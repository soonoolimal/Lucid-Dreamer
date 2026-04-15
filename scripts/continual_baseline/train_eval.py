import importlib
import inspect
import os
import pathlib
import sys
from datetime import datetime
from functools import partial as bind

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import elements
import numpy as np
import portal
import ruamel.yaml as yaml

import embodied
from dreamerv3.agent import Agent

DREAMER_CONFIGS = ROOT / 'dreamerv3' / 'configs.yaml'
CONFIGS_DIR = ROOT / 'configs'
LOGDIR_TPL = 'logs/continual_baseline/{scn}/{timestamp}'


def main(argv=None):
    # [elements.print(line) for line in Agent.banner]

    configs = load_configs('vizdoom.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_continual']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    scn_name = config.task.split('_', 1)[1]
    config = config.update(logdir=LOGDIR_TPL.format(scn=scn_name, timestamp=datetime.now().strftime('%y%m%d_%H%M%S')))
    logdir = elements.Path(config.logdir)
    print('Logdir:', logdir)
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
    print('Replica:', config.replica, '/', config.replicas)

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=config.errfile and logdir / 'error',
        clientkw=dict(logging_color='cyan'),
        serverkw=dict(logging_color='cyan'),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args,
    )


def load_configs(*extra_files):
    yml = yaml.YAML(typ='safe')
    configs = yml.load(elements.Path(DREAMER_CONFIGS).read())
    our_defaults = yml.load(elements.Path(CONFIGS_DIR / 'defaults.yaml').read())
    for key, val in our_defaults.items():
        if isinstance(val, dict) and key in configs['defaults']:
            configs['defaults'][key].update(val)
        else:
            configs['defaults'][key] = val
    for fname in extra_files:
        configs.update(yml.load(elements.Path(CONFIGS_DIR / fname).read()))
    return configs


def make_logger(config):
    step = elements.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
    for output in config.logger.outputs:
        if output == 'jsonl':
            outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
            outputs.append(elements.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'))
        elif output == 'tensorboard':
            outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
        elif output == 'scope':
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
        elif output == 'wandb':
            run_name = '/'.join(logdir.split('/')[-3:])
            outputs.append(elements.logger.WandBOutput(
                name=run_name, pattern=config.logger.wandb_filter,
                project='Lucid-Dreamer', group='continual_baseline', job_type='train',
            ))
        else:
            raise NotImplementedError(output)
    return elements.Logger(step, outputs, multiplier)


def make_env(config, _index, **overrides):
    suite, task = config.task.split('_', 1)
    ctor = {
        'vizdoom': 'embodied.envs.vizdoom:ContinualVizDoom',
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'atari100k': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'procgen': 'embodied.envs.procgen:ProcGen',
        'bsuite': 'embodied.envs.bsuite:BSuite',
    }[suite]
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
    kwargs = dict(config.env.get(suite, {}))
    kwargs.update(overrides)
    valid = inspect.signature(ctor).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, _config):
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_replay(config, folder, mode='train'):
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity

    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f'{config.replica:05}'
    kwargs = dict(
        length=length, capacity=int(capacity), online=config.replay.online,
        chunksize=config.replay.chunksize, directory=directory,
    )

    if config.replay.fracs.uniform < 1 and mode == 'train':
        assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
            'Gradient scaling for low-precision training can produce invalid loss outputs '
            'that are incompatible with prioritized replay.'
        )
        recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
        selectors = embodied.replay.selectors
        kwargs['selector'] = selectors.Mixture(
            dict(
                uniform=selectors.Uniform(),
                priority=selectors.Prioritized(**config.replay.prio),
                recency=selectors.Recency(recency),
            ), config.replay.fracs
        )

    return embodied.replay.Replay(**kwargs)


def make_stream(config, replay, mode):
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True,
    )
    return stream


def make_agent(config):
    env = make_env(config, 0)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


if __name__ == '__main__':
    main()
