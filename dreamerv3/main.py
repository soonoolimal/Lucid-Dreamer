import importlib
import inspect
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import numpy as np
import portal
import ruamel.yaml as yaml

import embodied

from .agent import Agent


def main(argv=None):
    [elements.print(line) for line in Agent.banner]

    # Set up
    # load config: defaults -> named config overrides -> CLI flags -> timestamp
    configs = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])                                          # defaults
    for name in parsed.configs:                                                            # named config overrides
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)                                           # CLI flags
    config = config.update(logdir=(config.logdir.format(timestamp=elements.timestamp())))  # timestamp

    # set replica index for parallel processing
    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
    print('Replica:', config.replica, '/', config.replicas)

    logdir = elements.Path(config.logdir)
    print('Logdir:', logdir)
    print('Run script:', config.script)
    if not config.script.endswith(('_env', '_replay')):  # worker scripts skip logdir creation
        logdir.mkdir()
        config.save(logdir / 'config.yaml')

    # worker process initializer: enable/disable global timer
    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    # inter-process communication setup for training and env workers
    portal.setup(
            errfile=config.errfile and logdir / 'error',
            clientkw=dict(logging_color='cyan'),
            serverkw=dict(logging_color='cyan'),
            initfns=[init],
            ipv6=config.ipv6,
    )

    # flat configs passed to embodied.run.* scripts
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

    # Running
    if config.script == 'train':
        embodied.run.train(
                bind(make_agent, config),
                bind(make_replay, config, 'replay'),
                bind(make_env, config),
                bind(make_stream, config),
                bind(make_logger, config),
                args,
        )
    elif config.script == 'train_eval':
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
    elif config.script == 'eval_only':
        embodied.run.eval_only(
                bind(make_agent, config),
                bind(make_env, config),
                bind(make_logger, config),
                args,
        )
    elif config.script == 'parallel':
        embodied.run.parallel.combined(
                bind(make_agent, config),
                bind(make_replay, config, 'replay'),
                bind(make_replay, config, 'replay_eval', 'eval'),
                bind(make_env, config),
                bind(make_env, config),
                bind(make_stream, config),
                bind(make_logger, config),
                args,
        )
    elif config.script == 'parallel_env':
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_env(
                bind(make_env, config),
                config.replica,
                args,
                is_eval,
        )
    elif config.script == 'parallel_envs':
        embodied.run.parallel.parallel_envs(
                bind(make_env, config),
                bind(make_env, config),
                args,
        )
    elif config.script == 'parallel_replay':
        embodied.run.parallel.parallel_replay(
                bind(make_replay, config, 'replay'),
                bind(make_replay, config, 'replay_eval', 'eval'),
                bind(make_stream, config),
                args,
        )
    else:
        raise NotImplementedError(config.script)


def make_agent(config):
    env = make_env(config, 0)                    # dummy env to get info
    notlog = lambda k: not k.startswith('log/')  # log/ keys are auxiliary, not model inputs
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
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


def make_logger(config):
    step = elements.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)  # action repeat scaling
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
    for output in config.logger.outputs:
        if output == 'jsonl':
            outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
            outputs.append(elements.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'))
        elif output == 'tensorboard':
            outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
        elif output == 'wandb':
            name = '/'.join(logdir.split('/')[-4:])
            outputs.append(elements.logger.WandBOutput(name))
        elif output == 'scope':
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
        else:
            raise NotImplementedError(output)
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, folder, mode='train'):
    """Create replay buffer for train or eval mode."""
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context  # consec chunks × batlen steps + context prefix
    assert config.batch_size * length <= capacity

    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f'{config.replica:05}'
    kwargs = dict(
            length=length, capacity=int(capacity), online=config.replay.online,
            chunksize=config.replay.chunksize, directory=directory
    )

    if config.replay.fracs.uniform < 1 and mode == 'train':
        assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
                'Gradient scaling for low-precision training can produce invalid loss outputs '
                'that are incompatible with prioritized replay.')
        recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp  # power-law recency weights
        selectors = embodied.replay.selectors
        kwargs['selector'] = selectors.Mixture(
            dict(
                uniform=selectors.Uniform(),
                priority=selectors.Prioritized(**config.replay.prio),
                recency=selectors.Recency(recency),
            ), config.replay.fracs
        )

    return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
    suite, task = config.task.split('_', 1)  # e.g. 'atari_pong' → ('atari', 'pong')
    # if suite == 'memmaze':
        # import memory_maze  # noqa
        # from embodied.envs import from_gym
    ctor = {
        'vizdoom': 'embodied.envs.vizdoom:VizDoom',
        'vizdoom_continual': 'embodied.envs.vizdoom:ContinualVizDoom',
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'atari100k': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'langroom': 'embodied.envs.langroom:LangRoom',
        'procgen': 'embodied.envs.procgen:ProcGen',
        'bsuite': 'embodied.envs.bsuite:BSuite',
        # 'memmaze': lambda task, **kw: from_gym.FromGym(f'MemoryMaze-{task}-v0', **kw),
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = dict(config.env.get(suite, {}))
    kwargs.update(overrides)
    if kwargs.pop('use_seed', False):    # consumed here, not passed to ctor
        kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
    if kwargs.pop('use_logdir', False):  # consumed here, not passed to ctor
        kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
    valid = inspect.signature(ctor).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, _config):
    """Apply standard action normalization and safety wrappers."""
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_stream(config, replay, mode):
    """Create a consecutive-chunk data stream from the replay buffer."""
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
            # sample consecutive sequence chunks for sequence training
            stream,
            length=config.batch_length if mode == 'train' else config.report_length,
            consec=config.consec_train if mode == 'train' else config.consec_report,
            prefix=config.replay_context,
            strict=(mode == 'train'),
            contiguous=True,
    )

    return stream


if __name__ == '__main__':
    main()
