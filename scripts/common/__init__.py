import importlib
import inspect
import pathlib
import re
from datetime import timedelta, timezone
from functools import partial as bind

import elements
import numpy as np
import ruamel.yaml as yaml
import wandb

import embodied
from dreamerv3.agent import Agent

KST = timezone(timedelta(hours=9))

ROOT = pathlib.Path(__file__).parents[2]
DREAMER_CONFIGS = ROOT / 'configs' / 'dreamerv3.yaml'
CONFIGS_DIR = ROOT / 'configs'


def resolve_checkpoint(argv, path_fn):
    """Pop --timestamp from argv and inject --run.from_checkpoint via path_fn.

    Args:
        argv: argument list
        path_fn: callable(scn, ts, argv) -> str, returns checkpoint path
    """
    argv = list(argv or [])
    if '--timestamp' not in argv:
        return argv
    idx = argv.index('--timestamp')
    ts = argv.pop(idx + 1)
    argv.pop(idx)
    task_idx = argv.index('--task') if '--task' in argv else -1
    if task_idx < 0:
        raise ValueError('--task is required when using --timestamp')
    scn = argv[task_idx + 1].split('_', 1)[1]
    return argv + ['--run.from_checkpoint', path_fn(scn, ts, argv)]


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


def make_env(config, _index, vizdoom_cls='VizDoom', **overrides):
    suite, task = config.task.split('_', 1)
    ctor = {
        'vizdoom': f'embodied.envs.vizdoom:{vizdoom_cls}',
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


def make_agent(config, make_env_fn):
    env = make_env_fn(0)
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


def _patch_wandb_video():
    _orig = wandb.Video
    wandb.Video = lambda data, **kw: _orig(data, format=kw.pop('format', 'gif'), **kw)

def make_logger(config, job_type):
    _patch_wandb_video()
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
            run_name = '/'.join(logdir.split('/')[1:])
            wb_group = config.logger.wandb_group
            project = config.logger.wandb_project
            if config.run.get('debug', False):
                run_name = 'debug_' + run_name
                wb_group = 'debug_' + wb_group
                project = project + '-Debug'
            wandb_id = config.run.get('wandb_id', '') or re.sub(r'[^a-zA-Z0-9_-]', '-', run_name)[:64]
            outputs.append(elements.logger.WandBOutput(
                name=run_name, pattern=config.logger.wandb_filter,
                project=project, group=wb_group, job_type=job_type,
                id=wandb_id, resume='allow',
                settings=wandb.Settings(init_timeout=300),
            ))
        else:
            raise NotImplementedError(output)
    return elements.Logger(step, outputs, multiplier)


# copied from dreamerv3/main.py
def make_replay(config, folder, mode='train'):
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity

    # directory = elements.Path(config.logdir) / folder
    # if config.replicas > 1:
    #     directory /= f'{config.replica:05}'
    kwargs = dict(
        length=length, capacity=int(capacity), online=config.replay.online,
        chunksize=config.replay.chunksize,
        # turn off disk I/O of replay buffer
        # since the replay buffer starts empty upon resuming,
        # an unstable warming-up phase occurs until the buffer fills up again
        # ckpt/, metrics.jsonl, config.yaml are still stored
        # so that agent weights are restored upon resuming
        directory=None,
        # directory=directory,
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


# copied from dreamerv3/main.py
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


# copied from dreamerv3/main.py
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
