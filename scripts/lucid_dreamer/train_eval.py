import collections
import os
import pathlib
import sys
from datetime import datetime
from functools import partial as bind

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import elements
import numpy as np
import portal
import torch
import yaml
from common import (KST, load_configs, make_env, make_logger, make_replay,
                    make_stream)

import embodied
import inception.models  # register all subclasses
from inception.models.base import DynEncModel, DynPredModel, ObsEncModel
from lucid_dreamer.agent import LucidDreamerAgent
from lucid_dreamer.kick import Kick

OBS_ENC_REGISTRY = {cls.__name__: cls for cls in ObsEncModel.__subclasses__()}
DYN_ENC_REGISTRY = {cls.__name__: cls for cls in DynEncModel.__subclasses__()}
DYN_PRED_REGISTRY = {cls.__name__: cls for cls in DynPredModel.__subclasses__()}

LOGDIR_TPL = 'logs/lucid_dreamer/{scn}/{timestamp}'


def _pop_arg(argv, flag):
    """Pop --flag value from argv list. Raises ValueError if not found."""
    if flag not in argv:
        raise ValueError(f'{flag} is required')
    idx = argv.index(flag)
    value = argv.pop(idx + 1)
    argv.pop(idx)
    return value


def _load_kick(config, is_discrete, device, ckpt_dir):
    """Build Inception moddel, then wrap in Kick."""
    with open(ROOT / 'configs' / 'inception.yaml') as f:
        inc_cfg = yaml.safe_load(f)['inception']

    ckpt_dir = pathlib.Path(ckpt_dir)
    seed = config.inception.seed

    obs_enc_cls_name = inc_cfg['obs_enc']['select']
    obs_enc = OBS_ENC_REGISTRY[obs_enc_cls_name](**inc_cfg['obs_enc'][obs_enc_cls_name])

    dyn_enc_cls_name = inc_cfg['dyn_enc']['select']
    dyn_enc_kw = inc_cfg['dyn_enc'][dyn_enc_cls_name]

    # infer max_ep_len and act_dim from saved weights to match training exactly
    dye_path = ckpt_dir / f'{dyn_enc_cls_name}_s{seed}_best.pt'
    dye_state = torch.load(dye_path, map_location='cpu')
    max_ep_len = dye_state['embed_tstep.weight'].shape[0]
    if is_discrete:
        act_dim = dye_state['embed_act.weight'].shape[0]  # Embedding: (n_act, H)
    else:
        act_dim = dye_state['embed_act.weight'].shape[1]  # Linear: (H, act_dim)
    dye = DYN_ENC_REGISTRY[dyn_enc_cls_name](
        obs_encoder=obs_enc,
        act_dim=act_dim,
        is_discrete=is_discrete,
        hidden_size=obs_enc.enc_dim,
        seq_len=inc_cfg['data']['seq_len'],
        max_ep_len=max_ep_len,
        **dyn_enc_kw,
    )
    dye.load_state_dict(dye_state)

    dyn_pred_cls_name = inc_cfg['dyn_pred']['select']
    dyn_pred_kw = inc_cfg['dyn_pred'][dyn_pred_cls_name]
    dyp = DYN_PRED_REGISTRY[dyn_pred_cls_name](
        hidden_size=dye.gpt2.config.n_embd,
        last_k=dyn_enc_kw['last_k'],
        pass_h=dyn_enc_kw['pass_h'],
        n_dynamics=inc_cfg['n_dynamics'],
        **dyn_pred_kw,
    )
    dyp_path = ckpt_dir / f'{dyn_pred_cls_name}_s{seed}_best.pt'
    dyp.load_state_dict(torch.load(dyp_path, map_location='cpu'))

    dye.eval().to(device)
    dyp.eval().to(device)

    kick_cfg = config.kick
    return Kick(
        dye, dyp, device,
        seq_len=kick_cfg.seq_len,
        n_confirm=kick_cfg.n_confirm,
        target_rtg=kick_cfg.target_rtg,
        gamma=kick_cfg.gamma,
    )


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    ds_type = _pop_arg(argv, '--ds_type')
    inc_timestamp = _pop_arg(argv, '--inc_timestamp')

    configs = load_configs('vizdoom.yaml', 'lucid_dreamer.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_lucid', 'lucid_dreamer']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    scn_name = config.task.split('_', 1)[1]
    ckpt_dir = ROOT / f'logs/inception/{scn_name}/{ds_type}/{inc_timestamp}'
    ts = config.run.resume_timestamp or datetime.now(tz=KST).strftime('%y%m%d_%H%M%S')
    logdir_str = LOGDIR_TPL.format(scn=scn_name, timestamp=ts)
    if config.run.debug:
        logdir_str = logdir_str.replace('logs/', 'logs/debug/', 1)
    config = config.update(logdir=logdir_str)
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

    _env = make_env(config, 0, vizdoom_cls='ContinualVizDoom')
    is_discrete = 'action_cont' not in _env.act_space
    _env.close()

    device = torch.device(config.inception.device)
    kick = _load_kick(config, is_discrete, device, ckpt_dir)

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

    make_env_fn = bind(make_env, config, vizdoom_cls='ContinualVizDoom')

    def make_agent():
        env = make_env_fn(0)
        obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
        act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
        env.close()
        return LucidDreamerAgent(obs_space, act_space, elements.Config(
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

    train_eval(
        make_agent,
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        make_env_fn,
        make_env_fn,
        bind(make_stream, config),
        bind(make_logger, config, 'train'),
        args,
        kick,
        is_discrete,
        config.kick.target_rtg,
        config.kick.ema_alpha,
    )


def train_eval(
    make_agent, make_replay_train, make_replay_eval,
    make_env_train, make_env_eval,
    make_stream, make_logger,
    args,
    kick, is_discrete, init_target_rtg, ema_alpha,
):
    agent = make_agent()
    replay_train = make_replay_train()
    replay_eval = make_replay_eval()
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    train_episodes = collections.defaultdict(elements.Agg)
    train_epstats = elements.Agg()
    eval_episodes = collections.defaultdict(elements.Agg)
    eval_epstats = elements.Agg()
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_video = elements.when.Every(args.report_every)
    should_save = embodied.LocalClock(args.save_every)

    @elements.timer.section('logfn')
    def logfn(tran, worker, mode):
        episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
        epstats = dict(train=train_epstats, eval=eval_epstats)[mode]
        episode = episodes[worker]
        if tran['is_first']:
            episode.reset()
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')
        for key, value in tran.items():
            if value.dtype == np.uint8 and value.ndim == 3:
                if worker == 0:
                    episode.add(f'policy_{key}', value, agg='stack')
            elif key.startswith('log/'):
                assert value.ndim == 0, (key, value.shape, value.dtype)
                episode.add(key + '/avg', value, agg='avg')
                episode.add(key + '/max', value, agg='max')
                episode.add(key + '/sum', value, agg='sum')
        if tran['is_last']:
            result = episode.result()
            logger.add({
                'score': result.pop('score'),
                'length': result.pop('length'),
            }, prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    # kick state
    # worker 0 only; dynamics shift is globally synchronized
    kick_flag = [None]                     # current shift flag; None during seq_len warmup
    ep_step = [0]                          # in-episode step counter
    ep_rew_sum = [0.0]                     # episode reward accumulator
    target_rtg = [float(init_target_rtg)]  # EMA of episode returns for kick.reset()

    @elements.timer.section('kickfn')
    def kickfn(tran, worker):
        if worker != 0:
            return
        if tran['is_first']:
            kick.reset(target_rtg=target_rtg[0])
            ep_step[0] = 0
            ep_rew_sum[0] = 0.0

        ep_rew_sum[0] += float(tran['reward'])

        obs = torch.from_numpy(np.asarray(tran['image'])).permute(2, 0, 1).float().div(255.0)
        if is_discrete:
            act = torch.tensor(int(tran['action']), dtype=torch.long)
        else:
            act = torch.from_numpy(np.concatenate([
                np.array([tran['action']], dtype=np.float32),
                np.asarray(tran['action_cont'], dtype=np.float32),
            ]))
        rew = torch.tensor(float(tran['reward']), dtype=torch.float32)
        ts = torch.tensor(ep_step[0], dtype=torch.long)

        flag = kick.step(obs, act, rew, ts)
        if flag is not None:
            kick_flag[0] = flag
        ep_step[0] += 1

        if tran['is_last']:
            target_rtg[0] = (1 - ema_alpha) * target_rtg[0] + ema_alpha * ep_rew_sum[0]

    fns = [bind(make_env_train, i) for i in range(args.envs)]
    driver_train = embodied.Driver(fns, parallel=(not args.debug))
    driver_train.on_step(lambda tran, _: step.increment())
    driver_train.on_step(lambda tran, _: policy_fps.step())
    driver_train.on_step(replay_train.add)
    driver_train.on_step(kickfn)
    driver_train.on_step(bind(logfn, mode='train'))

    fns = [bind(make_env_eval, i) for i in range(args.eval_envs)]
    driver_eval = embodied.Driver(fns, parallel=(not args.debug))
    driver_eval.on_step(replay_eval.add)
    driver_eval.on_step(bind(logfn, mode='eval'))
    driver_eval.on_step(lambda tran, _: policy_fps.step())

    stream_train = iter(agent.stream(make_stream(replay_train, 'train')))
    stream_report = iter(agent.stream(make_stream(replay_train, 'report')))
    stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))

    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)
    carry_eval = agent.init_report(args.batch_size)

    def trainfn(tran, worker):
        if len(replay_train) < args.batch_size * args.batch_length:
            return
        for _ in range(should_train(step)):
            with elements.timer.section('stream_next'):
                batch = next(stream_train)
            real = bool(kick_flag[0]) if kick_flag[0] is not None else False
            carry_train[0], outs, mets = agent.train(carry_train[0], batch, real=real)
            train_fps.step(batch_steps)
            if 'replay' in outs:
                replay_train.update(outs['replay'])
            train_agg.add(mets, prefix='train')
    driver_train.on_step(trainfn)

    def reportfn(carry, stream):
        agg = elements.Agg()
        for _ in range(args.report_batches):
            carry, mets = agent.report(carry, next(stream))
            agg.add(mets)
        return carry, agg.result()

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay_train = replay_train
    cp.replay_eval = replay_eval
    if args.from_checkpoint:
        elements.checkpoint.load(args.from_checkpoint, dict(
            agent=bind(agent.load, regex=args.from_checkpoint_regex)
        ))
    cp.load_or_save()

    print('Start training loop')
    train_policy = lambda *args: agent.policy(*args, mode='train')
    eval_policy = lambda *args: agent.policy(*args, mode='eval')
    driver_train.reset(agent.init_policy)
    while step < args.steps:
        driver_train(train_policy, steps=10)

        if should_video(step) and len(replay_train):
            print('Video / Evaluation')
            driver_eval.reset(agent.init_policy)
            driver_eval(eval_policy, episodes=args.eval_eps)
            logger.add(eval_epstats.result(), prefix='epstats')
            carry_report, mets = reportfn(carry_report, stream_report)
            logger.add(mets, prefix='report')
            if len(replay_eval):
                carry_eval, mets = reportfn(carry_eval, stream_eval)
                logger.add(mets, prefix='eval')

        if should_log(step):
            logger.add(train_agg.result())
            logger.add(train_epstats.result(), prefix='epstats')
            logger.add(replay_train.stats(), prefix='replay')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.add({'timer': elements.timer.stats()['summary']})
            logger.write()

        if should_save(step):
            cp.save()

    logger.close()


if __name__ == '__main__':
    main()
