import collections
import os
import pathlib
import sys
from datetime import datetime
from functools import partial as bind

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import elements
import portal
from common import (load_configs, make_agent, make_env, make_logger,
                    make_replay, make_stream)

import embodied

LOGDIR_TPL = 'logs/per_dy_dreamer/{scn}/dy{dy_type}{suffix}/{timestamp}'


def main(argv=None):
    configs = load_configs('vizdoom.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_fixed']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    # activate cheat mode for random agent
    if config.random_agent:
        config = config.update({'env.vizdoom.cheat': True})

    scn_name = config.task.split('_', 1)[1]
    dy_type = int(config.env.vizdoom.dy_type)
    config = config.update(logdir=LOGDIR_TPL.format(
        scn=scn_name, dy_type=dy_type,
        suffix='_random' if config.random_agent else '',
        timestamp=datetime.now().strftime('%y%m%d_%H%M%S'),
    ))
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

    train(
        bind(make_agent, config, bind(make_env, config)),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config, 'per_dy_dreamer', 'train'),
        args,
        config,
        dy_type,
    )


def train(make_agent, make_replay, make_env, make_stream, make_logger, args, config, dy_type):
    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    sample_timestamp = pathlib.Path(str(logdir)).name
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_report = elements.when.Every(args.report_every)
    should_save = embodied.LocalClock(args.save_every)

    # step-based trigger: collect HDF5 samples every sample_every env steps
    next_sample_step = [int(args.sample_every)]
    last_video = [None]  # most recent worker-0 episode frames for random agent video logging

    @elements.timer.section('logfn')
    def logfn(tran, worker):
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
            if config.random_agent and worker == 0 and 'policy_image' in result:
                last_video[0] = result.pop('policy_image')
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.envs)]
    driver = embodied.Driver(fns, parallel=not args.debug)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    stream_train = iter(agent.stream(make_stream(replay, 'train')))
    stream_report = iter(agent.stream(make_stream(replay, 'report')))

    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def trainfn(tran, worker):
        if len(replay) < args.batch_size * args.batch_length:
            return
        for _ in range(should_train(step)):
            with elements.timer.section('stream_next'):
                batch = next(stream_train)
            carry_train[0], outs, mets = agent.train(carry_train[0], batch)
            train_fps.step(batch_steps)
            if 'replay' in outs:
                replay.update(outs['replay'])
            train_agg.add(mets, prefix='train')
    driver.on_step(trainfn)

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    if args.from_checkpoint:
        elements.checkpoint.load(args.from_checkpoint, dict(
            agent=bind(agent.load, regex=args.from_checkpoint_regex)
        ))
    cp.load_or_save()

    print('Start training loop')
    policy = lambda *args: agent.policy(*args, mode='train')
    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)

        if should_report(step):
            if config.random_agent:
                if last_video[0] is not None:
                    logger.add({'policy_image': last_video[0]}, prefix='report')
                    logger.write()
            elif len(replay):
                agg = elements.Agg()
                for _ in range(args.consec_report * args.report_batches):
                    carry_report, mets = agent.report(carry_report, next(stream_report))
                    agg.add(mets)
                logger.add(agg.result(), prefix='report')

        if should_log(step):
            if not config.random_agent:
                logger.add(train_agg.result())
                logger.add(replay.stats(), prefix='replay')
                logger.add({'fps/train': train_fps.result()})
                logger.add({'timer': elements.timer.stats()['summary']})
            logger.add(epstats.result(), prefix='epstats')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.write()

        if should_save(step):
            cp.save()

        if args.sample_every > 0 and int(step) >= next_sample_step[0]:
            _collect_samples(agent, make_env, config, dy_type, int(step), int(args.n_sample_eps), timestamp=sample_timestamp)
            next_sample_step[0] = int(step) + int(args.sample_every)

    logger.close()


def _collect_samples(agent, make_env, config, dy_type, step, n_eps, timestamp=None):
    """Runs n_eps eval episodes to HDF5 for Alarm training.

    Output: data/{scn}/{ds_type}/{timestamp}_dy{dy_type}_s{seed}.hdf5
        observations  (N, H, W, 3)     uint8
        actions
            discrete: (N, 1)           int64
            mixed:    (N, 1+cont_dim)  float32
        rewards       (N,)             float32
        timeouts      (N,)             float32  1.0 at episode end
        attrs: timeout, num_episodes, act_dim, is_discrete, num_transitions
    """
    env = make_env(0)
    act_space = env.act_space['action']
    is_discrete = act_space.discrete
    if is_discrete:
        act_dim = 1
        ac_dtype = np.int64
    else:
        # mixed: action (discrete button) + action_cont (continuous deltas)
        cont_dim = env.act_space['action_cont'].shape[0]
        act_dim = 1 + cont_dim
        ac_dtype = np.float32

    all_obs, all_acs, all_rews, all_timeouts = [], [], [], []
    carry = agent.init_policy(1)

    for _ in range(n_eps):
        tran = env.step({'action': np.zeros((), np.int32), 'reset': np.array(True)})
        while True:
            obs_in = {k: tran[k][None] for k in tran if not k.startswith('log/')}
            carry, action_out, _ = agent.policy(carry, obs_in, mode='eval')

            if is_discrete:
                ac_vec = np.array([int(action_out['action'][0])], dtype=np.int64)
                gym_action = {'action': np.int32(ac_vec[0]), 'reset': np.array(False)}
            else:
                binary = int(action_out['action'][0])
                cont = np.asarray(action_out['action_cont'][0], dtype=np.float32)
                ac_vec = np.concatenate([[float(binary)], cont]).astype(np.float32)
                gym_action = {'action': np.int32(binary), 'action_cont': cont, 'reset': np.array(False)}

            next_tran = env.step(gym_action)

            all_obs.append(tran['image'])
            all_acs.append(ac_vec)
            all_rews.append(float(next_tran['reward']))
            all_timeouts.append(1.0 if next_tran['is_last'] else 0.0)

            if next_tran['is_last']:
                break

            tran = next_tran

    env.close()

    obs_arr = np.stack(all_obs).astype(np.uint8)
    acs_arr = np.stack(all_acs).astype(ac_dtype)
    rews_arr = np.array(all_rews, dtype=np.float32)
    tout_arr = np.array(all_timeouts, dtype=np.float32)

    max_ep_len = config.env.vizdoom.timeout // config.env.vizdoom.skip

    scn_name = config.task.split('_', 1)[1]
    ds_type = 'random' if config.random_agent else 'dreamer'
    data_dir = ROOT / 'data' / scn_name / ds_type
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    hdf5_path = str(data_dir / f'{timestamp}_dy{dy_type}_s{config.seed}.hdf5')

    datasets = [
        ('observations', obs_arr),
        ('actions', acs_arr),
        ('rewards', rews_arr),
        ('timeouts', tout_arr),
    ]
    chunk = min(256, obs_arr.shape[0])

    def create_ds(hf, name, arr):
        tail = arr.shape[1:]
        hf.create_dataset(
            name, data=arr, maxshape=(None, *tail), chunks=(chunk, *tail),
            compression='gzip', shuffle=True,
        )

    if not os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'w') as hf:
            for name, arr in datasets:
                create_ds(hf, name, arr)
            hf.attrs['timeout'] = max_ep_len
            hf.attrs['num_episodes'] = n_eps
            hf.attrs['act_dim'] = act_dim
            hf.attrs['is_discrete'] = is_discrete
            hf.attrs['num_transitions'] = len(all_obs)
    else:
        with h5py.File(hdf5_path, 'a') as hf:
            n = hf['observations'].shape[0]
            m = len(all_obs)
            for name, arr in datasets:
                hf[name].resize(n + m, axis=0)
                hf[name][n:] = arr
            hf.attrs['num_episodes'] = int(hf.attrs['num_episodes']) + n_eps
            hf.attrs['num_transitions'] = n + m

    print(f'[HDF5] step={step} n_eps={n_eps} n_trans={len(all_obs)} → {hdf5_path}')


if __name__ == '__main__':
    main()
