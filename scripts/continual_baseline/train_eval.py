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
from common import (KST, load_configs, make_agent, make_env, make_logger,
                    make_replay, make_stream)

import embodied

LOGDIR_TPL = 'logs/continual_baseline/{scn}/{timestamp}'


def main(argv=None):
    configs = load_configs('vizdoom.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_continual']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    scn_name = config.task.split('_', 1)[1]
    config = config.update(logdir=LOGDIR_TPL.format(
        scn=scn_name,
        timestamp=datetime.now(tz=KST).strftime('%y%m%d_%H%M%S'),
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

    train_eval(
        bind(make_agent, config, bind(make_env, config, vizdoom_cls='ContinualVizDoom')),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config, vizdoom_cls='ContinualVizDoom'),
        bind(make_env, config, vizdoom_cls='ContinualVizDoom'),
        bind(make_stream, config),
        bind(make_logger, config, 'train'),
        args,
    )


def train_eval(make_agent, make_replay_train, make_replay_eval, make_env_train, make_env_eval, make_stream, make_logger, args):
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

    fns = [bind(make_env_train, i) for i in range(args.envs)]
    driver_train = embodied.Driver(fns, parallel=(not args.debug))
    driver_train.on_step(lambda tran, _: step.increment())
    driver_train.on_step(lambda tran, _: policy_fps.step())
    driver_train.on_step(replay_train.add)
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
            carry_train[0], outs, mets = agent.train(carry_train[0], batch)
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
