import os
import pathlib
import sys
from datetime import datetime
from functools import partial as bind

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import elements
import portal
from common import (KST, load_configs, make_agent, make_env, make_logger,
                    resolve_checkpoint)

import embodied

LOGDIR_TPL = 'logs/continual_baseline/{scn}/eval_{timestamp}'


def main(argv=None):
    argv = resolve_checkpoint(argv, lambda scn, ts, _: str(ROOT / f'logs/continual_baseline/{scn}/{ts}/ckpt'))
    configs = load_configs('vizdoom.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_continual']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    scn_name = config.task.split('_', 1)[1]
    config = config.update(logdir=LOGDIR_TPL.format(scn=scn_name, timestamp=datetime.now(tz=KST).strftime('%y%m%d_%H%M%S')))
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

    embodied.run.eval_only(
        bind(make_agent, config, bind(make_env, config, vizdoom_cls='ContinualVizDoom')),
        bind(make_env, config, vizdoom_cls='ContinualVizDoom'),
        bind(make_logger, config, 'continual_baseline', 'eval'),
        args,
    )



if __name__ == '__main__':
    main()
