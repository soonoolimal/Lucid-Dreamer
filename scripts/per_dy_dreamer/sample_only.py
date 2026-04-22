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
from common import KST, load_configs, make_agent, make_env, resolve_checkpoint
from per_dy_dreamer.train import _collect_samples


def _ckpt_path(scn, ts, argv):
    dy_type_idx = argv.index('--env.vizdoom.dy_type') if '--env.vizdoom.dy_type' in argv else -1
    dy_type = argv[dy_type_idx + 1] if dy_type_idx >= 0 else '0'
    is_random = (
        '--random_agent' in argv
        and argv[argv.index('--random_agent') + 1].lower() == 'true'
    )
    subdir = f'dy{dy_type}_random' if is_random else f'dy{dy_type}'
    return str(ROOT / f'logs/per_dy_dreamer/{scn}/{subdir}/{ts}/ckpt')


def main(argv=None):
    argv = resolve_checkpoint(argv, _ckpt_path)
    configs = load_configs('vizdoom.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_fixed']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    scn_name = config.task.split('_', 1)[1]
    dy_type = int(config.env.vizdoom.dy_type)
    timestamp = datetime.now(tz=KST).strftime('%y%m%d_%H%M%S')

    ds_type = 'random' if config.random_agent else 'dreamer'
    data_dir = pathlib.Path(ROOT / 'data' / scn_name / ds_type)
    data_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(data_dir / f'{timestamp}_dy{dy_type}_s{config.seed}_config.yaml'))

    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
    print('Replica:', config.replica, '/', config.replicas)

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        clientkw=dict(logging_color='cyan'),
        serverkw=dict(logging_color='cyan'),
        initfns=[init],
        ipv6=config.ipv6,
    )

    assert config.run.from_checkpoint, '--run.from_checkpoint is required'

    agent = make_agent(config, bind(make_env, config))
    cp = elements.Checkpoint(config.run.from_checkpoint)
    elements.checkpoint.load(cp.latest(), dict(
        agent=bind(agent.load, regex=config.run.from_checkpoint_regex)
    ))

    _collect_samples(
        agent,
        bind(make_env, config),
        config,
        dy_type,
        step=0,  # dummy; only inference
        n_eps=int(config.run.n_sample_eps),
        timestamp=timestamp,
    )


if __name__ == '__main__':
    main()
