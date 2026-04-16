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
from common import load_configs, make_agent, make_env
from per_dy_dreamer.train import _collect_samples


def main(argv=None):
    configs = load_configs('vizdoom.yaml')
    parsed, other = elements.Flags(configs=['defaults', 'vizdoom_fixed']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    scn_name = config.task.split('_', 1)[1]
    dy_type = int(config.env.vizdoom.dy_type)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

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
    elements.checkpoint.load(config.run.from_checkpoint, dict(
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
