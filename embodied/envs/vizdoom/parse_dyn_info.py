from pathlib import Path

import ruamel.yaml as yaml

DYNAMICS = yaml.YAML(typ='safe').load((Path(__file__).parent / 'dynamics.yaml').read_text())
SCN_DIR = Path(__file__).parent / DYNAMICS.pop('scn_dir')
REW_SHIFTS = set(DYNAMICS.pop('rew_shifts'))
