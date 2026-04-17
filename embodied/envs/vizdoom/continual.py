import functools

import embodied

from .parse_dyn_info import DYNAMICS
from .vizdoom import VizDoom


class ContinualVizDoom(embodied.Env):
    def __init__(self, scn_name, switch_every, cheat, skip, level, timeout, size):
        if scn_name not in DYNAMICS:
            raise ValueError(f'Unknown scenario: {scn_name}')
        self._scn_name = scn_name
        self._switch_every = switch_every
        self._vzd_kw = dict(cheat=cheat, skip=skip, level=level, timeout=timeout, size=size)
        self._n_dynamics = DYNAMICS[scn_name]['n_dynamics']

        self._dy_type = 0
        self._step = 0
        self._env = VizDoom(scn_name, dy_type=0, **self._vzd_kw)

    @functools.cached_property
    def obs_space(self):
        return self._env.obs_space

    @functools.cached_property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)

        if not obs['is_first']:
            self._step += 1

        if self._step > 0 and self._step % self._switch_every == 0:
            obs = {**obs, 'is_last': True, 'is_terminal': False}
            self._switch()

        return obs

    def _switch(self):
        self._env.close()
        self._dy_type = (self._dy_type + 1) % self._n_dynamics
        self._env = VizDoom(self._scn_name, dy_type=self._dy_type, **self._vzd_kw)

    def close(self):
        self._env.close()
