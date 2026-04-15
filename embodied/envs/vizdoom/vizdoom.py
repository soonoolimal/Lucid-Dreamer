import functools

import elements
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from vizdoom import gymnasium_wrapper  # noqa: F401 registers VizDoom gym envs

import embodied

from . import vizdoom_wrappers as vdw
from .parse_dyn_info import DYNAMICS, REW_SHIFTS, SCN_DIR


class VizDoom(embodied.Env):
    def __init__(self, scn_name, dy_type, cheat, skip, level, timeout, size):
        if scn_name not in DYNAMICS:
            raise ValueError(f'Unknown scenario: {scn_name}')
        meta = DYNAMICS[scn_name]
        dynamics = meta[dy_type]

        self._env = self._make_env(scn_name, meta, dynamics, cheat, skip, level, timeout, size)
        self._done = True

        self._size = size

    def _make_env(self, scn_name, meta, dynamics, cheat, skip, level, timeout, size):
        scn_dir = SCN_DIR / ('cheat' if cheat else 'default')
        scn_id = meta['id']
        obs_shift = dynamics['obs_shift']
        rew_shift = dynamics['rew_shift']

        if rew_shift is not None and rew_shift not in REW_SHIFTS:
            raise ValueError(f'Unknown rew_shift: {rew_shift}')

        if obs_shift is not None:
            cfg_path = scn_dir / obs_shift / f'{scn_id}.cfg'
        else:
            cfg_path = scn_dir / f'{scn_id}.cfg'

        env = gym.make(
            f'Vizdoom{scn_name}-v1',
            scenario_config_file=str(cfg_path),
            render_mode='rgb_array',
        )

        if skip > 1:
            env = vdw.SkipFrame(env, skip)

        if rew_shift is not None:
            env = vdw.ShiftReward(env, rew_shift)
        elif cheat and meta.get('natural_hp') is not None:
            env = vdw.VirtualDeathReward(env, meta['natural_hp'], meta['death_penalty'])

        game: vzd.DoomGame = env.unwrapped.game

        if level is not None:
            game.set_doom_skill(level)

        if timeout is not None:
            game.set_episode_timeout(timeout)

        env = vdw.ResizeObservation(env, size)

        return env

    @functools.cached_property
    def obs_space(self):
        n_vars = self._env.observation_space.spaces['gamevariables'].shape[0]
        space = {
            'image': elements.Space(np.uint8, (self._size, self._size, 3)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        for i in range(n_vars):
            space[f'log/gamevar_{i}'] = elements.Space(np.float32)
        return space

    @functools.cached_property
    def act_space(self):
        n = self._env.action_space.n
        return {
            'action': elements.Space(np.int32, (), 0, n),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs, _ = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)

        obs, rew, term, trunc, _ = self._env.step(action['action'])
        self._done = term or trunc
        return self._obs(
            obs,
            float(rew),
            is_last=self._done,
            is_terminal=bool(term),
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        gamevars = np.asarray(obs['gamevariables'], dtype=np.float32)
        result = {
            'image': np.asarray(obs['screen'], dtype=np.uint8),
            'reward': np.float32(reward),
            'is_first': bool(is_first),
            'is_last': bool(is_last),
            'is_terminal': bool(is_terminal),
        }
        for i, v in enumerate(gamevars):
            result[f'log/gamevar_{i}'] = v
        return result

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
