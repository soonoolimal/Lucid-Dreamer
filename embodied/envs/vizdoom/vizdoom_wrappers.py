import cv2
import gymnasium as gym
import vizdoom as vzd


class ResizeObservation(gym.Wrapper):
    def __init__(self, env, size):
        super().__init__(env)
        self._size = size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._resize(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._resize(obs), rew, term, trunc, info

    def _resize(self, obs):
        screen = obs['screen']
        screen = cv2.resize(screen, (self._size, self._size), interpolation=cv2.INTER_AREA)
        return {**obs, 'screen': screen}


class VirtualDeathReward(gym.Wrapper):
    """Cheat mode only: tracks a virtual HP to issue death penalties as if the agent had natural HP."""
    def __init__(self, env, natural_hp, death_penalty):
        super().__init__(env)
        self._game: vzd.DoomGame = env.unwrapped.game

        self._natural_hp = natural_hp
        self._death_penalty = death_penalty

        self._virtual_hp = natural_hp
        self._prev_hp = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._virtual_hp = self._natural_hp
        self._prev_hp = self._game.get_game_variable(vzd.GameVariable.HEALTH)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        now_hp = self._game.get_game_variable(vzd.GameVariable.HEALTH)

        self._virtual_hp += now_hp - self._prev_hp
        self._prev_hp = now_hp

        while self._virtual_hp <= 0:
            rew -= self._death_penalty
            self._virtual_hp += self._natural_hp

        return obs, rew, term, trunc, info


class ShiftReward(gym.Wrapper):
    def __init__(self, env, rew_shift):
        super().__init__(env)
        self._game: vzd.DoomGame = env.unwrapped.game
        self._rew_shift = rew_shift
        self._prev_hp = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_hp = self._game.get_game_variable(vzd.GameVariable.HEALTH)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        now_hp = self._game.get_game_variable(vzd.GameVariable.HEALTH)

        if self._rew_shift == 'survive':
            rew = 1.0 if now_hp == self._prev_hp else 0.0
        elif self._rew_shift == 'recover':
            rew = 1.0 if now_hp > self._prev_hp else 0.0

        self._prev_hp = now_hp

        return obs, rew, term, trunc, info
