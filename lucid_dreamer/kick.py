from collections import deque

import torch
from torch import nn


class Kick:
    """Runs Inception to kick Dreamer out of imagination on dynamics shift."""
    def __init__(
        self,
        dye,
        dyp,
        device,
        seq_len,
        n_confirm,
        target_rtg,
        gamma,
    ):
        self.dye: nn.Module = dye
        self.dyp: nn.Module = dyp
        self.device = device
        self.seq_len = seq_len
        self.n_confirm = n_confirm
        self._default_target_rtg = target_rtg
        self._gamma = gamma

        self._obs = deque(maxlen=seq_len)
        self._act = deque(maxlen=seq_len)
        self._rew = deque(maxlen=seq_len)
        self._rtg = deque(maxlen=seq_len)
        self._ts = deque(maxlen=seq_len)

        self._curr_rtg = target_rtg
        self._stable_label = None
        self._cand_label = None
        self._cand_count = 0

    def reset(self, target_rtg=None):
        """Call at the start of each episode."""
        self._obs.clear()
        self._act.clear()
        self._rew.clear()
        self._rtg.clear()
        self._ts.clear()

        self._curr_rtg = target_rtg if target_rtg is not None else self._default_target_rtg
        self._stable_label = None
        self._cand_label = None
        self._cand_count = 0

    @torch.no_grad()
    def step(self, obs, act, rew, ts):
        """
        Called on every Dreamer env step:
            Feeds real interaction into the sliding buffer.
            Returns None until buffer reaches seq_len, then False/True each step.

        RTG is computed internally: rtg_{t+1} = (rtg_t - r_t) / gamma

        Returns:
            None: buffer warming up
            False: dynamics stable
            True: dynamics shifted
        """
        rtg = torch.tensor(self._curr_rtg, dtype=torch.float32)
        self._curr_rtg = (self._curr_rtg - float(rew)) / self._gamma

        self._obs.append(obs)
        self._act.append(act)
        self._rew.append(rew)
        self._rtg.append(rtg)
        self._ts.append(ts)

        if len(self._obs) < self.seq_len:
            return None

        obs_t = torch.stack(list(self._obs)).unsqueeze(0).to(self.device)                # (1, T, C, H, W)
        act_t = torch.stack(list(self._act)).unsqueeze(0).to(self.device)                # (1, T)/(1, T, act_dim)
        rew_t = torch.stack(list(self._rew)).unsqueeze(0).unsqueeze(-1).to(self.device)  # (1, T, 1)
        rtg_t = torch.stack(list(self._rtg)).unsqueeze(0).unsqueeze(-1).to(self.device)  # (1, T, 1)
        ts_t = torch.stack(list(self._ts)).unsqueeze(0).to(self.device)                  # (1, T)

        self.dye.eval()
        self.dyp.eval()

        enc_out = self.dye(obs_t, act_t, rew_t, rtg_t, ts_t)
        logits = self.dyp(enc_out)  # (1, n_dynamics)
        curr_label = int(logits.argmax(dim=-1).item())

        if self._stable_label is None:
            self._stable_label = curr_label
            self._cand_label = curr_label
            self._cand_count = 1
            return False

        if curr_label == self._cand_label:
            self._cand_count += 1
        else:
            self._cand_label = curr_label
            self._cand_count = 1

        if self._cand_count >= self.n_confirm:
            self._stable_label = self._cand_label

        return curr_label != self._stable_label
