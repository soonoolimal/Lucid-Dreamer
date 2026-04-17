from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import transformers
from einops import rearrange, repeat

from .backbone.gpt2 import GPT2Model
from .base import DynEncModel, ObsEncModel


@dataclass
class DTOutput:
    rtg_preds: torch.Tensor          # (B, T, 1)
    obs_preds: torch.Tensor          # (B, T, H)
    ac_logits: torch.Tensor          # (B, T, act_dim)
    obs_enc: torch.Tensor            # (B, T, H)
    rtg_h_k: Optional[torch.Tensor]  # (B, last_k, H) or None
    obs_h_k: Optional[torch.Tensor]  # (B, last_k, H) or None


class DecisionTransformer(DynEncModel):
    """Dynamics encoder based on Decision Transformer.

    Encodes (rtg, obs, act) trajectories via causal transformer.
    Produces self-supervised predictions for loss and hidden states for the paired DynPredModel.
    """
    def __init__(
        self,
        obs_encoder: ObsEncModel,
        act_dim,
        is_discrete,
        hidden_size,
        seq_len,
        last_k,
        pass_h,    # 'all' | 'rtg' | 'obs'
        max_ep_len,
        **kwargs,  # passed to GPT2Config (n_layer, n_head, etc.)
    ):
        super().__init__()

        if obs_encoder.enc_dim != hidden_size:
            raise ValueError(
                f'obs_encoder.enc_dim ({obs_encoder.enc_dim}) must match hidden_size ({hidden_size})'
            )
        if pass_h not in ('all', 'rtg', 'obs'):
            raise ValueError(f"pass_h must be 'all', 'rtg', or 'obs', got {pass_h!r}")

        self.obs_encoder = obs_encoder
        self.is_discrete = is_discrete
        self.last_k = last_k
        self.pass_h = pass_h

        config = transformers.GPT2Config(
            vocab_size=1,  # unused
            n_ctx=3 * seq_len,
            n_positions=3 * seq_len,
            n_embd=hidden_size,
            **kwargs,
        )
        self.gpt2 = GPT2Model(config)

        self.embed_tstep = nn.Embedding(max_ep_len, hidden_size)

        if is_discrete:
            self.embed_act = nn.Embedding(act_dim, hidden_size)
        # mixed: actions are (B, T, 1+cont_dim) float — discrete button + continuous deltas
        else:
            self.embed_act = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_obs = nn.Linear(hidden_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.n_ctx, hidden_size))

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.pred_act = nn.Linear(hidden_size, act_dim)      # h(obs_t) -> act_t
        self.pred_rtg = nn.Linear(hidden_size, 1)            # h(act_t) -> R_{t+1}
        self.pred_obs = nn.Linear(hidden_size, hidden_size)  # h(act_t) -> enc(obs_{t+1})

    def forward(self, observations, actions, rewards, returns_to_go, timesteps, mask=None):
        B, T, C, H, W = observations.size()

        obs_enc = self.obs_encoder(observations.reshape(B*T, C, H, W)).reshape(B, T, -1)

        if mask is None:
            mask = torch.ones((B, T), dtype=torch.long, device=obs_enc.device)

        # raws -> embeddings (B, T, H)
        rtg_emb = self.embed_rtg(returns_to_go)
        obs_emb = self.embed_obs(obs_enc)
        act_emb = self.embed_act(actions)
        time_emb = self.embed_tstep(timesteps)

        rtg_emb = rtg_emb + time_emb
        obs_emb = obs_emb + time_emb
        act_emb = act_emb + time_emb

        # interleave as [R_0, obs_0, act_0, R_1, obs_1, act_1, ...]
        stacked = rearrange(
            torch.stack((rtg_emb, obs_emb, act_emb), dim=2),
            'B T M H -> B (T M) H',
        )  # (B, 3*T, H)

        stacked_len = stacked.size(1)
        if stacked_len > self.pos_emb.size(1):
            raise ValueError(
                f'3 * seq_len ({stacked_len}) > n_ctx ({self.pos_emb.size(1)})'
            )
        stacked = self.embed_ln(stacked + self.pos_emb[:, :stacked_len, :])

        mask = repeat(mask, 'B T -> B (T M)', M=3)
        h = self.gpt2(stacked, mask)
        h = rearrange(h, 'B (T M) H -> B M T H', M=3)  # (B, 3*T, H) -> (B, 3, T, H)

        ac_logits = self.pred_act(h[:, 1])  # h(obs_t) -> act_t          (B, T, act_dim)
        rtg_preds = self.pred_rtg(h[:, 2])  # h(act_t) -> R_{t+1}        (B, T, 1)
        obs_preds = self.pred_obs(h[:, 2])  # h(act_t) -> enc(obs_{t+1}) (B, T, H)

        rtg_h_k = h[:, 0, -self.last_k:, :] if self.pass_h in ('all', 'rtg') else None
        obs_h_k = h[:, 1, -self.last_k:, :] if self.pass_h in ('all', 'obs') else None

        return DTOutput(
            rtg_preds=rtg_preds,
            obs_preds=obs_preds,
            ac_logits=ac_logits,
            obs_enc=obs_enc,
            rtg_h_k=rtg_h_k,
            obs_h_k=obs_h_k,
        )
