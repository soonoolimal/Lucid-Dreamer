import torch
import torch.nn as nn

from .base import DynPredModel
from .dyn_encoder import DTOutput


class DTPredictor(DynPredModel):
    """Dynamics predictor paired with DecisionTransformer.

    Classifies dynamics type from DT hidden states (rtg_h_k, obs_h_k).
    """
    def __init__(self, hidden_size, last_k, pass_h, p_dim, n_dynamics):
        super().__init__()

        if pass_h not in ('all', 'rtg', 'obs'):
            raise ValueError(f"pass_h must be 'all', 'rtg', or 'obs', got {pass_h!r}")

        self.pass_h = pass_h
        in_dim = last_k * hidden_size  # hidden_size and last_k must match those of the paired DecisionTransformer

        if pass_h in ('all', 'rtg'):
            self.rtg_branch = nn.Sequential(
                nn.Linear(in_dim, p_dim),
                nn.LayerNorm(p_dim),
                nn.ReLU(inplace=True),
                nn.Linear(p_dim, p_dim),
            )

        if pass_h in ('all', 'obs'):
            self.obs_branch = nn.Sequential(
                nn.Linear(in_dim, p_dim),
                nn.LayerNorm(p_dim),
                nn.ReLU(inplace=True),
                nn.Linear(p_dim, p_dim),
            )

        clf_in_dim = 2 * p_dim if pass_h == 'all' else p_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, p_dim),
            nn.ReLU(inplace=True),
            nn.Linear(p_dim, n_dynamics),
        )

    def forward(self, enc_out: DTOutput):
        """
        Args:
            enc_out: DTOutput from DecisionTransformer.
        Returns:
            logits: (B, n_dynamics).
        """
        parts = []
        if enc_out.rtg_h_k is not None:
            parts.append(self.rtg_branch(enc_out.rtg_h_k.flatten(start_dim=1)))
        if enc_out.obs_h_k is not None:
            parts.append(self.obs_branch(enc_out.obs_h_k.flatten(start_dim=1)))

        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.classifier(fused)
