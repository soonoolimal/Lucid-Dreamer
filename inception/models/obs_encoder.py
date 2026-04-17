import warnings

import torch
import torch.nn as nn

from .base import ObsEncModel


class BasicObsEncoder(ObsEncModel):
    """CNN + AdaptiveAvgPool + Linear projection.

    Encodes a regularized RGB image into a vector of enc_dim.
    """
    def __init__(self, enc_dim, in_ch, out_ch, n_layers, img_size, pool_size):
        super().__init__()

        if out_ch % (2 ** (n_layers - 1)) != 0:
            raise ValueError(
                f'out_ch ({out_ch}) must be divisible by '
                f'2^(n_layers-1) ({2 ** (n_layers - 1)})'
            )

        channels = [out_ch // (2 ** (n_layers - 1 - i)) for i in range(n_layers)]
        self.cnn = nn.Sequential(
            *[self._conv_block(a, b) for a, b in zip([in_ch] + channels[:-1], channels)]
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_size, img_size)
            spatial = self.cnn(dummy).shape[-1]

        if pool_size is None:
            resolved = spatial
        elif pool_size > spatial:
            warnings.warn(
                f'pool_size ({pool_size}) > spatial size after CNN ({spatial}). '
                f'Clamping to {spatial}.',
                UserWarning,
            )
            resolved = spatial
        else:
            resolved = pool_size

        self.pool = nn.AdaptiveAvgPool2d((resolved, resolved))
        self.proj = nn.Linear(out_ch * resolved * resolved, enc_dim)

        self._enc_dim = enc_dim

    @property
    def enc_dim(self):
        return self._enc_dim

    def forward(self, observations):
        # (B*T, C, H, W) -> (B*T, enc_dim)
        z = self.cnn(observations)
        z = self.pool(z)
        z = z.flatten(start_dim=1)
        return self.proj(z)

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
