from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class ObsEncModel(ABC, nn.Module):
    """Base class for observation encoders.

    Encodes images into fixed-size vectors.
    """
    @property
    @abstractmethod
    def enc_dim(self) -> int: ...

    @abstractmethod
    def forward(self, observations: Tensor) -> Tensor:
        """
        Args:
            observations: (B*T, C, H, W) in [0, 1]
        Returns:
            (B*T, enc_dim)
        """


class DynEncModel(ABC, nn.Module):
    """Base class for dynamics encoders.

    Encodes a trajectory sequence into hidden representations
    used for both self-supervised loss and DynPredModel input.

    Output type is implementation-defined.
    """
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Returns implementation-defined output."""


class DynPredModel(ABC, nn.Module):
    """Base class for dynamics predictors.

    Takes output from a paired DynEncModel and produces class logits.

    Input type is coupled to the paired DynEncModel implementation.
    """
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Returns logits: (B, n_dynamics)"""
