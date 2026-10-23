"""Explicit critic-output-bound ablation; actor and task remain unchanged."""

import torch
from torch import nn

from .bc_ppo_policy import BCMeanGaussianPolicy


class ValueBound(nn.Module):
    def __init__(self, bound=100.0):
        super().__init__()
        self.bound = float(bound)

    def forward(self, value):
        return self.bound * torch.tanh(value / self.bound)


class BoundedCriticPolicy(BCMeanGaussianPolicy):
    """Soft ±100 value bound, preserving slope1 near zero.

    With normalized rewards in[0,1] and gamma.99, true continuing values lie
    in[0,100]. Negative outputs remain permitted; no target/reward is clipped.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The existing Linear parameters remain the same optimizer objects.
        self.value_net = nn.Sequential(self.value_net, ValueBound())
