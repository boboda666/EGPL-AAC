"""Event Residual Adapter used by EGPL-AAC.

The module consumes frozen EAT patch tokens and returns temporal event logits
and a zero-initialized residual in the EAT feature dimension. Residual fusion
is kept explicit so the same checkpoint can be evaluated at different alpha
values without retraining.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def eat_tokens_to_temporal(
    tokens: Tensor, time_patches: int = 64, frequency_patches: int = 8
) -> Tensor:
    """Restore the EAT patch grid and average over frequency only."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B,N,D], got {tokens.shape}")
    expected = time_patches * frequency_patches
    if tokens.shape[1] == expected + 1:
        tokens = tokens[:, 1:]
    elif tokens.shape[1] != expected:
        raise ValueError(
            f"expected {expected} patches with an optional CLS token, "
            f"got {tokens.shape[1]}"
        )
    grid = tokens.reshape(
        tokens.shape[0], time_patches, frequency_patches, tokens.shape[-1]
    )
    return grid.mean(dim=2)


class GatedTemporalBlock(nn.Module):
    """Depthwise temporal convolution with a learned sigmoid gate."""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size, padding=(kernel_size - 1) // 2, groups=dim
        )
        self.gate_projection = nn.Conv1d(dim, 2 * dim, 1)
        self.output_projection = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.depthwise(self.norm(inputs).transpose(1, 2))
        value, gate = self.gate_projection(hidden).chunk(2, dim=1)
        hidden = self.output_projection(F.gelu(value) * torch.sigmoid(gate))
        return inputs + self.dropout(hidden.transpose(1, 2))


class EventResidualAdapter(nn.Module):
    """Extract event logits and an EAT-dimensional temporal residual."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 527,
        temporal_bins: int = 64,
        depth: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_bins = temporal_bins
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            GatedTemporalBlock(hidden_dim, kernel_size, dropout)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.event_head = nn.Linear(hidden_dim, num_classes)
        self.residual_projection = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, eat_tokens: Tensor) -> dict[str, Tensor]:
        hidden = eat_tokens_to_temporal(eat_tokens, self.temporal_bins, 8)
        hidden = F.gelu(self.input_projection(self.input_norm(hidden)))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        frame_logits = self.event_head(hidden)
        clip_logits = torch.logsumexp(frame_logits, dim=1) - math.log(
            self.temporal_bins
        )
        return {
            "event_tokens": hidden,
            "frame_logits": frame_logits,
            "clip_logits": clip_logits,
            "event_residual": self.residual_projection(hidden),
        }

    @staticmethod
    def fuse(
        eat_features: Tensor,
        frame_logits: Tensor,
        event_residual: Tensor,
        alpha: float = 0.1,
    ) -> Tensor:
        """Apply the temporally gated residual to EAT features."""
        gate = torch.sigmoid(frame_logits.amax(dim=-1, keepdim=True))
        target_length = eat_features.shape[1]
        if gate.shape[1] != target_length:
            gate = F.interpolate(
                gate.float().transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(eat_features.dtype)
            event_residual = F.interpolate(
                event_residual.float().transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(eat_features.dtype)
        return eat_features + alpha * gate * event_residual
