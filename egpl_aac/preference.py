"""Preference objectives used in the final EGPL-AAC experiments."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def length_normalized_log_likelihood(
    token_log_probs: Tensor, token_mask: Tensor
) -> Tensor:
    """Return mean token log-likelihood for every caption in a batch."""
    mask = token_mask.to(token_log_probs.dtype)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def dpo_loss(
    chosen_logp: Tensor,
    rejected_logp: Tensor,
    reference_chosen_logp: Tensor,
    reference_rejected_logp: Tensor,
    beta: float = 0.05,
    anchor_weight: float = 0.0,
    anchor_tolerance: float = 0.02,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute ordinary DPO with the optional Clotho chosen-caption anchor.

    All log-probability inputs are expected to be length-normalized. The final
    AudioCaps setting uses ``anchor_weight=0``; the final Clotho setting uses
    ``anchor_weight=0.01``.
    """
    policy_margin = chosen_logp - rejected_logp
    reference_margin = reference_chosen_logp - reference_rejected_logp
    logits = beta * (policy_margin - reference_margin)
    preference_loss = -F.logsigmoid(logits).mean()

    degradation = reference_chosen_logp - chosen_logp - anchor_tolerance
    anchor_loss = degradation.clamp_min(0).square().mean()
    total = preference_loss + anchor_weight * anchor_loss
    return total, {
        "preference_loss": preference_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "preference_accuracy": (logits > 0).float().mean().detach(),
    }
