"""Small training utilities for the final LoRA-only preference stage."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .preference import length_normalized_log_likelihood


def freeze_all_except_lora(model: nn.Module) -> list[str]:
    """Freeze a model and enable gradients only for parameters named LoRA.

    Returns the trainable parameter names and fails loudly if the supplied
    model has no LoRA parameters, preventing a silent no-op training run.
    """
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError("no LoRA parameters were found in the model")
    return trainable


def token_log_probabilities(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return teacher-forced token log-probabilities and their valid mask.

    ``labels == -100`` follows the Hugging Face convention for ignored prompt
    or padding positions. Inputs are shifted for causal language modelling.
    """
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~valid, 0), valid


def sequence_log_likelihood(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute the length-normalized caption log-likelihood used by DPO."""
    token_logp, mask = token_log_probabilities(logits, labels)
    return length_normalized_log_likelihood(token_logp, mask)


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    """Count scalar parameters in an iterable."""
    return sum(parameter.numel() for parameter in parameters)
