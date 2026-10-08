import torch
from torch import nn

from egpl_aac import (
    EventResidualAdapter,
    dpo_loss,
    freeze_all_except_lora,
    length_normalized_log_likelihood,
    sequence_log_likelihood,
)


def test_event_adapter_shapes_and_zero_initial_residual():
    model = EventResidualAdapter()
    outputs = model(torch.randn(2, 513, 768))
    assert outputs["frame_logits"].shape == (2, 64, 527)
    assert outputs["clip_logits"].shape == (2, 527)
    assert outputs["event_residual"].shape == (2, 64, 768)
    assert torch.count_nonzero(outputs["event_residual"]) == 0


def test_length_normalized_log_likelihood():
    values = torch.tensor([[-1.0, -3.0, -9.0], [-2.0, -4.0, -6.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    result = length_normalized_log_likelihood(values, mask)
    assert torch.allclose(result, torch.tensor([-2.0, -4.0]))


def test_dpo_anchor_is_nonnegative():
    loss, stats = dpo_loss(
        chosen_logp=torch.tensor([-2.0]),
        rejected_logp=torch.tensor([-3.0]),
        reference_chosen_logp=torch.tensor([-1.0]),
        reference_rejected_logp=torch.tensor([-3.0]),
        anchor_weight=0.01,
    )
    assert loss.ndim == 0
    assert stats["anchor_loss"] > 0


def test_anchor_matches_final_linear_hinge():
    _, stats = dpo_loss(
        chosen_logp=torch.tensor([-1.12, -1.01]),
        rejected_logp=torch.tensor([-2.0, -2.0]),
        reference_chosen_logp=torch.tensor([-1.0, -1.0]),
        reference_rejected_logp=torch.tensor([-2.0, -2.0]),
        anchor_weight=0.01,
        anchor_tolerance=0.02,
    )
    # Per-example hinge values are 0.10 and 0.00; the batch mean is 0.05.
    assert torch.allclose(stats["anchor_loss"], torch.tensor(0.05))


def test_sequence_log_likelihood_and_lora_freezing():
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, 0, 1, 2]])
    result = sequence_log_likelihood(logits, labels)
    assert torch.allclose(result, torch.tensor([-torch.log(torch.tensor(3.0))]))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.lora_A = nn.Linear(2, 1, bias=False)

    model = TinyModel()
    names = freeze_all_except_lora(model)
    assert names == ["lora_A.weight"]
    assert not model.backbone.weight.requires_grad
    assert model.lora_A.weight.requires_grad
