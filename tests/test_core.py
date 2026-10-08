import torch

from egpl_aac import (
    EventResidualAdapter,
    dpo_loss,
    length_normalized_log_likelihood,
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
