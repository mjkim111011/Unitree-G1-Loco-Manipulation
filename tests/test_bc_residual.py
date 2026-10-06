import pytest
import torch

from unitree_rl_lab.utils.bc_residual import compose_residual_action, smoothstep_handoff


def test_compose_residual_action_clips_and_scales():
    nominal = torch.tensor([[1.0, -1.0, 0.5]])
    residual = torch.tensor([[2.0, -2.0, 0.5]])
    result = compose_residual_action(nominal, residual, residual_scale=0.2)
    torch.testing.assert_close(result, torch.tensor([[1.2, -1.2, 0.6]]))


def test_compose_residual_action_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same"):
        compose_residual_action(torch.zeros(1, 2), torch.zeros(2, 2), residual_scale=0.2)


def test_smoothstep_handoff_reaches_proposal_inclusively():
    proposed = torch.ones(3, 2)
    reference = torch.zeros(3, 2)
    step = torch.tensor([0, 1, 2])
    result = smoothstep_handoff(proposed, reference, step, handoff_steps=3)
    torch.testing.assert_close(result[0], torch.zeros(2))
    torch.testing.assert_close(result[1], torch.full((2,), 0.5))
    torch.testing.assert_close(result[2], torch.ones(2))
