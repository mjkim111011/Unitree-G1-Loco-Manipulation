import pytest
import torch

from unitree_rl_lab.utils.teacher.transition_gate import (
    TransitionGateCfg,
    com_jacobian,
    com_shift_joint_offset,
    compose_transition_action,
    fixed_blend_weight,
    quintic_blend,
    update_alpha,
)


def _legacy_blend_weight(carry_step: int, blend_steps: int) -> float:
    # The inline formula evaluate_bc2_rollout.py used before the refactor.
    if not 0 <= carry_step < blend_steps:
        return 1.0
    fraction = (carry_step + 1) / float(blend_steps)
    return fraction ** 3 * (10.0 - 15.0 * fraction + 6.0 * fraction ** 2)


@pytest.mark.parametrize("blend_steps", [0, 1, 25, 50, 100])
def test_fixed_blend_weight_reproduces_baseline(blend_steps):
    for step in range(blend_steps + 5):
        assert fixed_blend_weight(step, blend_steps) == pytest.approx(
            _legacy_blend_weight(step, blend_steps), abs=1.0e-12
        )
    if blend_steps:
        assert fixed_blend_weight(blend_steps - 1, blend_steps) == pytest.approx(1.0)


def test_quintic_blend_matches_tensor_and_float():
    fractions = torch.linspace(-0.5, 1.5, 21, dtype=torch.float64)
    expected = torch.tensor([quintic_blend(float(f)) for f in fractions], dtype=torch.float64)
    torch.testing.assert_close(quintic_blend(fractions), expected)


def test_update_alpha_is_monotone_bounded_and_forced_to_complete():
    cfg = TransitionGateCfg().validate()
    torch.manual_seed(0)
    alpha = torch.zeros(64)
    for step in range(cfg.max_steps):
        gate = torch.randn(64) * 3.0
        updated = update_alpha(alpha, gate, torch.full((64,), step), cfg)
        assert torch.all(updated >= alpha)
        assert torch.all((updated >= 0.0) & (updated <= 1.0))
        if step < cfg.max_steps - cfg.forced_ramp_steps:
            assert torch.all(updated - alpha <= cfg.max_alpha_rate + 1.0e-6)
        alpha = updated
    torch.testing.assert_close(alpha, torch.ones(64))


def test_update_alpha_hold_and_full_rate():
    cfg = TransitionGateCfg().validate()
    alpha = torch.zeros(2)
    gate = torch.tensor([-1.0, 1.0])
    for step in range(cfg.max_steps - cfg.forced_ramp_steps):
        alpha = update_alpha(alpha, gate, torch.full((2,), step), cfg)
    assert alpha[0] == 0.0
    assert alpha[1] == 1.0


def test_update_alpha_rejects_bad_shapes():
    cfg = TransitionGateCfg()
    with pytest.raises(ValueError, match="shape"):
        update_alpha(torch.zeros(2), torch.zeros(3), torch.zeros(2), cfg)


def test_cfg_validation():
    with pytest.raises(ValueError):
        TransitionGateCfg(min_seconds=3.0, max_seconds=2.0).validate()


def test_compose_transition_action_endpoints_and_lock():
    anchor = torch.zeros(2, 4)
    bc3 = torch.ones(2, 4)
    reference = torch.full((2, 4), 7.0)
    offset = torch.full((2, 4), 0.5)
    alpha = torch.tensor([0.0, 1.0])
    action = compose_transition_action(
        anchor, bc3, alpha, locked_indices=[3], locked_reference=reference, com_offset=offset
    )
    torch.testing.assert_close(action[0, :3], torch.full((3,), 0.5))
    torch.testing.assert_close(action[1, :3], torch.ones(3))
    torch.testing.assert_close(action[:, 3], torch.full((2,), 7.0))


def test_com_jacobian_is_mass_weighted_mean():
    jacobians = torch.zeros(1, 2, 3, 2)
    jacobians[0, 0, 0, 0] = 1.0
    jacobians[0, 1, 1, 1] = 1.0
    masses = torch.tensor([[3.0, 1.0]])
    result = com_jacobian(jacobians, masses)
    torch.testing.assert_close(result[0, 0], torch.tensor([0.75, 0.0]))
    torch.testing.assert_close(result[0, 1], torch.tensor([0.0, 0.25]))


def test_com_shift_joint_offset_realizes_small_shift_and_clips():
    torch.manual_seed(1)
    jacobian = torch.randn(4, 2, 15, dtype=torch.float64)
    shift = torch.tensor([[0.01, -0.02]] * 4, dtype=torch.float64)
    offset = com_shift_joint_offset(jacobian, shift, max_shift_m=0.03, damping=1.0e-4)
    realized = (jacobian @ offset.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(realized, shift, atol=1.0e-6, rtol=0.0)
    big = com_shift_joint_offset(jacobian, shift * 100.0, max_shift_m=0.03, damping=1.0e-4)
    realized_big = (jacobian @ big.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(realized_big, torch.tensor([[0.03, -0.03]] * 4, dtype=torch.float64), atol=1.0e-6, rtol=0.0)
