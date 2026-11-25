import pytest
import torch

from unitree_rl_lab.utils.teacher.com_consistent_blend import (
    ComConsistentBlendCfg,
    consistent_correction,
    faded_gain,
    predicted_capture,
    target_gain,
)


def setup(dtype=torch.float64):
    torch.manual_seed(0)
    jacobian = torch.randn(2, 15, dtype=dtype) * 0.2
    wbc = torch.zeros(15, dtype=dtype)
    bc3 = torch.randn(15, dtype=dtype) * 0.05
    return jacobian, wbc, bc3


def test_prediction_is_first_order():
    jacobian, wbc, bc3 = setup()
    capture = torch.tensor([0.01, -0.02], dtype=torch.float64)
    torch.testing.assert_close(predicted_capture(capture, jacobian, wbc, wbc), capture)
    torch.testing.assert_close(predicted_capture(capture, jacobian, bc3, wbc), capture + jacobian @ bc3)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.6, 0.9])
def test_general_error_equals_the_anchor_drift_form(alpha):
    # e = xi_des - xi_pred(q_blend) reduces to (1 - alpha)(xi_start - xi_pred(q_WBC)) for a linear model.
    jacobian, wbc, bc3 = setup()
    applied = torch.randn(15, dtype=torch.float64) * 0.02
    capture = torch.tensor([0.02, -0.01], dtype=torch.float64)
    start = torch.tensor([0.0, 0.005], dtype=torch.float64)
    _, info = consistent_correction(alpha, capture, jacobian, wbc, bc3, applied, start, ComConsistentBlendCfg())
    expected = (1 - alpha) * (start - predicted_capture(capture, jacobian, wbc, applied))
    torch.testing.assert_close(torch.tensor(info["capture_error_xy_m"], dtype=torch.float64), expected)


def test_correction_vanishes_at_the_end_of_the_blend():
    jacobian, wbc, bc3 = setup()
    capture = torch.tensor([0.03, 0.02], dtype=torch.float64)  # drifted from the start
    offsets, info = consistent_correction(1.0, capture, jacobian, wbc, bc3, wbc,
                                          torch.zeros(2, dtype=torch.float64), ComConsistentBlendCfg())
    torch.testing.assert_close(offsets, torch.zeros_like(offsets))


def test_no_drift_means_no_correction():
    jacobian, wbc, bc3 = setup()
    start = torch.tensor([0.02, 0.0], dtype=torch.float64)
    for alpha in (0.0, 0.3, 0.7):
        offsets, _ = consistent_correction(alpha, start, jacobian, wbc, bc3, wbc, start, ComConsistentBlendCfg())
        torch.testing.assert_close(offsets, torch.zeros_like(offsets))


def test_corrected_blend_lands_on_the_straight_capture_path():
    jacobian, wbc, bc3 = setup()
    cfg = ComConsistentBlendCfg(damping=1.0e-5)
    start = torch.zeros(2, dtype=torch.float64)
    drifted = torch.tensor([0.0, -0.015], dtype=torch.float64)  # sideways sway during the blend
    alpha = 0.4
    offsets, info = consistent_correction(alpha, drifted, jacobian, wbc, bc3, wbc, start, cfg)
    blended = wbc + alpha * (bc3 - wbc)
    corrected = predicted_capture(drifted, jacobian, blended + offsets, wbc)
    torch.testing.assert_close(corrected, torch.tensor(info["desired_capture_xy_m"], dtype=torch.float64),
                               atol=1e-6, rtol=0)


def test_shift_and_joint_offsets_are_clipped():
    jacobian, wbc, bc3 = setup()
    cfg = ComConsistentBlendCfg(max_shift_m=0.01, max_joint_offset_rad=0.02)
    offsets, info = consistent_correction(0.0, torch.tensor([0.5, -0.5], dtype=torch.float64), jacobian,
                                          wbc, bc3, wbc, torch.zeros(2, dtype=torch.float64), cfg)
    assert max(abs(v) for v in info["com_shift_xy_m"]) <= 0.01 + 1e-12
    assert float(offsets.abs().max()) <= 0.02 + 1e-12


def test_gain_targets_by_contact_mode():
    cfg = ComConsistentBlendCfg(gain=2.0, single_support_gain_scale=0.25)
    assert target_gain("double", cfg) == 2.0
    assert target_gain("left", cfg) == 0.5
    assert target_gain(None, cfg) == 0.0


def test_gain_fades_out_over_the_fade_time_instead_of_jumping():
    # A foot lifting must not remove the correction in one control step.
    cfg = ComConsistentBlendCfg().validate()
    steps = round(cfg.gain_fade_seconds / cfg.step_dt)
    gain, history = cfg.gain, []
    for _ in range(steps + 2):
        gain = faded_gain(gain, 0.0, cfg)
        history.append(gain)
    max_step = cfg.gain * cfg.step_dt / cfg.gain_fade_seconds
    assert all(abs(b - a) <= max_step + 1e-12 for a, b in zip([cfg.gain] + history, history))
    assert history[0] > 0.0 and history[-1] == 0.0


def test_invalid_inputs():
    jacobian, wbc, bc3 = setup()
    with pytest.raises(ValueError):
        consistent_correction(1.5, torch.zeros(2), jacobian, wbc, bc3, wbc, torch.zeros(2), ComConsistentBlendCfg())
    with pytest.raises(ValueError):
        consistent_correction(0.5, torch.zeros(2), jacobian[:, :3], wbc, bc3, wbc, torch.zeros(2),
                              ComConsistentBlendCfg())
    with pytest.raises(ValueError):
        ComConsistentBlendCfg(gain_fade_seconds=0.0).validate()


def test_lead_one_is_the_original_m2():
    jacobian, wbc, bc3 = setup()
    capture = torch.tensor([0.02, -0.01], dtype=torch.float64)
    start = torch.tensor([0.0, 0.005], dtype=torch.float64)
    for alpha in (0.0, 0.3, 0.8):
        original, _ = consistent_correction(alpha, capture, jacobian, wbc, bc3, wbc, start, ComConsistentBlendCfg())
        lead_one, _ = consistent_correction(alpha, capture, jacobian, wbc, bc3, wbc, start,
                                            ComConsistentBlendCfg(lead=1.0))
        torch.testing.assert_close(lead_one, original)


def test_lead_assists_bc3s_intended_weight_shift():
    # No drift: the original M2 does nothing, a lead moves the capture point toward BC3's prediction.
    jacobian, wbc, bc3 = setup()
    start = torch.zeros(2, dtype=torch.float64)
    bc3_shift = jacobian @ (bc3 - wbc)
    for alpha in (0.1, 0.3, 0.45):
        _, info = consistent_correction(alpha, start, jacobian, wbc, bc3, wbc, start,
                                        ComConsistentBlendCfg(lead=2.0))
        error = torch.tensor(info["capture_error_xy_m"], dtype=torch.float64)
        torch.testing.assert_close(error, (min(1.0, 2.0 * alpha) - alpha) * bc3_shift)
        assert float(error @ bc3_shift) > 0.0


@pytest.mark.parametrize("lead", [1.0, 2.0, 4.0])
def test_correction_still_vanishes_at_alpha_one_with_lead(lead):
    jacobian, wbc, bc3 = setup()
    capture = torch.tensor([0.03, 0.02], dtype=torch.float64)
    offsets, info = consistent_correction(1.0, capture, jacobian, wbc, bc3, wbc,
                                          torch.zeros(2, dtype=torch.float64), ComConsistentBlendCfg(lead=lead))
    torch.testing.assert_close(offsets, torch.zeros_like(offsets))
    assert info["beta"] == 1.0


def test_invalid_lead():
    with pytest.raises(ValueError):
        ComConsistentBlendCfg(lead=0.0).validate()
