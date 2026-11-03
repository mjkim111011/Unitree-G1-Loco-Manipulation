import math

import pytest
import torch

from unitree_rl_lab.utils.teacher.com_stabilizer import (
    ComStabilizerCfg,
    build_reference,
    contact_mode,
    desired_com_shift,
    filter_shift,
    joint_offsets,
    mode_gain,
)


def test_contact_modes():
    assert contact_mode(True, True) == "double"
    assert contact_mode(True, False) == "left"
    assert contact_mode(False, True) == "right"
    assert contact_mode(False, False) is None


def test_gain_schedule_ramps_in_and_is_reduced_in_single_support():
    cfg = ComStabilizerCfg().validate()
    full = int(cfg.ramp_in_seconds / cfg.step_dt)
    assert mode_gain("double", 0, cfg) == pytest.approx(cfg.gain / full)
    assert mode_gain("double", full, cfg) == pytest.approx(cfg.gain)
    assert mode_gain("left", full, cfg) == pytest.approx(cfg.gain * cfg.single_support_gain_scale)
    assert mode_gain(None, full, cfg) == 0.0


def test_shift_opposes_capture_point_error_and_is_clipped():
    cfg = ComStabilizerCfg().validate()
    shift = desired_com_shift(torch.tensor([0.03, -0.01]), (0.01, 0.0), 1.0, cfg)
    torch.testing.assert_close(shift, torch.tensor([-0.02, 0.01]))
    big = desired_com_shift(torch.tensor([1.0, -1.0]), (0.0, 0.0), 1.0, cfg)
    torch.testing.assert_close(big, torch.tensor([-cfg.max_shift_m, cfg.max_shift_m]))


def test_filter_converges_to_target():
    cfg = ComStabilizerCfg().validate()
    shift = torch.zeros(2)
    target = torch.tensor([0.02, -0.01])
    for _ in range(100):
        shift = filter_shift(shift, target, cfg)
    torch.testing.assert_close(shift, target, atol=1e-6, rtol=0)


def test_joint_offsets_realize_the_shift_within_limits():
    cfg = ComStabilizerCfg(damping=1e-4).validate()
    torch.manual_seed(0)
    jacobian = torch.randn(2, 15, dtype=torch.float64)
    shift = torch.tensor([0.01, -0.02], dtype=torch.float64)
    offsets = joint_offsets(jacobian, shift, cfg)
    torch.testing.assert_close(jacobian @ offsets, shift, atol=1e-6, rtol=0)
    assert float(offsets.abs().max()) <= cfg.max_joint_offset_rad


def test_reference_is_median_per_mode_and_requires_every_mode():
    samples = [("double", (0.0, 0.0)), ("double", (0.02, 0.01)), ("double", (0.5, 0.5)),
               ("left", (0.01, 0.1)), ("right", (0.01, -0.1))]
    reference = build_reference(samples)
    assert reference["reference_xy"]["double"] == (0.02, 0.01)
    assert reference["sample_counts"] == {"double": 3, "left": 1, "right": 1}
    with pytest.raises(ValueError, match="right"):
        build_reference(samples[:4])


def test_invalid_cfg():
    with pytest.raises(ValueError):
        ComStabilizerCfg(single_support_gain_scale=2.0).validate()
    with pytest.raises(ValueError):
        ComStabilizerCfg(reference_xy={"double": (0.0, 0.0)}).validate()
