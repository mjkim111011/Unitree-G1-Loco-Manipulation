import math

import pytest
import torch

from unitree_rl_lab.utils.teacher.capture_command_guard import CommandGuardCfg, command_scale
from unitree_rl_lab.utils.teacher.transition_diagnostics import support_polygon
from unitree_rl_lab.utils.teacher.com_predictive_blend import signed_support_margin


def test_inside_envelope_and_deadband_leave_commands_unchanged():
    cfg = CommandGuardCfg()
    for margin in (.1, 0., -.01, -.02):
        assert command_scale(1., margin, 10, cfg) == (1., 1.)


def test_feedback_attenuates_only_outside_and_is_rate_limited():
    cfg = CommandGuardCfg()
    scale = 1.
    for i in range(80):
        new, target = command_scale(scale, -.2, i, cfg)
        assert cfg.minimum_scale <= new <= 1.
        assert abs(new-scale) <= cfg.attenuation_rate_per_second*cfg.step_dt+1e-9
        assert target == pytest.approx(cfg.minimum_scale)
        scale = new
    assert scale == pytest.approx(cfg.minimum_scale)
    # Recovery is also bounded: it cannot release the guard in one step.
    new, target = command_scale(scale, .1, 80, cfg)
    assert new == pytest.approx(scale+cfg.recovery_rate_per_second*cfg.step_dt)
    assert target == 1.


@pytest.mark.parametrize('margin', [None, float('nan'), float('inf')])
def test_missing_measurement_does_not_invent_a_balance_error(margin):
    assert command_scale(1., margin, 0, CommandGuardCfg()) == (1., 1.)


def test_guard_releases_gradually_after_its_window():
    cfg = CommandGuardCfg()
    scale, target = command_scale(.2, -.2, 150, cfg)
    assert scale == pytest.approx(.22) and target == 1.


def test_support_geometry_rotates_with_the_robot_and_handles_flight():
    feet = torch.tensor([[0., .1, 0.], [0., -.1, 0.]])
    quat = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])
    polygon = support_polygon(feet, quat, [True, True], torch.zeros(3), torch.eye(2))
    cp = torch.tensor([[.02, -.03]])
    margin = signed_support_margin(cp, polygon)
    yaw = .7
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    translated = feet.clone()
    translated[:, :2] = feet[:, :2]@rotation.T + torch.tensor([3., -2.])
    quat[:, 0], quat[:, 3] = math.cos(yaw/2), math.sin(yaw/2)
    transformed = support_polygon(translated, quat, [True, True], torch.tensor([3., -2., 0.]), rotation.T)
    torch.testing.assert_close(signed_support_margin(cp, transformed), margin)
    assert support_polygon(feet, quat, [False, False], torch.zeros(3), torch.eye(2)).shape == (0, 2)
