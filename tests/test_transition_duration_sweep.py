import copy
import math

import pytest

from scripts.teacher.evaluate_transition_duration_sweep import duration_steps, paired_reports


@pytest.mark.parametrize("duration,steps", [(0.26, 13), (0.5, 25), (1, 50), (1.5, 75), (2, 100)])
def test_duration_uses_exact_control_steps(duration, steps):
    assert duration_steps(duration) == steps


@pytest.mark.parametrize("duration", [0, -1, 0.25, math.nan, math.inf])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError):
        duration_steps(duration)


def episode(snapshot_id, success, valid=True):
    return {
        "snapshot_id": snapshot_id, "restore_success": valid,
        "carry": {"success": success, "reason": "carry_success" if success else "carry_horizon"},
        "start_features": {
            "robot_com_relative_feet_yaw_xy_m": [0.02, 0.01],
            "foot_load_fraction": [0.5, 0.5], "base_angular_velocity_body": [0.1, 0, 0],
            "joint_position": [0, 0], "box_state": [0.6, 0, 1.0],
        },
    }


def test_reports_retain_state_dependent_duration_outcomes():
    results = {
        0.5: {"episode_results": [episode("A", True), episode("B", False)]},
        1.5: {"episode_results": [episode("A", False), episode("B", True)]},
    }
    summary, states = paired_reports(results)
    assert [row["common_state_success_rate"] for row in summary] == [0.5, 0.5]
    assert states[0]["successful_durations_s"] == "[0.5]"
    assert states[1]["successful_durations_s"] == "[1.5]"
    assert states[0]["postwarm_joint_difference_max_rad"] == 0


def test_restore_failures_are_not_attributed_to_duration():
    results = {
        0.5: {"episode_results": [episode("A", True), episode("B", False, valid=False)]},
        1.5: {"episode_results": [episode("A", True), episode("B", True)]},
    }
    summary, states = paired_reports(results)
    assert [r["common_valid_states"] for r in summary] == [1, 1]
    assert [r["common_state_success_rate"] for r in summary] == [1.0, 1.0]
    assert not states[1]["common_valid_restore"]


def test_state_order_mismatch_is_rejected():
    results = {
        0.5: {"episode_results": [episode("A", True), episode("B", False)]},
        1.5: {"episode_results": [episode("B", True), episode("A", False)]},
    }
    with pytest.raises(ValueError, match="selection/order"):
        paired_reports(results)


def test_no_common_valid_states_reports_undefined_rate():
    summary, _ = paired_reports({0.5: {"episode_results": [episode("A", False, valid=False)]}})
    assert summary[0]["common_state_success_rate"] is None


def test_reports_measure_actual_postwarm_state_difference():
    first = episode("A", True)
    second = copy.deepcopy(first)
    second["start_features"]["joint_position"][0] = 0.001
    _, states = paired_reports({0.5: {"episode_results": [first]}, 1.5: {"episode_results": [second]}})
    assert states[0]["postwarm_joint_difference_max_rad"] == pytest.approx(0.001)
