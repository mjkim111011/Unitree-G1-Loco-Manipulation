from __future__ import annotations

import numpy as np
import pytest
import torch

from unitree_rl_lab.utils.teacher.mimic_xy_retarget import (
    FixedBoxSceneParameters,
    XYPositionRandomization,
    episode_passes_bimanual_lift_hold,
    retarget_object_and_bimanual_wrists,
)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (x, y, z)
    return pose


def _reference_mimic_transform(
    source_sequence: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Equivalent to Mimic's delta transform, without importing Kit."""

    return delta.unsqueeze(0) @ source_sequence


def _reference_delta_pose(source_object, current_object, constraint):
    assert constraint["coordination_scheme"] == "translate"
    delta = torch.eye(4, dtype=source_object.dtype, device=source_object.device)
    delta[:3, 3] = current_object[:3, 3] - source_object[:3, 3]
    return delta


def test_retarget_moves_object_and_both_wrists_with_same_xy_delta():
    source_object = np.stack(
        [_translation(0.5, 0.0, 0.8), _translation(0.5, 0.0, 0.92)]
    )
    source_left = np.stack(
        [_translation(0.48, 0.12, 0.86), _translation(0.48, 0.12, 0.98)]
    )
    source_right = np.stack(
        [_translation(0.48, -0.12, 0.86), _translation(0.48, -0.12, 0.98)]
    )
    target = _translation(0.63, -0.03, 0.8)

    result = retarget_object_and_bimanual_wrists(
        source_object_world=source_object,
        source_left_wrist_world=source_left,
        source_right_wrist_world=source_right,
        target_initial_object_world=target,
        mimic_transform=_reference_mimic_transform,
        mimic_delta_pose=_reference_delta_pose,
        translate_coordination_scheme="translate",
    )

    expected_delta = np.array([0.13, -0.03, 0.0])
    np.testing.assert_allclose(
        result.object_world[:, :3, 3], source_object[:, :3, 3] + expected_delta
    )
    np.testing.assert_allclose(
        result.left_wrist_world[:, :3, 3], source_left[:, :3, 3] + expected_delta
    )
    np.testing.assert_allclose(
        result.right_wrist_world[:, :3, 3], source_right[:, :3, 3] + expected_delta
    )


def test_xy_sampler_is_seeded_and_never_changes_fixed_scene_values():
    randomization = XYPositionRandomization((0.618, 0.622), (-0.002, 0.002))
    first = randomization.sample(np.random.default_rng(9))
    second = randomization.sample(np.random.default_rng(9))
    assert first == second
    assert 0.618 <= first[0] <= 0.622
    assert -0.002 <= first[1] <= 0.002

    fixed = FixedBoxSceneParameters(
        box_size_m=(0.2, 0.2, 0.2),
        box_mass_kg=0.2,
        static_friction=1.5,
        dynamic_friction=1.2,
        restitution=0.0,
        table_surface_height_m=0.74,
    )
    assert fixed.as_dict()["box_yaw_rad"] == 0.0


@pytest.mark.parametrize(
    ("longest_run", "expected"),
    [(99, False), (100, True), (125, True)],
)
def test_success_filter_requires_continuous_bilateral_lift_hold(longest_run, expected):
    length = 150
    lift = np.zeros(length, dtype=bool)
    left = np.zeros(length, dtype=bool)
    right = np.zeros(length, dtype=bool)
    lift[20 : 20 + longest_run] = True
    left[20 : 20 + longest_run] = True
    right[20 : 20 + longest_run] = True
    assert (
        episode_passes_bimanual_lift_hold(
            lift=lift,
            left_contact=left,
            right_contact=right,
            required_hold_steps=100,
        )
        is expected
    )
