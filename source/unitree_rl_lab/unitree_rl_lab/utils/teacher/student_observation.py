"""Contract for the Box task's student-only QR/proprioceptive observation.

The teacher datasets intentionally contain privileged poses for validation and
retargeting.  A student policy must never concatenate those datasets into its
input.  This module makes the actor boundary explicit and provides the one
supported way of applying a synthetic QR dropout during DAgger collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


STUDENT_OBSERVATION_SCHEMA = "unitree_box_qr_proprio_v1"
STUDENT_OBSERVATION_DIM = 517
POLICY_HISTORY_DIM = 460


@dataclass(frozen=True, slots=True)
class FeatureSlice:
    """Named half-open slice in the flattened student observation."""

    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


# policy_history is 5 * (base_ang_vel[3] + projected_gravity[3] +
# joint_pos_rel[43] + joint_vel_rel[43]).  policy_current follows it in the
# exact declaration order in BoxPickPlaceObservationsCfg.PolicyCurrentCfg.
STUDENT_FEATURE_SLICES: Mapping[str, FeatureSlice] = {
    "policy_history": FeatureSlice(0, 460),
    "last_action": FeatureSlice(460, 503),
    "velocity_commands": FeatureSlice(503, 506),
    "box_visible": FeatureSlice(506, 507),
    "box_position_camera": FeatureSlice(507, 510),
    "bin_visible": FeatureSlice(510, 511),
    "bin_position_camera": FeatureSlice(511, 514),
    "table_position_camera": FeatureSlice(514, 517),
}

PRIVILEGED_DATASET_NAMES = frozenset(
    {
        "box_pose",
        "bin_pose",
        "root_pose",
        "left_wrist_pose",
        "right_wrist_pose",
        "left_fingertip_positions",
        "right_fingertip_positions",
        "box_size",
        "left_contact",
        "right_contact",
        "left_contact_count",
        "right_contact_count",
        "lift",
        "drop",
        "success",
    }
)


def schema_metadata() -> dict[str, Any]:
    """Return JSON-serializable metadata for newly recorded demonstrations."""

    return {
        "student_observation_schema": STUDENT_OBSERVATION_SCHEMA,
        "student_observation_dim": STUDENT_OBSERVATION_DIM,
        "student_observation_groups": ["policy_history", "policy_current"],
        "student_observation_feature_slices": {
            name: [feature.start, feature.stop]
            for name, feature in STUDENT_FEATURE_SLICES.items()
        },
        "student_privileged_inputs": [],
        "privileged_fields_recorded_for_audit_only": sorted(PRIVILEGED_DATASET_NAMES),
    }


def validate_student_observation(value: Any, *, allow_batch: bool = True) -> np.ndarray:
    """Validate and return a float32 student observation without copying if possible."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    valid_rank = array.ndim == 1 or (allow_batch and array.ndim == 2)
    if not valid_rank or array.shape[-1] != STUDENT_OBSERVATION_DIM:
        expected = f"({STUDENT_OBSERVATION_DIM},)"
        if allow_batch:
            expected += f" or (N, {STUDENT_OBSERVATION_DIM})"
        raise ValueError(
            f"student observation has shape {array.shape}; expected {expected}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("student observation must contain only finite numeric values")
    return array.astype(np.float32, copy=False)


def mask_qr_observation(
    observation: Any,
    *,
    box: bool = False,
    bin: bool = False,
    copy: bool = True,
) -> np.ndarray:
    """Mask QR visibility and camera position using the deployed observation layout.

    This is an observation-only intervention.  The simulator object pose and
    the privileged teacher state are not changed, so a teacher can still label
    the same physical state during a QR-occlusion DAgger segment.
    """

    if not box and not bin:
        raise ValueError("at least one of box=True or bin=True is required")
    result = validate_student_observation(observation)
    if copy:
        result = result.copy()
    names: list[str] = []
    if box:
        names.extend(("box_visible", "box_position_camera"))
    if bin:
        names.extend(("bin_visible", "bin_position_camera"))
    for name in names:
        result[..., STUDENT_FEATURE_SLICES[name].as_slice()] = 0.0
    return result


def replace_velocity_command(
    observation: Any, command: Any, *, copy: bool = True
) -> np.ndarray:
    """Write the phase-owned ``vx, vy, wz`` command into a student vector."""

    result = validate_student_observation(observation)
    command_array = np.asarray(command, dtype=np.float32)
    expected = (3,) if result.ndim == 1 else (result.shape[0], 3)
    if command_array.shape != expected or not np.isfinite(command_array).all():
        raise ValueError(f"velocity command has shape {command_array.shape}; expected {expected}")
    if copy:
        result = result.copy()
    result[..., STUDENT_FEATURE_SLICES["velocity_commands"].as_slice()] = command_array
    return result


def validate_dataset_student_boundary(episode: Any) -> int:
    """Validate the only dataset a BC/DAgger student loader may consume.

    ``episode`` is deliberately duck-typed so this helper does not require an
    HDF5 import.  Privileged datasets may coexist in the group, but this
    function returns only the verified student input dimension and never a
    privileged value.
    """

    if "student_observation" not in episode:
        raise KeyError("episode has no student_observation dataset")
    dataset = episode["student_observation"]
    if len(dataset.shape) != 2 or int(dataset.shape[1]) != STUDENT_OBSERVATION_DIM:
        raise ValueError(
            f"student_observation has shape {dataset.shape}; expected (N, {STUDENT_OBSERVATION_DIM})"
        )
    return STUDENT_OBSERVATION_DIM
