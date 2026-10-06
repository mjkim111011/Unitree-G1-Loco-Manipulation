"""Audit and retarget IsaacLab-Arena G1 loco-manipulation demonstrations.

The Arena demonstration recorder mixes pre-step observations/actions with
post-step states.  This module preserves the raw post-step states and also
constructs an aligned pre-step object pose sequence (initial state followed by
the previous post-step state).  All wrist/object retargeting transforms use the
aligned sequence.

Quaternion conventions in the source file are intentionally explicit:

* articulation and rigid-object pose vectors are ``[x, y, z, qw, qx, qy, qz]``;
* the 23-D raw action wrist quaternion fields are ``[qx, qy, qz, qw]``;
* observed pelvis-relative wrist poses are already homogeneous 4x4 matrices.

No camera arrays are read or copied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SCHEMA_VERSION = "unitree-arena-teacher-v1"
PHASE_NAMES = ("approach", "grasp", "carry", "place")

# Source: IsaacLab-Arena/isaaclab_arena_g1/g1_env/config/
#         lab_g1_joints_order_43dof.yaml (Arena 0.3.0).
# The source HDF5 does not embed joint names, so keeping this exact Lab order next
# to every extracted processed target is essential.
ARENA_G1_LAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)

LEFT_DEX3_INDICES = tuple(
    i
    for i, name in enumerate(ARENA_G1_LAB_JOINT_NAMES)
    if name.startswith("left_hand_")
)
RIGHT_DEX3_INDICES = tuple(
    i
    for i, name in enumerate(ARENA_G1_LAB_JOINT_NAMES)
    if name.startswith("right_hand_")
)
DEX3_INDICES = LEFT_DEX3_INDICES + RIGHT_DEX3_INDICES

# (name, start inclusive, end exclusive, quaternion convention/semantic)
ARENA_G1_ACTION_LAYOUT = (
    ("left_hand_state", 0, 1, "binary: 0=open, 1=closed"),
    ("right_hand_state", 1, 2, "binary: 0=open, 1=closed"),
    ("left_wrist_position_pelvis", 2, 5, "xyz metres, pelvis frame"),
    ("left_wrist_quaternion_xyzw", 5, 9, "xyzw, pelvis frame"),
    ("right_wrist_position_pelvis", 9, 12, "xyz metres, pelvis frame"),
    ("right_wrist_quaternion_xyzw", 12, 16, "xyzw, pelvis frame"),
    ("base_velocity_command", 16, 19, "vx, vy, wz"),
    ("hip_height_command", 19, 20, "metres"),
    ("torso_orientation_rpy_command", 20, 23, "roll, pitch, yaw radians"),
)


@dataclass(frozen=True)
class PhaseInference:
    """Signal-derived task phase labels for one trajectory."""

    phase_id: np.ndarray
    boundaries: tuple[int, int, int, int, int]
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...] = ()

    @property
    def intervals(self) -> dict[str, tuple[int, int]]:
        return {
            name: (self.boundaries[index], self.boundaries[index + 1])
            for index, name in enumerate(PHASE_NAMES)
        }


@dataclass(frozen=True)
class ArenaTeacherTrajectory:
    """Camera-free, in-memory view used by teacher rollout integrations.

    All arrays are detached NumPy values; the source HDF5 is closed before this
    object is returned.  ``*_observed`` wrist transforms describe realized link
    poses, while ``*_command`` transforms describe the Pink IK targets encoded in
    the raw 23-D action.
    """

    source_path: Path
    episode_name: str
    control_dt_s: float
    control_frequency_hz: float
    high_level_action_23d: np.ndarray
    processed_43d_joint_target: np.ndarray
    phase_id: np.ndarray
    phase_boundaries: tuple[int, int, int, int, int]
    robot_joint_position_pre_step: np.ndarray
    robot_joint_velocity_pre_step: np.ndarray
    root_pose_wxyz_pre_step: np.ndarray
    box_pose_wxyz_pre_step: np.ndarray
    bin_pose_wxyz_pre_step: np.ndarray
    left_world_wrist_observed: np.ndarray
    right_world_wrist_observed: np.ndarray
    left_world_wrist_command: np.ndarray
    right_world_wrist_command: np.ndarray
    left_box_relative_wrist_observed: np.ndarray
    right_box_relative_wrist_observed: np.ndarray
    left_box_relative_wrist_command: np.ndarray
    right_box_relative_wrist_command: np.ndarray
    bin_relative_box: np.ndarray
    left_dex3_joint_target: np.ndarray
    right_dex3_joint_target: np.ndarray
    source_lab_joint_names: tuple[str, ...] = ARENA_G1_LAB_JOINT_NAMES


def _as_python(value: Any) -> Any:
    """Recursively convert NumPy/HDF5 scalars into JSON-safe Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized-or-unnormalized WXYZ quaternions to rotation matrices."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected (..., 4) WXYZ quaternion, got {quaternion.shape}.")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("Cannot convert a zero-norm quaternion.")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions (the Arena raw wrist convention) to matrices."""

    quaternion = np.asarray(quaternion)
    return quaternion_wxyz_to_matrix(quaternion[..., [3, 0, 1, 2]])


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build batched homogeneous transforms from positions and 3x3 rotations."""

    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[:-2] != position.shape[:-1] or rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Incompatible position {position.shape} and rotation {rotation.shape} shapes."
        )
    transform = np.zeros(position.shape[:-1] + (4, 4), dtype=np.float64)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = position
    transform[..., 3, 3] = 1.0
    return transform


def pose7_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``xyz + wxyz`` pose vectors to homogeneous transforms."""

    pose = np.asarray(pose)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected (..., 7) pose, got {pose.shape}.")
    return pose_matrix(pose[..., :3], quaternion_wxyz_to_matrix(pose[..., 3:7]))


def rigid_inverse(transform: np.ndarray) -> np.ndarray:
    """Invert batched homogeneous rigid transforms."""

    transform = np.asarray(transform, dtype=np.float64)
    rotation_t = np.swapaxes(transform[..., :3, :3], -1, -2)
    inverse = np.zeros_like(transform)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", rotation_t, transform[..., :3, 3]
    )
    inverse[..., 3, 3] = 1.0
    return inverse


def _first_sustained_true(mask: np.ndarray, start: int, length: int) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or start >= mask.size:
        return None
    length = max(1, min(length, mask.size - start))
    running = np.convolve(
        mask.astype(np.int32), np.ones(length, dtype=np.int32), mode="valid"
    )
    candidates = np.flatnonzero(running[start:] == length)
    if candidates.size:
        return int(start + candidates[0])

    # A demonstration can terminate soon after placement, leaving less than the
    # nominal one-second confirmation window.  Accept only a terminal true run
    # that reaches EOF and still spans at least one quarter of that window.
    suffix = mask[start:]
    false_indices = np.flatnonzero(~suffix)
    terminal_start = start + (int(false_indices[-1]) + 1 if false_indices.size else 0)
    minimum_terminal_length = max(3, length // 4)
    if mask.size - terminal_start >= minimum_terminal_length:
        return terminal_start
    return None


def infer_task_phases(
    raw_actions: np.ndarray,
    box_pose_wxyz: np.ndarray,
    bin_pose_wxyz: np.ndarray,
    control_dt_s: float,
) -> PhaseInference:
    """Infer approach/grasp/carry/place boundaries from physical/action signals.

    The detector does not encode trajectory indices.  It uses the first confirmed
    box motion, the first subsequent navigation command while the object is lifted,
    and the first sustained navigation stop after transport.  The returned
    diagnostics record thresholds and signal values so inferred labels remain
    auditable rather than masquerading as ground-truth annotations.
    """

    actions = np.asarray(raw_actions)
    box_pose = np.asarray(box_pose_wxyz)
    bin_pose = np.asarray(bin_pose_wxyz)
    if actions.ndim != 2 or actions.shape[1] != 23:
        raise ValueError(f"Expected N x 23 raw actions, got {actions.shape}.")
    if box_pose.shape != (actions.shape[0], 7) or bin_pose.shape != (
        actions.shape[0],
        7,
    ):
        raise ValueError("Box/bin pose sequences must both be N x 7 and match actions.")
    if control_dt_s <= 0:
        raise ValueError("control_dt_s must be positive.")

    num_steps = actions.shape[0]
    if num_steps < 8:
        raise ValueError(
            "At least eight samples are required for four-phase inference."
        )

    warnings: list[str] = []
    settle_steps = min(max(5, int(round(0.4 / control_dt_s))), max(5, num_steps // 5))
    baseline_count = min(max(5, int(round(0.5 / control_dt_s))), max(5, num_steps // 4))
    baseline_position = np.median(box_pose[:baseline_count, :3], axis=0)
    box_displacement = np.linalg.norm(box_pose[:, :3] - baseline_position, axis=1)

    seed_displacement_m = 1.0e-4
    confirm_displacement_m = 2.0e-3
    confirmation_steps = max(3, int(round(0.2 / control_dt_s)))
    grasp_start: int | None = None
    for index in range(settle_steps, num_steps - 3):
        stop = min(num_steps, index + confirmation_steps)
        if (
            box_displacement[index] >= seed_displacement_m
            and np.max(box_displacement[index:stop]) >= confirm_displacement_m
        ):
            grasp_start = index
            break
    if grasp_start is None:
        grasp_start = max(1, num_steps // 4)
        warnings.append(
            "No confirmed box-motion onset; grasp boundary fell back to 25% of the episode."
        )

    nav_norm = np.linalg.norm(actions[:, 16:19], axis=1)
    initial_box_z = float(baseline_position[2])
    lifted = box_pose[:, 2] - initial_box_z >= 0.03
    navigating = nav_norm >= 0.05
    carry_candidates = np.flatnonzero(
        navigating & lifted & (np.arange(num_steps) > grasp_start)
    )
    if carry_candidates.size:
        carry_start = int(carry_candidates[0])
    else:
        fallback_candidates = np.flatnonzero(
            navigating & (np.arange(num_steps) > grasp_start)
        )
        if fallback_candidates.size:
            carry_start = int(fallback_candidates[0])
            warnings.append("Carry onset used navigation without a verified 3 cm lift.")
        else:
            carry_start = max(grasp_start + 1, num_steps // 2)
            warnings.append(
                "No carry navigation onset; carry boundary fell back to 50% of the episode."
            )

    # A short zero-command gap can occur between P-controller subgoals, so require
    # one second of sustained stopping.  Also require meaningful progress toward
    # the bin relative to the distance at carry onset.
    stop_steps = max(3, int(round(1.0 / control_dt_s)))
    stopped = nav_norm < 0.01
    box_bin_distance = np.linalg.norm(box_pose[:, :3] - bin_pose[:, :3], axis=1)
    reference_distance = float(box_bin_distance[carry_start])
    progressed = box_bin_distance <= max(0.75 * reference_distance, 0.75)
    candidate_mask = stopped & progressed
    place_start = _first_sustained_true(candidate_mask, carry_start + 1, stop_steps)
    if place_start is None:
        place_start = _first_sustained_true(stopped, carry_start + 1, stop_steps)
        if place_start is not None:
            warnings.append(
                "Place onset used sustained stop without verified bin-distance progress."
            )
    if place_start is None:
        place_start = max(carry_start + 1, 3 * num_steps // 4)
        warnings.append(
            "No sustained navigation stop; place boundary fell back to 75% of the episode."
        )

    # Guarantee non-empty, monotonically ordered intervals even on malformed data.
    grasp_start = int(np.clip(grasp_start, 1, num_steps - 3))
    carry_start = int(np.clip(carry_start, grasp_start + 1, num_steps - 2))
    place_start = int(np.clip(place_start, carry_start + 1, num_steps - 1))
    boundaries = (0, grasp_start, carry_start, place_start, num_steps)
    phase_id = np.empty(num_steps, dtype=np.int8)
    for phase, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        phase_id[start:stop] = phase

    diagnostics = {
        "method": "signal_heuristic_v1",
        "signals": {
            "grasp": "first confirmed box displacement after settle window",
            "carry": "first |[vx,vy,wz]| >= threshold while box is >= 3 cm above initial z",
            "place": "first sustained navigation stop after progress toward bin",
        },
        "thresholds": {
            "settle_steps": settle_steps,
            "box_motion_seed_m": seed_displacement_m,
            "box_motion_confirmation_m": confirm_displacement_m,
            "box_motion_confirmation_steps": confirmation_steps,
            "carry_lift_m": 0.03,
            "navigation_norm": 0.05,
            "place_stop_navigation_norm": 0.01,
            "place_stop_steps": stop_steps,
        },
        "boundary_signal_values": {
            "grasp_box_displacement_m": float(box_displacement[grasp_start]),
            "carry_navigation_norm": float(nav_norm[carry_start]),
            "carry_box_lift_m": float(box_pose[carry_start, 2] - initial_box_z),
            "place_navigation_norm": float(nav_norm[place_start]),
            "place_box_bin_distance_m": float(box_bin_distance[place_start]),
        },
    }
    return PhaseInference(
        phase_id=phase_id,
        boundaries=boundaries,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )


def _all_dataset_paths(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            paths.append(name)

    group.visititems(visitor)
    return paths


def _dataset_schema(group: h5py.Group) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}

    def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            schema[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

    group.visititems(visitor)
    return schema


def _explicit_phase_paths(group: h5py.Group) -> list[str]:
    tokens = ("phase", "subtask", "datagen_info")
    return [
        path
        for path in _all_dataset_paths(group)
        if any(token in path.lower() for token in tokens)
    ]


def _parse_env_args(data_group: h5py.Group) -> dict[str, Any]:
    raw = data_group.attrs.get("env_args", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid data/env_args JSON: {raw!r}") from error
    return parsed


def _control_timing(data_group: h5py.Group) -> tuple[float, int, float, float]:
    env_args = _parse_env_args(data_group)
    sim_args = env_args.get("sim_args", {})
    physics_dt_s = float(sim_args["dt"])
    decimation = int(sim_args["decimation"])
    if physics_dt_s <= 0 or decimation <= 0:
        raise ValueError(f"Invalid timing: dt={physics_dt_s}, decimation={decimation}.")
    control_dt_s = physics_dt_s * decimation
    return physics_dt_s, decimation, control_dt_s, 1.0 / control_dt_s


def _require_shape(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{name} expected shape {shape}, got {array.shape}.")


def _pre_step_pose(initial_pose: np.ndarray, post_step_pose: np.ndarray) -> np.ndarray:
    initial_pose = np.asarray(initial_pose)
    post_step_pose = np.asarray(post_step_pose)
    if (
        initial_pose.shape not in ((7,), (1, 7))
        or post_step_pose.ndim != 2
        or post_step_pose.shape[1] != 7
    ):
        raise ValueError("Expected initial 7-D pose and N x 7 post-step poses.")
    result = np.empty_like(post_step_pose)
    result[0] = initial_pose.reshape(-1, 7)[0]
    result[1:] = post_step_pose[:-1]
    return result


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return (
        float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        if left.size
        else 0.0
    )


def _episode_arrays(episode: h5py.Group, data_group: h5py.Group) -> dict[str, Any]:
    actions = episode["actions"][:]
    processed = episode["processed_actions"][:]
    _require_shape(actions, (None, 23), "actions")
    _require_shape(processed, (actions.shape[0], 43), "processed_actions")
    num_steps = actions.shape[0]

    root_post = episode["states/articulation/robot/root_pose"][:]
    root_velocity_post = episode["states/articulation/robot/root_velocity"][:]
    joint_position_post = episode["states/articulation/robot/joint_position"][:]
    joint_velocity_post = episode["states/articulation/robot/joint_velocity"][:]
    box_post = episode["states/rigid_object/brown_box/root_pose"][:]
    box_velocity_post = episode["states/rigid_object/brown_box/root_velocity"][:]
    bin_post = episode["states/rigid_object/blue_sorting_bin/root_pose"][:]
    bin_velocity_post = episode["states/rigid_object/blue_sorting_bin/root_velocity"][:]
    for name, value, width in (
        ("root_pose", root_post, 7),
        ("root_velocity", root_velocity_post, 6),
        ("joint_position", joint_position_post, 43),
        ("joint_velocity", joint_velocity_post, 43),
        ("box_pose", box_post, 7),
        ("box_velocity", box_velocity_post, 6),
        ("bin_pose", bin_post, 7),
        ("bin_velocity", bin_velocity_post, 6),
    ):
        _require_shape(value, (num_steps, width), name)

    initial_root = episode["initial_state/articulation/robot/root_pose"][:]
    initial_joint_position = episode["initial_state/articulation/robot/joint_position"][
        :
    ]
    initial_joint_velocity = episode["initial_state/articulation/robot/joint_velocity"][
        :
    ]
    initial_box = episode["initial_state/rigid_object/brown_box/root_pose"][:]
    initial_bin = episode["initial_state/rigid_object/blue_sorting_bin/root_pose"][:]
    box_pre = _pre_step_pose(initial_box, box_post)
    bin_pre = _pre_step_pose(initial_bin, bin_post)

    # Robot position/quaternion observations share the same pre-step timestamp as
    # the observed wrist matrices.  Prefer them over shifting the post-step state.
    root_pre_shifted = _pre_step_pose(initial_root, root_post)
    if "obs/robot_pos" in episode and "obs/robot_quat" in episode:
        root_pre = np.concatenate(
            (episode["obs/robot_pos"][:], episode["obs/robot_quat"][:]), axis=1
        )
        _require_shape(root_pre, (num_steps, 7), "pre-step observed root pose")
        root_alignment_error = _max_abs_error(root_pre, root_pre_shifted)
    else:
        root_pre = root_pre_shifted
        root_alignment_error = 0.0

    joint_position_pre_shifted = np.empty_like(joint_position_post)
    joint_position_pre_shifted[0] = initial_joint_position.reshape(-1, 43)[0]
    joint_position_pre_shifted[1:] = joint_position_post[:-1]
    joint_velocity_pre_shifted = np.empty_like(joint_velocity_post)
    joint_velocity_pre_shifted[0] = initial_joint_velocity.reshape(-1, 43)[0]
    joint_velocity_pre_shifted[1:] = joint_velocity_post[:-1]
    joint_position_pre = (
        episode["obs/robot_joint_pos"][:]
        if "obs/robot_joint_pos" in episode
        else joint_position_pre_shifted
    )
    joint_velocity_pre = (
        episode["obs/robot_joint_vel"][:]
        if "obs/robot_joint_vel" in episode
        else joint_velocity_pre_shifted
    )
    _require_shape(joint_position_pre, (num_steps, 43), "pre-step robot joint position")
    _require_shape(joint_velocity_pre, (num_steps, 43), "pre-step robot joint velocity")

    left_pelvis_observed = episode["obs/left_wrist_pose_pelvis_frame"][:]
    right_pelvis_observed = episode["obs/right_wrist_pose_pelvis_frame"][:]
    _require_shape(left_pelvis_observed, (num_steps, 4, 4), "left pelvis wrist pose")
    _require_shape(right_pelvis_observed, (num_steps, 4, 4), "right pelvis wrist pose")

    root_matrix = pose7_wxyz_to_matrix(root_pre)
    box_matrix = pose7_wxyz_to_matrix(box_pre)
    bin_matrix = pose7_wxyz_to_matrix(bin_pre)
    left_world_observed = root_matrix @ left_pelvis_observed
    right_world_observed = root_matrix @ right_pelvis_observed

    left_pelvis_command = pose_matrix(
        actions[:, 2:5], quaternion_xyzw_to_matrix(actions[:, 5:9])
    )
    right_pelvis_command = pose_matrix(
        actions[:, 9:12], quaternion_xyzw_to_matrix(actions[:, 12:16])
    )
    left_world_command = root_matrix @ left_pelvis_command
    right_world_command = root_matrix @ right_pelvis_command

    box_inverse = rigid_inverse(box_matrix)
    bin_inverse = rigid_inverse(bin_matrix)
    box_left_observed = box_inverse @ left_world_observed
    box_right_observed = box_inverse @ right_world_observed
    box_left_command = box_inverse @ left_world_command
    box_right_command = box_inverse @ right_world_command
    bin_box = bin_inverse @ box_matrix

    physics_dt_s, decimation, control_dt_s, control_frequency_hz = _control_timing(
        data_group
    )
    phase = infer_task_phases(actions, box_post, bin_post, control_dt_s)
    phase_starts = np.asarray(phase.boundaries[:-1], dtype=np.int64)
    step_anchor_index = phase_starts[phase.phase_id]

    # Two related retargeting representations are emitted:
    # 1. pose relative to the box pose frozen at the current phase start;
    # 2. motion delta from the dynamic object-relative wrist pose at phase start.
    box_anchor_matrix = box_matrix[step_anchor_index]
    box_anchor_inverse = rigid_inverse(box_anchor_matrix)
    box_anchor_left_observed = box_anchor_inverse @ left_world_observed
    box_anchor_right_observed = box_anchor_inverse @ right_world_observed
    box_anchor_left_command = box_anchor_inverse @ left_world_command
    box_anchor_right_command = box_anchor_inverse @ right_world_command
    left_phase_delta_observed = (
        rigid_inverse(box_left_observed[step_anchor_index]) @ box_left_observed
    )
    right_phase_delta_observed = (
        rigid_inverse(box_right_observed[step_anchor_index]) @ box_right_observed
    )
    left_phase_delta_command = (
        rigid_inverse(box_left_command[step_anchor_index]) @ box_left_command
    )
    right_phase_delta_command = (
        rigid_inverse(box_right_command[step_anchor_index]) @ box_right_command
    )

    return {
        "num_steps": num_steps,
        "actions": actions,
        "processed": processed,
        "root_post": root_post,
        "root_velocity_post": root_velocity_post,
        "joint_position_post": joint_position_post,
        "joint_velocity_post": joint_velocity_post,
        "box_post": box_post,
        "box_velocity_post": box_velocity_post,
        "bin_post": bin_post,
        "bin_velocity_post": bin_velocity_post,
        "root_pre": root_pre,
        "joint_position_pre": joint_position_pre,
        "joint_velocity_pre": joint_velocity_pre,
        "box_pre": box_pre,
        "bin_pre": bin_pre,
        "left_pelvis_observed": left_pelvis_observed,
        "right_pelvis_observed": right_pelvis_observed,
        "left_pelvis_command": left_pelvis_command,
        "right_pelvis_command": right_pelvis_command,
        "left_world_observed": left_world_observed,
        "right_world_observed": right_world_observed,
        "left_world_command": left_world_command,
        "right_world_command": right_world_command,
        "box_left_observed": box_left_observed,
        "box_right_observed": box_right_observed,
        "box_left_command": box_left_command,
        "box_right_command": box_right_command,
        "bin_box": bin_box,
        "box_anchor_left_observed": box_anchor_left_observed,
        "box_anchor_right_observed": box_anchor_right_observed,
        "box_anchor_left_command": box_anchor_left_command,
        "box_anchor_right_command": box_anchor_right_command,
        "left_phase_delta_observed": left_phase_delta_observed,
        "right_phase_delta_observed": right_phase_delta_observed,
        "left_phase_delta_command": left_phase_delta_command,
        "right_phase_delta_command": right_phase_delta_command,
        "phase": phase,
        "phase_anchor_index": step_anchor_index,
        "physics_dt_s": physics_dt_s,
        "decimation": decimation,
        "control_dt_s": control_dt_s,
        "control_frequency_hz": control_frequency_hz,
        "root_pre_step_alignment_max_abs_error": root_alignment_error,
    }


def audit_episode(episode: h5py.Group, data_group: h5py.Group) -> dict[str, Any]:
    """Validate and summarize one source Arena episode without reading RGB."""

    arrays = _episode_arrays(episode, data_group)
    actions = arrays["actions"]
    processed = arrays["processed"]
    phase: PhaseInference = arrays["phase"]
    explicit_paths = _explicit_phase_paths(episode)
    source_schema = _dataset_schema(episode)

    subgroup_checks = {
        "left_eef_pos": _max_abs_error(
            actions[:, 2:5], episode["action/left_eef_pos"][:]
        ),
        "left_eef_quat_wxyz": _max_abs_error(
            actions[:, 5:9][:, [3, 0, 1, 2]], episode["action/left_eef_quat"][:]
        ),
        "right_eef_pos": _max_abs_error(
            actions[:, 9:12], episode["action/right_eef_pos"][:]
        ),
        "right_eef_quat_wxyz": _max_abs_error(
            actions[:, 12:16][:, [3, 0, 1, 2]], episode["action/right_eef_quat"][:]
        ),
        "navigate_cmd": _max_abs_error(
            actions[:, 16:19], episode["action/navigate_cmd"][:]
        ),
        "base_height_cmd": _max_abs_error(
            actions[:, 19:20], episode["action/base_height_cmd"][:]
        ),
        "torso_orientation_rpy_cmd": _max_abs_error(
            actions[:, 20:23], episode["action/torso_orientation_rpy_cmd"][:]
        ),
    }

    warnings = list(phase.warnings)
    if not explicit_paths:
        warnings.append(
            "No explicit approach/grasp/carry/place or subtask annotation exists; phase_id is inferred."
        )
    hand_state_unique = {
        "left": np.unique(actions[:, 0]).tolist(),
        "right": np.unique(actions[:, 1]).tolist(),
    }
    if len(hand_state_unique["left"]) == 1 and len(hand_state_unique["right"]) == 1:
        warnings.append(
            "Both high-level hand-state channels are constant; this file does not demonstrate an open/close transition."
        )
    dex3_target = processed[:, DEX3_INDICES]
    if float(np.max(np.ptp(dex3_target, axis=0))) < 1.0e-7:
        warnings.append(
            "All extracted Dex3 processed joint targets are constant (zero in the inspected small dataset)."
        )
    if not any("contact" in path.lower() for path in source_schema):
        warnings.append(
            "No explicit hand/object contact signal is stored in this source episode."
        )
    if not any(
        "box_size" in path.lower() or path.lower().endswith("/size")
        for path in source_schema
    ):
        warnings.append("No box-size field is stored in this source episode.")

    raw_quat_norms = np.concatenate(
        (
            np.linalg.norm(actions[:, 5:9], axis=1),
            np.linalg.norm(actions[:, 12:16], axis=1),
        )
    )
    pose_quat_norms = np.concatenate(
        (
            np.linalg.norm(arrays["root_post"][:, 3:7], axis=1),
            np.linalg.norm(arrays["box_post"][:, 3:7], axis=1),
            np.linalg.norm(arrays["bin_post"][:, 3:7], axis=1),
        )
    )

    return _as_python(
        {
            "num_samples": arrays["num_steps"],
            "duration_s": arrays["num_steps"] * arrays["control_dt_s"],
            "success_attribute": bool(episode.attrs.get("success", False)),
            "source_dataset_schema": source_schema,
            "timing": {
                "physics_dt_s": arrays["physics_dt_s"],
                "decimation": arrays["decimation"],
                "control_dt_s": arrays["control_dt_s"],
                "control_frequency_hz": arrays["control_frequency_hz"],
            },
            "actions": {
                "raw_shape": actions.shape,
                "processed_shape": processed.shape,
                "layout": [
                    {"name": name, "slice": [start, end], "semantic": semantic}
                    for name, start, end, semantic in ARENA_G1_ACTION_LAYOUT
                ],
                "decomposed_subgroup_max_abs_error": subgroup_checks,
                "hand_state_unique_values": hand_state_unique,
                "raw_wrist_quaternion_convention": "xyzw",
                "decomposed_action_subgroup_wrist_quaternion_convention": "wxyz",
                "raw_wrist_quaternion_norm_range": [
                    float(raw_quat_norms.min()),
                    float(raw_quat_norms.max()),
                ],
            },
            "states": {
                "root_and_object_pose_layout": "xyz + wxyz",
                "root_and_object_quaternion_norm_range": [
                    float(pose_quat_norms.min()),
                    float(pose_quat_norms.max()),
                ],
                "recorder_timing": "actions/obs are pre-step; states/processed_actions are post-step",
                "aligned_root_pose_max_abs_error_vs_shifted_state": arrays[
                    "root_pre_step_alignment_max_abs_error"
                ],
            },
            "phase_annotations": {
                "explicit_annotations_present": bool(explicit_paths),
                "explicit_annotation_paths": explicit_paths,
                "source": "explicit" if explicit_paths else "signal_heuristic_v1",
                "phase_names": PHASE_NAMES,
                "boundaries": phase.boundaries,
                "intervals": phase.intervals,
                "diagnostics": phase.diagnostics,
            },
            "retargeting": {
                "dynamic_transforms": [
                    "T_box^-1 T_left_wrist_observed",
                    "T_box^-1 T_right_wrist_observed",
                    "T_box^-1 T_left_wrist_command",
                    "T_box^-1 T_right_wrist_command",
                    "T_bin^-1 T_box",
                ],
                "phase_anchor_transforms_and_deltas_included": True,
                "world_wrist_calculation": "T_world_root(pre-step) @ T_pelvis_wrist",
                "object_pose_alignment": "initial state for step 0, previous post-step state thereafter",
            },
            "dex3": {
                "left_processed_target_indices": LEFT_DEX3_INDICES,
                "right_processed_target_indices": RIGHT_DEX3_INDICES,
                "left_joint_names": [
                    ARENA_G1_LAB_JOINT_NAMES[index] for index in LEFT_DEX3_INDICES
                ],
                "right_joint_names": [
                    ARENA_G1_LAB_JOINT_NAMES[index] for index in RIGHT_DEX3_INDICES
                ],
            },
            "camera": {
                "source_rgb_present": "camera_obs/robot_head_cam_rgb" in episode,
                "rgb_copied": False,
            },
            "warnings": warnings,
        }
    )


def load_episode_trajectory(
    source_path: str | Path,
    episode_name: str = "demo_0",
) -> ArenaTeacherTrajectory:
    """Load one Arena episode for runtime teacher replay without camera data.

    This public adapter is intentionally simulator-independent.  A rollout may
    index it by ``phase_id``, consume world or object-relative Pink wrist targets,
    and map ``processed_43d_joint_target`` using ``source_lab_joint_names``.
    """

    resolved_path = Path(source_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    with h5py.File(resolved_path, "r") as source:
        if "data" not in source or episode_name not in source["data"]:
            available = (
                sorted(source.get("data", {}).keys()) if "data" in source else []
            )
            raise KeyError(f"Unknown episode {episode_name!r}; available: {available}")
        arrays = _episode_arrays(source[f"data/{episode_name}"], source["data"])
    phase: PhaseInference = arrays["phase"]
    return ArenaTeacherTrajectory(
        source_path=resolved_path,
        episode_name=episode_name,
        control_dt_s=float(arrays["control_dt_s"]),
        control_frequency_hz=float(arrays["control_frequency_hz"]),
        high_level_action_23d=arrays["actions"],
        processed_43d_joint_target=arrays["processed"],
        phase_id=phase.phase_id,
        phase_boundaries=phase.boundaries,
        robot_joint_position_pre_step=arrays["joint_position_pre"],
        robot_joint_velocity_pre_step=arrays["joint_velocity_pre"],
        root_pose_wxyz_pre_step=arrays["root_pre"],
        box_pose_wxyz_pre_step=arrays["box_pre"],
        bin_pose_wxyz_pre_step=arrays["bin_pre"],
        left_world_wrist_observed=arrays["left_world_observed"],
        right_world_wrist_observed=arrays["right_world_observed"],
        left_world_wrist_command=arrays["left_world_command"],
        right_world_wrist_command=arrays["right_world_command"],
        left_box_relative_wrist_observed=arrays["box_left_observed"],
        right_box_relative_wrist_observed=arrays["box_right_observed"],
        left_box_relative_wrist_command=arrays["box_left_command"],
        right_box_relative_wrist_command=arrays["box_right_command"],
        bin_relative_box=arrays["bin_box"],
        left_dex3_joint_target=arrays["processed"][:, LEFT_DEX3_INDICES],
        right_dex3_joint_target=arrays["processed"][:, RIGHT_DEX3_INDICES],
    )


def _create_numeric(group: h5py.Group, name: str, value: np.ndarray) -> h5py.Dataset:
    value = np.asarray(value)
    options = (
        {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        if value.size
        else {}
    )
    return group.create_dataset(name, data=value, **options)


def _create_strings(
    group: h5py.Group, name: str, values: Iterable[str]
) -> h5py.Dataset:
    dtype = h5py.string_dtype(encoding="utf-8")
    return group.create_dataset(
        name, data=np.asarray(tuple(values), dtype=object), dtype=dtype
    )


def _write_episode(
    output: h5py.Group,
    episode: h5py.Group,
    arrays: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    output.attrs["num_samples"] = arrays["num_steps"]
    output.attrs["success"] = bool(episode.attrs.get("success", False))
    output.attrs["control_dt_s"] = arrays["control_dt_s"]
    output.attrs["control_frequency_hz"] = arrays["control_frequency_hz"]
    output.attrs["phase_annotation_source"] = audit["phase_annotations"]["source"]

    metadata = output.create_group("metadata")
    _create_strings(metadata, "source_lab_joint_names", ARENA_G1_LAB_JOINT_NAMES)
    _create_strings(metadata, "phase_names", PHASE_NAMES)
    _create_strings(
        metadata,
        "raw_action_field_names",
        (field[0] for field in ARENA_G1_ACTION_LAYOUT),
    )
    _create_numeric(
        metadata,
        "raw_action_field_slices",
        np.asarray([(field[1], field[2]) for field in ARENA_G1_ACTION_LAYOUT]),
    )
    metadata.attrs["source_joint_order_provenance"] = (
        "IsaacLab-Arena 0.3.0 isaaclab_arena_g1/g1_env/config/lab_g1_joints_order_43dof.yaml"
    )
    metadata.attrs["root_object_pose_quaternion"] = "wxyz"
    metadata.attrs["raw_wrist_action_quaternion"] = "xyzw"
    metadata.attrs["rgb_copied"] = False

    teacher = output.create_group("teacher")
    _create_numeric(teacher, "high_level_action_23d", arrays["actions"])
    _create_numeric(teacher, "processed_43d_joint_target", arrays["processed"])

    commands = output.create_group("commands")
    _create_numeric(commands, "base_velocity_vx_vy_wz", arrays["actions"][:, 16:19])
    _create_numeric(commands, "hip_height", arrays["actions"][:, 19:20])
    _create_numeric(commands, "torso_orientation_rpy", arrays["actions"][:, 20:23])

    wrist = output.create_group("wrist")
    _create_numeric(wrist, "left_position_pelvis_command", arrays["actions"][:, 2:5])
    left_quat = _create_numeric(
        wrist, "left_quaternion_xyzw_command", arrays["actions"][:, 5:9]
    )
    left_quat.attrs["quaternion_convention"] = "xyzw"
    _create_numeric(wrist, "right_position_pelvis_command", arrays["actions"][:, 9:12])
    right_quat = _create_numeric(
        wrist, "right_quaternion_xyzw_command", arrays["actions"][:, 12:16]
    )
    right_quat.attrs["quaternion_convention"] = "xyzw"
    _create_numeric(wrist, "T_pelvis_left_observed", arrays["left_pelvis_observed"])
    _create_numeric(wrist, "T_pelvis_right_observed", arrays["right_pelvis_observed"])
    _create_numeric(wrist, "T_pelvis_left_command", arrays["left_pelvis_command"])
    _create_numeric(wrist, "T_pelvis_right_command", arrays["right_pelvis_command"])

    dex3 = output.create_group("dex3")
    _create_numeric(
        dex3, "left_processed_joint_target", arrays["processed"][:, LEFT_DEX3_INDICES]
    )
    _create_numeric(
        dex3, "right_processed_joint_target", arrays["processed"][:, RIGHT_DEX3_INDICES]
    )
    _create_numeric(
        dex3, "both_processed_joint_target", arrays["processed"][:, DEX3_INDICES]
    )
    _create_strings(
        dex3,
        "left_joint_names",
        (ARENA_G1_LAB_JOINT_NAMES[index] for index in LEFT_DEX3_INDICES),
    )
    _create_strings(
        dex3,
        "right_joint_names",
        (ARENA_G1_LAB_JOINT_NAMES[index] for index in RIGHT_DEX3_INDICES),
    )
    _create_numeric(dex3, "left_high_level_state", arrays["actions"][:, 0:1])
    _create_numeric(dex3, "right_high_level_state", arrays["actions"][:, 1:2])

    state = output.create_group("state")
    post = state.create_group("post_step")
    for name, value in (
        ("root_pose_xyz_wxyz", arrays["root_post"]),
        ("root_velocity", arrays["root_velocity_post"]),
        ("robot_joint_position", arrays["joint_position_post"]),
        ("robot_joint_velocity", arrays["joint_velocity_post"]),
        ("box_pose_xyz_wxyz", arrays["box_post"]),
        ("box_velocity", arrays["box_velocity_post"]),
        ("bin_pose_xyz_wxyz", arrays["bin_post"]),
        ("bin_velocity", arrays["bin_velocity_post"]),
    ):
        dataset = _create_numeric(post, name, value)
        if "pose" in name:
            dataset.attrs["quaternion_convention"] = "wxyz"
    aligned = state.create_group("aligned_pre_step")
    for name, value in (
        ("root_pose_xyz_wxyz", arrays["root_pre"]),
        ("robot_joint_position", arrays["joint_position_pre"]),
        ("robot_joint_velocity", arrays["joint_velocity_pre"]),
        ("box_pose_xyz_wxyz", arrays["box_pre"]),
        ("bin_pose_xyz_wxyz", arrays["bin_pre"]),
    ):
        dataset = _create_numeric(aligned, name, value)
        dataset.attrs["quaternion_convention"] = "wxyz"

    transforms = output.create_group("transforms")
    for name, value in (
        ("T_world_left_wrist_observed", arrays["left_world_observed"]),
        ("T_world_right_wrist_observed", arrays["right_world_observed"]),
        ("T_world_left_wrist_command", arrays["left_world_command"]),
        ("T_world_right_wrist_command", arrays["right_world_command"]),
        ("T_box_inv_T_left_wrist_observed", arrays["box_left_observed"]),
        ("T_box_inv_T_right_wrist_observed", arrays["box_right_observed"]),
        ("T_box_inv_T_left_wrist_command", arrays["box_left_command"]),
        ("T_box_inv_T_right_wrist_command", arrays["box_right_command"]),
        ("T_bin_inv_T_box", arrays["bin_box"]),
        (
            "T_box_phase_anchor_inv_T_left_wrist_observed",
            arrays["box_anchor_left_observed"],
        ),
        (
            "T_box_phase_anchor_inv_T_right_wrist_observed",
            arrays["box_anchor_right_observed"],
        ),
        (
            "T_box_phase_anchor_inv_T_left_wrist_command",
            arrays["box_anchor_left_command"],
        ),
        (
            "T_box_phase_anchor_inv_T_right_wrist_command",
            arrays["box_anchor_right_command"],
        ),
        (
            "T_left_wrist_object_relative_phase_delta_observed",
            arrays["left_phase_delta_observed"],
        ),
        (
            "T_right_wrist_object_relative_phase_delta_observed",
            arrays["right_phase_delta_observed"],
        ),
        (
            "T_left_wrist_object_relative_phase_delta_command",
            arrays["left_phase_delta_command"],
        ),
        (
            "T_right_wrist_object_relative_phase_delta_command",
            arrays["right_phase_delta_command"],
        ),
    ):
        _create_numeric(transforms, name, np.asarray(value, dtype=np.float32))

    phase: PhaseInference = arrays["phase"]
    phase_group = output.create_group("phase")
    phase_ids = _create_numeric(phase_group, "id", phase.phase_id)
    phase_ids.attrs["labels"] = json.dumps(
        {index: name for index, name in enumerate(PHASE_NAMES)}
    )
    _create_numeric(
        phase_group, "boundaries", np.asarray(phase.boundaries, dtype=np.int64)
    )
    _create_numeric(phase_group, "anchor_index", arrays["phase_anchor_index"])
    phase_group.attrs["explicit_annotations_present"] = bool(
        audit["phase_annotations"]["explicit_annotations_present"]
    )
    phase_group.attrs["source"] = audit["phase_annotations"]["source"]
    phase_group.attrs["diagnostics_json"] = json.dumps(
        audit["phase_annotations"]["diagnostics"], sort_keys=True
    )


def extract_dataset(
    source_path: str | Path,
    output_path: str | Path,
    audit_json_path: str | Path,
    episode_names: Iterable[str] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Audit and convert Arena demonstrations into a camera-free retargeting HDF5."""

    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    audit_json_path = Path(audit_json_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for target in (output_path, audit_json_path):
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {target}; pass overwrite=True explicitly."
            )
        target.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(source_path),
        "derived_hdf5_path": str(output_path),
        "rgb_copied": False,
        "episodes": {},
    }
    with h5py.File(source_path, "r") as source:
        if "data" not in source:
            raise ValueError("Source file has no /data group.")
        source_data = source["data"]
        available = sorted(
            name for name, item in source_data.items() if isinstance(item, h5py.Group)
        )
        selected = list(episode_names) if episode_names is not None else available
        missing = sorted(set(selected) - set(available))
        if missing:
            raise KeyError(f"Unknown episode(s): {missing}; available: {available}")
        if not selected:
            raise ValueError("No episodes selected.")

        env_args = _parse_env_args(source_data)
        report["source_env_args"] = env_args
        with h5py.File(output_path, "w") as derived:
            derived.attrs["schema_version"] = SCHEMA_VERSION
            derived.attrs["source_path"] = str(source_path)
            derived.attrs["source_env_args"] = json.dumps(env_args, sort_keys=True)
            derived.attrs["rgb_copied"] = False
            output_data = derived.create_group("data")
            for episode_name in selected:
                source_episode = source_data[episode_name]
                arrays = _episode_arrays(source_episode, source_data)
                audit = audit_episode(source_episode, source_data)
                report["episodes"][episode_name] = audit
                _write_episode(
                    output_data.create_group(episode_name),
                    source_episode,
                    arrays,
                    audit,
                )
            output_data.attrs["total"] = sum(
                item["num_samples"] for item in report["episodes"].values()
            )

    with audit_json_path.open("w", encoding="utf-8") as stream:
        json.dump(_as_python(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report
