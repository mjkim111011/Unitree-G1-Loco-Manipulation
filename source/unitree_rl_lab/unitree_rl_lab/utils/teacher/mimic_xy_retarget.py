"""Isaac Lab Mimic retargeting helpers for the first Box randomization gate.

The first data-generation curriculum intentionally varies only the initial
Box ``x/y`` position.  Object size, mass, material, table height, object
height, and yaw remain fixed.  This module keeps that contract explicit and
uses the same object-relative transform as Isaac Lab Mimic for the source Box
and both wrist trajectories.

The Isaac Lab Mimic callable is injected instead of imported at module import
time.  Importing ``isaaclab_mimic.datagen.data_generator`` requires an active
Isaac Sim application, while the validation helpers in this module are also
used by lightweight unit tests and offline dataset tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


MIMIC_TRANSFORM_QUALNAME = (
    "isaaclab_mimic.datagen.data_generator."
    "get_delta_pose_with_scheme(TRANSLATE) + "
    "transform_source_data_segment_using_delta_object_pose"
)


@dataclass(frozen=True, slots=True)
class FixedBoxSceneParameters:
    """Scene constants held fixed while the Box initial ``x/y`` is sampled."""

    box_size_m: tuple[float, float, float]
    box_mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    table_surface_height_m: float
    box_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        size = np.asarray(self.box_size_m, dtype=np.float64)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            raise ValueError("box_size_m must contain three finite positive values")
        scalar_values = (
            self.box_mass_kg,
            self.static_friction,
            self.dynamic_friction,
            self.restitution,
            self.table_surface_height_m,
            self.box_yaw_rad,
        )
        if not np.isfinite(scalar_values).all():
            raise ValueError("fixed scene parameters must be finite")
        if self.box_mass_kg <= 0.0:
            raise ValueError("box_mass_kg must be positive")
        if min(self.static_friction, self.dynamic_friction, self.restitution) < 0.0:
            raise ValueError("friction and restitution must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "box_size_m": list(self.box_size_m),
            "box_mass_kg": self.box_mass_kg,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "table_surface_height_m": self.table_surface_height_m,
            "box_yaw_rad": self.box_yaw_rad,
        }


@dataclass(frozen=True, slots=True)
class XYPositionRandomization:
    """Small initial Box translation range with all non-``x/y`` pose fixed."""

    x_range_m: tuple[float, float] = (0.618, 0.622)
    y_range_m: tuple[float, float] = (-0.002, 0.002)

    def __post_init__(self) -> None:
        for name, bounds in (
            ("x_range_m", self.x_range_m),
            ("y_range_m", self.y_range_m),
        ):
            values = np.asarray(bounds, dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain two finite bounds")
            if values[0] > values[1]:
                raise ValueError(f"{name} lower bound exceeds upper bound")

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        """Return one deterministic-with-seed uniform ``(x, y)`` sample."""

        return (
            float(rng.uniform(*self.x_range_m)),
            float(rng.uniform(*self.y_range_m)),
        )

    def as_event_range(self) -> dict[str, tuple[float, float]]:
        return {"x": self.x_range_m, "y": self.y_range_m}


@dataclass(frozen=True, slots=True)
class MimicRetargetedTrajectories:
    """Object and bimanual wrist trajectories under one rigid Mimic transform."""

    object_world: np.ndarray
    left_wrist_world: np.ndarray
    right_wrist_world: np.ndarray


MimicDeltaTransform = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MimicDeltaPose = Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]


def _pose_sequence(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (T, 4, 4), got {array.shape}")
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    if not np.allclose(array[:, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-6):
        raise ValueError(f"{name} contains invalid homogeneous transforms")
    return array


def retarget_object_and_bimanual_wrists(
    *,
    source_object_world: np.ndarray,
    source_left_wrist_world: np.ndarray,
    source_right_wrist_world: np.ndarray,
    target_initial_object_world: np.ndarray,
    mimic_transform: MimicDeltaTransform,
    mimic_delta_pose: MimicDeltaPose,
    translate_coordination_scheme: object,
    device: str | torch.device = "cpu",
) -> MimicRetargetedTrajectories:
    """Apply Isaac Lab Mimic's object-relative transform to all three paths.

    ``mimic_transform`` must be Isaac Lab Mimic's
    :func:`transform_source_data_segment_using_delta_object_pose` in production,
    and ``mimic_delta_pose`` must be :func:`get_delta_pose_with_scheme` using
    Mimic's ``TRANSLATE`` scheme with zero noise.  Translation-only is
    intentional: the Arena and current Box assets use different semantic yaw
    conventions, so a generic SE(3) transform would incorrectly flip the
    bimanual paths.  The source Box path is still transformed alongside both
    wrists, rather than translating only the end-effectors independently.
    """

    source_object = _pose_sequence(source_object_world, "source_object_world")
    source_left = _pose_sequence(source_left_wrist_world, "source_left_wrist_world")
    source_right = _pose_sequence(source_right_wrist_world, "source_right_wrist_world")
    if not (len(source_object) == len(source_left) == len(source_right)):
        raise ValueError(
            "object and bimanual wrist trajectories must have equal lengths"
        )
    target_initial = np.asarray(target_initial_object_world, dtype=np.float64)
    if target_initial.shape != (4, 4) or not np.isfinite(target_initial).all():
        raise ValueError("target_initial_object_world must be a finite 4x4 transform")

    # Mimic's get_delta_pose_with_scheme creates its identity transform with
    # the environment's default float32 dtype, matching normal simulator data.
    src_object_t = torch.as_tensor(source_object, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target_initial, dtype=torch.float32, device=device)
    source_initial_t = src_object_t[0]
    delta_pose = mimic_delta_pose(
        source_initial_t,
        target_t,
        {
            "coordination_scheme": translate_coordination_scheme,
            "coordination_scheme_pos_noise_scale": 0.0,
            "coordination_scheme_rot_noise_scale": 0.0,
        },
    )
    if not isinstance(delta_pose, torch.Tensor) or delta_pose.shape != (4, 4):
        raise ValueError("Isaac Lab Mimic returned an invalid delta object pose")
    if not torch.allclose(
        delta_pose[:3, :3], torch.eye(3, dtype=delta_pose.dtype, device=device)
    ):
        raise RuntimeError(
            "x/y curriculum requires Mimic's translation-only coordination scheme"
        )

    def transform(sequence: np.ndarray) -> np.ndarray:
        sequence_t = torch.as_tensor(sequence, dtype=torch.float32, device=device)
        transformed = mimic_transform(sequence_t, delta_pose)
        if (
            not isinstance(transformed, torch.Tensor)
            or transformed.shape != sequence_t.shape
        ):
            raise ValueError("Isaac Lab Mimic transform returned an invalid trajectory")
        return transformed.detach().cpu().numpy().astype(np.float64, copy=False)

    result = MimicRetargetedTrajectories(
        object_world=transform(source_object),
        left_wrist_world=transform(source_left),
        right_wrist_world=transform(source_right),
    )

    # A single rigid change of object frame must preserve wrist/object relative
    # transforms at every step.  This catches accidental one-arm or object-only
    # translations before they become training data.
    for source_wrist, target_wrist, side in (
        (source_left, result.left_wrist_world, "left"),
        (source_right, result.right_wrist_world, "right"),
    ):
        source_relative = np.linalg.inv(source_object) @ source_wrist
        target_relative = np.linalg.inv(result.object_world) @ target_wrist
        if not np.allclose(source_relative, target_relative, atol=2.0e-6, rtol=1.0e-6):
            raise RuntimeError(
                f"Mimic retargeting did not preserve {side} wrist/Box poses"
            )
    if not np.allclose(
        result.object_world[0, :3, 3], target_initial[:3, 3], atol=2.0e-6, rtol=1.0e-6
    ):
        raise RuntimeError(
            "Mimic-retargeted initial Box position does not match the sampled position"
        )
    return result


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive true run in a one-dimensional mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {values.shape}")
    if not values.any():
        return 0
    padded = np.pad(values.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2]))


def episode_passes_bimanual_lift_hold(
    *,
    lift: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    required_hold_steps: int,
) -> bool:
    """Strict success filter used before a randomized demo is published."""

    if required_hold_steps <= 0:
        raise ValueError("required_hold_steps must be positive")
    lift_mask = np.asarray(lift, dtype=bool)
    left_mask = np.asarray(left_contact, dtype=bool)
    right_mask = np.asarray(right_contact, dtype=bool)
    if (
        not (lift_mask.shape == left_mask.shape == right_mask.shape)
        or lift_mask.ndim != 1
    ):
        raise ValueError(
            "lift and bilateral contact masks must be equal one-dimensional arrays"
        )
    return longest_true_run(lift_mask & left_mask & right_mask) >= required_hold_steps
