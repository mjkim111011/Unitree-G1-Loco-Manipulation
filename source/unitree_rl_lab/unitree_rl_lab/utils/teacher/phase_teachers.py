"""Privileged high-level carry and place teachers built from realized states.

These teachers operate on wrist/object transforms captured from physics.  The
carry teacher maintains the realized bilateral grasp while the camera-derived
navigation command drives the base.  The place teacher starts only from a real
carry-at-staging snapshot and moves that same grasp over the Bin before opening
both hands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix_wxyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite xyz+wxyz vector")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_pose_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quat_xyzw[3], quat_xyzw[:3]].astype(np.float32)


def _wrist_action_pose_xyzw(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32)
    return matrix[:3, 3].astype(np.float32), quat


@dataclass(frozen=True, slots=True)
class RealizedBimanualGrasp:
    """Object-relative wrists measured after strict bilateral lift/hold."""

    box_orientation_w: np.ndarray
    box_to_left_wrist: np.ndarray
    box_to_right_wrist: np.ndarray

    @classmethod
    def capture(
        cls,
        *,
        box_pose_wxyz: np.ndarray,
        left_wrist_pose_wxyz: np.ndarray,
        right_wrist_pose_wxyz: np.ndarray,
    ) -> "RealizedBimanualGrasp":
        box = pose_matrix_wxyz(box_pose_wxyz)
        left = pose_matrix_wxyz(left_wrist_pose_wxyz)
        right = pose_matrix_wxyz(right_wrist_pose_wxyz)
        return cls(
            box_orientation_w=box[:3, :3].copy(),
            box_to_left_wrist=np.linalg.inv(box) @ left,
            box_to_right_wrist=np.linalg.inv(box) @ right,
        ).validate()

    def validate(self) -> "RealizedBimanualGrasp":
        orientation = np.asarray(self.box_orientation_w)
        if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
            raise ValueError("box_orientation_w must be a finite 3x3 rotation")
        if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1.0e-5):
            raise ValueError("box_orientation_w is not orthonormal")
        for name in ("box_to_left_wrist", "box_to_right_wrist"):
            value = np.asarray(getattr(self, name))
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 4x4 transform")
        return self

    def desired_world_wrists(
        self, desired_box_position_w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(desired_box_position_w, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("desired_box_position_w must be a finite 3-vector")
        desired_box = np.eye(4, dtype=np.float64)
        desired_box[:3, :3] = self.box_orientation_w
        desired_box[:3, 3] = position
        return (
            desired_box @ self.box_to_left_wrist,
            desired_box @ self.box_to_right_wrist,
        )


class CarryTeacher:
    """Maintain a realized bimanual hold while following camera navigation."""

    def __init__(
        self,
        grasp: RealizedBimanualGrasp,
        *,
        base_height: float = 0.75,
        minimum_box_height_w: float | None = None,
    ) -> None:
        self.grasp = grasp.validate()
        self.base_height = float(base_height)
        if minimum_box_height_w is not None and not np.isfinite(minimum_box_height_w):
            raise ValueError("minimum_box_height_w must be finite when provided")
        self.minimum_box_height_w = (
            None if minimum_box_height_w is None else float(minimum_box_height_w)
        )

    def action(
        self,
        *,
        current_box_pose_wxyz: np.ndarray,
        current_pelvis_pose_wxyz: np.ndarray,
        camera_navigation_velocity: np.ndarray,
    ) -> np.ndarray:
        box_pose = np.asarray(current_box_pose_wxyz, dtype=np.float64)
        if box_pose.shape != (7,) or not np.isfinite(box_pose).all():
            raise ValueError("current_box_pose_wxyz must be a finite xyz+wxyz pose")
        box_position = box_pose[:3].copy()
        if self.minimum_box_height_w is not None:
            # Follow horizontal motion so the realized cage moves naturally
            # with the walking robot, but never chase a slipping Box downward.
            # This gives the teacher an actual recovery target during carry.
            box_position[2] = max(box_position[2], self.minimum_box_height_w)
        left_w, right_w = self.grasp.desired_world_wrists(box_position)
        pelvis_inv = np.linalg.inv(pose_matrix_wxyz(current_pelvis_pose_wxyz))
        left_b = pelvis_inv @ left_w
        right_b = pelvis_inv @ right_w
        nav = np.asarray(camera_navigation_velocity, dtype=np.float32)
        if nav.shape != (3,) or not np.isfinite(nav).all():
            raise ValueError(
                "camera_navigation_velocity must be a finite vx/vy/wz vector"
            )
        result = np.zeros(23, dtype=np.float32)
        result[:2] = 1.0
        result[2:5], result[5:9] = _wrist_action_pose_xyzw(left_b)
        result[9:12], result[12:16] = _wrist_action_pose_xyzw(right_b)
        result[16:19] = nav
        result[19] = self.base_height
        return result


@dataclass(frozen=True, slots=True)
class PlaceTrajectoryCfg:
    raise_steps: int = 25
    translate_steps: int = 100
    lower_steps: int = 60
    release_steps: int = 50
    clearance_m: float = 0.08
    # The current Bin walls end at z=0.54 m and the fixed curriculum Box is
    # 0.20 m tall.  Releasing its centre at 0.70 m leaves 6 cm of clearance
    # under the Box instead of asking both wrists to enter the Bin.  The old
    # 0.46 m target drove the still-closed grasp through the wall height and
    # consistently broke bilateral contact before the release stage.
    release_height_m: float = 0.70

    @property
    def total_steps(self) -> int:
        return (
            self.raise_steps
            + self.translate_steps
            + self.lower_steps
            + self.release_steps
        )

    def validate(self) -> "PlaceTrajectoryCfg":
        if (
            min(
                self.raise_steps,
                self.translate_steps,
                self.lower_steps,
                self.release_steps,
            )
            <= 0
        ):
            raise ValueError(
                "every place trajectory stage must contain at least one step"
            )
        if self.clearance_m < 0.0 or self.release_height_m <= 0.0:
            raise ValueError("place clearance and release height must be valid")
        return self


class PlaceTeacher:
    """Translate the captured physical grasp over the Bin, lower, and release."""

    def __init__(
        self,
        grasp: RealizedBimanualGrasp,
        *,
        start_box_position_w: np.ndarray,
        bin_center_w: np.ndarray,
        cfg: PlaceTrajectoryCfg = PlaceTrajectoryCfg(),
        base_height: float = 0.75,
    ) -> None:
        self.grasp = grasp.validate()
        self.start = np.asarray(start_box_position_w, dtype=np.float64)
        self.bin_center = np.asarray(bin_center_w, dtype=np.float64)
        if self.start.shape != (3,) or self.bin_center.shape != (3,):
            raise ValueError("start Box and Bin center must be 3-vectors")
        self.cfg = cfg.validate()
        self.base_height = float(base_height)
        self.raised = self.start + np.array([0.0, 0.0, cfg.clearance_m])
        self.over_bin = np.array(
            [self.bin_center[0], self.bin_center[1], self.raised[2]]
        )
        self.release = np.array(
            [self.bin_center[0], self.bin_center[1], cfg.release_height_m]
        )

    @staticmethod
    def _lerp(
        first: np.ndarray, second: np.ndarray, numerator: int, denominator: int
    ) -> np.ndarray:
        fraction = np.clip(numerator / max(1, denominator), 0.0, 1.0)
        return first + fraction * (second - first)

    def desired_box_position(self, step: int) -> tuple[np.ndarray, bool]:
        if step < 0:
            raise ValueError("step must be non-negative")
        a = self.cfg.raise_steps
        b = a + self.cfg.translate_steps
        c = b + self.cfg.lower_steps
        if step < a:
            return self._lerp(
                self.start, self.raised, step + 1, self.cfg.raise_steps
            ), False
        if step < b:
            return self._lerp(
                self.raised, self.over_bin, step - a + 1, self.cfg.translate_steps
            ), False
        if step < c:
            return self._lerp(
                self.over_bin, self.release, step - b + 1, self.cfg.lower_steps
            ), False
        return self.release.copy(), True

    def action(
        self,
        *,
        step: int,
        current_pelvis_pose_wxyz: np.ndarray,
        camera_navigation_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        box_position, release = self.desired_box_position(step)
        return self.action_for_box_position(
            box_position=box_position,
            release=release,
            current_pelvis_pose_wxyz=current_pelvis_pose_wxyz,
            camera_navigation_velocity=camera_navigation_velocity,
        )

    def action_for_box_position(
        self,
        *,
        box_position: np.ndarray,
        release: bool,
        current_pelvis_pose_wxyz: np.ndarray,
        left_recovery_compression_m: float = 0.0,
        right_recovery_compression_m: float = 0.0,
        camera_navigation_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a PLACE command for a feedback-selected Box target.

        The normal trajectory uses :meth:`action`.  A physical collector may
        pause the nominal clock and follow the measured Box when contact is
        marginal.  Positive recovery compression moves only the affected
        wrist toward the Box centre and is intentionally unavailable to the
        Student policy at inference time.
        """
        box_position = np.asarray(box_position, dtype=np.float64)
        if box_position.shape != (3,) or not np.isfinite(box_position).all():
            raise ValueError("box_position must be a finite 3-vector")
        if min(left_recovery_compression_m, right_recovery_compression_m) < 0.0:
            raise ValueError("place recovery compression must be non-negative")
        left_w, right_w = self.grasp.desired_world_wrists(box_position)
        for wrist, compression in (
            (left_w, float(left_recovery_compression_m)),
            (right_w, float(right_recovery_compression_m)),
        ):
            if compression <= 0.0:
                continue
            direction = box_position - wrist[:3, 3]
            distance = float(np.linalg.norm(direction))
            if distance > 1.0e-9:
                wrist[:3, 3] += min(compression, distance) * direction / distance
        pelvis_inv = np.linalg.inv(pose_matrix_wxyz(current_pelvis_pose_wxyz))
        left_b = pelvis_inv @ left_w
        right_b = pelvis_inv @ right_w
        result = np.zeros(23, dtype=np.float32)
        result[:2] = 0.0 if release else 1.0
        result[2:5], result[5:9] = _wrist_action_pose_xyzw(left_b)
        result[9:12], result[12:16] = _wrist_action_pose_xyzw(right_b)
        if camera_navigation_velocity is not None:
            navigation = np.asarray(camera_navigation_velocity, dtype=np.float32)
            if navigation.shape != (3,) or not np.isfinite(navigation).all():
                raise ValueError(
                    "camera_navigation_velocity must be a finite vx/vy/wz vector"
                )
            result[16:19] = navigation
        result[19] = self.base_height
        return result
