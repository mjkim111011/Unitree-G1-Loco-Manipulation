"""L-shaped flat-palm grip for the rule-based grasp teacher.

Palm and nearly straight index/middle lie on the Box side faces, the thumb tip
on the near face.  Before the close index the demo wrists are blended to the
outside of this grip while the robot steps in; after it the wrists move in and
index/middle curl slightly so the fingertips press the faces.

The constants come from an offline optimisation over the URDF collision meshes
(wrist poses in the Box frame, left grip = mirrored right grip).  This module
mirrors the ``--flat_palm_*`` logic of ``collect_nominal_box_teacher.py`` so the
BC-2 evaluator can reproduce the teacher approach up to the hand-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FLAT_PALM_BOX_WRIST = {
    "left": np.array(
        [[0.999, 0.0319, -0.0322, -0.1803],
         [-0.0296, 0.9972, 0.0691, 0.119],
         [0.0343, -0.0681, 0.9971, -0.0363],
         [0.0, 0.0, 0.0, 1.0]]
    ),
    "right": np.array(
        [[0.9991, -0.0287, -0.0325, -0.1805],
         [0.0263, 0.997, -0.0722, -0.1184],
         [0.0344, 0.0712, 0.9969, -0.0366],
         [0.0, 0.0, 0.0, 1.0]]
    ),
}
FLAT_PALM_HAND = {
    "left_hand_thumb_0_joint": -0.237,
    "left_hand_thumb_1_joint": 0.075,
    "left_hand_thumb_2_joint": 0.133,
    "left_hand_index_0_joint": -0.118,
    "left_hand_index_1_joint": -0.067,
    "left_hand_middle_0_joint": -0.045,
    "left_hand_middle_1_joint": -0.12,
    "right_hand_thumb_0_joint": 0.237,
    "right_hand_thumb_1_joint": -0.075,
    "right_hand_thumb_2_joint": -0.133,
    "right_hand_index_0_joint": 0.118,
    "right_hand_index_1_joint": 0.069,
    "right_hand_middle_0_joint": 0.039,
    "right_hand_middle_1_joint": 0.134,
}


def _smooth(fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _interp_pose(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = (1.0 - fraction) * a[:3, 3] + fraction * b[:3, 3]
    rotations = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    out[:3, :3] = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
    return out


def _action_pose(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = position
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return pose


def _pose_to_action(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pose[:3, 3].astype(np.float32), Rotation.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)


@dataclass
class FlatPalmConfig:
    blend_steps: int = 100
    close_steps: int = 40
    out_m: float = 0.04
    squeeze_m: float = 0.005
    dx_m: float = 0.02
    stand_dist_m: float = 0.42
    feedback_gain: float = 0.05
    press_curl: tuple[float, float] | None = (0.35, 0.5)


@dataclass
class FlatPalmApproach:
    """Per-episode state; call ``reset`` after every environment reset."""

    config: FlatPalmConfig
    corr: dict = field(default_factory=lambda: {"left": np.zeros(3), "right": np.zeros(3)})
    corr_rot: dict = field(default_factory=lambda: {"left": np.zeros(3), "right": np.zeros(3)})

    def reset(self) -> None:
        self.corr = {"left": np.zeros(3), "right": np.zeros(3)}
        self.corr_rot = {"left": np.zeros(3), "right": np.zeros(3)}

    def override(
        self,
        index: int,
        trajectory,
        retargeted,
        box_initial_pose_w: np.ndarray,
        pelvis_pose_w: np.ndarray,
        live_box_pose_w: np.ndarray,
        wrist_poses_w: dict[str, np.ndarray],
        high_level_action: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        """Rewrite ``high_level_action`` in place.

        Returns the hand-shape fraction (0 = untouched) and the base velocity
        command to apply after any camera command (``None`` = leave it).
        """
        cfg = self.config
        blend_start = trajectory.close_index - int(cfg.blend_steps)
        if index < blend_start:
            return 0.0, None
        i = index
        virtual_box = trajectory.box_world[i].copy()
        virtual_box[:3, :3] = box_initial_pose_w[:3, :3]
        virtual_box[:3, 3] += 0.5 * (
            retargeted.left_wrist_world[i][:3, 3] - trajectory.left_target_world[i][:3, 3]
            + retargeted.right_wrist_world[i][:3, 3] - trajectory.right_target_world[i][:3, 3]
        )
        lift_steps = max(1, trajectory.freeze_index - trajectory.lift_index)
        virtual_box[2, 3] += trajectory.hold_lift_margin_m * float(
            np.clip((i - trajectory.lift_index) / lift_steps, 0.0, 1.0)
        )
        closing = float(np.clip((i - trajectory.close_index + 1) / int(cfg.close_steps), 0.0, 1.0))
        lateral = (1.0 - _smooth(closing)) * cfg.out_m - _smooth(closing) * cfg.squeeze_m
        approach = _smooth((i - blend_start + 1) / int(cfg.blend_steps))
        fraction = approach if i < trajectory.close_index else 1.0
        high_level_action[0:2] = 0.0
        inverse_pelvis = np.linalg.inv(pelvis_pose_w)
        command = None
        if cfg.stand_dist_m > 0.0 and i < trajectory.freeze_index:
            box_in_pelvis = inverse_pelvis @ live_box_pose_w
            ex = float(box_in_pelvis[0, 3]) - cfg.stand_dist_m
            ey = float(box_in_pelvis[1, 3])
            command = np.zeros(3)
            if ex > 0.01:
                command[0] = float(np.clip(3.0 * ex, 0.13, 0.25))
                command[1] = float(np.clip(1.5 * ey, -0.08, 0.08))
        for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
            outward = 1.0 if side == "left" else -1.0
            offset = np.eye(4)
            offset[0, 3] = cfg.dx_m
            offset[1, 3] = outward * lateral
            grip = virtual_box @ offset @ FLAT_PALM_BOX_WRIST[side]
            if i < trajectory.close_index:
                demo = _action_pose(high_level_action[position_slice], high_level_action[quaternion_slice])
                grip = _interp_pose(pelvis_pose_w @ demo, grip, approach)
            if cfg.feedback_gain > 0.0:
                if i < trajectory.close_index:
                    actual = wrist_poses_w[side]
                    self.corr[side] = np.clip(self.corr[side] + cfg.feedback_gain * (grip[:3, 3] - actual[:3, 3]), -0.08, 0.08)
                    rot_error = Rotation.from_matrix(grip[:3, :3] @ actual[:3, :3].T).as_rotvec()
                    self.corr_rot[side] = np.clip(self.corr_rot[side] + cfg.feedback_gain * rot_error, -0.5, 0.5)
                grip = grip.copy()
                grip[:3, 3] += self.corr[side]
                grip[:3, :3] = Rotation.from_rotvec(self.corr_rot[side]).as_matrix() @ grip[:3, :3]
            high_level_action[position_slice], high_level_action[quaternion_slice] = _pose_to_action(inverse_pelvis @ grip)
        return fraction, command

    def hand_targets(self, index: int, close_index: int) -> dict[str, float]:
        """L-shape joint goals, with the fingertip press curl ramped in after close."""
        hand_goal = dict(FLAT_PALM_HAND)
        if self.config.press_curl is not None and index >= close_index:
            press = _smooth((index - close_index + 1) / int(self.config.close_steps))
            proximal, distal = self.config.press_curl
            for side, sign in (("left", -1.0), ("right", 1.0)):
                for finger in ("index", "middle"):
                    for joint, value in (("0", proximal), ("1", distal)):
                        name = f"{side}_hand_{finger}_{joint}_joint"
                        hand_goal[name] = (1.0 - press) * hand_goal[name] + press * sign * value
        return hand_goal
