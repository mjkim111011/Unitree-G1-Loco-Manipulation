"""Shared Isaac Lab runtime helpers for Arena Pink/Homie teacher rollouts.

This module must be imported only after ``AppLauncher`` has started Isaac Sim.
Arena itself remains in a version-isolated sidecar process; only numeric state
and named joint targets cross the bridge.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs import SubTaskConstraintCoordinationScheme
from isaaclab.utils import math as math_utils
from isaaclab_mimic.datagen.data_generator import (
    get_delta_pose_with_scheme,
    transform_source_data_segment_using_delta_object_pose,
)

from .arena_dataset import load_episode_trajectory, pose7_wxyz_to_matrix
from .mimic_xy_retarget import retarget_object_and_bimanual_wrists


BOX_CONTACT_THRESHOLD_N = 1.0
CONTROL_FREQUENCY_HZ = 50.0
DEFAULT_WRIST_COMPRESSION_M = 0.0
DEFAULT_TEACHER_HAND_STIFFNESS = 4.0
DEFAULT_TEACHER_HAND_DAMPING = 0.5
LEFT_FINGERTIP_NAMES = (
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
)
RIGHT_FINGERTIP_NAMES = (
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
)


def _matrix_to_action_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        transform[:3, 3].astype(np.float32),
        Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32),
    )


class ArenaLiftTrajectory:
    """Reusable Arena GRASP/LIFT target with Mimic TRANSLATE retargeting.

    The source and destination Box assets have different semantic yaw frames,
    so this deliberately uses Mimic's ``TRANSLATE`` coordination scheme, not a
    generic object SE(3) transform.  The exact same translation is applied to
    the source object and both wrists. The collector can then freeze a realized
    bilateral cage through ``realized_hold_wrist_world``; optional symmetric
    compression defaults to zero. Neither mechanism weakens the strict gate.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        close_lead_steps: int = 5,
        hold_lift_margin_m: float = 0.05,
        wrist_compression_m: float = DEFAULT_WRIST_COMPRESSION_M,
        episode_name: str = "demo_0",
    ) -> None:
        if close_lead_steps < 0 or hold_lift_margin_m < 0.0:
            raise ValueError("close lead and hold lift margin must be non-negative")
        if not 0.0 <= wrist_compression_m <= 0.03:
            raise ValueError("wrist compression must be in [0, 0.03]")
        self.path = Path(path).expanduser().resolve()
        source = load_episode_trajectory(self.path, episode_name)
        with h5py.File(self.path, "r") as dataset:
            self.recorded_success = bool(
                dataset[f"data/{episode_name}"].attrs.get("success", False)
            )

        self.actions = source.high_level_action_23d.astype(np.float32, copy=True)
        self.robot_root_pose = source.root_pose_wxyz_pre_step.astype(
            np.float64, copy=True
        )
        self.box_pose = source.box_pose_wxyz_pre_step.astype(np.float64, copy=True)
        self.box_world = pose7_wxyz_to_matrix(self.box_pose)
        self.left_target_world = source.left_world_wrist_command.astype(
            np.float64, copy=True
        )
        self.right_target_world = source.right_world_wrist_command.astype(
            np.float64, copy=True
        )
        self.phase_boundaries = source.phase_boundaries
        if not math.isclose(
            source.control_frequency_hz, CONTROL_FREQUENCY_HZ, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"Arena source must be 50 Hz, got {source.control_frequency_hz:g} Hz"
            )
        if self.actions.ndim != 2 or self.actions.shape[1] != 23:
            raise ValueError(
                f"Arena high-level action must be 23-D, got {self.actions.shape}"
            )
        if not (len(self.actions) == len(self.robot_root_pose) == len(self.box_pose)):
            raise ValueError(
                "Arena action/root/Box trajectories have inconsistent lengths"
            )

        displacement = np.linalg.norm(
            self.box_pose[:, :3] - self.box_pose[0, :3], axis=1
        )
        moving_indices = np.flatnonzero(displacement > 0.005)
        if moving_indices.size == 0:
            raise ValueError("Arena sample never moves the Box by 5 mm")
        self.pick_start = int(moving_indices[0])
        lifted_indices = np.flatnonzero(
            self.box_pose[:, 2] >= self.box_pose[0, 2] + 0.12
        )
        if lifted_indices.size == 0:
            raise ValueError("Arena sample never reaches a 12 cm lift")
        self.lift_index = int(lifted_indices[0])
        self.freeze_index = min(self.lift_index + 12, len(self.actions) - 1)
        self.close_index = max(0, self.pick_start - int(close_lead_steps))
        self.hold_lift_margin_m = float(hold_lift_margin_m)
        self.wrist_compression_m = float(wrist_compression_m)

    def retarget(self, current_box_initial_pose_w: np.ndarray):
        return retarget_object_and_bimanual_wrists(
            source_object_world=self.box_world,
            source_left_wrist_world=self.left_target_world,
            source_right_wrist_world=self.right_target_world,
            target_initial_object_world=current_box_initial_pose_w,
            mimic_transform=transform_source_data_segment_using_delta_object_pose,
            mimic_delta_pose=get_delta_pose_with_scheme,
            translate_coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSLATE,
        )

    def high_level_action(
        self,
        source_index: int,
        current_pelvis_pose_w: np.ndarray,
        retargeted: Any,
        *,
        realized_hold_wrist_world: tuple[np.ndarray, np.ndarray] | None = None,
        hold_support_offset_m: float = 0.0,
    ) -> np.ndarray:
        if not 0.0 <= hold_support_offset_m <= 0.02:
            raise ValueError("hold support offset must be in [0, 0.02]")
        source_index = min(int(source_index), self.freeze_index)
        if realized_hold_wrist_world is not None:
            # Once live physics has produced strict lift + bilateral contact,
            # retain that realized cage instead of replacing it with a source
            # final pose carrying accumulated IK/tracking error.
            source_index = self.freeze_index
        action = self.actions[source_index].copy()
        if realized_hold_wrist_world is None:
            left_world = retargeted.left_wrist_world[source_index].copy()
            right_world = retargeted.right_wrist_world[source_index].copy()
        else:
            if len(realized_hold_wrist_world) != 2:
                raise ValueError(
                    "realized hold requires left and right wrist transforms"
                )
            left_world = np.asarray(
                realized_hold_wrist_world[0], dtype=np.float64
            ).copy()
            right_world = np.asarray(
                realized_hold_wrist_world[1], dtype=np.float64
            ).copy()
            if left_world.shape != (4, 4) or right_world.shape != (4, 4):
                raise ValueError("realized hold wrist targets must be 4x4 transforms")
            left_world[2, 3] += hold_support_offset_m
            right_world[2, 3] += hold_support_offset_m

        if realized_hold_wrist_world is None:
            # Preserve the demonstrated approach/cage geometry exactly. Normal
            # force is optionally ramped only after bilateral lift begins.
            compression_steps = max(1, self.freeze_index - self.lift_index)
            compression_fraction = float(
                np.clip((source_index - self.lift_index) / compression_steps, 0.0, 1.0)
            )
            compression_axis = right_world[:2, 3] - left_world[:2, 3]
            axis_norm = float(np.linalg.norm(compression_axis))
            if axis_norm > 1.0e-6:
                delta = (
                    self.wrist_compression_m
                    * compression_fraction
                    * compression_axis
                    / axis_norm
                )
                left_world[:2, 3] += delta
                right_world[:2, 3] -= delta

            lift_tail_steps = max(1, self.freeze_index - self.lift_index)
            lift_fraction = np.clip(
                (source_index - self.lift_index) / lift_tail_steps, 0.0, 1.0
            )
            lift_compensation = self.hold_lift_margin_m * float(lift_fraction)
            left_world[2, 3] += lift_compensation
            right_world[2, 3] += lift_compensation
        pelvis_pose = np.asarray(current_pelvis_pose_w, dtype=np.float64)
        if pelvis_pose.shape == (7,):
            # Runtime callers naturally read ``xyz + wxyz`` from the
            # articulation.  The nominal collector historically passed a
            # homogeneous matrix instead, so support both representations at
            # this shared boundary and reject every ambiguous shape.
            pelvis_pose = pose7_wxyz_to_matrix(pelvis_pose)
        elif pelvis_pose.shape != (4, 4):
            raise ValueError(
                "current_pelvis_pose_w must be a finite xyz+wxyz pose or 4x4 transform"
            )
        if not np.isfinite(pelvis_pose).all():
            raise ValueError("current_pelvis_pose_w must contain only finite values")
        pelvis_inverse = np.linalg.inv(pelvis_pose)
        left_pelvis = pelvis_inverse @ left_world
        right_pelvis = pelvis_inverse @ right_world
        action[2:5], action[5:9] = _matrix_to_action_pose(left_pelvis)
        action[9:12], action[12:16] = _matrix_to_action_pose(right_pelvis)
        close = float(source_index >= self.close_index)
        action[0] = close
        action[1] = close
        if source_index >= self.freeze_index:
            action[16:19] = 0.0
        return action

    def corrective_high_level_action(
        self,
        source_index: int,
        current_pelvis_pose_w: np.ndarray,
        current_box_pose_wxyz: np.ndarray,
        *,
        reference_box_initial_height_w: float,
    ) -> np.ndarray:
        """Create a feedback recovery label around the live privileged Box pose.

        DAgger must not label a deviated state with a wrist target anchored to
        the reset Box. The demonstrated object-to-wrist transforms at the
        current subtask time are applied to the live Box orientation/x/y,
        while the desired height preserves demonstrated lift progress.
        """

        index = min(max(int(source_index), self.close_index), self.freeze_index)
        live_box = pose7_wxyz_to_matrix(
            np.asarray(current_box_pose_wxyz, dtype=np.float64)
        )
        source_box = self.box_world[index]
        source_lift_progress = max(
            0.0, float(source_box[2, 3] - self.box_world[0, 2, 3])
        )
        desired_box = live_box.copy()
        desired_box[2, 3] = max(
            float(live_box[2, 3]),
            float(reference_box_initial_height_w) + source_lift_progress,
        )
        left_relative = np.linalg.inv(source_box) @ self.left_target_world[index]
        right_relative = np.linalg.inv(source_box) @ self.right_target_world[index]
        left_world = desired_box @ left_relative
        right_world = desired_box @ right_relative

        pelvis = np.asarray(current_pelvis_pose_w, dtype=np.float64)
        if pelvis.shape == (7,):
            pelvis = pose7_wxyz_to_matrix(pelvis)
        if pelvis.shape != (4, 4) or not np.isfinite(pelvis).all():
            raise ValueError("current pelvis pose must be finite xyz+wxyz or 4x4")
        pelvis_inverse = np.linalg.inv(pelvis)
        action = self.actions[index].copy()
        action[:2] = 1.0
        action[2:5], action[5:9] = _matrix_to_action_pose(
            pelvis_inverse @ left_world
        )
        action[9:12], action[12:16] = _matrix_to_action_pose(
            pelvis_inverse @ right_world
        )
        action[16:19] = 0.0
        return action


def configure_arena_teacher_robot(
    robot_cfg: Any,
    *,
    hand_stiffness: float = DEFAULT_TEACHER_HAND_STIFFNESS,
    hand_damping: float = DEFAULT_TEACHER_HAND_DAMPING,
) -> None:
    """Match Arena's G1 Pink teacher dynamics without altering the global asset."""

    robot_cfg.spawn.rigid_props.solver_position_iteration_count = 4
    robot_cfg.spawn.rigid_props.solver_velocity_iteration_count = 0
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 4
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 0
    robot_cfg.soft_joint_pos_limit_factor = 1.0
    robot_cfg.actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={".*_hip_.*": 0.03, ".*_knee_joint": 0.03},
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=50.0,
            velocity_limit=37.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
            friction=0.03,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 250.0,
                "waist_roll_joint": 250.0,
                "waist_pitch_joint": 250.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.03,
            friction=0.03,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_.*_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 5.0,
                ".*_shoulder_roll_joint": 5.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_.*_joint": 2.0,
            },
            armature={
                ".*_shoulder_.*": 0.03,
                ".*_elbow_.*": 0.03,
                ".*_wrist_.*_joint": 0.03,
            },
            friction=0.03,
        ),
        "hands": IdealPDActuatorCfg(
            joint_names_expr=[".*_hand_.*"],
            effort_limit=5.0,
            velocity_limit=10.0,
            stiffness=float(hand_stiffness),
            damping=float(hand_damping),
            armature=0.03,
            friction=0.03,
        ),
    }
    joint_names = list(getattr(robot_cfg, "joint_sdk_names", ()) or ())
    if len(joint_names) != 43 or any(not name for name in joint_names):
        raise ValueError(
            f"Expected 43 named G1 joints in teacher config, got {joint_names}"
        )
    robot_cfg.init_state.joint_pos = {name: 0.0 for name in joint_names}
    robot_cfg.init_state.joint_pos.update(
        {
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "right_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "right_ankle_pitch_joint": -0.2,
        }
    )


def configure_fixed_teacher_scene(
    cfg: Any,
    *,
    box_size_m: float = 0.2,
    box_mass_kg: float = 0.2,
    static_friction: float = 1.5,
    dynamic_friction: float = 1.2,
    restitution: float = 0.0,
    teacher_hand_stiffness: float = DEFAULT_TEACHER_HAND_STIFFNESS,
    keep_work_area_randomization: bool = False,
) -> None:
    """Disable every randomizer except an already configured reset Box x/y range."""

    configure_arena_teacher_robot(
        cfg.scene.robot, hand_stiffness=teacher_hand_stiffness
    )
    cfg.scene.box.spawn.size = (box_size_m,) * 3
    cfg.scene.box.spawn.mass_props.mass = box_mass_kg
    cfg.scene.box.spawn.physics_material.static_friction = static_friction
    cfg.scene.box.spawn.physics_material.dynamic_friction = dynamic_friction
    cfg.scene.box.spawn.physics_material.restitution = restitution
    cfg.scene.box_qr.init_state.pos = (0.0, 0.0, 0.5 * box_size_m + 0.001)
    for name in (
        "randomize_box_scale",
        "preserve_box_qr_scale",
        "randomize_table_height",
        "randomize_box_material",
        "randomize_box_mass",
        "retry_pick_place_scene",
        "physics_material",
        "add_base_mass",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    if hasattr(cfg.events, "reset_pick_place_work_area") and not keep_work_area_randomization:
        cfg.events.reset_pick_place_work_area = None
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.reset_pick_place_scene.params["box_base_size"] = (box_size_m,) * 3
    cfg.events.reset_pick_place_scene.params["box_xy_relative_to_table"] = bool(
        keep_work_area_randomization
    )
    cfg.observations.policy_history.enable_corruption = False
    cfg.observations.policy_current.enable_corruption = False
    cfg.scene.front_camera = None
    cfg.terminations.bad_orientation = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    cfg.rewards.bad_orientation_penalty = None


def student_observation(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    values = []
    for name in ("policy_history", "policy_current"):
        value = observation[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"observation group {name!r} must be concatenated")
        values.append(value)
    result = torch.cat(values, dim=-1)
    if result.shape[-1] != 517:
        raise ValueError(f"expected a 517-D student observation, got {result.shape}")
    return result


def box_contact_counts(env: Any, env_id: int = 0) -> tuple[int, int]:
    counts = {"left": 0, "right": 0}
    for name, sensor in env.scene.sensors.items():
        if not name.endswith("_box_contact") or not (
            name.startswith("left_") or name.startswith("right_")
        ):
            continue
        forces = sensor.data.force_matrix_w
        if forces is None:
            continue
        force = (
            torch.linalg.vector_norm(forces, dim=-1)
            .flatten(start_dim=1)
            .amax(dim=-1)[env_id]
        )
        counts["left" if name.startswith("left_") else "right"] += int(
            force >= BOX_CONTACT_THRESHOLD_N
        )
    return counts["left"], counts["right"]


def body_id(robot: Any, name: str) -> int:
    matches, _ = robot.find_bodies(name)
    if len(matches) != 1:
        raise ValueError(f"Expected one body {name!r}, got {matches}")
    return int(matches[0])


def body_pose_wxyz(robot: Any, body_index: int, env_id: int = 0) -> torch.Tensor:
    return torch.cat(
        (
            robot.data.body_link_pos_w[env_id, body_index],
            robot.data.body_link_quat_w[env_id, body_index],
        ),
        dim=-1,
    )


def bridge_request(
    env: Any, high_level_action: np.ndarray, env_id: int = 0
) -> dict[str, Any]:
    robot = env.scene["robot"]
    torso = body_id(robot, "torso_link")
    torso_quat_w = robot.data.body_link_quat_w[env_id, torso]
    torso_ang_vel_w = robot.data.body_link_ang_vel_w[env_id, torso]
    torso_ang_vel_b = math_utils.quat_apply_inverse(
        torso_quat_w.unsqueeze(0), torso_ang_vel_w.unsqueeze(0)
    )[0]
    return {
        "high_level_action": np.asarray(high_level_action).tolist(),
        "joint_names": list(robot.data.joint_names),
        "joint_position": robot.data.joint_pos[env_id].detach().cpu().tolist(),
        "joint_velocity": robot.data.joint_vel[env_id].detach().cpu().tolist(),
        "default_joint_position": robot.data.default_joint_pos[env_id]
        .detach()
        .cpu()
        .tolist(),
        "floating_base_pose_wxyz": torch.cat(
            (robot.data.root_link_pos_w[env_id], robot.data.root_link_quat_w[env_id])
        )
        .detach()
        .cpu()
        .tolist(),
        "floating_base_velocity_body": torch.cat(
            (
                robot.data.root_link_lin_vel_b[env_id],
                robot.data.root_link_ang_vel_b[env_id],
            )
        )
        .detach()
        .cpu()
        .tolist(),
        "projected_gravity_body": robot.data.projected_gravity_b[env_id]
        .detach()
        .cpu()
        .tolist(),
        "root_angular_velocity_body": robot.data.root_ang_vel_b[env_id]
        .detach()
        .cpu()
        .tolist(),
        "torso_quaternion_wxyz": torso_quat_w.detach().cpu().tolist(),
        "torso_angular_velocity_body": torso_ang_vel_b.detach().cpu().tolist(),
    }


def action_order_target(
    env: Any, target_names: list[str], target_values: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    term = env.action_manager.get_term("JointPositionAction")
    if len(target_names) != len(target_values):
        raise ValueError("Arena target names and values have different lengths")
    if len(set(target_names)) != len(target_names):
        raise ValueError("Arena target contains duplicate joint names")
    target_by_name = dict(zip(target_names, target_values))
    missing = [name for name in term._joint_names if name not in target_by_name]
    if missing:
        raise ValueError(f"Arena target is missing current action joints: {missing}")
    absolute = torch.tensor(
        [[target_by_name[name] for name in term._joint_names]],
        dtype=torch.float32,
        device=env.device,
    )
    raw = (absolute - term._offset) / term._scale
    if not torch.isfinite(raw).all():
        raise ValueError("non-finite normalized teacher action")
    return raw, absolute
