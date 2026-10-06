#!/usr/bin/env python3
"""Collect nominal or x/y-randomized Box demonstrations with Arena teacher.

The current task and IsaacLab-Arena use incompatible Isaac Sim/Lab versions.
This driver keeps the current environment in this process and delegates only
the controller calculation to ``arena_wbc_worker.py`` running in Arena's own
virtual environment.  Isaac Lab Mimic's object-pose transform moves the source
Box and both wrist paths together.  The hand-close command is injected because
the bundled sample leaves both Dex3 hands open for its entire episode.
"""

# Isaac Sim must be launched before the runtime imports below.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher
from unitree_rl_lab.runtime_paths import default_arena_root, default_arena_dataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-G1-43dof-BoxPickPlace")
parser.add_argument(
    "--arena_dataset",
    default=default_arena_dataset(),
)
parser.add_argument("--arena_root", default=default_arena_root())
parser.add_argument("--output", default="data/teacher/current_box_nominal_teacher.hdf5")
parser.add_argument(
    "--snapshot_output",
    default=None,
    help="Optional HDF5 path for one live bilateral-hold physical snapshot per accepted demo.",
)
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument(
    "--target_successes",
    type=int,
    default=0,
    help="Stop after this many accepted demos; zero preserves --num_trials attempt semantics.",
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=None,
    help="Maximum physics attempts when --target_successes is used (default: 2x target).",
)
parser.add_argument(
    "--success_only",
    action="store_true",
    help="Delete failed episodes instead of publishing them in the HDF5 dataset.",
)
parser.add_argument(
    "--randomize_box_xy",
    action="store_true",
    help="Randomize only initial Box x/y; all other object/scene parameters stay fixed.",
)
parser.add_argument(
    "--box_x_range", type=float, nargs=2, default=(0.618, 0.622), metavar=("MIN", "MAX")
)
parser.add_argument(
    "--box_y_range",
    type=float,
    nargs=2,
    default=(-0.002, 0.002),
    metavar=("MIN", "MAX"),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--randomize_work_area", action="store_true", help="Move table/bin and navigate before Mimic grasp; Box x/y ranges become table-relative.")
parser.add_argument("--table_x_range", type=float, nargs=2, default=(0.8, 1.5))
parser.add_argument("--table_y_range", type=float, nargs=2, default=(-0.4, 0.4))
parser.add_argument("--navigation_timeout_seconds", type=float, default=45.0)
parser.add_argument("--navigation_settle_seconds", type=float, default=2.0)
parser.add_argument("--navigation_torso_pitch_rad", type=float, default=-0.25, help="Look farther ahead with the torso-mounted camera while navigating.")
parser.add_argument("--minimum_success_rate", type=float, default=0.80)
parser.add_argument("--hold_seconds", type=float, default=2.0)
parser.add_argument(
    "--hold_grace_seconds",
    type=float,
    default=4.0,
    help="Extra frozen-target time allowed for the continuous hold window to begin.",
)
parser.add_argument(
    "--hold_lift_margin_m",
    type=float,
    default=0.05,
    help="Gradual upward wrist compensation applied over the source lift tail.",
)
parser.add_argument(
    "--wrist_compression_m",
    type=float,
    default=0.0,
    help="Symmetric horizontal inward wrist ramp used to stabilize a physical bimanual hold.",
)
parser.add_argument(
    "--teacher_hand_stiffness",
    type=float,
    default=4.0,
    help="Fixed teacher-only Dex3 Ideal-PD stiffness during data collection.",
)
parser.add_argument(
    "--realized_hold_support_m",
    type=float,
    default=0.003,
    help="World-up support offset used only with --freeze-realized-hold-wrists.",
)
parser.add_argument(
    "--freeze_realized_hold_wrists",
    action="store_true",
    help=(
        "Experimental: replace the validated Mimic hold target with the first realized bilateral "
        "wrist poses. Disabled by default because it shortened strict contact in live validation."
    ),
)
parser.add_argument(
    "--box_size",
    type=float,
    default=0.20,
    help="Fixed nominal cube edge length in metres.",
)
parser.add_argument(
    "--box_mass", type=float, default=0.20, help="Fixed nominal Box mass in kilograms."
)
parser.add_argument("--box_static_friction", type=float, default=1.5)
parser.add_argument("--box_dynamic_friction", type=float, default=1.2)
parser.add_argument("--box_restitution", type=float, default=0.0)
parser.add_argument("--close_lead_steps", type=int, default=5)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument(
    "--disable_fabric", action="store_true", help="Use USD I/O instead of Fabric."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_trials <= 0:
    parser.error("--num_trials must be positive")
if args_cli.target_successes < 0:
    parser.error("--target_successes must be non-negative")
if args_cli.max_attempts is not None and args_cli.max_attempts <= 0:
    parser.error("--max_attempts must be positive")
if not 0.0 < args_cli.minimum_success_rate <= 1.0:
    parser.error("--minimum_success_rate must be in (0, 1]")
if args_cli.hold_seconds <= 0.0:
    parser.error("--hold_seconds must be positive")
if args_cli.hold_grace_seconds < 0.0:
    parser.error("--hold_grace_seconds must be non-negative")
if args_cli.hold_lift_margin_m < 0.0:
    parser.error("--hold_lift_margin_m must be non-negative")
if not 0.0 <= args_cli.wrist_compression_m <= 0.03:
    parser.error("--wrist_compression_m must be in [0, 0.03]")
if args_cli.teacher_hand_stiffness <= 0.0:
    parser.error("--teacher_hand_stiffness must be positive")
if not 0.0 <= args_cli.realized_hold_support_m <= 0.02:
    parser.error("--realized_hold_support_m must be in [0, 0.02]")
if args_cli.close_lead_steps < 0:
    parser.error("--close_lead_steps must be non-negative")
if args_cli.box_size <= 0.0 or args_cli.box_mass <= 0.0:
    parser.error("--box_size and --box_mass must be positive")
if (
    min(
        args_cli.box_static_friction,
        args_cli.box_dynamic_friction,
        args_cli.box_restitution,
    )
    < 0.0
):
    parser.error("fixed Box friction and restitution must be non-negative")
for range_name in ("box_x_range", "box_y_range", "table_x_range", "table_y_range"):
    bounds = getattr(args_cli, range_name)
    if (
        not math.isfinite(bounds[0])
        or not math.isfinite(bounds[1])
        or bounds[0] > bounds[1]
    ):
        parser.error(f"--{range_name} requires finite MIN <= MAX")
if not all(math.isfinite(value) and value > 0 for value in (args_cli.navigation_timeout_seconds, args_cli.navigation_settle_seconds)):
    parser.error("navigation durations must be finite and positive")
if not math.isfinite(args_cli.navigation_torso_pitch_rad) or abs(args_cli.navigation_torso_pitch_rad) > 0.35:
    parser.error("navigation torso pitch must be finite within +/-0.35 rad")
if args_cli.randomize_work_area:
    args_cli.randomize_box_xy = True
if args_cli.randomize_box_xy:
    supported_offset = 0.2 - 0.5 * args_cli.box_size - 0.02
    local_x = args_cli.box_x_range if args_cli.randomize_work_area else [v - 0.62 for v in args_cli.box_x_range]
    if supported_offset < 0 or any(abs(v) > supported_offset + 1e-6 for interval in (local_x, args_cli.box_y_range) for v in interval):
        parser.error("Box x/y ranges must keep the full Box on the 0.4m tabletop with 2cm edge margin; work-area ranges are table-relative")
if args_cli.randomize_box_xy:
    # The randomized dataset is a success-only BC corpus by construction.
    args_cli.success_only = True
if args_cli.target_successes > 0:
    args_cli.success_only = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import hashlib
import importlib
import json

import gymnasium as gym
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs import SubTaskConstraintCoordinationScheme
from isaaclab.utils import math as math_utils
from isaaclab_mimic.datagen.data_generator import (
    get_delta_pose_with_scheme,
    transform_source_data_segment_using_delta_object_pose,
)

from unitree_rl_lab.utils.arena_wbc_bridge import ArenaWBCBridge
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.teacher.arena_teacher_runtime import (
    ArenaLiftTrajectory as SharedArenaLiftTrajectory,
    configure_arena_teacher_robot as configure_shared_arena_teacher_robot,
)
from unitree_rl_lab.utils.teacher import (
    MIMIC_TRANSFORM_QUALNAME,
    FixedBoxSceneParameters,
    XYPositionRandomization,
    episode_passes_bimanual_lift_hold,
    load_episode_trajectory,
    retarget_object_and_bimanual_wrists,
)
from unitree_rl_lab.utils.teacher_demo_recorder import (
    TeacherDemoRecorder,
    TeacherDemoStep,
)
from unitree_rl_lab.utils.teacher.physical_snapshot_runtime import (
    capture_physical_snapshot,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import (
    PhysicalSnapshotStore,
    SnapshotKind,
)
from unitree_rl_lab.utils.teacher.student_observation import (
    STUDENT_FEATURE_SLICES,
    schema_metadata,
)


CONTROL_FREQUENCY_HZ = 50.0
BOX_CONTACT_THRESHOLD_N = 1.0
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


def _pose_matrix_wxyz(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    quaternion_xyzw = np.r_[pose[4:7], pose[3]]
    transform[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _matrix_to_action_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3].astype(np.float32), Rotation.from_matrix(
        transform[:3, :3]
    ).as_quat().astype(np.float32)


def _torch_pose_wxyz(
    position: torch.Tensor, quaternion_wxyz: torch.Tensor
) -> np.ndarray:
    pose = torch.cat((position, quaternion_wxyz), dim=-1).detach().cpu().numpy()
    return _pose_matrix_wxyz(pose)


def _configure_arena_teacher_robot(robot_cfg) -> None:
    """Match the source Arena G1 dynamics without changing the task asset globally."""
    # Arena's non-AGILE g1_wbc_pink embodiment uses an explicit Ideal-PD
    # articulation with self-collision disabled.  The Box task normally uses
    # weaker implicit drives and self-collision, which materially changes how
    # the same absolute WBC target tracks, especially at the shoulders/hands.
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
            stiffness=float(args_cli.teacher_hand_stiffness),
            damping=0.7,
            armature=0.03,
            friction=0.03,
        ),
    }

    # Arena resets every upper-body and hand joint to zero.  Reusing the Box
    # task's flexed-arm/closed-hand defaults creates a large first-step error
    # against Pink's zero-based warm start.
    joint_names = list(getattr(robot_cfg, "joint_sdk_names", ()) or ())
    if len(joint_names) != 43 or any(not name for name in joint_names):
        raise ValueError(
            f"Expected 43 named G1 joints in the teacher robot config, got {joint_names}"
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


class ArenaLiftTrajectory:
    """Retarget the sample's Box and both wrists with Isaac Lab Mimic."""

    def __init__(
        self,
        path: str | Path,
        close_lead_steps: int,
        hold_lift_margin_m: float,
        wrist_compression_m: float,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        source = load_episode_trajectory(self.path, "demo_0")
        with h5py.File(self.path, "r") as dataset:
            episode = dataset["data/demo_0"]
            self.recorded_success = bool(episode.attrs.get("success", False))

        self.actions = source.high_level_action_23d.astype(np.float32, copy=True)
        self.robot_root_pose = source.root_pose_wxyz_pre_step.astype(
            np.float64, copy=True
        )
        self.box_pose = source.box_pose_wxyz_pre_step.astype(np.float64, copy=True)
        self.box_world = np.stack([_pose_matrix_wxyz(pose) for pose in self.box_pose])
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
                f"Arena source must be {CONTROL_FREQUENCY_HZ:g} Hz, got {source.control_frequency_hz:g} Hz"
            )

        if self.actions.ndim != 2 or self.actions.shape[1] != 23:
            raise ValueError(
                f"Arena high-level action must be 23-D, got {self.actions.shape}"
            )
        if len(self.actions) != len(self.robot_root_pose) or len(self.actions) != len(
            self.box_pose
        ):
            raise ValueError(
                "Arena action/root/Box trajectories have inconsistent lengths"
            )

        displacement = np.linalg.norm(
            self.box_pose[:, :3] - self.box_pose[0, :3], axis=1
        )
        moving = displacement > 0.005
        moving_indices = np.flatnonzero(moving)
        if moving_indices.size == 0:
            raise ValueError("Arena sample never moves the Box by 5 mm")
        self.pick_start = int(moving_indices[0])

        lifted = self.box_pose[:, 2] >= self.box_pose[0, 2] + 0.12
        lifted_indices = np.flatnonzero(lifted)
        if lifted_indices.size == 0:
            raise ValueError("Arena sample never reaches a 12 cm lift")
        self.lift_index = int(lifted_indices[0])
        # Let the source lift settle briefly, but stop before its long carry
        # navigation begins.  The current nominal objective is lift + hold.
        self.freeze_index = min(self.lift_index + 12, len(self.actions) - 1)
        self.close_index = max(0, self.pick_start - int(close_lead_steps))
        self.hold_lift_margin_m = float(hold_lift_margin_m)
        self.wrist_compression_m = float(wrist_compression_m)

        self.source_box_initial_position = self.box_pose[0, :3].copy()

    def retarget(self, current_box_initial_pose_w: np.ndarray):
        """Build one trial's coherent source-object/bimanual target paths."""

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
        retargeted,
    ) -> np.ndarray:
        source_index = min(source_index, self.freeze_index)
        action = self.actions[source_index].copy()

        left_world = retargeted.left_wrist_world[source_index].copy()
        right_world = retargeted.right_wrist_world[source_index].copy()
        # Ramp both wrist targets a small, symmetric distance toward the
        # retargeted Box center.  This creates sustained normal force instead
        # of merely touching the cube while preserving the source trajectory's
        # height and orientation.  The opposing horizontal forces cancel at
        # the object level and therefore do not reward tossing the Box.
        compression_steps = max(1, self.freeze_index - self.lift_index)
        compression_fraction = float(
            np.clip((source_index - self.lift_index) / compression_steps, 0.0, 1.0)
        )
        compression_axis = right_world[:2, 3] - left_world[:2, 3]
        axis_norm = float(np.linalg.norm(compression_axis))
        if axis_norm > 1.0e-6:
            compression_delta = (
                self.wrist_compression_m
                * compression_fraction
                * compression_axis
                / axis_norm
            )
            left_world[:2, 3] += compression_delta
            right_world[:2, 3] -= compression_delta
        # The source Box reaches 12 cm but settles a few millimetres below that
        # threshold under the current scene's object/contact model.  Ramp a
        # small, explicit world-up compensation over the lift tail instead of
        # relaxing the success criterion or applying an abrupt hold command.
        lift_tail_steps = max(1, self.freeze_index - self.lift_index)
        lift_fraction = np.clip(
            (source_index - self.lift_index) / lift_tail_steps, 0.0, 1.0
        )
        lift_compensation = self.hold_lift_margin_m * float(lift_fraction)
        left_world[2, 3] += lift_compensation
        right_world[2, 3] += lift_compensation
        pelvis_inverse = np.linalg.inv(current_pelvis_pose_w)
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


def _configure_nominal_environment():
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    cfg.seed = int(args_cli.seed)
    size = float(args_cli.box_size)
    configure_shared_arena_teacher_robot(
        cfg.scene.robot,
        hand_stiffness=float(args_cli.teacher_hand_stiffness),
        hand_damping=0.5,
    )
    cfg.scene.box.spawn.size = (size, size, size)
    cfg.scene.box.spawn.mass_props.mass = float(args_cli.box_mass)
    cfg.scene.box.spawn.physics_material.static_friction = float(
        args_cli.box_static_friction
    )
    cfg.scene.box.spawn.physics_material.dynamic_friction = float(
        args_cli.box_dynamic_friction
    )
    cfg.scene.box.spawn.physics_material.restitution = float(args_cli.box_restitution)
    cfg.scene.box_qr.init_state.pos = (0.0, 0.0, 0.5 * size + 0.001)

    # This gate permits reset-time Box/table/bin x/y randomization. Scale, mass,
    # material, table height, Box z/yaw, robot state, and observation corruption
    # are fixed for the complete run.
    cfg.events.randomize_box_scale = None
    cfg.events.preserve_box_qr_scale = None
    cfg.events.randomize_table_height = None
    cfg.events.randomize_box_material = None
    cfg.events.randomize_box_mass = None
    cfg.events.retry_pick_place_scene = None
    # These two inherited startup events randomize the robot rather than the
    # Box, but they still make a supposedly nominal rollout nondeterministic.
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    xy_randomization = XYPositionRandomization(
        tuple(float(value) for value in args_cli.box_x_range),
        tuple(float(value) for value in args_cli.box_y_range),
    )
    cfg.events.reset_pick_place_scene.params["grasp_box_xy_range"] = (
        xy_randomization.as_event_range()
        if args_cli.randomize_box_xy
        else {"x": (0.62, 0.62), "y": (0.0, 0.0)}
    )
    cfg.events.reset_pick_place_scene.params["box_base_size"] = (size, size, size)
    cfg.events.reset_pick_place_scene.params["box_xy_relative_to_table"] = bool(args_cli.randomize_work_area)
    if args_cli.randomize_work_area:
        placement = cfg.events.reset_pick_place_work_area.params
        placement["table_xy_range"] = {"x": tuple(args_cli.table_x_range), "y": tuple(args_cli.table_y_range)}
        placement["bin_xy_range"] = {"x": (-0.5, 0.5), "y": (-1.8, -1.3)}
        nav = cfg.commands.base_velocity
        # Match the original collector's camera-guided pre-grasp endpoint,
        # not the initial robot-to-table separation (0.62m).
        nav.pregrasp_standoff_distance = 0.42
        nav.pregrasp_world_heading_rad = 0.0
        # Homie stops tracking tiny walking commands around 3cm residual.
        # This is a navigation tolerance, not a relaxed lift/contact gate.
        nav.pregrasp_position_tolerance = 0.05
        nav.pregrasp_lateral_tolerance = 0.05
        nav.pregrasp_yaw_tolerance = 0.10
        nav.pregrasp_position_gain = 2.0
        nav.pregrasp_alignment_exit_multiplier = 1.5
        nav.pregrasp_search_forward_speed = 0.0
    else:
        cfg.events.reset_pick_place_work_area = None

    cfg.observations.policy_history.enable_corruption = False
    cfg.observations.policy_current.enable_corruption = False
    # QR-derived compact observations are geometric; an RGB render is not
    # required for teacher collection and would only reduce control rate.
    cfg.scene.front_camera = None
    # Avoid auto-reset before post-step state is recorded.  The driver computes
    # the same 0.8-rad bad-orientation condition and ends the trial itself.
    cfg.terminations.bad_orientation = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    # ``bad_orientation_penalty`` is implemented through
    # ``is_terminated_term(term_keys="bad_orientation")``.  Removing the
    # termination while leaving this reward active makes RewardManager fail
    # during construction, even though rewards are irrelevant to collection.
    cfg.rewards.bad_orientation_penalty = None
    cfg.episode_length_s = max(12.0, args_cli.hold_seconds + 8.0) + (args_cli.navigation_timeout_seconds if args_cli.randomize_work_area else 0.0)
    return cfg


def _student_observation(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    groups = []
    for name in ("policy_history", "policy_current"):
        value = observation[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Observation group {name!r} must be concatenated, got {type(value)}"
            )
        groups.append(value)
    return torch.cat(groups, dim=-1)[0]


def _grasp_scope_observation(env) -> dict[str, torch.Tensor]:
    """Refresh the phase-0 camera command without advancing history twice."""

    if hasattr(env, "task_phase"):
        env.task_phase[0] = 0
    env.command_manager.compute(0.0)
    # env.step already updated proprioceptive history exactly once.
    return env.observation_manager.compute(update_history=False)


def _phase_student_observation(
    env, observation: dict[str, torch.Tensor], *, grasp_lift: bool
) -> torch.Tensor:
    """Expose camera navigation to APPROACH and a zero command to BC-2."""

    value = _student_observation(observation).clone()
    command_slice = STUDENT_FEATURE_SLICES["velocity_commands"]
    command = env.command_manager.get_term("base_velocity").command[0]
    value[command_slice.start : command_slice.stop] = 0.0 if grasp_lift else command
    return value


def _box_contact_counts(env) -> tuple[int, int]:
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
            .amax(dim=-1)[0]
        )
        side = "left" if name.startswith("left_") else "right"
        counts[side] += int(force >= BOX_CONTACT_THRESHOLD_N)
    return counts["left"], counts["right"]


def _body_ids(robot, names: tuple[str, ...]) -> list[int]:
    """Resolve a fixed body-name list without depending on scene ordering."""
    ids: list[int] = []
    for name in names:
        matches, _ = robot.find_bodies(name)
        if len(matches) != 1:
            raise ValueError(f"Expected one robot body named {name!r}, got {matches}")
        ids.append(int(matches[0]))
    return ids


def _body_pose_wxyz(robot, body_id: int) -> torch.Tensor:
    return torch.cat(
        (
            robot.data.body_link_pos_w[0, body_id],
            robot.data.body_link_quat_w[0, body_id],
        ),
        dim=-1,
    )


def _bridge_request(env, high_level_action: np.ndarray) -> dict:
    robot = env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    torso_quat_w = robot.data.body_link_quat_w[0, torso_id]
    torso_ang_vel_w = robot.data.body_link_ang_vel_w[0, torso_id]
    torso_ang_vel_b = math_utils.quat_apply_inverse(
        torso_quat_w.unsqueeze(0), torso_ang_vel_w.unsqueeze(0)
    )[0]
    return {
        "high_level_action": high_level_action.tolist(),
        "joint_names": list(robot.data.joint_names),
        "joint_position": robot.data.joint_pos[0].detach().cpu().tolist(),
        "joint_velocity": robot.data.joint_vel[0].detach().cpu().tolist(),
        "default_joint_position": robot.data.default_joint_pos[0]
        .detach()
        .cpu()
        .tolist(),
        "floating_base_pose_wxyz": torch.cat(
            (robot.data.root_link_pos_w[0], robot.data.root_link_quat_w[0])
        )
        .detach()
        .cpu()
        .tolist(),
        "floating_base_velocity_body": torch.cat(
            (robot.data.root_link_lin_vel_b[0], robot.data.root_link_ang_vel_b[0])
        )
        .detach()
        .cpu()
        .tolist(),
        "projected_gravity_body": robot.data.projected_gravity_b[0]
        .detach()
        .cpu()
        .tolist(),
        "root_angular_velocity_body": robot.data.root_ang_vel_b[0]
        .detach()
        .cpu()
        .tolist(),
        "torso_quaternion_wxyz": torso_quat_w.detach().cpu().tolist(),
        "torso_angular_velocity_body": torso_ang_vel_b.detach().cpu().tolist(),
    }


def _action_order_target(
    env, target_names: list[str], target_values: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    action_term = env.action_manager.get_term("JointPositionAction")
    action_names = list(action_term._joint_names)
    target_by_name = dict(zip(target_names, target_values))
    missing = [name for name in action_names if name not in target_by_name]
    if missing:
        raise ValueError(f"Arena target is missing current action joints: {missing}")
    absolute_target = torch.tensor(
        [[target_by_name[name] for name in action_names]],
        dtype=torch.float32,
        device=env.device,
    )
    offset = action_term._offset
    scale = action_term._scale
    raw_action = (absolute_target - offset) / scale
    if not torch.isfinite(raw_action).all():
        raise ValueError("Non-finite normalized teacher action")
    return raw_action, absolute_target


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    print("[teacher] loading Arena GRASP/LIFT trajectory", flush=True)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    trajectory = SharedArenaLiftTrajectory(
        args_cli.arena_dataset,
        close_lead_steps=args_cli.close_lead_steps,
        hold_lift_margin_m=args_cli.hold_lift_margin_m,
        wrist_compression_m=args_cli.wrist_compression_m,
    )
    cfg = _configure_nominal_environment()
    print("[teacher] creating the current Box environment", flush=True)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    print("[teacher] environment ready", flush=True)
    if not math.isclose(
        env.step_dt, 1.0 / CONTROL_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise RuntimeError(f"Expected a 50 Hz task, got step_dt={env.step_dt}")

    output_path = Path(args_cli.output).expanduser().resolve()
    box_size = np.full(3, float(args_cli.box_size), dtype=np.float32)
    task_cfg_module = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.box_carry_with_hand_env_cfg"
    )
    fixed_scene = FixedBoxSceneParameters(
        box_size_m=(float(args_cli.box_size),) * 3,
        box_mass_kg=float(args_cli.box_mass),
        static_friction=float(args_cli.box_static_friction),
        dynamic_friction=float(args_cli.box_dynamic_friction),
        restitution=float(args_cli.box_restitution),
        table_surface_height_m=float(task_cfg_module.TABLE_SURFACE_HEIGHT),
        box_yaw_rad=0.0,
    )
    xy_randomization = XYPositionRandomization(
        tuple(float(value) for value in args_cli.box_x_range),
        tuple(float(value) for value in args_cli.box_y_range),
    )
    bin_pose = np.array(
        [*task_cfg_module.BIN_CENTER_E, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    phase_names = {0: "grasp", 1: "carry", 2: "place", 3: "complete"}
    action_joint_names = list(
        env.action_manager.get_term("JointPositionAction")._joint_names
    )
    metadata = {
        "task": args_cli.task,
        "teacher": "Arena g1_wbc_pink + HomieV2 sidecar",
        "arena_root": str(Path(args_cli.arena_root).expanduser().resolve()),
        "arena_source_dataset": str(trajectory.path),
        "arena_source_dataset_sha256": _file_sha256(trajectory.path),
        "arena_source_success_attribute": trajectory.recorded_success,
        "arena_source_hands_are_open": bool(np.all(trajectory.actions[:, :2] == 0.0)),
        "injected_bimanual_close_start_step": trajectory.close_index,
        "source_pick_start_step": trajectory.pick_start,
        "source_12cm_lift_step": trajectory.lift_index,
        "source_freeze_step": trajectory.freeze_index,
        "source_inferred_phase_boundaries": list(trajectory.phase_boundaries),
        "hold_lift_margin_m": trajectory.hold_lift_margin_m,
        "wrist_compression_m": trajectory.wrist_compression_m,
        "teacher_hand_stiffness": float(args_cli.teacher_hand_stiffness),
        "realized_hold_support_m": float(args_cli.realized_hold_support_m),
        "hold_stabilization": (
            "freeze realized bilateral wrist world poses at first strict lift/contact"
            if args_cli.freeze_realized_hold_wrists
            else "validated Mimic wrist trajectory and frozen source lift target"
        ),
        "hold_grace_seconds": float(args_cli.hold_grace_seconds),
        "fixed_scene_parameters": fixed_scene.as_dict(),
        "randomization_mode": "table_and_box_xy" if args_cli.randomize_work_area else ("box_xy_only" if args_cli.randomize_box_xy else "none"),
        "box_xy_frame": "table_relative" if args_cli.randomize_work_area else "environment",
        "table_xy_range_m": {"x": args_cli.table_x_range, "y": args_cli.table_y_range} if args_cli.randomize_work_area else None,
        "navigation_required": bool(args_cli.randomize_work_area),
        "navigation_standoff_distance_m": 0.42 if args_cli.randomize_work_area else None,
        "navigation_world_heading_rad": 0.0 if args_cli.randomize_work_area else None,
        "navigation_torso_pitch_rad": float(args_cli.navigation_torso_pitch_rad) if args_cli.randomize_work_area else None,
        "box_xy_range_m": xy_randomization.as_event_range(),
        "random_seed": int(args_cli.seed),
        "mimic_object_wrist_transform": MIMIC_TRANSFORM_QUALNAME,
        "failed_demonstrations_published": not args_cli.success_only,
        "teacher_skill_scope": "approach_grasp_lift",
        "recorded_phase_id_contract": "0=grasp collector scope; BC splitter uses close/contact signals",
        "student_velocity_command_contract": (
            "camera pregrasp command for APPROACH; forced [0,0,0] from BC-2 close onset"
        ),
        "success_definition": (
            "12cm lift + bilateral >=1N fingertip contact held continuously for "
            f"{args_cli.hold_seconds:g}s"
        ),
    }
    metadata.update(schema_metadata())
    print("[teacher] recorder metadata and fixed-scene gate ready", flush=True)

    target_successes = int(args_cli.target_successes)
    if target_successes > 0:
        max_attempts = (
            int(args_cli.max_attempts)
            if args_cli.max_attempts is not None
            else max(2 * target_successes, args_cli.num_trials)
        )
    else:
        max_attempts = int(args_cli.max_attempts or args_cli.num_trials)
    grasp_lift_successes = 0
    bad_orientations = 0
    hold_successes = 0
    attempts = 0
    recorded_demos = 0
    trial_results: list[dict[str, object]] = []
    recorder: TeacherDemoRecorder | None = None
    snapshot_store: PhysicalSnapshotStore | None = None
    snapshots_recorded = 0
    output_published = False
    snapshots_published = False
    try:
        recorder = TeacherDemoRecorder(
            output_path,
            control_frequency_hz=CONTROL_FREQUENCY_HZ,
            joint_names=action_joint_names,
            phase_names=phase_names,
            quaternion_convention="wxyz",
            metadata=metadata,
            overwrite=args_cli.overwrite,
        )
        print(f"[teacher] streaming recorder opened: {output_path}", flush=True)
        if args_cli.snapshot_output is not None:
            snapshot_store = PhysicalSnapshotStore(
                args_cli.snapshot_output,
                joint_names=list(env.scene["robot"].data.joint_names),
                overwrite=args_cli.overwrite,
                metadata={
                    "task": args_cli.task,
                    "source_teacher_dataset": str(output_path),
                    "capture_gate": metadata["success_definition"],
                    "randomization_mode": metadata["randomization_mode"],
                },
            )
        with ArenaWBCBridge(
            arena_root=args_cli.arena_root, step_timeout_s=10.0
        ) as bridge:
            print("[teacher] Arena Pink/Homie sidecar ready", flush=True)
            while attempts < max_attempts and (
                target_successes == 0 or hold_successes < target_successes
            ):
                trial = attempts
                attempts += 1
                observation, _ = env.reset()
                print(
                    f"[teacher] attempt {attempts}/{max_attempts}: reset complete",
                    flush=True,
                )
                observation = _grasp_scope_observation(env)
                bridge.reset()
                robot = env.scene["robot"]
                box = env.scene["box"]
                pelvis_id = robot.find_bodies("pelvis")[0][0]
                left_wrist_id = _body_ids(robot, ("left_wrist_yaw_link",))[0]
                right_wrist_id = _body_ids(robot, ("right_wrist_yaw_link",))[0]
                left_fingertip_ids = _body_ids(robot, LEFT_FINGERTIP_NAMES)
                right_fingertip_ids = _body_ids(robot, RIGHT_FINGERTIP_NAMES)
                box_initial_position = (
                    box.data.root_link_pos_w[0].detach().cpu().numpy().copy()
                )
                box_initial_pose = _torch_pose_wxyz(
                    box.data.root_link_pos_w[0], box.data.root_link_quat_w[0]
                )
                retargeted = trajectory.retarget(box_initial_pose)
                initial_root_xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
                table_xy = (env.table_center_e[0, :2].detach().cpu().numpy().copy() if args_cli.randomize_work_area else np.array([0.62, 0.0]))
                if args_cli.randomize_work_area:
                    bin_pose[:3] = env.bin_center_e[0].detach().cpu().numpy() + env.scene.env_origins[0].detach().cpu().numpy()
                navigation_complete = not args_cli.randomize_work_area
                navigation_steps = 0
                navigation_stable_steps = 0
                navigation_budget = int(math.ceil(args_cli.navigation_timeout_seconds / env.step_dt)) if args_cli.randomize_work_area else 0
                navigation_required_stable = int(math.ceil(args_cli.navigation_settle_seconds / env.step_dt))
                navigation_hold = None
                navigation_end_root_pose = None
                initial_yaw = float(
                    Rotation.from_matrix(box_initial_pose[:3, :3]).as_euler("xyz")[2]
                )
                if not math.isclose(
                    initial_yaw, fixed_scene.box_yaw_rad, abs_tol=1.0e-5
                ):
                    raise RuntimeError(
                        f"Box yaw changed during x/y-only randomization: {initial_yaw}"
                    )
                recorder.start_episode(
                    metadata={
                        "attempt_index": trial,
                        "randomized_box_xy": bool(args_cli.randomize_box_xy),
                        "initial_box_pose_wxyz": torch.cat(
                            (box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        )
                        .detach()
                        .cpu()
                        .tolist(),
                        "mimic_transform": MIMIC_TRANSFORM_QUALNAME,
                    }
                )
                box_initial_height = float(box_initial_position[2])
                required_hold_steps = int(
                    math.ceil(args_cli.hold_seconds / env.step_dt)
                )
                stable_hold_steps = 0
                nominal_success = False
                grasp_lift_achieved = False
                bad_orientation = False
                termination_reason = "teacher_horizon"
                grace_steps = int(math.ceil(args_cli.hold_grace_seconds / env.step_dt))
                max_steps = trajectory.freeze_index + required_hold_steps + grace_steps + navigation_budget
                max_lift_m = 0.0
                max_tilt_rad = 0.0
                max_left_contacts = 0
                max_right_contacts = 0
                max_stable_hold_steps = 0
                min_left_tip_distance_m = math.inf
                min_right_tip_distance_m = math.inf
                lift_history: list[bool] = []
                left_contact_history: list[bool] = []
                right_contact_history: list[bool] = []
                accepted_snapshot = None
                realized_hold_wrist_world = None
                realized_hold_lock_step: int | None = None

                for step in range(max_steps):
                    navigating = not navigation_complete
                    source_index = 0 if navigating else min(step - navigation_steps, trajectory.freeze_index)
                    pelvis_pose = _torch_pose_wxyz(
                        robot.data.body_link_pos_w[0, pelvis_id],
                        robot.data.body_link_quat_w[0, pelvis_id],
                    )
                    high_level_action = trajectory.high_level_action(
                        source_index,
                        pelvis_pose,
                        retargeted,
                        realized_hold_wrist_world=realized_hold_wrist_world,
                        hold_support_offset_m=(
                            float(args_cli.realized_hold_support_m)
                            if realized_hold_wrist_world is not None
                            else 0.0
                        ),
                    )
                    grasp_lift_scope = source_index >= trajectory.close_index
                    if navigating:
                        if navigation_hold is None:
                            navigation_hold = high_level_action.copy()
                            navigation_hold[:2] = 0.0
                            inverse_pelvis = np.linalg.inv(pelvis_pose)
                            for wrist_id, position_slice, quaternion_slice in (
                                (left_wrist_id, slice(2, 5), slice(5, 9)),
                                (right_wrist_id, slice(9, 12), slice(12, 16)),
                            ):
                                wrist_pose = _torch_pose_wxyz(robot.data.body_link_pos_w[0, wrist_id], robot.data.body_link_quat_w[0, wrist_id])
                                navigation_hold[position_slice], navigation_hold[quaternion_slice] = _matrix_to_action_pose(inverse_pelvis @ wrist_pose)
                        high_level_action = navigation_hold.copy()
                        high_level_action[21] = float(args_cli.navigation_torso_pitch_rad)
                    camera_command = env.command_manager.get_term(
                        "base_velocity"
                    ).command[0]
                    high_level_action[16:19] = (
                        0.0
                        if grasp_lift_scope
                        else camera_command.detach().cpu().numpy()
                    )
                    pre_step_observation = _phase_student_observation(
                        env, observation, grasp_lift=grasp_lift_scope
                    )
                    # This collector produces only APPROACH + GRASP/LIFT.  The
                    # task reward manager may internally advance to CARRY after
                    # its shorter phase gate, but that must neither relabel the
                    # static 2 s hold tail nor expose Bin navigation commands to
                    # the grasp student.
                    phase_id = 0
                    target_names, target_values = bridge.step(
                        _bridge_request(env, high_level_action)
                    )
                    raw_action, absolute_target = _action_order_target(
                        env, target_names, target_values
                    )

                    observation, _, _, _, _ = env.step(raw_action)
                    observation = _grasp_scope_observation(env)
                    if navigating:
                        navigation_steps += 1
                        nav_term = env.command_manager.get_term("base_velocity")
                        navigation_stable_steps = navigation_stable_steps + 1 if bool(nav_term.at_pregrasp[0]) else 0
                        navigation_complete = navigation_stable_steps >= navigation_required_stable
                        if navigation_complete:
                            navigation_end_root_pose = robot.data.root_link_state_w[0, :7].detach().cpu().tolist()
                        if navigation_steps % 250 == 0 or navigation_complete:
                            print(f"[teacher navigation] step={navigation_steps}/{navigation_budget} visible={bool(nav_term.box_visible[0])} error={nav_term.pregrasp_position_error_b[0].tolist()} settle={navigation_stable_steps}/{navigation_required_stable}", flush=True)
                    left_count, right_count = _box_contact_counts(env)
                    box_height = float(box.data.root_link_pos_w[0, 2])
                    lift_height = box_height - box_initial_height
                    max_lift_m = max(max_lift_m, lift_height)
                    max_left_contacts = max(max_left_contacts, left_count)
                    max_right_contacts = max(max_right_contacts, right_count)
                    left_fingertips = robot.data.body_link_pos_w[0, left_fingertip_ids]
                    right_fingertips = robot.data.body_link_pos_w[
                        0, right_fingertip_ids
                    ]
                    box_position = box.data.root_link_pos_w[0]
                    min_left_tip_distance_m = min(
                        min_left_tip_distance_m,
                        float(
                            torch.linalg.vector_norm(
                                left_fingertips - box_position, dim=-1
                            ).amin()
                        ),
                    )
                    min_right_tip_distance_m = min(
                        min_right_tip_distance_m,
                        float(
                            torch.linalg.vector_norm(
                                right_fingertips - box_position, dim=-1
                            ).amin()
                        ),
                    )
                    lifted = box_height >= box_initial_height + 0.12
                    bilateral_contact = left_count > 0 and right_count > 0
                    lift_history.append(lifted)
                    left_contact_history.append(left_count > 0)
                    right_contact_history.append(right_count > 0)
                    if lifted and bilateral_contact:
                        stable_hold_steps += 1
                        grasp_lift_achieved = True
                    else:
                        stable_hold_steps = 0
                    max_stable_hold_steps = max(
                        max_stable_hold_steps, stable_hold_steps
                    )
                    projected_gravity = robot.data.projected_gravity_b[0]
                    tilt = torch.acos(torch.clamp(-projected_gravity[2], -1.0, 1.0))
                    max_tilt_rad = max(max_tilt_rad, float(tilt))
                    bad_orientation = bool(tilt > 0.8)
                    dropped = bool(box_height < box_initial_height - 0.10)
                    nominal_success = (
                        stable_hold_steps >= required_hold_steps
                        and navigation_complete
                        and not bad_orientation
                        and not dropped
                    )
                    if (
                        args_cli.freeze_realized_hold_wrists
                        and realized_hold_wrist_world is None
                        and lifted
                        and bilateral_contact
                        and not bad_orientation
                        and not dropped
                    ):
                        realized_hold_wrist_world = (
                            _torch_pose_wxyz(
                                robot.data.body_link_pos_w[0, left_wrist_id],
                                robot.data.body_link_quat_w[0, left_wrist_id],
                            ),
                            _torch_pose_wxyz(
                                robot.data.body_link_pos_w[0, right_wrist_id],
                                robot.data.body_link_quat_w[0, right_wrist_id],
                            ),
                        )
                        realized_hold_lock_step = step
                    recorder.append(
                        TeacherDemoStep(
                            student_observation=pre_step_observation,
                            teacher_high_level_action=high_level_action,
                            processed_43d_joint_target=absolute_target[0],
                            phase_id=phase_id,
                            box_pose=box.data.root_link_state_w[0, :7],
                            bin_pose=bin_pose,
                            root_pose=robot.data.root_link_state_w[0, :7],
                            box_size=box_size,
                            left_contact_count=left_count,
                            right_contact_count=right_count,
                            left_wrist_pose=_body_pose_wxyz(robot, left_wrist_id),
                            right_wrist_pose=_body_pose_wxyz(robot, right_wrist_id),
                            left_fingertip_positions=left_fingertips,
                            right_fingertip_positions=right_fingertips,
                            lift=lifted,
                            drop=dropped,
                            success=nominal_success,
                        )
                    )
                    if (
                        nominal_success
                        and snapshot_store is not None
                        and accepted_snapshot is None
                    ):
                        accepted_snapshot = capture_physical_snapshot(
                            env,
                            kind=SnapshotKind.BILATERAL_HOLD,
                            env_id=0,
                            source_episode=recorder.active_episode
                            or f"attempt_{trial}",
                            source_step=step,
                            student_observation=_phase_student_observation(
                                env, observation, grasp_lift=True
                            ),
                            box_initial_height_m=box_initial_height,
                            left_contact_count=left_count,
                            right_contact_count=right_count,
                            stable_hold_steps=stable_hold_steps,
                            at_staging=False,
                            bad_orientation=bad_orientation,
                            box_dropped=dropped,
                            box_size=box_size,
                            box_mass_kg=float(args_cli.box_mass),
                        )
                    if bad_orientation:
                        termination_reason = "bad_orientation"
                        break
                    if dropped:
                        termination_reason = "box_drop"
                        break
                    if nominal_success:
                        termination_reason = "nominal_lift_hold_success"
                        break
                    if navigating and not navigation_complete and navigation_steps >= navigation_budget:
                        termination_reason = "navigation_timeout"
                        break
                    if not navigating and step - navigation_steps + 1 >= trajectory.freeze_index + required_hold_steps + grace_steps:
                        break

                strict_filter_pass = episode_passes_bimanual_lift_hold(
                    lift=np.asarray(lift_history),
                    left_contact=np.asarray(left_contact_history),
                    right_contact=np.asarray(right_contact_history),
                    required_hold_steps=required_hold_steps,
                ) and navigation_complete and not bad_orientation and not dropped
                if strict_filter_pass != nominal_success:
                    raise RuntimeError(
                        "Online success state disagrees with the strict dataset filter"
                    )
                episode_metadata = {
                    "navigation_required": bool(args_cli.randomize_work_area),
                    "navigation_completed": navigation_complete,
                    "navigation_steps": navigation_steps,
                    "navigation_end_root_pose_wxyz": navigation_end_root_pose,
                    "initial_table_xy_m": table_xy.tolist(),
                    "initial_box_table_relative_xy_m": (box_initial_position[:2] - table_xy).tolist(),
                    "initial_root_xy_m": initial_root_xy.tolist(),
                    "strict_filter_pass": strict_filter_pass,
                    "initial_box_xy_m": box_initial_position[:2].tolist(),
                    "initial_box_z_m": float(box_initial_position[2]),
                    "initial_box_yaw_rad": initial_yaw,
                }
                if args_cli.success_only and not strict_filter_pass:
                    recorder.discard_episode(
                        reason=termination_reason,
                        metadata=episode_metadata,
                    )
                else:
                    recorder.end_episode(
                        success=strict_filter_pass,
                        termination_reason=termination_reason,
                        metadata=episode_metadata,
                    )
                    recorded_demos += 1
                    if snapshot_store is not None:
                        if accepted_snapshot is None:
                            raise RuntimeError(
                                "Accepted demo is missing its live bilateral-hold snapshot"
                            )
                        snapshot_store.append(accepted_snapshot)
                        snapshots_recorded += 1
                grasp_lift_successes += int(grasp_lift_achieved)
                hold_successes += int(strict_filter_pass)
                bad_orientations += int(bad_orientation)
                trial_results.append(
                    {
                        "trial_index": trial,
                        **episode_metadata,
                        "seed": int(args_cli.seed),
                        "success": strict_filter_pass,
                        "published": not args_cli.success_only or strict_filter_pass,
                        "initial_box_xy_m": box_initial_position[:2].tolist(),
                        "initial_box_z_m": float(box_initial_position[2]),
                        "initial_box_yaw_rad": initial_yaw,
                        "bimanual_grasp_lift_achieved": grasp_lift_achieved,
                        "termination_reason": termination_reason,
                        "num_steps": step + 1,
                        "max_lift_m": max_lift_m,
                        "max_tilt_rad": max_tilt_rad,
                        "max_left_contact_count": max_left_contacts,
                        "max_right_contact_count": max_right_contacts,
                        "min_left_fingertip_box_distance_m": min_left_tip_distance_m,
                        "min_right_fingertip_box_distance_m": min_right_tip_distance_m,
                        "max_continuous_bimanual_lift_hold_steps": max_stable_hold_steps,
                        "realized_hold_lock_step": realized_hold_lock_step,
                        "final_root_pose_wxyz": robot.data.root_link_state_w[0, :7].detach().cpu().tolist(),
                        "final_left_wrist_pose_wxyz": _body_pose_wxyz(robot, left_wrist_id).detach().cpu().tolist(),
                        "final_right_wrist_pose_wxyz": _body_pose_wxyz(robot, right_wrist_id).detach().cpu().tolist(),
                        "final_high_level_action": high_level_action.tolist(),
                    }
                )
                print(
                    f"[attempt {attempts}/{max_attempts}] accepted={hold_successes} "
                    f"success={strict_filter_pass} "
                    f"bad_orientation={bad_orientation} hold_steps={stable_hold_steps}/{required_hold_steps} "
                    f"reason={termination_reason}",
                    flush=True,
                )
        if recorded_demos > 0:
            recorder.finalize()
            output_published = True
        else:
            recorder.abort()
        if snapshot_store is not None:
            if snapshots_recorded > 0:
                snapshot_store.finalize()
                snapshots_published = True
            else:
                snapshot_store.abort()
    except BaseException:
        if recorder is not None:
            recorder.abort()
        if snapshot_store is not None:
            snapshot_store.abort()
        raise
    finally:
        env.close()

    denominator = max(1, attempts)
    attempt_success_rate = hold_successes / denominator
    randomized_success_gate = attempt_success_rate >= args_cli.minimum_success_rate
    target_reached = target_successes == 0 or hold_successes >= target_successes
    result = {
        "attempts": attempts,
        "accepted_demonstrations": hold_successes,
        "recorded_demonstrations": recorded_demos,
        "rejected_demonstrations": attempts - hold_successes,
        "target_successes": target_successes,
        "max_attempts": max_attempts,
        "success_only": bool(args_cli.success_only),
        "randomization_mode": metadata["randomization_mode"],
        "navigation_required": bool(args_cli.randomize_work_area),
        "fixed_scene_parameters": fixed_scene.as_dict(),
        "box_xy_range_m": xy_randomization.as_event_range(),
        "grasp_lift_success_rate": grasp_lift_successes / denominator,
        "attempt_success_rate": attempt_success_rate,
        "two_second_hold_success_rate": attempt_success_rate,
        "bad_orientation_rate": bad_orientations / denominator,
        "trial_results": trial_results,
        "gates": {
            "random_position_teacher_at_least_minimum_success_rate": randomized_success_gate,
            "minimum_success_rate": float(args_cli.minimum_success_rate),
            "target_accepted_demonstrations_reached": target_reached,
            "all_published_demos_pass_strict_filter": bool(
                not args_cli.success_only or recorded_demos == hold_successes
            ),
            "nominal_grasp_lift_at_least_90_percent": grasp_lift_successes / denominator
            >= 0.90,
            "bad_orientation_at_most_5_percent": bad_orientations / denominator <= 0.05,
            "nominal_hold_at_least_90_percent": attempt_success_rate >= 0.90,
        },
        "output": str(output_path),
        "output_published": output_published,
        "snapshot_output": (
            str(Path(args_cli.snapshot_output).expanduser().resolve())
            if args_cli.snapshot_output is not None
            else None
        ),
        "snapshots_recorded": snapshots_recorded,
        "snapshots_published": snapshots_published,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args_cli.randomize_box_xy and not randomized_success_gate:
        raise RuntimeError(
            f"Random-position teacher success {attempt_success_rate:.1%} is below "
            f"the required {args_cli.minimum_success_rate:.1%}"
        )
    if not target_reached:
        raise RuntimeError(
            f"Collected {hold_successes}/{target_successes} accepted demos in {attempts} attempts"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
