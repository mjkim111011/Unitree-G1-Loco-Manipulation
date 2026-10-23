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
    "--wrist_z_offset_m",
    type=float,
    default=0.0,
    help="Constant world-up offset on retargeted wrist targets (tabletop fingertip clearance).",
)
parser.add_argument(
    "--wrist_orientation_source",
    choices=("command", "observed"),
    default="command",
    help="Arena wrist orientation used for retargeting: recorded command or realized link pose.",
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
parser.add_argument(
    "--support_regrasp",
    action="store_true",
    help=(
        "After the lift, curl both middle fingers, briefly relax the side squeeze so the Box "
        "settles onto them, then re-squeeze; the strict hold is counted only afterwards"
    ),
)
parser.add_argument(
    "--support_middle_curl",
    type=float,
    nargs=2,
    default=(1.30, 1.70),
    help="Left-hand middle_0/middle_1 flexion magnitudes (rad); the right hand is mirrored",
)
parser.add_argument("--support_release_m", type=float, default=0.006)
parser.add_argument("--support_drop_m", type=float, default=0.0)
parser.add_argument("--support_phase_steps", type=int, default=50)
parser.add_argument(
    "--support_wrist_pitch_rad",
    type=float,
    default=0.0,
    help=(
        "Rotate both wrist targets about the pelvis y-axis (palm normal) before curling, so the "
        "middle fingers point down along the side faces and curl under the Box"
    ),
)
parser.add_argument(
    "--hook_grasp",
    action="store_true",
    help=(
        "Prototype far-edge hook grasp: from the pre-close pose, slide open hands forward along the "
        "side faces, squeeze, tilt the Box about its robot-side bottom edge, then curl index/middle "
        "around the far face and swing both thumbs under the far bottom edge"
    ),
)
parser.add_argument(
    "--hug_after_hold",
    action="store_true",
    help=(
        "After the strict side-squeeze hold succeeds, pull the lifted Box toward the body at waist "
        "height (keeping the realized hand-Box pose), tilt it toward the torso and hold again"
    ),
)
parser.add_argument("--hug_pos", type=float, nargs=2, default=(0.26, 0.10), help="Box centre x,z in pelvis frame")
parser.add_argument("--hug_tilt_deg", type=float, default=15.0)
parser.add_argument("--hug_move_steps", type=int, default=150)
parser.add_argument("--hug_hold_steps", type=int, default=150)
parser.add_argument(
    "--regrasp_hug",
    action="store_true",
    help=(
        "After the strict hold: set the Box down at the robot-side table edge, release, step in, "
        "re-grasp the far half of the side faces with bent elbows, then lift and lean it on the torso"
    ),
)
parser.add_argument("--regrasp_place_margin_m", type=float, default=0.015)
parser.add_argument("--regrasp_place_dz_m", type=float, default=-0.015, help="Placement target z offset from resting height (compensates the hold lift margin)")
parser.add_argument("--regrasp_target_dist_m", type=float, default=0.27)
parser.add_argument("--regrasp_slide_m", type=float, default=0.18)
parser.add_argument("--regrasp_reach_dz_m", type=float, default=0.0, help="Vertical offset of the re-grasp wrist targets (compensates the hold lift margin)")
parser.add_argument("--regrasp_open_m", type=float, default=0.04)
parser.add_argument("--regrasp_squeeze_m", type=float, default=0.005)
parser.add_argument("--regrasp_hug_pose", type=float, nargs=3, default=(0.22, 0.15, 12.0), help="Box centre x,z [m] in pelvis frame and tilt toward torso [deg]")
parser.add_argument(
    "--flat_palm_grip",
    action="store_true",
    help="L-shaped grip: palm and straight index/middle flat on the side faces, thumb tip on the near face (FLAT_PALM_* constants)",
)
parser.add_argument("--flat_palm_blend_steps", type=int, default=60, help="Steps before close to move from the demo wrists to the outside of the grip")
parser.add_argument("--flat_palm_out_m", type=float, default=0.04, help="Outward clearance of the palms before closing")
parser.add_argument("--flat_palm_squeeze_m", type=float, default=0.015, help="Inward wrist push after contact (normal force with straight fingers)")
parser.add_argument(
    "--flat_palm_stand_dist",
    type=float,
    default=0.0,
    help="If >0, step in to this Box-centre distance before closing and hold it until lift (the demo stance leaves the palms out of reach)",
)
parser.add_argument("--flat_palm_feedback", type=float, default=0.0, help="Integral gain correcting wrist sag before the palms touch")
parser.add_argument(
    "--flat_palm_press_curl",
    type=float,
    nargs=2,
    default=None,
    help="After the palms reach the faces, curl index/middle (proximal, distal rad) so the fingertips press the side faces",
)
parser.add_argument("--flat_palm_dx", type=float, default=0.0, help="Shift the grip along the Box x axis (negative = toward the robot, eases the thumb push)")
parser.add_argument("--flat_palm_close_steps", type=int, default=40, help="Steps after close to move in and shape the hand")
parser.add_argument("--proto_self_collision", action="store_true", help="Enable robot self-collisions (arms vs torso) for prototypes")
parser.add_argument("--pull_test", action="store_true", help="Prototype: step in, hook the far face with both hands (palms on the sides), drag the Box toward the robot while stepping back")
parser.add_argument("--pull_box_dist", type=float, default=0.33, help="Box centre distance ahead of the pelvis to keep")
parser.add_argument("--pull_slide_m", type=float, default=0.18)
parser.add_argument("--pull_distance_m", type=float, default=0.20)
parser.add_argument("--pull_base_gain", type=float, default=2.0)
parser.add_argument("--pull_open_lateral", type=float, default=0.045, help="Outward clearance of the open hook hands while walking in (pull_by_walking)")
parser.add_argument("--pull_closed_lateral", type=float, default=-0.005, help="Outward offset of the hook hands after the inward sweep (pull_by_walking)")
parser.add_argument("--pull_walkin_min_speed", type=float, default=0.06, help="Minimum walking speed command while walking in (pull_by_walking)")
parser.add_argument(
    "--pull_finger_curl",
    type=float,
    nargs=2,
    default=None,
    help="Index/middle (proximal, distal) curl in rad while hooking, instead of the teacher closed hand (pull_by_walking)",
)
parser.add_argument(
    "--pull_regrasp_three_face",
    action="store_true",
    help="After the pull, regrasp with thumb on the near face, index on the side, middle under the overhanging bottom, then keep pulling",
)
parser.add_argument("--pull2_distance_m", type=float, default=0.25, help="Walk-back distance after the three-face regrasp")
parser.add_argument("--pull2_squeeze_m", type=float, default=0.005, help="Inward wrist squeeze after the three-face regrasp")
parser.add_argument("--pull2_stand_dist", type=float, default=0.42, help="Box-center distance held while regrasping")
parser.add_argument("--pull2_dz", type=float, default=0.03, help="Raise the three-face wrist targets (sag compensation)")
parser.add_argument("--pull2_wrist_feedback", type=float, default=0.0, help="Integral gain correcting wrist sag from the measured wrist position (free-space stages only)")
parser.add_argument("--pull2_insert_dx", type=float, default=0.0, help="Stop the insertion this far (box x) from the planned pose")
parser.add_argument("--pull2_insert_steps", type=int, default=100)
parser.add_argument("--pull2_lift_m", type=float, default=0.03, help="Wrist lift at the end of the three-face pull")
parser.add_argument("--pull_hand_dz", type=float, default=0.0, help="Raise the hook hand targets above the Box center (pull_by_walking)")
parser.add_argument(
    "--pull_by_walking",
    action="store_true",
    help="Extend the arms to the final pelvis-frame hook pose far from the Box, then walk in, close and walk back (arms fixed)",
)
parser.add_argument("--push_test", action="store_true", help="Prototype: walk to the table front, push the Box sideways (robot's right) with the left palm")
parser.add_argument("--push_distance_m", type=float, default=0.20)
parser.add_argument(
    "--hold_base_walk",
    action="store_true",
    help="During manipulation keep |velocity command| >= 0.06 m/s (walk policy) with P-control on the table-relative stance and heading",
)
parser.add_argument("--push_stance", type=float, nargs=2, default=(0.40, 0.0), help="Table centre x,y in the pelvis frame before pushing")
parser.add_argument("--push_hand_x", type=float, default=-0.15, help="Left wrist x offset in the Box frame while pushing")
parser.add_argument("--squat_test", action="store_true", help="Prototype: retract arms, walk to a table-relative stance, squat and lean, then hold")
parser.add_argument("--squat_height", type=float, default=0.60)
parser.add_argument("--squat_pitch", type=float, default=0.35)
parser.add_argument("--squat_stance", type=float, nargs=2, default=(0.34, 0.15), help="Table centre x,y in the pelvis frame at the stance")
parser.add_argument("--hook_slide_m", type=float, default=0.18)
parser.add_argument("--hook_target_dist_m", type=float, default=0.36)
parser.add_argument("--hook_torso_pitch_rad", type=float, default=0.0)
parser.add_argument(
    "--hook_hug",
    action="store_true",
    help="After the tilt, lift slightly, pull the Box toward the robot and hold it against the body (no finger curl)",
)
parser.add_argument("--hook_pull_m", type=float, default=0.12)
parser.add_argument("--hook_lift_m", type=float, default=0.03)
parser.add_argument("--hook_hug_pos", type=float, nargs=2, default=(0.24, 0.10), help="Box centre x,z in the pelvis frame")
parser.add_argument("--hook_hug_tilt_deg", type=float, default=20.0)
parser.add_argument("--hook_camera_dir", default=None, help="Save D435 RGB frames at hook milestones (needs --enable_cameras)")
parser.add_argument(
    "--hook_hold_base",
    action="store_true",
    help="Keep a proportional base-distance command after step-in instead of a zero (stand) command",
)
parser.add_argument("--hook_stepin_steps", type=int, default=150)
parser.add_argument("--hook_lower_m", type=float, default=0.02)
parser.add_argument("--hook_open_m", type=float, default=0.03)
parser.add_argument("--hook_squeeze_m", type=float, default=0.005)
parser.add_argument("--hook_tilt_deg", type=float, default=25.0)
parser.add_argument("--hook_phase_steps", type=int, default=50)
parser.add_argument("--hook_hold_steps", type=int, default=150)
parser.add_argument("--hook_finger_curl", type=float, nargs=2, default=(0.9, 1.3))
parser.add_argument(
    "--video_output",
    default=None,
    help="Optional MP4 path for a headless recording of every control step; requires --enable_cameras",
)
parser.add_argument("--video_eye", type=float, nargs=3, default=(1.35, -0.45, 1.30))
parser.add_argument("--video_lookat", type=float, nargs=3, default=(0.55, 0.0, 0.85))
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
if not 0.0 <= args_cli.wrist_z_offset_m <= 0.10:
    parser.error("--wrist_z_offset_m must be in [0, 0.10]")
if args_cli.support_regrasp and (
    args_cli.support_phase_steps <= 0
    or not 0.0 <= args_cli.support_release_m <= 0.03
    or not 0.0 <= args_cli.support_drop_m <= 0.06
    or not 0.0 <= args_cli.support_middle_curl[0] <= 1.571
    or not 0.0 <= args_cli.support_middle_curl[1] <= 1.745
):
    parser.error("invalid --support_* parameters")
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
    if args_cli.proto_self_collision:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.scene.box.spawn.size = (size, size, size)
    cfg.scene.box.spawn.mass_props.mass = float(args_cli.box_mass)
    if args_cli.video_output is not None:
        cfg.viewer.eye = tuple(args_cli.video_eye)
        cfg.viewer.lookat = tuple(args_cli.video_lookat)
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.resolution = (960, 540)
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
    if args_cli.hook_camera_dir is not None:
        cfg.scene.front_camera.width = 320
        cfg.scene.front_camera.height = 240
    else:
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


# Wrist poses in the Box frame measured from a successful strict side-squeeze hold
# (BC-2 rollout snapshot); the hook prototype slides these along the side faces.
HOOK_BOX_WRIST = {
    "left": np.array(
        [[0.9283, -0.1272, 0.3494, -0.2002],
         [0.1604, 0.9847, -0.0675, 0.1218],
         [-0.3355, 0.1187, 0.9345, 0.0043],
         [0.0, 0.0, 0.0, 1.0]]
    ),
    "right": np.array(
        [[0.9489, 0.1096, 0.2959, -0.2076],
         [-0.1421, 0.9857, 0.0906, -0.1303],
         [-0.2817, -0.128, 0.9509, 0.0024],
         [0.0, 0.0, 0.0, 1.0]]
    ),
}
# Offline FK (pinocchio, URDF limits): thumbs reach under the far bottom edge.
HOOK_THUMB_TARGETS = {
    "left_hand_thumb_0_joint": 1.05,
    "left_hand_thumb_1_joint": 1.05,
    "left_hand_thumb_2_joint": 0.0,
    "right_hand_thumb_0_joint": 1.05,
    "right_hand_thumb_1_joint": -0.84,
    "right_hand_thumb_2_joint": -0.22,
}


# Offline optimisation (URDF collision meshes, tmp flat_palm.py): palm and straight index/middle
# flat on the side face (palm centre 3 cm below the Box centre, near half), thumb tip on the near face.
# Wrist poses in the Box frame; the left grip mirrors the right one.
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
# Offline optimisation (pinocchio, URDF limits, tmp three_face.py): thumb on the near face,
# index straight along the side face, middle under the bottom 3-4 cm from the near edge.
THREE_FACE_BOX_WRIST = {
    "left": np.array(
        [[0.8916, -0.2382, 0.3852, -0.2253],
         [0.0, 0.8505, 0.5259, 0.0891],
         [-0.4529, -0.4689, 0.7583, -0.0185],
         [0.0, 0.0, 0.0, 1.0]]
    ),
    "right": np.array(
        [[0.881, 0.2745, 0.3853, -0.2258],
         [0.0, 0.8144, -0.5803, -0.0877],
         [-0.4731, 0.5112, 0.7175, -0.0086],
         [0.0, 0.0, 0.0, 1.0]]
    ),
}
THREE_FACE_HAND = {
    "left_hand_thumb_0_joint": 0.18,
    "left_hand_thumb_1_joint": 0.606,
    "left_hand_thumb_2_joint": 1.73,
    "left_hand_index_0_joint": 0.0,
    "left_hand_index_1_joint": 0.0,
    "left_hand_middle_0_joint": -0.469,
    "left_hand_middle_1_joint": -1.74,
    "right_hand_thumb_0_joint": 0.26,
    "right_hand_thumb_1_joint": -0.818,
    "right_hand_thumb_2_joint": -1.333,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 0.175,
    "right_hand_middle_1_joint": 0.448,
}


def _smooth(fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _hook_box_pose(box_initial_pose: np.ndarray, tilt_rad: float) -> np.ndarray:
    """Box pose tilted about its robot-side bottom edge (far edge rising)."""

    half = 0.5 * float(args_cli.box_size)
    pivot = np.eye(4)
    pivot[:3, 3] = (-half, 0.0, -half)
    tilt = np.eye(4)
    tilt[:3, :3] = Rotation.from_euler("y", -tilt_rad).as_matrix()
    return box_initial_pose @ pivot @ tilt @ np.linalg.inv(pivot)


def _hook_wrist_world(
    box_initial_pose: np.ndarray, side: str, tilt_rad: float, lateral_open_m: float
) -> np.ndarray:
    offset = np.eye(4)
    outward = 1.0 if side == "left" else -1.0
    offset[:3, 3] = (
        float(args_cli.hook_slide_m),
        outward * lateral_open_m,
        -float(args_cli.hook_lower_m),
    )
    return _hook_box_pose(box_initial_pose, tilt_rad) @ offset @ HOOK_BOX_WRIST[side]


def _interp_pose(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    from scipy.spatial.transform import Slerp

    out = np.eye(4)
    out[:3, 3] = (1.0 - fraction) * a[:3, 3] + fraction * b[:3, 3]
    rotations = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    out[:3, :3] = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
    return out


def _hook_finger_targets(
    env, absolute_target: torch.Tensor, curl_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    action_term = env.action_manager.get_term("JointPositionAction")
    action_names = list(action_term._joint_names)
    target = absolute_target.clone()
    proximal, distal = (float(v) for v in args_cli.hook_finger_curl)
    goals = dict(HOOK_THUMB_TARGETS)
    for side, sign in (("left", -1.0), ("right", 1.0)):
        for finger in ("index", "middle"):
            goals[f"{side}_hand_{finger}_0_joint"] = sign * proximal
            goals[f"{side}_hand_{finger}_1_joint"] = sign * distal
    for name, goal in goals.items():
        index = action_names.index(name)
        target[0, index] = (1.0 - curl_fraction) * target[0, index] + curl_fraction * goal
    raw_action = (target - action_term._offset) / action_term._scale
    return raw_action, target


REGRASP_PHASES = (
    ("place", 150),
    ("settle", 75),
    ("open", 50),
    ("retract", 50),
    ("stepin", 200),
    ("reach", 100),
    ("squeeze", 50),
    ("hug", 150),
    ("hold", 200),
)


def _regrasp_phase(elapsed: int) -> tuple[str, int, int]:
    start = 0
    for name, length in REGRASP_PHASES:
        if elapsed < start + length:
            return name, elapsed - start, length
        start += length
    return "done", elapsed - start, 1


def _flat_box_pose(box_pose: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    yaw = Rotation.from_matrix(box_pose[:3, :3]).as_euler("zyx")[0]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    out[:3, 3] = position
    return out


SUPPORT_MIDDLE_JOINT_SIGNS = {
    "left_hand_middle_0_joint": (-1.0, 0),
    "left_hand_middle_1_joint": (-1.0, 1),
    "right_hand_middle_0_joint": (1.0, 0),
    "right_hand_middle_1_joint": (1.0, 1),
}


def _support_regrasp_targets(
    env, absolute_target: torch.Tensor, curl_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend both middle fingers toward the bottom-support flexion."""

    action_term = env.action_manager.get_term("JointPositionAction")
    action_names = list(action_term._joint_names)
    target = absolute_target.clone()
    for name, (sign, curl_index) in SUPPORT_MIDDLE_JOINT_SIGNS.items():
        index = action_names.index(name)
        support = sign * float(args_cli.support_middle_curl[curl_index])
        target[0, index] = (1.0 - curl_fraction) * target[0, index] + curl_fraction * support
    raw_action = (target - action_term._offset) / action_term._scale
    return raw_action, target


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
        wrist_z_offset_m=args_cli.wrist_z_offset_m,
        wrist_orientation_source=args_cli.wrist_orientation_source,
    )
    cfg = _configure_nominal_environment()
    print("[teacher] creating the current Box environment", flush=True)
    env = gym.make(
        args_cli.task,
        cfg=cfg,
        render_mode="rgb_array" if args_cli.video_output is not None else None,
    ).unwrapped
    video_writer = None
    video_frame_count = 0
    if args_cli.video_output is not None:
        import imageio

        video_path = Path(args_cli.video_output).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            video_path,
            fps=int(round(1.0 / env.step_dt)),
            codec="libx264",
            quality=7,
            macro_block_size=8,
        )
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
        "wrist_z_offset_m": trajectory.wrist_z_offset_m,
        "wrist_orientation_source": trajectory.wrist_orientation_source,
        "support_regrasp": bool(args_cli.support_regrasp),
        "support_middle_curl": list(args_cli.support_middle_curl),
        "support_release_m": float(args_cli.support_release_m),
        "support_drop_m": float(args_cli.support_drop_m),
        "support_phase_steps": int(args_cli.support_phase_steps),
        "support_wrist_pitch_rad": float(args_cli.support_wrist_pitch_rad),
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
                support_extra_steps = 3 * int(args_cli.support_phase_steps) if args_cli.support_regrasp else 0
                if args_cli.regrasp_hug:
                    support_extra_steps += sum(length for _, length in REGRASP_PHASES)
                if args_cli.squat_test:
                    support_extra_steps += 50 + 300 + 100 + 200
                if args_cli.pull_test:
                    support_extra_steps += 50 + 300 + 100 + 50 + 250 + 100
                    if args_cli.pull_regrasp_three_face:
                        support_extra_steps += 1300
                if args_cli.push_test:
                    support_extra_steps += 50 + 300 + 100 + 50 + 200 + 50 + 100
                if args_cli.hug_after_hold:
                    support_extra_steps += int(args_cli.hug_move_steps) + int(args_cli.hug_hold_steps)
                if args_cli.hook_grasp:
                    support_extra_steps += 7 * int(args_cli.hook_phase_steps) + int(args_cli.hook_hold_steps) + int(args_cli.hook_stepin_steps)
                    if args_cli.hook_hug:
                        support_extra_steps += 5 * int(args_cli.hook_phase_steps)
                max_steps = trajectory.freeze_index + required_hold_steps + grace_steps + navigation_budget + support_extra_steps
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
                support_start_step: int | None = None
                hook_start_step: int | None = None
                hook_start_wrist_world: dict[str, np.ndarray] = {}
                hook_start_wrist_pelvis: dict[str, np.ndarray] = {}
                hook_hug_anchor: np.ndarray | None = None
                hug_start_step: int | None = None
                rg_start: int | None = None
                sq_start: int | None = None
                pl_start: int | None = None
                pl: dict = {}
                flat_palm_corr = {"left": np.zeros(3), "right": np.zeros(3)}
                flat_palm_corr_rot = {"left": np.zeros(3), "right": np.zeros(3)}
                pu_start: int | None = None
                pu: dict = {}
                sq: dict = {}
                rg: dict = {}
                hug_anchor_box: np.ndarray | None = None
                hug_wrist_in_box: dict[str, np.ndarray] = {}
                hook_log_every = 25
                support_total_steps = 3 * int(args_cli.support_phase_steps)

                for step in range(max_steps):
                    navigating = not navigation_complete
                    source_index = 0 if navigating else min(step - navigation_steps, trajectory.freeze_index)
                    if (args_cli.hook_grasp or args_cli.squat_test or args_cli.push_test or args_cli.pull_test) and not navigating:
                        source_index = min(source_index, trajectory.close_index)
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
                    flat_palm_fraction = 0.0
                    flat_palm_command = None
                    if args_cli.flat_palm_grip and not navigating and realized_hold_wrist_world is None:
                        blend_start = trajectory.close_index - int(args_cli.flat_palm_blend_steps)
                        if source_index >= blend_start:
                            i = source_index
                            # Box pose the demo expects at this index (TRANSLATE retarget + hold lift margin)
                            # position from the demo, orientation from this scene's Box (the Arena source Box frame is rotated)
                            virtual_box = trajectory.box_world[i].copy()
                            virtual_box[:3, :3] = box_initial_pose[:3, :3]
                            virtual_box[:3, 3] += 0.5 * (
                                retargeted.left_wrist_world[i][:3, 3] - trajectory.left_target_world[i][:3, 3]
                                + retargeted.right_wrist_world[i][:3, 3] - trajectory.right_target_world[i][:3, 3]
                            )
                            lift_steps = max(1, trajectory.freeze_index - trajectory.lift_index)
                            virtual_box[2, 3] += trajectory.hold_lift_margin_m * float(
                                np.clip((i - trajectory.lift_index) / lift_steps, 0.0, 1.0)
                            )
                            closing = float(np.clip((i - trajectory.close_index + 1) / int(args_cli.flat_palm_close_steps), 0.0, 1.0))
                            lateral = (1.0 - _smooth(closing)) * float(args_cli.flat_palm_out_m) - _smooth(closing) * float(args_cli.flat_palm_squeeze_m)
                            approach = _smooth((i - blend_start + 1) / int(args_cli.flat_palm_blend_steps))
                            # Hold the L shape from the start of the approach so the fingers never move during contact;
                            # the Arena hand stays open (all zeros, itself L-shaped) so its fixed curled pose never kicks in.
                            flat_palm_fraction = approach if i < trajectory.close_index else 1.0
                            high_level_action[0:2] = 0.0
                            inverse_pelvis = np.linalg.inv(pelvis_pose)
                            if args_cli.flat_palm_stand_dist > 0.0 and i < trajectory.freeze_index:
                                live_box = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                                box_in_pelvis = inverse_pelvis @ live_box
                                ex = float(box_in_pelvis[0, 3]) - float(args_cli.flat_palm_stand_dist)
                                ey = float(box_in_pelvis[1, 3])
                                flat_palm_command = np.zeros(3)
                                if ex > 0.01:
                                    flat_palm_command[0] = float(np.clip(3.0 * ex, 0.13, 0.25))
                                    flat_palm_command[1] = float(np.clip(1.5 * ey, -0.08, 0.08))
                            for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
                                outward = 1.0 if side == "left" else -1.0
                                offset = np.eye(4); offset[0, 3] = float(args_cli.flat_palm_dx); offset[1, 3] = outward * lateral
                                grip = virtual_box @ offset @ FLAT_PALM_BOX_WRIST[side]
                                if i < trajectory.close_index:
                                    demo = np.eye(4)
                                    demo[:3, 3] = high_level_action[position_slice]
                                    demo[:3, :3] = Rotation.from_quat(high_level_action[quaternion_slice]).as_matrix()
                                    grip = _interp_pose(pelvis_pose @ demo, grip, approach)
                                if args_cli.flat_palm_feedback > 0.0:
                                    wrist_id = left_wrist_id if side == "left" else right_wrist_id
                                    if i < trajectory.close_index:
                                        actual_wrist = _torch_pose_wxyz(robot.data.body_link_pos_w[0, wrist_id], robot.data.body_link_quat_w[0, wrist_id])
                                        flat_palm_corr[side] = np.clip(
                                            flat_palm_corr[side] + float(args_cli.flat_palm_feedback) * (grip[:3, 3] - actual_wrist[:3, 3]), -0.08, 0.08
                                        )
                                        rot_error = Rotation.from_matrix(grip[:3, :3] @ actual_wrist[:3, :3].T).as_rotvec()
                                        flat_palm_corr_rot[side] = np.clip(flat_palm_corr_rot[side] + float(args_cli.flat_palm_feedback) * rot_error, -0.5, 0.5)
                                    grip = grip.copy(); grip[:3, 3] += flat_palm_corr[side]
                                    grip[:3, :3] = Rotation.from_rotvec(flat_palm_corr_rot[side]).as_matrix() @ grip[:3, :3]
                                high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(inverse_pelvis @ grip)
                                if (i - blend_start) % 20 == 0 and side == "right":
                                    live_box = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                                    actual = _torch_pose_wxyz(robot.data.body_link_pos_w[0, right_wrist_id], robot.data.body_link_quat_w[0, right_wrist_id])
                                    print(
                                        f"[flat] i={i} close={trajectory.close_index} lift={trajectory.lift_index} vbox-live={np.round((virtual_box[:3,3]-live_box[:3,3])*100,1)}cm "
                                        f"R target_in_box={np.round((np.linalg.inv(live_box)@grip)[:3,3]*100,1)} actual_in_box={np.round((np.linalg.inv(live_box)@actual)[:3,3]*100,1)} "
                                        f"rot_err={np.degrees(Rotation.from_matrix(grip[:3,:3].T@actual[:3,:3]).magnitude()):.0f}deg box_ahead={(np.linalg.inv(pelvis_pose)@live_box)[0,3]:.3f}",
                                        flush=True,
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
                    if flat_palm_command is not None:
                        high_level_action[16:19] = flat_palm_command
                    if args_cli.hook_grasp and not navigating and source_index >= trajectory.close_index:
                        if hook_start_step is None:
                            hook_start_step = step
                            for side, position_slice, quaternion_slice in (
                                ("left", slice(2, 5), slice(5, 9)),
                                ("right", slice(9, 12), slice(12, 16)),
                            ):
                                start_pelvis = np.eye(4)
                                start_pelvis[:3, 3] = high_level_action[position_slice]
                                start_pelvis[:3, :3] = Rotation.from_quat(
                                    high_level_action[quaternion_slice]
                                ).as_matrix()
                                hook_start_wrist_pelvis[side] = start_pelvis
                                hook_start_wrist_world[side] = pelvis_pose @ start_pelvis
                        stepin_steps = int(args_cli.hook_stepin_steps)
                        if step - hook_start_step < stepin_steps:
                            # Step toward the table with the arms held in the pelvis frame.
                            box_in_pelvis = np.linalg.inv(pelvis_pose) @ box_initial_pose
                            distance_error = float(box_in_pelvis[0, 3]) - float(args_cli.hook_target_dist_m)
                            for side, position_slice, quaternion_slice in (
                                ("left", slice(2, 5), slice(5, 9)),
                                ("right", slice(9, 12), slice(12, 16)),
                            ):
                                retracted = hook_start_wrist_pelvis[side].copy()
                                retracted[:3, 3] = (0.22, 0.22 if side == "left" else -0.22, 0.10)
                                widened = _interp_pose(
                                    hook_start_wrist_pelvis[side],
                                    retracted,
                                    _smooth((step - hook_start_step + 1) / 40.0),
                                )
                                high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(widened)
                                hook_start_wrist_world[side] = pelvis_pose @ widened
                            high_level_action[0:2] = 0.0
                            walking = distance_error > 0.01 and step - hook_start_step >= 40
                            high_level_action[16] = float(np.clip(1.5 * distance_error, 0.0, 0.25)) if walking else 0.0
                            high_level_action[17] = float(np.clip(-1.5 * float(box_in_pelvis[1, 3]), -0.1, 0.1)) if walking else 0.0
                            high_level_action[18] = 0.0
                        hook_elapsed = step - hook_start_step - stepin_steps
                        hook_p = int(args_cli.hook_phase_steps)
                        tilt_goal = math.radians(float(args_cli.hook_tilt_deg))
                    if args_cli.hook_grasp and hook_start_step is not None and step - hook_start_step >= int(args_cli.hook_stepin_steps):
                        hook_elapsed = step - hook_start_step - int(args_cli.hook_stepin_steps)
                        hook_p = int(args_cli.hook_phase_steps)
                        tilt_goal = math.radians(float(args_cli.hook_tilt_deg))
                        place_fraction = _smooth((hook_elapsed + 1) / (2 * hook_p))
                        squeeze_fraction = _smooth((hook_elapsed - 2 * hook_p + 1) / hook_p)
                        tilt_fraction = _smooth((hook_elapsed - 3 * hook_p + 1) / (3 * hook_p))
                        lateral = (1.0 - squeeze_fraction) * float(args_cli.hook_open_m) - squeeze_fraction * float(args_cli.hook_squeeze_m)
                        inverse_pelvis = np.linalg.inv(pelvis_pose)
                        desired_box = _hook_box_pose(box_initial_pose, tilt_fraction * tilt_goal)
                        if args_cli.hook_hug:
                            pull_fraction = _smooth((hook_elapsed - 6 * hook_p + 1) / (2 * hook_p))
                            hug_fraction = _smooth((hook_elapsed - 8 * hook_p + 1) / (3 * hook_p))
                            pulled = np.eye(4)
                            pulled[:3, 3] = (-float(args_cli.hook_pull_m) * pull_fraction, 0.0, float(args_cli.hook_lift_m) * pull_fraction)
                            desired_box = pulled @ desired_box
                            if hug_fraction > 0.0:
                                hug_pelvis = np.eye(4)
                                hug_pelvis[:3, :3] = Rotation.from_euler("y", -math.radians(float(args_cli.hook_hug_tilt_deg))).as_matrix()
                                hug_pelvis[:3, 3] = (float(args_cli.hook_hug_pos[0]), 0.0, float(args_cli.hook_hug_pos[1]))
                                if hook_hug_anchor is None:
                                    hook_hug_anchor = desired_box.copy()
                                desired_box = _interp_pose(hook_hug_anchor, pelvis_pose @ hug_pelvis, hug_fraction)
                        for side, position_slice, quaternion_slice in (
                            ("left", slice(2, 5), slice(5, 9)),
                            ("right", slice(9, 12), slice(12, 16)),
                        ):
                            outward = 1.0 if side == "left" else -1.0
                            hand_offset = np.eye(4)
                            hand_offset[:3, 3] = (float(args_cli.hook_slide_m), outward * lateral, -float(args_cli.hook_lower_m))
                            goal = desired_box @ hand_offset @ HOOK_BOX_WRIST[side]
                            world = _interp_pose(hook_start_wrist_world[side], goal, place_fraction) if place_fraction < 1.0 else goal
                            high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(inverse_pelvis @ world)
                        high_level_action[0:2] = 0.0
                        high_level_action[16:19] = 0.0
                        if args_cli.hook_hold_base:
                            box_in_pelvis = inverse_pelvis @ box_initial_pose
                            distance_error = float(box_in_pelvis[0, 3]) - float(args_cli.hook_target_dist_m)
                            high_level_action[16] = float(np.clip(1.5 * distance_error, -0.15, 0.20))
                        high_level_action[21] = float(args_cli.hook_torso_pitch_rad) * _smooth((hook_elapsed + 1) / (2 * hook_p))
                    if args_cli.pull_test and not navigating and source_index >= trajectory.close_index:
                        if pl_start is None:
                            pl_start = step
                            pl["heading0"] = float(Rotation.from_matrix(pelvis_pose[:3, :3]).as_euler("zyx")[0])
                            for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
                                start = np.eye(4)
                                start[:3, 3] = high_level_action[position_slice]
                                start[:3, :3] = Rotation.from_quat(high_level_action[quaternion_slice]).as_matrix()
                                retracted = start.copy()
                                retracted[:3, 3] = (0.15, 0.25 if side == "left" else -0.25, 0.18)
                                pl[side] = (start, retracted)
                        e = step - pl_start
                        box_now = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        inverse_pelvis = np.linalg.inv(pelvis_pose)
                        box_in_pelvis = inverse_pelvis @ box_now
                        high_level_action[16:19] = 0.0
                        hand_close = 0.0
                        targets = {}

                        def base_hold(min_speed: float = 0.06, dist: float | None = None, deadband: float = 0.0) -> None:
                            ex = float(box_in_pelvis[0, 3]) - float(args_cli.pull_box_dist if dist is None else dist)
                            ey = float(box_in_pelvis[1, 3])
                            if abs(ex) < deadband and abs(ey) < deadband:
                                return
                            heading = float(Rotation.from_matrix(pelvis_pose[:3, :3]).as_euler("zyx")[0])
                            yaw_error = math.atan2(math.sin(pl["heading0"] - heading), math.cos(pl["heading0"] - heading))
                            gain = float(args_cli.pull_base_gain)
                            vx = float(np.clip(gain * ex, -0.2, 0.2)); vy = float(np.clip(gain * ey, -0.12, 0.12))
                            norm = math.hypot(vx, vy)
                            if norm < min_speed:
                                vx, vy = (vx * min_speed / norm, vy * min_speed / norm) if norm > 1e-3 else (-min_speed, 0.0)
                            high_level_action[16] = vx; high_level_action[17] = vy
                            high_level_action[18] = float(np.clip(1.5 * yaw_error, -0.3, 0.3))

                        if args_cli.pull_by_walking:
                            box_z_rel = float(box_in_pelvis[2, 3])
                            final_box = np.eye(4)
                            final_box[:3, 3] = (float(args_cli.pull_box_dist), 0.0, box_z_rel + float(args_cli.pull_hand_dz))
                            if "hook_pelvis" not in pl:
                                pl["hook_pelvis"] = {}
                                for side in ("left", "right"):
                                    outward = 1.0 if side == "left" else -1.0
                                    for name, lateral in (("open", float(args_cli.pull_open_lateral)), ("closed", float(args_cli.pull_closed_lateral))):
                                        off = np.eye(4); off[:3, 3] = (float(args_cli.pull_slide_m), outward * lateral, 0.0)
                                        pl["hook_pelvis"][(side, name)] = final_box @ off @ HOOK_BOX_WRIST[side]
                            ex = float(box_in_pelvis[0, 3]) - float(args_cli.pull_box_dist)
                            ey = float(box_in_pelvis[1, 3])
                            stage = pl.get("stage", "extend")
                            if stage == "extend":
                                f = _smooth((e + 1) / 150)
                                for side in ("left", "right"):
                                    targets[side] = pelvis_pose @ _interp_pose(pl[side][0], pl["hook_pelvis"][(side, "open")], f)
                                if e >= 200:
                                    pl["stage"] = "walk_in"; pl["stage_e"] = e
                            elif stage == "walk_in":
                                for side in ("left", "right"):
                                    targets[side] = pelvis_pose @ pl["hook_pelvis"][(side, "open")]
                                if (abs(ex) < 0.012 and abs(ey) < 0.015) or e - pl["stage_e"] > 300:
                                    pl["stage"] = "close"; pl["stage_e"] = e
                                else:
                                    base_hold(float(args_cli.pull_walkin_min_speed))
                            elif stage == "close":
                                f = _smooth((e - pl["stage_e"] + 1) / 75)
                                for side in ("left", "right"):
                                    targets[side] = pelvis_pose @ _interp_pose(pl["hook_pelvis"][(side, "open")], pl["hook_pelvis"][(side, "closed")], f)
                                hand_close = f
                                if e - pl["stage_e"] >= 75:
                                    pl["stage"] = "walk_back"; pl["stage_e"] = e
                                    pl["box_start_x"] = float(box_now[0, 3])
                                    pl["pelvis_start"] = pelvis_pose.copy()
                            elif stage in ("release", "around", "preshape", "insert", "squeeze", "walk2", "lift", "hold2"):
                                k = e - pl["stage_e"]
                                if "tf_box" not in pl:
                                    pl["tf_box"] = box_now.copy()
                                    pl["tf_start"] = {side: pelvis_pose @ pl["hook_pelvis"][(side, "closed")] for side in ("left", "right")}
                                    pl["finger_goal"] = dict(THREE_FACE_HAND)
                                    pl["finger_frac"] = 0.0

                                def tf_pose(side: str, dx: float = 0.0, dy_out: float = 0.0, dz: float = 0.0) -> np.ndarray:
                                    outward = 1.0 if side == "left" else -1.0
                                    off = np.eye(4); off[:3, 3] = (dx, outward * dy_out, dz)
                                    lift = np.eye(4); lift[2, 3] = float(args_cli.pull2_dz)
                                    return lift @ pl["tf_box"] @ off @ THREE_FACE_BOX_WRIST[side]

                                def advance(next_stage: str, length: int) -> None:
                                    if k >= length:
                                        pl["stage"] = next_stage; pl["stage_e"] = e
                                        pl["tf_from"] = {side: targets[side].copy() for side in ("left", "right")}

                                hand_close = 0.0
                                if stage in ("around", "preshape", "insert", "squeeze"):
                                    base_hold(0.06, float(args_cli.pull2_stand_dist), 0.02)
                                if stage == "release":
                                    # open the hand and back the wrists out sideways
                                    # slide out 9 cm with the hand still closed (fingers clear the far corners),
                                    # then open in free space, then rise 3 cm
                                    for side in ("left", "right"):
                                        outward = 1.0 if side == "left" else -1.0
                                        out1 = pl["tf_start"][side].copy(); out1[1, 3] += outward * 0.09
                                        out2 = out1.copy(); out2[2, 3] += 0.03
                                        targets[side] = (
                                            _interp_pose(pl["tf_start"][side], out1, _smooth((k + 1) / 60))
                                            if k < 90
                                            else _interp_pose(out1, out2, _smooth((k - 89) / 30))
                                        )
                                    hand_close = 1.0 if k < 60 else 1.0 - _smooth((k - 59) / 30)
                                    advance("around", 120)
                                elif stage == "around":
                                    # swing to the outside-front of the near face, shaping the hand on the way
                                    if k == 0:
                                        pl["tf_box"] = box_now.copy()
                                    for side in ("left", "right"):
                                        targets[side] = _interp_pose(pl["tf_from"][side], tf_pose(side, -0.09, 0.12), _smooth((k + 1) / 120))
                                    advance("preshape", 120)
                                elif stage == "preshape":
                                    for side in ("left", "right"):
                                        targets[side] = _interp_pose(pl["tf_from"][side], tf_pose(side, -0.09, 0.0), _smooth((k + 1) / 80))
                                    # shape the hand only once it is clear in front of the near face
                                    pl["finger_frac"] = _smooth((k + 1) / 80)
                                    advance("insert", 80)
                                elif stage == "insert":
                                    # slide in: middle goes under the bottom, thumb meets the near face;
                                    # stop where we are as soon as the Box starts to be pushed
                                    if "insert_box" not in pl:
                                        pl["insert_box"] = box_now.copy()
                                    pushed = float(np.linalg.norm(box_now[:2, 3] - pl["insert_box"][:2, 3])) > 0.008
                                    if pushed and "insert_hold" not in pl and "last_insert" in pl:
                                        pl["insert_hold"] = {side: pl["last_insert"][side].copy() for side in ("left", "right")}
                                        print(f"[pull]   insert stopped at k={k}: Box pushed", flush=True)
                                    for side in ("left", "right"):
                                        if "insert_hold" in pl:
                                            targets[side] = pl["insert_hold"][side]
                                        else:
                                            targets[side] = _interp_pose(
                                                pl["tf_from"][side], tf_pose(side, float(args_cli.pull2_insert_dx)), _smooth((k + 1) / int(args_cli.pull2_insert_steps))
                                            )
                                    pl["last_insert"] = {side: targets[side].copy() for side in ("left", "right")}
                                    if "insert_hold" in pl and k < int(args_cli.pull2_insert_steps):
                                        k = int(args_cli.pull2_insert_steps)
                                    advance("squeeze", int(args_cli.pull2_insert_steps))
                                elif stage == "squeeze":
                                    squeeze = float(args_cli.pull2_squeeze_m)
                                    for side in ("left", "right"):
                                        outward = 1.0 if side == "left" else -1.0
                                        squeezed = pl["tf_from"][side].copy(); squeezed[:3, 3] -= outward * squeeze * pl["tf_box"][:3, 1]
                                        targets[side] = _interp_pose(pl["tf_from"][side], squeezed, _smooth((k + 1) / 40))
                                    if k >= 40:
                                        pl["tf_pelvis"] = {side: np.linalg.inv(pelvis_pose) @ targets[side] for side in ("left", "right")}
                                        pl["pelvis_start2"] = pelvis_pose.copy()
                                    advance("walk2", 40)
                                elif stage == "walk2":
                                    for side in ("left", "right"):
                                        targets[side] = pelvis_pose @ pl["tf_pelvis"][side]
                                    travelled = float(np.linalg.norm(pelvis_pose[:2, 3] - pl["pelvis_start2"][:2, 3]))
                                    if travelled < float(args_cli.pull2_distance_m) and k < 400:
                                        heading = float(Rotation.from_matrix(pelvis_pose[:3, :3]).as_euler("zyx")[0])
                                        yaw_error = math.atan2(math.sin(pl["heading0"] - heading), math.cos(pl["heading0"] - heading))
                                        high_level_action[16] = -0.10
                                        high_level_action[17] = float(np.clip(1.0 * ey, -0.08, 0.08))
                                        high_level_action[18] = float(np.clip(1.5 * yaw_error, -0.3, 0.3))
                                    else:
                                        pl["stage"] = "lift"; pl["stage_e"] = e
                                elif stage == "lift":
                                    for side in ("left", "right"):
                                        raised = pl["tf_pelvis"][side].copy(); raised[2, 3] += float(args_cli.pull2_lift_m) * _smooth((k + 1) / 100)
                                        targets[side] = pelvis_pose @ raised
                                    advance("hold2", 100)
                                else:
                                    for side in ("left", "right"):
                                        raised = pl["tf_pelvis"][side].copy(); raised[2, 3] += float(args_cli.pull2_lift_m)
                                        targets[side] = pelvis_pose @ raised
                            else:
                                for side in ("left", "right"):
                                    targets[side] = pelvis_pose @ pl["hook_pelvis"][(side, "closed")]
                                hand_close = 1.0
                                travelled = float(np.linalg.norm(pelvis_pose[:2, 3] - pl["pelvis_start"][:2, 3]))
                                if stage == "walk_back" and travelled < float(args_cli.pull_distance_m) and e - pl["stage_e"] < 400:
                                    heading = float(Rotation.from_matrix(pelvis_pose[:3, :3]).as_euler("zyx")[0])
                                    yaw_error = math.atan2(math.sin(pl["heading0"] - heading), math.cos(pl["heading0"] - heading))
                                    high_level_action[16] = -0.10
                                    high_level_action[17] = float(np.clip(1.0 * ey, -0.08, 0.08))
                                    high_level_action[18] = float(np.clip(1.5 * yaw_error, -0.3, 0.3))
                                elif stage == "walk_back" and args_cli.pull_regrasp_three_face:
                                    pl["stage"] = "settle"; pl["stage_e"] = e
                                elif stage == "settle":
                                    if e - pl["stage_e"] >= 50:
                                        pl["stage"] = "release"; pl["stage_e"] = e
                                else:
                                    pl["stage"] = "hold"
                            if args_cli.pull2_wrist_feedback > 0.0 and pl.get("stage") in ("around", "preshape", "insert", "squeeze", "walk2", "lift", "hold2"):
                                corr = pl.setdefault("wrist_corr", {"left": np.zeros(3), "right": np.zeros(3)})
                                corr_rot = pl.setdefault("wrist_corr_rot", {"left": np.zeros(3), "right": np.zeros(3)})
                                for side, wrist_id in (("left", left_wrist_id), ("right", right_wrist_id)):
                                    if pl["stage"] in ("around", "preshape", "insert"):
                                        actual = _torch_pose_wxyz(robot.data.body_link_pos_w[0, wrist_id], robot.data.body_link_quat_w[0, wrist_id])
                                        gain = float(args_cli.pull2_wrist_feedback)
                                        position_error = targets[side][:3, 3] - actual[:3, 3]
                                        if pl["stage"] == "insert":
                                            position_error[:2] = 0.0  # in contact range: correct sag only, never push forward
                                        corr[side] = np.clip(corr[side] + gain * position_error, -0.08, 0.08)
                                        rot_error = Rotation.from_matrix(targets[side][:3, :3] @ actual[:3, :3].T).as_rotvec()
                                        corr_rot[side] = np.clip(corr_rot[side] + gain * rot_error, -0.6, 0.6)
                                    targets[side] = targets[side].copy()
                                    targets[side][:3, 3] += corr[side]
                                    targets[side][:3, :3] = Rotation.from_rotvec(corr_rot[side]).as_matrix() @ targets[side][:3, :3]
                                if e % 25 == 0:
                                    print(
                                        f"[pull]   wrist correction L={np.round(corr['left'], 3)} R={np.round(corr['right'], 3)} "
                                        f"rot(deg) L={np.round(np.degrees(corr_rot['left']), 1)} R={np.round(np.degrees(corr_rot['right']), 1)}",
                                        flush=True,
                                    )
                            pl["arrive"] = pl.get("arrive", 0)
                        elif e < 50 or "arrive" not in pl:
                            for side in ("left", "right"):
                                start, retracted = pl[side]
                                targets[side] = pelvis_pose @ _interp_pose(start, retracted, _smooth((e + 1) / 50))
                            if e >= 50:
                                ex = float(box_in_pelvis[0, 3]) - float(args_cli.pull_box_dist)
                                ey = float(box_in_pelvis[1, 3])
                                if (abs(ex) < 0.015 and abs(ey) < 0.015) or e >= 350:
                                    pl["arrive"] = e
                                    pl["reach_box"] = box_now.copy()
                                    pl["reach_start"] = {side: pelvis_pose @ pl[side][1] for side in ("left", "right")}
                                else:
                                    base_hold()
                        if "arrive" in pl and not args_cli.pull_by_walking:
                            k = e - pl["arrive"]

                            def hook_target(box_pose: np.ndarray, side: str, lateral: float, pull: float = 0.0) -> np.ndarray:
                                outward = 1.0 if side == "left" else -1.0
                                off = np.eye(4); off[:3, 3] = (float(args_cli.pull_slide_m) - pull, outward * lateral, 0.0)
                                return box_pose @ off @ HOOK_BOX_WRIST[side]

                            for side in ("left", "right"):
                                if k < 100:
                                    targets[side] = _interp_pose(pl["reach_start"][side], hook_target(pl["reach_box"], side, 0.04), _smooth((k + 1) / 100))
                                elif k < 150:
                                    f = _smooth((k - 99) / 50)
                                    targets[side] = hook_target(pl["reach_box"], side, (1 - f) * 0.04 - f * 0.005)
                                    hand_close = f
                                elif k < 400:
                                    f = _smooth((k - 149) / 250)
                                    targets[side] = hook_target(pl["reach_box"], side, -0.005, f * float(args_cli.pull_distance_m))
                                    hand_close = 1.0
                                else:
                                    targets[side] = hook_target(pl["reach_box"], side, -0.005, float(args_cli.pull_distance_m))
                                    hand_close = 1.0
                            base_hold()
                        for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
                            high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(inverse_pelvis @ targets[side])
                        high_level_action[0:2] = hand_close
                        pl["curl"] = hand_close
                        if e % 25 == 0:
                            yaw = math.degrees(Rotation.from_matrix(box_now[:3, :3]).as_euler("zyx")[0])
                            overhang = (float(table_xy[0]) - 0.20) - (float(box_now[0, 3]) - 0.5 * float(args_cli.box_size))
                            print(
                                f"[pull] e={e:4d} phase={pl.get('stage', 'approach' if 'arrive' not in pl else 'k='+str(e - pl['arrive']))} "
                                f"box_xy=({box_now[0,3]:.3f},{box_now[1,3]:+.3f}) box_yaw={yaw:+.1f} box_ahead={box_in_pelvis[0,3]:.3f} "
                                f"near_overhang={overhang*100:+.1f}cm box_z={box_now[2,3]:.3f} contacts L{left_count} R{right_count} pelvis_x={pelvis_pose[0,3]:.3f} frame={video_frame_count}",
                                flush=True,
                            )
                            inverse_box = np.linalg.inv(box_now)
                            for tip in ("left_hand_index_1_link", "left_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_thumb_2_link"):
                                tip_id = _body_ids(robot, (tip,))[0]
                                tip_box = inverse_box @ np.append(robot.data.body_link_pos_w[0, tip_id].detach().cpu().numpy(), 1.0)
                                print(f"[pull]   {tip} box-frame=({tip_box[0]:+.3f},{tip_box[1]:+.3f},{tip_box[2]:+.3f})", flush=True)
                            for side, wrist_id in (("left", left_wrist_id), ("right", right_wrist_id)):
                                actual = inverse_pelvis @ _torch_pose_wxyz(robot.data.body_link_pos_w[0, wrist_id], robot.data.body_link_quat_w[0, wrist_id])
                                wanted = inverse_pelvis @ targets[side]
                                print(
                                    f"[pull]   {side} wrist pelvis-frame actual=({actual[0,3]:.3f},{actual[1,3]:+.3f},{actual[2,3]:+.3f}) "
                                    f"target=({wanted[0,3]:.3f},{wanted[1,3]:+.3f},{wanted[2,3]:+.3f})",
                                    flush=True,
                                )
                    if args_cli.push_test and not navigating and source_index >= trajectory.close_index:
                        if pu_start is None:
                            pu_start = step
                            for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
                                start = np.eye(4)
                                start[:3, 3] = high_level_action[position_slice]
                                start[:3, :3] = Rotation.from_quat(high_level_action[quaternion_slice]).as_matrix()
                                retracted = start.copy()
                                retracted[:3, 3] = (0.15, 0.25, 0.18) if side == "left" else (0.12, -0.25, 0.10)
                                pu[side] = (start, retracted)
                        e = step - pu_start
                        box_now = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        inverse_pelvis = np.linalg.inv(pelvis_pose)
                        high_level_action[0:2] = 0.0
                        high_level_action[16:19] = 0.0
                        # right arm stays retracted throughout
                        start_r, retracted_r = pu["right"]
                        high_level_action[9:12], high_level_action[12:16] = _matrix_to_action_pose(_interp_pose(start_r, retracted_r, _smooth((e + 1) / 50)))
                        start_l, retracted_l = pu["left"]
                        left_pelvis = _interp_pose(start_l, retracted_l, _smooth((e + 1) / 50))
                        table_center_w = np.eye(4)
                        table_center_w[:3, 3] = (float(table_xy[0]), float(table_xy[1]), 0.74)
                        table_in_pelvis = inverse_pelvis @ table_center_w
                        walk_end = pu.get("walk_end")
                        if e >= 50 and walk_end is None:
                            ex = float(table_in_pelvis[0, 3]) - float(args_cli.push_stance[0])
                            ey = float(table_in_pelvis[1, 3]) - float(args_cli.push_stance[1])
                            if (abs(ex) < 0.015 and abs(ey) < 0.015) or e >= 350:
                                pu["walk_end"] = e
                                pu["box0"] = box_now.copy()
                                pu["left_start_w"] = pelvis_pose @ left_pelvis
                            else:
                                vx = float(np.clip(1.5 * ex, -0.15, 0.25)); vy = float(np.clip(1.5 * ey, -0.15, 0.15))
                                norm = np.hypot(vx, vy)
                                if norm < 0.06:
                                    vx, vy = vx * 0.06 / max(norm, 1e-6), vy * 0.06 / max(norm, 1e-6)
                                high_level_action[16] = vx; high_level_action[17] = vy
                        walk_end = pu.get("walk_end")
                        if walk_end is None:
                            high_level_action[2:5], high_level_action[5:9] = _matrix_to_action_pose(left_pelvis)
                        else:
                            k = e - walk_end
                            def left_target(y_off, z_off=0.0):
                                off = np.eye(4); off[:3, 3] = (float(args_cli.push_hand_x), y_off, z_off)
                                rot = np.eye(4); rot[:3, :3] = HOOK_BOX_WRIST["left"][:3, :3]
                                return pu["box0"] @ off @ rot
                            y_contact = 0.122
                            if k < 100:
                                world = _interp_pose(pu["left_start_w"], left_target(y_contact + 0.05), _smooth((k + 1) / 100))
                            elif k < 150:
                                world = _interp_pose(left_target(y_contact + 0.05), left_target(y_contact), _smooth((k - 99) / 50))
                            elif k < 350 and not pu.get("pushed", False):
                                # Closed loop: keep the palm 2 cm "inside" the Box's current left face,
                                # yaw-aligned with the Box, until it has travelled the push distance.
                                travelled = float(pu["box0"][1, 3]) - float(box_now[1, 3])
                                if travelled >= float(args_cli.push_distance_m):
                                    pu["pushed"] = True
                                    pu["box_pushed"] = box_now.copy()
                                yaw_now = Rotation.from_matrix(box_now[:3, :3]).as_euler("zyx")[0]
                                frame = np.eye(4)
                                frame[:3, :3] = Rotation.from_euler("z", yaw_now).as_matrix()
                                frame[:3, 3] = box_now[:3, 3]
                                off = np.eye(4); off[:3, 3] = (float(args_cli.push_hand_x), y_contact - 0.02, 0.0)
                                rot = np.eye(4); rot[:3, :3] = HOOK_BOX_WRIST["left"][:3, :3]
                                world = frame @ off @ rot
                                pu["last_push_world"] = world
                            else:
                                pu.setdefault("release_k", k)
                                start_w = pu.get("last_push_world", left_target(y_contact))
                                away = start_w.copy(); away[1, 3] += 0.06; away[2, 3] += 0.05
                                world = _interp_pose(start_w, away, _smooth((k - pu["release_k"] + 1) / 50))
                            high_level_action[2:5], high_level_action[5:9] = _matrix_to_action_pose(inverse_pelvis @ world)
                            if args_cli.hold_base_walk:
                                ex = float(table_in_pelvis[0, 3]) - float(args_cli.push_stance[0])
                                ey = float(table_in_pelvis[1, 3]) - float(args_cli.push_stance[1])
                                heading_error = math.atan2(float(table_in_pelvis[1, 3]), float(table_in_pelvis[0, 3])) - math.atan2(float(args_cli.push_stance[1]), float(args_cli.push_stance[0]))
                                vx = float(np.clip(2.0 * ex, -0.2, 0.2)); vy = float(np.clip(2.0 * ey, -0.15, 0.15))
                                norm = math.hypot(vx, vy)
                                if norm < 0.06:
                                    if norm < 1e-3:
                                        vx, vy = 0.06, 0.0
                                    else:
                                        vx, vy = vx * 0.06 / norm, vy * 0.06 / norm
                                high_level_action[16] = vx
                                high_level_action[17] = vy
                                high_level_action[18] = float(np.clip(1.5 * heading_error, -0.3, 0.3))
                        if e % 25 == 0:
                            yaw = math.degrees(Rotation.from_matrix(box_now[:3, :3]).as_euler("zyx")[0])
                            tilt = math.degrees(Rotation.from_matrix(box_now[:3, :3]).magnitude() if False else 0.0)
                            rp = Rotation.from_matrix(box_now[:3, :3]).as_euler("xyz")
                            overhang = -(float(box_now[1, 3]) - float(table_xy[1])) + 0.5 * float(args_cli.box_size) - 0.20
                            print(
                                f"[push] e={e:4d} phase={'walk' if pu.get('walk_end') is None else ('k='+str(e-pu['walk_end']))} box_xy=({box_now[0,3]:.3f},{box_now[1,3]:+.3f}) "
                                f"box_yaw={yaw:+.1f} box_roll_pitch=({math.degrees(rp[0]):+.1f},{math.degrees(rp[1]):+.1f}) overhang={overhang*100:+.1f}cm "
                                f"pelvis_xy=({pelvis_pose[0,3]:.3f},{pelvis_pose[1,3]:+.3f}) table_in_pelvis=({table_in_pelvis[0,3]:+.3f},{table_in_pelvis[1,3]:+.3f}) contacts L{left_count}",
                                flush=True,
                            )
                    if args_cli.squat_test and not navigating and source_index >= trajectory.close_index:
                        if sq_start is None:
                            sq_start = step
                            for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
                                start = np.eye(4)
                                start[:3, 3] = high_level_action[position_slice]
                                start[:3, :3] = Rotation.from_quat(high_level_action[quaternion_slice]).as_matrix()
                                retracted = start.copy()
                                retracted[:3, 3] = (0.15, 0.24 if side == "left" else -0.24, 0.05)
                                sq[side] = (start, retracted)
                        e = step - sq_start
                        for side, position_slice, quaternion_slice in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
                            start, retracted = sq[side]
                            high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(_interp_pose(start, retracted, _smooth((e + 1) / 50)))
                        high_level_action[0:2] = 0.0
                        high_level_action[16:19] = 0.0
                        table_center_w = np.eye(4)
                        table_center_w[:3, 3] = (float(table_xy[0]), float(table_xy[1]), 0.74)
                        table_in_pelvis = np.linalg.inv(pelvis_pose) @ table_center_w
                        ex = float(table_in_pelvis[0, 3]) - float(args_cli.squat_stance[0])
                        ey = float(table_in_pelvis[1, 3]) - float(args_cli.squat_stance[1])
                        if 50 <= e < 350 and not sq.get("arrived", False):
                            if abs(ex) < 0.015 and abs(ey) < 0.015:
                                sq["arrived"] = True
                                sq["arrive_step"] = e
                            else:
                                high_level_action[16] = float(np.clip(1.5 * ex, -0.15, 0.25))
                                high_level_action[17] = float(np.clip(1.5 * ey, -0.15, 0.15))
                                if np.hypot(high_level_action[16], high_level_action[17]) < 0.06:
                                    scale = 0.06 / max(1e-6, np.hypot(high_level_action[16], high_level_action[17]))
                                    high_level_action[16] *= scale
                                    high_level_action[17] *= scale
                        if e >= 350 or sq.get("arrived", False):
                            sq.setdefault("squat_start", e)
                            f = _smooth((e - sq["squat_start"] + 1) / 100)
                            high_level_action[19] = 0.75 + f * (float(args_cli.squat_height) - 0.75)
                            high_level_action[21] = f * float(args_cli.squat_pitch)
                        if e % 25 == 0:
                            _rpy = Rotation.from_matrix(pelvis_pose[:3, :3]).as_euler("xyz")
                            torso_id = robot.find_bodies("torso_link")[0][0]
                            _torso = robot.data.body_link_pos_w[0, torso_id].detach().cpu().numpy()
                            print(
                                f"[squat] e={e:4d} pelvis_z={pelvis_pose[2,3]:.3f} pelvis_rp=({math.degrees(_rpy[0]):+.1f},{math.degrees(_rpy[1]):+.1f})deg "
                                f"table_in_pelvis=({table_in_pelvis[0,3]:+.3f},{table_in_pelvis[1,3]:+.3f}) cmd_h={high_level_action[19]:.3f} cmd_pitch={high_level_action[21]:.2f} "
                                f"torso_xyz=({_torso[0]:.3f},{_torso[1]:.3f},{_torso[2]:.3f}) arrived={sq.get('arrived', False)}",
                                flush=True,
                            )
                    if rg_start is not None:
                        settled_height = 0.74 + 0.5 * float(args_cli.box_size)
                        if _regrasp_phase(step - rg_start)[0] == "open" and not rg.get("released", False):
                            if float(box.data.root_link_pos_w[0, 2]) - settled_height > 0.015 and rg.get("wait", 0) < 150:
                                rg["wait"] = rg.get("wait", 0) + 1
                                rg_start += 1  # hold the timeline in 'settle' until the Box rests on the table
                            else:
                                rg["released"] = True
                        phase, k, length = _regrasp_phase(step - rg_start)
                        frac = _smooth((k + 1) / length)
                        inverse_pelvis = np.linalg.inv(pelvis_pose)
                        box_now = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        high_level_action[16:19] = 0.0
                        targets = {}
                        hand_close = 1.0
                        if phase in ("place", "settle"):
                            desired_box = _interp_pose(rg["anchor_box"], rg["place_box"], frac if phase == "place" else 1.0)
                            for side in ("left", "right"):
                                targets[side] = desired_box @ rg["grip"][side]
                        elif phase in ("open", "retract", "stepin"):
                            if "placed_box" not in rg:
                                rg["placed_box"] = box_now.copy()
                            for side in ("left", "right"):
                                outward = 1.0 if side == "left" else -1.0
                                opened = np.eye(4)
                                opened[:3, 3] = (0.0, outward * 0.05, 0.0)
                                raised = np.eye(4)
                                raised[:3, 3] = (0.0, outward * 0.08, 0.12)
                                open_world = rg["placed_box"] @ opened @ rg["grip"][side]
                                raised_world = rg["placed_box"] @ raised @ rg["grip"][side]
                                if phase == "open":
                                    targets[side] = _interp_pose(rg["placed_box"] @ rg["grip"][side], open_world, frac)
                                elif phase == "retract":
                                    targets[side] = _interp_pose(open_world, raised_world, frac)
                                else:
                                    targets[side] = raised_world
                                rg.setdefault("raised_world", {})[side] = raised_world
                            if phase == "stepin":
                                box_in_pelvis = inverse_pelvis @ box_now
                                error = float(box_in_pelvis[0, 3]) - float(args_cli.regrasp_target_dist_m)
                                if error > 0.01:
                                    high_level_action[16] = float(np.clip(1.5 * error, 0.06, 0.25))
                                    high_level_action[17] = float(np.clip(-1.5 * float(box_in_pelvis[1, 3]), -0.1, 0.1))
                            hand_close = 0.0
                        elif phase in ("reach", "squeeze"):
                            if "reach_box" not in rg:
                                rg["reach_box"] = box_now.copy()
                                rg["reach_start"] = dict(rg["raised_world"])
                            for side in ("left", "right"):
                                outward = 1.0 if side == "left" else -1.0
                                lateral = float(args_cli.regrasp_open_m) if phase == "reach" else (1.0 - frac) * float(args_cli.regrasp_open_m) - frac * float(args_cli.regrasp_squeeze_m)
                                offset = np.eye(4)
                                offset[:3, 3] = (float(args_cli.regrasp_slide_m), outward * lateral, float(args_cli.regrasp_reach_dz_m))
                                goal = rg["reach_box"] @ offset @ rg["grip"][side]
                                targets[side] = _interp_pose(rg["reach_start"][side], goal, frac) if phase == "reach" else goal
                                if phase == "squeeze" and k == length - 1:
                                    rg.setdefault("regrip", {})[side] = np.linalg.inv(box_now) @ goal
                            hand_close = 0.0 if phase == "reach" else frac
                        elif phase in ("hug", "hold", "done"):
                            if "hug_anchor" not in rg:
                                rg["hug_anchor"] = box_now.copy()
                                if "regrip" not in rg:
                                    rg["regrip"] = {side: np.linalg.inv(box_now) @ (rg["reach_box"] @ np.eye(4) @ rg["grip"][side]) for side in ("left", "right")}
                            hx, hz, htilt = (float(v) for v in args_cli.regrasp_hug_pose)
                            hug_goal = np.eye(4)
                            hug_goal[:3, :3] = Rotation.from_euler("y", -math.radians(htilt)).as_matrix()
                            hug_goal[:3, 3] = (hx, 0.0, hz)
                            desired_box = _interp_pose(rg["hug_anchor"], pelvis_pose @ hug_goal, frac if phase == "hug" else 1.0)
                            for side in ("left", "right"):
                                targets[side] = desired_box @ rg["regrip"][side]
                        for side, position_slice, quaternion_slice in (
                            ("left", slice(2, 5), slice(5, 9)),
                            ("right", slice(9, 12), slice(12, 16)),
                        ):
                            high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(inverse_pelvis @ targets[side])
                        high_level_action[0:2] = hand_close
                    if hug_start_step is not None:
                        hug_elapsed = step - hug_start_step
                        hug_fraction = _smooth((hug_elapsed + 1) / int(args_cli.hug_move_steps))
                        hug_goal_pelvis = np.eye(4)
                        hug_goal_pelvis[:3, :3] = Rotation.from_euler("y", -math.radians(float(args_cli.hug_tilt_deg))).as_matrix()
                        hug_goal_pelvis[:3, 3] = (float(args_cli.hug_pos[0]), 0.0, float(args_cli.hug_pos[1]))
                        desired_box = _interp_pose(hug_anchor_box, pelvis_pose @ hug_goal_pelvis, hug_fraction)
                        inverse_pelvis = np.linalg.inv(pelvis_pose)
                        for side, position_slice, quaternion_slice in (
                            ("left", slice(2, 5), slice(5, 9)),
                            ("right", slice(9, 12), slice(12, 16)),
                        ):
                            high_level_action[position_slice], high_level_action[quaternion_slice] = _matrix_to_action_pose(
                                inverse_pelvis @ desired_box @ hug_wrist_in_box[side]
                            )
                        high_level_action[16:19] = 0.0
                    if support_start_step is not None:
                        support_elapsed = step - support_start_step
                        phase_steps = int(args_cli.support_phase_steps)
                        if support_elapsed < phase_steps:
                            release_fraction = 0.0
                        elif support_elapsed < 2 * phase_steps:
                            release_fraction = (support_elapsed - phase_steps + 1) / phase_steps
                        else:
                            release_fraction = max(
                                0.0, 1.0 - (support_elapsed - 2 * phase_steps + 1) / phase_steps
                            )
                        # Pelvis-frame lateral release (left wrist +y, right wrist -y)
                        # and an optional downward slide of both wrists.
                        release = float(args_cli.support_release_m) * release_fraction
                        drop = float(args_cli.support_drop_m) * min(
                            1.0, max(0.0, (support_elapsed - phase_steps + 1) / phase_steps)
                        )
                        if args_cli.support_wrist_pitch_rad != 0.0:
                            pitch = float(args_cli.support_wrist_pitch_rad) * min(
                                1.0, (support_elapsed + 1) / phase_steps
                            )
                            pitch_rotation = Rotation.from_euler("y", pitch)
                            for quat_slice in (slice(5, 9), slice(12, 16)):
                                high_level_action[quat_slice] = (
                                    pitch_rotation * Rotation.from_quat(high_level_action[quat_slice])
                                ).as_quat().astype(np.float32)
                        high_level_action[3] += release
                        high_level_action[10] -= release
                        high_level_action[4] -= drop
                        high_level_action[11] -= drop
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
                    if hook_start_step is not None:
                        hook_elapsed = step - hook_start_step - int(args_cli.hook_stepin_steps)
                        hook_p = int(args_cli.hook_phase_steps)
                        curl_fraction = 0.0 if args_cli.hook_hug else _smooth((hook_elapsed - 6 * hook_p + 1) / hook_p)
                        if curl_fraction > 0.0:
                            raw_action, absolute_target = _hook_finger_targets(
                                env, absolute_target, curl_fraction
                            )
                    if flat_palm_fraction > 0.0:
                        action_term = env.action_manager.get_term("JointPositionAction")
                        action_names = list(action_term._joint_names)
                        hand_goal = dict(FLAT_PALM_HAND)
                        if args_cli.flat_palm_press_curl is not None and source_index >= trajectory.close_index:
                            press = _smooth((source_index - trajectory.close_index + 1) / int(args_cli.flat_palm_close_steps))
                            proximal, distal = (float(v) for v in args_cli.flat_palm_press_curl)
                            for side, sign in (("left", -1.0), ("right", 1.0)):
                                for finger in ("index", "middle"):
                                    for joint, value in (("0", proximal), ("1", distal)):
                                        name = f"{side}_hand_{finger}_{joint}_joint"
                                        hand_goal[name] = (1.0 - press) * hand_goal[name] + press * sign * value
                        for name, goal in hand_goal.items():
                            index = action_names.index(name)
                            absolute_target[0, index] = (1.0 - flat_palm_fraction) * absolute_target[0, index] + flat_palm_fraction * goal
                        raw_action = (absolute_target - action_term._offset) / action_term._scale
                    if pl.get("finger_frac", 0.0) > 0.0:
                        action_term = env.action_manager.get_term("JointPositionAction")
                        action_names = list(action_term._joint_names)
                        for name, goal in pl["finger_goal"].items():
                            index = action_names.index(name)
                            absolute_target[0, index] = (1.0 - pl["finger_frac"]) * absolute_target[0, index] + pl["finger_frac"] * goal
                        raw_action = (absolute_target - action_term._offset) / action_term._scale
                    elif args_cli.pull_finger_curl is not None and pl.get("curl", 0.0) > 0.0:
                        action_term = env.action_manager.get_term("JointPositionAction")
                        action_names = list(action_term._joint_names)
                        proximal, distal = (float(v) for v in args_cli.pull_finger_curl)
                        for side, sign in (("left", -1.0), ("right", 1.0)):
                            for finger in ("index", "middle"):
                                for joint, goal in (("0", proximal), ("1", distal)):
                                    index = action_names.index(f"{side}_hand_{finger}_{joint}_joint")
                                    absolute_target[0, index] = (1.0 - pl["curl"]) * absolute_target[0, index] + pl["curl"] * sign * goal
                        raw_action = (absolute_target - action_term._offset) / action_term._scale
                    if support_start_step is not None:
                        support_elapsed = step - support_start_step
                        phase_steps = int(args_cli.support_phase_steps)
                        curl_start = phase_steps if args_cli.support_wrist_pitch_rad != 0.0 else 0
                        curl_fraction = min(
                            1.0, max(0.0, (support_elapsed - curl_start + 1) / phase_steps)
                        )
                        raw_action, absolute_target = _support_regrasp_targets(
                            env, absolute_target, curl_fraction
                        )

                    observation, _, _, _, _ = env.step(raw_action)
                    if video_writer is not None:
                        frame = env.render()
                        if frame is not None:
                            video_writer.append_data(np.asarray(frame, dtype=np.uint8))
                            video_frame_count += 1
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
                    if hug_start_step is not None and args_cli.hook_camera_dir is not None and (step - hug_start_step) in (0, int(args_cli.hug_move_steps), int(args_cli.hug_move_steps) + int(args_cli.hug_hold_steps) - 2):
                        import imageio

                        _rgb = env.scene["front_camera"].data.output["rgb"][0].detach().cpu().numpy()
                        _dir = Path(args_cli.hook_camera_dir).expanduser()
                        _dir.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(_dir / f"d435_hug_k{step - hug_start_step:04d}.png", _rgb[..., :3].astype(np.uint8))
                    if hook_start_step is not None and args_cli.hook_camera_dir is not None:
                        _k = step - hook_start_step - int(args_cli.hook_stepin_steps)
                        _p = int(args_cli.hook_phase_steps)
                        if _k in (6 * _p - 1, 8 * _p - 1, 11 * _p - 1, 11 * _p + int(args_cli.hook_hold_steps) - 2):
                            import imageio

                            _rgb = env.scene["front_camera"].data.output["rgb"][0].detach().cpu().numpy()
                            _dir = Path(args_cli.hook_camera_dir).expanduser()
                            _dir.mkdir(parents=True, exist_ok=True)
                            imageio.imwrite(_dir / f"d435_k{_k:04d}.png", _rgb[..., :3].astype(np.uint8))
                    if hook_start_step is not None and (step - hook_start_step) % hook_log_every == 0:
                        _pelvis_now = _torch_pose_wxyz(robot.data.body_link_pos_w[0, pelvis_id], robot.data.body_link_quat_w[0, pelvis_id])
                        _box_dist = float((np.linalg.inv(_pelvis_now) @ _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0]))[0, 3])
                        box_pose_now = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        half = 0.5 * float(args_cli.box_size)
                        near_bottom = (box_pose_now @ np.array([-half, 0.0, -half, 1.0]))[2]
                        far_bottom = (box_pose_now @ np.array([half, 0.0, -half, 1.0]))[2]
                        pitch = math.degrees(Rotation.from_matrix(box_pose_now[:3, :3]).as_euler("xyz")[1])
                        print(
                            f"[hook] k={step - hook_start_step:4d} pitch={pitch:+6.1f}deg "
                            f"near_bottom_z={near_bottom:.3f} far_bottom_z={far_bottom:.3f} "
                            f"box_xy=({float(box.data.root_link_pos_w[0,0]):.3f},{float(box.data.root_link_pos_w[0,1]):.3f}) "
                            f"contacts L{left_count} R{right_count} box_ahead_of_pelvis={_box_dist:.3f} "
                            f"box_z_rel_pelvis={float((np.linalg.inv(_pelvis_now) @ box_pose_now)[2, 3]):+.3f}",
                            flush=True,
                        )
                    lifted = box_height >= box_initial_height + 0.12
                    bilateral_contact = left_count > 0 and right_count > 0
                    lift_history.append(lifted)
                    left_contact_history.append(left_count > 0)
                    right_contact_history.append(right_count > 0)
                    if (
                        args_cli.support_regrasp
                        and support_start_step is None
                        and source_index >= trajectory.freeze_index
                        and lifted
                        and bilateral_contact
                    ):
                        support_start_step = step + 1
                    support_ready = not args_cli.support_regrasp or (
                        support_start_step is not None
                        and step - support_start_step + 1 >= support_total_steps
                    )
                    if lifted and bilateral_contact and support_ready:
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
                    if args_cli.regrasp_hug and nominal_success and rg_start is None:
                        rg_start = step + 1
                        rg["anchor_box"] = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        rg["grip"] = {}
                        for side, position_slice, quaternion_slice in (
                            ("left", slice(2, 5), slice(5, 9)),
                            ("right", slice(9, 12), slice(12, 16)),
                        ):
                            command_pelvis = np.eye(4)
                            command_pelvis[:3, 3] = high_level_action[position_slice]
                            command_pelvis[:3, :3] = Rotation.from_quat(high_level_action[quaternion_slice]).as_matrix()
                            rg["grip"][side] = np.linalg.inv(rg["anchor_box"]) @ (pelvis_pose @ command_pelvis)
                        half = 0.5 * float(args_cli.box_size)
                        table_near_edge_x = float(table_xy[0]) - 0.20
                        rg["place_box"] = _flat_box_pose(
                            rg["anchor_box"],
                            (table_near_edge_x + half + float(args_cli.regrasp_place_margin_m), float(rg["anchor_box"][1, 3]), 0.74 + half + float(args_cli.regrasp_place_dz_m)),
                        )
                        print(f"[regrasp] start at step {step}", flush=True)
                    if rg_start is not None:
                        _phase, _k, _len = _regrasp_phase(step - rg_start)
                        _pel = _torch_pose_wxyz(robot.data.body_link_pos_w[0, pelvis_id], robot.data.body_link_quat_w[0, pelvis_id])
                        _bx = np.linalg.inv(_pel) @ _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        if (step - rg_start) % 25 == 0:
                            print(
                                f"[regrasp] t={step - rg_start:4d} {_phase:8s} box_in_pelvis=({_bx[0,3]:+.3f},{_bx[1,3]:+.3f},{_bx[2,3]:+.3f}) "
                                f"pitch={math.degrees(Rotation.from_matrix(_bx[:3,:3]).as_euler('xyz')[1]):+.1f} box_world_z={box_height:.3f} "
                                f"contacts L{left_count} R{right_count} pelvis_x={float(robot.data.root_link_pos_w[0,0]):.3f}",
                                flush=True,
                            )
                        if args_cli.hook_camera_dir is not None and _k == _len - 1 and _phase in ("retract", "squeeze", "hug", "hold"):
                            import imageio

                            _rgb = env.scene["front_camera"].data.output["rgb"][0].detach().cpu().numpy()
                            _dir = Path(args_cli.hook_camera_dir).expanduser()
                            _dir.mkdir(parents=True, exist_ok=True)
                            imageio.imwrite(_dir / f"d435_regrasp_{_phase}.png", _rgb[..., :3].astype(np.uint8))
                        nominal_success = False
                    if args_cli.hug_after_hold and nominal_success and hug_start_step is None:
                        hug_start_step = step + 1
                        hug_anchor_box = _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        # Keep the *commanded* (compressed) wrist targets relative to the Box;
                        # the realized wrist poses would drop the squeeze force.
                        for side, position_slice, quaternion_slice in (
                            ("left", slice(2, 5), slice(5, 9)),
                            ("right", slice(9, 12), slice(12, 16)),
                        ):
                            command_pelvis = np.eye(4)
                            command_pelvis[:3, 3] = high_level_action[position_slice]
                            command_pelvis[:3, :3] = Rotation.from_quat(high_level_action[quaternion_slice]).as_matrix()
                            hug_wrist_in_box[side] = np.linalg.inv(hug_anchor_box) @ (pelvis_pose @ command_pelvis)
                        print(f"[hug] start at step {step}", flush=True)
                    if args_cli.hug_after_hold:
                        hug_done = hug_start_step is not None and step - hug_start_step + 1 >= int(args_cli.hug_move_steps) + int(args_cli.hug_hold_steps)
                        _pel = _torch_pose_wxyz(robot.data.body_link_pos_w[0, pelvis_id], robot.data.body_link_quat_w[0, pelvis_id])
                        _bx = np.linalg.inv(_pel) @ _torch_pose_wxyz(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
                        hug_height_ok = abs(float(_bx[2, 3]) - float(args_cli.hug_pos[1])) < 0.04
                        if hug_start_step is not None and (step - hug_start_step) % 25 == 0:
                            print(
                                f"[hug] k={step - hug_start_step:4d} box_in_pelvis=({_bx[0,3]:+.3f},{_bx[1,3]:+.3f},{_bx[2,3]:+.3f}) "
                                f"pitch={math.degrees(Rotation.from_matrix(_bx[:3,:3]).as_euler('xyz')[1]):+.1f} "
                                f"lift={lift_height:.3f} contacts L{left_count} R{right_count} pelvis_x={float(robot.data.root_link_pos_w[0,0]):.3f}",
                                flush=True,
                            )
                        nominal_success = bool(
                            hug_done and hug_height_ok and bilateral_contact and not bad_orientation and not dropped
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
                    if not navigating and step - navigation_steps + 1 >= trajectory.freeze_index + required_hold_steps + grace_steps + support_extra_steps:
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
        import traceback

        traceback.print_exc()
        if recorder is not None:
            recorder.abort()
        if snapshot_store is not None:
            snapshot_store.abort()
        raise
    finally:
        if video_writer is not None:
            video_writer.close()
            print(
                f"[teacher] video written: {args_cli.video_output} ({video_frame_count} frames)",
                flush=True,
            )
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
