#!/usr/bin/env python3
"""Approach-only physical evaluation of a BC-1 Student (single env, headless).

Mirrors the APPROACH branch of scripts/teacher/play_bc_skill_fsm.py in the
narrow workspace (no passthrough, no QR endpoint alignment, no wrist servo,
EMA alpha 1): the camera pre-grasp command is written into the observation,
the 23-D Student action reaches the Arena Pink/Homie worker unmodified except
for quaternion normalization, and the same physical BC-1 -> BC-2 gate ends the
episode.  Failure is bad orientation (> 0.8 rad, the FSM criterion) or timeout.
BC-2, Carry, Place and a BC-2 gate are not needed.

Modes
  student          BC-1 checkpoint drives the robot.
  teacher_replay   Open-loop replay of a recorded Teacher episode's 43-D joint
                   targets (no WBC) to verify that this environment reproduces
                   the recorded observations.
  teacher_collect  After the optional navigation/settle pre-phase, run the
                   unmodified Arena Mimic Teacher (time-indexed retargeted wrists,
                   camera pre-grasp base command) from the realized handoff state
                   and record demos in the TeacherDemoRecorder schema, so BC-1 can
                   be trained on post-navigation start states.

Camera profiles
  fsm_narrow   55.2 deg lens and command overrides of play_bc_skill_fsm.py.
  collector    environment defaults (69.4 deg) as used by the Teacher collector.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unitree_rl_lab.runtime_paths import default_arena_dataset, default_arena_root

REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=("student", "teacher_replay", "teacher_collect"), default="student")
parser.add_argument("--arena_dataset", default=default_arena_dataset(), help="Arena source demo of the Mimic Teacher (teacher_collect)")
parser.add_argument("--collect_output", type=Path, help="teacher_collect: new HDF5 file for the recorded demos")
parser.add_argument("--collect_extra_steps", type=int, default=5, help="teacher_collect: steps recorded after the Teacher's close onset")
parser.add_argument("--settle_steps_range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="sample the settle steps uniformly per episode (inclusive; replaces --settle_steps)")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--teacher_corpus", type=Path, default=REPO_ROOT / "data/teacher/box_xy_original_physx_500")
parser.add_argument("--teacher_episode", default="shard_00000.hdf5::demo_0")
parser.add_argument("--camera_profile", choices=("fsm_narrow", "collector"), required=True)
parser.add_argument("--task", default="Unitree-G1-43dof-BoxPickPlace")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=20261019)
parser.add_argument("--episode_seconds", type=float, default=8.0)
parser.add_argument("--box_x_range", type=float, nargs=2, default=(0.618, 0.622))
parser.add_argument("--box_y_range", type=float, nargs=2, default=(-0.002, 0.002))
parser.add_argument("--robot_start_offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DYAW"))
parser.add_argument("--prewalk_steps", type=int, default=0, help="Homie-only walking before BC-1 starts")
parser.add_argument("--prewalk_command", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
parser.add_argument("--settle_steps", type=int, default=0, help="zero-command Homie steps between prewalk/navigation and BC-1")
parser.add_argument(
    "--robot_start_range",
    type=float,
    nargs=6,
    default=None,
    metavar=("DX_LO", "DX_HI", "DY_LO", "DY_HI", "DYAW_LO", "DYAW_HI"),
    help="sample the robot start offset uniformly per episode (replaces --robot_start_offset)",
)
parser.add_argument(
    "--pre_upper_body",
    choices=("current_wrist", "nominal"),
    default="current_wrist",
    help="upper-body command before BC-1: current_wrist re-commands the measured wrist poses every step "
    "(earlier runs; the arms drift), nominal keeps the wrist poses measured right after reset",
)
parser.add_argument("--navigate", action="store_true", help="ground-truth scripted navigator drives to the canonical BC-1 start first")
parser.add_argument("--nav_policy", type=Path, default=None,
                    help="TorchScript PPO navigation actor (eval_nav_ppo.py --export_jit); drives Homie through the Arena WBC "
                         "until the BC-1 handoff state (navigation-env HandoffSuccess definition) holds 25 steps")
parser.add_argument("--nav_policy_scale", type=float, nargs=3, default=None,
                    help="override the profile's positive command scale (default: the profile's, v5 = 0.6 0.4 0.8)")
parser.add_argument("--nav_action_profile", choices=("v5", "v6", "v7"), default="v5",
                    help="action-to-command processing, observation layout and handoff criterion of the PPO navigation policy; "
                         "must match its training env (v5: Unitree-G1-43dof-BoxNav-Homie, v6: Unitree-G1-43dof-BoxNav-HomieV6, "
                         "v7: Unitree-G1-43dof-BoxNav-HomieV7)")
parser.add_argument("--nav_handoff_criterion", choices=("profile", "loose25", "stable50"), default="profile",
                    help="navigation handoff condition: loose25 = v5 HandoffSuccess (window + speed <= 0.1, yaw rate <= 0.2, 25 steps), "
                         "stable50 = v6 StableHandoffSuccess (window + speed <= 0.05, yaw rate <= 0.1, leg joint speed <= 0.3, "
                         "zero command, 50 steps); profile = the policy's own (v5 loose25, v6/v7 stable50)")
parser.add_argument("--nav_target_offset_x", type=float, default=None, help="navigation target point offset from the canonical start (m); default -0.03")
parser.add_argument("--nav_handoff_x_range", type=float, nargs=2, default=None, help="handoff success window in x (m); default -0.08 0.015")
parser.add_argument("--start_polar", type=float, nargs=4, default=None, metavar=("R_LO", "R_HI", "BEARING_DEG", "YAW_MAX"),
                    help="teleport the robot after reset to a start sampled like the navigation env (around the target point)")
parser.add_argument("--start_fixed", type=float, nargs="+", default=None, metavar="R BEARING_DEG HEADING_DEG",
                    help="explicit start per episode as triples: distance to the target point, bearing from the approach axis "
                         "(0 = straight in front of the table, +-90 = side, 180 = behind the table) and heading offset from "
                         "facing the target (180 = back to the target); replaces --start_polar and --episodes")
parser.add_argument("--robot_start_list", type=float, nargs="+", default=None, metavar="DX DY DYAW",
                    help="per-episode reset offsets from the canonical start as triples (episode i uses triple i); replaces "
                         "--robot_start_offset / --robot_start_range and --episodes (pose replay of recorded handoff states)")
parser.add_argument("--episode_ids", type=int, nargs="+", default=None,
                    help="run only these episode indices (starts and settle draws use the same per-episode RNG)")
parser.add_argument("--post_handoff_hold_steps", type=int, default=0,
                    help="PPO navigation: keep the PPO policy in control this many steps after the 25-step handoff hold, "
                         "then hand over (stop-stability test)")
parser.add_argument("--nav_trace", action="store_true", help="save a per-episode step trace of every phase (trace_XXX.npz)")
parser.add_argument("--table_safe_gap", type=float, default=None,
                    help="PPO navigation, phase 7: keep the base at least this far (m) behind the table front edge (GT table pose): "
                         "the handoff window front edge becomes min(window front, edge - gap - canonical start) per episode and the "
                         "target point --table_safe_margin behind it (no retraining; e.g. 0.41)")
parser.add_argument("--table_safe_margin", type=float, default=0.016,
                    help="target point distance (m) behind the safe window front edge (v7 forward excursion while stopping: p95 1.6 cm)")
parser.add_argument("--table_contact_sensors", action="store_true",
                    help="add table-filtered contact sensors on the arm/hand links (table_contact.py); the trace then also records "
                         "per-link table forces, arm/hand link positions, the Box orientation and fingertip-Box forces, and the "
                         "summary the table contact per phase (diagnostics only)")
parser.add_argument("--safety_metrics", action="store_true",
                    help="phase 8: per phase, table contact (force > 1 N), peak force, contact impulse, minimum arm/hand-table "
                         "clearance of the collision geometry (hand_clearance.py) and of the commanded wrist targets, plus "
                         "collision-free success flags in the summary; implies --table_contact_sensors (diagnostics only)")
parser.add_argument("--pre_bc1_align", choices=("none", "teacher_start"), default="none",
                    help="phase 8 Method A: before BC-1 / the Teacher, move the open hands with WBC wrist commands (min-jerk, "
                         "lifted where needed for --safe_margin table clearance) from the measured post-navigation poses to "
                         "the Teacher's first wrist targets (retargeted source step 0), then hold them --align_hold_steps")
parser.add_argument("--align_seconds", type=float, default=1.2)
parser.add_argument("--align_hold_steps", type=int, default=10)
parser.add_argument("--teacher_path", choices=("source", "safe"), default="source",
                    help="teacher_collect: source = unmodified retargeted Teacher wrist path; safe = phase 8 Method B (first "
                         "--safe_blend_steps blended from the measured wrist poses, lifted where needed for --safe_margin "
                         "table clearance before the close onset; grasp/lift unchanged)")
parser.add_argument("--safe_margin", type=float, default=0.02, help="commanded open-hand clearance to the table [m] (Methods A/B)")
parser.add_argument("--safe_blend_steps", type=int, default=25)
parser.add_argument("--safe_margin_late", type=float, default=None,
                    help="Method B: margin in the last --safe_margin_late_steps before the close onset (slow pre-grasp); "
                         "default --safe_margin. A smaller value keeps the pre-grasp cage closer to the original height")
parser.add_argument("--safe_margin_late_steps", type=int, default=30)
parser.add_argument("--wrist_servo_gain", type=float, default=0.0,
                    help="phase 8: closed-loop wrist correction (safe_paths.WristServo) during the alignment and the safe Teacher path, so the realized wrists follow the planned clearance path; integral gain per step on the position error (0 = off)")
parser.add_argument("--wrist_servo_rot_gain", type=float, default=None, help="rotation gain of the wrist servo (default: --wrist_servo_gain)")
parser.add_argument("--wrist_servo_max_rot_deg", type=float, default=34.4,
                    help="limit of the servo rotation correction (phase-8 cause run E ep6: 24-31 deg tilted the fingers into the table)")
parser.add_argument("--teacher_servo_gain", type=float, default=0.0,
                    help="wrist servo gain during the safe Teacher path (0 = open loop; the phase-8 smoke test oscillated at 0.3)")
parser.add_argument("--teacher_servo_late_steps", type=int, default=None,
                    help="run the safe-path wrist servo only from this many steps before the close onset (slow pre-grasp); it starts "
                         "from the planned command, position only (rotation gain --teacher_servo_rot_gain), frozen from the close onset")
parser.add_argument("--teacher_servo_rot_gain", type=float, default=0.0)
parser.add_argument("--align_margin", type=float, default=None, help="alignment planning margin [m] (default: --safe_margin)")
parser.add_argument("--video", action="store_true", help="record one mp4 per episode from the viewport camera (needs --enable_cameras)")
parser.add_argument("--video_size", type=int, nargs=2, default=(640, 360), metavar=("W", "H"))
parser.add_argument("--video_every", type=int, default=2, help="render every k-th 50 Hz control step (real-time playback at 50/k fps)")
parser.add_argument("--video_label", default="", help="header text of the video overlay")
parser.add_argument("--gui_follow", action="store_true", help="GUI runs: move the viewport camera like the video camera")
parser.add_argument("--nav_max_steps", type=int, default=750)
parser.add_argument("--nav_position_tolerance_m", type=float, default=0.03)
parser.add_argument("--nav_yaw_tolerance_rad", type=float, default=0.03)
parser.add_argument("--nav_min_command_norm", type=float, default=0.10, help="keeps Homie on its walking policy until arrival")
parser.add_argument("--nav_target_jitter", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DYAW"),
                    help="uniform per-episode offset of the navigation target around the canonical BC-1 start (half-ranges)")
parser.add_argument("--handoff_wbc_reset", action="store_true", help="reset the WBC worker (Homie history, Pink warm start) when BC-1 starts")
parser.add_argument(
    "--settle_height",
    type=float,
    default=0.75,
    help="Homie base-height command during the settle phase (0.75 = BC-1's own command; ~0.79 stands at the reset posture)",
)
parser.add_argument(
    "--pre_default_hold_steps",
    type=int,
    default=0,
    help="after navigation/settle, hold every joint at its default target with plain PD (raw action 0, the reset "
    "posture of every Teacher demo) for N steps before BC-1",
)
parser.add_argument("--arm_joint_offset_rad", type=float, default=0.0, help="offset added to shoulder/elbow joints at reset")
parser.add_argument("--hand_stiffness", type=float, default=4.0)
parser.add_argument(
    "--hand_state_decoding",
    choices=("none", "binary"),
    default="none",
    help="binary: map the Student hand-state outputs to the Arena {0, 1} contract (>= 0.5 closes); "
    "none reproduces play_bc_skill_fsm.py, where any non-zero value closes the hand",
)
parser.add_argument("--approach_ready_hold_steps", type=int, default=1)
parser.add_argument("--then_bc2", type=Path, default=None,
                    help="after the BC-1 gate, run this 43-D BC-2 checkpoint exactly as evaluate_bc2_rollout.py does "
                         "(policy joint targets straight to the env, zero velocity command; success = 12 cm lift with "
                         "bilateral contact held --bc2_hold_seconds)")
parser.add_argument("--bc2_seconds", type=float, default=10.0)
parser.add_argument("--then_teacher", action="store_true",
                    help="after the BC-1 gate, let the unmodified Arena Teacher continue from the same time index "
                         "(grasp/lift through the WBC); with --collect_output the continuation is recorded as BC-2 demos")
parser.add_argument("--bc2_hold_seconds", type=float, default=2.0)
parser.add_argument("--force_handoff_step_range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="BC-2 data collection only (--then_teacher): per episode draw a step in [LO, HI] (own RNG from the seed "
                         "and the episode); if the BC-1 gate has not opened by then, hand over to the Teacher continuation at that "
                         "step anyway.  Gives BC-2 demos from the early-handoff states the deployed gate can produce; the gate "
                         "itself is unchanged and evaluations never use this option")
parser.add_argument("--teacher_continuation_path", choices=("source", "safe"), default="source",
                    help="--then_teacher: source = unmodified Teacher from the BC-1 gate (it lowers the hands to the source cage, "
                         "which rests on this table); safe = phase 8 BC-2 v3 data: the source grasp/lift path raised by a constant "
                         "lift that keeps the predicted realized open hands --safe_margin_late (or --safe_margin) above the table "
                         "around the close onset, blended over 10 steps from the BC-1 student's last command")
parser.add_argument("--handoff_tip_distance_limit_m", type=float, default=0.18)
parser.add_argument("--handoff_cage_midpoint_limit_m", type=float, default=0.11)
parser.add_argument("--handoff_wrist_midpoint_limit_m", type=float, default=0.222)
parser.add_argument("--handoff_wrist_asymmetry_limit_m", type=float, default=0.10)
parser.add_argument("--gate_max_cage_width_m", type=float, default=None,
                    help="phase 9 readiness (default off): BC-1 keeps acting until the existing gate holds AND the cage is closed, i.e. "
                         "the distance between the left and right fingertip centres along the Box y axis is <= this value. Early "
                         "gates opened by the body sway left the left hand about 5 cm outside the Box (cage 28-33 cm vs 25-28 cm)")
parser.add_argument("--gate_min_open_steps", type=int, default=None,
                    help="phase 9 fixed-delay control (default off): BC-1 keeps acting and hands over only when the existing gate holds "
                         "and at least this many control steps have passed since it first opened")
parser.add_argument("--gate_ready_max_wait_steps", type=int, default=None,
                    help="phase 9: with --gate_max_cage_width_m, hand over at most this many steps after the existing gate first opened, "
                         "even if the cage has not closed (bounded delay; recorded as a fallback)")
parser.add_argument("--gate_ready_timeout_action", choices=("handoff", "abort"), default="handoff",
                    help="phase 10: when --gate_ready_max_wait_steps passes without the cage closing: handoff = hand over to BC-2 "
                         "anyway (phase-9 behaviour); abort = safe stop: BC-2 is not started, both wrist commands are raised by "
                         "--gate_abort_lift_m over 10 steps with zero base velocity and held, and the episode ends as readiness_timeout")
parser.add_argument("--gate_abort_lift_m", type=float, default=0.03)
parser.add_argument("--gate_abort_hold_steps", type=int, default=40)
parser.add_argument("--reseed_each_episode", action="store_true",
                    help="phase 9: reseed torch / numpy from (seed, episode) before every reset, so an episode's start (Box position "
                         "included) does not depend on how long the previous episodes ran; makes gate variants pairable")
parser.add_argument("--gate_max_tip_height_m", type=float, default=None,
                    help="phase 8 grasp-readiness gate (default off = the existing BC-1 -> BC-2 gate): additionally require the mean "
                         "height of both middle-finger tips (middle_1 link origins) above the table top to be <= this value, so BC-2 "
                         "starts from a cage at grasp height (BC-1 v2 ends at 1.8-2.7 cm, BC-1 v3 reached the plain gate at 3.9-12 cm)")
parser.add_argument("--arena_root", default=default_arena_root())
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--no_step_log", action="store_true")
parser.add_argument("--dump_observations", action="store_true", help="save per-episode pre-step 517-D inputs and actions (.npz)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.safety_metrics:
    args_cli.table_contact_sensors = True
if args_cli.teacher_path != "source" and args_cli.mode != "teacher_collect":
    parser.error("--teacher_path applies to --mode teacher_collect")

if args_cli.mode == "student" and args_cli.checkpoint is None:
    parser.error("--checkpoint is required in student mode")
if args_cli.episodes <= 0 or args_cli.episode_seconds <= 0.0 or args_cli.approach_ready_hold_steps <= 0:
    parser.error("episodes, episode_seconds and hold steps must be positive")
if args_cli.navigate and args_cli.nav_policy is not None:
    parser.error("use either --navigate (scripted) or --nav_policy (PPO)")
if (args_cli.navigate or args_cli.nav_policy is not None) and (args_cli.prewalk_steps or args_cli.nav_max_steps <= 0):
    parser.error("--navigate replaces --prewalk_steps and needs --nav_max_steps > 0")
if args_cli.robot_start_range is not None and any(args_cli.robot_start_offset):
    parser.error("use either --robot_start_offset or --robot_start_range")
if args_cli.mode == "teacher_collect" and (args_cli.collect_output is None or args_cli.collect_output.exists()):
    parser.error("teacher_collect needs a new --collect_output file")
if args_cli.then_teacher and (args_cli.mode != "student" or args_cli.then_bc2 is not None):
    parser.error("--then_teacher runs after a student BC-1 and excludes --then_bc2")
if (args_cli.gate_max_cage_width_m is not None or args_cli.gate_min_open_steps is not None) and args_cli.mode != "student":
    parser.error("--gate_max_cage_width_m / --gate_min_open_steps apply to the student BC-1 gate")
if args_cli.gate_ready_max_wait_steps is not None and args_cli.gate_max_cage_width_m is None:
    parser.error("--gate_ready_max_wait_steps bounds the --gate_max_cage_width_m wait")
if args_cli.force_handoff_step_range is not None and not (
        args_cli.then_teacher and 0 < args_cli.force_handoff_step_range[0] <= args_cli.force_handoff_step_range[1]):
    parser.error("--force_handoff_step_range needs --then_teacher and 0 < LO <= HI")
if args_cli.then_teacher and args_cli.collect_output is not None and args_cli.collect_output.exists():
    parser.error("--collect_output must be a new file")
if args_cli.settle_steps_range is not None and not 0 <= args_cli.settle_steps_range[0] <= args_cli.settle_steps_range[1]:
    parser.error("--settle_steps_range needs 0 <= LO <= HI")
if args_cli.start_fixed is not None and (len(args_cli.start_fixed) % 3 or args_cli.start_polar is not None):
    parser.error("--start_fixed takes R BEARING_DEG HEADING_DEG triples and excludes --start_polar")
if args_cli.post_handoff_hold_steps and args_cli.nav_policy is None:
    parser.error("--post_handoff_hold_steps applies to --nav_policy")
if args_cli.video and not args_cli.enable_cameras:
    parser.error("--video needs --enable_cameras")
if args_cli.robot_start_list is not None and (len(args_cli.robot_start_list) % 3 or args_cli.start_fixed is not None
                                               or args_cli.start_polar is not None or args_cli.robot_start_range is not None):
    parser.error("--robot_start_list takes DX DY DYAW triples and excludes the other start options")
EPISODE_IDS = list(range(args_cli.episodes if args_cli.start_fixed is None else len(args_cli.start_fixed) // 3))
if args_cli.robot_start_list is not None:
    EPISODE_IDS = list(range(len(args_cli.robot_start_list) // 3))
if args_cli.episode_ids is not None:
    EPISODE_IDS = list(args_cli.episode_ids)
MAX_SETTLE_STEPS = args_cli.settle_steps if args_cli.settle_steps_range is None else args_cli.settle_steps_range[1]
OUTPUT_DIR = args_cli.output_dir.expanduser().resolve()
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    parser.error(f"refusing to write into non-empty {OUTPUT_DIR}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.arena_wbc_bridge import ArenaWBCBridge
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.teacher.arena_teacher_runtime import (
    LEFT_FINGERTIP_NAMES,
    RIGHT_FINGERTIP_NAMES,
    ArenaLiftTrajectory,
    action_order_target,
    body_id,
    body_pose_wxyz,
    box_contact_counts,
    bridge_request,
    configure_fixed_teacher_scene,
    student_observation,
)
from unitree_rl_lab.utils.teacher.phase_bc import load_bc_policy
from unitree_rl_lab.utils.teacher.phase_teachers import pose_matrix_wxyz
from unitree_rl_lab.utils.teacher_demo_recorder import TeacherDemoRecorder, TeacherDemoStep
from unitree_rl_lab.utils.teacher.skill_fsm import bimanual_pregrasp_ready, bimanual_wrists_centered
from unitree_rl_lab.utils.teacher.student_observation import STUDENT_FEATURE_SLICES, replace_velocity_command

from p0_bc1_diagnostics import feature_layout
from scripted_nav import NavGains, bc1_start_pose, nav_command, pose_error_body, wrap_to_pi

NARROW_FOV_DEG = 55.2
BOX_SIZE_M = 0.2
QUATERNION_SLICES = (slice(5, 9), slice(12, 16))
OBS_GROUP_SLICES = {
    "history": slice(0, 460),
    "last_action": slice(460, 503),
    "velocity_commands": slice(503, 506),
    "box": slice(506, 510),
    "bin": slice(510, 514),
    "table": slice(514, 517),
}


def configure_env(box_x_range, box_y_range):
    """Narrow-workspace configuration of play_bc_skill_fsm._configure_env."""
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    cfg.seed = args_cli.seed
    configure_fixed_teacher_scene(
        cfg,
        box_size_m=BOX_SIZE_M,
        box_mass_kg=0.20,
        teacher_hand_stiffness=args_cli.hand_stiffness,
        keep_work_area_randomization=False,
    )
    cfg.commands.base_velocity.standoff_distance = 0.30
    cfg.commands.base_velocity.position_tolerance = 0.10
    cfg.commands.base_velocity.yaw_tolerance = 0.15
    if args_cli.camera_profile == "fsm_narrow":
        cfg.commands.base_velocity.horizontal_fov_deg = NARROW_FOV_DEG
        cfg.commands.base_velocity.box_visibility_grace_steps = 0
        cfg.commands.base_velocity.pregrasp_search_forward_speed = 0.0
        cfg.commands.base_velocity.pregrasp_search_yaw_velocity = -0.20
        for name in ("box_visible", "box_position_camera", "bin_visible", "bin_position_camera"):
            getattr(cfg.observations.policy_current, name).params["horizontal_fov_deg"] = NARROW_FOV_DEG
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
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if args_cli.robot_start_range is not None:
        x_lo, x_hi, y_lo, y_hi, yaw_lo, yaw_hi = args_cli.robot_start_range
        cfg.events.reset_base.params["pose_range"] = {"x": (x_lo, x_hi), "y": (y_lo, y_hi), "yaw": (yaw_lo, yaw_hi)}
    else:
        dx, dy, dyaw = args_cli.robot_start_offset
        cfg.events.reset_base.params["pose_range"] = {"x": (dx, dx), "y": (dy, dy), "yaw": (dyaw, dyaw)}
    cfg.events.reset_pick_place_scene.params["box_base_size"] = (BOX_SIZE_M,) * 3
    cfg.events.reset_pick_place_scene.params["grasp_box_xy_range"] = {"x": tuple(box_x_range), "y": tuple(box_y_range)}
    cfg.events.reset_pick_place_scene.params["box_xy_relative_to_table"] = False
    cfg.observations.policy_history.enable_corruption = False
    cfg.observations.policy_current.enable_corruption = False
    cfg.scene.front_camera = None
    cfg.terminations.time_out = None
    cfg.terminations.bad_orientation = None
    cfg.terminations.success = None
    cfg.rewards.bad_orientation_penalty = None
    pre_steps = args_cli.prewalk_steps + MAX_SETTLE_STEPS + args_cli.pre_default_hold_steps
    pre_steps += args_cli.nav_max_steps + args_cli.post_handoff_hold_steps if (args_cli.navigate or args_cli.nav_policy is not None) else 0
    if args_cli.pre_bc1_align != "none":
        pre_steps += int(math.ceil(args_cli.align_seconds / 0.02)) + args_cli.align_hold_steps
    cfg.episode_length_s = args_cli.episode_seconds + 1.0 + 0.02 * pre_steps
    if args_cli.then_bc2 is not None or args_cli.then_teacher:
        cfg.episode_length_s += args_cli.bc2_seconds
    if args_cli.video:
        cfg.viewer.resolution = tuple(args_cli.video_size)
        cfg.viewer.origin_type = "world"
    if args_cli.table_contact_sensors:
        from table_contact import add_table_contact_sensors  # same directory

        add_table_contact_sensors(cfg.scene)
    return cfg


def rpy_from_wxyz(q) -> list[float]:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def normalize_quaternions(action: np.ndarray) -> np.ndarray:
    result = action.copy()
    for part in QUATERNION_SLICES:
        norm = np.linalg.norm(result[:, part], axis=1, keepdims=True)
        if np.any(norm <= 1.0e-8):
            raise RuntimeError("APPROACH policy produced a degenerate wrist quaternion")
        result[:, part] = result[:, part] / norm
    return result


def round_list(values, digits: int = 5) -> list:
    return [round(float(v), digits) for v in np.asarray(values).ravel()]


class Recorder:
    def __init__(self) -> None:
        self.stream = None if args_cli.no_step_log else (OUTPUT_DIR / "steps.jsonl").open("x", buffering=1)

    def write(self, record: dict) -> None:
        if self.stream is not None:
            self.stream.write(json.dumps(record) + "\n")


PHASE_CODES = {"ppo_nav": 0, "nav": 0, "ppo_hold": 1, "settle": 2, "prewalk": 2, "default_hold": 3, "bc1": 4, "bc2": 5, "align": 6}
PHASE_TITLES = {0: "NAVIGATION", 1: "PPO HOLD (stop test)", 2: "SETTLE (zero command)", 3: "DEFAULT HOLD", 4: "BC-1 v2 approach / pre-grasp",
                5: "BC-2 v2 grasp & lift", 6: "ARM ALIGNMENT (Method A)"}
SAFETY_PHASES = {"nav": (0, 1, 2, 3), "align": (6,), "bc1": (4,), "bc2": (5,)}
CONTACT_THRESHOLD_N = 1.0


class SafetyMonitor:
    """Phase-8 safety metrics of one episode (only reads the simulator).

    Per step: table force (largest substep force) and impulse of the 22 arm/hand links (table_contact.py), the minimum
    signed distance of each link's collision surface to the table (hand_clearance.TableClearance), and, when the step
    had a commanded wrist pose, the clearance of the open hand placed at that command (commanded clearance).
    """

    def __init__(self, env, robot) -> None:
        from hand_clearance import GROUPS, TableClearance
        from table_contact import TABLE_CONTACT_LINKS

        if tuple(TABLE_CONTACT_LINKS) != tuple(__import__("hand_clearance").CLEARANCE_LINKS):
            raise RuntimeError("table contact links and clearance links must have the same order")
        history = env.scene.sensors[f"{TABLE_CONTACT_LINKS[0]}_table_contact"].cfg.history_length
        if history != env.cfg.decimation:
            raise RuntimeError(f"table contact history ({history}) must equal the decimation ({env.cfg.decimation})")
        self.links = list(TABLE_CONTACT_LINKS)
        self.groups = {name: [self.links.index(link) for link in members] for name, members in GROUPS.items()}
        self.clearance = TableClearance(robot, self.table_centers(env), env.device)
        self.reset()

    @staticmethod
    def table_centers(env):
        return [env.scene[name].data.root_link_pos_w[0].detach().cpu().numpy() for name in ("table_top", "table_pedestal")]

    def reset(self, env=None) -> None:
        if env is not None:
            import torch

            self.clearance.centers = torch.as_tensor(np.asarray(self.table_centers(env), dtype=np.float32), device=env.device)
        self.series = {key: [] for key in ("phase", "force", "impulse", "clearance", "cmd_clearance")}

    def step(self, env, robot, code: int, commanded=None) -> dict:
        from table_contact import table_forces, table_impulse

        force = table_forces(env)[0].detach().cpu().numpy()
        impulse = table_impulse(env)[0].detach().cpu().numpy()
        clearance = self.clearance.measured(robot).detach().cpu().numpy()
        cmd = np.full(2, np.nan, dtype=np.float32)
        if commanded is not None:
            result = self.clearance.commanded(commanded["left"], commanded["right"])
            cmd[:] = (result["left"], result["right"])
        for key, value in (("phase", code), ("force", force), ("impulse", impulse), ("clearance", clearance), ("cmd_clearance", cmd)):
            self.series[key].append(value)
        return {"table_force": force, "table_impulse": impulse, "clearance": clearance.astype(np.float32), "cmd_clearance": cmd}

    def summary(self) -> dict:
        if not self.series["phase"]:
            return {}
        phase = np.asarray(self.series["phase"])
        force, impulse = np.asarray(self.series["force"]), np.asarray(self.series["impulse"])
        clearance, cmd = np.asarray(self.series["clearance"]), np.asarray(self.series["cmd_clearance"])
        out = {}
        for label, codes in SAFETY_PHASES.items():
            rows = np.flatnonzero(np.isin(phase, codes))
            if not len(rows):
                continue
            f, j, c = force[rows], impulse[rows], clearance[rows]
            touching = f.max(axis=1) > CONTACT_THRESHOLD_N
            over = np.flatnonzero(touching)
            segments, current = [], 0.0
            for k in range(len(rows)):
                if touching[k]:
                    current += float(j[k].sum())
                elif current:
                    segments.append(current)
                    current = 0.0
            if current:
                segments.append(current)
            entry = {
                "steps": int(len(rows)),
                "contact": bool(len(over)),
                "steps_over_1N": int(len(over)),
                "first_over_1N_step": int(over[0]) if len(over) else None,
                "max_N": round(float(f.max()), 2),
                "max_N_link": self.links[int(f.max(axis=0).argmax())],
                "impulse_Ns": round(float(j.sum()), 4),
                "max_contact_event_impulse_Ns": round(max(segments), 4) if segments else 0.0,
                "contact_links": sorted({self.links[i] for i in np.flatnonzero((f > CONTACT_THRESHOLD_N).any(axis=0))}),
            }
            for group, idx in self.groups.items():
                values = c[:, idx]
                k = int(values.min(axis=1).argmin())
                entry[f"min_clearance_{group}_m"] = round(float(values.min()), 4)
                entry[f"min_clearance_{group}_step"] = k
                entry[f"min_clearance_{group}_link"] = self.links[idx[int(values[k].argmin())]]
            commanded = cmd[rows]
            if np.isfinite(commanded).any():
                k = int(np.nanargmin(np.nanmin(commanded, axis=1)))
                entry["commanded_min_clearance_m"] = round(float(np.nanmin(commanded)), 4)
                entry["commanded_min_clearance_step"] = k
                entry["commanded_steps_below_0"] = int((np.nanmin(commanded, axis=1) <= 0.0).sum())
            out[label] = entry
        return out


def safety_flags(summary: dict, safety: dict) -> dict:
    """Collision-free success flags next to (never replacing) the existing success definitions."""
    def free(label):
        return None if label not in safety else not safety[label]["contact"]

    nav = summary.get("nav") or {}
    nav_ok = nav.get("handoff", nav.get("arrived")) if nav else None
    approach_free = all(v is not False for v in (free("align"), free("bc1"))) if ("bc1" in safety) else None
    gate = summary.get("outcome") == "handoff_reached"
    bc2 = summary.get("bc2")
    bc2_ok = None if bc2 is None else bc2.get("outcome") == "lift_hold_success"
    flags = {
        "nav_handoff": nav_ok,
        "nav_contact_free": free("nav"),
        "align_contact_free": free("align"),
        "approach_contact_free": approach_free,
        "bc1_gate": gate,
        "bc1_collision_free_success": bool(gate and approach_free),
        "bc2_success": bc2_ok,
        "bc2_contact_free": free("bc2"),
    }
    nav_good = True if nav_ok is None else bool(nav_ok and flags["nav_contact_free"] is not False)
    flags["e2e"] = bool((nav_ok is None or nav_ok) and gate and bc2_ok)
    flags["cf_e2e"] = bool(nav_good and flags["bc1_collision_free_success"] and bc2_ok)
    flags["cf_e2e_strict"] = bool(flags["cf_e2e"] and flags["bc2_contact_free"] is not False)
    return flags


class StepMonitor:
    """Optional per-step trace (--nav_trace) and overlay video (--video) of one evaluation episode.

    Called after every env.step of every phase.  It only reads simulator state (and moves the viewport
    camera for the video); nothing it computes is fed back to any controller.
    """

    def __init__(self, env, robot) -> None:
        names = list(robot.data.body_names)
        self.feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
        self.sensor = env.scene.sensors.get("contact_forces")
        self.sensor_names = list(self.sensor.body_names) if self.sensor is not None else []
        self.sensor_feet = [self.sensor_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")] if self.sensor else []
        self.joint_names = list(robot.data.joint_names)
        self.step_dt = float(env.step_dt)
        self.ppo_obs_width = 18 if args_cli.nav_action_profile == "v5" else 21  # navigation observation width (v5 18, v6/v7 21)
        self.table_link_ids = []
        if args_cli.table_contact_sensors:
            from table_contact import TABLE_CONTACT_LINKS, has_table_contact_sensors

            if has_table_contact_sensors(env):
                self.table_link_ids = [names.index(link) for link in TABLE_CONTACT_LINKS]
        self.safety = SafetyMonitor(env, robot) if args_cli.safety_metrics else None
        self.rows: dict[str, list] = {}
        self.writer = None
        self.k = 0
        self.eye = self.lookat = None
        self.side = None
        self.episode = None
        self.header = ""

    def begin(self, env, robot, episode: int, header: str) -> None:
        self.rows = {}
        self.k = 0
        self.episode = episode
        self.header = header
        self.eye = self.lookat = self.side = None
        self.path = []
        if self.safety is not None:
            self.safety.reset(env)
        if args_cli.gui_follow and not args_cli.video:
            self._camera(env, robot, snap=True)
        if args_cli.video:
            import imageio.v2 as imageio

            fps = 1.0 / (env.step_dt * args_cli.video_every)
            self.writer = imageio.get_writer(OUTPUT_DIR / f"episode_{episode:03d}.mp4", fps=fps, codec="libx264", quality=8,
                                             pixelformat="yuv420p", macro_block_size=2)
            self._camera(env, robot, snap=True)
            for _ in range(3):  # RTX warm-up frames are not written
                env.render()

    def _camera(self, env, robot, snap: bool = False) -> None:
        """Side view that keeps the robot and the table front in frame (zooms in as the robot arrives)."""
        from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

        root = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        focus = nav_mdp.target_point(env)[0][0].detach().cpu().numpy()
        box = env.scene["box"].data.root_link_pos_w[0, :2].detach().cpu().numpy()
        focus = 0.5 * (focus + box)
        path = root[:2] - focus
        d = float(np.linalg.norm(path))
        # Far away: look at the walk from the side of the path.  Near the table: 3/4 view of the robot's left
        # front (away from the bin), so the hands and the Box stay visible during BC-1 / BC-2.
        final_dir = np.array([0.45, 0.89])
        normal = np.array([-path[1], path[0]]) / max(d, 1.0e-6)
        if float(normal @ final_dir) < 0.0:
            normal = -normal
        w = float(np.clip((d - 0.6) / 2.0, 0.0, 1.0))
        side = w * normal + (1.0 - w) * final_dir
        side /= max(float(np.linalg.norm(side)), 1.0e-6)
        self.side = side if self.side is None else self.side + 0.15 * (side - self.side)
        mid = 0.5 * (root[:2] + focus)
        eye = np.array([*(mid + self.side * (2.6 + 0.80 * d)), 1.55 + 0.25 * d])
        lookat = np.array([*mid, 0.70])
        if snap or self.eye is None:
            self.eye, self.lookat = eye, lookat
        else:
            self.eye += 0.12 * (eye - self.eye)
            self.lookat += 0.12 * (lookat - self.lookat)
        env.sim.set_camera_view(eye=self.eye.tolist(), target=self.lookat.tolist())

    def after_step(self, env, robot, phase: str, cmd=None, ppo_obs=None, ppo_action=None, extra: str = "", commanded=None,
                   joint_target=None) -> None:
        if args_cli.gui_follow and not args_cli.video:
            self._camera(env, robot)
        safety = None if self.safety is None else self.safety.step(env, robot, PHASE_CODES[phase], commanded)
        if not (args_cli.nav_trace or args_cli.video):
            return
        from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

        data = robot.data
        code = PHASE_CODES[phase]
        nan3 = np.full(3, np.nan, dtype=np.float32)
        cmd = nan3 if cmd is None else np.asarray(cmd, dtype=np.float32)
        errors = nav_mdp.handoff_errors(env)[0].detach().cpu().numpy()
        root = data.root_link_pos_w[0].detach().cpu().numpy()
        lin_b = data.root_link_lin_vel_b[0].detach().cpu().numpy()
        box = env.scene["box"].data.root_link_pos_w[0].detach().cpu().numpy()
        if args_cli.nav_trace:
            if self.sensor is not None:
                forces = torch.linalg.vector_norm(self.sensor.data.net_forces_w[0], dim=-1).detach().cpu().numpy()
                other = forces.copy()
                other[self.sensor_feet] = 0.0
                feet_force, other_max, other_body = forces[self.sensor_feet], float(other.max()), int(other.argmax())
            else:
                feet_force, other_max, other_body = np.full(2, np.nan), np.nan, -1
            row = {
                "phase": code,
                "cmd": cmd,
                "ppo_obs": np.full(self.ppo_obs_width, np.nan, dtype=np.float32) if ppo_obs is None else np.asarray(ppo_obs, dtype=np.float32),
                "ppo_action": np.full(4, np.nan, dtype=np.float32) if ppo_action is None else np.asarray(ppo_action, dtype=np.float32),
                "root_pos": root,
                "root_quat": data.root_link_quat_w[0].detach().cpu().numpy(),
                "lin_vel_b": lin_b,
                "ang_vel_b": data.root_link_ang_vel_b[0].detach().cpu().numpy(),
                "projected_gravity": data.projected_gravity_b[0].detach().cpu().numpy(),
                "handoff_errors": errors,
                "feet_pos": data.body_link_pos_w[0, self.feet].detach().cpu().numpy(),
                "feet_vel": data.body_link_lin_vel_w[0, self.feet].detach().cpu().numpy(),
                "feet_force": feet_force,
                "nonfoot_force_max": np.float32(other_max),
                "nonfoot_force_body": np.int32(other_body),
                "table_dist": np.float32(float(nav_mdp._table_body_distance(env)[0])),
                "box_pos": box,
                "joint_pos": data.joint_pos[0].detach().cpu().numpy(),
                "joint_vel": data.joint_vel[0].detach().cpu().numpy(),
            }
            if self.table_link_ids:
                from table_contact import box_fingertip_forces, table_forces

                row["table_force"] = table_forces(env)[0].detach().cpu().numpy()
                row["arm_link_pos"] = data.body_link_pos_w[0, self.table_link_ids].detach().cpu().numpy()
                row["box_quat"] = env.scene["box"].data.root_link_quat_w[0].detach().cpu().numpy()
                tips = box_fingertip_forces(env)
                row["box_tip_force"] = np.full(6, np.nan, dtype=np.float32) if tips is None else tips[0].detach().cpu().numpy()
            if safety is not None:
                row["arm_link_quat"] = data.body_link_quat_w[0, self.table_link_ids].detach().cpu().numpy()
                row["table_impulse"] = safety["table_impulse"]
                row["clearance"] = safety["clearance"]
                row["cmd_clearance"] = safety["cmd_clearance"]
                for side in ("left", "right"):
                    wrist = np.full((4, 4), np.nan, dtype=np.float32) if commanded is None else np.asarray(commanded[side], dtype=np.float32)
                    row[f"cmd_wrist_{side}"] = wrist
                row["joint_target"] = (np.full(len(self.joint_names), np.nan, dtype=np.float32) if joint_target is None
                                       else np.asarray(joint_target, dtype=np.float32))
            for key, value in row.items():
                self.rows.setdefault(key, []).append(value)
        if self.writer is not None:
            self.path.append(root[:2].copy())
        if self.writer is not None and self.k % args_cli.video_every == 0:
            self._camera(env, robot)
            frame = np.ascontiguousarray(env.render())
            t = self.k * env.step_dt
            dist = float(nav_mdp.target_errors(env)[0][0])
            title, _, start_text = self.header.partition(" | ")
            lines = [title, start_text, f"[{PHASE_TITLES[code]}]  t = {t:5.2f} s"] if start_text else [title, f"[{PHASE_TITLES[code]}]  t = {t:5.2f} s"]
            if ppo_action is not None:
                a = np.asarray(ppo_action, dtype=np.float32)
                lines.append(f"PPO output vx {a[0]:+.2f} vy {a[1]:+.2f} wz {a[2]:+.2f} stop {a[3]:+.2f}" + ("  -> STOP" if a[3] > 0 else ""))
            if np.isfinite(cmd).all():
                walk = float(np.linalg.norm(cmd)) >= 0.05
                lines.append(f"Homie cmd [{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}]  {'WALK' if walk else 'STAND'} policy")
            lines.append(f"base speed {float(np.linalg.norm(lin_b[:2])):.2f} m/s | handoff err dx {100 * errors[0]:+.1f} dy {100 * errors[1]:+.1f} cm"
                         f" dyaw {errors[2]:+.3f} rad")
            if code == 0 and math.isfinite(dist):
                lines[-1] = f"dist to target {dist:.2f} m | " + lines[-1]
            if extra:
                lines.append(extra)
            frame = self._overlay(frame, lines)
            self.writer.append_data(self._minimap(env, frame, robot))
        self.k += 1

    @staticmethod
    def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
        import cv2

        frame = frame.copy()
        scale = frame.shape[0] / 540.0
        size, thick, step = 0.48 * scale, max(1, int(round(1.2 * scale))), int(20 * scale)
        width = int(max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0][0] for s in lines)) + 12
        box = frame[: step * len(lines) + 8, : min(frame.shape[1], width)]
        box[:] = (0.40 * box).astype(np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (6, step * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), thick, cv2.LINE_AA)
        return frame

    def _minimap(self, env, frame: np.ndarray, robot) -> np.ndarray:
        """Top-down inset: table, Box, BC-1 target point, start and the walked path, robot heading."""
        import cv2

        from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

        h = frame.shape[0]
        size = int(0.34 * h)
        x0, y0 = frame.shape[1] - size - 8, h - size - 8
        target = nav_mdp.target_point(env)[0][0].detach().cpu().numpy()
        table = np.asarray(nav_mdp.TABLE_CENTER_XY_E) + env.scene.env_origins[0, :2].detach().cpu().numpy()
        pts = np.asarray(self.path)
        lo = np.minimum(pts.min(axis=0), table - 0.35) - 0.3
        hi = np.maximum(pts.max(axis=0), table + 0.35) + 0.3
        span = float(max(hi - lo))
        center = 0.5 * (lo + hi)

        def px(p):  # world x up, world y left (top-down view from above, approach direction up)
            u = (center[1] - p[1]) / span + 0.5
            v = (center[0] - p[0]) / span + 0.5
            return int(x0 + 6 + u * (size - 12)), int(y0 + 6 + v * (size - 12))

        roi = frame[y0:y0 + size, x0:x0 + size]
        roi[:] = (0.35 * roi + 0.65 * 30).astype(np.uint8)
        half = nav_mdp.TABLE_HALF_EXTENT_M
        cv2.rectangle(frame, px(table + [half, half]), px(table - [half, half]), (200, 160, 110), -1)
        box = env.scene["box"].data.root_link_pos_w[0, :2].detach().cpu().numpy()
        cv2.rectangle(frame, px(box + [0.1, 0.1]), px(box - [0.1, 0.1]), (190, 120, 60), -1)
        cv2.polylines(frame, [np.asarray([px(p) for p in pts], dtype=np.int32)], False, (90, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, px(pts[0]), 3, (255, 255, 255), -1)
        t = px(target)
        cv2.drawMarker(frame, t, (60, 255, 60), cv2.MARKER_CROSS, 10, 1)
        yaw = rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2]
        r = pts[-1]
        cv2.circle(frame, px(r), 4, (255, 90, 90), -1)
        cv2.line(frame, px(r), px(r + 0.06 * span * np.array([math.cos(yaw), math.sin(yaw)])), (255, 90, 90), 2, cv2.LINE_AA)
        cv2.putText(frame, "top view", (x0 + 4, y0 + size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32 * h / 360, (230, 230, 230), 1, cv2.LINE_AA)
        return frame

    def end(self, summary: dict) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            summary["video"] = f"episode_{self.episode:03d}.mp4"
        if self.safety is not None:
            summary["safety"] = self.safety.summary()
            summary["safety_flags"] = safety_flags(summary, summary["safety"])
        if args_cli.nav_trace and self.rows:
            arrays = {k: np.asarray(v) for k, v in self.rows.items()}
            extra = {}
            if self.table_link_ids:
                from table_contact import BOX_FINGERTIP_SENSORS, TABLE_CONTACT_LINKS

                extra = {"table_link_names": np.asarray(TABLE_CONTACT_LINKS), "box_tip_sensor_names": np.asarray(BOX_FINGERTIP_SENSORS)}
                # table contact per phase: largest force, its link, steps above 1 N, first step above 1 N (within the phase)
                contact = {}
                for label, code in (("nav", 0), ("bc1", 4), ("bc2", 5)):
                    rows = np.flatnonzero(arrays["phase"] == code)
                    if not len(rows):
                        continue
                    force = arrays["table_force"][rows]
                    over = np.flatnonzero(force.max(axis=1) > 1.0)
                    contact[label] = {"max_N": round(float(force.max()), 2), "link": TABLE_CONTACT_LINKS[int(force.max(axis=0).argmax())],
                                      "steps_over_1N": int(len(over)), "first_over_1N_step": int(over[0]) if len(over) else None}
                summary["table_contact"] = contact
            np.savez_compressed(OUTPUT_DIR / f"trace_{self.episode:03d}.npz", **arrays, **extra,
                                sensor_body_names=np.asarray(self.sensor_names), joint_names=np.asarray(self.joint_names),
                                step_dt=np.float32(self.step_dt))
            summary["trace"] = f"trace_{self.episode:03d}.npz"


def main() -> None:
    if args_cli.nav_target_offset_x is not None or args_cli.nav_handoff_x_range is not None:
        from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

        nav_mdp.apply_nav_target_overrides(args_cli.nav_target_offset_x, args_cli.nav_handoff_x_range)
    teacher = None
    box_x_range, box_y_range = args_cli.box_x_range, args_cli.box_y_range
    if args_cli.mode == "teacher_replay":
        shard, demo = args_cli.teacher_episode.split("::")
        with h5py.File(args_cli.teacher_corpus / shard, "r") as stream:
            group = stream["data"][demo]
            teacher = {
                "observation": group["student_observation"][:],
                "target": group["processed_43d_joint_target"][:],
                "root_pose": group["root_pose"][:],
                "joint_names": [n.decode() if isinstance(n, bytes) else str(n) for n in stream["metadata/processed_joint_names"][:]],
            }
            initial_box = json.loads(group.attrs["metadata_json"])["initial_box_pose_wxyz"]
        box_x_range = (initial_box[0], initial_box[0])
        box_y_range = (initial_box[1], initial_box[1])

    cfg = configure_env(box_x_range, box_y_range)
    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array" if args_cli.video else None).unwrapped
    robot = env.scene["robot"]
    monitor = StepMonitor(env, robot)
    left_tips = [body_id(robot, n) for n in LEFT_FINGERTIP_NAMES]
    right_tips = [body_id(robot, n) for n in RIGHT_FINGERTIP_NAMES]
    left_wrist = body_id(robot, "left_wrist_yaw_link")
    right_wrist = body_id(robot, "right_wrist_yaw_link")
    pelvis = body_id(robot, "pelvis")
    action_term = env.action_manager.get_term("JointPositionAction")
    action_joint_names = list(action_term._joint_names)
    arm_joint_ids = [i for i, n in enumerate(robot.data.joint_names) if "shoulder" in n or "elbow" in n]
    upper_joint_ids = [
        i for i, n in enumerate(robot.data.joint_names) if any(k in n for k in ("shoulder", "elbow", "wrist", "waist"))
    ]

    policy = None
    mean = scale = None
    bc2_policy = None
    if args_cli.then_bc2 is not None:
        bc2_policy = load_bc_policy(args_cli.then_bc2, args_cli.device)
        if bc2_policy.action_key != "processed_43d_joint_target":
            raise ValueError("--then_bc2 expects the 43-D BC-2 joint-target policy")
    if args_cli.mode == "student":
        policy = load_bc_policy(args_cli.checkpoint, args_cli.device)
        if policy.action_key != "teacher_high_level_action":
            raise ValueError("this evaluator drives a 23-D high-level BC-1 policy")
        mean = policy.model.observation_mean.detach().cpu().numpy().astype(np.float64)
        scale = policy.model.observation_scale.detach().cpu().numpy().astype(np.float64)
    if teacher is not None and teacher["joint_names"] != action_joint_names:
        raise ValueError("recorded Teacher target order differs from the action term order")
    joint_order = [n.decode() if isinstance(n, bytes) else str(n) for n in robot.data.joint_names]
    feature_names, feature_groups, _ = feature_layout(action_joint_names)
    group_array = np.asarray(feature_groups)
    finger_dims = np.flatnonzero(np.char.find(group_array, ":finger") >= 0)
    gravity_dims = np.flatnonzero(group_array == "projected_gravity")
    bin_dims = np.arange(510, 514)

    needs_wbc = (args_cli.mode in ("student", "teacher_collect") or args_cli.prewalk_steps + MAX_SETTLE_STEPS > 0 or args_cli.navigate
                 or args_cli.nav_policy is not None)
    bridge = ArenaWBCBridge(arena_root=args_cli.arena_root, step_timeout_s=10.0) if needs_wbc else None
    recorder = Recorder()
    collector = None
    if args_cli.mode == "teacher_collect" or (args_cli.then_teacher and args_cli.collect_output is not None):
        import hashlib
        import importlib

        task_cfg_module = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.box_carry_with_hand_env_cfg")
        trajectory = ArenaLiftTrajectory(args_cli.arena_dataset)
        source_sha256 = hashlib.sha256(Path(args_cli.arena_dataset).read_bytes()).hexdigest()
        collector = {
            "trajectory": trajectory,
            "bin_pose": np.array([*task_cfg_module.BIN_CENTER_E, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "box_size": np.full(3, BOX_SIZE_M, dtype=np.float32),
            "recorded": 0,
            "recorder": TeacherDemoRecorder(
                args_cli.collect_output,
                control_frequency_hz=1.0 / env.step_dt,
                joint_names=action_joint_names,
                phase_names={0: "grasp", 1: "carry", 2: "place", 3: "complete"},
                quaternion_convention="wxyz",
                metadata={
                    "task": args_cli.task,
                    "teacher": "Arena g1_wbc_pink + HomieV2 sidecar; unmodified ArenaLiftTrajectory (Mimic TRANSLATE)",
                    "collector": "scripts/approach/eval_bc1_approach.py --mode teacher_collect",
                    "teacher_skill_scope": "grasp_lift_from_student_bc1_gate" if args_cli.then_teacher else "approach_from_handoff",
                    "arena_source_dataset": str(Path(args_cli.arena_dataset).resolve()),
                    "arena_source_dataset_sha256": source_sha256,
                    "injected_bimanual_close_start_step": int(trajectory.close_index),
                    "recorded_phase_id_contract": "0=grasp collector scope; BC splitter uses close/contact signals",
                    "student_velocity_command_contract": "camera pregrasp command for APPROACH; forced [0,0,0] from BC-2 close onset",
                    "camera_profile": args_cli.camera_profile,
                    "handoff_protocol": {
                        "navigate": args_cli.navigate,
                        "robot_start_range": args_cli.robot_start_range,
                        "robot_start_offset": list(args_cli.robot_start_offset),
                        "settle_steps": args_cli.settle_steps,
                        "settle_steps_range": args_cli.settle_steps_range,
                        "settle_height": args_cli.settle_height,
                        "pre_upper_body": args_cli.pre_upper_body,
                        "handoff_wbc_reset": args_cli.handoff_wbc_reset,
                    },
                    "success_definition": (
                        "12 cm lift with bilateral contact held 2 s after the student BC-1 gate" if args_cli.then_teacher else
                        "BC-1 physical gate (fingertip/cage/wrist-centring) reached before the Teacher's close onset, no fall"),
                    "student_bc1_checkpoint": str(args_cli.checkpoint) if args_cli.then_teacher else None,
                    "seed": args_cli.seed,
                },
            ),
        }
    if args_cli.pre_bc1_align != "none":
        global _ALIGN_TRAJECTORY
        _ALIGN_TRAJECTORY = collector["trajectory"] if collector is not None else ArenaLiftTrajectory(args_cli.arena_dataset)
    summaries = []
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args_cli).items()}
    config.update(
        {
            "camera_fov_deg": float(cfg.commands.base_velocity.horizontal_fov_deg),
            "box_visibility_grace_steps": int(cfg.commands.base_velocity.box_visibility_grace_steps),
            "observation_box_fov_deg": cfg.observations.policy_current.box_visible.params["horizontal_fov_deg"],
            "observation_bin_fov_deg": cfg.observations.policy_current.bin_visible.params["horizontal_fov_deg"],
            "robot_joint_order_equals_action_order": joint_order == action_joint_names,
            "checkpoint_metadata": None if policy is None else {
                k: policy.metadata.get(k) for k in ("best_epoch", "approach_research", "observation_scale_floor")
            },
        }
    )
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    try:
        for episode in EPISODE_IDS:
            summaries.append(run_episode(env, robot, bridge, recorder, policy, mean, scale, teacher, episode,
                                         left_tips, right_tips, left_wrist, right_wrist, pelvis, arm_joint_ids,
                                         upper_joint_ids, feature_names, finger_dims, gravity_dims, bin_dims,
                                         collector=collector, bc2_policy=bc2_policy, monitor=monitor))
            monitor.end(summaries[-1])
            keys = ("outcome", "steps", "handoff_step", "max_tilt_rad", "start_offset_canonical", "start_arm_dev_max_rad")
            brief = {k: summaries[-1].get(k) for k in keys}
            nav_brief = summaries[-1].get("nav") or {}
            brief.update({"nav_ok": nav_brief.get("handoff", nav_brief.get("arrived")), "nav_steps": nav_brief.get("steps"),
                          "bc2": (summaries[-1].get("bc2") or {}).get("outcome")})
            if "safety" in summaries[-1]:
                safety = summaries[-1]["safety"]
                brief["safety"] = {label: [v["max_N"], v["impulse_Ns"], v["min_clearance_finger_m"], v.get("commanded_min_clearance_m")]
                                   for label, v in safety.items()}
                brief["cf"] = {k: summaries[-1]["safety_flags"][k] for k in ("bc1_collision_free_success", "cf_e2e")}
            print(f"[eval] episode {episode}: {json.dumps(brief)}", flush=True)
    finally:
        if bridge is not None:
            bridge.close()
        if collector is not None:
            if collector["recorded"]:
                collector["recorder"].finalize()
            else:
                collector["recorder"].abort()
        outcomes = [s["outcome"] for s in summaries]
        aggregate = {
            "episodes": len(summaries),
            "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
            "handoff_success_rate": outcomes.count("handoff_reached") / max(1, len(summaries)),
            "bad_orientation_rate": outcomes.count("bad_orientation") / max(1, len(summaries)),
            "overwritten_steps_before_latch_total": sum(s.get("overwritten_steps_before_latch", 0) for s in summaries),
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps({"config": config, "aggregate": aggregate, "episodes": summaries}, indent=2, default=str) + "\n"
        )
        print("[eval] aggregate " + json.dumps(aggregate), flush=True)
        env.close()


def wrist_pose_block(robot, pelvis, left_wrist, right_wrist) -> np.ndarray:
    """Measured wrist poses in the pelvis frame as Arena's 14-D block (position + xyzw quaternion)."""
    from isaaclab.utils import math as math_utils

    block = np.zeros(14, dtype=np.float32)
    pelvis_pos = robot.data.body_link_pos_w[0, pelvis]
    pelvis_inv = math_utils.quat_inv(robot.data.body_link_quat_w[0, pelvis].unsqueeze(0))[0]
    for body, pos_slice, quat_slice in ((left_wrist, slice(0, 3), slice(3, 7)), (right_wrist, slice(7, 10), slice(10, 14))):
        rel = math_utils.quat_apply(pelvis_inv.unsqueeze(0), (robot.data.body_link_pos_w[0, body] - pelvis_pos).unsqueeze(0))[0]
        quat = math_utils.quat_mul(pelvis_inv.unsqueeze(0), robot.data.body_link_quat_w[0, body].unsqueeze(0))[0]
        block[pos_slice] = rel.detach().cpu().numpy()
        wxyz = quat.detach().cpu().numpy()
        block[quat_slice] = wxyz[[1, 2, 3, 0]]
    return block


def body_matrix(robot, body: int) -> np.ndarray:
    """Measured world pose of one body as a 4x4 matrix."""
    pose = body_pose_wxyz(robot, body).detach().cpu().numpy().astype(np.float64)
    return pose_matrix_wxyz(pose)


def action_wrists_world(pelvis_world: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    """World wrist targets of a 23-D Arena action (pelvis-frame position + xyzw quaternion) for a given pelvis pose."""
    from scipy.spatial.transform import Rotation

    out = {}
    for side, pos, quat in (("left", slice(2, 5), slice(5, 9)), ("right", slice(9, 12), slice(12, 16))):
        local = np.eye(4)
        local[:3, :3] = Rotation.from_quat(np.asarray(action[quat], dtype=np.float64)).as_matrix()
        local[:3, 3] = action[pos]
        out[side] = pelvis_world @ local
    return out


def world_wrists_block(pelvis_world: np.ndarray, left_world: np.ndarray, right_world: np.ndarray) -> np.ndarray:
    """World wrist targets -> Arena's 14-D pelvis-frame block (position + xyzw quaternion), like ArenaLiftTrajectory."""
    from scipy.spatial.transform import Rotation

    inverse = np.linalg.inv(pelvis_world)
    block = np.zeros(14, dtype=np.float32)
    for world, offset in ((left_world, 0), (right_world, 7)):
        local = inverse @ world
        block[offset:offset + 3] = local[:3, 3]
        block[offset + 3:offset + 7] = Rotation.from_matrix(local[:3, :3]).as_quat()
    return block


_LIFTER = None
_SOURCE_OBSERVED = None


def source_realized_paths(trajectory, retargeted) -> dict[str, np.ndarray]:
    """The source demo's observed (realized) wrist path, translated with the Box like the commanded path (Mimic
    TRANSLATE).  Used as the predicted realized path of the Teacher's commands (the Pink/Homie WBC sags 3-5 cm)."""
    global _SOURCE_OBSERVED
    if _SOURCE_OBSERVED is None:
        from unitree_rl_lab.utils.teacher.arena_dataset import load_episode_trajectory

        source = load_episode_trajectory(trajectory.path, "demo_0")
        _SOURCE_OBSERVED = {"left": np.asarray(source.left_world_wrist_observed, dtype=np.float64),
                            "right": np.asarray(source.right_world_wrist_observed, dtype=np.float64)}
    delta = np.asarray(retargeted.object_world[0])[:3, 3] - trajectory.box_world[0][:3, 3]
    out = {}
    for side in ("left", "right"):
        moved = _SOURCE_OBSERVED[side].copy()
        moved[:, :3, 3] += delta
        out[side] = moved
    return out


def block_matrices(pelvis_world: np.ndarray, block: np.ndarray) -> dict[str, np.ndarray]:
    """Arena 14-D pelvis-frame wrist block -> world 4x4 poses for the given pelvis pose."""
    action = np.zeros(23, dtype=np.float64)
    action[2:16] = block
    return action_wrists_world(pelvis_world, action)


def clearance_lifter(env):
    """ClearanceLifter for the current table pose (open-hand geometry is built once)."""
    global _LIFTER
    from hand_clearance import table_boxes
    from safe_paths import ClearanceLifter

    centers, half = table_boxes(*SafetyMonitor.table_centers(env))
    if _LIFTER is None:
        _LIFTER = ClearanceLifter(centers, half)
    else:
        _LIFTER.centers, _LIFTER.half = centers, half
    return _LIFTER


def run_alignment(env, robot, bridge, box, pelvis, left_wrist, right_wrist, trajectory, monitor, recorder, episode, start_cmd):
    """Method A: move the open hands from the post-navigation pose to the Teacher's first wrist target through the WBC.

    Commanded path: min-jerk from the current command (``start_cmd``, world) to the Teacher's step-0 command; predicted
    realized path: min-jerk from the measured wrists to the source demo's realized step-0 wrists.  The lift that keeps
    the predicted realized open hands ``--safe_margin`` away from the table is added to the commanded path.
    """
    from safe_paths import plan_tracked

    box_pose7 = torch.cat((box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])).detach().cpu().numpy()
    retargeted = trajectory.retarget(pose_matrix_wxyz(box_pose7))
    realized = source_realized_paths(trajectory, retargeted)
    steps = int(math.ceil(args_cli.align_seconds / env.step_dt))
    goal_cmd = {"left": retargeted.left_wrist_world[0], "right": retargeted.right_wrist_world[0]}
    goal_act = {side: realized[side][0] for side in ("left", "right")}
    start_act = {"left": body_matrix(robot, left_wrist), "right": body_matrix(robot, right_wrist)}
    base_cmd = {side: np.repeat(goal_cmd[side][None], steps, axis=0) for side in ("left", "right")}
    base_act = {side: np.repeat(goal_act[side][None], steps, axis=0) for side in ("left", "right")}
    margin = args_cli.safe_margin if args_cli.align_margin is None else args_cli.align_margin
    paths, predicted, info = plan_tracked(start_cmd, start_act, base_cmd, base_act, steps, np.ones(steps, dtype=bool),
                                          clearance_lifter(env), margin)
    servo = None
    if args_cli.wrist_servo_gain > 0.0:
        from safe_paths import WristServo

        rot_gain = args_cli.wrist_servo_gain if args_cli.wrist_servo_rot_gain is None else args_cli.wrist_servo_rot_gain
        servo = WristServo(start_cmd, start_act, args_cli.wrist_servo_gain, rot_gain, max_rot=math.radians(args_cli.wrist_servo_max_rot_deg))
    info = {k: v for k, v in info.items() if k != "lift_profile_m"} | {"max_lift_step": int(np.argmax(info["lift_profile_m"]))}
    info["start_to_goal_cmd_m"] = {side: round_list(goal_cmd[side][:3, 3] - start_cmd[side][:3, 3], 4) for side in ("left", "right")}
    info["start_to_goal_realized_m"] = {side: round_list(goal_act[side][:3, 3] - start_act[side][:3, 3], 4) for side in ("left", "right")}
    goal = goal_act
    fell = False
    for k in range(steps + args_cli.align_hold_steps):
        target = {side: paths[side][min(k, steps - 1)] for side in ("left", "right")}
        if servo is not None:
            measured = {"left": body_matrix(robot, left_wrist), "right": body_matrix(robot, right_wrist)}
            target = servo.command({side: predicted[side][min(k, steps - 1)] for side in ("left", "right")}, measured)
        wrists = world_wrists_block(body_matrix(robot, pelvis), target["left"], target["right"])
        names, values = bridge.step(bridge_request(env, pre_phase_command(wrists, (0.0, 0.0, 0.0), 0.75)))
        raw_action, absolute = action_order_target(env, names, values)
        env.step(raw_action)
        if monitor is not None:
            monitor.after_step(env, robot, "align", cmd=np.zeros(3, dtype=np.float32), commanded=target,
                               joint_target=absolute.detach().cpu().numpy()[0])
        tilt = tilt_of(robot)
        recorder.write({"ep": episode, "phase": "align", "k": k, "tilt": round(tilt, 5)})
        if tilt > 0.8:
            fell = True
            break
    final = {"left": body_matrix(robot, left_wrist), "right": body_matrix(robot, right_wrist)}
    info["final_wrist_error_cm"] = {side: round(100.0 * float(np.linalg.norm(final[side][:3, 3] - goal[side][:3, 3])), 2) for side in ("left", "right")}
    info["final_wrist_angle_error_deg"] = {
        side: round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(final[side][:3, :3].T @ goal[side][:3, :3]) - 1.0) / 2.0)))), 2)
        for side in ("left", "right")}
    info["steps"] = steps
    if servo is not None:
        info["servo_final"] = servo.state()
    return info, fell, target, servo


def pre_phase_command(wrists: np.ndarray, velocity, height: float = 0.75) -> np.ndarray:
    """Open hands, given wrist poses, given base velocity and height, zero torso RPY."""
    action = np.zeros(23, dtype=np.float32)
    action[2:16] = wrists
    action[16:19] = velocity
    action[19] = height
    return action


def compute_observation(env) -> np.ndarray:
    if hasattr(env, "task_phase"):
        env.task_phase[0] = 0
    env.command_manager.compute(0.0)
    observation = env.observation_manager.compute(update_history=False)
    obs = student_observation(observation).detach().cpu().numpy()
    command = env.command_manager.get_term("base_velocity").command.detach().cpu().numpy()
    return replace_velocity_command(obs, command), command[0]


def safe_abort(env, robot, bridge, sent, monitor, pelvis) -> dict:
    """Phase 10 safe stop when the cage did not close in time: BC-2 is not started; both wrist commands of the last BC-1
    action are raised by --gate_abort_lift_m over 10 steps (base velocity 0) and held.  Monitored as BC-1 steps, so any
    table contact here counts as a BC-1 contact."""
    base = np.asarray(sent, dtype=np.float32).copy()
    base[:, 16:19] = 0.0
    for k in range(args_cli.gate_abort_hold_steps):
        command = base.copy()
        lift = args_cli.gate_abort_lift_m * min(1.0, (k + 1) / 10.0)
        command[:, 4] += lift   # left_wrist_position_z (pelvis frame)
        command[:, 11] += lift  # right_wrist_position_z
        commanded = action_wrists_world(body_matrix(robot, pelvis), command[0]) if args_cli.safety_metrics else None
        names, values = bridge.step(bridge_request(env, command[0]))
        raw_action, absolute = action_order_target(env, names, values)
        env.step(raw_action)
        if monitor is not None:
            monitor.after_step(env, robot, "bc1", cmd=command[0, 16:19], commanded=commanded, joint_target=absolute.detach().cpu().numpy()[0])
    return {"steps": int(args_cli.gate_abort_hold_steps), "lift_m": float(args_cli.gate_abort_lift_m), "final_tilt_rad": round(tilt_of(robot), 4)}


def cage_width(robot, box, left_tips, right_tips) -> float:
    """Distance between the left and right fingertip centres along the Box y axis (Box yaw; the Box stays flat) [m]."""
    q = box.data.root_link_quat_w[0].detach().cpu().numpy()
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
    d = (robot.data.body_link_pos_w[0, left_tips].mean(dim=0) - robot.data.body_link_pos_w[0, right_tips].mean(dim=0)).detach().cpu().numpy()
    return float(-math.sin(yaw) * d[0] + math.cos(yaw) * d[1])


def gate_geometry(robot, box_position, left_tips, right_tips, left_wrist, right_wrist):
    lt = robot.data.body_link_pos_w[0, left_tips].detach().cpu().numpy()
    rt = robot.data.body_link_pos_w[0, right_tips].detach().cpu().numpy()
    lw = robot.data.body_link_pos_w[0, left_wrist].detach().cpu().numpy()
    rw = robot.data.body_link_pos_w[0, right_wrist].detach().cpu().numpy()
    ready = bool(
        bimanual_pregrasp_ready(
            box_position, lt, rt,
            maximum_tip_distance_m=args_cli.handoff_tip_distance_limit_m,
            maximum_cage_midpoint_error_m=args_cli.handoff_cage_midpoint_limit_m,
        )[0]
        and bimanual_wrists_centered(
            box_position, lw, rw,
            maximum_midpoint_error_m=args_cli.handoff_wrist_midpoint_limit_m,
            maximum_distance_asymmetry_m=args_cli.handoff_wrist_asymmetry_limit_m,
        )[0]
    )
    ld = np.linalg.norm(lt - box_position, axis=-1)
    rd = np.linalg.norm(rt - box_position, axis=-1)
    cage = 0.5 * (lt[int(np.argmin(ld))] + rt[int(np.argmin(rd))])
    geometry = {
        "left_tip": float(ld.min()),
        "right_tip": float(rd.min()),
        "cage_mid": float(np.linalg.norm(cage - box_position)),
        "wrist_mid": float(np.linalg.norm(0.5 * (lw + rw) - box_position)),
        "wrist_asym": float(abs(np.linalg.norm(lw - box_position) - np.linalg.norm(rw - box_position))),
    }
    return ready, geometry


def tilt_of(robot) -> float:
    gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()
    return float(math.acos(max(-1.0, min(1.0, -float(gravity[2])))))


_NAV_POLICY = None
_ALIGN_TRAJECTORY = None


def teleport_polar(env, robot, episode: int) -> None:
    """Place the robot like tasks/navigation reset_box_and_robot (around the target point, any heading)."""
    from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

    target, theta = nav_mdp.target_point(env)
    target = target[0].detach().cpu().numpy()
    theta = float(theta[0])
    if args_cli.start_fixed is not None:
        r, bearing_deg, heading_deg = args_cli.start_fixed[3 * episode: 3 * episode + 3]
        back = theta + math.pi + math.radians(bearing_deg)
        start = target + r * np.array([math.cos(back), math.sin(back)])
        yaw = math.atan2(target[1] - start[1], target[0] - start[0]) + math.radians(heading_deg)
    else:
        r_lo, r_hi, bearing_deg, yaw_max = args_cli.start_polar
        rng = np.random.default_rng([args_cli.seed, episode, 11])
        r = rng.uniform(r_lo, r_hi)
        back = theta + math.pi + math.radians(bearing_deg) * rng.uniform(-1.0, 1.0)
        start = target + r * np.array([math.cos(back), math.sin(back)])
        yaw = math.atan2(target[1] - start[1], target[0] - start[0]) + yaw_max * rng.uniform(-1.0, 1.0)
    root = robot.data.default_root_state[0:1].clone()
    root[0, 0:2] = torch.as_tensor(start, dtype=torch.float32, device=env.device)
    root[0, 2] += env.scene.env_origins[0, 2]
    root[0, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=env.device)
    root[0, 7:] = 0.0
    env_ids = torch.tensor([0], device=env.device)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def _actor_input_dim(policy) -> int:
    for name, tensor in policy.state_dict().items():
        if name.endswith("weight") and tensor.dim() == 2:
            return int(tensor.shape[1])
    raise ValueError("cannot infer the navigation actor input width")


def ppo_navigation(env, robot, pre_step, episode: int):
    """PPO navigation through the Arena WBC until the BC-1 handoff state holds 25 steps (or timeout/fall)."""
    global _NAV_POLICY
    from isaaclab.envs import mdp as base_mdp

    from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

    from unitree_rl_lab.tasks.navigation.homie_action import NAV_ACTION_PROFILES, heading_gate_gain, process_nav_command, stop_gate_state

    if _NAV_POLICY is None:
        _NAV_POLICY = torch.jit.load(str(args_cli.nav_policy), map_location=env.device).eval()
        nav_mdp.apply_nav_target_overrides(args_cli.nav_target_offset_x, args_cli.nav_handoff_x_range)
    table_safe = None
    if args_cli.table_safe_gap is not None:
        # Phase 7: table-edge-bounded stop from the GT table pose (approach frame theta = 0 -> world x).  The window front
        # edge keeps the base >= gap behind the table front edge; the target sits table_safe_margin behind that edge.
        base_target = nav_mdp.TARGET_OFFSET_X_M if args_cli.nav_target_offset_x is None else args_cli.nav_target_offset_x
        base_window = tuple(nav_mdp.HANDOFF_X_RANGE) if args_cli.nav_handoff_x_range is None else tuple(args_cli.nav_handoff_x_range)
        edge_x = float(env.scene["table_top"].data.root_link_pos_w[0, 0]) - nav_mdp.TABLE_HALF_EXTENT_M
        canonical_x = float(nav_mdp.canonical_start(env)[0][0, 0])
        dx_max = edge_x - args_cli.table_safe_gap - canonical_x
        front = min(base_window[1], dx_max)
        target = min(base_target, front - args_cli.table_safe_margin)
        table_safe = {"gap_m": args_cli.table_safe_gap, "margin_m": args_cli.table_safe_margin, "table_edge_x": round(edge_x, 4),
                      "canonical_x": round(canonical_x, 4), "dx_max_m": round(dx_max, 4), "target_offset_x_m": round(target, 4),
                      "handoff_x_range": [base_window[0], round(front, 4)], "feasible": bool(target >= base_window[0] and front > base_window[0])}
        nav_mdp.apply_nav_target_overrides(target, (base_window[0], front))
    profile = NAV_ACTION_PROFILES[args_cli.nav_action_profile]
    v6 = args_cli.nav_action_profile in ("v6", "v7")  # 21-D observation (+ the command Homie received)
    stable = args_cli.nav_handoff_criterion == "stable50" or (args_cli.nav_handoff_criterion == "profile" and v6)
    heading_gate = profile.get("heading_gate")
    gate_release = profile.get("gate_release")
    scale = torch.tensor(args_cli.nav_policy_scale or profile["scale"], device=env.device)
    scale_negative = None if profile["scale_negative"] is None else torch.tensor(profile["scale_negative"], device=env.device)
    rate_limit = None if profile["rate_limit"] is None else torch.tensor(profile["rate_limit"], device=env.device)
    expected_inputs = 21 if v6 else 18
    if _actor_input_dim(_NAV_POLICY) not in (expected_inputs, 17):
        raise ValueError(f"policy input width {_actor_input_dim(_NAV_POLICY)} does not match profile {args_cli.nav_action_profile}")
    width = 4 if _actor_input_dim(_NAV_POLICY) in (18, 21) else 3  # action width (3, or 4 with the stop gate)
    last = torch.zeros(1, width, device=env.device)
    command = torch.zeros(1, 3, device=env.device)  # the command Homie received at the last step (rate-limit state)
    gate_on = torch.zeros(1, dtype=torch.bool, device=env.device)  # v7 stop-gate latch (hysteresis state)
    hold = 0
    hold_steps = 50 if stable else 25
    nav = {"controller": "ppo", "policy": str(args_cli.nav_policy), "handoff": False, "steps": 0, "action_profile": args_cli.nav_action_profile,
           "handoff_criterion": "stable50" if stable else "loose25",
           "handoff_hold_steps": hold_steps, "target_offset_x_m": nav_mdp.TARGET_OFFSET_X_M, "handoff_x_range": list(nav_mdp.HANDOFF_X_RANGE),
           "table_safe": table_safe}
    start_errors = nav_mdp.handoff_errors(env)[0].tolist()
    start_distance = float(nav_mdp.target_errors(env)[0][0])
    arm_ids = [i for i, n in enumerate(robot.data.joint_names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    leg_ids = [i for i, n in enumerate(robot.data.joint_names) if any(k in n for k in ("hip", "knee", "ankle"))]

    def policy_step(phase: str):
        """One control step: GT observation -> PPO actor (deterministic) -> [vx, vy, wz] (stop gate) -> Arena WBC.

        The action -> command mapping is the training env's (homie_action.process_nav_command, same profile)."""
        nonlocal last, command, gate_on
        terms = [nav_mdp.gt_goal_pos_b(env), nav_mdp.gt_goal_heading(env), nav_mdp.gt_goal_distance(env),
                 base_mdp.base_lin_vel(env), base_mdp.base_ang_vel(env), base_mdp.projected_gravity(env), last]
        if v6:
            terms.append(command)
        obs = torch.cat(terms, dim=-1)
        with torch.inference_mode():
            action = torch.clamp(_NAV_POLICY(obs), -1.0, 1.0)
        last = action.clone()
        planar_gain = None if heading_gate is None else heading_gate_gain(terms[0], *heading_gate)
        gate = None
        if gate_release is not None and action.shape[1] == 4:
            gate_on = stop_gate_state(action[:, 3], gate_on, gate_release)
            gate = gate_on
        command = process_nav_command(action, command, scale, scale_negative, profile["ellipse"], action.shape[1] == 4, rate_limit, env.step_dt,
                                      gate_on=gate, planar_gain=planar_gain)
        stop = action.shape[1] == 4 and (bool(gate_on[0]) if gate is not None else float(action[0, 3]) > 0.0)
        tilt_now = pre_step(phase, command[0].detach().cpu().numpy(), ppo_obs=obs[0].detach().cpu().numpy(),
                            ppo_action=action[0].detach().cpu().numpy())
        e = nav_mdp.handoff_errors(env)[0]
        speed = float(torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[0, :2]))
        yaw_rate = abs(float(robot.data.root_link_ang_vel_b[0, 2]))
        arm_dev = float(torch.abs(robot.data.joint_pos[0, arm_ids]).max())
        in_window = (nav_mdp.HANDOFF_X_RANGE[0] <= float(e[0]) <= nav_mdp.HANDOFF_X_RANGE[1] and abs(float(e[1])) <= nav_mdp.HANDOFF_Y_MAX
                     and abs(float(e[2])) <= nav_mdp.HANDOFF_YAW_MAX and tilt_now <= 0.20 and arm_dev <= 0.25)
        if stable:  # StableHandoffSuccess of the v6/v7 training env
            leg_speed = float(torch.linalg.vector_norm(robot.data.joint_vel[0, leg_ids]))
            inside = (in_window and speed <= 0.05 and yaw_rate <= 0.10 and leg_speed <= 0.3
                      and float(torch.linalg.vector_norm(command[0])) < 1.0e-6)
        else:  # HandoffSuccess of the v5 training env
            inside = in_window and speed <= 0.10 and yaw_rate <= 0.20
        return tilt_now > 0.8, inside, stop, speed, yaw_rate, tilt_now

    fell = False
    for step in range(args_cli.nav_max_steps):
        fell, inside, *_ = policy_step("ppo_nav")
        if fell:
            break
        nav["steps"] = step + 1
        hold = hold + 1 if inside else 0
        if hold >= hold_steps:
            nav["handoff"] = True
            break
    nav["handoff_errors_at_handoff"] = round_list(nav_mdp.handoff_errors(env)[0].tolist(), 4)
    if nav["handoff"] and args_cli.post_handoff_hold_steps:
        # Stop-stability test: the PPO policy stays in control after the handoff condition was met.
        anchor = robot.data.root_link_pos_w[0, :2].clone()
        anchor_yaw = rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2]
        post = {"steps": 0, "stop_steps": 0, "inside_steps": 0, "max_speed": 0.0, "max_yaw_rate": 0.0, "max_tilt": 0.0,
                "max_drift_m": 0.0, "max_yaw_drift_rad": 0.0}
        for _ in range(args_cli.post_handoff_hold_steps):
            fell, inside, stop, speed, yaw_rate, tilt_now = policy_step("ppo_hold")
            drift = float(torch.linalg.vector_norm(robot.data.root_link_pos_w[0, :2] - anchor))
            yaw_drift = abs(wrap_to_pi(rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2] - anchor_yaw))
            post.update({"steps": post["steps"] + 1, "stop_steps": post["stop_steps"] + int(stop), "inside_steps": post["inside_steps"] + int(inside),
                         "max_speed": max(post["max_speed"], speed), "max_yaw_rate": max(post["max_yaw_rate"], yaw_rate),
                         "max_tilt": max(post["max_tilt"], tilt_now), "max_drift_m": max(post["max_drift_m"], drift),
                         "max_yaw_drift_rad": max(post["max_yaw_drift_rad"], yaw_drift)})
            if fell:
                break
        nav["post_handoff_hold"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in post.items()}
        nav["post_handoff_hold"]["fell"] = fell
    nav["start_handoff_errors"] = round_list(start_errors, 4)
    nav["start_distance_m"] = round(start_distance, 3)
    nav["final_handoff_errors"] = round_list(nav_mdp.handoff_errors(env)[0].tolist(), 4)
    nav["fell"] = fell
    return nav, fell


def run_episode(env, robot, bridge, recorder, policy, mean, scale, teacher, episode,
                left_tips, right_tips, left_wrist, right_wrist, pelvis, arm_joint_ids,
                upper_joint_ids, feature_names, finger_dims, gravity_dims, bin_dims, collector=None,
                bc2_policy=None, monitor=None) -> dict:
    started = time.monotonic()
    if args_cli.robot_start_list is not None:
        dx, dy, dyaw = args_cli.robot_start_list[3 * episode: 3 * episode + 3]
        env.event_manager.get_term_cfg("reset_base").params["pose_range"] = {"x": (dx, dx), "y": (dy, dy), "yaw": (dyaw, dyaw)}
    if args_cli.reseed_each_episode:
        episode_seed = int(np.random.default_rng([args_cli.seed, episode, 99]).integers(0, 2**31 - 1))
        torch.manual_seed(episode_seed)
        np.random.seed(episode_seed)
    env.reset()
    actuator = robot.actuators["hands"]
    actuator.stiffness[:] = float(args_cli.hand_stiffness)
    robot.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
    if args_cli.arm_joint_offset_rad != 0.0:
        joint_pos = robot.data.joint_pos.clone()
        joint_pos[0, arm_joint_ids] += float(args_cli.arm_joint_offset_rad)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    if args_cli.start_polar is not None or args_cli.start_fixed is not None:
        teleport_polar(env, robot, episode)
    if bridge is not None:
        bridge.reset()
    box = env.scene["box"]
    initial_box = box.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    initial_root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    initial_yaw = rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2]
    if monitor is not None:
        start_text = ""
        if args_cli.start_fixed is not None:
            r, bearing, heading = args_cli.start_fixed[3 * episode: 3 * episode + 3]
            start_text = f" | start {r:.1f} m, bearing {bearing:+.0f} deg, heading {heading:+.0f} deg"
        monitor.begin(env, robot, episode, (args_cli.video_label or "G1") + start_text)

    # Optional pre-phase that emulates arriving from navigation (Homie walking, open hands).
    nominal_wrists = None
    last_wrist_block = [None]  # pelvis-frame wrist command of the last pre-phase step
    last_cmd_world = [None]  # world wrist command at the end of the alignment phase
    align_servo = [None]  # wrist servo state carried from the alignment into the safe Teacher path
    pre_t = 0
    request_cmd_mismatch = 0.0

    def pre_step(phase: str, velocity, ppo_obs=None, ppo_action=None) -> float:
        nonlocal nominal_wrists, pre_t, request_cmd_mismatch
        measured = wrist_pose_block(robot, pelvis, left_wrist, right_wrist)
        if nominal_wrists is None:
            nominal_wrists = measured
        wrists = nominal_wrists if args_cli.pre_upper_body == "nominal" else measured
        last_wrist_block[0] = np.asarray(wrists, dtype=np.float64).copy()
        height = args_cli.settle_height if phase == "settle" else 0.75
        request = bridge_request(env, pre_phase_command(wrists, velocity, height))
        # The velocity command that reaches the Arena worker (Homie navigate_cmd), read back from the request.
        sent_cmd = np.asarray(request["high_level_action"][16:19], dtype=np.float32)
        request_cmd_mismatch = max(request_cmd_mismatch, float(np.abs(sent_cmd - np.asarray(velocity, dtype=np.float32)).max()))
        names, values = bridge.step(request)
        raw_action, _ = action_order_target(env, names, values)
        env.step(raw_action)
        if monitor is not None:
            monitor.after_step(env, robot, phase, cmd=sent_cmd, ppo_obs=ppo_obs, ppo_action=ppo_action)
        tilt = tilt_of(robot)
        recorder.write({
            "ep": episode, "pre_t": pre_t, "phase": phase, "cmd": round_list(velocity, 4), "tilt": round(tilt, 5),
            "root": round_list(robot.data.root_link_pos_w[0].detach().cpu().numpy(), 4),
            "yaw": round(rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2], 5),
            "lin_vel_b": round_list(robot.data.root_link_lin_vel_b[0].detach().cpu().numpy(), 4),
        })
        pre_t += 1
        return tilt

    nav = None
    pre_fell = False
    if args_cli.navigate:
        target = bc1_start_pose(initial_box[:2])
        jitter = np.random.default_rng([args_cli.seed, episode, 7]).uniform(-1.0, 1.0, 3) * np.asarray(args_cli.nav_target_jitter)
        target = (target[0] + float(jitter[0]), target[1] + float(jitter[1]), target[2] + float(jitter[2]))
        table_safe = None
        if args_cli.table_safe_gap is not None:
            # Same rule as the PPO navigation: keep the base >= gap behind the table front edge (GT), the target point
            # --table_safe_margin behind that limit (the canonical start faces +x, so the limit is on world x).
            edge_x = float(env.scene["table_top"].data.root_link_pos_w[0, 0]) - 0.20
            canonical_x = float(bc1_start_pose(initial_box[:2])[0])
            limit = edge_x - args_cli.table_safe_gap - args_cli.table_safe_margin
            table_safe = {"edge_x": round(edge_x, 4), "dx_limit_m": round(limit - canonical_x, 4), "clamped": target[0] > limit}
            target = (min(target[0], limit), target[1], target[2])
        gains = NavGains(min_command_norm=args_cli.nav_min_command_norm)
        nav = {"target_xy_yaw": round_list(target, 4), "target_jitter": round_list(jitter, 4), "arrived": False, "steps": 0,
               "table_safe": table_safe}
        for nav_step in range(args_cli.nav_max_steps):
            root_xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
            yaw = rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2]
            error = pose_error_body(root_xy, yaw, target[:2], target[2])
            if math.hypot(error[0], error[1]) <= args_cli.nav_position_tolerance_m and abs(error[2]) <= args_cli.nav_yaw_tolerance_rad:
                nav["arrived"] = True
                break
            nav["steps"] = nav_step + 1
            if pre_step("nav", nav_command(root_xy, yaw, target[:2], target[2], gains)) > 0.8:
                pre_fell = True
                break
        nav["stop_error_body"] = round_list(error, 4)
    if args_cli.nav_policy is not None:
        nav, pre_fell = ppo_navigation(env, robot, pre_step, episode)
    settle_steps = args_cli.settle_steps
    if args_cli.settle_steps_range is not None:
        settle_steps = int(np.random.default_rng([args_cli.seed, episode]).integers(args_cli.settle_steps_range[0], args_cli.settle_steps_range[1] + 1))
    for settle_t in range(0 if pre_fell else args_cli.prewalk_steps + settle_steps):
        prewalk = settle_t < args_cli.prewalk_steps
        if pre_step("prewalk" if prewalk else "settle", args_cli.prewalk_command if prewalk else (0.0, 0.0, 0.0)) > 0.8:
            pre_fell = True
            break
    for hold_t in range(0 if pre_fell else args_cli.pre_default_hold_steps):
        env.step(torch.zeros((1, env.action_manager.total_action_dim), device=env.device))
        if monitor is not None:
            monitor.after_step(env, robot, "default_hold")
        tilt = tilt_of(robot)
        recorder.write({
            "ep": episode, "pre_t": pre_t, "phase": "default_hold", "tilt": round(tilt, 5),
            "root": round_list(robot.data.root_link_pos_w[0].detach().cpu().numpy(), 4),
            "yaw": round(rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())[2], 5),
            "lin_vel_b": round_list(robot.data.root_link_lin_vel_b[0].detach().cpu().numpy(), 4),
        })
        pre_t += 1
        if tilt > 0.8:
            pre_fell = True
            break
    align_info = None
    if args_cli.pre_bc1_align != "none" and not pre_fell:
        measured_now = {"left": body_matrix(robot, left_wrist), "right": body_matrix(robot, right_wrist)}
        align_start_cmd = block_matrices(body_matrix(robot, pelvis), last_wrist_block[0]) if last_wrist_block[0] is not None else measured_now
        align_info, pre_fell, last_cmd_world[0], align_servo[0] = run_alignment(
            env, robot, bridge, env.scene["box"], pelvis, left_wrist, right_wrist, _ALIGN_TRAJECTORY, monitor, recorder, episode,
            align_start_cmd)
    if args_cli.handoff_wbc_reset and bridge is not None:
        bridge.reset()
    start_root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    start_rpy = rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())
    start_lin_vel = robot.data.root_link_lin_vel_b[0].detach().cpu().numpy().copy()
    start_ang_vel = robot.data.root_link_ang_vel_b[0].detach().cpu().numpy().copy()
    start_box = box.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    canonical = bc1_start_pose(start_box[:2])
    cos_yaw, sin_yaw = math.cos(start_rpy[2]), math.sin(start_rpy[2])
    box_rel = start_box[:2] - start_root[:2]
    q_rel = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).detach().cpu().numpy()
    upper_q_rel = {robot.data.joint_names[i]: round(float(q_rel[i]), 4) for i in upper_joint_ids}
    start_state = {
        "start_offset_canonical": round_list(
            (start_root[0] - canonical[0], start_root[1] - canonical[1], wrap_to_pi(start_rpy[2] - canonical[2])), 4
        ),
        "start_box_in_robot_xy": round_list((cos_yaw * box_rel[0] + sin_yaw * box_rel[1], -sin_yaw * box_rel[0] + cos_yaw * box_rel[1]), 4),
        "start_ang_vel_b": round_list(start_ang_vel, 4),
        "start_arm_dev_max_rad": round(max(abs(v) for k, v in upper_q_rel.items() if "waist" not in k), 4),
        "start_waist_dev_max_rad": round(max(abs(v) for k, v in upper_q_rel.items() if "waist" in k), 4),
        "start_upper_q_rel": upper_q_rel,
        "pre_upper_body": args_cli.pre_upper_body,
        "settle_steps": settle_steps,
        "nominal_wrist_block": None if nominal_wrists is None else round_list(nominal_wrists, 4),
        "nav": nav,
        "align": align_info,
        "initial_yaw": round(initial_yaw, 5),
        "pre_phase_request_cmd_mismatch_max": request_cmd_mismatch,
    }
    if pre_fell:
        return {
            "episode": episode, "seed": args_cli.seed, "outcome": "pre_phase_fall", "steps": 0, "handoff_step": None,
            "initial_box_xy": round_list(initial_box[:2], 5), "start_root_xyz": round_list(start_root, 4),
            "start_rpy": round_list(start_rpy, 4), "start_lin_vel_b": round_list(start_lin_vel, 4), **start_state,
            "wall_time_s": round(time.monotonic() - started, 2),
        }
    reset_box = initial_box
    initial_box = start_box  # Box displacement is measured from the BC-1 start.

    max_steps = int(math.ceil(args_cli.episode_seconds / env.step_dt))
    if teacher is not None:
        max_steps = min(max_steps, teacher["target"].shape[0])
    gate_step = None
    teacher_servo = None
    teacher_servo_start = None
    predicted_path = None
    if collector is not None and args_cli.mode == "teacher_collect":
        trajectory = collector["trajectory"]
        max_steps = trajectory.close_index + args_cli.collect_extra_steps + 1
        box_pose7 = torch.cat((box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])).detach().cpu().numpy()
        retargeted = trajectory.retarget(pose_matrix_wxyz(box_pose7))
        if args_cli.teacher_path == "safe":
            import dataclasses

            from safe_paths import margin_schedule, plan_tracked

            start_act = {"left": body_matrix(robot, left_wrist), "right": body_matrix(robot, right_wrist)}
            if last_cmd_world[0] is not None:
                start_cmd = last_cmd_world[0]
            elif last_wrist_block[0] is not None:
                start_cmd = block_matrices(body_matrix(robot, pelvis), last_wrist_block[0])
            else:
                start_cmd = start_act
            length = len(retargeted.left_wrist_world)
            paths, predicted_path, safe_info = plan_tracked(start_cmd, start_act,
                                               {"left": retargeted.left_wrist_world, "right": retargeted.right_wrist_world},
                                               source_realized_paths(trajectory, retargeted), args_cli.safe_blend_steps,
                                               np.arange(length) < trajectory.close_index, clearance_lifter(env),
                                               margin_schedule(length, args_cli.safe_margin, args_cli.safe_margin_late,
                                                               trajectory.close_index, args_cli.safe_margin_late_steps),
                                               hold_after=trajectory.close_index)
            if args_cli.teacher_servo_gain > 0.0:
                from safe_paths import WristServo

                if args_cli.teacher_servo_late_steps is None:
                    teacher_servo = WristServo(start_cmd, start_act, args_cli.teacher_servo_gain, args_cli.teacher_servo_rot_gain)
                else:
                    teacher_servo_start = max(0, trajectory.close_index - args_cli.teacher_servo_late_steps)
                    teacher_servo = "pending"  # created at teacher_servo_start from the planned command and path
                paths = {side: paths[side].copy() for side in ("left", "right")}  # overwritten online by the servo commands
            retargeted = dataclasses.replace(retargeted, left_wrist_world=paths["left"], right_wrist_world=paths["right"])
            start_state["safe_teacher_path"] = {k: v for k, v in safe_info.items() if k != "lift_profile_m"} | {
                "lift_profile_m_approach": safe_info["lift_profile_m"][: trajectory.close_index]}
        box_start_height = float(start_box[2])
        collector["recorder"].start_episode(metadata={
            "episode": episode, "seed": args_cli.seed, "initial_box_pose_wxyz": box_pose7.tolist(),
            "start_root_xyz": round_list(start_root, 5), "start_rpy": round_list(start_rpy, 5), **start_state,
        })
    ready_reference = None
    ready_steps = 0
    forced_step = None if args_cli.force_handoff_step_range is None else int(np.random.default_rng([args_cli.seed, episode, 4711]).integers(
        args_cli.force_handoff_step_range[0], args_cli.force_handoff_step_range[1] + 1))
    forced = False
    readiness = None
    if args_cli.gate_max_cage_width_m is not None or args_cli.gate_min_open_steps is not None:
        readiness = {"gate_first_open_step": None, "handoff_step": None, "waited_steps": None, "fallback": False,
                     "cage_width_at_first_open_m": None, "cage_width_at_handoff_m": None, "base_gate_open_steps": 0}
    previous_target = None
    servo_offsets_max = 0.0
    outcome = "timeout"
    handoff_step = None
    overwritten_before_latch = 0
    stats = {"tilt": [], "roll": [], "pitch": [], "z": [], "zmax": [], "raw_const": [], "box_dxy": [], "contacts": [], "jump": [], "obs_diff": [], "root_diff": []}
    best_score, best_geometry = math.inf, None
    dump = {"obs": [], "raw": [], "sent": [], "tilt": [], "root": [], "rpy": []}
    for step in range(max_steps):
        obs, command = compute_observation(env)
        record = {"ep": episode, "t": step, "cmd": round_list(command)}
        commanded = None
        dump["obs"].append(obs[0].copy())
        if teacher is not None:
            recorded = teacher["observation"][step]
            diff = np.abs(obs[0] - recorded)
            record["obs_diff_max"] = float(diff.max())
            record["obs_diff_by_group"] = {k: float(diff[s].max()) for k, s in OBS_GROUP_SLICES.items()}
            stats["obs_diff"].append(float(diff.max()))
            target = torch.as_tensor(teacher["target"][step:step + 1], dtype=torch.float32, device=env.device)
            raw_action = (target - action_term_offset(env)) / action_term_scale(env)
            absolute = target
        elif collector is not None and args_cli.mode == "teacher_collect":
            # Exactly the Teacher of collect_nominal_box_teacher.py / evaluate_bc2_rollout.py, indexed from the handoff.
            grasp_scope = step >= trajectory.close_index
            if teacher_servo == "pending" and step >= teacher_servo_start:
                from safe_paths import WristServo

                index = min(step, len(predicted_path["left"]) - 1)
                source_index = min(step, trajectory.freeze_index)
                planned_cmd = {"left": retargeted.left_wrist_world[source_index].copy(), "right": retargeted.right_wrist_world[source_index].copy()}
                # start from the planned command (no jump): offset = planned command - planned realized pose
                teacher_servo = WristServo(planned_cmd, {side: predicted_path[side][index] for side in ("left", "right")},
                                           args_cli.teacher_servo_gain, args_cli.teacher_servo_rot_gain)
            if teacher_servo is not None and teacher_servo != "pending":
                # Method B with the wrist servo: the commanded wrists make the realized wrists follow the planned path;
                # the correction is frozen from the close onset on (grasp/lift keep the last offsets).
                teacher_servo.frozen = grasp_scope
                index = min(step, len(predicted_path["left"]) - 1)
                measured = {"left": body_matrix(robot, left_wrist), "right": body_matrix(robot, right_wrist)}
                servo_cmd = teacher_servo.command({side: predicted_path[side][index] for side in ("left", "right")}, measured)
                source_index = min(step, trajectory.freeze_index)
                retargeted.left_wrist_world[source_index] = servo_cmd["left"]
                retargeted.right_wrist_world[source_index] = servo_cmd["right"]
                servo_offsets_max = max(servo_offsets_max, *(float(np.linalg.norm(v)) for v in teacher_servo.p_off.values()))
            sent = trajectory.high_level_action(step, body_pose_wxyz(robot, pelvis).detach().cpu().numpy(), retargeted)
            sent = np.asarray(sent, dtype=np.float32)[None, :]
            index = min(step, trajectory.freeze_index)
            commanded = {"left": retargeted.left_wrist_world[index], "right": retargeted.right_wrist_world[index]}
            sent[:, 16:19] = 0.0 if grasp_scope else command
            if grasp_scope:
                obs = replace_velocity_command(obs, np.zeros((1, 3), dtype=np.float32))
            names, values = bridge.step(bridge_request(env, sent[0]))
            raw_action, absolute = action_order_target(env, names, values)
            record["teacher_action"] = round_list(sent[0])
        else:
            raw = np.asarray(policy(obs), dtype=np.float32)
            decoded = raw.copy()
            if args_cli.hand_state_decoding == "binary":
                decoded[:, 0:2] = (raw[:, 0:2] >= 0.5).astype(np.float32)
            sent = decoded.copy()
            if ready_reference is not None:
                sent = ready_reference.copy()
                sent[:, 16:19] = 0.0
            sent = normalize_quaternions(sent)
            intact = normalize_quaternions(decoded)
            changed = np.flatnonzero(np.abs(sent[0] - intact[0]) > 1.0e-6)
            if changed.size and ready_reference is None:
                overwritten_before_latch += 1
            z = (obs[0].astype(np.float64) - mean) / scale
            top = np.argsort(-np.abs(z))[:5]
            record.update(
                {
                    "raw": round_list(raw[0]),
                    "hand_state_sent": round_list(sent[0, 0:2], 3),
                    "sent": round_list(sent[0]),
                    "overwritten_dims": changed.tolist(),
                    "latched": ready_reference is not None,
                    "z_max": float(np.abs(z).max()),
                    "z_top": [[feature_names[i], round(float(z[i]), 2)] for i in top],
                    "z_finger_max": float(np.abs(z[finger_dims]).max()),
                    "z_gravity_max": float(np.abs(z[gravity_dims]).max()),
                    "z_bin": round_list(z[bin_dims], 3),
                }
            )
            stats["zmax"].append(float(np.abs(z).max()))
            stats["raw_const"].append(np.r_[raw[0, 0:2], raw[0, 19:23]])
            commanded = action_wrists_world(body_matrix(robot, pelvis), sent[0]) if args_cli.safety_metrics else None
            names, values = bridge.step(bridge_request(env, sent[0]))
            raw_action, absolute = action_order_target(env, names, values)
            record["wbc_input"] = round_list(sent[0])
            dump["raw"].append(raw[0].copy())
            dump["sent"].append(sent[0].copy())
        absolute_np = absolute.detach().cpu().numpy()[0]
        jump = 0.0 if previous_target is None else float(np.abs(absolute_np - previous_target).max())
        previous_target = absolute_np
        env.step(raw_action)
        if monitor is not None:
            monitor.after_step(env, robot, "bc1", cmd=None if teacher is not None else np.asarray(sent, dtype=np.float32)[0, 16:19],
                               commanded=commanded, joint_target=absolute_np)

        # Post-step state, as play_bc_skill_fsm.py evaluates its signals.
        obs_after, _ = compute_observation(env)
        box_position = box.data.root_link_pos_w[0].detach().cpu().numpy()
        ready, geometry = gate_geometry(robot, box_position, left_tips, right_tips, left_wrist, right_wrist)
        if args_cli.gate_max_tip_height_m is not None:
            table_top_z = float(env.scene["table_top"].data.root_link_pos_w[0, 2]) + 0.04
            middles = [left_tips[2], right_tips[2]]  # *_hand_middle_1_link
            tip_height = float(robot.data.body_link_pos_w[0, middles, 2].mean()) - table_top_z
            geometry["tip_height"] = tip_height
            ready = ready and tip_height <= args_cli.gate_max_tip_height_m
        if readiness is not None and teacher is None:
            base_ready = ready
            if args_cli.gate_max_cage_width_m is not None:
                geometry["cage_width"] = cage_width(robot, box, left_tips, right_tips)
            if base_ready:
                readiness["base_gate_open_steps"] += 1
                if readiness["gate_first_open_step"] is None:
                    readiness["gate_first_open_step"] = step
                    readiness["cage_width_at_first_open_m"] = geometry.get("cage_width")
            first = readiness["gate_first_open_step"]
            waited = None if first is None else step - first
            extra = True
            if args_cli.gate_max_cage_width_m is not None:
                extra = extra and geometry["cage_width"] <= args_cli.gate_max_cage_width_m
            if args_cli.gate_min_open_steps is not None:
                extra = extra and waited is not None and waited + 1 >= args_cli.gate_min_open_steps
            fallback = (args_cli.gate_ready_max_wait_steps is not None and waited is not None
                        and waited >= args_cli.gate_ready_max_wait_steps)
            if fallback and not (base_ready and extra) and args_cli.gate_ready_timeout_action == "abort":
                readiness.update({"timeout_abort": True, "waited_steps": waited, "cage_width_at_timeout_m": geometry.get("cage_width"),
                                  "base_gate_at_timeout": bool(base_ready)})
                ready = False
            else:
                ready = bool((base_ready and extra) or fallback)
                if ready:
                    readiness.update({"handoff_step": step, "waited_steps": waited, "fallback": bool(fallback and not (base_ready and extra)),
                                      "cage_width_at_handoff_m": geometry.get("cage_width"), "base_gate_at_handoff": bool(base_ready)})
        gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()
        tilt = float(math.acos(max(-1.0, min(1.0, -float(gravity[2])))))
        root = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        rpy = rpy_from_wxyz(robot.data.root_link_quat_w[0].detach().cpu().numpy())
        left_count, right_count = box_contact_counts(env)
        box_dxy = float(np.linalg.norm(box_position[:2] - initial_box[:2]))
        score = max(geometry[k] / lim for k, lim in (
            ("left_tip", args_cli.handoff_tip_distance_limit_m), ("right_tip", args_cli.handoff_tip_distance_limit_m),
            ("cage_mid", args_cli.handoff_cage_midpoint_limit_m), ("wrist_mid", args_cli.handoff_wrist_midpoint_limit_m),
            ("wrist_asym", args_cli.handoff_wrist_asymmetry_limit_m)))
        if score < best_score:
            best_score, best_geometry = score, geometry
        record.update(
            {
                "tilt": round(tilt, 5),
                "rpy": round_list(rpy),
                "root": round_list(root),
                "lin_vel_b": round_list(robot.data.root_link_lin_vel_b[0].detach().cpu().numpy()),
                "gate": {k: round(v, 4) for k, v in geometry.items()},
                "gate_ready": ready,
                "contacts": [left_count, right_count],
                "box_dxy": round(box_dxy, 5),
                "wbc_target_jump": round(jump, 5),
                "box_visible_after": bool(obs_after[0, STUDENT_FEATURE_SLICES["box_visible"].start] > 0.5),
            }
        )
        if teacher is not None:
            root_diff = float(np.abs(root - teacher["root_pose"][step, :3]).max())
            record["root_pos_diff_vs_recorded"] = root_diff
            stats["root_diff"].append(root_diff)
        recorder.write(record)
        dump["tilt"].append(tilt)
        dump["root"].append(root.copy())
        dump["rpy"].append(rpy)
        for key, value in (("tilt", tilt), ("roll", rpy[0]), ("pitch", rpy[1]), ("z", float(root[2])), ("box_dxy", box_dxy), ("jump", jump)):
            stats[key].append(value)
        stats["contacts"].append(left_count + right_count)

        if collector is not None and args_cli.mode == "teacher_collect":
            box_height = float(box_position[2])
            collector["recorder"].append(TeacherDemoStep(
                student_observation=obs[0],
                teacher_high_level_action=sent[0],
                processed_43d_joint_target=absolute.detach().cpu().numpy()[0],
                phase_id=0,
                box_pose=box.data.root_link_state_w[0, :7].detach().cpu().numpy(),
                bin_pose=collector["bin_pose"],
                root_pose=robot.data.root_link_state_w[0, :7].detach().cpu().numpy(),
                box_size=collector["box_size"],
                left_contact_count=left_count,
                right_contact_count=right_count,
                left_wrist_pose=body_pose_wxyz(robot, left_wrist).detach().cpu().numpy(),
                right_wrist_pose=body_pose_wxyz(robot, right_wrist).detach().cpu().numpy(),
                left_fingertip_positions=robot.data.body_link_pos_w[0, left_tips].detach().cpu().numpy(),
                right_fingertip_positions=robot.data.body_link_pos_w[0, right_tips].detach().cpu().numpy(),
                lift=bool(box_height >= box_start_height + 0.12),
                drop=bool(box_height < box_start_height - 0.10),
                success=False,
            ))
            if ready and gate_step is None and step < trajectory.close_index:
                gate_step = step
        if tilt > 0.8:
            outcome = "bad_orientation"
            break
        if collector is not None and args_cli.mode == "teacher_collect":
            if step + 1 >= max_steps:
                outcome = "handoff_reached" if gate_step is not None else "timeout"
                handoff_step = gate_step
        elif teacher is None:
            if readiness is not None and readiness.get("timeout_abort"):
                outcome = "readiness_timeout"
                break
            if forced_step is not None and step >= forced_step and not ready and ready_reference is None:
                ready, forced = True, True  # BC-2 data collection only (--force_handoff_step_range)
            if ready and ready_reference is None:
                ready_reference = sent.copy()
                ready_reference[:, 16:19] = 0.0
            if ready_reference is not None:
                ready_steps += 1
                if ready_steps >= args_cli.approach_ready_hold_steps:
                    outcome = "handoff_reached"
                    handoff_step = step
                    break

    if outcome == "readiness_timeout":
        readiness["abort"] = safe_abort(env, robot, bridge, sent, monitor, pelvis)

    bc2 = None
    if args_cli.then_teacher and outcome == "handoff_reached":
        bc2 = teacher_continuation(env, robot, bridge, box, pelvis, left_wrist, right_wrist, left_tips, right_tips,
                                   start_box, handoff_step, collector, episode, last_sent=np.asarray(sent, dtype=np.float64)[0],
                                   monitor=monitor)
    if bc2_policy is not None and outcome == "handoff_reached":
        # Identical to evaluate_bc2_rollout.py after its Teacher handoff.
        term = env.action_manager.get_term("JointPositionAction")
        hold_required = int(math.ceil(args_cli.bc2_hold_seconds / env.step_dt))
        bc2 = {"outcome": "timeout", "steps": 0, "max_lift_m": 0.0, "max_hold_steps": 0, "max_tilt_rad": 0.0}
        hold = 0
        for k in range(int(math.ceil(args_cli.bc2_seconds / env.step_dt))):
            obs_bc2, _ = compute_observation(env)
            obs_bc2 = replace_velocity_command(obs_bc2, np.zeros((1, 3), dtype=np.float32))
            absolute_bc2 = torch.as_tensor(bc2_policy(obs_bc2)[0], dtype=torch.float32, device=env.device).unsqueeze(0)
            env.step((absolute_bc2 - term._offset) / term._scale)
            left_count, right_count = box_contact_counts(env)
            box_z = float(box.data.root_link_pos_w[0, 2])
            lift = box_z - float(start_box[2])
            if monitor is not None:
                monitor.after_step(env, robot, "bc2", extra=f"box lift {100.0 * lift:+.1f} cm | fingertip contacts L{left_count} R{right_count}")
            hold = hold + 1 if left_count > 0 and right_count > 0 and lift >= 0.12 else 0
            tilt = tilt_of(robot)
            bc2.update({"steps": k + 1, "max_lift_m": round(max(bc2["max_lift_m"], lift), 4),
                        "max_hold_steps": max(bc2["max_hold_steps"], hold), "max_tilt_rad": round(max(bc2["max_tilt_rad"], tilt), 4)})
            if tilt > 0.8:
                bc2["outcome"] = "bad_orientation"
                break
            if box_z < float(start_box[2]) - 0.10:
                bc2["outcome"] = "box_drop"
                break
            if hold >= hold_required:
                bc2["outcome"] = "lift_hold_success"
                break
    if args_cli.dump_observations:
        np.savez_compressed(
            OUTPUT_DIR / f"episode_{episode:03d}.npz",
            **{k: np.asarray(v, dtype=np.float32) for k, v in dump.items() if len(v)},
        )
    summary = {
        "episode": episode,
        "seed": args_cli.seed,
        "initial_box_xy": round_list(initial_box[:2], 5),
        "initial_root_xyz": round_list(initial_root, 4),
        "start_root_xyz": round_list(start_root, 4),
        "start_rpy": round_list(start_rpy, 4),
        "start_lin_vel_b": round_list(start_lin_vel, 4),
        "reset_box_xy": round_list(reset_box[:2], 5),
        **start_state,
        "outcome": outcome,
        "steps": step + 1,
        "handoff_step": handoff_step,
        "max_tilt_rad": round(max(stats["tilt"]), 4),
        "roll_range": [round(min(stats["roll"]), 4), round(max(stats["roll"]), 4)],
        "pitch_range": [round(min(stats["pitch"]), 4), round(max(stats["pitch"]), 4)],
        "pelvis_z_range": [round(min(stats["z"]), 4), round(max(stats["z"]), 4)],
        "max_box_displacement_m": round(max(stats["box_dxy"]), 4),
        "max_contacts": int(max(stats["contacts"])),
        "max_wbc_target_jump_rad": round(max(stats["jump"]), 4),
        "bc2": bc2,
        "best_gate_geometry": best_geometry,
        "best_gate_score": round(best_score, 4),
        "wall_time_s": round(time.monotonic() - started, 2),
    }
    if forced_step is not None:
        summary["forced_handoff"] = {"step": forced_step, "used": forced}
    if readiness is not None:
        summary["readiness"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in readiness.items()}
    if teacher_servo is not None and teacher_servo != "pending":
        summary["teacher_servo"] = teacher_servo.state() | {"max_position_offset_cm": round(100 * servo_offsets_max, 2)}
    if collector is not None and args_cli.mode == "teacher_collect":
        success = outcome == "handoff_reached"
        collector["recorder"].end_episode(success=success, termination_reason=outcome,
                                          metadata={"gate_step": gate_step, "best_gate_score": round(best_score, 4)})
        collector["recorded"] += 1
        summary.update({"collected": True, "teacher_gate_step": gate_step, "demo_success": success})
    elif teacher is None:
        const = np.asarray(stats["raw_const"])
        summary.update(
            {
                "overwritten_steps_before_latch": overwritten_before_latch,
                "max_abs_z": round(max(stats["zmax"]), 2),
                "max_abs_z_first10": round(max(stats["zmax"][:10]), 2),
                "raw_constant_channels_range": {
                    name: [round(float(const[:, i].min()), 4), round(float(const[:, i].max()), 4)]
                    for i, name in enumerate(("left_hand", "right_hand", "base_height", "torso_roll", "torso_pitch", "torso_yaw"))
                },
            }
        )
    else:
        summary.update(
            {
                "obs_diff_max_first10": round(max(stats["obs_diff"][:10]), 6),
                "obs_diff_max": round(max(stats["obs_diff"]), 6),
                "first_step_obs_diff_above_1e-3": next((i for i, d in enumerate(stats["obs_diff"]) if d > 1.0e-3), None),
                "root_pos_diff_max": round(max(stats["root_diff"]), 6),
            }
        )
    return summary


def safe_continuation_paths(env, robot, pelvis, left_wrist, right_wrist, trajectory, retargeted, start_index, last_sent):
    """Phase-8 BC-2 v3 data: the source grasp/lift wrist path from ``start_index`` on, raised by one constant lift.

    The lift is the smallest one that keeps the predicted realized open hands (source observed wrists) at least the late
    margin above the table over the cage formation (close onset - 15 .. + 5 steps); the first 10 steps are blended from
    the BC-1 student's last command.  Returns the modified retargeted trajectory and the plan info.
    """
    import dataclasses

    from safe_paths import blend_into

    realized = source_realized_paths(trajectory, retargeted)
    margin = args_cli.safe_margin if args_cli.safe_margin_late is None else args_cli.safe_margin_late
    lo, hi = max(0, trajectory.close_index - 15), min(len(retargeted.left_wrist_world), trajectory.close_index + 6)
    window = {side: realized[side][lo:hi] for side in ("left", "right")}
    _, info = clearance_lifter(env).lift(window, np.ones(hi - lo, dtype=bool), margin)
    lift = float(info["max_lift_m"])
    start_cmd = action_wrists_world(body_matrix(robot, pelvis), last_sent)
    out = {}
    for side in ("left", "right"):
        path = np.asarray(getattr(retargeted, f"{side}_wrist_world"), dtype=np.float64).copy()
        tail = path[start_index:].copy()
        tail[:, 2, 3] += lift
        path[start_index:] = blend_into(start_cmd[side], tail, 10)
        out[side] = path
    plan = {"lift_m": round(lift, 4), "margin_m": margin, "start_index": int(start_index),
            "predicted_min_clearance_before_m": info["min_clearance_before_m"], "predicted_min_clearance_after_m": info["min_clearance_after_m"]}
    return dataclasses.replace(retargeted, left_wrist_world=out["left"], right_wrist_world=out["right"]), plan


def teacher_continuation(env, robot, bridge, box, pelvis, left_wrist, right_wrist, left_tips, right_tips,
                         start_box, gate_step, collector, episode, last_sent=None, monitor=None) -> dict:
    """Arena Teacher from the student's BC-1 gate state, time index continued from the gate step (unmodified, or the
    phase-8 safe continuation with --teacher_continuation_path safe)."""
    trajectory = collector["trajectory"] if collector is not None else ArenaLiftTrajectory(args_cli.arena_dataset)
    box_pose7 = torch.cat((box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])).detach().cpu().numpy()
    retargeted = trajectory.retarget(pose_matrix_wxyz(box_pose7))
    plan = None
    if args_cli.teacher_continuation_path == "safe":
        retargeted, plan = safe_continuation_paths(env, robot, pelvis, left_wrist, right_wrist, trajectory, retargeted,
                                                   min(gate_step + 1, trajectory.freeze_index), last_sent)
    hold_required = int(math.ceil(args_cli.bc2_hold_seconds / env.step_dt))
    result = {"outcome": "timeout", "steps": 0, "max_lift_m": 0.0, "max_hold_steps": 0, "max_tilt_rad": 0.0, "policy": "teacher",
              "continuation_plan": plan}
    if collector is not None:
        collector["recorder"].start_episode(metadata={"episode": episode, "seed": args_cli.seed, "student_gate_step": gate_step,
                                                      "initial_box_pose_wxyz": box_pose7.tolist()})
    hold = 0
    for k in range(int(math.ceil(args_cli.bc2_seconds / env.step_dt))):
        source_index = gate_step + 1 + k
        grasp_scope = source_index >= trajectory.close_index
        obs, command = compute_observation(env)
        action = np.asarray(trajectory.high_level_action(source_index, body_pose_wxyz(robot, pelvis).detach().cpu().numpy(), retargeted),
                            dtype=np.float32)
        action[16:19] = 0.0 if grasp_scope else command
        if grasp_scope:
            obs = replace_velocity_command(obs, np.zeros((1, 3), dtype=np.float32))
        names, values = bridge.step(bridge_request(env, action))
        raw_action, absolute = action_order_target(env, names, values)
        env.step(raw_action)
        if monitor is not None:
            index = min(source_index, trajectory.freeze_index)
            monitor.after_step(env, robot, "bc2", commanded={"left": retargeted.left_wrist_world[index], "right": retargeted.right_wrist_world[index]},
                               joint_target=absolute.detach().cpu().numpy()[0])
        left_count, right_count = box_contact_counts(env)
        box_z = float(box.data.root_link_pos_w[0, 2])
        lift = box_z - float(start_box[2])
        hold = hold + 1 if left_count > 0 and right_count > 0 and lift >= 0.12 else 0
        tilt = tilt_of(robot)
        result.update({"steps": k + 1, "max_lift_m": round(max(result["max_lift_m"], lift), 4),
                       "max_hold_steps": max(result["max_hold_steps"], hold), "max_tilt_rad": round(max(result["max_tilt_rad"], tilt), 4)})
        if collector is not None:
            collector["recorder"].append(TeacherDemoStep(
                student_observation=obs[0], teacher_high_level_action=action, processed_43d_joint_target=absolute.detach().cpu().numpy()[0],
                phase_id=0, box_pose=box.data.root_link_state_w[0, :7].detach().cpu().numpy(), bin_pose=collector["bin_pose"],
                root_pose=robot.data.root_link_state_w[0, :7].detach().cpu().numpy(), box_size=collector["box_size"],
                left_contact_count=left_count, right_contact_count=right_count,
                left_wrist_pose=body_pose_wxyz(robot, left_wrist).detach().cpu().numpy(),
                right_wrist_pose=body_pose_wxyz(robot, right_wrist).detach().cpu().numpy(),
                left_fingertip_positions=robot.data.body_link_pos_w[0, left_tips].detach().cpu().numpy(),
                right_fingertip_positions=robot.data.body_link_pos_w[0, right_tips].detach().cpu().numpy(),
                lift=bool(lift >= 0.12), drop=bool(box_z < float(start_box[2]) - 0.10), success=hold >= hold_required))
        if tilt > 0.8:
            result["outcome"] = "bad_orientation"
            break
        if box_z < float(start_box[2]) - 0.10:
            result["outcome"] = "box_drop"
            break
        if hold >= hold_required:
            result["outcome"] = "lift_hold_success"
            break
    if collector is not None:
        collector["recorder"].end_episode(success=result["outcome"] == "lift_hold_success", termination_reason=result["outcome"],
                                          metadata={"student_gate_step": gate_step, **result})
        collector["recorded"] += 1
    return result


def action_term_offset(env):
    return env.action_manager.get_term("JointPositionAction")._offset


def action_term_scale(env):
    return env.action_manager.get_term("JointPositionAction")._scale


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
