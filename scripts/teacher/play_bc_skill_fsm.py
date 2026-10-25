#!/usr/bin/env python3
"""Run four independent BC skills through an explicit Box task FSM."""

# Isaac Sim must be launched before the runtime imports below.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher
from unitree_rl_lab.runtime_paths import default_arena_root, default_arena_dataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--approach_checkpoint", required=True)
parser.add_argument("--grasp_lift_checkpoint", required=True)
parser.add_argument("--carry_checkpoint", required=True)
parser.add_argument("--place_checkpoint", required=True)
parser.add_argument("--bc2_gate", required=True)
parser.add_argument(
    "--allow_uncertified_grasp_probe",
    action="store_true",
    help=(
        "Diagnostic only: allow the grasp checkpoint to differ from the valid "
        "upstream BC-2 gate; the mismatch is recorded and must not be promoted"
    ),
)
parser.add_argument("--task", default="Unitree-G1-43dof-BoxPickPlace")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--episode_seconds", type=float, default=30.0)
parser.add_argument("--strict_hold_seconds", type=float, default=2.0)
parser.add_argument(
    "--grasp_handoff_steps",
    type=int,
    default=25,
    help="Smoothstep blend from the realized Approach/WBC target into Grasp-Lift",
)
parser.add_argument(
    "--carry_handoff_steps",
    type=int,
    default=100,
    help="Lower-body smoothstep blend during the zero-command loaded stabilization",
)
parser.add_argument(
    "--loaded_stabilization_steps",
    type=int,
    default=100,
    help="Consecutive bilateral lifted zero-command CARRY steps required after grasp",
)
parser.add_argument(
    "--loaded_stand_steps",
    type=int,
    default=250,
    help="Additional zero-command loaded-balance steps required before walking",
)
parser.add_argument(
    "--carry_upper_residual_scale",
    type=float,
    default=0.0,
    help="Optional bounded shoulder/elbow/wrist residual around the grasp target; fingers stay locked",
)
parser.add_argument(
    "--carry_max_upper_residual_rad",
    type=float,
    default=0.05,
    help="Per-joint absolute bound for an enabled CARRY manipulation residual",
)
parser.add_argument(
    "--carry_lower_body_mode",
    choices=("student", "blend", "wbc"),
    default="student",
    help=(
        "Lower-body controller after loaded stand: the Carry Student, a smooth "
        "WBC-to-Student blend, or the live payload-aware Homie WBC controller"
    ),
)
parser.add_argument(
    "--carry_student_lower_blend_steps",
    type=int,
    default=250,
    help="Smoothstep duration when --carry_lower_body_mode=blend",
)
parser.add_argument(
    "--carry_staging_standoff_m",
    type=float,
    default=0.30,
    help="Camera-derived robot standoff from the Bin before PLACE starts",
)
parser.add_argument(
    "--carry_staging_position_tolerance_m",
    type=float,
    default=0.10,
    help="Maximum camera-derived radial staging error before PLACE may start",
)
parser.add_argument(
    "--carry_staging_yaw_tolerance_rad",
    type=float,
    default=0.15,
    help="Maximum camera-derived Bin yaw error before PLACE may start",
)
parser.add_argument(
    "--carry_velocity_scale",
    type=float,
    default=1.0,
    help="Scale the camera-derived CARRY velocity command after its startup ramp",
)
parser.add_argument(
    "--carry_current_wrist_wbc",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "For a 23-D CARRY Student, replace its wrist-pose setpoints with the "
        "currently realized proprioceptive wrist poses before the Pink/Homie "
        "solve. Navigation velocity remains the public FSM command."
    ),
)
parser.add_argument(
    "--place_release_when_in_bin",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Open the fingers once the physical FSM confirms that the Box is inside "
        "the Bin; this makes release timing an explicit safety gate instead of "
        "requiring the memoryless Place Student to infer elapsed phase time"
    ),
)
parser.add_argument(
    "--stop_after_grasp_snapshot",
    action="store_true",
    help="Snapshot-corpus mode: end each episode immediately after a real FSM grasp succeeds",
)
parser.add_argument(
    "--stop_after_handoff_gate",
    action="store_true",
    help=(
        "Evaluation mode: end an episode once the 100-step loaded Handoff gate "
        "passes, without entering loaded stand or walking evaluation."
    ),
)
parser.add_argument(
    "--stop_after_carry_snapshot",
    action="store_true",
    help=(
        "Snapshot-corpus mode: end each episode immediately after Student Carry "
        "reaches staging and its physical snapshot is recorded."
    ),
)
parser.add_argument(
    "--target_grasp_snapshots",
    type=int,
    default=0,
    help="Snapshot-corpus mode: stop the run after this many real FSM grasp snapshots",
)
parser.add_argument(
    "--grasp_phase_timeout_seconds",
    type=float,
    default=0.0,
    help="Optional snapshot-corpus timeout after APPROACH enters GRASP_LIFT; zero disables it",
)
parser.add_argument(
    "--max_grasp_retries",
    type=int,
    default=0,
    help=(
        "Maximum in-episode recovery transitions from a failed table-height "
        "grasp back to APPROACH; zero preserves the single-attempt contract"
    ),
)
parser.add_argument(
    "--grasp_retry_lost_contact_steps",
    type=int,
    default=50,
    help="Consecutive non-bilateral table-height steps required before a grasp retry",
)
parser.add_argument(
    "--grasp_retry_min_phase_steps",
    type=int,
    default=150,
    help="Minimum GRASP_LIFT steps before recovery may return to APPROACH",
)
parser.add_argument(
    "--handoff_wrist_midpoint_limit_m",
    type=float,
    default=0.222,
    help=(
        "Approach-to-Grasp wrist-midpoint gate. Keep the 0.222 m production default; "
        "larger values are explicit diagnostic probes and are recorded in metrics."
    ),
)
parser.add_argument(
    "--handoff_tip_distance_limit_m",
    type=float,
    default=0.18,
    help="Maximum closest fingertip-to-Box distance at the Approach handoff",
)
parser.add_argument(
    "--handoff_cage_midpoint_limit_m",
    type=float,
    default=0.11,
    help="Maximum fingertip-cage midpoint error at the Approach handoff",
)
parser.add_argument(
    "--handoff_wrist_asymmetry_limit_m",
    type=float,
    default=0.10,
    help="Maximum left/right wrist distance asymmetry at the Approach handoff",
)
parser.add_argument(
    "--approach_ready_hold_steps",
    type=int,
    default=1,
    help=(
        "Settling steps after latching the first physical pre-grasp-ready "
        "joint target before switching from Approach to Grasp/Lift"
    ),
)
parser.add_argument(
    "--grasp_lower_body_lock",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic: preserve the realized Approach leg/waist targets throughout "
        "Grasp-Lift while its policy controls shoulder/elbow/wrist/hand joints."
    ),
)
parser.add_argument(
    "--grasp_live_approach_lower_body",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Keep the 12 leg joints under a zero-velocity live APPROACH WBC during "
        "GRASP_LIFT while the certified grasp policy owns waist, arms, and hands"
    ),
)
parser.add_argument(
    "--grasp_latched_approach_lower_body",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "During GRASP_LIFT, keep the last in-distribution APPROACH high-level "
        "command as the live WBC stance instead of querying the APPROACH "
        "network on out-of-distribution grasp observations"
    ),
)
parser.add_argument(
    "--grasp_wbc_current_wrist_stance",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Build the GRASP_LIFT live WBC baseline from zero base velocity and "
        "the currently realized wrist poses, preventing stale wrist targets "
        "from driving lower-body translation during the lift"
    ),
)
parser.add_argument(
    "--grasp_wbc_current_wrist_after_upper_hold",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Use the dynamic APPROACH stance until lift capture, then switch the "
        "WBC lower-body baseline to zero velocity and current wrist poses"
    ),
)
parser.add_argument(
    "--grasp_wbc_current_wrist_after_contact",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Use zero-velocity current-wrist WBC as soon as bilateral contact has "
        "latched, preventing the Approach lower-body policy from moving the "
        "base during the remaining lift."
    ),
)
parser.add_argument(
    "--grasp_wbc_current_wrist_min_lift_m",
    type=float,
    default=0.0,
    help=(
        "Latch zero-velocity current-wrist WBC after reaching this positive "
        "partial lift height; zero disables the partial-lift latch."
    ),
)
parser.add_argument(
    "--grasp_latch_live_legs_on_contact",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Latch the live Approach WBC leg target at first bilateral contact so "
        "the remaining Grasp/Lift cannot walk away from the Box."
    ),
)
parser.add_argument(
    "--grasp_live_leg_max_residual_rad",
    type=float,
    default=0.0,
    help=(
        "Optional per-joint bound around the Approach handoff target for the "
        "12 live WBC leg targets; zero leaves them unbounded."
    ),
)
parser.add_argument(
    "--grasp_post_hold_stance_blend_steps",
    type=int,
    default=0,
    help="Smoothstep duration for the post-lift WBC stance transition",
)
parser.add_argument(
    "--grasp_freeze_upper_on_lift",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "After the first bilateral 12 cm lift, hold the realized waist/arm/hand "
        "target while live WBC legs stabilize the strict hold interval"
    ),
)
parser.add_argument(
    "--grasp_upper_freeze_min_lift_m",
    type=float,
    default=0.12,
    help=(
        "Lift height at which the realized waist/arm/hand target is frozen. "
        "Values above the strict 0.12 m gate provide settling margin."
    ),
)
parser.add_argument(
    "--grasp_freeze_hands_on_contact",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Freeze the realized 14 finger targets after bilateral contact and the "
        "configured minimum lift while waist/arms continue the learned motion"
    ),
)
parser.add_argument(
    "--grasp_hand_freeze_min_lift_m",
    type=float,
    default=0.0,
    help=(
        "Minimum lift before finger targets are frozen. This avoids locking a "
        "partially closed hand at the first transient contact."
    ),
)
parser.add_argument(
    "--grasp_refresh_hand_lock_on_upper_hold",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "When the stable lifted upper-body reference is captured, refresh "
        "the finger lock from that realized loaded grasp instead of retaining "
        "the weaker first-contact finger target"
    ),
)
parser.add_argument(
    "--grasp_freeze_loaded_posture_on_upper_hold",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "After a bilateral lifted posture is captured, retain its complete "
        "43-joint target during the strict hold instead of continuing to "
        "replace the legs with a changing out-of-distribution WBC stance"
    ),
)
parser.add_argument(
    "--grasp_canonical_hand_lock",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Replace the retained first-contact finger target with Arena's exact "
        "closed Dex3 target; this is a deterministic actuator postprocessor."
    ),
)
parser.add_argument(
    "--approach_action_alpha",
    type=float,
    default=1.0,
    help=(
        "EMA coefficient for APPROACH actions. It smooths both processed 43-D "
        "targets and 23-D high-level WBC commands; 1.0 disables smoothing"
    ),
)
parser.add_argument(
    "--approach_search_torso_pitch_rad",
    type=float,
    default=-0.40,
    help=(
        "Deployable active-vision torso pitch used only while the top QR is "
        "outside the camera frustum; zero disables the look-ahead posture"
    ),
)
parser.add_argument(
    "--approach_hand_state_decoding",
    choices=("none", "binary"),
    default="none",
    help=(
        "binary: map the 23-D APPROACH hand-state outputs to the Arena {0, 1} "
        "contract (>= 0.5 closes). With none, any non-zero regression output "
        "closes the Dex3 hand because Arena tests hand_state == 0."
    ),
)
parser.add_argument(
    "--narrow_camera_profile",
    choices=("certified_55", "collector_69"),
    default="certified_55",
    help=(
        "Narrow-workspace camera contract: certified_55 keeps the historical "
        "55.2 deg overrides; collector_69 keeps the environment defaults used "
        "by collect_nominal_box_teacher.py (69.4 deg)."
    ),
)
parser.add_argument(
    "--pre_navigate",
    action="store_true",
    help=(
        "Approach research: start each episode from a random pose behind the canonical BC-1 start "
        "and drive there with the ground-truth scripted navigator (Homie [vx, vy, wz], fixed nominal "
        "wrist targets, open hands), then settle, before the unchanged APPROACH phase"
    ),
)
parser.add_argument("--pre_navigate_start_range", type=float, nargs=6, default=(-0.8, -0.4, -0.25, 0.25, -0.4, 0.4),
                    metavar=("DX_LO", "DX_HI", "DY_LO", "DY_HI", "DYAW_LO", "DYAW_HI"))
parser.add_argument("--pre_navigate_max_steps", type=int, default=750)
parser.add_argument("--pre_navigate_tolerance", type=float, nargs=2, default=(0.03, 0.03), metavar=("POS_M", "YAW_RAD"))
parser.add_argument("--pre_navigate_target_jitter", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DYAW"))
parser.add_argument("--pre_navigate_settle_steps", type=int, default=25)
parser.add_argument(
    "--approach_task_target_passthrough",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Route camera navigation and fixed pelvis targets directly into the 23-D "
        "APPROACH command. Disabled by default to preserve the command distribution "
        "of the certified narrow-workspace Student; enable for 5 m search experiments."
    ),
)
parser.add_argument("--box_x_range", type=float, nargs=2, default=(0.618, 0.622))
parser.add_argument("--box_y_range", type=float, nargs=2, default=(-0.002, 0.002))
parser.add_argument(
    "--box_mass_kg",
    type=float,
    default=0.20,
    help=(
        "Fixed physical Box mass for this rollout. Curriculum evaluation runs "
        "the same checkpoint repeatedly at increasing values."
    ),
)
parser.add_argument(
    "--hand_stiffness",
    type=float,
    default=4.0,
    help="Dex3 implicit position-control stiffness used for physical FSM evaluation",
)
parser.add_argument(
    "--grasp_hand_stiffness",
    type=float,
    default=4.0,
    help=(
        "Dex3 stiffness applied only after the FSM enters GRASP_LIFT; the "
        "Approach phase retains --hand_stiffness."
    ),
)
parser.add_argument("--randomize_work_area", action="store_true")
parser.add_argument(
    "--work_area_size_m",
    type=float,
    default=5.0,
    help="Total side length of the square table/bin work area",
)
parser.add_argument(
    "--box_local_x_range", type=float, nargs=2, default=(-0.05, 0.05)
)
parser.add_argument(
    "--box_local_y_range", type=float, nargs=2, default=(-0.05, 0.05)
)
parser.add_argument("--seed", type=int, default=20260814)
parser.add_argument("--arena_root", default=default_arena_root())
parser.add_argument(
    "--arena_dataset",
    default=default_arena_dataset(),
)
parser.add_argument(
    "--approach_qr_endpoint_alignment",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "At camera-derived pre-grasp range, retarget the Arena open-hand endpoint "
        "to the observed QR pose before Pink IK"
    ),
)
parser.add_argument(
    "--approach_endpoint_offset_steps",
    type=int,
    default=5,
    help=(
        "Arena steps after the nominal hand-close boundary used for tighter "
        "open-hand pre-grasp wrist geometry"
    ),
)
parser.add_argument(
    "--approach_endpoint_centering_m",
    type=float,
    default=0.03,
    help="Maximum common QR-directed translation of both endpoint wrist targets",
)
parser.add_argument(
    "--approach_wrist_servo_gain",
    type=float,
    default=0.0,
    help=(
        "Closed-loop QR/proprio wrist-position correction gain near the pre-grasp "
        "boundary; zero disables the deployable alignment servo"
    ),
)
parser.add_argument(
    "--approach_wrist_servo_max_correction_m",
    type=float,
    default=0.08,
    help="Maximum per-wrist Cartesian correction added to the Arena endpoint",
)
parser.add_argument(
    "--metrics_output", default="data/teacher/bc_fsm_rollout.metrics.json"
)
parser.add_argument("--snapshot_output", default=None)
parser.add_argument(
    "--video_output",
    default=None,
    help="Optional MP4 prefix for an honest single-process FSM physics capture",
)
parser.add_argument(
    "--video_length_steps",
    type=int,
    default=2000,
    help="Maximum number of environment steps recorded by Gymnasium RecordVideo",
)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--real_time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if (
    args_cli.episodes <= 0
    or args_cli.episode_seconds <= 0.0
    or args_cli.strict_hold_seconds <= 0.0
    or args_cli.grasp_handoff_steps < 0
    or args_cli.carry_handoff_steps < 0
    or args_cli.loaded_stabilization_steps <= 0
    or args_cli.loaded_stand_steps <= 0
    or args_cli.carry_student_lower_blend_steps <= 0
    or args_cli.carry_staging_standoff_m <= 0.0
    or args_cli.carry_staging_position_tolerance_m <= 0.0
    or args_cli.carry_staging_yaw_tolerance_rad <= 0.0
    or args_cli.target_grasp_snapshots < 0
    or args_cli.grasp_phase_timeout_seconds < 0.0
    or args_cli.max_grasp_retries < 0
    or args_cli.grasp_retry_lost_contact_steps <= 0
    or args_cli.grasp_retry_min_phase_steps <= 0
    or args_cli.grasp_post_hold_stance_blend_steps < 0
    or args_cli.grasp_upper_freeze_min_lift_m < 0.12
    or args_cli.grasp_hand_freeze_min_lift_m < 0.0
    or args_cli.grasp_wbc_current_wrist_min_lift_m < 0.0
    or args_cli.grasp_live_leg_max_residual_rad < 0.0
    or args_cli.handoff_wrist_midpoint_limit_m <= 0.0
    or args_cli.handoff_tip_distance_limit_m <= 0.0
    or args_cli.handoff_cage_midpoint_limit_m <= 0.0
    or args_cli.handoff_wrist_asymmetry_limit_m <= 0.0
    or args_cli.approach_ready_hold_steps <= 0
    or args_cli.approach_endpoint_offset_steps < 0
    or args_cli.approach_endpoint_centering_m < 0.0
    or args_cli.approach_wrist_servo_gain < 0.0
    or args_cli.approach_wrist_servo_max_correction_m <= 0.0
    or args_cli.box_mass_kg <= 0.0
    or args_cli.hand_stiffness <= 0.0
    or args_cli.grasp_hand_stiffness <= 0.0
    or not 0.0 < args_cli.carry_velocity_scale <= 1.0
    or args_cli.video_length_steps <= 0
):
    parser.error("episodes/durations must be positive and handoff steps non-negative")
if not 0.0 < args_cli.approach_action_alpha <= 1.0:
    parser.error("--approach_action_alpha must be in (0, 1]")
if not 0.0 <= args_cli.carry_upper_residual_scale <= 1.0:
    parser.error("--carry_upper_residual_scale must be in [0, 1]")
if args_cli.carry_max_upper_residual_rad <= 0.0:
    parser.error("--carry_max_upper_residual_rad must be positive")
if args_cli.target_grasp_snapshots and not args_cli.stop_after_grasp_snapshot:
    parser.error("--target_grasp_snapshots requires --stop_after_grasp_snapshot")
if args_cli.stop_after_grasp_snapshot and args_cli.snapshot_output is None:
    parser.error("--stop_after_grasp_snapshot requires --snapshot_output")
if args_cli.stop_after_carry_snapshot and args_cli.snapshot_output is None:
    parser.error("--stop_after_carry_snapshot requires --snapshot_output")
if args_cli.grasp_lower_body_lock and args_cli.grasp_live_approach_lower_body:
    parser.error(
        "--grasp_lower_body_lock and --grasp_live_approach_lower_body are mutually exclusive"
    )
if args_cli.grasp_freeze_upper_on_lift and not args_cli.grasp_live_approach_lower_body:
    parser.error("--grasp_freeze_upper_on_lift requires live APPROACH legs")
if args_cli.grasp_freeze_hands_on_contact and not args_cli.grasp_live_approach_lower_body:
    parser.error("--grasp_freeze_hands_on_contact requires live APPROACH legs")
if any(
    bounds[0] > bounds[1]
    for bounds in (
        args_cli.box_x_range,
        args_cli.box_y_range,
        args_cli.box_local_x_range,
        args_cli.box_local_y_range,
    )
):
    parser.error("Box x/y ranges require MIN <= MAX")
if args_cli.work_area_size_m <= 1.5:
    parser.error("--work_area_size_m must exceed 1.5 m")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils import math as math_utils

from unitree_rl_lab.utils.arena_wbc_bridge import ArenaWBCBridge
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.teacher.arena_teacher_runtime import (
    ArenaLiftTrajectory,
    LEFT_FINGERTIP_NAMES,
    RIGHT_FINGERTIP_NAMES,
    action_order_target,
    body_id,
    body_pose_wxyz,
    box_contact_counts,
    bridge_request,
    configure_fixed_teacher_scene,
    student_observation,
)
from unitree_rl_lab.utils.teacher.carry_runtime_contract import (
    carry_navigation_scale,
    validate_carry_runtime_contract,
)
from unitree_rl_lab.utils.teacher.phase_bc import (
    file_sha256,
    load_bc_policy,
    validate_bc2_gate,
    validate_runtime_action_contract,
)
from unitree_rl_lab.utils.teacher.place_runtime_contract import (
    apply_place_runtime_postprocessor,
    validate_place_runtime_contract,
)
from unitree_rl_lab.utils.teacher.physical_snapshot_runtime import (
    capture_physical_snapshot,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import (
    PhysicalSnapshotStore,
    SnapshotKind,
)
from unitree_rl_lab.utils.teacher.phase_teachers import (
    CarryTeacher,
    RealizedBimanualGrasp,
    pose_matrix_wxyz,
)
from unitree_rl_lab.utils.teacher.skill_fsm import (
    ConditionalSuccessMetrics,
    SkillPhase,
    SkillRouter,
    SkillSignals,
    bimanual_pregrasp_ready,
    bimanual_wrists_centered,
    box_reached_ground_outside_bin,
)
from unitree_rl_lab.utils.teacher.student_observation import (
    STUDENT_FEATURE_SLICES,
    replace_velocity_command,
)
from unitree_rl_lab.utils.teacher_demo_recorder import DEFAULT_HIGH_LEVEL_ACTION_NAMES


BOX_SIZE_M = 0.20
TABLE_SURFACE_HEIGHT_M = 0.74

# Median wrist offsets measured at the open-hand boundary (step 125) over the
# 500 strict successful Arena-retargeted demonstrations.  They are expressed
# in the Box frame and are therefore reusable after QR/PnP pose estimation.
APPROACH_NOMINAL_LEFT_WRIST_OFFSET_BOX_M = np.array(
    (-0.202564, 0.119889, -0.015807), dtype=np.float32
)
APPROACH_NOMINAL_RIGHT_WRIST_OFFSET_BOX_M = np.array(
    (-0.214756, -0.152998, -0.021132), dtype=np.float32
)
BIN_CENTER_E = (1.55, -0.95, 0.28)
BIN_HALF_EXTENTS = (0.30, 0.35, 0.21)
TABLE_CENTER_E = (0.62, 0.0, 0.74)
PLACE_SETTLE_STEPS = 10
PLACE_SETTLE_MAX_BOX_SPEED_MPS = 0.15
CARRY_DROP_LIFT_HEIGHT_M = 0.08
CERTIFIED_NARROW_CAMERA_HORIZONTAL_FOV_DEG = 55.2
PLACE_PHASE_CLOCK_SCHEMA = "unitree_place_phase_clock_v1"
PLACE_OBSERVATION_ADAPTER_SCHEMA = "unitree_place_observation_adapter_v1"


def _place_phase_clock_contract(policy) -> dict | None:
    raw = policy.metadata.get("place_phase_clock")
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("schema") != PLACE_PHASE_CLOCK_SCHEMA:
        raise ValueError("PLACE checkpoint has an unsupported phase-clock contract")
    expected_index = STUDENT_FEATURE_SLICES["velocity_commands"].start
    feature_index = int(raw.get("feature_index", -1))
    horizon_steps = int(raw.get("horizon_steps", 0))
    if feature_index != expected_index:
        raise ValueError("PLACE phase clock does not target velocity_commands.x")
    if horizon_steps <= 0 or raw.get("privileged") is not False:
        raise ValueError("PLACE phase-clock contract is invalid")
    return {**raw, "feature_index": feature_index, "horizon_steps": horizon_steps}


def _place_observation_adapter_contract(policy) -> dict | None:
    raw = policy.metadata.get("place_observation_adapter")
    if raw is None:
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != PLACE_OBSERVATION_ADAPTER_SCHEMA
        or raw.get("privileged") is not False
    ):
        raise ValueError("PLACE checkpoint has an invalid observation adapter")
    masked = tuple(raw.get("masked_feature_names", ()))
    allowed = {"policy_history", "last_action"}
    if not set(masked).issubset(allowed):
        raise ValueError("PLACE observation adapter masks unsupported features")
    return {**raw, "masked_feature_names": masked}


def _configure_env():
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    cfg.seed = args_cli.seed
    # Direct 43-D checkpoints were trained from Arena teacher targets and must
    # be evaluated with the same teacher-only articulation dynamics.  The
    # policy itself still receives only deployed student observations.
    configure_fixed_teacher_scene(
        cfg,
        box_size_m=BOX_SIZE_M,
        box_mass_kg=args_cli.box_mass_kg,
        teacher_hand_stiffness=args_cli.hand_stiffness,
        keep_work_area_randomization=args_cli.randomize_work_area,
    )
    cfg.commands.base_velocity.standoff_distance = float(
        args_cli.carry_staging_standoff_m
    )
    cfg.commands.base_velocity.position_tolerance = float(
        args_cli.carry_staging_position_tolerance_m
    )
    cfg.commands.base_velocity.yaw_tolerance = float(
        args_cli.carry_staging_yaw_tolerance_rad
    )
    if not args_cli.randomize_work_area and args_cli.narrow_camera_profile == "certified_55":
        # Preserve the camera observation contract used to train and certify
        # the narrow-workspace Students.  The wider RealSense-like lens is a
        # separate 5 m workspace curriculum and changes QR-visible histories.
        narrow_fov = CERTIFIED_NARROW_CAMERA_HORIZONTAL_FOV_DEG
        cfg.commands.base_velocity.horizontal_fov_deg = narrow_fov
        cfg.commands.base_velocity.box_visibility_grace_steps = 0
        cfg.commands.base_velocity.pregrasp_search_forward_speed = 0.0
        cfg.commands.base_velocity.pregrasp_search_yaw_velocity = -0.20
        for term_name in (
            "box_visible",
            "box_position_camera",
            "bin_visible",
            "bin_position_camera",
        ):
            term = getattr(cfg.observations.policy_current, term_name)
            term.params["horizontal_fov_deg"] = narrow_fov
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
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    if args_cli.pre_navigate:
        x_lo, x_hi, y_lo, y_hi, yaw_lo, yaw_hi = args_cli.pre_navigate_start_range
        cfg.events.reset_base.params["pose_range"] = {"x": (x_lo, x_hi), "y": (y_lo, y_hi), "yaw": (yaw_lo, yaw_hi)}
    cfg.events.reset_pick_place_scene.params["box_base_size"] = (BOX_SIZE_M,) * 3
    if args_cli.randomize_work_area:
        centre_extent = 0.5 * args_cli.work_area_size_m - 0.45
        cfg.events.reset_pick_place_work_area.params["workspace_xy_range"] = {
            "x": (-centre_extent, centre_extent),
            "y": (-centre_extent, centre_extent),
        }
        cfg.events.reset_pick_place_scene.params["grasp_box_xy_range"] = {
            "x": tuple(args_cli.box_local_x_range),
            "y": tuple(args_cli.box_local_y_range),
        }
        cfg.events.reset_pick_place_scene.params["box_xy_relative_to_table"] = True
    else:
        cfg.events.reset_pick_place_scene.params["grasp_box_xy_range"] = {
            "x": tuple(args_cli.box_x_range),
            "y": tuple(args_cli.box_y_range),
        }
        cfg.events.reset_pick_place_scene.params["box_xy_relative_to_table"] = False
    cfg.observations.policy_history.enable_corruption = False
    cfg.observations.policy_current.enable_corruption = False
    cfg.scene.front_camera = None
    # Manual FSM accounting must inspect terminal physics before any automatic
    # reset.  In particular, a valid release into the Bin is not a drop.
    cfg.terminations.time_out = None
    cfg.terminations.bad_orientation = None
    cfg.terminations.success = None
    cfg.rewards.bad_orientation_penalty = None
    cfg.episode_length_s = args_cli.episode_seconds + 1.0
    if args_cli.pre_navigate:
        cfg.episode_length_s += 0.02 * (args_cli.pre_navigate_max_steps + args_cli.pre_navigate_settle_steps)
    if args_cli.video_output is not None:
        cfg.viewer.eye = (2.9, 2.5, 2.1)
        cfg.viewer.lookat = (0.65, -0.25, 0.82)
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
    return cfg


class _RecordingEnvProxy:
    """Use RecordVideo reset/step/close while exposing the manager environment."""

    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self._unwrapped_env = wrapped_env.unwrapped

    def __getattr__(self, name):
        return getattr(self._unwrapped_env, name)

    def reset(self, *args, **kwargs):
        return self._wrapped_env.reset(*args, **kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def close(self):
        return self._wrapped_env.close()


def _inside_bin(env) -> bool:
    box = env.scene["box"]
    center_e = (
        env.bin_center_e[0]
        if hasattr(env, "bin_center_e")
        else torch.tensor(BIN_CENTER_E, device=env.device)
    )
    center = env.scene.env_origins[0] + center_e
    half = torch.tensor(BIN_HALF_EXTENTS, device=env.device)
    return bool(torch.all(torch.abs(box.data.root_link_pos_w[0] - center) < half))


def _bad_orientation(env) -> bool:
    gravity = env.scene["robot"].data.projected_gravity_b[0]
    return bool(torch.acos(torch.clamp(-gravity[2], -1.0, 1.0)) > 0.8)


def _task_phase(skill_phase: SkillPhase) -> int:
    if skill_phase in (SkillPhase.APPROACH, SkillPhase.GRASP_LIFT):
        return 0
    if skill_phase == SkillPhase.CARRY:
        return 1
    if skill_phase == SkillPhase.PLACE:
        return 2
    return 3


def _yaw_from_quat_wxyz(quaternion: torch.Tensor) -> float:
    """Return the world yaw of one Isaac Lab wxyz quaternion."""
    w, x, y, z = (float(value) for value in quaternion.detach().cpu())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _box_top_position_camera(env, command_term) -> np.ndarray:
    """Unmasked Box-top marker position for active-vision diagnostics."""
    robot = env.scene["robot"]
    torso_id = command_term.robot_cfg.body_ids[0]
    torso_pos_w = robot.data.body_link_pos_w[0, torso_id]
    torso_quat_w = robot.data.body_link_quat_w[0, torso_id]
    camera_position_b = torch.tensor(
        command_term.cfg.camera_position_b, device=env.device
    )
    camera_orientation_b = torch.tensor(
        command_term.cfg.camera_orientation_b, device=env.device
    )
    camera_pos_w = torso_pos_w + math_utils.quat_apply(
        torso_quat_w.unsqueeze(0), camera_position_b.unsqueeze(0)
    )[0]
    camera_quat_w = math_utils.quat_mul(
        torso_quat_w.unsqueeze(0), camera_orientation_b.unsqueeze(0)
    )[0]
    box_top_w = env.scene["box"].data.root_link_pos_w[0].clone()
    box_top_w[2] += 0.5 * BOX_SIZE_M
    return (
        math_utils.quat_apply_inverse(
            camera_quat_w.unsqueeze(0), (box_top_w - camera_pos_w).unsqueeze(0)
        )[0]
        .detach()
        .cpu()
        .numpy()
    )


def _hold_current_wrists_in_pelvis(
    env,
    action: np.ndarray,
    pelvis_id: int,
    left_wrist_id: int,
    right_wrist_id: int,
) -> np.ndarray:
    """Keep upper-body reach neutral while camera navigation owns the base."""
    result = action.copy()
    robot = env.scene["robot"]
    pelvis_position = robot.data.body_link_pos_w[0, pelvis_id]
    pelvis_quaternion = robot.data.body_link_quat_w[0, pelvis_id]
    pelvis_inverse = math_utils.quat_inv(pelvis_quaternion.unsqueeze(0))[0]
    for body_id_value, position_slice, quaternion_slice in (
        (left_wrist_id, slice(2, 5), slice(5, 9)),
        (right_wrist_id, slice(9, 12), slice(12, 16)),
    ):
        wrist_position = robot.data.body_link_pos_w[0, body_id_value]
        wrist_quaternion = robot.data.body_link_quat_w[0, body_id_value]
        relative_position = math_utils.quat_apply(
            pelvis_inverse.unsqueeze(0),
            (wrist_position - pelvis_position).unsqueeze(0),
        )[0]
        relative_quaternion_wxyz = math_utils.quat_mul(
            pelvis_inverse.unsqueeze(0), wrist_quaternion.unsqueeze(0)
        )[0]
        result[0, position_slice] = relative_position.detach().cpu().numpy()
        wxyz = relative_quaternion_wxyz.detach().cpu().numpy()
        result[0, quaternion_slice] = wxyz[[1, 2, 3, 0]]
    result[:, :2] = 1.0
    return result


def _qr_observed_box_pose_wxyz(env, command_term) -> np.ndarray:
    """Reconstruct the fixed-yaw Box pose from the public camera measurement."""
    robot = env.scene["robot"]
    torso_id = command_term.robot_cfg.body_ids[0]
    torso_pos_w = robot.data.body_link_pos_w[0, torso_id]
    torso_quat_w = robot.data.body_link_quat_w[0, torso_id]
    camera_position_b = torch.tensor(
        command_term.cfg.camera_position_b, device=env.device
    )
    camera_orientation_b = torch.tensor(
        command_term.cfg.camera_orientation_b, device=env.device
    )
    camera_pos_w = torso_pos_w + math_utils.quat_apply(
        torso_quat_w.unsqueeze(0), camera_position_b.unsqueeze(0)
    )[0]
    camera_quat_w = math_utils.quat_mul(
        torso_quat_w.unsqueeze(0), camera_orientation_b.unsqueeze(0)
    )[0]
    measured_position_w = camera_pos_w + math_utils.quat_apply(
        camera_quat_w.unsqueeze(0), command_term.box_position_camera[0].unsqueeze(0)
    )[0]
    # Work-area randomization translates but does not rotate the table/Box.
    # Real deployment replaces this fixed quaternion with QR PnP orientation.
    return np.asarray(
        [*measured_position_w.detach().cpu().tolist(), 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )


def _apply_qr_proprio_wrist_servo(
    reference_action: np.ndarray,
    *,
    observed_box_pose_wxyz: np.ndarray,
    pelvis_pose_wxyz: np.ndarray,
    actual_left_wrist_position_w: np.ndarray,
    actual_right_wrist_position_w: np.ndarray,
    gain: float,
    maximum_correction_m: float,
) -> np.ndarray:
    """Correct endpoint tracking from QR pose and proprioceptive kinematics.

    The nominal offsets come only from successful demonstrations.  Runtime
    feedback uses the camera-derived Box pose and wrist forward kinematics;
    no privileged simulator Box state enters this controller.
    """

    result = np.asarray(reference_action, dtype=np.float32).copy()
    box_pose = pose_matrix_wxyz(observed_box_pose_wxyz)
    pelvis_pose = pose_matrix_wxyz(pelvis_pose_wxyz)
    pelvis_rotation_inverse = pelvis_pose[:3, :3].T
    for position_slice, actual_position, nominal_offset in (
        (
            slice(2, 5),
            actual_left_wrist_position_w,
            APPROACH_NOMINAL_LEFT_WRIST_OFFSET_BOX_M,
        ),
        (
            slice(9, 12),
            actual_right_wrist_position_w,
            APPROACH_NOMINAL_RIGHT_WRIST_OFFSET_BOX_M,
        ),
    ):
        desired_position_w = (
            box_pose
            @ np.asarray([*nominal_offset, 1.0], dtype=np.float32)
        )[:3]
        error_w = desired_position_w - np.asarray(actual_position, dtype=np.float32)
        correction_pelvis = gain * (pelvis_rotation_inverse @ error_w)
        norm = float(np.linalg.norm(correction_pelvis))
        if norm > maximum_correction_m:
            correction_pelvis *= maximum_correction_m / norm
        result[position_slice] += correction_pelvis
    return result


def _absolute_to_raw(env, absolute: np.ndarray) -> torch.Tensor:
    term = env.action_manager.get_term("JointPositionAction")
    target = torch.as_tensor(absolute, dtype=torch.float32, device=env.device)
    raw = (target - term._offset) / term._scale
    if not torch.isfinite(raw).all():
        raise RuntimeError("BC produced a non-finite normalized joint action")
    return raw


def _preserve_carry_grasp_targets(
    raw_action: torch.Tensor,
    reference_raw_action: torch.Tensor,
    carry_runtime_contract,
) -> torch.Tensor:
    """Apply the high-level CARRY policy with its recorded grasp-lock contract."""

    locked = tuple(carry_runtime_contract["locked_raw_joint_indices"])
    result = raw_action.clone()
    result[:, locked] = reference_raw_action[:, locked]
    return result


def _compose_split_carry_action(
    env,
    lower_body_wbc_raw: torch.Tensor,
    student_carry_raw: torch.Tensor,
    grasp_reference_raw: torch.Tensor,
    ordered_joint_names: tuple[str, ...],
    *,
    upper_residual_scale: float,
    max_upper_residual_rad: float,
) -> torch.Tensor:
    """Combine Homie legs/waist with a locked loaded grasp.

    The 15 non-arm/hand joints come only from the live lower-body controller.
    Fingers are always copied exactly from the successful grasp.  Shoulders,
    elbows, and wrists may use a small explicitly bounded residual, disabled by
    default, around that same physical reference.
    """

    term = env.action_manager.get_term("JointPositionAction")
    locked = tuple(
        index
        for index, name in enumerate(ordered_joint_names)
        if any(token in name for token in ("shoulder", "elbow", "wrist", "_hand_"))
    )
    fingers = tuple(
        index for index, name in enumerate(ordered_joint_names) if "_hand_" in name
    )
    arms = tuple(index for index in locked if index not in set(fingers))
    lower = tuple(index for index in range(len(ordered_joint_names)) if index not in set(locked))
    if len(locked) != 28 or len(fingers) != 14 or len(lower) != 15:
        raise RuntimeError(
            "split CARRY requires 28 arm/hand, 14 finger, and 15 leg/waist joints"
        )
    result = grasp_reference_raw.clone()
    result[:, lower] = lower_body_wbc_raw[:, lower]
    if upper_residual_scale > 0.0:
        action_scale = torch.as_tensor(
            term._scale, dtype=student_carry_raw.dtype, device=env.device
        )
        scale = action_scale if action_scale.ndim == 0 else action_scale[:, arms]
        delta_rad = (
            student_carry_raw[:, arms] - grasp_reference_raw[:, arms]
        ) * scale
        residual_rad = torch.clamp(
            upper_residual_scale * delta_rad,
            min=-max_upper_residual_rad,
            max=max_upper_residual_rad,
        )
        result[:, arms] = grasp_reference_raw[:, arms] + residual_rad / scale
    result[:, fingers] = grasp_reference_raw[:, fingers]
    return result


def _smoothstep_handoff(
    target: torch.Tensor,
    reference: torch.Tensor | None,
    step: int,
    duration_steps: int,
) -> torch.Tensor:
    """Blend raw action targets without a one-step controller discontinuity."""

    if reference is None or duration_steps <= 0 or step >= duration_steps:
        return target
    fraction = float(np.clip((step + 1) / duration_steps, 0.0, 1.0))
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    return reference + blend * (target - reference)


def _preserve_grasp_lower_body(
    target: torch.Tensor,
    reference: torch.Tensor | None,
    ordered_joint_names: tuple[str, ...],
) -> torch.Tensor:
    if reference is None:
        return target
    upper_body = tuple(
        index
        for index, name in enumerate(ordered_joint_names)
        if any(token in name for token in ("shoulder", "elbow", "wrist", "_hand_"))
    )
    if len(upper_body) != 28:
        raise RuntimeError(
            f"expected 28 arm/hand joints for grasp lower-body lock, got {len(upper_body)}"
        )
    result = reference.clone()
    result[:, upper_body] = target[:, upper_body]
    return result


def _preserve_grasp_live_legs(
    target: torch.Tensor,
    live_wbc: torch.Tensor,
    ordered_joint_names: tuple[str, ...],
) -> torch.Tensor:
    leg_indices = tuple(
        index
        for index, name in enumerate(ordered_joint_names)
        if any(token in name for token in ("_hip_", "_knee_", "_ankle_"))
    )
    if len(leg_indices) != 12:
        raise RuntimeError(
            f"expected 12 leg joints for live grasp stance, got {len(leg_indices)}"
        )
    result = target.clone()
    result[:, leg_indices] = live_wbc[:, leg_indices]
    return result


def _bound_live_leg_targets(
    env,
    live_wbc: torch.Tensor,
    reference: torch.Tensor | None,
    ordered_joint_names: tuple[str, ...],
    maximum_residual_rad: float,
) -> torch.Tensor:
    if reference is None or maximum_residual_rad <= 0.0:
        return live_wbc
    leg_indices = tuple(
        index
        for index, name in enumerate(ordered_joint_names)
        if any(token in name for token in ("_hip_", "_knee_", "_ankle_"))
    )
    term = env.action_manager.get_term("JointPositionAction")
    action_scale = torch.as_tensor(
        term._scale, dtype=live_wbc.dtype, device=env.device
    )
    scale = action_scale if action_scale.ndim == 0 else action_scale[:, leg_indices]
    maximum_raw = maximum_residual_rad / scale
    result = live_wbc.clone()
    result[:, leg_indices] = torch.maximum(
        torch.minimum(
            result[:, leg_indices], reference[:, leg_indices] + maximum_raw
        ),
        reference[:, leg_indices] - maximum_raw,
    )
    return result


def _preserve_grasp_hands(
    target: torch.Tensor,
    hand_reference: torch.Tensor,
    ordered_joint_names: tuple[str, ...],
) -> torch.Tensor:
    hand_indices = tuple(
        index for index, name in enumerate(ordered_joint_names) if "_hand_" in name
    )
    if len(hand_indices) != 14:
        raise RuntimeError(
            f"expected 14 finger joints for grasp contact lock, got {len(hand_indices)}"
        )
    result = target.clone()
    result[:, hand_indices] = hand_reference[:, hand_indices]
    return result


def _canonical_closed_hand_raw(
    env, reference: torch.Tensor, ordered_joint_names: tuple[str, ...]
) -> torch.Tensor:
    """Return ``reference`` with Arena's binary-closed Dex3 targets applied."""

    closed = {
        "left_hand_thumb_0_joint": 0.0,
        "left_hand_thumb_1_joint": 0.7,
        "left_hand_thumb_2_joint": 0.7,
        "left_hand_index_0_joint": -0.6,
        "left_hand_index_1_joint": -1.2,
        "left_hand_middle_0_joint": -0.6,
        "left_hand_middle_1_joint": -1.2,
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": -0.7,
        "right_hand_thumb_2_joint": -0.7,
        "right_hand_index_0_joint": 0.6,
        "right_hand_index_1_joint": 1.2,
        "right_hand_middle_0_joint": 0.6,
        "right_hand_middle_1_joint": 1.2,
    }
    if set(closed) != {name for name in ordered_joint_names if "_hand_" in name}:
        raise RuntimeError("runtime Dex3 joint names do not match the canonical close map")
    term = env.action_manager.get_term("JointPositionAction")
    absolute = reference * term._scale + term._offset
    for index, name in enumerate(ordered_joint_names):
        if name in closed:
            absolute[:, index] = closed[name]
    return (absolute - term._offset) / term._scale


def _set_hand_stiffness(robot, value: float) -> None:
    actuator = robot.actuators["hands"]
    actuator.stiffness[:] = float(value)
    robot.write_joint_stiffness_to_sim(
        actuator.stiffness,
        joint_ids=actuator.joint_indices,
    )


def _signals(
    env,
    observation,
    hold_steps: int,
    place_settle_steps: int,
    initial_box_height: float,
    current_phase: SkillPhase,
    left_fingertip_ids: list[int],
    right_fingertip_ids: list[int],
    left_wrist_id: int,
    right_wrist_id: int,
    handoff_tip_distance_limit_m: float,
    handoff_cage_midpoint_limit_m: float,
    handoff_wrist_midpoint_limit_m: float,
    handoff_wrist_asymmetry_limit_m: float,
) -> SkillSignals:
    observation = student_observation(observation)[0].detach().cpu().numpy()
    box_visible_slice = STUDENT_FEATURE_SLICES["box_visible"].as_slice()
    box_visible = bool(observation[box_visible_slice][0] > 0.5)
    command = env.command_manager.get_term("base_velocity")
    left, right = box_contact_counts(env)
    robot = env.scene["robot"]
    box_asset = env.scene["box"]
    box_position = box_asset.data.root_link_pos_w[0]
    box_height = float(box_position[2])
    in_bin = _inside_bin(env)
    hands_ready = bool(
        bimanual_pregrasp_ready(
            box_position.detach().cpu().numpy(),
            robot.data.body_link_pos_w[0, left_fingertip_ids].detach().cpu().numpy(),
            robot.data.body_link_pos_w[0, right_fingertip_ids].detach().cpu().numpy(),
            maximum_tip_distance_m=handoff_tip_distance_limit_m,
            maximum_cage_midpoint_error_m=handoff_cage_midpoint_limit_m,
        )[0]
    )
    wrists_centered = bool(
        bimanual_wrists_centered(
            box_position.detach().cpu().numpy(),
            robot.data.body_link_pos_w[0, left_wrist_id]
            .detach()
            .cpu()
            .numpy(),
            robot.data.body_link_pos_w[0, right_wrist_id]
            .detach()
            .cpu()
            .numpy(),
            maximum_midpoint_error_m=handoff_wrist_midpoint_limit_m,
            maximum_distance_asymmetry_m=handoff_wrist_asymmetry_limit_m,
        )[0]
    )
    released = left == 0 and right == 0
    lift_height = box_height - initial_box_height
    dropped_to_ground = box_reached_ground_outside_bin(
        box_height_m=box_height,
        environment_origin_z_m=float(env.scene.env_origins[0, 2]),
        dropped_object_height_m=0.12,
        box_in_bin=in_bin,
    )
    if current_phase == SkillPhase.CARRY:
        dropped = dropped_to_ground or lift_height < CARRY_DROP_LIFT_HEIGHT_M
    elif current_phase == SkillPhase.PLACE:
        # Release occurs above the Bin and ``in_bin`` includes z, so loss of
        # contact while the Box is still falling cannot be classified as a
        # drop. PLACE fails only after the Box reaches the ground outside the
        # delivered volume.
        dropped = dropped_to_ground
    else:
        dropped = dropped_to_ground
    return SkillSignals(
        box_visible=np.array([box_visible]),
        # Camera navigation's ``at_pregrasp`` is a base-position predicate and
        # becomes true well before the Arena reference closes its hands
        # (roughly step 30 versus close_index=126).  BC2 is certified only from
        # a realized bimanual cage, so never use the navigation predicate as a
        # manipulation handoff shortcut.
        at_pregrasp=np.array([hands_ready and wrists_centered]),
        left_contact=np.array([left > 0]),
        right_contact=np.array([right > 0]),
        lift_height_m=np.array([lift_height], dtype=np.float32),
        hold_steps=np.array([hold_steps]),
        at_staging=np.array([bool(command.at_staging[0])]),
        box_in_bin=np.array([in_bin]),
        released=np.array([released]),
        place_settle_steps=np.array([place_settle_steps], dtype=np.int64),
        box_dropped=np.array([dropped]),
        bad_orientation=np.array([_bad_orientation(env)]),
    )


def _handoff_geometry(
    env,
    left_fingertip_ids: list[int],
    right_fingertip_ids: list[int],
    left_wrist_id: int,
    right_wrist_id: int,
) -> dict[str, float]:
    """Measure the physical BC1->BC2 boundary without changing control."""

    robot = env.scene["robot"]
    box_position = (
        env.scene["box"].data.root_link_pos_w[0].detach().cpu().numpy()
    )
    left_tips = (
        robot.data.body_link_pos_w[0, left_fingertip_ids]
        .detach()
        .cpu()
        .numpy()
    )
    right_tips = (
        robot.data.body_link_pos_w[0, right_fingertip_ids]
        .detach()
        .cpu()
        .numpy()
    )
    left_tip_distances = np.linalg.norm(left_tips - box_position, axis=-1)
    right_tip_distances = np.linalg.norm(right_tips - box_position, axis=-1)
    left_closest = left_tips[int(np.argmin(left_tip_distances))]
    right_closest = right_tips[int(np.argmin(right_tip_distances))]
    left_wrist = (
        robot.data.body_link_pos_w[0, left_wrist_id].detach().cpu().numpy()
    )
    right_wrist = (
        robot.data.body_link_pos_w[0, right_wrist_id].detach().cpu().numpy()
    )
    left_wrist_distance = float(np.linalg.norm(left_wrist - box_position))
    right_wrist_distance = float(np.linalg.norm(right_wrist - box_position))
    return {
        "left_tip_distance_m": float(left_tip_distances.min()),
        "right_tip_distance_m": float(right_tip_distances.min()),
        "cage_midpoint_error_m": float(
            np.linalg.norm(0.5 * (left_closest + right_closest) - box_position)
        ),
        "wrist_midpoint_error_m": float(
            np.linalg.norm(0.5 * (left_wrist + right_wrist) - box_position)
        ),
        "wrist_distance_asymmetry_m": abs(
            left_wrist_distance - right_wrist_distance
        ),
    }


def _pre_navigate(env, bridge, episode: int) -> dict:
    """Approach research: GT scripted navigation to the canonical BC-1 start, then a Homie stand settle."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach"))
    from scripted_nav import NavGains, bc1_start_pose, nav_command, pose_error_body, wrap_to_pi

    robot = env.scene["robot"]
    pelvis = body_id(robot, "pelvis")
    pelvis_inv = math_utils.quat_inv(robot.data.body_link_quat_w[0, pelvis].unsqueeze(0))
    wrists = []
    for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
        index = body_id(robot, name)
        offset = (robot.data.body_link_pos_w[0, index] - robot.data.body_link_pos_w[0, pelvis]).unsqueeze(0)
        rel = math_utils.quat_apply(pelvis_inv, offset)[0].detach().cpu().numpy()
        quat = math_utils.quat_mul(pelvis_inv, robot.data.body_link_quat_w[0, index].unsqueeze(0))[0].detach().cpu().numpy()
        wrists.extend([*rel, *quat[[1, 2, 3, 0]]])
    target = bc1_start_pose(env.scene["box"].data.root_link_pos_w[0, :2].detach().cpu().numpy())
    jitter = np.random.default_rng([args_cli.seed, episode, 7]).uniform(-1.0, 1.0, 3) * np.asarray(args_cli.pre_navigate_target_jitter)
    target = (target[0] + float(jitter[0]), target[1] + float(jitter[1]), target[2] + float(jitter[2]))
    gains = NavGains(min_command_norm=0.10)

    def step(command) -> None:
        env.task_phase[0] = _task_phase(SkillPhase.APPROACH)
        env.command_manager.compute(0.0)
        action = np.zeros(23, dtype=np.float32)
        action[2:16] = wrists
        action[16:19] = command
        action[19] = 0.75
        names, values = bridge.step(bridge_request(env, action))
        env.step(action_order_target(env, names, values)[0])

    arrived, nav_steps = False, 0
    for nav_steps in range(args_cli.pre_navigate_max_steps):
        xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
        yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w[0])
        ex, ey, eyaw = pose_error_body(xy, yaw, target[:2], target[2])
        if math.hypot(ex, ey) <= args_cli.pre_navigate_tolerance[0] and abs(eyaw) <= args_cli.pre_navigate_tolerance[1]:
            arrived = True
            break
        step(nav_command(xy, yaw, target[:2], target[2], gains))
        if _bad_orientation(env):
            break
    for _ in range(args_cli.pre_navigate_settle_steps):
        step(np.zeros(3, dtype=np.float32))
    canonical = bc1_start_pose(env.scene["box"].data.root_link_pos_w[0, :2].detach().cpu().numpy())
    xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
    yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w[0])
    return {
        "arrived": arrived,
        "nav_steps": int(nav_steps),
        "target_jitter": [round(float(v), 4) for v in jitter],
        "handoff_offset_canonical": [round(float(xy[0] - canonical[0]), 4), round(float(xy[1] - canonical[1]), 4),
                                     round(float(wrap_to_pi(yaw - canonical[2])), 4)],
        "handoff_speed_mps": round(float(torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[0, :2])), 4),
        "fell": bool(_bad_orientation(env)),
    }


def main() -> None:
    checkpoints = {
        SkillPhase.APPROACH: args_cli.approach_checkpoint,
        SkillPhase.GRASP_LIFT: args_cli.grasp_lift_checkpoint,
        SkillPhase.CARRY: args_cli.carry_checkpoint,
        SkillPhase.PLACE: args_cli.place_checkpoint,
    }
    loaded = {
        phase: load_bc_policy(path, args_cli.device)
        for phase, path in checkpoints.items()
    }
    approach_reference_trajectory = (
        ArenaLiftTrajectory(args_cli.arena_dataset)
        if args_cli.approach_qr_endpoint_alignment
        else None
    )
    bc2_gate = validate_bc2_gate(
        args_cli.bc2_gate,
        required_action_key=loaded[SkillPhase.GRASP_LIFT].action_key,
    )
    grasp_checkpoint_certified_by_gate = (
        Path(bc2_gate["checkpoint_path"]).expanduser().resolve()
        == Path(args_cli.grasp_lift_checkpoint).expanduser().resolve()
    )
    if (
        not grasp_checkpoint_certified_by_gate
        and not args_cli.allow_uncertified_grasp_probe
    ):
        raise ValueError("--bc2_gate certifies a different grasp_lift checkpoint")
    for phase, policy in loaded.items():
        if policy.skill_phase != phase:
            raise ValueError(
                f"{checkpoints[phase]} is {policy.skill_phase.name}, expected {phase.name}"
            )
    cfg = _configure_env()
    video_path = (
        None
        if args_cli.video_output is None
        else Path(args_cli.video_output).expanduser().resolve()
    )
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
    gym_env = gym.make(
        args_cli.task,
        cfg=cfg,
        render_mode="rgb_array" if video_path is not None else None,
    )
    if video_path is not None:
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=str(video_path.parent),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length_steps,
            name_prefix=f"{video_path.stem}.raw",
            disable_logger=True,
        )
        env = _RecordingEnvProxy(gym_env)
    else:
        env = gym_env.unwrapped
    raw_joint_names = tuple(
        env.action_manager.get_term("JointPositionAction")._joint_names
    )
    action_keys = {
        phase: policy.action_key for phase, policy in loaded.items()
    }
    if (
        args_cli.grasp_live_approach_lower_body
        and action_keys[SkillPhase.APPROACH] != "teacher_high_level_action"
    ):
        raise ValueError(
            "live APPROACH lower-body control requires a 23-D teacher_high_level_action policy"
        )
    if action_keys[SkillPhase.APPROACH] != "teacher_high_level_action":
        raise ValueError(
            "split CARRY lower-body control requires the 23-D APPROACH policy as a Homie command source"
        )
    for policy in loaded.values():
        runtime_action_names = (
            raw_joint_names
            if policy.action_key == "processed_43d_joint_target"
            else DEFAULT_HIGH_LEVEL_ACTION_NAMES
        )
        validate_runtime_action_contract(policy, runtime_action_names)
    carry_runtime_contract = validate_carry_runtime_contract(
        loaded[SkillPhase.CARRY].carry_runtime_contract,
        expected_ordered_joint_names=raw_joint_names,
    )
    place_runtime_contract = validate_place_runtime_contract(
        loaded[SkillPhase.PLACE].place_runtime_contract,
        expected_ordered_joint_names=raw_joint_names,
    )
    place_phase_clock = _place_phase_clock_contract(loaded[SkillPhase.PLACE])
    place_observation_adapter = _place_observation_adapter_contract(
        loaded[SkillPhase.PLACE]
    )
    required_hold_steps = int(math.ceil(args_cli.strict_hold_seconds / env.step_dt))
    router = SkillRouter(
        loaded,
        num_envs=1,
        # The router owns only phase transitions here.  Individual policies
        # are invoked below because BC1 may emit 23-D high-level commands while
        # BC2--BC4 emit processed 43-D joint targets.
        action_dim=len(raw_joint_names),
        lift_threshold_m=0.12,
        hold_steps_required=required_hold_steps,
        place_settle_steps_required=PLACE_SETTLE_STEPS,
    )
    robot = env.scene["robot"]
    left_fingertip_ids = [body_id(robot, name) for name in LEFT_FINGERTIP_NAMES]
    right_fingertip_ids = [body_id(robot, name) for name in RIGHT_FINGERTIP_NAMES]
    left_wrist_id = body_id(robot, "left_wrist_yaw_link")
    right_wrist_id = body_id(robot, "right_wrist_yaw_link")
    pelvis_id = body_id(robot, "pelvis")
    metrics = ConditionalSuccessMetrics()
    episode_results = []
    snapshot_store = None
    if args_cli.snapshot_output:
        snapshot_store = PhysicalSnapshotStore(
            args_cli.snapshot_output,
            joint_names=list(env.scene["robot"].data.joint_names),
            overwrite=args_cli.overwrite,
            metadata={
                "source": "BC skill FSM physical rollout",
                "producer": "play_bc_skill_fsm",
                "box_teleport_into_hands": False,
                "actual_approach_grasp_chain": True,
                "approach_checkpoint": str(
                    Path(args_cli.approach_checkpoint).expanduser().resolve()
                ),
                "approach_checkpoint_sha256": file_sha256(
                    args_cli.approach_checkpoint
                ),
                "grasp_checkpoint": str(
                    Path(args_cli.grasp_lift_checkpoint).expanduser().resolve()
                ),
                "checkpoint_sha256": bc2_gate["checkpoint_sha256"],
                "action_key": bc2_gate["action_key"],
                "action_contract_sha256": bc2_gate["action_contract_sha256"],
                "bc2_gate": str(Path(args_cli.bc2_gate).expanduser().resolve()),
                "bc2_gate_sha256": file_sha256(args_cli.bc2_gate),
                "strict_hold_seconds": float(
                    bc2_gate["measured"]["strict_hold_seconds"]
                ),
                "strict_lift_m": float(bc2_gate["measured"]["strict_lift_m"]),
                "required_hold_steps": int(
                    bc2_gate["measured"]["required_hold_steps"]
                ),
                "carry_at_staging_contains_unprocessed_wbc_raw_action": True,
                "work_area_randomization": {
                    "enabled": args_cli.randomize_work_area,
                    "side_length_m": args_cli.work_area_size_m,
                    "snapshot_schema_preserves_table_and_bin_poses": True,
                },
            },
        )
    bridge = None
    if any(key == "teacher_high_level_action" for key in action_keys.values()):
        print("[FSM] starting Arena WBC bridge for high-level policy", flush=True)
        bridge = ArenaWBCBridge(arena_root=args_cli.arena_root, step_timeout_s=10.0)
        print("[FSM] Arena WBC bridge ready", flush=True)
    bad_orientation_count = 0
    handoff_eligible_episodes = 0
    handoff_survival_successes = 0
    loaded_stand_successes = 0
    walk_successes = 0
    handoff_geometry_limits = {
        "left_tip_distance_m": args_cli.handoff_tip_distance_limit_m,
        "right_tip_distance_m": args_cli.handoff_tip_distance_limit_m,
        "cage_midpoint_error_m": args_cli.handoff_cage_midpoint_limit_m,
        "wrist_midpoint_error_m": args_cli.handoff_wrist_midpoint_limit_m,
        "wrist_distance_asymmetry_m": args_cli.handoff_wrist_asymmetry_limit_m,
    }
    try:
        for episode in range(args_cli.episodes):
            if (
                args_cli.target_grasp_snapshots
                and snapshot_store is not None
                and snapshot_store.count >= args_cli.target_grasp_snapshots
            ):
                break
            print(f"[FSM] episode {episode} reset", flush=True)
            observation, _ = env.reset()
            _set_hand_stiffness(env.scene["robot"], args_cli.hand_stiffness)
            grasp_hand_stiffness_active = False
            print(f"[FSM] episode {episode} environment ready", flush=True)
            router.reset()
            if bridge is not None:
                bridge.reset()
                print(f"[FSM] episode {episode} WBC history reset", flush=True)
            pre_navigation = None
            if args_cli.pre_navigate:
                if bridge is None:
                    raise RuntimeError("--pre_navigate needs the Arena WBC bridge")
                pre_navigation = _pre_navigate(env, bridge, episode)
                print(f"[FSM] episode {episode} pre-navigation {json.dumps(pre_navigation)}", flush=True)
            initial_box_height = float(env.scene["box"].data.root_link_pos_w[0, 2])
            initial_box_xy = (
                env.scene["box"].data.root_link_pos_w[0, :2]
                .detach()
                .cpu()
                .tolist()
            )
            initial_table_xy = (
                env.table_center_e[0, :2].detach().cpu().tolist()
                if hasattr(env, "table_center_e")
                else [0.62, 0.0]
            )
            initial_bin_xy = (
                env.bin_center_e[0, :2].detach().cpu().tolist()
                if hasattr(env, "bin_center_e")
                else [BIN_CENTER_E[0], BIN_CENTER_E[1]]
            )
            initial_robot_xy = (
                env.scene["robot"].data.root_link_pos_w[0, :2]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            initial_robot_yaw = _yaw_from_quat_wxyz(
                env.scene["robot"].data.root_link_quat_w[0]
            )
            max_command_speed = 0.0
            max_command_yaw_speed = 0.0
            max_robot_displacement = 0.0
            box_visible_steps = 0
            approach_near_reach_steps = 0
            approach_ready_steps = 0
            maximum_approach_ready_steps = 0
            best_box_frustum_score = float("inf")
            final_box_top_camera = None
            max_steps = int(math.ceil(args_cli.episode_seconds / env.step_dt))
            hold_steps = 0
            max_hold_steps = 0
            max_lift_height_m = float("-inf")
            bilateral_contact_steps = 0
            place_settle_steps = 0
            carry_steps = 0
            carry_stable_steps = 0
            handoff_survived = False
            loaded_stand_survived = False
            loaded_stand_carry_step = None
            walked_to_staging = False
            place_steps = 0
            place_release_gate_steps = 0
            carry_grasp_reference = None
            grasp_handoff_reference = None
            grasp_steps = 0
            grasp_retry_count = 0
            grasp_lost_contact_steps = 0
            grasp_hold_reference = None
            grasp_hold_lower_reference = None
            grasp_hold_capture_step = None
            grasp_hand_reference = None
            grasp_hand_capture_step = None
            grasp_wbc_stance_latched = False
            grasp_contact_leg_reference = None
            live_stance_high_level = None
            grasp_lower_high_level_reference = None
            carry_lower_high_level_reference = None
            carry_teacher = None
            place_loaded_reference = None
            place_active_pink_baseline = None
            approach_filtered_action = None
            approach_ready_reference = None
            approach_qr_retargeted = None
            approach_qr_box_pose = None
            failed = False
            reason = "timeout"
            transitions_seen = []
            transition_steps = {}
            handoff_geometry_minima = {
                name: float("inf") for name in handoff_geometry_limits
            }
            best_handoff_score = float("inf")
            handoff_geometry_at_best = None
            fingertip_gate_steps = 0
            wrist_gate_steps = 0
            full_handoff_gate_steps = 0
            for step in range(max_steps):
                current_phase = SkillPhase(int(router.phase[0]))
                if (
                    current_phase == SkillPhase.GRASP_LIFT
                    and not grasp_hand_stiffness_active
                ):
                    _set_hand_stiffness(
                        env.scene["robot"], args_cli.grasp_hand_stiffness
                    )
                    grasp_hand_stiffness_active = True
                env.task_phase[0] = _task_phase(current_phase)
                env.command_manager.compute(0.0)
                # Recompute after phase synchronization so velocity command and
                # QR masks match the policy selected by the explicit router.
                # History advances only inside env.step(), once per control step.
                observation = env.observation_manager.compute(update_history=False)
                obs_np = student_observation(observation).detach().cpu().numpy()
                live_approach_lower_body = None
                if (
                    current_phase == SkillPhase.GRASP_LIFT
                    and args_cli.grasp_live_approach_lower_body
                ):
                    if bridge is None:
                        raise RuntimeError(
                            "live APPROACH lower-body control requires an Arena WBC bridge"
                        )
                    # GRASP_LIFT must hold a stance at the reached object.  A
                    # live WBC baseline is needed for balance, but carrying the
                    # camera-navigation command across the phase boundary keeps
                    # walking through the Box and prevents a stable cage.
                    approach_command = np.zeros((1, 3), dtype=np.float32)
                    approach_observation = replace_velocity_command(
                        obs_np, approach_command
                    )
                    if (
                        args_cli.grasp_latched_approach_lower_body
                        and grasp_lower_high_level_reference is not None
                    ):
                        approach_action = grasp_lower_high_level_reference.copy()
                    else:
                        approach_action = np.asarray(
                            loaded[SkillPhase.APPROACH](approach_observation),
                            dtype=np.float32,
                        )
                    use_current_wrist_stance = (
                        args_cli.grasp_wbc_current_wrist_stance
                        or (
                            args_cli.grasp_wbc_current_wrist_after_contact
                            and grasp_hand_reference is not None
                        )
                        or grasp_wbc_stance_latched
                        or (
                            args_cli.grasp_wbc_current_wrist_after_upper_hold
                            and grasp_hold_reference is not None
                        )
                    )
                    if use_current_wrist_stance:
                        approach_action = _hold_current_wrists_in_pelvis(
                            env,
                            approach_action,
                            pelvis_id,
                            left_wrist_id,
                            right_wrist_id,
                        )
                        approach_action[:, 16:19] = 0.0
                        approach_action[:, 19] = 0.75
                        approach_action[:, 20:23] = 0.0
                    # Preserve the complete learned APPROACH command here.  The
                    # certified narrow-workspace handoff used this exact output
                    # as its live stance baseline; overwriting indices 16:19 at
                    # runtime changes the reached waist/arm posture even though
                    # the observation already contains a zero velocity command.
                    live_stance_high_level = approach_action.copy()
                    names, values = bridge.step(
                        bridge_request(env, approach_action[0])
                    )
                    live_approach_lower_body, _ = action_order_target(
                        env, names, values
                    )
                    if (
                        use_current_wrist_stance
                        and grasp_hold_lower_reference is not None
                        and grasp_hold_capture_step is not None
                    ):
                        live_approach_lower_body = _smoothstep_handoff(
                            live_approach_lower_body,
                            grasp_hold_lower_reference,
                            max(0, step - grasp_hold_capture_step),
                            args_cli.grasp_post_hold_stance_blend_steps,
                        )
                if current_phase == SkillPhase.GRASP_LIFT:
                    phase_command = np.zeros((1, 3), dtype=np.float32)
                else:
                    navigation_term = env.command_manager.get_term("base_velocity")
                    phase_command = navigation_term.command.detach().cpu().numpy()
                    box_visible_steps += int(bool(navigation_term.box_visible[0]))
                    if (
                        current_phase == SkillPhase.APPROACH
                        and bool(navigation_term.box_visible[0])
                        and float(
                            torch.linalg.vector_norm(
                                navigation_term.pregrasp_position_error_b[0]
                            )
                        )
                        <= 0.35
                    ):
                        approach_near_reach_steps += 1
                    final_box_top_camera = _box_top_position_camera(
                        env, navigation_term
                    )
                    forward = max(float(final_box_top_camera[0]), 1.0e-6)
                    tan_horizontal = math.tan(
                        math.radians(navigation_term.cfg.horizontal_fov_deg) * 0.5
                    )
                    tan_vertical = (
                        tan_horizontal
                        * navigation_term.cfg.image_height
                        / navigation_term.cfg.image_width
                    )
                    best_box_frustum_score = min(
                        best_box_frustum_score,
                        max(
                            abs(float(final_box_top_camera[1]))
                            / (forward * tan_horizontal),
                            abs(float(final_box_top_camera[2]))
                            / (forward * tan_vertical),
                        ),
                    )
                    if current_phase == SkillPhase.CARRY:
                        zero_command_steps = (
                            args_cli.loaded_stabilization_steps
                            + args_cli.loaded_stand_steps
                        )
                        # Loaded stand is a one-way physical gate.  Once the
                        # required consecutive hold has passed, a transient
                        # one-frame contact/lift dip must not silently send
                        # the controller back to zero velocity for another
                        # full 350-step qualification window.  Actual drops
                        # remain terminal through the FSM failure signals.
                        if not loaded_stand_survived:
                            phase_command = np.zeros_like(phase_command)
                        else:
                            if loaded_stand_carry_step is None:
                                raise RuntimeError(
                                    "loaded stand latch is missing its CARRY phase step"
                                )
                            walk_step = carry_steps - loaded_stand_carry_step
                            phase_command = phase_command * carry_navigation_scale(
                                max(0, walk_step), carry_runtime_contract
                            )
                            # Slow loaded translation without weakening the
                            # yaw servo needed to keep the Bin centred in the
                            # camera during the curved approach.
                            phase_command = phase_command.copy()
                            phase_command[:, :2] *= args_cli.carry_velocity_scale
                obs_np = replace_velocity_command(obs_np, phase_command)
                max_command_speed = max(
                    max_command_speed,
                    float(np.linalg.norm(phase_command[0, :2])),
                )
                max_command_yaw_speed = max(
                    max_command_yaw_speed,
                    float(abs(phase_command[0, 2])),
                )
                current_policy = loaded[current_phase]
                current_action_key = current_policy.action_key
                if current_phase == SkillPhase.PLACE and place_phase_clock is not None:
                    obs_np = obs_np.copy()
                    obs_np[:, place_phase_clock["feature_index"]] = min(
                        place_steps / place_phase_clock["horizon_steps"], 1.0
                    )
                if (
                    current_phase == SkillPhase.PLACE
                    and place_observation_adapter is not None
                ):
                    obs_np = obs_np.copy()
                    for feature_name in place_observation_adapter[
                        "masked_feature_names"
                    ]:
                        obs_np[:, STUDENT_FEATURE_SLICES[feature_name].as_slice()] = 0.0
                policy_action = np.asarray(current_policy(obs_np), dtype=np.float32)
                expected_shape = (1, current_policy.action_dim)
                if policy_action.shape != expected_shape or not np.isfinite(
                    policy_action
                ).all():
                    raise ValueError(
                        f"{current_phase.name} policy returned {policy_action.shape}; "
                        f"expected finite {expected_shape}"
                    )
                if (
                    current_phase == SkillPhase.CARRY
                    and current_action_key == "teacher_high_level_action"
                ):
                    # The navigation velocity is an explicit task command that
                    # is already present in the 517-D actor observation.  Do
                    # not make loaded walking depend on an MLP reconstructing
                    # this known input: route it losslessly to Homie while the
                    # Student predicts the manipulation part of the 23-D
                    # command.  This uses no privileged simulator state.
                    policy_action[:, 16:19] = phase_command
                    # Pelvis height and torso attitude are fixed task-level
                    # objectives in every recorded Carry Teacher action.  As
                    # with the navigation command above, routing these known
                    # setpoints losslessly avoids asking the BC network to
                    # reconstruct constants whose small errors can bias the
                    # whole-body walking solve.
                    policy_action[:, 19] = 0.75
                    policy_action[:, 20:23] = 0.0
                    if args_cli.carry_current_wrist_wbc:
                        # The upper-body raw targets are locked to the
                        # realized grasp below.  Give Pink the matching live
                        # wrist constraints as well, otherwise a stale BC
                        # wrist proposal can make the lower-body solve fight
                        # the grasp lock during loaded walking.  Wrist poses
                        # are robot proprioception, not privileged Box state.
                        policy_action = _hold_current_wrists_in_pelvis(
                            env,
                            policy_action,
                            pelvis_id,
                            left_wrist_id,
                            right_wrist_id,
                        )
                elif (
                    current_phase == SkillPhase.APPROACH
                    and current_action_key == "teacher_high_level_action"
                ):
                    if args_cli.approach_hand_state_decoding == "binary":
                        policy_action[:, :2] = (policy_action[:, :2] >= 0.5).astype(
                            np.float32
                        )
                    if approach_ready_reference is not None:
                        policy_action = approach_ready_reference.copy()
                        policy_action[:, 16:19] = 0.0
                    if args_cli.approach_task_target_passthrough:
                        # Searching over a large workspace is an explicit
                        # camera-derived task command.  This mode is opt-in:
                        # the narrow-workspace checkpoint was physically
                        # certified with all 23 learned outputs intact.
                        policy_action[:, 16:19] = phase_command
                        policy_action[:, 19] = 0.75
                        policy_action[:, 20:23] = 0.0
                        if not bool(navigation_term.box_visible[0]):
                            policy_action[:, 21] = (
                                args_cli.approach_search_torso_pitch_rad
                            )
                    pregrasp_error_m = float(
                        torch.linalg.vector_norm(
                            navigation_term.pregrasp_position_error_b[0]
                        )
                    )
                    if args_cli.approach_task_target_passthrough and (
                        not bool(navigation_term.box_visible[0])
                        or pregrasp_error_m > 0.35
                    ):
                        policy_action = _hold_current_wrists_in_pelvis(
                            env,
                            policy_action,
                            pelvis_id,
                            left_wrist_id,
                            right_wrist_id,
                        )
                    elif approach_reference_trajectory is not None:
                        if approach_qr_retargeted is None:
                            observed_box_pose = _qr_observed_box_pose_wxyz(
                                env, navigation_term
                            )
                            approach_qr_box_pose = observed_box_pose.copy()
                            approach_qr_retargeted = (
                                approach_reference_trajectory.retarget(
                                    pose_matrix_wxyz(observed_box_pose)
                                )
                            )
                        reference_action = (
                            approach_reference_trajectory.high_level_action(
                                approach_reference_trajectory.close_index
                                + args_cli.approach_endpoint_offset_steps,
                                body_pose_wxyz(robot, pelvis_id)
                                .detach()
                                .cpu()
                                .numpy(),
                                approach_qr_retargeted,
                            )
                        )
                        reference_action[:2] = 0.0
                        pelvis_pose = pose_matrix_wxyz(
                            body_pose_wxyz(robot, pelvis_id)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        box_position_pelvis = (
                            np.linalg.inv(pelvis_pose)
                            @ pose_matrix_wxyz(approach_qr_box_pose)
                        )[:3, 3]
                        wrist_midpoint = 0.5 * (
                            reference_action[2:5] + reference_action[9:12]
                        )
                        centering_delta = box_position_pelvis - wrist_midpoint
                        centering_norm = float(np.linalg.norm(centering_delta))
                        if centering_norm > 1.0e-8:
                            centering_delta *= min(
                                args_cli.approach_endpoint_centering_m,
                                centering_norm,
                            ) / centering_norm
                            reference_action[2:5] += centering_delta
                            reference_action[9:12] += centering_delta
                        if args_cli.approach_wrist_servo_gain > 0.0:
                            reference_action = _apply_qr_proprio_wrist_servo(
                                reference_action,
                                observed_box_pose_wxyz=approach_qr_box_pose,
                                pelvis_pose_wxyz=(
                                    body_pose_wxyz(robot, pelvis_id)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                actual_left_wrist_position_w=(
                                    robot.data.body_link_pos_w[0, left_wrist_id]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                actual_right_wrist_position_w=(
                                    robot.data.body_link_pos_w[0, right_wrist_id]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                gain=args_cli.approach_wrist_servo_gain,
                                maximum_correction_m=(
                                    args_cli.approach_wrist_servo_max_correction_m
                                ),
                            )
                        policy_action[:, :16] = reference_action[:16]
                elif (
                    current_phase == SkillPhase.PLACE
                    and current_action_key == "teacher_high_level_action"
                ):
                    # PLACE is stationary while the wrists translate the Box
                    # into the Bin.  These are public FSM commands, not
                    # simulator-only state, so retain only the Student's
                    # manipulation proposal and provide the exact recorded
                    # base objective to Pink/Homie.
                    policy_action[:, 16:19] = 0.0
                    policy_action[:, 19] = 0.75
                    policy_action[:, 20:23] = 0.0
                    if args_cli.place_release_when_in_bin and _inside_bin(env):
                        policy_action[:, :2] = 0.0
                        place_release_gate_steps += 1
                if current_phase == SkillPhase.APPROACH:
                    if approach_filtered_action is None:
                        if current_action_key == "processed_43d_joint_target":
                            action_term = env.action_manager.get_term(
                                "JointPositionAction"
                            )
                            approach_filtered_action = (
                                action_term._offset.detach()
                                .cpu()
                                .numpy()
                                .reshape(1, -1)
                                .copy()
                            )
                        else:
                            approach_filtered_action = policy_action.copy()
                    approach_filtered_action += args_cli.approach_action_alpha * (
                        policy_action - approach_filtered_action
                    )
                    if current_action_key == "teacher_high_level_action":
                        for quaternion_slice in (slice(5, 9), slice(12, 16)):
                            quaternion = approach_filtered_action[
                                :, quaternion_slice
                            ]
                            norm = np.linalg.norm(
                                quaternion, axis=1, keepdims=True
                            )
                            if np.any(norm <= 1.0e-8):
                                raise RuntimeError(
                                    "APPROACH policy produced a degenerate wrist quaternion"
                                )
                            approach_filtered_action[:, quaternion_slice] = (
                                quaternion / norm
                            )
                    policy_action = approach_filtered_action.copy()
                unprocessed_wbc_raw_action = None
                if current_action_key == "processed_43d_joint_target":
                    raw_action = _absolute_to_raw(env, policy_action)
                    if current_phase == SkillPhase.GRASP_LIFT:
                        raw_action = _smoothstep_handoff(
                            raw_action,
                            grasp_handoff_reference,
                            grasp_steps,
                            args_cli.grasp_handoff_steps,
                        )
                        if args_cli.grasp_lower_body_lock:
                            raw_action = _preserve_grasp_lower_body(
                                raw_action,
                                grasp_handoff_reference,
                                raw_joint_names,
                            )
                        elif live_approach_lower_body is not None:
                            active_live_legs = (
                                grasp_contact_leg_reference
                                if grasp_contact_leg_reference is not None
                                else live_approach_lower_body
                            )
                            active_live_legs = _bound_live_leg_targets(
                                env,
                                active_live_legs,
                                grasp_handoff_reference,
                                raw_joint_names,
                                args_cli.grasp_live_leg_max_residual_rad,
                            )
                            raw_action = _preserve_grasp_live_legs(
                                raw_action,
                                active_live_legs,
                                raw_joint_names,
                            )
                            if (
                                args_cli.grasp_freeze_upper_on_lift
                                and grasp_hold_reference is not None
                            ):
                                if args_cli.grasp_freeze_loaded_posture_on_upper_hold:
                                    raw_action = grasp_hold_reference.detach().clone()
                                else:
                                    raw_action = _preserve_grasp_live_legs(
                                        grasp_hold_reference,
                                        active_live_legs,
                                        raw_joint_names,
                                    )
                            # Apply the first-contact finger cage last.  The
                            # later upper-body hold snapshot may already
                            # contain a reopening Student target and must not
                            # weaken the retained grasp.
                            if (
                                args_cli.grasp_freeze_hands_on_contact
                                and grasp_hand_reference is not None
                            ):
                                raw_action = _preserve_grasp_hands(
                                    raw_action,
                                    grasp_hand_reference,
                                    raw_joint_names,
                                )
                    elif current_phase == SkillPhase.CARRY:
                        if bridge is None or carry_grasp_reference is None:
                            raise RuntimeError(
                                "split CARRY requires Arena WBC and a realized grasp reference"
                            )
                        if carry_lower_high_level_reference is None or carry_teacher is None:
                            raise RuntimeError(
                                "CARRY began without the realized grasp controller state"
                            )
                        carry_student_raw = raw_action.detach().clone()
                        # Rebuild the wrist commands from the realized
                        # Box-relative grasp every step, exactly as the
                        # successful Carry Teacher collector does. Reusing the
                        # final GRASP wrist pose here makes Pink fight a stale
                        # world target and suppresses/perturbs loaded walking.
                        lower_high_level = carry_teacher.action(
                            current_box_pose_wxyz=(
                                env.scene["box"].data.root_link_state_w[0, :7]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            current_pelvis_pose_wxyz=(
                                body_pose_wxyz(robot, pelvis_id)
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            camera_navigation_velocity=phase_command[0],
                        ).reshape(1, -1)
                        names, values = bridge.step(
                            bridge_request(env, lower_high_level[0])
                        )
                        lower_body_raw, _ = action_order_target(env, names, values)
                        unprocessed_wbc_raw_action = lower_body_raw.detach().clone()
                        zero_command_steps = (
                            args_cli.loaded_stabilization_steps
                            + args_cli.loaded_stand_steps
                        )
                        if carry_stable_steps < zero_command_steps:
                            lower_body_source = _smoothstep_handoff(
                                lower_body_raw,
                                carry_grasp_reference,
                                carry_steps,
                                args_cli.carry_handoff_steps,
                            )
                        elif args_cli.carry_lower_body_mode == "wbc":
                            lower_body_source = lower_body_raw
                        elif args_cli.carry_lower_body_mode == "blend":
                            lower_body_source = _smoothstep_handoff(
                                carry_student_raw,
                                lower_body_raw,
                                max(0, carry_steps - zero_command_steps),
                                args_cli.carry_student_lower_blend_steps,
                            )
                        else:
                            lower_body_source = carry_student_raw
                        raw_action = _compose_split_carry_action(
                            env,
                            lower_body_source,
                            carry_student_raw,
                            carry_grasp_reference,
                            raw_joint_names,
                            upper_residual_scale=args_cli.carry_upper_residual_scale,
                            max_upper_residual_rad=args_cli.carry_max_upper_residual_rad,
                        )
                else:
                    if bridge is None:
                        raise RuntimeError("high-level policy requires an Arena WBC bridge")
                    names, values = bridge.step(bridge_request(env, policy_action[0]))
                    raw_action, _ = action_order_target(env, names, values)
                    unprocessed_wbc_raw_action = raw_action.detach().clone()
                    if current_phase == SkillPhase.CARRY:
                        if carry_grasp_reference is None:
                            raise RuntimeError(
                                "CARRY began without the realized GRASP_LIFT target"
                            )
                        if (
                            args_cli.carry_upper_residual_scale > 0.0
                            and loaded_stand_survived
                        ):
                            # For a high-level Student, Pink's arm solution is
                            # the deployable manipulation proposal.  Admit a
                            # small bounded shoulder/elbow/wrist correction
                            # around the realized grasp so walking-induced
                            # pelvis-height motion does not lower the Box;
                            # the 14 finger targets remain exactly locked.
                            residual_ramp = float(
                                np.clip(
                                    (carry_steps - loaded_stand_carry_step + 1)
                                    / args_cli.carry_student_lower_blend_steps,
                                    0.0,
                                    1.0,
                                )
                            )
                            raw_action = _compose_split_carry_action(
                                env,
                                raw_action,
                                raw_action,
                                carry_grasp_reference,
                                raw_joint_names,
                                upper_residual_scale=(
                                    args_cli.carry_upper_residual_scale
                                    * residual_ramp
                                ),
                                max_upper_residual_rad=(
                                    args_cli.carry_max_upper_residual_rad
                                ),
                            )
                        else:
                            raw_action = _preserve_carry_grasp_targets(
                                raw_action,
                                carry_grasp_reference,
                                carry_runtime_contract,
                            )
                    elif current_phase == SkillPhase.PLACE:
                        if (
                            place_loaded_reference is None
                            or place_active_pink_baseline is None
                        ):
                            raise RuntimeError(
                                "PLACE began without loaded/Pink CARRY transition references"
                            )
                        raw_action = (
                            apply_place_runtime_postprocessor(
                                raw_action,
                                place_loaded_reference,
                                place_active_pink_baseline,
                                policy_action[0, :2],
                                place_steps,
                                place_runtime_contract,
                            )
                        )
                observation, _, _, _, _ = env.step(raw_action)
                robot_xy = (
                    env.scene["robot"].data.root_link_pos_w[0, :2]
                    .detach()
                    .cpu()
                    .numpy()
                )
                max_robot_displacement = max(
                    max_robot_displacement,
                    float(np.linalg.norm(robot_xy - initial_robot_xy)),
                )

                handoff_geometry = _handoff_geometry(
                    env,
                    left_fingertip_ids,
                    right_fingertip_ids,
                    left_wrist_id,
                    right_wrist_id,
                )
                for name, value in handoff_geometry.items():
                    handoff_geometry_minima[name] = min(
                        handoff_geometry_minima[name], value
                    )
                handoff_score = max(
                    handoff_geometry[name] / limit
                    for name, limit in handoff_geometry_limits.items()
                )
                if handoff_score < best_handoff_score:
                    best_handoff_score = handoff_score
                    handoff_geometry_at_best = dict(handoff_geometry)
                fingertip_gate = all(
                    handoff_geometry[name] <= handoff_geometry_limits[name]
                    for name in (
                        "left_tip_distance_m",
                        "right_tip_distance_m",
                        "cage_midpoint_error_m",
                    )
                )
                wrist_gate = all(
                    handoff_geometry[name] <= handoff_geometry_limits[name]
                    for name in (
                        "wrist_midpoint_error_m",
                        "wrist_distance_asymmetry_m",
                    )
                )
                fingertip_gate_steps += int(fingertip_gate)
                wrist_gate_steps += int(wrist_gate)
                full_handoff_gate_steps += int(fingertip_gate and wrist_gate)

                # Reward-side phase logic may have advanced during env.step().
                # Restore the explicit router phase before querying navigation
                # predicates, without appending another history frame.
                env.task_phase[0] = _task_phase(current_phase)
                env.command_manager.compute(0.0)
                observation = env.observation_manager.compute(update_history=False)

                left_count, right_count = box_contact_counts(env)
                lift_height = (
                    float(env.scene["box"].data.root_link_pos_w[0, 2])
                    - initial_box_height
                )
                if (
                    current_phase == SkillPhase.GRASP_LIFT
                    and args_cli.grasp_wbc_current_wrist_min_lift_m > 0.0
                    and lift_height >= args_cli.grasp_wbc_current_wrist_min_lift_m
                ):
                    grasp_wbc_stance_latched = True
                hold_steps = (
                    hold_steps + 1
                    if left_count > 0 and right_count > 0 and lift_height >= 0.12
                    else 0
                )
                if current_phase == SkillPhase.CARRY:
                    carry_stable_steps = (
                        carry_stable_steps + 1
                        if left_count > 0 and right_count > 0 and lift_height >= 0.12
                        else 0
                    )
                    if (
                        not handoff_survived
                        and carry_stable_steps >= args_cli.loaded_stabilization_steps
                    ):
                        handoff_survived = True
                        transition_steps["handoff_survival"] = step
                    if (
                        not loaded_stand_survived
                        and carry_stable_steps
                        >= args_cli.loaded_stabilization_steps
                        + args_cli.loaded_stand_steps
                    ):
                        loaded_stand_survived = True
                        loaded_stand_carry_step = carry_steps
                        transition_steps["loaded_stand"] = step
                if (
                    current_phase == SkillPhase.GRASP_LIFT
                    and args_cli.grasp_latch_live_legs_on_contact
                    and grasp_contact_leg_reference is None
                    and live_approach_lower_body is not None
                    and left_count > 0
                    and right_count > 0
                ):
                    grasp_contact_leg_reference = (
                        live_approach_lower_body.detach().clone()
                    )
                if (
                    current_phase == SkillPhase.GRASP_LIFT
                    and args_cli.grasp_freeze_hands_on_contact
                    and grasp_hand_reference is None
                    and left_count > 0
                    and right_count > 0
                    and lift_height >= args_cli.grasp_hand_freeze_min_lift_m
                ):
                    grasp_hand_reference = env.action_manager.action.detach().clone()
                    if args_cli.grasp_canonical_hand_lock:
                        grasp_hand_reference = _canonical_closed_hand_raw(
                            env, grasp_hand_reference, raw_joint_names
                        )
                    grasp_hand_capture_step = step
                if (
                    current_phase == SkillPhase.GRASP_LIFT
                    and args_cli.grasp_freeze_upper_on_lift
                    and grasp_hold_reference is None
                    and left_count > 0
                    and right_count > 0
                    and lift_height >= args_cli.grasp_upper_freeze_min_lift_m
                ):
                    grasp_hold_reference = env.action_manager.action.detach().clone()
                    grasp_hold_capture_step = step
                    if live_approach_lower_body is not None:
                        grasp_hold_lower_reference = (
                            live_approach_lower_body.detach().clone()
                        )
                    if (
                        args_cli.grasp_freeze_hands_on_contact
                        and args_cli.grasp_refresh_hand_lock_on_upper_hold
                    ):
                        grasp_hand_reference = grasp_hold_reference.detach().clone()
                        if args_cli.grasp_canonical_hand_lock:
                            grasp_hand_reference = _canonical_closed_hand_raw(
                                env, grasp_hand_reference, raw_joint_names
                            )
                        grasp_hand_capture_step = step
                max_hold_steps = max(max_hold_steps, hold_steps)
                max_lift_height_m = max(max_lift_height_m, lift_height)
                bilateral_contact_steps += int(left_count > 0 and right_count > 0)
                in_bin = _inside_bin(env)
                released = left_count == 0 and right_count == 0
                box_speed = float(
                    torch.linalg.vector_norm(
                        env.scene["box"].data.root_link_lin_vel_w[0]
                    )
                )
                place_settled = (
                    current_phase == SkillPhase.PLACE
                    and in_bin
                    and released
                    and box_speed <= PLACE_SETTLE_MAX_BOX_SPEED_MPS
                )
                place_settle_steps = (
                    place_settle_steps + 1 if place_settled else 0
                )
                signals = _signals(
                    env,
                    observation,
                    hold_steps,
                    place_settle_steps,
                    initial_box_height,
                    current_phase,
                    left_fingertip_ids,
                    right_fingertip_ids,
                    left_wrist_id,
                    right_wrist_id,
                    args_cli.handoff_tip_distance_limit_m,
                    args_cli.handoff_cage_midpoint_limit_m,
                    args_cli.handoff_wrist_midpoint_limit_m,
                    args_cli.handoff_wrist_asymmetry_limit_m,
                )
                if current_phase == SkillPhase.APPROACH:
                    physically_ready = bool(signals.at_pregrasp[0])
                    if physically_ready and approach_ready_reference is None:
                        approach_ready_reference = policy_action.copy()
                        approach_ready_reference[:, 16:19] = 0.0
                    if approach_ready_reference is not None:
                        approach_ready_steps += 1
                    maximum_approach_ready_steps = max(
                        maximum_approach_ready_steps, approach_ready_steps
                    )
                    signals.at_pregrasp[:] = (
                        approach_ready_reference is not None
                        and
                        approach_ready_steps >= args_cli.approach_ready_hold_steps
                    )
                    if approach_ready_reference is not None:
                        # The target was latched from a valid QR-visible
                        # physical cage.  Preserve that public observation
                        # event through the short zero-velocity settling
                        # interval even if the top marker leaves the frustum.
                        signals.box_visible[:] = True
                if current_phase == SkillPhase.CARRY and not loaded_stand_survived:
                    signals.at_staging[:] = False
                transitions = router.update(signals)
                if transitions.approach_succeeded[0]:
                    transitions_seen.append("approach")
                    transition_steps["approach"] = step
                    grasp_handoff_reference = (
                        env.action_manager.action.detach().clone()
                    )
                    if current_action_key == "teacher_high_level_action":
                        grasp_lower_high_level_reference = policy_action.copy()
                    grasp_steps = 0
                if transitions.grasp_succeeded[0]:
                    transitions_seen.append("grasp_lift")
                    transition_steps["grasp_lift"] = step
                    carry_grasp_reference = env.action_manager.action.detach().clone()
                    carry_lower_high_level_reference = (
                        None
                        if live_stance_high_level is None
                        else live_stance_high_level.copy()
                    )
                    realized_grasp = RealizedBimanualGrasp.capture(
                        box_pose_wxyz=(
                            env.scene["box"].data.root_link_state_w[0, :7]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        left_wrist_pose_wxyz=(
                            body_pose_wxyz(robot, left_wrist_id)
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        right_wrist_pose_wxyz=(
                            body_pose_wxyz(robot, right_wrist_id)
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    )
                    carry_teacher = CarryTeacher(
                        realized_grasp,
                        minimum_box_height_w=float(
                            env.scene["box"].data.root_link_pos_w[0, 2]
                        ),
                    )
                    carry_steps = 0
                    carry_stable_steps = 0
                    if snapshot_store is not None:
                        snapshot_store.append(
                            capture_physical_snapshot(
                                env,
                                kind=SnapshotKind.BILATERAL_HOLD,
                                env_id=0,
                                source_episode=f"fsm_episode_{episode}",
                                source_step=step,
                                student_observation=student_observation(observation)[0],
                                box_initial_height_m=initial_box_height,
                                left_contact_count=left_count,
                                right_contact_count=right_count,
                                stable_hold_steps=hold_steps,
                                at_staging=False,
                                bad_orientation=False,
                                box_dropped=False,
                                box_size=np.full(3, BOX_SIZE_M),
                                box_mass_kg=args_cli.box_mass_kg,
                            )
                        )
                    if args_cli.stop_after_grasp_snapshot:
                        reason = "grasp_snapshot_collected"
                        break
                    handoff_eligible_episodes += 1
                if transitions.carry_succeeded[0]:
                    transitions_seen.append("carry")
                    transition_steps["carry"] = step
                    walked_to_staging = True
                    walk_successes += 1
                    if unprocessed_wbc_raw_action is None:
                        raise RuntimeError(
                            "CARRY transition has no unprocessed Pink/WBC baseline"
                        )
                    place_loaded_reference = env.action_manager.action.detach().clone()
                    place_active_pink_baseline = unprocessed_wbc_raw_action
                    place_steps = 0
                    if snapshot_store is not None:
                        snapshot_store.append(
                            capture_physical_snapshot(
                                env,
                                kind=SnapshotKind.CARRY_AT_STAGING,
                                env_id=0,
                                source_episode=f"fsm_episode_{episode}",
                                source_step=step,
                                student_observation=student_observation(observation)[0],
                                box_initial_height_m=initial_box_height,
                                left_contact_count=left_count,
                                right_contact_count=right_count,
                                stable_hold_steps=max(hold_steps, required_hold_steps),
                                at_staging=True,
                                bad_orientation=False,
                                box_dropped=False,
                                box_size=np.full(3, BOX_SIZE_M),
                                box_mass_kg=args_cli.box_mass_kg,
                                unprocessed_wbc_raw_action=(
                                    None
                                    if unprocessed_wbc_raw_action is None
                                    else unprocessed_wbc_raw_action[0]
                                ),
                            )
                        )
                    if args_cli.stop_after_carry_snapshot:
                        reason = "carry_snapshot_collected"
                        break
                if transitions.place_succeeded[0]:
                    transitions_seen.append("place")
                    transition_steps["place"] = step
                    reason = "full_task_success"
                    break
                if transitions.failed[0]:
                    failed = True
                    reason = (
                        "bad_orientation"
                        if bool(signals.bad_orientation[0])
                        else "box_drop"
                    )
                    break
                if current_phase == SkillPhase.GRASP_LIFT:
                    table_height_grasp_loss = (
                        lift_height < CARRY_DROP_LIFT_HEIGHT_M
                        and not (left_count > 0 and right_count > 0)
                    )
                    grasp_lost_contact_steps = (
                        grasp_lost_contact_steps + 1
                        if table_height_grasp_loss
                        else 0
                    )
                    if (
                        grasp_retry_count < args_cli.max_grasp_retries
                        and grasp_steps >= args_cli.grasp_retry_min_phase_steps
                        and grasp_lost_contact_steps
                        >= args_cli.grasp_retry_lost_contact_steps
                    ):
                        # The Box is back at table height and the current
                        # finger/upper-body latches no longer constitute a
                        # viable grasp. Re-open and realign through the normal
                        # camera-based APPROACH policy instead of replaying a
                        # dead GRASP_LIFT state until timeout.
                        router.phase[0] = int(SkillPhase.APPROACH)
                        _set_hand_stiffness(
                            env.scene["robot"], args_cli.hand_stiffness
                        )
                        grasp_hand_stiffness_active = False
                        grasp_retry_count += 1
                        grasp_lost_contact_steps = 0
                        grasp_steps = 0
                        grasp_handoff_reference = None
                        grasp_hold_reference = None
                        grasp_hold_lower_reference = None
                        grasp_hold_capture_step = None
                        grasp_hand_reference = None
                        grasp_hand_capture_step = None
                        grasp_contact_leg_reference = None
                        grasp_wbc_stance_latched = False
                        grasp_lower_high_level_reference = None
                        approach_filtered_action = None
                        approach_ready_reference = None
                        approach_ready_steps = 0
                        continue
                if args_cli.stop_after_handoff_gate and handoff_survived:
                    reason = "handoff_gate_passed"
                    break
                if current_phase == SkillPhase.GRASP_LIFT:
                    grasp_steps += 1
                    if (
                        args_cli.grasp_phase_timeout_seconds > 0.0
                        and grasp_steps
                        >= int(
                            math.ceil(
                                args_cli.grasp_phase_timeout_seconds / env.step_dt
                            )
                        )
                    ):
                        reason = "grasp_phase_timeout"
                        break
                elif current_phase == SkillPhase.CARRY:
                    carry_steps += 1
                elif current_phase == SkillPhase.PLACE:
                    place_steps += 1
                if args_cli.real_time:
                    env.sim.render()

            final_phase = SkillPhase(int(router.phase[0]))
            metrics.add_episode(final_phase, failed=failed)
            handoff_survival_successes += int(handoff_survived)
            loaded_stand_successes += int(loaded_stand_survived)
            bad_orientation_count += int(reason == "bad_orientation")
            result = {
                "episode": episode,
                "pre_navigation": pre_navigation,
                "final_phase": final_phase.name.lower(),
                "failed": failed,
                "reason": reason,
                "steps": step + 1,
                "transitions": transitions_seen,
                "transition_steps": transition_steps,
                "initial_box_xy_m": initial_box_xy,
                "initial_table_xy_m": initial_table_xy,
                "initial_bin_xy_m": initial_bin_xy,
                "final_robot_xy_m": robot_xy.tolist(),
                "robot_yaw_change_rad": _yaw_from_quat_wxyz(
                    env.scene["robot"].data.root_link_quat_w[0]
                )
                - initial_robot_yaw,
                "box_visible_steps": box_visible_steps,
                "approach_near_reach_steps": approach_near_reach_steps,
                "maximum_approach_ready_steps": maximum_approach_ready_steps,
                "best_box_frustum_normalized_score": best_box_frustum_score,
                "final_box_top_position_camera_m": (
                    None
                    if final_box_top_camera is None
                    else final_box_top_camera.tolist()
                ),
                "max_navigation_command_mps": max_command_speed,
                "max_navigation_yaw_command_rad_s": max_command_yaw_speed,
                "max_robot_displacement_m": max_robot_displacement,
                "handoff_geometry_limits_m": handoff_geometry_limits,
                "handoff_geometry_minima_m": handoff_geometry_minima,
                "best_handoff_normalized_score": best_handoff_score,
                "handoff_geometry_at_best_m": handoff_geometry_at_best,
                "fingertip_gate_steps": fingertip_gate_steps,
                "wrist_gate_steps": wrist_gate_steps,
                "full_handoff_gate_steps": full_handoff_gate_steps,
                "bilateral_contact_steps": bilateral_contact_steps,
                "max_lift_height_m": max_lift_height_m,
                "max_continuous_bimanual_lift_hold_steps": max_hold_steps,
                "carry_stable_steps": carry_stable_steps,
                "handoff_survival_success": handoff_survived,
                "loaded_stand_success": loaded_stand_survived,
                "walk_success": walked_to_staging,
                "grasp_hold_capture_step": grasp_hold_capture_step,
                "grasp_hand_capture_step": grasp_hand_capture_step,
                "grasp_retry_count": grasp_retry_count,
                "place_release_gate_steps": place_release_gate_steps,
            }
            episode_results.append(result)
            print(result, flush=True)
    finally:
        if bridge is not None:
            bridge.close()
        if snapshot_store is not None:
            if snapshot_store.count:
                snapshot_store.finalize()
            else:
                snapshot_store.abort()
        env.close()

    result = {
        **metrics.as_dict(),
        "seed": args_cli.seed,
        "box_mass_kg": args_cli.box_mass_kg,
        "hand_stiffness": args_cli.hand_stiffness,
        "grasp_hand_stiffness": args_cli.grasp_hand_stiffness,
        "work_area_randomization": {
            "enabled": args_cli.randomize_work_area,
            "side_length_m": args_cli.work_area_size_m,
            "box_local_x_range_m": list(args_cli.box_local_x_range),
            "box_local_y_range_m": list(args_cli.box_local_y_range),
        },
        "camera_horizontal_fov_deg": float(
            cfg.commands.base_velocity.horizontal_fov_deg
        ),
        "bimanual_grasp_lift_success_rate": metrics.grasp_successes / metrics.episodes,
        "bad_orientation_rate": bad_orientation_count / metrics.episodes,
        "loaded_stabilization_steps": args_cli.loaded_stabilization_steps,
        "loaded_stand_steps": args_cli.loaded_stand_steps,
        "carry_upper_residual_scale": args_cli.carry_upper_residual_scale,
        "carry_max_upper_residual_rad": args_cli.carry_max_upper_residual_rad,
        "carry_lower_body_mode": args_cli.carry_lower_body_mode,
        "carry_student_lower_blend_steps": args_cli.carry_student_lower_blend_steps,
        "carry_staging_standoff_m": args_cli.carry_staging_standoff_m,
        "carry_staging_position_tolerance_m": (
            args_cli.carry_staging_position_tolerance_m
        ),
        "carry_staging_yaw_tolerance_rad": (
            args_cli.carry_staging_yaw_tolerance_rad
        ),
        "carry_velocity_scale": args_cli.carry_velocity_scale,
        "carry_current_wrist_wbc": args_cli.carry_current_wrist_wbc,
        "place_release_when_in_bin": args_cli.place_release_when_in_bin,
        "place_phase_clock": place_phase_clock,
        "place_observation_adapter": place_observation_adapter,
        "carry_control_partition": {
            "legs_and_waist": (
                "live Homie WBC during loaded stabilization; "
                f"{args_cli.carry_lower_body_mode} after walking begins"
            ),
            "shoulder_elbow_wrist": "grasp reference plus bounded residual",
            "fingers": "exact grasp reference lock",
        },
        "handoff_gate": {
            "eligible_grasps": handoff_eligible_episodes,
            "successes": handoff_survival_successes,
            "conditional_probability": (
                handoff_survival_successes / handoff_eligible_episodes
                if handoff_eligible_episodes
                else None
            ),
        },
        "loaded_stand_gate": {
            "eligible_handoffs": handoff_survival_successes,
            "successes": loaded_stand_successes,
            "conditional_probability": (
                loaded_stand_successes / handoff_survival_successes
                if handoff_survival_successes
                else None
            ),
        },
        "walk_gate": {
            "eligible_loaded_stands": loaded_stand_successes,
            "successes": walk_successes,
            "conditional_probability": (
                walk_successes / loaded_stand_successes
                if loaded_stand_successes
                else None
            ),
        },
        "carry_given_grasp": (
            walk_successes / handoff_eligible_episodes
            if handoff_eligible_episodes
            else None
        ),
        "stop_after_grasp_snapshot": args_cli.stop_after_grasp_snapshot,
        "stop_after_handoff_gate": args_cli.stop_after_handoff_gate,
        "stop_after_carry_snapshot": args_cli.stop_after_carry_snapshot,
        "target_grasp_snapshots": args_cli.target_grasp_snapshots,
        "snapshot_count": (
            snapshot_store.count if snapshot_store is not None else 0
        ),
        "target_grasp_snapshots_reached": (
            snapshot_store is not None
            and args_cli.target_grasp_snapshots > 0
            and snapshot_store.count >= args_cli.target_grasp_snapshots
        ),
        "approach_action_alpha": args_cli.approach_action_alpha,
        "approach_search_torso_pitch_rad": args_cli.approach_search_torso_pitch_rad,
        "approach_hand_state_decoding": args_cli.approach_hand_state_decoding,
        "narrow_camera_profile": args_cli.narrow_camera_profile,
        "pre_navigate": {
            "enabled": args_cli.pre_navigate,
            "start_range": list(args_cli.pre_navigate_start_range),
            "max_steps": args_cli.pre_navigate_max_steps,
            "tolerance": list(args_cli.pre_navigate_tolerance),
            "target_jitter": list(args_cli.pre_navigate_target_jitter),
            "settle_steps": args_cli.pre_navigate_settle_steps,
        },
        "approach_task_target_passthrough": (
            args_cli.approach_task_target_passthrough
        ),
        "approach_qr_endpoint_alignment": args_cli.approach_qr_endpoint_alignment,
        "approach_endpoint_offset_steps": args_cli.approach_endpoint_offset_steps,
        "approach_endpoint_centering_m": args_cli.approach_endpoint_centering_m,
        "approach_wrist_servo_gain": args_cli.approach_wrist_servo_gain,
        "approach_wrist_servo_max_correction_m": (
            args_cli.approach_wrist_servo_max_correction_m
        ),
        "grasp_handoff_steps": args_cli.grasp_handoff_steps,
        "carry_handoff_steps": args_cli.carry_handoff_steps,
        "handoff_wrist_midpoint_limit_m": (
            args_cli.handoff_wrist_midpoint_limit_m
        ),
        "handoff_tip_distance_limit_m": args_cli.handoff_tip_distance_limit_m,
        "handoff_cage_midpoint_limit_m": (
            args_cli.handoff_cage_midpoint_limit_m
        ),
        "handoff_wrist_asymmetry_limit_m": (
            args_cli.handoff_wrist_asymmetry_limit_m
        ),
        "approach_ready_hold_steps": args_cli.approach_ready_hold_steps,
        "grasp_lower_body_lock": args_cli.grasp_lower_body_lock,
        "grasp_live_approach_lower_body": (
            args_cli.grasp_live_approach_lower_body
        ),
        "grasp_latched_approach_lower_body": (
            args_cli.grasp_latched_approach_lower_body
        ),
        "grasp_wbc_current_wrist_stance": (
            args_cli.grasp_wbc_current_wrist_stance
        ),
        "grasp_wbc_current_wrist_after_upper_hold": (
            args_cli.grasp_wbc_current_wrist_after_upper_hold
        ),
        "grasp_wbc_current_wrist_after_contact": (
            args_cli.grasp_wbc_current_wrist_after_contact
        ),
        "grasp_wbc_current_wrist_min_lift_m": (
            args_cli.grasp_wbc_current_wrist_min_lift_m
        ),
        "grasp_latch_live_legs_on_contact": (
            args_cli.grasp_latch_live_legs_on_contact
        ),
        "grasp_live_leg_max_residual_rad": (
            args_cli.grasp_live_leg_max_residual_rad
        ),
        "grasp_post_hold_stance_blend_steps": (
            args_cli.grasp_post_hold_stance_blend_steps
        ),
        "grasp_checkpoint_certified_by_gate": (
            grasp_checkpoint_certified_by_gate
        ),
        "grasp_freeze_upper_on_lift": args_cli.grasp_freeze_upper_on_lift,
        "grasp_upper_freeze_min_lift_m": (
            args_cli.grasp_upper_freeze_min_lift_m
        ),
        "grasp_freeze_hands_on_contact": (
            args_cli.grasp_freeze_hands_on_contact
        ),
        "grasp_hand_freeze_min_lift_m": (
            args_cli.grasp_hand_freeze_min_lift_m
        ),
        "grasp_refresh_hand_lock_on_upper_hold": (
            args_cli.grasp_refresh_hand_lock_on_upper_hold
        ),
        "grasp_freeze_loaded_posture_on_upper_hold": (
            args_cli.grasp_freeze_loaded_posture_on_upper_hold
        ),
        "grasp_canonical_hand_lock": args_cli.grasp_canonical_hand_lock,
        "max_grasp_retries": args_cli.max_grasp_retries,
        "grasp_retry_lost_contact_steps": (
            args_cli.grasp_retry_lost_contact_steps
        ),
        "grasp_retry_min_phase_steps": args_cli.grasp_retry_min_phase_steps,
        "total_grasp_retries": sum(
            item["grasp_retry_count"] for item in episode_results
        ),
        "allow_uncertified_grasp_probe": (
            args_cli.allow_uncertified_grasp_probe
        ),
        "action_key": (
            next(iter(set(action_keys.values())))
            if len(set(action_keys.values())) == 1
            else "mixed"
        ),
        "action_keys": {
            phase.name.lower(): key for phase, key in action_keys.items()
        },
        "carry_runtime_contract_sha256": carry_runtime_contract[
            "contract_sha256"
        ],
        "carry_navigation_ramp_steps": carry_runtime_contract[
            "navigation_ramp_steps"
        ],
        "carry_grasp_lock_postprocessor_version": carry_runtime_contract[
            "grasp_lock_postprocessor_version"
        ],
        "place_runtime_contract_sha256": place_runtime_contract[
            "contract_sha256"
        ],
        "place_postprocessor_version": place_runtime_contract[
            "postprocessor_version"
        ],
        "place_arm_offset_decay_steps": place_runtime_contract[
            "arm_offset_decay_steps"
        ],
        "place_arm_offset_decay_profile": place_runtime_contract[
            "arm_offset_decay_profile"
        ],
        "strict_hold_seconds": args_cli.strict_hold_seconds,
        "video_output_prefix": None if video_path is None else str(video_path),
        "checkpoints": {
            phase.name.lower(): str(Path(path).resolve())
            for phase, path in checkpoints.items()
        },
        "episode_results": episode_results,
    }
    output = Path(args_cli.metrics_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac/Kit can otherwise make a SystemExit during startup/reset look
        # like an ordinary clean shutdown with no actionable Python context.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
