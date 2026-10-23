#!/usr/bin/env python3
"""Collect carry or place teacher data from provenance-checked physical snapshots."""

# Isaac Sim must be launched before the runtime imports below.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher
from unitree_rl_lab.runtime_paths import default_arena_root, default_arena_dataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("carry", "place"), required=True)
parser.add_argument("--rule_based_place", action="store_true",
                    help="Metrics-only align/open/withdraw diagnostic with live balance; no BC4 required")
parser.add_argument("--snapshot_input", required=True)
parser.add_argument(
    "--bc2_gate",
    required=True,
    help="Passing SHA-bound BC-2 certification required before downstream teacher collection",
)
parser.add_argument("--output", required=True)
parser.add_argument("--staging_snapshot_output", default=None)
parser.add_argument(
    "--student_checkpoint",
    default=None,
    help=(
        "Optional 23-D phase or 43-D Carry Student checkpoint. When supplied, run a metrics-only "
        "physical evaluation from the same provenance-checked snapshots instead of "
        "publishing Teacher demonstrations."
    ),
)
parser.add_argument(
    "--student_teacher_relabel",
    action="store_true",
    help=(
        "Roll out --student_checkpoint but publish the Teacher high-level label "
        "at every Student-induced state, including failed episodes, for DAgger retraining"
    ),
)
parser.add_argument("--task", default="Unitree-G1-43dof-BoxPickPlace")
parser.add_argument("--arena_root", default=default_arena_root())
parser.add_argument("--num_trials", type=int, default=100)
parser.add_argument(
    "--repeat_snapshots",
    action="store_true",
    help=(
        "Cycle provenance-checked source snapshots until --num_trials physical "
        "rollouts have been attempted. Metrics retain the unique source count."
    ),
)
parser.add_argument(
    "--snapshot_start_offset",
    type=int,
    default=0,
    help=(
        "Start cyclic snapshot selection at this non-negative offset; used to "
        "diversify independent parallel collection shards"
    ),
)
parser.add_argument(
    "--target_successes",
    type=int,
    default=0,
    help="Stop after this many successful trajectories; zero runs all available trials",
)
parser.add_argument("--max_phase_seconds", type=float, default=16.0)
parser.add_argument(
    "--controller_handoff_steps",
    type=int,
    default=100,
    help="Zero-command lower-body blend steps used to stabilize a loaded grasp",
)
parser.add_argument(
    "--loaded_stand_steps",
    type=int,
    default=250,
    help="Additional zero-command Homie balance steps required before Carry walking",
)
parser.add_argument(
    "--staging_stabilization_steps",
    type=int,
    default=100,
    help=(
        "Consecutive zero-navigation, bilateral lifted steps required after "
        "reaching staging before publishing Carry success and its snapshot"
    ),
)
parser.add_argument(
    "--navigation_ramp_steps",
    type=int,
    default=50,
    help="Carry steps over which the camera navigation command ramps from zero",
)
parser.add_argument(
    "--carry_staging_standoff_m",
    type=float,
    default=0.30,
    help="Camera-derived robot standoff from the Bin used by Carry navigation",
)
parser.add_argument(
    "--place_arm_offset_decay_steps",
    type=int,
    default=100,
    help=(
        "Recorded PLACE steps over which the loaded arm preload offset decays "
        "to zero (inclusive smoothstep)"
    ),
)
parser.add_argument(
    "--place_lower_steps",
    type=int,
    default=60,
    help="Recorded PLACE steps used to lower a still-closed grasp above the Bin",
)
parser.add_argument("--place_raise_steps", type=int, default=25)
parser.add_argument("--place_translate_steps", type=int, default=100)
parser.add_argument(
    "--place_release_height_m",
    type=float,
    default=0.70,
    help=(
        "World-height target for bilateral release; default clears the 0.54 m Bin wall "
        "for the fixed 0.20 m Box"
    ),
)
parser.add_argument(
    "--adaptive_place",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Advance the PLACE trajectory only after bilateral contact and Box "
        "tracking gates pass; release only after a stable over-Bin gate"
    ),
)
parser.add_argument("--place_tracking_tolerance_m", type=float, default=0.06)
parser.add_argument("--place_release_xy_tolerance_m", type=float, default=0.06)
parser.add_argument("--place_release_z_tolerance_m", type=float, default=0.08)
parser.add_argument("--place_release_speed_mps", type=float, default=0.18)
parser.add_argument("--place_recovery_compression_m", type=float, default=0.012)
parser.add_argument(
    "--place_relabel_wrist_error_m",
    type=float,
    default=0.035,
    help=(
        "In Student-state PLACE relabel mode, hold the measured Box and issue a "
        "compressed recovery label when either wrist deviates this far from its "
        "captured object-relative grasp"
    ),
)
parser.add_argument("--place_contact_loss_grace_steps", type=int, default=40)
parser.add_argument(
    "--place_base_follow_max_speed_mps",
    type=float,
    default=0.0,
    help="Maximum PLACE base speed toward the Bin; zero disables base following",
)
parser.add_argument("--place_base_follow_gain", type=float, default=0.5)
parser.add_argument("--place_base_follow_stop_distance_m", type=float, default=0.10)
parser.add_argument(
    "--place_landing_overshoot_m",
    type=float,
    default=0.0,
    help=(
        "Bias the wrist target beyond the Bin centre along the incoming Box-to-Bin "
        "direction to compensate loaded arm tracking lag"
    ),
)
parser.add_argument("--box_size", type=float, default=0.20)
parser.add_argument("--box_mass", type=float, default=0.20)
parser.add_argument("--box_static_friction", type=float, default=1.5)
parser.add_argument("--box_dynamic_friction", type=float, default=1.2)
parser.add_argument("--box_restitution", type=float, default=0.0)
parser.add_argument("--randomize_work_area", action="store_true")
parser.add_argument("--work_area_size_m", type=float, default=5.0)
parser.add_argument("--keep_failures", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument(
    "--video_output",
    default=None,
    help="Optional MP4 target prefix; requires --enable_cameras",
)
parser.add_argument("--video_length_steps", type=int, default=750)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.rule_based_place and (args_cli.phase != "place" or args_cli.student_checkpoint is not None):
    parser.error("--rule_based_place requires --phase place without a Student checkpoint")

if args_cli.student_teacher_relabel and args_cli.student_checkpoint is None:
    parser.error("--student_teacher_relabel requires --student_checkpoint")
if args_cli.snapshot_start_offset and not args_cli.repeat_snapshots:
    parser.error("--snapshot_start_offset requires --repeat_snapshots")

if (
    args_cli.num_trials <= 0
    or args_cli.snapshot_start_offset < 0
    or args_cli.target_successes < 0
    or args_cli.target_successes > args_cli.num_trials
    or args_cli.max_phase_seconds <= 0.0
    or args_cli.controller_handoff_steps <= 0
    or args_cli.loaded_stand_steps <= 0
    or args_cli.staging_stabilization_steps <= 0
    or args_cli.navigation_ramp_steps <= 0
    or args_cli.carry_staging_standoff_m <= 0.0
    or args_cli.place_arm_offset_decay_steps < 2
    or args_cli.place_lower_steps <= 0
    or args_cli.place_raise_steps <= 0
    or args_cli.place_translate_steps <= 0
    or args_cli.place_release_height_m <= 0.0
    or args_cli.place_tracking_tolerance_m <= 0.0
    or args_cli.place_release_xy_tolerance_m <= 0.0
    or args_cli.place_release_z_tolerance_m <= 0.0
    or args_cli.place_release_speed_mps <= 0.0
    or args_cli.place_recovery_compression_m < 0.0
    or args_cli.place_relabel_wrist_error_m <= 0.0
    or args_cli.place_contact_loss_grace_steps <= 0
    or args_cli.place_base_follow_max_speed_mps < 0.0
    or args_cli.place_base_follow_gain <= 0.0
    or args_cli.place_base_follow_stop_distance_m <= 0.0
    or args_cli.place_landing_overshoot_m < 0.0
    or args_cli.video_length_steps <= 0
):
    parser.error(
        "trial/time/handoff/ramp/place-lower values must be positive, "
        "PLACE feedback thresholds must be positive, and "
        "--place_arm_offset_decay_steps must be >= 2"
    )
if args_cli.box_size <= 0.0 or args_cli.box_mass <= 0.0:
    parser.error("fixed Box size and mass must be positive")
if args_cli.work_area_size_m <= 1.5:
    parser.error("--work_area_size_m must exceed 1.5 m")
if (
    args_cli.phase == "carry"
    and args_cli.staging_snapshot_output is None
    and args_cli.student_checkpoint is None
):
    parser.error("carry collection requires --staging_snapshot_output")
if args_cli.student_checkpoint is not None and args_cli.keep_failures:
    parser.error("Student evaluation is metrics-only and cannot use --keep_failures")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib
import json
import uuid

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from unitree_rl_lab.utils.arena_wbc_bridge import ArenaWBCBridge
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.teacher.arena_teacher_runtime import (
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
    build_carry_runtime_contract,
    carry_grasp_lock_indices,
    carry_navigation_scale,
)
from unitree_rl_lab.utils.teacher.phase_teachers import (
    CarryTeacher,
    PlaceTeacher,
    PlaceTrajectoryCfg,
    RealizedBimanualGrasp,
    pose_matrix_wxyz,
)
from unitree_rl_lab.utils.teacher.place_runtime_contract import (
    apply_place_runtime_postprocessor,
    build_place_runtime_contract,
    place_finger_lock_active,
)
from unitree_rl_lab.utils.teacher.phase_bc import (
    file_sha256,
    load_bc_policy,
    validate_bc2_gate,
    validate_runtime_action_contract,
)
from unitree_rl_lab.utils.teacher.physical_snapshot_runtime import (
    capture_physical_snapshot,
    restore_physical_snapshot,
    restored_snapshot_gate,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import (
    PhysicalSnapshotStore,
    SnapshotKind,
    load_snapshots,
    load_snapshot_store_metadata,
)
from unitree_rl_lab.utils.teacher.skill_fsm import (
    SkillPhase,
    box_reached_ground_outside_bin,
)
from unitree_rl_lab.utils.teacher.student_observation import (
    STUDENT_FEATURE_SLICES,
    replace_velocity_command,
    schema_metadata,
)
from unitree_rl_lab.utils.teacher_demo_recorder import (
    DEFAULT_HIGH_LEVEL_ACTION_NAMES,
    TeacherDemoRecorder,
    TeacherDemoStep,
)


CONTROL_HZ = 50.0
TABLE_SURFACE_HEIGHT_M = 0.74
STRICT_LIFT_M = 0.12
STRICT_HOLD_SECONDS = 2.0
CONTACT_LOSS_GRACE_STEPS = 5
PLACE_SETTLE_STEPS = 10
PLACE_POST_RELEASE_HORIZON_STEPS = 100
PLACE_SETTLE_MAX_BOX_SPEED_MPS = 0.15
SNAPSHOT_MIN_HOLD_STEPS = int(math.ceil(STRICT_HOLD_SECONDS * CONTROL_HZ))
PLACE_PHASE_CLOCK_SCHEMA = "unitree_place_phase_clock_v1"
PLACE_OBSERVATION_ADAPTER_SCHEMA = "unitree_place_observation_adapter_v1"


def _place_student_observation(
    policy, observation: np.ndarray, step: int
) -> np.ndarray:
    """Apply the same public PLACE observation contract used by the full FSM."""

    result = np.asarray(observation, dtype=np.float32).copy()
    phase_clock = policy.metadata.get("place_phase_clock")
    if phase_clock is not None:
        expected_index = STUDENT_FEATURE_SLICES["velocity_commands"].start
        if (
            not isinstance(phase_clock, dict)
            or phase_clock.get("schema") != PLACE_PHASE_CLOCK_SCHEMA
            or int(phase_clock.get("feature_index", -1)) != expected_index
            or int(phase_clock.get("horizon_steps", 0)) <= 0
            or phase_clock.get("privileged") is not False
        ):
            raise ValueError("PLACE checkpoint has an invalid phase-clock contract")
        result[expected_index] = min(step / int(phase_clock["horizon_steps"]), 1.0)

    adapter = policy.metadata.get("place_observation_adapter")
    if adapter is not None:
        if (
            not isinstance(adapter, dict)
            or adapter.get("schema") != PLACE_OBSERVATION_ADAPTER_SCHEMA
            or adapter.get("privileged") is not False
        ):
            raise ValueError("PLACE checkpoint has an invalid observation adapter")
        masked = tuple(adapter.get("masked_feature_names", ()))
        if not set(masked).issubset({"policy_history", "last_action"}):
            raise ValueError("PLACE checkpoint masks unsupported observation features")
        for feature_name in masked:
            result[STUDENT_FEATURE_SLICES[feature_name].as_slice()] = 0.0
    return result


def _configure_environment():
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    configure_fixed_teacher_scene(
        cfg,
        box_size_m=float(args_cli.box_size),
        box_mass_kg=float(args_cli.box_mass),
        static_friction=float(args_cli.box_static_friction),
        dynamic_friction=float(args_cli.box_dynamic_friction),
        restitution=float(args_cli.box_restitution),
        keep_work_area_randomization=args_cli.randomize_work_area,
    )
    cfg.commands.base_velocity.standoff_distance = float(
        args_cli.carry_staging_standoff_m
    )
    if args_cli.randomize_work_area:
        centre_extent = 0.5 * args_cli.work_area_size_m - 0.45
        cfg.events.reset_pick_place_work_area.params["workspace_xy_range"] = {
            "x": (-centre_extent, centre_extent),
            "y": (-centre_extent, centre_extent),
        }
    # A complete physical snapshot overwrites the reset robot/Box state.  The
    # reset event remains deterministic only to keep manager buffers valid.
    cfg.events.reset_pick_place_scene.params["grasp_box_xy_range"] = {
        "x": (0.62, 0.62),
        "y": (0.0, 0.0),
    }
    cfg.episode_length_s = max(15.0, args_cli.max_phase_seconds + 2.0)
    if args_cli.video_output is not None:
        cfg.viewer.eye = (2.9, 2.5, 2.5)
        cfg.viewer.lookat = (0.70, -0.35, 0.75)
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
    return cfg


class _RecordingEnvProxy:
    """Route reset/step/close through RecordVideo while exposing manager APIs."""

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


def _pose7(asset, env_id: int = 0) -> torch.Tensor:
    return asset.data.root_link_state_w[env_id, :7]


def _fingertip_positions(robot, ids: list[int]) -> torch.Tensor:
    return robot.data.body_link_pos_w[0, ids]


def _tilt_rad(robot) -> float:
    gravity = robot.data.projected_gravity_b[0]
    return float(torch.acos(torch.clamp(-gravity[2], -1.0, 1.0)))


def _inside_bin(
    box_position_w: torch.Tensor, env_origin: torch.Tensor, center_e, half_extents
) -> bool:
    center = env_origin + torch.as_tensor(
        center_e, dtype=box_position_w.dtype, device=box_position_w.device
    )
    half = torch.tensor(
        half_extents, dtype=box_position_w.dtype, device=box_position_w.device
    )
    return bool(torch.all(torch.abs(box_position_w - center) < half))


def _synchronize_task_phase(env, phase: int) -> None:
    """Let this collector, rather than the RL reward FSM, own phase routing."""

    env.task_phase[0] = int(phase)
    # ``env.step`` already advances command time once.  A zero-dt refresh is
    # enough to make camera navigation and its staging predicate agree with
    # the explicitly selected phase without shortening command timers twice.
    env.command_manager.compute(0.0)


def _preserve_realized_grasp_targets(
    env,
    raw_action: torch.Tensor,
    reference_raw_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Homie for locomotion while retaining the loaded arm/hand targets.

    Pink is redundant: even when commanded to the measured wrist poses, a
    freshly initialized solve may choose arm joint targets that differ from a
    contact-loaded BC-2 grasp.  During carry the Box and wrists should move
    with the torso, so the physically realized shoulder/elbow/wrist/finger
    targets are the safest exact null-space choice.  Lower-body and waist
    joints remain controlled by Arena's locomotion policy.
    """

    term = env.action_manager.get_term("JointPositionAction")
    names = tuple(term._joint_names)
    locked = carry_grasp_lock_indices(names)
    result = raw_action.clone()
    result[:, locked] = reference_raw_action[:, locked]
    absolute = result * term._scale + term._offset
    return result, absolute


def _carry_controller_handoff(
    env,
    bridge,
    teacher: CarryTeacher,
    reference_raw_action: torch.Tensor,
    *,
    robot,
    box,
    pelvis: int,
    box_table_height: float,
    record_buffer: list[TeacherDemoStep],
    left_wrist: int,
    right_wrist: int,
    left_tips: list[int],
    right_tips: list[int],
    bin_pose: torch.Tensor,
) -> tuple[bool, str, int, int, float]:
    """Synchronize Arena WBC without exposing the loaded grasp to a command jump."""

    left_count = right_count = 0
    lift_height = float("-inf")
    for handoff_step in range(args_cli.controller_handoff_steps):
        _synchronize_task_phase(env, 1)
        observation = env.observation_manager.compute(update_history=False)
        pre_observation = torch.as_tensor(
            replace_velocity_command(
                student_observation(observation)[0].detach().cpu().numpy(),
                np.zeros(3, dtype=np.float32),
            ),
            dtype=torch.float32,
            device=env.device,
        )
        high_level = teacher.action(
            current_box_pose_wxyz=_pose7(box).detach().cpu().numpy(),
            current_pelvis_pose_wxyz=body_pose_wxyz(robot, pelvis)
            .detach()
            .cpu()
            .numpy(),
            camera_navigation_velocity=np.zeros(3, dtype=np.float32),
        )
        names, values = bridge.step(bridge_request(env, high_level))
        proposed_raw, _ = action_order_target(env, names, values)
        fraction = (handoff_step + 1) / args_cli.controller_handoff_steps
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        blended = reference_raw_action + fraction * (
            proposed_raw - reference_raw_action
        )
        applied_raw, absolute_target = _preserve_realized_grasp_targets(
            env, blended, reference_raw_action
        )
        env.step(applied_raw)
        _synchronize_task_phase(env, 1)
        left_count, right_count = box_contact_counts(env)
        lift_height = float(box.data.root_link_pos_w[0, 2]) - box_table_height
        bad_orientation = _tilt_rad(robot) > 0.8
        dropped = lift_height < 0.08
        record_buffer.append(
            TeacherDemoStep(
                student_observation=pre_observation,
                teacher_high_level_action=high_level,
                processed_43d_joint_target=absolute_target[0],
                phase_id=1,
                box_pose=_pose7(box).detach().clone(),
                bin_pose=bin_pose.detach().clone(),
                root_pose=_pose7(robot).detach().clone(),
                left_wrist_pose=body_pose_wxyz(robot, left_wrist),
                right_wrist_pose=body_pose_wxyz(robot, right_wrist),
                left_fingertip_positions=_fingertip_positions(robot, left_tips)
                .detach()
                .clone(),
                right_fingertip_positions=_fingertip_positions(robot, right_tips)
                .detach()
                .clone(),
                box_size=np.full(3, args_cli.box_size, dtype=np.float32),
                left_contact_count=left_count,
                right_contact_count=right_count,
                lift=lift_height >= STRICT_LIFT_M,
                drop=dropped,
                success=False,
            )
        )
        if bad_orientation:
            return (
                False,
                "handoff_bad_orientation",
                left_count,
                right_count,
                lift_height,
            )
        if dropped:
            return False, "handoff_box_drop", left_count, right_count, lift_height
        if left_count < 1 or right_count < 1:
            return (
                False,
                "handoff_bilateral_hold_lost",
                left_count,
                right_count,
                lift_height,
            )
    return True, "handoff_success", left_count, right_count, lift_height


def _carry_loaded_stand(
    env,
    bridge,
    teacher: CarryTeacher,
    reference_raw_action: torch.Tensor,
    *,
    robot,
    box,
    pelvis: int,
    box_table_height: float,
) -> tuple[bool, str, int, int, float]:
    """Verify five seconds of loaded balance before any navigation command."""

    left_count = right_count = 0
    lift_height = float("-inf")
    for _ in range(args_cli.loaded_stand_steps):
        _synchronize_task_phase(env, 1)
        high_level = teacher.action(
            current_box_pose_wxyz=_pose7(box).detach().cpu().numpy(),
            current_pelvis_pose_wxyz=body_pose_wxyz(robot, pelvis)
            .detach()
            .cpu()
            .numpy(),
            camera_navigation_velocity=np.zeros(3, dtype=np.float32),
        )
        names, values = bridge.step(bridge_request(env, high_level))
        proposed_raw, _ = action_order_target(env, names, values)
        applied_raw, _ = _preserve_realized_grasp_targets(
            env, proposed_raw, reference_raw_action
        )
        env.step(applied_raw)
        _synchronize_task_phase(env, 1)
        left_count, right_count = box_contact_counts(env)
        lift_height = float(box.data.root_link_pos_w[0, 2]) - box_table_height
        if _tilt_rad(robot) > 0.8:
            return (
                False,
                "loaded_stand_bad_orientation",
                left_count,
                right_count,
                lift_height,
            )
        if lift_height < 0.08:
            return False, "loaded_stand_box_drop", left_count, right_count, lift_height
        if left_count < 1 or right_count < 1:
            return (
                False,
                "loaded_stand_bilateral_hold_lost",
                left_count,
                right_count,
                lift_height,
            )
    return True, "loaded_stand_success", left_count, right_count, lift_height


def _warm_place_controller_without_state_transition(
    env,
    bridge,
    teacher: CarryTeacher,
    reference_raw_action: torch.Tensor,
    *,
    robot,
    box,
    pelvis: int,
    box_table_height: float,
) -> tuple[bool, str, int, int, float]:
    """Warm Pink/Homie while retaining the stable loaded staging target.

    PLACE starts from a completed Carry snapshot.  Blending its standing legs
    toward a freshly initialized Homie proposal caused falls before PLACE had
    issued a single motion.  Pink still receives every measured pose here so
    its arm baseline becomes valid, while physics retains the captured loaded
    leg/arm/hand target.  The v5 PLACE runtime contract keeps only the legs
    locked after this warm-up and lets Pink move the arms.
    """

    left_count = right_count = 0
    lift_height = float("-inf")
    for _ in range(args_cli.controller_handoff_steps):
        _synchronize_task_phase(env, 1)
        high_level = teacher.action(
            current_box_pose_wxyz=_pose7(box).detach().cpu().numpy(),
            current_pelvis_pose_wxyz=body_pose_wxyz(robot, pelvis)
            .detach()
            .cpu()
            .numpy(),
            camera_navigation_velocity=np.zeros(3, dtype=np.float32),
        )
        bridge.step(bridge_request(env, high_level))
        env.step(reference_raw_action)
        _synchronize_task_phase(env, 1)
        left_count, right_count = box_contact_counts(env)
        lift_height = float(box.data.root_link_pos_w[0, 2]) - box_table_height
        if _tilt_rad(robot) > 0.8:
            return (
                False,
                "handoff_bad_orientation",
                left_count,
                right_count,
                lift_height,
            )
        if lift_height < 0.08:
            return False, "handoff_box_drop", left_count, right_count, lift_height
        if left_count < 1 or right_count < 1:
            return (
                False,
                "handoff_bilateral_hold_lost",
                left_count,
                right_count,
                lift_height,
            )
    return True, "handoff_success", left_count, right_count, lift_height


def _apply_place_loaded_arm_delta_and_finger_lock(
    env,
    raw_action: torch.Tensor,
    reference_raw_action: torch.Tensor,
    active_pink_baseline_raw_action: torch.Tensor,
    hand_state: np.ndarray,
    place_step: int,
    place_runtime_contract,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply decayed loaded arm preload and retain fingers until release."""

    hand_state = np.asarray(hand_state, dtype=np.float32)
    if hand_state.shape != (2,) or not np.isfinite(hand_state).all():
        raise ValueError("PLACE hand_state must be a finite two-vector")
    term = env.action_manager.get_term("JointPositionAction")
    result = apply_place_runtime_postprocessor(
        raw_action,
        reference_raw_action,
        active_pink_baseline_raw_action,
        hand_state,
        place_step,
        place_runtime_contract,
    )
    absolute = result * term._scale + term._offset
    return result, absolute


def _run_trial(
    env,
    bridge,
    snapshot,
    recorder,
    *,
    task_cfg,
    snapshot_store,
    carry_runtime_contract,
    place_runtime_contract,
    student_policy,
    trial_index: int,
) -> dict:
    observation, _ = env.reset()
    bridge.reset()
    restore_physical_snapshot(
        env,
        snapshot,
        expected_box_size=np.full(3, args_cli.box_size),
        expected_box_mass_kg=args_cli.box_mass,
    )
    # A staging snapshot is normally captured after the task manager has
    # advanced to PLACE.  Temporarily evaluate it in CARRY so the camera
    # command recomputes the geometric ``at_staging`` predicate; PLACE itself
    # deliberately clears that flag.
    _synchronize_task_phase(env, 1)
    robot = env.scene["robot"]
    box = env.scene["box"]
    pelvis = body_id(robot, "pelvis")
    left_wrist = body_id(robot, "left_wrist_yaw_link")
    right_wrist = body_id(robot, "right_wrist_yaw_link")
    left_tips = [body_id(robot, name) for name in LEFT_FINGERTIP_NAMES]
    right_tips = [body_id(robot, name) for name in RIGHT_FINGERTIP_NAMES]
    box_table_height = float(snapshot.box_root_link_state_w[2]) - float(
        snapshot.lift_height_m
    )
    bin_center_e = (
        env.bin_center_e[0]
        if hasattr(env, "bin_center_e")
        else torch.tensor(task_cfg.BIN_CENTER_E, device=env.device)
    )
    bin_pose = torch.cat(
        [
            bin_center_e.to(dtype=torch.float32),
            torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device),
        ]
    )
    handoff_records: list[TeacherDemoStep] = []

    # One ordinary controlled step proves that the complete saved state remains
    # physically valid after restore.  This is never replaced with a Box pose
    # assignment independent of the robot state.
    settle_action = torch.as_tensor(snapshot.last_action, device=env.device).unsqueeze(
        0
    )
    observation, _, _, _, _ = env.step(settle_action)
    _synchronize_task_phase(env, 1)
    left_count, right_count = box_contact_counts(env)
    lift_height = float(box.data.root_link_pos_w[0, 2]) - box_table_height
    bad_orientation = _tilt_rad(robot) > 0.8
    dropped = float(box.data.root_link_pos_w[0, 2]) < box_table_height - 0.10
    navigation = env.command_manager.get_term("base_velocity")
    if not restored_snapshot_gate(
        snapshot,
        left_contact_count=left_count,
        right_contact_count=right_count,
        lift_height_m=lift_height,
        bad_orientation=bad_orientation,
        box_dropped=dropped,
        # CARRY_AT_STAGING snapshots are published only after a latched
        # camera-derived arrival followed by the configured physical standing
        # gate.  The Bin QR may leave the frustum during that stationary hold,
        # so revalidation relies on the provenance-bound latch plus the live
        # contact/lift/orientation predicates above.
        at_staging=(bool(navigation.at_staging[0]) or bool(snapshot.at_staging)),
    ):
        return {
            "success": False,
            "recorded": False,
            "reason": "restored_snapshot_failed_physical_gate",
            "steps": 1,
            "source_episode": snapshot.source_episode,
        }

    grasp = RealizedBimanualGrasp.capture(
        box_pose_wxyz=_pose7(box).detach().cpu().numpy(),
        left_wrist_pose_wxyz=body_pose_wxyz(robot, left_wrist).detach().cpu().numpy(),
        right_wrist_pose_wxyz=body_pose_wxyz(robot, right_wrist).detach().cpu().numpy(),
    )
    reference_raw_action = torch.as_tensor(
        snapshot.last_action, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    if args_cli.rule_based_place:
        from unitree_rl_lab.utils.teacher.rule_based_place import run_rule_based_place
        return run_rule_based_place(
            env, bridge,
            bin_center=bin_pose[:3].detach().cpu().numpy(),
            bin_half_extents=task_cfg.BIN_INNER_HALF_EXTENTS,
            box_size=np.full(3, args_cli.box_size),
            max_seconds=args_cli.max_phase_seconds,
            source_episode=snapshot.source_episode,
        )
    place_source_pink_baseline_raw_action = None
    place_active_pink_baseline_raw_action = None
    if args_cli.phase == "carry":
        _synchronize_task_phase(env, 1)
        teacher = CarryTeacher(
            grasp,
            minimum_box_height_w=float(box.data.root_link_pos_w[0, 2]),
        )
        handoff_ok, handoff_reason, left_count, right_count, lift_height = (
            _carry_controller_handoff(
                env,
                bridge,
                teacher,
                reference_raw_action,
                robot=robot,
                box=box,
                pelvis=pelvis,
                box_table_height=box_table_height,
                record_buffer=handoff_records,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                left_tips=left_tips,
                right_tips=right_tips,
                bin_pose=bin_pose,
            )
        )
        if not handoff_ok:
            return {
                "success": False,
                "recorded": False,
                "reason": handoff_reason,
                "steps": 1 + args_cli.controller_handoff_steps,
                "source_episode": snapshot.source_episode,
                "left_contact_count": left_count,
                "right_contact_count": right_count,
                "lift_height_m": lift_height,
            }
        loaded_stand_ok, loaded_stand_reason, left_count, right_count, lift_height = (
            _carry_loaded_stand(
                env,
                bridge,
                teacher,
                reference_raw_action,
                robot=robot,
                box=box,
                pelvis=pelvis,
                box_table_height=box_table_height,
            )
        )
        if not loaded_stand_ok:
            return {
                "success": False,
                "recorded": False,
                "reason": loaded_stand_reason,
                "steps": 1
                + args_cli.controller_handoff_steps
                + args_cli.loaded_stand_steps,
                "source_episode": snapshot.source_episode,
                "left_contact_count": left_count,
                "right_contact_count": right_count,
                "lift_height_m": lift_height,
            }
        max_steps = int(math.ceil(args_cli.max_phase_seconds / env.step_dt))
        recorded_phase = 1
    else:
        # The staging snapshot contains valid robot/Box physics and the
        # student observation history, but the out-of-process Arena WBC has
        # just been reset.  Warm its Pink/Homie state while preserving the
        # realized grasp before asking Pink to move either wrist toward the
        # Bin.  These zero-navigation steps are deliberately not recorded as
        # PLACE supervision.
        hold_teacher = CarryTeacher(
            grasp,
            minimum_box_height_w=float(box.data.root_link_pos_w[0, 2]),
        )
        handoff_ok, handoff_reason, left_count, right_count, lift_height = (
            _warm_place_controller_without_state_transition(
                env,
                bridge,
                hold_teacher,
                reference_raw_action,
                robot=robot,
                box=box,
                pelvis=pelvis,
                box_table_height=box_table_height,
            )
        )
        if not handoff_ok:
            return {
                "success": False,
                "recorded": False,
                "reason": f"place_{handoff_reason}",
                "steps": 1 + args_cli.controller_handoff_steps,
                "source_episode": snapshot.source_episode,
                "left_contact_count": left_count,
                "right_contact_count": right_count,
                "lift_height_m": lift_height,
            }
        # Keep the original CARRY transition baseline for provenance only.
        # Delta arithmetic must use a baseline from this warmed controller;
        # mixing a proposal from this process with the source process caused
        # a 4-rad offset and immediate loss of bilateral contact.
        place_source_pink_baseline_raw_action = torch.as_tensor(
            snapshot.unprocessed_wbc_raw_action,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        _synchronize_task_phase(env, 1)
        baseline_high_level = hold_teacher.action(
            current_box_pose_wxyz=_pose7(box).detach().cpu().numpy(),
            current_pelvis_pose_wxyz=body_pose_wxyz(robot, pelvis)
            .detach()
            .cpu()
            .numpy(),
            camera_navigation_velocity=np.zeros(3, dtype=np.float32),
        )
        baseline_names, baseline_values = bridge.step(
            bridge_request(env, baseline_high_level)
        )
        place_active_pink_baseline_raw_action, _ = action_order_target(
            env, baseline_names, baseline_values
        )
        _synchronize_task_phase(env, 2)
        bin_center_e = (
            env.bin_center_e[0].detach().cpu().numpy()
            if hasattr(env, "bin_center_e")
            else np.asarray(task_cfg.BIN_CENTER_E)
        )
        bin_center_w = env.scene.env_origins[0].detach().cpu().numpy() + bin_center_e
        start_box_position_w = box.data.root_link_pos_w[0].detach().cpu().numpy()
        landing_target_w = bin_center_w.copy()
        incoming_xy = bin_center_w[:2] - start_box_position_w[:2]
        incoming_distance = float(np.linalg.norm(incoming_xy))
        if incoming_distance > 1.0e-9:
            landing_target_w[:2] += (
                args_cli.place_landing_overshoot_m * incoming_xy / incoming_distance
            )
        teacher = PlaceTeacher(
            grasp,
            start_box_position_w=start_box_position_w,
            bin_center_w=landing_target_w,
            cfg=PlaceTrajectoryCfg(
                raise_steps=args_cli.place_raise_steps,
                translate_steps=args_cli.place_translate_steps,
                lower_steps=args_cli.place_lower_steps,
                release_height_m=args_cli.place_release_height_m,
            ),
        )
        phase_horizon_steps = int(math.ceil(args_cli.max_phase_seconds / env.step_dt))
        max_steps = (
            phase_horizon_steps
            if args_cli.adaptive_place
            else min(
                phase_horizon_steps,
                teacher.cfg.total_steps + PLACE_POST_RELEASE_HORIZON_STEPS,
            )
        )
        recorded_phase = 2

    recorder.start_episode(
        metadata={
            "trial_index": trial_index,
            "phase": args_cli.phase,
            "physical_snapshot_source_episode": snapshot.source_episode,
            "physical_snapshot_source_step": snapshot.source_step,
            "physical_snapshot_provenance": snapshot.provenance,
            "carry_grasp_lock_reference_raw_action": (
                snapshot.last_action.tolist() if args_cli.phase == "carry" else None
            ),
            "place_loaded_raw_reference_action": (
                snapshot.last_action.tolist() if args_cli.phase == "place" else None
            ),
            "place_source_pink_baseline_raw_action": (
                place_source_pink_baseline_raw_action[0].detach().cpu().tolist()
                if place_source_pink_baseline_raw_action is not None
                else None
            ),
            "place_active_pink_baseline_raw_action": (
                place_active_pink_baseline_raw_action[0].detach().cpu().tolist()
                if place_active_pink_baseline_raw_action is not None
                else None
            ),
            "place_source_vs_active_baseline_max_abs_raw": (
                float(
                    torch.max(
                        torch.abs(
                            place_source_pink_baseline_raw_action
                            - place_active_pink_baseline_raw_action
                        )
                    )
                )
                if place_source_pink_baseline_raw_action is not None
                and place_active_pink_baseline_raw_action is not None
                else None
            ),
            "high_level_runtime_postprocessor": (
                carry_runtime_contract["grasp_lock_postprocessor_version"]
                if args_cli.phase == "carry"
                else place_runtime_contract["postprocessor_version"]
            ),
            "carry_runtime_contract_sha256": (
                carry_runtime_contract["contract_sha256"]
                if args_cli.phase == "carry"
                else None
            ),
            "place_runtime_contract_sha256": (
                place_runtime_contract["contract_sha256"]
                if args_cli.phase == "place"
                else None
            ),
            "place_trajectory": (
                {
                    "raise_steps": teacher.cfg.raise_steps,
                    "translate_steps": teacher.cfg.translate_steps,
                    "lower_steps": teacher.cfg.lower_steps,
                    "release_steps": teacher.cfg.release_steps,
                    "clearance_m": teacher.cfg.clearance_m,
                    "release_height_m": teacher.cfg.release_height_m,
                }
                if args_cli.phase == "place"
                else None
            ),
        }
    )
    for handoff_record in handoff_records:
        recorder.append(handoff_record)
    hold_steps = (
        int(snapshot.stable_hold_steps)
        + args_cli.controller_handoff_steps
        + (args_cli.loaded_stand_steps if args_cli.phase == "carry" else 0)
    )
    contact_loss_steps = 0
    place_settle = 0
    staging_stable_steps = 0
    staging_reached = False
    place_progress_step = 0
    place_paused_steps = 0
    place_recovery_steps = 0
    place_release_commanded = False
    relabel_recovery_steps = 0
    relabel_left_recovery_steps = 0
    relabel_right_recovery_steps = 0
    relabel_max_left_wrist_error_m = 0.0
    relabel_max_right_wrist_error_m = 0.0
    success = False
    reason = "phase_horizon"
    staging_snapshot = None
    for step in range(max_steps):
        requested_task_phase = 1 if args_cli.phase == "carry" else 2
        _synchronize_task_phase(env, requested_task_phase)
        observation = env.observation_manager.compute(update_history=False)
        pre_observation = student_observation(observation)[0]
        pelvis_pose = body_pose_wxyz(robot, pelvis).detach().cpu().numpy()
        if args_cli.phase == "carry":
            navigation_scale = carry_navigation_scale(step, carry_runtime_contract)
            scaled_navigation = (
                navigation.command[0].detach().cpu().numpy() * navigation_scale
            )
            if staging_reached:
                scaled_navigation = np.zeros(3, dtype=np.float32)
            pre_observation = torch.as_tensor(
                replace_velocity_command(
                    pre_observation.detach().cpu().numpy(), scaled_navigation
                ),
                dtype=pre_observation.dtype,
                device=pre_observation.device,
            )
            high_level = teacher.action(
                current_box_pose_wxyz=_pose7(box).detach().cpu().numpy(),
                current_pelvis_pose_wxyz=pelvis_pose,
                camera_navigation_velocity=scaled_navigation,
            )
        else:
            current_box_position = box.data.root_link_pos_w[0].detach().cpu().numpy()
            place_navigation = np.zeros(3, dtype=np.float32)
            if args_cli.place_base_follow_max_speed_mps > 0.0:
                world_error = teacher.bin_center[:2] - current_box_position[:2]
                if (
                    np.linalg.norm(world_error)
                    > args_cli.place_base_follow_stop_distance_m
                ):
                    pelvis_matrix = pose_matrix_wxyz(pelvis_pose)
                    base_error = pelvis_matrix[:2, :2].T @ world_error
                    velocity = args_cli.place_base_follow_gain * base_error
                    speed = float(np.linalg.norm(velocity))
                    if speed > args_cli.place_base_follow_max_speed_mps:
                        velocity *= args_cli.place_base_follow_max_speed_mps / speed
                    place_navigation[:2] = velocity.astype(np.float32)
            if args_cli.adaptive_place:
                current_box_speed = float(
                    torch.linalg.vector_norm(box.data.root_link_lin_vel_w[0])
                )
                pre_left_count, pre_right_count = box_contact_counts(env)
                pre_bilateral = pre_left_count > 0 and pre_right_count > 0
                closed_horizon = (
                    teacher.cfg.raise_steps
                    + teacher.cfg.translate_steps
                    + teacher.cfg.lower_steps
                )
                desired_position, _ = teacher.desired_box_position(place_progress_step)
                tracking_error = float(
                    np.linalg.norm(current_box_position - desired_position)
                )
                if not place_release_commanded:
                    at_release_pose = (
                        np.linalg.norm(
                            current_box_position[:2] - teacher.bin_center[:2]
                        )
                        <= args_cli.place_release_xy_tolerance_m
                        and abs(float(current_box_position[2] - teacher.release[2]))
                        <= args_cli.place_release_z_tolerance_m
                        and current_box_speed <= args_cli.place_release_speed_mps
                    )
                    if (
                        place_progress_step >= closed_horizon - 1
                        and pre_bilateral
                        and at_release_pose
                    ):
                        place_release_commanded = True
                        place_progress_step = closed_horizon
                    elif (
                        pre_bilateral
                        and tracking_error <= args_cli.place_tracking_tolerance_m
                        and current_box_speed <= args_cli.place_release_speed_mps * 2.0
                    ):
                        place_progress_step = min(
                            place_progress_step + 1, closed_horizon - 1
                        )
                    else:
                        place_paused_steps += 1

                if not pre_bilateral and not place_release_commanded:
                    place_recovery_steps += 1
                    high_level = teacher.action_for_box_position(
                        box_position=current_box_position,
                        release=False,
                        current_pelvis_pose_wxyz=pelvis_pose,
                        left_recovery_compression_m=(
                            args_cli.place_recovery_compression_m
                            if pre_left_count == 0
                            else 0.0
                        ),
                        right_recovery_compression_m=(
                            args_cli.place_recovery_compression_m
                            if pre_right_count == 0
                            else 0.0
                        ),
                        camera_navigation_velocity=place_navigation,
                    )
                else:
                    high_level = teacher.action(
                        step=place_progress_step,
                        current_pelvis_pose_wxyz=pelvis_pose,
                        camera_navigation_velocity=place_navigation,
                    )
            else:
                high_level = teacher.action(
                    step=step,
                    current_pelvis_pose_wxyz=pelvis_pose,
                    camera_navigation_velocity=place_navigation,
                )
        if args_cli.student_teacher_relabel and args_cli.phase == "place":
            pre_left_count, pre_right_count = box_contact_counts(env)
            current_box_matrix = pose_matrix_wxyz(_pose7(box).detach().cpu().numpy())
            box_inverse = np.linalg.inv(current_box_matrix)
            actual_left_in_box = box_inverse @ pose_matrix_wxyz(
                body_pose_wxyz(robot, left_wrist).detach().cpu().numpy()
            )
            actual_right_in_box = box_inverse @ pose_matrix_wxyz(
                body_pose_wxyz(robot, right_wrist).detach().cpu().numpy()
            )
            left_wrist_error_m = float(
                np.linalg.norm(
                    actual_left_in_box[:3, 3] - grasp.box_to_left_wrist[:3, 3]
                )
            )
            right_wrist_error_m = float(
                np.linalg.norm(
                    actual_right_in_box[:3, 3] - grasp.box_to_right_wrist[:3, 3]
                )
            )
            recover_left = (
                pre_left_count == 0
                or left_wrist_error_m >= args_cli.place_relabel_wrist_error_m
            )
            recover_right = (
                pre_right_count == 0
                or right_wrist_error_m >= args_cli.place_relabel_wrist_error_m
            )
            closed_horizon = (
                teacher.cfg.raise_steps
                + teacher.cfg.translate_steps
                + teacher.cfg.lower_steps
            )
            if step < closed_horizon and (recover_left or recover_right):
                relabel_max_left_wrist_error_m = max(
                    relabel_max_left_wrist_error_m, left_wrist_error_m
                )
                relabel_max_right_wrist_error_m = max(
                    relabel_max_right_wrist_error_m, right_wrist_error_m
                )
                recovery_box_position = current_box_position.copy()
                recovery_box_position[2] = max(
                    recovery_box_position[2], teacher.start[2]
                )
                high_level = teacher.action_for_box_position(
                    box_position=recovery_box_position,
                    release=False,
                    current_pelvis_pose_wxyz=pelvis_pose,
                    left_recovery_compression_m=(
                        args_cli.place_recovery_compression_m if recover_left else 0.0
                    ),
                    right_recovery_compression_m=(
                        args_cli.place_recovery_compression_m if recover_right else 0.0
                    ),
                    camera_navigation_velocity=place_navigation,
                )
                relabel_recovery_steps += 1
                relabel_left_recovery_steps += int(recover_left)
                relabel_right_recovery_steps += int(recover_right)
        teacher_high_level = np.asarray(high_level, dtype=np.float32).copy()
        if student_policy is not None:
            policy_observation = pre_observation.detach().cpu().numpy()
            if args_cli.phase == "place":
                policy_observation = _place_student_observation(
                    student_policy, policy_observation, step
                )
            student_prediction = student_policy(policy_observation)[0]
            if student_policy.action_key == "teacher_high_level_action":
                high_level = student_prediction
            if args_cli.phase == "place" and not place_finger_lock_active(
                high_level[:2], place_runtime_contract
            ):
                place_release_commanded = True
        place_grasp_locked = args_cli.phase == "place" and place_finger_lock_active(
            high_level[:2], place_runtime_contract
        )
        names, values = bridge.step(bridge_request(env, high_level))
        raw_action, absolute_target = action_order_target(env, names, values)
        unprocessed_wbc_raw_action = raw_action.detach().clone()
        if (
            student_policy is not None
            and student_policy.action_key == "processed_43d_joint_target"
        ):
            term = env.action_manager.get_term("JointPositionAction")
            absolute_target = torch.as_tensor(
                student_prediction, dtype=torch.float32, device=env.device
            ).unsqueeze(0)
            raw_action = (absolute_target - term._offset) / term._scale
        if args_cli.phase == "carry":
            raw_action, absolute_target = _preserve_realized_grasp_targets(
                env, raw_action, reference_raw_action
            )
        else:
            raw_action, absolute_target = _apply_place_loaded_arm_delta_and_finger_lock(
                env,
                raw_action,
                reference_raw_action,
                place_active_pink_baseline_raw_action,
                high_level[:2],
                step,
                place_runtime_contract,
            )
        observation, _, _, _, _ = env.step(raw_action)
        # The task's sparse-reward term also owns a training-time FSM.  It may
        # have advanced CARRY->PLACE during reward computation, which clears
        # ``navigation.at_staging`` before ``env.step`` returns.  Restore this
        # collector's phase and recompute the camera predicate from the actual
        # post-physics state before evaluating success or recording a label.
        _synchronize_task_phase(env, requested_task_phase)
        observation = env.observation_manager.compute(update_history=False)

        left_count, right_count = box_contact_counts(env)
        bilateral = left_count > 0 and right_count > 0
        lift_height = float(box.data.root_link_pos_w[0, 2]) - box_table_height
        lifted = lift_height >= STRICT_LIFT_M
        hold_steps = hold_steps + 1 if bilateral and lifted else 0
        contact_loss_steps = 0 if bilateral else contact_loss_steps + 1
        bad_orientation = _tilt_rad(robot) > 0.8
        in_bin = _inside_bin(
            box.data.root_link_pos_w[0],
            env.scene.env_origins[0],
            (
                env.bin_center_e[0]
                if hasattr(env, "bin_center_e")
                else task_cfg.BIN_CENTER_E
            ),
            task_cfg.BIN_INNER_HALF_EXTENTS,
        )
        released = left_count == 0 and right_count == 0
        dropped_to_ground = box_reached_ground_outside_bin(
            box_height_m=float(box.data.root_link_pos_w[0, 2]),
            environment_origin_z_m=float(env.scene.env_origins[0, 2]),
            dropped_object_height_m=task_cfg.DROPPED_OBJECT_HEIGHT,
            box_in_bin=in_bin,
        )
        if args_cli.phase == "carry":
            dropped = dropped_to_ground or lift_height < 0.08
        else:
            # A valid PLACE release necessarily loses bilateral contact while
            # the Box is still falling through the Bin aperture.  Only a Box
            # that actually reaches the ground outside the delivered volume
            # is a fatal PLACE drop.
            dropped = dropped_to_ground
        at_staging = bool(navigation.at_staging[0])
        if args_cli.phase == "carry":
            staging_reached = staging_reached or at_staging
            staging_stable_steps = (
                staging_stable_steps + 1
                if staging_reached and bilateral and lifted
                else 0
            )
            task_success = (
                staging_reached
                and bilateral
                and lifted
                and hold_steps >= SNAPSHOT_MIN_HOLD_STEPS
                and staging_stable_steps >= args_cli.staging_stabilization_steps
            )
        else:
            box_speed = float(torch.linalg.vector_norm(box.data.root_link_lin_vel_w[0]))
            settled = (
                in_bin
                and released
                and (
                    place_release_commanded
                    if args_cli.adaptive_place
                    else bool(high_level[0] <= 0.5 and high_level[1] <= 0.5)
                )
                and box_speed <= PLACE_SETTLE_MAX_BOX_SPEED_MPS
            )
            place_settle = place_settle + 1 if settled else 0
            task_success = place_settle >= PLACE_SETTLE_STEPS

        # Safety predicates dominate a task predicate on the same physics
        # frame.  A success snapshot/demo must never be published from a fall
        # or from a Box-drop frame.
        success = task_success and not bad_orientation and not dropped

        recorder.append(
            TeacherDemoStep(
                student_observation=pre_observation,
                teacher_high_level_action=(
                    teacher_high_level
                    if args_cli.student_teacher_relabel
                    else high_level
                ),
                processed_43d_joint_target=absolute_target[0],
                phase_id=recorded_phase,
                box_pose=_pose7(box),
                bin_pose=bin_pose,
                root_pose=_pose7(robot),
                left_wrist_pose=body_pose_wxyz(robot, left_wrist),
                right_wrist_pose=body_pose_wxyz(robot, right_wrist),
                left_fingertip_positions=_fingertip_positions(robot, left_tips),
                right_fingertip_positions=_fingertip_positions(robot, right_tips),
                box_size=np.full(3, args_cli.box_size, dtype=np.float32),
                left_contact_count=left_count,
                right_contact_count=right_count,
                lift=lifted,
                drop=dropped,
                success=success,
            )
        )
        if success and args_cli.phase == "carry" and snapshot_store is not None:
            staging_snapshot = capture_physical_snapshot(
                env,
                kind=SnapshotKind.CARRY_AT_STAGING,
                env_id=0,
                source_episode=recorder.active_episode or f"carry_{trial_index}",
                source_step=step,
                student_observation=student_observation(observation)[0],
                box_initial_height_m=box_table_height,
                left_contact_count=left_count,
                right_contact_count=right_count,
                stable_hold_steps=hold_steps,
                at_staging=True,
                bad_orientation=bad_orientation,
                box_dropped=dropped,
                box_size=np.full(3, args_cli.box_size),
                box_mass_kg=args_cli.box_mass,
                unprocessed_wbc_raw_action=unprocessed_wbc_raw_action[0],
            )
        if bad_orientation:
            reason = "bad_orientation"
            break
        if dropped:
            reason = "box_drop"
            break
        if success:
            reason = f"{args_cli.phase}_success"
            break
        if (args_cli.phase == "carry" or place_grasp_locked) and contact_loss_steps >= (
            args_cli.place_contact_loss_grace_steps
            if args_cli.phase == "place"
            and (args_cli.adaptive_place or args_cli.student_teacher_relabel)
            else CONTACT_LOSS_GRACE_STEPS
        ):
            reason = "bilateral_hold_lost"
            break

    final_metadata = {
        "source_episode": snapshot.source_episode,
        "source_step": snapshot.source_step,
        "success": success,
        "termination_reason": reason,
        "adaptive_place": args_cli.adaptive_place
        if args_cli.phase == "place"
        else None,
        "place_progress_step": place_progress_step
        if args_cli.phase == "place"
        else None,
        "place_paused_steps": place_paused_steps if args_cli.phase == "place" else None,
        "place_recovery_steps": place_recovery_steps
        if args_cli.phase == "place"
        else None,
        "place_release_commanded": place_release_commanded
        if args_cli.phase == "place"
        else None,
        "student_teacher_relabel_recovery_steps": (
            relabel_recovery_steps if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_left_recovery_steps": (
            relabel_left_recovery_steps if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_right_recovery_steps": (
            relabel_right_recovery_steps if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_max_left_wrist_error_m": (
            relabel_max_left_wrist_error_m if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_max_right_wrist_error_m": (
            relabel_max_right_wrist_error_m
            if args_cli.student_teacher_relabel
            else None
        ),
        "staging_stable_steps": staging_stable_steps
        if args_cli.phase == "carry"
        else None,
        "staging_reached": staging_reached if args_cli.phase == "carry" else None,
    }
    if student_policy is not None and not args_cli.student_teacher_relabel:
        recorder.discard_episode(reason=reason, metadata=final_metadata)
    elif success or args_cli.keep_failures or args_cli.student_teacher_relabel:
        recorder.end_episode(
            success=success, termination_reason=reason, metadata=final_metadata
        )
    else:
        recorder.discard_episode(reason=reason, metadata=final_metadata)
    if success and staging_snapshot is not None:
        snapshot_store.append(staging_snapshot)
    return {
        "success": success,
        "recorded": student_policy is None or args_cli.student_teacher_relabel,
        "reason": reason,
        "steps": step + 1,
        "source_episode": snapshot.source_episode,
        "max_hold_steps": hold_steps,
        "staging_reached": staging_reached if args_cli.phase == "carry" else None,
        "staging_stable_steps": (
            staging_stable_steps if args_cli.phase == "carry" else None
        ),
        "place_progress_step": (
            place_progress_step if args_cli.phase == "place" else None
        ),
        "student_teacher_relabel_recovery_steps": (
            relabel_recovery_steps if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_left_recovery_steps": (
            relabel_left_recovery_steps if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_right_recovery_steps": (
            relabel_right_recovery_steps if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_max_left_wrist_error_m": (
            relabel_max_left_wrist_error_m if args_cli.student_teacher_relabel else None
        ),
        "student_teacher_relabel_max_right_wrist_error_m": (
            relabel_max_right_wrist_error_m
            if args_cli.student_teacher_relabel
            else None
        ),
    }


def main() -> None:
    print("[teacher] validating BC-2 gate and snapshot provenance", flush=True)
    bc2_gate = validate_bc2_gate(
        args_cli.bc2_gate, required_action_key="processed_43d_joint_target"
    )
    gate_measured = bc2_gate["measured"]
    gate_strict_hold_seconds = float(gate_measured["strict_hold_seconds"])
    gate_strict_lift_m = float(gate_measured["strict_lift_m"])
    gate_required_hold_steps = int(gate_measured["required_hold_steps"])
    if (
        gate_strict_hold_seconds < STRICT_HOLD_SECONDS
        or gate_strict_lift_m < STRICT_LIFT_M
        or gate_required_hold_steps < SNAPSHOT_MIN_HOLD_STEPS
    ):
        raise ValueError("BC-2 gate weakened the downstream snapshot contract")
    required_kind = (
        SnapshotKind.BILATERAL_HOLD
        if args_cli.phase == "carry"
        else SnapshotKind.CARRY_AT_STAGING
    )
    snapshot_input_path = Path(args_cli.snapshot_input).expanduser().resolve()
    snapshot_input_sha256 = file_sha256(snapshot_input_path)
    gate_snapshot_path = Path(bc2_gate["snapshot_output_path"]).expanduser().resolve()
    gate_snapshot_sha256 = bc2_gate["snapshot_output_sha256"]
    snapshot_metadata = load_snapshot_store_metadata(args_cli.snapshot_input)
    producer = snapshot_metadata.get("producer")
    allowed_producers = (
        {"evaluate_bc2_rollout", "play_bc_skill_fsm"}
        if required_kind == SnapshotKind.BILATERAL_HOLD
        else {"collect_carry_place_teacher:carry", "play_bc_skill_fsm"}
    )
    if producer not in allowed_producers:
        raise ValueError(
            f"{required_kind.name.lower()} snapshots must come from one of "
            f"{sorted(allowed_producers)!r}"
        )
    if snapshot_metadata.get("checkpoint_sha256") != bc2_gate.get("checkpoint_sha256"):
        raise ValueError("snapshot store is not bound to the certified BC-2 checkpoint")
    if snapshot_metadata.get("action_contract_sha256") != bc2_gate.get(
        "action_contract_sha256"
    ):
        raise ValueError("snapshot store action contract does not match the BC-2 gate")
    if snapshot_metadata.get("action_key") != bc2_gate.get("action_key"):
        raise ValueError("snapshot store action key does not match the BC-2 gate")
    if args_cli.phase == "carry":
        if producer == "evaluate_bc2_rollout":
            if snapshot_input_path != gate_snapshot_path:
                raise ValueError(
                    "carry snapshot input path is not the snapshot file bound to the BC-2 gate"
                )
            if snapshot_input_sha256 != gate_snapshot_sha256:
                raise ValueError(
                    "carry snapshot input SHA-256 is not the snapshot file bound to the BC-2 gate"
                )
        else:
            if snapshot_metadata.get("actual_approach_grasp_chain") is not True:
                raise ValueError(
                    "FSM carry snapshots must contain the actual approach-grasp chain"
                )
            if snapshot_metadata.get("bc2_gate_sha256") != file_sha256(
                args_cli.bc2_gate
            ):
                raise ValueError("FSM snapshot BC-2 gate SHA-256 mismatch")
            if (
                Path(str(snapshot_metadata.get("grasp_checkpoint", ""))).resolve()
                != Path(bc2_gate["checkpoint_path"]).resolve()
            ):
                raise ValueError(
                    "FSM snapshots used a grasp checkpoint not certified by the gate"
                )
    else:
        if producer == "play_bc_skill_fsm":
            if snapshot_metadata.get(
                "actual_approach_grasp_chain"
            ) is not True or snapshot_metadata.get("bc2_gate_sha256") != file_sha256(
                args_cli.bc2_gate
            ):
                raise ValueError(
                    "Student FSM staging snapshots are not bound to the certified BC-2 gate"
                )
        else:
            source_snapshot_value = snapshot_metadata.get("source_snapshot")
            if not isinstance(source_snapshot_value, str):
                raise ValueError("staging snapshots have no physical parent snapshot")
            source_snapshot_path = Path(source_snapshot_value).expanduser().resolve()
            source_snapshot_recorded_sha = snapshot_metadata.get(
                "source_snapshot_sha256"
            )
            if source_snapshot_recorded_sha != file_sha256(source_snapshot_path):
                raise ValueError(
                    "staging snapshot parent SHA-256 does not match its file"
                )
            if source_snapshot_path == gate_snapshot_path:
                if source_snapshot_recorded_sha != gate_snapshot_sha256:
                    raise ValueError(
                        "staging snapshot parent SHA-256 does not match the BC-2 gate"
                    )
            else:
                # Production Carry may descend from a deployed FSM grasp.
                source_metadata = load_snapshot_store_metadata(source_snapshot_path)
                if (
                    source_metadata.get("producer") != "play_bc_skill_fsm"
                    or source_metadata.get("actual_approach_grasp_chain") is not True
                    or source_metadata.get("bc2_gate_sha256")
                    != file_sha256(args_cli.bc2_gate)
                    or source_metadata.get("checkpoint_sha256")
                    != bc2_gate.get("checkpoint_sha256")
                    or source_metadata.get("action_contract_sha256")
                    != bc2_gate.get("action_contract_sha256")
                    or source_metadata.get("action_key") != bc2_gate.get("action_key")
                ):
                    raise ValueError(
                        "staging snapshot FSM parent is not bound to the certified BC-2 gate"
                    )
            publication_metrics_value = snapshot_metadata.get("publication_metrics")
            if not isinstance(publication_metrics_value, str):
                raise ValueError("staging snapshots have no carry publication binding")
            publication_metrics_path = (
                Path(publication_metrics_value).expanduser().resolve()
            )
            if not publication_metrics_path.is_file():
                raise FileNotFoundError(
                    f"carry publication metrics are missing: {publication_metrics_path}"
                )
            publication_metrics = json.loads(
                publication_metrics_path.read_text(encoding="utf-8")
            )
            if publication_metrics.get("collection_run_id") != snapshot_metadata.get(
                "collection_run_id"
            ):
                raise ValueError("staging snapshot collection run id is stale")
            if publication_metrics.get("staging_snapshot_published") is not True:
                raise ValueError("latest carry run did not publish staging snapshots")
            if (
                Path(
                    str(
                        publication_metrics.get("published_staging_snapshot_output", "")
                    )
                ).resolve()
                != snapshot_input_path
            ):
                raise ValueError("staging snapshot publication path mismatch")
            if (
                publication_metrics.get("staging_snapshot_sha256")
                != snapshot_input_sha256
            ):
                raise ValueError("staging snapshot publication SHA-256 mismatch")
    metadata_hold_seconds = float(snapshot_metadata.get("strict_hold_seconds", 0.0))
    metadata_lift_m = float(snapshot_metadata.get("strict_lift_m", 0.0))
    metadata_hold_steps = int(snapshot_metadata.get("required_hold_steps", 0))
    if (
        metadata_hold_seconds < STRICT_HOLD_SECONDS
        or metadata_lift_m < STRICT_LIFT_M
        or metadata_hold_steps < gate_required_hold_steps
        or not math.isclose(
            metadata_hold_seconds, gate_strict_hold_seconds, abs_tol=1.0e-12
        )
        or not math.isclose(metadata_lift_m, gate_strict_lift_m, abs_tol=1.0e-12)
    ):
        raise ValueError(
            "snapshot store does not preserve the gate's strict hold/lift contract"
        )
    print("[teacher] snapshot provenance valid; creating Isaac environment", flush=True)
    cfg = _configure_environment()
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
    # Full joint state arrays are restored by index, so provenance also
    # requires the exact articulation order used by the producer.
    try:
        snapshots = load_snapshots(
            args_cli.snapshot_input,
            kind=required_kind,
            expected_joint_names=env.scene["robot"].data.joint_names,
            lift_threshold_m=gate_strict_lift_m,
            hold_steps_required=gate_required_hold_steps,
            require_unprocessed_wbc_raw_action=args_cli.phase == "place",
        )
        if not snapshots:
            raise RuntimeError(
                f"No validated {required_kind.name.lower()} snapshots found"
            )
        unique_source_snapshot_count = len(snapshots)
        if args_cli.repeat_snapshots:
            snapshots = tuple(
                snapshots[
                    (args_cli.snapshot_start_offset + index)
                    % unique_source_snapshot_count
                ]
                for index in range(args_cli.num_trials)
            )
        else:
            snapshots = snapshots[: args_cli.num_trials]
    except BaseException:
        env.close()
        raise
    if not math.isclose(env.step_dt, 1.0 / CONTROL_HZ, abs_tol=1.0e-6):
        raise RuntimeError(f"Expected 50 Hz control, got {env.step_dt}")
    task_cfg = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.box_carry_with_hand_env_cfg"
    )
    output = Path(args_cli.output).expanduser().resolve()
    metrics_output = output.with_suffix(".metrics.json")
    collection_run_id = uuid.uuid4().hex
    preexisting_output_sha256 = file_sha256(output) if output.is_file() else None
    staging_output = (
        None
        if args_cli.staging_snapshot_output is None
        else Path(args_cli.staging_snapshot_output).expanduser().resolve()
    )
    preexisting_staging_sha256 = (
        file_sha256(staging_output)
        if staging_output is not None and staging_output.is_file()
        else None
    )
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(
            {
                "status": "running",
                "collection_run_id": collection_run_id,
                "phase": args_cli.phase,
                "requested_output": str(output),
                "dataset_published": False,
                "staging_snapshot_published": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def write_aborted_metrics(exc: BaseException, *, completed_trials: int) -> None:
        metrics_output.write_text(
            json.dumps(
                {
                    "status": "aborted",
                    "collection_run_id": collection_run_id,
                    "phase": args_cli.phase,
                    "requested_output": str(output),
                    "dataset_published": False,
                    "staging_snapshot_published": False,
                    "completed_trials": completed_trials,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        joint_names = list(
            env.action_manager.get_term("JointPositionAction")._joint_names
        )
        carry_runtime_contract = (
            build_carry_runtime_contract(
                joint_names,
                navigation_ramp_steps=args_cli.navigation_ramp_steps,
            )
            if args_cli.phase == "carry"
            else None
        )
        place_runtime_contract = (
            build_place_runtime_contract(
                joint_names,
                arm_offset_decay_steps=args_cli.place_arm_offset_decay_steps,
            )
            if args_cli.phase == "place"
            else None
        )
        student_policy = None
        if args_cli.student_checkpoint is not None:
            student_policy = load_bc_policy(
                args_cli.student_checkpoint, args_cli.device
            )
            expected_skill = (
                SkillPhase.CARRY if args_cli.phase == "carry" else SkillPhase.PLACE
            )
            if student_policy.skill_phase != expected_skill:
                raise ValueError(
                    f"Student checkpoint is {student_policy.skill_phase.name}, "
                    f"expected {expected_skill.name}"
                )
            if student_policy.action_key == "processed_43d_joint_target" and args_cli.phase == "carry":
                if args_cli.student_teacher_relabel:
                    raise ValueError("43-D Carry evaluation does not support Teacher relabeling")
                validate_runtime_action_contract(student_policy, joint_names)
            elif student_policy.action_key == "teacher_high_level_action":
                validate_runtime_action_contract(
                    student_policy, DEFAULT_HIGH_LEVEL_ACTION_NAMES
                )
            else:
                raise ValueError(
                    "snapshot phase evaluation requires a 23-D phase or 43-D Carry checkpoint"
                )
            if args_cli.phase == "carry":
                checkpoint_contract = student_policy.carry_runtime_contract
                runtime_contract = carry_runtime_contract
            else:
                checkpoint_contract = student_policy.place_runtime_contract
                runtime_contract = place_runtime_contract
            if checkpoint_contract is None or (
                checkpoint_contract["contract_sha256"]
                != runtime_contract["contract_sha256"]
            ):
                raise ValueError(
                    "Student checkpoint runtime contract does not match the "
                    "requested physical evaluation configuration"
                )
    except BaseException as exc:
        write_aborted_metrics(exc, completed_trials=0)
        env.close()
        raise

    metadata = {
        **schema_metadata(),
        "collection_run_id": collection_run_id,
        "task": args_cli.task,
        "teacher_phase": args_cli.phase,
        "teacher": "Arena Pink/Homie from complete physical prior-phase snapshots",
        "action_source": (
            "teacher_relabels_on_student_states"
            if args_cli.student_teacher_relabel
            else (
                "student_checkpoint"
                if args_cli.student_checkpoint is not None
                else "teacher"
            )
        ),
        "student_checkpoint": (
            str(Path(args_cli.student_checkpoint).expanduser().resolve())
            if args_cli.student_checkpoint is not None
            else None
        ),
        "student_checkpoint_sha256": (
            file_sha256(args_cli.student_checkpoint)
            if args_cli.student_checkpoint is not None
            else None
        ),
        "student_teacher_relabel_config": (
            {
                "wrist_error_threshold_m": args_cli.place_relabel_wrist_error_m,
                "recovery_compression_m": args_cli.place_recovery_compression_m,
                "contact_loss_grace_steps": args_cli.place_contact_loss_grace_steps,
                "recovery_target": ("measured_box_position_with_initial_height_floor"),
                "nominal_clock": "place_wall_step_matches_phase_clock_v1",
            }
            if args_cli.student_teacher_relabel
            else None
        ),
        "snapshot_input": str(Path(args_cli.snapshot_input).expanduser().resolve()),
        "snapshot_required_kind": required_kind.name.lower(),
        "box_teleport_into_hands": False,
        "fixed_box_size_m": [args_cli.box_size] * 3,
        "fixed_box_mass_kg": args_cli.box_mass,
        "work_area_randomization": {
            "enabled": args_cli.randomize_work_area,
            "side_length_m": args_cli.work_area_size_m,
            "snapshot_schema_preserves_table_and_bin_poses": True,
        },
        "bc2_gate": str(Path(args_cli.bc2_gate).expanduser().resolve()),
        "bc2_checkpoint_sha256": bc2_gate["checkpoint_sha256"],
        "snapshot_input_sha256": snapshot_input_sha256,
        "strict_hold_seconds": gate_strict_hold_seconds,
        "strict_lift_m": gate_strict_lift_m,
        "required_hold_steps": gate_required_hold_steps,
        "controller_handoff_steps": args_cli.controller_handoff_steps,
        "loaded_stand_steps": args_cli.loaded_stand_steps,
        "staging_stabilization_steps": args_cli.staging_stabilization_steps,
        "source_snapshot_producer": producer,
        "actual_approach_grasp_chain": producer == "play_bc_skill_fsm",
        "repeat_snapshots": args_cli.repeat_snapshots,
        "snapshot_start_offset": args_cli.snapshot_start_offset,
        "unique_source_snapshot_count": unique_source_snapshot_count,
        "carry_navigation_ramp_steps": args_cli.navigation_ramp_steps,
        "carry_staging_standoff_m": args_cli.carry_staging_standoff_m,
        "place_arm_offset_decay_steps": (
            args_cli.place_arm_offset_decay_steps if args_cli.phase == "place" else None
        ),
        "place_trajectory": (
            {
                "raise_steps": args_cli.place_raise_steps,
                "translate_steps": args_cli.place_translate_steps,
                "lower_steps": args_cli.place_lower_steps,
                "release_steps": 50,
                "clearance_m": 0.08,
                "release_height_m": args_cli.place_release_height_m,
            }
            if args_cli.phase == "place"
            else None
        ),
        "adaptive_place": (
            {
                "enabled": args_cli.adaptive_place,
                "tracking_tolerance_m": args_cli.place_tracking_tolerance_m,
                "release_xy_tolerance_m": args_cli.place_release_xy_tolerance_m,
                "release_z_tolerance_m": args_cli.place_release_z_tolerance_m,
                "release_speed_mps": args_cli.place_release_speed_mps,
                "recovery_compression_m": args_cli.place_recovery_compression_m,
                "contact_loss_grace_steps": args_cli.place_contact_loss_grace_steps,
                "base_follow_max_speed_mps": args_cli.place_base_follow_max_speed_mps,
                "base_follow_gain": args_cli.place_base_follow_gain,
                "base_follow_stop_distance_m": args_cli.place_base_follow_stop_distance_m,
                "landing_overshoot_m": args_cli.place_landing_overshoot_m,
            }
            if args_cli.phase == "place"
            else None
        ),
        "carry_realized_arm_hand_targets_locked": args_cli.phase == "carry",
        "teacher_high_level_action_requires_grasp_lock_postprocessor_for_carry": (
            args_cli.phase == "carry"
        ),
    }
    if carry_runtime_contract is not None:
        metadata["carry_runtime_contract"] = carry_runtime_contract
        metadata["carry_runtime_contract_sha256"] = carry_runtime_contract[
            "contract_sha256"
        ]
    if place_runtime_contract is not None:
        metadata["place_runtime_contract"] = place_runtime_contract
        metadata["place_runtime_contract_sha256"] = place_runtime_contract[
            "contract_sha256"
        ]
    try:
        recorder = TeacherDemoRecorder(
            output,
            control_frequency_hz=CONTROL_HZ,
            joint_names=joint_names,
            phase_names={0: "approach", 1: "carry", 2: "place", 3: "complete"},
            metadata=metadata,
            overwrite=args_cli.overwrite,
        )
    except BaseException as exc:
        write_aborted_metrics(exc, completed_trials=0)
        env.close()
        raise
    snapshot_store = None
    if args_cli.phase == "carry" and args_cli.student_checkpoint is None:
        try:
            snapshot_store = PhysicalSnapshotStore(
                args_cli.staging_snapshot_output,
                joint_names=list(env.scene["robot"].data.joint_names),
                overwrite=args_cli.overwrite,
                metadata={
                    "collection_run_id": collection_run_id,
                    "publication_metrics": str(metrics_output),
                    "producer": "collect_carry_place_teacher:carry",
                    "source": str(output),
                    "gate": "actual carry rollout at camera staging",
                    "checkpoint_sha256": bc2_gate["checkpoint_sha256"],
                    "action_key": bc2_gate["action_key"],
                    "action_contract_sha256": bc2_gate["action_contract_sha256"],
                    "source_snapshot": str(
                        Path(args_cli.snapshot_input).expanduser().resolve()
                    ),
                    "source_snapshot_sha256": snapshot_input_sha256,
                    "strict_hold_seconds": gate_strict_hold_seconds,
                    "strict_lift_m": gate_strict_lift_m,
                    "required_hold_steps": gate_required_hold_steps,
                    "carry_at_staging_contains_unprocessed_wbc_raw_action": True,
                },
            )
        except BaseException as exc:
            recorder.abort()
            write_aborted_metrics(exc, completed_trials=0)
            env.close()
            raise
    results = []
    dataset_published = False
    staging_snapshot_published = False
    try:
        with ArenaWBCBridge(
            arena_root=args_cli.arena_root, step_timeout_s=10.0
        ) as bridge:
            for index, snapshot in enumerate(snapshots):
                result = _run_trial(
                    env,
                    bridge,
                    snapshot,
                    recorder,
                    task_cfg=task_cfg,
                    snapshot_store=snapshot_store,
                    carry_runtime_contract=carry_runtime_contract,
                    place_runtime_contract=place_runtime_contract,
                    student_policy=student_policy,
                    trial_index=index,
                )
                results.append(result)
                displayed_result = {key: value for key, value in result.items() if key != "trace"}
                print(f"[{index + 1}/{len(snapshots)}] {displayed_result}", flush=True)
                if (index + 1) % 5 == 0:
                    progress_successes = sum(int(item["success"]) for item in results)
                    metrics_output.write_text(
                        json.dumps(
                            {
                                "status": "running",
                                "collection_run_id": collection_run_id,
                                "phase": args_cli.phase,
                                "completed_trials": len(results),
                                "successes": progress_successes,
                                "target_successes": args_cli.target_successes,
                                "dataset_published": False,
                                "staging_snapshot_published": False,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                if (
                    args_cli.target_successes
                    and sum(int(item["success"]) for item in results)
                    >= args_cli.target_successes
                ):
                    break
        successes = sum(int(result["success"]) for result in results)
        recorded = (
            sum(int(result.get("recorded", False)) for result in results)
            if args_cli.student_teacher_relabel
            else (
                0
                if student_policy is not None or args_cli.rule_based_place
                else sum(
                    int(result["success"])
                    + int(
                        args_cli.keep_failures
                        and not result["success"]
                        and result.get("recorded", False)
                    )
                    for result in results
                )
            )
        )
        if recorded == 0:
            recorder.abort()
        else:
            recorder.finalize()
            dataset_published = True
        if snapshot_store is not None:
            if successes:
                snapshot_store.finalize()
                staging_snapshot_published = True
            else:
                snapshot_store.abort()
    except BaseException as exc:
        recorder.abort()
        if snapshot_store is not None:
            snapshot_store.abort()
        write_aborted_metrics(exc, completed_trials=len(results))
        raise
    finally:
        env.close()

    handoff_gate_successes = sum(
        not str(result["reason"]).startswith("handoff_")
        and result["reason"] != "restored_snapshot_failed_physical_gate"
        for result in results
    )
    loaded_stand_gate_successes = sum(
        not str(result["reason"]).startswith("handoff_")
        and not str(result["reason"]).startswith("loaded_stand_")
        and result["reason"] != "restored_snapshot_failed_physical_gate"
        for result in results
    )

    def conditional_gate(success_count: int, eligible_count: int) -> dict:
        return {
            "eligible": eligible_count,
            "successes": success_count,
            "conditional_probability": (
                success_count / eligible_count if eligible_count else None
            ),
        }

    metrics = {
        "status": "completed",
        "evaluation_mode": (
            "rule_based_place" if args_cli.rule_based_place else "student_teacher_relabel"
            if args_cli.student_teacher_relabel
            else (
                "student_checkpoint"
                if args_cli.student_checkpoint is not None
                else "teacher_collection"
            )
        ),
        "student_checkpoint": (
            str(Path(args_cli.student_checkpoint).expanduser().resolve())
            if args_cli.student_checkpoint is not None
            else None
        ),
        "student_checkpoint_sha256": (
            file_sha256(args_cli.student_checkpoint)
            if args_cli.student_checkpoint is not None
            else None
        ),
        "collection_run_id": collection_run_id,
        "phase": args_cli.phase,
        "trials": len(results),
        "successes": successes,
        "target_successes": args_cli.target_successes,
        "target_successes_reached": (
            successes >= args_cli.target_successes
            if args_cli.target_successes
            else None
        ),
        "conditional_success_rate": successes / len(results),
        "physical_gates": (
            {
                "handoff_survival_given_grasp": conditional_gate(
                    handoff_gate_successes, len(results)
                ),
                "loaded_stand_given_handoff": conditional_gate(
                    loaded_stand_gate_successes, handoff_gate_successes
                ),
                "walk_given_loaded_stand": conditional_gate(
                    successes, loaded_stand_gate_successes
                ),
            }
            if args_cli.phase == "carry"
            else None
        ),
        "source_snapshot_producer": producer,
        "actual_approach_grasp_chain": producer == "play_bc_skill_fsm",
        "repeat_snapshots": args_cli.repeat_snapshots,
        "snapshot_start_offset": args_cli.snapshot_start_offset,
        "unique_source_snapshot_count": unique_source_snapshot_count,
        "handoff_survival_gate_steps": args_cli.controller_handoff_steps,
        "loaded_stand_gate_steps": (
            args_cli.loaded_stand_steps if args_cli.phase == "carry" else None
        ),
        "source_snapshot_kind": required_kind.name.lower(),
        "video_output_prefix": None if video_path is None else str(video_path),
        "results": results,
        "requested_output": str(output),
        "dataset_published": dataset_published,
        "published_output": str(output) if dataset_published else None,
        "dataset_sha256": file_sha256(output) if dataset_published else None,
        "stale_existing_output_ignored": (
            not dataset_published and preexisting_output_sha256 is not None
        ),
        "preexisting_output_sha256": (
            preexisting_output_sha256 if not dataset_published else None
        ),
        "requested_staging_snapshot_output": (
            None if staging_output is None else str(staging_output)
        ),
        "staging_snapshot_published": staging_snapshot_published,
        "published_staging_snapshot_output": (
            str(staging_output) if staging_snapshot_published else None
        ),
        "staging_snapshot_sha256": (
            file_sha256(staging_output) if staging_snapshot_published else None
        ),
        "stale_existing_staging_snapshot_ignored": (
            not staging_snapshot_published and preexisting_staging_sha256 is not None
        ),
        "preexisting_staging_snapshot_sha256": (
            preexisting_staging_sha256 if not staging_snapshot_published else None
        ),
    }
    metrics_output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    displayed_metrics = dict(metrics)
    if args_cli.rule_based_place:
        displayed_metrics["results"] = [
            {key: value for key, value in result.items() if key != "trace"}
            for result in results
        ]
    print(json.dumps(displayed_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac Sim shutdown may otherwise swallow the active traceback and
        # return status zero, making provenance/configuration failures opaque.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
