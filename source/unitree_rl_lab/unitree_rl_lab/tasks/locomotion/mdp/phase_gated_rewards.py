"""Phase-aware locomotion and posture rewards for pick-and-place."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import action_rate_l2 as _action_rate_l2
from isaaclab.envs.mdp import ang_vel_xy_l2 as _ang_vel_xy_l2
from isaaclab.envs.mdp import joint_deviation_l1 as _joint_deviation_l1
from isaaclab.envs.mdp import track_ang_vel_z_exp as _track_ang_vel_z_exp
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    track_lin_vel_xy_yaw_frame_exp as _track_lin_vel_xy_yaw_frame_exp,
)

from .rewards import feet_gait as _feet_gait
from .rewards import foot_clearance_reward as _foot_clearance_reward
from .task_phases import (
    CARRY_PHASE,
    GRASP_PHASE,
    ensure_task_phase_buffers,
    grasp_stance_valid,
    grasp_walk_stance_valid,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _carry_phase_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the phase in which the robot is both carrying and holding.

    Initializing the buffers makes this safe if a reward is evaluated before a
    reset event has created the task state.  A newly initialized environment is
    in GRASP, where separate low-weight stance terms—not the navigation
    terms—are active.
    """

    ensure_task_phase_buffers(env)
    # ``phase_stable_hold`` is bimanual in this task.  Gating navigation on it
    # means walking toward the Bin after a drop cannot receive locomotion or
    # gait rewards during the brief drop grace interval.
    return (env.task_phase == CARRY_PHASE) & env.phase_stable_hold


def _grasp_phase_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the phase in which zero-command stance stabilization is needed."""

    ensure_task_phase_buffers(env)
    return env.task_phase == GRASP_PHASE


def _scale_in_grasp(env: ManagerBasedRLEnv, value: torch.Tensor, grasp_scale: float) -> torch.Tensor:
    """Scale a regularization cost in GRASP while preserving later phases."""

    ensure_task_phase_buffers(env)
    scale = torch.where(
        env.task_phase == GRASP_PHASE,
        torch.full_like(value, grasp_scale),
        torch.ones_like(value),
    )
    return value * scale


def grasp_phase_track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    max_base_tilt_angle: float = 0.55,
    min_base_height: float = 0.66,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
    moving_command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track GRASP xy command, allowing one-foot support while pre-grasp walks."""

    reward = _track_lin_vel_xy_yaw_frame_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    standing_stance = grasp_stance_valid(
        env,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        asset_cfg,
    )
    walking_stance = grasp_walk_stance_valid(
        env,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        asset_cfg,
    )
    command_xy_norm = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    stance = torch.where(command_xy_norm > moving_command_threshold, walking_stance, standing_stance)
    return reward * stance.to(dtype=reward.dtype)


def grasp_phase_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    max_base_tilt_angle: float = 0.55,
    min_base_height: float = 0.66,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
    moving_command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track GRASP yaw command, allowing one-foot support while searching/walking."""

    reward = _track_ang_vel_z_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    standing_stance = grasp_stance_valid(
        env,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        asset_cfg,
    )
    walking_stance = grasp_walk_stance_valid(
        env,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        asset_cfg,
    )
    yaw_command = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    stance = torch.where(yaw_command > moving_command_threshold, walking_stance, standing_stance)
    return reward * stance.to(dtype=reward.dtype)


def grasp_phase_stance_stability_reward(
    env: ManagerBasedRLEnv,
    lin_vel_std: float,
    ang_vel_std: float,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg,
    min_foot_contact_force: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small GRASP-only bonus for a quiet, double-supported upright stance.

    This is deliberately not a gait or foot-clearance reward.  It only makes
    the zero-command posture a slightly better local optimum while the hands
    approach the Box, and all arm-reaching rewards remain unavailable outside
    the same strict stance envelope.
    """
    robot = env.scene[asset_cfg.name]
    linear_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    roll_pitch_speed = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
    uprightness = torch.clamp(-robot.data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    quietness = torch.exp(-torch.square(linear_speed / lin_vel_std)) * torch.exp(
        -torch.square(roll_pitch_speed / ang_vel_std)
    )
    stance = grasp_stance_valid(
        env,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        asset_cfg,
    )
    return quietness * uprightness * stance.to(dtype=quietness.dtype)


def grasp_phase_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return roll/pitch angular velocity cost only while grasping.

    Unlike yaw tracking, this directly damps the angular motion that precedes
    a bad-orientation termination while an arm reaches for the Box.
    """

    cost = _ang_vel_xy_l2(env, asset_cfg=asset_cfg)
    return cost * _grasp_phase_mask(env).to(dtype=cost.dtype)


def grasp_scaled_action_rate_l2(
    env: ManagerBasedRLEnv,
    grasp_scale: float = 1.0,
) -> torch.Tensor:
    """Relax action-rate regularization only enough to permit arm reaching."""

    return _scale_in_grasp(env, _action_rate_l2(env), grasp_scale)


def grasp_scaled_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grasp_scale: float = 1.0,
) -> torch.Tensor:
    """Relax a selected joint group's nominal-pose cost in GRASP only."""

    return _scale_in_grasp(env, _joint_deviation_l1(env, asset_cfg=asset_cfg), grasp_scale)


def phase_gated_track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track xy velocity exactly as upstream, except return zero in GRASP."""

    reward = _track_lin_vel_xy_yaw_frame_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    return reward * _carry_phase_mask(env).to(dtype=reward.dtype)


def phase_gated_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track yaw velocity exactly as upstream, except return zero in GRASP."""

    reward = _track_ang_vel_z_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    return reward * _carry_phase_mask(env).to(dtype=reward.dtype)


def phase_gated_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward foot clearance exactly as before, only while carrying."""

    reward = _foot_clearance_reward(
        env,
        asset_cfg=asset_cfg,
        target_height=target_height,
        std=std,
        tanh_mult=tanh_mult,
    )
    return reward * _carry_phase_mask(env).to(dtype=reward.dtype)


def phase_gated_bimanual_feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str = "base_velocity",
    max_base_tilt_angle: float = 0.55,
    min_base_height: float = 0.66,
    min_foot_contact_force: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward gait only for safe Box approach or a stable bimanual carry.

    GRASP receives gait reward only when the camera-derived pre-grasp command
    is nonzero and the base has at least one supported foot.  CARRY receives
    it only while ``phase_stable_hold`` is true, preventing an unladen robot
    from earning gait reward after dropping the Box.
    """
    reward = _feet_gait(
        env,
        period=period,
        offset=offset,
        sensor_cfg=sensor_cfg,
        threshold=threshold,
        command_name=command_name,
    )
    ensure_task_phase_buffers(env)
    command_norm = torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=-1)
    grasp_approach = (
        grasp_walk_stance_valid(
            env,
            max_base_tilt_angle,
            min_base_height,
            sensor_cfg,
            min_foot_contact_force,
            asset_cfg,
        )
        & (command_norm > 0.1)
    )
    active = grasp_approach | _carry_phase_mask(env)
    return reward * active.to(dtype=reward.dtype)
