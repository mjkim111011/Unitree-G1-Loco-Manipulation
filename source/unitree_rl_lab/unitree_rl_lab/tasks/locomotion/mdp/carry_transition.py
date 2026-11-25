"""MDP terms for the learned WBC-to-CARRY transition policy.

Every term reads ``env.transition_controller``
(:class:`unitree_rl_lab.utils.transition_policy.CarryTransitionController`).
Observation terms return zeros of the correct width before the controller is
attached, because the observation manager evaluates each term once while it is
being constructed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


BOX_CONTACT_FORCE_THRESHOLD_N = 1.0
STRICT_LIFT_M = 0.12
# Half extents of the G1 sole used to approximate the support region.
FOOT_HALF_LENGTH_M = 0.10
FOOT_HALF_WIDTH_M = 0.04
SWING_LOAD_FRACTION = 0.05
STANCE_LOAD_FRACTION = 0.60


def _controller(env: ManagerBasedRLEnv):
    return getattr(env, "transition_controller", None)


def _zeros(env: ManagerBasedRLEnv, width: int) -> torch.Tensor:
    return torch.zeros(env.num_envs, width, device=env.device)


def _to_yaw_frame(env: ManagerBasedRLEnv, vector_w: torch.Tensor) -> torch.Tensor:
    quat = yaw_quat(env.scene["robot"].data.root_quat_w)
    return quat_apply_inverse(quat, vector_w)


def box_contact_counts(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Box-only fingertip contact counts for the left and right hands."""

    left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    right = torch.zeros_like(left)
    for name, sensor in env.scene.sensors.items():
        if not name.endswith("_box_contact") or sensor.data.force_matrix_w is None:
            continue
        force = torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).flatten(start_dim=1).amax(dim=-1)
        touching = (force >= BOX_CONTACT_FORCE_THRESHOLD_N).long()
        if name.startswith("left_"):
            left += touching
        elif name.startswith("right_"):
            right += touching
    return left, right


def box_lift_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    box_z = env.scene["box"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rest = controller.box_rest_height if controller is not None else box_z
    return box_z - rest


def foot_loads(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vertical left/right ankle contact force as a fraction of robot weight."""

    controller = _controller(env)
    if controller is None:
        return _zeros(env, 2)
    sensor = env.scene.sensors["contact_forces"]
    if not hasattr(controller, "_foot_sensor_ids"):
        controller._foot_sensor_ids = sensor.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]
    force_z = sensor.data.net_forces_w[:, controller._foot_sensor_ids, 2].clamp_min(0.0)
    return force_z / (controller.robot_mass[:, None] * 9.81)


def com_relative_to_feet_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """CoM minus feet midpoint (xy) in the yaw-aligned base frame."""

    controller = _controller(env)
    if controller is None:
        return _zeros(env, 2)
    feet = env.scene["robot"].data.body_link_pos_w[:, controller.foot_body_ids]
    relative = controller.com_position_w() - feet.mean(dim=1)
    relative[:, 2] = 0.0
    return _to_yaw_frame(env, relative)[:, :2]


def feet_positions_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left and right foot xy relative to the CoM, in the yaw frame: (N, 2, 2)."""

    controller = _controller(env)
    feet = env.scene["robot"].data.body_link_pos_w[:, controller.foot_body_ids]
    relative = feet - controller.com_position_w()[:, None, :]
    quat = yaw_quat(env.scene["robot"].data.root_quat_w)[:, None, :].expand(-1, 2, -1)
    return quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 2, 3)[..., :2]


# ---------------------------------------------------------------- observations
def transition_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    if controller is None:
        return _zeros(env, 2)
    elapsed = controller.elapsed.float() / controller.cfg.gate.max_steps
    return torch.stack((controller.alpha, elapsed.clamp(max=2.0)), dim=-1)


def com_state_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """CoM xy offset from the feet midpoint and CoM xy velocity, yaw frame."""

    controller = _controller(env)
    if controller is None:
        return _zeros(env, 4)
    robot = env.scene["robot"]
    com_vel = (controller.body_masses[:, :, None] * robot.data.body_com_lin_vel_w).sum(dim=1)
    com_vel = com_vel / controller.robot_mass[:, None]
    return torch.cat((com_relative_to_feet_b(env), _to_yaw_frame(env, com_vel)[:, :2]), dim=-1)


def foot_load_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return foot_loads(env)


def com_joint_error_to_anchor(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    if controller is None:
        return _zeros(env, 15)
    ids = controller.com_action_ids
    anchor_target = controller.anchor[:, ids] * controller.scale[ids] + controller.offset[:, ids]
    return env.scene["robot"].data.joint_pos[:, controller.com_joint_ids] - anchor_target


def bc3_preview(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Where BC3 would drive the leg/waist targets right now, relative to the anchor (rad)."""

    controller = _controller(env)
    if controller is None:
        return _zeros(env, 15)
    ids = controller.com_action_ids
    return (controller.bc3_raw_action()[:, ids] - controller.anchor[:, ids]) * controller.scale[ids]


def transition_velocity_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    if controller is None:
        return _zeros(env, 3)
    return controller.navigation_velocity()


def box_state_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Box position/velocity relative to the pelvis plus per-hand contact counts."""

    if _controller(env) is None:
        return _zeros(env, 8)
    robot, box = env.scene["robot"], env.scene["box"]
    position = _to_yaw_frame(env, box.data.root_link_pos_w - robot.data.root_link_pos_w)
    velocity = _to_yaw_frame(env, box.data.root_com_lin_vel_w - robot.data.root_com_lin_vel_w)
    left, right = box_contact_counts(env)
    return torch.cat((position, velocity, left[:, None].float() / 3.0, right[:, None].float() / 3.0), dim=-1)


def box_lift_height_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if _controller(env) is None:
        return _zeros(env, 1)
    return box_lift_height(env)[:, None]


def transition_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    if controller is None:
        return _zeros(env, 3)
    return controller.gate_action


TRANSITION_OBSERVATION_DIM = 61


def transition_policy_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The transition actor's input; shared verbatim by training and the live chain."""

    robot = env.scene["robot"].data
    features = torch.cat(
        (
            transition_progress(env),  # 2
            com_state_b(env),  # 4
            foot_load_obs(env),  # 2
            robot.projected_gravity_b,  # 3
            0.2 * robot.root_ang_vel_b,  # 3
            robot.root_lin_vel_b,  # 3
            com_joint_error_to_anchor(env),  # 15
            bc3_preview(env),  # 15
            transition_velocity_command(env),  # 3
            box_state_b(env),  # 8
            transition_last_action(env),  # 3
        ),
        dim=-1,
    )
    if features.shape[-1] != TRANSITION_OBSERVATION_DIM:
        raise RuntimeError(f"transition observation has {features.shape[-1]} features")
    return features


# ---------------------------------------------------------------- rewards
def bilateral_box_hold(env: ManagerBasedRLEnv) -> torch.Tensor:
    left, right = box_contact_counts(env)
    return ((left > 0) & (right > 0) & (box_lift_height(env) >= STRICT_LIFT_M)).float()


def joint_target_jump_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared per-step change of the unlocked joint targets (the boundary discontinuity)."""

    controller = _controller(env)
    return controller.target_delta[:, controller.unlocked].square().sum(dim=-1)


def transition_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    return (controller.gate_action - controller.previous_gate_action).square().sum(dim=-1)


def box_relative_velocity_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot, box = env.scene["robot"], env.scene["box"]
    return (box.data.root_com_lin_vel_w - robot.data.root_com_lin_vel_w).square().sum(dim=-1)


def com_support_violation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance (m) of the CoM outside the feet bounding box expanded by the sole size."""

    feet = feet_positions_b(env)
    half = torch.tensor((FOOT_HALF_LENGTH_M, FOOT_HALF_WIDTH_M), device=env.device)
    low = feet.amin(dim=1) - half
    high = feet.amax(dim=1) + half
    # The CoM is the origin of this frame.
    outside = torch.clamp(low, min=0.0) + torch.clamp(-high, min=0.0)
    return torch.linalg.vector_norm(outside, dim=-1)


def alpha_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Alpha increase rate; integrates to exactly 1 over a completed handoff."""

    controller = _controller(env)
    return (controller.alpha - controller.previous_alpha) / env.step_dt


def pending_handoff(env: ManagerBasedRLEnv) -> torch.Tensor:
    return 1.0 - _controller(env).alpha


def bc3_velocity_tracking(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """Downstream quality: BC3 tracking of the carry command once it fully owns the robot."""

    controller = _controller(env)
    command = controller.navigation_velocity()
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error = (command[:, :2] - velocity).square().sum(dim=-1)
    return torch.exp(-error / std**2) * (controller.alpha >= 1.0).float()


def weight_shift_at_first_swing(env: ManagerBasedRLEnv, std: float = 0.04) -> torch.Tensor:
    """Sparse bonus when the first foot leaves the ground with the CoM over the stance foot."""

    controller = _controller(env)
    loads = foot_loads(env)
    left_swing = (loads[:, 0] < SWING_LOAD_FRACTION) & (loads[:, 1] > STANCE_LOAD_FRACTION)
    right_swing = (loads[:, 1] < SWING_LOAD_FRACTION) & (loads[:, 0] > STANCE_LOAD_FRACTION)
    event = (left_swing | right_swing) & ~controller.first_swing_done
    feet = feet_positions_b(env)
    stance = torch.where(left_swing[:, None], feet[:, 1], feet[:, 0])
    bonus = torch.exp(-stance.square().sum(dim=-1) / std**2) * event.float()
    controller.first_swing_done |= event
    return bonus / env.step_dt


# ---------------------------------------------------------------- terminations
def transition_box_dropped(
    env: ManagerBasedRLEnv, table_surface_height: float = 0.74, minimum_lift_m: float = 0.08
) -> torch.Tensor:
    """The live chain's ``box_drop`` criterion."""

    if _controller(env) is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    box_z = env.scene["box"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (box_z < table_surface_height - 0.10) | (box_lift_height(env) < minimum_lift_m)


# ---------------------------------------------------------------- events
def reset_carry_transition_from_snapshot(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    controller = _controller(env)
    if controller is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    controller.reset_from_snapshots(env_ids)
