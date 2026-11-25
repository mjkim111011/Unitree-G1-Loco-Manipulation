"""MDP terms for the grip-aware residual over the direct BC2 -> BC3 handoff.

Every term reads ``env.grip_residual_controller``
(:class:`unitree_rl_lab.utils.grip_residual_policy.GripResidualController`).
Observation terms return zeros of the correct width before the controller is
attached, because the observation manager evaluates each term once while it is
being constructed.

Actor observations use only robot-side signals plus the controller's *Box
estimate* (delayed, noisy simulator truth); simulator truth and contact forces
go to the critic only.  Rewards may use simulator truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unitree_rl_lab.tasks.locomotion.mdp.carry_transition import box_contact_counts, foot_loads
from unitree_rl_lab.utils.teacher.grip_residual import (
    OBS_CLIP,
    OBS_SCALES,
    actor_components,
    assemble,
    blend_weight,
    critic_components,
    excess,
    layout_dim,
    one_sided_excess,
    residual_envelope,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _controller(env: ManagerBasedRLEnv):
    return getattr(env, "grip_residual_controller", None)


def _scaled(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: (v * OBS_SCALES.get(k, 1.0)).clamp(-OBS_CLIP, OBS_CLIP) for k, v in values.items()}


def actor_feature_values(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Unscaled actor features (robot-side signals and the Box estimate)."""

    controller = _controller(env)
    controller.prepare_observation()
    robot = env.scene["robot"].data
    cfg = controller.residual_cfg
    k = controller.elapsed
    active = (~controller.warmup_pending).float()
    progress = torch.stack(
        (
            blend_weight(k, cfg.schedule) * active,
            k.float() / cfg.schedule.fade_end_steps * active,
            residual_envelope(k, cfg.schedule) * active,
        ),
        dim=-1,
    )
    joints = controller.residual_joint_ids
    values = {
        "projected_gravity_b": robot.projected_gravity_b,
        "base_ang_vel_b": robot.root_ang_vel_b,
        "blend_progress": progress,
        "velocity_command": controller.navigation_velocity(),
        "joint_pos_rel": robot.joint_pos[:, joints] - controller.default_joint_pos,
        "joint_vel": robot.joint_vel[:, joints],
        "bc3_target_minus_joint_pos": controller.bc3_target_minus_joint_pos() * active[:, None],
        "previous_residual": controller.residual / cfg.max_residual_rad,
    }
    if cfg.variant == "grip":
        rel_change, rel_vel, lift_change = controller.box_estimate()
        values.update({
            "box_wrist_rel_pos_change": rel_change,
            "box_wrist_rel_vel": rel_vel,
            "box_lift_change": lift_change,
        })
    return values


def privileged_feature_values(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Simulator truth for the critic."""

    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    controller = _controller(env)
    measured = controller.measure()
    robot, box = env.scene["robot"].data, env.scene["box"].data
    quat = yaw_quat(robot.root_quat_w)
    forces = measured["hand_force_w"]
    forces_b = quat_apply_inverse(quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4), forces.reshape(-1, 3))
    box_quat = box.root_link_quat_w
    box_up = 1.0 - 2.0 * (box_quat[:, 1].square() + box_quat[:, 2].square())
    left, right = box_contact_counts(env)
    return {
        "box_wrist_rel_pos_change_true": measured["rel_pos_change"],
        "box_wrist_rel_vel_true": measured["rel_vel"],
        "box_lift_change_true": measured["lift_change"][:, None],
        "hand_box_force_b": forces_b.reshape(-1, 6),
        "hand_friction_ratio": measured["friction_ratio"],
        "box_tilt": torch.acos(box_up.clamp(-1.0, 1.0))[:, None],
        "box_ang_vel_b": quat_apply_inverse(quat, box.root_link_ang_vel_w),
        "base_lin_vel_b": robot.root_lin_vel_b,
        "foot_loads": foot_loads(env),
        "box_contact_counts": torch.stack((left, right), dim=-1).float() / 3.0,
    }


# ---------------------------------------------------------------- observations
def residual_policy_observation(env: ManagerBasedRLEnv, variant: str) -> torch.Tensor:
    """The residual actor's input; shared verbatim by training and the live chain."""

    components = actor_components(variant)
    controller = _controller(env)
    if controller is None:
        return torch.zeros(env.num_envs, layout_dim(components), device=env.device)
    if controller.residual_cfg.variant != variant:
        raise RuntimeError(f"observation variant {variant} != controller variant {controller.residual_cfg.variant}")
    return assemble(components, _scaled(actor_feature_values(env)))


def residual_critic_observation(env: ManagerBasedRLEnv, variant: str) -> torch.Tensor:
    components = critic_components(variant)
    controller = _controller(env)
    if controller is None:
        return torch.zeros(env.num_envs, layout_dim(components), device=env.device)
    values = actor_feature_values(env)
    values.update(privileged_feature_values(env))
    return assemble(components, _scaled(values))


# ---------------------------------------------------------------- rewards
def _active(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (~_controller(env).warmup_pending).float()


def tilt_excess(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Body tilt beyond the normal walking band, in threshold units."""

    return excess(_controller(env).measure()["tilt"], threshold)


def ang_vel_excess(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    return excess(_controller(env).measure()["ang_vel_xy"], threshold)


def box_wrist_height_excess(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Box-to-wrist height change since the handoff beyond the band (either direction)."""

    return excess(_controller(env).measure()["rel_pos_change"][:, 2], threshold)


def lift_overshoot(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Box rise since the handoff beyond the band (overshoot only; lowering is not rewarded)."""

    return one_sided_excess(_controller(env).measure()["lift_change"], threshold)


def slip_velocity_excess(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Vertical Box velocity relative to the wrists beyond the band."""

    return excess(_controller(env).measure()["rel_vel"][:, 2], threshold)


def friction_ratio_excess(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Worst hand's |vertical| / horizontal Box force beyond the band (simulator contact forces)."""

    return one_sided_excess(_controller(env).measure()["friction_ratio"].amax(dim=-1), threshold)


def velocity_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """BC3's carry command tracking, so stopping is never a way to keep the grip."""

    controller = _controller(env)
    command = controller.navigation_velocity()[:, :2]
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return torch.exp(-(command - velocity).square().sum(dim=-1) / std**2) * _active(env)


def residual_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    return (controller.applied / controller.residual_cfg.max_residual_rad).square().mean(dim=-1)


def residual_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    controller = _controller(env)
    change = controller.applied - controller.previous_applied
    return (change / controller.residual_cfg.max_rate_rad).square().mean(dim=-1)


def command_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Raw actor-output change, only while the residual has authority."""

    controller = _controller(env)
    live = residual_envelope(controller.elapsed - 1, controller.residual_cfg.schedule) > 0
    change = (controller.command - controller.previous_command).square().mean(dim=-1) / 4.0
    return change * live.float() * _active(env)
