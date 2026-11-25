"""Grip-aware residual over the direct BC2 -> BC3 handoff: pure primitives.

The nominal command is exactly the fixed-blend baseline of the live chain
(``evaluate_bc2_rollout.py --wbc_bridge_seconds 0 --carry_blend_steps 50``):

    a_base = a_anchor + w(k) * (a_bc3 - a_anchor),  locked arm/hand joints = a_anchor

with ``a_anchor`` BC2's last raw action and ``w`` the quintic blend.  A PPO
actor adds a bounded joint-target correction on the 15 leg/waist joints only:

    q_target = q_base + beta(k) * r_k,
    r_k = r_{k-1} + clip(delta_max * u_k - r_{k-1}, +-rate_max),   |r_k| <= delta_max

``beta`` ramps in over the first steps, holds through the blend and the first
second of BC3 walking, and fades to exactly zero (quintic) by the end of the
active window, after which the command is BC3's own.  Together with the rate
limit this bounds the per-step target change the residual can add, so it can
never introduce a new target discontinuity at the blend end or the fade end.

Every function here is shared by training, the live evaluation chain and the
tests, and has no Isaac Lab dependency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import torch

from unitree_rl_lab.utils.teacher.transition_gate import compose_transition_action, quintic_blend


GRIP_RESIDUAL_VERSION = "grip_residual_v1"
# Leg and waist joints (12 + 3); arms and fingers stay on the grasp lock.
RESIDUAL_JOINT_TOKENS = ("hip", "knee", "ankle", "waist")
RESIDUAL_JOINT_COUNT = 15
VARIANTS = ("body", "grip")


@dataclass(frozen=True)
class ResidualScheduleCfg:
    """Carry-step schedule (50 Hz).  Defaults: 1 s blend, residual active 0-3 s."""

    step_dt: float = 0.02
    blend_steps: int = 50
    fade_in_steps: int = 10
    full_until_steps: int = 100
    fade_end_steps: int = 150

    def validate(self) -> "ResidualScheduleCfg":
        if not math.isfinite(self.step_dt) or self.step_dt <= 0:
            raise ValueError("step_dt must be positive")
        if self.blend_steps < 1:
            raise ValueError("the residual is defined over a fixed blend of at least one step")
        if not 0 <= self.fade_in_steps <= self.full_until_steps < self.fade_end_steps:
            raise ValueError("need 0 <= fade_in_steps <= full_until_steps < fade_end_steps")
        return self


@dataclass(frozen=True)
class GripResidualCfg:
    variant: str = "grip"
    max_residual_rad: float = 0.05
    # Per-step rate limit of the residual itself (0.005 rad/step = 0.25 rad/s).
    max_rate_rad: float = 0.005
    schedule: ResidualScheduleCfg = field(default_factory=ResidualScheduleCfg)
    # Deployable Box estimate: simulator truth delayed and corrupted for the actor only.
    box_obs_noise_m: float = 0.005
    box_obs_delay_steps: int = 2
    # Velocity estimate = finite difference over this many steps of the estimate.
    box_velocity_window_steps: int = 5

    def validate(self) -> "GripResidualCfg":
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not 0.0 < self.max_residual_rad <= 0.1:
            raise ValueError("max_residual_rad must be in (0, 0.1]")
        if not 0.0 < self.max_rate_rad <= self.max_residual_rad:
            raise ValueError("max_rate_rad must be in (0, max_residual_rad]")
        if not math.isfinite(self.box_obs_noise_m) or self.box_obs_noise_m < 0:
            raise ValueError("box_obs_noise_m must be finite and nonnegative")
        if self.box_obs_delay_steps < 0 or self.box_velocity_window_steps < 1:
            raise ValueError("delay must be >= 0 and the velocity window >= 1")
        self.schedule.validate()
        return self

    @property
    def action_dim(self) -> int:
        return RESIDUAL_JOINT_COUNT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "GripResidualCfg":
        data = dict(data)
        schedule = ResidualScheduleCfg(**data.pop("schedule", {}))
        return cls(schedule=schedule, **data).validate()


def residual_joint_indices(joint_names: Sequence[str]) -> list[int]:
    ids = [i for i, name in enumerate(joint_names) if any(tok in name for tok in RESIDUAL_JOINT_TOKENS)]
    if len(ids) != RESIDUAL_JOINT_COUNT:
        raise ValueError(f"expected 12 leg + 3 waist joints, got {len(ids)}")
    if any("hand" in joint_names[i] or "shoulder" in joint_names[i] or "elbow" in joint_names[i] for i in ids):
        raise ValueError("residual joints must not include arm or hand joints")
    return ids


def blend_weight(carry_step: torch.Tensor, cfg: ResidualScheduleCfg) -> torch.Tensor:
    """Vectorized ``transition_gate.fixed_blend_weight``: quintic((k + 1) / blend_steps), 1 afterwards."""

    k = carry_step.to(torch.float32)
    return quintic_blend((k + 1.0) / float(cfg.blend_steps))


def residual_envelope(carry_step: torch.Tensor, cfg: ResidualScheduleCfg) -> torch.Tensor:
    """beta(k): quintic ramp-in, hold, quintic fade to exactly 0 at ``fade_end_steps``."""

    k = carry_step.to(torch.float32) + 1.0
    if cfg.fade_in_steps > 0:
        ramp_in = quintic_blend(k / float(cfg.fade_in_steps))
    else:
        ramp_in = torch.ones_like(k)
    fade = 1.0 - quintic_blend((k - cfg.full_until_steps) / float(cfg.fade_end_steps - cfg.full_until_steps))
    beta = ramp_in * fade
    return torch.where(carry_step < 0, torch.zeros_like(beta), beta)


def rate_limited_residual(
    previous: torch.Tensor, command: torch.Tensor, cfg: GripResidualCfg
) -> torch.Tensor:
    """Move the residual (rad) toward ``max_residual_rad * command`` by at most ``max_rate_rad``."""

    goal = cfg.max_residual_rad * command.clamp(-1.0, 1.0)
    step = (goal - previous).clamp(-cfg.max_rate_rad, cfg.max_rate_rad)
    return (previous + step).clamp(-cfg.max_residual_rad, cfg.max_residual_rad)


def residual_raw_offset(
    applied_rad: torch.Tensor, residual_ids: Sequence[int], scale: torch.Tensor, action_dim: int = 43
) -> torch.Tensor:
    """(N, 15) joint-target correction in rad -> (N, A) raw action offset (zero off the residual joints)."""

    offset = torch.zeros(applied_rad.shape[0], action_dim, dtype=applied_rad.dtype, device=applied_rad.device)
    offset[:, list(residual_ids)] = applied_rad / scale[list(residual_ids)]
    return offset


def compose_residual_action(
    anchor: torch.Tensor,
    bc3: torch.Tensor,
    weight: torch.Tensor,
    *,
    locked_indices: Sequence[int],
    raw_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-blend baseline (grasp lock included) plus the residual raw offset.

    ``raw_offset`` must be zero on the locked joints; it is added after the lock,
    exactly as the live chain adds it after its own blend and lock.
    """

    action = compose_transition_action(
        anchor, bc3, weight, locked_indices=locked_indices, locked_reference=anchor
    )
    if raw_offset is not None:
        if raw_offset.shape != action.shape:
            raise ValueError("raw_offset must have the action shape")
        if bool((raw_offset[:, list(locked_indices)] != 0).any()):
            raise ValueError("the residual must not touch locked arm/hand joints")
        action = action + raw_offset
    return action


# ---------------------------------------------------------------- reward shaping
def excess(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft-threshold penalty in threshold units: max(|x| - t, 0) / t (0 inside the normal band)."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return (value.abs() - threshold).clamp_min(0.0) / threshold


def one_sided_excess(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """Like :func:`excess` but only for positive deviations (e.g. lift overshoot)."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return (value - threshold).clamp_min(0.0) / threshold


def tilt_from_projected_gravity(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    return torch.acos((-projected_gravity_b[..., 2]).clamp(-1.0, 1.0))


# ---------------------------------------------------------------- observation layout
# Actor components: name -> width.  Every entry is available on the robot
# (IMU, encoders, the controller's own state) or is a *Box estimate* (marked),
# which the controller derives from delayed, noisy simulator truth.
BODY_ACTOR_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("projected_gravity_b", 3),
    ("base_ang_vel_b", 3),
    ("blend_progress", 3),          # blend weight, carry time / window, envelope beta
    ("velocity_command", 3),
    ("joint_pos_rel", 15),
    ("joint_vel", 15),
    ("bc3_target_minus_joint_pos", 15),
    ("previous_residual", 15),
)
BOX_ESTIMATE_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("box_wrist_rel_pos_change", 3),   # estimate
    ("box_wrist_rel_vel", 3),          # estimate
    ("box_lift_change", 1),            # estimate
)
# Critic-only, simulator truth.  Never part of an actor layout.
PRIVILEGED_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("box_wrist_rel_pos_change_true", 3),
    ("box_wrist_rel_vel_true", 3),
    ("box_lift_change_true", 1),
    ("hand_box_force_b", 6),
    ("hand_friction_ratio", 2),
    ("box_tilt", 1),
    ("box_ang_vel_b", 3),
    ("base_lin_vel_b", 3),
    ("foot_loads", 2),
    ("box_contact_counts", 2),
)


def actor_components(variant: str) -> tuple[tuple[str, int], ...]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return BODY_ACTOR_COMPONENTS + (BOX_ESTIMATE_COMPONENTS if variant == "grip" else ())


def critic_components(variant: str) -> tuple[tuple[str, int], ...]:
    return actor_components(variant) + PRIVILEGED_COMPONENTS


def layout_dim(components: Sequence[tuple[str, int]]) -> int:
    return sum(width for _, width in components)


def assemble(components: Sequence[tuple[str, int]], values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate named (N, width) features in layout order, checking every width."""

    parts = []
    for name, width in components:
        value = values[name]
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"feature {name} has shape {tuple(value.shape)}, expected (N, {width})")
        parts.append(value)
    return torch.cat(parts, dim=-1)


# Fixed observation scales (checked against the zero-residual baseline statistics).
OBS_SCALES: Mapping[str, float] = {
    "projected_gravity_b": 5.0,
    "base_ang_vel_b": 0.25,
    "velocity_command": 2.0,
    "joint_pos_rel": 1.0,
    "joint_vel": 0.05,
    "bc3_target_minus_joint_pos": 2.0,
    "box_wrist_rel_pos_change": 20.0,
    "box_wrist_rel_vel": 3.0,
    "box_lift_change": 10.0,
    "box_wrist_rel_pos_change_true": 20.0,
    "box_wrist_rel_vel_true": 3.0,
    "box_lift_change_true": 10.0,
    "hand_box_force_b": 0.1,
    "box_ang_vel_b": 0.1,
    "base_lin_vel_b": 2.0,
}
OBS_CLIP = 5.0
