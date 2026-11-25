"""WBC-to-CARRY policy transition: fixed baseline and learned gate primitives.

At CARRY start the frozen WBC anchor (the last raw action the WBC actually
sent) is blended into the live BC3 output:

    a = (1 - alpha) * a_anchor + alpha * a_bc3 + (1 - alpha) * com_offset

The fixed baseline drives ``alpha`` with a quintic schedule.  The learned
transition policy instead chooses the per-step increase of ``alpha`` (and, in
stage 2, a CoM shift realized through leg/waist joint offsets).  ``alpha`` is
monotone and forced to complete before ``max_seconds``, so a learned gate can
never oscillate between controllers or stall the CARRY phase.

Every function is shared by training, the live evaluation chain, and tests, and
has no Isaac Lab dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


TRANSITION_GATE_VERSION = "monotonic_rate_gate_v1"


def quintic_blend(fraction: torch.Tensor | float) -> torch.Tensor | float:
    """Minimum-jerk ``10f^3 - 15f^4 + 6f^5`` on a fraction clamped to [0, 1]."""

    if isinstance(fraction, torch.Tensor):
        f = fraction.clamp(0.0, 1.0)
    else:
        f = min(1.0, max(0.0, float(fraction)))
    return f ** 3 * (10.0 - 15.0 * f + 6.0 * f ** 2)


def fixed_blend_weight(carry_step: int, blend_steps: int) -> float:
    """Baseline BC3 weight at a CARRY step; ``blend_steps == 0`` is an instant switch."""

    if carry_step < 0 or blend_steps < 0:
        raise ValueError("carry_step and blend_steps must be nonnegative")
    if carry_step >= blend_steps:
        return 1.0
    return float(quintic_blend((carry_step + 1) / float(blend_steps)))


@dataclass(frozen=True)
class TransitionGateCfg:
    step_dt: float = 0.02
    # Fastest possible full handoff when the policy requests the maximum rate.
    min_seconds: float = 0.25
    # alpha is forced to 1.0 by this time ...
    max_seconds: float = 2.0
    # ... by a linear floor that starts rising this long before max_seconds.
    forced_ramp_seconds: float = 0.5
    # Stage-2 CoM shift bound in the pelvis xy plane.
    max_com_shift_m: float = 0.03

    def validate(self) -> "TransitionGateCfg":
        values = (
            self.step_dt, self.min_seconds, self.max_seconds,
            self.forced_ramp_seconds, self.max_com_shift_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("transition gate durations and bounds must be finite and positive")
        if not self.step_dt <= self.min_seconds <= self.max_seconds:
            raise ValueError("transition gate requires step_dt <= min_seconds <= max_seconds")
        if self.forced_ramp_seconds > self.max_seconds:
            raise ValueError("forced_ramp_seconds cannot exceed max_seconds")
        return self

    @property
    def max_alpha_rate(self) -> float:
        return self.step_dt / self.min_seconds

    @property
    def max_steps(self) -> int:
        return int(round(self.max_seconds / self.step_dt))

    @property
    def forced_ramp_steps(self) -> int:
        return max(1, int(round(self.forced_ramp_seconds / self.step_dt)))


def forced_alpha_floor(elapsed_steps: torch.Tensor, cfg: TransitionGateCfg) -> torch.Tensor:
    """Lower bound on alpha after ``elapsed_steps + 1`` gate updates; 1.0 at ``max_steps``."""

    start = cfg.max_steps - cfg.forced_ramp_steps
    return ((elapsed_steps.float() + 1.0 - start) / cfg.forced_ramp_steps).clamp(0.0, 1.0)


def update_alpha(
    alpha: torch.Tensor,
    gate_action: torch.Tensor,
    elapsed_steps: torch.Tensor,
    cfg: TransitionGateCfg,
) -> torch.Tensor:
    """Advance alpha by ``max_rate * (clip(u, -1, 1) + 1) / 2`` with the forced floor.

    ``u = -1`` holds alpha, ``u = +1`` advances at the maximum rate.  The result
    is monotone, lies in [0, 1], and equals 1 once ``elapsed_steps + 1 >= max_steps``.
    """

    if alpha.ndim != 1 or gate_action.shape != alpha.shape or elapsed_steps.shape != alpha.shape:
        raise ValueError("alpha, gate_action, and elapsed_steps must have shape (N,)")
    if not torch.isfinite(gate_action).all():
        raise ValueError("gate action must be finite")
    rate = cfg.max_alpha_rate * 0.5 * (gate_action.clamp(-1.0, 1.0) + 1.0)
    proposed = torch.maximum(alpha + rate, forced_alpha_floor(elapsed_steps, cfg))
    return proposed.clamp(0.0, 1.0)


def compose_transition_action(
    anchor: torch.Tensor,
    bc3: torch.Tensor,
    alpha: torch.Tensor,
    *,
    locked_indices: Sequence[int],
    locked_reference: torch.Tensor,
    com_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend raw actions, add the ``(1 - alpha)``-faded CoM offset, then apply the grasp lock.

    All tensors are in raw ``JointPositionAction`` coordinates with shape (N, A);
    ``com_offset`` must already be divided by the action scale.
    """

    if anchor.ndim != 2 or bc3.shape != anchor.shape or locked_reference.shape != anchor.shape:
        raise ValueError("anchor, bc3, and locked_reference must share an (N, A) shape")
    if alpha.shape != (anchor.shape[0],):
        raise ValueError("alpha must have shape (N,)")
    weight = alpha.to(anchor.dtype)[:, None]
    action = anchor + weight * (bc3 - anchor)
    if com_offset is not None:
        if com_offset.shape != anchor.shape:
            raise ValueError("com_offset must have the action shape")
        action = action + (1.0 - weight) * com_offset
    locked = list(locked_indices)
    action[:, locked] = locked_reference[:, locked]
    return action


def com_jacobian(body_linear_jacobians: torch.Tensor, body_masses: torch.Tensor) -> torch.Tensor:
    """Whole-body CoM Jacobian ``sum_b m_b J_b / M`` from (N, B, 3, J) body Jacobians."""

    if body_linear_jacobians.ndim != 4 or body_linear_jacobians.shape[2] != 3:
        raise ValueError("body Jacobians must have shape (N, B, 3, J)")
    if body_masses.shape != body_linear_jacobians.shape[:2]:
        raise ValueError("body masses must have shape (N, B)")
    masses = body_masses.to(body_linear_jacobians.dtype)
    total = masses.sum(dim=1).clamp_min(1.0e-6)
    weighted = (masses[:, :, None, None] * body_linear_jacobians).sum(dim=1)
    return weighted / total[:, None, None]


def com_shift_joint_offset(
    com_jacobian_xy: torch.Tensor,
    com_shift_xy: torch.Tensor,
    *,
    max_shift_m: float,
    damping: float = 1.0e-2,
) -> torch.Tensor:
    """Damped least-squares joint offset ``J^T (J J^T + lambda^2 I)^-1 dc`` for a clipped shift."""

    if com_jacobian_xy.ndim != 3 or com_jacobian_xy.shape[1] != 2:
        raise ValueError("CoM xy Jacobian must have shape (N, 2, J)")
    if com_shift_xy.shape != (com_jacobian_xy.shape[0], 2):
        raise ValueError("CoM shift must have shape (N, 2)")
    if max_shift_m <= 0.0 or damping <= 0.0:
        raise ValueError("max_shift_m and damping must be positive")
    shift = com_shift_xy.clamp(-max_shift_m, max_shift_m).to(com_jacobian_xy.dtype)
    jjt = com_jacobian_xy @ com_jacobian_xy.transpose(1, 2)
    regularizer = (damping ** 2) * torch.eye(2, dtype=jjt.dtype, device=jjt.device)
    solved = torch.linalg.solve(jjt + regularizer, shift.unsqueeze(-1))
    return (com_jacobian_xy.transpose(1, 2) @ solved).squeeze(-1)
