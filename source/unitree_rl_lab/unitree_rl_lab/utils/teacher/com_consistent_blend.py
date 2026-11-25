"""CoM-consistent WBC-to-BC3 interpolation (M2), not a stability guarantee.

The fixed baseline blends joint targets with a quintic ``alpha(t)``:

    q_blend(alpha) = (1 - alpha) q_WBC + alpha q_BC3

Smooth joint targets do not make the payload-augmented capture point move
smoothly: mixing two controllers' postures lets it drift while the robot is
still in double support.  M2 keeps the same posture blend and adds a small
leg/waist correction so the capture point follows the straight path between
its value at the handoff and the one BC3's target posture leads to:

    xi_des(alpha)  = (1 - alpha) xi_start + alpha xi_pred(q_BC3)
    xi_pred(q)     = xi_meas + J (q - q_applied)          (first-order prediction)
    e              = xi_des(alpha) - xi_pred(q_blend(alpha))
    q_target       = q_blend + DLS(J) clip(e)              (clipped per joint)

The code evaluates ``e`` in this general form.  Because ``xi_pred`` is linear
in ``q``, it equals ``(1 - alpha) (xi_start - xi_pred(q_WBC))``: the
correction cancels the drift of the anchor posture, weighted by how much of
the anchor is still in the blend.  Straightness therefore holds to first order
around the current state at every step; it is closed through the measured
capture point each step rather than guaranteed for the whole path.

Lead (anticipatory weight transfer).  The 100-state paired evaluation showed
that the capture-point "drift" during the blend is mostly BC3's own lateral
weight shift before its first step (about 2.5 cm), and that ``lead = 1``
opposed it in 97 % of blend steps, making the first step sharper.  With
``lead > 1`` the desired path moves toward BC3's predicted capture point
ahead of the posture blend,

    beta = min(1, lead * alpha),   xi_des = (1 - beta) xi_start + beta xi_pred(q_BC3),

so the correction assists that weight shift instead.  ``lead = 1`` is the
original M2.

Properties:
* ``alpha = 1`` gives ``e = 0`` and BC3's own targets (no lasting change).
* The correction acts only in double support, where the capture-point
  prediction was accurate (0.3 s error 0.6 cm vs 4.9 cm while stepping).  The
  gain is rate-limited, so a foot lifting fades the correction out instead of
  removing it in one step.
* BC2, the WBC, BC3 and the arm/hand grasp lock are untouched.

``J`` is the payload-augmented CoM Jacobian relative to the load-weighted feet
(``ComStabilizer._relative_jacobian_yaw``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .transition_gate import com_shift_joint_offset, fixed_blend_weight


@dataclass(frozen=True)
class ComConsistentBlendCfg:
    step_dt: float = 0.02
    blend_steps: int = 50
    gain: float = 1.0
    # How far the desired capture point runs ahead of the posture blend (1 = original M2).
    lead: float = 1.0
    # Gain outside double support (the prediction is unreliable once a foot lifts).
    single_support_gain_scale: float = 0.0
    # Time for the gain to move across its full range; avoids a target jump at foot lift.
    gain_fade_seconds: float = 0.15
    max_shift_m: float = 0.04
    max_joint_offset_rad: float = 0.15
    damping: float = 1.0e-2

    def validate(self) -> "ComConsistentBlendCfg":
        if self.blend_steps <= 0:
            raise ValueError("blend_steps must be positive")
        positive = (self.step_dt, self.gain_fade_seconds, self.max_shift_m, self.max_joint_offset_rad, self.damping)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("step, fade time, limits and damping must be finite and positive")
        if not math.isfinite(self.gain) or self.gain < 0 or not 0 <= self.single_support_gain_scale <= 1:
            raise ValueError("gain must be nonnegative and the single-support scale in [0, 1]")
        if not math.isfinite(self.lead) or self.lead <= 0:
            raise ValueError("lead must be finite and positive")
        return self


def predicted_capture(capture_xy: torch.Tensor, jacobian_xy: torch.Tensor,
                      targets: torch.Tensor, applied_targets: torch.Tensor) -> torch.Tensor:
    """Capture point after the joints move from the applied targets to ``targets`` (first order)."""

    return capture_xy + jacobian_xy @ (targets - applied_targets)


def target_gain(mode: str | None, cfg: ComConsistentBlendCfg) -> float:
    if mode is None:
        return 0.0
    return cfg.gain * (1.0 if mode == "double" else cfg.single_support_gain_scale)


def faded_gain(current: float, target: float, cfg: ComConsistentBlendCfg) -> float:
    """Move the gain toward ``target`` by at most ``gain * step_dt / gain_fade_seconds`` per step."""

    rate = max(cfg.gain, 1e-12) * cfg.step_dt / cfg.gain_fade_seconds
    return current + max(-rate, min(rate, target - current))


def consistent_correction(
    alpha: float,
    capture_xy: torch.Tensor,
    jacobian_xy: torch.Tensor,
    wbc_targets: torch.Tensor,
    bc3_targets: torch.Tensor,
    applied_targets: torch.Tensor,
    start_capture_xy: torch.Tensor,
    cfg: ComConsistentBlendCfg,
    gain: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Joint offsets (J,) that put the blended posture's predicted capture point on the straight path.

    Targets are absolute joint angles of the controlled joints (J,); capture
    points are (2,) in the yaw frame relative to the support origin.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if jacobian_xy.shape != (2, wbc_targets.shape[0]):
        raise ValueError("jacobian must be (2, J) for J controlled joints")
    gain = cfg.gain if gain is None else gain
    blended = wbc_targets + alpha * (bc3_targets - wbc_targets)
    bc3_capture = predicted_capture(capture_xy, jacobian_xy, bc3_targets, applied_targets)
    beta = min(1.0, cfg.lead * alpha)
    desired = (1.0 - beta) * start_capture_xy + beta * bc3_capture
    blended_capture = predicted_capture(capture_xy, jacobian_xy, blended, applied_targets)
    error = desired - blended_capture
    shift = (gain * error).clamp(-cfg.max_shift_m, cfg.max_shift_m)
    if not bool(torch.any(shift != 0)):
        offsets = torch.zeros_like(wbc_targets)
    else:
        offsets = com_shift_joint_offset(
            jacobian_xy[None], shift[None], max_shift_m=cfg.max_shift_m, damping=cfg.damping,
        )[0].clamp(-cfg.max_joint_offset_rad, cfg.max_joint_offset_rad)
    return offsets, {
        "desired_capture_xy_m": desired.tolist(), "blended_capture_xy_m": blended_capture.tolist(),
        "capture_error_xy_m": error.tolist(), "com_shift_xy_m": shift.tolist(),
        "bc3_capture_xy_m": bc3_capture.tolist(), "beta": beta,
    }


class ComConsistentBlender:
    """Single-env runtime replacing the fixed blend from the WBC anchor into BC3.

    ``step`` takes BC3's raw action and the absolute target applied at the
    previous step, and returns the raw action to apply.  It reads simulation
    state but does not mutate it.
    """

    def __init__(self, env, anchor: torch.Tensor, locked_indices, cfg: ComConsistentBlendCfg | None = None):
        from .com_stabilizer import ComStabilizer

        self.cfg = (cfg or ComConsistentBlendCfg(step_dt=env.step_dt)).validate()
        # Reuse the stabilizer's payload-augmented capture-point and Jacobian model, measuring only.
        self.model = ComStabilizer(env, locked_indices, active=False)
        term = env.action_manager.get_term("JointPositionAction")
        self.scale = torch.as_tensor(term._scale, device=env.device, dtype=torch.float32).expand(43)
        self.offset = torch.as_tensor(term._offset, device=env.device, dtype=torch.float32).expand(1, 43)[0]
        self.anchor = anchor.detach().clone()
        self.locked = list(locked_indices)
        self.ids = self.model.action_ids
        self.start_capture = None
        self.alpha = 0.0
        self.gain = 0.0
        self.previous_offsets = torch.zeros(len(self.ids), device=env.device)

    def _absolute(self, raw: torch.Tensor) -> torch.Tensor:
        return raw[0] * self.scale + self.offset

    def step(self, bc3_raw: torch.Tensor, carry_step: int, previous_target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        self.alpha = fixed_blend_weight(carry_step, self.cfg.blend_steps)
        state = self.model.measure()
        if self.start_capture is None:
            self.start_capture = state["capture_xy"].clone()
            # Start from the full gain if the handoff begins in double support.
            self.gain = target_gain(state["mode"], self.cfg)
        self.gain = faded_gain(self.gain, target_gain(state["mode"], self.cfg), self.cfg)
        action = self.anchor + self.alpha * (bc3_raw - self.anchor)
        action[:, self.locked] = self.anchor[:, self.locked]
        log = {"alpha": self.alpha, "mode": state["mode"], "gain": self.gain,
               "capture_xy_m": state["capture_xy"].tolist(), "start_capture_xy_m": self.start_capture.tolist()}
        offsets = torch.zeros_like(self.previous_offsets)
        if self.alpha < 1.0 and self.gain > 0.0:
            jacobian = self.model._relative_jacobian_yaw(state["weights"], state["rotation"])
            offsets, info = consistent_correction(
                self.alpha, state["capture_xy"], jacobian,
                self._absolute(self.anchor)[self.ids], self._absolute(bc3_raw)[self.ids],
                previous_target[0, self.ids].to(self.scale.dtype), self.start_capture, self.cfg, gain=self.gain,
            )
            action[0, self.ids] += offsets / self.scale[self.ids]
            log.update(info)
        log["max_joint_offset_rad"] = float(offsets.abs().max())
        # Per-step change of the correction: checks continuity at foot lift and at alpha = 1.
        log["offset_step_change_rad"] = float((offsets - self.previous_offsets).abs().max())
        self.previous_offsets = offsets
        return action, log
