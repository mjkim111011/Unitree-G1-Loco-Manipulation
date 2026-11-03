"""CoM-predictive *schedule* selection for the WBC-to-BC3 handoff (v2), not a stability guarantee.

v1 (:mod:`com_predictive_blend`) scored one-step alpha increments over a 0.1 s
horizon.  A single step changes the joint targets by ~0.001-0.03 rad, so its
capture-point predictions were identical across candidates and it always took
the largest increment, i.e. it behaved like a fixed 0.26 s blend.

v2 compares whole remaining schedules (hold, keep the committed schedule, or
finish with a new quintic in 0.26/0.5/1.0 s) over a common 0.5 s horizon.  It
is anchored to the fixed 1 s baseline: it follows the preferred schedule (the
committed one, initially a 1 s quintic) and only deviates when that schedule's
predicted margin falls below ``keep_margin_m``.  The prediction model:

* joints follow *changes* of their blended targets, relative to the target
  applied now, with a first-order lag ``tau`` (under gravity load the PD has a
  steady-state error of up to ~0.13 rad at the ankles, so ``q -> target`` would
  predict a large spurious CoM shift even for a hold);
* the resulting joint motion moves the CoM relative to the loaded feet through
  the payload-augmented CoM Jacobian (rigid-grasp approximation);
* the current CoM velocity diverges as a linear inverted pendulum whose ZMP
  stays at the current CoM projection (the WBC stand is statically balanced);
* the support polygon of the currently loaded feet is held fixed, and BC3's
  current target is held constant over the horizon.

Contact changes, impacts and full rigid-body dynamics are not predicted, which
is why every decision logs predictions for an offline error check.  Only alpha
(and, optionally, the start of BC3's walking-command ramp) is changed: there is
no posture correction and no retraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .com_predictive_blend import convex_hull, signed_support_margin
from .transition_gate import quintic_blend


HOLD = -1.0
KEEP = 0.0


@dataclass(frozen=True)
class ComScheduleCfg:
    step_dt: float = 0.02
    horizon_seconds: float = 0.5
    # First-order joint tracking time constant of the PD-actuated legs.
    tracking_tau_seconds: float = 0.06
    new_schedule_seconds: tuple[float, ...] = (0.26, 0.5, 1.0)
    support_margin_m: float = 0.005
    # Schedule followed until the prediction flags risk (the fixed baseline).
    default_schedule_seconds: float = 1.0
    # Keep the preferred schedule while its worst predicted margin is above this.
    keep_margin_m: float = 0.02
    # Otherwise candidates within this of the best margin are treated as equal.
    margin_tie_m: float = 0.005
    foot_half_length_m: float = 0.10
    foot_half_width_m: float = 0.04
    minimum_foot_load_fraction: float = 0.05
    command_jump_limit_rad: float = 0.10
    max_seconds: float = 2.0
    forced_ramp_seconds: float = 0.5
    # Optional walking-command gate (method variant "+cmd").
    gate_inner_margin_m: float = 0.02
    gate_max_com_speed_mps: float = 0.10
    gate_max_delay_seconds: float = 1.0

    def validate(self) -> "ComScheduleCfg":
        positive = (self.step_dt, self.horizon_seconds, self.tracking_tau_seconds, self.foot_half_length_m,
                    self.foot_half_width_m, self.minimum_foot_load_fraction, self.command_jump_limit_rad,
                    self.max_seconds, self.forced_ramp_seconds, self.gate_max_com_speed_mps,
                    self.gate_max_delay_seconds)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("CoM schedule parameters must be finite and positive")
        if not self.new_schedule_seconds or any(d < self.step_dt for d in self.new_schedule_seconds):
            raise ValueError("schedule durations must be at least one control step")
        if self.forced_ramp_seconds > self.max_seconds or self.horizon_seconds > self.max_seconds:
            raise ValueError("invalid deadline or horizon")
        if not 0 <= self.support_margin_m < min(self.foot_half_length_m, self.foot_half_width_m):
            raise ValueError("invalid support margin")
        if self.margin_tie_m < 0 or self.gate_inner_margin_m < 0 or self.keep_margin_m < self.support_margin_m:
            raise ValueError("tie/gate margins must be nonnegative and keep_margin >= support_margin")
        if self.default_schedule_seconds not in self.new_schedule_seconds:
            raise ValueError("the default schedule must be one of the new schedule durations")
        return self

    @property
    def horizon_steps(self) -> int:
        return max(1, int(round(self.horizon_seconds / self.step_dt)))

    @property
    def max_steps(self) -> int:
        return int(round(self.max_seconds / self.step_dt))


@dataclass(frozen=True)
class Schedule:
    """A quintic from ``start_alpha`` at ``start_step`` to 1 at ``finish_step`` (or a hold)."""

    kind: float  # HOLD, or the duration in seconds of a quintic schedule
    start_alpha: float
    start_step: int
    finish_step: int

    def alpha(self, step: int) -> float:
        if self.kind == HOLD:
            return self.start_alpha
        span = max(1, self.finish_step - self.start_step)
        fraction = (step + 1 - self.start_step) / span
        return self.start_alpha + (1.0 - self.start_alpha) * float(quintic_blend(fraction))


def forced_floor(step: int, cfg: ComScheduleCfg) -> float:
    elapsed = (step + 1) * cfg.step_dt
    return max(0.0, min(1.0, (elapsed - cfg.max_seconds + cfg.forced_ramp_seconds) / cfg.forced_ramp_seconds))


def candidate_schedules(alpha: float, step: int, committed: Schedule | None, cfg: ComScheduleCfg):
    """Hold, the committed schedule (if any), and new quintics clipped to the deadline."""

    schedules = [Schedule(HOLD, alpha, step, step)]
    if committed is not None and committed.kind != HOLD:
        schedules.append(committed)
    for duration in cfg.new_schedule_seconds:
        finish = min(step + int(round(duration / cfg.step_dt)), cfg.max_steps)
        schedules.append(Schedule(duration, alpha, step, max(finish, step + 1)))
    return schedules


def schedule_alpha_trajectories(schedules, step: int, cfg: ComScheduleCfg, device=None) -> torch.Tensor:
    """(C, H) alpha over the horizon, monotone and never below the deadline floor."""

    rows = []
    for schedule in schedules:
        row, previous = [], schedule.start_alpha
        for k in range(cfg.horizon_steps):
            value = max(schedule.alpha(step + k), forced_floor(step + k, cfg), previous)
            row.append(min(1.0, value))
            previous = row[-1]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def predict_capture_points(
    alpha_trajectories: torch.Tensor,
    anchor_targets: torch.Tensor,
    bc3_targets: torch.Tensor,
    applied_targets: torch.Tensor,
    relative_com_jacobian_xy: torch.Tensor,
    com_xy: torch.Tensor,
    com_velocity_xy: torch.Tensor,
    omega: float,
    cfg: ComScheduleCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted capture points (C, H, 2) and CoM positions (C, H, 2) relative to the support origin.

    ``anchor_targets``/``bc3_targets``/``applied_targets`` are absolute joint
    targets of the controlled joints (J,), the last being the target currently
    commanded; the Jacobian is (2, J).
    """

    dt = cfg.step_dt
    decay = math.exp(-dt / cfg.tracking_tau_seconds)
    targets = anchor_targets + alpha_trajectories[..., None] * (bc3_targets - anchor_targets)  # (C, H, J)
    change = targets - applied_targets  # target change relative to now
    motion = torch.zeros_like(change[:, 0])
    shifts = []
    for k in range(targets.shape[1]):
        motion = change[:, k] + (motion - change[:, k]) * decay
        shifts.append(motion @ relative_com_jacobian_xy.T)
    shift = torch.stack(shifts, dim=1)  # (C, H, 2)
    times = dt * torch.arange(1, targets.shape[1] + 1, device=targets.device, dtype=targets.dtype)
    free_position = com_xy + com_velocity_xy * (torch.sinh(omega * times) / omega)[:, None]
    free_velocity = com_velocity_xy * torch.cosh(omega * times)[:, None]
    shift_velocity = torch.diff(shift, dim=1, prepend=torch.zeros_like(shift[:, :1])) / dt
    position = free_position[None] + shift
    capture = position + (free_velocity[None] + shift_velocity) / omega
    return capture, position


def choose_schedule(
    schedules,
    capture: torch.Tensor,
    polygon: torch.Tensor,
    first_step_jumps: torch.Tensor,
    bilateral_contact: bool,
    preferred_index: int,
    cfg: ComScheduleCfg,
):
    """Keep the preferred schedule unless its worst predicted margin over the horizon is too small.

    Otherwise pick the largest-margin schedule (ties: preferred, then faster).

    Returns ``(index, min_margins, feasible, fallback_reasons)``.  When every
    candidate is infeasible the least risky one is a logged best-effort
    fallback, not a safe-set proof.
    """

    if not torch.isfinite(capture).all() or not torch.isfinite(first_step_jumps).all():
        raise ValueError("nonfinite schedule predictions")
    count, horizon = capture.shape[:2]
    margins = signed_support_margin(capture.reshape(-1, 2), polygon).reshape(count, horizon)
    min_margins = margins.amin(dim=1)
    margin_ok = min_margins >= cfg.support_margin_m
    jump_ok = first_step_jumps <= cfg.command_jump_limit_rad
    feasible = margin_ok & jump_ok & bool(bilateral_contact)
    reasons = []
    if not bilateral_contact:
        reasons.append("hand_contact_lost")
    if polygon.shape[0] < 3:
        reasons.append("no_support_polygon")
    if not bool(margin_ok.any()):
        reasons.append("capture_margin")
    if not bool(jump_ok.any()):
        reasons.append("command_jump")
    if not bool(feasible.any()) and not reasons:
        reasons.append("margin_and_jump_disjoint")

    if not bilateral_contact:
        # Do not advance toward BC3 while the payload grasp is broken.
        return 0, min_margins, feasible, reasons
    pool = torch.nonzero(feasible).flatten() if bool(feasible.any()) else torch.nonzero(jump_ok).flatten()
    if pool.numel() == 0:
        pool = torch.arange(count, device=capture.device)
    if bool(feasible[preferred_index]) and float(min_margins[preferred_index]) >= cfg.keep_margin_m:
        return preferred_index, min_margins, feasible, reasons
    best = float(min_margins[pool].max())
    tied = [int(i) for i in pool if float(min_margins[i]) >= best - cfg.margin_tie_m]
    if preferred_index in tied:
        return preferred_index, min_margins, feasible, reasons

    def speed(i: int) -> float:
        s = schedules[i]
        return -1.0 if s.kind == HOLD else 1.0 / max(1, s.finish_step - s.start_step)

    return max(tied, key=speed), min_margins, feasible, reasons


def walking_gate_open(alpha: float, capture_margin: float, com_speed: float, steps_since_takeover: int,
                      cfg: ComScheduleCfg) -> bool:
    """Start BC3's walking-command ramp only once it owns the robot over a centered, slow CoM."""

    if alpha < 1.0:
        return False
    if steps_since_takeover * cfg.step_dt >= cfg.gate_max_delay_seconds:
        return True
    return capture_margin >= cfg.gate_inner_margin_m and com_speed <= cfg.gate_max_com_speed_mps


class ComScheduleBlender:
    """Single-env runtime adapter; reads simulation state but does not mutate it."""

    def __init__(self, env, anchor, locked_indices, cfg: ComScheduleCfg | None = None):
        if env.num_envs != 1:
            raise ValueError("CoM schedule evaluation currently supports one environment")
        self.cfg = (cfg or ComScheduleCfg(step_dt=env.step_dt)).validate()
        self.env, self.robot, self.box = env, env.scene["robot"], env.scene["box"]
        self.anchor = anchor.clone()
        self.locked = list(locked_indices)
        self.alpha = 0.0
        self.committed: Schedule | None = None
        term = env.action_manager.get_term("JointPositionAction")
        self.scale = torch.as_tensor(term._scale, device=env.device).expand(43)
        self.offset = torch.as_tensor(term._offset, device=env.device).expand_as(anchor)[0]
        joint_ids = term._joint_ids
        joint_ids = list(range(self.robot.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        self.unlocked = [i for i in range(43) if i not in set(self.locked)]
        self.control_joints = [joint_ids[i] for i in self.unlocked]
        self.columns = [6 + i for i in self.control_joints]
        self.masses = self.robot.root_physx_view.get_masses().to(env.device)[0]
        self.payload_mass = self.box.root_physx_view.get_masses().to(env.device)[0].sum()
        self.total_mass = self.masses.sum() + self.payload_mass
        self.feet = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
        self.wrists = self.robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True)[0]
        self.sensor = env.scene.sensors["contact_forces"]
        self.sensor_feet = self.sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"],
                                                   preserve_order=True)[0]

    def measure(self) -> dict:
        """Payload-augmented CoM state relative to the loaded-feet support, in world-aligned xy."""

        cfg, robot = self.cfg, self.robot
        com = ((self.masses[:, None] * robot.data.body_com_pos_w[0]).sum(0)
               + self.payload_mass * self.box.data.root_com_pos_w[0]) / self.total_mass
        velocity = ((self.masses[:, None] * robot.data.body_com_lin_vel_w[0]).sum(0)
                    + self.payload_mass * self.box.data.root_com_lin_vel_w[0]) / self.total_mass
        loads = self.sensor.data.net_forces_w[0, self.sensor_feet, 2].clamp_min(0)
        loaded = loads >= cfg.minimum_foot_load_fraction * self.total_mass * 9.81
        weights = loads * loaded
        weights = weights / weights.sum().clamp_min(1e-8) if loaded.any() else torch.full_like(weights, 0.5)
        feet_pos = robot.data.body_link_pos_w[0, self.feet]
        origin = (weights[:, None] * feet_pos).sum(0)
        origin_velocity = (weights[:, None] * robot.data.body_link_lin_vel_w[0, self.feet]).sum(0)
        corners = []
        for i in range(2):
            if not bool(loaded[i]):
                continue
            w, x, y, z = robot.data.body_link_quat_w[0, self.feet[i]].detach().cpu().tolist()
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = math.cos(yaw), math.sin(yaw)
            center = (feet_pos[i] - origin)[:2].detach().cpu().tolist()
            for dx in (-cfg.foot_half_length_m, cfg.foot_half_length_m):
                for dy in (-cfg.foot_half_width_m, cfg.foot_half_width_m):
                    corners.append((center[0] + c * dx - s * dy, center[1] + s * dx + c * dy))
        polygon = torch.tensor(convex_hull(corners), device=self.env.device, dtype=torch.float32).reshape(-1, 2)
        relative = (com - origin)[:2]
        relative_velocity = (velocity - origin_velocity)[:2]
        omega = math.sqrt(9.81 / max(0.1, float(com[2] - origin[2])))
        capture = relative + relative_velocity / omega
        return {
            "com_xy": relative, "com_velocity_xy": relative_velocity, "omega": omega, "polygon": polygon,
            "origin_w": origin, "weights": weights, "loaded": loaded,
            "capture_xy": capture,
            "capture_margin": float(signed_support_margin(capture[None], polygon)[0]),
        }

    def _relative_jacobian(self, weights) -> torch.Tensor:
        jac = self.robot.root_physx_view.get_jacobians()[0, :, :3, :][:, :, self.columns].to(self.env.device)
        com_j = ((self.masses[:, None, None] * jac).sum(0) + self.payload_mass * jac[self.wrists].mean(0)) / self.total_mass
        support_j = (weights[:, None, None] * jac[self.feet]).sum(0)
        return (com_j - support_j)[:2]

    def step(self, bc3, carry_step, previous_target, bilateral_contact):
        state = self.measure()
        if self.alpha >= 1.0:
            return bc3, {"alpha": 1.0, "active": False,
                         "measured_capture_w": (state["origin_w"][:2] + state["capture_xy"]).tolist(),
                         "measured_capture_margin_m": state["capture_margin"]}
        cfg = self.cfg
        schedules = candidate_schedules(self.alpha, carry_step, self.committed, cfg)
        committed_index = 1 if self.committed is not None and self.committed.kind != HOLD else None
        preferred_index = committed_index if committed_index is not None else next(
            i for i, sc in enumerate(schedules)
            if sc.kind == cfg.default_schedule_seconds and i != committed_index)
        alphas = schedule_alpha_trajectories(schedules, carry_step, cfg, device=self.env.device)
        anchor_t = (self.anchor[0] * self.scale + self.offset)[self.unlocked]
        bc3_t = (bc3[0] * self.scale + self.offset)[self.unlocked]
        applied_t = previous_target[0, self.unlocked]
        capture, position = predict_capture_points(
            alphas, anchor_t, bc3_t, applied_t, self._relative_jacobian(state["weights"]),
            state["com_xy"], state["com_velocity_xy"], state["omega"], cfg,
        )
        first_targets = anchor_t + alphas[:, :1] * (bc3_t - anchor_t)
        jumps = (first_targets - previous_target[0, self.unlocked]).abs().amax(dim=1)
        index, min_margins, feasible, reasons = choose_schedule(
            schedules, capture, state["polygon"], jumps, bilateral_contact, preferred_index, cfg,
        )
        chosen = schedules[index]
        old_alpha = self.alpha
        self.alpha = float(alphas[index, 0])
        self.committed = chosen if chosen.kind != HOLD else None
        action = self.anchor + self.alpha * (bc3 - self.anchor)
        action[:, self.locked] = self.anchor[:, self.locked]
        origin = state["origin_w"][:2]
        horizon = {"0.1s": 4, "0.3s": 14}
        debug = {
            "alpha": self.alpha, "previous_alpha": old_alpha, "active": True,
            "candidates": [{"kind": "hold" if s.kind == HOLD else f"quintic_{s.kind:g}s",
                            "finish_step": s.finish_step, "committed": i == committed_index,
                            "preferred": i == preferred_index}
                           for i, s in enumerate(schedules)],
            "selected": index, "kept_preferred": index == preferred_index,
            "selected_kind": "hold" if chosen.kind == HOLD else f"quintic_{chosen.kind:g}s",
            "candidate_min_margin_m": min_margins.tolist(),
            "candidate_first_jump_rad": jumps.tolist(),
            "feasible_candidates": int(feasible.sum()),
            "fallback": not bool(feasible.any()), "fallback_reasons": reasons,
            "deadline_floor": forced_floor(carry_step, cfg),
            "deadline_forced": forced_floor(carry_step, cfg) > chosen.alpha(carry_step) + 1e-6,
            "loaded_feet": state["loaded"].tolist(), "bilateral_contact": bilateral_contact,
            "measured_capture_w": (origin + state["capture_xy"]).tolist(),
            "measured_capture_margin_m": state["capture_margin"],
            "measured_com_velocity_xy_mps": state["com_velocity_xy"].tolist(),
            # Offline prediction-error check: prediction k is the state after k + 1
            # actions, i.e. measured_capture_w logged at carry_step + k + 1.
            "predicted_capture_w": {key: {"carry_step": carry_step + k + 1,
                                          "xy": (origin + capture[index, k]).tolist()}
                                    for key, k in horizon.items() if k < capture.shape[1]},
        }
        return action, debug
