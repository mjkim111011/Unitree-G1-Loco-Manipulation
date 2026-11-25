"""Payload-aware CoM stabilizer added on top of a frozen BC3 (no retraining).

BC3 was trained carrying a 0.2 kg Box.  With a heavier, untrained payload the
same joint configuration places the payload-augmented CoM elsewhere relative to
the feet than BC3 ever experienced.  This layer keeps the capture point
``xi = c + v / omega`` (payload-augmented CoM, relative to the loaded-feet
support origin, yaw-aligned) close to the reference BC3 had at the nominal
payload, separately for double support and each single-support stance:

    dc = clip(-K * (xi - xi_ref[mode]), +-max_shift)          (low-pass filtered)
    dq = J_rel^T (J_rel J_rel^T + lambda^2 I)^-1 dc           (leg + waist joints)

``J_rel`` is the payload-augmented CoM Jacobian minus the load-weighted foot
Jacobian.  The gain is scheduled by contact mode (full in double support,
reduced on a single stance foot so the layer does not fight BC3's gait) and is
ramped in at the handoff.  ``dq`` is added to BC3's (blended) joint targets.
State comes from simulation ground truth, including the payload mass.

The pipeline BC2 -> WBC -> BC3 itself is unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .transition_gate import com_shift_joint_offset


MODES = ("double", "left", "right")
REFERENCE_SCHEMA = "unitree_box_com_stabilizer_reference_v1"


@dataclass(frozen=True)
class ComStabilizerCfg:
    step_dt: float = 0.02
    gain: float = 1.0
    single_support_gain_scale: float = 0.3
    max_shift_m: float = 0.04
    max_joint_offset_rad: float = 0.15
    damping: float = 1.0e-2
    filter_seconds: float = 0.06
    ramp_in_seconds: float = 0.5
    minimum_foot_load_fraction: float = 0.05
    # Reference capture point (yaw frame, relative to the support origin) per mode.
    reference_xy: dict = field(default_factory=lambda: {mode: (0.0, 0.0) for mode in MODES})

    def validate(self) -> "ComStabilizerCfg":
        positive = (self.step_dt, self.max_shift_m, self.max_joint_offset_rad, self.damping,
                    self.filter_seconds, self.ramp_in_seconds, self.minimum_foot_load_fraction)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("stabilizer limits and time constants must be finite and positive")
        if not math.isfinite(self.gain) or self.gain < 0 or not 0 <= self.single_support_gain_scale <= 1:
            raise ValueError("gain must be nonnegative and the single-support scale in [0, 1]")
        if set(self.reference_xy) != set(MODES) or any(
            len(v) != 2 or not all(math.isfinite(x) for x in v) for v in self.reference_xy.values()
        ):
            raise ValueError("reference_xy needs a finite (x, y) for double/left/right")
        return self


def contact_mode(loaded_left: bool, loaded_right: bool) -> str | None:
    if loaded_left and loaded_right:
        return "double"
    if loaded_left:
        return "left"
    if loaded_right:
        return "right"
    return None  # flight / no support: do not act


def mode_gain(mode: str | None, steps_since_start: int, cfg: ComStabilizerCfg) -> float:
    if mode is None:
        return 0.0
    ramp = min(1.0, (steps_since_start + 1) * cfg.step_dt / cfg.ramp_in_seconds)
    scale = 1.0 if mode == "double" else cfg.single_support_gain_scale
    return cfg.gain * scale * ramp


def desired_com_shift(capture_xy: torch.Tensor, reference_xy, gain: float, cfg: ComStabilizerCfg) -> torch.Tensor:
    error = capture_xy - torch.as_tensor(reference_xy, dtype=capture_xy.dtype, device=capture_xy.device)
    return (-gain * error).clamp(-cfg.max_shift_m, cfg.max_shift_m)


def filter_shift(previous: torch.Tensor, target: torch.Tensor, cfg: ComStabilizerCfg) -> torch.Tensor:
    blend = 1.0 - math.exp(-cfg.step_dt / cfg.filter_seconds)
    return previous + blend * (target - previous)


def joint_offsets(relative_jacobian_xy: torch.Tensor, shift_xy: torch.Tensor, cfg: ComStabilizerCfg) -> torch.Tensor:
    """(J,) joint offsets realizing ``shift_xy`` through the (2, J) relative CoM Jacobian."""

    offsets = com_shift_joint_offset(
        relative_jacobian_xy[None], shift_xy[None], max_shift_m=cfg.max_shift_m, damping=cfg.damping,
    )[0]
    return offsets.clamp(-cfg.max_joint_offset_rad, cfg.max_joint_offset_rad)


def build_reference(samples) -> dict:
    """Median capture point per contact mode from nominal-payload ``(mode, (x, y))`` samples."""

    grouped = {mode: [] for mode in MODES}
    for mode, xy in samples:
        if mode in grouped:
            grouped[mode].append(xy)
    reference, counts = {}, {}
    for mode, values in grouped.items():
        counts[mode] = len(values)
        if not values:
            raise ValueError(f"no nominal samples for contact mode {mode!r}")
        xs = sorted(v[0] for v in values)
        ys = sorted(v[1] for v in values)
        reference[mode] = (xs[len(xs) // 2], ys[len(ys) // 2])
    return {"schema": REFERENCE_SCHEMA, "reference_xy": reference, "sample_counts": counts}


def load_reference(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("not a CoM stabilizer reference file")
    return {mode: tuple(data["reference_xy"][mode]) for mode in MODES}


class ComStabilizer:
    """Single-env runtime: measures the payload-augmented capture point and adds leg/waist offsets.

    With ``active=False`` it only measures and logs (used to build the reference).
    """

    def __init__(self, env, locked_indices, cfg: ComStabilizerCfg | None = None, active: bool = True):
        if env.num_envs != 1:
            raise ValueError("the CoM stabilizer evaluation currently supports one environment")
        self.cfg = (cfg or ComStabilizerCfg(step_dt=env.step_dt)).validate()
        self.active = active
        self.env, self.robot, self.box = env, env.scene["robot"], env.scene["box"]
        term = env.action_manager.get_term("JointPositionAction")
        self.scale = torch.as_tensor(term._scale, device=env.device).expand(43)
        joint_ids = term._joint_ids
        joint_ids = list(range(self.robot.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        names = list(term._joint_names)
        locked = set(locked_indices)
        self.action_ids = [i for i, n in enumerate(names)
                           if i not in locked and any(t in n for t in ("hip", "knee", "ankle", "waist"))]
        self.columns = [6 + joint_ids[i] for i in self.action_ids]
        self.masses = self.robot.root_physx_view.get_masses().to(env.device)[0]
        self.payload_mass = self.box.root_physx_view.get_masses().to(env.device)[0].sum()
        self.total_mass = self.masses.sum() + self.payload_mass
        self.feet = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
        self.wrists = self.robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True)[0]
        self.sensor = env.scene.sensors["contact_forces"]
        self.sensor_feet = self.sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"],
                                                   preserve_order=True)[0]
        self.shift = torch.zeros(2, device=env.device)
        self.steps = 0

    def _yaw_rotation(self) -> torch.Tensor:
        w, x, y, z = self.robot.data.root_link_quat_w[0].detach().cpu().tolist()
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)
        # world xy -> yaw frame
        return torch.tensor([[c, s], [-s, c]], device=self.env.device)

    def measure(self) -> dict:
        robot, cfg = self.robot, self.cfg
        com = ((self.masses[:, None] * robot.data.body_com_pos_w[0]).sum(0)
               + self.payload_mass * self.box.data.root_com_pos_w[0]) / self.total_mass
        velocity = ((self.masses[:, None] * robot.data.body_com_lin_vel_w[0]).sum(0)
                    + self.payload_mass * self.box.data.root_com_lin_vel_w[0]) / self.total_mass
        loads = self.sensor.data.net_forces_w[0, self.sensor_feet, 2].clamp_min(0)
        loaded = loads >= cfg.minimum_foot_load_fraction * self.total_mass * 9.81
        weights = loads * loaded
        weights = weights / weights.sum().clamp_min(1e-8) if loaded.any() else torch.full_like(weights, 0.5)
        feet = robot.data.body_link_pos_w[0, self.feet]
        origin = (weights[:, None] * feet).sum(0)
        origin_velocity = (weights[:, None] * robot.data.body_link_lin_vel_w[0, self.feet]).sum(0)
        omega = math.sqrt(9.81 / max(0.1, float(com[2] - origin[2])))
        rotation = self._yaw_rotation()
        capture_w = (com - origin)[:2] + (velocity - origin_velocity)[:2] / omega
        return {
            "mode": contact_mode(bool(loaded[0]), bool(loaded[1])),
            "capture_xy": rotation @ capture_w, "rotation": rotation, "weights": weights,
        }

    def _relative_jacobian_yaw(self, weights, rotation) -> torch.Tensor:
        jac = self.robot.root_physx_view.get_jacobians()[0, :, :3, :][:, :, self.columns].to(self.env.device)
        com_j = ((self.masses[:, None, None] * jac).sum(0) + self.payload_mass * jac[self.wrists].mean(0)) / self.total_mass
        support_j = (weights[:, None, None] * jac[self.feet]).sum(0)
        return rotation @ (com_j - support_j)[:2]

    def step(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Return the raw action with CoM offsets added (if active) and a log record."""

        state = self.measure()
        mode = state["mode"]
        gain = mode_gain(mode, self.steps, self.cfg) if self.active else 0.0
        reference = self.cfg.reference_xy[mode] if mode is not None else (0.0, 0.0)
        target = desired_com_shift(state["capture_xy"], reference, gain, self.cfg)
        self.shift = filter_shift(self.shift, target, self.cfg)
        self.steps += 1
        record = {
            "mode": mode, "capture_xy_m": state["capture_xy"].tolist(), "reference_xy_m": list(reference),
            "gain": gain, "com_shift_xy_m": self.shift.tolist(), "active": self.active,
        }
        if not self.active or mode is None:
            return raw_action, record
        offsets = joint_offsets(self._relative_jacobian_yaw(state["weights"], state["rotation"]), self.shift, self.cfg)
        action = raw_action.clone()
        action[0, self.action_ids] += offsets / self.scale[self.action_ids]
        record["max_joint_offset_rad"] = float(offsets.abs().max())
        return action, record
