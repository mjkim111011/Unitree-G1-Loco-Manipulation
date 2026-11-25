"""Experimental CoM-conditioned interpolation, not a stability guarantee.

Ranks monotone alpha candidates using a local joint-tracking/Jacobian model and
a capture-point proxy. Support is approximated by rectangles at currently loaded
feet, held fixed over the prediction horizon. Contact changes, impacts and full
rigid-body dynamics are not predicted. No extra posture/CoM action is applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComBlendCfg:
    step_dt: float = 0.02
    prediction_seconds: float = 0.10
    tracking_fraction: float = 0.25
    predicted_joint_correction_limit_rad: float = 0.05
    support_margin_m: float = 0.005
    foot_half_length_m: float = 0.10
    foot_half_width_m: float = 0.04
    minimum_foot_load_fraction: float = 0.05
    command_jump_limit_rad: float = 0.10
    alpha_increments: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08)
    max_seconds: float = 2.0
    forced_ramp_seconds: float = 0.5

    def validate(self):
        positive = [self.step_dt, self.prediction_seconds, self.tracking_fraction,
                    self.predicted_joint_correction_limit_rad, self.foot_half_length_m,
                    self.foot_half_width_m, self.minimum_foot_load_fraction,
                    self.command_jump_limit_rad, self.max_seconds, self.forced_ramp_seconds]
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("CoM blend parameters must be finite and positive")
        if not 0 <= self.support_margin_m < min(self.foot_half_length_m, self.foot_half_width_m):
            raise ValueError("invalid support margin")
        if self.tracking_fraction > 1 or self.forced_ramp_seconds > self.max_seconds:
            raise ValueError("invalid tracking fraction or deadline")
        if (not self.alpha_increments or self.alpha_increments[0] != 0
            or tuple(sorted(set(self.alpha_increments))) != self.alpha_increments
            or any(not math.isfinite(v) or v < 0 for v in self.alpha_increments)):
            raise ValueError("alpha increments must be sorted, unique, nonnegative, and include zero")
        return self


def alpha_candidates(alpha: float, elapsed_step: int, cfg: ComBlendCfg):
    elapsed = (elapsed_step + 1) * cfg.step_dt
    floor = max(0.0, min(1.0, (elapsed - cfg.max_seconds + cfg.forced_ramp_seconds) / cfg.forced_ramp_seconds))
    candidates = [min(1.0, max(alpha + increment, floor)) for increment in cfg.alpha_increments]
    return sorted(set(candidates)), floor


def convex_hull(points):
    """Counterclockwise 2-D convex hull, without a SciPy dependency."""
    points = sorted(set(tuple(map(float, p)) for p in points))
    if len(points) < 3:
        return points

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def signed_support_margin(points: torch.Tensor, polygon: torch.Tensor):
    if polygon.shape[0] < 3:
        return torch.full((points.shape[0],), -1000.0, device=points.device)
    edge = polygon.roll(-1, dims=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    distance = (edge[None, :, 0] * relative[:, :, 1] - edge[None, :, 1] * relative[:, :, 0])
    return (distance / torch.linalg.vector_norm(edge, dim=-1).clamp_min(1e-8)).amin(dim=1)


def choose_candidate(alphas, capture_points, polygon, command_jumps, bilateral_contact, cfg):
    margins = signed_support_margin(capture_points, polygon)
    if (not torch.isfinite(capture_points).all() or not torch.isfinite(command_jumps).all()
        or not torch.isfinite(polygon).all()):
        raise ValueError("nonfinite candidate predictions")
    feasible = ((margins >= cfg.support_margin_m)
                & (command_jumps <= cfg.command_jump_limit_rad)
                & bool(bilateral_contact))
    indices = torch.nonzero(feasible).flatten()
    if not bilateral_contact:
        index = int(torch.argmin(alphas))
    elif indices.numel():
        index = int(indices[torch.argmax(alphas[indices])])
    else:
        # Best-effort fallback, explicitly logged; this is not a safe-set proof.
        risk = ((cfg.support_margin_m - margins).clamp_min(0) / 0.01
                + (command_jumps / cfg.command_jump_limit_rad - 1).clamp_min(0))
        index = int(torch.argmin(risk))
    return index, margins, feasible


class ComPredictiveBlender:
    """Single-env runtime adapter; reads simulation state but does not mutate it."""

    def __init__(self, env, anchor, locked_indices, cfg=None):
        if env.num_envs != 1:
            raise ValueError("CoM blend evaluation currently supports one environment")
        self.cfg = (cfg or ComBlendCfg(step_dt=env.step_dt)).validate()
        self.env, self.robot, self.box = env, env.scene["robot"], env.scene["box"]
        self.anchor = anchor.clone()
        self.locked = list(locked_indices)
        self.alpha = 0.0
        term = env.action_manager.get_term("JointPositionAction")
        self.scale = torch.as_tensor(term._scale, device=env.device).expand(43)
        self.offset = torch.as_tensor(term._offset, device=env.device).expand_as(anchor)[0]
        joint_ids = term._joint_ids
        self.joint_ids = list(range(self.robot.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        self.unlocked = [i for i in range(43) if i not in self.locked]
        self.control_joints = [self.joint_ids[i] for i in self.unlocked]
        self.columns = [6 + i for i in self.control_joints]
        self.masses = self.robot.root_physx_view.get_masses().to(env.device)[0]
        self.robot_mass = self.masses.sum()
        self.payload_mass = self.box.root_physx_view.get_masses().to(env.device)[0].sum()
        self.feet = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
        self.wrists = self.robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True)[0]
        self.sensor = env.scene.sensors["contact_forces"]
        self.sensor_feet = self.sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]

    def step(self, bc3, elapsed_step, previous_target, bilateral_contact):
        if self.alpha >= 1.0:
            return bc3, {"alpha": 1.0, "active": False}
        cfg, robot = self.cfg, self.robot
        values, floor = alpha_candidates(self.alpha, elapsed_step, cfg)
        alphas = torch.tensor(values, device=self.env.device)
        actions = self.anchor + alphas[:, None] * (bc3 - self.anchor)
        actions[:, self.locked] = self.anchor[:, self.locked]
        targets = actions * self.scale + self.offset
        q = robot.data.joint_pos[0, self.control_joints]
        dq = robot.data.joint_vel[0, self.control_joints]
        correction = cfg.tracking_fraction * (targets[:, self.unlocked] - q - cfg.prediction_seconds * dq)
        correction = correction.clamp(-cfg.predicted_joint_correction_limit_rad, cfg.predicted_joint_correction_limit_rad)

        total = self.robot_mass + self.payload_mass
        com = ((self.masses[:, None] * robot.data.body_com_pos_w[0]).sum(0)
               + self.payload_mass * self.box.data.root_com_pos_w[0]) / total
        velocity = ((self.masses[:, None] * robot.data.body_com_lin_vel_w[0]).sum(0)
                    + self.payload_mass * self.box.data.root_com_lin_vel_w[0]) / total
        loads = self.sensor.data.net_forces_w[0, self.sensor_feet, 2].clamp_min(0)
        loaded = loads >= cfg.minimum_foot_load_fraction * total * 9.81
        weights = loads * loaded
        weights = weights / weights.sum().clamp_min(1e-8)
        if not loaded.any():
            weights = torch.full_like(weights, 0.5)
        feet_pos = robot.data.body_link_pos_w[0, self.feet]
        support_origin = (weights[:, None] * feet_pos).sum(0)
        support_velocity = (weights[:, None] * robot.data.body_link_lin_vel_w[0, self.feet]).sum(0)
        jac = robot.root_physx_view.get_jacobians()[0, :, :3, :][:, :, self.columns]
        # Rigid-grasp approximation for payload motion: average wrist Jacobian.
        com_j = ((self.masses[:, None, None] * jac).sum(0)
                 + self.payload_mass * jac[self.wrists].mean(0)) / total
        support_j = (weights[:, None, None] * jac[self.feet]).sum(0)
        relative_j = com_j - support_j
        relative_velocity = velocity - support_velocity
        shift = correction @ relative_j[:2].T
        predicted_com = (com - support_origin)[:2] + cfg.prediction_seconds * relative_velocity[:2] + shift
        predicted_velocity = relative_velocity[:2] + shift / cfg.prediction_seconds
        omega = torch.sqrt(9.81 / (com[2] - support_origin[2]).clamp_min(0.1))
        capture = predicted_com + predicted_velocity / omega

        corners = []
        for i in range(2):
            if not bool(loaded[i]):
                continue
            w, x, y, z = robot.data.body_link_quat_w[0, self.feet[i]].detach().cpu().tolist()
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            c, s = math.cos(yaw), math.sin(yaw)
            center = (feet_pos[i] - support_origin)[:2].detach().cpu().tolist()
            for dx in (-cfg.foot_half_length_m, cfg.foot_half_length_m):
                for dy in (-cfg.foot_half_width_m, cfg.foot_half_width_m):
                    corners.append((center[0] + c*dx-s*dy, center[1] + s*dx+c*dy))
        polygon = torch.tensor(convex_hull(corners), device=self.env.device).reshape(-1, 2)
        jumps = (targets[:, self.unlocked] - previous_target[0, self.unlocked]).abs().amax(1)
        index, margins, feasible = choose_candidate(alphas, capture, polygon, jumps, bilateral_contact, cfg)
        old_alpha = self.alpha
        self.alpha = float(alphas[index])
        debug = {
            "alpha": self.alpha, "previous_alpha": old_alpha, "active": True,
            "candidate_alphas": values, "candidate_capture_xy_m": capture.detach().cpu().tolist(),
            "candidate_margin_m": margins.detach().cpu().tolist(),
            "candidate_command_jump_rad": jumps.detach().cpu().tolist(),
            "feasible_candidates": int(feasible.sum()), "fallback": not bool(feasible.any()),
            "selected_margin_m": float(margins[index]), "support_polygon_xy_m": polygon.detach().cpu().tolist(),
            "loaded_feet": loaded.detach().cpu().tolist(), "bilateral_contact": bilateral_contact,
            "deadline_floor": floor, "deadline_forced": floor > old_alpha + 1e-6,
            "measured_com_relative_support_xy_m": (com-support_origin)[:2].detach().cpu().tolist(),
            "measured_com_velocity_relative_support_xy_mps": relative_velocity[:2].detach().cpu().tolist(),
        }
        return actions[index:index+1], debug
