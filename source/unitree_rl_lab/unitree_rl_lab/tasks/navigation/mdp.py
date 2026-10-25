"""MDP terms of the goal-conditioned navigation task (Homie ``[vx, vy, yaw_rate]``, BC-1 handoff target).

Geometry (approach frame).  The BC-1 canonical start is the Box centre minus 0.62 m along the approach
direction n = (cos theta, sin theta), facing n.  ``theta`` is the work-area (table) approach yaw, *not*
the Box yaw: the Box is a 90-degree-symmetric cube whose yaw is reset to 0, BC-1/BC-2 never observe
the Box orientation, and the Teacher retargets with Mimic TRANSLATE (translation only).  In this scene
the table is axis-aligned and never rotated, so theta = 0 for every environment.

Handoff errors are expressed relative to the canonical start in the approach frame:
    e_x  > 0 = closer to the Box,  e_y = left,  e_yaw = robot yaw - theta.
The measured BC-1 v2 region (40 post-navigation handoffs, 2026-10-19) is
    e_x in [-0.08, +0.015] m, |e_y| <= 0.04 m, |e_yaw| <= 0.08 rad, speed <= 0.1 m/s.
The navigation target point is the centre of that window: e_x = -0.03 m, e_y = 0, e_yaw = 0.

GT marking: observation terms whose name starts with ``gt_`` read the simulator Box pose (privileged).
They are the parts to replace by perception (camera Box/table estimate) later.
"""

from __future__ import annotations

import math
from collections import deque

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi

CANONICAL_STANDOFF_M = 0.62
TARGET_OFFSET_X_M = -0.03
HANDOFF_X_RANGE = (-0.08, 0.015)
HANDOFF_Y_MAX = 0.04
HANDOFF_YAW_MAX = 0.08
TABLE_CENTER_XY_E = (0.62, 0.0)
TABLE_HALF_EXTENT_M = 0.20
TABLE_TOP_Z = 0.74
BOX_SIZE_M = 0.2
BOX_CENTER_OFFSET = (0.0, 0.0)  # evaluation only: fixed Box offset from the table centre added to the stage's random offset

def apply_nav_target_overrides(target_offset_x: float | None, handoff_x_range: tuple[float, float] | None) -> dict:
    """Override the navigation target point / handoff x-window (module constants read at call time)."""
    global TARGET_OFFSET_X_M, HANDOFF_X_RANGE
    if target_offset_x is not None:
        TARGET_OFFSET_X_M = float(target_offset_x)
    if handoff_x_range is not None:
        HANDOFF_X_RANGE = (float(handoff_x_range[0]), float(handoff_x_range[1]))
    return {"target_offset_x_m": TARGET_OFFSET_X_M, "handoff_x_range": list(HANDOFF_X_RANGE)}


# Curriculum stages (start sampled around the canonical start in the approach frame):
#   r: distance from the target point, bearing: angle of the start position from the -n axis
#   (0 = straight behind the target, +-90 deg = beside it), yaw: heading offset from "facing the target".
STAGES = (
    {"name": "S1_0.5-1m_front", "r": (0.5, 1.0), "bearing": math.radians(15.0), "yaw": 0.2, "box_x": 0.0, "box_y": 0.0},
    {"name": "S2_1-2m_side_yaw", "r": (1.0, 2.0), "bearing": math.radians(60.0), "yaw": 0.8, "box_x": 0.0, "box_y": 0.0},
    {"name": "S3_3-5m_any_heading", "r": (3.0, 5.0), "bearing": math.radians(90.0), "yaw": math.pi, "box_x": 0.0, "box_y": 0.0},
    {"name": "S4_3-5m_box_on_table", "r": (3.0, 5.0), "bearing": math.radians(90.0), "yaw": math.pi, "box_x": 0.03, "box_y": 0.08},
)


# ---------------------------------------------------------------------------------------------- geometry
def approach_yaw(env) -> torch.Tensor:
    """Work-area approach yaw per env (table-normal heading).  0 in the current, axis-aligned scene."""
    value = getattr(env, "nav_approach_yaw", None)
    if value is None:
        value = torch.zeros(env.num_envs, device=env.device)
        env.nav_approach_yaw = value
    return value


def canonical_start(env) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy and yaw of the BC-1 canonical start for every env (from the GT Box position)."""
    theta = approach_yaw(env)
    box_xy = env.scene["box"].data.root_link_pos_w[:, :2]
    direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    return box_xy - CANONICAL_STANDOFF_M * direction, theta


def target_point(env) -> tuple[torch.Tensor, torch.Tensor]:
    canonical, theta = canonical_start(env)
    direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    return canonical + TARGET_OFFSET_X_M * direction, theta


def robot_yaw(env) -> torch.Tensor:
    q = env.scene["robot"].data.root_link_quat_w
    return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def handoff_errors(env) -> torch.Tensor:
    """(e_x, e_y, e_yaw) of the robot root relative to the canonical start, in the approach frame."""
    canonical, theta = canonical_start(env)
    rel = env.scene["robot"].data.root_link_pos_w[:, :2] - canonical
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], wrap_to_pi(robot_yaw(env) - theta)), dim=-1)


def target_errors(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance to the target point and heading error (theta - yaw)."""
    target, theta = target_point(env)
    distance = torch.linalg.vector_norm(target - env.scene["robot"].data.root_link_pos_w[:, :2], dim=-1)
    return distance, wrap_to_pi(theta - robot_yaw(env))


# ------------------------------------------------------------------------------------------ observations
def gt_goal_pos_b(env) -> torch.Tensor:
    """[GT] Target point in the robot yaw frame (m)."""
    target, _ = target_point(env)
    rel = target - env.scene["robot"].data.root_link_pos_w[:, :2]
    yaw = robot_yaw(env)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]), dim=-1)


def gt_goal_heading(env) -> torch.Tensor:
    """[GT] sin/cos of the final-heading error (theta - yaw)."""
    _, error = target_errors(env)
    return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


def gt_goal_distance(env, clip: float = 6.0) -> torch.Tensor:
    """[GT] Distance to the target point (m), clipped."""
    distance, _ = target_errors(env)
    return torch.clamp(distance, max=clip).unsqueeze(-1)


def homie_command(env) -> torch.Tensor:
    """The velocity command Homie received at the last step (m/s, rad/s)."""
    return env.action_manager.get_term("homie").processed_actions


# ------------------------------------------------------------------------------------------- terminations
class HandoffSuccess(ManagerTermBase):
    """BC-1 handoff state held for ``hold_steps`` consecutive control steps (real simulator state)."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.hold[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_steps: int = 25, speed_max: float = 0.10, yaw_rate_max: float = 0.20,
                 tilt_max: float = 0.20, arm_dev_max: float = 0.25) -> torch.Tensor:
        robot = env.scene["robot"]
        e = handoff_errors(env)
        speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=-1)
        yaw_rate = torch.abs(robot.data.root_link_ang_vel_b[:, 2])
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        arm_ids = env.action_manager.get_term("homie")._arm_ids
        arm_dev = torch.abs(robot.data.joint_pos[:, arm_ids]).amax(dim=-1)
        self.inside = ((e[:, 0] >= HANDOFF_X_RANGE[0]) & (e[:, 0] <= HANDOFF_X_RANGE[1]) & (torch.abs(e[:, 1]) <= HANDOFF_Y_MAX)
                       & (torch.abs(e[:, 2]) <= HANDOFF_YAW_MAX) & (speed <= speed_max) & (yaw_rate <= yaw_rate_max)
                       & (tilt <= tilt_max) & (arm_dev <= arm_dev_max))
        self.hold = torch.where(self.inside, self.hold + 1, torch.zeros_like(self.hold))
        return self.hold >= hold_steps


LEG_JOINT_KEYS = ("hip", "knee", "ankle")


class StableHandoffSuccess(ManagerTermBase):
    """v6: the BC-1 handoff window held by a *settled* robot for ``hold_steps`` control steps.

    In addition to the v5 window (HandoffSuccess) every hold step needs
      - base speed <= ``speed_max`` and |yaw rate| <= ``yaw_rate_max``
      - leg joint speed norm <= ``leg_joint_speed_max`` (the gait has died out, not only the base)
      - the policy commanding exactly zero (stop gate on and the rate-limited command decayed): no re-walking.
    v5 ended the episode after a 25-step hold, so it never learned to stay still afterwards.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        names = list(env.scene["robot"].joint_names)
        self.leg_ids = [i for i, n in enumerate(names) if any(k in n for k in LEG_JOINT_KEYS)]

    def reset(self, env_ids=None) -> None:
        self.hold[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_steps: int = 50, speed_max: float = 0.05, yaw_rate_max: float = 0.10,
                 leg_joint_speed_max: float = 0.3, tilt_max: float = 0.20, arm_dev_max: float = 0.25) -> torch.Tensor:
        robot = env.scene["robot"]
        e = handoff_errors(env)
        speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=-1)
        yaw_rate = torch.abs(robot.data.root_link_ang_vel_b[:, 2])
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        arm_ids = env.action_manager.get_term("homie")._arm_ids
        arm_dev = torch.abs(robot.data.joint_pos[:, arm_ids]).amax(dim=-1)
        leg_speed = torch.linalg.vector_norm(robot.data.joint_vel[:, self.leg_ids], dim=-1)
        stopped = torch.linalg.vector_norm(homie_command(env), dim=-1) < 1.0e-6
        self.inside = ((e[:, 0] >= HANDOFF_X_RANGE[0]) & (e[:, 0] <= HANDOFF_X_RANGE[1]) & (torch.abs(e[:, 1]) <= HANDOFF_Y_MAX)
                       & (torch.abs(e[:, 2]) <= HANDOFF_YAW_MAX) & (speed <= speed_max) & (yaw_rate <= yaw_rate_max)
                       & (tilt <= tilt_max) & (arm_dev <= arm_dev_max) & (leg_speed <= leg_joint_speed_max) & stopped)
        self.hold = torch.where(self.inside, self.hold + 1, torch.zeros_like(self.hold))
        return self.hold >= hold_steps


class ContactCollision(ManagerTermBase):
    """v6: any robot body except the feet touching something (net contact force > ``threshold`` N).

    Self-collision is disabled for this robot (configure_arena_teacher_robot), so this is contact with the
    table, the Box or the ground.  v5 only checked link origins against the table box (1 cm margin) and
    missed fingertip contacts (5/60 Arena-WBC episodes).
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        sensor = env.scene.sensors["contact_forces"]
        names = list(sensor.body_names)
        self.other_ids = [i for i, n in enumerate(names) if "ankle_roll" not in n]
        self.body_names = [names[i] for i in self.other_ids]
        self.last_force = torch.zeros(env.num_envs, device=env.device)
        self.last_body = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # diagnostics: index into body_names

    def __call__(self, env, threshold: float = 1.0) -> torch.Tensor:
        sensor = env.scene.sensors["contact_forces"]
        force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, :, self.other_ids], dim=-1).amax(dim=1)
        self.last_force, self.last_body = force.max(dim=-1)
        return self.last_force > threshold


def fall(env, min_height: float = 0.55, max_tilt: float = 0.8) -> torch.Tensor:
    robot = env.scene["robot"]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    return (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < min_height) | (tilt > max_tilt)


class Collision(ManagerTermBase):
    """Any robot body inside the (inflated) table volume, or the Box pushed from its reset position."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.box_ref = torch.zeros(env.num_envs, 2, device=env.device)
        self.needs_ref = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.needs_ref[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, margin: float = 0.05, box_shift_max: float = 0.03) -> torch.Tensor:
        box_xy = env.scene["box"].data.root_link_pos_w[:, :2]
        self.box_ref = torch.where(self.needs_ref.unsqueeze(-1), box_xy, self.box_ref)
        self.needs_ref[:] = False
        bodies = env.scene["robot"].data.body_link_pos_w  # (N, B, 3)
        origin = env.scene.env_origins
        rel = bodies - origin.unsqueeze(1)
        inside_bodies = ((torch.abs(rel[..., 0] - TABLE_CENTER_XY_E[0]) <= TABLE_HALF_EXTENT_M + margin)
                         & (torch.abs(rel[..., 1] - TABLE_CENTER_XY_E[1]) <= TABLE_HALF_EXTENT_M + margin)
                         & (rel[..., 2] <= TABLE_TOP_Z + margin))
        self.last_inside_bodies = inside_bodies
        inside = inside_bodies.any(dim=-1)
        pushed = torch.linalg.vector_norm(box_xy - self.box_ref, dim=-1) > box_shift_max
        self.last_inside, self.last_pushed = inside, pushed  # diagnostics
        return inside | pushed


# ------------------------------------------------------------------------------------------------ rewards
class TargetProgress(ManagerTermBase):
    """Potential-based progress: Phi = -(d + near_gain * min(d, near_cap) + yaw_weight * |heading error| * near).

    reward = dPhi/dt.  ``near_gain`` steepens the last ``near_cap`` metres so that entering the handoff
    window is worth more than parking just outside it (rollout of reward v2 stopped 15 cm short).

    The episode sum telescopes to Phi(end) - Phi(start), so loitering near the target earns nothing.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.prev = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, yaw_weight: float = 0.5, near_radius: float = 0.5, near_gain: float = 0.0,
                 near_cap: float = 0.3) -> torch.Tensor:
        distance, heading = target_errors(env)
        near = torch.clamp(1.0 - distance / near_radius, 0.0, 1.0)
        phi = -(distance + near_gain * torch.clamp(distance, max=near_cap) + yaw_weight * torch.abs(heading) * near)
        reward = torch.where(self.fresh, torch.zeros_like(phi), phi - self.prev) / env.step_dt
        self.prev = phi
        self.fresh[:] = False
        return reward


def _table_body_distance(env) -> torch.Tensor:
    """Per env: smallest distance of any robot body origin to the table volume (0 inside)."""
    rel = env.scene["robot"].data.body_link_pos_w - env.scene.env_origins.unsqueeze(1)
    dx = torch.clamp(torch.abs(rel[..., 0] - TABLE_CENTER_XY_E[0]) - TABLE_HALF_EXTENT_M, min=0.0)
    dy = torch.clamp(torch.abs(rel[..., 1] - TABLE_CENTER_XY_E[1]) - TABLE_HALF_EXTENT_M, min=0.0)
    dz = torch.clamp(rel[..., 2] - TABLE_TOP_Z, min=0.0)
    return torch.sqrt(dx**2 + dy**2 + dz**2).amin(dim=-1)


def table_proximity(env, margin: float = 0.05) -> torch.Tensor:
    """1 while any robot body is within ``margin`` of the table volume (near-miss zone, penalised)."""
    return (_table_body_distance(env) < margin).float()


def termination_event(env, term: str) -> torch.Tensor:
    """1/dt on the step the named termination fires (a one-off bonus/penalty of size ``weight``)."""
    return env.termination_manager.get_term(term).float() / env.step_dt


def constant(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def near_target_motion(env, radius: float = 0.15) -> torch.Tensor:
    """Base speed + 0.5 |yaw rate| inside ``radius`` of the target (penalised: stop once there)."""
    robot = env.scene["robot"]
    distance, _ = target_errors(env)
    motion = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=-1) + 0.5 * torch.abs(robot.data.root_link_ang_vel_b[:, 2])
    return motion * (distance < radius).float()


def stopped_outside_window(env, radius: float = 0.3) -> torch.Tensor:
    """v6b: 1 while the policy commands exactly zero (stopped) within ``radius`` of the target but outside the
    handoff window.  The deterministic v6a policy stopped cleanly 1-7 cm outside the window and never moved again
    (nothing rewarded a small correction once it stood still)."""
    distance, _ = target_errors(env)
    e = handoff_errors(env)
    in_window = ((e[:, 0] >= HANDOFF_X_RANGE[0]) & (e[:, 0] <= HANDOFF_X_RANGE[1]) & (torch.abs(e[:, 1]) <= HANDOFF_Y_MAX)
                 & (torch.abs(e[:, 2]) <= HANDOFF_YAW_MAX))
    stopped = torch.linalg.vector_norm(homie_command(env), dim=-1) < 1.0e-6
    return (stopped & (distance < radius) & ~in_window).float()


def command_magnitude_near_target(env, radius: float = 0.15) -> torch.Tensor:
    """|command|^2 inside ``radius`` of the target: commanding Homie to stand there is free."""
    distance, _ = target_errors(env)
    return torch.sum(homie_command(env) ** 2, dim=-1) * (distance < radius).float()


# ------------------------------------------------------------------------------------- events / curriculum
class NavStageCurriculum(ManagerTermBase):
    """Global start-distribution stage advanced on the rolling handoff success rate of its own episodes."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        self.stage = int(params.get("start_stage", 0))
        self.window = deque(maxlen=int(params.get("window", 400)))
        self.episode_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.nav_stage_state = self

    def __call__(self, env, env_ids, start_stage: int = 0, window: int = 400, thresholds=(0.8, 0.8, 0.75),
                 mix_prob: float = 0.2) -> dict:
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.arange(env.num_envs, device=env.device)
        done = ids[self.started[ids]]
        if len(done):
            success = env.termination_manager.get_term("handoff_success")[done]
            same = self.episode_stage[done] == self.stage
            self.window.extend(success[same].tolist())
        rate = sum(self.window) / len(self.window) if self.window else 0.0
        if (self.stage < len(STAGES) - 1 and len(self.window) == self.window.maxlen
                and rate >= thresholds[min(self.stage, len(thresholds) - 1)]):
            self.stage += 1
            self.window.clear()
        return {"stage": float(self.stage), "stage_success_rate": float(rate)}


def reset_box_and_robot(env, env_ids, fixed_stage: int = 2, mix_prob: float = 0.2,
                        start_range_override: dict | None = None) -> None:
    """Place the Box on the table (stage-dependent offset) and the robot around the BC-1 start.

    The stage comes from ``env.nav_stage_state`` (training curriculum) or ``fixed_stage``.  With
    probability ``mix_prob`` an episode re-uses a random earlier stage (anti-forgetting).
    ``start_range_override`` = {"r": (lo, hi), "bearing": rad, "yaw": rad, "box_x": m, "box_y": m}
    replaces the stage parameters (evaluation of unseen start ranges).
    """
    device = env.device
    n = len(env_ids)
    state = getattr(env, "nav_stage_state", None)
    current = state.stage if state is not None else int(fixed_stage)
    stages = torch.full((n,), current, dtype=torch.long, device=device)
    if current > 0 and mix_prob > 0.0:
        mix = torch.rand(n, device=device) < mix_prob
        stages = torch.where(mix, torch.randint(0, current, (n,), device=device), stages)
    if state is not None:
        state.episode_stage[env_ids] = stages
        state.started[env_ids] = True

    if start_range_override is not None:
        params = [start_range_override] * n
    else:
        params = [STAGES[s] for s in stages.tolist()]
    r_lo = torch.tensor([p["r"][0] for p in params], device=device)
    r_hi = torch.tensor([p["r"][1] for p in params], device=device)
    bearing_max = torch.tensor([p["bearing"] for p in params], device=device)
    yaw_max = torch.tensor([p["yaw"] for p in params], device=device)
    box_x = torch.tensor([p["box_x"] for p in params], device=device)
    box_y = torch.tensor([p["box_y"] for p in params], device=device)

    origins = env.scene.env_origins[env_ids]
    # Box on the table top, axis-aligned (yaw 0).
    box = env.scene["box"]
    box_pos = torch.zeros(n, 3, device=device)
    box_pos[:, 0] = TABLE_CENTER_XY_E[0] + BOX_CENTER_OFFSET[0] + (2.0 * torch.rand(n, device=device) - 1.0) * box_x
    box_pos[:, 1] = TABLE_CENTER_XY_E[1] + BOX_CENTER_OFFSET[1] + (2.0 * torch.rand(n, device=device) - 1.0) * box_y
    box_pos[:, 2] = TABLE_TOP_Z + 0.5 * BOX_SIZE_M + 0.001
    box_quat = torch.zeros(n, 4, device=device)
    box_quat[:, 0] = 1.0
    box.write_root_pose_to_sim(torch.cat((box_pos + origins, box_quat), dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    # Robot start around the target point (approach yaw theta = 0 in this scene).
    theta = approach_yaw(env)[env_ids]
    target_xy = box_pos[:, :2] - (CANONICAL_STANDOFF_M - TARGET_OFFSET_X_M) * torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    r = r_lo + torch.rand(n, device=device) * (r_hi - r_lo)
    bearing = (2.0 * torch.rand(n, device=device) - 1.0) * bearing_max
    back = theta + math.pi + bearing  # direction from the target to the start
    start_xy = target_xy + r.unsqueeze(-1) * torch.stack((torch.cos(back), torch.sin(back)), dim=-1)
    facing = torch.atan2(target_xy[:, 1] - start_xy[:, 1], target_xy[:, 0] - start_xy[:, 0])
    yaw = wrap_to_pi(facing + (2.0 * torch.rand(n, device=device) - 1.0) * yaw_max)
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, 0:2] = start_xy + origins[:, :2]
    root[:, 2] += origins[:, 2]
    root[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
