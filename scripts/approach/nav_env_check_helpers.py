"""Helpers shared by the navigation validation / evaluation scripts (import after AppLauncher)."""

from __future__ import annotations

import torch

from unitree_rl_lab.tasks.navigation import mdp as nav_mdp


def _with_gate(actions: torch.Tensor, stop: torch.Tensor) -> torch.Tensor:
    """Append the stop-gate column (+1 stop, -1 walk) expected by HomieNavAction(stop_gate=True)."""
    gate = torch.where(stop, torch.ones_like(stop, dtype=actions.dtype), -torch.ones_like(stop, dtype=actions.dtype))
    return torch.cat((actions, gate.unsqueeze(-1)), dim=-1)


def scripted_actions(env, scale: torch.Tensor, max_forward: float) -> torch.Tensor:
    """Vectorized scripts/approach/scripted_nav.nav_command toward the target point (normalized actions)."""
    target, theta = nav_mdp.target_point(env)
    robot = env.scene["robot"]
    yaw = nav_mdp.robot_yaw(env)
    rel = target - robot.data.root_link_pos_w[:, :2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    ex, ey = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
    eyaw = torch.atan2(torch.sin(theta - yaw), torch.cos(theta - yaw))
    distance = torch.hypot(ex, ey)
    path_heading = torch.atan2(ey, ex)
    far = distance > 0.3
    forward_far = torch.where(torch.abs(path_heading) < 0.35, distance, torch.zeros_like(distance)) * torch.clamp(torch.cos(path_heading), min=0.0)
    cmd = torch.stack((
        torch.where(far, forward_far, ex),
        torch.where(far, torch.zeros_like(ey), ey),
        torch.where(far, 1.5 * path_heading, 1.5 * eyaw),
    ), dim=-1)
    cmd[:, 0] = torch.clamp(cmd[:, 0], -0.15, max_forward)
    cmd[:, 1] = torch.clamp(cmd[:, 1], -0.2, 0.2)
    cmd[:, 2] = torch.clamp(cmd[:, 2], -0.4, 0.4)
    norm = torch.linalg.vector_norm(cmd, dim=-1, keepdim=True)
    cmd = torch.where((norm > 0) & (norm < 0.10), cmd * 0.10 / norm.clamp(min=1e-6), cmd)
    arrived = (distance <= 0.02) & (torch.abs(eyaw) <= 0.03)
    cmd[arrived] = 0.0
    return _with_gate(cmd / scale, arrived)


def scripted_staging_actions(env, scale: torch.Tensor, max_forward: float, staging_m: float = 0.5) -> torch.Tensor:
    """Scripted v2: align the heading at a staging point ``staging_m`` behind the target, then approach
    straight along the approach axis (no rotation next to the table)."""
    target, theta = nav_mdp.target_point(env)
    robot = env.scene["robot"]
    yaw = nav_mdp.robot_yaw(env)
    pos = robot.data.root_link_pos_w[:, :2]
    ct, st = torch.cos(theta), torch.sin(theta)
    rel_t = pos - target
    ax, ay = ct * rel_t[:, 0] + st * rel_t[:, 1], -st * rel_t[:, 0] + ct * rel_t[:, 1]  # robot in the approach frame
    eyaw = torch.atan2(torch.sin(theta - yaw), torch.cos(theta - yaw))
    corridor = (ax >= -(staging_m + 0.05)) & (ax <= 0.05) & (torch.abs(ay) <= 0.06) & (torch.abs(eyaw) <= 0.15)
    staging = target - staging_m * torch.stack((ct, st), dim=-1)
    goal = torch.where(corridor.unsqueeze(-1), target, staging)
    rel = goal - pos
    c, s = torch.cos(yaw), torch.sin(yaw)
    ex, ey = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
    distance = torch.hypot(ex, ey)
    path_heading = torch.atan2(ey, ex)
    far = (distance > 0.3) & ~corridor
    forward_far = torch.where(torch.abs(path_heading) < 0.35, distance, torch.zeros_like(distance)) * torch.clamp(torch.cos(path_heading), min=0.0)
    cmd = torch.stack((
        torch.where(far, forward_far, ex),
        torch.where(far, torch.zeros_like(ey), ey),
        torch.where(far, 1.5 * path_heading, 1.5 * eyaw),
    ), dim=-1)
    cmd[:, 0] = torch.clamp(cmd[:, 0], -0.15, max_forward)
    cmd[:, 1] = torch.clamp(cmd[:, 1], -0.2, 0.2)
    cmd[:, 2] = torch.clamp(cmd[:, 2], -0.4, 0.4)
    norm = torch.linalg.vector_norm(cmd, dim=-1, keepdim=True)
    cmd = torch.where((norm > 0) & (norm < 0.10), cmd * 0.10 / norm.clamp(min=1e-6), cmd)
    arrived = corridor & (torch.linalg.vector_norm(target - pos, dim=-1) <= 0.02) & (torch.abs(eyaw) <= 0.03)
    cmd[arrived] = 0.0
    return _with_gate(cmd / scale, arrived)
