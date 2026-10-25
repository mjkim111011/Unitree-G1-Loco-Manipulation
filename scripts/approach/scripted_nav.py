"""Privileged (ground-truth) go-to-pose navigator for the Homie ``[vx, vy, yaw_rate]`` interface.

This is the "Teacher Navigation" baseline and the future LP-NavOA-style Teacher:
it reads the simulator Box pose (privileged), computes the BC-1 canonical start
pose behind the Box, and emits body-frame velocity commands for the frozen
Homie locomotion policy.  Pure functions only; no simulator imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Teacher corpus: robot root at the env origin facing +x, Box centre 0.62 m ahead.
BC1_CANONICAL_STANDOFF_M = 0.62
HOMIE_STAND_THRESHOLD = 0.05  # Homie switches to its stand policy below this |cmd|


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bc1_start_pose(box_xy, box_yaw: float = 0.0, standoff: float = BC1_CANONICAL_STANDOFF_M) -> tuple[float, float, float]:
    """World pose (x, y, yaw) of the BC-1 canonical start relative to a Box."""
    x = float(box_xy[0]) - standoff * math.cos(box_yaw)
    y = float(box_xy[1]) - standoff * math.sin(box_yaw)
    return x, y, float(box_yaw)


@dataclass(frozen=True)
class NavGains:
    position_gain: float = 1.0
    yaw_gain: float = 1.5
    max_forward: float = 0.30
    max_backward: float = 0.15
    max_lateral: float = 0.20
    max_yaw_rate: float = 0.40
    heading_first_distance_m: float = 0.30  # beyond this, turn toward the path first
    heading_first_tolerance_rad: float = 0.35
    min_command_norm: float = 0.0  # > HOMIE_STAND_THRESHOLD keeps Homie walking until the caller stops


def pose_error_body(robot_xy, robot_yaw: float, target_xy, target_yaw: float) -> tuple[float, float, float]:
    """Target position in the robot yaw frame and the yaw error."""
    dx = float(target_xy[0]) - float(robot_xy[0])
    dy = float(target_xy[1]) - float(robot_xy[1])
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return c * dx + s * dy, -s * dx + c * dy, wrap_to_pi(float(target_yaw) - float(robot_yaw))


def nav_command(robot_xy, robot_yaw: float, target_xy, target_yaw: float, gains: NavGains = NavGains()) -> np.ndarray:
    """Body-frame ``[vx, vy, yaw_rate]`` that drives the base to a target pose.

    Far from the target the base first turns toward the path and walks forward
    (Homie tracks forward walking best); close to it the command becomes a
    holonomic pose servo that also aligns the final yaw.
    """
    ex, ey, eyaw = pose_error_body(robot_xy, robot_yaw, target_xy, target_yaw)
    distance = math.hypot(ex, ey)
    if distance > gains.heading_first_distance_m:
        path_heading_error = math.atan2(ey, ex)
        yaw_rate = gains.yaw_gain * path_heading_error
        forward = gains.position_gain * distance if abs(path_heading_error) < gains.heading_first_tolerance_rad else 0.0
        forward *= max(0.0, math.cos(path_heading_error))
        command = np.array([forward, 0.0, yaw_rate])
    else:
        command = np.array([gains.position_gain * ex, gains.position_gain * ey, gains.yaw_gain * eyaw])
    command[0] = float(np.clip(command[0], -gains.max_backward, gains.max_forward))
    command[1] = float(np.clip(command[1], -gains.max_lateral, gains.max_lateral))
    command[2] = float(np.clip(command[2], -gains.max_yaw_rate, gains.max_yaw_rate))
    norm = float(np.linalg.norm(command))
    if 0.0 < norm < gains.min_command_norm:
        command *= gains.min_command_norm / norm
    return command.astype(np.float32)


@dataclass(frozen=True)
class HandoffRegion:
    """Start-state region in which BC-1 is handed control (to be filled from the basin sweep)."""

    max_forward_error_m: float
    max_lateral_error_m: float
    max_yaw_error_rad: float
    max_speed_mps: float
    max_yaw_rate_rad_s: float
    min_settle_steps: int


def in_handoff_region(errors_body, speed_mps: float, yaw_rate: float, region: HandoffRegion) -> bool:
    ex, ey, eyaw = errors_body
    return (
        abs(ex) <= region.max_forward_error_m
        and abs(ey) <= region.max_lateral_error_m
        and abs(eyaw) <= region.max_yaw_error_rad
        and speed_mps <= region.max_speed_mps
        and abs(yaw_rate) <= region.max_yaw_rate_rad_s
    )


def _self_test() -> None:
    target = bc1_start_pose((0.62, 0.0))
    assert np.allclose(target, (0.0, 0.0, 0.0))
    # At the target: zero command (Homie stand policy).
    assert np.allclose(nav_command((0.0, 0.0), 0.0, target[:2], target[2]), 0.0)
    # 1 m behind, facing the target: walk forward at the cap, no turning.
    cmd = nav_command((-1.0, 0.0), 0.0, target[:2], target[2])
    assert cmd[0] == np.float32(0.30) and abs(cmd[2]) < 1e-6
    # 1 m to the side, facing +x: turn toward the path before walking.
    cmd = nav_command((0.0, -1.0), 0.0, target[:2], target[2])
    assert cmd[0] == 0.0 and cmd[2] > 0.0
    # Close range: holonomic correction plus final yaw alignment.
    cmd = nav_command((-0.05, 0.03), 0.1, target[:2], target[2])
    assert cmd[0] > 0.0 and cmd[1] < 0.0 and cmd[2] < 0.0
    # A small residual error is scaled up so that Homie does not fall back to its stand policy.
    cmd = nav_command((-0.01, 0.0), 0.0, target[:2], target[2], NavGains(min_command_norm=0.1))
    assert abs(float(np.linalg.norm(cmd)) - 0.1) < 1e-6 and cmd[0] > 0.0
    # Body-frame error is rotation-consistent.
    ex, ey, eyaw = pose_error_body((0.0, 0.0), math.pi / 2, (0.0, 1.0), math.pi / 2)
    assert abs(ex - 1.0) < 1e-9 and abs(ey) < 1e-9 and abs(eyaw) < 1e-9
    print("scripted_nav self-test passed")


if __name__ == "__main__":
    _self_test()
