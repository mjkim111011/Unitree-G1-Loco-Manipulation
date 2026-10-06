from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .task_phases import PLACE_PHASE, ensure_task_phase_buffers, final_task_succeeded
from .workspace import environment_point_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_error_above(
    env: ManagerBasedRLEnv,
    target_pos_b: tuple[float, float, float],
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate when an object drifts too far from its target pose in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_b = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    target_pos_b = torch.tensor(target_pos_b, device=env.device, dtype=torch.float32)
    return torch.linalg.norm(obj_pos_b - target_pos_b, dim=-1) > threshold


def object_in_receptacle(
    env: ManagerBasedRLEnv,
    target_pos_e: tuple[float, float, float],
    inner_half_extents: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate successfully after the object's center enters the target box."""
    obj: RigidObject = env.scene[object_cfg.name]
    target_pos_w = environment_point_w(env, target_pos_e, "bin_center_e")
    half_extents = torch.tensor(inner_half_extents, device=env.device)
    return torch.all(torch.abs(obj.data.root_pos_w - target_pos_w) < half_extents, dim=-1)


def object_out_of_workspace(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate if the movable object leaves its environment workspace."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos_e = obj.data.root_pos_w - env.scene.env_origins
    too_far = torch.linalg.norm(pos_e[:, :2], dim=-1) > maximum_distance
    too_low = pos_e[:, 2] < minimum_height
    return torch.logical_or(too_far, too_low)


def final_pick_place_success(
    env: ManagerBasedRLEnv,
    target_pos_e: tuple[float, float, float],
    inner_half_extents: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate immediately when Phase 3 places the object inside the bin."""
    ensure_task_phase_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    target_pos_w = environment_point_w(env, target_pos_e, "bin_center_e")
    half_extents = torch.tensor(inner_half_extents, device=env.device)
    delivered = torch.all(torch.abs(obj.data.root_pos_w - target_pos_w) < half_extents, dim=-1)
    return final_task_succeeded(env) | ((env.task_phase == PLACE_PHASE) & delivered)
