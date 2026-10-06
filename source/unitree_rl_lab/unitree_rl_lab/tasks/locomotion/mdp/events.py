from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, UsdGeom

from isaaclab.envs import mdp as isaac_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import get_current_stage

from .task_phases import ensure_task_phase_buffers, reset_task_phase_attempt, reset_task_phase_episode

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_object_relative_to_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_b: tuple[float, float, float],
    position_range: dict[str, tuple[float, float]] | None = None,
    orientation_b: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    velocity_range: dict[str, tuple[float, float]] | None = None,
    z_relative_to_env_origin: bool = False,
    yaw_only: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Reset a rigid object to a pose expressed relative to the robot.

    When ``z_relative_to_env_origin`` is enabled, the sampled x-y offset is
    still relative to the robot, while z is measured from the environment
    origin (normally the terrain height). This is useful for placing objects
    around a spawned robot instead of relative to its pelvis height.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_envs = len(env_ids)
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]
    reference_quat_w = math_utils.yaw_quat(root_quat_w) if yaw_only else root_quat_w

    pos_b = torch.tensor(position_b, device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    if position_range is not None:
        ranges = torch.tensor(
            [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
            device=env.device,
            dtype=torch.float32,
        )
        pos_b += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 3), device=env.device)

    obj_pos_w = root_pos_w + math_utils.quat_apply(reference_quat_w, pos_b)
    if z_relative_to_env_origin:
        obj_pos_w[:, 2] = env.scene.env_origins[env_ids, 2] + pos_b[:, 2]

    obj_quat_b = torch.tensor(orientation_b, device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    obj_quat_w = math_utils.quat_mul(reference_quat_w, obj_quat_b)

    vel_w = torch.zeros(num_envs, 6, device=env.device)
    if velocity_range is not None:
        ranges = torch.tensor(
            [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=env.device,
            dtype=torch.float32,
        )
        vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=env.device)

    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def _get_box_half_heights(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    box_base_size: tuple[float, float, float],
) -> torch.Tensor:
    """Read per-environment USD scales once so boxes can be placed flush on a surface."""
    cache_name = f"_{object_cfg.name}_half_heights"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    obj: RigidObject = env.scene[object_cfg.name]
    prim_paths = sim_utils.find_matching_prim_paths(obj.cfg.prim_path)
    stage = get_current_stage()
    scale_z = []
    for prim_path in prim_paths:
        scale = stage.GetPrimAtPath(prim_path).GetAttribute("xformOp:scale").Get()
        scale_z.append(1.0 if scale is None else float(scale[2]))
    half_heights = 0.5 * box_base_size[2] * torch.tensor(scale_z, device=env.device)
    setattr(env, cache_name, half_heights)
    return half_heights


def preserve_child_world_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    relative_child_path: str,
    child_axis_to_parent_axis: tuple[int, int, int] = (0, 1, 2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Counter-scale a visual child after its parent is randomized anisotropically."""
    if env.sim.is_playing():
        raise RuntimeError("Child scale compensation must run in prestartup mode.")
    asset: RigidObject = env.scene[asset_cfg.name]
    parent_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = get_current_stage()
    child_suffix = relative_child_path if relative_child_path.startswith("/") else f"/{relative_child_path}"
    for env_id in env_ids.tolist():
        parent_prim = stage.GetPrimAtPath(parent_paths[env_id])
        parent_scale = parent_prim.GetAttribute("xformOp:scale").Get()
        if parent_scale is None:
            parent_scale = (1.0, 1.0, 1.0)
        child_prim = stage.GetPrimAtPath(parent_paths[env_id] + child_suffix)
        xformable = UsdGeom.Xformable(child_prim)
        child_scale = tuple(
            1.0 / float(parent_scale[parent_axis]) for parent_axis in child_axis_to_parent_axis
        )
        scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
        scale_op = scale_ops[0] if scale_ops else xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*child_scale))


def randomize_table_height_once(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    table_asset_positions_e: dict[str, tuple[float, float, float]],
    base_surface_height: float,
    height_delta_range: tuple[float, float],
):
    """Sample one table height per environment at startup and keep it fixed."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "table_surface_height"):
        env.table_surface_height = torch.full(
            (env.num_envs,), base_surface_height, device=env.device
        )

    num_envs = len(env_ids)
    delta = math_utils.sample_uniform(
        height_delta_range[0],
        height_delta_range[1],
        (num_envs,),
        device=env.device,
    )
    env.table_surface_height[env_ids] = base_surface_height + delta
    origins = env.scene.env_origins[env_ids]
    orientation = torch.zeros((num_envs, 4), device=env.device)
    orientation[:, 0] = 1.0
    zero_velocity = torch.zeros((num_envs, 6), device=env.device)

    for asset_name, position_e in table_asset_positions_e.items():
        table_part: RigidObject = env.scene[asset_name]
        position = origins + torch.tensor(position_e, device=env.device)
        # Moving all parts together preserves the collision-free U shape. The
        # small range can make the leg ends sit slightly above/below the ground.
        position[:, 2] += delta
        table_part.write_root_pose_to_sim(
            torch.cat([position, orientation], dim=-1), env_ids=env_ids
        )
        table_part.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def reset_pick_place_work_area(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    workspace_xy_range: dict[str, tuple[float, float]],
    table_origin_distance_range: tuple[float, float],
    bin_origin_distance_range: tuple[float, float],
    table_bin_distance_range: tuple[float, float],
    table_center_e: tuple[float, float, float],
    bin_center_e: tuple[float, float, float],
    bin_qr_center_e: tuple[float, float, float],
    table_asset_positions_e: dict[str, tuple[float, float, float]],
    bin_asset_positions_e: dict[str, tuple[float, float, float]],
    maximum_sampling_attempts: int = 64,
    table_xy_range: dict[str, tuple[float, float]] | None = None,
    bin_xy_range: dict[str, tuple[float, float]] | None = None,
):
    """Randomize the table and target bin inside one shared work square.

    The published ``table_center_e``, ``bin_center_e`` and
    ``bin_qr_position_e`` tensors are the single source of truth used by actor
    observations, camera navigation, rewards and success predicates.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if maximum_sampling_attempts <= 0:
        raise ValueError("maximum_sampling_attempts must be positive")

    num_envs = len(env_ids)
    xy_bounds = torch.tensor(
        [workspace_xy_range[key] for key in ("x", "y")],
        device=env.device,
        dtype=torch.float32,
    )

    def sample_square(requested=None) -> torch.Tensor:
        bounds = xy_bounds if requested is None else torch.tensor(
            [requested[key] for key in ("x", "y")], device=env.device, dtype=torch.float32
        )
        if not bool(torch.isfinite(bounds).all()) or bool((bounds[:, 0] > bounds[:, 1]).any()):
            raise ValueError("placement bounds must be finite with MIN <= MAX")
        if bool((bounds[:, 0] < xy_bounds[:, 0]).any()) or bool((bounds[:, 1] > xy_bounds[:, 1]).any()):
            raise ValueError("placement bounds must fit inside workspace_xy_range")
        return math_utils.sample_uniform(
            bounds[:, 0], bounds[:, 1], (num_envs, 2), device=env.device
        )

    def in_distance_range(points: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        distance = torch.linalg.vector_norm(points, dim=-1)
        return (distance >= bounds[0]) & (distance <= bounds[1])

    table_xy = sample_square(table_xy_range)
    table_valid = in_distance_range(table_xy, table_origin_distance_range)
    for _ in range(maximum_sampling_attempts):
        if bool(torch.all(table_valid)):
            break
        candidates = sample_square(table_xy_range)
        replace = ~table_valid
        table_xy[replace] = candidates[replace]
        table_valid = in_distance_range(table_xy, table_origin_distance_range)
    if not bool(torch.all(table_valid)):
        raise RuntimeError("unable to sample valid table positions inside the work area")

    bin_xy = sample_square(bin_xy_range)
    table_bin_distance = torch.linalg.vector_norm(bin_xy - table_xy, dim=-1)
    bin_valid = in_distance_range(bin_xy, bin_origin_distance_range) & (
        (table_bin_distance >= table_bin_distance_range[0])
        & (table_bin_distance <= table_bin_distance_range[1])
    )
    for _ in range(maximum_sampling_attempts):
        if bool(torch.all(bin_valid)):
            break
        candidates = sample_square(bin_xy_range)
        replace = ~bin_valid
        bin_xy[replace] = candidates[replace]
        table_bin_distance = torch.linalg.vector_norm(bin_xy - table_xy, dim=-1)
        bin_valid = in_distance_range(bin_xy, bin_origin_distance_range) & (
            (table_bin_distance >= table_bin_distance_range[0])
            & (table_bin_distance <= table_bin_distance_range[1])
        )
    if not bool(torch.all(bin_valid)):
        raise RuntimeError("unable to sample valid bin positions inside the work area")

    if not hasattr(env, "table_center_e"):
        env.table_center_e = torch.tensor(
            table_center_e, device=env.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)
        env.bin_center_e = torch.tensor(
            bin_center_e, device=env.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)
        env.bin_qr_position_e = torch.tensor(
            bin_qr_center_e, device=env.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)

    table_center = torch.tensor(table_center_e, device=env.device)
    bin_center = torch.tensor(bin_center_e, device=env.device)
    table_delta_xy = table_xy - table_center[:2]
    bin_delta_xy = bin_xy - bin_center[:2]
    env.table_center_e[env_ids, :2] = table_xy
    env.bin_center_e[env_ids, :2] = bin_xy
    env.bin_qr_position_e[env_ids, :2] = (
        torch.tensor(bin_qr_center_e[:2], device=env.device) + bin_delta_xy
    )

    origins = env.scene.env_origins[env_ids]
    identity = torch.zeros((num_envs, 4), device=env.device)
    identity[:, 0] = 1.0
    zero_velocity = torch.zeros((num_envs, 6), device=env.device)
    table_height_delta = torch.zeros(num_envs, device=env.device)
    if hasattr(env, "table_surface_height"):
        table_height_delta = env.table_surface_height[env_ids] - table_center[2]
        env.table_center_e[env_ids, 2] = env.table_surface_height[env_ids]

    for asset_name, base_position_e in table_asset_positions_e.items():
        asset: RigidObject = env.scene[asset_name]
        position_e = torch.tensor(base_position_e, device=env.device).repeat(num_envs, 1)
        position_e[:, :2] += table_delta_xy
        position_e[:, 2] += table_height_delta
        asset.write_root_pose_to_sim(
            torch.cat([origins + position_e, identity], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    for asset_name, base_position_e in bin_asset_positions_e.items():
        asset: RigidObject = env.scene[asset_name]
        position_e = torch.tensor(base_position_e, device=env.device).repeat(num_envs, 1)
        position_e[:, :2] += bin_delta_xy
        asset.write_root_pose_to_sim(
            torch.cat([origins + position_e, identity], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def reset_pick_place_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    grasp_box_xy_range: dict[str, tuple[float, float]],
    base_table_surface_height: float,
    box_base_size: tuple[float, float, float],
    box_xy_relative_to_table: bool = False,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Start every episode at grasp Phase 1 with the object on the table."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ensure_task_phase_buffers(env)
    if not hasattr(env, "table_surface_height"):
        env.table_surface_height = torch.full(
            (env.num_envs,), base_table_surface_height, device=env.device
        )

    num_envs = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    half_heights = _get_box_half_heights(env, object_cfg, box_base_size)[env_ids]
    grasp_ranges = torch.tensor(
        [grasp_box_xy_range[key] for key in ["x", "y"]],
        device=env.device,
        dtype=torch.float32,
    )
    grasp_xy = math_utils.sample_uniform(
        grasp_ranges[:, 0], grasp_ranges[:, 1], (num_envs, 2), device=env.device
    )
    if box_xy_relative_to_table:
        if not hasattr(env, "table_center_e"):
            raise RuntimeError(
                "box_xy_relative_to_table requires reset_pick_place_work_area"
            )
        grasp_xy += env.table_center_e[env_ids, :2]
    object_pos_w = origins.clone()
    object_pos_w[:, :2] += grasp_xy
    object_pos_w[:, 2] += env.table_surface_height[env_ids] + half_heights
    object_quat_w = torch.zeros((num_envs, 4), device=env.device)
    object_quat_w[:, 0] = 1.0

    obj: RigidObject = env.scene[object_cfg.name]
    obj.write_root_pose_to_sim(torch.cat([object_pos_w, object_quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=env.device), env_ids=env_ids)
    reset_task_phase_episode(env, env_ids, object_pos_w)


def retry_pick_place_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Restore only failed attempts to the grasp setup without ending the episode.

    The state machine sets ``phase_retry_pending`` during reward computation.
    This interval event runs after command computation, so the next observation
    consistently contains the restored robot-box pair and zero grasp command.
    Episode-level phase-bonus latches remain intact.
    """
    ensure_task_phase_buffers(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    retry_env_ids = env_ids[env.phase_retry_pending[env_ids]]
    if len(retry_env_ids) == 0:
        return

    # Restore a stable, box-front grasp setup rather than leaving the robot at
    # the bin after a failed carry.  This intentionally does not call _reset_idx
    # so episode time and already-claimed phase bonuses are preserved.
    isaac_mdp.reset_root_state_uniform(
        env,
        retry_env_ids,
        pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        velocity_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        asset_cfg=robot_cfg,
    )
    isaac_mdp.reset_joints_by_scale(
        env,
        retry_env_ids,
        position_range=(1.0, 1.0),
        velocity_range=(0.0, 0.0),
        asset_cfg=robot_cfg,
    )

    obj: RigidObject = env.scene[object_cfg.name]
    object_quat_w = torch.zeros((len(retry_env_ids), 4), device=env.device)
    object_quat_w[:, 0] = 1.0
    obj.write_root_pose_to_sim(
        torch.cat([env.phase_object_start_pos[retry_env_ids], object_quat_w], dim=-1),
        env_ids=retry_env_ids,
    )
    obj.write_root_velocity_to_sim(torch.zeros((len(retry_env_ids), 6), device=env.device), env_ids=retry_env_ids)

    # Clear policy/action history tied to the pre-drop state, while retaining
    # the episode and phase-bonus latches.
    env.action_manager.reset(retry_env_ids)
    env.command_manager.reset(retry_env_ids)
    env.observation_manager.reset(retry_env_ids)
    reset_task_phase_attempt(env, retry_env_ids)
