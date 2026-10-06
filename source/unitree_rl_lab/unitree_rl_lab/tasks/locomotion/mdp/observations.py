from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from .workspace import environment_point_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Object root position expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)


def object_lin_vel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Object root linear velocity relative to the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)


def _point_pos_camera(
    env: ManagerBasedRLEnv,
    point_pos_w: torch.Tensor,
    camera_position_b: tuple[float, float, float],
    camera_orientation_b: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Transform a world point into the D435 frame derived from its parent body."""
    robot: Articulation = env.scene[robot_cfg.name]
    torso_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    torso_quat_w = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    camera_position_b_tensor = torch.tensor(camera_position_b, device=env.device).repeat(env.num_envs, 1)
    camera_orientation_b_tensor = torch.tensor(camera_orientation_b, device=env.device).repeat(env.num_envs, 1)
    camera_pos_w = torso_pos_w + quat_apply(torso_quat_w, camera_position_b_tensor)
    camera_quat_w = quat_mul(torso_quat_w, camera_orientation_b_tensor)
    return quat_apply_inverse(camera_quat_w, point_pos_w - camera_pos_w)


def _camera_visible(
    point_pos_c: torch.Tensor,
    horizontal_fov_deg: float,
    image_width: int,
    image_height: int,
    clipping_range: tuple[float, float],
) -> torch.Tensor:
    """Check whether a point center lies inside a pinhole camera frustum."""
    tan_half_horizontal_fov = math.tan(math.radians(horizontal_fov_deg) * 0.5)
    tan_half_vertical_fov = tan_half_horizontal_fov * image_height / image_width
    forward = point_pos_c[:, 0]
    return (
        (forward >= clipping_range[0])
        & (forward <= clipping_range[1])
        & (torch.abs(point_pos_c[:, 1]) <= forward * tan_half_horizontal_fov)
        & (torch.abs(point_pos_c[:, 2]) <= forward * tan_half_vertical_fov)
    )


def _object_top_marker_pos_w(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Approximate the attached QR center at the randomized box's top surface."""
    half_heights = getattr(env, f"_{object_cfg.name}_half_heights", None)
    if half_heights is None:
        return obj.data.root_pos_w
    marker_offset = torch.zeros((env.num_envs, 3), device=env.device)
    marker_offset[:, 2] = half_heights
    return obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, marker_offset)


def object_visible_camera(
    env: ManagerBasedRLEnv,
    horizontal_fov_deg: float,
    image_width: int,
    image_height: int,
    clipping_range: tuple[float, float],
    camera_position_b: tuple[float, float, float],
    camera_orientation_b: tuple[float, float, float, float],
    visibility_on_top_surface: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Return one when the selected object marker point is inside the camera frustum."""
    obj: RigidObject = env.scene[object_cfg.name]
    visibility_point_w = (
        _object_top_marker_pos_w(env, obj, object_cfg) if visibility_on_top_surface else obj.data.root_pos_w
    )
    point_pos_c = _point_pos_camera(
        env,
        visibility_point_w,
        camera_position_b,
        camera_orientation_b,
        robot_cfg,
    )
    visible = _camera_visible(
        point_pos_c,
        horizontal_fov_deg,
        image_width,
        image_height,
        clipping_range,
    )
    return visible.float().unsqueeze(-1)


def object_pos_camera(
    env: ManagerBasedRLEnv,
    horizontal_fov_deg: float,
    image_width: int,
    image_height: int,
    clipping_range: tuple[float, float],
    camera_position_b: tuple[float, float, float],
    camera_orientation_b: tuple[float, float, float, float],
    visibility_on_top_surface: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Object position in camera coordinates, zeroed when it is not visible."""
    obj: RigidObject = env.scene[object_cfg.name]
    point_pos_c = _point_pos_camera(
        env,
        obj.data.root_pos_w,
        camera_position_b,
        camera_orientation_b,
        robot_cfg,
    )
    visibility_point_c = point_pos_c
    if visibility_on_top_surface:
        visibility_point_c = _point_pos_camera(
            env,
            _object_top_marker_pos_w(env, obj, object_cfg),
            camera_position_b,
            camera_orientation_b,
            robot_cfg,
        )
    visible = _camera_visible(
        visibility_point_c,
        horizontal_fov_deg,
        image_width,
        image_height,
        clipping_range,
    )
    return point_pos_c * visible.unsqueeze(-1)


def point_visible_camera(
    env: ManagerBasedRLEnv,
    point_pos_e: tuple[float, float, float],
    horizontal_fov_deg: float,
    image_width: int,
    image_height: int,
    clipping_range: tuple[float, float],
    camera_position_b: tuple[float, float, float],
    camera_orientation_b: tuple[float, float, float, float],
    point_attr: str | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Return one when an environment-relative point is inside the camera frustum."""
    point_pos_w = environment_point_w(env, point_pos_e, point_attr)
    point_pos_c = _point_pos_camera(
        env,
        point_pos_w,
        camera_position_b,
        camera_orientation_b,
        robot_cfg,
    )
    visible = _camera_visible(
        point_pos_c,
        horizontal_fov_deg,
        image_width,
        image_height,
        clipping_range,
    )
    return visible.float().unsqueeze(-1)


def point_pos_camera(
    env: ManagerBasedRLEnv,
    point_pos_e: tuple[float, float, float],
    point_attr: str | None = None,
    visibility_point_pos_e: tuple[float, float, float] | None = None,
    visibility_point_attr: str | None = None,
    horizontal_fov_deg: float | None = None,
    image_width: int = 160,
    image_height: int = 120,
    clipping_range: tuple[float, float] = (0.05, 8.0),
    mask_if_not_visible: bool = False,
    use_randomized_table_height: bool = False,
    camera_position_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_orientation_b: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Environment-relative point in camera coordinates with optional visibility masking."""
    point_pos_w = environment_point_w(env, point_pos_e, point_attr)
    if use_randomized_table_height and hasattr(env, "table_surface_height"):
        point_pos_w[:, 2] = env.scene.env_origins[:, 2] + env.table_surface_height
    point_pos_c = _point_pos_camera(
        env,
        point_pos_w,
        camera_position_b,
        camera_orientation_b,
        robot_cfg,
    )
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if mask_if_not_visible:
        if horizontal_fov_deg is None:
            raise ValueError("horizontal_fov_deg is required when mask_if_not_visible is enabled.")
        visibility_pos_c = point_pos_c
        if visibility_point_pos_e is not None:
            visibility_pos_w = environment_point_w(
                env, visibility_point_pos_e, visibility_point_attr
            )
            visibility_pos_c = _point_pos_camera(
                env,
                visibility_pos_w,
                camera_position_b,
                camera_orientation_b,
                robot_cfg,
            )
        valid &= _camera_visible(
            visibility_pos_c,
            horizontal_fov_deg,
            image_width,
            image_height,
            clipping_range,
        )
    return point_pos_c * valid.unsqueeze(-1)


def point_pos_b(
    env: ManagerBasedRLEnv,
    point_pos_e: tuple[float, float, float],
    point_attr: str | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of an environment-relative point expressed in the robot frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    point_pos_w = environment_point_w(env, point_pos_e, point_attr)
    return quat_apply_inverse(robot.data.root_quat_w, point_pos_w - robot.data.root_pos_w)


def task_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current grasp/carry/full-task curriculum phase as a scalar critic observation."""
    if not hasattr(env, "task_phase"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.task_phase.float().unsqueeze(-1)


def object_lin_vel_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Object linear velocity in the inertial frame for the RND curiosity state."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w


def object_target_distance(
    env: ManagerBasedRLEnv,
    target_pos_e: tuple[float, float, float],
    maximum_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Clipped object-to-target distance for the RND curiosity state."""
    obj: RigidObject = env.scene[object_cfg.name]
    target_pos_w = environment_point_w(env, target_pos_e, "bin_center_e")
    distance = torch.linalg.norm(obj.data.root_pos_w - target_pos_w, dim=-1)
    return torch.clamp(distance, max=maximum_distance).unsqueeze(-1)


def camera_object_distance(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    camera_position_b: tuple[float, float, float],
    camera_orientation_b: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Clipped camera-to-object distance for the RND curiosity state."""
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos_c = _point_pos_camera(
        env,
        obj.data.root_pos_w,
        camera_position_b,
        camera_orientation_b,
        robot_cfg,
    )
    return torch.clamp(torch.linalg.norm(object_pos_c, dim=-1), max=maximum_distance).unsqueeze(-1)


def min_body_object_distance(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Minimum selected-body-origin to object-root distance, clipped to a safe range.

    The term is intentionally independent of the task phase, so it can be
    used by an exploration module as well as by policy/critic observations.
    Pass hand proxy bodies or fingertip bodies through ``body_cfg`` depending
    on the desired manipulation signal.
    """
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids, :]
    distances = torch.linalg.vector_norm(body_pos_w - obj.data.root_pos_w.unsqueeze(1), dim=-1)
    return torch.clamp(distances.amin(dim=-1), min=0.0, max=maximum_distance).unsqueeze(-1)


def filtered_contact_count(
    env: ManagerBasedRLEnv,
    contact_sensor_names: tuple[str, ...],
    contact_force_threshold: float,
    maximum_count: int | None = None,
) -> torch.Tensor:
    """Number of configured filtered-contact sensors touching their filtered object.

    Each sensor must be attached to one body and configured with
    ``filter_prim_paths_expr`` for the target object.  Returning a scalar
    count (rather than raw forces) gives RND a bounded, contact-sensitive
    state component without exposing every contact matrix element.
    """
    if not contact_sensor_names:
        raise ValueError("contact_sensor_names must contain at least one filtered ContactSensor.")

    count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for sensor_name in contact_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        forces = sensor.data.force_matrix_w
        if forces is None:
            raise RuntimeError(
                f"'{sensor_name}' must use filter_prim_paths_expr to provide filtered contact forces."
            )
        force_norm = torch.linalg.vector_norm(forces, dim=-1).flatten(start_dim=1).amax(dim=-1)
        count += (force_norm >= contact_force_threshold).float()

    if maximum_count is not None:
        count = torch.clamp(count, min=0.0, max=float(maximum_count))
    return count.unsqueeze(-1)


def object_lift_height(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Positive object lift from its episode-start height, clipped to a safe range.

    Pick-and-place environments record ``phase_object_start_height`` at reset.
    Other environments safely fall back to their local environment origin,
    keeping this term reusable outside the phase-task implementation.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    start_height = getattr(env, "phase_object_start_height", env.scene.env_origins[:, 2])
    lift_height = obj.data.root_pos_w[:, 2] - start_height
    return torch.clamp(lift_height, min=0.0, max=maximum_height).unsqueeze(-1)


def rnd_safety_mask(
    env: ManagerBasedRLEnv,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg,
    min_foot_contact_force: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return whether an RND reward is allowed for this physical state.

    GRASP and PLACE require double support; CARRY allows either foot because a
    normal walking step is single-supported for part of its cycle.  Training
    additionally masks terminal transitions in the PPO hook, because Isaac
    Lab returns a reset observation after a termination.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    base_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    base_height_e = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    foot_contact_force = torch.linalg.vector_norm(
        contact_sensor.data.net_forces_w_history[:, :, feet_sensor_cfg.body_ids, :], dim=-1
    ).amax(dim=1)
    support_count = torch.sum(foot_contact_force >= min_foot_contact_force, dim=-1)

    # Phase values are defined in task_phases.py as GRASP=0, CARRY=1,
    # PLACE=2.  Falling back to GRASP preserves the most conservative mask if
    # this observation is evaluated before task buffers are initialized.
    phase = getattr(env, "task_phase", torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    required_support = torch.where(phase == 1, 1, 2)
    safe = (
        (base_tilt <= max_base_tilt_angle)
        & (base_height_e >= min_base_height)
        & (support_count >= required_support)
    )
    return safe.float().unsqueeze(-1)


def object_height_above_env_origin(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Absolute positive object root height above the local environment origin."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(height, min=0.0, max=maximum_height).unsqueeze(-1)
