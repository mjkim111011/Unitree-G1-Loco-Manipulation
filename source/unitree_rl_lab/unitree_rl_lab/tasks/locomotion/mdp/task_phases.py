from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .workspace import environment_point_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


GRASP_PHASE = 0
CARRY_PHASE = 1
PLACE_PHASE = 2
COMPLETE_PHASE = 3
# Kept as an alias for observation/reward code written before the phases became
# a within-episode state machine.
FULL_TASK_PHASE = PLACE_PHASE


def ensure_task_phase_buffers(env: ManagerBasedRLEnv) -> None:
    """Create per-environment state for the within-episode pick-and-place task."""
    if hasattr(env, "task_phase"):
        # This branch is mainly useful for an interactive reload.  A normal
        # environment creates every buffer below together, but do not make a
        # newly added shaping term fail on an already initialized environment.
        if not hasattr(env, "phase_best_cage_distance"):
            env.phase_best_cage_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
        if not hasattr(env, "phase_best_close_contact_score"):
            env.phase_best_close_contact_score = torch.zeros(env.num_envs, device=env.device)
        if not hasattr(env, "phase_best_pre_lift_fraction"):
            env.phase_best_pre_lift_fraction = torch.zeros(env.num_envs, device=env.device)
        if not hasattr(env, "phase_pre_lift_escrow"):
            env.phase_pre_lift_escrow = torch.zeros(env.num_envs, device=env.device)
        if not hasattr(env, "phase_pre_lift_escrow_steps"):
            env.phase_pre_lift_escrow_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        return

    env.task_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.phase_object_start_height = torch.zeros(env.num_envs, device=env.device)
    env.phase_object_start_pos = torch.zeros(env.num_envs, 3, device=env.device)
    env.final_task_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Phase bonuses are deliberately latched across retries.  A retry is still
    # possible, but lift/drop cycles cannot repeatedly collect the same bonus.
    env.phase_bonus_claimed = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=env.device)
    env.phase_task_reward_event = torch.zeros(env.num_envs, device=env.device)
    env.phase_drop_event = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.phase_retry_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.phase_retry_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env.phase_stable_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.phase_hold_lost_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.phase_stable_hold = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Stable contact is deliberately tracked separately from a lifted hold: it
    # is the safe intermediate condition that can be rewarded before the box
    # has cleared the table.
    env.phase_stable_contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.phase_stable_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.phase_contact_reward_claimed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Monotonic progress potentials prevent approach/lift/transport shaping from
    # being collected by moving away and returning to the same state.
    env.phase_best_hand_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
    env.phase_best_cage_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
    env.phase_best_close_contact_score = torch.zeros(env.num_envs, device=env.device)
    # This has a separate potential from the strict lift term.  A low-force
    # contact can earn the short pre-lift bridge without consuming the later
    # stable-contact lift reward.
    env.phase_best_pre_lift_fraction = torch.zeros(env.num_envs, device=env.device)
    # The short pre-lift bridge is paid only after a continuous safe-stability
    # window.  Until then it is escrowed and discarded as soon as the posture
    # or the grasp ceases to be safe.
    env.phase_pre_lift_escrow = torch.zeros(env.num_envs, device=env.device)
    env.phase_pre_lift_escrow_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.phase_best_lift_fraction = torch.zeros(env.num_envs, device=env.device)
    env.phase_best_staging_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
    env.phase_best_bin_distance = torch.full((env.num_envs,), float("inf"), device=env.device)


def reset_task_phase_episode(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_start_pos_w: torch.Tensor,
) -> None:
    """Reset all task/reward state at a true environment reset."""
    ensure_task_phase_buffers(env)
    env.task_phase[env_ids] = GRASP_PHASE
    env.phase_object_start_pos[env_ids] = object_start_pos_w
    env.phase_object_start_height[env_ids] = object_start_pos_w[:, 2]
    env.final_task_success[env_ids] = False
    env.phase_bonus_claimed[env_ids] = False
    env.phase_task_reward_event[env_ids] = 0.0
    env.phase_drop_event[env_ids] = False
    env.phase_retry_pending[env_ids] = False
    env.phase_retry_count[env_ids] = 0
    env.phase_stable_hold_steps[env_ids] = 0
    env.phase_hold_lost_steps[env_ids] = 0
    env.phase_stable_hold[env_ids] = False
    env.phase_stable_contact_steps[env_ids] = 0
    env.phase_stable_contact[env_ids] = False
    env.phase_contact_reward_claimed[env_ids] = False
    env.phase_best_hand_distance[env_ids] = float("inf")
    env.phase_best_cage_distance[env_ids] = float("inf")
    env.phase_best_close_contact_score[env_ids] = 0.0
    env.phase_best_pre_lift_fraction[env_ids] = 0.0
    env.phase_pre_lift_escrow[env_ids] = 0.0
    env.phase_pre_lift_escrow_steps[env_ids] = 0
    env.phase_best_lift_fraction[env_ids] = 0.0
    env.phase_best_staging_distance[env_ids] = float("inf")
    env.phase_best_bin_distance[env_ids] = float("inf")


def reset_task_phase_attempt(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Finish a non-terminal retry without clearing episode-level reward latches."""
    ensure_task_phase_buffers(env)
    env.task_phase[env_ids] = GRASP_PHASE
    env.phase_retry_pending[env_ids] = False
    env.phase_stable_hold_steps[env_ids] = 0
    env.phase_hold_lost_steps[env_ids] = 0
    env.phase_stable_hold[env_ids] = False
    env.phase_stable_contact_steps[env_ids] = 0
    env.phase_stable_contact[env_ids] = False
    env.phase_pre_lift_escrow[env_ids] = 0.0
    env.phase_pre_lift_escrow_steps[env_ids] = 0


def _filtered_box_contact_count(
    env: ManagerBasedRLEnv,
    box_contact_sensor_names: tuple[str, ...],
    contact_force_threshold: float,
) -> torch.Tensor:
    """Return the number of fingertip sensors that currently touch only the box."""
    count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for sensor_name in box_contact_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        forces = sensor.data.force_matrix_w
        if forces is None:
            raise RuntimeError(
                f"'{sensor_name}' must use filter_prim_paths_expr so Box-only contact forces are available."
            )
        force_norm = torch.linalg.vector_norm(forces, dim=-1).flatten(start_dim=1).amax(dim=-1)
        count += (force_norm >= contact_force_threshold).long()
    return count


def _bimanual_box_contact_counts(
    env: ManagerBasedRLEnv,
    left_box_contact_sensor_names: tuple[str, ...],
    right_box_contact_sensor_names: tuple[str, ...],
    contact_force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Box-only fingertip-contact counts for the left and right hands.

    The aggregate six-fingertip count alone cannot distinguish a real
    two-handed grasp from two fingers on the same hand.  Keep the two groups
    explicit rather than inferring sides from body indices, so the task stays
    correct for robot-asset variants with a different body ordering.
    """
    if not left_box_contact_sensor_names or not right_box_contact_sensor_names:
        raise ValueError(
            "Bimanual grasp metrics require non-empty left and right Box-contact sensor name tuples."
        )
    return (
        _filtered_box_contact_count(env, left_box_contact_sensor_names, contact_force_threshold),
        _filtered_box_contact_count(env, right_box_contact_sensor_names, contact_force_threshold),
    )


def _body_ids_from_cfg(robot: Articulation, body_cfg: SceneEntityCfg | None) -> list[int]:
    """Resolve configured body ids without assuming a particular config lifecycle."""
    if body_cfg is None:
        return []
    if body_cfg.body_names is not None:
        body_ids, _ = robot.find_bodies(body_cfg.body_names, preserve_order=True)
        return [int(body_id) for body_id in body_ids]

    body_ids = body_cfg.body_ids
    if isinstance(body_ids, slice):
        return list(range(robot.num_bodies))[body_ids]
    if isinstance(body_ids, int):
        return [body_ids]
    return [int(body_id) for body_id in body_ids]


def _resolve_grasp_body_ids(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    box_contact_sensor_names: tuple[str, ...],
    hand_body_cfg: SceneEntityCfg | None,
    robot_cfg: SceneEntityCfg,
    *,
    require_fingertips: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Box-sensor fingertip ids and fingertip-plus-palm distance ids.

    The hand's fixed palm links are merged into the wrist links by the G1 URDF
    conversion.  ``hand_body_cfg`` therefore contributes the available palm
    proxy, while the body names owned by the filtered Box sensors provide the
    six actual fingertip links.  Resolving them here keeps the task predicate
    tied to the bodies that can genuinely report a Box contact.
    """
    proxy_ids = _body_ids_from_cfg(robot, hand_body_cfg)
    cache_key = (robot_cfg.name, tuple(box_contact_sensor_names), tuple(proxy_ids))
    cache = getattr(env, "_phase_grasp_body_ids_cache", None)
    if cache is None:
        cache = {}
        env._phase_grasp_body_ids_cache = cache
    if cache_key in cache:
        fingertip_ids, distance_ids = cache[cache_key]
    else:
        fingertip_names: list[str] = []
        for sensor_name in box_contact_sensor_names:
            sensor: ContactSensor = env.scene.sensors[sensor_name]
            fingertip_names.extend(sensor.body_names)
        fingertip_ids, _ = robot.find_bodies(fingertip_names, preserve_order=True) if fingertip_names else ([], [])

        # Preserve order while removing duplicates: the contact sensor list
        # can contain aliases in variants of the robot asset.
        fingertip_ids = list(dict.fromkeys(int(body_id) for body_id in fingertip_ids))
        distance_ids = list(dict.fromkeys([*fingertip_ids, *proxy_ids]))
        if not distance_ids:
            raise RuntimeError(
                "No grasp-distance bodies resolved. Configure a palm/wrist proxy or filtered Box fingertip sensors."
            )
        fingertip_ids = torch.tensor(fingertip_ids, dtype=torch.long, device=env.device)
        distance_ids = torch.tensor(distance_ids, dtype=torch.long, device=env.device)
        cache[cache_key] = (fingertip_ids, distance_ids)

    if require_fingertips and fingertip_ids.numel() == 0:
        raise RuntimeError(
            "A fingertip-based grasp metric requires non-empty 'box_contact_sensor_names' with robot link sensors."
        )
    return fingertip_ids, distance_ids


def _hand_object_distances(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    obj: RigidObject,
    box_contact_sensor_names: tuple[str, ...],
    hand_body_cfg: SceneEntityCfg | None,
    robot_cfg: SceneEntityCfg,
    *,
    require_fingertips: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-fingertip and minimum fingertip/palm-proxy Box distances."""
    fingertip_ids, distance_ids = _resolve_grasp_body_ids(
        env,
        robot,
        box_contact_sensor_names,
        hand_body_cfg,
        robot_cfg,
        require_fingertips=require_fingertips,
    )
    object_pos_w = obj.data.root_pos_w.unsqueeze(1)
    distance_positions_w = robot.data.body_pos_w[:, distance_ids, :]
    hand_distance = torch.linalg.vector_norm(distance_positions_w - object_pos_w, dim=-1).amin(dim=-1)

    if fingertip_ids.numel() == 0:
        # Only legacy reach shaping may omit fingertip sensors.  It never uses
        # this value for a contact/cage decision.
        fingertip_distances = hand_distance.unsqueeze(-1)
    else:
        fingertip_positions_w = robot.data.body_pos_w[:, fingertip_ids, :]
        fingertip_distances = torch.linalg.vector_norm(fingertip_positions_w - object_pos_w, dim=-1)
    return fingertip_distances, hand_distance


def _bimanual_fingertip_distances(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    obj: RigidObject,
    left_box_contact_sensor_names: tuple[str, ...],
    right_box_contact_sensor_names: tuple[str, ...],
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-fingertip object distances separately for both hands."""
    left_fingertip_distances, _ = _hand_object_distances(
        env,
        robot,
        obj,
        left_box_contact_sensor_names,
        None,
        robot_cfg,
        require_fingertips=True,
    )
    right_fingertip_distances, _ = _hand_object_distances(
        env,
        robot,
        obj,
        right_box_contact_sensor_names,
        None,
        robot_cfg,
        require_fingertips=True,
    )
    return left_fingertip_distances, right_fingertip_distances


def _nth_nearest_fingertip_distance(fingertip_distances: torch.Tensor, min_fingertips: int) -> torch.Tensor:
    """Distance of the ``min_fingertips``-th closest fingertip to the Box."""
    k = min(max(1, min_fingertips), fingertip_distances.shape[-1])
    return torch.topk(fingertip_distances, k=k, largest=False, dim=-1).values[:, -1]


def _supported_upright_stance(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg | None,
    min_foot_contact_force: float,
    min_supported_feet: int | None = None,
) -> torch.Tensor:
    """Return an upright, high-enough supported-base predicate.

    A three-frame maximum force history is used rather than a single contact
    flag.  It matches the existing feet-slide reward's contact convention and
    avoids flickering the manipulation reward on a one-physics-step foot
    sensor transition.
    """
    base_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    base_height_e = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    supported = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if feet_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
        contact_force = torch.linalg.vector_norm(
            contact_sensor.data.net_forces_w_history[:, :, feet_sensor_cfg.body_ids, :], dim=-1
        ).amax(dim=1)
        in_contact = contact_force >= min_foot_contact_force
        # Manipulation shaping uses the conservative default of both feet.
        # Pre-grasp locomotion deliberately asks for only one supported foot,
        # because a normal walking step has a single-support interval.
        supported = (
            torch.all(in_contact, dim=-1)
            if min_supported_feet is None
            else torch.sum(in_contact, dim=-1) >= min_supported_feet
        )
    return (base_tilt <= max_base_tilt_angle) & (base_height_e >= min_base_height) & supported


def grasp_stance_valid(
    env: ManagerBasedRLEnv,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg | None,
    min_foot_contact_force: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the strict stability gate used by all positive GRASP shaping."""
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    return (env.task_phase == GRASP_PHASE) & _supported_upright_stance(
        env,
        robot,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
    )


def grasp_walk_stance_valid(
    env: ManagerBasedRLEnv,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg | None,
    min_foot_contact_force: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the safe one-foot-support gate used while walking to pre-grasp.

    This is intentionally separate from :func:`grasp_stance_valid`: arm and
    lift shaping still require double support, while the camera-derived
    pre-grasp command must survive normal single-support gait frames.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    return (env.task_phase == GRASP_PHASE) & _supported_upright_stance(
        env,
        robot,
        max_base_tilt_angle,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        min_supported_feet=1,
    )


def _safe_pregrasp_state(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    robot: Articulation,
    max_base_tilt_angle: float,
    max_pregrasp_object_displacement: float,
    max_pregrasp_object_speed: float,
    max_pregrasp_lift_height: float,
    min_base_height: float = 0.0,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
) -> torch.Tensor:
    """Gate pre-contact shaping so pushing, tossing, and falling do not pay."""
    ensure_task_phase_buffers(env)
    object_displacement = torch.linalg.vector_norm(obj.data.root_pos_w - env.phase_object_start_pos, dim=-1)
    object_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    lift_height = obj.data.root_pos_w[:, 2] - env.phase_object_start_height
    return (
        (env.task_phase == GRASP_PHASE)
        & ~env.phase_stable_contact
        & _supported_upright_stance(
            env,
            robot,
            max_base_tilt_angle,
            min_base_height,
            feet_sensor_cfg,
            min_foot_contact_force,
        )
        & (object_displacement <= max_pregrasp_object_displacement)
        & (object_speed <= max_pregrasp_object_speed)
        & (lift_height <= max_pregrasp_lift_height)
    )


def _grasp_metrics(
    env: ManagerBasedRLEnv,
    box_contact_sensor_names: tuple[str, ...],
    left_box_contact_sensor_names: tuple[str, ...],
    right_box_contact_sensor_names: tuple[str, ...],
    contact_force_threshold: float,
    min_box_contacts: int,
    min_box_contacts_per_hand: int,
    max_hand_object_distance: float,
    min_hold_lift_height: float,
    max_relative_object_speed: float,
    max_base_tilt_angle: float,
    hand_body_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Measure box-only fingertip contact and physically plausible hand hold state."""
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    fingertip_distances, hand_distance = _hand_object_distances(
        env,
        robot,
        obj,
        box_contact_sensor_names,
        hand_body_cfg,
        robot_cfg,
        require_fingertips=True,
    )
    box_contact_count = _filtered_box_contact_count(env, box_contact_sensor_names, contact_force_threshold)
    left_box_contact_count, right_box_contact_count = _bimanual_box_contact_counts(
        env,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        contact_force_threshold,
    )
    has_box_contact = box_contact_count > 0

    lift_height = obj.data.root_pos_w[:, 2] - env.phase_object_start_height
    relative_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w - robot.data.root_lin_vel_w, dim=-1)
    base_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    bimanual_contact = (
        (left_box_contact_count >= min_box_contacts_per_hand)
        & (right_box_contact_count >= min_box_contacts_per_hand)
    )
    contact_candidate = (
        (box_contact_count >= min_box_contacts)
        & bimanual_contact
        & (hand_distance <= max_hand_object_distance)
        & (relative_speed <= max_relative_object_speed)
        & (base_tilt <= max_base_tilt_angle)
    )
    hold_candidate = contact_candidate & (lift_height >= min_hold_lift_height)
    return {
        "hand_distance": hand_distance,
        "fingertip_distances": fingertip_distances,
        "box_contact_count": box_contact_count,
        "left_box_contact_count": left_box_contact_count,
        "right_box_contact_count": right_box_contact_count,
        "bimanual_contact": bimanual_contact,
        "has_box_contact": has_box_contact,
        "lift_height": lift_height,
        "relative_speed": relative_speed,
        "base_tilt": base_tilt,
        "contact_candidate": contact_candidate,
        "hold_candidate": hold_candidate,
    }


def _update_stable_hold(
    env: ManagerBasedRLEnv,
    hold_candidate: torch.Tensor,
    stable_hold_duration_s: float,
) -> torch.Tensor:
    required_steps = max(1, math.ceil(stable_hold_duration_s / env.step_dt))
    env.phase_stable_hold_steps = torch.where(
        hold_candidate,
        env.phase_stable_hold_steps + 1,
        torch.zeros_like(env.phase_stable_hold_steps),
    )
    env.phase_stable_hold = env.phase_stable_hold_steps >= required_steps
    return env.phase_stable_hold


def _update_stable_contact(
    env: ManagerBasedRLEnv,
    contact_candidate: torch.Tensor,
    stable_contact_duration_s: float,
) -> torch.Tensor:
    """Track a physically plausible, upright multi-fingertip contact before lifting."""
    required_steps = max(1, math.ceil(stable_contact_duration_s / env.step_dt))
    env.phase_stable_contact_steps = torch.where(
        contact_candidate,
        env.phase_stable_contact_steps + 1,
        torch.zeros_like(env.phase_stable_contact_steps),
    )
    env.phase_stable_contact = env.phase_stable_contact_steps >= required_steps
    return env.phase_stable_contact


def _mark_retry(env: ManagerBasedRLEnv, retry_envs: torch.Tensor) -> None:
    if not torch.any(retry_envs):
        return
    env.task_phase[retry_envs] = GRASP_PHASE
    env.phase_retry_pending[retry_envs] = True
    env.phase_retry_count[retry_envs] += 1
    env.phase_drop_event[retry_envs] = True
    env.phase_stable_hold_steps[retry_envs] = 0
    env.phase_hold_lost_steps[retry_envs] = 0
    env.phase_stable_hold[retry_envs] = False
    env.phase_stable_contact_steps[retry_envs] = 0
    env.phase_stable_contact[retry_envs] = False
    env.phase_pre_lift_escrow[retry_envs] = 0.0
    env.phase_pre_lift_escrow_steps[retry_envs] = 0


def sequential_phase_task_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    grasp_success_reward: float,
    carry_success_reward: float,
    place_success_reward: float,
    carry_command_name: str,
    dropped_object_height: float,
    stable_contact_duration_s: float,
    stable_hold_duration_s: float,
    drop_grace_duration_s: float,
    box_contact_sensor_names: tuple[str, ...],
    left_box_contact_sensor_names: tuple[str, ...],
    right_box_contact_sensor_names: tuple[str, ...],
    contact_force_threshold: float,
    min_box_contacts: int,
    min_box_contacts_per_hand: int,
    max_hand_object_distance: float,
    min_hold_lift_height: float,
    max_relative_object_speed: float,
    max_base_tilt_angle: float,
    grasp_stance_max_base_tilt_angle: float,
    grasp_stance_min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg,
    min_foot_contact_force: float,
    bin_target_pos_e: tuple[float, float, float],
    bin_inner_half_extents: tuple[float, float, float],
    hand_body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Advance the task once per control step using a stable, Box-contact hold predicate.

    Repeated attempts are allowed inside an episode, but grasp and carry bonuses
    are latched after their first legitimate transition.  On a drop, a retry is
    requested; the interval event restores the robot and box to the grasp setup.
    """
    ensure_task_phase_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    phase = env.task_phase.clone()
    env.phase_task_reward_event.zero_()
    env.phase_drop_event.zero_()

    metrics = _grasp_metrics(
        env,
        box_contact_sensor_names,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        contact_force_threshold,
        min_box_contacts,
        min_box_contacts_per_hand,
        max_hand_object_distance,
        min_hold_lift_height,
        max_relative_object_speed,
        max_base_tilt_angle,
        hand_body_cfg,
        robot_cfg,
        object_cfg,
    )
    # A carry can legitimately use single support while walking.  In contrast,
    # an arm-reaching GRASP must qualify its first real contact from a stable,
    # double-supported stance.  This prevents an unstable reach from latching
    # a contact/hold state or advancing the phase bonus.
    grasp_stance = grasp_stance_valid(
        env,
        grasp_stance_max_base_tilt_angle,
        grasp_stance_min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
        robot_cfg,
    )
    grasp_phase = phase == GRASP_PHASE
    contact_candidate = metrics["contact_candidate"] & (~grasp_phase | grasp_stance)
    hold_candidate = metrics["hold_candidate"] & (~grasp_phase | grasp_stance)
    manipulation_phase = (phase == GRASP_PHASE) | (phase == CARRY_PHASE)
    _update_stable_contact(
        env,
        manipulation_phase & contact_candidate,
        stable_contact_duration_s,
    )
    stable_hold = _update_stable_hold(
        env,
        manipulation_phase & hold_candidate,
        stable_hold_duration_s,
    )

    lifted = metrics["lift_height"] >= lift_height
    grasp_succeeded = (phase == GRASP_PHASE) & stable_hold & lifted
    new_grasp = grasp_succeeded & ~env.phase_bonus_claimed[:, GRASP_PHASE]
    env.phase_task_reward_event[new_grasp] += grasp_success_reward
    env.phase_bonus_claimed[new_grasp, GRASP_PHASE] = True
    env.task_phase[grasp_succeeded] = CARRY_PHASE

    carry_command = env.command_manager.get_term(carry_command_name)
    carry_succeeded = (phase == CARRY_PHASE) & stable_hold & carry_command.at_staging
    new_carry = carry_succeeded & ~env.phase_bonus_claimed[:, CARRY_PHASE]
    env.phase_task_reward_event[new_carry] += carry_success_reward
    env.phase_bonus_claimed[new_carry, CARRY_PHASE] = True
    env.task_phase[carry_succeeded] = PLACE_PHASE

    # Contact loss must persist briefly, otherwise a single noisy contact frame
    # would reset a valid carry.  PLACE deliberately permits releasing the box.
    grace_steps = max(1, math.ceil(drop_grace_duration_s / env.step_dt))
    carry_phase = phase == CARRY_PHASE
    env.phase_hold_lost_steps = torch.where(
        carry_phase & ~metrics["hold_candidate"],
        env.phase_hold_lost_steps + 1,
        torch.zeros_like(env.phase_hold_lost_steps),
    )
    carry_failed = carry_phase & (env.phase_hold_lost_steps >= grace_steps)

    bin_target_pos_w = environment_point_w(env, bin_target_pos_e, "bin_center_e")
    bin_half_extents = torch.tensor(bin_inner_half_extents, device=env.device)
    delivered = torch.all(torch.abs(obj.data.root_pos_w - bin_target_pos_w) < bin_half_extents, dim=-1)
    place_succeeded = (phase == PLACE_PHASE) & delivered
    new_place = place_succeeded & ~env.phase_bonus_claimed[:, PLACE_PHASE]
    env.phase_task_reward_event[new_place] += place_success_reward
    env.phase_bonus_claimed[new_place, PLACE_PHASE] = True
    env.task_phase[place_succeeded] = COMPLETE_PHASE
    env.final_task_success[place_succeeded] = True

    object_height_e = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    place_failed = (phase == PLACE_PHASE) & ~delivered & (object_height_e < dropped_object_height)
    _mark_retry(env, carry_failed | place_failed)
    return env.phase_task_reward_event


def phase_drop_penalty(env: ManagerBasedRLEnv, penalty: float) -> torch.Tensor:
    """One-shot penalty paired with a retry-producing carry/place drop."""
    ensure_task_phase_buffers(env)
    return -penalty * env.phase_drop_event.float()


def grasp_approach_progress(
    env: ManagerBasedRLEnv,
    max_hand_object_distance: float,
    max_base_tilt_angle: float,
    hand_body_cfg: SceneEntityCfg,
    box_contact_sensor_names: tuple[str, ...] = (),
    left_box_contact_sensor_names: tuple[str, ...] = (),
    right_box_contact_sensor_names: tuple[str, ...] = (),
    max_pregrasp_object_displacement: float = 0.05,
    max_pregrasp_object_speed: float = 0.50,
    max_pregrasp_lift_height: float = 0.03,
    min_base_height: float = 0.0,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Monotonic bimanual fingertip reach progress while the Box is stationary.

    The farther hand is the potential bottleneck, so a single reaching hand
    cannot consume the bimanual approach reward by itself.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    left_fingertip_distances, right_fingertip_distances = _bimanual_fingertip_distances(
        env,
        robot,
        obj,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        robot_cfg,
    )
    # The farther hand is the bottleneck.  Rewarding the overall nearest hand
    # would reproduce the observed one-handed reach local optimum.
    distance = torch.maximum(
        left_fingertip_distances.amin(dim=-1), right_fingertip_distances.amin(dim=-1)
    )
    active = _safe_pregrasp_state(
        env,
        obj,
        robot,
        max_base_tilt_angle,
        max_pregrasp_object_displacement,
        max_pregrasp_object_speed,
        max_pregrasp_lift_height,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
    ) & (distance <= max_hand_object_distance)
    initialized = torch.isfinite(env.phase_best_hand_distance)
    improvement = torch.where(
        active & initialized,
        torch.clamp(env.phase_best_hand_distance - distance, min=0.0),
        torch.zeros_like(distance),
    )
    # Update the baseline throughout GRASP, including unsafe frames.  This
    # deliberately burns reach progress made while falling so it cannot be
    # paid after the robot later recovers an upright stance.
    env.phase_best_hand_distance = torch.where(
        env.task_phase == GRASP_PHASE,
        torch.minimum(env.phase_best_hand_distance, distance),
        env.phase_best_hand_distance,
    )
    return improvement / env.step_dt


def grasp_cage_progress(
    env: ManagerBasedRLEnv,
    cage_distance: float,
    max_base_tilt_angle: float,
    box_contact_sensor_names: tuple[str, ...],
    left_box_contact_sensor_names: tuple[str, ...],
    right_box_contact_sensor_names: tuple[str, ...],
    min_cage_fingertips: int = 2,
    min_cage_fingertips_per_hand: int = 1,
    max_pregrasp_object_displacement: float = 0.05,
    max_pregrasp_object_speed: float = 0.50,
    max_pregrasp_lift_height: float = 0.03,
    min_base_height: float = 0.0,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward monotonic two-handed fingertip enclosure before Box contact.

    The potential is the farther hand's configured closest fingertip.  It
    cannot be collected by repeatedly opening/closing or moving the Box: only
    an upright, nearly stationary Box in the grasp phase is eligible and each
    improvement is latched by the per-episode best potential.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    left_fingertip_distances, right_fingertip_distances = _bimanual_fingertip_distances(
        env,
        robot,
        obj,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        robot_cfg,
    )
    cage_metric = torch.maximum(
        _nth_nearest_fingertip_distance(left_fingertip_distances, min_cage_fingertips_per_hand),
        _nth_nearest_fingertip_distance(right_fingertip_distances, min_cage_fingertips_per_hand),
    )
    active = _safe_pregrasp_state(
        env,
        obj,
        robot,
        max_base_tilt_angle,
        max_pregrasp_object_displacement,
        max_pregrasp_object_speed,
        max_pregrasp_lift_height,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
    ) & (cage_metric <= cage_distance)
    initialized = torch.isfinite(env.phase_best_cage_distance)
    improvement = torch.where(
        active & initialized,
        torch.clamp(env.phase_best_cage_distance - cage_metric, min=0.0),
        torch.zeros_like(cage_metric),
    )
    env.phase_best_cage_distance = torch.where(
        env.task_phase == GRASP_PHASE,
        torch.minimum(env.phase_best_cage_distance, cage_metric),
        env.phase_best_cage_distance,
    )
    return improvement / env.step_dt


def grasp_close_progress(
    env: ManagerBasedRLEnv,
    cage_distance: float,
    contact_force_threshold: float,
    min_box_contacts: int,
    max_base_tilt_angle: float,
    box_contact_sensor_names: tuple[str, ...],
    left_box_contact_sensor_names: tuple[str, ...],
    right_box_contact_sensor_names: tuple[str, ...],
    min_box_contacts_per_hand: int = 1,
    min_cage_fingertips: int = 2,
    min_cage_fingertips_per_hand: int = 1,
    max_pregrasp_object_displacement: float = 0.05,
    max_pregrasp_object_speed: float = 0.50,
    max_pregrasp_lift_height: float = 0.03,
    min_base_height: float = 0.0,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-way pre-grasp signal for accumulating real fingertip contacts.

    This deliberately uses Box-filtered force sensors rather than finger joint
    angles: closing an empty hand earns nothing.  The score is the lower of
    the left and right hand contact scores, so it rises only when both hands
    make near, safe contact.  It is latched so collision/release cycles cannot
    farm it.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    left_fingertip_distances, right_fingertip_distances = _bimanual_fingertip_distances(
        env,
        robot,
        obj,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        robot_cfg,
    )
    cage_metric = torch.maximum(
        _nth_nearest_fingertip_distance(left_fingertip_distances, min_cage_fingertips_per_hand),
        _nth_nearest_fingertip_distance(right_fingertip_distances, min_cage_fingertips_per_hand),
    )
    safe = _safe_pregrasp_state(
        env,
        obj,
        robot,
        max_base_tilt_angle,
        max_pregrasp_object_displacement,
        max_pregrasp_object_speed,
        max_pregrasp_lift_height,
        min_base_height,
        feet_sensor_cfg,
        min_foot_contact_force,
    ) & (cage_metric <= cage_distance)
    left_contact_count, right_contact_count = _bimanual_box_contact_counts(
        env,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        contact_force_threshold,
    )
    # Both hands must close on the Box before this score can reach one.
    contact_score = torch.minimum(
        torch.clamp(left_contact_count, max=max(1, min_box_contacts_per_hand)).float()
        / max(1, min_box_contacts_per_hand),
        torch.clamp(right_contact_count, max=max(1, min_box_contacts_per_hand)).float()
        / max(1, min_box_contacts_per_hand),
    )
    improvement = torch.where(
        safe,
        torch.clamp(contact_score - env.phase_best_close_contact_score, min=0.0),
        torch.zeros_like(contact_score),
    )
    env.phase_best_close_contact_score = torch.where(
        env.task_phase == GRASP_PHASE,
        torch.maximum(env.phase_best_close_contact_score, contact_score),
        env.phase_best_close_contact_score,
    )
    return improvement / env.step_dt


def grasp_pre_lift_progress(
    env: ManagerBasedRLEnv,
    pre_lift_height: float = 0.04,
    cage_distance: float = 0.42,
    box_contact_sensor_names: tuple[str, ...] = (),
    left_box_contact_sensor_names: tuple[str, ...] = (),
    right_box_contact_sensor_names: tuple[str, ...] = (),
    hand_body_cfg: SceneEntityCfg | None = None,
    contact_force_threshold: float = 0.25,
    min_box_contacts: int = 2,
    min_box_contacts_per_hand: int = 1,
    min_cage_fingertips: int = 2,
    min_cage_fingertips_per_hand: int = 1,
    max_hand_object_distance: float = 0.36,
    max_relative_object_speed: float = 0.30,
    max_object_speed: float = 0.50,
    max_horizontal_object_displacement: float = 0.05,
    max_base_tilt_angle: float = 0.60,
    min_base_height: float = 0.0,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    min_foot_contact_force: float = 1.0,
    pre_lift_escrow_duration_s: float = 0.25,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a safe, low-force grasp for its first 0--``pre_lift_height`` of lift.

    This is deliberately a bridge *before* the strict stable-contact predicate:
    one Box-filtered fingertip from each hand must make low-force contact while
    enclosing the object, and the hands, object, and base must all be physically quiet.  The
    absolute-speed and horizontal-displacement gates make a tossed or pushed
    Box ineligible even if it passes close to the hand.  Progress is a
    per-episode monotonic potential, so a lift/drop/re-lift cycle cannot
    repeatedly collect it.  The strict 1 N stable-contact/0.12 m transition is
    untouched and uses its own lift potential.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    left_fingertip_distances, right_fingertip_distances = _bimanual_fingertip_distances(
        env,
        robot,
        obj,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        robot_cfg,
    )
    _, hand_distance = _hand_object_distances(
        env,
        robot,
        obj,
        box_contact_sensor_names,
        hand_body_cfg,
        robot_cfg,
        require_fingertips=True,
    )
    cage_metric = torch.maximum(
        _nth_nearest_fingertip_distance(left_fingertip_distances, min_cage_fingertips_per_hand),
        _nth_nearest_fingertip_distance(right_fingertip_distances, min_cage_fingertips_per_hand),
    )
    left_contact_count, right_contact_count = _bimanual_box_contact_counts(
        env,
        left_box_contact_sensor_names,
        right_box_contact_sensor_names,
        contact_force_threshold,
    )
    contact_count = _filtered_box_contact_count(env, box_contact_sensor_names, contact_force_threshold)

    lift_height = obj.data.root_pos_w[:, 2] - env.phase_object_start_height
    relative_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w - robot.data.root_lin_vel_w, dim=-1)
    object_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    horizontal_displacement = torch.linalg.vector_norm(
        obj.data.root_pos_w[:, :2] - env.phase_object_start_pos[:, :2], dim=-1
    )
    safe_low_force_grasp = (
        (contact_count >= min_box_contacts)
        & (left_contact_count >= min_box_contacts_per_hand)
        & (right_contact_count >= min_box_contacts_per_hand)
        & (cage_metric <= cage_distance)
        & (hand_distance <= max_hand_object_distance)
        & (relative_speed <= max_relative_object_speed)
        & (object_speed <= max_object_speed)
        & (horizontal_displacement <= max_horizontal_object_displacement)
        & grasp_stance_valid(
            env,
            max_base_tilt_angle,
            min_base_height,
            feet_sensor_cfg,
            min_foot_contact_force,
            robot_cfg,
        )
    )
    lift_fraction = torch.clamp(lift_height / max(pre_lift_height, 1.0e-6), min=0.0, max=1.0)
    improvement = torch.clamp(lift_fraction - env.phase_best_pre_lift_fraction, min=0.0)
    # Burn all uplift made in an invalid GRASP state.  Only the progress seen
    # during a continuously safe low-force grasp can enter the escrow.
    env.phase_best_pre_lift_fraction = torch.where(
        env.task_phase == GRASP_PHASE,
        torch.maximum(env.phase_best_pre_lift_fraction, lift_fraction),
        env.phase_best_pre_lift_fraction,
    )
    # Strict contact normally becomes true after 0.10 s.  An escrow already
    # opened before that point may keep maturing, but it cannot be started
    # after strict contact takes over the later lift reward.
    can_start_escrow = safe_low_force_grasp & ~env.phase_stable_contact
    can_continue_escrow = safe_low_force_grasp & (env.phase_pre_lift_escrow_steps > 0)
    escrow_active = can_start_escrow | can_continue_escrow
    escrow_steps = torch.where(
        escrow_active,
        env.phase_pre_lift_escrow_steps + 1,
        torch.zeros_like(env.phase_pre_lift_escrow_steps),
    )
    escrow_value = torch.where(
        escrow_active,
        env.phase_pre_lift_escrow + improvement,
        torch.zeros_like(env.phase_pre_lift_escrow),
    )
    required_steps = max(1, math.ceil(pre_lift_escrow_duration_s / env.step_dt))
    payout_ready = escrow_active & (escrow_steps >= required_steps)
    payout = torch.where(payout_ready, escrow_value, torch.zeros_like(escrow_value))
    env.phase_pre_lift_escrow = torch.where(payout_ready, torch.zeros_like(escrow_value), escrow_value)
    env.phase_pre_lift_escrow_steps = torch.where(
        payout_ready,
        torch.zeros_like(escrow_steps),
        escrow_steps,
    )
    return payout / env.step_dt


def grasp_box_contact_reward(
    env: ManagerBasedRLEnv,
    contact_reward: float,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg,
    min_foot_contact_force: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-time reward for a stable, upright multi-fingertip contact qualification."""
    ensure_task_phase_buffers(env)
    qualified = (
        (env.task_phase == GRASP_PHASE)
        & env.phase_stable_contact
        & ~env.phase_contact_reward_claimed
        & grasp_stance_valid(
            env,
            max_base_tilt_angle,
            min_base_height,
            feet_sensor_cfg,
            min_foot_contact_force,
            robot_cfg,
        )
    )
    env.phase_contact_reward_claimed[qualified] = True
    # RewardManager multiplies by dt.  Divide here so the configured value is
    # an actual one-shot reward rather than a single-frame fraction of one.
    return qualified.float() * (contact_reward / env.step_dt)


def grasp_lift_progress(
    env: ManagerBasedRLEnv,
    lift_height: float,
    max_base_tilt_angle: float,
    min_base_height: float,
    feet_sensor_cfg: SceneEntityCfg,
    min_foot_contact_force: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Monotonic lift progress after stable upright multi-fingertip contact.

    ``phase_stable_contact`` is updated once by ``sparse_task`` earlier in the
    RewardManager ordering.  This intentionally prevents a one-frame collision
    or a falling body from opening the lift reward.
    """
    ensure_task_phase_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    lift_fraction = torch.clamp((obj.data.root_pos_w[:, 2] - env.phase_object_start_height) / lift_height, 0.0, 1.0)
    grasp_active = (
        (env.task_phase == GRASP_PHASE)
        & env.phase_stable_contact
        & grasp_stance_valid(
            env,
            max_base_tilt_angle,
            min_base_height,
            feet_sensor_cfg,
            min_foot_contact_force,
            robot_cfg,
        )
    )
    # CARRY keeps its former predicate: normal walking may briefly use one
    # foot and must not make a valid hold/lift reward flicker off.
    carry_active = (env.task_phase == CARRY_PHASE) & env.phase_stable_contact
    active = grasp_active | carry_active
    improvement = torch.where(
        active,
        torch.clamp(lift_fraction - env.phase_best_lift_fraction, min=0.0),
        torch.zeros_like(lift_fraction),
    )
    # GRASP lift made while unsupported is intentionally not recoverable as a
    # later reward.  CARRY retains its original, stable-hold-only potential.
    update_best = (env.task_phase == GRASP_PHASE) | carry_active
    env.phase_best_lift_fraction = torch.where(
        update_best,
        torch.maximum(env.phase_best_lift_fraction, lift_fraction),
        env.phase_best_lift_fraction,
    )
    return improvement / env.step_dt


def carry_hold_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward a stable bimanual Box hold only while navigating in CARRY.

    ``phase_stable_hold`` is bimanual because the phase predicate updates it
    from one Box-filtered fingertip contact on each hand.  A one-handed grasp
    therefore receives neither this survival reward nor the navigation terms.
    """
    ensure_task_phase_buffers(env)
    return ((env.task_phase == CARRY_PHASE) & env.phase_stable_hold).float()


def carry_staging_progress(env: ManagerBasedRLEnv, carry_command_name: str) -> torch.Tensor:
    """Monotonic camera-command staging progress while the box is stably held."""
    ensure_task_phase_buffers(env)
    command = env.command_manager.get_term(carry_command_name)
    staging_distance = torch.linalg.vector_norm(command.position_error_b, dim=-1)
    active = (env.task_phase == CARRY_PHASE) & env.phase_stable_hold & command.bin_visible
    initialized = torch.isfinite(env.phase_best_staging_distance)
    improvement = torch.where(
        active & initialized,
        torch.clamp(env.phase_best_staging_distance - staging_distance, min=0.0),
        torch.zeros_like(staging_distance),
    )
    env.phase_best_staging_distance = torch.where(
        active,
        torch.minimum(env.phase_best_staging_distance, staging_distance),
        env.phase_best_staging_distance,
    )
    return improvement / env.step_dt


def place_bin_progress(
    env: ManagerBasedRLEnv,
    bin_target_pos_e: tuple[float, float, float],
    max_base_tilt_angle: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Monotonic object-to-bin progress after staging, before final placement."""
    ensure_task_phase_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    target_pos_w = environment_point_w(env, bin_target_pos_e, "bin_center_e")
    distance = torch.linalg.vector_norm(obj.data.root_pos_w - target_pos_w, dim=-1)
    robot: Articulation = env.scene[robot_cfg.name]
    base_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    active = (env.task_phase == PLACE_PHASE) & (base_tilt <= max_base_tilt_angle)
    initialized = torch.isfinite(env.phase_best_bin_distance)
    improvement = torch.where(
        active & initialized,
        torch.clamp(env.phase_best_bin_distance - distance, min=0.0),
        torch.zeros_like(distance),
    )
    env.phase_best_bin_distance = torch.where(
        active,
        torch.minimum(env.phase_best_bin_distance, distance),
        env.phase_best_bin_distance,
    )
    return improvement / env.step_dt


def final_task_succeeded(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the latched Phase-3 completion flag used for termination."""
    ensure_task_phase_buffers(env)
    return env.final_task_success
