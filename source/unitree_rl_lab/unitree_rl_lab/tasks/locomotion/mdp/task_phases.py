from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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


def _nth_nearest_fingertip_distance(fingertip_distances: torch.Tensor, min_fingertips: int) -> torch.Tensor:
    """Distance of the ``min_fingertips``-th closest fingertip to the Box."""
    k = min(max(1, min_fingertips), fingertip_distances.shape[-1])
    return torch.topk(fingertip_distances, k=k, largest=False, dim=-1).values[:, -1]


def _safe_pregrasp_state(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    robot: Articulation,
    max_base_tilt_angle: float,
    max_pregrasp_object_displacement: float,
    max_pregrasp_object_speed: float,
    max_pregrasp_lift_height: float,
) -> torch.Tensor:
    """Gate pre-contact shaping so pushing, tossing, and falling do not pay."""
    ensure_task_phase_buffers(env)
    base_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    object_displacement = torch.linalg.vector_norm(obj.data.root_pos_w - env.phase_object_start_pos, dim=-1)
    object_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    lift_height = obj.data.root_pos_w[:, 2] - env.phase_object_start_height
    return (
        (env.task_phase == GRASP_PHASE)
        & ~env.phase_stable_contact
        & (base_tilt <= max_base_tilt_angle)
        & (object_displacement <= max_pregrasp_object_displacement)
        & (object_speed <= max_pregrasp_object_speed)
        & (lift_height <= max_pregrasp_lift_height)
    )


def _grasp_metrics(
    env: ManagerBasedRLEnv,
    box_contact_sensor_names: tuple[str, ...],
    contact_force_threshold: float,
    min_box_contacts: int,
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
    has_box_contact = box_contact_count > 0

    lift_height = obj.data.root_pos_w[:, 2] - env.phase_object_start_height
    relative_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w - robot.data.root_lin_vel_w, dim=-1)
    base_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    contact_candidate = (
        (box_contact_count >= min_box_contacts)
        & (hand_distance <= max_hand_object_distance)
        & (relative_speed <= max_relative_object_speed)
        & (base_tilt <= max_base_tilt_angle)
    )
    hold_candidate = contact_candidate & (lift_height >= min_hold_lift_height)
    return {
        "hand_distance": hand_distance,
        "fingertip_distances": fingertip_distances,
        "box_contact_count": box_contact_count,
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
    contact_force_threshold: float,
    min_box_contacts: int,
    max_hand_object_distance: float,
    min_hold_lift_height: float,
    max_relative_object_speed: float,
    max_base_tilt_angle: float,
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
        contact_force_threshold,
        min_box_contacts,
        max_hand_object_distance,
        min_hold_lift_height,
        max_relative_object_speed,
        max_base_tilt_angle,
        hand_body_cfg,
        robot_cfg,
        object_cfg,
    )
    manipulation_phase = (phase == GRASP_PHASE) | (phase == CARRY_PHASE)
    _update_stable_contact(
        env,
        manipulation_phase & metrics["contact_candidate"],
        stable_contact_duration_s,
    )
    stable_hold = _update_stable_hold(
        env,
        manipulation_phase & metrics["hold_candidate"],
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

    bin_target_pos_w = env.scene.env_origins + torch.tensor(bin_target_pos_e, device=env.device)
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
    max_pregrasp_object_displacement: float = 0.05,
    max_pregrasp_object_speed: float = 0.50,
    max_pregrasp_lift_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Monotonic fingertip/palm-proxy reach progress while the Box is stationary.

    ``box_contact_sensor_names`` should name the six filtered fingertip sensors.
    Supplying it replaces the old wrist-only distance with the minimum over
    those fingertips plus ``hand_body_cfg``'s palm/wrist proxy.  The empty
    default preserves compatibility with legacy configurations.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    _, distance = _hand_object_distances(
        env,
        robot,
        obj,
        box_contact_sensor_names,
        hand_body_cfg,
        robot_cfg,
        require_fingertips=False,
    )
    active = _safe_pregrasp_state(
        env,
        obj,
        robot,
        max_base_tilt_angle,
        max_pregrasp_object_displacement,
        max_pregrasp_object_speed,
        max_pregrasp_lift_height,
    ) & (distance <= max_hand_object_distance)
    initialized = torch.isfinite(env.phase_best_hand_distance)
    improvement = torch.where(
        active & initialized,
        torch.clamp(env.phase_best_hand_distance - distance, min=0.0),
        torch.zeros_like(distance),
    )
    env.phase_best_hand_distance = torch.where(
        active,
        torch.minimum(env.phase_best_hand_distance, distance),
        env.phase_best_hand_distance,
    )
    return improvement / env.step_dt


def grasp_cage_progress(
    env: ManagerBasedRLEnv,
    cage_distance: float,
    max_base_tilt_angle: float,
    box_contact_sensor_names: tuple[str, ...],
    min_cage_fingertips: int = 2,
    max_pregrasp_object_displacement: float = 0.05,
    max_pregrasp_object_speed: float = 0.50,
    max_pregrasp_lift_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward monotonic approach of multiple fingertips before Box contact.

    The potential is the distance of the second-nearest (or configured
    ``min_cage_fingertips``-nearest) fingertip.  It cannot be collected by
    repeatedly opening/closing or moving the Box: only an upright, nearly
    stationary Box in the grasp phase is eligible and each improvement is
    latched by the per-episode best potential.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    fingertip_distances, _ = _hand_object_distances(
        env,
        robot,
        obj,
        box_contact_sensor_names,
        None,
        robot_cfg,
        require_fingertips=True,
    )
    cage_metric = _nth_nearest_fingertip_distance(fingertip_distances, min_cage_fingertips)
    active = _safe_pregrasp_state(
        env,
        obj,
        robot,
        max_base_tilt_angle,
        max_pregrasp_object_displacement,
        max_pregrasp_object_speed,
        max_pregrasp_lift_height,
    ) & (cage_metric <= cage_distance)
    initialized = torch.isfinite(env.phase_best_cage_distance)
    improvement = torch.where(
        active & initialized,
        torch.clamp(env.phase_best_cage_distance - cage_metric, min=0.0),
        torch.zeros_like(cage_metric),
    )
    env.phase_best_cage_distance = torch.where(
        active,
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
    min_cage_fingertips: int = 2,
    max_pregrasp_object_displacement: float = 0.05,
    max_pregrasp_object_speed: float = 0.50,
    max_pregrasp_lift_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-way pre-grasp signal for accumulating real fingertip contacts.

    This deliberately uses Box-filtered force sensors rather than finger joint
    angles: closing an empty hand earns nothing.  The score rises only as the
    number of near, safe fingertip contacts increases, is capped at the strict
    hold-contact count, and is latched so collision/release cycles cannot farm
    it.  It does not alter the stable-contact or sparse-success predicate.
    """
    ensure_task_phase_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    fingertip_distances, _ = _hand_object_distances(
        env,
        robot,
        obj,
        box_contact_sensor_names,
        None,
        robot_cfg,
        require_fingertips=True,
    )
    cage_metric = _nth_nearest_fingertip_distance(fingertip_distances, min_cage_fingertips)
    safe = _safe_pregrasp_state(
        env,
        obj,
        robot,
        max_base_tilt_angle,
        max_pregrasp_object_displacement,
        max_pregrasp_object_speed,
        max_pregrasp_lift_height,
    ) & (cage_metric <= cage_distance)
    contact_count = _filtered_box_contact_count(env, box_contact_sensor_names, contact_force_threshold)
    capped_contacts = torch.clamp(contact_count, max=max(1, min_box_contacts)).float()
    contact_score = capped_contacts / max(1, min_box_contacts)
    improvement = torch.where(
        safe,
        torch.clamp(contact_score - env.phase_best_close_contact_score, min=0.0),
        torch.zeros_like(contact_score),
    )
    env.phase_best_close_contact_score = torch.where(
        safe,
        torch.maximum(env.phase_best_close_contact_score, contact_score),
        env.phase_best_close_contact_score,
    )
    return improvement / env.step_dt


def grasp_box_contact_reward(
    env: ManagerBasedRLEnv,
    contact_reward: float,
) -> torch.Tensor:
    """One-time reward for a stable, upright multi-fingertip contact qualification."""
    ensure_task_phase_buffers(env)
    qualified = (
        (env.task_phase == GRASP_PHASE)
        & env.phase_stable_contact
        & ~env.phase_contact_reward_claimed
    )
    env.phase_contact_reward_claimed[qualified] = True
    # RewardManager multiplies by dt.  Divide here so the configured value is
    # an actual one-shot reward rather than a single-frame fraction of one.
    return qualified.float() * (contact_reward / env.step_dt)


def grasp_lift_progress(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Monotonic lift progress after stable upright multi-fingertip contact.

    ``phase_stable_contact`` is updated once by ``sparse_task`` earlier in the
    RewardManager ordering.  This intentionally prevents a one-frame collision
    or a falling body from opening the lift reward.
    """
    ensure_task_phase_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    lift_fraction = torch.clamp((obj.data.root_pos_w[:, 2] - env.phase_object_start_height) / lift_height, 0.0, 1.0)
    active = (
        ((env.task_phase == GRASP_PHASE) | (env.task_phase == CARRY_PHASE))
        & env.phase_stable_contact
    )
    improvement = torch.where(
        active,
        torch.clamp(lift_fraction - env.phase_best_lift_fraction, min=0.0),
        torch.zeros_like(lift_fraction),
    )
    env.phase_best_lift_fraction = torch.where(
        active,
        torch.maximum(env.phase_best_lift_fraction, lift_fraction),
        env.phase_best_lift_fraction,
    )
    return improvement / env.step_dt


def carry_hold_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward stable Box-contact hold only while navigating in the carry phase."""
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
    target_pos_w = env.scene.env_origins + torch.tensor(bin_target_pos_e, device=env.device)
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
