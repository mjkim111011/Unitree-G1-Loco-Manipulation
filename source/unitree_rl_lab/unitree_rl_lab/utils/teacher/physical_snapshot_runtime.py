"""Isaac Lab adapter for :mod:`physical_snapshots`.

Import this module only after the Isaac Lab application has been launched.
The on-disk schema itself remains simulator independent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .physical_snapshots import PhysicalSnapshot, ROLLOUT_PROVENANCE, SnapshotKind
from .student_observation import STUDENT_FEATURE_SLICES, validate_student_observation


def restore_policy_history_from_snapshot(
    env: Any,
    snapshot: PhysicalSnapshot,
    *,
    env_id: int = 0,
) -> None:
    """Restore the actor's exact five-frame proprioceptive history.

    Isaac Lab does not expose a public observation-history deserializer.  The
    snapshot nevertheless contains the exact flattened 460-D group, so for the
    single-environment teacher/evaluation runtimes we reconstruct each term's
    circular buffer in chronological order and verify the manager emits the
    same vector before continuing physics.
    """

    if env.num_envs != 1 or env_id != 0:
        raise ValueError(
            "physical snapshot history restore currently requires the single-env teacher runtime"
        )
    manager = env.observation_manager
    group_name = "policy_history"
    history_buffers = manager._group_obs_term_history_buffer.get(group_name, {})
    term_names = manager._group_obs_term_names.get(group_name, ())
    if not term_names or set(term_names) != set(history_buffers):
        raise RuntimeError("policy_history term buffers are unavailable or incomplete")
    if any(buffer._buffer is None for buffer in history_buffers.values()):
        manager.compute_group(group_name, update_history=True)

    observation = validate_student_observation(snapshot.student_observation)
    feature = STUDENT_FEATURE_SLICES["policy_history"]
    saved = torch.as_tensor(
        observation[feature.start : feature.stop],
        dtype=torch.float32,
        device=env.device,
    )
    offset = 0
    for term_name in term_names:
        buffer = history_buffers[term_name]
        chronological = buffer.buffer[env_id]
        width = int(chronological.numel())
        if offset + width > saved.numel():
            raise RuntimeError("saved policy history is shorter than manager layout")
        restored = saved[offset : offset + width].reshape(chronological.shape)
        offset += width
        buffer.reset()
        for frame in restored:
            buffer.append(frame.unsqueeze(0))
    if offset != saved.numel():
        raise RuntimeError(
            f"saved policy history has {saved.numel() - offset} unconsumed values"
        )
    emitted = manager.compute_group(group_name, update_history=False)[env_id]
    if emitted.shape != saved.shape or not torch.allclose(
        emitted, saved, atol=1.0e-6, rtol=0.0
    ):
        raise RuntimeError("restored policy history does not match the saved 460-D state")


def capture_physical_snapshot(
    env: Any,
    *,
    kind: SnapshotKind,
    env_id: int,
    source_episode: str,
    source_step: int,
    student_observation: Any,
    box_initial_height_m: float,
    left_contact_count: int,
    right_contact_count: int,
    stable_hold_steps: int,
    at_staging: bool,
    bad_orientation: bool,
    box_dropped: bool,
    box_size: Any,
    box_mass_kg: float,
    unprocessed_wbc_raw_action: Any | None = None,
    robot_name: str = "robot",
    box_name: str = "box",
) -> PhysicalSnapshot:
    """Capture a complete, realized robot/Box state after a physics step."""

    if env_id < 0 or env_id >= env.num_envs:
        raise IndexError(f"env_id {env_id} is outside [0, {env.num_envs})")
    robot = env.scene[robot_name]
    box = env.scene[box_name]
    box_state = box.data.root_link_state_w[env_id]
    lift_height = float(box_state[2].item()) - float(box_initial_height_m)
    wbc_raw_action = unprocessed_wbc_raw_action
    if wbc_raw_action is not None and hasattr(wbc_raw_action, "detach"):
        wbc_raw_action = wbc_raw_action.detach().cpu().numpy()
    table_asset_names = ("table_top", "table_pedestal")
    bin_asset_names = ("bin_bottom", "bin_front", "bin_back", "bin_left", "bin_right")

    def scene_has(name: str) -> bool:
        try:
            env.scene[name]
        except (KeyError, IndexError):
            return False
        return True

    has_work_area_assets = hasattr(env.scene, "env_origins") and all(
        scene_has(name) for name in (*table_asset_names, *bin_asset_names)
    )

    def asset_poses(names: tuple[str, ...]) -> np.ndarray:
        return np.stack(
            [
                env.scene[name].data.root_link_state_w[env_id, :7]
                .detach()
                .cpu()
                .numpy()
                .copy()
                for name in names
            ]
        )

    table_center_e = None
    bin_center_e = None
    bin_qr_position_e = None
    table_asset_root_poses_w = None
    bin_asset_root_poses_w = None
    if has_work_area_assets:
        env_origin = env.scene.env_origins[env_id]
        table_center_e = (
            env.table_center_e[env_id]
            if hasattr(env, "table_center_e")
            else torch.tensor((0.62, 0.0, 0.74), device=env.device)
        )
        bin_center_e = (
            env.bin_center_e[env_id]
            if hasattr(env, "bin_center_e")
            else env.scene["bin_bottom"].data.root_link_pos_w[env_id]
            - env_origin
            + torch.tensor((0.0, 0.0, 0.25), device=env.device)
        )
        bin_qr_position_e = (
            env.bin_qr_position_e[env_id]
            if hasattr(env, "bin_qr_position_e")
            else bin_center_e + torch.tensor((-0.386, 0.0, 0.02), device=env.device)
        )
        table_asset_root_poses_w = asset_poses(table_asset_names)
        bin_asset_root_poses_w = asset_poses(bin_asset_names)
    snapshot = PhysicalSnapshot(
        kind=SnapshotKind(kind),
        robot_root_link_state_w=robot.data.root_link_state_w[env_id]
        .detach()
        .cpu()
        .numpy()
        .copy(),
        robot_joint_position=robot.data.joint_pos[env_id].detach().cpu().numpy().copy(),
        robot_joint_velocity=robot.data.joint_vel[env_id].detach().cpu().numpy().copy(),
        box_root_link_state_w=box_state.detach().cpu().numpy().copy(),
        last_action=env.action_manager.action[env_id].detach().cpu().numpy().copy(),
        student_observation=validate_student_observation(student_observation).copy(),
        task_phase=int(env.task_phase[env_id].item()),
        source_episode=source_episode,
        source_step=int(source_step),
        source_env_id=int(env_id),
        provenance=ROLLOUT_PROVENANCE,
        left_contact_count=int(left_contact_count),
        right_contact_count=int(right_contact_count),
        lift_height_m=lift_height,
        stable_hold_steps=int(stable_hold_steps),
        at_staging=bool(at_staging),
        bad_orientation=bool(bad_orientation),
        box_dropped=bool(box_dropped),
        box_size=np.asarray(box_size, dtype=np.float32).copy(),
        box_mass_kg=float(box_mass_kg),
        unprocessed_wbc_raw_action=(
            None
            if wbc_raw_action is None
            else np.asarray(wbc_raw_action, dtype=np.float32).copy()
        ),
        table_center_e=(
            None if table_center_e is None else table_center_e.detach().cpu().numpy().copy()
        ),
        bin_center_e=(
            None if bin_center_e is None else bin_center_e.detach().cpu().numpy().copy()
        ),
        bin_qr_position_e=(
            None
            if bin_qr_position_e is None
            else bin_qr_position_e.detach().cpu().numpy().copy()
        ),
        table_asset_root_poses_w=table_asset_root_poses_w,
        bin_asset_root_poses_w=bin_asset_root_poses_w,
    )
    return snapshot.validate(
        joint_count=robot.num_joints,
        action_dim=env.action_manager.total_action_dim,
    )


def restore_physical_snapshot(
    env: Any,
    snapshot: PhysicalSnapshot,
    *,
    env_id: int = 0,
    expected_box_size: Any | None = None,
    expected_box_mass_kg: float | None = None,
    robot_name: str = "robot",
    box_name: str = "box",
) -> None:
    """Restore one provenance-checked physical state into a matching scene.

    The Box and robot root/joint velocities are restored together.  A caller
    must run one normal controlled physics step and re-check bilateral contact
    before accepting the state for training.
    """

    robot = env.scene[robot_name]
    box = env.scene[box_name]
    snapshot.validate(
        joint_count=robot.num_joints,
        action_dim=env.action_manager.total_action_dim,
    )
    if expected_box_size is not None and not np.allclose(
        np.asarray(expected_box_size), snapshot.box_size, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("snapshot Box size does not match the fixed current scene")
    if expected_box_mass_kg is not None and not np.isclose(
        expected_box_mass_kg, snapshot.box_mass_kg, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("snapshot Box mass does not match the fixed current scene")
    if env_id < 0 or env_id >= env.num_envs:
        raise IndexError(f"env_id {env_id} is outside [0, {env.num_envs})")

    device = env.device
    ids = torch.tensor([env_id], dtype=torch.long, device=device)
    robot_root = torch.as_tensor(
        snapshot.robot_root_link_state_w, device=device
    ).unsqueeze(0)
    joint_pos = torch.as_tensor(snapshot.robot_joint_position, device=device).unsqueeze(
        0
    )
    joint_vel = torch.as_tensor(snapshot.robot_joint_velocity, device=device).unsqueeze(
        0
    )
    box_root = torch.as_tensor(snapshot.box_root_link_state_w, device=device).unsqueeze(
        0
    )

    robot.write_root_link_state_to_sim(robot_root, env_ids=ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    box.write_root_link_state_to_sim(box_root, env_ids=ids)

    if snapshot.table_center_e is not None:
        table_asset_names = ("table_top", "table_pedestal")
        bin_asset_names = (
            "bin_bottom",
            "bin_front",
            "bin_back",
            "bin_left",
            "bin_right",
        )
        zero_velocity = torch.zeros((1, 6), device=device)
        for name, pose in zip(table_asset_names, snapshot.table_asset_root_poses_w):
            asset = env.scene[name]
            asset.write_root_pose_to_sim(
                torch.as_tensor(pose, device=device).unsqueeze(0), env_ids=ids
            )
            asset.write_root_velocity_to_sim(zero_velocity, env_ids=ids)
        for name, pose in zip(bin_asset_names, snapshot.bin_asset_root_poses_w):
            asset = env.scene[name]
            asset.write_root_pose_to_sim(
                torch.as_tensor(pose, device=device).unsqueeze(0), env_ids=ids
            )
            asset.write_root_velocity_to_sim(zero_velocity, env_ids=ids)
        if not hasattr(env, "table_center_e"):
            env.table_center_e = torch.zeros((env.num_envs, 3), device=device)
            env.bin_center_e = torch.zeros((env.num_envs, 3), device=device)
            env.bin_qr_position_e = torch.zeros((env.num_envs, 3), device=device)
        env.table_center_e[env_id] = torch.as_tensor(snapshot.table_center_e, device=device)
        env.bin_center_e[env_id] = torch.as_tensor(snapshot.bin_center_e, device=device)
        env.bin_qr_position_e[env_id] = torch.as_tensor(
            snapshot.bin_qr_position_e, device=device
        )

    # Synchronize last-action observations and the action term's processed
    # target.  Preserve every non-target environment in vectorized scenes.
    actions = env.action_manager.action.clone()
    actions[env_id] = torch.as_tensor(
        snapshot.last_action, dtype=actions.dtype, device=device
    )
    env.action_manager.process_action(actions)
    env.action_manager._prev_action[env_id] = actions[env_id]
    if hasattr(env, "task_phase"):
        env.task_phase[env_id] = int(snapshot.task_phase)
    restore_policy_history_from_snapshot(env, snapshot, env_id=env_id)
    env.sim.forward()


def restored_snapshot_gate(
    snapshot: PhysicalSnapshot,
    *,
    left_contact_count: int,
    right_contact_count: int,
    lift_height_m: float,
    bad_orientation: bool,
    box_dropped: bool,
    at_staging: bool | None = None,
    lift_threshold_m: float = 0.12,
) -> bool:
    """Validate a restored snapshot after one ordinary controlled sim step."""

    base = (
        left_contact_count >= 1
        and right_contact_count >= 1
        and lift_height_m >= lift_threshold_m
        and not bad_orientation
        and not box_dropped
    )
    if snapshot.kind == SnapshotKind.CARRY_AT_STAGING:
        return base and bool(at_staging)
    return base
