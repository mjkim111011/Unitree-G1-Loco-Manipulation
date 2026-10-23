"""Feedback placement: align the held Box over the Bin, open, and withdraw.

This simulator diagnostic uses measured object geometry and live Pink/Homie.
It neither changes physical state nor produces BC training demonstrations.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .phase_teachers import PlaceTeacher, RealizedBimanualGrasp, pose_matrix_wxyz


def box_fits_opening(box_pose, box_size, bin_center, bin_half_extents, margin=0.02):
    """Require the whole oriented Box footprint to clear the inner Bin walls."""
    transform = pose_matrix_wxyz(np.asarray(box_pose))
    half = np.abs(transform[:3, :3]) @ (np.asarray(box_size) / 2)
    offset = np.abs(transform[:2, 3] - np.asarray(bin_center)[:2])
    return bool(np.all(offset + half[:2] + margin < np.asarray(bin_half_extents)[:2]))


def run_rule_based_place(env, bridge, *, bin_center, bin_half_extents, box_size,
                         max_seconds=16.0, source_episode=None, step_env=None):
    from .arena_teacher_runtime import (
        action_order_target, body_id, body_pose_wxyz, box_contact_counts, bridge_request,
    )
    robot, box = env.scene["robot"], env.scene["box"]
    pelvis = body_id(robot, "pelvis")
    left_wrist, right_wrist = (body_id(robot, f"{side}_wrist_yaw_link") for side in ("left", "right"))
    term = env.action_manager.get_term("JointPositionAction")
    names = term._joint_names
    arms = [i for i, name in enumerate(names) if any(part in name for part in ("shoulder", "elbow", "wrist"))]
    fingers = [i for i, name in enumerate(names) if "_hand_" in name]
    reference = env.action_manager.action.detach().clone()
    start_pose = box.data.root_link_state_w[0, :7].detach().cpu().numpy().copy()
    grasp = RealizedBimanualGrasp.capture(
        box_pose_wxyz=start_pose,
        left_wrist_pose_wxyz=body_pose_wxyz(robot, left_wrist).detach().cpu().numpy(),
        right_wrist_pose_wxyz=body_pose_wxyz(robot, right_wrist).detach().cpu().numpy(),
    )
    teacher = PlaceTeacher(grasp, start_box_position_w=start_pose[:3], bin_center_w=np.asarray(bin_center))
    target = start_pose[:3].copy()
    goal = np.r_[np.asarray(bin_center)[:2], max(float(start_pose[2]), 0.70)]
    initial_rotation = pose_matrix_wxyz(robot.data.root_link_state_w[0, :7].detach().cpu().numpy())[:3, :3]
    initial_yaw = math.atan2(initial_rotation[1, 0], initial_rotation[0, 0])

    def proposal(position, release=False, withdraw=0.0, navigation=None, compression=0.008):
        pelvis_pose = body_pose_wxyz(robot, pelvis).detach().cpu().numpy()
        high_level = teacher.action_for_box_position(
            box_position=position, release=release,
            current_pelvis_pose_wxyz=pelvis_pose,
            left_recovery_compression_m=0.0 if release else compression,
            right_recovery_compression_m=0.0 if release else compression,
            camera_navigation_velocity=(np.zeros(3, dtype=np.float32) if navigation is None else navigation),
        )
        if withdraw:
            # Positions are in the pelvis frame: separate laterally, then retract.
            high_level[3] += 0.12 * withdraw
            high_level[10] -= 0.12 * withdraw
            high_level[2] -= 0.08 * withdraw
            high_level[9] -= 0.08 * withdraw
        ordered_names, values = bridge.step(bridge_request(env, high_level))
        return action_order_target(env, ordered_names, values)[0]

    # Seed the controller at the realized wrists without a frozen-physics wait.
    baseline = proposal(target, compression=0.0)
    arm_offset = reference[:, arms] - baseline[:, arms]
    state = "align"
    state_steps = stable_alignment = settled = contact_loss = 0
    released = False
    trace = []
    filtered_velocity = box.data.root_link_lin_vel_w[0].detach().cpu().numpy().copy()
    reason = "alignment_timeout"
    success = False
    max_steps = math.ceil(max_seconds / env.step_dt)
    print("[place-rule] ALIGN with live balance", flush=True)
    for step in range(max_steps):
        env.task_phase[0] = 2
        env.command_manager.compute(0.0)
        pose = box.data.root_link_state_w[0, :7].detach().cpu().numpy()
        position = pose[:3]
        speed = float(torch.linalg.vector_norm(box.data.root_link_lin_vel_w[0]))
        filtered_velocity = 0.8 * filtered_velocity + 0.2 * box.data.root_link_lin_vel_w[0].detach().cpu().numpy()
        fits = box_fits_opening(pose, box_size, bin_center, bin_half_extents)
        if state in ("align", "settle"):
            # Move both wrists together, preserving their captured grasp geometry.
            delta = goal - target
            wrist_speed = 0.08 if state == "align" else 0.04
            target += delta * min(1.0, wrist_speed * env.step_dt / max(float(np.linalg.norm(delta)), 1e-9))
            # The geometric goal is clearance of all four walls, rather than
            # an arbitrary distance from the exact centre. Account for drift
            # during the fall as well as the current rotated footprint.
            predicted = pose.copy()
            fall_time = math.sqrt(2 * max(0.0, float(position[2] - bin_center[2])) / 9.81)
            # Contact impulses cause rapid velocity sign flips while the held
            # object is stationary. Estimate drift over several control frames.
            predicted[:2] += filtered_velocity[:2] * fall_time
            landing_fits = box_fits_opening(predicted, box_size, bin_center, bin_half_extents)
            half_height = (np.abs(pose_matrix_wxyz(pose)[:3, :3]) @ (np.asarray(box_size) / 2))[2]
            above_rim = position[2] - half_height > bin_center[2] + bin_half_extents[2] + 0.07
            approach_fits = box_fits_opening(pose, box_size, bin_center, bin_half_extents, margin=0.03)
            aligned = (approach_fits and above_rim) if state == "align" else (fits and landing_fits and above_rim and np.linalg.norm(filtered_velocity) < 0.12)
            stable_alignment = stable_alignment + 1 if aligned else 0
            if state == "align" and stable_alignment >= 5:
                state, state_steps, stable_alignment = "settle", 0, 0
                target = position.copy()
                reference[:, arms] = env.action_manager.action[:, arms].detach().clone()
                arm_offset = reference[:, arms] - proposal(target)[:, arms]
                print("[place-rule] SETTLE base over Bin", flush=True)
            elif state == "settle" and state_steps >= 25 and stable_alignment >= 5:
                state, state_steps = "open", 0
                released = True
                target = position.copy()
                reference[:, arms] = env.action_manager.action[:, arms].detach().clone()
                arm_offset = reference[:, arms] - proposal(target)[:, arms]
                print("[place-rule] OPEN fingers over Bin", flush=True)
            elif state == "settle" and state_steps >= 100 and not fits:
                state, state_steps, stable_alignment = "align", 0, 0
                print("[place-rule] RE-ALIGN after stopping drift", flush=True)
        opening = (1.0 if state == "withdraw" else min(1.0, (state_steps + 1) / 30.0)) if released else 0.0
        withdraw = min(1.0, (state_steps + 1) / 60.0) if state == "withdraw" else 0.0
        navigation = np.zeros(3, dtype=np.float32)
        if state == "align":
            # Staging is outside comfortable arm reach. Follow the held object
            # slowly with the base rather than stretching the arms indefinitely.
            world_error = goal[:2] - position[:2]
            pelvis_transform = pose_matrix_wxyz(body_pose_wxyz(robot, pelvis).detach().cpu().numpy())
            velocity = pelvis_transform[:2, :2].T @ world_error * 0.8
            velocity *= min(1.0, 0.12 / max(float(np.linalg.norm(velocity)), 1e-9))
            navigation[:2] = velocity
        root_rotation = pose_matrix_wxyz(robot.data.root_link_state_w[0, :7].detach().cpu().numpy())[:3, :3]
        root_yaw = math.atan2(root_rotation[1, 0], root_rotation[0, 0])
        yaw_error = math.atan2(math.sin(initial_yaw - root_yaw), math.cos(initial_yaw - root_yaw))
        navigation[2] = np.clip(2.0 * yaw_error, -0.3, 0.3)
        action = proposal(target, release=released, withdraw=withdraw, navigation=navigation)
        action[:, arms] += arm_offset
        if not released:
            reach_blend = min(1.0, max(0.0, (step - 50) / 200.0)) if state == "align" else 1.0
            action[:, arms] = reference[:, arms] + reach_blend * (action[:, arms] - reference[:, arms])
            action[:, fingers] = reference[:, fingers]
        else:
            action[:, fingers] = reference[:, fingers] * (1.0 - opening) + action[:, fingers] * opening
        if step < 25:
            # Arm/finger targets already match the loaded reference at entry.
            fraction = (step + 1) / 25.0
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            lower = [i for i in range(len(names)) if i not in arms and i not in fingers]
            action[:, lower] = reference[:, lower] + blend * (action[:, lower] - reference[:, lower])
        (env if step_env is None else step_env).step(action)
        env.task_phase[0] = 2
        env.command_manager.compute(0.0)
        pose = box.data.root_link_state_w[0, :7].detach().cpu().numpy()
        position = pose[:3]
        speed = float(torch.linalg.vector_norm(box.data.root_link_lin_vel_w[0]))
        left, right = box_contact_counts(env)
        tilt = float(torch.acos(torch.clamp(-robot.data.projected_gravity_b[0, 2], -1.0, 1.0)))
        contact_loss = contact_loss + 1 if not (left and right) else 0
        inside = bool(np.all(np.abs(position - np.asarray(bin_center)) < np.asarray(bin_half_extents)))
        settled = settled + 1 if released and inside and left == 0 and right == 0 and speed < 0.15 else 0
        if step % 10 == 0:
            trace.append({"step": step, "state": state, "box": position.tolist(),
                          "box_pose": pose.tolist(), "fits_opening": fits,
                          "target": target.tolist(), "contacts": [left, right], "tilt": tilt,
                          "speed": speed, "root_height": float(robot.data.root_link_pos_w[0, 2]),
                          "root_position": robot.data.root_link_pos_w[0].detach().cpu().tolist()})
        if step % 100 == 0:
            print(f"[place-rule] step={step} state={state} xy_error={np.linalg.norm(position[:2] - np.asarray(bin_center)[:2]):.3f}m contacts={left}/{right}", flush=True)
        if tilt > 0.8:
            reason = "bad_orientation"
            break
        if position[2] < 0.15 and not inside:
            reason = "box_drop_outside_bin"
            break
        if not released and contact_loss > 10:
            reason = "grasp_lost_before_release"
            break
        state_steps += 1
        if state == "open" and state_steps >= 30:
            state, state_steps = "withdraw", 0
            print("[place-rule] WITHDRAW arms", flush=True)
        if settled >= 10 and state == "withdraw" and state_steps >= 60:
            success, reason = True, "place_success"
            break
        if released:
            reason = "release_did_not_settle"
    return {"success": success, "reason": reason, "recorded": False,
            "steps": step + 1, "source_episode": source_episode,
            "controller": "rule_based_place_v1", "release_commanded": released,
            "final_box_position": position.tolist(), "trace": trace}
