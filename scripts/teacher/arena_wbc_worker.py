#!/usr/bin/env python3
"""Run Arena's G1 Pink + Homie controller behind a JSON-lines protocol.

This worker is intentionally executed with ``IsaacLab-Arena/.venv``.  The
BoxPickPlace environment uses an older Isaac Lab/Isaac Sim stack, so importing
both stacks in one Python process is unsupported.  Only plain numeric state,
the Arena 23-D high-level action, and the resulting named 43-D joint target
cross the process boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


PROTOCOL_PREFIX = "ARENA_WBC_JSON:"


def _emit(payload: dict[str, Any]) -> None:
    print(PROTOCOL_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _as_batch(values: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape == (width,):
        array = array[None, :]
    if array.shape != (1, width):
        raise ValueError(
            f"{name} must have shape ({width},) or (1, {width}); got {array.shape}"
        )
    return array


def _pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    transform[:3, 3] = position
    return transform


class ArenaG1Controller:
    """Small simulation-independent facade over Arena's controller classes."""

    def __init__(self) -> None:
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.config.configs import (
            HomieV2Config,
        )
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik.g1_wbc_upperbody_controller import (
            G1WBCUpperbodyController,
        )
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.wbc_policy_factory import (
            get_wbc_policy,
        )
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 import (
            instantiate_g1_robot_model,
        )

        self.robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        self.upper_body = G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=["arms"],
            extra_active_joints=[],
        )
        self.whole_body = get_wbc_policy(
            "g1", self.robot_model, HomieV2Config(), num_envs=1
        )
        self.joint_names = list(self.robot_model.joint_names)
        self._name_to_index = {
            name: index for index, name in enumerate(self.joint_names)
        }

    def reset(self) -> None:
        """Reset recurrent locomotion history and re-warm the iterative IK state."""
        import torch

        # Mirror Arena's ``reset_decoupled_wbc_pink_policy`` event.  Merely
        # setting ``in_warmup`` leaves Pink's configuration at the previous
        # episode's terminal solution and makes later demonstrations depend on
        # rollout order.
        self.upper_body.body_ik_solver.initialize()
        self.upper_body.in_warmup = True
        lower_body = getattr(self.whole_body, "lower_body_policy", None)
        if lower_body is not None and hasattr(lower_body, "reset"):
            lower_body.reset(torch.tensor([0], dtype=torch.long))

    def _reorder(
        self, values: Any, source_names: list[str], field_name: str
    ) -> np.ndarray:
        values_array = np.asarray(values, dtype=np.float32)
        if values_array.shape != (len(source_names),):
            raise ValueError(
                f"{field_name} has shape {values_array.shape}; expected ({len(source_names)},)"
            )
        if len(set(source_names)) != len(source_names):
            raise ValueError("joint_names contains duplicates")
        missing = sorted(set(self.joint_names) - set(source_names))
        extra = sorted(set(source_names) - set(self.joint_names))
        if missing or extra:
            raise ValueError(f"joint name mismatch: missing={missing}, extra={extra}")
        output = np.empty((1, len(self.joint_names)), dtype=np.float32)
        source_index = {name: index for index, name in enumerate(source_names)}
        for target_index, name in enumerate(self.joint_names):
            output[0, target_index] = values_array[source_index[name]]
        return output

    def _seed_upper_body_ik(self, joint_position: np.ndarray) -> None:
        """Warm-start Pink from the realized simulator arm configuration.

        Arena normally starts an episode from its own default posture, so its
        reset hook initializes Pink at that same posture.  Carry/place
        collection is different: it resumes a physical snapshot whose arms
        already realize a loaded bilateral grasp.  Re-initializing Pink at the
        default posture makes the first otherwise-equivalent wrist command
        choose a different redundant IK solution and can pull the fingertips
        out of the Box cage in a few control steps.

        The bridge has already reordered ``joint_position`` into Arena's exact
        43-joint model order.  Seed only the reduced arm configuration used by
        Pink and make the posture task regard that realized posture as its
        local rest target.  No Box or robot state is altered by this operation.
        """

        full_q = np.asarray(joint_position, dtype=np.float64)
        if full_q.shape != (1, len(self.joint_names)):
            raise ValueError(
                "joint_position must be reordered before seeding upper-body IK"
            )
        if self.upper_body.using_reduced_robot_model:
            seed_q = self.upper_body.body.full_to_reduced_configuration(full_q[0])
        else:
            seed_q = full_q[0].copy()
        solver = self.upper_body.body_ik_solver
        solver.configuration.q = self.upper_body.body.clip_configuration(seed_q)
        solver.configuration.update()
        self.upper_body.body.cache_forward_kinematics(solver.configuration.q)
        solver.tasks["posture"].set_target_from_configuration(solver.configuration)

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        high_level_action = np.asarray(request["high_level_action"], dtype=np.float32)
        if high_level_action.shape != (23,):
            raise ValueError(
                f"high_level_action must have shape (23,), got {high_level_action.shape}"
            )

        joint_names = list(request["joint_names"])
        joint_position = self._reorder(
            request["joint_position"], joint_names, "joint_position"
        )
        joint_velocity = self._reorder(
            request["joint_velocity"], joint_names, "joint_velocity"
        )
        default_joint_position = self._reorder(
            request["default_joint_position"], joint_names, "default_joint_position"
        )

        # Arena action layout: two binary hand states, two wrist poses in the
        # pelvis frame (position + xyzw quaternion), then vx/vy/wz, pelvis
        # height, and torso roll/pitch/yaw.
        left_pose = _pose_matrix(high_level_action[2:5], high_level_action[5:9])
        right_pose = _pose_matrix(high_level_action[9:12], high_level_action[12:16])
        body_targets = {
            "left_wrist_yaw_link": left_pose,
            "right_wrist_yaw_link": right_pose,
        }

        if self.upper_body.in_warmup:
            self._seed_upper_body_ik(joint_position)
            for _ in range(50):
                ik_target = self.upper_body.inverse_kinematics(
                    body_targets,
                    float(high_level_action[0]),
                    float(high_level_action[1]),
                )
            self.upper_body.in_warmup = False
        else:
            ik_target = self.upper_body.inverse_kinematics(
                body_targets, float(high_level_action[0]), float(high_level_action[1])
            )

        upper_indices = self.robot_model.get_joint_group_indices("upper_body")
        upper_target = np.asarray(ik_target, dtype=np.float64)[upper_indices][None, :]

        observation = {
            "q": joint_position,
            "dq": joint_velocity,
            "default_q": default_joint_position,
            "ddq": np.zeros_like(joint_position),
            "tau_est": np.zeros_like(joint_position),
            "floating_base_pose": _as_batch(
                request["floating_base_pose_wxyz"], 7, "floating_base_pose_wxyz"
            ),
            "floating_base_vel": _as_batch(
                request["floating_base_velocity_body"], 6, "floating_base_velocity_body"
            ),
            "floating_base_acc": np.zeros((1, 6), dtype=np.float32),
            "projected_gravity_b": _as_batch(
                request["projected_gravity_body"], 3, "projected_gravity_body"
            ),
            "root_ang_vel_b": _as_batch(
                request["root_angular_velocity_body"], 3, "root_angular_velocity_body"
            ),
            "torso_quat": _as_batch(
                request["torso_quaternion_wxyz"], 4, "torso_quaternion_wxyz"
            ),
            "torso_ang_vel": _as_batch(
                request["torso_angular_velocity_body"], 3, "torso_angular_velocity_body"
            ),
        }
        goal = {
            "navigate_cmd": high_level_action[16:19][None, :],
            "base_height_command": high_level_action[19:20][None, :],
            "torso_orientation_rpy_cmd": high_level_action[20:23][None, :],
        }
        self.whole_body.set_goal(goal)
        self.whole_body.set_observation(observation)
        target = self.whole_body.get_action(upper_target)["q"][0]
        return {
            "ok": True,
            "joint_names": self.joint_names,
            "joint_target": np.asarray(target, dtype=np.float32).tolist(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one request (useful for smoke tests).",
    )
    args = parser.parse_args()

    try:
        controller = ArenaG1Controller()
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "stage": "initialization",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 2

    _emit({"ok": True, "ready": True, "joint_names": controller.joint_names})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                _emit({"ok": True, "closed": True})
                return 0
            if request.get("command") == "reset":
                controller.reset()
                _emit({"ok": True, "reset": True})
                continue
            _emit(controller.step(request))
        except Exception as exc:
            _emit(
                {
                    "ok": False,
                    "stage": "step",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        if args.once:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
