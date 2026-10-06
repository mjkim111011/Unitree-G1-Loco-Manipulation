"""Behavior-cloning nominal controller for residual reinforcement learning.

The PPO actor emits only a bounded correction.  Four frozen 43-D behavior-
cloning policies produce absolute joint targets from the same 517-D student
observation, and this wrapper converts those targets to the environment's raw
``JointPositionAction`` coordinates before adding the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import gymnasium as gym
import torch

from unitree_rl_lab.utils.teacher.phase_bc import (
    LoadedBCPolicy,
    load_bc_policy,
    validate_bc2_gate,
    validate_runtime_action_contract,
)
from unitree_rl_lab.utils.teacher.skill_fsm import SkillPhase


# Kept local so the pure residual-composition helpers remain importable without
# launching Isaac Sim.  These are the same physical fingertip links used by
# the teacher recorder and the strict grasp gate.
LEFT_FINGERTIP_NAMES = (
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
)
RIGHT_FINGERTIP_NAMES = (
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
)


@dataclass(frozen=True)
class ResidualNominalCfg:
    """Runtime bounds for the frozen nominal controller."""

    residual_scale: float = 0.20
    handoff_steps: int = 100
    maximum_tip_distance_m: float = 0.18
    maximum_cage_midpoint_error_m: float = 0.11
    maximum_wrist_midpoint_error_m: float = 0.222
    maximum_wrist_distance_asymmetry_m: float = 0.10

    def validate(self) -> "ResidualNominalCfg":
        values = (
            self.residual_scale,
            self.maximum_tip_distance_m,
            self.maximum_cage_midpoint_error_m,
            self.maximum_wrist_midpoint_error_m,
            self.maximum_wrist_distance_asymmetry_m,
        )
        if self.residual_scale <= 0.0 or self.handoff_steps <= 0 or any(value <= 0.0 for value in values[1:]):
            raise ValueError("residual scale, handoff steps, and geometry limits must be positive")
        return self


def compose_residual_action(
    nominal_raw_action: torch.Tensor,
    residual_action: torch.Tensor,
    *,
    residual_scale: float,
) -> torch.Tensor:
    """Return ``nominal + scale * clip(residual, -1, 1)`` with strict shapes."""

    if nominal_raw_action.shape != residual_action.shape or nominal_raw_action.ndim != 2:
        raise ValueError("nominal and residual actions must have the same (N, A) shape")
    if residual_scale <= 0.0:
        raise ValueError("residual_scale must be positive")
    if not torch.isfinite(nominal_raw_action).all() or not torch.isfinite(residual_action).all():
        raise ValueError("nominal and residual actions must be finite")
    return nominal_raw_action + float(residual_scale) * residual_action.clamp(-1.0, 1.0)


def smoothstep_handoff(
    proposed: torch.Tensor,
    reference: torch.Tensor,
    steps_since_transition: torch.Tensor,
    *,
    handoff_steps: int,
) -> torch.Tensor:
    """Vectorized inclusive smoothstep blend from the previous phase target."""

    if proposed.shape != reference.shape or proposed.ndim != 2:
        raise ValueError("proposed and reference targets must have matching (N, A) shapes")
    if steps_since_transition.shape != (proposed.shape[0],):
        raise ValueError("steps_since_transition must have shape (N,)")
    if handoff_steps <= 0:
        raise ValueError("handoff_steps must be positive")
    denominator = max(1, handoff_steps - 1)
    fraction = (steps_since_transition.to(proposed.dtype) / denominator).clamp(0.0, 1.0)
    alpha = fraction * fraction * (3.0 - 2.0 * fraction)
    return reference + alpha[:, None] * (proposed - reference)


class BcResidualActionWrapper(gym.Wrapper):
    """Run frozen four-skill BC nominal actions underneath a PPO residual."""

    def __init__(
        self,
        env: gym.Env,
        *,
        checkpoints: Mapping[SkillPhase, str | Path],
        bc2_gate: str | Path,
        device: str | torch.device,
        cfg: ResidualNominalCfg | None = None,
    ) -> None:
        super().__init__(env)
        self.cfg = (cfg or ResidualNominalCfg()).validate()
        self.device = torch.device(device)
        self.base_env = env.unwrapped
        self.num_envs = int(self.base_env.num_envs)
        self.action_dim = int(self.base_env.action_manager.total_action_dim)
        if self.action_dim != 43:
            raise ValueError(f"BC residual controller requires 43 actions, got {self.action_dim}")

        required = {SkillPhase.APPROACH, SkillPhase.GRASP_LIFT, SkillPhase.CARRY, SkillPhase.PLACE}
        if set(checkpoints) != required:
            raise ValueError("exactly four approach/grasp/carry/place checkpoints are required")
        self.checkpoint_paths = {phase: Path(path).expanduser().resolve() for phase, path in checkpoints.items()}
        self.policies: dict[SkillPhase, LoadedBCPolicy] = {
            phase: load_bc_policy(path, self.device) for phase, path in self.checkpoint_paths.items()
        }
        raw_joint_names = tuple(
            self.base_env.action_manager.get_term("JointPositionAction")._joint_names
        )
        for phase, policy in self.policies.items():
            if policy.skill_phase != phase:
                raise ValueError(f"{self.checkpoint_paths[phase]} has the wrong skill phase")
            if policy.action_key != "processed_43d_joint_target" or policy.action_dim != self.action_dim:
                raise ValueError("residual nominal policies must all output processed 43-D joint targets")
            validate_runtime_action_contract(policy, raw_joint_names)
        gate = validate_bc2_gate(bc2_gate, required_action_key="processed_43d_joint_target")
        if self.checkpoint_paths[SkillPhase.GRASP_LIFT] != Path(gate["checkpoint_path"]).expanduser().resolve():
            raise ValueError("the residual grasp policy is not the checkpoint certified by the BC2 gate")

        action_term = self.base_env.action_manager.get_term("JointPositionAction")
        self._offset = torch.as_tensor(
            action_term._offset, dtype=torch.float32, device=self.device
        )
        self._scale = torch.as_tensor(
            action_term._scale, dtype=torch.float32, device=self.device
        )
        if torch.any(self._scale == 0.0):
            raise ValueError("JointPositionAction scale must be non-zero")
        self._left_tip_ids = self._body_ids(LEFT_FINGERTIP_NAMES)
        self._right_tip_ids = self._body_ids(RIGHT_FINGERTIP_NAMES)
        self._left_wrist_id = self._body_ids(("left_wrist_yaw_link",))[0]
        self._right_wrist_id = self._body_ids(("right_wrist_yaw_link",))[0]
        self._grasp_subphase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._selected_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_task_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._handoff_reference = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._handoff_step = torch.full(
            (self.num_envs,), self.cfg.handoff_steps, dtype=torch.long, device=self.device
        )
        self._last_nominal_raw = torch.zeros_like(self._handoff_reference)
        self._last_observation: dict[str, torch.Tensor] | None = None

    def _body_ids(self, names: tuple[str, ...]) -> torch.Tensor:
        ids, _ = self.base_env.scene["robot"].find_bodies(list(names), preserve_order=True)
        if len(ids) != len(names):
            raise RuntimeError(f"could not resolve all residual-controller bodies: {names}")
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _student_observation(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        history = observation["policy_history"]
        current = observation["policy_current"]
        value = torch.cat((history, current), dim=-1).to(self.device)
        if value.shape != (self.num_envs, 517) or not torch.isfinite(value).all():
            raise RuntimeError(f"expected finite ({self.num_envs}, 517) student observation")
        return value

    def _pregrasp_ready(self) -> torch.Tensor:
        robot = self.base_env.scene["robot"]
        box = self.base_env.scene["box"].data.root_link_pos_w
        body_pos = robot.data.body_link_pos_w
        left = body_pos[:, self._left_tip_ids]
        right = body_pos[:, self._right_tip_ids]
        left_distance = torch.linalg.vector_norm(left - box[:, None, :], dim=-1)
        right_distance = torch.linalg.vector_norm(right - box[:, None, :], dim=-1)
        row = torch.arange(self.num_envs, device=self.device)
        left_closest = left[row, left_distance.argmin(dim=1)]
        right_closest = right[row, right_distance.argmin(dim=1)]
        cage_error = torch.linalg.vector_norm(0.5 * (left_closest + right_closest) - box, dim=-1)
        left_wrist = body_pos[:, self._left_wrist_id]
        right_wrist = body_pos[:, self._right_wrist_id]
        wrist_error = torch.linalg.vector_norm(0.5 * (left_wrist + right_wrist) - box, dim=-1)
        wrist_asymmetry = torch.abs(
            torch.linalg.vector_norm(left_wrist - box, dim=-1)
            - torch.linalg.vector_norm(right_wrist - box, dim=-1)
        )
        return (
            (left_distance.amin(dim=1) <= self.cfg.maximum_tip_distance_m)
            & (right_distance.amin(dim=1) <= self.cfg.maximum_tip_distance_m)
            & (cage_error <= self.cfg.maximum_cage_midpoint_error_m)
            & (wrist_error <= self.cfg.maximum_wrist_midpoint_error_m)
            & (wrist_asymmetry <= self.cfg.maximum_wrist_distance_asymmetry_m)
        )

    def _phase_ids(self) -> torch.Tensor:
        task_phase = self.base_env.task_phase.to(self.device)
        retry = (task_phase == 0) & (self._previous_task_phase > 0)
        self._grasp_subphase[retry] = False
        self._grasp_subphase |= (task_phase == 0) & self._pregrasp_ready()
        phase = torch.where(
            task_phase == 0,
            torch.where(
                self._grasp_subphase,
                torch.full_like(task_phase, int(SkillPhase.GRASP_LIFT)),
                torch.full_like(task_phase, int(SkillPhase.APPROACH)),
            ),
            torch.where(
                task_phase == 1,
                torch.full_like(task_phase, int(SkillPhase.CARRY)),
                torch.full_like(task_phase, int(SkillPhase.PLACE)),
            ),
        )
        self._previous_task_phase.copy_(task_phase)
        return phase

    def _nominal_raw_action(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        student_obs = self._student_observation(observation)
        phase = self._phase_ids()
        absolute = torch.empty(self.num_envs, self.action_dim, device=self.device)
        with torch.inference_mode():
            for skill in (SkillPhase.APPROACH, SkillPhase.GRASP_LIFT, SkillPhase.CARRY, SkillPhase.PLACE):
                mask = phase == int(skill)
                if mask.any():
                    absolute[mask] = self.policies[skill].model(student_obs[mask])
        raw = (absolute - self._offset) / self._scale
        changed = phase != self._selected_phase
        if changed.any():
            self._handoff_reference[changed] = self._last_nominal_raw[changed]
            self._handoff_step[changed] = 0
        blended = smoothstep_handoff(
            raw,
            self._handoff_reference,
            self._handoff_step,
            handoff_steps=self.cfg.handoff_steps,
        )
        not_blending = self._handoff_step >= self.cfg.handoff_steps
        blended[not_blending] = raw[not_blending]
        self._handoff_step.add_(1).clamp_(max=self.cfg.handoff_steps)
        self._selected_phase.copy_(phase)
        self._last_nominal_raw.copy_(blended)
        return blended

    def reset(self, **kwargs):
        observation, extras = self.env.reset(**kwargs)
        self._grasp_subphase.zero_()
        self._selected_phase.zero_()
        self._previous_task_phase.zero_()
        self._handoff_reference.zero_()
        self._handoff_step.fill_(self.cfg.handoff_steps)
        self._last_nominal_raw.zero_()
        self._last_observation = observation
        return observation, extras

    def step(self, residual_action: torch.Tensor):
        if self._last_observation is None:
            raise RuntimeError("residual wrapper must be reset before stepping")
        nominal = self._nominal_raw_action(self._last_observation)
        action = compose_residual_action(
            nominal, residual_action.to(self.device), residual_scale=self.cfg.residual_scale
        )
        observation, reward, terminated, truncated, extras = self.env.step(action)
        done = terminated | truncated
        if done.any():
            self._grasp_subphase[done] = False
            self._selected_phase[done] = int(SkillPhase.APPROACH)
            self._previous_task_phase[done] = 0
            self._handoff_step[done] = self.cfg.handoff_steps
            self._last_nominal_raw[done] = 0.0
        self._last_observation = observation
        return observation, reward, terminated, truncated, extras

    def provenance(self) -> dict[str, object]:
        return {
            "schema": "unitree_box_bc_residual_nominal_v1",
            "residual_scale": self.cfg.residual_scale,
            "handoff_steps": self.cfg.handoff_steps,
            "student_observation_dim": 517,
            "nominal_action_dim": 43,
            "residual_action_dim": 43,
            "checkpoints": {phase.name.lower(): str(path) for phase, path in self.checkpoint_paths.items()},
        }
