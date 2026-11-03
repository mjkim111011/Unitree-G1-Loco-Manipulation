"""Learned WBC-to-CARRY transition: frozen BC3 under a PPO gate (and CoM guide).

Each training episode starts from a physical ``LOADED_STAND_END`` snapshot, i.e.
the exact state in which the live chain hands the WBC's loaded stand to BC3.
The WBC itself runs in a separate Arena process and cannot be vectorized, so -
exactly as in the fixed-schedule baseline - its contribution is the frozen
anchor: the last raw action it sent, stored in the snapshot as ``last_action``.

The PPO actor emits ``u`` (stage 1) or ``[u, dc_x, dc_y]`` (stage 2).  The
environment itself keeps its 43-D ``JointPositionAction`` so that BC3 continues
to observe the composed 43-D command as its ``last_action``, identical to the
live chain.  Only the RSL-RL wrapper exposes the low-dimensional action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from unitree_rl_lab.tasks.locomotion.mdp.task_phases import CARRY_PHASE, ensure_task_phase_buffers
from unitree_rl_lab.utils.teacher.carry_runtime_contract import validate_carry_runtime_contract
from unitree_rl_lab.utils.teacher.phase_bc import (
    LoadedBCPolicy,
    load_bc_policy,
    validate_runtime_action_contract,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import (
    SnapshotKind,
    load_snapshot_store_metadata,
    load_snapshots,
)
from unitree_rl_lab.utils.teacher.skill_fsm import SkillPhase
from unitree_rl_lab.utils.teacher.student_observation import STUDENT_FEATURE_SLICES
from unitree_rl_lab.utils.teacher.transition_gate import (
    TRANSITION_GATE_VERSION,
    TransitionGateCfg,
    com_jacobian,
    com_shift_joint_offset,
    compose_transition_action,
    update_alpha,
)


# Leg and waist joints carry the CoM guide and the BC3 preview observation.
COM_JOINT_TOKENS = ("hip", "knee", "ankle", "waist")
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
WORK_AREA_ASSETS = (
    "table_top", "table_pedestal",
    "bin_bottom", "bin_front", "bin_back", "bin_left", "bin_right",
)


@dataclass(frozen=True)
class TransitionPolicyCfg:
    stage: int = 1
    gate: TransitionGateCfg = TransitionGateCfg()
    com_damping: float = 1.0e-2

    def validate(self) -> "TransitionPolicyCfg":
        if self.stage not in (1, 2):
            raise ValueError("transition stage must be 1 (gate) or 2 (gate + CoM)")
        self.gate.validate()
        if self.com_damping <= 0.0:
            raise ValueError("com_damping must be positive")
        return self

    @property
    def action_dim(self) -> int:
        return 1 if self.stage == 1 else 3


class CarryTransitionController:
    """Per-environment transition state shared by the wrapper and the MDP terms.

    The controller attaches itself as ``env.transition_controller``; the reset
    event, observation, reward, and termination terms read it from there.
    """

    def __init__(
        self,
        env,
        *,
        carry_checkpoint: str | Path,
        snapshot_paths: Sequence[str | Path] = (),
        cfg: TransitionPolicyCfg | None = None,
    ) -> None:
        self.env = env
        self.cfg = (cfg or TransitionPolicyCfg()).validate()
        self.device = torch.device(env.device)
        self.num_envs = int(env.num_envs)
        if abs(env.step_dt - self.cfg.gate.step_dt) > 1.0e-9:
            raise ValueError(f"gate step_dt {self.cfg.gate.step_dt} != env step_dt {env.step_dt}")

        term = env.action_manager.get_term("JointPositionAction")
        if env.action_manager.total_action_dim != 43:
            raise ValueError("the transition controller requires the 43-D JointPositionAction")
        self.joint_names = tuple(term._joint_names)
        self.scale = torch.as_tensor(term._scale, dtype=torch.float32, device=self.device).expand(43).clone()
        self.offset = torch.as_tensor(term._offset, dtype=torch.float32, device=self.device).expand(
            self.num_envs, 43
        ).clone()

        self.carry_checkpoint = Path(carry_checkpoint).expanduser().resolve()
        self.bc3: LoadedBCPolicy = load_bc_policy(self.carry_checkpoint, self.device)
        if self.bc3.skill_phase != SkillPhase.CARRY or self.bc3.action_key != "processed_43d_joint_target":
            raise ValueError("carry_checkpoint must be a 43-D Carry policy")
        validate_runtime_action_contract(self.bc3, self.joint_names)
        self.contract = validate_carry_runtime_contract(
            self.bc3.carry_runtime_contract, expected_ordered_joint_names=self.joint_names
        )
        self.locked = list(self.contract["locked_raw_joint_indices"])
        self.unlocked = [i for i in range(43) if i not in set(self.locked)]
        self.navigation_ramp_steps = int(self.contract["navigation_ramp_steps"])
        self.com_action_ids = [
            i for i, name in enumerate(self.joint_names) if any(tok in name for tok in COM_JOINT_TOKENS)
        ]
        if len(self.com_action_ids) != 15:
            raise ValueError(f"expected 12 leg + 3 waist joints, got {len(self.com_action_ids)}")
        self._velocity_slice = STUDENT_FEATURE_SLICES["velocity_commands"].as_slice()

        robot = env.scene["robot"]
        self.robot = robot
        self.box = env.scene["box"]
        joint_ids = term._joint_ids
        joint_ids = list(range(robot.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        self.com_joint_ids = [joint_ids[i] for i in self.com_action_ids]
        # Floating-base PhysX Jacobians put the 6 root DoFs first.
        self._com_jacobian_columns = torch.as_tensor(
            [6 + joint_ids[i] for i in self.com_action_ids], dtype=torch.long, device=self.device
        )
        self.foot_body_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
        self.body_masses = robot.root_physx_view.get_masses().to(self.device)
        self.robot_mass = self.body_masses.sum(dim=1)

        # Training resets from snapshots; the live chain passes none and calls begin_live().
        self.pool_size = 0
        if snapshot_paths:
            self._load_snapshot_pool(snapshot_paths)

        n = self.num_envs
        self.alpha = torch.zeros(n, device=self.device)
        self.previous_alpha = torch.zeros(n, device=self.device)
        self.elapsed = torch.zeros(n, dtype=torch.long, device=self.device)
        self.anchor = torch.zeros(n, 43, device=self.device)
        self.gate_action = torch.zeros(n, 3, device=self.device)
        self.previous_gate_action = torch.zeros(n, 3, device=self.device)
        self.com_shift = torch.zeros(n, 2, device=self.device)
        self.previous_target = self.offset.clone()
        self.target_delta = torch.zeros(n, 43, device=self.device)
        self.box_rest_height = torch.zeros(n, device=self.device)
        self.staging_reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_swing_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.snapshot_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self._bc3_raw = torch.zeros(n, 43, device=self.device)
        self._bc3_raw_stamp = -1
        env.transition_controller = self

    # ------------------------------------------------------------------ snapshots
    def _load_snapshot_pool(self, paths: Sequence[str | Path]) -> None:
        snapshots = []
        origins = []
        for path in paths:
            metadata = load_snapshot_store_metadata(path)
            origin = metadata.get("source_env_origin_w", (0.0, 0.0, 0.0))
            loaded = load_snapshots(
                path, kind=SnapshotKind.LOADED_STAND_END, expected_joint_names=self.robot.data.joint_names
            )
            snapshots.extend(loaded)
            origins.extend([origin] * len(loaded))
        if not snapshots:
            raise ValueError("no LOADED_STAND_END snapshots were found")

        def stack(values) -> torch.Tensor:
            return torch.as_tensor([list(map(float, v.flatten())) for v in values], device=self.device)

        self.pool_size = len(snapshots)
        source_origin = torch.as_tensor(origins, dtype=torch.float32, device=self.device)
        self.pool_robot_root = stack([s.robot_root_link_state_w for s in snapshots])
        self.pool_robot_root[:, :3] -= source_origin
        self.pool_joint_pos = stack([s.robot_joint_position for s in snapshots])
        self.pool_joint_vel = stack([s.robot_joint_velocity for s in snapshots])
        self.pool_box_root = stack([s.box_root_link_state_w for s in snapshots])
        self.pool_box_root[:, :3] -= source_origin
        self.pool_anchor = stack([s.last_action for s in snapshots])
        self.pool_box_rest_height = torch.as_tensor(
            [float(s.box_root_link_state_w[2] - s.lift_height_m) for s in snapshots], device=self.device
        ) - source_origin[:, 2]
        self.pool_has_work_area = all(s.table_center_e is not None for s in snapshots)
        if self.pool_has_work_area:
            poses = [
                torch.as_tensor(
                    [*s.table_asset_root_poses_w.tolist(), *s.bin_asset_root_poses_w.tolist()],
                    dtype=torch.float32,
                )
                for s in snapshots
            ]
            self.pool_work_area = torch.stack(poses).to(self.device)
            self.pool_work_area[:, :, :3] -= source_origin[:, None, :]
            self.pool_table_center_e = stack([s.table_center_e for s in snapshots])
            self.pool_bin_center_e = stack([s.bin_center_e for s in snapshots])
            self.pool_bin_qr_e = stack([s.bin_qr_position_e for s in snapshots])

    def reset_from_snapshots(self, env_ids: torch.Tensor) -> None:
        """Write sampled snapshot states into ``env_ids`` and reset their transition state."""

        env = self.env
        ids = env_ids.to(self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        pick = torch.randint(0, self.pool_size, (ids.numel(),), device=self.device)
        origins = env.scene.env_origins[ids]

        root = self.pool_robot_root[pick].clone()
        root[:, :3] += origins
        self.robot.write_root_link_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(self.pool_joint_pos[pick], self.pool_joint_vel[pick], env_ids=ids)
        box_root = self.pool_box_root[pick].clone()
        box_root[:, :3] += origins
        self.box.write_root_link_state_to_sim(box_root, env_ids=ids)
        if self.pool_has_work_area:
            zero_velocity = torch.zeros(ids.numel(), 6, device=self.device)
            for asset_index, name in enumerate(WORK_AREA_ASSETS):
                pose = self.pool_work_area[pick, asset_index].clone()
                pose[:, :3] += origins
                asset = env.scene[name]
                asset.write_root_pose_to_sim(pose, env_ids=ids)
                asset.write_root_velocity_to_sim(zero_velocity, env_ids=ids)
            for attr, pool in (
                ("table_center_e", self.pool_table_center_e),
                ("bin_center_e", self.pool_bin_center_e),
                ("bin_qr_position_e", self.pool_bin_qr_e),
            ):
                if not hasattr(env, attr):
                    setattr(env, attr, torch.zeros(self.num_envs, 3, device=self.device))
                getattr(env, attr)[ids] = pool[pick]

        ensure_task_phase_buffers(env)
        env.task_phase[ids] = CARRY_PHASE
        self.snapshot_index[ids] = pick
        self._reset_state(ids, self.pool_anchor[pick], self.pool_box_rest_height[pick])

    def begin_live(self, box_rest_height: float) -> None:
        """Start a transition in the live chain from the WBC command that was just applied."""

        ids = torch.arange(self.num_envs, device=self.device)
        anchor = self.env.action_manager.action.detach().clone().to(self.device)
        self._reset_state(ids, anchor, torch.full((self.num_envs,), float(box_rest_height), device=self.device))

    def _reset_state(self, ids: torch.Tensor, anchor: torch.Tensor, box_rest_height: torch.Tensor) -> None:
        self.anchor[ids] = anchor
        self.alpha[ids] = 0.0
        self.previous_alpha[ids] = 0.0
        self.elapsed[ids] = 0
        self.gate_action[ids] = 0.0
        self.previous_gate_action[ids] = 0.0
        self.com_shift[ids] = 0.0
        self.previous_target[ids] = anchor * self.scale + self.offset[ids]
        self.target_delta[ids] = 0.0
        self.box_rest_height[ids] = box_rest_height
        self.staging_reached[ids] = False
        self.first_swing_done[ids] = False
        self._bc3_raw_stamp = -1

    def sync_anchor_actions(self, env_ids: torch.Tensor) -> None:
        """Make the WBC anchor the processed target and BC3's last-action input.

        ``ManagerBasedRLEnv._reset_idx`` zeroes the action buffers after the
        reset events have run, so this must be called once the reset returns.
        """

        ids = env_ids.to(self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        manager = self.env.action_manager
        actions = manager.action.clone()
        actions[ids] = self.anchor[ids]
        # process_action() also shifts prev_action for every env; keep the
        # untouched environments' history intact.
        previous = manager._prev_action.clone()
        manager.process_action(actions)
        previous[ids] = self.anchor[ids]
        manager._prev_action.copy_(previous)
        self._bc3_raw_stamp = -1

    # ------------------------------------------------------------------ BC3
    def navigation_velocity(self) -> torch.Tensor:
        """The same ramped, staging-latched command the live chain feeds BC3."""

        navigation = self.env.command_manager.get_term("base_velocity")
        self.staging_reached |= navigation.at_staging.to(self.device).bool()
        ramp = ((self.elapsed.float() + 1.0) / self.navigation_ramp_steps).clamp(max=1.0)
        velocity = navigation.command.to(self.device)[:, :3] * ramp[:, None]
        return torch.where(self.staging_reached[:, None], torch.zeros_like(velocity), velocity)

    def bc3_raw_action(self) -> torch.Tensor:
        """Frozen BC3 output in raw action coordinates, cached per control step."""

        stamp = int(self.env.common_step_counter)
        if stamp == self._bc3_raw_stamp:
            return self._bc3_raw
        manager = self.env.observation_manager
        student = torch.cat(
            (
                manager.compute_group("policy_history", update_history=False),
                manager.compute_group("policy_current", update_history=False),
            ),
            dim=-1,
        ).to(self.device)
        if student.shape != (self.num_envs, 517):
            raise RuntimeError(f"expected a ({self.num_envs}, 517) student observation, got {student.shape}")
        student[:, self._velocity_slice] = self.navigation_velocity()
        with torch.inference_mode():
            absolute = self.bc3.model(student)
        self._bc3_raw = ((absolute - self.offset) / self.scale).clone()
        self._bc3_raw_stamp = stamp
        return self._bc3_raw

    # ------------------------------------------------------------------ CoM
    def com_position_w(self) -> torch.Tensor:
        com = self.robot.data.body_com_pos_w
        return (self.body_masses[:, :, None] * com).sum(dim=1) / self.robot_mass[:, None]

    def com_offset_raw(self, com_shift_yaw: torch.Tensor) -> torch.Tensor:
        """Raw-action offset that shifts the CoM relative to the feet by ``com_shift_yaw``."""

        from isaaclab.utils.math import quat_apply, yaw_quat

        jacobians = self.robot.root_physx_view.get_jacobians()[..., :3, :]
        jacobians = jacobians[..., self._com_jacobian_columns].to(self.device)
        com_j = com_jacobian(jacobians, self.body_masses)
        feet_j = jacobians[:, self.foot_body_ids].mean(dim=1)
        relative_xy = (com_j - feet_j)[:, :2, :]
        shift_b = torch.cat((com_shift_yaw, torch.zeros_like(com_shift_yaw[:, :1])), dim=-1)
        shift_w = quat_apply(yaw_quat(self.robot.data.root_quat_w), shift_b)[:, :2]
        delta_q = com_shift_joint_offset(
            relative_xy, shift_w, max_shift_m=self.cfg.gate.max_com_shift_m, damping=self.cfg.com_damping
        )
        offset = torch.zeros(self.num_envs, 43, device=self.device)
        offset[:, self.com_action_ids] = delta_q / self.scale[self.com_action_ids]
        return offset

    # ------------------------------------------------------------------ step
    def compose(self, policy_action: torch.Tensor) -> torch.Tensor:
        """Turn the PPO action into the 43-D raw command for this control step."""

        action = policy_action.to(self.device).reshape(self.num_envs, -1)
        if action.shape[1] != self.cfg.action_dim:
            raise ValueError(f"expected {self.cfg.action_dim}-D transition actions, got {action.shape[1]}")
        action = action.clamp(-1.0, 1.0)
        self.previous_gate_action.copy_(self.gate_action)
        self.gate_action.zero_()
        self.gate_action[:, : action.shape[1]] = action

        bc3 = self.bc3_raw_action()
        self.previous_alpha.copy_(self.alpha)
        self.alpha = update_alpha(self.alpha, action[:, 0], self.elapsed, self.cfg.gate)
        com_offset = None
        if self.cfg.stage == 2:
            self.com_shift = action[:, 1:3] * self.cfg.gate.max_com_shift_m
            com_offset = self.com_offset_raw(self.com_shift)
        raw = compose_transition_action(
            self.anchor, bc3, self.alpha,
            locked_indices=self.locked, locked_reference=self.anchor, com_offset=com_offset,
        )
        target = raw * self.scale + self.offset
        self.target_delta = target - self.previous_target
        self.previous_target = target
        self.elapsed += 1
        return raw

    def provenance(self) -> dict[str, object]:
        return {
            "schema": "unitree_box_carry_transition_policy_v1",
            "gate_version": TRANSITION_GATE_VERSION,
            "stage": self.cfg.stage,
            "action_dim": self.cfg.action_dim,
            "gate": vars(self.cfg.gate),
            "com_damping": self.cfg.com_damping,
            "carry_checkpoint": str(self.carry_checkpoint),
            "snapshot_pool_size": self.pool_size,
        }


class TransitionPolicyWrapper(gym.Wrapper):
    """Map low-dimensional transition actions to the env's 43-D raw command."""

    def __init__(self, env: gym.Env, controller: CarryTransitionController) -> None:
        super().__init__(env)
        self.controller = controller

    def reset(self, **kwargs):
        observation, extras = self.env.reset(**kwargs)
        self.controller.sync_anchor_actions(torch.arange(self.controller.num_envs))
        return observation, extras

    def step(self, action: torch.Tensor):
        observation, reward, terminated, truncated, extras = self.env.step(self.controller.compose(action))
        self.controller.sync_anchor_actions((terminated | truncated).nonzero(as_tuple=False).flatten())
        return observation, reward, terminated, truncated, extras


class TransitionRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL view whose action dimension is the transition actor's, not the 43-D term's."""

    def __init__(self, env: gym.Env, clip_actions: float | None = None) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.num_actions = env.unwrapped.transition_controller.cfg.action_dim
        self._modify_action_space()


# Gate output bias that advances alpha at a quarter of the maximum rate: with
# the default 0.25 s minimum handoff this is a 1 s linear blend, i.e. the actor
# starts at (the linear version of) the fixed-schedule baseline.
INITIAL_GATE_BIAS = -0.5


def initialize_transition_actor(policy: torch.nn.Module, *, stage1_checkpoint: str | Path | None = None) -> None:
    """Start the gate at the 1 s baseline; stage 2 inherits a stage-1 model with a zero CoM head."""

    last = [m for m in policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        if stage1_checkpoint is not None:
            source = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
            target = policy.state_dict()
            for key, value in source.items():
                if key not in target:
                    continue
                if target[key].shape == value.shape:
                    target[key].copy_(value)
                elif target[key].ndim == value.ndim and target[key].shape[1:] == value.shape[1:]:
                    # Gate output rows / std entries; the CoM rows stay as set below.
                    target[key][: value.shape[0]].copy_(value)
            policy.load_state_dict(target)
            last.weight[1:].zero_()
            last.bias[1:].zero_()
            print(f"[INFO] Stage-2 transition actor initialized from {stage1_checkpoint}; CoM head zeroed.")
            return
        last.weight.zero_()
        last.bias.zero_()
        last.bias[0] = INITIAL_GATE_BIAS
    print("[INFO] Transition gate initialized at the 1 s linear-blend baseline.")


def load_transition_actor(checkpoint: str | Path, device: str | torch.device):
    """Rebuild the deterministic RSL-RL actor (Linear/ELU MLP) from a training checkpoint."""

    state = torch.load(Path(checkpoint).expanduser(), map_location=device, weights_only=False)
    weights = state["model_state_dict"]
    indices = sorted({int(key.split(".")[1]) for key in weights if key.startswith("actor.") and key.endswith(".weight")})
    layers: list[torch.nn.Module] = []
    for position, index in enumerate(indices):
        weight = weights[f"actor.{index}.weight"]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(weights[f"actor.{index}.bias"])
        layers.append(linear)
        if position < len(indices) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor
