"""Grip-aware residual PPO over the direct BC2 -> BC3 handoff (Isaac runtime).

Training episodes start from physical ``BILATERAL_HOLD`` snapshots (BC2's
grasp-success state), i.e. exactly where the direct-handoff evaluation
(``evaluate_bc2_rollout.py --grasp_snapshot_input ... --wbc_bridge_seconds 0``)
starts.  Per episode, matching that evaluation step by step:

1. reset: the Box mass is set to the snapshot's mass and the robot and Box are
   parked apart in free air for one *contact-clearing* step.  Without it, a
   reset that teleports the Box back into hands that were still holding it
   (e.g. after a timeout) keeps the PhysX contact pairs - and their stale
   friction anchors - alive, and the next episode slips far more often than a
   fresh evaluator process (feet/ground likewise);
2. the snapshot is then written exactly, the anchor is its ``last_action``
   (BC2's final command), and one warm-up step applies the anchor unchanged
   (the evaluator's controlled post-restore step);
3. carry steps: frozen BC3 is queried with the same ramped navigation command,
   blended into the anchor with the fixed 1 s quintic schedule, the 28 arm/hand
   joints stay locked to the anchor, and the residual adds
   ``beta(k) * r_k`` on the 15 leg/waist joints only.

BC2 is represented only by its recorded final command; BC3 is frozen.  With a
zero residual the command is identical to the baseline.  See
:mod:`unitree_rl_lab.utils.teacher.grip_residual` for the shared primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from unitree_rl_lab.utils.teacher.grip_residual import (
    GRIP_RESIDUAL_VERSION,
    GripResidualCfg,
    blend_weight,
    compose_residual_action,
    residual_envelope,
    residual_joint_indices,
    residual_raw_offset,
    rate_limited_residual,
    tilt_from_projected_gravity,
)
from unitree_rl_lab.utils.teacher.phase_bc import file_sha256
from unitree_rl_lab.utils.teacher.physical_snapshots import SnapshotKind
from unitree_rl_lab.utils.transition_policy import CarryTransitionController


WRIST_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# Contact-clearing park offsets (m): robot and Box apart, both clear of the ground and table.
PARK_ROBOT_RISE_M = 1.5
PARK_BOX_RISE_M = 3.0
# Full-hand Box force sensors added by the residual env: one per hand link, named
# "<side>_<link>_box_load" (not "*_box_contact", so the fingertip contact counts
# BC3 and the gates use are unchanged).
HAND_LOAD_SUFFIX = "_box_load"


class GripResidualController(CarryTransitionController):
    """Per-environment residual state shared by the wrapper, the MDP terms and the live chain."""

    def __init__(
        self,
        env,
        *,
        carry_checkpoint: str | Path,
        snapshot_paths: Sequence[str | Path] = (),
        residual_cfg: GripResidualCfg | None = None,
        balance_snapshot_sources: bool = True,
        seed: int | None = None,
    ) -> None:
        self.residual_cfg = (residual_cfg or GripResidualCfg()).validate()
        self.balance_snapshot_sources = balance_snapshot_sources
        self.snapshot_paths = [str(Path(p).expanduser().resolve()) for p in snapshot_paths]
        super().__init__(
            env, carry_checkpoint=carry_checkpoint, snapshot_paths=snapshot_paths,
            snapshot_kind=SnapshotKind.BILATERAL_HOLD,
        )
        if abs(env.step_dt - self.residual_cfg.schedule.step_dt) > 1.0e-9:
            raise ValueError("residual schedule step_dt must match the env step_dt")
        self.residual_ids = residual_joint_indices(self.joint_names)
        if self.residual_ids != self.com_action_ids or set(self.residual_ids) & set(self.locked):
            raise RuntimeError("residual joints must be the unlocked leg/waist joints")
        robot = self.robot
        self.wrist_body_ids = robot.find_bodies(list(WRIST_BODY_NAMES), preserve_order=True)[0]
        self.residual_joint_ids = list(self.com_joint_ids)
        self.default_joint_pos = robot.data.default_joint_pos[:, self.residual_joint_ids].clone()
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        n, cfg = self.num_envs, self.residual_cfg
        self.warmup_pending = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.clearing_pending = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.command = torch.zeros(n, 15, device=self.device)
        self.previous_command = torch.zeros(n, 15, device=self.device)
        self.residual = torch.zeros(n, 15, device=self.device)
        self.applied = torch.zeros(n, 15, device=self.device)
        self.previous_applied = torch.zeros(n, 15, device=self.device)
        self.handoff_rel_pos = torch.zeros(n, 3, device=self.device)
        self.handoff_box_z = torch.zeros(n, device=self.device)
        self.handoff_captured = torch.zeros(n, dtype=torch.bool, device=self.device)
        history = max(cfg.box_obs_delay_steps, 0) + cfg.box_velocity_window_steps + 1
        self.box_history = torch.zeros(history, n, 4, device=self.device)  # rel_pos (3) + box z
        self.box_estimate_history = torch.zeros(cfg.box_velocity_window_steps + 1, n, 4, device=self.device)
        self.pool_box_mass = None
        if self.pool_size:
            self.pool_box_mass = torch.as_tensor(
                [float(s.box_mass_kg) for s in self.pool_snapshots], device=self.device
            )
            box_view = self.box.root_physx_view
            self._spawn_box_mass = box_view.get_masses().clone()
            self._spawn_box_inertia = box_view.get_inertias().clone()
            self.current_box_mass = self._spawn_box_mass.sum(dim=-1).to(self.device)
        self._measure_cache = None
        self._prepared_stamp = None
        env.grip_residual_controller = self

    # ------------------------------------------------------------------ resets
    def sample_snapshots(self, count: int) -> torch.Tensor:
        """Uniform over snapshot stores (e.g. 0.2 / 0.3 kg), then uniform within a store."""

        if not self.balance_snapshot_sources:
            return super().sample_snapshots(count)
        sources = int(self.pool_source.max()) + 1
        source = torch.randint(0, sources, (count,), device=self.device)
        pick = torch.empty(count, dtype=torch.long, device=self.device)
        for s in range(sources):
            members = (self.pool_source == s).nonzero(as_tuple=False).flatten()
            mask = source == s
            if mask.any():
                pick[mask] = members[torch.randint(0, members.numel(), (int(mask.sum()),), device=self.device)]
        return pick

    def reset_from_snapshots(self, env_ids: torch.Tensor) -> None:
        super().reset_from_snapshots(env_ids)
        ids = env_ids.to(self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self._apply_box_mass(ids, self.pool_box_mass[self.snapshot_index[ids]])
        self._write_snapshot_state(ids, park=True)
        self.clearing_pending[ids] = True

    def _write_snapshot_state(self, ids: torch.Tensor, *, park: bool) -> None:
        """Write the sampled snapshot's robot and Box state (optionally parked apart, at rest)."""

        pick = self.snapshot_index[ids]
        origins = self.env.scene.env_origins[ids]
        root = self.pool_robot_root[pick].clone()
        root[:, :3] += origins
        box_root = self.pool_box_root[pick].clone()
        box_root[:, :3] += origins
        joint_vel = self.pool_joint_vel[pick]
        if park:
            root[:, 2] += PARK_ROBOT_RISE_M
            box_root[:, 2] += PARK_BOX_RISE_M
            root[:, 7:] = 0.0
            box_root[:, 7:] = 0.0
            joint_vel = torch.zeros_like(joint_vel)
        self.robot.write_root_link_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(self.pool_joint_pos[pick], joint_vel, env_ids=ids)
        self.box.write_root_link_state_to_sim(box_root, env_ids=ids)

    def finish_clearing(self, env_ids: torch.Tensor) -> None:
        """After the contact-clearing step: write the exact snapshot; the warm-up step follows."""

        ids = env_ids.to(self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self._write_snapshot_state(ids, park=False)
        self.clearing_pending[ids] = False
        self.sync_anchor_actions(ids)
        # The Box estimate of these envs starts at their handoff (after the warm-up), so the
        # once-per-step observation stamp is kept: other envs must not be pushed twice.
        self._measure_cache = None

    def _apply_box_mass(self, ids: torch.Tensor, mass: torch.Tensor) -> None:
        """Per-env Box mass (inertia scaled with it, as PhysX derives it from the spawn mass)."""

        changed = (self.current_box_mass[ids] - mass).abs() > 1.0e-9
        if not bool(changed.any()):
            return
        ids, mass = ids[changed], mass[changed]
        view = self.box.root_physx_view
        cpu_ids = ids.cpu()
        ratio = (mass / self._spawn_box_mass[cpu_ids].sum(dim=-1).to(self.device)).cpu()
        masses = view.get_masses().clone()
        inertias = view.get_inertias().clone()
        masses[cpu_ids] = self._spawn_box_mass[cpu_ids] * ratio[:, None]
        inertias[cpu_ids] = self._spawn_box_inertia[cpu_ids] * ratio[:, None]
        view.set_masses(masses, cpu_ids)
        view.set_inertias(inertias, cpu_ids)
        self.current_box_mass[ids] = mass

    def box_mass(self) -> torch.Tensor:
        return self.box.root_physx_view.get_masses().sum(dim=-1).to(self.device)

    def _reset_state(self, ids: torch.Tensor, anchor: torch.Tensor, box_rest_height: torch.Tensor) -> None:
        super()._reset_state(ids, anchor, box_rest_height)
        # elapsed counts CARRY steps; the first step after a reset is the warm-up.
        self.warmup_pending[ids] = True
        for buffer in (self.command, self.previous_command, self.residual, self.applied, self.previous_applied):
            buffer[ids] = 0.0
        self.handoff_captured[ids] = False
        self._measure_cache = None

    def begin_live_residual(self, anchor: torch.Tensor, box_rest_height: float) -> None:
        """Live chain: start at CARRY step 0 (the evaluator already ran its warm-up step)."""

        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_state(ids, anchor.to(self.device), torch.full((self.num_envs,), float(box_rest_height),
                                                                   device=self.device))
        self.warmup_pending[ids] = False
        self._capture_handoff(ids)
        self._prepared_stamp = int(self.env.common_step_counter)

    def live_step(self, carry_step: int, bc3_raw: torch.Tensor) -> None:
        """Live chain, before the observation: the CARRY step and the BC3 output it already computed."""

        self.elapsed[:] = int(carry_step)
        self._bc3_raw = bc3_raw.detach().to(self.device).reshape(self.num_envs, -1).clone()
        self._bc3_raw_stamp = int(self.env.common_step_counter)

    # ------------------------------------------------------------------ Box state
    def box_wrist_state_true(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Box position relative to the wrist midpoint (pelvis yaw frame), its velocity, and Box height."""

        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        robot, box = self.robot.data, self.box.data
        wrists = robot.body_link_pos_w[:, self.wrist_body_ids]
        wrist_vel = robot.body_link_lin_vel_w[:, self.wrist_body_ids]
        quat = yaw_quat(robot.root_quat_w)
        rel_pos = quat_apply_inverse(quat, box.root_link_pos_w - wrists.mean(dim=1))
        rel_vel = quat_apply_inverse(quat, box.root_link_lin_vel_w - wrist_vel.mean(dim=1))
        box_z = box.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        return rel_pos, rel_vel, box_z

    def _capture_handoff(self, ids: torch.Tensor) -> None:
        rel_pos, _, box_z = self.box_wrist_state_true()
        self.handoff_rel_pos[ids] = rel_pos[ids]
        self.handoff_box_z[ids] = box_z[ids]
        self.handoff_captured[ids] = True
        sample = torch.cat((rel_pos, box_z[:, None]), dim=-1)
        self.box_history[:, ids] = sample[ids]
        self.box_estimate_history[:, ids] = sample[ids]

    def prepare_observation(self) -> None:
        """Once per physics state, before any residual observation: capture handoff references for
        environments whose warm-up just ended and push the Box estimate."""

        stamp = int(self.env.common_step_counter)
        if self._prepared_stamp == stamp:
            return
        self._prepared_stamp = stamp
        fresh = (~self.handoff_captured) & (~self.warmup_pending)
        if bool(fresh.any()):
            self._capture_handoff(fresh.nonzero(as_tuple=False).flatten())
        self.update_box_estimate()

    def bc3_target_minus_joint_pos(self) -> torch.Tensor:
        """Where frozen BC3 drives the leg/waist joints at this CARRY step, relative to the measured joints."""

        ids = self.residual_ids
        target = self.bc3_raw_action()[:, ids] * self.scale[ids] + self.offset[:, ids]
        return target - self.robot.data.joint_pos[:, self.residual_joint_ids]

    def update_box_estimate(self) -> None:
        """Push the current truth; the estimate is the delayed truth plus white noise."""

        rel_pos, _, box_z = self.box_wrist_state_true()
        sample = torch.cat((rel_pos, box_z[:, None]), dim=-1)
        self.box_history = torch.roll(self.box_history, 1, dims=0)
        self.box_history[0] = sample
        cfg = self.residual_cfg
        estimate = self.box_history[cfg.box_obs_delay_steps].clone()
        if cfg.box_obs_noise_m > 0:
            estimate += cfg.box_obs_noise_m * torch.randn(
                estimate.shape, device=self.device, generator=self.generator
            )
        self.box_estimate_history = torch.roll(self.box_estimate_history, 1, dims=0)
        self.box_estimate_history[0] = estimate
        idle = ~self.handoff_captured
        if bool(idle.any()):
            self.box_history[:, idle] = sample[idle]
            self.box_estimate_history[:, idle] = sample[idle]

    def box_estimate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(relative position change since handoff, relative velocity, lift change) from the estimate."""

        cfg = self.residual_cfg
        now, then = self.box_estimate_history[0], self.box_estimate_history[cfg.box_velocity_window_steps]
        rel_change = now[:, :3] - self.handoff_rel_pos
        rel_vel = (now[:, :3] - then[:, :3]) / (cfg.box_velocity_window_steps * self.env.step_dt)
        lift_change = (now[:, 3] - self.handoff_box_z)[:, None]
        gate = self.handoff_captured[:, None].float()
        return rel_change * gate, rel_vel * gate, lift_change * gate

    # ------------------------------------------------------------------ measurements (rewards)
    def measure(self) -> dict[str, torch.Tensor]:
        """Simulator-truth quantities for rewards/critic, cached per physics state."""

        stamp = int(self.env.common_step_counter)
        if self._measure_cache is not None and self._measure_cache[0] == stamp:
            return self._measure_cache[1]
        robot = self.robot.data
        rel_pos, rel_vel, box_z = self.box_wrist_state_true()
        gate = self.handoff_captured.float()
        forces = self.hand_box_forces_w()
        horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
        ratio = forces[..., 2].abs() / horizontal.clamp_min(1.0)
        values = {
            "tilt": tilt_from_projected_gravity(robot.projected_gravity_b),
            "ang_vel_xy": torch.linalg.vector_norm(robot.root_ang_vel_b[:, :2], dim=-1),
            "rel_pos_change": (rel_pos - self.handoff_rel_pos) * gate[:, None],
            "rel_vel": rel_vel * gate[:, None],
            "lift_change": (box_z - self.handoff_box_z) * gate,
            "hand_force_w": forces,
            "friction_ratio": ratio * (horizontal > 1.0).float(),
        }
        self._measure_cache = (stamp, values)
        return values

    def hand_box_forces_w(self) -> torch.Tensor:
        """(N, 2, 3) Box-filtered contact force on each hand (sum over all hand + wrist links).

        The sensors report the force *on the hand*; the force on the Box is its negative.
        Zero when the env has no ``*_box_load`` sensors (e.g. the live evaluation scene).
        """

        total = torch.zeros(self.num_envs, 2, 3, device=self.device)
        for name, sensor in self.env.scene.sensors.items():
            if not name.endswith(HAND_LOAD_SUFFIX) or sensor.data.force_matrix_w is None:
                continue
            side = 0 if name.startswith("left_") else 1 if name.startswith("right_") else None
            if side is not None:
                total[:, side] += sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(dim=1)
        return total

    # ------------------------------------------------------------------ step
    def step_residual(self, command: torch.Tensor) -> torch.Tensor:
        """Advance the rate-limited residual; returns the applied (N, 15) correction in rad."""

        cfg = self.residual_cfg
        command = command.to(self.device).reshape(self.num_envs, -1)
        if command.shape[1] != cfg.action_dim:
            raise ValueError(f"expected {cfg.action_dim}-D residual actions, got {command.shape[1]}")
        command = command.clamp(-1.0, 1.0)
        active = ~self.warmup_pending
        self.previous_command.copy_(self.command)
        self.command = torch.where(active[:, None], command, torch.zeros_like(command))
        self.previous_applied.copy_(self.applied)
        residual = rate_limited_residual(self.residual, self.command, cfg)
        self.residual = torch.where(active[:, None], residual, torch.zeros_like(residual))
        beta = residual_envelope(self.elapsed, cfg.schedule) * active.float()
        self.applied = beta[:, None] * self.residual
        return self.applied

    def residual_offset(self, command: torch.Tensor) -> torch.Tensor:
        """Live chain: the (N, 43) raw offset for this CARRY step (added after blend and lock)."""

        return residual_raw_offset(self.step_residual(command), self.residual_ids, self.scale)

    def compose(self, policy_action: torch.Tensor) -> torch.Tensor:
        """Training: warm-up anchor, or fixed-blend baseline + residual, as a 43-D raw command."""

        bc3 = self.bc3_raw_action()
        weight = blend_weight(self.elapsed, self.residual_cfg.schedule)
        applied = self.step_residual(policy_action)
        raw = compose_residual_action(
            self.anchor, bc3, weight, locked_indices=self.locked,
            raw_offset=residual_raw_offset(applied, self.residual_ids, self.scale),
        )
        warm = self.warmup_pending | self.clearing_pending
        raw = torch.where(warm[:, None], self.anchor, raw)
        target = raw * self.scale + self.offset
        self.target_delta = target - self.previous_target
        self.previous_target = target
        self.elapsed = torch.where(warm, self.elapsed, self.elapsed + 1)
        # The warm-up ends with this step unless the env is still in its clearing step.
        self.warmup_pending &= self.clearing_pending
        return raw

    def provenance(self) -> dict[str, object]:
        return {
            "schema": "unitree_grip_residual_policy_v1",
            "version": GRIP_RESIDUAL_VERSION,
            "residual": self.residual_cfg.to_dict(),
            "carry_checkpoint": str(self.carry_checkpoint),
            "carry_checkpoint_sha256": file_sha256(self.carry_checkpoint),
            "snapshot_paths": self.snapshot_paths,
            "snapshot_sha256": {p: file_sha256(p) for p in self.snapshot_paths},
            "snapshot_pool_size": self.pool_size,
            "balance_snapshot_sources": self.balance_snapshot_sources,
            "residual_joint_names": [self.joint_names[i] for i in self.residual_ids],
            "locked_joint_count": len(self.locked),
        }


class GripResidualWrapper(gym.Wrapper):
    """15-D residual -> 43-D raw command; runs the contact-clearing step of every reset."""

    def __init__(self, env: gym.Env, controller: GripResidualController) -> None:
        super().__init__(env)
        self.controller = controller

    def reset(self, **kwargs):
        observation, extras = self.env.reset(**kwargs)
        self.controller.sync_anchor_actions(torch.arange(self.controller.num_envs))
        return observation, extras

    def step(self, action: torch.Tensor):
        controller = self.controller
        clearing = controller.clearing_pending.clone()
        observation, reward, terminated, truncated, extras = self.env.step(controller.compose(action))
        done = terminated | truncated
        controller.sync_anchor_actions(done.nonzero(as_tuple=False).flatten())
        # Envs whose clearing step just ran (and were not reset again) get their exact snapshot.
        cleared = (clearing & ~done).nonzero(as_tuple=False).flatten()
        if cleared.numel():
            controller.finish_clearing(cleared)
            observation = self.env.unwrapped.observation_manager.compute(update_history=False)
        return observation, reward, terminated, truncated, extras


class GripResidualRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL view whose action dimension is the residual actor's (15), not the 43-D term's."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.num_actions = env.unwrapped.grip_residual_controller.residual_cfg.action_dim
        self._modify_action_space()


def zero_residual_actor_output(policy: torch.nn.Module) -> None:
    """Zero the actor's last layer: the mean residual starts at exactly zero (the baseline)."""

    last = [m for m in policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    if last.out_features != 15:
        raise RuntimeError(f"expected a 15-D residual output layer, got {last.out_features}")
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    print("[INFO] Residual actor output zero-initialized at the fixed-blend baseline.")


def load_residual_config(checkpoint: str | Path) -> dict:
    """``params/grip_residual.json`` next to an RSL-RL run's ``model_*.pt``."""

    import json

    path = Path(checkpoint).expanduser().resolve().parent / "params" / "grip_residual.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing residual provenance {path}")
    return json.loads(path.read_text())
