"""Task-local safety hook for RSL-RL Random Network Distillation.

RSL-RL adds intrinsic reward after the environment returns its observation.
Isaac Lab returns a reset observation on a terminal transition, so an
environment-side reward term cannot reliably suppress curiosity for a fall.
This hook applies a dedicated post-step safety mask before intrinsic reward is
added to the PPO rollout, without modifying the installed ``rsl_rl`` package.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch


def _masked_intrinsic_reward(rnd: Any, obs: Any, safety_mask: torch.Tensor) -> torch.Tensor:
    """Compute the upstream RND reward, but normalize and return only safe rows."""
    # Keep the upstream step counter/weight schedule semantics unchanged.
    rnd.update_counter += 1
    rnd_state = rnd.get_rnd_state(obs)
    rnd_state = rnd.state_normalizer(rnd_state)
    target_embedding = rnd.target(rnd_state).detach()
    predictor_embedding = rnd.predictor(rnd_state).detach()
    intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)

    # Applying the mask before reward normalization also prevents unsafe rows
    # from inflating the running intrinsic-reward scale.  The state/predictor
    # networks may still see those rows, but they cannot create a policy
    # incentive because their final intrinsic reward is exactly zero.
    intrinsic_reward = intrinsic_reward * safety_mask.to(dtype=intrinsic_reward.dtype)
    intrinsic_reward = rnd.reward_normalizer(intrinsic_reward)
    if rnd.weight_scheduler is not None:
        rnd.weight = rnd.weight_scheduler(step=rnd.update_counter, **rnd.weight_scheduler_params)
    else:
        rnd.weight = rnd.initial_weight
    return intrinsic_reward * safety_mask.to(dtype=intrinsic_reward.dtype) * rnd.weight


def install_rnd_safety_mask(algorithm: Any, observation_key: str = "rnd_safety_mask") -> None:
    """Install a hard intrinsic-reward mask on one RSL-RL PPO algorithm.

    The observation must have shape ``(num_envs, 1)`` and contain ``1`` for a
    physically safe state.  Terminal rows are always masked separately: by the
    time PPO receives them, Isaac Lab has already produced the reset state.
    """
    if getattr(algorithm, "rnd", None) is None:
        return
    if getattr(algorithm, "_unitree_rnd_safety_mask_key", None) == observation_key:
        return

    def process_env_step_with_rnd_safety_mask(
        self: Any,
        obs: Any,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        if observation_key not in obs.keys():
            raise KeyError(
                f"RND safety mask observation '{observation_key}' is missing from the PPO observation dictionary."
            )

        # This mirrors PPO.process_env_step() from RSL-RL, replacing only its
        # unconditional RND addition below.
        self.policy.update_normalization(obs)
        self.rnd.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        safety_mask = obs[observation_key]
        if safety_mask.ndim != 2 or safety_mask.shape[-1] != 1:
            raise ValueError(
                f"RND safety mask '{observation_key}' must have shape (num_envs, 1); got {tuple(safety_mask.shape)}."
            )
        safety_mask = safety_mask.squeeze(-1).to(dtype=torch.bool)
        # A terminal result's observation is already the reset pose.  Without
        # this explicit condition, a fall would receive curiosity for that
        # apparently safe reset observation.
        safety_mask = safety_mask & ~dones.reshape(-1).to(dtype=torch.bool)
        self.intrinsic_rewards = _masked_intrinsic_reward(self.rnd, obs, safety_mask)
        self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    algorithm.process_env_step = MethodType(process_env_step_with_rnd_safety_mask, algorithm)
    algorithm._unitree_rnd_safety_mask_key = observation_key

