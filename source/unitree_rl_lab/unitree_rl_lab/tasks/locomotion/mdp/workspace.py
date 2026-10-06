"""Helpers for per-environment randomized pick-and-place targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def environment_point_e(
    env: ManagerBasedRLEnv,
    fallback_point_e: tuple[float, float, float],
    attribute_name: str | None = None,
) -> torch.Tensor:
    """Return an ``(num_envs, 3)`` environment-relative point tensor.

    Randomized work-area events publish points directly on the environment.
    The tuple fallback preserves compatibility with fixed-scene tools and old
    checkpoints that do not enable the work-area event.
    """
    value = getattr(env, attribute_name, None) if attribute_name else None
    if value is None:
        return torch.tensor(
            fallback_point_e, device=env.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)
    result = torch.as_tensor(value, device=env.device, dtype=torch.float32)
    if result.ndim == 1:
        result = result.repeat(env.num_envs, 1)
    if result.shape != (env.num_envs, 3):
        raise ValueError(
            f"{attribute_name} must have shape ({env.num_envs}, 3), got {tuple(result.shape)}"
        )
    return result


def environment_point_w(
    env: ManagerBasedRLEnv,
    fallback_point_e: tuple[float, float, float],
    attribute_name: str | None = None,
) -> torch.Tensor:
    """Resolve a per-environment point and translate it to world coordinates."""
    return env.scene.env_origins + environment_point_e(
        env, fallback_point_e, attribute_name
    )
