"""Grip-readiness re-grip between BC2 (grasp-lift) and BC3 (carry).

With the WBC bridge removed, BC3 inherits BC2's final grasp as its arm/hand
lock reference.  In the direct-handoff evaluation the only early failure began
with a single Box-filtered fingertip contact per hand and slipped out once BC3
walked; the 7 s WBC stand had re-seated the same grasp (more contacts) and
succeeded.

Before the handoff this layer holds BC2's last target and, for every hand with
fewer than ``min_contacts_per_hand`` contacts, closes that hand's finger
targets further (in the direction they already moved from the open default)
at a bounded rate.  It stops once both hands keep the contact condition for
``hold_steps`` consecutive steps, or at ``max_steps`` (logged as a timeout;
the handoff then proceeds).  The resulting targets become BC3's arm/hand lock
reference, so the tighter grip persists through Carry.  Only finger joints
move; arms, legs and waist keep BC2's final targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


HAND_TOKEN = "_hand_"


@dataclass(frozen=True)
class GripReadinessCfg:
    min_contacts_per_hand: int = 2
    hold_steps: int = 5
    max_steps: int = 50
    finger_close_rate_rad: float = 0.01
    finger_close_max_rad: float = 0.25
    # Joints whose target is within this of the open default have no closing direction.
    min_closed_offset_rad: float = 1.0e-3

    def validate(self) -> "GripReadinessCfg":
        if self.min_contacts_per_hand < 1 or self.hold_steps < 1 or self.max_steps < self.hold_steps:
            raise ValueError("contacts and hold must be >= 1 and max_steps >= hold_steps")
        positive = (self.finger_close_rate_rad, self.finger_close_max_rad, self.min_closed_offset_rad)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("finger close rate/limit and the direction threshold must be positive")
        return self


def closing_directions(targets: torch.Tensor, open_positions: torch.Tensor, cfg: GripReadinessCfg) -> torch.Tensor:
    """Per-joint sign of further closing: the direction the grasp already moved from open (0 if none)."""

    delta = targets - open_positions
    return torch.where(delta.abs() > cfg.min_closed_offset_rad, torch.sign(delta), torch.zeros_like(delta))


def hand_ready(left_contacts: int, right_contacts: int, cfg: GripReadinessCfg) -> tuple[bool, bool]:
    return left_contacts >= cfg.min_contacts_per_hand, right_contacts >= cfg.min_contacts_per_hand


def advance_offsets(offsets: torch.Tensor, directions: torch.Tensor, close_mask: torch.Tensor,
                    cfg: GripReadinessCfg) -> torch.Tensor:
    """Close the masked finger joints by one rate step, capped at ``finger_close_max_rad``."""

    step = cfg.finger_close_rate_rad * directions * close_mask.to(offsets.dtype)
    proposed = offsets + step
    return torch.sign(proposed) * proposed.abs().clamp(max=cfg.finger_close_max_rad)


class GripReadiness:
    """Single-env runtime.  ``run`` steps the simulation and returns the new raw reference."""

    def __init__(self, env, cfg: GripReadinessCfg | None = None):
        if env.num_envs != 1:
            raise ValueError("grip readiness currently supports one environment")
        self.cfg = (cfg or GripReadinessCfg()).validate()
        self.env = env
        robot = env.scene["robot"]
        term = env.action_manager.get_term("JointPositionAction")
        names = list(term._joint_names)
        joint_ids = term._joint_ids
        joint_ids = list(range(robot.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        self.scale = torch.as_tensor(term._scale, device=env.device, dtype=torch.float32).expand(43)
        self.offset = torch.as_tensor(term._offset, device=env.device, dtype=torch.float32).expand(1, 43)[0]
        self.sides = {}
        for side in ("left", "right"):
            ids = [i for i, n in enumerate(names) if n.startswith(side) and HAND_TOKEN in n]
            if not ids:
                raise ValueError(f"no {side} hand joints in the action term")
            self.sides[side] = ids
        self.open_positions = robot.data.default_joint_pos[0, joint_ids].to(torch.float32)

    def run(self, reference_raw: torch.Tensor, step_fn, contact_fn, lift_fn=None) -> tuple[torch.Tensor, dict]:
        """Re-grip from ``reference_raw`` (1, 43).

        ``step_fn(raw_action)`` advances the controlled simulation by one step
        (the caller keeps its task phase); ``contact_fn()`` returns the
        (left, right) Box-filtered fingertip contact counts; ``lift_fn()``
        optionally returns the current lift height for the log.
        """

        cfg = self.cfg
        targets = reference_raw[0] * self.scale + self.offset
        directions = closing_directions(targets, self.open_positions, cfg)
        offsets = torch.zeros_like(targets)
        consecutive = 0
        left, right = contact_fn()
        history = []
        step = 0
        for step in range(cfg.max_steps):
            left_ok, right_ok = hand_ready(left, right, cfg)
            mask = torch.zeros_like(targets)
            if not left_ok:
                mask[self.sides["left"]] = 1.0
            if not right_ok:
                mask[self.sides["right"]] = 1.0
            offsets = advance_offsets(offsets, directions, mask, cfg)
            raw = reference_raw.clone()
            raw[0] = (targets + offsets - self.offset) / self.scale
            step_fn(raw)
            left, right = contact_fn()
            history.append({"step": step, "contacts": [int(left), int(right)],
                            "lift_height_m": None if lift_fn is None else float(lift_fn())})
            consecutive = consecutive + 1 if all(hand_ready(left, right, cfg)) else 0
            if consecutive >= cfg.hold_steps:
                break
        ready = consecutive >= cfg.hold_steps
        final = reference_raw.clone()
        final[0] = (targets + offsets - self.offset) / self.scale
        log = {
            "ready": ready, "timed_out": not ready, "steps": step + 1,
            "final_contacts": [int(left), int(right)],
            "max_finger_offset_rad": float(offsets.abs().max()),
            "finger_offsets_rad": {side: offsets[ids].tolist() for side, ids in self.sides.items()},
            "history": history,
        }
        return final, log
