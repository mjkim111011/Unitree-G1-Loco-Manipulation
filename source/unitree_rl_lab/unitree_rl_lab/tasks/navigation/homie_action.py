"""High-level ``[vx, vy, yaw_rate]`` action executed by the frozen Homie V2 lower body (batched torch port).

Per control step (50 Hz) the term builds Homie's 86-D frame from the simulator state, appends it to the
6-frame history, runs the stand/walk network (|cmd| < 0.05 selects stand) and writes absolute joint
position targets for all 43 joints: Homie for legs + waist, the nominal upper body for the arms (all
zeros = the default posture; the Arena WBC's Pink output under the nominal wrist hold, verified to be
exactly 0) and open hands (Arena hand_state 0 = all zeros).  This reproduces the Arena WBC worker that
drives BC-1/BC-2 (closed-loop parity: scripts/approach/homie_parity_rollout.py).
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from unitree_rl_lab.utils.homie_batch import BODY_JOINTS, LOWER_BODY_JOINTS, HomieBatch

from . import mdp as nav_mdp

# Action-to-command profiles shared by the training env (HomieNavActionCfg) and the Arena-WBC evaluator.
#   scale / scale_negative: |command| per unit action for positive / negative actions, [vx, vy, wz]
#   ellipse: clip (vx, vy) to the ellipse of the per-sign axis scales (diagonal walking no faster than straight)
#   rate_limit: max change of [vx, vy, wz] per second (m/s^2, m/s^2, rad/s^2); None = none
#   heading_gate: (near_radius_m, on_deg, off_deg) planar-command gain from the goal bearing (heading_gate_gain); None = none
#   gate_release: stop-gate hysteresis, a latched stop is released only when the gate action < gate_release; None = none
NAV_ACTION_PROFILES = {
    "v5": {"scale": (0.6, 0.4, 0.8), "scale_negative": None, "ellipse": False, "rate_limit": None,
           "heading_gate": None, "gate_release": None},
    "v6": {"scale": (0.5, 0.25, 0.6), "scale_negative": (0.2, 0.25, 0.6), "ellipse": True, "rate_limit": (0.8, 0.8, 2.0),
           "heading_gate": None, "gate_release": None},
    # v7 = v6 + turn in place while the goal is far off the heading + stop-gate hysteresis (no stop/walk flicker)
    "v7": {"scale": (0.5, 0.25, 0.6), "scale_negative": (0.2, 0.25, 0.6), "ellipse": True, "rate_limit": (0.8, 0.8, 2.0),
           "heading_gate": (0.5, 45.0, 90.0), "gate_release": -0.5},
}


def heading_gate_gain(goal_pos_b: torch.Tensor, near_radius: float, on_deg: float, off_deg: float) -> torch.Tensor:
    """Planar-command gain (N,) from the goal point in the robot yaw frame (N, 2): 1 while the goal lies within
    ``on_deg`` of the heading or closer than ``near_radius`` (final alignment needs side / back steps), 0 beyond
    ``off_deg`` (turn in place first), linear in the cosine of the bearing in between."""
    distance = torch.linalg.vector_norm(goal_pos_b, dim=-1)
    cos_bearing = goal_pos_b[:, 0] / torch.clamp(distance, min=1.0e-6)
    cos_on, cos_off = math.cos(math.radians(on_deg)), math.cos(math.radians(off_deg))
    gain = torch.clamp((cos_bearing - cos_off) / (cos_on - cos_off), 0.0, 1.0)
    return torch.where(distance < near_radius, torch.ones_like(gain), gain)


def stop_gate_state(gate_action: torch.Tensor, previous_on: torch.Tensor, release: float) -> torch.Tensor:
    """Stop gate with hysteresis (N,): on when the gate action > 0; a stop that is on stays on until the gate action
    drops below ``release`` (< 0).  Without it a deterministic gate hovering at 0 toggles stop/walk every few steps."""
    return (gate_action > 0.0) | (previous_on & (gate_action >= release))


def process_nav_command(actions: torch.Tensor, previous: torch.Tensor, scale: torch.Tensor, scale_negative: torch.Tensor | None,
                        ellipse: bool, stop_gate: bool, rate_limit: torch.Tensor | None, dt: float,
                        gate_on: torch.Tensor | None = None, planar_gain: torch.Tensor | None = None) -> torch.Tensor:
    """Policy action (N, 3 or 4) -> Homie command [vx, vy, wz] (N, 3).

    With the v5 settings (no negative scale, no ellipse, no rate limit) this is exactly
    clip(a, -1, 1) * scale followed by the stop gate (a[3] > 0 -> [0, 0, 0]).
    ``gate_on`` (N,) replaces a[3] > 0 as the stop decision (v7 hysteresis, stop_gate_state); ``planar_gain`` (N,)
    scales [vx, vy] after the caps (v7 heading gate).  Both None = v5/v6 behaviour, unchanged.
    """
    clipped = torch.clamp(actions[:, :3], -1.0, 1.0)
    if scale_negative is None:
        command = clipped * scale
    else:
        command = torch.where(clipped >= 0.0, clipped * scale, clipped * scale_negative)
    if ellipse:
        sx = scale[0] if scale_negative is None else torch.where(command[:, 0] >= 0.0, scale[0], scale_negative[0])
        sy = scale[1] if scale_negative is None else torch.where(command[:, 1] >= 0.0, scale[1], scale_negative[1])
        radius = torch.sqrt((command[:, 0] / sx) ** 2 + (command[:, 1] / sy) ** 2)
        shrink = 1.0 / torch.clamp(radius, min=1.0)
        command = torch.cat((command[:, :2] * shrink.unsqueeze(-1), command[:, 2:3]), dim=-1)
    if planar_gain is not None:
        command = torch.cat((command[:, :2] * planar_gain.unsqueeze(-1), command[:, 2:3]), dim=-1)
    if stop_gate:
        # Positive gate = zero target command (Homie stand policy once |command| < 0.05).
        stop = actions[:, 3:4] > 0.0 if gate_on is None else gate_on.unsqueeze(-1)
        command = torch.where(stop, torch.zeros_like(command), command)
    if rate_limit is not None:
        step = rate_limit * dt
        command = previous + torch.clamp(command - previous, -step, step)
    return command


class HomieNavAction(ActionTerm):
    cfg: "HomieNavActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "HomieNavActionCfg", env) -> None:
        super().__init__(cfg, env)
        names = list(self._asset.joint_names)
        self._body_ids = torch.tensor([names.index(n) for n in BODY_JOINTS], device=self.device)
        self._lower_ids = torch.tensor([names.index(n) for n in LOWER_BODY_JOINTS], device=self.device)
        self._arm_ids = torch.tensor([names.index(n) for n in BODY_JOINTS[15:]], device=self.device)
        self._hand_ids = torch.tensor([i for i, n in enumerate(names) if "_hand_" in n], device=self.device)
        if len(self._hand_ids) != 14:
            raise ValueError(f"expected 14 Dex3 hand joints, found {len(self._hand_ids)}")
        self._homie = HomieBatch(self.num_envs, self.device)
        self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._scale = torch.tensor(cfg.command_scale, device=self.device)
        self._scale_negative = None if cfg.command_scale_negative is None else torch.tensor(cfg.command_scale_negative, device=self.device)
        self._rate_limit = None if cfg.command_rate_limit is None else torch.tensor(cfg.command_rate_limit, device=self.device)
        self._gate_on = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # v7 stop-gate latch
        self._height = torch.full((self.num_envs,), cfg.base_height, device=self.device)
        self._rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self._targets = self._asset.data.default_joint_pos.clone()
        self._targets[:, self._arm_ids] = cfg.arm_target
        self._targets[:, self._hand_ids] = cfg.hand_target

    @property
    def action_dim(self) -> int:
        return 4 if self.cfg.stop_gate else 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        """The velocity command actually sent to Homie ``[vx, vy, wz]`` (m/s, rad/s)."""
        return self._command

    @property
    def homie(self) -> HomieBatch:
        return self._homie

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        planar_gain = None if self.cfg.heading_gate is None else heading_gate_gain(nav_mdp.gt_goal_pos_b(self._env), *self.cfg.heading_gate)
        gate_on = None
        if self.cfg.stop_gate and self.cfg.stop_gate_release is not None:
            self._gate_on[:] = stop_gate_state(actions[:, 3], self._gate_on, self.cfg.stop_gate_release)
            gate_on = self._gate_on
        # Positive gate = zero command (Homie stand policy): lets a stochastic policy stand still.
        command = process_nav_command(actions, self._command, self._scale, self._scale_negative, self.cfg.planar_ellipse_cap,
                                      self.cfg.stop_gate, self._rate_limit, self._env.step_dt, gate_on=gate_on, planar_gain=planar_gain)
        self._command[:] = command
        data = self._asset.data
        lower = self._homie.step(
            self._command,
            self._height,
            self._rpy,
            data.root_link_ang_vel_b,
            data.projected_gravity_b,
            data.joint_pos[:, self._body_ids],
            data.joint_vel[:, self._body_ids],
        )
        self._targets[:, self._lower_ids] = lower

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._targets)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._command[ids] = 0.0
        self._gate_on[ids] = False
        self._homie.reset(env_ids)


@configclass
class HomieNavActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HomieNavAction
    asset_name: str = MISSING
    command_scale: tuple[float, float, float] = (0.6, 0.4, 0.8)
    """Action[:3] in [-1, 1]^3 is clipped and multiplied by this scale to give [vx, vy, wz]."""
    command_scale_negative: tuple[float, float, float] | None = None
    """Scale for negative actions (None = command_scale; v6: slow backward walking)."""
    planar_ellipse_cap: bool = False
    """Clip (vx, vy) to the ellipse of the axis scales so that diagonal walking is not faster than straight (v6)."""
    command_rate_limit: tuple[float, float, float] | None = None
    """Max change of [vx, vy, wz] per second (m/s^2, m/s^2, rad/s^2); None = immediate (v5)."""
    heading_gate: tuple[float, float, float] | None = None
    """(near_radius_m, on_deg, off_deg): scale [vx, vy] by heading_gate_gain of the goal bearing (v7); None = none."""
    stop_gate_release: float | None = None
    """Stop-gate hysteresis: a stop stays on until action[3] < this value (v7, e.g. -0.5); None = on iff action[3] > 0."""
    stop_gate: bool = True
    """Action[3] > 0 sends exactly [0, 0, 0] (stand), independent of the exploration noise on action[:3]."""
    base_height: float = 0.75
    """Homie base-height command; BC-1/BC-2 and the evaluator use 0.75."""
    arm_target: float = 0.0
    hand_target: float = 0.0
