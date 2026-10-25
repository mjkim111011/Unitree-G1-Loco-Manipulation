#!/usr/bin/env python3
"""Batched evaluation of navigation controllers in the Homie navigation environment.

  --controller ppo       trained RSL-RL checkpoint (deterministic actor mean); also exports TorchScript
  --controller scripted  ground-truth scripted go-to-pose navigator (scripts/approach/scripted_nav logic)

Every env runs exactly one episode from the chosen start distribution.  With the same --seed both
controllers see identical start states (the start sampling is the only random draw; neither controller
uses the RNG); the per-episode start states are saved so this can be checked.  Metrics: handoff success
(BC-1 v2 window held 25 steps), fall, collision, time-out, final handoff errors and steps to success.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Unitree-G1-43dof-BoxNav-Homie")
parser.add_argument("--controller", choices=("ppo", "scripted", "scripted_staging"), required=True)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--stage", type=int, default=2, help="start distribution stage (0..3) when no override")
parser.add_argument("--override", default=None,
                    help='JSON start range, e.g. {"r": [5.0, 6.0], "bearing_deg": 120, "yaw": 3.1416, "box_x": 0.0, "box_y": 0.12}')
parser.add_argument("--max_forward", type=float, default=0.3)
parser.add_argument("--episode_seconds", type=float, default=40.0)
parser.add_argument("--export_jit", type=Path, default=None)
parser.add_argument("--nav_target_offset_x", type=float, default=None, help="navigation target point offset from the canonical start (m); default -0.03")
parser.add_argument("--nav_handoff_x_range", type=float, nargs=2, default=None, help="handoff success window in x (m); default -0.08 0.015")
parser.add_argument("--post_hold_steps", type=int, default=0,
                    help="stop-stability test: after the 25-step handoff hold keep the same controller running this many steps "
                         "and measure base motion, drift, tilt, foot slip and window occupancy (default 0 = original evaluation)")
parser.add_argument("--handoff_criterion", choices=("env", "stable50"), default="env",
                    help="env = the task's own success term; stable50 = replace it by the v6 StableHandoffSuccess (window + speed <= 0.05, "
                         "yaw rate <= 0.1, leg joint speed <= 0.3, zero command, 50 steps) to compare policies under one handoff condition")
parser.add_argument("--table_contact_sensors", action="store_true",
                    help="add table-filtered contact sensors on the arm/hand links (table_contact.py) and record, for every contact "
                         "termination, whether the table was touched, by which link, and whether it happened after the first stop")
parser.add_argument("--box_offset", type=float, nargs=2, default=None, metavar=("DX", "DY"),
                    help="place the Box at exactly this offset (m) from the table centre for every env (overrides the stage's Box range)")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.output.exists():
    parser.error(f"refusing to overwrite {args_cli.output}")
if args_cli.controller == "ppo" and args_cli.checkpoint is None:
    parser.error("--checkpoint is required for --controller ppo")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.navigation import mdp as nav_mdp
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from nav_env_check_helpers import scripted_actions, scripted_staging_actions  # noqa: E402  (same directory)


STANCE_FORCE_N = 20.0  # a foot with a larger net contact force is in stance
SLIP_SPEED = 0.10  # m/s horizontal ankle-link speed while in stance counted as slip (proxy: includes foot pivoting)


class _Feet:
    def __init__(self, base) -> None:
        robot = base.scene["robot"]
        self.robot = robot
        self.body_ids = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.sensor = base.scene.sensors.get("contact_forces")
        self.sensor_ids = [list(self.sensor.body_names).index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")] if self.sensor else None

    def stance_speed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 2) horizontal ankle-link speed and (N, 2) stance mask."""
        speed = torch.linalg.vector_norm(self.robot.data.body_link_lin_vel_w[:, self.body_ids, :2], dim=-1)
        if self.sensor is None:
            stance = self.robot.data.body_link_pos_w[:, self.body_ids, 2] < 0.10
        else:
            stance = torch.linalg.vector_norm(self.sensor.data.net_forces_w[:, self.sensor_ids], dim=-1) > STANCE_FORCE_N
        return speed, stance


class WalkMonitor:
    """Navigation-phase motion quality per env (until the handoff hold): turning, speed, slip, gate / Homie switching."""

    def __init__(self, base) -> None:
        n, dev = base.num_envs, base.device
        self.base, self.feet = base, _Feet(base)
        self.term = base.action_manager.get_term("homie")
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.max_speed, self.max_wz, self.max_dwz = z(), z(), z()
        self.slip_steps, self.stance_steps, self.max_stance_speed, self.sum_stance_speed = z(), z(), z(), z()
        self.gate_toggles, self.mode_switches, self.steps = z(), z(), z()
        self.crab_sum, self.crab_n, self.yaw_flips, self.saturated, self.backward = z(), z(), z(), z(), z()
        self.back_time, self.back_dist, self.cap_sat, self.cmd_moving, self.wz_rev = z(), z(), z(), z(), z()
        self.prev_wz_dir = z()
        scale = self.term._scale
        scale_neg = self.term._scale_negative if self.term._scale_negative is not None else scale
        self.caps = (float(scale[0]), float(scale_neg[0]), float(scale[1]), float(scale[2]))
        self.prev_wz_sign = z()
        self.prev_wz = z()
        self.prev_gate = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_walk = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seen = torch.zeros(n, dtype=torch.bool, device=dev)
        # phase 5: translation while the goal is >= 90 deg off the heading (farther than 0.5 m), stop breaks near the target
        self.off_path, self.off_time, self.stop_breaks = z(), z(), z()
        self.first_stop = torch.full((n,), -1.0, device=dev)
        self.prev_zero = torch.zeros(n, dtype=torch.bool, device=dev)

    def update(self, mask: torch.Tensor, actions: torch.Tensor) -> None:
        data = self.base.scene["robot"].data
        m = mask.float()
        speed = torch.linalg.vector_norm(data.root_link_lin_vel_b[:, :2], dim=-1)
        wz = data.root_link_ang_vel_b[:, 2]
        dwz = torch.where(self.seen, torch.abs(wz - self.prev_wz) / self.base.step_dt, torch.zeros_like(wz))
        gate = actions[:, 3] > 0.0 if actions.shape[1] == 4 else torch.zeros_like(mask)
        walk = torch.linalg.vector_norm(self.term.processed_actions, dim=-1) >= 0.05
        self.gate_toggles += m * (self.seen & (gate != self.prev_gate)).float()
        self.mode_switches += m * (self.seen & (walk != self.prev_walk)).float()
        moving = speed > 0.2
        crab = torch.abs(torch.atan2(data.root_link_lin_vel_b[:, 1], data.root_link_lin_vel_b[:, 0]))
        self.crab_sum += m * moving.float() * crab
        self.crab_n += m * moving.float()
        self.backward += m * (moving & (data.root_link_lin_vel_b[:, 0] < 0.0)).float()
        wz_cmd = self.term.processed_actions[:, 2]
        sign = torch.where(torch.abs(wz_cmd) < 0.05, torch.zeros_like(wz_cmd), torch.sign(wz_cmd))
        flip = (sign != 0) & (self.prev_wz_sign != 0) & (sign != self.prev_wz_sign)
        self.yaw_flips += m * flip.float()
        self.prev_wz_sign = torch.where(mask & (sign != 0), sign, self.prev_wz_sign)
        self.saturated += m * ((torch.abs(actions[:, :3]) >= 0.99).any(-1) & ~gate).float()
        vx = data.root_link_lin_vel_b[:, 0]
        back = vx < -0.05
        self.back_time += m * back.float() * self.base.step_dt
        self.back_dist += m * torch.where(back, -vx, torch.zeros_like(vx)) * self.base.step_dt
        c = self.term.processed_actions
        cap_x = torch.where(c[:, 0] >= 0, torch.full_like(c[:, 0], self.caps[0]), torch.full_like(c[:, 0], self.caps[1]))
        at_cap = (torch.abs(c[:, 0]) >= 0.97 * cap_x) | (torch.abs(c[:, 1]) >= 0.97 * self.caps[2]) | (torch.abs(c[:, 2]) >= 0.97 * self.caps[3])
        moving_cmd = torch.linalg.vector_norm(c, dim=-1) >= 0.05
        self.cap_sat += m * (at_cap & moving_cmd).float()
        self.cmd_moving += m * moving_cmd.float()
        wz_dir = torch.where(wz > 0.5, torch.ones_like(wz), torch.where(wz < -0.5, -torch.ones_like(wz), torch.zeros_like(wz)))
        self.wz_rev += m * ((wz_dir != 0) & (self.prev_wz_dir != 0) & (wz_dir != self.prev_wz_dir)).float()
        self.prev_wz_dir = torch.where(mask & (wz_dir != 0), wz_dir, self.prev_wz_dir)
        self.max_speed = torch.where(mask, torch.maximum(self.max_speed, speed), self.max_speed)
        self.max_wz = torch.where(mask, torch.maximum(self.max_wz, torch.abs(wz)), self.max_wz)
        self.max_dwz = torch.where(mask, torch.maximum(self.max_dwz, dwz), self.max_dwz)
        foot_speed, stance = self.feet.stance_speed()
        stance_speed = torch.where(stance, foot_speed, torch.zeros_like(foot_speed))
        self.stance_steps += m * stance.float().sum(-1)
        self.slip_steps += m * (stance & (foot_speed > SLIP_SPEED)).float().sum(-1)
        self.sum_stance_speed += m * stance_speed.sum(-1)
        self.max_stance_speed = torch.where(mask, torch.maximum(self.max_stance_speed, stance_speed.amax(-1)), self.max_stance_speed)
        goal = nav_mdp.gt_goal_pos_b(self.base)
        dist = torch.linalg.vector_norm(goal, dim=-1)
        off = (goal[:, 0] <= 0.0) & (dist > 0.5)  # bearing >= 90 deg
        self.off_path += m * off.float() * speed * self.base.step_dt
        self.off_time += m * off.float() * self.base.step_dt
        zero = torch.linalg.vector_norm(c, dim=-1) < 1.0e-6
        near = dist < 0.3
        newly_stopped = mask & zero & near & (self.first_stop < 0)
        self.first_stop = torch.where(newly_stopped, self.steps, self.first_stop)
        self.stop_breaks += m * ((self.first_stop >= 0) & self.prev_zero & ~zero).float()
        self.prev_zero = torch.where(mask, zero, self.prev_zero)
        self.steps += m
        self.prev_wz = torch.where(mask, wz, self.prev_wz)
        self.prev_gate = torch.where(mask, gate, self.prev_gate)
        self.prev_walk = torch.where(mask, walk, self.prev_walk)
        self.seen |= mask

    def summary(self, i: int) -> dict:
        stance = max(1.0, float(self.stance_steps[i]))
        return {"max_speed": round(float(self.max_speed[i]), 3), "max_abs_yaw_rate": round(float(self.max_wz[i]), 3),
                "max_abs_yaw_accel": round(float(self.max_dwz[i]), 2), "stance_slip_fraction": round(float(self.slip_steps[i]) / stance, 4),
                "mean_stance_foot_speed": round(float(self.sum_stance_speed[i]) / stance, 4),
                "max_stance_foot_speed": round(float(self.max_stance_speed[i]), 3),
                "stop_gate_toggles": int(self.gate_toggles[i]), "homie_mode_switches": int(self.mode_switches[i]), "steps": int(self.steps[i]),
                "mean_abs_crab_deg_moving": round(float(self.crab_sum[i]) / max(1.0, float(self.crab_n[i])) * 57.29578, 2),
                "yaw_cmd_sign_flips_per_s": round(float(self.yaw_flips[i]) / max(1e-6, float(self.steps[i]) * self.base.step_dt), 3),
                "action_saturated_fraction": round(float(self.saturated[i]) / max(1.0, float(self.steps[i])), 4),
                "backward_fraction_moving": round(float(self.backward[i]) / max(1.0, float(self.crab_n[i])), 4),
                "backward_time_s": round(float(self.back_time[i]), 3), "backward_distance_m": round(float(self.back_dist[i]), 3),
                "cmd_any_axis_at_cap_fraction": round(float(self.cap_sat[i]) / max(1.0, float(self.cmd_moving[i])), 4),
                "yaw_rate_reversals_per_s": round(float(self.wz_rev[i]) / max(1e-6, float(self.steps[i]) * self.base.step_dt), 3),
                "off_heading_path_m": round(float(self.off_path[i]), 3), "off_heading_time_s": round(float(self.off_time[i]), 3),
                "stop_breaks_after_first_stop": int(self.stop_breaks[i]),
                "first_stop_s": round(float(self.first_stop[i]) * self.base.step_dt, 2) if float(self.first_stop[i]) >= 0 else None}


class PostHoldMonitor:
    """After the 25-step handoff hold (HandoffSuccess hold counter), keep the controller running ``n_steps`` and
    measure whether the robot stays stopped: base speed / yaw rate / tilt, drift from the handoff pose, window
    occupancy, stop-gate usage and stance-foot slip."""

    def __init__(self, base, n_steps: int) -> None:
        n, dev = base.num_envs, base.device
        self.base, self.n_steps, self.feet = base, n_steps, _Feet(base)
        self.hs = base.termination_manager.get_term_cfg("handoff_success").func
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.count = torch.zeros(n, dtype=torch.long, device=dev)
        self.handoff_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.anchor = torch.zeros(n, 2, device=dev)
        self.anchor_yaw = z()
        self.handoff_errors = torch.zeros(n, 3, device=dev)
        self.max_speed, self.max_wz, self.max_tilt, self.max_drift, self.max_yaw_drift = z(), z(), z(), z(), z()
        self.inside_steps, self.stop_steps, self.walk_steps, self.slip_steps, self.stance_steps, self.max_stance_speed = z(), z(), z(), z(), z(), z()
        self.height_min, self.height_max = torch.full((n,), 9.0, device=dev), z()
        self.failure = {}

    def update(self, active: torch.Tensor, actions: torch.Tensor, step: int) -> torch.Tensor:
        data = self.base.scene["robot"].data
        newly = active & ~self.started & (self.hs.hold >= 25)
        if bool(newly.any()):
            self.started |= newly
            self.handoff_step = torch.where(newly, torch.full_like(self.handoff_step, step + 1), self.handoff_step)
            self.anchor = torch.where(newly.unsqueeze(-1), data.root_link_pos_w[:, :2], self.anchor)
            self.anchor_yaw = torch.where(newly, nav_mdp.robot_yaw(self.base), self.anchor_yaw)
            self.handoff_errors = torch.where(newly.unsqueeze(-1), nav_mdp.handoff_errors(self.base), self.handoff_errors)
        m = active & self.started & ~newly
        mf = m.float()
        speed = torch.linalg.vector_norm(data.root_link_lin_vel_b[:, :2], dim=-1)
        wz = torch.abs(data.root_link_ang_vel_b[:, 2])
        tilt = torch.acos(torch.clamp(-data.projected_gravity_b[:, 2], -1.0, 1.0))
        drift = torch.linalg.vector_norm(data.root_link_pos_w[:, :2] - self.anchor, dim=-1)
        yaw_drift = torch.abs(nav_mdp.wrap_to_pi(nav_mdp.robot_yaw(self.base) - self.anchor_yaw))
        height = data.root_link_pos_w[:, 2] - self.base.scene.env_origins[:, 2]
        for name, value in (("max_speed", speed), ("max_wz", wz), ("max_tilt", tilt), ("max_drift", drift), ("max_yaw_drift", yaw_drift),
                            ("height_max", height)):
            setattr(self, name, torch.where(m, torch.maximum(getattr(self, name), value), getattr(self, name)))
        self.height_min = torch.where(m, torch.minimum(self.height_min, height), self.height_min)
        self.inside_steps += mf * self.hs.inside.float()
        gate = actions[:, 3] > 0.0 if actions.shape[1] == 4 else torch.zeros_like(m)
        self.stop_steps += mf * gate.float()
        self.walk_steps += mf * (torch.linalg.vector_norm(self.base.action_manager.get_term("homie").processed_actions, dim=-1) >= 0.05).float()
        foot_speed, stance = self.feet.stance_speed()
        self.stance_steps += mf * stance.float().sum(-1)
        self.slip_steps += mf * (stance & (foot_speed > SLIP_SPEED)).float().sum(-1)
        stance_speed = torch.where(stance, foot_speed, torch.zeros_like(foot_speed)).amax(-1)
        self.max_stance_speed = torch.where(m, torch.maximum(self.max_stance_speed, stance_speed), self.max_stance_speed)
        self.count += m.long()
        return m & (self.count >= self.n_steps)

    def fail(self, i: int, reason: str) -> None:
        self.failure[i] = reason

    def summary(self, i: int) -> dict:
        steps = max(1, int(self.count[i]))
        return {"post_hold_steps": int(self.count[i]), "failure": self.failure.get(i), "max_speed": round(float(self.max_speed[i]), 4),
                "max_abs_yaw_rate": round(float(self.max_wz[i]), 4), "max_tilt": round(float(self.max_tilt[i]), 4),
                "max_drift_m": round(float(self.max_drift[i]), 4), "max_yaw_drift_rad": round(float(self.max_yaw_drift[i]), 4),
                "height_range": [round(float(self.height_min[i]), 4), round(float(self.height_max[i]), 4)],
                "inside_window_fraction": round(float(self.inside_steps[i]) / steps, 4), "stop_gate_fraction": round(float(self.stop_steps[i]) / steps, 4),
                "homie_walk_fraction": round(float(self.walk_steps[i]) / steps, 4),
                "stance_slip_fraction": round(float(self.slip_steps[i]) / max(1.0, float(self.stance_steps[i])), 4),
                "max_stance_foot_speed": round(float(self.max_stance_speed[i]), 4)}


def main() -> None:
    nav_target = nav_mdp.apply_nav_target_overrides(args_cli.nav_target_offset_x, args_cli.nav_handoff_x_range)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point")
    cfg.seed = args_cli.seed
    cfg.episode_length_s = args_cli.episode_seconds
    override = None
    if args_cli.override:
        raw = json.loads(args_cli.override)
        override = {"r": tuple(raw["r"]), "bearing": math.radians(raw.get("bearing_deg", 90.0)), "yaw": float(raw.get("yaw", math.pi)),
                    "box_x": float(raw.get("box_x", 0.0)), "box_y": float(raw.get("box_y", 0.0))}
    cfg.events.reset_box_and_robot.params = {"fixed_stage": args_cli.stage, "mix_prob": 0.0, "start_range_override": override}
    if args_cli.box_offset is not None:
        # a zero-width Box range around the requested offset (reset_box_and_robot samples centre +- half-range)
        base_params = override or dict(nav_mdp.STAGES[args_cli.stage])
        override = {**base_params, "box_x": 0.0, "box_y": 0.0}
        cfg.events.reset_box_and_robot.params["start_range_override"] = override
        nav_mdp.BOX_CENTER_OFFSET = (float(args_cli.box_offset[0]), float(args_cli.box_offset[1]))
    if args_cli.table_contact_sensors:
        from table_contact import add_table_contact_sensors  # same directory

        add_table_contact_sensors(cfg.scene)
    if args_cli.handoff_criterion == "stable50":
        from isaaclab.managers import TerminationTermCfg

        cfg.terminations.handoff_success = TerminationTermCfg(
            func=nav_mdp.StableHandoffSuccess, params={"hold_steps": 50, "speed_max": 0.05, "yaw_rate_max": 0.10, "leg_joint_speed_max": 0.3})
    if args_cli.post_hold_steps:
        # The HandoffSuccess term still evaluates the window and its hold counter every step; it just never terminates.
        cfg.terminations.handoff_success.params["hold_steps"] = 10**9
        cfg.episode_length_s += args_cli.post_hold_steps * cfg.sim.dt * cfg.decimation
    env = gym.make(args_cli.task, cfg=cfg)
    policy = None
    if args_cli.controller == "ppo":
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit

        from unitree_rl_lab.tasks.navigation.agents import BoxNavHomiePPORunnerCfg

        agent_cfg = BoxNavHomiePPORunnerCfg()
        agent_cfg.device = args_cli.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args_cli.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=args_cli.device)
        if args_cli.export_jit is not None:
            export_policy_as_jit(runner.alg.policy, runner.alg.policy.actor_obs_normalizer, str(args_cli.export_jit.parent), args_cli.export_jit.name)
        stepper = wrapped
    else:
        stepper = None
    base = env.unwrapped
    term = base.action_manager.get_term("homie")
    n = base.num_envs
    if stepper is not None:
        obs = stepper.get_observations()  # the wrapper already reset the env
    else:
        env.reset()
    start = {"handoff_errors": nav_mdp.handoff_errors(base).tolist(), "distance": nav_mdp.target_errors(base)[0].tolist(),
             "root_xy": (base.scene["robot"].data.root_link_pos_w[:, :2] - base.scene.env_origins[:, :2]).tolist(),
             "box_xy": (base.scene["box"].data.root_link_pos_w[:, :2] - base.scene.env_origins[:, :2]).tolist()}
    contact_body = [None] * n
    done = torch.zeros(n, dtype=torch.bool, device=base.device)
    outcome = ["timeout"] * n
    steps = [None] * n
    final = [None] * n
    max_steps = int(base.max_episode_length)
    tm = base.termination_manager
    # every termination term except the time-out and the success (v6 adds "contact")
    failure_terms = [name for name in tm.active_terms if name not in ("time_out", "handoff_success")]
    post = PostHoldMonitor(base, args_cli.post_hold_steps) if args_cli.post_hold_steps else None
    probe = None
    if args_cli.table_contact_sensors and "contact" in tm.active_terms:
        from table_contact import TABLE_CONTACT_LINKS, table_forces

        class _ContactProbe:
            """Wraps the contact termination term: same decision, plus the table-filtered forces at that moment (the
            env resets terminated envs inside step(), which clears the sensor buffers before step() returns)."""

            def __init__(self, inner) -> None:
                self.inner = inner
                self.table = torch.zeros(n, len(TABLE_CONTACT_LINKS), device=base.device)

            def __call__(self, env, **params):
                out = self.inner(env, **params)
                self.table = table_forces(env)
                return out

            def reset(self, env_ids=None):
                return self.inner.reset(env_ids)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        contact_cfg = tm.get_term_cfg("contact")
        probe = _ContactProbe(contact_cfg.func)
        contact_cfg.func = probe
    table_contact = [None] * n
    walk = WalkMonitor(base)
    for step in range(max_steps):
        errors = nav_mdp.handoff_errors(base).clone()
        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, extras = stepper.step(actions)
            truncated = extras.get("time_outs", torch.zeros_like(dones)).bool()
        else:
            controller = scripted_staging_actions if args_cli.controller == "scripted_staging" else scripted_actions
            actions = controller(base, term._scale, args_cli.max_forward)
            _, _, _, truncated, _ = env.step(actions)
        # Envs that terminate this step were auto-reset by the env: their post-step state belongs to a new episode.
        terminated_now = torch.zeros_like(done)
        for name in failure_terms:
            terminated_now |= tm.get_term(name)
        terminated_now |= truncated.bool()
        if post is None:
            terminated_now |= tm.get_term("handoff_success")
        active = ~done & ~terminated_now
        walk.update(active & ~(post.started if post is not None else torch.zeros_like(done)), actions)
        if post is not None:
            finished = post.update(active, actions, step)
            for i in torch.nonzero(finished & ~done).flatten().tolist():
                outcome[i], final[i], steps[i] = "handoff_success", [round(float(v), 4) for v in post.handoff_errors[i]], int(post.handoff_step[i])
            done |= finished
            # A fall / collision / time-out after the handoff hold is a post-hold failure of a successful handoff.
            for i in torch.nonzero(terminated_now & post.started & ~done).flatten().tolist():
                post.fail(i, "post_hold_" + next(nm for nm in failure_terms if bool(tm.get_term(nm)[i])) if not bool(truncated[i]) else "post_hold_timeout")
                outcome[i], final[i], steps[i] = "handoff_success", [round(float(v), 4) for v in post.handoff_errors[i]], int(post.handoff_step[i])
                done[i] = True
        for name in ("handoff_success", *failure_terms):
            if name == "handoff_success" and post is not None:
                continue
            fired = tm.get_term(name) & ~done
            for i in torch.nonzero(fired).flatten().tolist():
                outcome[i], final[i], steps[i] = name, [round(float(v), 4) for v in errors[i]], step + 1
                if name == "contact":  # diagnostics: the body with the largest contact force at termination
                    term_c = tm.get_term_cfg("contact").func
                    contact_body[i] = term_c.body_names[int(term_c.last_body[i])]
                    if probe is not None:
                        force = probe.table[i]
                        first_stop = walk.first_stop[i]
                        table_contact[i] = {"table_N": round(float(force.max()), 2),
                                            "table_link": TABLE_CONTACT_LINKS[int(force.argmax())] if float(force.max()) > 1.0 else None,
                                            "after_first_stop": bool(float(first_stop) >= 0.0),
                                            "contact_s": round((step + 1) * base.step_dt, 2)}
            done |= fired
        timed = truncated.bool() & ~done
        for i in torch.nonzero(timed).flatten().tolist():
            final[i], steps[i] = [round(float(v), 4) for v in errors[i]], step + 1
        done |= timed
        if bool(done.all()):
            break
    ok = [i for i in range(n) if outcome[i] == "handoff_success"]
    ok_steps = sorted(steps[i] for i in ok)
    finals = [final[i] for i in ok]
    report = {
        "controller": args_cli.controller, "checkpoint": str(args_cli.checkpoint) if args_cli.checkpoint else None,
        "seed": args_cli.seed, "stage": args_cli.stage, "override": override, "num_envs": n, "nav_target": nav_target,
        "task": args_cli.task, "handoff_criterion": args_cli.handoff_criterion, "box_offset": args_cli.box_offset,
        "outcomes": {k: outcome.count(k) for k in sorted(set(outcome))},
        "handoff_success_rate": round(len(ok) / n, 4),
        "fall_rate": round(outcome.count("fall") / n, 4),
        "steps_to_success": {"median": ok_steps[len(ok_steps) // 2] if ok_steps else None, "max": ok_steps[-1] if ok_steps else None},
        "final_errors_success_abs_max": [round(max(abs(f[k]) for f in finals), 4) for k in range(3)] if finals else None,
        "episodes": [{"outcome": outcome[i], "steps": steps[i], "final_handoff_errors": final[i], "start_distance": round(start["distance"][i], 3),
                      "start_handoff_errors": [round(v, 4) for v in start["handoff_errors"][i]], "walk": walk.summary(i),
                      "box_xy": [round(v, 4) for v in start["box_xy"][i]], "contact_body": contact_body[i], "table_contact": table_contact[i],
                      **({"post_hold": post.summary(i)} if post is not None and outcome[i] == "handoff_success" else {})} for i in range(n)],
    }
    report["slip_definition"] = {"stance_force_N": STANCE_FORCE_N, "slip_speed_mps": SLIP_SPEED,
                                 "note": "horizontal ankle_roll_link speed while its net contact force > threshold (proxy; foot pivoting counts)"}

    def describe(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]  # noqa: E731
        return {"median": round(pick(0.5), 4), "p95": round(pick(0.95), 4), "max": round(values[-1], 4)}

    walks = [report["episodes"][i]["walk"] for i in ok]
    report["walk_aggregate_success"] = {k: describe([w[k] for w in walks]) for k in walks[0]} if walks else None
    if post is not None:
        posts = [report["episodes"][i]["post_hold"] for i in ok]
        report["post_hold_steps"] = args_cli.post_hold_steps
        report["post_hold_failures"] = {r: sum(p["failure"] == r for p in posts) for r in sorted({p["failure"] for p in posts if p["failure"]})}
        report["post_hold_aggregate"] = {k: describe([p[k] for p in posts if p["failure"] is None])
                                         for k in posts[0] if k not in ("failure", "height_range", "post_hold_steps")} if posts else None
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=1))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
