#!/usr/bin/env python3
"""Train the goal-conditioned Homie navigation policy with PPO (RSL-RL OnPolicyRunner).

Task: Unitree-G1-43dof-BoxNav-Homie (tasks/navigation).  PPO outputs [vx, vy, yaw_rate] for the frozen
Homie V2 lower body; reward = BC-1 v2 handoff state (see tasks/navigation/mdp.py).  The start-distance
curriculum (S1 0.5-1 m -> S2 1-2 m -> S3 3-5 m -> S4 Box offsets on the table) advances on the rolling
handoff success rate.  Logs (TensorBoard + checkpoints + configs) go to a new directory under
outputs/approach_p2/ppo/; an existing directory is never reused.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Unitree-G1-43dof-BoxNav-Homie")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run_name", required=True)
parser.add_argument("--resume_checkpoint", type=Path, default=None, help="continue training from a model_*.pt (new run dir)")
parser.add_argument("--start_stage", type=int, default=0)
parser.add_argument("--entropy_coef", type=float, default=None, help="override the PPO entropy coefficient")
parser.add_argument("--set_std", type=float, default=None, help="after loading, set the action noise std (fine-tuning)")
parser.add_argument("--nav_target_offset_x", type=float, default=None, help="navigation target point offset from the canonical start (m); default -0.03")
parser.add_argument("--nav_handoff_x_range", type=float, nargs=2, default=None, help="handoff success window in x (m); default -0.08 0.015")
parser.add_argument("--log_root", type=Path, default=Path("outputs/approach_p2/ppo"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.navigation.agents import BoxNavHomiePPORunnerCfg
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main() -> None:
    from unitree_rl_lab.tasks.navigation import mdp as nav_mdp

    nav_target = nav_mdp.apply_nav_target_overrides(args_cli.nav_target_offset_x, args_cli.nav_handoff_x_range)
    print(f"[INFO] navigation target {nav_target}", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = BoxNavHomiePPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = args_cli.device
    agent_cfg.run_name = args_cli.run_name
    env_cfg.seed = agent_cfg.seed
    env_cfg.curriculum.stage.params["start_stage"] = args_cli.start_stage
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef

    log_dir = (args_cli.log_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}").resolve()
    if log_dir.exists():
        raise SystemExit(f"refusing to reuse {log_dir}")
    log_dir.mkdir(parents=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if args_cli.resume_checkpoint is not None:
        print(f"[INFO] resuming weights from {args_cli.resume_checkpoint}")
        runner.load(str(args_cli.resume_checkpoint))
        if args_cli.set_std is not None:
            import torch

            policy = runner.alg.policy
            with torch.no_grad():
                if hasattr(policy, "std"):
                    policy.std.fill_(args_cli.set_std)
                else:
                    policy.log_std.fill_(float(torch.log(torch.tensor(args_cli.set_std))))
            print(f"[INFO] action noise std set to {args_cli.set_std}")
    else:
        # Start from "stand still" (zero mean command) instead of a random, env-correlated walking bias.
        import torch

        last = [m for m in runner.alg.policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        if last.out_features == 4:
            with torch.no_grad():
                last.bias[3] = -0.5  # stop gate mostly off at the start
        print("[INFO] zero-initialised the actor output layer (mean command 0; stop-gate bias -0.5)")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    (log_dir / "params" / "nav_target.json").write_text(__import__("json").dumps(nav_target) + "\n")
    print(f"[INFO] log dir {log_dir}", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
