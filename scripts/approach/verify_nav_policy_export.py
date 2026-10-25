#!/usr/bin/env python3
"""Check that the TorchScript navigation policy used by the WBC evaluator is exactly the PPO checkpoint.

Compares (1) every actor / observation-normalizer tensor of the RSL-RL checkpoint with the TorchScript
module, bit for bit, and (2) the TorchScript output with a re-implementation of the deterministic actor
(EmpiricalNormalization (x - mean) / (std + 1e-2), MLP {18|21}-256-128-64-4 with ELU) on random inputs and,
optionally, on observations recorded in WBC rollouts (--trace_glob, *.npz with a ``ppo_obs`` array).
CPU only, no simulator.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manual_actor(state: dict, obs: torch.Tensor) -> torch.Tensor:
    x = (obs - state["actor_obs_normalizer._mean"]) / (state["actor_obs_normalizer._std"] + 1.0e-2)
    for i in (0, 2, 4, 6):
        x = x @ state[f"actor.{i}.weight"].T + state[f"actor.{i}.bias"]
        if i != 6:
            x = torch.nn.functional.elu(x)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jit", type=Path, required=True)
    parser.add_argument("--trace_glob", default=None, help="WBC trace files (*.npz) whose ppo_obs rows are re-evaluated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    jit = torch.jit.load(str(args.jit), map_location="cpu").eval()
    jit_state = jit.state_dict()
    pairs = {f"actor.{i}.{p}": f"actor.{i}.{p}" for i in (0, 2, 4, 6) for p in ("weight", "bias")}
    pairs.update({f"actor_obs_normalizer.{k}": f"normalizer.{k}" for k in ("_mean", "_var", "_std", "count")})
    tensor_equal = {k: bool(torch.equal(state[k], jit_state[v])) for k, v in pairs.items()}

    generator = torch.Generator().manual_seed(0)
    mean, std = state["actor_obs_normalizer._mean"], state["actor_obs_normalizer._std"]
    random_obs = mean + 3.0 * std * torch.randn(20000, mean.shape[-1], generator=generator)  # 18 (v5) or 21 (v6/v7)
    with torch.inference_mode():
        diff_random = float((jit(random_obs) - manual_actor(state, random_obs)).abs().max())
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_iter": int(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["iter"]),
        "jit": str(args.jit), "jit_sha256": sha256(args.jit),
        "all_actor_and_normalizer_tensors_bit_identical": all(tensor_equal.values()), "tensor_equal": tensor_equal,
        "random_obs": {"n": int(random_obs.shape[0]), "max_abs_diff_jit_vs_manual": diff_random},
        "actor_std_parameter (exploration only, unused by the deterministic policy)": state["std"].tolist(),
    }
    if args.trace_glob:
        obs_rows, act_rows, files = [], [], sorted(glob.glob(args.trace_glob))
        for f in files:
            d = np.load(f)
            keep = np.isfinite(d["ppo_obs"]).all(axis=1) & np.isfinite(d["ppo_action"]).all(axis=1)
            obs_rows.append(d["ppo_obs"][keep])
            act_rows.append(d["ppo_action"][keep])
        if obs_rows and sum(len(r) for r in obs_rows):
            recorded = torch.as_tensor(np.concatenate(obs_rows), dtype=torch.float32)
            applied = torch.as_tensor(np.concatenate(act_rows), dtype=torch.float32)
            with torch.inference_mode():
                recomputed = torch.clamp(manual_actor(state, recorded), -1.0, 1.0)
                report["recorded_rollouts"] = {
                    "trace_files": len(files), "steps": int(recorded.shape[0]),
                    "max_abs_diff_jit_vs_manual": float((jit(recorded) - manual_actor(state, recorded)).abs().max()),
                    "max_abs_diff_applied_action_vs_checkpoint(clamped)": float((recomputed - applied).abs().max()),
                    "stop_gate_on_fraction": float((applied[:, 3] > 0).float().mean()),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
