#!/usr/bin/env python3
"""Measure uninterrupted Teacher approach -> BC2 -> BC3 -> rule-based Bin placement.

Each trial uses a fresh simulator, without snapshot restore or mid-trial resets.
This configuration does NOT contain the unavailable team BC1/Nav checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def live_result(document, expected_seed, mass, method):
    config = document["evaluator_config"]
    if (config.get("snapshot_selection") or document.get("snapshot_metrics")
            or not config.get("rule_based_place") or config.get("seed") != expected_seed
            or config.get("box_mass_kg") != mass
            or bool(config.get("residual_checkpoint")) != (method == "grip")):
        raise ValueError("Result is not the requested uninterrupted full-task trial")
    episodes = document["episode_results"]
    if len(episodes) != 1:
        raise ValueError("Expected one complete trial per simulator")
    episode = episodes[0]
    grasp = episode.get("success")
    carry = episode.get("carry", {}).get("success")
    place = episode.get("place", {}).get("success")
    if any(type(value) is not bool for value in (grasp, carry, place)):
        raise ValueError("Incomplete task record; do not count an infrastructure error as a failure")
    if (carry and not grasp) or (place and not carry):
        raise ValueError("Impossible stage ordering")
    success = grasp and carry and place
    if episode.get("chain_success") is not success:
        raise ValueError("Full-task outcome disagrees with individual stages")
    stage = ("approach_or_grasp" if not grasp else "carry" if not carry else "place" if not place else None)
    reason = (episode.get("reason") if not grasp else episode["carry"].get("reason") if not carry
              else episode["place"].get("reason") if not place else "full_task_success")
    return {"seed": expected_seed, "success": success, "grasp_success": grasp,
            "carry_success": carry, "place_success": place, "failure_stage": stage, "reason": reason}


def report(records, planned):
    n = len(records)
    return {"planned_trials": planned, "completed_trials": n, "complete": n == planned,
            "full_task_successes": sum(r["success"] for r in records),
            "full_task_success_rate": sum(r["success"] for r in records) / n if n else None,
            "failure_stages": dict(Counter(r["failure_stage"] for r in records if not r["success"])),
            "failure_reasons": dict(Counter(r["reason"] for r in records if not r["success"])),
            "stage_successes_over_all_starts": {stage: sum(r[stage + "_success"] for r in records)
                                                for stage in ("grasp", "carry", "place")},
            "trials": records}


def guarded_run(command, log, env, cwd):
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=cwd, env=env, start_new_session=True)
        started = changed = time.monotonic()
        size = 0
        try:
            while process.poll() is None:
                time.sleep(1)
                current = log.stat().st_size
                if current != size:
                    changed, size = time.monotonic(), current
                if time.monotonic() - changed > 300 or time.monotonic() - started > 1200:
                    raise TimeoutError(f"Simulator timeout; inspect {log}")
            if process.returncode:
                raise RuntimeError(f"Simulator exit {process.returncode}; inspect {log}")
        finally:
            # Also clean up the worker if its simulator exited unexpectedly.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=250100)
    parser.add_argument("--box_mass_kg", type=float, choices=(0.2, 0.3), default=0.2)
    parser.add_argument("--methods", nargs="+", choices=("baseline", "grip"), default=["grip"])
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs/full_task_r01/eval_0p2kg")
    parser.add_argument("--runtime_root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/teacher/bc/bc2_grasp_lift_43d.pt")
    parser.add_argument("--carry_checkpoint", type=Path, help="BC3 checkpoint for the direct handoff")
    parser.add_argument("--residual_checkpoint", type=Path, help="Grip residual checkpoint used by the grip method")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    args.runtime_root = args.runtime_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.carry_checkpoint = (
        args.carry_checkpoint or args.runtime_root / "outputs/bc3_direct_r01/bc3_direct_mix.pt"
    ).expanduser().resolve()
    if "grip" in args.methods:
        if args.residual_checkpoint is None:
            parser.error("the grip method requires --residual_checkpoint")
        args.residual_checkpoint = args.residual_checkpoint.expanduser().resolve()
    evaluator = args.runtime_root / "scripts/teacher/evaluate_bc2_rollout.py"
    files = [args.checkpoint, args.carry_checkpoint, evaluator]
    if "grip" in args.methods:
        files += [args.residual_checkpoint, args.residual_checkpoint.parent / "params/env.yaml"]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        parser.error("Missing inputs: " + ", ".join(missing))
    # Pin runtime sources as well as weights: resume must not mix changed code.
    sources = list((args.runtime_root / "source/unitree_rl_lab").rglob("*.py"))
    manifest = {"format": "unitree_full_task_evaluation_v1", "episodes": args.episodes,
                "seed": args.seed, "mass_kg": args.box_mass_kg, "methods": args.methods,
                "scope": "teacher_near_approach_bc2_mixed_bc3_rule_based_place",
                "team_nav_bc1_included": False, "mid_episode_resets": False,
                "snapshot_restore": False, "carry_seconds": 36, "place_seconds": 16,
                "wbc_bridge_seconds": 0, "carry_blend_steps": 50,
                "input_sha256": {str(f.resolve()): sha(f) for f in files + sources},
                "launcher_sha256": sha(__file__)}
    if args.check_only:
        print(json.dumps({"ready": True, "scope": manifest["scope"], "episodes": args.episodes,
                          "methods": args.methods, "team_nav_bc1_included": False}, indent=2))
        return
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    previous = out / "manifest.json"
    if previous.exists():
        if not args.resume or json.loads(previous.read_text()) != manifest:
            parser.error("Existing output requires --resume and identical code, models and protocol")
    else:
        write_json(previous, manifest)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.runtime_root / "source/unitree_rl_lab") + os.pathsep + str(args.runtime_root)
    env["UNITREE_WORKSPACE"] = str(ROOT)
    env["UNITREE_ARENA_ROOT"] = str(ROOT / "external/IsaacLab-Arena")
    records = {method: [] for method in args.methods}
    for index in range(args.episodes):
        seed = args.seed + index
        for method in args.methods:
            trial = out / method / f"trial_{index:04d}"
            trial.mkdir(parents=True, exist_ok=True)
            result_path = trial / "result.json"
            print(f"[full-task] trial={index + 1}/{args.episodes} method={method} seed={seed}", flush=True)
            if not result_path.exists():
                command = [sys.executable, "-u", str(evaluator), "--checkpoint", str(args.checkpoint),
                           "--carry_checkpoint", str(args.carry_checkpoint), "--episodes", "1", "--seed", str(seed),
                           "--box_mass_kg", str(args.box_mass_kg), "--episode_seconds", "20", "--carry_seconds", "36",
                           "--wbc_bridge_seconds", "0", "--carry_blend_steps", "50", "--rule_based_place",
                           "--place_seconds", "16", "--headless", "--device", args.device, "--output", str(result_path)]
                if method == "grip":
                    command += ["--residual_checkpoint", str(args.residual_checkpoint)]
                write_json(trial / "command.json", command)
                guarded_run(command, trial / "run.log", env, args.runtime_root)
            result = live_result(json.loads(result_path.read_text()), seed, args.box_mass_kg, method)
            records[method].append(result)
            summaries = {m: report(r, args.episodes) for m, r in records.items()}
            write_json(out / "summary.json", {"scope": manifest["scope"], "methods": summaries})
            print(f"[full-task] {method}: {summaries[method]['full_task_successes']}/{len(records[method])} "
                  f"last={result['reason']}", flush=True)


if __name__ == "__main__":
    main()
