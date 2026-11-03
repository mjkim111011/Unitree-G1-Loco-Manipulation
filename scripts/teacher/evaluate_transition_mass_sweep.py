#!/usr/bin/env python3
"""Evaluate four frozen transition methods at fixed physical Box masses.

This is a subprocess orchestrator, not an Isaac Sim process. Each condition
has its own metrics and console log; --resume validates existing results before
reusing them. No policy is retrained and placing is excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("instant", "fixed_1s", "stage1", "stage2")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(data: dict, mass: float, method: str) -> dict:
    episodes = data["episode_results"]
    grasp = sum(bool(e["success"]) for e in episodes)
    reached = sum(
        any(r.get("carry_step", -1) >= 0 for r in e.get("carry", {}).get("trace", []))
        for e in episodes
    )
    successes = sum(bool(e.get("carry", {}).get("success")) for e in episodes)
    reasons = Counter(
        e.get("carry", {}).get("reason", e.get("reason"))
        for e in episodes if not e.get("carry", {}).get("success")
    )
    jumps, tilts, takeover_times = [], [], []
    step_dt = 1.0 / data.get("evaluator_config", {}).get("control_frequency_hz", 50.0)
    for episode in episodes:
        trace = episode.get("carry", {}).get("trace", [])
        boundary = [r for r in trace if 0 <= r.get("carry_step", -1) * step_dt < 2.0]
        for key, values in (("target_jump_rad", jumps), ("root_tilt_rad", tilts)):
            measured = [r[key] for r in boundary if key in r]
            if measured:
                values.append(max(measured))
        done = next((r for r in trace if r.get("carry_step", -1) >= 0 and r.get("bc3_blend_weight", 0) >= 1 - 1e-5), None)
        if done is not None:
            takeover_times.append((done["carry_step"] + 1) * step_dt)

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "mass_kg": mass, "method": method, "episodes": len(episodes),
        "grasp_successes": grasp, "transition_reached": reached,
        "pre_transition_failures": len(episodes) - reached,
        "carry_successes": successes,
        "full_chain_success_rate": successes / len(episodes),
        "carry_success_given_transition": successes / reached if reached else None,
        "box_drop_failures": reasons["box_drop"],
        "carry_horizon_failures": reasons["carry_horizon"],
        "mean_takeover_seconds": mean(takeover_times),
        "mean_first2s_peak_target_jump_rad": mean(jumps),
        "mean_first2s_peak_root_tilt_rad": mean(tilts),
        "failure_reasons": json.dumps(dict(reasons), sort_keys=True),
    }


def validate_result(data: dict, *, mass: float, method: str, args, hashes: dict) -> None:
    config = data["evaluator_config"]
    expected = {
        "box_mass_kg": mass, "episodes": args.episodes, "seed": args.seed,
        "episode_seconds": args.episode_seconds, "carry_seconds": args.carry_seconds,
        "box_size_m": 0.2, "box_x_range_m": [0.618, 0.622],
        "box_y_range_m": [-0.002, 0.002], "rule_based_place": False,
        "carry_blend_steps": 50 if method == "fixed_1s" else 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"result configuration mismatch: {key}={config.get(key)!r}, expected {value!r}")
    for key, name in (("checkpoint_sha256", "bc2"), ("carry_checkpoint_sha256", "bc3")):
        if data.get(key) != hashes[name]:
            raise ValueError(f"result checkpoint mismatch: {name}")
    if config.get("transition_checkpoint_sha256") != hashes.get(method):
        raise ValueError("result transition checkpoint mismatch")
    episodes = data["episode_results"]
    if len(episodes) != args.episodes or [e["episode"] for e in episodes] != list(range(args.episodes)):
        raise ValueError("result does not contain every requested episode")
    if any(not math.isclose(e.get("actual_box_mass_kg", math.nan), mass, rel_tol=1e-4) for e in episodes):
        raise ValueError("result has missing or incorrect measured physical Box mass")


def command_for(args, mass: float, method: str, output: Path) -> list[str]:
    command = [
        sys.executable, "-u", str(ROOT / "scripts/teacher/evaluate_bc2_rollout.py"),
        "--checkpoint", str(args.checkpoint),
        "--carry_checkpoint", str(args.carry_checkpoint),
        "--box_mass_kg", str(mass), "--episodes", str(args.episodes),
        "--seed", str(args.seed), "--episode_seconds", str(args.episode_seconds),
        "--carry_seconds", str(args.carry_seconds), "--output", str(output),
        "--device", args.device,
    ]
    if method in ("stage1", "stage2"):
        command += ["--transition_checkpoint", str(getattr(args, f"{method}_checkpoint"))]
    else:
        command += ["--carry_blend_steps", "50" if method == "fixed_1s" else "0"]
    if args.headless:
        command += ["--headless"]
    return command


def write_summary(root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/teacher/bc/bc2_grasp_lift_43d.pt")
    parser.add_argument("--carry_checkpoint", type=Path, default=ROOT / "data/teacher/carry_expanded_r01/bc3_carry_candidate.pt")
    parser.add_argument("--stage1_checkpoint", type=Path, required=True)
    parser.add_argument("--stage2_checkpoint", type=Path, required=True)
    parser.add_argument("--masses", type=float, nargs="+", default=[0.2, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per mass and method")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--episode_seconds", type=float, default=20.0)
    parser.add_argument("--carry_seconds", type=float, default=16.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs/robustness/transition_mass_sweep_r01")
    parser.add_argument("--resume", action="store_true", help="Reuse only matching, complete result files")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without launching Isaac Sim")
    args = parser.parse_args()
    if args.episodes <= 0 or any(not math.isfinite(v) or v <= 0 for v in [*args.masses, args.episode_seconds, args.carry_seconds]):
        parser.error("masses, episodes and durations must be finite and positive")
    if len(set(args.masses)) != len(args.masses):
        parser.error("duplicate masses are not allowed")
    hashes = {}
    for name, key in (("bc2", "checkpoint"), ("bc3", "carry_checkpoint"),
                      ("stage1", "stage1_checkpoint"), ("stage2", "stage2_checkpoint")):
        path = getattr(args, key).expanduser().resolve()
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")
        setattr(args, key, path)
        hashes[name] = sha256(path)
    args.output_root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.masses) * len(METHODS)
    for index, (mass, method) in enumerate(((m, k) for m in args.masses for k in METHODS), 1):
        directory = args.output_root / f"mass_{str(mass).replace('.', 'p')}kg"
        output = directory / f"{method}.json"
        command = command_for(args, mass, method, output)
        print(f"[sweep] {index}/{total}: mass={mass}kg method={method}", flush=True)
        if args.dry_run:
            print(shlex.join(command))
            continue
        if output.exists():
            if not args.resume:
                raise FileExistsError(f"{output} exists; use --resume or a new --output_root")
            data = json.loads(output.read_text())
            validate_result(data, mass=mass, method=method, args=args, hashes=hashes)
            print("[sweep] reusing verified complete result", flush=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            log_path = output.with_suffix(".log")
            print(f"[sweep] console log: {log_path}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith(("[chain]", "[progress]", "[snapshot]")):
                            print(line, end="", flush=True)
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if returncode:
                raise RuntimeError(f"evaluation failed (exit={returncode}); inspect {log_path}. Completed results remain reusable with --resume.")
            data = json.loads(output.read_text())
            validate_result(data, mass=mass, method=method, args=args, hashes=hashes)
        row = summarize(data, mass, method)
        rows.append(row)
        write_summary(args.output_root, rows)
        print(f"[sweep] success={row['carry_successes']}/{row['episodes']}; transition reached={row['transition_reached']}", flush=True)
    if not args.dry_run:
        print(f"[sweep] summary: {args.output_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
