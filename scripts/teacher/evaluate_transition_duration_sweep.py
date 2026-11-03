#!/usr/bin/env python3
"""Paired fixed-duration experiments from identical saved transition starts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEP_DT = 0.02


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def duration_steps(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("durations must be finite and positive")
    steps = round(seconds / STEP_DT)
    if steps < 1 or not math.isclose(steps * STEP_DT, seconds, abs_tol=1e-9):
        raise ValueError("durations must be positive multiples of 0.02 seconds")
    return steps


def save_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def paired_reports(results):
    """Only compare states whose common warm-up passes for every duration."""
    durations = list(results)
    reference = results[durations[0]]["episode_results"]
    ids = [e["snapshot_id"] for e in reference]
    for data in results.values():
        if [e["snapshot_id"] for e in data["episode_results"]] != ids:
            raise ValueError("snapshot selection/order differs across durations")
    common = [all(results[t]["episode_results"][i]["restore_success"] for t in durations) for i in range(len(ids))]
    states = []
    for i, snapshot_id in enumerate(ids):
        initial = reference[i]["start_features"]
        successful = [t for t in durations if results[t]["episode_results"][i]["carry"]["success"]]
        joint_delta = box_delta = 0.0
        for data in results.values():
            other = data["episode_results"][i]["start_features"]
            joint_delta = max(joint_delta, max(abs(a - b) for a, b in zip(initial["joint_position"], other["joint_position"])))
            box_delta = max(box_delta, max(abs(a - b) for a, b in zip(initial["box_state"][:3], other["box_state"][:3])))
        row = {
            "state_index": i, "snapshot_id": snapshot_id,
            "common_valid_restore": common[i],
            "robot_com_x_m": initial["robot_com_relative_feet_yaw_xy_m"][0],
            "robot_com_y_m": initial["robot_com_relative_feet_yaw_xy_m"][1],
            "left_foot_load_fraction": initial["foot_load_fraction"][0],
            "right_foot_load_fraction": initial["foot_load_fraction"][1],
            "angular_speed_rad_s": math.sqrt(sum(v*v for v in initial["base_angular_velocity_body"])),
            "postwarm_joint_difference_max_rad": joint_delta,
            "postwarm_box_position_difference_max_m": box_delta,
            "successful_durations_s": json.dumps(successful),
        }
        for t in durations:
            episode = results[t]["episode_results"][i]
            row[f"success_{t:g}s"] = int(episode["carry"]["success"]) if episode["restore_success"] else "restore_invalid"
            row[f"reason_{t:g}s"] = episode["carry"]["reason"]
        states.append(row)
    summary = []
    for t, data in results.items():
        episodes = data["episode_results"]
        successes = sum(e["carry"]["success"] for i, e in enumerate(episodes) if common[i])
        summary.append({
            "duration_s": t, "blend_steps": duration_steps(t), "states": len(ids),
            "restore_valid": sum(e["restore_success"] for e in episodes),
            "common_valid_states": sum(common), "common_state_carry_successes": successes,
            "common_state_success_rate": successes / sum(common) if any(common) else None,
            "total_carry_successes": sum(e["carry"]["success"] for e in episodes),
        })
    return summary, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/teacher/bc/bc2_grasp_lift_43d.pt")
    parser.add_argument("--carry_checkpoint", type=Path, default=ROOT / "data/teacher/carry_expanded_r01/bc3_carry_candidate.pt")
    parser.add_argument("--durations", type=float, nargs="+", default=[0.26, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--carry_seconds", type=float, default=16.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs/robustness/duration_sweep_r01")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.states <= 0 or not math.isfinite(args.carry_seconds) or args.carry_seconds <= 0:
        parser.error("states and carry_seconds must be positive")
    try:
        for t in args.durations:
            duration_steps(t)
    except (ValueError, OverflowError) as exc:
        parser.error(str(exc))
    if len(set(args.durations)) != len(args.durations) or args.carry_seconds <= max(args.durations):
        parser.error("durations must be unique and carry_seconds must exceed every duration")
    args.snapshots = sorted({p.expanduser().resolve() for p in args.snapshots})
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.carry_checkpoint = args.carry_checkpoint.expanduser().resolve()
    for p in [*args.snapshots, args.checkpoint, args.carry_checkpoint]:
        if not p.is_file():
            parser.error(f"input not found: {p}")
    evaluator = ROOT / "scripts/teacher/evaluate_bc2_rollout.py"
    signature = {
        "states": args.states, "seed": args.seed, "durations": args.durations,
        "carry_seconds": args.carry_seconds, "device": args.device,
        "inputs": {str(p): sha256(p) for p in [evaluator, args.checkpoint, args.carry_checkpoint, *args.snapshots]},
    }
    root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.json"
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text()) != signature:
                raise ValueError("existing experiment differs, or --resume is missing; use a new output_root")
        else:
            manifest.write_text(json.dumps(signature, indent=2, sort_keys=True))
    results = {}
    for t in args.durations:
        output = root / f"blend_{str(t).replace('.', 'p')}s.json"
        command = [
            sys.executable, "-u", str(evaluator), "--checkpoint", str(args.checkpoint),
            "--carry_checkpoint", str(args.carry_checkpoint),
            "--transition_snapshot_input", *map(str, args.snapshots),
            "--carry_blend_steps", str(duration_steps(t)), "--episodes", str(args.states),
            "--seed", str(args.seed), "--carry_seconds", str(args.carry_seconds),
            "--device", args.device, "--output", str(output),
        ] + (["--headless"] if args.headless else [])
        print(f"[duration-sweep] {t:g}s: {args.states} paired states", flush=True)
        if args.dry_run:
            import shlex
            print(shlex.join(command))
            continue
        if not output.exists():
            log_path = output.with_suffix(".log")
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith(("[snapshot-eval]", "[chain]", "[progress]")):
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
                raise RuntimeError(f"evaluation failed: exit={returncode}, log={log_path}")
        data = json.loads(output.read_text())
        config = data["evaluator_config"]
        if (len(data["episode_results"]) != args.states
            or config["rollout_mode"] != "snapshot_transition"
            or config["carry_blend_steps"] != duration_steps(t)
            or config["seed"] != args.seed or config["carry_seconds"] != args.carry_seconds
            or data["carry_checkpoint_sha256"] != sha256(args.carry_checkpoint)):
            raise ValueError(f"incomplete or mismatched result: {output}")
        results[t] = data
        summary, states = paired_reports(results)
        save_csv(root / "summary.csv", summary)
        save_csv(root / "states.csv", states)
        print(f"[duration-sweep] {data['snapshot_metrics']}", flush=True)
    if not args.dry_run:
        print(f"[duration-sweep] done: {root / 'summary.csv'} and {root / 'states.csv'}", flush=True)


if __name__ == "__main__":
    main()
