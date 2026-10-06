#!/usr/bin/env python3
"""Balanced, resumable Teacher collection for table-local and walking diversity.

The driver does not import Isaac Sim. Each shard starts the existing physical
collector; a failed cell stops collection instead of silently oversampling easy
cells. Success counts are targets, never synthetic duplicates.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

import h5py
import numpy as np

from scripts.teacher.generate_randomized_box_teacher import audit_success_only_shard
from unitree_rl_lab.utils.teacher.phase_bc import DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA
from unitree_rl_lab.runtime_paths import default_arena_dataset, default_arena_root


def bounds(values):
    if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
        raise ValueError("bounds must be finite MIN <= MAX")
    return tuple(values)


def split_bounds(interval, index, cells=4):
    lo, hi = bounds(interval)
    return (lo + (hi - lo) * index / cells, lo + (hi - lo) * (index + 1) / cells)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_diverse_shard(path, config, cell):
    audit = audit_success_only_shard(path, required_hold_steps=100, expected_box_size_m=0.2,
                                    expected_initial_box_height_m=0.84)
    if not audit.valid:
        raise ValueError(f"strict physical/actor contract failed: {audit}")
    positions = []
    with h5py.File(path, "r") as dataset:
        run = json.loads(dataset.attrs["metadata_json"])
        walking = config["case"] == "table_approach"
        if run.get("randomization_mode") != ("table_and_box_xy" if walking else "box_xy_only"):
            raise ValueError("wrong randomization mode")
        for episode in dataset["data"].values():
            final = json.loads(episode.attrs["final_metadata_json"])
            box = np.asarray(final["initial_box_xy_m"])
            table = np.asarray(final["initial_table_xy_m"])
            local = np.asarray(final["initial_box_table_relative_xy_m"])
            if not np.allclose(box - table, local, atol=1e-6):
                raise ValueError("Box/table coordinate-frame mismatch")
            xy = table if walking else local
            if not np.isfinite(xy).all() or any(not lo-1e-6 <= value <= hi+1e-6 for value, (lo, hi) in zip(xy, cell)):
                raise ValueError("episode outside assigned coverage cell")
            if walking and (not final.get("navigation_completed") or final.get("navigation_steps", 0) < 100):
                raise ValueError("missing physical approach and 2-second settling")
            if walking:
                nav_end = np.asarray(final.get("navigation_end_root_pose_wxyz", []))
                initial_root = np.asarray(final["initial_root_xy_m"])
                if nav_end.shape != (7,) or not np.isfinite(nav_end).all():
                    raise ValueError("missing realized navigation endpoint")
                relative_box = box - nav_end[:2]
                if np.linalg.norm(relative_box - [0.42, 0]) > 0.12 or np.linalg.norm(nav_end[:2] - initial_root) < 0.05:
                    raise ValueError("navigation endpoint does not demonstrate actual approach")
            if any(not lo-1e-6 <= value <= hi+1e-6 for value, (lo, hi) in zip(local, config["box_local_range"])):
                raise ValueError("Box outside supported table region")
            positions.append({"box_xy": box.tolist(), "table_xy": table.tolist(), "box_table_xy": local.tolist()})
    return audit, positions


def collector_command(args, output, cell_id, shard_id, count):
    walking = args.case == "table_approach"
    intervals = (args.table_x_range, args.table_y_range) if walking else (args.box_local_x_range, args.box_local_y_range)
    cell = (split_bounds(intervals[0], cell_id % 4), split_bounds(intervals[1], cell_id // 4))
    local = (args.box_local_x_range, args.box_local_y_range) if walking else cell
    x = local[0] if walking else tuple(v + 0.62 for v in local[0])
    command = [sys.executable, str(Path(__file__).with_name("collect_nominal_box_teacher.py")),
               "--arena_root", args.arena_root, "--arena_dataset", args.arena_dataset,
               "--output", str(output), "--snapshot_output", str(output.with_suffix(".snapshots.hdf5")),
               "--headless", "--device", args.device, "--randomize_box_xy", "--success_only",
               "--target_successes", str(count), "--max_attempts", str(count * args.max_attempt_factor),
               "--minimum_success_rate", str(args.minimum_success_rate),
               "--seed", str(args.seed + shard_id), "--box_x_range", *map(str, x),
               "--box_y_range", *map(str, local[1]), "--hold_seconds", "2"]
    if walking:
        command += ["--randomize_work_area", "--table_x_range", *map(str, cell[0]),
                    "--table_y_range", *map(str, cell[1]),
                    "--navigation_timeout_seconds", str(args.navigation_timeout_seconds)]
    return command, cell


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case", required=True, choices=("table_box", "table_approach"))
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--target_demos", type=int, default=500)
    result.add_argument("--shard_size", type=int, default=25)
    result.add_argument("--max_attempt_factor", type=int, default=4)
    result.add_argument("--minimum_success_rate", type=float, default=0.8)
    result.add_argument("--box_local_x_range", nargs=2, type=float, default=(-0.08, 0.08))
    result.add_argument("--box_local_y_range", nargs=2, type=float, default=(-0.08, 0.08))
    result.add_argument("--table_x_range", nargs=2, type=float, default=(0.8, 1.5))
    result.add_argument("--table_y_range", nargs=2, type=float, default=(-0.4, 0.4))
    result.add_argument("--navigation_timeout_seconds", type=float, default=45)
    result.add_argument("--arena_root", default=default_arena_root())
    result.add_argument("--arena_dataset", default=default_arena_dataset())
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=20261006)
    result.add_argument("--dry-run", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if min(args.target_demos, args.shard_size, args.max_attempt_factor) <= 0:
        raise ValueError("counts must be positive")
    if not 0.8 <= args.minimum_success_rate <= 1 or not math.isfinite(args.navigation_timeout_seconds) or args.navigation_timeout_seconds <= 2:
        raise ValueError("invalid rate/navigation timeout")
    for interval in (args.box_local_x_range, args.box_local_y_range):
        lo, hi = bounds(interval)
        if lo < -0.08-1e-9 or hi > 0.08+1e-9:
            raise ValueError("0.4m tabletop / 0.2m cube require offsets within +/-0.08m with 2cm margin")
    for interval in (args.table_x_range, args.table_y_range):
        lo, hi = bounds(interval)
        if lo < -2.05 or hi > 2.05:
            raise ValueError("table bounds outside existing 5m work area")
    output_dir = args.output_dir.resolve()
    config = {"case": args.case, "target_demos": args.target_demos, "shard_size": args.shard_size,
              "max_attempt_factor": args.max_attempt_factor, "minimum_success_rate": args.minimum_success_rate,
              "box_local_range": [list(args.box_local_x_range), list(args.box_local_y_range)],
              "table_range": [list(args.table_x_range), list(args.table_y_range)], "seed": args.seed,
              "arena_root": str(Path(args.arena_root).resolve()), "arena_dataset": str(Path(args.arena_dataset).resolve()),
              "navigation_timeout_seconds": args.navigation_timeout_seconds,
              "strict_hold_steps": 100, "collector_sha256": sha256(Path(__file__).with_name("collect_nominal_box_teacher.py")),
              "source_sha256": sha256(args.arena_dataset) if not args.dry_run else "dry_run",
              "implementation_sha256": {name: sha256(Path(__file__).resolve().parents[2] / name) for name in (
                  "scripts/teacher/collect_diverse_box_teacher.py", "scripts/teacher/arena_wbc_worker.py",
                  "source/unitree_rl_lab/unitree_rl_lab/utils/arena_wbc_bridge.py",
                  "source/unitree_rl_lab/unitree_rl_lab/utils/teacher/arena_teacher_runtime.py",
                  "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events.py",
                  "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/camera_navigation_command.py")}}
    # Equal success quota per spatial cell; never reuse easier predecessor states.
    quotas = [args.target_demos // 16 + int(i < args.target_demos % 16) for i in range(16)]
    plans = []
    for cell_id, quota in enumerate(quotas):
        remaining = quota
        while remaining:
            count = min(args.shard_size, remaining)
            plans.append((cell_id, count))
            remaining -= count
    if args.dry_run:
        for shard_id, (cell_id, count) in enumerate(plans):
            command, cell = collector_command(args, output_dir / f"shard_{shard_id:05d}.hdf5", cell_id, shard_id, count)
            print(json.dumps({"cell": cell_id, "bounds": cell, "target": count, "command": command}))
        return
    if not Path(args.arena_dataset).is_file() or not (Path(args.arena_root) / ".venv/bin/python").is_file():
        raise FileNotFoundError("Arena source HDF5 or worker Python is missing; set LM_ARENA_ROOT/LM_ARENA_DATASET")
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_lock = (output_dir / ".collection.lock").open("a")
    fcntl.flock(collection_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = output_dir / "manifest.json"
    manifest = {"schema": DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA, "config": config, "complete": False, "accepted_demonstrations": 0,
                "target_successes_per_cell": quotas, "shards": [], "coverage": [], "attempt_logs": [], "status": "collecting"}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("config") != config:
            raise ValueError("resume configuration/code differs; use a new output directory")
        previous_hashes = {item["path"]: item["sha256"] for item in old["shards"]}
        previous_snapshot_hashes = {item["path"]: item.get("snapshot_sha256") for item in old["shards"]}
    else:
        if any(output_dir.glob("shard_*")):
            raise ValueError("unmanifested artifacts present; use a clean output directory")
        previous_hashes = {}
        previous_snapshot_hashes = {}

    def publish():
        accepted = manifest["accepted_demonstrations"]
        attempts = sum(item["attempts"] for item in manifest["attempt_logs"])
        rate = accepted / max(1, attempts)
        counts = [sum(item["episodes"] for item in manifest["shards"] if item["cell"] == cell) for cell in range(16)]
        manifest.update(accepted_demos=accepted, target_demos=args.target_demos, attempts=attempts,
                        aggregate_teacher_success_rate=rate, accepted_successes_per_cell=counts,
                        gates={"minimum_success_rate": args.minimum_success_rate,
                               "demo_count_reached": accepted == args.target_demos,
                               "success_rate_at_least_minimum": rate >= args.minimum_success_rate,
                               "all_demos_strict_bilateral_lift_hold": bool(manifest["shards"]),
                               "balanced_spatial_coverage_reached": counts == quotas})
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    process = None
    try:
        for shard_id, (cell_id, count) in enumerate(plans):
            output = output_dir / f"shard_{shard_id:05d}.hdf5"
            command, cell = collector_command(args, output, cell_id, shard_id, count)
            exit_code = 0
            manifest["active_cell"] = cell_id
            manifest["active_shard"] = shard_id
            publish()
            if output.exists():
                if str(output) in previous_hashes and sha256(output) != previous_hashes[str(output)]:
                    raise ValueError("resumed shard SHA mismatch")
            else:
                # Never overwrite unpublished/interrupted HDF5 transactions.
                if any(output_dir.glob(f".{output.name}.*.partial")):
                    raise ValueError("interrupted shard transaction exists; use a fresh run directory")
                print(f"[{args.case}] cell={cell_id} shard={shard_id} target={count}", flush=True)
                process = subprocess.Popen(command, start_new_session=True)
                exit_code = process.wait()
                process = None
            metric_path = output.with_suffix(".metrics.json")
            if metric_path.is_file():
                metrics = json.loads(metric_path.read_text())
                manifest["attempt_logs"].append({"cell": cell_id, "shard": shard_id, "path": str(metric_path),
                                                 "attempts": metrics["attempts"], "accepted": metrics["accepted_demonstrations"],
                                                 "trial_results": metrics["trial_results"]})
                publish()
            if output.exists():
                audit, positions = audit_diverse_shard(output, config, cell)
                if audit.episodes > count:
                    raise ValueError("shard exceeds balanced cell quota")
                metrics = json.loads(metric_path.read_text())
                snapshot_path = output.with_suffix(".snapshots.hdf5")
                with h5py.File(snapshot_path, "r") as snapshots:
                    if int(snapshots.attrs.get("num_snapshots", -1)) != audit.episodes:
                        raise ValueError("live snapshot count does not match accepted demonstrations")
                if previous_snapshot_hashes.get(str(output)) is not None and sha256(snapshot_path) != previous_snapshot_hashes[str(output)]:
                    raise ValueError("resumed live snapshot SHA mismatch")
                manifest["shards"].append({"path": str(output), "sha256": sha256(output), "cell": cell_id,
                                           "episodes": audit.episodes, "accepted": audit.episodes, "steps": audit.total_steps,
                                           "snapshot_path": str(snapshot_path), "snapshot_sha256": sha256(snapshot_path),
                                           "attempts": metrics["attempts"], "attempt_success_rate": metrics["attempt_success_rate"]})
                manifest["accepted_demonstrations"] += audit.episodes
                manifest["coverage"].extend(positions)
                publish()
                if audit.episodes != count or metrics["attempt_success_rate"] < args.minimum_success_rate:
                    raise RuntimeError("cell target/Teacher reliability unmet; partial successes retained, no easy-cell substitution")
            if exit_code or not output.exists():
                raise RuntimeError(f"collector failed (exit={exit_code}); inspect log/metrics before another production run")
        manifest["complete"] = manifest["accepted_demonstrations"] == args.target_demos
        manifest["status"] = "complete"
        manifest["active_cell"] = None
        manifest["active_shard"] = None
        publish()
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        publish()
        raise


if __name__ == "__main__":
    main()
