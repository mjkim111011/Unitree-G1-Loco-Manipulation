#!/usr/bin/env python3
"""Collect fresh Carry demonstrations, train a candidate, and compare live rollouts.

Each shard gets new BC2 physical states. Snapshots are never cycled. Existing
checkpoints are retained; only an explicitly chosen run directory is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-successes", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_successes < 20 or args.shard_size < 20:
        parser.error("target-successes and shard-size must each be at least 20")
    run = args.run_dir.expanduser().resolve()
    bc2 = ROOT / "data/teacher/bc/bc2_grasp_lift_43d.pt"
    baseline = ROOT / "data/teacher/bc/bc3_carry_43d.pt"
    candidate = run / "bc3_carry_candidate.pt"
    candidate_metrics = run / "bc3_carry_candidate.train_metrics.json"
    if not args.dry_run:
        run.mkdir(parents=True, exist_ok=True)

    def stage(name, script, arguments, outputs, inputs=()):
        command = [sys.executable, str(ROOT / "scripts/teacher" / script), *map(str, arguments)]
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            return
        marker = run / f"{name}.done.json"
        input_hashes = {str(p): sha256(p) for p in [ROOT / "scripts/teacher" / script, *inputs]}
        if marker.exists():
            previous = json.loads(marker.read_text())
            if (previous["command"] == command and previous["inputs"] == input_hashes
                    and all(Path(p).is_file() and sha256(p) == digest for p, digest in previous["outputs"].items())):
                print(f"[{name}] verified completed outputs; skipping", flush=True)
                return
            raise RuntimeError(f"Changed inputs or outputs for {name}; use a new --run-dir")
        # An interrupted stage may leave partial files; existing verified stages
        # are skipped above. The underlying snapshot writers support overwrite.
        environment = os.environ.copy()
        environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        environment.setdefault("ACCEPT_EULA", "Y")
        with (run / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps({"command": command, "inputs": input_hashes,
                                     "outputs": {str(p): sha256(p) for p in outputs}}, indent=2) + "\n")

    def states(name, count, seed):
        metrics = run / f"{name}.bc2.json"
        snapshots = run / f"{name}.snapshots.hdf5"
        gate = run / f"{name}.gate.json"
        stage(f"{name}_states", "evaluate_bc2_rollout.py", [
            "--checkpoint", bc2, "--episodes", count, "--episode_seconds", 20,
            "--seed", seed,
            "--snapshot_output", snapshots, "--output", metrics,
            "--device", args.device, "--headless", "--overwrite",
        ], [metrics, snapshots], [bc2])
        stage(f"{name}_certify", "phase_bc.py", [
            "certify-bc2", "--checkpoint", bc2, "--rollout-metrics", metrics,
            "--output", gate,
        ], [gate], [bc2, metrics, snapshots])
        return snapshots, gate

    datasets = []
    shards = math.ceil(args.target_successes / args.shard_size)
    # Balance shard quotas, keeping each certification above the 20-episode minimum.
    quotient, remainder = divmod(args.target_successes, shards)
    for index in range(shards):
        quota = quotient + int(index < remainder)
        name = f"shard_{index:03d}"
        snapshots, gate = states(name, max(20, 2 * quota), 17000 + index)
        dataset = run / f"{name}.hdf5"
        metrics = dataset.with_suffix(".metrics.json")
        staging = run / f"{name}.staging.hdf5"
        stage(f"{name}_carry", "collect_carry_place_teacher.py", [
            "--phase", "carry", "--snapshot_input", snapshots, "--bc2_gate", gate,
            "--output", dataset, "--staging_snapshot_output", staging,
            "--num_trials", max(20, 2 * quota), "--target_successes", quota,
            "--device", args.device, "--headless", "--overwrite",
        ], [dataset, metrics, staging], [snapshots, gate, bc2])
        if not args.dry_run:
            report = json.loads(metrics.read_text())
            if not report.get("dataset_published") or report["successes"] < quota:
                raise RuntimeError(f"{name}: only {report['successes']}/{quota} successes; inspect {metrics}")
        datasets.append(dataset)

    stage("train", "phase_bc.py", [
        "train", *datasets, "--skill", "carry", "--action-key", "processed_43d_joint_target",
        "--bc2-gate", gate, "--output", candidate, "--epochs", 200, "--patience", 30,
        "--metrics-json", candidate_metrics,
        "--phase-start-steps", 350, "--phase-start-weight", 4, "--device", args.device,
    ], [candidate, candidate_metrics],
        [*datasets, *[p.with_suffix(".metrics.json") for p in datasets], gate])

    snapshots, gate = states("holdout", 25, 27000)
    reports = {}
    for name, checkpoint in (("baseline", baseline), ("candidate", candidate)):
        output = run / f"{name}_holdout.hdf5"
        report = output.with_suffix(".metrics.json")
        stage(f"evaluate_{name}", "collect_carry_place_teacher.py", [
            "--phase", "carry", "--snapshot_input", snapshots, "--bc2_gate", gate,
            "--student_checkpoint", checkpoint, "--output", output,
            "--num_trials", 20, "--device", args.device, "--headless",
        ], [report], [checkpoint, snapshots, gate])
        if not args.dry_run:
            reports[name] = json.loads(report.read_text())
    if not args.dry_run:
        summary = {"target_successes": args.target_successes,
                   "accepted_successes": sum(json.loads(p.with_suffix(".metrics.json").read_text())["successes"] for p in datasets),
                   "candidate": str(candidate), "candidate_sha256": sha256(candidate),
                   "baseline_sha256": sha256(baseline),
                   "holdout": {name: {key: report[key] for key in ("successes", "trials", "conditional_success_rate", "physical_gates")}
                               for name, report in reports.items()},
                   "default_checkpoint_replaced": False}
        (run / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
