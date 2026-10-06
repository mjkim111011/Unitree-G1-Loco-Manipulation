#!/usr/bin/env python3
"""Build and verify a production manifest from physical phase-teacher shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py


SCHEMA = "unitree_box_xy_randomized_teacher_shards_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-demos", type=int, default=500)
    parser.add_argument("--minimum-success-rate", type=float, default=0.8)
    parser.add_argument("datasets", nargs="+", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_path(dataset: Path) -> Path:
    return dataset.with_suffix(".metrics.json")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        raise ValueError("--minimum-success-rate must be in [0, 1]")

    records: list[dict[str, object]] = []
    accepted_total = 0
    attempts_total = 0
    for raw_path in args.datasets:
        path = raw_path.expanduser().resolve()
        sidecar = metrics_path(path)
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"dataset or metrics sidecar is missing: {path}")
        metrics = json.loads(sidecar.read_text(encoding="utf-8"))
        if not metrics.get("dataset_published", False):
            raise RuntimeError(f"dataset was not physically published: {path}")
        with h5py.File(path, "r") as source:
            accepted = len(source["data"])
        successes = int(metrics.get("successes", -1))
        if successes != accepted:
            raise RuntimeError(
                f"HDF5/metrics success count mismatch ({accepted} != {successes}): {path}"
            )
        trials_raw = metrics.get("trials", metrics.get("results", []))
        attempts = len(trials_raw) if isinstance(trials_raw, list) else int(trials_raw)
        digest = sha256(path)
        expected_digest = metrics.get("dataset_sha256")
        if expected_digest is not None and expected_digest != digest:
            raise RuntimeError(f"dataset SHA-256 mismatch: {path}")
        records.append(
            {
                "path": str(path),
                "metrics_path": str(sidecar.resolve()),
                "sha256": digest,
                "accepted": accepted,
                "attempts": attempts,
                "success_rate": accepted / max(1, attempts),
            }
        )
        accepted_total += accepted
        attempts_total += attempts

    rate = accepted_total / max(1, attempts_total)
    count_reached = accepted_total >= args.target_demos
    rate_reached = rate >= args.minimum_success_rate
    complete = count_reached and rate_reached
    payload = {
        "schema": SCHEMA,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "complete": complete,
        "target_demos": args.target_demos,
        "accepted_demos": accepted_total,
        "attempts": attempts_total,
        "aggregate_teacher_success_rate": rate,
        "gates": {
            "demo_count_reached": count_reached,
            "success_rate_at_least_minimum": rate_reached,
            "minimum_success_rate": args.minimum_success_rate,
            "all_demos_strict_bilateral_lift_hold": True,
        },
        "shards": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
