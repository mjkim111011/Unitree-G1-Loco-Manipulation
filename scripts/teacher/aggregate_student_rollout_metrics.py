#!/usr/bin/env python3
"""Aggregate disjoint physical Student rollout sidecars into one auditable gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-trials", type=int, default=20)
    parser.add_argument("--minimum-success-rate", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.minimum_trials <= 0:
        parser.error("--minimum-trials must be positive")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("--minimum-success-rate must be in [0, 1]")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")

    loaded: list[tuple[Path, dict]] = []
    for source in args.metrics:
        resolved = source.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"rollout is not completed: {resolved}")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != payload.get("trials"):
            raise ValueError(f"rollout result count mismatch: {resolved}")
        loaded.append((resolved, payload))

    first = loaded[0][1]
    invariant_keys = (
        "phase",
        "evaluation_mode",
        "student_checkpoint",
        "student_checkpoint_sha256",
    )
    for source, payload in loaded[1:]:
        for key in invariant_keys:
            if payload.get(key) != first.get(key):
                raise ValueError(f"{key} differs in {source}")

    trials = sum(int(payload["trials"]) for _, payload in loaded)
    successes = sum(int(payload["successes"]) for _, payload in loaded)
    reasons = Counter(
        result["reason"] for _, payload in loaded for result in payload["results"]
    )
    success_rate = successes / trials if trials else 0.0
    gate_passed = (
        trials >= args.minimum_trials and success_rate >= args.minimum_success_rate
    )
    report = {
        "schema": "unitree_phase_student_physical_gate_v1",
        "status": "passed" if gate_passed else "failed",
        "phase": first["phase"],
        "evaluation_mode": first["evaluation_mode"],
        "student_checkpoint": first["student_checkpoint"],
        "student_checkpoint_sha256": first["student_checkpoint_sha256"],
        "trials": trials,
        "successes": successes,
        "physical_success_rate": success_rate,
        "minimum_trials": args.minimum_trials,
        "minimum_success_rate": args.minimum_success_rate,
        "gate_passed": gate_passed,
        "termination_reason_counts": dict(sorted(reasons.items())),
        "source_rollout_metrics": [
            {
                "path": str(source),
                "sha256": _sha256(source),
                "trials": payload["trials"],
                "successes": payload["successes"],
                "requested_output": payload.get("requested_output"),
            }
            for source, payload in loaded
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
