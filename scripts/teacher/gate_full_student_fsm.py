#!/usr/bin/env python3
"""Audit a full Student FSM rollout and emit a production pass/fail gate."""

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
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument("--minimum-full-task-success-rate", type=float, default=0.6)
    parser.add_argument("--maximum-bad-orientation-rate", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.minimum_episodes <= 0:
        parser.error("--minimum-episodes must be positive")
    for name, value in (
        ("--minimum-full-task-success-rate", args.minimum_full_task_success_rate),
        ("--maximum-bad-orientation-rate", args.maximum_bad_orientation_rate),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be in [0, 1]")

    source = args.metrics.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    episode_results = payload.get("episode_results")
    episodes = payload.get("episodes")
    if not isinstance(episode_results, list) or len(episode_results) != episodes:
        raise ValueError("FSM episode count does not match episode_results")
    expected_ids = list(range(episodes))
    if [item.get("episode") for item in episode_results] != expected_ids:
        raise ValueError("FSM episode IDs must be sequential")

    transitions = [set(item.get("transitions", [])) for item in episode_results]
    recomputed = {
        "approach_successes": sum("approach" in item for item in transitions),
        "grasp_successes": sum("grasp_lift" in item for item in transitions),
        "carry_successes": sum("carry" in item for item in transitions),
        "place_successes": sum("place" in item for item in transitions),
    }
    for key, value in recomputed.items():
        if payload.get(key) != value:
            raise ValueError(f"FSM aggregate mismatch for {key}")

    full_successes = sum(
        item.get("reason") == "full_task_success"
        and item.get("final_phase") == "complete"
        and not item.get("failed")
        for item in episode_results
    )
    if full_successes != recomputed["place_successes"]:
        raise ValueError("PLACE transition count does not match full-task successes")
    bad_orientation_count = sum(
        item.get("reason") == "bad_orientation" for item in episode_results
    )
    full_rate = full_successes / episodes if episodes else 0.0
    bad_rate = bad_orientation_count / episodes if episodes else 0.0
    if abs(float(payload.get("full_task_success", -1.0)) - full_rate) > 1.0e-12:
        raise ValueError("full_task_success metric is inconsistent")
    if abs(float(payload.get("bad_orientation_rate", -1.0)) - bad_rate) > 1.0e-12:
        raise ValueError("bad_orientation_rate metric is inconsistent")

    checkpoint_bindings = {}
    for phase, path_value in payload.get("checkpoints", {}).items():
        checkpoint = Path(path_value).expanduser().resolve()
        checkpoint_bindings[phase] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }
    if set(checkpoint_bindings) != {"approach", "grasp_lift", "carry", "place"}:
        raise ValueError("FSM metrics must bind exactly four phase checkpoints")

    gate_passed = (
        episodes >= args.minimum_episodes
        and full_rate >= args.minimum_full_task_success_rate
        and bad_rate <= args.maximum_bad_orientation_rate
    )
    report = {
        "schema": "unitree_full_student_fsm_physical_gate_v1",
        "status": "passed" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "source_metrics": str(source),
        "source_metrics_sha256": _sha256(source),
        "episodes": episodes,
        "minimum_episodes": args.minimum_episodes,
        "full_task_successes": full_successes,
        "full_task_success_rate": full_rate,
        "minimum_full_task_success_rate": args.minimum_full_task_success_rate,
        "bad_orientation_count": bad_orientation_count,
        "bad_orientation_rate": bad_rate,
        "maximum_bad_orientation_rate": args.maximum_bad_orientation_rate,
        **recomputed,
        "termination_reason_counts": dict(
            sorted(Counter(item["reason"] for item in episode_results).items())
        ),
        "handoff_gate": payload.get("handoff_gate"),
        "loaded_stand_gate": payload.get("loaded_stand_gate"),
        "walk_gate": payload.get("walk_gate"),
        "checkpoint_bindings": checkpoint_bindings,
    }
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
