#!/usr/bin/env python3
"""Audit an Approach-to-Grasp/Lift Student FSM rollout."""

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


def build_report(
    payload: dict,
    *,
    source: Path,
    minimum_episodes: int,
    minimum_approach_rate: float,
    minimum_grasp_given_approach_rate: float,
    maximum_bad_orientation_rate: float,
) -> dict:
    episode_results = payload.get("episode_results")
    episodes = payload.get("episodes")
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("FSM metrics must contain a positive integer episode count")
    if not isinstance(episode_results, list) or len(episode_results) != episodes:
        raise ValueError("FSM episode count does not match episode_results")
    if [item.get("episode") for item in episode_results] != list(range(episodes)):
        raise ValueError("FSM episode IDs must be sequential")

    transitions = [set(item.get("transitions", [])) for item in episode_results]
    approach_successes = sum("approach" in item for item in transitions)
    grasp_successes = sum("grasp_lift" in item for item in transitions)
    if grasp_successes > approach_successes:
        raise ValueError("grasp successes cannot exceed approach successes")
    if payload.get("approach_successes") != approach_successes:
        raise ValueError("FSM approach aggregate is inconsistent")
    if payload.get("grasp_successes") != grasp_successes:
        raise ValueError("FSM grasp aggregate is inconsistent")

    bad_orientation_count = sum(
        item.get("reason") == "bad_orientation" for item in episode_results
    )
    approach_rate = approach_successes / episodes
    grasp_given_approach_rate = (
        grasp_successes / approach_successes if approach_successes else 0.0
    )
    bad_orientation_rate = bad_orientation_count / episodes
    expected_rates = {
        "P(approach)": approach_rate,
        "P(grasp|approach)": grasp_given_approach_rate,
        "bad_orientation_rate": bad_orientation_rate,
    }
    for key, expected in expected_rates.items():
        measured = payload.get(key)
        if measured is None or abs(float(measured) - expected) > 1.0e-12:
            raise ValueError(f"FSM metric is inconsistent: {key}")

    checkpoint_bindings = {}
    for phase, path_value in payload.get("checkpoints", {}).items():
        checkpoint = Path(path_value).expanduser().resolve()
        checkpoint_bindings[phase] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }
    if set(checkpoint_bindings) != {"approach", "grasp_lift", "carry", "place"}:
        raise ValueError("FSM metrics must bind exactly four phase checkpoints")

    checks = {
        "minimum_episodes": episodes >= minimum_episodes,
        "approach_success_rate": approach_rate >= minimum_approach_rate,
        "grasp_given_approach_success_rate": (
            grasp_given_approach_rate >= minimum_grasp_given_approach_rate
        ),
        "bad_orientation_rate": (
            bad_orientation_rate <= maximum_bad_orientation_rate
        ),
        "grasp_checkpoint_certified": bool(
            payload.get("grasp_checkpoint_certified_by_gate")
        ),
    }
    gate_passed = all(checks.values())
    return {
        "schema": "unitree_approach_grasp_transition_physical_gate_v1",
        "status": "passed" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "checks": checks,
        "source_metrics": str(source),
        "source_metrics_sha256": _sha256(source),
        "episodes": episodes,
        "minimum_episodes": minimum_episodes,
        "approach_successes": approach_successes,
        "approach_success_rate": approach_rate,
        "minimum_approach_success_rate": minimum_approach_rate,
        "grasp_successes": grasp_successes,
        "grasp_given_approach_success_rate": grasp_given_approach_rate,
        "minimum_grasp_given_approach_success_rate": (
            minimum_grasp_given_approach_rate
        ),
        "bad_orientation_count": bad_orientation_count,
        "bad_orientation_rate": bad_orientation_rate,
        "maximum_bad_orientation_rate": maximum_bad_orientation_rate,
        "termination_reason_counts": dict(
            sorted(Counter(item["reason"] for item in episode_results).items())
        ),
        "checkpoint_bindings": checkpoint_bindings,
        "promotion_allowed": gate_passed,
        "next_stage": "workspace_5m_full_fsm" if gate_passed else "retain_deployed_policy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument("--minimum-approach-success-rate", type=float, default=0.8)
    parser.add_argument(
        "--minimum-grasp-given-approach-success-rate", type=float, default=0.8
    )
    parser.add_argument("--maximum-bad-orientation-rate", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.minimum_episodes <= 0:
        parser.error("--minimum-episodes must be positive")
    for name, value in (
        ("--minimum-approach-success-rate", args.minimum_approach_success_rate),
        (
            "--minimum-grasp-given-approach-success-rate",
            args.minimum_grasp_given_approach_success_rate,
        ),
        ("--maximum-bad-orientation-rate", args.maximum_bad_orientation_rate),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be in [0, 1]")

    source = args.metrics.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = build_report(
        payload,
        source=source,
        minimum_episodes=args.minimum_episodes,
        minimum_approach_rate=args.minimum_approach_success_rate,
        minimum_grasp_given_approach_rate=(
            args.minimum_grasp_given_approach_success_rate
        ),
        maximum_bad_orientation_rate=args.maximum_bad_orientation_rate,
    )
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
