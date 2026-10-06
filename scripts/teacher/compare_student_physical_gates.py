#!/usr/bin/env python3
"""Compare two phase Student physical gates and emit an auditable promotion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gate(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "unitree_phase_student_physical_gate_v1":
        raise ValueError(f"unsupported gate schema: {resolved}")
    checkpoint = Path(payload["student_checkpoint"]).expanduser().resolve()
    if _sha256(checkpoint) != payload.get("student_checkpoint_sha256"):
        raise ValueError(f"checkpoint SHA mismatch: {checkpoint}")
    return payload


def _source_episodes(gate: dict) -> list[str]:
    episodes: list[str] = []
    for source in gate["source_rollout_metrics"]:
        path = Path(source["path"]).expanduser().resolve()
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"rollout sidecar SHA mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes.extend(result["source_episode"] for result in payload["results"])
    if len(episodes) != len(set(episodes)):
        raise ValueError("a gate reuses one or more source episodes")
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    baseline = _load_gate(args.baseline)
    candidate = _load_gate(args.candidate)
    for key in ("phase", "trials", "minimum_trials", "minimum_success_rate"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"baseline and candidate differ in {key}")

    baseline_episodes = _source_episodes(baseline)
    candidate_episodes = _source_episodes(candidate)
    if baseline_episodes != candidate_episodes:
        raise ValueError("baseline and candidate were not evaluated on the same ordered episodes")

    baseline_rate = float(baseline["physical_success_rate"])
    candidate_rate = float(candidate["physical_success_rate"])
    promoted = bool(candidate["gate_passed"] and candidate_rate > baseline_rate)
    report = {
        "schema": "unitree_phase_student_checkpoint_promotion_v1",
        "status": "promoted" if promoted else "rejected",
        "phase": baseline["phase"],
        "evaluation_episode_count": len(baseline_episodes),
        "evaluation_source_episodes": baseline_episodes,
        "minimum_success_rate": candidate["minimum_success_rate"],
        "baseline": {
            "gate": str(args.baseline.expanduser().resolve()),
            "checkpoint": baseline["student_checkpoint"],
            "checkpoint_sha256": baseline["student_checkpoint_sha256"],
            "successes": baseline["successes"],
            "trials": baseline["trials"],
            "physical_success_rate": baseline_rate,
            "gate_passed": baseline["gate_passed"],
        },
        "candidate": {
            "gate": str(args.candidate.expanduser().resolve()),
            "checkpoint": candidate["student_checkpoint"],
            "checkpoint_sha256": candidate["student_checkpoint_sha256"],
            "successes": candidate["successes"],
            "trials": candidate["trials"],
            "physical_success_rate": candidate_rate,
            "gate_passed": candidate["gate_passed"],
        },
        "success_rate_delta": candidate_rate - baseline_rate,
        "candidate_promoted": promoted,
        "selected_checkpoint": (
            candidate["student_checkpoint"] if promoted else baseline["student_checkpoint"]
        ),
        "decision_rule": "candidate gate must pass and physical success rate must exceed baseline",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
