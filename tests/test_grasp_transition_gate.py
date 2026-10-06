from __future__ import annotations

import json

import pytest

from scripts.teacher.gate_grasp_transition_fsm import build_report


def _payload(tmp_path, *, grasp_successes: int) -> tuple[dict, object]:
    checkpoints = {}
    for phase in ("approach", "grasp_lift", "carry", "place"):
        checkpoint = tmp_path / f"{phase}.pt"
        checkpoint.write_bytes(phase.encode())
        checkpoints[phase] = str(checkpoint)
    results = []
    for episode in range(5):
        transitions = ["approach"]
        if episode < grasp_successes:
            transitions.append("grasp_lift")
        results.append(
            {
                "episode": episode,
                "transitions": transitions,
                "reason": "handoff_gate_passed"
                if episode < grasp_successes
                else "grasp_phase_timeout",
            }
        )
    payload = {
        "episodes": 5,
        "episode_results": results,
        "approach_successes": 5,
        "grasp_successes": grasp_successes,
        "P(approach)": 1.0,
        "P(grasp|approach)": grasp_successes / 5,
        "bad_orientation_rate": 0.0,
        "grasp_checkpoint_certified_by_gate": True,
        "checkpoints": checkpoints,
    }
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return payload, source


def test_grasp_transition_gate_passes_only_complete_conditional_evidence(tmp_path):
    payload, source = _payload(tmp_path, grasp_successes=4)
    report = build_report(
        payload,
        source=source,
        minimum_episodes=5,
        minimum_approach_rate=0.8,
        minimum_grasp_given_approach_rate=0.8,
        maximum_bad_orientation_rate=0.05,
    )
    assert report["gate_passed"]
    assert report["next_stage"] == "workspace_5m_full_fsm"
    assert report["grasp_given_approach_success_rate"] == pytest.approx(0.8)


def test_grasp_transition_gate_fails_closed_on_screening_run(tmp_path):
    payload, source = _payload(tmp_path, grasp_successes=0)
    report = build_report(
        payload,
        source=source,
        minimum_episodes=20,
        minimum_approach_rate=0.8,
        minimum_grasp_given_approach_rate=0.8,
        maximum_bad_orientation_rate=0.05,
    )
    assert not report["gate_passed"]
    assert not report["checks"]["minimum_episodes"]
    assert not report["checks"]["grasp_given_approach_success_rate"]
    assert report["next_stage"] == "retain_deployed_policy"
