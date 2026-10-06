#!/usr/bin/env python3
"""Audit the promoted G1 43-DOF loco-manipulation artifacts without Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {
    "teacher_manifest": "data/teacher/box_xy_randomized_500/manifest.json",
    "approach_checkpoint": "data/teacher/bc/box_approach_pregrasp_23d_dagger_open_r05.pt",
    "approach_metrics": "data/teacher/bc/box_approach_pregrasp_23d_dagger_open_r05.pt.metrics.json",
    "grasp_checkpoint": "data/teacher/bc/box_grasp_lift_43d_from_approach_r05_dagger_r02_probe.pt",
    "grasp_gate": "data/teacher/bc/box_grasp_lift_43d_from_approach_r05_dagger_r02.gate.json",
    "grasp_handoff_gate": "data/teacher/bc/fsm_grasp_handoff_margin5cm.seed271828.rollout20.gate.json",
    "grasp_handoff_metrics": "data/teacher/bc/fsm_grasp_handoff_margin5cm.seed271828.rollout20.metrics.json",
    "carry_manifest": "data/teacher/production/carry_teacher_success_500_r04/manifest.json",
    "carry_checkpoint": "data/teacher/bc/box_carry_23d_teacher500_r05.pt",
    "carry_checkpoint_metrics": "data/teacher/bc/box_carry_23d_teacher500_r05.pt.metrics.json",
    "carry_student_physical_metrics": "data/teacher/evaluation/carry_student_teacher500_r05.rollout5.discarded.metrics.json",
    "place_manifest": "data/teacher/production/place_teacher_success_diverse_600_r07/manifest.json",
    "place_checkpoint": "data/teacher/bc/box_place_23d_diverse600_weighted_history_r12.pt",
    "place_checkpoint_metrics": "data/teacher/bc/box_place_23d_diverse600_weighted_history_r12.pt.metrics.json",
    "place_student_physical_gate": "data/teacher/evaluation/place_student_diverse600_r12.physical_gate20.json",
    "chained_grasp_metrics": "data/teacher/bc/fsm_approach_r05_grasp_r02_grip_hold_center_probe.metrics.json",
    "full_fsm_metrics": "data/teacher/evaluation/full_fsm_latest_students_qr_place_r06.seed314159.rollout5.metrics.json",
    "residual_approach_checkpoint": "data/teacher/bc/box_approach_pregrasp_43d_residual_nominal_r01.pt",
    "residual_approach_metrics": "data/teacher/bc/box_approach_pregrasp_43d_residual_nominal_r01.pt.metrics.json",
    "residual_adapter": "source/unitree_rl_lab/unitree_rl_lab/utils/bc_residual.py",
    "residual_log_root": "logs/rsl_rl/g1_43dof_box_pick_place_bc_residual_rnd",
    "residual_training_metrics": "data/teacher/bc/residual_ppo_rnd_100iter_r01.metrics.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_contract_ok(
    metrics: dict[str, Any], *, phase: str, action_dim: int
) -> bool:
    return bool(
        metrics.get("skill_phase") == phase
        and metrics.get("action_dim") == action_dim
        and metrics.get("student_observation_dim") == 517
        and metrics.get("privileged_student_inputs") == []
    )


def audit(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    paths = {name: repo_root / relative for name, relative in ARTIFACTS.items()}
    manifest = _load_json(paths["teacher_manifest"])
    approach = _load_json(paths["approach_metrics"])
    gate = _load_json(paths["grasp_gate"])
    grasp_handoff_gate = _load_json(paths["grasp_handoff_gate"])
    grasp_handoff = _load_json(paths["grasp_handoff_metrics"])
    carry_data = _load_json(paths["carry_manifest"])
    carry_policy = _load_json(paths["carry_checkpoint_metrics"])
    carry_student_physical = _load_json(paths["carry_student_physical_metrics"])
    place_data = _load_json(paths["place_manifest"])
    place_policy = _load_json(paths["place_checkpoint_metrics"])
    place_student_physical = _load_json(paths["place_student_physical_gate"])
    chained = _load_json(paths["chained_grasp_metrics"])
    fsm = _load_json(paths["full_fsm_metrics"])
    residual_approach = _load_json(paths["residual_approach_metrics"])
    residual_training = _load_json(paths["residual_training_metrics"])
    residual_runs = sorted(
        path for path in paths["residual_log_root"].glob("*") if path.is_dir()
    )
    residual_run = residual_runs[-1] if residual_runs else None
    residual_models = (
        sorted(
            residual_run.glob("model_*.pt"),
            key=lambda path: int(path.stem.removeprefix("model_")),
        )
        if residual_run is not None
        else []
    )
    residual_provenance = (
        _load_json(residual_run / "params" / "bc_residual.json")
        if residual_run is not None
        else {}
    )
    carry_corpus_paths = {
        str(Path(record["path"]).expanduser().resolve())
        for record in carry_data.get("shards", [])
        if isinstance(record, dict) and record.get("path")
    }
    carry_policy_paths = {
        str(Path(path).expanduser().resolve())
        for path in carry_policy.get("source_paths", [])
    }
    carry_policy_trained_from_complete_corpus = bool(
        carry_corpus_paths and carry_corpus_paths.issubset(carry_policy_paths)
    )
    place_corpus_paths = {
        str(Path(record["path"]).expanduser().resolve())
        for record in place_data.get("shards", [])
        if isinstance(record, dict) and record.get("path")
    }
    place_policy_paths = {
        str(Path(path).expanduser().resolve())
        for path in place_policy.get("source_paths", [])
    }
    place_policy_trained_from_complete_corpus = bool(
        place_corpus_paths and place_corpus_paths.issubset(place_policy_paths)
    )

    grasp_sha_matches = False
    if paths["grasp_checkpoint"].is_file() and gate.get("checkpoint_sha256"):
        grasp_sha_matches = (
            _sha256(paths["grasp_checkpoint"]) == gate["checkpoint_sha256"]
        )
    grasp_handoff_sha_matches = bool(
        paths["grasp_handoff_metrics"].is_file()
        and grasp_handoff_gate.get("rollout_metrics_sha256")
        and _sha256(paths["grasp_handoff_metrics"])
        == grasp_handoff_gate["rollout_metrics_sha256"]
        and grasp_handoff_gate.get("checkpoint_sha256") == gate.get("checkpoint_sha256")
    )

    fsm_criteria = {
        "min_episodes": 20,
        "min_conditional_success_rate": 0.8,
        "min_full_task_success_rate": 0.6,
        "max_bad_orientation_rate": 0.05,
    }
    conditional_metrics = (
        "P(approach)",
        "P(grasp|approach)",
        "P(carry|grasp)",
        "P(place|carry)",
    )
    stages = {
        "teacher_500_strict_demos": {
            "passed": bool(
                manifest.get("complete")
                and manifest.get("accepted_demos", 0) >= 500
                and manifest.get("target_demos", 0) >= 500
            ),
            "accepted_demos": manifest.get("accepted_demos"),
        },
        "box_approach_pregrasp_policy": {
            "passed": paths["approach_checkpoint"].is_file()
            and _policy_contract_ok(approach, phase="approach", action_dim=23),
            "checkpoint": str(paths["approach_checkpoint"]),
        },
        "bimanual_grasp_lift_policy": {
            "passed": bool(gate.get("passed") and grasp_sha_matches),
            "checkpoint_sha_matches_gate": grasp_sha_matches,
            "physical_success_rate": gate.get("measured", {}).get(
                "bimanual_grasp_lift_success_rate"
            ),
        },
        "approach_grasp_handoff_20_episode_gate": {
            "passed": bool(
                grasp_handoff_gate.get("passed")
                and grasp_handoff_sha_matches
                and grasp_handoff.get("episodes", 0) >= 20
                and grasp_handoff.get("P(grasp|approach)", 0.0) >= 0.8
                and grasp_handoff.get("handoff_gate", {}).get(
                    "conditional_probability", 0.0
                )
                >= 0.8
            ),
            "metrics_sha_matches_gate": grasp_handoff_sha_matches,
            "episodes": grasp_handoff.get("episodes", 0),
            "P(grasp|approach)": grasp_handoff.get("P(grasp|approach)"),
            "P(handoff|grasp)": grasp_handoff.get("handoff_gate", {}).get(
                "conditional_probability"
            ),
        },
        "carry_500_success_teacher_corpus": {
            "passed": bool(
                paths["carry_manifest"].is_file()
                and carry_data.get("complete")
                and carry_data.get("accepted_demos", 0) >= 500
                and carry_data.get("gates", {}).get("demo_count_reached")
                and carry_data.get("gates", {}).get("success_rate_at_least_minimum")
            ),
            "teacher_successes": carry_data.get("accepted_demos", 0),
            "teacher_trials": carry_data.get("attempts", 0),
            "aggregate_teacher_success_rate": carry_data.get(
                "aggregate_teacher_success_rate"
            ),
            "shards": len(carry_data.get("shards", [])),
        },
        "loaded_carry_policy": {
            "passed": bool(
                paths["carry_checkpoint"].is_file()
                and _policy_contract_ok(carry_policy, phase="carry", action_dim=23)
                and carry_policy_trained_from_complete_corpus
                and carry_student_physical.get("trials", 0) >= 20
                and carry_student_physical.get("conditional_success_rate", 0.0) >= 0.8
            ),
            "teacher_successes": carry_data.get("accepted_demos", 0),
            "trained_from_complete_500_corpus": (
                carry_policy_trained_from_complete_corpus
            ),
            "physical_smoke_trials": carry_student_physical.get("trials", 0),
            "physical_smoke_successes": carry_student_physical.get("successes", 0),
            "physical_smoke_success_rate": carry_student_physical.get(
                "conditional_success_rate"
            ),
            "required_physical_gate_trials": 20,
        },
        "place_release_settle_policy": {
            "passed": bool(
                place_data.get("complete")
                and place_data.get("accepted_demos", 0) >= 600
                and paths["place_manifest"].is_file()
                and paths["place_checkpoint"].is_file()
                and _policy_contract_ok(place_policy, phase="place", action_dim=23)
                and place_policy_trained_from_complete_corpus
                and place_policy.get("place_phase_clock", {}).get("privileged") is False
                and place_policy.get("place_observation_adapter", {}).get("privileged")
                is False
                and place_student_physical.get("gate_passed") is True
            ),
            "teacher_successes": place_data.get("accepted_demos", 0),
            "teacher_trials": place_data.get("attempts", 0),
            "diversified_new_initial_states": 100,
            "trained_from_complete_600_corpus": (
                place_policy_trained_from_complete_corpus
            ),
            "physical_gate_trials": place_student_physical.get("trials", 0),
            "physical_gate_successes": place_student_physical.get("successes", 0),
            "physical_gate_success_rate": place_student_physical.get(
                "physical_success_rate"
            ),
            "required_physical_success_rate": place_student_physical.get(
                "minimum_success_rate", 0.8
            ),
        },
        "chained_approach_grasp_smoke": {
            "passed": bool(
                chained.get("episodes", 0) >= 1
                and chained.get("P(approach)") == 1.0
                and chained.get("P(grasp|approach)") == 1.0
                and chained.get("grasp_checkpoint_certified_by_gate")
            ),
            "episodes": chained.get("episodes", 0),
        },
        "full_fsm_20_episode_gate": {
            "passed": bool(
                fsm.get("episodes", 0) >= fsm_criteria["min_episodes"]
                and all(
                    isinstance(fsm.get(metric), (int, float))
                    and fsm[metric] >= fsm_criteria["min_conditional_success_rate"]
                    for metric in conditional_metrics
                )
                and fsm.get("full_task_success", 0.0)
                >= fsm_criteria["min_full_task_success_rate"]
                and fsm.get("bad_orientation_rate", 1.0)
                <= fsm_criteria["max_bad_orientation_rate"]
                and fsm.get("grasp_checkpoint_certified_by_gate")
            ),
            "criteria": fsm_criteria,
            "episodes": fsm.get("episodes", 0),
            "full_task_success": fsm.get("full_task_success"),
            "conditional_success": {
                metric: fsm.get(metric) for metric in conditional_metrics
            },
        },
    }
    return {
        "format": "unitree_g1_43dof_loco_manipulation_pipeline_audit_v3",
        "production_complete": all(stage["passed"] for stage in stages.values()),
        "teacher_success_inventory": {
            "approach": manifest.get("accepted_demos", 0),
            "grasp_lift": manifest.get("accepted_demos", 0),
            "carry": carry_data.get("accepted_demos", 0),
            "place": place_data.get("accepted_demos", 0),
        },
        "stages": stages,
        "rl_fine_tuning": {
            "environment_ready": True,
            "mode": "four_frozen_43d_bc_nominal_plus_bounded_43d_ppo_residual_with_rnd",
            "bc_fsm_residual_training_implemented": bool(
                paths["residual_adapter"].is_file()
                and paths["residual_approach_checkpoint"].is_file()
                and _policy_contract_ok(
                    residual_approach, phase="approach", action_dim=43
                )
            ),
            "training_execution_completed": bool(
                residual_models
                and residual_provenance.get("schema")
                == "unitree_box_bc_residual_nominal_v1"
                and residual_training.get("update_iterations", 0) >= 100
            ),
            "production_gate_passed": bool(
                residual_training.get("production_gate_passed")
            ),
            "update_iterations": residual_training.get("update_iterations"),
            "transitions": residual_training.get("transitions"),
            "measured_final": residual_training.get("measured_final", {}),
            "latest_training_run": None if residual_run is None else str(residual_run),
            "latest_training_checkpoint": (
                None if not residual_models else str(residual_models[-1])
            ),
            "note": (
                "The adapter, smoke run, and 100-update PPO+RND run completed. The final "
                "success rate is zero and bad-orientation termination rate is one, so the "
                "checkpoint is rejected rather than promoted. The latest BC/DAgger FSM "
                "can complete the full task, but has not yet passed the 20-episode gate."
            ),
        },
        "artifact_paths": {name: str(path) for name, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every production stage has passed",
    )
    args = parser.parse_args()
    result = audit(args.repo_root.expanduser().resolve())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and not result["production_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
