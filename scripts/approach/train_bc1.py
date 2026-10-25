#!/usr/bin/env python3
"""Train an approach-only BC-1 Student from the Teacher corpus.

Data loading, phase split, episode split, model, loss and checkpoint format are
exactly those of ``scripts/teacher/phase_bc.py train --skill approach``.  The
only differences are:

* no ``--bc2-gate``: that CLI flag is a pipeline-order precondition only and
  never enters BC-1 data, labels, normalization or weights.  The checkpoint
  records ``bc2_gate: null`` instead of pretending to be gated;
* ``--obs-floor env_noise`` optionally lower-bounds observation scales by the
  environment's own sensor-noise half-ranges (see ``observation_floor``).

This is a re-training on the locally regenerated corpus, not a reproduction of
the senior's r05 checkpoint (whose DAgger data is unavailable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_bc1_diagnostics import DEFAULT_CORPUS, feature_layout  # noqa: E402

# Observation-unit floors.  History terms use the Unoise half-ranges of
# PolicyHistoryCfg times the term scale (noise is applied before scale).
ENV_NOISE_FLOOR = {
    "base_ang_vel": 0.2 * 0.2,
    "projected_gravity": 0.05,
    "joint_pos_rel": 0.01,
    "joint_vel_rel": 1.5 * 0.05,
    "last_action": 0.01 / 0.25,  # 0.01 rad expressed in raw action units
    "velocity_commands": 0.01,
    "box_visible": 0.1,
    "bin_visible": 0.1,
    "box_position_camera": 0.01,
    "bin_position_camera": 0.01,
    "table_position_camera": 0.01,
}


def observation_floor(joint_names: list[str]) -> np.ndarray:
    _, groups, _ = feature_layout(joint_names)
    return np.asarray([ENV_NOISE_FLOOR[g.split(":")[0]] for g in groups], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-floor", choices=("none", "env_noise"), default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-lead-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    if output.exists() or metrics_path.exists():
        raise SystemExit(f"refusing to overwrite {output} or its metrics")
    output.parent.mkdir(parents=True, exist_ok=True)

    from unitree_rl_lab.utils.teacher import phase_bc as pbc

    import h5py

    corpus = Path(args.corpus).absolute()
    pbc.validate_complete_teacher_corpus([corpus])
    teacher_paths = pbc.expand_teacher_dataset_paths([corpus])
    episodes = pbc.discover_episodes(teacher_paths, success_only=True)
    split = pbc.split_episodes(episodes, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    inference = pbc.PhaseInferenceConfig(hand_close_threshold=0.5, grasp_lead_steps=args.grasp_lead_steps)
    arrays = {
        subset: pbc.load_phase_arrays(
            episodes,
            split.ids(subset),
            skill="approach",
            action_key="teacher_high_level_action",
            inference_config=inference,
        )
        for subset in ("train", "validation", "test")
    }
    with h5py.File(teacher_paths[0], "r") as stream:
        joint_names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in stream["metadata/processed_joint_names"][:]
        ]
    floor = observation_floor(joint_names) if args.obs_floor == "env_noise" else None
    extra = {
        "approach_research": {
            "trainer": "scripts/approach/train_bc1.py",
            "bc2_gate": None,
            "bc2_gate_note": (
                "phase_bc.py train requires --bc2-gate only as a pipeline-order "
                "precondition; it does not enter BC-1 data, labels, normalization or weights"
            ),
            "observation_normalization_variant": args.obs_floor,
            "env_noise_floor_by_term": ENV_NOISE_FLOOR if floor is not None else None,
            "teacher_corpus_camera_fov_deg": 69.4,
            "reproduces_senior_r05": False,
        }
    }
    report = pbc.train_phase_policy(
        train_arrays=arrays["train"],
        validation_arrays=arrays["validation"],
        test_arrays=arrays["test"],
        split=split,
        output_path=output,
        config=pbc.TrainConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        ),
        device=args.device,
        source_paths=list(teacher_paths),
        selection_validation_arrays=arrays["validation"],
        checkpoint_metadata_extra=extra,
        observation_scale_floor=floor,
    )
    pbc.write_metrics_json(report, metrics_path)
    summary = {
        "checkpoint": report["checkpoint"],
        "best_epoch": report["best_epoch"],
        "samples": report["sample_counts"],
        "normalized_mse": {k: v["normalized_mse"] for k, v in report["metrics"].items() if v},
        "observation_scale_floor": report.get("observation_scale_floor", {}).get("raised_dims"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
