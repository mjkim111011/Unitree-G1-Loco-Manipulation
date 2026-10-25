#!/usr/bin/env python3
"""Train BC-1 on the original Teacher corpus plus post-navigation handoff demos.

The original corpus is split, labelled and loaded exactly as in train_bc1.py
(seed 0, 80/10/10, approach = phase 0 before the first close/contact/lift
minus the 2-step grasp lead).  Handoff demos recorded by
``eval_bc1_approach.py --mode teacher_collect`` are split the same way on
their own and appended to each subset, so no handoff episode leaks across
train/validation/test.  ``--extra-repeat k`` repeats the handoff training
samples k times (validation/test are never repeated).  Model, loss,
normalization (optionally with the env-noise floor) and checkpoint format are
those of ``phase_bc.train_phase_policy``.  There is no BC-2 gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_bc1_diagnostics import DEFAULT_CORPUS  # noqa: E402
from train_bc1 import ENV_NOISE_FLOOR, observation_floor  # noqa: E402


def concat_arrays(pbc, parts: list, repeats: list[int]):
    observations, actions, episode_ids, index = [], [], [], []
    for arrays, repeat in zip(parts, repeats):
        offset = len(episode_ids)
        episode_ids.extend(arrays.episode_ids)
        for _ in range(repeat):
            observations.append(arrays.observations)
            actions.append(arrays.actions)
            index.append(arrays.sample_episode_index + offset)
    return pbc.PhaseArrays(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        episode_ids=tuple(episode_ids),
        sample_episode_index=np.concatenate(index, axis=0),
        skill=parts[0].skill,
        action_key=parts[0].action_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--extra", nargs="+", required=True, help="handoff demo HDF5 files (teacher_collect output)")
    parser.add_argument("--extra-repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-floor", choices=("none", "env_noise"), default="env_noise")
    parser.add_argument("--skill", choices=("approach", "grasp_lift"), default="approach")
    parser.add_argument("--action-key", default="teacher_high_level_action",
                        help="teacher_high_level_action (BC-1) or processed_43d_joint_target (BC-2)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-lead-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.extra_repeat < 1:
        raise SystemExit("--extra-repeat must be >= 1")

    output = args.output.expanduser().resolve()
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    if output.exists() or metrics_path.exists():
        raise SystemExit(f"refusing to overwrite {output} or its metrics")
    output.parent.mkdir(parents=True, exist_ok=True)

    import h5py

    from unitree_rl_lab.utils.teacher import phase_bc as pbc

    corpus = Path(args.corpus).absolute()
    pbc.validate_complete_teacher_corpus([corpus])
    original_paths = pbc.expand_teacher_dataset_paths([corpus])
    original = pbc.discover_episodes(original_paths, success_only=True)
    extra_paths = pbc.expand_teacher_dataset_paths([Path(p).absolute() for p in args.extra])
    extra = pbc.discover_episodes(extra_paths, success_only=True)
    split_original = pbc.split_episodes(original, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    split_extra = pbc.split_episodes(extra, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    inference = pbc.PhaseInferenceConfig(hand_close_threshold=0.5, grasp_lead_steps=args.grasp_lead_steps)

    def load(episodes, split, subset):
        return pbc.load_phase_arrays(episodes, split.ids(subset), skill=args.skill,
                                     action_key=args.action_key, inference_config=inference)

    per_source = {}
    arrays = {}
    for subset in ("train", "validation", "test"):
        a, b = load(original, split_original, subset), load(extra, split_extra, subset)
        per_source[subset] = {"original": [a.num_episodes, a.num_samples], "handoff": [b.num_episodes, b.num_samples]}
        arrays[subset] = concat_arrays(pbc, [a, b], [1, args.extra_repeat if subset == "train" else 1])
    split = pbc.EpisodeSplit(
        train=split_original.train + split_extra.train,
        validation=split_original.validation + split_extra.validation,
        test=split_original.test + split_extra.test,
        seed=args.seed,
        ratios=(0.8, 0.1, 0.1),
    )
    with h5py.File(original_paths[0], "r") as stream:
        joint_names = [n.decode() if isinstance(n, bytes) else str(n) for n in stream["metadata/processed_joint_names"][:]]
    with h5py.File(extra_paths[0], "r") as stream:
        extra_names = [n.decode() if isinstance(n, bytes) else str(n) for n in stream["metadata/processed_joint_names"][:]]
    if extra_names != joint_names:
        raise SystemExit("handoff demos use a different joint order than the original corpus")
    floor = observation_floor(joint_names) if args.obs_floor == "env_noise" else None
    extra_meta = {
        "approach_research": {
            "trainer": "scripts/approach/train_bc1_handoff.py",
            "skill": args.skill,
            "action_key": args.action_key,
            "bc2_gate": None,
            "original_corpus": str(corpus),
            "handoff_demo_files": [str(p) for p in extra_paths],
            "handoff_extra_repeat": args.extra_repeat,
            "episodes_and_samples_by_source": per_source,
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
        source_paths=list(original_paths) + list(extra_paths),
        selection_validation_arrays=arrays["validation"],
        checkpoint_metadata_extra=extra_meta,
        observation_scale_floor=floor,
    )
    pbc.write_metrics_json(report, metrics_path)
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "best_epoch": report["best_epoch"],
        "episodes_and_samples_by_source": per_source,
        "normalized_mse": {k: v["normalized_mse"] for k, v in report["metrics"].items() if v},
    }, indent=2))


if __name__ == "__main__":
    main()
