#!/usr/bin/env python3
"""Phase 8 Method C: train BC-1 v3 (and BC-2 v3, --skill grasp_lift) on collision-free Teacher demos only.

Same model, loss, approach labelling (phase 0 before the first close/contact/lift minus a 2-step grasp lead),
normalization (env-noise floor) and checkpoint format as BC-1 v2 (train_bc1_handoff.py / phase_bc.train_phase_policy),
but the data are only the demos listed by select_safe_demos.py: Teacher demos with the Method A/B collision-free
approach (no table contact, minimum clearance kept).  The original 500-demo corpus and the post-navigation v1 demos
all touch the table (phase-8 geometric analysis: 500/500 and 100/100) and are not used.  Episodes are split 80/10/10
(seed 0) per source file group, so no episode leaks across train/validation/test.  There is no BC-2 gate.

BC-2 v3 (--skill grasp_lift): the demos are the safe Teacher grasp/lift continuations recorded from BC-1 v3 gate states
(run_p8_collect_bc2.sh, accepted by select_safe_demos.py --phase bc2), the targets the 43-D joint targets as for BC-2 v2.
--bc2-labels from_handoff uses every recorded continuation step, i.e. everything BC-2 is responsible for in the FSM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bc1 import ENV_NOISE_FLOOR, observation_floor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accepted", type=Path, nargs="+", required=True, help="select_safe_demos.py outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-floor", choices=("none", "env_noise"), default="env_noise")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-lead-steps", type=int, default=2)
    parser.add_argument("--skill", choices=("approach", "grasp_lift"), default="approach",
                        help="approach = BC-1 (23-D high-level action); grasp_lift = BC-2 (43-D joint targets) from the safe continuations")
    parser.add_argument("--action-key", default=None, help="default: teacher_high_level_action (approach) / processed_43d_joint_target (grasp_lift)")
    parser.add_argument("--bc2-labels", choices=("phase_bc", "from_handoff"), default="phase_bc",
                        help="grasp_lift only: phase_bc = the BC-2 v2 rule (phase 0 from the first close/contact/lift minus the grasp "
                             "lead); from_handoff = every phase-0 step of a continuation demo (recorded from the BC-1 gate on), so "
                             "BC-2 also learns the steps between an early gate and the close onset, as it must in the FSM")
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

    import h5py

    from unitree_rl_lab.utils.teacher import phase_bc as pbc

    if args.skill == "grasp_lift" and args.bc2_labels == "from_handoff":
        phase_bc_labels = pbc.infer_bc_skill_labels

        def from_handoff(group, *, config=pbc.PhaseInferenceConfig()):
            labels = phase_bc_labels(group, config=config)
            labels[labels == int(pbc.SkillPhase.APPROACH)] = int(pbc.SkillPhase.GRASP_LIFT)
            return labels

        pbc.infer_bc_skill_labels = from_handoff  # load_phase_arrays looks the labeller up at call time

    accepted, rules = set(), []
    for path in args.accepted:
        data = json.loads(path.read_text())
        accepted.update(data["accepted"])
        rules.append({"file": str(path), "rule": data["rule"], "runs": data["runs"]})
    paths = sorted({uid.split("::")[0] for uid in accepted})
    episodes = [e for e in pbc.discover_episodes(paths, success_only=True) if e.uid in accepted]
    if len(episodes) != len(accepted):
        missing = sorted(accepted - {e.uid for e in episodes})
        raise SystemExit(f"{len(missing)} accepted demos were not found or are not successful: {missing[:5]}")
    split = pbc.split_episodes(episodes, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    inference = pbc.PhaseInferenceConfig(hand_close_threshold=0.5, grasp_lead_steps=args.grasp_lead_steps)
    action_key = args.action_key or ("teacher_high_level_action" if args.skill == "approach" else "processed_43d_joint_target")
    arrays = {subset: pbc.load_phase_arrays(episodes, split.ids(subset), skill=args.skill, action_key=action_key,
                                            inference_config=inference) for subset in ("train", "validation", "test")}
    with h5py.File(paths[0], "r") as stream:
        joint_names = [n.decode() if isinstance(n, bytes) else str(n) for n in stream["metadata/processed_joint_names"][:]]
    floor = observation_floor(joint_names) if args.obs_floor == "env_noise" else None
    composition = Counter(Path(e.path).name for e in episodes)
    extra_meta = {
        "approach_research": {
            "trainer": "scripts/approach/train_bc1_safe.py",
            "phase": 8,
            "skill": args.skill,
            "action_key": action_key,
            "bc2_labels": args.bc2_labels if args.skill == "grasp_lift" else None,
            "bc2_gate": None,
            "original_corpus": None,
            "data": "collision-free Teacher demos only (select_safe_demos.py)",
            "demos_by_file": dict(composition),
            "selection_rules": rules,
            "episodes_and_samples": {k: [v.num_episodes, v.num_samples] for k, v in arrays.items()},
            "observation_normalization_variant": args.obs_floor,
            "env_noise_floor_by_term": ENV_NOISE_FLOOR if floor is not None else None,
            "teacher_corpus_camera_fov_deg": 69.4,
            "reproduces_senior_r05": False,
        }
    }
    report = pbc.train_phase_policy(
        train_arrays=arrays["train"], validation_arrays=arrays["validation"], test_arrays=arrays["test"], split=split,
        output_path=output,
        config=pbc.TrainConfig(learning_rate=args.learning_rate, weight_decay=args.weight_decay, batch_size=args.batch_size,
                               epochs=args.epochs, patience=args.patience, seed=args.seed),
        device=args.device, source_paths=paths, selection_validation_arrays=arrays["validation"],
        checkpoint_metadata_extra=extra_meta, observation_scale_floor=floor)
    pbc.write_metrics_json(report, metrics_path)
    print(json.dumps({"checkpoint": report["checkpoint"], "best_epoch": report["best_epoch"],
                      "episodes_and_samples": extra_meta["approach_research"]["episodes_and_samples"],
                      "normalized_mse": {k: v["normalized_mse"] for k, v in report["metrics"].items() if v}}, indent=2))


if __name__ == "__main__":
    main()
