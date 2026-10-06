#!/usr/bin/env python3
"""Train, evaluate, and certify phase-wise Box behavior-cloning policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.utils.teacher.phase_bc import (  # noqa: E402
    ACTION_DIMS,
    SKILL_BY_NAME,
    PhaseArrays,
    PhaseInferenceConfig,
    TrainConfig,
    certify_bc2_gate,
    discover_dagger_episodes,
    discover_episodes,
    evaluate_checkpoint,
    expand_teacher_dataset_paths,
    load_bc_policy,
    load_dagger_phase_arrays,
    load_phase_arrays,
    merge_phase_arrays,
    parse_skill,
    split_episodes,
    train_phase_policy,
    validate_bc2_gate,
    validate_complete_teacher_corpus,
    validate_strict_grasp_corpus,
    validate_teacher_action_contract,
    file_sha256,
    write_metrics_json,
)
from unitree_rl_lab.utils.teacher.skill_fsm import SkillPhase  # noqa: E402
from unitree_rl_lab.utils.teacher.student_observation import (  # noqa: E402
    STUDENT_FEATURE_SLICES,
)


PLACE_PHASE_CLOCK_SCHEMA = "unitree_place_phase_clock_v1"


def _tuple_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "hidden dims must be comma-separated integers"
        ) from error
    if not result or any(width <= 0 for width in result):
        raise argparse.ArgumentTypeError("hidden dims must contain positive integers")
    return result


def _ratios(value: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "split ratios must be comma-separated numbers"
        ) from error
    if len(parsed) != 3 or any(item < 0.0 for item in parsed) or sum(parsed) <= 0.0:
        raise argparse.ArgumentTypeError(
            "split ratios must be three non-negative numbers with positive sum"
        )
    return parsed  # type: ignore[return-value]


def _add_datasets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Teacher HDF5 file(s), shard directory, manifest.json, or glob",
    )
    parser.add_argument(
        "--dagger-datasets",
        nargs="+",
        type=Path,
        default=(),
        help="Optional flat DAgger correction HDF5 file(s), grouped by source_episode for splitting",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed demonstrations (off by default; randomized BC corpora should stay success-only)",
    )
    parser.add_argument("--hand-close-threshold", type=float, default=0.5)
    parser.add_argument(
        "--grasp-lead-steps",
        type=int,
        default=2,
        help="Include two pre-close frames in BC-2 so the physical hand-cage FSM handoff is covered",
    )


def _inference_config(args: argparse.Namespace) -> PhaseInferenceConfig:
    return PhaseInferenceConfig(
        hand_close_threshold=args.hand_close_threshold,
        grasp_lead_steps=args.grasp_lead_steps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-wise BC with a strict 517-D QR/proprio student boundary. "
            "Train and physically certify grasp_lift (BC-2) before other skills."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train one phase policy")
    _add_datasets(train)
    train.add_argument("--skill", choices=sorted(SKILL_BY_NAME), required=True)
    train.add_argument(
        "--action-key",
        choices=sorted(ACTION_DIMS),
        default="processed_43d_joint_target",
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--metrics-json", type=Path)
    train.add_argument("--split-ratios", type=_ratios, default=(0.8, 0.1, 0.1))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--hidden-dims", type=_tuple_ints, default=(512, 512, 256))
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-5)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument(
        "--checkpoint-selection",
        choices=("validation", "last"),
        default="validation",
        help=(
            "Checkpoint selection rule; 'last' is intended only for a "
            "small targeted DAgger probe followed by physical gating"
        ),
    )
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--device", default="cuda:0")
    train.add_argument(
        "--initial-checkpoint",
        type=Path,
        help=(
            "Fine-tune from a same-skill checkpoint while preserving its "
            "normalization; supports corrective DAgger or successful boundary demos"
        ),
    )
    train.add_argument(
        "--dagger-loss-weight",
        type=float,
        default=1.0,
        help="Relative training loss weight for DAgger samples (nominal samples use 1.0)",
    )
    train.add_argument(
        "--source-weight-substring",
        help=(
            "Give nominal training episodes whose resolved source UID contains "
            "this substring an additional multiplicative weight"
        ),
    )
    train.add_argument("--source-weight", type=float, default=1.0)
    train.add_argument(
        "--phase-start-steps",
        type=int,
        default=0,
        help="Give the first N selected samples of every training episode extra loss weight",
    )
    train.add_argument(
        "--phase-start-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for --phase-start-steps (used for the CARRY handoff boundary)",
    )
    train.add_argument(
        "--place-phase-clock-steps",
        type=int,
        default=0,
        help=(
            "For PLACE only, encode clipped within-phase progress in the public "
            "velocity_commands.x observation feature; 0 disables the adapter"
        ),
    )
    train.add_argument(
        "--place-mask-history-and-last-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For PLACE only, mask policy_history and last_action during both "
            "training and runtime so snapshot-history mismatch cannot drive the policy"
        ),
    )
    train.add_argument(
        "--bc2-gate",
        type=Path,
        help="Required passing BC-2 certification marker when training approach/carry/place",
    )
    train.add_argument(
        "--allow-incomplete-manifest",
        action="store_true",
        help="Diagnostic/smoke-test only: allow training from an incomplete production shard manifest",
    )

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a checkpoint on teacher data"
    )
    _add_datasets(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--subset", choices=("train", "validation", "test", "all"), default="test"
    )
    evaluate.add_argument("--batch-size", type=int, default=2048)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--output-json", type=Path)

    certify = commands.add_parser(
        "certify-bc2",
        help="Bind held-out BC error and physical rollout metrics into a BC-2 gate",
    )
    certify.add_argument("--checkpoint", type=Path, required=True)
    certify.add_argument("--rollout-metrics", type=Path, required=True)
    certify.add_argument("--output", type=Path, required=True)
    certify.add_argument("--max-validation-normalized-mse", type=float, default=0.25)
    certify.add_argument("--min-bimanual-lift-success-rate", type=float, default=0.80)
    certify.add_argument("--max-bad-orientation-rate", type=float, default=0.05)
    certify.add_argument("--min-rollout-episodes", type=int, default=20)
    return parser


def _inject_place_phase_clock(arrays: PhaseArrays, horizon_steps: int) -> PhaseArrays:
    """Add a public within-PLACE progress signal without changing the 517-D ABI."""

    if horizon_steps <= 0 or arrays.num_samples == 0:
        return arrays
    observations = arrays.observations.copy()
    feature_index = STUDENT_FEATURE_SLICES["velocity_commands"].start
    for episode_index in np.unique(arrays.sample_episode_index):
        sample_indices = np.flatnonzero(arrays.sample_episode_index == episode_index)
        observations[sample_indices, feature_index] = np.minimum(
            np.arange(sample_indices.size, dtype=np.float32) / horizon_steps,
            1.0,
        )
    return PhaseArrays(
        observations=observations,
        actions=arrays.actions,
        episode_ids=arrays.episode_ids,
        sample_episode_index=arrays.sample_episode_index,
        skill=arrays.skill,
        action_key=arrays.action_key,
    )


def _mask_place_dynamic_history(arrays: PhaseArrays) -> PhaseArrays:
    if arrays.num_samples == 0:
        return arrays
    observations = arrays.observations.copy()
    for name in ("policy_history", "last_action"):
        observations[:, STUDENT_FEATURE_SLICES[name].as_slice()] = 0.0
    return PhaseArrays(
        observations=observations,
        actions=arrays.actions,
        episode_ids=arrays.episode_ids,
        sample_episode_index=arrays.sample_episode_index,
        skill=arrays.skill,
        action_key=arrays.action_key,
    )


def _train(args: argparse.Namespace) -> dict:
    skill = parse_skill(args.skill)
    if not np.isfinite(args.dagger_loss_weight) or args.dagger_loss_weight <= 0.0:
        raise ValueError("--dagger-loss-weight must be finite and greater than zero")
    if not np.isfinite(args.source_weight) or args.source_weight < 1.0:
        raise ValueError("--source-weight must be finite and at least 1")
    if args.source_weight > 1.0 and not args.source_weight_substring:
        raise ValueError("--source-weight > 1 requires --source-weight-substring")
    if args.phase_start_steps < 0:
        raise ValueError("--phase-start-steps must be non-negative")
    if not np.isfinite(args.phase_start_weight) or args.phase_start_weight < 1.0:
        raise ValueError("--phase-start-weight must be finite and at least 1")
    if args.place_phase_clock_steps < 0:
        raise ValueError("--place-phase-clock-steps must be non-negative")
    if args.place_phase_clock_steps and skill != SkillPhase.PLACE:
        raise ValueError("--place-phase-clock-steps is supported only for PLACE")
    if args.place_mask_history_and_last_action and skill != SkillPhase.PLACE:
        raise ValueError(
            "--place-mask-history-and-last-action is supported only for PLACE"
        )
    if skill != SkillPhase.GRASP_LIFT:
        if args.bc2_gate is None:
            raise RuntimeError(
                "BC-2 must be physically validated first: pass --bc2-gate from the certify-bc2 command"
            )
        # The gate certifies the upstream GRASP_LIFT boundary, not the output
        # representation chosen by the downstream skill being trained.  BC1
        # may intentionally predict 23-D high-level targets while the
        # certified BC2 remains a processed 43-D joint-target policy.
        validate_bc2_gate(args.bc2_gate)
    if not args.allow_incomplete_manifest:
        validate_complete_teacher_corpus(args.datasets)
    teacher_paths = expand_teacher_dataset_paths(args.datasets)
    episodes = discover_episodes(teacher_paths, success_only=not args.include_failed)
    if skill == SkillPhase.GRASP_LIFT:
        validate_strict_grasp_corpus(episodes, required_hold_steps=100)
    dagger_episodes = discover_dagger_episodes(args.dagger_datasets)
    split = split_episodes(
        [*episodes, *dagger_episodes], ratios=args.split_ratios, seed=args.seed
    )
    inference = _inference_config(args)
    teacher_uids = {episode.uid for episode in episodes}
    dagger_uids = {episode.uid for episode in dagger_episodes}
    arrays = {}
    nominal_arrays = {}
    corrective_arrays = {}
    train_sample_weights = None
    for subset in ("train", "validation", "test"):
        subset_ids = set(split.ids(subset))
        nominal = load_phase_arrays(
            episodes,
            subset_ids & teacher_uids,
            skill=skill,
            action_key=args.action_key,
            inference_config=inference,
        )
        corrective = load_dagger_phase_arrays(
            dagger_episodes,
            subset_ids & dagger_uids,
            skill=skill,
            action_key=args.action_key,
        )
        if args.place_phase_clock_steps:
            nominal = _inject_place_phase_clock(nominal, args.place_phase_clock_steps)
            corrective = _inject_place_phase_clock(
                corrective, args.place_phase_clock_steps
            )
        if args.place_mask_history_and_last_action:
            nominal = _mask_place_dynamic_history(nominal)
            corrective = _mask_place_dynamic_history(corrective)
        nominal_arrays[subset] = nominal
        corrective_arrays[subset] = corrective
        arrays[subset] = merge_phase_arrays(nominal, corrective)
        if subset == "train":
            nominal_weights = np.ones((nominal.num_samples,), dtype=np.float32)
            if args.source_weight_substring and args.source_weight > 1.0:
                for episode_index, episode_uid in enumerate(nominal.episode_ids):
                    if args.source_weight_substring in episode_uid:
                        nominal_weights[
                            nominal.sample_episode_index == episode_index
                        ] *= args.source_weight
            if args.phase_start_steps and args.phase_start_weight > 1.0:
                for episode_index in np.unique(nominal.sample_episode_index):
                    sample_indices = np.flatnonzero(
                        nominal.sample_episode_index == episode_index
                    )
                    nominal_weights[sample_indices[: args.phase_start_steps]] *= (
                        args.phase_start_weight
                    )
            corrective_weights = np.full(
                (corrective.num_samples,),
                args.dagger_loss_weight,
                dtype=np.float32,
            )
            if (
                corrective.num_samples
                or args.phase_start_steps
                and args.phase_start_weight > 1.0
            ):
                train_sample_weights = np.concatenate(
                    (nominal_weights, corrective_weights)
                )
    if skill == SkillPhase.GRASP_LIFT and arrays["validation"].num_samples == 0:
        raise RuntimeError(
            "BC-2 requires a nonempty episode-level validation split before certification"
        )
    report = train_phase_policy(
        train_arrays=arrays["train"],
        validation_arrays=nominal_arrays["validation"],
        test_arrays=nominal_arrays["test"],
        split=split,
        output_path=args.output,
        config=TrainConfig(
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
        ),
        device=args.device,
        source_paths=[*teacher_paths, *args.dagger_datasets],
        initial_checkpoint=args.initial_checkpoint,
        train_sample_weights=train_sample_weights,
        selection_validation_arrays=arrays["validation"],
        auxiliary_evaluation_arrays={
            "dagger_validation": corrective_arrays["validation"],
            "dagger_test": corrective_arrays["test"],
        }
        if dagger_episodes
        else None,
        select_last_epoch=args.checkpoint_selection == "last",
        checkpoint_metadata_extra=(
            {
                **(
                    {
                        "place_phase_clock": {
                            "schema": PLACE_PHASE_CLOCK_SCHEMA,
                            "feature_index": STUDENT_FEATURE_SLICES[
                                "velocity_commands"
                            ].start,
                            "feature_name": "velocity_commands.x",
                            "horizon_steps": args.place_phase_clock_steps,
                            "encoding": "clip(place_step / horizon_steps, 0, 1)",
                            "privileged": False,
                        }
                    }
                    if args.place_phase_clock_steps
                    else {}
                ),
                "place_observation_adapter": {
                    "schema": "unitree_place_observation_adapter_v1",
                    "masked_feature_names": ["policy_history", "last_action"]
                    if args.place_mask_history_and_last_action
                    else [],
                    "privileged": False,
                },
            }
            if args.place_phase_clock_steps or args.place_mask_history_and_last_action
            else None
        ),
    )
    report["checkpoint_selection"] = args.checkpoint_selection
    report["dagger_loss_weight"] = args.dagger_loss_weight
    report["source_weighting"] = {
        "substring": args.source_weight_substring,
        "weight": args.source_weight,
    }
    report["phase_start_weighting"] = {
        "steps_per_episode": args.phase_start_steps,
        "weight": args.phase_start_weight,
    }
    report["place_phase_clock_steps"] = args.place_phase_clock_steps
    report["place_mask_history_and_last_action"] = (
        args.place_mask_history_and_last_action
    )
    metrics_path = args.metrics_json or args.output.with_suffix(
        args.output.suffix + ".metrics.json"
    )
    write_metrics_json(report, metrics_path)
    report["metrics_json"] = str(metrics_path.expanduser().resolve())
    return report


def _evaluate(args: argparse.Namespace) -> dict:
    loaded = load_bc_policy(args.checkpoint, args.device)
    teacher_paths = expand_teacher_dataset_paths(args.datasets)
    validate_teacher_action_contract(
        [*teacher_paths, *args.dagger_datasets],
        loaded.action_key,
        expected_names=loaded.ordered_action_names,
    )
    episodes = discover_episodes(teacher_paths, success_only=not args.include_failed)
    dagger_episodes = discover_dagger_episodes(args.dagger_datasets)
    all_episodes = [*episodes, *dagger_episodes]
    if args.subset == "all":
        episode_ids = tuple(episode.uid for episode in all_episodes)
    else:
        split = loaded.metadata.get("episode_split", {})
        if args.subset not in split:
            raise ValueError(
                f"checkpoint does not contain a {args.subset!r} episode split"
            )
        episode_ids = tuple(split[args.subset])
    known_uids = {episode.uid for episode in all_episodes}
    unknown = set(episode_ids).difference(known_uids)
    if unknown:
        raise ValueError(
            f"evaluation dataset is missing {len(unknown)} episodes stored in the checkpoint split"
        )
    teacher_uids = {episode.uid for episode in episodes}
    dagger_uids = {episode.uid for episode in dagger_episodes}
    nominal = load_phase_arrays(
        episodes,
        set(episode_ids) & teacher_uids,
        skill=loaded.skill_phase,
        action_key=loaded.action_key,
        inference_config=_inference_config(args),
    )
    corrective = load_dagger_phase_arrays(
        dagger_episodes,
        set(episode_ids) & dagger_uids,
        skill=loaded.skill_phase,
        action_key=loaded.action_key,
    )
    phase_clock = loaded.metadata.get("place_phase_clock")
    if phase_clock is not None:
        if loaded.skill_phase != SkillPhase.PLACE:
            raise ValueError("non-PLACE checkpoint declares a PLACE phase clock")
        if phase_clock.get("schema") != PLACE_PHASE_CLOCK_SCHEMA:
            raise ValueError("checkpoint has an unsupported PLACE phase clock")
        nominal = _inject_place_phase_clock(nominal, int(phase_clock["horizon_steps"]))
        corrective = _inject_place_phase_clock(
            corrective, int(phase_clock["horizon_steps"])
        )
    adapter = loaded.metadata.get("place_observation_adapter")
    if adapter is not None:
        if adapter.get("schema") != "unitree_place_observation_adapter_v1":
            raise ValueError("checkpoint has an unsupported PLACE observation adapter")
        masked_names = tuple(adapter.get("masked_feature_names", ()))
        if set(masked_names) != {"policy_history", "last_action"}:
            if masked_names:
                raise ValueError("checkpoint masks unsupported PLACE features")
        if masked_names:
            nominal = _mask_place_dynamic_history(nominal)
            corrective = _mask_place_dynamic_history(corrective)
    arrays = merge_phase_arrays(nominal, corrective)
    report = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "skill_phase": loaded.metadata["skill_phase"],
        "action_key": loaded.action_key,
        "subset": args.subset,
        "metrics": evaluate_checkpoint(
            args.checkpoint, arrays, device=args.device, batch_size=args.batch_size
        ),
    }
    if args.output_json:
        write_metrics_json(report, args.output_json)
    return report


def _certify(args: argparse.Namespace) -> dict:
    with args.rollout_metrics.expanduser().open("r", encoding="utf-8") as stream:
        rollout_metrics = json.load(stream)
    return certify_bc2_gate(
        args.checkpoint,
        rollout_metrics,
        args.output,
        max_validation_normalized_mse=args.max_validation_normalized_mse,
        min_bimanual_lift_success_rate=args.min_bimanual_lift_success_rate,
        max_bad_orientation_rate=args.max_bad_orientation_rate,
        min_rollout_episodes=args.min_rollout_episodes,
        rollout_metrics_sha256=file_sha256(args.rollout_metrics),
        rollout_metrics_path=args.rollout_metrics,
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "train":
        report = _train(args)
    elif args.command == "evaluate":
        report = _evaluate(args)
    else:
        report = _certify(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
