#!/usr/bin/env python3
"""Build a SHA-bound manifest for PLACE Student-state Teacher relabel shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCHEMA = "unitree_place_student_teacher_relabel_shards_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: object, *, label: str) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-episodes", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.target_episodes <= 0:
        raise ValueError("--target-episodes must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")

    shards: list[dict[str, object]] = []
    checkpoint_sha256: str | None = None
    source_references: set[tuple[str, str]] = set()
    total_episodes = 0
    total_samples = 0
    total_successes = 0
    total_recovery_steps = 0
    total_left_recovery_steps = 0
    total_right_recovery_steps = 0

    for raw_path in args.datasets:
        path = raw_path.expanduser().resolve()
        sidecar = path.with_suffix(".metrics.json")
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"dataset or metrics sidecar is missing: {path}")
        metrics = json.loads(sidecar.read_text(encoding="utf-8"))
        if metrics.get("status") != "completed":
            raise ValueError(f"collection did not complete: {path}")
        if metrics.get("evaluation_mode") != "student_teacher_relabel":
            raise ValueError(f"dataset is not a Student-state relabel run: {path}")
        if metrics.get("dataset_published") is not True:
            raise ValueError(f"dataset is not publication-bound: {path}")
        if Path(str(metrics.get("published_output"))).resolve() != path:
            raise ValueError(f"sidecar points to a different published output: {path}")
        digest = _sha256(path)
        if metrics.get("dataset_sha256") != digest:
            raise ValueError(f"dataset SHA-256 mismatch: {path}")

        shard_episodes = 0
        shard_samples = 0
        shard_successes = 0
        shard_recovery_steps = 0
        shard_left_recovery_steps = 0
        shard_right_recovery_steps = 0
        with h5py.File(path, "r") as source:
            if not bool(source.attrs.get("complete", False)):
                raise ValueError(f"HDF5 dataset is incomplete: {path}")
            metadata = _json_object(
                source.attrs.get("metadata_json", "{}"), label="metadata_json"
            )
            if metadata.get("teacher_phase") != "place":
                raise ValueError(f"dataset phase is not PLACE: {path}")
            if metadata.get("action_source") != "teacher_relabels_on_student_states":
                raise ValueError(f"dataset action source is not Teacher relabels: {path}")
            if metadata.get("collection_run_id") != metrics.get("collection_run_id"):
                raise ValueError(f"collection run id mismatch: {path}")
            current_checkpoint_sha = metadata.get("student_checkpoint_sha256")
            if not current_checkpoint_sha:
                raise ValueError(f"Student checkpoint SHA is missing: {path}")
            if checkpoint_sha256 is None:
                checkpoint_sha256 = str(current_checkpoint_sha)
            elif current_checkpoint_sha != checkpoint_sha256:
                raise ValueError("relabel shards use different Student checkpoints")
            snapshot_input = str(metadata.get("snapshot_input", ""))
            if not snapshot_input:
                raise ValueError(f"snapshot input provenance is missing: {path}")

            for episode_name in sorted(source["data"]):
                group = source["data"][episode_name]
                if not bool(group.attrs.get("complete", False)):
                    raise ValueError(f"incomplete episode {episode_name}: {path}")
                final = _json_object(
                    group.attrs.get("final_metadata_json", "{}"),
                    label=f"{episode_name}.final_metadata_json",
                )
                source_episode = str(final.get("source_episode", ""))
                if not source_episode:
                    raise ValueError(
                        f"source episode provenance is missing for {episode_name}: {path}"
                    )
                source_references.add((snapshot_input, source_episode))
                samples = int(group.attrs.get("num_samples", 0))
                if samples <= 0 or samples != len(group["student_observation"]):
                    raise ValueError(f"invalid sample count for {episode_name}: {path}")
                expected_shapes = {
                    "student_observation": (samples, 517),
                    "teacher_high_level_action": (samples, 23),
                    "processed_43d_joint_target": (samples, 43),
                }
                for key, expected_shape in expected_shapes.items():
                    values = np.asarray(group[key][:])
                    if values.shape != expected_shape:
                        raise ValueError(
                            f"invalid {key} shape {values.shape} for "
                            f"{episode_name}: {path}"
                        )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"non-finite {key} values for {episode_name}: {path}"
                        )
                shard_episodes += 1
                shard_samples += samples
                shard_successes += int(bool(group.attrs.get("success", False)))
                shard_recovery_steps += int(
                    final.get("student_teacher_relabel_recovery_steps", 0)
                )
                shard_left_recovery_steps += int(
                    final.get("student_teacher_relabel_left_recovery_steps", 0)
                )
                shard_right_recovery_steps += int(
                    final.get("student_teacher_relabel_right_recovery_steps", 0)
                )

        metric_results = metrics.get("results")
        if not isinstance(metric_results, list):
            raise ValueError(f"metrics results list is missing: {path}")
        recorded_results = [item for item in metric_results if item.get("recorded")]
        if shard_episodes != len(recorded_results):
            raise ValueError(f"HDF5/metrics recorded episode count mismatch: {path}")
        recorded_successes = sum(int(bool(item.get("success"))) for item in recorded_results)
        if shard_successes != recorded_successes:
            raise ValueError(f"HDF5/metrics success count mismatch: {path}")
        shards.append(
            {
                "path": str(path),
                "metrics_path": str(sidecar.resolve()),
                "sha256": digest,
                "metrics_sha256": _sha256(sidecar),
                "collection_run_id": metrics["collection_run_id"],
                "episodes": shard_episodes,
                "physical_trials": int(metrics.get("trials", len(metric_results))),
                "samples": shard_samples,
                "successes": shard_successes,
                "failures": shard_episodes - shard_successes,
                "recovery_steps": shard_recovery_steps,
                "left_recovery_steps": shard_left_recovery_steps,
                "right_recovery_steps": shard_right_recovery_steps,
            }
        )
        total_episodes += shard_episodes
        total_samples += shard_samples
        total_successes += shard_successes
        total_recovery_steps += shard_recovery_steps
        total_left_recovery_steps += shard_left_recovery_steps
        total_right_recovery_steps += shard_right_recovery_steps

    count_reached = total_episodes >= args.target_episodes
    has_recovery_labels = total_recovery_steps > 0
    unique_sources = len(source_references)
    source_diversity_reached = unique_sources >= args.target_episodes
    complete = count_reached and has_recovery_labels and source_diversity_reached
    payload = {
        "schema": SCHEMA,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "student_checkpoint_sha256": checkpoint_sha256,
        "target_episodes": args.target_episodes,
        "episodes": total_episodes,
        "samples": total_samples,
        "successes": total_successes,
        "failures": total_episodes - total_successes,
        "student_rollout_success_rate": total_successes / max(1, total_episodes),
        "unique_source_snapshot_references": unique_sources,
        "recovery_steps": total_recovery_steps,
        "left_recovery_steps": total_left_recovery_steps,
        "right_recovery_steps": total_right_recovery_steps,
        "gates": {
            "episode_count_reached": count_reached,
            "recovery_labels_present": has_recovery_labels,
            "unique_source_snapshot_per_episode": source_diversity_reached,
            "all_shards_sha_bound": True,
            "all_labels_generated_on_student_states": True,
            "all_observation_and_action_tensors_valid": True,
        },
        "shards": shards,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
