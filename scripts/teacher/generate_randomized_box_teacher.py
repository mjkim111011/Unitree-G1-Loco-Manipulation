#!/usr/bin/env python3
"""Generate a resumable 500--1000 demo Box-x/y teacher dataset.

Each shard is collected in live physics by ``collect_nominal_box_teacher.py``.
That collector uses Isaac Lab Mimic's translation coordination scheme for the
Box and both wrists, and discards every rollout that does not pass the strict
bilateral 12 cm lift + continuous hold filter.  Sharding makes a multi-hour
generation run resumable without appending to or repairing an open HDF5 file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from unitree_rl_lab.utils.teacher import episode_passes_bimanual_lift_hold
from unitree_rl_lab.runtime_paths import default_arena_root, default_arena_dataset
from unitree_rl_lab.utils.teacher.student_observation import (
    STUDENT_FEATURE_SLICES,
    STUDENT_OBSERVATION_DIM,
)


@dataclass(frozen=True, slots=True)
class ShardAudit:
    path: Path
    episodes: int
    total_steps: int
    invalid_episodes: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.episodes > 0 and not self.invalid_episodes


def audit_success_only_shard(
    path: str | Path,
    *,
    required_hold_steps: int,
    expected_box_size_m: float | None = None,
    expected_box_x_range: tuple[float, float] | None = None,
    expected_box_y_range: tuple[float, float] | None = None,
    expected_initial_box_height_m: float | None = None,
) -> ShardAudit:
    """Independently verify that every published episode passes the hard gate."""

    shard_path = Path(path).expanduser().resolve()
    invalid: list[str] = []
    total_steps = 0
    with h5py.File(shard_path, "r") as dataset:
        if not bool(dataset.attrs.get("complete", False)) or "data" not in dataset:
            raise ValueError(f"Incomplete teacher shard: {shard_path}")
        if "metadata/processed_joint_names" not in dataset:
            raise ValueError(f"Shard has no ordered 43-D action contract: {shard_path}")
        joint_names = tuple(dataset["metadata/processed_joint_names"].asstr()[:])
        if len(joint_names) != 43 or len(set(joint_names)) != 43:
            raise ValueError(f"Shard has an invalid 43-D action contract: {shard_path}")
        episodes = sorted(name for name in dataset["data"] if name.startswith("demo_"))
        for name in episodes:
            episode = dataset[f"data/{name}"]
            total_steps += int(episode.attrs.get("num_samples", len(episode["lift"])))
            required_fields = {
                "lift",
                "left_contact",
                "right_contact",
                "drop",
                "root_pose",
                "box_size",
                "student_observation",
                "teacher_high_level_action",
            }
            if not required_fields.issubset(episode.keys()):
                invalid.append(name)
                continue
            strict_pass = episode_passes_bimanual_lift_hold(
                lift=episode["lift"][:],
                left_contact=episode["left_contact"][:],
                right_contact=episode["right_contact"][:],
                required_hold_steps=required_hold_steps,
            )
            root_pose = np.asarray(episode["root_pose"][:], dtype=np.float64)
            quat = root_pose[:, 3:7]
            quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1.0e-12, None)
            tilt = np.arccos(
                np.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0)
            )
            observations = np.asarray(episode["student_observation"][:], dtype=np.float32)
            high_level = np.asarray(episode["teacher_high_level_action"][:], dtype=np.float32)
            close = np.flatnonzero(np.any(high_level[:, :2] > 0.5, axis=1))
            command_slice = STUDENT_FEATURE_SLICES["velocity_commands"]
            command = observations[:, command_slice.start : command_slice.stop]
            command_contract_valid = bool(close.size) and observations.shape[1:] == (
                STUDENT_OBSERVATION_DIM,
            )
            if close.size:
                close_index = int(close[0])
                command_contract_valid &= np.allclose(
                    command[:close_index], high_level[:close_index, 16:19], atol=1.0e-6
                ) and np.allclose(command[close_index:], 0.0, atol=1.0e-7)
                command_contract_valid &= np.allclose(
                    high_level[close_index:, 16:19], 0.0, atol=1.0e-7
                )
            scene_valid = True
            if expected_box_size_m is not None:
                scene_valid &= np.allclose(
                    episode["box_size"][:], expected_box_size_m, atol=1.0e-6
                )
            if (
                expected_box_x_range is not None
                or expected_box_y_range is not None
                or expected_initial_box_height_m is not None
            ):
                metadata = json.loads(episode.attrs.get("metadata_json", "{}"))
                initial = np.asarray(metadata.get("initial_box_pose_wxyz", ()), dtype=np.float64)
                scene_valid &= initial.shape == (7,) and np.isfinite(initial).all()
                if initial.shape == (7,):
                    if expected_box_x_range is not None:
                        scene_valid &= expected_box_x_range[0] <= initial[0] <= expected_box_x_range[1]
                    if expected_box_y_range is not None:
                        scene_valid &= expected_box_y_range[0] <= initial[1] <= expected_box_y_range[1]
                    if expected_initial_box_height_m is not None:
                        scene_valid &= math.isclose(
                            initial[2], expected_initial_box_height_m, abs_tol=1.0e-5
                        )
                    scene_valid &= np.allclose(initial[3:], (1.0, 0.0, 0.0, 0.0), atol=1.0e-5)
            if (
                not strict_pass
                or not bool(episode.attrs.get("success", False))
                or np.any(episode["drop"][:])
                or np.any(tilt > 0.8)
                or not command_contract_valid
                or not scene_valid
            ):
                invalid.append(name)
    return ShardAudit(shard_path, len(episodes), total_steps, tuple(invalid))


def _read_metrics(path: Path) -> dict[str, Any]:
    metrics_path = path.with_suffix(".metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing shard metrics: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _numeric_match(value: Any, expected: Any) -> bool:
    recorded = np.asarray(value, dtype=np.float64)
    target = np.asarray(expected, dtype=np.float64)
    return recorded.shape == target.shape and bool(
        np.allclose(recorded, target, atol=1.0e-9, rtol=0.0)
    )


def _snapshot_path(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}.snapshots.hdf5")


def _snapshot_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing live physical snapshots: {path}")
    with h5py.File(path, "r") as dataset:
        if dataset.attrs.get("schema_name") != "unitree_box_physical_phase_snapshot":
            raise ValueError(f"Invalid physical snapshot schema: {path}")
        return int(dataset.attrs.get("num_snapshots", len(dataset["snapshots"])))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/teacher/box_xy_randomized_500")
    parser.add_argument("--target_demos", type=int, default=500)
    parser.add_argument("--shard_size", type=int, default=25)
    parser.add_argument("--minimum_success_rate", type=float, default=0.80)
    parser.add_argument("--max_attempt_factor", type=float, default=2.0)
    parser.add_argument("--hold_seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--box_x_range", type=float, nargs=2, default=(0.618, 0.622))
    parser.add_argument("--box_y_range", type=float, nargs=2, default=(-0.002, 0.002))
    parser.add_argument("--box_size", type=float, default=0.20)
    parser.add_argument("--box_mass", type=float, default=0.20)
    parser.add_argument("--box_static_friction", type=float, default=1.5)
    parser.add_argument("--box_dynamic_friction", type=float, default=1.2)
    parser.add_argument("--box_restitution", type=float, default=0.0)
    parser.add_argument("--wrist_compression_m", type=float, default=0.0)
    parser.add_argument("--teacher_hand_stiffness", type=float, default=4.0)
    parser.add_argument("--realized_hold_support_m", type=float, default=0.003)
    parser.add_argument(
        "--freeze_realized_hold_wrists",
        action="store_true",
        help="Experimental hold controller; disabled for the validated default pipeline.",
    )
    parser.add_argument("--task", default="Unitree-G1-43dof-BoxPickPlace")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--arena_dataset",
        default=default_arena_dataset(),
    )
    parser.add_argument("--arena_root", default=default_arena_root())
    parser.add_argument(
        "--collector",
        default=str(Path(__file__).with_name("collect_nominal_box_teacher.py")),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_headless", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.target_demos <= 0:
        raise ValueError(
            "--target_demos must be positive (the production gate is 500--1000)"
        )
    if not args.dry_run and not 500 <= args.target_demos <= 1000:
        raise ValueError("production --target_demos must be in [500, 1000]")
    if args.shard_size <= 0 or args.shard_size > args.target_demos:
        raise ValueError("--shard_size must be in [1, target_demos]")
    if not 0.0 < args.minimum_success_rate <= 1.0:
        raise ValueError("--minimum_success_rate must be in (0, 1]")
    if args.max_attempt_factor < 1.0:
        raise ValueError("--max_attempt_factor must be at least 1")
    if args.hold_seconds <= 0.0:
        raise ValueError("--hold_seconds must be positive")
    if (
        not 0.0 <= args.wrist_compression_m <= 0.03
        or args.teacher_hand_stiffness <= 0.0
    ):
        raise ValueError("invalid teacher wrist compression or hand stiffness")
    if not 0.0 <= args.realized_hold_support_m <= 0.02:
        raise ValueError("invalid realized hold support offset")
    for name in ("box_x_range", "box_y_range"):
        bounds = getattr(args, name)
        if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"--{name} requires finite MIN <= MAX")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    required_hold_steps = int(math.ceil(args.hold_seconds * 50.0))

    shard_records: list[dict[str, Any]] = []
    accepted_total = 0
    attempts_total = 0
    next_index = 0
    # Resume only from independently valid, success-only shards.  A corrupt or
    # failed-demo shard is never silently counted toward the target.
    for path in sorted(output_dir.glob("shard_*.hdf5")):
        if path.name.endswith(".snapshots.hdf5"):
            continue
        audit = audit_success_only_shard(
            path,
            required_hold_steps=required_hold_steps,
            expected_box_size_m=args.box_size,
            expected_box_x_range=tuple(args.box_x_range),
            expected_box_y_range=tuple(args.box_y_range),
            expected_initial_box_height_m=0.74 + 0.5 * args.box_size,
        )
        if not audit.valid:
            raise RuntimeError(
                f"Existing shard failed strict filter: {path}: {audit.invalid_episodes}"
            )
        metrics = _read_metrics(path)
        if metrics.get("randomization_mode") != "box_xy_only":
            raise RuntimeError(f"Existing shard is not Box-x/y-only: {path}")
        recorded_ranges = metrics.get("box_xy_range_m", {})
        if not _numeric_match(recorded_ranges.get("x", ()), args.box_x_range):
            raise RuntimeError(
                f"Existing shard x range does not match this run: {path}"
            )
        if not _numeric_match(recorded_ranges.get("y", ()), args.box_y_range):
            raise RuntimeError(
                f"Existing shard y range does not match this run: {path}"
            )
        fixed = metrics.get("fixed_scene_parameters", {})
        expected_fixed = {
            "box_size_m": [args.box_size] * 3,
            "box_mass_kg": args.box_mass,
            "static_friction": args.box_static_friction,
            "dynamic_friction": args.box_dynamic_friction,
            "restitution": args.box_restitution,
            "box_yaw_rad": 0.0,
        }
        for name, expected in expected_fixed.items():
            if not _numeric_match(fixed.get(name, ()), expected):
                raise RuntimeError(
                    f"Existing shard fixed {name} does not match this run: {path}"
                )
        attempts = int(metrics["attempts"])
        accepted = int(metrics["accepted_demonstrations"])
        if accepted != audit.episodes:
            raise RuntimeError(f"Shard metrics/HDF5 episode mismatch: {path}")
        if accepted / max(1, attempts) < args.minimum_success_rate:
            raise RuntimeError(
                f"Existing shard is below the teacher success gate: {path}"
            )
        snapshot_path = _snapshot_path(path)
        if _snapshot_count(snapshot_path) != accepted:
            raise RuntimeError(f"Shard demo/snapshot count mismatch: {path}")
        shard_records.append(
            {
                "path": str(path),
                "snapshot_path": str(snapshot_path),
                "accepted": accepted,
                "attempts": attempts,
                "total_steps": audit.total_steps,
                "success_rate": accepted / max(1, attempts),
                "seed": metrics.get("trial_results", [{}])[0].get("seed"),
            }
        )
        accepted_total += accepted
        attempts_total += attempts
        next_index = max(next_index, int(path.stem.rsplit("_", 1)[1]) + 1)

    def publish_manifest(*, complete: bool) -> None:
        aggregate_rate = accepted_total / max(1, attempts_total)
        _atomic_json(
            manifest_path,
            {
                "schema": "unitree_box_xy_randomized_teacher_shards_v1",
                "updated_utc": _utc_now(),
                "complete": complete,
                "target_demos": args.target_demos,
                "accepted_demos": accepted_total,
                "attempts": attempts_total,
                "aggregate_teacher_success_rate": aggregate_rate,
                "gates": {
                    "demo_count_reached": accepted_total >= args.target_demos,
                    "success_rate_at_least_minimum": aggregate_rate
                    >= args.minimum_success_rate,
                    "minimum_success_rate": args.minimum_success_rate,
                    "all_demos_strict_bilateral_lift_hold": True,
                },
                "randomization": {
                    "only": ["box_initial_x", "box_initial_y"],
                    "box_x_range_m": list(args.box_x_range),
                    "box_y_range_m": list(args.box_y_range),
                    "seed": args.seed,
                },
                "fixed": {
                    "box_size_m": [args.box_size] * 3,
                    "box_mass_kg": args.box_mass,
                    "box_static_friction": args.box_static_friction,
                    "box_dynamic_friction": args.box_dynamic_friction,
                    "box_restitution": args.box_restitution,
                    "box_yaw_rad": 0.0,
                    "table_height": "task TABLE_SURFACE_HEIGHT, startup randomization disabled",
                    "teacher_wrist_compression_m": args.wrist_compression_m,
                    "teacher_hand_stiffness": args.teacher_hand_stiffness,
                    "realized_hold_support_m": args.realized_hold_support_m,
                    "freeze_realized_hold_wrists": bool(
                        args.freeze_realized_hold_wrists
                    ),
                },
                "shards": shard_records,
            },
        )

    publish_manifest(complete=accepted_total >= args.target_demos)
    while accepted_total < args.target_demos:
        requested = min(args.shard_size, args.target_demos - accepted_total)
        max_attempts = int(math.ceil(requested * args.max_attempt_factor))
        shard_path = output_dir / f"shard_{next_index:05d}.hdf5"
        snapshot_path = _snapshot_path(shard_path)
        command = [
            str(Path(args.python).expanduser()),
            str(Path(args.collector).expanduser()),
            "--task",
            args.task,
            "--arena_dataset",
            args.arena_dataset,
            "--arena_root",
            args.arena_root,
            "--output",
            str(shard_path),
            "--snapshot_output",
            str(snapshot_path),
            "--target_successes",
            str(requested),
            "--max_attempts",
            str(max_attempts),
            "--randomize_box_xy",
            "--success_only",
            "--minimum_success_rate",
            str(args.minimum_success_rate),
            "--hold_seconds",
            str(args.hold_seconds),
            "--seed",
            str(args.seed + next_index),
            "--box_x_range",
            *(str(value) for value in args.box_x_range),
            "--box_y_range",
            *(str(value) for value in args.box_y_range),
            "--box_size",
            str(args.box_size),
            "--box_mass",
            str(args.box_mass),
            "--box_static_friction",
            str(args.box_static_friction),
            "--box_dynamic_friction",
            str(args.box_dynamic_friction),
            "--box_restitution",
            str(args.box_restitution),
            "--wrist_compression_m",
            str(args.wrist_compression_m),
            "--teacher_hand_stiffness",
            str(args.teacher_hand_stiffness),
            "--realized_hold_support_m",
            str(args.realized_hold_support_m),
            "--device",
            args.device,
        ]
        if not args.no_headless:
            command.append("--headless")
        if args.freeze_realized_hold_wrists:
            command.append("--freeze_realized_hold_wrists")
        print(" ".join(command), flush=True)
        if args.dry_run:
            return
        subprocess.run(command, check=True)

        audit = audit_success_only_shard(
            shard_path,
            required_hold_steps=required_hold_steps,
            expected_box_size_m=args.box_size,
            expected_box_x_range=tuple(args.box_x_range),
            expected_box_y_range=tuple(args.box_y_range),
            expected_initial_box_height_m=0.74 + 0.5 * args.box_size,
        )
        metrics = _read_metrics(shard_path)
        if not audit.valid or audit.episodes != requested:
            raise RuntimeError(
                f"Generated shard failed strict count/filter gate: {shard_path}; "
                f"episodes={audit.episodes}, invalid={audit.invalid_episodes}"
            )
        if _snapshot_count(snapshot_path) != audit.episodes:
            raise RuntimeError(
                f"Generated shard demo/snapshot count mismatch: {shard_path}"
            )
        attempts = int(metrics["attempts"])
        success_rate = audit.episodes / max(1, attempts)
        if success_rate < args.minimum_success_rate:
            raise RuntimeError(
                f"Shard teacher success {success_rate:.1%} is below {args.minimum_success_rate:.1%}"
            )
        shard_records.append(
            {
                "path": str(shard_path),
                "snapshot_path": str(snapshot_path),
                "accepted": audit.episodes,
                "attempts": attempts,
                "total_steps": audit.total_steps,
                "success_rate": success_rate,
                "seed": args.seed + next_index,
            }
        )
        accepted_total += audit.episodes
        attempts_total += attempts
        next_index += 1
        publish_manifest(complete=accepted_total >= args.target_demos)

    aggregate_rate = accepted_total / max(1, attempts_total)
    if aggregate_rate < args.minimum_success_rate:
        raise RuntimeError(
            f"Aggregate teacher success {aggregate_rate:.1%} is below {args.minimum_success_rate:.1%}"
        )
    print(
        f"complete: accepted={accepted_total}, attempts={attempts_total}, success={aggregate_rate:.1%}"
    )


if __name__ == "__main__":
    main()
