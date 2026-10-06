#!/usr/bin/env python3
"""Validate current-task Box teacher HDF5 schema and nominal lift/hold gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from unitree_rl_lab.utils.teacher.student_observation import (
    STUDENT_FEATURE_SLICES,
    STUDENT_OBSERVATION_DIM,
)


REQUIRED_TRAILING_SHAPES = {
    "teacher_high_level_action": (23,),
    "processed_43d_joint_target": (43,),
    "phase_id": (),
    "box_pose": (7,),
    "bin_pose": (7,),
    "root_pose": (7,),
    "box_size": (3,),
    "left_contact": (),
    "right_contact": (),
    "left_contact_count": (),
    "right_contact_count": (),
    "lift": (),
    "drop": (),
    "success": (),
}
OPTIONAL_TRAILING_SHAPES = {
    "left_wrist_pose": (7,),
    "right_wrist_pose": (7,),
    "left_fingertip_positions": (3, 3),
    "right_fingertip_positions": (3, 3),
}


def _maximum_true_run(mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _tilt_from_pose(pose: np.ndarray, convention: str) -> np.ndarray:
    quaternion = np.asarray(pose[:, 3:7], dtype=np.float64)
    if convention == "xyzw":
        quaternion = quaternion[:, [3, 0, 1, 2]]
    elif convention != "wxyz":
        raise ValueError(f"Unsupported quaternion convention: {convention!r}")
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("root_pose contains a zero quaternion")
    _, x, y, _ = (quaternion / norm).T
    body_up_z = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(body_up_z, -1.0, 1.0))


def _validate_episode(
    name: str,
    group: h5py.Group,
    *,
    quaternion_convention: str,
    lift_height_m: float,
    hold_steps: int,
    bad_orientation_limit_rad: float,
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    if not bool(group.attrs.get("complete", False)):
        errors.append(f"{name}: episode is not marked complete")
    sample_count = int(group.attrs.get("num_samples", -1))
    if sample_count <= 0:
        errors.append(f"{name}: invalid num_samples={sample_count}")

    shapes = {**REQUIRED_TRAILING_SHAPES, "student_observation": None}
    for dataset_name, trailing_shape in shapes.items():
        if dataset_name not in group:
            errors.append(f"{name}: missing dataset {dataset_name!r}")
            continue
        dataset = group[dataset_name]
        if dataset.shape[0] != sample_count:
            errors.append(
                f"{name}/{dataset_name}: first dimension {dataset.shape[0]} != {sample_count}"
            )
        if trailing_shape is not None and dataset.shape[1:] != trailing_shape:
            errors.append(
                f"{name}/{dataset_name}: trailing shape {dataset.shape[1:]} != {trailing_shape}"
            )
        if dataset_name == "student_observation" and (
            dataset.ndim != 2 or dataset.shape[1] != STUDENT_OBSERVATION_DIM
        ):
            errors.append(
                f"{name}/student_observation must have shape (N, {STUDENT_OBSERVATION_DIM})"
            )

    for dataset_name, trailing_shape in OPTIONAL_TRAILING_SHAPES.items():
        if dataset_name in group and group[dataset_name].shape != (
            sample_count,
            *trailing_shape,
        ):
            errors.append(
                f"{name}/{dataset_name}: shape {group[dataset_name].shape} "
                f"!= {(sample_count, *trailing_shape)}"
            )

    if errors:
        return {"episode": name, "schema_valid": False}, errors

    for dataset_name in (
        "student_observation",
        "teacher_high_level_action",
        "processed_43d_joint_target",
        "box_pose",
        "bin_pose",
        "root_pose",
        "box_size",
    ):
        if not np.isfinite(group[dataset_name][:]).all():
            errors.append(f"{name}/{dataset_name}: contains non-finite values")
    if np.any(group["box_size"][:] <= 0.0):
        errors.append(f"{name}/box_size: values must be positive")

    box_height = np.asarray(group["box_pose"][:, 2], dtype=np.float64)
    lifted = box_height >= box_height[0] + lift_height_m
    bilateral_contact = np.asarray(group["left_contact"][:], dtype=bool) & np.asarray(
        group["right_contact"][:], dtype=bool
    )
    qualified_hold = lifted & bilateral_contact
    max_hold_steps = _maximum_true_run(qualified_hold)
    tilt = _tilt_from_pose(group["root_pose"][:], quaternion_convention)
    bimanual_grasp_lift = bool(np.any(qualified_hold))
    hold_success = max_hold_steps >= hold_steps
    bad_orientation = bool(np.any(tilt > bad_orientation_limit_rad))
    any_drop = bool(np.any(np.asarray(group["drop"][:], dtype=bool)))
    safe_hold_success = hold_success and not bad_orientation and not any_drop
    recorded_success = bool(group.attrs.get("success", False))
    if recorded_success != safe_hold_success:
        errors.append(
            f"{name}: recorded success={recorded_success} disagrees with safe hold={safe_hold_success}"
        )
    high_level = np.asarray(group["teacher_high_level_action"][:], dtype=np.float32)
    observations = np.asarray(group["student_observation"][:], dtype=np.float32)
    command_feature = STUDENT_FEATURE_SLICES["velocity_commands"]
    observed_command = observations[:, command_feature.start : command_feature.stop]
    close_indices = np.flatnonzero(np.any(high_level[:, :2] > 0.5, axis=1))
    if close_indices.size == 0:
        errors.append(f"{name}: no bimanual close onset for BC-2")
        close_index = sample_count
    else:
        close_index = int(close_indices[0])
    if not np.allclose(
        observed_command[:close_index], high_level[:close_index, 16:19], atol=1.0e-6
    ):
        errors.append(f"{name}: APPROACH observation/action camera commands disagree")
    if not np.allclose(observed_command[close_index:], 0.0, atol=1.0e-7):
        errors.append(f"{name}: BC-2 observation velocity command is not zero")
    if not np.allclose(high_level[close_index:, 16:19], 0.0, atol=1.0e-7):
        errors.append(f"{name}: BC-2 teacher locomotion command is not zero")

    return (
        {
            "episode": name,
            "schema_valid": not errors,
            "num_samples": sample_count,
            "student_observation_dim": int(group["student_observation"].shape[1]),
            "bimanual_grasp_lift": bimanual_grasp_lift,
            "two_second_hold": hold_success,
            "safe_two_second_hold": safe_hold_success,
            "max_continuous_hold_steps": max_hold_steps,
            "max_lift_m": float(np.max(box_height - box_height[0])),
            "max_tilt_rad": float(np.max(tilt)),
            "bad_orientation": bad_orientation,
            "drop": any_drop,
            "grasp_lift_close_index": close_index,
            "recorded_success": recorded_success,
            "termination_reason": str(group.attrs.get("termination_reason", "")),
        },
        errors,
    )


def validate(args: argparse.Namespace) -> dict[str, object]:
    dataset_path = args.dataset.expanduser().resolve()
    errors: list[str] = []
    with h5py.File(dataset_path, "r") as dataset:
        if not bool(dataset.attrs.get("complete", False)):
            errors.append("file is not marked complete")
        if "data" not in dataset or not dataset["data"]:
            errors.append("file has no episodes under /data")
            episodes: list[dict[str, object]] = []
            control_frequency_hz = float(
                dataset.attrs.get("control_frequency_hz", math.nan)
            )
        else:
            if "metadata/processed_joint_names" not in dataset:
                errors.append("missing ordered /metadata/processed_joint_names action contract")
            elif dataset["metadata/processed_joint_names"].shape != (43,):
                errors.append("processed_joint_names action contract must contain 43 names")
            control_frequency_hz = float(
                dataset.attrs.get("control_frequency_hz", math.nan)
            )
            if not math.isfinite(control_frequency_hz) or control_frequency_hz <= 0.0:
                errors.append(f"invalid control_frequency_hz={control_frequency_hz}")
            hold_steps = int(math.ceil(args.hold_seconds * control_frequency_hz))
            convention = str(dataset.attrs.get("quaternion_convention", ""))
            episodes = []
            for name in sorted(dataset["data"]):
                result, episode_errors = _validate_episode(
                    name,
                    dataset["data"][name],
                    quaternion_convention=convention,
                    lift_height_m=args.lift_height_m,
                    hold_steps=hold_steps,
                    bad_orientation_limit_rad=args.bad_orientation_limit_rad,
                )
                episodes.append(result)
                errors.extend(episode_errors)

    count = len(episodes)
    grasp_lift_rate = sum(
        bool(item.get("bimanual_grasp_lift")) for item in episodes
    ) / max(1, count)
    hold_rate = sum(bool(item.get("safe_two_second_hold")) for item in episodes) / max(
        1, count
    )
    bad_orientation_rate = sum(
        bool(item.get("bad_orientation")) for item in episodes
    ) / max(1, count)
    drop_rate = sum(bool(item.get("drop")) for item in episodes) / max(1, count)
    gates = {
        "schema_valid": not errors,
        "grasp_lift_at_least_minimum": grasp_lift_rate >= args.minimum_grasp_lift_rate,
        "hold_at_least_minimum": hold_rate >= args.minimum_hold_rate,
        "bad_orientation_at_most_maximum": bad_orientation_rate
        <= args.maximum_bad_orientation_rate,
        "no_published_drop": drop_rate == 0.0,
    }
    return {
        "dataset": str(dataset_path),
        "control_frequency_hz": control_frequency_hz,
        "episode_count": count,
        "criteria": {
            "lift_height_m": args.lift_height_m,
            "hold_seconds": args.hold_seconds,
            "bad_orientation_limit_rad": args.bad_orientation_limit_rad,
            "minimum_grasp_lift_rate": args.minimum_grasp_lift_rate,
            "minimum_hold_rate": args.minimum_hold_rate,
            "maximum_bad_orientation_rate": args.maximum_bad_orientation_rate,
        },
        "rates": {
            "bimanual_grasp_lift": grasp_lift_rate,
            "two_second_hold": hold_rate,
            "bad_orientation": bad_orientation_rate,
            "drop": drop_rate,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "errors": errors,
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--lift-height-m", type=float, default=0.12)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--bad-orientation-limit-rad", type=float, default=0.8)
    parser.add_argument("--minimum-grasp-lift-rate", type=float, default=0.90)
    parser.add_argument("--minimum-hold-rate", type=float, default=0.90)
    parser.add_argument("--maximum-bad-orientation-rate", type=float, default=0.05)
    args = parser.parse_args()
    for name in ("lift_height_m", "hold_seconds", "bad_orientation_limit_rad"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "minimum_grasp_lift_rate",
        "minimum_hold_rate",
        "maximum_bad_orientation_rate",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    report = validate(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
