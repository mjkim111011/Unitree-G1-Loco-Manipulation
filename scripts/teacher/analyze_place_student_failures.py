#!/usr/bin/env python3
"""Summarize hand-loss geometry in published PLACE Student rollout datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


BOX_VISIBLE_INDEX = 506
BIN_VISIBLE_INDEX = 510


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_attr(value: object) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("metadata attribute must contain a JSON object")
    return parsed


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return math.pi
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    world_z_dot_body_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(world_z_dot_body_z, -1.0, 1.0)))


def _phase_region(step: int, trajectory: dict) -> str:
    raise_end = int(trajectory["raise_steps"])
    translate_end = raise_end + int(trajectory["translate_steps"])
    lower_end = translate_end + int(trajectory["lower_steps"])
    if step < raise_end:
        return "raise"
    if step < translate_end:
        return "translate"
    if step < lower_end:
        return "lower"
    return "release_or_settle"


def _finite_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")

    records: list[dict] = []
    source_records: list[dict] = []
    checkpoint_sha: str | None = None
    for raw_path in args.datasets:
        path = raw_path.expanduser().resolve()
        sidecar = path.with_suffix(".metrics.json")
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"dataset or sidecar is missing: {path}")
        metrics = json.loads(sidecar.read_text(encoding="utf-8"))
        if metrics.get("dataset_published") is not True:
            raise ValueError(f"dataset is not publication-bound: {path}")
        if metrics.get("dataset_sha256") != _sha256(path):
            raise ValueError(f"dataset SHA-256 mismatch: {path}")
        with h5py.File(path, "r") as source:
            metadata = _json_attr(source.attrs.get("metadata_json", "{}"))
            if metadata.get("teacher_phase") != "place":
                raise ValueError(f"dataset is not PLACE: {path}")
            current_sha = metadata.get("student_checkpoint_sha256")
            if checkpoint_sha is None:
                checkpoint_sha = current_sha
            elif current_sha != checkpoint_sha:
                raise ValueError("datasets use different Student checkpoints")
            trajectory = metadata["place_trajectory"]
            for episode_name in sorted(source["data"]):
                group = source["data"][episode_name]
                final = _json_attr(group.attrs.get("final_metadata_json", "{}"))
                reason = str(group.attrs.get("termination_reason", "unknown"))
                left = np.asarray(group["left_contact_count"][:], dtype=np.int64)
                right = np.asarray(group["right_contact_count"][:], dtype=np.int64)
                missing = np.flatnonzero((left == 0) | (right == 0))
                first_loss = int(missing[0]) if missing.size else None
                record = {
                    "dataset": str(path),
                    "episode": episode_name,
                    "source_episode": final.get("source_episode"),
                    "success": bool(group.attrs.get("success", False)),
                    "termination_reason": reason,
                    "steps": int(len(left)),
                    "first_contact_loss_step": first_loss,
                }
                if first_loss is not None:
                    index = first_loss
                    box = np.asarray(group["box_pose"][index], dtype=np.float64)
                    bin_pose = np.asarray(group["bin_pose"][index], dtype=np.float64)
                    left_wrist = np.asarray(
                        group["left_wrist_pose"][index, :3], dtype=np.float64
                    )
                    right_wrist = np.asarray(
                        group["right_wrist_pose"][index, :3], dtype=np.float64
                    )
                    observation = np.asarray(
                        group["student_observation"][index], dtype=np.float64
                    )
                    record.update(
                        {
                            "first_contact_loss_seconds": index
                            / float(source.attrs["control_frequency_hz"]),
                            "first_loss_side": (
                                "both"
                                if left[index] == 0 and right[index] == 0
                                else "left"
                                if left[index] == 0
                                else "right"
                            ),
                            "first_loss_phase_region": _phase_region(index, trajectory),
                            "left_wrist_to_box_center_m": float(
                                np.linalg.norm(left_wrist - box[:3])
                            ),
                            "right_wrist_to_box_center_m": float(
                                np.linalg.norm(right_wrist - box[:3])
                            ),
                            "wrist_distance_asymmetry_m": float(
                                abs(
                                    np.linalg.norm(left_wrist - box[:3])
                                    - np.linalg.norm(right_wrist - box[:3])
                                )
                            ),
                            "box_to_bin_xy_m": float(
                                np.linalg.norm(box[:2] - bin_pose[:2])
                            ),
                            "box_tilt_rad": _tilt_rad(box[3:7]),
                            "box_visible": bool(observation[BOX_VISIBLE_INDEX] > 0.5),
                            "bin_visible": bool(observation[BIN_VISIBLE_INDEX] > 0.5),
                        }
                    )
                records.append(record)
        source_records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "metrics_path": str(sidecar.resolve()),
                "metrics_sha256": _sha256(sidecar),
            }
        )

    failures = [record for record in records if not record["success"]]
    losses = [
        record for record in failures if record["first_contact_loss_step"] is not None
    ]
    report = {
        "schema": "unitree_place_student_failure_analysis_v1",
        "student_checkpoint_sha256": checkpoint_sha,
        "episodes": len(records),
        "successes": sum(int(record["success"]) for record in records),
        "failures": len(failures),
        "termination_reason_counts": dict(
            sorted(Counter(record["termination_reason"] for record in records).items())
        ),
        "first_loss_side_counts": dict(
            sorted(Counter(record["first_loss_side"] for record in losses).items())
        ),
        "first_loss_phase_region_counts": dict(
            sorted(
                Counter(record["first_loss_phase_region"] for record in losses).items()
            )
        ),
        "first_loss_seconds": _finite_summary(
            [record["first_contact_loss_seconds"] for record in losses]
        ),
        "wrist_distance_asymmetry_m": _finite_summary(
            [record["wrist_distance_asymmetry_m"] for record in losses]
        ),
        "box_to_bin_xy_m": _finite_summary(
            [record["box_to_bin_xy_m"] for record in losses]
        ),
        "box_tilt_rad": _finite_summary([record["box_tilt_rad"] for record in losses]),
        "visibility_at_first_loss": {
            "box_visible": sum(int(record["box_visible"]) for record in losses),
            "bin_visible": sum(int(record["bin_visible"]) for record in losses),
            "eligible": len(losses),
        },
        "sources": source_records,
        "episode_records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
