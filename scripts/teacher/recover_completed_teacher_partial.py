#!/usr/bin/env python3
"""Atomically republish complete demo groups left by an interrupted collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("recovery output must differ from the partial input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as input_file:
        if input_file.attrs.get("schema_name") != "unitree_box_teacher_demo":
            raise ValueError("input is not a Unitree Box teacher corpus")
        group_names = tuple(
            name
            for name, group in input_file["data"].items()
            if bool(group.attrs.get("complete"))
            and int(group.attrs.get("num_samples", 0)) > 0
            and np.any(group["success"][:])
        )
        groups = tuple(input_file["data"][name] for name in group_names)
        if not groups:
            raise ValueError("no complete successful episodes are recoverable")
        if any(int(group.attrs.get("num_samples", 0)) <= 0 for group in groups):
            raise ValueError("every recovered episode must contain samples")
        if any(not np.any(group["success"][:]) for group in groups):
            raise ValueError("recovery accepts only physically successful episodes")

    fd, partial_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".recovery", dir=output.parent
    )
    os.close(fd)
    recovery_path = Path(partial_name)
    try:
        with h5py.File(source, "r") as input_file, h5py.File(
            recovery_path, "w", track_order=True
        ) as output_file:
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value
            for name in input_file:
                if name != "data":
                    input_file.copy(name, output_file)
            data = output_file.create_group("data", track_order=True)
            for name in group_names:
                input_file.copy(input_file["data"][name], data, name=name)
            metadata = json.loads(str(output_file.attrs["metadata_json"]))
            metadata["recovered_from_interrupted_partial"] = str(source)
            metadata["recovery_preserved_completed_episode_groups_only"] = True
            output_file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            output_file.attrs["complete"] = True
            output_file.attrs["completed_utc"] = datetime.now(timezone.utc).isoformat()
            output_file.flush()
        os.replace(recovery_path, output)
    finally:
        recovery_path.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with h5py.File(output, "r") as output_file:
        metadata = json.loads(str(output_file.attrs["metadata_json"]))
        successes = len(output_file["data"])
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_run_id": metadata["collection_run_id"],
                "phase": metadata["teacher_phase"],
                "trials": successes,
                "successes": successes,
                "dataset_published": True,
                "published_output": str(output),
                "dataset_sha256": digest,
                "recovered_from_interrupted_partial": str(source),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "metrics": str(metrics_path),
                "recovered_episodes": successes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
