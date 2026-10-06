#!/usr/bin/env python3
"""Merge provenance-compatible physical snapshot stores without changing states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.utils.teacher.phase_bc import file_sha256  # noqa: E402
from unitree_rl_lab.utils.teacher.physical_snapshots import (  # noqa: E402
    PhysicalSnapshotStore,
    load_snapshot_store_metadata,
    load_snapshots,
)


CRITICAL_METADATA_KEYS = (
    "producer",
    "actual_approach_grasp_chain",
    "checkpoint_sha256",
    "action_key",
    "action_contract_sha256",
    "bc2_gate_sha256",
    "strict_hold_seconds",
    "strict_lift_m",
    "required_hold_steps",
)


def _joint_names(path: Path) -> tuple[str, ...]:
    with h5py.File(path, "r") as h5_file:
        return tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5_file["joint_names"][:]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-source-episode",
        action="append",
        default=[],
        help="Exclude a known failed phase-boundary state by its provenance episode",
    )
    parser.add_argument(
        "--exclude-source-ref",
        action="append",
        default=[],
        metavar="EPISODE:STEP",
        help="Exclude one exact failed boundary state without removing its episode siblings",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = tuple(path.expanduser().resolve() for path in args.inputs)
    if len(set(paths)) != len(paths):
        raise ValueError("input snapshot paths must be unique")
    if args.output.expanduser().resolve() in paths:
        raise ValueError("output must differ from every input")

    metadata = [load_snapshot_store_metadata(path) for path in paths]
    reference = metadata[0]
    for path, current in zip(paths[1:], metadata[1:], strict=True):
        differences = {
            key: (reference.get(key), current.get(key))
            for key in CRITICAL_METADATA_KEYS
            if reference.get(key) != current.get(key)
        }
        if differences:
            raise ValueError(
                f"incompatible snapshot provenance in {path}: {differences}"
            )

    joint_names = _joint_names(paths[0])
    merged_metadata = {
        **reference,
        "merged_without_state_modification": True,
        "merged_sources": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "snapshot_count": len(
                    load_snapshots(path, expected_joint_names=joint_names)
                ),
            }
            for path in paths
        ],
    }
    all_snapshots = tuple(
        snapshot
        for path in paths
        for snapshot in load_snapshots(path, expected_joint_names=joint_names)
    )
    excluded = set(args.exclude_source_episode)
    excluded_refs = set()
    for value in args.exclude_source_ref:
        episode, separator, step_text = value.rpartition(":")
        if not separator or not episode:
            raise ValueError("--exclude-source-ref must use EPISODE:STEP")
        excluded_refs.add((episode, int(step_text)))
    snapshots = tuple(
        snapshot
        for snapshot in all_snapshots
        if snapshot.source_episode not in excluded
        and (snapshot.source_episode, snapshot.source_step) not in excluded_refs
    )
    if not snapshots:
        raise RuntimeError("snapshot filters removed every input state")
    merged_metadata["excluded_source_episodes"] = sorted(excluded)
    merged_metadata["excluded_source_refs"] = sorted(
        f"{episode}:{step}" for episode, step in excluded_refs
    )
    with PhysicalSnapshotStore(
        args.output,
        joint_names=joint_names,
        overwrite=args.overwrite,
        metadata=merged_metadata,
    ) as store:
        for snapshot in snapshots:
            store.append(snapshot)

    report = {
        "output": str(args.output.expanduser().resolve()),
        "output_sha256": file_sha256(args.output),
        "snapshot_count": len(snapshots),
        "excluded_snapshot_count": len(all_snapshots) - len(snapshots),
        "source_count": len(paths),
        "sources": merged_metadata["merged_sources"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
