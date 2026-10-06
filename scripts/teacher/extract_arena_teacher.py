#!/usr/bin/env python3
"""Create a camera-free retargeting dataset and audit from an Arena HDF5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.utils.teacher.arena_dataset import extract_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit IsaacLab-Arena G1 demonstrations and extract 23-D/43-D actions, "
            "Dex3 targets, inferred phases, and object-relative wrist transforms. RGB is never copied."
        )
    )
    parser.add_argument("source", type=Path, help="Source Arena demonstration HDF5.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Derived camera-free HDF5 path."
    )
    parser.add_argument(
        "--audit-json", type=Path, required=True, help="Human-readable audit JSON path."
    )
    parser.add_argument(
        "--episode",
        action="append",
        dest="episodes",
        help="Episode group such as demo_0. Repeat to select multiple; default extracts all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = extract_dataset(
        args.source,
        args.output,
        args.audit_json,
        episode_names=args.episodes,
        overwrite=args.overwrite,
    )
    summary = {
        "derived_hdf5": str(args.output.resolve()),
        "audit_json": str(args.audit_json.resolve()),
        "episodes": {
            name: {
                "samples": episode["num_samples"],
                "control_frequency_hz": episode["timing"]["control_frequency_hz"],
                "phase_intervals": episode["phase_annotations"]["intervals"],
                "warnings": episode["warnings"],
            }
            for name, episode in report["episodes"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
