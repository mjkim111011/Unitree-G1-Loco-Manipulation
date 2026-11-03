#!/usr/bin/env python3
"""Build the per-contact-mode capture-point reference from nominal-payload rollouts.

Reads ``evaluate_bc2_rollout.py --com_stabilizer observe`` result files, keeps
only successful carries, and writes the median capture point (yaw frame,
relative to the loaded-feet support origin) for double/left/right support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unitree_rl_lab.utils.teacher.com_stabilizer import build_reference


def samples_from(path: Path, min_carry_step: int):
    data = json.loads(path.read_text())
    if data["evaluator_config"].get("com_stabilizer") != "observe":
        raise ValueError(f"{path} was not recorded with --com_stabilizer observe")
    for episode in data["episode_results"]:
        carry = episode.get("carry") or {}
        if not carry.get("success"):
            continue
        for record in carry.get("trace", []):
            log = record.get("com_stabilizer")
            if log and log["mode"] is not None and record["carry_step"] >= min_carry_step:
                yield log["mode"], tuple(log["capture_xy_m"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="observe-mode result JSON files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_carry_step", type=int, default=0,
                        help="Ignore samples before this CARRY step (e.g. the blend)")
    args = parser.parse_args()
    samples = [s for path in args.results for s in samples_from(path, args.min_carry_step)]
    reference = build_reference(samples)
    reference["sources"] = [str(p.resolve()) for p in args.results]
    reference["min_carry_step"] = args.min_carry_step
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: reference[k] for k in ("reference_xy", "sample_counts")}, indent=2))


if __name__ == "__main__":
    main()
