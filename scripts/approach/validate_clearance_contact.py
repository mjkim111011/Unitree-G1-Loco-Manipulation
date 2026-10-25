#!/usr/bin/env python3
"""Phase 8: consistency of the geometric clearance model with the PhysX table contact sensors (CPU only).

Over every link and step of the given ``--safety_metrics --nav_trace`` runs:
  - clearance of link-steps with a table force > 1 N (should be ~0: the model's collision surface touches the table)
  - table force of link-steps with clearance > 5 mm (should be 0: no contact reported away from the table)
  - fraction of link-steps with clearance <= 0 that report > 1 N
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    touching, far_force, penetrating, penetrating_with_force, steps, traces = [], 0, 0, 0, 0, 0
    for directory in args.dirs:
        for path in sorted(directory.glob("trace_*.npz")):
            d = np.load(path)
            if "clearance" not in d.files:
                continue
            force, clearance = d["table_force"], d["clearance"]
            traces += 1
            steps += force.size
            touching.extend(clearance[force > 1.0].tolist())
            far_force += int(((clearance > 0.005) & (force > 1.0)).sum())
            penetrating += int((clearance <= 0.0).sum())
            penetrating_with_force += int(((clearance <= 0.0) & (force > 1.0)).sum())
    t = 100.0 * np.asarray(touching)
    report = {
        "traces": traces, "link_steps": steps,
        "contact_link_steps": int(len(t)),
        "clearance_cm_at_contact": {q: round(float(np.percentile(t, q)), 3) for q in (0, 1, 50, 99, 100)} if len(t) else None,
        "contact_link_steps_with_clearance_over_5mm": far_force,
        "link_steps_with_clearance_le_0": penetrating,
        "of_which_force_over_1N": penetrating_with_force,
    }
    print(json.dumps(report, indent=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
