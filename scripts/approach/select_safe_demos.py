#!/usr/bin/env python3
"""Phase 8 Method C: pick the collision-free Teacher demos of ``--mode teacher_collect --safety_metrics`` runs (CPU only).

A recorded demo (``<run>.hdf5`` next to the run directory) is accepted only when its episode in ``<run>/summary.json``
  - reached the BC-1 gate before the Teacher's close onset (demo success, the existing definition),
  - had no table contact (> 1 N on any of the 22 arm/hand links) during the alignment and the approach,
  - kept at least --min_clearance_m between the arm/hand collision geometry and the table during the approach,
  - had no table contact during the navigation pre-phase (start states with a nav-phase contact are not used).
Demos with any remaining contact are never used, so BC-1 v3 is not trained on colliding demonstrations.

--scope recorded  (default) contact / clearance over the whole recorded approach (until 5 steps after the Teacher's
                  close onset; includes Teacher steps after the BC-1 gate that are not trained)
--scope trained   only the steps that become BC-1 training samples (phase_bc approach labels: before the first close /
                  Box contact / lift minus the grasp lead): no table contact before the segment end (summary
                  ``first_over_1N_step``), and the open-hand clearance of the recorded wrist poses over the segment
                  (hand_clearance.OpenHandGeometry) >= --min_clearance_m
Writes the accepted demo ids (``<hdf5 path>::demo_<k>``) and the per-run counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, type=Path, help="teacher_collect run directories (HDF5 = <dir>.hdf5)")
    parser.add_argument("--min_clearance_m", type=float, default=0.005)
    parser.add_argument("--scope", choices=("recorded", "trained"), default="recorded")
    parser.add_argument("--phase", choices=("bc1", "bc2"), default="bc1",
                        help="bc2: Teacher grasp/lift continuations recorded after a BC-1 gate (--then_teacher --collect_output): "
                             "accepted when the continuation succeeded (12 cm lift held 2 s), and neither the BC-1 approach nor the "
                             "continuation touched the table, with --min_clearance_m during the continuation")
    parser.add_argument("--grasp_lead_steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted, report = [], {}
    if args.scope == "trained":
        import sys

        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from hand_clearance import OpenHandGeometry, table_boxes
        from unitree_rl_lab.utils.teacher import phase_bc as pbc

        geometry = OpenHandGeometry()
        inference = pbc.PhaseInferenceConfig(hand_close_threshold=0.5, grasp_lead_steps=args.grasp_lead_steps)

        def pose_matrices(poses):
            w, x, y, z = (poses[:, i] for i in range(3, 7))
            m = np.zeros((len(poses), 4, 4))
            m[:, 0, 0], m[:, 0, 1], m[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
            m[:, 1, 0], m[:, 1, 1], m[:, 1, 2] = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
            m[:, 2, 0], m[:, 2, 1], m[:, 2, 2] = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
            m[:, :3, 3], m[:, 3, 3] = poses[:, :3], 1.0
            return m
    for run in args.run:
        hdf5 = run.with_suffix(".hdf5") if run.suffix != ".hdf5" else run
        hdf5 = Path(str(run) + ".hdf5") if not hdf5.exists() else hdf5
        summaries = {e["episode"]: e for e in json.loads((run / "summary.json").read_text())["episodes"]}
        counts = {"recorded": 0, "gate": 0, "contact_free": 0, "clearance_ok": 0, "nav_contact_free": 0, "accepted": 0}
        with h5py.File(hdf5, "r") as stream:
            for name in sorted(stream["data"], key=lambda n: int(n.split("_")[-1])):
                group = stream["data"][name]
                episode = json.loads(group.attrs["metadata_json"])["episode"]
                e = summaries[episode]
                flags, safety = e.get("safety_flags") or {}, e.get("safety") or {}
                counts["recorded"] += 1
                gate = bool(group.attrs.get("success", False))
                free = bool(flags.get("approach_contact_free"))
                approach = [safety[p] for p in ("align", "bc1") if p in safety]
                clearance = min(min(v[f"min_clearance_{g}_m"] for g in ("finger", "wrist_palm", "forearm")) for v in approach) if approach else -1.0
                if args.phase == "bc2":
                    bc2 = safety.get("bc2")
                    free = bool(free and bc2 is not None and not bc2["contact"])
                    clearance = min(bc2[f"min_clearance_{g}_m"] for g in ("finger", "wrist_palm", "forearm")) if bc2 else -1.0
                if args.scope == "trained" and "bc1" in safety:
                    labels = pbc.infer_bc_skill_labels(group, config=inference)
                    segment = int(np.sum(labels == int(pbc.SkillPhase.APPROACH)))
                    first = safety["bc1"]["first_over_1N_step"]
                    free = (safety.get("align") is None or not safety["align"]["contact"]) and (first is None or first >= segment)
                    centers, half = table_boxes((0.62, 0.0, 0.70), (0.62, 0.0, 0.33))
                    hands = [geometry.clearance(side, pose_matrices(group[f"{side}_wrist_pose"][:segment].astype(np.float64)), centers, half)[0].min()
                             for side in ("left", "right")]
                    clearance = float(min(hands))
                    counts.setdefault("segment_steps", []).append(segment)
                nav_free = flags.get("nav_contact_free") is not False
                counts["gate"] += gate
                counts["contact_free"] += gate and free
                counts["clearance_ok"] += gate and free and clearance >= args.min_clearance_m
                counts["nav_contact_free"] += nav_free
                if gate and free and clearance >= args.min_clearance_m and nav_free:
                    counts["accepted"] += 1
                    accepted.append(f"{hdf5.resolve()}::{name}")
        report[str(run)] = counts
        print(run.name, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for counts in report.values():
        if "segment_steps" in counts:
            counts["segment_steps"] = [min(counts["segment_steps"]), max(counts["segment_steps"])]
    args.output.write_text(json.dumps({"rule": {"min_clearance_m": args.min_clearance_m, "contact_threshold_N": 1.0, "scope": args.scope,
                                                "phase": args.phase},
                                       "runs": report, "accepted": accepted}, indent=1) + "\n")
    print(f"accepted {len(accepted)} demos -> {args.output}")


if __name__ == "__main__":
    main()
