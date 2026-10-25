#!/usr/bin/env python3
"""Phase-8 per-episode safety table of eval_bc1_approach.py --safety_metrics runs (CPU only, reads summary.json).

One row per episode: start offset to the BC-1 canonical start, Box xy, gate step, BC-1 table contact (max force, first
step over 1 N, links) and minimum finger clearance (value, step), what followed the gate (BC-2 policy or the Teacher
continuation: outcome, lifted height, contact, minimum clearance).  The footer counts the contacts by when they began
(BC-1 step < 20 = start-up strike, otherwise the pre-grasp; BC-2 step < 5 = carried over from the handoff state).
Usage: summarize_p8_episodes.py <run dir> [<run dir> ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(e: dict) -> dict:
    safety = e.get("safety") or {}
    b1, b2 = safety.get("bc1") or {}, safety.get("bc2")
    follow = e.get("bc2") or {}
    return {
        "episode": e["episode"],
        "start_dx_dy_dyaw": e.get("start_offset_canonical"),
        "box_xy": e.get("initial_box_xy"),
        "gate_step": e.get("handoff_step"),
        "bc1_max_N": b1.get("max_N"),
        "bc1_first_contact_step": b1.get("first_over_1N_step"),
        "bc1_contact_links": b1.get("contact_links"),
        "bc1_impulse_Ns": b1.get("impulse_Ns"),
        "bc1_min_finger_m": b1.get("min_clearance_finger_m"),
        "bc1_min_finger_step": b1.get("min_clearance_finger_step"),
        "after_gate": follow.get("outcome"),
        "after_gate_lift_m": follow.get("max_lift_m"),
        "bc2_max_N": None if b2 is None else b2.get("max_N"),
        "bc2_first_contact_step": None if b2 is None else b2.get("first_over_1N_step"),
        "bc2_contact_links": None if b2 is None else b2.get("contact_links"),
        "bc2_min_finger_m": None if b2 is None else b2.get("min_clearance_finger_m"),
        "bc2_min_finger_step": None if b2 is None else b2.get("min_clearance_finger_step"),
        "lift_plan_m": ((follow.get("continuation_plan") or {}).get("lift_m")),
        "flags": e.get("safety_flags"),
    }


def fmt(v, scale=1.0, digits=1):
    return "-" if v is None else f"{scale * v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    out = {}
    for run in args.runs:
        episodes = json.loads((run / "summary.json").read_text())["episodes"]
        rows = [row(e) for e in episodes]
        out[str(run)] = rows
        print(f"== {run} ({len(rows)} episodes)")
        print(" ep | start dx dy cm  | box x y cm   | gate | BC-1 N  step  min cm @step | after gate        lift | BC-2 N  step  min cm")
        for r in rows:
            dx, dy, _ = r["start_dx_dy_dyaw"] or (None, None, None)
            bx, by = r["box_xy"] or (None, None)
            print(f" {r['episode']:2d} | {fmt(dx, 100):>6} {fmt(dy, 100):>6} | {fmt(bx and bx - 0.62, 100):>5} {fmt(by, 100):>5} | "
                  f"{r['gate_step'] if r['gate_step'] is not None else '-':>4} | {fmt(r['bc1_max_N'], 1, 0):>5} {r['bc1_first_contact_step'] if r['bc1_first_contact_step'] is not None else '-':>4} "
                  f"{fmt(r['bc1_min_finger_m'], 100, 2):>6} @{r['bc1_min_finger_step']} | {str(r['after_gate']):<17} {fmt(r['after_gate_lift_m'], 100):>4} | "
                  f"{fmt(r['bc2_max_N'], 1, 0):>4} {r['bc2_first_contact_step'] if r['bc2_first_contact_step'] is not None else '-':>4} {fmt(r['bc2_min_finger_m'], 100, 2):>6}")
        n = len(rows)
        bc1_contact = [r for r in rows if r["bc1_max_N"]]
        early = sum(1 for r in bc1_contact if r["bc1_first_contact_step"] is not None and r["bc1_first_contact_step"] < 20)
        bc2_rows = [r for r in rows if r["bc2_max_N"] is not None]
        bc2_contact = [r for r in bc2_rows if r["bc2_max_N"]]
        carried = sum(1 for r in bc2_contact if r["bc2_first_contact_step"] is not None and r["bc2_first_contact_step"] < 5)
        gate = sum(1 for r in rows if r["gate_step"] is not None)
        success = sum(1 for r in rows if r["after_gate"] == "lift_hold_success")
        strict = sum(1 for r in rows if (r["flags"] or {}).get("cf_e2e_strict"))
        print(f" gate {gate}/{n} | BC-1 contact {len(bc1_contact)}/{n} (start-up strike {early}, pre-grasp {len(bc1_contact) - early}) | "
              f"after-gate success {success}/{gate} | BC-2-phase contact {len(bc2_contact)}/{len(bc2_rows)} (from the handoff state {carried}) | "
              f"contact-free in every phase + success {strict}/{n}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
