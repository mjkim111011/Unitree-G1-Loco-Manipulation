#!/usr/bin/env python3
"""Phase 8: safety and success table of Arena-WBC runs recorded with ``eval_bc1_approach.py --safety_metrics`` (CPU only).

Per run (label = output directory), the existing success definitions are reported unchanged next to the new safety
metrics:
  BC-1 gate success          outcome == handoff_reached (the evaluator's existing BC-1 success)
  table contact              any of the 22 arm/hand links > 1 N against the table top / pedestal
  peak force, impulse        largest substep force; impulse = sum over substeps of force x 5 ms
  clearance                  minimum signed distance of the collision geometry to the table (finger, wrist/palm, forearm)
  commanded clearance        same for the open hand placed at the commanded wrist pose (does the target itself collide?)
  collision-free BC-1        gate success and no table contact during alignment + approach
  BC-2 success               lift_hold_success after the gate
  collision-free E2E         navigation handoff without table contact, collision-free BC-1, BC-2 success
                             (strict: also no table contact during BC-2)
Paired tests (same seed/episode): exact McNemar for binary outcomes, Wilcoxon signed-rank for force/clearance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def describe(values) -> dict | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"median": round(float(np.median(a)), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "n": len(a)}


def load(directory: Path) -> list[dict]:
    return json.loads((directory / "summary.json").read_text())["episodes"]


def legacy_safety(e: dict) -> tuple[dict, dict]:
    """Phase 6/7 runs (``--table_contact_sensors`` only): contact and peak force per phase, no impulse / clearance."""
    contact = e.get("table_contact") or {}
    safety = {}
    for phase in ("nav", "bc1", "bc2"):
        if phase in contact:
            safety[phase] = {"contact": contact[phase]["max_N"] > 1.0, "max_N": contact[phase]["max_N"],
                             "first_over_1N_step": contact[phase]["first_over_1N_step"],
                             "steps_over_1N": contact[phase]["steps_over_1N"], "contact_links": [contact[phase]["link"]]}
    nav = e.get("nav") or {}
    nav_ok = nav.get("handoff", nav.get("arrived")) if nav else None
    gate = e.get("outcome") == "handoff_reached"
    bc2 = e.get("bc2")
    bc2_ok = None if bc2 is None else bc2.get("outcome") == "lift_hold_success"
    nav_free = None if "nav" not in safety else not safety["nav"]["contact"]
    bc1_free = None if "bc1" not in safety else not safety["bc1"]["contact"]
    flags = {"nav_handoff": nav_ok, "nav_contact_free": nav_free, "approach_contact_free": bc1_free, "bc1_gate": gate,
             "bc1_collision_free_success": bool(gate and bc1_free), "bc2_success": bc2_ok,
             "bc2_contact_free": None if "bc2" not in safety else not safety["bc2"]["contact"],
             "e2e": bool((nav_ok is None or nav_ok) and gate and bc2_ok)}
    flags["cf_e2e"] = bool(flags["e2e"] and nav_free is not False and bc1_free)
    flags["cf_e2e_strict"] = bool(flags["cf_e2e"] and flags["bc2_contact_free"] is not False)
    return safety, flags


def episode_row(e: dict) -> dict:
    safety, flags = e.get("safety") or {}, e.get("safety_flags") or {}
    if not safety and e.get("table_contact"):
        safety, flags = legacy_safety(e)
    bc1, align, nav, bc2 = safety.get("bc1") or {}, safety.get("align"), safety.get("nav"), safety.get("bc2")
    row = {
        "episode": e["episode"],
        "outcome": e["outcome"],
        "nav_handoff": flags.get("nav_handoff"),
        "nav_contact": None if nav is None else nav["contact"],
        "nav_max_N": None if nav is None else nav["max_N"],
        "align_contact": None if align is None else align["contact"],
        "align_max_N": None if align is None else align["max_N"],
        "align_min_finger_m": None if align is None else align["min_clearance_finger_m"],
        "align_cmd_min_m": None if align is None else align.get("commanded_min_clearance_m"),
        "bc1_contact": bc1.get("contact"),
        "approach_contact": None if not bc1 else bool(bc1.get("contact") or (align is not None and align["contact"])),
        "bc1_max_N": bc1.get("max_N"),
        "bc1_impulse_Ns": bc1.get("impulse_Ns"),
        "bc1_event_impulse_Ns": bc1.get("max_contact_event_impulse_Ns"),
        "bc1_first_contact_step": bc1.get("first_over_1N_step"),
        "bc1_contact_steps": bc1.get("steps_over_1N"),
        "bc1_contact_links": bc1.get("contact_links"),
        "bc1_min_finger_m": bc1.get("min_clearance_finger_m"),
        "bc1_min_finger_step": bc1.get("min_clearance_finger_step"),
        "bc1_min_finger_link": bc1.get("min_clearance_finger_link"),
        "bc1_min_wrist_palm_m": bc1.get("min_clearance_wrist_palm_m"),
        "bc1_min_forearm_m": bc1.get("min_clearance_forearm_m"),
        "bc1_cmd_min_m": bc1.get("commanded_min_clearance_m"),
        "bc1_cmd_min_step": bc1.get("commanded_min_clearance_step"),
        "bc1_cmd_steps_below_0": bc1.get("commanded_steps_below_0"),
        "bc1_gate": flags.get("bc1_gate"),
        "bc1_cf_success": flags.get("bc1_collision_free_success"),
        "bc2_run": e.get("bc2") is not None,
        "bc2_success": flags.get("bc2_success"),
        "bc2_contact": None if bc2 is None else bc2["contact"],
        "bc2_max_N": None if bc2 is None else bc2["max_N"],
        "bc2_min_finger_m": None if bc2 is None else bc2.get("min_clearance_finger_m"),
        "bc2_first_contact_step": None if bc2 is None else bc2.get("first_over_1N_step"),
        "bc2_max_lift_m": None if e.get("bc2") is None else e["bc2"].get("max_lift_m"),
        "e2e": flags.get("e2e"),
        "cf_e2e": flags.get("cf_e2e"),
        "cf_e2e_strict": flags.get("cf_e2e_strict"),
        "gate_geometry": e.get("best_gate_geometry"),
        "align_info": e.get("align"),
        "safe_path": e.get("safe_teacher_path"),
        "handoff_dx_cm": None if not (e.get("nav") or {}).get("handoff_errors_at_handoff") else round(100 * e["nav"]["handoff_errors_at_handoff"][0], 2),
        "start_key": (tuple(e.get("initial_root_xyz") or ()), tuple(e.get("initial_box_xy") or ())),
    }
    return row


def run_summary(rows: list[dict]) -> dict:
    n = len(rows)

    def count(key, condition=None):
        sel = [r for r in rows if r[key] is not None and (condition is None or condition(r))]
        k = sum(bool(r[key]) for r in sel)
        return {"k": k, "n": len(sel), "wilson95": wilson(k, len(sel))}

    out = {
        "episodes": n,
        "nav_handoff": count("nav_handoff"),
        "nav_contact": count("nav_contact"),
        "align_contact": count("align_contact"),
        "bc1_contact": count("bc1_contact"),
        "approach_contact": count("approach_contact"),
        "bc1_gate": count("bc1_gate"),
        "bc1_cf_success": count("bc1_cf_success"),
        "bc2_success_given_gate": count("bc2_success", lambda r: r["bc2_run"]),
        "bc2_contact": count("bc2_contact"),
        "e2e": count("e2e"),
        "cf_e2e": count("cf_e2e"),
        "cf_e2e_strict": count("cf_e2e_strict"),
        "bc1_max_N": describe(r["bc1_max_N"] for r in rows),
        "bc1_max_N_contact_only": describe(r["bc1_max_N"] for r in rows if r["bc1_contact"]),
        "bc1_impulse_Ns": describe(r["bc1_impulse_Ns"] for r in rows),
        "bc1_event_impulse_Ns": describe(r["bc1_event_impulse_Ns"] for r in rows if r["bc1_contact"]),
        "bc1_first_contact_step": describe(r["bc1_first_contact_step"] for r in rows),
        "bc1_contact_steps": describe(r["bc1_contact_steps"] for r in rows if r["bc1_contact"]),
        "bc1_min_finger_m": describe(r["bc1_min_finger_m"] for r in rows),
        "bc1_min_wrist_palm_m": describe(r["bc1_min_wrist_palm_m"] for r in rows),
        "bc1_min_forearm_m": describe(r["bc1_min_forearm_m"] for r in rows),
        "bc1_cmd_min_m": describe(r["bc1_cmd_min_m"] for r in rows),
        "bc1_cmd_steps_below_0": describe(r["bc1_cmd_steps_below_0"] for r in rows),
        "align_max_N": describe(r["align_max_N"] for r in rows),
        "align_min_finger_m": describe(r["align_min_finger_m"] for r in rows),
        "align_cmd_min_m": describe(r["align_cmd_min_m"] for r in rows),
        "bc2_max_N": describe(r["bc2_max_N"] for r in rows),
        "bc2_min_finger_m": describe(r["bc2_min_finger_m"] for r in rows),
        "bc2_max_lift_m": describe(r["bc2_max_lift_m"] for r in rows),
        "contact_links": {},
    }
    for r in rows:
        for link in r["bc1_contact_links"] or []:
            out["contact_links"][link] = out["contact_links"].get(link, 0) + 1
    aligns = [r["align_info"] for r in rows if r["align_info"]]
    if aligns:
        out["align_final_wrist_error_cm"] = describe(max(a["final_wrist_error_cm"].values()) for a in aligns)
        out["align_planned_max_lift_m"] = describe(a["max_lift_m"] for a in aligns)
        out["align_planned_min_clearance_after_m"] = describe(min(a["min_clearance_after_m"].values()) for a in aligns)
    safe = [r["safe_path"] for r in rows if r["safe_path"]]
    if safe:
        out["safe_path_max_lift_m"] = describe(s["max_lift_m"] for s in safe)
        out["safe_path_steps_lifted"] = describe(s["steps_lifted"] for s in safe)
        out["safe_path_min_clearance_before_m"] = describe(min(s["min_clearance_before_m"].values()) for s in safe)
        out["safe_path_min_clearance_after_m"] = describe(min(s["min_clearance_after_m"].values()) for s in safe)
    return out


def paired(a: list[dict], b: list[dict]) -> dict:
    """Same seed, same episode index.  The start states are identical only while the random stream has not diverged (it
    depends on how long the earlier episodes ran), so the exact-pairing count is reported and every binary outcome also
    gets an unpaired Fisher exact test over all episodes of both runs."""
    from scipy.stats import fisher_exact, wilcoxon

    ra, rb = {r["episode"]: r for r in a}, {r["episode"]: r for r in b}
    eps = sorted(set(ra) & set(rb))
    out = {"episodes": len(eps), "identical_start_episodes": sum(1 for e in eps if ra[e]["start_key"] == rb[e]["start_key"])}
    for key in ("approach_contact", "bc1_contact", "bc1_gate", "bc1_cf_success", "bc2_success", "bc2_contact", "e2e", "cf_e2e",
                "cf_e2e_strict", "nav_handoff"):
        pairs = [(ra[e][key], rb[e][key]) for e in eps if ra[e][key] is not None and rb[e][key] is not None]
        b_ = sum(1 for x, y in pairs if x and not y)
        c_ = sum(1 for x, y in pairs if y and not x)
        out[key] = {"from": sum(bool(x) for x, _ in pairs), "to": sum(bool(y) for _, y in pairs), "n": len(pairs),
                    "discordant": [b_, c_], "p": round(mcnemar(b_, c_), 4)}
        xa = [bool(r[key]) for r in a if r[key] is not None]
        xb = [bool(r[key]) for r in b if r[key] is not None]
        if xa and xb:
            table = [[sum(xa), len(xa) - sum(xa)], [sum(xb), len(xb) - sum(xb)]]
            out[key]["unpaired"] = {"from": f"{sum(xa)}/{len(xa)}", "to": f"{sum(xb)}/{len(xb)}",
                                    "fisher_p": float(f"{fisher_exact(table)[1]:.3g}")}
    for key in ("bc1_max_N", "bc1_impulse_Ns", "bc1_min_finger_m", "bc2_min_finger_m", "bc2_max_lift_m"):
        pairs = [(ra[e][key], rb[e][key]) for e in eps if ra[e][key] is not None and rb[e][key] is not None]
        if len(pairs) >= 5 and any(x != y for x, y in pairs):
            stat = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
            out[key] = {"median_from": round(float(np.median([x for x, _ in pairs])), 4),
                        "median_to": round(float(np.median([y for _, y in pairs])), 4), "p": float(f"{stat.pvalue:.3g}")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--pair", action="append", default=[], metavar="LABEL_A:LABEL_B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"runs": {}, "paired": {}}
    rows = {}
    for item in args.run:
        label, directory = item.split("=", 1)
        rows[label] = [episode_row(e) for e in load(Path(directory))]
        report["runs"][label] = {"directory": directory, "summary": run_summary(rows[label]), "episodes": rows[label]}
        s = report["runs"][label]["summary"]
        fmt = lambda d: f"{d['k']}/{d['n']}" if d["n"] else "-"  # noqa: E731
        med = lambda d, scale=1.0: "-" if d is None else f"{scale * d['median']:.1f}"  # noqa: E731
        print(f"{label:28s} n={s['episodes']:2d} nav {fmt(s['nav_handoff'])} navC {fmt(s['nav_contact'])} alignC {fmt(s['align_contact'])} "
              f"| approach contact {fmt(s['approach_contact'])} (BC-1 {fmt(s['bc1_contact'])}) maxN {med(s['bc1_max_N'])} imp {med(s['bc1_impulse_Ns'])} "
              f"minFinger {med(s['bc1_min_finger_m'], 100)}cm cmdMin {med(s['bc1_cmd_min_m'], 100)}cm "
              f"| gate {fmt(s['bc1_gate'])} CF-BC1 {fmt(s['bc1_cf_success'])} BC-2 {fmt(s['bc2_success_given_gate'])} "
              f"E2E {fmt(s['e2e'])} CF-E2E {fmt(s['cf_e2e'])} strict {fmt(s['cf_e2e_strict'])}")
    for item in args.pair:
        a, b = item.split(":")
        report["paired"][item] = paired(rows[a], rows[b])
        print(item, json.dumps(report["paired"][item]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("written", args.output)


if __name__ == "__main__":
    main()
