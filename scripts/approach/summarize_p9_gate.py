#!/usr/bin/env python3
"""Phase 9: compare BC-1 -> BC-2 handoff-condition variants against the existing gate on the same episodes (CPU only).

Runs come from run_p9_gate.sh (--reseed_each_episode, same seed and episode ids under every variant, so each episode
starts from the same state; an episode is replayed exactly until a variant changes it, later ones may drift slightly).
Per variant and condition, against the existing gate (G0):
  BC-1 gate, BC-2 success after the gate, strict collision-free end-to-end success (no table contact in navigation /
  BC-1 / BC-2 and the existing E2E success), contacts per phase, BC-1 contacts that began after the step at which G0
  handed over (contact caused by the delay), BC-1 minimum finger clearance, the handoff delay against G0, readiness
  fallbacks and BC-1 timeouts.  "early" = episodes in which G0 handed over at step <= 116 or with a cage wider than
  29 cm (computed from the G0 trace), i.e. the states the readiness condition targets.
Tests: exact McNemar on the paired outcomes and Fisher exact (unpaired).
Usage: summarize_p9_gate.py --eval_root outputs/approach_p9/eval --output outputs/approach_p9/reports/p9_gate_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

TIPS = ("hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def cage_at(trace: Path, step_in_bc1: int) -> float | None:
    if not trace.exists():
        return None
    d = np.load(trace, allow_pickle=True)
    links = [str(n) for n in d["table_link_names"]]
    bc1 = np.flatnonzero(d["phase"] == 4)
    if not len(bc1) or step_in_bc1 >= len(bc1):
        return None
    k = int(bc1[step_in_bc1])
    q = d["box_quat"][k]
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    diff = d["arm_link_pos"][k, [links.index("left_" + t) for t in TIPS]].mean(0) - d["arm_link_pos"][k, [links.index("right_" + t) for t in TIPS]].mean(0)
    return float(-math.sin(yaw) * diff[0] + math.cos(yaw) * diff[1])


def episode_row(e: dict) -> dict:
    s = e.get("safety") or {}
    f = e.get("safety_flags") or {}
    b1 = s.get("bc1") or {}
    return {
        "episode": e["episode"],
        "start": (tuple(e.get("initial_root_xyz") or ()), tuple(e.get("initial_box_xy") or ()), tuple(e.get("start_offset_canonical") or ())),
        "handoff": e.get("handoff_step"),
        "gate": bool(f.get("bc1_gate")),
        "bc2_ok": bool(f.get("bc2_success")),
        "strict": bool(f.get("cf_e2e_strict")),
        "nav_handoff": bool(f.get("nav_handoff")),
        # the part a handoff condition can change: collision-free BC-1 (gate, no table contact) -> BC-2 success without contact
        "cf_bc1_bc2": bool(f.get("bc1_collision_free_success") and f.get("bc2_success") and f.get("bc2_contact_free") is not False),
        "e2e": bool(f.get("e2e")),
        "nav_contact": bool((s.get("nav") or {}).get("contact")),
        "bc1_contact": bool(b1.get("contact")),
        "bc1_first_contact": b1.get("first_over_1N_step"),
        "bc1_max_N": b1.get("max_N"),
        "bc2_contact": bool((s.get("bc2") or {}).get("contact")),
        "bc1_min_finger_m": b1.get("min_clearance_finger_m"),
        "readiness": e.get("readiness"),
        "lift": (e.get("bc2") or {}).get("max_lift_m"),
    }


def largest(root: Path, pattern: str) -> Path | None:
    """The finished run matching the pattern with the most episodes (smoke subsets lose to the full run)."""
    runs = [d for d in root.glob(pattern) if (d / "summary.json").exists()]
    if not runs:
        return None
    return max(runs, key=lambda d: len(json.loads((d / "summary.json").read_text())["episodes"]))


def frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_root", type=Path, default=Path("outputs/approach_p9/eval"))
    parser.add_argument("--conditions", nargs="+", default=["v6far", "win"])
    parser.add_argument("--variants", nargs="+", default=["cage", "fix", "freeze"])
    parser.add_argument("--baseline", default="g0", help="run name part of the reference gate (g0, or g0s for the stress runs)")
    parser.add_argument("--early_step", type=int, default=116)
    parser.add_argument("--cage_threshold_m", type=float, default=0.29)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    md = ["| 조건 | 변형 | n | 같은 navigation 종료 | nav handoff | BC-1 gate | BC-2 / gate | **무접촉 BC-1 → BC-2 성공** | **strict collision-free E2E** | 접촉 nav / BC-1 / BC-2 | "
          "지연 중 시작된 BC-1 접촉 | BC-1 최소 손가락 거리 중앙값 [최소] | handoff 지연 (step) 중앙값 [최대] | fallback | 조기·넓은 cage episode: BC-2 / 무접촉 BC-1→BC-2 | McNemar p (BC-1→BC-2) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---|---:|"]
    for cond in args.conditions:
        g0_dir = largest(args.eval_root, f"{cond}_{args.baseline}_seed4002_ep*")
        if g0_dir is None:
            continue
        g0 = {e["episode"]: episode_row(e) for e in json.loads((g0_dir / "summary.json").read_text())["episodes"]}
        early = set()
        for ep, r in g0.items():
            if r["handoff"] is None:
                continue
            width = cage_at(g0_dir / f"trace_{ep:03d}.npz", r["handoff"])
            r["cage_at_handoff_m"] = width
            if r["handoff"] <= args.early_step or (width is not None and width > args.cage_threshold_m):
                early.add(ep)
        report[cond] = {"g0": str(g0_dir), "early_episodes": sorted(early), "variants": {}}
        for variant in [args.baseline, *args.variants]:
            run_dir = largest(args.eval_root, f"{cond}_{variant}_seed4002_ep*")
            if run_dir is None:
                continue
            rows = {e["episode"]: episode_row(e) for e in json.loads((run_dir / "summary.json").read_text())["episodes"]}
            eps = sorted(set(rows) & set(g0))
            same = sum(1 for ep in eps if rows[ep]["start"] == g0[ep]["start"])
            n = len(eps)
            k = lambda key, sel=eps: sum(1 for ep in sel if rows[ep][key])  # noqa: E731
            gated = [ep for ep in eps if rows[ep]["gate"]]
            delay_contacts = sum(1 for ep in eps if rows[ep]["bc1_contact"] and g0[ep]["handoff"] is not None
                                 and rows[ep]["bc1_first_contact"] is not None and rows[ep]["bc1_first_contact"] > g0[ep]["handoff"])
            mins = np.array([100 * rows[ep]["bc1_min_finger_m"] for ep in eps if rows[ep]["bc1_min_finger_m"] is not None])
            delays = [rows[ep]["handoff"] - g0[ep]["handoff"] for ep in eps if rows[ep]["handoff"] is not None and g0[ep]["handoff"] is not None]
            fallbacks = sum(1 for ep in eps if (rows[ep]["readiness"] or {}).get("fallback"))
            ee = [ep for ep in eps if ep in early]
            b = sum(1 for ep in eps if g0[ep]["strict"] and not rows[ep]["strict"])
            c = sum(1 for ep in eps if rows[ep]["strict"] and not g0[ep]["strict"])
            b2 = sum(1 for ep in eps if g0[ep]["cf_bc1_bc2"] and not rows[ep]["cf_bc1_bc2"])
            c2 = sum(1 for ep in eps if rows[ep]["cf_bc1_bc2"] and not g0[ep]["cf_bc1_bc2"])
            fisher = fisher_exact([[k("strict"), n - k("strict")], [sum(g0[ep]["strict"] for ep in eps), n - sum(g0[ep]["strict"] for ep in eps)]])[1]
            entry = {
                "run": str(run_dir), "episodes": n, "identical_starts": same,
                "bc1_gate": [k("gate"), n], "bc2_success_given_gate": [sum(rows[ep]["bc2_ok"] for ep in gated), len(gated)],
                "strict_cf_e2e": [k("strict"), n], "e2e": [k("e2e"), n], "nav_handoff": [k("nav_handoff"), n], "cf_bc1_bc2": [k("cf_bc1_bc2"), n],
                "contacts": {"nav": k("nav_contact"), "bc1": k("bc1_contact"), "bc2": k("bc2_contact")},
                "bc1_contacts_started_after_g0_handoff": delay_contacts,
                "bc1_min_finger_cm": None if not len(mins) else [round(float(np.median(mins)), 2), round(float(mins.min()), 2)],
                "handoff_delay_steps": None if not delays else [float(np.median(delays)), int(max(delays))],
                "fallbacks": fallbacks,
                "early_subset": {"n": len(ee), "bc2_success": sum(rows[ep]["bc2_ok"] for ep in ee), "strict": sum(rows[ep]["strict"] for ep in ee),
                                 "cf_bc1_bc2": sum(rows[ep]["cf_bc1_bc2"] for ep in ee)},
                "paired_strict_vs_g0": {"discordant_g0_only": b, "discordant_variant_only": c, "mcnemar_p": round(mcnemar(b, c), 4),
                                        "fisher_p": float(f"{fisher:.3g}")},
                "paired_cf_bc1_bc2_vs_g0": {"discordant_g0_only": b2, "discordant_variant_only": c2, "mcnemar_p": round(mcnemar(b2, c2), 4)},
                "per_episode": {ep: {"handoff": rows[ep]["handoff"], "g0_handoff": g0[ep]["handoff"], "bc2_ok": rows[ep]["bc2_ok"],
                                     "strict": rows[ep]["strict"], "g0_strict": g0[ep]["strict"], "readiness": rows[ep]["readiness"],
                                     "g0_cage_m": g0[ep].get("cage_at_handoff_m")} for ep in eps},
            }
            report[cond]["variants"][variant] = entry
            clearance_text = "-" if entry["bc1_min_finger_cm"] is None else f"{entry['bc1_min_finger_cm'][0]:.2f} [{entry['bc1_min_finger_cm'][1]:.2f}] cm"
            md.append(f"| {cond} | {variant.upper()} | {n} | {same}/{n} | {frac(k('nav_handoff'), n)} | {frac(k('gate'), n)} | {frac(*entry['bc2_success_given_gate'])} | "
                      f"**{frac(k('cf_bc1_bc2'), n)}** | **{frac(k('strict'), n)}** | {k('nav_contact')} / {k('bc1_contact')} / {k('bc2_contact')} | {delay_contacts} | "
                      f"{clearance_text} | "
                      f"{'-' if not delays else f'{np.median(delays):.0f} [{max(delays)}]'} | {fallbacks} | "
                      f"{entry['early_subset']['bc2_success']}/{len(ee)} / {entry['early_subset']['cf_bc1_bc2']}/{len(ee)} | "
                      f"{'-' if variant == args.baseline else entry['paired_cf_bc1_bc2_vs_g0']['mcnemar_p']} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
    text = "\n".join(md) + "\n"
    args.output.with_suffix(".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
