#!/usr/bin/env python3
"""Phase 10: the cage readiness gate (existing gate AND cage width <= 0.29 m, safe stop after 30 steps) against the stored
phase-8 baseline (existing gate, 149/160) on the 10 phase-8 conditions, episode by episode (CPU only).

The cage runs repeat the phase-8 runs exactly (same evaluator settings, seeds, episode order, no reseeding) plus the
readiness options, so every episode is replayed exactly until the first episode in which the new gate changes the
behaviour; after that, an episode keeps its initial robot pose (per-episode random stream) but its Box position (+-2 mm,
shared random stream) and its navigation (simulator history) can drift.  Each pair is therefore classified as
  exact     identical initial pose, Box position and navigation end state
  drifted   identical initial pose, but the Box position or the navigation end state differ
and every outcome change is attributed to "gate acted" (the cage condition delayed or stopped this episode) or to drift.
Per episode: navigation handoff, BC-1 gate, BC-2 lift-hold, table contact per phase (> 1 N), strict collision-free E2E,
the wait (first gate opening, handoff, cage width over the wait, minimum finger-table clearance and peak table force during
the wait) and the BC-2 start state (cage width, left/right fingertip gap to the Box faces, fingertip and base speed, finger
clearance).  Tests: exact McNemar on the paired outcomes, per condition and pooled.
Usage: summarize_p10_cage_gate.py --output outputs/approach_p10/reports/p10_cage_gate.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TIPS = ("hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")
FINGERS = ("hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_index_0_link", "hand_index_1_link",
           "hand_middle_0_link", "hand_middle_1_link")
P8 = Path("outputs/approach_p8/eval")
P10 = Path("outputs/approach_p10/eval")
CONDITIONS = [
    ("nominal (v7 nav)", "nominal_v3_seed4002"), ("nominal (v6 nav)", "nominal_v6nav_v3_seed4002"), ("cause starts", "cause_seed8102_v3"),
    ("reset start", "reset_v3_seed4007"), ("box +1 cm", "box+1cm_v3_seed4003"), ("box +2 cm", "box+2cm_v3_seed4003"),
    ("box +3 cm", "box+3cm_v3_seed4003"), ("box -2 cm (toward)", "toward2cm_v3_seed4004"), ("box left 2 cm", "lat+2cm_v3_seed4005"),
    ("box right 2 cm", "lat-2cm_v3_seed4006"),
]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def outcomes(e: dict) -> dict:
    f = e.get("safety_flags") or {}
    s = e.get("safety") or {}
    nav = e.get("nav")
    return {
        "nav_handoff": None if nav is None else bool(f.get("nav_handoff")),
        "bc1_gate": bool(f.get("bc1_gate")),
        "bc2_lift_hold": bool(f.get("bc2_success")),
        "contact_nav": bool((s.get("nav") or {}).get("contact")),
        "contact_bc1": bool((s.get("bc1") or {}).get("contact")),
        "contact_bc2": bool((s.get("bc2") or {}).get("contact")),
        "bc1_max_N": (s.get("bc1") or {}).get("max_N"),
        "bc2_max_N": (s.get("bc2") or {}).get("max_N"),
        "strict": bool(f.get("cf_e2e_strict")),
        "e2e": bool(f.get("e2e")),
        "outcome": e.get("outcome"),
        "bc2_outcome": (e.get("bc2") or {}).get("outcome"),
        "lift_cm": None if not e.get("bc2") else round(100 * e["bc2"].get("max_lift_m", 0.0), 1),
    }


def trace_state(trace: Path, first: int | None, handoff: int | None) -> dict:
    """Cage width over the wait, clearance and table force during the wait, and the BC-2 start state."""
    if not trace.exists() or handoff is None:
        return {}
    d = np.load(trace, allow_pickle=True)
    links = [str(n) for n in d["table_link_names"]]
    bc1 = np.flatnonzero(d["phase"] == 4)
    if not len(bc1) or handoff >= len(bc1):
        return {}
    li, ri = [links.index("left_" + t) for t in TIPS], [links.index("right_" + t) for t in TIPS]
    fingers = [links.index(s + "_" + f) for s in ("left", "right") for f in FINGERS]

    def cage(k):
        q = d["box_quat"][k]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        diff = d["arm_link_pos"][k, li].mean(0) - d["arm_link_pos"][k, ri].mean(0)
        return 100 * (-math.sin(yaw) * diff[0] + math.cos(yaw) * diff[1])

    def tips_box(k, idx):
        q = d["box_quat"][k]
        w, x, y, z = q
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return (d["arm_link_pos"][k, idx].mean(0) - d["box_pos"][k]) @ r

    k_h = int(bc1[handoff])
    out = {}
    if first is not None and first < handoff:
        span = [int(bc1[s]) for s in range(first, handoff + 1)]
        widths = [cage(k) for k in span]
        out["wait"] = {"steps": handoff - first, "cage_start_cm": round(widths[0], 2), "cage_min_cm": round(min(widths), 2),
                       "cage_end_cm": round(widths[-1], 2),
                       "min_finger_clearance_cm": round(100 * float(d["clearance"][span][:, fingers].min()), 2),
                       "max_table_force_N": round(float(d["table_force"][span].max()), 2)}
    lo = int(bc1[max(0, handoff - 2)])
    gap = {}
    for side, idx in (("left", li), ("right", ri)):
        b = tips_box(k_h, idx)
        gap[side] = round(100 * float(np.linalg.norm(np.clip(np.abs(b) - 0.10, 0.0, None))), 2)
    speed = max(float(np.linalg.norm(d["arm_link_pos"][k_h, idx].mean(0) - d["arm_link_pos"][lo, idx].mean(0))) / (max(1, k_h - lo) * 0.02) for idx in (li, ri))
    out["bc2_start"] = {"cage_cm": round(cage(k_h), 2), "left_tips_gap_cm": gap["left"], "right_tips_gap_cm": gap["right"],
                        "tips_speed_max_cms": round(100 * speed, 1), "finger_clearance_cm": round(100 * float(d["clearance"][k_h, fingers].min()), 2),
                        "base_speed_cms": round(100 * float(np.linalg.norm(d["lin_vel_b"][k_h, :2])), 1)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suffix", default="_v31cage")
    parser.add_argument("--new_root", type=Path, default=P10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"conditions": {}, "pooled": {}}
    keys = ("nav_handoff", "bc1_gate", "bc2_lift_hold", "contact_nav", "contact_bc1", "contact_bc2", "strict", "e2e")
    pooled = {k: [0, 0, 0, 0] for k in keys}  # base true, new true, base-only, new-only
    pooled_n = 0
    transitions = []
    rows_md = []
    for label, stem in CONDITIONS:
        base_dir, new_dir = P8 / f"{stem}_v31", args.new_root / f"{stem}{args.suffix}"
        if not (base_dir / "summary.json").exists() or not (new_dir / "summary.json").exists():
            continue
        base = {e["episode"]: e for e in json.loads((base_dir / "summary.json").read_text())["episodes"]}
        new = {e["episode"]: e for e in json.loads((new_dir / "summary.json").read_text())["episodes"]}
        eps = sorted(set(base) & set(new))
        cond = {"n": len(eps), "pairing": {"exact": 0, "drifted": 0}, "gate_acted": 0, "timeouts": 0, "episodes": {}}
        stats = {k: [0, 0, 0, 0] for k in keys}
        for ep in eps:
            b, n = base[ep], new[ep]
            ob, on = outcomes(b), outcomes(n)
            same_pose = b.get("initial_root_xyz") == n.get("initial_root_xyz")
            same_box = b.get("initial_box_xy") == n.get("initial_box_xy")
            same_nav = b.get("start_offset_canonical") == n.get("start_offset_canonical")
            pairing = "exact" if same_pose and same_box and same_nav else "drifted"
            cond["pairing"][pairing] += 1
            r = n.get("readiness") or {}
            first, handoff = r.get("gate_first_open_step"), r.get("handoff_step")
            acted = bool(r.get("timeout_abort")) or (r.get("waited_steps") or 0) > 0
            cond["gate_acted"] += acted
            cond["timeouts"] += bool(r.get("timeout_abort"))
            state = trace_state(new_dir / f"trace_{ep:03d}.npz", first, handoff if handoff is not None else (first + (r.get("waited_steps") or 0) if first is not None else None))
            base_state = trace_state(base_dir / f"trace_{ep:03d}.npz", None, b.get("handoff_step"))
            for k in keys:
                if ob[k] is None or on[k] is None:
                    continue
                stats[k][0] += ob[k]
                stats[k][1] += on[k]
                stats[k][2] += ob[k] and not on[k]
                stats[k][3] += on[k] and not ob[k]
            entry = {"pairing": pairing, "same_pose": same_pose, "same_box": same_box, "same_nav_end": same_nav, "gate_acted": acted,
                     "base": ob | {"handoff_step": b.get("handoff_step"), "bc2_start": base_state.get("bc2_start")},
                     "new": on | {"readiness": r, **state}}
            cond["episodes"][ep] = entry
            for k, name in (("strict", "strict collision-free E2E"), ("bc2_lift_hold", "BC-2 lift-hold")):
                if ob[k] != on[k] and ob[k] is not None and on[k] is not None:
                    transitions.append({"condition": label, "episode": ep, "metric": name, "change": "S->F" if ob[k] else "F->S",
                                        "attribution": "gate acted" if acted else "drift (gate did not act)", "pairing": pairing,
                                        "base": {x: ob[x] for x in ("nav_handoff", "bc1_gate", "bc2_lift_hold", "contact_nav", "contact_bc1", "contact_bc2", "lift_cm", "outcome")},
                                        "new": {x: on[x] for x in ("nav_handoff", "bc1_gate", "bc2_lift_hold", "contact_nav", "contact_bc1", "contact_bc2", "lift_cm", "outcome")},
                                        "readiness": r, "wait": state.get("wait"), "bc2_start_new": state.get("bc2_start"), "bc2_start_base": base_state.get("bc2_start")})
        cond["metrics"] = {k: {"baseline": [stats[k][0], len(eps)], "cage": [stats[k][1], len(eps)], "baseline_only": stats[k][2],
                               "cage_only": stats[k][3], "mcnemar_p": round(mcnemar(stats[k][2], stats[k][3]), 4)} for k in keys}
        report["conditions"][label] = cond
        for k in keys:
            for i in range(4):
                pooled[k][i] += stats[k][i]
        pooled_n += len(eps)
        m = cond["metrics"]
        f = lambda k: f"{m[k]['baseline'][0]} → {m[k]['cage'][0]}"  # noqa: E731
        rows_md.append(f"| {label} | {len(eps)} | {cond['pairing']['exact']} | {cond['gate_acted']} ({cond['timeouts']}) | {f('nav_handoff') if m['nav_handoff']['baseline'][1] and any(outcomes(base[e])['nav_handoff'] is not None for e in eps) else '–'} | "
                       f"{f('bc1_gate')} | {f('bc2_lift_hold')} | {m['contact_nav']['baseline'][0]}/{m['contact_bc1']['baseline'][0]}/{m['contact_bc2']['baseline'][0]} → "
                       f"{m['contact_nav']['cage'][0]}/{m['contact_bc1']['cage'][0]}/{m['contact_bc2']['cage'][0]} | **{f('strict')}** | "
                       f"{m['strict']['baseline_only']} / {m['strict']['cage_only']} | {m['strict']['mcnemar_p']} |")
    report["pooled"] = {"n": pooled_n, **{k: {"baseline": pooled[k][0], "cage": pooled[k][1], "baseline_only": pooled[k][2], "cage_only": pooled[k][3],
                                             "mcnemar_p": round(mcnemar(pooled[k][2], pooled[k][3]), 4)} for k in keys}}
    report["transitions"] = transitions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
    p = report["pooled"]
    md = ["| 조건 | n | 정확히 같은 episode | 새 gate가 작동 (안전 정지) | nav handoff | BC-1 gate | BC-2 lift-hold | 접촉 nav/BC-1/BC-2 | **strict collision-free E2E** | 기존만 성공 / 새 gate만 성공 | McNemar p |",
          "|---|---:|---:|---:|---|---|---|---|---|---|---:|", *rows_md,
          f"| **전체** | {pooled_n} | {sum(c['pairing']['exact'] for c in report['conditions'].values())} | "
          f"{sum(c['gate_acted'] for c in report['conditions'].values())} ({sum(c['timeouts'] for c in report['conditions'].values())}) | "
          f"{p['nav_handoff']['baseline']} → {p['nav_handoff']['cage']} | {p['bc1_gate']['baseline']} → {p['bc1_gate']['cage']} | "
          f"{p['bc2_lift_hold']['baseline']} → {p['bc2_lift_hold']['cage']} | {p['contact_nav']['baseline']}/{p['contact_bc1']['baseline']}/{p['contact_bc2']['baseline']} → "
          f"{p['contact_nav']['cage']}/{p['contact_bc1']['cage']}/{p['contact_bc2']['cage']} | **{p['strict']['baseline']} → {p['strict']['cage']}** | "
          f"{p['strict']['baseline_only']} / {p['strict']['cage_only']} | {p['strict']['mcnemar_p']} |"]
    text = "\n".join(md) + "\n"
    args.output.with_suffix(".md").write_text(text)
    print(text)
    for t in transitions:
        print(f"{t['condition']} ep {t['episode']} {t['metric']}: {t['change']} ({t['attribution']}, {t['pairing']}) readiness {t['readiness']} wait {t['wait']}")


if __name__ == "__main__":
    main()
