#!/usr/bin/env python3
"""Phase-8 figure: pre-grasp margin, navigation end position and BC-2 handoff timing (CPU only, reads summary.json files).

Panels:
  1  Teacher demos: minimum finger-table clearance over the recorded approach (until close + 5 steps), BC-1 v3 demos
     (pre-grasp margin 3.0/3.5 cm, outputs/approach_p8/collect) vs BC-1 v3.1 demos (7 cm, collect_v31), same seeds
  2  navigation-phase minimum finger clearance against the PPO handoff position dx (all phase-8 PPO runs with safety
     metrics); episodes with a navigation table contact marked
  3  BC-2 v3.1 success against the BC-1 gate step (evaluation runs *_v31)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P8 = Path("outputs/approach_p8")


def episodes(pattern):
    for path in sorted(glob.glob(str(P8 / pattern))):
        if "stopped" in path:
            continue
        yield from json.loads(Path(path).read_text())["episodes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=P8 / "figures/fig_p8_margin_nav.png")
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    bins = np.linspace(-0.5, 4.5, 41)
    for pattern, label, color in (("collect/*/summary.json", "BC-1 v3 demos (pre-grasp margin 3.0/3.5 cm)", "#C0504D"),
                                  ("collect_v31/*/summary.json", "BC-1 v3.1 demos (7 cm)", "#2E9D5B")):
        rows = [e["safety"]["bc1"] for e in episodes(pattern)]
        clearance = 100 * np.array([b["min_clearance_finger_m"] for b in rows])
        contact = sum(bool(b["contact"]) for b in rows)
        ax.hist(clearance, bins=bins, alpha=0.6, color=color, label=f"{label}: contact {contact}/{len(rows)}")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("minimum finger-table clearance in the Teacher approach [cm]")
    ax.set_ylabel("demos")
    ax.set_title("Teacher demos, same seeds (240 each)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pts = []
    for pattern in ("eval/*/summary.json", "collect*/ppo_nav_*/summary.json", "cause/*_nav_seed8102/summary.json", "margin_check/**/summary.json"):
        for e in episodes(pattern):
            nav, safety = e.get("nav") or {}, e.get("safety") or {}
            if "nav" not in safety or nav.get("controller") != "ppo":
                continue
            dx = (nav.get("handoff_errors_at_handoff") or nav.get("final_handoff_errors") or [None])[0]
            if dx is not None:
                pts.append((100 * dx, 100 * safety["nav"]["min_clearance_finger_m"], bool(safety["nav"]["contact"])))
    pts = np.array(sorted(set(pts)), dtype=float)
    free, hit = pts[pts[:, 2] == 0], pts[pts[:, 2] == 1]
    ax.scatter(free[:, 0], free[:, 1], s=8, alpha=0.5, color="#4F81BD", label=f"no contact ({len(free)})")
    ax.scatter(hit[:, 0], hit[:, 1], s=30, marker="x", color="#C0504D", label=f"navigation table contact ({len(hit)})")
    ax.axhline(0, color="k", lw=1)
    ax.axvspan(-4.5, 1.5, color="0.85", alpha=0.5, label="existing handoff window")
    ax.axvline(-0.5, color="#2E9D5B", ls="--", label="proposed front edge (-0.5 cm)")
    ax.set_xlabel("PPO handoff position dx relative to the BC-1 start [cm]")
    ax.set_ylabel("navigation min finger clearance [cm]")
    ax.set_title("navigation end position and table clearance (PPO v6/v7)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    gates = [(e["handoff_step"], (e.get("bc2") or {}).get("outcome") == "lift_hold_success") for e in episodes("eval/*_v31/summary.json")
             if e.get("handoff_step") is not None and e.get("bc2") is not None]
    edges = [(0, 109, "<=109"), (110, 116, "110-116"), (117, 120, "117-120"), (121, 999, ">=121")]
    xs = np.arange(len(edges))
    rates, ns, ks = [], [], []
    for lo, hi, _ in edges:
        sel = [ok for g, ok in gates if lo <= g <= hi]
        ns.append(len(sel))
        ks.append(sum(sel))
        rates.append(sum(sel) / len(sel) if sel else np.nan)
    ax.bar(xs, rates, color="#2E9D5B")
    for x, k, n in zip(xs, ks, ns):
        ax.text(x, 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(xs, [e[2] for e in edges])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("BC-1 gate step (handoff to BC-2)")
    ax.set_ylabel("BC-2 v3.1 lift-and-hold success")
    ax.set_title(f"BC-2 v3.1 success by handoff time (all v3.1 evaluation runs, n={len(gates)})", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print("written", args.output)


if __name__ == "__main__":
    main()
