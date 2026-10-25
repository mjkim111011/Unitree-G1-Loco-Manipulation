#!/usr/bin/env python3
"""Phase-9 figure: handoff-condition variants against the existing gate (CPU only; summarize_p9_gate.py outputs + traces).

Panels:
  1  contact-free BC-1 -> BC-2 success per variant in the three experiments (enriched PPO v6 far start, deployment PPO v7
     window, stress = existing gate loosened to 0.21 m), with Wilson 95 % intervals
  2  BC-1 minimum finger-table clearance per variant (median and minimum) in the same experiments
  3  stress experiment: cage width at the handoff per episode and variant, BC-2 failures marked
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TIPS = ("hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")
NAMES = {"g0": "existing gate", "cage": "cage <= 29 cm", "fix": "fixed delay 10", "freeze": "frozen hold 10"}
COLORS = {"g0": "#7F7F7F", "cage": "#2E9D5B", "fix": "#4F81BD", "freeze": "#C0504D"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def cage_at(trace: Path, k_in_bc1: int) -> float:
    d = np.load(trace, allow_pickle=True)
    links = [str(n) for n in d["table_link_names"]]
    bc1 = np.flatnonzero(d["phase"] == 4)
    k = int(bc1[min(k_in_bc1, len(bc1) - 1)])
    q = d["box_quat"][k]
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    diff = d["arm_link_pos"][k, [links.index("left_" + t) for t in TIPS]].mean(0) - d["arm_link_pos"][k, [links.index("right_" + t) for t in TIPS]].mean(0)
    return 100 * (-math.sin(yaw) * diff[0] + math.cos(yaw) * diff[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, default=Path("outputs/approach_p9/reports"))
    parser.add_argument("--eval_root", type=Path, default=Path("outputs/approach_p9/eval"))
    parser.add_argument("--output", type=Path, default=Path("outputs/approach_p9/figures/fig_p9_gate_variants.png"))
    args = parser.parse_args()
    experiments = [("enriched\n(PPO v6 far)", "p9_gate_v6far.json", "v6far", ""), ("deployment\n(PPO v7 window)", "p9_gate_win.json", "win", ""),
                   ("stress\n(gate 0.21 m)", "p9_gate_stress.json", "win", "s")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    width = 0.2
    for i, (label, report, cond, suffix) in enumerate(experiments):
        data = json.loads((args.reports / report).read_text())[cond]["variants"]
        for j, v in enumerate(("g0", "cage", "fix", "freeze")):
            entry = data.get(v + suffix)
            if entry is None:
                continue
            k, n = entry["cf_bc1_bc2"]
            lo, hi = wilson(k, n)
            x = i + (j - 1.5) * width
            axes[0].bar(x, k / n, width * 0.95, color=COLORS[v], label=NAMES[v] if i == 0 else None)
            axes[0].errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], color="k", capsize=2, lw=0.8)
            axes[0].text(x, 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=7, rotation=90, color="white")
            med, mn = entry["bc1_min_finger_cm"]
            axes[1].bar(x, med, width * 0.95, color=COLORS[v], label=NAMES[v] if i == 0 else None)
            axes[1].plot([x], [mn], "kv", ms=5)
    for ax, ylabel, title in ((axes[0], "contact-free BC-1 -> BC-2 success", "handoff success without table contact (navigation excluded)"),
                              (axes[1], "BC-1 min finger clearance [cm] (bar median, triangle minimum)", "BC-1 table clearance")):
        ax.set_xticks(range(len(experiments)), [e[0] for e in experiments], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="upper center", ncol=4, frameon=False)
    axes[0].set_ylim(0, 1.22)
    axes[0].set_yticks(np.linspace(0, 1, 6))
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.18)
    ax = axes[2]
    for j, v in enumerate(("g0s", "cages", "fixs", "freezes")):
        run = args.eval_root / f"win_{v}_seed4002_ep0-14"
        if not (run / "summary.json").exists():
            continue
        for e in json.loads((run / "summary.json").read_text())["episodes"]:
            c = cage_at(run / f"trace_{e['episode']:03d}.npz", e["handoff_step"])
            ok = (e.get("bc2") or {}).get("outcome") == "lift_hold_success"
            ax.scatter(j + np.random.default_rng(e["episode"]).uniform(-0.15, 0.15), c, s=40 if not ok else 18,
                       marker="o" if ok else "X", color=COLORS[v[:-1]], edgecolors="k" if not ok else "none")
    ax.axhline(29.0, color="k", ls="--", lw=1, label="29 cm readiness threshold")
    ax.set_xticks(range(4), [NAMES[v] for v in ("g0", "cage", "fix", "freeze")], fontsize=8)
    ax.set_ylabel("cage width at the handoff [cm]")
    ax.set_title("stress experiment: handoff state (X = BC-2 failure)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print("written", args.output)


if __name__ == "__main__":
    main()
