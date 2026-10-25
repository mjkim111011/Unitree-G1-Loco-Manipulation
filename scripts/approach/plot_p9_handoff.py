#!/usr/bin/env python3
"""Phase-9 figure: what BC-1 hands over at an early gate and what BC-2 v3.1 does with it (CPU only).

Input: analyze_p9_early_handoff.py output (groups early_fail / early_ok / normal) and the traces it read.
Panels:
  1  handoff state: cage width (left - right fingertip centre along the Box y axis) against the left fingertips' gap to
     the Box face, per group; dashed line = the 29 cm readiness threshold tested in phase 9
  2  cage width during the last 40 BC-1 steps before the handoff (median and p10-p90 per group)
  3  Box lift during BC-2 (median and p10-p90 per group); 12 cm = the lift criterion
  4  steps with both hands touching the Box but the Box below 12 cm (the hold-breaking condition), per group
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TIPS = ("hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")
COLORS = {"early_fail": "#C0504D", "early_ok": "#F79646", "normal": "#2E9D5B"}


def cage_series(trace) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(trace, allow_pickle=True)
    links = [str(n) for n in d["table_link_names"]]
    li = [links.index("left_" + t) for t in TIPS]
    ri = [links.index("right_" + t) for t in TIPS]
    q = d["box_quat"]
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    diff = d["arm_link_pos"][:, li].mean(axis=1) - d["arm_link_pos"][:, ri].mean(axis=1)
    width = -np.sin(yaw) * diff[:, 0] + np.cos(yaw) * diff[:, 1]
    phase = d["phase"]
    bc1, bc2 = np.flatnonzero(phase == 4), np.flatnonzero(phase == 5)
    lift = d["box_pos"][bc2, 2] - d["box_pos"][bc1[-1], 2]
    return 100 * width[bc1[-40:]] if len(bc1) >= 40 else 100 * np.r_[np.full(40 - len(bc1), np.nan), width[bc1]], 100 * lift


def band(ax, series, color, label, x0=0):
    n = max(len(s) for s in series)
    m = np.full((len(series), n), np.nan)
    for i, s in enumerate(series):
        m[i, : len(s)] = s
    x = np.arange(n) + x0
    ax.plot(x, np.nanmedian(m, axis=0), color=color, label=label)
    ax.fill_between(x, np.nanpercentile(m, 10, axis=0), np.nanpercentile(m, 90, axis=0), color=color, alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=Path("outputs/approach_p9/reports/p9_early_handoff.json"))
    parser.add_argument("--eval_root", type=Path, default=Path("outputs/approach_p8/eval"))
    parser.add_argument("--output", type=Path, default=Path("outputs/approach_p9/figures/fig_p9_early_handoff.png"))
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    early = data["early_step"]
    groups = {"early_fail": [], "early_ok": [], "normal": []}
    for r in data["rows"]:
        g = "normal" if r["gate_step"] > early else ("early_ok" if r["bc2_ok"] else "early_fail")
        groups[g].append(r)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    ax = axes[0]
    for g, rows in groups.items():
        ax.scatter([r["left_tips_gap_cm"] for r in rows], [r["cage_width_cm"] for r in rows], s=14 if g == "normal" else 40,
                   alpha=0.5 if g == "normal" else 0.9, color=COLORS[g], label=f"{g} ({len(rows)})")
    ax.axhline(29.0, color="k", ls="--", lw=1, label="readiness threshold 29 cm")
    ax.set_xlabel("left fingertips outside the Box face at handoff [cm]")
    ax.set_ylabel("cage width at handoff [cm]")
    ax.set_title("handoff state (gate step <= %d = early)" % early, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    cages, lifts = {g: [] for g in groups}, {g: [] for g in groups}
    for g, rows in groups.items():
        for r in rows:
            w, lift = cage_series(args.eval_root / r["run"] / f"trace_{r['episode']:03d}.npz")
            cages[g].append(w)
            lifts[g].append(lift)
    ax = axes[1]
    for g in groups:
        band(ax, cages[g], COLORS[g], g, x0=-39)
    ax.axhline(29.0, color="k", ls="--", lw=1)
    ax.set_xlabel("BC-1 step relative to the handoff")
    ax.set_ylabel("cage width [cm]")
    ax.set_title("cage width before the handoff (median, p10-p90)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[2]
    for g in groups:
        band(ax, [x[:300] for x in lifts[g]], COLORS[g], g)
    ax.axhline(12.0, color="k", ls="--", lw=1, label="12 cm lift criterion")
    ax.set_xlabel("BC-2 step")
    ax.set_ylabel("Box lift [cm]")
    ax.set_title("Box lift during BC-2 v3.1 (median, p10-p90)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[3]
    keys = list(groups)
    vals = [[r.get("bc2_steps_both_but_low", np.nan) for r in groups[g]] for g in keys]
    ax.boxplot(vals, labels=[f"{g}\n(n={len(v)})" for g, v in zip(keys, vals)], showfliers=True)
    ax.set_ylabel("BC-2 steps: both hands touch the Box, Box below 12 cm")
    ax.set_title("why the 2 s hold breaks", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print("written", args.output)


if __name__ == "__main__":
    main()
