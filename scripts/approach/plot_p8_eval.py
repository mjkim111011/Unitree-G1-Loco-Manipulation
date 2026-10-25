#!/usr/bin/env python3
"""Phase-8 evaluation figure: BC-1 v2 versus BC-1 v3 (collision-free Teacher demos) under identical conditions (CPU only).

Input: summarize_p8_runs.py output whose run labels are "<condition> | <policy>" for the --policies (default v2 v3), plus
optional "sweep dx <cm> cm | <last policy>" runs.  Panels:
  1  approach table-contact rate and collision-free BC-1 success per condition (Wilson 95 % intervals)
  2  BC-2 success after the gate and strict collision-free end-to-end success (no table contact in any phase)
  3  minimum finger-table clearance during the approach (median, p10-p90; v2 runs without clearance are skipped)
  4  start-region sweep: collision-free BC-1 and end-to-end success against the navigation target dx
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ("#C0504D", "#2E9D5B", "#4F81BD", "#8064A2")


def rate(d):
    return (d["k"] / d["n"], d["wilson95"]) if d and d["n"] else (np.nan, [np.nan, np.nan])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True, help="condition names in plotting order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["v2", "v3"], help="policy labels in plotting order")
    parser.add_argument("--names", nargs="+", default=None, help="legend names of the policies (default: BC-1 <label>)")
    args = parser.parse_args()
    C = dict(zip(args.policies, PALETTE))
    NAME = dict(zip(args.policies, args.names or [f"BC-1 {p}" for p in args.policies]))
    npol = len(args.policies)
    width = 0.36 / npol
    runs = json.loads(args.summary.read_text())["runs"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))
    x = np.arange(len(args.conditions))
    for ax, keys, title in ((axes[0], (("approach_contact", "table contact", "//"), ("bc1_cf_success", "collision-free BC-1", None)),
                             "approach: table contact (hatched) and collision-free BC-1 success"),
                            (axes[1], (("bc2_success_given_gate", "BC-2 | gate", "//"), ("cf_e2e_strict", "collision-free E2E (strict)", None)),
                             "BC-2 after the gate (hatched) and collision-free end-to-end (strict)")):
        for j, pol in enumerate(args.policies):
            for k, (key, name, hatch) in enumerate(keys):
                values = [rate(runs.get(f"{c} | {pol}", {}).get("summary", {}).get(key)) for c in args.conditions]
                r = np.array([v[0] for v in values])
                lo = np.array([v[1][0] for v in values])
                hi = np.array([v[1][1] for v in values])
                pos = x + (k - 0.5) * 0.4 + (j - (npol - 1) / 2) * width
                ax.bar(pos, r, width=width * 0.95, color=C[pol], alpha=0.5 if hatch else 0.9, hatch=hatch,
                       label=f"{NAME[pol]}: {name}")
                ax.errorbar(pos, r, yerr=[np.clip(r - lo, 0, None), np.clip(hi - r, 0, None)], fmt="none", color="k", lw=0.8, capsize=2)
                for xp, rv in zip(pos, r):
                    if rv == 0:
                        ax.text(xp, 0.01, "0", ha="center", va="bottom", fontsize=6, color=C[pol])
        ax.set_xticks(x, args.conditions, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.45)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=6.5, loc="upper center", ncol=2, frameon=False)
    ax = axes[2]
    for j, pol in enumerate(args.policies):
        med, lo, hi = [], [], []
        for c in args.conditions:
            d = runs.get(f"{c} | {pol}", {}).get("summary", {}).get("bc1_min_finger_m")
            med.append(np.nan if d is None else 100 * d["median"])
            lo.append(np.nan if d is None else 100 * d["p10"])
            hi.append(np.nan if d is None else 100 * d["p90"])
        med, lo, hi = map(np.asarray, (med, lo, hi))
        pos = x + (j - (npol - 1) / 2) * (0.7 / npol)
        ax.bar(pos, med, width=0.66 / npol, color=C[pol], label=NAME[pol])
        ax.errorbar(pos, med, yerr=[np.clip(med - lo, 0, None), np.clip(hi - med, 0, None)], fmt="none", color="k", capsize=2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x, args.conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cm")
    ax.set_title("minimum finger-table clearance in the approach (median, p10-p90)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    ax = axes[3]
    last = re.escape(args.policies[-1])
    sweep = sorted((float(m.group(1)), label) for label in runs if (m := re.match(r"sweep dx ([-+\d.]+) cm \| " + last + "$", label)))
    if sweep:
        dx = [s[0] for s in sweep]
        for key, name, style in (("bc1_cf_success", "collision-free BC-1", "o-"), ("cf_e2e_strict", "collision-free E2E (strict)", "s--"),
                                 ("bc1_gate", "BC-1 gate", "^:")):
            values = [rate(runs[label]["summary"][key]) for _, label in sweep]
            ax.plot(dx, [v[0] for v in values], style, label=name)
        ax.set_xlabel("navigation target dx to the BC-1 canonical start [cm]")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{NAME[args.policies[-1]]} start region (nominal Box, PPO v7 navigation)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print("written", args.output)


if __name__ == "__main__":
    main()
