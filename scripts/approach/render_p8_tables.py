#!/usr/bin/env python3
"""Phase-8 report tables (Markdown, Korean headers) from summarize_p8_runs.py output (CPU only).

For every condition with a "<condition> | v31" run (and, when present, "<condition> | v2"): navigation handoff, table
contact per phase, peak force, minimum finger clearance, BC-1 gate, collision-free BC-1, BC-2 after the gate, E2E and
strict collision-free E2E, with Fisher exact p (unpaired, same seeds) for contact, gate, E2E and strict collision-free E2E.
Usage: render_p8_tables.py --summary reports/eval_v31_summary.json [--output reports/eval_v31_tables.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def frac(d) -> str:
    return "–" if not d or not d["n"] else f"{d['k']}/{d['n']}"


def med(d, scale=1.0, unit="", digits=1) -> str:
    if not d:
        return "–"
    return f"{scale * d['median']:.{digits}f}{unit} [{scale * d['p10']:.{digits}f}–{scale * d['p90']:.{digits}f}]"


def fp(paired, key) -> str:
    u = (paired or {}).get(key, {}).get("unpaired")
    if not u:
        return ""
    p = u["fisher_p"]
    return f"p = {p:.2g}" if p >= 1e-4 else f"p = {p:.1e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    runs, pairs = data["runs"], data["paired"]
    conditions = []
    for label in runs:
        cond, pol = label.rsplit(" | ", 1)
        if pol == "v31" and cond not in conditions:
            conditions.append(cond)
    lines = ["| 조건 | 시스템 | n | nav handoff | nav 접촉 | BC-1 접촉 | BC-1 최대 힘 (접촉 시) | 최소 손가락 거리 [p10–p90] | BC-1 gate | "
             "**Collision-free BC-1** | BC-2 / gate | BC-2 중 접촉 | E2E (기존) | **Collision-free E2E (strict)** |",
             "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    tests = ["| 조건 | 테이블 접촉 | gate | E2E | strict collision-free E2E | 같은 시작 episode |", "|---|---|---|---|---|---:|"]
    for cond in conditions:
        for pol, name in (("v2", "v2 + BC-2 v2"), ("v31", "**v3.1 + BC-2 v3.1**")):
            label = f"{cond} | {pol}"
            if label not in runs:
                continue
            s = runs[label]["summary"]
            force = s.get("bc1_max_N_contact_only")
            force_text = "–" if not force else f"{force['median']:.0f} N"
            lines.append(
                f"| {cond} | {name} | {s['episodes']} | {frac(s['nav_handoff'])} | {frac(s['nav_contact'])} | {frac(s['approach_contact'])} | "
                f"{force_text} | "
                f"{med(s['bc1_min_finger_m'], 100, ' cm', 2)} | {frac(s['bc1_gate'])} | {frac(s['bc1_cf_success'])} | "
                f"{frac(s['bc2_success_given_gate'])} | {frac(s['bc2_contact'])} | {frac(s['e2e'])} | {frac(s['cf_e2e_strict'])} |")
        pair = pairs.get(f"{cond} | v2:{cond} | v31")
        if pair:
            tests.append(f"| {cond} | {fp(pair, 'approach_contact')} | {fp(pair, 'bc1_gate')} | {fp(pair, 'e2e')} | {fp(pair, 'cf_e2e_strict')} | "
                         f"{pair['identical_start_episodes']}/{pair['episodes']} |")
    sweep = [(float(label.split("sweep dx ")[1].split(" cm")[0]), label) for label in runs if label.startswith("sweep dx ")]
    sweep_lines = []
    if sweep:
        sweep_lines = ["| 목표 dx | n | nav handoff | nav 접촉 | BC-1 접촉 | 최소 손가락 거리 | gate | BC-2 / gate | strict collision-free E2E |",
                       "|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
        for dx, label in sorted(sweep):
            s = runs[label]["summary"]
            sweep_lines.append(f"| {dx:+.1f} cm | {s['episodes']} | {frac(s['nav_handoff'])} | {frac(s['nav_contact'])} | {frac(s['approach_contact'])} | "
                               f"{med(s['bc1_min_finger_m'], 100, ' cm', 2)} | {frac(s['bc1_gate'])} | {frac(s['bc2_success_given_gate'])} | "
                               f"{frac(s['cf_e2e_strict'])} |")
    text = "\n".join(["### 조건별 결과", *lines, "", "### v2 대 v3.1 검정 (Fisher exact, 비짝)", *tests, "", "### 시작 영역 sweep", *sweep_lines]) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
