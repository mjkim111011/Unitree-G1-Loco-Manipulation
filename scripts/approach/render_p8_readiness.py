#!/usr/bin/env python3
"""Phase-8 grasp readiness at the BC-1 end (Markdown table) from analyze_p6_handoff.py output (CPU only).

Per run, over the episodes that reached the gate (state BC-2 starts from, last BC-1 step):
  fingertip height     mean of thumb / index / middle tips of each hand above the table top (Box frame z + Box half height)
  tip gap              distance of each hand's fingertip centre outside the Box surface (0 = touching the side)
  lateral span         distance between the left and right fingertip centres along the Box y axis (Box width 20 cm)
  arm joint speed      norm of the 7 arm joint speeds at the handoff (rad/s, mean of both arms)
  Box shift            Box displacement in the plane since the episode start
  BC-2 success         after the gate
Usage: render_p8_readiness.py --handoff reports/eval_v31_handoff.json [--output reports/eval_v31_readiness.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOX_HALF = 0.10


def q(values, scale=100.0, digits=1) -> str:
    a = np.asarray([v for v in values if v is not None], dtype=float) * scale
    if not len(a):
        return "–"
    return f"{np.median(a):.{digits}f} [{np.percentile(a, 10):.{digits}f}–{np.percentile(a, 90):.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    runs = json.loads(args.handoff.read_text())["runs"]
    lines = ["| 실행 | gate n | 손끝 높이 (테이블 위, cm) 왼 / 오 | 손끝–박스 옆면 간격 (cm) 왼 / 오 | 좌우 손끝 간격 (cm) | 팔 관절 속도 (rad/s) | 박스 이동 (cm) | BC-2 성공 |",
             "|---|---:|---|---|---|---|---|---:|"]
    for label, run in runs.items():
        rows = [r for r in run["episode_rows"] if r.get("bc1_ok") and r.get("bc1_end")]
        if not rows:
            continue
        end = [r["bc1_end"] for r in rows]
        height = {s: [e[f"{s}_tips_box_frame_m"][2] + BOX_HALF for e in end] for s in ("left", "right")}
        gap = {s: [e[f"{s}_tips_gap_m"] for e in end] for s in ("left", "right")}
        span = [e["left_tips_box_frame_m"][1] - e["right_tips_box_frame_m"][1] for e in end]
        speed = [(e["left_arm_joint_speed"] + e["right_arm_joint_speed"]) / 2 for e in end]
        shift = [e["box_shift_since_start_m"] for e in end]
        ok = sum(bool(r.get("bc2_ok")) for r in rows)
        lines.append(f"| {label.replace(' | ', ' — ')} | {len(rows)} | {q(height['left'])} / {q(height['right'])} | {q(gap['left'])} / {q(gap['right'])} | {q(span)} | "
                     f"{q(speed, 1.0, 2)} | {q(shift, 100.0, 2)} | {ok}/{len(rows)} |")
    text = "\n".join(lines) + "\n\n값: 중앙값 [p10–p90]. 손끝 높이는 엄지·검지·가운뎃손가락 끝마디 링크 원점 높이의 평균입니다(손가락 충돌 형상의 최저점이 아님; 테이블과의 실제 최소 거리는 조건별 결과 표의 '최소 손가락 거리').\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
