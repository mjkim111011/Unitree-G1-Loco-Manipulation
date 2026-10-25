#!/usr/bin/env python3
"""Phase 9: why does BC-2 v3.1 fail after an early BC-1 -> BC-2 handoff? (CPU only, reads summary.json + trace_XXX.npz)

For every episode that reached the BC-1 gate and ran BC-2, the state BC-2 starts from (last BC-1 control step) is
described by
  hands      fingertip centre (thumb_2, index_1, middle_1) and palm (wrist_yaw) of each hand in the Box frame
             (x towards the robot is negative, y left, z up from the Box centre; Box half size 0.10 m), the fingertip
             centre's distance outside the Box surface, the width of the cage (left - right fingertip y)
  motion     wrist and fingertip speed of each hand (central difference over the last control steps), vertical part,
             arm joint speed norm, base linear velocity (body frame) and yaw rate
  clearance  minimum finger-table clearance of each hand (collision geometry) at the handoff
  tracking   commanded minus realized wrist height (WBC sag) of each hand
  gate       the gate quantities (closest fingertip of each hand to the Box centre, cage midpoint, wrist midpoint and
             wrist asymmetry) recomputed from the trace
and the BC-2 rollout by: maximum lift, longest hold, and why the hold broke (a hand without fingertip contact while
the Box was >= 12 cm up, or the Box below 12 cm while both hands touched it), the Box tilt, and the grasp height (Box-
frame z of the fingertip centres when both hands first touch the Box).
Groups: early_fail / early_ok (gate step <= --early_step, BC-2 failed / succeeded) and normal (later gates).
Per feature: median [p10, p90] per group, Mann-Whitney p (early_fail vs early_ok, early vs normal) and the AUC of
early_fail against everything else (0.5 = no separation).
Usage: analyze_p9_early_handoff.py --run <dir> [<dir> ...] --output reports/p9_early_handoff.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

BOX_HALF = 0.10
TIPS = ("hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")
FINGER_LINKS = ("hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_index_0_link", "hand_index_1_link",
                "hand_middle_0_link", "hand_middle_1_link")


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return v @ r  # rows: R^T v for each v


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def features(summary: dict, trace: Path, dt: float = 0.02) -> dict | None:
    d = np.load(trace, allow_pickle=True)
    if "arm_link_pos" not in d.files or "clearance" not in d.files:
        return None
    links = [str(n) for n in d["table_link_names"]]
    joints = [str(n) for n in d["joint_names"]]
    phase = d["phase"]
    bc1, bc2 = np.flatnonzero(phase == 4), np.flatnonzero(phase == 5)
    if not len(bc1) or not len(bc2):
        return None
    k = int(bc1[-1])
    pos, box_p, box_q = d["arm_link_pos"], d["box_pos"], d["box_quat"]
    f = {"gate_step": int(summary["handoff_step"]), "bc2_ok": (summary.get("bc2") or {}).get("outcome") == "lift_hold_success"}
    tips_b, palm_b = {}, {}
    for side in ("left", "right"):
        tip_idx = [links.index(f"{side}_{t}") for t in TIPS]
        wrist = links.index(f"{side}_wrist_yaw_link")
        tips_w = pos[k, tip_idx].mean(axis=0)
        tips_b[side] = quat_rotate_inverse(box_q[k], tips_w - box_p[k])
        palm_b[side] = quat_rotate_inverse(box_q[k], pos[k, wrist] - box_p[k])
        for axis, j in (("x", 0), ("y", 1), ("z", 2)):
            f[f"{side}_tips_{axis}_cm"] = 100 * tips_b[side][j]
            f[f"{side}_palm_{axis}_cm"] = 100 * palm_b[side][j]
        f[f"{side}_tips_gap_cm"] = 100 * float(np.linalg.norm(np.clip(np.abs(tips_b[side]) - BOX_HALF, 0.0, None)))
        lo = max(int(bc1[0]), k - 2)
        span = max(1, k - lo)
        v_wrist = (pos[k, wrist] - pos[lo, wrist]) / (span * dt)
        v_tips = (pos[k, tip_idx].mean(axis=0) - pos[lo, tip_idx].mean(axis=0)) / (span * dt)
        f[f"{side}_wrist_speed_cms"] = 100 * float(np.linalg.norm(v_wrist))
        f[f"{side}_wrist_vz_cms"] = 100 * float(v_wrist[2])
        f[f"{side}_tips_speed_cms"] = 100 * float(np.linalg.norm(v_tips))
        f[f"{side}_tips_vz_cms"] = 100 * float(v_tips[2])
        fingers = [links.index(f"{side}_{n}") for n in FINGER_LINKS]
        f[f"{side}_finger_clearance_cm"] = 100 * float(d["clearance"][k, fingers].min())
        if "cmd_wrist_" + side in d.files:
            f[f"{side}_wrist_sag_cm"] = 100 * float(d[f"cmd_wrist_{side}"][k, 2, 3] - pos[k, wrist, 2])
        arm = [i for i, n in enumerate(joints) if n.startswith(side) and any(s in n for s in ("shoulder", "elbow", "wrist"))]
        f[f"{side}_arm_joint_speed"] = float(np.linalg.norm(d["joint_vel"][k, arm]))
        hand = [i for i, n in enumerate(joints) if n.startswith(f"{side}_hand")]
        f[f"{side}_hand_closure_rad"] = float(np.abs(d["joint_pos"][k, hand]).mean())
    f["cage_width_cm"] = f["left_tips_y_cm"] - f["right_tips_y_cm"]
    f["cage_centre_y_cm"] = (f["left_tips_y_cm"] + f["right_tips_y_cm"]) / 2
    f["tips_z_mean_cm"] = (f["left_tips_z_cm"] + f["right_tips_z_cm"]) / 2
    f["tips_x_mean_cm"] = (f["left_tips_x_cm"] + f["right_tips_x_cm"]) / 2
    f["wrist_speed_max_cms"] = max(f["left_wrist_speed_cms"], f["right_wrist_speed_cms"])
    f["tips_speed_max_cms"] = max(f["left_tips_speed_cms"], f["right_tips_speed_cms"])
    f["finger_clearance_min_cm"] = min(f["left_finger_clearance_cm"], f["right_finger_clearance_cm"])
    f["arm_joint_speed_max"] = max(f["left_arm_joint_speed"], f["right_arm_joint_speed"])
    vb = d["lin_vel_b"][k]
    f["base_vx_cms"], f["base_vy_cms"] = 100 * float(vb[0]), 100 * float(vb[1])
    f["base_speed_cms"] = 100 * float(np.linalg.norm(vb[:2]))
    f["base_yaw_rate"] = float(d["ang_vel_b"][k, 2])
    # gate quantities (as gate_geometry: closest fingertip of each hand, cage midpoint, wrist midpoint, wrist asymmetry)
    centre = box_p[k]
    near = {s: min(float(np.linalg.norm(pos[k, links.index(f"{s}_{t}")] - centre)) for t in TIPS) for s in ("left", "right")}
    f["gate_left_tip_m"], f["gate_right_tip_m"] = near["left"], near["right"]
    mids = [pos[k, [links.index(f"{s}_{t}") for t in TIPS]].mean(axis=0) for s in ("left", "right")]
    f["gate_cage_mid_m"] = float(np.linalg.norm((mids[0] + mids[1]) / 2 - centre))
    wl, wr = pos[k, links.index("left_wrist_yaw_link")], pos[k, links.index("right_wrist_yaw_link")]
    f["gate_wrist_mid_m"] = float(np.linalg.norm((wl + wr) / 2 - centre))
    f["gate_wrist_asym_m"] = abs(float(np.linalg.norm(wl - centre) - np.linalg.norm(wr - centre)))
    # BC-2 rollout
    start_z = float(box_p[k, 2])
    tip_force = d["box_tip_force"][bc2] if "box_tip_force" in d.files else None
    lift = box_p[bc2, 2] - start_z
    if tip_force is not None and np.isfinite(tip_force).all():
        left_c, right_c = (tip_force[:, :3] >= 1.0).sum(axis=1), (tip_force[:, 3:] >= 1.0).sum(axis=1)
        up = lift >= 0.12
        both = (left_c > 0) & (right_c > 0)
        f["bc2_steps_up_left_lost"] = int(np.sum(up & (left_c == 0)))
        f["bc2_steps_up_right_lost"] = int(np.sum(up & (right_c == 0)))
        f["bc2_steps_both_but_low"] = int(np.sum(both & ~up & (lift > 0.02)))
        f["bc2_steps_hold"] = int(np.sum(up & both))
        first = np.flatnonzero(both)
        if len(first):
            j = int(bc2[first[0]])
            f["bc2_first_bilateral_step"] = int(first[0])
            for side in ("left", "right"):
                tip_idx = [links.index(f"{side}_{t}") for t in TIPS]
                f[f"bc2_grasp_{side}_tips_z_cm"] = 100 * float(quat_rotate_inverse(box_q[j], pos[j, tip_idx].mean(axis=0) - box_p[j])[2])
    f["bc2_max_lift_cm"] = 100 * float(lift.max())
    f["bc2_max_hold_steps"] = int((summary.get("bc2") or {}).get("max_hold_steps") or 0)
    tilts = []
    for q in box_q[bc2]:
        w, x, y, z = q
        tilts.append(float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))))
    f["bc2_box_tilt_max_deg"] = max(tilts)
    f["bc2_table_contact"] = bool(((summary.get("safety") or {}).get("bc2") or {}).get("contact"))
    nav = summary.get("nav") or {}
    f["start_dx_cm"] = 100 * float((summary.get("start_offset_canonical") or [np.nan])[0])
    f["start_dy_cm"] = 100 * float((summary.get("start_offset_canonical") or [np.nan, np.nan])[1])
    f["start_dyaw_deg"] = float(np.degrees((summary.get("start_offset_canonical") or [np.nan] * 3)[2]))
    f["box_dx_cm"] = 100 * (float(summary["initial_box_xy"][0]) - 0.62)
    f["box_dy_cm"] = 100 * float(summary["initial_box_xy"][1])
    f["nav_steps"] = nav.get("steps")
    return f


def describe(values) -> list | None:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return None
    return [round(float(np.median(a)), 2), round(float(np.percentile(a, 10)), 2), round(float(np.percentile(a, 90)), 2), len(a)]


def auc(pos, neg) -> float | None:
    pos = [v for v in pos if v is not None and np.isfinite(v)]
    neg = [v for v in neg if v is not None and np.isfinite(v)]
    if not pos or not neg:
        return None
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return round(float(u / (len(pos) * len(neg))), 3)


def mw(a, b) -> float | None:
    a = [v for v in a if v is not None and np.isfinite(v)]
    b = [v for v in b if v is not None and np.isfinite(v)]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(f"{stats.mannwhitneyu(a, b, alternative='two-sided').pvalue:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, type=Path)
    parser.add_argument("--early_step", type=int, default=116)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.run:
        for e in json.loads((run / "summary.json").read_text())["episodes"]:
            if e.get("handoff_step") is None or e.get("bc2") is None:
                continue
            trace = run / f"trace_{e['episode']:03d}.npz"
            if not trace.exists():
                continue
            f = features(e, trace)
            if f is not None:
                f.update({"run": run.name, "episode": e["episode"]})
                rows.append(f)
    early_fail = [r for r in rows if r["gate_step"] <= args.early_step and not r["bc2_ok"]]
    early_ok = [r for r in rows if r["gate_step"] <= args.early_step and r["bc2_ok"]]
    normal = [r for r in rows if r["gate_step"] > args.early_step]
    print(f"episodes {len(rows)}: early_fail {len(early_fail)}, early_ok {len(early_ok)}, normal {len(normal)} "
          f"(normal BC-2 success {sum(r['bc2_ok'] for r in normal)}/{len(normal)})")
    keys = [k for k in rows[0] if k not in ("run", "episode", "bc2_ok") and isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
    table = {}
    for key in keys:
        col = lambda group: [r.get(key) for r in group]  # noqa: E731
        table[key] = {"early_fail": describe(col(early_fail)), "early_ok": describe(col(early_ok)), "normal": describe(col(normal)),
                      "p_fail_vs_ok": mw(col(early_fail), col(early_ok)), "p_early_vs_normal": mw(col(early_fail + early_ok), col(normal)),
                      "auc_fail_vs_rest": auc(col(early_fail), col(early_ok + normal))}
    order = sorted(table, key=lambda k: -abs((table[k]["auc_fail_vs_rest"] or 0.5) - 0.5))
    print(f"{'feature':32s} {'early_fail':>24s} {'early_ok':>24s} {'normal':>24s}  p(f/ok)  p(e/n)  AUC(f/rest)")
    for key in order:
        t = table[key]
        fmt = lambda v: "-" if v is None else f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]"  # noqa: E731
        print(f"{key:32s} {fmt(t['early_fail']):>24s} {fmt(t['early_ok']):>24s} {fmt(t['normal']):>24s}  "
              f"{t['p_fail_vs_ok'] if t['p_fail_vs_ok'] is not None else '-':>7}  {t['p_early_vs_normal'] if t['p_early_vs_normal'] is not None else '-':>6}  "
              f"{t['auc_fail_vs_rest'] if t['auc_fail_vs_rest'] is not None else '-'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"early_step": args.early_step, "counts": {"early_fail": len(early_fail), "early_ok": len(early_ok),
                                                                                   "normal": len(normal)},
                                       "features": table, "rows": rows}, indent=1, default=float) + "\n")
    print("written", args.output)


if __name__ == "__main__":
    main()
