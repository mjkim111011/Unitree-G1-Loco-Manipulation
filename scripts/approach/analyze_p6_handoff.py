#!/usr/bin/env python3
"""Phase-6 analysis of Arena-WBC runs recorded with --table_contact_sensors (CPU only).

Per episode (summary.json + trace_XXX.npz):
  table contact per phase   navigation / BC-1 / BC-2: largest table-filtered force on an arm/hand link, the link, steps
                            above 1 N (the summary's ``table_contact``; the trace for the time of the first contact)
  BC-1 end state            last BC-1 control step = the state BC-2 starts from:
                              fingertips (thumb_2, index_1, middle_1) and palm (wrist_yaw) of each hand in the Box frame,
                              distance of each hand's fingertip centre to the nearest Box face,
                              arm / hand joint speeds (norm, rad/s), base speed,
                              Box position in the robot yaw frame and the Box displacement since the episode start
Aggregates per run and BC-2 failure vs success comparisons (Mann-Whitney U).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

BOX_HALF = 0.10
TIPS = ("hand_thumb_2_link", "hand_index_1_link", "hand_middle_1_link")


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors v (..., 3) into the frame of quaternion q (w, x, y, z)."""
    w, x, y, z = q
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return v @ rot  # = (R^T v^T)^T


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def episode(summary: dict, trace: Path) -> dict:
    out = {"episode": summary["episode"], "nav_ok": bool((summary.get("nav") or {}).get("handoff")),
           "bc1_ok": summary["outcome"] == "handoff_reached", "bc2_ok": (summary.get("bc2") or {}).get("outcome") == "lift_hold_success",
           "bc2_outcome": (summary.get("bc2") or {}).get("outcome"), "bc2_max_lift_m": (summary.get("bc2") or {}).get("max_lift_m"),
           "bc2_max_hold_steps": (summary.get("bc2") or {}).get("max_hold_steps"), "table_contact": summary.get("table_contact"),
           "gate_geometry": summary.get("best_gate_geometry")}
    out["e2e_ok"] = out["nav_ok"] and out["bc1_ok"] and out["bc2_ok"]
    if not trace.exists():
        return out
    d = np.load(trace)
    if "arm_link_pos" not in d.files:
        return out
    links = [str(n) for n in d["table_link_names"]]
    joints = [str(n) for n in d["joint_names"]]
    phase = d["phase"]
    bc1 = np.flatnonzero(phase == 4)
    if not len(bc1) or not out["bc1_ok"]:
        return out
    k = int(bc1[-1])  # last BC-1 step: the state BC-2 starts from
    box_p, box_q = d["box_pos"][k], d["box_quat"][k]
    pos = d["arm_link_pos"][k]
    end = {}
    for side in ("left", "right"):
        tips = np.stack([pos[links.index(f"{side}_{t}")] for t in TIPS])
        palm = pos[links.index(f"{side}_wrist_yaw_link")]
        tips_b = quat_rotate_inverse(box_q, tips - box_p)
        palm_b = quat_rotate_inverse(box_q, palm - box_p)
        centre = tips_b.mean(axis=0)
        end[f"{side}_tips_box_frame_m"] = np.round(centre, 4).tolist()
        end[f"{side}_palm_box_frame_m"] = np.round(palm_b, 4).tolist()
        # outside distance of the fingertip centre from the Box surface (0 = on / inside the Box)
        end[f"{side}_tips_gap_m"] = round(float(np.linalg.norm(np.clip(np.abs(centre) - BOX_HALF, 0.0, None))), 4)
        arm = [i for i, n in enumerate(joints) if n.startswith(side) and any(s in n for s in ("shoulder", "elbow", "wrist"))]
        hand = [i for i, n in enumerate(joints) if n.startswith(f"{side}_hand")]
        end[f"{side}_arm_joint_speed"] = round(float(np.linalg.norm(d["joint_vel"][k, arm])), 4)
        end[f"{side}_hand_joint_speed"] = round(float(np.linalg.norm(d["joint_vel"][k, hand])), 4)
    root, yaw = d["root_pos"][k], yaw_of(d["root_quat"][k])
    rel = box_p[:2] - root[:2]
    end["box_in_robot_frame_m"] = [round(float(np.cos(yaw) * rel[0] + np.sin(yaw) * rel[1]), 4),
                                   round(float(-np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1]), 4)]
    end["box_shift_since_start_m"] = round(float(np.linalg.norm(box_p[:2] - d["box_pos"][0, :2])), 4)
    end["box_height_change_m"] = round(float(box_p[2] - d["box_pos"][0, 2]), 4)
    end["base_speed"] = round(float(np.linalg.norm(d["lin_vel_b"][k, :2])), 4)
    tip_force = d["box_tip_force"][k] if "box_tip_force" in d.files else None
    if tip_force is not None and np.isfinite(tip_force).all():
        end["box_tip_force_N"] = np.round(tip_force, 2).tolist()  # left thumb, index, middle, right thumb, index, middle
    out["bc1_end"] = end
    return out


def flat(e: dict) -> dict:
    """Scalar BC-1-end features for comparisons."""
    b = e.get("bc1_end") or {}
    f = {}
    for side in ("left", "right"):
        if f"{side}_tips_box_frame_m" in b:
            for j, axis in enumerate("xyz"):
                f[f"{side}_tips_{axis}"] = b[f"{side}_tips_box_frame_m"][j]
            f[f"{side}_tips_gap"] = b[f"{side}_tips_gap_m"]
            f[f"{side}_arm_speed"] = b[f"{side}_arm_joint_speed"]
            f[f"{side}_hand_speed"] = b[f"{side}_hand_joint_speed"]
    if "box_in_robot_frame_m" in b:
        f["box_x_robot"], f["box_y_robot"] = b["box_in_robot_frame_m"]
        f["box_shift"] = b["box_shift_since_start_m"]
        f["base_speed"] = b["base_speed"]
    g = e.get("gate_geometry") or {}
    for key in ("left_tip", "right_tip", "cage_mid", "wrist_mid", "wrist_asym"):
        if key in g:
            f[f"gate_{key}"] = g[key]
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = {"runs": {}}
    for item in args.run:
        label, directory = item.split("=", 1)
        summary = json.loads((Path(directory) / "summary.json").read_text())
        rows = [episode(e, Path(directory) / f"trace_{e['episode']:03d}.npz") for e in summary["episodes"]]
        n = len(rows)
        contact = {}
        for ph in ("nav", "bc1", "bc2"):
            hit = [r for r in rows if r["table_contact"] and ph in r["table_contact"] and r["table_contact"][ph]["max_N"] > 1.0]
            contact[ph] = {"episodes": len(hit), "of": sum(1 for r in rows if r["table_contact"] and ph in r["table_contact"]),
                           "links": sorted({r["table_contact"][ph]["link"] for r in hit}),
                           "max_N": round(max((r["table_contact"][ph]["max_N"] for r in hit), default=0.0), 2),
                           "episode_ids": [r["episode"] for r in hit]}
        ran_bc2 = [r for r in rows if r["nav_ok"] and r["bc1_ok"] and "bc1_end" in r]
        ok, bad = [flat(r) for r in ran_bc2 if r["bc2_ok"]], [flat(r) for r in ran_bc2 if not r["bc2_ok"]]
        compare = {}
        for key in (ok[0] if ok else {}):
            a, b = [x[key] for x in bad if key in x], [x[key] for x in ok if key in x]
            compare[key] = {"bc2_fail_median": round(float(np.median(a)), 4) if a else None, "bc2_ok_median": round(float(np.median(b)), 4),
                            "mannwhitney_p": round(float(stats.mannwhitneyu(a, b)[1]), 4) if len(a) >= 2 else None}
        report["runs"][label] = {
            "directory": directory, "episodes": n,
            "nav": sum(r["nav_ok"] for r in rows), "bc1_given_nav": sum(r["bc1_ok"] for r in rows if r["nav_ok"]),
            "bc2_given_nav_bc1": sum(r["bc2_ok"] for r in rows if r["nav_ok"] and r["bc1_ok"]), "e2e": sum(r["e2e_ok"] for r in rows),
            "table_contact_by_phase": contact, "bc1_end_bc2_fail_vs_ok": compare,
            "bc2_failures": [{"episode": r["episode"], "outcome": r["bc2_outcome"], "max_lift_m": r["bc2_max_lift_m"],
                              "max_hold_steps": r["bc2_max_hold_steps"], **flat(r)} for r in ran_bc2 if not r["bc2_ok"]],
            "episode_rows": rows,
        }
        a = report["runs"][label]
        print(f"{label}: nav {a['nav']}/{n} bc1|nav {a['bc1_given_nav']} bc2|nav,bc1 {a['bc2_given_nav_bc1']} e2e {a['e2e']} | table contact "
              + ", ".join(f"{ph} {c['episodes']}/{c['of']} {c['links']}" for ph, c in contact.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
