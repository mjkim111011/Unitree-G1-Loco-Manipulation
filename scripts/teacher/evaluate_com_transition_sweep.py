#!/usr/bin/env python3
"""Fresh-process, paired comparison of CoM-based interpolation and fixed baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("instant", "fixed_0p26s", "fixed_1s", "com_predictive")
# method -> (carry_blend_steps, CoM mode or None, CoM walking-command gate, walk only after takeover)
METHOD_SPECS = {
    "instant": (0, None, False, False),
    "fixed_0p26s": (13, None, False, False),
    "fixed_0p5s": (25, None, False, False),
    "fixed_1s": (50, None, False, False),
    "fixed_1s_walk": (50, None, False, True),  # ablation of com_v2_cmd without any CoM criterion
    "fixed_2s": (100, None, False, False),
    "com_predictive": (0, "v1", False, False),
    "com_v2": (0, "v2", False, False),
    "com_v2_cmd": (0, "v2", True, False),
}
# Box drops / falls within this long after the handoff starts are attributed
# to the transition; later failures and horizons are navigation/carry failures.
TRANSITION_FAILURE_SECONDS = 3.0
STEP_DT = 0.02
START_TOLERANCES = {
    "joint_position": 1e-5, "joint_velocity": 1e-4, "root_state": 1e-4,
    "box_state": 1e-4, "last_action": 1e-7, "student_observation": 1e-4,
    "foot_load_fraction": 1e-4,
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_starts(episodes):
    reference = episodes[0]
    if any(e["snapshot_id"] != reference["snapshot_id"] for e in episodes):
        raise ValueError("different snapshots were selected for the paired trial")
    differences = {}
    for key, tolerance in START_TOLERANCES.items():
        values = reference["start_features"][key]
        if any(len(e["start_features"][key]) != len(values) for e in episodes):
            raise ValueError(f"start feature shape differs: {key}")
        if any(not math.isfinite(v) for e in episodes for v in e["start_features"][key]):
            raise ValueError("nonfinite restored state")
        difference = max(abs(a-b) for e in episodes for a,b in zip(values, e["start_features"][key]))
        differences[key] = difference
        if not math.isfinite(difference):
            raise ValueError("nonfinite restored state")
    matched = all(differences[k] <= tolerance for k, tolerance in START_TOLERANCES.items())
    return matched, differences


def failure_seconds(episode):
    trace = episode["carry"].get("trace", [])
    return (trace[-1]["carry_step"] + 1) * STEP_DT if trace else None


def transition_failure(episode):
    carry = episode["carry"]
    if carry["success"] or carry["reason"] not in ("box_drop", "bad_orientation"):
        return False
    seconds = failure_seconds(episode)
    return seconds is not None and seconds <= TRANSITION_FAILURE_SECONDS


def prediction_errors(trace):
    """Distance between v2's predicted capture point and the one measured later, per horizon."""
    measured = {}
    for r in trace:
        debug = r.get("com_predictive_blend") or {}
        if "measured_capture_w" in debug:
            measured[r["carry_step"]] = debug["measured_capture_w"]
    errors = {}
    for r in trace:
        for key, item in ((r.get("com_predictive_blend") or {}).get("predicted_capture_w") or {}).items():
            actual = measured.get(item["carry_step"])
            if actual is not None:
                errors.setdefault(key, []).append(math.dist(item["xy"], actual))
    return errors


def reports(groups, methods=METHODS, reference="fixed_1s"):
    matrix = []
    for index, group in sorted(groups.items()):
        matched, differences = compare_starts(list(group.values()))
        common_valid = matched and all(e["restore_success"] for e in group.values())
        row = {"state_index": index, "snapshot_id": next(iter(group.values()))["snapshot_id"],
               "paired_start_matched": matched, "common_valid_restore": common_valid}
        row.update({f"start_max_difference_{k}": v for k,v in differences.items()})
        for method, e in group.items():
            row[f"success_{method}"] = int(e["carry"]["success"]) if e["restore_success"] else "restore_invalid"
            row[f"reason_{method}"] = e["carry"]["reason"]
        matrix.append(row)
    valid_indices = [r["state_index"] for r in matrix if r["common_valid_restore"]]
    summary = []
    for method in methods:
        trials = [groups[i][method] for i in valid_indices]
        success = sum(e["carry"]["success"] for e in trials)
        jumps, takeovers, active, fallback, deadline = [], [], 0, 0, 0
        lateral, reasons, errors, kinds = [], {}, {}, {}
        for e in trials:
            trace = e["carry"].get("trace", [])
            first = [r for r in trace if 0 <= r["carry_step"] < 100]
            if first:
                jumps.append(max(r["target_jump_rad"] for r in first))
                lateral.append(max(abs(r["root_velocity_body"][1]) for r in first))
            for key, values in prediction_errors(trace).items():
                errors.setdefault(key, []).extend(values)
            done = next((r for r in trace if r.get("bc3_blend_weight", 0) >= 1 - 1e-5), None)
            if done:
                takeovers.append((done["carry_step"] + 1) * 0.02)
            for r in first:
                debug = r.get("com_predictive_blend") or {}
                if debug.get("active"):
                    active += 1
                    fallback += bool(debug.get("fallback"))
                    deadline += bool(debug.get("deadline_forced"))
                    for reason in debug.get("fallback_reasons", []) if debug.get("fallback") else []:
                        reasons[reason] = reasons.get(reason, 0) + 1
                    if "selected_kind" in debug:
                        kinds[debug["selected_kind"]] = kinds.get(debug["selected_kind"], 0) + 1
        summary.append({
            "method": method, "states_evaluated": len(groups),
            "paired_valid_states": len(trials), "carry_successes": success,
            "carry_success_rate": success / len(trials) if trials else None,
            "mean_takeover_seconds": sum(takeovers) / len(takeovers) if takeovers else None,
            "mean_first2s_peak_target_jump_rad": sum(jumps) / len(jumps) if jumps else None,
            "com_active_decisions": active, "com_fallback_decisions": fallback,
            "com_deadline_floor_decisions": deadline,
            "transition_failures": sum(transition_failure(e) for e in trials),
            "other_failures": sum(not e["carry"]["success"] and not transition_failure(e) for e in trials),
            "mean_first2s_peak_lateral_base_speed_mps": sum(lateral) / len(lateral) if lateral else None,
            "com_fallback_reasons": json.dumps(reasons, sort_keys=True),
            "com_selected_schedules": json.dumps(kinds, sort_keys=True),
            **{f"prediction_error_{key}_mean_m": sum(v) / len(v) for key, v in sorted(errors.items())},
        })
    if reference in methods:
        # Paired comparison against the reference on the same valid starts.
        for row in summary:
            pairs = [(groups[i][row["method"]]["carry"]["success"], groups[i][reference]["carry"]["success"])
                     for i in valid_indices]
            row[f"only_this_succeeded_vs_{reference}"] = sum(a and not b for a, b in pairs)
            row[f"only_{reference}_succeeded"] = sum(b and not a for a, b in pairs)
    keys = sorted({k for row in summary for k in row}, key=lambda k: (k != "method", k))
    summary = [{k: row.get(k) for k in keys} for row in summary]
    return summary, matrix


def save_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_for(args, index, method, output):
    blend_steps, mode, gate, walk_after = METHOD_SPECS[method]
    command = [
        sys.executable, "-u", str(ROOT / "scripts/teacher/evaluate_bc2_rollout.py"),
        "--checkpoint", str(args.checkpoint), "--carry_checkpoint", str(args.carry_checkpoint),
        "--transition_snapshot_input", *map(str, args.snapshots),
        "--transition_snapshot_offset", str(index), "--episodes", "1",
        "--seed", str(args.seed), "--carry_seconds", str(args.carry_seconds),
        "--device", args.device, "--output", str(output),
        "--box_mass_kg", str(args.box_mass_kg),
    ]
    if args.box_mass_kg != args.snapshot_box_mass_kg:
        command += ["--snapshot_box_mass_kg", str(args.snapshot_box_mass_kg)]
    if mode is not None:
        command += ["--com_predictive_blend", "--com_predictive_mode", mode]
        if gate:
            command += ["--com_command_gate"]
    else:
        command += ["--carry_blend_steps", str(blend_steps)]
        if walk_after:
            command += ["--walk_after_takeover"]
    if args.headless:
        command += ["--headless"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/teacher/bc/bc2_grasp_lift_43d.pt")
    parser.add_argument("--carry_checkpoint", type=Path, default=ROOT / "data/teacher/carry_expanded_r01/bc3_carry_candidate.pt")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--carry_seconds", type=float, default=16.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs/robustness/com_blend_r01")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SPECS), default=list(METHODS))
    parser.add_argument("--reference_method", default="fixed_1s")
    parser.add_argument("--box_mass_kg", type=float, default=0.2,
                        help="Physical Box mass during the evaluated handoff and carry")
    parser.add_argument("--snapshot_box_mass_kg", type=float, default=0.2,
                        help="Box mass the snapshots were captured with")
    args = parser.parse_args()
    methods = tuple(args.methods)
    if args.states <= 0 or not math.isfinite(args.carry_seconds) or args.carry_seconds <= 2.0:
        parser.error("states must be positive and carry_seconds must exceed the 2 s deadline")
    args.snapshots = sorted({p.expanduser().resolve() for p in args.snapshots})
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.carry_checkpoint = args.carry_checkpoint.expanduser().resolve()
    inputs = [args.checkpoint, args.carry_checkpoint, *args.snapshots,
              ROOT / "scripts/teacher/evaluate_bc2_rollout.py",
              ROOT / "source/unitree_rl_lab/unitree_rl_lab/utils/teacher/com_predictive_blend.py",
              ROOT / "source/unitree_rl_lab/unitree_rl_lab/utils/teacher/com_schedule_blend.py",
              ROOT / "source/unitree_rl_lab/unitree_rl_lab/utils/teacher/physical_snapshot_runtime.py"]
    for path in inputs:
        if not path.is_file():
            parser.error(f"input not found: {path}")
    signature = {
        "states": args.states, "seed": args.seed, "carry_seconds": args.carry_seconds,
        "methods": list(methods), "device": args.device,
        "box_mass_kg": args.box_mass_kg, "snapshot_box_mass_kg": args.snapshot_box_mass_kg,
        "start_tolerances": START_TOLERANCES,
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "isolation": "one_new_simulator_process_per_state_and_method",
    }
    root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.json"
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text()) != signature:
                raise ValueError("existing experiment differs or --resume is missing; use a new output_root")
        else:
            manifest.write_text(json.dumps(signature, indent=2, sort_keys=True))
    groups = {}
    for index in range(args.states):
        group = {}
        for method in methods:
            output = root / f"state_{index:04d}" / f"{method}.json"
            command = command_for(args, index, method, output)
            print(f"[com-sweep] state={index + 1}/{args.states} method={method}", flush=True)
            if args.dry_run:
                import shlex
                print(shlex.join(command))
                continue
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                log_path = output.with_suffix(".log")
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    try:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            if line.startswith(("[snapshot-eval]", "[chain]", "[progress]")):
                                print(line, end="", flush=True)
                        returncode = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                if returncode:
                    raise RuntimeError(f"evaluation failed (exit={returncode}); inspect {log_path}")
            data = json.loads(output.read_text())
            config = data["evaluator_config"]
            if (len(data["episode_results"]) != 1 or config["rollout_mode"] != "snapshot_transition"
                or config["transition_snapshot_offset"] != index or config["seed"] != args.seed
                or config["carry_seconds"] != args.carry_seconds
                or not math.isclose(config["box_mass_kg"], args.box_mass_kg)
                or config.get("com_predictive_blend") != (METHOD_SPECS[method][1] is not None)
                or config.get("com_predictive_mode", "v1" if config.get("com_predictive_blend") else None)
                != METHOD_SPECS[method][1]
                or bool(config.get("com_command_gate")) != METHOD_SPECS[method][2]
                or bool(config.get("walk_after_takeover")) != METHOD_SPECS[method][3]
                or config["carry_blend_steps"] != METHOD_SPECS[method][0]
                or data["carry_checkpoint_sha256"] != signature["input_sha256"][str(args.carry_checkpoint)]):
                raise ValueError(f"incomplete or mismatched result: {output}")
            group[method] = data["episode_results"][0]
        if args.dry_run:
            continue
        groups[index] = group
        summary, matrix = reports(groups, methods, args.reference_method)
        save_csv(root / "summary.csv", summary)
        save_csv(root / "states.csv", matrix)
        if not matrix[-1]["paired_start_matched"]:
            raise RuntimeError(f"paired start validation failed for state {index}; inspect states.csv before running more trials")
    if not args.dry_run:
        print(f"[com-sweep] done: {root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
