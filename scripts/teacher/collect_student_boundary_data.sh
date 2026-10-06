#!/usr/bin/env bash
# Readiness-gated collection: this does NOT train Student policies or implement a
# successful Teacher Grasp-Lift rollout after Student Approach.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash scripts/teacher/collect_student_boundary_data.sh [--collect|--smoke] [--alignment-probe] [--dry-run]' \
        '  Default: evaluate 10 Student FSM episodes ONLY; do not collect Teacher/DAgger data.' \
        '  Require at least 3 Grasp and 2 Carry end states before permitting later stages.' \
        '  --collect: after readiness passes, run 2,000 FSM attempts, then target 500' \
        '             Place/Carry Teacher successes each and 500 Grasp DAgger episodes.' \
        '  --smoke: after readiness passes, reuse screening states for 5 Teacher' \
        '           successes per phase (20 attempts) and 5 DAgger episodes.' \
        '  --alignment-probe: test stronger wrist correction; NOT a validated improvement.' \
        '           Strict contact/lift/handoff success thresholds remain unchanged.' \
        '  Success counts are targets, not guarantees.' \
        '  --dry-run: print commands without creating files or starting simulation.' \
        'Environment overrides:' \
        '  LM_PY, LM_RUN_DIR (must not exist), LM_FSM_EPISODES, LM_TEACHER_TARGET,' \
        '  LM_TEACHER_TRIALS, LM_DAGGER_EPISODES, LM_SEED, LM_WORK_AREA_SIZE_M,' \
        '  LM_APPROACH_CHECKPOINT, LM_GRASP_CHECKPOINT, LM_CARRY_CHECKPOINT,' \
        '  LM_PLACE_CHECKPOINT, LM_GATE, LM_ARENA_ROOT, LM_ARENA_DATASET,' \
        '  LM_SCREEN_EPISODES, LM_MIN_GRASP_STATES, LM_MIN_CARRY_STATES,' \
        '  LM_EPISODE_SECONDS, LM_WRIST_SERVO_GAIN, LM_WRIST_SERVO_MAX_M,' \
        '  LM_ENDPOINT_OFFSET_STEPS, LM_ENDPOINT_CENTERING_M.' \
        '  Screening states are held out from production collection in --collect mode.' \
        'Exit: 0 = requested checks/targets met; 2 = readiness/targets unmet;' \
        '      other nonzero = execution or artifact validation error.'
}

DRY_RUN=0
SMOKE=0
COLLECT=0
ALIGNMENT_PROBE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --smoke) SMOKE=1 ;;
        --collect) COLLECT=1 ;;
        --alignment-probe) ALIGNMENT_PROBE=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; usage >&2; exit 1 ;;
    esac
done
if (( SMOKE && COLLECT )); then
    printf 'Choose either --smoke or --collect.\n' >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$REPO_ROOT"

if (( SMOKE )); then
    : "${LM_FSM_EPISODES:=10}" "${LM_TEACHER_TARGET:=5}"
    : "${LM_TEACHER_TRIALS:=20}" "${LM_DAGGER_EPISODES:=5}"
else
    : "${LM_FSM_EPISODES:=2000}" "${LM_TEACHER_TARGET:=500}"
    : "${LM_TEACHER_TRIALS:=2500}" "${LM_DAGGER_EPISODES:=500}"
fi
if [[ -z "${LM_PY:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        LM_PY="$REPO_ROOT/.venv/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        LM_PY="$VIRTUAL_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        LM_PY="$CONDA_PREFIX/bin/python"
    else
        LM_PY="python3.11"
    fi
fi
: "${LM_SEED:=20261005}" "${LM_WORK_AREA_SIZE_M:=5.0}"
: "${LM_SCREEN_EPISODES:=10}" "${LM_MIN_GRASP_STATES:=3}" "${LM_MIN_CARRY_STATES:=2}"
: "${LM_EPISODE_SECONDS:=60}"
if (( ALIGNMENT_PROBE )); then
    : "${LM_WRIST_SERVO_GAIN:=1.0}" "${LM_WRIST_SERVO_MAX_M:=0.12}"
    : "${LM_ENDPOINT_OFFSET_STEPS:=10}" "${LM_ENDPOINT_CENTERING_M:=0.05}"
else
    : "${LM_WRIST_SERVO_GAIN:=0.7}" "${LM_WRIST_SERVO_MAX_M:=0.08}"
    : "${LM_ENDPOINT_OFFSET_STEPS:=5}" "${LM_ENDPOINT_CENTERING_M:=0.03}"
fi
: "${LM_APPROACH_CHECKPOINT:=data/teacher/bc/box_approach_pregrasp_23d_workspace5m_r01.pt}"
: "${LM_GRASP_CHECKPOINT:=data/teacher/bc/box_grasp_lift_43d_from_approach_r05_dagger_r02_probe.pt}"
: "${LM_CARRY_CHECKPOINT:=data/teacher/bc/box_carry_23d_teacher500_r05.pt}"
: "${LM_PLACE_CHECKPOINT:=data/teacher/bc/box_place_retrain_20261005.pt}"
: "${LM_GATE:=data/teacher/bc/box_grasp_lift_43d_from_approach_r05_dagger_r02.gate.json}"
if [[ -z "${LM_ARENA_ROOT:-}" ]]; then
    if [[ -n "${UNITREE_ARENA_ROOT:-}" ]]; then
        LM_ARENA_ROOT="$UNITREE_ARENA_ROOT"
    else
        LM_ARENA_ROOT="$(python3 -c 'from pathlib import Path; print(Path.home() / "IsaacLab-Arena")')"
        if [[ -d "$REPO_ROOT/external/IsaacLab-Arena" && ( ! -d "$LM_ARENA_ROOT" || -n "${UNITREE_WORKSPACE:-}" ) ]]; then
            LM_ARENA_ROOT="$REPO_ROOT/external/IsaacLab-Arena"
        fi
    fi
fi
: "${LM_ARENA_DATASET:=$LM_ARENA_ROOT/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5}"
export LM_FSM_EPISODES LM_TEACHER_TARGET LM_TEACHER_TRIALS LM_DAGGER_EPISODES
export LM_SEED LM_WORK_AREA_SIZE_M
export LM_SCREEN_EPISODES LM_MIN_GRASP_STATES LM_MIN_CARRY_STATES LM_EPISODE_SECONDS
export LM_WRIST_SERVO_GAIN LM_WRIST_SERVO_MAX_M LM_ENDPOINT_OFFSET_STEPS LM_ENDPOINT_CENTERING_M
export PYTHONPATH="$REPO_ROOT/source/unitree_rl_lab${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# Offline validation also catches Isaac shutdown paths that report exit code 0
# despite a Python exception: a log or an HDF5 file alone is not proof of success.
audit() {
    "$LM_PY" - "$@" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

from unitree_rl_lab.utils.teacher.phase_bc import (
    discover_dagger_episodes, discover_episodes, validate_bc2_gate,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import load_snapshots, SnapshotKind

mode, *args = sys.argv[1:]
if mode == "preflight":
    gate_path, approach, grasp, carry, place, arena_root, arena_dataset = args
    for key in ("LM_FSM_EPISODES", "LM_TEACHER_TARGET", "LM_TEACHER_TRIALS", "LM_DAGGER_EPISODES",
                "LM_SCREEN_EPISODES", "LM_MIN_GRASP_STATES", "LM_MIN_CARRY_STATES"):
        if int(os.environ[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(os.environ["LM_TEACHER_TARGET"]) > int(os.environ["LM_TEACHER_TRIALS"]):
        raise ValueError("LM_TEACHER_TRIALS must be >= LM_TEACHER_TARGET")
    if not math.isfinite(float(os.environ["LM_WORK_AREA_SIZE_M"])) or float(os.environ["LM_WORK_AREA_SIZE_M"]) <= 1.5:
        raise ValueError("LM_WORK_AREA_SIZE_M must be finite and exceed 1.5")
    for key in ("LM_MIN_GRASP_STATES", "LM_MIN_CARRY_STATES"):
        if int(os.environ[key]) > int(os.environ["LM_SCREEN_EPISODES"]):
            raise ValueError(f"{key} must be <= LM_SCREEN_EPISODES")
    for key in ("LM_EPISODE_SECONDS", "LM_WRIST_SERVO_MAX_M"):
        if not math.isfinite(float(os.environ[key])) or float(os.environ[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("LM_WRIST_SERVO_GAIN", "LM_ENDPOINT_CENTERING_M"):
        if not math.isfinite(float(os.environ[key])) or float(os.environ[key]) < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if int(os.environ["LM_ENDPOINT_OFFSET_STEPS"]) < 0:
        raise ValueError("LM_ENDPOINT_OFFSET_STEPS must be non-negative")
    int(os.environ["LM_SEED"])
    for value in (approach, grasp, carry, place, arena_dataset, str(Path(arena_root) / ".venv/bin/python")):
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    gate = validate_bc2_gate(gate_path, required_action_key="processed_43d_joint_target")
    if Path(gate["checkpoint_path"]).resolve() != Path(grasp).resolve():
        raise ValueError("LM_GATE certifies a different LM_GRASP_CHECKPOINT; certify it first")
    print("[preflight] Input files and certified Grasp gate verified")
elif mode == "fsm":
    metrics_path, snapshot_path, *requested = args
    metrics = json.loads(Path(metrics_path).read_text())
    expected_episodes = int(requested[0]) if requested else int(os.environ["LM_FSM_EPISODES"])
    if metrics.get("episodes") != expected_episodes:
        raise ValueError("FSM did not complete the requested episode count")
    if len(metrics.get("episode_results", [])) != metrics["episodes"]:
        raise ValueError("FSM metrics have incomplete episode results")
    if metrics.get("grasp_checkpoint_certified_by_gate") is not True:
        raise ValueError("FSM used an uncertified Grasp checkpoint")
    if not Path(snapshot_path).exists():
        if metrics.get("snapshot_count") != 0:
            raise ValueError("FSM reported snapshots, but the snapshot file is missing")
        print("0 0")
    else:
        snapshots = load_snapshots(snapshot_path)
        if len(snapshots) != metrics.get("snapshot_count"):
            raise ValueError("FSM snapshot count mismatch")
        print(sum(s.kind == SnapshotKind.BILATERAL_HOLD for s in snapshots),
              sum(s.kind == SnapshotKind.CARRY_AT_STAGING for s in snapshots))
elif mode == "readiness":
    metrics_path, grasp_count, carry_count, report_path = args
    metrics = json.loads(Path(metrics_path).read_text())
    grasp_count, carry_count = int(grasp_count), int(carry_count)
    minimum_grasp = int(os.environ["LM_MIN_GRASP_STATES"])
    minimum_carry = int(os.environ["LM_MIN_CARRY_STATES"])
    rows = metrics["episode_results"]
    ready = grasp_count >= minimum_grasp and carry_count >= minimum_carry
    report = {
        "ready_for_collection": ready,
        "episodes": metrics["episodes"],
        "approach_successes": sum("approach" in row.get("transitions", []) for row in rows),
        "grasp_end_states": grasp_count, "carry_end_states": carry_count,
        "minimum_grasp_states": minimum_grasp, "minimum_carry_states": minimum_carry,
        "reason_counts": {reason: sum(row.get("reason") == reason for row in rows)
                          for reason in sorted({row.get("reason", "unknown") for row in rows})},
        "qr_seen_episodes": sum(row.get("box_visible_steps", 0) > 0 for row in rows),
        "near_box_episodes": sum(row.get("approach_near_reach_steps", 0) > 0 for row in rows),
        "handoff_gate_seen_episodes": sum(row.get("full_handoff_gate_steps", 0) > 0 for row in rows),
        "success_conditions_unchanged": True,
        "alignment_settings": {key: metrics.get(key) for key in (
            "approach_endpoint_offset_steps", "approach_endpoint_centering_m",
            "approach_wrist_servo_gain", "approach_wrist_servo_max_correction_m")},
        "next_action": "explicit_collection_permitted" if ready else "improve_approach_or_phase_transition_then_retest",
    }
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ready:
        print("[STOP] Too few successful Student end states. Teacher/DAgger collection will NOT start.")
        sys.exit(2)
elif mode in ("teacher", "dagger"):
    metrics_path, dataset_path = args
    metrics = json.loads(Path(metrics_path).read_text())
    if metrics.get("status") != "completed":
        raise ValueError(f"Incomplete {mode} collection")
    if not metrics.get("dataset_published"):
        if mode == "teacher" and metrics.get("successes") == 0:
            print("[incomplete] No successful Teacher trajectories published")
            sys.exit(2)
        raise ValueError("Dataset was not published")
    if mode == "teacher":
        if metrics.get("evaluation_mode") != "teacher_collection":
            raise ValueError("Expected Teacher execution, not Student-state relabels")
        episodes = discover_episodes([dataset_path], success_only=True)
        if len(episodes) != metrics["successes"]:
            raise ValueError("Teacher success count differs from the validated dataset")
        print(f"[verified] Teacher successes={len(episodes)}, source states={metrics['unique_source_snapshot_count']}")
        if metrics.get("target_successes_reached") is not True:
            print("[incomplete] Teacher success target was not reached")
            sys.exit(2)
    else:
        discover_dagger_episodes([dataset_path])
        if metrics.get("episodes") != int(os.environ["LM_DAGGER_EPISODES"]):
            raise ValueError("DAgger did not complete the requested episode count")
        print(f"[verified] DAgger samples={metrics['samples']} (not successful Teacher rollouts)")
else:
    raise ValueError(mode)
PY
}

if (( DRY_RUN )); then
    LM_RUN_DIR="${LM_RUN_DIR:-$REPO_ROOT/data/teacher/production/DRY_RUN_student_boundary}"
else
    [[ -x "$LM_PY" ]] || { printf 'Python not executable: %s\n' "$LM_PY" >&2; exit 1; }
    audit preflight "$LM_GATE" "$LM_APPROACH_CHECKPOINT" "$LM_GRASP_CHECKPOINT" \
        "$LM_CARRY_CHECKPOINT" "$LM_PLACE_CHECKPOINT" "$LM_ARENA_ROOT" "$LM_ARENA_DATASET"
    if [[ -n "${LM_RUN_DIR:-}" ]]; then
        # Deliberately refuse existing directories; never silently overwrite a run.
        mkdir -- "$LM_RUN_DIR"
        LM_RUN_DIR="$(cd -- "$LM_RUN_DIR" && pwd)"
    else
        LM_RUN_DIR="$(mktemp -d "$REPO_ROOT/data/teacher/production/student_boundary_XXXXXX")"
    fi
    mkdir -- "$LM_RUN_DIR/logs" "$LM_RUN_DIR/tmp"
    export TMPDIR="$LM_RUN_DIR/tmp"
fi

INCOMPLETE=0
trap 'printf "\nCollection stopped. Existing artifacts remain in: %s\n" "$LM_RUN_DIR" >&2' EXIT
run_step() {
    local name="$1"
    shift
    printf '\n[%s]\n' "$name"
    printf 'Command: '; printf '%q ' "$@"; printf '\n'
    if (( ! DRY_RUN )); then
        "$@" 2>&1 | tee "$LM_RUN_DIR/logs/$name.log"
    fi
}
check_collection() {
    local kind="$1" path="$2" code=0
    if (( DRY_RUN )); then return; fi
    audit "$kind" "${path%.hdf5}.metrics.json" "$path" || code=$?
    if (( code == 2 )); then
        INCOMPLETE=1
    elif (( code != 0 )); then
        exit "$code"
    fi
}

printf 'Output directory: %s\n' "$LM_RUN_DIR"
printf 'Grasp collection below is DAgger labels, NOT successful Teacher Grasp demonstrations.\n'
run_fsm() {
    local label="$1" episodes="$2" seed="$3" prefix="$4"
    run_step "$label" "$LM_PY" scripts/teacher/play_bc_skill_fsm.py \
    --headless --device cuda:0 --seed "$seed" \
    --arena_root "$LM_ARENA_ROOT" --arena_dataset "$LM_ARENA_DATASET" \
    --randomize_work_area --work_area_size_m "$LM_WORK_AREA_SIZE_M" \
    --approach_checkpoint "$LM_APPROACH_CHECKPOINT" \
    --grasp_lift_checkpoint "$LM_GRASP_CHECKPOINT" \
    --carry_checkpoint "$LM_CARRY_CHECKPOINT" --place_checkpoint "$LM_PLACE_CHECKPOINT" \
    --bc2_gate "$LM_GATE" \
    --approach_task_target_passthrough --approach_qr_endpoint_alignment \
    --approach_wrist_servo_gain "$LM_WRIST_SERVO_GAIN" \
    --approach_wrist_servo_max_correction_m "$LM_WRIST_SERVO_MAX_M" \
    --approach_endpoint_offset_steps "$LM_ENDPOINT_OFFSET_STEPS" \
    --approach_endpoint_centering_m "$LM_ENDPOINT_CENTERING_M" \
    --grasp_handoff_steps 100 --grasp_phase_timeout_seconds 20 \
    --grasp_live_approach_lower_body --grasp_freeze_hands_on_contact \
    --grasp_freeze_upper_on_lift --grasp_upper_freeze_min_lift_m 0.17 \
    --carry_handoff_steps 100 --loaded_stabilization_steps 100 \
    --loaded_stand_steps 250 --carry_upper_residual_scale 0.0 \
    --carry_lower_body_mode student --stop_after_carry_snapshot \
    --episodes "$episodes" --episode_seconds "$LM_EPISODE_SECONDS" \
    --snapshot_output "$prefix.snapshots.hdf5" --metrics_output "$prefix.metrics.json"
}

SCREEN_PREFIX="$LM_RUN_DIR/screen_student_end_states"
run_fsm 00_readiness_screen "$LM_SCREEN_EPISODES" "$LM_SEED" "$SCREEN_PREFIX"

GRASP_COUNT=1
CARRY_COUNT=1
if (( ! DRY_RUN )); then
    COUNTS="$(audit fsm "$SCREEN_PREFIX.metrics.json" "$SCREEN_PREFIX.snapshots.hdf5" "$LM_SCREEN_EPISODES")"
    read -r GRASP_COUNT CARRY_COUNT <<< "$COUNTS"
    audit readiness "$SCREEN_PREFIX.metrics.json" "$GRASP_COUNT" "$CARRY_COUNT" \
        "$LM_RUN_DIR/readiness_report.json"
else
    printf '[dry-run] Later commands are conditional on the readiness gate passing.\n'
fi

if (( ! COLLECT && ! SMOKE )); then
    trap - EXIT
    printf '\nScreening only; no Teacher or DAgger collection started. Results: %s\n' "$LM_RUN_DIR"
    printf 'Use --collect for guarded production collection, or --alignment-probe to test wrist alignment.\n'
    exit 0
fi

if (( SMOKE )); then
    SNAPSHOTS="$SCREEN_PREFIX.snapshots.hdf5"
else
    # Different reset seed and source file keep screening states out of training data.
    SNAPSHOTS="$LM_RUN_DIR/student_end_states.snapshots.hdf5"
    run_fsm 01_student_end_states "$LM_FSM_EPISODES" "$((LM_SEED + 1000))" \
        "$LM_RUN_DIR/student_end_states"
fi

GRASP_COUNT=1
CARRY_COUNT=1
if (( ! DRY_RUN && ! SMOKE )); then
    COUNTS="$(audit fsm "$LM_RUN_DIR/student_end_states.metrics.json" "$SNAPSHOTS")"
    read -r GRASP_COUNT CARRY_COUNT <<< "$COUNTS"
    printf 'Validated Student end states: Grasp=%s, Carry=%s\n' "$GRASP_COUNT" "$CARRY_COUNT"
    audit readiness "$LM_RUN_DIR/student_end_states.metrics.json" "$GRASP_COUNT" "$CARRY_COUNT" \
        "$LM_RUN_DIR/collection_readiness_report.json"
fi

if (( CARRY_COUNT > 0 )); then
    run_step 02_place_teacher "$LM_PY" scripts/teacher/collect_carry_place_teacher.py \
        --headless --device cuda:0 --phase place --arena_root "$LM_ARENA_ROOT" \
        --bc2_gate "$LM_GATE" --snapshot_input "$SNAPSHOTS" \
        --output "$LM_RUN_DIR/place_teacher_success.hdf5" \
        --num_trials "$LM_TEACHER_TRIALS" --target_successes "$LM_TEACHER_TARGET" --repeat_snapshots \
        --controller_handoff_steps 1 --max_phase_seconds 8 \
        --place_arm_offset_decay_steps 140 --place_raise_steps 15 \
        --place_translate_steps 50 --place_lower_steps 30 --no-adaptive_place \
        --box_static_friction 3.0 --box_dynamic_friction 2.5
    check_collection teacher "$LM_RUN_DIR/place_teacher_success.hdf5"
else
    printf '[skip] Place Teacher: no successful Student Carry end states.\n'
    INCOMPLETE=1
fi

if (( GRASP_COUNT > 0 )); then
    run_step 03_carry_teacher "$LM_PY" scripts/teacher/collect_carry_place_teacher.py \
        --headless --device cuda:0 --phase carry --arena_root "$LM_ARENA_ROOT" \
        --bc2_gate "$LM_GATE" --snapshot_input "$SNAPSHOTS" \
        --output "$LM_RUN_DIR/carry_teacher_success.hdf5" \
        --staging_snapshot_output "$LM_RUN_DIR/teacher_carry_end_states.snapshots.hdf5" \
        --num_trials "$LM_TEACHER_TRIALS" --target_successes "$LM_TEACHER_TARGET" --repeat_snapshots \
        --controller_handoff_steps 100 --loaded_stand_steps 250 \
        --staging_stabilization_steps 100 --max_phase_seconds 45
    check_collection teacher "$LM_RUN_DIR/carry_teacher_success.hdf5"
else
    printf '[skip] Carry Teacher: no successful Student Grasp end states.\n'
    INCOMPLETE=1
fi

run_step 04_grasp_dagger "$LM_PY" scripts/teacher/collect_dagger_corrections.py \
    --headless --device cuda:0 --skill grasp_lift \
    --arena_root "$LM_ARENA_ROOT" --arena_dataset "$LM_ARENA_DATASET" \
    --checkpoint "$LM_GRASP_CHECKPOINT" --bc2_gate "$LM_GATE" \
    --approach_checkpoint "$LM_APPROACH_CHECKPOINT" \
    --approach_handoff_steps 100 --grasp_live_approach_legs \
    --randomize_work_area --work_area_size_m "$LM_WORK_AREA_SIZE_M" \
    --episodes "$LM_DAGGER_EPISODES" --episode_seconds 30 \
    --seed "$((LM_SEED + 1))" --output "$LM_RUN_DIR/grasp_transition_dagger.hdf5"
check_collection dagger "$LM_RUN_DIR/grasp_transition_dagger.hdf5"

trap - EXIT
printf '\nArtifacts and per-step logs: %s\n' "$LM_RUN_DIR"
if (( INCOMPLETE )); then
    printf 'INCOMPLETE: one or more phases lacked source states or missed the success target.\n'
    exit 2
fi
if (( DRY_RUN )); then
    printf 'Dry run complete; no simulation or file creation performed.\n'
else
    printf 'Collection verified. Student retraining has NOT been started.\n'
fi
