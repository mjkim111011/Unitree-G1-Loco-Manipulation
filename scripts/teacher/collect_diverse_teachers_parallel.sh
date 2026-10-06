#!/usr/bin/env bash
# Both jobs are Teachers; this does not execute Student training/DAgger.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$ROOT"
DRY=0
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        --smoke) SMOKE=1 ;;
        -h|--help) printf '%s\n' 'Usage: bash scripts/teacher/collect_diverse_teachers_parallel.sh [--smoke] [--dry-run]' 'Default: target 500 successes per case, balanced across 16 spatial cells.' 'Overrides: LM_RUN_DIR (new directory or matching resume), LM_TEACHER_TARGET, LM_SHARD_SIZE,' 'LM_PY, LM_ARENA_ROOT, LM_ARENA_DATASET, LM_DEVICE_1, LM_DEVICE_2, LM_SEED.'; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
: "${LM_PY:=$ROOT/.venv/bin/python}"
: "${LM_SEED:=20261006}"
: "${LM_DEVICE_1:=cuda:0}" "${LM_DEVICE_2:=cuda:0}"
if (( SMOKE )); then
    : "${LM_TEACHER_TARGET:=2}" "${LM_SHARD_SIZE:=1}"
else
    : "${LM_TEACHER_TARGET:=500}" "${LM_SHARD_SIZE:=25}"
fi
# Resolve source location BEFORE activating the portable workspace; do not
# assume external/Arena/data contains the large upstream HDF5.
if [[ -z "${LM_ARENA_ROOT:-}" ]]; then
    if [[ -x "$ROOT/external/IsaacLab-Arena/.venv/bin/python" ]]; then
        LM_ARENA_ROOT="$ROOT/external/IsaacLab-Arena"
    else
        LM_ARENA_ROOT="$HOME/IsaacLab-Arena"
    fi
fi
if [[ -z "${LM_ARENA_DATASET:-}" ]]; then
    LM_ARENA_DATASET="$LM_ARENA_ROOT/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5"
    if [[ ! -f "$LM_ARENA_DATASET" && -f "$HOME/IsaacLab-Arena/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5" ]]; then
        LM_ARENA_DATASET="$HOME/IsaacLab-Arena/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5"
    fi
fi
unset PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD
export PYTHONPATH="$ROOT:$ROOT/source/unitree_rl_lab"
export UNITREE_WORKSPACE="$ROOT" UNITREE_ARENA_ROOT="$LM_ARENA_ROOT"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$ROOT/external/unitree_ros}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
if (( DRY )); then
    RUN_DIR="${LM_RUN_DIR:-$ROOT/data/teacher/production/teacher_diversity_DRY_RUN}"
else
    [[ -x "$LM_PY" && -x "$LM_ARENA_ROOT/.venv/bin/python" && -f "$LM_ARENA_DATASET" ]] || { printf 'Python/Arena source missing. Set LM_PY/LM_ARENA_ROOT/LM_ARENA_DATASET.\n' >&2; exit 2; }
    mkdir -p "$ROOT/data/teacher/production"
    RUN_DIR="${LM_RUN_DIR:-$(mktemp -d "$ROOT/data/teacher/production/teacher_diversity_XXXXXX")}"
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tmp_case1" "$RUN_DIR/tmp_case2"
fi
COMMON=(--target_demos "$LM_TEACHER_TARGET" --shard_size "$LM_SHARD_SIZE" --arena_root "$LM_ARENA_ROOT" --arena_dataset "$LM_ARENA_DATASET")
CMD1=("$LM_PY" -m scripts.teacher.collect_diverse_box_teacher --case table_box --output_dir "$RUN_DIR/01_table_box" --device "$LM_DEVICE_1" --seed "$LM_SEED" "${COMMON[@]}")
CMD2=("$LM_PY" -m scripts.teacher.collect_diverse_box_teacher --case table_approach --output_dir "$RUN_DIR/02_table_approach" --device "$LM_DEVICE_2" --seed "$((LM_SEED + 100000))" "${COMMON[@]}")
printf 'Output: %s\n' "$RUN_DIR"
if (( DRY )); then
    printf '%q ' "${CMD1[@]}" --dry-run; printf '\n'
    printf '%q ' "${CMD2[@]}" --dry-run; printf '\n'
    exit 0
fi
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT TERM
TMPDIR="$RUN_DIR/tmp_case1" "${CMD1[@]}" > "$RUN_DIR/logs/01_table_box.log" 2>&1 &
PIDS+=("$!")
TMPDIR="$RUN_DIR/tmp_case2" "${CMD2[@]}" > "$RUN_DIR/logs/02_table_approach.log" 2>&1 &
PIDS+=("$!")
printf 'Parallel workers: %s %s\nLogs: %s/logs/\n' "${PIDS[0]}" "${PIDS[1]}" "$RUN_DIR"
set +e
wait "${PIDS[0]}"; STATUS1=$?
wait "${PIDS[1]}"; STATUS2=$?
set -e
trap - INT TERM
printf 'Case 1 exit=%s; Case 2 exit=%s\nCheck each manifest.json: complete=true is required.\n' "$STATUS1" "$STATUS2"
(( STATUS1 == 0 && STATUS2 == 0 ))
