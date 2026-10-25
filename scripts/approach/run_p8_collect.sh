#!/usr/bin/env bash
# Phase-8 Method C data: collision-free Teacher demos (Method B safe path, open loop) from diverse post-navigation states
# and Box positions.  Every shard is a new HDF5 file next to its run directory (outputs/approach_p8/collect/); nothing
# existing is overwritten.  Demos are filtered afterwards (select_safe_demos.py: gate reached, no table contact,
# clearance kept), so colliding demos never enter BC-1 v3.
#   SHORT_<k>  scripted ground-truth navigation from 0.15-0.5 m behind the BC-1 start (lateral +-0.2 m, yaw +-0.4 rad),
#              target jitter +-4 cm / +-4 cm / +-0.06 rad clamped by the table-edge safe gap (0.41 m), settle 0-50 steps,
#              Box x in [0.60, 0.65] (2 cm towards the robot .. 3 cm into the table), y in [-0.03, 0.03]
#   PPO_<k>    PPO v7 navigation from 3-5 m (any heading), wide window + table-edge safe gap, Box as above
#   RESET_<k>  canonical reset start (no navigation, the original corpus start state), Box as above
# Usage: STAGES="SHORT_0 SHORT_1 PPO_0" EPISODES=40 scripts/approach/run_p8_collect.sh
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"  # REPO_ROOT: run a frozen copy of this script from another directory
source packaging/activate.sh
STAGES=${STAGES:-""}
ROOT=${ROOT:-outputs/approach_p8/collect}
LOGS=${LOGS:-outputs/approach_p8/logs/collect}
mkdir -p "$ROOT" "$LOGS"
EPISODES=${EPISODES:-40}
MARGIN=${MARGIN:-0.04}
MARGIN_LATE=${MARGIN_LATE:-0.04}   # pre-grasp margin (last 30 steps before the close onset); v3.1 data: 0.07 (run_p8_v31_pipeline.sh)
V7_JIT=outputs/approach_p5/eval/v7_nav_policy.pt
BASE=(--mode teacher_collect --teacher_path safe --safe_margin "$MARGIN" --safe_margin_late "$MARGIN_LATE" --camera_profile collector --episode_seconds 6
      --pre_upper_body nominal --no_step_log --headless --device cuda:0 --safety_metrics
      --box_x_range 0.60 0.65 --box_y_range -0.03 0.03 --table_safe_gap 0.41 --table_safe_margin 0.016)

run() {  # name, seed, args...
  local name=$1 seed=$2; shift 2
  [[ -f $ROOT/$name/summary.json ]] && { echo "skip $name"; return; }
  [[ -e $ROOT/$name ]] && mv "$ROOT/$name" "$ROOT/$name.failed.$(date +%s)"
  [[ -e $ROOT/$name.hdf5 ]] && mv "$ROOT/$name.hdf5" "$ROOT/$name.hdf5.failed.$(date +%s)"
  echo "$(date +%T) start $name"
  timeout 36000 python scripts/approach/eval_bc1_approach.py "${BASE[@]}" "$@" --seed "$seed" --episodes "$EPISODES" \
    --collect_output "$ROOT/$name.hdf5" --output_dir "$ROOT/$name" > "$LOGS/$name.log" 2>&1
  echo "$(date +%T) $name exit=$? $(grep -c '\[eval\] episode' "$LOGS/$name.log") episodes"
}

for stage in $STAGES; do
  k=${stage##*_}
  case $stage in
    SHORT_*) run "short_nav_$k" $((9100 + k)) --navigate --robot_start_range -0.5 -0.15 -0.2 0.2 -0.4 0.4 --nav_max_steps 600 \
               --nav_target_jitter 0.04 0.04 0.06 --settle_steps_range 0 50 ;;
    PPO_*)   run "ppo_nav_$k" $((9200 + k)) --nav_policy "$V7_JIT" --nav_action_profile v7 --nav_target_offset_x -0.008 \
               --nav_handoff_x_range -0.045 0.015 --nav_max_steps 2000 --settle_steps 0 --start_polar 3.0 5.0 90 3.1416 ;;
    RESET_*) run "reset_$k" $((9300 + k)) ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "P8 COLLECT DONE ($STAGES)"
