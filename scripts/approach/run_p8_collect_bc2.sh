#!/usr/bin/env bash
# Phase-8 BC-2 v3 data: BC-1 v3 (collision-free approach) up to the existing BC-1 -> BC-2 gate, then the Arena Teacher
# continues grasp/lift on the safe continuation path (source path raised by a constant lift that keeps the hands off the
# table) and records BC-2 demos.  The grasp-readiness gate (GATE_TIP_MAX, default off) is not used: it lowered the BC-1
# end clearance by ~1 cm in paired box +1 cm runs.  (The table contacts of the first BC-1 v3 collection recurred
# identically without it: they came from BC-1 v3's thin pre-grasp margin, which v3.1 raises to 7 cm.)
# MARGIN_LATE sets the continuation lift (late margin over the cage formation).  FORCE_HANDOFF="LO HI" adds BC-2 demos from
# early handoffs (BC-2 v3.1 failed 7/13 after gates at steps <= 116, where it had 6 demos; the deployed gate is unchanged).  New HDF5 files next to the run directories
# (outputs/approach_p8/collect_bc2/); nothing existing is overwritten.  Demos are filtered afterwards
# (select_safe_demos.py --phase bc2).
#   SHORT_<k>  scripted navigation from 0.15-0.5 m behind the BC-1 start (as for the BC-1 v3 data), Box x 0.60-0.65, y +-3 cm
#   PPO_<k>    PPO v7 navigation from 3-5 m
#   RESET_<k>  canonical reset start
# Usage: STAGES="SHORT_0 PPO_0" EPISODES=40 scripts/approach/run_p8_collect_bc2.sh
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"
source packaging/activate.sh
STAGES=${STAGES:-""}
ROOT=${ROOT:-outputs/approach_p8/collect_bc2}
LOGS=${LOGS:-outputs/approach_p8/logs/collect_bc2}
mkdir -p "$ROOT" "$LOGS"
EPISODES=${EPISODES:-40}
MARGIN_LATE=${MARGIN_LATE:-0.035}   # continuation lift: late margin over the cage formation
BC1_V3=${BC1_V3:-outputs/approach_p8/checkpoints/bc1_v3_safe.pt}
V7_JIT=outputs/approach_p5/eval/v7_nav_policy.pt
BASE=(--checkpoint "$BC1_V3" --camera_profile collector --hand_state_decoding binary --episode_seconds 6 --pre_upper_body nominal
      --no_step_log --headless --device cuda:0 --safety_metrics
      --then_teacher --teacher_continuation_path safe --safe_margin 0.04 --safe_margin_late "$MARGIN_LATE" --bc2_seconds 10
      --box_x_range 0.60 0.65 --box_y_range -0.03 0.03 --table_safe_gap 0.41 --table_safe_margin 0.016)
GATE_TIP_MAX=${GATE_TIP_MAX:-}
[[ -n $GATE_TIP_MAX ]] && BASE+=(--gate_max_tip_height_m "$GATE_TIP_MAX")
FORCE_HANDOFF=${FORCE_HANDOFF:-}   # "LO HI": early-handoff augmentation (hand over at a random step in [LO, HI] unless the gate opened first)
[[ -n $FORCE_HANDOFF ]] && BASE+=(--force_handoff_step_range $FORCE_HANDOFF)

run() {  # name, seed, args...
  local name=$1 seed=$2; shift 2
  [[ -f $ROOT/$name/summary.json ]] && { echo "skip $name"; return; }
  [[ -e $ROOT/$name ]] && mv "$ROOT/$name" "$ROOT/$name.failed.$(date +%s)"
  echo "$(date +%T) start $name"
  timeout 36000 python scripts/approach/eval_bc1_approach.py "${BASE[@]}" "$@" --seed "$seed" --episodes "$EPISODES" \
    --collect_output "$ROOT/$name.hdf5" --output_dir "$ROOT/$name" > "$LOGS/$name.log" 2>&1
  echo "$(date +%T) $name exit=$? $(grep -c '\[eval\] episode' "$LOGS/$name.log") episodes"
}

for stage in $STAGES; do
  k=${stage##*_}
  case $stage in
    SHORT_*) run "short_nav_$k" $((9410 + k)) --navigate --robot_start_range -0.5 -0.15 -0.2 0.2 -0.4 0.4 --nav_max_steps 600 \
               --nav_target_jitter 0.04 0.04 0.06 --settle_steps_range 0 50 ;;
    PPO_*)   run "ppo_nav_$k" $((9510 + k)) --nav_policy "$V7_JIT" --nav_action_profile v7 --nav_target_offset_x -0.008 \
               --nav_handoff_x_range -0.045 0.015 --nav_max_steps 2000 --settle_steps 0 --start_polar 3.0 5.0 90 3.1416 ;;
    RESET_*) run "reset_$k" $((9610 + k)) ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "P8 COLLECT BC2 DONE ($STAGES)"
