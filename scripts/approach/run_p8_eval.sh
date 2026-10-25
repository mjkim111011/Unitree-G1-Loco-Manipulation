#!/usr/bin/env bash
# Phase-8 evaluation of BC-1 v3 (collision-free Teacher demos) against BC-1 v2 under identical conditions.
# Every run: Arena WBC, PPO v7 navigation (frozen), BC-1 under test, BC-2 v2 (frozen), trace + safety metrics.
#   NOMINAL_<pol>      Box at the table centre (+-2 mm), seed 4002 = phase-6 experiment-1 starts (v7, wide window)
#   NOMV6_<pol>        as NOMINAL with PPO v6 navigation (the deployment baseline; phase-6 E1V6 starts)
#   NARROW_<pol>       as NOMINAL (same starts) with a navigation end condition kept off the table edge: target dx -2.5 cm,
#                      handoff window [-4.5, -0.5] cm (navigation clearance shrinks for handoffs at dx >= 0)
#   NARROWV6_<pol>     as NARROW with PPO v6 navigation
#   WIN_<pol>          recommended end condition from the phase-8 sweep: handoff window [-6, -0.5] cm (BC-1 v3.1 is contact-free
#                      for starts at dx -6..0 cm, the navigation itself touches the table for dx > -0.5 cm), v7 target -2.5 cm
#   WINV6_<pol>        same window with PPO v6, target -1.0 cm (v6 stops about 1.5-2 cm behind its target)
#   CAUSE_<pol>        Box fixed at the table centre, seed 8102 = cause-analysis starts (B Teacher / C BC-1 v2 / F safe Teacher)
#   BOX<cm>_<pol>      Box moved <cm> cm into the table, seed 4003 = phase-6/7 starts, table-edge safe stop (gap 0.41)
#   TOWARD2_<pol>      Box 2 cm towards the robot, seed 4004
#   LAT<+|->2_<pol>    Box 2 cm to the left / right, seed 4005 / 4006
#   RESET_<pol>        canonical reset start (no navigation; the original corpus start), seed 4007
#   SWEEP<dx>_<pol>    BC-1 start region: PPO v7 navigation to a target dx (cm) with a 2 cm window, nominal Box, seed 4008
#   <pol> = V2 (bc1_v2_handoff_s01_r3.pt) or V3 (BC1_V3 checkpoint)
# Usage: BC1_V3=outputs/approach_p8/checkpoints/bc1_v3_safe.pt STAGES="NOMINAL_V3 BOX2_V3" EPISODES=20 scripts/approach/run_p8_eval.sh
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"  # REPO_ROOT: run a frozen copy of this script from another directory
source packaging/activate.sh
STAGES=${STAGES:-""}
ROOT=${ROOT:-outputs/approach_p8/eval}
LOGS=${LOGS:-outputs/approach_p8/logs/eval}
mkdir -p "$ROOT" "$LOGS"
EPISODES=${EPISODES:-20}
V7_JIT=outputs/approach_p5/eval/v7_nav_policy.pt
V6_JIT=outputs/approach_p4/eval/v6_nav_policy.pt
BC2_V2=outputs/approach_p1/checkpoints/bc2_v2_from_bc1.pt
declare -A CKPT=([V2]=outputs/approach_p1/checkpoints/bc1_v2_handoff_s01_r3.pt [V3]=${BC1_V3:-outputs/approach_p8/checkpoints/bc1_v3_safe.pt})
NAV=(--nav_policy "$V7_JIT" --nav_action_profile v7 --nav_max_steps 2000 --settle_steps 0 --start_polar 3.0 5.0 90 3.1416)
WIDE=(--nav_target_offset_x -0.008 --nav_handoff_x_range -0.045 0.015)
SAFE=(--table_safe_gap 0.41 --table_safe_margin 0.016)
BASE=(--camera_profile collector --hand_state_decoding binary --episode_seconds 6 --pre_upper_body nominal --no_step_log --headless
      --device cuda:0 --nav_trace --safety_metrics --then_bc2 "$BC2_V2" --bc2_seconds 10)
GATE_TIP_MAX=${GATE_TIP_MAX:-}   # V3 runs: grasp-readiness gate (m); empty = the existing BC-1 -> BC-2 gate
SUFFIX=${NAME_SUFFIX:-}          # V3 runs: appended to the run name (e.g. _rg45 or _v3v3)
BC2_V3=${BC2_V3:-}               # V3 runs: BC-2 checkpoint after BC-1 v3 (default BC-2 v2)
EXTRA_V3=${EXTRA_V3:-}           # V3 runs: extra evaluator arguments (phase 10: the cage readiness gate)
# V2 runs always use the existing gate, BC-2 v2 and no name suffix (the readiness gate is a no-op for BC-1 v2).

run() {  # name, policy, args...
  local name=$1 pol=$2; shift 2
  local extra=()
  if [[ $pol == V3 ]]; then
    name=$name$SUFFIX
    [[ -n $GATE_TIP_MAX ]] && extra+=(--gate_max_tip_height_m "$GATE_TIP_MAX")
    [[ -n $BC2_V3 ]] && extra+=(--then_bc2 "$BC2_V3")
    # shellcheck disable=SC2206
    [[ -n $EXTRA_V3 ]] && extra+=($EXTRA_V3)
  fi
  [[ -f $ROOT/$name/summary.json ]] && { echo "skip $name"; return; }
  [[ -e $ROOT/$name ]] && mv "$ROOT/$name" "$ROOT/$name.failed.$(date +%s)"
  [[ -f ${CKPT[$pol]} ]] || { echo "missing checkpoint for $pol: ${CKPT[$pol]}"; return; }
  echo "$(date +%T) start $name"
  timeout 36000 python scripts/approach/eval_bc1_approach.py "${BASE[@]}" "${extra[@]}" --checkpoint "${CKPT[$pol]}" "$@" --output_dir "$ROOT/$name" \
    > "$LOGS/$name.log" 2>&1
  echo "$(date +%T) $name exit=$? $(grep -c '\[eval\] episode' "$LOGS/$name.log") episodes"
}
box() { python3 -c "c = $1; print(f'{c - 0.002:.4f} {c + 0.002:.4f}')"; }

for stage in $STAGES; do
  pol=${stage##*_}
  case $stage in
    NOMINAL_*) run nominal_${pol,,}_seed4002 "$pol" "${NAV[@]}" "${WIDE[@]}" --seed 4002 --episodes "$EPISODES" ;;
    NARROW_*)  run narrow_${pol,,}_seed4002 "$pol" "${NAV[@]}" --nav_target_offset_x -0.025 --nav_handoff_x_range -0.045 -0.005 \
                 --seed 4002 --episodes "$EPISODES" ;;
    NARROWV6_*) run narrow_v6nav_${pol,,}_seed4002 "$pol" --nav_policy "$V6_JIT" --nav_action_profile v6 --nav_max_steps 2000 --settle_steps 0 \
                 --start_polar 3.0 5.0 90 3.1416 --nav_target_offset_x -0.025 --nav_handoff_x_range -0.045 -0.005 --seed 4002 --episodes "$EPISODES" ;;
    WIN_*)     run win_${pol,,}_seed4002 "$pol" "${NAV[@]}" --nav_target_offset_x -0.025 --nav_handoff_x_range -0.06 -0.005 \
                 --seed 4002 --episodes "$EPISODES" ;;
    WINV6_*)   run win_v6nav_${pol,,}_seed4002 "$pol" --nav_policy "$V6_JIT" --nav_action_profile v6 --nav_max_steps 2000 --settle_steps 0 \
                 --start_polar 3.0 5.0 90 3.1416 --nav_target_offset_x -0.010 --nav_handoff_x_range -0.06 -0.005 --seed 4002 --episodes "$EPISODES" ;;
    NOMV6_*)   run nominal_v6nav_${pol,,}_seed4002 "$pol" --nav_policy "$V6_JIT" --nav_action_profile v6 --nav_max_steps 2000 --settle_steps 0 \
                 --start_polar 3.0 5.0 90 3.1416 "${WIDE[@]}" --seed 4002 --episodes "$EPISODES" ;;
    CAUSE_*)   # same starts as the cause-analysis runs B/C/F (seed 8102, Box fixed at the table centre)
               run cause_seed8102_${pol,,} "$pol" "${NAV[@]}" "${WIDE[@]}" --seed 8102 --episodes "$EPISODES" --box_x_range 0.62 0.62 --box_y_range 0.0 0.0 ;;
    BOX[123]_*) cm=${stage%%_*}; cm=${cm#BOX}
               run box+${cm}cm_${pol,,}_seed4003 "$pol" "${NAV[@]}" "${WIDE[@]}" "${SAFE[@]}" --seed 4003 --episodes "$EPISODES" \
                 --box_x_range $(box "0.62 + $cm / 100") ;;
    TOWARD2_*) run toward2cm_${pol,,}_seed4004 "$pol" "${NAV[@]}" "${WIDE[@]}" "${SAFE[@]}" --seed 4004 --episodes "$EPISODES" \
                 --box_x_range $(box 0.60) ;;
    LAT+2_*)   run lat+2cm_${pol,,}_seed4005 "$pol" "${NAV[@]}" "${WIDE[@]}" "${SAFE[@]}" --seed 4005 --episodes "$EPISODES" \
                 --box_y_range $(box 0.02) ;;
    LAT-2_*)   run lat-2cm_${pol,,}_seed4006 "$pol" "${NAV[@]}" "${WIDE[@]}" "${SAFE[@]}" --seed 4006 --episodes "$EPISODES" \
                 --box_y_range $(box -0.02) ;;
    RESET_*)   run reset_${pol,,}_seed4007 "$pol" --seed 4007 --episodes "$EPISODES" ;;
    SWEEP*_*)  dx=${stage%%_*}; dx=${dx#SWEEP}
               lo=$(python3 -c "print(($dx - 1.0) / 100)"); hi=$(python3 -c "print(($dx + 1.0) / 100)"); tgt=$(python3 -c "print($dx / 100)")
               run sweep_dx${dx}cm_${pol,,}_seed4008 "$pol" "${NAV[@]}" --nav_target_offset_x "$tgt" --nav_handoff_x_range "$lo" "$hi" \
                 --seed 4008 --episodes "$EPISODES" ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "P8 EVAL DONE ($STAGES)"
