#!/usr/bin/env bash
# Phase 9: controlled experiments on the BC-1 -> BC-2 handoff condition (no training; frozen PPO v6/v7, BC-1 v3.1, BC-2 v3.1).
# Every run: Arena WBC, BC-1 v3.1 -> BC-2 v3.1, trace + safety metrics, --reseed_each_episode (an episode's start does not
# depend on the previous episodes, so the same episode id is the same start under every gate variant).
# Conditions (STAGES "<COND>_<VARIANT>[_<first>-<last>]", episode ids optional):
#   V6FAR   PPO v6 navigation, target dx -2.5 cm, window [-4.5, -0.5] cm, nominal Box, seed 4002 (early gates 4/20 in phase 8)
#   WIN     deployment end condition: PPO v7, target -2.5 cm, window [-6, -0.5] cm, nominal Box, seed 4002
# Gate variants:
#   G0      existing gate (handoff at the first step it holds)
#   CAGE    existing gate AND cage width <= 29 cm (BC-1 keeps acting), at most 30 steps after the gate first opened
#   FIX     existing gate, handoff 10 steps after it first opened (BC-1 keeps acting; fixed delay control)
#   FREEZE  existing --approach_ready_hold_steps 10 (command frozen at the first gate for 9 steps; naive delay control)
#   *S      stress versions (G0S, CAGES, FIXS, FREEZES): the existing gate with the fingertip limit loosened 0.18 -> 0.21 m,
#           which opens it about 12 steps early with the cage still open (offline replay of 70 traces: cage > 29 cm in 68/70),
#           i.e. the natural early-gate state on demand; the same loosened gate in every arm
# Usage: STAGES="V6FAR_G0_0-19 WIN_CAGE" scripts/approach/run_p9_gate.sh
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"
source packaging/activate.sh
STAGES=${STAGES:-""}
ROOT=${ROOT:-outputs/approach_p9/eval}
LOGS=${LOGS:-outputs/approach_p9/logs/eval}
mkdir -p "$ROOT" "$LOGS"
EPISODES=${EPISODES:-20}
BC1=outputs/approach_p8/checkpoints/bc1_v31_safe.pt
BC2=outputs/approach_p8/checkpoints/bc2_v31_safe.pt
V6=outputs/approach_p4/eval/v6_nav_policy.pt
V7=outputs/approach_p5/eval/v7_nav_policy.pt
BASE=(--checkpoint "$BC1" --then_bc2 "$BC2" --bc2_seconds 10 --camera_profile collector --hand_state_decoding binary --episode_seconds 6
      --pre_upper_body nominal --no_step_log --headless --device cuda:0 --nav_trace --safety_metrics --reseed_each_episode
      --nav_max_steps 2000 --settle_steps 0 --start_polar 3.0 5.0 90 3.1416 --seed 4002)
declare -A COND=([V6FAR]="--nav_policy $V6 --nav_action_profile v6 --nav_target_offset_x -0.025 --nav_handoff_x_range -0.045 -0.005"
                 [WIN]="--nav_policy $V7 --nav_action_profile v7 --nav_target_offset_x -0.025 --nav_handoff_x_range -0.06 -0.005")
STRESS="--handoff_tip_distance_limit_m 0.21"   # looser position gate: opens about 12 steps early with a wide cage (offline replay)
declare -A GATE=([G0]="" [CAGE]="--gate_max_cage_width_m 0.29 --gate_ready_max_wait_steps 30" [FIX]="--gate_min_open_steps 10"
                 [FREEZE]="--approach_ready_hold_steps 10"
                 [G0S]="$STRESS" [CAGES]="$STRESS --gate_max_cage_width_m 0.29 --gate_ready_max_wait_steps 30"
                 [FIXS]="$STRESS --gate_min_open_steps 10" [FREEZES]="$STRESS --approach_ready_hold_steps 10")

for stage in $STAGES; do
  IFS=_ read -r cond variant range <<< "$stage"
  [[ -n ${COND[$cond]:-} && -v GATE[$variant] ]] || { echo "unknown stage $stage"; continue; }
  name=${cond,,}_${variant,,}_seed4002
  ids=()
  if [[ -n ${range:-} ]]; then
    name+=_ep$range
    ids=(--episode_ids $(seq "${range%-*}" "${range#*-}"))
  else
    ids=(--episodes "$EPISODES")
  fi
  [[ -f $ROOT/$name/summary.json ]] && { echo "skip $name"; continue; }
  [[ -e $ROOT/$name ]] && mv "$ROOT/$name" "$ROOT/$name.failed.$(date +%s)"
  echo "$(date +%T) start $name"
  # shellcheck disable=SC2086
  timeout 36000 python scripts/approach/eval_bc1_approach.py "${BASE[@]}" ${COND[$cond]} ${GATE[$variant]} "${ids[@]}" --output_dir "$ROOT/$name" \
    > "$LOGS/$name.log" 2>&1
  echo "$(date +%T) $name exit=$? $(grep -c '\[eval\] episode' "$LOGS/$name.log") episodes"
done
echo "P9 GATE DONE ($STAGES)"
