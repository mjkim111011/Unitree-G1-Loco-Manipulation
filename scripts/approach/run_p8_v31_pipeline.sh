#!/usr/bin/env bash
# Phase-8 v3.1 pipeline: the BC-1 v3 Teacher demos used a 3.0-3.5 cm pre-grasp margin (chosen for BC-2 v2's lift height).
# The realized hands then end 0.6 cm (median) above the table at the close onset and 44/240 demos touch it with the right
# middle finger tip at steps 120-125, so BC-1 v3 learned to end millimetres above the table.  With BC-2 v3 trained on its
# own continuations, the pre-grasp margin can be raised (MARGIN_LATE, default 0.07).  Same pipeline as v3, new paths only:
#   1 Teacher demos, same stages and seeds as BC-1 v3 (run_p8_collect.sh)            -> outputs/approach_p8/collect_v31
#   2 select: no table contact over the whole recorded approach (until close + 5), clearance >= SEL_CLR (recorded scope)
#   3 train BC-1 v3.1                                                                    -> checkpoints/bc1_v31_safe.pt
#   4 BC-2 data: BC-1 v3.1 -> safe Teacher continuation (same late margin)             -> outputs/approach_p8/collect_bc2_v31
#   5 select (phase bc2) and train BC-2 v3.1 (labels from the handoff on)              -> checkpoints/bc2_v31_safe.pt
#   6 evaluation queues (run_p8_eval.sh, BC-1 v3.1 + BC-2 v3.1, run names *_v31; BC-1 v2 runs where missing)
# Every stage runs through run_p8_chain.sh (two Isaac processes at most, frozen script copies).  Stops on any failure.
# The chain runner is called from a frozen copy (FROZEN below), so the repository scripts can be edited while this runs;
# run this script itself from a frozen copy too (REPO_ROOT=<repo> bash <copy>).
# Usage: WAIT_PIDS="<pid>" MARGIN_LATE=0.07 REPO_ROOT=$(pwd) nohup bash <frozen copy> > outputs/approach_p8/logs/v31_pipeline.log 2>&1 &
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"
export REPO_ROOT=$(pwd)
source packaging/activate.sh
FROZEN=outputs/approach_p8/logs/frozen_scripts/v31_pipeline_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FROZEN"
cp scripts/approach/run_p8_chain.sh "$FROZEN/"
CHAIN="bash $FROZEN/run_p8_chain.sh"
echo "$(date +%T) frozen chain runner in $FROZEN"
export MARGIN_LATE=${MARGIN_LATE:-0.07}
SEL_CLR=${SEL_CLR:-0.01}
SEL_CLR_BC2=${SEL_CLR_BC2:-0.005}
P8=outputs/approach_p8
BC1=$P8/checkpoints/bc1_v31_safe.pt
BC2=$P8/checkpoints/bc2_v31_safe.pt
for pid in ${WAIT_PIDS:-}; do
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
echo "$(date +%T) start (MARGIN_LATE=$MARGIN_LATE)"
complete() {  # root, run names...: every run has summary.json and its HDF5
  local root=$1; shift
  for name in "$@"; do [[ -f $root/$name/summary.json && -f $root/$name.hdf5 ]] || { echo "incomplete $root/$name"; return 1; }; done
}

# 1-3 BC-1 v3.1
if [[ ! -f $BC1 ]]; then
  ROOT=$P8/collect_v31 LOGS=$P8/logs/collect_v31 COLLECT_EPISODES=40 \
    QUEUE_A="collect SHORT_0 SHORT_2;collect:20 PPO_0;collect RESET_0" QUEUE_B="collect SHORT_1 SHORT_3;collect:20 PPO_1" $CHAIN
  complete $P8/collect_v31 short_nav_0 short_nav_1 short_nav_2 short_nav_3 ppo_nav_0 ppo_nav_1 reset_0 || exit 1
  python scripts/approach/select_safe_demos.py --scope recorded --min_clearance_m "$SEL_CLR" \
    --run $P8/collect_v31/short_nav_0 $P8/collect_v31/short_nav_1 $P8/collect_v31/short_nav_2 $P8/collect_v31/short_nav_3 \
          $P8/collect_v31/ppo_nav_0 $P8/collect_v31/ppo_nav_1 $P8/collect_v31/reset_0 \
    --output $P8/reports/accepted_v31.json || exit 1
  python scripts/approach/train_bc1_safe.py --accepted $P8/reports/accepted_v31.json --output "$BC1" --device cpu --epochs 400 --patience 40 || exit 1
  echo "$(date +%T) trained $BC1"
fi

# 4-5 BC-2 v3.1
if [[ ! -f $BC2 ]]; then
  ROOT=$P8/collect_bc2_v31 LOGS=$P8/logs/collect_bc2_v31 BC1_V3=$BC1 COLLECT_EPISODES=40 \
    QUEUE_A="collect_bc2 SHORT_0 SHORT_2" QUEUE_B="collect_bc2 SHORT_1;collect_bc2:20 PPO_0 RESET_0" $CHAIN
  complete $P8/collect_bc2_v31 short_nav_0 short_nav_1 short_nav_2 ppo_nav_0 reset_0 || exit 1
  python scripts/approach/select_safe_demos.py --phase bc2 --min_clearance_m "$SEL_CLR_BC2" \
    --run $P8/collect_bc2_v31/short_nav_0 $P8/collect_bc2_v31/short_nav_1 $P8/collect_bc2_v31/short_nav_2 \
          $P8/collect_bc2_v31/ppo_nav_0 $P8/collect_bc2_v31/reset_0 \
    --output $P8/reports/accepted_bc2_v31.json || exit 1
  python scripts/approach/train_bc1_safe.py --accepted $P8/reports/accepted_bc2_v31.json --output "$BC2" --skill grasp_lift \
    --bc2-labels from_handoff --device cpu --epochs 400 --patience 40 || exit 1
  echo "$(date +%T) trained $BC2"
fi

# 6 evaluation: the first item of each queue is a quick look (cause starts / reset start) before the long conditions
[[ ${SKIP_EVAL:-0} == 1 ]] && { echo "$(date +%T) V31 PIPELINE DONE (no evaluation)"; exit 0; }
BC1_V3=$BC1 BC2_V3=$BC2 NAME_SUFFIX=_v31 EVAL_EPISODES=20 \
QUEUE_A="eval:10 CAUSE_V3;eval NOMINAL_V3;eval BOX2_V3;eval:10 LAT+2_V3;eval:10 LAT+2_V2;eval:10 TOWARD2_V3;eval:8 SWEEP-6_V3;eval:8 SWEEP-3_V3;eval:8 SWEEP0_V3" \
QUEUE_B="eval RESET_V3;eval BOX1_V3;eval BOX3_V3;eval:10 TOWARD2_V2;eval:10 LAT-2_V3;eval:10 LAT-2_V2;eval NOMV6_V3;eval:8 SWEEP-4.5_V3;eval:8 SWEEP1.5_V3" \
  $CHAIN
echo "$(date +%T) V31 PIPELINE DONE"
