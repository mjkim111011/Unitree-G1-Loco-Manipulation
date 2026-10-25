#!/usr/bin/env bash
# Phase-8 GPU chain: waits for running processes, then starts two queues (at most two Isaac processes).  The queue scripts
# are executed from frozen copies (copied at start), so editing the repository scripts cannot change running queues.
#   QUEUE_A / QUEUE_B: ';'-separated commands of the form "suite STAGES" / "collect STAGES" / "eval STAGES"
# Usage: WAIT_PIDS="123 456" MARGIN_LATE=0.03 QUEUE_A="suite F_SAFE_TEACHER_BOX3;collect SHORT_0 RESET_0" \
#        QUEUE_B="suite F_SAFE_TEACHER_BOX2;collect SHORT_1" scripts/approach/run_p8_chain.sh
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"  # REPO_ROOT: run a frozen copy of this script from another directory
export REPO_ROOT=$(pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOGS=${CHAIN_LOGS:-outputs/approach_p8/logs}   # where the queue logs and the frozen script copies go
FROZEN=$CHAIN_LOGS/frozen_scripts/$STAMP
mkdir -p "$FROZEN"
cp scripts/approach/run_p8_suite.sh scripts/approach/run_p8_collect.sh scripts/approach/run_p8_eval.sh scripts/approach/run_p8_collect_bc2.sh "$FROZEN/"
echo "$(date +%T) frozen scripts in $FROZEN"
for pid in ${WAIT_PIDS:-}; do
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
done
echo "$(date +%T) waited for ${WAIT_PIDS:-nothing}"
run_queue() {  # name, spec
  local name=$1 spec=$2
  IFS=';' read -ra items <<< "$spec"
  for item in "${items[@]}"; do
    kind=${item%% *}; stages=${item#* }
    episodes=""; [[ $kind == *:* ]] && { episodes=${kind#*:}; kind=${kind%%:*}; }  # e.g. "collect:20 PPO_0"
    case $kind in
      suite)   STAGES="$stages" EPISODES=${episodes:-${SUITE_EPISODES:-10}} bash "$FROZEN/run_p8_suite.sh" ;;
      collect) STAGES="$stages" EPISODES=${episodes:-${COLLECT_EPISODES:-40}} bash "$FROZEN/run_p8_collect.sh" ;;
      eval)    STAGES="$stages" EPISODES=${episodes:-${EVAL_EPISODES:-20}} bash "$FROZEN/run_p8_eval.sh" ;;
      collect_bc2) STAGES="$stages" EPISODES=${episodes:-${COLLECT_EPISODES:-40}} bash "$FROZEN/run_p8_collect_bc2.sh" ;;
      *) echo "unknown queue item $item" ;;
    esac
  done
  echo "$(date +%T) QUEUE $name DONE"
}
run_queue A "${QUEUE_A:-}" > "$CHAIN_LOGS/queue_A_$STAMP.log" 2>&1 &
sleep 30
run_queue B "${QUEUE_B:-}" > "$CHAIN_LOGS/queue_B_$STAMP.log" 2>&1 &
wait
echo "$(date +%T) CHAIN DONE"
