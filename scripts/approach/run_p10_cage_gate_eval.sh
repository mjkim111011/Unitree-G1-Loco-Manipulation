#!/usr/bin/env bash
# Phase 10 (not run yet): full re-evaluation of the cage readiness gate on the 10
# phase-8 conditions (160 episodes), paired with the stored phase-8 baseline (existing gate, strict collision-free E2E
# 149/160).  Run it as is (about 6 h on two Isaac processes):
#
#   nohup scripts/approach/run_p10_cage_gate_eval.sh > outputs/approach_p10/logs/cage_gate_eval.log 2>&1 &
#
# then compare episode by episode against the stored baseline (CPU):
#
#   python3 scripts/approach/summarize_p10_cage_gate.py --output outputs/approach_p10/reports/p10_cage_gate.json
#
# Design (why the stored baseline can be reused):
#   - The runs repeat the phase-8 runs exactly: run_p8_eval.sh stages with BC-1 v3.1 + BC-2 v3.1, the same seeds, episode
#     counts and order, no --reseed_each_episode; only the readiness options are added (EXTRA_V3).
#   - Episodes are therefore replayed exactly until the first episode in which the cage gate changes the behaviour (the
#     phase-10 pre-check replayed the first stored episodes of two conditions with the current evaluator); afterwards an
#     episode keeps its initial robot pose but its Box position (+-2 mm) and navigation can drift, which the comparison
#     script reports per episode ("exact" / "drifted", "gate acted" / drift).
#   - Readiness: existing gate AND cage width <= 0.29 m; if the cage has not closed 30 steps after the existing gate
#     first opened, BC-2 is NOT started: safe stop (wrists raised 3 cm over 10 steps, base velocity 0, held 40 steps),
#     outcome readiness_timeout (counted as a failure; any table contact during the stop counts as a BC-1 contact).
#   - Paired comparison with the exact McNemar test (per condition and pooled).
set -uo pipefail
cd "$(dirname "$0")/../.."
export BC1_V3=outputs/approach_p8/checkpoints/bc1_v31_safe.pt
export BC2_V3=outputs/approach_p8/checkpoints/bc2_v31_safe.pt
export NAME_SUFFIX=_v31cage
export EXTRA_V3="--gate_max_cage_width_m 0.29 --gate_ready_max_wait_steps 30 --gate_ready_timeout_action abort"
export ROOT=outputs/approach_p10/eval
export LOGS=outputs/approach_p10/logs/eval
export CHAIN_LOGS=outputs/approach_p10/logs   # queue logs and frozen script copies stay out of outputs/approach_p8
# same episode counts as the stored phase-8 runs: 70 + 90 = 160 episodes
QUEUE_A="eval:10 CAUSE_V3;eval NOMINAL_V3;eval BOX2_V3;eval:10 LAT+2_V3;eval:10 TOWARD2_V3" \
QUEUE_B="eval RESET_V3;eval BOX1_V3;eval BOX3_V3;eval:10 LAT-2_V3;eval NOMV6_V3" \
  scripts/approach/run_p8_chain.sh
