#!/usr/bin/env bash
# Phase-8 final tables and figure (CPU only): BC-1 v2 + BC-2 v2 (existing runs, same seeds) against
# BC-1 v3.1 + BC-2 v3.1 (run_p8_v31_pipeline.sh evaluation, run names *_v31).  Missing runs are skipped.
# Output: outputs/approach_p8/reports/eval_v31_summary.json, eval_v31_episodes.json, figures/fig_p8_eval_v31.png
set -uo pipefail
cd "$(dirname "$0")/../.."
P6=outputs/approach_p6/eval
P7=outputs/approach_p7/eval
E=outputs/approach_p8/eval
args=() pairs=() conds=() dirs=()
add() {  # condition, v2 dir, v31 dir
  local cond=$1 v2=$2 v31=$3
  [[ -f $v31/summary.json ]] || { echo "skip $cond (no $v31)"; return; }
  args+=(--run "$cond | v31=$v31"); dirs+=("$v31"); conds+=("$cond")
  if [[ -f $v2/summary.json ]]; then args+=(--run "$cond | v2=$v2"); pairs+=(--pair "$cond | v2:$cond | v31"); fi
}
add "nominal (v7 nav)" $P6/e1_v7_wide_seed4002 $E/nominal_v3_seed4002_v31
add "nominal (v6 nav)" $P6/e1_v6_wide_seed4002 $E/nominal_v6nav_v3_seed4002_v31
add "cause starts" outputs/approach_p8/cause/C_bc1v2_nav_seed8102 $E/cause_seed8102_v3_v31
add "reset start" $E/reset_v2_seed4007 $E/reset_v3_seed4007_v31
add "box +1 cm" $P7/v7safe_box+1cm_seed4003 $E/box+1cm_v3_seed4003_v31
add "box +2 cm" $P7/v7safe_box+2cm_seed4003 $E/box+2cm_v3_seed4003_v31
add "box +3 cm" $P7/v7safe_box+3cm_seed4003 $E/box+3cm_v3_seed4003_v31
add "box -2 cm (toward)" $E/toward2cm_v2_seed4004 $E/toward2cm_v3_seed4004_v31
add "box left 2 cm" $E/lat+2cm_v2_seed4005 $E/lat+2cm_v3_seed4005_v31
add "box right 2 cm" $E/lat-2cm_v2_seed4006 $E/lat-2cm_v3_seed4006_v31
add "nominal, nav window -4.5..-0.5 cm (v7 nav)" "" $E/narrow_v3_seed4002_v31
add "nominal, nav window -4.5..-0.5 cm (v6 nav)" "" $E/narrow_v6nav_v3_seed4002_v31
add "nominal, nav window -6..-0.5 cm, target -2.5 (v7 nav)" "" $E/win_v3_seed4002_v31
add "nominal, nav window -6..-0.5 cm, target -1.0 (v6 nav)" "" $E/win_v6nav_v3_seed4002_v31
[[ -f $E/narrow_v3_seed4002_v31/summary.json && -f $E/nominal_v3_seed4002_v31/summary.json ]] && \
  pairs+=(--pair "nominal (v7 nav) | v31:nominal, nav window -4.5..-0.5 cm (v7 nav) | v31")
[[ -f $E/narrow_v6nav_v3_seed4002_v31/summary.json && -f $E/nominal_v6nav_v3_seed4002_v31/summary.json ]] && \
  pairs+=(--pair "nominal (v6 nav) | v31:nominal, nav window -4.5..-0.5 cm (v6 nav) | v31")
for dx in -6 -4.5 -3 0 1.5; do
  d=$E/sweep_dx${dx}cm_v3_seed4008_v31
  [[ -f $d/summary.json ]] && { args+=(--run "sweep dx $dx cm | v31=$d"); dirs+=("$d"); }
done
# intermediate BC-1 v3 (+ BC-2 v2) on the cause starts, for the record
[[ -f $E/cause_seed8102_v3/summary.json ]] && args+=(--run "cause starts | v3+bc2v2=$E/cause_seed8102_v3")
python3 scripts/approach/summarize_p8_runs.py "${args[@]}" "${pairs[@]}" --output outputs/approach_p8/reports/eval_v31_summary.json || exit 1
python3 scripts/approach/summarize_p8_episodes.py "${dirs[@]}" --json outputs/approach_p8/reports/eval_v31_episodes.json > outputs/approach_p8/reports/eval_v31_episodes.txt || exit 1
plotconds=(); for c in "${conds[@]}"; do [[ $c == *"nav window"* ]] || plotconds+=("$c"); done  # window runs: tables only
python3 scripts/approach/plot_p8_eval.py --summary outputs/approach_p8/reports/eval_v31_summary.json --conditions "${plotconds[@]}" \
  --policies v2 v31 --names "BC-1 v2 + BC-2 v2" "BC-1 v3.1 + BC-2 v3.1" --output outputs/approach_p8/figures/fig_p8_eval_v31.png
python3 scripts/approach/render_p8_tables.py --summary outputs/approach_p8/reports/eval_v31_summary.json \
  --output outputs/approach_p8/reports/eval_v31_tables.md > /dev/null || exit 1
echo "written outputs/approach_p8/reports/eval_v31_{summary.json,episodes.json,episodes.txt,tables.md}, figures/fig_p8_eval_v31.png"
# grasp readiness at the BC-1 end (state BC-2 starts from)
hargs=()
for i in "${!conds[@]}"; do hargs+=(--run "${conds[$i]} | v31=${dirs[$i]}"); done
python3 scripts/approach/analyze_p6_handoff.py "${hargs[@]}" --run "nominal (v7 nav) | v2=$P6/e1_v7_wide_seed4002" --run "box +3 cm | v2=$P7/v7safe_box+3cm_seed4003" \
  --output outputs/approach_p8/reports/eval_v31_handoff.json > /dev/null || exit 1
python3 scripts/approach/render_p8_readiness.py --handoff outputs/approach_p8/reports/eval_v31_handoff.json --output outputs/approach_p8/reports/eval_v31_readiness.md > /dev/null || exit 1
echo "written outputs/approach_p8/reports/eval_v31_handoff.json, eval_v31_readiness.md"
