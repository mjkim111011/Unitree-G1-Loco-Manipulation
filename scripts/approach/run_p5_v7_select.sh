#!/usr/bin/env bash
# PPO v7 checkpoint selection (deterministic actor, batched Homie env, *selection* seed 2001 - the reported batched seed
# is 3001 and the Arena-WBC seed 4001, as in phases 2-4).  S3 = 3-5 m any heading, S4 = + Box offsets on the table.
# Usage: RUN=<v7 run dir> CKPTS="1500 1600 ..." scripts/approach/run_p5_v7_select.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source packaging/activate.sh
RUN=${RUN:?set RUN}
CKPTS=${CKPTS:?set CKPTS}
STAGES_SEL=${STAGES_SEL:-"2 3"}
B=outputs/approach_p5/eval/select
mkdir -p "$B"
for c in $CKPTS; do
  for st in $STAGES_SEL; do
    name=v7_m${c}_S$((st + 1))_seed2001
    [[ -f $B/$name.json ]] && { echo "skip $name"; continue; }
    echo "$(date +%T) start $name"
    timeout 3600 python scripts/approach/eval_nav_ppo.py --task Unitree-G1-43dof-BoxNav-HomieV7 --controller ppo --checkpoint "$RUN/model_$c.pt" \
      --stage "$st" --num_envs 256 --seed 2001 --nav_target_offset_x -0.008 --nav_handoff_x_range -0.03 0.015 \
      --headless --device cuda:0 --output "$B/$name.json" > outputs/approach_p5/logs/select_$name.log 2>&1
    echo "$(date +%T) $name exit=$?"
  done
done
echo SELECT_DONE
