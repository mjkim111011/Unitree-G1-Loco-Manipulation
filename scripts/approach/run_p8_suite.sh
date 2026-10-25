#!/usr/bin/env bash
# Phase-8 runs (Arena WBC, single env, trace + safety metrics).  Outputs: outputs/approach_p8.  No checkpoint or Teacher
# data is modified; Teacher runs write new HDF5 files next to their run directory.
#   Cause analysis (box fixed at the table centre unless noted; navigation = PPO v7 from 3-5 m, wide window, stable 50,
#   identical seed so every condition starts BC-1 from the same post-navigation state):
#     A_TEACHER        canonical reset start (no navigation, Box +-2 mm like the original corpus) + unmodified Teacher
#     B_TEACHER        navigation + unmodified Teacher
#     C_STUDENT        navigation + BC-1 v2
#     D_ALIGN_TEACHER  navigation + Method A alignment (to the Teacher's first target) + unmodified Teacher
#     E_ALIGN_STUDENT  navigation + Method A alignment + BC-1 v2
#     F_SAFE_TEACHER   navigation + Method B safe Teacher path
#     G_ALIGN_SAFE     navigation + Method A alignment + Method B safe Teacher path
#   Every condition continues with BC-2 v2 after a BC-1 gate.
# Usage: STAGES="A_TEACHER B_TEACHER" EPISODES=10 scripts/approach/run_p8_suite.sh   (stages skip existing outputs)
set -uo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/../..}"  # REPO_ROOT: run a frozen copy of this script from another directory
source packaging/activate.sh
STAGES=${STAGES:-""}
ROOT=${ROOT:-outputs/approach_p8}
LOGS=$ROOT/logs
mkdir -p "$LOGS" "$ROOT/cause"
EPISODES=${EPISODES:-10}
SEED=${SEED:-8102}
MARGIN=${MARGIN:-0.04}        # Method B: clearance of the predicted realized open hands (open loop; smoke test: 2 cm -> 1 mm realized)
ALIGN_MARGIN=${ALIGN_MARGIN:-0.02}  # Method A: alignment path, realized wrists follow it through the wrist servo
MARGIN_LATE=${MARGIN_LATE:-$MARGIN}  # Method B: pre-grasp margin (last 30 steps before the close onset)
SERVO=${SERVO:-0.3}
V7_JIT=outputs/approach_p5/eval/v7_nav_policy.pt
BC1_V2=outputs/approach_p1/checkpoints/bc1_v2_handoff_s01_r3.pt
BC2_V2=outputs/approach_p1/checkpoints/bc2_v2_from_bc1.pt
NAV=(--nav_policy "$V7_JIT" --nav_action_profile v7 --nav_target_offset_x -0.008 --nav_handoff_x_range -0.045 0.015 --nav_max_steps 2000
     --settle_steps 0 --start_polar 3.0 5.0 90 3.1416)
BASE=(--camera_profile collector --hand_state_decoding binary --episode_seconds 6 --pre_upper_body nominal --no_step_log --headless
      --device cuda:0 --nav_trace --safety_metrics --then_bc2 "$BC2_V2" --bc2_seconds 10 --safe_margin "$MARGIN"
      --align_margin "$ALIGN_MARGIN" --wrist_servo_gain "$SERVO" --safe_margin_late "$MARGIN_LATE")
BOX_FIXED=(--box_x_range 0.62 0.62 --box_y_range 0.0 0.0)

run() {  # name, args...
  local name=$1; shift
  [[ -f $ROOT/$name/summary.json ]] && { echo "skip $name"; return; }
  echo "$(date +%T) start $name"
  timeout 21600 python scripts/approach/eval_bc1_approach.py "${BASE[@]}" "$@" --output_dir "$ROOT/$name" > "$LOGS/${name//\//_}.log" 2>&1
  echo "$(date +%T) $name exit=$? $(grep -c '\[eval\] episode' "$LOGS/${name//\//_}.log") episodes"
}
teacher() {  # name, args...
  local name=$1; shift
  run "$name" --mode teacher_collect --collect_output "$ROOT/$name.hdf5" "$@"
}

for stage in $STAGES; do
  case $stage in
    A_TEACHER)       teacher cause/A_teacher_reset_seed8101 --seed 8101 --episodes "$EPISODES" ;;
    B_TEACHER)       teacher cause/B_teacher_nav_seed$SEED "${NAV[@]}" "${BOX_FIXED[@]}" --seed "$SEED" --episodes "$EPISODES" ;;
    C_STUDENT)       run cause/C_bc1v2_nav_seed$SEED --checkpoint "$BC1_V2" "${NAV[@]}" "${BOX_FIXED[@]}" --seed "$SEED" --episodes "$EPISODES" ;;
    D_ALIGN_TEACHER) teacher cause/D_align_teacher_nav_seed$SEED "${NAV[@]}" "${BOX_FIXED[@]}" --pre_bc1_align teacher_start --seed "$SEED" --episodes "$EPISODES" ;;
    E_ALIGN_STUDENT) run cause/E_align_bc1v2_nav_seed$SEED --checkpoint "$BC1_V2" "${NAV[@]}" "${BOX_FIXED[@]}" --pre_bc1_align teacher_start --seed "$SEED" --episodes "$EPISODES" ;;
    F_SAFE_TEACHER)  teacher cause/F_safe_teacher_nav_seed$SEED "${NAV[@]}" "${BOX_FIXED[@]}" --teacher_path safe --seed "$SEED" --episodes "$EPISODES" ;;
    G_ALIGN_SAFE)    teacher cause/G_align_safe_teacher_nav_seed$SEED "${NAV[@]}" "${BOX_FIXED[@]}" --pre_bc1_align teacher_start --teacher_path safe --seed "$SEED" --episodes "$EPISODES" ;;
    F_SAFE_TEACHER_BOX[123])  # Method B with the Box moved into the table (navigation with the phase-7 table-edge safe stop)
                     cm=${stage#F_SAFE_TEACHER_BOX}
                     teacher cause/F_safe_teacher_nav_box+${cm}cm_seed$SEED "${NAV[@]}" --table_safe_gap 0.41 --table_safe_margin 0.016 \
                       --box_x_range $(python3 -c "print(0.62 + $cm / 100)") $(python3 -c "print(0.62 + $cm / 100)") --box_y_range 0.0 0.0 \
                       --teacher_path safe --seed "$SEED" --episodes "$EPISODES" ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "P8 SUITE DONE ($STAGES)"
