# Arena teacher audit and nominal Box collection

The Arena checkout and this project use incompatible Isaac Sim/Lab versions.
Do not import them into one Python process. `arena_wbc_worker.py` runs Pink IK
and HomieV2 in Arena's `.venv`; `arena_wbc_bridge.py` exchanges only JSON
numeric state and named 43-DOF targets with the current Box environment.

## Production status (2026-08-27)

- The strict x/y-only corpus in `data/teacher/box_xy_randomized_500` is complete:
  500/500 accepted demonstrations in five audited shards.
- The promoted 23-D approach policy uses open-hand boundary DAgger and reaches
  the production wrist/fingertip handoff in the verified central rollout.
- The promoted 43-D grasp/lift policy adds 453 balanced handoff corrections to
  the strict corpus. It passed 19/20 independent physical lift/hold rollouts
  with 0% bad orientation and has a SHA-bound gate plus 19 physical snapshots.
- Carry now has 500 successful physical trajectories: the previous 100 plus
  400 new rollouts cycling 22 states captured from the actual Student
  approach-to-grasp chain. The production manifest records 500 successes from
  533 attempts across 14 SHA-256-bound shards.
- The Carry Student was retrained from all 500 episodes with the first 350
  recorded phase-boundary steps weighted 4x. It uses only the 517-D QR/proprio
  input and reached validation/test normalized MSE 0.000515/0.000544. This new
  checkpoint passed a 5/5 physical Carry smoke evaluation, but remains a
  candidate until its 20-episode Carry and full-FSM gates run.
- Place now has 600 successful physical trajectories. The previous 500 were
  retained, and 100 new successes were collected across five Carry-at-staging
  stores with 100 distinct source episode references. The combined manifest
  records 600 successes from 795 attempts and passes its publication gate.
- The diversified Place Student uses only the 517-D QR/proprio observation,
  retains proprioceptive history, and applies a non-privileged phase clock.
  It was trained from all 600 successes with the 100 diversified trajectories
  and phase-start samples up-weighted. Its disjoint 20-rollout physical check
  reached 9/20; the 45% result is recorded but does not pass the 80% gate.
- Place Student-state Teacher relabel collection is implemented and produced
  20 DAgger episodes. The first fine-tuned candidate regressed to 0/4 and was
  rejected, so the physically better 600-demo checkpoint remains the default
  candidate without being marked as production-gate passed.
- The latest completed full-FSM smoke evaluation still predates this Place
  checkpoint. A new 20-episode full-FSM evaluation remains required.

Smoke artifacts are not substitutes for the production paths above. The
authoritative executable sequence is also available as numbered entries in
`.vscode/launch.json`. Run `18 Audit: Current Loco-Manipulation Pipeline` to write
the current machine-readable status to `data/teacher/pipeline_status.json`.

## 1. Audit and retarget the supplied Arena episode

### Teacher demonstration collection at a glance

The source is an Arena G1 demonstration with 29 body joints and 14 Dex3 hand
joints (43 joints total). We do not copy its joint-angle sequence directly
into the current simulator. Instead, the collector:

1. extracts the Box path, both wrist targets, and the 23-D high-level command;
2. samples the Box initial `x/y` in the current
   `Unitree-G1-43dof-BoxPickPlace` environment;
3. applies the same translation from the Arena Box to the current Box to the
   Box, left-wrist, and right-wrist paths, preserving each wrist pose relative
   to the Box;
4. expresses the translated wrist targets in the live simulated pelvis frame;
5. uses Pink inverse kinematics for the arms/hands and the learned Homie policy
   for lower-body balance/locomotion, then maps the returned target by joint
   name into the current 43-joint action order; and
6. executes that target in the current physics environment and publishes only
   rollouts with bilateral contact, at least 12 cm lift, and a continuous
   100-step (2 s at 50 Hz) hold without a fall or drop.

The teacher may read the simulator's exact Box pose to retarget and label an
action. The student never receives that world pose: its 517-D input contains
proprioception plus compact camera-frame Box/Bin visibility and position
features. These camera features currently use simulator geometry to emulate
an ideal QR result; real RGB QR/AprilTag detection remains a sim-to-real
integration step. Carry and Place are not copied from the Arena episode:
Carry starts from a physically successful grasp snapshot in this environment,
and Place starts from a physically successful Carry-at-staging snapshot.

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/extract_arena_teacher.py \
  /home/mjkim/IsaacLab-Arena/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
  --output data/teacher/arena_g1_loco_manipulation_small_retarget.hdf5 \
  --audit-json data/teacher/arena_g1_loco_manipulation_small_audit.json \
  --overwrite
```

The camera-free output contains:

- raw 23-D high-level action and processed named 43-D joint target;
- base velocity, hip-height, and torso commands;
- left/right Dex3 targets;
- inferred approach/grasp/carry/place phase IDs and boundaries;
- realized and commanded world wrist poses;
- `T_box^-1 T_left_wrist`, `T_box^-1 T_right_wrist`, and
  `T_bin^-1 T_box`, plus phase-anchor deltas.

The supplied episode has no explicit phase/contact/Box-size fields. Its phase
labels are signal-based inference. Both hand-state channels and all 14 Dex3
targets remain zero/open for all 853 steps, so it must not be used directly as
closed-hand bimanual supervision.

## 2. Replay in Arena

Use Arena's own environment and the Kit visualizer:

```bash
cd /home/mjkim/IsaacLab-Arena
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \
  submodules/IsaacLab/scripts/tools/replay_demos.py \
  --viz kit --device cpu \
  --dataset_file data/arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
  --select_episodes 0 --validate_success_rate --reset_sim_buffer_each_episode \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task galileo_g1_locomanip_pick_and_place \
  --object brown_box --embodiment g1_wbc_pink
```

Action replay and Arena success validation pass `1/1`. `--validate_states`
does not pass because this HDF5's older recorder pose shape differs from the
installed Arena 0.3 state schema.

## 3. Collect nominal current-environment demonstrations

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/collect_nominal_box_teacher.py \
  --headless --device cuda:0 --num_trials 10 \
  --output data/teacher/current_box_nominal_teacher.hdf5 \
  --overwrite
```

This is deliberately a fixed `0.20 m`, `0.20 kg` nominal Box gate, not yet a
randomized full carry/place dataset. The driver:

- translates Arena wrist targets to the current fixed Box;
- writes the live camera/QR-derived pregrasp velocity command into both the
  517-D student observation and high-level action channels 16:19 during
  `APPROACH`, then writes exactly `[0, 0, 0]` to both from the BC-2 close onset;
- injects bilateral Dex3 close because the source hand signal is open;
- applies a gradual 5 cm lift-tail compensation while retaining the strict
  12 cm success threshold;
- allows up to 4 seconds for the strict continuous hold window to begin after
  freezing the final lift target;
- matches Arena's teacher-only PD/solver/self-collision/reset configuration;
- records the current student's 517-D proprio/QR input without RGB;
- records both 23-D high-level and absolute named 43-D actions.

This command split is part of the dataset contract. It prevents the actor from
seeing a stationary command while its approach teacher walks, and prevents a
Bin-navigation command from leaking into the stationary grasp/lift skill.

Each HDF5 step includes student observation, both teacher action levels,
phase, Box/Bin/root and wrist poses, fingertip positions, Box size, bilateral
contact counts, and lift/drop/success flags.

## 4. Independently validate schema and gates

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/validate_teacher_dataset.py \
  data/teacher/current_box_nominal_teacher.hdf5 \
  --output-json data/teacher/current_box_nominal_teacher.validation.json
```

The validator recomputes bilateral grasp/lift, the continuous 100-step
(2-second at 50 Hz) hold, root tilt, dimensions, finite values, and recorder
success agreement. It exits nonzero if schema or the 90%/5%/90% gates fail.

## 5. First curriculum: Box x/y-only data generation

The randomized collector keeps the following values fixed for the entire run:

- Box size, mass, static/dynamic friction, and restitution;
- table height and Box height on the table;
- Box yaw (identity), initial velocity, and robot reset state.

Only the Box's reset-time `x/y` is sampled. The reset event still computes
`z = TABLE_SURFACE_HEIGHT + box_half_height`, writes identity orientation, and
zeros velocity. All scale/material/mass/table-height randomizers and the
in-episode retry event remain disabled.

The initial default is deliberately conservative: `x=[0.618, 0.622] m` and
`y=[-0.002, 0.002] m` around the validated `(0.62, 0)` nominal pose. Widen this
range only after the generated teacher manifest continues to pass the 80%
gate; this is the first position curriculum, not the final domain range.

Retargeting calls Isaac Lab Mimic's
`get_delta_pose_with_scheme(..., TRANSLATE)` and
`transform_source_data_segment_using_delta_object_pose` for the source Box,
left wrist, and right wrist. `TRANSLATE` is intentional: the Arena source Box
and this task's Box use different semantic yaw conventions, so the generic
full-SE(3) transform flips an otherwise valid bimanual path. The validated
default preserves that transformed wrist trajectory and freezes the final
source lift target. Optional wrist compression is zero. An experimental
`--freeze-realized-hold-wrists` mode is available, but is disabled because live
validation showed that switching to the first realized wrist poses reduced the
strict continuous hold to only two steps. No success threshold was relaxed.

Use the resumable production driver for 500 demonstrations:

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/generate_randomized_box_teacher.py \
  --target_demos 500 --shard_size 100 \
  --output_dir data/teacher/box_xy_randomized_500
```

The same command can be re-run after interruption. It audits existing shards,
continues at the next shard, and updates `manifest.json` atomically. Every
shard must reach at least 80% live teacher success. Failed attempts are deleted
inside the streaming recorder, so only demonstrations satisfying continuous
bilateral Box-filtered contact, 12 cm lift, and a two-second hold are published.

Each accepted demo also has one `BILATERAL_HOLD` entry in the matching
`shard_XXXXX.snapshots.hdf5`. This snapshot comes directly from the realized
physics rollout and stores complete robot root/joint position and velocity,
Box state and velocity, last action, and the 517-D student observation. These
shard snapshots remain useful for corpus audit, but the carry collector no
longer accepts them as phase-start state. Carry may start only from the
`BILATERAL_HOLD` snapshot store emitted by the certified BC-2 physical
evaluator, so its loaded grasp is tied to the exact deployed BC-2 checkpoint.

The later-phase restore path is intentionally single-environment and restores
the saved 460-D `policy_history` as the exact five chronological frames in the
observation manager's circular buffers. It checks that the manager emits the
same flattened history before continuing. The action manager's saved last
action is restored separately; current proprio/QR values are then obtained
from the restored physical scene. A normal controlled physics step must still
retain bilateral contact and lift before the snapshot is accepted for data
collection.

The recorded student input is always `policy_history + policy_current` (517D,
QR/proprio only). Privileged Box/Bin/root/wrist/contact fields coexist only for
retargeting and audit, and the root metadata declares that boundary. This
GRASP/LIFT collector writes phase ID 0 for its full episode and resets the live
task phase back to grasp before recomputing commands/observations, preventing
the two-second hold tail from being mislabeled or contaminated by Bin carry
commands.

Only `policy_history` is temporally stacked: it is five frames (460D) of
proprioception. Both `policy_current` and the critic's `critic_current` use
`history_length=0`, so command/action/QR and privileged current-state features
are recomputed from the present frame even when a tool calls the observation
manager with `update_history=False`. The critic retains its separate five-frame
proprioceptive `critic_history`; current task fields are not stacked.

## 6. Train phase-wise BC policies (BC-2 first)

`phase_bc.py` has one hard actor boundary: the network receives only the
recorded 517-D `student_observation` (`policy_history + policy_current`). Box,
Bin, root, wrist, contact, and lift datasets remain available for audit and
phase labeling, but cannot be concatenated into a student input through this
loader. The checkpoint repeats this contract as metadata and the runtime loader
rejects checkpoints that declare privileged inputs.

Task phase 0 is split into two skills. The first hand-close command, real hand
contact, or lift flag begins `grasp_lift`; earlier samples are `approach`. Task
phase 1 maps to `carry`, task phase 2 maps to `place`, and complete steps are
excluded. Train/validation/test assignment is made by whole episode, so
transitions from one rollout never leak across splits.

The CLI includes two pre-close frames in BC-2. At runtime the APPROACH→BC-2
handoff requires a measured two-sided fingertip cage near the Box. Camera/base
`at_pregrasp` alignment is deliberately insufficient: it becomes true much
earlier than the demonstrated hand-close boundary and previously handed an
uncaged state to BC-2.

Train BC-2 first:

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/phase_bc.py train \
  data/teacher/box_xy_randomized_500 \
  --skill grasp_lift \
  --action-key processed_43d_joint_target \
  --output data/teacher/bc/bc2_grasp_lift_43d.pt \
  --device cuda:0
```

`processed_43d_joint_target` trains the directly deployable 43-D joint-target
policy. Selecting `teacher_high_level_action` instead trains the 23-D
wrist/base/Dex3 command and therefore still requires the Pink/Homie WBC bridge
at runtime.

Both targets have an exact ordered-channel action contract. Every contributing
HDF5 must declare the same 43 processed joint names or the same 23 high-level
channel names. Training writes `ordered_action_names` and a SHA-256 over the
canonical `(action_key, ordered names)` payload into the checkpoint. Loading
recomputes that hash, and the physical evaluator, DAgger collector, and FSM
compare the checkpoint order with the live action-manager/WBC order before an
action can be applied. Equal dimensionality alone is not accepted.

Corrective DAgger labels can be mixed without converting their flat schema:

```bash
# Add this option to any `phase_bc.py train` command.
--dagger-datasets data/teacher/dagger/corrections_round_01.hdf5 \
                  data/teacher/dagger/corrections_round_02.hdf5
```

When the input is the production shard directory or `manifest.json`, training
refuses to start until the manifest proves 500--1000 accepted demos, at least
80% aggregate live-teacher success, and the strict bilateral lift/hold gate.
`--allow-incomplete-manifest` exists only for deliberate diagnostic smoke tests
on partial shards; it must not be used for a production checkpoint.

The loader groups every correction file by `source_episode`. All corrective
steps from one physical rollout enter exactly one split, and normalization is
fit only after nominal and corrective training samples have been merged. The
student input is still only `/samples/student_observation`; correction tags and
teacher targets are labels, never actor features.

Offline held-out error is not enough to validate BC-2. The gate accepts only
the full physical-rollout schema produced by `evaluate_bc2_rollout.py`; a
hand-written JSON containing only success and orientation scalars is rejected.
It requires at least 20 per-episode results, recomputes both aggregate rates
from those records, and verifies a 12 cm lift and at least a 2-second hold. It
also audits sequential episode IDs, positive step counts, integer continuous
hold counts, Box x/y within the declared evaluator range, and consistent
success/failure reasons. A success record must contain the required number of
hold steps and cannot also contain bad orientation.

Bind that result and the checkpoint hash into a gate marker:

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/phase_bc.py certify-bc2 \
  --checkpoint data/teacher/bc/bc2_grasp_lift_43d.pt \
  --rollout-metrics data/teacher/bc/bc2_rollout_metrics.json \
  --output data/teacher/bc/bc2_grasp_lift_43d.gate.json
```

The default gate requires validation normalized MSE <= 0.25, physical
bimanual grasp/lift success >= 80%, and bad orientation <= 5%. Certification
also verifies the checkpoint path/SHA-256, action key/action-contract hash, and
evaluator-config hash. The absolute rollout-metrics path and its SHA-256 are
required inputs and are stored in the marker. Every later gate validation
reopens that file, checks its SHA-256 and checkpoint/action/evaluator bindings,
and compares its episode count and measured rates with the marker. The
evaluator's successful `BILATERAL_HOLD` snapshot store is also bound by its
canonical absolute path and SHA-256 in both the rollout metrics and gate. A
missing or edited metrics/snapshot file, changed checkpoint, or mismatched
action contract therefore invalidates the gate. The lift corpus can then train
`approach`, but it cannot train carry or place because it contains no
phase-1/phase-2 rollout:

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/phase_bc.py train \
  data/teacher/box_xy_randomized_500 \
  --skill approach \
  --dagger-datasets data/teacher/dagger/bc1_approach_round_01.hdf5 \
  --action-key teacher_high_level_action \
  --bc2-gate data/teacher/bc/bc2_grasp_lift_43d_dagger_r02.gate.json \
  --output data/teacher/bc/bc1_approach_23d_dagger_r01.pt \
  --device cuda:0
```

Carry/place datasets must contain real phase-1/phase-2 physics rollouts. A
nominal lift-only file cannot silently produce either missing skill: training
stops with a no-samples error. At runtime:

```python
from unitree_rl_lab.utils.teacher.phase_bc import load_bc_policy

policy = load_bc_policy("data/teacher/bc/bc2_grasp_lift_43d.pt", device="cuda:0")
joint_targets = policy(student_observation_batch)  # (N, 517) -> (N, 43)
print(policy.skill_phase, policy.action_key, policy.metadata)
```

Evaluate the checkpoint on its episode-level test split with:

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/phase_bc.py evaluate \
  data/teacher/box_xy_randomized_500 \
  --checkpoint data/teacher/bc/bc2_grasp_lift_43d.pt \
  --subset test --device cuda:0 \
  --output-json data/teacher/bc/bc2_test_metrics.json
```

Reported values include native action MSE/RMSE/MAE, normalized MSE,
per-action RMSE, sample count, and episode count.

The CLI accepts an individual HDF5, a shell glob, the generated shard
directory, or its `manifest.json`. This is important for VS Code, whose Python
launcher does not expand shell globs.

## 7. Physically validate and certify BC-2

Offline MSE does not satisfy the BC-2 gate. The physical evaluator lets the
teacher execute only the approach prefix, then gives control to the BC-2
student. Its default Box range is the same first-curriculum x/y range.

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/evaluate_bc2_rollout.py \
  --headless --device cuda:0 \
  --checkpoint data/teacher/bc/bc2_grasp_lift_43d.pt \
  --episodes 20 --hold_seconds 2.0 \
  --output data/teacher/bc/bc2_rollout_metrics.json \
  --snapshot_output data/teacher/bc/bc2_success.snapshots.hdf5 \
  --overwrite

/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/phase_bc.py certify-bc2 \
  --checkpoint data/teacher/bc/bc2_grasp_lift_43d.pt \
  --rollout-metrics data/teacher/bc/bc2_rollout_metrics.json \
  --output data/teacher/bc/bc2_grasp_lift_43d.gate.json
```

The gate requires at least 20 physical episodes and is bound to the checkpoint
SHA-256, action contract, hashed evaluator configuration, the absolute
rollout-metrics file path/SHA-256, and the absolute published evaluator-snapshot
path/SHA-256. Validation normalized MSE must be <= 0.25, physical bimanual
grasp/lift success >= 80%, and bad orientation <= 5%. Editing, moving, or
replacing the checkpoint, metrics file, or evaluator snapshot invalidates the
gate.

## 8. Corrective DAgger collection

DAgger starts with teacher approach, then runs the certified BC-2 policy. The
teacher still reads privileged current physics to relabel a state, but the
stored student input remains the 517-D QR/proprio vector. The collector tags:

- one-hand-first contact;
- wrist midpoint/asymmetry away from the Box center;
- Box rotation while both hands contact;
- lift slip or bilateral-contact loss;
- temporary QR loss.

It also records `visible_anchor`, a teacher-labelled post-close state with an
intact Box QR observation. This is not a sixth recovery gate. It prevents the
five corrective categories, especially QR loss, from removing the normal
visible boundary from the DAgger mix.

QR dropout changes only `box_visible` and `box_position_camera` in the actor
observation. It never hides the Box from the teacher or edits physics. After
the close boundary, corrective teacher labels are recomputed from the live
privileged Box pose/orientation and current pelvis pose; they are not a replay
of an initial-Box reference trajectory.

By default a seeded schedule cycles through four short external-Box-wrench
campaigns: lateral impulses around close for one-hand-first/off-center states,
a yaw torque after bilateral contact for in-hand rotation, and a diagonal/down
force after initial lift for slip. These are physical forces/torques, never
pose teleportation. QR loss remains the fifth observation-only campaign. The
published file must contain at least 10 labeled samples for each of the five
categories by default (`--minimum_samples_per_correction_type 10`); otherwise
the atomic recorder aborts instead of publishing a partial correction set.
Only the active physical campaign, explicit QR-loss samples, and visible
anchors are retained. Each rollout is capped at 128 samples, invisible samples
are bounded to three times the accepted visible count, at least 20% of the
published samples must remain visible, and no single recovery category may
cover more than 95% of all samples. The 22.2 cm wrist-midpoint threshold was
chosen from the measured nominal loaded-grasp distribution; the independent
10 cm left/right distance-asymmetry threshold remains in force. Publication
also binds the source BC-2 checkpoint and its certification gate by canonical
path and SHA-256, and the loader revalidates both bindings.

The Box's permanent wrench composer is reset at episode start, reset again on
every control step before the current pulse is added, and reset during cleanup.
Nonzero pulses are added with `is_global=True` after the reset, forcing the
composer to transform the world-frame wrench using the Box's current link pose
rather than a stale cached orientation.

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/collect_dagger_corrections.py \
  --headless --device cuda:0 \
  --checkpoint data/teacher/bc/bc2_grasp_lift_43d.pt \
  --bc2_gate data/teacher/bc/bc2_grasp_lift_43d.gate.json \
  --output data/teacher/dagger/bc2_round_02_balanced.hdf5 \
  --episodes 100 --overwrite
```

Retrain BC-2 with `--dagger-datasets ...`, physically evaluate it again, and
issue a new certification for that new checkpoint. For a certified checkpoint,
`--initial-checkpoint`, `--dagger-loss-weight`, and a conservative learning rate
support bounded fine-tuning without discarding the already validated nominal
behavior. Nominal validation remains the checkpoint-selection metric; DAgger
validation/test metrics are reported separately. Corrections are grouped by
their physical `source_episode`, so a rollout cannot leak across train,
validation, and test splits.

## 9. Carry and place data from physical prior-phase snapshots

Carry accepts only `BILATERAL_HOLD` snapshots whose store metadata declares
`producer=evaluate_bc2_rollout`. The metadata's checkpoint SHA-256, action key,
and action-contract SHA-256 must exactly match the passing BC-2 gate, and the
gate itself revalidates its bound checkpoint and rollout-metrics file first.
The carry collector specifically requires a certified
`processed_43d_joint_target` BC-2 gate. Production-collector shard snapshots
are intentionally rejected. Its input path and SHA-256 must exactly match the
evaluator snapshot bound into that gate. The snapshot store must preserve the
gate's >=2-second and >=12-cm thresholds, and every loaded bilateral snapshot
is revalidated against the gate's measured required-hold step count, bilateral
contact, lift, orientation, and drop status. A complete robot root state, all 43 joint
positions/velocities, Box pose/velocity, last action, task state, and exact
five-frame policy history are restored together. One ordinary controlled
physics step must retain bilateral contact and lift before the state is
accepted.

Place similarly accepts only `CARRY_AT_STAGING` snapshots whose producer is
`collect_carry_place_teacher:carry`. Each staging store records the absolute
parent BC-2 snapshot path and its SHA-256, in addition to the BC-2 checkpoint,
action-key, and action-contract bindings. There is no Box-only teleport path.
Both snapshot kinds require the 100-step (2-second at 50 Hz) continuous
bilateral lift gate.

```bash
# BC-3 teacher: starts from real bilateral hold and writes real staging states.
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/collect_carry_place_teacher.py \
  --headless --device cuda:0 --phase carry \
  --bc2_gate data/teacher/bc/bc2_grasp_lift_43d.gate.json \
  --snapshot_input data/teacher/bc/bc2_success.snapshots.hdf5 \
  --output data/teacher/carry_teacher.hdf5 \
  --staging_snapshot_output data/teacher/carry_at_staging.snapshots.hdf5 \
  --num_trials 25 --overwrite

# BC-4 teacher: starts only from those physically reached staging states.
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/collect_carry_place_teacher.py \
  --headless --device cuda:0 --phase place \
  --bc2_gate data/teacher/bc/bc2_grasp_lift_43d.gate.json \
  --snapshot_input data/teacher/carry_at_staging.snapshots.hdf5 \
  --output data/teacher/place_teacher.hdf5 \
  --num_trials 25 --overwrite
```

Carry begins with a recorded 100-step controller handoff. Its camera command
is zero while lower-body/waist targets are smoothly blended from the snapshot's
last action to Arena WBC, followed by an unrecorded 250-step loaded-stand gate.
At every handoff and recorded carry step, the 28
shoulder/elbow/wrist/Dex3 targets are overwritten with the phase-start
snapshot's last-action targets that produced the realized loaded grasp; this
prevents a freshly initialized redundant Pink solve from opening or rotating
the grasp. Bilateral contact, lift, and upright safety are checked throughout
the handoff.

Carry's default `--max_phase_seconds` is now 16 seconds (up to 800 recorded
steps at 50 Hz), allowing the unchanged camera controller more time to reach
staging. This is a horizon change only: `at_staging` still requires a visible
Bin QR, distance from the 0.30 m standoff within 0.10 m, and absolute yaw error
within 0.15 rad. Carry success still additionally requires bilateral contact,
the strict lift, and the continuous hold gate.

The first 50 recorded carry steps linearly ramp the camera/QR Bin navigation
command from `1/50` to full scale. The same scaled command is written to the
student observation and high-level teacher action before WBC processing. The
recorded 43-D target already contains the 28-joint grasp lock. Replaying or
training `teacher_high_level_action` for carry additionally requires the
documented postprocessor: after WBC, overwrite those same 28 raw action channels
with the per-episode `carry_grasp_lock_reference_raw_action`. The recorder marks
this requirement in root and episode metadata.

These semantics are now machine-enforced as
`unitree_box_carry_runtime_contract_v1`, not inferred from prose at replay time.
Every carry HDF5 stores the ramp length/profile, postprocessor version, exact
ordered 43-D raw joint names, 28 locked names/indices/mask, reference semantics,
and a canonical SHA-256. BC-3 refuses mixed carry sources when any of those
fields differ, embeds the validated contract and hash in either a 23-D or 43-D
checkpoint, and validates it again on load. `play_bc_skill_fsm.py` takes its
ramp and lock indices from the carry checkpoint and verifies the stored raw
joint order against the live articulation. The ramp affects the 517-D input for
both action representations; the post-WBC lock runs only for the 23-D action
because a processed 43-D target already contains it. Older carry data or carry
checkpoints without this contract are intentionally rejected and must be
re-collected/re-trained.

PLACE now performs the same unrecorded 25-step zero-navigation handoff after
restoring a staging snapshot. During this warm-up, lower-body/waist targets are
smoothly handed to Arena WBC while the same 28 arm/hand targets stay locked and
bilateral hold/lift/upright safety is checked. Only after it passes does the
collector switch to phase 2, create the Place trajectory, and begin recording;
the fixed 28-joint carry lock is replaced by the loaded PLACE transform so Pink
can move the arms/wrists over the Bin without discarding their contact-loaded
PD offset.

The carry-at-staging snapshot stores both the last applied loaded raw action and
the exact unprocessed Pink/WBC raw proposal from the original CARRY→PLACE
transition. Snapshot writer schema 2.0 adds the latter field; schema 1.0 remains
readable for BC-2/BILATERAL_HOLD, but PLACE fails closed if its staging snapshot
does not contain this provenance baseline. A restored PLACE collector does not
use that old-process value in delta arithmetic: after its 25-step physical
handoff, it queries the warmed active controller's current carry proposal
without applying it and uses that as its baseline. The source and active
baselines are both recorded for audit.

During recorded PLACE, `unitree_box_place_runtime_contract_v4` applies each
Pink arm proposal as
`pink_proposed + decay(place_step) * (loaded_ref - active_pink_baseline)` on
the exact 14 shoulder/elbow/wrist channels. The inclusive smoothstep decay is
1.0 on the first recorded PLACE step and reaches 0.0 on step 99 by default
(`--place_arm_offset_decay_steps 100`). Thus the contact-loaded preload is
continuous at handoff but no longer remains as a permanent offset during the
large raise/translate motion. Continuous FSM rollout uses its last unprocessed
CARRY proposal; restored collection uses the warmed controller query described
above. The decay step count/profile is part of the canonical contract hash,
stored in PLACE HDF5/checkpoints, and reused by 23-D FSM replay. The contract
also preserves the exact 14 loaded Dex3
finger targets until both high-level hand-state channels are `<= 0.5`.
One-sided open intent keeps both hands locked; only bilateral release unlocks
them. Both ordered masks/names/indices, both references, the predicate,
postprocessor version
`place_decayed_loaded_arm_offset_and_finger_lock_until_release_v1`, and canonical
SHA-256 are stored in each PLACE HDF5 and BC-4 checkpoint. As with CARRY, 23-D
replay applies the postprocessor after WBC, while processed 43-D targets already
contain it.

Carry and place use a `0.75 m` pelvis-height command. The place teacher
preserves the realized grasp, raises it by 8 cm, translates for 100 control
steps, then lowers for 60 steps only to a `0.70 m` Box-centre release target.
The target clears the `0.54 m` Bin wall for the fixed `0.20 m` Box, so the
closed wrists do not collide with the Bin; release lets the Box fall through
the aperture. These values are written into HDF5 metadata and can be adjusted
only explicitly with `--place_lower_steps` and `--place_release_height_m`.
PLACE success is not a one-frame release: the Box
must remain inside the Bin, both hands must be released, and Box linear speed
must stay <= 0.15 m/s for 10 consecutive control steps.
Releasing the Box above the Bin aperture is not classified as a drop merely
because contact is gone or the Box has not yet entered the Bin's z-bounded
volume. PLACE becomes fatal only if the Box reaches the ground threshold
outside the Bin; CARRY's strict lift/drop failure remains unchanged.

Every carry/place collection invocation creates a new UUID
`collection_run_id` and writes it into the published teacher HDF5 metadata;
carry also writes it into the staging snapshot store. The latest same-stem
`.metrics.json` sidecar (for example, `carry_teacher.metrics.json`) records that
run ID, phase, success count, whether the teacher dataset was actually
published, its canonical path and SHA-256, and—during carry—whether the staging
store was published plus its path and SHA-256. If a failed atomic overwrite
leaves an older HDF5 or staging file on disk, the new sidecar explicitly marks
that old file as stale rather than claiming it as output of the current run.

`phase_bc.py` refuses carry/place data when the sidecar is missing, its run ID
or phase differs from the HDF5 metadata, `dataset_published` is false, its
published path/SHA-256 does not match the current HDF5 bytes, or the run has no
successful physical rollout. This prevents an older successful-looking file
from being trained after a newer failed collection attempt. Train BC-3/BC-4
only from these current, publication-bound rollouts using the passing BC-2
gate:

The one explicit exception is `--student_teacher_relabel`: the Student still
controls physics, while the Teacher labels each Student-induced state for
DAgger. These files may contain only failures, but remain trainable only when
their run ID, checkpoint SHA-256, published path, and dataset SHA-256 all match
the sidecar. A newly trained candidate is promoted only after physical rollout
comparison; the first 20-episode Place relabel round was retained as correction
data, while its regressed checkpoint was rejected.

For PLACE, relabel collection compares each live wrist with the wrist pose
captured in the Box frame at the start of the rollout. Missing contact or an
object-relative translation error above the configured threshold switches the
Teacher label from the nominal placement trajectory to a side-specific recovery
target around the measured live Box pose. The Student action continues to drive
the simulator, so this is an on-policy correction dataset rather than another
Teacher rollout. `build_place_relabel_manifest.py` admits a corpus only after
checking every HDF5/sidecar SHA binding, checkpoint identity, unique source
snapshot reference, and the presence of recovery-labelled steps.

The current `place_recovery_relabels_r15` corpus contains 108 valid episodes
from 108 distinct Carry-at-staging snapshot references (14,712 samples). It
contains 4,854 wrist/contact recovery-labelled steps and is bound to the Place
Student checkpoint by SHA-256. The 69 successful and 39 failed Student
rollouts describe the collection-state distribution; they are not a claim of
post-DAgger policy improvement. The verified inventory is stored in
`data/teacher/dagger/place_recovery_relabels_r15/manifest.json`.

The 108 relabel episodes were conservatively fine-tuned from the diversified
600-demo Place Student with relabel source weights 1, 2, and 4. The selected
weight-2 candidate reduced correction-set RMSE from 0.2035 to 0.0697 while
changing the held-out Teacher RMSE from 0.0118 to 0.0127. This offline gain did
not transfer to the physical gate: on the same 20 previously unused
Carry-at-staging states (`demo_30`--`demo_49`), the existing Student succeeded
9/20 and the candidate 7/20. Both missed the 80% gate, so the candidate was
rejected and the FSM continues to use the existing Place checkpoint. The
SHA-bound reports are
`data/teacher/evaluation/place_r12_true_holdout30_49.physical_gate20.json`,
`data/teacher/evaluation/place_r16_w2_true_holdout30_49.physical_gate20.json`,
and `data/teacher/evaluation/place_r16_w2.promotion_decision.json`.

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python scripts/teacher/phase_bc.py train \
  data/teacher/carry_teacher.hdf5 --skill carry \
  --action-key processed_43d_joint_target \
  --bc2-gate data/teacher/bc/bc2_grasp_lift_43d.gate.json \
  --output data/teacher/bc/bc3_carry_43d.pt --device cuda:0

/home/mjkim/anaconda3/envs/env_isaaclab/bin/python scripts/teacher/phase_bc.py train \
  data/teacher/production/place_teacher_success_diverse_600_r07/manifest.json \
  --skill place --action-key teacher_high_level_action \
  --bc2-gate data/teacher/bc/bc2_grasp_lift_43d.gate.json \
  --place-phase-clock-steps 95 \
  --output data/teacher/bc/box_place_23d.pt --device cuda:0
```

## 10. Explicit four-policy FSM

The runtime does not merge the four networks. It routes one shared 517-D
student observation through exactly one checkpoint at a time. The policies may
use different output contracts: Box approach/Pre-grasp predicts the 23-D
wrist/base/Dex3 command and runs it through Pink/Homie each live step, while
grasp/lift predicts a certified 43-D processed joint target. Loaded Carry and
Place predict 23-D wrist/base/Dex3 commands, run Pink/Homie, and then apply
their phase-specific grasp-preservation runtime contracts.

```text
APPROACH --visible + pregrasp--> GRASP_LIFT
         --bilateral + 12 cm + continuous hold--> CARRY
         --camera staging + strict hold retained--> PLACE
         --inside Bin + released + 10-step low-speed settle--> COMPLETE
```

Bad orientation is fatal in every phase. Box drop is fatal in carry/place; the
FSM never silently retries grasp or skips more than one phase in a single
update.

The actor command channels follow the same phase contract as collection:
`APPROACH` receives the camera/QR Box-pregrasp command, `GRASP_LIFT` receives
exactly zero velocity, and `CARRY`/`PLACE` receive the live camera/QR Bin
navigation command. The four BC policies therefore see the command semantics
on which their demonstrations were recorded.

```bash
/home/mjkim/anaconda3/envs/env_isaaclab/bin/python \
  scripts/teacher/play_bc_skill_fsm.py --device cuda:0 \
  --approach_checkpoint data/teacher/bc/box_approach_pregrasp_23d_dagger_open_r05.pt \
  --grasp_lift_checkpoint data/teacher/bc/box_grasp_lift_43d_from_approach_r05_dagger_r02_probe.pt \
  --carry_checkpoint data/teacher/bc/box_carry_43d_fsm_handoff_r03.pt \
  --place_checkpoint data/teacher/bc/bc4_place_43d.pt \
  --bc2_gate data/teacher/bc/box_grasp_lift_43d_from_approach_r05_dagger_r02.gate.json \
  --grasp_handoff_steps 100 --grasp_live_approach_lower_body \
  --grasp_freeze_hands_on_contact --grasp_freeze_upper_on_lift \
  --carry_handoff_steps 100 --loaded_stabilization_steps 100 \
  --loaded_stand_steps 250 --carry_upper_residual_scale 0.0 \
  --grasp_phase_timeout_seconds 12 --episodes 20 \
  --metrics_output data/teacher/bc/fsm_loaded_split_r03.rollout20.metrics.json \
  --overwrite --headless
```

The metrics report `P(approach)`, `P(grasp|approach)`, handoff survival,
loaded-stand survival, `P(carry|grasp)`, `P(place|carry)`, bad-orientation rate,
and final full-task success. In the latest narrow-scene 20-episode evaluation,
Approach reached 20/20, Grasp/Lift reached 5/20, all five successful grasps
survived both stabilization gates, two reached Place, and one completed the
full task. The conditional rates are therefore 25% for Grasp after Approach,
40% for Carry after Grasp, and 50% for Place after Carry; full-task success is
5% with one bad-orientation failure. The SHA-bound 60% production gate failed.
The metrics and gate are stored in
`data/teacher/evaluation/full_fsm_latest_students_place_r12.rollout20.metrics.json`
and
`data/teacher/evaluation/full_fsm_latest_students_place_r12.physical_gate20.json`.
The dominant first bottleneck is continuous bilateral Grasp/Lift hold
(14 grasp-phase timeouts); after a valid grasp, loaded stabilization is 5/5,
but loaded Student walking reaches staging only 2/5.

#### Latest Approach-to-Grasp/Lift transition candidate

The next transition round combines the 500 strict nominal demonstrations with
DAgger rounds 10--13.  These correction sets cover the deployed Approach
endpoint, live lower-body motion, one-hand-first contact, wrist offset, Box
rotation, lift slip, and temporary QR loss.  Fine-tuning uses a conservative
DAgger loss weight of 2 so the canonical Grasp/Lift behavior is not displaced.

`box_grasp_lift_43d_transition_dagger_r15_w2.pt` passed its isolated physical
Grasp/Lift certification in 20/20 rollouts with no bad-orientation termination,
but failed the actual Approach-to-Grasp/Lift screen in 0/5 rollouts.  Approach
itself reached 5/5; every episode then timed out in Grasp/Lift and the maximum
Box lift remained below 3 cm.  The SHA-bound transition screen gate therefore
rejects promotion and retains the deployed Grasp/Lift policy.  The 20-episode
transition gate and 5 m full-FSM evaluation are intentionally not run for this
candidate.  The evidence is stored in:

- `data/teacher/bc/box_grasp_lift_43d_transition_dagger_r15_w2.gate.json`
- `data/teacher/evaluation/grasp_transition_r15_w2.seed271828.screen5.metrics.json`
- `data/teacher/evaluation/grasp_transition_r15_w2.seed271828.screen_gate5.json`

The result isolates the remaining problem as a policy-boundary mismatch rather
than loss of canonical lifting ability.  The next data round must reproduce the
FSM post-processing contract at collection time, especially contact-triggered
hand locking and the live lower-body state, before another candidate is trained.

### Grasp-to-Carry stabilization and split control

The production FSM treats the grasp-to-CARRY boundary as a physical subtask,
not an instantaneous policy switch. It first holds zero navigation command for
100 consecutive control steps while retaining the exact shoulder, elbow,
wrist, and finger target captured at grasp success. Homie alone updates the 12
leg and three waist targets. It then verifies a five-second loaded stand before
enabling the ramped navigation command and handing the 12 leg plus three waist
targets to the Carry Student. Any optional manipulation residual is
bounded, while finger targets remain exactly locked so a CARRY policy cannot
command opening.

Carry data should be collected from snapshots produced by the real
APPROACH-to-GRASP FSM chain, rather than only from isolated GRASP rollouts.
This retains the wrist asymmetry, Box rotation, and incipient slip present at
the deployed boundary. The target corpus is at least 100 successful Carry
trajectories, and training gives extra weight to the first 100 recorded Carry
steps. Evaluation reports the following separate physical gates:

When a smaller valid snapshot set is cycled with `--repeat_snapshots`, metrics
report both the independent physical rollout count and the unique source-state
count. Repetition can satisfy the trajectory-count gate, but is not presented
as added phase-boundary diversity.

- `P(handoff survival | grasp)`: 100-step zero-command bilateral lifted hold;
- `P(loaded stand | handoff)`: a further five seconds of loaded balance;
- `P(walk | loaded stand)`: arrival at the staging target with the grasp held;
- `P(carry | grasp)`: the complete conditional Carry result.

Payload-aware locomotion reinforcement learning is a later improvement and is
not part of this data recollection/gating run.

## 11. Implementation map

- `mimic_xy_retarget.py`: fixed-scene contract, true Isaac Lab Mimic transform
  boundary, and strict continuous demo filter.
- `arena_teacher_runtime.py`: shared Arena trajectory, Pink/Homie bridge inputs,
  contact queries, and fixed teacher dynamics.
- `student_observation.py`: immutable 517-D student feature layout and
  observation-only QR masking.
- `phase_bc.py`: phase inference, episode-level splits, DAgger merge,
  normalization, exact ordered-action contracts, training, checkpoint loading,
  production-manifest checks, and checkpoint/metrics-file-bound BC-2
  certification with strict per-episode audit.
- `dagger.py`: five recovery-state predicates, seeded physical Box-wrench
  campaigns, observation-only QR dropout, and atomic correction recorder.
- `physical_snapshots.py` / `physical_snapshot_runtime.py`: provenance-checked
  full dynamic snapshot storage, exact history-5 restoration, and simulator
  restore.
- `phase_teachers.py`: object-relative carry/place teachers and the 0.75 m
  pelvis-height command.
- `collect_carry_place_teacher.py`: BC-2/snapshot provenance, configurable
  loaded handoff, Carry grasp lock, Place loaded-arm/finger preservation,
  Student physical evaluation, Student-state Teacher relabeling, parent SHA
  propagation, and run-ID/HDF5-SHA publication sidecars.
- `aggregate_student_rollout_metrics.py`: disjoint physical rollout sidecar
  aggregation with checkpoint SHA consistency and explicit pass/fail gates.
- `compare_student_physical_gates.py`: same-episode baseline/candidate
  comparison, checkpoint/sidecar SHA validation, and auditable promotion or
  rejection selection.
- `gate_full_student_fsm.py`: recomputed stage counts, four checkpoint SHA
  bindings, termination distribution, and the final 20-episode/60% production
  pass/fail decision.
- `analyze_place_student_failures.py`: first hand-loss timing, side, trajectory
  region, wrist asymmetry, Box tilt, Bin distance, and QR visibility analysis
  for publication-bound PLACE Student rollouts.
- `build_place_relabel_manifest.py`: SHA-bound episode/sample/recovery totals
  and source-snapshot diversity gate for PLACE Student-state relabel shards.
- `skill_fsm.py`: explicit routing and conditional success accounting.
- `generate_randomized_box_teacher.py`: resumable 500--1000-demo sharding,
  manifest, independent shard audit, and snapshot-count audit.
- `box_skill_pipeline.json`: machine-readable scene/gate/pipeline contract.
