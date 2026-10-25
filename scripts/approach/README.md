# G1 장거리 Approach: PPO Navigation → Box 접근(BC-1) → 양손 파지·리프트(BC-2)

3–5 m 떨어진 곳에서 걸어와 테이블에 닿지 않고 Box 접근을 끝낸 뒤, 양손 파지·리프트로 넘기는 **시뮬레이션** 파이프라인입니다. Isaac Lab과 Arena WBC(Pink IK 상체, Homie V2 하체) 위에서 동작합니다.

> 파일명과 checkpoint의 `bc1` / `bc2`는 lab 문서(`G1_43DOF_LOCO_MANIPULATION.md`)의 1단계 **Box 접근 및 양손 Pre-grasp 정렬**과 2단계 **양손 파지·리프트·안정 유지**를 가리킵니다. 이 문서에서는 BC-1 / BC-2로 줄여 씁니다.

> **실제 로봇 배포 불가 (현재 상태)**
> - PPO navigation은 시뮬레이터의 GT(ground truth) 값을 관측합니다: 목표(BC-1 시작 자세)의 위치·방향·거리, base 선속도.
> - 테이블 안전 정지와 BC-1 → BC-2 gate는 GT 테이블·박스 위치를 쓰고, scripted navigator는 GT 로봇 자세를 씁니다.
> - BC-1 / BC-2는 lab의 Student 관측 규약(`unitree_box_qr_proprio_v1`, 517-D)만 입력으로 받습니다. lab 규약상 privileged 데이터(정답 world pose, 접촉, lift 등)는 입력에 없습니다.
> - 다만 이 규약의 카메라 항목(Box·Bin QR 위치와 보임 여부, Table 위치)은 시뮬레이터가 GT 위치를 카메라 모델로 변환해 만든 값입니다. 실제 검출기의 잡음이나 오검출은 반영되지 않았습니다(1.1절).
> - 따라서 인식과 상태 추정으로 바꾸고 다시 검증하기 전에는 **실제 로봇에 바로 배포할 수 없습니다.**
> - 모든 결과는 시뮬레이션 결과입니다.

## 1. 구성

```
PPO navigation (Homie V2 [vx, vy, wz] + stop)  ──handoff 창──▶  BC-1 v3.1 (23-D 고수준 명령 → Pink/Homie WBC)
                                               ──기존 gate──▶   BC-2 v3.1 (43-D joint target) ──▶ 2 s lift-hold
```

| 구성 요소 | 내용 |
|---|---|
| PPO navigation v7 | 목표 지점까지 걷는 goal-conditioned PPO. task `Unitree-G1-43dof-BoxNav-HomieV7`. 출력 `[vx, vy, wz, stop]`을 Homie V2 하체가 실행합니다. |
| Collision-aware Teacher | 기존 Mimic Teacher 경로를 WBC 처짐(3–6 cm)을 반영해 예측한 실제 손 경로 기준으로 들어 올립니다. 열린 손 전체가 테이블에서 margin(접근 4 cm, 파지 직전 7 cm)을 유지합니다(`safe_paths.py`, `hand_clearance.py`). |
| BC-1 v3.1 | 위 Teacher로 모은 무접촉 시연 239개로 학습(lab `phase_bc` 학습 코드, observation scale floor) |
| BC-2 v3.1 | BC-1 v3.1의 gate 상태에서 Teacher가 이어 수행한 무접촉 파지·리프트 시연 159개로 학습(handoff 시점부터 label) |
| Handoff gate | FSM의 Approach→Grasp 조건과 같은 4개 기하 조건: 손끝 거리 0.18 m, cage 중점 0.11 m, 손목 중점 0.222 m, 좌우 비대칭 0.10 m |
| 안전 지표 | `--safety_metrics`: 단계별(navigation / BC-1 / BC-2) 테이블 접촉(팔·손 22개 link, > 1 N), 최대 힘, impulse, 손가락–테이블 최소 거리. 기존 성공 지표와 따로 기록합니다. |

### 1.1 관측과 GT 사용 범위
각 입력이 코드에서 만들어지는 경로를 추적한 결과입니다(2026-10-25).

| 구성 | 쓰임 | 시뮬레이션에서 값의 출처 | 실제 로봇에서 필요한 것 |
|---|---|---|---|
| PPO navigation 관측 | 정책 입력 | GT 목표 위치·방향·거리, GT base 선속도 | 인식 기반 목표 추정, 속도 추정 |
| BC 고유감각(IMU 각속도·중력 방향, 관절 위치·속도, 5 step 이력) | 정책 입력 | 시뮬레이터 값. 평가와 데이터 수집에서는 관측 잡음을 끔 | 실제 센서(잡음에 대한 강건성은 미검증) |
| BC 직전 action(43-D) | 정책 입력 | 직전 관절 목표 | 같음 |
| Box·Bin QR 위치(카메라 좌표)와 보임 여부 | 정책 입력 | GT 위치를 D435 카메라 모델(장착 위치, 시야각, 해상도, 거리 범위)로 변환. 시야 밖이면 0, 잡음 없음 | QR 검출기(오차·지연·오검출 미검증) |
| Table 위치(카메라 좌표) | 정책 입력 | GT 위치. 보임 여부와 관계없이 항상 제공 | 테이블 인식 |
| 속도 명령(3-D) | 정책 입력 | BC-1: 위 Box QR 측정과 로봇 기구학으로 계산한 pre-grasp 명령. BC-2 실행 중에는 0 | QR 검출기 |
| BC-1 → BC-2 gate | 전환 판단 | 손끝·손목 위치(기구학)와 GT 박스 위치. lab FSM과 같은 방식 | 인식 기반 판단 |
| 테이블 안전 정지 | navigation 목표 조정 | GT 테이블 앞 모서리 위치 | 테이블 인식 |
| 안전 지표 | 기록 전용 | GT 접촉 센서와 형상 | 해당 없음 |

- 학습은 HDF5의 `student_observation` 데이터만 읽습니다(lab `phase_bc`가 검사). 같은 파일에 기록된 privileged 값(박스·손목 pose, 접촉 등)은 확인용이며 입력으로 쓰지 않습니다.
- 평가 중 정책 입력은 학습 데이터와 같은 함수(`arena_teacher_runtime.student_observation`)로 만듭니다.

## 2. 결과 (시뮬레이션, Arena WBC)

| 표시 | 의미 |
|---|---|
| **[확정]** | 같은 seed로 짝지은 평가까지 끝난 결과 |
| **[예비]** | 소규모 통제 실험에서만 확인한 결과 |
| **[미검증]** | 아직 측정하지 않음 |

### 2.1 최종 결과: 기존 gate [확정]
조건은 10개입니다: nominal(PPO v7 / v6), 박스 고정 시작, 리셋 시작, 박스 +1 / +2 / +3 cm, 박스 −2 cm, 좌 / 우 2 cm.

- 리셋 시작 조건을 빼고는 모두 PPO navigation 3–5 m → BC-1 → BC-2 순서입니다. 리셋 시작은 navigation 없이 BC-1부터 시작합니다.
- 기존 시스템(BC-1 v2 + BC-2 v2)도 같은 시작 조건으로 평가했습니다. 일부 조건의 episode 수가 더 많아 합계가 240회입니다.

| 지표 | BC-1 v2 + BC-2 v2 | **BC-1 v3.1 + BC-2 v3.1** |
|---|---:|---:|
| **Strict collision-free E2E** (navigation handoff·BC-1·BC-2 모두 테이블 무접촉 + BC-2 2 s lift-hold 성공) | 2/240 | **149/160** |
| BC-1 중 테이블 접촉 | 238/240 | 0/160 |
| BC-1 gate 도달 (기존 지표) | 214/240 | 160/160 |
| 기존 E2E (접촉 무관) | 196/240 | 150/160 |
| BC-2 중 테이블 접촉 | 147/214 | 1/160 |

- strict 실패 11건
  - navigation handoff 시간 초과 6
  - BC-2 유지 실패 4: 모두 조기 gate 뒤
  - BC-2 중 접촉 1: 76 N
- 권장 handoff 창은 dx −6 ~ −0.5 cm(PPO v7 목표 −2.5 cm)입니다. 결과: PPO v7 20/20 무접촉 성공, PPO v6 16/20(실패 4회는 정지 조건 미충족, 접촉 없음).

### 2.2 Cage gate (조기 handoff 대응)
| 항목 | 상태 |
|---|---|
| 조기 handoff 재현 실험(gate 손끝 기준 0.21 m로 완화, 15회): 기존 gate 10/15 → cage 폭 ≤ 0.29 m 조건 15/15 | **[예비]** McNemar exact p = 0.0625, 0.05 기준 유의하지 않음 |
| 배포 조건(PPO v7 권장 창, 10회): 결과 변화 없음, 접촉 0 | **[예비]** |
| cage gate를 넣은 **전체 160 episode 평가** | **[미검증] 수행하지 않음** (`run_p10_cage_gate_eval.sh`로 재개 가능) |

cage gate 옵션은 기본값이 꺼져 있습니다. 위 149/160은 **기존 gate 기준**입니다.

## 3. 프로젝트 구조

```
source/unitree_rl_lab/unitree_rl_lab/
├── tasks/navigation/          PPO box-navigation task (gym ID: Unitree-G1-43dof-BoxNav-Homie, -HomieV6, -HomieV6b, -HomieV7)
│   ├── __init__.py            gym 등록 (문자열 entry point)
│   ├── nav_env_cfg.py         장면(lab의 box_carry_with_hand_env_cfg 재사용)·관측·보상·종료·curriculum
│   ├── mdp.py                 관측·보상·종료 항목, 시작 거리 curriculum
│   ├── homie_action.py        PPO 출력 [vx, vy, wz, stop] → Homie V2 하체 관절 목표
│   └── agents.py              RSL-RL PPO 설정
└── utils/
    ├── homie_batch.py         batched torch Homie V2 (walk/stand ONNX 가중치 로드)
    └── teacher/phase_bc.py    (수정) train_phase_policy의 선택 인자 observation_scale_floor (기본값은 기존과 같음)

scripts/teacher/play_bc_skill_fsm.py   (수정) --pre_navigate 등 기본값 꺼짐 옵션 (5.5절)

scripts/approach/
├── README.md
├── Navigation 학습·평가
│   train_nav_ppo.py, eval_nav_ppo.py, nav_env_check_helpers.py, verify_nav_policy_export.py, run_p5_v7_select.sh
├── 장거리 체인 평가 · collision-aware Teacher 수집
│   eval_bc1_approach.py (navigation → BC-1 → gate → BC-2 / Teacher, 안전 지표)
│   scripted_nav.py, safe_paths.py, hand_clearance.py, table_contact.py,
│   p0_bc1_diagnostics.py (feature layout), validate_clearance_contact.py
├── BC 학습
│   select_safe_demos.py, train_bc1_safe.py (BC-1·BC-2 v3.1), train_bc1.py, train_bc1_handoff.py (v2 기준 모델)
├── 실행 묶음
│   run_p8_collect.sh, run_p8_collect_bc2.sh, run_p8_eval.sh, run_p8_chain.sh (Isaac 2 process 큐),
│   run_p8_suite.sh (원인 실험), run_p8_v31_pipeline.sh (데이터 → 학습 → 평가)
├── 결과 정리 (CPU)
│   run_p8_v31_summary.sh, summarize_p8_runs.py, summarize_p8_episodes.py, render_p8_tables.py,
│   render_p8_readiness.py, plot_p8_eval.py, plot_p8_margin_nav.py, analyze_p6_handoff.py
└── Handoff readiness 실험 [예비]
    run_p9_gate.sh, analyze_p9_early_handoff.py, summarize_p9_gate.py, plot_p9_gate_variants.py, plot_p9_handoff.py,
    run_p10_cage_gate_eval.sh, summarize_p10_cage_gate.py
```

스크립트 이름의 `p5`, `p8` 등은 연구 단계 번호입니다. 실험 기록과 맞추기 위해 그대로 두었습니다.

## 4. 준비

### 4.1 환경
lab 표준 환경을 그대로 씁니다. `run_*.sh`는 스스로 `packaging/activate.sh`를 불러옵니다.

```bash
packaging/bootstrap.sh          # 처음 한 번 (.venv, external/ 준비)
source packaging/activate.sh
```

### 4.2 외부 자원
git에 포함되지 않지만, 위 환경에서 제공되는 자원입니다.

| 자원 | 쓰는 곳 | 찾는 위치 |
|---|---|---|
| G1 URDF와 collision mesh | `hand_clearance.py`(안전 지표, collision-aware Teacher) | `UNITREE_ROS_DIR`(activate.sh가 설정) 또는 `external/unitree_ros` |
| Arena source demo | Mimic Teacher(`--mode teacher_collect`, `--then_teacher`) | `--arena_dataset`, 기본값은 lab `runtime_paths`(`UNITREE_ARENA_DATASET`) |
| Homie V2 ONNX (`walk.onnx`, `stand.onnx`) | navigation 환경(`utils/homie_batch.py`) | 1) `UNITREE_HOMIE_V2_DIR` 2) `outputs/approach_p1/assets/homie_v2/` 3) Arena asset cache `/tmp/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy/models/homie_v2/` |

### 4.3 필요한 Checkpoint (GitHub에 올리지 않음)
- 모델 파일(`*.pt`)은 `.gitignore` 대상이며 이 저장소에 포함하지 않습니다.
- **배포 위치: 미정** (연구실 공유 드라이브 예정)
- 아래 파일을 **같은 경로**에 두면 실행 스크립트를 수정 없이 쓸 수 있습니다.

| 모델 | 경로 | 용도 | sha256 (앞 12자리) |
|---|---|---|---|
| BC-1 v3.1 | `outputs/approach_p8/checkpoints/bc1_v31_safe.pt` | 최종 시스템 | `6386be2b17ad` |
| BC-2 v3.1 | `outputs/approach_p8/checkpoints/bc2_v31_safe.pt` | 최종 시스템 | `c4fb01297ef0` |
| PPO v7 TorchScript | `outputs/approach_p5/eval/v7_nav_policy.pt` | WBC 평가·데이터 수집 | `26a79568bcba` |
| PPO v7 RSL-RL checkpoint | `outputs/approach_p5/ppo/2026-10-22_01-20-31_nav_v7_from_v6b1399_headinggate_gatelatch/model_1898.pt` | batched 평가, export 확인 | `59da6f848221` |
| PPO v6 TorchScript | `outputs/approach_p4/eval/v6_nav_policy.pt` | 비교 조건(`NOMV6`) | `1fc9aabe566e` |
| PPO v6b RSL-RL checkpoint | `outputs/approach_p4/ppo/2026-10-21_14-50-13_nav_v6b_from_v6a800_stopped_outside_penalty/model_1399.pt` | v7 fine-tune 시작점 | `7eb2131a044c` |
| BC-1 v2 / BC-2 v2 | `outputs/approach_p1/checkpoints/bc1_v2_handoff_s01_r3.pt`, `bc2_v2_from_bc1.pt` | 비교 기준(`*_V2` 조건) | `f33e9bcf485b` / `7397a88cd9a3` |

전체 sha256 목록은 모델과 함께 배포합니다.

### 4.4 재학습용 데이터 (git에 포함하지 않음, 배포 위치 미정)
- v3.1 학습에 쓴 무접촉 시연
  - BC-1: `outputs/approach_p8/collect_v31/`(36 MB)
  - BC-2: `outputs/approach_p8/collect_bc2_v31/`(23 MB)
- 선택 목록(`accepted_*.json`)에는 절대 경로가 들어 있습니다. 새 위치에서는 `select_safe_demos.py`로 다시 만듭니다(6.2절).
- 원본 Teacher corpus(`data/teacher/`)는 v3.1 학습에 쓰지 않습니다.

## 5. 평가
WBC 평가는 GPU에서 Isaac 1 process(1 env)로 돌고, 결과는 `<output_dir>/summary.json`에 저장됩니다. 여기에는 episode별 결과와 `safety`, `safety_flags`가 들어 있습니다.

### 5.1 빠른 확인 (권장 handoff 창, 3 episode)
```bash
BC1_V3=outputs/approach_p8/checkpoints/bc1_v31_safe.pt \
BC2_V3=outputs/approach_p8/checkpoints/bc2_v31_safe.pt \
NAME_SUFFIX=_v31 STAGES="WIN_V3" EPISODES=3 \
ROOT=outputs/approach_quick/eval LOGS=outputs/approach_quick/logs \
  scripts/approach/run_p8_eval.sh

python3 scripts/approach/summarize_p8_runs.py \
  --run "win v7=outputs/approach_quick/eval/win_v3_seed4002_v31" --output outputs/approach_quick/summary.json
```

### 5.2 149/160 재현 (10개 조건, 160 episode)
Isaac 2 process를 동시에 실행하며, 약 6시간 걸립니다.

```bash
BC1_V3=outputs/approach_p8/checkpoints/bc1_v31_safe.pt \
BC2_V3=outputs/approach_p8/checkpoints/bc2_v31_safe.pt NAME_SUFFIX=_v31 \
QUEUE_A="eval:10 CAUSE_V3;eval NOMINAL_V3;eval BOX2_V3;eval:10 LAT+2_V3;eval:10 TOWARD2_V3" \
QUEUE_B="eval RESET_V3;eval BOX1_V3;eval BOX3_V3;eval:10 LAT-2_V3;eval NOMV6_V3" \
  scripts/approach/run_p8_chain.sh

scripts/approach/run_p8_v31_summary.sh   # 표·그림 (CPU)
```
- 조건 정의(seed 4002–4008, 8102)는 `run_p8_eval.sh` 머리말에 있습니다.
- 출력 폴더에 결과(`summary.json`)가 이미 있는 run은 건너뜁니다.
- `run_p8_v31_summary.sh`는 v2 비교 run이 없으면 그 행을 건너뜁니다.
- 같은 seed와 조건이라도 GPU나 driver가 다르면 수치가 조금 달라질 수 있습니다.

### 5.3 Navigation 단독 (batched, 256 env)
```bash
source packaging/activate.sh
python scripts/approach/eval_nav_ppo.py --task Unitree-G1-43dof-BoxNav-HomieV7 --controller ppo \
  --checkpoint outputs/approach_p5/ppo/2026-10-22_01-20-31_nav_v7_from_v6b1399_headinggate_gatelatch/model_1898.pt \
  --stage 2 --num_envs 256 --seed 3001 --nav_target_offset_x -0.008 --nav_handoff_x_range -0.03 0.015 \
  --headless --device cuda:0 --output outputs/approach_quick/nav_v7_S3.json
```
- `--stage 2`는 3–5 m, 임의 방향 시작입니다. `--stage 3`은 박스 위치 변화가 더해집니다.
- TorchScript 내보내기: 위 명령에 `--export_jit <path>`를 붙입니다.
- 내보낸 파일 확인(CPU): `python scripts/approach/verify_nav_policy_export.py --checkpoint <model.pt> --jit <path> --output <json>`

### 5.4 결과 집계만 다시 하기 (CPU)
저장된 평가 결과에서 표를 다시 만듭니다.

```bash
python3 scripts/approach/summarize_p8_runs.py --run "<라벨>=<run 폴더>" [--run ...] [--pair "<라벨 A>:<라벨 B>"] --output <json>
```

### 5.5 Skill FSM에서 실행
`scripts/teacher/play_bc_skill_fsm.py`에 기본값이 꺼진 옵션을 추가했습니다. 기본값에서는 기존 동작과 같습니다.

| 옵션 | 내용 |
|---|---|
| `--pre_navigate` (+ `--pre_navigate_start_range`, `--pre_navigate_target_jitter` 등) | 각 episode를 BC-1 시작점 뒤 무작위 위치에서 시작해, GT scripted navigator로 걸어간 뒤 기존 APPROACH를 실행 |
| `--approach_hand_state_decoding binary` | BC-1의 손 상태 출력을 Arena의 {0, 1} 규약으로 변환(≥ 0.5이면 닫음). `none`이면 0이 아닌 회귀 출력에서 손이 바로 닫힘 |
| `--narrow_camera_profile collector_69` | Teacher 수집 때와 같은 69.4° 카메라 사용. 기본 `certified_55`는 기존 55.2° |

- 검증 범위: BC-1 v2 + BC-2 v2를 0.4–0.8 m 뒤에서 scripted navigation으로 시작해, `--grasp_handoff_steps 0` 설정에서 10/10 성공했습니다.
- **PPO v7, BC-1 v3.1 / BC-2 v3.1은 FSM에서 아직 시험하지 않았습니다.** 장거리 전체 체인은 `eval_bc1_approach.py`에서 검증했습니다.

## 6. 학습

### 6.1 PPO navigation v7
v6(처음부터) → v6b → v7 순서로 fine-tune합니다.

- 공통 인자: `--num_envs 1024 --nav_target_offset_x -0.008 --nav_handoff_x_range -0.03 0.015 --headless`
- 각 run의 정확한 설정은 run 폴더의 `params/`(env.yaml, agent.yaml)에 저장되고, 모델과 함께 배포합니다.

| 단계 | task | 시작점과 주요 인자 |
|---|---|---|
| v6, iteration 0–300 | `Unitree-G1-43dof-BoxNav-HomieV6` | 처음부터, `--seed 1` |
| v6, 300–800 | `...-HomieV6` | `--resume_checkpoint <v6 model_300.pt> --set_std 0.5 --entropy_coef 0.002 --start_stage 2` |
| v6b, 800–1399 | `...-HomieV6b` | `--resume_checkpoint <v6 model_800.pt> --set_std 0.4 --entropy_coef 0.002 --start_stage 3` |
| v7, 1399–1898 | `...-HomieV7` | 아래 명령 |

```bash
python scripts/approach/train_nav_ppo.py --task Unitree-G1-43dof-BoxNav-HomieV7 --num_envs 1024 --max_iterations 500 --seed 5 \
  --run_name nav_v7_from_v6b1399_headinggate_gatelatch \
  --resume_checkpoint outputs/approach_p4/ppo/2026-10-21_14-50-13_nav_v6b_from_v6a800_stopped_outside_penalty/model_1399.pt \
  --start_stage 3 --entropy_coef 0.002 --set_std 0.3 --nav_target_offset_x -0.008 --nav_handoff_x_range -0.03 0.015 \
  --log_root outputs/approach_p5/ppo --headless
```
- checkpoint 선택은 seed 2001로 합니다(보고용 seed 3001과 분리): `RUN=<run 폴더> CKPTS="1500 1600 1700 1800 1898" scripts/approach/run_p5_v7_select.sh`
- 선택한 checkpoint는 TorchScript로 내보내고(5.3절), `verify_nav_policy_export.py`로 확인합니다.

### 6.2 Collision-aware Teacher → BC-1 v3.1 → BC-2 v3.1
한 번에 실행합니다(데이터 수집 → 선별 → BC-1 학습 → BC-2 데이터 → BC-2 학습).

```bash
MARGIN_LATE=0.07 SKIP_EVAL=1 scripts/approach/run_p8_v31_pipeline.sh
```
- 결과 경로: `outputs/approach_p8/collect_v31`, `collect_bc2_v31`, `checkpoints/bc1_v31_safe.pt`, `bc2_v31_safe.pt`
- **같은 경로에 checkpoint가 있으면 그 학습 단계는 건너뜁니다.** 배포된 모델을 둔 작업 폴더와 다른 폴더에서 실행하세요.
- `SKIP_EVAL=1`을 빼면 5.2절의 평가까지 이어서 실행합니다.
- 데이터 수집에는 GPU가 필요하고, 선별과 학습은 CPU로 됩니다(각 수 분).

단계별로 실행할 때는 다음 순서입니다.

```bash
# 1) 무접촉 Teacher 시연 수집 (GPU): 짧은 scripted navigation / PPO v7 navigation / 리셋 시작
MARGIN_LATE=0.07 STAGES="SHORT_0 SHORT_1 SHORT_2 SHORT_3 PPO_0 PPO_1 RESET_0" EPISODES=40 \
  ROOT=<dir> LOGS=<dir>/logs scripts/approach/run_p8_collect.sh
# 2) 선별 (CPU): gate 도달, 기록 구간 전체 무접촉, 최소 여유 1 cm
python scripts/approach/select_safe_demos.py --scope recorded --min_clearance_m 0.01 \
  --run <dir>/short_nav_0 <dir>/short_nav_1 ... --output accepted_bc1.json
# 3) BC-1 학습 (CPU)
python scripts/approach/train_bc1_safe.py --accepted accepted_bc1.json --output bc1.pt --device cpu --epochs 400 --patience 40
# 4) BC-2 시연 (GPU): BC-1 → gate → Teacher의 무접촉 파지·리프트
MARGIN_LATE=0.07 BC1_V3=bc1.pt STAGES="SHORT_0 SHORT_1 SHORT_2 PPO_0 RESET_0" EPISODES=40 \
  ROOT=<dir2> LOGS=<dir2>/logs scripts/approach/run_p8_collect_bc2.sh
# 5) 선별과 BC-2 학습 (CPU, handoff 시점부터 label)
python scripts/approach/select_safe_demos.py --phase bc2 --min_clearance_m 0.005 \
  --run <dir2>/short_nav_0 ... --output accepted_bc2.json
python scripts/approach/train_bc1_safe.py --accepted accepted_bc2.json --output bc2.pt --skill grasp_lift \
  --bc2-labels from_handoff --device cpu --epochs 400 --patience 40
```
`run_p8_v31_pipeline.sh`는 일부 stage(`PPO_*`, BC-2의 `RESET_0`)를 20 episode로 줄여 실행합니다. v3.1과 같은 데이터를 만들려면 이 스크립트의 stage와 episode 수를 따르세요.

## 7. Handoff readiness 실험 [예비]
- `run_p9_gate.sh`
  - 조기 handoff가 많은 조건과 배포 조건에서 gate 변형을 같은 시작 조건으로 비교합니다: 기존 / cage ≤ 0.29 m / 고정 지연 / 정지 hold.
  - 분석은 `analyze_p9_early_handoff.py`와 `summarize_p9_gate.py`(McNemar exact), 그림은 `plot_p9_*.py`입니다.
- `run_p10_cage_gate_eval.sh`
  - cage 조건을 넣은 10개 조건 160 episode 재평가입니다(약 6시간, **아직 실행하지 않음**).
  - 저장된 기존 gate 결과와의 짝지은 비교는 `summarize_p10_cage_gate.py`로 합니다.
- cage 조건이 30 step 안에 만족되지 않으면 BC-2로 넘기지 않습니다. 대신 손목을 3 cm 들어 올리고 정지합니다(`--gate_ready_timeout_action abort`).

## 8. `eval_bc1_approach.py` 주요 옵션

| 옵션 | 내용 |
|---|---|
| `--mode {student, teacher_replay, teacher_collect}` | BC-1 평가 / Teacher open-loop 재생 / Teacher 시연 수집(`--collect_output`) |
| `--nav_policy <jit> --nav_action_profile v7` | PPO navigation 뒤 BC-1 시작. `--start_polar 3.0 5.0 90 3.1416`(거리·방위·방향 범위) |
| `--nav_target_offset_x`, `--nav_handoff_x_range` | navigation 목표와 handoff 창(BC-1 시작점 기준 x, m) |
| `--table_safe_gap 0.41 --table_safe_margin 0.016` | 박스가 테이블 안쪽에 있을 때 손가락이 테이블 앞 모서리에 닿지 않게 목표와 창을 줄임(GT 테이블 위치 사용) |
| `--navigate` | PPO 대신 GT scripted navigator |
| `--then_bc2 <ckpt>` | gate 뒤 BC-2 실행 |
| `--then_teacher --teacher_continuation_path safe` | gate 뒤 Teacher가 무접촉 경로로 파지·리프트(BC-2 데이터) |
| `--teacher_path safe --safe_margin 0.04 --safe_margin_late 0.07` | collision-aware Teacher 경로 |
| `--hand_state_decoding binary`, `--camera_profile collector` | 손 상태 변환, 69.4° 카메라 |
| `--safety_metrics`, `--nav_trace` | 안전 지표, step별 trace |
| `--gate_max_cage_width_m 0.29 --gate_ready_max_wait_steps 30 --gate_ready_timeout_action abort` | cage readiness 조건 [예비], 기본값 꺼짐 |
| `--reseed_each_episode` | episode마다 (seed, episode)로 다시 seed를 정해, 변형 간 시작 조건을 맞춤 |

## 9. 한계
- **GT 관측**: 맨 위 안내와 1.1절을 참고하세요. 실제 로봇에 배포하기 전에 인식과 상태 추정으로 바꾸고, 잡음과 오검출에 대해 다시 검증해야 합니다.
- **시뮬레이션 전용 설정**: pre-grasp margin 7 cm는 이 WBC의 처짐(PD, 중력 보상 없음)에 맞춘 값입니다. 다른 제어기나 실제 로봇에서는 다시 정해야 합니다.
- **FSM 통합**: PPO v7과 BC v3.1은 FSM에서 아직 시험하지 않았습니다.
- **표본 크기**: 조건당 10–20 episode입니다.
- **잡는 위치**: v3.1은 v2보다 4–5 cm 높게 잡습니다. 20 cm 박스 외의 크기와 모양은 시험하지 않았습니다.
- **시뮬레이터 이력**: 같은 seed라도 앞 episode의 이력 때문에 궤적이 조금 달라질 수 있습니다. 변형 간 비교에는 `--reseed_each_episode`를 씁니다.

## 10. 검증 환경과 기록
- 환경
  - Ubuntu 22.04, Python 3.11.14
  - Isaac Sim 5.1.0, Isaac Lab 0.54.3, rsl-rl-lib 3.1.2, PyTorch 2.7.0+cu128
  - GPU: RTX 5060 Ti 8 GB
- GPU 평가: 위 결과는 2026-10-24에 실행했습니다(8단계 평가).
- 2026-10-25 통합 시 CPU 검증
  - BC-1 / BC-2 v3.1: 6.2절의 선별과 학습 명령을 CPU에서 다시 실행
    - 선별 결과가 같은 시연 집합(239개 / 159개)
    - 학습 결과가 저장된 checkpoint와 bit 단위로 같음: 가중치, 정규화, 학습 이력, test 오차
  - 저장된 BC checkpoint를 lab loader(`phase_bc.load_bc_policy`)로 읽어 계산한 test 오차가 학습 기록과 같음
  - 저장된 평가 결과를 이 코드로 다시 집계해 149/160을 재현
  - lab 테스트 결과는 통합 전후로 같음
  - PPO v7의 TorchScript가 checkpoint와 bit 단위로 같음
  - 같은 날 GPU smoke test는 하지 않았습니다. Isaac 실행 경로(navigation, BC-1, BC-2 rollout)는 이번 통합 뒤 다시 실행하지 않았습니다.
