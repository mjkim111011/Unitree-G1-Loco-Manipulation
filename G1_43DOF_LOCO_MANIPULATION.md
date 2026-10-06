# Unitree G1 43-DOF Loco-Manipulation 파이프라인

## 1. 해결하려는 작업

Unitree G1 43-DOF가 상체 조작과 하체 보행을 함께 사용하여 다음 작업을 연속 수행하는 문제임.

```text
Box 탐색·접근 → 양손 정렬 → 파지·리프트 → 적재 상태 보행 → Bin 투입·안착
```

- 입력: 5프레임 proprioception history와 현재 joint/action, 속도 명령, Box/Bin QR 기반 위치·가시성
- 출력: 23-D 고수준 WBC 명령 또는 즉시 적용 가능한 43-D joint target
- 성공 조건: 양손 접촉, 12 cm 이상 리프트, 2초 연속 유지, Bin staging 도달, 양손 해제 후 Bin 내부 10-step 저속 안착
- 핵심 제약: 학생 정책에 Box/Bin의 정답 world pose, contact, lift flag 등의 privileged state 미사용

## 2. 전체 학습 흐름

| 순서 | 실제 작업명 | 학습 대상과 데이터 | 현재 상태 |
|---|---|---|---|
| 1 | **Box 접근 및 양손 Pre-grasp 정렬** | QR 기반 접근 속도와 양 손목·손가락의 23-D 고수준 명령 학습, Pink/Homie WBC를 거쳐 43관절 제어 | open-hand DAgger r05 학습 완료, 중앙 조건 FSM handoff 1/1 성공·무작위 반복 검증 필요 |
| 2 | **양손 파지·리프트·안정 유지** | 500개 strict teacher demo와 balanced DAgger correction으로 43-D joint target 학습 | 독립 physical rollout 19/20 성공, SHA gate 통과, 접근 이후 strict hold 1/1 성공 |
| 3 | **적재 상태 운반 및 Bin staging 접근** | 인증된 파지 snapshot에서 시작한 실제 physics rollout으로 43-D 정책 학습 | teacher 12/20 성공, 정책 학습 완료 |
| 4 | **Bin 투입·양손 해제·안착** | 실제 staging snapshot에서 시작한 Place v4 rollout으로 43-D 정책 학습 | teacher 6/12 성공, 정책 학습 완료·데이터 확대 필요 |
| 5 | **전체 작업 연결** | 물리 조건 기반 FSM이 위 네 정책 중 하나만 선택하여 실행 | 접근→파지·리프트 통과, Carry 진입 직후 drop 진단 후 100-step Carry blend 구현, 20회 end-to-end 검증 대기 |

기존 코드와 checkpoint의 `bc1`~`bc4`는 위 네 작업을 가리키는 호환용 파일명임. 보고와 설명에서는 번호 대신 실제 작업명 사용 권장.

## 3. Teacher 데이터와 Behavior Cloning

### 3.1 Teacher 원본과 500-demo corpus

- Arena의 G1 loco-manipulation trajectory를 현재 Isaac Lab 환경과 43-DOF joint order로 변환
- Box 기준 손목 궤적을 Isaac Lab Mimic의 translation-only 방식으로 재타기팅
- 첫 curriculum에서는 물체의 초기 `x/y`만 무작위화하고 크기·질량·마찰·yaw·table 높이는 고정
- 100개씩 5개 shard, 총 **500/500 accepted demo** 생성
- 양손 Box 접촉, 12 cm lift, 100-step 연속 hold를 모두 통과한 episode만 HDF5에 게시
- 학생 입력은 항상 `policy_history(460) + policy_current(57) = 517-D`; privileged field는 검증과 teacher label 생성에만 사용

### 3.2 작업별 BC 학습

- **Box 접근 및 Pre-grasp 정렬**: 접근 구간만 분리하여 23-D base/wrist/Dex3 명령 회귀 학습
- **양손 파지·리프트**: 첫 hand-close 직전부터 lift/hold까지를 분리하여 43-D processed joint target 회귀 학습
- **적재 상태 운반**: 인증된 파지 snapshot에서 복원한 carry rollout만 사용하여 학습
- **Bin 투입·안착**: carry가 실제로 staging에 도달해 저장한 snapshot에서 시작한 Place v4 rollout만 사용하여 학습
- train/validation/test를 step이 아니라 episode 단위로 분리하여 동일 rollout의 split 누출 방지
- action name 순서와 SHA-256 contract를 dataset/checkpoint/runtime에서 재검증하여 43-D 순서 오류 방지

## 4. 정해진 궤적을 넘어선 DAgger 데이터 증대

단순히 초기 Arena 궤적을 반복하지 않고, **현재 학생 정책을 physics에서 실행한 뒤 실패하기 쉬운 상태만 teacher가 다시 라벨링**하는 방식임.

1. 학생 정책으로 rollout 수행
2. 짧은 외력·토크 또는 관측 dropout으로 recovery 상태 생성
3. teacher가 현재 Box pose와 robot state를 읽어 그 시점의 교정 action 계산
4. 학생에게 실제 보였던 517-D 관측과 teacher action만 저장
5. nominal demo와 correction을 episode 단위로 혼합하여 재학습
6. 새 checkpoint를 다시 20회 physical rollout으로 평가하고 SHA-bound gate 재발급

교정 category는 다음과 같음.

- 한 손이 먼저 닿는 비대칭 접촉
- 손목 중심 또는 좌우 대칭이 Box 중심에서 벗어난 상태
- 양손 안에서 Box가 회전한 상태
- lift 중 미끄러짐 또는 양손 접촉 손실
- 일시적인 QR 관측 손실
- 정상 QR-visible 경계를 유지하기 위한 visible anchor

외란은 Box pose teleport가 아니라 lateral impulse, yaw torque, diagonal/down force로 적용하며, QR dropout은 actor observation만 가리고 teacher와 physics에는 영향을 주지 않음. 현재 승격된 접근 정책에는 open-hand 경계 correction 4,000개, 파지·리프트 정책에는 접근 종료 상태에서 수집한 balanced correction 453개가 직접 반영됨. 파지 correction은 `box_rotated 54`, `lift_slip 14`, `one_hand 108`, `qr_temporarily_missing 88`, `qr_visible_boundary 355`, `wrist_off_center 228`을 포함함.

## 5. FSM 구성

네 네트워크를 하나로 합치지 않고, 공통 517-D 관측을 현재 단계의 정책 하나에만 전달하는 구조임.

```text
BOX_APPROACH
  └─ Box visible + base 정렬 + 양손 fingertip cage
       → BIMANUAL_GRASP_AND_LIFT
          └─ 양손 접촉 + lift ≥ 0.12 m + hold ≥ 100 steps
               → LOADED_CARRY_TO_BIN
                  └─ Bin visible + 거리/yaw staging gate + hold 유지
                       → PLACE_AND_RELEASE
                          └─ Bin 내부 + 양손 release + 저속 settle ≥ 10 steps
                               → COMPLETE
```

- Box bad orientation은 모든 단계의 fatal failure
- carry/place 중 drop은 fatal failure
- 한 update에서 둘 이상의 단계를 건너뛰지 않는 전이
- 접근 단계는 Box QR navigation command, 파지·리프트 단계는 zero velocity, carry/place 단계는 Bin QR command 사용
- 파지 구간에는 12개 다리 관절을 접근 WBC의 zero-velocity stance로 계속 갱신하고, 최초 양손 접촉 시 14개 손가락 target 고정, 12 cm lift 시 상체 target 고정으로 100-step hold 보존
- Carry 진입 시 strict-hold의 실현 joint target에서 Carry 정책 target까지 smoothstep blend 적용 가능, 현재 후보값 100 step
- 평가 지표: `P(approach)`, `P(grasp|approach)`, `P(carry|grasp)`, `P(place|carry)`, 최종 task success
- 현재 실측 결과: 중앙 조건 1회에서 `P(approach)=1`, `P(grasp|approach)=1`, 최대 lift 0.425 m, 양손 lift hold 100 step 달성 후 Carry 진입
- 신규 Carry·Place 후보와 Carry 100-step blend의 20회 실측: 접근 19/20, 파지 8/19, Carry 0/8, 전체 0/20
- 현재 핵심 미완료 항목: 파지→Carry handoff 성공률 개선과 전체 FSM production gate 통과

## 6. 강화학습 파인튜닝

### 현재 구현

- Gym task: `Unitree-G1-43dof-BoxPickPlace`
- 4096개 병렬 환경의 RSL-RL PPO 실행 구성
- actor는 BC와 같은 QR/proprio 계열 관측 사용, critic만 object pose·target·task phase의 privileged state 사용
- 단계 성공 보상 `grasp +10`, `carry +25`, `place +80`과 접근·cage·lift·staging·placement progress shaping 사용
- 낙하, bad orientation, 불필요한 접촉, 에너지·action rate에 대한 penalty 사용
- Box 크기·질량·마찰, table 높이, robot reset pose를 무작위화하여 robustness 학습
- RND intrinsic reward로 희소한 조작 상태 탐색 보조, 넘어짐·불안정 상태에서는 safety mask로 intrinsic reward 차단

scratch 경로와 별도로 네 동결 Behavior Cloning 정책을 공통 43차원 명목 제어기로 사용하는 residual 경로를 구현함. 접근도 동일한 500-demo/DAgger 원천에서 43차원 정책을 추가 학습했으며, 단계 전환에는 100-step blend를 적용함. PPO actor는 43차원 residual만 출력하고 평균을 0으로 초기화하며, 실제 action은 `nominal + 0.20 × residual`로 구성함.

`RL Residual: Fine-tune Four BC Skills with PPO+RND`와 resume launch를 추가함. 128개 환경의 3회 smoke 이후 100회 추가 학습으로 307,200 transition을 처리하고 최종 checkpoint를 저장함. Random Network Distillation 안전 마스크까지 정상 작동했으나 최종 성공률 0%, 비정상 자세 종료율 100%로 gate에 실패하여 checkpoint를 승격하지 않음.

### 이후 학습 방식

1. 현재 구현된 네 작업 명목 제어기와 bounded residual 유지
2. 실패가 집중된 파지→Carry snapshot curriculum부터 학습
3. Carry → Place → 전체 episode 순으로 초기 상태 범위 확대
4. conditional physical gate로 기존 Behavior Cloning 성능 보존 여부 확인
5. actor에는 517-D 관측만 제공하고 critic에만 privileged state 제공하는 asymmetric PPO 유지
6. domain randomization과 DAgger 외란을 함께 적용하여 trajectory 주변 recovery 능력 학습
7. 단계별 conditional success와 fall/drop rate가 기존 BC gate보다 악화되면 checkpoint 승격 중단

직접 BC weight를 단일 PPO actor에 복사하는 방식은 23-D/43-D 출력과 네 정책 구조가 서로 달라 부적합함. FSM 위 residual RL 또는 단계별 actor fine-tuning이 현재 구조와 가장 자연스러운 연결 방식임.

## 7. 주요 파일

| 파일 | 역할 |
|---|---|
| `.vscode/launch.json` | 데이터 생성, BC/DAgger 학습, physical gate, FSM, PPO 실행 entry |
| `scripts/teacher/generate_randomized_box_teacher.py` | resumable 500-demo shard 생성과 manifest 관리 |
| `scripts/teacher/phase_bc.py` | 작업 구간 분리, BC/DAgger 학습, 평가와 checkpoint 인증 |
| `scripts/teacher/collect_dagger_corrections.py` | 학생 rollout 기반 교정 데이터 수집 |
| `scripts/teacher/evaluate_bc2_rollout.py` | 파지·리프트 physical rollout과 snapshot 게시 |
| `scripts/teacher/collect_carry_place_teacher.py` | 파지/carry snapshot 기반 운반·투입 teacher 수집 |
| `scripts/teacher/play_bc_skill_fsm.py` | 네 정책의 FSM 실행과 conditional metric 출력 |
| `scripts/teacher/audit_loco_manipulation_pipeline.py` | promoted artifact 존재·517-D 계약·checkpoint/gate SHA·FSM 완료 여부의 read-only 감사 |
| `source/.../utils/teacher/student_observation.py` | 517-D QR/proprio 관측 계약 |
| `source/.../utils/teacher/dagger.py` | recovery predicate, 외란 campaign, atomic recorder |
| `source/.../utils/teacher/skill_fsm.py` | 정책 routing, 물리 전이 gate, 성공률 계산 |
| `source/.../tasks/locomotion/robots/g1/29dof/box_carry_with_hand_env_cfg.py` | 43-DOF BoxPickPlace RL scene·관측·보상·randomization |
| `source/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | PPO 및 RND hyperparameter |

## 8. 결과 영상 위치

저장 디렉터리: `/home/mjkim/unitree_rl_lab/data/teacher/videos/`

- [Randomized teacher 접근·파지·리프트](data/teacher/videos/01_production_teacher_30s.mp4)
- [학습된 양손 파지·리프트 정책 3회 rollout](data/teacher/videos/02_bc2_student_grasp_lift_30s.mp4)
- [적재 상태 운반 및 Bin 투입·안착](data/teacher/videos/03_carry_place_phasewise_30s.mp4)
- [영상 manifest](data/teacher/videos/weekly_video_manifest.json): 해상도, FPS, 길이와 영상 성격 기록

세 번째 영상은 snapshot으로 연결한 phase-wise 성공 결과이며, 처음부터 끝까지 한 episode로 수행한 FSM 성공 증거는 아님.

## 9. 현재 결론

- 500-demo teacher corpus, 접근/파지 DAgger, 파지 physical gate 19/20, carry/place teacher와 작업별 정책까지 확보
- 정해진 trajectory 복제에서 벗어나 외력·관측 손실 상태의 online correction을 학습 데이터로 환류하는 구조 확보
- 학생 정책 연쇄에서 Box 접근부터 양손 파지·리프트·2초 유지와 Carry 단계 진입까지 확인
- Place 데이터 확대, 전체 FSM 20회 평가, BC 명목 제어기와 PPO residual 연결, 100회 추가 학습까지 완료
- 전체 production 완료를 위해 43차원 접근 명목 제어기 안정화, 파지→Carry handoff 개선, 전체 FSM gate 재통과 필요

## 10. 이어서 실행할 순서

1. 43차원 접근 명목 제어기를 물리 안정화한 뒤 파지→Carry 중심 curriculum 재학습
2. Carry 조건부 성공이 확보되면 신규 checkpoint로 전체 FSM 20회 재평가
3. conditional success 기준 통과 후 randomization 범위 확대와 sim-to-real 검증

현재 감사 결과는 `data/teacher/pipeline_status.json`에 저장됨. `production_complete=false`는 전체 FSM 20회 평가에서 Carry가 0/8이고, 100회 residual 학습도 성공률 0%로 gate를 통과하지 못했기 때문임. 구현과 실행은 완료되었으나 수렴된 robustness 결과는 아직 아님.
