# Unitree G1 43-DOF Loco-Manipulation 상세 설명

## 1. 목표 작업

Unitree G1이 보행과 양손 조작을 연속으로 사용해 다음 작업을 수행하도록 만드는 프로젝트임.

```text
Box 탐색·접근
→ 양손 Pre-grasp 정렬
→ 양손 파지·리프트·안정 유지
→ Box를 든 상태로 Bin까지 운반
→ Bin 안에 투입·양손 해제·안착
```

작업별 정책 네 개를 따로 학습하고, 물리 조건을 검사하는 유한 상태 기계가 현재 실행할 정책을 선택하는 구조임.

---

## 2. Arena 원본 데이터의 정확한 관절 구성

사용한 원본 파일은 다음과 같음.

```text
/home/mjkim/IsaacLab-Arena/data/
arena_g1_loco_manipulation_dataset_generated_small.hdf5
```

실제 HDF5를 확인한 결과는 다음과 같음.

- Arena task: `galileo_g1_locomanip_pick_and_place`
- 성공 episode: 1개
- episode 길이: 853 step
- simulation time step: 0.005초
- control decimation: 4
- 실제 control interval: 0.02초, 50 Hz
- episode 실제 시간: 17.06초
- raw high-level action: 23차원
- robot joint position: 43차원
- processed joint target: 43차원

따라서 원본은 29관절 데이터가 아님. G1 몸체 29관절과 양손 Dex3 14관절을 합친 43관절 데이터임.

```text
몸체 29관절 + 왼손 7관절 + 오른손 7관절 = 43관절
```

현재 Box task 코드가 `robots/g1/29dof/` 아래에 있는 것은 기존 repository 디렉터리 이름을 유지한 결과임. 실제 환경은 `UNITREE_G1_43DOF_WITH_HAND_CFG`를 사용함.

### Arena와 현재 환경의 joint action order 비교

production HDF5의 `metadata/processed_joint_names`와 Arena의 `lab_g1_joints_order_43dof.yaml` 계약을 직접 비교한 결과, 현재 저장된 두 순서는 완전히 동일함.

| Index | Arena order | 현재 BoxPickPlace action order | 비교 |
|---:|---|---|---|
| 0 | left_hip_pitch_joint | left_hip_pitch_joint | 동일 |
| 1 | right_hip_pitch_joint | right_hip_pitch_joint | 동일 |
| 2 | waist_yaw_joint | waist_yaw_joint | 동일 |
| 3 | left_hip_roll_joint | left_hip_roll_joint | 동일 |
| 4 | right_hip_roll_joint | right_hip_roll_joint | 동일 |
| 5 | waist_roll_joint | waist_roll_joint | 동일 |
| 6 | left_hip_yaw_joint | left_hip_yaw_joint | 동일 |
| 7 | right_hip_yaw_joint | right_hip_yaw_joint | 동일 |
| 8 | waist_pitch_joint | waist_pitch_joint | 동일 |
| 9 | left_knee_joint | left_knee_joint | 동일 |
| 10 | right_knee_joint | right_knee_joint | 동일 |
| 11 | left_shoulder_pitch_joint | left_shoulder_pitch_joint | 동일 |
| 12 | right_shoulder_pitch_joint | right_shoulder_pitch_joint | 동일 |
| 13 | left_ankle_pitch_joint | left_ankle_pitch_joint | 동일 |
| 14 | right_ankle_pitch_joint | right_ankle_pitch_joint | 동일 |
| 15 | left_shoulder_roll_joint | left_shoulder_roll_joint | 동일 |
| 16 | right_shoulder_roll_joint | right_shoulder_roll_joint | 동일 |
| 17 | left_ankle_roll_joint | left_ankle_roll_joint | 동일 |
| 18 | right_ankle_roll_joint | right_ankle_roll_joint | 동일 |
| 19 | left_shoulder_yaw_joint | left_shoulder_yaw_joint | 동일 |
| 20 | right_shoulder_yaw_joint | right_shoulder_yaw_joint | 동일 |
| 21 | left_elbow_joint | left_elbow_joint | 동일 |
| 22 | right_elbow_joint | right_elbow_joint | 동일 |
| 23 | left_wrist_roll_joint | left_wrist_roll_joint | 동일 |
| 24 | right_wrist_roll_joint | right_wrist_roll_joint | 동일 |
| 25 | left_wrist_pitch_joint | left_wrist_pitch_joint | 동일 |
| 26 | right_wrist_pitch_joint | right_wrist_pitch_joint | 동일 |
| 27 | left_wrist_yaw_joint | left_wrist_yaw_joint | 동일 |
| 28 | right_wrist_yaw_joint | right_wrist_yaw_joint | 동일 |
| 29 | left_hand_index_0_joint | left_hand_index_0_joint | 동일 |
| 30 | left_hand_middle_0_joint | left_hand_middle_0_joint | 동일 |
| 31 | left_hand_thumb_0_joint | left_hand_thumb_0_joint | 동일 |
| 32 | right_hand_index_0_joint | right_hand_index_0_joint | 동일 |
| 33 | right_hand_middle_0_joint | right_hand_middle_0_joint | 동일 |
| 34 | right_hand_thumb_0_joint | right_hand_thumb_0_joint | 동일 |
| 35 | left_hand_index_1_joint | left_hand_index_1_joint | 동일 |
| 36 | left_hand_middle_1_joint | left_hand_middle_1_joint | 동일 |
| 37 | left_hand_thumb_1_joint | left_hand_thumb_1_joint | 동일 |
| 38 | right_hand_index_1_joint | right_hand_index_1_joint | 동일 |
| 39 | right_hand_middle_1_joint | right_hand_middle_1_joint | 동일 |
| 40 | right_hand_thumb_1_joint | right_hand_thumb_1_joint | 동일 |
| 41 | left_hand_thumb_2_joint | left_hand_thumb_2_joint | 동일 |
| 42 | right_hand_thumb_2_joint | right_hand_thumb_2_joint | 동일 |

따라서 현재 데이터에 대해 index permutation은 identity임.

```text
Arena index 0..42 → 현재 환경 index 0..42
```

그럼에도 코드가 이름 기반으로 재배열하는 이유는 다음과 같음.

- URDF 또는 USD asset 변경으로 action order가 달라질 수 있음
- Arena controller 내부 model order와 simulator articulation order가 달라질 수 있음
- 43이라는 차원만 같고 좌우 관절 channel이 바뀌는 silent error를 방지해야 함
- dataset, checkpoint, runtime 사이의 정확한 의미를 SHA-256으로 묶어야 함

즉 “실제로 달라서 바꾼 변환”이라기보다 “순서가 달라도 안전하도록 이름으로 정렬하고, 현재는 동일함을 확인한 계약”임.

---

## 3. 리타게팅과 두 Python 환경의 역할

Arena와 현재 Unitree RL Lab은 Isaac Sim과 Isaac Lab stack이 달라 한 Python process에 동시에 import할 수 없음. 이를 해결하기 위해 simulator와 Whole-Body Control을 서로 다른 process에 분리함.

```text
현재 Unitree RL Lab process
  ├─ Unitree-G1-43dof-BoxPickPlace physics
  ├─ Box/Bin/robot state
  ├─ QR/proprioception observation
  ├─ contact/lift/drop 판정
  └─ HDF5 기록

Arena worker process
  ├─ Pink upper-body inverse kinematics
  ├─ HomieV2 lower-body locomotion controller
  └─ 23-D high-level command → named 43-D joint target
```

### 3.1 현재 process가 만드는 23차원 명령

각 step에서 현재 process는 다음 고수준 명령을 구성함.

| 범위 | 의미 | 차원 |
|---|---|---:|
| 0:2 | 좌우 손 open/close state | 2 |
| 2:5 | pelvis frame 왼쪽 손목 위치 | 3 |
| 5:9 | pelvis frame 왼쪽 손목 quaternion, xyzw | 4 |
| 9:12 | pelvis frame 오른쪽 손목 위치 | 3 |
| 12:16 | pelvis frame 오른쪽 손목 quaternion, xyzw | 4 |
| 16:19 | base velocity `vx, vy, yaw rate` | 3 |
| 19:20 | pelvis 높이 | 1 |
| 20:23 | torso roll, pitch, yaw | 3 |

목표 손목 pose만 보내는 것이 아님. Whole-Body Control이 현재 물리 상태를 기준으로 action을 계산할 수 있도록 다음 값도 JSON request에 포함함.

- 현재 simulator joint name 43개
- 현재 joint position 43개
- 현재 joint velocity 43개
- default joint position 43개
- floating-base position과 quaternion
- floating-base body-frame linear/angular velocity
- body-frame projected gravity
- root angular velocity
- torso quaternion
- torso body-frame angular velocity

JSON에는 plain number와 array만 포함되며 simulator object나 tensor를 process 경계 너머로 전달하지 않음.

### 3.2 JSON-lines bridge

현재 process가 Arena의 `.venv/bin/python`으로 `arena_wbc_worker.py`를 persistent subprocess로 실행함.

```text
현재 process stdin
  → 한 줄의 JSON request
Arena worker
  → ARENA_WBC_JSON:{...} 한 줄의 JSON response
현재 process stdout reader
```

- startup timeout: 기본 45초
- step response timeout: 기본 5초
- Arena worker의 일반 log와 protocol response는 prefix로 구분
- reset 시 Pink iterative state와 Homie locomotion history를 함께 초기화
- caller의 Python path와 native library path가 Arena process로 섞이지 않도록 환경 변수를 분리

### 3.3 Arena worker 내부 계산

1. 전달받은 joint state를 joint name 기준 Arena robot model order로 정렬
2. 23차원 명령의 좌우 손목 위치·quaternion을 pelvis-frame 4×4 transform으로 변환
3. 현재 arm configuration으로 Pink inverse kinematics를 warm-start
4. 초기 step에는 Pink inverse kinematics를 50회 반복해 수렴 상태를 준비
5. Pink가 양팔·손목과 손가락에 해당하는 upper-body target 계산
6. HomieV2에 현재 43관절 state, base state, gravity, torso state 전달
7. HomieV2에 base velocity, pelvis height, torso orientation goal 전달
8. Pink upper-body target과 HomieV2 lower-body locomotion을 결합
9. absolute joint-position target 43개와 해당 joint name 반환

### 3.4 현재 환경에서 target 적용

Arena worker가 반환한 순서를 그대로 index로 신뢰하지 않음. 현재 `JointPositionAction`의 이름 순서로 다시 lookup함.

```text
target_by_name = {joint_name: absolute_target}
absolute = [target_by_name[name] for name in current_action_joint_names]
```

현재 action term은 default joint position offset과 0.25 scale을 사용하므로 simulator에 전달할 raw action은 다음과 같음.

```text
raw_action = (absolute_joint_target - default_joint_offset) / 0.25
```

이 raw action을 `env.step()`에 전달하고 현재 환경의 physics가 다음 state를 계산함. 이후 현재 환경에서 contact, lift, robot orientation, drop을 판정하고 HDF5에 기록함.

```text
현재 physics state
→ 23-D high-level command와 current state를 JSON 전달
→ Arena Pink/Homie가 named 43-D absolute target 계산
→ 현재 action order로 이름 기반 정렬
→ raw JointPositionAction으로 변환
→ 현재 환경 env.step()
→ 현재 환경에서 성공 조건 판정 및 데이터 저장
```

---

## 4. Box 위치 리타게팅과 500개 Teacher demonstration

500개 demonstration은 Arena simulator에서 생성한 것이 아니라 현재 `Unitree-G1-43dof-BoxPickPlace` 환경의 실제 physics rollout으로 생성함. Arena는 원본 trajectory와 Pink/Homie controller 계산을 제공함.

### 4.1 Translation delta

원본 Arena trajectory에서 시간별 Box world pose와 좌우 손목 world pose를 읽음.

```text
source Box trajectory: T_source_box(t)
source wrist trajectories: T_source_left(t), T_source_right(t)
sampled current Box initial pose: T_target_box(0)
```

Isaac Lab Mimic의 `get_delta_pose_with_scheme(..., TRANSLATE)`를 이용해 다음 translation-only delta를 구함.

```text
Delta = T_target_box(0) × inverse(T_source_box(0))
```

여기서 rotation part는 identity로 강제함. 같은 delta를 Box와 양쪽 손목 trajectory에 모두 적용함.

```text
T_target_box(t)   = Delta × T_source_box(t)
T_target_left(t)  = Delta × T_source_left(t)
T_target_right(t) = Delta × T_source_right(t)
```

이 방식은 모든 step에서 Box-relative wrist transform을 보존함.

```text
inverse(T_source_box(t)) × T_source_wrist(t)
= inverse(T_target_box(t)) × T_target_wrist(t)
```

full rotation retargeting을 사용하지 않은 이유는 Arena Box와 현재 Box asset의 semantic yaw convention 차이로 양손 궤적이 뒤집힐 수 있기 때문임.

### 4.2 첫 curriculum의 randomization

무작위화한 값:

- Box 초기 x: 0.618–0.622 m
- Box 초기 y: -0.002–0.002 m

고정한 값:

- Box 크기: 0.2 × 0.2 × 0.2 m
- Box 질량: 0.2 kg
- Box yaw: 0 rad
- 정·동 마찰과 restitution
- Table 높이
- Robot 초기 root pose와 joint velocity

### 4.3 Demonstration 길이

원본 Arena episode와 길이가 같지 않음.

| 데이터 | 포함 범위 | Step | 실제 시간 |
|---|---|---:|---:|
| Arena 원본 | 접근·파지·Carry·Place 전체 | 853 | 17.06초 |
| 500-demo 평균 | 접근·파지·lift·strict hold | 260.788 | 5.216초 |
| 500-demo 최소 | 동일 | 260 | 5.20초 |
| 500-demo 최대 | hold 성공이 늦어진 경우 | 309 | 6.18초 |

500-demo 중 접근 구간은 모든 episode에서 124 step이고, 이후 파지·리프트·hold 구간은 평균 136.788 step임. 성공 조건에 도달하면 episode를 끝내므로 길이가 일부 달라짐.

### 4.4 저장 성공 조건

다음 조건을 동시에 연속으로 만족한 episode만 저장함.

#### 양손 Box 접촉

- 왼손 fingertip 중 하나 이상이 Box에 접촉
- 오른손 fingertip 중 하나 이상이 Box에 접촉
- Box만 filter한 contact force 사용
- threshold: 1 N

한 손의 여러 손가락만 닿은 상태는 양손 접촉으로 인정하지 않음.

#### 12 cm 이상 lift

Table의 world height를 매 step 직접 빼는 방식이 아니라 episode 초기 Box 중심 높이를 기준으로 계산함.

```text
lift_height = current_box_center_z - initial_box_center_z
```

현재 고정 curriculum에서는 Table 표면이 0.74 m, Box 높이가 0.20 m이므로 Box 중심 초기 높이는 약 0.84 m임. 12 cm lift 성공 시 Box 중심은 0.96 m 이상임. Table과 Box 크기가 고정이므로 이는 Box 바닥이 초기 Table 표면에서 12 cm 이상 올라간 것과 동치임.

#### 100-step 연속 hold

제어 주기가 50 Hz이므로 다음과 같음.

```text
100 step × 0.02초 = 2.0초
```

2초 동안 매 step 다음 조건을 모두 만족해야 함.

```text
left_contact AND right_contact AND lift_height >= 0.12 m
```

한 step이라도 접촉이 끊기거나 높이가 내려가면 counter가 0으로 초기화됨. bad orientation 또는 Box drop episode도 게시하지 않음.

---

## 5. 네 작업의 정책 입력과 출력

### 5.1 모든 학생 정책의 공통 입력: 517차원

학생 정책 네 개는 같은 입력 contract를 사용함.

```text
5-frame proprioception history 460
+ current observation 57
= 517차원
```

#### 과거 5프레임: 460차원

한 프레임은 다음 92차원임.

| Feature | 차원 |
|---|---:|
| Base angular velocity | 3 |
| Projected gravity | 3 |
| 43개 joint relative position | 43 |
| 43개 joint relative velocity | 43 |
| 합계 | 92 |

`92 × 5 = 460`임.

#### 현재 관측: 57차원

| Feature | 차원 |
|---|---:|
| 이전 action | 43 |
| 현재 base velocity command | 3 |
| Box QR visibility | 1 |
| Camera-frame Box position | 3 |
| Bin QR visibility | 1 |
| Camera-frame Bin position | 3 |
| Camera-frame Table position | 3 |
| 합계 | 57 |

학생 입력에서 제외한 privileged 정보:

- Box와 Bin world pose
- 실제 양손 contact
- 실제 lift와 drop flag
- Box 크기
- 손목·fingertip world pose
- 성공 label

이 정보는 Teacher action 생성, 데이터 filtering, 물리 gate 검증에만 사용함.

### 5.2 작업별 정책

| 작업 | 학습 데이터 | 입력 | 출력 | Runtime 적용 |
|---|---|---:|---:|---|
| Box 접근·Pre-grasp | 500-demo 접근 prefix + open-hand DAgger | 517 | 23-D high-level | Pink/Homie → 43-D target |
| Box 파지·리프트·hold | 500-demo close/lift/hold suffix + balanced DAgger | 517 | 43-D absolute joint target | 직접 JointPositionAction 변환 |
| Box Carry | 인증 파지 snapshot에서 수집한 Carry rollout | 517 | 43-D absolute joint target | 직접 JointPositionAction 변환 |
| Bin Place·release | 실제 Carry staging snapshot에서 수집한 Place v4 rollout | 517 | 43-D absolute joint target | 직접 JointPositionAction 변환 |

#### Box 접근·Pre-grasp

- Box QR로 생성한 접근 velocity command 사용
- 양손은 open 상태 유지
- 양손 손목을 Box의 pre-grasp cage 위치로 정렬
- 출력은 손 상태, 손목 pose, base velocity, pelvis 높이, torso orientation을 포함한 23차원
- Runtime마다 Pink/Homie가 43차원 target으로 변환

#### Box 파지·리프트·hold

- 첫 hand-close, 첫 contact 또는 lift trigger보다 2 step 앞에서 시작
- velocity command는 `[0, 0, 0]`
- 양손 close, contact, 12 cm lift, 2초 hold 전체를 담당
- 출력은 43차원 absolute joint-position target

#### Box Carry

- SHA로 인증된 파지·hold physical snapshot에서만 Teacher rollout 시작
- velocity command는 Bin QR 기반 navigation command
- 실제로 측정한 Box-relative 양손 grasp를 유지하면서 보행
- 출력은 43차원 absolute joint-position target

#### Bin Place·release

- Carry가 실제 staging에 도달한 physical snapshot에서만 시작
- Box를 들어 올리고 Bin 위로 이동한 뒤 낮추고 손을 여는 과정 담당
- Bin 내부, 양손 release, 10-step low-speed settle을 성공 기준으로 사용
- 출력은 43차원 absolute joint-position target

---

## 6. 시연 모방 학습 방식

Adversarial Motion Priors 기반 적대적 모방학습이 아님. 일반적인 supervised Behavior Cloning을 사용함.

```text
517-D observation
→ Linear 512 + ELU
→ Linear 512 + ELU
→ Linear 256 + ELU
→ 23-D 또는 43-D action
```

- optimizer: AdamW
- 기본 loss: normalized Teacher action에 대한 Mean Squared Error
- train/validation/test split: step이 아닌 episode 단위
- checkpoint 선택: validation normalized Mean Squared Error
- DAgger sample: 같은 Mean Squared Error에 correction sample weight 적용 가능

사용하지 않은 요소:

- discriminator
- adversarial reward
- Adversarial Motion Priors motion prior
- Generative Adversarial Imitation Learning

즉 네 작업 정책의 학습은 `Teacher action을 직접 회귀하는 Behavior Cloning`임.

---

## 7. DAgger 교정 데이터 증대

한 개 원본 궤적을 단순 복사하는 데서 끝내지 않고 현재 학생 정책을 physics에서 실행해 trajectory 주변의 실패 상태를 수집함.

```text
현재 학생 정책 rollout
→ 외력·yaw torque·down force 또는 QR dropout 적용
→ 실패 가능 상태 감지
→ privileged Teacher가 현재 state의 correction action 계산
→ 학생이 실제 본 517-D observation과 Teacher action 저장
→ nominal data와 episode 단위로 병합
→ Behavior Cloning 재학습
```

교정 category:

- 한 손만 먼저 접촉한 상태
- 손목 중심이나 좌우 대칭이 어긋난 상태
- Box가 양손 안에서 회전한 상태
- lift 중 미끄러지거나 한쪽 접촉을 잃은 상태
- Box QR이 일시적으로 보이지 않는 상태
- QR이 보이는 정상 경계 상태

QR dropout은 학생 observation만 가리며 physics와 Teacher의 Box state는 바꾸지 않음. Box 위치를 순간이동해 correction을 만들지 않고 실제 force와 torque를 사용함.

---

## 8. 물리 rollout과 checkpoint 무결성 gate

offline action error가 작아도 실제 robot이 파지와 hold를 수행한다는 보장은 없음. 따라서 파지·리프트 정책은 별도 physical rollout gate를 통과해야 함.

검사 항목:

- held-out validation normalized Mean Squared Error ≤ 0.25
- physical rollout 20회 이상
- 양손 12 cm lift와 2초 hold 성공률 ≥ 80%
- bad orientation rate ≤ 5%
- 성공 episode별 strict 100-step hold audit
- checkpoint SHA-256과 rollout metrics SHA-256 일치
- 성공 physical snapshot 수와 provenance 검사

현재 승격된 파지·리프트 정책 결과:

- 20회 중 19회 성공
- 성공률 95%
- bad orientation 0%
- 성공 physical snapshot 19개

Carry는 이 gate가 가리키는 정확한 checkpoint의 성공 snapshot만 입력으로 허용함.

---

## 9. 유한 상태 기계로 네 정책 연결

유한 상태 기계는 학습 network가 아니라 현재 물리 상태를 검사해 실행할 정책을 선택하는 router임.

```text
BOX_APPROACH
  └─ Box visible + base 정렬 + fingertip cage
       → BIMANUAL_GRASP_AND_LIFT
          └─ 양손 접촉 + 12 cm lift + 100-step hold
               → LOADED_CARRY_TO_BIN
                  └─ Bin staging + bilateral hold 유지
                       → PLACE_AND_RELEASE
                          └─ Bin 내부 + 양손 release + 10-step 저속 settle
                               → COMPLETE
```

주요 안전 조건:

- 한 update에 한 단계만 전이
- 모든 단계에서 robot bad orientation은 fatal failure
- Carry와 Place에서 Box drop은 fatal failure
- 접근→파지 target은 smoothstep으로 blending
- 파지 중 12개 다리 관절은 zero-velocity Whole-Body Control로 안정화
- 최초 양손 접촉 시 손가락 target 고정
- 17 cm lift 시 상체 target을 고정해 settling 후에도 12 cm strict hold 여유 보존
- Carry 진입 시 strict-hold target에서 Carry target으로 candidate 100-step blending

현재 학생 연쇄에서 확인된 범위:

```text
Box 접근 성공
→ 양손 접촉
→ 최대 0.425 m lift
→ 100-step bilateral hold
→ Carry 진입
```

첫 12 cm 도달 즉시 상체를 고정하면 물리 settling으로 높이가 임계값 아래로 내려가던 문제를 확인해, Teacher와 동일한 5 cm lift 여유를 적용함. 별도 고정-seed 20회 평가 결과는 접근 20/20, 접근 조건부 파지·리프트 16/20, 파지 조건부 2초 Handoff 15/16임. 파지와 Handoff의 80% gate는 통과했으며, 이후 병목은 Loaded Stand에서 실제 운반 보행으로 넘어가는 구간임.

---

## 10. 강화학습 기반 robustness 파인튜닝 준비

별도 `Unitree-G1-43dof-BoxPickPlace` RSL-RL 환경에 Proximal Policy Optimization과 Random Network Distillation 구성을 마련함.

- actor: QR/proprioception 계열 관측
- critic: object pose, target, task phase를 포함한 privileged observation
- 단계 성공 보상: grasp, carry, place
- shaping: 접근, fingertip cage, lift, staging, placement progress
- penalty: fall, drop, bad orientation, energy, action rate, 불필요한 contact
- domain randomization: Box 크기·질량·마찰, Table 높이, robot reset pose
- unsafe state에서는 Random Network Distillation intrinsic reward 차단

scratch 경로 외에 다음 residual 경로를 구현함.

```text
action = frozen 43-D Behavior Cloning nominal action + 0.20 × clipped PPO residual
```

기존 접근 정책의 23차원 출력 문제를 해소하기 위해 같은 500-demo와 접근 DAgger 데이터로 43차원 접근 명목 정책을 추가 학습함. 네 동결 정책의 517차원 입력과 43차원 출력을 통일하고, 물리 단계에 따라 하나를 선택하며 전환 시 100-step smoothstep blend를 적용함. Proximal Policy Optimization residual 평균은 0으로 초기화하고 Random Network Distillation 안전 마스크를 유지함.

128개 병렬 환경에서 3회 smoke를 통과한 뒤 100회 추가 학습과 307,200 transition 처리를 완료함. 최종 checkpoint까지 정상 저장되었으나 성공률 0%, 비정상 자세 종료율 100%로 production gate를 통과하지 못함. 공통 43차원 접근 명목 제어기가 무작위 reset 분포에서 넘어지며 후속 보상에 도달하지 못한 결과이므로 해당 checkpoint를 승격하지 않음.

---

## 11. 현재 완료 경계와 남은 작업

완료된 항목:

- Arena 43관절 원본 데이터 audit
- 현재 43관절 환경과 이름 기반 action contract
- Box x/y randomization teacher demonstration 500개
- 접근, 파지·리프트, Carry, Place 정책 학습
- 파지·리프트 독립 physical gate 19/20
- 학생 접근→파지→2초 hold→Carry 진입 확인
- DAgger와 checkpoint provenance pipeline
- 유한 상태 기계, 잔차 강화학습 환경, 100회 Proximal Policy Optimization 실행

남은 핵심 항목:

1. 43차원 접근 명목 제어기의 물리 안정화 후 파지→Carry handoff snapshot curriculum 적용
2. Carry conditional success가 확보된 checkpoint의 전체 유한 상태 기계 20회 재평가
3. production conditional success 기준 통과 후 sim-to-real 안전 검증

## 12. 주요 코드 위치

| 파일 | 역할 |
|---|---|
| `scripts/teacher/arena_wbc_worker.py` | Arena Pink/Homie JSON worker |
| `source/unitree_rl_lab/unitree_rl_lab/utils/arena_wbc_bridge.py` | 별도 Arena Python process 관리와 JSON-lines protocol |
| `source/unitree_rl_lab/unitree_rl_lab/utils/bc_residual.py` | 네 동결 43차원 정책 선택, 전이 blend, PPO residual 합성 |
| `source/unitree_rl_lab/unitree_rl_lab/utils/teacher/arena_dataset.py` | Arena 43관절/action/pose audit와 extraction |
| `source/unitree_rl_lab/unitree_rl_lab/utils/teacher/mimic_xy_retarget.py` | translation-only Box·양손 궤적 리타게팅 |
| `scripts/teacher/collect_nominal_box_teacher.py` | 현재 환경에서 teacher rollout과 strict 저장 |
| `scripts/teacher/generate_randomized_box_teacher.py` | 500-demo shard 생성·resume·audit |
| `source/unitree_rl_lab/unitree_rl_lab/utils/teacher/student_observation.py` | 학생 517차원 input contract |
| `scripts/teacher/phase_bc.py` | 작업별 Behavior Cloning train/evaluate/certify command |
| `source/unitree_rl_lab/unitree_rl_lab/utils/teacher/phase_bc.py` | phase split, network, Mean Squared Error, checkpoint contract |
| `scripts/teacher/collect_dagger_corrections.py` | 학생 rollout 기반 DAgger correction 수집 |
| `scripts/teacher/evaluate_bc2_rollout.py` | 파지·리프트 physical gate와 snapshot 생성 |
| `scripts/teacher/collect_carry_place_teacher.py` | physical snapshot 기반 Carry·Place Teacher 수집 |
| `source/unitree_rl_lab/unitree_rl_lab/utils/teacher/skill_fsm.py` | 단계 전이와 conditional metric |
| `scripts/teacher/play_bc_skill_fsm.py` | 네 정책 유한 상태 기계 runtime |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/box_carry_with_hand_env_cfg.py` | 43관절 BoxPickPlace 강화학습 환경 |
| `.vscode/launch.json` | 전체 실행 entry |
