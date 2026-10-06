# Unitree G1 43관절 Loco-Manipulation 개발 및 학습 프레임워크

작성일: 2026년 10월 5일  
대상 프로젝트: `/home/mjkim/unitree_rl_lab`  
정리 기준: 현재 구현 코드, Teacher 데이터 manifest, Student checkpoint의 학습 기록, 물리 평가 및 수집 로그

이 문서는 지금까지 진행한 작업을 **환경 구성, Teacher 데이터 수집, Student 정책 학습, FSM 구성**의 네 항목으로 정리한 문서입니다. 과거 실험의 성공 사례와 현재 설정의 결과를 구분하며, 기능이 구현되어 있다는 사실을 전체 작업의 성공 인증과 동일하게 취급하지 않습니다.

최종 목표는 로봇이 상자를 인식하여 접근하고, 양손으로 들어 올린 뒤, 목적지까지 운반하여 상자함에 내려놓는 연속 작업을 수행하는 것입니다. 현재는 시뮬레이션의 단계별 정책과 연결 제어를 개발·검증하는 단계이며, 5 m 랜덤 배치에서 전체 작업을 안정적으로 성공하는 상태나 실물 적용이 완료된 상태는 아닙니다.

전체 개발 흐름은 다음과 같습니다.

`현재 환경 구성 → Arena 기준 궤적의 작업 공간 리타게팅 → 물리적으로 성공한 Teacher 시연 선별 → 작업별 Student 모방학습 → Student 상태의 Teacher 교정 데이터 추가 → FSM 연결 및 평가`

목차:

1. [환경 구성](#1-환경-구성)
2. [Teacher 데이터 수집](#2-teacher-데이터-수집)
3. [Student 정책 학습](#3-student-정책-학습)
4. [FSM 구성](#4-fsm-구성)

## 1. 환경 구성

### 1.1 수행하려는 작업과 네 가지 하위 작업

Loco-manipulation은 이동을 뜻하는 locomotion과 물체 조작을 뜻하는 manipulation을 결합한 작업입니다. 단순히 상자를 집는 동작뿐 아니라, 상자를 유지하면서 걷고 목적지에 내려놓는 동작까지 포함합니다.

긴 작업을 하나의 신경망으로 한 번에 해결하기보다, 다음 네 작업에 해당하는 Student 정책을 별도로 학습하고 연결하는 구조를 사용합니다.

| 작업 | 로봇이 수행하는 동작 | 해당 단계의 목표 |
|---|---|---|
| 상자 접근 및 양손 정렬 — Approach | 상자를 찾고 앞으로 이동한 뒤 양손을 파지 직전 위치로 이동 | 다음 정책이 상자를 잡을 수 있는 자세 확보 |
| 양손 파지 및 들어 올리기 — Grasp & Lift | 양손으로 상자를 잡고 위로 들어 올림 | 양손 접촉, 12 cm 이상 상승, 연속 유지 |
| 상자를 든 상태의 운반 — Carry | 상자를 유지하며 상자함 앞까지 이동 | 파지를 잃지 않고 투입 가능한 위치 도달 |
| 상자함 투입 — Place | 상자를 상자함 안으로 이동하고 손을 열어 내려놓음 | 상자가 상자함 안에서 안정된 상태 확보 |

여기서 정책은 로봇의 관측을 입력받아 행동을 출력하는 신경망입니다. 뒤에서 설명할 Teacher는 현재 프로젝트에서 이 네 신경망을 공동 학습하는 모델이 아니라, 학습용 정답 행동을 계산하는 데이터 생성 제어기입니다.

### 1.2 로봇과 물리 환경

환경 이름은 `Unitree-G1-43dof-BoxPickPlace`입니다. G1 몸체와 Dex3 양손, 테이블, 상자, 상자함 및 카메라 관측을 구성한 Isaac Lab 환경입니다.

| 항목 | 현재 구성 |
|---|---|
| 로봇 | Unitree G1 몸체 29관절 + Dex3 양손 14관절 = 43관절 |
| 관절 구성 | 하체 12, 허리 3, 양팔·손목 14, 양손 손가락 14 |
| 물리 계산 간격 | 0.005초 |
| 행동 적용 간격 | 물리 4 step마다 1회, 즉 0.02초 / 50 Hz |
| 환경의 기본 에피소드 길이 | 30초. 수집·평가 스크립트에서 별도로 변경 가능 |
| 강화학습용 환경 기본값 | 2,048개 병렬 환경 |
| Teacher 수집 및 FSM 재생 | 현재 스크립트에서는 1개 물리 환경 |
| Behavior Cloning 학습 | 이미 저장된 데이터의 미니배치 학습. 2,048개 시뮬레이터를 동시에 실행하는 방식이 아님 |
| 지형 | 현재 상자 작업 환경은 평지 |

환경 파일 경로에 `29dof`가 들어 있지만 실제로 사용하는 자산은 `UNITREE_G1_43DOF_WITH_HAND_CFG`입니다. 파일 경로 이름만 보고 로봇이 29관절이라고 판단하면 안 됩니다.

테이블과 상자함은 이동 가능한 배치 구성을 갖는 충돌 물체이고, 상자는 중력과 접촉에 의해 움직이는 동적 물체입니다. 상자를 손 안으로 임의로 옮겨 성공을 만드는 방식은 사용하지 않습니다.

### 1.3 환경 랜덤화의 범위와 적용 시점

정해진 위치에서만 성공하는 정책이 되지 않도록, 배치와 물성을 바꿀 수 있는 기능을 구현했습니다.

| 항목 | 환경 설정 범위 | 적용 시점 |
|---|---|---|
| 테이블·상자함 중심 | 기본 x/y 각각 약 −2.05~2.05 m의 안전 영역 | 에피소드 reset |
| 전체 작업 공간 | 물체 여유 공간을 포함한 약 5 m × 5 m 범위 | 배치 설정 |
| 로봇 원점–테이블 거리 | 0.75~2.20 m | 배치 표본화 시 검사 |
| 로봇 원점–상자함 거리 | 0.75~2.30 m | 배치 표본화 시 검사 |
| 테이블–상자함 거리 | 1.00~3.20 m | 배치 표본화 시 검사 |
| 테이블 위 상자 위치 | 테이블 중심 기준 x/y 각각 ±0.05 m | 에피소드 reset |
| 테이블 높이 | 기준 0.74 m에서 ±0.03 m | 환경 시작 시 표본화 |
| 상자 크기 | 각 축 0.10~0.30 m | 환경 자산 생성/초기화 시 표본화 |
| 상자 질량 | 0.10~0.40 kg | 환경 시작 시 표본화 |
| 상자 마찰 | 정지 0.8~1.8, 동적 0.6~1.5 | 환경 시작 시 표본화 |
| 상자 반발 계수 | 0~0.05 | 환경 시작 시 표본화 |
| 로봇 초기 상태 | x/y ±0.05 m, yaw ±0.08 rad, 관절 속도 ±0.1 | 에피소드 reset |

테이블·상자함을 사각형 안에서 아무렇게나 생성하는 것이 아니라, 원점 및 서로 사이의 거리 조건을 만족하는 배치를 선택합니다. 테이블 높이와 상자 크기를 이용해 상자의 초기 높이도 계산합니다.

**환경에 랜덤화 기능이 있는 것과, 모든 학습 데이터가 그 범위에서 수집됐다는 것은 다릅니다.** 기존 접근·파지 성공 시연 500개는 상자 x=0.618~0.622 m, y=−0.002~0.002 m의 좁은 범위이고, 크기·질량·마찰은 고정한 데이터입니다. 최근 5 m 수집에서는 넓은 배치를 실제로 켰지만, 수집 스크립트가 물성을 고정하는 설정도 사용합니다.

또한 저장된 이전 단계 상태를 복원하면 로봇, 상자, 테이블 및 상자함의 배치까지 함께 복원됩니다. 복원 후 같은 상태를 반복 실행한다고 해서 매번 새로운 5 m 배치가 생성되는 것은 아닙니다.

### 1.4 Student가 보는 정보: 517차원 관측

Student 신경망에는 이미지 전체나 시뮬레이션의 상자 절대 좌표를 넣지 않습니다. 로봇 내부 센서에 대응하는 정보와 카메라 기준 상대 위치를 묶은 517차원 벡터를 입력합니다.

| 구성 | 차원 | 내용 |
|---|---:|---|
| 최근 5 frame의 로봇 고유감각 | 460 | frame당 몸체 각속도 3, 중력 방향 3, 관절 위치 43, 관절 속도 43 |
| 직전 저수준 관절 행동 | 43 | 이전에 적용한 관절 행동. 23차원 작업 명령과 구분 |
| 현재 속도 명령 | 3 | 전후·좌우 이동 및 yaw 회전 명령 |
| 상자 관측 | 4 | 보이는지 1 + 카메라 기준 위치 3 |
| 상자함 관측 | 4 | 보이는지 1 + 카메라 기준 위치 3 |
| 테이블 위치 | 3 | 카메라 기준 위치 |
| 합계 | **517** | 모든 작업에 공통으로 사용하는 기본 입력 계약 |

이력 frame 하나는 92차원이며, 5개 frame을 결합하면 460차원입니다. 관절 위치는 기준 자세와의 차이를 사용하고, 일부 속도에는 스케일과 센서 잡음 설정이 적용됩니다.

Student 학습 loader는 HDF5의 `student_observation`만 입력으로 읽습니다. 같은 파일에 저장된 `box_pose`, `bin_pose`, 실제 접촉 수, lift, success 등의 정보는 Teacher 계산·검증·분석을 위한 것이며 Student 입력에 이어 붙이지 않습니다.

다만 이 정보 경계는 **Student 신경망 입력에 대한 경계**입니다. 현재 물리 성공 판정과 일부 FSM 안전 조건은 정확한 시뮬레이션 접촉·상자 위치를 사용합니다. 전체 제어 시스템이 이미 모든 특권 상태 없이 실물에서 동작한다는 뜻은 아닙니다.

### 1.5 QR 관측 구현과 실제 카메라의 차이

현재 QR 관측은 시뮬레이션의 물체 위치를 카메라 좌표계로 변환하고 시야각·거리·상자 윗면 방향 등을 검사하는 기하 기반 관측입니다. RealSense D435의 위치·방향과 약 69.4° 수평 시야각을 반영하지만, 실제 영상에서 QR을 검출하는 알고리즘을 실행하는 구조는 아닙니다.

- 보이지 않는 상자·상자함에 대해서는 관측 마스킹을 사용합니다.
- DAgger에서는 관측만 일시적으로 가려 QR 손실 상황을 생성합니다. 실제 상자를 이동시키는 처리가 아닙니다.
- 현재 입력에는 상자·상자함의 상대 위치가 들어가며, 검출된 6차원 자세 전체가 들어가는 것은 아닙니다.
- `table_position_camera`는 환경에 정의된 테이블 위치를 변환하는 항목입니다. 실제 적용에는 테이블 위치를 얻는 별도 추정 또는 보정 방법이 필요합니다.

실제 카메라를 사용하려면 카메라 보정, QR/AprilTag 검출, 자세 추정, 시간 필터 및 가림 처리와 현재 관측 계약을 연결하는 작업이 남아 있습니다.

### 1.6 환경 관련 주요 코드

| 파일 | 역할 |
|---|---|
| [box_carry_with_hand_env_cfg.py](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/box_carry_with_hand_env_cfg.py) | 로봇·물체·카메라·관측·보상·종료 조건 및 랜덤화 구성 |
| [unitree.py](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) | G1 43관절 자산, 초기 관절 자세와 손가락 actuator 설정 |
| [events.py](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events.py) | 배치 reset, 테이블 높이 및 물체 상태 갱신 |
| [workspace.py](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/workspace.py) | 작업 공간 배치와 거리 조건 관련 처리 |
| [camera_navigation_command.py](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/camera_navigation_command.py) | QR 상대 관측 기반 목표 접근·탐색·운반 속도 명령 |
| [student_observation.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/student_observation.py) | 517차원 입력 순서, 특권 입력 제외 및 QR dropout 처리 |

## 2. Teacher 데이터 수집

### 2.1 Teacher의 정확한 의미

Teacher는 원본 시연과 정확한 물리 상태를 이용해 정답 행동을 계산하는 제어기입니다. 현재 프로젝트에서 Teacher 네 개를 각각 새로 강화학습한 것이 아닙니다.

접근·파지에는 Arena의 손목 궤적과 작업 공간 변환을 사용하고, 운반·투입에는 실제 이전 단계 종료 상태에서 구성한 손목 목표와 이동·놓기 절차를 사용합니다. 이를 기존 Pink/Homie 제어기가 실행 가능한 관절 목표로 변환합니다.

Student는 그 결과를 학습하는 별도 신경망입니다. 따라서 Teacher와 Student를 비교할 때에는 동일한 초기 상태에서 각각 실제로 물리 실행한 성공률을 봐야 합니다. Teacher 행동을 Student 상태에 붙였다는 사실만으로 Teacher가 그 상태에서 끝까지 성공했다고 판단하지 않습니다.

### 2.2 Arena 원본 데이터와 관절 구성

현재 사용하는 원본 파일은 다음입니다.

`/home/mjkim/IsaacLab-Arena/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5`

기준 궤적에서 상자, 몸체 및 양 손목의 pose, 23차원 작업 명령과 43차원 관절 target을 읽습니다. 현재 사용 중인 원본 데이터 계약은 이미 **몸체 29관절 + 양손 14관절의 43관절 구성**입니다. 이번 과정은 손이 없는 29차원 배열에 새 관절을 임의로 덧붙이는 변환이 아닙니다.

리타게팅은 주로 **작업 공간과 현재 물리 상태에 맞춘 목표 pose의 변환**입니다. 현재 corpus의 `processed_joint_names`와 Arena Lab의 43관절 순서는 비교 결과 동일합니다. 따라서 현재 사례에서는 순서 정렬이 그대로 유지되는 경우이지만, 구현은 항상 joint 이름을 기준으로 확인·재배열하여 다른 순서의 target을 잘못 적용하지 않도록 합니다.

상자·몸체 pose의 quaternion은 `wxyz`, 23차원 손목 명령의 quaternion은 `xyzw`를 사용하므로 변환 경계에서 구분합니다. 원본의 행동·관측은 물리 step 이전, 물체 상태는 step 이후에 저장된 항목이 있어 시간 정렬도 별도로 처리합니다.

### 2.3 현재 환경으로 옮기는 과정

1. 원본 상자 초기 pose와 새로 생성한 상자 초기 pose를 읽습니다.
2. 두 상자의 x/y 위치 차이인 이동량을 계산합니다.
3. 원본 상자 및 좌우 손목 목표 궤적에 동일한 평면 이동량을 적용합니다.
4. 매 step의 실제 pelvis pose를 이용해 손목 목표를 현재 몸체 기준으로 변환합니다.
5. 변환된 목표를 Pink/Homie에 전달하여 43관절 target을 계산합니다.
6. 현재 Unitree RL Lab의 물리 환경에 target을 적용합니다.
7. 실제 접촉·상승·유지·자세 조건을 만족한 시연만 저장합니다.

핵심 변환은 다음과 같습니다.

```text
상자 이동량 = 새 상자 초기 위치 − 원본 상자 초기 위치  (x/y만 사용)
새 손목 world 목표 = 원본 손목 world 목표 + 상자 이동량
몸체 기준 손목 목표 = inverse(현재 pelvis pose) × 새 손목 world 목표
```

이는 원본의 모든 관절각을 그대로 재생하는 방식이 아닙니다. 손목과 상자의 상대 관계를 유지하는 목표를 만든 뒤, 현재 자세에 맞는 관절 해를 다시 계산하고 현재 물리 환경에서 검증합니다.

기준 파일의 손 명령이 열린 상태로 남아 있는 특성을 보완하기 위해, 필요한 구간에 양손 닫힘 명령을 주입하는 처리도 포함됩니다. 들어 올리기 후에는 검증된 목표 유지와 추가 상승 여유를 사용합니다.

### 2.4 Pink/Homie와 별도 worker의 역할

Whole-Body Control은 손목·몸통·이동 목표를 현재 로봇 상태에서 실행할 수 있는 전신 관절 목표로 바꾸는 중간 제어 계층입니다.

| 구성 요소 | 역할 |
|---|---|
| Pink | 목표 손목 위치·방향을 만족하는 팔·손목 자세를 찾는 역기구학 solver |
| HomieV2 | 현재 로봇 상태와 속도·몸통 목표를 이용해 균형·보행 관절 목표를 만드는 기존 학습 기반 제어기 |
| Arena worker | Arena 전용 Python 환경에서 Pink/Homie 계산을 실행하는 별도 프로세스 |
| 현재 환경의 actuator | 반환된 관절 위치 target을 물리 시뮬레이션에서 추종 |

Pink는 상자를 직접 인식하지 않습니다. Teacher 또는 Student·보정 제어기가 손목 목표를 만들면 Pink가 그 목표를 관절각으로 풉니다. HomieV2 역시 새로 학습한 Carry Student와 같지 않습니다. Carry Student는 작업 목표를 출력하는 상위 정책이고, HomieV2는 이를 실행하는 하위 균형·보행 제어기입니다.

두 프로젝트의 Isaac 환경 의존성을 분리하기 위해 JSON 요청·응답 방식을 사용합니다.

```text
현재 Unitree RL Lab
  현재 robot/pelvis 상태 + 손목·이동 목표를 JSON으로 전송
    → Arena 전용 Python worker의 Pink/Homie 계산
    → joint 이름이 붙은 43관절 target 반환
  현재 환경에서 물리 step 수행, 성공 판정 및 데이터 저장
```

**물리 rollout과 최종 성공 판정은 현재 환경에서 수행합니다.** Arena worker에서 다른 장면을 성공시킨 결과를 그대로 가져오는 구조가 아닙니다. 복원 상태에서는 현재 팔 자세로 역기구학을 초기화하여 다른 관절 해로 갑자기 바뀌는 문제도 줄입니다.

### 2.5 23차원 명령의 의미

23차원 명령은 Arena 제어기의 기존 인터페이스를 재사용한 표현입니다. 이 프로젝트에서 임의로 차원 수를 정한 새 인터페이스가 아닙니다. 다만 파지 닫힘, 작업별 속도, 정렬 보정 등 명령을 생성·수정하는 로직은 현재 작업에 맞게 구성했습니다.

| index | 차원 | 의미 |
|---|---:|---|
| 0–1 | 2 | 좌우 손 열림·닫힘 상태 |
| 2–4 | 3 | pelvis 기준 왼쪽 손목 위치 |
| 5–8 | 4 | 왼쪽 손목 방향 quaternion, xyzw |
| 9–11 | 3 | pelvis 기준 오른쪽 손목 위치 |
| 12–15 | 4 | 오른쪽 손목 방향 quaternion, xyzw |
| 16–18 | 3 | 전후·좌우 이동 속도 및 yaw 회전 속도 |
| 19 | 1 | pelvis 목표 높이 |
| 20–22 | 3 | 몸통 roll/pitch/yaw 목표 |

### 2.6 작업별 시연의 시작 상태와 성공 조건

**접근·파지**는 같은 전반부 성공 rollout에서 각각의 학습 구간을 추출합니다. 양손이 각각 상자와 접촉하고, 상자 중심이 초기 중심 높이보다 0.12 m 이상 상승한 상태를 100 step 연속 유지해야 합니다. 50 Hz 기준 100 step은 2초입니다.

여기서 12 cm는 테이블 윗면에서 상자 중심까지의 절대 높이가 아니라, **테이블에 놓인 상자의 초기 중심 높이에 대한 상승량**입니다. 중간에 한쪽 접촉이나 상승 조건이 깨지면 연속 유지 counter가 다시 시작됩니다.

**운반**은 실제 파지 성공 상태에서 시작합니다. 상자 기준 양 손목의 실제 상대 자세를 유지하고, 목적지 방향으로 이동한 뒤 상자함 앞에서 파지·정지 조건을 확인합니다. 상자 slip을 따라 손을 계속 아래로 내리는 현상을 막는 목표 높이 제한도 포함됩니다.

**투입**은 실제 운반 종료 상태에서 시작합니다. 상자를 들어 올려 상자함 위로 이동하고 내려놓은 뒤 손을 여는 절차와 release·settle 조건을 검사합니다. 이전 Place v2/v3 artifact와 현재 v4 계약은 구분하며, 호환되지 않는 오래된 데이터를 무조건 섞어 사용하지 않습니다.

시연 길이는 원본 Arena의 전체 궤적 길이와 같지 않습니다. 전반부는 파지·유지가 끝난 지점까지 사용하고, 운반·투입은 해당 단계의 목표 달성 또는 제한 시간에 따라 종료하는 별도 물리 rollout입니다.

### 2.7 확보한 Teacher 데이터와 다양성

| 작업 | 확보한 성공 시연 | 근거 | 해석 |
|---|---:|---|---|
| 접근 | 500개 rollout의 접근 구간 | [전반부 manifest](data/teacher/box_xy_randomized_500/manifest.json) | 파지와 같은 500개 rollout 공유 |
| 파지·리프트 | 같은 500개 rollout의 파지·유지 구간 | 동일 manifest | 독립적인 추가 500개 파일이 아님 |
| 운반 | 500개 | [운반 manifest](data/teacher/production/carry_teacher_success_500_r04/manifest.json) | 533회 시도에서 확보 |
| 투입 | 600개 | [투입 manifest](data/teacher/production/place_teacher_success_diverse_600_r07/manifest.json) | 기존 500개 + 다양화 100개, 총 795회 시도 |

전반부 corpus는 100개씩 5개 shard로 구성되어 있습니다. 운반 500개는 기존 100개에 추가 400개를 결합했고, 투입은 500개에서 600개로 확대했습니다.

다만 다음 한계가 있습니다.

- 전반부 500개는 한 기준 궤적을 좁은 x/y 범위로 변환한 데이터입니다.
- 운반 수집의 주요 시작 상태 집합은 Student 파지 성공 상태 22개입니다. 500개의 전혀 다른 파지 종료 상태를 한 번씩 사용한 것이 아닙니다.
- 투입 기존 500개는 Student 운반 종료 상태 두 개를 반복 사용한 결과입니다.
- 추가 투입 100개와 투입 DAgger 일부는 Teacher 운반 종료 상태 집합에서 생성했습니다. 모두가 최신 Student 운반 종료 상태는 아닙니다.
- 다른 Student 운반 종료 상태에서 성공한 추가 투입 Teacher 시연은 [별도 기록](data/teacher/production/place_teacher_latest_student_boundary.smoke5.metrics.json)상 1개이며, 최근 두 재학습 명령에는 포함되지 않았습니다.

`--repeat_snapshots`는 저장된 상태를 순서대로 반복 선택하는 기능입니다. 성공한 상태에 가중치를 줘 자동으로 선택하는 옵션은 아니지만, 실패를 버리고 성공만 저장하기 때문에 최종 데이터에서는 성공하기 쉬운 시작 상태가 더 많이 남을 수 있습니다. **성공 시연 수, 서로 다른 시작 상태 수, 시작 상태별 성공률을 함께 확인해야 합니다.**

### 2.8 물리 상태 저장과 안전한 데이터 게시

이전 단계의 종료 상태 파일에는 로봇 root pose·속도, 43관절 위치·속도, 상자 pose·속도, 직전 행동, Student 관측, 접촉·lift·유지 상태와 테이블·상자함 배치를 저장합니다. 운반 종료 상태에는 WBC의 전처리 전 기준 행동도 저장합니다.

현재 상태 종류는 `bilateral_hold`와 `carry_at_staging`입니다. 전자는 파지 성공 후 운반 시작용, 후자는 운반 성공 후 투입 시작용입니다. 실제 접근 종료 상태를 Teacher 파지 성공 rollout의 시작점으로 저장·복원하는 전용 경로는 아직 없습니다.

HDF5는 수집 중 임시 파일에 기록한 뒤 완료된 결과를 게시합니다. manifest와 sidecar metrics에는 데이터·checkpoint·gate의 SHA-256을 기록합니다. 파일이 존재하는 것만으로 완료로 간주하지 않고, 성공 수, 게시 여부, 계약 및 hash를 확인합니다.

### 2.9 최신 대량 수집 실행 파일과 남은 수집 기능

[collect_student_boundary_data.sh](scripts/teacher/collect_student_boundary_data.sh)는 다음 순서의 실행 파일입니다.

1. 5 m 랜덤 배치에서 Student를 10회 평가합니다.
2. 파지 종료 상태 3개와 운반 종료 상태 2개 이상이 확보됐는지 검사합니다.
3. 기준을 통과하고 `--collect`를 지정한 경우에만 별도 seed로 최대 2,000회 Student 종료 상태 수집을 진행합니다.
4. 이 상태를 사용해 투입·운반 Teacher 성공 시연을 각각 500개 목표로 수집합니다. 각 작업의 최대 시도 수는 2,500회입니다.
5. 파지 DAgger 교정 데이터를 500회 실행에서 수집합니다.

기본 실행은 사전 평가만 수행합니다. `--smoke`는 기준 통과 후 소규모 수집을 수행하며, `--dry-run`은 명령만 출력합니다. screening 상태는 `--collect`의 후속 학습용 상태와 seed·파일을 분리합니다.

```bash
# 명령 확인만 수행
bash scripts/teacher/collect_student_boundary_data.sh --dry-run

# 손목 보정 시험을 포함한 10회 사전 평가
bash scripts/teacher/collect_student_boundary_data.sh --alignment-probe

# 사전 평가 통과 시에만 대량 수집
bash scripts/teacher/collect_student_boundary_data.sh --alignment-probe --collect
```

`--alignment-probe`는 손목 보정 gain 0.7→1.0, 보정 한도 0.08→0.12 m, endpoint offset 5→10 step, 중심 보정 한도 0.03→0.05 m를 시험하는 설정입니다. 성공 조건은 완화하지 않으며, 아직 검증된 성능 개선이라고 볼 수 없습니다. 소규모 기준도 production 인증이 아니라, 실패만 반복하는 대량 실행을 막는 최소 준비 조건입니다.

기존 대량 실행에서는 완료 로그 182개가 모두 접근 timeout이었고, 파지·운반 성공 상태가 없었습니다. 이를 반영해 실행 파일에 사전 평가와 차단 조건을 추가했습니다. 문서 작성 시점에는 개선 설정의 소규모 평가가 시작된 상태이며 최종 결과는 미확정입니다.

**파지의 남은 기능:** `Student 접근 → 실제 접근 종료 상태 유지 → Teacher가 이어서 파지·리프트 수행 → 성공한 연속 궤적 저장` 경로는 아직 구현되지 않았습니다. 현재 파지 수집은 Student 상태에 Teacher 정답을 붙이는 DAgger 방식입니다. 최신 셸 실행 파일이 이 미구현 기능까지 자동으로 수행한다고 해석하면 안 됩니다.

### 2.10 Teacher 관련 주요 코드

| 파일 | 역할 |
|---|---|
| [extract_arena_teacher.py](scripts/teacher/extract_arena_teacher.py) / [arena_dataset.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/arena_dataset.py) | Arena 궤적 추출, 관절·좌표·시간 정렬 계약 |
| [mimic_xy_retarget.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/mimic_xy_retarget.py) | 상자·양손 궤적의 x/y 이동 변환 및 유지 조건 검사 |
| [collect_nominal_box_teacher.py](scripts/teacher/collect_nominal_box_teacher.py) | 현재 환경에서 전반부 Teacher 물리 시연 생성 |
| [generate_randomized_box_teacher.py](scripts/teacher/generate_randomized_box_teacher.py) | 전반부 shard 수집·검증·재개 및 manifest 관리 |
| [arena_wbc_bridge.py](source/unitree_rl_lab/unitree_rl_lab/utils/arena_wbc_bridge.py) / [arena_wbc_worker.py](scripts/teacher/arena_wbc_worker.py) | 두 Python 환경 사이 JSON 요청·응답 및 WBC 계산 |
| [phase_teachers.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/phase_teachers.py) | 실제 파지 상태 기반 Carry·Place 목표 생성 |
| [collect_carry_place_teacher.py](scripts/teacher/collect_carry_place_teacher.py) | 상태 복원 후 Teacher 시연 수집 또는 Student 평가·재라벨링 |
| [physical_snapshots.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/physical_snapshots.py) / [physical_snapshot_runtime.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/physical_snapshot_runtime.py) | 상태 파일 계약, 실제 환경 캡처·복원 |
| [build_phase_teacher_manifest.py](scripts/teacher/build_phase_teacher_manifest.py) | 작업별 성공 데이터 shard 통합 |

## 3. Student 정책 학습

### 3.1 네 정책은 각각 별도로 학습

Student 학습 명령 한 번은 `--skill`로 지정한 작업 하나를 학습합니다. 한 번의 학습으로 네 정책을 동시에 생성하는 방식은 아닙니다. 접근·파지가 같은 시연 파일을 읽어도 작업별 구간과 정답 행동이 다르므로 별도 checkpoint를 만듭니다.

| 정책 | 기본 입력 | 현재 주력 출력 | 출력 이후 처리 |
|---|---|---|---|
| 접근·양손 정렬 | 517차원 QR·고유감각 | 23차원 작업 명령 | Pink/Homie가 43관절 목표로 변환 |
| 파지·리프트 | 같은 517차원 | 43차원 절대 관절 위치 target | 환경 action scale·offset에 맞춰 적용 |
| 운반 | 같은 517차원 | 23차원 작업 명령 | WBC + 파지 보호 및 상·하체 분리 처리 |
| 투입 | 같은 517차원, 정책 metadata에 따른 adapter | 23차원 작업 명령 | WBC + 팔 offset·손가락 유지·release 처리 |

과거 Carry·Place에는 43차원 직접 출력 후보도 있었습니다. 현재 주력 23차원 정책과 혼동하지 않도록 checkpoint의 `skill_phase`, `action_key`, `action_dim`을 확인합니다. 강화학습 residual 실험용 공통 43차원 명목 정책도 별도 경로입니다.

### 3.2 사용하는 모방학습 방법

현재 주력 방식은 **Behavior Cloning과 DAgger를 결합한 지도학습**입니다. AMP와 같은 판별기를 이용하는 적대적 모방학습을 현재 네 작업 정책의 학습 방식으로 사용하지 않았습니다.

Behavior Cloning은 저장된 관측과 Teacher 정답 행동의 쌍을 읽고, Student가 같은 행동을 출력하도록 회귀하는 방식입니다.

```text
입력: 해당 시점의 517차원 Student 관측
정답: 같은 시점의 Teacher 23차원 명령 또는 43차원 관절 target
학습: Student 출력과 정답 사이의 정규화된 평균제곱오차 최소화
```

Teacher가 가진 상자 world pose 등의 정보는 정답 행동을 만드는 과정에서 사용하지만, Student가 그 정보를 입력받는 것은 아닙니다. Behavior Cloning 자체에는 critic도 필요하지 않습니다. 환경에 준비된 critic 관측은 강화학습 경로를 위한 구성입니다.

### 3.3 신경망과 공통 학습 설정

| 항목 | 기본 구성 |
|---|---|
| 모델 | 다층 퍼셉트론: 입력 517 → 은닉 512 → 512 → 256 → 행동 출력 |
| 활성화 | ELU |
| optimizer | AdamW |
| 손실 | 정규화된 행동 평균제곱오차. 데이터 출처·구간별 가중치 적용 가능 |
| 미니배치 | 기본 1,024개 frame |
| 기본 학습률 | 3×10⁻⁴. 기존 정책 미세조정은 더 작은 값 사용 |
| weight decay | 10⁻⁵ |
| 데이터 분리 | 에피소드 단위 80% / 10% / 10% |
| checkpoint 선택 | 기본적으로 검증 손실이 가장 낮은 epoch |
| 조기 종료 | 검증 개선이 없으면 patience 설정에 따라 종료 |

입력과 행동의 정규화 통계는 checkpoint에 포함합니다. 23차원 행동에는 위치·방향·속도·손 상태가 섞여 있으므로 모든 native 오차를 하나의 거리나 각도 단위로 해석하면 안 됩니다.

80/10/10 분리는 같은 에피소드의 frame이 서로 다른 subset에 들어가는 문제를 막지만, 다른 에피소드가 같은 복원 시작 상태를 반복 사용한 경우까지 자동으로 막지는 않습니다. 일반화 평가에는 원본 시작 상태·파일·seed를 기준으로 학습과 평가를 추가 분리해야 합니다.

epoch는 저장된 학습 데이터를 한 번 도는 단위이고, optimizer update는 그 안에서 미니배치마다 수행됩니다. 저장 데이터 회귀 및 작은 학습률의 미세조정은 시뮬레이션 강화학습보다 빨리 끝날 수 있습니다. 빨리 끝났다는 사실이나 작은 검증 손실만으로 충분한 물리 성능을 판단하지 않습니다.

### 3.4 학습한 주요 정책과 설정

아래 값은 각 checkpoint의 metadata 또는 학습 metrics에 기록된 값입니다. epoch는 최대 설정이며, best epoch는 0부터 기록된 index를 사람이 읽는 순번으로 바꾼 값입니다. best epoch가 곧 물리 성능의 수렴 시점이라는 뜻은 아닙니다.

| 정책 | 학습 데이터 및 설정 | 대표 checkpoint |
|---|---|---|
| 5 m 접근 후보 | 전반부 500개 + 5 m 접근 DAgger smoke 1회 데이터, 학습률 3×10⁻⁶, 최대 60 epoch | `box_approach_pregrasp_23d_workspace5m_r01.pt` |
| 현재 인증된 파지 | 전반부 500개 + 접근 후 파지 교정 데이터, 기존 정책에서 미세조정 | `box_grasp_lift_43d_from_approach_r05_dagger_r02_probe.pt` |
| 운반 500개 후보 | 400/50/50 에피소드 분리, 학습률 3×10⁻⁴, 최대 100 epoch, 초반 350 step 가중치 4배 | `box_carry_23d_teacher500_r05.pt` |
| 투입 600개 후보 | 480/60/60 분리, 다양화 시연 5배, 초반 120 step 8배, 학습률 3×10⁻⁴, 최대 150 epoch | `box_place_23d_diverse600_weighted_history_r12.pt` |
| 최근 파지 재학습 | 기존 500개 + DAgger round 10~13, 학습률 10⁻⁶, 교정 가중치 2배, 최대 30 epoch, best 30번째 | `box_grasp_lift_retrain_20261005.pt` |
| 최근 투입 재학습 | 기존 600개 + Student 상태 재라벨링 108개, 교정 출처 2배, 학습률 10⁻⁵, 최대 60 epoch, best 57번째 | `box_place_retrain_20261005.pt` |

checkpoint는 [data/teacher/bc](data/teacher/bc)에 위치합니다. 최근 두 재학습은 기존 실험과 같은 데이터를 다시 사용한 실험이며, 새로운 날짜의 파일 이름이 새 Teacher 성공 데이터를 추가했다는 뜻은 아닙니다.

5 m 접근 후보에도 데이터 한계가 있습니다. 실제 metadata상 넓은 배치 교정 데이터가 대규모 성공 corpus가 아니라 소규모 smoke 데이터이므로, 해당 checkpoint 이름만으로 5 m 접근·정렬 일반화가 완료됐다고 판단하지 않습니다.

### 3.5 투입 진행 정보와 관측 adapter

투입은 들어 올림, 이동, 내려놓음, release처럼 순서가 있는 동작입니다. 현재 일부 투입 정책은 작업이 얼마나 진행됐는지를 나타내는 phase clock을 사용합니다.

- 값은 `clip(place_step / 95, 0, 1)`입니다.
- 기존 입력의 `velocity_commands.x`, 즉 index 503에 기록하므로 전체 입력은 517차원입니다.
- 투입 중 속도를 0으로 유지하는 대신 사용 가능한 채널에 공개된 작업 진행 정보를 넣는 adapter입니다.
- 정확한 상자 world pose를 추가하는 특권 입력이 아닙니다.
- 최근 투입 재학습본은 관측 이력과 직전 행동을 유지하는 설정입니다.

학습과 실행에서 같은 adapter를 적용해야 합니다. 또한 목표 행동에 포함된 팔 offset·손가락 유지 처리와 실행 시의 Place v4 계약이 일치해야 합니다.

### 3.6 DAgger로 데이터가 늘어나는 원리

원본 Teacher 시연만 학습한 Student는 Teacher가 거의 방문하지 않은 자세에서 행동 오차가 커질 수 있습니다. 예를 들어 접근 종료 자세가 조금 다르거나 한쪽 손만 닿으면 이후 관측도 계속 달라져 실패가 누적됩니다.

DAgger는 다음 순서로 이 상태 분포 차이를 보완합니다.

1. Student를 실제 물리 환경에서 실행합니다.
2. Student 때문에 생긴 경계·실패 상태를 관측합니다.
3. 같은 상태에서 Teacher가 낼 정답 행동을 계산합니다.
4. Student 관측과 Teacher 행동을 교정 데이터로 저장합니다.
5. 기존 성공 시연에 교정 데이터를 섞어 다시 Behavior Cloning합니다.
6. 재학습 정책의 실제 성공률과 정상 시연 성능 저하 여부를 확인합니다.

파지 교정에는 한쪽 손 먼저 접촉, 손목 중심 이탈, 상자 회전, 들어 올리는 중 slip, 일시적 QR 손실과 정상 관측 표본을 포함하는 기능을 구현했습니다. 외란은 상자에 짧은 힘·토크를 가하는 방식이며, QR dropout은 관측만 가리는 방식입니다.

투입 재라벨링은 Student가 실행한 상태에서 Teacher 손목 목표를 계산하여 기록합니다. 실패 episode도 유효한 교정 행동을 갖는 경우 학습에 사용할 수 있습니다.

**DAgger 데이터의 성공 flag는 해당 Student rollout의 결과이지, Teacher가 그 궤적을 직접 실행해 성공했다는 인증이 아닙니다.** 전용 Teacher 성공 시연과 DAgger label을 별도 집계합니다.

### 3.7 교정 데이터 현황과 재학습 해석

| 구분 | 현재 기록 | 최근 재학습 사용 여부 |
|---|---|---|
| 파지 DAgger round 10~13 | 각 20회 실행, 합계 4,157개 교정 표본 | 최근 파지 재학습에 사용 |
| 이후 파지 경계 교정 | 5회 실행, 416개 표본 | 최근 명령에는 미포함. Teacher 성공 시연은 아님 |
| 투입 Student 상태 재라벨링 | 108개 에피소드, 14,712개 frame | 최근 투입 재학습에 사용 |
| 다른 Student 운반 종료 상태의 투입 Teacher 성공 | 1개 | 최근 명령에는 미포함 |

투입 108개에는 Student 성공 69개, 실패 39개가 포함됩니다. 이 69개를 새로운 Teacher 성공 시연 69개로 바꾸어 표현하지 않습니다.

교정 데이터의 비중을 너무 크게 하면 기존 정상 시연 성능이 나빠질 수 있어, 가중치를 낮춘 파지 후보도 비교했습니다. 실제 접근 뒤의 소규모 파지 성공을 확인한 보수적 후보는 `box_grasp_lift_43d_transition_nominal_r16_w025.pt`이며, 교정 가중치 0.25를 사용했습니다. 소규모 성공과 정식 checkpoint 인증은 별개입니다.

### 3.8 최신 Teacher–Student 비교와 한계

최근 재학습본으로 수행한 [비교 기록](data/teacher/evaluation/retrain_compare_20261005_H1xYja/physical_comparison.json)은 다음과 같습니다.

| 평가 | Student | Teacher | 초기 상태와 해석 |
|---|---:|---:|---|
| 파지·리프트 | 5/5 | 5/5 | 좁은 상자 배치에서 Teacher 접근 후 실행. 실제 Student 접근 이후의 평가가 아님 |
| 투입 | 5/15 | 8/15 | 같은 15개 운반 종료 상태에서 비교. 해당 시작 상태는 Teacher 운반 결과 |

투입은 같은 조건에서 Teacher가 Student보다 20%p 높았지만, Teacher 자체도 모든 시작 상태에서 성공하지는 않았습니다. 파지 5/5도 실제 Student 접근 이후의 높은 성공률이나 5 m 일반화를 보장하지 않습니다.

이 비교에서 새로 저장한 Teacher 투입 성공 시연 8개는 평가 상태에서 생성된 데이터입니다. 이를 학습에 넣은 뒤 같은 상태의 결과를 독립 평가로 제시하면 안 됩니다. 평가용으로 유지하거나 다른 평가 시작 상태를 분리해야 합니다.

### 3.9 강화학습 미세조정: 구현·실험과 현재 채택 상태

지도학습 이후 부족한 구간을 개선하기 위한 강화학습 경로도 구성했습니다. 기존 네 43차원 명목 정책을 동결하고 작은 PPO residual을 더하는 실험 경로입니다.

```text
관절 행동 = 동결된 Behavior Cloning 명목 행동 + 0.20 × 제한된 PPO residual
```

과거에는 RND를 포함한 128개 환경, 100회 update의 실험을 실행했고 307,200 transition과 checkpoint를 저장했습니다. 그러나 [실험 기록](data/teacher/bc/residual_ppo_rnd_100iter_r01.metrics.json)상 성공률 0%, 비정상 자세 종료율 100%로 production gate를 통과하지 못했습니다. 해당 강화학습 정책을 현재 성공 정책으로 승격하지 않았습니다.

따라서 현재 주력 Student 구성은 Behavior Cloning + DAgger입니다. 기존 강화학습 환경과 보행 설정이 있다는 사실을 적재 보행 PPO 학습 완료로 해석하지 않습니다. 앞으로 필요할 경우 파지 유지가 검증된 상태에서 하체 보행 구간 중심의 제한적인 PPO를 검토하며, RND를 필수 구성으로 두지 않습니다.

### 3.10 학습 관련 주요 코드

| 파일 | 역할 |
|---|---|
| [scripts/teacher/phase_bc.py](scripts/teacher/phase_bc.py) | 작업 선택, corpus·DAgger 입력, 가중 학습, 평가 및 인증 명령 |
| [utils/teacher/phase_bc.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/phase_bc.py) | 데이터 loader, phase 분리, MLP, 정규화, 학습 및 checkpoint 계약 |
| [collect_dagger_corrections.py](scripts/teacher/collect_dagger_corrections.py) | 접근·파지 Student 상태의 Teacher 행동 수집 |
| [build_place_relabel_manifest.py](scripts/teacher/build_place_relabel_manifest.py) | 투입 재라벨링 shard·교정 표본 검증 |
| [place_runtime_contract.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/place_runtime_contract.py) | 투입 팔 offset, 손가락 유지 및 release 처리의 의미 일치 |
| [bc_residual.py](source/unitree_rl_lab/unitree_rl_lab/utils/bc_residual.py) | 동결 명목 정책과 residual 행동의 합성 |
| [scripts/rsl_rl/train.py](scripts/rsl_rl/train.py) / [rsl_rl_ppo_cfg.py](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py) | 강화학습 실행 및 PPO 설정 |

## 4. FSM 구성

### 4.1 단순 정책 재생과 FSM의 차이

FSM은 Finite State Machine, 즉 유한 상태 기계입니다. 일정 시간이 됐다고 다음 정책으로 넘어가는 대신, 각 작업의 실제 조건을 확인하고 현재 사용할 정책을 선택합니다.

```text
접근 및 양손 정렬
  → 양손 파지 및 들어 올리기
  → 정지 상태의 파지 유지·균형 안정화
  → 상자를 든 보행
  → 상자함 투입 및 release
  → 완료
```

정지 안정화는 독립적으로 학습한 다섯 번째 Student 정책이 아니라, 파지→운반 사이에 추가한 제어 구간입니다. 작업 신경망은 네 개를 유지하면서, 파지 이후 바로 걷기 시작하지 않도록 제어합니다.

환경 내부의 coarse task phase는 접근·파지를 한 구간으로 묶는 등 별도 구성을 갖습니다. 네 작업별 `SkillPhase`와 환경의 phase 숫자가 항상 같은 의미라고 가정하지 않고 loader와 FSM에서 각각 변환합니다.

### 4.2 접근→파지 전환

로봇이 상자 근처에 도달하는 것만으로 파지를 시작하지 않습니다. 현재 기본 전환 검사에는 다음 조건이 들어갑니다.

| 검사 항목 | 기본 한도 |
|---|---:|
| 왼손·오른손의 가까운 손끝과 상자 중심 거리 | 각 0.18 m 이하 |
| 양손 손끝으로 만든 중심과 상자 중심의 차이 | 0.11 m 이하 |
| 양 손목 중심과 상자 중심의 차이 | 0.222 m 이하 |
| 좌우 손목–상자 거리의 비대칭 | 0.10 m 이하 |

각 최소값이 다른 시점에 만족한 것만으로 성공하지는 않습니다. 필요한 조건이 같은 시점에 만족하고 준비 상태가 유지되어야 전환합니다.

전환 때에는 실제 접근 마지막 행동에서 파지 행동으로 부드럽게 보간합니다. 현재 수집 설정은 100 step을 사용합니다. 파지 중 하체 균형 제어를 유지하고, 접촉한 손가락 목표와 충분히 들어 올린 이후의 상체 목표를 고정하는 옵션도 추가했습니다.

5 m 접근에서는 카메라 기반 이동 명령과 몸통 목표를 명시적으로 전달하고, 가까워졌을 때 Arena endpoint의 QR 기반 손목 정렬 및 보정을 사용하는 실행 구성이 있습니다. 따라서 현재 일부 실행은 **Student 신경망 단독이 아니라 Student + WBC + 작업별 보정 제어**입니다. 이 보정을 포함한 결과를 보정 없는 신경망의 순수 성공률로 제시하면 안 됩니다.

### 4.3 파지→정지 안정화→보행

파지 직후의 주요 문제는 앞 정책의 종료 자세와 다음 정책의 예상 초기 자세가 달라 관절 명령이 갑자기 바뀌는 것이었습니다. 이를 줄이기 위해 다음 구조를 적용했습니다.

1. 양손 접촉과 12 cm 상승을 100 step 연속 확인합니다.
2. 속도 명령을 0으로 유지하며 손·팔 목표를 파지 성공 시점 값으로 유지합니다.
3. 하체 행동을 100 step 동안 부드럽게 바꾸며, 추가로 양손 파지 유지 조건을 확인합니다.
4. 기본 250 step, 즉 5초의 정지 균형 구간을 통과합니다.
5. 보행 명령을 점진적으로 증가시키고 운반을 시작합니다.

첫 파지 성공을 확인하는 2초와 이후 전환 안정화의 2초는 서로 다른 조건입니다. 이 시간들을 모두 합친 정확한 전환 시간은 준비 조건과 설정에 따라 달라집니다.

정지 균형 조건을 한 번 통과한 뒤 작은 접촉 변화 때문에 보행 시작이 계속 초기화되지 않도록, 통과 여부를 유지하는 일방향 상태와 보행 ramp 시작 시점을 수정했습니다. 실제 낙하·위험 자세 판정은 계속 적용합니다.

### 4.4 운반의 상·하체 역할 분리

운반 정책이 손가락·팔·하체를 모두 크게 바꾸면 상자를 잡은 자세가 쉽게 무너질 수 있습니다. 다음과 같이 역할을 분리하는 처리를 추가했습니다.

- 손가락은 파지 성공 시점의 닫힌 target을 유지합니다.
- 팔·손목은 성공 target을 유지하거나 작은 범위의 residual만 허용합니다.
- 하체는 균형·보행 명령을 수행합니다.
- 보행 시작에는 정지→이동 ramp를 사용합니다.
- 실험 목적에 따라 하체 Student, WBC, 또는 WBC→Student blend를 선택하는 옵션이 있습니다.

현재 수집 스크립트의 `carry_lower_body_mode=student`는 Student 기반 운반 실행을 사용한다는 뜻입니다. 다만 23차원 정책을 관절 목표로 바꾸는 Pink/Homie와 파지 보호 postprocessor는 계속 사용합니다. 별도의 payload-aware 보행 PPO가 완성됐다는 뜻은 아닙니다.

### 4.5 운반→투입 및 투입 완료

상자함 앞의 위치·방향과 파지 상태를 확인한 뒤 투입으로 전환합니다. 투입에는 운반에서 받은 팔의 하중 유지 offset을 점진적으로 줄이는 처리와, release 전까지 손가락을 닫힌 값으로 유지하는 Place v4 처리가 있습니다.

양손을 너무 일찍 열거나 투입 시작 시 팔 목표가 급격히 바뀌면 상자가 상자함 바깥으로 떨어질 수 있습니다. 따라서 투입 정책 출력뿐 아니라 상자함 내부 도달, 양손 release 및 안정 상태 유지 조건을 검사합니다.

높은 곳에서 손 접촉이 끊어진 것을 바로 바닥 낙하로 판정하지 않도록, 상자함 내부 여부와 실제 상자의 높이를 함께 검사하는 낙하 조건도 구성했습니다.

### 4.6 성공률은 단계별 조건부 확률로 해석

| 지표 | 의미 |
|---|---|
| P(approach) | 전체 시도 중 접근·정렬 완료 비율 |
| P(grasp \| approach) | 접근에 성공한 시도 중 파지·리프트 성공 비율 |
| P(handoff survival \| grasp) | 파지 성공 후 정지 유지·전환 구간을 버틴 비율 |
| P(loaded stand \| handoff) | 전환 구간 통과 후 정지 균형 성공 비율 |
| P(walk \| loaded stand) | 정지 균형 통과 후 목적지 이동 성공 비율 |
| P(carry \| grasp) | 파지 성공에서 운반 종료까지의 성공 비율 |
| P(place \| carry) | 운반 성공 후 투입 성공 비율 |
| 전체 성공률 | 전체 시도 중 접근부터 투입까지 모두 성공한 비율 |

예를 들어 접근 20개, 파지 5개, 운반 2개, 투입 1개라면 파지는 5/20, 운반은 2/5, 투입은 1/2이며 전체 성공률은 1/20입니다. 앞 단계가 실패해 다음 단계에 도달하지 않은 시도를 다음 단계의 실패 데이터와 혼동하지 않습니다.

### 4.7 검증된 결과와 아직 일반화되지 않은 결과

| 평가 구성 | 접근 | 파지 | 운반 | 투입/전체 성공 | 해석 |
|---|---:|---:|---:|---:|---|
| 기존 주력 후보의 좁은 배치 전체 20회 | 20/20 | 5/20 | 2/5 | 1/20, 전체 5% | 넓은 배치 평가가 아님 |
| R16 후보의 좁은 배치 3회 | 3/3 | 3/3 | 3/3 | 2/3, 전체 66.7% | 소규모 진단 결과, production 인증 아님 |
| 손 강성 변경을 포함한 좁은 배치 5회 | 5/5 | 3/5 | 3/3 | 2/5, 전체 40% | 다른 설정이며 반복 변동성 존재 |
| 최근 5 m 대량 수집의 완료 로그 182개 | 0/182 | 미도달 | 미도달 | 전체 성공 없음 | 모두 접근 timeout, 다음 단계 시작 상태 미확보 |

근거: [전체 20회](data/teacher/evaluation/full_fsm_latest_students_place_r12.rollout20.metrics.json), [R16 3회](data/teacher/evaluation/full_fsm_r16_latchedstand_staging20cm.seed31415.screen3.metrics.json), [강성 변경 5회](data/teacher/evaluation/full_fsm_r16_stiffness8_latchedstand_staging20cm.seed31415.screen5.metrics.json), [5 m 수집 로그](data/teacher/production/student_boundary_pEKwmF/logs/01_student_end_states.log).

단일 전체 성공 영상과 소규모 성공을 확보했지만, 이를 모든 현재 checkpoint 조합이나 5 m 환경의 성공률로 확장하지 않습니다. 과거 고정 조건에서 파지·전환 성공률이 높았던 결과도 다른 정책·상태·설정의 결과와 구분합니다.

현재 5 m 실패의 의미는 단순히 파지 Student가 학습이 덜 됐다는 것이 아니라, **파지 정책을 실행할 수 있는 접근·양손 정렬 상태 자체를 확보하지 못했다는 것**입니다. 이 상태에서 투입 Student만 재학습하거나 같은 수집을 수천 회 반복해도 앞 단계 문제는 해결되지 않습니다.

### 4.8 checkpoint 인증과 실행 구조

파지 gate는 지정한 checkpoint의 SHA, 검증 오차, 실제 물리 rollout, 12 cm·100-step 유지 조건과 성공 상태 파일의 SHA를 연결합니다. 다른 checkpoint로 교체하면서 기존 gate를 그대로 사용하는 것은 허용하지 않습니다.

단독 파지 평가가 Teacher 접근 상태에서 성공한 것과, Student 접근 뒤의 전환 평가에 성공한 것은 별도입니다. `evaluate_bc2_rollout.py`의 기본 전반부는 Teacher 접근이므로 실제 전환 성공 판정에는 FSM 기반 평가를 추가해야 합니다.

전체 승격도 데이터나 checkpoint의 생성만으로 완료하지 않습니다. 실제 조건부 성공률과 전체 반복 결과, 시작 상태 분리, 파일 무결성을 확인합니다. 소규모 시험 기준을 넘었다고 즉시 production 정책으로 교체하지 않습니다.

[.vscode/launch.json](.vscode/launch.json)에는 데이터 수집, 정책별 학습, 파지 인증, 전환 및 전체 평가 실행 항목이 있습니다. 다만 최신 사전 평가 차단 로직은 셸 실행 파일에 구현되어 있어 launch 항목을 순서대로 실행하는 것과 같지 않습니다. 또한 문서 확인 시 launch 파일의 `configurations` 시작 부분에 비 JSON 문자열 `dndpsms g`가 있어, 현재 상태로는 문법 수정 없이 정상 파싱되지 않을 수 있습니다. 이 문서 작성에서는 해당 사용자 파일을 수정하지 않았습니다.

### 4.9 FSM 및 검증 관련 주요 코드

| 파일 | 역할 |
|---|---|
| [skill_fsm.py](source/unitree_rl_lab/unitree_rl_lab/utils/teacher/skill_fsm.py) | 작업 상태, 물리 전환 조건, 조건부 성공률 집계 |
| [play_bc_skill_fsm.py](scripts/teacher/play_bc_skill_fsm.py) | 네 checkpoint 실행, 정렬·blend·안정화·운반·release 및 종료 상태 기록 |
| [evaluate_bc2_rollout.py](scripts/teacher/evaluate_bc2_rollout.py) | Teacher 접근 이후 파지 Student 또는 Teacher의 물리 평가 |
| [gate_grasp_transition_fsm.py](scripts/teacher/gate_grasp_transition_fsm.py) | 실제 접근 이후의 파지 성공률 검사 |
| [gate_full_student_fsm.py](scripts/teacher/gate_full_student_fsm.py) | 전체 반복 성공 및 승격 조건 검사 |
| [aggregate_student_rollout_metrics.py](scripts/teacher/aggregate_student_rollout_metrics.py) / [compare_student_physical_gates.py](scripts/teacher/compare_student_physical_gates.py) | 평가 통합 및 후보 정책 비교 |
| [collect_student_boundary_data.sh](scripts/teacher/collect_student_boundary_data.sh) | 소규모 준비 평가, 대량 수집 차단·허용 및 실행 결과 검증 |
| [test_student_boundary_collection_script.py](tests/test_student_boundary_collection_script.py) | 명령 생성, 입력 옵션, 데이터 검증, 실패 시 후속 수집 차단 테스트 |

최신 실행 파일 수정은 구문 및 검증 테스트 21개를 통과했습니다. 이는 실행 흐름과 실패 차단 로직의 테스트이며, 정렬 시험 설정이 물리 성공률을 높였다는 결과는 아닙니다.

### 4.10 현재 완료 범위와 앞으로 연결할 작업

현재까지 구현·수행한 범위는 다음과 같습니다.

- G1 43관절 평지 상자 작업 환경과 랜덤화 기능 구성입니다.
- 현재 환경에서의 Teacher 리타게팅, 전반부 500개·운반 500개·투입 600개 성공 데이터 확보입니다.
- 네 Student의 Behavior Cloning 및 DAgger 재학습과 여러 checkpoint 생성입니다.
- 실제 전환 상태 사용, 파지 목표 유지, 상·하체 역할 분리, 정지 균형과 release를 포함한 FSM 구성입니다.
- 좁은 배치의 전체 성공 사례 및 실패 구간별 평가, 최신 5 m 수집 실패 확인과 준비 평가 차단 로직 추가입니다.

남은 우선순위는 다음과 같습니다.

1. 5 m 접근 말단 정렬을 실제 물리 시험으로 개선하고 충분한 파지 가능 상태를 확보하는 작업입니다.
2. Student 접근 뒤에서 Teacher가 파지·리프트를 성공하는 연속 시연 수집 경로를 추가하는 작업입니다.
3. 다양한 실제 Student 종료 상태에서 운반·투입 Teacher 성공 데이터를 보강하고, 시작 상태 기준의 학습·평가 분리를 적용하는 작업입니다.
4. 새 데이터로 필요한 Student만 재학습한 뒤 단독·실제 전환·전체 FSM 평가를 각각 통과시키는 작업입니다.
5. 시뮬레이션 연속 작업을 안정화한 뒤 적재 보행 PPO, 물성·험지 일반화, 실제 QR 인식 및 sim-to-real로 확장하는 작업입니다.

요약하면, **네 작업의 데이터 생성·Student 학습·FSM 실행 기반은 구축됐지만, 다양한 5 m 배치에서 Student 전체 작업을 안정적으로 수행하는 파이프라인은 아직 완성되지 않았습니다.** 현재 개발의 중심은 추가 학습을 무조건 반복하는 것이 아니라, 실제 정책 전환 상태를 확보하고 그 상태에서 성공하는 데이터를 추가하여 연결 성능을 검증하는 것입니다.
