# 위치 다양화를 위한 Teacher 시연 수집

기존 성공 시연 500개는 상자 위치가 x/y 각각 약 ±2 mm에 집중되어 있었음.
아래 실행기는 기존 데이터를 보존하고, 두 종류의 **새로운 물리 시연**을 동시에 수집함.
Student 학습이나 실패 상태 교정 데이터 수집(DAgger)을 실행하는 스크립트는 아님.

## 두 수집 경로

| 구분 | 바뀌는 조건 | Teacher 동작 |
|---|---|---|
| 1. 테이블 위 상자 위치 | 고정 테이블 중심에서 상자 x/y 각각 ±8 cm | Arena 손목 궤적을 새 상자 위치로 이동 → 양손 정렬 → 파지·리프트 → 2초 유지 |
| 2. 테이블 배치 및 접근 | 테이블 x=0.8~1.5 m, y=−0.4~0.4 m, 테이블 위 상자 x/y 각각 ±8 cm | 실제 보행 → 파지 시작 위치에서 2초 정지 → 같은 로봇 상태에서 정렬·파지·리프트 → 2초 유지 |

- 0.4 m × 0.4 m 테이블과 0.2 m 상자 기준, ±8 cm는 상자 전체를 테이블 위에 두고 가장자리에 2 cm 여유를 남기는 범위임.
- 2번은 기존 5 m × 5 m 환경에서 **로봇 전방 배치를 먼저 다루는 학습 단계**임. 5 m 전체·360도 배치가 검증됐다는 뜻은 아님. 상자함은 전방 보행 경로와 겹치지 않도록 x=−0.5~0.5 m, y=−1.8~−1.3 m에 배치함.
- 두 경우 모두 상자 크기 0.2 m, 질량 0.2 kg, 테이블 높이 0.74 m, 마찰을 고정하여 먼저 위치 변화만 검증함.
- 기본 목표는 **각 경로 성공 시연 500개**임. 성공 데이터의 복제나 기존 500개의 재사용이 아님.

## 실행

작업 공간 루트에서 실행:

```bash
cd /home/mjkim/unitree_rl_lab
bash scripts/teacher/collect_diverse_teachers_parallel.sh --dry-run
bash scripts/teacher/collect_diverse_teachers_parallel.sh --smoke
bash scripts/teacher/collect_diverse_teachers_parallel.sh
```

- `--dry-run`: 두 실행 명령만 출력. 파일 생성·시뮬레이션 실행 없음.
- `--smoke`: 각 경로 2개 성공 시연을 목표로 소규모 검사. 전체 범위 성공을 보장하는 검사는 아님.
- 옵션 없음: 두 경로를 병렬 실행, 각각 500개 목표. GPU가 하나면 두 작업 모두 `cuda:0` 사용. 소규모 검사 결과를 먼저 확인한 뒤 본 수집 실행 권장.
- 각 경로는 독립 프로세스·로그·임시 폴더 사용. 한 경로가 실패하면 해당 경로를 중단하고 다른 경로는 계속 진행. Ctrl+C는 이 실행기가 시작한 작업만 중단.

장치 및 출력 폴더 지정 예:

```bash
LM_RUN_DIR=/home/mjkim/unitree_rl_lab/data/teacher/production/diverse_teacher_run01 \
LM_DEVICE_1=cuda:0 LM_DEVICE_2=cuda:0 \
bash scripts/teacher/collect_diverse_teachers_parallel.sh
```

같은 명령·같은 `LM_RUN_DIR`로 재개 가능. 기존 shard의 SHA와 수집 설정을 검사하며 덮어쓰지 않음.
미완료 shard, 실패 구역, 변경된 코드·설정이 있으면 완료 처리하지 않고 중단함.
Arena 위치가 다른 장비에서는 `LM_ARENA_ROOT`, `LM_ARENA_DATASET`, 필요하면 `LM_PY` 지정.

## 성공 판정과 분포

**현재 Unitree RL Lab 물리 시뮬레이션**에서 양손 접촉력 ≥1 N, 상자 초기 높이보다 ≥12 cm 상승,
두 조건 100 step 연속 유지(50 Hz에서 2초), 낙하 없음, 로봇 기울기 ≤0.8 rad를 만족한 시연만 저장함.
2번은 보행 후 도착·정지 조건도 만족해야 함. 시뮬레이터 종료 코드만으로 성공 처리하지 않음.

1번은 상자의 테이블 상대 위치, 2번은 테이블 위치를 각각 4×4 구역으로 나누고 성공 개수를 고르게 할당함.
500개일 때 각 구역에 31~32개를 확보하므로, 성공하기 쉬운 위치만 반복 수집하여 500개를 채우지 않음.
각 shard의 시도 예산은 목표의 4배이고, 시도 대비 성공률 80% 또는 구역 목표에 미달하면 중단함.
성공 궤적은 보존하되 manifest는 미완료로 표시하고 실패 위치·원인을 metrics/log에 남김.
2번의 테이블 상대 상자 위치는 구역 내부에서 무작위 샘플링하므로, 테이블·상자 조합 전체의 균등 분포를 보장하지는 않음.

## 저장 위치 및 다음 학습 연결

```text
data/teacher/production/teacher_diversity_<고유 문자열>/
├── 01_table_box/          # 성공 HDF5, 물리 상태, metrics, manifest
├── 02_table_approach/     # 보행부터 파지·유지까지 연결된 성공 HDF5
└── logs/
    ├── 01_table_box.log
    └── 02_table_approach.log
```

```bash
tail -f data/teacher/production/teacher_diversity_<문자열>/logs/01_table_box.log \
        data/teacher/production/teacher_diversity_<문자열>/logs/02_table_approach.log
```

완료 확인은 각 `manifest.json`의 `complete=true`, `accepted_demos=500`, 구역별 목표 달성 여부로 수행함.
완료된 디렉터리/manifest는 기존 Student 학습의 `--dataset`에 전달 가능하도록 loader를 연결함.
한 시연 안에서 손 닫기 전 구간은 접근 학습, 손 닫기 이후 구간은 파지·리프트 학습에 사용됨.
**이 실행기는 운반·투입 Teacher 데이터를 수집하지 않음.** 해당 단계 데이터는 새 파지 종료 상태에서 별도로 수집해야 함.

## 제어 및 관측

Arena worker의 Pink는 손목 목표를 팔 관절로 변환하고, Homie는 속도·몸통 목표를 보행 관절로 변환함.
계산 결과를 현재 환경의 43관절 순서로 적용하며 물리 rollout은 현재 환경에서 진행됨.
2번의 보행 속도는 기존 QR 카메라 관측과 로봇 자체 관측으로 계산함.
보행 중 몸통 pitch를 −0.25 rad로 설정하여 멀리 놓인 상자도 카메라 시야에 들어오도록 함.
도착 목표는 상자 앞 0.42 m이며, 도착 뒤 실제 자세에서 정지한 다음 시연 궤적을 시작함.
방향 목표 0 rad는 고정 작업 조건이며 상자의 숨겨진 절대 좌표를 Student 입력에 추가하지 않음.
Student 입력은 기존 517차원, 접근 Teacher 목표는 기존 23차원, 처리된 관절 목표는 기존 43차원을 유지함.
상자 절대 좌표는 Teacher 리타게팅·환경 설정·성공 판정·감사용 metadata에만 사용함.
현재 QR는 시뮬레이션 기하 관측이며 실제 카메라 영상 검출 알고리즘은 아님.

## 구현 시 확인한 소규모 결과 (2026-10-06)

- 병렬 실행: 1번 2회 시도에서 2개 성공, 2번 2회 시도에서 2개 성공.
- 별도 먼 배치 시험: 테이블 (1.5 m, 0.4 m)에서 보행→파지→리프트→100-step 유지 성공.
- 시험 데이터: `data/teacher/production/teacher_diversity_xsIF44/`.
- 먼 배치 시험: `data/teacher/production/teacher_navigation_far_probe_oqbA9I/`.
- 각 500개 본 수집 및 16구역 전체의 성공률은 **아직 검증되지 않았음**.
  앞선 진단 시험의 실패 로그는 보존했고 실패 궤적은 성공 데이터에 포함하지 않았음.
