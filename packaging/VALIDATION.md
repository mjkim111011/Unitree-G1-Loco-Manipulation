# 패키징 검증 결과 — 2026-10-05

## 실제 설치한 새 환경

기존 학습용 Conda 환경이나 기존 `/home/mjkim/IsaacLab-Arena/.venv`를 수정하지 않고, 현재 저장소에 별도 개발 환경을 설치했습니다.

| 항목 | 확인 결과 |
| --- | --- |
| 새 학습 환경 | `.venv`, CPython **3.11.14**, `pip list` 기준 **281개** 배포 패키지 |
| 새 Arena 환경 | `external/IsaacLab-Arena/.venv`, CPython **3.12.12**, **314개** 배포 패키지 |
| 학습 환경의 전체 버전 검사 | `verify_environment.py --full --assets` **통과** |
| 학습 환경 `python -m pip check` | **No broken requirements found** |
| CUDA 실제 연산 | PyTorch **2.7.0+cu128**에서 GPU tensor 연산 **통과** |
| Student 모듈 로딩 | **517차원** 관측 모듈 import **통과** |
| Isaac Sim 실행 | 앱 시작 → context 생성/reset → **10 step** 실행 → 정상 종료 **통과** |
| Arena 전체 원본 버전 비교 | `--profile arena --full` **통과**, 원본 314개 버전과 일치 |
| 두 Python 환경의 연결 | 새 Arena worker 초기화 및 **43개 named joint** 반환 **통과** |
| 테스트 | 새 학습 환경에서 패키징·수집 스크립트·리타게팅 테스트 **42개 통과** |
| 루트 wheel / source 배포 | `python -m build --no-isolation` **성공** |
| 기존 extension wheel | `pip wheel ... source/unitree_rl_lab` **성공** |
| wheel의 모듈 누락 검사 | `source/.../unitree_rl_lab`의 Python 파일 **누락 없음** |
| source 배포의 대용량 파일 검사 | data/외부 소스/venv/체크포인트/HDF5/MP4 **포함 없음** |
| portable VS Code 설정 | **35개** launch 구성 JSON 파싱·실행 스크립트 경로 검사 **통과** |

학습 환경은 원본 284개 중 이 저장소에서 사용하지 않는 `lab.cocelo`, `robot_lab`, `swanlab`을 제외하여 281개입니다. **starlette 0.45.3, wheel 0.45.1, wrapt 1.16.0만 원본과 다르며 나머지 포함 패키지 버전은 일치**합니다. torchaudio는 로컬 버전 접미사까지 원본과 같도록 `===2.7.0`으로 지정했습니다.

## 소스 재현

- Main Isaac Lab: `73930be06597075d6e587af1f0a55e0439931f83`에 원래 로컬 변경 8개 파일 패치와 starlette 의존성 패치 적용.
- Arena: `c5826c87aa181442328c2c74c49f9e551b5c3d8c` 및 원래의 submodule 커밋 사용.
- Arena Isaac Lab submodule: `af1bab4dc173ba69b08fab779c14ead61d13fd33`.
- Arena `uv.lock` SHA-256: `64d912c667eb64776e3c543697cdbcdf16cc97bf7b117d8f2f052128771ac9bf`.
- G1 robot description: unitree_ros `d96d8f63ae17a7108d4f7229c00ef875ba7129c9`.

설치 과정에서 GitHub SSH submodule 접근 실패를 실제 확인하여 **해당 설치 명령에서만 HTTPS로 재작성**하도록 수정했습니다. 사용자 전역 Git 설정은 변경하지 않았습니다. 원래 버전 전체를 그대로 설치했을 때의 의존성 충돌은 숨기지 않고 원본 snapshot과 설치용 lock으로 분리했습니다.

작은 시뮬레이션을 종료할 때 Isaac Lab의 standalone timeline-stop callback이 paused-render loop를 유지하는 상황도 확인했습니다. 검증 스크립트에서 공개 API인 `SimulationContext.clear_instance()`로 callback을 해제한 후 Kit를 종료하도록 수정했고, 수정된 스크립트의 정상 종료를 재확인했습니다. 연구용 rollout 제어 로직이나 성공 조건은 변경하지 않았습니다.

## 남아 있는 주의점

- Arena의 공식 lock override에 대응하는 metadata 충돌 **3개는 경고로 남음**: NumPy/openpi-client, coverage/Isaac Lab, websockets/Isaac Sim. 무조건적인 예외 허용이 아니라 현재 확인한 패키지·버전 조합만 허용합니다. Arena가 `pip check` 무경고 상태라는 의미는 아닙니다.
- GPU 검증은 현재 Ubuntu 22.04.5 / RTX 3090 Ti / driver 590.48.01 머신에서 수행했습니다. 다른 OS·GPU·드라이버의 동일 동작을 보장하지 않습니다.
- 메타데이터/시뮬레이터/WBC 기동 검증이며, 새 환경에서 네 Student 정책의 전체 FSM 성공률을 다시 측정한 것은 아닙니다.
- 기존 데이터·checkpoint·gate의 새로운 경로 이전은 수행하지 않았습니다. 별도 데이터 공개와 SHA/출처 검증을 마쳐야 새 clone에서 이미 학습된 정책을 실행할 수 있습니다.
- 원래 `.vscode/launch.json`의 사용자 내용은 보존했습니다. 배포 시 파싱 가능한 `packaging/vscode.launch.json`을 사용해야 합니다.
- GitHub push, PyPI 업로드 및 기존 데이터 삭제는 수행하지 않았습니다.

## 검증 재실행

```bash
source packaging/activate.sh
python -m pip check
unitree-lab doctor --full --assets
unitree-lab doctor --simulation
unitree-lab doctor --wbc
external/IsaacLab-Arena/.venv/bin/python packaging/verify_environment.py --profile arena --full
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_packaging_workspace.py \
  tests/test_student_boundary_collection_script.py \
  tests/test_mimic_xy_retarget.py
```

현재 설치 폴더는 학습 환경 약 **19 GB**, Arena 환경 약 **27 GB**입니다. 추가 Git 소스·pip/uv 다운로드 캐시 및 모델 캐시는 이 용량에 별도로 더해지며 `external/`, `.venv*`는 Git에서 제외됩니다.
