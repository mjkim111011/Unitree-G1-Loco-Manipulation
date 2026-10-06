# Unitree G1 43-DOF Loco-Manipulation 진행 현황

## 목표

```text
Box 접근 → 양손 파지·리프트 → Box 운반 → Bin 투입·안착
```

Unitree G1의 몸체 29관절과 양손 Dex3 14관절을 합친 43관절을 사용해 보행과 양손 조작을 연결하는 프로젝트임.

## 데이터 준비

- Arena의 성공 pick-and-place episode 1개를 원본으로 사용함
- 원본은 29관절이 아니라 43관절 데이터이며 853 step, 17.06초임
- Arena와 현재 환경의 production 43-joint action order는 실제로 동일함
- Runtime에서는 asset 변경에 대비해 joint name 기준으로 항상 target을 정렬함
- Arena와 현재 Isaac Lab stack이 달라 Pink/Homie Whole-Body Control만 별도 Arena Python worker에서 실행함
- 물리 rollout, 성공 판정, HDF5 저장은 현재 `Unitree-G1-43dof-BoxPickPlace` 환경에서 수행함

## 500개 Teacher demonstration

- 현재 환경에서 Box 초기 x/y만 무작위화해 500개를 생성함
- 원본 Box와 새 Box의 translation delta를 Box·좌우 손목 궤적에 동일하게 적용함
- demonstration 평균 길이는 260.788 step, 약 5.216초임
- 다음 조건을 2초 동안 연속 만족한 성공 episode만 저장함
  - 왼손과 오른손이 각각 Box에 접촉
  - Box 중심이 episode 초기 높이보다 12 cm 이상 상승
  - 위 두 조건을 100 step 연속 유지

## 작업별 정책

모든 학생 정책은 과거 5프레임 proprioception과 현재 action·속도 명령·Box/Bin QR 관측으로 구성된 517차원 입력을 사용함. Box/Bin world pose와 실제 contact·lift flag는 학생 입력에서 제외함.

| 작업 | 출력 | 현재 상태 |
|---|---|---|
| Box 접근·Pre-grasp | 23차원 손목·손·base·torso 명령, Pink/Homie가 43관절로 변환 | DAgger 학습 및 중앙 조건 handoff 확인 |
| Box 파지·리프트·hold | 43차원 absolute joint target | 실제 접근 종료 상태 평가 16/20 성공, 조건부 Handoff 15/16 성공 |
| Box Carry | 43차원 absolute joint target | 기존 성공 12개와 최신 gate snapshot 성공 2개를 결합해 후보 재학습 완료 |
| Bin Place·release | 43차원 absolute joint target | Place v4 기존 성공 6개와 신규 성공 1개를 결합해 후보 재학습 완료 |

## 학습 방식

- Adversarial Motion Priors 기반 적대적 모방학습이 아님
- 517차원 관측에서 Teacher의 23차원 또는 43차원 action을 회귀하는 supervised Behavior Cloning 사용
- 3층 Multilayer Perceptron, ELU activation, AdamW optimizer, normalized Mean Squared Error loss 사용
- 학생 rollout의 한 손 접촉, 손목 이탈, Box 회전, lift slip, QR 손실 상태를 DAgger correction으로 추가해 다시 Behavior Cloning함

## 검증과 연결

- 파지 checkpoint는 offline error뿐 아니라 20회 physical rollout, bad orientation, 2초 hold를 검사함
- checkpoint와 rollout metrics, 성공 snapshot을 SHA-256으로 묶어 무결성을 검증함
- 유한 상태 기계가 물리 조건에 따라 접근·파지·Carry·Place 정책을 순서대로 선택함
- strict 12 cm 기준보다 5 cm 높은 17 cm에서 상체 target을 고정해 물리 settling 여유를 확보함
- 현재 학생 연쇄는 접근, 양손 접촉, 100-step strict hold, 추가 100-step Handoff까지 확인됨
- 신규 Carry·Place 후보와 100-step target blending을 포함한 전체 20회 검증 수행 완료
- 최신 고정-seed 20회 실측: `P(approach)=20/20`, `P(grasp|approach)=16/20`, `P(handoff|grasp)=15/16`
- 파지와 2초 Handoff는 각각 80% 이상 gate를 통과했으며 다음 병목은 Loaded Stand 이후 보행임

## 강화학습

- 네 단계 모두를 공통 43차원 관절 target으로 출력하는 동결 Behavior Cloning 명목 제어기 구성 완료
- `action = nominal + 0.20 × clipped residual` 합성과 100-step 단계 전이 blend 구현 완료
- Proximal Policy Optimization actor의 residual 평균을 0으로 초기화하고 Random Network Distillation 안전 마스크 연결 완료
- 128개 환경의 3회 smoke 후 100회 추가 학습, 307,200 transition, 최종 checkpoint 생성 완료
- 최종 측정은 성공률 0%, 비정상 자세 종료율 100%로 production gate 실패
- 구현과 학습 실행은 완료되었으나 접근 명목 제어기의 무작위 reset 물리 안정성이 부족하여 checkpoint 승격 거부

## 다음 단계

1. 통과한 파지→Handoff 설정을 고정하고 실제 FSM 성공 snapshot corpus 확장
2. Loaded Stand 이후 상자를 든 보행 정책 학습과 Carry drop 감소 검증
3. `P(walk|loaded stand)`가 확보된 후보로 Place까지 전체 20회 gate 재평가

상세 기술 설명은 `README_G1_43DOF_LOCO_MANIPULATION_DETAILED.md` 참고.
