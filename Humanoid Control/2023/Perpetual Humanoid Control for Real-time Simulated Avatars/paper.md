# Perpetual Humanoid Control for Real-time Simulated Avatars

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Humanoid Control
- Physics-based Simulation

---
url:
- [paper](https://arxiv.org/abs/2305.06456)
- [project](https://www.zhengyiluo.com/PHC-Site/)
---
- **Authors**: Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, Weipeng Xu
- **Venue**: ICCV 2023

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)
  - [3.1. Goal Conditioned Motion Imitation with Adversarial Motion Prior](#31-goal-conditioned-motion-imitation-with-adversarial-motion-prior)

---

## ⚡ 요약 (Summary)

---

## 📖 Paper Review

## Abstract

물리 기반 humanoid 컨트롤러 제시
- noisy input(비디오에서 추정된 포즈 또는 언어에서 생성된 포즈)과 예상치 못한 실패가 있는 경우에도 고충실도의 동작 모방과 fault-tolerant(결함 허용) 행동을 달성
- 외부 안정화력 없이 만 개의 motion clips를 학습할 수 있음
- 실패 상태에서 자연스럽게 회복하는 법을 배움
- reference motion이 주어졌을 때, 리셋 없이 시뮬레이션 된 아바타를 영원히 컨트롤 할 수 있음
- 점점 더 어려운 동작 sequence를 학습하기 위해 새로운 네트워크 용량을 동적으로 할당하는 progressive multiplicative control policy(PMCP)를 제안
    - catastrophic forgetting 없이 large-scale motion database와 새로운 작업(예: 실패 상태에서 회복)을 추가하기 위한 효율적인 scaling을 가능하게 함
- 실시간 multi-person avatar 사용 사례에서 비디오 기반 포즈 추정기와 언어 기반 모션 생성기에서 나온 노이즈가 있는 포즈를 모방하여 컨트롤러의 효과를 보임

## 1. Introduction

물리 기반 모션 모방
- 현실적인 인간 동작을 생성
- 그럴듯한 환경 상호작용을 가능하게 함
- 미래의 가상 아바타 기술을 발전시킬 수 있음

시뮬레이션에서 high-degree-of-freedom(DOF) 휴머노이드를 컨트롤하는 것은 상당한 도전을 제시
- 넘어지거나, 발을 헛디디거나, 기준 동작에서 벗어나거나 회복하려 애쓸 수 있음
- noisy 비디오 관찰에서 추정한 포즈를 사용하여 시뮬레이션된 휴머노이드를 컨트롤하는 것은 종종 휴머노이드가 넘어질 수 있게 함
- 이러한 한계는 physics-based 방법의 광범위한 적용을 막음
- 현재의 제어 정책은 noisy 관측(예: 비디오, 언어)을 처리할 수 없음

아바타용으로 물리적으로 시뮬레이션된 휴머노이드를 적용
- 첫 번째 과제: 높은 성공률로 인간과 유사한 동작을 충실히 재현할 수 있는 동작 모방기(컨트롤러) 학습
    - 강화학습 기반의 모방 정책이 유망한 결과를 보여줌
        - 단일 정책으로 AMASS(만 개의 클립, 40시간 동작)과 같은 대규모 데이터셋에서 모션을 성공적으로 모방하는 것은 달성되지 않음
    - 더 크거나 mixture of expoert 정책들을 사용하는 것은 약간의 성공을 보임
        - 아직 가장 큰 데이터셋에는 적용되지 못함
    - 휴머노이드를 안정화시키기 위해 외부 힘을 사용하는 방법
        - Residual force control(RFC)
            - AMASS 데이터셋을 97%까지 따라할 수 있는 모션 모방기를 만드는 데 도움이 됨
            - 비디오에서 human pose 추정 및 언어 기반 모션 생성에서 성공적으로 응용됨
        - 외부 힘은 humanoid를 조종하는 "신의 손"처럼 작용하여 물리적 현실성을 훼손
        - flying, floating과 같은 아티팩트 발생
        - RFC를 사용하면 모델이 humanoid에 비물리적인 힘을 자유롭게 적용할 수 있기에 현실성이 훼손됨

- 두 번째 과제: noisy input과 실패 사례를 처리하는 방법
    - 비디오 입력과 관련하여 인기 있는 포즈 추정 방법에서는 폐색, 어려운 시점과 조명, 빠른 움직임 등으로 인해 floating, foot sliding, 물리적으로 불가능한 포즈와 같은 아티팩트가 흔하게 나타남
    - 이러한 경우를 처리하기 위해 대부분의 물리 기반 방법들은 실패 조건이 발생했을 때 휴머노이드를 reset하는 방법에 의존
    - 리셋은 고품질의 레퍼런스 포즈를 요구
        - 포즈 추정은 잡음이 많은 특성을 갖기에 얻기 어려운 경우가 많음
        - 넘어졌다가 불안정한 자세로 리셋되는 악순환을 초래
    - 예상치 못한 낙하와 noisy 입력을 자연스럽게 처리하고, 실패 상태에서 자연스럽게 회복하며, 모방을 재개할 수 있는 컨트롤러를 갖는 것이 중요

**논문의 목표**: 
사람의 비디오 관측을 사용하여 아바타를 컨트롤하는 실시간 가상 아바타를 컨트롤하기 위해 설계된 humanoid controller 설계

**Perpetual Humanoid Control(PHC)**
- 모션 모방에서 높은 성공률을 달성하기 위한 단일 정책
- 실패 상테에서 자연스럽게 회복할 수 있음
- Progressive multiplicative control policy(PMCP) 제안
    - catastrophic forgetting 없이 전체 AMASS 데이터셋에서 모션 시퀀스를 학습하기 위함
    - 더 어렵고 어려운 동작 시퀀스를 다른 "작업"으로 취급
    - 학습할 새로운 네트워크의 용량을 점진적으로 할당
    - 이를 통해 PMCP는 더 어려운 동작을 학습할 때도 더 쉬운 동작 clip을 모방하는 능력을 유지
    - 컨트롤러가 모션 모방 능력을 손상시키지 않으면서 실패 상태 회복 작업을 학습할 수 있게 함
- 전체 파이프라인에서 Adversarial Motion Prior(AMP)를 채택하고 실패 상태 복구 동안 자연스럽고 인간과 유사한 행동을 보장
- 대부분의 동작 모방 방법은 link position과 회전 추정치 모두를 입력으로 해야 함
    - 제안 방법은 link position만 요구
    - 이 입력은 vision-based 3D keypoint 추정기나 VR 컨트롤러의 3D pose 추정치를 통해 더 쉽게 생성될 수 있음

**논문의 기여**
1. Perpetual Humanoid Controller 제안
    - 외부력 없이 AMASS 데이터셋의 98.9% 성공적인 모방이 가능
2. Progressive Multiplicative Control Policy(PMCP)
    - 대규모 동작 데이터셋에서 catastrophic forgetting 없이 학습하고, 실패 상태 복구와 같은 추가 능력이 가능하도록 함
3. 우리의 컨트롤러는 task에 구애받지 않음
    - off-the-shelf 비디오 기반 포즈 추정기와 drop-in 해결책으로 호환됨
- MoCap과 비디오에서 추정된 모션 모두를 평가하여 컨트롤러의 능력 입증
- 웹캠 영상을 입력으로 사용하여 지속적으로 시뮬레이션된 아바타를 구동하는 실시간(30fps) 데모를 보임

## 2. Related Works


## 3. Method
- $\hat{q}_t \triangleq (\hat{\theta}_t, \hat{p}_t)$: reference pose
- $\hat{\theta}_t \in \mathbb{R}^{J \times 6}$: 3D joint rotation(6DoF 회전 표현 사용)
- $\hat{p}_t \in \mathbb{R}^{J \times 3}$: 3D joint position
- $J$: 휴머노이드의 link 수

reference pose $\hat{q}_{1:T}$로부터, finite difference(유한 차분)을 통해 reference 속도 $\hat{\dot{q}}_{1:T}$를 계산할 수 있음
- $\hat{\dot{q}}_{1:T} \triangleq (\hat{w}_t, \hat{v}_t)$: 각속도 $\hat{w}_t \in \mathbb{R}^{J \times 3}$ 과 선속도 $\hat{v}_t \in \mathbb{R}^{J \times 3}$으로 구성됨
- 회전 기반과 keypoint 기반 동작 모방을 입력으로 구분
    - 회전 기반 모방: reference poses $\hat{q}_{1:T}$에 의존(회전과 키포인트)
    - keypoint 기반 모방: 3D keypoints $\hat{p}_{1:T}$만 요구
> $\tilde{\cdot}$ : 포즈 추정기/키포인트 검출기에서의 운동학적 양(물리 시뮬레이션에서의)  
> $\hat{\cdot}$ : Motion Capture(MoCAP)으로부터 얻은 GT 양
> 기호 없음: 물리 시뮬레이션에서 나온 값

### 3.1. Goal Conditioned Motion Imitation with Adversarial Motion Prior

controller는 goal-conditioned RL의 일반적인 프레임워크(Fig 3 참조)를 따름 
- goal-conditioned policy $\pi_{PHC}$: reference motion $\hat{q}_{1:t}$ 또는 keypoints $\hat{p}_{1:T}를 모방하는 임무를 수행
- tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$로 정의되는 Markov Decision Process(MDP)로 task를 공식화
    - state, action, transition dynamics(전이 동역학), reward function, discount factor로 구성
- 물리 시뮬레이션은 상태 $s_t \in S$와 transition dynamics $\mathcal{T}$를 결정
    - policy $\pi_{PHC}$가 per-step action $a_t \in \mathcal{A}$를 계산
- simulation state $s_t$와 reference motion $\hat{q}_t$에 기반하여 보상 함수 $\mathcal{R}$은 정책 학습 신호로서 보상 $r_t = \mathcal{R}(s_t, \hat{q}_t)$를 계산
- 정책의 목표: discounted reward $\mathbb{E} \big[ \Sigma_{t=1}^T \gamma^{t-1} r_t \big]$를 최대화하는 것
- $\pi_{PHC}$를 학습하기 위해 proximal policy gradient(PPO)를 사용

**상태**