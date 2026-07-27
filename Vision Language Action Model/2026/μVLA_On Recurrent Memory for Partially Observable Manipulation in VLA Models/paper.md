# μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Vision Language Action Model
- Embodied AI & Robotics
- Recurrent Memory & POMDP

---
url:
- [paper](https://arxiv.org/abs/2606.12497) (arXiv 2026)
- [project](https://avanturist322.github.io/mu-vla/)

---
- **Authors**: Egor Cherepanov, Nikita Kachaev, Daniil Zelezetsky, Aydar Bulatov, Artem Pshenitsyn, Yuri Kuratov, Alexey Skrynnik, Aleksandr I. Panov, Alexey K. Kovalev
- **Venue**: arXiv 2026

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Preliminaries](#3-preliminaries)
- [4. μVLA: A Controlled Family of Recurrent VLAs](#4-μvla-a-controlled-family-of-recurrent-vlas)
- [5. Experiments](#5-experiments)
  - [Suites and training mixture](#suites-and-training-mixture)
  - [Backbone and training recipe](#backbone-and-training-recipe)
  - [Members of the family and reference points](#members-of-the-family-and-reference-points)
  - [Comparison policy](#comparison-policy)
  - [Evaluation](#evaluation)
  - [5.1 Results on MIKASA-Robo](#51-results-on-mikasa-robo)
  - [5.2 Recurrence is benign on a Markovian suite (LIBERO)](#52-recurrence-is-benign-on-a-markovian-suite-libero)
  - [5.3 Memory diagnostics and inference-time ablations](#53-memory-diagnostics-and-inference-time-ablations)
- [6. Discussion](#6-discussion)
- [7. Conclusion](#7-conclusion)
- [Appendix Details & Diagnostics](#appendix-details--diagnostics)

---

## Abstract

- **문제 정의**: 기존 Vision-Language-Action (VLA) 모델들은 현재 관측(observation)만으로 미래 행동 덩어리(action chunk)를 예측하는 Markovian 가정을 바탕으로 작동함. 그러나 관측 정보가 가려지거나(occluded) 과거 사건에 의존하는 부분 관측(Partially Observable) 환경에서는 이러한 가정이 실패함.
- **기존 연구의 한계**: 기존 메모리 강화 VLA 모델들은 순환성(recurrence), 검색(retrieval), 압축 모듈, 보조 손실(auxiliary loss), 계층적 메모리, 타스크 특화 아키텍처 변경을 동시에 도입하여, "순환성(recurrence) 자체의 순수한 기여"가 외부 수식 장치와 얽혀(entangled) 명확히 분리되지 않았음.
- **핵심 제안 ($\mu\text{VLA}$)**: 강력한 사전학습 VLA 백본(OpenVLA-OFT) 내부에서 순환성만을 격리하여 조절 연구(controlled isolation study)를 수행함.
  - 트랜스포머 아키텍처에 추가 모듈이나 보조 손실 없이, 타임스텝 간 전파되고 셀프 어텐션(self-attention)으로 업데이트되는 소수의 학습 가능한 메모리 토큰($m$)을 도입함.
  - 시간 절단 역전파(Truncated Backpropagation Through Time; TBPTT, 길이 $K$) 및 EMA 메모리 업데이트 규칙으로 엔드투엔드 파인튜닝을 수행함.
- **주요 성과**:
  - **MIKASA-Robo (부분 관측 벤치마크)**: 가장 강력한 설정($m=64, K=2$)에서 5개 학습 타스크의 평균 성공률(SR)을 기존 0.42에서 **0.84**로 비약적으로 상승시킴. 동일한 메모리 구조를 갖는 held-out 타스크에서도 메모리리스 베이스라인(0.07) 대비 **0.23**으로 향상됨.
  - **LIBERO (완전 관측 벤치마크)**: 평균 성공률 **96.2%**를 달성하여 완전 관측 환경에서도 성능 저하(regression)가 없음을 실증함.

![alt text](./images/Fig%201.png)
> **Figure 1. µVLA: 사전 학습된 VLA 백본 안에서의 recurrent memory**
> - 각 timestep t에서 모델은 visual 관찰, 고유 감각, language, memory tokens를 받아들임
> - 그 다음 행동과 업데이트된 메모리 토큰을 예측해서 t + 1로 반복적으로 넘김

![alt text](./images/Fig%202.png)
> **Figure 2. TBPTT training of µVLA**
> - 사전 학습된 VLA는 K step의 window 위에 펼쳐져 있음
> - memory token은 $t_0$에서 $M^{init}$에서 초기화됨
> - 이어지는 단계들 사이에서 반복적으로 전달됨
> - gradient는 unrolled pathway를 따라 흐르다가 window 경계에서 분리됨

---

## 1. Introduction

- **배경 및 동기**:
  - 기존 VLA 모델들은 멀티모달 시퀀스 상의 다음 토큰 예측(next-token prediction)으로 로봇 조종(manipulation)을 모델링하며, 현재 관측 $o_t$만으로 미래 행동 덩어리 $\boldsymbol{a}_{t:t+H-1}$를 예측함.
  - 그러나 실제 로봇 환경은 결정 관련 정보가 일시적이거나(transient) 가려지는(occluded) Partial Observability(POMDP) 시나리오가 빈번함.
  - 기존 역사 정보 반영 시도:
    - **콘텍스트 윈도우 확장 / KV-cache 재사용**: 문맥 길이는 늘어나지만 바운드되어 있고, 유지해야 할 정보를 결정하는 구조적 메커니즘이 부족함 (Jang et al., 2025; Fan et al., 2025).
    - **외부 메모리 및 검색 (External Memory & Retrieval)**: 메모리 저장 및 검색 유연성은 증가하나 별도의 저장 정책과 추가 컴포넌트가 필요함 (Koo et al., 2025; Sridhar et al., 2025; Haresh et al., 2026).
    - **순환 상태 전달 (Recurrence)**: 잠재 상태(latent state)를 전달하여 무엇을 기억할지 학습하지만, 보조 손실이나 전용 아키텍처 변경과 결합되어 순환성의 pure effect를 측정하기 어려웠음 (Li et al., 2026; Xiao et al., 2025; Tur et al., 2026).
- **핵심 기여 (Contributions)**:
  1. **최초의 통제된 순환성 격리 연구 (Controlled Isolation Study)**: 외부 검색, 압축, 계층 메모리, 보조 손실 없이 사전학습 VLA 백본 내부에서 순환성만을 단일 실험 축으로 격리함.
  2. **$\mu\text{VLA}$ 가족 제안**: 메모리 너비 $m$, TBPTT 길이 $K$, 메모리 쓰기 규칙(write rule: 시간 절단 역전파 또는 분리된 EMA)으로 파라미터화된 OpenVLA-OFT 순환 파인튜닝 모델 패밀리를 구축함.
  3. **순환 성능 한계 영역 정밀 측정 (Capability Envelope Calibration)**: MIKASA-Robo 및 LIBERO 벤치마크에서 3가지 역량 영역(학습 분포 내부, 동일 메모리 구조 전이, 신규 메모리 구조 전이)을 명확히 정의하고 수많은 진단 실험(대표표현 역학, 어텐션 롤아웃, 메모리 차단 인터벤션 등)을 수행함.

---

## 2. Related Work

- **Vision-Language-Action (VLA) Models**:
  - RT-1 (Brohan et al., 2023), RT-2 (Zitkovich et al., 2023), OpenVLA (Kim et al., 2024), OpenVLA-OFT (Kim et al., 2025), Octo (Ghosh et al., 2024), $\pi_0$ (Black et al., 2025) 등은 이미지, 언어 지시문, 프롬프리오셉션을 입력으로 받아 행동 시퀀스를 예측함.
  - 대다수가 현재 관측 $o_t$만을 사용하는 Markovian 정책 가정을 따름.

- **VLA 모델에서의 이력 정보(History) 활용**:
  - **콘텍스트 프레임 연결 (Context Window)**: ContextVLA (Jang et al., 2025), Long-VLA (Fan et al., 2025).
  - **외부 메모리 및 검색 (External Memory & Retrieval)**: Hamlet (Koo et al., 2025), Memer (Sridhar et al., 2025), MAP-VLA (Li et al., 2025), Notes-to-Self (Haresh et al., 2026), Mem (Torne et al., 2026).
  - **순환 상태 전달 (Recurrence)**: ReMem-VLA (Li et al., 2026), AVA-VLA (Xiao et al., 2025), VPWEM (Lei et al., 2026), Recurrent-Depth VLA (Tur et al., 2026), MemoryVLA (Shi et al., 2025).
- **본 논문의 고유 위치 (Position)**:
  - 복잡한 모듈과 보조 손실을 배제하고, Recurrent Memory Transformer (RMT; Bulatov et al., 2022; Cherepanov et al., 2026a,b) 스타일의 소수 메모리 토큰을 VLA 백본의 셀프 어텐션 내부에서 환경 스텝 간 전파하도록 설계함.
  - 타스크 간 의존 구조가 명확한 MIKASA-Robo (Cherepanov et al., 2026)와 완전 관측 제어 환경인 LIBERO (Liu et al., 2023)에서 순환성 자체의 기여를 정밀하게 분석함.

---

## 3. Preliminaries

- **POMDP 문제 정의**:
  - 조종 과제를 POMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, \Omega, \mu_0 \rangle$ 로 모델링함. 관측이 부분적이므로 최적 정책은 과거 이력 전체에 의존함: $\pi^\star(a_t \mid o_{1:t}, \ell)$.
- **표준 VLA 학습 (Standard VLA Training)**:
  - 데모 에피소드에서 무작위 타임스텝 $t$를 독립적(i.i.d.)으로 샘플링하여 오픈루프 행동 덩어리 $\boldsymbol{a}_{t:t+H-1}$를 예측함. 타임스텝 간 지속 상태를 유지하지 않음.
- **순환 VLA 학습 (Recurrent VLA Training)**:
  - 시간 순서가 보장된 에피소드 스트림 상에서 순환 상태 $\boldsymbol{m}_t$를 전달함:
    $$
    (\hat{a}_t^{(t)}, \ldots, \hat{a}_{t+H-1}^{(t)}) = \pi_\theta(o_t, \boldsymbol{m}_{t-1}, \ell), \quad \boldsymbol{m}_t = f_\theta(o_t, \boldsymbol{m}_{t-1}, \ell)
    $$
  - 데모 관측을 따라가면서 잠재 상태 $\boldsymbol{m}_t$는 모델 스스로 생성하는 교사 강제(Teacher-Forcing) 유사 방식을 적용함.
- **베이스라인 백본 및 벤치마크**:
  - **OpenVLA-OFT**: SigLIP (Zhai et al., 2023) / DINOv2 (Oquab et al., 2024) 비전 인코더 + Llama-2-7B 백본 (Touvron et al., 2023) 기반으로 $H=8$ 연속 행동 덩어리를 예측.
  - **MIKASA-Robo**: 큐 회상(cue-recall), 차폐 추적(occlusion), 순차/예측 메모리 타스크로 구성된 대표적 부분 관측 로봇 조종 벤치마크 (Cherepanov et al., 2026).

---

## 4. μVLA: A Controlled Family of Recurrent VLAs


![alt text](./images/Fig%203.png)
> **Figure 3. memory-action guard가 있는 attention mask**
> - memory token은 observation, proprioception, language, 이전 메모리 상태에만 주의를 기울일 수 있음
> - action token은 읽을 수 없음
> - 이렇게 하면 recurrent state가 단순히 시연된 action을 복사하는 걸 막음
> - 대신 작업과 관련된 observation을 encoding하도록 유도

- **메모리 토큰 구성 및 위치 (Memory Tokens & Recurrence)**:
  - OpenVLA-OFT 멀티모달 토큰 시퀀스의 PROPRIO와 TEXT 사이 위치에 $m$개의 학습 가능한 메모리 토큰 $\boldsymbol{M}_t \in \mathbb{R}^{m \times d}$를 삽입함:
    $$
    [\text{BOS}] \, [\text{VIS}] \, [\text{PROPRIO}] \, [\boldsymbol{M}_t] \, [\text{TEXT}] \, [\text{ACT}] \, [\text{STOP}]
    $$
  - 에피소드 첫 스텝에서는 학습 가능한 초기 파라미터 $\boldsymbol{M}_0 \triangleq \boldsymbol{M}^{\text{init}}$ 사용.
  - $t \ge 1$ 스텝에서는 이전 스텝의 셀프 어텐션 출력 중 메모리 위치의 은닉 상태를 추출하여 다음 스텝의 메모리로 전파:
    $$
    \boldsymbol{M}_t = h_\theta(o_t, \ell; \boldsymbol{M}_{t-1})[\text{mem positions}]
    $$
- **3가지 핵심 기술적 함정 완화 요소**:
  1. **행동 복사 방지 어텐션 마스크 (Attention-Mask Guard)**:
     - 양방향 어텐션 사용 시 메모리 토큰이 예측된 행동 토큰을 그대로 복사하는 자가참조적 무의미 해(degenerate self-referential solution) $\boldsymbol{M}_t^{(i)} = \phi(\text{ACTION}_t^{(i)})$에 빠지는 위험 방지.
     - **Context-to-Action 어텐션 차단**: 메모리 토큰(및 멀티모달 프라이머)이 ACTION 영역을 어텐션하지 못하도록 마스크의 해당 블록을 0으로 차단함.
  2. **Round-Robin 에피소드 데이터로더 (Episodic Dataloader)**:
     - 무작위 타임스텝 셔플링 대신 $B$개의 독립된 에피소드 스트림을 타임스텝 순서대로 보장하며, 에피소드 경계에서 `is_first` 플래그로 해당 스트림만 $\boldsymbol{M}^{\text{init}}$으로 리셋함.
  3. **시간 절단 역전파 (TBPTT) & EMA 쓰기 규칙**:
     - **TBPTT ($K$)**: $K$개 연속 스텝에 대해 L1 행동 손실을 누적하여 엔드투엔드 역전파 수행.
     - **EMA 쓰기**: 역전파를 단일 스텝에 국한하고 메모리를 차단된 지수 이동 평균 $\boldsymbol{M}_{t+1} = \alpha \boldsymbol{M}'_t + (1-\alpha) \boldsymbol{M}_t$로 업데이트.
  4. **후퇴 지평선 추론 (Receding-Horizon Inference)**:
     - 매 스텝마다 모델을 재질의하고 예측 덩어리의 첫 번째 행동만 실행함. 메모리 업데이트 주기와 환경 스텝 간의 불일치(mismatch)를 제거함 (Chi et al., 2025).

---

## 5. Experiments

- **연구 질문 (Research Questions; RQs)**:
  - **RQ1**: 백본을 처음부터 다시 학습하지 않고, 메모리리스 VLA의 파인튜닝 단계에서 순환 메모리를 추가할 수 있는가?
  - **RQ2**: 순환성의 엔드투엔드 파인튜닝이 스텝 간 경사도 흐름(TBPTT vs 스텝별 detach $K=1$ vs 학습 없는 EMA 쓰기)의 이점을 얻는가? 그렇다면 최적의 절단 길이 $K$는 얼마인가?
  - **RQ3**: 학습된 메모리는 추론 시 (a) 학습 타스크와 메모리 의미론(semantics)을 공유하는 held-out 타스크 대 (b) 메모리 의미론이 개념적으로 다른 held-out 타스크에서 어떻게 동작하는가?
  - **RQ4**: 순환 채널 추가가 메모리가 필요 없는 Markovian (MDP) 타스크(예: 완전 관측 LIBERO 수트)에서 성능 저하(degradation)를 일으키는가?

- **Suites and training mixture**:
  - **MIKASA-Robo (부분 관측)**:
    - 5개 학습 타스크 믹스처: 큐 회상(`RememberColor5`, `RememberShapeAndColor3x3`), 차폐 추적(`ShellGamePush`), 순차/예측 메모리(`TakeItBack`, `InterceptMedium`).
    - 18개 held-out 평가 환경:
      - **Matched memory semantics (11개)**: 학습 타스크와 동일 메모리 구조나 난이도/행동 원형이 다른 타스크 (`ShellGame` touch/pick, `Intercept` slow/fast, `InterceptGrab`, 다양한 크기의 `RememberColor`/`RememberShapeAndColor`).
      - **Novel memory semantics (7개)**: 학습 믹스처에 없는 새로운 메모리 구조 (`RememberShape` 패밀리, 내부 진행 상황 추적이 필요한 `Rotate` 패밀리).
  - **LIBERO (완전 관측)**: 완전 관측 제어 벤치마크 4개 수트.

- **Backbone and training recipe**:
  - 공개된 OpenVLA 체크포인트를 기반으로 백본 및 행동 헤드에 LoRA (rank 32) 적용, 메모리 위치 임베딩 및 $\boldsymbol{M}^{\text{init}}$ 파라미터 파인튜닝.

- **Members of the family and reference points**:
  - **MIKASA-Robo 10개 실험 조건**:
    - **메모리리스 레퍼런스 포인트 (3개)**:
      - (R1) `OpenVLA-OFT`: 원본 학습 레시피 및 오픈루프 청크 실행.
      - (R2) `OpenVLA-OFT† (episodic)`: episodic 데이터로더 및 $m=0$ 메모리 토큰 적용 (데이터로더 변경 효과 격리).
      - (R3) `OpenVLA-OFT† (episodic) + 1st observation`: 첫 번째 타임스텝의 상단 탑뷰 프레임을 매 스텝 입력에 추가 (초기 관측에 큐가 존재하는 타스크의 오라클 레퍼런스).
    - **$\mu\text{VLA}$ 패밀리 변형 (7개)**:
      - $m=64$: TBPTT ($K=1, 2, 8$), EMA 쓰기 ($\alpha=0.1$, 그래프 분리), 행동 복사 방지 마스크를 제거한 EMA 변형.
      - 대역폭 탐침: $m=1$ ($K=8$).
      - 싱글 타스크 컨트롤: $m=64, K=8$ (`RememberColor5` 전용 학습).
  - **LIBERO 조건**: OpenVLA-OFT, $\mu\text{VLA}$ ($m=64, K=8$), $\mu\text{VLA}$ ($m=64, \text{EMA}$).

- **Comparison policy**:
  - 복잡한 외부 메모리 시스템으로 벤치마크 점수를 극대화하는 것이 아니라, 고정된 VLA 백본 내 순환성(recurrence) 고유의 기여를 격리 분석하는 것이 목표임.
  - 최신 메모리 강화 VLA (CronusVLA, MemoryVLA 등)의 기존 발표 성능은 맥락을 제공하기 위해 함께 제시함.

- **Evaluation**:
  - 환경당 100회 결정론적 에피소드 평가, Receding-horizon 추론 제어 적용.

![alt text](./images/Table%201.png)
> **Table 1. 환경별 SR on all 23 MIKASA-Robo-VLA 환경
> - 행: RQ3의 memory-semantic split로 그룹화
> - 열: $\pi_{0.5}$ memoryless baseline
> - 세 가지 OpenVLA-OFT reference points:
>   - (R1): 원레 dataloader recipe
>   - (R2): recurrence 없이 episode dataloader
>   - (R3): 첫 번째 프레임에 접근 가능한 episodic dataloader
> - (+1st) 열은 episode의 첫 번째 관찰을 모든 timestep에서 모델 컨텍스트에 추가
>   - 잠재적 단서가 초기 관찰에서 완전히 보이는 task에 대해 oracle-stype의 upper bound 역할을 함
> - 나머지 열은 $m=64$에서 다른 recurrent write(K = 1, 2, 8, EMA, EMA without the action-copy guard)를 갖는 µVLA 변형 및 $m=1$ bandwidth probe와 single-task RemenberColor5 control
> - $\dagger$: episodic dataloader
> - 최고의 memory 구성(m=64, K=2, orange)와 oracle 열(초록색)이 하이라이트 되어 있음

![alt text](./images/Table%202.png)
> **Table 2. LIBERO 수트에서의 성능 비교**
> - 4개 LIBERO suites 및 전체 평균 성공률 보고
> - *: memory-augmented VLA 모델 표시

### 5.1 Results on MIKASA-Robo

- **Headline numbers per RQ**:
  - **RQ1 (파인튜닝 단계 순환성 추가)**: 5개 학습 타스크 평균 성공률이 메모리리스 베이스라인 (OpenVLA-OFT: 0.42, OpenVLA-OFT† episodic: 0.48) 대비 $\mu\text{VLA}$ ($m=64, K=2$)에서 **0.84**로 대폭 상승. 특히 `RememberColor5`에서 0.04/0.09에서 **0.93**으로 폭발적 향상.
  - **RQ2 (스텝 간 경사도 및 최적 $K$)**: TBPTT 길이 $K$는 U자형 성과를 나타냄 ($K=1$: 0.57, $K=8$: 0.57 대비 **$K=2$: 0.84**가 최고 성과). EMA 쓰기는 $K=8$ 수준(0.57)을 추종함.
  - **RQ3 (메모리 의미론적 이동 전이)**: Matched semantics 11개 환경에서는 $K=2$에서 **0.23** 달성 (episodic 베이스라인 0.07 대비 큰 향상). Novel semantics 7개 환경에서는 메모리리스 레퍼런스와 유사한 수준에 머물러 순환성의 전이 한계 영역을 측정함.
  - **RQ4 (Markovian 타스크 영향)**: LIBERO 벤치마크 결과로 연결됨.

### 5.2 Recurrence is benign on a Markovian suite (LIBERO)

- **완전 관측 환경에서의 성능 검증**:
  - 완전 관측 제어 벤치마크인 LIBERO 4개 수트 평균 성공률: OpenVLA-OFT (95.4%) 대비 **$\mu\text{VLA}$ ($m=64, K=8$) 96.2%** 달성.
  - 순환 메모리 채널을 도입하더라도 완전 관측 타스크에서 성능 저하(regression)가 발생하지 않고 무해함(benign)을 입증함.
  - 특히 Object 및 Long-10 수트에서 성능 향상이 두드러짐.
- **Receding-Horizon 추론의 필수성**:
  - 학습 시 스텝별 메모리 업데이트와 추론 시 오픈루프 $H$-step 청크 실행 간의 미스매치(train-test mismatch)로 인해, 오픈루프 실행 시 Long-10 성공률이 95.8%에서 5.4%로, Goal이 96.6%에서 35.8%로 급락함을 확인함.

### 5.3 Memory diagnostics and inference-time ablations

![alt text](./images/Fig%204.png)
> **Figure 4. Cosine Distance $M_t$ vs. $M_{t-1}$ on RemeberColor5(K=2, m=64)**
> - 연한 파랑 선: 개별 메모리 토큰
> - 짙은 회색: 64개 전체의 평균
> - episode에는 두 단계 전환이 있음
> - cue cube는 $t=4$까지 보이며, $t=5$에 사라지고 agent는 $t=0까지 빈 테이블을 관찰
> - $t=10$에는 후보 큐브가 나타나고 task가 기억 유지에서 검색 및 행동 수행으로 전환됨
> - 색상은 삽입된 snapshot의 프레임과 일치
> - 두 전환 모두 메모리 변화에서 뚜렷한 스파이크를 만듦

![alt text](./images/Fig%205.png)
> **Figure 5. Memory intervention on RememberColor5**
> - baseline, noise, freeze_first 환경에서 100 episode 기준 SR

![alt text](./images/Fig%206.png)
> **Figure 6. Memory -> vision attention rollout on RememberColor5(K=2, m=64)**
> - 에피소드 초반에 queue 패널이 보이고 이후 가려짐
> - memory attention은 cue grid가 보이는 종안 집중되고, 가려진 후에는 퍼지면서 Figure 13의 phase transition과 일치함

- **Representation dynamics and attention rollouts (대표표현 역학 및 어텐션 롤아웃)**:
  - **위기 시점 재조직화**: `RememberColor5` 타스크에서 큐 제시 종료 시점($t=4 \to 5$)과 캔디데이트 등장 및 선택 시점($t=9 \to 10$)에서 메모리 코사인 거동 $1 - \cos(\boldsymbol{M}_t, \boldsymbol{M}_{t-1})$에 정교한 피크 스파이크 발생. 라텐트 페이즈 전환 시 메모리 상태가 동적으로 재조직됨.
  - **어텐션 롤아웃 시각화**: `memory → vision` 어텐션 분석 시, 초기 스텝에서는 큐 물체 위치에 어텐션이 집중되고 큐가 사라진 후에도 내부 은닉 상태로 저장 및 전파됨을 검증.
- **Causal intervention on the memory channel (메모리 채널 인과적 개입)**:
  - **Gaussian Noise 주입**: 추론 시 $\boldsymbol{M}_t$에 노이즈 주입 시 성공률 폭락 (`RememberColor5` $K=2$: $0.94 \to 0.09$, `TakeItBack` $K=2$: $0.99 \to 0.21$). 추론 시 메모리 채널을 실제로 읽어 구동함을 입증.
  - **Freeze-First ($\boldsymbol{M}_1$ 고정)**: 메모리를 첫 프레임 상태 $\boldsymbol{M}_1$로 고정 시 첫 프레임 큐 타스크는 유지되나, 시간적 동적 추적 타스크(`InterceptMedium` $K=2$: $0.99 \to 0.07$)는 완전 붕괴.
- **Robustness to chunked inference (청크 추론 강인성)**:
  - 오픈루프 청크 길이를 $\{1, 2, 4, 8, 16\}$으로 스윕한 결과, receding-horizon ($chunk=1$) 일 때 메모리 유/무의 성능 격차가 가장 큼. $chunk \ge 2$에서는 큐 회상 타스크 성능이 붕괴함.
- **Length and OOD-cue generalization (길이 및 분포 외 큐 일반화)**:
  - **페이즈 길이 스윕 ($N \in \{3,5,10,20,50,100\}$)**: $K=2$는 $N=3$ (0.93)에서 가장 우수하나 $N=20$ 이상에서는 회상 성능의 약 75%를 상실함.
  - **OOD 컬러 팔레트 일반화**: OOD 컬러 팔레트로 대체된 환경에서도 $K=2$가 가장 뛰어난 견고성(5개 스왑 시 0.48)을 보여, 큐 정보가 단순 색상 복사가 아닌 일정 수준 추상화된 형태로 인코딩됨을 확인.

---

## 6. Discussion

- **RQ1 (파인튜닝 단계 순환성 추가)**: 백본 재학습 없이 파인튜닝만으로 소수 메모리 토큰을 추가하여 부분 관측 성공률을 0.42/0.48에서 0.84로 대폭 향상시킴.
- **RQ2 (스텝 간 경사도 및 최적 $K$)**: $K=2$가 신용 할당(credit assignment)과 신호 해상도 간 최고의 트레이드오프를 제공함.
- **RQ3 (메모리 의미론적 이동 전이)**: 동일 메모리 구조를 갖는 held-out 타스크에는 효과적으로 전이되나(0.23 vs 0.07), 완전히 새로운 메모리 의미론(shape 전용, 회전 추적 등)에서는 베이스라인 수준으로 수렴함.
- **RQ4 (Markovian 타스크 영향)**: LIBERO 수트에서 성능 감소가 없어 순환 채널이 무해함(benign)을 실증.

---

## 7. Conclusion

- 본 논문은 VLA 백본 내부의 최소한의 순환성(minimal in-backbone recurrence)이 부분 관측 로봇 조종 문제에 미치는 효과를 제어된 실험으로 명확히 격리 분석함.
- 소수의 메모리 토큰과 짧은 TBPTT 파인튜닝만으로도 부분 관측 성능을 크게 향상시키며 완전 관측 성능도 유지함을 입증함.
- 거대 외부 검색이나 복잡한 구조 없이도 순환성 자체가 VLA 모델의 성능 한계를 확장하는 강력한 핵심 요소임을 제시함.

---

## Appendix Details & Diagnostics

- **Appendix A & B (상세 논의 및 한계)**:
  - TBPTT 특성상 임의의 아주 긴 장기 메모리(arbitrarily long-horizon)에는 한계가 있으나, 온라인 상태 추정기(online state estimator)로서 안정적으로 동작함.
- **Appendix C (기존 이력 기법 비교)**:
  - ReMem-VLA (Li et al., 2026), AVA-VLA (Xiao et al., 2025), Recurrent-Depth VLA (Tur et al., 2026), VPWEM (Lei et al., 2026)과의 구조적 차이점 정리 (어텐션 마스크 행동 복사 방지 및 Receding-Horizon 추론 결합).
- **Appendix D & F & G (하이퍼파라미터 및 연산 비용)**:
  - LoRA rank 32 (Hu et al., 2022), AdamW, $B=4$/GPU, NVIDIA H100 8장 환경에서 파인튜닝.
  - $K=2$ 설정 시 추론 지연 시간은 약 66.65ms로 OpenVLA-OFT(61.33ms)와 유사하며 매우 효율적임.