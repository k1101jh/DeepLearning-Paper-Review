# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 📌 Metadata
---
분류
- Large Language Models
- State Space Models (SSM)
- Sequence Modeling

---
url:
- [paper](https://arxiv.org/abs/2312.00752) (arXiv 2023)
- [github](https://github.com/state-spaces/mamba)

---
- **Authors**: Albert Gu, Tri Dao
- **Venue**: arXiv 2023

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. State Space Models](#2-state-space-models)
- [3. Selective State Space Models](#3-selective-state-space-models)
- [4. Mamba Architecture](#4-mamba-architecture)
- [5. Hardware-aware Algorithms](#5-hardware-aware-algorithms)
- [6. Experiments](#6-experiments)
- [7. Conclusion](#7-conclusion)

---

## Abstract

Foundation Model(원시 데이터에서 대개 비지도 학습으로 훈련된 신경망)은 대부분 Transformer 아키텍처와 attention 모듈을 기반으로 한다.

Linear attention, gated convolution 및 recurrent models, structured state space models(SSMs)와 같은 많은 subquadratic-time(o(n^2). big-O가 아닌 little-o) 아키텍처가 긴 sequence에 대한 transformer의 계산 비효율성을 해결하기 위해 개발되었지만, attention만큼 성능이 뛰어나진 않았다.

이러한 모델들의 주요 약점: content-based 추론을 진행할 수 없다.
개선 사항
1. SSM 매개 변수를 입력의 함수로 두는 것만으로도 이진 modality에서의 약점을 해결한다.
-> 모델이 현재 토큰에 따라 sequence 길이 차원에서 정보를 선택적으로 전파하거나 잊을 수 있다.
2. 이러한 변경으로 인해 효율적인 convolution을 사용할 수 없더라도, 순환 모드에서 하드웨어 인식 병렬 알고리즘을 설계한다.
이러한 선택적 SSM을 attention이나 MLP 블록(Mamba) 없이 단순화된 end-to-end 신경망 아키텍처에 통합한다.

Mamba
- 빠른 추론(Transformer보다 5배 높은 처리량)과 sequence 길이의 선형 확장을 제공
- 최대 수백만 길이의 sequence까지 실제 데이터에서 성능이 향상
- 다양한 Modality(language, audio, 유전체학)에서 SOTA 달성
- 언어 모델링에서, Mamba-3B 모델은 pre-training 및 downstream 평가에서 동일한 크기의 transformer보다 성능이 뛰어나다.
(두 배 크기의 transformer와 성능이 일치하다)


## 1. Introduction

Foundation Model(FM) 또는 대규모 데이터에 대해 pre-train 후 downstream 작업에 맞게 조정된 대규모 모델은 현대 기계 학습에서 효과적인 패러다임
이러한 FM의 중추는 Sequence Model인 경우가 많다.

현대 FM은 주로 Transformer와 attention layer을 기반으로 한다.
self-attention의 효율성은 context window 내에서 정보를 조밀하게 라우팅하는 능력에 기인한다.
self-attention의 단점
- 유한한 window 외부의 어떤 것도 모델링할 수 없다.
- 창 길이에 대한 2차(quadratic) scaling

단점을 극복하기 위해 보다 효율적인 attention 변형에 대한 연구가 등장했지만, 종종 attention을 효과적으로 만드는 속성을 희생해야 한다.

-> 아직까지 attention 변형 중 어느 것도 domain 전반에 걸친 규모에서 경험적으로 효과적인 것이 없다.


**구조화된 상태 공간 sequence 모델(structured state space sequence models. SSMs)**
- sequence modeling을 위한 아키텍처
- 고전적 상태 공간 모델에서 영감을 받은 RNN과 CNN의 조합으로 생각할 수 있다.
- sequence 길이에서 선형 또는 거의 선형에 가까운 scaling을 사용
- recurrence 또는 convolution 처럼 매우 효율적으로 계산할 수 있다.
- 특정 데이터 modalities에 대한 long-range dependency 모델링을 위한 원칙적인 메커니즘을 갖고 있다.
- Long Range Arena와 같은 벤치마크를 지배했다.

다양한 종류의 SSMs는 audio 및 vision과 같은 연속 신호 데이터와 관련된 영역에서 성공을 거두었다.
하지만, text와 같은 불연속적이고 정보 밀도가 높은 데이터를 모델링하는데는 덜 효과적이다.

sequence 길이를 선형적으로 확장하면서, Transformer의 모델링 능력을 달성하기 위해 여러 관점에서 이전 작업을 개선한 새로운 종류의 selective state space model을 제안

**Selection Mechanism**

이전 모델의 주요 제한 사항:  
입력 종속 방식으로 데이터를 효율적으로 선택하는 기능(특정 입력에 집중하거나 무시하는 기능)을 식별

selective copy 및 induction head와 같은 중요한 합성 작업을 기반으로 하는 직관에 의해  
입력을 기반으로 SSM 매개변수를 매개변수화하여 간단한 선택 매커니즘을 설계
-> 모델은 관련 없는 정보를 필터링하고 관련 정보를 무기한으로 기억할 수 있다.

**Hardware-aware Algorithm**

Selective Mechanism은 모델의 계산에 기술적 문제를 제기

이전의 모든 SSM 모델은 계산 효율성을 높이기 위해 시간 및 입력에 불변이어야 한다.
convolution 대신 scan으로 모델을 반복적으로 계산하는 hardware-aware 알고리즘으로 이 문제를 극복
GPU 메모리 계층 구조의 서로 다른 level간의 IO access를 피하기 위해 확장된 상태를 구체화하지 않는다.
-> 구현은 이론적으로(모든 convolution-based SSms의 pseudo-linear과 비교하여 sequence 길이에서 선형으로 scaling) 이전 방법보다 빠르고 최신 하드웨어에서도 이전 방법보다 빠르다(A100 GPUs에서 최대 3배 빠르다).

**Architecture**
이전 SSM 아키텍처[(Hungry Hungry Hippos. Dao, Fu, Saab, et al. 2023)](https://arxiv.org/abs/2212.14052)의 설계와 Transformers의 MLP 블록을 단일 블록으로 결합하여 이전의 deep sequence model 아키텍처를 단순화하여 선택적 상태 공간을 통합하는 간단하고 동질적인 아키텍처 설계(Mamba)로 이어진다.

선택적 SSM, Mamba 아키텍처
- Sequence에서 작동하는 general foundation model의 backbone으로 적합한 주요 속성(key properties)를 가진 fully recurrent model
1. 고품질  
selectivity는 language 및 genomics와 같은 조밀한 modalities에서 강력한 성능을 제공
2. 빠른 학습 및 추론  
계산 및 메모리는 학습 중에 sequence 길이에 따라 선형적으로 확장되며, 추론 중에 모델을 자동 회귀적(autoregressively)으로 unrolling(반복문을 명령어의 나열로 바꾸는 방식)하는 것은 이전 요소의 캐시가 필요하지 않기 때문에 step당 일정한 시간만 필요하다.
3. 긴 context  
품질과 효율성이 함께 제공되어 sequence 길이 1M까지 실제 데이터에서 성능 향상이 가능함

Mamba의 잠재력을 경험적으로 검증(general sequence FM backbone으로서)
- pretraing quality와 도메인별 작업 성능 모두에서
- 여러 유형의 modalities 및 settings:
    - Synthetics(합성)  (합성 데이터 생성 작업으로 추정)  
        - LLM의 핵심으로 제안된 copying 및 induction heads와 같은 중요한 synthetic task에서 Mamba는 이를 쉽게 해결하고 솔루션을 무기한(>1M tokens) 외삽(extrapolate)할 수 있다.  
        (외삽: 관찰 범위를 넘어섰을 때 다른 변수와의 관계에 기초하여 변수의 값을 추정)
    - Audio 및 Genomics  
        - pretraining 품질 및 downstream 지표 모두에서 오디오 파형 및 DNA sequence 모델링에서 SaShiMi, Hyena, Transformers와 같은 이전 SOTA 모델을 능가(예: 까다로운 음성 생성 데이터셋의 FID를 절반 이상 감소)
        - 두 setting 모두에서 최대 백만 길이의 sequence까지 더 긴 context로 성능 향상
    - Language Modeling
        - Mamba는 pretraining perplexity(언어 모델 평가 지표) 및 downstream 평가 모두에서 Transformer 품질의 성능을 달성하는 최초의 linear-time sequence model.
        - 최대 1B 매개변수의 scaling 범칙을 통해 Mamba가 LLaMa를 기반으로 하는 매우 강력한 최신 Transformer training recipes를 포함하여 광범위한 baseline의 성능을 능가함을 보임
        - 비슷한 크기의 Transformer에 비해 5배의 generation throughput을 갖는다.
        - Mamba-3B의 품질은 Transformer의 두 배 크기와 일치  
        (예: Pythia-3B보다 common sense reasoning에서 평균 4포인트 더 높고, Pythia-7B를 능가하기도 함)

## 2. State Space Models

Structured State Space Sequence Models(S4)
- RNN, CNN 및 고전적인 State space models와 광범위하게 관련된 딥러닝을 위한 최신 sequence 모델
- implicit latent state $h(t) \in \mathbb{R}^N$을 통해 1차원 함수 또는 sequence $x(t) \in \mathbb{R} \mapsto y(t) \in \mathbb{R}$을 매핑하는 particular continuous system에서 영감을 받음
- S4 모델은 4개의 매개변수($\Delta, A, B, C$)로 정의되며, 이는 sequence-to-sequence transformation을 두 단계로 정의

$$
\displaystyle
\begin{aligned}
h'(t) &= Ah(t) + Bx(t)
&(1a)
\quad h_t &= \bar{A}h_{t-1} + \bar{B}x_t
&(2a)
\quad \bar{K} &= (C\bar{B}, C\bar{A}\bar{B}, \dots, C\bar{A}^k\bar{B}, \dots)
&(3a)
\\
y(t) &= Ch(t)
&(1b)
\quad y_t &= Ch_t
&(2b)
\quad y &= x * \bar{K}
&(3b)
\end{aligned}
$$

**이산화**

첫 번째 단계에서 "continuous parameters(연속 매개변수)" $(\Delta, A, B)$를 "discrete parameters(이산 매개변수)" $(A, B)$로 fixed formulas $\bar{A}=f_A(\Delta, A)$와 $\bar{B}=f_B(\Delta, A, B)$를 통해 변환.  
$(f_A, f_B)$: discretization rule
식 4에 정의된 zero-order hold(ZOH)와 같은 다양한 규칙을 사용할 수 있다.

$$
\displaystyle
\begin{aligned}
&\bar{A} = \exp(\Delta A) \quad &\bar{B}=(\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
\quad &(4)
\end{aligned}
$$

이산화
- continuous-time system과 깊은 관련이 있다
    - 해상도 불변성과 같은 추가 속성을 부여
    - 모델이 적절하게 정규화되도록 자동으로 보장할 수 있다.
- RNN의 gating mechanism과도 연결되어 있다.
- mechanical 관점에서 이산화는 SSM의 순방향 패스에서 계산 그래프의 첫 번째 단계로 볼 수 있다.
- SSM의 대체 버전은 이산화 단계를 우회하고 $(\bar{A}, \bar{B})$를 직접 매개변수화 할 수 있고, 추론하기 더 쉬울 수 있다.

**계산**

매개변수가 $(\Delta, A, B, C) \mapsto (\bar{A}, \bar{B}, C)$로 변환된 후 모델은 linear recurrence 또는 global convolution으로 계산할 수 있다.

일반적으로 모델은 효율적인 병렬화 가능한 학습(전체 입력 sequence가 미리 표시되는 경우)를 위해 convolutional mode(3)을 사용.
효율적인 자동 회귀 추론(입력이 한 번에 하나의 timestep으로 표시되는 경우)를 위해 recurrent mode(2)로 전환

**Linear Time Invariance(LTI)**

식 (1)에서 (3)가지의 중요한 속성
- 모델의 역학이 시간이 지나도 일정하다.  
: $(\Delta, A, B, C)$와 결과적으로 $(\bar{A}, \bar{B})$도 모든 time-step에 대해 고정이다.
이를 Linear Time Invariance(LTI)라고 함
- recurrence와 convolution에 깊이 연결되어 있따.

논문의 저자는 LTI SSM을 linear recurrence(식 2a)또는 convolution(식 3b)와 동등하다고 생각하여 LTI를 이에 대해 포괄적인 용어로 사용

모든 구조화된 SSM(S4)은 3.3절의 근본적인 효율성 제약으로 인해 LTI(예: convolution으로 계산)이었다.
이 논문에서는 LTI 모델이 특정 유형의 데이터를 모델링하는데 근본적인 한계가 있다고 주장.
-> 효율성 병목 현상을 극복하면서 LTI 제약 조건을 제거

**구조와 차원**

구조화된 SSM(S4)은 효율적으로 계산하려면 A 행렬에 구조를 부여해야 하기 때문에 이러한 이름을 가짐

대각선(diagonal) 구조 형태
- 가장 인기 있는 형태
- 이 경우, $A \in \mathbb{R}^{N \times N}, B \in \mathbb{R}^{N \times 1}, C \in \mathbb{R}^{1 \times N}$ 행렬은 모두 N개의 숫자로 나타낼 수 있다.
- batch size: $B$, 길이가 $L$이고 채널이 $D$인 입력 sequence $x$에 대해 연산을 수행하기 위해 SSM은 각 채널에 독립적으로 적용됨  
이 경우, 총 은닉 상태에는 입력당 차원 $DN$이 있으며 sequence 길이에 대해 계산하면 O(BLDN) 시간과 메모리가 필요
-> 효율성 병목 현상의 근본 원인

**General State Space Models**

state space model이라는 용어는 매우 광범위한 의미를 갖는다
- 잠재 상태를 가진 모든 반복(recurrent) process의 개념을 의미
- Markov Decision Processes(MDP, 마르코프 결정 과정) (강화학습, Dynamic Causal Modeling(DCM, 동적 인과 모델링), 계산 신경 과학), 칼만 필터, Hidden Markov Model(HMM, 은닉 마르코프 모델), Linear Dynamical Systems(LDS, 선형 역학 시스템), 대규모의 recurrent(또는 convolutional) 딥러닝 models을 포함하여 다양한 분야에서 많은 이질적인 개념을 참조하는데 사용

이 논문에서 SSM은 structured SSM 또는 S4 모델을 지칭
linear-recurrence 또는 global-convolution 관점에 초점을 맞춘 모델과 파생 모델을 포함할 수도 있다.

**SSM Architecture**

SSM은 end-to-end NN 아키텍처에 통합할 수 있는 독립형 sequence transformations

가장 잘 알려진 SSM 중 일부
- Linear attention
    - 퇴화된 선형 SSM으로 볼 수 있는 recurrence를 포함하는 self-attention의 근사치
- H3
    - 이 recurrence를 S4를 소용하도록 일반화
    - SSM이 두 개의 gated connection 사이에 끼워져 있는 아키텍처로 볼 수 있다.
    - standard local convolution을 삽입  
    main SSM layer 이전에 standard local convolution을 끼워 넣어서 shift-SSM이라고 함
- Hyena
    - H3과 동일한 아키텍처를 사용하지만, S4 계층을 MLP-매개변수화된 전역 convolution으로 대체
- RetNet
    - 아키텍처에 추가 gate를 추가하고 더 간단한 SSM을 사용하여 convolution 대신 Multi-Head Attention(MHA)의 변형을 사용하여 병렬 처리 가능한 대체 계산 경로를 허용
- RWKV
    - linear attention 근사치인 attention-free Transformer를 기반으로 하는 언어 모델링을 위해 설계된 최신 RNN
    - 주요 WKV 메커니즘은 LTI recurrence와 관련이 있으며, 두 SSM의 비율로 볼 수 있다.

## 3. Selective State Space Models

synthetic tasks에서 직관을 사용하여 selection mechanism에 동기 부여
이 메커니즘을 state space model에 통합하는 방법 설명

- 시간에 따라 달라지는 SSMs은 convolution을 사용할 수 없으므로, 이를 효율적으로 계산하는 방법에 대한 기술적 과제 발생
- 최신 하드웨어의 메모리 계층 구조를 활용하는 hardware-aware 알고리즘으로 이를 극복
- attention 없거나 MLP block이 없는 간단한 SSM 아키텍처에 대해 설명
- 마지막으로, selection mechanism의 몇 가지 추가 속성에 대해 설명