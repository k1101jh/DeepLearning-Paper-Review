# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Large Language Models
- State Space Models (SSM)
- Sequence Modeling

---
- **논문 제목**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **저자**: Albert Gu, Tri Dao
- **발표 연도**: arXiv 2023
- **공식 링크**: [arXiv](https://arxiv.org/abs/2312.00752), [GitHub](https://github.com/state-spaces/mamba)

---

## ⚡ 요약 (Summary)

### 1. 한 줄 요약
- 기존 구조화된 상태 공간 모델(SSM)의 시점 불변성(LTI) 한계를 극복하기 위해 입력 종속적 선택 메커니즘(Selection Mechanism)과 하드웨어 인식 병렬 알고리즘(Selective Scan)을 도입하고, Attention과 MLP 없이 단순화된 Mamba 아키텍처를 통해 시퀀스 길이에 선형 확장되면서 Transformer 수준의 성능과 5배 높은 추론 처리량을 달성하였다.

### 2. 핵심 아이디어
- **선택 메커니즘 (Selective SSM / S6)**: SSM의 주요 매개변수 $(\Delta, B, C)$를 입력 토큰 $x$의 함수로 매개변수화하여, 입력 내용(content)에 따라 시퀀스 길이를 따라 정보를 선택적으로 기억하거나 잊을 수 있도록 개선하였다.
- **하드웨어 인식 알고리즘 (Selective Scan)**: 입력 종속적 매개변수로 인해 글로벌 컨볼루션(Global Convolution)을 사용할 수 없게 된 한계를 극복하기 위해, GPU 메모리 계층 구조(HBM 과 SRAM 간의 IO 최소화)를 고려한 커널 퓨전(Kernel Fusion), 병렬 스캔(Parallel Scan), 그리고 재계산(Recomputation) 기법을 결합하여 고속 계산을 구현하였다.
- **단순화된 균일 아키텍처 (Mamba Architecture)**: 이전 SSM 아키텍처(H3 등)의 설계와 Transformer의 MLP 블록(SwiGLU 변형)을 단일 블록으로 통합하여 Attention이나 MLP 블록 없이 Mamba 블록만으로 엔드투엔드 모델을 구성하였다.

### 3. 문제 정의
- Transformer의 Self-Attention은 맥락(context) 내 정보를 라우팅하는 능력이 뛰어나지만, 시퀀스 길이가 길어짐에 따라 연산 및 메모리 복잡도가 2차(Quadratic, $O(L^2)$)로 증가하고 유한한 맥락 창 외부를 모델링하지 못한다.
- 기존 Subquadratic/Linear 모델들(Linear Attention, Gated Convolution, S4/H3 등 LTI SSM)은 선형 Scalability를 제공하지만, 매개변수가 시점 및 입력에 독립적인 LTI(Linear Time Invariance) 제약으로 인해 불필요한 정보를 선택적으로 필터링하고 맥락에 따라 기억하는 콘텐츠 기반 추론(Content-based Reasoning) 능력이 결여되어 텍스트와 같은 불연속적 정보 밀집 데이터에서 성능이 저하된다.

### 4. 실험 결과 및 의의
- **합성 과제 (Synthetic Tasks)**: Selective Copying 과제 및 Induction Heads 과제(최대 100만 토큰 외삽)를 완벽하게 해결하였다.
- **언어 모델링 (Language Modeling)**: 125M부터 2.8B 매개변수 스케일링 평가에서 최신 Transformer++ 레시피(LLaMa 구조 기반)와 대등하거나 우수한 성능을 보였으며, Mamba-3B는 Pythia-3B를 능가하고 Pythia-7B 수준에 도달함과 동시에 5배 빠른 Autoregressive 추론 처리량을 달성하였다.
- **DNA 및 오디오 (DNA & Audio)**: 백만 길이 시퀀스 pre-training 및 다운스트림 분류/생성 과제에서 기존 SOTA 모델들(HyenaDNA, SaShiMi)을 큰 폭으로 능가하였다.

### 5. 한계 및 의견
- continuous signal(오디오 파형 등) 데이터에서는 실수 매개변수화(Real-valued)보다 복소수 매개변수화(Complex-valued)가 더 유효한 경우가 있어 모달리티 특성에 따른 트레이드오프 검토가 필요하다.
- 7B~70B 이상 초대형 스케일에서의 Scaling behavior 및 Transformer 생태계의 다양한 다운스트림 기법(Fine-tuning, RLHF, Quantization 등)과의 호환성에 대한 지속적인 검증이 요구된다.

---

## 📖 Paper Review

## Abstract
- Foundation Model은 주로 Transformer 아키텍처와 core attention 모듈에 기반을 두고 있다.
- Linear attention, gated convolution, recurrent models, structured state space models(SSMs)와 같은 서브-쿼드라틱(subquadratic-time, $o(N^2)$) 아키텍처들이 긴 시퀀스에 대한 Transformer의 계산 비효율성을 해결하기 위해 개발되었으나, 언어(language)와 같은 중요한 모달리티에서는 attention만큼의 성능을 내지 못했다.
- 저자들은 이러한 기존 모델들의 핵심 약점이 **콘텐츠 기반 추론(content-based reasoning)**을 수행하지 못하는 능력 결여에 있음을 밝혀내고 다음과 같이 개선을 진행하였다:
    - **1) 선택 메커니즘**: SSM 매개변수를 입력의 함수로 설정함으로써 불연속(discrete) 모달리티에서의 약점을 해결하고, 모델이 현재 토큰에 따라 시퀀스 길이 차원에서 정보를 선택적으로 전파하거나 필터링(잊기)할 수 있게 하였다.
    - **2) 하드웨어 인식 병렬 알고리즘**: 입력 종속적 변경으로 인해 효율적인 글로벌 컨볼루션(convolution)을 사용할 수 없게 됨에도 불구하고, 순환 모드(recurrent mode)에서 메모리 계층 구조를 활용한 하드웨어 인식 병렬 알고리즘을 설계하였다.
    - **3) 단순화된 Mamba 아키텍처**: 선택적 SSM을 attention이나 MLP 블록 없이 단일화된 엔드투엔드 신경망 아키텍처에 통합하였다.
- Mamba는 다음과 같은 뛰어난 이점을 제공한다:
    - 빠른 추론(Transformer 대비 5배 높은 처리량)과 시퀀스 길이에 대한 선형 scaling을 제공하며, 실제 데이터에서 최대 백만(1M) 길이의 시퀀스까지 성능이 지속 향상된다.
    - 언어, 오디오, 유전체학(genomics) 등 다양한 모달리티에서 SOTA 성능을 달성하였다.
    - 언어 모델링에서 Mamba-3B 모델은 pre-training 및 downstream 평가 모두에서 동일 크기의 Transformer를 능가하고, 2배 크기의 Transformer(Pythia-7B 등)와 대등한 성능을 달성하였다.

---

## 1. Introduction
- 대규모 데이터로 Pre-train된 후 Downstream 작업에 맞게 조정되는 Foundation Model(FM)은 현대 머신러닝의 핵심 패러다임이다.
- 이러한 FM의 백본은 언어, 이미지, 음성, 오디오, 시계열, 유전체 등 다양한 도메인의 시퀀스를 처리하는 시퀀스 모델인 경우가 많다 (`Brown et al. (2020)`, `Dosovitskiy et al. (2020)`, `Oord et al. (2016)`, `Poli et al. (2023)`).
- 현대 FM은 주로 Transformer (`Vaswani et al. (2017)`)와 핵심 Self-attention 레이어 (`Bahdanau et al. (2015)`)를 기반으로 구축된다. Self-attention은 맥락 창(context window) 내에서 정보를 조밀하게 라우팅하는 뛰어난 능력을 지니지만, 두 가지 근본적인 단점을 갖는다:
    - 유한한 창(finite window) 외부의 정보를 모델링할 수 없다.
    - 창 길이에 대해 2차(quadratic) 계산 복잡도 및 메모리 스케일링을 갖는다.
- 이러한 단점을 극복하기 위해 많은 효율적인 Attention 변형 연구 (`Tay et al. (2022)`)가 시도되었으나, 그 과정에서 Attention을 효과적으로 만드는 본질적 특성이 훼손되는 경우가 많았다.
- 최근 **구조화된 상태 공간 시퀀스 모델(Structured State Space Sequence Models, SSMs)** (`Gu et al. (2021)`, `Gu et al. (2022)`)이 시퀀스 모델링을 위한 유망한 아키텍처 클래스로 부상하였다:
    - 고전 상태 공간 모델 (`Kalman (1960)`)에서 영감을 받아 RNN과 CNN의 속성을 결합하였다.
    - 시퀀스 길이에 대해 선형 또는 거의 선형(near-linear) 스케일링으로 recurrence 또는 convolution 형태 모두 효율적 연산이 가능하다.
    - 오디오, 비전 등 연속 신호(continuous signal) 데이터 영역 및 Long Range Arena (`Tay et al. (2021)`) 벤치마크에서 뛰어난 성과를 거두었다.
    - 그러나 텍스트와 같은 불연속적(discrete)이고 정보 밀도가 높은 데이터 모델링에서는 덜 효과적이었다.
- 본 연구에서는 시퀀스 길이에 선형 확장을 유지하면서 Transformer 수준의 표현력을 달성하는 새로운 **Selective State Space Model**을 제안한다.
- **Selection Mechanism (선택 메커니즘)**:
    - 기존 모델의 핵심 한계가 입력 종속적 방식(input-dependent manner)으로 데이터를 선택하는 능력(특정 입력에 집중하거나 무시하는 능력)의 부재임을 식별하였다.
    - Selective Copying 및 Induction Heads 과제 기반의 직관을 통해 SSM 매개변수를 입력에 따라 동적으로 매개변수화하는 단순한 선택 메커니즘을 설계하였다.
    - 이를 통해 모델은 관련 없는 정보를 필터링하고 관련 정보를 무기한 유지할 수 있다.
- **Hardware-aware Algorithm (하드웨어 인식 알고리즘)**:
    - 선택 메커니즘은 매개변수가 시점(time) 및 입력에 따라 바뀌게 만들어 효율적인 컨볼루션 연산을 불가능하게 만든다.
    - 저자들은 컨볼루션 대신 병렬 스캔(scan)으로 모델을 계산하되, GPU 메모리 계층 구조(HBM과 SRAM) 간의 IO 전송을 최소화하는 하드웨어 인식 알고리즘을 도입하였다.
    - 메모리에 전체 확장 상태(expanded state)를 실체화(materialize)하지 않는 방식을 통해 기존 컨볼루션 기반 SSM 대비 이론적으로나 실체적으로(A100 GPU 기준 최대 3배) 빠른 계산 속도를 달성한다.
- **Architecture (아키텍처)**:
    - 이전 SSM 아키텍처 H3 (`Dao et al. (2023)`)와 Transformer의 MLP 블록을 단일 블록으로 결합하여 단순하고 균일한(homogenous) Mamba 아키텍처를 설계하였다.
- Selective SSM 및 Mamba는 다음과 같은 핵심 특성을 갖는 Fully Recurrent 모델이다:
    - **1) High Quality**: Selectivity를 통해 텍스트, 유전체 등 정보 밀도가 높은 모달리티에서 강력한 성능 발휘.
    - **2) Fast Training & Inference**: 학습 시 계산/메모리가 시퀀스 길이에 선형 비례하며, 추론 시 KV 캐시가 필요 없어 Step당 일정한 시간만 소요.
    - **3) Long Context**: 뛰어난 품질과 효율성 덕분에 최대 1M 길이 시퀀스까지 실제 성능 향상 지속.

---

## 2. State Space Models
- Structured State Space Sequence Models (S4)는 implicit latent state $h(t) \in \mathbb{R}^N$을 통하여 1차원 신호 $x(t) \in \mathbb{R} \mapsto y(t) \in \mathbb{R}$을 매핑하는 연속 시스템(Continuous System)에서 영감을 얻었다:

$$
\begin{aligned}
h'(t) &= Ah(t) + Bx(t) &(1a) \\
y(t) &= Ch(t) &(1b)
\end{aligned}
$$

- S4 모델은 4개의 매개변수 $(\Delta, A, B, C)$로 정의되며, continuous parameters $(\Delta, A, B)$를 이산 매개변수 $(\bar{A}, \bar{B})$로 변환하는 이산화 단계와 시퀀스 변환 단계로 구성된다.

$$
\begin{aligned}
h_t &= \bar{A}h_{t-1} + \bar{B}x_t &(2a) \quad &\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, \dots, C\bar{A}^k\bar{B}, \dots) &(3a) \\
y_t &= Ch_t &(2b) \quad &y = x * \bar{K} &(3b)
\end{aligned}
$$

### Discretization (이산화)
- 고정 공식 $\bar{A} = f_A(\Delta, A)$, $\bar{B} = f_B(\Delta, A, B)$를 사용하여 이산화를 수행한다. 일반적으로 Zero-Order Hold (ZOH) 규칙이 사용된다:

$$
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B \quad (4)
$$

- 이산화는 연속 시간 시스템과의 연결을 통해 해상도 불변성(resolution invariance)을 부여하고 정규화(normalization)를 보장하며, RNN의 게이팅 메커니즘과도 밀접하게 연결된다.

### Computation (계산 방식)
- 매개변수가 $(\bar{A}, \bar{B}, C)$로 변환되면, 모델은 **Linear Recurrence (식 2)** 또는 **Global Convolution (식 3)**의 두 가지 이중 형태로 연산이 가능하다.
- 전체 시퀀스가 주어진 학습 시에는 병렬 처리가 용이한 Convolutional Mode (3)을 사용하고, Autoregressive 추론 시에는 Step별 계산이 빠른 Recurrent Mode (2)로 전환한다.

### Linear Time Invariance (LTI)
- 식 (1)~(3)의 결정적인 속성은 모델의 역학(dynamics)이 시간에 따라 일정하다는 점이다. 즉 $(\Delta, A, B, C)$ 및 $(\bar{A}, \bar{B})$가 모든 시점(time-step)에서 고정된다. 이를 **Linear Time Invariance (LTI)**라 칭한다.
- 모든 기존 구조화된 SSM(S4 등)은 효율성 제약으로 인해 LTI 모델(즉, 글로벌 컨볼루션 계산 가능)로 제한되었으나, LTI 속성은 특정 유형의 데이터를 모델링하는 데 근본적인 한계를 지닌다.

### Structure and Dimensions (구조와 차원)
- $A$ 행렬에 구조를 부여해야 연산이 효율적이다. 가장 대중적인 구조는 대각선(Diagonal) 형태이며, 이 경우 $A \in \mathbb{R}^{N \times N}, B \in \mathbb{R}^{N \times 1}, C \in \mathbb{R}^{1 \times N}$은 $N$개 숫자로 표현된다.
- Batch size $B$, 시퀀스 길이 $L$, 채널 수 $D$인 입력 $x$에 대해 SSM은 각 채널별로 독립 적용된다. 입력당 은닉 상태 차원은 $DN$이 되며, 시퀀스에 걸쳐 연산 시 $O(BLDN)$의 시간과 메모리가 요구되어 효율성 병목이 발생한다.

### General State Space Models
- State Space Model은 은닉 상태를 갖는 모든 순환 프로세스(MDP, 칼만 필터, HMM, LDS 등)를 통칭하는 광범위한 용어이다. 본 논문에서 SSM은 Structured SSM (S4 및 그 변형)에 한정하여 지칭한다.

### SSM Architectures
- SSM은 독립적인 Sequence Transformation 모듈로서 딥러닝 아키텍처에 내재될 수 있다:
    - **Linear Attention**: Self-attention의 이산 순환 근사 형태.
    - **H3**: S4를 두 개의 게이트 연결 사이에 배치하고 메인 SSM 전에 로컬 컨볼루션(Shift-SSM)을 추가한 아키텍처.
    - **Hyena**: H3와 동일하나 S4 계층을 MLP 매개변수화 전역 컨볼루션으로 대체.
    - **RetNet**: 게이트를 추가하고 MHA 변형 병렬 경로를 갖는 아키텍처.
    - **RWKV**: Linear attention 근사 기반의 아키텍처로 WKV 메커니즘이 LTI recurrence에 해당.

> **Figure 1. Selective State Space Model Overview.** 기존 구조화된 SSM은 각 채널($D$)을 고차원 은닉 상태($N$)로 독립 매핑하며, LTI 제약(시간에 대해 고정된 $\Delta, A, B, C$)을 통해 컨볼루션 형태 경로를 사용하여 $DN$ 차원의 상태 확장을 피했다. 반면 Mamba의 선택 메커니즘은 입력 종속 동역학을 복원하며, GPU 메모리 계층(HBM/SRAM)을 고려한 하드웨어 인식 알고리즘을 통해 확장 상태를 필요한 순간에만 구현한다.

---

## 3. Selective State Space Models

### 3.1 Motivation: Selection as a Means of Compression (동기: 압축 수단으로서의 선택)
- 시퀀스 모델링의 근본 과제는 맥락(context)을 더 작은 상태(state)로 효율적으로 압축하는 것이다:
    - **Attention**: 맥락을 전혀 압축하지 않는다 (KV 캐시를 모두 유지). 결과적으로 추론 시 느리고 학습 시 2차(quadratic) 계산을 유도한다.
    - **Recurrent Models**: 유한한 상태(finite state)를 유지하므로 추론이 상수 시간, 학습이 선형 시간이지만, 상태가 맥락을 얼마나 잘 압축했는지에 따라 성능이 제한된다.
- 합성 과제(Synthetic Tasks)를 통해 LTI 모델의 한계를 분석한다:
    - **Selective Copying**: 기억해야 할 토큰의 위치를 무작위화한 과제로, 관련 토큰을 기억하고 관련 없는 토큰을 필터링하는 **콘텐츠 기반 추론**이 필요하다.
    - **Induction Heads**: 이전 문맥의 패턴(예: "Harry Potter")을 기반으로 다음 토큰을 예측하는 연상 기억(associative recall) 과제이다.
- LTI 모델은 시점 불변 동역학으로 인해 문맥에 따른 선택적 정보 전달이 불가능하다. 전역 컨볼루션은 단순 Copying은 해결하지만 토큰 간 간격이 가변적인 Selective Copying은 해결하지 못한다.

> **Figure 2. Synthetic Tasks Comparison.** (좌) 표준 Copying 과제는 고정된 간격을 가져 LTI 모델로도 쉽게 해결된다. (우상) Selective Copying 과제는 간격이 무작위이며 입력 내용에 따라 선택적으로 기억/무시할 수 있는 시간 가변(Time-varying) 모델이 필요하다. (우하) Induction Heads 과제는 맥락에 기반한 연상 기억을 요구하는 LLM의 핵심 기능이다.

### 3.2 Improving SSMs with Selection (선택 메커니즘을 통한 SSM 개선)
- SSM의 매개변수인 $(\Delta, B, C)$를 입력 $x$의 함수(Linear projection)로 정의하여 시점 가변(Time-varying) 매개변수화를 적용한다:
    - $s_B(x) = \text{Linear}_N(x)$
    - $s_C(x) = \text{Linear}_N(x)$
    - $s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$
    - $\tau_\Delta = \text{softplus}$

```
Algorithm 1: Standard SSM (S4)
Input: x (B, L, D)
Output: y (B, L, D)
1: A: (D, N) <- Parameter
2: B: (D, N) <- Parameter
3: C: (D, N) <- Parameter
4: Δ: (D)    <- τ_Δ(Parameter)
5: Ā, B̄: (D, N) <- discretize(Δ, A, B)
6: y <- SSM(Ā, B̄, C)(x)  # Time-invariant: recurrence or convolution
7: return y
```

```
Algorithm 2: SSM + Selection (S6)
Input: x (B, L, D)
Output: y (B, L, D)
1: A: (D, N) <- Parameter
2: B: (B, L, N) <- s_B(x)
3: C: (B, L, N) <- s_C(x)
4: Δ: (B, L, D) <- τ_Δ(Parameter + s_Δ(x))
5: Ā, B̄: (B, L, D, N) <- discretize(Δ, A, B)
6: y <- SSM(Ā, B̄, C)(x)  # Time-varying: recurrence (scan) only
7: return y
```

### 3.3 Efficient Implementation of Selective SSMs (선택적 SSM의 효율적 구현)

#### 3.3.1 Motivation of Prior Models
- 순환 연산은 $O(BLDN)$ FLOPs를 사용하고 전역 컨볼루션 연산은 $O(BLD \log L)$ FLOPs를 사용한다. 은닉 상태 차원 $N$이 크지 않으면 순환 연산의 FLOPs가 실제로는 더 낮다.
- 기존 LTI 모델은 $N$을 확대하면서 순환-컨볼루션 이중성을 이용해 메모리 IO를 줄였으나, 선택적 SSM은 컨볼루션을 활용할 수 없다.

#### 3.3.2 Overview of Selective Scan: Hardware-Aware State Expansion
- **Kernel Fusion**: 느린 GPU HBM에 $h$ 상태 $(B, L, D, N)$를 구체화(materialize)하지 않고, SSM 매개변수 $(\Delta, A, B, C)$를 HBM에서 빠르고 작은 SRAM으로 직접 로드하여 SRAM 내부에서 이산화 및 순환 연산(Scan)을 실행한 후 결과 $y(B, L, D)$만 HBM에 다시 쓴다.
- **Parallel Scan**: 순차적 재귀 계산 대신 병렬 스캔 알고리즘 (`Blelloch (1990)`)을 사용하여 GPU 상에서 일정한 병렬성을 확보한다.
- **Recomputation**: 역전파(backward pass) 시 중간 상태 $h$를 HBM에 저장하지 않고, 역전파 계산 시 SRAM으로 입력을 재로드하여 상태를 재계산(recompute)함으로써 메모리 소모를 최적화한다. (FlashAttention과 동일한 메모리 요구량)

### 3.4 A Simplified SSM Architecture (단순화된 SSM 아키텍처)
- 기존 H3 아키텍처와 Transformer의 Gated MLP(SwiGLU) 구조를 결합하여 Attention 및 MLP 블록 없이 단일 Mamba 블록을 균일하게(homogenously) 쌓아 올린 아키텍처를 제안한다.
- 확장 계수 $E=2$를 적용하여 파라미터 수를 조정하고, 활성화 함수로 SiLU/Swish를 사용하며 LayerNorm을 normalization에 활용한다.

> **Figure 3. Architecture Comparison.** H3 블록과 Gated MLP 블록을 결합하여 Mamba 아키텍처 블록을 구성한다. Mamba는 첫 번째 곱셈 게이트 대신 활성화 함수를 적용하고 메인 브랜치에 SSM 계층을 추가하였다.

### 3.5 Properties of Selection Mechanisms (선택 메커니즘의 특성)

#### 3.5.1 Connection to Gating Mechanisms
- **Theorem 1**: $N=1, A=-1, B=1, s_\Delta = \text{Linear}(x), \tau_\Delta = \text{softplus}$ 일 때, 선택적 SSM 순환식(Algorithm 2)은 전통적인 RNN 게이팅 메커니즘 형태와 수학적으로 일치한다:

$$
g_t = \sigma(\text{Linear}(x_t)), \quad h_t = (1 - g_t)h_{t-1} + g_t x_t \quad (5)
$$

- SSM의 이산화 과정에서의 $\Delta$는 RNN 게이팅 메커니즘의 원리를 제공하는 일반화된 패러다임으로 해석할 수 있다.

#### 3.5.2 Interpretation of Selection Mechanisms
- **Variable Spacing (가변 간격)**: 중요한 입력 사이에 존재하는 불필요한 노이즈 토큰(예: 텍스트의 추임새 "um")을 $g_t \to 0$을 통해 선택적으로 무시할 수 있다.
- **Filtering Context (맥락 필터링)**: 기존 LTI 모델은 무관한 맥락을 필터링하지 못해 시퀀스가 길어져도 성능이 정체/하락하는 반면, 선택적 모델은 상태를 리셋하여 무관한 역사를 제거하므로 시퀀스 길이에 따라 성능이 단조 증가한다.
- **Boundary Resetting (경계 리셋)**: 여러 독립 문서가 병합된 환경에서 경계 조건($g_t \to 1$)을 만나면 이전 상태를 완전히 리셋하여 정보의 이탈(bleed)을 방지한다.
- **Interpretation of $\Delta, A, B, C$**:
    - $\Delta$: 현재 입력 $x_t$에 얼마나 집중하거나 무시할지 제어한다 ($\Delta \to \infty$는 현재 입력 집중 및 상태 리셋, $\Delta \to 0$은 무시 및 이전 상태 유지).
    - $A$: discretization을 통해 $\Delta$와 상호작용하므로 $\Delta$의 선택성만으로도 충분하다.
    - $B, C$: 입력 $x_t$를 은닉 상태 $h_t$로 얼마나 들여보낼지, 은닉 상태를 출력 $y_t$로 얼마나 내보낼지 정교하게 제어한다.

### 3.6 Additional Model Details
- **Real vs. Complex**: 기존 SSM은 복소수(complex) 상태를 주로 사용했으나, 텍스트 등 불연속 데이터에는 실수(real-valued) 매개변수가 더 뛰어난 효율과 성능을 보임을 확인하여 기본값으로 실수를 사용한다.
- **Initialization**: 실수 상태에 대해서는 S4D-Real, 복소수 상태에 대해서는 S4D-Lin 초기화를 기본으로 채택한다.
- **Parameterization of $\Delta$**: $s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$로 정의되며 low-rank projection 형태로 확장 가능하다.

---

## 4. Empirical Evaluation

### 4.1 Synthetic Tasks

#### 4.1.1 Selective Copying
- 토큰 간 간격을 무작위화한 과제로 LTI 모델(S4 등)이나 아키텍처 게이팅(H3, Hyena)은 부분적 개선만 가능하지만, Selective SSM(S6 계층)을 탑재한 Mamba는 99.8%의 정확도로 완전히 해결하였다.

> **Table 1. Selective Copying Accuracy.** 아키텍처 블록과 내부 시퀀스 계층 결합에 따른 정확도 비교. S6 선택적 계층 적용 시 정확도가 99% 이상으로 비약적 향상된다.

#### 4.1.2 Induction Heads
- 훈련 길이(256) 대비 테스트 시 시퀀스 길이를 최대 1,048,576(1M, 4000배)까지 확장 평가하였다.

> **Table 2. Induction Heads Extrapolation Accuracy.** Mamba(selective SSM)는 1M 토큰 길이까지 100% 완벽한 외삽 능력을 보여주지만 다른 Attention 및 기존 SSM 기법들은 길이가 늘어나면 실패한다.

### 4.2 Language Modeling

#### 4.2.1 Scaling Laws
- The Pile 데이터셋에서 125M부터 1.3B 매개변수 범위로 Chinchilla 프로토콜 하에 평가를 진행하였다.

> **Figure 4. Scaling Laws on The Pile.** Mamba는 기존의 모든 Attention-free 모델을 능가하며, 최신 강력한 Transformer++ 레시피와 대등한 성능을 달성하는 최초의 선형 시간 시퀀스 모델이다.

#### 4.2.2 Downstream Evaluations
- Zero-shot 다운스트림 평가 결과, 동일한 데이터셋 및 토크나이저로 훈련된 동급 Pythia 및 RWKV 모델들을 전 영역에서 능가하였다.

> **Table 3. Zero-shot Downstream Evaluations.** 동일 파라미터 체급에서 Mamba는 모든 단일 평가 지표에서 Best-in-class 성능을 나타내며, 2배 크기의 모델(Pythia-7B 등)과 대등한 성과를 거둔다.

### 4.3 DNA Modeling

#### 4.3.1 Scaling: Model Size
- HG38 유전체 데이터셋 pretraining에서 모델 크기를 200K에서 40M까지 확장하며 평가하였다. Mamba는 HyenaDNA 및 Transformer++ 대비 3~4배 적은 파라미터로 동일 Perplexity에 도달하였다.

#### 4.3.2 Scaling: Context Length
- 시퀀스 길이를 1024에서 1,048,576(1M)까지 확장 시, Mamba는 컨텍스트가 길어질수록 Perplexity가 지속적으로 개선된 반면 HyenaDNA(LTI 모델)는 컨텍스트가 길어지면 오히려 성능이 악화되었다.

> **Figure 5. DNA Scaling Laws on HG38.** (좌) 모델 크기 증가 시 Mamba가 가장 우수한 Scaling을 보임. (우) 시퀀스 길이 증가 시 Mamba는 1M 토큰까지 단조적 성능 향상을 보여주나 HyenaDNA는 저하된다.

#### 4.3.3 Synthetic Species Classification
- DNA 99%를 공유하는 영장류 5종(인간, 침팬지, 고릴라, 오랑우탄, 보노보) 분류 과제에서 Fine-tuning 정확도를 평가하였다.

> **Figure 6. Great Apes DNA Classification Accuracy.** 시퀀스 길이가 1M에 도달함에 따라 Mamba 기반 fine-tuning 정확도가 독보적으로 상승한다.

### 4.4 Audio Modeling and Generation

#### 4.4.1 Long-Context Autoregressive Autoregressive Pretraining
- YouTubeMix 피아노 오디오 데이터셋에서 시퀀스 길이를 8,192에서 1M까지 확장 평가하였다. 복소수(complex) 파라미터화를 적용한 Mamba는 기존 SaShiMi(S4+MLP) 대비 뛰어난 bpb(bits per byte) 성과를 달성하였다.

> **Figure 7. Audio Pretraining (YouTubeMix).** 1분 길이(약 1M 시퀀스)에 이르는 long context 오디오 pretraining에서 Mamba가 SaShiMi 대비 우수한 bpb 성능을 보인다.

#### 4.4.2 Autoregressive Speech Generation
- SC09 음성 생성 데이터셋 평가에서 소형 Mamba 모델이 기존 GAN 및 Diffusion 모델 기반 SOTA를 능가하였고, 큰 모델은 품질 지표(FID, IS 등)를 획기적으로 개선하였다.

> **Table 4. SC09 Automated Metrics.** 무조건부 음성 생성 평가에서 Mamba는 기존 Autoregressive 및 GAN/Diffusion 기법 대비 월등히 낮은 FID와 높은 IS/mIS를 기록하였다.

> **Table 5. SC09 Model Ablations.** U-Net 아키텍처의 Outer 및 Center 블록에 Mamba를 적용 시 S4+MLP 및 MHA+MLP 조합보다 항상 우수한 품질을 보여준다.

### 4.5 Speed and Memory Benchmarks
- **Training Speed**: Mamba의 융합된 Selective Scan 커널은 PyTorch 기본 스캔 구현보다 20~40배 빠르며, 2K 시퀀스 이상부터 FlashAttention-2보다 빠른 계산 속도를 나타낸다.
- **Inference Throughput**: KV 캐시가 필요 없으므로 대용량 배치를 사용할 수 있어 Transformer 대비 4~5배 높은 추론 처리량을 제공한다.

> **Figure 8. Efficiency Benchmarks.** (좌) 학습 시 융합 스캔 커널은 PyTorch 스캔 대비 40배 빠름. (우) 추론 시 Mamba-6.9B는 5배 작은 Transformer-1.3B보다 높은 처리량을 보여준다.

### 4.6 Model Ablations

#### 4.6.1 Architecture
- LTI 모델 간에는 복소수/실수 파라미터화의 차이가 적으나, Selective SSM(S6)을 적용할 경우 아키텍처 종류와 상관없이 Perplexity가 대폭 개선(10.5 -> 8.6)되었다.

> **Table 6. Ablations: Architecture and SSM layer.** H3 및 Mamba 블록 모두 S6 선택적 계층을 장착할 때 Perplexity가 압도적으로 개선된다.

#### 4.6.2 Selective SSM
- 매개변수 선택성 조합 중 $\Delta$의 선택성이 가장 결정적인 역할을 수행하며, $(\Delta, B, C)$를 모두 선택적으로 구성할 때 시너지가 극대화된다.
- SSM 상태 차원 $N$을 확대할 때, $B, C$가 선택적일 때만 파라미터 증가(1% 미만) 대비 Perplexity가 1.0 이상 대폭 개선되는 효과를 거두었다.

> **Table 7. Ablations: Selective parameters.** $\Delta, B, C$의 선택성 여부에 따른 Perplexity 변화. $\Delta$의 선택성이 가장 핵심적이다.

> **Table 8. Ablations: Parameterization of A.** 선택적 SSM 환경에서는 실수 기반 S4D-Real 및 무작위 초기화가 복소수 기반 초기화보다 더 좋은 performance를 나타낸다.

> **Table 9. Ablations: Expressivity of $\Delta$.** $\Delta$ 사상 차원을 1차원에서 64차원까지 확장 시 적은 파라미터 추가로 성능이 꾸준히 향상된다.

> **Table 10. Ablations: SSM state dimension N.** $B, C$가 선택적(Bottom)일 때 상태 차원 $N$을 늘리면 계산 비용 증가 없이 Perplexity가 비약적으로 개선된다.

---

## 5. Discussion
- **No Free Lunch: Continuous-Discrete Spectrum**:
    - SSM은 본래 연속 시간 시스템의 이산화에서 발원하여 연속 신호(오디오 등)에 강한 귀납적 편향(inductive bias)을 가진다.
    - 저자들의 선택 메커니즘은 텍스트, DNA 등 불연속 모달리티에서의 약점을 제거하지만, 오디오 실험 결과에서 보듯 LTI SSM이 강점을 보이던 일부 연속 신호 분야에서는 트레이드오프가 존재할 수 있다.
- **Downstream Affordances**:
    - Fine-tuning, Prompting, In-context Learning, RLHF, Quantization 등 Transformer 중심의 풍부한 다운스트림 생태계 속성이 SSM에서도 동일하게 발휘될 수 있는지 추가적인 연구가 요구된다.
- **Scaling**:
    - 본 논문의 실험은 주로 3B 이하 모델에 국한되어 있으므로, 7B, 70B 이상의 초대형 스케일에서도 Mamba의 경쟁력이 지속 유지되는지에 대한 지속적인 검증이 필요하다.

---

## 6. Conclusion
- 구조화된 상태 공간 모델에 **선택 메커니즘(Selection Mechanism)**을 도입하여, 시퀀스 길이에 선형적으로 확장되면서 맥락 종속적 추론을 수행하는 Mamba 모델을 개발하였다.
- Mamba는 텍스트, 유전체, 오디오 등 다채로운 모달리티에서 Transformer 이상의 SOTA 성과를 보여주었으며, 향후 foundation model 백본의 강력한 대안으로 자리매김할 가능성을 입증하였다.