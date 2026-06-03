# Attention is All you Need

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- LLM
- Transformers

---
url:
- [paper](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) (NeurIPS 2017)

---
- **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, Illia Polosukhin
- **Venue**: NeurIPS 2017

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)



---

## Abstract
- 지배적인 sequence 변환 모델은 encoder와 decoder을 포함하는 RNN 또는 CNN을 기반으로 함
- 최고의 성능을 내는 모델은 encoder와 decoder을 attention 메커니즘으로 연결
- Transformers
    - recurrence나 convolution을 배제하고 attention만을 기반으로 하는 네트워크
    - 두 가지 번역 과제
        - 품질 면에서 우수
        - 병렬화 가능
        - 훈련에 필요한 시간이 더 적음
        - WMT 2014 영어-독일어 번역 과제
            - 28.4 BELU
            - 기존 SOTA보다 2 BELU 이상 개선
        - WMT 2014 영어-프랑스어 번역 과제
            - 8개의 GPU에서 3.5일동안 훈련
            - 단일 모델 기준 SOTA인 41.0 BELU 달성


## 1. Introduction

**RNN**
- long short-term memory와 gated RNN은 언어 모델링과 기계 번역과 같은 sequence 모델링 및 변환 문제에서 SOTA로 자리잡음
- 입력과 출력 sequence의 symbol 위치에 따라 계산
    - 위치를 계산 시간의 step에 맞추면, 이전의 hidden state $h_{t - 1}$과 위치 $t$에 대한 입력의 함수로 hidden state $h_t$의 sequence를 생성
- 이러한 순차적인 특성은 예제 내에서 병렬화를 불가하게 함
- 긴 sequence 길이에서는 메모리 제약이 예제 간 배치 작업을 제한하여 병렬화가 중요해짐
- factorization trick과 조건부 계산을 통해 계산 효율성이 향상됨
    - 하지만 순차 계산의 근본적인 제약이 남아있음

**Attention 메커니즘**
- 강력한 sequence 모델링과 transduction model의 필수 요소가 됨
- 입력 또는 출력 sequence에서 거리 없이 의존성 모델링 가능
- 대부분의 경우 RNN과 함께 사용됨

**Transformer**
- 더 많은 병렬화를 가능하게 함
- 8개의 P100 GPU에서 12시간만 훈련 후 번역 품질 SOTA 달성 가능

## 2. Background

## 3. Model Architecture

![alt text](./images/Fig%201.png)
> **Figure 1. Transformer 아키텍처**

- 대부분의 경쟁 neural sequence transduction 모델은 encoder-decoder 구조를 갖는다.
    - encoder는 입력된 symbol sequence $(x_1, ..., x_n)$을 연속 표현 sequence $z = (z_1, ..., z_n)$으로 매핑
    - z가 주어지면, decoder는 한 번에 한 요소씩 기호 $(y_1, ..., y_m)$으로 출력 sequence 생성

### 3.1 Encoder and Decoder Stacks

**Encoder**
- N=6개의 동일한 층으로 구성된 스택으로 이뤄짐
- 각 층에는 두 개의 하위 층이 있음
    1. multi-head self-attention 메커니즘
    2. 단순한 position-wise fully connected feed-forward 네트워크
- 각 하위 층 주위에 residual connection을 사용
    - 이후 layer norm을 적용
- 즉, 각 하위 층의 출력은 $LayerNorm(x + Sublayer(x))$
- residual connection을 용이하게 하기 위해, 모든 sub-layer과 embedding layer은 $d_{model} = 512$ 차원의 출력을 생성

**Decoder**
- N = 6개의 동일한 층으로 구성된 stack
- 각 encoder 층의 두 하위 층 외에도, 세 번째 하위 층을 추가
    - encoder stack의 출력에 대해 multi-head attention 수행
- encoder와 유사하게, 각 하위 층 주의에 residual connection을 사용
    - 이후 layer norm 적용
- decoder stack의 self-atention sub-layer을 수정하여 위치가 이후 위치를 참조하지 못하도록 함
- masking과 출력 embedding이 한 위치씩 offset되는 사실을 결합하면, 위치 i에 대한 예측이 i보다 작은 위치에서 알려진 출력에만 의존하도록 보장됨

### 3.2 Attention

- Attention 함수는 query, key-value 쌍의 집합을 출력으로 매핑
- query, key, value는 모두 벡터
- 출력은 값들의 가중 합으로 계산됨
- 각 값에 할당된 가중치는 해당 key와의 query의 compability function에 의해 계산도미

#### 3.2.1 Scaled Dot-Production Attention

- 특정한 attention을 "Scaled Dot-Production Attention"이라고 부름(Figure 2)
    - 입력은 차원 $d_k$의 query와 key, 차원 $d_v$의 value로 구성됨
    - query와 모든 key의 내적을 계산하고, 각각을 $\sqrt{d_k}$로 나눈 뒤, softmax 함수를 적용하여 값에 대한 가중치를 얻음
- query 집합에 대해 attention function을 동시에 계산
    - 이를 행렬 Q에 패킹
    - key와 value 또한 각각 행렬 K, V에 패킹
    - 출력 행렬 계산:
    $$
    \displaystyle
    Attention(Q, K, V) = \rm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
    \tag{1}
    $$
- 가장 일반적으로 사용되는 두 가지 attention
    - additive attention
        - single hidden layer을 갖는 feed-forward 네트워크를 사용하여 호환성 함수 계산
    - dot-production attention
        - scaling 계수 $\frac{1}{\sqrt{d_k}}$를 제외하면 제안 알고리즘과 동일
    - 두 방법은 이론적 복잡도에서 유사하지만, dot-production attention은 고도로 최적화된 matrix multiplication 코드로 구현할 수 있기 때문에 빠르고 메모리 효율적
    - $d_k$가 작은 경우 두 메커니즘 성능이 유사하지만, $d_k$가 커지면 scaling 없이 addivite attention이 더 우수한 성능을 보임
        - $d_k$값이 큰 경우 dot production이 크게 증가하여 softmax 함수가 작은 기울기를 갖게 된다고 추측
        - 이를 완화하기 위해 dot product를 $\frac{1}{\sqrt{d_k}}$로 scaling
        - (두 벡터를 내적하게 되면, $d_k$개의 값들이 더하지게 되어 분산이 $d_k$로 커지게 됨. 분산을 1로 만들려면, d_k의 제곱근으로 나눠줘야 함)

### 3.2 Multi-Head Attention

- $d_{model}$-차원의 key, value, query를 사용하여 single attention을 수행하는 대신 Q, K, V 각각을 h번 서로 다른 학습된 linear projection으로 $d_k, d_k, d_v$ 차원으로 선형 투영
- 이렇게 투영된 Q, K, V에 대해 병렬로 attention 함수를 수행하여 $d_v$ 차원의 출력 값을 얻음
- 이 값들은 연결된 후 다시 한 번 투영되어 최종 값을 얻음(Fig 2 참조)
- 모델에서 서로 다른 위치에 있는 다양한 표현 하위 공간에서 정보를 동시에 참조할 수 있도록 함
    - single attention head에서는 평균화로 인해 이것이 억제됨

#### 3.2.3 Applications of Attention in our model