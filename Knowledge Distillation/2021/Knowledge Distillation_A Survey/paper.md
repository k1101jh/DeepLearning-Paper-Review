# Knowledge Distillation: A Survey

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Knowledge Distillation
- Model Compression
- Survey

---
url:
- [paper](https://link.springer.com/article/10.1007/s11263-021-01453-z) (IJCV 2021)
- [arXiv](https://arxiv.org/abs/2006.05525)

---
- **Authors**: Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao
- **Venue**: IJCV 2021

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Knowledge](#2-knowledge)
  - [2.1 Response-Based Knowledge](#21-response-based-knowledge)
  - [2.2 Feature-Based Knowledge](#22-feature-based-knowledge)
  - [2.3 Relation-Based Knowledge](#23-relation-based-knowledge)
- [3. Distillation Schemes](#3-distillation-schemes)
  - [3.1 Offline Distillation](#31-offline-distillation)
  - [3.2 Online Distillation](#32-online-distillation)
  - [3.3 Self-Distillation](#33-self-distillation)
- [4. Teacher-Student Architectures](#4-teacher-student-architectures)
  - [4.1 Teacher-Student Model Setups](#41-teacher-student-model-setups-구조적-설정-분류)
  - [4.2 Model Capacity Gap](#42-model-capacity-gap-용량-차이-완화-기법)
  - [4.3 Neural Architecture Search (NAS)](#43-neural-architecture-search-nas-기반-아키텍처-탐색)
- [5. Distillation Algorithms](#5-distillation-algorithms)
  - [5.1 Adversarial Distillation](#51-adversarial-distillation)
  - [5.2 Multi-Teacher Distillation](#52-multi-teacher-distillation)
  - [5.3 Cross-Modal Distillation](#53-cross-modal-distillation)
  - [5.4 Graph-Based Distillation](#54-graph-based-distillation)
  - [5.5 Attention-Based Distillation](#55-attention-based-distillation)
  - [5.6 Data-Free Distillation](#56-data-free-distillation)
  - [5.7 Quantized Distillation](#57-quantized-distillation)
  - [5.8 Lifelong Distillation](#58-lifelong-distillation)
  - [5.9 NAS-Based Distillation](#59-nas-based-distillation)
- [6. Performance Comparison](#6-performance-comparison)
- [7. Applications](#7-applications)
  - [7.1 Visual Recognition](#71-visual-recognition)
  - [7.2 Natural Language Processing (NLP)](#72-natural-language-processing-nlp)
  - [7.3 Speech Recognition](#73-speech-recognition)
  - [7.4 Other Applications](#74-other-applications)
- [8. Challenges and Future Directions](#8-challenges-and-future-directions)
- [9. Conclusion](#9-conclusion)



---

## Abstract

딥 러닝의 성공은 주로 대규모 데이터를 인코딩하고 수십억 개의 모델 매개변수를 조작할 수 있는 확장성 덕분
-> 자원이 제한된 장치에 배포하는 것은 높은 계산 복잡성과 큰 저장공간을 요구

Knowledge Distillation
- 대규모 교사 모델로 소규모 학생 모델을 효과적으로 학습

## 1. Instroduction

대규모 딥러닝 모델은 성공을 거두었지만 엄청난 컴퓨팅 복잡성과 방대한 저장공간을 요규하여 자원이 제한된 장치에 배포하기 힘들다.

효율적인 딥 러닝 모델을 만들기 위한 최근 연구
1. MobileNets와 같이 depthwise 분리 가능한 convolution을 포함한 효율적으로 block을 쌓는 방법
2. 모델 압축과 가속화 기법
    - 매개변수 가지치기(pruning) 과 sharing
        - 성능에 중요한 영향을 미치지 않는 불필요한 매개변수를 제거
        - Model quantization, Model binarization, structural matrices, parameter sharing 등이 있다.
    - Low-rank factorization
        - 행렬 및 tensor decomposition을 사용하여 Neural Network(NN)의 여분 매개변수를 파악
    - Transferred compact convolutional filters
        - 컨볼루션 필터를 transferring 또는 압축하여 불필요한 매개변수 제거
    - Knowledge Distillation(KD)
        - 큰 네트워크에서 지식을 추출하여 작은 네트워크로 전달

Knowledge Distillation
- Bucilua et al. (2006)
    - 큰 모델 또는 모델의 앙상블의 정보를 작은 모델로 전달
- Urner et al. (2011)
    - semi-supervised learning을 위해 fully-supervised teacher model과 student model 간의 지식 전달을 도입
- Hinton et al. (2015)
    - 대형 모델에서 작은 모델을 학습하는 것을 "Knowledge Distillation"으로 공식적으로 대중화

주요 아이디어
- 경쟁력 있거나 우수한 성과를 얻기 위해 학생 모델이 교사 모델을 모방

주요 문제
- 대규모 교사 모델에서 소규모 학생 모델로 지식을 전달

KD의 세 가지 구성
- Knowledge
- Distillation algorithm
- Teacher-student architecture

![alt text](./images/Fig%201.png)

> **Figure 1. Knowledge distillation을 위한 일반적인 teacher-student 프레임워크**

- label이 지정되지 않은 데이터를 사용하여 교사 모델에서 학생 모델로의 지식 전달이 PAC learnable임을 입증
- Phuong and Lampert (2019)
    - deep linear classifiers 시나리오에서 learning distilled 학생 네트워크의 빠른 수렴으로 일반화에 대한 이론적 정당성을 얻음
    - 학생이 무엇을 얼마나 빨리 배우는지에 대한 답을 제시하고 distillation의 성공을 결정하는 요인을 드러냄
    - 성공적인 증류는 데이터 기하학, 증류 목표의 최적화 편향 및 student classifier의 강력한 단조성에 달려 있음
- Cheng et al. (2020)
    - 지식 증류를 설명하기 위해 심층 신경망의 중간 layer에서 시각적 개념 추출을 정량화
- Ji & Zhu. 2020
    - risk bound, data efficiency, 불완전한 교사의 관점에서 넓은 Neural Network에서 지식 증류를 이론적으로 설명
- Cho & Hariharan. 2019
    - 지식 증류의 효능을 세부적으로 분석
    - 모델 크기가 크다고 더 나은 교사가 되지는 않음
        - 모델의 용량 차이 때문
    - 증류가 student 학습에 부정적인 영향을 미침
    - 다음 내용은 다루지 않음
        - teacher과 student 사이의 knowledge, distillation, 상호 영향에 대한 다양한 형태의 knowledge distillation
- Tang et al. 2020
    - label smoothing, teacher의 정확도 평가, 최적의 출력층 구조에 대한 prior을 얻기 위해 연구

모델 압축을 위한 Knowledge distillation
- 인간이 배우는 방식과 비슷
- 확장 방법
    - teacher-student learning
    - mutual learning
    - assistant learning
    - lifelong learning
    - self-learning
- 확장 방법들은 대부분 Deep neural network를 압축하는데 집중
- 생성된 가벼운 네트워크들은 visual recognition, speech recognition, natural language processing과 같은 application에서 쉽게 사용될 수 있음
- 적대적 공격, 데이터 증강, 데이터 프라이버시와 보안같은 작업으로 확장 가능
- 데이터셋 증류
    - knowledge transfer 아이디어는 핵심 데이터를 압축하는 데 적용
    - 큰 데이터셋에서 작은 데이터셋으로 knowledge를 전이시켜 모델 학습의 부담을 줄임

![alt text](./images/Fig%202.png)

> **Figure 2. knowledge distillation의 개략적 구조와 인접 section 간의 관계**

**논문의 기여**
1. 여러 전형적인 knowledge, distillation 방법 및 아키텍처를 포함한 지식 증류에 대한 개요 제공
2. 알고리즘과 다양한 실제 시나리오에 대한 적용을 포함한 최근 knowledge distillation의 진전 검토
3. 다양한 knowledge transfer 관점을 기반으로 knowledge distillation에 대한 통찰 제공 및 장애물을 다룸
    - knowledge 종류
    - 학습 방식
    - 증류 알고리즘과 구조
    - applications
- 유사한 survey 연구 존재(Wand and Yoon. 2020)
    - vision 학습을 위한 teacher-student 학습의 다른 관점에서의 종합적인 인전과 도전 과제를 제시
    - 이 논문에서는 knowledge 유형, distillation 방식, distillation 알고리즘, 성능 비교 및 다양한 응용 분야에 대해 다룸

## 2. Knowledge

![alt text](./images/Fig%203.png)

> **Figure 3. deep teacher network에서의 response-based knowledge, feature-based knowledge, relation-based knowledge**

- 지식의 종류, 증류 전략, teacher-student 구조가 학생의 학습에 중요한 역할을 함
- 일반적인 knowledge distillation은 큰 deep model의 logit을 teacher knowledge로 사용
- 중간 계층의 활성화 값, neurons, sample pair 간의 관계에는 teacher 모델이 학습한 풍부한 정보가 담겨있음
- teacher model의 parameter(혹은 layer 간 연결)도 또 다른 knowledge를 갖고 있음
- 다양한 형태의 지식
    - response-based knowledge
    - feature-based knowledge
    - relation-based knowledge

![alt text](./images/Table%201.png)
> **Table 1. 지식 증류에서의 지식(Knowledge) 카테고리 비교 요약 (Response-based, Feature-based, Relation-based)**



### 2.1 Response-based knowledge


![alt text](./images/Fig%204.png)
> **Figure 4. 일반적인 response-based knowledge distillation**

![alt text](./images/Fig%205.png)
> **Figure 5. benchmark knowledge distillation의 구체적인 아키텍처**

- teacher model의 마지막 출력층의 neural 응답을 의미
- teacher model의 최종 예측을 직접 따라함
- 간단하면서도 모델 압축에 효과적
- 마지막 fully connected layer의 출력으로 logit vector $z$가 주어지면, 증류 loss는 다음과 같음
$$
L_{ResD}(z_t, z_s) = L_R(z_t, z_s)
\tag{1}
$$
> $L_R(\cdot)$: logits의 divergence loss
> $z_t, z_s$: teacher와 student의 logits
- 여러 종류의 모델 예측에 활용 가능
    - 객체 탐지 작업
        - 응답이 logit과 함께 bounding box의 offset을 포함 가능
    - semantic landmark localization 작업
        - teacher 모델의 응답이 각 landmark에 대한 heatmap 포함 가능
- response-based knowledge가 실제 label 정보를 conditional target으로 활용하는 방향으로 탐구되고 있음
- soft targets
    - 가장 인기 있는 response-based knowledge
    - 입력이 각 class에 속할 확률
    - softmax로 추정 가능
    $$
    p(z_i, T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
    \tag{2}
    $$
    > $z_i$: i-th class의 logit
    - 각 soft target의 중요도를 조절하기 위해 온도 계수 $T$가 도입됨
    - soft target은 teacher 모델로부터 유용한 dark knowledge를 담고 있음
    - 다음과 같이 distillation loss 표현 가능
    $$
    L_{ResD}(p(z_t, T), p(z_s, T)) = L_R(p(z_t, T), p(z_s, T))
    \tag{3}
    $$
    > $L_R(p(z_t, T), p(z_s, T))$: KL divergence loss
- 식 (1)이나 (3)을 최적화하면 student model의 logit $z_s$가 teacher model의 logit $z_t$와 맞춰질 수 있음
- distillation & student loss를 합친 모델(Fig 5 참고)
    - vanilla knowledge distillation 벤치마크 모델
    - student loss는 항상 GT label과 학생 모델의 soft logits 사이의 cross-entropy loss로 정의됨
    $L_{CE}(y, p(z_s, T=1))$
- 장점
    - 직관적이고 이해하기 쉬운 방법
        - dark knowledge 맥락에서 도드라짐
    - label smoothing이나 regularization과 유사
- 단점
    - 마지막 층의 출력에 의존
        - teacher model의 중간 레벨 supervision을 다루지 못함
        - 이는 아주 깊은 신경망으로 representation learning을 할 때 중요
    - soft logit이 class 확률 분포이기 때문에, response based knowledge는 supervised learning에 한정된다고 할 수 있음

### 2.2 Feature-based knowledge

![alt text](./images/Fig%206.png)
> **Figure 6. 일반적인 feature-based knowledge distillation**

- Deep Neural Network는 점점 더 추상화되는 다양한 level의 feature representation을 학습하는 데 능숙함
    - 이를 Representation learning이라고 함
- 마지막 층 뿐만 아니라, 중간 층의 feature map도 student model의 supervision으로 사용 가능
    - 중간 층의 feature based knowledge는 response-based knowledge의 좋은 확장임
    - 얇고 긴 네트워크 학습에 유용
- 관련 논문들
    - 처음에 Fitnets(Romero et al. 2015)에서 student model의 학습을 향상시키기 위한 hint를 제공하는 방식으로 소개됨
        - teacher와 student의 feature activation을 직접 맞추는 게 중요
    - Zagoruyko and Komodakis(2017)
        - original feature map에서 knowledge를 표현하기 위해 attention map 도출
    - Huang and Wan(2017)
        - neuron selectivity transfer을 통해 attention map을 일반화
    - Passalis and Tefas(2018)
        - feature space에서 확률 분포를 맞추는 방식으로 knowledge transfer
    - Kim et al.(2018)
        - teacher knowledge를 더 쉽게 전달하기 위해 중간 표현을 더 이해하기 쉬운 형태로 만든 'factors' 도입
    - Jin et al.(2019)
        - teacher와 student 간 성능 격차를 줄이기 위해 hint layer의 출력을 이용해 student를 supevise하는 route constrained hint learning을 제안
    - Heo et al.(2019c)
        - hidden neurons의 activation boundary를 knowledge transfer에 사용
    - Zhou et al.(2018)
        - teacher model의 중간층 파라미터 공유와 response-based knowledge를 함께 사용하는 방식도 teacher knowledge로 사용됨
    - Chen et al.(2021)
        - teacher layer을 student layer에 맞게 적절히 할당하는 cross-layer knowledge distillation 제안
- feature-based knowledge를 위한 loss는 다음과 같이 공식화 가능
$$
L_{FeaD}\big(f_t(x), f_s(x)\big) = L_F\big(\Phi_t(f_t(x)), \Phi_s(f_s(x))\big)
\tag{4}
$$
> - $(f_t(x))$: Teacher의 중간층 **feature map**  
> - $(f_s(x))$: Student의 중간층 **feature map**  
> - $(\Phi_t(\cdot), \Phi_s(\cdot))$: 서로 다른 크기/형태의 feature를 맞추기 위한 **변환 함수**  
> - $(L_F(\cdot,\cdot))$: feature 유사성 측정에 사용하는 **distillation loss**
- teacher와 student model의 feature map이 동일한 형태가 아닐 때는 변환 함수가 적용됨
- $L_F(\cdot)$: teacher와 student model의 feature map을 맞추는 데 사용되는 유사도 함수
- 일반적인 feature-based KD model은 Fig 6 참고
- feature types, source layers, distillation loss 관점에서 다양한 feature-based knowledge 유형을 Table 1에 정리
    - $L_2$: L2 norm
    - $L_1$: L1 norm
    - $L_{CE}$: Cross Entropy loss
    - $L_{MMD}$: 최대 평균 차이 loss
- feature-based knowledge가 student model 학습에 유리한 정보를 제공
    - teacher model에서 hint layer을, student model에서 guided layer을 어떻게 선택할지는 연구 필요(Romero et al. 2015)
    - hint layer과 guided layer 크기 차이가 크기 때문에, feature representation을 적절히 맞추는 방법 탐구 필요

### 2.3 Relation-Based Knowledge

![alt text](./images/Fig%207.png)
> **Figure 7. 일반적인 instance relation-based knowledge distillation**

- Relation-Based knowledge는 서로 다른 층이나 데이터 샘플 간의 관계를 탐구
- 관련 연구
    - Yim et al. (2017)
        - 두 layer 간의 gram matrix로 정의되는 Flow of Solution Process(FSP)를 제안
        - FSP 행렬은 feature map 쌍 간의 관계를 요약
        - 두 layer에서 나온 feature map 간의 내적을 이용해 계산됨
    - Lee et al. (2018)
        - 주요 정보를 추출하기 위해 singular value decomposition을 통한 knowledge distillation 제안
        - feature map 간의 correlation을 distilled knowledge로 사용
    - Zhang and Peng (2018)
        - 각 teacher model의 logit과 feature을 각각 노드로 사용해서 두 개의 그래프 생성
        - 서로 다른 teacher model의 중요성과 관계는 knowledge transfer 전에 logit과 representation graph를 통해 모델링됨
    - Lee and Song (2019)
        - multi-head graph-based knowledge distillation
        - graph knowledge는 multi-head attention network를 통해 두 feature map 간의 데이터 내 관계를 의미
    - Passalis et al. (2020b)
        - student model은 teacher model의 hint layer pair 간의 mutual information flow도 모방
- feature map 간의 관계에 기반한 relation-based knowledge의 distillation loss는 다음과 같이 표현 가능
$$
L_{\text{RelD}}(f_t, f_s) \;=\; L_{R^1}\!\big(\Psi_t(\hat{f}_t, \breve{f}_t),\; \Psi_s(\hat{f}_s, \breve{f}_s)\big)
\tag{5}
$$
> - $f_t, f_s$: teacher과 student model의 feature map
>   - $\hat{f}_t, \breve{f}_t$은 teacher model에서 선택
>   - $\hat{f}_s, \breve{f}_s$은 student model에서 선택
> - $\Psi_t(\cdot), \Psi_s(\cdot)$: teacher/student 모델의 feature map 쌍의 유사도 함수
> - $L_{R^1}(\cdot)$: teacher과 student feature map 간의 correlation function

- 전통적인 knowledge transfer 방법은 종종 individual knowledge distillation을 포함
    - 증류된 지식에는 feature 정보뿐만 아니라 데이터 샘플 간의 상호 관계도 포함됨
    - Liu et al. (2019g)
        - instance relationship graph를 통한 강력하고 효과적인 knowledge distillation 방법 제안
        - instance relationship graph에서 전달된 지식이 포함하는 내용
            - instance relationship
            - instance features
            - layer간의 feature space transformation
    - Park et al. (2019)
        - relational knowledge distillation 제안
        - instance 관계에서 knowledge를 전달하는 방법
    - Chen et al. (2021)
        - manifold learning 아이디어에 기반해서, student network는 teacher network의 중간 layer에서 sample 간 feature similarities를 보존하는 feature embedding을 통해 학습됨
    - Passalis and Tefas (2018), Passalis et al. (2020a)
        - 데이터 샘플 간의 관계는 데이터의 feature representation을 사용하여 확률 분포로 모델링됨
    - Tung and Mori (2019)
        - similarity preserving knowledge distillation 방법 제안
        - teacher network에서 입력 쌍의 비슷한 activation에서 생기는 similarity-preserving knowledge를 student network로 전달
        - pairwise similarity 보존
    - Peng et al. (2019a)
        - 상관 일치(correlation congruence)에 기반한 knowledge distillation 방법 제안
        - 증류된 지식에는 instance-level 정보뿐만 아니라 instance 간 상관관계도 포함됨

- instance 관계를 기반으로 한 relation-based knowledge distillation loss는 다음과 같이 공식화 가능
    $$
    L_{\text{RelD}}(F_t, F_s) \;=\; L_{R2}\!\big(\psi_t(t_i,t_j),\; \psi_s(s_i,s_j)\big)
    \tag{6}
    $$
    - $(t_i, t_j) \in F_t$, $(s_i, s_j) \in F_s$
    - $\psi_t(\cdot), \psi_s(\cdot)$: $(t_i, t_j), $(s_i, s_j)$의 유사도 함수
    - $L_{R^2}(\cdot)$: teacher와 student feature representation 사이의 correlation function
- 증류된 지식은 데이터의 구조화된 지식이나 입력 특징에 대한 특권 정보처럼 다양한 관점에서 분류 가능

![alt text](./images/Table%202.png)
> **Table 2. 다양한 형태의 지식(Knowledge) 유형별 대표 기법 및 특성 요약**

---

## 3. Distillation Schemes

![alt text](./images/Fig%208.png)
> **Figure 8. Offline distillation, online distillation, self-distillation의 개략적 비교**

- 교사 및 학생 모델의 학습 업데이트 타이밍(동시 업데이트 여부)에 따라 **Offline Distillation**, **Online Distillation**, **Self-Distillation**의 3가지 주요 카테고리로 분류됨
- 인간의 학습 방식에 따른 직관적 비유:
    - Offline Distillation: 박식한 교사(teacher)가 학생(student)을 가르침
    - Online Distillation: 교사와 학생(또는 동료 학생들)이 함께 학습함
    - Self-Distillation: 학생 스스로 지식을 학습함
    - 이 3가지 증류 방식은 각자의 장점을 살려 상호 보완적으로 결합될 수 있음 (예: Sun et al., 2021)

### 3.1 Offline Distillation
- 기존의 대부분 지식 증류 방법이 속하는 전통적인 방식 (Hinton et al., 2015)
- 사전 학습된(pre-trained) 교사 모델의 지식을 학생 모델로 전이하는 2단계(two-stage) 학습 절차:
    - 1단계: 대규모 교사 모델을 특정 데이터셋에서 독립적으로 사전 학습
    - 2단계: 사전 학습된 교사 모델로부터 logit이나 중간 feature 형태의 지식을 추출하여 학생 모델 학습에 지침으로 활용
- 교사 모델 구조나 학생과의 관계는 이미 정의된 것으로 간주하며, 지식의 표현 방식 및 손실 함수(loss function) 개선에 주력:
    - 지식 표현 설계: Hinton et al. (2015), Romero et al. (2015)
    - 특징/분포 매칭 손실 함수: Huang and Wang (2017), Passalis and Tefas (2018), Zagoruyko and Komodakis (2017), Mirzadeh et al. (2020), Li et al. (2020d), Heo et al. (2019b), Asif et al. (2020)
- **장점**:
    - 구조가 단순하고 구현이 쉬움 (여러 머신에서 개별 패키지로 사전 학습된 다양한 모델의 지식을 캐시로 저장해 재활용 가능)
- **단점**:
    - 복잡한 대용량 교사 모델을 사전 학습하는 데 막대한 시간과 계산 비용 소요
    - 1방향(one-way) 지식 전이에 의존하여 학생이 교사에 과도하게 의존함
    - 대형 교사와 소형 학생 간의 모델 용량 차이(capacity gap) 문제 해결이 어려움 (Mirzadeh et al., 2020)

### 3.2 Online Distillation
- 고성능 대용량 교사 모델을 미리 준비하기 어려운 한계를 극복하기 위해 제안됨 (Mirzadeh et al., 2020; Zhang et al., 2018b; Chen et al., 2020a)
- 교사 모델과 학생 모델이 동시에 업데이트되며 전체 지식 증류 프레임워크가 단일 단계 엔드투엔드(end-to-end)로 학습됨
- **주요 연구 기법**:
    - **Deep Mutual Learning (DML)** (Zhang et al., 2018b)
        - 다수의 신경망 모델이 협력(collaborative) 방식으로 지식을 서로 교환
        - 학습 도중 임의의 모델이 학생이자 동시에 다른 모델의 교사 역할을 수행
    - **Collaborative Learning Soft Logits Ensemble (KDCL)** (Guo et al., 2020)
        - soft logit 앙상블을 활용하여 DML의 일반화 성능 확장
    - **Auxiliary Peers & Group Leader** (Chen et al., 2020a)
        - DML 프레임워크에 보조 피어(peer)들과 그룹 리더를 도입하여 피어 모델 간 다양성 강화
    - **Multi-branch Architecture** (Zhu and Gong, 2018)
        - 계산 비용 절감을 위해 각 브랜치를 학생 모델로 설정하고 메인 백본 네트워크를 공유
    - **Feature Fusion Module** (Kim et al., 2019b)
        - logit 앙상블 대신 feature fusion 모듈을 도입해 효율적인 교사 분류기 구축
    - **Cheap Convolution Student** (Xie et al., 2019)
        - 표준 컨볼루션을 가벼운 연산(cheap conv operations)으로 대체하여 학생 모델 생성
    - **Co-Distillation** (Anil et al., 2018)
        - 대규모 분산 신경망 학습을 위해 동일 구조를 가진 다수의 모델을 병렬 학습시키며 서로 지식을 전달
    - **Online Adversarial KD** (Chung et al., 2020)
        - 클래스 확률과 feature map 지식을 discriminator를 통해 동시에 온라인에서 학습
    - **Adversarial Co-Distillation** (Zhang et al., 2021a)
        - GAN을 활용하여 서로 다른 다변화된 예제(divergent examples)를 생성하며 co-distillation 수행
    - **Multi-knowledge Transfer Integration** (Sun et al., 2021)
        - Online distillation과 Self-distillation을 프레임워크 내에서 효과적으로 통합
- **특징 및 한계**:
    - **장점**: 병렬 연산을 지원하는 효율적인 1단계 엔드투엔드 학습 프로세스
    - **한계점**: 고용량 교사 모델을 설정하기 어려워 교사-학생 간 복잡한 관계 모델링에 대한 추가 연구 필요

### 3.3 Self-Distillation
- 교사와 학생 모델에 동일한 네트워크(same network)를 활용하는 방식 (Zhang et al., 2019b; Hou et al., 2019; Zhang and Sabuncu, 2020)
- Online distillation의 특수한 형태로 간주될 수 있음
- **주요 연구 기법**:
    - **Deep to Shallow Section Distillation** (Zhang et al., 2019b)
        - 동일 네트워크의 깊은 층(deeper sections)에서 추출한 지식을 얕은 층(shallow sections)으로 증류
    - **Self-Attention Distillation (SAD)** (Hou et al., 2019)
        - 차선 변경 감지(lane detection)를 위해 자체 네트워크의 상위 레이어 attention map을 하위 레이어의 타겟으로 전달
    - **Snapshot Distillation** (Yang et al., 2019b)
        - 동일 네트워크 내에서 이전 에폭(earlier epoch, 교사)의 지식을 나중 에폭(later epoch, 학생)으로 전이
    - **Early Exit Distillation** (Phuong and Lampert, 2019b)
        - 조기 종료(early exit) 레이어가 학습 도중 나중 종료 레이어의 출력을 모방하여 추론 시간 단축
    - **Theoretical & Empirical Analysis** (Mobahi et al., 2020; Zhang and Sabuncu, 2020)
        - Self-distillation이 정규화(regularization) 및 분산 감소에 미치는 수학적/실험적 효과 증명
    - **Teacher-free KD** (Yuan et al., 2020)
        - Label smoothing 정규화 분석을 기반으로 사전 학습된 교사 없이 스스로 증류 수행
    - **Self-Knowledge Probabilities** (Hahn and Choi, 2019)
        - 기존 soft probability 대신 임베딩 공간의 데이터 유사도를 반영한 예측 확률(predicted probabilities)을 self-knowledge로 활용
    - **Class-wise Self-KD** (Yun et al., 2020)
        - 동일 클래스 내 intra-class 샘플 및 데이터 증강 샘플 간의 출력 분포를 맞춤
    - **Data Augmentation Self-KD** (Lee et al., 2019a)
        - 증강된 데이터로부터 추출된 self-knowledge를 모델 자체에 다시 증류
    - **Sequential Multi-step Self-KD / Born Again** (Furlanello et al., 2018; Bagherinezhad et al., 2018)
        - 동일한 아키텍처를 가진 네트워크들을 단계별(step-by-step)로 순차적(sequential)으로 학습시키며 직전 네트워크를 교사로 활용

---

## 4. Teacher-Student Architectures

![alt text](./images/Fig%209.png)
> **Figure 9. Teacher-student 모델 구조 간 관계 및 설정**

- 지식 증류에서 교사-학생 아키텍처는 지식 전달을 담는 일반적인 틀(carrier)이며, 올바른 교사를 선택하고 적절한 학생 구조를 설계하는 것이 핵심 과제임
- 기존 연구에서는 교사와 학생의 구조가 고정(pre-fixed)되어 있어 용량 차이(capacity gap) 문제가 빈번히 발생함

#### 4.1 Teacher-Student Model Setups (구조적 설정 분류)
- 깊이(depth)와 폭(width)의 차이에 따른 5가지 주요 학생 네트워크 설정:
    1. **단순화된 소형 구조 (Simplified version of teacher)** (Wang et al., 2018a; Zhu and Gong, 2018; Li et al., 2020d)
        - 레이어 수와 채널 수가 적은 구조
    2. **양자화된 구조 (Quantized version of teacher)** (Polino et al., 2018; Mishra and Marr, 2018; Wei et al., 2018; Shin et al., 2019)
        - 전체적인 네트워크 형태를 유지하면서 가중치/피처를 양자화
    3. **효율적인 연산 블록 적용 구조 (Efficient basic operations)** (Howard et al., 2017; Zhang et al., 2018a; Huang et al., 2017)
        - Depthwise separable convolution 등 경량 모듈 적용
    4. **최적화된 글로벌 구조 (Optimized global structure)** (Liu et al., 2019i; Xie et al., 2020; Gu and Tresp, 2020)
        - 전역 네트워크 구조를 탐색하여 간소화
    5. **교사와 동일한 구조 (Same network as teacher)** (Zhang et al., 2018b; Furlanello et al., 2018; Tarvainen and Valpola, 2017)

#### 4.2 Model Capacity Gap (용량 차이) 완화 기법
- 대형 교사와 소형 학생 간의 학습 격차를 줄이기 위한 접근:
    - **Teacher Assistant (TA)** (Mirzadeh et al., 2020)
        - 교사와 학생 사이에 중간 크기의 보조(assistant) 모델을 두어 단계별 지식 전이
    - **Residual Learning TA** (Gao et al., 2021)
        - 보조 모델이 잔차 오차(residual error)를 학습하도록 하여 격차 축소
    - **Structure Compression** (Nowak and Corso, 2018)
        - 다중 레이어에서 학습된 지식을 단일 레이어로 압축 전이
    - **Progressive Block-wise Transfer** (Wang et al., 2018a)
        - 수용 영역(receptive field)을 유지하며 블록 단위로 단계적 지식 전이
    - **Online Ensemble Architecture** (Zhang et al., 2018b; Zhu and Gong, 2018; Furlanello et al., 2018; Chen et al., 2020a)
        - 온라인 학습 환경에서 동일하거나 유사한 구조를 가진 학생 모델 간 앙상블 활용

#### 4.3 Neural Architecture Search (NAS) 기반 아키텍처 탐색
- 엣지 디바이스용 경량 신경망 설계를 위한 depthwise separable conv 활용 (Chollet, 2017; Howard et al., 2017; Sandler et al., 2018; Zhang et al., 2018a; Ma et al., 2018)
- 메타 연산/블록 검색 기반 NAS 발전 (Wu et al., 2019; Tan et al., 2019; Tan and Le, 2019; Radosavovic et al., 2020)
- 지식 증류 내 NAS 결합 연구:
    - **Reinforcement Learning Layer Pruning** (Ashok et al., 2018)
        - 강화학습을 이용해 불필요한 레이어를 데이터 드라이븐 방식으로 자동 제거
    - **Optimal Student Search** (Liu et al., 2019i; Xie et al., 2020; Gu and Tresp, 2020)
        - 주어진 교사 모델에 최적인 학생 아키텍처를 자율적으로 탐색
    - **Future Direction**: 교사의 가이드하에 학생 구조와 지식 전이 기법을 동적으로 공동 탐색(joint search)하는 연구가 핵심 미래 과제임


---

## 5. Distillation Algorithms

다양한 복잡한 설정에서 지식 전달 프로세스를 개선하기 위해 여러 지식 증류 알고리즘이 제안됨 (Romero et al., 2015; Hinton et al., 2015; Passalis and Tefas, 2018)

### 5.1 Adversarial Distillation
![alt text](./images/Fig%2010.png)
> **Figure 10. Adversarial learning 기반 지식 증류의 세 가지 주요 범주**

- 교사 모델이 실제 데이터 분포를 완벽히 학습하기 어렵거나 학생 모델의 용량 한계로 교사를 정확히 모방하지 못하는 문제를 해결하기 위해 GAN(Goodfellow et al., 2014) 도입
- 세 가지 주요 범주 (Wang et al., 2018e; Xu et al., 2018a,b; Micaelli and Storkey, 2019; Liu et al., 2018; Wang et al., 2018f; Chen et al., 2019a; Shen et al., 2019d; Shu et al., 2019; Liu et al., 2020a; Belagiannis et al., 2018):
    1. **생성자(Generator) 기반 데이터 생성 및 증강** (Fig 10a)
        - 합성 데이터를 생성하여 학습 데이터셋으로 직접 사용하거나 데이터 증강에 활용 (Chen et al., 2019a; Ye et al., 2020; Liu et al., 2018)
        - 지식 전이를 위한 난해한 예제(hard examples) 생성 (Micaelli and Storkey, 2019)
        $$
        L_G(F_t(G(z)), F_s(G(z)))
        \tag{7}
        $$
    2. **판별자(Discriminator) 기반 모방** (Fig 10b)
        - logit (Xu et al., 2018a,b) 또는 feature (Wang et al., 2018f)를 이용하여 학생과 교사 모델의 샘플을 구분하는 discriminator 도입
        - 비라벨(unlabeled) 데이터 활용 지식 전이 (Belagiannis et al., 2018), 다중 discriminator 도입 (Shen et al., 2019d), 용량 격차 완화를 위한 squeezed knowledge 압축 지식 (Shu et al., 2019)
        $$
        L_{GAN}(G, D)
        \tag{8}
        $$
    3. **Online Adversarial Distillation** (Fig 10c)
        - 교사와 학생 모델을 매 이터레이션마다 동시에 공동 최적화 (Wang et al., 2018e; Chung et al., 2020)
- **GAN 모델 압축**: 소형 GAN 학생 네트워크가 대형 GAN 교사 네트워크를 지식 증류로 모방 및 압축 (Aguinaldo et al., 2019; Li et al., 2020c)

### 5.2 Multi-Teacher Distillation
![alt text](./images/Fig%2011.png)
> **Figure 11. Multi-teacher distillation의 일반적인 프레임워크**

- 다수의 교사 네트워크가 각자의 유용한 지식을 제공하여 학생 네트워크를 학습 (Sau and Balasubramanian, 2016; You et al., 2017; Chen et al., 2019b; Furlanello et al., 2018; Yang et al., 2019a; Zhang et al., 2018b; Lee et al., 2019c; Park and Kwak, 2020)
- **단순 평균 응답 활용**: 모든 교사의 logit 평균을 supervision 신호로 사용 (Hinton et al., 2015)
- **주요 접근법**:
    - 샘플 간 비유사성을 장려하기 위한 중간층 feature 통합 (You et al., 2017)
    - 2개의 교사 모델 활용 (한 교사는 response 기반, 다른 교사는 feature 기반 지식 전달) (Chen et al., 2019b)
    - 이터레이션마다 교사 풀에서 무작위로 교사 선택 (Fukuda et al., 2017)
    - 추가 교사 브랜치(teacher branches)를 학생 네트워크에 부착하여 중간 feature 모방 (Park and Kwak, 2020; Asif et al., 2020)
    - Born Again Networks: $t$ 단계의 학생을 $t+1$ 단계 학생의 교사로 순차 활용 (Furlanello et al., 2018; Yang et al., 2019a)
    - 교사 모델에 노이즈를 추가하여 다중 교사 시뮬레이션 (Sau and Balasubramanian, 2016) 또는 스토캐스틱 블록/스킵 연결 활용 (Lee et al., 2019c)
    - **지식 통합 (Knowledge Amalgamation)**: 공개된 다중 교사 모델들의 feature 앙상블을 통해 새로운 학생 모델로 지식 통합 (Shen et al., 2019a,b; Luo et al., 2019, 2020)
    - **응용 확장**: 도메인 적응(Domain Adaptation)을 위한 지식 적응 (Ruder et al., 2017) 및 개인정보/보안 보호 (Vongkulbhisal et al., 2019; Papernot et al., 2017)

![alt text](./images/Table%203.png)
> **Table 3. 대표적인 Multi-Teacher Distillation 기법들의 지식 유형 및 증류 전략 요약**

### 5.3 Cross-Modal Distillation
![alt text](./images/Fig%2012.png)
> **Figure 12. Cross-modal distillation의 일반적인 프레임워크**

- 학습 또는 테스트 중 특정 모달리티의 데이터나 레이블이 없을 때 모달리티 간 지식 전이 수행 (Gupta et al., 2016; Garcia et al. 2018; Zhao et al., 2018; Roheda et al., 2018; Zhao et al., 2020b)
- **주요 예시 및 구현 기법**:
    - RGB 이미지 사전학습 교사의 지식을 비라벨 쌍(paired) 샘플을 통해 깊이(depth) 또는 광학 흐름(optical flow) 학생으로 전이 (Gupta et al., 2016)
    - 샘플 페어링 기반 라벨 정보 등록 및 전이 (Albanie et al., 2018; Zhao et al., 2018; Thoker and Gall, 2019)
    - 동기화된 라디오 신호와 카메라 이미지를 이용한 벽 너머 인체 포즈 추정 (Zhao et al., 2018)
    - RGB 비디오 지식을 스켈레톤(skeleton) 기반 행동 인식 모델로 전이 (Thoker and Gall, 2019)
    - 추가 깊이 모달리티의 환각 스트림(hallucination stream)을 생성하여 RGB 행동 인식 강화 (Garcia et al., 2018)
    - 모달리티 간 쌍 관계 전이를 위한 대조 손실(contrastive loss) 도입 (Tian et al., 2020)
    - 결측 모달리티 작성을 위한 GAN 기반 cross-modal 증류 (Roheda et al., 2018)
    - Visual Question Answering (VQA)에서 Image-Question-Answer 3선(trilinear) 교사 지식을 Image-Question 2선(bilinear) 학생으로 증류 (Do et al., 2019)
    - 텍스트 모달리티에서 시각 모달리티로의 확률적 지식 전이 (Passalis and Tefas, 2018)
    - 교차 도메인 지식 전이 (Kundu et al., 2019; Chen et al., 2019c; Su and Maji, 2017)

![alt text](./images/Table%204.png)
> **Table 4. 모달리티별 Cross-Modal Distillation 기법 및 지식 전이 전략 요약**


### 5.4 Graph-Based Distillation
![alt text](./images/Fig%2013.png)
> **Figure 13. Graph-based distillation의 일반적인 프레임워크**

- 데이터 내 상호 관계(intra-data relationships)를 그래프로 모델링하여 지식 전이 (Chen et al., 2021; Zhang and Peng, 2018; Lee and Song, 2019; Park et al., 2019; Yao et al., 2020; Ma and Mei, 2019; Hou et al., 2020)
- **두 가지 핵심 역할**:
    1. **그래프를 지식의 반송파(carrier)로 사용**:
        - 자율 학습 교사의 logit 그래프 및 representation 그래프 구축 (Zhang and Peng, 2018)
        - 고차원 공간 샘플 관계 보존을 위한 국소 보존 손실(locality preserving loss) (Chen et al., 2021)
        - CNN 레이어 특징 기반 multi-head 그래프 분석 (Lee and Song, 2019)
        - 교사-학생 그래프 간선(edge) 매칭 Relational KD (Park et al., 2019)
        - 활성화 유사도 행렬 매칭 (Tung and Mori, 2019; Peng et al., 2019a; Liu et al., 2019g)
    2. **그래프를 통한 지식 전달 제어 (Message Passing Control)**:
        - 모달리티 간 관계 탐색을 위한 방향성 Distillation Graph (Luo et al., 2018)
        - 양방향 그래프 기반 다변화 협력 학습 (Minami et al., 2019)
        - GNN을 활용한 그래프 지식 전이 (Yao et al., 2020)
        - 그래프 컨볼루션 네트워크 위상 의미(topological semantics) 전이 (Yang et al., 2020b)

### 5.5 Attention-Based Distillation
- CNN 뉴런 활성화를 반영하는 attention 메커니즘을 지식 증류에 활용 (Zagoruyko and Komodakis, 2017; Huang and Wang, 2017; Srinivas and Fleuret, 2018; Crowley et al., 2018)
- 레이어별 특징 임베딩에 대한 attention map을 정의하여 교사 지식 전이 (Zagoruyko and Komodakis, 2017)
- 신뢰도 규칙(confidence rules) 지정을 위한 attention 메커니즘 적용 (Song et al., 2018)

### 5.6 Data-Free Distillation
- 프라이버시, 보안, 법적 이슈로 원본 학습 데이터를 사용할 수 없는 환경 극복 (Chen et al., 2019a; Lopes et al., 2017; Nayak et al., 2019; Micaelli and Storkey, 2019; Haroush et al., 2020; Ye et al., 2020; Nayak et al., 2021; Chawla et al., 2021)
- **주요 접근법**:
    - GAN 기반 합성 전이 데이터 생성 (Chen et al., 2019a; Ye et al., 2020; Micaelli and Storkey, 2019; Yoo et al., 2019; Hu et al., 2020)
    - 교사 레이어 활성화 및 스펙트럼 활성화 신호를 역전파하여 데이터 재구성 (Lopes et al., 2017)
    - DeepInversion을 통한 고품질 합성 이미지 생성 (Yin et al., 2020)
    - Softmax 공간 모델링 기반 Zero-Shot KD (Nayak et al., 2019)
    - 제한된 라벨 데이터 환경에서의 Few-Shot KD (Kimura et al., 2018; Shen et al., 2021)
    - 비라벨 데이터의 교사 어노테이션을 활용하는 Data Distillation (Radosavovic et al., 2018; Liu et al., 2019d; Zhang et al., 2020d)

### 5.7 Quantized Distillation
![alt text](./images/Fig%2015.png)
> **Figure 15. Quantized distillation의 일반적인 프레임워크**

- 고정밀 모델(32-bit float)을 저정밀 모델(2-bit, 8-bit)로 변환하는 양자화 과정과 지식 증류 결합 (Polino et al., 2018; Mishra and Marr, 2018; Wei et al., 2018; Shin et al., 2019; Kim et al., 2019a)
- **주요 구현 방식**:
    - 가중치 양자화 학생 네트워크에 지식 전이 (Polino et al., 2018)
    - Apprentice 모델: 고정밀 교사가 소형 저정밀 학생 전이 가이드 (Mishra and Marr, 2018)
    - 양자화된 Feature map 교사로부터 양자화된 학생으로 지식 전이 (Wei et al., 2018)
    - Quantization-Aware KD (자율학습 및 공동학습 프레임워크) (Kim et al., 2019a)
    - 교사 크기 및 증류 온도 등 하이퍼파라미터 영향 실증 분석 (Shin et al., 2019)
    - 파라미터를 공유하는 양자화 모델용 Self-Distillation (Boo et al., 2021)

### 5.8 Lifelong Distillation
- 지속적 학습(Continual learning) 및 메타 학습(Meta-learning)에서 이전 지식을 보존하고 멸망적 망각(Catastrophic forgetting) 방지 (Chen and Liu, 2018; Jang et al., 2019; Flennerhag et al., 2019; Peng et al., 2019b; Liu et al., 2019e; Lee et al., 2019b; Zhai et al., 2019; Zhou et al., 2020; Shmelkov et al., 2017; Li and Hoiem, 2017; Caccia et al., 2020)
- Meta-Transfer Networks (전이할 지식과 위치 동적 결정) (Jang et al., 2019)
- "Leap" 메타학습 프레임워크 (Flennerhag et al., 2019)
- Few-shot 이미지 인식 전이 네트워크 (Peng et al., 2019b)
- 이미지 검색을 위한 의미 인식 지식 보존 (Semantic-aware knowledge preservation) (Liu et al., 2019e)
- **망각 방지 기법**: Global Distillation (Lee et al., 2019b), Lifelong GAN (Zhai et al., 2019), Multi-Model Distillation (Zhou et al., 2020), Learning without Forgetting (LwF; Li and Hoiem, 2017)

### 5.9 NAS-Based Distillation
- 교사-학생 간 용량 차이(capacity gap) 문제를 해결하기 위해 Neural Architecture Search (NAS)로 학생 구조 자동 탐색 (Kang et al., 2020; Liu et al., 2019i)
- **지식 증류를 통한 NAS 탐색 효율성 향상**:
    - AdaNAS (Macko et al., 2019)
    - 증류된 구조 지식 활용 NAS (Li et al., 2020a)
    - Teacher Guided Search for Architectures (TGSA) (Bashivan et al., 2019) - 중간 feature 표현 모방 가이드
    - One-Shot NAS (Peng et al., 2020)


---

## 6. Performance Comparison

- CIFAR-10 및 CIFAR-100 데이터셋에 대해 다양한 지식 증류 기법의 분류 성능 비교 분석 (Table 5, Table 6)

![alt text](./images/Table%205.png)
> **Table 5. CIFAR-10 데이터셋에서의 다양한 지식 증류 기법 분류 정확도(%) 비교**

![alt text](./images/Table%206.png)
> **Table 6. CIFAR-100 데이터셋에서의 다양한 지식 증류 기법 분류 정확도(%) 비교**

- 주요 관찰 사항:

    1. 지식 증류는 다양한 심층 모델 아키텍처에 보편적으로 적용 가능함
    2. 다양한 딥 모델의 모델 압축을 효과적으로 달성함
    3. 협력 학습(collaborative learning) 기반 Online KD는 모델 성능을 유의미하게 향상시킴 (Zhang et al. 2018b)
    4. Self-KD는 단일 모델 성능 자체를 향상시키는 데 뛰어난 효과를 보임 (Yang et al. 2019b, Yuan et al. 2020)
    5. Offline KD는 주로 feature 기반 지식을, Online KD는 주로 response 기반 지식을 전이함
    6. 고용량 교사 모델의 지식 전이를 통해 소형 학생 모델의 성능을 향상시킬 수 있음

---

## 7. Applications

Knowledge Distillation은 딥러닝 모델의 경량화 및 가속화를 위해 시각 인식, 음성 인식, 자연어 처리(NLP), 추천 시스템, 적대적 공격 방어 및 데이터 프라이버시 보호 등 딥러닝 전반에 걸쳐 넓게 적용됨 (Li and Hoiem, 2017; Peng et al., 2019b; Bagherinezhad et al., 2018)

### 7.1 Visual Recognition
- **Face Recognition (얼굴 인식)**:
    - 효율적 배포와 높은 인식 정확도를 동시에 달성하기 위한 지식 전이 (Luo et al., 2016; Kong et al., 2019; Yan et al., 2019; Ge et al., 2018; Wang et al., 2018b, 2019c; Duong et al., 2019; Wu et al., 2020)
    - 상위 힌트 레이어의 유익한 뉴런 전이 (Luo et al., 2016), 가중치 적용 힌트 레이어 피처 손실 (Wang et al., 2018b), 재귀적 지식 증류 (Yan et al., 2019)
    - 오픈셋(open-set) 얼굴 인식을 위한 특징간 거리를 반영한 loss: Angular loss (Duong et al., 2019), Correlated embedding loss (Wu et al., 2020)
    - **저해상도 얼굴 인식**: 고해상도 교사의 지식을 희소 그래프 최적화(sparse graph optimization; Ge et al., 2018), 해상도 불변 모델(Kong et al., 2019), Multi-kernel MMD 특징 손실(Wang et al., 2019c)을 통해 저해상도 학생으로 전이
- **Image Classification (이미지 분류)**:
    - 복잡한 이미지 분류를 위한 불완전/모호한 레이블 개선 (Label Refinery model; Bagherinezhad et al., 2018)
    - 멸망적 망각 방지를 위한 융합 기법 (Learning without Forgetting, LwF; Li and Hoiem, 2017)
    - GAN 기반 Feature Map 지식 전이 (Chen et al., 2018a), 이미지 분류기 시각적 해석 및 진단 (Wang et al., 2019b), 저해상도 이미지 분류 (Zhu et al., 2019)
- **Action Recognition (행동 인식)**:
    - 시공간 모달리티 증류: 상호 교사-학생 네트워크 (Thoker and Gall, 2019), 다중 스트림 네트워크 (Garcia et al., 2018), 시공간 증류 밀집 연결 네트워크 (Hao and Zhang, 2019), 그래프 증류 (Luo et al., 2018), 다중 교사-다중 학생 (Wu et al., 2019b; Zhang et al., 2020c)
- **Object Detection & Segmentation 및 기타 컴퓨터 비전**:
    - 객체 검출 (Li et al., 2017; Shmelkov et al., 2017; Wang et al., 2019d; Chawla et al., 2021)
    - 세그멘테이션 (He et al., 2019; Liu et al., 2019h; Hou et al., 2020; Bergmann et al., 2020)
    - 차선 감지 (Hou et al., 2019), 보행자 재식별 (Wu et al., 2019a), 포즈 추정 (Zhao et al., 2018), 비디오 캡셔닝 (Pan et al., 2020; Zhang et al., 2020f), 이미지 검색 (Liu et al., 2019e), 이상 감지 (Bergmann et al., 2020)

### 7.2 Natural Language Processing (NLP)
- **Neural Machine Translation (NMT, 신경망 기계 번역)**:
    - **Sequence-Level KD**: 단어 수준을 넘어 시퀀스 생성 분포 모방 (Kim and Rush, 2016)
    - **데이터 증강 및 정규화**: 시퀀스 수준 지식 증류의 해석 (Gordon and Duh, 2019)
    - **비자귀적 번역 (NAT)**: 모델 용량과 전이 데이터에 의한 성능 향상 실증 (Zhou et al., 2019a)
    - **다국어 번역**: 이중 언어 교사들로부터 다국어 학생으로 multi-teacher distillation (Tan et al., 2019)
    - **온라인 증류 기법**: 검증 성능이 우수한 최적 평가 모델을 동적으로 교사로 설정하여 학습 진동 완화 (Wei et al., 2019)
- **BERT Model Compression (BERT 경량화)**:
    - **BERT-PKD (Patient KD)**: 교사 힌트 레이어의 [CLS] 토큰 피처 표현을 학생으로 전달 (Sun et al., 2019)
    - **TinyBERT**: 일반 영역 및 타스크 특화 2단계 transformer 지식 증류 (Jiao et al., 2020)
    - **DistilBERT**: BERT와 동일한 일반 구조를 지닌 경량 학생 모델 사전학습 (Sanh et al., 2019)
    - **Task-Specific KD**: BERT 교사의 지식을 BiLSTM 경량 신경망으로 전이 (Tang et al., 2019)
    - **Internal Distillation**: 거대 교사 BERT의 내부 표현을 소형 학생으로 전이 (Aguilar et al., 2020)
- **기타 NLP 응용**:
    - 질의 응답 (QA/MRC): 어텐션 가이드 정답 증류 (Hu et al., 2018)
    - 사전학습-증류-파인튜닝 통합 (Turc et al., 2019)
    - 다중 타스크 증류: 단일 타스크 교사들이 다중 타스크 학생을 지도하는 Single-Multi Born-Again (Clark et al., 2019)
    - 다국어 사전 유도 및 저자원 언어 확장 (Nakashole and Flauger, 2017; Cui et al., 2017)

### 7.3 Speech Recognition
- **Acoustic Model Compression (음향 모델 압축)**:
    - **시계열 정보 전이**: RNN 교사 모델의 시계열 지식을 소형 DNN/RNN 학생 모델로 전달 (Chan et al., 2015)
    - **앙상블 음향 모델 증류**: 2,000시간 분량의 LVCSR(대용량 어휘 연속 음성 인식) 5개 언어 타스크 적용 (Chebotar and Waters, 2016)
    - **단문 발화 언어 식별 (LID)**: 장문 발화 교사 피처를 단문 발화 학생으로 증류 (Shen et al., 2018), 상호작용 온라인 증류 (Shen et al., 2019c), 내부 표현 증류 (Shen et al., 2020)
    - **음성 향상 및 오디오 분류**: 오디오-비주얼 다중 모달 지식 증류 (Perez et al., 2020), 잡음 환경 고화질 음성 향상 (Watanabe et al., 2017), 양자화 지식 증류 오디오 이벤트 감지 (Shi et al., 2019a)
- **Sequence-Level KD in Speech**:
    - CTC(Connectionist Temporal Classification) 시퀀스 모델에 시퀀스 수준 증류 적용하여 프레임 수준 한계 극복 (Wong and Gales, 2016; Huang et al., 2018)
    - 양방향 LSTM-CTC 교사의 지식을 실시간 단방향 LSTM-CTC 학생으로 증류 (Takashima et al., 2018)
- **도메인 적응 및 언어 모델 결합**:
    - 소량 데이터 소스 도메인 과적합 방지를 위한 정규화 (Asami et al., 2017)
    - 다중 억양(multi-accent) 음성 인식 (Ghorbani et al., 2018)
    - 외부 언어 모델(LM 교사)을 Seq2Seq 학생 모델로 통합하여 문자 오류율(CER) 감소 (Bai et al., 2019)

### 7.4 Other Applications
- **Recommender Systems (추천 시스템)**:
    - **Ranking Distillation**: 추천을 랭킹 문제로 정의하고 지식 증류 최초 적용 (Tang and Wang, 2018)
    - **Adversarial KD**: 리뷰 예측 교사 네트워크와 유저-아이템 예측 학생 네트워크 간 적대적 학습 (Chen et al., 2018b)
    - **ECAE (Enhanced Collaborative Denoising Autoencoder)**: 피드백 지식 캡처 및 노이즈 제거 증류 레이어 설계 (Pan et al., 2019)
- **Adversarial Robustness & Data Privacy (적대적 공격 방어 및 데이터 프라이버시)**:
    - **적대적 공격 방어**: 교사 모델의 강건한 출력을 증류하여 노이즈/적대적 교란 극복 (Papernot et al., 2016; Ross and Doshi-Velez, 2018; Goldblum et al., 2020)
    - **민감 데이터 프라이버시 보호**: 데이터 파티션별 다중 교사를 통한 비라벨/민감 데이터 보호 (Papernot et al., 2017; Vongkulbhisal et al., 2019)
    - **Private Model Compression**: 공개 데이터와 민감 데이터를 분리하여 프라이버시 손실 및 배치 손실 적용 (Wang et al., 2019a)
    - **Few-shot 프라이버시 압축**: 클래스당 소수 샘플 기반 레이어별 지식 증류 (Bai et al., 2020)
- **기타 신흥 분야**:
    - 딥러닝 모델의 해석 가능성(Interpretability) 진단 (Liu et al., 2018b)
    - 연합 학습 (Federated Learning) 내 지식 증류 적용 (Bistritz et al., 2020; Lin et al., 2020; Seo et al., 2020; He et al., 2020a)

---

## 8. Challenges and Future Directions

- **지식의 원천과 이해(Understanding of Knowledge)**:
    - 교사 네트워크에서 추출된 지식이 왜 학생의 학습을 돕는지에 대한 이론적/수학적 정립 부족 (Phuong and Lampert 2019a, Ji and Zhu 2020)
    - 지식의 품질 및 표현 방식에 대한 정량적 평가 메커니즘 개발 필요
- **용량 차이 및 증류 한계(Capacity Gap & Distillation Limit)**:
    - 교사 모델의 용량이 매우 커지면 오히려 학생 모델 학습에 부정적인 영향을 미치는 문제 발생 (Cho and Hariharan 2019, Mirzadeh et al. 2020)
    - 교사와 학생 간 능률적인 가이드라인 및 중간 전달체 설계 필수
- **다양한 지식 및 모달리티의 효율적 통합(Knowledge Integration)**:
    - response, feature, relation 기반 지식 및 멀티 교사/멀티 모달리티 지식을 최적으로 조합하는 범용적 프레임워크 연구 필요
- **Data-Free 및 프라이버시 환경(Data-Free & Privacy-Preserving KD)**:
    - 원본 데이터를 사용할 수 없는 제한적 환경에서의 고품질 데이터 합성 및 안정적인 증류 기술 발전 필요
- **NAS와의 자동화 결합(Automated Architecture & KD Search)**:
    - 교사-학생 구조 설계 및 지식 전이 전략을 자율적으로 탐색하는 자동화 연구의 중요성 증대

---

## 9. Conclusion

- 본 논문은 딥러닝 모델의 경량화 및 가속화를 위한 대표적 핵심 기술인 Knowledge Distillation(지식 증류) 분야에 대한 종합적인 Survey를 제공함
- 지식의 유형(Response, Feature, Relation), 학습 전략(Offline, Online, Self-distillation), 교사-학생 구조, 다양한 알고리즘(Adversarial, Multi-teacher, Cross-modal, Graph-based, Quantized, Lifelong, NAS-based 등)과 실제 적용 분야(Vision, NLP, Speech, Recommender system)를 체계적으로 정립함
- 자원 제약이 있는 엣지 컴퓨팅 환경 배포 및 초거대 모델 압축 시대에서 Knowledge Distillation은 향후 딥러닝 연구 및 산업 현장에서 더욱 핵심적인 위치를 차지할 것임


