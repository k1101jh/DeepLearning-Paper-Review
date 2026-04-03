# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Vision Language Action Model
- Chain-of-Thought Reasoning

---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html) (arXiv 2025)
- [project](https://cot-vla.github.io/)

---
- **Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin
- **Affiliation**: NVIDIA, Stanford University, MIT
- **Venue**: CVPR 2025

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

**Vision-Language-Action Models(VLAs)**
- 사전 학습된 vision-language model과 다양한 로봇 시연을 활용
- 일반화 가능한 센서모터 제어를 학습하는데 잠재력을 보임
- 현재의 VLAs 문제점
    - 주로 직접적인 입력-출력 매핑에 초점을 맞춤
        - 복잡한 조작 작업에 필수적인 중간 추론 단계가 결여됨
        - 시간적 계획 / 추론 능력이 부족함

**CoT-VLA**
- VLAs에 명시적 시각적 chain-of-throught(CoT) reasoning을 통합하는 방법 제안
    - 시각적 목표를 미래 이미지 frame으로 autoregressively하게 예측
    - 짧은 행동 sequence를 생성
- SOTA 7B VLA
- real-world 조작 작업에서 SOTA VLA 모델을 17% 능가
- 시뮬레이션 벤치마크에서 6% 우수한 성능을 보임

![alt text](./images/Fig%201.png)

> **Figure 1. Vanilla VLA와 CoT-VLA 프레임워크 간의 비교**  
> (위): 기존 VLA 모델은 명시적인 추론 단계를 거치지 않고 입력으로부터 로봇 행동을 직접 예측  
> 학습을 위해 action-annotated 로봇 시연 데이터만 사용  
> (아래): CoT-VLA는 행동이 없는 EPIC-KITCHEN-100[27]과 같은 데이터도 활용하여 subgoal 이미지 생성 능력을 향상시킬 수 있음  
> 풍부한 unlabeld video 데이터를 사용하여 VLA의 시각적 추론 능력을 개선할 잠재력을 해방할 수 있음  
> 작업 흐름: 중간 추론 단계로 subgoal image 생성 -> subgoal을 달성하기 위한 짧은 action sequence 생성


## 1. Introduction

- 로봇 학습에서의 VLA
    - 자연어 지시와 시각적 관찰을 로봇 동작으로 매핑
    - VLM을 로봇 시연으로 훈련
        - VLA가 다양한 장면과 객체, 자연어 지시를 이해하는 능력을 물려받음
        - 하위 테스트 시나리오에서 fine-tuning할 때 더 나은 일반화 능력 발휘 가능
    - 일반적으로 명시적인 중간 reasoning step 없이 관측에서 행동으로 directly mapping함
        - 중간 단계는 해석 가능성을 높이고 잠재적으로 성능을 향상시킬 수 있음

- Language domain에서 Chain-of-Thought(CoT) prompting
    - step-by-step 사고를 장려하여 LLMs의 reasoning 능력을 향상시키는 강력한 기술
- 로보틱스에 이러한 컨셉을 적용
    - 텍스트, 시각적 관측, 물리적 행동에서 추론을 기반으로 하는 기회 제공
- 최근 작업
    - [15, 44, 45, 63]
        - language descriptions, keypoints, bounding boxes와 같은 중간 reasoning step을 결합
        - 이러한 중간 표현은 장면의 상태, 객체, 작업의 추상적인 상태를 포착하여 종종 추가 pre-processing pipeline이 필요함
- 제안 방법
    - subgoal image를 action generation 이전에 중간 reasoning step으로 탐구
        - model의 reasoning 과정의 상태를 포착
        - robot demonstration dataset에서 자연스럽게 얻을 수 있음
    - subgoal을 중간 CoT reasoning step으로서 VLAs와 통합한 최초의 사례
        - 이전 연구에서는 하위 목표 생성과 goal-conditioned 모방 학습을 탐구
    - subgoal 이미지 생성을 사용하는 VLA를 위한 시각적 CoT reasoning을 제안
    - 방법
        1. 로봇의 계획된 상태를 pixel 공간에서 나타내는 subgoal 이미지 생성
        2. 현재 관찰과 생성된 subgoal image를 기반으로 행동을 조건화
    - 모델이 행동하기 전에 task를 수행하는 방법을 시각적으로 생각할 수 있게 함
    - 최소한의 전처리만으로 로봇 조작 데이터에 이미 존재하는 정보 활용 가능
    - subgoal 이미지 생성은 행동 주석을 필요로 하지 않음
        - 시각적 추론 및 이해를 향상시킬 잠재력을 열 수 있음

**CoT-VLA**
- visual CoT reasoning을 활용하는 통합 multimodal foundation model 기반 시스템
- Open X-Embodiment dataset & action-less video dataset을 사용해서 base model 훈련
- downstream 로봇 환경에서 수집된 task 시연으로 모델을 fine-tune
- hybrid attention mechanism을 설계
    - text 및 image 생성을 위해 다음 token 예측과 함께 causal attention을 사용
    - 모든 action 차원을 한 번에 예측하기 위해 full attention 활용

- 각 timestep마다 단일 행동을 예측하기 보다 행동 sequence를 예측(action chunking)
- action chunking & hybrid attention 메커니즘이 모델 성능을 향상시킴을 입증
- visual chain-of-thought reasoning이 기존 VLA 접근법과 비교하여 정책 성능을 향상시키는데 도움이 됨
    - 시뮬레이션 벤치마크[37] & real-world 실험[48, 60] 모두에서 광범위한 실험을 통해 입증

**논문의 기여**
- 중간 reasoning step으로서 subgoal image 생성을 통한 visual CoT reasoning을 수행하는 방법 제안
- visual CoT reasoning을 통합하고 hybrid attention 메커니즘을 갖는 시스템 CoT-VLA 소개
- 시뮬레이션과 실제 환경에서 포괄적인 평가를 통해 visual CoT reasoning이 VLA 성능을 향상시키고 제안 시스템이 여러 로봇 플랫폼과 tasks에서 SOTA를 달성함을 보임

## 2. Related Work

**Chain-of-Thought (CoT) Reasoning**
- LLM에서 문제를 순차적이고 설명 가능한 단계로 나누어 모델이 복잡하고 multi-step reasoning 작업을 수행할 수 있도록 하는데 중요함
- 초기 연구[62]는 CoT reasoning이 LLM 모델이 중간 reasoning step을 생성하도록 LLM에 prompting하는 것이 효과적임을 입증
- Visual domain으로 확장
    - 시각 정보를 단계적으로 처리하여 향후 결과나 상태를 추론하는 multimodal CoT 방법 탐구
    - 예시:
        - bounding box 생성[53]
        - Stable Diffusion[50]
        - 표준 python package를 이용한 중간 이미지 채우기[24]
        - CLIP embedding 생성[22]
    - CoT reasoning이 embodied application으로 확장됨
        - multi-stage 실행을 위한 texture 계획 생성[44, 45]
        - point 궤적[63]
        - 객체와 gripper 위치의 bounding box를 추가 관찰로 라벨링[44]
        - open-loop following을 위한 future image 궤적 생성[35, 47]
        - 강화 학습을 위한 fine-grained reward guidance 생성
- Visual-CoT reasoning
    - 로봇 조작을 위해 예측된 subgoal 이미지를 closed-loop action 생성의 중간 reasoning step으로 활용
    - 추가 주석 없이 시연 동영상을 자연스러운 중간 추론 상태로 활용

**Vision-Language-Action Models**


## 3. CoT-VLA

![alt text](./images/Fig%202.png)

> **Figure 2. CoT-VLA 프레임워크 개요**  
> VILA-U[67] 위에 모델 구축  
> VILA-U는 interleaved text-image 데이터로 pretrain된 생성형 multimodal 모델  
> 기본 모델은 이후 로봇 시연과 action-less video로 학습됨  
> 배포 시, 시각적 관찰과 텍스트 명령이 주어지면 모델은 causal attention을 사용하여 subgoal image를 생성하여 시각적 CoT reasoning을 수행  
> 이후, 로봇 실행을 위해 full attention을 사용하여 짧은 행동 sequence를 생성  
> 시스템은 예측된 행동 sequence를 실행한 후 새로운 관찰을 포착하는 closed-loop control 방식으로 작동

### 3.1. Visual Chain-of-Thought Reasoning

- VLA pretraining에 두 종류 훈련 데이터 고려
    1. 로봇 시연 데이터셋 $D_r$
        - $D_r = \{(l, \rm{a}_{1...T}, \rm{s}_{1...T}) \}$
        > $l$: 자연어 지시  
        > $\rm{a}_{1...T} = \{ \rm{a}_1, ..., \rm{a}_T \}$: 로봇 행동 sequence  
        > $\rm{s}_{1...T} = \{ \rm{s}_1, ..., \rm{s}_T \}$: 시각적 관측 이미지 sequence  
    2. action-less videos dataset $D_v$
        - $D_v = \{ (l, \rm{s}_{1...T}) \}$
            - language descriptions, 액션 주석이 없는 images로 구성
- **VLA**
    - vanilla VLA 접근
        - 사전학습된 VLM, $P_\theta$, $D_r$을 fine-tune
        - 현재 관찰 $\rm{s}_t$와 language 지시 $l$로부터 직접 행동 $\hat{\rm{a}}_{t+1}$을 예측하는 방법 학습
        $$
        \displaystyle
        \hat{\rm{a}}_t ~ P_\theta(\rm{a}_t | \rm{s}_t, l)
        \tag{1}
        $$
- **CoT_VLA**
    - 주요 insight: 행동 생성 전에 명시적인 visual reasoning을 통합
    - 두 개의 sequential phases 수행:
    $$
    \displaystyle
    \hat{\rm{s}_{t + n} ~ P_\theta(\rm{s}_{t + n} | \rm{s}_t, l)}
    \tag{2}
    $$
    $$
    \displaystyle
    \{ \hat{\rm{a}}_t, ..., \hat{\rm{a}}_{t + m} \} ~ P_\theta (\{\rm{a}_t, ..., \rm{a}_{t+m} | \rm{s}_t, l, \rm{s}_{t + n} \})
    \tag{3}
    $$
    1. 충간 추론 단계로서 $n$ 프레임 이후 subgoal image $\hat{\rm{s}}_{t+n}$을 예측(식 2.)
    2. subgoal을 달성하기 위해 $m$개의 action sequence 생성(식 3.)
    - 이는 모델이 시각적으로 생각하는 걸 가능케 함
        - 행동 예측 전, 명시적으로 바라는 미래 상태에 대해 reasoning
    - 식 2는 로봇 시연 $D_r$과 action-less video $D_v$ 모두에서 학습됨
    - 식 3은 로봇 시연 $D_r$에서만 학습됨



### 3.2 The Base Vision-Language Model

- VILA-U[67] 기반으로 구축
    - 이미지와 text token을 이해하고 생성할 수 있는 unified multimodal foundation model
    - autoregressive next-token 예측 프레임워크를 통해 video, image, language 이해 통합
    - visual input을 textual 정보와 정렬된 이산 token으로 encode하는 통합된 vision tower가 있음
        - 이산 visual features를 활용하는 VLM의 이해 능력을 크게 향상시키면서 autoregressive 이미지 & video 생성을 가능하게 함
    - residual quantization[32]를 활용하여 이산 visual feature의 표현 능력을 향상시킴
    - residual token을 예측하기 위해 depth transformer을 통합(RQ-VAE[32]에서 제안됨)
    - 추출된 visual feature은 LLM backbone에 의해 처리되기 전에 projector을 통해 전달됨
    - base model은 [image, text], [text, image], [video, text], [text, video]를 포함한 멀티 모달 쌍에서 학습됨
    - $256 \times 256$ 해상도로 훈련된 VILA-U 모델을 사용
    - 각 이미지는 $16 \times 16 \times 4$ 토큰으로 encode됨.

### 3.3 Training Procedures

![alt text](./images/Fig%203.png)

> **Figure 3. CoT-VLA 내 Hybrid attention mechanism**  
> image/text 생성을 위한 causal attention과 action 생성을 위한 full attention을 사용  
> $[x], [\theta], [g]$: actions의 병렬 decoding을 위한 special token

- base 7B VILA-U
    - robot demonstrations $D_r$과 action-less videos $D_v$를 조합하여 훈련
    - 훈련 중, vision tower을 고정한 채로 LLM backbone, projector, depth transformer 훈련
    - objective:
        1. causal attention을 사용하여 subgoal image 생성
        2. full attention을 사용하여 action 생성

**Visual Tokens Prediction**

- subgoal image 생성을 위해, 각 training sequence는 다음과 같은 형태를 가짐 $(l, \rm{s}_t, \rm{s}_{t + n})$
- 각 visual position $j$에 대해 depth transformer $P_\delta$는 LLM-generated code embedding $h_j$에 기반하여 $D$개의 residual tokens $(k_{j1}, ..., k_{jD})$를 autoregressively 하게 예측
- visual tokens에 대한 training objective:
$$
\displaystyle
\mathcal{L}_{\rm{visual}} = - \Sigma_j \Sigma_{d=1}^D \log P_\delta (k_{jd} | k_{j<d})
\tag{4}
$$
> $j$: visual token을 포함하는 위치

**Action Tokens Prediction**

- 행동 예측을 위해, 각 training sequence는 다음과 같은 형태를 가짐 $(l, \rm{s}_t, \rm{s}_{t + n}, \rm{a}_t, ..., \rm{a}_{t+m})$
- 각 행동 $\rm{a}_i$는 7개의 token으로 표현됨
- 각 행동의 차원은 독립적으로 이산화됨
- 각 연속적인 행동 차원을 256개의 이산 구역으로 매핑
- 구간의 너비: 학습 데이터의 행동 분포에서 1% 값과 99% 값 사이를 균등하게 나누어 결정

- text tokenizer의 vocabulary에서 가장 적게 사용되는 256개의 token을 행동 구간 token으로 재사용
- 행동 토큰을 처리하고 예측하기 위해 full attention을 사용
    - 모든 행동 token이 서로 상호작용할 수 있도록 함
- 훈련 중에는 행동 예측에 대한 cross-entropy loss를 최소화
$$
\displaystyle
\mathcal{L}_{\rm{action}} = - \Sigma_{i=1}^{m} \log P_\theta(\rm{a}_t ... \rm{a}_{t + m} | l, s_t, s_{t + n})
\tag{5}
$$

- input sequence 배치가 주어지면, 전반적인 훈련 objective는 action과 visual loss를 결합함
$$
\displaystyle
\mathcal{L} = \mathcal{L}_{\rm{action}} + \mathcal{L}_{\rm{visual}}
\tag{6}
$$

**Pretraining Phase**

- 로봇 시연 $D_r$과 action-less video $D_v$ 모두에서 CoT-VLA를 사전학습
- $D_r$
    - Open X-Embodiment 데이터셋[48]의 하위 집합 선별
    - Open VLA[29]에서 설정된 전처리 파이프라인을 따라, 3인칭 카메라 view와 single-arm end-effector control(7-DoF)를 갖는 데이터셋을 선택하고 처리
- $D_v$:
    - EPIC-KITCHENS[27] 및 Something-Something V2[20] 데이터셋을 통합
    - 모든 이미지는 $256 \times 256$ 해상도로 처리
    - 시각적 추론을 위해, 미래 시점 $n$에서 subgoal 이미지를 dataset-specific range $[n_l, n_u]$에서 균등하게 샘플링
        - $n_l, n_u$: prediction horizion의 하한과 상한
    - action chunk 크기: 10

**Adaptation Phase for Downstream Closed-Loop Deployment**

- downstream 작업에 적응하기 위해 pretrained model을 task-specific robot 시연 데이터 $D_r$을 사용하여 fine-tune
- vision tower을 고정한 채 LLM 백본, projector, depth transformer을 최적화
- pre-training 단계와 동일한 training setup 유지
- 모델이 자연어 명령에 기반하여 새로운 조작 작업을 수행할 수 있도록 함
- 테스트 시 로봇 제어 절차:

![alt text](./images/Algorithm%201.png)

## 4. Experiemnts

**실험의 목표**
- 여러 벤치마크와 구현에서 SOTA 기준과 비교했을 때의 성능
- pretraining, visual CoT reasoning, hybrid attention이 작업 성능에 미치는 영향
- 시각적 reasoning에서 향상된 일반화가 행동 예측 능력을 얼마나 강화하는지

### 4.1 Experimental Setup

- 세 가지 상호 보완적인 환경에서 평가 수행

- **LIBERO Simulation Benchmark[37]**
    - 네 가지 서로 다른 작업 모음으로 

- **Bridge-V2 Real-Robot Experiments[60]**

- **Franka-Tabletop Real-Robot Experiments**
    - 각 테스트 시나리오당 10~150개의 제한된 로봇 시연을 갖는 table-mounted Franka Emika Panda 로봇

![alt text](./images/Table%201.png)
> **Table 1. LIBERO 벤치마크 실험 결과**  
> 각 task suite(Spatial, Object, Goal, Long)에 대해 평균 성공률, 표준 오차를 보고  
> CoT-VLA는 모든 LIBERO 벤치마크 세트에서 baseline 접근 방식과 비교했을 때 가장 우수하거나 경쟁력있는 성능을 달성  
> 굵은 글씨: 가장 높은 성공률  
> 밑줄: 두 번째로 높은 성공률

