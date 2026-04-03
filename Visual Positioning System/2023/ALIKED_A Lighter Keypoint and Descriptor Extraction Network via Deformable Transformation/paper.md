# ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- Keypoint Extraction
---
url:
- [paper](https://ieeexplore.ieee.org/abstract/document/10111017)
- [arXiv](https://arxiv.org/abs/2304.03608)

---
- **Authors**: Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter C. Y. Chen, Qingsong Xu, Zhengguo Li
- **Venue**: IEEE Transactions on Instrumentation and Measurement (TIM) 2023

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [3. Network Architecture of ALIKED](#3-network-architecture-of-aliked)
  - [A. Feature Encoding](#a-feature-encoding)
  - [B. Feature Aggregation](#b-feature-aggregation)
  - [C. Differentiable Keypoint Detection](#c-differentiable-keypoint-detection)

---

## ⚡ 요약 (Summary)


---

## 📖 Paper Review

## Abstract

기존의 convolutional 연산은 descriptor에 필요한 기하학적 불변성을 제공하지 않음  
-> 각 keypoint에 대한 supporting features의 deformable(변형 가능한) position을 학습하고, deformable descriptor을 구성하는 sparse deformable descriptor head(SDDH)를 제안

SDDH
- dense descriptor map 대신 sparse keypoints에서 descriptor을 추출하여 강력한 표현력을 가진 descriptor을 효율적으로 추출할 수 있게 함
- 추출한 sparse descriptors를 훈련하기 위해 neural reprojection error(NRE) 손실을 dense에서 sparse로 완화함

실험 결과는 제안된 네트워크가 image matching, 3D reconstruction 및 visual relocalization 등의 다양한 작업에서 효율적이고 강력함을 보임

## 1. Introduction

효율적이고 강인한 이미지 keypoint & descriptor 추출은 SLAM, computational photography, visual place recognition과 같은 많은 자원이 제한된 cisual measurement application에 매우 중요
- 초기의 keypoint 감지 및 descriptor 추출 방법은 인간의 heuristics에 의존
- 이러한 수작업 방식은 효율적이지 않고, 강력하지 않음
- DNN 기반 데이터 중심 접근 방법
    - 초기에는 미리 정의된 keypoint에서 image patch의 descriptor을 추출
    - 이후 단일 네트워크를 사용한 keypoint & descriptor 추출 방식으로 발전
    - 이를 지도 기반 방법(map-based methods)라고 함
        - 두 개의 헤드(SMH, DMH)를 사용하여 score map과 descriptor map을 추정
        - 이후 socre map과 descriptor map에서 각각 keypoint와 descriptor을 추출

## 2. Related Works


## 3. Network Architecture of ALIKED

### A. Feature Encoding
Feature Encoder
- 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$을 4개의 encoding block을 사용하여 $F_1, F_2, F_3, F_4$로 변환.
(각 block은 $c_1$에서 $c_4$까지의 채널 범위를 가짐)
- 첫 번째 블록은 Fig. 1에 표시된 것처럼 low-level image features $F_1$을 추출하는 두 개의 convolution으로 구성됨
- 두 번째 블록은 $2 \times 2$ average pooling을 사용하여 $F_1$을 downsampling(더 큰 수용 영역 커버, 계산 효율성 높이기 위해)
- 세 번째, 네 번째 블록은 $4 \times 4$ average pooling을 사용하여 feature을 downsampling한 후, residual block과 함께 $3 \times 3$ DCN(Deformable Convolution Network)을 사용하여 이미지 feature을 추출
- ReLU 대신 SeLU(self-normalizing neural networks) 활성화 함수 사용(수렴을 개선하기 위해)

### B. Feature Aggregation

- localization 및 representation 능력을 위해 MS features {$F_1, F_2, F_3, F_4$}를 집계하는 역할을 함
- 이러한 feawture을 집계하기 위해 4개의 ublock 사용(Fig 1. 참조)
- 각 ublock은 $1 \times 1$ convolution과 upsample layer으로 구성되어 MS 특징의 차원과 해상도를 맞춤
- 이러한 정렬된 features {$F_1, F_2, F_3, F_4$}를 연결하여 keypoint 및 descriptor 추출을 위한 집계된 feature $F$를 얻음

### C. Differentiable Keypoint Detection

- keypoint 감지를 위해 SMH(Score Map Head)는 집계된 feature $F$를 사용하여 score map $S \in R^{H \times W}$을 추정
- SMH는 $1 \times 1$ convolution layer을 사용하여 feature channel을 8개로 줄인 다음, 두 개의 $3 \times 3$ convolution layer을 통해 feature encoding을 수행
- 마지막으로, $3 \times 3$ 합성곱 계층과 sigmoid layer을 사용하여 score map $S$를 얻음