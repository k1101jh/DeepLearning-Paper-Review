# Lightglue: Local feature matching at light speed

---

- Feature Matching
- Visual Localization
- Camera Localization

---

url
 - [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)

---

목차

0. [Abstract](#abstract)
1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Conclusion](#5-conclusion)

---


## Abstract

**LightGlue**
 - 이미지 간의 local feature matching을 학습하는 Deep Neural Network
 - sparse matching에서 SOTA인 SuperGlue의 설계를 개선
 - 메모리 및 계산 측면에서 더 효율적이고, 더 정확하고 훈련하기 쉽게 만듦
 - 주요 특징: 문제의 난이도에 적응함
    - 추론은 직관적으로 일치하기 쉬운 이미지 쌍(시각적 중첩이 더 크거나 외관 변화가 제한적일 때) 빠름


## 1. Introduction

이미지 매칭의 일반적인 접근법
 - local visual appearance를 encoding하는 고차원 표현을 사용하여 매칭되는 sparse interest points 에 의존
 - 어려운 점
    - 대칭
    - 약한 texture
    - 외관 변화
 - 가려짐이나 누락된 포인트로 발생하는 이상치들을 거부하기 위해 이러한 표현들은 판별적이어야 함
 - 즉, 강인하며 독창적이어야 함

**SuperGlue**
 - 두 이미지를 동시에 고려해서 sparse points를 동시에 매칭하고 이상치를 거부하는 deep neural network 도입
 - transformer 모델을 사용하여 대규모 데이터세트에서 이미지쌍을 매칭하는 방법 학습
 - 실내 및 실외 환경에서 강인한 매칭을 제공
 - 도전적인 조건에서 visual localization에 효과적
 - 여러 작업에서 잘 일반화됨
    - aerial matching - 하늘에서 촬영한 이미지(항공 이미지) 매칭
    - 객체 자세 추정
    - 물고기 재식별
 - 계산 비용이 높지만 효율성이 좋음
 - 다른 transformer 기반 모델과 마찬가지로 훈련이 어렵고 많은 자원을 필요로 함

**LightGlue**

![alt text](./images/Fig%201.png)
> **Figure 1. LightGlue는 sparse features를 기존 방법보다 빠르고 정확하게 매칭**  
> adaptive stopping 메커니즘은 속도와 정확도 간의 균형을 세밀하게 조정할 수 있게 함  
> 일반적인 야외 조건에서 8배 더 높은 속도로 LoFTR에 가까운 정확도를 제공

 - SuperGlue보다 정확하고 효율적
 - 훈련이 더 쉬움


![alt text](./images/Fig%202.png)
> **Figure 2. Depth adaptivity**  
> 

 - 각 이미지 쌍의 난이도에 적응
 - 직관적으로 매칭하기 쉬운 쌍에서 훨씬 더 빠름
 - 매칭할 수 없는 점들을 초기에 버려 공동 가시(covisible) 영역에 집중

매칭 방법 
1. 각 계산 블록 이후 대응 관계 예측
2. 모델이 이를 스스로 검토하여 추가 계산이 필요한지 예측


## 2. Related work

**Matching images**
 - 고전 알고리즘은 수작업으로 만든 기준과 gradient 통계에 의존
 - 최근 연구는 detection과 description을 위한 CNN을 설계
    - CNN은 matching의 정확성과 강건성을 크게 향상시킴
 - local feature은 다양한 형태로 제공됨
    - 일부는 더 잘 localize됨
    - 높은 반복성
    - 저장 및 매칭 비용이 적음
    - 특징 변화에 불변
    - 신뢰할 수 없는 객체는 무시
 - local feature은 descriptor 공간에서 nearest neighbor search로 매칭됨
    - 매칭되지 않는 keypoint와 불완전한 descriptor 때문에 일부 대응은 올바르지 않음
    - Lowe's ratio test 또는 mutual check, inlier classifier, 강건하게 geometric model을 적합하는 등의 방법을 사용하여 필터링
        - 광범위한 도메인 전문 지식과 튜닝 필요
        - 조건이 너무 어려우면 실패하기 쉬움

**Deep Matcher**
 - 이미지 쌍을 입력받아 local feature을 동시에 매칭하고 이상치를 거부하도록 훈련
 - SuperGlue
    - Transformer의 expressive 표현과 최적의 운송을 결합하여 부분 할당 문제(partial assignment problem)을 해결.
    - 장면 기하학 및 카메라 움직임에 대한 강력한 prior을 학습  
        -> 극단적인 변화에 강함  
        -> 데이터 domain 전반에 잘 일반화 됨
    - 훈련이 어렵고, keypoint의 수에 따라 복잡도가 제곱으로 증가
 - 후속작업은 이를 더 효율적으로 만듦
    - 작은 seed match로 제한[7]하거나 유사한 keypoint의 클러스터 내로 제한[62]  
    -> 많은 수의 keypoint에 대한 실행 시간을 크게 줄이지만, 적은 수에서는 이점이 없음  
    -> 강인성이 낮아짐
 - LightGlue
    - SLAM과 같은 일반적인 작동 조건에서 큰 개선을 가져옴
    - 성능을 저하시키지 않음
    - 네트워크 크기를 동적으로 조정하여 달성
 - Dense matcher
    - LoFTR[65]나 후속 연구[8, 73]의 dense matcher은 sparse location이 아닌 dense grid에서 분포된 포인트를 매칭
    - 강건성을 높이지만 더 느림
    - 입력 이미지의 해상도에 따라 그에 따른 공간 정확도를 제한

**Making Transformers efficient**

~

## 3. Fast feature matching

**Problem formulation**

 - 이미지 $A$ 와 $B$에서 추출된 두 집합의 local feature 간의 partial assignment를 예측

 - 각 local feature $i$: 2D point 위치 $\text{p}_i := (x, y)_i \in {[0, 1]}^2$ 로 구성됨
    - 이미지 크기로 정규화됨
    - visual descriptor인 $\text{d}_i \in \mathbb{R}^d$를 포함

 - 이미지 $A$ 와 $B$는 각각 $M$, $N$ 개의 local feature을 가짐

 - LightGlue가 $M = {(i, j)} \subset A \times B$를 출력하도록 함
    - 각 point는 고유한 3D point에서 유래하므로 최소 한 번은 일치할 수 있음
    - 일부 point는 가려짐이나 비반복성(non-repeatability. 다른 이미지에서 등장하지 않는 의미인 듯)으로 인해 일치할 수 없음
 - $A$와 $B$의 local feature간의 soft partial assignment matrix $P \in [0, 1]^{M \times N}$을 찾고, 여기서 매칭 쌍을 추출할 수 있음

![alt text](./images/Fig%203.png)

> **Figure 3. LightGlue 아키텍처**  
> 입력 local features 쌍 $\text{d}, \text{p}$에 대해 각 layer은 self- 및 cross-attention unit과 positional encoding을 기반으로 visual descriptors(:red_circle:, :blue_circle:)을 강화.  
> confidence classifier $c$는 inference를 중단할 지 여부를 결정하는데 도움을 줌  
> 신뢰할 점이 많지 않으면 다음 레이어로 진행. 확실하게 매칭되지 않는 점들은 가지치기.  
> 신뢰 가능한 상태에 도달하면 점 간의 pairwise similarity와 하나만 매칭되는 가능성을 기반으로 assignment를 예측

### 3.1 Transformer backbone

이미지 $I \in (A, B)$의 각 local feature $i$를 상태 $\text{x}_i^I \in \mathbb{R}^d$과 연결.
 - 상태는 해당하는 시각적 descriptor $\text{x}_i^I \leftarrow \text{d}_i^I$로 초기화됨
 - 이후 각 layer에 의해 업데이트됨
 - layer은 self-attention unit 1개와 cross-attention unit 1개의 연속으로 정의됨

**Attention unit**
