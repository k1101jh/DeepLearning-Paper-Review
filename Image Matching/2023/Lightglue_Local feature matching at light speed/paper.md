# Lightglue: Local feature matching at light speed

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Feature Matching
- Visual Localization
- Camera Localization

---

url
- [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html) (ICCV 2023)

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

각 유닛에서 MLP는 원본 이미지 $S \in {A, B}$로부터 메시지 $m_i^{I \leftarrow S}$를 받아 상태를 업데이트함  

$$
\displaystyle
\begin{aligned}

& \text{x}_i^I \leftarrow \text{x}_i^I + MLP ([\text{x}_i^I | m_i^{I \leftarrow S}])
& (1)

\end{aligned}
$$

> $[\cdot | \cdot]:$ 두 개의 벡터를 stack. 두 이미지의 모든 점에 대해 병렬로 계산됨

- self-attention unit에서 각 이미지 I는 동일한 이미지의 점에서 정보를 가져오므로 $S = I$.
- cross-attention unit에서 각 이미지는 다른 이미지와 $S = \{A, B\} \backslash I$에서 정보를 가져옴

메시지는 이미지 $S$의 모든 상태 $j$의 가중 평균으로 attention mechanism으로 계산됨

$$
\displaystyle
\begin{aligned}

& m_i^{I \leftarrow S} = \sum_{j \in S} \text{Softmax}_{k \in S} (a_{ik}^{IS})_j \text{Wx}_j^S
& (2)

\end{aligned}
$$

> $W:$ projection matrix  
> $a_{ij}^{IS}:$ 이미지 I와 S의 점 i와 j 사이의 attention score. 이 점수는 self와 cross attention unit에서 계산 방법이 다르다


**Self-attention**

각 point는 동일한 이미지의 모든 point에 주의를 기울임


각 지점 $i$에 대해 현재 상태 $x_i$는 서로 다른 linear transformation을 통해 key vector $k_i$와 $q_i$로 분해됨
이후 point $i$와 $j$ 사이의 attention 점수를 정의(이미지 I를 나타내는 위첨자 생략)

$$
\displaystyle
\begin{aligned}

& a_{ij} = q_i^T R(p_j - p_i) k_j
& (3)

\end{aligned}
$$

> $R(\cdot) \in \mathbb{R}^{d \times d}:$ point 간의 상대 위치에 대한 회전 encoding[64].

 - 공간을 $d/2$개의 2D subspace로 나누고 각각을 학습된 기초 $b_k \in R^2$에 대한 projection에 따라 푸리에 feature을 따르는 각도로 회전

$$
\displaystyle
\begin{aligned}

& R(p) =
\begin{pmatrix}
\hat{\text{R}}(\text{b}_1^T \text{p}) & & 0  \\ 
& \ddots & \\  
0 & & \hat{\text{R}}(\text{b}_{d/2}^T \text{p})
\end{pmatrix}

, \hat{\text{R}}(\theta) = 
\begin{pmatrix}
cos \theta & -sin \theta \\
sin \theta & cos \theta
\end{pmatrix}

& (4)

\end{aligned}
$$

위치 인코딩
- 서로 다른 요소들을 해당 위치로 지정하게 해 줌
- projective camera geometry에서 시각 관측의 위치가 이미지 평면 내에서 카메라 이동에 대해 동등함
   - 동일한 fronto-parallel plane(카메라 뷰에 해당하는 평면)에 있는 3D point에서 나온 2D points는 동일한 방식으로 이동함. 상대적 거리는 일정하게 유지됨.  
    -> 점들의 절대 위치가 아닌 상대적인 위치만을 포착하는 인코딩이 필요
- 회전 인코딩은 모델이 위치 $i$에서 학습된 상대 위치로 재위치한 포인트 $j$를 검색할 수 있도록 함.
- 위치 인코딩은 값 $\text{v}_j$에 적용되지 않으므로 상태 $\text{x}_i$로 넘겨지지 않음
- 인코딩은 모든 레이어에 대해 동일하며 한 번 계산되어 캐시됨

**Cross-attention**
- $I$의 각 point는 다른 이미지 $S$의 모든 point에 주의를 기울임
- 각 요소에 대해 key $\text{k}_i$를 계산하지만 쿼리는 없음. 이는 다음과 같이 점수를 표현할 수 있게 함

$$
\displaystyle
\begin{aligned}

& a_{ij}^{IS} = \text{k}_i^{I\top} \text{k}_j^S \stackrel{!}{=} a_{ji}^{SI}.
& (5)

\end{aligned}
$$

**bidirectional attention**
- $I \leftarrow S$와 $S \leftarrow I$ 메시지에 대해 유사성을 한 번만 검사하는 트릭
- $O(NMd)$ 시간복잡도. 비용이 많이 드는 단계인데 bidirectional로 해서 $/2$ 만큼 비용 절약
- 상대 위치가 이미지 간에 의미가 없으므로 추가적인 위치정보를 추가하지 않음

### 3.2 Correspondence prediction

모든 레이어에서 업데이트 된 상태를 고려하여 매칭을 예측하는 경량 헤드 설계

**Assignment score(할당 점수, 또는 매칭 점수)**

1. 두 이미지의 point 간의 pairwise score matrix $S \in \mathbb{R}^{M \times N}$을 계산

$$
\displaystyle
\begin{aligned}

& \text{S}_{ij} = \text{Linear}(\text{x}_i^A)^\top \text{Linear}(\text{x}_j^B)
& \forall(i, j) \in A \times B 
& (6)

\end{aligned}
$$

> Linear $(\cdot):$ bias가 있는 학습된 linear transformation

- 이 점수는 각 point 쌍의 대응 관계(같은 3D point의 2D projection 간의 밀접한 정도)를 인코딩
- 각 점에 대해 매치 가능성 점수를 계산:

$$
\displaystyle
\begin{aligned}

& \sigma_i = \text{Sigmoid} (\text{Linear} (\text{x}_i)) \in [0,1]
& (7)

\end{aligned}
$$

- 이 점수는 해당 point가 대응점을 가질 가능성을 인코딩
- 다른 이미지에서 나오지 않는 경우(예: 가려져 있는 경우), $\sigma_i \rightarrow 0$

**Correspondences**

유사성 및 일치 가능성 점수를 soft partial assignment matrix $\text{P}$에 결합

$$
\displaystyle
\begin{aligned}

& \text{P}_{ij} = \sigma_i^A \sigma_j^B \text{Softmax}_{k \in A} (\text{S}_{kj})_i \text{Softmax}_{k \in B} (\text{S}_{ik})_j
& (8)

\end{aligned}
$$

- point 쌍 $(i, j)$는 두 point가 매치 가능성이 있다고 예측되고 두 이미지에서의 유사성이 다른 모든 점보다 높을 때 대응 관계를 생성
- $\text{P}_{ij}$가 임계값 $\tau$보다 크고 두 점의 행렬과 열에서 다른 모든 요소보다 큰 쌍을 선택


### 3.3 Adaptive depth and width

불필요한 계산을 피하고 추론 시간을 절약하는 두 가지 메커니즘
1. 입력 이미지 쌍의 난이도에 따라 레이어 수를 줄임
2. 확실하게 탈락한 point를 조기에 가지치기

**Confidence classifier**

입력 visual descriptor에 context를 추가
- 이미지 쌍이 쉽고(시각적 중첩이 많이 되어있고) 외관 변화가 적을 때 신뢰할 수 있음
- 이러한 경우 초기 레이어에서의 예측은 신뢰할 수 있고, 후반 레이어의 예측과 동일함
- 이 경우, 추론을 중단할 수 있음

각 레이어의 끝에서, 각 point의 예측 assignment에 대한 신뢰도를 추론

$$
\displaystyle
\begin{aligned}

& c_i = \text{Sigmoid} (\text{MLP}(\text{x}_i)) \in [0, 1]
& (9)

\end{aligned}
$$

높은 값은 i의 표현이 신뢰할 수 있고 최종적임을 나타냄
- 확실하게 매칭되는 경우 / 매칭되지 않는 경우
- compact MLP는 최악의 경우 추론 시간의 2%를 추가하지만 대부분의 경우 더 많이 절약함

**Exit criterion**

- 주어진 레이어 $\ell$에 대해, 한 point가 신뢰할 수 있다고 여겨지기 위한 조건은 $c_i > \lambda_\ell$.
- 모든 point의 충분한 비율 $\alpha$가 신뢰할 수 있는 경우, 추론을 중단

$$
\displaystyle
\begin{aligned}

& \text{exit} = \left( \frac{1}{N + M}  \sum_{I \in {A, B}} \sum_{i \in I} [[c_i^I > \lambda_\ell ]]  \right) > \alpha
& (10)

\end{aligned}
$$

- [59]에서와 같이, classifier가 초기 layer에서 덜 확신함을 관찰  
-> 각 classifiere의 검증 정확도에 따라 layer 전반에 걸쳐 $\lambda_\ell$을 감소시킴
- 종료 임계값 $\alpha$는 정확도와 추론 시간 사이의 균형을 직접적으로 조절

**Point pruning**

![alt text](./images/Fig%204.png)

> 

- 종료 기준이 충족되지 않은 경우, 확신이 있고 매칭이 불가능한 것으로 예측된 point는 후속 레이어에서 매칭에 도움이 될 가능성이 낮음
- 각 레이어에서 이를 제거하고 나머지 point만 다음 레이어로 전달
- attention의 quadratic 복잡성을 고려할 때 계산량을 크게 줄이고 정확도에 영향을 미치지 않음

### 3.4 Supervision

LightGlue를 두 단계로 훈련
1. correspondence를 예측하도록 훈련
2. confidence classifier을 훈련

후자는 최종 레이어의 정확도나 훈련의 수렴에 영향을 미치지 않음

**Correspondences**

- 두 개의 view 변환에서 추정된 실제 label을 사용하여 assignment matrix $\text{P}$를 감독(supervise)
- homography 또는 pixel 단위 깊이와 상대 자세가 주어지면, A에서 B로, 그리고 반대로 point들을 wrap.
- GT matches $M$은 두 이미지 모두에서 낮은 재투영 오차와 일관된 depth를 가진 point 쌍
- 일부 점 $\bar{A} \subseteq A$와 $\bar{B} \subseteq B$는 다른 모든 점들과 비교하여 재투영 또는 깊이 오차가 충분히 클 경우 매칭 불가능한 것으로 label이 지정됨.
- 각 층 $\ell$에서 예측된 할당의 log-likelihood를 최소화하여 LightGlue가 초기에 올바른 대응을 예측하도록 유도

$$
\displaystyle
\begin{aligned}

loss = -\frac{1}{L} \sum_\ell \left( \frac{1}{|\mathcal{M}|} log^\ell \right)




\end{aligned}
$$

loss는 positive & negative label 사이 균형을 이룸

**Confidence classifier**

- 식 (9)의 MLP를 훈련시켜 각 레이어의 예측이 최종 예측과 동일한지 예측
- $^\ell m_i^A \in \mathbb{B} \cup \{\bullet\}$ 을 레이어 $\ell$에서 $i$에 맞춰진 $B$의 point index로 두고, $i$가 매칭될 수 없는 경우 $^\ell m_i^A = \bullet$ 로 설정
- 각 지점의 GT binary label은 $[[ ^\ell m_i^A = ^L m_i^A]]$이며, $B$에 대해서도 동일함
- 이후 layer $\ell \in \{1, \dots, L - 1 \}$의 classifier들의 binary cross-entropy를 최소화

### 3.5 SuperGlue와 비교

LightGlue는 SuperGlue에서 영감을 받았지만 정확성, 효율성, 훈련 용이성이 더 뛰어나다

**Positional encoding**

- SuperGlue
   - MLP로 절대 위치를 인코딩하고 이를 초기에 descriptor과 융합
   - layer을 거치면서 위치 정보를 잊어버리는 경향이 있음
- LightGlue
   - 이미지 간에 더 잘 비교할 수 있는 상대 인코딩에 의존
   - 각 self-attention unit에 추가됨
   - 더 깊은 layer의 정확도가 향상됨

**Prediction head**

- SuperGlue
   - Sinkhorn 알고리즘을 사용하여 미분 가능한 최적 수송 문제를 해결하여 assignment를 예측
      - 행 방향 및 열 방향 정규화의 많은 반복으로 구성
      - 계산 및 메모리 비용이 많이 듦
   - 일치할 수 없는 점을 거부하기 위해 쓰레기통 추가
   - 쓰레기통이 모든 점의 유사성 점수를 얽히게 함
      - 덜 최적의 훈련 동적을 초래
- LightGlue
   - 유사성과 매치 가능성을 분리
      - 더 깨끗한 gradient 생성

**Deep supervision**

- SuperGlue
   - 각 layer 후에 예측을 할 수 없고 마지막 레이어에서만 감독됨
      - Sinkhorn 비용 때문
- LightGlue
   - 가벼운 head는 각 layer에서 할당을 예측하고 감독할 수 있게 함
   - 수렴 속도를 높이고 임의의 레이어 후에 추론을 종료할 수 있게 함


## 4. Details that matter

**Recipe**

- SuperGlue의 supervised training 설정을 따름
   - 100만 개 이미지에서 샘플링한 synthetic homographies(합성된 대응 관계?)를 사용하여 모델을 사전 훈련
   - 이러한 증강은 완전하고 noise-free 감독을 제공하지만 신중한 조정이 필요
- 이후 MegaDepth 데이터셋으로 미세조정
   - 196개의 관광 명소를 묘사한 100만개 crowd-sourced 이미지 포함
   - 카메라 캘리브레이션과 제사는 SfM으로 복구됨
   - dense depth는 다중 뷰 스테레오로 복구됨

**Training tricks**

![alt text](./images/Fig%205.png)

> **Figure 5. 훈련이 쉬움**

LightGlue 아키텍처의 세부 사항이 큰 영향을 미침(Fig 5 참조)
- AUC-RANSAC과 AUC-DLT가 SuperGlue에 비해 모델 훈련에 필요한 자원을 줄임
   - 훈련 비용을 줄이고 깊은 matcher을 많은 사람들이 활용하기 쉽게 만듦

