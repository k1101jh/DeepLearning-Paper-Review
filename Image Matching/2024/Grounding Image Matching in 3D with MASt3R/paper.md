#  Grounding Image Matching in 3D with MASt3R

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- Image Matching
- 3D Reconstruction
---
url:
- [paper](https://link.springer.com/chapter/10.1007/978-3-031-73220-1_5)
- [arXiv](https://arxiv.org/abs/2406.09756)
---
- **Authors**: Vincent Leroy, Yohann Cabon, Jérôme Revaud, Boris Chidlovskii, et al.
- **Venue**: ECCV 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related works](#2-related-works)
- [3. Method](#3-method)
  - [3.1 The DUSt3R framework](#31-the-dust3r-framework)
  - [3.2 Matching prediction head and loss](#32-matching-prediction-head-and-loss)
  - [3.3 Fast reciprocal matching](#33-fast-reciprocal-matching)
  - [3.4 Coarse-to-fine matching](#34-coarse-to-fine-matching)
- [4. Experimental results](#4-experimental-results)
- [5. 결론](#5-결론)

---

## ⚡ 요약 (Summary)
- **Problem**: 기존 이미지 매칭은 주로 2D 문제로 다루어졌으나, 실제 매칭은 본질적으로 3D 기하학적 문제임. DUSt3R은 3D 재구성에는 강력하지만 매칭 정확도가 다소 제한적이었음.
- **Goal**: DUSt3R 프레임워크를 기반으로 매칭 기능을 강화하여, 극단적인 시점 변화에서도 정확하고 조밀한 대응 관계(Dense Correspondences)를 추출하는 MASt3R 제안.
- **Key Method**: 
    - **Matching Head**: 3D 포인트 회귀와 함께 미세한 매칭을 위한 밀집 로컬 특징(Local Features) 맵을 출력하는 헤드 추가 (InfoNCE loss로 학습).
    - **Fast Reciprocal Matching**: 고차원 특징 공간에서의 상호 근접 이웃(Mutual NN) 검색의 복잡도($O(N^2)$)를 서브샘플링과 반복 전파를 통해 획기적으로 개선함.
    - **Coarse-to-fine Matching**: 고해상도 이미지 처리를 위해 저해상도 매칭 후 중요 윈도우 쌍을 탐욕적으로 선택하여 미세하게 최적화하는 전략 도입.
- **Result**: Map-free localization, Relative pose estimation 등 극도로 도전적인 벤치마크에서 SOTA를 크게 경신함.

![image.png](./images/Fig%202.png)

- DUSt3R 프레임워크를 기반으로 함. 파란색 부분이 MASt3R의 기여
- 이미지 매칭을 3D 문제로 간주
- 빠른 상호 매칭
    - 두 개의 feature map이 주어졌을 때 pixel correspondences(mutual nearest neighbor) 추출
    - 전체 픽셀에 대해 진행하면 $O(W^2H^2)$를 가지게 됨
    - 제안 방법
        1. 첫 번째 이미지에서 격자 형식으로 규칙적으로 샘플링($U^0$)
        2. 두 번째 이미지로 NN 매핑하여 $V^1$생성
        3. 이를 다시 첫 번째 이미지로 NN 매핑하여 $U^{t+1}$생성. 이를 반복.
        이 과정에서 $U^t$와 $U^{t+1}$에서 일치하는 correspondence(수렴한 경우)는 제외하고 반복
        - 몇 번의 iteration 후 수렴되지 않은 점이 급격히 감소함
- Coarse-to-fine 매칭
    - MASt3R은 최대 512픽셀인 이미지만 처리함
    - 더 큰 이미지의 경우
        1. 다운스케일링 버전에서 매칭
        2. 각 이미지의 전체 해상도에서 겹치는 window crop을 생성
        3. window 쌍을 선택.
        선택 시에는 1.의 correspondence를 포함하는 조합으로 선택
        대응 관계의 90%가 포함될 때까지 window pair을 하나씩 탐욕적으로 추가
        4. 각 window 쌍에 대해 독립적으로 매칭을 수행
        5. 각 window 쌍에서 얻은 correspondence를 원본 이미지 좌표로 매핑
- 실험 결과
    ![image.png](./images/Table%202.png)
    
    - Map-free dataset 테스트 결과, 모든 SOTA 방식을 크게 능가함
    - direct reg는 이미지 매칭 없이 point map에서 PnP 적용(2D-3D 정보가 있으므로)
        - PnP나 essential matrix와 같은 intrinsic을 활용한 pose estimation이 일반적으로 더 안전

---

## 📖 Paper Review

## Abstract

![alt text](./images/Fig%201.png)

> **Figure 1. Dense Correspondences**  
> MASt3R은 카메라 움직임이 시각적 유사성을 상당히 감소시키는 지역에서도 밀집된 대응 관계를 예측  
> focal length는 예측된 3D 기하학에서 파생될 수 있음  
> 카메라 보정, 카메라 위치 추정 및 3D scene 재구성을 위한 독립적인 방법으로 접근 방식을 만들고, 여러 극도로 도전적인 벤치마크에서 SOTA를 달성하고 개선함


이미지 매칭
- 3D vision 알고리즘의 핵심 구성 요소
- 매칭은 3D 문제이지만 일반적으로 2D 문제로 다뤄짐
- 본 논문에서는 매칭을 Transformer 기반의 3D 재구성 프레임워크인 DUSt3R을 사용하여 3D 작업으로 제안
    - DUSt3R
        - pointmap 회귀에 기반하여 극단적인 viewpoint 변화가 있는 뷰를 매칭하는데 탁월한 강인성을 보임
        - 정확도는 제한적
    - 추가 matching loss로 훈련된 밀집 local feature을 출력하는 새로운 헤드로 DUSt3R 네트워크 증강을 제안
    - dense matching의 이차 복잡성 문제를 해결
    - 빠른 쌍방 매칭 scheme을 도입
        - 매칭을 수십 배속으로 가속화하며 이론적인 보장을 제공
        - 개선된 결과 제공
    - MASt3R가 여러 매칭 작업에서 SOTA를 크게 초월함
    - 도전적인 Map-free localization 데이터셋에서 VCRE AUC에서 가장 뛰어난 방법보다 30%(절대 지표) 더 뛰어남

## 1. Introduction

이미지 매칭
- 관련 3D 비전 애플리케이션
    - mapping
    - localization
    - navigaion
    - photogrammetry
    - 자율 로봇 공학
- Visual Localization의 SOTA 방법은
    - offline mapping 단계:에서 COLMAP을 사용하는 것과 같이 이미지 매칭에 압도적으로 의존.
    - online localization: 일반적으로 PnP를 사용

제안 방법
- 목적:
    - 두 장의 이미지가 주어졌을 때 일치하는 쌍의 리스트를 생성
    - 시점과 조명 변화에 견딜 수 있는 정확하고 조밀한 위치 출력

전통적 매칭 방법
- 파이프라인
    1. 희소하고 반복 가능한 keypoint 추출
    2. 이를 지역적으로 불변한 feature로 설명
    3. feature space에서 거리를 비교해서 이산 keypoint 집합을 쌍으로 연결
- 장점
    - keypoint 탐지기는 낮거나 중간 조명 및 시점 변화에서 정확
    - keypoint의 희소성은 문제를 계산적으로 처리 가능하게 하여 밀리초 이내에 정밀한 매칭을 가능하게 함
- 단점
    - 매칭을 bag-of-keypoint 문제로 축소
        - matching 작업의 전역 기하학적 맥락을 버리게 됨
    - 반복 패턴이나 텍스처가 적은 영역의 상황에서 오류가 발생하기 쉬움
- 해결 방법
    - 매칭에 대한 학습된 사전 정보를 활용하여 pairing 단계에서 global 최적화 전략 도입
        - SuperGlue 등의 방법이 이를 성공적으로 구현
        - keypoint와 그 descriptor가 충분한 정보를 인코딩하지 않는다면 매칭 중 전역 context를 활용하는 건 너무 늦을 수 있다.
    - dense 전체적인 매칭 고려
        - keypoint를 완전히 피하고 전체 이미지를 한 번에 매칭
        - global attention을 위한 매커니즘이 등장하며 가능해짐
        - LoFTR과 같은 접근 방식은 이미지를 전체로 고려하여 생성된 대응 집합은 밀집해있고 반복 패턴 및 low-texture 영역에 더 강력함
        - Map-free localization 벤치마크와 같은 도전적인 벤치마크에서 SOTA를 달성
        - LoFTR과 같은 방법조차 Map-free localization 벤치마크에서 상대적으로 실망스러운 정확도 34%를 기록

모든 매칭 접근법이 매칭을 이미지 공간의 2D 문제로 다루고 있기 때문.
- 매칭은 본질적으로 3D 문제
- 일치하는 픽셀은 같은 3D 점을 관찰하는 픽셀
- 2D pixel correspondence와 3D 공간에서의 상대 카메라 포즈는 epipolar 행렬에 의해 직접적으로 연결됨
- Map-free 벤치마크에서 최상위 성능을 기록한 DUSt3R은 매칭이 아닌 3D 재구성을 위해 처음 설계된 방법
- 현재 3D 출력에서 단순히 얻은 correspondences들은 Map-free 벤치마크에서 모든 다른 keypoint 및 매칭 기반 방법보다 우수한 성능을 보임

MASt3R(Matching And Stereo 3D Reconstruction)
- DUSt3R가 매칭에 사용될 수 있음
    - viewpoint 변화에는 매우 강건하지만 상대적으로 부정확함
    - 이를 보완하기 위해 dense local feature map을 회귀하는 두 번째 헤드를 부착하고 InfoNCE loss로 훈련을 제안
- 여러 벤치마크에서 DUSt3R을 능가함
- pixel-accurate 매칭을 얻기 위해 여러 scale에서 매칭이 수행되는 coarse-to-fine 매칭 scheme을 제안
- 각 매칭 단계는 dense feature map에서 상호 매칭을 추출하는 과정을 포함
    - 이는 직관에 반하며, dense feature map 자체를 계산하는 것보다 더 많은 시간을 소모
    - 상호 매칭을 찾기 위한 더 빠른 알고리즘 제안. 약 두 배의 속도로 포즈 추정 품질을 향상시킴

논문의 기여
1. MASt3R 제안. DUSt3R 프레임워크를 기반으로 하며 정확하고 강력한 매칭을 가능하게 하는 feature map 출력
2. 고해상도 이미지에서 작동할 수 있도록 빠른 매칭 알고리즘과 관련된 coarse-to-fine 매칭 방식 제안
3. MASt3R은 여러 절대 및 상대 자세 localization 벤치마크에서 SOTA를 크게 능가

## 2. Related works


## 3. Method

![alt text](./images/Fig%202.png)

> **Figure 2. 제안 방법의 개요**  
> 매칭할 두 입력 이미지가 주어지면 네트워크에서 각 이미지와 입력 픽셀에 대해 3D point, 신뢰도, local feature을 회귀  
> 3D point 또는 local feature을 fast reciprocal NN matcher에 입력하면 강건한 대응을 얻을 수 있음
> DUSt3R과 비교하여 기여를 파란색으로 강조

카메라 파라미터가 알려지지 않은 두 카메라 $C^1$, $C^2$로부터 촬영된 두 이미지 $I^1$과 $I^2$가 주어졌을 때 픽셀 대응 집합 $\{(i, j)\}$를 복원
- $i, j$: 픽셀 $i = (u_i, v_i)$, $j = (u_j, v_j) \in \{1, \dots, W\} \times \{1, \dots, H\}$
- 두 이미지는 동일한 해상도를 가진다고 가정 (하지만 그렇다고 일반성을 잃지는 않음)
    - 최종 네트워크는 다양한 종횡비의 이미지 쌍을 처리 가능
- 두 장의 입력 이미지를 바탕으로 3D 장면 복원과 매칭을 동시에 수행하고자 함
    - DUSt3R 프레임워크에 기반함

### 3.1 The DUSt3R framework

DUSt3R
- 이미지로부터 calibration과 3D reconstruction 문제를 동시에 해결
- transformer-based 네트워크를 사용하여 두 입력 이미지로부터 local 3D reconstruction을 수행
    - 두 개의 dense 3D point-clouds $X^{1, 1}$과 $X^{2, 1}$로 표현되며, 이후 pointmaps로 칭함

- Pointmap $X^{a,b} \in \mathbb{R}^{H \times W \times 3}$는 다음 둘 사이의 dense 2D-to-3D 매핑을 나타냄
    - 이미지 $I^a$의 각 픽셀 $i = (u, v)$
    - 그에 대응되는 3D 점 $X_{u,v}^{a,b} \in \mathbb{R}^3$ (카메라 $C^b$ 좌표계로 표현됨)

- DUSt3R는 두 pointmap $X^{1,1}$과 $X^{2,1}$를 동일한 좌표계 (카메라 $C_1$ 기준)로 회귀함으로써 캘리브레이션 문제와 3D 재구성 문제를 동시에 해결

- 두 장 이상의 이미지가 주어질 경우, global alignment의 두 번째 step을 통해 모든 pointmap을 같은 좌표계로 병합할 수 있음
    - 본 논문에서는 이 단계를 사용하지 않고, 이안(binocular) 설정에만 집중함.


1. 두 이미지는 Siamese 방식으로 ViT로 encoding됨

$$
\displaystyle
\begin{aligned}

&H^1 = \text{Encoder}(I^1)  &(1) \\
&H^2 = \text{Encoder}(I^2)  &(2)

\end{aligned}
$$

2. 두 표현은 얽힌(intertwined) 디코더에 의해 공동 처리됨
    - scene의 viewpoint와 global 3D geometry 간의 공간 관계를 이해하기 위해 cross-attention으로 정보를 교환

$$
\displaystyle
\begin{aligned}

&H'^1, H'^2 = \text{Decoder}(H^1, H^2)  &(3)

\end{aligned}
$$

3. 두 prediction head는 encoder과 decoder 출력을 concatenate하여 최종 pointmaps와 confidence maps를 회귀

$$
\displaystyle
\begin{aligned}

&X^{1,1}, C^1 = \text{Head1}^1_{3D}([H^1, H'^1])  &(4) \\
&X^{2,1}, C^2 = \text{Head2}^1_{3D}([H^2, H'^2])  &(5)

\end{aligned}
$$

**Regression loss**
- DUSt3R은 간단한 regression loss를 사용하여 fully-supervised 방식으로 학습됨


$$
\displaystyle
\begin{aligned}

&\ell_{\text{regr}}(v, i) = \parallel\frac{1}{z} X^{v,1}_i - \frac{1}{\hat{z}}\hat{X}^{v,1}_i\parallel  &(6)

\end{aligned}
$$

> $v \in {1, 2}$: view  
> $i$: GT 3D point $\hat{X}^{v, 1} \in \mathbb{R}^3$이 주어졌을 때 픽셀

원래 공식에는 normalizing factor $z, \hat{z}$를 사용해서 재구성이 scale에 대해 불변하도록 함
- 이들은 단순히 모든 유효 3D point의 원점에 대한 평균 거리를 정의한 것.

**Metric predictions**

scale invariance는 반드시 바람직하지는 않음
- map-free visual localization과 같은 일부 사용 사례에서는 metric scale 예측이 필요
- GT pointmap이 metric으로 알려져 있을 때 예측된 pointmap에 대한 정규화를 무시하도록 regression loss를 수정
- GT가 metric일 때, $z := \hat{z}$로 설정하여 $\ell_{\text{regr}}(v, i) = \frac{||X^{v,1}_i - \hat{X}^{v,1}_i||}{\hat{z}}$ 가 됨
- 최종 confidence-aware regression loss는 다음과 같이 정의됨

$$
\displaystyle
\begin{aligned}
& L_{\text{conf}} = \sum_{v \in \{1,2\}} \sum_{i \in V^v} C_i^v \ell_{\text{regr}}(v, i) - \alpha \log C_i^v
& (7)
\end{aligned}
$$

### 3.2 Matching prediction head and loss

pointmap에서 신뢰할 수 있는 pixel correspondences를 얻기 위한 일반적인 방법
- 불변 feature space에서 상호 일치 여부를 찾는 것
    - 극단적인 시점 변화가 존재할 때도 DUSt3R의 regressed pointmaps(3차원 공간)과 잘 작동
    - 이로 얻은 correspondence가 다소 부정확함
    - 원인:
        1. regression은 본질적으로 noise의 영향을 받음
        2. DUSt3R은 매칭을 위해 명시적으로 훈련된 적이 없음

**Matching head**

- 두 개의 dense feature map $D^1$ 및 $D^2 \in \mathbb{R}^{H \times W \times d}$를 출력하는 두 번째 헤드 추가

$$
\displaystyle
\begin{aligned}

& D^1 = \text{Head}^1_{\text{desc}}([H^1, H'^1]) &(8) \\
& D^2 = \text{Head}^2_{\text{desc}}([H^2, H'^2]) &(9)

\end{aligned}
$$

- 간단한 2-layer MLP로 구현
    - non-linear GELU 활성화 함수가 중첩됨
- 마지막으로 각 local feature을 unit norm으로 정규화

**Matching objective**
- 두 이미지간의 descriptor matching(같은 3D point를 가리키는)
- 실제 GT correspondence를 기준으로 infoNCE[95] loss를 활용

$$
\displaystyle
\begin{aligned}

& L_{\text{match}} = -\sum_{(i, j) \in \hat{M}} \log \frac{s_\tau(i, j)}{\sum_{k \in \mathcal{P}_1} s_\tau(k, j)} + \frac{s_\tau(i, j)}{\sum_{k \in \mathcal{P}^2} s_\tau(i, k)}
& (10)
\\
& \text{with} \quad \ s_\tau(i, j) = \exp(-\tau D_i^{1 \top} D_j^2)
& (11)

\end{aligned}
$$

> $\mathcal{P}_1 \{ i | (i, j) \in \hat{\mathcal{M}} \}$ 및 $\mathcal{P}_2 \{ j | (i, j) \in \hat{\mathcal{M}} \}$: 각 이미지에서 고려된 pixel의 하위 집합  
> $\tau$: 온도 hyper parameter

matching 목표는 본질적으로 cross-entropy classification loss
- 식 6의 회귀와 달리, 올바른 픽셀을 맞추었을 때만 보상을 받고 근처 pixel에 대해서는 보상이 잆음
    - 네트워크가 높은 정밀도를 달성하도록 유도


최종 훈련 목표:

$$
\displaystyle
\begin{aligned}

& L_{\text{total}} = L_{\text{conf}} + \beta L_{\text{match}}
& (12)

\end{aligned}
$$


### 3.3 Fast reciprocal matching

![alt text](./images/Fig%203.png)

> **Figure 3. 빠른 상호 매칭**  
> 좌: fast matching process
>   - 초기 pixel 하위 집합 $U^0$에서 시작해서 NN 검색을 사용하여 이를 반복적으로 전파하는 빠른 매칭 프로세스
>   - 사이클(파란 화살표)를 검색하면 상호 correspondences를 감지하고 수렴한 점을 제거하여 이후 단계를 가속화 할 수 있음
> 중간: 반복 $t=1 ... 6$에서 $U^t$의 남은 점의 평균 수
>   - 5회의 반복 후, 거의 모든 점이 상호 매칭으로 수렴
> 우: Map-free dataset에서 성능과 시간의 절충
>   - 성능은 실제로 개선되며 중간 레벨의 subsampling을 수행할 때 매칭 속도도 함께 향상됨


두 개의 예측된 feature map $D^1, D^2 \in \mathbb{R}^{H \times W \times d}$ 이 주어질 때, 신뢰할 수 있는 pixel correspondences(서로의 mutual nearest neighbors)를 추출하는 것을 목표로 함

$$
\displaystyle
\begin{aligned}

\mathcal{M} = \{(i, j) \mid j = \text{NN}_2(D_i^1) \text{ and } i = \text{NN}_1(D_j^2)\} &
& (13)
\\
\text{with} \quad \text{NN}_A(D_j^B) = \arg\min_i ||D_i^A - D_j^B|| &
& (14)

\end{aligned}
$$

- 쌍방 매칭의 단순 구현은 $O(W^2H^2)$라는 높은 계산 복잡성을 가짐
    - 이미지의 각 픽셀이 다른 이미지의 모든 픽셀과 비교되어야 하기 때문
    - NN(nearest neighbor)을 최적화하는 것은 가능하지만(예: K-d tree 사용) 이러한 최적화는 고차원 feature space에서 일반적으로 매우 비효율적이 되고, 모든 경우에서 MASt3R가 $D^1$과 $D^2$를 출력하는 추론 시간보다 수십 배 느림

**Fast matching**

- sub-sampling을 기반으로 한 빠른 접근 방식 제안
- 초기 sparse set of $k$ pixels $U^0 = \{U^0_n\}^k_{n=1}$ 에서 시작하는 반복 과정을 기반으로 함
- 일반적으로 첫 번쨰 이미지 $I^1$에서 격자에 규칙적으로 샘플링됨
- 각 픽셀은 $I^2$에서 NN으로 매핑되어 $V^1$을 생성하고, 결과 픽셀은 동일 방식으로 $I^1$에 다시 매핑됨:

$$
\displaystyle
\begin{aligned}

& U^t \longmapsto[{\text{NN}_2(D_u^1)}]_{u \in U^t} \equiv V^t \longmapsto[{\text{NN}1(D_v^2)}]_{v \in V^t} \equiv U^{t+1}
& (15)

\end{aligned}
$$

상호 매치 집합이 모아짐(사이클을 형성하는. 예: $M_t^k = \{(U_t^n, V_t^n) \mid U_t^n = U_{t+1}^n\}$)
- 다음 iteration에서는 이미 수렴된 pixel이 필터링됨(즉, $U^{t+1} := U^{t+1} \setminus U^t$ 업데이트)
- 마찬가지로, $t=1$ 부터 시작해서 $V^{t+1}을 검증하고 필터링하여 유사한 방식으로 $V^t$와 비교
- 대부분 correspondence가 안정된 쌍으로 수렴할 때까지 고정된 횟수만큼 반복(그림 3(좌) 참조)
- 몇 번의 iteration 후 수렴되지 않은 점 $U^t$가 0으로 급격히 감소(그림 3(중앙) 참조)
- output set은 모든 상호 쌍들을 병합하여 구성. $\mathcal{M}_k = \bigcup_t \mathcal{M}_k^t$

**이론적 보증**

- 빠른 매칭의 전반적인 복잡성: $O(kWH)$
    - naive 방식보다 $WH/k \gg 1$배 빠름(그림 3(오른쪽) 참조)
    - 제안한 빠른 일치 알고리즘은 전체 집합 $\mathcal{M}$의 부분 집합을 추출하며, 이는 $|\mathcal{M}_k| \leq k$ 크기를 가짐
    - 보충 자료에서 알고리즘의 수렴 보장과 이상치 필터링 속성을 입증하는 방법 연구. 전체 대응 집합 M을 사용할 때보다 최종 정확도가 더 높은 이유를 설명함(그림 3(오른쪽) 참조)

### 3.4 Coarse-to-fine matching

attention 연산은 입력 이미지 영역 $(W \times H)$에 대해 2차 복잡도를 가짐.
- MASt3R은 최대 크기가 512픽셀인 이미지만 처리
- 더 큰 이미지는 학습을 위해 상당히 많은 양의 컴퓨팅 파워를 요구
- ViT는 더 큰 test 시간 해상도로 일반화되지 않음
- 고해상도 이미지는 매칭되기 위해 다운스케일링 되어야 함. 이후 대응 관계를 원본 해상도로 업스케일링
    - 성능 손실 발생 가능
    - 위치 정확도나 재구성 품질에서 상당한 저하 발생 가능

**Coarse-to-fine matching**
- 고해상도 이미지를 저해상도 알고리즘으로 매칭할 때 이점을 보존하기 위한 기술
1. 두 이미지의 다운스케일링 버전에서 매칭 수행
    - 서브샘플링 $k$로 얻은 coarse correspondence 집합을 $\mathcal{M}_k^0$이라 함
2. 각 전체 해상도 이미지에서 독립적으로 겹치는 window crop의 grid $W^1, W^2 \in \mathbb{R}^{w \times 4}$를 생성
    - 각 window의 crop은 최대 512픽셀
    - 연속적인 window는 50% 겹침
3. 모든 window 쌍 $(w_1, w_2) \in W^1 \times W^2$의 집합을 나열할 수 있음.
    - 대부분의 coarse corresponcence $M_k^0$을 포함하는 부분 집합을 선택
    - 대응 관계의 90%가 포함될때까지 window pair을 하나씩 탐욕적으로 추가
4. 각 window pair에 대해 독립적으로 매칭을 수행:

$$
\displaystyle
\begin{aligned}

D^{w_1}, D^{w_2} &= \text{MASt3R}(I_{w_1}^1, I_{w_2}^2) &(16) \\
M^{w_1, w_2}_k &= \text{fast\_reciprocal\_NN}(D^{w_1}, D^{w_2}) &(17)

\end{aligned}
$$

- 각 window pair에서 얻은 correspondences를 원래 이미지 좌표로 다시 매핑하고 concatenate하여 고밀도 full-resolution match를 제공

## 4. Experimental results

### 4.1 Training

**Training data**

- 종류
    - Habitat
    - ARKitScenes
    - Blended MVS
    - MegaDepth
    - Static Scenes 3D
    - ScanNet++
    - CO3D-v2
    - Waymo
    - Map-free
    - WildRgb
    - VirtualKitti
    - UnrealK
    - TartanAir
    - 내부 데이터셋
- 이러한 데이터셋은 indoor, outdoor, 합성, 실제 세계, 객체 중심 등 다양한 장면 유형을 포함함
- 10개의 데이터셋은 metric GT를 갖고 있음
- 이미지 쌍이 직접 제공되지 않을 경우, [104]에서 설명된 방법을 기반으로 이미지를 추출함
- off-the-shelf 이미지 검색 및 point matching 알고리즘을 활용하여 이미지 쌍을 매칭하고 검증

**Training**

- DUSt3R 모델을 기반으로 아키텍처를 구성하고 동일한 백본(ViT Large encoder 및 ViT-Base decoder) 사용
- 사전훈련된 모델 사용
- 각 epoch동안, 모든 데이터셋에 균등하게 분배된 650k 쌍을 무작위로 샘플링
- cosine schedule과 lr=0.0001로 35epoch동안 네트워크 훈련
- 이미지 종횡비를 무작위로 설정하여 가장 큰 이미지 차원이 512픽셀로 설정되도록 함
- local feature 차원을 $d=24$로 설정하고 매칭 손실 가중치 $\beta=1$로 설정
- 훈련 중 무작위 crop 방식으로 데이터 증강 수행. crop된 이미지는 principal point의 중심 위치를 유지하기 위해 homography 변환 적용
(crop 이미지는 원본 이미지의 중심과 일치하지 않을 수 있음. crop된 이미지의 카메라 중심은 원본 이미지의 중심으로 되어 있는 상태. 카메라의 중심을 crop 이미지의 중심으로 이동시키기 위해 이미지에 homography 변환을 적용)
    - coarse-to-fine 매칭을 위해 네트워크가 다양한 스케일을 보도록 함

**Correspondence sampling**

- 매칭 손실(식 10)에 필요한 GT correspondences를 생성하기 위해 단순히 실제 3D pointmap $\hat{X}^{1, 1} \leftrightarrow \hat{X}^{}2, 1$ 간의 상호 correspondences를 찾음
- 이후 이미지 쌍당 4096개의 correspondences를 무작위로 subsampling
- 충분한 correspondences를 찾을 수 없는 경우, 무작위 잘못된 correspondence로 패딩
    - 실제 매칭이 유지될 가능성을 일정하게 유지하기 위해

**Fast nearest neighbors**

- $x$의 차원에 따라 다르게 식 (14)에서 최근접 이웃 함수 NN($x$)를 구현
- 3D point $x \in \mathbb{R}^3$을 매칭할 때 K-d tree를 사용하여 NN($x$)를 구현
- $d=24$의 local feature을 매칭할 때는 K-d tree가 차원의 저주로 인해 매우 비효율적이 됨.
    - 이 경우, FAISS 라이브러리에 의존

### 4.2 Map-free localization

**Dataset description**

- Map-free relocalization benchmark[5]로 실험
    - 지도 없이 단일 참조 이미지만을 갖고 metric space(절대 좌표계)에서 카메라 localizing을 하는 데이터셋
    - 장면 수
        - training: 460
        - validation: 65
        - test: 130
    - 평가 지표
        - Virtual Correspondence Reprojection Error(VCRE)
        - 카메라 포즈 정확도

**Impact of subsampling**

- 이 데이터셋에서 coarse-to-fine matching을 사용하지 않음
    - 입력 해상도($720 \times 540$)가 MASt3R의 작업 해상도($512 \times 384$)에 근접하기 때문
- dense 상호 매칭 계산은 최적화된 NN 탐색 코드를 사용해도 매우 느림
    - 대응 집합 $\mathcal{M}$(식 13)에서 최대 $k$개의 correspondence만 유지하면서 subsampling 사용
- 그림 3(오른쪽)은 AUC(VCRE) 성능 및 시간 측면에서 서브샘플링의 영향을 보여줌
    - 서브샘플링의 중간 값들에 대해 상당히 개선됨
- $k=3000$을 사용하면 매칭을 64배 가속할 수 있으며 성능도 향상됨(supplimentary material 참조)

**Ablations on losses and matching modes (표 1 참조)**

![alt text](./images/Table%201.png)

> **Table 1. Map-free 데이터셋에서 검증 결과**  
> 가장 좋은 항목은 굵은 글씨  
> 두 번째로 좋은 항목은 밑줄

표 1에서 다양한 접근 방법의 validations set 실험을 보고
DUSt3R - 3D points 매칭(I)
MASt3R - 3D points 매칭(II) local feature 매칭(III, IV, V)

- 예측된 matches 에서 essential matrix를 추정하고 여기서 relative pose 계산 (PnP가 이와 유사하게 동작함)
- metric scene scale은 KITTI[65]로 finetune된 off-the-shelf DPT로 추출한 depth(I-IV)에서 추정하거나 MASt3R에서 직접 출력된 depth(V)로부터 가져옴
- 제안된 모든 방법이 DUSt3R 기준선보다 현저히 뛰어난 성능을 보임
    - MASt3R가 더 오랜 시간동안 더 많은 데이터로 훈련했기 때문으로 추측
    - 모든 것이 동일할 경우, matching descriptor가 3D 포인트를 매칭하는 것보다 현저히 더 나은 성능을 보임(II vs V)
        - 회귀가 pixel correspondence를 계산하는데 본질적으로 부적합함을 확인해 줌

single matching objective만 사용한 경우(식 10의 $\mathcal{L}_{match}$. III)
- 3D와 matching losses로 훈련한 경우(IV)와 비교했을 때 전반적인 성능이 감소함
- pose estimation accuracy(예: III의 median rotation: 10.8, IV의 media rotation: 3.0)
    - decoder가 두 가지 3D 재구성을 동시에 수행하는 대신 단일 작업을 수행할 수 있는 더 많은 용량을 가지게 되었음에도 발생
    - 3D에서 매칭을 기반으로 하는 것이 실제로 매칭을 개선하는 데 중요함
- MASt3R에 의해 직접 출력되는 metric 깊이를 사용할 때 성능이 크게 향상됨
    - 깊이 예측 작업이 3D 장면 이해와 밀접한 상관관계가 있음
    - 깊이 예측 작업과 3D 장면 이해는 서로 이점을 제공함

**비교 (표 2 참조)**

![alt text](./images/Table%202.png)

> **Table 2. Map-free dataset의 테스트셋으로 SOTA와 비교**

- MASt3R은 모든 SOTA 방식을 크게 능가함
    - VCRE AUC에서 93%이상을 달성
    - median translation error은 2m에 비해 36cm로 크게 감소
- DPT-KITTI에서 깊이를 활용하는 순수 매칭 기반 방법이 모든 SOTA를 능가
- Direct regression 방법
    - 매칭 없이(!!) point맵에서 PnP 적용(2D-3D 정보가 있으므로 가능)
        - 카메라의 gt calibration을 사용하지 않았음에도 매칭 기반 변형 방법과 유사함
    - 다른 localization 데이터셋에서는 그렇지 않음. PnP나 essential matrix와 같은 intrinsic을 활용한 매칭을 통한 pose estimation이 일반적으로 더 안전해 보임

**Qualitative results (그림 4 참조)**

![alt text](./images/Fig%204.png)

> **Figure 4. Map-free dataset에서 정성적 샘플**  
> 윗줄: 강한 viewpoint 변화가 있는 경우, 세 번째는 실패 케이스. 전체 correspondences 중 일부만 그림  
> 아랫줄: 흥미로운 지점 확대. local keypoint 매칭이 힘들 것으로 예상되는 부분들

- 강한 시점 변화(최대 180도)가 있는 쌍에 대한 일부 매칭 결과를 보임
- 극적인 외관 변화에도 불구하고 MASt3R에 의해 정확하게 매칭된 경우를 보임
    - 이러한 correspondences는 2D 기반 매칭에서 거의 불가능할 것
    - 매칭이 3D라고 생각하면 상대적으로 간단하게 해결 가능

### 4.3 Relative pose estimation

**Datasets and protocol**

- CO3Dv2[67]
    - 약 37k개 비디오에서 추출한 600만 프레임 포함
    - 51 MS-COCO 범주를 포함함
    - 각 비디오의 200프레임을 사용하여 COLMAP을 통해 카메라 포즈를 얻음
- RealEstate10k[121]
    - YouTube의 80k 비디오 클립
    - 총 1000만 프레임을 포함하는 실내/실외 데이터셋
    - SLAM 및 Bundle Adjustment로 카메라 포즈를 얻음
- [100]에 따라 CO3Dv2의 41개 범주와 RealEstate10k 테스트셋에서 1.8k 비디오 클립을 평가.
- 각 시퀀스는 10프레임 길이이며, 모든 가능한 45쌍 간의 상대 카메라 포즈를 평가. GT focal을 사용하지 않음

**Baselines and metrics**

- MASt3R로 얻어진 match는 essential matrics 및 상대 포즈를 추정하는데 사용됨
- 예측은 항상 쌍별로 이뤄지며, DUSt3R PnP를 제외한 모든 다른 방법이 여러 뷰를 활용하는 것과 대조적
- 최신 데이터 기반 접근 방식
    - RelPose
    - RelPose++
    - PoseReg
    - PoseDiff
    - RayDiff
    - DUSt3R
- 전통적인 SfM 방식
    - PixSFM
    - SuperPoint
    - SuperGlue(COLMAP+SPSG)
- 각 이미지 쌍에 대해 Relative Rotation Accuracy(RRA) 및 Relative Translation Accuracy(RTA)를 보고
- 임계값 $\tau=15$를 선택하여 RTA@15 및 RRA@15를 보고
- min(RRA@30, RTA@30)에서 각도 차원의 정확도 곡선 아래 면적 정의된 mean avearge accuracy(mAA30)을 계산

**Results (표 3 참조)**

![alt text](./images/Table%203.png)

> **Table 3.**  
> 좌: CO3Dv2 및 RealEstate10K 데이터셋의 10개의 무작위 프레임으로 Multi-view pose 회귀  
> 괄호는 이 방법으로 보고된 결과가 아님  
> 비교를 위해 최상의 결과(8개 뷰)를 보고  
> (a) 다중 뷰 방법과 (b) pairwise 방법을 구분  
> 우: DTU 데이터셋에서의 dense MVS 결과 (mm)  
> (c) 수작업 방법은 (d) 특정 도메인에서 학습된 학습 기반 방법보다 성능이 저조함  
> (e) zero-shot 설정에서 작동하는 방법 중 MASt3R이 유일하게 합리적인 성능을 달성

- SfM 방식은 주로 시각적 지원이 부족하기 때문에 상당히 부진하게 수행되는 경향이 있음
    - 이미지가 보통 작은 물체를 관찰하는 경우가 많고, 많은 쌍이 최대 180도까지 넓은 baseline을 갖기 때문
- RayDiffusion, DUSt3R, MASt3R과 같은 3D 기반 접근 방식은 이 데이터셋에서 가장 경쟁력있는 방법
    - MASt3R은 두 데이터셋 모두에서 translation 및 mAA가 뛰어남
    - RealEstate에서는 mAA점수가 최고 multi-view 방법보다 8.7 point, pairwise DUSt3R보다 15.2 point 향상됨
    - few input view 설정에서 제안 방식의 정확성과 견고성을 보임

### 4.4 Visual localization

**Dataset**

- Aachen Day-Night[118]과 InLoc[84] 데이터셋에서 절대 포즈 추정
    - Aachen Day-Night
        - 4328개 참조 이미지
        - 독일 Aachen 구시가지에서 모바일 폰으로 촬영된 824개 주간 이미지와 98개 야간 이미지로 구성
    - InLoc
        - 9972개 RGB-D + 6DoF 포즈 데이터베이스 이미지
        - iPhone 7로 촬여한 329개 쿼리 이미지 간의 도전적인 외관 변화를 가진 실내 데이터셋
    
**Metric**

- Aachen
    - (0.25m, 2°), (0.5m, 5°), (5m, 10°)
- InLoc
    - (0.25m, 10°), (0.5m, 10°), (1m, 10°)

**결과 (표 4 참조)**

![alt text](./images/Table%204.png)

> **Table 4. Aachen Day-Night와 InLoc 데이터셋에서 Visual Localization결과**  
> 다른 숫자의 검색 데이터베이스 이미지 개수를 사용한 결과를 보고(topN)

- 더 많은 이미지가 검색될 때(top40) 더 나은 성능 제공
    - Aachen에서 경쟁력 있는 성능 달성
    - InLoc에서 SOTA를 크게 초월
- 검색된 이미지가 하나(top1)일 때도 잘 작동함
- 직접 회귀 결과는 상당히 좋지 않음
    - 데이터셋 규모가 localization 오류에 미치는 극적인 영향을 보여줌
    - 작은 장면은 훨씬 덜 영향을 받음
    - 신뢰할 수 있는 포즈를 추정하기 위한 feature matching의 중요성을 확인

### 4.5 Multiview 3D reconstruction

## 5. 결론

- MASt3R을 통한 3D Grounding 이미지 매칭은 많은 공개 벤치마크에서 카메라 포즈 및 localization 작업의 기준을 크게 높임
- 매칭을 통해 DUSt3R을 성공적으로 개선하여 강화된 견고성을 갖추면서 pixel 매칭만으로 할 수 있는 것 이상을 달성
- 효율적인 처리를 위한 빠른 reciprocal matcher와 coarse-to-fine 방식을 도입하여 정확성과 속도 간의 균형을 맞출 수 있도록 함
- 몇 가지 뷰 환경(top1의 경우도)에서도 수행할 수 있어 localization의 다재다능성을 크게 증가시킬 것