# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors



---

- SLAM

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html) (CVPR 2025)

---
짧은 요약



요약

- 2D 이미지만으로는 dense SLAM 시스템을 수행하기에 부족함.
- 3D 정보는 view 간에 불변하기 때문에 3D 기하학 공간 활용 필요

---

**문제점 & 한계점**


---

목차

0. [Abstract](#abstract)
1. 

---


## Abstract

MASt3R-SLAM
- 두 시점의 3D 재구성과 matching prior로 설계된 SLAM 시스템
- 고유한 카메라 중심만 있으면(fixed 또는 parametric 카메라 모델에 대한 가정을 하지 않아도) 야외 비디오 시퀀스에서 강력함
- pointmap 매칭, 카메라 추적 및 local 융합, 그래프 구성 및 loop closure, 2차 전역 최적화를 위한 효율적인 방법을 소개
- calibration이 있는 경우 간단한 시스템 수정으로 다양한 벤치마크에서 SOTA 달성
- 15FPS로 작동하면서 일관된 포즈와 dense geometry를 생성할 수 있는 plug and play monocular SLAM 시스템 제안

## 1. Introduction

SLAM(Simultaneous Localization and Mapping)
- 하드웨어 전문 지식과 calibration을 요구하기 때문에 아직 plug-and-play 알고리즘이 아님
- IMU와 같은 추가 센서 없이 최소한의 단일 카메라 설정에서는 정확한 포즈와 일관된 dense map을 제공하는 in-the-wild SLAM이 존재하지 않음

신뢰할 수 있는 dense SLAM 시스템
- 2D 이미지만으로 dense SLAM 수행에는 다음이 필요
    - 시간에 따라 변화하는 포즈와 카메라 모델
    - 3D scene 기하학
    - 이러한 고차원의 역문제를 해결하기 위해 다양한 선험적 priors가 제안됨
        - Single-view priors
            - monocular 깊이 및 normal
                - 단일 이미지에서 기하학을 예측하고자 함
                - 이러한 가정은 모호성을 포함하고 있으며, view 간의 일관성이 부족함
        - Multi-view priors
            - 광학 흐름 등
            - 모호성을 줄이지만, 포즈와 기하학을 분리하는 건 도전적
            - 픽셀 움직임은 외부 요소와 카메라 모델 모두에 의존
- 3D 장면은 view 간에 불변함.
    - 이미지를 통해 포즈, 카메라 모델, dense 기하학을 해결하는 데 필요한 통합 선험적 가정은 공통 좌표계에서의 3D 기하학 공간에 있음

DUSt3R[50]과 MASt3R[21]이 선도한 두 개의 view 3D 재구성 prior이 정제된 3D 데이터셋을 활용하여 SfM에서 패러다임 전환을 일으킴
- 공통 좌표계에서 두 개의 이미지로부터 pointmap을 직접 출력하여 앞서 언급한 하위 문제를 공동 프레임워크에서 암시적으로 해결
- 추후에는 이러한 prior이 상당한 왜곡을 가진 모든 종류의 카메라 모델에 대해 훈련될 것
- 3D prior은 더 많으 view를 수용할 수 있지만, SfM과 SLAM은 공간 희소성을 활용하고 중복을 피하여 대규모 일관성 달성
- two-view 아키텍처는 SfM의 기본 블록으로써 two-view 기하학을 반영하며, 이러한 모듈성은 효율적인 의사 결정과 강력한 합의가 이후 이뤄질 수 있는 기회를 제공함

**최초의 실시간 SLAM 프레임워크 제안(그림 1 참조)**

![alt text](./images/Fig%201.png)

> **Figure 1. Burghers 시퀀스에서 dense monocular SLAM 시스템을 통한 재구성**  
> 좌측의 MASt3R의 two-view 예측을 사용하여 known camera model 없이도 실시간으로 전역적으로 일관된 자세와 기하학을 달성

- two-view 3D 재구성 prior을 추적, 맵핑, relocalization을 위한 통합 기초로써 활용
- 이전 작업에서는 이러한 prior을 비정렬 이미지 컬렉션에서 offline 환경의 SfM에 적용
- SLAM은 데이터를 점진적으로 수신하고 실시간 작업을 유지해야 함
    - 저지연 매칭, 신중한 맵 유지보수 및 대규모 최적화를 위한 효율적인 방법에 대한 새로운 관점을 요구
- SLAM에서 필터링 및 최적화 기술에서 영감을 받아, 프론트엔드에서 pointmap의 local filtering을 수행하여 백엔드에서 대규모 global 최적화를 가능하게 함
- 모든 광선이 통과하는 고유한 카메라 중심을 가지고 있다는 것 이외에 각 이미지의 카메라 모델에 대해 어떤 가정도 하지 않음
    - 일반적이고 시간에 따라 변하는 카메라 모델로 장면을 재구성할 수 있는 실시간 밀집 단안 SLAM 시스템이 구현됨
- calibration이 주어지면 경로 정확도와 dense geomtry estimation에서 SOTA 성능을 보임

논문의 기여
- two-view 3D 재구성 prior인 MASt3R를 기반으로 하는 최초의 실시간 SLAM 시스템
- pointmap 매칭, 추적 및 local fusion, 그래프 구축 및 loop closure, 이차 클로벌 최적화를 위한 효율적인 기술
- 일반적이고 시간에 따라 변하는 카메라 모델을 처리할 수 있는 SOTA dense SLAM 시스템

## 2. Related Work

## 3. Method

### 3.1 Preliminaries

DUSt3R
- 입력: $I^i, I^j \in \mathbb{R}^{H \times W \times 3}$
- 출력: 
    - 포인트맵: $X^i_i, X^j_i \in \mathbb{R}^{H \times W \times 3}$
    - 신뢰도: $C^i_i, C^j_i \in \mathbb{R}^{H \times W \times 1}$
- $X^j_i$는 이미지 $i$의 포인트맵을 카메라 $j$의 좌표계로 표현한 것

MASt3R
- 일치하는 d차원 특징을 예측하기 위해 $D^i_i, D^j_i \in \mathbb{R}^{H \times W \times d}$와 특징 신뢰도 $Q^i_i, Q^j_i \in \mathbb{R}^{H \times W \times 1}$를 예측
- forward pass를 정의: $\mathcal{F}_\mathrm{M}(I^i, I^j)$
    - 이전에 논의된 출력을 산출
    - 이후 본문에서는 이 출력을 그대로 사용
- MASt3R을 훈련하느 데 사용된 데이터 중 일부는 metric scale을 가짐
    - 스케일이 종종 예측 간 큰 불일치의 원인임을 발견
    - 서로 다른 scale의 예측을 최적화하기 위해 모든 포즈를 $T \in \mathrm{Sim}(3)$로 정의하고, Lie 대수 $\tau \in \mathfrak{sim}(3)$와 left-plus 연산자를 이용해 업데이트한다.

$$
\displaystyle
\begin{aligned}

\mathrm{T} = \begin{bmatrix} sR & t \\ 0 & 1 \end{bmatrix}, \quad \mathrm{T} \leftarrow \tau \oplus \mathrm{T} \equiv \text{Exp}(\tau) \circ \mathrm{T}

\end{aligned}
$$

> $R \in \mathrm{SO}(3)$  
> $t \in \mathbb{R}^3$  
> $s \in \mathbb{R}$

카메라 모델에 대한 가정
- 일반 central camera
    - 모든 광선이 고유한 카메라 중심을 통과
- $\psi (\mathrm{X}^i_i)$: pointmap $\mathrm{X}^i_i$를 unit norm의 광선으로 정규화하는 함수
    - 시간에 따라 변하는 카메라 모델(예: 줌) 이나 렌즈 왜곡과 같은 요소를 통합 처리할 수 있음

### 3.2 Pointmap Matching

목표: MASt3R로 얻은 pointmap과 feature을 이용해 두 이미지 간 pixel 매칭 집합 $m_{i,j} = \mathcal{M}\bigl(X^i_i,X^j_i,D^i_i,,D^j_i \bigr) $를 찾기
- 단순 brute-force matching은 계산 복잡도가 이차적으로 증가
- DUSt3R은 3D 점에 대한 k-d tree를 사용하여 이 문제를 완화
    - k-d 트리 구축의 병렬화가 어려움
    - pointmap 예측 오차가 있으면 3D NN 탐색이 부정확한 매칭을 많이 생성
- MASt3R은 네트워크가 예측하는 고차원 특징을 활용해 더 넓은 baseline에서도 매칭이 가능하도록 함
    - coarse-to-fine scheme을 제안함
        - dense pixel 매칭 시 런타임이 수 초 단위로 느림
        - sparse matching도 k-d tree보다 느림
- SLAM의 projective data-association 기법을 활용
    - parametric 카메라 모델의 closed-form projection이 필요하지만, 이 시스템은 각 프레임이 유일한 카메라 중심을 가진다고 가정
    - 출력된 pointmap $\mathrm{X}^i_i, \mathrm{X}^j_i$로부터 함수 $\psi(\mathrm{X}^i_i)$를 이용해 제너릭 카메라 모델을 구성
    - 참조 프레임에서 광선 간 오차(ray error)를 최소화하는 픽셀 좌표 $\text{p}$를 반복적으로 최적화하여 각 point $x \in X^j_i$를 독립적으로 projection

$$
\displaystyle
\begin{aligned}

& \mathrm{p}^* = \argmin_{\mathrm{p}} \bigl\|\psi([\mathrm{X}^i_i]_\mathrm{p}) - \psi(\mathrm{x})\bigr\|^2
& (2)

\end{aligned}
$$

![alt text](./images/Fig%202.png)

> **Figure 2. iterative projective 매칭 개요**  
> MASt3R로부터 두 개의 pointmap 예측이 주어졌을 때, 참조 pointmap은 ray mapping을 하기 위해 smooth pixel을 제공하기 위해 $\psi(\mathrm{X}^i_i)$로 정규화됨  
> pointmap \mathrm{X}^i_i의 3D point x의 projection 추정 $\text{p}_0$를 처음 계산하기 위해 pixel은 queried ray $\psi([\mathrm{X}^i_i]_\mathrm{p})$와 target ray $\psi(\mathrm{x})$의 angular difference $\theta$를 최소화하기 위해 단계적으로 업데이트됨  
> minimum error을 갖는 pixel $\mathrm{p}^*$를 찾게 되면, $\mathcal{I}^i$와 $\mathcal{I}^j$ 사이의 pixel correspondence를 갖게 됨

- 정규화된 벡터 간의 euclidean 거리 최소화: 두 개의 정규화된 광선 간의 각도 $\theta$를 최소화하는 것과 동등함

$$
\displaystyle
\begin{aligned}

& \|\psi_1 - \psi_2\|^2 = 2(1 - \cos\theta), \quad \cos\theta = \psi_1^\top \psi_2
& (3)

\end{aligned}
$$

[35]와 유사한 nonlinear least-squares 형식을 사용하여 analytical Jacobians을 계산하고 Levenberg-Marquardt를 해결하여 투영된 위치에 대한 업데이트를 반복적으로 해결할 수 있음
- 각 점에 대해 별도로 수행 가능. 광선 이미지가 smooth하기 때문에 거의 모든 유효 픽셀에 대해 10번의 반복 안에 수렴함
- 이 과정이 끝나면 초기 매칭 $m_{i, j}$를 갖게 됨
- projection $\mathrm{p}$에 대한 초기 추정이 없을 때(예: 새로운 keyframe에 대한 추적 중이거나 loop closure edge를 매칭할 때), 모든 픽셀이 identity mapping으로 초기화됨
- 추적 중에는 항상 이전 프레임의 매칭이 있기에 수렴 속도를 높일 수 있음
- 가려짐과 이상치를 처리하기 위해 3D 공간에서 큰 거리의 매칭을 무효화
- 매칭은 GPU에서 병렬 처리되며 SLAM의 점진적 특성을 활용할 수도 있음

이 픽셀들이 기하학을 사용하여 매칭에 대한 좋은 추정을 제공하는 동안, MASt3R은 픽셀마다의 특징을 활용하는 것이 포즈 추정의 downstream 성능을 크게 향상시킴을 보임
- 이전 단계에서 좋은 초기 결과를 얻었으므로, 지역 파치 윈도우에서 maximum feature 유사성으로 픽셀 위치를 업데이트하여 coarse-to-fine 이미지 기반 검색을 수행
- 반복 projection 및 feature extraction 단계를 사용자 정의 CUDA 커널에서 구현
    - 각 pixel에 대해 병렬 처리 가능
    - 2ms 소요
    - 그래프에서 edge를 구성하는 데 초기 projection 추정 없이 새로 추가된 모든 edge에 대해 몇 ms만 소요됨
- 매치가 pose 추정에 의해 편향되지 않음
    - MASt3R 출력에만 의존하므로 projection data association에 비전형적임 (3D 점을 투영하고 근처 점을 찾는 과정)

### 3.3 Tracking and Pointmap Fusion

현재 프레임 $\mathcal{I}^f$와 이전 keyframe $\mathcal{I}^k$간의 상대 변화 $\mathrm{T}_{kf}$를 추정
- 네트워크를 한 번만 실행하여 $\mathrm{T}_{kf}$를 추정
- 마지막 keyframe의 pointmap 추정치 $\tilde{\mathrm{X}}_k^k$가 있다고 가정
- $\mathcal{I}^f$ 프레임에 대한 point는 $\mathcal{F}_M(\mathcal{I}^f, \mathcal{I}^k)$를 통해 얻을 수 있음
- 포즈를 추정하기 위한 

$$
E_{\text{prior}}(T_{kf}) = \lambda_t \cdot \left\|\, \log(T_{kf}^{\text{pred}}{}^{-1} \cdot T_{kf}^{\text{current}}) \,\right\|^2


$$


$$
w(q, \sigma^2) =
\begin{cases}
\frac{\sigma^2}{q}, & \text{if } q > q_{\text{min}} \\
\infty, & \text{otherwise}
\end{cases}
$$

### 3.4 Graph Construction and Loop Closure