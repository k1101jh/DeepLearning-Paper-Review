# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- SLAM

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html) (CVPR 2025)

---
요약

- 문제점
    - 2D 이미지만으로는 dense SLAM 시스템을 수행하기에 부족함.
    - 3D 정보는 view 간에 불변하기 때문에 3D 기하학 공간 활용 필요

- 카메라 모델 가정
    - 일반 central camera로 가정
        - 모든 광선이 고유한 카메라 중심을 통과함
- MASt3R
    - MASt3R의 훈련 데이터 일부는 metric scale을 가짐
        - 이는 종종 예측 간 큰 불일치를 발생시킴
        - 모든 포즈를 Sim(3)으로 정의하고, sim(3)과 left-plus 연산자로 업데이트
- 두 이미지 간 pointmap 매칭
    - pointmap으로부터 generic camera 모델을 구성
    - 참조 프레임에서 ray error을 최소화하는 픽셀 좌표 p를 반복적으로 최적화
    - 전역 검색 없이도 정확한 대응점 획득
    - 간단히 말해서, 카메라 두 개가 있을 때(연속 프레임)
        - A 카메라에서 점을 하나 선택
        - B 카메라에서 임의의 점을 선택
        - B 카메라에서 선택한 점이 A 카메라의 점과 같아질 때까지 B 카메라의 점을 최적화
- Tracking
    - MASt3R로 $T_kf$를 추정
    - 마지막 keyframe의 pointmap 추정치가 있다고 가정
    - Graph Construction and Loop Closure
- encoded feature로 데이터베이스 쿼리
- 특징 벡터들을 codebook의 중심점들 중 하나로 양자화
- backend optimization
    - 맨 첫 번째 키프레임의 포즈를 고정
    - 모든 edge의 rayerror을 공동으로 최소화
    - 새 keyframe마다 최대 10회의 gauss-newton 반복

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
- 3D prior은 더 많은 view를 수용할 수 있지만, SfM과 SLAM은 공간 희소성을 활용하고 중복을 피하여 대규모 일관성 달성
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

Sparse Monocular SLAM
- 목표: 카메라 자세와 편향 없는 일부 3D 랜드마크 동시 추정
- 최적화의 희소성(sparsity)를 활용한 알고리즘 발전과 정교한 graph 구성  
-> 실시간 pose 추정 및 sparse reconstruction 가능
- 충분한 특징점과 시차(parallax)가 있으면 매우 정확함
    - dense scene model이 없어 강인한 추적이나 기하 구조에 대한 명시적 추론에는 한계가 있음

Dense Monocular SLAM
- 자세와 밀집 깊이(Dense Depth)를 수작업으로 규제하고 번갈아 최적화하는 방식 제안
- 제어된 환경에서만 적용 가능
- 최근 경향
    - 데이터 기반 prior과 backend 최적화 결합
    - 단일 이미지에서 depth나 surface normal을 예측하는 기술이 크게 발전
        - SLAM에서의 활용은 제한적
    - 단일 뷰 기반 gemoetry 예측은 모호함
        - 편향되고 불일치하는 3D 기하 생성
- SLAM에서의 대안 접근
    - 가능한 깊이의 가설 공간(hypothesis space)에 대한 prior 예측에 집중
    - 형태:
        - latent spaces[2, 6]
        - subspaces[41]
        - local primitives[24]
        - distributions[8, 9]
    - 이러한 prior은 더 높은 일관성을 제공할 수 있지만, multiple view 간 강인한 대응 관계 확보가 필수임

~

DUSt3R
- 새로운 two-view 3D 재구성 prior 제안
- 두 이미지의 dense 3D point cloud를 공통 좌표계에서 출력
- 기존 prior들이 작업의 subprolems만 해결하는 것과 달리, DUSt3R은 correspondence, pose, camera model, dense geometry를 암묵적으로 추론하여 직접적인 two-view 3D 장면의 pseudo-measurement 제공

MASt3R
- DUSt3R의 후속 모델
- pixel 단위 특징을 추가로 예측하여 localization과 SfM에서의 pixel 매칭 성능 향상
- 모든 prior과 마찬가지로, 여전히 3D geometry의 불일치와 상관된 오차 존재

DUSt3R & MASt3R의 한계
- global consistency를 위해 대규모 최적화 필요
- 이미지가 많아질수록 시간 복잡도가 급격히 증가

Spann3R
- backend 최적화를 생략하고, DUSt3R을 fine-tuning하여 global 좌표계로 직접 pointmap stream 예측
- token 메모리 제한으로 큰 장면에서 drift 발생 가능

본 연구의 제안
- two-view 3D 재구성 prior을 기반으로 dense SLAM 시스템 설계
- 일반적인 central camera model만 가정
- 효율적인 pointmap 매칭, tracking, pointmap 융합, loop closure, global optimization 기법 제안
- pairwise 예측의 대규모 일관성을 실시간으로 달성

## 3. Method

![alt text](./images/Fig%203.png)

> **Figure 3. System diagram of MASt3R-SLAM**  
> 새로운 이미지는 MASt3R로 pointmap을 예측하고 반복 투영 pointmap 매칭을 사용하여 픽셀 일치를 찾아 현재 keyframe과 비교하여 추적됨  
> 추적은 현재 자세를 추정하고 local pointmap fusion을 수행  
> 새로운 keyframe이 backend에 추가되면 인코딩된 MASt3R 특성을 사용하여 검색 DB에 질의하여 loop closure 후보를 선택  
> 후보는 MASt3R에 의해 디코딩되며, 충분한 수의 일치가 발견되면 edge가 backend 그래프에 추가됨  
> 대규모 2차 최적화는 자세와 밀집 geometry의 전역 일관성을 달성함

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
- MASt3R을 훈련하는 데 사용된 데이터 중 일부는 metric scale을 가짐
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
> pointmap $\mathrm{X}^i_i$의 3D point x의 projection 추정 $\text{p}_0$를 처음 계산하기 위해 pixel은 queried ray $\psi([\mathrm{X}^i_i]_\mathrm{p})$와 target ray $\psi(\mathrm{x})$의 angular difference $\theta$를 최소화하기 위해 단계적으로 업데이트됨  
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
- 포즈를 추정하기 위한 방법 중 하나는 3D point error을 최소화 하는 것

$$
\displaystyle
\begin{aligned}

& E_p = \sum_{m,n \in \text{m}_{f,k}} \left\| \frac{\tilde{\text{X}}_{k, n}^k - \text{T}_{kf} \text{X}_f^f(m)}{w(\text{q}_{m,n}, \sigma_p^2)} \right\|_\rho
& (4)

\end{aligned}
$$

> $q_{m, n} = \sqrt{\text{Q}_{f, m}^f \text{Q}_{f, n}^k} $: MASt3R-SfM에서 제안한 match confidence weighting
- 강인함을 위해 Huber norm $\left\| \cdot \right\|_\rho$에 이어 per-match weighting을 적용


$$
\displaystyle
\begin{aligned}

& w(\text{q}, \sigma^2) =
\begin{cases}
\sigma^2 / \text{q}, & \text{if } \text{q} > \text{q}_{\text{min}} \\
\infty, & \text{otherwise}
\end{cases}

& (5)

\end{aligned}
$$

$\text{X}_f^f$ 보다 $\text{X}_f^k$를 사용했을 때 $\text{X}_k^k$와 매칭될 수 있음
- pixel 정렬이 되어 있어서 명시적인 matching없이 정렬 가능
- $\text{X}_f^f$와 명시적인 매칭을 사용하면 larger baseline 시나리오에서 더 정확함
- 3D point error는 적절하지만, 깊이에 대한 일관되지 않은 예측이 상대적으로 자주 발생하므로 pointmap 예측의 오류에 의해 쉽게 왜곡됨
- 모든 예측을 평균내어 하나의 pointmap으로 통합
    - tracking 에러는 keyframe의 pointmap을 감소시킬 것(이 pointmap은 backend에서 사용됨)


중앙 카메라 가정을 기반으로 pointmap을 광선으로 변환할 수 있음
- 정확하지 않은 깊이 예측에 덜 민감한 directional ray error을 사용할 수 있음

$$
\displaystyle
\begin{aligned}

& E_r = \sum_{m,n \in \text{m}_{f,k}} \left\| \frac{\psi(\tilde{\text{X}}_{k,n}^k) - \psi(\text{T}_{kf} \text{X}_{f,m}^f)}{w(\text{q}_{m,n}, \sigma_r^2)} \right\|_\rho
& (6)

\end{aligned}
$$

- similar angular error 결과는 식 3과 그림 2 참조
    - 많은 known correspondences를 갖고 있고, 현재 프레임에서 예측된 연관된 ray와 canonical rays 사이의 angular error을 최소화시키는 pose를 찾고자 함
- angular error이 제한되어있기에, ray-based errors는 outlier에 강인함[30]

카메라 중심 거리 차이에 작은 가중치를 둔 error term 추가
- 시스템이 순수 회전하에 퇴화되는 것을 방지
- 깊이 오류로 인한 중대한 편향을 피함
- 반복적으로 가중치가 조정된 least-squares(IRLS)프레임워크에서 Gauss-Newton을 사용하여 자세 업데이트를 효율적으로 해결
- 상대 자세 $\text{T}_{kf}$에 대한 perturbation $\tau$에 대한 ray & distance error의 해석적 Jacobian을 계산
- residuals, Jacobians, weights를 각각 행렬 $\text{r}, \text{J}, \text{W}$로 표기
- 선형 시스템을 반복적으로 해결하고 자세를 업데이트:

$$
\displaystyle
\begin{aligned}
& (\text{J}^T \text{W} \text{J}) \tau = -\text{J}^T \text{W} r,\quad \text{T}_{kf} \leftarrow \tau \oplus \text{T}_{kf}
& (7)
\end{aligned}
$$

각 pointmap이 새 정보를 제공할 수 있음
- 기하학 추정치뿐만 아니라 ray로 정의되는 카메라 모델도 필터링하여 이를 활용
- 상대 포즈를 해결한 후, transform $\text{T}_{kf}$를 사용하여 실행 중인 weighted average filter을 통해 canonical pointmap $\tilde{X}_k^k$를 업데이트 할 수 있음[5, 28]


$$
\displaystyle
\begin{aligned}

&\tilde{X}_k^k \leftarrow \frac{ \tilde{C}_k^k \tilde{X}_k^k + C_f^k \cdot (T_{kf} X_k^f) }{ \tilde{C}_k^k + C_f^k }, \quad \tilde{C}_k^k \leftarrow \tilde{C}_k^k + C_f^k

&(8)

\end{aligned}
$$

Pointmap은 초기에 큰 error을 갖고 낮은 신뢰도를 가짐
- 작은 baseline frame만을 사용했기 때문
- filtering은 많은 viewpoints로부터 정보를 합칠 수 있게 함
- weighted average가 일관성을 유지하면서 noise를 필터링하는 데 가장 좋음을 실험적으로 발견
- MASt3R-SfM[10]의 canonical pointmap에 비해 본 논문에서는 이를 점진적으로 계산
    - $\text{X}_k^k$의 추가 네트워크 예측이 tracking 속도를 늦출 수 있기 때문에 points의 transformation이 필요
- 필터링은 모든 카메라 자세에 대해 명시적으로 최적화하거나 decoder에서 모든 예측된 pointmap을 backend에 저장할 필요 없이 모든 프레임에서 정보를 활용하는 이점을 제공

### 3.4 Graph Construction and Loop Closure

- Tracking 중 valid match 수 또는 $\mathrm{m}_{f, k}$의 unique keyframe 픽셀 수가 threshold $w_k$ 미만으로 낮아지면 새로운 keyframe $\mathcal{K}_i$ 추가
- $\mathcal{K}_i$ 추가 후, 이전 keyframe $\mathcal{K}_{i - 1}$과의 bidirectional edge가 edge-list $\mathcal{E}$에 추가됨
    - 예측 포즈를 시간 순차적이도록 제약을 걸지만 드리프트는 여전히 발생할 수 있음(오차가 발생할 수 있음)

**Loop Closure**
- Aggregated Selective Match Kernel(ASMK)[46, 47] 적용
    - MASt3R-SfM[10]에서 사용함
    - 모든 이미지를 처음부터 사용 가능한 batch 설정으로 사용되었지만, 이를 incrementally하게 작동하도록 수정
- top-K 이미지를 얻기 위해 encoded features $\mathcal{K}_i$로 데이터베이스 쿼리
    - codebook은 수만개의 centroid만 존재. Dense L2 거리 계산이 feature을 양자화하는 데 충분히 빠름
        - ASMK에서 검색하려면 특징 벡터들을 codebook의 중심점들 중 하나로 양자화해야 함. 이 중에서 가장 가까운 벡터를 찾을 때 L2 거리를 사용
        - 중심점 수가 수만 개 정도로 작기 때문에 dense 방식으로도 계산이 빠름
    - 검색 점수가 임계값 $w_r$ 이상이면 MASt3R 디코더에 전달하고 3.2절의 match 수가 $w_l$ 이상이면 양방향 edge를 추가
    - 새로운 keyframe의 인코딩된 feature을 inverted 파일 인덱스에 추가하여 검색 데이터베이스를 업데이트

### 3.5 Backend Optimization

- 입력: keyframe 포즈 $\text{T}_{WC_i}$와 각 $\mathcal{K}_i$의 canonical pointmaps $\tilde{X}_i^i$
- 목적: 모든 pose와 기하에 대해 전역적 일관성(global consistency)달성

최적화 방식
- 과거: 1차(gradient-based) 최적화를 사용하고, 매 iteration 후 rescaling 필요
- 제안 방식:
    - 효율적인 2차 최적화 도입
    - 문제의 gauge freedom(전역 Sim(3))는 첫 keyframe의 7-Dof Sim(3) 포즈를 고정하여 제거
    - graph의 모든 edge $\mathcal{E}의 ray error을 공동으로 최소화

$$
\displaystyle
\begin{aligned}

& E_g = \sum_{i,j \in \mathcal{E}} \sum_{m,n \in \mathrm{m}_{i,j}}
\left\| \frac{\psi\!\left(\tilde{X}_{i,m}^{i}\right) - \psi\!\left(\mathrm{T}_{ij}\,\tilde{\mathrm{X}}_{j,n}^{j}\right)}{w\!\left(\mathrm{q}_{m,n}, \sigma_r^2\right)}\right\|_\rho,\;

& (9)

\end{aligned}
$$

> $\text{T}_{ij} = \text{T}_{WC_i}^{-1}\, \text{T}_{WC_j}$

- N개의 keyframe이 주어졌을 때 식 (9)는 $14 \times 14$ 블록을 $7N \times 7N$ Hessian으로 누적 (7-DoF이므로)
    - 각 edge는 두 pose를 연결하므로 $14 \times 14$ 블록이 생성됨
    - Gauss-Newton 업데이트를 사용. dense 하지 않은 경우 sparse Cholesky decomposition으로 해결
    - Hessian의 효율적 구성
        - 해석적 Jacobian 사용과 CUDA로 구현된 parallel reduction
        - pure-rotation 상황의 퇴화를 피하기 위해 거리 일관성에 대한 작은 규제 항을 추가
        - 새 keyframe마다 최대 10회의 Gauss-Newton 반복. 수렴 시 조기 종료
- 2차 항을 사용하여 전역 최적화를 상당히 가속
- 효율적 구현으로 병목이 되지 않음

### 3.6 Relocalization

시스템이 충분한 수의 match 부족으로 tracking을 잃으면 relocalization이 트리거됨
- 새 프레임의 경우, 검색 데이터베이스가 점수에 대한 더 엄격한 기준으로 쿼리됨
    - 검색된 이미지가 현재 프레임과 충분한 수의 일치 항목을 가지면 그래프에 새로운 keyframe으로 추가되고 추적 재개

### 3.7 Known Calibration

Camera calibration이 있다면 두 가지 변경으로 정확성을 향상시킬 수 있음

1. canonical pointmaps를 사용하기 전에 depth dimension만을 쿼리하고 pointmap을 known camera model이 정의한 광선에 따라 역투영되도록 제한
    - 3D point가 실제 카메라의 광선 방향에 맞게 배치되도록 보정
2. 잔차 계산을 기존 광선 공간에서 픽셀 공간으로 변경
    - 추가적인 거리 residual 은 일관성을 위해 깊이값으로 변환됨

$$
\displaystyle
\begin{aligned}

&E_Π = \sum_{i,j \in E} \sum_{m,n \in m_{i,j}} 
\left\| \frac{p^i_{i,m} - \Pi(T_{ij} \tilde{X}^j_{j,n})}{w(q_{m,n}, \sigma^2_Π)} \right\|_{\rho}
& (10)

\end{aligned}
$$

> $Π$: 주어진 카메라 모델에서 pixel 공간으로 projection 함수

## 4. Results

데이터셋(단안 RGB)
- TUM RGB-D
- 7-Scenes
    - Depth 카메라 측정값이 있음
- ETH3D-SLAM
    - 기하학적 평가 진행
        - 3D 구조에 대한 실제 값 제공
- EuRoC Vision room

실험 환경
- I9-12900K
- RTX 4090

시스템
- 15FPS로 실행됨
- 실시간 성능을 시뮬레이션하기 위해 데이터셋의 매 2프레임마다 하위 샘플링 진행
- MASt3R에서 나온 전체 해상도 출력
- 가장 큰 차원 크기를 512로 조정


### 4.1 Camera Pose Estimation

- ATE(Absolute Trajectory Error, 절대 궤적 오류)의 RMSE를 m단위로 보고
- 모든 시스템이 monocular이므로 스케일링된 궤적 정렬을 수행

TUM RGB-D
- calibration을 사용할 때 SOTA trajectory error을 달성(표 1 참고)
- DROID-SLAM, DPV-SLAM 및 GO SLAM과 같은 이전 SOTA 알고리즘은 DROID-SLAM이 제안한 기초적인 매칭 및 end-to-end system 위에 구축됨
- 제안 방법은 즉각적으로 작동하면서 다른 시스템보다 더 나은 성능을 낼 수 있는 two-view geometric prior을 활용하는 시스템 제안
- uncalibrated system은 DROID-SLAM보다 현저히 우수한 성능을 발휘
    - DROID-SLAM - 첫 번째 이미지에서 Geo Calib[48]을 사용하여 내재적 값을 보정하여 사용
- 전체 시퀀스에 걸쳐 고정된 카메라 모델을 가정하지 않고 이를 달성
- 하위 문제를 해결하는 이전 방법보다 dense uncalibrated SLAM을 위한 3D prior의 가치를 입증
- uncalibrated SLAM 결과는 known calibration을 사용하는 최근의 DPV-SLAM과도 비교할만함

7-Scenes
- NICER-SLAM와 동일한 시퀀스 사용
- NICER-SLAM[58]과 DROID SLAM 모두를 능가
    - 제안 방법은 단일 3D 재구성 우선 순위를 사용
    - NICER-SLAM은 다수의 깊이, 법선 및 광학 흐름 네트워크 사용

ETH3D-SLAM
- RGB-D 방법에 대해서만 평가
- 공식 private evaluation을 위한 ATE 기준이 monocular 방법에는 너무 엄격하여 train sequence에서 여러 SOTA monocular 시스템을 평가하고 ATE curve 생성
- 빠른 카메라 움직임을 포함한 시퀀스가 포함되어 있어 모든 방법에 대해 프레임을 subsample하지 않음
- 다른 방법은 더 정확한 궤적을 가질 수 있지만, 제안 방법은 강건성 측면에서 긴 꼬리를 가지고 있어 최고의 ATE와 AUC(Area-Under-Curve)를 모두 달성

EuRoC(표 3)
- 11개의 EuRoC sequence 전반에 걸친 평균 ATE를 보고
- uncalibrated인 경우, 왜곡이 상당하여 MASt3R이 아직 그러한 카메라 모델에 대해 훈련되지 않음을 발견
- 이미지를 undistort했지만 파이프라인의 나머지 부분에 보정을 제공하지 않음
- 제안 방법은 DROID-SLAM에 의해 능가되지만, 10% grayscale 이미지를 사용하여 훈련을 명시적으로 보강
- 0.041m ATE는 매우 정확하며 [22]의 비교에서 모든 성능이 우수한 방법들이 DROID-SLAM의 기초 위에 구축됨
- 본 논문은 3D recostruction prior을 사용하는 새로운 방법을 제시

### 4.2 Dense Geometry Evaluation

### 4.4 Component Analysis

매칭 방법 비교(표 4)
- feature refinement를 통한 병렬 투영 매칭은 가장 높은 정확도와 가장 빠른 실행 시간 달성
    - MASt3R 매칭을 모든 픽셀에 수행하면 2초 소요
    - 제안 매칭은 2ms 만 소요하여 FPS 40배 향상

canonical pointmap 업데이트 방법 비교(표 5)
- 가장 최근 pointmap 선택  
-> 드리프트 발생
- 첫 번째 pointmap 선택  
-> 충분한 기준선 부족
- calibration이 주어진 경우
    - weighted fusuion과 highest median confidence를 사용한 pointmap 선택이 비슷한 성능
- calibration이 없는 경우
    - weighted fusuion이 가장 낮은 ATE 기록
    - EuRoC에서 ATE를 1.3cm 개선
    - 카메라 모델 간의 융합이 중요함
- uncalibrated tracking and backend optimization(표 6)
    - ray error formulation이 부정확한 depth 예측을 포함하는 3D 점 오류를 사용할 때보다 성능을 개선
- loop closure(표 7)
    - pose 및 기하학적 정확성을 개선
    - 더 긴 sequence에서 의미 있는 향상을 보임
    - MASt3R의 출력에 여전히 편향이 포함되어 드리프트를 유발함
        - 제안한 구성요소는 이를 완화


## 5. Limitations and Future Work

frontend에서 pointmap을 필터링하여 정확한 geometry를 추정할 수 있지만, 전체 Global optimization에서 모든 geometry를 정제하지는 않음
- DROID-SLAM은 BA를 통해 픽셀당 깊이를 최적화하지만 이 프레임은 불일치하는 geometry를 허용함

향후 방향
- 3D에서 pointmap을 global에서 일관되게 만들면서 원본 MASt3R 예측의 일관성을 유지할 수 있도록 함

MASt3R은 핀홀 이미지에서만 훈련되기 때문에 geometry 예측은 왜곡이 증가함에 따라 저하됨
- 미래에는 다양한 카메라 모델에 대해 모델이 훈련되어 parametric 카메라 모델을 가정하지 않는 본 논문의 프레임워크와 호환될 것

전체 해상도에서 decoder을 사용하는 것은 low-latency tracking 및 

우리가 프론트엔드에서 포인트맵을 필터링하여 정확한 기하학을 추정할 수 있지만, 현재 전체 글로벌 최적화에서 모든 기하학을 정제하지는 않습니다. DROID SLAM은 번들 조정을 통해 픽셀당 깊이를 최적화하지만, 이 프레임워크는 불일치하는 기하학을 허용합니다. 3D에서 포인트맵을 전 세계적으로 일관되게 만들면서 원본 MASt3R 예측의 일관성을 유지할 수 있는 방법이 향후 작업을 위한 흥미로운 방향이 될 것입니다. MASt3R는 핀홀 이미지가 있는 이미지에서만 훈련되기 때문에, 기하학 예측은 왜곡이 증가함에 따라 저하됩니다. 그러나 미래에는 다양한 카메라 모델에 대해 모델이 훈련되어, 파라메트릭 카메라 모델을 가정하지 않는 우리의 프레임워크와 호환될 것입니다. 또한, 현재 전체 해상도에서 디코더를 사용하는 것은 특히 저지연 추적 및 루프 닫기 후보 체크에 있어 병목 현상입니다. 네트워크 효율성을 향상시키는 것은 전체 시스템 효율성에 도움이 될 것입니다.