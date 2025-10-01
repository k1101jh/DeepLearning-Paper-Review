# LoopSplat: Loop Closure by Registering 3D Gaussian Splats



---

- Visual SLAM
- Gaussian Splatting

---

url
- [paper](https://ieeexplore.ieee.org/abstract/document/11125565/) (3DV 2025)
- [project](https://loopsplat.github.io/)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

3DGS를 기반으로 한 SLAM은 최근 보다 정확하고 밀도 높은 3D 장면 지도를 만드는 데 가능성을 보임
- 기존 3DGS 기반 방법들은 loop closure 및/또는 global bundle adjustment를 통해 장면의 전역 일관성을 해결하지 못함

LoopSplat
- RGB-D 이미지를 입력으로 받아 3DGS submap과 frame-to-model tracking을 통해 밀도 있는 매핑을 수행
- 온라인으로 loop closure을 trigger하고 3DGS registration을 통해 submap 간의 상대 loop edge 제약을 직접 계산
    - 기존 global-to-local point cloud registration 대비 효율성과 정확도 향상
- robust pose graph 최적화 formulation을 사용하고 submap을 강제로 정렬하여 전역 일관성 달성
- 합성 Replica와 실제 TUM-RGBD, ScanNet, ScanNet++ 데이터셋 평가 결과 기존의 dense RGB-D SLAM 방법에 비해 tracking, mapping, rendering에서 경쟁력 있거나 우수한 성능을 보임

## 1. Introduction

기존 Dense SLAM
- mapping과 tracking을 분리/결합
    - 분리: 맵 정보를 tracking에 활용하지 않아 정보 공유 비효율과 연산 중복 초래
    - 결합: frame-to-model 프레임워크로 tracking

coupled 3DGS SLAM
- map과 pose에서 global consistency를 달성하기 위한 전략 부족
    - 자세 오차 누적, 지도 왜곡 발생
- loop closure 및/또는 Global Bundle Adjustment를 통해 글로벌 일관성을 강제하는 방법이 있음
    - GO-SLAM[101]
        - 지도를 변형시키기 위해 hash grid feature을 재학습해야 함
        - 메모리에 모든 매핑된 프레임을 저장해야 함
    - Photo-SLAM[30]
        - ORB-SLAM[52] tracker로부터 pose update를 수행하기 위해 3D 가우시안 parameter의 추가 최적화 필요
        - 메모리에 모든 매핑된 프레임을 저장해야 함
    - Loopy-SLAM[40]
        - neural point cloud의 submap을 사용
        - loop closure 이후 이를 rigidly update
        - loop edge 제약 조건 계산을 위해 global-to-local point cloud registration 사용
        - 속도가 느리고 장면 표현 자체의 특성을 활용하지 못함
        
제안 방법
- 모든 매핑된 입력 프레임을 저장하지 않고도 loop 제약을 dense map에서 직접 추출할 수 있는 결합형 SLAM 시스템 구축
- local frame-to-model tracking 및 dense mapping을 위해 3D gaussian submap을 사용하는 dense RGB-D SLAM 시스템 제안
- loop closure 탐지 및 pose graph 최적화를 통해 global consistency 달성
- 전통적인 point cloud 등록 기법이 3D 가우시안에서 loop edge 제약을 도출하는데 적합하지 않음
- 3DGS 표현에서 직접 작동하는 새로운 등록 기법 제안
- 3DGS를 tracking, mapping, global consistency 유지를 위한 통합 scene 표현으로 사용

주요 기여
1. Gaussian Splatting을 기반으로 한 결합형 RGB-D SLAM 시스템인 LoopSplat을 소개.
    - 새로운 loop closure 모듈은 Gaussian Splat에 직접 작동
    - 3D geometry와 visual scene content를 통합하여 강력한 loop detection 및 closure 제공
2. 두 개의 3DGS 표현을 등록하는 효율적인 방법 개발
    - pose graph 최적화를 위한 edge 제약을 효과적으로 추출
    - 3DGS의 빠른 rasterization을 활용하여 system에 원활하게 통합됨
    - 속도와 정확도 모두에서 전통적인 기법보다 우수함
3. 3DGS 기반 RGB-D SLAM 시스템의 tracking 및 reconstruction 성능을 향상시킴
    - 다양한 데이터셋 전반에서 눈에 띄는 향상과 강인성을 입증

## 2. Related Work

## 3. LoopSplat

### 3.1 Gaussian Splatting SLAM

- 장면을 submap 모음으로 표현
- 각 submap은 3D 가우시안 point cloud $P^s$로, 여러 주요 프레임을 모델링
$$
\displaystyle
P^s = \{ G_i(μ, Σ, o, C) | i = 1, …, N \}  
\tag{1}
$$

**Submap Initialization**

- 첫 번째 keyframe $I_f^s$로 submap 생성 시작
- 현재 프레임이 첫 keyframe 대비 변위 또는 회전 임계치 $d_{thre}, \theta_{thre}$를 초과하면 새로운 submap을 동적으로 초기화

**Frame-to-model Tracking**

- 입력 frame $I_j^s$를 현재 submap $P^s$를 사용해 localize하기 위해
    - 일정한 motion 가정으로 camera pose $T_j$ 초기화:
    $$
    \displaystyle
    T_j = T_{j−1} · T_{j−2}^{−1} · T_{j−1}
    $$
    - 렌더링된 색상 $\hat{I}_j$, depth $\hat{D}_j^s$ 와 입력 색상 $I_j$, depth $D_j$ 사이의 차이를 측정하는 tracking loss $\mathcal{L}_{tracking}(\hat{I}_j^s, \hat{D}_j^s, I_j^s, D_j^s, T_j)$를 최소화하여 $T_j$ 최적화
    - tracking을 안정화하기 위해 잘못 재구성되었거나 unobserved areas로 발생하는 큰 오류를 처리하기 위해 알파 마스크 $M_a$와 인라이어 마스크 $M_{in}$을 사용
    - 최종 tracking 손실은 다음과 같음. 유효한 픽셀에 대한 합으로 표현됨
    $$
    \displaystyle
    L_{tracking} = ∑ M_{in}·M_a·(λ_c · | \hat{I}^s_j − I^s_j |_1 + (1−λ_c) · | \hat{D}^s_j − D^s_j |_1)  
    \tag{2}
    $$
    > $\lambda_c$: 색상 loss와 depth loss의 균형을 조절하는 가중치  
    > $ || \cdot ||$: 두 이미지 간 L1 loss

**Submap Expansion**

- 고정된 간격으로 keyframe 선택
- 현재 keyframe $I_j^s$가 localize되면 효율적인 매핑을 위해 주로 sparsely covered region에서 3D Gaussian map 확장
    1. 누적 alpha 값이 임계치 $\alpha_{thre}$ 미만이거나 유의미한 깊이 차이가 발생하는 영역에서 $M_k$ point를 균일하게 샘플링
    2. 이러한 point는 anistropic 3D 가우시안으로 초기화됨. scale은 현재 submap 내 nearest neighbor distance 기준으로 정의됨
    3. 새로운 3D Gaussian splat은 반지름 $\rho$ 내에 기존 3D gaussian mean이 존재하지 않을 경우에만 현재 submap에 추가됨

**Submap Update**

- 새 가우시안 추가 후, 현재 submap 내 모든 가우시안은 고정된 횟수의 iteration만큼 최적화됨
    - rendering loss $L_{render}$을 최소화하여 최적화
- submap의 모든 keyframe에 대해 최적화
- 최소 40%의 계산은 가장 최근의 keyframe에 할당
- rendering loss는 세 가지 항으로 구성됨
$$
\displaystyle
\mathcal{L}_{render} = λ_{color}·\mathcal{L}_{color} + λ_{depth}·\mathcal{L}_{depth} + λ_{reg}·\mathcal{L}_{reg}  
\tag{3}
$$
> $\lambda$: hyperparameter

- depth loss는 렌더링된 깊이와 gt 깊이 간의 $L_1$ loss
- color supervision에는 $L_1$과 SSIM loss 사용
$$
\displaystyle
\mathcal{L}_{col} = (1 - \lambda{SSIM} \cdot |\hat{I} - I|_1 + \lambda{SSIM}(1 - SSIM(\hat{I}, I)))
\tag{4}
$$
> $\lambda{SSIM} \in [0, 1]$

- 희소하게 덮이거나 거의 관찰되지 않은 영역에서 지나치게 긴 3D gaussian을 정규화하기 위해 등방성(isotropic) 정규화 항 추가
$$
\displaystyle
\mathcal{L}_{reg} = \frac{1}{K} \Sigma_{k \in K} |s_k - \bar{s}_k|_1
\tag{5}
$$
> $s_k \in \mathbb{R}^3$: 3D 가우시안의 크기

- 최적화 과정에서 depth sensor로 측정된 geometry 유지
- 계산 시간을 줄이기 위해 가우시안을 복제하거나 가지치기하지 않음

### 3.2 Registration of Gaussian Splats

- 서로 다른 keyframe을 사용하여 재구성되고 정렬되지 않은 두 개의 겹치는 gaussian submap $\text{P}, \text{Q}$를 고려
- 목표: P를 Q에 맞추는 rigid transformation $T_{P \rightarrow Q} \in SE(3)$ 추정
- 각 submap은 다음과 같이 viewpoint 집합 $\text{V}^{\text{P}}$와 연관됨
$$
\displaystyle
\text{V}^{\text{P}} = \{ \text{v}_i^{\text{P}} = (\text{I}, \text{D}, \text{T})_i | i = 0, ..., N \}
\tag{6}
$$
> $\text{I, D}$: 개별 RGB 및 depth 측정값
> $T$: 추정한 카메라 포즈

**Overlap Estimation**

