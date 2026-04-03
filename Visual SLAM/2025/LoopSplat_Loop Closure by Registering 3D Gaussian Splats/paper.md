# LoopSplat: Loop Closure by Registering 3D Gaussian Splats

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
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

![alt text](./images/Fig%201.png)

> **Figure 1. ScanNet scene0054에서 dense reconstruction**  
> LoopSplat은 기하학적 정확성, 견고한 tracking, 고품질 re-rendering에서 우수한 성능을 보임  
> 3DGS를 활용한 전역적으로 일관된 재구성 접근 방식 덕분에 가능함

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

**Dense Visual SLAM**

- Curless & Levoy[16]
    - truncated signed distance function을 활용한 dense 3D mapping의 길을 열음
- KinectFusion[53]
    - 실시간 SLAM이 일반 depth sensor에서 가능함을 보임
- [18, 34, 49, 54, 55]
    - 장면 크기에 따른 cubic memory scaling 문제를 해결하기 위해 맵 압축을 위해 voxel hashing과 oc-trees를 활용
- [10, 41, 47, 69, 90]
    - point-based representation 사용
- [36, 48, 65, 88, 95, 98]
    - surfel과 neural points 또는 3D Gaussian 활용

포즈 누적 오류 문제를 해결하기 위해 전역적으로 일관된 dense SLAM 방법들이 개발됨
- [5, 8, 13, 18, 22, 27, 34, 35, 40, 44-46]
    - global map을 submap으로 나눔
    - 이후 phose graph 최적화를 통해 submap간의 변형을 수행  
    [8, 13, 19, 20, 26-28, 35, 39, 40, 45, 46, 49, 60, 68, 72, 78, 82,89]
- [8, 15, 18, 28, 49, 68, 78, 79, 89, 91, 101]
    - Global Bundle Adjustment를 통해 개선
- [36, 48, 88, 95]
    - RGB-D 입력을 사용하는 3D Gaussian SLAM
    - global 일관성을 고려하지 않아 지도와 pose 추정에서 오류가 누적됨

가장 유사한 연구
- Loopy-SLAM[40]
    - Point-SLAM[63]의 명시적 neural point cloud 표현을 활용
    - submap에서의 loop closure을 통해 global 일관성을 갖춤

Loop Splat의 세 가지 개선
1. 3DGS를 직접 등록하여 상대 포즈 제약조건의 정확성과 효율성 향상  
FPFH[62]와 RANSAC 후 ICP[4]와 같은 전통적인 기법을 사용하지 않음
2. 등록을 위해 별도의 submap mesh작업을 생략하고 3D 가우시안을 직접 활용
3. loop detection에서 이미지 매칭과 submap 간 overlap을 결합하여 탐지 성능을 개선  
[40]에서처럼 이미지 내용만 사용했을 때보다 더 나은 결과 제공

**Geometric Registration**

- pose graph를 위한 edge 제약을 구축하는 중요한 요소
- point cloud 정합은 두 point cloud fragment를 동일한 좌표 체계로 정렬하는 rigid transformation을 찾는 것을 목표로 함
- 전통적인 방법
    - feature matching을 위해 hand-crafted local descriptor을 활용
    - RANSAC을 사용하여 pose 추정
- 최근 학습 기반 방법
    - patch 기반 local descriptor 또는 효율적인 fully-convolutional 방법 사용
- BUFFER[1]
    - keypoint 검출을 위한 fully convolutional backbone과 feature description을 위한 patch-based network 결합
    - local descriptor의 효율성과 일반화를 균형있게 구현
- Predator[32]
    - 낮은 중복률을 갖는 fragment 정합 문제 해결을 위해 keypoint sampling을 안내하기 위해 attention 메커니즘을 사용하여 알고리즘의 강인성 크게 향상
    - coarse-to-fine matching을 통해 더욱 강화됨
- point cloud는 연속적이고 시점에 따른 변화를 반영함
- 다중 scale 표현 능력을 가진 NeRFs와 달리 SLAM에서 복잡한 3D 장면을 완전히 포착하는 능력 제한됨

Neural Radiance Fields
- 장면 재구성, 장면 이해, 자율 주행, SLAM 등 다양한 응용 분야에서 널리 사용
- 대규모 장면 모델링 시, 메모리 관리 및 장면을 블록 단위로 나눠야 함
- 서로 다른 블록들을 합치기 위해 등록(register)하는 문제가 연구 주제로 등장
    - iNeRF[92]
        - analysis-by-synthesis를 통해 쿼리 이미지를 NeRF 맵에 정렬
        - 카메라의 위치를 최적화하여 렌더링된 이미지가 쿼리 이미지와 일치하도록 함
        - non-convex 특성으로 인해 local minima에 갇힐 수 있으므로 local refinement에만 적합
    - NeRF2NeRF[24]
        - density field에서 surface points를 추출하고 수동으로 선택한 keypoint를 정렬하여 카메라 위치를 추정
    - DReg-NeRF[11]
        - 표면 점을 먼저 추출한 후 완전 합성곱(feature extraction backbone)을 적용하는 방식
        - point cloud 등록 기법과 유사하게 NeRF 등록 문제를 해결
- 최근에는 효율적인 rasterization과 명시적 표현(explicit representation)에 따른 유연한 편집 기능 덕에 Gaussian Splatting이 NeRF를 대체하기 시작함
    - GaussReg[9]
        - 3DGS의 빠른 렌더링을 활용하여 learning-based 3DGS registration을 개척
        - 이전의 모든 NeRF 및 3DGS 등록 방법[9, 11, 24, 92]는 training view에 대해 실제 카메라 위치를 가정하고 있어 실제 SLAM 시나리오에는 적합하지 않음
        - 이러한 방법들은 소규모 장면에서 pairwise 등록만 탐구함
- 제안 방법은 어떠한 훈련이나 전처리 없이 SLAM Frontend에서 추정된 카메라 위치를 직접 사용
    - loop closure에 즉시 통합 가능


## 3. LoopSplat

![alt text](./images/Fig%202.png)

> **Figure 2. LoopSplat overview**  
> LoopSplat은 tracking, mapping, maintaining global consistency를 위한 통합 장면 표현으로 GS를 사용하는 coupled RGB-D SLAM system  
> Frontend에서는 카메라 위치를 지속적으로 추정하면서 가우시안 splat을 사용하여 장면 구성  
> 카메라가 미리 정의된 임계값을 초과하여 이동하면 현재 submap이 확정되고 새로운 submap을 시작  
> backend loop closure 모듈은 위치 재방문을 모니터링  
> loop가 감지되면 제안된 3DGS 등록에서 파생된 loop edge 제약 조건을 통합하여 pose graph 생성  
> 이후 Pose Graph Optimization(PGO)가 실행되어 카메라 포즈와 submap을 모두 정밀하게 조정하여 전체 공간적 일관성 보장

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
    (즉, 겹쳐있지 않을 때만)

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

- 견고하고 정확한 정합을 위해 source와 target submaps $P$와 $Q$ 간의 대략적인 겹침 정도를 알아야 함
    - 특징 유사성을 비교하여 추출 가능
- Gaussian splat의 평균값이 point cloud를 형성
    - 하지만 local feature을 매칭하여 overlap 영역을 직접 추정하는 것은 잘 작동하지 않음
- 본 논문에서는 각 submap에서 시각적 내용이 유사한 시점들을 식별함
    - 모든 keyframe을 NetVLAD를 통해 통과시켜 전역 descriptor 추출
    - 두 세트의 keyframe 간 cosine 유사도를 계산하고 정합을 위해 상위 k개의 쌍을 유지

**Registration as Keyframe Localization**

- 3DGS submap과 그 관측 지점을 하나의 rigid body로 처리할 수 있다는 점을 고려하여 3DGS 정합을 keyframe 위치 추정 문제로 접근 제안
- 선택된 관측 지점 $v_i^P$에 대해, Q 내에서의 카메라 자세 $T_i^q$를 결정하면 Q에서 $v_i^p$와 동일한 RGB-D 이미지 렌더링 가능
- rigid transformation $T_{P \rightarrow Q}$는 $T_i^q \cdot T_i^{-1}$로 계산될 수 있음
    - keyframe 위치 추정 과정에서 Q의 파라미터는 고정한 채, 렌더링 loss $\mathcal{L} = \mathcal{L}_{col} + \mathcal{L}_{depth}$를 최소화하여 rigid transformation 최적화
    - $\mathcal{L}_{col}, \mathcal{L}_{depth}$은 $\mathcal{L}_1$ loss
- 선택된 관측 지점에 대해, $V^P$의 관측 지점에서는 $P$에서 $Q$로, $V^Q$의 관측 지점에서는 $Q$에서 $P$로 rigid transformation을 병렬로 추정
- 최적화 완료 시 렌더링 잔차 $\epsilon$도 저장됨
- 추정된 겹침 영역에서 샘플링된 상위 $k$개의 관측 지점을 선택 관측 지점으로 사용함으로써 비겹침 관측 지점에서의 중복 없이 정합 효율 크게 향상됨
- 관측 지점 변환을 먼저 추정 후, 이를 사용하여 submap의 전역 변환을 계산

**Multi-view Pose Refinement**

transformation set $\{ (T_{P \curvearrowleft Q}, \epsilon)_i \}_{i=1}^{2k}$가 주어졌을 때
- 처음 $k$개의 추정치는 $P \rightarrow Q$에서, 마지막 $k$개의 추정치는 $Q \rightarrow P$에서 나온 것이라면 transformation $\bar{T}_{P \rightarrow Q}$에 대한 global 합의를 찾아야 함
- 렌더링 residual은 transformed viewpoint가 original 관측을 얼마나 잘 맞추는지를 나타냄
- 각 추정치에 대한 가중치로 residual의 역수를 사용하고 weighted rotation averaging을 적용하여 전체 회전을 계산

$$
\bar{R} = \arg \min_{R \in SO(3)} 
\sum_{i=1}^{k} \frac{1}{\varepsilon_i} \| R - R_i \|_F^2 
+ \sum_{i=k+1}^{2k} \frac{1}{\varepsilon_i} \| R - R_i^{-1} \|_F^2
\tag{7}
$$

> $||\cdot||_F^2$: Frobenius norm

- global translation은 개별 추정치들의 weighted mean으로 계산됨

### 3.3 Loop Closure with 3DGS

Loop closure
- 과거의 submap과 keyframe에 대해 현재 추정치와의 상대 변환을 찾아 전역적 일관성을 보장하는 과정
- 새로운 submap이 생성될 때 이 과정이 시작됨
    - 새로운 loop 가 감지되면 모든 과거 submap을 포함하는 pose graph가 구성됨
- loop edge 제약은 3DGS 기반 정합(registration)을 통해 계산됨
- Pose Graph Optimization(PGO)를 수행하여 전역적으로 일관된 multi-way registration을 달성

Loop Closure Detection
- 동일한 장소를 재방문했는지 효과적으로 탐지하기 위해 사전 학습된 NetVLAD를 사용하여 global descriptor $d \in \mathbb{R}^{1024}$를 추출
- $i$번째 submap 내 모든 keyframe의 cosine 유사도를 계산하고, $p$번째 백분위수에 해당하는 self-similarity score $s_{self}^i$를 구함
- 같은 방식으로 $i, j$번째 submap 간의 cross-similarity $s_{cross}^{i,j}$를 계산
- 새로운 loop는 다음 조건을 만족할 때 추가됨  
$ s_{cross}^{i, j} > \min(s_{self}^i, s_{self}^j)$
- 시각적 유사도만으로 loop closure을 판단하면 잘못된 loop edge가 생겨 PGO 성능을 저하시킬 수 있음
- 이를 방지하기 위해, 두 submap의 gaussian 간 초기 geometric overlap 비율 $r$을 추가로 평가하고 $r > 0.2$인 경우에만 loop를 유지

Pose Graph Optimization
- 새로운 loop가 감지될 때마다 새로운 pose graph 생성
    - 이 graph는 이전 graph의 연결을 유지하면서 새로운 submap으로 인해 추가된 edge를 포함
- 각 submap에 대한 상대 pose 보정은 다음과 같음
$\{ T_{c^i} \in SE(3) \}$
> $T_{c^i}$: i번째 submap에 적용되는 보정

- 인접 node에 연결된 node와 edge들(예: odometry edges)는 identity matrix로 초기화됨
- loop가 감지되었을 때 loop edge 제약이 추가되고 Gaussian splatting 정합 결과에 따라 초기화됨
- edge의 information matrix는 Gaussian의 중심에서 직접 계산되어 pose 그래프에 포함됨
- loop detection 이후 PGO가 실행됨
    - online process 기반 robust formulation 사용

Globally Consistent Map Adjustment
- PGO 결과로 $N_s$개의 submap에 대한 pose 보정 집합을 얻음  
$
\displaystyle
\{ T_{c^i} = [R_{c^i} | t_{c^i}] \}_{i=1}^{N_s}
$
> $c_i$: submap $i$에 대한 보정

- 각 submap에 대해 카메라 포즈, 가우시안 평균, 공분산을 업데이트

$$
T_j ← T_{c^i} T_j
\tag{8}
$$
$$
μ_i ← R_{c^i} μ_{S^i} + t_{c^i}, \quad
Σ_i ← R_{c^i} Σ_{S^i} R_{c^i}^T
\tag{9}
$$

- 가우시안 map 크기를 줄이고 pose 추정 정확도를 향상시키기 위해 spherical harmonics는 생략함