# RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes

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
- [paper](https://arxiv.org/abs/2502.15633) (arXiv 2025)

---
요약

- 문제점
    - 

---

## Abstract

3D Gaussian Splatting(3DGS)
- 고충실도의 새로운 뷰 생성 가능

이전 GS 기반 방법의 한계
- 주로 실내 장면을 목표로 함
- RGB-D 센서나 pre-trained depth estimation model에 의존
- 실외 시나리오에서의 성능 저조

OpenGS-SLAM
- 무한한 실외 장면을 위한 RGB-only GS SLAM
- 포즈 추정을 위해 프레임 간 일관된 pointmap을 생성하는 pointmap regression network 사용
    - 일반적인 깊이 맵과 비교하여 여러 view에 걸쳐 공간적 관계와 장면 기하학을 포함하여 강력한 카메라 포즈 추정을 가능하게 함
- 추정된 카메라 포즈를 3DGS 렌더링과 통합하여 최적화 가능한 end-to-end 파이프라인으로 제안
- 카메라 포즈와 3DGS 장면 매개변수의 동시 최적화를 달성하여 시스템 추적 정확성 크게 향상
    - pointmap 회귀 네트워크를 위한 적응형 scale mapper을 설계하여 3DGS 맵 표현에 대한 pointmap 매핑을 더 정확하게 제공
- Waymo 데이터셋 실험
    - 이전 3DGS 방법의 추적 오류를 9.8% 줄임
    - 새로운 view 합성에서 SOTA 달성


## 1. Introduction

SLAM
- 자율 주행, 로봇 공학 및 VR 분야에서 광범위하게 적용됨
- 3D 표현 문제
    - 고충실도 시각 효과
    - 정밀한 위치 추정 기능 달성
- 이전 연구의 두 가지 분류
    - dense representation based method
        - 관찰된 영역을 효과적으로 렌더링
        - 새로운 view 합성 능력에 부족함이 있음
    - neural implicit representation based method
        - end-to-end 미분 가능한 dense visual slam 시스템을 개발하여 강력한 성능을 보임
        - 낮은 계산 효율성, 명시적 모델링 포즈 부족

3DGS scene representation
- 고충실도 새로운 view 합성 가능
- 낮은 메모리 요구 사항으로 실시간 렌더링 달성
- 기존 연구의 문제
    - 고품질 depth 입력에 크게 의존
    - 제한된 카메라 이동이 있는 소규모 실내 장면 시나리오에서만 작동
- RGB만 사용하는 무한한 외부 장면 처리의 어려움
    - 포즈 정확도와 3DGS 초기화에 영향을 미치는 정확한 깊이와 scale 추정의 어려움
    - 효과적인 제약이 부족한 이미지 overlap 및 singular viewing angle로 수렴 어려움

**OpenGS-SLAM**
- RGB 정보만을 사용하여 장면 표현
- 고충실도 이미지 생성을 위해 3DGS 활용
- 프레임 간의 일관된 pointmap 생성을 위해 pointmap 회귀 네트워크 사용
    - 여러 표준 view에서 3D 구조를 저장
    - 시점 관계, 2D-to-3D  대응 및 장면 기하학이 포함됨
    - 더 강력한 카메라 포즈 추정을 가능하게 하며, pre-trained depth network의 부정확성 문제를 효과적으로 완화함
- 카메라 포즈 추정과 3DGS 렌더링 통합
- 포즈와 3DGS 매개변수의 공동 최적화를 달성하여 시스템 추적 정확성을 향상시킴
- 더 정확하게 pointmap을 3DGS 맵 표현으로 매핑하기 위해 적응형 scale mapper와 dynamic learning rate 조정 전략 설계
- Waymo 데이터셋 실험
    - OpenGS-SLAM은 기존 3DGS 방법의 추적 오류를 9.8% 줄임
- 새로운 view 합성에서 SOTA 결과 달성

논문의 기여
1. 무한한 야외 장면을 위한 RGB-only 3DGS SLAM 처음으로 제안
2. 자세 추정에서 3DGS 렌더링까지 이르는 end-to-end 파이프라인 및 pointmap 회귀 네트워크를 통합한 시스템 제안
    - 자세와 장면 매개변수를 동시에 최적화하여 추적 정확도와 안정성 향상
3. 적응형 scale mapper와 동적 learning rate 조정을 통해 Waymo 데이터셋에서 novel view 합성에서 SOTA 달성



## 2. Related Work


## 3. Method

### A. SLAM System Overview

![alt text](./images/Figure%201.png)
> **Figure 1. SLAM System Pipeline**  
> 각 프레임은 추적을 위한 RGB 이미지  
> 현재와 이전 프레임은 자세 추정을 위한 pointmap 회귀 네트워크에 쌍으로 입력됨  
> 이후 현재 가우시안 맵을 기반으로 자세 최적화 진행  
> 중요 프레임에서는 매핑 수행

Figure 1은 system overview를 제공

### B. Tracking

1. Pairwise Pointmap Regression and Pose Estimation
    - 이전 연구[2, 9]에서는 3DGS와 NeRF가 주로 카메라 이동이 최소화되고 시점 각도가 밀집한 실내 및 소규모 장면에 중점을 둠
        - NeRF 또는 3DGS를 사용하여 카메라 자세 직접 회귀 가능
    - 야외 장면은 일반적으로 차량 기반 사진 촬영을 포함
        - 상당한 이동 진폭
        - 상대적으로 희박한 시점 각도
        - 카메라 자세의 직접적인 회귀가 어려워짐
    - pointmap이 viewpoint relationships를 포함하고 있다는 점을 고려하여 현재 프레임에 대한 강건하고 신속한 카메라 자세 추정을 목표로 하는 pairwise pointmap regression network 기반의 새로운 자세 추정 방법 제안
        - 구체적인 방법
            - 두 개의 입력 이미지 2D 이미지 $I^1, I^2 \in \mathbb{R}^{W \times H}$ 를 가정
            - 이러한 이미지의 각 픽셀에 해당하는 3D point $X^1, X^2 \in \mathbb{R}^{W \times H \times 3}$ 을 pointmap으로 정의
        - pre-trained pointmap regression network
            - ViT 인코더
            - transformer decoder
                - self-attention 및 cross-attention layers이 있음
            - MLP regression head
                - consecutive frame 이미지에 대한 pointmap 생성을 위함
        - 두 개의 이미지 branch 간 정보 공유는 pointmap의 올바른 정렬을 촉진
        - 네트워크는 예측된 pointmap과 실제 point간의 euclidean 거리를 최소화함으로써 훈련됨
        $$
        \displaystyle
        \begin{aligned}
        & L_reg = \sum_{v=(1,2)} \sum_{i \in D} \| \frac{1}{z} X_i^v - \frac{1}{\bar{z}} \bar{X}_i^v \|
        & \tag{1}
        \end{aligned}
        $$
        > $D \subseteq \{1 ... W\} \times \{1 ... H\}$  
        > $z$: scale normalization factor. $z = \frac{1}{2|D|} \sum_{v=(1,2)} \sum_{i \in D} \| X_i^v \|$
        - pointmap은 비직관적으로 보일 수 있지만, 이미지 공간에서 3D 형상을 효과적으로 표현하고 깊이 추정 품질에 얽매이지 않고 서로 다른 viewpoint의 ray들 사이를 삼각 측량할 수 있게 함
        - 상대 포즈 $T_{trans}^k 추정: well-established RANSAC[35]과 PnP[36]로 추정
        - k번째 프레임의 포즈 계산: $T_k = T_{trans}^k T^{k-1}$

2. 포즈 최적화
    - 목표: 추정된 포즈로부터 렌더링한 이미지에 기반한 photometric 손실이 카메라 포즈에 대해 미분 가능하도록 만들어 정밀한 카메라 포즈 추적 수행
    - 카메라 포즈 $T$는 3D 공간의 회전 및 translation으로 이루어지며, Lie group $SE(3)$에 속함
        - $SE(3)$은 비선형 군 구조를 가지며 덧셈에 대해 닫혀 있지 않아 표준적인 gradient-based methods를 적용하기 어려움
        - 따라서, $SE(3)$을 그에 대항하는 리 대수 $se(3)으로 선형화하여 표준 경사 하강 기법을 사용할 수 있게 함
    - $SE(3)$ 선형화
        - exponential mapping $\exp(\xi)$를 사용
        - $\xi = (w, v)$는 회전($w$)와 translation($v$)의 무한소 생성자(infinitesimal generators)를 나타냄
        - 리 대수에서 유도된 Jacobian 행렬 사용 시
            - 포토메트릭 손실 $L_{pho}$에 대한 카메라 포즈 $T_{CW}$의 미분 가능성을 보장
            - 불필요한 연산 제거 가능
        - 미분 계산 절차
            - chain rule을 사용해 2D 가우시안 $N(\mu_I, \Sigma_I)$의 $T_{CW}$에 대한 도함수 계산
            - 이 2ED 가우시안은 EWA(Elliptical Weighted Average) splatting[38]을 3D 가우시안 $N(\mu_W, \Sigma_W)$에 적용해 얻음
            $$
            \displaystyle
            \frac{\partial \mu_I}{\partial T_{CW}} = \frac{\partial \mu_I}{\partial \mu_C} \cdot [I - \mu_c^{\times}]
            \tag{2}
            $$
            $$
            \displaystyle
            \frac{\partial \Sigma_I}{\partial T_{CW}}
            = \frac{\partial \Sigma_I}{\partial J} \cdot \frac{\partial J}{\partial \mu_C} \cdot [I - \mu_c^{\times}]
            + \frac{\partial \Sigma_I}{\partial W} \cdot
            \begin{bmatrix}
            0 & -W^{\times}_{:,1} \\
            0 & -W^{\times}_{:,2} \\
            0 & -W^{\times}_{:,3}
            \end{bmatrix}
            \tag{3}
            $$
            > $^\times$: 3D 벡터의 skew-symmetric matrix 변환 연산  
            > $W$: $T_{CW}$의 회전 성분  
            > $W_{:, 3}^\times$: W의 skew-symmetric matrix에서 i번째 열
    - photometric loss $L_{pho}$
    $$
    \displaystyle
    L_{\text{pho}} = \| r(G, T_{CW}) - \bar{I} \|_1
    \tag{4}
    $$
    > $r$: 픽셀 단위 미분 가능 렌더링 함수. 가우시안 집합 $\mathcal{G}$와 카메라 포즈 $T_{CW}$를 입력받아 이미지 생성
    > $\bar{I}$: GT 이미지
    - 포토메트릭 손실 $L_{pho}$의 포즈 $T_{CW}$에 대한 기울기 계산
    $$
    \displaystyle
    \nabla_T L_{\text{pho}} =
    \frac{\partial L_{\text{pho}}}{\partial r} \cdot
    \left(
    \frac{\partial r}{\partial \mu_I} \cdot
    \frac{\partial \mu_I}{\partial T_{CW}}
    +
    \frac{\partial r}{\partial \Sigma_I} \cdot
    \frac{\partial \Sigma_I}{\partial T_{CW}}
    \right)
    \tag{5}
    $$
    - 렌더링 함수 미분을 통해 incremental pose update를 hpotometric loss와 연결
    - end-to-end 최적화로 인해 고정밀, 강건한 pose tracking

### C. 3D Gaussian Scene Representation

1. 3D Gaussian Map
    - volumetric scene 표현의 장점을 유지하며 실시간 렌더링 구현
    - 장면을 3D 가우시안 집합으로 모델링
        - 각 가우시안은 중심 $\mu$와 공분산 행렬 $\Sigma$로 정의
        - 가우시안 분포:
        $$
        \displaystyle
        G(x) = e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}
        \tag{6}
        $$
        > $x$: 3D 공간 내 임의의 위치
        - 공분산 행렬 분해:
        $$
        \displaystyle
        \Sigma = R S S^T R^T
        \tag{7}
        $$
        - 분해를 통해 scene geometry 제어가 용이해짐
    - View dependent radiance를 표현하는 spherical harmonics 생략
    - 3D Gaussian을 2D 평면으로 투영
    - tile-based rasterization으로 효율적인 정렬 및 blending
    - 빠르고 정확한 색상 렌더링 달성
    - 픽셀 $x'$ 색상 계산:
    $$
    \displaystyle
    C(x') = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
    \tag{8}
    $$
    > $N: 픽셀 $x'$에 영향을 주는 가우시안 집합
    - dynamic 환경에서도 빠른 적응 가능
    - 전체 과정이 미분 가능하여 학습 및 조정이 용이함
2. Adaptive Scale Mapper
    - 목적:
        - pointmap을 3D 가우시안 맵에 매핑
        - pointmap 회귀 오차에 대한 강건성 향상
    - 동작 방식
        - 새로운 가우시안을 삽입하기 전에 pointmap에 scale-mapping 수행
        - 연속된 프레임 간 3D 매칭을 통해 scale-mapping
        - 새 프레임(k - 1, k, k + 1)의 cross-view 이미지로부터 pointmap 생성
            - $\{X^{k-1}, X^k\}, \{X'^k, X'^{k + 1}\}$ 매칭
    - 스케일 변화 비율 정의:
    $$
    \displaystyle
    \rho_{ij} = \frac{\| X'^k_i - X'^{k+1}_j \|}{\| X^{k-1}_i - X^k_j \|}
    \tag{9}
    $$
    > $X^{k-1}_i, X^k_j$: 프레임 k에서의 대응점  
    > $X'^k_i, X'^{k+1}_j$: 프레임 k+1에서의 대응점
    - 이 비율은 프레임 k에서 k+1에서의 스케일 변화를 나타냄
    - 다중 $\rho_{ij}$ 값을 평균내어 전체 scene의 평균 scale 변화를 추정
    - sequence동안 scale consistency를 유지하려면 첫 번째 프레임을 기준으로 각 프레임의 scale factor을 계산하고, 이를 누적 곱
    $$
    \displaystyle
    S_k = S_{k-1} \cdot \frac{1}{N} \sum_{ij} \frac{\| X'_k{}_i - X'_{k+1}{}_j \|}{\| X_{k-1}{}_i - X_k{}_j \|}
    \tag{10}
    $$
    - frame-to-frame scale consistency를 보장
        - scale factor을 amp subsequent frame pointmap 좌표에서 사용할 수 있게 함
        - 정밀한 3D 매핑 및 위치 추적에 중요
    - Sparse Subsampling
        - 모든 3D Gaussian point가 mapping에 기여하지 않음
        - hierarchical 구조를 사용해 3D 가우시안 포인트 수를 효과적으로 제어
        - 매핑 품질을 유지하고 처리 시간 단축

### D. Mapping

1. Keyframe Management

키프레임 선택 전략
- 충분한 viewpoint 중복성을 보장하면서 중복된 키프레임을 피해야 함
- 모든 키프레임과 함께 가우시안 장면과 카메라 포즈를 공동 최적화하는 것은 불가능함
    - 동일한 영역을 관찰하는 비중복 키프레임을 선택하기 위해 local keyframe window $W$를 관리하여 후속 매핑 최적화에 더 나은 multi-view 제약 조건을 제공
- [2]에서의 키프레임 관리 전략을 채택
    - 가시성을 기반으로 한 키프레임 선택
    - 가장 최근 키프레임과 중복 평가를 통한 로컬 윈도우 관리

2. Gaussian Map Optimization
- Local Window BA
    - 각 키프레임마다 관리 중인 로컬 키프레임 윈도우 $W$ 내에서 가우시안 속성과 카메라 포즈를 공동 최적화
- photometric loss를 최소화하는 방식으로 최적화 수행
- 문제점: 타원체가 과도하게 늘어나는 현상 발생 가능
    - 이를 방지하기 위해 등방성 정규화(isotropic regularization) 적용
    $$
    \displaystyle
    L_{iso} = \sum_{i=1}^{|G|} \| s_i - \tilde{s}_i \cdot \mathbf{1} \|_1
    \tag{11}
    $$
    - scaling parameter $s_i$에 패널티를 적용하기 위하 mapping 최적화 작업은 다음과 같이 요약됨
    $$
    \displaystyle
    \min_{\substack{T^k_{CW} \in SE(3) \\ \forall k \in W},\ \mathcal{G}} \sum_{k \in W} L^{k}_{pho} + \lambda_{iso} L_{iso}
    $$

3. Adaptive Learning Rate Adjustment

- 전통적인 실내 SLAM 데이터셋에서는 카메라가 작은 장면을 반복적으로 촬영  
-> 누적 반복 횟수(N_{iter})가 커질수록 학습률이 점진적으로 감소
- 본 연구의 실외 데이터는 동일 지역 재방문이 거의 없음
- 직선 도로 주행 시에는 학습률이 서서히 감소하는 것이 좋지만, 경사로나 회전 구간에서는 새로운 장면 최적화를 위해 학습률을 높여야 함
- 회전 각도 기반 적응형 학습률 조정 제안
- 방법
    1. 누적 반복 횟수 $N_{iter}$을 기준으로 기본 학습률 조정
    2. 현재 키프레임 $R_1$과 이전 키프레임 $R_0$의 회전 행렬을 사용해 상대 회전행렬 계산: $R_{diff} = R_0^T R_1$
    3. 회전 라디안 계산
    $$
    \displaystyle
    \theta_{rad} = \cos^{-1} \frac{\mathrm{trace}(R_{diff}) - 1}{2}
    $$
    4. 라디안을 도($\theta$) 단위로 변환
    5. 만약 $\theta > 2$면 누적 반복 횟수 조정
    $$
    \displaystyle
    N^{new}_{iter} = N_{iter} \times \left( 1 - \frac{\theta}{90} \right)
    $$
    6. 회전이 90도에 도달하면 반복 횟수 리셋
    7. 제곱근 조정을 통해 작은 각도 변화에도 학습률이 더 크게 증가하도록 함
- 후반부 매핑 품질 향상됨

## 4. Experiments