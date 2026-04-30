# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Gaussian Splatting

---
url:
- [paper](https://arxiv.org/abs/2312.00109) (CVPR 2024)
- [project](https://city-super.github.io/scaffold-gs/)

---
- **Authors**: Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, Bo Dai
- **Affiliation**: Shanghai Artificial Intelligence Laboratory, The Chinese University of Hong Kong, Nanjing University, Cornell University
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---

## Abstract

![alt text](./images/Fig%201.png)
> **Figure 1.**  
> Scaffold-GS는 장면을 3D Gaussian set을 사용하여 dual-layered 계층 구조로 나타냄  
> 초기 point의 희박한 grid에 기반하여, 각 anchor에서 적은 수의 neural gaussian이 생성되어 다양한 시점과 거리에 동적으로 적응  
> 더 간결한 모델로 3D-GS에 필적하는 렌더링 품질과 속도 달성(마지막 행 척도: PSNR / 저장 용량 / FPS)  
> 투명성, 반사광, 반사, 무텍스처 영역 및 미세한 세부 사항 등 관찰이 어려운 view에서도 대규모 야외 장면과 복잡한 실내 환경에서 높은 견고성을 보임

**3D Gaussian Splatting**
- 종종 모든 학습 view에 맞추려는 과도하게 중복된 가우시안 생성
    - 기본 장면 기하 구조를 무시하는 경우가 많음
- 중요한 view 변화, texture가 없는 영역 및 조명 효과에 덜 견고해짐

**Scaffold-GS**
- anchor points를 사용하여 local 3D Gaussians를 분포시킴
- view frustum 내에서 시점 방향과 거리에 따라 속성을 실시간으로 예측
- neural Gaussian의 중요성이 기반한 anchor growing 및 가지치기 전략이 개발되어 장면 커버리지를 안정적으로 향상시킴
- 고품질 렌더링을 제공하면서 중복 가우시안을 효과적으로 줄임
- 렌더링 속도를 희생하지 않으면서 다양한 수준의 세부 사항과 view-dependent 관측을 갖는 장면을 처리하는 향상된 능력 입증

## 1. Introduction

**3D 장면 렌더링**
- mesh와 point 기반 전통적인 primitive-based 표현
    - 현대 GPU에 맞춘 rasterization 기법을 사용
    - 단점
        - 종종 불연속성과 흐릿한 아티팩트를 나타냄
        - 낮은 품질의 렌더링을 초래
- volumetric 표현과 neural radiance field
    - 학습 기반의 parametric model 활용
    - 많은 세부 정보가 보존된 연속적인 렌더링 결과 생성 가능
    - 단점
        - 확률적 샘플링이 시간 소모적임
            - 성능이 느려지고 잠재적인 노이즈 발생 가능

**3DGS**
- SOTA 렌더링 품질 및 속도 달성
- 일련의 3D 가우시안을 최적화
- 부피 표현에서 발견되는 고유한 연속성을 유지
- 3D 가우시안을 2D 이미지 평면에 splatting하여 빠른 rasterization을 용이하게 함
- 단점
    - 가우시안 ball을 과도하게 확장하는 경향이 있음
        - 장면 구조를 무시하게 됨
        - 상당한 중복 발생
        - 복잡한 대규모 장면의 경우, 확장성이 제한됨
    - view 의존 효과가 개별 가우시안 매개변수에 포함됨
        - 보간 능력이 거의 없음
        - view 변화와 조명 효과에 대한 강인성이 떨어짐

**Scaffold-GS**
- SfM point로부터 시작된 anchor point의 sparse grid 구성
- 각 anchor는 학습 가능한 offset을 갖는 neural gaussian set을 연결
    - 이들의 속성은 anchor feature과 시점에 따라 동적으로 예측됨
- 3D 가우시안 분포를 장면 구조를 사용하여 안내하고 제약하면서 지역적으로 시각 각도와 거리 변화에 적응할 수 있도록 함
    - 기존 3DGS는 3D 가우시안이 자유롭게 이동하고 분할될 수 있음
- scene 커버리지를 향상시키기 위해 anchor에 대한 growing 및 가지치기 연산 개발
- 원본 3DGS와 동등하거나 더 나은 렌더링 품질 제공
    - 추론 시, view frustum 내 anchor에 대해서만 neural gaussian 예측 제한
    - 필터링 단계(학습 가능한 selector)을 통해 불필요한 neural gaussian을 불투명도 기준으로 걸러냄
- 추가적인 계산 비용이 거의 없이 원본 3DGS와 유사한 속도(1K 해상도에서 약 100FPS)로 렌더링 가능
- 각 장면에 대해 anchor point와 MLP 예측기만 저장하면 되므로 저장 요구량이 크게 줄어듦

**논문의 기여**
1. 장면 구조를 활용하여, sparse voxel grid에서 anchor points를 초기화하여 local 3D Gaussian의 분포를 안내
    - 계층적이고 region-aware한 장면 표현 형성
2. view frustum 내에서, 다양한 시점과 거리를 수용하기 위해 각 anchor에서 neural gaussian을 즉시 예측
    - 보다 견고한 novel view 합성을 실현
3. 예측된 neural Gaussian을 활용하여 보다 나은 scene coverage를 위해 더 신뢰할 수 있는 anchor growing 및 가지치기 전략 개발

## 2. Related work

## 3. Methods

**원본 3DGS**
- 각 training view를 재구성하기 위해 가우시안 최적화
- heuristic 분할 및 가지치기 연산을 사용하지만 기본적인 장면 구조는 무시
    - 종종 중복된 가우시안 초래
    - novel view의 시점과 거리에 덜 견고하게 함

**제안 방법**
- 계층적 3D 가우시안 장면 표현 제안
- SfM에서 초기화된 anchor point로 지역 장면 정보 인코딩하고 지역 neural Gaussian 생성
- neural gaussian의 물리적 특성은 학습된 anchor features에서 view 의존적으로 실시간 디코딩

### 3.1 Preliminaries

### 3.2 Scaffold-GS

![alt text](./images/Fig%202.png)
> **Figure 2. Scaffold-GS 개요**  
>

#### 3.2.1 Anchor Point Initialization

- 초기 입력으로 COLMAP의 sparse point cloud 사용
    - point cloud $\rm{P} \in \mathbb{R}^{M \times 3}$에서 장면을 다음과 같이 voxel화
    $$
    \displaystyle
    \rm{V} = \bigg\{ \bigg\lfloor \frac{\rm{P}}{\epsilon} \bigg\rfloor \bigg\}
    \tag{4}
    $$
    > $\rm{V} \in \mathbb{R}^{N \times 3}$: voxel centers  
    > $\epsilon$: voxel size  
    - 이후 중복 항목 제거($\{ cdot \}$로 표시)
        - $\rm{P}$ 중복성과 불규칙성 줄이기 위함
- 각 voxel $v \in \rm{V}$의 중심은 anchor point($v$)로 간주됨
    - local context feature $f_v \in \mathbb{R}^{32}$
    - scaling factor $l_v \in \mathbb{r}^3$
    - $k$ 개의 learnable offsets $\rm{O}_v \in \mathbb{R}^{k \times 3}$
- 각 anchor $v$에 대해
    1. feature bank 생성 ${f_v, f_{v_{\downarrow_1}}, f_{v_{\downarrow_2}}}$  
    ($\downarrow_n$은 $f_v$가 $2^n$배로 downsampling됨을 나타냄)
    2. feature bank를 view-dependent weight와 혼합하여 통합된 anchor feature $\hat{f}_v$를 형성
        - 카메라가 $x_c$에 있고, anchor가 $x_v$에 있을 때, 다음과 같이 위치 간의 거리 및 방향 계산
        $$
        \displaystyle
        \delta_{vc} = ||\rm{x}_v - \rm{x}_c||_2, \quad \vec{\rm{d}}_{vc} = \frac{\rm{x}_v - \rm{x}_c}{||\rm{x}_v - \rm{x}_c||_2}
        \tag{5}
        $$
        - 작은 MLP $F_w$에서 예측된 가중치로 feature bank의 가중합 계산
        $$
        \displaystyle
        \begin{aligned}
        \{w, w_1, w_2\} = \rm{Softmax} (F_w(\delta_{vc}, \vec{\rm{d}}_{vc})),
        \tag{6}
        \end{aligned}
        $$
        $$
        \displaystyle
        \begin{aligned}
        \hat{f}_v = w \cdot f_v + w_1 \cdot f_{v_{\downarrow_1}} + w_2 \cdot f_{v_{\downarrow_2}}
        \tag{7}
        \end{aligned}

#### 3.2.2 Neural Gaussian Derivation

- $F_*$는 특정 MLP를 나타냄
- MLP 오버헤드를 줄이기 위한 두 가지 효율적인 사전 필터링 전략 소개
- neural Gaussian 매개변수
    - 위치 $\mu \in \mathbb{R}^3$
    - 불투명도 $\alpha \in \mathbb{R}$
    - covariance-related quaternion $q \in \mathbb{R}^4$
    - scaling $s \in \mathbb{R}^3$
    - color $c \in \mathbb{R}^3$
- view frustum 내의 각 visible anchor point에 대해 $k$개의 neural Gaussian을 생성하고 그 속성을 예측
    - $\rm{x}_v$에 위치한 anchor point가 주어지면 해당 neural Gaussian은 다음과 같이 계산됨
    $$
    \displaystyle
    \{\mu_0, \cdots, \mu_{k-1} \} = \rm{x}_v + \{ \mathcal{O}_0, \cdots, \mathcal{O}_{k-1} \} \cdot l_v
    \tag{8}
    $$
    > $\{ \mathcal{O}_0, \cdots, \mathcal{O}_{k-1} \} \in \mathbb{R}^{k \times 3}$: 학습 가능한 offset  
    > $l_v$: 해당 anchor와 관련된 scaling factor
    - $k$개의 neural gaussian 속성은 anchor feature $\hat{f}_v$ 카메라와 anchor point 사이의 