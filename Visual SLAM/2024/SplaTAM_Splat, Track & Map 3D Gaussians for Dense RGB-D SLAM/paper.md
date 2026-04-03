# SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- SLAM
- 3D Gaussian Splatting
---
url:
- [paper](https://arxiv.org/abs/2312.02126)
- [project](https://spla-tam.github.io/)
- [github](https://github.com/spla-tam/SplaTAM)
---
- **Authors**: Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, Jonathon Luiten
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Method](#3-method)

---

## ⚡ 요약 (Summary)


---

## 📖 Paper Review

## Abstract

![alt text](./images/Fig%201.png)
> **Figure 1.**

현재의 Dense SLAM 방법은 장면을 나타내는 non-volumetric 또는 암묵적인 방식 때문에 종종 방해를 받음

SplaTAM
- 명시적 volumetric 표현
    - 3D 가우시안을 활용하여 unposed RGB-D 카메라로부터 고충실도 재구성을 가능하게 함
- silhouette mask를 사용하여 장면 밀도를 우아하게 포착
    - 빠른 렌더링
    - 밀도 최적화
    - 해당 영역이 이전에 매핑되었는지 신속하게 판단
    - 더 많은 가우시안을 추가하여 구조화된 map 확장
- 기존 방법보다 카메라 pose 추정, map 구축, novel view 합성에서 최대 2배 우수한 성능을 보임


## 1. Introduction

**SLAM**
- vision sensor의 포즈 & 환경 map 추정
- 이전에 본 적 없는 3D 환경에서 작동하기 위해 vision/로봇 시스템에 필수적
- map 표현
    - SLAM 시스템 내의 모든 processing 블록 설계와 후속 작업에 영향을 미침

**dense visual SLAM**

- radiance-field 기반 SLAM
    - 미분 가능한 렌더링
        - dense photometric 정보를 포착하는 image reconstruction loss
        - 고충실도 전역 지도
    - implicit neural 표현을 사용하면 발생하는 문제
        - 계산 효율이 낮음
        - 편집이 쉽지 않음
        - 공간적 기하를 명시적으로 모델링하지 않음
        - catastrophic forgetting
    
- 3D 가우시안 기반 radiance field가 갖는 장점
    - **빠른 렌더링 & 풍부한 최적화**
        - 최대 400FPS 속도로 렌더링 가능
        - implicit한 다른 방법보다 시각화 및 최적화 속도가 빠름
        - 몇 가지 최적화 방법 소개
            - view 의존적 외관 제거
            - 등방성 가우시안 사용
        - 기존 방법의 map 표현과 달리, sparse 3D 기하학적 특징이나 pixel sampling에 의존하지 않음
        - 실시간으로 dense photometric loss 사용 가능
    - **명시적인 공간 범위를 갖는 지도**
        - 지도의 공간 경계는 과거 장면의 일부에 가우시안을 추가하여 쉽게 제어 가능
        - 실루엣 렌더링을 통해 장면의 어떤 부분이 새로운 콘텐츠인지 식별 가능
            - 카메라 추적에 중요한 역할
        - implicit 지도 표현에서는 새로운 공간에 대해 최적화 시 네트워크가 전역적으로 변경됨
    - **명시적 지도**
        - 더 많은 가우시안을 추가하여 지도의 용량을 늘릴 수 있음
        - 장면의 일부를 편집할 수 있으면서 사실적인 렌더링 가능
        - implicit 방법은 용량을 늘리거나 편집하기 어려움
    - **parameter로의 직접적인 gradient flow**
        - 장면이 물리적인 3D 위치, 색상, 크기를 갖는 가우시안으로 표현됨  
        -> 파라미터와 렌더링 사이에 직접적이고 거의 선형적인 gradient 흐름이 존재
        - 카메라 움직임 == '카메라를 고정하고 장면을 이동'으로 생각 가능  
        카메라 파라미터에도 gradient가 직접적으로 흐름
        - 신경망 기반 표현은 비선형 신경망 층을 통해 gradient가 흐르기에 이러한 직접적인 흐름이 없음

## 2. Related Work

## 3. Introduction

![alt text](./images/Fig%202.png)
> **Figure 2. SplaTAM 개요**
> 각 timestep에서 입력: 현재 RGB-D 프레임 & 구축한 3D gaussian map 표현
> 우상단: 단계 1에서는 silhouette guided 미분가능 렌더링을 사용하여 새 이미지의 카메라 포즈 추정
> 우하단: 단계 2에서는 렌더링된 실루엣과 입력 깊이를 기반으로 새로운 가우시안 추가
> 좌하단: 단계 3에서는 미분 가능 렌더링을 통해 gaussian map 업데이트

**SplaTAM**
- 3DGS를 사용하는 최초의 dense RGB-D SLAM
- 세계를 고충실도 색상 및 깊이 이미지로 렌더링할 수 있는 3D 가우시안의 집합으로 모델링
- 각 프레임의 카메라 포즈와 volumetric 이산 map 모두를 최적화 가능
- 미분 렌더링 & gradient 기반 최적화를 사용

**Gaussian Map Representation**
- 장면의 기본 지도를 3DGS로 표현
- view-independent 색상 사용
- 가우시안을 등방성으로 강제(아주 최근에는 이등방성을 사용하는 것으로 파악)
- 각 가우시안이 8개 값으로만 매개변수화됨
    - $\bm{c}$: RGB 색상
    - $\mu \in \mathbb{R}^3$: 중심 위치
    - $r$: 반지름
    - $o$: 불투명도
- 각 가우시안은 불투명도로 가중치를 두어 3D 공간의 점 $\rm{x} \in \mathbb{R}^3$에 영향을 미침
$$
\displaystyle
\begin{aligned}
f(x) = o \exp{\big( - \frac{||\rm{x} - \mu||^2}{2r^2} \big)}
\tag{1}
\end{aligned}
$$

**Differentiable Rendering via Splatting**
- 고충실도 색상, 깊이 및 실루엣 이미지를 모든 가능한 카메라 레퍼런스 프레임으로 렌더링 가능
- 렌더링 이미지와 제공된 RGB-D 프레임 간의 오차에 대해 기본 장면 표현(Gaussian)과 카메라 매개변수의 gradient를 직접 계산
- Gaussian과 카메라 매개변수 모두를 업데이트 가능  
-> 정확한 카메라 pose, 정확한 부피 표현을 모두 맞출 수 있음
- 렌더링 방법
    1. 3D Gaussian과 카메라 pose 집합이 주어지면 모든 gaussian을 앞에서부터 정렬
    2. 각 gaussian의 2D 투영을 픽셀 공간에서 순서대로 alpha-compositing해서 효율적으로 렌더링
    3. 픽셀 $\rm{p} = (u, v)$의 렌더링 색상:
    $$
    \displaystyle
    \begin{aligned}
    C(\rm{p}) = \Sigma_{i=1}^{n} \rm{c}_i f_i (\rm{p}) \prod_{j=1}^{i-1} (1 - f_j(\rm{p}))
    \tag{2}
    \end{aligned}
    $$
    > $f_i(p)$: 식 1에 의해 계산됨  
    > $\mu, r$: pixel-space에서 뿌려진 2D Gaussian의 값
    $$
    \displaystyle
    \begin{aligned}
    \mu^{2D} = K \frac{E_t \mu}{d} \quad r^{2D} = \frac{fr}{d}, \quad \rm{where} \quad d = (E_t \mu)_z
    \tag{3}
    \end{aligned}
    $$
    > $K$: (unknown) 카메라 intrinsic  
    > $E_t$: t frame에서 카메라의 extrinsic matrix  
    > $f$: (known) focal length  
    > $d$: $i^{th}$ 가우시안의 depth(카메라 좌표계에서)
- 렌더링 depth:
    - input depth map과 비교 가능
    - 3D map에 대한 기울기 반환
    $$
    \displaystyle
    D(p) = \Sigma_{i=1}^{n} d_i f_i(p) \prod_{j=1}^{i-1} (1 - f_j(p))
    \tag{4}
    $$
- 실루엣 이미지:
    - 픽셀이 현재 맵의 정보를 포함하고 있는지 여부 확인
    $$
    \displaystyle
    S(p) = \Sigma_{i=1}^{n} f_i(p) \prod_{j=1}^{i-1} (1 - f_j(p))
    \tag{5}
    $$

**SLAM System**

- 가정:
    - 3D Gaussian으로 구성된 맵이 있다고 가정
    - 카메라 프레임은 1부터 t까지
    - 새로운 RGB-D 프레임 t+1이 주어졌을 겅우를 고려

1. **Camera Tracking**
- 이미지 & depth 재구성 error 최소화
- 

2. **Gaussian Densification**
3. **Map Update**

**Initialization**

**Camera Tracking**

**Gaussian Densification**
- tracking 이후 정확하게 추정된 카메라 포즈를 얻음
- depth image가 있으면 가우시안이 있어야 할 위치에 대한 추정치를 알 수 있음
- 가우시안이 이미 존재하는 곳에는 추가하고 싶지 않음
- 어떤 픽셀을 조밀하게 해야 할지를 결정하기 위해 densification mask 생성
$$
\displaystyle
M(p) = \big( S(p) < 0.5 \big) + \big( D_{GT}(p) < D(p) \big) \big( L_1(D(p)) > \lambda \rm{MDE} \big)
\tag{9}
$$
- map이 적절하게 조밀하지 않은 곳을 보임
- 또는 현재 추정된 geometry의 앞에 새로운 geometry가 있어야 하는 곳을 나타냄
    - (예: 예측 depth 앞에 gt depth가 있고, depth error가 Median Depth Error(MDE)보다 $\lambda$ 배 큰 경우)
    - $\lambda$는 경험적으로 50으로 설정
- 마스크의 각 픽셀마다 새로운 gaussian 추가

**Gaussiaan Map Updating**

## 4. Experimental Setup

**Datasets and Evaluation Settings**

**Evaluation Metrics**
