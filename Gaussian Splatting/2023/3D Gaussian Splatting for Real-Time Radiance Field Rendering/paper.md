# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Gaussian Splatting
- Radiance Field

---
url:
- [paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) (SIGGRAPH 2023)
- [project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
---
- **Authors**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **Venue**: SIGGRAPH 2023

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [3. OVERVIEW](#3-overview)
- [4. Differentiable 3D Gaussian Splatting](#4-differentiable-3d-gaussian-splatting)
- [5. Optimization with Adaptive Density Control](#5-optimization-with-adaptive-density-control-of-3d-gaussians)
- [6. Fast Differentiable Rasterizer](#6-fast-differentiable-rasterizer-for-gaussians)

---

## ⚡ 요약 (Summary)
- **Problem**: 기존 Neural Radiance Fields(NeRF) 방식은 고품질의 새로운 뷰 합성이 가능하나, 렌더링 속도가 매우 느리고 훈련 비용이 커서 실시간 인터랙티브 환경에 적용하기 어려움.
- **Idea**: 장면을 불투명도와 색상을 가진 수백만 개의 3D 가우시안 타원체로 표현하고, 타일 기반 미분 가능 래스터라이저와 점진적인 밀도 제어(Adaptive Density Control)를 통해 최적화함.
- **Result**: 1080p 해상도에서 30FPS 이상의 실시간 렌더링을 달성하면서도 최신 NeRF 모델과 대등한 수준의 화질을 유지하며 훈련 시간을 대폭 단축함.


- SfM(Structure from Motion)으로 얻은 sparse point 집합 → 고품질 장면 표현 최적화
- Gaussian Splatting
    - Gaussian
        - 평균(\mu)를 중심으로 3D 공분산 행렬로 정의됨
        - 렌더링 시 2D로 투영해야 함(rasterizer 사용)
    - 최적화
        - 렌더링 → 훈련 데이터와 결과 이미지 비교하여 최적화 반복
            - optimization warm-up 이후 100iteration마다 밀도화 & 투명도 작은 가우시안 제거
        - SGD 적용
        - 위치, 공분산, 색상, 불투명도 최적화
    - 밀도화
        - 작은 가우시안 + under-reconstruction 영역
            - 가우시안을 복제
        - 큰 가우시안 + over-reconstruction 영역
            - 가우시안을 분할
        - 카메라 근처의 부유물로 인해 가우시안 밀도가 증가할 수 있음
            - 주기적으로 투명도를 0에 가깝게 초기화
            - 투명도가 임계값보다 작은 가우시안 제거

---

## 📖 Paper Review

## Abstract

Radience Field
- 높은 시각적 품질을 달성하기 위해 신경망을 훈련하고 렌더링을 해야 함
- 속도와 품질 사이의 절충 필요

경계가 없고 완전한 장면(고립된 객체가 아닌)과 1080p 해상도 렌더링을 다루는 현재의 어떤 방법도 실시간 디스플레이 속도를 달성할 수 없음

제안 방법
- 특징
    - 최첨단 시각적 품질 달성
    - 경쟁력 있는 훈련 시간 유지
    - 1080p 해상도에서 고품질 실시간(>= 30fps)
    - 새로운 시점 합성 가능
- 방법
    - 카메라 calibration 동안 생성된 sparse point에서 시작해서 3D 가우시안으로 장면 표현
    - 3D 가우시안의 교차 최적화/밀도 제어를 수행. 장면의 정확한 표현을 달성하기 위해 비구면 공분산 최적화
    - 비가시성 인식 렌더링 알고리즘 개발
        - 비구면 splatting 지원
        - 훈련 가속화
        - 실시간 렌더링을 가능하게 함


## 1. Introduction

![alt text](./images/Fig%201.png)

> **Fig 1.**  
> 3D 가우시안 장면 표현과 실시간 미분 가능한 렌더러를 사용하여 장면 최적화와 새로운 view 합성을 가속화
> 이전 SOTA 방법 Mip-NeRF360[Barron et al. 2022]와 동등한 품질로 radience field 렌더링  
> 가장 빠른 이전 방법들[Fridovich-Keil and Yu et al. 2022; Müller et al. 2022]과 경쟁할 수 있는 최적화 시간 필요  
> InstantNGP와 비슷한 훈련 시간을 가지며 유사한 품질 달성  
> 51분 훈련으로 SOTA 품질을 달성하며, Mip-NeRF360[Barron et al. 2022]보다 약간 더 나은 품질




## 3. OVERVIEW

## 4. Differentiable 3D Gaussian Splatting

목표: 희소한(SfM) 점 집합에서 시작하여 고품질의 새로운 뷰 합성이 가능한 장면 표현 최적화
- 미분 가능한 volume 표현의 속성을 가지면서 빠른 렌더링을 위해 구조화되지 않은 명시적인 형태 필요
- 3D Gaussian 선택
    - 미분 가능
    - 2D splat에 쉽게 투영될 수 있음
    - 렌더링을 위한 빠른 $\alpha$-blending을 허용

기존 방법[Kopanas et al. 2021; Yifan et al. 2019]
- 각 점이 법선이 있는 작은 평면 원이라고 가정
- SfM의 극단적인 희소성으로 법선 추정이 어려움

제안 방법
- normal이 필요 없는 3D 가우시안의 집합으로 기하학 모델링
- world 좌표계에서 점(평균) $\mu$를 중심으로 전체 3D 공분산 행렬 $\Sigma$로 정의

$$
\displaystyle
\begin{aligned}

& G(x) = e^{-\frac{1}{2} x^T \Sigma^{-1} x}
& (4)

\end{aligned}
$$

- 렌더링 시 3D 가우시안을 2D로 투영해야 함
- [Zwicker et al. 2001a] 방법 사용
    - 주어진 viewing transformation $W$에 대한 공분산 행렬 $\Sigma'$
    - $\Sigma'$의 세 번째 행과 열을 생략하면 평면 점과 normal에서 시작하는 것과 같은 구조와 특성을 가진 $2 \times 2$ 분산 행렬을 얻을 수 있음

$$
\displaystyle
\begin{aligned}

& \Sigma' = J W \Sigma W^T J^T
& (5)

\end{aligned}
$$

> $J$: 투영 변환의 affine 근사의 Jacobian



공분산 행렬 $\Sigma$를 최적화하여 radiance field를 나타내는 3D 가우시안 계산
- 공분산 행렬은 양의 준정(definite)일 때만 의미가 있음
- gradient descent로 최적화 시 유효하지 않은 공분산 행렬이 쉽게 발생 가능

대안
- 3D 가우시안의 공분산을 타원체 형태로 표현
- Scale 행렬 $S$와 회전 행렬 $R$을 사용:

$$
\displaystyle
\begin{aligned}

& \Sigma = R S S^T R^T
& (6)

\end{aligned}
$$

각 요소의 독립 최적화를 위해 별도로 저장
- 3D scale vector $s$ 저장
- 회전은 quaternion $q$로 저장(정규화하여 단위화)
- 이들은 각각의 행렬로 간단히 변화 가능

학습 시 효율성 개선
- 자동미분 시 큰 오버헤드를 방지하기 위해 모든 파라미터의 기울기를 명시적으로 유도

비등방성 공분산 표현
- 최적화를 위해 적합
- 캡처된 장면의 다양한 기하학에 맞게 3D 가우시안 최적화 가능
    - 압축된 표현 제공

## 5. Optimization with Adaptive Density Control of 3D Gaussians

### 5.1 Optimization

최적화
- 렌더링, 데이터셋의 training set과 결과 이미지 비교를 반복
- 3D에서 2D로의 투영의 모호성 때문에 geometry가 잘못 배치될 수 있음
    - geometry를 생성, 파괴, 이동할 수 있어야 함
- 3D 가우시안의 공분산 매개변수의 품질은 표현의 compact성에 중요
    - 큰 균일 영역은 적은 수의 큰 anisotropic(특정 방향으로 쏠려 있는) 가우시안으로 캡처될 수 있음
- SGD(Stochastic Gradient Descent) 적용
    - 일부 작업에 사용자 정의 CUDA 커널 추가 가능
- 빠른 resterization은 최적화 효율성에 중요
- 범위를 $[0 - 1)$ 로 제한하기 위해 불투명도(Opacity, $\alpha$)에 sigmoid activation 활성화 함수 사용
- 같은 이유로 scale에 exponential activation function을 적용
- 초기 공분산 행렬
    - 등방성 Gaussian으로 초기화
    - 축 길이: 가장 가까운 3개의 점까지의 거리 평균
- 위치에만 적용되는 Exponential decay scheduling 기술 사용
    - Plenoxels방식과 유사, 하지만 position에만 적용
- loss function:

$$
\displaystyle
\begin{aligned}

& \mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}
& (7)

\end{aligned}
$$

- $\lambda = 0.2$ 사용

### 5.2 Adaptive Control of Gaussians

- 초기 상태: SfM(Structure-from-Motion)으로 얻은 sparse point 집합에서 시작
- 이후 단위 부피당 가우시안 밀도와 개수를 적응적으로 조절
    - 초기 희소 집합 -> 더 조밀하고 장면을 잘 표현하는 집합으로 확장

- 순서
    1. optimization warm-up
    2. 100 iteration마다 밀도화 및 $\alpha$가 threshold $\epsilon_\alpha$보다 작은 가우시안(사실상 투명한) 제거

**밀도화**
- 빈 영역 채우기
- 다음 두 경우 모두 밀도화 필요:
    1. Under-reconstruction: 기하 정보가 부족한 영역
    2. Over-reconstruction: 한 개의 가우시안이 젋은 장면 영역을 덮고 있는 경우
- 공통적인 특징: view-space에서 positional 기울기가 큼
    - 직관적으로, 재구성이 부족한 지역은 최적화가 가우시안을 이동시키려고 함
- view-space position 기울기의 크기 평균이 임계값 $\tau_{pos} = 0.0002$ 이상인 경우 밀도화 수행
- 작은 가우시안 + under-reconstructed 영역
    - 목표: 새로 필요한 geometry를 커버
    - 방법: clone
        - 동일 크기의 복사본 생성
        - 위치는 기울기 방향으로 이동
    - 총 부피 및 가우시안 개수 동시 증가
- 큰 가우시안 + Over-reconstructed 영역
    - 목표: 지역을 세밀하게 표현
    - 방법: 분할
        - 큰 가우시안을 2개로 분리
        - scale은 계수 $\phi = 1.6$으로 나눔
        - 위치 초기화는 기존 3D 가우시안을 PDF(Probability Density Function, 확률 밀도 함수)로 사용해 샘플링
    - 총 부피 유지, 가우시안 개수 증가

가우시안 개수 폭증 방지
- 최적화가 입력 카메라 근처의 floaters(부유물)에 갇힐 수 있음(stuck)
    - 가우시안 밀도가 증가할 수 있음
- 해결책
    - 매 $N = 3000$ iteration 마다 $\alpha$를 거의 0으로 초기화
    - 이후 최적화 과정에서 필요한 gaussian은 $\alpha$가 다시 증가함
    - $\alpha < \epsilon_\alpha$인 가우시안은 제거
- 주기적 제거
    - 월드 공간에서 매우 큰 가우시안
    - viewspace에서 footprint가 큰 가우시안

모든 가우시안은 항상 euclidean 공간의 primitive로 유지
- 다른 방법처럼 공간 압축, warping, 원거리/대형 가우시안 투영 전략 불필요

6. Fast Differentiable Rasterizer for Gaussians



## Supplemental

![alt text](./images/sup_algorithm%201.png)