#  CAMERAS AS RAYS: POSE ESTIMATION VIA RAY DIFFUSION

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Relative Camera Pose Estimation

---
url:
- [paper](https://openreview.net/forum?id=EanCFCwAjM)
- [arXiv](https://arxiv.org/abs/2402.14817)
---
- **Authors**: Jason Y. Zhang, Amy Lin, Manolis Savva, Angel X. Chang, Deva Ramanan, Shubham Tulsiani
- **Venue**: ICLR 2024 (Oral)

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Method](#3-method)
  - [3.1 Representing Cameras with Rays](#31-representing-cameras-with-rays)
  - [3.2 Pose Estimation via Ray Regression](#32-pose-estimation-via-ray-regression)
  - [3.3 Pose Estimation via Denoising Ray Diffusion](#33-pose-estimation-via-denoising-ray-diffusion)
  - [3.4 Implementation Details](#34-implementation-details)

---

## ⚡ 요약 (Summary)
- **Problem**: 기존의 학습 기반 카메라 포즈 추정 방식은 전역적 포즈(회전 및 변환)를 직접 회귀하려 시도하지만, 이는 이미지 패치별 지역적 기하학 특징과의 결합력이 낮고 대칭적 객체에서 발생하는 다중 해(Multi-modality)와 불확실성을 표현하기 어려움.
- **Idea**: 카메라를 6차원 Plücker 좌표계로 표현된 광선(Ray)의 집합으로 분산 모델링하고, 트랜스포머 기반의 디퓨전(Diffusion) 모델을 활용하여 노이즈 섞인 초기 광선으로부터 정밀한 카메라 자세를 복원하는 'Ray Diffusion' 기법을 제안함.
- **Result**: 희소 시점(Sparse-view) 환경에서도 고수준의 기하학적 일관성을 확보하며 CO3D 데이터셋에서 기존 SOTA를 능가하는 성능을 달성했으며, 학습되지 않은 범주의 객체 및 실제 환경 촬영 데이터에서도 강력한 일반화 성능을 입증함.

---

## 📖 Paper Review

## Abstract

카메라 포즈 추정
- 3D 재구성을 위한 기본 작업
- 희소하게 샘플링된 view로 인해 도전적(<10개)
- 기존 방식은 카메라 extrinsics를 global parameterization하여 top-down 예측을 수행

카메라를 광선의 집합으로 취급하는 분산 표현을 제안
- 공간 이미지 feature과의 긴밀한 결합을 가능하게 하여 위치 정밀도를 향상시킴
- 이 표현이 set-level transformers에 적합함을 관찰
- 이미지 패치를 해당 광선에 매핑하는 regression 기반 접근 방식을 개발
- 성능을 향상시키면서 그럴듯한 mode를 샘플링할 수 있는 denoising diffusion을 조정
    - sparse-view 포즈 추론에서 내재된 불확실성을 포착하기 위해
- 회귀 및 확산 기반 방법 모두는 CO3D에서 카메라 위치 추정에 대한 SOTA 성능을 보임
- unseen 객체 및 in-the-wild 촬영에 일반화됨

### 1. Introduction

![alt text](./images/Fig%201.png)

> **Figure 1. Denoising Rays를 사용하여 Sparse-view 카메라 파라미터 복원**  
> 상단: 주어진 희소하게 샘플링된 이미지에서, 제안 방법은 camera ray(Plücker coordinates 사용)를 denoise하는 방법을 학습  
> 이후 카메라 intrinsic과 extrinsic을 광선을 사용하여 복원  
> 하단: seen (teddybear)과 unseen object 카테고리(couch, sandwich)에서의 일반화를 보임


희소한 입력 이미지만을 가지고 high-fidelity 3D representation을 얻기 위한 최근의 빠른 진행을 목격
- 2D 입력 이미지에 해당하는 카메라 포즈의 가용성이 중요함
- SfM 방법이 희소하게 샘플링된 뷰를 가진 설정에서 카메라 포즈를 신뢰성있게 추론할 수 없음

최근의 희소한 입력 이미지에서 카메라를 예측하는 학습 기반 접근 방식
- 회귀 기반 접근법
- energy-based 모델링
- denoising diffusion

이러한 방법들은 학습 기반 방법이 어떤 카메라 포즈 표현을 예측해야 하는지에 대해 회피함
- 이전 방법들은 카메라 extrinsic(단일 회전과 변환으로 매개변수화)을 사용. (회전은 matrix, quaternion, 각도 등 다양함)
- 간결한 global pose 표현이 신경망 학습에 최적이 아닐 수 있음
    - 신경망은 종종 over-parameterized 분산된 표현에서 이점을 얻음
- 기하학적 관점에서, 고전적인 bottom-up 방법은 pixel/patches 간의 저수준 대응에서 이점을 얻음
    - global 카메라 표현 예측은 이러한 연관성에서 이점을 얻지 못할 수 있음

포즈 추론 작업을 patch 별 광선 예측으로 변환하는 alternate camera parameterization을 제안
- 각 입력 이미지에 대해 전역 회전 및 이동을 예측하는 대신, 이미지의 각 패치를 통과하는 광선을 예측
    - 이 표현이 transformer 기반 set-to-set 추론 모델에 적합함을 보임
- 카메라 extrinsic과 intrinsic을 복원하기 위해, 예측된 ray bundle을 바탕으로 least square objective를 최적화
- 예측된 광선 다발 자체는 Grossberg & Nayar(2001)에서 소개된 일반 카메라의 인코딩으로 볼 수 있음
    - projection의 중심에서 교차하지 않을 수 있는 catadioptric imager 또는 직교 카메라와 같은 비원근 카메라를 포착할 수 있음

patch-based transformer을 표준 회귀 손실로 훈련시켜 분산된 광선 표현의 효과성을 설명
- 연산이 더 많이 소모되는 최신 포즈 예측 방법의 성능을 초월
- 대칭성과 부분 관측 때문에 예측된 광선에는 자연스러운 모호성이 존재함  
-> 회귀 기반 방법을 denoising diffusion-based 확률 모델로 확장
    - 성능을 향상시키고, 뚜련한 분포 mode를 복원할 수 있음을 발견
- CO3D 데이터셋에서 실험하고, seen 항목에 대한 성능 및 unseen 항목에 대한 일반화 연구
- 제안 방법이 unseen 데이터셋에서도 일반화 할 수 있으며, in-the-wild self-capture에 대한 정성적 결과 제시

**논문의 기여**
- 전역 카메라 매개변수 추론 대신 패치별 광선 방정식을 추론
- 희소하게 샘플링된 뷰를 고려하여 간단한 회귀 기반 접근법 제시. 최신 기술을 능가함
- 광선 기반 카메라 매개변수화를 통해 denoising diffusion model을 학습하여 성능 향상


## 2. Related Work

## 3. Method

목표: 희소 이미지 셋에서 카메라 포즈 추정

![alt text](./images/Fig%202.png)

> **Figure 2. Camera와 ray 표현 사이의 변환**  
> 카메라를 6-D Plücker ray의 집합으로 표현  
> 카메라의 전통적인 표현을 광선 다발 표현으로 변환하기 위해 카메라 중심에서 픽셀 좌표로 광선을 unprojectin  
> 카메라 중심, intrinsic, rotation 행렬에 대해 least-square 최적화를 해결하여 광선을 카메라 표현으로 변환  


![alt text](./images/Fig%203.png)

> **Figure 3. Denoising Ray Diffuser 네트워크**  
> 이미지 패치에 해당하는 노이즈가 있는 광선이 주어졌을 때, denoising ray diffusion 모델은 denoised ray를 예측  
> noisy ray와 함께 공간 이미지 feature을 결합. 이는 6차원 Plücker 좌표로 표현되어 3채널 방향 맵과 3채널 모멘트 맵으로 시각화됨  
> Transformer을 사용하여 모든 이미지 패치와 관련된 noisy rays를 공동으로 처리하여 original denoised rays를 예측

### 3.1 Representing Cameras with Rays

**Distributed Ray Representation**

- 일반적으로, 카메라는 외부 매개변수와 내부 매개변수 행렬로 매개변수화됨
- 신경망이 이 저차원 표현을 직접 회귀하는 것이 어려울 수 있다고 가설을 세움
- 카메라를 광선의 집합으로 over-parameterize하는 것을 제안

$$
\displaystyle
\begin{aligned}

& \mathcal{R} = {r_1, ..., r_m},
& (1)

\end{aligned}
$$

- 각 광선 $r_i \in \mathbb{R}^6$은 알려진 픽셀 좌표 $u_i$에 연관됨
- Plücker 좌표를 사용하여 모든 점 $p \in \mathbb{R}^3$을 통해 방향 $d \in \mathbb{R}^3$로 이동하는 각 광선 $r$을 매개변수화

$$
\displaystyle
\begin{aligned}

& r = <d, m> \in mathbb{R}^6,
& (2)

\end{aligned}
$$

> $m = p \times d \in \mathbb{R}^3$: moment 벡터  

- 모멘트 벡터를 계산하는데 사용되는 광선의 특정 point와는 무관
- $d$가 unit length일 때, moment $m$의 norm은 광선과 원점 사이의 거리

~

### 3.2 Pose Estimation via Ray Regression

$N$개 이미지 $\{ I_1, ..., I_N\}$이 주어졌을 때 광선 표현을 예측하는 방법
- 카메라 매개변수가 주어지면 ray bundles $\{ \mathcal{R}_1, ..., \mathcal{R}_N \}$을 예측할 수 있음
- 균일한 $p \times p$ grid에서 광선을 계산하여 각 광선 다발이 $m = p^2$ 광선(식(1))으로 구성되도록 함

광선과 이미지 patch 간의 상관관계를 보장하기 위해 spatial image feature extractor을 사용하고 각 patch feature을 토큰으로 간주:

$$
\displaystyle
\begin{aligned}

& f_{feat}(I) = f \in \mathcal{R}^{p \times p \times d}
& (6)

\end{aligned}
$$

crop parameter을 사용하기 위해 각 spatial feature에 대해 pixel coordiante $u$(uncropped image에서 정규화된 device 좌표)를 연결
- transformer-based 아키텍처를 사용하여 $N$개 이미지의 $p^2$ 토큰을 공동으로 처리하고 각 patch에 해당하는 광선을 예측:

$$
\displaystyle
\begin{aligned}

& \{ \hat{\mathcal{R} \} }_{i=1}^N = f_{Regress} \large( \{ f_i, u_i \}_{i=1}^{N \cdot p^2} \large)
& (7)

\end{aligned}
$$

loss:

$$
\displaystyle
\begin{aligned}

& \mathcal{L}_{recon} = \sum_{i=1}^N || \hat{\mathcal{R}}_i - \mathcal{R}_i ||_2^2
& (8)

\end{aligned}
$$



### 3.3 Pose Estimation via Denoising Ray Diffusion

patchwise regression-based 아키텍처는 분산된 ray-based parameterization을 효과적으로 예측
- 희박한 view에서 광선 형태의 포즈를 예측하는 작업은 모호할 수 있음
- 분산된 광선 표현 위에 diffusion-based probabilistic model을 학습하도록 이전 회귀 기반 방식 확장
    - 대칭성과 부분 관측으로 인한 예측의 내재적 불확실성을 처리하기 위함

![alt text](./images/Fig%204.png)

> **Figure 4. Ray Diffuser을 사용한 Denoising 과정 시각화**  
> 여행가방에 대한 2장의 이미지가 주어졌을 때, 무작위로 초기화된 카메라 광선에서 시작한 Denoising 과정 시각화  
> 아래: Plücker 표현을 사용한 시각화  
> 위: 해당하는 3D 위치  
> 좌측: 예측 카메라 포즈(녹색)과 GT(검정) 비교

~

### 3.4 Implementation Details

- 훈련 카메라의 optical 축에 가장 가까운 지점에 world origin을 둠
    - 중심을 향한 카메라 설정을 위한 유용한 유도 편향을 나타냄
- 좌표계의 모호성을 처리
    - 첫 번째 카메라가 항상 identity rotation을 갖도록 world 좌표를 회전시킴
    - 첫 번째 카메라의 변환이 unit norm을 갖도록 scene 크기 조정
- 객체 bounding box 주위에서 촘촘히 정사각형 이미지 crop을 수행. 이에 따라 광선과 관련된 pixel 좌표의 균일한 격자를 조정
- frozen DINOv2(S/14)(Oquab et al. 2023)을 image feature extractor로 사용
- $f_{Regress}$(t는 항상 100으로 설정됨)와 $f_{Diffusion}$ 두 가지 아키텍처에 대해 16개의 transformer block을 가진 DiT(Peebles & Xie, 2023)을 사용
- $T=100$ timestep으로 diffusion 모델 훈련
- denoiser을 훈련할 때, 광선의 방향과 모멘트 표현에 noise를 추가
- 광선 회귀 및 광선 확산 모델은 8개의 A6000 GPU에서 훈련하는데 각각 2일/4일 소요됨


ray denoiser로 카메라를 예측하기 위해, 약간의 수정이 가해진 DDPM inference를 사용
- 경험적으로 DDPM 추론에서 stochastic noise를 제거하고, backward diffusion 과정을 조기에 중단하며 (예측된 $x_0$을 추정으로 사용) 더 나은 성능을 보임
- earlier diffusion 과정에 개별적으로 그럴듯한 모드를 선택하는 데 도움이 되는 반면, later steps가 이러한 모드를 중심으로 샘플을 생성 - 이는 distribution modes를 선호하는 accuray 지표에 해가 될 수 있다고 추측

![alt text](./images/Fig%205.png)


![alt text](./images/Fig%206.png)