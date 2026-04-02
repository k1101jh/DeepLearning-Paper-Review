# Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras

---

## 📌 Metadata
---
분류
- Photo-SLAM
- 3DGS
- SLAM
- Photorealistic Mapping

---
url:
- [paper](https://arxiv.org/abs/2311.16728)
---
- **Authors**: Huajian Huang, Longwei Li, Hui Cheng, Sai-Kit Yeung
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Photo-SLAM](#3-photo-slam)
  - [3.1 Hyper Primitives Map](#31-hyper-primitives-map)
  - [3.2 Localization and Geometry Mapping](#32-localization-and-geometry-mapping)
  - [3.3 Photorealistic Mapping](#33-photorealistic-mapping)
  - [3.4 Gaussian-Pyramid-Based Learning](#34-gaussian-pyramid-based-learning)
  - [3.5 Loop Closure](#35-loop-closure)
- [4. Experiment](#4-experiment)

---

## ⚡ 요약 (Summary)
- **Problem**: 기존의 암시적 표현(Implicit Representation) 기반 SLAM은 MLP 디코딩 연산량이 많아 임베디드 장치에서 실시간 구동이 어렵고, 기하학적 정확도와 사진급 화질을 동시에 만족하기 힘듦.
- **Idea**: 명시적인 특징점(ORB)과 암시적인 텍스처(3D Gaussian)를 결합한 'Hyper Primitive Map' 구조를 제안하고, 가우시안 피라미드 기반 학습 및 기하학 기반 고밀도화(Geometry-based Densification) 전략을 도입함.
- **Result**: 단안, 스테레오, RGB-D 카메라에서 실시간 추적과 함께 SOTA 대비 30% 높은 PSNR의 고화질 지도를 생성하며, Jetson Orin 등 임베디드 플랫폼에서도 실시간 성능을 입증함.


**Photo-SLAM**
- hyper primitive map(고차원 표현 방식) 기반 설계
   - 인접한 프레임 사이에서 충분한 2D-to-2D 대응관계를 기반으로 변환 행렬을 성공적으로 추정하면(에센셜 행렬을 구하는 것으로 추측), 삼각 측량을 통해 hyper primitive map이 초기화되고 자세 추정이 시작됨
   - 추적하는 동안
      - Localization: 2D-to-3D 관계를 활용하여 현재 카메라 자세를 계산. PnP가 아니라 Motion-only bundle adjustment 방법 사용. Levenberg-Marquardt(LM) 알고리즘으로 최적화(재투영 오차를 줄임)
      - Geometric mapping: 점진적으로 sparse hyper primitive를 생성하고 초기화. local bundle adjustment 수행
      - Photorealistic: hyper primitive를 점진적으로 최적화하고 밀도를 높임

- Geometry-based Densification
   - 실험적으로, RGB-D가 아닌 상황에서 frame의 2D feature point 중 30% 미만이 활성화되어 있고 3D 포인트와 매칭됨(Fig 4 참조)
      - frame에 공간적으로 분포된 2D 특징점들이 사실상 복잡한 텍스처를 가진 영역을 나타냄
      - 이러한 영역에는 더 많은 hyper primitives가 필요함
   - photorealistic한 매핑을 위해 keyframe이 생성되는 시점에 비활성화된 2D 특징점을 기반으로 임시 hyper primitives를 추가로 생성
      - RGB-D 카메라를 사용하는 경우, 비활성화된 2D 특징점(매칭이 안 된 특징점)에 깊이 정보를 직접 투영하여 임시 hyper primitives를 생성 불가
      - monocular 카메라의 경우, 비활성화된 특징점의 깊이를 인접한 활성화된 2D 특징점들의 깊이를 해석하여 추정
      - stereo 카메라의 경우, stereo-matching 알고리즘을 통해 비활성화된 특징점의 깊이를 추정

- Gaussian-Pyramid-Based Learning
   - 다양한 장점을 최대한 활용하기 위해 제안한 새로운 점진적 학습 방식
   - 가우시안 피라미드: 이미지의 다양한 세부 수준을 포함한 다중 스케일 표현 방식(Fig 5)
   - 원본 이미지에 가우시안 블러링과 다운샘플링 작업을 반복적으로 적용
      - 초기 학습 단계에서는 하이퍼 프리미티브들이 피라미드의 가장 높은 수준, 즉 레벨 n에 의해 감독됨
      - 학습 반복이 늘어남에 따라 하이퍼 프리미티브의 밀도를 높이는 동시에(Sec. 3.3.1 참조) 피라미드의 수준을 줄이고 새로운 정답(ground truth)을 얻음
      - 가우시안 피라미드의 최하단에 도달할 때까지 이어짐

**의문점**
1. 새로운 프레임이 들어오면 Gaussian을 학습하는데, 고해상도로 학습중일 때 새로운 프레임이 들어와서 동일한 가우시안을 저해상도로 학습시키면 문제가 발생하는게 아닌가?  
-> 관측 가능한 gaussian만을 최적화? 저해상도에서 사용되는 가우시안과 고해상도에서 사용되는 가우시안이 다른가?

**문제점 & 한계점**

- Tracking 병목 존재(Rendering 성능이 크게 향상되어서 상대적으로 Tracking 성능이 낮아짐)
    - 딥러닝 기반 pose estimation 방법 적용 고려


---

## 📖 Paper Review

## Abstract

neural rendering 및 SLAM 시스템의 통합은 최근 joint localization 및 photorealistic view 재구성에서 유망한 결과를 보임
- 하지만 기존 방법은 implicit representations에 완전히 의존. 자원을 많이 소모하여 휴대용 장치에서 실행할 수 없음

**Photo-SLAM**
- hyper primitive map을 가진 새로운 SLAM 프레임워크
- 위치 추정을 위해 명시적 기하학적 특징을 동시에 활용
- 관찰된 환경의 texture 정보를 표현하기 위해 implicit photometric features를 학습
- 기하학적 특징에 기반하여 hyper primitives map을 적극적으로 밀집화
- multi-level features를 점진적으로 학습하여 photorealistic mapping 성능을 향상시키기 위해 Gaussian-Pyramid 기반의 훈련 방법을 추가로 도입
- monocular, stereo, RGB-D 데이터셋을 사용한 광범위한 실험 결과는 Photo-SLAM이 online photorealistic mapping에서 현재 SOTA SLAM 시스템을 현저히 능가함을 입증
   - PSNR이 30% 더 높음
   - Replica dataset에서 렌더링 속도가 수백 배 더 빠름
   - Jetson AGX Orin과 같은 임베디드 플랫폼에서 실시간 속도로 실행 가능
   - 로봇 응용의 가능성을 보임

## 1. Introduction

카메라를 이용한 SLAM의 목표: 자율 시스템이 환경을 탐색하고 이해할 수 있게 함

전통적인 SLAM
- 기하학적 mapping에 초점을 맞춤
- 정확하지만 시각적으로 단순한 feature 제공

neural rendering의 발전은 SLAM 파이프라인에 photorealistic view 재구성을 통합할 수 있는 가능성을 보임  
-> 로봇 시스템의 인식 능력 향상
- 둘의 결합을 통해 얻은 유망한 결과에도 불구하고, 기존 방법들은 단순하고 심하게 암묵적인(implicit) 표현에 의존  
-> 계산 집약적. 자원이 제한된 장치에 배포하기 적합하지 않음
   - 예: Nice-SLAM은 환경을 나타내는 학습 가능한 feature을 저장하기 위해 hierarchical grid를 활용하는 반면, ESLAM은 multi-scale 압축 텐서 components를 이용.
   - 이후 카메라 포즈를 공동으로 추정하고 ray sampling 배치의 reconstruction loss를 최소화하여 feature을 최적화
   - 이러한 최적화 과정은 시간이 많이 소요됨.
- RGB-D 카메라, dense optical flow estimator 또는 monocular depth estimator과 같은 다양한 소스에서 얻은 해당 깊이 정보를 통합하여 효율적인 수렴을 보장하는 것이 필수적
- 암묵적 feature은 MLP에 의해 디코딩되기 때문에 최적의 성능을 위해 ray sampling을 정규화하기 위해 경계(bounding area)를 신증하게 정의하는 과정이 일반적으로 필요함[14]  
-> 이는 시스템의 확장성을 제한
- 이러한 한계는 휴대 가능한 플랫폼을 사용하여 알려지지 않은 환경에서 실시간 탐색 및 매핑 기능을 제공할 수 없음을 의미함

**Photo-SLAM**

![alt text](./images/Fig%201.png)

> **Figure 1. Rendering 및 궤적 결과**  
> Photo-SLAM은 단안, 스테레오, RGB-D 카메라를 사용하여 장면의 high-fidelity 뷰를 재구성 할 수 있음  
> 렌더링 속도는 최대 1000FPS

![alt text](./images/Fig%202.png)

> **Figure 2.**  
> Photo-SLAM은 localization, 명시적 geometry mapping, 암시적 photorealistic mapping, loop closure 구성 요소로 이루어짐  
> hyper primitives로 지도를 유지

- 기존 방법의 확장성 및 계산 자원 제약을 해결하면서 정밀한 위치 추적 및 온라인 photorealistic mapping을 달성하는 프레임워크
- ORB features, rotation, scaling, 밀도, spherical harmonic(SH) coefficients를 저장하는 point cloud로 구성된 hyper primitive map을 유지
- hyper primitive map
   - system이 factor graph solver을 사용하여 효율적으로 추적을 최적화
   - 원본 이미지와 렌더링 이미지 간의 loss를 역전파하여 해당 매핑을 학습할 수 있도록 함
   - 이미지는 ray sampling 대신 3D Gaussian splatting을 사용하여 렌더링됨
- 3D Gaussian splatting renderer
   - view 재구성 비용이 줄어들 수 있지만, online incremental mapping을 위한 high-fidelity rendering 생성을 가능하게 하지는 않음. monocular 시나리오에서 특히
- dense depth 정보에 의존하지 않고 고품질 mapping을 달성하기 위해 geometry-based densification 전략과 Gaussian-Pyramid-based(GP) 학습 방법을 추가로 제안
   - GP 학습은 multi-level feature의 점진적 습득을 촉진하여 시스템의 매핑 성능을 효과적으로 향상시킴

제안된 접근법의 효능을 평가하기 위해, monocular, stereo, RGB-D 카메라로 캡처한 다양한 데이터셋을 사용하여 광범위한 실험을 수행
- Photo SLAM이 localization 효율성, photorealistic mapping 품질 및 렌더링 속도에서 SOTA를 달성함을 보임
- embedded device에서 Photo-SLAM 시스템의 실시간 실행은 실용적인 로봇 응용 프로그램의 잠재력을 보임

**논문의 기여**
- hyper primitives map에 기반한 최초의 simultaneous localization and photorealistic mapping 시스템 개발. 실내외 환경에서 monocular, stereo, RGB-D 카메라를 지원
- 모델의 high-fidelity mapping을 실현할 수 있도록 multi-level feature을 효율적이고 효과적으로 학습할 수 있는 Gaussian-Pyrarmid-based learning을 제안
- C++ 및 CUDA로 완전히 구현된 시스템은 SOTA를 달성하며 embedded platform에서 실시간 속도로 실행 가능

## 2. Related Work

Visual localization and mapping은 카메라를 통해 알려지지 않은 환경의 적절한 표현을 구축하면서 해당 환경 내의 자세를 추정하는 문제

SfM과 달리, Visual SLAM은 일반적으로 정확성과 실시간 성능 간의 더 나은 균형을 추구함

**Graph Solver vs Neural Solver**

- 고전적인 SLAM 방법은 변수(pose와 landmark)와 측정값(관측 및 제약 조건) 간의 복잡한 최적화 문제를 모델링하기 위해 factor graph를 널리 채택
- SLAM은 실시간 성능을 달성하기 위해 비용이 많이 드는 연산을 피하면서 자세 추정을 점진적으로 전파
   - 예: ORB-SLAM 시리즈[2, 23, 24]는 연속 프레임에서 가벼운 기하학적 특징을 추출 및 추적하는데 의존. 전역적으로가 아니라 지역적으로 bundle adjustment를 수행
   - LSD-SLAM[7] 및 DSO[8]과 같은 직접 SLAM은 기하학적 feature 추출 비용 없이 raw image intensities에서 작동
   - 이들은 제한된 자원 시스템에서도 point cloud로 표현된 sparse / semi dense map을 온라인으로 유지
- 딥러닝을 통해 SLAM에 학습 가능한 매개변수와 모델이 도입되어 파이프라인이 미분 가능해짐
   - DeepTAM[45]와 같은 일부 방법은 신경망을 통해 카메라 자세를 end-to-end로 예측하지만, 정확도가 제한적임
   - 성능을 향상시키기 위해, D3VO[41] 및 Droid-SLAM[34]와 같은 일부 방법은 SLAM 파이프라인에 monocular 깊이 추정[12] 또는 dense optical flow estimation[33] 모델을 감독 신호로 도입  
   -> 장면 기하를 명시적으로 나타내는 depth map을 생성할 수 있음
   - 훈련에 사용할 수 있는 대규모 합성 SLAM 데이터셋인 TartanAir[37] 덕분에 RAFT[33]에 기반한 Droid-SLAM은 SOTA를 달성.  
   -> 순수 신경망 기반 solver는 계산 비용이 많이 들고 unseen scenes에서 성능이 크게 저하될 수 있음

**Explicit Represenatation vs Implicit Representation**
- dense reconstruction을 얻기 위해 KinectFusion[15], BundleFusion[6], InfiniTAM[25]를 포함한 일부 방법은 암묵적 표현이 Truncated Signed Distance Function(TSDF)[5]를 사용하여 들어오는 RGB-D 이미지를 통합하고 연속적인 표면을 재구성  
   - GPU에서 실시간으로 실행될 수 있음
   - dense reconstruction을 얻을 수 있지만, view rendering 품질은 제한적임
- NNeRF[21] 등 neural rendering 기술은 놀라운 novel view 합성을 달성
   - 카메라 포즈가 주어지면 MLP를 통해 장면 기하학과 색상을 암묵적으로 모델링
   - MLP는 렌더링 이미지와 훈련 view의 손실을 최소화하여 최적화
   - iMAP[30]은 이후 NeRF를 incremental mapping에 적응시켜 MLP와 카메라 포즈 모두를 최적화
   - Nice-SLAM[46]은 깊은 MLP 쿼리 비용을 줄이기 위해 multi-resolution grids를 도입
- Co-SLAM[36]과 ESLAM[16]은 각각 InstantNGP[22]와 TensoRF[3]을 탐색하여 매핑 속도를 가속화
   - 카메라 포즈와 기하학적 표현의 암묵적 공동 최적화는 여전히 불안정함
   - 불가피하게, RGB-D 카메라의 명시적 깊이 정보나 radiance field의 빠른 수렴을 위한 추가 모델 예측에 의존

Photo-SLAM
- dense mesh를 재구성하기 보다는 몰입형 탐색을 위한 관찰된 환경의 간결한 표현을 회복하는 것을 목표로 함  
-> 명확한 기하학적 특징점을 활용하여 정확하고 효율적인 위치 추정을 수행하는 online hyper primitive로 지도를 유지
- 텍스쳐 정보를 포착하고 모델링하기 위해 암묵적 표현을 활용
- 밀집 깊이 정보에 의존하지 않고도 고품질 매핑을 달성하므로 RGB-D 카메라 뿐만 아니라 monocular 및 stereo 카메라도 지원할 수 있음


## 3. Photo-SLAM

### 3.1 Hyper Primitives Map

Hyper primitives: SLAM에서 사용되는 고차원 표현 방식
> Point clouds: $P \in \mathbb{R}^3$  
> ORB features[26]: $O \in \mathbb{R}^{256}$  
> Rotation: $\text{P} \in \mathbb{R}^3$  
> Scaling: $\text{s} \in \mathbb{R}^3$  
> Density: $\sigma \in \mathbb{R}^1$  
> Spherical harmonic 계수: $\text{SH} \in \mathbb{R}^{16}$

- ORB features은 2D-to-2D 및 2D-to-3D 대응관계를 설정하는 역할을 함
- 인접한 프레임 사이에서 충분한 2D-to-2D 대응관계를 기반으로 변환 행렬을 성공적으로 추정하면(에센셜 행렬을 구하는 것으로 추측), 삼각 측량을 통해 hyper primitive map이 초기화되고 자세 추정이 시작됨
- 추적하는 동안
   - localization component는 들어오는 이미지를 처리하고, 2D-to-3D 관계를 활용하여 현재 카메라 자세를 계산
   - 기하학 mapping componnet는 점진적으로 sparse hyper primitive를 생성하고 초기화
   - photorealistic component는 hyper primitives를 점진적으로 최적화하고 밀도를 높임

### 3.2 Localization and Geometry Mapping

- localization 및 기하학 mapping components는 다음 요소를 제공
   - 입력 이미지의 효율적인 6-DoF 카메라 포즈 추정
   - sparese 3D points
- 최적화 문제는 Levenberg-Marquardt(LM) 알고리즘으로 해결되는 factor graph로 공식화됨

**localization thread**
- motion-only bundle 조정을 사용
   - 카메라 방향 $\text{R} \in SO(3)$와 위치 $t \in \mathbb{R}^3$을 최적화하여 키포인트 $p_i$와 3D 점 $P^i$ 간의 재투영 오차를 최소화
- $i \in \mathcal{X}$를 매칭 집합 $\mathcal{X}$의 인덱스라고 하면, LM으로 다음을 최적화 하려 함

$$
\displaystyle
\begin{aligned}

& {R, t} = \text{argmin}_{R, t} \sum_{i \in \mathcal{X}} \rho ( || \text{p}_i - \pi (\text{RP}_i + \text{t}) ||_{\sum_g}^2 )
& (1)

\end{aligned}
$$

> $\sum_g$: keypoint의 scale-associated 공분산 행렬  
> $\pi (\cdot)$: 3D에서 2D로의 투영 함수  
> $\rho$: 강건한 huber 비용 함수

**geometry mapping thread**
- 양쪽에서 볼 수 있는 점들 $\mathcal{P}_L$과 keyframes $\mathcal{K}_L$ 집합에 대해 local bundle adjustment 수행
- keyframe은 입력 카메라 시퀀스에서 선택된 프레임.
   - 좋은 시각 정보를 제공
- 각 keyframe이 노드인 factor graph를 구성
   - edge는 keyframe과 matched 3D point 간의 제약을 나타냄
- error function의 1차 미분을 사용하여 keyframe pose와 3D point를 정제함으로써 reprojection 잔차를 반복적으로 최소화
- $\mathcal{K}_L$에는 없지만 $\mathcal{P}_L$을 관찰하는 keyframe $\mathcal{K}_F$의 포즈를 고정
- $\mathcal{K} = \mathcal{K}_L \cup \mathcal{K}_F$로 두고, $\mathcal{X}_k$를 keyframe k의 2D keypoint와 $\mathcal{P}_L$의 3D 점 간의 매칭 집합으로 둔다.
- 최적화 과정의 목표: $\mathcal{K}$와 $\mathcal{P}_L$간의 기하학적 불일치를 줄이기
- 최적화 과정:

$$
\displaystyle
\begin{aligned}

& {\text{P}_i, \text{R}_l, \text{t}_l | i \in \mathcal{P}_L, l \in \mathcal{K}_L} = \text{argmin}_{\text{P}_i, \text{R}_l, \text{t}_l} \sum_{k \in \mathcal{K}} \sum_{j \in \mathcal{X}_k} \rho (E(k, j)),

& (2)

\end{aligned}
$$

reprojection residual:

$$
\displaystyle
\begin{aligned}

& E(k, j) = || \text{p}_j - \pi (\text{R}_k \text{P}_j + \text{t}_k) ||_{\sum_g}^2

\end{aligned}
$$

### 3.3 Photorealistic Mapping

**photorealistic mapping thread**
- geometry mapping thread에 의해 생성된 hyper primitives를 최적화
- hyper primitives는 tile-based renderer에 의해 rasterized되어 keyframe pose와 일치하는 이미지를 합성할 수 있음
- 렌더링 프로세스: 

$$
\displaystyle
\begin{aligned}

& C(\text{R, t}) = \sum_{i \in N} \text{c}_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_i)
& (3)

\end{aligned}
$$

> $N:$ hyper primitives 개수  
> $\text{c}_i$: $\text{SH} \in \mathbb{R}^{16}$으로부터 변환된 색상  
> $\alpha$: $\sigma \cdot \mathcal{G}(\text{R}, \text{t}, \text{P}_i, \text{r}_i, \text{s}_i)$ 와 동일. $\mathcal{G}$는 3D gaussian splatting 알고리즘

position $\text{P}$, rotation $\text{r}$, scaling $\text{s}$, 밀도 $\sigma$와 spherical harmonic 계수 $\text{SH}$에 대한 최적화는 렌더링 이미지 $I_r$과 실제 이미지 $I_{gt}$ 간의 photometric loss $\mathcal{L}$을 최소화하여 수행됨

$$
\displaystyle
\begin{aligned}

& \mathcal{L} = (1 - \lambda) | I_r - I_{gt} |_1 + \lambda (1 - \text{SSIM}(I_r, I_{gt}))
& (4)

\end{aligned}
$$

> $\text{SSIM} (I_r, I_{gt})$: 두 이미지 간의 구조적 유사성  
> $\lambda$: 균형을 위한 가중치 계수

### 3.3.1 Geometry-based Densification

![alt text](./images/Fig%204.png)

> **Figure 4. 초기 기하학적 정보를 사용하여 hyper primitives를 밀집화**

photorealistic mapping을 scene의 회귀 모델로 고려할 때, 더 많은 매개변수를 가진 더 밀집된 hyper primitives가 일반적으로 더 높은 렌더링 품질을 위한 장면의 복잡성을 더 잘 모델링 할 수 있음
- 실시간 매핑을 위해 기하학 매핑 요소는 sparse hyper primitives만 수립
- 기하학 매핑에 의해 생성된 조잡한 hyper primitives는 photorealistic mapping의 최적화 과정에서 밀집화되어야 함
- [18]과 유사하게 큰 loss gradients를 가진 hyper primitives를 분할하거나 복제하는 것 되에도 추가적인 기하학 기반 밀집화 전략을 도입

실험적으로, RGB-D가 아닌 상황에서 frame의 2D 기하학적 특징점(feature point) 중 30% 미만이 활성화되어 있고 3D 포인트와 매칭됨(Fig 4 참조)
- frame에 공간적으로 분포된 2D 특징점들이 사실상 복잡한 텍스처를 가진 영역을 나타냄
- 이러한 영역에는 더 많은 hyper primitives가 필요함

photorealistic한 매핑을 위해 keyframe이 생성되는 시점에 비활성화된 2D 특징점을 기반으로 임시 hyper primitives를 추가로 생성
- RGB-D 카메라를 사용하는 경우, 비활성화된 2D 특징점에 깊이 정보를 직접 투영하여 임시 hyper primitives를 생성 불가
- monocular 카메라의 경우, 비활성화된 특징점의 깊이를 인접한 활성화된 2D 특징점들의 깊이를 해석하여 추정
- stereo 카메라의 경우, stereo-matching 알고리즘을 통해 비활성화된 특징점의 깊이를 추정


### 3.4 Gaussian-Pyramid-Based Learning

![alt text](./images/Fig%203.png)

> **Figure 3. 다양한 progressive training 방법 비교**  
> 인코더 $\mathcal{E}_n$은 feature $\mathcal{F}_n$을 회귀하는 구조를 나타냄  
> 이는 MLP, voxel grid, hash table, positional encoding 등일 수 있음  
> 디코더 $\mathcal{D}_n$은 $\mathcal{F}_n$을 밀도, 색상, 또는 기타 정보로 변환하는 구조를 나타냄  
> multi-level features를 효율적으로 학습화기 위해 Gaussian pyramid를 기반으로 한 새로운 방법 제안

**Progressive Training**
- neural rendering에서 최적화 속도를 높이기 위해 사용되는 기술
- 더 나은 렌더링 품질을 유지하면서 학습 시간을 줄이기 위한 방법들이 제안됨
   - 기본 방법: 모델의 해상도와 파라미터 수를 점진적으로 증가시키기
      - 예시:
         - NSVF[20]와 DVGO[31]는 학습 중 feature grid 해상도를 점차 높이는 방식. 기존 방식보다 학습 효율이 크게 향상됨. 초기 단계의 저해상도 모델은 고해상도 모델을 초기화하는데 사용되며, 최종 추론 시에는 사용되지 않음(Fig 3a)
         - NGLoD[32]는 다중 해상도의 특징을 향상시키기 위해 여러 개의 MLP를 인코더와 디코더로 점진적으로 학습시키며, 마지막 디코더만 유지하여 통합된 다중 해상도 특징을 디코딩(Fig 3b)
         - Neuralangelo[19]는 학습 중 하나의 MLP만 유지하며, 점진적으로 서로 다른 수준의 해시 테이블을 활성화함으로써 대규모 장면 재구성에서 우수한 성능을 달성(Fig 3c)
         - 3D Gaussian Splatting[18]도 이와 유사하게 3D 가우시안을 점진적으로 고밀도화하여 radiance field 렌더링에서 최고 수준 성능 발휘
      - 이러한 방법들은 같은 학습 이미지를 활용하여 서로 다른 수준의 모델을 supervised training함
   - 대안 방식
      - BungeeNeRF[39]에서 사용된 네 번째 방법(Fig 3d)는 서로 다른 해상도의 이미지에 대해 서로 다른 모델을 적용
         - 다중 해상도 학습 이미지들을 명시적으로 그룹화하여, 각 모델이 다양한 수준의 특징을 학습할 수 있도록 설계됨
         - 이러한 방법은 모든 상황에 적용 가능한 일반적인 방식은 아니며, 다중 해상도 이미지가 제공되지 않는 대부분의 경우에는 사용이 불가

![alt text](./images/Fig%205.png)

> **Figure 5. Gaussian pyramid 기반 Training process**

**가우시안 피라미드 기반(GP) 학습(Fig. 3e)**
- 다양한 장점을 최대한 활용하기 위해 제안한 새로운 점진적 학습 방식
- 가우시안 피라미드는 이미지의 다양한 세부 수준을 포함한 다중 스케일 표현 방식(Fig 5)
- 원본 이미지에 가우시안 블러링과 다운샘플링 작업을 반복적으로 적용
   - 초기 학습 단계에서는 하이퍼 프리미티브들이 피라미드의 가장 높은 수준, 즉 레벨 n에 의해 감독됨
   - 학습 반복이 늘어남에 따라 하이퍼 프리미티브의 밀도를 높이는 동시에(Sec. 3.3.1 참조) 피라미드의 수준을 줄이고 새로운 정답(ground truth)을 얻음
   - 가우시안 피라미드의 최하단에 도달할 때까지 이어짐

가우시안 피라미드의 n+1 수준을 활용한 최적화 과정:

$$
\displaystyle
\begin{aligned}

& t_0: \text{argmin} \mathcal{L}(I_r^n, \text{GP}^n(I_{gt})), \\
& t_1: \text{argmin} \mathcal{L}(I_r^{n-1}, \text{GP}^{n-1}(I_{gt})), \\
& \dots
& (5) \\
& t_n: \text{argmin} \mathcal{L}(I_r^0, \text{GP}^0(I_{gt})),

\end{aligned}
$$

> $I_r, \text{GP}(I_{gt})$: 식 4.  
> $\text{GP}^n(I_{gt})$: Gaussian pyramid에서 level n에 있는 ground image

- GP 학습이 photorealistic mapping 성능을 크게 향상시킴(특히 단안 카메라에서)

### 3.5 Loop Closure

Loop Closure [11]
- 누적된 오류와 드리프트 문제를 해결하는 데 핵심적인 역할
- 이러한 문제는 위치 추정 및 기하학적 지도 작성 과정에서 발생할 수 있음
- 루프가 닫혔다고 판단되면, 유사 변환(Similarity Transformation)을 통해 local keyframes과 hyper primitives를 보정할 수 있음
- 보정된 카메라 포즈를 활용하면
   - 광학 매핑(photorealistic mapping) 구성 요소가 시각적 유령 현상(ghosting)을 제거하고
   - 주행 기반 위치 추정 오류(odometry drift)로 인한 왜곡을 줄임
   - 전체 매핑 품질을 향상시킬 수 있음


## 4. Experiment

Photo-SLAM과 SOTA SLAM 및 real-time 3D 재구성 시스템과 비교
- 단안, 스테레오, RGB-D 카메라 및 실내 및 야외 환경을 포함한 다양한 시나리오를 포함
- Photo-SLAM의 성능을 다양한 하드웨어 구성에서 평가하여 효율성을 입증
- 제안된 알고리즘의 효과성을 검증하기 위한 ablation study를 수행

### 4.1 Implementation and Experiment Setup

- ORB-SLAM3 [2], 3D 가우시안 스플래팅 [18], LibTorch 프레임워크를 활용하여 C++ 및 CUDA로 Photo-SLAM을 구현
- Photorealistic mapping의 최적화: 확률적 경사 하강법 알고리즘 사용. 고정 learning rate 및 λ = 0.2를 사용
- 테스트 데이터셋의 이미지 해상도를 고려하여 gaussian pyramid의 level은 3으로 설정(기본값: n = 2)
- 비교 기준:
   - ORB-SLAM3 [2]: 전통 SLAM 시스템
   - BundleFusion [6]: 실시간 RGB-D 밀집 재구성 시스템
   - DROID-SLAM [34]: 딥러닝 기반 시스템
   - 최근의 뷰 합성을 지원하는 SLAM 시스템
      - Nice-SLAM [46]
      - Orbeez-SLAM [4]
      - ESLAM [16]
      - Co SLAM [36]
      - Point-SLAM [27]
      - Go-SLAM [44]
   
**Hardware**

- 데스크탑에서 Photo-SLAM 및 비교된 모든 방법을 공식 코드를 사용하여 실행
   - 데스크탑: NVIDIA RTX 4090 24 GB GPU, Intel Core i9-13900K CPU, 64 GB RAM
- 노트북과 Jetson AGX Orin 개발 키트에서 Photo-SLAM을 테스트
   - 노트북: NVIDIA RTX 3080ti 16GB laptop GPU, Intel Core i9-12900HX, 32 GB RAM

**Datasets and Metrics**

단안 및 RGB-D 테스트
- Replica
- TUM RGB-D

스테레오 테스트
- EuRoC MAV

실내 장면 외 추가 평가
- ZED 2 스테레오 카메라를 사용하여 야외 장면 수집

Metrics
- Absolute Trajectory Error(ATE): localization 정확도 추정
- ATE의 RMSE 및 STD 보고
- PSNR, SSIM, LPIPS: photorealistic mapping 성능 분석
- 추적 FPS, 렌더링 FPS 및 GPU 메모리 사용량 표시

다중 스레딩 및 머신 러닝 시스템의 비결정론적 특성의 영향을 줄이기 위해 각 시퀀스를 5번 실행하고 각 지표의 평균 결과를 보고

### 4.2 Results and Evaluation

**On Replica**

![alt text](./images/Table%201.png)

![alt text](./images/Fig%206.png)

![alt text](./images/Table%202.png)

![alt text](./images/Fig%207.png)

![alt text](./images/Fig%208.png)

![alt text](./images/Table%203.png)

![alt text](./images/Fig%209.png)

![alt text](./images/Table%204.png)

![alt text](./images/Fig%2010.png)