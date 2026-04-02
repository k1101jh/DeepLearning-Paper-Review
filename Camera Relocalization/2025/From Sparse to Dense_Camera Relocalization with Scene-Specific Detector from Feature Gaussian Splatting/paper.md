# From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting

---

## 📌 Metadata
---
분류
- Camera Relocalization
- Gaussian Splatting
---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.html) (CVPR 2025)
- [project page](https://zju3dv.github.io/STDLoc/)
---
- **Authors**: Zhiwei Huang, Hailin Yu, Yichun Shentu, Jin Yuan, Guofeng Zhang
- **Venue**: CVPR 2025

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2.3 Radiance Field-Based Methods](#23-radiance-field-based-methods)
- [3. Method](#3-method)
  - [3.1 Feature Gaussian Training](#31-feature-gaussian-training)
  - [3.2 Matching-Oriented Sampling](#32-matching-oriented-sampling)
  - [3.3 Scene-Specific Detector](#33-scene-specific-detector)
  - [3.4 Sparse-to-Dense Localization](#34-sparse-to-dense-localization)

---

## ⚡ 요약 (Summary)
### 문제점 (사용자 작성)
- Gaussian을 의도적으로 균등하게 분포시킴  
- -> 사실적인 map은 생성 불가. Gaussian Splatting의 의미가 퇴색됨

### 핵심 요약
- **Problem**: 전통적인 특징점 기반 재위치 추정은 텍스처가 부족한(Weak-texture) 환경에서 매칭 실패가 잦으며, SCR 방식은 대규모 실외 장면에서의 정확도가 제한적임.
- **Idea**: Feature Gaussian을 활용한 새로운 'Sparse-to-Dense' 패러다임을 제안하고, 효율적인 초기 포즈 추정을 위한 Matching-Oriented 샘플링과 장면 맞춤형 검출기(Scene-specific detector)를 도입함.
- **Result**: 쿼리 특징 맵을 가우시안 특징 필드와 정밀하게 정렬하여 위치 추정 정확도 및 재현율 측면에서 SOTA를 달성함.

---

## 📖 Paper Review

## Abstract

**STDLoc**
- Feature Gaussian을 활용하는 Full relocalization pipeline
- 자세 정보에 의존하지 않고도 정확한 relocalization 가능
- 이전 방식
    1. 이미지 검색
    2. feature matching
- 새로운 sparse-to-dense localization 패러다임 제안
- 새로운 matching-oriented gaussian sampling 전략과 scene-specific detector 도입
    - 효율적이고 견고한 초기 자세 추정을 위해
- 초기 위치 지정 결과를 기반으로 dense feature matching  
-> query feature map을 gaussian feature field와 정렬하여 정확한 위치 지정을 가능하게 함
- 위치 지정 정확도 및 재현율 측면에서 SOTA localization 방법 능가

## 1. Introduction

![alt text](./images/Fig%201.png)
> **Figure 1. STDLoc: Sparse-to-Dense Localization**  
> 장면 표현으로 Feature Gaussian을 활용하여 랜드마크에서 직접 2D-3D sparse matching을 지원  
> dense matching을 통해 query feature map을 feature field에 정렬할 수 있음

Camera Relocalization
- 사전에 구축된 scene map과 비교하여 쿼리 이미지의 6DoF 카메라 자세를 추정

Scene representation
- 적절한 scene representation은 카메라 relocalization에 핵심적인 역할을 함
- 전통 방법
    - 사전 SfM으로 재구성된 희소 3D pointCloud 사용
        - 각 3D 포인트는 하나 이상의 2D descriptors와 연결됨
    - 쿼리 이미지에서 reliable local feature을 추출
    - 추출한 feature을 reference 이미지 또는 직접 3D point cloud와 매칭
    - PnP(Perspective-n-Point) 알고리즘을 사용하여 2D-3D 대응점 기반으로 6DoF 포즈 추정
    - 장점: 풍부한 texture 환경에서 높은 정확도
    - 한계: weak-texture 환경에서 feature 대응점 부족으로 정확도 저하
        - Semi-dense / dense matching으로 완화 가능
            - 매핑 단계에서 SfM 계산량이 과도하여 실용성 낮음
        - mesh 활용 -> 깊이 정보는 제공 가능하지만 artifact 발생 시 위치 정확도에 심각한 영향을 줌

- 딥러닝 기반 접근
    - 장면 정보를 newral network에 인코딩
    - APR(Absolute Pose Regression)[9, 20]
        - 절대 포즈 직접 회귀
        - 일반적으로 정확도 한계
        - unseen view에 대한 정확도 저하
    - SCR(Scene Coordinate Regression)
        - 장면 좌표를 조밀하게 회귀
        - 실내 weak-texture 환경에서 feature matching 기반 방법보다 정확도 높음
        - 한계
            - 네트워크 가중치에 장면 정보가 직접 인코딩
                - 목표 장면 크기에 따라 네트워크 용량을 동적으로 조정 불가
            - 대규모 실외 장면에서의 정확도는 상대적으로 제한됨


### 2.3 Radiance Field-Based Methods

- 최근 연구 동향: NeRF[34]와 Gaussian Splatting[19, 21]을 사용한 활발한 연구 진행중
- Inverse Rendering
    - 역렌더링을 통해 카메라 포즈 직접 최적화
    - iNeRF[64]
        - 최초로 inverse rendering을 적용한 카메라 포즈 최적화 방법
    - PNeRFLoc[68]
        - 2D-3D feature matching으로 초기 포즈 추정
            -> 새로운 view 합성을 활용해 포즈 정제
    - NeRFMatch[69]
        - NeRF 내부 특징을 활용한 정밀 2D-3D 매칭
            -> 포토메트릭 오류 최소화로 포즈 최적화
    - CROSSFIRE[37]
        - 포토메트릭 정합 대신 volumetric rendering의 조밀 로컬 특징을 활용해 robustness 향상
    - NeuraLoc[67]
        - 상호 보완적인 특징 학습으로 정확한 2D-3D 대응 관계 수립
    - Lens[36]
        - NeRF를 이용해 추가 시점 합성하여 학습 데이터셋 확장
- Gaussian 기반 localization
    - NeRF보다 효율적
    - 6DGS[2]
        - 사전 포즈 없이 3DGS 모델에서 직접 6DoF 카메라 포즈 추정
    - GSplatLoc[51]
        - dense keypoint descriptors를 3DGS에 통합
            -> 초기 포즈 추정 후 photometric warping loss 최적화
    - LoGS[13]
        - 이미지 검색과 PnP solver로 로컬 feature 매칭
        - 이후 analysis-by-synthesis(분석 기반 합성)으로 정제
    - GS-CPR[28]
        - 3DGS로 고품질 이미지를 렌더링해 신경망 기반 방법의 정합 정확도 향상

- 제안 방법의 차별점
    - 완전한 localization 파이프라인 제안
    - 새로운 Sparse-to-Dense 패러다임
        1. 샘플링된 scene landmarks와 scene-specific detector을 사용해 효율적 초기 포즈 추정
        2. feature gaussian이 제공하는 feature field를 활용하여 포즈 추가 정제
    - 높은 포즈 추정 정확도
    - 조명 변화나 weak-texture 환경에서도 강건한 성능 유지

## 3. Method

### 3.1 Feature Gaussian Training

![alt text](./images/Fig%202.png)

> **Figure 2.**  
> Feature Gaussian은 radiance field loss $\mathcal{L}_{rgb}$와 feature field loss $\mathcal{L}_f$를 공동으로 최적화하도록 훈련됨

scene representation
- feature field로 augment된 Gaussian primitive로 구성
- 학습 가능한 속성 집합($\Theta$)
    - $x_i$: 중심(Center)
    - $q_i$: 회전(Rotation)
    - $s_i$: 스케일(Scale)
    - $\alpha_i$: 불투명도(Opacity)
    - $c_i$: 색상(Color)
    - $f_i$: 특징(Feature)
- Feature 3DGS[70] 방식 기반으로 radiance field과 feature field를 공동 최적화
- 모든 명시적 primitives기반 3DGS 변형에 적용 가능

Gaussian Primitive
- SfM(Structure-from-Motion) 포인트 클라우드로 Gaussian Primitive 초기화
- Feature Map
    - $F_t(I) \in \mathbb{R}^{D \times H' \times W'}$: 학습 이미지 $I \in \mathbb{R}^{3 \times H \times W}$에서 추출한 Dense Feature Map
    - $D$: 로컬 특징의 차원 수
- GT Feature Map $F_t(I)$:
    - SuperPoint [14]와 같은 범용 로컬 특징 추출기로 획득할 수 있음

렌더링 과정
- Gaussian radiance field
    - alpha blending 기법을 사용하여 색상 속성 $c$를 렌더링 RGB 이미지 $\hat{i}$로 변환
    - 같은 방식으로 Feature 속성 $f$를 사용해 렌더링 Feature Map $\hat{F}_s$ 생성

loss
- Radiance Field Loss($L_{rgb}$)
    - $L_1$ 손실 + D-SSIM 손실 조합:
    $$
    \displaystyle
    \begin{aligned}
    & \mathcal{L}_{rgb} = (1 - \lambda) \mathcal{L}_1(I, \hat{I}) + \lambda \mathcal{L}_{D-SSIM}(I, \hat{I}) \tag{1}
    \end{aligned}
    $$
- Feature Field Loss ($L_f$)
    - GT Feature Map과 렌더링 Feature Map 간 $L_1$ 거리:
    $$
    \displaystyle
    \begin{aligned}
    & \mathcal{L}_f = \mathcal{L}_1(F_t(I), \hat{F}_s) \tag{2}
    \end{aligned}
    $$
- 최종 손실 ($L$)
    - 두 손실을 가중합:
    $$
    \displaystyle
    \begin{aligned}
    & \mathcal{L} = \alpha \mathcal{L}_{rgb} + \beta \mathcal{L}_f \tag{3}
    \end{aligned}
    $$

- $\lambda = 0.2, \alpha = 1.0, \beta = 1.0$ 사용
- 훈련으로 얻은 Feature Gaussian scene을 $\mathcal{G}$로 표현

### 3.2 Matching-Oriented Sampling

![alt text](./images/Fig%203.png)

> **Figure 3. Matching-Oriented Sampling**  
> 각 가우시안은 anchor sampling에 따라 매칭 점수가 할당됨  
> 각 앵커에 대해, 공간적 거리 기준으로 k개의 가까운 가우시안이 식별되며, 그 중 가장 높은 점수를 가진 가우시안이 선택됨


- 장면 내 모든 Gaussian과 쿼리 feature을 모두 매칭하면 시간 소모가 큼
- 모호한 Gaussian이 많으면 feature matching 정확도 저하 가능
- 목표
    - 수백만 개의 Gaussian primitive 중 매칭에 적합한 것만 선택
    - 장면 전체에 균등 분포
    - 여러 시점에서 인식 가능한 gaussian을 유지

가우시안 품질 평가
- 각 가우시안 $g_i$와 학습 이미지 $I$에 대해
    - $g_i$의 중심을 $I$에 투영하여 2D 좌표 $(u_i, v_i)$ 획득
    - feature map $F_t(I)$에서 해당 위치의 2D feature 추출  
    -> Bilinear interpolation 사용: $F_t(I)[u_i, v_i]$
    - Gaussian의 3D feature $f_i$와 2D feature 간 코사인 유사도 계산
- $V_i$: $g_i$가 보이는 이미지들의 집합

- 최종 점수 $s(g_i)$:

$$
\displaystyle
\begin{aligned}
& s(g_i) = \frac{1}{|\mathcal{V}_i|} \sum_{I \in \mathcal{V}_i} \langle f_i, F_t(I)[u_i, v_i] \rangle
\tag{4}
\end{aligned}
$$

- 점수가 높을수록 매칭 적합도가 높음

점수 기반 단일 선택의 문제
- 불균형한 공간 분포 발생
    - 고텍스처 영역: Gaussian이 과밀 집적
    - 저텍스처 영역: Gaussian이 희소, 매칭 성능 저하
- 해결 방법
    - downsampling
        - Random/Farthest point sampling 진행
        - 균등 분포 확보를 위해 고정 개수의 gaussian을 anchor로 sample
        - 각 anchor마다, 공간 거리 기준 k nearset neighbor을 식별
        - 해당 집합 중 점수가 가장 높은 gaussian 선택
- 결과
    - 원본 대비 gaussian 수 대폭 축소
    - 최종 집합:
        - 공간적으로 균등함
        - 여러 시점에서 높은 인식률 보유
    - 실험 결과 수천 개 수준의 gaussian만으로도 충분히 효과적인 localization 성능 달성
    - sampled gaussian을 scene landmarks $\tilde{\mathcal{G}}$라고 함

### 3.3 Scene-Specific Detector

![alt text](./images/Fig%204.png)

> **Figure 4. Scene-Specific Detector Training**  
> 샘플된 랜드마크의 중심은 2D 이미지에 투영되어 scene-specific detector의 훈련을 guide함

샘플링된 랜드마크와 dense feature map을 직접적으로 매칭하는 것은 불가능
- dense feature map에는 많은 수의 feature이 있음
- 하늘과 같은 유효하지 않은 영역의 feature은 매칭에 부적합함
- 기존의 SuperPoint와 같은 detector은 장면에 무관한 사전 정의된 keypoint 검출
    - Feature Gaussian 장면에서 샘플된 랜드마크와 잘 맞지 않음

제안 방법
- scene-specific feature detector $\mathcal{D}_\theta$
    - 입력: 이미지 $I$로부터 추출된 feature map $F_t(I)$
    - 출력: heatmap $\hat{K} \in \mathbb{R}^{1 \times H' \times W'}$
        - 각 2D feature이 랜드마크일 확률을 나타냄

$$
\displaystyle
\begin{aligned}

& \hat{K} = D_{\theta}(F_t(I))
& \tag{5}

\end{aligned}
$$

네트워크 구조
- $\mathcal{D}_\theta$는 얕은(shallow) convolutional neural network(CNN)으로 구성됨

학습 방식
- self-supervised 학습
- 샘플링된 랜드마크 집합 $\tilde{\mathcal{G}}$의 각 gaussian 중심을 이미지 평면에 투영
- 해당 픽셀 위치를 1로 설정하여 GT(ground truth) heatmap $K$ 생성
- Binary Cross-Entropy Loss로 최적화

$$
\displaystyle
\begin{aligned}

& L_{\text{det}} = -K \log(\hat{K}) - (1 - K) \log(1 - \hat{K})
& \tag{6}

\end{aligned}
$$

추론
- 생성된 heatmap $\hat{K}$에 Non-Maximum Suppression(NMS) 적용
- 검출된 keypoint들이 균일하게 분포되도록 함

### 3.4 Sparse-to-Dense Localization

![alt text](./images/Fig%205.png)

> **Figure 5. Feature Gaussian 기반 sparse-to-dense localization pipeline**

Sparse-to-dense localization pipeline
- sparse feature matching은 sampled landmark $\tilde{\mathcal{G}}$와 $\mathcal{D}_\theta$로 탐지한 sparse local feature로 구성됨
- sparse matching을 통해 얻은 2D-3D correspondences 기반으로 initial camera pose는 PnP 알고리즘으로 계산 가능
- dense feature map은 full Feature Gaussian $\mathcal{G}$로 렌더링 될 수 있음
- 이후 coarse-to-fine dense feature matching으로 포즈 refine

Sparse Stage
- 입력: 쿼리 이미지 $I_q$
- dense feature map $F_t(I_q)$ 추출
- $\mathcal{D}_\theta$를 사용해 sparse local feature 검출
- sparse local feature와 $\tilde{\mathcal{G}}$의 모든 랜드마크 간의 cosine similarity계산
- 각 local feature에 대해 가장 유사도가 높은 landmark를 매칭으로 선택
- 로컬 특징의 2D 좌표와 해당 랜드마크의 3D 중심 좌표를 2D-3D correspondences로 설정  
-> sparse match set $\mathcal{M}_{sparse}$ 형성
- PnP + RANSAC으로 초기 포즈 $\xi_{sparse}$ 추정

Dense Stage
- 초기 포즈 $\xi_{sparse}$를 사용해서 Feature Gaussian 장면 $\mathcal{G}$에서
    - 고해상도 Dense feature map $\hat{F}_s$ 렌더링
    - depth map $\hat{D}$ 렌더링
- LoFTR[52] 방식을 참고하여 저해상도 $D \times H_f / 8 \times W_f / 8$에서 coarse matching
- 이후 full 해상도 $D \times H_f \times W_f$에서 refine
- 고해상도 feature map을 직접 렌더링 후, bilinear interpolation으로 저해상도 버전 생성
- Coarse Matching
    - 코사인 유사도로 coarse feature map 간 상관 행렬 $S_c$ 계산
    - Dual-softmax op로 확률 행렬 $P_c$ 생성
    - MNN 검색으로 coarse correspondence 집합 $\mathcal{M}_c$ 획득

$$
\displaystyle
\begin{aligned}

& P_c = \text{softmax}\left(\frac{1}{\tau} S_c\right)_{\text{row}} \cdot \text{softmax}\left(\frac{1}{\tau} S_c\right)_{\text{col}}
& \tag{7}

\end{aligned}
$$

> $\tau$: 온도 파라미터


- Fine Matching
    - 각 coarse correspondence 위치에서 $8 \times 8$ 패치 추출
    - 동일한 방식으로
        - correlation matrix $S_f$ 계산
        - probability matrix $P_f$ 계산
        - MNN 검색으로 refined matches $\mathcal{M}_f$ 획득
