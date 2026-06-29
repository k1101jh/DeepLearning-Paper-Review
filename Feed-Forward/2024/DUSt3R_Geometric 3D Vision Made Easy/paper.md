# DUSt3R: Geometric 3D Vision Made Easy

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Image Matching
- 3D Reconstruction
- MVS
---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html)
- [project](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)
---
- **Authors**: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [3. Method](#3-method)
  - [3.1 개요](#31-개요)
  - [3.2 훈련 목표](#32-훈련-목표)
  - [3.3 Downstream Applications](#33-downstream-applications)

---

## ⚡ 요약 (Summary)
- **Problem**: 기존 Multi-view Stereo(MVS)는 카메라의 내/외부 파라미터(Calibration, Poses) 추정이 필수적이며, 이는 실제 환경(In-the-wild)에서 큰 병목 현상이 됨.
- **Goal**: 카메라 정보에 대한 사전 지식 없이 임의의 이미지 컬렉션으로부터 밀집한(Dense) 3D 재구성을 수행하는 새로운 패러다임 제안.
- **Key Method**: 
    - **Pointmap Regression**: 3D 재구성 문제를 포인트맵($X \in \mathbb{R}^{W \times H \times 3}$) 회귀 문제로 변환하여 엄격한 투영 모델 제약을 완화함.
    - **Siamese ViT Encoder-Decoder**: 두 이미지의 특징을 정보를 교환하며 인코딩하고, 첫 번째 이미지의 좌표계에서 두 개의 포인트맵을 직접 출력함.
    - **Confidence-aware Loss**: 하늘이나 반투명 객체 등 불확실한 영역을 처리하기 위해 신뢰도 점수를 함께 학습함.
- **Result**: 단안/다중 뷰 깊이 추정 및 상대 포즈 추정에서 SOTA 성능을 기록하며, Pixel matching 및 Focal length 복구 등 다양한 하위 작업에 적용 가능함.

---

## 📖 Paper Review

## Abstract

![alt text](./images/Fig%201.png)

> **Figure 1.**  
> 상단: DUSt3R은 제약이 없는 이미지 모음을 입력으로 받아서 pointmap을 출력  
> 하단: 입력 카메라 자세나 내부 매개변수 없이 완전히 일관된 3D 재구성  
> 죄->우: 입력 이미지, 색이 입혀진 point cloud, 음영이 있는 렌더링.  
> 우측 상단: 시각적 겹침 없이 장면을 재구성 할 수도 있음

Multi-view stereo reconstruction(MVS)
- in-the-wild에서 할 때, 카메라의 내부 및 외부 매개변수를 추정해야 함
    - 3D 공간에서 대응하는 픽셀을 삼각 측량하는 데 필수적

**DUSt3R**
- 본 연구에서는 카메라 calibration이나 viewpoint poses에 대한 prior 없이 임의의 이미지 컬렉션에 대한 밀집하고(Dense) 제약 없는(Unconstrained) Stereo 3D Reconstruction을 위한 패러다임인 DUSt3R을 소개
- pairwise 재구성 문제를 pointmap의 회귀로 바꾸어 일반적인 투영 카메라 모델의 엄격한 제약을 완화
- 단안 및 쌍안 재구성 사례를 부드럽게 통합
- 두 개 이상의 이미지가 제공되는 경우, 모든 pairwise pointmaps을 common reference frame에서 표현하는 간단하면서도 효과적인 글로벌 alignment 전략을 제안
- 표준 Transformer 인코더 및 디코더를 기반으로 강력한 사전 훈련된 모델을 활용할 수 있도록 함
- 제안 공식은 장면의 3D 모델과 깊이 정보를 직접 제공
- pixel match, focal length, 상대 및 절대 카메라를 원활하게 복구할 수 있음
- 단안 및 다중 뷰 깊이 추정 및 상대 포즈 추정에서 새로운 성능 기록을 세움

## 3. Method

**Pointmap**

- 3D 점의 밀집 2D 필드를 pointmap $X \in \mathbb{R}^{W \times H \times 3}$으로 표기
- $W \times H$ 해상도의 해당 RGB 이미지 $I$와 함께, $X$는 이미지 픽셀과 3D 장면 점 사이의 일대일 매핑을 형성
    - 모든 픽셀 좌표 $(i, j) \in \{1 ... W\} \times \{ 1 ... H \}에 대해 $I_{i, j} \leftrightarrow X_{i, j}$
- 카메라 광선이 단일 3D 점에 부딪힌다고 가정(반투명 점은 무시)

**Cameras and scenes**

- 카메라 내부 파라미터 $K \in \mathbb{R}^{3 \times 3}$을 고려할 때, pointmap $X$는 GT depthmap $D \in \mathbb{R}^{W \times H}$로부터 간단히 얻을 수 있음: $X_{i, j} = K^{-1}D_{i, j} [i, j, 1]^\top$ ($X$는 카메라 좌표계에서 표현됨)
- 카메라 $m$의 좌표계로 표현된 카메라 $n$의 pointmap $X^n$을 $X^{n, m}$으로 표기:

$$
\displaystyle
\begin{aligned}

& X^{n ,m} = P_m P_n^{-1} h(X^n)
& (1)

\end{aligned}
$$

> $P_m, P_n \in \mathbb{R}^{3 \times 4}$: 이미지 m과 n의 world-to-camera 포즈
> $h : (x, y, z) \rightarrow (x, y, z, 1)$: homogeneous 매핑


### 3.1 개요

- 목표: direct regression을 통해 일반화된 스테레오 경우에 대한 3D 재구성 작업을 해결하는 네트워크 구축
- 입력: 두 장의 RGB 이미지 $I^1, I^2 \in \mathbb{R}^{W \times H \times 3}$
- 출력:
    - 두 개의 대응되는 pointmap $X^{1, 1}, X^{2, 1} \in \mathbb{R}^{W \times H \times 3}$
    - 두 개의 연관된 신뢰도 맵 $C^{1, 1}, C^{2, 1} \in \mathbb{R}^{W \times H}$
- 두 pointmap 모두 $I^1$의 좌표계에서 표현됨  
  -> 기존 방식과 차별화되어 여러 장점 제공(1, 2, 3.3, 3.4절 참조)
- 두 이미지 해상도를 $W \times H$로 동일하게 가정(실제로는 다를 수 있음)

**네트워크 아키텍처**

![alt text](./images/Fig%202.png)
> **Figure 2. 네트워크 아키텍처**  
> 씬의 두 가지 뷰($I^1, I^2$)는 shared ViT encoder을 사용하여 Siamese 방식으로 encode됨  
> 결과 토큰 표현 $F^1, F^2$는 두 개의 transformer decoder을 거침(cross-attention으로 정보를 교환)  
> 최종적으로, 두 개의 regression head는 두 개의 pointmap과 연관된 confidence map을 출력  
> 두 개의 pointmap은 첫 번째 이미지의 좌표로 표현됨  
> 네트워크는 simple regression loss(식 4)를 통해 훈련됨

네트워크 $\mathcal{f}$
- CroCo[114]에서 영감을 받음(CroCo 사전학습 활용 가능)
- 두 개의 동일한 branch(각각 하나의 이미지 처리)
    - 이미지 인코더(encoder)
        - ViT(Visual TranseFormer) 기반
        - 두 입력 이미지는 Siamese 방식으로 처리되며 가중치 공유
        - $F^1=\mathrm{Encoder(I^1)}, F^2=\mathrm{Encoder}(I^2)$
    - Decoder
        - CroCo와 유사하게, cross-attention을 포함한 generic transformer network사용
        - 각 decoder block은 self-attention, cross-attention, MLP순으로 구성
        - decoder pass 중 두 브랜치는 정보를 지속적으로 공유  
        -> pointmap 정렬에 필수
    
$$
\displaystyle
\begin{aligned}

& G^1_i = \mathrm{DecoderBlock}^1_i (G^1_{i-1}, G^2_{i-1}), \\
& G^2_i = \mathrm{DecoderBlock}^2_i (G^2_{i-1}, G^1_{i-1})

\end{aligned}
$$

> $i=1, \dots, B$, B는 decoder 블록 개수  
> encoder token 초기값: $G^1_0 := F^1,\; G^2_0 := F^2$  
> $\mathrm{DecoderBlock}_i^v(G^1, G^2)$: 브랜치 $v \in {1, 2}$의 i번째 블록. $G^1, G^2$는 입력 토큰. $G^2$는 다른 브랜치의 입력  

- 각 브랜치의 regression head가 decoder token set을 받아서 pointmap과 연관된 confidencemap을 출력

$$
\displaystyle
\begin{aligned}

& (X^{1,1}, C^{1,1}) = \mathrm{Head}^1(G^1_0, ..., G^1_B) \\
& (X^{2,1}, C^{2,1}) = \mathrm{Head}^2(G^2_0, ..., G^2_B)

\end{aligned}
$$

**Discussion**

- 출력 pointmap $X^{1, 1}, X^{2, 1}$은 unknown scale factor에 대해 회귀됨
- 제안한 architecture은 명시적으로 어떤 기하학적 제약도 강제하지 않음
    - pointmap은 물리적으로 완전히 일치하는 카메라 모델을 보장하지 않음
    - 실제로는 근접하게 맞음
- 기하학적으로 일관된 pointmap만 포함하는 훈련 세트에 존재하는 모든 관련 사전 지식을 학습하도록 허용
- 일반 아키텍처를 사용하면 강력한 사전학습 기술을 사용할 수 있음
    - 기존의 작업 별 아키텍처보다 성능 우위 가능


### 3.2 훈련 목표

**3D Regression loss**

- 목표: 3D 공간에서의 회귀
- GT pointmap을 $\bar{X}^{1, 1}, \bar{X}^{2, 1}$로 표시
    - GT가 정의된 두 개의 유효 픽셀 집합 $\mathcal{D}^1, \mathcal{D}^2 \subseteq \{ 1 ... W \} \times \{ 1 ... H \}$ 포함
- 유효 픽셀 $i \in \mathcal{D}^v$에 대한 regression loss는 view $v \in \{1, 2\}$에서 단순히 euclidean 거리로 정의됨:

$$
\displaystyle
\begin{aligned}

& \ell_{\mathrm{regr}}(v,i) = \big\| \tfrac{1}{z}\,X_i^{v,1} \;-\; \tfrac{1}{\bar z}\,\bar X_i^{v,1} \big\|.
& (2)

\end{aligned}
$$

prediction과 GT 간의 스케일 모호성을 처리
- 각각 scaling factor $z=\text{norm}(X^{1, 1}, X^{2, 1})과 \bar{z} = \text{norm}(\bar{X}^{1, 1}, \bar{X}^{2, 1})$로 predicted & GT pointmap을 정규화
    - 이는 단순히 모든 유효 점의 거리 평균을 나타냄

$$
\displaystyle
\begin{aligned}

& \mathrm{norm}(X^1, X^2) = \frac{1}{|D^1| + |D^2|} \sum_{v\in\{1,2\}} \sum_{i\in D^v} \big\| X_i^v \big\|.
& (3)

\end{aligned}
$$

**Confidence-aware loss**
- 하늘이나 반투명 객체와 같은 잘 정의되지 않은 3D 포인트 존재
- 네트워크가 각 픽셀에 대해 신뢰도를 예측하는 방법을 학습
- 모든 유효 픽셀에 대한 confidence-weighted regression loss:

$$
\displaystyle
\begin{aligned}

& L_{\mathrm{conf}} = \sum_{v \in \{1,2\}} \sum_{i \in D^v} C^{v,1}_i · ℓ_{regr}(v,i) − α · \mathrm{log} C^{v,1}_i
& (4)

\end{aligned}
$$

> $C^{v,1}_i$: $i$번째 픽셀의 confidence score  
> $\alpha$: 정규화 항을 제어하는 하이퍼 파라미터

- confidence값이 양수가 되도록 보장하기 위해 다음과 같이 정의
    - 네트워크가 예측이 어려운 영역(예: 단일 뷰에서만 관측된 영역)에 대해서도 추론을 강제하는 효과
    - 네트워크 $f$를 학습하면 명시적인 supervision(정답) 없이 confidence score 추정 가능

$$ C_i^{v,1} = 1 + \exp(c_i^{v,1}) \gg 0, \quad c_i^{v,1} \in \mathbb{R} $$

### 3.3 Downstream Applications

**Point matching**
- 두 이미지간 픽셀 대응은 3D pointmap 매칭 공간에서 NN(Nearest Neighbor) 검색으로 간단히 구함
- 오차 최소화를 위해 서로의 최근접 이웃이 일치하는 상호 대응(mutual correspondences)만 유지.

$$
\displaystyle
\begin{aligned}

& \mathcal{M}_{1,2} = \{(a,b) \mid a = \mathrm{NN}^{1,2}(b) \ \text{and} \ b = \mathrm{NN}^{2,1}(a) \} \\
& \mathrm{with} \quad \mathrm{NN}^{n,m}(a) = \argmin_{b \in \{0, \dots, WH\}} \left\| X_b^{n,1} - X_a^{m,1} \right\|

\end{aligned}
$$

**Recovering intrinsics**

- $X^{1, 1}은 이미지 $I^1$의 좌표계로 표현됨
- principal point는 중앙, pixel은 정사각형이라 가정  
-> focal length $f_1^*$만 추정하면 됨

$$
\displaystyle
\begin{aligned}

& f_1^* = \argmin_{f_1} \sum_{i=0}^W \sum_{j=0}^H C_{i,j}^{1,1} \left\| (i', j') - f_1 \frac{\big(X_{i,j,0}^{1,1}, \ X_{i,j,1}^{1,1}\big)}{X_{i,j,2}^{1,1}} \right\|

\end{aligned}
$$

> $i' = i - \frac{W}{2}, \quad j' = j - \frac{H}{2}$

- weiszfeld algorithm등 빠른 iterative solver로 수 iteration 내 optimal $f_1^*$을 찾을 수 있음
- 두 번째 카메라의 focal length $f_2^*$
    - 가장 쉬운 방법: 이미지만 서로 바꿔서 위의 공식으로 계산($X^{1, 1}$ 대신 $X^{2, 2}$를 사용. 이미지가 바뀌었으므로)

**Relative pose estimation**

방법
1. 2D 매칭 + intrinsic 추정 -> Epipolar matrix 추정 -> 상대 포즈 복원
2. pointmap $X^{1,1} \leftrightarrow X^{1,2}$(혹은 $X^{2,2} \leftrightarrow X^{1,2}$) 비교
    - Procrustes alignment[54]를 사용해서 scaled relative pose $P^* = \sigma^* [R^* | t^*]$ 추정

$$
\displaystyle
\begin{aligned}

& P^* = \arg\min_{\sigma, R, t} \sum_i C_i^{1,1} C_i^{1,2} \left\| \sigma \big( R X_i^{1,1} + t \big) - X_i^{1,2} \right\|^2

\end{aligned}
$$

- closed-form 해 존재
- noise, 이상치에 민감  
-> RANSAC + PnP로 더 안정적인 추정 가능

**Absolute pose estimation**

- 쿼리 이미지: $I^Q$
- 참조 이미지: $I^B$(2D-3D 대응이 있는 경우)
- 방법
    1. $X^{Q, Q}로 $I^Q$의 intrinsic 추정
    2. 
        - 방법 1. $I^Q$와 몇몇 $I^B$ 간의 2D pixel 매칭을 사용해서 $I^Q$의 2D-3D 대응 계산  
        -> PnP-RANSAC 실행
        - 방법 2. $I^Q$와 $I^B$ 상대 자세 추정  
        -> $X^{B, B}$와 $I^B$의 GT pointmap 간의 scale을 맞춰서 자세를 월드 좌표계로 변환