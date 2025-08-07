# DUSt3R: Geometric 3D Vision Made Easy



---

- Image Matching

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html) (CVPR 2024)
- [project page](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)

---
짧은 요약



요약


---

**문제점 & 한계점**


---

목차

0. [Abstract](#abstract)
1. 

---


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

DUSt3R
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

우리는 직접 회귀를 통해 일반화된 스테레오 경우에 대한 3D 재구성 작업을 해결하는 네트워크를 구축하고자 합니다. 이를 위해, 두 개의 RGB 이미지 I1, I2 ∈ RW×H×3를 입력으로 받고 두 개의 해당 포인트 맵 X1,1, X2,1 ∈ RW×H×3과 관련된 신뢰 맵 C1,1, C2,1 ∈ RW×H을 출력하는 네트워크 f를 훈련합니다. 두 포인트 맵은 I1의 같은 좌표 프레임으로 표현된다는 점에 유의하세요. 이는 기존 접근 방식과는 본질적으로 다르지만 주요 장점을 제공합니다(섹션 1, 2, 3.3 및 3.4 참조). 명확성을 위해 그리고 일반화를 잃지 않기 위해, 우리는 여기서 두 이미지의 해상도가 W × H로 동일하다고 가정하지만, 실제로 해상도가 다를 수 있습니다. 네트워크 아키텍처. 우리의 네트워크 f의 아키텍처는 CroCo [114]에서 영감을 받아 단순하게 CroCo의 사전 훈련 [113]에서 큰 이점을 얻을 수 있게 합니다. 그림 2에 표시된 바와 같이, 이 네트워크는 두 개의 동일한 가지로 구성되어 있으며(각 이미지에 하나씩) 각 가지는 이미지 인코더, 디코더 및 회귀 헤드로 구성됩니다. 두 입력 이미지는 먼저 동일한 가중치를 공유하는 ViT 인코더 [25]에 의해 시암 브랜드 방식으로 인코딩되어 두 개의 토큰 표현 F1과 F2를 생성합니다.



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


