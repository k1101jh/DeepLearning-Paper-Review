# Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition


---

- Camera Calibration

---

url
- [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Hausler_Patch-NetVLAD_Multi-Scale_Fusion_of_Locally-Global_Descriptors_for_Place_Recognition_CVPR_2021_paper.html)

---

목차

0. [Abstract](#abstract)
1. 

---


## Abstract

**Visual Place Recognition**
- 항상 변화하는 세계에서 외형 변화와 시점 변화라는 두 가지 문제를 다루어야 함

**Patch-NetVLAD**
- NetVLAD의 residual에서 patch-level feature을 유도
    - local 및 global descriptor 방법의 장점을 결합
- 기존의 fixed spatial neighborhood regime의 local keypoint features와 달리, feature-space grid에서 정의된 deep-learned local features의 집합 및 매칭을 가능하게 함
- 통합 특징 공간(integral feature space)를 통해 상호 보완적인 scale(patch 크기)를 가진 patch features의 multi-scale fusion을 소개
- fused features는 condition(계절, 구조, 조명) 및 시점(translation 및 rotation) 변화에 대해 매우 불변적임을 보임
- 계산이 제한된 시나리오에서 SOTA visual place recognition 결과를 달성
- 다양한 도전적인 실제 데이터셋에서 검증됨
- 독립적인 place recognition 능력과 SLAM systemd의 전반적인 성능을 향상시키기에 적합함


## 1. Introduction

![alt text](./images/Fig 1.png)
> **Figure 1. Patch-NetVLAD**  
> - Patch-NetVLAD는 두 이미지 간의 유사성 점수를 생성하는 새로운 condition and viewpoint 불변 visual place recognition system
> - 각 이미지의 feature space에서 patches 집합에서 추출한 locally-global descriptors의 local matching을 통해 수행됨
> - 적분 feature space를 도입하여 여러 patch 크기를 융합하는 multi-scale 접근 방식을 도출
> - 전체 feature space의 외관만을 집계하여 단일 global descriptor로 만드는 원래의 NetVLAD 논문과 대조적임

**Visual Place Recognition(VPR)**
- prior map을 사용할 때 stand-alone positioning 기능으로 작용
- 전체 Simultaneous Localization And Mapping(SLAM) system의 핵심 구성 요소로 작용
- 외관, 조명 및 시점의 변화로 인해 어려울 수 있다.

**작동 방식**
- 이미지 검색 작업: 쿼리 이미지가 주어지면 가장 유사한 데이터베이스 이미지(카메라 포즈같은 메타데이터와 함께)를 검색
- query 이미지와 reference 이미지를 나타내는 두 가지 일반적인 방법
  - 전체 이미지를 설명하는 global descriptor 사용
    - 매칭은 일반적으로 query 이미지와 ref 이미지 간의 nearest neighbor을 통해 수행
    - 일반적으로 외관 및 조명 변화에 대한 강인성이 뛰어남
  - 관심 영역(area of interest)을 설명하는 local descriptor 사용
    - 일반적으로 교차 매칭되고, 기하학적 검증이 뒤따름
    - 픽셀 단위 수준에서 공간 정밀도를 우선시함
    - 개별 구성 요소가 고정 크기의 spatial neighborhood를 사용하여 매우 정확한 6-DoF pose estimation을 용이하게 함
- Patch-NetVLAD 시스템은 local 및 global 접근 방식의 상호 강점을 결합하여 이들의 약점 최소화

**논문의 기여**
1. local-global descriptors를 통한 exhaustive matching(모든 경우를 비교하는 것)을 통해 얻은 공간 점수를 통해 이미지 쌍 간의 유사도 점수를 생성하는 새로운 place recognition system을 소개  
이러한 descriptors는 VPR-optimized 집계 기법(이 논문에서는 NetVLAD)를 사용하여 feature space 내에서 밀집 샘플링된 local patches에 대해 추출
2. 성능 향상을 위해 다른 크기의 hybrid descriptors를 생성 및 결합하는 multi-scale fusion technique 제안  
multi-scale 접근 방식으로 전환할 때 계산이 증가하는 문제를 최소화하기 위해 다양한 patch 크기에 대한 local features를 유도하기 위해 적분 feature space(적분 이미지와 유사한)를 개발  
1번과 2번 기여는 사용자에게 작업 요구 사항에 따라 유연성을 제공
3. 다른 성능 및 계산 균형을 달성하는 시스템 구성의 시연
    - performance: 엄격한 error threshold가 필요할 때 SOTA recall 성능을 달성
    - balanced: SOTA 기술과 거의 동일한 성능을 내면서 SuperGlue보다 3배 빠르고 DELG보다 28배 빠름
    - speed-focused: SOTA보다 최소한 한 차원 빠름



....


## 2. Related Workd


## 3. methodology

![alt text](./images/Fig 2.png)
> **Figure 2. 제안 알고리즘 설계도**  
> 1. Patch-NetVLAD는 NetVLAD descriptor 비교를 통해 쿼리 이미지에 대해 가장 가능성이 높은 reference match를 초기 입력으로 받음
> 2. Top-ranked 후보 이미지에 대해, 여러 scale에서 새로운 local-global patch-level descriptors를 계산
> 3. 이 descriptors를 query 및 후보 이미지 간에 기하학적 검증을 통해 local cross-matching을 수행
> 4. 이러한 매치 점수를 사용하여 초기 목록의 순서를 재정렬하여 최종 이미지 검색 결과 생성

Patch-NetVLAD
- 이미지 쌍 간의 유사성 점수를 생성하여 이미직 간의 공간적 및 외관적 일관성을 측정
- 접근 방식
    1. original NetVLAD descriptors를 사용하여 쿼리 이미지에 대해 가장 가능성이 높은 상위 k(논문에서는 k=100)을 검색
    2. NetVLAD에서 사용된 새로운 유형의 patch descriptor을 계산하고 patch-level descriptor의 local matching을 수행하여 초기 match 목록을 재정렬하고 최종 이미지 검색을 정제
- 이러한 접근 방식은 최종 이미지 검색 단계에서 recall 성능을 희생하지 않으면서 patch features를 cross matching하여 발생하는 추가 전체 계산 비용을 최소화

### 3.1 Original NetVLAD Architecture

- 이미지 분류에 사용되는 pre-trained CNN에서 추출한 중간 feature map을 집계하여 이미지를 위한 condition and viewpoint 불변 embedding을 생성하는 Vector-of-Locally-Aggregated-Descriptors(VLAD) 접근 방식을 사용
- $f_\theta : I \rightarrow \mathbb{R}^{H \times W \times D}$를 기본 아키텍처로 두고, 이미지 $I$를 제공하면 $H \times W \times D$ 차원의 feature map $F$(예: VGG의 conv5레이어)를 출력
- 이러한 $D$ 차원의 feature을 각 특징 $x_i \in \mathbb{R}^D$와 soft-assignment로 가중치가 부여된 $K$개의 학습된 클러스터 중심 간의 residual을 합산하여 $K \times D$ 차원의 행렬로 집계
- 공식적으로, $N \times D$ 차원 features에 대해, VLAD의 집계 레이어 $f_{VLAD}: \mathbb{R}^{N \times D} \rightarrow \mathbb{R}^{K \times D}$는 다음과 같이 주어진다.

$$
\displaystyle
\begin{aligned}
&f_{VLAD}(F)(j, k) = \sum_{i=1}^{N} \bar{a}_k(\text{x}_i)(x_i(j) - c_k(j))
&(1)
\end{aligned}
$$

> $x_i(j)$: $i^{th}$ descriptor의 $j^{th}$ 요소  
> $\bar{a}_k$: soft-assignment function  
> $c_k$: $k^{th}$ cluster center

- VLAD 집계 후, 결과 행렬은 projection layer $f_{proj}: \mathbb{R}^{K \times D} \rightarrow \mathbb{R}^{D_{proj}}$를 사용하여 차원 축소된 벡터로 투사된다.
- intra(column)-wise normalizeation을 적용한 후, single vector로 전개한 다음 전체를 L2-normalization한 다음 백색화(각 주성분들이 서로 독립적인 값이 되도록) 및 L2-normalization을 적용한 PCA(training set으로 훈련된)을 적용
- 이 feature-map 집계 방법을 사용하여 전체 feature map($N \ll H \times W$) 내의 local patches에 대한 descriptor을 추출하고 다양한 scale에서 query/reference image pair 간의 cross-matching하여 이미지 검색에 사용되는 최종 유사성 점수를 생성
- 이는 $N = H \times W$를 설정하고 feature map 내의 모든 descriptors를 집계하여 전역 이미지를 생성하는 NetVLAD와 대조적임

### 3.2 Patch-level Global Features

- global descriptors 내에서 조밀하게 sample된 sub-regions(patch 형태)의 global descriptor을 추출
- feature map $F \in \mathbb{R}^{H \times W \times D}$에서 stride $s_p$로 $d_x \times d_y$ 패치 세트 ${P_i, x_i, y_i}_{i=1}^{n_p}$를 추출
- 패치의 개수:

$$
\displaystyle
\begin{aligned}
& n_p = \left\lfloor \frac{H - d_y}{s_p} + 1 \right\rfloor * \left\lfloor \frac{W - d_x}{s_p} + 1 \right\rfloor , d_y, d_x, \leq H, W
& (2)
\end{aligned}
$$

> $P_i \in \mathbb{R}^{(d_x \times d_y) \times D}$: patch features 세트  
> $x_i, y_i$: 패치의 중심 위치

- 정사각형 패치가 다양한 환경에서 가장 일반화된 성능을 제공.  
추후 연구에서는 특정 상황(예: 수직과 수평 방향에 서로 다른 texture frequencies를 갖는 환경)에서 다양한 패치 형태를 고려할 수 있음
- 각 패치에서 descriptor을 추출하여 patch descriptor set $\{\text{f}_i\}_{i=1}^{n_p}$를 생성


각 패치에 대해, 우리는 이후에 패치 설명자를 추출하여 패치 설명자 집합 {fi}np i=1을 생성합니다. 여기서 fi = fproj (fVLAD (Pi)) ∈ RDproj는 관련된 패치 특징 집합에 NetVLAD 집계 및 투영 레이어를 사용합니다. 모든 실험에서 우리는 PCA를 이용하여 패치 특징에 대한 차원 축소의 정도를 변화시킴으로써 사용자 선호에 맞는 계산 시간과 이미지 검색 성능 간의 균형을 달성하는 방법을 보여줍니다(섹션 4.5 참조). 또한, 여러 스케일에서 패치를 추출하여 장소 인식 성능을 더욱 향상시킬 수 있으며, 원본 이미지 내에서 더 큰 하위 영역을 나타내는 패치 크기 조합을 사용하는 것이 검색 성능을 개선하는 것을 관찰합니다(섹션 3.5 참조). 이러한 다중 스케일 융합은 섹션 3.6에서 소개된 우리의 IntegralVLAD 공식을 사용하여 계산적으로 효율적입니다.