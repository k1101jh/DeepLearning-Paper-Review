# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- 3D Segmentation
- Gaussian Splatting
- Scene Editing

---
url:
- [paper](https://arxiv.org/abs/2312.00732)
- [project](https://ymq2017.github.io/gaussian-grouping/)
- [github](https://github.com/lkeab/gaussian-grouping)
---
- **Authors**: Mingqiao Ye, Martin Danelljan, Fisher Yu, Lei Ke
- **Venue**: ECCV 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)
  - [3.1 Preliminaries: 3D Gaussian Splatting](#31-preliminaries-3d-gaussian-splatting)
  - [3.2 3D Gaussian Grouping](#32-3d-gaussian-grouping)
  - [3.3 Gaussian Grouping for Scene Editing](#33-gaussian-grouping-for-scene-editing)
- [4. Experiments](#4-experiments)

---

## ⚡ 요약 (Summary)
- **Problem**: 기존 3D Gaussian Splatting(3DGS)은 장면의 외형 재구수에는 탁월하지만, 개별 객체에 대한 의미론적 이해가 부족하여 특정 사물을 선택하거나 편집하는 작업에 한계가 있음.
- **Idea**: 각 가우시안 입자에 고유 ID(Identity Encoding)를 부여하고, SAM(Segment Anything Model)으로부터 얻은 2D 마스크 정보를 3D 공간 일관성 정규화(3D Regularization Loss)와 함께 학습시켜 가우시안을 그룹화함.
- **Result**: 고품질 렌더링 성능을 유지하면서 실시간 3D 세그멘테이션을 달성하고, 객체별 제거, 인페인팅, 스타일 변환 등 복잡한 3D 장면 편집 작업을 효율적으로 수행함.

---

## 📖 Paper Review

## Abstract

Gaussian Splatting
- 3D 장면의 고품질 real-time novel-view 합성 달성
- 외형과 geometry modeling에만 집중되어 있고, 세밀한 객체 수준의 장면 이해는 부족함

Gaussian Grouping
- open world 3D scenes에서 모든 것을 공동으로 재구성하고 분할할 수 있게 함
- 각 가우시안에 compact Identity Encoding을 추가
    - 3D 장면에서 객체 instance 또는 stuff membership(그룹으로 묶기 위한 분류 기준. 간단히는 semantic class 정보)에 따라 가우시안을 그룹화할 수 있도록 함
- 미분 가능한 렌더링동안 identity encoding을 감독
    - Segment Anything Model(SAM)의 2D 마스크 예측과 도입된 3D 공간 일관성 규제를 활용
- NeRF 표현과 비교했을 때, 이산적이고 그룹화된 3D 가우시안이 높은 시각적 품질, 세밀한 수준, 효율성을 갖춘 상태에서 3D 내의 모든 것을 재구성, 분할, 편집할 수 있음을 보여줌
- Gaussian Grouping을 기반으로 local Gaussian Editing 방식을 제안하여 3D 객체 제거, inpainting, 색칠, style transfer 및 장면 재구성 등 다양한 장면 편집 응용에서 효과적임을 보임

## 1. Introduction

논문의 목표
- posed RGB 이미지 집합이 주어졌을 때, 효과적인 3D 표현을 배우는 것
- 3D scene의 모든 것을 공동으로 재구성하고 segment
    - 다양한 downstream 장면 편집 앱을 쉽게 지원 가능

3D scene understanding
- SAM 및 그 변형들에 의해 2D scene understanding의 3D로의 확장이 제한됨
    - 3D scene dataset의 제작 과정의 가용성과 노동 집약적인 과정 때문
- 기존 방법은 수작업으로 라벨링된 데이터셋에 의존
    - 고비용
    - 범위가 제한적일 수 있음
    - 입력으로 정확하게 스캔된 point cloud 필요

기존 NeRF 기반 방법들
- multi-view 캡처를 통해 2D mask를 가져오거나, Neural field rendering을 통해 CLIP/DINO feature 추출
    - NeRF의 암묵적이고 연속적인 표현 때문에 이러한 방법은 비용이 많이 드는 무작위 샘플링이 필요함
- MLP와 같은 학습된 신경망은 3D 장면의 각 부분이나 모듈을 쉽게 분해할 수 없음
    - downstream local 편집 작업에 직접 적용하기 어려움
- 일부 방법은 NeRF나 stable-diffusion 방법과 SAM mask를 결합하지만, 단일 객체에만 집중

3D Gaussian Splatting
- 높은 훈련 및 렌더링 효율
- 인상적인 재구성 품질
- 객체 인스턴스나 semantic 이해를 모델링하지 않음
- fine-grained scene 이해를 달성하기 위해 3D 환경을 구성하는 개별 객체와 요소들을 포함하도록 확장
    - Gaussian Grouping 제안
    - 3D 장면에서 무엇이든 재구성하고 segmenting하기 위한 이산적인 그룹화된 3D 표현 학습
        - multi-view 캡처와 SAM에 의해 자동으로 생성된 mask를 입력
    - SAM의 강력한 zero-shot 2D scene 이해 능력을 계승하며, 일관된 novel view 합성과 segmentation을 생성하여 3D로 확장

논문의 방법
- 3D 장면의 각 Gaussian의 identity를 그룹화하여 포착
    - Identity Encoding을 각 가우시안에 추가
        - compact하고 저차원인 학습 가능한 임베딩
        - 미분 가능한 가우시안 렌더링을 통해 학습됨
            - 2D에서의 segmentation 감독을 활용하기 위함
    - 다양한 가우시안의 encoding 벡터가 2D 렌더링 뷰에 투영됨
    - 2D로 렌더링된 identity feature을 활용해 추가적인 linear layer을 통해 각 2D 위치에서 투영된 embedding을 분류하여 identity classification
- 그룹화 정확도 향상을 위해 identity classification을 위한 standard cross-entropy loss 외에도 un-supervised 3D Regularization loss 도입
    - 상위 K개의 가장 가까운 3D 가우시안의 Identity Encoding이 feature space에서 서로 가깝도록 강제
    - 3D 객체 내부 혹은 심하게 가려진 가우시안도 학습 중에 충분히 감독될 수 있음을 확인

Gaussian Grouping의 장점
- 높은 시각적 품질
- 빠른 학습 속도
- Dense 3D segmentation을 통해 후속 장면 편집 응용이 가능해짐
    - 각 3D gaussian 그룹이 독립적으로 동작하여 구성 요소를 완전히 분리하거나 나눌 수 있음
    - 개별 구성 요소를 식별, 조작, 교체해야 하는 시나리오에서 중요

논문의 기여
1. Gaussian Grouping 제안
    - SAM의 knowledge를 3D 장면의 zero-shot segmentation으로 확장하는 최초의 3D Gaussian Splatting 기반 segmentation framework
    - 3D mask label 없이 가능
2. Gaussian Grouping은 제안된 Local Gaussian Editing 방식으로 다양한 후속 작업 지원
    - 개별 구성 요소를 식별, 조작, 교체해도 전체 장면 구조에 영향을 주지 않음
    - 여러 3D scene 편집 사례를 보임
3. 학습 및 렌더링이 빠르며, real-time 운영 요구 사항을 충족할 수 있음


## 2. Related Works

**SAM in 3D**
Segment Anything Model(SAM) [17]은 zero-shot 2D segmentation을 위한 foundation vision model로 출시됨
- 여러 연구에서 SAM의 2D mask를 NeRF[4, 5] 또는 3D point cloud를 통해 3D segmentation으로 확장
    - NeRF 기반 접근법은 3D 장면의 단일 객체나 일부 객체에만 집중
- Gaussian Grouping은 자동 everything mode에서 작동하여 전체 장면의 각 인스턴스 / 물건을 총체적으로 이해할 수 있음

## 3. Method

목표
- 외형과 geometry 모델링을 넘어, 장면의 모든 객체와 배경 요소의 정체성을 포착하는 표현력 있는 3D 장면 표현 구축

Gaussian Grouping
1. scene의 각 3D 부분을 외형, geometry 및 mask identities와 하께 모델링
2. 3D scene을 개별 그룹으로 완전히 분해하여 편집 가능하게 다양한 객체 인스턴스를 나타냄
3. 원래의 3D 재구성 품질을 해치지 않으면서 빠른 학습 및 렌더링 가능

Gaussian Grouping
- SAM의 dense 2D mask proposal을 효과적으로 활용
    - radiance fields rendering을 통해 3D 장면 내 모든 객체를 분할하는 데 확장


### 3.1 Preliminaries: 3D Gaussian Splatting


### 3.2 3D Gaussian Grouping


![alt text](./images/Fig%202.png)
> **Figure 2**  
> Gaussian Grouping은 세 가지 주요 단계로 구성됨  
> (a) 각 view별로 독립적으로 everything mode에서 SAM을 사용해서 mask를 자동으로 생성하여 입력 준비  
> (b) 학습 view 전반에 걸쳐 일관된 mask ID를 얻기 위해 범용 temporal propagation model[7]을 사용해서 mask label을 연결하고 일관된 multi-view segmentation을 생성  
> (c) 준비된 학습 입력을 사용하여, 미분 가능한 렌더링을 통해 group Identity Encoding을 포함한 3D Gaussian의 모든 속성을 공동으로 학습  
> encoding은 일관된 segmentaion view를 활용한 2D Identity loss와 3D Regularization loss로 감독됨  
> 입력 view에서 프레임 간 객체 아이디를 나타내기 위해 색상 사용
> 다른 Gaussian paramter와 density control 부분의 렌더링 과정은 [15]에서 상속됨


![alt text](./images/Algorithm%201.png)

(a) 2D Image and Mask Input
- SAM을 사용하여 multi-view collection의 각 이미지에 대한 mask 자동 생성(그림 2(a) 참고)
    - 2D mask는 이미지별로 개별 생성됨
- 각 2D 마스크에 3D 장면에서 고유한 ID를 할당하기 위해, 서로 다른 뷰에서 동일한 identity를 가진 마스크를 연관시키고 3D 장면에서 instance/객체의 총 개수 $K$를 얻어야 함

(b) view 간 Identity Consistency
- 학습 중 cost-based linear assignment를 사용하는 대신, 3D 장면의 multi-view 이미지를 점진적으로 변화하는 뷰를 가진 video sequence로 간주
- view간 2D mask 일관성을 달성하기 위해, 잘 학습된 zero-shot tracker을 사용해서 mask를 propagate하고 연관시킴
    - 이를 통해 3D 장면에서 mask identities의 총 개수를 제공받음
    - 연관된 2D mask label 시각화(그림 2(b) 참고)
- 비용 기반 linear assignment와 비교
    - 학습 난이도 단순화
    - 각 rendering iteration에서 matching 관계를 반복 계산하지 않아도 됨
    - 60배 이상의 속도 향상을 제공
- SAM으로 생성된 dense & overlapping mask 환경에서도 cost-based linear assignment보다 나은 성능 달성
- 2D associated masks가 video에서 명백한 오류를 포하하고 있음에도, 3D mask association의 강건함을 보임(그림 5 참고)

(c) 3D Gaussian Rendering and Grouping
- 3D 일관 mask identities를 생성하기 위해 동일한 instance / stuff에 속하는 3D 가우시안을 group화
- 가우시안에 Identity Encoding 파라미터 추가
    - 길이 16의 학습 가능한 벡터
    - 장면에서 서로 다른 객체 / 부분을 계산 효율적으로 구분하는데 충분함
    - 학습 중 각 가우시안의 색상을 나타내는 SH 계수와 유사하게 Identity Encoding 벡터를 최적화
        - 장면의 instance ID를 표현하기 위해
- 장면의 view 의존적 외형 modeling과 달리, instance ID는 다양한 렌더링 view에서도 일관됨
    - identity encoding의 SH degree를 0으로 설정하여 Direct-current 성분만 모델링
- NeRF 기반 방법에서 추가적인 semantic MLP 레이어를 설계하는 것과 달리, Identity encoding은 각 가우시안에 대한 학습 가능한 속성으로서 3D 장면을 그룹화

Identity Encoding 최적화
- 2D 이미지로 미분 가능하게 렌더링
- 미분 가능한 3D gaussian renderer[15] 사용
- 렌더링 과정을 SH 계수 최적화와 유사하게 처리
- 3D gaussian splatting은 neural point-based $\alpha'$-rendering을 채택
    - 각 가우시안의 영향 가중치 $\alpha'$는 각 pixel에 대해 2D에서 평가될 수 있음
- 단일 pixel 위치에 대한 모든 gaussian의 영향은 gaussian을 깊이 순서로 정렬하고 pixel에 겹치는 N개의 정렬된 point를 블렌딩해서 계산됨[15]
- 각 pixel에 대해 최종 렌더링된 2D mask identity feature $E_{id}$:  
    각 gaussian의 길이 16인 identity encoding $e_i$에 대한 weighted sum
    - 해당 pixel에서 gaussian의 영향 계수는 $\alpha_i'$로 가중됨
- 학습된 per-point 불투명도 $\alpha_i$와 곱해진 공분산 $\sum^{2D}$를 갖는 2D 가우시안을 측정하여 $\alpha_i'$를 계산
$$
\displaystyle
\sum^{2D} = J W \sum^{3D} W^T J^T
\tag{2}
$$
> $\sum^{3D}$: 3D covariance matrix  
> $\sum^{2D}$: splatted 2D version  
> $J$: 3D-2D 투영의 affine 근사의 Jacobian  
> $W$: world-to-camera 변환 행렬

(d) Grouping Loss
- 2D instance label을 연관시킨 후, 3D scene에 총 $K$개의 mask가 있다고 가정
    - 각 3D Gaussian을 instance / stuff mask identities로 그룹화하기 위해 grouping loss $\mathcal{L}_{id}$ 설계
    - 두 가지 구성 요소
        1. 2D Identity Loss
            - mask identity label이 2D에 있으므로 3D Gaussian의 Identity Encoding $e_i$를 직접 감독하지 않음
            - rendered 2D features $E_{id}$ 가 입력으로 주어지면, linear layer $f$를 추가하여 feature dimension을 $K$로 되돌림($K$: 3D scene 내 mask 개수)
            - $softmax (f(E_{id}))$
        2. 3D Regularization Loss
            - gaussian grouping 정확도를 높이기 위해 indirect 2D supervision을 위한 표준 cross-entropy loss 외에도 unsupervised 3D Regularization Loss를 도입하여 Identity Encoding 학습을 직접 정규화
            - 3D 정규화 loss는 3D 공간 일관성을 활용하여 top k-nearest 3D Gaussian들의 Identity Encodings가 feature 거리에서 가깝도록 강제
                - 3D 객체 내부의 가우시안이나 point-based 렌더링 중 심하게 가려진 가우시안들이 더 충분히 감독될 수 있도록 함
            - 식 3에서는 $F$를 linear layer $f$ 이후에 결합된 softmax 연산으로 나타냄(2D Identity loss 계산에 공유됨)
            - $m$개의 샘플링 point에서 KL divergence loss를 다음과 같이 형식화
            $$
            \displaystyle
            \mathcal{L}_{3d} = \frac{1}{m} \sum_{j=1}^m D_{kl} (P || Q) = \frac{1}{mk} \sum_{j=1}^m \sum_{i=1}^k F(e_j) \log \big(\frac{F(e_j)}{F(e'_i)}\big)
            \tag{3}
            $$
            - $P$: 3D gaussian의 sample된 Identity Encoding $e$를 포함함
            - 집합 $Q = {e'_1, e'_2, ..., e'_k}$는 3D euclidean 공간에서의 $k$개의 nearest neighbor로 구성됨
            - 단순화를 위해 linear layer $f$ 뒤에 결합된 softmax 연산은 생략
    - 전체 loss $\mathcal{L}_{render}$:
    $$
    \displaystyle
    \mathcal{L}_{render} = \mathcal{L}_{rec} + \mathcal{L}_{id} = \mathcal{L}_{rec} + \lambda_{2d} \mathcal{L}_{2d} + \lambda_{3d}\mathcal{L}_{3d}
    \tag{4}
    $$

### 3.3 Gaussian Grouping for Scene Editing

3D gaussian field 훈련 및 그룹화 후, 전체 3D scene을 그룹화된 3D Gaussian 집합으로 나타냄
- 다양한 downstream local scene editing 작업을 수행하기 위해 효율적인 local gaussian 편집 제안
- 모든 3D Gaussian을 finetuning하는 대신, 대부분의 잘 훈련된 gaussian의 속성을 고정하고 편집 대상과 관련된 기존 or 새로 추가된 일부 3D 가우시안만 조정
- 3D 객체 제거
    - 편집 대상의 3D 가우시안을 단순히 제거
- 3D 장면 재구성
    - 두 가우시안 그룹 간 3D 위치 교환
- 3D 객체 inpainting
    - 관련 3D 가우시안 제거 후 약간의 새로운 gaussian을 추가하고 렌더링 시 LAMA[46]의 2D inpalinting 결과를 통해 감독
- 3D 객체 colorization
    - 학습된 3D scene geometry를 유지하기 위해 해당 가우시안 그룹의 SH 매개변수만 조정
- 3D 객체 style transfer
    - 3D 위치와 크기를 추가로 조정
- local gaussian 편집 방식은 시간 효율적
- 세밀한 mask modeling 덕분에 간섭 없이 여러 local 편집을 동시에 지원하거나 전체 global 3D 장면 표현을 재학습할 필요 없음


## 4. Experiments

### 4.1 Dataset and Experiment Setup



## 6. Appendix

### 6.1 Supplementary Experiments

LERF-Mask 데이터셋에서 SA3D[4]와 open world 3D segmentation의 분할 결과 비교
- Gaussian Grouping은 segmentation 효율성에서 큰 장점을 보임
    - 3D 장면의 모든 객체를 9분몬에 jointly segment
    - SA3D는 3D voxel grid에서 inverse rendering 설계 때문에 객체마다 35분 소요
    