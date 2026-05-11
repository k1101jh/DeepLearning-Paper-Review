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
    - $k$개의 neural gaussian 속성은 anchor feature $\hat{f}_v$ 카메라와 anchor point 사이의 상대 시야 거리 $\delta_{vc}$ 및 방향 $\vec{\rm{d}}_{vc}$를 통해 개별 MLP로 직접 디코딩됨
    - 속성은 one-pass로 디코딩됨
    - 예시) anchor point에서 생성된 neural gaussian의 불투명도 값:
    $$
    \displaystyle
    \{\alpha_0, \dots, \alpha_{k - 1}\} = F_\alpha (\hat{f}_v, \delta_{vc}, \vec{\rm{d}}_{vc})
    \tag{9}
    $$
- neural gaussian 속성의 예측이 on-the-fly로 수행됨(즉시 수행됨)
    - frustum 내에서 보이는 anchor만 neural Gaussian을 생성하도록 활성화됨
    - 불투명 값이 미리 정의된 임계값 $\tau_\alpha$보다 큰 neural gaussian만 유지
        - 계산 부하를 줄이고 높은 렌더링 속도 유지에 도움

### 3.3 Anchor Points Refinement

**Growing Operation**

![alt text](./images/Fig%203.png)
> **Figure 3. Growing operation**  
> neural Gaussian의 gradient에 의해 guid되는 anchor growing policy 개발  
> 왼쪽에서 오른쪽으로 neural Gaussian을 공간적으로 크기 ${\epsilon_g^{(m)}}$인 multi-resolution voxels ($m \in \{1, 2, 3\}$)으로 양자화  
> 새로운 anchor는 집계된 gradient가 $\{\tau_g^{(m)}\}$보다 큰 voxel에 추가됨

- Neural Gaussian은 SfM point에서 초기화된 anchor point와 밀접하게 연관됨
- 각 anchor point의 모델링 능력은 국소 영역에 지한됨
    - 초기 anchor point 배치가 중요
- **Error-based anchor growing policy(Fig 3 참조)**
    - neural gaussian이 유의미하다고 판단되는 곳에 새로운 anchor을 성장시킴
    - 방법
        1. neural Gaussian을 크기 $\epsilon_g$의 복셀로 공간적으로 양자화
        2. 각 복셀에 대해, $N$번의 training iteration동안 포함된 neural Gaussian의 평균 기울기 $\nabla_g$를 계산
        3. $\nabla_g > \tau_g$인 voxel은 유의미한 것으로 간주. $\tau_g$는 임계값
            - 해당 voxel 중심에 anchor point가 없으면 새 anchor point 배치
    - 공간을 multi-resolution voxel grid로 양자화
        - 다양한 수준에서 새로운 anchor를 추가할 수 있도록
        $$
        \displaystyle
        \epsilon_g^{(m)} = \epsilon_g / 4^{m - 1}, \tau_g^{(m)} = \tau_g * 2 ^{m - 1}
        \tag{10}
        $$
        > $m$: 양자화 수준
        - 새로운 anchor의 추가를 더 규제하기 위해 이러한 후보들에 무작위 제거 적용
            - anchor의 급속한 확장을 효과적으로 억제
**Pruining Operation**
- 사소한 anchor 제거를 위해, $N$회의 training iteration동안 해당 anchor와 관련된 neural Gaussian의 불투명 값들을 누적
- anchor가 만족스러운 수준의 불투명을 갖는 neural Gaussian을 생성하지 못하면 해당 anchor을 장면에서 제거
(즉, anchor 내 가우시안들이 일정 수준 이상 불투명도를 가져야 함)

## 3.4 Losses Design
- 렌더링된 픽셀 색상의 L1 loss에 대해 학습 가능한 매개변수와 MLP 최적화
    - $\mathcal{L}_{SSIM}$ 및 volume 정규화[28] $\mathcal{L}_{vol}$을 포함
    - 전체 supervision:
    $$
    \displaystyle
    \mathcal{L} = \mathcal{L}_1 + \lambda_{\rm{SSIM}} \mathcal{L}_{\rm{SSIM}} + \lambda_{\rm{vol}} \mathcal{L}_{\rm{vol}}
    \tag{11}
    $$
    - volume 정규화 $\mahtcal{L}_{\rm{vol}}$:
    $$
    \displaystyle
    \mathcal{L}_{\rm{vol}} = \Sum_{i=1}^{N_{ng}} \rm{Prod}(s_i)
    \tag{12}
    $$
    > $N_{\rm{ng}}$: 장면 내 neural Gaussian 수  
    > $\rm{Prod}(\cdot)$: vector 곱  
    > $s_i$: 각 가우시안의 scale
    - 부피 정규화 항은 neural Gaussian이 최소한의 겹침으로 작게 유지되도록 장려

## 4. Experiments

### 4.1 Experimental Setup

**Dataset and Metrics**

Public dataset에서 27개 scene 대상으로 포괄적 평가 수행
- 3D-GS[22]에서 테스트된 모든 사용 가능한 scene에서 테스트
    - Mip-NeRF360[4]의 7개 scene
    - Tanks&Temples[23]에서 2개 scene
    - DeepBlending[18]에서 2개 scene
    - 합성 Blender 데이터셋[30]
- view 적응 렌더링의 장점을 보여주기 위해 다양한 LOD에서 캡처된 콘텐츠를 포함한 데이터셋에서 평가 수행
    - BungeeNeRF[49]의 6개 scene
        - multi-scale outdoor 관측
    - VR-NeRF[51]의 2개 scene
        - 복잡한 실내 환경 캡처
- 평가 지표
    - PSNR
    - SSIM
    - LPIPS
    - 저장 용량(MB)
    - 렌더링 속도(FPS)

**Baseline and Implementation**

- 3D-GS[22]
    - 주요 baseline
- 3D-GS 및 제안 방법 모두 30k iteration으로 학습
- Mip NeRF360, iNGP, Plenoxels 결과도 기록
- 모든 실험에서 $k=10$으로 설정
- 모든 MLP
    - ReLU activation을 갖는 2-layer MLP
    - hidden unit의 차원: 32
- anchor point refinement
    - $N = 100$ iteration동안 gradient 평균화
    - $\tau_g = 64 \epsilon$ 사용
    - 복잡한 장면 & texture가 없는 영역이 우세할 경우 $\tau_g = 16\epsilon$ 사용
- anchor의 neural gaussian 누적 불투명도가 각 refinement round에서 0.5 미만이면 해당 anchor 제거
- $lambda_{\rm{SSIM}}$: 0.2
- $lambda_{\rm{vol}}$: 0.001

### 4.2 Results Analysis

![alt text](./images/Fig%204.png)

> **Figure 4. 다양한 데이터셋에서 Scaffold-GS와 3D-GS의 정성적 비교**  
> - 제안 방법은 이러한 장면에서 지속적으로 3D-GS보다 뛰어남
> - 도전적인 시나리오에서 명백한 장점을 보임
>   - 예:
>       - 얇은 기하 구조 & 세밀한 디테일  
>       (MIP360-ROOM(a), MIP360-COUNTER(a))
>       - 텍스처 없는 영역
>       (DB-DRJOHNSON, DB-PLAYROOM)
>       - 조명 효과  
>       (MIP360-COUNTER(b), DB-DRJOHNSON)
>       - 충분한 관찰  
>       (TANDT-TRAIN, VR-KITCHEN)
> - VR-APARTMENT 등에서 제안 방법이 다양한 scale과 viewing distance에서 contents를 표현하는데 우수함

- 다양한 데이터셋에서 수행
    - 합성 object-level scene
    - 실내 및 실외
    - 대규모 도시 장면 & 풍경
- 질감이 없는 영역, 관찰 부족, 세밀한 디테일, view-dependnt light effects와 같은 경우에서 다양한 개선 사항 관찰

**Comparisons**

- 비교 대상
    - 3D-GS[22]
    - Mip-NeRF360[4]
    - iNGP[31]
    - Plenoxels[13]
- MipNeRF360 데이터셋에서 SOTA 알고리즘과 유사한 결과 달성(표 1 참조)
    ![alt text](./images/Table%201.png)
    - Tanks&Temples, DeepBlending에서 SOTA 능가
        - 이 데이터셋은 조명의 변화, 질감이 없는 영역 및 반사 등의 존재로 더 도전적인 환경 포착
- 효율성(표 2 참조)
    ![alt text](./images/Table%202.png)
    - 제안 방법과 3D-GS의 렌더링 속도와 저장 용량 평가
    - 저장 공간을 적게 사용하면서 실시간 렌더링 달성
        - 렌더링 품질과 속도를 희생하지 않고 3D-GS보다 컴팩트한 모델임
    - 이전의 grid 기반 방법과 유사하게, 3D-GS보다 더 빠르게 수렴함
- 합성 Blender 데이터셋에서 실험(표 3 참조)
    ![alt text](./images/Table%203.png)
    - 좋은 초기 SfM 점 set이 쉽게 제공되지 않음
        - 100k grid point에서 시작해서 anchor refinement 연산을 통해 point grow 및 prune
    - 30k iteration 후, 남은 점들을 초기화된 anchor로 사용하고 framework 다시 실행
    - 제안 방법은 신뢰할 수 있는 기하 구조와 texture details로 더 나은 시각적 품질 달성 가능함

**Multi-scale Scene Contents**

![alt text](./images/Fig%205.png)
> **Figure 5. multi-scale scenes(w/zoom-in cases)에서의 비교**  
> - BungeeNeRF의 AMSTERDAM scene에서 unseen closer scale에서 렌더링 결과를 보임
> - 제안 방법은 refined neural Gaussian 특성을 사용하여 새로운 vieweing distances로 부드럽게 외삽함
> - 고정된 gaussian scaling 값으로 발생하는 원본 3D-GS의 바늘 모양 artifact 수정

- multi-scale scene detail을 처리하는 모델의 능력 검증
    - 데이터셋:
        - BungeeNeRF
        - VR-NeRF
- 3D-GS와 비교해서 모델을 저장하는 데 적은 저장 공간으로도 우수한 품질 달성
    - 3D-GS
        - 종종 눈에 띄는 흐림과 바늘 모양의 artifact 발생
        - Gaussian 속성이 multi-scale training view에 과적합되도록 최적화되었기 때문일 수 있음
            - 관찰 거리에 따라 작동하는 과도한 가우시안 생성
            - novel view를 합성할 때 각도와 거리에 대한 추론 능력이 부족
            - 쉽게 모호합과 불확실성을 초래할 수 있음
    - 제안 방법:
        - 장면 내 다양한 detail 수준을 다루는데 뛰어남
        - 지역 구조를 효율적으로 compact한 neural feature로 인코딩하여 렌더링 품질과 수렴 속도 모두 향상

**Feature Analysis(Fig 6 참조)**

![alt text](./images/Fig%206.png)
> **Figure 6. Anchor feature clustering**
> - K-means를 사용하여 anchor features(DB-PLAYROOM)을 3개의 cluster로 clustering
> - clustered feature은 scene 구성 요소(예: 난간, 유모차 등)을 명확하게 식별 가능
> - 벽과 바닥의 anchor도 각각 함께 그룹화됨
>   - 제안 방식이 3D-GS 모델의 해석 가능성을 향상시킴
>   - 재사용 가능한 feature을 활용하여 더 큰 장면에서도 확장될 가능성이 있음

- 클러스터링된 패턴은 압축된 anchor feature space가 유사한 시각적 속성과 기하학적 구조를 갖는 영역을 능숙하게 포착함
    - encoding된 feature space에서의 근접성으로 증명됨

**View Adaptability(Fig 7 참조)**

![alt text](./images/Fig%207.png)

- 동일한 가우시안을 서로 다른 위치에서 관찰할 때 속성 값이 어떻게 변하는지 탐구
    - 서로 다른 관찰 위치에서 속성 강도의 다양한 분포를 보임
    - 일정한 수준의 local 연속성을 유지  
    -> 3D-GS와 비교해서 우수한 시점 적응성을 갖는다

**Selection Process by Opacity**

![alt text](./images/Fig%208.png)





![alt text](./images/Fig%209.png)

### 4.3 Ablation Studies

![alt text](./images/Table%204.png)

![alt text](./images/Table%205.png)

### 4.4 Discussions and Limitations