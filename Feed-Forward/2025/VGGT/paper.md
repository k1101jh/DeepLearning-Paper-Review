# VGGT: Visual Geometry Grounded Transformer

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Feed Forward
- 3D Reconstruction
---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)(CVPR 2025)
- [github](https://github.com/facebookresearch/vggt)

---
- **Authors**: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
- **Venue**: CVPR 2025

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

![alt text](./images/Fig%201.png)
> **Figure 1**
> - VGGT는 방대한 3D annotation data로 학습된 3D-inductive biases가 최소화된 대형 feed-forward transformer
> - 최대 수백 장의 이미지를 입력받아 1초 이내에 모든 이미지에 대한 카메라, point map, depth map, point track을 예측 가능
> - 추가 처리 없이 최적화 기반 대안보다 뛰어난 성능을 보이는 경우가 많음

**VGGT**
- Feed-forward neural network
- 하나의 view, 몇 개의 view, 수백 개의 view에서 장면의 모든 주요 3D 속성들을 직접 추론
    - 카메라 파라미터
    - point map
    - depth map
    - 3D point track
- 간단하고효율적
- 1초 이하로 이미지 재구성 가능
- visual geometry 최적화 기법을 사용한 후 post-processing이 필요한 기존 방법보다 뛰어남
- 사전 학습된 VGGT를 feature backbone으로 사용하는 경우
    - 후속 작업 성능이 크게 향상됨
        - 비강체 point tracking
        - feed-forward novel view 합성

## 1. Introduction

3D 재구성
- 전통적인 방법
    - 반복 최적화 기법을 활용한 visual-geometry 방법
- 머신러닝
    - 기하학만으로 해결할 수 없는 작업 보완
        - feature matching
        - monocular depth 예측
- VGGSfM
    - 미분 가능한 BA를 통해 머신러닝과 visual geometry를 end-to-end로 결합
    - visual geometry가 3D 재구성에서 중요한 역할
        - 복잡성과 계산 비용을 닐림
- DUSt3R & MASt3R
    - 기하학적 후처리를 거의 배제하고 신경망으로 직접 해결할 수 있음
    - 한 번에 두 개의 이미지만 처리 가능
    - 더 많은 이미지 재구성을 위해 후처리를 통해 재구성 결과를 합쳐야 함

**VGGT**
- 1장, 몇장, 수백 장의 입력 장면에서 3D 재구성 수행
- 카메라 파라미터, depth map, point map, 3D point track을 포함한 전체 3D 속성 예측
- 단일 forward 패스로 몇 초만에 수행
- 추가 처리 없이도 최적화 기반 방법보다 나은 성능을 보이기도 함
- large tranformer 기반
    - 3D 나 기타 귀납적 편향은 없음
    - 3D 주석이 있는 공개 데이터셋에서 학습
- 다용도 백본
- 공유 백본을 사용하여 관심 있는 모든 3D 데이터를 합께 예측
    - 상호 관련된 3D 속성들을 예측하도록 학습
    - 잠재적 중복에도 전체 정확도를 높일 수 있음
    - 추론 중에 별도로 예측된 깊이와 카메라 매개변수로부터 point map을 유도할 수 있음
    - 전용 point map 헤드를 직접 사용하는 것보다 더 나은 정확도를 얻을 수 있음


**요약**
1. VGGT라는 large feed-forward transformer 소개
    - 1~수백 개의 장면 이미지로부터 카메라 내부/외부 파라미터, point map, depth map, 3D point track 예측 가능
2. 후처리 최적화 기법을 사용하는 최신 방법보다 경쟁력이 높거나 일반적으로 더 좋음
3. BA 후처리와 함께 사용될 때 3D 작업의 일부에 특화된 방법들과 비교해도 모든 분야에서 최고 수준의 결과 달성
    - 종종 품질을 크게 개선

## 2. Related Work


## 3. Method

![alt text](./images/Fig%202.png)
> **Figure 2. 아키텍처 개요**
> - DINO를 사용해서 입력 이미지를 token으로 패치
> - 카메라 예측을 위해 카메라 토큰을 추가
> - frame-wise & global self attention layer을 번갈아 사용
> - 카메라 head는 카메라 extrainsic & intrinsic에 대한 최종 예측 수행
> - DPT head는 모든 dense output에 대해 예측 수행

아키텍처 개요. 본 모델은 먼저 DINO를 사용하여 입력 이미지를 토큰으로 패치하고, 카메라 예측을 위해 카메라 토큰을 추가합니다. 그런 다음 프레임별 셀프 어텐션 레이어와 전역 셀프 어텐션 레이어를 번갈아 사용합니다. 카메라 헤드는 카메라 외적 및 내적 요소에 대한 최종 예측을 수행하고, DPT[87] 헤드는 모든 밀집 출력에 대해 예측을 수행합니다.

### 3.1. Problem definition and notation

- 입력: N개의 RGB 이미지 $I_i \in \mathbb{R}^{3 \times H \times W}$로 구성된 sequence $(I_i)_{i=1}^{N}$
- 이 sequence를 각 프레임마다 하나씩 대응되는 3D annotation set으로 매핑
    - 프레임마다:
    $$
    \displaystyle
    f ((I_i)_{i=1}^N) = (g_i, D_i, P_i, T_i)_{i=1}^N
    \tag{1}
    $$
    - transformer은 각 이미지 $I_i$를 따라 다음을 매핑
        - 카메라 파라미터 $g_i \in \mathbb{R}^9$
            - [125]의 parameterization 사용
            - $g = [q, t, f]$로 설정
            - q: 쿼터니언
            - t: 이동 벡터
            - f: 시야각
            - 카메라 주점이 이미지 중심에 있다고 가정
        - depth map $D_i \in \mathbb{R}^{H \times W}
        - point map $P_i \in \mathbb{R}^{3 \times H \times W}
        - grid $T_i \in \mathbb{R}^{C \times H \times W}
            - point tracking을 위함
- depth map $D_i$는 각 픽셀 위치 $y \in \mathcal{I}(I_i)$를 i번째 카메라에서 관찰된 해당 깊이 값 $D_i(y) \in \mathbb{R}^+$와 연결
- point map $P_i$는 각 픽셀을 해당 3D 장면 포인트 $P_i(y) \in mathbb{R}^3$과 연결
    - point map은 시점에 불변
    - 첫 번째 카메라의 좌표계에서 정의됨
        - 이를 world 좌표계로 사용
- keypoint tracking
    - track-any-point 방법을 따름
    - query image $I_q$에서 고정된 query image point $y_q$가 주어지면, 모든 이미지 $I_i$에서 해당 2D 점 $y_i \in mathbb{R}^2$로 이루어진 트랙 $\mathcal{T}^*(y_q) = (y_i)_{i=1}^N$을 출력
    - 트랜스포머 f는 track을 직접 출력하는 게 아니라 추척에 사용되는 feature $T_i$를 출력

**Order of Predictions**

- 입력 sequence에서 이미지의 순서는 임의적
- 첫 번째 이미지는 기준 프레임으로 선택됨
- 네트워크는 첫 프레임을 제외하고 순열에 대해 동등하도록 설계됨

**Over-complete Predictions**
- VGGT가 예측하는 모든 값이 독립적이지는 않음
- 불변 point map $P$로부터 카메라 매개변수 g를 추론 가능
    - PnP 문제를 풀어서 알 수 있음
- depth maps는 point map과 카메라 파라미터에서 추론될 수 있음
    - 훈련 중에 VGGT에게 이 모든 값을 명시적으로 예측하게 하면, 이 값들이 closed-form 관계로 연결되어 있어도 성능이 크게 향상됨
    - 추론 시에는 독립적으로 추정된 depth maps와 카메라 파라미터를 결합하는 것이 specialized point map branch를 직접 사용하는 것보다 정확한 3D point를 만들어 냄

### 3.2 Feature Backbone

- 최소한의 3D 귀납 편향을 갖는 간단한 아키텍처 설계
- 모델이 충분한 양의 3D 주석 데이터를 학습하도록 함
- 모델 f를 large transformer로 구현
- 각 입력 이미지 I는 먼저 DINO를 통해 K개의 token $t^I \in mathbb{R}^{K \times C}로 패치화
- 모든 frame에서 추출된 이미지 토큰들의 결합 set $t^I = \cup_{i=1}^N \{t_i^I\}$는 이후 메인 네트워크 구조를 통해 처리됨
- frame-wise&global self-attention layer가 번갈아 가며 적용됨


**Alternating-Attention**
- Alternating-Attention(AA)를 도입해서 transformer 설계를 약간 조정
    - transformer가 각 프레임 안에서 집중했다가 전역적으로 번갈아 가며 집중 가능
    - frame-wise self-attention은 각 프레임 안의 토큰 $t_k^I$에만 집중
    - global self-attention은 모든 프레임에서 토큰 $t^I$를 함께 봄
    - 이렇게 하는 경우, 다른 이미지 간 정보를 통합하는 것과 각 이미지 안에서 token의 activations를 정규화하는 것 사이의 균형을 맞출 수 있음
    - $L=24$층의 global & frame-wise attention을 사용
- 아키텍처에 cross-attention layer가 없고 self-attention layer만 존재

### 3.3 Prediction heads

- 각 입력 이미지 $I_i$에 대해 해당 이미지 토큰 $t_i^I$에 추가 카메라 토큰 $t_i^g \in \mathbb{R}^{1 \times C'}$와 네 개의 register token $t_i^R \in \mathbb{R}^{4 \times C'}$를 더함
- $(t_i^I, t_i^g, t_i^R j)_{i=1}^N$을 이어 붙여 AA transformer에 입력하여 출력 토큰 $(\hat{t}_i^I, \hat{t}_i^g, \hat{t}_i^R)_{i=1}^N$을 얻음
- 첫 번째 frame의 카메라 토큰과 register token은 다른 학습 가능한 토큰으로 설정됨
    - 나머지 모든 프레임의 토큰도 학습 가능
    - 이는 모델이 첫 번째 프레임과 나머지 프레임을 구분할 수 있게 함
    - 3D 예측을 첫 번째 카메라의 좌표 프레임에서 표현 가능
    - refined camera와 register token은 프레임별로 특화됨
        - AA transformer가 frame-wise self-attention layer을 포함하고 있기 때문
        - transformer가 카메라와 register token을 같은 이미지의 대응 토큰과 맞출 수 있음
- 일반적인 방식대로, 출력 레지스터 토큰 $\hat{t}_i^R$은 버리고 $\hat{t}_i^I, \hat{t}_i^g$만 예측에 사용

**Coordinate Frame**

- 첫 번째 카메라 좌표계에서 카메라, point map, depth map 예측
- 첫 번째 카메라에 대한 extrinsic output은 항등행렬

**Camera Predictions**

- 카메라 파라미터 $(\hat{g}^i)_{i=1}^N$은 출력 카메라 토큰 $(\hat{t}_i^g)_{i=1}^N$에서 예측되며, 네 개의 추가 self-attention layer와 이어지는 linear layer을 사용

**Dense Predictions**

- 출력 이미지 토큰 $\hat{t}_i^I$는 dense output, depth map $D_i$, point maps $P_i$, tracking features $T_i$를 예측하는 데 사용됨
- \hat{t}_i^I$는 DPT 레이어를 통해 dense feature map $F_i \in \mathbb{R}^{C'' \times H \times W}$로 변환됨
- 이후 각 $F_i$는 3x3 conv layer을 통해 해당 depth&point map $D_i, P_i$로 매핑됨
- DPT 헤드는 tracking head의 입력으로 사용되는 dense features $T_i \in \mathbb{R}^{C \times H \times W}도 출력
- 각 depth&point map에 대해 비결정적 불확실성 $\Sigma_i^D \in \mathbb{R}_+^{H \times W}$와 $\Sigma_i^P \in \mathbb{R}_+^{H \times W}$도 예측
- 불확실성 map은 loss 계산에 사용되며, 훈련 후에는 모델의 예측 확산에 비례

**Tracking**

- CoTracker2 아키텍처 사용
    - dense tracking features $T_i$를 입력으로 받음
    - query 이미지 $I_q$에서 쿼리 포인트 $y_j$가 주어졌을 때, tracking head T는 모든 이미지 $I_i$에서 y와 같은 3D 포인트에 해당하는 2D 포인트 집합 $T((y_j)^M_{j=1}, (T_i)^N_{i=1}) = ((\hat{y}_{j,i})^N_{i=1})^M_{j=1}$를 예측
        - 학습 중에는 q=1로 설정하지만, 다른 이미지도 잠재적으로 쿼리로 사용 가능
    - 이 feature은 $T_i, i \neq q$인 다른 모든 feature map과 상관관계를 계산하여 correlation map set을 얻는데 사용됨
    - 이 맵들은 self-attention layer을 통해 처리되어, y_i와 모두 대응하는 최종 2D point $\hat{y}_i$를 예측
- VGSfM과 비슷하게, tracker는 입력 frame의 시간적 순서에 대한 가정을 하지 않음
    - 비디오 뿐만 아니라 어떤 이미지 집합에도 적용 가능


### 3.4 Training

**Training Losses**

$$
\mathcal{L} = \mathcal{L}_{camera} + \mathcal{L}_{depth} + \mathcal{L}_{pmap} + λ \mathcal{L}_{track}
\tag{2}
$$
- camera loss
    - $| \cdot |_\epsilon$: Huber loss
$$
L_{\text{camera}} = \sum_{i=1}^{N} \left\| \hat g_i - g_i \right\|_{\epsilon}
$$

- depth loss
    - DUSt3R을 따름
    - aleatoric-uncertainty loss
    - 예측 깊이와 정답의 차이를 예측된 불확실성 map $\Sigma^D_i$으로 가중
    - 공간적 기울기 항도 포함하여 경계 및 형태 보존을 다움
    - $\odot$: channel-broadcast element-wise 곱
    - $\alpha$: 불확실성 항의 scale 조정
$$
L_{\text{depth}} = \sum_{i=1}^{N} \Big( \|\,\Sigma^D_i \odot (\hat D_i - D_i)\,\| \;+\; \|\,\Sigma^D_i \odot (\nabla \hat D_i - \nabla D_i)\,\| \;-\; \alpha \log \Sigma^D_i \Big)
$$
- point map loss
$$
L_{\text{pmap}} = \sum_{i=1}^{N} \Big( \|\,\Sigma^P_i \odot (\hat P_i - P_i)\,\| \;+\; \|\,\Sigma^P_i \odot (\nabla \hat P_i - \nabla P_i)\,\| \;-\; \alpha \log \Sigma^P_i \Big)
$$
- tracking loss
    - query point $y_i$에 대해 모든 프레임에서의 예측 2D 좌표 $\hat{y}_{j, i}$와 정답 $y_{j, i}$의 유클리드 거리 합
        - $\hat{y}_{j, i}$는 tarcking module로 얻은 예측
    - CoTracker2에 따라 가시성에 대해 binary cross-entropy loss를 추가 적용
$$
L_{\text{track}} = \sum_{j=1}^{M} \sum_{i=1}^{N} \left\| y_{j,i} - \hat y_{j,i} \right\|
$$

**Ground Truth Coordinate Normalization**
- 장면을 확대하거나 전역 기준 좌표계를 바꿔도 장면 이미지는 영향을 받지 않음
- 첫 번째 이미지를 기준으로 출력하도록 함
- 방법
    - 모든 수치를 첫 번째 카메라 $g_1$의 좌표계로 표현
    - point map P의 모든 3D 점이 원점에서 얼마나 떨어져 있는지 euclidean 거리 평균을 계산
        - 이 scale을 사용해 카메라 이동 $t$, point map $P$, depth map $D$를 정규화
    - [129]와 달리 transformer가 출력하는 예측값에는 이런 정규화를 적용하지 않음
        - transformer가 이 정규화를 배우도록 함

**Implementation Details**
- 전역 및 프레임별 attention에 각각 24개의 layer($L=24$)를 사용
- 모델은 약 1.2B 파라미터로 구성
- AdamW optimizer 사용
    - 160K iteration
- cosine learning rate scheduler
    - 최고 학습률은 0.0002
    - 8K 동안 warmup
- 각 배치마다 random 학습 장면에서 2~24 프레임을 무작위 샘플링
- 입력 프레임, depth map, point map은 최대 518 픽셀로 크기 조정
- 화면 비율은 0.33에서 1.0 사이로 무작위 설정
- 64개 A100 GPU에서 9일동안 학습
- gradient norm clipping을 threshold 1.0으로 사용
- bfloat 16과 gradient checkpointing도 사용

**Training Data**
- Co3Dv2
- BlendMVS
- DL3DV
- MegaDepth
- Kubric
- WildRGB
- ScanNet
- HyperSim
- Mapillary
- Habitat
- Replica
- MVS-Synth
- PointOdyssey
- Virtual KITTI
- Aria Synthetic Environments
- Aria Digital Twin
- Objaverse와 유사한 아티스트 제작 에셋의 합성 데이터

## 4. Experiments

![alt text](./images/Fig%203.png)

> **Figure 3. in-the-wild 이미지에서 3D point 예측을 DUSt3R과 질적 비교**
> - 위쪽 행: 제안 방법은 유화의 기하학적 구조를 성공적으로 예측
>   - DUSt3R은 왜곡된 평면 예측
> - 두 번째 행: 제안 방법은 겹치지 않는 두 이미지에서 3D 장면을 올바르게 복원
>   - DUSt3R은 실해
> - 세 번째 행: 반복된 텍스처가 있는 어려운 예시. 제안 방법은 높은 품질 유지
> - DUSt3R이 한계를 넘으면 메모리가 부족해지므로 32프레임이 넘는 예시는 포함하지 않음

![alt text](./images/Fig%204.png)

> **Figure 4. Point map 추정의 추가 시각화
> - 카메라 frustum은 추정된 카메라 위치를 보여줌

![alt text](./images/Fig%205.png)

> **Figure 5. Rigid & Dynamic point tracking 시각화**
> - 위쪽: VGGT의 tracking module $\mathcal{T}$는 정적인 장면을 나타내는 순서가 없는 입력 이미지 집합에 대해 keypoint track을 출력
> - 아래쪽: 순차적 입력을 처리하는 동적 point tracker CoTracker을 향상시키기 위해 VGGT의 backbone을 finetune

### 4.1 Camera Pose Estimation (표 1)

![alt text](./images/Table%201.png)

> **Table 1. Camera Pose Estimation on RealEstate10K and CO3Dv2$$
> - 10 random frames 사용
> - 모든 지표는 높을수록 좋음
> - 어떤 방법도 Re10K로 학습되지 않음
> - 실행 시간은 H100 GPU 하나로 측정
> - $\ddagger$: concurrent work

- CO3Dv2와 RealEstate10K 데이터셋에서 평가(Table 1 참고)
    - [124]를 따라, 각 장면에서 임의로 10장의 이미지를 선택하고 AUC@30으로 평가
        - RRA(Relative Rotation Accuracy)와 RTA(Relative Translation Accuracy)를 결합한 것
        - 각각 이미지 쌍마다 회전과 이동에서 상대 각도 오차를 계산
        - 임계값 처리됨
    - Table 1에서, 학습 가능한 방법들은 Co3Dv2에서 학습되었고 RealEstate10K에서는 학습되지 않음
    - Feedforward model은 두 데이터셋 모두에서 경쟁 방법들을 모든 지표에서 일관되게 능가함
    - Global Alignment for DUSt3R/MASt3R 또는 VGGSfM의 Bundle Adjustment와 같이 계산적으로 비용이 많이 드는 후처리 단계를 사용하는 방법들과 비교해도 우수한 성능을 보임
        - 이런 방법들은 일반적으로 10초 이상 소요
        - VGGT는 0.2초
    - VGGT는 Fast3R과 유사한 속도와 더 좋은 성능을 보임
        - RealEstate10K에서 우위가 더 뚜렷함
            - 일반화가 뛰어남
    - VGGT가 visual geometry optimization과 결합하여 더 향상될 수 있음
        - 예측된 카메라 포즈와 track을 BA로 정제하면 정확도가 더 향상됨
        - 거의 정확한 point/depth map을 직접 예측하여 BA의 초기값으로 사용 가능
            - 삼각측량과 반복적인 정제가 필요 없음
            - BA와 함께 해도 2초 정도로 빠르게 수행됨
    - 후처리 최적화가 이점을 제공하므로 개선 여지 있음

### 4.2 Multi-view Depth Estimation (표 2)

![alt text](./images/Table%202.png)

> **Table 2. Dense MVS Estimation on the DTU Dataset**
> - 상단: 실제 카메라 정보를 아는 방법
> - 하단: 실제 카메라 정보를 모르는 방법

- MASt3R[62]처럼, DTU 데이터셋에서 multi-view depth estimation 결과 평가
- 지표
    - Euclidean 거리 정확도(Accuracy)
    - 완전성(Completeness)
        - 실제 값과 예측 사이의 가장 작은 euclidean 거리
    - 전체 평균(Overall)
- DUSt3R과 VGGT는 카메라 위치 정보를 사용하지 않고 작동
- MASt3R은 실제 카메라 정보를 사용해 매칭을 삼각측량하여 depth map 생성
- GeoMVSNet과 같은 deep multi-view stereo 방법은 실제 카메라 정보를 사용해 cost volume을 생성
- VGGT는 DUSt3R보다 성능이 뛰어남
    - 전체 점수를 1.741에서 0.382로 낮춤
    - 실제 카메라 정보를 아는 방법과 비교해도 비슷한 결과 달성
    - 모델이 multi-image training 방식을 통해 multi-view 삼각측량을 자연스럽게 학습하도록 만듦

### 4.3 Point Map Estimation (표 3)

![alt text](./images/Table%203.png)

> **Table 3. Point Map Estimation on ETH3D**
> - DUSt3R와 MASt3R은 global alignment 사용
> - VGGT는 feed-forward라서 훨씬 빠름
> - 'Ours(Point)'는 point map head를 사용한 결과
> - 'Ours(Depth+Cam)'은 depth map head와 camera head를 결합하여 point cloud를 만든 결과

- ETH3D 데이터셋에서 예측한 point cloud의 정확도를 DUSt3R과 MASt3R과 비교
- 각 장면마다 임의로 10프레임 샘플링
- 예측한 point cloud는 Umeyama 알고리즘을 사용해 실제값과 정렬
- 결과는 공식 mask를 사용해 유효하지 않은 point를 걸러낸 후 보고
- 지표
    - point map 추정 정확도
    - 완전성
    - 전체 Chamfer 거리
- DUSt3R과 MASt3R은 비용이 많이 드는 최적화(global 정렬. scene 당 10초)를 수행
- VGGT는 feed-forward 방식에서도 장면당 0.2초
- 추정 point map과 depth와 카메라 예측의 결과를 비교했을 때, 후자가 높은 정확도 보임
    - 복잡한 작업을 단순한 하위 문제로 나누는 이점
- Fig 3: in-the-wild 장면에서 DUSt3R과의 비교
- Fig 4: 추가 예시
- VGGT 성능 분석
    - 고품질 예측을 제공
    - 일반화 성능이 뛰어남
    - 유화, 겹치지 않는 프레임, 사막과 같은 반복되거나 균질된 텍스처가 있는 out-of-domain 예제에서도 잘 작동

### 4.4 Image Matching (표 4)

![alt text](./images/Table%204.png)

> **Table 4. ScanNet-1500에서 Two-View matching 비교**
> - VGGT의 tracking head는 two-view 환경에 특화되지 않았지만, 최신 two-view 매칭 방법인 Roma를 능가
- AUC 기준으로 측정(높을수록 좋음)

- two-view 이미지 매칭
    - rigid point tracking의 특정한 경우를 나타냄
- ScanNet 데이터셋에서 표준 프로토콜을 따르고, 결과를 표 4에 보고
- 각 이미지 쌍에 대해 matching을 추출하고, 이를 사용해 essential matrix를 추정
- 이후 이를 상대 카메라 포즈로 분해
- 최종 지표
    - 상대 포즈 정확도(AUC로 측정)
- ALIKED를 사용하여 keypoint를 검출하고, 이를 query point $y_q$로 취급
- 이를 tracking branch $\mathcal{T}$에 전달하여 두 번째 프레임에서 대응점을 찾음
- 매칭 수, RANSAC 임계값 등 평가 hyper parameter은 Roma[33]에서 가져옴
- VGGT는 모든 기준 모델 중 가장 높은 정확도 달성

### 4.5 Ablation Studies

**Feature Backbone (표 5)**

![alt text](./images/Table%205.png)

> **ETH3D에서 Transformer Backbone에 대한 Ablation Study**
> - alternating-attention 아키텍처를 두 가지 변형과 비교
>   - global self-attention만 사용
>   - cross-attention을 사용

- Alternating Attention 설계의 효과를 두 가지 다른 attention 아키텍처와 비교하여 검증
    - (a) global self-attention
    - (b) cross-attention
- 모든 모델 변형은 동일한 수의 파라미터를 유지
    - 총 2L 개의 attention layer을 사용
- cross-attention 변형
    - 각 프레임은 다른 모든 프레임의 토큰을 독립적으로 참조하며 프레임 간 정보 융합 극대화
    - 입력 프레임 수가 많아질수록 실행 시간이 크게 늘어남
- hidden dimension과 헤드 수 등의 하이퍼 파라미터는 동일하게 유지
- point map 추정 정확도를 평가 지표로 선택
- Alternating-Aetention 아키텍처가 두 가지 기본 변형보다 명확히 뛰어난 성능을 보임
- 다른 실험에서도 cross-attention을 사용하는 아키텍처는 일반적으로 self-attention만 사용하는 아키텍처보다 성능이 낮다는 것이 일관되게 나옴

**Multi-task Learning (표 6)**

![alt text](./images/Table%206.png)

> **Table 6. Ablation Study for Multi-task Learning
> - 카메라, depth, track 추정을 동시에 학습하면 ETH3D에서 point map 추정 정확도가 가장 높음

- 단일 네트워크를 훈련시켜 여러 3D 양(quantities)을 동시에 학습하는 이점 확인
- 출력들이 잠재적으로 겹칠 수 있음
    - 예: point map과 카메라 파라미터&depth map
- 카메라, depth, track 추정을 하지 않고 훈련하면 pointmap 추정 정확도 감소함
- 카메라 파라미터 추정을 포함하면 point map 정확도 향상
- depth map 추정은 약간의 개선만 보임

### 4.6 Finetuning for Downstream Tasks (표 7)

**Feed-forward Novel View Synthesis**

- 대부분 기존 방법은 카메라 파라미터가 알려진 이미지를 입력받아 새로운 카메라 시점에 해당하는 target 이미지 예측
- 명시적인 3D 표현에 의존하는 대신, LVSM을 따라 VGGT를 수정해 바로 타겟 이미지를 출력하게 함
    - 입력 프레임에 대해 알려진 카메라 파라미터를 가정하지 않음
- LVSM의 학습 및 평가 프로토콜을 따름
    - 4개의 입력 뷰 사용
    - Plucker ray를 사용해 target 시점을 표현
- 입력 이미지는 DINO로 토큰화됨
- target view는 convolutional layer을 사용해 Plucker ray 이미지를 토큰으로 인코딩
    - 이 토큰들은 입력 이미지와 target view를 모두 나타냄
    - AA Transformer을 통해 처리됨
    - 이후 DPT head를 사용해 target view의 RGB 색상을 회귀
    - source 이미지에 대해 Plucker ray를 입력하지 않음
        - 모델은 입력 frame의 카메라 파라미터를 제공받지 않음
- LVSM은 Objaverse 데이터셋에서 학습됨
- Objaverse의 약 20% 규모의 내부 데이터셋을 비슷하게 사용
- 학습과 평가에 대한 자세한 내용은 [53]에서 확인 가능
- Tab 7에 나타난 것처럼, 입력 카메라 파라미터가 필요 없고, LVSM보다 적은 학습 데이터를 사용했음에도 GSO 데이터셋에서 경쟁력있는 결과를 보여줌
- 더 큰 Training dataset을 사용하면 더 나은 결과가 나올 것으로 예상됨

![alt text](./images/Fig%206.png)

> **Figure 6. Novel View 합성의 질적 예시**
> - 맨 윗 줄: 입력 이미지
> - 가운데 줄: 목표 시점의 실제 이미지
> - 맨 아래 줄: 직접 합성한 이미지

![alt text](./images/Table%207.png)

> **Table 7. GSO 데이터셋에서 view 합성에 대한 정량적 비교**
> - feed-forward novel view 합성을 위해 VGGT를 finetuning하면, 입력 이미지의 카메라 외부 및 내부 파라미터를 몰라도 경쟁력 있는 성능을 보임
> - *: 작은 학습 세트를 사용했음을 나타냄(20%만 사용)

**Dynamic Point Tracking (표 8)**


![alt text](./images/Table%208.png)

- 추적 지표
    - Occlusion Accuracy(OA)
        - occlusion 예측의 이진 정확도를 포함
    - $\delta_{avg}^{vis}$
        - 특정 픽셀 임계값 내에서 정확하게 추적된 visible 점의 평균 비율
    - Average Jaccard(AJ)
        - 추적과 occlusion 예측 정확도를 함꼐 측정
- CoTracker2 모델을 사전학습된 feature backbone으로 교체해 조정
    - VGGT가 순차적인 비디오가 아니라 정렬되지 않은 이미지 모음으로 학습됨
    - backbone은 추적 feature $T_i$를 예측
    - feature extractor의 출력을 대체
    - 이후 CoTracker2 구조의 나머지 부분에 들어가서 track을 예측
- 전체 수정된 tracker을 Kubric에서 finetune
- 사전학습된 VGGT를 통합하면 TAP-Vid benchmark에서 CoTracker 성능이 크게 향상됨(Table 8 참고)
    - VGGT tracking feature은 TAP-Vid RGB-S 데이터셋에서 $\delta_{avg}^{vis}$ 지표를 78.9에서 84.0으로 끌어올림
- VGGT는 명시적으로 설계되지 않은 시나리오에서도 feature의 일반화 능력을 보임
    - TAP-Vid 벤치마크에 다양한 데이터 소스에서 빠른 동적 움직임이 있는 video가 포함되어 있음에도 불구하고

## 5. Discussions

**Limitations**

- fisheye 또는 파노라마 이미지를 지원하지 않음
- 극단적인 입력 회전이 있는 경우 복원 성능 떨어짐
- 약간의 비탄성 움직임이 있는 장면은 처리 가능하지만, 상당한 비탄성 변형이 있는 시나리오에서는 실패함

- 장점
    - 유연성 및 적용 용이성
- 위의 한계들은 최소한의 구조 변경으로 데이터셋에 맞게 모델을 finetuning하여 해결 가능
- 기존 방법은 특수한 상황에 맞게 대규모 재설계가 필요한 경우가 많음


**Runtime and Memory**

![alt text](./images/Table%209.png)

> **Table 9. 다양한 입력 프레임 수에 따른 runtime과 최대 GPU 메모리 사용량**
> - runtime은 초 단위로 측정됨
> - GPU 메모리 사용량은 기가바이트 단위로 보고됨

- 입력 트레임 수가 달라질 때 feature backbone의 추론 실행 시간과 최대 GPU 메모리 사용량을 평가
- 환경
    - flash attention v3을 사용한 H100 GPU로 수행
    - 이미지 해상도: 336x518
- 카메라 헤드
    - feature backbone 시간의 약 5%와 GPU 메모리의 약 2% 차지
- DPT head
    - 프레임당 평균 0.03초와 0.2GB GPU 메모리 사용
- GPU 메모리가 충분한 경우, 여러 프레임을 하나의 forward pass로 효율적으로 처리 가능
- 프레임 간 관계가 feature backbone 내에서만 처리되며, DPT 헤드는 frame별로 독립적으로 예측
    - GPU 리소스에 제한이 있는 사용자는 프레임 단위로 예측 가능
- global self-attention을 단순하게 구현하면 token 수가 많을 때 메모리 소모가 커질 수 있음
    - LLM의 기술 활용해서 메모리 절약 혹은 추론 속도 향상 가능
        - 예: Fast3R은 tensor 병렬 처리를 사용하여 여러 GPU에서 추론 가속화

**Patchifying**

- 이미지를 14x14 conv layer을 이용하거나 사전 학습된 DINOv2 모델을 사용해서 토큰으로 나누는 방법 탐구
- DINOv2 모델이 더 나은 성능을 보임
    - 초기 단계에서 안정적인 학습 보장
    - 학습률이나 모멘텀 같은 하이퍼파라미터 변화에 덜 민감

**Differentiable BA**

- 미분 가능한 BA 사용 아이디어
    - 소규모 예비 실험에서는 유망한 성능을 보임
    - Theseus[85]를 사용해서 Pytorch에서 미분 가능한 BA를 활성화하면 각 훈련 단계가 약 4배 느려져서 대규모 훈련에는 비용이 많이 들음
    - 훈련을 빠르게 하기 위해 프레임워크를 커스터마이징 하는 작업은 연구의 범위를 벗어남
    - 명시적인 3D 주석이 없는 상황에서는 효과적인 감독 신호로 작용할 수도 있음
    - 대규모 비지도 학습에서는 유망할 수 있음

**Single-view Reconstruction**

- 단일 이미지 입력을 지원
    - 이전에는 DUSt3R/MASt3R같은 이미지 쌍을 만들기 위해 이미지를 복제해야 함
    - global attention이 frame 단위 attention으로 자연스럽게 전환됨
- 단일 view 재구성을 위해 명시적으로 학습되지 않았지만, 좋은 결과를 보임(Fig 3, 7 참고)

**Normalizing Prediction**

- 3D point의 평균 euclidean 거리를 사용해서 GT를 정규화
- DUSt3R같은 몇몇 방법도 네트워크 예측값에 이런 정규화를 적용
    - 수렴에 꼭 필요하지는 않음
    - 최종 모델 성능에 도움되지도 않음
    - 훈련 단계에서 추가적인 불안정을 불러오기도 함

## 6. Conclusions

- VGGT
    - 수백 개 입력 뷰에서 모든 주요 3D 장면 속성을 직접 추정할 수 있는 Feed-forward 신경망
    - 카메라 파라미터 추정, multi-view depth 추정, dense point cloud 복원, 3D point 추적 등 여러 3D 작업에서 SOTA 성능을 보임
    - 간단하고 신경망 중심 접근 방식은 최적화와 후처리에 의존해 정확하고 작업별 결과를 얻는 전통적인 시각적 기하학 기반 방법과 다름
    - 단순함과 효율성은 실시간 application에 잘맞음