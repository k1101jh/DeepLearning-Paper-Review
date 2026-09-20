# Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Gaussian Splatting

---
- **논문 제목**: Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats
- **저자**: Ziwen Chen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Fuxin Li, Zexiang Xu
- **발표 연도**: 2025 (ICCV 2025 Highlight / arXiv 2024.10)
- **공식 링크**: [arXiv](https://arxiv.org/abs/2410.12781), [Project Page](https://arthurhero.github.io/projects/llrm/), [GitHub](https://github.com/arthurhero/Long-LRM)

---

## ⚡ 요약 (Summary)

### 1. 한 줄 요약
**Long-LRM**은 Mamba2와 Transformer를 결합한 하이브리드 아키텍처, 경량 토큰 병합(Token Merging), 그리고 투명도 정규화 기반 가우시안 가지치기(Pruning)를 도입하여 **32장의 고해상도(960×540) 입력 이미지(약 25만 토큰)**로부터 **단 1초 만에 360° 광범위 장면 수준의 3D Gaussian Splatting 재구성을 피드포워드(Feed-forward) 방식으로 달성**한 대형 재구성 모델이다.

### 2. 핵심 아이디어
- **Mamba2 & Transformer 하이브리드 아키텍처 ({7M1T}×3 구조)**:
  - 25만 개에 달하는 초장열 토큰(Long-sequence Token) 처리의 이차 시간/메모리 복잡도($$O(L^2)$$) 문제를 해결하기 위해, 선형 복잡도($$O(L)$$)를 가진 양방향 Mamba2 블록 7개와 전역 맥락(Global Context) 인지능력이 뛰어난 Transformer 블록 1개를 1개 세트로 묶어 총 24개 블록으로 배치.
- **경량 토큰 병합(Light-weight Token Merging)**:
  - 네트워크 중간(9번째 블록 직전)에 2D 픽셀 패치 시퀀스를 2×2 보타이 컨볼루션(Stride 2)으로 압축하여 토큰 수를 1/4로 줄이고 차원을 256에서 1024로 확장함으로써 메모리 소모 및 학습 시간을 획기적으로 축소.
- **투명도 정규화 손실 및 가우시안 가지치기(Pruning)**:
  - 픽셀별 대규모 가우시안 생성(약 1,700만 개)에 따른 메모리 과부하 및 부유물(Floaters) 아티팩트를 억제하기 위해 $$L_1$$ Opacity Loss와 DepthAnything 기반 Soft Depth Loss를 적용하고, 불투명도 0.001 이하의 가우시안을 가지치기하여 추론 및 렌더링 효율을 극대화.
- **다양한 Gaussian 변형(2D GS 등)과의 호환성**:
  - 3D Gaussian뿐만 아니라 2D Gaussian Splatting (Huang et al., 2024) 파라미터로 확장 fine-tuning을 수행하여 유효한 정밀 기하(Geometry) 및 깊이(Depth) 지도 재구성 기능 확보.

### 3. 문제 정의
- **기존 3D GS Optimization 방법의 한계**:
  - 3D Gaussian Splatting (Kerbl et al., 2023), Mip-Splatting, Scaffold-GS 등은 높은 렌더링 품질을 보이지만, 새로운 장면마다 최소 10분~16분 이상의 경사하강법 기반 최적화(Per-scene Optimization)가 필수적이어서 즉각적인(Instant) 3D 생성에 부적합함.
- **기존 Feed-forward 3D GS 모델의 한계**:
  - pixelSplat (Charatan et al., 2024), MVSplat (Chen et al., 2025) 등은 에피폴라 지오메트리나 Plane-sweep Cost Volume 등 3D 기하학적 귀속 편향(Inductive Bias)에 의존하여 1~4장의 좁은 시점(Narrow-view) 입력에만 한정됨.
  - GS-LRM (Zhang et al., 2024)은 트랜스포머의 전역 어텐션을 활용했으나 이차 복잡도($$O(L^2)$$) 문제로 인해 2~4장 입력(약 4,000 토큰) 이상으로 확장하지 못하고 광범위 360° 시점 재구성이 불가능했음.

### 4. 실험 결과 및 의의
- **광범위 장면 재구성 성능 (DL3DV-140 & Tanks&Temples)**:
  - 32장 960×540 입력 기준, 13분 이상 소요되는 최적화 기반 3D GS 대비 **800배 빠른 1초의 피드포워드 시간**으로 대등하거나 더 뛰어난 PSNR (DL3DV-140 기준 24.10 dB) 및 시각적 품질 달성.
  - 3~10 step의 후처리 최적화(Ours10: 37초) 적용 시 PSNR 25.60 dB로 최적화 기반 SOTA 모델들을 능가함.
- **Sparse-view 성능 (RealEstate10K 2-view 256×256)**:
  - 기존 피드포워드 모델(pixelSplat, MVSplat, GS-LRM) 대비 가장 우수한 PSNR 28.54 dB 기록.
- **부유물 아티팩트 제거**:
  - Soft Depth 정규화와 Opacity 정규화를 통해 최적화 기반 방법에서 흔히 발생하는 카메라 전면 부유물(Floaters) 아티팩트를 획기적으로 제거함.

### 5. 한계 및 의견
- **장점**:
  - 대규모 360° 카메라 시점(32장)과 초장열 토큰(25만 개)을 실용적인 메모리/시간 범위 내에서 피드포워드로 처리한 최초의 대형 3D 재구성 모델.
  - Mamba2와 Transformer를 효과적으로 혼합한 구조적 기여.
- **한계점 및 향후 과제**:
  - 32장 이상의 극단적으로 길어지는 비디오 스트림(100~300장 전체)을 직접 통째로 처리하기에는 여전히 메모리 한계가 있어 카메라 포즈 기반 K-means 시점 추출이 필요함.
  - 카메라 Pose 정보가 정확하게 입력되어야 하므로 Pose-free 피드포워드 3D LRM으로의 확장이 요구됨.

---

## Abstract
- **Instant Feed-forward 3D Gaussian Reconstruction**:
  - Long-LRM은 32장의 고해상도(960×540) 입력 이미지로부터 단 1초 만에 인스턴트, 고해상도, 360° 광범위 커버리지 장면 수준의 3D Gaussian 재구성을 수행하는 피드포워드 모델임.
- **250K 토큰 수용을 위한 하이브리드 설계**:
  - 대규모 입력으로 생성되는 약 25만(250K) 개에 달하는 극단적인 초장열 토큰을 처리하기 위해 Mamba2 블록과 고전적 Transformer 블록을 혼합하고, 경량 토큰 병합 모듈과 가우시안 가지치기 단계를 결합함.
- **실험 결과 및 속도 가속**:
  - DL3DV 및 Tanks&Temples 벤치마크 평가 결과, 최적화 기반 방법 대비 **800배 가속**을 달성하면서 대등한 재구성 품질을 입증함. 기존 피드포워드 모델 대비 60배 이상 큰 입력 크기를 처리함.
- **2D GS 호환성**:
  - 2D Gaussian Splatting과의 결합을 통해 표면 기하(Geometry) 재구성 능력 향상 가능성을 확인 함.

---

## 1. Introduction
- **3D 재구성 연구의 발전과 한계**:
  - NeRF (Barron et al., 2022) 및 3D Gaussian Splatting (3D GS, Kerbl et al., 2023) 등의 렌더링 필드 기법은 뛰어난 quality의 3D 재구성을 가능케 했으나, 각 장면마다 독립적인 최적화(Per-scene Optimization)를 필요로 하므로 수분~수시간이 소요되고 새로운 장면에 대한 일반화가 불가능함.
- **기존 Feed-forward GS 모델의 한계**:
  - pixelSplat (Charatan et al., 2024), MVSplat (Chen et al., 2025), MVSGaussian (Liu et al., 2024) 등 피드포워드 모델이 등장했으나, 에피폴라 어텐션이나 plane-sweep cost volume 등 3D inductive bias에 의존하여 1~4장의 좁은 커버리지 입력에만 한정됨.
  - GS-LRM (Zhang et al., 2024)은 트랜스포머의 전역 어텐션을 활용해 우수한 품질을 보여주었으나, 트랜스포머 어텐션의 이차 복잡도($$O(L^2)$$)로 인해 2~4장 입력에 갇혀 실경계 대규모 장면 재구성이 불가능했음.

![Figure 1](./images/Fig%201.png)
> **Figure 1. Long-LRM의 광범위 시점 재구성 결과.** 32장의 고해상도(960×540) 이미지를 입력받아 단 1초 만에 실경계 시점 전체를 피드포워드로 재구성하며, 13분 이상 최적화한 3D GS (Kerbl et al., 2023)와 대등하거나 더 뛰어난 visual rendering 품질을 달성함.

- **Long-LRM의 제안 및 핵심 설계**:
  - 32장의 960×540 이미지를 토큰화하면 8×8 패치 기준 약 25만 개(250K)의 토큰 시퀀스가 형성되며 이는 LLaMA3-128K 맥락 길이보다도 긺.
  - 선형 복잡도($$O(L)$$)를 지닌 SSM 계열의 Mamba2 (Dao & Gu, 2024) 블록과 Transformer 블록을 체계적으로 유기 결합하여 전역 맥락 유지와 계산 효율을 동시 확보함.
  - 네트워크 중간 토큰 병합 모듈과 가우시안 가지치기를 통해 GS-LRM과 유사한 계산 자원으로 토큰 길이를 60배 이상 확장 성공.

---

## 2. Related Work
- **3D Reconstruction**:
  - 전통적 기하 재구성(Bozic et al., 2021; Cheng et al., 2020)은 메쉬나 깊이 맵에 집중하였으나, 본 연구는 3D GS 표현을 활용하여 기하와 색상을 동시에 명시적으로 재구성함.
- **Neural reconstruction and rendering**:
  - NeRF (Mildenhall et al., 2021)는 볼륨 렌더링 기반 학습으로 화려한 시각적 품질을 달성했으나 수시간의 장면별 최적화가 필요함.
  - 3D GS (Kerbl et al., 2023) 및 VastGaussian (Lin et al., 2024) 등 변형 모델은 수십 분 수준으로 시간을 줄였으나 여전히 즉각적인 피드포워드 재구성이 불가능함.
- **Generalizable NeRF and 3D GS**:
  - Epipolar projection 및 Plane-sweep cost volume에 의존하는 일반화 모델(Chen et al., 2021; Johari et al., 2022)은 오직 1~4장의 좁은 뷰 입력에만 국한됨.
  - Mamba 기반의 Gamba (Shen et al., 2024) 및 MVGamba (Wu et al., 2024)가 제안되었으나 물체 단위(Object-level) 및 1~4 시점 재구성에 머무름.
  - Long-LRM은 Mamba2와 Transformer의 하이브리드 구조를 통해 360° 광범위 실내외 장면 수준 재구성을 32장 입력으로 달성함.
- **Efficient models for long sequences**:
  - Linear Attention (Katharopoulos et al., 2020) 및 Structured SSM (Gu et al., 2021), Mamba (Gu & Dao, 2023), Mamba2 (Dao & Gu, 2024) 등은 선형 복잡도로 초장문 처리를 가능케 함.
  - empirical 연구(Park et al., 2024)에 따르면 Mamba2 단독으로는 in-context reasoning 능력이 트랜스포머보다 떨어지므로, Jamba (Lieber et al., 2024) 스타일의 하이브리드 조합을 선택함.

---

## 3. Method

![Figure 2](./images/Fig%202.png)
> **Figure 2. Long-LRM 모델 전체 개요.** 32장의 입력 이미지 및 Plücker ray 임베딩을 시퀀스 토큰화한 뒤, 7개의 Mamba2 블록과 1개의 Transformer 블록 조합({7M1T}×3)을 거쳐 픽셀 단위 3D Gaussian 파라미터로 디코딩하고 Gaussian Pruning을 수행함.

### 3.1. Overall Architecture
- **입력 토큰화 (Tokenization)**:
  - 32장의 입력 RGB 이미지와 카메라 원점 및 방향을 표현하는 Plücker Ray 임베딩을 채널 방향으로 결합(Concatenation)한 후 패치화(Patchify)하여 선형 투영(Linear Projection)함.
  - Patch Size 8일 때 960×540 이미지 32장은 약 **250K (248,400) 개**의 토큰 시퀀스를 형성함.
- **하이브리드 모듈 배치 ({7M1T}×3)**:
  - Mamba2 블록 7개와 Self-Attention 기반 Transformer 블록 1개를 하나의 레이어 그룹으로 형성하여 총 3회 반복(총 24개 블록)함.
- **픽셀별 가우시안 디코딩 & 가지치기**:
  - 최종 출력 토큰을 픽셀별 가우시안 3D 위치(Position), 투명도(Opacity), 스케일(Scale), 회전(Rotation), 구면 조화 함수(SH Color) 파라미터로 디코딩한 뒤 Gaussian Pruning 과정을 적용함.

### 3.2. Mamba2 Block
- **선형 복잡도 연산 수식**:
  - SSM 기본 수식: $$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$
  - Mamba2 (Dao & Gu, 2024)는 대각 행렬 $$A$$를 스칼라 곱하기 단위 행렬로 제한하여 효율적인 블록 곱셈 연산을 구현하고 상태 차원을 확장함.
- **양방향 스캐닝 (Bi-directional Scan)**:
  - 1D 언어 모델과 달리 이미지 토큰 시퀀스 처리를 위해 순방향(Forward)과 역방향(Backward) 스캔을 동시 수행한 후 결과 토큰을 합산(Summation)함.

### 3.3. Token Merging and Gaussian Pruning
- **경량 토큰 병합 (Token Merging)**:
  - 9번째 블록 직전에 2D Conv (Kernel Size 2×2, Stride 2)를 적용하여 시퀀스 길이를 1/4 (약 62.5K)로 축소하고 토큰 차원을 256에서 1024로 확장함.
- **가우시안 가지치기 (Gaussian Pruning)**:
  - 픽셀별 가우시안 예측에 의해 생성되는 약 1,700만 개의 가우시안 메모리 과부하를 방지하기 위해, 불투명도 0.001 이하의 가우시안을 제거(Prune)함.

### 3.4. Training Objectives
- **Rendering Loss (이미지 렌더링 손실)**:
  $$\mathcal{L}_{\text{image}} = \frac{1}{M} \sum_{i=1}^M \left( \text{MSE}(\mathbf{I}_i^{\text{gt}}, \mathbf{I}_i^{\text{pred}}) + \lambda \cdot \text{Perceptual}(\mathbf{I}_i^{\text{gt}}, \mathbf{I}_i^{\text{pred}}) \right)$$
  (여기서 $$\lambda = 0.5$$)

- **Depth Regularization for Training Stability (깊이 정규화)**:
  $$\mathcal{L}_{\text{depth}} = \frac{1}{M} \sum_{i=1}^M \text{Smooth-L1}(\mathbf{D}_i^{\text{da}}, \mathbf{D}_i^{\text{pred}})$$
  (DepthAnything 지도 정규화를 통한 카메라 전면 부유물(Floaters) 생성 차단)

- **Opacity Regularization for Inference Efficiency (불투명도 정규화)**:
  $$\mathcal{L}_{\text{opacity}} = \frac{1}{N} \sum_{i=1}^N |o_i|$$

- **Overall Training Loss (최종 학습 손실 함수)**:
  $$\mathcal{L} = \mathcal{L}_{\text{image}} + \lambda_{\text{opacity}} \cdot \mathcal{L}_{\text{opacity}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{depth}}$$
  (여기서 $$\lambda_{\text{opacity}} = 0.1$$, $$\lambda_{\text{depth}} = 0.01$$)

---

## 4. Experiments

### 4.1. Implementation and Experiment Details
- **모델 세부 구조**: 24 블록 (7 Mamba2 + 1 Transformer 3회 반복), FlashAttention-2 연동.
- **커리큘럼 학습 (3 Stage Schedule)**:
  - Stage 1: 256×256 해상도 (60K steps, Batch Size 256, Peak LR 4e-4)
  - Stage 2: 512×512 해상도 (10K steps, Batch Size 64, Peak LR 4e-5)
  - Stage 3: 960×540 해상도 (10K steps, Batch Size 64, 광범위 시점 커버)
- **Post-prediction Optimization**: Direct Feed-forward (1초) 후 3~10 step 추가 최적화 가능.

### 4.2. Datasets
- DL3DV-10K & DL3DV-140 Benchmark (Ling et al., 2024), Tanks & Temples (Knapitsch et al., 2017), RealEstate10K (Zhou et al., 2018).

### 4.3. Results

![Table 1](./images/Table%201.png)
> **Table 1. 광범위 장면 재구성 정량적 비교 (960×540 해상도, DL3DV-140 및 Tanks&Temples).** Long-LRM 피드포워드 모델(1초)은 13분 이상 최적화한 3D GS (Kerbl et al., 2023), Mip-Splatting (Yu et al., 2024), Scaffold-GS (Lu et al., 2024) 대비 **800배 빠른 속도**와 대등하거나 우수한 PSNR 수치를 달성함.

![Figure 3](./images/Fig%203.png)
> **Figure 3. Long-LRM (1초)과 최적화 기반 3D GS (13분+)의 정성적 비교.** 기존 최적화 기반 방법들이 빈번하게 발생하는 부유물 아티팩트(빨간색 점선 타원)를 Long-LRM은 학습된 사전 지식과 깊이 정규화를 통해 깨끗하게 제거함.

![Table 2](./images/Table%202.png)
> **Table 2. RealEstate10K (2-view, 256×256) 비교.** 기존 Sparse-view 전용 피드포워드 모델(pixelSplat, MVSplat, GS-LRM) 대비 최고의 PSNR (28.54 dB) 달성.

![Table 3](./images/Table%203.png)
> **Table 3. 2D GS 변형과의 결합 수치 비교.**

![Figure 4](./images/Fig%204.png)
> **Figure 4. 2D GS 기반 렌더링 및 Depth Map 생성 결과.** 정교한 물체 경계선과 매끄러운 3D 표면 기하 재구성 확인.

---

## 5. Analysis

### 5.1. Ablation Studies of Model Designs

![Table 4](./images/Table%204.png)
> **Table 4. 모델 아키텍처 변형 및 스케일링 소단원 분석.**

- **All Transformer vs All Mamba2 vs Hybrid**:
  - All Transformer (GS-LRM)는 32개 입력 시 iteration time 50.5초로 학습 불가능.
  - All Mamba2는 속도는 빠르나 PSNR이 최대 3.3 dB 하락함.
  - {7M1T}×3 하이브리드 모델만이 우수한 재구성 품질과 연산 속도의 상호 균형을 달성함.

![Table 5](./images/Table%205.png)
> **Table 5. Token Merging 위치 소단원 분석.** 9번째 블록(@9) 배치가 가장 우수한 시가공간적 밸런스를 입증.

### 5.2. Ablation Studies of Training Objectives

![Table 6](./images/Table%206.png)
> **Table 6. 학습 손실 정규화 소단원 분석.** Opacity loss가 유효 가우시안 비율을 99.2%에서 68.3%로 절감시키며, Depth supervision이 PSNR 및 부유물 제거를 추가 견인함.

![Table 7](./images/Table%207.png)
> **Table 7. 입력 크기 및 손실 함수에 따른 가우시안 사용율 분석.**

---

## 6. Conclusions
- Long-LRM은 Mamba2와 Transformer의 결합, 토큰 병합, 가우시안 가지치기를 통해 32장 고해상도 이미지로부터 1초 만에 360° 광범위 장면 3D Gaussian 재구성을 피드포워드로 구현한 대형 모델임.
- 광범위 및 희소 시점 벤치마크 모두에서 SOTA급 성능을 기록하며 피드포워드 3D 대형 재구성 분야에 새로운 패러다임을 제시함.
