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
  - 네트워크 중간(9번째 블록 직전)에서 토큰 시퀀스를 이미지 패치 격자로 되돌린 뒤 channel-wise 2×2 2D 컨볼루션(Stride 2)으로 압축하여 토큰 수를 1/4로 줄이고(패치 크기 8→16) 차원을 256에서 1024로 확장함으로써 메모리 소모 및 학습 시간을 축소.
- **불투명도 정규화 손실 및 가우시안 가지치기(Pruning)**:
  - 픽셀별 가우시안 예측으로 생성되는 대량의 가우시안(약 1,700만 개)을 효율적으로 쓰도록 $$L_1$$ Opacity Loss를 적용하고, 불투명도 상위 일정 비율만 남기는 가지치기(학습 시 상위 40% + 나머지 중 무작위 10%, 평가 시 상위 50%)로 메모리와 렌더링 시간을 절감.
  - 학습 안정성을 위해 DepthAnything의 disparity를 이용한 scale-invariant Soft Depth Loss를 추가하여, 실제 표면 위에 놓이지 않는 부유물(Floaters) 가우시안을 억제.
- **다양한 Gaussian 변형(2D GS 등)과의 호환성**:
  - 3D Gaussian뿐만 아니라 예측된 파라미터를 2D Gaussian Splatting (Huang et al., 2024) 파라미터로 해석하여 fine-tuning함으로써 기하(Geometry) 및 깊이(Depth) 지도 재구성 능력 확보.

### 3. 문제 정의
- **기존 3D GS Optimization 방법의 한계**:
  - 3D Gaussian Splatting (Kerbl et al., 2023), Mip-Splatting, Scaffold-GS 등은 높은 렌더링 품질을 보이지만, 새로운 장면마다 일반적으로 10분 이상(Table 1 기준 13~16분)의 장면별 최적화(Per-scene Optimization)가 필요하여 즉각적인(Instant) 3D 생성에 부적합함.
- **기존 Feed-forward 3D GS 모델의 한계**:
  - pixelSplat (Charatan et al., 2024), MVSplat (Chen et al., 2025) 등은 에피폴라 지오메트리나 Plane-sweep Cost Volume 등 3D 귀납적 편향(Inductive Bias)에 의존하여 1~4장의 좁은 시점(Narrow-view) 입력에만 한정됨.
  - GS-LRM (Zhang et al., 2024)은 트랜스포머의 전역 어텐션을 활용했으나 이차 복잡도($$O(L^2)$$) 문제로 인해 2~4장 입력의 국소(local) 재구성에 머물러 광범위 360° 장면 재구성이 불가능했음.

### 4. 실험 결과 및 의의
- **광범위 장면 재구성 성능 (DL3DV-140 & Tanks&Temples)**:
  - 32장 960×540 입력 기준, 약 13분 소요되는 최적화 기반 3D GS 대비 **800배 빠른 1초의 피드포워드 시간**으로 대등한 품질 달성 (DL3DV-140 PSNR 24.10 dB; 3D GS 23.60, Mip-Splatting 23.32, Scaffold-GS 24.77).
  - 3~10 step의 후처리 최적화(Ours3: 12초, Ours10: 37초) 적용 시 PSNR 25.60 dB로 최적화 기반 방법들보다 PSNR이 높아짐(3D GS 대비 +2 dB). 단, LPIPS는 최적화 기반 방법이 더 우수함.
- **Sparse-view 성능 (RealEstate10K 2-view 256×256)**:
  - 기존 피드포워드 모델(pixelSplat 25.89, MVSplat 26.39, GS-LRM 28.10) 대비 가장 우수한 PSNR 28.54 dB 기록.
- **부유물 아티팩트 제거**:
  - 대규모 학습 데이터에서 얻은 사전 지식과 Opacity/Soft Depth 정규화 덕분에 최적화 기반 방법에서 흔히 발생하는 부유물(Floaters) 아티팩트가 크게 줄어듦.

### 5. 한계 및 의견
- **장점**:
  - 32장 고해상도 입력(약 25만 토큰)을 실용적인 메모리/시간 범위 내에서 처리하여, 광범위 장면 수준 재구성을 수 초 내에 수행한 최초의 피드포워드 GS 방법 (저자 주장).
  - Mamba2와 Transformer를 효과적으로 혼합한 구조적 기여.
- **한계점 및 향후 과제**:
  - (논문 Appendix E) 입력을 64장으로 늘려도 PSNR 향상이 1 dB 미만이며, 64장 고해상도 입력은 500K 토큰을 넘는 초장문 시퀀스가 되어 현재 시퀀스 모델로 다루기 어려움.
  - (논문 Appendix E) DL3DV 학습 데이터의 FOV가 고정되어 있어, FOV가 크게 다른 데이터셋(예: MipNeRF360)에서 일반화가 어려움. 저자들은 위치별 토큰 의미가 FOV에 따라 바뀌는 Mamba2 블록을 원인으로 추정함.
  - (의견) 입력 카메라 포즈가 필요하므로 Pose-free 피드포워드 재구성으로의 확장이 과제로 남음. 또한 평가 시 전체 시퀀스에서 K-means로 32장을 골라 입력하므로, 전체 프레임을 직접 쓰는 방식은 아님.

---

## Abstract
- **Instant Feed-forward 3D Gaussian Reconstruction**:
  - Long-LRM은 32장의 고해상도(960×540) 입력 이미지로부터 단일 A100 GPU에서 단 1초 만에 고해상도, 360° 광범위 커버리지 장면 수준의 3D Gaussian 재구성을 수행하는 피드포워드 모델임.
- **250K 토큰 수용을 위한 하이브리드 설계**:
  - 대규모 입력으로 생성되는 약 25만(250K) 개에 달하는 극단적인 초장열 토큰을 처리하기 위해 Mamba2 블록과 고전적 Transformer 블록을 혼합하고, 경량 토큰 병합 모듈과 가우시안 가지치기 단계를 결합함.
- **실험 결과 및 속도 가속**:
  - DL3DV 및 Tanks&Temples 벤치마크 평가 결과, 최적화 기반 방법 대비 **800배 가속**을 달성하면서 대등한 재구성 품질을 입증함. 기존 피드포워드 모델 대비 최소 60배 큰 입력 크기를 처리함.
  - 모델 설계 선택에 대해 렌더링 품질과 연산 효율 양 측면에서 광범위한 ablation을 수행함.
- **2D GS 호환성**:
  - 2D Gaussian Splatting과의 결합을 통해 기하(Geometry) 재구성 능력이 향상됨을 확인함.

---

## 1. Introduction
- **3D 재구성 연구의 발전과 한계**:
  - NeRF (Mildenhall et al., 2021) 및 다양한 radiance field 기법은 고품질 3D 재구성을 가능케 했으나, 각 장면마다 독립적인 최적화(Per-scene Optimization)가 필요하여 느리고 새로운 장면에 일반화되지 않음.
  - 3D Gaussian Splatting (3D GS, Kerbl et al., 2023)은 효율을 크게 개선했지만 여전히 장면당 최소 10분 정도의 최적화가 필요하여 즉각적인 재구성은 불가능함.
- **기존 Feed-forward GS 모델의 한계**:
  - pixelSplat (Charatan et al., 2024), MVSplat (Chen et al., 2025), MVSGaussian (Liu et al., 2024) 등 피드포워드 모델이 등장했으나, 에피폴라 어텐션이나 plane-sweep cost volume 등 3D inductive bias에 의존하여 1~4장의 좁은 커버리지 입력에만 한정됨.
  - GS-LRM (Zhang et al., 2024)은 트랜스포머의 전역 어텐션을 활용해 우수한 품질을 보여주었으나, 이전 피드포워드 GS 모델들은 1~4장 수준의 좁은 커버리지 입력만 다루도록 설계되어, 최소 수십 장이 필요한 대규모 실세계 장면은 여전히 최적화 기반 방법만이 유일한 선택지였음.

![Figure 1](./images/Fig%201.png)
> **Figure 1. Long-LRM의 광범위 시점 재구성 결과.** 32장의 고해상도(960×540) 이미지로부터 넓은 시야 범위의 실세계 장면을 단 1초 만에 재구성하며, 장면당 10분 이상 최적화가 필요한 3D GS 방법들과 대등한 렌더링 품질을 달성함.

- **Long-LRM의 제안 및 핵심 설계**:
  - 32장의 960×540 이미지를 토큰화하면 8×8 패치 기준 약 25만 개(250K)의 토큰 시퀀스가 형성되며 이는 LLaMA3의 128K 맥락 길이보다도 긺.
  - 선형 복잡도($$O(L)$$)를 지닌 SSM 계열의 Mamba2 (Dao & Gu, 2024) 블록과 Transformer 블록을 체계적으로 유기 결합하여 전역 맥락 유지와 계산 효율을 동시 확보함.
  - 네트워크 중간 토큰 병합 모듈과 가우시안 가지치기를 통해 GS-LRM과 유사한 계산 자원으로 토큰 길이를 60배 이상 확장 성공.
  - DL3DV-10K(약 10K 장면)로 학습하고 DL3DV-140 및 Tanks and Temples에서 32장 입력으로 평가하여, 최적화 기반 3D GS와 대등한 품질을 1초(vs. 13분)에 달성함.

---

## 2. Related Work
- **3D Reconstruction**:
  - 전통적 기하 재구성(Bozic et al., 2021; Cheng et al., 2020)은 메쉬나 깊이 맵에 집중하였으나, 본 연구는 3D GS 표현을 활용하여 기하와 색상을 동시에 명시적으로 재구성함.
- **Neural reconstruction and rendering**:
  - NeRF (Mildenhall et al., 2021)는 미분 가능한 볼륨 렌더링으로 3D GT 없이 높은 시각적 품질을 달성했으나, 장면마다 MLP를 최적화하는 데 수 시간~수 일이 필요함.
  - 3D GS (Kerbl et al., 2023)는 재구성 시간을 수십 분 수준으로 줄이면서 실시간 렌더링을 가능케 했고, CityGaussian, Octree-GS, VastGaussian (Lin et al., 2024) 등 변형 모델은 도시 규모로 확장했으나 여전히 장면마다 수 분 이상의 최적화가 필요함.
- **Generalizable NeRF and 3D GS**:
  - Epipolar line 및 Plane-sweep cost volume에 의존하는 일반화 NeRF/GS 모델(Chen et al., 2021; Johari et al., 2022 등)은 입력 뷰 간 큰 겹침이 필요하므로 1~4장의 좁은 뷰 입력에 국한됨.
  - GS-LRM (Zhang et al., 2024)은 이러한 3D 특화 구조 없이 전역 어텐션 트랜스포머로 SOTA를 달성했으나, 이차 복잡도 때문에 2~4장 입력에 머묾.
  - Mamba 기반의 Gamba (Shen et al., 2024) 및 MVGamba (Yi et al., 2024)가 제안되었으나 물체 단위(Object-level) 및 1~4 시점 재구성에 머무름.
  - Long-LRM은 Mamba2와 Transformer의 하이브리드 구조를 통해 360° 광범위 실내외 장면 수준 재구성을 32장 입력으로 달성함.
- **Efficient models for long sequences**:
  - Linear Attention (Katharopoulos et al., 2020) 및 Structured SSM (Gu et al., 2021), Mamba (Gu & Dao, 2023), Mamba2 (Dao & Gu, 2024) 등은 선형 복잡도로 초장문 처리를 가능케 함.
  - Mamba2 (Dao & Gu, 2024)는 상태 행렬 $$A$$를 제한하고 상태 차원을 확장하여 여러 언어 과제에서 트랜스포머에 필적하는 성능을 보임.
  - 그러나 실증 연구(Waleffe et al., 2024)에 따르면 3D 재구성에 중요한 in-context learning과 long-context reasoning에서는 여전히 트랜스포머가 우세하므로, 이 연구와 Jamba (Lieber et al., 2024)에서 영감을 받아 트랜스포머와 Mamba2의 하이브리드 구조를 채택함.

---

## 3. Method

![Figure 2](./images/Fig%202.png)
> **Figure 2. Long-LRM 모델 전체 개요.** 32장의 입력 이미지 및 Plücker ray 임베딩을 시퀀스 토큰화한 뒤, 7개의 Mamba2 블록과 1개의 Transformer 블록 조합({7M1T}×3)을 거쳐 픽셀 단위 3D Gaussian 파라미터로 디코딩하고 Gaussian Pruning을 수행함.

### 3.1. Overall Architecture
- **입력 토큰화 (Tokenization)**:
  - 32장의 입력 RGB 이미지와 카메라 원점 및 방향을 표현하는 Plücker Ray 임베딩을 채널 방향으로 결합(Concatenation)한 후 패치화(Patchify)하여 선형 투영(Linear Projection)함.
  - Patch Size 8일 때 960×540 이미지 32장은 약 **250K 개**의 토큰 시퀀스를 형성함.
  - GS-LRM과 마찬가지로 픽셀별 가우시안 예측을 sequence-to-sequence 매핑으로 간주함.
- **하이브리드 모듈 배치 ({7M1T}×3)**:
  - Mamba2 블록 7개와 전역 Self-Attention 기반 Transformer 블록 1개를 하나의 하이브리드 블록으로 구성하여 총 3회 반복(총 24개 블록)함. 저자들은 이 구성이 균형이 좋고 수렴이 쉬웠다고 보고함.
  - 하이브리드 블록 사이에 토큰 병합 모듈을 삽입하여 처리 속도를 높임.
- **픽셀별 가우시안 디코딩 & 가지치기**:
  - 최종 출력 토큰을 linear layer로 픽셀별 가우시안 파라미터(위치, 불투명도, 스케일, 회전, SH degree 3 색상)로 디코딩하고, 학습/테스트 시 Gaussian Pruning을 적용함. 가우시안 위치는 각 픽셀의 카메라 ray 위에 정렬됨 (Appendix B).

### 3.2. Mamba2 Block
- **선형 복잡도 연산 수식**:
  - SSM 기본 수식: $$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$
  - 이전 SSM(S4)과 달리 Mamba는 $$A, B, C$$를 모델 파라미터로 저장하지 않고 입력으로부터 계산함. 트랜스포머 블록은 $$O(L^2)$$, Mamba는 $$O(L)$$ 복잡도를 가짐.
  - Mamba2 (Dao & Gu, 2024)는 상태 행렬 $$A$$를 스칼라 × 단위 행렬로 제한하여 효율적인 블록 행렬 곱 연산과 더 큰 상태 차원으로의 확장을 가능케 함.
- **양방향 스캐닝 (Bi-directional Scan)**:
  - Mamba2는 언어용으로 설계되어 한 방향으로만 스캔하므로 이미지에는 부적합함. Vision Mamba (Zhu et al., 2024)를 따라 linear layer로 상태 파라미터를 계산한 뒤 순방향(Forward)과 역방향(Backward) 스캔을 수행하고, 두 출력 토큰을 합산(Summation)한 후 다시 linear layer를 통과시킴.
  - VMamba, LocalMamba 같은 복잡한 스캔 패턴도 시도했으나 속도 저하가 커서 채택하지 않음.

### 3.3. Token Merging and Gaussian Pruning
- **경량 토큰 병합 (Token Merging)**:
  - 250K 토큰은 선형 복잡도 모델에도 부담이 커서, all-Mamba2 변형조차 최고 해상도에서 메모리 부족(OOM)이 발생함.
  - 다단계 CNN 인코더에서 영감을 받아, 토큰 시퀀스 $$L \times D$$를 $$N \times \frac{H}{p} \times \frac{W}{p} \times D$$로 되돌린 뒤 channel-wise 2×2 2D Conv (Stride 2)를 적용하고 다시 $$\frac{L}{4} \times D'$$로 펼침. 각 토큰의 유효 패치 크기는 $$2p$$가 됨.
  - 최종 모델은 9번째 블록 직전에 병합하며 토큰 차원을 256에서 1024로 확장함.
- **가우시안 가지치기 (Gaussian Pruning)**:
  - 픽셀별 가우시안 예측은 32장 960×540 입력에서 약 1,700만 개의 가우시안을 만들어 메모리와 속도에 큰 부담을 줌. 고해상도 입력과 카메라 frustum 간 겹침 때문에 중복이 클 것으로 봄.
  - 단순히 투명한 가우시안(불투명도 5% 수준)만 제거해도 불투명한 물체가 불완전해지는데, 이는 모델이 하나의 불투명한 색을 여러 반투명 가우시안으로 분산시키기 때문임. 따라서 Opacity Loss로 소수의 가우시안만 쓰도록 유도한 뒤 저불투명도 가우시안을 제거함.
  - 불투명도 0.001 미만 가우시안을 제거해도 렌더링 품질 차이는 없었으나, 학습 메모리를 일정하게 유지하기 위해 실제로는 임계값 대신 **고정 비율 pruning**을 사용함 (960×540 학습/추론 단계에서 렌더링 전에 적용).

### 3.4. Training Objectives
- **Rendering Loss (이미지 렌더링 손실)**:
  $$\mathcal{L}_{\text{image}} = \frac{1}{M} \sum_{i=1}^M \left( \text{MSE}(\mathbf{I}_i^{\text{gt}}, \mathbf{I}_i^{\text{pred}}) + \lambda \cdot \text{Perceptual}(\mathbf{I}_i^{\text{gt}}, \mathbf{I}_i^{\text{pred}}) \right)$$
  (여기서 $$\lambda = 0.5$$)

- **Depth Regularization for Training Stability (깊이 정규화)**:
  $$\mathcal{L}_{\text{depth}} = \frac{1}{M} \sum_{i=1}^M \text{Smooth-L1}(\mathbf{D}_i^{\text{da}}, \mathbf{D}_i^{\text{pred}})$$
  - 렌더링 손실만으로 학습하면 실제 표면에 놓이지 않는 부유물(Floaters) 가우시안이 생기고 학습이 불안정해짐.
  - $$\mathbf{D}^{\text{da}}$$는 DepthAnything이 예측한 disparity map, $$\mathbf{D}^{\text{pred}}$$는 예측된 가우시안 위치로 얻은 disparity map이며, 두 map 모두 median을 빼고 평균 절대 편차로 나누어 정규화하는 scale-invariant 손실임. 이 soft depth supervision이 학습 발산 가능성을 줄임.

- **Opacity Regularization for Inference Efficiency (불투명도 정규화)**:
  $$\mathcal{L}_{\text{opacity}} = \frac{1}{N} \sum_{i=1}^N |o_i|$$
  - 불투명도는 0~1 범위이며, L1 정규화가 희소성을 유도함. 이 손실로 불투명도 0.001 초과 가우시안 비율이 99%에서 약 40%로 감소함.

- **Overall Training Loss (최종 학습 손실 함수)**:
  $$\mathcal{L} = \mathcal{L}_{\text{image}} + \lambda_{\text{opacity}} \cdot \mathcal{L}_{\text{opacity}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{depth}}$$
  (여기서 $$\lambda_{\text{opacity}} = 0.1$$, $$\lambda_{\text{depth}} = 0.01$$)

---

## 4. Experiments

### 4.1. Implementation and Experiment Details
- **모델 세부 구조**: 24 블록 (7 Mamba2 + 1 Transformer 3회 반복). 초기 패치 크기 8, 토큰 차원 256이며, 9번째 블록 시작 시 토큰 병합으로 패치 크기 16, 차원 1024로 확장. Mamba2는 상태 차원 256, expansion rate 2, head 차원 64를 사용하고, Transformer는 head 차원 64, MLP ratio 4이며 FlashAttention-2 구현을 사용.
- **커리큘럼 학습 (3 Stage Schedule)**:
  - Stage 1: 256×256 해상도 (60K steps, Batch Size 256, Peak LR 4e-4, AdamW, weight decay 0.05, 2K step warmup 후 cosine decay). 64~128 프레임의 연속 구간에서 입력 32장을 균등 샘플링하고 타깃 8장을 무작위 샘플링함.
  - Stage 2: 512×512 해상도 (10K steps, Batch Size 64, Peak LR 4e-5)
  - Stage 3: 960×540 해상도 (10K steps, Batch Size 64). 정사각 crop 없이 전체 시퀀스(DL3DV 기준 약 200~300 프레임)로 샘플링 범위를 넓히고, 불투명도 상위 40% + 나머지 중 무작위 10%만 남기는 Gaussian pruning을 적용함. FOV augmentation(0.77~1.0 center crop) 사용.
- **평가 설정**: 시퀀스의 매 8번째 이미지를 테스트로 사용하고, 나머지에서 카메라 위치/방향 기반 K-means로 입력 뷰를 선택함. 평가 시 불투명도 상위 50% 가우시안만 유지함.
- **Post-prediction Optimization**: 피드포워드 예측 후 입력 뷰에 대한 MSE로 가우시안 위치(LR 5e-4)와 색상(LR 1e-3)만 추가 최적화하며, 불투명도/스케일/회전은 고정함. Table 1의 Ours3, Ours10은 각각 3, 10 step 최적화 결과임.
- **2D GS 학습**: 처음 두 stage 후 예측 파라미터를 2D Gaussian 파라미터로 해석하여 512×512에서 10K step fine-tuning하고, 이후 동일한 stage 3 학습을 수행함.

### 4.2. Datasets
- **DL3DV-10K** (Ling et al., 2024): 10,510개 비디오(각 200~300 keyframe, COLMAP 포즈)로 학습하고 **DL3DV-140** Benchmark(140개 장면)로 평가.
- **Tanks & Temples** (Knapitsch et al., 2017): train, truck 장면으로 zero-shot 평가.
- **RealEstate10K** (Zhou et al., 2018): pixelSplat 설정을 따른 저해상도 2-view 비교.
- **ScanNetv2** (Dai et al., 2017): 2D GS 버전의 zero-shot 깊이 평가 (부록).

### 4.3. Results

![Table 1](./images/Table%201.png)
> **Table 1. 광범위 장면 재구성 정량적 비교 (960×540 해상도, DL3DV-140 및 Tanks&Temples).** 16장/32장 입력 설정에서 30K step 최적화한 3D GS (Kerbl et al., 2023), Mip-Splatting (Yu et al., 2024), Scaffold-GS (Lu et al., 2024)와 비교함. Long-LRM(1초)은 **800배 빠른 속도**로 대등한 품질을 보이며, 후처리 최적화(Ours3, Ours10)로 PSNR이 더 향상됨. LPIPS는 최적화 기반 방법이 더 우수함.

![Figure 3](./images/Fig%203.png)
> **Figure 3. Long-LRM (1초)과 최적화 기반 3D GS (13분+)의 정성적 비교.** 32장 960×540 입력으로 재구성한 결과임. 빨간 타원은 최적화 기반 방법에 아티팩트나 부정확한 부분이 있는 영역이며, Long-LRM은 더 깨끗한 결과를 보임.

![Table 2](./images/Table%202.png)
> **Table 2. RealEstate10K (2-view, 256×256) 비교.** 기존 피드포워드 모델 대비 최고 PSNR (28.54 dB) 달성. pixelSplat, MVSplat보다 2 dB 이상 높고 GS-LRM(28.10)보다 약간 높음.

![Table 3](./images/Table%203.png)
> **Table 3. Long-LRM (2D GS 버전)과 최적화 기반 2D GS의 색상 재구성 비교 (DL3DV-140, 960×540).** Long-LRM 1초 결과는 PSNR 23.24로 2D GS 30k(13분, 24.35)보다 낮지만, 10 step 후처리 최적화(37초) 시 24.74로 더 높아짐.

![Figure 4](./images/Fig%204.png)
> **Figure 4. 두 개의 광범위 장면(두 번째는 360°)에 대한 색상 및 깊이 재구성 결과.** RGB는 3D Gaussian 버전에서, 깊이 맵은 2D GS로 fine-tuning한 버전에서 렌더링함. 깊이 맵에서 날카로운 물체 경계와 매끄러운 표면을 확인할 수 있음.

---

## 5. Analysis

### 5.1. Ablation Studies of Model Designs

![Table 4](./images/Table%204.png)
> **Table 4. 모델 아키텍처 ablation.** 입력 크기(4-view 256, 32-view 256, 32-view 512, 32-view 960×540)에 따른 학습 시간, 메모리, 재구성 품질을 비교함.

- **All Transformer vs All Mamba2 vs Hybrid**:
  - All Transformer (GS-LRM과 동일)는 4-view 256 설정에서는 하이브리드와 대등하지만, 32-view 512 설정에서 batch size 1 기준 iteration time이 50.5초로 감당할 수 없을 만큼 느려짐.
  - All Mamba2는 시간 증가가 완만하지만 품질이 떨어짐. 하이브리드 대비 PSNR이 4-view 256에서 1.8 dB, 32-view 256에서 2.5 dB, 32-view 512에서 3.3 dB 낮음.
  - 하이브리드 모델은 트랜스포머의 품질과 Mamba2의 속도를 함께 얻음. 다만 32-view 960×540 설정에서는 토큰 병합이 필수이며, 토큰 길이를 1/4로 줄이고(차원 1024 유지) 학습 시간이 2/3로, 초기 차원을 256으로 낮추면 시간과 메모리가 추가로 1/3 줄어 다른 모든 변형이 OOM인 설정에서도 학습이 가능해짐.

![Table 5](./images/Table%205.png)
> **Table 5. Token Merging 위치 ablation (4-view 256).** @1이 PSNR은 가장 높지만(21.35) 파라미터가 가장 많고(198M), @17은 파라미터가 가장 적지만(86M) PSNR이 가장 낮음(20.99). @9(142M, 21.25)를 모델 크기와 품질의 균형점으로 선택함.

### 5.2. Ablation Studies of Training Objectives

![Table 6](./images/Table%206.png)
> **Table 6. 학습 손실 ablation (4-view 256).** Opacity loss는 불투명도 0.001 초과 가우시안 비율을 99.2%에서 68.3%로 줄이면서 렌더링 성능에 미치는 영향은 미미함(PSNR 20.43→20.96). Depth supervision은 PSNR을 21.25로 추가 향상시키고, 부유물 가우시안을 실제 표면으로 유도하여 gradient explosion을 방지함.

![Table 7](./images/Table%207.png)
> **Table 7. Opacity loss와 입력 크기에 따른 가우시안 사용률.** 해상도가 높아질수록 사용률이 감소하지만(41.8% → 34.1%), 960×540에서는 시야 범위가 넓어지고 뷰 간 겹침이 줄어 감소폭이 미미함(33.3%).

---

## 6. Conclusions
- Long-LRM은 Mamba2와 Transformer의 결합, 토큰 병합, 가우시안 가지치기를 통해 32장 고해상도 이미지로부터 1초 만에 360° 광범위 장면 3D Gaussian 재구성을 피드포워드로 구현한 대형 모델임.
- 광범위 장면에서는 최적화 기반 3D GS와 대등한 렌더링 품질을, 저해상도 sparse-view 설정에서는 기존 피드포워드 GS 방법보다 우수한 성능을 보이며, 2D GS 등 다른 GS 변형과도 호환됨.
  - 광범위한 ablation으로 하이브리드 구조, 토큰 병합, 정규화 항의 효과를 확인함.
