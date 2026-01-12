# DINOv3


---

- Segmentation

---

url:
- [paper](https://arxiv.org/abs/2508.10104) (arXiv 2025)
- [blog](https://ai.meta.com/dinov3/)

---
요약


---

## Abstract

**Self-Supervised Learning**
- 수동 데이터 주석 작성의 필요성을 없앰
- 방대한 데이터셋과 더 큰 아키텍처로 모델 확장 가능

**DINOv3**
- 데이터셋과 모델 크기 모두를 확장
    - 데이터 준비, 설계 최적화를 통해 이뤄짐
- Gram anchoring 도입
    - 긴 훈련 스케쥴 동안 dense feature map이 열화되는 문제를 해결
- 해상도, 모델 크기, 텍스트와의 정렬에 대한 유연성을 강화하는 사후 전략 사용

**결과**
- 다양한 vision task에서 뛰어난 성능을 내는 고품질 dense feature 생성
- 이전의 self-, weakly-supervised 기반 foundation model을 크게 능가함
- 다양한 자원 제약과 배치 시나리오에 대한 확장 가능 솔루션을 제공

## 1. Introduction

**Self-Supervised Learning**
- 방대한 원시 이미지 집합에 대한 학습을 가능하게 함
- 분포 이동 입력에 강인함
- 강력한 전역 및 국소 특징을 제공
- 물리적 장면 이해를 돕는 풍부한 임베딩 생성
- 특정 하위 작업을 위해 훈련되지 않았기에, 다재다능하고 견고한 범용 feature 제공

![alt text](./images/Fig%201.png)

> **Figure 1.**  
> (a) ImageNet1k(In1k)에서의 linear probing 결과. Fully-Supervised Learning(SL), Weakly-Supervised Learning(WSL), Self-Supervised Learning(SSL) 방법 비교. SSL은 나중에 도입되었음에도 빠르게 발전하여 Imagenet 정확도 고원에 도달.  
> 본 논문에서는 SSL이 고품질 dense feature을 제공한다는 점을 입증  
> DINOv3에서는 dense task에서 WSL보다 현저히 향상되었음. (b) 참고(동급 최고 WSL 모델들의 DINOv3에 대한 상대적 성능).
> 자연 이미지 (c) & 항공 이미지 (d)를 학습한 고해상도 DINOv3 이미지에서 얻은 feature PCA 지도

대량의 데이터로 SSL을 통해 강력한 모델을 생성하는데에는 규모 확장에 어려운 문제가 남아 있음
- Oquab et al. (2024)가 제안한 휴리스틱을 통해 모델 불안정성과 붕괴가 완화되지만, 규모를 확장하면 더 많은 문제가 발생함
    1. label이 지정되지 않은 데이터셋에서 유용한 데이터를 추출하는 방법이 불분명함
    2. 일반적인 학습 방식에서 cosine scheduling을 사용하려면 optimization 기간을 사전에 알아야 함
    3. 초기 학습 후 feature의 성능이 점차 저하됨(patch 유사도 맵을 시각적으로 보면 알 수 있음)
- ViT-Large(3M 파라미터) 이상의 모델로 장시간 학습할 때 발생함

**DINOv3**

- 단일 frozen SSL backbone이 하위작업에서 SOTA를 달성할 수 있는 범용 visual encoder로 작용할 수 있음을 입증
- supervised & metadata 의존 pre-training 전략보다 뛰어난 성능을 발휘

연구 목표
1. task와 domain 전반에 걸쳐 다양한 foundation model 훈련
2. dense features에 대해 SSL 모델의 한계 개선
3. off-the-shelf에서 사용할 수 있는 모델 계열 배포

![alt text](./images/Fig%202.png)

> **Figure 2. 다른 self-supervised 또는 weakly-supervised 모델과의 비교**  
> DINOv3은 AM-RADIO(Heinrich et al. 2025)와 같은 mask annotation priors를 활용하는 모델을 포함한 dense benchmarks에서 다른 모델을 크게 능가함

**세 가지 목표**

1. **Strong & Versatile Foundational Models**

- DINOv3은 모델 크기와 학습 데이터의 확장성을 통해 두 가지 측면에서 높은 수준의 다재다능함을 제공하는 것을 목표로 함
    1. SSL 모델은 frozen 상태로 우수한 성능을 달성해야 함.
        - 이상적으로는 specialized model과 유사한 SOTA 결과를 달성해야 함
        - single forward pass만으로도 여러 작업에서 SOTA 결과를 얻을 수 있음
            - 연산 비용 절감
            - Edge 장치 적용에 필수적
    2. 메타데이터에 의존하지 않는 확장 가능한 SSL 훈련 파이프라인은 많은 과학적 응용을 가능하게 함
        - 다양한 이미지 집합을 pre-training함으로써 SSL 모델은 광범위한 도메인과 작업에 걸쳐 일반화됨
        - 1(d)에서는 고해상도 항공 이미지에서 추출한 DINOv3 feature의 PCA를 통해 도로, 주택, 녹지를 명확히 구분할 수 있음

2. **Gram Anchoring을 통한 Superior Feature Maps**

- DINOv3은 dense feature maps에 상당한 개선을 이룸
    -  고수준 semantic 작업에서 뛰어난 model을 생성하는 동시에 depth 추정이나 3D 매칭과 같은 기하학적 과제를 풀기에 적합한 feature map 생성을 목표로 함
    - 모델은 off-the-shelf 또는 적은 post-processing 만으로 dense feature을 만들어야 함
    - dense 및 global 표현은 방대한 많은 양의 이미지를 훈련할 때 최적화가 특히 어려움
        - 고수준 이해와 dense feature map의 품질이 충돌할 수 있음
        - 대규모 모델과 긴 훈련 일정으로 인한 dense feature들의 붕괴를 초래함
    - Gram Anchoring 전략은 이 붕괴를 효과적으로 완화함
    - DINOv3은 DINOv2보다 훨씬 더 높은 dense feature map을 얻으며, 고해상도에서 깔끔함을 유지함

3. **The DINOv3 Family of Models**

- Gram 앵커링으로 dense feature map의 열화를 해결하면 scaling의 힘을 얻을 수 있음
    - SSL로 더 큰 모델을 학습시키면 성능이 크게 향상됨 (논문에서는 7B DINO 모델 학습)
    - distillation을 통해 knowledge를 더 작은 변형으로 압축
- DINOv3
    - 다양한 자원 제약과 배치 시나리오에 적응할 수 있는 확장 솔루션 제공
    - 증류는 ViT Small, Base, Large, ConvNeXt 기반 아키텍처 등 여러 규모의 모델 변형을 생성
    - ViT-L은 7B 교사와 거의 동일한 성능을 보임
    - 전반적으로 광범위한 벤치마크에서 강력한 성능을 보이며, global 작업에서 경쟁 모델과 동등하거나 능가하는 정확도를 보임
    - 밀도가 높은 예측 작업에서는 크게 우수함

![alt text](./images/Fig%203.png)

**Overview of Contributions**

~


## 2. Related Work


## 3. Training at Scale Without Supervision

DINOv3
- Self-supervised learning의 경계를 넓혀 견고하고 유연한 시각적 표현을 구현
- SSL은 특정 supervision이나 작업에 편향되지 않은 풍부하고 고품질의 시각적 특징을 제공
    - 다양한 하위 응용 분야에 다재다능한 기반을 제공

### 3.1. Data Preparation

- 데이터 크기를 단순히 늘리는것만으로는 모델 품질이 높아지거나 성능이 높아지지 않음
- 데이터 다양성 및 균형 개선에 초점을 맞추거나, 데이터 유용성(실용적인 관련성)에 초점을 맞출 수 있음
- DINOv3 개발에는 두가지 상호 보완적인 접근법을 결합

**Data Collection and Curation**

사전 학습 데이터셋 구축
- 인스타그램의 공개 개시물에서 수집한 방대한 웹 이미지 데이터 풀을 활용하여 구축
- 플랫폼에서 검열을 거침
- 약 170억 장의 초기 데이터 풀 확보
- 이를 사용하여 3개의 데이터셋 part를 생성
    1. 계층적 k-means를 적용하여 구성
        - 이미지 embedding으로 DINOv2를 사용
        - 5단계 클러스터링 사용(200M, 8M, 800k, 100k, 25k)
        - 이후 balanced sampling 알고리즘 적용
        - 16억 8,900만장 이미지(LVD-1689M) 생성됨
            - 웹상에 등장하는 모든 시각적 concepts에 균형 잡힌 coverage를 보장
    2. Oquab et al.(2024)가 제안한 절차와 유사한 검색 기반 관리 시스템 채택
        - 데이터 풀에서 선택한 seed dataset과 유사한 이미지를 가져와 후속 작업에 관련된 시각적 개념을 포함하는 데이터셋을 생성함
    3. ImageNet1k, ImageNet22k, Mapillary Street-level Sequences를 포함한 공개 데이터를 사용
        - 모델 성능 최적화

**데이터 샘플링**

- pre-training 중에 샘플러를 사용해 서로 다른 데이터 part를 혼합
    - 여러 옵션이 존재
        - 각 iteration에서 무작위로 선정된 단일 구성 요소에서 나오는 homogeneous(동질적인) 데이터 배치로 훈련
        - 특정 비율로 모든 구성 요소의 데이터를 조합한 heterogeneous(이질적인) 배치로 훈련
    - Charton & Kempe(2024)는 작은 데이터셋에서 고품질로 이루어진 동질적 배치가 유익함을 관찰
    - 각 반복에서 ImageNet1k 만의 동질 배치 또는 다른 모든 구성 요소의 데이터를 혼합한 이기종 배치를 무작위로 샘플링
        - 본 논문의 훈련에서는 ImageNet1k의 동질 배치가 학습의 10% 차지


**데이터 Ablation(제거)**

- 큐레이션 기법의 영향 평가
    - 각 데이터셋에 대해 모델을 학습시키고 표준 downstream 작업에서의 성능을 비교
    - 효율성을 위해 100만회 대신 20만 iteration 사용
    - 모든 벤치마크에서 단일 큐레이션 기법이 잘 작동하지 않음(표 1 참고)
    - 전체 파이프라인을 통해 두 가지 장점을 모두 얻을 수 있음

### 3.2 Large-Scale Training with Self-Supervision

SSL로 훈련된 모델들은 흥미로운 특성을 보임
- 대부분의 SSL 알고리즘은 아직 더 큰 모델 크기로 확장되지 않음
    - 훈련 안정성 문제 때문일 수 있음
    - 혹은 visual world의 복잡성을 완전히 포착하지 못하는 지나치게 단순한 해결책 때문일 수 있음
- 대규모 학습에서 SSL로 학습된 모델은 반드시 인상적인 성능을 보이지는 않음
    - DINOv2는 예외로, CLIP과 같은 weakly-supervised 모델과 유사한 성능을 보임
- DINOv2를 7B개 매개변수로 확장하려는 시도는 global tasks에서 유망한 결과를 보였지만, dense prediction에서 실망스러운 결과를 보임

**Learning Objective**

- 전역 & 지역 loss를 모두 포함하는 여러 self-supervised objectives를 혼합한 discriminative(판별적) self-supervised 전략으로 모델 훈련
- $\mathcal{L}_{DINO}$: DINOv2의 image-level objective
- $\mathcal{L}_{iBOT}$: patch-level latent reconstruction objective
- 두 objective 모두에서 DINO의 centering 방식을 SwAV(Caron et al. 2020)의 sinkhorn-knopp 방식으로 대체
- 각 objective는 backbone 네트워크 위에 있는 전용 head 출력을 사용하여 계산됨
    - loss 계산 전에 feature에 특화 가능함
- local & global crops의 backbone 출력에 전용 layer normalization을 적용
- 이러한 변경 사항은 학습 후반에 ImageNet KNN 분류를 안정화시키고(+0.2 정확도) 밀집 성능을 향상시킴(ADE20k segmentation에서 +1 mIoU, NYUv2 depth estimation에서 -0.02 RMSE)
- $\mathcal{L}_{Koleo}$: batch 내 feature들이 공간에 균일하게 분포되도록
    - 16개 sample의 작은 batch 단위로 적용되는 Koleo의 분산 구현을 사용
    - 필요에 따라 GPU 활용 가능
- 초기 학습은 다음 loss를 최적화하여 수행

$$
\displaystyle
\begin{aligned}
\mathcal{L}_{Pre} = \mathcal{L}_{DINO} + \mathcal{L}_{iBOT} + 0.1 * \mathcal{L}_{DKoleo}
\tag{1}
\end{aligned}
$$

**Update Model Architecture**

- 모델 크기를 7B 매개변수로 확장
- 해당 hyperparameter을 DINOv2 1.1B 매개변수 모델과 비교(표 2 참고)
- RoPE의 맞춤형 변형을 사용
    - 기본 구현
        - 각 patch에 [-1, 1]로 정규화된 박스 내의 좌표 할당
        - 두 patch의 상대적 위치에 따라 multi-head attention 연산에 bias 적용
    - 해상도, 스케일 및 화면비 변화에 대한 모델의 견고성 향상을 위해 RoPE-box jittering 사용
    - 좌표 박스 [-1, 1]은 [-s, s]로 무작위로 확장됨($s \in [0.5, 2]$)
- DINOv3가 더 상세하고 견고한 시각적 특징을 학습하도록 함

**Optimization**

- 적절한 최적화 시점을 추측하는 것은 불가능
    - 모델 용량과 학습 데이터 복잡성 간의 상호작용을 사전에 평가하기 어려움
- 해결 방법
    - 모든 매개변수 스케줄링을 없애기
    - 일정한 학습률
    - weight decay
    - teacher EMA momentum
- 두 가지 이점
    1. downstream 성능이 계속 향상되는 한 훈련 가능
    2. 최적화 hyper parameter 수가 줄어듦
- 훈련 설정
    - 훈련이 제대로 시작되도록, learning rate와 teature temperature에 대해 linear warmup 사용
    - AdamW 사용
    - 배치 사이즈: 4096 이미지. 256개 GPU에 분산 처리
    - 이미지 당 2개의 global crop과 8개의 local crop을 사용하는 multi-crop 전략으로 모델 학습
        - global crop에는 256x256픽셀 정사각형, local crop에는 112x112픽셀 정사각형 이미지 사용
    - 패치 크기 변경을 통해 DINOv2와 동일한 이미지 당 유효 sequence 길이와 3.7M 토큰의 sequence 길이를 얻음


## 4. Gram Anchoring: A Regularization for Dense Features

- 장기간 훈련은 global benchmark 대비 향상을 이뤄냄
- 하지만 훈련이 진행될수록 dense task에서는 성능이 저하됨
    - feature representation의 patch-level 불일치로 인해 발생

### 4.1 Loss of Patch-Level Consistency Over Training

- 장기간 훈련 동안 전역 지표는 꾸준히 개선되었지만 dense prediction 작업에서는 성능이 눈에 띄게 감소(그림 5 참고)
- 이미지 분류 및 분할 작업 모두에서 iteration 전반에 걸쳐 모델의 성능을 보여줌
    - classification
        - CLS 토큰을 사용해 ImageNet-1k에서 linear classifier을 학습하고 top-1 정확도 보고
    - segmentation
        - Pascal VOC에서 추출한 patch feature에 linear layer을 훈련하고 mIoU를 보고
    - ViT-g와 ViT-7B 모두에서 classification 정확도가 훈련 전반에 걸쳐 단조적으로 향상됨을 관찰
    - 두 경우 모두 약 20만번의 iteration 이후 segmentation 성능이 저하되어 ViT-7B 초기 수준 이하로 떨어짐

patch feature의 품질을 분석하여 patch 간 cosine 유사성 시각화(그림 6)
- 20만 iteration 시 유사도 map은 매끄럽고 잘 국소화됨
- 60만회 이상 반복되면 map이 크게 저하되고, 참조 patch와 유사한 무관한 patch가 많아짐
- patch-level 일관성 상실은 dense task 성능 저하와 연관되어 있음

patch-level 불규칙성은 high-norm patch outlier(Darcet et al. 2024)과는 다름
- register token 통합으로 patch norm은 훈련 내내 안정적으로 유지됨
- 하지만 학습 중에 CLS 토큰과 patch 출력 간의 cosine 유사성이 점차 증가함을 관찰
    - patch feature의 지역성이 줄어든다는 뜻
- 그림 5는 200k와 1M 반복에서 cosine map을 나타냄
- patch feature을 정규화하고 좋은 patch-level 일관성을 보장하면서도 높은 global 성능을 유지하는 새로운 목표를 제안

### 4.2 Gram Anchoring Objective

강한 discriminative(판별) features를 배우는 것과 local consistency 유지 사이에 상대적 독립성이 있음을 확인
- global & dense 성능 간의 correlation 부재에서 관찰됨
- global DINO loss와 local iBOT loss 결합으로 해결 시도
    - 학습이 진행됨에 따라 균형이 불안정하여 global representation이 지배적임

새로운 해결책
- patch-level 일관성 저하를 완화하는 새로운 objective 도입
    - feature에 영향을 주지 않으면서 patch-level 일관성의 품질 강제
- 새로운 loss 함수는 이미지 내 patch feature의 모든 pairwise dot proeuct 행렬인 Gram 행렬에서 작동
- student의 Gram 행렬을 Gram 교사의 matrix로 밀어 붙이고자 함
    - teacher 네트워크의 초기 반복을 통해 Gram teacher 선정
        - 우수한 밀도 특성을 보임
- feature 자체가 아니라 Gram 행렬에 적용함으로써, local feature들은 구조가 동일하게 유지된다면 자유롭게 이동 가능함
- 가정
    - $d$ 차원에서 작동하는 네트워크와 $P$개의 패치가 있다고 가정
    - $X_S$가($X_G$도 마찬가지로) $P \times d$ 크기의 student의 L2-normalized local feature(gram teacher도 마찬가지로)
    - Loss:
    $$
    \displaystyle
    \begin{aligned}
    \mathcal{L}_{Gram} = ||X_S \cdot X_S^T - X_G \cdot X_G^T||_F^2
    \tag{2}
    \end{aligned}
    $$
    - 이 loss는 global crops에서만 계산
        - 훈련 초기에 적용할 수 있지만 효율성을 위해 1M iteration 이후 적용
    - $\mathcal{L}_{Gram}$을 늦게 적용했어도 심하게 손상된(degraded) local feature을 복구할 수 있음을 관찰
    - 성능을 더욱 향상시키기 위해, 1M번의 iteration 마다 Gram teacher을 업데이트
        - Gram teacher가 main EAM teacher과 동일해지도록 함
        - 이 두 번째 학습 단계를 정제 단계라고 부름
        - 이 단계에서 목적 함수 $\mathcal{L}_{Ref}$를 최적화
        $$
        \displaystyle
        \mathcal{L}_{Ref} = \omega_D\mathcal{L}_{DINO} + \mathcal{L}_{iBOT} + \omega_{DK} \mathcal{L}_{DKoleo} + \omega_{Gram} \mathcal{L}_{Gram}
        \tag{3}
        $$

- Gram objective를 적용했을 때 iBOT 손실 함수가 상당히 빠르게 감소(그림 7 참고)
    - 안정적인 Gram teacher 모델이 제공한느 안정성이 iBOT 목적 함수에 긍정적인 영향을 미침
    - 반면, Gram 목적 함수는 DINO 손실 함수에는 큰 영향을 미치지 않음
- Gram 목적 함수와 iBOT 목적 함수가 feature에 유사한 방식으로 영향을 미치는 반면, DINO 손실 함수는 서로 다른 방식으로 영향을 미침

- 새로운 loss 함수 효과는 거의 즉각적으로 나타남(그림 8 참고)
- Gram Anchoring을 도입하면, 처음 10,000번의 iteration 내에 dense 작업에서 상당한 성능 향상을 가져옴
- Gram teacher 업데이트 후 ADE20k 벤치마크에서 눈에 띄는 성능 향상을 확인할 수 있음
- 학습 시간을 늘리면 ObjectNet 벤치마크에서 성능이 더욱 향상됨
- 다른 global 벤치마크에서는 새로운 loss 함수의 영향이 미미하게 나타남

![alt text](images/Fig%208.png)

> **Figure 8. **

### 4.3 Leveraging Higher-Resolution Features

- patch feature의 weighted average는 outlier patches를 smoothing하고 patch-level 일관성을 향상시켜 더 강력한 local 표현을 생성할 수 있음
- 고해상도 이미지를 backbone에 입력하면 더 세밀하고 상세한 feature map이 생성됨
- 이러한 두 가지 관찰 결과를 활용하여 Gram teacher에 사용할 고품질 feature을 계산
    - 일반 해상도의 두 배에 해당하는 이미지를 Gram teacher에 입력
    - 생성된 feature map을 2배 downsampling하고 3차 보간법을 사용하여 student 출력 크기와 일치하는 원하는 smooth feature maps를 얻음
- 그림 9(a)는 256 및 512 해상도 이미지로 얻은 patch feature의 Gram 행렬과 512 해상도에서 2배 downsampling한 feature의 Gram 행렬을 시각화
    - 고해상도 feature의 우수한 patch-level 일관성이 downsampling을 통해 유지되어 더욱 부드럽고 일관성 있는 patch-level 표현이 생성됨
    - RoPE(Rotary Positional Embeddings)(Su et al. 2024)을 채택하여 adaptation(적응) 과정 없이 다양한 해상도의 이미지를 원활하게 처리할 수 있음


down-sampled features의 Gram 행렬을 계산하고 이를 objective 함수 $\mathcal{L}_{Gram}$의 $X_G$ 대신 사용
- 이를 $\mathcal{L}_{HRef}$로 정의
- Gram objective function은 smoothed 고해상도 feature의 향상된 patch 일관성을 student 모델에 효과적으로 반영 가능
    - 이러한 반영은 dense task에서 더 나은 예측 성능으로 이어짐(그림 8, 9 참고)
    - $\mathcal{L}_{Ref}$가 제공하는 이점(ADE20k에서 +2 mIoU)에 더해 추가적인 성능 향상을 가져옴
- Gram teacher 선택 과정을 제거(그림 9(b) 참고)
- 10만개 또는 20만개의 Gram teacher 중에서 선택하는 것은 결과에 큰 영향을 미치지 않음
    - 더 나중에 생성된 Gram teacher을 사용하면 patch-level 일관성이 떨어짐


그림 10은 Gram Anchoring이 patch-level 일관성에 미치는 영향을 정성적으로 보임
- 초기 학습과 고해상도 Gram Anchoring refinement를 통해 얻은 Gram matrices patch feature을 시각화한 것
- 고해상도 refinement 절차를 통해 feature 간 상관관계가 크게 향상되는 것을 확인할 수 있음


## 5. Post-Training

### 5.1 Resolution Scaling


### 5.2 Model Distillation
