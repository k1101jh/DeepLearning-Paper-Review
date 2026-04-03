# SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---



---

- 3D Segmentation
- 3DGS SLAM

---

url:
- [paper](https://arxiv.org/abs/2403.07494) (arXiv 2025)
- [github](https://github.com/IRMVLab/SemGauss-SLAM)

---
요약

- 3D Gaussian Semantic 표현을 위해 가우시안에 semantic feature embedding 값을 추가
- mapping에 4가지 loss 사용
    - semantic loss: cross-entropy
    - feature-level loss: DINOv2 feature extractor에서 얻은 feature과 렌더링 reature 간의 L1
    - color loss: L1 loss
    - depth loss: L1 loss
- tracking은 color rloss와 depth loss만 사용
- Semantic informed BA
    - multi-view 제약 & semantic association을 활용하여 BA
    - frame i에서 렌더링 된 semantic feature을 상대 자세 변환을 적용해 co-visible frame j에 투영
    - 위의 j로 투영한 semantic feature과 j에서 렌더링 한 semantic feature을 비교하여 loss 구성
    - RGB & Depth도 마찬가지로 적용

---

## Abstract

SemGauss-SLAM
- 3D 가우시안 표현을 활용한 밀도 높은 semantic SLAM
- 정확한 3D semantic mapping, 견고한 카메라 트래킹, 고품질 렌더링이 가능함
- 3D 가우시안 표현에 semantic feature embedding 결합
    - 환경의 공간적 배치 내에서 의미 정보를 효과적으로 인코딩
- 3D 가우시안 표현 업데이트를 위한 feature level loss를 제안
- semantic informed BA 제안
    - tracking에서 누적 drift 줄이기 위함
    - semantic 재구성 정확도 향상을 위함
    - 3D 가우시안 표현과 카메라 포즈를 공동 최적화
    - 2D feature embedding을 통해 semantic 정보를 효과적으로 인코딩
        - low-drift tracking과 정확한 semantic mapping
- 기존 radiance field-based SLAM 방법보다 우수한 성능을 보임
    - Replica, ScanNet 데이터셋
    - 고정밀 semantic segmentation과 dense semantic mapping

## 1. Introduction

- 대부분의 기존 3DGS SLAM은 RGB map을 얻기 위해 시각적 매핑에 집중
    - 색상 정보만으로는 navigation과 같은 후속 작업에 충분하지 않음
- 현재의 semantic NeRF 기반 SLAM 방법은 누적 드리프트 문제를 겪어 SLAM 정확도가 저하됨
- 이러한 방법들은 새로운 view에서 정확한 semantic segmantation을 달성하는데 어려움
    - 온라인에서 재구성된 3D semantic 표현에서 정밀도가 제한적임

![alt text](./images/Fig%201.png)
> **Figure 1.**  
> SemGauss-SLAM은 3D 가우시안 표현에 semantic feature embedding을 통합하여 밀집된 의미 SLAM을 수행  
> 정확한 semantic mapping을 달성하고, 다른 radiance field 기반 semantic SLAM과 비교해 고정밀 semantic novel view 합성을 가능하게 함  
> semantic embedding을 적용한 3D 가우시안 blob을 시각화하여 semantic gaussian 표현의 공간적 배치를 보임
> semantic mapping은 semantic feature embedding을 통해 시각화됨


- 3DGS와 semantic-informed BA를 기반으로 한 새로운 dense semantic SLAM 시스템 도입
- 높은 정확도의 tracking, dense mapping, 새로운 view에 대한 semantic segmentation이 가능함
- SGS-SLAM
    - 색상을 semantic 표현으로 사용하여 3DGS semantic SLAM 달성
- 제안 방법은 semantic feature embedding을 3D 가우시안에 통합하여 semantic modeling 수행
    - 장면에 대한 보다 구별적이고 포괄적인 이해를 가능하게 함
- 3D 가우시안 표현의 명시적 3D 구조를 활용하여 embedding-based semantic Gaussian 표현은 semantic information의 공간적 분포를 포착하여 고정확도로 novel-view semantic segmentation을 달성할 수 있음
- radiance field-based semantic SLAM에서 누적된 drift를 줄이기 위해
    - co-visible frame 간의 의미적 연관성을 활용
    - 카메라 자세와 3D 가우시안 표현의 공동 최적화를 위한 semantic-informed BA 제안
    - 이는 multi-view 의미론의 일관성을 활용해 제약 조건을 설정함
        - 추적의 누적 drift를 줄이고, semantic mapping의 정밀도 향상

논문의 기여
- SemGauss-SLAM 제안
    - 3D Gaussian 기반 dense semantic SLAM system
    - 정확한 semantic mapping, photo-realistic 재구성, robust tracking을 구현
    - 3D 가우시안에 semantic feature embedding을 통합하여 정밀한 의미론적 장면 표현 구현
    - feature-level loss를 도입하여 3D gaussian 최적화를 위한 higher-level guidance 제공
        - 고정밀 장면 최적화 결과를 얻을 수 있음
- multi-view semantic 제약을 활용해 카메라 포즈와 3D gaussian 표현의 공동 최적화를 위해 semantic-informed BA를 수행
    - low-drift tracking과 정확한 semantic mapping을 달성
- 도전적인 데이터셋에 대해 광범위한 평가 수행
    - 기존 radiance field-based SLAM 보다 mapping, tracking, semantic segmentation, novel-view synthesis에서 우수한 성능을 달성

## 2. Related Works

## 3. Method

![alt text](./images/Fig%202.png)
> **Figure 2. SemGauss-SLAM 개요**  
> 

### 3-A. 3D Gaussian Semantic Mapping and Tracking

**Semantic Gaussian Representation**

- 장면 표현을 위해 특정 properties를 갖는 isotropy-Gaussians 집합 사용
- semantic Gaussian mapping을 달성하기 위해, 각 가우시안에 있는 semantic representation을 위한 새로운 매개변수 "semantic feature embedding"을 도입
    - 이는 환경의 공간적 의미 정보를 포착할 수 있는 간결하고 효율적인 gaussian semantic representation을 가능하게 함
- semantic feature embedding으로 보강된 3D 가우시안 표현은 실시간 mapping을 위한 최적화 과정에서 빠르게 수렴하는 것이 중요
    - 이미지에서 추출한 2D semantic feature을 3D 가우시안의 초기 값으로 전달하여 semantic gaussian 최적화의 수렴을 빠르게 달성
    - 이 과정에서 범용 feature 추출기 DINOv2[35] 사용
        - 이후 semantic network를 구성하는 pretrained classifier이 이어짐
- 각 가우시안은 다음을 포함함
    - 3D center position $\mu$
    - radius $r$
    - color $c = (r, g, b)$
    - opacity $\alpha$
    - 16-channel semantic feature embedding $e$
- 이는 $\alpha$로 곱해진 standard Gaussian equation으로 정의됨

$$
\displaystyle
g(x) = \alpha \exp( - \frac{||x-\mu||^2}{2r^2} )
\tag{1}
$$

**3D Gaussain Rendering**


**Tracking Process**

**Mapping Process**

### 3-B. Loss Functions

semantic scene 표현을 최적화하기 위해
- semantic loss $\mathcal{L}_s$을 구성하기 위해 cross-entropy loss를 사용
- feature-level loss $\mathcal{L}_f$를 도입
    - semantic optimization을 위한 higher-level guidance
    $$
    \displaystyle
    \begin{aligned}
    \mathcal{L}_f = \sum_{p \in P_M} | F_e - E(p) |
    \tag{6}
    \end{aligned}
    $$
    > $F_e$: DINOv2-based feature extractor에서 추출한 feature  
    > $P_M$: rendered image에서 모든 pixel set 표현
    - semantic loss와 비교해서, feature loss는 명시적인 semantic gaussian 최적화를 위한 중간 feature에 명시적인 supervision을 제공
        - 장면에 대해 보다 견고하고 정확한 의미론적 이해가 가능하도록 함
- $\mathcal{L}_c$와 $\mathcal{L}_d$는 L1_loss 사용
- mapping process에서, 3D gaussian 최적화를 위해 모든 렌더링된 픽셀에 대한 loss를 구성
    - RGB loss에 SSIM 항 추가
$$
\displaystyle
\begin{aligned}
\mathcal{L}_{mapping} = \sum_{p \in P_M} (\lambda_{f_m}\mathcal{L}_f(p) + \lambda_{s_m}\mathcal{L}_s(p) + \lambda_{c_m}\mathcal{L}_c(p) + \lambda_{d_m}\mathcal{L}_d(p))
\tag{7}
\end{aligned}
$$

- tracking process에서 과도하게 제한된 loss 함수를 사용 시
    - 카메라 포즈 정확도 저하 가능
    - 처리 시간이 증가할 수 있음
- tracking용 loss 함수는 RGB loss와 depth loss의 가중 합만을 기반으로 구성
$$
\displaystyle
\begin{aligned}
\mathcal{L}_{tracking} = \sum_{p \in P_T} (\lambda_{c_t}\mathcal{L}_c (p) + \lambda_{d_t}\mathcal{L}_d (p))
\end{aligned}
$$
> $P_T$: 3D Gaussian map의 잘 최적화된 부분을 렌더링 했을 때 픽셀. visibility silhouette $Sil(p)$가 0.99 이상인 부분

### 3-C. Semantic-informed Bundle Adjustment

기존의 radiance field-based semantic SLAM 시스템
- 최신 input RGB-D frame을 활용하여 포즈 추정을 위한 RGB & depth loss를 계산
- 이 SLAM 시스템의 장면 표현은 추정된 포즈와 최신 프레임을 사용하여 최적화됨
- 단일 프레임 제약만으로 포즈 최적화 의존 시
    - 전역 제약 조건이 없어 추적 과정에서 누적 드리프트 발생 가능
- 장면 표현을 최적화하기 위해 단일 프레임 정보만 사용 시
    - 의미론적 수전에서 장면의 전역적 일관성 없는 업데이트 발생 가능

**semantic-informed Bundle Adjustment**
- 다중 view 제약과 semantic association을 활용하여 3D 가우시안 표현과 카메라 자세 공동 최적화
- multi-view 의미론적 일관성을 활용하여 제약 조건 설정
    - 렌더링 된 semantic feature를 추정된 상대 자세 변환 $T_i^j$를 이용해 co-visible frame $j$ 에 warp 됨
    - $\mathcal{L}_{sem}$을 얻기 위해 프레임 $j$에서 렌더링된 semantic feature $\mathcal{G}(T_j, e)$ 로 loss를 구성
    $$
    \displaystyle
    \begin{aligned}
    \mathcal{L}_{BA-sem} = \sum_{i=1}^{N-1} \sum_{j=i+1}^N (|T_i^j \cdot \mathcal{G}(T_i, e) - \mathcal{G}(T_j, e)|)
    \tag{9}
    \end{aligned}
    $$
    > $\mathcal{G}(T_i, e)$: 카메라 포즈 $T_i$를 사용한 3D gaussian $\mathcal{G}$의 semantic embedding
    - 기하학&시각적 일관성 달성을 위해 rendered RGB와 depth를 co-visible frames에 warp하여 loss 구성
    $$
    \displaystyle
    \begin{aligned}
    \mathcal{L}_{BA-rgb} = \sum_{i=1}^{N-1} \sum_{j=i+1}^N (|T_i^j \cdot \mathcal{G}(T_i, c) - \mathcal{G}(T_j, c)|)
    \\
    \mathcal{L}_{BA-depth} = \sum_{i=1}^{N-1} \sum_{j=i+1}^N (|T_i^j \cdot \mathcal{G}(T_i, d) - \mathcal{G}(T_j, d)|)
    \tag{10}
    \end{aligned}
    $$
- 전체 loss
$$
\displaystyle
\begin{aligned}
\mathcal{L}_{BA} = \lambda_e \mathcal{L}_{BA-sem} + \lambda_c \mathcal{L}_{BA-rgb} + \lambda_d \mathcal{L}_{BA-depth}
\tag{11}
\end{aligned}
$$

## 4. Experiments

### 4-A Experimental Setup

**Datasets**

- Replica[36]
    - 8개 scene 사용
- ScanNet[37]
    - 5개 scene 사용

**Metrics**

- reconstruction
    - Depth L1(cm)
- tracking accuracy
    - ATE RMSE(cm)
- rendering performance
    - PSNR(dB)
    - SSIM
    - LPIPS
- semantic segmentation
    - mIoU(%)

![alt text](./images/Fig%203.png)
> **Figure 3. 렌더링 품질 정성적 비교**  
> Replica와 ScanNet 데이터셋에서 5개의 scene을 선택해서 시각화  
> 디테일은 빨간색 박스로 표시  
> 제안 방법은 풍부한 텍스처 정보가 있는 영역에서 사실적인 렌더링 품질과 더 높은 완성도를 달성

![alt text](./images/Fig%204.png)
> **Figure 4. Replica의 3개의 scene에 대해 semantic novel view 정성적 비교**

**Baselines**
- dense visual SLAM
    - NeRF-based SLAM[23, 26, 25, 27, 24]
    - 3DGS-based SLAM[12]
- dense semantic SLAM
    - NeRF-based semantic SLAM[7, 8, 9]
    - SGS-SLAM[16]

**Implementation Details**
- NVIDIA RTX 4090 GPU
- 8프레임마다 mapping
- weighting coefficients
    - mapping
        - $\lambda_{f_m}$: 0.01
        - $\lambda_{s_m}$: 0.01
        - $\lambda_{c_m}$: 0.5
        - $\lambda_{d_m}$: 1
    - tracking
        - $\lambda_{c_t}$: 0.5
        - $\lambda_{d_t}$: 1
    - semantic-informed bundle adjustment
        - $\lambda_{e}$: 0.004
        - $\lambda_{c}$: 0.5
        - $\lambda_{d}$: 1


> **Table 1. SLAM 정확도와 렌더링 품질 비교**  
> Replica의 8개 scene에 대해 평균
![alt text](./images/Table%201.png)

### 4-B Experimental Results

**SLAM and Rendering Quality Results**

제안 방법은 모든 지표에서 가장 높은 정확도 달성. 재구성 정확도는 최대 35%의 상대적 향상을 보임(Table 1 참고)
- semantic-informed bundle adjustment의 도입 덕분
    - 여러 카메라 포즈와 scene 표현을 공동 최적화 달성을 위해 multi-view 제약 도입
- real-world dataset ScanNet[37]에서 우리의 방법이 baseline method보다 뛰어남을 보임(Table 2 참고)
    - 결과의 신뢰도를 위해, 각 scene은 5번의 독립적인 실행으로 테스트되고 평균됨

> **Table 2. ScanNet 데이터셋의 tracking metric RMSE(cm) ↓ 정확도 비교
![alt text](./images/Table%202.png)

**Novel View Semantic Evaluation Results**

- 제안 방법은 semantic novel view 합성에서 SNI-SLAM[7]보다 최대 49%의 상대적인 향상을 보임
- 하나의 scene에 대해, 평가를 위해 무작위로 100개의 새로운 viewpoint를 선택
    - mIoU는 splatted semantic label과 GT label 간에 계산됨
- semantic 표현을 위해 gaussian feature embedding을 도입
    - 제안 방법은 연속적인 semantic modeling이 가능
    - 이 모델은 semantically 일관된 scenes 생성에 중요.
    - 불일치를 발생시키는 sharp transitions(급격한 전환)를 줄임
    - novel viewpoints의 정확한 semantic representation을 보장

> **Table 3. Replica의 semantic novel view 합성 성능의 정량적 비교. mIoU(%)**
![alt text](./images/Table%203.png)

> **Table 4. Replica의 4개 scene의 Input view의 semantic segmentation 성능. mIoU(%)**
![alt text](./images/Table%204.png)

**Semantic Segmentation Results**

- [9, 8, 16, 7]은 감독을 위해 GT 라벨 사용
    - 공정한 결과를 위해, **(GT)**로 표시한 GT 라벨을 사용한 결과 제시
- 제안 방법은 기존의 radiance field-based semantic SLAM 방법보다 우수한 성능을 보임
    - semantic feature embedding을 3D 가우시안에 통합하여 의미론적 표현을 풍부하게 함
    - semantic feature-level loss를 통해 semantic optimization을 direct guidance하는데 기여
    - 제안한 semantic-informed BA는 여러 co-visible frames를 활용해 고정밀 semantic representation을 위한 전역적으로 일관된 semantic map을 구축
        - 높은 semantic 정밀도에 기여

**Visualization**

- Fig 3은 4개의 scene에서 흥미로운 영역을 색상이 있는 box로 표현하여 렌더링 품질 비교
    - 자주 관측되지 않는 영역(소파의 옆 부분 또는 바닥)은 다른 방법의 경우 저품질로 재구성하거나 구멍을 남김
    - 제안 방법
        - 3D 가우시안 표현의 고품질 렌더링 능력을 활용
        - semantic-informed BA를 도입하여 상세하고 완전한 기하학적 재구성 결과 달성
            - multi-view 의 일관성을 향상하여 재구성된 장면 기하가 모든 관측 시점에서 정확히 정렬되도록 함
- 제안 방법은 baseline SNI-SLAM[7]에 비해 novel view semantic segmentation 정확도가 우수함(Fig 4 참고)
    - SNI-SLAM은 매핑 과정에서 관측되는 빈도가 적어 novel view 합성에 segmenting ceilings(천장 segmentation?)에 어려움을 겪음
        - ceiling feature이 semantic 장면 표현에서 부적절하게 모델링 됨
    - 제안 방법은 semantic-informed BA를 도입하여 multi-view 제약을 구성. sparse하게 관찰되는 영역이 co-visible frames에서 얻는 제한된 정보를 효과적으로 활용할 수 있게 하여 정확한 semantic 재구성을 위한 충분한 제약을 확립함

![alt text](./images/Fig%205.png)

> **Figure 5. Replica의 두 개 scene에 대한 feature-level loss 제거의 semantic rendering 결과 및 GT 라벨**


> **Table 5. Replica에 대한 기여 사항의 제거 연구**

![alt text](./images/Table%205.png)

### 4-C Ablation Study

**Feature-level Loss**
- feature-level loss를 결합하면 semantic segmentation 성능이 향상됨
    - tracking과 geometric 재구성에 약간의 영향을 줌
    - feature level loss는 semantic features 최적화에만 영향을 주고 geometry나 pose estimation에는 영향을 주지 않음
    - feature level loss를 활용하면 boundary segmentation이나 작은 객체의 finer segmentation이 향상됨
        - feature loss가 장면 표현이 feature space 내에서 고차원적이고 직접적인 정보를 포착하도록 강제함
        - 장면 내 복잡한 세부 사항과 미묘한 변화를 구별할 수 있게 함

**Semantic-informed BA**
- 재구성, semantic segmentation이 향상됨
    - multi-view 제약에 의해 결정된 카메라 포즈와 scene 표현의 공동 최적화 덕분
    - semantic 제약 부족 색상과 깊이 제약 부재에 비해 추적 성능이 크게 저하됨
    - multi-view semantic 제약이 여러 관점에서 semantic 일관성을 유지하기 때문에 더 포괄적이고 정확한 정보를 제공
    - semantic 제약이 없으면 semantic 정밀도가 떨어질 수 있음
    - color, depth 제약이 없으면 재구성 정확도가 떨어질 수 있음


> **Table 6. Replica에서 semantic inform BA 제거 연구**  
> (w/o semantic): $\lambda_{BA-sem}$ 제거  
> (w/o RGB and depth): $\lambda_{BA-rgb}, \lambda{BA-depth}$ 제거

![alt text](./images/Table%206.png)

## 5. 결론

**SemGauss-SLAM**
- 3D 가우시안 표현을 활용한 dense semantic SLAM system
- dense visual mapping, 견고한 카메라 tracking, 전체 장면의 3D semantic mapping이 가능함
- dense semantic mapping을 위한 gaussian semantic 표현을 위해 3D 가우시안에 semantic feature embedding을 통합
- 3D 가우시안 장면 최적화를 위한 feature-level loss 제안
- multi-view semantic 제약을 설정하여 카메라 자세와 3D 가우시안 표현의 공동 최적화를 가능하게 하는 의미론적 기반 BA를 도입
    - low-drift tracking과 정밀한 매핑을 가능하게 함