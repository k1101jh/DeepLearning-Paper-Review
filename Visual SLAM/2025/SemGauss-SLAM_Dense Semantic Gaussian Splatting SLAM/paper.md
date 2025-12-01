# SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM


---

- 3D Segmentation
- 3DGS SLAM

---

url:
- [paper](https://arxiv.org/abs/2403.07494) (arXiv 2025)

---
요약

- 문제점
    - 

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
> SemGauss-SLAm은 3D 가우시안 표현에 semantic feature embedding을 통합하여 밀집된 의미 SLAM을 수행  
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

### 3-A 3D Gaussian Semantic Mapping and Tracking

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

### 3-B Loss Functions

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

tracking process

~



### 3-C Semantic-informed Bundle Adjustment
