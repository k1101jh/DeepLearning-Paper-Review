# Improving Gaussian Splatting with Localized Points Management

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Gaussian Splatting

---

url
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Improving_Gaussian_Splatting_with_Localized_Points_Management_CVPR_2025_paper.html) (CVPR 2025)
- [project](https://happy-hsy.github.io/projects/LPM/)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

3D Gaussian Splatting 모델에서 point 관리가 중요
- 초기 point 분포(예: SfM)는 종종 부적절함
- 일반적으로 Adaptive Density Control(ADC) 알고리즘 채택
    - point 밀도 증가를 위한 view-averaged gradient 크기 thresholding 사용
    - 모든 point 불투명도 초기화
- 이 전략이 복잡하거나 특수한 이미지 영역(예: 투명 영역)을 처리하는데 한계가 있음
- point 밀도 증가가 필요한 모든 3D 영역을 식별하지 못하고 부정적인 영향을 미치는 ill-conditioned points(예: false high opacity로 인한 가림)을 처리할 적절한 메커니즘이 없음

Localized Point Management(LPM)
- point addition 및 geometry calibration 에서 error-contributing zone을 식별할 수 있음
- 이미지 렌더링 오류를 고려한 underlying multiview geometry constraints를 활용하여 이루어짐
- 식별된 구역에서 point densification을 높임
    - 이러한 영역 앞의 point의 불투명도를 재설정
        - 잘못 조건화된 점을 수정할 기회를 만듦
- LPM은 기존 정적 3D 및 동적 4D Gaussian splatting 모델에 최소한의 추가 비용으로 원활하게 통합될 수 있음
- 실험적 평가를 통해 LPM이 다양한 기존 3D/4D 모델을 양적 및 질적으로 향상시키는 효과가 검증됨
- 특히, 정적 3DGS와 동적 SpaceTimeGS의 렌더링 품질을 SOTA 수준으로 향상시키면서 실시간 속도를 유지
- Tanks & Temples 및 Neural 3D Video datset과 같은 도전적인 데이터셋에서도 뛰어난 성능 발휘

## 1. Introduction

Neural Rendering
- 임의의 카메라 포즈에서 photorealistic novel view 합성(NVS)를 가능하게 하는 범용적이고 유연한 접근법
- AR/VR/MR, 로보틱스, 생성 등 다양한 분야에 활용됨
- NeRF(Neural Radiance Field)
    - 복잡한 장면의 방사선 분포를 MLP와 같은 신경망으로 암묵적으로 모델링
    - Geometry, texture, 조명 변화에 대한 수작업 설계가 필요없음
    - ray sampling 기반 렌더링이 계산적으로 비효율적
    - 고해상도 및 대규모 장면 모델링에 확장이 어려움

3D Gaussian Splatting(3DGS)
- 명시적 표현(explicit representation)을 사용하여 모델 최적화를 빠르게 수행
- 실시간 렌더링 가능
- 방법
    1. Structure from Motion(SfM)을 통해 3D Gaussian pointset을 초기화
    2. view reconstruction loss로 point parameter 최적화
    3. 미분 가능한 splatting-based rasterization으로 렌더링 결과 생성
- 초기화된 point 분포가 최적이 아니어서 장면 공간에 point가 부족하거나 과도하게 분포될 수 있음
    - Adaptive Density Control(ADC)와 같은 point 관리 기법 필요
        - 평균 gradient 임계값을 기준으로 point 추적
- ADC의 한계
    - 평균 그래디언트가 낮은 큰 Gaussian 포인트는 여러 뷰에서 자주 등장하여 최적화가 덜 된 point를 놓치는 경우가 많음
    - point가 너무 적게 분포된 영역에서는 충분하고 신뢰할 만한 point를 추가하기 어려움
    - 잘못 최적화된 point는 다른 유효 point를 가려버려 부정확한 깊이 추정으로 이어질 수 있음

Localized Point Management(LPM)
1. 오류 맵 생성
    - 특정 뷰의 렌더링 오류 맵을 계산
2. region correspondence(영역 대응)
    - view 간 feature mapping과 multiview 기하 제약을 이용해 오류가 대응되는 영역 쌍을 찾음
3. 오류 원인 영역 추출
    - 각 대응 영역에서 카메라 view를 따라 원뿔 형태로 광선을 쏘고, 이 광선들의 교차영역을 error source zone으로 정의
4. 국소적 point 관리
    - 각 error source zone마다 두 가지 situation 고려
        1. point가 존재하면 낮은 임계값으로 point 증식을 수행
        2. point가 없으면 새 Gaussian point를 추가
    - 동시에, 이 zone 앞쪽에 위치한 렌더링에 과도한 영향을 주는 고불투명도 point는 불투명도를 리셋해 재조정 기회 제공
5. 밀도 인식 pruning
    - 모델 확장 최소화를 위해 불투명도 기반 density-aware pruning으로 불필요한 point 제거

주요 기여
1. gaussian splatting의 표준 point management 메커니즘 분석을 통해 최적화 방해 요인들을 규명
2. error-contributing 3D zone을 찾아 국소적으로 point densification과 opacity reset을 수행하는 Localized Point Management(LPM) 제안
3. 정적 및 동적 장면 모두에서 다양한 3D/4D Gaussian Splatting 모델의 novel view 합성 성능을 개선했음을 실험적으로 검증

## 2. Related Work

## 3. Method

### 3.1 Preliminaries: 3D Gaussian Splatting

Gaussian Splatting
- EWA Splatting을 기반으로 3D 장면을 ${G_i | i = 1, ..., K}$ 개의 3D Gaussian 점들의 집합으로 모델링한 후 volume splatting으로 렌더링
- 각 gaussian은 다음 식으로 정의됨
$$
\displaystyle
G(x) = e^{-\tfrac{1}{2}(x - \mu)^{T}\Sigma^{-1}(x - \mu)}
$$

~

픽셀 색상 계산
~
$$
\displaystyle
c(p) = \sum_{i=1}^{N} c_i\,\alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)  
$$

~

**Point management**

- SfM 초기화로부터 얻은 Gaussian point들은 공간 분포가 조밀하지 못하거나 과도하게 밀집된 상태로 시작됨

Adaptive Density Control(ADC)
- 각 point $G_i$의 위치 gradient $\frac{\partial L_\pi}{\partial \mu_i}$ 크기를 모든 렌더링 view에 걸쳐 추적한 뒤 평균값 $T_i$를 구함
- gradient $T_i$가 미리 정해진 임계값을 넘으면, 해당 영역을 충분히 표현하지 못했다고 판단하여
    - 큰 Gaussian은 둘로 분할
    - 작은 Gaussian은 복제
- 한계점
    - 단일 임계값 기반으로는 scene geometry 복잡도가 다르게 변하는 모든 영역을 포괄하기 힘듦
    - ill-conditioned point를 처리할 적절한 메커니즘이 부족함(예: 여기저기 분포된 point로 학습하는 동안 잘못 추정된 불투명도 값)


### 3.2 Localized Gaussian Point Management

Localized Point Management(LPM)
- 이미지 렌더링 오류의 guidance를 기반으로 multiview gemoetry 제약을 활용하여 오류를 유발하는 3D point를 식별
- 기존 3DGS 모델에 아키텍처 수정을 필요로 하지 않고 원활하게 통합될 수 있음
- 특정 view에 대한 이미지 렌더링 error map으로 시작함(Fig 2)
- multiview geometry 제약 하에서, referred view의 대응 영역은 feature mapping을 통해 일치됨
- 각 대응 영역 쌍에 대해, 그 영역들을 카메라 view에서 원뿔 형태로 ray를 투사하고 그 교차점을 error source zone으로 식별
- 각 영역 내에서, localized point manipulation을 수행

**Error map generation**

- 3D 공간에서 점 밀도와 geometry calibration이 필요한 영역을 정확하게 위치 지정하기 위해, 3D 가우시안 splatting을 통해 현재 view 이미지를 렌더링하는 과정으로 시작