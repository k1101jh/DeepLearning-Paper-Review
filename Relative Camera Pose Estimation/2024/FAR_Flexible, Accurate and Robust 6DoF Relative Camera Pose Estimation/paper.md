# FAR: Flexible, Accurate and Robust 6DoF Relative Camera Pose Estimation

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Relative Camera Pose Estimation
---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Rockwell_FAR_Flexible_Accurate_and_Robust_6DoF_Relative_Camera_Pose_Estimation_CVPR_2024_paper.html) (CVPR 2024)
- [project](https://crockwell.github.io/far)
- [github](https://github.com/crockwell/far)
---
- **Authors**: Chris Rockwell, Nilesh Kulkarni, Linyi Jin, Jeong Joon Park, Justin Johnson, David F. Fouhey
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [3. 접근 방법](#3-접근-방법)
  - [3.1 Approach Outline](#31-approach-outline)
  - [3.2 Pose Transformer](#32-pose-transformer)
  - [3.3 Prior-Guided Robust Pose Estimator](#33-prior-guided-robust-pose-estimator)

---

## ⚡ 요약 (Summary)
**이미지 사이의 상대 카메라 자세 추정**
- 대응 관계를 찾고 기본 행렬을 해결하는 방법
    - 높은 정확도 제공
- 신경망을 사용하여 자세 직접 예측
    - 겹침이 제한될 때 더 강력
    - 절대 변환 스케일(absolute translation scale)을 추론할 수 있음
    - 정확도가 감소함

**FAR**

![alt text](./images/Fig%203.png)

1. dense feature을 입력으로 받음
2. Pose Transformer에서 6DoF pose $\text{T}_t$와 이에 대한 relative weight $\text{w}$도 예측
3. $\text{T}_t$와 solver로 예측한 pose $\text{T}_s$, weight $\text{w}$를 결합하여 포즈 추정값 $\text{T}_1$을 얻음
4. $\text{T}_1$은 solver의 prior으로 사용되며 업데이트 된 solver 출력 $\text{T}_u$를 생성
5. 이는 $\text{w}$를 통해 $\text{T}_t$와 결합되어 최종 출력 $\text{T}$를 얻음

- 이 방법은 network가 데이터 상태에 따라 다르게 행동하도록 학습할 수 있게 함
- 많은 고품질 correspondence가 있는 경우
    - 고전적인 solver가 일반적으로 정확. prior은 solver에 미치는 영향이 거의 없어서 solver의 출력에 크게 의존(낮은 $\text{w}$)하도록 학습
- 적고 저품질의 correspondence인 경우
    - solver 성능이 떨어지므로 prior은 solver 출력에 강한 영향을 미치도록 transformer 예측(높은 $\text{w}$)에 더 의존하도록 함

### 문제점 & 한계점
- 딥러닝 기반 포즈 추정 방법과 기존 포즈 추정 방법 같이 사용
    - 파이프라인이 긺
- 3D 공간 정보 활용 부재
- Feature Extractor에 의존
- 동적 객체에 대한 대처가 부족함
- 느린 속도. 초당 3.3 iteration(1080Ti)

---

## 📖 Paper Review

## Abstract

**이미지 사이의 상대 카메라 자세 추정**
- 대응 관계를 찾고 기본 행렬을 해결하는 방법
    - 높은 정확도 제공
- 신경망을 사용하여 자세 직접 예측
    - 겹침이 제한될 때 더 강력
    - 절대 변환 스케일(absolute translation scale)을 추론할 수 있음
    - 정확도가 감소함

**제안 방법**
- 두 방법의 장점 결합
- 정확하고 강건한 결과를 보임
- 변환 scale을 정확하게 추론
- Transformer 기반
    1. 해결된 pose estimation과 학습된 pose estimation 간의 균형을 배움
    2. 해결자(solver)을 안내하는 사전 정보를 제공
- Matterport3D, InteriorNet, StreetLearn, Map-free Relocalization에서 6DoF 포즈 추정에서 SOTA 성능을 보임

## 1. Introduction

![alt text](./images/Fig%201.png)

> **Figure 1. 정확하고 강건한 6DoF 포즈 추정**  
> - correspondence 예측 + solver 방법(LoFTR[70], RANSAC[22])
>   - 중간 정도의 회전에 대해 정밀한 출력을 생성
>   - 큰 회전(좌)에 대해 강건하지 않음
>   - translation sclae을 생성하지 않음
> - Learning-based 방법(LoFTR & 8-Point ViT[62] head)
>   - scale(우)를 생성하고 더 강건함
>   - 정밀도가 떨어짐
>
> FAR은 정밀하고 견고한 예측을 위해 두 가지를 활용. scale도 예측함

**Relative camera pose estimation**
- 대응 관계를 추정하고 이를 해결하여 포즈를 추정
    - 종종 sub-degree 에러 발생
    - large view 변환(Fig 1. 왼쪽)이 발생하는 경우 잘 작동하지 않음
    - Fundametal 또는 Essential matrix를 계산하기 때문에 scale을 복구할 수 없음
- 포즈를 직접 계산
    - 정확하지는 않음
    - 더 강건하고 translation scale을 제공(Fig 1. 왼쪽 및 오른쪽)

**제안 방법**
- 두 방법을 기반으로 함
- 종종 두 방법보다 나은 결과 제공
- 학습된 correspondence 예측을 입력으로 사용해서 학습된 포즈 추정과 해결자(solver)를 결합
- 밀집 feature 또는 correspondence를 처리할 수 있는 Transformer을 선택
- 유연함
    - correspondence와 feature backbone에 무관함
- 정확함
    - correspondence-based방법과 정밀도 일치
- 강건함
    - 학습된 포즈 방법의 회복성을 기반으로 구축

**FAR**
- 학습 기반 방법과 solver 기반 방법이 서로 개선될 수 있도록 함
- 학습된 예측은 solver 출력보다 강건함
    - solver의 편향을 조정하는 prior로 사용됨
- solver 출력은 학습된 예측보다 정확함 (성공 시)
    - 최종 출력을 형성하기 위해 Transformer 예측과 결합(solver 출력이 성공할 때)
- 예측은 Transformer에 의해 예측된 가중치를 통해 결합됨
    - Transformer가 각 방법이 효율성에 따라 어느 방법에 더 의존하도록 학습할 수 있음

**FAR 분석**
- 좋은 input correspondences 개수에 따라 오류 측정
- correspondences가 많은 경우
    - solver은 정확하여 prior을 개선할 여지가 없음
- correspondence가 적은 경우
    - solver 성능이 줄어들지만, 학습된 prior을 사용하여 완화 가능
- 학습된 가중치는 강건성에 기여
    - Transformer은 대응 수가 많을 경우 solver 출력을 사용하고, 적을 때는 regressor을 더 많이 사용(Figure 2. 오른쪽)
- correspondence가 많은 경우 희생하지 않으면서 correspondence가 적을 때 큰 이득을 얻는 방법

**FAR 실험 및 성능 분석**
- 이론적 강건성 평가
    - 정확한 correspondences에서 시작해서 noise 및 outlier을 점진적으로 추가
- 네 개의 데이터셋에서 성능 평가
    - 실내 데이터: Matterport3D, InteriorNet
    - 실외 데이터: StreetLearn, Map-free Relocalization
    - 제안 방법은 SOTA 혹은 그 이상의 수준 달성
- ablation 실험
    - 다양한 correspondence와 feature estimation backbone에 적용
- 데이터셋의 크기가 모델의 행동에 미치는 영향을 연구

## 2. Related Work

![alt text](./images/Fig%202.png)

## 3. 접근 방법

**목표**
- 두 개의 겹치는 이미지를 기반으로 상대 카메라 포즈, translation scale 추정
- 6DoF 자세는 $\text{T} \in \text{SE(3)}$으로 매개변수화 가능
    - $\text{R} \in \text{SO(3)}$ 및 $\text{t} \in \mathcal{R}^3$ 포함
- translation scale 예측에 초점
    - correspondence만으로는 해결 불가
    - 3D reconstruction이나 neural rendering 등 실제 응용 프로그램에 유용
- two-view의 경우 image collection을 사용하는 응용 프로그램을 용이하게 함(동영상을 사용하는 프로그램 등? neural rendering?)
- 카메라 내부 파라미터는 알고 있다고 가정

**FAR**
- 기존 두 방법의 강점을 융합
    - 학습된 correspondence 예측 후 robust solver 적용
    - end-to-end 포즈 추정
- 두 방법보다 결과가 나쁘지 않으며 종종 더 좋음
- 유연함
    - 최소한의 수정으로 기존의 방법에 적용 가능

### 3.1 Approach Outline

![alt text](./images/Fig%203.png)

> **Figure 3. Overview**  
> 밀집된 feature과 correspondence를 고려하여 FAR의 transformer과 classical solver을 통해 카메라 자세 생성  
> 1. solver이 $\text{T}_s$ 생성  
> 2. pose transformer은 가중치 w를 사용해서 $\text{T}_s$를 자신의 예측 $\text{T}_t$와 평균내어 Round 1 포즈 $\text{T}_1$을 생성  
> 3. $\text{T}_1$는 classic solver의 prior로 제공되어 $\text{T}_u$ 생성  
> 4. $\text{T}_t$의 추가 추정 및 가중치 $w$와 결합되어 최종 결과 $\text{T}$ 생성  
>
> correspondence가 적은 경우 $\text{T}_1$은 solver의 출력을 돕고, 네트워크는 Transformer 예측에 더 많은 비중을 두도록 학습됨
> correspondence가 많은 경우, solver 출력이 종종 좋기 때문에 네트워크는 주로 solver 출력에 의존


**접근 방식**
1. dense feature을 입력으로 받음
2. Pose Transformer에서 6DoF pose $\text{T}_t$와 이에 대한 relative weight $\text{w}$도 예측
3. $\text{T}_t$와 solver로 예측한 pose $\text{T}_s$, weight $\text{w}$를 결합하여 포즈 추정값 $\text{T}_1$을 얻음
4. $\text{T}_1$은 solver의 prior으로 사용되며 업데이트 된 solver 출력 $\text{T}_u$를 생성
5. 이는 $\text{w}$를 통해 $\text{T}_t$와 결합되어 최종 출력 $\text{T}$를 얻음

- 이 방법은 network가 데이터 상태에 따라 다르게 행동하도록 학습할 수 있게 함
- 많은 고품질 correspondence가 있는 경우
    - 고전적인 solver가 일반적으로 정확. prior은 solver에 미치는 영향이 거의 없어서 solver의 출력에 크게 의존(낮은 $\text{w}$)하도록 학습
- 적고 저품질의 correspondence인 경우
    - solver 성능이 떨어지므로 prior은 solver 출력에 강한 영향을 미치도록 transformer 예측(높은 $\text{w}$)에 더 의존하도록 함
- 입력 feature 및 correspondence에 대해 무관한 방법

### 3.2 Pose Transformer

**Transformer의 목표**
1. 두 장의 wide-baseline 이미지(서로 다른 시점에서 찍은 이미지) 간의 6DoF 상대 카메라 포즈 $\text{T}_t$ 추정
2. weight $\text{w} \in [0, 1]$ 추정
    - transformer 예측과 solver 예측의 가중치
- 입력
    - 2D correspondence matches $\mathcal{M} = \{(\text{p, q})\} | \text{p, q} \in \mathcal{R}^2$
    - dense 2D image-wise features $f_i, f_j$ (선택적)

- 최종 출력: Transformer pose $\text{T}_t$와 solver pose $\text{T}_s$의 가중치 $\text{w}$로 조합된 선형 결합
- 가중치는 translation $w_t$ 와 rotation $w_r$로 나눠짐
    - Transformer가 두 문제애 대해 서로 다른 신뢰도를 가질 수 있도록 함

두 가지 문제점
- 회전의 선형 결합은 종종 rotation matrix가 아님
    - Zhou et al [84]의 6D 표현 사용 후 Gram-Schmidt 정규화를 통해 유효한 회전으로 변환
- solver translation은 scale이 없음
    - Transformer의 translation magnitude $||\text{t}_t||$로부터 sclae translation $\text{t}_s$를 사용
    - 선형 결합 이전에 solver 출력에 스케일 보정

$\text{t}_s$와 $\text{t}_{tf}$의 각도를 평균화한 다음 정규화된 예측에 scale을 적용하는 방법과 비교했을 때 훈련을 안정화함

최종 공식:

$$
\displaystyle
\begin{aligned}

& \hat{R} = w_r R_t + (1 - w_r) R_s \\
& \hat{t} = w_t t_t + (1 - w_t) \|t_t\| t_s & (1)

\end{aligned}
$$

**Transformer backbone**

입력 종류에 따라 두 가지 아키텍처 사용
- modified ViT: feature을 사용 가능한 경우
    - dense features를 생성하는 correspondence 또는 regression 기반 방법과 같이 사용 가능
- Vanilla Transformer: correspondences만 가능한 경우  
    - correspondence만 출력하는 방법에도 적용 가능

 각각의 경우, Transformer은 두 개의 MLP head에 입력으로 사용되는 feature $f_o$ 생성

**8-Point ViT**
- 입력: 두 이미지의 pairwise dense features $f_i, f_j$
- 출력: feature vector $f_o$
- 구성:
    - LoFTR[70] self-attention과 cross attention layer
    - 이후 8-Point ViT cross-attention layer
- 각 네트워크는 포즈 추정을 위한 좋은 feature 생성을 목표로 함

**Vanilla Transformer**
- 입력: correspondences 집합 $\mathbb{M} = \{(\text{p, q})\} | \text{p, q} \in \mathbb{R}^2$ 및 연관된 descriptors(선택적으로)
- 출력: feature set $f_o$
- 구성: 
    - $N$개의 vanilla Transformer encoder
    - correspondences와 descriptors를 입력으로 사용
    - K밴드로 Sin 함수 방식으로 correspondences를 인코딩한 후, Transformer에 입력되는 $c$ 크기로 선형 매핑
    - 각 correspondence point의 descriptive features이 사용 가능한 경우, 차원 $d < c$의 특징은 선형 레이어에 의해 $\frac{c}{4}$로 매핑됨
        - $\frac{3c}{4}$로 선형 매핑된 correspondence 위치에 연결됨

- dense features $f$를 입력으로 사용할 수도 있음
    - 네트워크가 두 이미지의 joint feature encoding을 생성하면 Transformer을 두 저해상도 feature에 positional encoding 없이 직접 적용 가능
    - Arnold et al[2]의 regression model 사용

**Regression MLP**
- Transformer features $f_0$을 두 개의 hidden layer을 사용하여 $R \in \mathbb{R}^6, t \in \mathbb{R}^3$로 매핑

**Gating MLP**
- 입력:
    - Transformer features
    - Regression MLP 예측
    - classical solver 예측
    - solver output의 inlier correspondences 개수(여러 임계값 사용)
- 예측값들과 inlier 수는 정규화 된 후 scalar features로 입력됨
- inlier correspondence 수는 solver pose estimate $\text{T}_s$ 성능과 높은 상관관계를 가짐
- 두 개의 hidden layer가 있으며 Sigmoid로 끝나 $w_t, w_r \in (0, 1)$을 생성

### 3.3 Prior-Guided Robust Pose Estimator

learning based 방법이 solver을 돕는 방법
- RANSAC과 같은 search-based solver의 성능은 모델 공간 내에서 적절한 가설을 샘플링하고 점수화 하는 방식에 달림
- 점수화 함수: 가설 하에 데이터의 확률 측정
- 이러한 solver은 correspondences 셋으로 포즈를 추정할 때 사용됨
    - 불확실한 셋으로 correspondence 추정을 수행하면 강인하지 않음
- 아이디어: 예측된 포즈 추정값 $\text{T}_1$을 사용하여 데이터가 부족한 시나리오에서 검색 및 점수 함수 모두에게 영향을 미치기

RANSAC 기반 알고리즘에서 sampling과 selection을 개선하기 위해 학습을 사용하는 연구들에서 영감을 얻음
- learning-based 모델에서 추정값을 재활용하고 이 추정값을 간단하게 도입
    1. 초기 추정 포즈 $\text{T}_1$이 검색 함수를 수정하여 $\text{T}_1$에 더 가까운 더 많은 가설을 샘플링하도록 함
    2. $\text{T}_1$까지의 거리를 고려하여 scoring function을 수정하고 inlier 수를 함께 고려

**RANSAC Preliminaries**
- correspondence로부터 자세 추정을 위한 전형적인 접근 방식:
    - model fitting에 무작위 샘플 일치(및 변종)을 적용(예: RANSAC, USAC, MAGSAC)하여 model fitting(예: 5, 7, 8-point 알고리즘)
    - 이러한 방법은 inlier 임계값(soft 및 hard)을 위해 Sampson Error와 같은 Epipolar 거리 개념을 사용
    - 2D correspondence 매치 $\mathbb{M} = {(\text{p, q})}|\text{p, q} \in \mathbb{R}^2$ 집합이 주어지면, 최소한의 점 집합을 무작위로 샘플링하여 n-point 알고리즘을 통해 모델 H를 맞춤
    - scoring function은 고정된 임계값 $\sigma$보다 Sampson Error가 작은 inlier의 수를 계산
    - 주어진 가설 $\text{H}$와 correspondence 집합 $\text{M}$에 대해 $\text{E(p, q|H)}$는 $\text{H}$ 하에서 점 $\text{p}$와 $\text{q}$ 간의 Sampson Error
    - 점수 함수: $(\text{H}) = \sum_{(p, q) \in \mathbb{M}} \mathbb{1}[E(p, q|H) < \sigma]$
    - 샘플링은 $N$회까지 반복되거나 중지 heuristics가 충족될 때까지 반복
    - 가장 높은 점수를 갖는 모델(가장 많은 inlier을 갖는 모델)이 선택됨
    - MAGSAC, MAGSAC++와 같은 연구들은 점수 함수를 개선하여 더 나은 성능을 보임
    - 간단함을 위해 고전 RANSAC에서 인기있는 임계값 기반 함수로 설명 진행

**Limitations in Few-Correspondence Case**
- inlier을 세는 heuristic 점수는 특히 low-correspondences 경우에 효과적이지 않음[5]
- correspondences 수가 모델을 최소한으로 정의하는 데 필요한 점의 수의 작은 배수에 불과할 때 알고리즘을 신뢰할 수 없음
    - 9개의 점에서 보정된 카메라를 사용하여 포즈 복구를 수행한다고 가정 시 5개가 inlier
    - 실제 모델은 5개의 inlier가 있으며, 샘플링된 다른 가설도 5개의 inlier 존재. 따라서 아무 가설이나 선택될 수 있음

**Prior-Guided Estimator**
- learning based 예측 통합
- $\Beta(\cdot|\text{T}_1)$ 함수를 사용하여 네트워크의 예측 하에 가설의 likelihood를 추정하는 prior 모델을 통합함으로써 실행
- $\Beta(\text{H}|\text{T}_1)$ 함수는 $\text{T}_1$ 하에서 가설화된 모델 $\text{H}$의 log probability를 측정
- 회전과 translation의 확률을 추정하고 가중치를 부여하는 것은 어렵다고 생각  
-> 모델이 고정된 grid point 집합이 어떻게 변환(transform)시키는지 비교
- $\text{T}_1$에 의해 변환된 고정된 grid point 집합과 $H$에 의해 변환된 동일한 고정된 grid point 집합 사이의 평균 제곱 거리의 negative를 측정 (부록 참조)

$\beta$ 함수가 점수를 변경하는 방법
- 수정된 점수 함수는 $\text{T}_1$ 하에서 가설 $\text{H}$의 likelihood를 측정하면서 $\text{H}$ 하에서 데이터의 likelihood도 측정

$$
\displaystyle
\begin{aligned}

& \text{score}(\text{H}) = \alpha \beta(\text{H}|\text{T}_1) + \sum_{(p, q) \in \mathbb{\text{M}}} \mathbb{1}[\text{E}(\text{p, q}|\text{H}) < \sigma]
& (2)

\end{aligned}
$$

- $\Beta$ 사전 함수에 따라 가설의 확률과 $\text{H}$ 아래의 데이터 $\mathbb{M}$의 확률의 (log) 곱
- 스칼라 $\alpha \in \mathbb{R}$로 prior을 가중
- 두 가설이 비슷한 수의 inlier을 가질 때 prior이 애매한 경우를 타협
- $|\mathbb{M}|$이 커질수록 영향력 감소
- $|\mathbb{M}| \rightarrow \infty$에 가까워질수록 prior의 영향은 완전히 사라짐
- correspondence가 적고 편향되지 않은 가설이 좋지 않을 때 큰 효과를 가짐
- correspondence가 많을 때는 작은 영향을 미침

**Sampling Good Hypotheses**
- 포인트를 무작위로 샘플링하고 $\text{H}$를 추정하는 것은 $\text{T}_1$모델과 일치하는 가설로 이어질 가능성이 낮음
- 일관된 가설을 샘플링 할 확률을 높이기 위해 최소 subset을 샘플링
- correspondences에 따라 가중치를 부여하여 이를 달성
    - 이는 $ w(\text{p}, q) = \exp\left(- \text{Sampson}(\text{p}, q \mid \text{T}_1) / {\tau}\right) $로 영향을 미침
- biased 샘플링을 사용하여 가설의 절반을 샘플링하고 나머지 절반은 uniform sampling 사용
    - correspondnece가 많은 경우 샘플 다양성을 개선. 이 경우 비편향 샘플링이 효과적
