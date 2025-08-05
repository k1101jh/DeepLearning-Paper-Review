# Quantifying Epistemic Uncertainty in Absolute Pose Regression

---

- Camera Relocalization
- Uncertainty Estimation
- VAE

---

url
- [paper](https://link.springer.com/chapter/10.1007/978-3-031-95918-9_13) (SCIA 2025)
- [paper](https://arxiv.org/pdf/2504.07260?) (arXiv 2025)

---

주요 참고논문
- [43]: [Conditional Variational Autoencoders for Probabilistic Pose Regression](https://ieeexplore.ieee.org/abstract/document/10802091) (IEEE/RSJ 2024)


---

요약

- VAE를 사용하여 입력 이미지 $x_i$를 조건으로 넣어 카메라 포즈 $y_i$를 추정
    - VAE가 해당 공간에 대해 훈련되어 있어야 함. 훈련되지 않은 이미지에 대해서는 잘못된 카메라 포즈를 추정
- likelihood 계산
    - epistemic(인식적) 불확실성(모델이 샘플을 얼마나 잘 알 수 있는지에 대한 불확실성)을 정량화
    - 중요도 샘플링 기법을 통해 근사
    - 테스트 시점에서는 이미지 $x$만 사용 가능하고 실제 자세 $y$는 알 수 없음  
    -> Monte Carlo 생성 $\hat{y}_i = f_\theta (z_i, x), z_i \sim \mathcal{N} (0, I) $을 사용하여 예상 $\log p(\hat{y}_i | x)$ 추정을 제안.
    - 이 likelihood 측정은 $x$로 조건화된 인코더와 디코더 간의 일치를 정량화. VAE가 어떻게 암묵적으로 epistemic 불확실성을 포착하는지를 보임
    - likelihood $\log p (\hat{y} | x)$를 관측 x에 대한 네트워크의 예측 자신감으로 해석

---

**문제점 & 한계점**

- VAE로 공간을 학습시켜서 이미지 한 장으로 절대 포즈를 추정하기 때문에, 학습되지 않은 이미지에 대해서는 잘못된 자세 추정

---


목차

0. [Abstract](#abstract)


---

## Abstract

**Visual ReLocalization**
- 카메라가 보는 이미지를 기준으로 카메라 자세를 추정
- absolute pose regression은 신경망을 훈련시켜 이미지 특성에서 직접 카메라 자세를 회귀
    - 메모리와 계산 효율성이 좋지만, 훈련 도메인 바깥에서는 부정확하고 신뢰할 수 없음
- 새로운 방법 제안
    - variational framework 내에서 관찰의 가능성을 추정하여 절대 자세 회귀 모델의 인식적 불확실성을 정량화
    - 예측에 대한 신뢰도 측정 제공 & 반복 구조가 존재할 때 카메라를 확률적으로 localizing하는 관찰 모호성도 처리하는 통합 모델 제공
    - 불확실성과 예측 오류 간의 관계 포착에서 기존 방식을 능가

## 1. Introduction

**Visual Localization**
- 환경에서 캡처한 이미지에서 카메라의 자세를 추정
- full visual localization pipeline
    1. 카메라 프레임 간 추적(상대 자세 추정)
    2. 글로벌 relocalization(절대 자세 추정)
        이전에 매핑된 환경에서 단일 이미지로부터 카메라 자세 추정
- 전통적인 방법에는 이미지 DB부터 sparse 3D point map까지 다양함.
    - 새로운 query 이미지의 자세를 추정하기 위해 이미지 검색 또는 키포인트 매칭에 사용
    - 효율성과 정확성 간의 trade-off 존재
- 최근의 visual relocalization은 신경망을 사용하여 이미지 기반으로 카메라 자세를 직접 회귀하는 end-to-end 학습 파이프라인을 사용
    - 신경망의 가중치에 맵 표현을 저장
    - 전통적인 방법에 비해 메모리 및 계산 효율성이 높고 강건

![alt text](./images/Fig%201.png)
> **Figure 1.**  
> 훈련 데이터와 다른 경로(또는 다른 도메인)에서 절대 자세 회귀 네트워크를 query하면 높은 예측 오류 발생  
> (회색 선은 예측과 실제 값 사이의 오류)  
> 인식 불확실성 측정 방식은 테스트 샘플이 훈련 분포에 속할 가능성을 추정 (color map으로 시각화)  
> 예측 값과 실제 값의 색상 차이는 예측 오류와 높은 상관관계를 보임

**end-to-end 절대 자세 회귀 방법**
- 효율성이 좋음
- visual relocalization 벤치마크에서 전통적인 기하학적 접근법보다 정확도가 떨어짐
    - 신경망을 사용하면 장면에 대한 공간적 이해를 배울 수 있는 잠재성이 존재
    - 종종 이미지 검색 기반 자세 추정과 유사한 성능을 보임
    - 훈련 데이터가 아닐 때 일반화가 제한됨
- 예측 과정에서 기하학적 제약이나 검증 단계의 부재  
    - 모든 입력은 고도로 부정확할 수 있는 예측 생성
    - 신뢰할 수 있는 예측을 하기 위한 증거가 부족할 때 이를 식별할 수 있는 전통적인 방법과 대조적
- 자세 회귀 네트워크 일반화[25] 및 정확도[6, 35] 개선 연구 노력이 있음
- 네트워크 예측에 신뢰도 측정을 동반해야 함
- Fig 1의 경우, 훈련과 테스트 간의 강한 조명 변화로 인해 시각적 relocalization 시스템을 악화시킴
    - 네트워크는 "자신이 모르는 경우"를 알아야 함

절대 자세 회귀에서 불확실성 추정에 관한 기존 연구
- 두 가지 방법
    1. aleatoric 불확실성을 모델링하여 모호한 관측을 처리[9, 24, 42]
    2. 인지 불확실성을 정량화하여 예측에 대한 신뢰성을 제공[12, 14]. (이 논문의 방법)

**논문의 방법**
- Variational Autoencoder(VAE) 기반 솔루션 구축
    - 인지 불확실성을 정량화하는데 사용할 수 있음
    - 두 가지 유형의 불확실성을 모두 처리할 수 있는 통합 프레임워크 제공
- 절대 자세 회귀에서 인지 불확실성을 정량화하여 예측의 신뢰성을 추정하는 새로운 방법 제안 (Fig 1 참조)

**요약**
1. 절대 자세 회귀에서 인지적 및 aleatoric 불확실성을 처리하기 위한 통합 프레임워크 제안
2. variational framework 내에서 절대 자세 회귀에서 인지 불확실성의 정량화를 위한 공식 도출
3. 제안한 인지 불확실성 정량화가 네트워크 예측의 신뢰성을 어떻게 추정할 수 있는지를 보임
4. 제안 방법이 기존 인지 불확실성 정량화 방법보다 우수함을 보임


## 2. Related Work

### 2.1 Visual relocalization

## 3. Method

이상적인 visual relocalization solution
- 주어진 이미지 $x \in \mathbb{R}^{H \times W \times 3}$에 대해 진짜 카메라 자세 $y \in \text{SE(3)}$을 예측.
> $(x, y) \sim p_{true}$. $(x, y)$는 $p_{true}$에서 샘플링
- real-world distribution $p_{true}$는 경험적 분포 $p_{train}$을 형성하는 유한한 훈련 샘플 세트로만 모델링 할 수 있음
- $p_{train}$의 성공적인 모델링은 $p_{true}$의 다른 unseen 경험적 분포 $p_{test}$에 대한 일반화를 의미
    - 실제로는 $p_{train}$과 $p_{test}$에서 query 간에 항상 모델 성능 격차가 존재
- 훈련 후 테스트 샘플 $(x, y) \sim p_{true}$이 모델링된 분포 $p_{train}$에 얼마나 잘 부합하는지를 측정할 수 있는 visual relocalization 모델에 관심이 있음
    - visual relocalization 작업을 분포 $p_{train}(y | x)$를 학습하는 것으로 정리
    - 이 조건부 분포에서 샘플링 = camera relocalization
    - 주어진 샘플 $(x, y)$에 대한 가능성 추정 = 모델링된 분포 $p_{train}$에 대한 일치 정도를 정량화

### 3.1 Learning a generative model

주어진 이미지 $x \in \mathbb{R}^{H \times W \times C}$에 조건화된 신경망 $\mathcal{f}_\theta (\cdot)$을 훈련
- 이 신경망은 noise 분포 $p(\mathcal{z}) = \mathcal{N}(0, I)$로부터 샘플 $z \in \mathbb{R}^d$를 카메라 포즈 $y \in \text{SE(3)} ~ p(y|x)$의 posterior distribution으로 변환
- [43]에서 보인 것과 같이, 이러한 생성 네트워크는 장면의 이미지가 주어졌을 때 카메라 포즈를 재구성하는 conditional VAE pipeline의 decoder로 훈련될 수 있다.
- 설계 원칙은 [43]을 참조

**Setup and optimization**

conditional VAE는 주어진 이미지 $x_i$에 대한 카메라 포즈 $y_i$를 재구성하기 위해 최적화된 인코더 $g_{\phi}(\cdot)$와 conditional decoder 네트워크 $f_{\theta}(\cdot)$으로 구성됨
- 훈련 세트 $(x_i, y_i) \in \mathcal{D}_{train}$에서 이뤄짐
- 인코더는 각 훈련 포즈 $y_i$를 Gaussian posterior $q(\mathcal{z} | y_i)$의 평균과 공분산으로 매핑.
    - 실제 잠재 posterior 분포 $p(\mathcal{z} | y_i)$를 닮도록 설계됨
- 디코더는 해당 이미지 $x_i$에 조건화되어 이 추정된 posetrior $\mathcal{z}_j \sim q(\mathcal{z} | y_i)$를 원래 포즈 $y_i$의 재구성 $\hat{y}_{i,j} \in \text{SE(3)}$으로 매핑

파이프라인의 최적화 목표: Evidence lower bound(ELBO)
- training sample의 likelihood로부터 다음을 도출:

$$

$$

> $\phi, \theta$: 분포와 이들이 구현된 네트워크 가중치 간의 관계

- $\phi$와 $\theta$를 사용하여 $\mathcal{D}_{train}$에 대해 ELBO를 최대화 = $p(y|x)를 모델링하는 네트워크 $g_\phi(\cdot)$와 $f_\theta(\cdot)$을 최적화
- 가우시안 모델을 따르며, $q_\phi(z|y)$에서 몬테 카를로 샘플을 사용하여 재구성 가능성의 기댓값을 계산

$$

$$

- 모든 샘플의 훈련을 위해 공유되는 homoscedastic $6 \times 6$ 공분산 행렬 $\sum$는 네트워크 가중치 $\theta$와 $\phi$와 함께 최적화됨
- 재구성 오류를 예측에 대한 지역적 수정으로 간주하여 $y = \hat{y} exp(\xi^∧)$로 정의  
-> 오류 벡터를 $\xi = \log (\hat{y}^{−1} y)^∨ \in \mathbb{R}^6$로 리 대수 $\mathfrak{se}(3)^1$에서 정의
    - $\mathfrak{se}(3) \mapsto SE(3)$ 는 리 대수에서 리 군으로의 exponential map. log는 그 역
    - 연산자 ^는 $\xi$를 리 대수 $\mathfrak{se}(3)$의 구성원으로 변환. ∨는 그 역
- 접선 공간에서 오류를 정의하고 공유된 $\sum$를 학습하면 훈련 동안 카메라 자세의 다양한 자유도에 걸쳐 손실 가중치의 자동 조정이 가능해짐
- 이 자동 조정은 [15]에서 영감을 받았으며 데이터셋마다 수행된 비최적 수동 조정의 필요성을 제거[9, 42, 43].
    - [15]는 변환 및 회전 오류 구성 요소 간의 두 개의 손실 가중치 매개변수만 학습하는 것을 탐구
    - 우리는 전체 공분산 행렬을 모델링하여 모든 여섯 개의 자유도로 이 접근 방식을 확장

**Sample generation**

- 디코더를 이미지 $x$에 대해 조건화하고 사전에서 샘플을 디코더를 통해 전달하여 $p(y | x)$에서 쉽게 샘플링할 수 있음
- 이를 통해 $\mathcal{Y} = \{\hat{y}_i = f_\theta (z_i, x) | z_i \sim \mathcal{N}(0,I)\}$를 얻음(그림 2 왼쪽 점선 상자 참고)
- 디코더는 각 이미지와 관련된 불확실한 포즈 모호성의 공간을 학습하며, 모호한 이미지의 경우 서로 다른 잠재 영역이 이미지의 다양한 카메라 포즈에 매핑되도록 적절하게 잠재 공간을 분할[43]

### 3.2 Likelihood estimation

marginal likelihood 계산
- epistemic(인식적) 불확실성(모델이 샘플을 얼마나 잘 알 수 있는지에 대한 불확실성)을 정량화
- 중요도 샘플링 기법을 통해 근사:

$$
\displaystyle
\begin{aligned}

\log p(y|x) &= \log \int p_\theta (y | z, x) p(z | x) dz  & \\
&= \log \int q_\phi(z | y) \frac{p_\theta(y | z, x)\, p(z)}{q_\phi(z | y)}\, dz & \\
& = \log \mathbb{E}_{q_\phi(z | y)} \frac{p_\theta(y | z, x)\, p(z)}{q_\phi(z | y)} & (3) \\
& \approx \log \frac{1}{M} \sum_{z_j} \frac{p_\theta(y | z_j, x)\, p(z_j)}{q_\phi(z_j | y)}

\end{aligned}
$$

> $p(z | x) = p(z)$: 모든 관측값 $x$가 같은 prior distribution $p(z)$를 공유한다는 가정에서 발생

- 테스트 시점에서는 이미지 $x$만 사용 가능하고 실제 자세 $y$는 알 수 없음  
-> Monte Carlo 생성 $\hat{y}_i = f_\theta (z_i, x), z_i \sim \mathcal{N} (0, I) $을 사용하여 예상 $\log p(\hat{y}_i | x)$ 추정을 제안.
- 이 likelihood 측정은 $x$로 조건화된 인코더와 디코더 간의 일치를 정량화. VAE가 어떻게 암묵적으로 epistemic 불확실성을 포착하는지를 보임
- 두 네트워크는 훈련 분포를 기반으로 잠재 공간이 구조화되어 있음  
-> 오직 분포 내 샘플에 대해서만 일치할 것(= 훈련한 샘플에 대해서만 예측 가능할 것)
- likelihood $\log p (\hat{y} | x)$를 관측 x에 대한 네트워크의 예측 자신감으로 해석

![alt text](./images/Fig%202.png)

> **Figure 2. 포즈 추정과 인식적 불확실성 정량화를 위한 파이프라인.**  
> conditional VAE로 장면을 모델링  
> test-time 이미지 관측 $x$가 주어지면, decoder은 카메라 자세의 posterior distribution $p(y | x)$에서 포즈 $\hat{y}$를 샘플링하는 데 사용됨  
> VAE 파이프라인을 통해 $\hat{y}$를 재구성하면 훈련 분포에 대한 테스트 샘플의 likelihood를 나타내는 $\log p(\hat{y} | x)$에 대한 추정을 얻음  
> 이는 모델이 관찰 $x$에 대해 갖는 인식적 불확실성을 반영
