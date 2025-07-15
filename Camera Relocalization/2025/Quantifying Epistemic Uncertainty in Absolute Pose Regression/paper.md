# Quantifying Epistemic Uncertainty in Absolute Pose Regression

---

- Camera Relocalization
- Uncertainty Estimation
- VAE

---

url
- [paper](https://link.springer.com/chapter/10.1007/978-3-031-95918-9_13) (SCIA 2025)
- [paper](https://arxiv.org/pdf/2504.07260?) (arxiv 2025)

---

주요 참고논문
- [43]: [Conditional Variational Autoencoders for Probabilistic Pose Regression](https://ieeexplore.ieee.org/abstract/document/10802091) (IEEE/RSJ 2024)

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

### 1. Introduction

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
> $(x, y) ~ p_{true}$. $(x, y)$는 $p_{true}$에서 샘플링
- real-world distribution $p_{true}$는 경험적 분포 $p_{train}$을 형성하는 유한한 훈련 샘플 세트로만 모델링 할 수 있음
- $p_{train}$의 성공적인 모델링은 $p_{true}$의 다른 unseen 경험적 분포 $p_{test}$에 대한 일반화를 의미
    - 실제로는 $p_{train}$과 $p_{test}$에서 query 간에 항상 모델 성능 격차가 존재
- 훈련 후 테스트 샘플 $(x, y) ~ p_{true}$이 모델링된 분포 $p_{train}$에 얼마나 잘 부합하는지를 측정할 수 있는 visual relocalization 모델에 관심이 있음
    - visual relocalization 작업을 분포 $p_{train}(y | x)$를 학습하는 것으로 정리
    - 이 조건부 분포에서 샘플링 = camera relocalization
    - 주어진 샘플 $(x, y)$에 대한 가능성 추정 = 모델링된 분포 $p_{train}$에 대한 일치 정도를 정량화

### 3.1 Learning a generative model

주어진 이미지 $x \in \mathbb{R}^{H \times W \times C}$에 조건화된 신경망 $\mathcal{f}_\theta (\cdot)$을 훈련
- 이 신경망은 noise 분포 $p(\mathcal{z}) = \mathcal{N}(0, I)$로부터 샘플 $z \in \mathbb{R}^d를 카메라 포즈 $y \in \text{SE(3)} ~ p(y|x)$의 posterior distribution으로 변환
- [43]에서 보인 것과 같이, 이러한 생성 네트워크는 장면의 이미지가 주어졌을 때 카메라 포즈를 재구성하는 conditional VAE pipeline의 decoder로 훈련될 수 있다.
- 설계 원칙은 [43]을 참조
