# Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization

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
- [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.pdf) (CVPR 2025)
---
- **Authors**: Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, Yanchao Yang
- **Venue**: CVPR 2025

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Method](#3-method)
  - [3.1 Relative Camera Pose Regression](#31-relative-camera-pose-regression)
  - [3.2 Motion Averaging](#32-motion-averaging)
- [4. Experiments](#4-experiments)
  - [4.1 Relative Camera Pose Estimation](#41-relative-camera-pose-estimation)
  - [4.2 Visual Localization](#42-visual-localization)
  - [4.3 Analyses](#43-analyses)

---

## ⚡ 요약 (Summary)
- ViT 기반 완전 대칭 구조 아키텍처
- Translation을 크기 말고 방향만 예측
    - 크기는 motion averaging을 통해 계산
    - translation을 rotation처럼 각도로 표현하여 둘 사이의 가중치를 결정할 필요를 없앰
- Motion Averaging
    - Rotation averaging
        - 상대 회전으로 절대 회전 계산
        - 중앙값 회전을 사용
    - Camera center triangulation
        - 모든 유효 쌍을 사용하여 평균 교차점 계산
        - 일반적으로 반복 최적화가 필요하지만, 상대 자세 추정치에서 유도된 각 translation 방향까지의 카메라 중심에서의 제곱 거리 합을 최소화하는 최소 자승법 사용
        - 특이값 분해(SVD)를 사용하여 해를 얻음

### 문제점 & 한계점
- 쿼리 이미지와 검색된 데이터베이스 이미지가 일직선상에 있을 경우, motion averaging을 사용하여 metric scale을 해결할 수 없음
- Visual Localization 과정에서 데이터베이스에서 유사 이미지를 찾아야 함. NetVLAD를 사용해서 진행. 상위 10개의 이미지를 검색. 개선의 여지 있음
- 동적 장면에 대한 대응 부족

---

## 📖 Paper Review

## Abstract

**Visual Localization**
- query 이미지의 카메라 포즈를 데이터베이스를 기준으로 결정
- 기존의 신경망 기반 방법
    - 빠른 추론
    - 새로운 장면에 잘 일반화 되지 않음
    - 정확하지 않을 수 있음

**Reloc3r**
- 상대 포즈 회귀 네트워크 + 절대 포즈 추정을 위한 minimalist motion 평균 모듈
- 800만 개 포즈가 지정된 이미지 쌍으로 훈련
- 좋은 성능 & 일반화 능력
- 6개 공개 데이터셋으로 실험

## 1. Introduction

![alt text](./images/Fig%201.png)

> **Figure 1. 자세 정확도와 런타임 효율성 비교**  
> ScanNet1500 데이터셋에서 AUC@5와 image pairs per second(FPS)를 비교  
> 두 가지 버전의 Reloc3r
> - 이미지 너비 512로 훈련
>   - 다른 모든 방법 능가
>   - 24FPS 효율성 유지
>   - 가장 좋은 AUC@5 달성
> - 이미지 너비 224로 훈련
>   - ROMA[34]와 정확도가 일치하면서 20배 빠름

**전통적인 Visual Localization**
- struction-from-motion(SfM) 기술에 의존하여 3D 모델 재구성
    1. 쿼리 이미지의 픽셀을 3D 장면의 포인트와 일치시킴
    2. 기하학적 최적화를 통해 카메라 자세 계산
- 높은 정확도
- 테스트 시간에 비효율성으로 어려움을 겪는 경우가 많음  
-> 확장성에 한계를 가짐

**Scene coordinate regression 방식**
- pixel-point 대응에 대한 대안적인 관점
- 신경망을 사용하여 암묵적인 장면 포현을 학습. 이를 통해 dense correspondence 유추
- 일반화에 한계를 가짐
- 종종 GT keypoint 매치 또는 3D point map과 같은 집중적인 감독이 필요하여 학습 데이터를 확장하기 어렵게 함

**Absolute Pose Regression(APR)**
- 이미지에서 카메라 포즈를 직접 회귀
- 빠른 추론 & 높은 정확성
- 장면 특정적, training 중 밀집한 view point coverage를 요구  
-> 실제 적용 가능성 제한됨
- 합성 데이터셋 생성을 통해 정확성을 향상시키려는 방법이 있음
    - 상당한 계산 오버헤드 발생
    - 광범위한 배치를 저해함

**Relative Pose Regression(RPR)**
- 데이터베이스-쿼리 이미지 쌍 간의 상대 포즈를 추정
- 각각의 장면에 대한 훈련의 필요성 완화
- 빠른 추론(테스트 시 효율성)을 유지
- APR 방식의 위치 확인 정확성을 따라잡지 못함
    - 몇몇 RPR 방법은 서로 다른 데이터셋에서 일반화할 수 있지만, 정확도 감소

각 방법들은 새로운 장면 일반화, 테스트 시간 효율성, 카메라 포즈 정확성 세 가지 중 하나의 문제를 가짐

**Reloc3r**
- Transformer과 대규모 훈련을 활용
- DUSt3R의 아키텍처를 베이스로 사용하여 수정
- 훈련 중 상대 포즈의 metric 척도를 무시
- minimalist 모션 평균화 모듈과 통합하여 절대 포즈를 추정
- 객체 중심, 실내외 장면에 걸쳐 다양한 소스에서 약 800만 개의 이미지 쌍을 처리
- 6개의 포즈 추정 데이터셋에서 우수한 성능을 보임

**논문의 기여**
- Reloc3r 제안
    - 새로운 장면에 대해 탁월한 생성
    - 빠른 테스트시간 효율성
    - 높은 카메라 포즈 정확도
- 제안된 완전 대칭 relative pose 회귀 네트워크와 모션 평균화 모듈은 간소화의 원리를 따름
    - 효율적인 대규모 학습이 가능
- 6개의 데이터셋에 대한 결과는 제안 방식의 효과를 일관되게 보임

## 2. Related Work

**Structure-based visual localization**
- multi-veiw 기하학을 통해 카메라 자세 문제를 푸는 방법
- 최신 방법은 2가지 단계를 따름
    1. 이미지 간 or 픽셀과 사전 구축된 3D 모델간의 대응관계를 설정
    2. 노이즈가 있는 대응 관계를 설정하고 카메라 자세를 견고하게 해결
    - 이러한 correspondences는 keypoint 매칭 또는 장면 좌표 회귀를 통해 얻어짐
    - 이후 견고한 추정기를 적용하여 카메라 자세를 추정
    - 이런 방법은 종종 느린 추론 시간을 가짐
- 최근 효율성 지향적인 변형이 매칭을 가속화하기 위해 제안됨[60, 112[]]
    - 복잡한 시스템 설계 & 높은 계산 비용은 해결되지 않음
- 이 방법은 일반적으로 supervision을 위해 실제 대응 관계나 3D point map을 필요로 하여 대규모 학습의 확장성을 제한함

**Camera pose regression**
- 실시간 추론
- 두 가지 방법
    - APR(Absolute Pose Regression)
        - 이미지를 통해 월드 좌표계에서 카메라의 위치와 방향을 밀리초 이내에 직접 회귀
        - 구조 기반 접근 방식의 정확도에 미치지 못함
        - 종종 이미지 검색을 통한 자세 근사와 유사함
        - 일부 방법은 훈련을 위한 dense viewpoint를 만들기 위해 새로운 시점 합성을 사용
            - 상당한 계산 비용 초래. 각 특정 장면에 대해 훈련이 몇 시간 또는 며칠 걸릴 수 있음
        - 포즈 추정기와 scene-specific map 간의 연결을 구축하여 교육 시간을 줄이려는 시도가 있음
            - 장면별 훈련 및 평가에 한정됨
    - RPR(Relative Pose Regression)
        - 쿼리 이미지와 가장 유사한 데이터베이스 간의 상대 자세를 회귀하여 localization
        - 상대적 변환의 metric scale은 단일 데이터베이스 쿼리 쌍에서 대략적으로 추정 가능
        - 정밀한 절대 위치 측정은 multi-view 삼각측량을 통해 가능함
        - 가장 좋은 RPR 방법은 APR 방법들에 비해 상대적으로 뒤쳐짐
        - 기존 RPR 모델의 일반화 능력은 여전히 제한적
            - Relative PN[49] 및 Relpose-GNN[101]과 같은 방법은 새로운 데이터셋 및 장면에 적응 가능.  
                - 오류가 거의 두 배로 증가
                - map 없이 대규모 데이터셋(약 523K)로 훈련되었음에도 여전히 실내외 설정에 대해 별도의 모델에 의존
            - 다른 방법은 multi-view 자세 추정 뿐만 아니라 넓은 베이스라인 및 파노라마 방법을 탐색
                - 데이터셋 특정 훈련에 그치므로 확장성이 제한적
- 기존 방식은 주로 기술 설계에 집중되어 있고, 더 큰 데이터셋에서의 훈련에는 덜 집중함
- 본 논문에서는 객체 중심, 실내외 데이터셋의 다양한 혼합으로 훈련된 첫 번쨰 포즈 회귀 접근법을 제시
- 이전 연구[129] 에서는 포즈 휘귀의 부정확성이 조잡한 특징 위치 지정에서 기인한다고 제안
    - 본 논문에서는 잘 훈련된 패치 수준 회귀 네트워크가 픽셀 수준 특징 매칭의 성과를 달성할 수 있고, 때로 이를 초과할 수 있음을 발견

**Foundtaion Models**
- 다양한 작업에서 강력한 일반화 성능을 보임
- DUSt3R
    - two-view 기하학과 관련된 거의 모든 작업을 처리할 수 있는 3D 기반 모델
    - 다양한 후속 작업에서 성능을 향상시키기 위해 여러번 fine-tuning됨
- 본 논문에서는 DUSt3R의 Transformer 백본을 채택하여 대칭적인 디자인의 Reloc3r을 개발

## 3. Method

![alt text](./images/Fig%202.png)

> **Figure 2.**  
> Reloc3r은 두 가지 모듈로 구성됨
> - 상대 카메라 자세 회귀 네트워크
>   - 입력 이미지 쌍이 주어지면, 네트워크 모듈은 두 이미지 간의 상대 카메라 자세(정확한 스케일은 알 수 없음)을 추론  
>   - 이 모듈은 공유 가중치를 가진 two-branch Vision Transformer(ViT)로 구성됨
>   - 이미지는 패치로 나눠지고, 토큰이로 변환되어 별도의 인코더를 통해 잠재적 feature로 임베딩됨
>   - 두 개의 잠재적 feature 집합 간에 정보를 교환
>   - 각 헤드는 자신의 잠재적 feature을 집계하여 상대 카메라 자세를 추정
>   - 데이터베이스에 상대적인 쿼리 이미지의 절대 카메라 자세를 추정하기 휘애, 최소 두 개의 데이터베이스-쿼리 쌍을 검색
>   - 이러한 쌍은 상대 자세 추정을 위해 네트워크에 의해 처리됨
> - 모션 평균 모듈
>   - 상대 추정을 집계하여 절대 metric 자세를 계산

**Proble statement**

Visual Localization
- 입력:
    - scene에서 posed images 데이터베이스 $\text{D} = \{ I_{d_n} \in \mathbb{R}^{H \times W \times 3} | n = 1, ..., N \}$
    - 동일한 scene에서 query image $I_q$
- 출력:
    - $I_q$를 데이터베이스 이미지로 정의된 월드 좌표계에 등록할 수 있는 6-DoF 카메라 포즈 $P \in \mathbb{R}^{3 \times 4}$를 추정하는 것
    - $P$는 카메라 회전 $R \in \mathbb{R}^{3 \times 3}$ 과 translation $t \in \mathbb{R}^{3}$으로 표현됨

**Method overview**

제안된 Visual Localization 방법은 두 개의 메인 요소로 구성(Fig 2 참조)
- relative pose regression network
    - query 이미지 $I_q$는 off-the-shelf 이미지 검색 방법[3]을 사용하여 데이터베이스에서 top-K개의 이미지와 쌍을 이루어 이미지 쌍 $\text{Q} = \{ (I_{d_k}, I_q) | k = 1,...,K \}$를 생성
    - 네트워크는 Q에서의 각 이미지 쌍에 대해 독립적으로 상대 포즈 $P_{d,q}$와 $P_{q,d}$를 결정
    - 이미 알고 있는 데이터베이스 이미지의 포즈 $\{\hat{P}_{d_1}, ..., \hat{P}_{d_K} \}$
- motion averaging module
    - relative pose regression network의 결과는 noisy할 수 있고, translation vector metric scale은 불확실할 수 있음
    - rotation averaging과 camera center 삼각측량을 수행하여 query 이미지의 절대 metric pose를 생성

### 3.1 Relative Camera Pose Regression

relative camera pose regression network
- DUSt3R 기반
- 입력: 이미지 쌍 $I_1, I_2$
    - 해상도가 같다고 가정하지만, 다를 수 있음
1. 이미지를 패치로 나누고 ViT encoder을 통해 토큰으로 처리
2. ViT Decoder가 cross-attention을 사용하여 두 가지 branch에서 토큰 간의 정보를 공유
3. 상대 포즈 $\hat{P}_{I_1, I_2}$와 $\hat{P}_{I_2, I_1}$을 예측하는 회귀 헤드가 이어짐
    - 두 가지 분기는 대칭적이고 가중치를 공유함

**ViT encoder-decoder architecture**
- pre-training의 이점을 얻음
1. 각 입력 이미지 $I_i$를 $T$개의 토큰($d$ 차원) 시퀀스로 나눔
2. 각 토큰에 대해 RoPE positional embedding을 계산하여 이미지 내 상대 공간 위치를 인코딩
3. $m$개의 ViT encoder blocks를 통해 토큰을 처리하며, 각 블록은 self-attention 및 feed-forward 레이어로 구성되어 인코딩된 feature token $F_1$ 및 $F_2$ 생성:

$$
\displaystyle
\begin{aligned}

& F_i(T \times d) = \mathrm{Encoder}(\mathrm{Patchify}(I_i^{H \times W \times 3})), \mathrm{where} \quad i=1, 2

\end{aligned}
$$

디코더는 $n$ 개의 ViT Decoder blocks로 구성되며, 각 블록은 동일한 RoPE 위치 임베딩을 사용
- 인코딩 블럭과 달리 decoder block은 self-attention과 feed-forward 레이어 사이에 추가적인 cross-attention layer을 통합
- 이 구조는 모델이 두 세트의 feature token 간의 공간적 관계를 추론할 수 있도록 함.
- Decoded 된 토큰:

$$
\displaystyle
\begin{aligned}

& G_1^{(T \times d)} = \mathrm{Decoder}\bigl(F_1^{(T \times d)},F_2^{(T \times d)} \bigr) \\
& G_2^{(T \times d)} = \mathrm{Decoder}\bigl(F_2^{(T \times d)},F_1^{(T \times d)} \bigr)

\end{aligned}
$$


**Pose regression head**

- pose regression head는 $h$ 개의 feed-forward layers로 구성
- 이어서 average pooling과 relative rotation 및 translation을 예측하는 추가 레이어를 포함
- rotation은 초기에 9D 표현을 사용하여 표현됨
- SVD orthogonalization을 통해 $3 \times 3$으로 변환됨
- 이 matrix는 3D translation vector와 결합되어 최종 transformation matrix를 형성
- 최종 상대 포즈 출력:

$$
\displaystyle
\begin{aligned}
& \hat P_{I_1,I_2}^{(3 \times 4)} = \mathrm{Head}\bigl(G_1^{(T \times d)}\bigr)
& \hat P_{I_2,I_1}^{(3 \times 4)} = \mathrm{Head}\bigl(G_2^{(T \times d)}\bigr)
\end{aligned}
$$

**Supervision signal**

네트워크에 의해 예측된 자세는 두 가지 정보를 전달.
1. Rotation: 방향의 상대적인 변화를 측정
2. Translation: 카메라 중심 이동의 상대적인 방향

- 두 값 모두 상대 각도로 표현될 수 있음.
- 예측된 상대 값과 실제 값 간의 차이를 최소화하여 네트워크를 학습시킴: 

$$
\displaystyle
\begin{aligned}

& \mathcal{L}=\ell_R + \ell_t \\

& \ell_R = \arccos\!\Bigl(\frac{\mathrm{tr}(\hat R^{-1}R) - 1}{2}\Bigr)

& \ell_t = \arccos\!\Bigl(\frac{\hat t \cdot t}{\|\hat t\|\;\|t\|\!}\Bigr)

\end{aligned}
$$

> $\mathrm{tr}(\cdot)$: 행렬의 자취(trace)를 나타냄  
> $\hat{R}, \hat{t}$: 예측된 회전과 변환  
> $R, t$: 실제 값

**Discussion**
- 좌표 정렬을 위해 사용되는 DUSt3R의 비대칭 branches와 달리, 상대적 포즈 추정에 더 적합한 완전 대칭 아키텍처를 사용
- 이미지 순서로 인한 편향을 제거하여 훈련을 단순화
- branch 간의 가중치 공유를 허용하여 계산 복잡성과 저장 요구 사항을 줄임

최근 연구[4, 116]에서는 relative pose regression에서 metric pose 학습을 선호
- 본 논문에서는 translation의 방향만 학습하기로 선택
    - motion averaging이 metric scale을 효과적으로 해결할 수 있기 때문
    - translation을 rotation과 동일한 차원에서 각도로 표현하여 훈련 중 rotation과 translation의 가중치를 부여할 필요를 없앰
    - 다양한 데이터셋에서 metric scale을 balancing하는 도전 과제를 피할 수 있음

### 3.2 Motion Averaging

- pose regression network와 minimalist motion averaging 모듈 결합
- 네트워크 예측이 매우 정확하여 추가적인 robust 추정은 적용하지 않음
- 각 이미지 쌍에서 두 개의 상대 포즈 $\hat P_{I_1,I_2}$와 $\hat P_{I_2,I_1}$ 생성
    - 서로 역변환 관계. 둘 다 비슷한 정확도를 보임
- 쿼리 -> 데이터베이스 변환을 motion averaging 입력으로 사용

**Rotation averaging**

- 데이터베이스-쿼리 쌍으로부터 얻은 상대 회전 $\hat R_{q,di}$로 절대 회전 $\hat R_q = R_{di}\,\hat R_{q,di}$ 계산
- 사용 가능한 모든 쌍의 모든 절대 회전 추정치를 모아 예측 노이즈 감소
    - 이 집계는 quaternion 표현을 통해 평균 회전을 효율적으로 계산
- 중앙값 회전을 계산하면 노이즈에 대한 강인성을 향상시킬 수 있으며, 최소한의 추가적인 계산 비용으로 가능함  
-> 중앙값 회전을 최종 추정으로 사용

**Camera center triangulation**

- 데이터베이스-쿼리 쌍 두 개로부터 절대 카메라 중심을 삼각측량
- 모든 유효 쌍을 사용하여 평균 교차점을 계산
- 교차점의 기하학적 중앙값은 해석적으로 해결할 수 없고, 일반적으로 반복 최적화가 필요
    - 본 논문에서는 상대 자세 추정치에서 유도된 각 translation 방향까지의 카메라 중심에서의 제곱 거리의 합을 최소화하는 최소 자승법을 사용
    - 특이값 분해(SVD. Singular Value Decomposition)를 사용하여 해를 얻음


## 4. Experiments

**Training Data**

- DUSt3R과 유사하게, 7개 공개 데이터셋에서 약 800만 개의 이미지 쌍을 처리
- GT relative pose를 OpenCV 형식으로 변환
- 각 이미지는 주요 점을 기준으로 중앙에서 잘리고 512 픽셀 너비로 크기가 조정됨

![alt text](./images/Table%201.png)

> **Table 1. Reloc3r의 훈련 데이터**  
> 객체 중심, 실내외 환경 등 다양한 장면을 포함

**Implementation details**

- 24개의 인코더 블록, 12개의 디코더 블록을 사용
- $h = 2$ convolutional layer로 구성된 pose regression head
- self & cross-attention에서 메모리 사용량과 속도 개선
    - memory-efficient attention 사용
    - 훈련 중 GPU 메모리의 25% 절약
    - 약 14%의 속도 향상을 달성
- Reloc3r을 DUSt3R의 사전 훈련된 512-DPT 가중치로 초기화
- Decoder 초기화에 두 번째 DUSt3R 디코더의 가중치를 사용
    - 좌표 변환을 수행하기 위해 사전 훈련됨
- 훈련 시
    - batch size: 8
    - 학습률: 1e-5에서 1e-7로 감소
    - MI250x-40G GPU 8개 사용
- Visual localization task
    - 이미지 검색에 NetVLAD 적용
    - 상위 10개의 유사 이미지 쌍 사용
    - 검색된 이미지 쌍을 distance-based clustering, filtering, 기타 휴리스틱 없이 직접 사용하여 공간 분포를 향상시킴
    - 평가 시: 24GB NVIDIA GeForce RTX 4090 GPU에서 수행
    - 혼합 정밀도 fp16/fp32 사용

### 4.1 Relative Camera Pose Estimation

**pair-wise relative pose**

![alt text](./images/Table%203.png)

> **Table 3. 상대 카메라 포즈 추정**  
> 제안 방법은 모든 pose regression 경쟁자보다 우월함  
> 밑줄: 여러 데이터셋과 metric에서 경쟁자들과 비교해서 최고 성능 달성  
> 제안 방법은 real-time으로 동작. Non-PR 방법 SOTA보다 약 50배 빠름

- ScanNet1500, RealEstate10K, ACID 데이터셋 사용
- 다양한 카메라 경로로 포착된 다양한 실내 및 실외 장면을 보임
- ScanNet1500
    - 실내 장면에 중점
- RealEstate10K
    - 실내외 장면
- ACID
    - 공중 촬영한 실외 데이터
- evaluation test set은 훈련 데이터와 장면이 겹치지 않음
- ACID는 훈련에 사용되지 않음
- 세 가지 metric
    - AUC@ 5/10/20
        - 최소 회전 및 변환 각도 오류에 대한 임계값 $\tau=5/10/20$에서 자세 정확도의 곡선 아래 면적을 계산

기존 Pose regression 방법이 이러한 데이터셋을 위해 설계된 것이 없음
- 비-PR 방법과 본 논문의 접근 방식 비교
- 현재 SOTA relative pose regression 방법(ExReNet[93, 116], Map-free[4])를 평가 (표 3.)
- Reloc3r은 세 가지 데이터셋에서 다른 PR 방법보다 우수한 성능을 보임
- 비-PR 방법과 비교했을 때, DUSt3R 보다 약 13% 더 높은 AUC@20 성능 제공
- Reloc3r은 512 픽셀 해상도에서 42ms 추론 시간으로 실행됨
    - NoPoSplat[122](>2000ms) 및 ROMA[34](300ms) 등 많은 비-PR 방법보다 빠르고 PR 방법과 동등함


**multi-view relative pose**

![alt text](./images/Table%202.png)

> **Table 2. CO3Dv2 데이터셋에서 상대 자세 추정(multi-view)**  
> 밑줄: 제안 방법이 모든 경쟁자 중에서 최고 성능을 달성  
> *: 8프레임을 사용하여 평가

- Co3dv2 데이터셋 사용
    - inward-facing 카메라 경로로 캡처한 object-level scene으로 구성됨
    - 시각적 대칭, 텍스처가 없는 객체, 이미지 간의 넓은 baseline(각도가 큰) 포함
- 41개 범주 테스트셋에서 평가
- 각 시퀀스에서 무작위로 10프레임을 샘플링하고 평가를 위해 45쌍을 형성
- 세 가지 metric
    - 15도 내의 상대 회전 (RRA@15)
    - 15도 내의 상대 translation (RTA@15)
    - mean avearage accuracy(mAA@30 또는 AUC@30)
- 비교 대상
    - PixSfM[59]
    - VGGSfM[107]
    - RelPose[127]
    - RelPose++[57]
    - PoseDiffusion[106]
    - RayDiffusion[128]
    - DUSt3R(PnP 사용) [111]
    - MASt3R[51]
    - PoseReg[106]
    - RayReg[128]
- DUSt3R 및 MASt3R과 유사하게 pair-wise 평가만 사용하여 평가를 위한 더 많은 맥락 정보를 가짐
- Reloc3r은 여러 metric에서 SOTA를 달성
- multi-view 설정 및 wide baselines에서 강인하고 정확하게 localizing


### 4.2 Visual Localization

![alt text](./images/Fig%203.png)

> **Figure 3. 포즈 추정 결과 시각화**  
> 7 Scenes 데이터셋의 Chess와 Cambridge Landmarks의 KingsCollege 시각화  
> ExReNet[116]과 Map-free[4]의 결과와 비교  
> Reloc3r의 포즈 추정 결과가 GT 포즈와 더 유사함

7 Scenes 데이터셋과 cambridge landmarks를 사용하여 실험 수행
- 7 Scenes
    - 서로 다른 이동 궤적에서 캡처된 여러 비디오 시퀀스를 포함하는 7개의 실내 방 장면으로 구성
- Cambridge Landmarks
    - 6개의 장면을 특징으로 하는 교외 규모 야외 데이터셋
    - 이전 접근 방식[15, 54, 101]을 따르며, 평가를 위해 6개 중 5개 장면 사용
- 각 장면에 대한 median translation 및 rotation 오류 측정
- 두 데이터셋 모두 훈련에 사용되지 않음

**Indoor visual localization**

![alt text](./images/Table%204.png)

> **Table 4. 7 Scenes 데이터셋에서 Visual localization 결과**  
> 미터와 각도 단위로 median pose error 보고  
> $\dagger$로 표시된 경우는 추가 geometry solver와 feature matching을 결합한 하이브리드 포즈 추정 방법을 나타냄

- 7 Scenes 데이터셋에서 SOTA Absolute Pose Regression(APR) 및 Relative Pose Regression(RPR) 방법과 Reloc3r 비교
- RPR 방법의 두 가지 분류
    - 동일한 데이터셋으로 모델을 훈련하고 평가한 'seen' 그룹
    - 데이터셋으로 모델을 훈련하지 않은 'unseen' 그룹
- EssNet[129], Relative PN[49], RelocNet[7] 과 같은 방법들이 unseen 데이터셋으로 평가할 때 상당한 성능 저하를 보임  
-> 장면 일반화의 한계
- 제안 방법은 'seen' 그룹 방법보다도 뛰어남. 평균 중간 오차 0.04m/1.02º를 달성
- APR 방법과 비교할 때, Reloc3r은 장면 특정 훈련 없이도 비슷한 성능을 보여 강건성과 다양한 장면에 대한 적응 능력을 보임

**Outdoor visual localization**

![alt text](./images/Table%205.png)

> **Table 5. Cambridge Landmark 데이터셋에서 Visual localization 결과**  
> 미터와 각도 단위로 median pose error 보고  
> $\dagger$로 표시된 경우는 추가 geometry solver와 feature matching을 결합한 하이브리드 포즈 추정 방법을 나타냄

- Cambridge 데이터셋을 사용하여 모든 RPR 방법 평가
- Reloc3r은 특정 장면에 대한 재훈련이나 미세 조정 없이 이전 RPR 방법을 초월. 모든 장면에서 일관된 개선을 달성
    - 'unseen' 조건에서 Reloc3r은 이전 최첨단 RPR 방법인 ImageNet+NCM[129]에 비해 평균 포즈 오류를 절반으로 줄임
    - 마지막 네 장면에서 평균 오류는 0.38m/0.52º
    - 모든 APR 기반 방법보다 더 나은 평균 회전 오류를 보임


### 4.3 Analyses

![alt text](./images/Table%206.png)

> **Table 6. assymetric network 아키텍처 및 metric scale을 통한 포즈 예측을 조사하는 절제 연구**

![alt text](./images/Fig%204.png)

> **Figure 4.**  
> 윗줄: Efficient LoFTR의 매칭 결과  
> 아랫줄: Reloc3r의 decoder에서 상위 3개의 cross attention 응답  
> Reloc3r이 pose supervision으로만 훈련되었음에도 상관된 영역이 Efficient LoFTR의 상관된 영역보다 우수함

**이미지 해상도**
- 두 가지 해상도(224, 512)로 훈련하고 평가
- DUSt3R과 유사하게, 더 높은 해상도는 정확도를 향상시키지만 런타임을 증가시킴

**비대칭 네트워크 아키텍처와 비교 (표 6 참조)**
- rkr branch에 별도의 decoder과 head를 사용하는 비대칭 버전의 Reloc3r-512를 훈련.
- 더 많은 계산 자원을 필요로 하면서 기본 Reloc3r보다 성능이 떨어짐

**Metric scale로 포즈 예측과 비교 (표 6 참조)**
- metric pose를 출력으로 가진 Reloc3r-512를 훈련
- metric scale이 없는 기본 설계가 효과적

**Interesting findings (그림 4 참조)**
- Reloc3r의 디코더의 일부 블록에서 cross-attention 응답을 시각화
- 여러 층이 patch 대응을 맞추는 능력을 개발함

**Limitations**
- 실패 사례는 쿼리 이미지와 모든 검색된 데이터베이스 이미지가 완벽하게 일직선상에 있을 때 발생
- motion averaging 방법을 사용하여 metric scale을 해결할 수 없는 degeneracy 문제를 초래

## 5. 결론

Reloc3r
- 간단하면서도 효과적인 visual localization 프레임워크
- minimalist motion averaging module과 relative pose regression network로 구성
- 약 800만 개 이미지 쌍에 대한 대규모 훈련
- 강력한 일반화 능력, 높은 효율성 및 정확한 포즈 추정