# Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization



---

- Relative Camera Pose Estimation

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.pdf) (CVPR 2025)

---
짧은 요약

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
---

요약



---

목차

0. [Abstract](#abstract)
1. 

---


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
