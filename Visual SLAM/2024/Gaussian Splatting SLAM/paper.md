# Gaussian Splatting SLAM


---

- SLAM
- 3DGS SLAM

---

url:
- [paper](https://arxiv.org/abs/2312.06741) (arXiv)
- [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.pdf)(CVPR 2024)

---
요약

- 문제점
    - 

---

## Abstract

제안 방법
- 3FPS로 실행됨
- 3DGS에 대한 direct 최적화를 사용하여 카메라 추적을 설계
    - 넓은 수렴 영역에서 빠르고 강력한 추적이 가능함을 보임
- 3D dense reconstruction에서 발생하는 모호성을 처리하기 위해 기하학적 검증과 정규화 도입
- 새로운 시점 합성과 궤적 추정에서 SOTA 달성 및 작은 객체나 투명한 객체 재구성

## 1. Introduction

- 3DGS 표현을 기반으로 한 최초의 online visual SLAM 제안
- 세 가지 혁신
    1. 3DGS map에 대한 카메라 위치의 Lie group 상 해석적 야코비안 도출
    이를 기존의 미분 가능 raterisation pipeline에 통합하여 카메라 위치와 장면 기하를 동시에 최적화할 수 있음을 보여줌
    2. 점진적 재구성에서 중요하다고 확인된 기하학적 일관성을 보장하기 위해 새로운 gaussian isotropic shape 정규화 도입
    3. 기하 구조를 깨끗하게 유지하고 정확한 카메라 추적을 가능하게 하는 새로운 gaussian 자원 할당 및 가지치기 방법 제안
- 사전 학습된 depth 예측기나 다른 기존 tracking 모듈을 사용하지 않고, RGB 입력만을 사용

논문의 기여
- 3DGS를 유일한 장면 표현으로 사용하며, 단안 입력만으로도 동작하는 최초의 online SLAM 시스템
- direct camera pose estimation을 위한 Lie group 상의 해석적 Jacobian, 가우시안 형태의 isotropic 정규화, 기하학적 검증 등 SLAM 프레임워크 내의 새로운 기법들을 포함함
- monocular 및 RGB-D 환경 모두에서 다양한 데이터셋에 대한 광범위한 평가를 수행하였으며, 실제 환경 시나리오에서 경쟁력 있는 성능을 입증

## 2. Related Work

## 3. Method

### 3.1 Gaussian Splatting

- 장면을 anisotropic Gaussians $\mathcal{G}$로 매핑
- 각 가우시안 $\mathcal{G}^i$는 색상 $c^i$와 불투명도 $\alpha^i$와 같은 광학적 특성을 포함
- 연속적인 3D 표현을 위해, world 좌표계에서 정의된 평균 $\mu^i_W$와 공분산 $\Sigma^i_W$는 가우시안의 위치와 타원의 형태를 나타냄
- spherical harmonics 표현 생략(단순성, 속도를 위해)
- N개의 가우시안을 splatting하고 blending하여, 픽셀 색상 $\mathcal{C}_p$를 합성

$$
\displaystyle
\begin{aligned}

\mathcal{C}_p = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
\tag{1}

\end{aligned}
$$

~~

### 3.2 Camera Pose Optimisation

- 프레임당 적어도 50 iteration의 gradient descent 요구
- 자동 미분의 오버헤드를 피하기 위해 3DGS는 모든 매개변수에 대한 도함수를 명시적으로 계산하여 CUDA로 rasterisation 구현
- 카메라 야코비안을 명시적으로 도출
    - rasterization이 performance critical하기 때문

- EWA splatting 및 3DGS에서 사용되는 3DGS에 대한 $SE(3)$ 카메라 자세의 분석적 야코비안을 제공
    - 최소 야코비안을 도출하기 위해 Lie 대수를 사용하여 야코비안의 차원이 freedom과 일치하도록 하고 불필요한 계산을 제거함
    - 식 (2)의 항들은 카메라 자세 $T_{CW}$에 대해 미분 가능
    - 연쇄 법칙: 
    $$
    \displaystyle
    $$


### 3.3 SLAM

#### 3.3.1 Tracking


- 다음 식으로 photometric residual 최소화(monocular)
$$
\displaystyle
E_{pho} = || I(\mathcal{G}, T_{CW}) - \bar{I} ||_2
\tag{7}
$$
- 다음 식으로 다른 노출도에 대해 affine brightness 파라미터 최적화(depth가 있는 경우)
$$
\displaystyle
E_{geo} = || D(\mathcal{G}, T_{CW}) - \bar{D} ||_1
\tag{8}
$$
> $D(\mathcal{G}, T_{CW})$: depth rasterisation  
> $\bar{D}$: 관측 깊이

- depth 측정값을 gaussian 초기값으로 사용하는 것보다, photometric과 geometric residual을 최소화
$\lambda_{pho}E_{pho} + (1 - \lambda_{pho})E_{geo}$
- 식 1처럼, 픽셀별 depth는 alpha blending을 통해 rasterised됨
$$
\displaystyle
\mathcal{D}_p = \sum_{i \in \mathcal{N}} z_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
\tag{9}
$$



#### 3.3.2 Keyframing

**Selection and Management**

현재 프레임 $i$와 마지막 키프레임 $j$ 사이에 관찰된 가우시안들의 교집합을 통해 공동 가시성을 측정
- 공동 가시성이 임계값 아래로 떨어지거나, 상대 변위 $t_{ij}$가 중간 깊이에 비해 클 경우 프레임 $i$를 keyframe으로 등록
- DSO[4]의 keyframe 관리 휴리스틱을 따라 현재 윈도우 $W_k$에 소수의 키프레임만 유지
- 최신 키프레임과 겹침 계수가 임계값 아래면 윈도우에서 키프레임 제거

**Gaussian covisibility**

covisibility에 대한 정확한 추정은 keyframe 선택과 관리를 단순화함
- 설계상으로 occlusion을 처리할 수 있음
- 가우시안은 rasterisation에 사용됨
    - 누적 $\alpha$ 값이 0.5에 도달하지 않았다면 가시 가능한 것으로 표시됨

**Gaussian Insertion and Pruning**

- 가우시안 추가
    - depth 사용 시
        - 가우시안의 평균 $\mu_W$는 깊이를 역투영하여 초기화
    - depth 미사용 시
        - 현재 프레임에서 깊이를 렌더링
        - 깊이 추정값이 있는 픽셀의 경우, $\mu_W$는 낮은 분산으로 해당 깊이 주변 값으로 초기화
        - 없는 경우, 렌더링된 이미지의 중간 깊이 주변 값에서 높은 분산으로 초기화

- pruning
    - monocular의 경우, 현재 윈도우 $W_k$ 내에서 가시성을 확인해서 가우시안 제거
        - 마지막 3개의 keyframe 내에 추가된 가우시안이 다른 최소 3개의 프레임에서 관찰되지 않았다면 제거

### 3.3.3 Mapping

- $W_k$의 keyframe을 사용하여 현재 보이는 영역 재구성
- global map을 잊어버리는 것을 방지하기 위해 매 반복마다 두 개의 임의의 과거 keyframe $W_r$ 선택
- 3DGS의 rasterisation은 깊이 관측이 있더라도 시선 방향을 따라 가우시안에 제약을 가하지 않음
    - continuous SLAM에서는 많은 아티팩트를 발생시킴
    -> 등방성 정규화(isotropic regularisation) 도입
    $$
    \displaystyle
    \begin{aligned}
    E_{iso} = \sum_{i=1}^{|\mathcal{G}|} || s_i - \tilde{s}_i \cdot 1 ||_1
    \tag{10}
    \end{aligned}
    $$
- scaling parameter $s_i$(타원의 늘어남 정도)를 평균 $\tilde{s}_i$와의 차이로 페널티를 줌
    - 구형성을 촉진
    - 시선 방향을 따라 크게 늘어난 가우시안이 아티팩트를 생성하는 문제를 피하도록 함
- 매핑:
$$
\displaystyle
\min_{T_{CW}^k \in SE(3), \mathcal{G}, \forall k \in \mathcal{W}} \sum_{\forall k \in \mathcal{W}} E_{pho}^k + \lambda_{iso}E_{iso}
\tag{11}
$$