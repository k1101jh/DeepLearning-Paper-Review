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

