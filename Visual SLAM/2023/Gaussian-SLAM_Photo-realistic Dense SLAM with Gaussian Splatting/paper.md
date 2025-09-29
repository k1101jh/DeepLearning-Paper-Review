# Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting


---

- RGBD SLAM

---

url
- [paper](https://arxiv.org/abs/2312.10070) (arXiv 2023)
- [project](https://vladimiryugay.github.io/gaussian_slam/)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

Gaussian-SLAM
- 단일 카메라 RGBD 영상만으로도 실시간에 가까운 속도로 재구성 및 photo-realistic rendering 달성
- 가우시안 seeding 전략
    - 새로 탐색된 공간에 가우시안을 효율적으로 추가하기 위한 독창적 전략 제안
    - seed된 가우시안은 장면 크기에 독립적으로 효과적으로 online 최적화됨
    - 대규모 장면에도 확장 가능
- sub-map 기반 최적화
    - 전체 장면을 여러 개의 sub-map으로 분할
    - 각 sub-map은 독립적으로 최적화되며 최적화 완료 후 메모리에서 해제 가능
- frame-to-model 카메라 tracking
    - 입력과 렌더링된 프레임 간의 photometric&geometric loss를 최소화
- 가우시한 표현으로 인해 high-quality photo-realistic real-time rendering 가능
- 평가
    - 합성 및 실제 데이터셋에서 mapping, tracking, rendering 분야에서 기존 neural dense SLAM 방법과 경쟁적이거나 우수함


## 1. Introduction

배경 및 동기
SLAM은 지난 20년간 활발히 연구되어 왔으며, 트래킹 성능 향상과 고품질 지도를 위한 다양한 장면 표현이 제안되어 왔습니다. 초기에는 피처 포인트 클라우드[14,26,42], 서펠[56,72], 깊이 맵[45,61], 암시적 표현[13,44,46] 등을 이용해 카메라 트래킹에 집중했습니다. 최근에는 NeRF[40]를 활용한 고충실도 뷰 합성 기술 덕분에 밀집 신경 SLAM[18,34,53,63,65,66,85,89]이 크게 발전했지만, 여전히 소규모 합성 장면에 한정되고 실제 렌더링은 포토리얼리즘에 부족함이 있습니다.

Gaussian splatting 기반 표현
- Gaussian splatting[25]은 NeRF와 동등한 렌더링 품질을 유지하면서 렌더링 및 최적화 속도를 10배 이상 개선합니다.
- 이 표현은 직접 해석 및 조작이 가능해 경로 계획, 시맨틱 이해 같은 다운스트림 작업에도 유리합니다.
- 이러한 장점을 온라인 SLAM 시스템에 적용하면 포토리얼리스틱 밀집 SLAM 구현의 문을 열어 줍니다.

Gaussian-SLAM 시스템 개요
- Gaussian-SLAM은 3D 가우시안을 장면 표현으로 사용해 다음을 동시에 달성합니다.
    - 실시간에 가까운 속도로 장면 재구성
    - 고품질 사진 같은 실시간 렌더링
    - RGBD 입력만으로 매핑 및 카메라 트래킹

- 그림 1에는 Gaussian-SLAM이 재현해 내는 고충실도 렌더링 예시가 제시되어 있습니다.

주요 기여
- 3D 가우시안 기반 장면 표현을 활용해 실제 환경에서 SOTA 렌더링 결과를 얻는 밀집 RGBD SLAM 기법 제안
- 단일 카메라 설정에서도 NeRF를 넘는 지오메트리 표현과 재구성을 가능케 하는 Gaussian splatting 확장
- 맵을 하위 지도(sub-map) 단위로 처리하며 효율적인 가우시안 시딩 및 최적화 전략을 도입한 온라인 최적화 기법 개발
- 광도 및 기하학적 손실을 최소화하는 프레임-투-모델 트래커 구현
- 모든 소스 코드와 데이터 공개 예정

## 2. Related Work


## 3. Method

![alt text](./images/Fig%202.png)

> **Fig 2. Gaussian-SLAM Architecture**  
> 모든 input keyframe에 대해 camera pose는 active sub-map에 대해 depth와 color loss를 추정  
> 추정된 camera pose가 주어졌을 때, RGBD 프레임은 3D로 변환되고 color gradient에 subsample된 

효율적인 맵 구축
- sequential single-camera RGBD 데이터 처리 시 계산량을 상한
- 미분 가능한 depth rendering을 도입해 기하학 정보 손실 없이 정확한 gradient를 계산
- 3D map 표현에 기반한 frame-to-model tracking 방법을 개발

### 3.1 Gaussian Splatting

초기화
- sparse Structure-from-Motion point cloud로부터 3D Gaussian 초기화
- 각 가우시안의 parameter
    - 평균(중심 위치) $\mu \in \mathbb{R}^3$
    - 공분산 $\Sigma \in \mathbb{R}^{3 \times 3}$
    - 불투명도 $o \in \mathbb{R}$
    - RGB 색상 $C \in \mathbb{R}^3$

2D 이미지에 투영할 떄 위치 계산
$$
\displaystyle
\mu^I = \pi(P(T_{wc} \mu_{homogeneous}))  
\tag{1}
$$

> $T_{wc} \in SE(3)$: world-to-camera transformation  
> $P \in \mathbb{R}^{4 \times 4}$: OpenGL-style projection matrix  
> $\pi: \mathbb{R}^4 \rightarrow \mathbb{R}^2$: pixel 좌표계로 projection

splatted Gaussian의 2D 공분산 $\Sigma^I$:
$$
\displaystyle
\Sigma^I = JR_{wc}\Sigma R_{wc}^T J^T
\tag{2}
$$

> $J \in \mathbb{R}^{2 \times 3}$: affine transformation

한 채널 $ch$에서 픽셀 $i$에 영향을 주는 $m$개의 정렬된 가우시안을 이용해 색상 $C$는 다음과 같이 렌더링됨

$$
\displaystyle
C_i^{ch} = \sum_{j \le m} C_j^{ch} \cdot \alpha_j \cdot T_j
, \quad \text{with}\quad 
T_j = \prod_{k < j} (1 - \alpha_k)
\tag{3}
$$

$\alpha_j$:

$$
\displaystyle
\alpha_j = o_j \cdot \exp(-\sigma_j)
\quad \text{and} \quad 
\sigma_j = \frac{1}{2} \Delta_j^{T} \Sigma_{j}^{I-1} \Delta_j
$$

> $\Delta_j \in \mathbb{R}^2$: 픽셀 좌표와 2D로 splatted된 가우시안의 평균(중심) 간의 offset

- 3D Gaussian의 파라미터는 렌더링 이미지와 학습 이미지 간의 photometric loss를 최소화하도록 반복적으로 업데이트됨
- 최적화동안, 색상 $C$는 direction-based color variations를 모델링하기 위해 spherical harmonics $SH \in \mathbb{R}^{15}$로 인코딩됨
- 공분산은 다음과 같이 분해됨. $\Sigma = R S S^T R^T$ ($R \in \mathbb{R}^{3 \times 3}, S = \text{diag}(s) \in \mathbb{R}^{3 \times 3}$)
    - rotation과 scale은 gradient 기반 최적화 중 공분산의 positive semi-definite 성질을 보존하도록 설계됨

### 3.2 3D Gaussian-based Map

- 입력을 chunk(sub-maps) 단위로 처리
    - catastrophic forgetting과 overfitting을 피하기 위함
- 각 sub-map은 이를 관측하는 여러 keyframe을 포함하며 별도의 3D gaussian point cloud로 표현됨
- sub-map의 gaussian point cloud $P^s$는 $N$개의 3D Gaussian 모임으로 정의됨
$$
P^s = \{ G(\mu_i^s, \Sigma_i^s, o_i^s, C_i^s) \mid i = 1, \ldots, N \}
\tag{5}
$$

**Sub-map 초기화**
- sub-map 시작은 첫 프레임에서 시작하며 새로운 keyframe이 들어올 때마다 점진적으로 커짐
- 탐색 영역이 커질수록, unseen regions를 커버하고 GPU 메모리에 모든 가우시안을 저장하는 상황을 피하기 위해 새로운 sub-map 필요
- 새로운 sub-map을 생성할 때 고정된 interval을 사용하는 대신 카메라 움직임에 의존하는 초기화 전략 사용
    - 두 가지 경우
        - 현재 프레임이 활성 submap의 첫 번째 프레임에 대해 추정한 변위가 미리 정의된 임계값 $d_{thre}$를 초과하는 경우
        - 추정된 오일러 각이 $\theta_{thre}$를 초과하는 경우
    - 항상 활성 sub-map만 처리
    - 계산 비용을 제한하고, 더 큰 장면을 탐색하면서 최적화가 빠르게 유지되도록 함

**Sub-map 구축**
- 각 새로운 keyframe은 장면에서 새로 관찰된 부분을 반영하기 위해 활성 sub-map에 3D 가우시안을 추가할 수 있음
- 현재 keyframe의 자세 추정이 완료된 후, keyframe의 RGBD 측정값으로부터 dense point-cloud가 계산됨
- 각 sub-map의 시작에서는 새로운 가우시안을 추가하기 위해 color gradient가 높은 영역에서 keyframe point cloud에서 $M_u$와 $M_c$ point를 균일하게 샘플링
- sub-map의 다음 keyframes에 대해서는 렌더링된 알파 값이 기준 $\alpha_n$보다 낮은 영역에서 $M_k$ point를 균일하게 샘플링
    - 3D 가우시안이 sparese하게 커버된 영역에서도 맵을 확장할 수 있음
- 새로운 gaussian은 현재 sub-map에서 탐색 반경 $\rho$ 내에 이웃이 없는 샘플 포인트를 사용하여 sub-map에 추가됨
- 새로운 가우시안은 anisotropic(비등방성)이며, 활성 sub-map 내의 nearest neighbor distance를 기준으로 규모가 정의됨
- 이러한 densification 전략은 최적화 중 gradient 값을 기반으로 새로운 가우시안을 추가하고 가지치기한 [25]와는 크게 다름
- 가우시안 수에 대한 세밀한 제어를 제공

**Sub-map 최적화**
- 활성 sub-map의 모든 가우시안은 새로운 가우시안이 sub-map에 추가될 때마다 loss(12)를 최소화하도록 고정된 횟수만큼 반복하여 공동 최적화
- [25]에서 최적화동안 가우시안을 복제하거나 가지치기하는 대신, 깊이 센서에서 얻은 기하학적 밀도를 유지하고, 계산 시간을 줄이며 가우시안 수를 더 잘 제어함
- 활성 sub-map을 최적화하여 그 모든 keyframe의 깊이와 색상을 렌더링함
- 최적화를 빠르게 하기 위해 spherical harmonics 함수를 사용하지 않고 RGB 색상을 직접 최적화
- Gaussian splatting[25]에서는 scene 표현이 모든 학습 view에 걸쳐 여러 번 반복 최적화됨
    - 이러한 접근 방식은 속도가 중요한 SLAM 환경에는 적합하지 않음
    - 모든 keyframe에 대해 동일한 횟수로 반복 최적화를 수행하면 과소적합이 발생하거나 최적화에 과도한 시간이 소요됨
    - 활성 sub-map의 keyframe만 최적화하고 새로운 keyframe에는 최소 40%의 반복을 사용

### 3.3 Geometry and Color Encoding

Gaussian Splatting은 이미지 렌더링에는 우수하지만, 직접적인 depth supervision이 없기 때문에 렌더링 된 depth maps는 정확도가 제한적
- 추가적인 depth loss를 통해 문제 해결
- $m$개의 ordered Gaussian에 의해 영향을 받는 $i$번째 픽셀에서 깊이
$$
\displaystyle
D_i = \sum_{j \le m} \mu_{z,j} \cdot \alpha_j \cdot T_j
\tag{6}
$$
> $\mu_j^z$: 3D 가우시안 평균의 z 성분  

- 관측된 depth로 3D gaussian parameter을 업데이트하기 위해 3D 가우시안의 평균, 공분산, opacity에 대한 depth loss의 gradient를 계산
- Gaussian $j$의 평균 업데이트를 위한 gaussian 계산
$$
\displaystyle
\frac{\partial L_\text{depth}}{\partial \mu_j}
= \frac{\partial L_\text{depth}}{\partial D_i}
  \;\frac{\partial D_i}{\partial \alpha_j}
  \;\frac{\partial \alpha_j}{\partial \mu_j}
\tag{7}
$$

- $\displaystyle \frac{\partial L_{\text{depth}}}{\partial D_i}$: 식 (9)를 사용하여 pytorch autograd로 자동 계산
- $\displaystyle \frac{\partial \alpha_j}{\partial \mu_j}$: [25]와 동일하게 유도
- $\displaystyle \frac{\partial D_i}{\partial \alpha_j}$:
$$
\displaystyle
\frac{\partial D_i}{\partial \alpha_j}
= \mu_{z,j} \cdot T_j
  \;-\;\frac{\sum_{u > j} \mu_{z,u}\,\alpha_u\,T_u}{1 - \alpha_j}
\tag{8}
$$
