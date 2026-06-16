# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Gaussian Splatting

---
url:
- [paper](https://dl.acm.org/doi/abs/10.1145/3641519.3657428)(SIGGRAPH 2024)
- [project](https://surfsplatting.github.io/)

---
- **Authors**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao
- **Venue**: SIGGRAPH 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

3DGS
- 3D 가우시안의 multi-view 불일치 특성으로 표현을 정확하게 표현하지 못함

2DGS
- multi-view 이미지에서 기하학적으로 정확한 radiance field를 모델링하고 복원
- 3D volume을 2D 방향 평면 Gaussian Disk 집합으로 축소
- 2D 가우시안은 시점 일관된 기하학을 제공
- perspective-accurate 2D splatting 과정 도입
    - 얇은 표면을 정확하게 복원
    - 안정적인 최적화 달성
    - ray-splat 교차 및 rasterization을 활용
- 깊이 왜곡과 normal 일관성 항목을 포함하여 복원 품질 향상
- 미분 가능 renderer
    - 경쟁력 있는 외형 품질
    - 빠른 학습 속도
    - 실시간 렌더링을 유지하며 노이즈 없는 세밀한 기하학적 복원


## 1. Introduction

3DGS
- 평평한 표면을 표현하지 못함
    - 복잡한 기하학 캡셔닝에 부족

이전 연구들
- surfel이 complex geometry를 효과적으로 표현
- 도형과 음영 속성을 통해 객체 표면을 국소적으로 근사
- 일반적으로 GT geometry, depth sensor 데이터, known lighting이 제한된 상황에서 작동해야 함

2DGS

![alt text](images/Fig%201.png)

~

![alt text](images/Fig%202.png)

2DGS
- 3D 장면을 각기 방향이 지정된 2D Gaussian primitives로 표현
- 렌더링 시 정확한 기하학적 표현
- surfels 기반 모델과 달리, 기하학이 알려지지 않은 상태에서도 gradient 기반 최적화를 통해 복원 가능
- 기하학적 모델링은 뛰어나지만 photometric loss만으로 최적화하면 noise가 있는 재구성이 발생할 수 있음
    - 두 가지 정규화 항 도입
        - Depth distortion
            - ray를 따라 촘촘히 분포된 2D primitives를 집중시킴
        - normal consistency
            - 렌더링된 법선 맵과 렌더링된 깊이의 기울기 사이를 최소화
            - 깊이와 법선으로 정의된 기하학 정렬 보장


## 3. 3D Gaussian Splatting


## 4. 2D Gaussian Splatting


### 4.1 Modeling

![alt text](./images/Fig%203.png)

- 전체 각도 방사도를 하나의 blob으로 모델링하는 대신, 평평한 2D Gaussian으로 3차원 모델링 단순화
    - primitive는 평면 disk 내에서 밀도를 분포시킴
    - 법선은 밀도가 가장 급격히 변하는 방향으로 정의됨
    - 이 덕에, 얇은 표면과 더 잘 맞출 수 있음
- 이전 방식[Kopanas et al. 2021; Yifan et al. 2019]
    - Geometry 재구성을 위해 2D Gaussian 사용
    - 입력으로 dense point cloud나 실제 법선을 필요로 함
- 제안 방법은 sparse calibration point cloud와 photometric 감독만으로 외형과 geometry를 동시에 재구성 가능
- 2D Splat 구성 요소
    - 중심 점 $p_k$
    - 두 개의 주요 접선 벡터 $t_u, t_v$
    - 분산을 조절하는 scaling vector $S = (s_u, s_v)$
        - primitive normal은 두 직교 법선 벡터 $t_w = t_u \times t_v$로 정의됨
    - 방향을 $3 \times 3$인 $R = [t_u, t_v, t_w]$로 배치 가능
    - scaling factor을 $3 \times 3$인 S로 배치 가능
        - 마지막 항목이 0임
            - 2D이기 때문에?
- 2D 가우시안은 world space 내 local tangent plane에서 정의됨

$$

P(u,v) = p_k + s_u t_u u + s_v t_v v = H(u,v,1,1)^T
\tag{4}
$$

$$
\rm{where} \
H =
\begin{bmatrix}
s_u t_u & s_v t_v & 0 & p_k \\
0 & 0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
R S &
p_k \\
0 &
1
\end{bmatrix}
\tag{5}
$$

- 2D Gaussian의 geometry를 나타내는 $H \in 4 \times 4$는 homogeneous transformation matrix
- uv 공간에 있는 point $\rm{u} = (u, v)$에 대해, 2D gaussian 값은 표준 가우시안으로 계산될 수 있음

$$
G(u) = \exp\left(-\frac{u^2 + v^2}{2}\right)
\tag{6}
$$

- 각 2D Gaussian primitive는 불투명도 $\alpha$와 spherical harmonics로 표현되는 view-dependent 외관 $c$를 갖고 있음

### 4.2 Splatting

2D 가우시안을 렌더링하는 방법
- perspective projection의 affine approximation을 사용해서 이미지 공간에 투사
    - 가우시안의 중심에서만 정확
    - 중심에서 멀어질수록 근사 오차가 커짐
- Zwicker et al은 homogeneous coordinate를 기반으로 한 수식 제안
    - 2D splat을 이미지 평면에 투사하는 작업을 homogeneous coordinate에서 일반적인 2D-to-2D 매핑으로 설명 가능
- $W \in 4 \times 4$가 world space에서 screen space로의 combined transformation matrix라고 할 때 screen space의 점은 다음과 같이 얻어짐
$$
\displaystyle
x = (xz, yz, z, z)^T = W P(u,v) = W H (u,v,1,1)^T
\tag{7}
$$

> x: 카메라에서 발사되어 pixel $(x, y)$를 통과하고 깊이 $z$에서 splat과 교차하는 homogeneous ray
- 2D Gaussian을 rasterize하기 위해 Zwicker et al.은 $M = (WH)^{-1}$을 사용한 implicit 방법으로 원뿔을 화면 공간으로 투영하는 방법 제안
    - inverse transformation은 숫자적 불안정을 초래할 수 있음
    - splat 이 선분으로 퇴화될 때(옆에서 볼 때) 문제가 됨
- 이를 해결하기 위해 surface splatting rendering 방법들은 임계값을 사용해 조건이 안좋은 변화들을 버림
    - 미분 가능 rendering framework에 어려움을 줌
    - 임계값 적용이 불안정한 최적화 초래 가능


**explicit ray-splat intersection**
- 세 개의 평행하지 않은 평면의 교점을 찾아서 ray-splat intersection을 효율적으로 찾음
- 이미지 좌표 $x = (x, y)$가 주어지면, projective space에서 pixel의 ray를 두 개의 orthogonal plane, x-plane과 y-plane의 교점으로 매개변수화
    - x-plane
        - 법선 벡터(-1, 0, 0)과 offset x로 정의됨
        - 4D homogeneous plane $h_x = (-1, 0, 0, x)^T$처럼 나타낼 수 있음
    - y-plane
        - $h_y = (0, -1, 0, y)^T$
    - $x = (x, y)는 이 두 평면의 교차점으로 결정됨
- 두 평면을 2D Gaussian primitive의 local 좌표(uv 좌표계)로 변환
    - 평면상의 점을 변환 행렬 $M$으로 변환하는 것은 homogeneous plane parameters를 inverse transpose $M^{-T}$로 변환하는 것과 같음
    - $M = (WH)^{-1}$ 적용은 $M = (WH)^T$ 적용과 같음
    $$
    \displaystyle
    h_u = (WH)^T h_x \quad h_v = (WH)^T h_y
    \tag{8}
    $$
    - 2D Gaussian plane은 $(u, v, 1, 1)$로 표현됨
    - 동시에, 교차점은 x-plane과 y-plane 안에 있어야 함
    $$
    h_u \cdot (u, v, 1, 1)^T = h_v \cdot (u, v, 1, 1)^T = 0
    \tag{9}
    $$
    - intersection point u(x)에 대한 효율적인 해결책
    $$
    \displaystyle
    u(x) = \frac{h2_u h4_v - h4_u h2_v}{h1_u h2_v - h2_u h1_v}
    \quad
    v(x) = \frac{h4_u h1_v - h1_u h4_v}{h1_u h2_v - h2_u h1_v}
    \tag{10}
    $$
    > $h_u^i, h_v^i$: 4D plane의 i번째 파라미터
    - $h_u^3과 h_v^3$은 항상 0
    - local 좌표 (u, v)를 구하면
        - 식 7을 통해 교차점의 깊이 z를 계산
        - 식 6으로 가우시안 값을 평가

**Degenerate Solutions**
- 2D Gaussian을 기울어진 시점에서 보면 화면 공간에서 선으로 표현됨
    - rasterization중에 놓칠 수 있음
- 이를 처리하고 최적화 안정화를 위해 object-space low-pass filter 사용
$$
\hat{\mathcal{G}(x)} = \max{\{\mathcal{G}(u(x)), \mathcal{G} (\frac{x - c}{\sigma})\}}
\tag{11}
$$
> $u(x)$는 식 10에 의해 주어짐
> $c$: $p_k$ 중심의 투영값
- $\hat{\mathcal{G}}$의 중심이 c이고 반지름이 $\sigma$인 고정된 screen space gaussian low-pass filter로 하한이 정해짐
- 렌더링 중 충분한 픽셀이 사용되도록 $\sigma = \sqrt{2}/2$로 설정

**Rasterization**
- 3DGS와 유사한 rasterization 과정을 따름
    - 각 가우시안 primitive에 대해 screen space bounding box를 계산
    - 2D Gaussian은 중심 깊이에 따라 정렬됨
    - bounding box를 기준으로 tile로 조작됨
    - volumetric alpha blending을 상요해서 앞에서 뒤로 알파 가중치를 적용한 외형 통일
        - 누적된 불푸명도가 포화되면 반복 종료