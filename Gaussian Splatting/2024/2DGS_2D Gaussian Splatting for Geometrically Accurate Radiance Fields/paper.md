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

2D 가우시안을 