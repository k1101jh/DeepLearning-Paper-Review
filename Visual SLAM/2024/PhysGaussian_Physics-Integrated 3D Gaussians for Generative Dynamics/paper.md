# PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- SLAM
- Dynamic SLAM
- 3D Gaussian Splatting
- Physics-Integrated

---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.html) (CVPR 2024)
- [project](https://xpandora.github.io/PhysGaussian/)
- [github](https://github.com/XPandora/PhysGaussian)
---
- **Authors**: Tianyi Xie, Zeshun Zong, Yuxin Qiao, Liwen Wan, Xuan Li, Bin Wang, Chenfanfu Jiang
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Method Overview](#3-method-overview)
  - [3.1 3D Gaussian Splatting](#31-3d-gaussian-splatting)
  - [3.2 Continuum Mechanics](#32-continuum-mechanics)
  - [3.3 Material Point Method](#33-material-point-method)
  - [3.4 Physics-Integrated 3D Gaussians](#34-physics-integrated-3d-gaussians)
  - [3.5 Spherical Harmonics Evolution](#35-spherical-harmonics-의-진화하는-방향)
  - [3.6 Incremental Evolution](#36-gaussian-의-점진적-진화)
  - [3.7 Internal Filling](#37-internal-filling)
  - [3.8 Anisotropy Regularizer](#38-anisotropy-regularizer)
- [4. Experiments](#4-experiments)

---

## ⚡ 요약 (Summary)
- **Problem**: 3D 가우스 스플래팅(3DGS)은 정적인 장면 복원에는 탁월하나, 중력이나 충돌 등 물리 법칙에 따른 동적인 시뮬레이션과 렌더링을 통합적으로 처리하기 어려움.
- **Idea**: 3D 가우시안 커널에 속도, 변형기울기, 응력 등 물리적 속성을 직접 부여하고, 연속체 역학(Continuum Mechanics) 기반의 MPM 시뮬레이션 프레임워크를 3DGS에 통합함.
- custom Material Point Method(MPM)를 적용
    - 연속체 역학 원리에 따라 발전하는 물리적으로 의미 있는 운동 변형 및 기계적 응력 속성을 3D 가우시안 커널에 통합


![alt text](images/Fig%201.png)
> **Figure 1.**  
> PhysGaussian은 3D 가우시안과 연속체 역학을 기반으로 하는 통합된 simulation-rendering 파이프라인

**Continuum Mechanics(연속체 역학)**
- 변형되지 않은 물질 공간과 시간 $t$에서의 변형된 world space 사이의 time-dependent 연속 변형 지도에 의해 운동을 설명함
- 상태가 시간에 따라 변할 때, 질량과 운동량 보존 법칙을 따름
- 변형은 탄성과 소성으로 분해될 수 있음

**Material Point Method(MPM)**
- 라그랑주 입자와 오일러 격자의 장점을 결합하여 연속체 방정식을 풀어냄
    - 라그랑주(입자 기반): 
        - 연속체는 작은 물질 영역을 나타내는 입자들의 집합으로 이산화됨
        - 입자들은 위치 $x_p$, 속도 $v_p$, 변형기울기 $F_p$와 같은 여러 시간에 따라 변화하는 라그랑주 양을 추적
        - 라그랑주 입자에서의 질량 보존은 이동 동안 전체 질량의 불변성을 보장
    - 오일러(격자 기반):
        - 운동량 보존은 격자 구성을 오일러 표현에서 더 자연스러움
        - 공간에 고정된 격자를 두고, 그 격자를 통과하는 물질의 변화를 봄
        - 물체 간의 충돌이나 내부에서 발생하는 힘(응력)을 계산하기 쉬움

**Physics-Integrated 3D Gaussians**
- 시뮬레이션된 연속체를 공간적으로 이산화하기 위해 Gaussian kernel을 이산 입자 cloud로 취급
- 연속체가 변형됨에 따라, 가우시안 커널도 변형되게 함
- 변형 map $\phi(X, t)$를 사용해서 커널을 변형하면 $\phi$가 비선형적이어서 가우시안이 아니게 변형됨
    - 이를 해결하기 위해 야코비안 행렬 $F_p$를 적용

정적 장면의 3D GS가 주어졌을 때, 시뮬레이션을 사용하여 장면을 동적으로 만듦

물리적 속성 초기화
- 투명도와 sh 계수가 시간에 따라 불변이라고 가정
- harmonics는 회전될 것
- 각 파티클을 나타내는 volume $V_p^0$: 배경 셀 volume을 포함된 파티클의 수로 나눈 값으로 초기화
    - 왜 나누는가?: 가우시안이 많이 겹친 부분은 밀도가 비정상적으로 커지게 됨
    - MPM의 grid를 활용해서(배경 셀: 격자 한 칸)각 파티클의 부피 초기화
        - 파티클의 부피 = 격자 한 칸의 부피 / 격자 내 파티클 수
- 질량 $m_p$는 user-specified 밀도 $p_p$에 영향을 받음. $m_p = p_p V_p^0$
- 원본 GS 프레임워크에서 splatting을 사용하여 렌더링

**Spherical Harmonics의 변화(진화) 방향**
객체가 회전하게 되면 spherical harmonic으로 인해 시야 방향이 물체에 상대적으로 고정되었음에도 다양한 외관을 보임

해결책:
- 타원체가 시간에 따라 회전할 때, 해당 spherical harmonics 방향도 회전

**Gaussian의 점진적 진화**

기존 방식은 시간 0부터 현재 t까지의 전체 변형 기울기 $F$를 누적해서 계산

대안 방법:
- 직전 step에서 현재 step으로 넘어올 때의 변화량으로 가우시안 업데이트

**Internal Filling**
- 내부 구조는 물체의 표면에 의해 가려짐
- 광선이 불투명도가 낮은 grid $(\alpha_i < \alpha_{th})$에서 불투명도가 높은 곳 $(\alpha_j > \alpha_{th})$을 관통할 때를 고려
    - 물체의 내부 공간 및 바깥 공간은 불투명도가 낮음
- 후보 grid에서 6개의 축을 따라 광선을 투사하고 교차
    - 광선이 모두 표면 grid와 부딪히면 물체 내부로 판단.
- 후보 grid 선택을 정제
    - 교차 횟수(condition 2)를 평가하기 위해 추가 광선을 사용하여 더 높은 정확성을 보장
    - 6축과 겹치지 않는 방향으로 추가 ray를 쏨
    - 홀수 번 관통하면 내부로 판단.
    - 짝수 번 관통하면 외부로 판단
- 채워진 입자들은 가장 가까운 가우시안 커널로부터 $\alpha_p, \mathcal{C}_p$를 상속받음


**가우시안 커널의 이방성**
- 3D 표현의 효율성을 높임
- 시뮬레이션을 할 때, 가늘어진 가우시안에 물리 법칙을 적용하면 렌더링 에러가 발생할 수 있음
- 지나치게 가는 커널은 큰 변형 시 객체 표면 밖으로 향할 수 있음
    - 예상치 못한 plush artifacts를 유발할 수 있음
- major axis 길이와 minor axis 길이의 비율이 $r$을 넘지 않도록 제한하는 학습 loss 추가 제안

---

문제점 및 개선방안
- 물체의 내부를 검사할 때 6축 + 추가 ray를 사용해서 검사
    - 서로 다른 물체로 둘러쌓인 외부 공간을 구분할 수 없음
    - 물체 내부가 아님에도 물체 내부로 판별될 수 있음
    - 개선방안
        - semantic 정보 활용
            - ray와 교차한 grid 들의 semantic feature이 서로 다르면, 물체의 사이 공간으로 인식
        - 뷰 다양성 활용
            - NeRF 처럼 카메라들의 시야 활용하기. 3DGS를 만들고, 시뮬레이션 이전에 한 번만 수행하면 됨
        - depth 정보 활용
            - 모든 view에서 해당 grid의 depth가 관측 depth보다 큰 경우(항상 물체 뒤에 있는 경우) 내부로 판별
        - 사전 정보 활용(semantic 및 객체 정보)
            - 생성 모델 등을 통해 prior을 통해 물체 구분하고 채우기
- 사용자 정의 밀도를 사용
    - semantic 정보를 학습시키고 CLIP 등을 활용해서 해당 물질의 밀도를 자동으로 할당
- 시뮬레이션 시 sh등의 특징이 변할 수도 있을것
- 물체 간 충돌 처리(world model?)

---

## 📖 Paper Review

## Abstract

PhysGaussian
- 뉴턴 역학을 3D 가우시안 내에 원활하게 통합
- custom Material Point Method(MPM)를 적용
    - 연속체 역학 원리에 따라 발전하는 물리적으로 의미 있는 운동 변형 및 기계적 응력 속성을 3D 가우시안 커널에 통합
- 물리 시뮬레이션과 시각적 렌더링 간의 원활한 통합
    - 두 구성 요소 모두 같은 3D Gaussian 커널을 사용
    - 삼각형/사면체 meshing, marching cubes, cage mesh 또는 다른 어떤 기하학적 임베딩의 필요성을 없앰
    - "What you see is what you simuate"($WS^2$) 원칙 강조
    - 탄성체, 플라스틱 금속, 비-뉴턴 유체, 입상 재료를 포함한 다양한 재료에서 탁월한 다재다능성을 보임
    - 새로운 시점과 움직임으로 다양한 시각적 콘텐츠를 생성하는 능력을 보여줌

## 1. Introduction

Neural Radiance Fields(NeRFs)의 최근 발전은 3D 그래픽스와 비전에 상당한 발전을 보임
- 3D Gaussian Splatting(GS) 프레임워크에 의해 더욱 증대됨
- 새로운 역학을 생성하는 적용에는 눈에 띄는 격차가 남아 있음
- NeRFs에 새로운 포즈를 생성하려는 노력들
    - quasi-static 기하학적 형태 편집 작업
    - meshing, tetrahedra와 같은 조잡한 proxy mesh에 시각적 기하학을 포함시킴

전통적인 물리-기반 시각적 콘텐츠 생성 파이프라인은 다단계 과정을 가짐
- 역학 구성, 시뮬레이션 준비 상태로 만들기(종종 tetrahedralization과 같은 기술 사용), 물리 시뮬레이션, 렌더링
- 이 과정은 효과적이지만 시뮬레이션과 최종 시각화 사이에 불일치를 초래할 수 있는 중간 단계를 도입
- NeRF 패러다임에서도 유사한 경향을 보임
    - 렌더링 역학이 시뮬레이션 역학에 내장되기 때문
- 본질적으로 이러한 구분은 물질의 물리적 행동과 시각적 외관이 얽혀 있는 자연 세계와 대조됨  
-> 이 두 측면을 일치시키고자 함
- $WS^2$ 원칙을 지지하며 시뮬레이션, capturihng, rendering의 일관된 통합을 목표로 함

**PhysGaussian**
- 동역학 생성을 위한 physics-integrated 3D Gaussian
- 3D Gaussian이 뉴턴 역학을 포함하여 고체 물질에 내재된 현실적인 행동과 관성 효과를 포착할 수 있게 함
    - 3D Gaussian 커널에 물리학을 부여하여, 속도와 변형과 같은 운동학적 속성과 탄성에너지, 응력, 소성 같은 기계적 특성을 갖게 함
- 연속체 역학 원리와 custom Material Point Method(MPM)을 통해 물리적 시뮬레이션과 시각적 렌더링 모두가 3D 가우시안에 의해 구동되도록 보장
    - 임베딩 메커니즘의 필요성을 없앰
    - 시뮬레이션과 렌더링 사이의 불일치 및 해상도 불일치를 제거

논문의 기여
- **3D Gaussian 운동학을 위한 연속체 역학**
    - 물리적 편미분방정식(PDE) 기반 변위장 내에서 진화하는 3D 가우시안 커널과 관련된 spherical harmonics를 위해 맞춤화된 연속체 역학 기반 전략 소개
- **통합 시뮬레이션-렌더링 파이프라인**
    - 통합 3D 가우시안 표현을 갖춘 효율적인 시뮬레이션 및 렌더링 파이프라인
    - 명시적 객체 meshing에 대한 추가 노력을 제거하여 모션 생성 과정을 크게 단순화
- **다목적 벤치마킹 및 실험**
    - 다양한 재료를 대상으로 포괄적인 벤치마크와 실험 수행
    - 실시간 GS 렌더링과 효율적인 MPM 시뮬레이션으로 강화하여 간단한 역학이 적용되는 장면에서 실시간 성능 달성

## 2. Related Work

**Radiance Fields Rendering for View Synthesis**


**Dynamic Nerual Radiance Field**


이전의 동적 GS 프레임워크가 가우시안 커널의 형태를 유지하거나 이를 수정하도록 학습하는 것과 달리, 변위 맵(변형 기울기)로부터 1차 정보를 고유하게 활용하여 동적 시뮬레이션 보조
- 가우시안 커널을 변형하고 GS 프레임워크 내에서 시뮬레이션을 원활하게 통합 가능

**Material Point Method**

- 다양한 다중 물리 현상을 시뮬레이션하기 위해 널리 사용되는 프레임워크
- topology 변화, 마찰 상호작용을 허용
- 탄성 물체, 유체, 모래, 눈 등을 포함
- 이에 국한되지 않는 다양한 재료의 시뮬레이션에 적합하게 만듦
- MPM은 codimensional(여차원. 주 공간 차원에서 부분 공간의 차원을 뺀 값. 차원이 낮은 구조물이 고차원 공간에서 어떻게 위치하는지 설명) 특성을 가진 물체를 시뮬레이션하도록 확장될 수 있음
- MPM 구현 가속화를 위해 GPU를 활용하는 효능이 입증됨
- Gaussian Splatting 프레임 워크와 함께 공유된 입자 표현을 통해 다양한 시나리오에 역학을 효율적으로 가져올 수 있게 함

## 3. Method Overview

![alt text](images/Fig%202.png)
> **Method Overview**
> PhysGaussian은 3DGS 표현과 연속체 역학을 통합하여 물리 기반 동역학과 photo-realistic 렌더링을 동시에 원활하게 생성하는 통합 시뮬레이션-렌더링 파이프라인

PhysGaussian
- 통합 시뮬레이션-렌더링 프레임워크
- 연속체 역학 및 3D GS를 기반으로 한 생성 역학을 위함
- 정적 장면의 GS 표현을 재구성
- 선택적으로 지나치게 가느다란 커널을 정규화하기 위한 비등방성 손실 항을 사용
    - 이 가우시안들은 시뮬레이션될 장면의 이산화로 간주됨
- 사진처럼 실감나는 렌더링을 위해 변형된 가우시안을 직접 분산시킴
- 물리적 준수를 더 잘하기 위해 객체 내부 영역을 선택적으로 채움

### 3.1. 3D Gaussian Splatting

3D Gaussian Splatting
- NeRF를 3D 가우시안 커널 집합을 사용하여 재매개변수화
- view를 렌더링하기 위해 3D 가우시안을 2D 가우시안으로 투사
- view마다의 최적화는 $L_1$ loss와 SSIM loss를 사용해서 수행됨
- 훈련과 렌더링 속도 가속화
- NeRF 장면의 직접적인 조작도 가능하게 함
- 비디오에 걸쳐 $x_p, A_p$를 시간 결정적으로 생성하고 렌더링 loss를 최소화하여 data-driven 동역학을 지원

### 3.2. Continuum Mechanics

연속체 역학은 변형되지 않은 물질 공간 $\Omega^0$과 시간 $t$에서의 변형된 world space $\Omega^t$ 사이의 time-dependent 연속 변형 지도 $x=\phi(\bm{X}, t)$에 의해 운동을 설명함  
($\bm{X}$ 위치에 있던 입자가 시간 $t$에 $x$ 위치로 이동함을 나타내는 매핑 함수)
- 변형 기울기 $F(X, t) = \triangledown_x \phi(X, t)$
    - 변형 맵 $\phi$를 초기 좌표 $\bm{X}$에 대해 미분한 야코비안 행렬
    - stretch, rotation, shear을 포함한 국소 변환을 encode
- deformation $\phi$의 진화(상태가 시간에 따라 변하는 과정)는 질량 및 운동량 보존에 의해 지배됨
    - 물리 법칙을 따라서 변한다는 뜻
- 질량 보존은 임의의 무한소 영역 $B_\epsilon^0 \in \Omega^0$에 있는 질량이 시간에 따라 일정하게 유지됨을 보장
$$
\displaystyle
\begin{aligned}
\int_{B_\epsilon^t} \rho(x, t) \equiv \int_{B_\epsilon^0} \rho(\phi^{-1} (x, t), 0)
\tag{2}
\end{aligned}
$$
> $B_\epsilon^t = \phi(B_\epsilon^0, t), \rho(x, t)$: 재료 분포를 직징짓는 density field
- 시간 $t$일 때의 총 질량과 시간 0일 때의 총 질량이 동일
- $\rho$: 해당 위치에서의 밀도
- $\phi^{-1}(\bm{x}, t)$: 역함수. 현재 위치 $\bm{x}$에 있는 입자의 원래 초기 위치 $X$를 거슬러 찾아가는 것을 의미

속도장을 $v(x, t)$로 나타낼 때의 운동량 보존
$$
\displaystyle
p(x, t) \dot{v}(x, t) = \triangledown \cdot \sigma(x, t) + f^{ext},
\tag{3}
$$
> $\sigma = \frac{1}{det(F)} \frac{\partial \Psi}{\partial F}(F^E) F^{E^T}$: cauchy stress tensor. $\Phi(F)$: 하이퍼탄성 에너지 밀도. $f^{ext}$: 단위 부피당 외력  
> stress: 응력. 외부 힘을 받을 때 물체 내부에 발생하는 단위 면적당 힘  
> tensor: 어떤 좌표계로 봔환하더라고 변하지 않는 물리량  
> $\rho \dot{v}$: 질량 밀도 $\times$ 가속도. 특정 위치의 물건이 얼마나 가속하며 움직이는지  
> $\triangledown \cdot \sigma$: 코시 응력 텐서의 발산. 물질 내부에서 서로 밀어내거나 당기는 힘  
> $f^{ext}$: 외부에서 가해지는 힘
- 가속도는 내부에서 발생하는 응력과 외부에서 가해진 힘의 합에 의해 결정됨
- 전체 변형 기울기는 탄성 / 소성 부분으로 분해될 수 있음
$F = F^E F^P$
    - 탄성 변형 $F^E$: 힘을 줬다가 빼면 복원되는 변형. 응력의 주 원인
    - 소성 변형 $F^P$: 한계치 이상으로 힘을 받아 영구적으로 형태가 변하는 변형
- 소성으로 인해 발생하는 영구적인 원래의 형상 변화를 지원
- $F^E$의 변화는 특정 소성 흐름에 따라 진행됨
    - 탄성 변형이 무한정 커지지 않도록 제한
- 항상 미리 정의된 탄성 영역 내에 제한됨


### 3.3. Material Point Method

Material Point Method(MPM)
- 라그랑주 입자와 오일러 격자의 장점을 결합하여 위의 방정식을 해결
    - 라그랑주(입자 기반): 
        - 연속체는 작은 물질 영역을 나타내는 입자들의 집합으로 이산화됨
        - 입자들은 위치 $x_p$, 속도 $v_p$, 변형기울기 $F_p$와 같은 여러 시간에 따라 변화하는 라그랑주 양을 추적
        - 라그랑주 입자에서의 질량 보존은 이동 동안 전체 질량의 불변성을 보장
    - 오일러(격자 기반):
        - 운동량 보존은 격자 구성을 오일러 표현에서 더 자연스러움
        - 공간에 고정된 격자를 두고, 그 격자를 통과하는 물질의 변화를 봄
        - 물체 간의 충돌이나 내부에서 발생하는 힘(응력)을 계산하기 쉬움
- Stomakhin et al.[39]를 따라 두-방향 전송을 위해 $C^1$ 연속 B-spline 커널을 사용해 이러한 표현을 통합
- 시간 단계 $t^n$에서 $t^{n + 1}$까지, 전방 오일러 방식으로 이산화된 운동량 보존은 다음과 같이 표현됨

$$
\displaystyle
\frac{m_i}{\Delta t} \left( v_i^{n+1} - v_i^n \right) 
= - \sum_p V_p^0 \frac{\partial \Psi}{\partial F} \left( F_p^{E,n} \right) 
F_p^{{E,n}^T} \nabla w_{ip}^n + f_i^{ext}
\tag{4}
$$
> $i$: 오일러리안 격자 인덱스  
> $p$: 라그랑지안 입자 인덱스  
> $w_{ip}^n$: B-스플라인 커널. $x_p^n$에서 계산된 i-th grid에서 정의된 커널  
> $V_p^0$: 초기 대표 부피  
> $\nabla t$: time step 크기  
> $\Psi(F)$: 탄성 에너지 밀도  
> $f_i^{ext}$: 외력  

- 업데이트된 grid velocity field $v_i^{n+1}$은 입자 $v_p^{n+1}$로 다시 전송되어 입자 위치 업데이트:
$$
\displaystyle
x_p^{n+1} = x_p^n + \Delta t \, v_p^{n+1}
$$
- $F^E$는 다음과 같이 갱신됨:
$$
\displaystyle
F_p^{E,n+1} = (I + \Delta t \nabla v_p) F_p^{E, n} = \left( I + \Delta t \sum_i v_i^{n+1} \nabla w_{ip}^n{}^T \right) F_p^{E,n}
$$
- 소성 변형을 지원하기 위해 추가적인 return mapping으로 규제됨
$$
\displaystyle
F_p^{E,n+1} \leftarrow Z(F_p^{E,n+1})
$$
- 서로 다른 소성 모델은 서로 다른 복귀 매핑을 정의
    - 세부 사항은 supplemental 참조


### 3.4. Physics-Integrated 3D Gaussians

- 시뮬레이션된 연속체를 공간적으로 이산화하기 위해 Gaussian kernel을 이산 입자 cloud로 취급
- 연속체가 변형됨에 따라, 가우시안 커널도 변형되게 함
- 가우시안 커널은 재료 공간에서 $X_p$에서 정의됨
    - $G_p(X) = e^{-\frac{1}{2}(X - X_p)^T A_p^{-1}(X - X_p)}$
    - 변형 map $\phi(X, t)$ 하에 변형된 커널은 world 공간에서의 가우시안에게 필수적이지 않음:
    $$
    \displaystyle
    G_p(x, t) = e^{-\frac{1}{2}(\phi^{-1}(x,t) - X_p)^T A_p^{-1} (\phi^{-1}(x,t) - X_p)}
    \tag{5}
    $$
    - 이는 스플래팅 과정의 요구 사항을 위반
    - 입자를 1차 근사로 특성화되는 local affine 변형을 겪는다고 가정하면(가우시안이 충분히 작다고 가정하고 선형 근사)
    $$
    \displaystyle
    \tilde{\phi}_p(X, t) = x_p + F_p(X - X_p)
    \tag{6}
    $$
    - 다음과 같이 전개 가능
    $$
    \frac{x - x_p}{F_p} = X - X_p
    $$
    - 변형된 커널은 원하는 대로 가우시안이 됨
    - 가우시안 내부의 점 $X$에 대한 수식
    $$
    \displaystyle
    G_p(x, t) = e^{-\frac{1}{2}(x - x_p)^T(F_p A_p F_p^T)^{-1} (x - x_p)}
    \tag{7}
    $$
- 이 변환은 자연스럽게으로 3D GS 프레임워크에 대한 $x_p$와 $A_p$의 time-dependent 버전을 제공
$$
\displaystyle
\begin{aligned}
x_p(t) &= \phi(X_p, t),\\
a_p(t) &= F_p(t)A_p F_p(t)^T
\end{aligned}
\tag{8}
$$

정적 장면의 3D GS $\{X_p, A_p, \alpha_p, C_p\}$가 주어졌을 때, 시뮬레이션을 사용하여 이러한 가우시안을 진화시켜 dynamic Gaussians $\{x_p (t), a_p (t), \alpha_p, C_p\}$을 생성하여 장면을 동적으로 만듦
- 투명도와 sh 계수가 시간에 따라 불변이라고 가정
- harmonics는 회전될 것

식(4)에서 다른 물리적 속성 초기화
- 각 파티클을 나타내는 volume $V_p^0$: 배경 셀 volume을 포함된 파티클의 수로 나눈 값으로 초기화
    - 왜 나누는가?: 가우시안이 많이 겹친 부분은 밀도가 비정상적으로 커지게 됨
    - MPM의 grid를 활용해서(배경 셀: 격자 한 칸)각 파티클의 부피 초기화
        - 파티클의 부피 = 격자 한 칸의 부피 / 격자 내 파티클 수
- 질량 $m_p$는 user-specified 밀도 $p_p$에 영향을 받음. $m_p = p_p V_p^0$
- 이러한 변형된 가우시안 커널들을 렌더링하기 위해, 원본 GS 프레임워크에서 splatting 사용

물리학을 3D 가우시안에 통합하는 것이 매끄러움
- 가우시안 자체는 연속체의 이산화로 간주됨
    - 이는 직접 시뮬레이션 가능
- 변형된 가우시안은 splatting 절차를 통해 직접적으로 렌더링 될 수 있음
    - 전통적인 애니메이션 파이프라인에서 상용 렌더링 소프트웨어가 필요하지 않음
- $WS^2$를 달성하여 현실 데이터에서 장면을 직접적으로 시뮬레이션 가능

### 3.5. Spherical Harmonics의 진화하는 방향

world-space의 3D 가우시안을 렌더링하여 고품질의 결과를 얻을 수 있음
- 하지만 객체가 회전하게 되면 spherical harmonic의 base는 재질 공간에서 표현됨
- 시야 방향이 물체에 상대적으로 고정되었음에도 다양한 외관을 보임

해결책:
- 타원체가 시간에 따라 회전할 때, 해당 spherical harmonics 방향도 회전
- base가 GS 프레임워크 내에 하드코딩되어 있음
    - 시야 방향에 역회전을 적용하여 달성
- Wu et al. [45]에서 시야 방향의 회전은 고려되지 않음
- Chen et al. [6]은 Point-NeRF 프레임워크에서 이 문제를 다루지만 표면 방향 추적이 필요함
- 제안 방법에서, local 회전은 변형 기울기 $F_p$에서 쉽게 얻을 수 있음
- $f^0(d)$: 재질 공간(material space)에서의 spherical harmonic basis. $d$: unit sphere의 점(시야 방향을 표현)
- 극좌표 분해 $F_p = R_p S_p$는 rotated harmonic basis를 유도:
$$
\displaystyle
\begin{aligned}
f^t (d) = f^0 (R^T d)
\tag{9}
\end{aligned}
$$

### 3.6. Gaussian의 점진적 진화

기존 방식(식 8)은 시간 0부터 현재 t까지의 전체 변형 기울기 $F$를 누적해서 계산
- 이는 원래 모양을 기억하는 고체(고무 등)에는 적합
- 진흙, 액체, 모래같은 물질은 초기 상태를 추적하는 의미가 없고 에러를 유발

Gaussian 동역학의 대안 방법 제안
- 직전 step에서 현재 step으로 넘어올 때의 변화량으로 가우시안 업데이트
- Lagrangian 프레임워크 업데이트에 더 적합
- 전체 변형 기울기 $F$에 대한 의존성을 회피
- 변형 측정으로 $F$를 사용하는 것에 의존하지 않는 물리적 재료 모델의 길을 열어줌
- 계산 유체 역학[4, 23]의 관례를 따라, world-space 공분산 행렬 $a$의 업데이트 규칙은 운동학의 rate 형태 $\dot{a} = (\triangledown) a + a (\triangledown v)^T$를 이산화하여 유도할 수 있음
$$
\displaystyle
\begin{aligned}
a_p^{n + 1} = a_i^n + \Delta t (\triangledown v_p a_p^n + a_p^n \triangledown v_p^T)
\tag{10}
\end{aligned}
$$
- $F_p$를 얻을 필요 없이 Gaussian kernel 모양을 time step $t^n$에서 $t^{n+1}$까지 점진적으로 업데이트를 용이하게 함
- 회전 행렬 $R_p$의 각 spherical harmonics basis는 같은 방법으로 점진적으로 업데이트 될 수 있음
- 극좌표 분해를 사용하여 $(I + \Delta t v_p) R_p^n$에서 회전 행렬 $R_p^{n + 1}$를 추출

### 3.7. Internal Filling
내부 구조는 물체의 표면에 의해 가려짐
 - 가우시안은 표면 근처에 분포되려고 함
 - volumetric 객체의 부정확한 동작을 초래
 - 빈 내부 공간을 particles로 채우기 위해 Tang et al[42]에서 영감을 받아 3D Gaussian에서 3D opacity field를 가져옴
$$
\displaystyle
\begin{aligned}
d(x) = \Sigma_p \alpha_p \exp(-\frac{1}{2}(x - x_p)^T A_p^{-1} (x - x_p))
\tag{11}
\end{aligned}
$$

이 연속적인 필드는 3D grid에서 이산화됨
- 강인한 내부 채우기를 달성하기 위해 불투명도 필드에서 user-defined 임계값 $\alpha_{th}$를 사용하여 "intersection"의 개념을 정의함

- 광선이 불투명도가 낮은 grid $(\alpha_i < \alpha_{th})$에서 불투명도가 높은 곳 $(\alpha_j > \alpha_{th})$을 관통할 때를 고려
    - 물체의 내부 공간 및 바깥 공간은 불투명도가 낮음
- 후보 grid에서 6개의 축을 따라 광선을 투사하고 교차
    - 광선이 모두 표면 grid와 부딪히면 물체 내부로 판단.
- 후보 grid 선택을 정제
    - 교차 횟수(condition 2)를 평가하기 위해 추가 광선을 사용하여 더 높은 정확성을 보장
    - 6축과 겹치지 않는 방향으로 추가 ray를 쏨
    - 홀수 번 관통하면 내부로 판단.
    - 짝수 번 관통하면 외부로 판단

큰 변형으로 인해 내부 입자가 노출될 수 있음
- 채워진 입자들은 가장 가까운 가우시안 커널로부터 $\alpha_p, \mathcal{C}_p$를 상속받음
- 각 입자의 공분산 행렬: $diag(r_p^2, r_p^2, r_p^2)$로 초기화됨
    - $r$: 입자의 부피 $r_p = (3V_p^0 / 4 \pi)^{\frac{1}{3}}$으로부터 계산된 반지름
- 혹은 생성 모델 사용을 고려할 수도 있음

### 3.8. Anisotropy Regularizer

가우시안 커널의 이방성
- 3D 표현의 효율성을 높임
- 지나치게 가는 커널은 큰 변형 시 객체 표면 밖으로 향할 수 있음
    - 예상치 못한 plush artifacts를 유발할 수 있음
- 3D Gaussian reconstruction동안 다음 학습 손실 제안
$$
\displaystyle
\mathcal{L}_{aniso} = \frac{1}{|P|} \Sigma_{p \in P} \max \{\max(S_p) / \min(S_p), r\} - r
\tag{12}
$$

> $S_p$: 3D Gaussian의 scalings
- 이 loss는 major axis 길이와 minor axis 길이의 비율이 $r$을 넘지 않도록 제한
- 이 항을 학습 loss에 추가할 수도 있음

## 4. Experiments

### 4.1. Evaluation of Generative Dynamics

**Dataset**
- BlenderNeRF[34]에 의해 생성된 합성 데이터(sofa suite)
- Instant NGP[26] (fox)
- Nerfstudio[41] (plane)
- DroneDeploy NeRF[29] (ruins)
- iPhone으로 수집한 데이터 (toast, jam)
- 각 scene에는 150장의 사진 포함
- 초기 point cloud와 카메라 매개변수는 COLMAP을 사용하여 얻음

**Simulation Setups**
- Zong et al.[53]의 MPM을 기반으로 구축
- 시뮬레이션 영역을 수동으로 선택
    - 변의 길이가 2인 큐브로 정규화
- 내부 입자 채우기는 시뮬레이션 전에 수행 가능
- 직육면체 시뮬레이션 도메인은 3D dense grid로 이산화
- 특정 입자의 속도를 선택적으로 수정하여 제어된 움직임 유도
- 나머지 입자는 자연스러운 운동 패턴을 따름
- 실험 환경
    - RTX3090
    - 24코어 Intel i9-10920X

**Results**
- 탄성
    - 변형 중에도 객체의 원래 형태가 변하지 않는 성질
- 금속
    - 영구적인 형태 변화를 겪을 수 있음
    - von-Mises 소성 모델을 따름
- 파괴
    - MPM 시뮬레이션에서 자연스럽게 지원됨
- Sand
    - 입자 간 세분화된 마찰 효과를 포착할 수 있는 Druker-Prager 소성 모델을 따름
- Paste
    - 비뉴턴 viscoplastic(점탄성) 유체로 모델링
    - Herschel-Bulkley plasticity model을 따름
- 충돌
    - MPM 시뮬레이션을 통해 지원
- 일부 사례가 1/24초 프레임 지속 시간을 기준으로 실시간 달성 가능
    - plane(30FPS)
    - toast(25FPS)
    - jam(36FPS)
- FEM을 활용하여 탄성 시뮬레이션 가속화 가능
    - 추가적인 mesh 추출이 필요
    - 비탄성 시뮬레이션에서 MPM의 일반성을 잃게 됨

### 4.2. Lattice Deformation Benchmarks

- BlenderNeRF[34]를 사용하여 여러 장면 합성
- 격자 변형 도구를 사용하여 굽힘과 비틀림 적용
- 각 장면에 대해 변형되지 않은 multi-view 렌더링 100개를 훈련
- 변형된 상태의 multi-view 렌더링 100개를 변형된 NeRF의 GT로 사용
- 공정한 비교를 위해 격자 변형은 입력값으로 설정됨

**Comparison**

- NeRF-Editing[51]
    - 추출된 표면 mesh를 사용하여 NeRF를 변형
- Deforming-NeRF[47]
    - 변형을 위해 cage mesh 활용
- PAC-NeRF[18]
    - 개별 초기 입자를 조작

![alt text](images/Fig%204.png)

> **Comparisons**  
> 각 벤치마크의 경우, 하나의 test viewpoint를 선택하고 시각화  
> 변형 후에도 고충실도의 렌더링 품질을 유지

![alt text](images/Fig%205.png)

> **Ablation Studies**  
> 확장 불가능한 가우시안은 변형 중에 심각한 시각적 아티팩트를 생성할 수 있음  
> sh에 추가 회전을 적용하면 렌더링 품질 향상 가능

![alt text](images/Table%201.png)

- NeRF-Editing
    - 장면 표현으로 NeuS[43] 사용
        - 표면 재구성에 적합
    - 3DGS보다 렌더링 품질 낮음
    - 변형이 추출된 표면 메쉬와 확장된 cage 메쉬의 정확성에 크게 의존
    - 너무 tight한 mesh는 radiance field를 포함하지 못할 수 있음
    - 지나치게 큰 mesh는 공허한 경계를 초래할 수 있음
- Deforming-NeRF
    - 선명한 렌더링 제공
    - 더 높은 변형 cage를 제공하면 향상된 결과를 도출할 가능성이 있음
    - 모든 cage 정점에서 부드러운 보간을 적용하므로 세밀한 국부 세부 정보를 필터링하고 격자 변형과 일치하지 못함
- PAC-NeRF
    - 시스템 식별 작업에서 더 단순한 객체와 텍스처를 위해 설계됨
    - 입자 표현을 통해 유연성 제공
    - 고품질 렌더링을 달성하지는 못함
- 제안 방법
    - 각 격자 셀에서 0차 정보(변형 맵)과 1차 정보(변형 기울기)를 모두 활용
    - 모든 경우에서 다른 방법 능가
    - 변형 후에도 고품질 렌더링이 잘 유지됨

**Ablation Study**
1. Fixed Covariance
    - 가우시안 커널 이동만
2. Rigid Covariance
    - 가우시안에 강체 변환만 적용
3. Fixed harmonics
    - 구변 조화의 방향을 회전시키지 않음
- Fig 5.에서, 가우시안이 확장 불가능하면 변형 후 표면을 제대로 덮지 못하여 심각한 시각적 이상 현상이 발생
- 

> **Table 1.**  
> ablation study를 위해 격자 변형 벤치마크 데이터셋 합성  
> 제안 방법은 다른 모든 방법보다 우수

### 4.3. Additional Qualitative Studies

![alt text](image.png)

**Internal Filling**
- 3DGS 프레임워크는 객체의 표면 외관에 초점을 맞춰 내부 구조는 포착하지 못하는 경우가 많음
- 그림 6
    - 내부 입자가 없는 객체는 사용된 재료와 상관없이 중력에 의해 붕괴되는 경향이 있음
    - 제안 방법은 객체 동역학을 세밀하게 제어할 수 있게 하며, 다양한 재료 특성에 효과적으로 적응 가능


![alt text](images/Fig%207.png)
> **Figure 7. 부피 보존**  

**Volume Conservation**

- 기존의 NeRF 조작 접근법은 주로 기하학적 조정에 중점을 둠
- 실제 물체는 변형 동안 체적을 보존해야 함
- 제안 방법은 NeRF-Editing[51]과 달리 물체의 체적을 정확하게 포착하고 유지


![alt text](images/Fig%208.png)
> **이방성 정규화**

**Anisotropy Regularizer**
- 지나치게 날씬한 가우시안 커널은 아티팩트를 초래할 수 있음
- 식 12를 도입하여 제약


## 5. Discussion

**결론**
- PhysGaussian
    - 물리 기반 동역학과 photo-realistic renderining을 동시에 원활하게 생성하는 통합 시뮬레이션-렌더링 파이프라인

**한계**
- 그림자의 변화가 고려되지 않음
- 재질 매개변수는 수동으로 설정됨
    - 자동 매개변수 할당은 GS segmentation과 미분 가능한 MPM 시뮬레이터를 결합하여 비디오로부터 파생될 수 있음
- Geometry 인식 3DGS 재구성 방법을 통합하면 생성 동역학을 향상시킬 수 있음