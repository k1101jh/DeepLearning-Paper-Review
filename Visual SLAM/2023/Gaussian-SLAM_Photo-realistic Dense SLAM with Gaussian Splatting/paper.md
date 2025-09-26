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