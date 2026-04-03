# ElasticFusion: Dense SLAM Without A Pose Graph

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- SLAM

---

url:
- [paper](https://roboticsproceedings.org/rss11/p01.pdf) (Robotics: science and systems 2015)

---
요약

- 문제점
    - 

---

## Abstract

Real-time dense visual SLAM 제안
- RGB-D 카메라를 사용하여 room-scale 환경을 점진적, 온라인 방식으로 탐사하면서, 전역적으로 일관된 고밀도 지도 생성 가능
- pose graph optimisation 또는 어떤 post-processing step도 필요로 하지 않음

다음 방법 사용
- dense frame-to-model 카메라 tracking
- windowed surfel-based fusion
- non-rigid 표면 변형을 통한 frequent model refinement

- local model-to-model 표면 loop closure 최적화를 가능한 한 자주 적용
- map distribution의 mode 근처를 유지
- global loop closure을 활용하여 임의의 drift를 보정하고, 전역 일관성 유지

## 1. Introduction

## 2. Approach Overview

## 3. Fused Predicted Tracking

scene 표현 방식
- sufel의 unordered list $M$
- 각 sufel의 속성
    - 위치
    - 법선
    - 색상
    - 가중치
    - 반경
        - 해당 점 주변의 국소 표면 면적을 나타냄. visible hole을 최소화하도록 계산
    - 초기화 시각
    - 마지막 업데이트 시각

- Sufel 초기화 및 융합 규칙

- 이미지 공간 정의
    - 이미지 공간 $\Omega$
    - RGB-D 프레임 구성
        - 깊이 맵 $D$
        - 컬러 이미지 $C$
    - 필요 시 central difference로 법선 맵 계산

- 투영 및 역투영 정의
    - 3D 역투영
    $$
    p(u, D) = K^{-1} u \, d(u)
    $$
    - 원근 투영
    $$
    u = \pi(Kp), \quad \pi(p) = \left( \frac{x}{z}, \frac{y}{z} \right)^\top
    $$
    - 픽셀 강도값
    $$
    I(u, C) = \frac{c_1 + c_2 + c_3}{3}
    $$

- 카메라 포즈 추정
    - 각 입력 프레임 시각 $t$에서, 카메라의 전역 포즈 $P_t$를 추정
    - 방법:
        - 현재 RGB-D 프레임과 이전 포즈 추정치로부터 예측된 활성 sufel 모델의 깊이·컬러 맵을 정합
    - 포즈 표현:
    $$
    P_t =
    \begin{bmatrix}
    R_t & t_t \\
    0 & 1
    \end{bmatrix}
    \in SE(3)
    $$

A. Geometric Pose Estimation
