# DUSt3R: Geometric 3D Vision Made Easy

---

- 3D Reconstruction

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html) (CVPR 2024)

---
짧은 요약



요약

---

**문제점 & 한계점**


---

목차

0. [Abstract](#abstract)
1. 

---


## Abstract

Multi-view stereo reconstruction(MVS)
- 카메라 내,외부 파라미터를 추정해야 함
    - 3D 삼각측량에 필수적

DUSt3R
- 임의의 이미지 컬렉션에 대한 Dense and Unconstrained Stereo 3D Reconstruction
- 카메라 보정이나 viewpoint pose에 대한 사전 정보가 없어도 됨
- pairwise reconstruction 문제를 pointmap 회귀로 캐스팅
    - 투영 카메라 모델의 강한 제약을 완화함
    - 단안 및 양안 재구성 케이스가 자연스럽게 통합됨
- 두 장보다 많은 이미지가 주어지면 모든 pairwise pointmap을 기준 좌표계로 표현하는 효과적인 global alignment 전략을 제안
- 표준 Transformer 인코더, 디코더 기반으로 설계
- 제안한 formulation은 장면의 3D 모델과 깊이 정보를 제공
    - 픽셀 매칭, 초점거리, 상대/절대 포즈를 복원 가능
- 다양한 3D 비전 작업을 통합
    - 단안 & 다중 view 깊이 추정, 상대 포즈 추정에서 sota 달성


## 1. Introduction

## 2. 

## 3. Method



### 3.1 Overview

목표: 일반화된 streo상황에서 direct regression 방식으로 3D 재구성을 해결하는 신경망 구축

입력
- 두 장의 RGB 이미지 $I^1, I^2 \in \mathbb{R}^{W \times H \times 3}$

출력
- 각 이미지에 대응되는 포인트맵(pointmap): $X^{1,1}, X^{2,1} \in \mathbb{R}^{W \times H \times 3}$
- 각 포인트맵에 대응하는 신뢰도 맵(confidence map): $C^{1,1}, C^{2,1} \in \mathbb{R}^{W \times H}$

- 출력되는 두 pointmap은 모두 $I^1$ 좌표계에서 표현됨
    - 여러 이점을 제공(1장, 2장, 3.3절, 3.4절 참조)

### 3.2 Training Objective


### 3.3 Downstream Applications


### 3.4 Global Alignment