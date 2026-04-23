# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

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
- [paper](https://arxiv.org/abs/2312.00109) (CVPR 2024)
- [project](https://city-super.github.io/scaffold-gs/)

---
- **Authors**: Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, Bo Dai
- **Affiliation**: Shanghai Artificial Intelligence Laboratory, The Chinese University of Hong Kong, Nanjing University, Cornell University
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---

## Abstract

![alt text](./images/Fig%201.png)
> **Figure 1.**  
> Scaffold-GS는 장면을 3D Gaussian set을 사용하여 dual-layered 계층 구조로 나타냄  
> 초기 point의 희박한 grid에 기반하여, 각 anchor에서 적은 수의 neural gaussian이 생성되어 다양한 시점과 거리에 동적으로 적응  
> 더 간결한 모델로 3D-GS에 필적하는 렌더링 품질과 속도 달성(마지막 행 척도: PSNR / 저장 용량 / FPS)  
> 투명성, 반사광, 반사, 무텍스처 영역 및 미세한 세부 사항 등 관찰이 어려운 view에서도 대규모 야외 장면과 복잡한 실내 환경에서 높은 견고성을 보임

**3D Gaussian Splatting**
- 종종 모든 학습 view에 맞추려는 과도하게 중복된 가우시안 생성
    - 기본 장면 기하 구조를 무시하는 경우가 많음
- 중요한 view 변화, texture가 없는 영역 및 조명 효과에 덜 견고해짐

**Scaffold-GS**
- anchor points를 사용하여 local 3D Gaussians를 분포시킴
- view frustum 내에서 시점 방향과 거리에 따라 속성을 실시간으로 예측
- neural Gaussian의 중요성이 기반한 anchor growing 및 가지치기 전략이 개발되어 장면 커버리지를 안정적으로 향상시킴
- 고품질 렌더링을 제공하면서 중복 가우시안을 효과적으로 줄임
- 렌더링 속도를 희생하지 않으면서 다양한 수준의 세부 사항과 view-dependent 관측을 갖는 장면을 처리하는 향상된 능력 입증

## 1. Introduction

**3D 장면 렌더링**
- mesh와 point 기반 전통적인 primitive-based 표현
    - 현대 GPU에 맞춘 rasterization 기법을 사용
    - 단점
        - 종종 불연속성과 흐릿한 아티팩트를 나타냄
        - 낮은 품질의 렌더링을 초래
- volumetric 표현과 neural radiance field
    - 학습 기반의 parametric model 활용
    - 많은 세부 정보가 보존된 연속적인 렌더링 결과 생성 가능
    - 단점
        - 확률적 샘플링이 시간 소모적임
            - 성능이 느려지고 잠재적인 노이즈 발생 가능

**3DGS**
- SOTA 렌더링 품질 및 속도 달성
- 일련의 3D 가우시안을 최적화
- 부피 표현에서 발견되는 고유한 연속성을 유지
- 3D 가우시안을 2D 이미지 평면에 splatting하여 빠른 rasterization을 용이하게 함
- 단점
    - 