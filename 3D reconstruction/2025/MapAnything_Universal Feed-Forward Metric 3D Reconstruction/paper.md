# MapAnything: Universal Feed-Forward Metric 3D Reconstruction

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- 3D Reconstruction

---
url:
- [paper](https://arxiv.org/abs/2509.13414)
- [project](https://map-anything.github.io/)
---
- **Authors**: Nikhil Keetha, Norman Müller, Johannes Schönberger, et al.
- **Venue**: 3DV 2026 (arXiv 2025)

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [3. MapAnything Archtiecture](#3-mapanything)

---

## ⚡ 요약 (Summary)

---

## 📖 Paper Review

## Abstract

**MapAnything**
- unified transformer-based feed-forward model
- 입력
    - 하나 이상의 이미지
    - (선택적) 기하학적 정보
        - camera intrinsic
        - pose
        - depth
        - partial reconstructions
- 출력
    - metric 3D scene geometry
    - camera
- multi-view scene geometry를 factored(분해된) 표현으로 모델링
    - 예:
        - depth maps
        - local ray maps
        - camera poses
        - metric scale factor
            - local reconstruction을 전역적으로 일관된 metric frame으로 변환
- single feed-forward pass로 광범위한 3D vision 과제 해결 가능
    - 다양한 데이터셋에 대해 supervision과 training을 표준화
    - 유연한 입력 증강을 적용
    - 가능 작업
        - uncalibrated structure-from-motion
        - calibrated multi-view stereo
        - monocular depth estimation
        - camera localization
        - depth completion 등
- 광범위한 실험 분석 및 모델 절제 연구를 통해
    - MapAnything은 특화된 feedforward 모델들보다 우수하거나 동등한 성능을 보임
    - 효율적인 공동 학습(joint training)
- 범용 3D reconstruction backbone으로 나아가는 길을 제시


## 1. Introduction

기존 image-based 3D reconstruction 접근 방식
- Structure-from-Motion(SfM)
- Photometric stereo
- shape-from-shading

고전적 접근 방식
- Feature Detection 및 매칭
- two-view pose estimation
- camera calibration 및 resectioning
- rotation & translation averaging
- bundle adjustment(BA)
- multi-view stereo(MVS)
- monocular surface estimation

연구 동향
- feed-forward 아키텍처를 활용해 이러한 문제들을 통합적으로 해결하려는 시도가 진행됨
- 기존 feed-forward 연구는 각 과제를 분리(disjoint)하거나 모든 입력 모달리티를 활용하지 못함

MapAnything
- 통합 end-to-end 모델을 제시하여 다양한 3D 재구성 과제를 동시에 해결
- 지원 과제
    - uncalibrated SfM
    - calibrated SfM
    - Multi-view Stereo
    - monocular depth estimation
    - camera localization
    - metric depth completion

unified model 학습 전략
1. 유연한 입력 체계
    - 다양한 gemoetric modalities를 지원
2. 적합한 출력 공간
    - 여러 task를 동시에 지원할 수 있는 출력 표현 설계
3. 데이터셋 집계 및 표준화
    - 다양한 데이터셋을 통합적으로 학습할 수 있도록 설계

핵심 아이디어: multi-view scene geometry에서 factored representation 사용
- scene을 pointmap으로 직접 표현하지 않고 다음 요소들의 집합으로 표현
    - depth maps
    - local raymaps
    - camera poses
    - metric scale factor
- 이 표현의 특징:
    - 출력 뿐 아니라 입력(선택적)으로도 활용 가능
    - 로봇 응용(예: 카메라 내, 외부 파라미터가 이미 주어진 경우)에 적합
    - 부분 주석(partial annotations)만 있는 데이터셋에서도 효과적 학습 가능

주요 기여
1. 통합 feed-forward 모델
    - 12개 이상의 문제 설정을 지원하는 multi-view metric 3D reconstruction 모델
    - end-to-end transformer로 효율적 학습 가능
    - 이미지 뿐만 아니라 카메라 내,외부 파라미터, 깊이, scale factor 등도 활용
2. Factored Scene Representation
    - 유연하게 분리된 입력 가능
    - metric 3D reconstruction의 효과적인 예측 가능
    - 중복이나 비용이 큰 후처리 없이 직접적으로 multi-view pixel-wise scene geometry와 camera를 계산
3. SOTA 성능
    - 특정 작업을 위한 기존 feed-forward 전문과 모델과 동등하거나 우수함
4. Open Source 공개
    - 데이터 처리, 추론, 벤치마킹, 학습 및 절제 연구 코드 공개
    - Apache 2.0 라이선스 하에 사전 학습된 모델 제공
    - 향후 3D/4D foundation model 연구를 위한 확장 가능하고 모듈화된 프레임워크 제공

## 2. 


## 3. MapAnything

입력
- N개의 RGB 이미지 $\hat{\mathcal{I}} = (\hat{I}_i)$
- 선택적 geometry 입력
    - 모든 뷰 혹은 일부 뷰에 대해 제공 가능
    - 종류
        - ray directions로 표현한 generic central camera calibration $\hat{\mathcal{R}} = (\hat{R}_i)_{i \in S_r}$
        - 카메라 포즈
            - 첫 번째 view $\hat{I}_1$을 기준 좌표계로 표현
            - Quaternion $\hat{\mathcal{Q}} = (\hat{Q}_i)_{i \in S_q}$
            - Translation $\hat{\mathcal{T}} = (\hat{T}_i)_{i \in S_d}$
        > $S_r, S_q, S_t, S_d$는 frame index 집합 $[1, N]$의 subset
    
출력
- 위 입력들을 