# SOIC: Semantic Online Initialization and Calibration  for LiDAR and Camera

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Camera Calibration

---
url:
- [paper](https://arxiv.org/abs/2003.04260)

---
- **Authors**: 
- **Venue**: 

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)

---

## Abstract



SOIC
- 이전의 온라인 calibration 방법은 일반적으로 최적화를 위해 대략적인 초기 값에 대한 사전 지식이 필요
    - 의미론적 중심(SCs, Semantic Centroids)의 도입과 함께 초기화 문제를 Perspective-n-Point(PnP) 문제로 변환하여 이러한 제한을 제거
- 이 PnP 문제의 폐쇄형 해결책은 잘 연구되었으며 기존 PnP 방법에서 찾을 수 있다.

- point cloud의 의미론적 중심은 일반적으로 해당 이미지의 의미론적 중심과 정확하게 일치하지 않기 때문에 nonlinear refinement 프로세스 후에도 매개변수의 정확도가 향상되지 않는다.  
-> point cloud와 이미지 데이터 모두의 의미론적 요소들 사이의 대응에 대한 제약을 기반으로 하는 비용 함수가 공식화된다.
- 비용 함수를 최소화하여 최적의 외재적 매개변수를 추정
- KITTI 데이터셋에서 GT 또는 예측된 semantic을 사용하여 제안된 방법을 평가
- 실험 결과 및 baseline과의 비교를 통해 초기화 전략의 타당성과 보정 접근법의 정확성을 검증

## 1. Introduction

LiDAR(Light Detection and Ranging) 센서
- 넓은 범위에서 공간 데이터를 강력하게 얻을 수 있다.
- 해상도가 낮고 색상이 없다.

카메라 센서
- 고해상도
- 빛에 민감하다
- 거리 정보가 없는 RGB 이미지를 얻을 수 있다.

두 센서의 조합은 모바일 로봇 공학 및 자율주행 차량의 애플리케이션을 위한 일반적이고 필수적인 설정

이러한 조합을 기반으로 MV3D, AVOD, FPonitNet과 같은 신경망을 제안하여 기존 객체 감지 및 세분화 작업의 성능을 향상시킬 수 있다. 

결합을 위한 중요한 전제 조건:
- 정확한 외부 보정: 두 센서의 좌표계 간의 변환 매트릭스를 추정

많은 LiDAR 카메라 보정 방법이 제안됨
수동 개입
- 일반적으로 feature을 선택하거나 point cloud와 이미지의 feature 간의 상관 관계를 조정하기 위해 필요

프로세스의 편의성을 향상시키기 위해 감지된 feature에 자동으로 대응할 수 있는 방법이 있다.
- 체스판과 같은 특정 목표가 필요

유연성을 높이기 위해 온라인 목표가 없는 방법이 제안됨
- 그 중 한 가지는 관측된 point cloud와 이미지 데이터 사이의 강도 또는 가장자리의 상관 관계를 활용하여 외부 매개변수를 찾는 관측치 기반 방법
    - 신경망을 통해 이러한 상관관계를 학습하기 위한 학습 기반 방법이 제안됨
    - 일반적으로 초기 추측값의 정확도에 따라 성능이 크게 달라진다.
- 다른 방법으로는 두 센서의 움직임을 일치시켜 보정 매개변수를 얻는 방법이 있다.
    - 높은 정확도를 달성하기 위해서는 충분하고 정확한 ego-motion 추정이 필요하다.

본 논문에서는 pointcloud와 이미지 데이터의 semantic 분할 결과를 활용하여 초기화 문제를 해결하기 위해 Semantic calibration 방법인 SOIC(Semantic Online and Initialization Calibration)을 제안

Fig 1에서 