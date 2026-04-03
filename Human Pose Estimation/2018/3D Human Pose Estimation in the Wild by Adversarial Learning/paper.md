# 3D Human Pose Estimation in the Wild by Adversarial Learning

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Human Pose Estimation
- Adversairal Learning

---

url:
- [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_3D_Human_Pose_CVPR_2018_paper.html)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

강력한 Deep Convolutional Neural Networks(DCNN) 덕분에 단안 이미지에서 3D HPE에 놀라운 발전이 이뤄짐

실제 이미지에 대한 3D 포즈 주석을 얻기는 어렵다.  
-> in-the-wild 이미지에서 3D 인간 포즈를 추정하는 것이 어렵다.

본 논문에서는 fully annotated dataset(주석이 달린 데이터)에서 학습한 3D 인간 포즈 구조를 2D 포즈 주석만 있는 in-the-wild 이미지로 추출하는 적대적 학습 프레임워크를 제안

포즈 추정 결과를 제한하기 위해 hard coding된 규칙을 정의하는 대신 예측된 3D 포즈와 Ground-Truth를 구별하기 위해 새로운 multi-source discriminator(판별기)을 설계
- pose estimator가 in-the-wild 이미지에서도 인체측정학적으로 유효한 포즈를 생성하는 데 도움이 된다. 

판별기를 위해 신중하게 설계된 information source가 성능을 향상시키는 데 필수적

신체 관절 사이의 pair-wise 상대적인 위치와 거리를 계산하는 기하학적 descriptor을 판별자에 대한 새로운 information source로 설계

새로운 기하학적 descriptor을 사용한 적대적 학습 프레임워크의 효율성은 공개 벤치마크에 대한 실험으로 입증
기존 SOTA 접근 방식에 비해 성능을 크게 향상시킨다.

## 1. Introduction

인간의 자세 추정은 컴퓨터 비전에서 근본적이지만 어려운 문제

이미지나 비디오가 주어졌을 때 신체 부위의 2D 또는 3D 위치 추정