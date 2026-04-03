# Monocular Depth Estimation Based On Deep Learning: An Overview

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Depth Estimation
- Monocular Depth Estimation
- Survey

---
url:
- [paper](https://link.springer.com/article/10.1007/s11431-020-1582-8) (Science China Technological Sciences 2020)
- [arXiv](https://arxiv.org/abs/2003.06620)

---
- **Authors**: Chaoqiang Zhao, Qiyu Sun, Chongzhen Zhang, Yang Tang, Feng Qian
- **Venue**: Science China Technological Sciences 2020

---

## 📑 Table of Contents
- [Abstract](#abstract)


---

## Abstract

깊이 정보는 자율 시스템이 환경을 인식하고 자체 상태를 추정하는 데 중요

Structure from motion 및 stero vision matching과 같은 기존의 깊이 추정 방법은 여러 관점의 feature 대응을 기반으로 함
- 예측된 깊이 맵은 희박하다.

단일 이미지에서 깊이 정보를 추론하는 것(monocular 깊이 추정)은 ill-posed problem이다.

최근 딥러닝을 기반으로 한 monocular 깊이 추정이 널리 연구되고 있으며, 정확도 면에서 유망한 성능을 달성

조밀한 깊이 맵은 심층 신경망에 의해 단일 이미지에서 end-to-end로 추정

본 논문에서는 딥러닝을 기반으로 한 monocular 깊이 추정 방법을 조사

1. 딥러닝 기반 깊이 추정에서 널리 사용되는 데이터셋과 평가 지표를 정리
2. 다양한 training 방식에 따라 기존의 대표적인 방법을 검토: supervised, unsupervised, semi-supervised 방법
3. 도전 과제에 대해 논의하고 단안 깊이 추정에 대한 향후 연구를 위한 아이디어 제공

## 1. Introduction


## 3. Monocular depth estimation based on deep learning

인간들은 세계의 priori information을 사용할 수 있기 때문에 단일 이미지에서 깊이 정보를 인식할 수 있다.
이에 영감을 받은 이전 작업들은 일부 기하학적 구조(하늘, 땅, 건물)간의 관계와 같은 일부 사전 정보를 결합하여 단일 이미지 깊이 추정을 달성

심층 신경망은 블랙박스로 간주될 수 있다.
- supervised signals의 도움으로 깊이 추론을 위한 일부 구조적 정보를 학습

가장 큰 과제:
- GT 데이터셋이 충분하지 않다.
    - 획득에 비용이 많이 든다.

supervised, unsupervised, semi-supervised 방법을 리뷰

unsupervised & semi-supervised의 훈련 과정은 단안 비디오 또는 스테레오 이미지쌍에 의존
훈련된 깊이 네트워크는 테스트 중에 단일 이미지에서 깊이 맵을 예측

### A. Supervised monocular depth estimation

supervised methods의 기본 모델:
supervisory 신호는 깊이 맵의 실측 자료를 기반으로 함 -> 단안 깊이 추정은 회귀 문제로 간주될 수 있다.
