# RelPose++: Recovering 6D Poses from Sparse-view Observations

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- Relative Camera Pose Estimation

---
url:
- [paper](https://ieeexplore.ieee.org/abstract/document/10550461)

---
- **Authors**: Amy Lin, Jason Y. Zhang, Deva Ramanan, Shubham Tulsiani
- **Venue**: 3DV 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)

---

## ⚡ 요약 (Summary)


---

## 📖 Paper Review

## Abstract

- sparse-view 이미지 셋(2~8 이미지)에서 6D 카메라 자세 추정 작업
    - 거의 모든 재구성 알고리즘의 중요한 전처리 단계
    - 시각적 대칭성과 텍스처가 없는 표면을 가진 객체에 대해 희소한 view에서 도전적
- 제안 방법
    - RelPose 프레임워크 기반
    - attentional transformer layer을 사용하여 여러 이미지를 함께 처리
        - 객체의 추가 뷰가 특정 이미지 쌍에서 모호한 대칭을 해결할 수 있기 때문
    - 회전 추정의 모호성을 translation 예측과 분리하는 적절한 좌표계를 정의하여 카메라 translation도 보고할 수 있도록 네트워크 확장
    - seen 및 unseen 객체 범주 모두에서 이전 기술보다 6D pose prediction에서 큰 개선을 이룸
    - in-the-wild 객체에 대한 자세 추정 및 3D 재구성을 가능하게 함
