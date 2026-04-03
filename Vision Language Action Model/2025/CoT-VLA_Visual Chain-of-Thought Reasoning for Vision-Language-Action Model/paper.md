# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Vision Language Action Model
- Chain-of-Thought Reasoning

---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html) (arXiv 2025)
- [project](https://cot-vla.github.io/)

---
- **Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin
- **Affiliation**: NVIDIA, Stanford University, MIT
- **Venue**: CVPR 2025

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

**Vision-Language-Action Models(VLAs)**
- 사전 학습된 vision-language model과 다양한 로봇 시연을 활용
- 일반화 가능한 센서모터 제어를 학습하는데 잠재력을 보임
- 현재의 VLAs 문제점
    - 주로 직접적인 입력-출력 매핑에 초점을 맞춤
        - 복잡한 조작 작업에 필수적인 중간 추론 단계가 결여됨
        - 시간적 계획 / 추론 능력이 부족함

**CoT-VLA**
- VLAs에 명시적 시각적 chain-of-throught(CoT) reasoning을 통합하는 방법 제안
    - 시각적 목표를 미래 이미지 frame으로 autoregressively하게 예측
    - 짧은 행동 sequence를 생성
- SOTA 7B VLA
- real-world 조작 작업에서 SOTA VLA 모델을 17% 능가
- 시뮬레이션 벤치마크에서 6% 우수한 성능을 보임

![alt text](./images/Fig%201.png)


## 1. Introduction

