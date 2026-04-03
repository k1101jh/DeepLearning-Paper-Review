# Speedy MASt3R

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Image Matching

---

url:
- [paper](https://arxiv.org/abs/2503.10017) (CVPR 2025)

---
짧은 요약



요약

- 2D 이미지만으로는 dense SLAM 시스템을 수행하기에 부족함.
- 3D 정보는 view 간에 불변하기 때문에 3D 기하학 공간 활용 필요

---

**문제점 & 한계점**


---

목차

0. [Abstract](#abstract)
1. 

---


## Abstract

MASt3R
- DUSt3R을 활용하여 이미지 매칭을 3D 작업으로 재정의
- 일치성을 몇 배 향상시켜 이론적 보장을 유지하는 빠른 reciprocal 매칭 기법 도입
- 단점
    - 추론 속도
        - A40 GPU에서는 이미지 쌍당 198.16ms 지연 발생
        - ViT 인코더-디코더와 Fast Reciprocal Nearest Neighbor(FastNN) 매칭 단계의 계산 오버헤드

Speedy MASt3R
- 정확도를 유지하면서 추론 효율성을 향상
- FlashMatch를 포함한 여러 최적화 기법 통합
    - FlashAttention v2와 tiling 전략을 활용하여 계산 효율성 향상
- layer 및 tensor 융합을 통한 계산 그래프 최적화
- TensorRT(GraphFusion)을 통한 커널 자동 조정 및 메모리 접근 시간을 2차에서 선형으로 줄임
- block-wise correlation 점수를 vectorized 계산(FastNN-Lite)을 통해 가속
- mixed-precision 추론을 통해 FP16/FP32 하이브리드 계산(HybridCast)를 적용하여 속도를 높이면서 수치 정밀성을 보장
- Aachen Day-Night, InLoc, 7-Scenes, ScanNet1500, MegaDepth1500에서 평가
    - 정확도를 손상시키지 않으면서 추론 시간을 54% 단축(198ms -> 91ms 이미지 쌍당)

