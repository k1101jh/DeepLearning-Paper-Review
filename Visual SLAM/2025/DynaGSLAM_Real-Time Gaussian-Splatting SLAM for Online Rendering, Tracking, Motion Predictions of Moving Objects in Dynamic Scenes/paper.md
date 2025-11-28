# RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes


---

- SLAM
- Dynamic SLAM
- 3D Gaussian Splatting

---

url:
- [paper](https://arxiv.org/abs/2503.11979) (arXiv 2025)
- [project](https://blarklee.github.io/dynagslam/)

---
요약

- 문제점
    - 

---

## Abstract

SLAM
- 고품질과 빠른 매핑이 중요함
- 3DGS SLAM은 pointcloud-SLAM과 비교했을 때 고품질 텍스쳐로 보이지 않는 뷰를 합성함

GS-SLAM
- GS-SLAM은 bundle adjustment의 정적인 가정을 위반하는 이동 객체가 장면을 점유하면 실패함
- 움직이는 GS의 업데이트 실패는 정적인 GS에 영향을 미침.  
-> 전체 맵을 오염시킴
- 이동 객체 고려 연구
    - GS 렌더링에서 움직이는 영역을 감지하고 제거함(anti dynamic GS-SLAM). 정적 배경만 GS-SLAM의 혜택을 받음

![alt text](./images/Fig1.png)

> **Figure 1. **

## 1. Introduction

3DGS SOTA 방법은 정적인 장면에서만 고려됨
- Fig 2의 정적인 영역도 동적 영역으로 잘못 취급되어 결과가 나빠질 수 있음

동적 객체를 감지하고 제거하는 방법
- localization에는 도움이 됨
- 동적 객체를 제거하면 GS 맵에서 정적인 배경만 렌더링 가능

동적 객체를 GS로 표현하기 위한 SOTA 방법
- GS에 직접 시간 차원을 추가하는 방법 등
- GS를 오프라인 방식으로 영상 시퀀스 당 몇시간씩 학습하므로 온라인 SLAM에 적합하지 않음

본 논문의 기여
- DynaGSLAM
    - moving object의 모션 예측을 지원하는 첫 번째 real-time Gaussian-Splatting based SLAM
- novel dynamic GS management 알고리즘 제안
    - adding, deleting, tracking, updating, predicting dynamic GS

## 2. Related Works

## 3. Problem Formulation



## 4. Dynamic GS Architecture

목표:  
동적 객체가 있는 scene에서, unknown camera pose에서 촬영된 RGBD 입력을 받아서
카메라 포즈와 time-varying scene 표현을 찾아내는 것

