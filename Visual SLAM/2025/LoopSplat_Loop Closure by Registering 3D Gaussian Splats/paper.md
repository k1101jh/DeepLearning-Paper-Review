# LoopSplat: Loop Closure by Registering 3D Gaussian Splats



---

- Visual SLAM
- Gaussian Splatting

---

url
- [paper](https://ieeexplore.ieee.org/abstract/document/11125565/) (3DV 2025)
- [project](https://loopsplat.github.io/)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

3DGS를 기반으로 한 SLAM은 최근 보다 정확하고 밀도 높은 3D 장면 지도를 만드는 데 가능성을 보임
- 기존 3DGS 기반 방법들은 loop closure 및/또는 global bundle adjustment를 통해 장면의 전역 일관성을 해결하지 못함

LoopSplat
- RGB-D 이미지를 입력으로 받아 3DGS submap과 frame-to-model tracking을 통해 밀도 있는 매핑을 수행
- 온라인으로 loop closure을 trigger하고 3DGS registration을 통해 submap 간의 상대 loop edge 제약을 직접 계산
    - 기존 global-to-local point cloud registration 대비 효율성과 정확도 향상
- robust pose graph 최적화 formulation을 사용하고 submap을 강제로 정렬하여 전역 일관성 달성
- 합성 Replica와 실제 TUM-RGBD, ScanNet, ScanNet++ 데이터셋 평가 결과 기존의 dense RGB-D SLAM 방법에 비해 tracking, mapping, rendering에서 경쟁력 있거나 우수한 성능을 보임

## 1. Introduction

## 2. Related Work

## 3. LoopSplat