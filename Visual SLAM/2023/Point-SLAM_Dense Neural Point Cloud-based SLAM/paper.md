# Point-SLAM: Dense Neural Point Cloud-based SLAM


---

- RGBD SLAM

---

url
- [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Sandstrom_Point-SLAM_Dense_Neural_Point_Cloud-based_SLAM_ICCV_2023_paper.html) (ICCV 2023)
- [github](https://github.com/eriksandstroem/Point-SLAM)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

Point-SLAM
- 단안 RGBD 입력을 처리하기 위함
- neural scene representation의 feature을 point cloud에 고정
    - input-dependent data-driven manner 방식으로 반복적으로 생성된 point cloud
- tracking과 mapping은 동일한 point-based neural scene representation을 사용한 RGBD-based re-rendering loss를 최소화하여 수행됨
- 기존 dense neural SLAM 방법은 scene features를 sparse grid에 고정
- 제안한 point-based 접근 방법은 입력의 information density에 따라 anchor point density를 동적으로 조정 가능
    - 세부 정보가 적은 영역에서는 실행 시간과 메모리 사용을 줄임
    - 세밀한 세부 정보를 해결하기 위해 높은 point 밀도 달성
- Replica, TUM-RGBD, ScanNet 데이터셋에서 tracking, mapping, 및 렌더링 정확도 측면에서 기존의 dense neural RGBD SLAM 방법보다 더 우수하거나 경쟁력 있는 성능을 보임


## 1. Introduction


## 2. Related Works