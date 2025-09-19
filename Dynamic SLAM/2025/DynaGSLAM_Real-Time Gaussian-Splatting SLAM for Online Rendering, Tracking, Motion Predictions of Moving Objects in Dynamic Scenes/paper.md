# DynaGSLAM: Real-Time Gaussian-Splatting SLAM for Online Rendering, Tracking, Motion Predictions of Moving Objects in Dynamic Scenes

---

- SLAM
- Dynamic SLAM

---

url:
- [paper](https://arxiv.org/abs/2503.11979) (arXiv 2025)

---
요약

---


## Abstract

![alt text](./images/Fig%201.png)

> **Figure 1**  
> DynaGSLAM은 동적 장면에서 동적 객체의 온라인 고품질 렌더링을 위한 최초의 실시간 GS 기반 SLAM  
> 온라인 RGBD 프레임을 통해 과거/미래의 연속적인 객체 움직임을 추적(보간)/예측(외삽, extrapolate)할 수 있게 함  
> TUM 데이터셋에서의 GSmapping 렌더링을 보임  
> 첫 번째 행: RGB 렌더링  
> 두 번째 행: 렌더링과 실제 값 사이의 절대 오류

3D Gaussian Splatting
- 우수한 렌더링 품질과 속도를 갖춘 명시적 3D 표현 방식
- 기존 pointcloud 기반 SLAM과 달리, GS-SLAM은 입력 카메라 뷰로부터 photometric 정보를 학습하여 관측되지 않은 시점의 뷰를 고품질 텍스처로 합성 가능

기존 GS-SLAM의 한계
- 장면에 움직이는 객체가 존재하면, Bundle Adjustment의 정적(scene static) 가정이 깨짐
- 움직이는 GS의 잘못된 업데이트가 정적 GS까지 오염시켜, 장기 프레임에서 전체 맵 품질이 저하됨
- 일부 동시 연구들은 움직이는 객체를 고려했으나, 단순히 동적 영역을 검출하고 제거하는 방식("anti" dynamic GS-SLAM)만 사용  
-> 정적 배경만 GS의 이점을 활용

DynaGSLAM
- 최초의 실시간 GS-SLAM
- 고품질 온라인 GS 렌더링
- 동적 객체 추적
- 동적 객체의 움직임 예측(motion prediction)
- 정확한 ego motion(카메라의 움직임) 추정


## 1. Introduction



## 2. 

## 3. 

## 4. 

## 5. 

## 6. Experiments

**Baselines**
- RTGSLAM[36]
- SplatTAM[17]
- GSSLAM[31]
- GSLAM[51]

- dynamic 객체를 제거하는 "anti" dynamic GS-SLAM 방법은 코드를 사용할 수 없기에 재현 불가

**데이터셋:**
- OMD[14]
- TUM[42]
- BONN[34]

**실험 설정:**
- TUM, OMD 데이터셋
    - DepthAnythingV2[49]를 사용하여 부드러운 깊이를 얻고 원래 깊이 맵으로 실제 scale을 복구
- BONN 데이터셋
    - raw depth sensor 측정을 사용

- 2D motion mask 내의 동적 개체에 대해서만 PSNR로 평가
- 카메라 위치 파악의 ATE(Absolute Trajectory Error)을 평가

**Dynamic Mapping Results**

**정량적 비교(Tables 1 & 2)**

![alt text](./images/Table%201.png)

> **Table 1. Comparison on Bonn Dataset**  
> 제안 방법은 다른 baseline을 모두 능가  
> '*': reproduction 없이 [46]에서 보고된 성능 리스트  
> [46]에 대해 DynaPSNR 측정 불가

![alt text](./images/Table%202.png)

> **Table 2. Comparison on TUM Dataset**  
> 제안 방법은 모든 지표에서 다른 방법보다 우수함

- DynaGSLAM은 모든 dnyamic sequence에서 다른 SOTA GS-SLAM보다 우수한 성능 달성
- DynaPSNR 지표에서 향상은 제안한 동적 GS management 알고리즘의 효과를 입증

**정성적 비교(Figures 1 & 4)**
- 렌더링 품질이 다른 방법보다 뛰어나며, 동적 객체 주변에서 차이가 두드러짐
- 소수의 GS만 사용하여 효율성을 높였기 때문에 일부 floater 아티팩트 발생
- SplaTAM[17]은 500프레임 이후 메모리 부족(OOM) 발생  
-> 동적 GS의 이상치(outlier)을 삭제하지 못해 메모리 해제 실패

**Dynamic Motion Tracking & Prediction**

평가 방법(Figure 5)
- Tracking
    - 시작 프레임 $t_0$과 종료 프레임 $t_5$만 주어졌을 때, 중간 시점 $t_3$을 보간하여 렌더링
- Prediction
    - 동일한 입력으로 미래 시점 $t_10$을 외삽(extrapolation)하여 렌더링
- 5프레임마다 단 하나의 프레임만 제공하기 때문에 시간적으로 희소한 데이터에서 unseen 프레임($t_3, t_10$)을 재구성해야 하기에 매우 어려운 작업

비교 기준
- 기존 방법들은 GS에서 동적 객체 모션을 모델링하지 않음
- RTGSLAM[36]을 baseline으로 사용. $t_5$ 시점에 동적 객체가 없다고 가정하고 목표 시점 viewpoint에서 렌더링

결과
- DynaGSLAM은 이동하는 객체(박스, 사람)을 정확히 예측
- 예측 결과가 모션 마스크(투명 흰색)과 잘 겹침
-> 동적 모션 추정 및 렌더링 정확도 입증