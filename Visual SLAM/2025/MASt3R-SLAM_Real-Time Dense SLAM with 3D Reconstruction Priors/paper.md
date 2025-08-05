# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors



---

- SLAM

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html) (CVPR 2025)

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

MASt3R-SLAM
- 두 시점의 3D 재구성과 matching prior로 설계된 SLAM 시스템
- 고유한 카메라 중심만 있으면(fixed 또는 parametric 카메라 모델에 대한 가정을 하지 않아도) 야외 비디오 시퀀스에서 강력함
- pointmap 매칭, 카메라 추적 및 local 융합, 그래프 구성 및 loop closure, 2차 전역 최적화를 위한 효율적인 방법을 소개
- calibration이 있는 경우 간단한 시스템 수정으로 다양한 벤치마크에서 SOTA 달성
- 15FPS로 작동하면서 일관된 포즈와 dense geometry를 생성할 수 있는 plug and play monocular SLAM 시스템 제안

## 1. Introduction

SLAM(Simultaneous Localization and Mapping)
- 하드웨어 전문 지식과 calibration을 요구하기 때문에 아직 plug-and-play 알고리즘이 아님
- IMU와 같은 추가 센서 없이 최소한의 단일 카메라 설정에서는 정확한 포즈와 일관된 dense map을 제공하는 in-the-wild SLAM이 존재하지 않음

신뢰할 수 있는 dense SLAM 시스템
- 2D 이미지만으로 dense SLAM 수행에는 다음이 필요
    - 시간에 따라 변화하는 포즈와 카메라 모델
    - 3D scene 기하학
    - 이러한 고차원의 역문제를 해결하기 위해 다양한 선험적 priors가 제안됨
        - Single-view priors
            - monocular 깊이 및 normal
                - 단일 이미지에서 기하학을 예측하고자 함
                - 이러한 가정은 모호성을 포함하고 있으며, view 간의 일관성이 부족함
        - Multi-view priors
            - 광학 흐름 등
            - 모호성을 줄이지만, 포즈와 기하학을 분리하는 건 도전적
            - 픽셀 움직임은 외부 요소와 카메라 모델 모두에 의존
- 3D 장면은 view 간에 불변함.
    - 이미지를 통해 포즈, 카메라 모델, dense 기하학을 해결하는 데 필요한 통합 선험적 가정은 공통 좌표계에서의 3D 기하학 공간에 있음

DUSt3R[50]과 MASt3R[21]이 선도한 두 개의 view 3D 재구성 prior이 정제된 3D 데이터셋을 활용하여 SfM에서 패러다임 전환을 일으킴
- 공통 좌표계에서 두 개의 이미지로부터 pointmap을 직접 출력하여 앞서 언급한 하위 문제를 공동 프레임워크에서 암시적으로 해결
- 추후에는 이러한 prior이 상당한 왜곡을 가진 모든 종류의 카메라 모델에 대해 훈련될 것
- 3D prior은 더 많으 view를 수용할 수 있지만, SfM과 SLAM은 공간 희소성을 활용하고 중복을 피하여 대규모 일관성 달성
- two-view 아키텍처는 SfM의 기본 블록으로써 two-view 기하학을 반영하며, 이러한 모듈성은 효율적인 의사 결정과 강력한 합의가 이후 이뤄질 수 있는 기회를 제공함

최초의 실시간 SLAM 프레임워크 제안
- two-view 3D 재구성 prior을 추적, 맵핑, relocalization을 위한 통합 기초로써 활용
- 이전 작업에서는 이러한 prior을 비정렬 이미지 컬렉션에서 offline 환경의 SfM에 적용
- SLAM은 데이터를 점진적으로 수신하고 실시간 작업을 유지해야 함
    - 저지연 매칭, 신중한 맵 유지보수 및 대규모 최적화를 위한 효율적인 방법에 대한 새로운 관점을 요구
- SLAM에서 필터링 및 최적화 기술에서 영감을 받아, 프론트엔드에서 pointmap의 local filtering을 수행하여 백엔드에서 대규모 global 최적화를 가능하게 함
- 모든 광선이 통과하는 고유한 카메라 중심을 가지고 있다는 것 이외에 각 이미지의 카메라 모델에 대해 어떤 가정도 하지 않음
    - 일반적이고 시간에 따라 변하는 카메라 모델로 장면을 재구성할 수 있는 실시간 밀집 단안 SLAM 시스템이 구현됨
- calibration이 주어지면 경로 정확도와 dense geomtry estimation에서 SOTA 성능을 보임

논문의 기여
- two-view 3D 재구성 prior인 MASt3R를 기반으로 하는 최초의 실시간 SLAM 시스템
- pointmap 매칭, 추적 및 local fusion, 그래프 구축 및 loop closure, 이차 클로벌 최적화를 위한 효율적인 기술
- 일반적이고 시간에 따라 변하는 카메라 모델을 처리할 수 있는 SOTA dense SLAM 시스템
요약하자면, 우리의 기여는 다음과 같습니다: • MASt3R [21] 두 뷰 3D 재구성 프라이어를 기초로 하는 최초의 실시간 SLAM 시스템. • 포인트 맵 매칭, 추적 및 로컬 융합, 그래프 구축 및 루프 클로저, 그리고 이차 글로벌 최적화를 위한 효율적인 기술. • 일반적이고 시간에 따라 변하는 카메라 모델을 처리할 수 있는 최첨단 밀집 SLAM 시스템.