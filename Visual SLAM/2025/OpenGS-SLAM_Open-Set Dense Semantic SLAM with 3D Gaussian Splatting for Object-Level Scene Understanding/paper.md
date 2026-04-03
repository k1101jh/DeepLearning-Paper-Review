# OpenGS-SLAM: Open-Set Dense Semantic SLAM with 3D Gaussian Splatting for Object-Level Scene Understanding

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---



---

- Segmentation
- 3DGS SLAM

---

url:
- [paper](https://arxiv.org/abs/2503.01646) (arXiv 2025)

---
요약

---

## Abstract

기존 3DGS Dense Semantic SLAM 문제점
- 일반적으로 제한된 범주의 pre-trained classifier과 implicit semantic 표현에 제약을 받음
    - open-set 환경에서의 성능 저하
    - 3D object-level 장면 이해에 제약을 받음

OpenGS-SLAM
- open-set 환경에서 3DGS를 사용한 Dense Semantic SLAM 프레임워크
- 2D 기반 모델의 semantic label을 3D Gaussian 프레임워크에 통합
    - 견고한 3D object-level scene 이해 지원
- Gaussian voting splatting 도입
    - 빠른 2D label map 렌더링 및 scene 업데이트를 가능하게 함
- Confidence-based 2D Label Consensus 제안
    - 여러 view에서 일관된 labeling을 보장하기 위함
- Segmentation Counter Pruning 전략 사용
    - semantic 장면 표현의 정확도를 향상시키기 위함
- 기존 방법 대비 semantic rendering 속도 10배 향상, 저장 비용이 2배 낮음


## 1. Introduction

- Dense semantic SLAM은 로보틱스와 embodied intelligence의 근본적인 과제
- 3DGS 기반 Semantic SLAM
    - 2D semantic feature을 3DGS 표현에 embed
    - 2D 분할 결과를 생성하기 위해 제한된 수의 범주를 갖는 pre-trained classifier을 사용하는 경우가 많음
        - open-set 시나리오에서 효율성 제한됨
    - 모든 semantic 정보가 feature-embedded 표현 내에 암묵적으로 저장됨
        - 각 가우시안 행렬의 semantic label에 직접 접근하기 어려움
    - 이로 인해, 3D object-level scene 이해 및 상호작용을 필요로 하는 embodied intelligence와는 호환되지 않음

![alt text](./images/Fig%201.png)
> **Figure 1.**

OpenGS-SLAM
- 2D기반 비전 모델 통합
- 각 가우시안에 명확한 semantic label을 할당
- 온라인 semantic mapping을 지원
- foundation model은 쉽게 사용자 정의가 가능함
    - 예를 들어, SAM[14] 또는 MobileSAMv2[15]로 대체될 수 있음
- 생성된 semantic label을 3D 가우시안에 매핑하여 open-set 시나리오에서의 적용을 용이하게 함
- 각 가우시안에 명시적인 semantic label을 할당
    - 3D object-level scene 이해를 효율적으로 지원

세 가지 주요 과제
1. semantic label은 미분 불가능하여 빠른 label map 렌더링 및 scene 업데이트에 3DGS raterization을 사용할 수 없음
2. 여러 view에서 SAM 결과의 불일치로 인해 동일한 객체가 다양한 label을 받거나, 부분 또는 전체로 분할됨
3. training 중 view 제약이 적은 영역의 가우시안이 상당히 확장되어 부정확한 segmentation이 발생할 수 있음

해결방법
1. Gaussian Voting Splatting
    - 신속한 2D label map 렌더링 및 scene update를 용이하게 하기 위함
2. Confidence-based 2D Label Consensus
    - 일관된 segmentation을 달성하고 새로운 의미 정보를 장면에 효율적으로 통합하기 위함
3. Segmentation Counter Pruning 전략 제안
    - 더 정확한 장면 분할을 위함
- 추가: 2D semantic label을 생성하기 위한 ensemble semantic 정보 생성기 설계

논문의 기여
- OenGS SLAM 제안
    - 3DGS 기반 open-set semantic SLAM
    - 추가 학습 없이 다양한 기성 2D 기반 모델 지원
- Gaussian Voting Splatting 방법을 이용한 새로운 semantic 3DGS 표현 제안
    - 다른 방법에 비해 새로운 view에서 10배 빠른 렌더링, 2배 낮은 저장 비용
    - 더 정확한 semantic segmentaion을 통해 3D object-level scene 이해
- tracking, mapping, scene understanding에서 합성 및 실제 시나리오 모두에서 경쟁력 있는 성능 달성