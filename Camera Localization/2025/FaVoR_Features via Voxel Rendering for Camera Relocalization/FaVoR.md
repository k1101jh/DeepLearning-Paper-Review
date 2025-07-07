# FaVoR: Features via Voxel Rendering for Camera Relocalization


---

- SLAM

---

url
- [paper](https://ieeexplore.ieee.org/abstract/document/10943362)

---

목차

0. [Abstract](#abstract)
1. 

---


## Abstract

Camera Relocalization 방법
- dense image 정렬
- query 이미지로부터 camera pose 회귀

sparse feature matching은 효율적이고 일반적인 경량 접근 방식
- feature-based 방법은 종종 중요한 시점 및 외관 변화에 어려움을 겪어 매칭 실패 및 부정확한 자세 추정을 초래함
-> 2D feature의 global적으로 희소하지만 지역적으로 밀집한 3D 표현을 활용하는 새로운 접근 방식을 제안

- 여러 프레임에 걸쳐 landmark를 추적하고 삼각 측량하여 tracking 중에 관측된 image patch descriptors를 렌더링하는데 최적화된 sparse voxel map을 생성
- 상세 방법
    - 초기 pose가 주어지면 volumetric rendering을 사용하여 voxel에서 descriptors를 합성한 후, feature matching을 수행하여 

카메라 재위치 파악 방법은 밀집 이미지 정렬에서 쿼리 이미지로부터 직접 카메라 자세 회귀에 이르기까지 다양합니다. 이 중에서 희소 특징 매칭은 효율적이고 다재다능하며 일반적으로 경량한 접근 방식으로 두드러지며 수많은 응용 프로그램이 있습니다. 그러나 특징 기반 방법은 종종 중요한 시점 및 외관 변화에 어려움을 겪어 매칭 실패 및 부정확한 자세 추정을 초래합니다. 이러한 한계를 극복하기 위해, 우리는 2D 특징의 전 세계적으로 희소하지만 지역적으로 밀집한 3D 표현을 활용하는 새로운 접근 방식을 제안합니다. 우리는 여러 프레임에 걸쳐 랜드마크를 추적하고 삼각 측량하여 추적 중에 관찰된 이미지 패치 기술자를 렌더링하는 데 최적화된 희소 복셀 맵을 구성합니다. 초기 자세 추정이 주어지면, 우리는 먼저 부피 렌더링을 사용하여 복셀에서 기술자를 합성한 다음, 기능 매칭을 수행하여 카메라 자세를 추정합니다. 이 방법은 보지 못한 뷰에 대한 기술자를 생성할 수 있게 하여 시점 변화에 대한 강건성을 향상시킵니다. 우리는 7-Scenes 및 Cambridge Landmarks 데이터 세트에서 우리의 방법을 평가합니다. 우리의 결과는 실내 환경에서 기존의 최신 기능 표현 기술보다 우리의 접근 방식이 상당히 우수하다는 것을 보여주며, 중앙 번역 오류에서 최대 39%의 개선을 달성했습니다. 또한, 우리의 접근 방식은 야외 장면에 대해서도 다른 방법들과 비슷한 결과를 보이지만 더 낮은 계산 및 메모리 발자국을 가지고 있습니다.