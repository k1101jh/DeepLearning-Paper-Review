# MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM

---

- SLAM

---

url:
- [paper](https://arxiv.org/pdf/2504.02437) (arXiv 2025)

---
요약

---


## Abstract

MonoGS++ 개요
- 정의: MonoGS++는 3D Gaussian 표현을 활용하며, RGB 입력만으로 동작하는 빠르고 정확한 SLAM(Simultaneous Localization and Mapping) 기법.
- 기존 방식과 차별점:
    - 기존 3D Gaussian Splatting(GS) 기반 방법들은 깊이 센서에 크게 의존
    - MonoGS++는 RGB 입력만 사용
    - 실시간 온라인 비주얼 오도메트리(VO)를 통해 희소 포인트 클라우드 생성

3D Gaussian 매핑 개선 사항
- 동적 3D Gaussian 삽입(Dynamic Insertion)
    - 이미 잘 재구성된 영역에 중복 Gaussian이 추가되는 것을 방지
- 선명도 향상 Gaussian 밀집화 모듈(Clarity-Enhancing Densification)
    - 질감이 부족한 영역(texture-less areas)과 평면(flat surfaces) 처리 성능 향상
- 평면 정규화(Planar Regularization)
    - 평면 구조의 정확한 재구성을 지원

성능
- 정확도:
    - 합성 데이터셋 Replica와 실제 데이터셋 TUM-RGBD에서 최첨단 수준(state-of-the-art)에 필적하는 정밀한 카메라 추적 성능 달성

- 속도:
    - 이전 SOTA인 MonoGS [8] 대비 FPS 5.57배 향상


## 3. Method
