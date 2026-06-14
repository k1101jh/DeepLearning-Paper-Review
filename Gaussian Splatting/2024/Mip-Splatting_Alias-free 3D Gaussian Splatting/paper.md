# Mip-Splatting: Alias-free 3D Gaussian Splatting

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Gaussian Splatting

---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)(CVPR 2024)
- [project](https://niujinshuchong.github.io/mip-splatting/)

---
- **Authors**: Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

3DGS
- 인상적인 novel view 합성 결과를 보임
- 높은 화질과 효율성 달성
- sampling 속도를 바꿀 때, focal length나 camera distance를 바꿀 때 artifact가 나올 수 있음
    - 3D frequency 제약이 부족함
    - 2D dilation filter을 사용하기 때문
- 이를 해결하기 위해
    - 3D smoothing 필터 도입
        - 입력 view에서 유도된 최대 sampling frequency에 기반하여 3D gaussian primitive 크기를 제한
        - zoom-in 시 high-frequency artifact를 제거할 수 있ㅇ므
    - 2D dilation을 2D Mip filter로 교체하면 aliasing과 dilation 이슈를 완화 가능
        - 2D mip filter는 2D box filter를 simulation


## 3. Preliminaries


## 4. Sensitivity to Sampling Rate

원본 가우시안의 최적화가 애매함을 겪음(Figure 1 참조)
- (a)의 3D 객체와 이를 3D 가우시안으로 근사한 것, 화면 공간으로 투사(파란색 픽셀)을 고려
    - 가우시안 커널(크기 $\approx$ 1픽셀)을 사용한 화면 공간 팽창(식 5) 때문에, (b)에서 Dirac $\delta$ 함수로 표현된 퇴화된 gaussian이 비슷한 이미지를 만들게 됨
    - 실제 장면의 high frequency 정보를 나타내기 위해 팽창된 2D 가우시안은 작아지게 됨
        - 가우시안이 작을수록 나타내는  frequency가 더 높기 때문
        - 크기가 체계적으로 과소평가됨
    - 유사한 샘플링 속도에서는 렌더링이 영향을 주지 않음(그림 1(a)와 (b) 비교)
    - 카메라를 확대하거나 가까이 이동하면 침식 효과가 나타남
        - 화면 공간에서 팽창된 2D 가우시안이 더 작아지기 때문
        - 렌더링 된 이미지는 고주파 잡음을 보임
        - 실제보다 객체 구조가 더 얇게 나타남(그림 1 (d))




## 5. Mip Gaussian Splatting

