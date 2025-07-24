# FAR: Flexible, Accurate and Robust 6DoF Relative Camera Pose Estimation



---

- Visual SLAM

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Rockwell_FAR_Flexible_Accurate_and_Robust_6DoF_Relative_Camera_Pose_Estimation_CVPR_2024_paper.html) (CVPR 2024)
- [github](https://crockwell.github.io/far)

---

요약

---


---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

이미지 사이의 상대 카메라 자세 추정
- 대응 관계를 찾고 기본 행렬을 해결하는 방법
    - 높은 정확도 제공
- 신경망을 사용하여 자세 직접 예측
    - 겹침이 제한될 때 더 강력
    - 절대 변환 스케일(absolute translation scale)을 추론할 수 있음
    - 정확도가 감소함

제안 방법
- 두 방법의 장점 결합
- 정확하고 강건한 결과를 보임
- 변환 scale을 정확하게 추론
- Transformer 기반
    1. 해결된 pose estimation과 학습된 pose estimation 간의 균형을 배움
    2. 해결자(solver)을 안내하는 사전 정보를 제공
- Matterport3D, InteriorNet, StreetLearn, Map-free Relocalization에서 6DoF 포즈 추정에서 SOTA 성능을 보임

## 1. Introduction

Relative camera pose estimation
- 대응 관계를 추정하고 이를 해결하여 포즈를 추정
    - 종종 sub-degree 에러 발생
    - large view 변환(Fig 1. 왼쪽)이 발생하는 경우 잘 작동하지 않음
    - Fundametal 또는 Essential matrix를 계산하기 때문에 scale을 복구할 수 없음
- 포즈를 직접 계산
    - 정확하지는 않음
    - 더 강건하고 translation scale을 제공(Fig 1. 왼쪽 및 오른쪽)

