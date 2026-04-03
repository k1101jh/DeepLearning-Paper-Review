# Segment Any 3D Gaussians

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
- [paper](https://ojs.aaai.org/index.php/AAAI/article/view/32193) (aaai 2025)
- [github](https://github.com/Jumpat/SegAnyGAussians)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract


![alt text](./images/Fig%201.png)

## 3. Method

### 3.1 Preliminary

**3D Gaussian Splatting(3D-GS)**


### 3.2 Overall Pipeline

![alt text](./images/Fig%202.png)

**SAGA**
- $\mathcal{G}$: 사전훈련된 3D-GS model
- SAGA는 3D Gaussian $g$의 gaussian affinity feature $f_g \in \mathbb{R}^D$를 $\mathcal{G}$에 붙임
- 3D prompt 가능한 segmentation의 inherent multi-granularity 모호성을 처리하기 위해, SAGA는 이러한 features를 다양한 scale $s$에 맞는 서로 다른 scale-gated feature sub-space로 투영하기 위해 soft scale gate 메커니즘을 사용함
- affinity feature을 학습하기 위해, SAGA는 training set $\mathcal{I}$ 내의 각 이미지 I에 대해 SAM을 사용해서 multi-granularity mask $\mathcal{M}_I = {\text{M}_\text{I}^i \in {0, 1} ^ {HW} | i=1, ..., N_\text{I}}를 추출  
($N_\text{I}$: 추출된 마스크 개수)
- 각 마스크 $M_I^i$에 대해, 3D physical scale $s_{M_I^i}$는 카메라 포즈와 함께 $\mathcal{G}$에 의해 예측된 깊이를 사용해서 식 (2)와 같이 계산됨
- 이후 SAGA는 scale-aware contrastive learning 전략을 사용하여 multi-view 2D mask에 내재된 multi-granularity segmentation 능력을 scale-gated affinity features로 증류
- 학습 후, 주어진 scale에서 두 가우시안 간의 affinity feawture 유사성은 이들이 동일한 3D 타겟에 속하는지 여부를 나타냄
- inference
    - 특정 시점이 주어지면 2D visual prompts를 대응하는 3D scale-gated query features로 변환하여 3D affinity features와의 유사도를 평가하여 3D target을 segment
    - 잘 학습된 affinity features를 사용하면 단순 클러스터링을 통해 3D scene 분해가 가능
    - CLIP과 통합하면 language fields 없이도 open-vocabulary segmentation을 수행할 수 있음


### 3.3 Gaussian Affinity Feature


### 3.4 Scale-Aware Contrastive Learning
