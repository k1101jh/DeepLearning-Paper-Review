# A ConvNet for the 2020s

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Vision Model
- Classification
- CNN

---
url:
- [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)(CVPR 2022)

---
- **Authors**: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
- **Venue**: CVPR 2022

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

- 일반적인 컴퓨터 비전 과제(객체 검출, semantic 등)에는 일반 ViT 적용이 어려움
    - ConvNet prior을 도입한 다시 도입한 Swin Transformer가 이러한 문제를 해결
- 이러한 하이브리드 접근법의 효과는 Transformer 자체의 우수성 덕분
    - Convolution의 귀납적 편향 덕이 아님




## 2. Modernizing a ConvNet: a Roadmap


### 2.5. Large Kernel Sizes

![alt text](./images/Fig%203.png)
> **Figure 3.** block 수정 및 사양
> - (a) ResNeXt block
> - (b) inverted bottleneck block 생성
> - (c) spatial depthwise conv lyaer을 위로 이동시킴

- ViT는 non-local self attention으로 각 레이어가 global 수용 영역을 갖게 함
- 표준 방법은 작은 커널의 컨볼루션 레이어를 쌓는 것

**Moving up depthwise conv layer**

- 큰 커널을 위해서는 depthwise conv 레이어 위치를 위로 올려야 함(Fig 3. (b) 참조)
- 복잡하고 비효율적인 모듈(MSA, 큰 커널 conv)는 채널 수가 적고, 효율적이고 조밀한 1x1 layer가 주요 작업을 수행
- 이 중간 단계 덕분에 FLOPs가 4.1G로 줄고, 성능이 79.9%로 떨어짐

**Increasing the kernel size**

- 위 과정이 끝나면 MLP는 입력 차원보다 네 배 넓어짐
- 큰 커널 사이즈의 conv를 채택하는 이점이 상당함
- 3, 5, 7, 9, 11등 여러 커널 크기로 테스트해봤을 때 성능은 79.9%(3x3)에서 80.6%(7x7)로 증가했지만 FLOPs는 거의 변하지 않음
- 7x7에서 큰 커널의 이점이 포화점에 도달