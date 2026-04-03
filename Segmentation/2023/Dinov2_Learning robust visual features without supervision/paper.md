# DINOv2: Learning Robust Visual Features without Supervision

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


---

## 📌 Metadata
---
분류
- Segmentation
- Self-supervised Learning
- Foundation Model
---
url:
- [paper](https://arxiv.org/abs/2304.07193)
- [project](https://dinov2.metademolab.com/)
---
- **Authors**: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, et al.
- **Venue**: TMLR 2023

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [3. Data Processing](#3-data-processing)
- [4. Discriminative Self-supervised Pre-training](#4-discriminative-self-supervised-pre-training)

---

## ⚡ 요약 (Summary)
- **Problem**: 컴퓨터 비전에서 텍스트 가이드 사전 학습은 캡션이 이미지의 풍부한 정보를 충분히 담지 못해 저해상도 픽셀 레벨 정보를 잃기 쉬움. 또한 기존 자기주도 학습(SSL)은 대규모 데이터 확장 시 성능 저하나 불안정성 문제가 있었음.
- **Goal**: 대량의 정제된 데이터를 활용하여 이미지 및 픽셀 레벨 모두에서 즉시 사용 가능한(Frozen) 범용 시각적 특징을 학습하는 DINOv2 제안.
- **Key Method**: 
    - **Discriminative SSL**: DINO와 iBOT 손실을 결합하고, 대규모 학습을 위해 안정화된 아키텍처와 하이퍼파라미터를 재검토함.
    - **Data Pipeline**: NLP 파이프라인에서 영감을 받아 메타데이터 없이 데이터 유사성만을 활용해 비선별 이미지 컬렉션(LVD-142M)을 필터링 및 재조정하는 자동 파이프라인 구축.
    - **Efficiency**: 대규모 배치 학습을 지원하도록 최적화하여 이전 SSL 방식보다 약 2배 빠르고 메모리 사용량은 3배 적음.
- **Result**: 별도의 파인튜닝 없이 고정된 특징만으로도 다양한 벤치마크(세그멘테이션, 깊이 추정 등)에서 최첨단 성능을 기록하며 공개된 최고의 약지도(weakly-supervised) 모델과 경쟁함.

---

## 📖 Paper Review

## Abstract


## 1. Introduction

학습 과제에 구애받지 않는 사전 학습 표현은 NLP에서 표준이 됨
- 이러한 feature은 fine-tuning 없이 그대로 사용할 수 있음
- task-specific models가 만들어내는 것보다 더 나은 성능을 하위 과제에서 달성 가능
- language modeling이나 world vector같은 감독이 필요 없는 사전 목표를 사용하여 대량의 raw text로 사전 학습

컴퓨터 비전에서의 유사한 'foundation' model
- 이미지 수준(예: 이미지 분류)과 픽셀 수준(예: segmentation) 모두에서 즉시 사용할 수 있는 시각적 특징을 생성해야 함
- 대부분 text-guided 사전학습을 사용하여 feature 학습을 안내하는 것에 집중되어 있음
- 텍스트 기반 사전 학습은 caption이 이미지의 풍부한 정보를 단순히 근사하기 때문에 보존할 수 있는 저옵를 제한함
- 복잡한 pixel-level 정보는 이러한 지도학습에서는 드러나지 않을 수 있음
- 이미지 encoder은 text-image 정렬 corpora를 필요로 함으로, raw data로 학습하는 유연성은 제공하지 않음

컴퓨터 비전에서 self-supervised learning
- text-guided pretraining의 대안
- 개념적으로 language modeling과 같은 pretext tasks에 더 가까움
- 이미지 및 pixel-level 정보를 포착 가능
- self-supervised learning 모델이 출력하는 특징은 다양한 유용한 특성을 나타냄
- 대부분의 self-supervised learning은 소규모 선별 데이터셋인 ImageNet-1k에서의 pretrining 맥락에서 이루어짐
- ImageNet-1k를 넘어 접근법을 확장한 시도돌은 주로 비선별 데이터셋에 초점을 맞춰 feature의 품질이 크게 떨어지는 결과를 초래하는 경우가 많았음
    - 좋은 feature을 생성하는데 필수적인 데이터 품질과 다양성에 대한 통제가 부족하기 때문

이 논문의 방법
- 대량의 정제된 데이터로 pretraining할 경우, self-supervised learning이 범용 시각적 특징을 학습할 잠재력이 있는지 탐구
- iBOT과 같이 이미지 및 patch-level에서 feature을 학습하는 기존 self-supervised 방법을 다시 살펴봄
    - 더 큰 데이터셋 관점에서 설계 선택을 재검토
- 모델과 데이터 크기를 확장할 때 discriminative self-supervised learning을 안정화하고 가속화하는데 초점
    - 유사한 discriminative self-supervised learning보다 약 2배 빠르고 메모리 사용량은 3배 적음
    - 더 큰 배치로 더 긴 학습 수행 가능

방대한 비정제 이미지 컬렉션에서 데이터셋을 필터링하고 재조정하는 자동 파이프라인 구축
- NLP에서 사용되는 파이프라인(Wenzek et al. 2020)에 영감을 받음
- metadata 대신 데이터 유사성을 활용
- 수동 주석이 필요하지 않음

실제 이미지 처리에서의 어려움
- concept 재조정 및 몇 가지 지배적인 mode에 과적합되는 것을 피하는 것
- 단순한 clustering 접근 방식이 문제를 해결하는데 비교적 효과적임

DINOv2
- 다양한 사전 학습된 visual models
- 서로 다른 ViT 아키텍처로 저자의 데이터로 학습됨
- 코드 및 모델 공개
- 이미지 및 pixel-level의 다양한 컴퓨터 비전 벤치마크에서 DINOv2 검증
    - transferable frozen feature 학습에는 self-supervised pretraining만으로도 좋은 후보임
        - 공개된 최고의 weakly-supervised model과 경쟁할 수 있음

## 3. Data Processing

## 4. Discriminative Self-supervised Pre-training

- discriminative self-supervised 방법으로 feature 학습
    - DINO와 iBOT loss를 SwAV의 centering과 결합한 것으로 볼 수 있음
- feature을 분산시키기 위한 정규화 항과 짧은 고해상도 학습 단계를 추가

- Image-level objective(Caron et al. 2021)
    - student 및 teacher network에서 추출된 feature 간의 cross-entropy loss를 고려
    - 두 feature 모두 같은 이미지의 서로 다른 crop에서 얻어진 ViT 클래스 토큰에서 나옴
    - student class token을 student DINO head를 통해 전달
        - 이 head는 vector 형태의 점수(prototype scores)를 출력하는 MLP 모델
    - 이후 softmax를 적용해서 $p_s$를 얻음
    - 마찬가지로 teacher class token에 teacher DINO head를 적용해서 teature prototype score을 얻고 softmax를 적용
    - 이후 이동 평균으로 centering(또는 Sinkhorn-Knopp centering)을 적용하여 $p_t$를 얻음
    - DINO loss:
    $$
    \displaystyle
    \mathcal{L}_{DINO} = - \sum p_t \log p_s
    $$