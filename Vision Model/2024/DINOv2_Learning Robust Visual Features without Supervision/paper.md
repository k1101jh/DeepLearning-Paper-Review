# DINOv2: Learning Robust Visual Features without Supervision

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Vision Model

---
url:
- [paper](https://openreview.net/forum?id=a68SUt6zFt) (TMLR 2024)
- [paper](https://arxiv.org/abs/2304.07193)(arXiv 2023)
- [github](https://github.com/facebookresearch/dinov2)

---
- **Authors**: Maxime Oquab, Timothée Darcet, Théo Moutakanni,
Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza,
Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal1,
Patrick Labatut, Armand Joulin, Piotr Bojanowski
- **Venue**: TMLR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

대규모 데이터에서 model pretraining을 위한 컴퓨터 비전
- 미세 조정 없이도 이미지 분포와 작업 전반에서 작동하는 feature을 생성
- 어떤 시스템에서든 이미지 사용을 크게 단순화 할 수 있음

논문의 작업
- Self-supervised learning 방법이 다양한 출처의 충분히 선별된 데이터로 학습되면 이러한 특징을 생성할 수 있음
- 기술적 기여
    - 대규모 학습 가속화 및 안정화
- 데이터 측면
    - self-supervised 방법에서 선별되지 않은 데이터 대신, 전용의 다양하고 선별된 이미지 데이터셋을 구축할 자동화된 파이프라인 제안
- 모델 측면
    - 10억개의 파라미터를 갖는 ViT 모델을 학습
    - 이를 일련의 작은 모델로 증류


## 1. Introduction

- NLP에서는 학습 작업에 구애받지 않는 pretrained 표현이 표준이 됨
    - 이러한 feautre을 finetuning 없이 사용할 수 있음
    - 작업 특화 모델이 생성한 성능보다 뛰어난 하위 작업 성능 달성 가능
    - supervision이 필요 없는 전처리 목표(예: 언어 모델링) 또는 word vector을 사용하여 대량의 raw text에 대한 pretraining에서 비롯됨
- 컴퓨터 비전에서도 유사한 foundation model이 등장할 것
    - 이미지 수준(분류 등) 및 픽셀 수준(분할 등) 모두에서 어떤 작업에도 바로 적용 가능한 visual feature을 생성해야 함
    - text-guided pretraining(text 형태의 감독을 사용하여 feature 학습을 guide)에 초점을 맞춤
        - 이미지에 대한 정보 유지에 제한을 둠
        - caption은 이미지의 정보를 단지 근사할 뿐
        - 복잡한 픽셀 수준 정보는 이러한 supervision에서 나타나지 않을 수 있음
        - 이러한 이미지 encoder은 정렬된 text-image corpora가 필요하므로 raw data만으로 학습할 수 있는 유연성이 없음

self-supervised learning
- text-guided pretraining의 대안
- 개념적으로 언어 모델링과 같은 pretext task(전제 과제)에 더 가까움
- 이미지와 픽셀 수준에서 정보를 포착 가능
- 출력 feature이 다양한 유용한 속성을 나타내는 것으로 보여짐
- 다양한 응용 가능
- 문제점
    - 소규모 curated 데이터셋인 ImageNet-1k에서의 pretrainng 맥락에서 이뤄짐
    - ImageNet-1k 이상의 규모로 접근을 확장하려는 시도가 있었지만
        - feature의 품질이 현저히 낮아지는 비 curated 데이터셋에 초점을 맞춤
- 양질의 feature 생성에 필수적인 데이터 품질과 다양성에 대한 통제가 부족했음

논문의 제안
- 대량의 선별된 데이터로 사전학습 시, self-supervised learning이 일반 복적의 visual feature을 학습할 가능성이 있는지 탐구
- 이미지 및 patch 수준 모두에서 feature을 학습하는 기존의 discriminative(판별적) self-supervised 접근법(iBOT 등)을 다시 살펴봄
    - 더 큰 데이터셋 관점에서 일부 설계 선택을 재고
- 모델과 데이터 크기를 확장할 때 판별적 self-supervised learning을 안정화하고 가속
- 유사한 판별적 self-supervised 방법보다 약 2배 빠르고 3배 적은 메모리 요구
- 더 큰 배치로 더 긴 학습 활용 가능

pretraining dataset
- 자동화 파이프라인 구축
    - 방대한 uncurated 이미지로부터 데이터셋을 필터링하고 재균형하기 위함
    - NLP에서 사용되는 파이프라인(Wenzek et al. 2020)에서 영감을 받음
    - 외부 메타데이터 대신 데이터 유사성을 사용
    - 수동 주석이 필요하지 않음
- wild image 처리에서의 주요 어려움
    - 개념 재균형화(rebalance concepts)
    - 몇 가지 지배적인 모드에 과적합되는 것을 피하는 것
    - 단순한 clustering 접근법이 이 문제 해결에 잘 작동
- 다양한 142M 이미지 corpus 수집

DINOv2
- 다양한 pre-trained visual model
- 다양한 Vision Transformer(ViT) 아키텍처로 저자의 데이터에서 학습됨
- DINOv2의 품질을 이미지 및 픽셀 수준에서 검증
- self-supevised pretraining 만으로도 transferable frozen feature 학습에 좋은 후보임
    - 최고 수준의 공개적으로 사용 가능한 weakly-supervised model들과 경쟁할 수 있음

## 2. Related Work


## 3. Data Processing


## 4. Discriminative Self-supervised Pre-training

## 5. Efficient implementation

## 6. Ablation Studies

