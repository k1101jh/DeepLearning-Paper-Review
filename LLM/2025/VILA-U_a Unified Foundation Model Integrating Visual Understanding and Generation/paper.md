# VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- LLM
- Foundation Model

---
url:
- [paper](https://arxiv.org/abs/2409.04429#)
- [project](https://hanlab.mit.edu/projects/vila-u)
- [github](https://github.com/mit-han-lab/vila-u)

---
- **Authors**: Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, Yao Lu
- **Venue**: ICLR 2025

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. Method](#3-method)

---


## Abstract

**VILA-U**
- Video, Imagej, Language 이해 및 생성을 통합한 unified foundation model
- 기존의 VLM은 시각적 콘텐츠를 이해하고 생성하기 위해 별도의 모듈을 사용
    - 불일치, 복잡성 증가를 초래
- VILA-U는 single autoregressive next-token 예측 프레임워크를 사용하여 diffusion model과 같은 추가 구성요소가 필요하지 않음
    - 모델 단순화 & visual language 이해 및 생성에서 SOTA에 근접한 성과 달성
- 두 가지 주요 요인에 기인
    1. 사전학습동안 이산 visual token을 text 입력과 정렬하는 unified vision tower가 visual perception을 향상시킴
    2. autoregressive 이미지 생성이 고품질 데이터셋을 사용할 경우 diffusion model과 유사한 품질을 달성할 수 있음

## 1. Introduction

**LLM**
- 다양한 language 작업에서 우수한 능력을 보임
- 많은 연구자들이 VLM을 구축하고자 함
- visual language understanding
    - CLIP과 같은 vision 모델을 통해 LLM의 semantic 공간으로 투영되어 text-image 정렬 목표를 포함하믕로써 두 모달리티를 연결
- visual generation
    - text-guided image generation의 두 가지 방법
        1. diffusion model 사용
        2. Vector Quantization(VQ)를 통해 visual content를 discrete token으로 변환. 이후 autoregressive transformer을 사용하여 고품질&다양한 생성 수행
- multimodal framework
    - 두 가지 주요 접근 방식
        1. VQGAN 기반 tokenizer을 활용하여 visual input을 이산 토큰으로 변환. understanding과 generation을 위해 autoregressive model 활용
            - VQGAM 기반 encoder에서 나온 visual token의 semantic 정보가 부족함
            - downstream visual understanding 작업에서 심각한 성능 저하를 초래
        2. codebook을 활용하여 CLIP과 같은 pre-trained vision model에서 생성된 feature 양자화
            - CLIP feature은 풍부한 semantic 정보를 encoding
            - understanding 작업에서 일반적으로 더 뛰어난 성능 달성
            - tokenizer의 decoding 능력이 부족하여 생성된 visual token을 조건으로 visual output 생성을 위해 외부 visual generation model(Diffusion model 등)이 필요
            - infra 설계의 복잡성을 증가시킴
            - 사용 가능한 large-scale foundation model 훈련 파이프라인과 deployment 시스템은 next-token 예측을 사용하는 language modeling에 대해 고도로 최적화되어 있음
            - diffusion model을 지원하기 위해 추가 stack을 설계하고 유지하는 것은 상당한 엔지니어링 비용을 발생시킬 것

**VILA-U**
- end-to-end autoregressive framework
- visual & text 입력 모두에 대해 unified next-token 예측 목표를 가짐
- diffusion model과 같은 외부 구성 요소의 도움 없이 visual language understanding 및 generation task 모두에서 경쟁력 있는 성능을 달성할 수 있음
- vision & language modalities를 통합하기 위한 두 가지 핵심 원칙 확인
    1. 기존의 unified end-to-end autoregressive VLMs는 분리된 VQGAN token이 image reconstruction loss로만 학습되고 textual input과 정렬되지 않았기 때문에 경쟁력있는 visual understanding 성능을 달성할 수 없음
        - perception 능력을 향상시키기 위해 VQ vision tower pretraining동안 tex alignment를 도입하는 것이 중요
    2. 충분한 크기의 고품질 데이터셋으로 학습할 경우 Autoregressive image 생성은 diffusion model과 유사한 품질을 달성 가능
- unified foundation vision tower을
    - visual 입력을 vector quantization을 통해 분리된 토큰으로 변환하고 constrastive learning을 통해 이러한 token을 text 입력과 정렬
- multi-modal training
    - 소규모 고품질 image-text corpus에서 visual&textual token 모두에 대해 unified next-token 예측 목표를 활용
- 평가 과제
    - image-language understanding
    - video-language understanding
    - image generation
    - video generation 등
- end-to-end autoregressive model과 continuous-token VLMs 간의 visual understanding 성능 격차를 크게 좁힘 & 경쟁력 있는 native visual 생성 능력 도입

## 2. Related Work

## 3. Methods

### 3.1 Unified Foundation Vision Tower

- 다양한 visual understanding 및 generation 작업을 지원하기 위해, 적절한 visual feature을 제공하기 위해 unified foundation vision tower을 구축
- vision tower training에 text-image contrastive loss와 VQ-based 이미지 재구성 loss를 포함할 것을 제안
    - vision tower의 text alignment와 discrete tokenization 능력을 강화
    - 그림 2에 나타난 바와 같이, 이미지에서 추출된 visual feature은 주로 residual quantization을 통해 이산화됨
    - 이후, 하나의 경로에서 이산 visual feature가 decoder에 입력되어 이미지를 재구성하고 reconstruction loss 계산
    - 다른 경로에서는 discrete visual features와 text encoder가 제공한 textual feature 간의 image-text constrastive loss 계산
    - 이를 통해 vision tower은 VLM에서 understanding과 generation을 위해 discrete feature을 적절하게 추출하는 방법을 학습

**Unified Training Recipe**
- unified vision tower을 처음부터 훈련하는 것은 어려움
    - alignment와 reconstruction 작업이 각각 high-level semantic & low-level appearance feature을 요구하기 때문
    - 두 objective를 모두 사용하여 훈련 시 상충되는 목표가 발생할 수 있음
        - image reconstruction & contrastive loss를 모두 사용하여 vector-quantized vision tower을 처음부터 훈련하는 경우, Top-1 정확도가 5%에 불과함
- 해결책
    1. 모델에 text-image 정렬 능력을 갖추도록 함
    2. 정렬 능력을 유지하면서 reconstruction 학습
        - text-image 정렬을 잘 보장하기 위해 CLIP 모델에서 pretrained 가중치로 vision encoder와 text encoder 초기화
    3. text encoder을 freeze하고 모든 vision 구성 요소는 contrastive & reconstruction loss 모두를 사용하여 학습 가능하도록 함
        - constrastive loss는정렬 능력을 유지
        - reconstruction loss는 reconstruction 능력을 개발
    - 이 방법은 빠르게 수렴하며 강력한 성능 제공
    - pre-trained CLIP 가중치는 high-level priors 제공
        - 이러한 가중치로 초기화하면 vision encoder가 low-level & high-level feature을 더 빠르고 효율적으로 결합 가능
    - text 정렬 능력과 이미지 재구성 능력을 모두 갖춘 vision tower 훈련 가능
    $$
    \mathcal{L}_{total} = w_{contra}\mathcal{L}_{contra} + w_{recon}\mathcal{L}_{recon}
    \tag{1}
    $$
    > 각 가중치는 1로 설정됨

**Residual Vector Quantization**
- visual feature은 이산적으로 양자화 되어 있음
    - 표현 능력은 quantizer에서 사용하는 code size에 의존
    - 이 feature들이 high-level & low-level 모두 포함하기를 원하기 때문에, vector feature space에서 더 많은 용량이 필요함
    - 하위 작업에서 좋은 성능을 위해 더 큰 코드 필요
    - 각 이미지에 대해 코드가 너무 많으면 visual generation 과정에서 LLM이 생성해야 하는 token 수가 너무 많아져서 지연 시간이 크게 증가
- RQ-VAE를 따라 residual vector quantization 채택
    - 벡터 z를 D개의 이산 코드로 양자화
    $$
    \mathcal{RQ}(\rm{z};\mathcal{C}, D) = (k_1, \cdots, k_D) \in [K]^D
    \tag{2}
    $$


### 3.2 Unified Multi-Modal Generative Pre-Training

## 4. Experiments

### 4.1 Experimental Setup

