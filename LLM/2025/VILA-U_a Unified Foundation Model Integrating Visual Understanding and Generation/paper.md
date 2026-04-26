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
- Video, Image, Language 이해 및 생성을 통합한 unified foundation model
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
    - CLIP과 같은 vision 모델을 통해 LLM의 semantic 공간으로 투영되어 text-image 정렬 목표를 포함함으로써 두 모달리티를 연결
- visual generation
    - text-guided image generation의 두 가지 방법
        1. diffusion model 사용
        2. Vector Quantization(VQ)를 통해 visual content를 discrete token으로 변환. 이후 autoregressive transformer을 사용하여 고품질&다양한 생성 수행
- multimodal framework
    - 두 가지 주요 접근 방식
        1. VQGAN 기반 tokenizer을 활용하여 visual input을 이산 토큰으로 변환. understanding과 generation을 위해 autoregressive model 활용
            - VQGAN 기반 encoder에서 나온 visual token의 semantic 정보가 부족함
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
        - perception 능력을 향상시키기 위해 VQ vision tower pretraining동안 text alignment를 도입하는 것이 중요
    2. 충분한 크기의 고품질 데이터셋으로 학습할 경우 Autoregressive image 생성은 diffusion model과 유사한 품질을 달성 가능
- unified foundation vision tower
    - visual 입력을 vector quantization을 통해 이산 토큰으로 변환
    - constrastive learning을 통해 이러한 token을 text 입력과 정렬
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

![alt text](images/Fig%201.png)
> **Figure 1. 제안 프레임워크의 multi-modal 훈련 및 inference 과정 개요**  
> visual 입력은 이산 token으로 변환되고 text token과 연결되어 multi-modal token sequence를 형성  
> 모든 token은 다음 token 예측 과정에 포함되어 unified training objective를 가능하게 함  
> inference 중에는 output token은 text detokenizer 또는 vision tower decoder에 의해 디코딩되어 multi-modal 콘텐츠 생성

### 3.1 Unified Foundation Vision Tower

![alt text](images/Fig%202.png)
> **Figure 2. unified foundation vision tower 개요**  
> 입력 이미지가 주어지면, vision encoder에 의해 추출된 feature은 residual quantization을 사용하여 이산화  
> 이산화된 vision features는 이미지 재구성을 위해 vision decoder에 입력되고 text-image 정렬 수행에 사용됨  
> 이 과정동안 reconstruction loss & contrastive loss가 계산됨

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
    \displaystyle
    \mathcal{RQ}(\rm{z};\mathcal{C}, D) = (k_1, \cdots, k_D) \in [K]^D
    \tag{2}
    $$
    > $C$: codebook  
    > $K = |C|, k_d$: depth d에서의 z의 code
    - $\rm{r}_0 = \rm{z}$에서 시작해서, vector 양자화를 재귀적으로 수행
    $$
    \displaystyle
    \begin{aligned}
    k_d &= \mathcal{Q}(\rm{r}_{d-1}, \mathcal{C}), \\
    \rm{r}_d &= \rm{r}_{d - 1} - \rm{e}(k_d),
    \tag{3}
    \end{aligned}
    $$
    > $\rm{e}$: codebook embedding table  
    > $\mathcal{Q}$: standard vector quantization:
    $$
    \displaystyle
    \mathcal{Q}(\rm{z};\mathcal{C}) = \argmin_{k \in [K]} ||\rm{z} - \rm{e}(k)||_2^2
    \tag{4}
    $$
    > $\rm{z}$: depth dim에 대한 합: $\hat{\rm{z}} = \Sigma_{i=1}^D \rm{e}(k_i)$
- 각 depth에서 양자화 error을 줄이기 위해 code를 선택
- 하나의 vector를 양자화하기 위해 $D$개의 code를 갖고 있어 더 세밀한 근사와 더 큰 feature space를 제공
- multi-modal 학습 및 추론 동안, LLM은 code embedding만 예측하면 됨
- 서로 다른 depth에 있는 code는 code embedding을 초기 입력으로 사용하는 depth transformer에 의해 순차적으로 생성됨
- latency를 거의 증가시키지 않으면서 vision tower의 표현 능력 향상 가능

### 3.2 Unified Multi-Modal Generative Pre-Training

**Encoder**
- visual input을 순차적으로 처리하여 1D token sequence 생성
    - 이 sequence는 text token과 concat되어 multi-modal sequence를 형성
    - modality를 구분하고 visual content 생성을 가능하게 하기 위해
        - image token의 시작과 끝에 <image_start> 및 <image_end> 토큰 삽입
        - video token의 시작과 끝에 <video_start> 및 <video_end> 토큰 삽입

**Pre-training data form**
- text와 visual token 간의 다양한 결합 형태를 활용하여 이해 및 생성 모두를 용이하게 함
- [image, text], [text, image], [text, video] 형태를 사용
- 각 쌍에서 후자 modality에만 supervision 추가
    - 무조건적인 content 생성을 피함
    - modality 정렬을 촉진
- 향상된 이해를 위해 text와 이미지를 교차 결합하는 형태 사용
    - supervision loss는 text에만 적용됨
    - [video, text] 형태는 효율성 문제로 pre-training중에는 제외
    - supervised fine-tuning 시 이를 포함하면 뛰어난 video 이해 능력을 효과적으로 얻을 수 있음

**Training Objective**
- next-token prediction objective로 LLM 훈련 가능
    - 시각적 token과 text token이 모두 이산적
- visual token에 residual quantization을 사용하기 때문에 text token과 visual token의 훈련 목표가 약간 다름
- text token:
    - negative log-likelihood loss
    $$
    \displaystyle
    \mathcal{L}_{text} = - \sum_{i=1}^T \log P_\theta(y_i | y_{<i}),
    \tag{5}
    $$
    > $T$: multi-modal sequence 길이  
    > $i$: text token이 position i에 나타날 때에만 계산됨
- visual token:
    - 각 visual position $j$에서 code의 depth-tacked 구조를 도입
    - RQ-VAE에서 소개된 depth transformer을 활용
        - position $j$에서 visual token에 대해 LLM이 생성한 코드 embedding $h_j$가 주어졌을 때, depth transformer은 autoregressively하게 $D$개의 residual token($k_{j1}, \dots, k_{jD}$)를 예측
        - 학습 중 depth transformer의 깊이 $d$에서 입력 $v_{jd}$는 $d > 1$일 경우, 깊이 $d-1$까지의 code embedding의 합으로 정의됨
        $$
        \displaystyle
        v_{jd} = \sum_{d'=1}^{d-1} \rm{e}(k_{jd'})
        \tag{6}
        $$
    > $v_{j1} = h_j$
    - depth transformer은 이전 $d-1$까지의 추정치를 바탕으로 feature $\hat{z}_j$에 대한 finer estimation을 위해 다음 code를 예측
    - visual token에 대한 negative log-likelihood loss:
    $$
    \displaystyle
    \mathcal{L}_{visual} = -\sum_{j=1}^T \sum_{d=1}^D \log P_{\delta}(k_{jd} | k_{j,<d})
    \tag{7}
    $$
    > $T$: multi-modal sequence 길이  
    > $j$: visual token이 position $j$에 나타날 때만 계산됨  
    - multi-modal pre-training동안, depth transformer의 가중치는 무작위로 초기화되고 LLM과 함께 업데이트됨

## 4. Experiments

### 4.1 Experimental Setup

- base LLM: LLaMA-2-7B
- Vision Tower: SigLIP-Large-patch16-256/SigLIP-SO400M-patch14-384를 vision encoder architecture로 선택
- RQ-VAE에서 residual quantizer, depth transformer 및 decoder architecture 채택
- quantizer codebook 크기: 16384
- 이미지/비디오 해상도: 256 x 256 / 384 x 384
- 각 이미지는 residual depth D=4을 갖는 16 x 16 x 4 코드로 변환됨
- 각 비디오는 residual depth D=16을 갖는 27 x 27 x 16 코드로 변환됨
    - patch14 모델을 사용해서 27 x 27 크기
    - 14 * 27 = 378(6pixel 버림)
- COYO-700M에서 vision tower을 학습
- 평가
    - zeroshot classification, reconstruction
        - ImageNet 사용
    - visual understanding
        - ShareGPT4V의 1M [image, text] 데이터
        - MMC4의 6M interleaved text & image 데이터 활용
    - visual generation
        - 내부 데이터셋에서 선별한 15M 고품질 [text, image] 데이터와 OpenVid의 1M [text, video] 데이터 통합
        - Classifier-free guidance사용
            - CFG 값: 3
        - 이미지 생성을 위해 MJHQ-30K & GenAI-Bench 사용
            - MJHQ-30K:
            생성된 이미지와 30K 고품질 이미지 간의 FID를 채택하여 이미지 생성의 전반적인 능력 반영
            - GenAI-Bench:
            이미지 생성 모델의 종합적인 생성 능력을 반영하는 도전적인 image-to-text 생성 벤치마크
        - 비디오 생성을 위해 VBench 사용
            - VBench:
            종합 벤치마크. 생성 품질을 여러 잘 정의된 차원으로 분해하여 세밀하고 객관적인 평가를 용이하게 함

### 4.2 Unified Foundation Vision Tower

![alt text](./images/Table%201.png)
> **Table 1.**  
> unified vision tower가 ImageNet에서 zero-shot image classification을 수행할 때 reconstruction FID(rFID)와 Top-1 정확도

- 재구성 및 ImageNet에서 zero-shot 이미지 classification 수행
- 제안 모델은 VQ-GAN보다 우수한 재구성 결과를 달성
- 동일한 코드 형태를 사용할 때, rFID가 RQ-VAE보다 약간 낮음
    - 이미지 이해 향상을 목표로 한 constrastive loss 도입으로 인해 재구성 품질이 감소한 것으로 예상
- text 정렬 능력
    - unified vision tower은 256/384 해상도에서 Top-1 정확도 73.3/78.0을 달성

### 4.3 Quantitative Evaluation

**Visual Understanding Tasks**

![alt text](./images/Table%202.png)
> **Table 2. image-based visual language benchmarks에서 leading methods와 비교**  
> 제안 방법의 성능은 leading VLMs와 거의 비슷  
> 동일한 LLM 크기 하에서도 많은 방법을 큰 차이로 능가  
> * 표시는 해당 데이터셋의 training split에 있는 이미지가 VLM 학습 중 관찰되었음을 나타냄

![alt text](./images/Table%203.png)
> **Table 3. video-based visual language benchmarks에서 leading methods와 비교**  
> 제안 방법 성능은 SOTA VLMs에 근접  
> 동일한 LLM 크기에서도 많은 방법을 능가함.  
> (discrete visual token type을 사용하더라도)

- Table 2/Table 3: 각각 image-language/video-language 벤치마크에서 제안 방법과 다른 주요 VLMs의 비교 요약
- CLIP과 같은 foundation model이 생성하는 연속적인 visual tokens가 주류
- VQGAN 기반의 이산 visual token은 text와의 정렬이 덜 되어 있어 VLM의 visual understanding  작업 성능에 부정적인 영향을 미침
- unified foundation vision tower을 통해, 제안 모델은 이산 visual token을 사용하더라도 leading VLM에 가까운 성능을 낼 수 있음

**Visual Generation Tasks**

![alt text](./images/Table%204.png)
> **Table 4. MJHQ-30K evaluation benchmark에서 다른 visual generation 방법과의 비교**

![alt text](./images/Table%205.png)
> **Table 5. GenAI-Bench에서 다른 visual generation 방법과의 비교**  
> 제안 방법이 이전의 autoregressive visual generation 방법보다 성능이 뛰어남  
> 더 나은 text following 능력이 필요한 고급 프롬프트에 대해서도 제안 방법은 훨씬 적은 학습 데이터만으로도 diffusion-based 방법과 비교해 상대적으로 작은 성능 차이를 가질 수 있음

![alt text](./images/Table%206.png)
> **Table 6. VBench에서 다른 visual generation 방법과의 비교**

- Table 4.
    - VILA-U는 다른 autoregressive 방법보다 더 나은 FID를 달성할 수 있음
    - 일부 diffusion-based 방법과 비교할 만한 성능을 보임
- Table 5.
    - GenAI-Bench에서 제안 방법과 다른 visual generator의 정량적 결과를 요약
    - 제안 방법
        - 수십억 수준의 image-text 쌍으로 학습된 diffusion-based 방법에는 뒤쳐짐
        - SDv2.1 및 SD-XL과 비교하면 고급 프롬프트에도 비슷한 성능을 보임
        - VILA-U가 unified training framework를 통해 visual&textual modalities 간의 상관관계를 효과적으로 학습할 수 있음을 추가로 보임
- Table 6.
    - 영상 생성 평가.
    - 제안 방법을 VBench에서 평가
    - 제안 방법은 CogVideo보다 더 나은 성능을 달성하고, Open-Sora와 비교할 만한 성능을 보임

### 4.4 Qualitative Evaluation

**Visual Understanding**

![alt text](./images/Fig%203.png)
> **Figure 3. VILA-U는 vision encoder의 text alignment 덕분에 비디오의 자막을 정확하게 작성하고 모든 세부 사항을 포착**

![alt text](./images/Fig%204.png)
> **Figure 4.**  
> VILA-U는 좋은 visual question answering 능력을 가짐  
> image와 질문은 VQAv2 데이터셋의 test 분할에서 가져옴

![alt text](./images/Fig%205.png)
> **Figure 5.**  
> VILA-U는 좋은 in-context learning 능력을 가짐  
> 두 쌍의 image-text pairs와 세 번째 이미지를 context로 유도하기 위해 제공

![alt text](./images/Fig%206.png)
> **Figure 6.**  
> VILA-U는 여러 이미지에 대해 올바르게 추론하지 못함

- VILA-U의 종합적인 visual understanding 과제에서의 효율성 검증(Fig 3, Fig 4 참조)
- visual captioning 및 visual question answering을 포함한 다양한 과제에서 VILA-U의 다재다능함 확인
- multi-image understanding, in-context learning과 같은 VILA의 중요한 기능을 일부 계승(Fig 5, Fig 6 참조)

**Visual Generation**

![alt text](./images/Fig%207.png)
> **Figure 7.**  
> VILA-U는 여러 이미지에 대해 올바르게 추론할 수 있음

- Fig 7에서 visual generation 결과의 예시 제시
- 비교적 작은 data corpus로도 훈련될 수 있음
- 이미지 생성 & 비디오 생성 모두에 사용 가능
- 사용자의 입력에 맞는 보기 좋은 이미지와 연속적인 비디오 생성 가능

## 5. Ablation Study

### 5.1 Impact of Constrastive Loss to Visual Understanding

![alt text](./images/Table%207.png)
> **Table 7. visual understanding에서 contrastive loss의 영향**

- Vision Tower 학습에서 contrastive loss을 포함하여 텍스트 정렬 능력을 부여
    - multi-modal 학습 동안 이러한 text 정렬 능력은 modality 융합을 향상시킴
    - downstream visual language 작업에서 성능을 높이는데 중요함
- 정렬의 중요성 검증(표 7 참조)
    - contrastive loss를 사용하여 학습한 경우와 사용하지 않은 경우의 Vision Tower을 학습시키고 visual language 이해 성능에 미치는 영향 평가
    - COYO-700M에서 2,500만개 데이터를 무작위로 샘플링하여 Vision Tower 학습
    - multi-modal 학습에는 ShareGPT4V와 MMC4를 text-image 및 text-video 데이터 없이 사용
    - 데이터셋 크기를 25M에서 700M으로 확장하면 성능이 더욱 향상됨

### 5.2 Impact of Constrastive Loss to Visual Generation

![alt text](./images/Table%208.png)
> **Table 8. visual generation에서 contrastive loss의 영향**

- 생성 성능에 대한 contrastive loss의 영향 입증
- text-to-image pretraining 수행
- LLaMA-2-7B 대신 Sheared-LLaMA-1.3B를 LLM으로 활용
    1. rFID가 1.3인 RQ-VAE를 vision tower로 사용
    2. Unified Vision Tower을 사용
- 실험 결과(표 8 참조)
    - Unified Vision Tower은 contrastive loss로 인한 낮은 rFID 결과 때문에 MJHQ-30K에서 RQ-VAE보다 약간 더 안좋은 FID 결과를 보임

### 5.3 Impact of Classifier-Free Guidance

![alt text](./images/Table%209.png)
> **Table 9. CFG의 영향**

- visual content generation에서 classifier-free guidance를 채택
- 256 해상도 모델에서 CFG 값의 영향 조사
- 실험 결과, CFG 3.0이 가장 좋은 FID 점수를 제공(표 9 참조)

## 6. Conclusion and Limitation

**VILA-U**
- video, image 및 언어 이해와 생성 작업을 하나의 autoregressive next-token 예측 프레임워크로 통합
- visual generation과 이해를 통합하기 위해 diffusion과 같은 추가 요소를 사용하는 VLMs보다 간결함
- autoregressive 방법이 SOTA VLMs와 비교할 만한 성능을 달성할 수 있음을 보여줌
- contrastive loss의 도입은 vision tower의 재구성 능력에 영향을 미침
- 생성 및 이해 능력의 균형을 맞추는 것은 추가적인 탐구가 필요
- 이해 작업과 생성 작업 간의 유의미한 시너지나 상호 강화 효과가 관찰되지 않음