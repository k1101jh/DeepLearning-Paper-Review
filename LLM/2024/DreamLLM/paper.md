# DreamLLM: Synergistic Multimodal Comprehension and Creation

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Multimodal LLM

---
url:
- [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1a7a22152cd21f0ca3c0f8139bb32905-Abstract-Conference.html) (ICLR 2024)

---
- **Authors**: Runpei Dong, chunrui han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, Li Yi
- **Venue**: ICLR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)



---

## Abstract

**DreamLLM**
- 다재다능한 Multimodal Large Language Model(MLLMs)를 처음으로 달성
    - multimodal의 이해와 생성 간의 시너지를 바탕으로 함
- 두 가지 기본 원칙
    1. raw multimodal 공간에서 직접 샘플링하여 언어와 이미지 posteriors 모두를 생성 모델링 하는데 중점을 둠
        - 외부 특징 추출기(CLIP 등)에서 발생하는 제한 및 정보 손실을 회피
    2. raw interleaved documnets를 생성하여 텍스트와 이미지 내용을 비롯해 비구조적 레이아웃까지 모델링하도록 촉진
        - 조건부, 주변부 및 결합 multimodal 분포를 효과적으로 학습할 수 있도록 함
- free-form interleaved content를 생성할 수 있는 최초의 MLLM
- zero-shot multimodal generalist로서 뛰어난 성능을 보임

## 1. Introduction

![alt text](./images/Fig%201.png)
> **Figure 1. Vision-language(VL) foundation model의 개념적 비교**  
> - (a) CLIP과 유사한 모델은 두 개의 tower을 활용하여 VL 표현을 명시적으로 정렬
> - (b) Flamingo/BLIP 유사 모델은 단일 MLLM을 사용하여 VL 표현을 통합 매니폴드 공간으로 인코딩
>     - 이러한 모델은 언어만 출력하기에 완전한 autoregressivity를 갖추지 못함
> - (c) 동시 MLLMs는 시각적 출력을 CLIP 표현과 정렬하지만, 이 정렬이 raw 데이터 공간이 아닌 중간 공간에서 발생
>     - Emu와 같은 모델은 원시 이미지 생성을 위해 Stable Diffusion의 2단계 미세 조정이 필요함
>     - raw interleaved 문서 생성에도 한계가 있음
> - (d) DreamLLM은 raw language와 이미지 입력을 통합된 auto-regressive 방식으로 생성하여 자연스럽게 섞인 생성을 가능하게 함
>     - non-autoregressive generation loss만 기록됨

**MLLM**
- GPT 스타일 LLM의 확장으로 등장
- 일반적으로 CLIP feature과 같은 이미지를 multi modal 입력으로 통합
- 목적: language posterior을 통해 multi modal 조건부 또는 marginal distribution을 포착
- 이미지, text 또는 두 가지를 생성하는 멀티모달 창조는 언어와 이미지 posterior을 동시에 학습하는 보편적 생성 모델 필요
- 일부 연구에서는 MLLM을 이용한 조건부 이미지 생성에 성공을 거둠
    - MLLM이 사전 학습된 CLIP encoder와 명시적으로 일치하는 이산적 또는 연속적 조건부 embedding을 생성하도록 강제(Fig 1 참조)
    - 이후 사전 학습된 Stable Diffusion 모델에서 이미지 생성에 사용될 수 있음
    - 내재된 modality gap 때문에 CLIP semantics는 주로 modality-shared-information을 중심으로 함
        - modality-specific knowledge를 간과함
    - 창의성에서 미미한 개선만을 보임

**DreamVLA의 두 가지 설계 원칙**
1. Generating Everything as It Is
    - DreamLLM은 모든 modalities raw data를 입력으로 사용하며, 출력으로도 사용함(truly end-to-end)
        - 기존 작업은 CLIP embedding처럼 중간 이미지 표현을 생성
    - 이해력을 희생하지 않고 이미지 posterior을 배우게 함
        - dream query 도입
            - MLLM이 encode한 semantic을 캡슐화하는 학습 가능한 embedding 집합
        - MLLM의 출력 공간을 변경하는 것을 피함
    - 원시 이미지는 semantic에 조건을 둔 SD image decoder에 의해 디코딩됨
    - 사전 학습된 SD는 score function으로 작용
    - image posterior은 pixel 공간에서 직접 샘플링을 통해 모델링됨
        - score distillation에 의해 촉진됨
2. Interleaved Generative Pre-Training($\mathcal{I}$-GPT)
    - DreamLLM은 인터넷에서 교차된 multimodal corpora를 생성하도록 학습됨
    - 교차된 이미지-텍스트 multimodal 입력을 인코딩하고 디코딩
    - 기존 방법처럼 multimodal 입력을 인코딩하는 것과 달리 교차된 multimodal 출력을 디코딩하는 것은 복잡한 교차 레이아웃 구조와 이미지의 긴 context 요구사항 때문에 어려움
    - text 내 이미지의 위치를 예측하는 고유한 \<dream\> 토큰을 사용하여 교차 레이아웃 학습을 해결
    - DreamLLM의 인과적(causal) nature을 활용하여, 모든 콘텐츠는 길이에 상관없이 모든 이전 multimodal context와 함께 생성됨
    - 이러한 interleaved generative pretraining($\mathcal{I}$-GPT)은 문서 내 이미지와 텍스트의 모든 결합, 주변 및 조건부 분포를 본질적으로 형성
        - 창작을 기반으로 한 DreamLLM의 이해력과 그 반대의 학습적 시너지를 이끌어냄

실험
- 다양한 vision-language 이해, 콘텐츠 생성, language only 작업에 걸쳐 실험
- DreamLLM이 zero-shot multimodal generalist로서 우수한 성능을 보임
    - MS-COCO에서 8.46 FID를 달성
    - MMBench와 MM-Vet 평가에서 각각 49.1/35.9 점수를 기록하여 새로운 기준을 설정
- 이해와 생성 간의 학습 시너지에 대해 탐구
    - $\mathcal{I}$-GPT 사전학습을 통해, GPT-4로 선별된 교육 지침 데이터에 대한 supervised fine-tuning 이후 사람의 프롬프트를 따라서 interleaved content 생성
    - 양쪽에서 학습 시너지를 발휘하며 free-form interleaved content를 생성할 수 있도록 한 최초의 MLLM 연구


## 2. Background & Problem Statement


## 3. DreamLLM

![alt text](./images/Fig%202.png)
> **Figure 2. DreamLLM framework 개요**  
> - 상호 연관된 문서들이 입력으로 사용됨
> - text와 이미지는 모두 순차적이고 이산적인 token embedding으로 인코딩되어 MLLM 입력으로 사용됨
> - 특수 <dream> 토큰은 이미지를 생성할 위치 예측
> - 일련의 dream query가 MLLM에 입력되어 전체적인 과거 의미를 포착
> - 이미지들은 queried semantics를 조건으로 하는 SD 이미지 decoder에 의해 합성됨
> - 합성된 이미지는 이후 이해를 위해 다시 MLLM에 입력됨

**DreamLLM**
- ShareGPT에서 학습된 LLaMA를 기반으로 한 Vicuna
    - causal decoder-only LLM $\mathcal{F}_\theta$를 model foundation으로 구축
- visual encoder $\mathcal{H}_\phi$: CLIP-Large
- visual embedding projection을 위한 linear layer $\mathcal{M}_\zeta$를 사용
- 이미지 합성을 위해 이미지 decoder로 Stable Diffusion(SD)를 사용
- linear layer인 condition projector $\mathcal{M}_\psi$ 사용

### 3.1 End-to-End Interleaved Generative Pretraining ($\mathcal{I}$-GPT)

- 모든 natural 문서는 text-image interleaved information의 전달자로 간주될 수 있음
- text만 있는 데이터, 이미지만 있는 데이터, text-image 쌍 데이터는 서로 다른 modality 구성으로 된 interleaved corpora의 특수한 구성으로 볼 수 있음
- 모델이 모든 가능한 분포를 형성하는 free-form interleaved documnet를 학습하고 생성할 수 있는 능력을 갖추는 것이 중요

**Interleaved Structure Learning**
- 교차 구조를 모델링하기 위해 교차 sequence는 이미지 앞에 특수 토큰 \<dream\>을 추가하여 조작됨
- 이미지가 나타나는 위치를 나타내는 \<dream\> 토큰을 예측하도록 학습됨
- 이후 조건부 이미지 합성이 수행됨
- DreamLLM이 이 토큰이 예측될 때 "자유 의지"로 이미지 생성

**Conditional Synthesis through Score Distillation**

- CLIP semantics와 MLLM 간의 잠재적인 충동을 피하기 위해 학습 목표와 조건부 embedding 설계
- 길이가 Q인 학습 가능한 일련의 dream query 도입
$$
d = {d_q}_{q=1}^Q
$$
- t번째 토큰이 \<dream\> 토큰으로 예측되었다고 가정할 때, 조건부 embedding $\mathcal{C}_{K(t) + 1}^{DREAMLLM}$는 (K(t)+1) 번째 이미지 합성을 이전 sequence를 인과적으로 query하여 얻을 수 있음
$$
\mathcal{C}_{K(t) + 1}^{DREAMLLM} := \mathcal{F}_\theta(d, x_{<t+1}, V_{<K(t)+1})
\tag{3}
$$
- latent $z$를 이용한 denoising score matching은 식 2와 유사한 형태
$$
\mathcal{L}_{DM}^{DREAMLLM}(\theta, d, \zeta, \psi, z) := \mathbb{E}_{t~\mu(0,1), \epsilon~\mathcal{N}(0, I)} [||\epsilon_\xi(z_t;\mathcal{C}^{DREAMLLM}, t) - \epsilon||^2],
\tag{4}
$$
- $\xi$는 SD가 frozen되어 있어서 업데이트되지 않음
- 식 4는 text inversion의 일반화된 공식으로 볼 수 있음
    - 모든 condition embedding은 모델 탐색에 의해 학습 가능함
    - score distillation의 관점에서, condition과 pre-learned score function에 의해 정의된 KL divergence는 조건부 이미지 합성에서 학습된 확률 밀도를 증류하기 위해 동일하게 최소화됨
    $$
    \min_{\theta,d,\zeta,\psi}\;
    L^{\mathrm{DREAMLLM}}_{\mathrm{DM}}
    := \mathbb{E}_{t,\;C^{\mathrm{DREAMLLM}}}\;
    D_{\mathrm{KL}}\!\big( q(z_{t-1}\mid z_t, z_1, C^{\mathrm{DREAMLLM}})\;\|\; p_{\xi}(z_{t-1}\mid z_t)\big)
    \tag{5}
    $$

**Universal Multimodal Generative Modeling**
- 교차된 문서 sequence $x = {x_t}_{t=1}^T는 단어 $w = {w_i}_{i=1}^N$과 이미지 $I = {I_k}_{k=1}^K$를 모두 포함
- 자기 회귀적 성질은 이미지 conditional multimodal comprehension $p(w|I)$ 또는 text-to-image 합성 $p(I|w)$와 같은 모든 가능한 조건부 분포를 형성
- 이미지는 인과적 이해를 위해 visual embeddings $V$로 처리됨
- 사전 학습된 SD가 최적의 점수 함수라고 가정하면, 식 5는 합성 posterior에 대한 MLE 최적화로 볼 수 있음
- 식 1과 달리, 목표 sequence $x_t$는 인코딩된 이미지나 단어 모두가 될 수 있음
    - 목표는 임의 형태의 모든 인과적으로 조건된 posteriors에 대한 MLE로 통합됨
    $$
    \mathcal{L}_{MLLM}^{DREAMLLM}(\Theta = \{\theta, d, \zeta, \psi\}, x) := -\mathbb{E}_t [\log_{p \Theta}(x_t | x_{<t})]
    \tag{6}
    $$

### 3.2 Model Training
1. Alignment Training
    - multimodal 입력을 LLM에 적응시키는데 사용됨
    - 다음 항목들은 frozen LLMs, visual encoder, SD 간 cross-modal manifold 정렬을 사전 학습함
        - linear visual projector, linear condition projector, dream embedding
    - 약 30M의 image-text 조합 데이터를 사용하여 image-to-text 이해와 text-to-image 합성을 모두 훈련
2. $\mathcal{I}$-GPT Training
    - 이후, LLM은 $\mathcal{I}$-GPT 사전 학습을 위해 unfrozen 과정을 거침
        - joint vision-language 분포 학습을 촉진
        - training은 약 2개의 MMC4-Core에서 선별적으로 필터링된 문서를 포함
        - CLIP 점수 임계값: 0.25
        - text-to-image 학습을 강화하고 일부 저품질 노이즈 이미지와 텍스트의 영향을 완화
            - LAION400M에서 2개의 페어링된 데이터 샘플을 사용
            - BLIP 자막을 제공
3. Supervised Fine-tuning
    - 인간 지시에 따라 일반적인 multimodal 이해 및 창의적 삭업을 수행할 수 있게 함
    - Liu et al.이 수집한 약 80K의 visual instruction tuning data를 활용
    - instruction-following 콘텐츠 생성을 위해 GPT-4 문서 요약 또는 이미지 캡션으로 프롬프트를 받음
    - 약 20K개의 instruction following 문서 합성을 MMC4에서 수집
    - BLIP에서 20K의 이미지 합성 데이터 수집

## 4. Experiments

### 4.1 Multimodal Comprehension

![alt text](./images/Table%201.png)

- DreamLLM의 multimodal vision 및 언어 능력을 여러 벤치마크에서 평가
    - COCO, Image2Paragraph에서의 image-to-text captioning
    - VQAv2, OKVQA, VizWiz에서의 VQA
    - TextVQA에서의 text-related VQA
- MMBench, MM-Vet에서 zero-shot 평가 수행
- DreamLLM이 모든 벤치마크에서 다른 MLLM을 능가
    - DreamLLM-7B는 이미지 생성 능력을 갖는 동시대 MLLM보다 큰 차이로 우수
    - Emu-13B 대비 VQAv2에서 +16.6 높은 정확도 달성
- MMBench 및 MM-Vet과 같은 종합 벤치마크에서도 모든 7B 상대 모델을 상대로 SOTA 달성
- 다른 MLLM과 비교해 우수한 공간/관계 추론 능력을 보임
    - 이미지 생성 학습의 결과일 가능성이 높음

### 4.2 Text-conditional Image Synthesis

![alt text](./images/Table%202.png)
> **Table 2. MS-COCO와 LN-COCO에서의 Zero-shot text-to-image generation FID**
> - LM: Language Model based 방법
> - MG: Multimodal Generation 방법
> - FIG: Free-form Interleaved Generation 방법
> - $\dagger$: 우리 상태 $I$ 데이터에서 fine-tune된 SDv2.1
> - *: 검색 보강(retrieval-augmentation)
> - $\triangleright$: stage I alignment training 후의 결과

Text2Image
- 자유 형식 언어를 통해 창의적 콘텐츠 생성
- MS-COCO validation set, LN-COCO(Localized Narratives의 COCO subset)에서 text 조건부 이미지 합성 평가
- MS-COCO 데이터셋
    - 더 짧은 캡션을 갖는 고수준 이미지 추상화를 포함
    - 각 text prompt당 CLIP 점수 순위에 따라 8개 이미지를 샘플링
- LN-COCO
    - 더 포괄적인 이미지 설명 제공
    - CLIP 순위 없이 각 prompt당 한 개의 이미지만 샘플링
        - text가 길고 CLIP 길이 제한을 초과함
- 평가 지표: zero-shot Fréchet Inception Distance(FID)
- 세 가지 주요 관찰 사항
    1. DreamLLM은 stage-I 정렬 이후 SD baseline에 비해 FID가 크게 향상됨
        - MS-COCO와 LN-COCO에서 각각 3.67, 11.83만큼 점수 감소(SDv2.1과 비교)
        - pretraining과 supervised fine-tuning을 통해 FID가 각각 3.97, 13.73 향상됨
        - DreamLLM은 긴 문맥 정보 처리에 탁월한 능력을 지님
            - LN-COCO에서 큰 향상을 보임
    2. 이전 specialist 모델과 비교할 때, SD image decoder을 기반으로 경쟁력 있는 결과 제공
    3. DreamLLM은 동시대 MLLMs-based image 합성 방법을 지속적으로 능가

### 4.3 Multimodal Joint Createion & Comprehension

![alt text](./images/Fig%203.png)
> **Figure 3. 선택된 DreamLLM instruction 준수 interleaved content 생성 예시**
> - 각 이미지는 DreamLLM이 결정한 위치에서 자동으로 생성됨
> - 이후 해당 이미지는 다음 content 생성의 multimodal 이해 입력으로 피드백됨

**Free-form Interleaved Document Creation**

- $\mathcal{I}$-GPT의 interleaved generative 모델링을 활용하여 자유로윤 형태의 interleaved document 생성 가능(그림 3 참조)
    - 이 문서에서는 다음을 보임
        1. DreamLLM이 지침에 따라 의미있는 콘텐츠를 생성 가능
        2. 시스템은 제안된 token을 예측하여 지정된 위치에서 자율적으로 이미지를 생성할 수 있음
**Image Quality**
- 문서 품질은 text 내용, 이미지 품질(image-text 정렬 포함), 일러스트 위치 지정 등의 요인에 의해 영향을 받을 수 있음
- InstrcutMMC4에서 제시된 instruction-following subset을 시연 도구로 사용
    - 하위 집합은 30개의 MMC4-defined topics에 걸쳐 15000개의 문서로 구성됨
    - 각 주제당 500개의 샘플 존재
    - 부분집합에 대해 FID를 사용해 이미지 품질을 평가
    - 실제 택스트 기반으로 각 이미지 생성
    - 이미지 합성에 매칭된 text input만 사용할 경우, SD는 FID 점수 74.77을 달성
    - DreamLLM은 FID 점수 36.62로 SD보다 더 우수한 성과를 보임

**Human Evaluation**
- 생성된 샘플의 품질 평가를 위해 포괄적인 human evaluation 수행
- 150개 샘플(주제별 5개)을 무작위로 선정하여 instruction-following 문서 생성을 수행
- 생성된 문서와 실제 MMC4 문서를 식별 정보 없이 혼합하여 작성
- 다섯 명의 편향되지 않은 자원봉사자가 주어진 샘플이 지지되는지 여부를 판단
- MMC4에서 중복 및 저화질 이미지가 존재할 때, 지지율은 77.24%에 불과
- DreamLLM은 60.68%의 지지율을 달성하여 30%인 튜링 테스트 요구사항을 초과
- 생성된 문서들이 논리적으로 배치된 고품질 이미지를 포함하고 있음을 나타냄

## 5. Discussions

### 5.1 Creation과 이해 사이의 시너지

![alt text](./images/Table%203.png)
> **Table 3. multimodal 이해와 생성 간의 시너지에 대한 구체적인 분석**
> - ID는 사전 학습의 두 번째 단계에서 interleaved dataset이 사용되는지를 나타냄

- 동일한 학습 데이터를 사용하면서 학습 목표가 다른 세 가지 방법을 DreamLLM 아키텍처와 비교
    - 멀티모달 생성과 이해 간의 시너지를 설명하기 
    위해
- a. creation-only baseline: text/문서 조건부 이미지 합성에만 집중
- b. comprehension-only baseline: 단어 생성에만 집중
- c. Joint-learning: 이미지와 언어 모델링 모두 학습. DreamLLM의 기본 설정

**Quantitative Analysis**
Table 3에서 다음과 같은 관찰이 이뤄짐
1. LLM의 강력한 언어 이해 능력은 SD와 같은 text-image 전문가의 성능을 크게 향상시킴
    - 8.50 FID(1행)로 입증
2. MMC4와 같은 interleaved data의 사용은 multimodal 이해 성능을 잠재적으로 향상시킬 수 있음(4행)
3. 제안된 $\mathcal{I}$-GPT는 이해와 생성의 시너지를 더욱 강화하여 성능 향상으로 이어짐(5행)
4. CLIP 정렬 loss를 통합하는 경우, DreamLLM은 수렴하지 못하고 붕괴 지점에서 끝남(6행)
    - query가 실제 데이터분포를 적응적으로 학습하고 있음
    - CLIP semantics가 MLLM-encoded semantics와 충돌하고 있음

**Qualitative Analysis**

![alt text](./images/Fig%204.png)
> **Figure 4. Qualitative comparison**  
> - 답변 A: interleaved training 없이 comprehension-only 모델에서 나온 답변
> - 답변 B: joint-leaning model에서 나온 답변

comprehension-only와 joint learning 모듈 각각의 일부 예제 VQA 과제에 대한 답변 비교(Fig 4)
1. joint-learning 방법은 객체 크기와 같은 객체 관계 및 속성을 식별하는 데 있어 우수한 multimodal 이해를 보임
2. 여러 이미지 입력을 포함하는 multimodal 이해 시나리오에서 joint-learning 방법은 정확성 향상을 보임  
-> $\mathcal{I}$-GPT 사전학습의 자연스러운 결과. 다양한 interleaved 문서에서 multimodal 상관관계를 더 잘 모델링할 수 있게 함

**Multimodal In-Context Generation**

- In-context VQA는 상당한 진전이 있었지만, In-context 이미지 합성은 상대적으로 탐구가 부족
- in-context 이미지 편집, subject-driven image generation, compositional generation과 같은 작업은 zero-shot 환경에서는 상당한 어려움을 제시
    - DreamBooth에서와 같은 downstream finetuning이나 Prompt2Prompt에서의 attention 수정 기술 등이 없이는 어려움
- Fig 5: DreamLLM의 multimodal context 조건부 이미지 합성
    - DreamLLM이 주제, identity, semantic context를 유지하는데 있어 유망한 잠재력을 갖고 있음

![alt text](./images/Fig%205.png)
> **Figure 5. 선택된 DreamLLM in-context 이미지 생성 예시**  
> - 생성된 이미지 아래에 표시된 텍스트 프롬프트로 X의 multimodal 입력이 교체됨
> - (c)에서는 text prompt X만 사용한 SD ㅠㅁㄴ디ㅑㅜㄷdml rufrhkfmf qhdla


### 5.2 What is Learned by DreamLLM?

![alt text](./images/Fig%206.png)
> **Figure 6. dream queries와 Diffusion U-Net latent의 cross-attention**  
> - 64개의 query는 64개의 단어로 볼 수 있음
> - 각 attention map은 각각의 query와 U-Net의 latent feature 간의 cross-attention으로 계산됨
> - 64개의 querys느 8x8 grid 순서로 배열됨
> - 각 attention map은 모든 timestamp에 걸쳐 평균화된 결과

**Dream Query Attention**
- DreamLLM에서는 conditional embedding이 일부 학습된 꿈 쿼리와 함께 MLLM으로부터 유도됨
- Fig 6은 이러한 query와 diffusion latent 간의 학습된 cross-attention을 시각화
- (Hertz et al. 2023)과 유사하게, 모든 timestamp에 걸쳐 평균화된 attention map을 시각화
1. query attention은 구조화되어 있고, 분리되어 있음. 의미론적으로 지향적임
    - 서로 다른 query가 주제와 배경 의미를 능숙하게 포착한다는 사실에서 입증됨
2. 다양한 프롬프트에도 불구하고, attention pattern은 놀라운 유사성을 보임
    - 일반적인 text token에 종속적인 원본 SD의 token attention과는 대조적
- 모델의 인과적 특성 때문에 발생하여 일관된 의미 구조 순서를 초래한다고 가정