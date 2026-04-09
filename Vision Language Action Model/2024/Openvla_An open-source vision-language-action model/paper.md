# Openvla: An open-source vision-language-action model

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---


## 📌 Metadata
---
분류
- Vision Language Action Model

---
url:
- [paper](https://arxiv.org/abs/2406.09246) (arXiv 2024)
- [project](https://openvla.github.io/)

---
- **Authors**: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti,
Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam,
Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn,
- **Affiliation**: Stanford University, UC Berkeley, Toyota Research Institute, Google DeepMind, physical Intelligence, MIT
- **Venue**: arXiv 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Works](#2-related-works)
- [3. The OpenVLA Model](#3-the-openvla-model)

---


## Abstract

![alt text](./images/Fig%201.png)
> **Figure 1. **  
> 7B 매개변수 오픈소스 VLM인 OpenVLA 제안  
> Open X-Embodiment 데이터셋의 970k 로봇 에피소드로 훈련됨  
> 일반적인 로봇 조작 정책에서 SOTA 달성  
> 여러 로봇을 바로 제어할 수 있음  
> 파라미터 효율적인 fin-tuning을 통해 새로운 로봇 domain에 신속하게 적응할 수 있음  
> 체크포인트 및 학습 파이프라인을 오픈소스로 공개

- 인터넷 규모 vision-language 데이터와 다양한 로봇 시연을 조합하여 pretrained 대규모 정책은 로봇에게 새로운 기술을 가르치는 방식을 바꿀 잠재력을 가짐
- 새로운 행동을 처움부터 학습시키는 대신, Vision-Language-Action(VLA) 모델을 fine-tune하여 시각 운동 제어를 위한 견고하고 일반화 가능한 정책을 얻을 수 있음
- VLA를 Robotics에 적용하는 데 어려움
    1. 기존 VLA가 대부분 폐쇄적이로 public으로 접근 불가
    2. 기존 연구에서는 adoption(채택)의 핵심 요소인 새로운 작업에 VLA를 효율적으로 fine-tune하는 방법을 탐구하지 않음
- **OpenVLA**
    - 다양한 970k 건의 실제 로봇 시연 데이터를 기반으로 학습됨
    - 7B 매개변수 오픈소스
    - Llama2 언어 모델에 DINOv2와 SigLIP에서 사전 학습된 feature을 융합하는 visual encoder 결합
    - 추가된 데이터 다양성과 새로운 모델 구성 요소의 결과로 OpenVLA는 일반적인 조작에서 강력한 성능을 보임
    - RT-2-X(55B)와 같은 폐쇄형 모델을 29개 작업과 여러 로봇 구현에서 절대 작업 성공률 기준으로 16.5% 더 뛰어난 성능으로 능가
        - 매개변수수는 7배 적음
    - OpenVLA를 새로운 환경에 효과적으로 fine-tune 할 수 있음을 보임
        - 특히 여러 객체가 포함된 multi-task 환경에서 강한 일반화 결과를 보임
        - 강력한 언어 기반 능력을 갖춤
        - Diffusion Policy와 같은 from-scratch 모방 학습 방법보다 20.4% 더 높은 성능을 보임
    - 최신 low-rank adaptation 방법을 통해 일반 소비자용 GPU에서 fine tune 가능
    - 양자화를 통해 downstream 성공률에 영향을 주지 않으면서 효율적으로 서비스될 수 있음
    - Open X-Embodiment 데이터셋에서 대규모로 VLA를 학습할 수 있는 built-in 지원 제공

## 1. Introduction

- 로봇 조작을 위한 learned policies의 주요 약점:  
훈련 데이터 이상의 상황에 일반화할 수 없음
    - 개별 기술이나 언저 지침에 대해 훈련된 기존 정책은 물체 위치나 조명과 같은 새로운 초기 조건으로 행동을 외삽할 수 있는 능력을 가짐
    - 하지만 장면의 방해 요소, 새로운 물체에 대한 강인함이 부족하고, unseen 작업 지침 실행에 어려움을 겪음
- 기존의 vision & Language 기반 foundation 모델(CLIP, SigLIP, Llama2 등)은 이러한 유형의 일반화뿐만 아니라 그 이상의 능력을 가짐
    - 로봇 공학에서 이러한 규모의 사전 학습을 재현하는게 과제
    - 가장 큰 로봇 데이터셋조차 100k~1M 개의 예제만 갖고 있음
- 기존 방법
    - 로봇 표현 학습을 위해 다음을 탐구
        - pretrained language & vision-language model 통합[12-14]
        - 위 모델을 작업 계획 및 실행을 위한 모듈식 시스템의 구성 요소로 사용[15, 16]
    - 최근에는 제어를 위해 Vision-language-action 모델을 직접 학습하는데 사용
        - 로봇 공학을 위해 사전 학습된     vision & language foundation model을 사용하기도 함
        - PaLI[19, 20]은 Visually conditioned Language models(VLMs)를 직접 finetuning하여 로봇 제어 행동을 생성
        - 인터넷 규모 데이터로 학습된 강력한 foundation model을 기반으로 구축된 VLA
            - RT-2[7]과 같이 뛰어난 견고성 결과를 보여줌
            - 새로운 객체와 작업에 일반화될 수 있는 능력을 보여줌
            - 일반적인 로봇 정책에 대한 새로운 기준을 설정
        - VLA 사용을 방해하는 두 가지 이유
            1. 현재 모델[1, 7, 17, 18]은 폐쇄적임
                - 모델 아키텍처, 학습 절차, 데이터 혼합에 대해 잘 알려지지 않음
            2. 기존 연구는 VLA를 새로운 로봇, 환경, 작업에 배포하고 적응시키는 최선의 방법을 제공하지 않음
    - 일반적인 하드웨어(예: 소비자용 GPUs)에서, 향후 연구 개발을 위한 풍부한 기반을 구축하기 위해 기존 오픈소스 언어 모델 주변 생태계[21-24]와 유사하게 효과적인 fine-tuning 및 적응을 지원하는 오픈소스 범용 VLA가 필요함

**OpenVLA**
- 7B 매개변수 오픈소스 VLA
- 범용 로봇 조작 정책에 대해 새로운 SOTA 성능 수립
- 다양한 세부 수준에서 시각적 특징을 포착하는 pretrained visually-conditioned language model 백본으로 구성됨
- Open-X embodiment 데이터셋에서 가져온 97만개의 로봇 조작 경로의 크고 다양한 데이터셋으로 fine-tuning됨
    - 다양한 로봇 구현, 작업, 장면을 포함
- 이전 최고 성능 VLA였던 55B 매개변수 RT-2-X 모델[1, 7]보다 WidowX 및 Google Robot 구현에서 29개 평가 작업에 걸쳐 전체 성공률이 16.5% 높음
- VLA를 위한 효율적인 fine-tuning 전략 조사
    - 7가지 다양한 조작 작업에 걸쳐 수행됨
- fine-tuned OpenVLA 정책이 Octo[5]와 같은 pretrained 정책을 fine-tune한 경우보다 명확히 우수함
- Diffusion 정책[3]을 사용한 from-scratch 모방 학습과 비교
    - fine-tuned OpenVLA는 다중 객체가 있는 다중 작업 환경에서 언어를 행동으로 연결하려는 작업에서 상당한 향상을 보임
- 소비자용 GPU에서 OpenVLA 모델을 적응시키기 위해 low-rank adaptation[LoRA, 26] 및 모델 양자화[27]을 활용한 계산 효율적인 fine-tuning 방법의 효과를 입증
- 모든 모델, 배포 및 fine-tuning 노트북, 대규모 VLA 훈련을 위한 OpenVLA 코드베이스를 오픈소스로 공개

## 2. Related Work

## 3. The OpenVLA Model

![alt text](./images/Fig%202.png)
> **Figure 2. OpenVLA model 아키텍처**  
> 관찰 이미지와 언어 명령이 주어졌을 때, 모델은 7차원 로봇 제어 액션을 예측  
> 세 가지 주요 요소  
> 1. vision encoder(DinoV2 & SigLIP feature concat)  
> 2. projector(visual feature을 language embedding space로 mapping)  
> 3. LLM backbone(Llama 2 7B 매개변수 LLM)

### 3.1 Preliminaries: Vision-Language Models

- 최신 VLM 아키텍처의 주요 파트
    1. Visual Encoder:
        - 이미지 입력을 여러개의 patch embedding으로 매핑
    2. Projector:
        - Visual encoder의 출력 임베딩을 받아 language model의 입력 공간으로 매핑
    3. LLM 백본:
- VLM 학습 중 모델은 다양한 인터넷 소스에서 선별된 vision&language 데이터에 대해 쌍으로 이뤄지거나 교차된 데이터를 사용하여 end-to-end로 text token 예측 목표로 학습됨
- Prismatic-7B VLM
    - 본 논문에서 기반으로 사용한 모델
    - 주요 파트
        - 600M 매개변수 visual encoder
        - 작은 2-layer MLP projector
        - 7B 매개변수 Llama2 언어 모델 백본
    - Prismatic은 사전 학습된 SigLIP[79]와 DinoV2[25] 모델로 구성된 two-part visual encoder을 사용
        - 입력 이미지 patch는 두 encoder 모두를 통해 개별적으로 전달됨
        - 결과 feature vector은 channel-wise로 concat됨
    - CLIP[80] 또는 SigLIP 단일 encoder와 같은 더 일반적으로 사용되는 visual encoder와 달리, DinoV2 feature의 추가는 공간적 reasoning[44]에 도움이 됨
    - SigLIP, DinoV2, Llama2는 학습 데이터에 대한 세부 정보를 공개하지 않음
        - 각각 인터넷에서 수집된 image-text, image-only, text-only 데이터 토큰으로 구성될 가능성이 높음
        - Prismatic VLM은 이러한 구성 요소 위해 LLaVA 1.5 data mixture[43]을 사용하여 fine-tuning
            - 오픈 소스 데이터셋[29, 42, 81-83]에서 총 약 1M개의 image-text, text-only 데이터 샘플이 포함됨

### 3.2 OpenVLA Training Procedure

- 로봇 행동 예측을 위해 Pretrained Prismatic-7B VLM 백본을 fine-tune(Fig 2 참조)
- 로봇 행동 예측 문제를 입력 관찰 이미지와 자연어 작업 지침을 예측된 로봇 행동 문자열로 매핑하는 Vision-Language 과제로 공식화
- 로봇 행동을 언어 모델의 tokenizer가 사용하는 이산 토큰으로 매핑
- Brohan et al.[7]을 따라, 로봇 행동의 각 차원을 별도로 256개 구간 중 하나로 이산화
    - 각 행동 차원의 상위 1% 및 99% 사이 구간을 균등하게 나누도록 구간 폭을 설정
    - Brohan et al.이 사용한 min-bax bound 대신 이렇게 하는 경우, 데이터의 이상치 행동을 무시할 수 있음
        - 이상치 행동: 이산화 구간을 크게 확장하고 행동 이산화의 유효한 세분성을 줄일 수 있음
    - N차원 로봇 행동에 대해 N개의 이산 정수 $\in [0 ... 255]$를 얻음
    - Llama tokenizer[10]은 fine-tuning 중 새로 도입된 token을 위해 100개의 "special token"을 예약하고 있어 256개 행동 이산화 토큰에는 부족함  
    -> Llama tokenizer 어휘에서 가장 적게 사용되는 256개 token을 행동 토큰으로 덮어씀
- 행동이 token sequence로 처리되면, OpenVLA는 표준 next-token 예측 목적으로 학습됨
- 예측된 행동 토큰에 대해서만 cross-entropy loss를 평가

### 3.3 훈련 데이터
- Open X-Embodiment 데이터셋[1](OpenX)를 기반으로 학습 데이터셋 마련
- 전체 OpenX 데이터셋은 70개 이상의 개별 로봇 데이터셋과 200만개 이상의 로봇 궤적 포함
- 원시 데이터셋에 여러 단계의 데이터 큐레이션 적용
    - 큐레이션의 목표
        1. 모든 학습 데이터셋에서 일관된 입력 및 출력 공간 보장
        2. 최종 training mixture에서 구현, 작업, 장면의 balanced mix 확보
    - (1)을 위해, [1, 5]를 따르며 학습 데이터셋을 3인칭 카메라가 최소 하나 있는 조작 데이터셋으로 제한. single-arm end-effector 제어 사용
    - (2)를 위해, 첫 번째 필터링 단계를 통과한 모든 데이터셋에 대해 Octo[5]의 data mixture weights 활용
        - Octo는 경험적으로 다양성이 낮은 데이터셋의 weights를 낮추거나 제거
        - 작업과 장면의 다양성이 큰 데이터셋의 가중치를 높이는 방식으로 처리
- Octo 출시 이후 OpenX 데이터셋에 추가된 몇 가지 데이터셋을 training mixture에 보수적인 mixture weight 10%로 통합(예: DROID 데이터셋[11])
    - DROID의 행동 토큰 정확도가 학습 내내 낮게 유지됨을 발견
    - 향후 그 다양성을 맞추기 위해 더 큰 mixture weight나 모델이 필요할 수 있음
    - 학습 마지막 1/3 구간에서는 DROID를 data mixture에서 제외
        - 최종 모델의 품질을 위해

### 3.4 OpenVLA Design Decisions

- 최종 모델 학습 전 소규모 실험에서 다양한 설계 결정을 탐색
    - OpenX mixture에서 학습하는 대신 BridgeData V2[6]에서 OpenVLA 모델 학습 및 평가를 통해 iteration 속도를 높이고 계산 비용을 줄임

**VLM Backbone**
- IDEFICS-1[84], LLaVA[85] finetuning 테스트
    - 단일 객체만 있는 장면에서 둘의 성능이 비슷
    - 여러 객체가 포함되고 정책이 언어 지시에 명시된 올바른 객체를 조작해야 하는 과제에서는 LLaVA가 더 강력한 언어 기반 성능을 보임
        - BridgeData V2 sink 환경에서 5개의 언어 기반 과제에서 평균 절대 성공률 기준으로 IDEFICS-1 대비 35% 향상
    - fine-tuned Prismatic VLM 정책은 단일 객체 과제와 다중 객체 언어 기반 과제 모두에서 LLaVA 정책을 약 10% 절대 성공률로 능가
        - 통합된 SigLIP-DinoV2 백본이 제공하는 향상된 공간적 추론 능력 때문일 수 있음
    
**Image Resolution**
- $224 \times 224$px와 $384 \times 384$px 입력을 갖는 VLA를 비교했지만 평가에서 성능 차이를 발견하지 못함
    - $384 \times 384$는 훈련 시간이 3배 더 걸림
- 최종 OpenVLA 모델에서는 $224 \times 224$ 해상도 선택
- 많은 VLM 벤치마크에서는 해상도가 증가하면 성능이 향상되지만, VLAs에서는 이 경향을 아직 보지 못함

**Fine-Tuning Vision Encoder**
- 이전 연구에서, VLM 훈련 중 vision encoder을 freezing하면 일반적으로 더 높은 성능을 보임[44]
    - frozen vision encoder은 internet-scale pretrianing에서 학습된 강인한 특징을 더 잘 보존할 수 있음
- 본 논문에서는 VLA 훈련 중 vision encoder을 fine-tuning하는 것이 좋은 VLA 성능에 결정적임을 발견
    - pretrained vision backbone이 정확한 로봇 제어를 가능하게 하기 위해 장면의 중요한 부분에 대한 충분한 fine-grained 공간적 세부 정보를 포착하지 못할 수 있다고 가설을 세움

**Training Epochs**
- 일반적인 LLM 또는 VLM 훈련 과정:
    - 훈련 데이터셋을 1~2 epoch만 학습
- 본 논문에서는 VLA 훈련에서 훈련 데이터셋을 훨씬 많이 반복하는 것이 중요함을 발견
    - 실제 로봇 성능은 훈련 action token 정확도가 95%를 초과할 때까지 지속적으로 중가
    - 본 논문에서 최종 훈련 과정은 27 epoch를 거침

**Learning Rate**
- 고정 학습률 2e-5를 사용했을 때 가장 좋은 결과를 얻음
    - VLM 사전 학습동안 사용한 lr과 동일
- lr warmup이 이점을 제공하지 않음

### 3.5 Infrastructure for Training and Inference

- 최종 OpenVLA 모델은 64개의 A100GPU 클러스터에서 14일간 총 21,500 A100시간동안 batch size 2048로 학습됨
- 추론
    - bfloat16 정밀도로 로드될 때 15GB VRAM 요구
    - 하나의 RTX 4090 GPU에서 약 6hz로 실행됨
        - 컴파일, speculative decoding, 다른 추론 속도 향상 기법 없이 했을 때
    - 양자화를 통해 추론 중 메모리 사용량을 추가로 줄일 수 있음
    - 로봇으로 action 예측을 실시간 원격 스트리밍할 수 있도록 원격 VLA 추론 서버를 구현

## 4. The OpenVLA Codebase

- OpenVLA 코드베이스 공개
    - AMP
    - FlashAttention
    - fully sharded data parallelism
    - AutoModel 클래스와 통합
    - Lora fine-tuning
    - 양자화 모델 추론 지원

## 5. Experiments

### 5.1 Direct Evaluations on Multiple Robot Platforms

**Robot Setups and Tasks**

- 두 가지 로봇 구현에서 OpenVLA의 성능을 "out-of-the-box"(바로 사용 가능한 상태)로 평가
    - BridgeData V2 평가[6]에서의 WidowX 로봇
    - RT-1, RT-2 평가[2, 7]에서의 mobile manipulation 로봇(Google 로봇. Fig 1 중앙 참조)
- 각 환경에서 다양한 일반화 축을 포함하는 포괄적인 평가 작업 집합 정의
    - 예시:
        - Visual: unseen background, 방해 객체, 객체의 색상/외관
        - Motion: unseen object positions/orientations
        - Physical: unseen object sizes/shapes
        - Semantic: unseen target objects, instructions, 인터넷에서 가져온 개념
        - language conditioning: 장면 내 여러 객체가 있는 상태에서 능력을 평가.  
        정책이 사용자의 prompt에 명시된 올바른 객체를 조작할 수 있는지 테스트
- 평가 방법
    - BridgeData V2: 170 rollout(17개 작업, 각각 10회)
    - Google 로봇: 60 rollout(12개 작업, 각각 5회)

**Comparisons**

![alt text](./images/Fig%203.png)
> **Figure 3. BridgeData V2 WidowX 로봇 평가 작업 및 결과**  
> OpenVLA와 이전 SOTA 일반적인 로봇 정책을 여러 일반화 축을 포괄하는 포괄적인 작업 모음과 language conditioning 능력을 구체적으로 평가하는 과제에서 평가  
> OpenVLA는 semantic generalization을 제외한 모든 카테고리에서 closed-source model인 RT-2-X보다 뛰어나거나 전반적으로 압도적인 성능을 달성(표 4 참조)

![alt text](./images/Fig%204.png)
> **Figure 4. Google robot evaluation results**  
> RT-1 및 RT-2 평가[2, 7]에서 사용된 mobile manipulator에서 일반 로봇 정책을 in-distribution 및 out-of-distribution 작업에서 평가  
> OpenVLA와 RT-2-X가 유사한 성능을 달성  
> 전반적으로 RT-1-X와 Octo보다 훨씬 뛰어남  
> 평균 성공률 $\pm$ 표준편차는 접근 방식별로 총 60회 rollout에서 계산됨(표 6 참조)

- 세 가지 이전 generalist manipulation policies:
    - RT-1-X[1]
        - 35M 매개변수
    - RT-2-X[1]
        - 55B 매개변수
        - SOTA
        - Internet-pretrained vision & language 백본을 활용하는 closed-source VLA
    - Octo[5]
        - 93M 매개변수
        - 오픈소스 manipulation policies 중에서는 SOTA
- RT-1-X와 Octo는 OpenX dataset의 subset에서 처음부터 훈련된 transformer policies
- 실험 결과
    - BridgeData V2 평가(Fig 3 참조)
    - Google 로봇 평가(Fig 4 참조)
    - RT-1-X와 Octo는 tested tasks에서 고군분투함
        - 특히 방해물이 있을 때 종종 정확한 객체를 조종하는데 실패
        - 일부 경우에는 로봇이 팔을 목적 없이 휘두르게 함
- 이전 연구에서 수행된 평가보다 훨씬 더 큰 일반화 수준까지 테스트
    - internet pretraining이 없는 모델의 낮은 성능이 예상됨
    - RT-2-X는 RT-1-X와 Octo를 모두 능가
        - 대규모 pretrained VLM의 이점을 보임
    - OpenVLA
        - Google 로봇 평가에서 RT-2-X와 비슷한 성능을 보임
        - BridgeData V2 평가에서 RT-2-X보다 훨씬 뛰어난 성능을 보임(파라미터가 더 적음에도 불구. 7B vs 55B)
    - RT-2-X와 OpenVLA 모두 다른 모델보다 더 견고한 행동을 보임
        - 방해 객체가 있더라도 올바른 객체에 접근
        - 로봇의 end-effector을 목표 객체의 방향에 맞게 정확히 조정
        - 객체를 불안정하게 잡는 실수에서 회복
    - RT-2-X는 semantic generalization 작업에서 더 높은 성능을 보임(Fig 3 참조)
        - RT-2-X가 더 큰 규모의 인터넷 사전학습 데이터 사용
        - 로봇 행동 데이터와 인터넷 사전학습 데이터를 co-fine-tune하여 사전학습 지식을 더 잘 보존하도록 설계
        - OpenVLA는 로봇 데이터만으로 finetuning됨
    - OpenVLA는 BridgeData V2와 Google 로봇 평가의 다른 모든 과제 카테고리에서 동등하거나 더 나은 성능을 보임
        - 더 큰 학습 데이터셋(OpenVLA: 970k, RT-2-X: 350k)
        - 학습 데이터셋을 더 신중하게 정제
            - 예: bridge 데이터셋에서 모든 값이 0인 행동 필터링
            - pretrained semantic & spatial features를 결합한 fused vision encoder을 사용

### 5.2 Data-Efficient Adaptation to New Robot Setups

- 이전 연구에서는 주로 VLAs를 바로 사용 가능한 상태"로 평가하는데 초점
    - 새로운 작업 및 로봇 환경에 맞춘 효과적인 fine-tuning은 대부분 탐구되지 않음

![alt text](./images/Fig%205.png)
> **Figure 5. 새로운 robot setup에 적응**  
> 7개의 Franka Emika Panda 작업(각 10~150개 시연)에서 처음부터 학습된 SOTA Diffusion Policy, fine-tuned Octo, OpenVLA를 평가  
> Diffusion Policy: 단일 지시 작업에서는 강력한 성능을 보임  
> Octo, OpenVLA: 여러 지시 및 방해 객체가 포함된 다양한 fine-tuning 작업에서 더 나은 성능을 보임  
> OpenVLA는 두 setup모두에서 가장 높은 종합 성능 달성  
    - downstream task에서 정책을 학습하는데 효과적임  
> 평균 성공률 $\pm$ 표준편차는 접근 방식별로 129회 rollout(Franka-Tabletop 99회, Franka-DROID 30회) 기반으로 계산(표 7 참조)

**Robot setups and tasks**
- 간단한 fine-tuning 방법을 테스트
    - 목표 작업에 대한 10~150개의 시연이 포함된 소규모 데이터셋을 사용한 full fine-tuning
    - 두 가지 환경
        - Franka-Tabletop
            - 고정된 테이블에 장착된 Franka Emika Panda 7-DoF 로봇 팔
            - 5Hz non-blocking 컨트롤러 사용
        - Franka-DROID
            - 이동 가능한 책상에 장착된 DROID 데이터셋의 Franka 로봇 팔
            - 15Hz non-blocking 컨트롤러 사용
    - 각각Franka 로봇 팔을 fine-tuning 실험 대상으로 선택한 이유
        - 로봇 학습 커뮤니티에서 널리 사용됨
        - OpenVLA fine-tuning의 대상이 될 가능성이 높음

**비교**
- Diffusion Policy와 처음부터 학습한 경우 비교
- 입출력 사양을 OpenVLA에 맞춘 버전인 Diffusion Policy(matched)도 비교
- Octo[5]를 대상 데이터셋에 맞춰 fine-tuning한 결과도 평가
    - 현재 fine-tuining을 지원하는 최고의 범용 policy
- 동일한 대상 데이터셋에서 OpenVLA를 finetuning
    - OpenVLA로 표시
- Ablation study
    - OpenVLA(scratch)
        - OpenX-pretrained OpenVLA를 finetuning하지 않고, 기본 Prismatic VLM을 직접 대상 로봇 환경에 맞춰 finetuning

- 결과(Fig 5. 참고)
    - Diffusion Policy의 두 버전 모두 더 좁은 단일 지시 과제에서 Octo 및 OpenVLA와 경쟁하거나 능가
        - "당근을 그릇에 넣기"
        - "옥수수를 냄비에 붓기"
    - 여러 객체 포함 & 언어 조건이 필요한 다양한 fine-tuining 과제에서는 pre-trained 범용 정책이 더 나은 성능을 보임
    - Octo와 OpenVLA의 OpenX 사전 학습은 언어 기반 적응이 중요한 이런 더 다양한 과제에 모델이 더 잘 적응할 수 있게 함
        - OpenVLA(scatch)의 낮은 성능이 이를 보임
    - OpenVLA가 가장 높은 평균 성능을 달성
        - 대부분의 이전 연구들은 narrow single-instruction 또는 diverse multi-instruction 작업 중 한쪽에서만 강력한 성능 달성
        - OpenVLA는 모든 작업에서 최소 50%의 성공률을 달성
        - 좁지만 매우 정밀한 작업의 경우, Diffusion Policy는 더 부드럽고 정밀한 경로를 보임
        - Diffusion Policy에서 구현된 action chunking과 temporal smoothing을 도입하면 OpenVLA가 동일한 수준의 정밀성을 달성하는데 도움이 될 수 있음

### 5.3 Parameter-Efficient Fine-Tuning
- OpenVLA의 전체 fine-tuning은 작업당 5~15시간동안 8개의 A100 GPU 사용

![alt text](./images/Table%201.png)
> **Table 1. Parameter-efficient fine-tuning evaluation**  
> LoRA fine-tuning은 최고의 성능-계산 trade-off를 달성  
> 모델 파라미터의 1.4%만 학습하면서 full fine-tuning 성능과 동일한 성능을 보임  
> 평균 성공률 $\pm$ 표준편차는 선택된 Franka-Tabletop 작업에서 접근 방식별로 33회 시행한 결과 기반으로 계산(표 8. 참조)  
> *: FSDP[77]을 사용해 2개의 GPU에 분할됨

**fine-tuning 접근 방식 비교**
- full fine-tuning
    - fine-tuning동안 모든 가중치를 업데이트
- last layer only
    - OpenVLA의 transformer 백본의 마지막 레이어와 token embedding 행렬만을 미세 조정
- frozen vision
    - vision encoder을 freeze
- sandwich fine-tuning
    - vision encoder, token embedding matrix, last layer을 unfreeze
- LoRA
    - Hu et al.[26]에서 제안한 low-rank adaptation 기법 사용
        - rank 값 $r$을 곱하고 모델의 모든 linear layer에 적용

여러 Franka-Tabletop 작업에 걸친 fine-tuning 성공률과 학습 파라미터 수 및 GPU 메모리 요구사항을 보고(표 1 참조)
- 네트워크의 마지막 layer만 fine-tuning하거나 vision encoder을 freezing하면 성능이 저하됨
    - 대상 장면에 대한 visual feature의 추가 adaptation이 중요
- "sandwich fine-tuining"은 vision encoder을 fine-tuning하기 때문에 더 나은 성능 달성
    - 전체 LLM backbone을 fine-tuning하지 않기 때문에 GPU 메모리를 더 적게 소비
- LoRA는 성능과 학습 메모리 소비 간의 최상의 균형 달성
    - "sandwich fine-tuning"보다 우수한 성능을 보임
    - 전체 fine-tuning 성능과 일치하면서도 파라미터의 단 1.4%만을 fine-tuning
    - LoRA rank가 policy 성능에 거의 영향을 미치지 않음
        - 기본 rank $r=32$ 사용 권장
    - 단일 A100 GPU에서 10~15시간 내에 OpenVLA를 새로운 작업에 맞게 fine-tuning 가능
        - 전체 fine-tuning에 비해 8배의 연산량 감소

### 5.4 Memory-Efficient Inference via Quantization

![alt text](./images/Fig%206.png)
> **Figure 6. 다양한 GPU에서 OpenVLA 추론 속도**  
> bfloat16과 int4 양자화 모두높은 처리량 달성  
> 특히 Ada Lovelace 아키텍처(4090, H100)에서 두드러짐  
> TensorRT-LLM[89]와 같은 최신 LLM 추론 프레임워크를 이용하면 추가 속도 향상이 가능함  
> ♠: 모델이 두 개의 GPU에 걸쳐 나눠 적재됨

![alt text](./images/Table%202.png)
> **Table 2. 양자화 추론 성능**  
> 4-bit 양자화는 GPU 메모리 사용량을 절반 이상 줄이면서 bfloat16 추론(기본 접근 방식)과 성능이 일치함  
> 평균 성공 $\pm$ 표준편차는 8개의 대표적인 BridgeData V2 작업[6]과 접근 방식별 80회 롤아웃을 기준으로 계산됨(표 5 참조)

- OpenVLA는 7B 매개변수 모델
- 추론 시 OpenVLA를 bfloat16 정밀도로 저장하고 로드(기본 접근 방식)
    - 메모리 사용량을 절반으로 줄여 16GB VRAM 만으로도 OpenVLA 제공 가능
- 8개의 대표적인 BridgeData V2 과제에서 OpenVLA 모델을 8-bit 및 4-bit 정밀도로 제공하는 방법 조사
    - 8bit
        - 추가된 양자화 연산의 오버헤드때문에 대부분의 GPU에서 추론 속도를 늦춤
        - 상당한 성능 저하 발생
    - 4bit
        - GPU 메모리 전송 감소가 양자화 오버헤드를 상쇄하여 더 높은 처리량 달성
        - VRAM이 절반 이하임에도 불구하고 bfloat16 half-precision 추론과 유사한 성능 제공
        - A5000에서 3Hz로 실행 가능하여 데이터 수집 중 시스템 동역학과 더 밀접하게 일치
    - A5000 GPu에서는 모델을 1.2Hz로만 실행 가능
        - BridgeData V2 과제에서 사용된 5Hz non-blocking 컨트롤러의 훈련 데이터셋과 비교했을 때 시스템 동역학을 크게 변경

## 6. Discussion and Limitations

**OpenVLA**
- SOTA open-source VLA 모델
- 그대로 사용해도 다양한 로봇 형태간 제어에서 강력한 성능 발휘
- 파라미터 효율적인 fine-tuning 기법을 통해 새로운 로봇 설정에 쉽게 적응 가능
- 한계점
    1. 단일 이미지 관측만을 지원.
        - 현실에는 다양한 센서 입력이 존재
        - 여러 이미지와 고유감각 입력 및 관측 기록을 지원하도록 확장할 필요가 있음
        - 이미지와 text 데이터를 교차 학습한 VLM을 활용하면 flexible-input VLA fine-tuning이 용이해질 수 있음
    2. OpenVLA의 추론 처리량 향상이 필요
        - 고주파 제어 환경에서 VLA 제어 필요(ALOHA[90]는 50Hz로 동작)
        - 더 정교하고 양손 조작이 필요한 작업에서 VLA를 테스트할 수 있음
        - action chunking 또는 추론 시 최적화 기법(예: speculative decoding[91])  사용을 탐구하는 방법이 있을 수 있음
    3. 성능 향상의 여지가 있음
        - 테스트된 작업에서 아주 높은 신뢰성을 제공하지는 않음
        - 일반적으로 성공률이 <90%
    4. 계산 자원 제한으로 인해 많은 VLA 설계 관련해서 충분히 탐구되지 않음
        - 기본 VLM 크기가 VLA 성능에 어떤 영향을 미치는지?
        - 로봇 action 예측 데이터와 인터넷 규모 vision-language 데이터를 공동학습하면 VLA 성능이 크게 향상되는가?
        - VLA 모델에 가장 적합한 visual feature은?

