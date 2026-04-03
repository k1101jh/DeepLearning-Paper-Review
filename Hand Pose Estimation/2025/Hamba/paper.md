

url: https://proceedings.neurips.cc/paper_files/paper/2024/hash/03e9a69e5b686c316a07d73f0cf5e225-Abstract-Conference.html

## GPT 정리

Hamba - Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba
1. 개요
본 논문은 단일 RGB 이미지에서 3D 손 메쉬 복원을 수행하는 Hamba라는 새로운 모델을 제안함.
기존 SOTA(최신 기술) 모델들은 Transformer 기반의 Attention 메커니즘을 사용하여 3D 손 포즈와 형태를 학습하지만, 손 관절 간의 공간적 관계를 효과적으로 모델링하지 못함.
Hamba는 Graph-guided Bi-Scanning Mamba (GBS-Mamba) 프레임워크를 도입하여 그래프 학습과 상태 공간 모델(State Space Model, SSM) 을 결합하여 성능을 향상시킴.
2. 주요 기여
그래프 학습과 상태 공간 모델(SSM)을 결합하여 3D 손 메쉬 복원 모델 개발
기존 Transformer 기반 모델보다 88.5% 적은 토큰(token)을 사용하면서도 성능을 유지.
Graph-guided State Space (GSS) 블록 설계
손 관절 간의 구조적 관계를 학습하는 그래프 컨볼루션 네트워크(GCN) 와 Mamba 블록을 결합.
Token Sampler 및 Fusion 모듈 개발
효율적인 토큰 샘플링 및 전역-지역 특징 통합으로 성능 향상.
SOTA 모델 대비 성능 향상
FreiHAND 벤치마크에서 PA-MPVPE 5.3mm, F@15mm 0.992 기록.
HO3Dv2 및 HO3Dv3 데이터셋에서도 1위 성능 달성.
3. Hamba 모델 아키텍처
백본(Backbone) 모델
Vision Transformer (ViT) 기반 백본을 사용하여 이미지 특징을 추출.
Token Sampler (TS)
2D 손 관절 위치를 예측한 후, 이를 기반으로 중요한 토큰을 선택.
Graph-guided State Space (GSS) 블록
Graph-guided Bi-Scanning Mamba (GBS-Mamba) 기법을 적용하여 손 관절 간 공간적 관계를 학습.
기존 Transformer의 Attention을 대체하는 Mamba 기반 상태 공간 모델을 사용하여 긴 범위(long-range) 의존성을 효율적으로 학습.
Fusion 모듈
GSS 블록에서 생성된 특징을 전역 특징(Global Feature)과 결합하여 최종 출력 생성.
MANO 모델 활용
최종적으로 MANO 모델을 이용하여 3D 손 메쉬 복원 수행.
4. 실험 및 성능 비교
FreiHAND 데이터셋 성능 (3D 손 메쉬 복원)

PA-MPJPE (↓): 5.8mm
PA-MPVPE (↓): 5.3mm
F@5mm (↑): 0.806
F@15mm (↑): 0.992 (SOTA 모델 중 최고 성능)
HO3Dv2, HO3Dv3 데이터셋 성능

기존 HaMeR 모델보다 우수한 성능을 기록하며 Rank 1 기록.
실험 결과 요약

기존 Transformer 기반 모델 (METRO, MeshGraphormer 등)보다 토큰 수 88.5% 감소하면서도 성능 향상.
복잡한 손 동작 및 가려진 손(occlusion) 환경에서도 강건한 성능을 보임.
영상 데이터에 대한 일반화 성능 우수 (HInt-EpicKitchens, NewDays 벤치마크 테스트).
5. Ablation Study (성능 기여도 분석)
GSS 블록 제거 시 성능 하락
F@5mm 0.738 → 0.717, PA-MPJPE 6.6 → 6.9mm
GCN 제거 시 성능 하락
PA-MPJPE 6.6 → 7.3mm, PA-MPVPE 6.3 → 7.2mm
Mamba (SSM) 제거 시 성능 하락
PA-MPJPE 6.6 → 7.3mm, PA-MPVPE 6.3 → 7.2mm
6. 결론 및 한계점
결론

Hamba는 3D 손 메쉬 복원을 위한 새로운 그래프 기반 Mamba 모델로서, SOTA 대비 성능을 향상시키면서도 연산량을 줄임.
Graph-guided Bi-Scanning Mamba 기법을 통해 손 관절 간의 공간적 관계를 효과적으로 학습.
실험 결과, FreiHAND 및 HO3Dv2, HO3Dv3 데이터셋에서 SOTA를 달성.
한계점

현재는 비디오 기반 데이터에 대한 학습 및 평가가 부족, 향후 시계열 학습을 통한 개선이 필요.
모델이 학습한 데이터셋과 다소 차이가 있는 특정 환경(극한 조명, 비정상적 손 모양 등)에서는 성능 저하 가능성.
요약
Hamba는 Graph-guided Bi-Scanning Mamba 기법을 적용한 3D 손 메쉬 복원 모델로, 기존 Transformer 기반 모델보다 88.5% 적은 토큰을 사용하면서도 높은 정확도를 달성.
Graph-guided State Space (GSS) 블록을 설계하여, 손 관절 간의 공간적 관계를 효과적으로 학습.
FreiHAND, HO3Dv2, HO3Dv3 데이터셋에서 최고 성능 기록하며, 특히 손 가림(occlusion) 상황에서도 강건한 성능을 보임.
→ 기존 Transformer 기반 모델을 대체할 수 있는 새로운 접근 방식 제안.


## Abstract

단일 RGB 이미지에서 3D 손 재구성은 AR 티큘레이트 모션, 자체 폐색 및 물체와의 상호 작용으로 인해 어렵습니다. 기존 SOTA 방법은 어텐션 기반 트랜스포머를 사용하여 3D 손 포즈와 모양을 학습하지만, 주로 관절 간의 공간 관계를 비효율적으로 모델링하기 때문에 강력하고 정확한 성능을 완전히 달성하지 못합니다. 이 문제를 해결하기 위해 그래프 학습과 상태 공간 모델링을 연결하는 Hamba라는 새로운 그래프 유도 Mamba 프레임워크를 제안합니다. 우리의 핵심 아이디어는 몇 가지 효과적인 토큰을 사용하여 3D 재구성을 위해 Mamba의 스캐닝을 그래프 유도 양방향 스캐닝으로 재구성하는 것입니다. 이를 통해 재건 성능을 개선하기 위해 관절 간의 공간적 관계를 효율적으로 학습할 수 있습니다. 구체적으로 말하자면, 그래프로 구조화된 관계와 조인트의 공간 시퀀스를 학습하고 어텐션 기반 방법보다 88.5% 적은 토큰을 사용하는 GSS(Graph-guided State Space) 블록을 설계합니다. 또한 융합 모듈을 사용하여 상태 공간 기능과 전역 기능을 통합합니다. GSS 블록과 융합 모듈을 활용함으로써 Hamba는 그래프 유도 상태 공간 기능을 효과적으로 활용하고 글로벌 및 로컬 기능을 공동으로 고려하여 성능을 개선합니다. 여러 벤치 마크와 현장 테스트에 대한 실험은 함바가 FreiHAND에서 5.3mm의 PA-MPVPE와 0.992의 F@15mm을 달성하여 기존 SOTA를 크게 능가하는 성능을 보여주었습니다. 이 논문이 채택될 당시, Hamba는 3D 손 재구성에 관한 두 개의 대회 순위표1에서 1위를 차지하고 있습니다

## 1. Introduction

3D 손 재구성은 로봇 공학, 애니메이션, 인간-컴퓨터 상호 작용 및 AR/VR을 포함한 여러 분야에 걸쳐 수많은 응용 분야를 가지고 있습니다[11, 34, 71, 18, 103]. 그러나 신체 컨텍스트나 카메라 매개변수 없이 단일 RGB 이미지에서 3D 손을 재구성하는 것은 컴퓨터 비전에서 어려운 과제로 남아 있습니다. 최근 연구에서는 주로 이 작업을 위해 트랜스포머[14, 19, 51, 52, 70, 95, 73, 45, 55]를 탐색하고 어텐션 메커니즘을 활용하여 SOTA 성능을 달성했습니다. METRO[51]는 정점-정점 및 정점-접합 관계를 학습하기 위해 self-attention을 사용하는 다층 변압기를 도입했습니다. MeshGraphormer[52]는 재구성 성능을 더욱 향상시키기 위해 트랜스포머와 그래프 컨볼루션을 통합했습니다. 최근 HaMeR[70]은 더 나은 성능을 달성하기 위해 ViTPose[95] 가중치와 대규모 데이터 세트를 사용하여 ViT 기반 모델[19]을 설계했습니다.

## 2. 