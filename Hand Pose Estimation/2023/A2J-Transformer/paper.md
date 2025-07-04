# A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image


---

---


---


## GPT 요약

1. 연구 배경

    - **3D 상호작용 손 포즈 추정(Interacting Hand Pose Estimation, IHPE)**은 VR, AR, HCI(Human-Computer Interaction) 등의 다양한 분야에서 활용 가능하지만, 다음과 같은 난점이 있음:

        - 심각한 자기 가림(self-occlusion) 및 상호 가림(inter-occlusion) 문제 발생.
        - 두 손의 유사한 외형 패턴으로 인해 혼동 발생.
        - 2D에서 3D로의 변환이 비정형적인 문제(ill-posed problem).

    - 기존 방법들은 모델 기반(model-based)과 모델 프리(model-free) 방식으로 나뉨.

        - 모델 기반 방식: 사전 정의된 손 모델을 활용하지만, 초기화 민감도 및 개인별 모델 조정 문제가 있음.
        - 모델 프리 방식: CNN 또는 Transformer를 활용해 3D 포즈를 직접 추정하지만, 손가락 간의 글로벌 관계 학습이 부족함.

2. A2J-Transformer의 핵심 아이디어
    - 기존 A2J(Anchor-to-Joint) 모델을 RGB 도메인으로 확장하여 3D 상호작용 손 포즈를 예측.
    - A2J-Transformer는 Transformer 기반의 비지역적(non-local) 인코딩-디코딩 프레임워크를 활용하여 로컬(local) 및 글로벌(global) 관계를 동시에 학습함.
    - 기존 A2J 대비 세 가지 주요 개선점:
        1. Anchor 간의 Self-attention을 도입하여 손 관절의 전역 관계 학습 → 가림(occlusion) 문제 해결.
        2. Anchor를 학습 가능한 Query로 변환하여 손 모양의 세밀한 표현 개선.
        3. Anchor를 2D에서 3D 공간으로 확장하여 2D-3D 변환 문제 해결.

3. A2J-Transformer의 아키텍처
    1. Pyramid Feature Extractor
        - ResNet-50 기반 피라미드 특징 추출기 사용하여 다중 해상도(multiscale) 특징 맵 생성.
    2. Anchor Refinement Model
        - Feature Enhancement Module: Self-Attention을 적용하여 로컬-글로벌 특징을 결합.
        - Anchor Interaction Module: Cross-Attention을 활용하여 Anchor 간 관계 학습.
    3. Anchor Offset-Weight Estimation Model
        - 각 Anchor가 손 관절에 대한 오프셋(offset)과 가중치(weight)를 학습.
        - 최종적으로 각 손 관절의 위치를 여러 Anchor의 가중합(weighted sum)으로 결정.
4. 실험 및 성능 평가
    - InterHand 2.6M 데이터셋 성능 비교
        - 기존 모델 대비 3.38mm MPJPE(MPJPE 감소) 개선.
        - 기존 A2J 대비 5mm 이상 성능 향상.
        - 기존 모델 대비 속도(FPS) 및 모델 크기(42M) 최적화.
    - HANDS2017, NYU 데이터셋에서도 SOTA 성능 기록
        - Depth 기반 데이터에서도 높은 성능을 유지하며 일반화 가능성 증명.

| 모델 | MPJPE (mm) (낮을수록 좋음)	| FPS(높을수록 좋음) | 모델 크기(M) |
|---|---|---|---|
| Zhang et al. | 13.48 | 17.02 | 143M |
| Meng et al. | 10.97 | 15.47 | 55M |
| Moon et al. | 14.22 | 107.08 | 47M |
| A2J-Transformer | 9.63 | 25.65 | 42M |

5. Ablation Study (성능 기여도 분석)
    - Transformer 제거 시 성능 하락: MPJPE 9.63 → 14.44mm.
    - 3D Anchor가 아닌 2D Anchor 사용 시 성능 저하: MPJPE 9.63 → 15.36mm.
    - Multi-scale Deformable Attention(MSDAM) 제거 시 성능 저하: MPJPE 9.63 → 10.69mm.

6. 결론 및 향후 연구 방향
    - A2J-Transformer는 기존 A2J 모델을 Transformer 기반으로 확장하여, 3D 상호작용 손 포즈 추정에서 최고의 성능을 달성.
    - 기존 2D Anchor 기반 모델을 3D Anchor로 확장하여 2D-3D 변환 문제 해결.
    - Self-Attention을 활용하여 로컬 및 글로벌 특징을 동시에 학습.
    - 향후 연구 방향:
        - 비디오 기반 손 추적 적용.
        - 추가적인 데이터 증강 및 실환경 적용.


## Abstract