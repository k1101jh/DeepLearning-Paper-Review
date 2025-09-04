# DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras



---

- SLAM

---

url:
- [paper](https://proceedings.neurips.cc/paper/2021/hash/89fcd07f20b6785b92134bd6c1d0fa42-Abstract.html) (NeurIPS 2021)

---
짧은 요약



요약



---

**문제점 & 한계점**


---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

DROID-SLAM
- 딥러닝 기반 SLAM
- dense bundle adjustment layer을 통해 카메라 자세와 pixelwise depth의 반복적인 업데이트로 구성
- 이전 작업보다 뛰어난 정확성
- catastrophic failuer이 현저히 적음
- monocular video로 훈련되었지만, Streo 또는 RGB-D 비디오 활용 가능


## 1. Introduction

SLAM(Simultaneous Localization and Mapping)
- 목표
    - 환경의 지도 작성
    - 환경 내에서 agent 위치 추정
- SfM의 특수한 형태
    - 장기 궤적을 정확하게 추적하는데 중점

SLAM 접근 방식
- 초기
    - 확률론적/필터링 기반 접근[12, 30]
    - 지도와 카메라 자세를 번갈아 최적화하는 방식[34, 16]
- 최근
    - Least-Squares(최소자승) 최적화 활용
    - Full Bundle Adjustment(BA)
        - 카메라 자세와 3D 지도를 하나의 최적화 문제로 동시에 최적화
    - 최적화 기반 구성 장점: 다양한 센서로 쉽게 확장 가능
        - 예: ORB-SLAM3[5]. 단안, 스테레오, RGB-D, IMU 지원
        - 다양한 카메라 모델 지원[5, 27, 43, 6]

현재 SLAM의 한계
- 강인성 부족
- 실패 원인
    - feature track 손실
    - 최적화 알고리즘 발산
    - drift 축적 등


딥러닝 기반 개선 시도
- 기존 연구 방향
    - 수작업 특징 -> 학습된 특징으로 대체
    - 신경망 기반 3D 표현 활용
    - 학습된 에너지 항을 고전적 최적화 백엔드와 결합
    - SLAM/VO 시스템을 end-to-end로 학습
    - 일부 방법들은 더 강인하지만, 공통 벤치마크에서 정확도는 여전히 고전적 방법보다 낮음


DROID SLAM
- 딥러닝 기반의 새로운 SLAM
- 장점
    - 높은 정확도
        - 다양한 데이터셋과 방법론에서 이전 방법보다 큰 성능 향상 달성
        - TartanAir SLAM 대회[55]에서
            - monocular track 62%
            - stereo track 60% 오류율 줄임
        - ETH-3D RGB-DSLAM에서 1위 달성
            - AUC(오류와 치명적 실패율을 모두 고려) 기준 2위보다 35% 우수함
        - EuRoC
            - monocular 입력 기준 zero failure 오류를 82% 줄임
            - ORB-SLAM3가 성공한 10개 sequence(전체는 11개)에서만 성공을 고려할 경우 43% 줄임
            - stereo 입력의 경우 ORB-SLAM3에 비해 오류를 71% 줄임
        - TUM-RGBD
            - 실패 없는 방법들 중에서 오류를 83% 줄임
    - 높은 강인성
        - ETH-3D에서 32개의 RGB-D 데이터셋 중 30개를 성공적으로 추적
        - 그 다음으로 좋은 방법은 19/32개만 성공적으로 추적
        - TartanAir, EuRoC, TUM-RGBD에서는 실패하지 않음
    - 강력한 일반화
        - monocular 입력으로만 훈련된 제안 시스템은 재훈련 없이 stereo 또는 RGB-D 입력을 직접 사용하여 정확성을 향상시킬 수 있음
        - 4개 데이터셋과 3개의 modalities에서 단일 모델로 모든 결과 달성
        - 학습은 monocular 입력으로만 합성된 TartanAir 데이터셋으로만 훈련

DROID(Differentiable Recurrent Optimization-Inspired Design)
- 고전적인 SLAM 방식과 딥러닝의 장점을 결합한 end-to-end 미분 가능 아키텍처
- 반복 최적화 구조
    - 카메라 자세와 깊이를 반복적으로 업데이트
    - RAFT는 두 프레임 간의 optical flow만 업데이트하지만 DROID는 임의의 수의 프레임에 대해 카메라 자세와 깊이를 반복적으로 업데이트
    - 긴 궤적과 loop closure에 대한 drift 최소화에 필수
- Differentiable Dense Bundle Adjustment(DBA) 레이어
    - 각 업데이트는 DBA 레이어를 통해 수행됨
        - optical flow와의 호환성을 최대화하기 위해 Gauss-Newton 방식으로 카메라 자세와 픽셀 단위 깊이를 조정
    - 이 레이어는 기하학적 제약 조건을 활용하여 정확성과 견고성을 높이며, 단안 카메라로 학습된 모델이 스테레오 또는 RGB-D 입력도 재학습 없이 처리 가능

기존 Deeplearning 기반 SLAM과 비교
- DeepV2D
    - 깊이와 카메라 자세를 번갈아 업데이트하여 전통적인 BA를 사용하지 않음
- BA-Net
    - BA 레이어를 포함하지만 깊이 기준(미리 예측된 깊이 맵 세트)를 선형적으로 결합하는 데 사용되는 적은 수의 coefficient만 최적화
    - photometric reprojection error을 최적화(feature space에서)
- 제안 방법
    - 깊이 기준에 얽매이지 않고 픽셀 당 깊이를 직접 최적화


## 2. Related Work


## 3. Approach

두 가지 목표
- 비디오를 입력으로 받아 카메라 경로 추적
- 환경의 3D 지도 구축

Representation
- 제안 네트워크는 정렬된 이미지 컬렉션($ \{ I_t \}_{t = 0}^N$)에서 동작
- 각 이미지 $t$에 대해 두 개의 상태 변수 유지
    - 카메라 포즈 $G_t \in SE(3)$
    - 역 깊이 집합 $d_t \in \mathbb{R}^{H \times W}_+$
- 논문에서 "깊이"를 말할 때는 "역 깊이"를 의미

프레임 그래프 $(\mathcal{V}, \mathcal{E})$
- 프레임 간의 공동 가시성을 나타내기 위해 채택
- edge $(i, j) \in \mathcal{E}$: 이미지 $I_i$와 $I_j$가 시야가 겹치며 공통된 3D 포인트를 공유함을 의미
- 훈련 및 추론 중에 동적으로 구축됨
- 각 포즈 또는 깊이 업데이트 이후 가시성을 다시 계산하여 그래프 업데이트
- 카메라가 이전에 매핑된 영역에 돌아가면 loop closure을 수행하기 위해 그래프에 장거리 연결을 추가

### 3.1 Feature Extraction and Correlation

시스템에 새 이미지가 추가될 때마타 feature 추출
- 핵심 구성 요소는 RAFT[48]에서 차용

Feature Extraction
- 각 입력 이미지는 feature extraction network를 거침
- 네트워크 구성
    - 6개 residual block
    - 3개 downsampling layer
- 출력: 입력 이미지 해상도의 1/8 크기의 dense feature map 생성
- RAFT[48]와 동일하게 두 개의 별도 네트워크 사용
    - Feature Network -> correlation volumes 생성에 사용
    - Context Network -> 업데이트 연산자 적용 시 context feature 주입

Correlation Pyramid
- frame graph $(i, j) \in \mathcal{E}$에 대해 $g_\theta(I_i), g_\theta(I_j)$의 모든 feature vector 쌍 간의 내적을 계산하여 4D correlation volume을 생성

$$
\displaystyle
\begin{aligned}

& C_{u_1 v_1 u_2 v_2}^{ij} = g_\theta(I_i)_{u_1 v_1} \cdot g_\theta(I_j)_{u_2 v_2}
& \tag{1}

\end{aligned}
$$

- 이후 상관관계 volume의 마지막 두 차원에 대해 average pooling을 수행하여 4-level correlation pyramid 생성
    - RAFT[48] 방식

Correlation Lookup
- lookup operator $L_r: \mathbb{R}^{H \times W \times H \times W} \times \mathbb{R}^{H \times W \times 2} \mapsto \mathbb{R}^{H \times W \times (r + 1)^2}$ 정의($r$: radius)
- 입력: $H \times W$ 크기의 coordinate grid
- Bilinear interpolation을 사용하여 correlation volume에서 값을 조회
- 이 연산자는 피라미드의 각 상관관계 volume에 적용됨
- 최종 특징 벡터: 각 level에서 얻은 결과를 concatenating해서 생성

### Update Operator

SLAM 시스템의 핵심 구성 요소는 learned update operator
- 구조: $3 \times 3$ convolutional GRU
    - 은닉 상태(hidden state) $h$를 유지
    - 매번 적용 시:
        1. 은닉 상태 업데이트
        2. 포즈 업데이트. $\Delta \xi^{(k)}$
        3. 깊이 업데이트. $\Delta \mathrm{d}^{(k)}$
- 포즈 업데이트는 SE(3) manifold 상에서 retraction 연산을 통해 적용
- 깊이 업데이트는 벡터 덧셈

$$
\displaystyle
\begin{aligned}

& G^{(k+1)} = \mathrm{Exp}(\xi^{(k)}) \circ G^{(k)}
& \mathrm{d}^{(k+1)} = \mathrm{d}^{(k)} + \delta \mathrm{d}^{(k)}
& \tag{2}

\end{aligned}
$$

- 업데이트 연산자를 반복 적용하면 포즈와 깊이의 sequence가 생성됨
- 목표: 반복이 고정점 $\{ G^{(k)} \} \rightarrow G^*, \{ \mathrm{d}^{(k)} \} \rightarrow \mathrm{d}^*$에 수렴 -> 실제 장면의 참값 재구성에 근접

Correspondence
- 각 반복의 시작 시, 현재 포즈, 깊이 추정을 사용해 프레임 간 correspondence 계산
- 프레임 $i$의 픽셀 좌표 grid $\mathrm{p} \in \mathbb{R}^{H \times W \times 2}$에 대해, dense correspondence field $\mathrm{p}_{ij}$  계산

$$
\displaystyle
\begin{aligned}

& \mathrm{p}_{ij} = \Pi \big( \mathrm{G}_{ij} \circ \Pi^{-1}(\mathrm{p}_i, \mathrm{d}_i) \big),
\quad
& \mathrm{p}_{ij} \in \mathbb{R}^{H \times W \times 2}
\quad
& \mathrm{G}_{ij} = \mathrm{G}_j \, \mathrm{G}_i^{-1}
& \tag{3}

\end{aligned}
$$

- frame graph의 각 edge $(i, j) \in \mathcal{E}$에 대해, 
    - $\Pi_c$: 3D point를 이미지와 매핑하는 camera model
    - $\Pi_c^{-1}$: inverse depth map $\mathrm{d}$와 coordinate grid $\mathrm{p}_i$를 3D point cloud (formulas와 jacobians는 appendix에서 제공)를 매핑하는 inverse projection function
    - $\mathrm{p}_{ij}$: pixel $\mathrm{p}_i$가 추정된 포즈와 깊이를 통해 프레임 $j$로 매핑된 좌표 