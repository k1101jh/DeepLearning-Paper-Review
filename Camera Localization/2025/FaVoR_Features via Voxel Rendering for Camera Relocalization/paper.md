# FaVoR: Features via Voxel Rendering for Camera Relocalization

---

- Camera Localization

---

url
 - [paper](https://ieeexplore.ieee.org/abstract/document/10943362)

---

목차

0. [Abstract](#abstract)
1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Conclusion](#5-conclusion)

---

## Abstract

Camera Relocalization 방법
 - dense image 정렬
 - query 이미지로부터 camera pose 회귀

sparse feature matching은 효율적이고 일반적인 경량 접근 방식
 - feature-based 방법은 종종 중요한 시점 및 외관 변화에 어려움을 겪어 매칭 실패 및 부정확한 자세 추정을 초래함  
 -> 2D feature의 global적으로 희소하지만 지역적으로 밀집한 3D 표현을 활용하는 새로운 접근 방식을 제안

여러 프레임에 걸쳐 landmark를 추적하고 삼각 측량하여 tracking 중에 관측된 image patch descriptors를 렌더링하는데 최적화된 sparse voxel map을 생성
 - 상세 방법
    - 초기 pose가 주어지면 volumetric rendering을 사용하여 voxel에서 descriptors를 합성
    - feature matching을 수행하여 카메라 자세 추정
 - unseen view에 대한 descriptor을 생성할 수 있게 하여 시점 변화에 대한 강인성을 높임

사용 데이터:
 - 7-Scenes
 - Cambridge Landmark datasets

실험 결과
 - 실내 환경에서 기존 SOTA feature representation 방식보다 크게 우수함
   - median translation error을 최대 39% 개선
 - 야외 장면에서도 다른 방법들과 비슷한 결과를 보이지만 계산과 메모리 사용량이 적음


## 1. Introduction

**Visual localization**
 - 넓은 viewpoint와 외관 변화로 인해 힘든 과제

**성능 개선 전략들**
 - sequence-based 장소 인식
   - 프레임 간의 일관성을 활용하여 이전에 방문했던 위치를 식별[27]
   - point cloud 정렬 및 3D-3D 매칭에서 파생된 기하학적 제약은 유용한 정보를 제공할 수 있음
   - 카메라 포즈를 결정하는데 광범위하게 사용되는 low-level feature matcing의 신뢰성과 견고성에 따라 달라짐
 - NeRF 기반 방법[26]
   - feature podint 추출을 위한 프레임을 합성하거나 photometric 정렬을 수행하여 dense/semi-dense SLAM 프레임워크를 기반으로 함
 - neural view 합성을 활용하여 known map에서 위치를 지정하거나 query viewpoint에서 관찰된 sparse한 feature point set의 외관 변화를 캡쳐하는 dense descriptyor space를 렌더링[14] 

**이러한 방법의 단점**
 - dense descriptor representation은 배포 전에 dense하고 완전히 최적화된 radiance field를 학습해야 하기 때문에 더 많은 학습 시간과 메모리를 요구
 - sparse descriptors를 합성하는 작업은 많은 채널의 설명자를 렌더링하는데 어려움을 겪음

**FaVoR**
 - 사전 학습된 신경망을 활용하여 feature을 추출하고 sparse voxel-based 공식화를 사용하여 3D 공간에서 feature descriptors를 인코딩하고 렌더링하는 새로운 feature rendering 방법
 - 이를 통한 scene representation은 전반적으로 희소하지만 지역적으로 밀집함
   - 어떤 query 카메라 위치에서도 view-conditioned 3D point descriptor을 효율적으로 추출할 수 있음
 - 훈련 중 일련의 프레임에서 keypoint을 추출하고 추적한 후, known 카메라 자세를 사용하여 3D 랜드마크를 삼각측량함
   - 기존의 online localization pipeline[30]과 유사함
 - 각 랜드마크는 voxel로 표현되며, volume rendering을 통해 최적화되어 관련된 descriptor가 새로운 view에서 렌더링 될 수 있음  
 -> 넓은 시점 변화 하에 low-level feature matching을 수행할 수 있음
 - dense radiance fields를 학습하는 이전 기술들과 달리, sparse landmark descriptors만 훈련하므로 descriptor 견고성과 자원 효율성 사이의 유리한 균형을 제공  
 -> 확장성 향상
 - descriptor을 렌더링하기 위해 신경망을 사용하는 대신 명시적인 voxel 표현과 trilinear interpolation을 사용하여 렌더링을 수행하여 훈련 및 렌더링 과정을 가속화
 - 7-Scenes(실내) 데이터셋에서 SOTA implicit feature rendering 방식을 크게 능가하여 median translation error을 최대 39% 줄임

**논문의 기여**
 - 1
 - 2


## 2. Related Work


## 3. Methodology

![alt text](./images/Fig%201.png)

> **Figure 1. 제안 방법의 도식적 표현**  
> 1. feature points를 추적하고 삼각 측량하여 지속적인 랜드마크(많은 view에서 관찰된 랜드마크)를 위한 voxel 표현을 생성  
> 2. voxel은 feature tracking 중에 추출된 descriptor patches를 렌더링하기 위하 최적화됨  
> 3. voxel을 쿼리하여 주어진 query pose에서 보이는 descriptor을 렌더링하고, query 이미지와 랜드마크간의 2D-3D 매칭을 찾고, 포즈 추정 수행

훈련 이미지들을 사용해서 sparse landmark 집합을 추적

**Render + PnP-RANSAC** 과정
 - volumetric rendering 사용
    - 각 랜드마크의 local 밀집 복셀 grid 표현을 최적화하기 위해
 - 추적하는동안 관찰된 2D descriptor patches를 렌더링(그림 1)
 - inference 단계
    - 초기 카메라 포즈 추정에 따라 최적화된 voxel 집합을 query하여 각 랜드마크의 외관을 결정하여 카메라 추정을 반복적으로 개선
 - 렌더링된 descriptor은 query 이미지의 특징과 매치됨
 - 이후 RANSAC을 적용하여 PnP 문제를 해결하여 카메라 포즈 계산

### 3.1 Landmark Tracking

 - M 개의 RGB 이미지($H \times W$ \times 3) Sequence를 고려
 - 랜드마크 $\ell_i \in \mathbb{R}^3$ 은 월드 frame에서 정의됨
 - $S_j$ = $\ell_j$를 포함하는 훈련 이미지의 index 집합($\ell_j$를 관찰하는 이미지들)일 떄, 각 이미지 $I_i$에 대해, $i \in S_j$인 경우, keypoint $k_{ij} \in \mathbb{R}^2$($\ell_j$가 $I_i$에 투영된)가 존재
 - $I_i$에서 월드 frame 카메라 포즈는 $T_i \in \text{SE}(3)$
 - $F$ = 주어진 이미지 $I_i$에 대해 keypoints와 dense descriptor map $D_i \in \mathbb{R}^{H \times W \times C}$를 제공하는 feature extractor.
 - $C$ = descriptor channel 개수
 - $D_i$로부터 각 추출된 keypoint $k_{ij}를 중심으로 $S \times S$ 픽셀 크기의 패치 $P_{ij} \in \mathbb{R}^{S \times S \times C}$를 잘라냄

$$
\displaystyle
\begin{aligned}
& P_{ij} = crop(D_i, k_{ij}, S)
& (1)
\end{aligned}
$$

 - 각 랜드마크 $\ell_j$에 대해, $\ell_j$를 포함하는 이미지 sequence의 카메라 포즈 $T_i$, keypoint 집합 $k_{ij}$, 해당하는 descriptor patch $P_{ij}$를 포함하는 track을 저장
 - track이 주어지면, world frame에서 랜드마크 위치 $\ell_j$를 삼각 측량
 - $\ell_j$의 초기 추정치 = 선형 tranform 알고리즘[15]를 사용하여 찾음. 
 이후 Levenberg-Marquardt 최적화 기법을 통해 reprojection 오류를 최소화하여 보정
 - 이상치를 감안하여, 최적화 과정에서 robust cost를 적용[13]
 - 최적화의 수치적 안정성을 보장하기 위해, 카메라 프레임의 랜드마크 위치의 inverse depth 매개변수화를 사용
 - 모델 & 비용 함수 - supplementary material 참고
 - 이 접근 방식은 Vision-inertial odometry(VIO)와 같은 기존 online vision-based localization system에 적합

### 3.2 Voxel Creation

 - 각 track이 최소 길이보다 긴 경우, $\ell_j$를 중심으로 하는 새로운 voxel $V_j$를 생성  
 -> 지속적인 랜드마크 선택을 위한 위치 추적 시스템과 유사함[30]
 - track과 연관된 랜드마크는 유용한 위치 정보를 제공하기 위해 여러 pose에서 볼 수 있어야 함
 - 각 voxel $V_j$는 $R \times R \times R$ 해상도를 갖는 더 작은 하위 voxel로 구성된 grid를 포함
 - grid의 각 node(정점)은 $F$에 의해 제공되는 descriptor channel에 해당하는 $C$ 크기의 벡터를 저장
 - 적절한 voxel(및 그 하위 voxel)의 크기를 결정하는 방법
    1. 각 patch $P_{ij}$에 대해 $\text{T}_i$에서 점 $\ell_j$까지의 유클리드 거리 $l_{ij}$를 계산
    2. patch 크기가 $S \times S$ 픽셀일 때, voxel 크기 $s_{v_j}$(m 단위)를 다음과 같이 추정

$$
\displaystyle
\begin{aligned}
& s_{v_j} = min_{i \in S_j} (S \cdot \frac{l_{ij}}{f})
& (2)
\end{aligned}
$$

> $f$: 카메라 focal length

 - keypoint $k_{ij}$와 연관된 descriptor을 렌더링하는데 필요한 정보를 캡처하기에 충분하므로 최소 voxel 크기를 선택
 - 각 voxel $V_j$는 descriptors grid와 동일한 해상도를 갖는 density grid와 연결되지만, $C$ 대신 크기 1의 node를 사용. density grid는 volume rendering 과정에 사용됨

> Density grid:  
> - 희소성 마스크 역할  
> - 여기서는 voxel의 각 정점에 descriptor가 저장되므로, voxel의 어느 위치에 descriptor가 저장되었는지 확인하는 역할

### 3.3 Descriptor Learning and Rendering

 - voxel이 생성되면, 관련된 track을 따라서 관찰된 descriptor patches를 렌더링하기 위해 시스템을 훈련할 수 있음
 - 훈련 과정은 [26]에서 설명된 방법과 유사하지만 view-dependent rendering을 위한 MLP의 부재와 같은 차이점이 있음
 - 모든 patches와 poses $i \in S_j$에 대해, patch $P_{ij}$의 각 요소를 통해 지나가는 카메라 중심 $T_i$에서 광선을 추적
 - 각 ray $r$은 voxel grid $V_j$ 및 관련된 density grid와 두 개의 point에서 교차함
 - 카메라에 가까운 지점: $p_n$, 다른 지점: $p_f$
 - 두 교차점 사이의 광선에서 $N$개의 샘플, $d_t \in \mathbb{R}^C$와 $\hat{\sigma}_t \in \mathbb{R}$ $(t = 1, ..., N)$을 샘플링하고, $V_j$와 density grid에서 각각 trilinear interpolation을 사용(3차원 공간이므로 3차원 보간 사용)
 - 이 과정은 [25]에서 제안된 volume rendering 방식을 따르지만, RGB 색상을 렌더링하는 대신 feature descriptor을 렌더링

$$
\displaystyle
\begin{aligned}
& \delta = \frac{||p_f - p_n||_2}{N}
& (3) \\
& \hat{d}_{ij}^{uv} = \sum_{t=1}^N T_i (1 - exp(-\hat{\sigma}_t \delta)) d_t
& (4) \\
& T_t = \prod_{l=1}^{t-1} exp(-\hat{\sigma}\delta)
& (5)
\end{aligned}
$$

> $\hat{\text{d}}_{ij}^{uv}$: patch $P_{ij}$의 pixel 위치 $(u, v)$에서 포즈 $T_i$로 본 랜드마크 $l_j$에 대한 추정된 descriptor

 - descriptor와 밀도 grid를 학습하기 위해 descriptor 벡터 공간에서 $\hat{\text{d}}_{ij}^{uv}$가 GT descriptors $d_{ij}^{uv} \in \text{P}_{ij}$에 가능한 가까운 norm과 방향을 갖도록 보장하고자 함
 - GT descriptors $d_{ij}^{uv} \in \text{P}_{ij}$는 patch에서 모든 $(u, v) \in {(0, 0), (0, 1), ..., (S, S)}$에 대해 $F$에 의해 추출됨
 - MSE와 cosine similarity loss를 사용하여 렌더링된 descriptor의 norm과 방향이 target feature에 가능한 가깝도록 함
 - patch $P_{ij}$의 smooth representation을 보장하기 위해 전체 variation regularization[34]를 추가
 - 밀도의 경우, [42]와 동일한 entropy loss를 사용. loss에 대한 자세한 설명은 supplementary material 참고
 - 다른 voxel과 독립적으로 각 voxel에 학습 프로세스를 적용하여 학습 프로세스를 병렬화

새로운 view에서 관찰된 장면의 voxel로부터 descriptor을 렌더링
 - 추정하고자 하는 query pose $T_q$에 대해 초기 추정 $\hat{T}_q$이  필요
 - 이 추정이 제공된다고 가정(robotics localization 시스템에서는 흔함)
 - 포즈 $\hat{T_q}$ 와 장면 내 voxel 집합 $V = {V_0, V_1, ..., V_J}$가 주어지면, 주어진 query pose에서 볼 수 있는 모든 landmark의 descriptor을 렌더링 할 수 있음 (이 렌더링에는 깊이 정보의 부족으로 인해 가려진 point가 포함될 수 있음)
 - 각 $V_j$에 대해 쿼리 카메라 포즈 $\hat{T_q}$에서 voxel grid 중심 $l_j$(랜드마크의 위치)로 광선을 추적
 - 이후 방정식 (3, 4, 5)으로 광선을 따라 volumetric rendering을 수행하여 $\hat{T}_q$에서 보이는 예상 descriptor을 얻음

### 3.4 2D-3D Matching and Pose Estimation

모든 $\hat{T}_q$에서 보이는 descriptor가 렌더링되면, query 이미지 $I_q$와 함꼐 2D-3D 대응을 찾을 수 있음
 - feature extractor은 일반적으로 query 이미지에서 희소 2D keypoint를 찾아 각 keypoint를 $F$[11, 49]에 의해 제공된 dense descriptor map과 연결
 - 렌더링된 descriptor을 query 이미지에서 추출한 descriptor과 일치시키기 위해, 임계값 이상의 가장 높은 유사성 점수를 갖는 대응을 찾음.  
 => 두 descriptor 세트 간의 유사성 행렬을 계산하고 임계값 처리를 한 후, 최대 유사성 응답을 갖는 descriptor 쌍만 고려
 - 그림 2는 Alike-l[50] feature extractor가 unseen view에서 추출한 dense descriptor map과 렌더링된 descriptor의 유사성 응답을 보임

모든 rendering된 feature가 일치하고 가정된 2D-3D 대응이 가능해지면 PnP RANSAC을 사용해서 카메라 포즈를 결정
 - feature 표현은 새로운 view로부터 descriptor을 렌더링 할 수 있게 하여 반복적인 Render + PnP-RANSAC 정제 절차를 사용할 수 있게 함
 - 추정된 query pose $\hat{T}_q$가 $T_q$에 수렴함에 따라 렌더링된 descriptor은 query 이미지 descriptor의 외관과 점점 더 일치하게 되어 더 많은 대응을 나타냄(4. Results 참조)


## 4. Results

7-Scenes[40]와 Cambridge Landmarks[17] 데이터셋을 사용하여 실내 및 실외 환경에서 카메라 relocalization 시스템을 평가

### 4.1 Implementation Details and Baselines

 - Pytorch 사용
 - [42] 프레임워크 기반
 - Alike[50] 및 SuperPoint[11]을 특징 추출기로 사용
 - 64(Alike-t), 94(Alike-s), 128(Alike-n), 128(Alike-l), 256(SuperPoint) 채널로 descriptor 테스트
 - 훈련
    - epoch 당 1024개의 ray를 사용
    - voxel(landmark) 당 2000 epoch 수행
    - Adam optimizer 사용
    - subvoxel마다 가변 learning rate 사용[42]
 - 모델 크기, rendering 품질, signal-to-noise ratio(PSNR), grid 해상도 간의 trade-off를 조사
 - 메모리 사용과 렌더링 성능의 균형을 맞추기 위해 $3 \times 3 \times 3$ grid 해상도 선택
 - $7 \times 7$ pixel descriptor patch에서 voxel을 훈련하는 데 NVIDIA Geforce RTX 4060 노트북 GPU에서 10초 소요
 - 단일 descriptor을 렌더링하는 데 1ms 소요
 - FaVoR은 빠른 런타임 작업을 위한 최적화가 되어있지 않음
 - Voxel 간의 독립성 덕분에 전체 훈련 및 추론 파이프라인을 병렬화하여 성능을 향상시킬 수 있음

FaVoR과 FQN[14], CROSS-FIRE[28], NeRF-loc[21]을 비교
 - unseen view에서 descriptor을 렌더링하는 유사한 SFR 방법들
 - FQN, CROSS-FIRE은 각각 SfM과 dense feature matching을 사용하여 대량의 point set에 의존
 - NeRF-loc은 각 query 이미지와 1024개의 3D point를 매칭
 - FaVoR
    - 더 작은 point set을 사용
    - CROSS-FIRE, NeRF-loc은 완전히 훈련된 신경 렌더링 모델을 요구하는데 비해, FaVoR은 랜드마크의 sparse voxel 표현을 훈련하기 위해 일련의 프레임만을 필요로 함
    - FaVoR은 CROSS_FIRE과 같은 custom 장면 의존 네트워크를 요구하지 않고 out-of-the-box(즉시 사용 가능한) feature extractor과 함께 작동
    - FaVoR을 훈련함으로써, 제안한 방법이 FQN과 달리 서로 다른 descriptor 크기로 확장 가능함을 보임
    - FQN이 Render + PnP-RANSAC을 여러 번 반복하는 반면, FaVoR은 CROSS-FIRE과 유사하게 세 번의 반복만 사용
- 포괄성을 위해, 이미지 기반 및 구조 기반 카메라 relocalization 방법과 hybrid 방법의 localization 결과를 보고함 (Section 2 참고)
- HM 카테고리의 경우, SfM point cloud를 사용하여 relocalization을 수행하는 SuperPoint(SP)[11] 및 SuperGlue(SG) 기반의 HLoc[35]로부터의 결과를 보고
- SceneSqueezer[48] 결과도 포함 - 더 적은 메모리를 요구하지만 더 큰 localization error을 보임

### 4.2 Relocalization Evaluation


