# Loopy-SLAM: Dense Neural SLAM with Loop Closures

---

## 📌 Metadata
---
분류
- Dense SLAM
- Neural SLAM
- Loop Closure

---
url:
- [paper](https://arxiv.org/abs/2402.09944) (arXiv 2024)
- [project](https://notchla.github.io/Loopy-SLAM/)
---
- **Authors**: Lorenzo Liso, Erik Sandström, Vladimir Yugay, Luc Van Gool, Martin R. Oswald
- **Venue**: CVPR 2024

---

## 📑 Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [3. Method](#3-method)
  - [3.1 Neural Point Cloud-based SLAM](#31-neural-point-cloud-based-slam)
  - [3.2 Loop Closure and Refinement](#32-loop-closure-and-refinement)

---

## ⚡ 요약 (Summary)
- **Problem**: 신경망 기반 SLAM(Neural SLAM)은 미세한 재구성에 유리하지만, 대규모 환경에서 루프 폐쇄(Loop Closure) 기능이 부재하여 주행 거리가 길어질수록 궤적 드리프트와 지도 왜곡이 심화되는 문제가 있음.
- **Idea**: 장면을 여러 개의 독립적인 로컬 서브맵(Submap)으로 분할하여 관리하고, 전역 장소 인식을 통해 루프를 실시간 감지하여 서브맵 간의 자세를 보정하는 포즈 그래프 최적화(PGO)를 Neural SLAM에 통합함.
- **Result**: 포인트 기반 표현의 유연성을 극대화하여 비용이 큰 재학습 없이도 드리프트를 효과적으로 보완하였으며, 공개 데이터셋(Replica, ScanNet)에서 기존 Neural SLAM 대비 높은 궤적 정확도와 일관된 전역 지도를 구현함.

---

## 📖 Paper Review

## Abstract

신경망 기반 RGBD-SLAM
- dense SLAM에서 유망함을 보임
- 카메라 추적 중 오류 누적으로 인한 지도 왜곡과 같은 문제에 직면

Loopy-SLAM
- 포즈와 고밀도 3D 모델을 전역적으로 최적화
- data-driven point-based submap 생성 방식을 사용하여 frame-to-model tracking 수행
- global place recognition을 통해 online으로 loop closure 활성화
- 강건한 pose graph optimization을 사용하여 local submap을 rigidly 정렬
- 표현이 point 기반이므로, 맵 correction은 매핑에 사용된 전체 입력 프레임을 저장할 필요 없이 효율적으로 수행될 수 있음
    - 일반적으로 grid 기반 mapping 구조를 사용하는 기존 기법에서 요구되는 사항
- 합성 Replica 및 TUM-RGBD와 ScanNet 데이터셋 평가 결과, 기존의 dense neural RGBD SLAM 방법과 비교할 때 추적, 매핑 및 렌더링 정확도에서 경쟁력있거나 더 우수한 성능을 보임


## 1. Introduction

RGBD 카메라를 이용한 scene의 online dense 3D reconstruction
- 최근 몇몇 연구에서는 test time에서 encoder-free neural scene representation을 최적화하는 방법을 제안
    - 압축 성능 향상
    - unseen geometry 외삽(extrapolate)
    - 3D 의미 예측과 같은 고수준 추론으로의 보다 원활한 전환점 제공
    - 강력한 학습 가능한 prior을 활용
    - 온라인 최적화를 통해 test time의 제약에 적응할 가능성을 보임
- 두 가지 방식 존재
    - 결합형 솔루션
        - 추적과 매핑에서 동일한 표현 사용
    - 분리형 솔루션
        - 각각의 작업에 독립적인 프레임워크 사용
- 현재는 분리형 방법이 더 높은 추적 정확도를 달성
    - 하지만 분리형은 tracking이 고밀도 맵과 독립적으로 수행되므로 바람직하지 않은 데이터 중복성과 독립성 초래
- 궁극적으로는 동일한 장면 표현을 활용해야 한다고 판단
- 결합형 솔루션 중 MIPS-Fusion [55]를 제외한 모든 방법은 frame-to-model tracking만을 구현
    - noise가 있는 실제 데이터에서 카메라 드리프트가 심각하게 발생  
    -> 손상된 지도 생성
- 분리형 방법들은 모두 multi-resolution hash grid를 사용
    - map correction(loop closure 등)을 위해 쉽게 변환 불가
    - 비용이 큰 gradient 기반 update를 수행하고 mapping에 사용된 입력 프레임을 저장해야 함
- Point-SLAM[43]
    - 신경망 기반 point cloud 표현을 효율적이고 정확한 장면 표현으로 사용하여 매핑 및 추적에 적용할 수 있음을 보임
    - noise가 있는 실제 데이터에서 안정적으로 추적하는 데 어려움을 겪음
- point 기반 표현은 특히 서로 독립적이고 빠르게 변환할 수 있기 때문에 map correction 수행에 적합
- Point-SLAM의 data-adaptive scene encoding을 계승하고 loop closure을 확장하여 전역적으로 일관된 지도와 정확한 궤적 추적을 달성하는 Loopy-SLAM 소개

논문의 기여
- Loopy-SLAM 제안
    - scene 탐색 동안 data-driven 방식으로 반복적으로 성장하는 point cloud submap에 neurla features를 고정하는 dense RGBD SLAM
    - 카메라 움직임에 따라 submap을 동적으로 생성
    - submap keyframe 간에 pose graph를 점진적으로 구축
    - global place recognition은 loop closure을 실시간으로 감지하고 scene representation에서 간단하고 효율적인 rigid correction을 통해 trajectory와 submap을 전역적으로 정렬하는데 사용됨(그림 1 참고)
- 이전 연구와 달리 scene representation의 gradient 업데이트나 reintegration 전략 없이 dense neural SLAM에서 loop closure을 구현하는 직접적인 방법 제안
- 전통적으로 rigid submap registration은 overlapping 영역에서 visible 이음새를 생성할 수 있음
    - neural point cloud를 기반으로 한 논문의 접근법은 이를 회피함
    - trajectory capture가 끝날 때 색상과 geometry 구조의 feature refinement를 적용
    - 겹치는 영역에서 submap의 feature 융합 전략을 도입
        - 과도한 메모리 사용 방지
        - 렌더링 성능 향상

## 2. 


## 3. Method


이 섹션에서는 우리의 고밀도 RGBD SLAM 시스템에 대해 자세히 설명합니다. 구체적으로, 장면 공간이 탐색됨에 따라 신경 포인트 클라우드의 서브맵을 점진적으로 확장합니다. 프레임-대-모델 추적을 매 활성 서브맵에 대해 직접 손실 공식으로 매핑과 함께 적용합니다(3.1절). 카메라 움직임에 따라 새로운 글로벌 키프레임과 관련 서브맵을 동적으로 생성합니다. 서브맵이 완료되면, 잠재적 루프 클로저를 감지하기 위해 글로벌 장소 인식을 수행하고, 관련된 에지를 포즈 그래프에 추가하며, 이는 고밀도 표면 등록 제약을 사용하여 최적화됩니다. 장면 표현을 더욱 정교하게 하기 위해, 경로 캡처가 끝나면 먼저 서브맵이 중첩되는 부분에서 특징 융합을 적용한 후 색상 및 지오메트리 특징 정제를 수행합니다(3.2절). 그림 2는 개요를 보여줍니다.

### 3.1 Neural Point Cloud-based SLAM

[43]에서 제안된 point cloud 기반 SLAM
- Loop closure 시 dense scene representation을 변형시키는 데 적합
- geometry 및 외관 정보가 모두 point cloud에 고정된 특징으로 locally encoded 되어 있기 때문
- 이러한 anchor point는 원본 입력 데이터를 사용하여 dense representation을 처음부터 계산할 필요 없이 scene을 변형시키기 위해 지속적으로 이동될 수 있음
- loop closure 업데이트에 맞게 feature point cloud representation을 적응시키기 위해 이를 $s \in \mathbb{N}$개의 submap 집합으로 재정의
- 각 submap은 N개의 neural point collection을 포함하는 neural point cloud $P^s$를 포함
$$
\displaystyle
P^s = \{ (p^s_i, f^{s,g}_i, f^{s,c}_i) | i = 1, ..., N^s \}
\tag{1}
$$
- 각 point는 위치 $p_i^s \in \mathbb{R}^3$ 및 geometric & color feature descriptor $f_i^{s,g} \in mathbb{R}^32, f_i^{s,c} \in \mathbb{R}^32$를 갖는다.

**Building Submap Progressively**

- mapping과 tracking은 항상 active submap에서 수행됨(가장 최근에 생성된 submap)
- submap의 첫 번째 프레임을 global keyframe으로 연관시킴
- keyframe은 global 기준 좌표계에서 submap의 pose를 정의
- Point-SLAM[43]에서 point 추가 전략과 동적 해상도를 채택
    - 데이터에 의존적인 방식으로 각 submap을 점진적으로 확장하여 효율성과 정확성 보장
    - 깊이와 색상 렌더링은 [43]을 따름
    - 원점 O를 갖는 카메라 자세가 주어지면 점 집합 $x_i$를 다음과 같이 샘플링
    $$
    x_i = O + z_i d,   i ∈ {1, ..., M}
    \tag{2}
    $$
    > $z_i \in \mathbb{R}$: point depth  
    > $\text{d} \in \mathbb{R}^3$: ray direction
    - point x_i가 샘플링된 후, 투명도와 색상은 MLP를 사용하여 다음과 같이 디코딩됨
    $$
    o_i = h(x_i, P^{s, g} (x_i)) \quad c_i = g_\xi (x_i, P^{s, c} (x_i))
    \tag{3}
    $$
    > $P^{s, g}(x_i)$: submap $P^s$에서 보간된 기하 features  
    > $P^{s, c}(x_i)$: submap $P^s$에서 보간된 색상 features  
    > geometry 및 color decoder MLPs는 각각 $h$와 $g$로 표시됨
- 매핑 전략에 조정을 함
    - feature 외에도 decoder은 입력으로 3D 점 $x_i$를 받아 학습 가능한 gaussian positional encoding이 적용됨
    - 기하학적 MLP는 고정한 상태로, 인코딩은 실시간으로 최적화될 수 있도록 허용함
    - loop closure 시 점들이 이용하면 새로운 위치에서 이전과 정확히 같은 값으로 디코딩되지 않을 수 있음
    - 실시간 적응형 위치 인코딩을 사용하면 각 점의 특징을 업데이트하는데 더 비용이 많이 드는 대신 시스템에게 간단히 조정할 수 있는 방법을 제공
    - 색상 $\hat{I}$와 depth $\hat{D}$의 feature 보간 및 렌더링 방정식 내용은 [43] 참고

**Tracking and Mapping Losses**

- tracking과 mapping을 active submap에서 교대로 수행
    - [43]과 동일
- tracking
    - RGBD 프레임에서 얻은 $M_t$개의 픽셀 렌더링
    - 센서로부터 얻은 $D$ 와 $I$에 대해 re-rendering loss 최소화
    $$
    \displaystyle
    L_{track} = \sum_{k=1}^{M_t} \frac{|D_k - \hat{D}_k|_1}{\hat{S}_D} + \lambda_t |I_k - \hat{I}_k|_1
    \tag{4}
    $$
    > $\hat{D}$: 렌더링된 깊이  
    > $\hat{I}$: 렌더링된 색상  
    > $\hat{S}_D$: $\hat{D}$의 분산  
    > $\alpha_t$: 하이퍼파라미터
    - mapping을 위해 프레임 전체에서 $M$ 픽셀을 렌더링하고 다음 손실 최소화
    $$
    \displaystyle
    L_{map} = \sum_{k=1}^{M} |D_k - \hat{D}_k|_1 + \lambda_m |I_k - \hat{I}_k|_1
    \tag{5}
    $$

**Keyframe Selection and Submap Initialization**

- submap을 너무 자주 생성하면 pose drift 발생 가능  
(특히 작은 loop가 많은 궤적에서)
- 기존 연구[8, 12, 27] 처럼 고정된 간격으로 global keyframe을 생성하는 대신, 카메라 움직임 기반으로 동적으로 global keyframe 생성
- submap 생성 조건
    - active submap의 global keyframe에 대한 회전 각도가 임계값 $\sigma$를 초과하거나
    - relative translation이 임계값 $\theta$를 초과하는 경우
- submap 초기화
    - 새로운 submap $P_s$를 생성할 때, 매핑 속도를 높이기 위해 이전 submap $P^{s-1}$의 neural point cloud를 새로운 global keyframe에 투영하여 초기화
- 각 submap 내에서 일정한 간격으로 local keyframe을 생성하여 매핑 제약[43]
    - global scene representation이 아니라 submap 단위로 적용됨
    - 새로운 submap이 초기화되면 기존 local frame은 삭제됨


### 3.2 Loop Closure and Refinement

- 새로운 submap을 시작하기 전에 global place recognition을 수행하여 pose graph에 edge 추가
- loop closure edge 제약은 coarse to fine registration 전략으로 계산됨
- pose graph optimization(PGO)는 강건한 line process 기반 최적화를 통해 outlier edge 후보를 제거
- PGO 출력은 refined global keyframe poses
    - 이를 활용해 모든 frame 포즈와 submap의 포인트 보정
- 궤적 캡처가 끝난 후에는 feature fusion과 refinement를 모든 submap에 대해 수행

**Global Place Recognition**

- 임의의 drift를 보정하기 위해 모든 global keyframe을 Bag of visual Words(BoW) 데이터베이스에 추가
- global keyframe이 생성될 때마다 BoW DB에 삽입됨
- MIPS-Fusion[55]와 대비됨
    - submap 간 중첩을 통해 loop closure 검출
    - 큰 드리프트 보정에는 한계가 있음

**Pose Graph Optimization**

- 각 노드 $T_s \in \text{SE}(3)$은 global keyframe의 world 좌표 포즈에 대한 보정값으로 정의됨
- submap $P^s$와 $P^{s+1}$의 인접한 keyframe 사이에 identity 제약 $\{I_s\}$를 이용하여 odometry edge들을 추가적으로 채움
- submap 이 왼료되면 BoW DB를 조회하여 비인접 node들 사이에 loop edge 제약 $\{T_{st}\} \in \text{SE}(3)$을 추가
- BoW에서 상위 $K$개의 이웃을 쿼리하고 시각적 유사도 점수가 $s_{min}$보다 큰 경우에만 pose graph에 추가
    - $s_{min}$: global keyframe과 해당 submap 프레임들 간 최소 점수로 동적으로 계산됨
- 온라인 방식으로 PGO를 수행하여 실시간 submap 간 드리프트를 조기에 보정
- Dense surface registration 기반의 강건한 최적화 전략을 사용하여 이상치 loop edge를 제거[8]
- relative pose residual[32] 대신, submap 자체의 표면 대응점을 활용하는 objective 사용
- 잘못된 loop edge에 강건하도록 loop edge에 대한 jointly optimized weight($l_{st} \in [0, 1]$)로 line process $\mathbb{L} = \{ l_{st} \}$가 추가됨
- 다음 목적 함수를 최소화하여 loop weights $\mathbb{L}$와 함께 global keyframe pose corrections $\mathbb{T} = \{ T_s \}$를 최적화
$$
\displaystyle
\mathbb{E}(\mathbb{T}, \mathbb{L}) = ∑_s f(T_s, T_{s+1}, I_s) 
        + λ \left(
        ∑_{s,t} l_{st} f(T_s, T_t, T_{st})
        + μ ∑_{s,t} (\sqrt{l_{st}} - 1)^2
        \right)
\tag{6}
$$
> $\lambda,\mu$: 하이퍼파라미터  
> $f(T_s, T_t, X)$: 두 서브맵 $P_s, P_t$ 간의 대응점($p, q$)의 제곱 거리 합
$$
\displaystyle
f(T_s, T_t, X) = ∑_{(p,q)} ||T_s p - T_t q||^2 / ||T_s p - T_t Xp||^2
\tag{7}
$$
> 마지막 항: trivial solution(자명한 해)를 방지하는 정규화 항
- 최적화는 Levenberg-Marquardt 알고리즘으로 수행

최적화 절차
1. $\mathbb{T}$를 identity로 초기화
2. 1단계: $l_{st} < l_{min}$인 loop edge를 제거
3. 2단계: 남은 loop edge를 모두 사용하여 최적화

- PGO의 출력은 global keyframe pose에 대한 rigid correction terms $\mathbb{T}$ set
- 이 보정항은
    - keyframe world coordinate pose
    - 각 submap에 속한 frame pose
    - submap 자체  
    에 모두 적용됨
