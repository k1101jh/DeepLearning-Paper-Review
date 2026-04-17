# Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Gaussian Splatting

---


url:
- [paper](https://arxiv.org/abs/2403.17898)

---
요약


---

## Abstract

**3DGS**
- NeRF 기반 장면 표현에 비해 우수한 렌더링 충실도 & 효율성
- 대규모 장면에서 Gaussian 개수로 인해 어려움을 겪음
- zoom-out된 view에서는 모든 기본 요소가 투영 크기와 상관없이 렌더링됨
    - 모델 용량을 비효율적으로 사용
    - 다양한 scale의 세부 사항을 캡처하기 어려움

**Octree-GS**
- Level-of-Detail(LOD) 구조
- multi-scale 가우시안 primitives에서 적절한 레벨을 동적으로 선택
- densification을 위한 grow-and-prune 전략을 사용
- 적절한 LOD 레벨로 가우시안을 배치하기 위한 점진적인 학습 전략 제안
- 2D-GS 및 Scaffold-GS와 같은 다른 Gaussian 기반 방법으로 일반화됨
- 대규모 장면에서도 시각적 품질을 손상시키지 않으면서 실시간 속도를 달성
- SOTA 방법보다 최대 10배 빠른 성능을 보임

## 1. Instruction

**3DGS**
- anisotropic Gaussian primitives, tile-based splatting 기술 사용
    - 효율적인 학습 시간
    - 거의 완벽한 시각적 품질 구현
- 단점
    - 3D Gaussian 분포와 실제 장면 구조 사이의 불일치
        - Gaussian primitives는 학습 뷰에 맞게 분포됨
        - 부정확하고 비효율적인 배치를 초래
        - 두 가지 병목
            1. primitives가 일반화를 위해 최적화되지 않음
                - 학습 set과 크게 다른 view를 렌더링할 때 견고성 감소
            2. 실시간 렌더링을 위해 장면 detail을 효율적으로 표현하지 못하는 중복&겹침 primitives 발생
- 가우시안과 장면 구조 간 불일치를 해결하려는 시도
    - Scaffold-GS[3]
        - 규칙적으로 배치된 feature grid를 도입하여 구조 정렬을 향상시킴
        - 렌더링 품질&효율성을 위해 정렬&viewpoint-aware을 조정할 수 있게 함
    - Mip-Splatting[14]
        - 3D-GS 최적화 과정 동안 3D 가우시안의 중복 완화를 위해 3D smoothing과 2D Mip filter을 사용
    - 2D-GS[15]: primitives가 표면과 더 잘 정렬되도록 하여 더 빠른 재구성을 가능하게 함
- 새로운 문제
    - 점점 더 대규모 장면을 기록하고자 하지만, 기존 방법은 확장에 어려움을 겪음
        - primitive selection을 위해 visibility-based filtering에 의존하고 있기 때문
        - view frustum 내의 모든 primitives를 해당 투영 크기를 고려하지 않고 처리  
        -> 거리와 상관없이 모든 객체의 세부 사항 렌더링
            - 불필요한 계산 & 일관화되지 않은 렌더링 속도
            - 크고 복잡한 장면에서 zoom-out 시나리오에서 문제 발생
        - LOD adaptation이 없으면 모든 3D Gaussain이 view간 경쟁을 하도록 강요
            - 다양한 scale에서 rendering 품질을 저하시킴
        
**제안 방법**
- Gaussian 표현에 octree 구조 통합
- multi-resolution grid와 같은 공간 구조
    - 유연한 content 할당 및 실시간 렌더링에 효과적
- Gaussian 표현에 Octree 구조 통합
    - scene을 계층적 grid로 조직하여 LOD 요구 충족
    - 학습&추론에서 복잡하거나 대규모 장면에 효율적으로 적응
    - LOD 레벨은 관찰 범위와 장면 detail의 풍부함에 따라 선택됨
- 점진적 학습 전략
    - 새로운 growing/pruning 접근 방식 도입
- next-level growth 연산자는 LOD 간의 연결을 강화하여 high-frequency detail을 향상
- 불필요한 가우시안은 불투명도와 view frequency를 기준으로 prune됨
- LOD 레벨을 적응적으로 쿼리
    - 렌더링에 필요한 primitive 수를 최소화
    - 일관된 효율성 보장
- coarse & fine scene detail을 효과적으로 분리
    - 적절한 규모에서 정확한 가우시안 배치를 가능하게 함
    - 재구성 정화곧와 texture detail을 크게 향상

**제안 방법의 특징**
- 다른 concurrent LOD 방법과 달리, 단일 training round에서 LOD 효과를 달성하는 end-to-end algorithm
- 학습 시간과 저장 공간을 줄임
- explicit Gaussians, Neural Gaussian을 포함한 다양한 가우시안 표현과 호환됨
- 세밀한 실내 장면과 대규모 도시 관경을 포함한 다양한 데이터셋에서 시각적 성능과 렌더링 속도가 향상됨

**논문의 기여**
- 가우시안 표현에서 LOD 문제를 다루는 최초의 점근 방식
    - Octree 구조 설계를 통해 동적으로 가져온 LOD를 실시간으로 저종
    - 일관된 렌더링 속도를 가능하게 함
- LOD adaptation에 최적화된 새로운 grow-and-prune 전략 개발
- primitives의 보다 신뢰할 수 있는 분포를 장려하기 위해 progressive training 전략 도입
- LOD 전략은 모든 가우시안 기반 방법에 일반화될 수 있음
- 뛰어난 렌더링 품질을 유지하면서 large-scale 장면과 extreme-view sequence에서 SOTA 렌더링 속도 달성

## 2. Related Work

### A. Novel View Synthesis

**NeRF[4]**

### B. Spatial Structures for Neural Scene Representations

### C. Level-of-Detail (LOD)

## 3. Preliminaries

### A. 3D-GS

### B. Scaffold-GS

## 4. Methods

**Octree-GS**
- anchor을 octree 구조로 계층화하여 multiview 이미지에서 neural scene을 학습
- 각 anchor은 명시적 Gaussian, neural Gaussian과 같은 다양한 유형의 Gaussian Primitives를 방출할 수 있음
- 적절한 LOD 레벨에서 anchor을 동적으로 선택하여 일관되게 효율적인 학습 및 렌더링을 보장
- 복잡하거나 대규모 장면에도 효율적으로 적응 가능

### A.LOD-structured Anchors

1) Anchor Definition
    - Gaussian primitives를 관리하기 위해 anchor 도입
        - 다양한 voxel 크기를 갖는 sparse, uniform voxel grid의 중심에 배치됨
    - LOD $L$가 높은 anchor일수록 더 작은 voxel 크기를 갖는 grid 안에 배치됨
        - LOD 0을 가장 거친 수준으로 정의
    - LOD level이 증가함에 따라 더 많은 detail이 포착됨
        - LOD 설계는 누적되는 식
    - LOD $K$에서 렌더링된 이미지는 LOD 0부터 $K$까지의 모든 Gaussian primitives를 rasterize
    - 각 anchor에는 지역적 복잡성을 고려한 LOD bias $\Delta L$이 할당됨
    - 각 anchor은 이미지 렌더링을 위해 $k$개 가우시안 primitives와 연결됨
        - 식 3에 의해 위치 결정
    - 다양한 유형의 가우시안을 지원하도록 일반화되어 있음
        - Gaussian primitive 2D[15] 또는 3D Gaussian[5]와 같이 학습 가능한 개별 특성으로 명시적으로 정의될 수 있거나, 해당 anchor에서 decoding된 neural Gaussian일 수 있음

2) Anchor Initialization
    - sparse SfM points $\rm{P}$에서 octree 구조 anchor을 초기화
        1. 관찰된 거리 범위를 기반으로 octree layer의 수 $K$를 결정
            - 각 학습 이미지 $i$의 카메라 중심과 SfM 점 $j$ 사이의 거리 $d_{ij}$를 계산
        2. $r_d$번째 큰 거리와 $r_d$번째 작은 거리를 각각 $d_{max}, d_{min}$으로 정의
            - $r_d$는 이상치를 제거하기 위한 hyperparameter.
            - 일반적으로 0.999로 설정됨
            $K$ 계산 방법:
            $$
            \displaystyle
            K = \lfloor \log_2(\hat{d}_{max}/\hat{d}_{min}) \rceil + 1
            \tag{5}
            $$
            > $\lfloor \cdot \rceil$: round 연산자
        3. $K$개의 layer를 갖는 octree 구조 격자가 구성됨.
            - 각 layer의 anchor는 해당 voxel 크기로 voxel화됨:
            $$
            \displaystyle
            \rm{V}_L = \left\{\lfloor \frac{\rm{P}}{\delta / 2 ^ L} \rceil \cdot \delta / 2^L \right\}
            \tag{6}
            $$
            > $\delta$: LOD 0에서의 coarsest layer의 기본 voxel size  
            > $\rm{V}_L$: LOD L에서 초기화된 anchors  
            - anchor의 properties, 이와 연관된 Gaussian primitives 또한 초기화됨

3) Anchor Selection
    - 이상적인 anchor
        - 화면에 투영된 가우시안의 pixel footprint를 기반으로 $K$ LOD 레벨에서 동적으로 가져옴
            - 이를 관찰 거리 $d_{ij}$를 사용하여 단순화
                - 카메라 intrinsic이 일정하면 footprint와 비례하기 때문
            - intrinsic이 달라지는 경우, focal scale factor $s$를 적용하여 거리를 동등하게 조정
        - 관찰 거리만으로 $LOD$ 레벨을 추적하는 것은 최적이 아님
            - 각 anchor에 대해 학습 가능한 LOD bias $\Delta L$을 residual로 설정
                - 추론 과정에서 렌더링되는 고주파 영역의 세부 사항을 보다 일관되게 보완(Fig 13.의 객체의 가장자리처럼)
        - 주어진 시점 $i$에 대한 anchor $j$의 LOD level:
        $$
        \displaystyle
        \hat{L_{ij}} = \lfloor L_{ij}^* \rfloor  =\lfloor \Phi (\log_2(d_{max} / (d_{ij} * s))) + \Delta L_j \rfloor,
        \tag{7}
        $$
        > $d_{ij}$: viewpoint $i$와 anchor $j$ 사이의 거리  
        > $\Phi(\cdot)$: 분수 LOD 레벨 $L_{ij}^*$를 범위 $[0, K-1]$로 제한하는 clamping 함수  
        - progressive LOD 기법에서 영감을 받아 단일 LOD level이 아닌 누적 LOD 레벨을 사용하여 이미지 렌더링  
        - anchor은 LOD 레벨 $L_j \leq \hat{L_{ij}}$인 경우 선택됨(Fig 3 참조)
        - 선택된 anchor에서 방출된 gaussian primitives는 이후 rasterizer로 전달되어 렌더링됨
    - inference 단계
        - 서로 다른 LOD 레벨 간의 부드러운 렌더링 전환을 보장
        - 눈에 띄는 아티팩트를 도입하지 않기 위해 [16], [51]에서 영감받은 opacity blending 기법 채택
            - 인접한 level 간의 구간별 선형 보간을 사용하여 LOD 전환을 연속적으로 만듦
                - LOD aliasing을 효과적으로 제거
            - fully satisfied anchor  외에도 기준 $L_j = \hat{L_{ij}} + 1$을 만족하는 nearly satisfied anchor도 선택
            - anchor의 gaussian primitives는 불툼여도가 $L_{ij}^* - \hat{L_{ij}}$로 스케일되어 rasterizer에 전달됨

### B. Adaptive Anchor Gaussians Control

1) Anchor Growing
    - anchor 밀도를 조정하기 위한 기준으로 gaussian primitives의 view-space 위치 gradient를 사용
    - 새로운 anchor는 [3]의 방법에 따라 octree 구조 격자 내의 비어 있는 voxel에 성장됨
    - 매 $T$ 반복마다 생성된 gaussian primitives의 평균 누적 gradient $\nabla_g$를 계산
    - $\nabla_g$가 미리 정의된 임계값 $\tau_g$를 초과하는 gaussian primitive는 중요한 것으로 간주됨
        - 비어 있는 voxel에 위치할 경우, 새로운 anchor로 변환됨
    - 새로 변환된 anchor에는 몇의 LOD level을 적용해야 하는지?
        - 'next-level' 성장 연산 제안
            - 매우 높은 gradient를 갖는 gaussian primitives를 상위 레벨로 승격
            - 다양한 세분화 수준에서 새로운 anchor을 추가하여 성장 전략 조정
            - 높은 LOD level로 과도하게 성장하는 것을 방지하기 위해, 새로운 anchor을 더 높은 LOD level로 성장시키기 어렵게 함
                - 임계값: $\tau_g^L = \tau_g * 2^{\Beta L}$
                > $\tau_g, \Beta$: 하이퍼파라미터. 0.0002, 0.2
            - level $L$에 있는 가우시안은 $\nabla_g > \tau_g^{L + 1}$인 경우에만 다음 레벨로 승격됨

    - 장면의 복잡성 신호로 gradient를 활용하여 LOD bias $\Delta L$을 조정
    - anchor의 gradient는 생성된 gaussian primitive의 평균 gradient로 정의됨


### C. Progressive Training

### D. Appearance Embedding

## 5. Experiments