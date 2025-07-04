


## 2. Related Work

**Global Image Descriptors**

초기 이미지 descriptor 접근 방식(local key-point descriptors 집합)
 - Bag of Words(BoW)
 - Fisher Vectors(FV)
 - Vector of Locally Aggregated Descriptors(VLAD)

집합은 sparse keypoint location 또는 image grid의 dense sanpling에 기반할 수 있다.

**딥러닝 기반 아키텍처로 재정의한 방법:**
 - NetVLAD
 - NetBoW
 - NetFV

최신 접근 방법
 - ranking-loss based learning
 - novel pooling
 - contextual feature reweighting
 - large scale re-training
 - semantics-guided feature aggregation
 - 3D 사용
 - 추가 센서
 - sequence
 - 이미지 외형 변형

global descriptor matching을 통해 얻어진 장소 매칭은 종종 순차 정보, query 확장, geometric 검증 및 feature fusion을 사용해서 재정렬됨

이 논문은 이미지 description의 local-to-global process를 반전시켜 global descriptor인 NetVLAD로부터 multi-scale patch features를 유도하는 Patch-NetVLAD를 제안


##

### 3.6 IntegralVLAD
다양한 scale에서 patch descriptor을 추출하는 계산을 돕기 위해, 적분 이미지와 유사한 새로운 IntegralVLAD 공식을 제안

patch에 대해 집계된 VLAD descriptor는 (projection layer 이전) 각각 patch 내 단일 feature에 해당하는 모든 $1 \times 1$ patch descriptors의 합으로 계산될 수 있다.

-> multi-scale 융합을 위한 patch descriptor을 계산하는데 사용할 수 있는 integral patch feature map을 미리 계산할 수 있다.

integral patch feature map $\mathcal{I}$:

$$
\displaystyle
\begin{aligned}

&mathcal{I}(i, j) = \sum_{i'<i, j'<j} \text{f}_{i', j'} ^ 1
&(6)

\end{aligned}
$$

> $\text{f}_{i', j'}$: feature space의 공간 index $i', j'$위치의 patch 크기 1에 대한 VLAD 집계 patch descriptor(projection 이전)

integral feature map 내의 4개의 reference를 사용한 산술을 포함하는 일반적인 접근 방식을 통해 임의의 크기에 대한 patch 특징을 복구할 수 있다.  
-> kernel $K$를 가진 2D depth-wise diltated convolutions를 통해 실제로 구현됨

$$
\displaystyle
\begin{aligned}

&K = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
&(7)

\end{aligned}
$$

> dilation == 요구 patch size

## 4. Experimental Results

### 4.1 Implementation

 - Pytorch에서 구현
 - patch feature 추출 전에 모든 이미지를 $640 \times 480$ 픽셀로 resize
 - 두 가지 데이터셋으로 Vanilla NetVLAD feature extractor 훈련
    - Pittsburgh 30k[80] 데이터셋(Pittsburgh와 Tokyo dataset)로 도시 이미지 훈련
    - Mapillary Street Level Sequences [82]로 다른 모든 조건에 대해 훈련
 - 훈련을 위한 모든 hyper parameter은 [3]과 동일
    - 예외적으로, Mapillary 훈련에서 대규모 데이터셋에서 빠른 훈련을 위해 64개 대신 16개의 클러스터를 사용

patch 크기와 관련 가중치를 찾기 위해 RobotCar Seasons v2 훈련 데이터셋에서 Grid search 진행
 - patch 크기가 $d_x = d_y = 5$(원본 이미지의 228 \times 228 영역에 해당)
 - 단일 패치 크기가 사용될 때 stride $sp = 1$로 설정됨
 - multi-scale fusion을 위한 사각형 패치 크기: $2, 5, 8$. 각 가중치 $w_i = 0.45, 0.15, 0.4$
 - 이 단일 구성을 모든 데이터셋 실험에 사용

### 4.2 Datasets
