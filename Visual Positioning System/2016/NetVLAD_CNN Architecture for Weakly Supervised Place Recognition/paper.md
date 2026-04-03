# NetVLAD: CNN Architecture for Weakly Supervised Place Recognition

---
Reference

본 문서에 사용된 모든 이미지와 표는 해당 논문에서 발췌하였습니다.

---

## 📌 Metadata
---
분류
- Visual Positioning System

---

url:
- [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html) (CVPR 2016)

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

**대규모 시각 장소 인식(visual place recognition)**
- 주어진 query 사진의 위치를 빠르고 정확하게 인식하는 것

논문의 기여
1. 장소 인식 작업을 위해 직접적으로 end-to-end 방식으로 학습 가능한 CNN 아키텍처 개발
    - NetVLAD
        - 이 아키텍처의 주요 구성 요소
        - 이미지 검색에서 일반적으로 사용되는 'Vector of Locally Aggregated Descriptor' 이미지 표현에 영감을 받은 새로운 일반화된 VLAD 레이어
        - 모든 CNN 아키텍처에 쉽게 연결할 수 있으며 역전파를 통해 학습 가능
2. 학습 절차 개발
    - 새로운 weakly supervised ranking loss 기반
    - 시간에 따른 동일 장소를 묘사하는 이미지(Google Street View Time Machine에서 다운로드)로부터 아키텍처의 파라미터를 엔드 투 엔드 방식으로 학습하기 위함
3. 제안된 아키텍처가 두 개의 도전적인 장소 인식 벤치마크에서 비학습 이미지 표현 및 상용 CNN Descriptors들보다 상당히 우수한 성능을 보이며, 표준 이미지 검색 벤치마크에서 현재 SOTA의 압축 이미지 표현보다 개선됨

## 1. Introduction

**시각 장소 인식(visual place recognition)**
 - 자율 주행, 증강 현실, 저장된 이미지를 사용한 지리적 위치 지정(geo-localizing archival imagery)에서 핵심 기술로 사용됨

장소 인식의 어려움
 - 다양한 조명 조건에서 촬영될 수 있음
 - 시간이 지남에 따라 외관이 변할 수 있음

유사한 장소를 구분할 수 있을 만큼 풍부하면서 전체 도시나 국가를 표현할 수 있을 만큼 압축된 "장소에 대한 표현"은 무엇인가?

**장소 인식 문제**
 - 대규모 지리태그 데이터베이스에 쿼리하여 얻은 시각적으로 가장 유사한 이미지의 위치를 사용하여 쿼리 이미지의 위치를 추정
 - 각 데이터베이스 이미지는 SIFT와 같은 지역 불변 특징을 사용하여 표현됨  
    - 전체 이미지를 위한 단일 벡터 표현으로 집계됨
    - 일반적으로 압축되고 효율적으로 색인화됨
 - 이미지 데이터베이스는 정확한 카메라 포즈를 추정하기 위해 3D 구조로 확장될 수 있음

본 논문에서는 장소 인식을 위해 개발하고 훈련한 CNN 표현이 성능을 향상시킬 수 있는지 조사  
이를 위한 세 가지 과제
1. 장소 인식을 위한 좋은 CNN 아키텍처는 무엇인가?
2. 훈련을 위한 충분한 주석 데이터 수집 방법?
3. 장소 인식 작업에 맞춘 end-to-end 방식으로 개발된 아키텍처의 훈련 방법

과제를 해결하기 위한 세 가지 혁신 제안

1. NetVLAD 기반 CNN 아키텍처 개발
    - 전체 이미지를 통해 추출된 중간 수준(Conv5) 합성곱 feature을 집계하여 효율적인 indexing을 위한 단일 vector 표현으로 만드는 장소 인식을 위한 CNN 개발
    - 이미지 검색 및 장소 인식에서 우수한 성능을 보인 VLAD(Vector of Locally Aggregated Descriptors) 표현에 영감을 받아 학습 가능한 일반화된 VLAD 계층인 NetVLAD 설계
    - 이 계층은 모든 CNN 아키텍처에 연결 가능하며 역전파로 훈련 가능
    - 결과로 얻어진 집계 표현은 주성분 분석(PCA)를 사용하여 이미지의 최종 압축 descriptor을 얻기 위해 압축
2. Google Street View Time Machine을 활용한 대규모 학습 데이터 수집
    - 다양한 시점에서 동일 장소를 묘사하는 파노라마 이미지 수집
    - 이는 전 세계의 광범위한 지역에서 제공되지만, 약한 형태의 supervision만을 제공함
    - 두 파노라마가 GPS를 기반으로 대략 유사한 위치에서 촬영됨을 알고있지만, 파노라마의 어느 부분이 동일한 부분을 묘사하는지는 알지 못함
3. 장소 인식을 위한 학습 절차 개발
    - weakly labelled Time Machine Imagery를 통해 장소 인식 작업에 맞게 end-to-end 학습 진행
    - 얻어진 표현은 시점 및 조명 변화에 강건
    - 건물 외관 및 스카이라인과 같은 이미지의 관련 부분에 집중하도록 학습
    - 여러 장소에서 발생할 수 있는 자동차 및 사람과 같은 혼란스러운 요소는 무시하도록 학습

- 제안한 아키텍처가 두 가지 장소 인식 벤치마크에서 non-learnt 이미지 표현 및 기성 CNN descriptor보다 상당히 우수함
- 표준 이미지 검색 벤치마크에서 SOTA compact image 표현보다 향상됨을 보임

### 1.1 Related work

## 2. Method overview

## 3. Deep architecture for place recognition

대부분의 이미지 검색 파이프라인
1. local descriptor 추출
2. 순서 없는 방식으로 집계

 - 이 절차가 변환 및 부분 가림에 대한 상당한 강인성을 제공
 - descriptor은 조명 및 시점 변화에 대해 강인함
 - multiple scales에서 descriptor을 추출하여 scale 불변성을 보장

**CNN 아키텍처 설계**
 - 표현을 end-to-end로 학습하기 위해
 - 미분 가능한 모듈과 함께 통합되고 원칙 있는 방식으로 표준 검색 파이프라인을 모방함
 1. CNN을 마지막 convolutional layer에서 자르고 이를 dense descriptor extractor로 취급
    - 이는 instance 검색 및 texture recognition에 대해 잘 작동하는 것으로 관찰됨
    - 마지막 합성곱 layer의 출력: $H \times W \times D$ map  
    -> $H \times W$ 공간 위치에서 추출된 D-dimensional descriptor 집합으로 간주될 수 있음
 2. 추출된 descriptor을 고정된 이미지 표현으로 집계하는 새로운 pooling layer **NetVLAD**를 설계
    - VLAD(Vector of Locally Aggregated Descriptors)에서 영감을 받음
    - 이 레이어의 매개변수는 역전파를 통해 학습 가능

### 3.1 NetVLAD: A Generalized VLAD layer($f_{VLAD}$)


3.1. NetVLAD: A Generalized VLAD Layer ($f_{\text{VLAD}}$)
VLAD (Vector of Locally Aggregated Descriptors)

인스턴스 검색 및 이미지 분류에서 널리 사용되는 descriptor pooling 방법

이미지 전체에 걸쳐 로컬 디스크립터의 통계 정보를 집계함

Bag-of-Visual-Words(BoW)는 visual word의 등장 횟수를 저장하는 반면,
VLAD는 **descriptor와 대응하는 클러스터 중심 간의 차이 벡터(= residuals)**의 합을 저장

정의

$N$개의 $D$-차원 로컬 디스크립터 ${x_i}$

$K$개의 클러스터 중심(visual words) ${c_k}$

결과 VLAD 표현 $V$는 $K \times D$ 행렬 (나중에 벡터로 변환됨)

$V$의 $(j, k)$ 원소는 다음과 같이 계산됨:

𝑉(𝑗,𝑘)=∑𝑖=1𝑁𝑎𝑘(𝑥𝑖)⋅(𝑥𝑖(𝑗)−𝑐𝑘(𝑗))V(j,k)= i=1∑N​ a k​ (x i​ )⋅(x i​ (j)−c k​ (j))

여기서:

$x_i(j)$: $i$번째 디스크립터의 $j$번째 차원

$c_k(j)$: $k$번째 클러스터 중심의 $j$번째 차원

$a_k(x_i)$: 디스크립터 $x_i$가 클러스터 $c_k$에 속하는지 여부

$a_k(x_i) = 1$이면 $x_i$는 $c_k$에 속함

그렇지 않으면 $a_k(x_i) = 0$

직관적으로, $V$의 각 column $k$는 클러스터 $c_k$에 할당된 디스크립터들의 residual $(x_i - c_k)$의 합

정규화(normalization)

column-wise로 $L_2$ 정규화 (intra-normalization)

이후 행렬 $V$를 벡터로 변환

전체적으로 다시 $L_2$ 정규화하여 최종 이미지 표현 벡터로 사용