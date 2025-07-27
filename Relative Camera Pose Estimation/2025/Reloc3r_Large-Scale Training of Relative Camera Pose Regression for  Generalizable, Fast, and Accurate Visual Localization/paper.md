# Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization



---

- Relative Camera Pose Estimation

---

url:
- [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.pdf) (CVPR 2025)

---

요약



---

목차

0. [Abstract](#abstract)
1. 

---


## Abstract

**Visual Localization**
- query 이미지의 카메라 포즈를 데이터베이스를 기준으로 결정
- 기존의 신경망 기반 방법
    - 빠른 추론
    - 새로운 장면에 잘 일반화 되지 않음
    - 정확하지 않을 수 있음

**Reloc3r**
- 상대 포즈 회귀 네트워크 + 절대 포즈 추정을 위한 minimalist motion 평균 모듈
- 800만 개 포즈가 지정된 이미지 쌍으로 훈련
- 좋은 성능 & 일반화능력
- 6개 공가 데이터셋으로 실험

## 1. Introduction

**전통적인 Visual Localization**
- struction-from-motion(SfM) 기술에 의존하여 3D 모델 재구성
    1. 쿼리 이미지의 픽셀을 3D 장면의 포인트와 일치시킴
    2. 기하학적 최적화를 통해 카메라 자세 계산
- 높은 정확도
- 테스트 시간에 비효율성으로 어려움을 겪는 경우가 많음  
-> 확장성에 한계를 가짐

**Scene coordinate regression 방식**
- pixel-point 대응에 대한 대안적인 관점
- 신경망을 사용하여 암묵적인 장면 포현을 학습. 이를 통해 dense correspondence 유추
- 일반화에 한계를 가짐
- 종종 GT keypoint 매치 또는 3D point map과 같은 집중적인 감독이 필요하여 학습 데이터를 확장하기 어렵게 함

**Absolute Pose Regression(APR)**
- 이미지에서 카메라 포즈를 직접 회귀
- 빠른 추론 & 높은 정확성
- 장면 특정적, training 중 밀집한 view point coverage를 요구  
-> 실제 적용 가능성 제한됨
- 합성 데이터셋 생성을 통해 정확성을 향상시키려는 방법이 있음
    - 상당한 계산 오버헤드 발생
    - 광범위한 배치를 저해함

**Relative Pose Regression(RPR)**
- 데이터베이스-쿼리 이미지 쌍 간의 상대 포즈를 추정
- 각각의 장면에 대한 훈련의 필요성 완화
- 빠른 추론(테스트 시 효율성)을 유지
- APR 방식의 위치 확인 정확성을 따라잡지 못함
    - 몇몇 RPR 방법은 서로 다른 데이터셋에서 일반화할 수 있지만, 정확도 감소

각 방법들은 새로운 장면 일반화, 테스트 시간 효율성, 카메라 포즈 정확성 세 가지 중 하나의 문제를 가짐

**Reloc3r**
- Transformer과 대규모 훈련을 활용
- DUSt3R의 아키텍처를 베이스로 사용하여 수정
- 훈련 중 상대 포즈의 metric 척도를 무시
- minimalist 모션 평균화 모듈과 통합하여 절대 포즈를 추정
- 객체 중심, 실내외 장면에 걸쳐 다양한 소스에서 약 800만 개의 이미지 쌍을 처리
- 6개의 포즈 추정 데이터셋에서 우수한 성능을 보임

**논문의 기여**
- Reloc3r 제안
    - 새로운 장면에 대해 탁월한 생성
    - 빠른 테스트시간 효율성
    - 높은 카메라 포즈 정확도
- 제안된 완전 대칭 relative pose 회귀 네트워크와 모션 평균화 모듈은 간소화의 원리를 따름
    - 효율적인 대규모 학습이 가능
- 6개의 데이터셋에 대한 결과는 제안 방식의 효과를 일관되게 보임

## 2. Related Work

**Structure-based visual localization**
- multi-veiw 기하학을 통해 카메라 자세 문제를 푸는 방법
- 최신 방법은 2가지 단계를 따름
    1. 이미지 간 or 픽셀과 사전 구축된 3D 모델간의 대응관계를 설정
    2. 노이즈가 있는 대응 관계를 설정하고 카메라 자세를 견고하게 해결
    - 이러한 correspondences는 keypoint 매칭 또는 장면 좌표 회귀를 통해 얻어짐
    - 이후 견고한 추정기를 적용하여 카메라 자세를 추정
    - 이런 방법은 종종 느린 추론 시간을 가짐
- 최근 효율성 지향적인 변형이 매칭을 가속화하기 위해 제안됨[60, 112[]]
    - 복잡한 시스템 설계 & 높은 계산 비용은 해결되지 않음
- 이 방법은 일반적으로 supervision을 위해 실제 대응 관계나 3D point map을 필요로 하여 대규모 학습의 확장성을 제한함

**Camera pose regression**
- 실시간 추론
- 두 가지 방법
    - APR(Absolute Pose Regression)
        - 이미지를 통해 월드 좌표계에서 카메라의 위치와 방향을 밀리초 이내에 직접 회귀
        - 구조 기반 접근 방식의 정확도에 미치지 못함
        - 종종 이미지 검색을 통한 자세 근사와 유사함
        - 일부 방법은 훈련을 위한 dense viewpoint를 만들기 위해 새로운 시점 합성을 사용
            - 상당한 계산 비용 초래. 각 특정 장면에 대해 훈련이 몇 시간 또는 며칠 걸릴 수 있음
        - 포즈 추정기와 scene-specific map 간의 연결을 구축하여 교육 시간을 줄이려는 시도가 있음
            - 장면별 훈련 및 평가에 한정됨
    - RPR(Relative Pose Regression)
        - 쿼리 이미지와 가장 유사한 데이터베이스 간의 상대 자세를 회귀하여 localization
        - 상대적 변환의 metric scale은 단일 데이터베이스 쿼리 쌍에서 대략적으로 추정 가능
        - 정밀한 절대 위치 측정은 multi-view 삼각측량을 통해 가능함
        - 가장 좋은 RPR 방법은 APR 방법들에 비해 상대적으로 뒤쳐짐
        - 기존 RPR 모델의 일반화 능력은 여전히 제한적
            - Relative PN[49] 및 Relpose-GNN[101]과 같은 방법은 새로운 데이터셋 및 장면에 적응 가능.  
                - 오류가 거의 두 배로 증가
                - map 없이 대규모 데이터셋(약 523K)로 훈련되었음에도 여전히 실내외 설정에 대해 별도의 모델에 의존
            - 다른 방법은 multi-view 자세 추정 뿐만 아니라 넓은 베이스라인 및 파노라마 방법을 탐색
                - 데이터셋 특정 훈련에 그치므로 확장성이 제한적
- 기존 방식은 주로 기술 설계에 집중되어 있고, 더 큰 데이터셋에서의 훈련에는 덜 집중함
- 본 논문에서는 객체 중심, 실내외 데이터셋의 다양한 혼합으로 훈련된 첫 번쨰 포즈 회귀 접근법을 제시
- 이전 연구[129] 에서는 포즈 휘귀의 부정확성이 조잡한 특징 위치 지정에서 기인한다고 제안
    - 본 논문에서는 잘 훈련된 패치 수준 회귀 네트워크가 픽셀 수준 특징 매칭의 성과를 달성할 수 있고, 때로 이를 초과할 수 있음을 발견

**Foundtaion Models**
- 다양한 작업에서 강력한 일반화 성능을 보임
- DUSt3R
    - two-view 기하학과 관련된 거의 모든 작업을 처리할 수 있는 3D 기반 모델
    - 다양한 후속 작업에서 성능을 향상시키기 위해 여러번 fine-tuning됨
- 본 논문에서는 DUSt3R의 Transformer 백본을 채택하여 대칭적인 디자인의 Reloc3r을 개발

## 3. Method

**Proble statement**

