# Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras


---

- Visual SLAM

---

paper: 

---

목차

0. [Abstract](#abstract)
1. 

---

## Abstract

neural rendering 및 SLAM 시스템의 통합은 최근 joint localization 및 photorealistic view 재구성에서 유망한 결과를 보임
 - 하지만 기존 방법은 implicit representations에 완전히 의존. 자원을 많이 소모하여 휴대용 장치에서 실행할 수 없음

**Photo-SLAM**
 - hyper primitive map을 가진 새로운 SLAM 프레임워크
 - 위치 추정을 위해 명시적 기하학적 특징을 동시에 활용
 - 관찰된 환경의 texture 정보를 표현하기 위해 implicit photometric features를 학습
 - 기하학적 특징에 기반하여 hyper primitives map을 적극적으로 밀집화
 - multi-level features를 점진적으로 학습하여 photorealistic mapping 성능을 향상시키기 위해 Gaussian-Pyramid 기반의 훈련 방법을 추가로 도입
 - monocular, stereo, RGB-D 데이터셋을 사용한 광범위한 실험 결과는 Photo-SLAM이 online photorealistic mapping에서 현재 SOTA SLAM 시스템을 현저히 능가함을 입증
    - PSNR이 30% 더 높음
    - Replica dataset에서 렌더링 속도가 수백 배 더 빠름
    - Jetson AGX Orin과 같은 임베디드 플랫폼에서 실시간 속도로 실행 가능
    - 로봇 응용의 가능성을 보임

## 1. Introduction

카메라를 이용한 SLAM의 목표: 자율 시스템이 환경을 탐색하고 이해할 수 있게 함

전통적인 SLAM
 - 기하학적 mapping에 초점을 맞춤
 - 정확하지만 시각적으로 단순한 feature 제공

neural rendering의 발전은 SLAM 파이프라인에 photorealistic view 재구성을 통합할 수 있는 가능성을 보임  
-> 로봇 시스템의 인식 능력 향상
 - 둘의 결합을 통해 얻은 유망한 결과에도 불구하고, 기존 방법들은 단순하고 심하게 암묵적인(implicit) 표현에 의존  
 -> 계산 집약적. 자원이 제한된 장치에 배포하기 적합하지 않음
    - 예: Nice-SLAM은 환경을 나타내는 학습 가능한 feature을 저장하기 위해 hierarchical grid를 활용하는 반면, ESLAM은 multi-scale 압축 텐서 components를 이용.
    - 이후 카메라 포즈를 공동으로 추정하고 ray sampling 배치의 reconstruction loss를 최소화하여 feature을 최적화
    - 이러한 최적화 과정은 시간이 많이 소요됨.
 - RGB-D 카메라, dense optical flow estimator 또는 monocular depth estimator과 같은 다양한 소스에서 얻은 해당 깊이 정보를 통합하여 효율적인 수렴을 보장하는 것이 필수적
 - 암묵적 feature은 MLP에 의해 디코딩되기 때문에 최적의 성능을 위해 ray sampling을 정규화하기 위해 경계(bounding area)를 신증하게 정의하는 과정이 일반적으로 필요함[14]  
 -> 이는 시스템의 확장성을 제한
 - 이러한 한계는 휴대 가능한 플랫폼을 사용하여 알려지지 않은 환경에서 실시간 탐색 및 매핑 기능을 제공할 수 없음을 의미함

**Photo-SLAM**

 - 기존 방법의 확장성 및 계산 자원 제약을 해결하면서 정밀한 위치 추적 및 온라인 photorealistic mapping을 달성하는 프레임워크
 - ORB features, rotation, scaling, 밀ㄹ도, spherical harmonic(SH) coefficients를 저장하는 point cloud로 구성된 hyper primitive map을 유지
 - hyper primitive map
    - system이 factor graph solver을 사용하여 효율적으로 추적을 최적화
    - 원본 이미지와 렌더링 이미지 간의 loss를 역전파하여 해당 매핑을 학습할 수 있도록 함
    - 이미지는 ray sampling 대신 3D Gaussian splatting을 사용하여 렌더링됨
 - 3D Gaussian splatting renderer
    - view 재구성 비용이 줄어들 수 있지만, online incremental mapping을 위한 high-fidelity rendering 생성을 가능하게 하지는 않음. 특히 monocular 시나리오에서
 - dense depth 정보에 의존하지 않고 고품질 mapping을 달성하기 위해 geometry-based densification 전략과 Gaussian-Pyramid-based(GP) 학습 방법을 추가로 제안
    - GP 학습은 multi-level feature의 점진적 습득을 촉진하여 시스템의 매핑 성능을 효과적으로 향상시킴

제안된 접근법의 효능을 평가하기 위해, monocular, stereo, RGB-D 카메라로 캡처한 다양한 데이터셋을 사용하여 광범위한 실험을 수행
 - Photo SLAM이 localization 효율성, photorealistic mapping 품질 및 렌더링 속도에서 SOTA를 달성함을 보임
 - embedded device에서 Photo-SLAM 시스템의 실시간 실행은 실용적인 로봇 응용 프로그램의 잠재력을 보임

**논문의 기여**
 - hyper primitives map에 기반한 최초의 simultaneous localization and photorealistic mapping 시스템 개발. 실내외 환경에서 monocular, stereo, RGB-D 카메라를 지원
 - 모델의 high-fidelity mapping을 실현할 수 있도록 multi-level feature을 효율적이고 효과적으로 학습할 수 있는 Gaussian-Pyrarmid-based learning을 제안
 - C++ 및 CUDA로 완전히 구현된 시스템은 SOTA를 달성하며 embedded platform에서 실시간 속도로 실행 가능

## 2. Related Work

Visual localization and mapping은 카메라를 통해 알려지지 않은 환경의 적절한 표현을 구축하면서 해당 환경 내의 자세를 추정하는 문제

SfM과 달리, Visual SLAM은 일반적으로 정확성과 실시간 성능 간의 더 나은 균형을 추구함

**Graph Solver vs Neural Solver**

 - 고전적인 SLAM 방법은 변수(pose와 landmark)와 측정값(관측 및 제약 조건) 간의 복잡한 최적화 문제를 모델링하기 위해 factor graph를 널리 채택
 - SLAM은 실시간 성능을 달성하기 위해 비용이 많이 드는 연산을 피하면서 자세 추정을 점진적으로 전파
    - 예: ORB-SLAM 시리즈[2, 23, 24]는 연속 프레임에서 가벼운 기하학적 특징을 추출 및 추적하는데 의존. 전역적으로가 아니라 지역적으로 bundle adjustment를 수행
    - LSD-SLAM[7] 및 DSO[8]과 같은 직접 SLAM은 기하학적 feature 추출 비용 없이 raw image intensities에서 작동
    - 이들은 제한된 자원 시스템에서도 point cloud로 표현된 sparse / semi dense map을 온라인으로 유지
 - 딥러닝을 통해 SLAM에 학습 가능한 매개변수와 모델이 도입되어 파이프라인이 미분 가능해짐
    - DeepTAM[45]와 같은 일부 방법은 신경망을 통해 카메라 자세를 end-to-end로 예측하지만, 정확도가 제한적임
    - 성능을 향상시키기 위해, D3VO[41] 및 Droid-SLAM[34]와 같은 일부 방법은 SLAM 파이프라인에 monocular 깊이 추정[12] 또는 dense optical flow estimation[33] 모델을 감독 신호로 도입  
    -> 장면 기하를 명시적으로 나타내는 depth map을 생성할 수 있음
    - 훈련에 사용할 수 있는 대규모 합성 SLAM 데이터셋인 TartanAir[37] 덕분에 RAFT[33]에 기반한 Droid-SLAM은 SOTA를 달성.  
    -> 순수 신경망 기반 solver는 계산 비용이 많이 들고 unseen scenes에서 성능이 크게 저하될 수 있음

**Explicit Represenatation vs Implicit Representation**
 - dense reconstruction을 얻기 위해 KinectFusion[15], BundleFusion[6], InfiniTAM[25]를 포함한 일부 방법은 암묵적 표현이 Truncated Signed Distance Function(TSDF)[5]를 사용하여 들어오는 RGB-D 이미지를 통합하고 연속적인 표면을 재구성  
    - GPU에서 실시간으로 실행될 수 있음
    - dense reconstruction을 얻을 수 있지만, view rendering 품질은 제한적임
 - NNeRF[21] 등 neural rendering 기술은 놀라운 novel view 합성을 달성
    - 카메라 포즈가 주어지면 MLP를 통해 장면 기하학과 색상을 암묵적으로 모델링
    - MLP는 렌더링 이미지와 훈련 view의 손실을 최소화하여 최적화
    - iMAP[30]은 이후 NeRF를 incremental mapping에 적응시켜 MLP와 카메라 포즈 모두를 최적화
    - Nice-SLAM[46]은 깊은 MLP 쿼리 비용을 줄이기 위해 multi-resolution grids를 도입
 - Co-SLAM[36]과 ESLAM[16]은 각각 InstantNGP[22]와 TensoRF[3]을 탐색하여 매핑 속도를 가속화
    - 카메라 포즈와 기하학적 표현의 암묵적 공동 최적화는 여전히 불안정함
    - 불가피하게, RGB-D 카메라의 명시적 깊이 정보나 radiance field의 빠른 수렴을 위한 추가 모델 예측에 의존

Photo-SLAM
 - dense mesh를 재구성하기 보다는 몰입형 탐색을 위한 관찰된 환경의 간결한 표현을 회복하는 것을 목표로 함  
 -> 명확한 기하학적 특징점을 활용하여 정확하고 효율적인 위치 추정을 수행하는 online hyper primitive로 지도를 유지
 - 텍스쳐 정보를 포착하고 모델링하기 위해 암묵적 표현을 활용
 - 밀집 깊이 정보에 의존하지 않고도 고품질 매핑을 달성하므로 RGB-D 카메라 뿐만 아니라 monocular 및 stereo 카메라도 지원할 수 있음


## 3. Photo-SLAM

