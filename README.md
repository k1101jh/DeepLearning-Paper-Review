# DeepLearning Paper Review

딥러닝의 다양한 분야(Computer Vision, NLP, Multi-modal 등)에서 발표된 핵심 논문들을 분석하고 정리한 저장소입니다. 각 논문의 핵심 아이디어, 구조, 방법론을 파악하는 것을 목적으로 합니다.

---

## 📑 분야별 논문 목록 (Index)

### 🧊 3D Vision & Graphics
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**3D reconstruction**](./3D%20reconstruction) | Surface Reconstruction, Mesh Generation | 2024 - 2025 |
| [**Avatar Generation**](./Avatar%20Generation) | Digital Humans, Neural Avatars | 2024 |
| [**Gaussian Splatting**](./Gaussian%20Splatting) | 3D Scene Representation & Rendering | 2023 - 2025 |
| [**World Model**](./World%20Model) | Video Models, Prediction & Planning | 2025 |

### 📍 SLAM & Pose Estimation
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**Visual SLAM**](./Visual%20SLAM) | Simultaneous Localization and Mapping | 2015 - 2026 |
| [**Camera Calibration**](./Camera%20Calibration) | Intrinsic/Extrinsic Calibration | 2024 |
| [**Camera Pose Estimation**](./Camera%20Pose%20Estimation) | 6D Pose, PnP, Perspective-n-Point | 2023 - 2024 |
| [**Camera Relocalization**](./Camera%20Relocalization) | Absolute Pose Regression, Scene Coordinate | 2023 |
| [**Hand Pose Estimation**](./Hand%20Pose%20Estimation) | 3D Hand Mesh & Keypoint | 2023 - 2024 |
| [**Human Pose Estimation**](./Human%20Pose%20Estimation) | 2D/3D Human Pose Analysis | 2023 - 2024 |
| [**Humanoid Control**](./Humanoid%20Control) | Physics-based Motion, RL Control | 2023 - 2024 |
| [**Image Matching**](./Image%20Matching) | Feature Extraction, Matching | 2023 - 2025 |
| [**Relative Camera Pose**](./Relative%20Camera%20Pose%20Estimation) | Essential Matrix, Fundamental Matrix | 2023 - 2024 |
| [**Visual Positioning System**](./Visual%20Positioning%20System) | VPS, Geo-localization | 2023 |

### 🎨 Generative Models
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**Diffusion**](./Diffusion) | Generative Models, Image Synthesis | 2015 - 2024 |

### 💬 Natural Language & Multi-modal
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**LLM**](./LLM) | Large Language Models, Transformers | 2023 - 2024 |
| [**RAG**](./RAG) | Retrieval-Augmented Generation | 2024 |

### ⚡ Model Efficiency & Optimization
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**Knowledge Distillation**](./Knowledge%20Distillation) | Model Compression, Teacher-Student | 2021 - 2023 |
| [**Low-rank Adaptation**](./Low-rank%20Adaptation%20&%20Factorization) | LoRA, Parameter-Efficient Tuning | 2023 |
| [**Quantization**](./Quantization) | Model Compression & Efficiency | 2021 - 2023 |
| [**Reparameterization**](./Reparameterization) | Structural Reparam, Training vs Inference | 2023 - 2024 |

### 🔍 Fundamental Computer Vision
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**Attention**](./Attention) | Self-Attention, Cross-Attention | 2023 |
| [**Classification**](./Classification) | Image Recognition | 2023 - 2024 |
| [**Depth Estimation**](./Depth%20Estimation) | Monocular/Stereo Depth | 2023 - 2024 |
| [**Detection**](./Detection) | Object Detection | 2023 - 2024 |
| [**Segmentation**](./Segmentation) | Semantic, Instance, Panoptic | 2024 |
| [**Vision Transformer**](./Vision%20Transformer) | ViT, FastViT, Attention Mechanisms | 2020 - 2023 |

### 🔉 Others
| 분야 | 주제 | 연도 |
| :--- | :--- | :--- |
| [**Audio**](./Audio) | Audio Synthesis, Recognition | 2023 |

---

## 🏛️ 리포지토리 구조

```text
category/
└── year/
    └── paper_title/
        ├── paper.md   # 분석 및 정리 내용
        └── images/     # 논문에서 인용된 핵심 Figure
```

---

## ⚖️ 저작권 및 라이선스 (License & Disclaimer)

### License
본 리포지토리의 분석 텍스트 및 주석 내용은 **[MIT License](./LICENSE)**를 따릅니다.

### Disclaimer
1.  **이미지(Figure) 저작권**: 각 리뷰에 포함된 이미지(Figure)의 저작권은 해당 논문의 원저작자 및 출판사에 있습니다. 본 저장소에서는 '공정 이용(Fair Use)' 원칙에 따라 교육 및 연구 목적으로만 인용하고 있습니다.
2.  **공식 링크**: 가급적 공식 사이트(arXiv, CVF, IEEE 등)의 링크나 DOI를 제공하고 있습니다. 유료 논문의 경우 구독 환경에 따라 접근이 제한될 수 있습니다.

