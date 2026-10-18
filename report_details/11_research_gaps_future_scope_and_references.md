# Research Gaps, Future Scope, and Formal References

## 1. Critical Research Gaps Identified in Existing Literature

As critically analyzed in Chapter 2 (Section 2.5) of the Project Report, several fundamental research gaps characterize existing neuro-oncological AI frameworks:

### 1.1 Gap 1: Interpretability vs. Computational Cost
High-capacity models like pure Vision Transformers (ViT) or full 3D dense convolutional networks demand prohibitive computational hardware (multiple enterprise GPUs with 24GB+ VRAM), preventing deployment in standard hospital diagnostic workstations. Conversely, lightweight standalone models (e.g., MobileNet or shallow CNNs) suffer from reduced diagnostic sensitivity on heterogeneous tumor boundaries.

### 1.2 Gap 2: The "Skull Leakage" and Superficial XAI Problem
In standard academic literature, Explainable AI (XAI) techniques such as Grad-CAM are often applied as an uncalibrated post-hoc visualization step. Because convolutional neural networks are attracted to high-contrast calvarial bones, dura mater, and scanner markings, standard Grad-CAM frequently generates false-positive activations outside the brain. Almost no existing studies implement automated anatomical masking to enforce intracranial bounding constraints.

### 1.3 Gap 3: Gap Between Classification and Actionable Segmentation
Most published CAD papers stop at providing an image-level label (e.g., "Glioma: 98%"). However, neurosurgeons and oncologists cannot plan stereotactic biopsies or craniotomies from a simple label. Supervised segmentation architectures (like U-Net) require costly manual polygon ground-truth annotations. There is an acute need for weakly supervised pipelines that bridge classification feature gradients directly into pixel-level segmentations without requiring manual training masks.

### 1.4 Gap 4: Disconnect from Clinical Workflow
Academic papers rarely develop end-to-end clinical tooling. Native DICOM ingestion, dynamic Hounsfield/MRI window leveling, persistent clinical database audit logging, and automated generation of formal PDF radiology reports are universally omitted from typical deep learning benchmarks.

---

## 2. Future Scope and Architectural Evolution

Chapter 4 (Section 4.3) outlines the key avenues for continuing research and technical expansion:

### 2.1 Transition to Modern ConvNeXt Architectures
While DenseNet121 and InceptionV3 establish a powerful baseline, convolutional network design has undergone a major paradigm shift. Incorporating modern **ConvNeXt** backbones (Liu et al., CVPR 2022) introduces key architectural modernizations:
- 7x7 depthwise separable convolutions simulating Vision Transformer receptive fields.
- Inverted bottleneck designs (4x channel expansion) reducing FLOP overhead.
- LayerNorm replacing BatchNorm to eliminate minibatch dependency.
- GELU activations providing smooth, differentiable gradient propagation.
This roadmap bridges the pre-ConvNeXt dual-backbone framework directly into modern state-of-the-art medical benchmarks.

### 2.2 3D Volumetric Scan Aggregation
The current pipeline operates on 2D MRI cross-sections. Future expansions will implement 3D volumetric context, linking adjacent axial slices across the complete scan volume to calculate true 3D tumor volume (in cubic centimeters, cm^3) rather than 2D surface area percentages.

### 2.3 Multi-Parametric MRI Fusion
While the current system utilizes Contrast-Enhanced T1-weighted images, incorporating complementary sequences—T2-weighted (for vasogenic edema mapping), FLAIR (for non-enhancing infiltrative margins), and Diffusion-Weighted Imaging (DWI, for cellular density assessment)—will enable fine-grained automated tumor sub-region segmentation:
1. Enhancing Tumor Margin.
2. Central Necrotic Cavity.
3. Peritumoral Vasogenic Edema.

### 2.4 Multi-Center Prospective Clinical Validation
To achieve regulatory clearance (such as FDA 510(k) or CE-mark software as a medical device), the system will undergo multi-center prospective reader studies across multiple scanner hardware vendors (Siemens Magnetom, GE Signa, Philips Ingenia) to validate cross-manufacturer generalization.

---

## 3. Formal Bibliographic References

The following peer-reviewed literature forms the formal citation base of the project report (`Report Body_merged.pdf`):

```text
[1] S. Kundu, M. Hasan and S. Saha,
    "A Hybrid Approach to Brain Tumor Detection Using Deep Learning and Explainable AI,"
    2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN),
    Rangpur, Bangladesh, 2025, pp. 1-6.
    DOI: 10.1109/QPAIN66474.2025.11171665.

[2] K. M. Hosny, M. A. Mohammed, R. A. Salama et al.,
    "Explainable ensemble deep learning-based model for brain tumor detection and classification,"
    Neural Computing and Applications, vol. 37, pp. 1289–1306, 2025.
    DOI: 10.1007/s00521-024-10401-0.

[3] A. X. S, M. A. Reddy, N. S and R. D,
    "A Comprehensive Computer Vision Framework for Brain Tumor Classification and Explainable Analysis from MRI Scans,"
    2025 2nd Asia Pacific Conference on Innovation in Technology (APCIT),
    Mysore, India, 2025, pp. 1-10.
    DOI: 10.1109/APCIT65661.2025.11411270.

[4] S. Arun Kumar, S. Sasikala, P. Caroline, L. Chamily Nafia and M. Kavya,
    "Explainable AI Framework for Enhanced Brain Tumour Detection and Classification,"
    2024 First International Conference on Software, Systems and Information Technology (SSITCON),
    Tumkur, India, 2024, pp. 1-6.
    DOI: 10.1109/SSITCON62437.2024.10797113.

[5] M. S. H. Omi et al.,
    "Integrating Deep Neural Networks with Explainable AI for Precise Brain Tumor Detection and Classification,"
    2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN),
    Rangpur, Bangladesh, 2025, pp. 1-6.
    DOI: 10.1109/QPAIN66474.2025.11172007.

[6] S. J. Mohammed and G. A. Al-Suhail,
    "Explainable AI-based Deep Learning for Brain Tumor MRI Analysis: A Review,"
    2025 6th International Conference on Communications, Information, Electronic and Energy Systems (CIEES),
    Ruse, Bulgaria, 2025, pp. 1-7.
    DOI: 10.1109/CIEES66347.2025.11300122.

[7] S. Pereira, A. Pinto, V. Alves and C. A. Silva,
    "Brain Tumor Segmentation Using Convolutional Neural Networks in MRI Images,"
    IEEE Transactions on Medical Imaging (TMI), vol. 35, no. 5, pp. 1240-1251, May 2016.
    DOI: 10.1109/TMI.2016.2538465.

[8] A. Bala, A. P. Sharma, G. Kaur, H. M. Dixit, S. Kumari and P. Singla,
    "Explainable Deep Learning for Brain Tumor Detection: Integrating DeepLabV3+ and Bayesian Optimization,"
    2025 3rd World Conference on Communication & Computing (WCONF),
    Raipur, India, 2025, pp. 1-7.
    DOI: 10.1109/WCONF64849.2025.11233489.
```
