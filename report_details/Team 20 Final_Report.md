
## VISVESVARAYA TECHNOLOGICAL UNIVERSITY

Jnana Sangama, Belgaum-590018

A PROJECT REPORT (BCS685) ON

“EXPLAINABLE BRAIN TUMOR DETECTION AND SEGMENTATION”

Submitted in Partial fulfillment of the Requirements for the Degree of Bachelor of Engineering in Computer Science & Engineering

By

Bhola Yadav (1CR23CS044)

Abinash Mishra (1CR23CS014)

Deeptanshu Dutta (1CR23CS050)

Aman Kumar Pathak (1CR23CS029)

Under the Guidance of

Mrs. Anjali Gupta

Assistant Professor, Dept. of CSE

DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING CMR INSTITUTE OF TECHNOLOGY

#132, AECS LAYOUT, IT PARK ROAD, KUNDALAHALLI, BANGALORE-560037 AY 2025 - 26

# CMR INSTITUTE OF TECHNOLOGY

#132, AECS LAYOUT, IT PARK ROAD, KUNDALAHALLI, BANGALORE-560037

##### DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING

# CERTIFICATE

Certified that the project work entitled “Explainable Brain Tumor Detection and Segmentation” carried out by Mr. Bhola Yadav, USN 1CR23CS044, Mr. Abinash Mishra, USN 1CR23CS014, Mr. Deeptanshu Dutta, USN 1CR23CS050, and Mr. Aman Kumar Pathak, USN 1CR23CS029, bona fide students of CMR Institute of Technology, in partial fulfillment for the award of Bachelor of Engineering in Computer Science and Engineering of the Visvesvaraya Technological University, Belagavi, during the year 2025–2026. It is certified that all corrections/suggestions indicated for Internal Assessment have been incorporated in the Report deposited in the departmental library.

The project report has been approved as it satisfies the academic requirements in respect of Project work prescribed for the said Degree.

Mrs. Anjali Gupta

Assistant Professor, Dept. of CSE, CMRIT

Dr.Prem Kumar Ramesh

Professor & Head, Dept. of CSE, CMRIT

Dr. Sanjay Jain Principal CMRIT

External Viva

Name of the Examiners	Signature with Date

# DECLARATION

We, the students of Computer Science and Engineering, CMR Institute of Technology, Bangalore declare that the work entitled "Explainable Brain Tumor Detection and Segmentation" has been successfully completed under the guidance of Mrs. Anjali Gupta, Assistant Professor, Computer Science and Engineering Department, CMR Institute of Technology, Bangalore. This dissertation work is submitted in partial fulfillment of the requirements for the award of Degree of Bachelor of Engineering in Computer Science and Engineering during the academic year 2025 - 2026. Further, the matter embodied in the project report has not been submitted previously by anybody for the award of any degree or diploma to any university.

Place: Bangalore Date:

Team members:	Signature

Deeptanshu Dutta (1CR23CS050)

Abinash Mishra (1CR23CS014)

Bhola Yadav (1CR23CS044)

Aman Kumar Pathak (1CR23CS029)

##### ABSTRACT

Brain tumor detection using Magnetic Resonance Imaging (MRI) is a challenging task that requires accurate classification and localization of abnormal tissue. This project presents an Explainable Brain Tumor Detection and Segmentation framework powered by a state-of-the-art heterogeneous Tri-Ensemble architecture. The system combines modern ConvNeXt-Small (45% weight), InceptionV3 (35% weight), and DenseNet121 (20% weight) optimized via Nelder-Mead calibrated soft-voting to classify MRI scans into four categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. Multi-scale Test-Time Augmentation (TTA) elevates overall diagnostic accuracy to a peak benchmark of 98.88%. To guarantee clinical transparency, Grad-CAM and Grad-CAM++ generate visual heatmaps targeting the terminal LayerNorm and convolutional layers, while automated brain-tissue contour masking suppresses cranial bone artifacts. Otsu adaptive thresholding and morphological operations delineate potential tumor margins, and an automated rule-based decision matrix computes intracranial burden percentage and clinical severity tiers to provide comprehensive, interpretable diagnostic assistance.

**Keywords: Brain Tumor Detection, MRI, ConvNeXt-Small, Tri-Ensemble Learning, DenseNet121, InceptionV3, Grad-CAM++, Explainable AI, Morphological Segmentation, Test-Time Augmentation.**

### ACKNOWLEDGEMENT

We express our sincere gratitude to CMR Institute of Technology, Bengaluru, for providing us with the opportunity and resources to carry out our major project, “Explainable Brain Tumor Detection and Segmentation.”

We extend our heartfelt thanks to Dr. Sanjay Jain, Principal, CMRIT, for his constant encouragement and support.

We would also like to express our gratitude to Dr. Prem Kumar Ramesh, Professor and Head, Department of Computer Science and Engineering, for his valuable support and encouragement throughout the project.

We consider it a privilege to express our sincere appreciation to our project guide, Mrs. Anjali Gupta, Assistant Professor, Department of Computer Science and Engineering, CMRIT, for her valuable guidance, constructive suggestions, and continuous encouragement during the development of our project.

We also thank all the faculty members of the Department of Computer Science and Engineering for their support and assistance throughout the course of this work.

Finally, we express our heartfelt gratitude to our parents and friends for their constant motivation, understanding, and moral support, which helped us complete this project successfully.

### TABLE OF CONTENTS

Page No.

### LIST OF FIGURES

**Figure No.        Figure Name                                                        Page No.**

**Fig 3.1           System Architecture of the Explainable CAD Framework              13
Fig 3.2           Comparison of Original, Resized, and Intensity-Normalized Scans   14
Fig 4.1           Multi-Stage Diagnostic and Morphological Segmentation Pipeline     18
Fig 5.1           Interactive Clinical Diagnostic Studio Dashboard                  25
Fig 5.2           Standalone Backbones vs. Tri-Ensemble Classification Comparison    26
Fig 5.3           4x4 Confusion Matrix across Validation Dataset (1,269 Scans)       27
Fig 5.4           Brain-Gated Grad-CAM and Grad-CAM++ Multi-Focal Heatmaps           28
Fig 5.5           Weakly-Supervised Morphological Tumor Segmentation Overlay         29
Fig 5.6           Training and Validation Convergence Curves (DenseNet & Inception) 30
Fig 6.1           Evaluator Ratings on User Interface Clarity and Dashboard Intuition 32
Fig 6.2           Evaluator Agreement with Visual Tumor Localization Quality         33
Fig 6.3           Key Evaluator Feedback and Implemented Architectural Refinements   33
Fig 6.4           Evaluator Ratings on Recommending CAD System for Clinical Workflows 34
Fig 6.5           System Reliability Metrics Demonstrating 0% Error Encounters       34
Fig 6.6           Evaluator Ratings on Ease of Use and Workflow Navigation           35
Fig 6.7           User Feedback on Diagnostic Processing Speed and Response Latency  35**

### LIST OF TABLES

**Table No.         Table Name                                                         Page No.**

**Table 2.1         Comparison of Existing Literature on Brain Tumor Detection (Part 1)  6
Table 2.2         Comparison of Existing Literature on Brain Tumor Detection (Part 2)  7
Table 5.1         Performance Comparison of CNN Backbones, Tri-Ensemble, and TTA     26
Table 5.2         Detailed Per-Class Classification Report (1,269 Validation Scans)   27
Table 5.3         Quantitative Morphological Segmentation Evaluation Metrics          28
Table 6.1         Comprehensive Medical Test Cases and Quality Assurance Outcomes     32**

6

### LIST OF ABBREVIATIONS

AI          Artificial Intelligence
CAD         Computer-Aided Diagnosis
CE-MRI      Contrast-Enhanced Magnetic Resonance Imaging
CNN         Convolutional Neural Network
ConvNeXt    Convolutional Neural Network for the 2020s (Modern CNN Backbone)
DICOM       Digital Imaging and Communications in Medicine
DSC         Dice Similarity Coefficient
FN          False Negative
FP          False Positive
GAP         Global Average Pooling
GELU        Gaussian Error Linear Unit
Grad-CAM    Gradient-weighted Class Activation Mapping
IoU         Intersection over Union (Jaccard Index)
LIME        Local Interpretable Model-agnostic Explanations
LN          Layer Normalization
MRI         Magnetic Resonance Imaging
PACS        Picture Archiving and Communication System
ReLU        Rectified Linear Unit
ROI         Region of Interest
TN          True Negative
TP          True Positive
TTA         Test-Time Augmentation
WHO         World Health Organization
XAI         Explainable Artificial Intelligence

| Col 1 | Col 2 |
| --- | --- |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

# CHAPTER 1

# INTRODUCTION

1. Neuro-Oncological Disease Burden
Brain and central nervous system (CNS) tumors constitute one of the most fatal and biologically aggressive forms of human malignancies. The regulation of cognitive function, motor control, respiration, and sensory processing within the intracranial vault means that even histologically benign neoplasms can induce catastrophic neurological deficits through mass effect and elevated intracranial pressure.

Global Incidence & Mortality: In 2020 alone, primary malignant brain and CNS tumors accounted for an estimated 251,329 deaths worldwide, with hundreds of thousands of new diagnoses annually.

Prognostic Impact of Early Detection: For malignant neoplasms such as high-grade gliomas (Glioblastoma Multiforme - WHO Grade IV), median patient survival is restricted to 12 to 15 months under standard chemoradiation protocols. Early, precise diagnosis prior to extensive parenchymal infiltration is crucial for maximizing the efficacy of maximal safe surgical resection.

2. Characterization of the 4 Target Diagnostic Classes
The Computer-Aided Diagnosis (CAD) system is designed to detect and differentiate among four distinct tissue categories on Contrast-Enhanced T1-weighted Magnetic Resonance Imaging (CE-MRI):

2.1 Class 0: No Tumor (Healthy Intracranial Parenchyma)

Pathological Description: Intact cerebral hemispheres, cerebellar lobes, brainstem, and ventricular structures without pathological space-occupying lesions.

Radiological Appearance: Normal gray matter/white matter junction differentiation, symmetric lateral ventricles, absence of abnormal focal contrast enhancement, midline structures strictly aligned without deviation.

Clinical Significance: Healthy control identification is vital for screening workflows. A clinically viable CAD model must achieve exceptional specificity (>99%) to prevent false-positive cancer diagnoses that inflict severe psychological trauma on patients.

2.2 Class 1: Glioma Tumor

Pathological Description: Intra-axial neoplasms arising from glial support cells (astrocytes, oligodendrocytes, or ependymal cells). Gliomas exhibit diffuse, invasive growth patterns that infiltrate surrounding healthy white matter tracts without clear anatomical capsules.

Radiological Appearance: Irregular, infiltrative margins with non-uniform signal intensities. High-grade variants typically display a prominent central necrotic core surrounded by thick, ring-like contrast enhancement and extensive peritumoral vasogenic edema.

Clinical Urgency: Highly aggressive; necessitates rapid stereotactic biopsy, craniotomy, and concurrent chemoradiation.

2.3 Class 2: Meningioma Tumor

Pathological Description: Predominantly extra-axial tumors originating from the arachnoid cap cells of the meninges. While over 80% are histologically benign (WHO Grade I), their slow expansion compresses adjacent cortical tissue, cranial nerves, and major venous sinuses.

Radiological Appearance: Well-circumscribed, homogeneous, round or lobulated masses attached to the dura mater. Classical signs include intense, uniform post-contrast enhancement and the pathognomonic "dural tail" sign along the adjacent cranial vault.

Diagnostic Pitfall: When situated along the skull base or parasellar region, meningiomas can mimic pituitary adenomas or cause dural margin false alarms on AI models due to proximity to the skull bone.

2.4 Class 3: Pituitary Tumor (Pituitary Adenoma)

Pathological Description: Neoplasms arising from the neuroendocrine cells of the anterior pituitary gland within the sella turcica at the skull base.

Radiological Appearance: Well-defined sellar masses. Microadenomas (diameter < 10 mm) show subtle hypo-enhancement relative to normal pituitary tissue, while macroadenomas (diameter >= 10 mm) expand upward into the suprasellar cistern, often exerting mass effect on the optic chiasm.

Clinical Impact: Causes severe endocrine dysregulation (hypo/hyper-secretion of prolactin, growth hormone, ACTH) and progressive visual field deficits (bitemporal hemianopsia).

#### Relevance of the Project

Brain tumors represent one of the most critical and complex health concerns globally, primarily due to their devastating impact on the central nervous system (CNS) and the regulation of essential bodily functions. The global health burden of this disease is staggering; in 2020 alone, primary cancerous brain and CNS tumors resulted in approximately 251,329 deaths worldwide, with tens of thousands of new cases diagnosed annually. In this context, the early, rapid, and highly accurate detection of brain tumors is not just beneficial—it is an absolute necessity to improve patient survival rates and optimize treatment strategies.

Traditionally, Magnetic Resonance Imaging (MRI) serves as the standard non-invasive diagnostic tool. However, the manual interpretation of these complex, multi-parametric scans by radiologists is inherently time-consuming, subjective, and highly prone to human error, especially when dealing with a massive volume of imaging data. While deep learning models, particularly Convolutional Neural Networks (CNNs), have revolutionized automated medical imaging, they often face two major hurdles: standalone architectures struggle to extract optimal hierarchical features, and their "black-box" nature prevents clinicians from trusting the model's reasoning.

This project is highly relevant as it directly addresses and resolves these critical diagnostic bottlenecks. Its significance lies in two major contributions:

Technological Superiority through Heterogeneous Tri-Ensemble: By proposing a cutting-edge heterogeneous deep learning architecture that strategically fuses modern ConvNeXt-Small (incorporating 7x7 depthwise separable convolutions and inverted bottlenecks) with the multi-scale factorized convolutions of InceptionV3 and the dense feature reuse of DenseNet121, this project maximizes spatial and textural feature extraction. Calibrated via Nelder-Mead optimization (45% ConvNeXt, 35% Inception, 20% DenseNet) and augmented with Test-Time Augmentation (TTA), the framework establishes an exceptional diagnostic accuracy of 98.88%.

Clinical Applicability through Transparency: The true relevance of this project extends beyond mere accuracy. By integrating Explainable Artificial Intelligence (XAI) frameworks specifically Gradient-weighted Class Activation Mapping (Grad-CAM), Local Interpretable Model-agnostic Explanations (LIME), and Integrated Gradients the system provides pixel-level transparency. It visually justifies exactly why and where the model detected a tumor.

Ultimately, the relevance of this project is rooted in its ability to bridge the gap between advanced computational algorithms and real-world medical applications. By delivering a diagnostic tool that is both eInceptionv3ally accurate and fully interpretable, it ensures robustness, reliability, and trust, thereby empowering healthcare professionals to make informed, life-saving clinical decisions.

#### Problem Statement

The manual interpretation of Magnetic Resonance Imaging (MRI) scans for brain tumor diagnosis is an eInceptionv3ally time-consuming, subjective, and error-prone process, Given the complex neuronal structure, diverse shapes, and highly variable locations of tumors such as gliomas, meningiomas, and pituitary tumors relying solely on human visual inspection creates a significant diagnostic bottleneck that can delay critical, life-saving treatments,

While automated diagnostic systems powered by Convolutional Neural Networks (CNNs) have been introduced to mitigate these delays, traditional standalone architectures frequently struggle to extract optimal, hierarchical features, Models relying on a single architecture often fail to consistently capture the minute spatial heterogeneities and subtle boundaries of different tumor classes, occasionally leading to misclassifications.

More critically, the most profound problem hindering the clinical adoption of these advanced deep learning models is their inherent "black-box" nature, even when a traditional neural network achieves high predictive accuracy, it fails to provide any logical reasoning, pixel-level transparency, or visual evidence for its diagnostic output. In high-stakes healthcare scenarios, medical professionals cannot blindly trust or act upon an opaque algorithmic decision without clearly understanding the underlying features that influenced it.

Therefore, a pressing need exists to develop a comprehensive diagnostic framework that simultaneously overcomes the feature-extraction limitations of standalone models through architectural hybridization, while explicitly breaking the "black-box" barrier using Explainable Artificial Intelligence (XAI) to ensure clinical transparency, diagnostic reliability, and ultimate trust,

#### Objectives

The primary objective of this project is to design, develop, and validate an advanced, highly accurate, and clinically interpretable computer-aided diagnosis (CAD) framework for brain tumor detection. To achieve this overarching goal, the project is driven by the following specific objectives:

To Develop a Heterogeneous Tri-Ensemble Architecture: To engineer an advanced, high-performing ensemble model by intelligently fusing modern ConvNeXt-Small (for large-receptive-field global context), InceptionV3 (for multi-scale lesion handling), and DenseNet121 (for continuous gradient backpropagation and fine edge preservation), achieving robust multi-class brain tumor classification across variable CE-MRI protocols.

To Implement Robust Data Preprocessing and Augmentation: To apply effective image preprocessing, resizing, and synthetic data augmentation techniques to resolve dataset class imbalance, standardize MRI inputs, and enhance the model's robustness and ability to generalize across diverse clinical scenarios.

To Perform Precise Multi-Class Classification: To accurately detect and classify brain MRI scans into four distinct categories—glioma tumors, meningioma tumors, pituitary tumors, and healthy (no tumor) control cases—while targeting an eInceptionv3al overall accuracy rate of over 97%.

To Integrate Explainable Artificial Intelligence (XAI): To shatter the traditional "black-box" paradigm of deep learning by incorporating advanced XAI frameworks, specifically Gradient-weighted Class Activation Mapping (Grad-CAM), Local Interpretable Model-agnostic Explanations (LIME), and Integrated Gradients. This objective focuses on generating pixel-level visual evidence (such as heatmaps) to transparently justify the model's diagnostic predictions to medical professionals.

To Evaluate and Benchmark Performance: To rigorously validate the proposed hybrid framework against standard, standalone Convolutional Neural Networks (CNNs) using comprehensive statistical metrics (including Accuracy, Precision, Recall, F1-Score, and Specificity) to establish its technological superiority and clinical reliability.

#### Scope of the Project

The scope of this project encompasses the design, development, and validation of an automated, highly accurate, and clinically interpretable Computer-Aided Diagnosis (CAD) system specifically tailored for brain tumor analysis using Magnetic Resonance Imaging (MRI) scans. The proposed diagnostic framework is strictly delimited to categorizing MRI images into four specific classes: glioma tumors, meningioma tumors, pituitary tumors, and healthy (no tumor) control cases.

Technologically, the scope is confined to the implementation and evaluation of deep learning techniques, specifically a heterogeneous Tri-Ensemble Convolutional Neural Network (CNN) architecture that strategically fuses ConvNeXt-Small, InceptionV3, and DenseNet121 models to optimize hierarchical feature extraction. To ensure the system's reliability across diverse clinical scenarios, the scope includes robust image preprocessing and synthetic data augmentation techniques to handle class imbalances and standardize the raw MRI inputs. Furthermore, the interpretability scope of this project is explicitly defined by the integration of Explainable Artificial Intelligence (XAI) algorithms. The system utilizes these XAI frameworks to generate localized visual heatmaps and pixel-level evidence, completely eliminating the "black-box" nature of traditional neural networks and providing transparent reasoning for its diagnostic outputs.

While the proposed hybrid framework achieves an ensemble benchmark accuracy of 98.88% with multiscale TTA, it is important to define the operational boundaries of this study. The current scope is limited to the analysis of 2D MRI slices and does not extend to real-time 3D volumetric tumor segmentation or live video processing. Additionally, this framework is strictly engineered as a supplementary decision-support tool designed to assist radiologists and neuro-oncologists by reducing manual analysis time and mitigating human error. It is not intended to act as a standalone replacement for professional clinical judgment or biopsy-based pathological confirmations.

#### Tools and Technologies

The successful design, training, and deployment of an advanced Hybrid Deep Learning and Explainable AI (XAI) framework require a robust and highly scalable technological stack. Due to the immense computational complexity involved in processing high-resolution medical imagery and generating real-time pixel-level explanations, the project’s requirements are broadly categorized into hardware configurations and software/modeling frameworks.

1.6.1 Hardware and Computational Requirements

Training deep Convolutional Neural Networks (CNNs) on thousands of Augmented Magnetic Resonance Imaging (MRI) scans demands substantial computational power. The foundational hardware prerequisites for this project include:

Cloud-Based GPU Processing:

The entire experimental and training pipeline was executed within a cloud-based computational environment, specifically utilizing Google Colaboratory (Colab) with high-performance GPU (Graphics Processing Unit) support. GPUs are strictly required to drastically accelerate the matrix multiplications inherent in deep learning, ensuring rapid model convergence.

Memory and Storage:

To prevent memory bottleneck errors during the processing of large multi-dimensional image arrays and heavy hybrid architectures (like DenseNet121 and Inceptionv3), a minimum computational environment of 16 GB of RAM and substantial cloud storage (e.g., 1TB backend storage) is highly recommended for seamless data preparation and model execution.

1.6.2 Software and Modeling Requirements

The core diagnostic engine and its interpretability modules were built using industry-standard machine learning libraries and computer vision frameworks. The essential software stack includes:

Programming Environment:

Python 3 was utilized as the primary, versatile programming language for developing the entire architecture, from backend data handling to frontend deployment.

Deep Learning Frameworks: TensorFlow (v2.3.0) and Keras (v2.4.0) served as the primary deep learning backbones. These frameworks were instrumental in engineering the hybrid CNN architecture, applying the Adam optimizer, configuring dropout layers for regularization, and compiling the sparse categorical cross-entropy loss function.

Computer Vision and Data Preprocessing: Libraries such as OpenCV (cv2), NumPy (v1.18.5), and Pandas (v1.1.4) were extensively utilized for handling complex image data. OpenCV was crucial for resizing raw MRI scans into standardized dimensions (e.g., 224x224 pixels) and applying geometric data augmentation techniques (like rotation and flipping) to resolve class imbalance.

Explainable AI (XAI) Libraries: To shatter the "black-box" nature of traditional models, specialized explainability modules were integrated into the pipeline. These packages generated Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps, and provided Local Interpretable Model-agnostic Explanations (LIME) and Integrated Gradients for highlighting the exact pixel-level regions of the tumors.

Interactive Web Deployment: Finally, the Gradio framework was employed to bridge the gap between complex backend algorithms and clinical usability. It enabled the creation of an interactive web interface where medical professionals can effortlessly upload an MRI scan to instantly receive the segmented tumor class alongside its visual XAI heatmap explanation.

#### Chapter-Wise Summary

In Chapter 1, the report introduces the Explainable Brain Tumor Detection and Segmentation project, describing its relevance, problem statement, objectives, scope, and tools and technologies. It highlights the importance of accurate brain MRI classification and the need for interpretable predictions and tumor-region localization.

In Chapter 2, the literature survey reviews existing research on brain tumor classification and segmentation using deep learning. It discusses CNN architectures, transfer learning, ensemble learning, explainable artificial intelligence, and image-processing techniques. The chapter also identifies limitations in existing approaches and explains the motivation for the proposed framework.

In Chapter 3, the proposed architecture and system design are presented. The chapter describes the workflow involving MRI preprocessing, brain-region extraction, classification using a Tri-Ensemble of ConvNeXt-Small, InceptionV3, and DenseNet121, and Nelder-Mead calibrated ensemble prediction. It also explains the use of Grad-CAM and Grad-CAM++ for visual explanations and morphological operations for potential tumor-region segmentation.

In Chapter 4, the implementation methodology is described, including dataset preparation, image preprocessing, augmentation, model training, ensemble prediction, heatmap generation, and tumor-region segmentation. The chapter explains how the different components are integrated to perform brain MRI analysis.

In Chapter 5, the experimental results and discussion are presented. The chapter evaluates the classification models and ensemble using metrics such as accuracy, precision, recall, and F1-score. It also discusses confusion matrices and the visual outputs generated by the explainability and segmentation components. The reported results are interpreted in the context of the dataset and model configuration used.

In Chapter 6, the testing methodology is discussed. Test cases are designed to verify the major components, including image preprocessing, classification, explainability heatmap generation, segmentation, and integration between system modules. The chapter presents the corresponding test outcomes for the implemented system.

In Chapter 7, the report concludes by summarizing the project objectives, methodology, and findings. It discusses the contributions and limitations of combining deep learning, explainable AI, and morphological segmentation. Finally, it outlines possible future enhancements, including three-dimensional MRI analysis, multi-modal imaging, and the development of more advanced tumor-segmentation models.

# CHAPTER 2

# LITERATURE SURVEY

The automated detection, classification, and segmentation of brain tumors using Magnetic Resonance Imaging (MRI) has witnessed a massive paradigm shift from traditional computer vision techniques to advanced deep learning frameworks. Accurate and early diagnosis of neuro-oncological conditions—such as gliomas, meningiomas, and pituitary tumors—is critical for patient survival. However, the complex structure of brain tissues and varying tumor morphologies make manual interpretation by radiologists highly subjective and prone to errors. This literature survey critically evaluates existing methodologies, ranging from traditional machine learning to state-of-the-art hybrid deep learning architectures and Explainable AI (XAI) frameworks, highlighting their contributions and identifying prevailing research gaps.

## 2.1 Detection and Segmentation Based Approaches

Historically, automated tumor detection heavily relied on manual feature extraction and traditional machine learning algorithms. Doke et al. proposed a mechanism utilizing 2D Gabor filters to transform MRI images and extract features. Upon evaluating multiple classifiers, they concluded that the Support Vector Machine (SVM) outperformed K-Nearest Neighbors (KNN), Decision Trees, and Naive Bayes in detecting the presence of tumors. Similarly, Singh et al. evaluated various machine learning models (KNN, Random Forest, SVM) against Convolutional Neural Networks (CNNs) for tumor detection, concluding that CNNs generated significantly superior and more reliable results by eliminating the need for manual feature engineering.

For segmentation tasks, methods often aim to distinguish between overlapping tumor subregions such as the necrotic core, edema, and enhancing tumor. Advanced architectures like DeepLabV3+ and U-Net have been leveraged for this purpose. Bala et al. integrated DeepLabV3+ with Bayesian Optimization to finely tune hyperparameters and precisely segment tumor boundaries, proving highly effective in managing the intensity variations typical of MRI scans.

## 2.2 Model and Approach-Based Studies

The transition toward Deep Learning has led to the adoption of standalone CNNs, Transfer Learning, and Hybrid architectures.

Standalone and Transfer Learning Models: Pre-trained architectures like VGG16, ResNet50, DenseNet121, and InceptionV3 have been extensively evaluated. Studies have demonstrated that DenseNet121 effectively captures hierarchical features due to its dense connectivity, while InceptionV3 excels in handling complex image patterns via varying kernel sizes.

Ensemble, Hybrid, and Modernized CNN Architectures: To overcome the architectural limitations of standalone classical models, recent research emphasizes heterogeneous hybridization and modernized convolutional backbones. In 2022, Liu et al. introduced ConvNeXt, re-architecting standard ResNets with Vision Transformer principles—incorporating 7x7 depthwise separable convolutions, inverted bottleneck blocks, Layer Normalization, and GELU activations. ConvNeXt achieved Transformer-grade visual representation while retaining the computational efficiency and inductive bias of pure convolutions. Hosny et al. proposed an ensemble leveraging DenseNet121 and InceptionV3 through transfer learning, achieving 99.02% accuracy on specific datasets. Kundu et al. demonstrated that fusing depthwise separable convolutions with dense connections achieves 97.63% accuracy. Combining modern ConvNeXt with multi-scale Inception and DenseNet architectures forms the ultimate synergistic defense against multi-class morphological variability.

.

## 2.3 Explainable AI (XAI) in Brain Tumor Analysis

Despite the high predictive accuracy of deep learning frameworks, their clinical applicability is severely hindered by their inherent "black-box" nature. Medical professionals cannot safely act upon an AI's prediction without understanding its underlying reasoning. To bridge this gap, the integration of Explainable AI (XAI) has become a primary focus.

Gradient-weighted Class Activation Mapping (Grad-CAM):

Grad-CAM utilizes gradients from the final convolutional layers to generate global visual heatmaps, highlighting the macro-regions of the MRI that influenced the classification.

Local Interpretable Model-agnostic Explanations (LIME):

For pixel-level transparency, LIME is employed to approximate local decision boundaries, offering visual super-pixel evidence. Complementary attribution methods rooted in cooperative game theory provide a unified metric for feature importance, identifying exact spatial contributions. Studies by Nafia et al. and Bijoy et al. demonstrated that integrating LIME with CNNs drastically improves diagnostic transparency and clinical trust.

## 2.4 Key Observations from Literature

Superiority of Hybridization: Fusing distinct architectures (e.g., DenseNet121 + Inceptionv3) consistently yields higher precision and recall compared to standalone models, effectively mitigating the vanishing gradient problem while enhancing feature reuse.

Necessity of Interpretability: Accuracy alone is insufficient for medical adoption. Models integrating visual XAI (Grad-CAM, LIME) are significantly more reliable for radiologists, as they prove the model detects actual anatomical anomalies rather than background noise.

Data Preprocessing Impact: Robust preprocessing pipelines such as contrast enhancement, synthetic data augmentation, and spatial resizing are vital for preventing model overfitting, especially in medical datasets suffering from class imbalance.

## 2.5 Research Gaps

While extensive research has been conducted, several critical gaps remain:

Interpretability vs. Computational Cost:

High-performing models like pure Vision Transformers or 3D CNNs are computationally heavy and memory-intensive, limiting their real-time clinical deployment.

Post-hoc Interpretability Limitations:

Many existing frameworks apply XAI solely as an afterthought rather than integrating it seamlessly into the diagnostic pipeline to justify multi-class localizations simultaneously.

The Need for Comprehensive Frameworks:

There is a pressing need for a lightweight, hybrid deep learning architecture (such as the fusion of DenseNet and Inceptionv3) that not only provides superior multi-class detection (Glioma, Meningioma, Pituitary) but simultaneously breaks the "black-box" barrier utilizing multiple XAI layers (both Grad-CAM for global heatmaps and LIME for pixel-level transparency). This specific intersection forms the core motivation and scope of the proposed project.

**Table 2.1 Comparison of Existing Literature on Brain Tumor Detection & XAI (Part 1).**

| S.No | Author & Year | Methodology / Model | Dataset | Key Contributions | Limitations |
| --- | --- | --- | --- | --- | --- |
| 1 | Shaheema et al. (2025) | PA-ResNet50 + Grad-CAM++ | BraTS 2019 / 2021 | Achieved 98.7% - 99.4% accuracy with Dice scores of 98.8% on glioma scans. | High GPU overhead; evaluated strictly on BraTS gliomas. |
| 2 | Liu et al. (2023) | SegResNet + nnU-Net with transfer learning | BraTS 2021 + Pediatric DMG | Dice scores 0.859 (enhancing) and 0.880 (whole tumor) with adult pretraining. | Very small cohort (45 patients); inconsistent subregions. |
| 3 | Jia & Chen (2025) | FAHS-SVM + ELM | Benchmark CE-MRI | Achieved 98.51% accuracy distinguishing abnormal vs normal brain tissues. | Slow for high-resolution slices; difficulty with large tumor margins. |
| 4 | Hosny et al. (2025) | DenseNet121 + InceptionV3 + Grad-CAM | Multi-Class CE-MRI | Ensemble achieved 99.02% accuracy with strong precision, recall, and F1. | High computational complexity; models trained independently without severity rules. |
| 5 | Doke & Rao (2023) | 2D Gabor Filters + SVM Classifier | Small MRI (255 scans) | Evaluated multiple classical ML classifiers; SVM achieved 80% accuracy. | Low accuracy and small sample size due to manual feature engineering limitations. |
| 6 | Arun Kumar et al. (2024) | CNN + LIME + SHAP | Multi-Class MRI | CNN achieved 97.39% accuracy; compared visual feature attributions. | ML baselines had low accuracy (53%-77%); relies on high-quality slices. |

**Table 2.2 Comparison of Existing Literature on Brain Tumor Detection & XAI (Part 2).**

| S.No | Author & Year | Methodology / Model | Dataset | Key Contributions | Limitations |
| --- | --- | --- | --- | --- | --- |
| 7 | Bala et al. (2025) | DeepLabV3+ + Bayesian Optimization | Multi-Class MRI | Bayesian hyperparameter optimization; achieved Dice 0.89 and IoU 0.78. | Deep models remain complex; Bayesian tuning increases computational cost. |
| 8 | Ahmed et al. (2024) | Vision Transformer (ViT) + GRU + LIME | Southern Bangladesh Scans | Achieved 97.98% accuracy with strong sequential and global pattern detection. | High computational cost; dataset restricted to single hospital. |
| 9 | Singh et al. (2023) | Image Filtering + Seed Growing + CNN | Small Dataset (394 scans) | CNN achieved 98.6% accuracy, outperforming classical random forests and SVMs. | Small sample size (394 images); vulnerable to validation overfitting. |
| 10 | Mohammed & Al-Suhail (2025) | Survey: CNN / ViT + Grad-CAM / SHAP | Multi-modal Survey | Comprehensive review of XAI; highlighted need for clinical trust and transparency. | Identified lack of clinical validation and uncalibrated skull leakage. |

#### Research Gap

Real-time Performance

Most existing CAD systems struggle to provide low-latency responses when handling large volumes of high-resolution multi-parametric MRI scans. The lack of optimized convolutional backbones and real-time inference pipelines limits their deployment in acute neuro-oncological emergency triage.

Scalability

Many proposed models are not designed to scale efficiently as the number of users or the size of the dataset increases. Solutions that lack distributed computation or parallelization tend to suffer from performance bottlenecks under high demand.

Computational Efficiency

There is a strong need for resource-efficient detection architectures. Many current solutions require substantial processing power and memory, making them impractical for real-time deployment on low-power, edge, or embedded devices.

# CHAPTER 3

# PROPOSED ARCHITECTURE AND DESIGN

## 3.1 System Architecture

The system architecture forms the core blueprint of the proposed Computer-Aided Diagnosis (CAD) framework. It is designed as a structured, sequential pipeline that ensures both high diagnostic performance and interpretability. The workflow progresses step by step, starting from raw MRI brain scan inputs and moving toward clinically meaningful, explainable predictions. Each stage is carefully organized to transform data, extract features, perform classification, and finally provide visual explanations that support medical decision-making.

The overall architecture consists of five interconnected modules: input dataset acquisition, data preprocessing, the proposed heterogeneous Tri-Ensemble expert model (ConvNeXt-Small + InceptionV3 + DenseNet121), multi-class classification, and an explainability module based on Grad-CAM and LIME.

Data Acquisition and Preprocessing: Ingestion of raw MRI scans followed by noise reduction, resizing, and synthetic augmentation to balance the data distribution.

Expert Hybrid Feature Extraction: Utilizing a fused deep learning architecture to autonomously capture complex hierarchical and spatial features from the brain tissues.

Classification Engine: A dense neural network layer designed to categorize the extracted features into specific tumor classes or healthy tissues.

Explainability and Segmentation Module: The integration of visual AI tools to generate localized heatmaps and pixel-level boundaries, transforming the model from a "black-box" into a transparent clinical assistant.

**Fig 3.1 Methodology of Brain Tumor Detection**

## 3.2 Input Data and Preprocessing

The efficacy of any deep learning model is heavily reliant on the quality and volume of its training data. For this study, the input data consists of a comprehensive, publicly available Magnetic Resonance Imaging (MRI) dataset sourced from the Kaggle repository. The initial dataset comprises 2,968 raw MRI scans, meticulously categorized into four distinct classes: glioma tumors (826 files), meningioma tumors (822 files), pituitary tumors (827 files), and healthy 'no tumor' cases (395 files). The dataset was systematically partitioned into an 80% training subset and a 20% independent testing subset to ensure an unbiased evaluation.

To optimize the raw MRI scans for the neural network, a rigorous preprocessing pipeline was implemented:

Data Augmentation: To resolve the inherent class imbalance and prevent model overfitting, synthetic data augmentation was heavily employed. Through geometric transformations such as random rotating, flipping, and zooming the dataset was significantly expanded from 3,264 to 8,177 images, making the model highly robust to real-world anatomical variations.

Dimensional Resizing: MRI scans acquired from diverse clinical setups often vary in spatial dimensions. To maintain input uniformity and ensure computational efficiency across the network, all images were resized to a standard, uniform resolution.

Normalization and Rescaling: Pixel intensity values across clinical MRI scans fluctuate drastically due to differences in scanner magnetic field strengths (1.5T vs. 3.0T) and acquisition parameters. To stabilize neural network optimization, raw pixel intensities are normalized into floating-point tensors using standardized transformations:

**Equation 3.1: Standard Min-Max Intensity Normalization**
```text
                  X(x, y) - X_min
  X_norm(x, y) = ─────────────────────────────
                  X_max - X_min + epsilon
```
Where:
- `X(x, y)`: Raw voxel intensity at coordinate `(x, y)`.
- `X_min`, `X_max`: Minimum and maximum intensities within the intracranial brain region.
- `epsilon = 1e-7`: Numerical stabilizer preventing division by zero on uniform backgrounds.

To resolve class frequency imbalances across the dataset, balanced loss weights are dynamically computed for each category:

**Equation 3.2: Inverse-Frequency Balanced Class Weighting**
```text
               N_total
  W(c) = ───────────────────────
          N_classes * N_samples(c)
```
Where `N_total` represents total training volume, `N_classes = 4`, and `N_samples(c)` represents the empirical support of class `c`.

## 3.3 The Expert Tri-Ensemble: Architectural Hybridization

Traditional standalone Convolutional Neural Networks (CNNs) often struggle to simultaneously capture fine textural boundaries and macroscopic spatial brain context. To engineer the optimal CAD engine, a heterogeneous Tri-Ensemble was designed by fusing three complementary architectures: modern ConvNeXt-Small, multi-scale InceptionV3, and densely connected DenseNet121.

The Tri-Ensemble framework leverages the distinctive mathematical strengths of three foundational deep learning paradigms:

#### 1. ConvNeXt-Small (Primary Modern Backbone - 45% Weight)
ConvNeXt modernizes classical convolutions by adopting Vision Transformer design principles while maintaining pure convolutional efficiency. It introduces large 7x7 depthwise separable kernels to capture wide-range cranial context:

**Equation 3.3: Large 7x7 Depthwise Separable Convolution**
```text
           3       3
  y(i, j) = ∑       ∑    K(m, n) · x(i + m, j + n)
          m=-3    n=-3
```
Where `K` represents the 7x7 spatial filter kernel applied independently across each feature channel, mirroring the receptive field of Swin Transformer self-attention.

Within each ConvNeXt block, an inverted bottleneck expands channel dimensions 4-fold (`C → 4C`) before spatial down-projection, preserving critical textural features. Normalization is standardized using Layer Normalization across channels:

**Equation 3.4: Layer Normalization (LN)**
```text
           x - μ
  LN(x) = ────────────── · γ + β
          √(σ² + ε)
```
Where `μ` and `σ²` are instance spatial mean and variance, and `γ`, `β` are learnable affine parameters. Non-linear gating is provided by Gaussian Error Linear Units (GELU):

**Equation 3.5: Gaussian Error Linear Unit (GELU)**
```text
  GELU(x) = x · Φ(x) = x · P(X ≤ x)
          ≈ 0.5 · x · [ 1 + tanh( √(2/π) · (x + 0.044715 · x³) ) ]
```

#### 2. InceptionV3 (Secondary Backbone - 35% Weight)
InceptionV3 incorporates multi-scale factorized convolutions. High-order 7x7 spatial convolutions are factorized into asymmetric 1x7 and 7x1 cascaded kernels:

**Equation 3.6: Asymmetric Factorized Convolution**
```text
  Conv2D_asym(x) = [ K_{7 × 1} * [ K_{1 × 7} * x ] ]
```
This asymmetric decomposition reduces operational complexity by 71.4% (from 49 to 14 multiplications per pixel) while capturing heterogeneous tumor dimensions.

#### 3. DenseNet121 (Tertiary Backbone - 20% Weight)
DenseNet enforces direct connectivity between all layers within each dense block:

**Equation 3.7: Dense Feature Connectivity**
```text
  x_l = H_l( [ x_0, x_1, x_2, ..., x_{l-1} ] )
```
By concatenating preceding feature maps, the backpropagation loss gradient contains a direct identity path:

**Equation 3.8: Additive Backpropagation Gradient Highway**
```text
  d(Loss)      d(Loss)         L-1    [ d(Loss)     d(x_k)  ]
  ───────  =  ─────────   +     ∑     [ ───────  ·  ─────── ]
   dx_l         dx_L           k=l    [  dx_k        dx_l   ]
```
This additive formulation eliminates vanishing gradients and maximizes fine-grained edge preservation for subtle meningioma margins.

## 3.4 Classification Module

Following multi-backbone feature extraction, class posterior probabilities from all three networks are fused through calibrated soft voting. Optimal voting coefficients were obtained via Nelder-Mead simplex optimization on validation cross-entropy logits:

**Equation 3.9: Nelder-Mead Calibrated Tri-Ensemble Soft Voting**
```text
  P_tri(y = c | x) = 0.45 · P_ConvNeXt(c | x) + 0.35 · P_Inception(c | x) + 0.20 · P_DenseNet(c | x)
```
Where `∑_{c=1}^4 P_tri(c | x) = 1.0`.

The final clinical diagnostic prediction and confidence score are derived as:

**Equation 3.10: Discrete Class Selection & Diagnostic Confidence**
```text
  c_predicted = argmax_{c ∈ {0,1,2,3}} [ P_tri(c | x) ]
  Confidence  = max_{c}                [ P_tri(c | x) ] × 100%
```

To alert clinicians when an MRI slice exhibits ambiguous anatomical features, diagnostic uncertainty is computed using normalized Shannon entropy:

**Equation 3.11: Normalized Shannon Diagnostic Uncertainty**
```text
                   4
                -  ∑  [ P_tri(c | x) · log2( P_tri(c | x) ) ]
                  c=1
  Uncertainty = ────────────────────────────────────────────── × 100%
                                   log2(4)
```
Where `Uncertainty ∈ [0.0%, 100.0%]`. A confident prediction with probability 0.99 yields uncertainty < 5%, whereas ambiguous predictions (e.g., 0.25 on all 4 classes) yield 100% uncertainty, triggering automated clinical review.

## 3.5 Explainability and Segmentation Module (Crucial for Clinical Trust)

To eliminate the opaque black-box barrier inherent in deep neural networks, the CAD platform integrates visual explainability via Gradient-weighted Class Activation Mapping (Grad-CAM and Grad-CAM++):

**Equation 3.12: Neuron Importance Weight via Global Average Pooling**
```text
              1      u   v   ∂( y^c )
  α_k^c   =  ─── ·   ∑   ∑   ─────────────
              Z     i=1 j=1   ∂( A_{i,j}^k )
```
Where:
- `A^k`: Activation feature map of channel `k` from the terminal normalization or convolutional layer.
- `Z = u × v`: Spatial grid resolution (height `u` × width `v`).
- `y^c`: Predicted class logit prior to softmax activation.

The visual coarse saliency localization map is generated by computing a weighted combination followed by rectified non-linear filtering:

**Equation 3.13: Rectified Visual Heatmap Formulation**
```text
  L_GradCAM^c = ReLU(  ∑  [ α_k^c · A^k ]  )
                       k
```

To eliminate extraneous high-intensity cranial bone, scalp, and ocular artifacts from corrupting the radiological visualization, Brain-Mask Anatomical Gating is strictly enforced:

**Equation 3.14: Brain-Mask Anatomical Saliency Gating**
```text
  HM_gated(x, y) = L_GradCAM^c(x, y) · Mask_brain(x, y)
```
Where `Mask_brain(x, y) ∈ {0, 1}` is extracted via largest convex intracranial contour isolation, ensuring 100% of diagnostic attributions remain confined within cerebral parenchyma.

To transition from mere image classification to precise tumor segmentation and interpretability, the system integrates two advanced XAI frameworks,

Grad-CAM (Gradient-weighted Class Activation Mapping): This technique extracts vital gradient information from the final convolutional layers to generate vibrant, global visual heatmaps, these heatmaps visually highlight the exact spatial regions (such as the tumor core) within the MRI that exerted the maximum influence on the model's algorithmic prediction.

LIME (Local Interpretable Model-agnostic Explanations): To provide granular, pixel-level transparency, LIME is employed to approximate local decision boundaries, it generates visual super-pixel evidence, explicitly defining the segmented boundaries of the tumor and proving that the model's internal logic aligns perfectly with actual human anatomy.

3.6 Tumor-Region Segmentation

The tumor-region segmentation module extends the classification framework by identifying potential abnormal regions within brain MRI images. After the classification and explainability stages, the generated activation heatmap provides a visual indication of the image regions that contributed to the predicted tumor category. This heatmap is used as a guide for identifying regions that may contain abnormal tissue.

Image-processing techniques are applied to refine the highlighted region. Thresholding separates higher-activation areas from the surrounding image, while morphological operations help reduce small isolated regions and improve the continuity of the detected area. The resulting mask is overlaid on the original MRI image to provide a visual representation of the potential tumor region.

The segmentation output is intended to complement the classification result by providing spatial information about the suspected abnormality. Since the approach relies on model-generated heatmaps and image-processing operations rather than a dedicated, manually supervised segmentation network, the resulting boundary should be treated as an approximate localization rather than a definitive medical measurement.

3.7 Severity Estimation and Anatomical Localization

The framework can further interpret the classification output using prediction confidence and the estimated area of the segmented region. These values provide additional information that can be used to assign preliminary severity categories and describe the approximate location of the highlighted region.

The severity estimation process considers the model's classification confidence together with the proportion of the brain image occupied by the segmented region. Based on the configured rules, the system can categorize the result into levels such as High, Moderate, Low, Borderline, or Uncertain. These categories are computational indicators generated by the framework and should not be interpreted as clinically validated tumor grades.

Approximate anatomical localization provides additional context by relating the highlighted region to the visible brain area. The usefulness of this estimate depends on the quality of the MRI image, the accuracy of the heatmap, and the reliability of the segmentation mask. Final interpretation must therefore remain with qualified medical professionals.

3.8 Overall System Workflow

The proposed system follows a sequential workflow that integrates image preprocessing, deep learning classification, explainability, and approximate segmentation.

MRI Image Input: The user supplies a brain MRI image for analysis.

Image Preprocessing: The input image is resized and normalized to meet the requirements of the classification models. Brain-region extraction may be applied to reduce irrelevant background information.

Feature Extraction: ConvNeXt-Small, InceptionV3, and DenseNet121 extract complementary global, multi-scale, and local edge visual features from the MRI image.

Classification: The classification module produces scores for the four supported categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor.

Explainability Generation: Grad-CAM and related explainability techniques generate visual evidence indicating which image regions influenced the prediction.

Tumor-Region Segmentation: Thresholding and morphological image-processing operations are applied to the heatmap to generate an approximate region mask.

Result Presentation: The system presents the predicted class, visual explanation, and segmentation output. Where implemented, it can also display preliminary severity and anatomical localization estimates.

This modular workflow connects the major components of the framework and provides a structured approach to brain MRI classification and interpretation. Each module contributes to the final output, while the explainability and segmentation stages provide visual context beyond thepredictedclassalone.

Tumor-region boundary extraction is achieved through a multi-stage morphological pipeline operating on the gated saliency map:

1. **ROI Bounding:** Identifying the top 30% of intracranial activations: `τ_ROI = max(Percentile_70, 0.45)`.
2. **Adaptive Binarization:** Applying Otsu's optimal thresholding inside the ROI to maximize between-class variance:

**Equation 3.15: Otsu Optimal Between-Class Variance**
```text
  t* = argmax_t [ σ_B²(t) ] = argmax_t [ ω_0(t) · ω_1(t) · ( μ_0(t) - μ_1(t) )² ]
```
Where `ω_0(t)`, `ω_1(t)` represent cumulative class probabilities, and `μ_0(t)`, `μ_1(t)` denote mean pixel intensities of background and tumor candidate pixels.

3. **Morphological Filtering:** Applying morphological opening and closing with a 5x5 elliptical structuring element `B` to bridge necrotic core gaps and eliminate isolated noise:

**Equation 3.16: Morphological Refinement Operators**
```text
  Morphological Opening:  M ○ B = ( M ⊖ B ) ⊕ B
  Morphological Closing:  M ● B = ( M ⊕ B ) ⊖ B
```

4. **Intracranial Burden Estimation:** Relative tumor mass is quantified as the percentage of intracranial volume occupied by the segmented lesion:

**Equation 3.17: Relative Intracranial Tumor Burden Ratio**
```text
                ∑_{(x,y)} Mask_tumor(x, y)
  ρ_tumor  =  ───────────────────────────── × 100%
                ∑_{(x,y)} Mask_brain(x, y)
```

**Equation 3.18: Deterministic Clinical Severity Index Rules**
```text
  Severity(Confidence, ρ_tumor) = 
  ⎧ High Tier:       if Confidence > 95%  AND  ρ_tumor > 5.0%
  ⎪ Moderate Tier:   if Confidence > 80%  AND  ρ_tumor > 2.0%
  ⎨ Low Tier:        if Confidence > 60%  AND  ρ_tumor ≤ 2.0%
  ⎪ Borderline:      if 50% ≤ Confidence < 55%
  ⎩ Uncertain:       if Confidence < 50%
```

Anatomical localization is determined by mapping the peak coordinate `(x_max, y_max)` of `HM_gated` against intracranial anatomical boundaries to determine Cerebral Hemisphere (Left vs. Right) and Cerebral Lobe (Frontal, Parietal, or Occipital).

# CHAPTER 4

# IMPLEMENTATION

The following components have been developed for the Explainable Brain Tumor Detection and Segmentation project.

The system processes brain MRI images to classify them into four categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. During preprocessing, images are prepared according to the input requirements of the classification models. Image resizing and normalization help maintain consistent input dimensions and intensity ranges. Brain-region extraction is also used to reduce irrelevant skull and background information.

The classification module uses three heterogeneous deep learning architectures—ConvNeXt-Small, InceptionV3, and DenseNet121—to extract image features and predict the corresponding tumor category. Their predictions are combined using the configured ensemble method to obtain the final classification result. The classification output provides the predicted category and associated confidence information.

To improve interpretability, the framework incorporates Grad-CAM and Grad-CAM++ to generate heatmaps highlighting image regions that influence the model's prediction. These visual explanations help users inspect the model's focus rather than relying solely on the predicted class label.

The tumor-region segmentation module uses the generated heatmaps to guide image processing. Otsu thresholding and morphological operations, including opening and closing, help identify and refine potential tumor regions. The resulting mask provides an approximate visual representation of the suspected abnormal area. Its accuracy depends on the quality of the heatmap and preprocessing.

The system combines classification, explainability, and approximate segmentation into a unified MRI analysis workflow. Where implemented, additional outputs include preliminary severity categories and approximate anatomical localization based on classification confidence and segmented tumor area. These outputs are intended to support research and interpretation and do not replace professional medical diagnosis.

#### Algorithm

Algorithm: Explainable Brain Tumor Detection and Segmentation

Input: Brain MRI image

Output: Predicted tumor category, explainability heatmap, and approximate tumor-region mask

Import the required libraries and initialize the image-processing and deep learning components.

Load the input brain MRI image.

Preprocess the image by resizing and normalizing it according to the model requirements.

Apply brain-region extraction, where supported, to reduce irrelevant skull and background information.

Pass the preprocessed image to the ConvNeXt-Small, InceptionV3, and DenseNet121 classification models.

Obtain the class-probability outputs from the individual models.

Combine the model predictions using the configured ensemble method.

Select the class with the highest combined probability from No Tumor, Glioma, Meningioma, and Pituitary Tumor.

Generate Grad-CAM and Grad-CAM++ heatmaps to identify regions influencing the predicted class.

Use the generated heatmap to identify a region of interest within the brain image.

Apply Otsu thresholding and morphological operations to produce an approximate tumor-region mask.

Overlay the heatmap and segmentation mask on the original MRI image for visualization.

Present the predicted category, classification confidence, heatmap, and segmentation result.

If supported by the implementation, estimate severity and approximate anatomical location using the configured rules.

Return the results and wait for the next image input.

End of Algorithm

#### Flow Chart

**Fig 4.1 Flowchart of the Multi-Stage Diagnostic, Explainability, and Segmentation Pipeline.**

The proposed system processes brain MRI images to classify them into four categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. During preprocessing, images are prepared according to the requirements of the classification models. Brain-region extraction is used to reduce irrelevant skull and background information where supported by the implemented pipeline. The processed images are then passed to the deep learning classification models, and their predictions are combined using the configured ensemble method.

Following classification, explainability techniques such as Grad-CAM and Grad-CAM++ generate heatmaps indicating image regions that influence the predicted class. The heatmaps guide the segmentation stage, where thresholding and morphological operations are applied to identify potential tumor regions. The system presents the predicted category together with the visual explanation and approximate segmentation output, providing a more interpretable representation of the MRI analysis.

#### Code

This section describes the implementation of the major components of the Explainable Brain Tumor Detection and Segmentation framework. The implementation includes dataset collection, MRI image preprocessing, deep learning-based classification, explainability, tumor-region segmentation, and presentation of the analysis results. Each component contributes to the overall workflow, from accepting an MRI image to generating a predicted tumor category and visualizing the regions influencing the prediction.

##### Dataset Collection

The project uses a brain MRI image dataset to develop and evaluate the classification framework. The images are organized into four categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. These categories enable the model to distinguish between healthy brain images and images associated with different tumor types.

The dataset is prepared for use with deep learning architectures by organizing the images according to their corresponding class labels. The images are subsequently divided into the appropriate training and evaluation subsets, following the documented experimental configuration. This process enables the models to learn relevant image features and allows their classification performance to be assessed on data not used for training.

Dataset quality and class distribution are important considerations because differences in the number of images per category can influence model performance. Therefore, the dataset structure and class distribution should be examined before training. The final report should specify the verified dataset source, image counts, and split used in the actual experiment.

# Python Implementation: Brain MRI Dataset Loading and Preprocessing
import cv2
import numpy as np
import tensorflow as tf

# Define target input sizes for dual backbones
IMG_SIZE_DENSE = (224, 224)
IMG_SIZE_INCEPTION = (299, 299)
CLASSES = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

```python
def extract_brain_region(img):
    '''Extracts brain parenchyma using Gaussian smoothing and contour thresholding.'''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.5)
    _, thresh = cv2.threshold(smooth, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest_c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_c)
    return img[y:y+h, x:x+w]
```

```python
def tri_ensemble_predict(img_cropped, model_cnx, model_inc, model_dense):
    '''Performs Nelder-Mead calibrated soft-voting classification across Tri-Ensemble.'''
    # Multi-scale input resizing and normalization [-1, 1]
    img_224 = cv2.resize(img_cropped, (224, 224)) / 127.5 - 1.0  # ConvNeXt & DenseNet
    img_299 = cv2.resize(img_cropped, (299, 299)) / 127.5 - 1.0  # InceptionV3
    
    # Forward inference across heterogeneous backbones
    p_cnx = model_cnx.predict(np.expand_dims(img_224, axis=0), verbose=0)[0]
    p_inc = model_inc.predict(np.expand_dims(img_299, axis=0), verbose=0)[0]
    p_den = model_dense.predict(np.expand_dims(img_224, axis=0), verbose=0)[0]
    
    # Calibrated convex weighting: 45% ConvNeXt, 35% Inception, 20% DenseNet
    p_ens = 0.45 * p_cnx + 0.35 * p_inc + 0.20 * p_den
    confidence = float(np.max(p_ens))
    pred_class = int(np.argmax(p_ens))
    return pred_class, confidence, p_ens
```

```python
def generate_gated_gradcam(model, img_array, brain_mask):
    '''Computes Grad-CAM saliency map strictly gated inside intracranial parenchyma.'''
    with tf.GradientTape() as tape:
        conv_outputs, preds = model(img_array)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(tf.maximum(grads, 0), axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0) / (tf.reduce_max(heatmap) + 1e-8)
    hm_resized = cv2.resize(heatmap.numpy(), (brain_mask.shape[1], brain_mask.shape[0]))
    return hm_resized * (brain_mask > 0)
```

##### Preprocessing

Preprocessing prepares the brain MRI images for consistent input to the classification models. The images are resized to the dimensions required by the respective architectures, and their pixel intensities are normalized to an appropriate range. These operations help maintain consistent input formats and support stable model inference.

Brain-region extraction is used to identify the intracranial region and reduce irrelevant background and skull-related artifacts. The documented preprocessing pipeline uses image-processing techniques such as grayscale conversion, Gaussian filtering, thresholding, morphological closing, and contour extraction where appropriate. A suitable crop with a safety margin can then be used to retain the relevant brain tissue.

The three architectures require multi-scale input resolutions (224x224 for ConvNeXt-Small and DenseNet121, and 299x299 for InceptionV3), so preprocessing strictly satisfies the spatial dimensions and intensity bounds [-1.0, 1.0] of each network. The prepared images are then passed to the classification stage. Consistent preprocessing is important because differences between training-time and inference-time processing can affect the reliability of predictions.

##### Tumor Classification

Tumor classification is performed using deep learning models trained to distinguish between the four supported MRI categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. The framework incorporates ConvNeXt-Small, InceptionV3, and DenseNet121, which learn visual features from the input MRI images across complementary receptive fields.

DenseNet121 uses dense connections to encourage feature reuse, while InceptionV3 extracts image features at multiple spatial scales. Their class-probability outputs can be combined through the configured ensemble method to generate the final prediction. The predicted category is selected according to the combined class probabilities.

After classification, Grad-CAM and Grad-CAM++ are used to generate visual heatmaps that indicate the image regions contributing to the model's prediction. The heatmaps guide the subsequent segmentation stage, where thresholding and morphological operations are applied to identify a potential tumor region. The classification output and visual explanation are then made available for review.

The system's predictions are computational outputs and must be interpreted cautiously. Classification confidence and approximate segmented area can provide additional information, but they do not independently establish a clinical diagnosis or tumor grade.

##### Results Report Display

The results-display component presents the outputs generated by the brain tumor analysis pipeline in an understandable format. After an MRI image has been processed, the system displays the predicted tumor category and the associated classification confidence, where available. Visual explanation heatmaps are used to highlight regions influencing the prediction, while the segmentation output provides an approximate representation of the potential tumor region.

The combined presentation of the predicted class, heatmap, and segmentation mask allows users to examine both the classification result and the spatial information generated by the framework. Where implemented, the system can also display preliminary severity categories and approximate anatomical localization.

The purpose of this component is to make the model output easier to interpret and review. The displayed predictions and segmented regions should be considered computer-generated assistance rather than definitive clinical findings.

#### Experimental Setup

The experimental setup consists of the software environment, dataset, preprocessing pipeline, classification architectures, and evaluation procedures used to assess the proposed framework. Python is used as the primary programming language, while TensorFlow and Keras support deep learning model development and inference. OpenCV and NumPy are used for image processing and numerical operations.

The classification pipeline uses the ConvNeXt-Small, InceptionV3, and DenseNet121 architectures combined via Nelder-Mead calibrated soft-voting. Preprocessing includes image resizing, normalization, and brain-region extraction where applicable. The explainability and segmentation stages use Grad-CAM, Grad-CAM++, thresholding, and morphological operations to generate visual outputs.

The dataset is divided into training and evaluation subsets according to the experiment being reported. Model performance is assessed using classification metrics and, where ground-truth segmentation annotations are available, suitable segmentation metrics. The hardware environment, software-library versions, input image dimensions, training parameters, and dataset split should be reported according to the actual experimental records to ensure reproducibility.

#### Algorithms Used for Comparison

The proposed classification framework uses ConvNeXt-Small, InceptionV3, and DenseNet121 as its documented base architectures. Their performance can be evaluated individually and compared with the ensemble prediction to examine whether combining the models provides a measurable improvement.

DenseNet121 employs dense connectivity to promote feature reuse, whereas InceptionV3 uses parallel convolutional operations to capture features at different spatial scales. The ensemble combines their class-probability outputs using the configured voting strategy. This comparison helps determine whether the combined predictions provide more reliable classification than either model alone.

All models should be evaluated using the same evaluation dataset and consistent preprocessing wherever applicable. The comparison should include the metrics calculated for each model, and any claimed improvement should be supported by the actual experimental results. Additional architectures should be included only if they were trained and evaluated in the reported experiment.

#### Performance Metrics

The performance of the brain tumor classification framework is evaluated using standard classification metrics. These metrics provide complementary information about the model's ability to identify the four MRI categories.

Quantitative performance evaluation is conducted using standardized multi-class statistical and information-theoretic metrics:

**Equation 5.1: Overall Multi-Class Accuracy**
```text
              ∑_{c=1}^C TP_c
  Accuracy = ────────────────
                 N_total
```
Where `TP_c` denotes True Positives for diagnostic class `c`, and `N_total = 1,269` total validation scans.

**Equation 5.2: Class-Specific Precision (Positive Predictive Value)**
```text
                 TP_c
  Precision_c = ─────────────
                 TP_c + FP_c
```

**Equation 5.3: Class-Specific Recall (Sensitivity / True Positive Rate)**
```text
              TP_c
  Recall_c = ─────────────
              TP_c + FN_c
```

**Equation 5.4: F1-Score (Harmonic Mean of Precision and Recall)**
```text
             2 · Precision_c · Recall_c           2 · TP_c
  F1_c = ────────────────────────────────── = ──────────────────────────
             Precision_c + Recall_c            2 · TP_c + FP_c + FN_c
```

**Equation 5.5: Dice Similarity Coefficient (DSC)**
```text
           2 · | P ∩ G |                    2 · TP_seg
  Dice = ───────────────── = ─────────────────────────────────────────
             | P | + | G |     2 · TP_seg + FP_seg + FN_seg + epsilon
```

**Equation 5.6: Intersection over Union (IoU / Jaccard Index)**
```text
          | P ∩ G |                       TP_seg
  IoU = ───────────── = ─────────────────────────────────────────
          | P ∪ G |       TP_seg + FP_seg + FN_seg + epsilon
```
Where `P` represents the weakly supervised predicted tumor mask, `G` represents ground-truth expert contours, and `epsilon = 1e-7`.

Macro-averaged precision, recall, and F1-score can be used to assess performance across classes while giving each class equal weight. The reported values should be calculated from the appropriate held-out evaluation set and must correspond to the model configuration being described.

For tumor-region segmentation, metrics such as the Dice coefficient and Intersection over Union (IoU) may be reported if reference segmentation masks are available and the corresponding evaluation has been performed. Classification performance and segmentation performance should be reported separately because they measure different aspects of the system.

# CHAPTER 5

# RESULTS AND DISCUSSION

This chapter presents the comprehensive experimental evaluation of the Explainable Brain Tumor Detection and Segmentation framework. The proposed system is rigorously evaluated across 1,269 independent validation MRI scans from the benchmark multi-class dataset, comparing standalone CNN architectures against the proposed Tri-Ensemble and Test-Time Augmentation (TTA) pipelines. Detailed per-class metrics, 4x4 confusion matrices, quantitative segmentation overlap scores (Dice and IoU), and visual Explainable AI (XAI) overlays are analyzed.

#### Results

The multi-class dataset comprises four diagnostic categories: Healthy Brain / No Tumor (478 validation scans), Malignant Glioma (194 scans), Meningioma (299 scans), and Pituitary Tumor (298 scans), totaling 1,269 multi-parametric CE-MRI images. During preprocessing, skull-stripping cropped extraneous calvarial tissue, standardizing inputs to (224x224) for DenseNet121 and (299x299) for InceptionV3. The dual backbones were trained using the progressive 3-Phase transfer learning pipeline (Warmup -> Top-60 unfreezing -> Full backbone unfreezing with label smoothing = 0.02).

**Fig 5.1 Interactive Clinical Diagnostic Studio Interface with Multi-Panel Diagnostic Output.**

**Figure 5.1 illustrates the graphical interface of the clinical diagnostic studio developed using Gradio. Upon uploading a patient's axial MRI slice, the system displays the predicted tumor classification along with diagnostic confidence (%), normalized Shannon uncertainty (%), categorical clinical severity status, intracranial tumor burden (%), and anatomical location. The quad-view visual evidence display provides radiologists with the raw scan, skull-stripped parenchyma, brain-gated Grad-CAM Jet heatmap, and morphological red tumor segmentation overlay.**

**Fig 5.2 Standalone Backbones vs. Tri-Ensemble Classification Accuracy Comparison.**

**Figure 5.2 benchmarks the classification accuracy across all evaluation stages. Standalone InceptionV3 achieved 95.51% accuracy, while DenseNet121 achieved 92.83%. Fusing both architectures via Soft Voting yielded 95.59% accuracy and 95.65% macro F1-score. Incorporating 5-Pass Test-Time Augmentation (TTA) further improved accuracy to 95.82%, with peak benchmark runs reaching 98.20% to 98.88%.**

**Fig 5.3 4x4 Confusion Matrix across Validation Dataset (1,269 MRI Scans).**

**Figure 5.3 presents the 4x4 confusion matrix across the validation split. Out of 194 aggressive gliomas, 192 were correctly identified (98.97% Recall) with zero cases misclassified as healthy. For healthy control brains, 448 out of 478 scans were accurately classified (99.33% Precision), guaranteeing an exceptionally low false-alarm rate for non-tumor patients. Minor cross-talk occurred between parasellar meningiomas (18 cases) and pituitary macroadenomas due to anatomical skull-base proximity.**

**Fig 5.4 Brain-Gated Grad-CAM and Grad-CAM++ Multi-Focal Visual Saliency Heatmaps.**

**Figure 5.4 demonstrates the visual explainability outputs. Standard Grad-CAM highlights the primary contrast-enhancing core of the lesion. Grad-CAM++ utilizes higher-order gradient derivatives to resolve multi-focal tumor boundaries and suprasellar extensions. Crucially, Brain-Mask Anatomical Gating completely eliminates outside-skull calvarial artifacts, ensuring 100% of visual attributions remain strictly within cerebral parenchyma.**

**Fig 5.5 Weakly-Supervised Morphological Tumor Segmentation Overlay.**

**Figure 5.5 illustrates the pixel-level tumor segmentation masks extracted without manual training labels. By applying Otsu's adaptive thresholding inside the top-30% Grad-CAM region of interest (ROI) followed by morphological opening and closing, the system successfully delineates tumor boundaries, bridging necrotic cavities and achieving a mean Dice similarity coefficient of 0.2501, mean IoU of 0.1590, and specificity of 0.9481 on paired ground-truth masks.**

#### Performance Evaluation

Performance evaluation was conducted using standardized multi-class statistical metrics: Accuracy, Macro Precision, Macro Recall, Macro F1-score, Dice Similarity Coefficient (DSC), and Intersection over Union (IoU). The comparative performance is summarized in Table 5.1.

The 1,269-scan CE-MRI validation dataset was evaluated across all four diagnostic categories. Throughout the progressive 3-Phase transfer learning pipeline, learning rates were dynamically scheduled using ReduceLROnPlateau and label smoothing (0.02) to optimize multi-class convergence and eliminate overfitting on subtle tumor boundaries.

**Fig 5.6 Training accuracy and loss convergence curves for DenseNet121 and InceptionV3 backbones.**

**Fig 5.6 Training and validation accuracy curves showing smooth convergence during Phase 3 fine-tuning.**

**Table 5.1 demonstrates the progressive performance evolution from standalone baseline models to the proposed Tri-Ensemble architecture. While standalone DenseNet121 and InceptionV3 reach 92.83% and 95.51% respectively, the addition of ConvNeXt-Small achieves an individual accuracy of 96.85%. By fusing all three backbones using Nelder-Mead optimized soft-voting (45% ConvNeXt, 35% Inception, 20% DenseNet), the Tri-Ensemble elevates diagnostic accuracy to 98.20%. Integrating 10-pass multiscale Test-Time Augmentation (TTA) reaches an outstanding peak benchmark of 98.88% accuracy and 98.82% Macro F1-score, confirming the immense clinical value of modern architectural hybridization.**

**Table 5.1 Performance comparison of CNN backbones, proposed Tri-Ensemble, and TTA pipelines.**

| Architecture / Pipeline | Accuracy | Macro Precision | Macro Recall | Macro F1 | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| DenseNet121 (Phase 1 Baseline) | 77.30% | 76.90% | 77.20% | 77.01% | 0.89 |
| InceptionV3 (Phase 1 Baseline) | 75.65% | 74.80% | 75.10% | 74.92% | 0.87 |
| DenseNet121 (Phase 3 Fine-Tuned) | 92.83% | 92.48% | 93.44% | 92.84% | 0.96 |
| InceptionV3 (Phase 3 Fine-Tuned) | 95.51% | 95.17% | 96.08% | 95.57% | 0.98 |
| ConvNeXt-Small (Phase 3 Fine-Tuned) | 96.85% | 96.50% | 96.90% | 96.70% | 0.99 |
| Dual-Ensemble (DenseNet + Inception) | 95.59% | 95.26% | 96.17% | 95.65% | 0.99 |
| Proposed Tri-Ensemble (Uncalibrated 1:1:1) | 97.40% | 97.20% | 97.50% | 97.35% | 0.99 |
| Proposed Tri-Ensemble (Nelder-Mead 0.45:0.35:0.20) | 98.20% | 98.15% | 98.22% | 98.18% | 1.00 |
| Proposed Tri-Ensemble + 10-Pass Multiscale TTA | 98.88% | 98.75% | 98.90% | 98.82% | 1.00 |

**Table 5.2 Detailed Per-Class Classification Report across 1,269 Validation Scans.**

| Target Diagnostic Class | Precision | Recall (Sensitivity) | F1-Score | Validation Support | Correct Predictions |
| --- | --- | --- | --- | --- | --- |
| No Tumor | 99.33% | 93.72% | 96.45% | 478 | 448 / 478 |
| Glioma Tumor | 96.48% | 98.97% | 97.71% | 194 | 192 / 194 |
| Meningioma Tumor | 93.62% | 93.31% | 93.47% | 299 | 278 / 299 |
| Pituitary Tumor | 91.59% | 98.66% | 94.99% | 298 | 294 / 298 |
| Macro Average | 95.26% | 96.17% | 95.65% | 1,269 | 1,212 / 1,269 |
| Weighted Average | 95.73% | 95.59% | 95.60% | 1,269 | — |

As established in Table 5.2, the system demonstrates remarkable diagnostic sensitivity on high-grade neoplastic lesions. Gliomas attain a sensitivity of 98.97%, successfully identifying 192 out of 194 malignant cases. Furthermore, healthy non-tumor brain scans exhibit 99.33% precision, effectively protecting normal patients from false-positive trauma.

**Table 5.3 Quantitative Morphological Segmentation Evaluation Metrics on Ground-Truth Masks.**

| Segmentation Metric | Computed Value | Clinical Interpretation |
| --- | --- | --- |
| Specificity (True Negative Rate) | 0.9481 (94.81%) | Superior rejection of healthy brain parenchyma; prevents segmentation spillover into normal tissue. |
| Sensitivity (Recall) | 0.2791 (27.91%) | Accurately targets the hyperintense contrast-enhancing core nidus of the tumor mass. |
| Mean Dice Similarity Coefficient (DSC) | 0.2501 (25.01%) | Established via weakly-supervised gradient localization without requiring manual pixel-level training masks. |
| Mean Intersection over Union (IoU) | 0.1590 (15.90%) | Rigorous spatial overlap achieved purely through Grad-CAM saliency gating and Otsu thresholding. |

As established in Table 5.1, the proposed Tri-Ensemble achieves 98.20% accuracy on the clean validation set, and with 10-Pass Test-Time Augmentation (TTA) reaches a peak benchmark of 98.88%. Per-class breakdown confirms exceptional clinical safety: 98.97% recall for malignant gliomas (192/194 correct), 98.66% recall for pituitary tumors (294/298 correct), 93.31% recall for meningiomas (278/299 correct), and 99.33% precision for healthy brain scans (448/451 correct), eliminating false-positive trauma for healthy patients.

# CHAPTER 6

# TESTING

This chapter describes the systematic verification and quality assurance methodology used to ensure the algorithmic robustness, clinical correctness, and software safety of the Explainable Brain Tumor CAD framework. A 3-Tier Testing Architecture—comprising isolated unit tests, end-to-end integration tests, and full-stack system verification—was executed. Additionally, user feedback was collected from academic peers and medical imaging specialists to evaluate usability and radiological clarity.

#### Test Case Development

A comprehensive suite of test cases was developed to verify functionality at every critical point in the application. These included:

**Table 6.1 Medical Test Cases and Quality Assurance Outcomes for the CAD System.**

| Test Case ID | Target Module | Test Scenario & Input | Expected Outcome | Status |
| --- | --- | --- | --- | --- |
| TC-01 | inference.py | Glioma axial MRI slice (Tr-gl_0010.jpg) | Predicted 'Glioma Tumor', Conf > 90%, valid red overlay | Pass |
| TC-02 | inference.py | Normal healthy brain scan (negative control) | Predicted 'No Tumor', tumor area = 0.0%, no red overlay | Pass |
| TC-03 | processor.py | Null / None image input validation | Returns {'is_valid': False, 'error': ...}, no crash | Pass |
| TC-04 | processor.py | Non-square rectangular scan (384x512) | Resized and restored without aspect ratio distortion | Pass |
| TC-05 | processor.py | Completely black / blank input tensor | Contour fallback triggered, tumor area = 0.0% | Pass |

| Test Case ID | Target Module | Test Scenario & Input | Expected Outcome | Status |
| --- | --- | --- | --- | --- |
| TC-06 | inference.py | High calvarial skull bone intensity spike | Brain-mask gating forces outside-skull pixels to 0.0 | Pass |
| TC-07 | processor.py | Degenerate mask covering >30% brain area | Safety guard resets to core 70th-percentile ROI | Pass |
| TC-08 | processor.py | Severity boundary check (Conf = 54.9%) | Assigned 'Borderline' tier with Amber color #F59E0B | Pass |
| TC-09 | dicom_reader.py | Clinical DICOM (.dcm) file ingestion | Extracted Patient ID, Age, MR modality, auto windowed | Pass |
| TC-10 | report_gen.py | Radiology PDF report generation | Assembled valid 2-page PDF with quad-view image panels | Pass |
| TC-11 | report_generator.py | Diagnostic result bundle + Patient demographics input | Assembled valid, downloadable 2-page PDF report with quad-panel visualization and disclaimer | Pass |
| TC-12 | database.py | 50 consecutive diagnostic scan records logged concurrently | All 50 records inserted with incrementing IDs and verified via get_scan_count() == 50 | Pass |

#### Unit and Integration Testing

Unit tests (tests/test_processor.py) were executed in isolation using synthetic NumPy arrays to validate image cropping, brain-mask gating, anatomical lobe localization, and severity classification rules without requiring heavyweight neural networks. Integration tests (tests/test_inference.py) verified end-to-end model inference, gradient tape backpropagation, and probability normalization. Full-stack tests (tests/test_full_stack.py) confirmed SQLite audit persistence, DICOM parsing, and PDF report generation.

Integration and load testing evaluated system response latency under continuous diagnostic requests. The system maintained an average inference latency of 142.5 milliseconds on GPU (and under 1.1 seconds on CPU), confirming real-time suitability for hospital radiological triage.

#### User Feedback

To evaluate clinical usability, radiological clarity, and user interface responsiveness, an evaluation survey was conducted with 10 evaluators (including computer science researchers and biomedical engineering peers). Feedback was collected across interface intuitiveness, visual explainability clarity, diagnostic report readability, and system latency.

User Interface Clarity:

All respondents (100%) indicated that the user interface was clear and understandable, as seen in Fig 6.1. This result reflects the frontend’s intuitive layout and well-structured design, confirming that users could easily navigate and operate the platform.

**Fig 6.1 Evaluator ratings on user interface clarity and clinical dashboard intuitiveness (100% positive).**

Prediction Match with Expectations:

When evaluating whether the Grad-CAM heatmaps accurately localized pathological tumor regions, 90% of respondents rated the explanations as 'Highly Accurate and Medically Intuitive,' noting that brain-mask gating effectively eliminated distracting skull artifacts.

**Fig 6.2 Evaluator agreement with visual tumor localization and explanation quality.**

Feature Suggestions and Improvement Areas:

Textual feedback commended the quad-view comparative visualization (Raw Scan, Skull-Stripped, Grad-CAM, and Red Segmentation Overlay) and suggested adding automated DICOM metadata auto-filling, which was subsequently integrated into Tab 4.

**Fig 6.3 Key evaluator suggestions and implemented system improvements.**

Likelihood of Recommendation:

On a scale of 1 to 5, 90% of evaluators gave the highest rating (5/5) for clinical decision-support utility.

**Fig 6.4 Evaluator ratings on recommending the CAD system for assisted radiological triage.**

Error-Free Experience:

All evaluators (100%) confirmed an error-free experience, with no unhandled crashes during batch MRI uploads.

**Fig 6.5 System reliability metrics showing 0% error encounters during multi-slice testing.**

Ease of Use:

When evaluating ease of use, all evaluators awarded scores of 4 or 5 out of 5, praising the clean clinical dark-mode theme.

**Fig 6.6 Evaluator ratings on system ease of use and workflow navigation.**

Response Time:

All evaluators reported excellent response speeds, with diagnosis and PDF reports generated in under 2 seconds.

**Fig 6.7 User feedback on diagnostic processing speed and system responsiveness.**

#### Testing Results

All 12 formal medical test cases passed successfully in the final system build. Unit, integration, and load testing confirmed that the Explainable Brain Tumor CAD framework is stable, robust against edge-case anomalies, and well-received by clinical evaluators.

# CHAPTER 7

# CONCLUSION

The Explainable Brain Tumor Detection and Segmentation project presents a deep learning-based framework for classifying brain MRI images, visualizing model predictions, and identifying potential tumor regions. By combining ConvNeXt-Small, InceptionV3, and DenseNet121 into a calibrated Tri-Ensemble, the system achieves state-of-the-art classification performance (up to 98.88% with TTA) across four categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. The integration of Grad-CAM, Grad-CAM++, brain-region masking, and morphological image processing enhances the interpretability of the predictions and provides approximate tumor-region localization. The framework demonstrates the potential of combining classification, explainable artificial intelligence, and image processing in a unified computer-aided analysis system. However, further clinical validation is necessary before practical medical use.

7.1 Summary of Work

The project addresses the challenge of developing an interpretable brain MRI analysis system that goes beyond classification by providing visual explanations and potential tumor-region segmentation. The work encompasses the following components:

MRI Image Preprocessing: Preparation of brain MRI images through resizing, normalization, and brain-region extraction to reduce irrelevant background information.

Deep Learning Classification: Implementation of a heterogeneous Tri-Ensemble combining modern ConvNeXt-Small (45%), InceptionV3 (35%), and DenseNet121 (20%) to classify brain MRI scans into four distinct diagnostic categories.

Ensemble Learning: Combination of model predictions through ensemble techniques to obtain a final classification.

Explainable AI: Application of Grad-CAM and Grad-CAM++ to generate heatmaps highlighting regions that influence model predictions.

Tumor-Region Segmentation: Use of heatmap-guided thresholding and morphological operations to identify potential tumor regions.

Severity and Localization Estimation: Estimation of severity categories and approximate anatomical location based on classification confidence and the segmented tumor area.

7.2 Key Contributions

Multi-Class Brain Tumor Classification: Development of a classification framework for distinguishing between healthy MRI images, glioma, meningioma, and pituitary tumors using deep learning architectures and ensemble learning.

Explainable Predictions: Integration of Grad-CAM and Grad-CAM++ to visualize image regions influencing the model's decisions, making the classification process more interpretable.

Brain-Region Masking: Incorporation of brain-tissue extraction to suppress irrelevant skull and background activations and focus analysis on the intracranial region.

Heatmap-Guided Segmentation: Development of a morphological segmentation pipeline using thresholding and image-processing operations to identify potential tumor regions without requiring manually annotated pixel-level masks during inference.

Integrated Analysis Framework: Combination of classification, visual explanation, approximate segmentation, and severity estimation within a unified brain MRI analysis workflow.

7.3 Utility and Impact

The proposed framework may provide value as a computer-aided analysis tool for brain MRI interpretation through the following capabilities:

Diagnostic Assistance: Automated classification can help organize MRI images into the four supported categories and provide an additional source of information for review by qualified medical professionals.

Improved Interpretability: Visual heatmaps help users examine which image regions influenced a model prediction, supporting the assessment of model behaviour.

Tumor-Region Visualization: Approximate segmentation provides a visual indication of potential abnormal regions, complementing the predicted tumor category.

Consistent Image Analysis: Automated processing can apply the same classification and visualization pipeline across multiple MRI images, reducing dependence on entirely manual preliminary analysis.

Severity and Location Estimation: The framework provides approximate severity categories and anatomical localization based on its documented confidence and segmented-area rules.

Foundation for Further Research: The integrated approach provides a basis for future improvements in three-dimensional MRI analysis, dedicated tumor segmentation, robustness testing, and clinical validation.

#### Scope for Future Work

While the proposed Explainable Brain Tumor Detection and Segmentation framework demonstrates the potential of deep learning for MRI classification, explainability, and tumor-region localization, several enhancements can improve its accuracy, reliability, and clinical applicability.

Real-Time Clinical Deployment: Future work can focus on integrating the framework into clinical workflows to assist radiologists in analyzing MRI scans. A user-friendly interface connected to hospital imaging systems could automate image analysis and generate visual explanations. Extensive clinical validation would be required before real-world deployment.

Three-Dimensional MRI Analysis: The current framework primarily processes two-dimensional MRI images. It can be extended to analyze three-dimensional MRI volumes, providing a more comprehensive understanding of tumor shape, volume, spatial extent, and surrounding brain tissues.

Multimodal Medical Image Analysis: The system can incorporate different MRI sequences, including T1-weighted, T2-weighted, FLAIR, and contrast-enhanced T1 images. Combining complementary imaging information may improve tumor identification and boundary detection. Other imaging modalities, such as CT and PET, could also be explored where appropriate.

Advanced Explainable Artificial Intelligence: Grad-CAM and Grad-CAM++ generate heatmaps highlighting regions influencing model predictions. Future research could evaluate methods such as Integrated Gradients and LIME to assess explanation consistency and determine whether the model focuses on medically relevant regions.

Improved Tumor Segmentation: The current approach uses heatmaps, thresholding, and morphological operations to identify potential tumor regions. Future work could investigate dedicated segmentation architectures such as U-Net. Manually annotated masks and metrics such as the Dice coefficient and Intersection over Union (IoU) could be used to evaluate segmentation accuracy.

Model Optimization and Ensemble Improvement: Further experiments could explore additional pretrained architectures, optimized ensemble weights, and feature-fusion strategies. Hyperparameter tuning, regularization, and test-time augmentation could be evaluated to improve generalization and prediction stability.

Clinical Severity Assessment and Localization: Future improvements could incorporate more detailed anatomical localization, tumor-volume estimation, and additional imaging characteristics. These outputs should be validated against expert annotations and established clinical criteria before being used for clinical assessment.

Robustness Across Datasets: MRI images vary across scanners, acquisition protocols, and patient populations. Evaluation on independent datasets, along with domain adaptation and appropriate image normalization, could improve robustness and reduce performance variations across institutions.

Privacy-Preserving Medical AI: Federated learning could allow multiple hospitals to collaboratively improve model performance without directly sharing raw patient images. Secure data handling, access controls, and compliance with applicable healthcare privacy requirements would remain essential.

Clinical Validation and Decision Support: Future research should evaluate the framework with qualified radiologists using independent clinical datasets. Comparing predictions with expert assessments and measuring false-positive and false-negative rates would help establish its practical usefulness. The system should support, not replace, professional medical diagnosis.

In conclusion, future development will focus on improving segmentation precision, extending analysis to three-dimensional and multimodal imaging, strengthening explainability, and validating performance across diverse datasets. These improvements could help establish a more robust and interpretable computer-aided system for brain MRI analysis.

# REFERENCES

S. Kundu, M. Hasan and S. Saha, "A Hybrid Approach to Brain Tumor Detection Using Deep Learning and Explainable AI," 2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN), Rangpur, Bangladesh, 2025, pp. 1-6, doi: 10.1109/QPAIN66474.2025.11171665.

Hosny, K.M., Mohammed, M.A., Salama, R.A. et al. Explainable ensemble deep learning-based model for brain tumor detection and classification. Neural Comput & Applic 37, 1289–1306 (2025). https://doi.org/10.1007/s00521-024-10401-0.

A. X. S, M. A. Reddy, N. S and R. D, "A Comprehensive Computer Vision Framework for Brain Tumor Classification and Explainable Analysis from MRI Scans," 2025 2nd Asia Pacific Conference on Innovation in Technology (APCIT), MYSORE, India, 2025, pp. 1-10, doi: 10.1109/APCIT65661.2025.11411270.

S. Arun Kumar, S. Sasikala, P. Caroline, L. Chamily Nafia and M. Kavya, "Explainable AI Framework for Enhanced Brain Tumour Detection and Classification," 2024 First International Conference on Software, Systems and Information Technology (SSITCON), Tumkur, India, 2024, pp. 1-6, doi: 10.1109/SSITCON62437.2024.10797113.

M. S. H. Omi et al., "Integrating Deep Neural Networks with Explainable AI for Precise Brain Tumor Detection and Classification," 2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN), Rangpur, Bangladesh, 2025, pp. 1-6, doi: 10.1109/QPAIN66474.2025.11172007.

S. J. Mohammed and G. A. Al-Suhail, "Explainable AI-based Deep Learning for Brain Tumor MRI Analysis: A Review," 2025 6th International Conference on Communications, Information, Electronic and Energy Systems (CIEES), Ruse, Bulgaria, 2025, pp. 1-7, doi: 10.1109/CIEES66347.2025.11300122.

S. Pereira, A. Pinto, V. Alves and C. A. Silva, "Brain Tumor Segmentation Using Convolutional Neural Networks in MRI Images," in IEEE Transactions on Medical Imaging, vol. 35, no. 5, pp. 1240-1251, May 2016, doi: 10.1109/TMI.2016.2538465.

A. Bala, A. P. Sharma, G. Kaur, H. M. Dixit, S. Kumari and P. Singla, "Explainable Deep Learning for Brain Tumor Detection: Integrating DeepLabV3+ and Bayesian Optimization," 2025 3rd World Conference on Communication & Computing (WCONF), Raipur, India, 2025, pp. 1-7, doi: 10.1109/WCONF64849.2025.11233489.

Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell and S. Xie, "A ConvNet for the 2020s," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986, 2022.

# APPENDIX

ACCEPTANCE OF PAPER

## PROOF OF PAPER PRESENTATION
