# Dataset Architecture and Preprocessing

## 1. Dataset Provenance and Specifications

The experimental foundation of this research utilizes Contrast-Enhanced T1-weighted Magnetic Resonance Imaging (CE-MRI) scans originating from the benchmark clinical dataset compiled by Cheng et al. (Figshare) and aggregated within the Sartaj Kaggle CE-MRI repository.

### 1.1 Image Acquisition Characteristics
- **Imaging Modality:** T1-weighted Contrast-Enhanced MRI (gadolinium contrast agent injected intravenously).
- **Scan Orientations:** Axial (transverse), coronal, and sagittal cross-sectional slices.
- **Slice Resolution:** Variable raw clinical resolutions ranging from 512x512 down to 256x256 pixels, with 8-bit or 16-bit grayscale depth.
- **Pathological Classes:** 4 distinct diagnostic categories (No Tumor, Glioma, Meningioma, Pituitary).

### 1.2 Dataset Class Distributions Across Phases

| Class Index | Diagnostic Category | Raw Scans (Phase 1 Report) | Augmented Training Set | Validation Split (Evaluation Benchmark) |
|---|---|---|---|---|
| Class 0 | No Tumor (Normal) | 395 | 2,050 | 478 |
| Class 1 | Glioma Tumor | 826 | 2,042 | 194 |
| Class 2 | Meningioma Tumor | 822 | 2,040 | 299 |
| Class 3 | Pituitary Tumor | 827 | 2,045 | 298 |
| **Total** | **All 4 Classes** | **2,870 - 3,264** | **8,177** | **1,269** |

---

## 2. Image Preprocessing Pipeline

Raw clinical MRI images exhibit considerable variations in field-of-view, patient skull positioning, scanner contrast settings, and background illumination. To standardize input features for the convolutional backbones, a multi-stage preprocessing pipeline is executed.

### 2.1 Step 1: Grayscale Conversion
When reading three-channel RGB representations of grayscale MRI slices, the luminance channel is computed using standard ITU-R BT.601 perceptual weights:

```
Gray(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)
```

Where:
- R(x, y), G(x, y), B(x, y) represent red, green, and blue color channels at pixel coordinate (x, y).
- Gray(x, y) is the resulting single-channel luminance intensity [0 to 255].

### 2.2 Step 2: Floating-Point Rescaling and Normalization
To prevent numerical instability and exploding gradients during backpropagation, integer pixel intensities in the range [0, 255] are rescaled into standard floating-point values [0.0, 1.0]:

```
X_norm(x, y) = X(x, y) / 255.0
```

For general numeric contrast normalization across variable dynamic ranges (such as 12-bit or 16-bit DICOM pixel data):

```
X_minmax(x, y) = [ X(x, y) - X_min ] / [ X_max - X_min + epsilon ]
```

Where:
- X_min is the minimum intensity in the scan volume.
- X_max is the maximum intensity in the scan volume.
- epsilon is a minute stabilizing constant (1e-7) preventing division by zero on uniform blank scans.

### 2.3 Step 3: Automated OpenCV Brain Region Extraction (Skull Stripping)
A common failure mode of CNNs in medical imaging is attending to black background margins or bright skull bone edges. To enforce anatomical focus, an automated contour-detection skull-cropping routine is applied:

1. **Gaussian Smoothing:** Remove high-frequency thermal scanner noise:
   ```
   G(x, y) = [ 1 / (2 * pi * sigma^2) ] * exp( - (x^2 + y^2) / (2 * sigma^2) )
   I_smooth(x, y) = Gray(x, y) convolved with G(x, y, sigma=1.5)
   ```
2. **Binary Thresholding:** Convert the smoothed image to a binary tissue mask:
   ```
   Binary_Mask(x, y) = 255   if I_smooth(x, y) > Threshold_Intensity (e.g., 45)
   Binary_Mask(x, y) = 0     otherwise
   ```
3. **Largest Connected Component Identification:** Find all external contours and identify the contour with maximal enclosed area:
   ```
   Contour_brain = argmax_c [ Area(Contour_c) ]
   ```
4. **Bounding Extremities Calculation:** Identify the extreme Cartesian coordinates enclosing the intracranial tissue:
   ```
   x_min = min [ x for (x, y) in Contour_brain ]
   x_max = max [ x for (x, y) in Contour_brain ]
   y_min = min [ y for (x, y) in Contour_brain ]
   y_max = max [ y for (x, y) in Contour_brain ]
   ```
5. **Cropped Region Extraction:**
   ```
   I_cropped = I_raw[ y_min : y_max , x_min : x_max ]
   ```

This cropping step eliminates up to 45% of uninformative black background border pixels and aligns the cerebrum consistently across all samples.

### 2.4 Step 4: Multi-Scale Spatial Resizing
Because the two ensemble backbones possess differing native receptive field designs, the cropped scans are resized using bicubic interpolation:
- **DenseNet121 Input:** Resized to (224 x 224 x 3).
- **InceptionV3 Input:** Resized to (299 x 299 x 3).

---

## 3. Data Augmentation and Regularization

### 3.1 Anatomically-Constrained Geometric Transformations
Unlike generic computer vision datasets (e.g., ImageNet), medical neuroimaging data cannot undergo arbitrary distortion. Severe shearing or excessive rotations corrupt natural anatomical symmetry and deceive spatial diagnostic reasoning. The augmentation strategy is strictly constrained:

| Transformation Parameter | Applied Value | Clinical and Algorithmic Rationale |
|---|---|---|
| Rotation Range | +/- 15 degrees | Reflects slight patient head tilt within the scanner head coil. |
| Horizontal Flip | Enabled (True, p=0.5) | Preserves bilateral cerebral hemisphere symmetry (left/right). |
| Vertical Flip | Disabled (False) | Strict: Scans must remain anatomically upright (Superior to Inferior). |
| Zoom Range | +/- 10% [0.90, 1.10] | Simulates minor variation in scanner magnification and head size. |
| Shear Range | 0.0 (Disabled) | Strictly prohibited: Shearing distorts intracranial tissue geometry. |
| Brightness Range | [0.90, 1.10] | Simulates RF coil excitation inhomogeneity and gain variations. |
| Width Shift Range | +/- 8% | Accommodates off-center positioning in the coil bore. |
| Height Shift Range | +/- 8% | Accommodates vertical field-of-view shifts. |
| Fill Mode | Constant (cval=0) | Zero-padding (black) replicates the dark ambient MRI scanner void. |

---

## 4. Balanced Class Weight Formulation

Due to the unequal distribution of samples across tumor types (with the "No Tumor" class having fewer historical samples in the initial raw repository and "Glioma" having fewer validation samples), standard optimization produces a majority-class bias.

To penalize minority class errors proportionally, balanced class weights are computed using the inverse class frequency formulation:

```
Weight(c) = N_total / [ N_classes * N_samples(c) ]
```

Where:
- N_total is the total number of training images across all classes.
- N_classes is the number of distinct classes (4).
- N_samples(c) is the total count of training images in class c.

### Worked Example:
If N_total = 2,968:
- No Tumor (N_0 = 395): Weight(0) = 2968 / (4 * 395) = 1.8785
- Glioma (N_1 = 826): Weight(1) = 2968 / (4 * 826) = 0.8983
- Meningioma (N_2 = 822): Weight(2) = 2968 / (4 * 822) = 0.9027
- Pituitary (N_3 = 827): Weight(3) = 2968 / (4 * 827) = 0.8972

These weights directly scale the categorical cross-entropy loss gradients for each batch sample during training, guaranteeing that minority classes contribute equally to parameter updates.
