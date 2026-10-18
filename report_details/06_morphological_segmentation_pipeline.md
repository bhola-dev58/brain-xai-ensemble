# Morphological Segmentation Pipeline

## 1. Weakly-Supervised Segmentation Strategy

In standard medical deep learning, training supervised semantic segmentation models (such as U-Net or DeepLabV3+) requires thousands of manually contoured ground-truth masks—a process demanding hundreds of hours of expert neuro-radiologist labor.

To overcome this dependency, this project implements a **Weakly-Supervised Morphological Segmentation Pipeline**. The system leverages classification-trained feature gradients (Grad-CAM) to establish coarse pathological regions of interest (ROI), and subsequently applies mathematical morphology and adaptive Otsu thresholding to extract exact, pixel-level tumor boundaries automatically during inference.

---

## 2. Complete Mathematical Workflow

```
[Grad-CAM Heatmap HM(x, y)]
            │
            ▼
[Brain-Mask Anatomical Gating] ──► Discards all outside-skull pixels
            │
            ▼
[Top-30% ROI Extraction] ───────► Identifies primary tumor spatial zone
            │
            ▼
[Otsu Adaptive Thresholding] ───► Isolates hyperintense contrast-enhanced mass
            │
            ▼
[Morphological OPEN + CLOSE] ───► Eliminates noise spikes and bridges necrosis holes
            │
            ▼
[Largest Connected Component] ──► Removes spurious satellite fragments
            │
            ▼
[Tumor Area % Quantification] ──► Computes relative intracranial tumor burden
```

---

## 3. Step-by-Step Mathematical Derivations

### 3.1 Step 1: Heatmap Scaling and Intracranial Gating
The raw Grad-CAM activation map HM_raw is resized to the native image spatial dimensions (W x H) via bicubic interpolation and normalized to floating-point range [0.0, 1.0]:

```
HM(x, y) = [ HM_interp(x, y) - min(HM) ] / [ max(HM) - min(HM) + epsilon ]
```

The normalized map is strictly gated by the binary brain tissue mask Mask_brain:

```
HM_gated(x, y) = HM(x, y) * Mask_brain(x, y)
```

### 3.2 Step 2: Region of Interest (ROI) Thresholding
Rather than segmenting the entire image, the system isolates the candidate pathological zone using the 70th percentile of intracranial activations:

1. Extract all activation values falling within the cerebral parenchyma:
   ```
   V_brain = { HM_gated(x, y)  such that  Mask_brain(x, y) > 0 }
   ```
2. Compute the dynamic 70th percentile threshold tau_ROI, bounded by a minimum floor:
   ```
   tau_ROI = max( Percentile_70( V_brain ), 0.45 )
   ```
3. Generate the binary ROI candidate mask:
   ```
   Mask_ROI(x, y) = 1    if HM_gated(x, y) > tau_ROI and Mask_brain(x, y) > 0
   Mask_ROI(x, y) = 0    otherwise
   ```
4. Perform structural gap filling via morphological closing with an elliptical structuring element B_roi of size 13 x 13:
   ```
   Mask_ROI_closed = ( Mask_ROI (+) B_roi ) (-) B_roi
   ```
5. Extract the largest connected contour to discard weak secondary focal points:
   ```
   Mask_ROI_final = Largest_Connected_Component( Mask_ROI_closed )
   ```

### 3.3 Step 3: Adaptive Otsu Thresholding within the ROI
Within the isolated candidate ROI, actual tumor tissue exhibits distinct hyperintense or contrast-enhanced grayscale values compared to normal surrounding brain matter.

To find the optimal separation threshold without manual tuning, **Otsu's Thresholding Algorithm** is executed exclusively over the pixel intensity distribution of the ROI interior:

Let the grayscale intensity values inside Mask_ROI_final be normalized into an L-bin probability distribution p(i) for intensities i in {0, 1, 2, ..., 255}:

```
p(i) = n_i / N_ROI
```

Where:
- n_i is the count of pixels with intensity i inside the ROI.
- N_ROI is the total number of pixels inside the ROI.

For any candidate threshold t separating pixels into background class C_0 [intensities 0 to t] and tumor class C_1 [intensities t+1 to 255]:

1. **Cumulative Class Probabilities:**
   ```
   omega_0(t) = sum_{i=0}^t p(i)
   omega_1(t) = sum_{i=t+1}^{255} p(i) = 1 - omega_0(t)
   ```
2. **Class Mean Intensities:**
   ```
   mu_0(t) = [ sum_{i=0}^t ( i * p(i) ) ] / omega_0(t)
   mu_1(t) = [ sum_{i=t+1}^{255} ( i * p(i) ) ] / omega_1(t)
   ```
3. **Total Mean Intensity:**
   ```
   mu_total = omega_0(t) * mu_0(t) + omega_1(t) * mu_1(t)
   ```
4. **Between-Class Variance Formulation:**
   The optimal threshold t_opt maximizes the statistical variance between the two classes:
   ```
   sigma_B^2(t) = omega_0(t) * omega_1(t) * [ mu_0(t) - mu_1(t) ]^2
   ```
5. **Optimal Threshold Selection:**
   ```
   t_opt = argmax_{0 <= t < 255} [ sigma_B^2(t) ]
   ```

The raw tumor mask is then established:

```
Mask_tumor_raw(x, y) = 1   if [ Gray(x, y) > t_opt and Mask_ROI_final(x, y) == 1 and Mask_brain(x, y) == 1 ]
Mask_tumor_raw(x, y) = 0   otherwise
```

### 3.4 Step 4: Morphological Cleanup and Boundary Regularization
Raw thresholded masks contain isolated noisy pixels and internal holes (often caused by hypointense necrotic tumor cores). The mask is regularized using mathematical morphological operators:

1. **Morphological Opening (Noise Removal):**
   Erosion followed by dilation using an elliptical kernel B_clean of size 5 x 5:
   ```
   Mask_opened = ( Mask_tumor_raw (-) B_clean ) (+) B_clean
   ```
2. **Morphological Closing (Hole Filling):**
   Three successive dilations followed by erosions using kernel B_clean:
   ```
   Mask_closed = ( ( Mask_opened (+) B_clean ) ... ) (-) B_clean
   ```
3. **Primary Lesion Extraction:**
   ```
   Mask_tumor_main = Largest_Connected_Component( Mask_closed )
   ```
4. **Gaussian Edge Smoothing:**
   To produce radiologically natural smooth contours:
   ```
   Mask_blurred = GaussianBlur( Mask_tumor_main, kernel=(9, 9), sigma=2.0 )
   Mask_final(x, y) = 255   if Mask_blurred(x, y) >= 127  else 0
   ```

---

## 4. Tumor Area Percentage Quantification

Standard CAD systems often compute tumor area as a percentage of the entire rectangular image frame. This metric is clinically invalid because it fluctuates depending on the extent of black background margin present in the scan.

Our framework calculates **Relative Intracranial Tumor Burden (rho_tumor)** strictly against the patient's actual cerebral volume:

```
Tumor_Pixels = sum_{x=1}^W sum_{y=1}^H [ 1   if Mask_final(x, y) > 0  else 0 ]
Brain_Pixels = sum_{x=1}^W sum_{y=1}^H [ 1   if Mask_brain(x, y) > 0  else 0 ]

rho_tumor = [ Tumor_Pixels / Brain_Pixels ] * 100.0%
```

### Fallback Guard:
If the segmented mask produces 0.0% or an anatomically unrealistic volume (>30.0% of the entire brain), the system safely falls back to the core ROI contour to guarantee that no pathological lesion is overlooked in clinical reports.

---

## 5. Semi-Transparent Visual Alpha Blending

For interactive radiological display, the binary tumor mask is rendered as a semi-transparent colored overlay on top of the original MRI scan:

```
I_segmented(x, y) = (1.0 - alpha) * I_orig(x, y) + alpha * Color_Overlay(x, y)
```

Where:
- alpha = 0.45 represents the blending opacity.
- Color_Overlay = (255, 0, 0) (Vibrant Medical Red).
- Applied exclusively to coordinates where Mask_final(x, y) > 0.
- Pixels outside the segmented tumor retain 100% of their original anatomical clarity.
