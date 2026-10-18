# Explainable AI (XAI): Grad-CAM, Grad-CAM++, and LIME

## 1. The Imperative for Interpretability in Neuro-Oncology

Deep convolutional networks achieve exceptional classification accuracy but inherently function as black-box decision models. In high-consequence medical diagnostics, algorithmic opacity introduces severe vulnerabilities:
- Models may learn high-frequency non-biological correlations (e.g., scanner vendor watermarks, patient positioning markers, or intense calvarial bone reflections).
- Radiologists cannot verify if a predicted "Glioma" is based on the actual infiltrative parenchymal lesion or surrounding skull artifacts.

To solve this, the framework integrates Explainable Artificial Intelligence (XAI) at multiple granularities:
1. **Global Class Activation Saliency:** Standard Grad-CAM.
2. **Multi-Focal High-Resolution Saliency:** Grad-CAM++.
3. **Brain-Mask Anatomical Gating:** Zeroing outside-skull false positives.
4. **Local Perturbation Super-Pixel Attribution:** LIME (Local Interpretable Model-agnostic Explanations).

---

## 2. Standard Grad-CAM Mathematical Derivation

Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., 2017) computes the visual gradient of the predicted class score with respect to the spatial feature maps of the final convolutional layer.

### 2.1 Target Layer Selection
In DenseNet121, the feature representation just before the classification head is the final ReLU layer following Dense Block 4 (`relu` / 7x7 spatial grid with 1,024 channels). In InceptionV3, the target is the final concatenated mixed layer (`mixed10` / 8x8 spatial grid with 2,048 channels).

### 2.2 Gradient Computation
Let y^c denote the raw class score (logit) for diagnostic class c. Let A^k represent the k-th 2D feature map of the chosen convolutional layer, with spatial dimensions u x v (where u is height, v is width).

The gradient of class score y^c with respect to feature map activation A^k at spatial coordinate (i, j) is given by:

```
Gradient = d(y^c) / d(A^k_{i, j})
```

### 2.3 Neuron Importance Weight Formulation (Global Average Pooling)
To quantify the global importance of feature channel k in predicting class c, the gradients are pooled across all spatial coordinates (i, j):

```
alpha_k^c = [ 1 / Z ] * sum_{i=1}^u sum_{j=1}^v [ d(y^c) / d(A^k_{i, j}) ]
```

Where:
- Z = u * v represents the total number of spatial pixels in the feature map (e.g., 7 * 7 = 49 in DenseNet121).
- alpha_k^c is the resulting scalar importance weight for channel k and target class c.

### 2.4 Forward Activation Map Combination and Rectified Linear Activation
The final 2D Class Activation Map L_GradCAM^c is computed as a weighted linear summation of all feature maps A^k, followed by a Rectified Linear Unit (ReLU) operation:

```
L_GradCAM^c = ReLU( sum_{k=1}^K [ alpha_k^c * A^k ] )
```

### Mathematical Necessity of ReLU:
The ReLU function ensures that only feature channels whose presence **positively increases** the target class score y^c are visualized. Channels that suppress class c (negative gradients) typically correlate with competing pathological classes and are clipped to zero, preventing visual confusion.

### 2.5 Normalization to Floating-Point Range [0.0, 1.0]
To prepare the raw activation map for visualization and downstream segmentation:

```
L_norm(i, j) = [ L_GradCAM^c(i, j) - min(L) ] / [ max(L) - min(L) + epsilon ]
```

---

## 3. Noise Suppression via Top-K Channel Pooling

In deep architectures with 1,024 to 2,048 feature channels, a substantial proportion of channels exhibit low-magnitude noisy gradients. Standard summation accumulates this background noise, producing scattered, unfocused heatmaps.

To solve this, our inference pipeline (`src/inference.py`) implements **Top-K Channel Pooling**:
1. Global gradients are computed and restricted to positive updates:
   ```
   pooled_grads_pos = max( [ 1 / Z ] * sum_{i,j} [ d(y^c) / d(A^k_{i,j}) ] , 0 )
   ```
2. Only the top 25% highest-magnitude channels are retained:
   ```
   K_top = max( 1, int( Total_Channels * 0.25 ) )
   Top_Indices = argpartition( -pooled_grads_pos, K_top )[:K_top]
   ```
3. Non-top channels are set to zero in the sparse gradient weight vector:
   ```
   weights_sparse[k] = pooled_grads_pos[k]   if k in Top_Indices
   weights_sparse[k] = 0.0                   otherwise
   ```
4. The matrix multiplication produces a sharp, focused heatmap:
   ```
   Heatmap_Focused = sum_{k in Top_Indices} [ weights_sparse[k] * A^k ]
   ```

---

## 4. Grad-CAM++: Multi-Focal Higher-Order Gradient Formulation

Standard Grad-CAM fails when multiple distinct tumor nodules exist in a single slice or when fine-grained boundaries are needed. **Grad-CAM++** (Chattopadhay et al., 2018) resolves this by computing higher-order derivatives (second and third gradients).

### 4.1 Pixel-Wise Weight Derivation
In Grad-CAM++, the channel weight alpha_k^c is generalized into a spatially-weighted pixel coefficient alpha_{k, i, j}^c:

```
alpha_{k, i, j}^c = [ d^2(y^c) / d(A^k_{i, j})^2 ] / [ 2 * d^2(y^c) / d(A^k_{i, j})^2 + sum_{a=1}^u sum_{b=1}^v [ A^k_{a, b} * ( d^3(y^c) / d(A^k_{a, b})^3 ) ] ]
```

### 4.2 Aggregated Channel Weight
The aggregate channel importance is then weighted by the ReLU of the first-order gradient:

```
w_k^c = sum_{i=1}^u sum_{j=1}^v [ alpha_{k, i, j}^c * ReLU( d(y^c) / d(A^k_{i, j}) ) ]
```

### 4.3 Grad-CAM++ Heatmap Generation
```
L_GradCAM++^c = sum_{k=1}^K [ w_k^c * A^k ]
```

Grad-CAM++ provides significantly sharper localization around heterogeneous tumor margins, especially for small pituitary microadenomas.

---

## 5. Brain-Mask Gating (Eliminating Calvarial Skull Artifacts)

A primary flaw in uncalibrated medical XAI is "skull leakage"—heatmaps bleeding into high-contrast calvarial bones, scanner head holders, or ambient air space.

The system enforces strict **Brain-Mask Anatomical Gating**:
1. An intracranial tissue mask Mask_brain(x, y) in {0, 1} is extracted from the raw scan using automated morphological contouring.
2. The resized Grad-CAM heatmap HM(x, y) is multiplied element-wise by the binary mask:
   ```
   HM_gated(x, y) = HM(x, y) * Mask_brain(x, y)
   ```
3. Any activation occurring at coordinate (x, y) outside the cerebral parenchyma is instantly forced to 0.0:
   ```
   HM_gated(x, y) = 0.0    for all (x, y) where Mask_brain(x, y) == 0
   ```

This structural guarantee ensures 100% of reported visual saliency resides strictly inside the brain.

---

## 6. Local Interpretable Model-agnostic Explanations (LIME)

While Grad-CAM explains convolutional feature weighting globally, **LIME** (Ribeiro et al., 2016) provides model-agnostic, pixel-level local explanations by perturbing super-pixel clusters and observing output probability shifts.

### 6.1 LIME Optimization Objective
For an input image x and complex prediction model f, LIME trains an interpretable linear surrogate model g belonging to class G (e.g., sparse linear regression) by minimizing the following objective function:

```
Explanation(x) = argmin_{g in G} [ L( f, g, pi_x ) + Omega(g) ]
```

Where:
- f(z) is the output probability of the deep learning ensemble for perturbed input instance z.
- g(z) is the prediction of the simple local linear surrogate:
  ```
  g(z') = w_0 + sum_{d=1}^D [ w_d * z'_d ]
  ```
  Where z' is a binary vector {0, 1}^D indicating the presence (1) or occlusion (0) of super-pixel d.
- L(f, g, pi_x) is the squared loss measuring local discrepancy between f and g:
  ```
  L( f, g, pi_x ) = sum_{z, z' in Z} [ pi_x(z) * ( f(z) - g(z') )^2 ]
  ```
- Omega(g) is a regularization penalty enforcing model sparsity (limiting explanations to top K = 5 or 10 super-pixels).

### 6.2 Exponential Proximity Kernel Formulation
The local neighborhood weight pi_x(z) measures the cosine or Euclidean distance D(x, z) between the original scan x and the perturbed instance z:

```
pi_x(z) = exp( - [ D(x, z)^2 ] / [ sigma^2 ] )
```

Where sigma represents the kernel bandwidth parameter controlling neighborhood size.

### 6.3 Super-Pixel Segmentation
LIME divides the MRI image into visually coherent super-pixels using the Quickshift or SLIC (Simple Linear Iterative Clustering) algorithm. By turning individual super-pixels on or off (filling occluded segments with mean background gray intensity), LIME identifies the exact contiguous super-pixels that provide positive evidence for the tumor diagnosis, verifying alignment with radiological anatomy.
