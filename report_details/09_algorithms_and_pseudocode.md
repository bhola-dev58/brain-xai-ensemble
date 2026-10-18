# Complete Algorithms and Pseudocode

This document details the formal algorithmic procedures powering each module of the Computer-Aided Diagnosis (CAD) framework, written in structured, step-by-step pseudocode.

---

## Algorithm 1: Automated Brain Tissue Extraction and Preprocessing

```text
Input:  Raw input MRI image I_raw of dimensions (H_raw, W_raw, C)
Output: Cropped and normalized tensor X_norm of dimensions (H_target, W_target, 3), Binary Brain Mask Mask_brain

Procedure ExtractAndPreprocess(I_raw, target_size):
    1: If I_raw has 3 color channels:
    2:     I_gray = 0.299 * I_raw[:,:,0] + 0.587 * I_raw[:,:,1] + 0.114 * I_raw[:,:,2]
    3: Else:
    4:     I_gray = Copy(I_raw)
    5: End If

    6: // Step 1: Smooth scanner noise
    7: I_smooth = GaussianBlur(I_gray, kernel_size=(5, 5), sigma=1.5)

    8: // Step 2: Binary thresholding to segment brain matter from void
    9: Threshold_val = 10
   10: I_thresh = Zeros_Like(I_gray)
   11: For each pixel (x, y):
   12:     If I_smooth(x, y) > Threshold_val:
   13:         I_thresh(x, y) = 255
   14:     End If
   15: End For

   16: // Step 3: Find external contours
   17: Contours = FindContours(I_thresh, mode=EXTERNAL, method=CHAIN_APPROX_SIMPLE)
   18: If Contours is empty:
   19:     Return Resize(I_raw, target_size) / 255.0, Ones_Like(I_gray) * 255
   20: End If

   21: // Step 4: Extract the largest contour (the intracranial vault)
   22: C_max = argmax_{c in Contours} [ ContourArea(c) ]

   23: // Step 5: Compute extreme Cartesian extremities
   24: x_min = min_{point in C_max} [ point.x ]
   25: x_max = max_{point in C_max} [ point.x ]
   26: y_min = min_{point in C_max} [ point.y ]
   27: y_max = max_{point in C_max} [ point.y ]

   28: // Step 6: Crop skull background margins
   29: I_cropped = I_raw[ y_min : y_max, x_min : x_max ]

   30: // Step 7: Build binary brain mask
   31: Mask_brain = Zeros_Like(I_gray)
   32: DrawContours(Mask_brain, [C_max], color=255, thickness=FILLED)
   33: Kernel_close = EllipticalKernel(11, 11)
   34: Mask_brain = MorphologicalClose(Mask_brain, Kernel_close)
   35: Mask_brain_cropped = Mask_brain[ y_min : y_max, x_min : x_max ]

   36: // Step 8: Multi-scale spatial resize and float normalization
   37: I_resized = BicubicResize(I_cropped, target_size)
   38: X_norm = I_resized / 255.0

   39: Return X_norm, Mask_brain_cropped
End Procedure
```

---

## Algorithm 2: 3-Phase Progressive Transfer Learning Pipeline

```text
Input:  Pre-trained CNN Backbones (DenseNet121, InceptionV3), Augmented Training Set D_train, Validation Set D_val
Output: Optimized Fine-Tuned Checkpoints (DenseNet_best.keras, Inception_best.keras)

Procedure ProgressiveTrainingPipeline(Backbone, D_train, D_val):
    1: // ==========================================
    2: // Phase 1: Feature Adapter Warmup
    3: // ==========================================
    4: Set Backbone.trainable = False  // Freeze 100% of convolutional weights
    5: Head = GlobalAveragePooling2D(Backbone.output)
    6: Head = BatchNormalization()(Head)
    7: Head = Dense(512, activation='relu', kernel_regularizer=L2(0.0001))(Head)
    8: Head = Dropout(rate=0.25)(Head)
    9: Head = Dense(256, activation='relu', kernel_regularizer=L2(0.0001))(Head)
   10: Head = Dropout(rate=0.25)(Head)
   11: Outputs = Dense(4, activation='softmax')(Head)
   12: Model = CreateModel(inputs=Backbone.input, outputs=Outputs)

   13: Compile Model with:
   14:     Optimizer = Adam(learning_rate=1e-3, amsgrad=True)
   15:     Loss = CategoricalCrossentropy(label_smoothing=0.05)
   16:     Class_Weights = ComputeBalancedWeights(D_train)

   17: Train Model on D_train for 25 epochs with callbacks (ModelCheckpoint, EarlyStopping patience=6)
   18: Load best weights from Phase 1 checkpoint

   19: // ==========================================
   20: // Phase 2: High-Level Representation Adaptation
   21: // ==========================================
   22: Set Backbone.trainable = True
   23: For layer in Backbone.layers[:-60]:
   24:     Set layer.trainable = False  // Keep lower edge detectors frozen
   25: End For
   26: Recompile Model with:
   27:     Optimizer = Adam(learning_rate=1e-5, amsgrad=True)
   28:     Loss = CategoricalCrossentropy(label_smoothing=0.03)

   29: Train Model for 35 epochs with ReduceLROnPlateau(factor=0.4, patience=4)
   30: Load best weights from Phase 2 checkpoint

   31: // ==========================================
   32: // Phase 3: Deep Aggressive Fine-Tuning
   33: // ==========================================
   34: For layer in Backbone.layers:
   35:     Set layer.trainable = True  // Unlock all layers across network hierarchy
   36: End For
   37: Recompile Model with:
   38:     Optimizer = Adam(learning_rate=3e-6, amsgrad=True)
   39:     Loss = CategoricalCrossentropy(label_smoothing=0.02)

   40: Train Model for up to 80 epochs with EarlyStopping(patience=15, restore_best_weights=True)
   41: Save best model checkpoint: SaveModel(Model, f"{Backbone.name}_best.keras")
   42: Return Model
End Procedure
```

---

## Algorithm 3: Soft-Voting Ensemble Inference and TTA Decision Fusion

```text
Input:  Preprocessed scan X_norm, Models M_dense, M_inc, Weights w_dense=0.5, w_inc=0.5, TTA Passes K
Output: Predicted Class c_hat, Confidence Score, Shannon Uncertainty %, Probability Distribution P_ensemble

Procedure SoftVotingInference(X_norm, M_dense, M_inc, K=1):
    1: If K <= 1: // Standard Single-Pass Inference
    2:     P_dense = M_dense.predict(X_norm)   // Shape: (1, 4)
    3:     P_inc   = M_inc.predict(X_norm)     // Shape: (1, 4)
    4:     P_ensemble = (w_dense * P_dense + w_inc * P_inc) / (w_dense + w_inc)
    5: Else:        // Multi-Pass Test-Time Augmentation (TTA)
    6:     Predictions_Accum = Zeros(4)
    7:     For pass = 1 to K:
    8:         X_aug = ApplyRandomAffineTransform(X_norm, rot_range=15, flip=True, zoom_range=0.10)
    9:         P_d = M_dense.predict(X_aug)
   10:         P_i = M_inc.predict(X_aug)
   11:         P_pass = (w_dense * P_d + w_inc * P_i) / (w_dense + w_inc)
   12:         Predictions_Accum = Predictions_Accum + P_pass
   13:     End For
   14:     P_ensemble = Predictions_Accum / K
   15: End If

   16: // Decision selection
   17: c_hat = argmax_{c in {0, 1, 2, 3}} [ P_ensemble[c] ]
   18: Confidence = max_{c in {0, 1, 2, 3}} [ P_ensemble[c] ] * 100.0%

   19: // Calculate Normalized Shannon Entropy (Uncertainty)
   20: Epsilon = 1e-12
   21: P_clipped = Clip(P_ensemble, Epsilon, 1.0)
   22: Entropy = - sum_{c=0}^3 [ P_clipped[c] * log2(P_clipped[c]) ]
   23: Max_Entropy = log2(4) = 2.0
   24: Uncertainty = (Entropy / Max_Entropy) * 100.0%

   25: Return c_hat, Confidence, Uncertainty, P_ensemble
End Procedure
```

---

## Algorithm 4: Intracranial Brain-Gated Grad-CAM Heatmap Generation

```text
Input:  Model M_dense, Input Tensor X, Target Class Index c, Original Scan I_orig, Binary Mask Mask_brain
Output: Cleaned Intracranial Heatmap HM_clean of dimensions (H_orig, W_orig)

Procedure GenerateGatedGradCAM(M_dense, X, c, I_orig, Mask_brain):
    1: Locate final convolutional layer L_conv in M_dense (e.g. 'relu' in DenseNet121)
    2: Construct Grad_Model = SubModel(inputs=M_dense.input, outputs=[L_conv.output, M_dense.output])

    3: With GradientTape() as Tape:
    4:     Tape.watch(X)
    5:     Conv_Outputs, Predictions = Grad_Model(X)
    6:     Tape.watch(Conv_Outputs)
    7:     If c is None:
    8:         c = argmax(Predictions[0])
    9:     End If
   10:     Target_Class_Score = Predictions[:, c]
   11: End With

   12: Gradients = Tape.gradient(Target_Class_Score, Conv_Outputs) // Shape: (1, U, V, Channels)

   13: // Step 1: Global Average Pooling of positive gradients
   14: Pooled_Grads = Mean(Gradients, axis=(0, 1, 2))              // Shape: (Channels,)
   15: Pooled_Grads_Pos = Maximum(Pooled_Grads, 0.0)

   16: // Step 2: Top-K Channel Pooling (Retain top 25% channels)
   17: Num_Channels = Length(Pooled_Grads_Pos)
   18: K_top = max(1, int(Num_Channels * 0.25))
   19: Top_Indices = TopKIndices(Pooled_Grads_Pos, k=K_top)
   20: Weights_Sparse = Zeros_Like(Pooled_Grads_Pos)
   21: For idx in Top_Indices:
   22:     Weights_Sparse[idx] = Pooled_Grads_Pos[idx]
   23: End For

   24: // Step 3: Weighted linear combination of feature maps
   25: Heatmap_Raw = Sum_{ch in Top_Indices} [ Conv_Outputs[0, :, :, ch] * Weights_Sparse[ch] ]
   26: Heatmap_ReLU = Maximum(Heatmap_Raw, 0.0)
   27: Heatmap_Norm = Heatmap_ReLU / (max(Heatmap_ReLU) + 1e-8)

   28: // Step 4: Bicubic interpolation to original resolution
   29: Heatmap_Resized = BicubicResize(Heatmap_Norm, size=(W_orig, H_orig))

   30: // Step 5: Brain-Mask Gating (Forced suppression of outside-skull activations)
   31: Heatmap_Gated = Heatmap_Resized * (Mask_brain > 0)

   32: // Step 6: Adaptive Otsu thresholding on brain interior activations
   33: Brain_Vals = Heatmap_Gated[ Mask_brain > 0 ]
   34: Otsu_Threshold = ComputeOtsuThreshold(Brain_Vals * 255.0) / 255.0
   35: Adaptive_Cutoff = max(Otsu_Threshold, 0.40)
   36: Heatmap_Thresholded = Where(Heatmap_Gated > Adaptive_Cutoff, Heatmap_Gated, 0.0)

   37: // Step 7: Gaussian edge smoothing
   38: HM_clean = GaussianBlur(Heatmap_Thresholded, kernel_size=(7, 7), sigma=2.0)
   39: HM_clean = HM_clean / (max(HM_clean) + 1e-8)

   40: Return HM_clean
End Procedure
```

---

## Algorithm 5: Weakly-Supervised Morphological Segmentation

```text
Input:  Cleaned Heatmap HM_clean, Original Image I_orig, Binary Brain Mask Mask_brain
Output: Segmented Image I_overlay, Binary Tumor Mask Mask_tumor, Intracranial Area Percentage rho_tumor

Procedure WeaklySupervisedSegmentation(HM_clean, I_orig, Mask_brain):
    1: (H, W) = Dimensions(I_orig)
    2: Brain_Pixels_Total = CountNonZero(Mask_brain)
    3: If Brain_Pixels_Total == 0:
    4:     Brain_Pixels_Total = H * W
    5: End If

    6: // Step 1: Extract candidate ROI from 70th percentile of brain activations
    7: Brain_HM = HM_clean[ Mask_brain > 0 ]
    8: Tau_ROI = max( Percentile(Brain_HM, 70), 0.45 )
    9: Mask_ROI = (HM_clean > Tau_ROI) and (Mask_brain > 0)

   10: // Close gaps and keep largest connected component
   11: Kernel_ROI = EllipticalKernel(13, 13)
   12: Mask_ROI = MorphologicalClose(Mask_ROI, Kernel_ROI)
   13: Mask_ROI = KeepLargestComponent(Mask_ROI)

   14: // Step 2: Otsu thresholding within candidate ROI
   15: I_gray = Grayscale(I_orig)
   16: Pixels_ROI = I_gray[ Mask_ROI > 0 ]
   17: If Length(Pixels_ROI) < 50:
   18:     Return I_orig, Zeros(H, W), 0.0%
   19: End If
   20: Otsu_Cutoff = OtsuThreshold(Pixels_ROI)

   21: // Step 3: Segment hyperintense contrast-enhanced mass
   22: Mask_Tumor_Raw = (I_gray > Otsu_Cutoff) and (Mask_ROI > 0) and (Mask_brain > 0)

   23: // Step 4: Morphological opening (noise removal) & closing (hole bridge)
   24: Kernel_Clean = EllipticalKernel(5, 5)
   25: Mask_Clean = MorphologicalOpen(Mask_Tumor_Raw, Kernel_Clean, iterations=1)
   26: Mask_Clean = MorphologicalClose(Mask_Clean, Kernel_Clean, iterations=3)
   27: Mask_Tumor = KeepLargestComponent(Mask_Clean)

   28: // Smooth edges
   29: Mask_Tumor = GaussianBlur(Mask_Tumor, (9, 9), sigma=2.0)
   30: Mask_Tumor = (Mask_Tumor >= 127) * 255

   31: // Step 5: Calculate relative intracranial tumor burden
   32: Tumor_Pixels = CountNonZero(Mask_Tumor)
   33: rho_tumor = (Tumor_Pixels / Brain_Pixels_Total) * 100.0%

   34: // Fallback safety guard for degenerate segments
   35: If rho_tumor == 0.0% or rho_tumor > 30.0%:
   36:     Mask_Tumor = Mask_ROI
   37:     rho_tumor = (CountNonZero(Mask_Tumor) / Brain_Pixels_Total) * 100.0%
   38: End If

   39: // Step 6: Alpha-blended medical red overlay
   40: I_overlay = Copy(I_orig)
   41: Alpha = 0.45
   42: For (x, y) where Mask_Tumor(x, y) > 0:
   43:     I_overlay(x, y) = (1.0 - Alpha) * I_orig(x, y) + Alpha * [255, 0, 0]
   44: End For

   45: Return I_overlay, Mask_Tumor, rho_tumor
End Procedure
```

---

## Algorithm 6: Clinical Decision Rules and PDF Report Synthesis

```text
Input:  Ensemble Confidence %, Tumor Percentage rho_tumor, Peak Saliency (y_peak, x_peak), Patient Info
Output: Severity Tier, Severity Color, Anatomical Location, Generated PDF Report Path

Procedure ClinicalDecisionAndReport(Confidence, rho_tumor, y_peak, x_peak, Patient_Dict):
    1: // Severity evaluation
    2: If Confidence < 55.0%:
    3:     Severity = "Borderline"
    4:     Color = "#F59E0B"
    5: Else If Confidence > 95.0% and rho_tumor > 5.0%:
    6:     Severity = "High"
    7:     Color = "#EF4444"
    8: Else If Confidence > 80.0% and rho_tumor > 3.0%:
    9:     Severity = "Moderate"
   10:     Color = "#F97316"
   11: Else If Confidence > 60.0%:
   12:     Severity = "Low"
   13:     Color = "#10B981"
   14: Else:
   15:     Severity = "Uncertain"
   16:     Color = "#6B7280"
   17: End If

   18: // Anatomical location mapping
   19: Hemisphere = (x_peak < W / 2) ? "Left" : "Right"
   20: Vertical   = (y_peak < H / 2) ? "Superior" : "Inferior"
   21: If y_peak < 0.40 * H:
   22:     Lobe = "Frontal Lobe"
   23: Else If y_peak < 0.70 * H:
   24:     Lobe = "Parietal Lobe"
   25: Else:
   26:     Lobe = "Occipital Lobe"
   27: End If
   28: Location = Hemisphere + " " + Lobe + " (" + Vertical + ")"

   29: // Database logging
   30: InsertIntoDatabase(Patient_Dict, Confidence, rho_tumor, Severity, Location)

   31: // Generate ReportLab PDF Document
   32: PDF_Path = BuildRadiologyPDF(
   33:     Patient_Dict,
   34:     Severity, Color, Location,
   35:     Images=[Original_Scan, Skull_Stripped, GradCAM_Overlay, Segmentation_Overlay]
   36: )

   37: Return Severity, Color, Location, PDF_Path
End Procedure
```
