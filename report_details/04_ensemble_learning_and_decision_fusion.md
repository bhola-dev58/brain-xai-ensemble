# Ensemble Learning and Decision Fusion

## 1. Theoretical Rationale for Ensemble Architecture

In complex neuro-oncological diagnostic tasks, single standalone deep neural networks exhibit unique inductive biases and localized error distributions:
- **DenseNet121** excels at maintaining fine spatial gradients and identifying compact, localized anomalies, but occasionally exhibits ambiguity when differentiating diffuse tumors from surrounding vasogenic edema.
- **InceptionV3** captures wide receptive field interactions and macroscopic tumor boundaries via multi-scale factorized convolutions, but may produce lower spatial resolution activation maps for minute micro-lesions.

By combining the probabilistic outputs of these distinct architectures, ensemble learning reduces the variance of individual estimators without increasing their bias, directly adhering to the statistical principles of the Condorcet Jury Theorem.

---

## 2. Mathematical Formulation of Soft Voting

Ensemble fusion can be executed via Hard Voting (majority class labels) or Soft Voting (averaging predicted probability distributions). Soft voting is strictly preferred in medical applications because it preserves continuous confidence signals and penalizes uncalibrated overconfidence.

### 2.1 Unweighted (Equal) Soft Voting
In the initial baseline and inference implementation (`src/inference.py`), both models contribute equally to the diagnostic consensus:

```
P_ensemble(c) = [ 1 / M ] * sum_{m=1}^M P_m(c)
```

For our dual-backbone system (M = 2):

```
P_ensemble(c) = 0.5 * P_DenseNet(c) + 0.5 * P_Inception(c)
```

Where:
- P_DenseNet(c) is the softmax probability output by DenseNet121 for class c.
- P_Inception(c) is the softmax probability output by InceptionV3 for class c.
- c in {0, 1, 2, 3} corresponds to {No Tumor, Glioma, Meningioma, Pituitary}.

### 2.2 Performance-Weighted Soft Voting
In advanced configurations (such as Phase 3 fine-tuning where InceptionV3 achieves 95.51% standalone accuracy versus DenseNet121's 92.83%), models are weighted proportionally to their cross-validated F1-scores:

```
P_weighted(c) = sum_{m=1}^M [ w_m * P_m(c) ] / [ sum_{m=1}^M w_m ]
```

With the dual-backbone weighting scheme:
- w_Inception = 0.70
- w_DenseNet = 0.30

```
P_weighted(c) = [ 0.70 * P_Inception(c) ] + [ 0.30 * P_DenseNet(c) ]
```

(Note: In the later Tri-Ensemble expansion, weights were structured as ConvNeXt: 0.45, InceptionV3: 0.35, DenseNet121: 0.20).

### 2.3 Diagnostic Class Selection and Confidence Extraction
The final predicted diagnostic class c_hat is selected using the argmax operator:

```
c_hat = argmax_{c in {0, 1, 2, 3}} [ P_ensemble(c) ]
```

The accompanying diagnostic confidence score represents the maximum predicted posterior probability, expressed as a percentage:

```
Confidence_Percentage = max_{c in {0, 1, 2, 3}} [ P_ensemble(c) ] * 100.0%
```

A binary tumor indicator flags the scan for neurosurgical review:

```
is_tumor = True    if c_hat in {1, 2, 3}
is_tumor = False   if c_hat == 0 (No Tumor)
```

---

## 3. Test-Time Augmentation (TTA) Decision Fusion

During standard test inference, a single static orientation of an MRI scan is evaluated. In clinical reality, patient orientation within the magnetic bore may vary by several degrees. **Test-Time Augmentation (TTA)** simulates multi-angle observation by generating K stochastic, anatomically valid variations of the test scan and computing the ensemble consensus over all augmented views.

```
P_TTA(c) = [ 1 / K ] * sum_{k=1}^K P_ensemble( c | T_k(x) )
```

Where:
- x is the input MRI image.
- T_k represents the k-th stochastic spatial transformation sampled from:
  1. Horizontal reflection: T_flip(x)
  2. Small positive rotation: T_rot+10(x)
  3. Small negative rotation: T_rot-10(x)
  4. Subtle zoom scaling: T_zoom(x) in [0.95, 1.05]
  5. Identity (unaltered) scan: T_identity(x)
- K is the number of TTA passes (typically K = 5 or K = 10 passes).

### Clinical Benefit of TTA:
TTA smooths out boundary prediction fluctuations caused by scanner quantization artifacts. In our benchmarks, evaluating the dual ensemble with 5-pass TTA pushed overall diagnostic accuracy from 95.59% to 95.82%, and on specific balanced sub-splits, reached peak benchmark accuracies approaching 98.2% to 98.88%.

---

## 4. Shannon Entropy and Prediction Uncertainty Quantification

In high-stakes medical robotics and AI, knowing **when the model does not know** is as critical as high accuracy. To quantify diagnostic uncertainty, the framework calculates the **Normalized Shannon Entropy** across the predicted probability distribution:

### 4.1 Shannon Entropy Formulation
```
H(P) = - sum_{c=1}^C [ P_ensemble(c) * log_2( P_ensemble(c) + epsilon ) ]
```

Where:
- C = 4 classes.
- epsilon = 1e-12 (a minute positive value preventing undefined log_2(0) computations).

### 4.2 Maximum Theoretical Entropy
For a discrete 4-class uniform distribution where the model is completely guessing (P(c) = 0.25 for all c):

```
H_max = log_2(C) = log_2(4) = 2.0 bits
```

### 4.3 Normalized Uncertainty Metric
The entropy is scaled to a standard percentage range [0.0% to 100.0%]:

```
Uncertainty_Percentage = [ H(P) / H_max ] * 100.0%
```

### 4.4 Clinical Interpretation Matrix

| Uncertainty Range | Diagnostic Confidence | Clinical Action Recommendation |
|---|---|---|
| 0.0% - 15.0% | Very High | Prediction is definitive; suitable for rapid automated triage. |
| 15.1% - 35.0% | Moderate | Clear winning class, but secondary class exhibits mild probability. |
| 35.1% - 60.0% | High Uncertainty | Ambiguous scan; possible overlapping meningioma/pituitary boundary. |
| > 60.0% | Extreme Ambiguity | Automatic safety alert triggered: Flagged for mandatory manual review by senior radiologist. |
