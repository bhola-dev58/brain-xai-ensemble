"""
Script to compute Quantitative Segmentation Performance Metrics:
- Dice Coefficient
- Mean IoU (Jaccard Index)
- Sensitivity (Recall)
- Specificity
across paired ground truth masks and predicted Grad-CAM masks.
"""

import os
import sys
import glob
import numpy as np
import cv2
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_tumor_logic, make_gradcam_heatmap, model_dense
from src.processor import create_segmentation, compute_dice_coefficient, compute_iou

def compute_single_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(bool)
    gt = (gt_mask > 0).astype(bool)

    TP = int(np.sum(pred & gt))
    FP = int(np.sum(pred & ~gt))
    FN = int(np.sum(~pred & gt))
    TN = int(np.sum(~pred & ~gt))

    dice = compute_dice_coefficient(gt_mask, pred_mask)
    iou = compute_iou(gt_mask, pred_mask)
    sensitivity = TP / (TP + FN + 1e-7)
    specificity = TN / (TN + FP + 1e-7)

    return dice, iou, sensitivity, specificity

def evaluate_segmentation():
    base_img_dir = "datasets/image"
    base_mask_dir = "datasets/mask"

    classes = ['1', '2', '3'] # Tumor classes: Glioma, Meningioma, Pituitary
    dice_list = []
    iou_list = []
    sens_list = []
    spec_list = []

    per_class_metrics = {c: {'dice': [], 'iou': [], 'sens': [], 'spec': []} for c in classes}

    print("Evaluating Grad-CAM Guided Morphological Segmentation on Ground Truth Masks...")
    
    total_evaluated = 0

    for c in classes:
        img_folder = os.path.join(base_img_dir, c)
        mask_folder = os.path.join(base_mask_dir, c)

        if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
            continue

        images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")) + glob.glob(os.path.join(img_folder, "*.png")))
        # Use validation slice (last 30% of each folder for reproducibility)
        val_count = int(len(images) * 0.30)
        val_images = images[-val_count:]

        print(f"Processing Class {c}: {len(val_images)} validation scans...")

        for img_path in val_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Look for mask with _m or same name
            mask_path = os.path.join(mask_folder, f"{base_name}_m.jpg")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_folder, f"{base_name}_m.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_folder, f"{base_name}.jpg")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_folder, f"{base_name}.png")

            if not os.path.exists(mask_path):
                continue

            # Read image and ground truth mask
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue

            # Generate prediction & segmentation mask
            h, w = img_rgb.shape[:2]
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

            heatmap_raw, brain_mask = make_gradcam_heatmap(img_array, original_img=img_rgb)
            _, pred_mask, _ = create_segmentation(heatmap_raw, img_rgb, brain_mask=brain_mask)

            # Ensure same dimensions
            gt_mask_binary = (cv2.resize(gt_mask, (w, h)) > 127).astype(np.uint8)
            pred_mask_binary = (pred_mask > 127).astype(np.uint8)

            d, i, s, sp = compute_single_metrics(pred_mask_binary, gt_mask_binary)

            dice_list.append(d)
            iou_list.append(i)
            sens_list.append(s)
            spec_list.append(sp)

            per_class_metrics[c]['dice'].append(d)
            per_class_metrics[c]['iou'].append(i)
            per_class_metrics[c]['sens'].append(s)
            per_class_metrics[c]['spec'].append(sp)
            total_evaluated += 1

    print(f"\nCompleted evaluation on {total_evaluated} tumor validation masks.\n")

    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(iou_list)
    mean_sens = np.mean(sens_list)
    mean_spec = np.mean(spec_list)

    print("=========================================================================")
    print("        QUANTITATIVE SEGMENTATION PERFORMANCE EVALUATION RESULTS")
    print("=========================================================================")
    print(f" Mean Dice Coefficient:  {mean_dice:.4f}  ({mean_dice*100:.2f}%)")
    print(f" Mean IoU (Jaccard):     {mean_iou:.4f}  ({mean_iou*100:.2f}%)")
    print(f" Sensitivity (Recall):   {mean_sens:.4f}  ({mean_sens*100:.2f}%)")
    print(f" Specificity:            {mean_spec:.4f}  ({mean_spec*100:.2f}%)")
    print("=========================================================================\n")

    class_names = {'1': 'Glioma Tumor', '2': 'Meningioma Tumor', '3': 'Pituitary Tumor'}
    for c in classes:
        if len(per_class_metrics[c]['dice']) > 0:
            print(f"--- {class_names.get(c, c)} (N={len(per_class_metrics[c]['dice'])}) ---")
            print(f"  Dice: {np.mean(per_class_metrics[c]['dice']):.4f} | IoU: {np.mean(per_class_metrics[c]['iou']):.4f} | Sens: {np.mean(per_class_metrics[c]['sens']):.4f} | Spec: {np.mean(per_class_metrics[c]['spec']):.4f}")

if __name__ == "__main__":
    evaluate_segmentation()
