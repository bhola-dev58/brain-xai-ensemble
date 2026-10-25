import os
import sys
import numpy as np
import cv2

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASSES, CONVNEXT_VOTE_WEIGHT, INCEPTION_VOTE_WEIGHT, DENSENET_VOTE_WEIGHT
from src.inference import predict_tumor_logic
from src.report_generator import generate_pdf_report
from fastapi.testclient import TestClient
from src.api import app as fastapi_app

def test_full_stack():
    print("=" * 60)
    print("TESTING FULL-STACK BRAINTUMORXAI PIPELINE")
    print("=" * 60)

    # 1. Config Check
    print("\n[1/4] Verifying System Configuration...")
    print(f"  Classes: {CLASSES}")
    print(f"  Ensemble Weights: ConvNeXt={CONVNEXT_VOTE_WEIGHT}, Inception={INCEPTION_VOTE_WEIGHT}, DenseNet={DENSENET_VOTE_WEIGHT}")
    assert len(CLASSES) == 4
    assert np.isclose(CONVNEXT_VOTE_WEIGHT + INCEPTION_VOTE_WEIGHT + DENSENET_VOTE_WEIGHT, 1.0)
    print("  [PASSED] Configuration is valid.")

    # 2. Inference & XAI Check
    print("\n[2/4] Testing Tri-Ensemble & Multimodal XAI Engine...")
    test_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_images", "Tr-me_0025.jpg")
    
    if os.path.exists(test_img_path):
        img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
    else:
        # Create a synthetic brain slice tensor if test image is missing
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(img, (128, 128), 90, (180, 180, 180), -1)
        cv2.circle(img, (110, 100), 25, (240, 240, 240), -1)
    from src.inference import is_ensemble_ready
    if not is_ensemble_ready():
        import pytest
        pytest.skip("Tri-Ensemble model weights not available in models/ — skipping full-stack live weight inference.")

    result = predict_tumor_logic(img)
    assert result["is_valid"] is True
    print(f"  Diagnosis: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  Uncertainty Entropy: {result.get('uncertainty', 0.0):.2f}%")
    print(f"  Location: {result['location']}")
    print(f"  Tumor Area: {result['tumor_percentage']:.2f}%")
    print(f"  Inference Time: {result['inference_time']:.4f}s")
    assert "gradcam_overlay" in result
    assert "gradcam_pp_overlay" in result
    assert "segmentation_img" in result
    print("  [PASSED] Inference and XAI overlays generated successfully.")

    # 3. Clinical PDF Report Generation
    print("\n[3/4] Testing Clinical PDF Report Generator...")
    pdf_path = generate_pdf_report(
        diag_result=result,
        patient_id="TEST-001",
        patient_name="Verification Patient",
        patient_age="52",
        patient_gender="Female"
    )
    assert os.path.exists(pdf_path)
    file_size_kb = os.path.getsize(pdf_path) / 1024
    print(f"  Report generated at: {pdf_path} ({file_size_kb:.1f} KB)")
    print("  [PASSED] PDF Report generated successfully.")

    # 4. FastAPI PACS Workstation Verification
    print("\n[4/4] Testing FastAPI PACS Workstation...")
    client = TestClient(fastapi_app)
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    home_resp = client.get("/")
    assert home_resp.status_code == 200
    assert "BrainTumor" in home_resp.text
    print("  [PASSED] FastAPI PACS Workstation endpoints verified cleanly.")

    print("\n" + "=" * 60)
    print("ALL FULL-STACK TESTS PASSED SUCCESSFULLY! 🚀")
    print("=" * 60)

if __name__ == "__main__":
    test_full_stack()
