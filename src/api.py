"""
FastAPI REST backend for BrainTumorXAI.

Endpoints:
    GET  /api/health   — Liveness check; confirms models are loaded.
    GET  /api/models   — Architecture telemetry and active ensemble weights.
    POST /api/predict  — Accepts an MRI image and returns diagnosis JSON with Grad-CAM and Grad-CAM++ base64 images.
    POST /api/report   — Accepts an MRI image + patient metadata and returns downloadable PDF report.

Usage:
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""
import base64
import io
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from src.config import (
    CONVNEXT_VOTE_WEIGHT,
    INCEPTION_VOTE_WEIGHT,
    DENSENET_VOTE_WEIGHT,
    CLASSES,
    logger
)
from src.inference import predict_tumor_logic, model_convnext, model_inc, model_dense
from src.report_generator import generate_pdf_report

# ------------------------------------------------------------------ #
# Application factory
# ------------------------------------------------------------------ #
app = FastAPI(
    title="BrainTumorXAI Clinical AI API",
    description=(
        "REST API for the Explainable Brain Tumor Diagnostic System. "
        "Provides Tri-Ensemble (ConvNeXtSmall + InceptionV3 + DenseNet121) classification, "
        "Grad-CAM / Grad-CAM++ explainability, uncertainty estimation, and PDF report generation."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _ndarray_to_base64(img: np.ndarray) -> str:
    """Encodes a numpy image array to a base64 PNG string for JSON transport."""
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def _read_upload_as_numpy(file: UploadFile) -> np.ndarray:
    """Reads an uploaded image file and converts it to a uint8 RGB numpy array."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode the uploaded file as a valid image. "
                   "Please upload a valid JPEG or PNG MRI scan.",
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #
@app.get(
    "/api/health",
    summary="Health check",
    description="Returns server status and active ensemble status.",
    tags=["System"],
)
def health_check() -> dict:
    """Liveness endpoint."""
    return {
        "status": "ok",
        "models_active": {
            "ConvNeXtSmall": model_convnext is not None,
            "InceptionV3": model_inc is not None,
            "DenseNet121": model_dense is not None,
        },
        "pipeline_version": "3.0.0 (Tri-Ensemble)",
    }


@app.get(
    "/api/models",
    summary="Model Ensemble Telemetry",
    description="Returns active model architectures and dynamic ensemble voting weights.",
    tags=["System"],
)
def get_models_info() -> dict:
    """Telemetry endpoint."""
    return {
        "architecture": "Tri-Ensemble Deep Convolutional Framework",
        "classes": CLASSES,
        "weights": {
            "ConvNeXtSmall": CONVNEXT_VOTE_WEIGHT,
            "InceptionV3": INCEPTION_VOTE_WEIGHT,
            "DenseNet121": DENSENET_VOTE_WEIGHT,
        },
        "explainable_ai": ["Grad-CAM", "Grad-CAM++", "Morphological Contour Sizing"],
    }


@app.post(
    "/api/predict",
    summary="Predict brain tumor from MRI",
    description=(
        "Upload a brain MRI image (JPEG/PNG). "
        "Returns classification result, confidence score, uncertainty entropy, "
        "individual model votes, and base64-encoded Grad-CAM and Grad-CAM++ overlays."
    ),
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(..., description="Brain MRI image file (JPEG or PNG)"),
) -> JSONResponse:
    """Performs full Tri-Ensemble inference with multi-modal XAI overlays."""
    logger.info(f"Received prediction request: filename={file.filename}")

    img = await _read_upload_as_numpy(file)
    result = predict_tumor_logic(img)

    if not result.get("is_valid"):
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed."))

    gradcam_b64 = _ndarray_to_base64(result["gradcam_overlay"])
    gradcam_pp_b64 = _ndarray_to_base64(result["gradcam_pp_overlay"])
    seg_b64 = _ndarray_to_base64(result["segmentation_img"])

    pred_breakdown = {}
    for cls, prob in zip(result["classes"], result["avg_pred"]):
        pred_breakdown[cls] = round(float(prob) * 100, 4)

    return JSONResponse(
        status_code=200,
        content={
            "class_name": result["class_name"],
            "confidence": round(result["confidence"], 2),
            "uncertainty_entropy": round(result.get("uncertainty", 0.0), 2),
            "is_tumor": result["is_tumor"],
            "location": result["location"],
            "tumor_percentage": round(result["tumor_percentage"], 2),
            "severity": result["severity"],
            "severity_color": result["severity_color"],
            "inference_time_seconds": round(result["inference_time"], 4),
            "prediction_breakdown": pred_breakdown,
            "gradcam_overlay_b64": gradcam_b64,
            "gradcam_pp_overlay_b64": gradcam_pp_b64,
            "segmentation_b64": seg_b64,
        },
    )


@app.post(
    "/api/report",
    summary="Generate Clinical PDF Diagnostic Report",
    description="Upload an MRI image with patient details and receive a hospital-grade PDF report.",
    tags=["Reports"],
)
async def generate_report(
    file: UploadFile = File(..., description="Brain MRI image file (JPEG or PNG)"),
    patient_id: str = Form("PT-8942"),
    patient_name: str = Form("Anonymous Patient"),
    patient_age: str = Form("45"),
    patient_gender: str = Form("Unspecified"),
):
    """Generates and streams a clinical diagnostic PDF report."""
    logger.info(f"Generating PDF report for patient: {patient_id}")
    
    img = await _read_upload_as_numpy(file)
    diag_result = predict_tumor_logic(img)

    if not diag_result.get("is_valid"):
        raise HTTPException(status_code=500, detail="Failed to process image for report.")

    pdf_path = generate_pdf_report(
        diag_result=diag_result,
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender
    )

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="PDF generation failed on server.")

    return FileResponse(
        path=pdf_path,
        filename=os.path.basename(pdf_path),
        media_type="application/pdf"
    )


# ------------------------------------------------------------------ #
# Mount Static Frontend (UI/UX Pro Max Workstation)
# ------------------------------------------------------------------ #
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

