"""
FastAPI REST backend for BrainTumorXAI.

Endpoints:
    GET  /api/health               — Liveness and model readiness probe (public).
    GET  /api/models               — Architecture telemetry and ensemble weights (public).
    POST /api/predict              — Brain MRI classification with Grad-CAM and Grad-CAM++ overlays (protected).
    POST /api/report               — Generates downloadable clinical PDF diagnostic report (protected).
    GET  /api/reports/{analysis_id} — Securely retrieves a previously generated report by analysis ID (protected).
    GET  /api/history              — Securely retrieves paginated patient case history records (protected).

Usage:
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""
import base64
import io
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from src.config import (
    CONVNEXT_VOTE_WEIGHT,
    INCEPTION_VOTE_WEIGHT,
    DENSENET_VOTE_WEIGHT,
    CLASSES,
    REPORTS_DIR,
    HOST,
    PORT,
    logger
)
from src.logger import request_id_ctx
from src.inference import (
    predict_tumor_logic,
    is_ensemble_ready,
    get_model_status,
)
from src.report_generator import generate_pdf_report
from src.database import (
    log_scan,
    get_scan_by_analysis_id,
    get_scan_provenance,
    get_scans_paginated,
    get_scan_count,
)
from src.security import (
    validate_and_read_upload,
    validate_image_content,
    validate_image_dimensions,
    sanitize_filename,
    sanitize_patient_metadata,
    validate_analysis_id,
    require_api_key,
)

# ------------------------------------------------------------------ #
# Application Factory, Lifespan, and Settings
# ------------------------------------------------------------------ #
ENABLE_DOCS = os.getenv("ENABLE_API_DOCS", "true").lower() in ("true", "1", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle events with
    operational diagnostic logging.
    """
    ready = is_ensemble_ready()
    status_map = get_model_status()
    active_count = sum(status_map.values())
    logger.info(
        f"BrainTumorXAI API starting up. Host={HOST}, Port={PORT}, "
        f"Ensemble ready={ready} ({active_count}/3 models active: {status_map})"
    )
    yield
    logger.info("BrainTumorXAI API shutting down cleanly.")


app = FastAPI(
    title="BrainTumorXAI Clinical AI API",
    description=(
        "REST API for the Explainable Brain Tumor Diagnostic System. "
        "Provides Tri-Ensemble (ConvNeXtSmall + InceptionV3 + DenseNet121) classification, "
        "Grad-CAM / Grad-CAM++ explainability, uncertainty estimation, and PDF report generation."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# ------------------------------------------------------------------ #
# Request Observability & Tracing Middleware
# ------------------------------------------------------------------ #
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Tracks request lifecycle, assigns and propagates X-Request-ID,
    measures request processing duration, and securely logs status codes.
    Excludes query parameters, body contents, and patient PHI from logs.
    """
    # 1. Extract or generate unique request ID
    req_id = request.headers.get("X-Request-ID")
    if not req_id or not re.match(r"^[a-zA-Z0-9_\-]{8,64}$", req_id):
        req_id = str(uuid.uuid4())

    token = request_id_ctx.set(req_id)
    start_time = time.perf_counter()

    # Route and method (query strings explicitly excluded for privacy)
    method = request.method
    path = request.url.path

    try:
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id
        logger.info(
            f"{method} {path} status={response.status_code} duration={elapsed_ms:.2f}ms"
        )
        return response
    except Exception as unhandled_err:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            f"{method} {path} status=500 duration={elapsed_ms:.2f}ms error={unhandled_err}",
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred while processing the request.",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )
    finally:
        request_id_ctx.reset(token)


# Configurable CORS Settings
_cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors_env:
    if _cors_env == "*":
        allowed_origins = ["*"]
    else:
        allowed_origins = [orig.strip() for orig in _cors_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
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
    """
    Safely reads and decodes an uploaded medical scan file.
    Enforces maximum size bounds, magic byte checks, format validation,
    and dimension limits to prevent DoS and memory exhaustion.
    """
    # 1. Enforce size limits and non-empty file
    contents = await validate_and_read_upload(file)

    # 2. Inspect magic bytes / file signature
    detected_format = validate_image_content(contents)

    # 3. Decode based on format
    if detected_format == "DICOM":
        try:
            import pydicom
            from src.dicom_reader import dicom_to_rgb
            ds = pydicom.dcmread(io.BytesIO(contents), force=True)
            rgb_img = dicom_to_rgb(ds)
        except ValueError as val_err:
            logger.warning(f"DICOM validation failure: {val_err}")
            raise HTTPException(
                status_code=422,
                detail=f"Could not decode DICOM: {val_err}",
            )
        except Exception as dicom_err:
            logger.warning(f"DICOM decode error: {dicom_err}")
            raise HTTPException(
                status_code=422,
                detail="Could not decode the uploaded DICOM file as a valid medical scan.",
            )
        img = rgb_img
    else:
        np_arr = np.frombuffer(contents, np.uint8)
        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise HTTPException(
                status_code=422,
                detail="Could not decode the uploaded file as a valid image. "
                       "Please upload a valid JPEG, PNG, or DICOM MRI scan.",
            )
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 4. Check for non-finite pixel values
    if not np.all(np.isfinite(img)):
        raise HTTPException(
            status_code=422,
            detail="Uploaded image contains non-finite (NaN or Inf) pixel values.",
        )

    # 5. Standardize channel dimensionality to (H, W, 3)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image channel count: {img.shape}. Expected 2D or 3-channel RGB image.",
        )

    # 6. Enforce image dimension safety bounds
    validate_image_dimensions(height=img.shape[0], width=img.shape[1])

    return img


# ------------------------------------------------------------------ #
# Public Endpoints (Liveness and Telemetry)
# ------------------------------------------------------------------ #
@app.get(
    "/api/health",
    summary="Health check",
    description="Returns server status and active ensemble status (Public).",
    tags=["System"],
)
def health_check() -> dict:
    """Liveness and model readiness endpoint."""
    ready = is_ensemble_ready()
    status_map = get_model_status()
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "models_active": status_map,
        "pipeline_version": "3.0.0 (Tri-Ensemble)",
    }


@app.get(
    "/api/models",
    summary="Model Ensemble Telemetry",
    description="Returns active model architectures and dynamic ensemble voting weights (Public).",
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


# ------------------------------------------------------------------ #
# Protected Endpoints (Inference, Reporting, and History)
# ------------------------------------------------------------------ #
@app.post(
    "/api/predict",
    summary="Predict brain tumor from MRI",
    description=(
        "Upload a brain MRI image (JPEG/PNG/DICOM). "
        "Returns classification result, confidence score, uncertainty entropy, "
        "individual model votes, and base64-encoded Grad-CAM and Grad-CAM++ overlays."
    ),
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(..., description="Brain MRI image file (JPEG, PNG, or DICOM)"),
    use_tta: bool = Form(False, description="Enable 5-pass Test-Time Augmentation (TTA)"),
    _auth: Optional[str] = Depends(require_api_key),
) -> JSONResponse:
    """Performs full Tri-Ensemble inference with multi-modal XAI overlays."""
    safe_name = sanitize_filename(file.filename)
    logger.info(f"Processing prediction request: tta_enabled={use_tta}")

    img = await _read_upload_as_numpy(file)
    result = predict_tumor_logic(img, use_tta=use_tta)

    if not result.get("is_valid"):
        code = result.get("code", "UNKNOWN_ERROR")
        aid = result.get("analysis_id")
        ts = result.get("timestamp")
        p_ver = result.get("pipeline_version")
        try:
            log_scan(
                predicted_class="FAILED",
                confidence_pct=0.0,
                is_tumor=False,
                severity="N/A",
                tumor_area_pct=0.0,
                location="N/A",
                inference_time=0.0,
                scan_filename=safe_name,
                input_type="IMAGE",
                analysis_id=aid,
                timestamp=ts,
                status="FAILED",
                error_code=code,
                pipeline_version=p_ver,
            )
        except Exception as db_err:
            logger.warning(f"Could not record failed scan in database: {db_err}")

        status_code = 503 if code == "MODELS_UNAVAILABLE" else 400
        raise HTTPException(status_code=status_code, detail=result.get("error", "Inference failed."))

    # Log analysis to patient history for auditability
    try:
        log_scan(
            predicted_class=result["class_name"],
            confidence_pct=result["confidence"],
            is_tumor=result["is_tumor"],
            severity=result["severity"],
            tumor_area_pct=result["tumor_percentage"],
            location=result["location"],
            inference_time=result["inference_time"],
            scan_filename=safe_name,
            input_type="IMAGE",
            analysis_id=result.get("analysis_id"),
            timestamp=result.get("timestamp"),
            status=result.get("status", "COMPLETED"),
            pipeline_version=result.get("pipeline_version"),
            model_provenance=result.get("provenance"),
        )
    except Exception as log_err:
        logger.warning(f"Could not log scan to database: {log_err}")

    gradcam_b64 = _ndarray_to_base64(result["gradcam_overlay"])
    gradcam_pp_b64 = _ndarray_to_base64(result["gradcam_pp_overlay"])
    seg_b64 = _ndarray_to_base64(result["segmentation_img"])

    pred_breakdown = {}
    for cls, prob in zip(result["classes"], result["avg_pred"]):
        pred_breakdown[cls] = round(float(prob) * 100, 4)

    individual_probs = {}
    for m_name, probs in result.get("pred_dict", {}).items():
        individual_probs[m_name] = {
            cls: round(float(p) * 100, 4) for cls, p in zip(result["classes"], probs)
        }

    return JSONResponse(
        status_code=200,
        content={
            "analysis_id": result.get("analysis_id"),
            "timestamp": result.get("timestamp"),
            "status": result.get("status", "COMPLETED"),
            "pipeline_version": result.get("pipeline_version"),
            "provenance": result.get("provenance"),
            "tta_enabled": result.get("tta_enabled", False),
            "class_name": result["class_name"],
            "confidence": round(result["confidence"], 2),
            "confidence_metric": result.get("confidence_metric", "Ensemble Softmax Probability (Uncalibrated)"),
            "uncertainty_entropy": round(result.get("uncertainty", 0.0), 2),
            "uncertainty_metric": result.get("uncertainty_metric", "Normalized Shannon Entropy of Softmax Distribution"),
            "is_tumor": result["is_tumor"],
            "location": result["location"],
            "tumor_percentage": round(result["tumor_percentage"], 2),
            "severity": result["severity"],
            "severity_color": result["severity_color"],
            "inference_time_seconds": round(result["inference_time"], 4),
            "prediction_breakdown": pred_breakdown,
            "individual_model_breakdown": individual_probs,
            "gradcam_overlay_b64": gradcam_b64,
            "gradcam_pp_overlay_b64": gradcam_pp_b64,
            "segmentation_b64": seg_b64,
        },
    )


@app.post(
    "/api/report",
    summary="Generate Clinical PDF Diagnostic Report",
    description="Upload an MRI image with patient details and receive a structured decision-support PDF report.",
    tags=["Reports"],
)
async def generate_report(
    file: UploadFile = File(..., description="Brain MRI image file (JPEG, PNG, or DICOM)"),
    patient_id: str = Form("PT-8942"),
    patient_name: str = Form("Anonymous Patient"),
    patient_age: str = Form("45"),
    patient_gender: str = Form("Unspecified"),
    analysis_id: Optional[str] = Form(None),
    _auth: Optional[str] = Depends(require_api_key),
):
    """Generates and streams a structured decision-support PDF report."""
    # Sanitize inputs
    safe_pid, safe_pname, safe_page, safe_pgender = sanitize_patient_metadata(
        patient_id, patient_name, patient_age, patient_gender
    )
    safe_aid = validate_analysis_id(analysis_id)
    safe_scan_name = sanitize_filename(file.filename)

    logger.info(f"Processing report generation request: analysis_id={safe_aid or 'pending'}")

    img = await _read_upload_as_numpy(file)
    diag_result = predict_tumor_logic(img, analysis_id=safe_aid)

    if not diag_result.get("is_valid"):
        code = diag_result.get("code")
        status_code = 503 if code == "MODELS_UNAVAILABLE" else 400
        raise HTTPException(
            status_code=status_code,
            detail=diag_result.get("error", "Failed to process image for report.")
        )

    try:
        pdf_path = generate_pdf_report(
            diag_result=diag_result,
            patient_id=safe_pid,
            patient_name=safe_pname,
            patient_age=safe_page,
            patient_gender=safe_pgender,
            analysis_id=diag_result.get("analysis_id"),
        )
    except ValueError as ve:
        logger.warning(f"Validation failure in report generation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal report generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Report generation failed: an internal error occurred during report processing."
        )

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=500,
            detail="Report generation failed: output file not found."
        )

    # Record report path in audit database
    try:
        log_scan(
            predicted_class=diag_result["class_name"],
            confidence_pct=diag_result["confidence"],
            is_tumor=diag_result["is_tumor"],
            severity=diag_result["severity"],
            tumor_area_pct=diag_result["tumor_percentage"],
            location=diag_result["location"],
            inference_time=diag_result["inference_time"],
            patient_id=safe_pid,
            patient_name=safe_pname,
            patient_age=safe_page,
            patient_gender=safe_pgender,
            scan_filename=safe_scan_name,
            input_type="IMAGE",
            pdf_path=pdf_path,
            analysis_id=diag_result.get("analysis_id"),
            timestamp=diag_result.get("timestamp"),
            status=diag_result.get("status", "COMPLETED"),
            pipeline_version=diag_result.get("pipeline_version"),
            model_provenance=diag_result.get("provenance"),
        )
    except Exception as db_err:
        logger.warning(f"Could not record report path in database: {db_err}")

    return FileResponse(
        path=pdf_path,
        filename=os.path.basename(pdf_path),
        media_type="application/pdf"
    )


@app.get(
    "/api/reports/{analysis_id}",
    summary="Retrieve Generated PDF Report",
    description="Securely fetches a previously generated PDF report using its traceable analysis ID.",
    tags=["Reports"],
)
def get_report_by_analysis_id(
    analysis_id: str,
    _auth: Optional[str] = Depends(require_api_key),
):
    """Retrieves an existing PDF report securely with strict path-traversal prevention."""
    valid_id = validate_analysis_id(analysis_id)
    scan = get_scan_by_analysis_id(valid_id)
    if not scan:
        raise HTTPException(
            status_code=404,
            detail="Report not found for the requested analysis identifier.",
        )

    pdf_path = scan.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="Report file is missing or has expired from local storage.",
        )

    # Verify path confinement within reports directory
    reports_dir = os.path.abspath(REPORTS_DIR)
    norm_pdf = os.path.abspath(pdf_path)
    if not norm_pdf.startswith(reports_dir):
        logger.error(f"Access violation: report path {pdf_path} outside reports directory.")
        raise HTTPException(status_code=403, detail="Access denied to requested report.")

    return FileResponse(
        path=norm_pdf,
        filename=os.path.basename(norm_pdf),
        media_type="application/pdf",
    )


@app.get(
    "/api/provenance/{analysis_id}",
    summary="Retrieve Analysis Provenance",
    description="Returns model architecture, weights digests, and pipeline configuration that produced the given analysis.",
    tags=["System"],
)
def get_analysis_provenance(
    analysis_id: str,
    _auth: Optional[str] = Depends(require_api_key),
) -> dict:
    """Retrieves immutable model and preprocessing provenance for a given analysis."""
    safe_aid = validate_analysis_id(analysis_id)
    prov = get_scan_provenance(safe_aid)
    if not prov:
        raise HTTPException(
            status_code=404,
            detail="Analysis record not found for the requested identifier.",
        )
    return prov


@app.get(
    "/api/history",
    summary="Patient Case History",
    description="Retrieves paginated scan history records from the audit database.",
    tags=["System"],
)
def get_patient_history(
    limit: int = Query(50, ge=1, le=100, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    _auth: Optional[str] = Depends(require_api_key),
) -> dict:
    """Secure paginated scan history endpoint."""
    total = get_scan_count()
    records = get_scans_paginated(limit=limit, offset=offset)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "scans": records,
    }


# ------------------------------------------------------------------ #
# Mount Static Frontend (UI/UX Pro Max Workstation)
# ------------------------------------------------------------------ #
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
