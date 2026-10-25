# 06 — API Reference

The FastAPI REST backend (`src/api.py`) exposes the complete inference pipeline as HTTP endpoints while serving the PACS Clinical Diagnostic Studio workstation. It can be integrated with any external system, hospital PACS, or frontend application.

---

## Starting the API Server

```bash
python scripts/run_api.py
```

The server starts on `http://localhost:8000`.

Interactive Swagger documentation: `http://localhost:8000/docs`  
ReDoc documentation: `http://localhost:8000/redoc`

---

## CORS Policy

All origins, methods (GET, POST), and headers are allowed:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

This makes it easy to call the API from a separate frontend (React, Vue, etc.) without browser CORS errors.

---

## Endpoints

### `GET /api/health`

**Purpose:** Liveness check. Confirms the server is running and the ensemble models are loaded.

**Request:** No body or parameters required.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "ensemble": "DenseNet121 + InceptionV3",
  "version": "1.0.0"
}
```

**Example (curl):**
```bash
curl http://localhost:8000/api/health
```

---

### `POST /api/predict`

**Purpose:** Upload a brain MRI image and receive a full diagnostic JSON response with classification, tumor properties, and base64-encoded visual overlays.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body field: `file` — the MRI image file (JPEG or PNG)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/predict \
     -F "file=@test_images/Tr-me_0025.jpg"
```

**Example (Python requests):**
```python
import requests

with open("test_images/Tr-me_0025.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"file": ("scan.jpg", f, "image/jpeg")}
    )
    
result = response.json()
print(result["class_name"])       # e.g., "Meningioma Tumor"
print(result["confidence"])       # e.g., 84.7231
```

**Success Response (HTTP 200):**

```json
{
  "class_name": "Meningioma Tumor",
  "confidence": 84.7231,
  "is_tumor": true,
  "location": "Right Parietal Lobe (Superior)",
  "tumor_percentage": 6.4512,
  "severity": "Moderate",
  "severity_color": "#f59e0b",
  "inference_time": 1.3240,
  "prediction_breakdown": {
    "No Tumor": 3.2100,
    "Glioma Tumor": 8.4400,
    "Meningioma Tumor": 84.7231,
    "Pituitary Tumor": 3.6269
  },
  "gradcam_overlay_b64": "<base64 PNG string>",
  "segmentation_b64": "<base64 PNG string>"
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `class_name` | `string` | Predicted class (one of 4 tumor classes) |
| `confidence` | `float` | Prediction confidence as percentage (0–100) |
| `is_tumor` | `boolean` | `true` if any tumor class was predicted |
| `location` | `string` | Anatomical location of the tumor |
| `tumor_percentage` | `float` | Estimated tumor area as % of scan |
| `severity` | `string` | `High`, `Moderate`, `Low`, or `Uncertain` |
| `severity_color` | `string` | Hex color code matching the severity |
| `inference_time` | `float` | Total processing time in seconds |
| `prediction_breakdown` | `object` | Per-class probability breakdown (%) |
| `gradcam_overlay_b64` | `string` | Base64-encoded PNG of the Grad-CAM heatmap overlay |
| `segmentation_b64` | `string` | Base64-encoded PNG of the tumor segmentation overlay |

**Error Responses:**

| Status Code | Cause | Response |
|---|---|---|
| `422 Unprocessable Entity` | Invalid image file (not a valid JPEG/PNG) | `{"detail": "Could not decode the uploaded file..."}` |
| `500 Internal Server Error` | Inference pipeline failure | `{"detail": "An internal server error occurred: ..."}` |

---

## Decoding Base64 Images (Python)

```python
import base64
from PIL import Image
import io

# Decode Grad-CAM overlay
gradcam_bytes = base64.b64decode(result["gradcam_overlay_b64"])
gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
gradcam_img.save("gradcam_output.png")

# Decode segmentation overlay
seg_bytes = base64.b64decode(result["segmentation_b64"])
seg_img = Image.open(io.BytesIO(seg_bytes))
seg_img.save("segmentation_output.png")
```

---

## Decoding Base64 Images (JavaScript)

```javascript
// Display Grad-CAM in an <img> tag
const gradcamSrc = `data:image/png;base64,${result.gradcam_overlay_b64}`;
document.getElementById('gradcam-img').src = gradcamSrc;
```

---

## Workstation & API Features

The FastAPI application provides both the interactive Clinical Diagnostic Studio frontend and the headless REST API:

| Capability | Web PACS Studio (`/`) | REST API (`/api/*`) |
|---|---|---|
| Access URL | `http://localhost:8000/` | `http://localhost:8000/api` |
| Input method | Drag & drop file upload / DICOM | HTTP multipart form upload |
| Output format | Real-time Quad-View overlays + SVG metrics | JSON with base64-encoded overlays |
| Interactive documentation | Built-in Diagnostic Guide & Shortcuts | Swagger UI (`/docs`) & ReDoc (`/redoc`) |
| Use Case | Radiology review, clinical validation | PACS integration, EMR connectors, mobile apps |
