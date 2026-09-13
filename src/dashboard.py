"""
src/dashboard.py
================
Gradio-based Clinical Diagnostic Studio for Brain XAI Ensemble.

Tabs:
  1. Clinical Diagnostics & XAI      — Upload, infer, visualize overlays.
  2. Tri-Ensemble Analytics          — Per-model probability breakdown.
  3. Clinical Report Export (PDF)    — Patient form & downloadable PDF.
  4. DICOM File Import               — Upload .dcm hospital files directly.
  5. Patient Case History            — SQLite-backed scan log with trends.
  6. Architecture & Citations        — Technical reference.
"""

from typing import Tuple, Optional, Dict, Any
import os
import numpy as np
import gradio as gr

from src.inference import predict_tumor_logic
from src.report_generator import generate_pdf_report
from src.database import log_scan, get_all_scans, clear_all_scans, get_scan_count
from src.dicom_reader import read_dicom
from src.theme import get_clinical_theme

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSS_PATH = os.path.join(_BASE_DIR, "assets", "styles.css")

EXAMPLE_IMAGES = [
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0025.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0070.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0080.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-pi_0050.jpg")],
]

# ── State shared across tabs ────────────────────────────────────────────────────
_last_result: Dict[str, Any] = {}  # Stores latest inference result for logging


def get_custom_css() -> str:
    """Reads and returns the external CSS stylesheet."""
    if os.path.exists(_CSS_PATH):
        with open(_CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ── Empty State HTML Builders ────────────────────────────────────────────────────

def get_empty_states() -> Tuple[str, str, str, str]:
    diag = """
    <div class="result-card result-card-empty">
        <div class="result-empty-text">
            Upload an MRI axial slice and click <b>Analyze MRI Scan</b> to trigger
            Tri-Ensemble neural inference with multi-modal explainability.
        </div>
    </div>
    """
    conf = """
    <div class="result-card">
        <div class="card-title-sm">TRI-ENSEMBLE CONFIDENCE</div>
        <div class="conf-main-value conf-val-empty">-- %</div>
        <div class="conf-entropy-text">Awaiting scan input...</div>
    </div>
    """
    props = """
    <div class="result-card">
        <div class="card-title-sm">CLINICAL MEASUREMENTS</div>
        <div class="biomarkers-grid">
            <div class="biomarker-item"><div class="biomarker-label">Estimated Location</div><div class="biomarker-value">--</div></div>
            <div class="biomarker-item"><div class="biomarker-label">Tumor Tissue Area</div><div class="biomarker-value">-- %</div></div>
            <div class="biomarker-item"><div class="biomarker-label">Severity Grading</div><div class="biomarker-value">--</div></div>
            <div class="biomarker-item"><div class="biomarker-label">Uncertainty Entropy</div><div class="biomarker-value">-- %</div></div>
        </div>
    </div>
    """
    breakdown = """
    <div class="result-card">
        <div class="card-title-sm">MULTI-CLASS PROBABILITIES</div>
        <div class="conf-entropy-text">Glioma | Meningioma | Pituitary | No Tumor</div>
    </div>
    """
    return diag, conf, props, breakdown


# ── Result Formatter ─────────────────────────────────────────────────────────────

def format_results(data: Dict[str, Any]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    str, str, str, str, str
]:
    global _last_result
    _last_result = data

    if not data.get("is_valid"):
        err = f"<div class='result-card result-card-error'><b>Error:</b> {data.get('error', 'Inference failed')}</div>"
        return None, None, None, None, err, "", "", "", ""

    is_tumor = data["is_tumor"]
    cls_name = data["class_name"]
    conf = data["confidence"]
    uncertainty = data.get("uncertainty", 0.0)
    sev = data["severity"]
    loc = data["location"]
    pct = data["tumor_percentage"]
    inf_time = data["inference_time"]

    card_state_class = "result-card result-card-tumor" if is_tumor else "result-card result-card-normal"
    badge_class = "diag-badge diag-badge-tumor" if is_tumor else "diag-badge diag-badge-normal"
    conf_val_class = "conf-main-value conf-val-tumor" if is_tumor else "conf-main-value conf-val-normal"
    progress_fill_class = "progress-fill progress-fill-tumor" if is_tumor else "progress-fill progress-fill-normal"

    diagnosis_label = "TUMOR DETECTED" if is_tumor else "NO TUMOR DETECTED"
    sev_color_map = {
        "Critical": "#ef4444",
        "Severe":   "#f97316",
        "Moderate": "#eab308",
        "Mild":     "#22c55e",
        "N/A":      "#64748b",
    }
    sev_color = sev_color_map.get(sev, "#64748b")

    diag_html = f"""
    <div class="{card_state_class}">
        <div class="diag-header-row">
            <span class="diag-class-name">{cls_name}</span>
            <span class="{badge_class}">{diagnosis_label}</span>
        </div>
        <div class="severity-row">
            <span class="severity-label">Severity:</span>
            <span class="severity-value" style="color:{sev_color}; font-weight:700;">{sev}</span>
        </div>
        <div class="progress-bar-wrap">
            <div class="{progress_fill_class}" style="width:{min(conf,100):.1f}%"></div>
        </div>
        <div class="diag-footer-row">
            <span class="inf-time-label">Inference: {inf_time:.2f}s</span>
        </div>
    </div>
    """

    conf_html = f"""
    <div class="result-card">
        <div class="card-title-sm">TRI-ENSEMBLE CONFIDENCE</div>
        <div class="{conf_val_class}">{conf:.1f}%</div>
        <div class="conf-entropy-text">Prediction Uncertainty: {uncertainty:.4f} bits</div>
    </div>
    """

    props_html = f"""
    <div class="result-card">
        <div class="card-title-sm">CLINICAL MEASUREMENTS</div>
        <div class="biomarkers-grid">
            <div class="biomarker-item">
                <div class="biomarker-label">Estimated Location</div>
                <div class="biomarker-value">{loc}</div>
            </div>
            <div class="biomarker-item">
                <div class="biomarker-label">Tumor Tissue Area</div>
                <div class="biomarker-value">{pct:.1f}%</div>
            </div>
            <div class="biomarker-item">
                <div class="biomarker-label">Severity Grading</div>
                <div class="biomarker-value" style="color:{sev_color};">{sev}</div>
            </div>
            <div class="biomarker-item">
                <div class="biomarker-label">Uncertainty Entropy</div>
                <div class="biomarker-value">{uncertainty:.4f} bits</div>
            </div>
        </div>
    </div>
    """

    avg_pred = data.get("avg_pred", [])
    classes  = data.get("classes", [])
    pred_dict = data.get("pred_dict", {})

    def make_bar(label, prob, color="#3b82f6"):
        pct_v = prob * 100
        return f"""
        <div style="margin:6px 0;">
            <div style="display:flex;justify-content:space-between;font-size:12px;color:#cbd5e1;margin-bottom:2px;">
                <span>{label}</span><span>{pct_v:.1f}%</span>
            </div>
            <div style="background:#1e293b;border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{pct_v:.1f}%;height:100%;background:{color};border-radius:4px;transition:width 0.4s;"></div>
            </div>
        </div>"""

    colors = ["#ef4444", "#f97316", "#3b82f6", "#22c55e"]
    class_bars = ""
    for i, (cls, prob) in enumerate(zip(classes, avg_pred)):
        class_bars += make_bar(cls, float(prob), colors[i % len(colors)])

    model_section = ""
    for mname, probs in pred_dict.items():
        top_cls = classes[int(np.argmax(probs))]
        top_conf = float(np.max(probs)) * 100
        model_section += f"""
        <div style="background:#0f172a;border-radius:8px;padding:10px 14px;margin:6px 0;">
            <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">{mname}</div>
            <div style="font-size:13px;color:#e2e8f0;font-weight:600;">{top_cls} — {top_conf:.1f}%</div>
        </div>"""

    breakdown_html = f"""
    <div class="result-card">
        <div class="card-title-sm">MULTI-CLASS PROBABILITIES</div>
        {class_bars}
        <div style="margin-top:14px;">
            <div class="card-title-sm">PER-MODEL CONSENSUS BREAKDOWN</div>
            {model_section}
        </div>
    </div>
    """

    return (
        data.get("img"),
        data.get("gradcam_overlay"),
        data.get("gradcam_pp_overlay"),
        data.get("segmentation_img"),
        diag_html,
        conf_html,
        props_html,
        breakdown_html,
        breakdown_html,  # analytics tab also uses breakdown_html
    )


# ── Patient History HTML Builder ─────────────────────────────────────────────────

def build_history_html() -> str:
    """Generates an HTML table of all logged scans from the database."""
    scans = get_all_scans()
    count = len(scans)

    header = f"""
    <div class="result-card">
        <div class="card-title-sm">PATIENT CASE HISTORY — {count} SCAN(S) LOGGED</div>
    """
    if count == 0:
        return header + """
        <div style="color:#94a3b8;font-size:13px;text-align:center;padding:28px 0;">
            No scans logged yet. Run an analysis to record the first case.
        </div>
    </div>"""

    rows_html = ""
    for s in scans:
        tumor_badge = (
            "<span style='color:#ef4444;font-weight:600;'>Tumor</span>"
            if s["is_tumor"] else
            "<span style='color:#22c55e;font-weight:600;'>No Tumor</span>"
        )
        input_badge = (
            "<span style='background:#1e3a5f;color:#60a5fa;padding:2px 7px;border-radius:4px;font-size:11px;'>DICOM</span>"
            if s["input_type"] == "DICOM" else
            "<span style='background:#1a2e1a;color:#4ade80;padding:2px 7px;border-radius:4px;font-size:11px;'>IMAGE</span>"
        )
        sev_color_map = {
            "Critical": "#ef4444", "Severe": "#f97316",
            "Moderate": "#eab308", "Mild": "#22c55e", "N/A": "#64748b",
        }
        sev_c = sev_color_map.get(s["severity"], "#64748b")
        rows_html += f"""
        <tr style="border-bottom:1px solid #1e293b;">
            <td style="padding:8px 10px;color:#94a3b8;font-size:12px;">{s['timestamp']}</td>
            <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;">{s['patient_id']}</td>
            <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;">{s['patient_name']}</td>
            <td style="padding:8px 10px;color:#e2e8f0;font-weight:600;font-size:12px;">{s['predicted_class']}</td>
            <td style="padding:8px 10px;font-size:12px;">{tumor_badge}</td>
            <td style="padding:8px 10px;color:{sev_c};font-weight:600;font-size:12px;">{s['severity']}</td>
            <td style="padding:8px 10px;color:#94a3b8;font-size:12px;">{s['confidence_pct']:.1f}%</td>
            <td style="padding:8px 10px;color:#94a3b8;font-size:12px;">{s['tumor_area_pct']:.1f}%</td>
            <td style="padding:8px 10px;font-size:12px;">{input_badge}</td>
            <td style="padding:8px 10px;color:#64748b;font-size:11px;">{s['scan_filename']}</td>
        </tr>"""

    table_html = f"""
    <div style="overflow-x:auto;margin-top:10px;">
        <table style="width:100%;border-collapse:collapse;font-family:monospace;">
            <thead>
                <tr style="background:#0f172a;border-bottom:2px solid #334155;">
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Timestamp</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Patient ID</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Name</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Diagnosis</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Tumor</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Severity</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Confidence</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Area %</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Source</th>
                    <th style="padding:10px;text-align:left;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;">Filename</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>"""

    return header + table_html + "</div>"


# ── Event Handlers ───────────────────────────────────────────────────────────────

def handle_prediction(img: Optional[np.ndarray], filename: str = "N/A"):
    data = predict_tumor_logic(img)
    result = format_results(data)

    # Auto-log to database after every successful prediction
    if data.get("is_valid"):
        log_scan(
            predicted_class=data["class_name"],
            confidence_pct=data["confidence"],
            is_tumor=data["is_tumor"],
            severity=data["severity"],
            tumor_area_pct=data["tumor_percentage"],
            location=data["location"],
            inference_time=data["inference_time"],
            scan_filename=filename,
            input_type="IMAGE",
        )

    return result


def handle_dicom_upload(dicom_file) -> Tuple[
    Optional[np.ndarray], str, str, str, str, str
]:
    """
    Reads an uploaded DICOM file, runs inference, and returns:
      (image_for_tab1, patient_id, patient_name, patient_age, patient_gender, status_msg)
    """
    if dicom_file is None:
        return None, "PT-0000", "Anonymous Patient", "N/A", "N/A", "No file uploaded."

    filepath = dicom_file.name if hasattr(dicom_file, "name") else str(dicom_file)
    result = read_dicom(filepath)

    if not result["success"]:
        err_msg = f"DICOM read error: {result['error']}"
        return None, "PT-0000", "Anonymous Patient", "N/A", "N/A", err_msg

    meta = result["metadata"]
    image = result["image"]
    filename = result["filename"]

    # Auto-log to database
    dicom_data = predict_tumor_logic(image)
    if dicom_data.get("is_valid"):
        log_scan(
            predicted_class=dicom_data["class_name"],
            confidence_pct=dicom_data["confidence"],
            is_tumor=dicom_data["is_tumor"],
            severity=dicom_data["severity"],
            tumor_area_pct=dicom_data["tumor_percentage"],
            location=dicom_data["location"],
            inference_time=dicom_data["inference_time"],
            patient_id=meta.get("patient_id", "N/A"),
            patient_name=meta.get("patient_name", "Anonymous"),
            patient_age=meta.get("patient_age", "N/A"),
            patient_gender=meta.get("patient_gender", "N/A"),
            scan_filename=filename,
            input_type="DICOM",
        )

    status_lines = [
        f"DICOM loaded successfully: {filename}",
        f"Patient ID:  {meta.get('patient_id', 'N/A')}",
        f"Modality:    {meta.get('modality', 'N/A')}",
        f"Study Date:  {meta.get('study_date', 'N/A')}",
        f"Slice Thick: {meta.get('slice_thickness', 'N/A')}",
        f"Pixel Space: {meta.get('pixel_spacing', 'N/A')}",
        f"Institution: {meta.get('institution', 'N/A')}",
        "",
        "Switch to 'Clinical Diagnostics & XAI' tab to see the full inference results.",
    ]

    return (
        image,
        meta.get("patient_id", "PT-0000"),
        str(meta.get("patient_name", "Anonymous Patient")),
        meta.get("patient_age", "N/A"),
        meta.get("patient_gender", "N/A"),
        "\n".join(status_lines),
    )


def handle_pdf_export(
    img: Optional[np.ndarray],
    patient_id: str,
    patient_name: str,
    patient_age: str,
    patient_gender: str
) -> Optional[str]:
    if img is None:
        return None
    data = predict_tumor_logic(img)
    if not data["is_valid"]:
        return None
    pdf_path = generate_pdf_report(
        diag_result=data,
        patient_id=patient_id or "PT-8942",
        patient_name=patient_name or "Anonymous Patient",
        patient_age=patient_age or "45",
        patient_gender=patient_gender or "Unspecified"
    )
    # Update last log row with pdf_path (best-effort)
    return pdf_path


def handle_refresh_history() -> str:
    return build_history_html()


def handle_clear_history() -> str:
    cleared = clear_all_scans()
    return f"""
    <div class="result-card">
        <div class="card-title-sm">PATIENT CASE HISTORY</div>
        <div style="color:#22c55e;font-size:13px;text-align:center;padding:20px 0;">
            {cleared} scan record(s) cleared successfully. History is now empty.
        </div>
    </div>"""


# ── App Factory ──────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Brain XAI Ensemble — Clinical Studio", fill_width=True) as demo:

        # ── Header ───────────────────────────────────────────────────────────
        with gr.Row(elem_id="app-header"):
            gr.HTML("""
            <div class="header-container">
                <div class="header-brand">
                    <div class="header-logo-badge">
                        <span class="header-logo-text">BT</span>
                    </div>
                    <div>
                        <h1 class="header-title">
                            Brain XAI Ensemble <span class="header-title-highlight">Clinical Diagnostic Studio</span>
                        </h1>
                        <p class="header-subtitle">
                            Explainable AI Tri-Ensemble (ConvNeXt-Small 45% + InceptionV3 35% + DenseNet121 20%)
                            &nbsp;|&nbsp; 98.88% TTA Accuracy &nbsp;|&nbsp; DICOM Support &nbsp;|&nbsp; PDF Reports
                        </p>
                    </div>
                </div>
                <div class="header-badges-group">
                    <span class="badge-primary">Brain-Masked Grad-CAM++</span>
                    <span class="badge-primary">DICOM Native Support</span>
                    <span class="badge-primary">Patient Case History</span>
                </div>
            </div>
            """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: Clinical Diagnostics & XAI
            # ================================================================
            with gr.TabItem("Clinical Diagnostics & XAI"):
                with gr.Row():
                    with gr.Column(scale=5, min_width=240):
                        with gr.Column(elem_classes="card-panel"):
                            gr.HTML('<div class="card-heading">1. Upload Brain MRI Axial Slice</div>')
                            input_img = gr.Image(
                                type="numpy", label=None, show_label=False, height=280
                            )
                            with gr.Row():
                                predict_btn = gr.Button(
                                    "Analyze MRI Scan", elem_classes="action-btn", scale=3
                                )
                                clear_btn = gr.Button(
                                    "Reset", elem_classes="clear-btn", scale=1
                                )
                            gr.Examples(
                                examples=EXAMPLE_IMAGES,
                                inputs=input_img,
                                label="Sample MRI Scans",
                                examples_per_page=4,
                            )

                    with gr.Column(scale=7, min_width=260):
                        empty_diag, empty_conf, empty_props, empty_breakdown = get_empty_states()
                        diag_box      = gr.HTML(value=empty_diag)
                        conf_box      = gr.HTML(value=empty_conf)
                        props_box     = gr.HTML(value=empty_props)
                        breakdown_box = gr.HTML(value=empty_breakdown)

                gr.HTML('<div class="section-title">Multi-Modal Explainable AI (XAI) Overlays</div>')
                with gr.Row():
                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-orig">(A) Input MRI Slice</div>')
                        out_orig = gr.Image(label=None, show_label=False, interactive=False, height=240)

                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-grad">(B) Grad-CAM Attention</div>')
                        out_grad = gr.Image(label=None, show_label=False, interactive=False, height=240)

                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-pp">(C) Brain-Masked Grad-CAM++</div>')
                        out_grad_pp = gr.Image(label=None, show_label=False, interactive=False, height=240)

                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-seg">(D) Morphological Segmentation</div>')
                        out_seg = gr.Image(label=None, show_label=False, interactive=False, height=240)

            # ================================================================
            # TAB 2: Tri-Ensemble Analytics
            # ================================================================
            with gr.TabItem("Tri-Ensemble Analytics"):
                gr.HTML("""
                <div class="result-card">
                    <div class="card-title-sm">Ensemble Consensus Framework</div>
                    <p style="color:#94a3b8; font-size:13.5px; margin:0; line-height:1.6;">
                        Predictions from three distinct architectures fused via Nelder-Mead Weighted Soft Voting:
                        <b style="color:#cbd5e1;">ConvNeXt-Small (45%)</b> +
                        <b style="color:#cbd5e1;">InceptionV3 (35%)</b> +
                        <b style="color:#cbd5e1;">DenseNet121 (20%)</b>.
                    </p>
                </div>
                """)
                analytics_box = gr.HTML("""
                <div class="result-card analytics-empty-box">
                    <div>Run an analysis in the Diagnostics tab to view the live multi-model breakdown.</div>
                </div>
                """)

            # ================================================================
            # TAB 3: Clinical Report Export (PDF)
            # ================================================================
            with gr.TabItem("Clinical Report Export (PDF)"):
                with gr.Row():
                    with gr.Column(scale=6, min_width=240, elem_classes="card-panel"):
                        gr.HTML('<div class="card-heading">Patient & Clinical Details</div>')
                        with gr.Row():
                            p_id     = gr.Textbox(label="Patient ID",   value="PT-8942")
                            p_name   = gr.Textbox(label="Patient Name", value="Anonymous Patient")
                        with gr.Row():
                            p_age    = gr.Textbox(label="Age", value="45")
                            p_gender = gr.Radio(
                                choices=["Male", "Female", "Other"],
                                label="Gender", value="Male"
                            )
                        generate_pdf_btn = gr.Button(
                            "Generate Official PDF Report", elem_classes="download-btn"
                        )

                    with gr.Column(scale=6, min_width=240, elem_classes="card-panel"):
                        gr.HTML('<div class="card-heading">Downloadable PDF Medical Document</div>')
                        pdf_download_file = gr.File(label="Download Generated PDF", interactive=False)
                        gr.HTML("""
                        <div class="report-notice-box">
                            The generated PDF includes patient information, Tri-Ensemble classification
                            confidence, Brain-Masked Grad-CAM++ attention overlays, morphological
                            tumor area measurements, and clinical signature blocks.
                        </div>
                        """)

            # ================================================================
            # TAB 4: DICOM File Import
            # ================================================================
            with gr.TabItem("DICOM File Import (.dcm)"):
                gr.HTML("""
                <div class="result-card">
                    <div class="card-title-sm">NATIVE DICOM HOSPITAL FILE SUPPORT</div>
                    <p style="color:#94a3b8;font-size:13px;line-height:1.7;margin-top:6px;">
                        Upload a raw <b style="color:#60a5fa;">.dcm</b> DICOM file directly from a hospital scanner,
                        PACS system, or imaging archive.<br>
                        The system automatically extracts patient metadata, converts the pixel array to RGB,
                        runs the Tri-Ensemble inference pipeline, and populates the PDF report form.
                    </p>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=5, elem_classes="card-panel"):
                        gr.HTML('<div class="card-heading">Upload DICOM File</div>')
                        dicom_file_input = gr.File(
                            label="Select .dcm File",
                            file_types=[".dcm"],
                            type="filepath",
                        )
                        dicom_analyze_btn = gr.Button(
                            "Load & Analyze DICOM", elem_classes="action-btn"
                        )

                    with gr.Column(scale=7, elem_classes="card-panel"):
                        gr.HTML('<div class="card-heading">Extracted DICOM Metadata & Status</div>')
                        dicom_status = gr.Textbox(
                            label=None,
                            show_label=False,
                            lines=10,
                            interactive=False,
                            placeholder="DICOM metadata and import status will appear here after loading.",
                        )

                gr.HTML("""
                <div class="result-card" style="margin-top:12px;">
                    <div class="card-title-sm">HOW IT WORKS</div>
                    <ol style="color:#94a3b8;font-size:12.5px;line-height:2;margin:8px 0 0 16px;padding:0;">
                        <li>Upload your .dcm file using the button above.</li>
                        <li>Click <b style="color:#e2e8f0;">Load & Analyze DICOM</b>.</li>
                        <li>The image is auto-converted and sent to the Tri-Ensemble inference pipeline.</li>
                        <li>Patient ID, Name, and Age are auto-filled in the PDF Report tab.</li>
                        <li>The scan is automatically logged to the Patient Case History database.</li>
                        <li>Switch to the <b style="color:#e2e8f0;">Clinical Diagnostics & XAI</b> tab
                            to view the full Grad-CAM analysis.</li>
                    </ol>
                </div>
                """)

            # ================================================================
            # TAB 5: Patient Case History
            # ================================================================
            with gr.TabItem("Patient Case History"):
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_history_btn = gr.Button(
                            "Refresh History", elem_classes="action-btn"
                        )
                    with gr.Column(scale=1):
                        clear_history_btn = gr.Button(
                            "Clear All Records", elem_classes="clear-btn"
                        )

                gr.HTML("""
                <div style="color:#64748b;font-size:11.5px;margin:4px 0 10px 0;line-height:1.6;">
                    Every successful scan analysis is automatically saved to a local SQLite database
                    (<code>data/patient_history.db</code>). Records persist across sessions.
                    This database is excluded from version control to protect patient privacy.
                </div>
                """)
                history_box = gr.HTML(value=build_history_html())

            # ================================================================
            # TAB 6: Architecture & Citations
            # ================================================================
            with gr.TabItem("Architecture & Citations"):
                gr.HTML("""
                <div class="result-card doc-card">
                    <h3 class="doc-title">Deep Learning & Explainable AI Specifications</h3>
                    <ul class="doc-list">
                        <li><b>Primary Backbone: ConvNeXt-Small (Meta AI, CVPR 2022)</b>:
                            50M-parameter modernized ConvNet with 7×7 depthwise convolutions,
                            GELU activations, and LayerNorm. Voting weight: 45%.</li>
                        <li><b>Multi-Scale Context: InceptionV3</b>:
                            Parallel 1×1, 3×3, 5×5 receptive fields at native 299×299 resolution.
                            Captures both micro-adenoma foci and broad astrocytoma margins. Voting weight: 35%.</li>
                        <li><b>Feature Saliency: DenseNet121</b>:
                            Direct dense feature concatenation across all blocks for high-resolution
                            Grad-CAM gradient propagation. Voting weight: 20%.</li>
                        <li><b>Nelder-Mead Ensemble Optimization</b>:
                            Soft-voting weights calibrated via Nelder-Mead simplex optimization on
                            validation logits. Eliminates correlated single-model error modes.</li>
                        <li><b>Brain-Masked Grad-CAM++</b>:
                            Higher-order gradient derivatives element-wise gated with Otsu-derived
                            intracranial binary mask. Achieves 93.47% localization specificity
                            on 791 ground-truth tumor masks.</li>
                        <li><b>10-Pass Test-Time Augmentation (TTA)</b>:
                            Classification averaged across 10 augmented passes (rotation, zoom, shift).
                            Boosts accuracy from 94.96% (single-pass) to 98.88% (1,255/1,269 correct).</li>
                        <li><b>DICOM Support</b>:
                            Native .dcm file reading via pydicom 3.x with automatic VOI windowing,
                            photometric conversion (MONOCHROME1/2, RGB, YBR), and patient metadata extraction.</li>
                        <li><b>Patient History</b>:
                            Persistent SQLite patient case log with longitudinal scan tracking.</li>
                    </ul>
                </div>
                """)

        # ── Shared State ─────────────────────────────────────────────────────
        empty_diag, empty_conf, empty_props, empty_breakdown = get_empty_states()

        # ── Event Wiring ─────────────────────────────────────────────────────

        predict_btn.click(
            fn=handle_prediction,
            inputs=[input_img],
            outputs=[
                out_orig, out_grad, out_grad_pp, out_seg,
                diag_box, conf_box, props_box, breakdown_box, analytics_box
            ]
        )

        clear_btn.click(
            fn=lambda: (
                None, None, None, None, None,
                empty_diag, empty_conf, empty_props, empty_breakdown,
                "<div class='result-card analytics-empty-box'><div>Run an analysis in the Diagnostics tab to view the live multi-model breakdown.</div></div>"
            ),
            inputs=[],
            outputs=[
                input_img, out_orig, out_grad, out_grad_pp, out_seg,
                diag_box, conf_box, props_box, breakdown_box, analytics_box
            ]
        )

        generate_pdf_btn.click(
            fn=handle_pdf_export,
            inputs=[input_img, p_id, p_name, p_age, p_gender],
            outputs=[pdf_download_file]
        )

        # DICOM tab: Load file -> auto-fills metadata + sends image to Tab 1
        dicom_analyze_btn.click(
            fn=handle_dicom_upload,
            inputs=[dicom_file_input],
            outputs=[input_img, p_id, p_name, p_age, dicom_status]
            # Note: p_gender not wired as DICOM gender is a Radio, auto-fill not supported in Gradio Radio
        )

        # Patient History tab buttons
        refresh_history_btn.click(
            fn=handle_refresh_history,
            inputs=[],
            outputs=[history_box]
        )

        clear_history_btn.click(
            fn=handle_clear_history,
            inputs=[],
            outputs=[history_box]
        )

    return demo
