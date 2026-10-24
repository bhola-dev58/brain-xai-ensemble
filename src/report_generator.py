"""
src/report_generator.py
=======================
PDF Diagnostic Report Generator for BrainTumorXAI.

Generates structured decision-support radiological reports with:
- Unique analysis traceability identifiers and timestamps.
- Explicit non-definitive predictive terminology (Model Estimates, Not Confirmed Diagnoses).
- Multi-modal XAI visual localization and multi-model probability tables.
- Mandatory research and clinical review disclaimers.
- Safe temporary image lifecycle management.
"""
import os
import io
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from src.config import logger, REPORTS_DIR, TEMP_REPORTS_DIR


def generate_pdf_report(
    diag_result: Dict[str, Any],
    patient_id: str = "PT-8942",
    patient_name: str = "Anonymous Patient",
    patient_age: str = "45",
    patient_gender: str = "Unspecified",
    output_pdf_path: Optional[str] = None,
    analysis_id: Optional[str] = None
) -> str:
    """
    Generates a structured medical decision-support radiological report as a PDF.
    
    Args:
        diag_result: Dictionary output from predict_tumor_logic (must have is_valid=True).
        patient_id: Identifier for the patient/scan.
        patient_name: Name of the patient.
        patient_age: Age of the patient.
        patient_gender: Gender of the patient.
        output_pdf_path: Optional destination file path.
        analysis_id: Optional unique analysis identifier for end-to-end traceability.
        
    Returns:
        String path to the generated PDF report.
    """
    if not diag_result or not diag_result.get("is_valid"):
        err_msg = diag_result.get("error", "Inference was invalid or failed") if diag_result else "No inference result provided"
        raise ValueError(f"Cannot generate clinical report: {err_msg}")

    # Traceability resolution
    resolved_analysis_id = analysis_id or diag_result.get("analysis_id") or f"ANL-{uuid.uuid4().hex[:10].upper()}"
    timestamp = diag_result.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reports_dir = os.path.abspath(REPORTS_DIR)
    os.makedirs(reports_dir, exist_ok=True)

    if output_pdf_path is None:
        safe_pid = "".join(c for c in patient_id if c.isalnum() or c in ("-", "_"))[:64] or "ANON"
        safe_aid = "".join(c for c in resolved_analysis_id if c.isalnum() or c in ("-", "_"))[:64]
        output_pdf_path = os.path.join(reports_dir, f"Clinical_Report_{safe_pid}_{safe_aid}.pdf")
    else:
        norm_output = os.path.abspath(output_pdf_path)
        system_dirs = ("/etc", "/usr", "/bin", "/boot", "/dev", "/lib", "/root", "/sys", "/proc", "/sbin")
        if any(norm_output.startswith(s) for s in system_dirs) or ".." in output_pdf_path:
            raise ValueError(f"Path traversal detected: target path '{output_pdf_path}' is in a restricted location.")
        output_pdf_path = norm_output

    logger.info(f"Report generation started: analysis_id={resolved_analysis_id}")
    gen_start = time.time()
    try:
        if HAS_REPORTLAB:
            result_path = _generate_reportlab_pdf(diag_result, patient_id, patient_name, patient_age, patient_gender, output_pdf_path, resolved_analysis_id, timestamp)
        else:
            result_path = _generate_matplotlib_pdf(diag_result, patient_id, patient_name, patient_age, patient_gender, output_pdf_path, resolved_analysis_id, timestamp)
        gen_duration = time.time() - gen_start
        logger.info(f"Report generation completed: analysis_id={resolved_analysis_id} duration={gen_duration:.3f}s")
        return result_path
    except Exception as e:
        gen_duration = time.time() - gen_start
        logger.error(f"Report generation failed: analysis_id={resolved_analysis_id} duration={gen_duration:.3f}s error={e}", exc_info=True)
        raise


def _save_temp_image(np_img: np.ndarray, filename: str) -> str:
    """Saves a temporary PNG image for PDF inclusion inside reports/temp."""
    temp_dir = os.path.abspath(TEMP_REPORTS_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "_", "."))
    file_path = os.path.join(temp_dir, safe_name)
    Image.fromarray(np_img.astype(np.uint8)).save(file_path, format="PNG")
    return file_path


def _generate_reportlab_pdf(
    diag_result: Dict[str, Any],
    patient_id: str,
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    pdf_path: str,
    analysis_id: str,
    timestamp: str
) -> str:
    """Builds a formatted multi-section decision-support PDF using ReportLab."""
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    c_primary = colors.HexColor("#1e3a8a")     # Deep Medical Blue
    c_secondary = colors.HexColor("#0284c7")   # Bright Cyan-Blue
    c_dark = colors.HexColor("#0f172a")        # Slate Text
    c_card = colors.HexColor("#f8fafc")        # Background Tint

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b")
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#92400e")
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_primary,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=c_dark
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>BrainTumorXAI Clinical AI Decision Support</b>", title_style),
            Paragraph(
                f"<b>Report ID:</b> RPT-{analysis_id}<br/>"
                f"<b>Analysis ID:</b> {analysis_id}<br/>"
                f"<b>Timestamp:</b> {timestamp}",
                subtitle_style
            )
        ]
    ]
    header_table = Table(header_data, colWidths=[350, 190])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=8))

    # Mandatory Prominent Clinical & Research Disclaimer Box
    disclaimer_data = [
        [Paragraph(
            "<b>INVESTIGATIONAL DECISION SUPPORT NOTICE:</b> "
            "This report is generated by an artificial intelligence decision-support tool for investigational/research use. "
            "It does NOT constitute a confirmed medical diagnosis, histopathological grade, or autonomous clinical finding. "
            "All predicted classes, attention heatmaps, and lesion measurements are model-derived estimates and MUST be "
            "independently reviewed and verified by a licensed radiologist or attending physician.",
            disclaimer_style
        )]
    ]
    disclaimer_table = Table(disclaimer_data, colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # 2. Patient Demographics & Scan Information
    story.append(Paragraph("1. PATIENT & SCAN METADATA", section_heading))
    patient_info_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style),
            Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
            Paragraph(f"<b>Age / Sex:</b> {patient_age} / {patient_gender}", body_style)
        ],
        [
            Paragraph("<b>Modality:</b> MRI Brain Scan (T1/T2 CE)", body_style),
            Paragraph(f"<b>Scope:</b> {diag_result.get('slice_context', 'Single 2D Slice Evaluation')}", body_style),
            Paragraph("<b>Diagnostic Scope:</b> Single Slice (Non-Volumetric)", body_style)
        ]
    ]
    patient_table = Table(patient_info_data, colWidths=[180, 180, 180])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # 3. Model Predictions & Biomarkers
    story.append(Paragraph("2. PREDICTIVE MODEL ESTIMATES & QUANTITATIVE FINDINGS", section_heading))
    
    cls_name = diag_result.get("class_name", "Unknown")
    confidence = diag_result.get("confidence", 0.0)
    severity = diag_result.get("severity", "N/A")
    location = diag_result.get("location", "N/A")
    tumor_pct = diag_result.get("tumor_percentage", 0.0)
    uncertainty = diag_result.get("uncertainty", 0.0)

    is_tumor = diag_result.get("is_tumor", False)
    diag_bg = colors.HexColor("#fef2f2") if is_tumor else colors.HexColor("#f0fdf4")
    diag_border = colors.HexColor("#ef4444") if is_tumor else colors.HexColor("#22c55e")

    diag_data = [
        [
            Paragraph(f"<b>PREDICTED CLASSIFICATION:</b> <font size=11 color='{diag_border.hexval()}'><b>{cls_name.upper()} (Model Estimate)</b></font>", body_style),
            Paragraph(f"<b>Ensemble Softmax Confidence:</b> <b>{confidence:.2f}%</b> (Uncalibrated)", body_style)
        ],
        [
            Paragraph(f"<b>Peak Attention Coordinates:</b> {location} (Heuristic)", body_style),
            Paragraph(f"<b>Estimated Lesion Area Fraction:</b> {tumor_pct:.2f}% of intracranial area", body_style)
        ],
        [
            Paragraph(f"<b>Image Severity Indicator:</b> <b>{severity}</b> (Heuristic — Not WHO Grade)", body_style),
            Paragraph(f"<b>Softmax Entropy (Dispersion):</b> {uncertainty:.2f}% (Softmax dispersion across categories; not calibrated clinical uncertainty)", body_style)
        ]
    ]
    diag_table = Table(diag_data, colWidths=[270, 270])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), diag_bg),
        ('BOX', (0, 0), (-1, -1), 1.2, diag_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 12))

    # 4. Multi-Modal Visual Explainability Overlays
    story.append(Paragraph("3. EXPLAINABLE AI (XAI) VISUAL LOCALIZATION", section_heading))
    
    # Save temporary images with unique analysis_id to prevent concurrency race conditions
    img_orig_path = _save_temp_image(diag_result["img"], f"orig_{analysis_id}.png")
    img_grad_path = _save_temp_image(diag_result["gradcam_overlay"], f"grad_{analysis_id}.png")
    img_seg_path = _save_temp_image(diag_result["segmentation_img"], f"seg_{analysis_id}.png")

    img_w, img_h = 165, 165
    images_table_data = [
        [
            RLImage(img_orig_path, width=img_w, height=img_h),
            RLImage(img_grad_path, width=img_w, height=img_h),
            RLImage(img_seg_path, width=img_w, height=img_h)
        ],
        [
            Paragraph("<para align=center><b>(A) Input MRI Slice</b></para>", subtitle_style),
            Paragraph("<para align=center><b>(B) Grad-CAM Attention Heatmap<br/>(Network Feature Activation)</b></para>", subtitle_style),
            Paragraph("<para align=center><b>(C) Morphological Lesion Localization<br/>(Model-Derived Contour — Not Confirmed)</b></para>", subtitle_style)
        ]
    ]
    images_table = Table(images_table_data, colWidths=[180, 180, 180])
    images_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(images_table)
    story.append(Spacer(1, 10))

    # 5. Tri-Ensemble Model Probability Breakdown
    story.append(Paragraph("4. MULTI-MODEL PROBABILITY DISTRIBUTION", section_heading))
    
    prob_headers = ["Category", "ConvNeXt (45%)", "InceptionV3 (35%)", "DenseNet121 (20%)", "Ensemble Softmax"]
    prob_rows = [prob_headers]

    classes = diag_result.get("classes", [])
    avg_preds = diag_result.get("avg_pred", [])
    pred_dict = diag_result.get("pred_dict", {})

    cnx_preds = pred_dict.get("ConvNeXtSmall", [0.0]*len(classes))
    inc_preds = pred_dict.get("InceptionV3", [0.0]*len(classes))
    den_preds = pred_dict.get("DenseNet121", [0.0]*len(classes))

    for i, cname in enumerate(classes):
        prob_rows.append([
            Paragraph(f"<b>{cname}</b>", body_style),
            f"{cnx_preds[i]*100:.2f}%",
            f"{inc_preds[i]*100:.2f}%",
            f"{den_preds[i]*100:.2f}%",
            f"<b>{avg_preds[i]*100:.2f}%</b>"
        ])

    prob_table = Table(prob_rows, colWidths=[160, 95, 95, 95, 95])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_card]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # 6. Clinical Impression & Recommendations
    story.append(Paragraph("5. AI IMPRESSION & CLINICAL RECOMMENDATIONS", section_heading))
    
    if is_tumor:
        rec_text = (
            f"<b>Model Impression:</b> Algorithmic features are suggestive of <b>{cls_name}</b> in the <b>{location}</b> with an estimated lesion area ratio of {tumor_pct:.2f}%. "
            "Grad-CAM feature activation highlights localized neural network attention matching image contrast variations.<br/>"
            "<b>Clinical Recommendation:</b> Automated findings must be correlated with clinical history, neurological examination, and dedicated multi-sequence MRI (T1, T1-CE, T2, FLAIR, DWI/ADC). "
            "Neurosurgical consultation is advised. Definitive diagnosis and WHO tumor grading require histopathological evaluation via biopsy or surgical excision."
        )
    else:
        rec_text = (
            f"<b>Model Impression:</b> The ensemble classified the provided 2D MRI slice as 'No Tumor' (Softmax confidence: {confidence:.2f}%).<br/>"
            "<b>Clinical Caution:</b> A 'No Tumor' model classification on an isolated 2D slice does NOT exclude intracranial neoplasms, low-grade diffuse processes, non-axial pathologies, or lesions on other slices. "
            "Full diagnostic multi-sequence radiological interpretation is required."
        )

    rec_para = Paragraph(rec_text, body_style)
    story.append(rec_para)
    story.append(Spacer(1, 14))

    # 7. Signature Footer
    sig_data = [
        [
            Paragraph(f"<b>AI Decision Support System:</b><br/>BrainTumorXAI v3.0 (Tri-Ensemble)<br/>Analysis ID: {analysis_id}", subtitle_style),
            Paragraph("<para align=right><b>Attending Radiologist / Reviewer:</b><br/>___________________________<br/>MD / Specialist (Signature & Date)</para>", subtitle_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('TOPPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(sig_table)

    try:
        doc.build(story)
    finally:
        # Guarantee cleanup of temporary snapshot images
        for temp_img in [img_orig_path, img_grad_path, img_seg_path]:
            try:
                if os.path.exists(temp_img):
                    os.remove(temp_img)
            except Exception:
                pass

    return pdf_path


def _generate_matplotlib_pdf(
    diag_result: Dict[str, Any],
    patient_id: str,
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    pdf_path: str,
    analysis_id: str,
    timestamp: str
) -> str:
    """Fallback generator using Matplotlib when ReportLab is unavailable."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 12))
    plt.subplots_adjust(top=0.88, hspace=0.3)

    title = (
        f"BrainTumorXAI Clinical AI Decision Support (INVESTIGATIONAL ONLY)\n"
        f"Analysis ID: {analysis_id} | Report ID: RPT-{analysis_id}\n"
        f"Patient: {patient_name} ({patient_age}y/{patient_gender}) | Timestamp: {timestamp}\n"
        "NOTICE: Automated estimates for research decision-support only. Not a confirmed medical diagnosis."
    )
    fig.suptitle(title, fontsize=10, fontweight='bold')

    axes[0, 0].imshow(diag_result["img"])
    axes[0, 0].set_title("Input MRI Scan")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(diag_result["gradcam_overlay"])
    axes[0, 1].set_title(f"Grad-CAM Attention ({diag_result.get('location', 'N/A')})")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(diag_result["segmentation_img"])
    axes[1, 0].set_title(f"Estimated Lesion Contour ({diag_result.get('tumor_percentage', 0):.2f}%)")
    axes[1, 0].axis('off')

    classes = diag_result.get("classes", [])
    avg_preds = [p * 100 for p in diag_result.get("avg_pred", [])]
    axes[1, 1].barh(classes, avg_preds, color=['#22c55e', '#ef4444', '#f59e0b', '#8b5cf6'])
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel("Ensemble Softmax Confidence (%)")
    axes[1, 1].set_title(f"Prediction: {diag_result.get('class_name')} ({diag_result.get('confidence', 0):.2f}%)")

    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", dpi=150)
    plt.close()
    return pdf_path
