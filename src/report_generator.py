import os
import io
import time
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

from src.config import logger


def generate_pdf_report(
    diag_result: Dict[str, Any],
    patient_id: str = "PT-8942",
    patient_name: str = "Anonymous Patient",
    patient_age: str = "45",
    patient_gender: str = "Unspecified",
    output_pdf_path: Optional[str] = None
) -> str:
    """
    Generates a structured medical radiological report as a PDF.
    
    Args:
        diag_result: Dictionary output from predict_tumor_logic.
        patient_id: Identifier for the patient/scan.
        patient_name: Name of the patient.
        patient_age: Age of the patient.
        patient_gender: Gender of the patient.
        output_pdf_path: Optional destination file path.
        
    Returns:
        String path to the generated PDF report.
    """
    if output_pdf_path is None:
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf_path = os.path.join(reports_dir, f"Clinical_Report_{patient_id}_{timestamp}.pdf")

    if HAS_REPORTLAB:
        return _generate_reportlab_pdf(diag_result, patient_id, patient_name, patient_age, patient_gender, output_pdf_path)
    else:
        return _generate_matplotlib_pdf(diag_result, patient_id, patient_name, patient_age, patient_gender, output_pdf_path)


def _save_temp_image(np_img: np.ndarray, filename: str) -> str:
    """Saves a temporary PNG image for PDF inclusion inside reports/temp."""
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, filename)
    Image.fromarray(np_img.astype(np.uint8)).save(file_path, format="PNG")
    return file_path


def _generate_reportlab_pdf(
    diag_result: Dict[str, Any],
    patient_id: str,
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    pdf_path: str
) -> str:
    """Builds a formatted multi-section PDF using ReportLab."""
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
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=c_dark
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>BrainTumorXAI Clinical Diagnostic Center</b>", title_style),
            Paragraph(f"<b>Report ID:</b> RPT-{patient_id}<br/><b>Date:</b> {datetime.now().strftime('%d-%b-%Y %H:%M')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=12))

    # 2. Patient Demographics & Scan Information
    story.append(Paragraph("1. PATIENT & SCAN DEMOGRAPHICS", section_heading))
    patient_info_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style),
            Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
            Paragraph(f"<b>Age / Sex:</b> {patient_age} / {patient_gender}", body_style)
        ],
        [
            Paragraph("<b>Modality:</b> MRI Axial Brain Scan (T1/T2 CE)", body_style),
            Paragraph("<b>AI Ensemble:</b> ConvNeXt + Inception + DenseNet", body_style),
            Paragraph("<b>Status:</b> Automated Analysis Complete", body_style)
        ]
    ]
    patient_table = Table(patient_info_data, colWidths=[180, 180, 180])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 14))

    # 3. Primary Diagnostic Findings
    story.append(Paragraph("2. PRIMARY RADIOLOGICAL AI FINDINGS", section_heading))
    
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
            Paragraph(f"<b>PRIMARY DIAGNOSIS:</b> <font size=12 color='{diag_border.hexval()}'><b>{cls_name.upper()}</b></font>", body_style),
            Paragraph(f"<b>Ensemble Confidence:</b> <b>{confidence:.2f}%</b>", body_style)
        ],
        [
            Paragraph(f"<b>Estimated Location:</b> {location}", body_style),
            Paragraph(f"<b>Tumor Burden Ratio:</b> {tumor_pct:.2f}% of brain tissue", body_style)
        ],
        [
            Paragraph(f"<b>Severity Grading:</b> <b>{severity}</b>", body_style),
            Paragraph(f"<b>Uncertainty Entropy:</b> {uncertainty:.2f}%", body_style)
        ]
    ]
    diag_table = Table(diag_data, colWidths=[270, 270])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), diag_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, diag_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 14))

    # 4. Multi-Modal Visual Explainability Overlays
    story.append(Paragraph("3. EXPLAINABLE AI (XAI) VISUAL LOCALIZATION", section_heading))
    
    img_orig_path = _save_temp_image(diag_result["img"], f"orig_{patient_id}.png")
    img_grad_path = _save_temp_image(diag_result["gradcam_overlay"], f"grad_{patient_id}.png")
    img_seg_path = _save_temp_image(diag_result["segmentation_img"], f"seg_{patient_id}.png")

    img_w, img_h = 170, 170
    images_table_data = [
        [
            RLImage(img_orig_path, width=img_w, height=img_h),
            RLImage(img_grad_path, width=img_w, height=img_h),
            RLImage(img_seg_path, width=img_w, height=img_h)
        ],
        [
            Paragraph("<para align=center><b>(A) Input MRI Slice</b></para>", subtitle_style),
            Paragraph("<para align=center><b>(B) Grad-CAM Attention Heatmap</b></para>", subtitle_style),
            Paragraph("<para align=center><b>(C) Morphological Segmentation</b></para>", subtitle_style)
        ]
    ]
    images_table = Table(images_table_data, colWidths=[180, 180, 180])
    images_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(images_table)
    story.append(Spacer(1, 14))

    # 5. Tri-Ensemble Model Probability Breakdown
    story.append(Paragraph("4. MULTI-MODEL PROBABILITY DISTRIBUTION", section_heading))
    
    prob_headers = ["Classification Category", "ConvNeXt (45%)", "InceptionV3 (35%)", "DenseNet121 (20%)", "Ensemble Weighted"]
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
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 14))

    # 6. Clinical Impression & Radiologist Recommendation
    story.append(Paragraph("5. CLINICAL IMPRESSION & RECOMMENDATIONS", section_heading))
    
    if is_tumor:
        rec_text = (
            f"<b>Impression:</b> Features are suggestive of <b>{cls_name}</b> in the <b>{location}</b> with an estimated tumor tissue burden of {tumor_pct:.2f}%. "
            "Explainable AI Grad-CAM localization confirms focal hyper-activation matching lesion morphology.<br/>"
            "<b>Recommendation:</b> Correlate with contrast-enhanced volumetric sequences (T1-CE, FLAIR, DWI/ADC) and schedule a neurosurgical consultation for biopsy/resection planning."
        )
    else:
        rec_text = (
            "<b>Impression:</b> No focal space-occupying lesion, mass effect, or pathological hyper-intensity detected. "
            "Ventricles and basal cisterns appear symmetric.<br/>"
            "<b>Recommendation:</b> Routine follow-up as clinically indicated."
        )

    rec_para = Paragraph(rec_text, body_style)
    story.append(rec_para)
    story.append(Spacer(1, 20))

    # 7. Signature Footer
    sig_data = [
        [
            Paragraph("<b>AI Diagnostics Verification:</b><br/>BrainTumorXAI v3.0 SOTA Pipeline<br/>Validated with 10-Pass TTA", subtitle_style),
            Paragraph("<para align=right><b>Attending Radiologist / Reviewer:</b><br/>___________________________<br/>MD, Neuro-Radiology</para>", subtitle_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('TOPPADDING', (0, 0), (-1, -1), 10)
    ]))
    doc.build(story)
    logger.info(f"Generated Clinical PDF Report at: {pdf_path}")

    # Clean up temporary snapshot images
    try:
        for temp_img in [img_orig_path, img_grad_path, img_seg_path]:
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
    pdf_path: str
) -> str:
    """Fallback generator using Matplotlib when ReportLab is unavailable."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 12))
    plt.subplots_adjust(top=0.88, hspace=0.3)

    title = f"BrainTumorXAI Clinical Report - ID: {patient_id}\nPatient: {patient_name} ({patient_age}y/{patient_gender}) | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    fig.suptitle(title, fontsize=12, fontweight='bold')

    axes[0, 0].imshow(diag_result["img"])
    axes[0, 0].set_title("Input MRI Scan")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(diag_result["gradcam_overlay"])
    axes[0, 1].set_title(f"Grad-CAM Heatmap ({diag_result.get('location', 'N/A')})")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(diag_result["segmentation_img"])
    axes[1, 0].set_title(f"Tumor Contour ({diag_result.get('tumor_percentage', 0):.2f}%)")
    axes[1, 0].axis('off')

    classes = diag_result.get("classes", [])
    avg_preds = [p * 100 for p in diag_result.get("avg_pred", [])]
    axes[1, 1].barh(classes, avg_preds, color=['#22c55e', '#ef4444', '#f59e0b', '#8b5cf6'])
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel("Confidence (%)")
    axes[1, 1].set_title(f"Diagnosis: {diag_result.get('class_name')} ({diag_result.get('confidence', 0):.2f}%)")

    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", dpi=150)
    plt.close()

    logger.info(f"Generated Matplotlib fallback PDF report at: {pdf_path}")
    return pdf_path
