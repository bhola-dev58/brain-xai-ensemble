/**
 * NeuroScan XAI - Clinical Radiology Workstation Client
 * Pure Vanilla JavaScript Client Engine (ES6+)
 * Connects directly to FastAPI /api REST endpoints
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── State ──────────────────────────────────────────────────────────────────
  let currentFile = null;
  let lastInferenceData = null;

  // ── DOM References ────────────────────────────────────────────────────────
  const tabs = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  const dropzone = document.getElementById('mri-dropzone');
  const fileInput = document.getElementById('mri-file-input');
  const dropzoneFilename = document.getElementById('dropzone-filename');
  const btnAnalyze = document.getElementById('btn-analyze');
  const sampleBtns = document.querySelectorAll('.sample-btn');

  // Viewport DOM
  const imgRaw = document.getElementById('img-raw');
  const imgGradcam = document.getElementById('img-gradcam');
  const imgGradcamPP = document.getElementById('img-gradcam-pp');
  const imgSegmentation = document.getElementById('img-segmentation');

  const imgUnderlayGradcam = document.getElementById('img-underlay-gradcam');
  const imgUnderlayGradcamPP = document.getElementById('img-underlay-gradcam-pp');
  const imgUnderlaySeg = document.getElementById('img-underlay-seg');

  const phRaw = document.getElementById('ph-raw');
  const phGradcam = document.getElementById('ph-gradcam');
  const phGradcamPP = document.getElementById('ph-gradcam-pp');
  const phSeg = document.getElementById('ph-seg');

  const opacitySlider = document.getElementById('opacity-slider');
  const opacityLabel = document.getElementById('opacity-label');

  // Diagnostics DOM
  const tumorStatusPill = document.getElementById('tumor-status-pill');
  const tumorStatusText = document.getElementById('tumor-status-text');
  const diagClassName = document.getElementById('diag-class-name');
  const metricConfidence = document.getElementById('metric-confidence');
  const barConfidence = document.getElementById('bar-confidence');
  const metricEntropy = document.getElementById('metric-entropy');
  const barEntropy = document.getElementById('bar-entropy');
  const metricArea = document.getElementById('metric-area');
  const barArea = document.getElementById('bar-area');
  const metricSeverity = document.getElementById('metric-severity');
  const metricLocation = document.getElementById('metric-location');
  const latencyBadge = document.getElementById('latency-badge');

  // Analytics DOM
  const probGlioma = document.getElementById('prob-glioma');
  const fillGlioma = document.getElementById('fill-glioma');
  const probMeningioma = document.getElementById('prob-meningioma');
  const fillMeningioma = document.getElementById('fill-meningioma');
  const probPituitary = document.getElementById('prob-pituitary');
  const fillPituitary = document.getElementById('fill-pituitary');
  const probNoTumor = document.getElementById('prob-notumor');
  const fillNoTumor = document.getElementById('fill-notumor');

  // PDF Report DOM
  const btnGeneratePdf = document.getElementById('btn-generate-pdf');
  const pdfDownloadWrap = document.getElementById('pdf-download-wrap');
  const pdfDownloadLink = document.getElementById('pdf-download-link');

  // System status
  const systemStatusDot = document.getElementById('system-status-dot');
  const systemStatusText = document.getElementById('system-status-text');

  // ── 1. Tab Switching ───────────────────────────────────────────────────────
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tabPanels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');

      const targetId = tab.getAttribute('data-tab');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });

  // ── 2. Health Check Ping ───────────────────────────────────────────────────
  async function checkServerHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        systemStatusDot.style.backgroundColor = 'var(--status-success)';
        systemStatusText.textContent = `Tri-Ensemble Engine Online (${data.pipeline_version || 'Active'})`;
      } else {
        throw new Error('API degraded');
      }
    } catch (e) {
      systemStatusDot.style.backgroundColor = 'var(--status-warning)';
      systemStatusText.textContent = 'REST API Offline — Start scripts/run_api.py';
    }
  }
  checkServerHealth();

  // ── 3. File Input & Drag and Drop ──────────────────────────────────────────
  function handleFileSelected(file) {
    if (!file) return;
    currentFile = file;
    dropzoneFilename.textContent = file.name;
    btnAnalyze.disabled = false;

    // Preview Raw Scan & Underlays
    const reader = new FileReader();
    reader.onload = (e) => {
      const scanDataUrl = e.target.result;
      imgRaw.src = scanDataUrl;
      imgRaw.style.display = 'block';
      phRaw.style.display = 'none';

      if (imgUnderlayGradcam) {
        imgUnderlayGradcam.src = scanDataUrl;
        imgUnderlayGradcam.style.display = 'block';
      }
      if (imgUnderlayGradcamPP) {
        imgUnderlayGradcamPP.src = scanDataUrl;
        imgUnderlayGradcamPP.style.display = 'block';
      }
      if (imgUnderlaySeg) {
        imgUnderlaySeg.src = scanDataUrl;
        imgUnderlaySeg.style.display = 'block';
      }
    };
    reader.readAsDataURL(file);

    // Reset previous heatmaps
    imgGradcam.style.display = 'none';
    phGradcam.style.display = 'flex';
    imgGradcamPP.style.display = 'none';
    phGradcamPP.style.display = 'flex';
    imgSegmentation.style.display = 'none';
    phSeg.style.display = 'flex';

    diagClassName.textContent = '--';
    tumorStatusPill.className = 'diag-status-pill status-healthy';
    tumorStatusText.textContent = 'Scan Loaded — Click Analyze';
  }

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-active');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // ── 4. Quick Sample Picker ─────────────────────────────────────────────────
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const sampleUrl = btn.getAttribute('data-sample');
      const sampleLabel = btn.getAttribute('data-label');
      dropzoneFilename.textContent = `Loading ${sampleLabel}...`;

      try {
        const response = await fetch(sampleUrl);
        const blob = await response.blob();
        const file = new File([blob], `${sampleLabel.replace(/\s+/g, '_')}.jpg`, { type: 'image/jpeg' });
        handleFileSelected(file);
      } catch (err) {
        dropzoneFilename.textContent = 'Failed to load sample image';
        console.error(err);
      }
    });
  });

  // ── 5. Run Tri-Ensemble Inference ──────────────────────────────────────────
  btnAnalyze.addEventListener('click', async () => {
    if (!currentFile) return;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `
      <svg class="spin-icon" viewBox="0 0 24 24" style="animation: spin 1s linear infinite;">
        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" stroke-dasharray="32" stroke-linecap="round"></circle>
      </svg>
      <span>Computing Tri-Ensemble XAI...</span>
    `;

    const formData = new FormData();
    formData.append('file', currentFile);

    const startTime = performance.now();

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Inference failed');
      }

      const data = await res.json();
      lastInferenceData = data;
      const totalClientMs = Math.round(performance.now() - startTime);

      renderInferenceResults(data, totalClientMs);
    } catch (err) {
      alert(`Error during clinical inference: ${err.message}`);
      console.error(err);
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = `
        <svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        <span>Run Tri-Ensemble XAI</span>
      `;
    }
  });

  // ── 6. Render Inference Results ────────────────────────────────────────────
  function renderInferenceResults(data, clientLatencyMs) {
    // 1. Heatmap Viewports
    if (data.gradcam_overlay_b64) {
      imgGradcam.src = `data:image/png;base64,${data.gradcam_overlay_b64}`;
      imgGradcam.style.display = 'block';
      phGradcam.style.display = 'none';
    }

    if (data.gradcam_pp_overlay_b64) {
      imgGradcamPP.src = `data:image/png;base64,${data.gradcam_pp_overlay_b64}`;
      imgGradcamPP.style.display = 'block';
      phGradcamPP.style.display = 'none';
    }

    if (data.segmentation_b64) {
      imgSegmentation.src = `data:image/png;base64,${data.segmentation_b64}`;
      imgSegmentation.style.display = 'block';
      phSeg.style.display = 'none';
    }

    // 2. Classification Status Pill
    if (data.is_tumor) {
      tumorStatusPill.className = 'diag-status-pill status-tumor';
      tumorStatusText.textContent = 'Tumor Detected (Pathological Finding)';
      diagClassName.style.color = 'var(--status-danger)';
    } else {
      tumorStatusPill.className = 'diag-status-pill status-healthy';
      tumorStatusText.textContent = 'Normal Brain (No Tumor Detected)';
      diagClassName.style.color = 'var(--status-success)';
    }

    diagClassName.textContent = data.class_name;

    // 3. Quantitative Biomarkers
    metricConfidence.textContent = `${data.confidence.toFixed(2)} %`;
    barConfidence.style.width = `${Math.min(100, data.confidence)}%`;

    metricEntropy.textContent = `${data.uncertainty_entropy.toFixed(2)} %`;
    barEntropy.style.width = `${Math.min(100, data.uncertainty_entropy)}%`;

    metricArea.textContent = `${data.tumor_percentage.toFixed(2)} %`;
    barArea.style.width = `${Math.min(100, data.tumor_percentage * 3)}%`;

    metricSeverity.textContent = data.severity;
    metricSeverity.style.color = data.severity_color || 'var(--text-primary)';

    metricLocation.textContent = data.location || 'Diffuse / Central';
    latencyBadge.textContent = `Server: ${(data.inference_time_seconds * 1000).toFixed(0)}ms | Total: ${clientLatencyMs}ms`;

    // 4. Probability Breakdown (Tab 2)
    const probs = data.prediction_breakdown || {};
    const gli = probs['Glioma'] || 0;
    const men = probs['Meningioma'] || 0;
    const pit = probs['Pituitary'] || 0;
    const norm = probs['No Tumor'] || 0;

    probGlioma.textContent = `${gli.toFixed(2)}%`;
    fillGlioma.style.width = `${gli}%`;

    probMeningioma.textContent = `${men.toFixed(2)}%`;
    fillMeningioma.style.width = `${men}%`;

    probPituitary.textContent = `${pit.toFixed(2)}%`;
    fillPituitary.style.width = `${pit}%`;

    probNoTumor.textContent = `${norm.toFixed(2)}%`;
    fillNoTumor.style.width = `${norm}%`;

    // 5. Append to Case History (Tab 4)
    appendHistoryRow(data);
  }

  // ── 7. Opacity Slider ──────────────────────────────────────────────────────
  opacitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    opacityLabel.textContent = `${val}%`;
    const opacityFloat = val / 100;

    imgGradcam.style.opacity = opacityFloat;
    imgGradcamPP.style.opacity = opacityFloat;
    imgSegmentation.style.opacity = opacityFloat;
  });

  // ── 8. PDF Report Generation ───────────────────────────────────────────────
  btnGeneratePdf.addEventListener('click', async () => {
    if (!currentFile) {
      alert('Please upload or select an MRI scan in the Diagnostic Studio first.');
      return;
    }

    const patientId = document.getElementById('rep-patient-id').value || 'PT-2026-9812';
    const patientName = document.getElementById('rep-patient-name').value || 'Anonymous Patient';
    const patientAge = document.getElementById('rep-patient-age').value || '50';
    const patientGender = document.getElementById('rep-patient-gender').value || 'Unspecified';

    btnGeneratePdf.disabled = true;
    btnGeneratePdf.textContent = 'Generating Clinical PDF Report...';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('patient_id', patientId);
    formData.append('patient_name', patientName);
    formData.append('patient_age', patientAge);
    formData.append('patient_gender', patientGender);

    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        throw new Error('Failed to generate report on server');
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);

      pdfDownloadLink.href = url;
      pdfDownloadLink.download = `Diagnostic_Report_${patientId}.pdf`;
      pdfDownloadWrap.style.display = 'block';

      // Auto trigger download
      pdfDownloadLink.click();
    } catch (err) {
      alert(`Report Generation Error: ${err.message}`);
    } finally {
      btnGeneratePdf.disabled = false;
      btnGeneratePdf.textContent = 'Generate Hospital-Grade PDF Report';
    }
  });

  // ── 9. Case History Logger ─────────────────────────────────────────────────
  function appendHistoryRow(data) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    const tr = document.createElement('tr');
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const statusPillClass = data.is_tumor ? 'status-tumor' : 'status-healthy';

    tr.innerHTML = `
      <td>${now}</td>
      <td>PT-${Math.floor(1000 + Math.random() * 9000)}</td>
      <td><strong>${data.class_name}</strong></td>
      <td><strong style="color: var(--accent-orange);">${data.confidence.toFixed(2)}%</strong></td>
      <td>${data.uncertainty_entropy.toFixed(2)}%</td>
      <td><span class="diag-status-pill ${statusPillClass}" style="font-size: 0.68rem; margin: 0;">${data.severity}</span></td>
    `;

    tbody.insertBefore(tr, tbody.firstChild);
  }

  // ── 10. PACS Sidebar & Workstation Controls ──────────────────────────────
  const pacsSidebar = document.getElementById('pacs-sidebar');
  const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
  const sidebarCollapseBtn = document.getElementById('sidebar-collapse-btn');
  const presetPills = document.querySelectorAll('.preset-pill');
  const canvasWraps = document.querySelectorAll('.panel-canvas-wrap');
  const toggleCrosshair = document.getElementById('toggle-crosshair');
  const togglePhi = document.getElementById('toggle-phi');
  const sliderSaliencyCutoff = document.getElementById('slider-saliency-cutoff');
  const valSaliencyCutoff = document.getElementById('val-saliency-cutoff');
  const selectColormap = document.getElementById('select-colormap');
  const toggleTta = document.getElementById('toggle-tta');
  const toggleGating = document.getElementById('toggle-gating');
  const btnRestoreDefaults = document.getElementById('btn-restore-defaults');
  const toggleControlItems = document.querySelectorAll('.toggle-control-item');
  const dotTta = document.getElementById('dot-tta');
  const dotGating = document.getElementById('dot-gating');
  const dotPhi = document.getElementById('dot-phi');
  const dotCrosshair = document.getElementById('dot-crosshair');

  function toggleSidebar() {
    if (!pacsSidebar) return;
    pacsSidebar.classList.toggle('collapsed');
  }

  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSidebar();
    });
  }

  if (sidebarCollapseBtn) {
    sidebarCollapseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSidebar();
    });
  }

  // Keyboard shortcut Ctrl+B or Cmd+B to toggle sidebar
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      toggleSidebar();
    }
  });

  // Radiological Window Presets
  const filterClasses = [
    'preset-filter-default',
    'preset-filter-brain',
    'preset-filter-high-contrast',
    'preset-filter-bone',
    'preset-filter-invert',
    'preset-filter-angio'
  ];

  function applyPresetFilter(presetKey) {
    canvasWraps.forEach(wrap => {
      filterClasses.forEach(cls => wrap.classList.remove(cls));
      wrap.classList.add(`preset-filter-${presetKey}`);
    });
  }

  presetPills.forEach(pill => {
    pill.addEventListener('click', (e) => {
      e.stopPropagation();
      presetPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const presetKey = pill.getAttribute('data-preset') || 'default';
      applyPresetFilter(presetKey);
    });
  });

  // Saliency Cutoff Slider
  if (sliderSaliencyCutoff && valSaliencyCutoff) {
    sliderSaliencyCutoff.addEventListener('input', (e) => {
      valSaliencyCutoff.textContent = `Top ${e.target.value}%`;
    });
  }

  // Precision Crosshair
  function syncCrosshairState() {
    if (!toggleCrosshair) return;
    const isChecked = toggleCrosshair.checked;
    canvasWraps.forEach(wrap => {
      if (isChecked) {
        wrap.classList.add('crosshair-active');
      } else {
        wrap.classList.remove('crosshair-active');
      }
    });
    updateIndicatorDots();
  }

  if (toggleCrosshair) {
    toggleCrosshair.addEventListener('change', syncCrosshairState);
    syncCrosshairState(); // Initialize
  }

  // Anonymize Patient PHI
  if (togglePhi) {
    togglePhi.addEventListener('change', (e) => {
      if (e.target.checked) {
        document.body.classList.add('phi-anonymized');
      } else {
        document.body.classList.remove('phi-anonymized');
      }
      updateIndicatorDots();
    });
  }

  // Status Indicator Dots for Collapsed Icon Dock
  function updateIndicatorDots() {
    if (dotTta && toggleTta) dotTta.classList.toggle('active', toggleTta.checked);
    if (dotGating && toggleGating) dotGating.classList.toggle('active', toggleGating.checked);
    if (dotPhi && togglePhi) dotPhi.classList.toggle('active', togglePhi.checked);
    if (dotCrosshair && toggleCrosshair) dotCrosshair.classList.toggle('active', toggleCrosshair.checked);
  }

  // Click on protocol item directly toggles checkbox even in collapsed state
  toggleControlItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      if (e.target.tagName && e.target.tagName.toLowerCase() !== 'input') {
        const input = item.querySelector('input[type="checkbox"]');
        if (input) {
          input.checked = !input.checked;
          input.dispatchEvent(new Event('change'));
        }
      }
      updateIndicatorDots();
    });
  });

  [toggleTta, toggleGating, togglePhi, toggleCrosshair].forEach(cb => {
    if (cb) {
      cb.addEventListener('change', updateIndicatorDots);
    }
  });

  // Colormap cycle in collapsed mode
  const colormapRow = document.querySelector('.setting-row[title="Colormap Scheme"]');
  if (colormapRow) {
    colormapRow.addEventListener('click', (e) => {
      if (pacsSidebar && pacsSidebar.classList.contains('collapsed')) {
        e.stopPropagation();
        if (selectColormap) {
          const opts = Array.from(selectColormap.options);
          const nextIdx = (selectColormap.selectedIndex + 1) % opts.length;
          selectColormap.selectedIndex = nextIdx;
          selectColormap.dispatchEvent(new Event('change'));
        }
      }
    });
  }

  // Restore Default PACS Profile
  if (btnRestoreDefaults) {
    btnRestoreDefaults.addEventListener('click', (e) => {
      e.stopPropagation();
      // 1. Reset Presets
      presetPills.forEach(p => {
        if (p.getAttribute('data-preset') === 'default') {
          p.classList.add('active');
        } else {
          p.classList.remove('active');
        }
      });
      applyPresetFilter('default');

      // 2. Reset Colormap & Cutoff
      if (selectColormap) selectColormap.value = 'jet';
      if (sliderSaliencyCutoff) {
        sliderSaliencyCutoff.value = 30;
        if (valSaliencyCutoff) valSaliencyCutoff.textContent = 'Top 30%';
      }

      // 3. Reset Protocol Toggles
      if (toggleTta) toggleTta.checked = true;
      if (toggleGating) toggleGating.checked = true;
      if (togglePhi) {
        togglePhi.checked = false;
        document.body.classList.remove('phi-anonymized');
      }
      if (toggleCrosshair) {
        toggleCrosshair.checked = true;
        syncCrosshairState();
      }
      updateIndicatorDots();

      // Visual feedback
      btnRestoreDefaults.style.color = 'var(--status-success)';
      setTimeout(() => {
        btnRestoreDefaults.style.color = '';
      }, 1000);
    });
  }
});
