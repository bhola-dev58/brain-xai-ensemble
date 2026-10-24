/**
 * NeuroScan XAI — Brain MRI Analysis Workstation
 * Client Application Engine (ES6+, Vanilla JavaScript)
 *
 * Architecture:
 *   State      — Central application state object
 *   DOM        — Cached DOM references, grouped by concern
 *   Modules    — Discrete, single-responsibility functions
 *   Init       — Bootstrap on DOMContentLoaded
 *
 * API Endpoints consumed:
 *   GET  /api/health   — System health check
 *   POST /api/predict  — Tri-ensemble XAI inference
 *   POST /api/report   — PDF report generation
 */

document.addEventListener('DOMContentLoaded', () => {

  /* ============================================================
     1. APPLICATION STATE
     ============================================================ */
  const state = {
    /** @type {File|null} Currently loaded MRI image file */
    currentFile: null,

    /** @type {Object|null} Last successful inference API response */
    lastResult: null,

    /** @type {boolean} Whether the PACS sidebar is collapsed */
    sidebarCollapsed: false,
  };


  /* ============================================================
     2. DOM REFERENCES — Grouped by concern
     ============================================================ */
  const dom = {
    // System Header
    statusDot:    document.getElementById('status-dot'),
    statusText:   document.getElementById('status-text'),

    // Sidebar
    workstation:  document.getElementById('workstation-layout'),
    sidebar:      document.getElementById('pacs-sidebar'),
    sidebarToggleBtn:   document.getElementById('sidebar-toggle-btn'),
    sidebarCollapseBtn: document.getElementById('sidebar-collapse-btn'),

    // Tab Navigation
    tabBtns:    document.querySelectorAll('.tab-btn'),
    tabPanels:  document.querySelectorAll('.tab-panel'),

    // Patient Bar
    dispPatientId:   document.getElementById('disp-patient-id'),
    dispPatientName: document.getElementById('disp-patient-name'),
    phiBadge:        document.getElementById('phi-badge'),
    phiBadgeText:    document.getElementById('phi-badge-text'),
    ttaBadge:        document.getElementById('tta-badge'),
    ttaBadgeText:    document.getElementById('tta-badge-text'),

    // Upload / Dropzone
    dropzone:      document.getElementById('mri-dropzone'),
    fileInput:     document.getElementById('mri-file-input'),
    dropzoneLabel: document.getElementById('dropzone-label'),
    sampleBtns:    document.querySelectorAll('.sample-btn'),
    btnAnalyze:    document.getElementById('btn-analyze'),

    // Quad Viewport — Raw
    imgRaw:  document.getElementById('img-raw'),
    phRaw:   document.getElementById('ph-raw'),
    wrapRaw: document.getElementById('wrap-raw'),

    // Quad Viewport — Grad-CAM
    imgUnderlayGradcam: document.getElementById('img-underlay-gradcam'),
    imgGradcam:         document.getElementById('img-gradcam'),
    phGradcam:          document.getElementById('ph-gradcam'),

    // Quad Viewport — Grad-CAM++
    imgUnderlayGradcamPP: document.getElementById('img-underlay-gradcam-pp'),
    imgGradcamPP:         document.getElementById('img-gradcam-pp'),
    phGradcamPP:          document.getElementById('ph-gradcam-pp'),

    // Quad Viewport — Segmentation
    imgUnderlaySeg: document.getElementById('img-underlay-seg'),
    imgSeg:         document.getElementById('img-segmentation'),
    phSeg:          document.getElementById('ph-seg'),

    // Overlay opacity slider
    opacitySlider: document.getElementById('opacity-slider'),
    opacityLabel:  document.getElementById('opacity-label'),

    // All viewport canvas wrappers (for crosshair & filter application)
    viewportCanvases: document.querySelectorAll('.viewport-canvas'),

    // Clinical Findings — Tab 1
    verdictPill:      document.getElementById('verdict-pill'),
    verdictText:      document.getElementById('verdict-text'),
    verdictClass:     document.getElementById('verdict-class'),
    metricConfidence: document.getElementById('metric-confidence'),
    barConfidence:    document.getElementById('bar-confidence'),
    metricEntropy:    document.getElementById('metric-entropy'),
    barEntropy:       document.getElementById('bar-entropy'),
    metricArea:       document.getElementById('metric-area'),
    barArea:          document.getElementById('bar-area'),
    metricSeverity:   document.getElementById('metric-severity'),
    metricLocation:   document.getElementById('metric-location'),
    latencyBadge:     document.getElementById('latency-badge'),

    // Analytics — Tab 2
    probGlioma:    document.getElementById('prob-glioma'),
    fillGlioma:    document.getElementById('fill-glioma'),
    probMeningioma:document.getElementById('prob-meningioma'),
    fillMeningioma:document.getElementById('fill-meningioma'),
    probPituitary: document.getElementById('prob-pituitary'),
    fillPituitary: document.getElementById('fill-pituitary'),
    probNoTumor:   document.getElementById('prob-notumor'),
    fillNoTumor:   document.getElementById('fill-notumor'),

    // Gating panel labels
    tagGatingB: document.getElementById('tag-gating-b'),
    tagGatingC: document.getElementById('tag-gating-c'),

    // PDF Report — Tab 3
    repPatientId:   document.getElementById('rep-patient-id'),
    repPatientName: document.getElementById('rep-patient-name'),
    repPatientAge:  document.getElementById('rep-patient-age'),
    repPatientGender: document.getElementById('rep-patient-gender'),
    btnGeneratePdf:   document.getElementById('btn-generate-pdf'),
    pdfDownloadWrap:  document.getElementById('pdf-download-wrap'),
    pdfDownloadLink:  document.getElementById('pdf-download-link'),

    // Case History — Tab 4
    historyTbody: document.getElementById('history-tbody'),
    historyCount: document.getElementById('history-count'),

    // PACS Controls
    presetBtns:        document.querySelectorAll('.preset-btn'),
    selectColormap:    document.getElementById('select-colormap'),
    sliderSaliency:    document.getElementById('slider-saliency'),
    valSaliency:       document.getElementById('val-saliency'),
    toggleTta:         document.getElementById('toggle-tta'),
    toggleGating:      document.getElementById('toggle-gating'),
    togglePhi:         document.getElementById('toggle-phi'),
    toggleCrosshair:   document.getElementById('toggle-crosshair'),
    dotTta:            document.getElementById('dot-tta'),
    dotGating:         document.getElementById('dot-gating'),
    dotPhi:            document.getElementById('dot-phi'),
    dotCrosshair:      document.getElementById('dot-crosshair'),
    btnRestoreDefaults: document.getElementById('btn-restore-defaults'),
    protocolToggles:   document.querySelectorAll('.protocol-toggle'),
  };


  /* ============================================================
     3. HEALTH CHECK — Verify backend availability on load
     ============================================================ */
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('API returned non-OK status');

      const data = await res.json();
      dom.statusDot.classList.add('online');
      dom.statusText.textContent =
        `Tri-Ensemble Engine Online — v${data.pipeline_version || data.version || '3.0'}`;
    } catch {
      dom.statusDot.classList.add('offline');
      dom.statusText.textContent = 'Backend Offline — Start scripts/run_api.py';
    }
  }

  checkHealth();


  /* ============================================================
     4. TAB SWITCHING
     ============================================================ */
  function switchTab(targetId) {
    dom.tabBtns.forEach(btn => {
      const isTarget = btn.dataset.tab === targetId;
      btn.classList.toggle('active', isTarget);
      btn.setAttribute('aria-selected', String(isTarget));
    });
    dom.tabPanels.forEach(panel => {
      panel.classList.toggle('active', panel.id === targetId);
    });
  }

  dom.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });


  /* ============================================================
     5. SIDEBAR — Collapse / Expand
     ============================================================ */
  function setSidebarCollapsed(collapsed) {
    state.sidebarCollapsed = collapsed;
    dom.workstation.classList.toggle('sidebar-collapsed', collapsed);
    // On mobile, toggle a special class on the sidebar itself
    if (window.innerWidth <= 640) {
      dom.sidebar.classList.toggle('mobile-open', !collapsed);
    }
  }

  function toggleSidebar() {
    setSidebarCollapsed(!state.sidebarCollapsed);
  }

  dom.sidebarToggleBtn?.addEventListener('click', e => {
    e.stopPropagation();
    toggleSidebar();
  });

  dom.sidebarCollapseBtn?.addEventListener('click', e => {
    e.stopPropagation();
    toggleSidebar();
  });

  // Keyboard shortcut: Ctrl+B / Cmd+B
  window.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      toggleSidebar();
    }
  });


  /* ============================================================
     6. DRAG & DROP — File Upload
     ============================================================ */

  /**
   * Handles a selected MRI image file: previews it in the quad-view
   * and enables the analyze button.
   * @param {File} file
   */
  function handleFileSelected(file) {
    if (!file || !file.type.startsWith('image/')) return;

    state.currentFile = file;
    dom.dropzoneLabel.textContent = file.name;
    dom.btnAnalyze.disabled = false;
    dom.btnAnalyze.removeAttribute('aria-disabled');

    // Read and preview the raw scan + underlays
    const reader = new FileReader();
    reader.onload = e => {
      const dataUrl = e.target.result;

      // Panel A: raw scan
      dom.imgRaw.src = dataUrl;
      dom.imgRaw.classList.remove('hidden');
      dom.phRaw.style.display = 'none';

      // Underlay for Grad-CAM, Grad-CAM++, Segmentation
      [dom.imgUnderlayGradcam, dom.imgUnderlayGradcamPP, dom.imgUnderlaySeg].forEach(img => {
        if (img) {
          img.src = dataUrl;
          img.classList.remove('hidden');
        }
      });
    };
    reader.readAsDataURL(file);

    // Reset previous heatmap overlays
    resetOverlays();

    // Reset findings display
    setVerdictReady();
  }

  /** Hides all heatmap overlays and restores placeholders. */
  function resetOverlays() {
    const overlays = [dom.imgGradcam, dom.imgGradcamPP, dom.imgSeg];
    const placeholders = [dom.phGradcam, dom.phGradcamPP, dom.phSeg];

    overlays.forEach(img => { if (img) img.classList.add('hidden'); });
    placeholders.forEach(ph => { if (ph) ph.style.display = ''; });
  }

  /** Resets the findings panel to its ready/idle state. */
  function setVerdictReady() {
    dom.verdictPill.className = 'verdict-pill status--ready';
    dom.verdictText.textContent = 'Scan Loaded — Click Analyze';
    dom.verdictClass.textContent = '--';
    dom.verdictClass.style.color = '';
  }

  // File input change event
  dom.fileInput.addEventListener('change', e => {
    if (e.target.files?.[0]) handleFileSelected(e.target.files[0]);
  });

  // Drag events on the dropzone
  dom.dropzone.addEventListener('dragenter', e => { e.preventDefault(); dom.dropzone.classList.add('drag-over'); });
  dom.dropzone.addEventListener('dragover',  e => { e.preventDefault(); dom.dropzone.classList.add('drag-over'); });
  dom.dropzone.addEventListener('dragleave', e => { e.preventDefault(); dom.dropzone.classList.remove('drag-over'); });
  dom.dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dom.dropzone.classList.remove('drag-over');
    if (e.dataTransfer?.files?.[0]) handleFileSelected(e.dataTransfer.files[0]);
  });

  // Keyboard accessibility for the dropzone
  dom.dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      dom.fileInput.click();
    }
  });


  /* ============================================================
     7. SAMPLE PICKER — Load benchmark images
     ============================================================ */
  dom.sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const url   = btn.dataset.sample;
      const label = btn.dataset.label;

      dom.dropzoneLabel.textContent = `Loading ${label}…`;
      btn.disabled = true;

      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        const file = new File([blob], `${label.replace(/\s+/g, '_')}.jpg`, { type: 'image/jpeg' });
        handleFileSelected(file);
      } catch (err) {
        dom.dropzoneLabel.textContent = `Failed to load "${label}"`;
        console.error('[NeuroScan] Sample load error:', err);
      } finally {
        btn.disabled = false;
      }
    });
  });


  /* ============================================================
     8. INFERENCE — Run Tri-Ensemble XAI
     ============================================================ */

  /** Sets the analyze button into loading state. */
  function setAnalyzeBtnLoading(loading) {
    dom.btnAnalyze.disabled = loading;

    if (loading) {
      dom.btnAnalyze.innerHTML = `
        <svg viewBox="0 0 24 24" style="animation:spin 0.9s linear infinite; width:16px; height:16px; stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round;">
          <circle cx="12" cy="12" r="10" stroke-dasharray="31.4" stroke-dashoffset="10"/>
        </svg>
        <span>Computing XAI…</span>
      `;
    } else {
      dom.btnAnalyze.innerHTML = `
        <svg viewBox="0 0 24 24">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
        <span>Run Tri-Ensemble XAI</span>
      `;
    }
  }

  dom.btnAnalyze.addEventListener('click', async () => {
    if (!state.currentFile) return;

    const formData = new FormData();
    formData.append('file', state.currentFile);

    const t0 = performance.now();
    setAnalyzeBtnLoading(true);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      state.lastResult = data;

      const clientMs = Math.round(performance.now() - t0);
      renderResults(data, clientMs);
    } catch (err) {
      console.error('[NeuroScan] Inference error:', err);
      alert(`Inference failed: ${err.message}`);
    } finally {
      setAnalyzeBtnLoading(false);
    }
  });


  /* ============================================================
     9. RENDER RESULTS — Update UI from API response
     ============================================================ */

  /**
   * Populates all UI panels with inference results.
   * @param {Object} data      - API /api/predict response
   * @param {number} clientMs  - Total client-side round-trip ms
   */
  function renderResults(data, clientMs) {
    renderHeatmaps(data);
    renderVerdict(data);
    renderBiomarkers(data, clientMs);
    renderProbabilities(data);
    appendHistoryRow(data);
  }

  /** Renders heatmap overlays onto the quad-view panels. */
  function renderHeatmaps(data) {
    const overlayMap = [
      { b64: data.gradcam_overlay_b64,    img: dom.imgGradcam,    ph: dom.phGradcam    },
      { b64: data.gradcam_pp_overlay_b64, img: dom.imgGradcamPP,  ph: dom.phGradcamPP  },
      { b64: data.segmentation_b64,       img: dom.imgSeg,         ph: dom.phSeg        },
    ];

    overlayMap.forEach(({ b64, img, ph }) => {
      if (b64 && img) {
        img.src = `data:image/png;base64,${b64}`;
        img.classList.remove('hidden');
        if (ph) ph.style.display = 'none';
      }
    });
  }

  /** Updates the classification verdict pill and class label. */
  function renderVerdict(data) {
    if (data.is_tumor) {
      dom.verdictPill.className = 'verdict-pill status--tumor';
      dom.verdictText.textContent = 'Pathological Finding Detected';
      dom.verdictClass.style.color = 'var(--clr-danger)';
    } else {
      dom.verdictPill.className = 'verdict-pill status--healthy';
      dom.verdictText.textContent = 'No Tumor Detected';
      dom.verdictClass.style.color = 'var(--clr-success)';
    }
    dom.verdictClass.textContent = data.class_name;
  }

  /**
   * Updates quantitative biomarker tiles.
   * @param {Object} data
   * @param {number} clientMs
   */
  function renderBiomarkers(data, clientMs) {
    // Confidence
    setMetricAndBar(
      dom.metricConfidence, dom.barConfidence,
      `${data.confidence.toFixed(2)} %`,
      Math.min(100, data.confidence)
    );

    // Shannon Entropy (uncertainty)
    setMetricAndBar(
      dom.metricEntropy, dom.barEntropy,
      `${data.uncertainty_entropy.toFixed(2)} %`,
      Math.min(100, data.uncertainty_entropy)
    );

    // Estimated lesion area
    setMetricAndBar(
      dom.metricArea, dom.barArea,
      `${data.tumor_percentage.toFixed(2)} %`,
      Math.min(100, data.tumor_percentage * 3)
    );

    // Severity
    dom.metricSeverity.textContent   = data.severity;
    dom.metricSeverity.style.color   = data.severity_color || '';

    // Anatomical location
    dom.metricLocation.textContent = data.location || 'Diffuse / Central';

    // Latency badge
    const serverMs = Math.round((data.inference_time_seconds || 0) * 1000);
    dom.latencyBadge.textContent = `Server: ${serverMs} ms | Total: ${clientMs} ms`;
  }

  /**
   * Sets a metric value element and animates its progress bar.
   * @param {HTMLElement} valEl
   * @param {HTMLElement} barEl
   * @param {string} text
   * @param {number} pct  - 0 to 100
   */
  function setMetricAndBar(valEl, barEl, text, pct) {
    if (valEl) valEl.textContent = text;
    if (barEl) barEl.style.width = `${pct}%`;
  }

  /** Updates the softmax probability bars in Tab 2. */
  function renderProbabilities(data) {
    const breakdown = data.prediction_breakdown || {};

    const classMap = [
      { label: 'Glioma',      prob: dom.probGlioma,    fill: dom.fillGlioma    },
      { label: 'Meningioma',  prob: dom.probMeningioma, fill: dom.fillMeningioma },
      { label: 'Pituitary',   prob: dom.probPituitary,  fill: dom.fillPituitary  },
      { label: 'No Tumor',    prob: dom.probNoTumor,    fill: dom.fillNoTumor    },
    ];

    classMap.forEach(({ label, prob, fill }) => {
      const pct = breakdown[label] || 0;
      if (prob) prob.textContent = `${pct.toFixed(2)}%`;
      if (fill) fill.style.width = `${pct}%`;
    });
  }


  /* ============================================================
     10. CASE HISTORY — Append new session row
     ============================================================ */

  /** Prepends a new row to the case history table. */
  function appendHistoryRow(data) {
    if (!dom.historyTbody) return;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const mrn  = `PT-${Math.floor(1000 + Math.random() * 9000)}`;
    const pillClass = data.is_tumor ? 'status--tumor' : 'status--healthy';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${time}</td>
      <td>${mrn}</td>
      <td><strong>${escapeHtml(data.class_name)}</strong></td>
      <td><strong class="val--orange">${data.confidence.toFixed(2)}%</strong></td>
      <td>${data.uncertainty_entropy.toFixed(2)}%</td>
      <td><span class="verdict-pill ${pillClass} verdict-pill--sm">${escapeHtml(data.severity)}</span></td>
    `;
    dom.historyTbody.insertBefore(tr, dom.historyTbody.firstChild);

    // Update count badge
    if (dom.historyCount) {
      const count = dom.historyTbody.querySelectorAll('tr').length;
      dom.historyCount.textContent = `${count} Scan${count !== 1 ? 's' : ''} Logged`;
    }
  }

  /**
   * Escapes HTML to prevent XSS when inserting user-influenced text.
   * @param {string} str
   * @returns {string}
   */
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Seed the initial count badge from pre-populated rows
  if (dom.historyCount && dom.historyTbody) {
    const seedCount = dom.historyTbody.querySelectorAll('tr').length;
    dom.historyCount.textContent = `${seedCount} Scans Logged`;
  }


  /* ============================================================
     11. OPACITY SLIDER — Heatmap overlay blend
     ============================================================ */
  dom.opacitySlider?.addEventListener('input', e => {
    const val = parseInt(e.target.value, 10);
    dom.opacityLabel.textContent = `${val}%`;
    const opacity = val / 100;

    [dom.imgGradcam, dom.imgGradcamPP, dom.imgSeg].forEach(img => {
      if (img) img.style.opacity = opacity;
    });
  });


  /* ============================================================
     12. RADIOLOGICAL WINDOW PRESETS
     ============================================================ */
  const PRESET_FILTER_PREFIX = 'preset-filter-';

  /** Removes all preset filter classes from all viewport canvases. */
  function clearPresetFilters() {
    dom.viewportCanvases.forEach(canvas => {
      [...canvas.classList].forEach(cls => {
        if (cls.startsWith(PRESET_FILTER_PREFIX)) canvas.classList.remove(cls);
      });
    });
  }

  dom.presetBtns.forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      dom.presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const preset = btn.dataset.preset || 'default';
      clearPresetFilters();
      if (preset !== 'default') {
        dom.viewportCanvases.forEach(canvas => {
          canvas.classList.add(`${PRESET_FILTER_PREFIX}${preset}`);
        });
      }
    });
  });


  /* ============================================================
     13. PRECISION CROSSHAIR — Mouse tracker in viewports
     ============================================================ */

  /** Binds or unbinds crosshair mouse tracking on all viewport canvases. */
  function updateCrosshairState() {
    const enabled = dom.toggleCrosshair?.checked ?? true;

    dom.viewportCanvases.forEach(canvas => {
      canvas.classList.toggle('crosshair-active', enabled);
    });

    if (dom.dotCrosshair) {
      dom.dotCrosshair.classList.toggle('active', enabled);
    }
  }

  dom.viewportCanvases.forEach(canvas => {
    const guideH = canvas.querySelector('.crosshair-h');
    const guideV = canvas.querySelector('.crosshair-v');
    const hud    = canvas.querySelector('.crosshair-hud');

    canvas.addEventListener('mousemove', e => {
      if (!dom.toggleCrosshair?.checked) return;

      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width,  e.clientX - rect.left));
      const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

      if (guideH) guideH.style.top  = `${y}px`;
      if (guideV) guideV.style.left = `${x}px`;
      if (hud)    hud.textContent   = `X: ${Math.round(x)} | Y: ${Math.round(y)}`;
    });
  });

  dom.toggleCrosshair?.addEventListener('change', updateCrosshairState);
  updateCrosshairState(); // Apply initial state


  /* ============================================================
     14. PHI ANONYMIZATION
     ============================================================ */

  /** Toggles PHI anonymization mode on the document body. */
  function updatePhiState() {
    const anonymized = dom.togglePhi?.checked ?? false;

    document.body.classList.toggle('phi-anonymized', anonymized);

    if (dom.phiBadgeText) {
      dom.phiBadgeText.textContent = anonymized ? 'HIPAA DE-IDENTIFIED' : 'PHI Visible';
    }
    if (dom.dispPatientId) {
      dom.dispPatientId.textContent = anonymized ? 'PT-XXXX-REDACTED' : 'PT-2026-9812';
    }
    if (dom.dispPatientName) {
      dom.dispPatientName.textContent = anonymized
        ? '[HIPAA DE-IDENTIFIED]'
        : 'Anonymous Patient (52Y / M)';
    }

    if (dom.dotPhi) dom.dotPhi.classList.toggle('active', anonymized);
  }

  dom.togglePhi?.addEventListener('change', updatePhiState);
  updatePhiState(); // Apply initial state


  /* ============================================================
     15. TTA TOGGLE — 5-Pass Test-Time Augmentation
     ============================================================ */

  /** Syncs TTA badge and dot indicator with toggle state. */
  function updateTtaState() {
    const enabled = dom.toggleTta?.checked ?? true;

    if (dom.ttaBadge) dom.ttaBadge.classList.toggle('disabled', !enabled);
    if (dom.ttaBadgeText) {
      dom.ttaBadgeText.textContent = enabled ? 'TTA Active (+5 passes)' : 'Single-Pass (Fast)';
    }
    if (dom.dotTta) dom.dotTta.classList.toggle('active', enabled);
  }

  dom.toggleTta?.addEventListener('change', updateTtaState);
  updateTtaState(); // Apply initial state


  /* ============================================================
     16. CALVARIAL GATING TOGGLE
     ============================================================ */

  /** Syncs panel labels with the gating toggle state. */
  function updateGatingState() {
    const gated = dom.toggleGating?.checked ?? true;

    if (dom.tagGatingB) dom.tagGatingB.textContent = gated ? 'Brain-Gated' : 'Ungated';
    if (dom.tagGatingC) dom.tagGatingC.textContent = gated ? 'Multi-Focal Gated' : 'Ungated';
    if (dom.dotGating) dom.dotGating.classList.toggle('active', gated);
  }

  dom.toggleGating?.addEventListener('change', updateGatingState);
  updateGatingState(); // Apply initial state


  /* ============================================================
     17. SALIENCY CUTOFF SLIDER
     ============================================================ */
  dom.sliderSaliency?.addEventListener('input', e => {
    if (dom.valSaliency) dom.valSaliency.textContent = `Top ${e.target.value}%`;
  });


  /* ============================================================
     18. PROTOCOL TOGGLE ITEMS — Whole-row click proxy
     ============================================================ */
  // Clicking anywhere on a toggle row (including its icon / metadata)
  // will toggle the checkbox, enabling full-row hit-area.
  dom.protocolToggles.forEach(row => {
    row.addEventListener('click', e => {
      // Don't intercept if the click was directly on the switch label / checkbox
      if (e.target.closest('.toggle-switch')) return;
      e.stopPropagation();

      const checkbox = row.querySelector('input[type="checkbox"]');
      if (checkbox) {
        checkbox.checked = !checkbox.checked;
        checkbox.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  });


  /* ============================================================
     19. RESTORE DEFAULTS
     ============================================================ */
  dom.btnRestoreDefaults?.addEventListener('click', e => {
    e.stopPropagation();

    // 1. Reset preset buttons
    dom.presetBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.preset === 'default');
    });
    clearPresetFilters();

    // 2. Reset colormap & saliency slider
    if (dom.selectColormap)  dom.selectColormap.value = 'jet';
    if (dom.sliderSaliency) {
      dom.sliderSaliency.value = 30;
      if (dom.valSaliency) dom.valSaliency.textContent = 'Top 30%';
    }

    // 3. Reset protocol toggles to defaults
    if (dom.toggleTta)        dom.toggleTta.checked = true;
    if (dom.toggleGating)     dom.toggleGating.checked = true;
    if (dom.togglePhi)        dom.togglePhi.checked = false;
    if (dom.toggleCrosshair)  dom.toggleCrosshair.checked = true;

    // 4. Sync all derived state
    updateTtaState();
    updateGatingState();
    updatePhiState();
    updateCrosshairState();

    // 5. Visual feedback on the button
    dom.btnRestoreDefaults.style.color = 'var(--clr-success)';
    setTimeout(() => { dom.btnRestoreDefaults.style.color = ''; }, 1200);
  });


  /* ============================================================
     20. PDF REPORT GENERATION
     ============================================================ */
  dom.btnGeneratePdf?.addEventListener('click', async () => {
    if (!state.currentFile) {
      alert('Please upload or select an MRI scan in the Diagnostic Studio first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', state.currentFile);
    formData.append('patient_id',     dom.repPatientId?.value   || 'PT-2026-9812');
    formData.append('patient_name',   dom.repPatientName?.value || 'Anonymous Patient');
    formData.append('patient_age',    dom.repPatientAge?.value  || '50');
    formData.append('patient_gender', dom.repPatientGender?.value || 'Unspecified');

    dom.btnGeneratePdf.disabled = true;
    dom.btnGeneratePdf.textContent = 'Generating PDF…';

    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const pid  = dom.repPatientId?.value || 'report';

      dom.pdfDownloadLink.href     = url;
      dom.pdfDownloadLink.download = `NeuroScan_Report_${pid}.pdf`;
      dom.pdfDownloadWrap.classList.remove('hidden');

      // Auto-trigger download
      dom.pdfDownloadLink.click();
    } catch (err) {
      console.error('[NeuroScan] PDF generation error:', err);
      alert(`PDF generation failed: ${err.message}`);
    } finally {
      dom.btnGeneratePdf.disabled = false;
      dom.btnGeneratePdf.innerHTML = `
        <svg viewBox="0 0 24 24">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Generate PDF Report
      `;
    }
  });

}); // end DOMContentLoaded
