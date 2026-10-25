/**
 * BrainTumorXAI — Clinical Diagnostic Studio
 * Client Application Engine (ES6+, Vanilla JavaScript)
 *
 * Architecture:
 *   ThemeManager — System-aware Dark & Light theme controller with persistence
 *   State        — Central application state object
 *   DOM          — Cached DOM references, grouped by concern
 *   Modules      — Discrete, single-responsibility functions
 *   Init         — Bootstrap on DOMContentLoaded
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
    // System Header & Theme Toggle
    statusDot:        document.getElementById('status-dot'),
    statusText:       document.getElementById('status-text'),
    themeToggleBtn:   document.getElementById('theme-toggle-btn'),
    themeLabel:       document.getElementById('theme-label'),
    btnHeaderExport:  document.getElementById('btn-header-export'),

    // Navigation Rail
    navRail:          document.querySelector('.nav-rail'),
    navRailItems:     document.querySelectorAll('.nav-rail-item'),
    navRailSettings:  document.getElementById('nav-rail-settings'),
    navRailHelp:      document.getElementById('nav-rail-help'),

    // Sidebar & Layout
    workstation:        document.getElementById('workstation-layout'),
    sidebar:            document.getElementById('pacs-sidebar'),
    sidebarToggleBtn:   document.getElementById('sidebar-toggle-btn'),
    sidebarCollapseBtn: document.getElementById('sidebar-collapse-btn'),

    // Tab Panels
    tabPanels:  document.querySelectorAll('.tab-panel'),

    // Patient Bar
    dispPatientId:   document.getElementById('disp-patient-id'),
    dispPatientName: document.getElementById('disp-patient-name'),
    phiBadge:        document.getElementById('phi-badge'),
    phiBadgeText:    document.getElementById('phi-badge-text'),
    ttaBadge:        document.getElementById('tta-badge'),
    ttaBadgeText:    document.getElementById('tta-badge-text'),

    // Upload / Dropzone & Action Buttons
    dropzone:      document.getElementById('mri-dropzone'),
    fileInput:     document.getElementById('mri-file-input'),
    dropzoneLabel: document.getElementById('dropzone-label'),
    btnBrowse:     document.getElementById('btn-browse'),
    btnAnalyze:    document.getElementById('btn-analyze'),
    btnReset:      document.getElementById('btn-reset'),
    sampleBtns:    document.querySelectorAll('.sample-btn'),

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

    // Overlay blend slider
    opacitySlider: document.getElementById('opacity-slider'),
    opacityLabel:  document.getElementById('opacity-label'),

    // All viewport canvas wrappers (for crosshair & filter application)
    viewportCanvases: document.querySelectorAll('.viewport-canvas'),

    // Confidence Circular Gauge Meter
    gaugeMeter:           document.getElementById('gauge-meter'),
    gaugeVal:             document.getElementById('gauge-val'),
    gaugeStatusMsg:       document.getElementById('gauge-status-msg'),
    confidenceVerdictTag: document.getElementById('confidence-verdict-tag'),

    // Findings Verdict & Biomarkers
    verdictPill:      document.getElementById('verdict-pill'),
    verdictText:      document.getElementById('verdict-text'),
    verdictClass:     document.getElementById('verdict-class'),
    latencyBadge:     document.getElementById('latency-badge'),
    metricConfidence: document.getElementById('metric-confidence'),
    barConfidence:    document.getElementById('bar-confidence'),
    metricEntropy:    document.getElementById('metric-entropy'),
    barEntropy:       document.getElementById('bar-entropy'),
    metricArea:       document.getElementById('metric-area'),
    barArea:          document.getElementById('bar-area'),
    metricSeverity:   document.getElementById('metric-severity'),
    metricLocation:   document.getElementById('metric-location'),

    // Probabilities Panel (Tab 1 cards)
    probMeningiomaCard: document.getElementById('prob-meningioma-card'),
    barMeningiomaCard:  document.getElementById('bar-meningioma-card'),
    probPituitaryCard:  document.getElementById('prob-pituitary-card'),
    barPituitaryCard:   document.getElementById('bar-pituitary-card'),
    probNoTumorCard:    document.getElementById('prob-notumor-card'),
    barNoTumorCard:     document.getElementById('bar-notumor-card'),

    // Analytics — Tab 2
    probGlioma:     document.getElementById('prob-glioma'),
    fillGlioma:     document.getElementById('fill-glioma'),
    probMeningioma: document.getElementById('prob-meningioma'),
    fillMeningioma: document.getElementById('fill-meningioma'),
    probPituitary:  document.getElementById('prob-pituitary'),
    fillPituitary:  document.getElementById('fill-pituitary'),
    probNoTumor:    document.getElementById('prob-notumor'),
    fillNoTumor:    document.getElementById('fill-notumor'),

    // Gating panel labels
    tagGatingB: document.getElementById('tag-gating-b'),
    tagGatingC: document.getElementById('tag-gating-c'),

    // PDF Report — Tab 3
    repPatientId:     document.getElementById('rep-patient-id'),
    repPatientName:   document.getElementById('rep-patient-name'),
    repPatientAge:    document.getElementById('rep-patient-age'),
    repPatientGender: document.getElementById('rep-patient-gender'),
    btnGeneratePdf:   document.getElementById('btn-generate-pdf'),
    pdfDownloadWrap:  document.getElementById('pdf-download-wrap'),
    pdfDownloadLink:  document.getElementById('pdf-download-link'),

    // Case History — Tab 4
    historyTbody: document.getElementById('history-tbody'),
    historyCount: document.getElementById('history-count'),

    // PACS Controls
    presetBtns:         document.querySelectorAll('.preset-btn'),
    selectColormap:     document.getElementById('select-colormap'),
    sliderSaliency:     document.getElementById('slider-saliency'),
    valSaliency:        document.getElementById('val-saliency'),
    toggleTta:          document.getElementById('toggle-tta'),
    toggleGating:       document.getElementById('toggle-gating'),
    togglePhi:          document.getElementById('toggle-phi'),
    toggleCrosshair:    document.getElementById('toggle-crosshair'),
    dotTta:             document.getElementById('dot-tta'),
    dotGating:          document.getElementById('dot-gating'),
    dotPhi:             document.getElementById('dot-phi'),
    dotCrosshair:       document.getElementById('dot-crosshair'),
    btnRestoreDefaults: document.getElementById('btn-restore-defaults'),
    protocolToggles:    document.querySelectorAll('.protocol-toggle'),

    // Help Modal
    helpModal:      document.getElementById('help-modal'),
    helpModalClose: document.getElementById('help-modal-close'),
  };


  /* ============================================================
     3. THEME MANAGER — Dark / Light Mode with Persistence
     ============================================================ */

  /**
   * Retrieves the stored or system theme.
   * @returns {'dark'|'light'}
   */
  function getStoredTheme() {
    try {
      const saved = localStorage.getItem('btxai_theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch {
      // Fallback if localStorage is inaccessible
    }
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches)
      ? 'light'
      : 'dark';
  }

  /**
   * Applies the requested theme to documentElement and updates toggle UI.
   * @param {'dark'|'light'} theme
   */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);

    try {
      localStorage.setItem('btxai_theme', theme);
    } catch {
      // localStorage may fail in strict private browsing
    }

    if (dom.themeLabel) {
      dom.themeLabel.textContent = theme === 'dark' ? 'Dark' : 'Light';
    }

    if (dom.themeToggleBtn) {
      dom.themeToggleBtn.setAttribute(
        'aria-label',
        `Current theme: ${theme}. Click to switch to ${theme === 'dark' ? 'light' : 'dark'} mode.`
      );
      dom.themeToggleBtn.setAttribute(
        'title',
        `Theme: ${theme.toUpperCase()} (Click to toggle)`
      );
    }
  }

  // Initialize theme on load
  applyTheme(getStoredTheme());

  // Click listener for theme toggle button
  dom.themeToggleBtn?.addEventListener('click', e => {
    e.stopPropagation();
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
  });

  // Listen for system appearance change if user has no saved preference
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
      if (!localStorage.getItem('btxai_theme')) {
        applyTheme(e.matches ? 'dark' : 'light');
      }
    });
  }


  /* ============================================================
     4. HEALTH CHECK — Verify backend availability on load
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

  // On mobile/tablet screens, collapse sidebar initially to prioritize diagnostic workflow
  if (window.innerWidth <= 768) {
    setSidebarCollapsed(true);
  }


  /* ============================================================
     5. TAB & NAV RAIL SWITCHING
     ============================================================ */

  /**
   * Switches the active workstation tab and synchronizes the nav rail.
   * @param {string} targetId - ID of the target .tab-panel
   */
  function switchTab(targetId) {
    // 0. Toggle diagnostic lock mode on tab-content container
    const tabContent = document.querySelector('.tab-content');
    if (tabContent) {
      tabContent.classList.toggle('is-diagnostics', targetId === 'tab-diagnostics');
    }

    // 1. Show target tab panel
    dom.tabPanels.forEach(panel => {
      panel.classList.toggle('active', panel.id === targetId);
    });

    // 2. Sync vertical nav rail active indicators and aria-selected
    dom.navRailItems.forEach(item => {
      const railTab = item.dataset.railTab;
      if (railTab && railTab !== 'scans') {
        const isTarget = railTab === targetId;
        item.classList.toggle('active', isTarget);
        if (item.getAttribute('role') === 'tab') {
          item.setAttribute('aria-selected', String(isTarget));
        }
      }
    });
  }

  // Vertical Navigation Rail interactions
  dom.navRailItems.forEach(item => {
    item.addEventListener('click', e => {
      const target = item.dataset.railTab;
      if (target === 'scans') {
        // Switch to diagnostic tab and focus on dropzone
        switchTab('tab-diagnostics');
        dom.dropzone?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        dom.dropzone?.focus();
      } else if (target) {
        switchTab(target);
      }
    });
  });

  // Settings in rail toggles PACS controls sidebar
  dom.navRailSettings?.addEventListener('click', e => {
    e.stopPropagation();
    toggleSidebar();
  });

  // Help in rail opens shortcuts modal
  dom.navRailHelp?.addEventListener('click', e => {
    e.stopPropagation();
    openHelpModal();
  });

  // Header quick "Export Report" button
  dom.btnHeaderExport?.addEventListener('click', () => {
    switchTab('tab-report');
    if (state.currentFile && dom.btnGeneratePdf) {
      dom.btnGeneratePdf.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });


  /* ============================================================
     6. SIDEBAR — Collapse / Expand
     ============================================================ */
  function setSidebarCollapsed(collapsed) {
    state.sidebarCollapsed = collapsed;
    dom.workstation.classList.toggle('sidebar-collapsed', collapsed);
    dom.sidebarCollapseBtn?.setAttribute('aria-expanded', String(!collapsed));
    dom.sidebarToggleBtn?.setAttribute('aria-expanded', String(!collapsed));
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


  /* ============================================================
     7. FILE UPLOAD & PREVIEW
     ============================================================ */

  // Browse files button triggers hidden file input
  dom.btnBrowse?.addEventListener('click', e => {
    e.stopPropagation();
    dom.fileInput?.click();
  });

  // Clicking anywhere on the dropzone triggers file picker
  dom.dropzone?.addEventListener('click', e => {
    if (e.target !== dom.btnBrowse && e.target !== dom.fileInput) {
      dom.fileInput?.click();
    }
  });

  /**
   * Handles selected MRI image file, reads preview, and prepares UI.
   * @param {File} file
   */
  function handleFileSelected(file) {
    if (!file) return;

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

    // Reset previous overlays and verdict
    resetOverlays();
    setVerdictReady();
  }

  /** Hides all heatmap overlays and restores placeholders. */
  function resetOverlays() {
    const overlays = [dom.imgGradcam, dom.imgGradcamPP, dom.imgSeg];
    const placeholders = [dom.phGradcam, dom.phGradcamPP, dom.phSeg];

    overlays.forEach(img => { if (img) img.classList.add('hidden'); });
    placeholders.forEach(ph => { if (ph) ph.style.display = ''; });
  }

  /** Resets the findings panel to ready/idle state. */
  function setVerdictReady() {
    dom.verdictPill.className = 'verdict-pill status--ready';
    dom.verdictText.textContent = 'Scan Loaded — Click Analyze';
    dom.verdictClass.textContent = '--';
    dom.verdictClass.style.color = '';

    // Reset confidence gauge
    if (dom.gaugeMeter) dom.gaugeMeter.style.strokeDashoffset = '264';
    if (dom.gaugeVal) dom.gaugeVal.textContent = '--%';
    if (dom.gaugeStatusMsg) dom.gaugeStatusMsg.textContent = 'Awaiting analysis…';
    if (dom.confidenceVerdictTag) dom.confidenceVerdictTag.textContent = 'Ready';
  }

  /** Resets the entire studio to empty state */
  function resetStudio() {
    state.currentFile = null;
    state.lastResult = null;

    if (dom.fileInput) dom.fileInput.value = '';
    dom.dropzoneLabel.textContent = 'Drag & drop MRI image here';
    dom.btnAnalyze.disabled = true;

    // Reset raw preview
    if (dom.imgRaw) {
      dom.imgRaw.src = '';
      dom.imgRaw.classList.add('hidden');
    }
    if (dom.phRaw) dom.phRaw.style.display = '';

    // Reset underlays & overlays
    [dom.imgUnderlayGradcam, dom.imgUnderlayGradcamPP, dom.imgUnderlaySeg].forEach(img => {
      if (img) {
        img.src = '';
        img.classList.add('hidden');
      }
    });
    resetOverlays();

    // Reset verdict and gauge
    dom.verdictPill.className = 'verdict-pill status--ready';
    dom.verdictText.textContent = 'System Ready';
    dom.verdictClass.textContent = '--';
    dom.verdictClass.style.color = '';

    if (dom.gaugeMeter) dom.gaugeMeter.style.strokeDashoffset = '264';
    if (dom.gaugeVal) dom.gaugeVal.textContent = '--%';
    if (dom.gaugeStatusMsg) dom.gaugeStatusMsg.textContent = 'Awaiting scan input…';
    if (dom.confidenceVerdictTag) dom.confidenceVerdictTag.textContent = '--';

    // Reset measurements
    if (dom.metricLocation) dom.metricLocation.textContent = '--';
    if (dom.metricArea) dom.metricArea.textContent = '-- %';
    if (dom.barArea) dom.barArea.style.width = '0%';
    if (dom.metricSeverity) dom.metricSeverity.textContent = '--';
    if (dom.metricEntropy) dom.metricEntropy.textContent = '-- %';
    if (dom.barEntropy) dom.barEntropy.style.width = '0%';
    if (dom.metricConfidence) dom.metricConfidence.textContent = '0.00%';
    if (dom.barConfidence) dom.barConfidence.style.width = '0%';

    // Reset sample buttons
    dom.sampleBtns.forEach(b => b.classList.remove('active'));
  }

  dom.btnReset?.addEventListener('click', resetStudio);

  // File input change event
  dom.fileInput?.addEventListener('change', e => {
    if (e.target.files?.[0]) handleFileSelected(e.target.files[0]);
  });

  // Drag & drop events
  dom.dropzone?.addEventListener('dragenter', e => { e.preventDefault(); dom.dropzone.classList.add('dragover'); });
  dom.dropzone?.addEventListener('dragover',  e => { e.preventDefault(); dom.dropzone.classList.add('dragover'); });
  dom.dropzone?.addEventListener('dragleave', e => { e.preventDefault(); dom.dropzone.classList.remove('dragover'); });
  dom.dropzone?.addEventListener('drop', e => {
    e.preventDefault();
    dom.dropzone.classList.remove('dragover');
    if (e.dataTransfer?.files?.[0]) handleFileSelected(e.dataTransfer.files[0]);
  });

  // Dropzone keyboard navigation
  dom.dropzone?.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      dom.fileInput?.click();
    }
  });


  /* ============================================================
     8. SAMPLE BENCHMARK PICKER
     ============================================================ */
  dom.sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      dom.sampleBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

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
        console.error('[BrainTumorXAI] Sample load error:', err);
      } finally {
        btn.disabled = false;
      }
    });
  });


  /* ============================================================
     9. INFERENCE — Run Tri-Ensemble XAI
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
        <span>Analyze MRI Scan</span>
      `;
    }
  }

  dom.btnAnalyze?.addEventListener('click', async () => {
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
      console.error('[BrainTumorXAI] Inference error:', err);
      alert(`Inference failed: ${err.message}`);
    } finally {
      setAnalyzeBtnLoading(false);
    }
  });


  /* ============================================================
     10. RENDER RESULTS — Update UI from API response
     ============================================================ */

  /**
   * Populates all UI panels with inference results.
   * @param {Object} data      - API /api/predict response
   * @param {number} clientMs  - Total client round-trip ms
   */
  function renderResults(data, clientMs) {
    renderHeatmaps(data);
    renderVerdict(data);
    renderConfidenceGauge(data);
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
      dom.verdictText.textContent = 'Pathological Finding';
      dom.verdictClass.style.color = 'var(--clr-danger)';
    } else {
      dom.verdictPill.className = 'verdict-pill status--healthy';
      dom.verdictText.textContent = 'No Tumor Detected';
      dom.verdictClass.style.color = 'var(--clr-success)';
    }
    dom.verdictClass.textContent = data.class_name;
  }

  /** Animates the circular confidence gauge meter */
  function renderConfidenceGauge(data) {
    const conf = Math.max(0, Math.min(100, data.confidence || 0));
    // Circumference of r=42 is 2 * PI * 42 = 263.89
    const offset = 264 - (conf / 100) * 264;

    if (dom.gaugeMeter) {
      dom.gaugeMeter.style.strokeDashoffset = String(offset);
      dom.gaugeMeter.style.stroke = data.is_tumor ? 'var(--clr-accent)' : 'var(--clr-success)';
    }

    if (dom.gaugeVal) {
      dom.gaugeVal.textContent = `${conf.toFixed(1)}%`;
    }

    if (dom.gaugeStatusMsg) {
      dom.gaugeStatusMsg.textContent = `${data.class_name} · ${data.severity}`;
    }

    if (dom.confidenceVerdictTag) {
      dom.confidenceVerdictTag.textContent = data.is_tumor ? 'Tumor Confirmed' : 'Benign / Clear';
      dom.confidenceVerdictTag.style.color = data.is_tumor ? 'var(--clr-accent)' : 'var(--clr-success)';
    }
  }

  /** Updates quantitative biomarker measurements. */
  function renderBiomarkers(data, clientMs) {
    // Estimated location
    if (dom.metricLocation) {
      dom.metricLocation.textContent = data.location || 'Diffuse / Central';
    }

    // Tumor Tissue Area
    if (dom.metricArea) {
      dom.metricArea.textContent = `${(data.tumor_percentage || 0).toFixed(2)} %`;
    }
    if (dom.barArea) {
      dom.barArea.style.width = `${Math.min(100, (data.tumor_percentage || 0) * 3)}%`;
    }

    // Severity
    if (dom.metricSeverity) {
      dom.metricSeverity.textContent = data.severity || '--';
      dom.metricSeverity.style.color = data.severity_color || '';
    }

    // Uncertainty Entropy
    if (dom.metricEntropy) {
      dom.metricEntropy.textContent = `${(data.uncertainty_entropy || 0).toFixed(2)} %`;
    }
    if (dom.barEntropy) {
      dom.barEntropy.style.width = `${Math.min(100, data.uncertainty_entropy || 0)}%`;
    }

    // Latency badge
    const serverMs = Math.round((data.inference_time_seconds || 0) * 1000);
    if (dom.latencyBadge) {
      dom.latencyBadge.textContent = `Server: ${serverMs} ms | Total: ${clientMs} ms`;
    }
  }

  /** Updates probability distribution bars across tabs 1 & 2 */
  function renderProbabilities(data) {
    const breakdown = data.prediction_breakdown || {};

    const classMap = [
      { label: 'Glioma',      prob: dom.probGlioma,     fill: dom.fillGlioma     },
      { label: 'Meningioma',  prob: dom.probMeningioma, fill: dom.fillMeningioma  },
      { label: 'Pituitary',   prob: dom.probPituitary,  fill: dom.fillPituitary   },
      { label: 'No Tumor',    prob: dom.probNoTumor,    fill: dom.fillNoTumor     },
    ];

    classMap.forEach(({ label, prob, fill }) => {
      const pct = breakdown[label] || 0;
      if (prob) prob.textContent = `${pct.toFixed(2)}%`;
      if (fill) fill.style.width = `${pct}%`;
    });

    // Update Tab 1 cards
    if (dom.metricConfidence) dom.metricConfidence.textContent = `${(breakdown['Glioma'] || 0).toFixed(2)}%`;
    if (dom.barConfidence) dom.barConfidence.style.width = `${breakdown['Glioma'] || 0}%`;

    if (dom.probMeningiomaCard) dom.probMeningiomaCard.textContent = `${(breakdown['Meningioma'] || 0).toFixed(2)}%`;
    if (dom.barMeningiomaCard) dom.barMeningiomaCard.style.width = `${breakdown['Meningioma'] || 0}%`;

    if (dom.probPituitaryCard) dom.probPituitaryCard.textContent = `${(breakdown['Pituitary'] || 0).toFixed(2)}%`;
    if (dom.barPituitaryCard) dom.barPituitaryCard.style.width = `${breakdown['Pituitary'] || 0}%`;

    if (dom.probNoTumorCard) dom.probNoTumorCard.textContent = `${(breakdown['No Tumor'] || 0).toFixed(2)}%`;
    if (dom.barNoTumorCard) dom.barNoTumorCard.style.width = `${breakdown['No Tumor'] || 0}%`;
  }

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

    if (dom.historyCount) {
      const count = dom.historyTbody.querySelectorAll('tr').length;
      dom.historyCount.textContent = `${count} Case${count !== 1 ? 's' : ''}`;
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }


  /* ============================================================
     11. OPACITY SLIDER — Heatmap overlay blend
     ============================================================ */
  dom.opacitySlider?.addEventListener('input', e => {
    const val = parseInt(e.target.value, 10);
    if (dom.opacityLabel) dom.opacityLabel.textContent = `${val}%`;
    const opacity = val / 100;

    [dom.imgGradcam, dom.imgGradcamPP, dom.imgSeg].forEach(img => {
      if (img) img.style.opacity = opacity;
    });
  });


  /* ============================================================
     12. RADIOLOGICAL WINDOW PRESETS
     ============================================================ */
  const PRESET_FILTER_PREFIX = 'preset-filter-';

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
     13. PRECISION CROSSHAIR
     ============================================================ */
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
  updateCrosshairState();


  /* ============================================================
     14. PHI ANONYMIZATION, TTA, GATING TOGGLES
     ============================================================ */
  function updatePhiState() {
    const anonymized = dom.togglePhi?.checked ?? false;
    document.body.classList.toggle('phi-anonymized', anonymized);

    if (dom.phiBadgeText) dom.phiBadgeText.textContent = anonymized ? 'HIPAA REDACTED' : 'PHI Visible';
    if (dom.dispPatientId) dom.dispPatientId.textContent = anonymized ? 'PT-XXXX-REDACTED' : 'PT-2026-9812';
    if (dom.dispPatientName) dom.dispPatientName.textContent = anonymized ? '[REDACTED]' : 'Anonymous Patient (52Y / M)';
    if (dom.dotPhi) dom.dotPhi.classList.toggle('active', anonymized);
  }

  dom.togglePhi?.addEventListener('change', updatePhiState);
  updatePhiState();

  function updateTtaState() {
    const enabled = dom.toggleTta?.checked ?? true;
    if (dom.ttaBadge) dom.ttaBadge.classList.toggle('disabled', !enabled);
    if (dom.ttaBadgeText) dom.ttaBadgeText.textContent = enabled ? 'TTA Active' : 'Single-Pass';
    if (dom.dotTta) dom.dotTta.classList.toggle('active', enabled);
  }

  dom.toggleTta?.addEventListener('change', updateTtaState);
  updateTtaState();

  function updateGatingState() {
    const gated = dom.toggleGating?.checked ?? true;
    if (dom.tagGatingB) dom.tagGatingB.textContent = gated ? 'Brain-Gated' : 'Ungated';
    if (dom.tagGatingC) dom.tagGatingC.textContent = gated ? 'Multi-Focal Gated' : 'Ungated';
    if (dom.dotGating) dom.dotGating.classList.toggle('active', gated);
  }

  dom.toggleGating?.addEventListener('change', updateGatingState);
  updateGatingState();

  dom.sliderSaliency?.addEventListener('input', e => {
    if (dom.valSaliency) dom.valSaliency.textContent = `Top ${e.target.value}%`;
  });

  // Proxy clicks on full toggle row
  dom.protocolToggles.forEach(row => {
    row.addEventListener('click', e => {
      if (e.target.closest('.toggle-switch')) return;
      e.stopPropagation();
      const checkbox = row.querySelector('input[type="checkbox"]');
      if (checkbox) {
        checkbox.checked = !checkbox.checked;
        checkbox.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  });

  // Restore Defaults button
  dom.btnRestoreDefaults?.addEventListener('click', e => {
    e.stopPropagation();
    dom.presetBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === 'default'));
    clearPresetFilters();

    if (dom.selectColormap) dom.selectColormap.value = 'jet';
    if (dom.sliderSaliency) {
      dom.sliderSaliency.value = 30;
      if (dom.valSaliency) dom.valSaliency.textContent = 'Top 30%';
    }

    if (dom.toggleTta)       dom.toggleTta.checked = true;
    if (dom.toggleGating)    dom.toggleGating.checked = true;
    if (dom.togglePhi)       dom.togglePhi.checked = false;
    if (dom.toggleCrosshair) dom.toggleCrosshair.checked = true;

    updateTtaState();
    updateGatingState();
    updatePhiState();
    updateCrosshairState();
  });


  /* ============================================================
     15. PDF REPORT GENERATION
     ============================================================ */
  dom.btnGeneratePdf?.addEventListener('click', async () => {
    if (!state.currentFile) {
      alert('Please upload or select an MRI scan in the Diagnostic Studio first.');
      switchTab('tab-diagnostics');
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
      dom.pdfDownloadLink.download = `BrainTumorXAI_Report_${pid}.pdf`;
      dom.pdfDownloadWrap.classList.remove('hidden');

      dom.pdfDownloadLink.click();
    } catch (err) {
      console.error('[BrainTumorXAI] PDF error:', err);
      alert(`PDF generation failed: ${err.message}`);
    } finally {
      dom.btnGeneratePdf.disabled = false;
      dom.btnGeneratePdf.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Generate Clinical PDF
      `;
    }
  });


  /* ============================================================
     16. HELP MODAL & SHORTCUTS
     ============================================================ */
  function openHelpModal() {
    dom.helpModal?.classList.add('active');
    dom.helpModalClose?.focus();
  }

  function closeHelpModal() {
    dom.helpModal?.classList.remove('active');
    dom.navRailHelp?.focus();
  }

  dom.helpModalClose?.addEventListener('click', closeHelpModal);
  dom.helpModal?.addEventListener('click', e => {
    if (e.target === dom.helpModal) closeHelpModal();
  });

  // Global Keyboard Shortcuts
  window.addEventListener('keydown', e => {
    // Ctrl+B / Cmd+B: Toggle Sidebar
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      toggleSidebar();
    }
    // Ctrl+T / Cmd+T / Alt+T: Toggle Theme
    if ((e.altKey || ((e.ctrlKey || e.metaKey) && e.shiftKey)) && e.key.toLowerCase() === 't') {
      e.preventDefault();
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    }
    // Escape: Close modal or Reset
    if (e.key === 'Escape') {
      if (dom.helpModal?.classList.contains('active')) {
        closeHelpModal();
      }
    }
  });

}); // end DOMContentLoaded
