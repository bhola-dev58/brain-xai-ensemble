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

    /** @type {string} Active XAI Colormap preset (jet, turbo, inferno, viridis) */
    selectedColormap: 'jet',

    /** @type {number} Saliency ROI Cutoff percentage (10 - 60) */
    saliencyCutoff: 30,
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
    cachedBaseImg = null;
    cachedHmImg = null;
    cachedHmPPImg = null;
    cachedMaskImg = null;
    updateColormapBadges(state.selectedColormap || 'jet');

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
    formData.append('use_tta', String(dom.toggleTta?.checked ?? false));
    const activeColormap = dom.selectColormap ? dom.selectColormap.value : (state.selectedColormap || 'jet');
    formData.append('colormap', activeColormap);

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
    state.lastResult = data;
    if (data.colormap) {
      state.selectedColormap = data.colormap;
      if (dom.selectColormap) dom.selectColormap.value = data.colormap;
    }
    updateColormapBadges(state.selectedColormap);

    if (data.img_cropped_b64) {
      const scanSrc = `data:image/png;base64,${data.img_cropped_b64}`;
      if (dom.imgRaw) dom.imgRaw.src = scanSrc;
      [dom.imgUnderlayGradcam, dom.imgUnderlayGradcamPP, dom.imgUnderlaySeg].forEach(img => {
        if (img) {
          img.src = scanSrc;
          img.classList.remove('hidden');
        }
      });
    }

    renderHeatmaps(data);
    renderVerdict(data);
    renderConfidenceGauge(data);
    renderBiomarkers(data, clientMs);
    renderProbabilities(data);
    appendHistoryRow(data);

    if (data.heatmap_raw_b64) {
      loadCachedHeatmapImages(
        data.img_cropped_b64,
        data.heatmap_raw_b64,
        data.heatmap_pp_raw_b64,
        data.tumor_mask_b64,
        () => {
          recolorSaliencyOverlays();
        }
      );
    }
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

  /* ============================================================
     XAI SALIENCY COLORMAPS & ROI CUTOFF ENGINE
     ============================================================ */
  const COLORMAP_LUTS = {
    jet: Uint8Array.from(atob("AACAAACEAACIAACMAACQAACUAACYAACcAACgAACkAACoAACsAACwAAC0AAC4AAC8AADAAADEAADIAADMAADQAADUAADYAADcAADgAADkAADoAADsAADwAAD0AAD4AAD8AAD/AAT/AAj/AAz/ABD/ABT/ABj/ABz/ACD/ACT/ACj/ACz/ADD/ADT/ADj/ADz/AED/AET/AEj/AEz/AFD/AFT/AFj/AFz/AGD/AGT/AGj/AGz/AHD/AHT/AHj/AHz/AID/AIT/AIj/AIz/AJD/AJT/AJj/AJz/AKD/AKT/AKj/AKz/ALD/ALT/ALj/ALz/AMD/AMT/AMj/AMz/AND/ANT/ANj/ANz/AOD/AOT/AOj/AOz/APD/APT/APj/APz/Av/+Bv/6Cv/2Dv/yEv/uFv/qGv/mHv/iIv/eJv/aKv/WLv/SMv/ONv/KOv/GPv/CQv++Rv+6Sv+2Tv+yUv+uVv+qWv+mXv+iYv+eZv+aav+Wbv+Scv+Odv+Kev+Gfv+Cgv9+hv96iv92jv9ykv9ulv9qmv9mnv9iov9epv9aqv9Wrv9Ssv9Otv9Kuv9Gvv9Cwv8+xv86yv82zv8y0v8u1v8q2v8m3v8i4v8e5v8a6v8W7v8S8v8O9v8K+v8G/v8B//wA//gA//QA//AA/+wA/+gA/+QA/+AA/9wA/9gA/9QA/9AA/8wA/8gA/8QA/8AA/7wA/7gA/7QA/7AA/6wA/6gA/6QA/6AA/5wA/5gA/5QA/5AA/4wA/4gA/4QA/4AA/3wA/3gA/3QA/3AA/2wA/2gA/2QA/2AA/1wA/1gA/1QA/1AA/0wA/0gA/0QA/0AA/zwA/zgA/zQA/zAA/ywA/ygA/yQA/yAA/xwA/xgA/xQA/xAA/wwA/wgA/wQA/wAA/AAA+AAA9AAA8AAA7AAA6AAA5AAA4AAA3AAA2AAA1AAA0AAAzAAAyAAAxAAAwAAAvAAAuAAAtAAAsAAArAAAqAAApAAAoAAAnAAAmAAAlAAAkAAAjAAAiAAAhAAAgAAA"), c => c.charCodeAt(0)),
    turbo: Uint8Array.from(atob("MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUasQkmxQku1Q066RFG/RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjR27mR3HpR3PrR3buR3jwR3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXiIMffH8ndHsvaHM3YG9DVGtLSGtTQGdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeynJ+6kKu+hLPCeL/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5fdf5cef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3sfk2tPg2t/c1ufY1vPU0vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA339834d0349s45dk459c56dU569M57NE67s8678068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks/pYr/pMq/pAp/Y0n/Yom/Icl/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ7FMP61AO6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioEyCgDxSYDwyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4BlQ0BkgsBjgoBiwkCiAgChQcCgQYCfgUCegQD"), c => c.charCodeAt(0)),
    inferno: Uint8Array.from(atob("AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgpDgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtVKwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxrTAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZuahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFriCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxgpi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQwztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k4821A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E999NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k"), c => c.charCodeAt(0)),
    viridis: Uint8Array.from(atob("RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxuSB1vSB9wSCBxSCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWBRTeBRTiCRDmDRDqDRDuEQz2EQz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2KPU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaMOFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KNM2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6OLm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WOKnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOOJYSOJYWOJIaOJIeOI4iOI4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mKH5qKHpuKHpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62BKK6AKa9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9wRsBvSMFuSsFtTMJsTsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5YcM9Xc9BWddBUd9FTetFRfNJQf9NOgdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4ovd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg+OYh++cj/ecl"), c => c.charCodeAt(0)),
  };

  let cachedBaseImg = null;
  let cachedHmImg = null;
  let cachedHmPPImg = null;
  let cachedMaskImg = null;

  function loadCachedHeatmapImages(baseB64, hmB64, hmPPB64, maskB64, onLoaded) {
    let pending = 0;
    const checkDone = () => {
      pending--;
      if (pending <= 0 && onLoaded) onLoaded();
    };

    if (baseB64) {
      pending++;
      cachedBaseImg = new Image();
      cachedBaseImg.onload = checkDone;
      cachedBaseImg.onerror = checkDone;
      cachedBaseImg.src = `data:image/png;base64,${baseB64}`;
    }
    if (hmB64) {
      pending++;
      cachedHmImg = new Image();
      cachedHmImg.onload = checkDone;
      cachedHmImg.onerror = checkDone;
      cachedHmImg.src = `data:image/png;base64,${hmB64}`;
    }
    if (hmPPB64) {
      pending++;
      cachedHmPPImg = new Image();
      cachedHmPPImg.onload = checkDone;
      cachedHmPPImg.onerror = checkDone;
      cachedHmPPImg.src = `data:image/png;base64,${hmPPB64}`;
    }
    if (maskB64) {
      pending++;
      cachedMaskImg = new Image();
      cachedMaskImg.onload = checkDone;
      cachedMaskImg.onerror = checkDone;
      cachedMaskImg.src = `data:image/png;base64,${maskB64}`;
    }
    if (pending === 0 && onLoaded) {
      onLoaded();
    }
  }

  function updateColormapBadges(colormapName) {
    const cmLabel = String(colormapName || 'jet').toUpperCase();
    if (dom.tagGatingB) dom.tagGatingB.textContent = `Brain-Gated (${cmLabel})`;
    if (dom.tagGatingC) dom.tagGatingC.textContent = `Multi-Focal (${cmLabel})`;
  }

  function blendHeatmapToCanvas(hmImg, lut, threshold, targetElement) {
    if (!hmImg || !targetElement) return;
    const w = hmImg.naturalWidth || hmImg.width || 224;
    const h = hmImg.naturalHeight || hmImg.height || 224;
    if (w === 0 || h === 0) return;

    const cHm = document.createElement('canvas');
    cHm.width = w;
    cHm.height = h;
    const ctxHm = cHm.getContext('2d');
    ctxHm.drawImage(hmImg, 0, 0, w, h);
    const hmImgData = ctxHm.getImageData(0, 0, w, h);
    const hmPx = hmImgData.data;

    const cOut = document.createElement('canvas');
    cOut.width = w;
    cOut.height = h;
    const ctxOut = cOut.getContext('2d');
    const outImgData = ctxOut.createImageData(w, h);
    const outPx = outImgData.data;

    const len = hmPx.length;

    for (let i = 0; i < len; i += 4) {
      const v = hmPx[i]; // Grayscale activation intensity [0, 255]
      if (v < threshold || v === 0) continue;

      const lutIdx = v * 3;
      outPx[i]     = lut[lutIdx];
      outPx[i + 1] = lut[lutIdx + 1];
      outPx[i + 2] = lut[lutIdx + 2];
      // Soft alpha blend so underlying anatomy is visible through the overlay
      outPx[i + 3] = Math.min(235, Math.round(95 + (v / 255) * 140));
    }

    ctxOut.putImageData(outImgData, 0, 0);
    targetElement.src = cOut.toDataURL('image/png');
  }

  function renderSegmentationToCanvas(maskImg, targetElement) {
    if (!maskImg || !targetElement) return;
    const w = maskImg.naturalWidth || maskImg.width || 224;
    const h = maskImg.naturalHeight || maskImg.height || 224;
    if (w === 0 || h === 0) return;

    const cMask = document.createElement('canvas');
    cMask.width = w;
    cMask.height = h;
    const ctxMask = cMask.getContext('2d');
    ctxMask.drawImage(maskImg, 0, 0, w, h);
    const maskImgData = ctxMask.getImageData(0, 0, w, h);
    const maskPx = maskImgData.data;

    const cOut = document.createElement('canvas');
    cOut.width = w;
    cOut.height = h;
    const ctxOut = cOut.getContext('2d');
    const outImgData = ctxOut.createImageData(w, h);
    const outPx = outImgData.data;

    const len = maskPx.length;
    for (let i = 0; i < len; i += 4) {
      const v = maskPx[i];
      if (v < 50) continue;

      // Soft red lesion fill (RGBA: 239, 68, 68, 160)
      outPx[i]     = 239;
      outPx[i + 1] = 68;
      outPx[i + 2] = 68;
      outPx[i + 3] = 160;
    }

    ctxOut.putImageData(outImgData, 0, 0);
    targetElement.src = cOut.toDataURL('image/png');
  }

  function recolorSaliencyOverlays() {
    if (!state.lastResult) return;

    const cm = state.selectedColormap || 'jet';
    const lut = COLORMAP_LUTS[cm] || COLORMAP_LUTS.jet;
    const cutoff = state.saliencyCutoff != null ? state.saliencyCutoff : 30;
    const threshold = Math.round(255 * (1 - (cutoff / 100)));

    if (cachedHmImg && dom.imgGradcam) {
      blendHeatmapToCanvas(cachedHmImg, lut, threshold, dom.imgGradcam);
    }
    if (cachedHmPPImg && dom.imgGradcamPP) {
      blendHeatmapToCanvas(cachedHmPPImg, lut, threshold, dom.imgGradcamPP);
    }
    if (cachedMaskImg && dom.imgSeg) {
      renderSegmentationToCanvas(cachedMaskImg, dom.imgSeg);
    }
  }

  dom.selectColormap?.addEventListener('change', e => {
    state.selectedColormap = e.target.value;
    updateColormapBadges(state.selectedColormap);
    recolorSaliencyOverlays();
  });

  dom.sliderSaliency?.addEventListener('input', e => {
    state.saliencyCutoff = parseInt(e.target.value, 10);
    if (dom.valSaliency) dom.valSaliency.textContent = `Top ${state.saliencyCutoff}%`;
    recolorSaliencyOverlays();
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
    state.selectedColormap = 'jet';
    if (dom.sliderSaliency) {
      dom.sliderSaliency.value = 30;
      if (dom.valSaliency) dom.valSaliency.textContent = 'Top 30%';
    }
    state.saliencyCutoff = 30;
    updateColormapBadges('jet');
    recolorSaliencyOverlays();

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
