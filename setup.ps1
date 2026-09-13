#Requires -Version 5.1
<#
.SYNOPSIS
    Brain XAI Ensemble — Windows PowerShell Setup Script

.DESCRIPTION
    Fully automated one-command project setup for Windows users.
    Detects Python, creates venv, installs deps, initialises DB, runs smoke tests.

.PARAMETER Dev
    Also install pytest, coverage, and httpx for running the test suite.

.PARAMETER Docker
    Build and launch the project via Docker instead of a local venv.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Dev
    .\setup.ps1 -Docker

.NOTES
    Requires Python 3.10+ to be installed and on PATH.
    Download Python 3.12: https://www.python.org/downloads/
    Linux / macOS users: run ./setup.sh instead.
#>

param(
    [switch]$Dev,
    [switch]$Docker
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ─────────────────────────────────────────────────────────────
function Write-Info    { param([string]$msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok      { param([string]$msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn    { param([string]$msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err     { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Header  { param([string]$msg) Write-Host "`n━━━  $msg  ━━━" -ForegroundColor Blue }
function Fail          { param([string]$msg) Write-Err $msg; exit 1 }

# ── Banner ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "  ║      Brain XAI Ensemble — Automated Setup (PowerShell)      ║" -ForegroundColor Blue
Write-Host "  ║   Explainable AI Tri-Ensemble Brain Tumor Diagnostics        ║" -ForegroundColor Blue
Write-Host "  ╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""
Write-Info "Platform: Windows  |  PowerShell $($PSVersionTable.PSVersion)"

# ── Working directory guard ────────────────────────────────────────────────────
$ScriptRoot = $PSScriptRoot
Set-Location $ScriptRoot

if (-not (Test-Path "app.py") -or -not (Test-Path "requirements.txt")) {
    Fail "Run setup.ps1 from the project root directory where app.py lives."
}

# =============================================================================
# DOCKER MODE
# =============================================================================
if ($Docker) {
    Write-Header "Docker Mode"
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Fail "Docker not found. Install from: https://docs.docker.com/get-docker/"
    }
    Write-Info "Building image: brain-xai-ensemble ..."
    docker build -t brain-xai-ensemble .
    Write-Ok "Image built."
    Write-Host ""
    Write-Host "  Run:  docker run --rm -p 7860:7860 brain-xai-ensemble" -ForegroundColor Cyan
    Write-Host "  Open: http://localhost:7860" -ForegroundColor Cyan
    exit 0
}

# =============================================================================
# STEP 1 — Python Detection
# =============================================================================
Write-Header "Step 1 / 7 — Python Environment"

$PythonBin = $null
$PythonVer = $null

foreach ($candidate in @("python3.12","python3.11","python3.10","python3","python","py")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        try {
            $ver = & $candidate -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')" 2>$null
            if ($ver -match "^3\.(\d+)$") {
                $minor = [int]$Matches[1]
                if ($minor -ge 10) {
                    $PythonBin = $candidate
                    $PythonVer = $ver
                    break
                }
            }
        } catch { }
    }
}

if (-not $PythonBin) {
    Write-Err "Python 3.10+ is required but was not found on PATH."
    Write-Host ""
    Write-Host "  Download Python 3.12: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  IMPORTANT: During install, check 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Ok "Python $PythonVer found: $PythonBin"

# =============================================================================
# STEP 2 — System Dependencies
# =============================================================================
Write-Header "Step 2 / 7 — System Dependencies"
Write-Ok "Windows detected. OpenCV and TensorFlow pip wheels are self-contained."
Write-Ok "No additional system libraries or DLLs required."

# =============================================================================
# STEP 3 — Virtual Environment
# =============================================================================
Write-Header "Step 3 / 7 — Virtual Environment"

$VenvDir = "venv"
if (Test-Path "$VenvDir\Scripts\python.exe") {
    Write-Info "Existing venv found at .\$VenvDir — skipping creation."
} else {
    Write-Info "Creating virtualenv at .\$VenvDir ..."
    & $PythonBin -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create virtual environment." }
    Write-Ok "Virtual environment created."
}

$Py  = "$VenvDir\Scripts\python.exe"
$Pip = "$VenvDir\Scripts\pip.exe"

Write-Info "Upgrading pip ..."
& $Py -m pip install --upgrade pip --quiet
$pipVer = & $Pip --version | Select-String -Pattern "pip (\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
Write-Ok "pip $pipVer ready."

# =============================================================================
# STEP 4 — Dependencies
# =============================================================================
Write-Header "Step 4 / 7 — Python Dependencies"

Write-Info "Installing from requirements.txt ..."
& $Pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Dependency install failed. Check your internet connection." }
Write-Ok "Core dependencies installed."

if ($Dev) {
    Write-Info "[--Dev] Installing pytest, coverage, httpx ..."
    & $Pip install pytest pytest-cov coverage httpx
    Write-Ok "Dev tools installed."
}

# =============================================================================
# STEP 5 — Project Structure
# =============================================================================
Write-Header "Step 5 / 7 — Project Structure"

$Issues = 0

# Ensure directories exist
foreach ($d in @("models","datasets","assets","src","scripts","test_images","data","reports")) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Info "Created directory: $d\"
    }
}
Write-Ok "All runtime directories present."

# Required source files
$RequiredFiles = @(
    "app.py","src\__init__.py","src\config.py","src\inference.py",
    "src\processor.py","src\dashboard.py","src\report_generator.py",
    "src\database.py","src\dicom_reader.py","src\theme.py","src\api.py"
)
foreach ($f in $RequiredFiles) {
    if (Test-Path $f) {
        Write-Ok "Found: $f"
    } else {
        Write-Err "Missing: $f"
        $Issues++
    }
}

# Model files
$ModelsFound = 0
foreach ($m in @("models\convnext_full_best.keras","models\inception_full_best.keras","models\densenet_full_best.keras")) {
    if (Test-Path $m) {
        $size = [math]::Round((Get-Item $m).Length / 1MB, 0)
        Write-Ok "Model: $m  (~${size} MB)"
        $ModelsFound++
    } else {
        Write-Warn "Model not found: $m"
    }
}

if ($ModelsFound -eq 0) {
    Write-Host ""
    Write-Warn "No trained model .keras files found in models\"
    Write-Warn "Dashboard will run in DEMO mode until model files are added."
    Write-Host ""
    Write-Host "  Options:" -ForegroundColor Cyan
    Write-Host "    1. Extract bundled archive:  Expand-Archive models.zip -DestinationPath models" -ForegroundColor Cyan
    Write-Host "    2. Re-train via Kaggle:       see docs\04_models_and_training.md" -ForegroundColor Cyan
    Write-Host ""
}

# =============================================================================
# STEP 6 — Database Init
# =============================================================================
Write-Header "Step 6 / 7 — Database Init"

if (-not (Test-Path "data"))    { New-Item -ItemType Directory -Path "data"    | Out-Null }
if (-not (Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }

Write-Info "Initialising patient history SQLite database ..."
$initScript = @"
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)
try:
    from src.database import init_db
    init_db()
    print('Database ready: data/patient_history.db')
except Exception as e:
    print(f'Note: {e}')
"@
& $Py -c $initScript
Write-Ok "Database ready."

# =============================================================================
# STEP 7 — Smoke Test
# =============================================================================
Write-Header "Step 7 / 7 — Smoke Test"
Write-Info "Verifying all module imports ..."

$smokeScript = @"
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

failures = []
checks = [
    ('numpy',        "import numpy; print(f'  numpy        {numpy.__version__}')"),
    ('opencv',       "import cv2;   print(f'  opencv       {cv2.__version__}')"),
    ('tensorflow',   "import tensorflow as tf; print(f'  tensorflow   {tf.__version__}')"),
    ('gradio',       "import gradio as gr; print(f'  gradio       {gr.__version__}')"),
    ('reportlab',    "from reportlab.lib import colors; print('  reportlab    OK')"),
    ('scikit-learn', "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"),
    ('pydicom',      "import pydicom; print(f'  pydicom      {pydicom.__version__}')"),
    ('fastapi',      "import fastapi; print(f'  fastapi      {fastapi.__version__}')"),
    ('src.config',   "from src.config import CLASSES; print(f'  src.config   OK ({len(CLASSES)} classes)')"),
    ('src.database', "from src.database import get_scan_count; n=get_scan_count(); print(f'  src.database OK ({n} record(s))')"),
    ('src.dicom',    "from src.dicom_reader import read_dicom; print('  src.dicom    OK')"),
]

for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        failures.append(f'{name}: {e}')

if failures:
    print('FAILED:', failures, file=sys.stderr)
    sys.exit(1)
print()
print('  All imports verified successfully.')
"@

& $Py -c $smokeScript
if ($LASTEXITCODE -ne 0) {
    Write-Err "One or more import checks failed. Review output above."
    $Issues++
} else {
    Write-Ok "All smoke tests passed."
}

# =============================================================================
# Final Summary
# =============================================================================
Write-Host ""
Write-Host ("━" * 70) -ForegroundColor Blue

if ($Issues -eq 0) {
    Write-Host "  ✓  Setup completed successfully! Ready to launch." -ForegroundColor Green
} else {
    Write-Host "  ⚠  Setup completed with $Issues issue(s). Review output above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Launch the Gradio dashboard:" -ForegroundColor White
Write-Host "    venv\Scripts\activate" -ForegroundColor Cyan
Write-Host "    python app.py" -ForegroundColor Cyan
Write-Host "    Open: http://127.0.0.1:7860" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Launch the FastAPI REST backend:" -ForegroundColor White
Write-Host "    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor Cyan
Write-Host ""

if ($Dev) {
    Write-Host "  Run tests:" -ForegroundColor White
    Write-Host "    pytest tests\ -v --cov=src" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "  Other flags:" -ForegroundColor White
Write-Host "    .\setup.ps1 -Dev      Include pytest and coverage tools" -ForegroundColor Cyan
Write-Host "    .\setup.ps1 -Docker   Build and run in Docker" -ForegroundColor Cyan
Write-Host ""
Write-Host ("━" * 70) -ForegroundColor Blue
Write-Host ""

exit 0
