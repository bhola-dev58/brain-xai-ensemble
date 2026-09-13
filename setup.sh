#!/usr/bin/env bash
# =============================================================================
#  Brain XAI Ensemble — Automated Project Setup Script
#  GitHub: https://github.com/bhola-dev58/brain-xai-ensemble
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh            # Full interactive setup
#    ./setup.sh --dev      # Same, but also installs dev/test dependencies
#    ./setup.sh --docker   # Build and run via Docker instead of venv
#    ./setup.sh --help     # Show usage
#
#  Requirements:
#    - Linux / macOS (Ubuntu 20.04+ recommended)
#    - Python 3.10, 3.11, or 3.12
#    - git (for cloning)
#    - Internet connection (first run only)
#
#  After setup, run the dashboard with:
#    source venv/bin/activate && python3 app.py
# =============================================================================

set -euo pipefail   # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'

# ── Colour helpers ─────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; RESET=''
fi

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}━━━  $*  ━━━${RESET}"; }
die()     { error "$*"; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────
MODE="standard"    # standard | dev | docker
for arg in "$@"; do
    case "$arg" in
        --dev)    MODE="dev" ;;
        --docker) MODE="docker" ;;
        --help|-h)
            echo ""
            echo "  ${BOLD}Brain XAI Ensemble — Setup Script${RESET}"
            echo ""
            echo "  Usage:"
            echo "    ./setup.sh           Full automated setup (virtualenv + dependencies)"
            echo "    ./setup.sh --dev     Same + installs pytest, coverage, and dev tools"
            echo "    ./setup.sh --docker  Build and launch via Docker container"
            echo "    ./setup.sh --help    Show this help message"
            echo ""
            exit 0
            ;;
        *) warn "Unknown argument '$arg' — ignoring." ;;
    esac
done

# ── Banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${BLUE}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║        Brain XAI Ensemble — Automated Project Setup         ║"
echo "  ║   Explainable AI Tri-Ensemble for Brain Tumor Diagnostics   ║"
echo "  ║         github.com/bhola-dev58/brain-xai-ensemble           ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Working directory guard ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    die "setup.sh must be run from the project root directory (where app.py exists)."
fi

# ═════════════════════════════════════════════════════════════════════════════
# DOCKER MODE
# ═════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "docker" ]; then
    header "Docker Mode"

    if ! command -v docker &>/dev/null; then
        die "Docker is not installed. Install it from https://docs.docker.com/get-docker/ and retry."
    fi

    info "Docker version: $(docker --version)"

    if [ ! -d "models" ] || [ -z "$(ls -A models/*.keras 2>/dev/null)" ]; then
        warn "models/ directory is empty or missing .keras files."
        warn "Download and extract models.zip before building:"
        warn "  unzip models.zip -d models/"
    fi

    info "Building Docker image: brain-xai-ensemble..."
    docker build -t brain-xai-ensemble .

    success "Docker image built successfully."
    echo ""
    info "To run the dashboard:"
    echo -e "    ${BOLD}docker run --rm -p 7860:7860 brain-xai-ensemble${RESET}"
    echo ""
    echo -e "    Then open: ${CYAN}http://localhost:7860${RESET}"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# STANDARD / DEV SETUP
# ═════════════════════════════════════════════════════════════════════════════

# ── Step 1: Python version check ───────────────────────────────────────────────
header "Step 1 / 7 — Python Environment Check"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        VERSION=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            info "Using Python $VERSION  ($(which $candidate))"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    die "Python 3.10 or higher is required. Install it from https://www.python.org/downloads/"
fi

# ── Step 2: System dependencies check ─────────────────────────────────────────
header "Step 2 / 7 — System Dependencies Check"

MISSING_SYS=()
for pkg in libgl1 libglib2.0-0; do
    if ! dpkg -l "$pkg" &>/dev/null 2>&1; then
        MISSING_SYS+=("$pkg")
    fi
done

if [ ${#MISSING_SYS[@]} -gt 0 ]; then
    warn "Missing system packages required by OpenCV: ${MISSING_SYS[*]}"
    if command -v apt-get &>/dev/null; then
        info "Installing missing system packages (sudo required)..."
        sudo apt-get install -y --no-install-recommends "${MISSING_SYS[@]}" \
            || warn "Could not install system packages automatically. Install manually if OpenCV fails."
    else
        warn "apt-get not found. Install manually: ${MISSING_SYS[*]}"
    fi
else
    success "All system dependencies present (libgl1, libglib2.0-0)."
fi

# ── Step 3: Virtual environment ────────────────────────────────────────────────
header "Step 3 / 7 — Virtual Environment"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    info "Virtual environment already exists at ./$VENV_DIR — skipping creation."
else
    info "Creating virtual environment at ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Upgrade pip silently
info "Upgrading pip..."
"$PIP" install --upgrade pip --quiet
success "pip upgraded to $("$PIP" --version | awk '{print $2}')"

# ── Step 4: Install Python dependencies ───────────────────────────────────────
header "Step 4 / 7 — Python Dependencies"

info "Installing from requirements.txt..."
"$PIP" install -r requirements.txt
success "All core dependencies installed."

if [ "$MODE" = "dev" ]; then
    info "[--dev] Installing development & testing tools..."
    "$PIP" install pytest pytest-cov coverage httpx
    success "Dev tools installed: pytest, coverage, httpx."
fi

# ── Step 5: Runtime structure verification ────────────────────────────────────
header "Step 5 / 7 — Project Structure Verification"

ISSUES=0

# Required directories
for dir in models datasets assets src scripts test_images data; do
    if [ ! -d "$dir" ]; then
        info "Creating missing directory: $dir/"
        mkdir -p "$dir"
    fi
done
success "All required directories are present."

# Required source files
REQUIRED_SRC=(
    "app.py"
    "src/__init__.py"
    "src/config.py"
    "src/inference.py"
    "src/processor.py"
    "src/dashboard.py"
    "src/report_generator.py"
    "src/database.py"
    "src/dicom_reader.py"
    "src/theme.py"
    "src/api.py"
)

for f in "${REQUIRED_SRC[@]}"; do
    if [ -f "$f" ]; then
        success "Found: $f"
    else
        error "Missing source file: $f"
        ISSUES=$((ISSUES + 1))
    fi
done

# Check model files (warn but don't fail — user may need to download them)
MODEL_FILES=(
    "models/convnext_full_best.keras"
    "models/inception_full_best.keras"
    "models/densenet_full_best.keras"
)
MODELS_FOUND=0
for m in "${MODEL_FILES[@]}"; do
    if [ -f "$m" ]; then
        SIZE=$(du -sh "$m" | cut -f1)
        success "Model found: $m  ($SIZE)"
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        warn "Model file not found: $m"
    fi
done

if [ "$MODELS_FOUND" -eq 0 ]; then
    echo ""
    warn "No trained model files found in models/."
    warn "The dashboard will run in DEMO mode (synthetic predictions) until models are added."
    echo ""
    info "  To get models, either:"
    echo "    1. Extract the included archive:  unzip models.zip -d models/"
    echo "    2. Re-train using the Kaggle notebook in docs/04_models_and_training.md"
    echo ""
fi

# ── Step 6: Database and reports directories ───────────────────────────────────
header "Step 6 / 7 — Database & Reports Initialization"

mkdir -p data reports

# Trigger SQLite DB initialization
info "Initializing patient history database..."
"$PY" -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.disable(logging.CRITICAL)
try:
    from src.database import init_db
    init_db()
    print('Patient history database: ready')
except Exception as e:
    print(f'Warning: DB init returned: {e}')
"
success "Database initialized: data/patient_history.db"

# ── Step 7: Smoke test imports ─────────────────────────────────────────────────
header "Step 7 / 7 — Smoke Test (Import Verification)"

info "Running import checks..."
"$PY" -c "
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

failures = []

checks = [
    ('numpy',         'import numpy; print(f\"  numpy {numpy.__version__}\")'),
    ('opencv',        'import cv2; print(f\"  opencv {cv2.__version__}\")'),
    ('tensorflow',    'import tensorflow as tf; print(f\"  tensorflow {tf.__version__}\")'),
    ('gradio',        'import gradio as gr; print(f\"  gradio {gr.__version__}\")'),
    ('reportlab',     'from reportlab.lib import colors; print(\"  reportlab OK\")'),
    ('scikit-learn',  'import sklearn; print(f\"  scikit-learn {sklearn.__version__}\")'),
    ('pydicom',       'import pydicom; print(f\"  pydicom {pydicom.__version__}\")'),
    ('fastapi',       'import fastapi; print(f\"  fastapi {fastapi.__version__}\")'),
    ('src.config',    'from src.config import CLASSES; print(f\"  src.config OK — {len(CLASSES)} classes\")'),
    ('src.database',  'from src.database import get_scan_count; n=get_scan_count(); print(f\"  src.database OK — {n} record(s) in DB\")'),
    ('src.dicom',     'from src.dicom_reader import read_dicom; print(\"  src.dicom_reader OK\")'),
]

for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        failures.append(f'{name}: {e}')

if failures:
    print('FAILED imports:', failures)
    sys.exit(1)
else:
    print()
    print('All imports passed.')
" && success "All import checks passed." || {
    error "One or more import checks failed. Check the output above."
    ISSUES=$((ISSUES + 1))
}

# ── Final Summary ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

if [ "$ISSUES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  Setup completed successfully. Ready to launch!${RESET}"
else
    echo -e "${YELLOW}${BOLD}  Setup completed with ${ISSUES} warning(s). Review the output above.${RESET}"
fi

echo ""
echo -e "${BOLD}  Run the Gradio dashboard:${RESET}"
echo -e "    ${CYAN}source venv/bin/activate${RESET}"
echo -e "    ${CYAN}python3 app.py${RESET}"
echo ""
echo -e "${BOLD}  Then open your browser at:${RESET}  ${CYAN}http://127.0.0.1:7860${RESET}"
echo ""
echo -e "${BOLD}  Or run the FastAPI REST backend:${RESET}"
echo -e "    ${CYAN}source venv/bin/activate${RESET}"
echo -e "    ${CYAN}uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload${RESET}"
echo ""

if [ "$MODE" = "dev" ]; then
    echo -e "${BOLD}  Run tests:${RESET}"
    echo -e "    ${CYAN}source venv/bin/activate && pytest tests/ -v${RESET}"
    echo ""
fi

echo -e "${BOLD}  Useful flags for setup.sh:${RESET}"
echo -e "    ${CYAN}./setup.sh --dev${RESET}     Install dev/test tools (pytest, coverage)"
echo -e "    ${CYAN}./setup.sh --docker${RESET}  Build and run with Docker"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

exit 0
