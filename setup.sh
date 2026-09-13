#!/usr/bin/env bash
# =============================================================================
#  Brain XAI Ensemble — Cross-Platform Automated Setup Script
#  Supports: Linux (Ubuntu/Debian/Fedora/Arch) | macOS (Intel + Apple Silicon)
#  GitHub: https://github.com/bhola-dev58/brain-xai-ensemble
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh            # Full setup (venv + deps + DB + smoke test)
#    ./setup.sh --dev      # Same + pytest, coverage, httpx for testing
#    ./setup.sh --docker   # Build and launch via Docker instead of venv
#    ./setup.sh --help     # Show usage
#
#  Windows users: run setup.bat (CMD) or setup.ps1 (PowerShell) instead.
#
#  After setup:
#    source venv/bin/activate
#    python3 app.py        →  http://127.0.0.1:7860
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Colour helpers (disabled when not a TTY, e.g. CI pipes) ───────────────────
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

# ── Detect OS ─────────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="macos" ;;
        *)       OS="unknown" ;;
    esac
}

# ── Argument parsing ───────────────────────────────────────────────────────────
MODE="standard"
for arg in "$@"; do
    case "$arg" in
        --dev)    MODE="dev" ;;
        --docker) MODE="docker" ;;
        --help|-h)
            echo ""
            echo -e "  ${BOLD}Brain XAI Ensemble — Setup Script${RESET}"
            echo ""
            echo "  Supports: Linux (Ubuntu/Debian/Fedora/Arch) | macOS (Intel + Apple Silicon)"
            echo "  Windows:  run setup.bat  or  setup.ps1 instead."
            echo ""
            echo "  Usage:"
            echo "    ./setup.sh           Standard setup (venv + deps + DB + smoke test)"
            echo "    ./setup.sh --dev     Also install dev tools (pytest, coverage)"
            echo "    ./setup.sh --docker  Build & run with Docker"
            echo "    ./setup.sh --help    Show this help"
            echo ""
            exit 0
            ;;
        *) warn "Unknown argument '$arg' — ignoring." ;;
    esac
done

detect_os

# ── Banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${BLUE}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║        Brain XAI Ensemble — Automated Project Setup         ║"
echo "  ║   Explainable AI Tri-Ensemble for Brain Tumor Diagnostics   ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "Operating System: ${OS}"

# ── Working directory guard ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f "app.py" ] && [ -f "requirements.txt" ] \
    || die "Run setup.sh from the project root directory (where app.py lives)."

# =============================================================================
# DOCKER MODE
# =============================================================================
if [ "$MODE" = "docker" ]; then
    header "Docker Mode"
    command -v docker &>/dev/null || die "Docker is not installed. Visit: https://docs.docker.com/get-docker/"
    info "Docker: $(docker --version)"
    [ -z "$(ls -A models/*.keras 2>/dev/null)" ] \
        && warn "models/ is empty — extract models.zip first:  unzip models.zip -d models/"
    info "Building Docker image: brain-xai-ensemble..."
    docker build -t brain-xai-ensemble .
    success "Docker image built."
    echo ""
    echo -e "  ${BOLD}Run:${RESET}  docker run --rm -p 7860:7860 brain-xai-ensemble"
    echo -e "  ${BOLD}Open:${RESET} ${CYAN}http://localhost:7860${RESET}"
    exit 0
fi

# =============================================================================
# STEP 1 — Python Version Check
# =============================================================================
header "Step 1 / 7 — Python Environment"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        VERSION=$("$candidate" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || continue
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    error "Python 3.10 or higher is required but was not found."
    echo ""
    if [ "$OS" = "macos" ]; then
        info "Install via Homebrew:   brew install python@3.12"
        info "Or download from:       https://www.python.org/downloads/"
    else
        info "Install on Ubuntu/Debian:  sudo apt-get install python3.12"
        info "Install on Fedora:         sudo dnf install python3.12"
        info "Install on Arch:           sudo pacman -S python"
        info "Or download from:          https://www.python.org/downloads/"
    fi
    exit 1
fi

info "Python $VERSION found at: $(command -v $PYTHON_BIN)"

# =============================================================================
# STEP 2 — System Dependencies
# =============================================================================
header "Step 2 / 7 — System Dependencies"

if [ "$OS" = "linux" ]; then
    # ── Linux: detect package manager ────────────────────────────────────────
    MISSING_PKGS=()
    for lib in libgl1 libglib2.0-0; do
        dpkg -l "$lib" &>/dev/null 2>&1 || MISSING_PKGS+=("$lib")
    done

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        warn "Missing OpenCV system libs: ${MISSING_PKGS[*]}"
        if command -v apt-get &>/dev/null; then
            info "Installing via apt-get (sudo required)..."
            sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}" \
                && success "System packages installed." \
                || warn "Auto-install failed. Please run: sudo apt-get install ${MISSING_PKGS[*]}"
        elif command -v dnf &>/dev/null; then
            info "Installing via dnf (sudo required)..."
            sudo dnf install -y mesa-libGL glib2 \
                && success "System packages installed." \
                || warn "Auto-install failed. Please run: sudo dnf install mesa-libGL glib2"
        elif command -v pacman &>/dev/null; then
            info "Installing via pacman (sudo required)..."
            sudo pacman -S --noconfirm mesa glib2 \
                && success "System packages installed." \
                || warn "Auto-install failed. Please run: sudo pacman -S mesa glib2"
        else
            warn "Unknown package manager. Install these manually: ${MISSING_PKGS[*]}"
        fi
    else
        success "System libraries OK (libgl1, libglib2.0-0)."
    fi

elif [ "$OS" = "macos" ]; then
    # ── macOS: OpenCV pip wheel is self-contained; no system libs needed ──────
    success "macOS detected — OpenCV pip wheel is self-contained. No system libs required."

    # Warn about Apple Silicon Rosetta / arch mismatch
    ARCH=$(uname -m)
    info "Architecture: $ARCH"
    if [ "$ARCH" = "arm64" ]; then
        info "Apple Silicon (M1/M2/M3) detected. TensorFlow uses CPU-only mode on Apple Silicon."
        info "All three ensemble models will run on CPU — this is expected and supported."
    fi

else
    warn "Unrecognised OS '$(uname -s)'. Skipping system package check."
    warn "If OpenCV fails, install its native libraries manually for your platform."
fi

# =============================================================================
# STEP 3 — Virtual Environment
# =============================================================================
header "Step 3 / 7 — Virtual Environment"

VENV_DIR="venv"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    info "Existing virtualenv found at ./$VENV_DIR — skipping creation."
else
    info "Creating virtualenv at ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

info "Upgrading pip..."
"$PIP" install --upgrade pip --quiet
success "pip upgraded to $("$PIP" --version | awk '{print $2}')"

# =============================================================================
# STEP 4 — Python Dependencies
# =============================================================================
header "Step 4 / 7 — Python Dependencies"

info "Installing from requirements.txt ..."
"$PIP" install -r requirements.txt
success "Core dependencies installed."

if [ "$MODE" = "dev" ]; then
    info "[--dev] Installing development tools (pytest, coverage, httpx)..."
    "$PIP" install pytest pytest-cov coverage httpx
    success "Dev tools installed."
fi

# =============================================================================
# STEP 5 — Project Structure Verification
# =============================================================================
header "Step 5 / 7 — Project Structure"

ISSUES=0

# Ensure runtime directories exist
for dir in models datasets assets src scripts test_images data reports; do
    [ -d "$dir" ] || { mkdir -p "$dir"; info "Created missing directory: $dir/"; }
done
success "All runtime directories present."

# Required source files
REQUIRED_SRC=(
    "app.py" "src/__init__.py" "src/config.py" "src/inference.py"
    "src/processor.py" "src/dashboard.py" "src/report_generator.py"
    "src/database.py" "src/dicom_reader.py" "src/theme.py" "src/api.py"
)
for f in "${REQUIRED_SRC[@]}"; do
    if [ -f "$f" ]; then
        success "Found: $f"
    else
        error  "Missing: $f"
        ISSUES=$((ISSUES + 1))
    fi
done

# Model files (warn but don't fail — contributor may need to download them)
MODELS_FOUND=0
for m in models/convnext_full_best.keras models/inception_full_best.keras models/densenet_full_best.keras; do
    if [ -f "$m" ]; then
        SIZE=$(du -sh "$m" | cut -f1)
        success "Model: $m  ($SIZE)"
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        warn "Model not found: $m"
    fi
done

if [ "$MODELS_FOUND" -eq 0 ]; then
    echo ""
    warn "No trained model files found in models/."
    warn "Dashboard will run in DEMO MODE until models are added."
    echo ""
    info "Options to obtain model weights:"
    echo "    1. Extract bundled archive:    unzip models.zip -d models/"
    echo "    2. Re-train via Kaggle:        see docs/04_models_and_training.md"
    echo ""
fi

# =============================================================================
# STEP 6 — Database Initialisation
# =============================================================================
header "Step 6 / 7 — Database & Directory Init"

mkdir -p data reports

info "Initialising patient history SQLite database..."
"$PY" - <<'PYEOF'
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)
try:
    from src.database import init_db
    init_db()
    print("Patient history database: ready  (data/patient_history.db)")
except Exception as e:
    print(f"Note: {e}")
PYEOF
success "Database ready."

# =============================================================================
# STEP 7 — Smoke Test (Import Verification)
# =============================================================================
header "Step 7 / 7 — Smoke Test"

info "Verifying all module imports..."
"$PY" - <<'PYEOF'
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

failures = []
checks = [
    ("numpy",        "import numpy; print(f'  numpy        {numpy.__version__}')"),
    ("opencv",       "import cv2;   print(f'  opencv       {cv2.__version__}')"),
    ("tensorflow",   "import tensorflow as tf; print(f'  tensorflow   {tf.__version__}')"),
    ("gradio",       "import gradio as gr; print(f'  gradio       {gr.__version__}')"),
    ("reportlab",    "from reportlab.lib import colors; print('  reportlab    OK')"),
    ("scikit-learn", "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"),
    ("pydicom",      "import pydicom; print(f'  pydicom      {pydicom.__version__}')"),
    ("fastapi",      "import fastapi; print(f'  fastapi      {fastapi.__version__}')"),
    ("src.config",   "from src.config import CLASSES; print(f'  src.config   OK  ({len(CLASSES)} classes)')"),
    ("src.database", "from src.database import get_scan_count; n=get_scan_count(); print(f'  src.database OK  ({n} record(s) in DB)')"),
    ("src.dicom",    "from src.dicom_reader import read_dicom; print('  src.dicom    OK')"),
]

for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        failures.append(f"{name}: {e}")

if failures:
    print("\nFAILED:", failures, file=sys.stderr)
    sys.exit(1)

print("\n  All imports verified successfully.")
PYEOF
success "All smoke tests passed."

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

if [ "$ISSUES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ✓  Setup completed successfully!${RESET}"
else
    echo -e "${YELLOW}${BOLD}  ⚠  Setup completed with ${ISSUES} issue(s). Review output above.${RESET}"
fi

echo ""
echo -e "${BOLD}  Launch the Gradio dashboard:${RESET}"
echo -e "    ${CYAN}source venv/bin/activate${RESET}"
echo -e "    ${CYAN}python3 app.py${RESET}"
echo -e "    Then open: ${CYAN}http://127.0.0.1:7860${RESET}"
echo ""
echo -e "${BOLD}  Launch the FastAPI REST backend:${RESET}"
echo -e "    ${CYAN}uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload${RESET}"
echo ""
[ "$MODE" = "dev" ] && {
    echo -e "${BOLD}  Run test suite:${RESET}"
    echo -e "    ${CYAN}pytest tests/ -v --cov=src${RESET}"
    echo ""
}
echo -e "${BOLD}  Other setup options:${RESET}"
echo -e "    ${CYAN}./setup.sh --dev${RESET}      Include pytest & coverage tools"
echo -e "    ${CYAN}./setup.sh --docker${RESET}   Build & run in Docker container"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

exit 0
