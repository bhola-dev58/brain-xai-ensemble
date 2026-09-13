@echo off
:: =============================================================================
::  Brain XAI Ensemble — Windows Setup Script (Command Prompt)
::  GitHub: https://github.com/bhola-dev58/brain-xai-ensemble
::
::  Usage:
::    setup.bat          Full setup (venv + deps + DB + smoke test)
::    setup.bat --dev    Also install pytest, coverage, httpx
::    setup.bat --help   Show usage
::
::  Requires:
::    - Windows 10 / 11
::    - Python 3.10, 3.11, or 3.12  (https://www.python.org/downloads/)
::    - Internet connection (first run only)
::
::  After setup:
::    venv\Scripts\activate
::    python app.py        ->  http://127.0.0.1:7860
:: =============================================================================

setlocal EnableDelayedExpansion

:: ── Argument handling ─────────────────────────────────────────────────────────
set MODE=standard
for %%A in (%*) do (
    if /I "%%A"=="--dev"    set MODE=dev
    if /I "%%A"=="--docker" set MODE=docker
    if /I "%%A"=="--help"   goto :show_help
    if /I "%%A"=="-h"       goto :show_help
)
goto :main

:show_help
echo.
echo   Brain XAI Ensemble -- Windows Setup Script
echo.
echo   Usage:
echo     setup.bat           Full setup (venv + deps + DB + smoke test)
echo     setup.bat --dev     Also install pytest, coverage, httpx
echo     setup.bat --docker  Build and run with Docker
echo     setup.bat --help    Show this help
echo.
echo   Linux / macOS users: run  ./setup.sh  instead.
echo.
exit /b 0

:main

:: ── Banner ────────────────────────────────────────────────────────────────────
echo.
echo   ============================================================
echo     Brain XAI Ensemble -- Automated Setup for Windows
echo     Explainable AI Tri-Ensemble Brain Tumor Diagnostics
echo   ============================================================
echo.

:: ── Ensure we are in the project root ────────────────────────────────────────
if not exist "app.py" (
    echo [ERROR] Run setup.bat from the project root directory where app.py lives.
    exit /b 1
)
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found. Are you in the right folder?
    exit /b 1
)

:: =============================================================================
:: DOCKER MODE
:: =============================================================================
if /I "%MODE%"=="docker" (
    echo [INFO]  Docker mode selected.
    where docker >nul 2>&1 || (
        echo [ERROR] Docker not found. Install from https://docs.docker.com/get-docker/
        exit /b 1
    )
    echo [INFO]  Building image: brain-xai-ensemble ...
    docker build -t brain-xai-ensemble .
    echo.
    echo [OK]    Docker image built.
    echo.
    echo   Run:  docker run --rm -p 7860:7860 brain-xai-ensemble
    echo   Open: http://localhost:7860
    echo.
    exit /b 0
)

:: =============================================================================
:: STEP 1 — Python Check
:: =============================================================================
echo.
echo --- Step 1 / 7 -- Python Environment ---
echo.

set PYTHON_BIN=
for %%P in (python3.12 python3.11 python3.10 python3 python py) do (
    if not defined PYTHON_BIN (
        where %%P >nul 2>&1 && (
            for /f "tokens=*" %%V in ('%%P -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')" 2^>nul') do (
                set VER=%%V
                for /f "tokens=1,2 delims=." %%M in ("%%V") do (
                    if %%M EQU 3 (
                        if %%N GEQ 10 (
                            set PYTHON_BIN=%%P
                        )
                    )
                )
            )
        )
    )
)

if not defined PYTHON_BIN (
    echo [ERROR] Python 3.10 or higher is required but was not found.
    echo.
    echo   Download Python 3.12 from: https://www.python.org/downloads/
    echo   IMPORTANT: During install, tick "Add Python to PATH"
    echo.
    exit /b 1
)

echo [OK]    Python %VER% found: %PYTHON_BIN%

:: =============================================================================
:: STEP 2 — System Dependencies (Windows: none needed, pip wheels are bundled)
:: =============================================================================
echo.
echo --- Step 2 / 7 -- System Dependencies ---
echo.
echo [OK]    Windows detected. OpenCV and TensorFlow pip wheels are self-contained.
echo [OK]    No additional system libraries required.

:: =============================================================================
:: STEP 3 — Virtual Environment
:: =============================================================================
echo.
echo --- Step 3 / 7 -- Virtual Environment ---
echo.

if exist "venv\Scripts\python.exe" (
    echo [INFO]  Existing venv found at .\venv -- skipping creation.
) else (
    echo [INFO]  Creating virtual environment at .\venv ...
    %PYTHON_BIN% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK]    Virtual environment created.
)

set PY=venv\Scripts\python.exe
set PIP=venv\Scripts\pip.exe

echo [INFO]  Upgrading pip...
"%PY%" -m pip install --upgrade pip --quiet
echo [OK]    pip upgraded.

:: =============================================================================
:: STEP 4 — Python Dependencies
:: =============================================================================
echo.
echo --- Step 4 / 7 -- Python Dependencies ---
echo.

echo [INFO]  Installing from requirements.txt ...
"%PIP%" install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    exit /b 1
)
echo [OK]    Core dependencies installed.

if /I "%MODE%"=="dev" (
    echo [INFO]  [--dev] Installing pytest, coverage, httpx ...
    "%PIP%" install pytest pytest-cov coverage httpx
    echo [OK]    Dev tools installed.
)

:: =============================================================================
:: STEP 5 — Project Structure
:: =============================================================================
echo.
echo --- Step 5 / 7 -- Project Structure ---
echo.

set ISSUES=0

:: Create missing directories
for %%D in (models datasets assets src scripts test_images data reports) do (
    if not exist "%%D\" (
        mkdir "%%D"
        echo [INFO]  Created directory: %%D\
    )
)
echo [OK]    All runtime directories present.

:: Check required source files
set FILES=app.py src\__init__.py src\config.py src\inference.py src\processor.py src\dashboard.py src\report_generator.py src\database.py src\dicom_reader.py src\theme.py src\api.py

for %%F in (%FILES%) do (
    if exist "%%F" (
        echo [OK]    Found: %%F
    ) else (
        echo [ERROR] Missing: %%F
        set /a ISSUES+=1
    )
)

:: Check model files
set MODELS_FOUND=0
for %%M in (models\convnext_full_best.keras models\inception_full_best.keras models\densenet_full_best.keras) do (
    if exist "%%M" (
        echo [OK]    Model: %%M
        set /a MODELS_FOUND+=1
    ) else (
        echo [WARN]  Model not found: %%M
    )
)

if %MODELS_FOUND%==0 (
    echo.
    echo [WARN]  No trained model .keras files found in models\
    echo [WARN]  Dashboard will run in DEMO mode until models are added.
    echo.
    echo   Options:
    echo     1. Extract bundled archive:  tar -xf models.zip  (or use 7-Zip)
    echo     2. Re-train via Kaggle:      see docs\04_models_and_training.md
    echo.
)

:: =============================================================================
:: STEP 6 — Database Initialisation
:: =============================================================================
echo.
echo --- Step 6 / 7 -- Database Init ---
echo.

if not exist "data\" mkdir data
if not exist "reports\" mkdir reports

echo [INFO]  Initialising patient history database ...
"%PY%" -c "import os,logging; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; logging.disable(logging.CRITICAL); from src.database import init_db; init_db(); print('[OK]    Database ready: data/patient_history.db')"
if errorlevel 1 (
    echo [WARN]  Database init returned a warning. Check output above.
)

:: =============================================================================
:: STEP 7 — Smoke Test
:: =============================================================================
echo.
echo --- Step 7 / 7 -- Smoke Test ---
echo.
echo [INFO]  Verifying module imports ...

"%PY%" -c ^
"import os,sys;" ^
"os.environ['TF_CPP_MIN_LOG_LEVEL']='3';" ^
"os.environ['CUDA_VISIBLE_DEVICES']='-1';" ^
"failures=[];" ^
"checks=[('numpy','import numpy;print(f\"  numpy        {numpy.__version__}\")'),('opencv','import cv2;print(f\"  opencv       {cv2.__version__}\")'),('tensorflow','import tensorflow as tf;print(f\"  tensorflow   {tf.__version__}\")'),('gradio','import gradio as gr;print(f\"  gradio       {gr.__version__}\")'),('pydicom','import pydicom;print(f\"  pydicom      {pydicom.__version__}\")'),('fastapi','import fastapi;print(f\"  fastapi      {fastapi.__version__}\")'),('src.config','from src.config import CLASSES;print(f\"  src.config   OK ({len(CLASSES)} classes)\")'),('src.database','from src.database import get_scan_count;n=get_scan_count();print(f\"  src.database OK ({n} records)\")'),('src.dicom','from src.dicom_reader import read_dicom;print(\"  src.dicom    OK\")')];" ^
"[failures.append(f'{n}: {e}') or None for n,c in checks for e in [None] if not (lambda:exec(c) or True)()];" ^
"sys.exit(1) if failures else print('\n  All imports verified.')"

if errorlevel 1 (
    echo [ERROR] One or more import checks failed. See output above.
    set /a ISSUES+=1
) else (
    echo [OK]    All smoke tests passed.
)

:: =============================================================================
:: Final Summary
:: =============================================================================
echo.
echo ====================================================================
if %ISSUES%==0 (
    echo   SUCCESS: Setup completed. Ready to launch!
) else (
    echo   WARNING: Setup completed with %ISSUES% issue(s). Review output.
)
echo ====================================================================
echo.
echo   Launch the Gradio dashboard:
echo     venv\Scripts\activate
echo     python app.py
echo     Open:  http://127.0.0.1:7860
echo.
echo   Launch the FastAPI REST backend:
echo     venv\Scripts\activate
echo     uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
echo.
if /I "%MODE%"=="dev" (
    echo   Run tests:
    echo     venv\Scripts\activate ^&^& pytest tests\ -v --cov=src
    echo.
)
echo   Other flags:
echo     setup.bat --dev      Install pytest and coverage tools
echo     setup.bat --docker   Build and run in Docker
echo.

endlocal
exit /b 0
