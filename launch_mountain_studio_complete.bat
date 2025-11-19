@echo off
REM
REM Mountain Studio COMPLETE - Launcher Script (Windows)
REM Détecte l'environnement et lance l'application
REM

echo.
echo ================================================================
echo    Mountain Studio COMPLETE - Photorealistic Edition
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python non trouve!
    echo        Installez Python 3.8+ depuis https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python detecte: %PYTHON_VERSION%
echo.

REM Check dependencies
echo Verification des dependances...
echo.

python -c "import PySide6" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] PySide6 manquant
    echo          Installation...
    pip install PySide6
)

python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] NumPy manquant
    echo          Installation...
    pip install numpy
)

python -c "import scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] SciPy manquant
    echo          Installation...
    pip install scipy
)

python -c "import pyqtgraph" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] PyQtGraph manquant (viewer 3D limite)
    echo        Installation...
    pip install pyqtgraph
)

python -c "import PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Pillow manquant (exports limites)
    echo        Installation...
    pip install pillow
)

python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] OpenCV manquant
    echo        Installation...
    pip install opencv-python
)

echo.
echo [OK] Toutes les dependances critiques sont installees
echo.

REM Check ComfyUI (optional)
echo Verification ComfyUI (optionnel)...
curl -s http://127.0.0.1:8188/system_stats >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] ComfyUI detecte et actif (AI textures disponibles^)
) else (
    echo [WARN] ComfyUI non detecte (fallback procedural active^)
    echo        Pour activer l'AI: voir COMFYUI_GUIDE.md
)

echo.
echo ================================================================
echo    Lancement de Mountain Studio COMPLETE...
echo ================================================================
echo.

REM Launch application
python mountain_studio_complete.py

if %errorlevel% equ 0 (
    echo.
    echo [OK] Application fermee normalement
) else (
    echo.
    echo [ERREUR] Application terminee avec erreur
    echo          Verifiez les logs ci-dessus
    pause
    exit /b 1
)

pause
