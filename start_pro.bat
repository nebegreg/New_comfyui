@echo off
cls
echo ================================================
echo   🏔️ Mountain Studio Pro
echo   Interface Professionnelle PySide6
echo ================================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non trouvé
    echo Installez Python 3.8+ depuis python.org
    pause
    exit /b 1
)

echo ✓ Python détecté
python --version

REM Vérifier environnement virtuel
if not exist "venv\" (
    echo 📦 Création environnement virtuel...
    python -m venv venv
    echo ✓ Environnement créé
)

REM Activer venv
echo 🔧 Activation environnement...
call venv\Scripts\activate.bat

REM Installer dépendances si nécessaire
if not exist "venv\.installed" (
    echo 📥 Installation dépendances première fois...
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    type nul > venv\.installed
    echo ✓ Dépendances installées
) else (
    echo ✓ Dépendances déjà installées
)

echo.
echo 🚀 Lancement Mountain Studio Pro...
echo.
echo =========================================
echo   Interface PySide6 Professionnelle
echo   - Vue 3D temps réel
echo   - Génération heightmap/normal/depth
echo   - Export professionnel EXR/OBJ
echo   - Vidéo cohérente même montagne!
echo =========================================
echo.
echo 💡 Conseil: Utilisez l'onglet 🗻 Terrain pour commencer
echo.

REM Lancer l'application
python mountain_pro_ui.py

echo.
echo 👋 Mountain Studio Pro fermé
pause
