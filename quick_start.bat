@echo off
echo ============================================
echo  🏔️ Simulation de Montagne Ultra-Réaliste
echo ============================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou n'est pas dans le PATH
    echo Veuillez installer Python 3.8 ou supérieur depuis python.org
    pause
    exit /b 1
)

echo ✓ Python détecté
python --version
echo.

REM Créer l'environnement virtuel s'il n'existe pas
if not exist "venv\" (
    echo 📦 Création de l'environnement virtuel...
    python -m venv venv
    echo ✓ Environnement virtuel créé
) else (
    echo ✓ Environnement virtuel déjà existant
)

REM Activer l'environnement virtuel
echo 🔧 Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Installer les dépendances
echo 📥 Installation des dépendances...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

echo.
echo ✓ Installation terminée!
echo.
echo 🚀 Lancement de l'application...
echo L'interface sera accessible à http://localhost:7860
echo.
echo Appuyez sur Ctrl+C pour arrêter l'application
echo.

REM Lancer l'application
python mountain_app.py

pause
