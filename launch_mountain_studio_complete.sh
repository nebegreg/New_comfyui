#!/bin/bash
#
# Mountain Studio COMPLETE - Launcher Script
# Détecte l'environnement et lance l'application
#

echo "🏔️  Mountain Studio COMPLETE - Photorealistic Edition"
echo "======================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé!"
    echo "   Installez Python 3.8+ depuis https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✅ Python détecté: $PYTHON_VERSION"

# Check dependencies
echo ""
echo "Vérification des dépendances..."

MISSING_DEPS=0

if ! python3 -c "import PySide6" 2>/dev/null; then
    echo "❌ PySide6 manquant"
    MISSING_DEPS=1
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "❌ NumPy manquant"
    MISSING_DEPS=1
fi

if ! python3 -c "import scipy" 2>/dev/null; then
    echo "❌ SciPy manquant"
    MISSING_DEPS=1
fi

if ! python3 -c "import pyqtgraph" 2>/dev/null; then
    echo "⚠️  PyQtGraph manquant (viewer 3D limité)"
fi

if ! python3 -c "import PIL" 2>/dev/null; then
    echo "⚠️  Pillow manquant (exports limités)"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "Installation des dépendances manquantes..."
    pip3 install PySide6 numpy scipy pyqtgraph pillow opencv-python
fi

echo ""
echo "✅ Toutes les dépendances critiques sont installées"
echo ""

# Check ComfyUI (optionnel)
echo "Vérification ComfyUI (optionnel)..."
if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
    echo "✅ ComfyUI détecté et actif (AI textures disponibles)"
else
    echo "⚠️  ComfyUI non détecté (fallback procédural activé)"
    echo "   Pour activer l'AI: voir COMFYUI_GUIDE.md"
fi

echo ""
echo "======================================================"
echo "🚀 Lancement de Mountain Studio COMPLETE..."
echo "======================================================"
echo ""

# Launch application
python3 mountain_studio_complete.py

# Exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Application fermée normalement"
else
    echo ""
    echo "❌ Application terminée avec erreur"
    echo "   Vérifiez les logs ci-dessus"
    exit 1
fi
