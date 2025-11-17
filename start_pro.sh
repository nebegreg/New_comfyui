#!/bin/bash

echo "================================================"
echo "  🏔️ Mountain Studio Pro"
echo "  Interface Professionnelle PySide6"
echo "================================================"
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 non trouvé"
    echo "Installez Python 3.8+ depuis python.org"
    exit 1
fi

echo "✓ Python: $(python3 --version)"

# Vérifier environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Création environnement virtuel..."
    python3 -m venv venv
    echo "✓ Environnement créé"
fi

# Activer venv
echo "🔧 Activation environnement..."
source venv/bin/activate

# Installer dépendances si nécessaire
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installation dépendances (première fois)..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    touch venv/.installed
    echo "✓ Dépendances installées"
else
    echo "✓ Dépendances déjà installées"
fi

echo ""
echo "🚀 Lancement Mountain Studio Pro..."
echo ""
echo "========================================="
echo "  Interface PySide6 Professionnelle"
echo "  - Vue 3D temps réel"
echo "  - Génération heightmap/normal/depth"
echo "  - Export professionnel EXR/OBJ"
echo "  - Vidéo cohérente (même montagne!)"
echo "========================================="
echo ""
echo "💡 Conseil: Utilisez l'onglet 🗻 Terrain pour commencer"
echo ""

# Lancer l'application
python mountain_pro_ui.py

echo ""
echo "👋 Mountain Studio Pro fermé"
