#!/bin/bash

echo "🏔️  Simulation de Montagne Ultra-Réaliste"
echo "========================================"
echo ""

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 n'est pas installé"
    echo "Veuillez installer Python 3.8 ou supérieur"
    exit 1
fi

echo "✓ Python détecté: $(python3 --version)"
echo ""

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
    echo "✓ Environnement virtuel créé"
else
    echo "✓ Environnement virtuel déjà existant"
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "📥 Installation des dépendances..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo ""
echo "✓ Installation terminée!"
echo ""
echo "🚀 Lancement de l'application..."
echo "L'interface sera accessible à http://localhost:7860"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter l'application"
echo ""

# Lancer l'application
python mountain_app.py
