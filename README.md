# 🏔️ Simulation de Montagne Ultra-Réaliste

Application de génération d'images et de vidéos de montagnes photoréalistes utilisant Stable Diffusion et ComfyUI.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Fonctionnalités

- 🖼️ **Génération d'images ultra-réalistes** de paysages montagneux
- 🎬 **Création de vidéos** avec mouvements de caméra cinématiques
- 🎥 **Système de caméra complet** avec contrôle de l'angle, focale, hauteur et distance
- 🏔️ **Personnalisation totale** : type de montagne, végétation, ciel, météo, saison
- 🎨 **Interface graphique intuitive** avec Gradio
- 🔧 **Deux backends disponibles** : ComfyUI ou Stable Diffusion direct
- 🎞️ **Mouvements de caméra** : Orbit, Pan, Zoom, Flyover

## 📋 Prérequis

- Python 3.8 ou supérieur
- GPU NVIDIA avec CUDA (recommandé pour Stable Diffusion)
- 8 GB+ de VRAM recommandé
- (Optionnel) ComfyUI installé et en fonctionnement

## 🚀 Installation

1. **Cloner le dépôt**
```bash
git clone https://github.com/votre-repo/mountain-simulation.git
cd mountain-simulation
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration du backend**

### Option A : Stable Diffusion Direct (Recommandé pour commencer)
Aucune configuration supplémentaire nécessaire. Le modèle se téléchargera automatiquement au premier lancement.

### Option B : ComfyUI
1. Installez ComfyUI séparément : https://github.com/comfyanonymous/ComfyUI
2. Lancez le serveur ComfyUI
3. Notez l'adresse du serveur (par défaut: `127.0.0.1:8188`)

## 🎮 Utilisation

### Lancement de l'application

```bash
python mountain_app.py
```

L'interface sera accessible à l'adresse : `http://localhost:7860`

### Guide d'utilisation

#### 1. Configuration initiale
- Choisissez votre backend (ComfyUI ou Stable Diffusion Direct)
- Si ComfyUI : entrez l'adresse du serveur
- Cliquez sur "🚀 Initialiser"

#### 2. Génération d'une image unique

**Paramètres de Montagne :**
- **Type de montagne** : Alpine, Rolling, Volcanic, Massive, Rocky
- **Hauteur relative** : 0-100 (influence l'élévation des pics)

**Végétation :**
- **Type d'arbres** : Pine, Spruce, Mixed, Sparse, Dense
- **Densité** : 0-100 (0 = pas d'arbres, 100 = forêt dense)

**Ciel et Météo :**
- **Type de ciel** : Clear, Cloudy, Sunset, Sunrise, Stormy, etc.
- **Éclairage** : Golden hour, Midday, Dramatic, Soft, Backlit
- **Météo** : Clear, Fog, Snow, Rain
- **Saison** : Spring, Summer, Autumn, Winter

**Caméra :**
- **Angle horizontal** : -180° à 180° (rotation autour de la scène)
- **Angle vertical** : -90° à 90° (vue plongeante ou contre-plongée)
- **Focale** : 24mm-200mm (grand angle à téléobjectif)
- **Hauteur** : 0-100 (élévation de la caméra)
- **Distance** : 10-500 (distance à la scène)

**Génération :**
- **Dimensions** : Largeur et hauteur en pixels (recommandé : 1024x768)
- **Steps** : 20-100 (plus = meilleure qualité mais plus lent)
- **Seed** : Nombre aléatoire pour la reproductibilité
- **Niveau de détail** : 0-100 (influence les tags de qualité)

#### 3. Génération de vidéo

Utilisez l'onglet "🎬 Génération Vidéo" pour créer des animations :

**Types de mouvements :**
- **Orbit** : Rotation complète à 360° autour des montagnes
- **Pan** : Panoramique horizontal de gauche à droite
- **Zoom** : Zoom progressif sur la scène
- **Flyover** : Survol cinématique avec mouvement de hauteur
- **Static** : Aucun mouvement (pour tester les paramètres)

**Paramètres vidéo :**
- **Nombre de frames** : 3-30 (attention, chaque frame nécessite une génération)
- **FPS** : 12-60 (frames par seconde de la vidéo finale)
- **Transitions douces** : Interpole entre les frames pour un mouvement fluide

⚠️ **Note** : La génération de vidéo peut être longue. Pour 10 frames avec 30 steps chacune, comptez 5-10 minutes selon votre GPU.

## 📁 Structure du projet

```
mountain-simulation/
├── mountain_app.py           # Application principale avec interface Gradio
├── camera_system.py          # Système de caméra et gestion des mouvements
├── prompt_generator.py       # Génération de prompts optimisés
├── comfyui_integration.py    # Intégration ComfyUI et Stable Diffusion
├── video_generator.py        # Création de vidéos à partir d'images
├── requirements.txt          # Dépendances Python
├── README.md                # Cette documentation
└── outputs/                 # Dossier des images et vidéos générées
```

## 🎨 Exemples de prompts générés

L'application génère automatiquement des prompts optimisés. Exemple :

```
photorealistic, highly detailed, 8k uhd, professional photography,
jagged alpine peaks, snow-capped mountains, rocky cliffs, towering peaks,
extreme elevation, massive scale, dense pine forest, coniferous trees,
evergreen coverage, thick forest coverage, abundant vegetation,
golden hour, sunset lighting, warm orange glow, dramatic sky,
dramatic lighting, god rays, volumetric light, clear weather, high visibility,
autumn colors, fall foliage, orange and red leaves, elevated viewpoint,
overlooking mountain landscape, standard lens, natural perspective,
medium elevation, moderate depth of field, natural landscape,
realistic terrain, authentic mountain scene, high dynamic range,
rich colors, natural color grading, ultra detailed, hyper realistic
```

## 🔧 Paramètres avancés

### Optimisation des performances

- **Pour des générations rapides** : Réduisez les steps à 20-25
- **Pour la meilleure qualité** : 50-80 steps
- **Pour des vidéos** : 25-35 steps (compromis vitesse/qualité)

### Utilisation de la mémoire

- **GPU 8GB** : Résolution max recommandée 1024x768
- **GPU 12GB** : Résolution max recommandée 1536x1024
- **GPU 24GB+** : Jusqu'à 2048x2048

### Seeds utiles

Utilisez le même seed pour générer des variations cohérentes d'une même scène en changeant uniquement certains paramètres.

## 🐛 Dépannage

### Erreur : "CUDA out of memory"
- Réduisez la résolution de l'image
- Réduisez le nombre de steps
- Fermez les autres applications utilisant le GPU

### ComfyUI ne se connecte pas
- Vérifiez que ComfyUI est bien lancé
- Vérifiez l'adresse du serveur (défaut: 127.0.0.1:8188)
- Essayez le mode "Stable Diffusion Direct"

### Images de mauvaise qualité
- Augmentez le nombre de steps (50+)
- Augmentez le niveau de détail
- Essayez différents seeds
- Vérifiez que les paramètres de scène sont cohérents

### Vidéo saccadée
- Augmentez le nombre de frames
- Activez les "transitions douces"
- Augmentez le FPS (30-60)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Partager vos créations

## 📝 License

MIT License - Voir le fichier LICENSE pour plus de détails

## 🙏 Remerciements

- **Stable Diffusion** par Stability AI
- **ComfyUI** par comfyanonymous
- **Gradio** pour l'interface graphique
- La communauté open-source de l'IA générative

## 📞 Support

Pour toute question ou problème :
- Ouvrez une issue sur GitHub
- Consultez la documentation de Stable Diffusion
- Rejoignez la communauté ComfyUI

---

**Créé avec ❤️ pour les amoureux de la montagne et de l'IA générative**

Amusez-vous bien à créer des paysages montagneux époustouflants ! 🏔️✨
