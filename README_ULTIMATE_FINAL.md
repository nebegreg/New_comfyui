# MOUNTAIN STUDIO ULTIMATE FINAL 🏔️

**L'Application Ultime de Génération de Terrains Photorréalistes**

Version Finale - Tous les Features Intégrés

---

## ✨ Features Principales

### 🎯 One-Click Generation
- **Bouton "GENERATE ALL"** - Génère terrain + végétation + PBR + HDRI en un seul clic
- Workflow automatique avec tracking de progression
- Zéro configuration manuelle requise

### ⭐ Presets Professionnels
6 presets de montagnes iconiques prêts à l'emploi:
- **Evian Alps** - Style publicité (montagnes immaculées)
- **Three Peaks** - 3 sommets majestueux
- **Powder Ski Slope** - Piste de ski poudreuse
- **Matterhorn Peak** - Pic emblématique en pyramide
- **Mont Blanc Massif** - Plus haut sommet des Alpes
- **Dolomites Towers** - Formations rocheuses spectaculaires

### 🏔️ Génération Terrain Ultra-Réaliste
- Algorithmes avancés (Perlin, Ridge, Domain Warping)
- Érosion hydraulique et thermique
- Résolutions: 256x256 jusqu'à 2048x2048
- Contrôle total: scale, octaves, seed

### 🌲 Système de Végétation
- Placement basé sur les biomes
- Poisson Disc Sampling pour distribution naturelle
- Classification automatique (forêt, prairie, roche, neige)
- Clustering pour groupes d'arbres

### 🎨 Textures PBR Professionnelles
**Deux modes de génération:**
- **AI (ComfyUI)** - Ultra-réaliste via Stable Diffusion (automatique)
- **Procédural** - Rapide, bonne qualité

**Maps générées:**
- Diffuse (Albedo)
- Normal Map
- Roughness
- Ambient Occlusion
- Height (Displacement)
- Metallic

### 🗺️ Preview PBR Complet (NOUVEAU!)
- Visualisation de toutes les PBR maps
- Grille de thumbnails interactive
- Zoom et comparaison côte à côte

### 🎮 Rendu 3D Photorréaliste
- PBR lighting avec atmosphère
- Distance fog et scattering atmosphérique
- Ombres et spéculaire réalistes
- Visualisation OpenGL temps-réel

### 🌅 HDRI Panoramique
**7 times of day:**
- Sunrise, Morning, Midday, Afternoon, Sunset, Twilight, Night

**Features:**
- Scattering Rayleigh physiquement réaliste
- Color temperature simulation
- Nuages procéduraux
- Silhouettes de montagnes lointaines
- Export HDR (.hdr), EXR (.exr), PNG (preview)

### 🎬 Export Avancé (NOUVEAU!)
**Export pour Autodesk Flame:**
- Package complet optimisé VFX
- High-res OBJ mesh
- 16-bit EXR textures (linear color space)
- Camera data
- HDRI environment
- Python setup script

**Autres formats:**
- **OBJ** (Wavefront) - Universal
- **FBX** (Autodesk) - Maya, 3ds Max
- **ABC** (Alembic) - VFX pipelines

---

## 🚀 Installation

### Prérequis
```bash
Python 3.8+
PySide6
numpy
scipy
Pillow
pyqtgraph
PyOpenGL
```

### Installation Rapide
```bash
# Installer les dépendances
pip install PySide6 numpy scipy Pillow pyqtgraph PyOpenGL PyOpenGL_accelerate

# Lancer l'application
python mountain_studio_ultimate_final.py
```

### Installation Complète (avec AI)
Pour utiliser la génération AI via ComfyUI:

1. **Installer ComfyUI** (voir COMFYUI_GUIDE.md)
2. **Télécharger les modèles** (SDXL ou SD 1.5)
3. **Lancer ComfyUI server**
4. L'application détectera automatiquement ComfyUI

---

## 📖 Guide d'Utilisation

### Workflow Recommandé

#### Option 1: Utiliser un Preset (RAPIDE ⚡)
1. **Onglet "Presets"** → Sélectionner un preset (ex: "Evian Alps")
2. Cliquer **"Apply Preset"**
3. Cliquer **"GENERATE ALL"** (bouton vert en haut)
4. Attendre (2-5 minutes selon config)
5. Visualiser dans **"3D Rendering"** et **"PBR Preview"**
6. Exporter dans **"Advanced Export"** → **"Export for Autodesk Flame"**

**Temps total: 2-5 minutes** ⚡

#### Option 2: Configuration Manuelle (CONTRÔLE TOTAL)
1. **Onglet "Terrain"** → Régler paramètres → **"Generate Terrain"**
2. **Onglet "Vegetation"** → Régler spacing → **"Generate Vegetation"**
3. **Onglet "PBR Textures"** → Choisir material → **"Generate PBR"**
4. **Onglet "HDRI Sky"** → Choisir time of day → **"Generate HDRI"**
5. **Onglet "3D Rendering"** → **"Render 3D View"**
6. **Onglet "PBR Preview"** → Visualiser toutes les maps
7. **Onglet "Advanced Export"** → Exporter

### Les 9 Onglets

1. **⭐ Presets** - Configurations professionnelles prêtes à l'emploi
2. **🏔️ Terrain** - Génération terrain avec érosion
3. **🌲 Vegetation** - Placement arbres et végétation
4. **🎨 PBR Textures** - Génération textures (AI ou procedural)
5. **🗺️ PBR Preview** - Visualisation complète des maps (NOUVEAU!)
6. **🎮 3D Rendering** - Vue 3D photorréaliste temps-réel
7. **🌅 HDRI Sky** - Génération ciel panoramique HDR
8. **💾 Export** - Exports basiques (heightmap, textures, HDRI)
9. **🎬 Advanced Export** - Exports professionnels VFX/3D (NOUVEAU!)

---

## 🎯 Quick Start Résumé

```bash
# 1. Installer
pip install PySide6 numpy scipy Pillow pyqtgraph PyOpenGL

# 2. Lancer
python mountain_studio_ultimate_final.py

# 3. Dans l'app:
#    - Onglet "Presets" → "Evian Alps"
#    - Click "Apply Preset"
#    - Click "GENERATE ALL"
#    - Attendre 2-5 minutes
#    - Visualiser et exporter

# 🎉 DONE!
```

---

## 📁 Structure des Outputs

```
outputs_ultimate/
├── terrain_preview.png          # Heightmap preview
├── pbr_textures/                # PBR maps
│   ├── diffuse.png
│   ├── normal.png
│   ├── roughness.png
│   ├── ao.png
│   ├── height.png
│   └── metallic.png
├── hdri/                        # HDRI exports
│   ├── mountain_hdri.exr
│   └── mountain_hdri_preview.png
└── flame_export/                # Autodesk Flame package
    ├── terrain_flame.obj
    ├── terrain_flame.mtl
    ├── *.png (textures)
    └── README_FLAME.txt
```

---

## 🔧 Troubleshooting

### L'application ne se lance pas
```bash
pip install --upgrade PySide6 numpy scipy Pillow pyqtgraph PyOpenGL
python --version  # Doit être 3.8+
```

### ComfyUI ne se connecte pas
1. Vérifier ComfyUI: `http://127.0.0.1:8188`
2. Décocher "Use AI" et utiliser Procedural
3. Voir COMFYUI_GUIDE.md

### Pas de 3D view
```bash
pip install PyOpenGL PyOpenGL_accelerate
```

### Génération lente
- Réduire résolution (512x512)
- Réduire érosion iterations (20)
- Utiliser PBR Procedural

---

## 📝 Changelog

### Version ULTIMATE FINAL (2025)
- ✅ **NEW:** Onglet Presets (6 configurations pro)
- ✅ **NEW:** Onglet PBR Preview (visualisation complète)
- ✅ **NEW:** Onglet Advanced Export (Autodesk Flame)
- ✅ **NEW:** Bouton "Generate All" master
- ✅ **NEW:** ComfyUI auto-workflow (zéro config)
- ✅ **IMPROVED:** HDRI V2 avec Rayleigh scattering
- ✅ **IMPROVED:** Export OBJ avec normals et UVs
- ✅ **FIXED:** BiomeClassifier initialization
- ✅ **FIXED:** Texture resolution mismatch

---

## 👥 Credits

**Mountain Studio Pro Team**

Built with: Python, PySide6, NumPy, SciPy, PyQtGraph, OpenGL, ComfyUI

---

**🏔️ MOUNTAIN STUDIO ULTIMATE FINAL - L'application ultime pour terrains photorréalistes!**
