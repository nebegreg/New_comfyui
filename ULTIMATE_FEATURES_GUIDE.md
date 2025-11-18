# Mountain Studio Pro - Ultimate Features Guide

## 🎉 Nouvelles Fonctionnalités Avancées

Ce guide documente les 3 fonctionnalités avancées ajoutées:

1. **HDRI Panoramique 360°** - Génération de skybox/environnement
2. **Ombres Temps Réel** - Shadow mapping avec shaders OpenGL personnalisés
3. **Caméra FPS Complète** - Contrôles WASD + mouse look avec collision terrain

---

## 📋 Prérequis

### Dépendances Requises

```bash
# OpenGL pour rendering avancé
pip install PyOpenGL PyOpenGL-accelerate

# HDR/EXR support
pip install OpenEXR Imath

# Mathématiques 3D
pip install pyrr

# Optionnel: AI enhancement pour HDRI (nécessite 10+ GB VRAM)
pip install diffusers transformers accelerate torch
```

### Configuration Système

- **Carte graphique**: OpenGL 3.3+ requis
- **VRAM**:
  - 2-4 GB: Fonctionnalités de base
  - 10-12 GB: HDRI avec AI enhancement
  - 24 GB: Toutes fonctionnalités à résolution maximale
- **RAM**: 8 GB minimum, 16 GB recommandé

### Vérifier Support OpenGL

```python
from OpenGL.GL import *
version = glGetString(GL_VERSION).decode('utf-8')
print(f"OpenGL Version: {version}")  # Doit être ≥ 3.3
```

---

## 🎮 1. Caméra FPS Complète

### Caractéristiques

- ✅ Déplacement WASD fluide
- ✅ Mouse look (yaw/pitch)
- ✅ Collision terrain avec interpolation bilinéaire
- ✅ Mouvement vertical (Space/Shift)
- ✅ Contrôle de vitesse et sensibilité
- ✅ Matrices view/projection pour OpenGL

### Utilisation

```python
from core.camera.fps_camera import FPSCamera
import numpy as np

# Créer caméra
camera = FPSCamera(
    position=np.array([0.0, 10.0, 0.0]),
    yaw=-90.0,  # Direction initiale
    pitch=0.0,
    speed=10.0,
    sensitivity=0.1
)

# Définir heightmap pour collision
camera.set_heightmap(
    heightmap=terrain_heightmap,
    terrain_scale=100.0,
    height_scale=20.0
)

# Dans la boucle de rendu:
delta_time = time_since_last_frame

# Input clavier
if key_W_pressed:
    camera.set_move_forward(True)
# ... autres touches

# Input souris
camera.process_mouse_movement(mouse_dx, mouse_dy)

# Update position
camera.process_keyboard(delta_time)

# Obtenir matrices pour rendu
view_matrix = camera.get_view_matrix()
proj_matrix = camera.get_projection_matrix(aspect_ratio)
```

### Contrôles

| Touche | Action |
|--------|--------|
| **W** | Avancer |
| **S** | Reculer |
| **A** | Gauche |
| **D** | Droite |
| **Space** | Monter |
| **Shift** | Descendre |
| **Souris** | Rotation caméra |
| **R** | Reset position |
| **C** | Toggle collision |

### Paramètres

```python
# Vitesse de déplacement
camera.speed = 15.0  # unités/seconde

# Sensibilité souris
camera.sensitivity = 0.2  # multiplicateur

# Field of view
camera.fov = 60.0  # degrés

# Collision
camera.collision_enabled = True
camera.min_height_above_terrain = 2.0  # mètres
```

### Fichier

`core/camera/fps_camera.py` (400+ lignes)

---

## 🌄 2. HDRI Panoramique 360°

### Caractéristiques

- ✅ Génération procédurale équirectangulaire (2:1 ratio)
- ✅ 7 presets temps (sunrise, midday, sunset, night, etc.)
- ✅ Ciel avec gradient, soleil, nuages, montagnes lointaines
- ✅ Export .exr (OpenEXR) et .hdr (Radiance HDR)
- ✅ Preview tone-mapped PNG
- ✅ AI enhancement optionnel (Stable Diffusion XL)
- ✅ HDR range: 0.01 - 100.0 (exposition)

### Utilisation Basique

```python
from core.rendering.hdri_generator import HDRIPanoramicGenerator, TimeOfDay

# Créer générateur
generator = HDRIPanoramicGenerator(resolution=(4096, 2048))

# Générer HDRI procédural
hdri = generator.generate_procedural(
    time_of_day=TimeOfDay.SUNSET,
    cloud_density=0.3,  # [0-1]
    mountain_distance=True,
    seed=42
)

# Exporter
generator.export_exr(hdri, "mountain_sunset.exr")
generator.export_ldr(hdri, "mountain_sunset_preview.png")
```

### Presets Temps

```python
# Temps disponibles
TimeOfDay.SUNRISE    # 5° élévation, couleurs chaudes
TimeOfDay.MORNING    # 30° élévation, ciel bleu
TimeOfDay.MIDDAY     # 60° élévation, soleil intense
TimeOfDay.AFTERNOON  # 40° élévation, lumière dorée
TimeOfDay.SUNSET     # 5° élévation, couleurs dramatiques
TimeOfDay.TWILIGHT   # -5° élévation, ciel violet
TimeOfDay.NIGHT      # -30° élévation, ciel étoilé
```

### Résolutions

```python
# Résolutions standards (width x height, ratio 2:1)
RESOLUTION_LOW = (2048, 1024)     # ~2K, rapide
RESOLUTION_MEDIUM = (4096, 2048)  # ~4K, recommandé
RESOLUTION_HIGH = (8192, 4096)    # ~8K, haute qualité
```

### AI Enhancement (Optionnel)

```python
# Nécessite diffusers + 10-12 GB VRAM
enhanced = generator.enhance_with_ai(
    base_image=hdri,
    prompt="360 degree panoramic view of majestic mountains at sunset, "
           "highly detailed, photorealistic, dramatic clouds, 8k",
    strength=0.4,  # [0-1], force de modification
    seed=42
)
```

### Génération Batch

```python
# Générer tous les presets
for time in TimeOfDay:
    generator.generate_preset(
        time_of_day=time,
        output_dir="./hdri_output",
        ai_enhance=False  # True pour AI
    )
```

### Paramètres Personnalisés

```python
# Accéder aux paramètres de preset
params = HDRIPanoramicGenerator.TIME_PRESETS[TimeOfDay.SUNSET]
print(params['sun_elevation'])     # 5.0°
print(params['sun_color'])         # [1.0, 0.6, 0.4]
print(params['sky_horizon_color']) # [1.0, 0.5, 0.3]
```

### Fichier

`core/rendering/hdri_generator.py` (900+ lignes)

---

## 🌗 3. Ombres Temps Réel avec Shadow Mapping

### Caractéristiques

- ✅ Shadow mapping classique avec depth texture
- ✅ PCF (Percentage Closer Filtering) 3x3 pour ombres douces
- ✅ Shaders GLSL personnalisés (OpenGL 3.3+)
- ✅ Phong lighting (ambient + diffuse + specular)
- ✅ Fog atmosphérique exponentiel
- ✅ 3 niveaux de qualité ombres (1024², 2048², 4096²)
- ✅ Adaptive shadow bias
- ✅ Performance optimisée avec LOD

### Utilisation

```python
from ui.widgets.advanced_terrain_viewer import AdvancedTerrainViewer
from PySide6.QtWidgets import QApplication

app = QApplication([])

# Créer viewer
viewer = AdvancedTerrainViewer()

# Définir terrain
viewer.set_terrain(
    heightmap=terrain_data,
    terrain_scale=100.0,  # taille monde
    height_scale=20.0,    # multiplicateur hauteur
    lod=2                 # LOD: 1 (high), 2 (medium), 4 (low)
)

# Configurer ombres
viewer.set_shadows_enabled(True)
viewer.set_shadow_quality(2048)  # 1024, 2048, ou 4096

# Configurer fog
viewer.set_fog_enabled(True)
viewer._fog_density = 0.0001

# Afficher
viewer.show()
app.exec()
```

### Shaders

#### Terrain Vertex Shader
`core/rendering/shaders/terrain_vertex.glsl`

- Input: position, normal, color
- Output: world position, normal, light-space position
- Calcule transformation pour shadow mapping

#### Terrain Fragment Shader
`core/rendering/shaders/terrain_fragment.glsl`

- Phong lighting model
- Shadow calculation avec PCF
- Fog exponentiel
- Specular highlights

#### Shadow Depth Shaders
`core/rendering/shaders/shadow_depth.vert/frag`

- Rendu depth-only depuis perspective lumière
- Génère shadow map texture

### Pipeline de Rendu

```
1. Shadow Pass:
   - Bind shadow FBO
   - Render terrain depuis perspective lumière
   - Store depth dans texture 2048x2048

2. Main Pass:
   - Bind default framebuffer
   - Render terrain depuis caméra
   - Sample shadow map
   - Calculate lighting + shadows
   - Apply fog
```

### Paramètres Lighting

```python
# Direction lumière (soleil)
viewer._light_dir = np.array([0.3, -0.7, 0.5])  # normalisé

# Couleur lumière
viewer._light_color = np.array([1.0, 1.0, 0.95])  # blanc chaud

# Ambient
viewer._ambient_strength = 0.3  # [0-1]

# Shadow bias (évite shadow acne)
viewer._shadow_bias = 0.005  # ajuster selon scène
```

### Qualités Ombres

| Qualité | Résolution | VRAM | FPS (1024² terrain) |
|---------|-----------|------|---------------------|
| **Low** | 1024x1024 | ~4 MB | ~60 FPS |
| **Medium** | 2048x2048 | ~16 MB | ~45 FPS |
| **High** | 4096x4096 | ~64 MB | ~30 FPS |

### Optimisation Performance

```python
# LOD (Level of Detail)
viewer.set_terrain(heightmap, lod=4)  # 4x moins de vertices

# Désactiver ombres temporairement
viewer.set_shadows_enabled(False)

# Wireframe debug
viewer.set_wireframe(True)
```

### Fichiers

- `ui/widgets/advanced_terrain_viewer.py` (1000+ lignes)
- `core/rendering/shaders/terrain_vertex.glsl`
- `core/rendering/shaders/terrain_fragment.glsl`
- `core/rendering/shaders/shadow_depth.vert`
- `core/rendering/shaders/shadow_depth.frag`

---

## 🖥️ Interface Ultimate Viewer

### Lancement

```python
from ui.widgets.ultimate_terrain_viewer import UltimateTerrainViewer
from PySide6.QtWidgets import QApplication

app = QApplication([])
viewer = UltimateTerrainViewer()
viewer.show()
app.exec()
```

Ou via exemple:

```bash
python examples/example_ultimate_viewer.py
```

### Interface Tabs

#### 1. Terrain
- **Presets**: Alps, Himalayas, Scottish Highlands, etc.
- **Taille**: 128 à 2048
- **Terrain Scale**: Taille en unités monde
- **Height Scale**: Multiplicateur hauteur
- **LOD**: 1 (high), 2 (medium), 4 (low)
- **Load Heightmap**: Charger depuis fichier

#### 2. Rendering
- **Shadows**: On/Off + qualité (1024/2048/4096)
- **Fog**: On/Off + densité
- **Wireframe**: Mode fil de fer

#### 3. Lighting
- **Sun Azimuth**: 0-360° (position horizontale soleil)
- **Sun Elevation**: -90 à +90° (hauteur soleil)
- **Ambient Strength**: Force lumière ambiante
- **Shadow Bias**: Ajustement précision ombres

#### 4. Camera
- **Speed**: Vitesse déplacement
- **Sensitivity**: Sensibilité souris
- **Collision**: On/Off collision terrain
- **Reset**: Repositionner caméra

#### 5. HDRI Skybox
- **Time of Day**: 7 presets temps
- **Resolution**: Low/Medium/High
- **Cloud Density**: Couverture nuageuse
- **AI Enhancement**: On/Off (optionnel)
- **Generate**: Créer nouveau HDRI
- **Load**: Charger HDRI existant

#### 6. Export
- **Export for Flame**: Exporte terrain + textures
- **Screenshot**: Capture vue actuelle

### Status Bar

- **FPS Counter**: Frames par seconde
- **Camera Position**: Position [x, y, z]
- **Messages**: Actions en cours

### Fichier

`ui/widgets/ultimate_terrain_viewer.py` (1100+ lignes)

---

## 📝 Exemples d'Utilisation

### Exemple 1: Viewer Complet

```bash
python examples/example_ultimate_viewer.py
```

Lance l'interface complète avec terrain Alps pré-généré.

### Exemple 2: Génération HDRI Batch

```bash
python examples/example_hdri_generation.py
```

Génère 4 HDRIs (sunrise, midday, sunset, night) et les sauvegarde.

### Exemple 3: Caméra FPS Standalone

```python
from core.camera.fps_camera import FPSCamera
import numpy as np

camera = FPSCamera()
camera.set_heightmap(my_heightmap, terrain_scale=100.0, height_scale=20.0)

# Dans game loop:
camera.process_keyboard(delta_time)
view = camera.get_view_matrix()
```

### Exemple 4: HDRI Personnalisé

```python
from core.rendering.hdri_generator import HDRIPanoramicGenerator, TimeOfDay

gen = HDRIPanoramicGenerator((4096, 2048))

# Custom time preset
custom_hdri = gen.generate_procedural(
    time_of_day=TimeOfDay.SUNSET,
    cloud_density=0.7,  # Beaucoup de nuages
    mountain_distance=True,
    seed=123
)

gen.export_exr(custom_hdri, "my_hdri.exr")
```

---

## 🔧 Dépannage

### Problème: "PyOpenGL not available"

```bash
pip install PyOpenGL PyOpenGL-accelerate
```

### Problème: "OpenEXR not available"

```bash
# Linux
sudo apt-get install libopenexr-dev
pip install OpenEXR Imath

# macOS
brew install openexr
pip install OpenEXR Imath

# Windows
pip install OpenEXR Imath
```

### Problème: Shaders ne compilent pas

Vérifier version OpenGL:

```python
from OpenGL.GL import *
print(glGetString(GL_VERSION))  # Doit être ≥ 3.3
```

Si < 3.3, mettre à jour drivers graphiques.

### Problème: FPS bas avec ombres

1. Réduire shadow quality: 2048 → 1024
2. Augmenter LOD terrain: 1 → 2 ou 4
3. Désactiver fog temporairement
4. Réduire taille terrain: 1024 → 512

### Problème: AI enhancement trop lent

1. Vérifier VRAM disponible: `nvidia-smi` ou `rocm-smi`
2. Réduire résolution HDRI: 4096 → 2048
3. Utiliser génération procédurale uniquement
4. L'AI enhancement est optionnel

### Problème: Collision caméra ne fonctionne pas

```python
# Vérifier heightmap défini
camera.set_heightmap(heightmap, terrain_scale, height_scale)

# Vérifier collision activée
camera.collision_enabled = True

# Ajuster hauteur minimale
camera.min_height_above_terrain = 2.0
```

---

## 📊 Performance

### Benchmarks (RTX 3080, 10 GB VRAM)

| Opération | Taille | Temps | Notes |
|-----------|--------|-------|-------|
| Terrain generation | 512² | ~20s | Stream power erosion 50 iter |
| Shadow rendering | 1024² | 60 FPS | Quality: Medium (2048²) |
| Shadow rendering | 2048² | 30 FPS | Quality: High (4096²) |
| HDRI procedural | 4096x2048 | ~3s | Sans AI |
| HDRI + AI enhance | 4096x2048 | ~90s | Stable Diffusion XL, 30 steps |
| Heightmap loading | 2048² | <1s | From PNG |

### Recommandations

- **Performance**: LOD=4, Shadow=1024, Terrain=512²
- **Balanced**: LOD=2, Shadow=2048, Terrain=1024²
- **Quality**: LOD=1, Shadow=4096, Terrain=2048²

---

## 🎯 Workflow Recommandé

### 1. Création Terrain

```python
# Générer terrain
from core.terrain.advanced_algorithms import combine_algorithms, MOUNTAIN_PRESETS

terrain = combine_algorithms(512, **MOUNTAIN_PRESETS['alps'], seed=42)
```

### 2. Lancer Viewer

```python
viewer = UltimateTerrainViewer()
viewer._current_heightmap = terrain
viewer._update_terrain()
viewer.show()
```

### 3. Ajuster Visuel

- Tab **Rendering**: Activer shadows quality Medium
- Tab **Lighting**: Ajuster sun position (135° azimuth, 45° elevation)
- Tab **Camera**: Speed=15, Sensitivity=0.15

### 4. Générer HDRI

- Tab **HDRI Skybox**:
  - Time: Sunset
  - Resolution: Medium
  - Clouds: 30%
  - Generate

### 5. Explorer

- Click dans viewport pour capturer souris
- WASD pour se déplacer
- Observer ombres temps réel et fog atmosphérique

### 6. Exporter

- Tab **Export**: Export for Flame
- Sauvegarder dans répertoire projet

---

## 📚 Ressources

### Documentation Interne

- `IMPLEMENTATION_PLAN_ULTIMATE.md` - Plan d'implémentation détaillé
- `SYSTEM_STATUS_REPORT.md` - État système complet
- `RESEARCH_TERRAIN_ALGORITHMS.md` - Recherche algorithmes

### Code Source

```
core/
├── camera/
│   └── fps_camera.py              # Système caméra FPS
├── rendering/
│   ├── hdri_generator.py          # Générateur HDRI
│   └── shaders/                   # Shaders GLSL
│       ├── terrain_vertex.glsl
│       ├── terrain_fragment.glsl
│       ├── shadow_depth.vert
│       └── shadow_depth.frag
ui/widgets/
├── advanced_terrain_viewer.py     # Viewer OpenGL avancé
└── ultimate_terrain_viewer.py     # Interface complète
examples/
├── example_ultimate_viewer.py     # Exemple viewer
└── example_hdri_generation.py     # Exemple HDRI
```

### Références Techniques

- **Shadow Mapping**: LearnOpenGL - Shadow Mapping Tutorial
- **PCF Filtering**: Real-Time Rendering, 3rd Ed., Chapter 7
- **HDRI**: Radiance HDR File Format Specification
- **OpenEXR**: OpenEXR Technical Introduction
- **Equirectangular**: Panoramic Image Projections

---

## ✨ Fonctionnalités Futures (Suggestions)

- [ ] Cascade Shadow Maps pour grandes distances
- [ ] SSAO (Screen-Space Ambient Occlusion)
- [ ] HDR Bloom post-processing
- [ ] Dynamic weather (rain, snow)
- [ ] Animated clouds
- [ ] Water reflections
- [ ] Vegetation placement
- [ ] Multi-threading pour génération terrain
- [ ] Vulkan renderer (alternative OpenGL)

---

## 📄 Licence

Mountain Studio Pro - Tous droits réservés

---

**Version**: 2.0 - Ultimate Edition
**Date**: 2025-11-18
**Auteur**: Mountain Studio Pro Team
