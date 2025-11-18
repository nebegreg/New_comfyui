# Guide d'Intégration - Nouvelles Fonctionnalités

Guide complet pour utiliser les nouvelles fonctionnalités ajoutées à Mountain Studio Pro.

## 📋 Table des Matières

1. [Nouveaux Algorithmes de Terrain](#nouveaux-algorithmes)
2. [Installateur ComfyUI](#installateur-comfyui)
3. [Preview 3D Améliorée](#preview-3d)
4. [Système PBR](#système-pbr)
5. [Export Autodesk Flame](#export-flame)

---

## 1. Nouveaux Algorithmes de Terrain

### Spectral Synthesis (FFT-Based)

**Quand l'utiliser**: Pour terrains très variés avec contrôle précis des fréquences spatiales.

```python
from core.terrain.advanced_algorithms import spectral_synthesis

# Génération basique
heightmap = spectral_synthesis(
    size=1024,
    beta=2.0,      # 2.0 = terrain naturel, 2.5-3.0 = très rugueux
    amplitude=1.0,
    seed=42
)

# Terrain lisse (collines)
smooth_terrain = spectral_synthesis(size=1024, beta=1.5, seed=42)

# Terrain très rugueux (montagnes escarpées)
rugged_terrain = spectral_synthesis(size=1024, beta=2.8, seed=42)
```

**Avantages**:
- Pas d'artifacts de grille
- Contrôle précis du "bruit"
- Très rapide avec FFT

### Stream Power Erosion

**Quand l'utiliser**: Pour érosion géomorphologiquement réaliste avec chenaux.

```python
from core.terrain.advanced_algorithms import stream_power_erosion

# Appliquer érosion stream power
eroded = stream_power_erosion(
    heightmap,
    iterations=100,
    K_erosion=0.015,    # Coefficient d'érodabilité (0.01-0.02 typique)
    m_area_exp=0.5,     # Exposant aire drainage (0.4-0.6)
    n_slope_exp=1.0,    # Exposant pente (1.0-2.0)
    dt=0.01,            # Pas de temps (plus petit = plus stable)
    uplift_rate=0.0     # Soulèvement tectonique (optionnel)
)
```

**Paramètres Calibrés pour Différents Terrains**:

```python
# Montagnes jeunes (uplift actif)
eroded_young = stream_power_erosion(
    heightmap,
    K_erosion=0.012,
    uplift_rate=0.001  # Soulèvement actif
)

# Canyons profonds
eroded_canyon = stream_power_erosion(
    heightmap,
    iterations=200,
    K_erosion=0.025,   # Érosion forte
    n_slope_exp=1.5    # Incision profonde
)
```

### Érosion Glaciaire

**Quand l'utiliser**: Pour créer vallées en U caractéristiques.

```python
from core.terrain.advanced_algorithms import glacial_erosion

# Appliquer érosion glaciaire
glaciated = glacial_erosion(
    heightmap,
    altitude_threshold=0.7,  # Zones glaciaires (30% supérieur)
    strength=0.3,            # Force d'érosion (0.1-0.5)
    u_valley_factor=0.8      # Prononciation forme U (0-1)
)
```

**Combinaison Réaliste** (Alpes):
```python
# 1. Terrain de base
base = spectral_synthesis(1024, beta=2.2, seed=42)

# 2. Érosion fluviale (rivières)
with_rivers = stream_power_erosion(base, iterations=100)

# 3. Érosion glaciaire (vallées en U)
alpine = glacial_erosion(with_rivers, altitude_threshold=0.7)
```

### Soulèvement Tectonique

**Quand l'utiliser**: Pour simuler construction de montagnes.

```python
from core.terrain.advanced_algorithms import tectonic_uplift

# Soulèvement gaussien
uplifted = tectonic_uplift(
    heightmap,
    center=(512, 512),  # Centre du soulèvement
    magnitude=0.3,      # Hauteur (0-1)
    radius=0.4          # Rayon (fraction de la taille)
)
```

### Génération Combinée (Un Seul Appel)

**Utilisation Recommandée**: Pour workflow complet en une fonction.

```python
from core.terrain.advanced_algorithms import combine_algorithms, MOUNTAIN_PRESETS

# Utiliser preset
alps = combine_algorithms(
    size=1024,
    **MOUNTAIN_PRESETS['alps'],
    seed=42
)

# Himalaya avec tectonique active
himalaya = combine_algorithms(
    size=1024,
    **MOUNTAIN_PRESETS['himalayas'],
    seed=42
)

# Personnalisé
custom = combine_algorithms(
    size=1024,
    algorithm='spectral',
    beta=2.3,
    erosion_iterations=150,
    apply_glacial=True,
    apply_uplift=True,
    seed=42
)
```

**Presets Disponibles**:
- `alps`: Alpes (spectral + fluvial + glaciaire)
- `himalayas`: Himalaya (hybrid + tectonique + glaciaire)
- `scottish_highlands`: Écosse (spectral + glaciaire fort)
- `grand_canyon`: Grand Canyon (ridged + érosion massive)
- `rocky_mountains`: Rocheuses (hybrid + tectonique + glaciaire)

---

## 2. Installateur ComfyUI

### Widget d'Installation

**Intégration dans le GUI**:

```python
from ui.widgets.comfyui_installer_widget import ComfyUIInstallerWidget

# Dans votre main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Ajouter tab d'installation
        installer_widget = ComfyUIInstallerWidget()

        # Connecter signal de complétion
        installer_widget.installation_complete.connect(self.on_install_complete)

        # Ajouter au tab widget
        self.tabs.addTab(installer_widget, "ComfyUI Installer")

    def on_install_complete(self):
        print("Installation terminée!")
```

### Utilisation Programmatique

```python
from core.ai.comfyui_installer import ComfyUIInstaller

# Créer installateur
installer = ComfyUIInstaller()

# Définir chemin ComfyUI
if installer.set_comfyui_path("/path/to/ComfyUI"):
    print("✓ Chemin valide")

    # Voir statut
    status = installer.get_installation_status()
    print(f"Modèles: {len(status['models'])}")
    print(f"Nodes: {len(status['nodes'])}")

    # Installer modèle spécifique
    models = installer.get_recommended_models()
    realistic_vision = models[0]  # Realistic Vision V5.1

    def progress(current_mb, total_mb, pct):
        print(f"Téléchargement: {pct:.1f}%")

    success = installer.download_model(realistic_vision, progress)

    # Installer tous les composants pour fonctionnalités
    successful, failed = installer.install_all_required([
        'pbr_textures',
        'landscape_generation'
    ])
    print(f"Installé: {successful}, Échoué: {failed}")
```

### Modèles Recommandés

| Modèle | Taille | Usage |
|--------|--------|-------|
| Realistic Vision V5.1 | 2.1 GB | Textures réalistes |
| SD XL Base 1.0 | 6.9 GB | Haute qualité (lent) |
| VAE-ft-mse-840000 | 335 MB | Meilleures couleurs |
| ControlNet Normal | 1.4 GB | Normal maps |
| ControlNet Depth | 1.4 GB | Depth-guided |

### Custom Nodes Recommandés

| Node | Usage |
|------|-------|
| ComfyUI-Manager | Gestion nodes |
| ComfyUI_Comfyroll | Utilitaires |
| Impact-Pack | Traitement avancé |
| controlnet_aux | Preprocesseurs |

---

## 3. Preview 3D Améliorée

### Widget de Preview

**Intégration**:

```python
from ui.widgets.terrain_preview_3d import TerrainPreview3DWidget

# Créer widget
preview_3d = TerrainPreview3DWidget()

# Définir heightmap
preview_3d.set_heightmap(heightmap)

# Connecter à changements de caméra
preview_3d.camera_changed.connect(lambda state: print(f"Caméra: {state}"))
```

### Contrôles Disponibles

**Via UI**:
- Vertical Exaggeration: Slider 1.0x - 10.0x
- Render Mode: Solid / Wireframe / Textured
- Show Grid: Checkbox
- Show Normals: Checkbox (TODO)
- Lighting: Ambient et Diffuse sliders

**Via Code**:

```python
# Changer exagération verticale
preview_3d.vertical_exaggeration = 3.0
preview_3d._update_terrain_mesh()

# Changer mode rendu
preview_3d.render_mode = 'wireframe'  # 'solid', 'wireframe', 'textured'
preview_3d._update_terrain_mesh()

# Vues prédéfinies
preview_3d._set_view(elevation=90, azimuth=0)   # Vue du dessus
preview_3d._set_view(elevation=0, azimuth=90)   # Vue de côté
preview_3d._reset_camera()                       # Reset

# État caméra
state = preview_3d._get_camera_state()
# {'distance': 100.0, 'elevation': 30.0, 'azimuth': 45.0, 'center': (0,0,0)}

# Restaurer état
preview_3d.set_camera_state(state)

# Export snapshot
preview_3d.export_snapshot("terrain_view.png")
```

### Couleurs par Élévation

Le widget colorie automatiquement le terrain selon l'élévation:
- Bleu: Vallées / eau (0-20%)
- Vert: Végétation (20-40%)
- Brun: Roche (40-70%)
- Blanc: Neige (70-100%)

**Personnaliser les Couleurs**:

```python
# Modifier _calculate_vertex_colors() dans le widget

def custom_colors(self, z_values):
    """Couleurs custom basées sur élévation"""
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())

    colors = np.zeros((len(z_values), 4))

    # Votre gradient custom
    colors[:, 0] = z_norm  # R
    colors[:, 1] = 1 - z_norm  # G
    colors[:, 2] = 0.5  # B
    colors[:, 3] = 1.0  # A

    return colors
```

---

## 4. Système PBR

### Génération Automatique

```python
from core.ai.comfyui_integration import generate_terrain_pbr_auto

# UN SEUL APPEL pour tout générer!
files = generate_terrain_pbr_auto(
    heightmap,
    output_dir='terrain_pbr',
    resolution=2048,
    material_type='rock'  # 'rock', 'grass', 'snow', 'sand', 'dirt'
)

# Fichiers générés:
# files = {
#     'diffuse': 'terrain_pbr/terrain_rock_diffuse.png',
#     'normal': 'terrain_pbr/terrain_rock_normal.png',
#     'roughness': 'terrain_pbr/terrain_rock_roughness.png',
#     'ao': 'terrain_pbr/terrain_rock_ao.png',
#     'height': 'terrain_pbr/terrain_rock_height.png',
#     'metallic': 'terrain_pbr/terrain_rock_metallic.png'
# }
```

### Génération Manuelle (Plus de Contrôle)

```python
from core.rendering.pbr_texture_generator import PBRTextureGenerator

# Créer générateur
generator = PBRTextureGenerator(resolution=2048)

# Générer PBR set
pbr_textures = generator.generate_from_heightmap(
    heightmap,
    material_type='rock',
    make_seamless=True,
    detail_level=1.0  # 0.5-2.0
)

# Accéder aux maps
diffuse = pbr_textures['diffuse']  # (2048, 2048, 3) RGB [0, 255]
normal = pbr_textures['normal']    # (2048, 2048, 3) RGB [0, 255]
roughness = pbr_textures['roughness']  # (2048, 2048) [0, 255]
# etc.

# Export manuel
exported = generator.export_pbr_set(
    pbr_textures,
    output_dir='my_pbr',
    prefix='mountain'
)
```

### Matériaux Disponibles

| Material | Base Color | Roughness Range | Metallic |
|----------|------------|-----------------|----------|
| rock | Gris-brun | 0.7-0.95 | 0.0 |
| grass | Vert | 0.6-0.85 | 0.0 |
| snow | Blanc-bleu | 0.3-0.6 | 0.0 |
| sand | Jaune-tan | 0.5-0.75 | 0.0 |
| dirt | Brun foncé | 0.65-0.85 | 0.0 |

### Avec ComfyUI (Optionnel)

```python
from core.ai.comfyui_integration import generate_complete_pbr_set

# Tentative ComfyUI, fallback procédural
pbr_result = generate_complete_pbr_set(
    heightmap,
    material_type='rock',
    resolution=2048,
    use_comfyui=True,  # Tente ComfyUI d'abord
    comfyui_server="127.0.0.1:8188",
    make_seamless=True,
    output_dir='output'
)

# Vérifier la source
if pbr_result['source'] == 'comfyui':
    print("Généré par ComfyUI!")
else:
    print("Fallback procédural (haute qualité)")
```

---

## 5. Export Autodesk Flame

### Export Complet

```python
from core.export.professional_exporter import ProfessionalExporter
from core.rendering.pbr_texture_generator import PBRTextureGenerator

# Générer terrain et PBR
heightmap = spectral_synthesis(1024, beta=2.2)

pbr_gen = PBRTextureGenerator(resolution=1024)
pbr = pbr_gen.generate_from_heightmap(heightmap, material_type='rock')

# Générer maps dérivées
from core.terrain.heightmap_generator import HeightmapGenerator
terrain_gen = HeightmapGenerator(1024, 1024)
terrain_gen.heightmap = heightmap

normal_map = terrain_gen.generate_normal_map(heightmap=heightmap)
depth_map = terrain_gen.generate_depth_map(heightmap=heightmap)
ao_map = terrain_gen.generate_ambient_occlusion(heightmap=heightmap, samples=8)

# Export pour Flame
exporter = ProfessionalExporter('flame_export')

exported_files = exporter.export_for_autodesk_flame(
    heightmap=heightmap,
    normal_map=normal_map,
    depth_map=depth_map,
    ao_map=ao_map,
    diffuse_map=pbr['diffuse'],
    roughness_map=pbr['roughness'],
    splatmaps=None,  # Optionnel
    tree_instances=None,  # Optionnel
    mesh_subsample=2,  # Sous-échantillonnage (1-4)
    scale_y=50.0  # Échelle verticale
)

print(f"Exporté {len(exported_files)} fichiers")
print(f"OBJ: {exported_files['obj']}")
print(f"MTL: {exported_files['mtl']}")
```

### Fichiers Exportés

```
flame_export/
├── terrain.obj            # Mesh 3D
├── terrain.mtl            # Materials
├── textures/
│   ├── height.png         # Heightmap (8-bit)
│   ├── normal.png         # Normal map (8-bit RGB)
│   ├── depth.png          # Depth map (16-bit)
│   ├── ao.png             # AO (8-bit)
│   ├── diffuse.png        # Diffuse/Albedo (8-bit RGB)
│   └── roughness.png      # Roughness (8-bit)
└── README_FLAME.txt       # Instructions import
```

### Import dans Flame

1. Ouvrir Autodesk Flame 2025.2.2
2. File → Import → Media
3. Sélectionner `terrain.obj`
4. Les textures sont automatiquement chargées via le `.mtl`
5. Voir `README_FLAME.txt` pour détails

---

## 6. Workflow Complet Exemple

### Scénario: Créer Terrain Alpin Réaliste pour VFX

```python
#!/usr/bin/env python3
"""
Workflow complet: Terrain alpin pour Autodesk Flame
"""

from core.terrain.advanced_algorithms import combine_algorithms, MOUNTAIN_PRESETS
from core.ai.comfyui_integration import generate_terrain_pbr_auto
from core.terrain.heightmap_generator import HeightmapGenerator
from core.export.professional_exporter import ProfessionalExporter

# 1. Générer terrain avec algorithmes avancés
print("[1/4] Génération terrain alpin...")
heightmap = combine_algorithms(
    size=2048,
    **MOUNTAIN_PRESETS['alps'],
    seed=42
)

# 2. Générer textures PBR
print("[2/4] Génération textures PBR...")
pbr_files = generate_terrain_pbr_auto(
    heightmap,
    output_dir='alpine_pbr',
    resolution=2048,
    material_type='rock'
)

# 3. Générer maps dérivées
print("[3/4] Génération normal, depth, AO...")
terrain_gen = HeightmapGenerator(2048, 2048)
normal_map = terrain_gen.generate_normal_map(heightmap=heightmap, strength=1.0)
depth_map = terrain_gen.generate_depth_map(heightmap=heightmap)
ao_map = terrain_gen.generate_ambient_occlusion(heightmap=heightmap, samples=16)

# 4. Export pour Flame
print("[4/4] Export Autodesk Flame...")
exporter = ProfessionalExporter('alpine_export_flame')

exported = exporter.export_for_autodesk_flame(
    heightmap=heightmap,
    normal_map=normal_map,
    depth_map=depth_map,
    ao_map=ao_map,
    diffuse_map=pbr_files['diffuse'],  # Charger depuis fichier
    roughness_map=pbr_files['roughness'],
    mesh_subsample=2,
    scale_y=100.0  # Alpes: haute altitude
)

print("\n✅ Workflow terminé!")
print(f"📁 Fichiers dans: alpine_export_flame/")
print(f"   - {len(exported)} fichiers exportés")
print(f"   - Import terrain.obj dans Flame")
```

---

## 7. Performance et Optimisation

### Tailles Recommandées

| Usage | Résolution | Temps Génération* |
|-------|------------|-------------------|
| Preview/Test | 256x256 | ~1s |
| Production Standard | 1024x1024 | ~10s |
| Haute Qualité | 2048x2048 | ~40s |
| Ultra Haute | 4096x4096 | ~3min |

*Sur CPU moderne, avec érosion 100 iterations

### Optimisations

```python
# Utiliser GPU si disponible (pour érosion)
# (TODO: Implémentation CuPy)

# Réduire iterations pour preview
quick_preview = combine_algorithms(
    size=512,
    algorithm='spectral',
    erosion_iterations=20,  # Réduit
    apply_glacial=False,    # Désactivé
    seed=42
)

# Sous-échantillonner mesh export
exported = exporter.export_for_autodesk_flame(
    ...,
    mesh_subsample=4  # 4x moins de vertices
)
```

---

## 8. Dépannage

### Erreur: "ComfyUI path not set"

**Solution**:
```python
installer = ComfyUIInstaller()
installer.set_comfyui_path("/path/to/ComfyUI")
```

### Erreur: "Model download failed"

**Solution**:
- Vérifier connexion internet
- Essayer URL alternative
- Télécharger manuellement et placer dans `models/checkpoints/`

### Preview 3D ne s'affiche pas

**Solution**:
```python
# Vérifier que PyOpenGL est installé
pip install PyOpenGL PyOpenGL_accelerate

# Tester
python3 -c "import OpenGL.GL; print('OK')"
```

### Textures sombres/claires

**Solution**:
```python
# Ajuster detail_level
pbr = generator.generate_from_heightmap(
    heightmap,
    detail_level=1.5  # Plus de variation
)
```

---

## 9. Prochaines Étapes

### Fonctionnalités à Venir

- [ ] GPU acceleration (CuPy) pour érosion
- [ ] Export FBX (en plus d'OBJ)
- [ ] Vegetation placement 3D
- [ ] Animation de croissance de terrain
- [ ] Batch processing pour séquences
- [ ] Integration ComfyUI workflows custom

### Contribution

Pour contribuer:
1. Fork le repo
2. Créer branch `feature/ma-fonctionnalite`
3. Suivre conventions de nommage (voir NAMING_CONSISTENCY_ANALYSIS.md)
4. Tester avec `pytest`
5. Pull request

---

**Pour plus d'informations**:
- `RESEARCH_TERRAIN_ALGORITHMS.md`: Détails algorithmes
- `PBR_TEXTURE_SYSTEM.md`: Système PBR complet
- `INSTALL_ROCKY_LINUX.md`: Installation Rocky Linux
- `NAMING_CONSISTENCY_ANALYSIS.md`: Conventions code

**Support**: Ouvrir une issue sur GitHub
