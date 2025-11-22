# 🚀 Quick Start Guide - Mountain Studio ULTIMATE v2.0

## ⚡ Résolution Rapide des Erreurs ComfyUI

### Problème: "ImageSegmentation does not exist"
### Problème: "sd_xl_base_1.0.safetensors not in []"
### Problème: Seed -1 invalide

## 🔧 **SOLUTION AUTOMATIQUE**

```bash
# 1. Installer automatiquement tout ce qu'il faut
python3 comfyui_auto_setup.py

# Ou avec chemin spécifique:
python3 comfyui_auto_setup.py --comfyui-path /path/to/ComfyUI
```

**Ce script va:**
- ✅ Télécharger `sd_xl_base_1.0.safetensors` (7 GB)
- ✅ Installer les custom nodes manquants (ImageSegmentation, etc.)
- ✅ Créer un workflow fixé avec seed valide
- ✅ Installer toutes les dépendances Python

**Temps estimé**: 15-30 minutes (selon connexion internet)

---

## 📋 Vérifier l'Installation

```bash
# Voir ce qui est déjà installé
python3 comfyui_auto_setup.py --check-only
```

**Output exemple:**
```
✅ model_sd_xl_base_1.0.safetensors
✅ model_sdxl_vae.safetensors
✅ node_ComfyUI-Manager
✅ node_comfyui_controlnet_aux
✅ node_ComfyUI-Impact-Pack
```

---

## 🎯 **WORKFLOW COMPLET** (De zéro à génération)

### Étape 1: Installer ComfyUI (si pas encore fait)

```bash
# Cloner ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Installer dépendances
pip install -r requirements.txt
```

### Étape 2: Auto-Setup Mountain Studio

```bash
# Retourner au dossier Mountain Studio
cd /home/user/New_comfyui

# Lancer l'auto-setup
python3 comfyui_auto_setup.py --comfyui-path ../ComfyUI

# Attendre téléchargement modèles + installation nodes
# ⏱️ 15-30 minutes
```

### Étape 3: Démarrer ComfyUI

```bash
# Aller dans ComfyUI
cd ../ComfyUI

# Lancer le serveur
python main.py

# Attendre: "To see the GUI go to: http://127.0.0.1:8188"
```

### Étape 4: Charger le Workflow Fixé

Dans ComfyUI (navigateur http://127.0.0.1:8188):
1. Cliquer "Load" (en haut)
2. Sélectionner `mountain_studio_workflow_fixed.json`
3. Le workflow devrait charger sans erreurs!

### Étape 5: Lancer Mountain Studio

```bash
# Nouveau terminal, retourner au projet
cd /home/user/New_comfyui

# Lancer Mountain Studio
python3 mountain_studio_ultimate_v2.py
```

### Étape 6: Tester ComfyUI depuis Mountain Studio

Dans l'application:
1. Aller dans l'onglet **"🎨 AI Textures"**
2. Cliquer **"🔍 Check Connection"**
3. Statut devrait afficher: **"✅ Connected"**
4. Entrer un prompt: `ultra realistic mountain rock texture, 4k, PBR`
5. Cliquer **"🎨 Generate AI Textures"**

---

## 🏔️ **PRESETS INTÉGRÉS**

Mountain Studio inclut **10+ presets professionnels** prêts à l'emploi!

### Catégories de Presets:

#### 🎬 **VFX Production** (Films/Publicités)
- **VFX Epic Mountain**: Pics alpins dramatiques à l'heure dorée (4K)
- **VFX Misty Forest**: Forêt de montagne brumeuse, ambiance cinéma

#### 🎮 **Game Development** (Unreal/Unity)
- **Game: Unreal Engine Landscape**: Optimisé UE5, maps PBR complètes
- **Game: Unity Terrain**: Textures 2K, splatmaps, instances végétation

#### 📷 **Landscape Photography** (Style photo pro)
- **Photo: Golden Hour Alpine**: Style National Geographic
- **Photo: B&W Ansel Adams Style**: Noir & blanc dramatique

#### 🎨 **Artistic / Concept Art**
- **Art: Fantasy Mountain Peaks**: Montagnes fantastiques exagérées
- **Art: Minimalist Zen Mountain**: Paysage minimaliste, apaisant

#### ⚡ **Quick Test** (Tests rapides)
- **Test: Quick Preview**: 512x512, 5-10 secondes
- **Test: Erosion Comparison**: 1024x1024, test érosion

### Comment utiliser les presets:

```python
from config.professional_presets import PresetManager

# Charger le manager
manager = PresetManager()

# Lister tous les presets
presets = manager.list_presets()
print(presets)

# Lister par catégorie
vfx_presets = manager.list_presets(category='vfx_production')

# Charger un preset
preset = manager.get_preset('vfx_epic_mountain')

# Appliquer les paramètres
width = preset.terrain.width          # 4096
height = preset.terrain.height        # 4096
scale = preset.terrain.scale          # 150.0
octaves = preset.terrain.octaves      # 10
seed = preset.terrain.seed            # 42

# Paramètres végétation
density = preset.vegetation.density   # 0.4
min_spacing = preset.vegetation.min_spacing  # 4.0
```

---

## 🌲 **GÉNÉRATION RÉALISTE DE SAPINS**

Mountain Studio intègre un système complet de végétation écologique!

### Espèces disponibles:

1. **Pin (Pine)** - Pinus
   - Altitude: 20-80%
   - Hauteur: ~25m
   - Espacement: 5m
   - Description: "tall pine tree, coniferous, needle foliage, brown bark"

2. **Épicéa (Spruce)** - Picea
   - Altitude: 30-85%
   - Hauteur: ~30m
   - Espacement: 5.5m
   - Description: "tall spruce tree, conical shape, dense dark green foliage"

3. **Sapin (Fir)** - Abies
   - Altitude: 40-90%
   - Hauteur: ~28m
   - Espacement: 5.2m
   - Description: "tall fir tree, symmetrical, upward branches"

4. **Mélèze (Larch)** - Larix
   - Altitude: 50-95%
   - Hauteur: ~35m
   - Espacement: 6m
   - Description: "deciduous conifer, light green needles in summer"

5. **Autres**: Oak, Birch, Aspen, Willow...

### Paramètres de placement:

```python
from core.vegetation.vegetation_placer import VegetationPlacer

# Créer le placer
placer = VegetationPlacer(
    width=2048,
    height=2048,
    heightmap=terrain,
    biome_map=biomes
)

# Placer végétation avec clustering réaliste
trees = placer.place_vegetation(
    density=0.6,              # 60% coverage
    min_spacing=4.0,          # 4 mètres min entre arbres
    use_clustering=True,      # Groupes naturels
    cluster_size=8,           # 8 arbres par groupe
    cluster_radius=15.0,      # Rayon 15m
    seed=42                   # Reproductible
)

# Chaque arbre contient:
# - Position (x, y)
# - Élévation (altitude)
# - Espèce (pine, spruce, fir, etc.)
# - Échelle (variation 0.8-1.2)
# - Rotation (0-360°)
# - Âge (0-1, affecte apparence)
# - Santé (0-1)
```

### Algorithmes utilisés:

1. **Poisson Disc Sampling**: Distribution uniforme mais naturelle
2. **Clustering**: Groupes réalistes comme dans la nature
3. **Règles écologiques**: Altitude, pente, orientation, moisture
4. **Compétition**: Espacement minimum pour éviter superposition

### Export végétation:

```python
# Export JSON (pour tous moteurs)
import json

trees_data = [
    {
        'x': tree.x,
        'y': tree.y,
        'elevation': tree.elevation,
        'species': tree.species,
        'scale': tree.scale,
        'rotation': tree.rotation
    }
    for tree in trees
]

with open('vegetation_instances.json', 'w') as f:
    json.dump(trees_data, f, indent=2)
```

**Format Unreal Engine:**
```json
{
  "instances": [
    {
      "asset": "/Game/Trees/Pine_01",
      "transform": {
        "translation": [x, y, elevation],
        "rotation": [0, 0, rotation],
        "scale": [scale, scale, scale]
      }
    }
  ]
}
```

**Format Unity:**
```json
{
  "treeInstances": [
    {
      "prototypeIndex": 0,
      "position": {"x": x, "y": elevation, "z": y},
      "widthScale": scale,
      "heightScale": scale,
      "rotation": rotation,
      "color": {"r": 1, "g": 1, "b": 1, "a": 1},
      "lightmapColor": {"r": 1, "g": 1, "b": 1, "a": 1}
    }
  ]
}
```

---

## 🎯 **RECAPITULATIF: Workflow Optimal**

### Pour VFX Production (4K, qualité cinéma):

```bash
# 1. Setup ComfyUI (une seule fois)
python3 comfyui_auto_setup.py

# 2. Lancer ComfyUI
cd ComfyUI && python main.py &

# 3. Lancer Mountain Studio
python3 mountain_studio_ultimate_v2.py

# 4. Dans l'application:
#    - Charger preset: "VFX Epic Mountain"
#    - Générer terrain (4096x4096, ~2 min)
#    - Générer végétation (density=0.4)
#    - Générer PBR maps (2048x2048)
#    - Générer HDRI (4K, sunset)
#    - Générer AI textures (ComfyUI)
#    - Export Autodesk Flame pipeline

# Total: ~10-15 minutes pour package complet VFX!
```

### Pour Game Dev (Unreal/Unity, optimisé):

```bash
# 1. Lancer Mountain Studio
python3 mountain_studio_ultimate_v2.py

# 2. Dans l'application:
#    - Charger preset: "Game: Unreal Engine Landscape"
#    - Générer terrain (2048x2048, ~30 sec)
#    - Générer végétation (density=0.6, instances)
#    - Générer PBR maps (2048x2048)
#    - Export complet (OBJ + maps + vegetation JSON)

# 3. Import dans Unreal:
#    - Heightmap: Import as Landscape
#    - PBR maps: Create Landscape Material
#    - Vegetation: Use Foliage Tool with JSON positions

# Total: ~5 minutes pour assets game-ready!
```

### Pour Tests Rapides:

```bash
# 1. Lancer Mountain Studio
python3 mountain_studio_ultimate_v2.py

# 2. Dans l'application:
#    - Charger preset: "Test: Quick Preview"
#    - Générer terrain (512x512, ~5 sec)
#    - Ajuster lighting (Tab: Lighting)
#    - Export PNG 16-bit

# Total: ~10 secondes pour preview!
```

---

## 🐛 **Troubleshooting**

### ComfyUI ne se connecte pas

**Symptômes:**
- Status: "❌ Not connected"
- Mountain Studio ne peut pas générer de textures AI

**Solutions:**
1. Vérifier que ComfyUI tourne: `curl http://127.0.0.1:8188/system_stats`
2. Voir les logs ComfyUI pour erreurs
3. Redémarrer ComfyUI: `python main.py`
4. Vérifier firewall ne bloque pas port 8188

### Modèles manquants après setup

**Symptômes:**
- "sd_xl_base_1.0.safetensors not in []"

**Solutions:**
1. Vérifier chemin: `ls ComfyUI/models/checkpoints/`
2. Téléchargement manuel si nécessaire:
   ```bash
   cd ComfyUI/models/checkpoints/
   wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
   ```
3. Re-lancer setup: `python3 comfyui_auto_setup.py`

### Custom nodes ne marchent pas

**Symptômes:**
- "ImageSegmentation does not exist"
- Autres nodes manquants

**Solutions:**
1. Vérifier installation: `ls ComfyUI/custom_nodes/`
2. Installation manuelle:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
   cd comfyui_controlnet_aux
   pip install -r requirements.txt
   ```
3. Redémarrer ComfyUI

### Végétation ne s'affiche pas

**Symptômes:**
- Aucun arbre placé
- Erreur lors du placement

**Solutions:**
1. Vérifier densité n'est pas 0: `density > 0`
2. Vérifier biome map est valide
3. Augmenter max_attempts: `max_attempts=50`
4. Vérifier altitude range compatible avec espèces

### Erreurs d'export

**Symptômes:**
- Export échoue
- Fichiers incomplets

**Solutions:**
1. Vérifier permissions dossier output
2. Vérifier espace disque disponible
3. Vérifier PIL/Pillow installé: `pip install pillow`
4. Pour EXR: `pip install OpenEXR`

---

## 📚 **Ressources**

### Documentation
- Mountain Studio v2 README: `MOUNTAIN_STUDIO_V2_README.md`
- Config presets: `config/professional_presets.py`
- Végétation: `core/vegetation/`

### Communauté
- ComfyUI GitHub: https://github.com/comfyanonymous/ComfyUI
- SDXL Models: https://huggingface.co/stabilityai

### Support
- Report bugs: [GitHub Issues](lien vers votre repo)
- Questions: Documentation complète

---

**🏔️ Mountain Studio ULTIMATE v2.0** - Professional terrain generation made easy!

**Generate. Visualize. Populate. Export. Create.**
