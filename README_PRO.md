# 🏔️ Mountain Studio Pro - Outil Professionnel pour Graphistes

**Application professionnelle de génération de montagnes 3D ultra-réalistes avec IA**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![PySide6](https://img.shields.io/badge/UI-PySide6-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🎯 Nouveautés Version 2.0 PRO

### ✨ Interface PySide6 Professionnelle
- **Interface graphique moderne** avec dark theme
- **Vue 3D interactive** en temps réel (PyQt OpenGL)
- **Preview instantané** de toutes les maps
- **Workflow optimisé** pour graphistes

### 🗻 Génération 3D Réelle de Terrain
- **Heightmap** avec algorithmes Perlin/Simplex noise multi-octaves
- **Normal maps** haute résolution
- **Depth maps** (Z-depth pour rendu)
- **Ambient Occlusion** maps
- **Roughness maps** pour PBR
- **Simulation d'érosion** réaliste

### 🎬 Cohérence Temporelle Vidéo (SOLUTION AU PROBLÈME!)
Le système de **cohérence temporelle** résout le problème des montagnes qui changent à chaque frame:

- **ControlNet** pour guidance structurelle constante
- **Img2Img** avec faible strength pour cohérence frame-à-frame
- **Optical flow warping** pour interpolation fluide
- **AnimateDiff integration** pour stabilité temporelle
- **Même heightmap 3D** = même montagne, angles différents!

### 💾 Export Professionnel
- **Format EXR 32-bit** pour heightmaps (displacement)
- **Multi-channel export** (toutes les maps en un clic)
- **Presets pour Blender, Unreal, Unity, Substance**
- **Export OBJ** avec mesh 3D complet
- **Scripts auto-import** pour Blender

### 🔧 ComfyUI Amélioré
- **Fix erreur 400** : détection automatique des modèles disponibles
- **Diagnostic intelligent** des erreurs
- **Test de connexion** avant génération
- **Gestion robuste** des timeouts et erreurs réseau

---

## 📦 Installation

### Prérequis
- Python 3.8+
- GPU NVIDIA avec CUDA (recommandé 8GB+ VRAM)
- (Optionnel) ComfyUI installé

### Installation Rapide

```bash
# Cloner le dépôt
git clone https://github.com/nebegreg/New_comfyui.git
cd New_comfyui

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Lancer l'application PRO
python mountain_pro_ui.py
```

---

## 🚀 Guide d'Utilisation

### 1. Interface Principale

L'interface est divisée en 3 panels:

#### **Panel Gauche - Contrôles**
4 onglets principaux:
- **🗻 Terrain** : Génération heightmap et maps 3D
- **🎨 Texture AI** : Texturisation avec Stable Diffusion
- **🎥 Caméra** : Contrôle caméra et génération vidéo
- **💾 Export** : Export professionnel multi-format

#### **Panel Central - Vue 3D**
- Vue 3D interactive du terrain généré
- Rotation, zoom libre
- Visualisation wireframe/solid
- Preview en temps réel

#### **Panel Droit - Preview Maps**
- Tabs pour chaque map (Heightmap, Normal, Depth, Texture)
- Preview 350x350px
- Mise à jour automatique

### 2. Génération de Terrain 3D

#### Paramètres Terrain

**Type de Montagne:**
- **Alpine** : Pics aigus, montagnes alpines classiques
- **Volcanic** : Pic central prononcé, forme conique
- **Rolling** : Collines douces, terrain vallonné
- **Massive** : Massifs imposants, larges formations
- **Rocky** : Terrain rocheux irrégulier

**Résolution:** 512, 1024, 2048, ou 4096px
- 2048px recommandé pour qualité/performance
- 4096px pour production finale

**Scale (10-200):** Échelle du terrain
- Valeurs basses = plus de détails
- Valeurs hautes = formes plus larges

**Octaves (1-12):** Nombre de niveaux de détail
- Plus d'octaves = plus de complexité
- 6-8 recommandé

**Persistence (0.1-0.9):** Amplitude des détails
- 0.5 = équilibré
- Plus haut = plus chaotique

**Lacunarity (1.0-4.0):** Fréquence des détails
- 2.0 standard
- Plus haut = variations plus rapides

**Normal Map Strength (0.5-3.0):** Force des normales
- 1.0 = normal
- >1.5 pour relief très prononcé

**Seed:** Pour reproductibilité
- Même seed = même terrain

#### Processus de Génération

1. Cliquez sur **🗻 Générer Terrain 3D**
2. Attendez la progression (10-100%)
   - 10%: Génération heightmap
   - 40%: Normal map
   - 60%: Depth map
   - 80%: AO et Roughness
   - 100%: Terminé!
3. Les maps s'affichent automatiquement dans les previews
4. La vue 3D se met à jour

### 3. Texture AI (Résout le problème de cohérence!)

#### Configuration Backend

**Option 1: Stable Diffusion XL** (Recommandé pour débuter)
```
1. Sélectionner "Stable Diffusion XL"
2. Cliquer "🚀 Initialiser Backend"
3. Attendre le chargement du modèle (~5-10 min première fois)
```

**Option 2: ComfyUI** (Pour utilisateurs avancés)
```
1. Lancer ComfyUI séparément
2. Noter l'adresse (ex: 127.0.0.1:8188)
3. Entrer l'adresse dans l'interface
4. Cliquer "🚀 Initialiser Backend"
5. Vérifier la connexion (liste des modèles)
```

#### Génération de Texture

**Auto-génération de Prompt:**
- Cliquez sur **✨ Auto-générer Prompt**
- Le système crée un prompt optimisé basé sur vos paramètres terrain

**Prompt Manuel:**
```
Exemple de prompt pro:
"photorealistic mountain landscape, detailed rock texture,
alpine terrain, high resolution, 8k, professional photography,
natural lighting, realistic material, PBR ready"
```

**Steps:** 20-100
- 30-40 pour tests rapides
- 50-80 pour production

**Detail Level:** 0-100
- Influence les tags de qualité dans le prompt

### 4. Génération Vidéo Cohérente (NOUVELLE FONCTIONNALITÉ!)

**Le Problème Résolu:**
Avant, chaque frame générait une montagne différente. Maintenant, grâce au système de cohérence temporelle:

1. **Même heightmap 3D** pour toutes les frames
2. **ControlNet** maintient la structure
3. **Img2Img faible strength** assure la cohérence
4. **Interpolation optical flow** fluidifie le mouvement

**Paramètres Vidéo:**

**Nombre de Frames:** 3-60
- 12 frames = ~0.5 sec à 24fps
- 24 frames = 1 sec
- Attention: chaque frame prend ~30 sec à générer

**Type de Mouvement:**
- **Orbit** : Rotation 360° autour de la montagne
- **Pan** : Panoramique horizontal
- **Zoom** : Zoom progressif
- **Flyover** : Survol cinématique avec variation hauteur
- **Static** : Test sans mouvement

**Strength (cohérence):** 0.1-0.5
- 0.15-0.25 recommandé
- Plus bas = plus cohérent mais moins de variation
- Plus haut = plus de variation mais risque d'incohérence

**Interpolation:**
- Activée par défaut
- Génère 1-3 frames supplémentaires entre chaque frame générée
- Utilise optical flow pour fluidité maximale

**Processus:**
```
1. Configure les paramètres vidéo
2. Assure-toi d'avoir un terrain généré (heightmap)
3. Clique "🎬 Générer Vidéo Cohérente"
4. Attends (peut prendre 5-20 min selon nombre de frames)
5. La vidéo est sauvegardée automatiquement en MP4
```

### 5. Export Professionnel

#### Export Toutes les Maps

**Formats Disponibles:**
- ✅ **Heightmap** : EXR 32-bit float (ou TIFF 32-bit, ou PNG 16-bit)
- ✅ **Normal Map** : PNG RGB
- ✅ **Depth Map** : PNG grayscale
- ✅ **Ambient Occlusion** : PNG grayscale
- ✅ **Roughness Map** : PNG grayscale
- ✅ **Texture AI** : PNG RGB (si générée)

**Workflow:**
```
1. Coche les maps que tu veux exporter
2. Choisis le format (PNG / EXR / TIFF / Tous)
3. Clique "💾 Exporter Toutes les Maps"
4. Choisis le dossier de destination
5. Toutes les maps sont exportées avec préfixes cohérents
```

**Nomenclature:**
```
mountain_pro_heightmap.exr
mountain_pro_heightmap.png
mountain_pro_normal.png
mountain_pro_depth.png
mountain_pro_ao.png
mountain_pro_roughness.png
```

#### Export Mesh 3D (.OBJ)

```
1. Clique "📐 Exporter Mesh 3D (.OBJ)"
2. Choisis le nom et emplacement
3. Le mesh est exporté avec vertices et faces
4. Import direct dans Blender, Maya, 3ds Max, etc.
```

**Spécifications Mesh:**
- Vertices: X, Y, Z (Z = height)
- Faces: Triangles
- Résolution: basée sur la résolution du heightmap
- Format: OBJ standard ASCII

#### Export pour Logiciels Spécifiques

**Pour Blender:**
```python
# Dans l'export professionnel (code)
from professional_exporter import ProfessionalExporter

exporter = ProfessionalExporter()
exporter.export_to_blender(terrain_gen, "output/blender", "mountain")

# Génère:
# - mountain_displacement.exr (pour modifier Displace)
# - mountain_normal.png
# - mountain_mesh.obj
# - mountain_blender_import.py (script auto-setup)
```

**Pour Unreal Engine:**
```python
exporter.export_to_unreal(maps_dict, "output/unreal", "mountain")

# Génère:
# - mountain_Heightmap.png (16-bit)
# - mountain_Normal.png (DirectX format)
# - mountain_ORM.png (Occlusion-Roughness-Metallic packed)
```

**Pour Unity:**
```python
# Heightmap en RAW ou PNG 16-bit
# Normal maps en format Unity (OpenGL)
# Textures PBR standard
```

**Pour Substance Painter/Designer:**
```python
exporter.export_to_substance(maps_dict, "output/substance", "mountain")

# Format TIFF 16-bit préféré par Substance
# Nomenclature automatique correcte
```

---

## 🔧 Workflows Professionnels

### Workflow 1: Terrain pour Jeu Vidéo

```
1. Générer terrain (résolution 2048, type Alpine)
2. Exporter heightmap + normal map
3. Importer dans Unreal/Unity
4. Utiliser heightmap pour landscape
5. Appliquer normal map pour détails
6. Optionnel: Texturer avec Texture AI
```

### Workflow 2: Asset pour Film/VFX

```
1. Générer terrain haute résolution (4096)
2. Générer texture AI ultra-détaillée (80+ steps)
3. Exporter tout en EXR 32-bit
4. Importer dans Blender
5. Setup displacement + PBR materials
6. Render Cycles/EEVEE
```

### Workflow 3: Vidéo Cinématique

```
1. Générer terrain parfait (ajuster seed jusqu'à satisfaction)
2. Configurer mouvement caméra (Orbit ou Flyover)
3. Générer vidéo cohérente (12-24 frames)
4. Interpolation activée pour fluidité
5. Export vidéo MP4
6. Post-production si nécessaire
```

### Workflow 4: Texture Development

```
1. Générer terrain de base
2. Exporter heightmap
3. Importer dans Substance Painter
4. Texturer manuellement avec contrôle total
5. Export maps PBR
6. Réimport optionnel pour rendu AI
```

---

## 🎨 Exemples de Paramètres

### Montagnes Alpines Dramatiques
```
Type: Alpine
Résolution: 2048
Scale: 80
Octaves: 8
Persistence: 0.55
Lacunarity: 2.2
Normal Strength: 1.5
Seed: 1234
```

### Volcan Majestueux
```
Type: Volcanic
Résolution: 2048
Scale: 60
Octaves: 7
Persistence: 0.45
Lacunarity: 2.0
Normal Strength: 2.0
Seed: 5678
```

### Collines Douces
```
Type: Rolling
Résolution: 1024
Scale: 120
Octaves: 6
Persistence: 0.50
Lacunarity: 2.0
Normal Strength: 0.8
Seed: 9012
```

---

## 🐛 Troubleshooting

### Erreur 400 ComfyUI

**Causes:**
1. Checkpoint inexistant
2. ComfyUI pas lancé
3. Mauvaise adresse serveur
4. Workflow incompatible

**Solutions:**
```
1. Vérifier que ComfyUI est lancé (http://127.0.0.1:8188)
2. Tester la connexion dans l'interface
3. Vérifier la liste des modèles détectés
4. Utiliser le modèle par défaut proposé
```

L'interface affiche maintenant des messages détaillés:
```
❌ Erreur 400 - Bad Request
   Détails: {'error': 'checkpoint not found'}

   💡 Suggestion: Le checkpoint spécifié n'existe pas
      Checkpoints disponibles: ['model1.safetensors', 'model2.ckpt', ...]
```

### CUDA Out of Memory

**Solutions:**
```
1. Réduire résolution (2048 → 1024)
2. Réduire steps (50 → 30)
3. Fermer autres applications GPU
4. Utiliser CPU (plus lent)
```

### Vidéo Incohérente

**Problème:** Les montagnes changent entre frames

**Solutions:**
1. **Réduire Strength** (0.25 → 0.15)
2. **Activer ControlNet** guidance
3. **Vérifier** que la heightmap est bien utilisée
4. **Augmenter** steps pour meilleure qualité
5. **Réduire** nombre de frames si problème persiste

### Génération Lente

**Optimisations:**
```
1. Réduire résolution terrain (4096 → 2048)
2. Réduire steps AI (50 → 30)
3. Utiliser GPU au lieu de CPU
4. Fermer applications gourmandes
5. Pour vidéo: réduire nombre de frames
```

---

## 💡 Tips & Best Practices

### Génération Terrain

1. **Commencer avec résolution moyenne** (1024-2048) pour tests
2. **Expérimenter avec seeds** jusqu'à trouver la forme parfaite
3. **Sauvegarder le seed** des terrains réussis
4. **Ajuster octaves progressivement** (commencer à 6, augmenter si besoin)
5. **Normal strength >1.5** pour terrains très détaillés

### Texture AI

1. **Prompts détaillés** = meilleurs résultats
2. **Utiliser auto-generate** comme base, puis ajuster
3. **Steps 40-60** bon compromis qualité/vitesse
4. **Negative prompts** importants: "low quality, blurry, artificial"
5. **Seed cohérent** avec le terrain pour consistance

### Vidéo Cohérente

1. **TOUJOURS générer terrain d'abord**
2. **Strength 0.20-0.25** optimal pour cohérence
3. **12-16 frames** bon début (éviter 30+ pour premiers tests)
4. **Interpolation ON** pour fluidité
5. **Type Orbit** le plus spectaculaire
6. **Tester Static** avant longs rendus

### Export

1. **EXR pour displacement** (meilleure précision)
2. **PNG pour diffuse/color** maps
3. **Exporter contact sheet** pour validation rapide
4. **Nomenclature cohérente** importante pour pipeline
5. **Vérifier gamma/color space** avant import 3D

---

## 📊 Spécifications Techniques

### Formats Supportés

**Input:**
- Parameters (sliders, UI)
- Heightmap guidance (optional)

**Output:**
- PNG (8-bit, 16-bit)
- TIFF (16-bit, 32-bit float)
- EXR (32-bit float, multi-channel)
- OBJ (mesh 3D)
- MP4 (vidéo)

### Résolutions

- Terrain: 512×512 à 4096×4096
- Texture AI: 512×512 à 2048×2048
- Video: 1024×768 recommandé

### Performance

**Génération Terrain (2048×2048):**
- CPU i7: ~5-10 secondes
- Avec toutes maps: ~15-20 secondes

**Texture AI (1024×768, 40 steps):**
- GPU RTX 3060 (8GB): ~30-40 secondes
- GPU RTX 4090 (24GB): ~10-15 secondes

**Vidéo Cohérente (12 frames, interpolation):**
- Total frames générées: 12 × 2 (interpolation) = 24 frames
- Temps total: ~6-12 minutes (GPU moyen)
- Output: 1 seconde vidéo à 24fps

---

## 🔄 Comparaison v1.0 vs v2.0 PRO

| Fonctionnalité | v1.0 (Gradio) | v2.0 PRO (PySide6) |
|---|---|---|
| Interface | Web Gradio | Application native Qt |
| Vue 3D | ❌ | ✅ Temps réel |
| Génération Terrain | Prompts texte | Heightmap 3D réelle |
| Normal Maps | ❌ | ✅ Haute résolution |
| Depth Maps | ❌ | ✅ Z-depth précis |
| PBR Maps | ❌ | ✅ AO + Roughness |
| Cohérence Vidéo | ❌ Montagnes changent | ✅ Même montagne! |
| Export Pro | PNG basique | EXR/TIFF/OBJ/Multi |
| ComfyUI Errors | Peu d'info | Diagnostic détaillé |
| Presets Logiciels | ❌ | ✅ Blender/UE/Unity |
| Performance | Moyenne | Optimisée |
| Public | Amateurs | **Professionnels** |

---

## 🤝 Support & Contribution

### Bug Reports
Ouvrez une issue avec:
- Description du problème
- Étapes pour reproduire
- Logs (panel de droite)
- Specs GPU/CPU

### Feature Requests
- Expliquez le cas d'usage professionnel
- Référencez des exemples d'autres outils
- Priorité haute si demande récurrente

### Community
- Partagez vos créations!
- Tutoriels vidéo bienvenus
- Presets communautaires

---

## 📚 Ressources

### Tutoriels
- [Importer dans Blender](#) (à venir)
- [Setup Unreal Engine](#) (à venir)
- [Workflow Substance](#) (à venir)

### Documentation Externe
- [Stable Diffusion Docs](https://stable-diffusion-art.com/)
- [ComfyUI Wiki](https://github.com/comfyanonymous/ComfyUI/wiki)
- [PBR Texture Guide](https://marmoset.co/posts/pbr-texture-conversion/)

---

## 📝 License

MIT License - Libre utilisation commerciale et non-commerciale

---

## 🙏 Remerciements

- **Stable Diffusion** par Stability AI
- **ComfyUI** par comfyanonymous
- **PySide6** par Qt Company
- **PyQtGraph** pour visualisation 3D
- **ControlNet** pour cohérence temporelle
- **Communauté open-source AI**

---

**Mountain Studio Pro - L'outil professionnel pour graphistes qui veulent créer des montagnes ultra-réalistes avec contrôle total!**

🏔️✨ **Version 2.0 - Designed for Professionals** ✨🏔️
