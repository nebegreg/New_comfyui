# Guide ComfyUI pour Mountain Studio
## Configuration et Utilisation des Textures AI

Ce guide explique **comment configurer ComfyUI** pour générer des textures photoréalistes avec Mountain Studio.

---

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Installation ComfyUI](#installation-comfyui)
3. [Modèles Requis](#modèles-requis)
4. [Custom Nodes Recommandés](#custom-nodes-recommandés)
5. [Workflow de Base](#workflow-de-base)
6. [Workflow Avancé (PBR Complet)](#workflow-avancé-pbr-complet)
7. [Intégration avec Mountain Studio](#intégration-avec-mountain-studio)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

Mountain Studio peut utiliser ComfyUI pour générer des **textures photoréalistes** avec l'IA au lieu de textures procédurales.

**Avantages**:
- ✅ Textures ultra-réalistes (photogrammetry quality)
- ✅ Styles variés (granite, limestone, moss, etc.)
- ✅ Détails fins impossibles en procédural

**Inconvénients**:
- ⚠️ Nécessite GPU (NVIDIA recommandé)
- ⚠️ ~10-30 secondes par texture (dépend du GPU)
- ⚠️ Setup initial requis

---

## 🚀 Installation ComfyUI

### Méthode 1: Installation Portable (Recommandé Windows)

```bash
# Télécharger depuis https://github.com/comfyanonymous/ComfyUI/releases
# Version portable avec tout inclus

# Extraire et lancer
cd ComfyUI_windows_portable
run_nvidia_gpu.bat  # ou run_cpu.bat si pas de GPU NVIDIA
```

### Méthode 2: Installation depuis Source (Linux/Mac)

```bash
# Cloner le repo
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Lancer
python main.py
```

### Vérifier que ComfyUI Tourne

Une fois lancé, ouvrez votre navigateur:
```
http://127.0.0.1:8188
```

Vous devriez voir l'interface ComfyUI.

---

## 📦 Modèles Requis

ComfyUI nécessite des **modèles de diffusion**. Pour des textures photoréalistes, utilisez **SDXL** ou **SD 1.5**.

### SDXL (Recommandé pour qualité maximale)

**Télécharger**:
- **sd_xl_base_1.0.safetensors** (~6.5 GB)
  - Lien: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main

**Installation**:
```bash
# Placer dans ComfyUI/models/checkpoints/
cp sd_xl_base_1.0.safetensors ComfyUI/models/checkpoints/
```

### SD 1.5 (Alternative plus légère)

**Télécharger**:
- **v1-5-pruned-emaonly.safetensors** (~4 GB)
  - Lien: https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main

**Installation**:
```bash
cp v1-5-pruned-emaonly.safetensors ComfyUI/models/checkpoints/
```

### Modèles Spécialisés (Optionnel)

Pour encore plus de réalisme:

- **Realistic Vision** (portrait/réalisme)
  - https://civitai.com/models/4201/realistic-vision-v60-b1

- **DreamShaper** (polyvalent)
  - https://civitai.com/models/4384/dreamshaper

---

## 🔧 Custom Nodes Recommandés

Les **custom nodes** ajoutent des fonctionnalités à ComfyUI.

### Installation avec ComfyUI Manager

1. **Installer ComfyUI Manager**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

2. **Redémarrer ComfyUI**

3. **Dans l'interface**, cliquer sur "Manager" > "Install Custom Nodes"

### Nodes Recommandés pour Textures

| Node | Usage | Installation |
|------|-------|--------------|
| **ComfyUI-PBRify** | Génère Normal/Roughness/AO depuis diffuse | Via Manager |
| **ControlNet** | Guide génération avec heightmap | Via Manager |
| **WAS Node Suite** | Utilities (resize, blend, etc.) | Via Manager |
| **Image Saver** | Export avancé | Inclus |

---

## 📝 Workflow de Base

Voici un **workflow simple** pour générer une texture de terrain.

### Workflow JSON (À copier-coller dans ComfyUI)

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "ultra realistic alpine mountain rock texture, granite stone, high detail, 8k photogrammetry scan, seamless tileable, pbr material",
      "clip": ["1", 1]
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "blurry, low quality, cartoon, painted, artificial, tiling artifacts, watermark",
      "clip": ["1", 1]
    }
  },
  "4": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    }
  },
  "5": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 30,
      "cfg": 7.5,
      "sampler_name": "euler_a",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["4", 0]
    }
  },
  "6": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["5", 0],
      "vae": ["1", 2]
    }
  },
  "7": {
    "class_type": "SaveImage",
    "inputs": {
      "filename_prefix": "terrain_texture",
      "images": ["6", 0]
    }
  }
}
```

### Comment Utiliser

1. **Copier** le JSON ci-dessus
2. Dans ComfyUI, **Load** > **Paste Workflow**
3. **Queue Prompt** (bouton en bas à droite)
4. Attendre ~10-30 secondes
5. Image sauvegardée dans `ComfyUI/output/`

### Personnaliser le Prompt

Modifiez le node `2` (CLIPTextEncode positif):

**Pour différents matériaux**:
```
Rock/Granite:
"ultra realistic alpine granite rock texture, weathered stone, lichen patches, high detail, 8k scan, seamless, pbr"

Grass:
"photorealistic alpine grass texture, short mountain grass, moss, soil patches, 4k scan, seamless, pbr"

Snow:
"ultra realistic fresh snow texture, alpine snow, subtle footprints, crystals, 8k macro, seamless, pbr"

Sand:
"photorealistic mountain sand texture, fine grain, pebbles, natural weathering, 4k scan, seamless, pbr"
```

---

## 🎨 Workflow Avancé (PBR Complet)

Pour générer **toutes les maps PBR** (Diffuse, Normal, Roughness, AO) en une seule passe.

### Avec ComfyUI-PBRify (Recommandé)

**Installation**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-PBRify.git  # Lien hypothétique
```

**Workflow**:
1. Générer diffuse (comme ci-dessus)
2. Passer dans **PBRify node**
3. Obtenir:
   - Normal map
   - Roughness map
   - AO map
   - Height map

**Node Setup**:
```
SaveImage (diffuse)
    ↓
PBRify
    ↓
├─ SaveImage (normal)
├─ SaveImage (roughness)
├─ SaveImage (ao)
└─ SaveImage (height)
```

### Avec ControlNet + Heightmap

Pour guider la génération avec votre heightmap:

1. **Charger heightmap** comme image de contrôle
2. **ControlNet Depth** pour guider la structure
3. Génération respecte la topologie du terrain

**Workflow**:
```
LoadImage (heightmap)
    ↓
ControlNet Preprocessor (depth)
    ↓
Apply ControlNet
    ↓
KSampler (avec ControlNet)
    ↓
VAEDecode → SaveImage
```

---

## 🔗 Intégration avec Mountain Studio

### Vérifier la Connexion

Dans Mountain Studio:
1. Aller dans l'onglet **"Textures PBR"**
2. Vérifier le statut ComfyUI
3. Si ❌ rouge:
   - Vérifier que ComfyUI tourne sur `http://127.0.0.1:8188`
   - Tester dans le navigateur

### Utiliser ComfyUI dans Mountain Studio

1. **Générer un terrain** (onglet Terrain)
2. **Aller dans onglet "Textures PBR"**
3. **Activer** "Utiliser ComfyUI pour génération AI"
4. **Sélectionner matériau** (rock, grass, snow, etc.)
5. **Cliquer** "GÉNÉRER TEXTURES PBR"
6. Attendre 10-60 secondes (selon GPU)
7. ✅ Textures appliquées au viewer 3D!

### Que se passe-t-il ?

Mountain Studio:
1. Crée un **workflow ComfyUI** automatiquement
2. Envoie à `http://127.0.0.1:8188/prompt`
3. Attend la génération
4. Récupère l'image via `/view` endpoint
5. Génère les autres maps PBR (Normal, Roughness, AO) procéduralement
6. Applique au terrain 3D

### Fallback Automatique

Si ComfyUI **n'est pas disponible**:
- Mountain Studio bascule en **génération procédurale**
- Qualité moindre mais instantané
- Aucune action requise

---

## 🛠️ Troubleshooting

### ComfyUI ne se lance pas

**Erreur: "CUDA not available"**
```bash
# Réinstaller PyTorch avec CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Erreur: "Port 8188 already in use"**
```bash
# Tuer le processus existant
# Windows:
netstat -ano | findstr :8188
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8188 | xargs kill -9
```

### Génération très lente

**Solutions**:
1. **Réduire résolution**: 512x512 au lieu de 1024x1024
2. **Réduire steps**: 20 au lieu de 30
3. **Utiliser modèle plus léger**: SD 1.5 au lieu de SDXL
4. **Vérifier GPU utilisé**: Doit utiliser CUDA, pas CPU

**Vérifier GPU**:
```python
import torch
print(torch.cuda.is_available())  # Doit afficher True
print(torch.cuda.get_device_name(0))  # Nom du GPU
```

### Textures de mauvaise qualité

**Prompt trop vague**:
❌ `"mountain texture"`
✅ `"ultra realistic alpine granite rock texture, weathered, lichen, 8k photogrammetry, seamless, pbr material"`

**CFG trop bas/haut**:
- CFG = 5-8: Plus créatif, moins fidèle au prompt
- CFG = 10-15: Plus fidèle mais peut être "over-saturated"
- **Recommandé**: CFG = 7-7.5

**Steps trop faibles**:
- Minimum 20 steps
- **Recommandé**: 30 steps
- Au-delà de 50: peu de gain

### Mountain Studio ne détecte pas ComfyUI

**Vérifier que ComfyUI tourne**:
```bash
curl http://127.0.0.1:8188/system_stats
```

Doit retourner des stats JSON.

**Firewall/Antivirus**:
- Autoriser `python.exe` ou `main.py`
- Autoriser port `8188`

**Mauvaise adresse**:
- Mountain Studio utilise `127.0.0.1:8188` par défaut
- Si ComfyUI sur autre port/IP, modifier dans le code:
  ```python
  # core/ai/comfyui_integration.py, ligne ~38
  server_address: str = "127.0.0.1:8188"
  ```

---

## 🎓 Prompts Recommandés par Matériau

### Rock (Granite/Limestone)

**Positif**:
```
ultra realistic alpine granite rock texture, weathered stone surface,
lichen and moss patches, natural cracks and erosion,
high detail 8k photogrammetry scan, seamless tileable,
PBR material, physically accurate
```

**Négatif**:
```
blurry, low quality, cartoon, painted, artificial, smooth,
tiling artifacts, watermark, text, signature, unrealistic colors
```

### Grass (Alpine Meadow)

**Positif**:
```
photorealistic alpine mountain grass texture, short grass blades,
small wildflowers, moss patches, soil visible, natural variation,
4k macro photography, seamless tileable, PBR material
```

**Négatif**:
```
blurry, low res, plastic, artificial grass, uniform,
cartoon, painted, tiling visible, watermark
```

### Snow (Fresh Alpine Snow)

**Positif**:
```
ultra realistic fresh alpine snow texture, pristine white snow,
subtle surface details, ice crystals, natural shadows,
8k macro photography, seamless tileable, PBR material,
physically based rendering
```

**Négatif**:
```
blurry, dirty, footprints everywhere, yellow snow,
artificial, cartoon, painted, low quality, artifacts
```

### Dirt/Soil

**Positif**:
```
photorealistic mountain dirt texture, dark brown soil,
small pebbles and rocks, organic matter, natural variation,
4k photogrammetry scan, seamless tileable, PBR material
```

**Négatif**:
```
blurry, uniform, artificial, cartoon, painted,
too saturated, tiling artifacts, watermark
```

---

## 📚 Ressources Additionnelles

### Documentation

- **ComfyUI GitHub**: https://github.com/comfyanonymous/ComfyUI
- **ComfyUI Examples**: https://comfyanonymous.github.io/ComfyUI_examples/
- **Workflow Gallery**: https://openart.ai/workflows/comfyui

### Tutoriels Video

- **ComfyUI Basics**: Rechercher "ComfyUI tutorial" sur YouTube
- **PBR Workflow**: Rechercher "ComfyUI PBR textures"
- **ControlNet Guide**: Rechercher "ComfyUI ControlNet"

### Communauté

- **Reddit**: r/StableDiffusion, r/ComfyUI
- **Discord**: ComfyUI Official Discord
- **CivitAI**: Modèles et workflows communautaires

---

## ✅ Checklist de Setup

Avant d'utiliser ComfyUI avec Mountain Studio:

- [ ] ComfyUI installé et lancé sur port 8188
- [ ] Au moins un modèle téléchargé (SDXL ou SD 1.5)
- [ ] Workflow de base testé manuellement
- [ ] GPU CUDA fonctionnel (si disponible)
- [ ] Mountain Studio détecte la connexion (✅ vert)
- [ ] Test de génération réussi

**Une fois tout coché**, vous êtes prêt pour des textures AI photoréalistes!

---

## 🎯 Workflow Recommandé

1. **Générer terrain** dans Mountain Studio
2. **Générer textures PBR** avec ComfyUI (AI)
3. **Placer végétation** (arbres)
4. **Ajuster rendu 3D** (soleil, brouillard)
5. **Exporter tout** (heightmap + textures + végétation)

Résultat: **Terrain photoréaliste style Evian** prêt pour utilisation dans Blender, Unreal, Unity, etc.!

---

**Bon rendu! 🏔️✨**
