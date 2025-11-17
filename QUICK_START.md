# 🚀 Guide de Démarrage Rapide - Mountain Studio Pro

## Installation en 3 étapes

### 1. Installer Python
- Python 3.8+ requis
- Télécharger depuis [python.org](https://python.org)

### 2. Cloner et Installer
```bash
git clone https://github.com/nebegreg/New_comfyui.git
cd New_comfyui
pip install -r requirements.txt
```

### 3. Lancer l'Application
```bash
# Linux/Mac
./start_pro.sh

# Windows
start_pro.bat

# Ou directement
python mountain_pro_ui.py
```

---

## Premier Terrain en 5 Minutes

### Étape 1: Lancer l'Application
```bash
python mountain_pro_ui.py
```

### Étape 2: Onglet "🗻 Terrain"
- Choisir type: **Alpine**
- Résolution: **2048**
- Garder les paramètres par défaut
- Seed: **42**

### Étape 3: Générer
- Cliquer **🗻 Générer Terrain 3D**
- Attendre 10-20 secondes
- ✓ Terrain apparaît en 3D!

### Étape 4: Explorer
- **Panel Central**: Vue 3D interactive
  - Clic gauche + drag = rotation
  - Molette = zoom
  - Clic droit + drag = pan

- **Panel Droit**: Preview des maps
  - Tab "Heightmap" = élévation
  - Tab "Normal" = détails surface
  - Tab "Depth" = profondeur

### Étape 5: Exporter
- Onglet **💾 Export**
- Cocher toutes les maps
- **💾 Exporter Toutes les Maps**
- Choisir dossier
- ✓ Terminé!

---

## Premier Rendu AI en 10 Minutes

### Prérequis
- GPU NVIDIA avec 8GB+ VRAM recommandé
- Ou patience si CPU (plus lent)

### Étape 1: Initialiser Backend
- Onglet **🎨 Texture AI**
- Backend: **Stable Diffusion XL**
- **🚀 Initialiser Backend**
- ⏳ Attendre chargement modèle (5-10 min première fois)

### Étape 2: Auto-Prompt
- **✨ Auto-générer Prompt**
- Le système crée un prompt optimisé

Ou écrire manuellement:
```
photorealistic alpine mountain landscape, detailed rocky texture,
snow-capped peaks, natural lighting, 8k, professional photography
```

### Étape 3: Paramètres
- Steps: **40** (bon compromis)
- Detail Level: **85**

### Étape 4: Générer
- **🎨 Générer Texture AI**
- ⏳ Attendre 30-60 secondes (GPU)
- ✓ Texture apparaît dans preview!

---

## Première Vidéo Cohérente en 15 Minutes

### ⚠️ Important
La vidéo nécessite:
1. Un terrain déjà généré (heightmap)
2. Backend AI initialisé
3. ~10-15 minutes pour 12 frames

### Étape 1: Terrain
Si pas déjà fait:
- Générer un terrain (voir ci-dessus)

### Étape 2: Configuration Vidéo
- Onglet **🎥 Caméra**
- Nombre de Frames: **12** (pour test)
- Type Mouvement: **Orbit**
- Strength: **0.25**
- ✅ Interpolation activée

### Étape 3: Générer
- **🎬 Générer Vidéo Cohérente**
- ⏳ Patience: ~10-15 minutes
- Progress indiqué dans status

### Étape 4: Résultat
- Vidéo sauvegardée en MP4
- ~0.5 secondes à 24fps
- **Même montagne** sous différents angles!

---

## Troubleshooting Express

### "Python not found"
```bash
# Installer Python 3.8+
# https://python.org/downloads
```

### "CUDA out of memory"
```
Solution 1: Réduire résolution (2048 → 1024)
Solution 2: Réduire steps (40 → 25)
Solution 3: Fermer autres apps GPU
Solution 4: Utiliser CPU (plus lent)
```

### "ComfyUI erreur 400"
```
1. Vérifier ComfyUI lancé (http://127.0.0.1:8188)
2. Tester connexion dans l'interface
3. Utiliser "Stable Diffusion XL" à la place
```

### "Application ne démarre pas"
```bash
# Vérifier dépendances
pip install -r requirements.txt

# Vérifier imports
python -c "import PySide6; print('OK')"

# Logs détaillés
python mountain_pro_ui.py 2>&1 | tee log.txt
```

### "Vue 3D ne s'affiche pas"
```
1. Vérifier OpenGL support
2. Mettre à jour drivers GPU
3. Essayer sans vue 3D (preview 2D fonctionne)
```

---

## Raccourcis Clavier (à venir)

- `Ctrl+G` : Générer Terrain
- `Ctrl+T` : Générer Texture
- `Ctrl+E` : Export Rapide
- `Ctrl+R` : Reset Vue 3D
- `F5` : Refresh Preview

---

## Workflows Recommandés

### Débutant: Premier Essai
```
1. Générer terrain (défaut)
2. Exporter PNG
3. Voir résultat
Total: 1 minute
```

### Intermédiaire: Terrain + Texture
```
1. Générer terrain custom
2. Initialiser SD
3. Auto-prompt + générer texture
4. Export multi-maps
Total: 10-15 minutes
```

### Avancé: Vidéo Production
```
1. Terrain optimisé (ajuster seed)
2. Texture AI haute qualité (80 steps)
3. Vidéo cohérente (24 frames)
4. Export EXR + OBJ
5. Import Blender
Total: 30-45 minutes
```

---

## Ressources

### Documentation
- **README_PRO.md** : Documentation complète
- **README.md** : Version originale Gradio

### Support
- GitHub Issues pour bugs
- Discussions pour questions

### Communauté
- Partagez vos créations!
- #MountainStudioPro

---

## Prochaines Étapes

Après avoir maîtrisé les bases:

1. **Expérimenter paramètres**
   - Différents types montagne
   - Jouer avec octaves/persistence
   - Trouver vos seeds préférés

2. **Apprendre prompts AI**
   - Lire guide Stable Diffusion
   - Tester différents styles
   - Créer vos presets

3. **Workflow professionnel**
   - Export vers Blender/Unreal
   - Pipeline production
   - Automatisation

4. **Contribuer**
   - Partager presets
   - Créer tutoriels
   - Suggérer features

---

**Amusez-vous bien! 🏔️✨**

Si vous avez des questions, consultez le README_PRO.md complet ou ouvrez une issue GitHub.
