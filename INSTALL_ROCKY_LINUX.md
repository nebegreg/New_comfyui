# Installation sur Rocky Linux - Guide pour Autodesk Flame 2025.2.2

Ce guide couvre l'installation complète de Mountain Studio Pro sur Rocky Linux, optimisé pour l'utilisation avec Autodesk Flame 2025.2.2.

## Prérequis Système

- Rocky Linux 8 ou 9
- Python 3.9 ou 3.10 (compatible avec Flame 2025.2.2)
- 8 GB RAM minimum (16 GB recommandé)
- GPU NVIDIA optionnel (pour génération AI avec ComfyUI)

## 1. Installation des Dépendances Système

### Activer EPEL et PowerTools/CRB

```bash
# Installer EPEL
sudo dnf install -y epel-release

# Activer PowerTools (Rocky 8) ou CRB (Rocky 9)
sudo dnf config-manager --set-enabled powertools  # Rocky 8
sudo dnf config-manager --set-enabled crb         # Rocky 9
```

### Outils de Développement

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y python3-devel python3-pip git
```

### Bibliothèques OpenGL/Qt (pour PySide6 et PyOpenGL)

```bash
sudo dnf install -y mesa-libGL mesa-libGLU \
                    libxcb libxcb-xinerama \
                    xcb-util-wm xcb-util-image \
                    xcb-util-keysyms xcb-util-renderutil \
                    libX11-xcb
```

### Bibliothèques OpenCV

```bash
sudo dnf install -y libSM libXrender libXext
```

### FFmpeg (pour export vidéo)

```bash
# Installer RPM Fusion
sudo dnf install -y --nogpgcheck \
    https://download1.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm

# Installer FFmpeg
sudo dnf install -y ffmpeg
```

### CUDA (Optionnel - pour GPU NVIDIA)

```bash
# Suivre le guide officiel NVIDIA:
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=8

# Pour Rocky 8:
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install -y cuda

# Pour Rocky 9:
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y cuda
```

## 2. Installation Python

### Vérifier la Version Python de Flame

```bash
/opt/Autodesk/flame_*/python/bin/python --version
```

### Installer les Requirements

```bash
# Cloner le repository
git clone https://github.com/yourusername/New_comfyui.git
cd New_comfyui

# Installer les dépendances Python
pip3 install -r requirements.txt

# OU installer dans l'environnement Python de Flame:
/opt/Autodesk/flame_*/python/bin/pip install -r requirements.txt
```

### PyTorch - Version CPU ou CUDA

```bash
# Version CPU (pas de GPU)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Version CUDA 11.8 (si GPU NVIDIA)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Version CUDA 12.1 (si GPU NVIDIA récent)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Tests de Validation

### Test 1: Core Dependencies

```bash
python3 -c "import numpy, scipy, PIL; print('✓ Core libs OK')"
```

### Test 2: OpenCV

```bash
python3 -c "import cv2; print('✓ OpenCV OK')"
```

### Test 3: PyTorch

```bash
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} OK')"
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### Test 4: PySide6 (UI)

```bash
python3 -c "from PySide6.QtWidgets import QApplication; print('✓ PySide6 OK')"
```

### Test 5: PyOpenGL

```bash
python3 -c "import OpenGL.GL; print('✓ PyOpenGL OK')"
```

### Test 6: Export Flame

```bash
python3 test_flame_export.py
```

Si tout fonctionne, vous verrez:
```
✅ TEST RÉUSSI!
📁 Tous les fichiers sont dans: test_flame_export_output/
```

## 4. Configuration pour Autodesk Flame

### Chemins Importants

- **Scripts Flame**: `/opt/Autodesk/flame_*/python/`
- **Projets Flame**: `~/flame_projects/` (ou configuré dans Flame)
- **Export Mountain Studio**: Utiliser chemin absolu ou relatif au projet

### Format d'Export

Mountain Studio Pro exporte pour Flame dans les formats suivants:

✅ **OBJ + MTL** (implémenté)
- Mesh terrain en .obj
- Materials en .mtl
- Textures automatiquement liées

✅ **Textures** (implémentées)
- Diffuse: PNG 8-bit RGB
- Normal: PNG 8-bit RGB
- Depth/Height: PNG 16-bit Grayscale
- AO: PNG 8-bit Grayscale
- Roughness: PNG 8-bit Grayscale

🔧 **FBX** (TODO)
- Nécessite fbx-sdk ou pyfbx
- En développement

### Import dans Flame 2025.2.2

1. **Générer le terrain**:
   ```bash
   python3 test_flame_export.py
   ```

2. **Dans Flame**:
   - File > Import > Media
   - Sélectionner `test_flame_export_output/terrain.obj`
   - Les textures seront automatiquement chargées via le .mtl
   - Voir `README_FLAME.txt` pour détails

## 5. Résolution de Problèmes

### Erreur: "cannot import name 'xxx'"

```bash
# Réinstaller les requirements
pip3 install -r requirements.txt --force-reinstall
```

### Erreur: "libGL.so.1: cannot open shared object file"

```bash
# Installer Mesa libraries
sudo dnf install -y mesa-libGL mesa-libGLU
```

### Erreur: "Qt platform plugin 'xcb' not found"

```bash
# Installer xcb libraries
sudo dnf install -y libxcb libxcb-xinerama xcb-util-wm xcb-util-image
```

### SELinux Bloque l'Exécution

```bash
# Désactiver temporairement
sudo setenforce 0

# Ou désactiver définitivement (déconseillé en production)
sudo sed -i 's/SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config
```

### Performance Lente

```bash
# Vérifier que vous utilisez EXT4 (préféré par Flame)
df -Th

# Désactiver swap si RAM > 16GB
sudo swapoff -a

# Augmenter file descriptors
ulimit -n 65536
```

## 6. Optimisations Rocky Linux

### Pour Workstations VFX

```bash
# Installer TuneD pour profils de performance
sudo dnf install -y tuned
sudo systemctl enable --now tuned

# Utiliser le profil latency-performance
sudo tuned-adm profile latency-performance

# Vérifier
sudo tuned-adm active
```

### Pour GPU NVIDIA

```bash
# Installer nvidia-smi
sudo dnf install -y nvidia-driver

# Vérifier GPU
nvidia-smi

# Définir mode performance
sudo nvidia-smi -pm 1
```

## 7. Utilisation avec ComfyUI (Optionnel)

Si vous utilisez ComfyUI pour génération de textures AI:

```bash
# Installer ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip3 install -r requirements.txt

# Lancer ComfyUI
python3 main.py

# Dans un autre terminal, lancer Mountain Studio
cd ../New_comfyui
python3 mountain_pro_ui.py
```

Mountain Studio détectera automatiquement ComfyUI si il tourne sur `127.0.0.1:8188`.

## 8. Support et Aide

### Tests Automatisés

```bash
# Test export Flame
python3 test_flame_export.py

# Test génération PBR
python3 -c "from core.ai.comfyui_integration import generate_terrain_pbr_auto; print('PBR OK')"
```

### Vérifier Logs

```bash
# Logs Mountain Studio
tail -f ~/.mountain_studio/logs/latest.log

# Logs ComfyUI
tail -f ComfyUI/comfyui.log
```

## 9. Compatibilité Testée

✅ Rocky Linux 8.8
✅ Rocky Linux 9.2
✅ Autodesk Flame 2025.2.2
✅ Python 3.9, 3.10
✅ CUDA 11.8, 12.1
✅ PyTorch 2.0+

## 10. Checklist Installation

- [ ] EPEL et PowerTools/CRB activés
- [ ] Development Tools installés
- [ ] Bibliothèques OpenGL/Qt installées
- [ ] FFmpeg installé
- [ ] Python requirements installés
- [ ] PyTorch (CPU ou CUDA) installé
- [ ] Tests de validation passés
- [ ] Test export Flame réussi
- [ ] Import test réussi dans Flame

---

**Pour questions ou problèmes**: Ouvrir une issue sur GitHub

**Documentation complète**: Voir README.md et PBR_TEXTURE_SYSTEM.md
