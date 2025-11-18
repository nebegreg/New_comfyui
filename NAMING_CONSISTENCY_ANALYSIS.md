# Analyse de Cohérence - Nommage et Architecture

## 1. Conventions de Nommage Actuelles

### Fichiers Python
- **Snake_case**: La majorité des fichiers (✓ cohérent)
  - `heightmap_generator.py`
  - `hydraulic_erosion.py`
  - `pbr_texture_generator.py`

### Classes
- **PascalCase**: (✓ cohérent)
  - `HeightmapGenerator`
  - `PBRTextureGenerator`
  - `ComfyUIClient`

### Fonctions et Méthodes
- **Snake_case**: (✓ cohérent)
  - `generate_terrain()`
  - `apply_erosion()`
  - `export_for_flame()`

### Variables
- **Snake_case**: (✓ cohérent)
  - `heightmap`
  - `erosion_strength`
  - `vertical_exaggeration`

## 2. Incohérences Détectées

### 2.1 Noms de Modules Mixtes

**Problème**: Certains fichiers à la racine ne suivent pas la structure `core/`

Fichiers concernés:
```
❌ /mountain_app.py          → Devrait être core/ui/ ou ui/
❌ /comfyui_integration.py   → Devrait être core/ai/
❌ /professional_exporter.py → Devrait être core/export/
```

**Solution Recommandée**: Déplacer vers l'architecture `core/`

### 2.2 Duplication de Fonctionnalité

**Problème**: Deux générateurs de heightmap

```python
/core/terrain/heightmap_generator.py      # V1 - Ancien
/core/terrain/heightmap_generator_v2.py   # V2 - Nouveau (ultra-realistic)
```

**Solution Recommandée**:
- Garder V2 comme principal
- Renommer V1 en `heightmap_generator_legacy.py`
- Ou fusionner les fonctionnalités

### 2.3 Noms Français vs Anglais

**Inconsistance Linguistique**:

Français (à éviter dans le code):
```python
# Dans mountain_pro_ui.py
def generer_terrain()  # ❌
```

Anglais (préféré):
```python
# Partout ailleurs
def generate_terrain()  # ✓
```

**Solution**: Tout en anglais pour cohérence internationale

## 3. Architecture Recommandée

### Structure Actuelle (Simplifiée)
```
New_comfyui/
├── core/                      # ✓ Bon
│   ├── terrain/
│   ├── ai/
│   ├── rendering/
│   ├── vegetation/
│   └── export/
├── ui/                        # ✓ Bon
│   └── widgets/
├── config/                    # ✓ Bon
├── mountain_app.py            # ❌ À déplacer
├── comfyui_integration.py     # ❌ À déplacer
└── professional_exporter.py   # ❌ À déplacer
```

### Structure Cible
```
New_comfyui/
├── core/                      # Core functionality
│   ├── terrain/
│   │   ├── generators/        # ← NOUVEAU: générateurs séparés
│   │   │   ├── __init__.py
│   │   │   ├── spectral.py
│   │   │   ├── ridged.py
│   │   │   └── combined.py
│   │   ├── erosion/           # ← NOUVEAU: érosions séparées
│   │   │   ├── __init__.py
│   │   │   ├── hydraulic.py
│   │   │   ├── thermal.py
│   │   │   ├── stream_power.py
│   │   │   └── glacial.py
│   │   └── advanced_algorithms.py
│   ├── ai/
│   │   ├── comfyui_client.py
│   │   ├── comfyui_installer.py
│   │   ├── comfyui_workflows.py
│   │   └── comfyui_integration.py
│   ├── rendering/
│   ├── vegetation/
│   └── export/
│       └── professional_exporter.py  # ← Déplacé
├── ui/
│   ├── main_window.py         # ← NOUVEAU: GUI principal
│   └── widgets/
│       ├── terrain_preview_3d.py
│       ├── comfyui_installer_widget.py
│       └── ...
├── config/
├── tests/                     # ← NOUVEAU: tests organisés
│   ├── test_terrain.py
│   ├── test_erosion.py
│   └── test_export.py
└── main.py                    # ← Point d'entrée principal
```

## 4. Plan de Migration

### Phase 1: Réorganisation Fichiers (Priorité: Haute)
```bash
# Déplacer vers core/
mv comfyui_integration.py core/ai/  # Déjà fait
mv professional_exporter.py core/export/  # Déjà fait

# Renommer mountain_app.py
mv mountain_app.py ui/mountain_app_legacy.py

# Créer nouveau point d'entrée
# main.py → utilise mountain_pro_ui.py
```

### Phase 2: Consolidation Générateurs (Priorité: Moyenne)
```python
# core/terrain/generators/__init__.py

from .spectral import spectral_synthesis
from .ridged import ridged_multifractal
from .combined import TerrainGenerator  # Classe unifiée

class TerrainGenerator:
    """
    Unified terrain generator with all algorithms

    Replaces:
    - HeightmapGenerator (v1)
    - HeightmapGeneratorV2 (v2)

    Supports:
    - Spectral synthesis
    - Ridged multifractal
    - Hybrid
    - Stream power erosion
    - Glacial erosion
    """

    def generate(self, algorithm='ultra_realistic', **params):
        """Generate with specified algorithm"""
        if algorithm == 'spectral':
            return self._spectral(**params)
        elif algorithm == 'ridged':
            return self._ridged(**params)
        # ...
```

### Phase 3: Nettoyage UI (Priorité: Moyenne)
```python
# Fusionner:
# - mountain_app.py (ancien)
# - mountain_pro_ui.py (nouveau)
# → ui/main_window.py (final)
```

### Phase 4: Tests (Priorité: Haute)
```python
# Créer tests pour tous les nouveaux modules
# tests/test_advanced_algorithms.py
# tests/test_comfyui_installer.py
# tests/test_preview_3d.py
```

## 5. Conventions de Nommage - Standards

### 5.1 Fichiers et Modules
```
snake_case_with_underscores.py
```

### 5.2 Classes
```python
class PascalCaseClassName:
    """DocString"""
```

### 5.3 Fonctions et Méthodes
```python
def snake_case_function_name():
    """DocString"""
```

### 5.4 Constantes
```python
CONSTANT_NAME_UPPERCASE = 42
```

### 5.5 Variables Privées
```python
class MyClass:
    def __init__(self):
        self._private_var = 0      # Convention: private
        self.__really_private = 0  # Name mangling
```

## 6. Nomenclature Spécifique au Projet

### Terrain
- **Heightmap** (pas "elevation map" ou "height map")
- **Normal map** (pas "normalmap")
- **Splatmap** (pas "splat map")
- **PBR textures** (pas "textures PBR")

### Erosion
- **Hydraulic erosion** (pas "water erosion")
- **Thermal erosion** (pas "slope erosion")
- **Stream power** (pas "river erosion")
- **Glacial erosion** (pas "ice erosion")

### Export
- **Autodesk Flame export** (pas "Flame export")
- **OBJ/MTL format** (pas "Wavefront")
- **FBX format** (pas "Filmbox")

### AI/ComfyUI
- **ComfyUI client** (pas "Comfy client")
- **Custom nodes** (pas "plugins")
- **Checkpoint** (pas "model file")
- **Workflow** (pas "pipeline")

## 7. TODO: Renommages Nécessaires

### Priorité Haute (Faire maintenant)
- [ ] Créer `ui/main_window.py` comme point d'entrée GUI
- [ ] Tester `advanced_algorithms.py`
- [ ] Documenter nouveaux widgets

### Priorité Moyenne (Prochaine session)
- [ ] Fusionner HeightmapGenerator V1 et V2
- [ ] Créer `core/terrain/generators/` module
- [ ] Créer `core/terrain/erosion/` module
- [ ] Organiser tests dans `tests/`

### Priorité Basse (Future)
- [ ] Traduire commentaires français → anglais
- [ ] Standardiser tous les docstrings (format Google)
- [ ] Créer diagrammes d'architecture

## 8. Checklist de Cohérence

Avant chaque commit, vérifier:
- [ ] Noms de fichiers en snake_case
- [ ] Classes en PascalCase
- [ ] Fonctions en snake_case
- [ ] Pas de noms français dans le code
- [ ] DocStrings en anglais
- [ ] Imports organisés (stdlib, third-party, local)
- [ ] Type hints ajoutés
- [ ] Tests passent

## 9. Exemples de Bon Nommage

```python
# Fichier: core/terrain/generators/spectral.py

"""
Spectral Synthesis Terrain Generator

FFT-based terrain generation using power-law spectrum.
Based on Fournier et al. (1982).
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectralTerrainGenerator:
    """
    Generate terrain using spectral synthesis

    Attributes:
        beta: Power spectrum exponent
        size: Output resolution
    """

    def __init__(self, size: int, beta: float = 2.0):
        """
        Initialize spectral generator

        Args:
            size: Output resolution (power of 2 recommended)
            beta: Spectral exponent (2.0 = natural terrain)
        """
        self.size = size
        self.beta = beta
        logger.info(f"SpectralTerrainGenerator initialized: size={size}, beta={beta}")

    def generate(
        self,
        seed: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate heightmap

        Args:
            seed: Random seed for reproducibility
            normalize: Whether to normalize output to [0, 1]

        Returns:
            Heightmap array of shape (size, size)
        """
        # Implementation...
        pass
```

## 10. Résumé

**État Actuel**: ⚠️ Partiellement cohérent
- Architecture core/ bien organisée
- Mais quelques fichiers legacy à la racine
- Duplication HeightmapGenerator v1/v2

**Actions Prioritaires**:
1. Tester nouveaux modules (advanced_algorithms, widgets)
2. Créer point d'entrée unifié (`main.py`)
3. Documenter nouvelles fonctionnalités
4. Commit avec nommage cohérent

**Objectif Final**: 🎯 Projet 100% cohérent
- Structure modulaire claire
- Nommage uniforme (anglais, snake_case/PascalCase)
- Tests complets
- Documentation à jour
