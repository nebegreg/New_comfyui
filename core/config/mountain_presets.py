"""
Mountain Presets System - Professional Terrain Templates
========================================================

Presets prédéfinis pour générer rapidement des terrains iconiques:
- Evian Alps (montagnes immaculées style publicité)
- Three Peaks (3 sommets majestueux)
- Ski Slope (piste de ski poudreuse)
- Matterhorn (pic emblématique)
- Mont Blanc (plus haut sommet des Alpes)
- Dolomites (formations rocheuses spectaculaires)
- Scottish Highlands (paysages vallonnés)
- Grand Canyon (canyon profond)

Chaque preset inclut:
✅ Paramètres de génération terrain
✅ Configuration érosion
✅ Paramètres végétation
✅ Setup caméra
✅ Lighting/HDRI preset
✅ Matériaux PBR

Author: Mountain Studio Pro Team
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np


class PresetCategory(Enum):
    """Catégories de presets"""
    ALPS = "alps"
    PEAKS = "peaks"
    SKI = "ski"
    CANYON = "canyon"
    HIGHLANDS = "highlands"
    VOLCANIC = "volcanic"


@dataclass
class TerrainPreset:
    """Configuration complète d'un preset de terrain"""

    # Metadata
    name: str
    description: str
    category: PresetCategory
    difficulty: str  # "easy", "medium", "hard"

    # Terrain generation
    resolution: Tuple[int, int] = (512, 512)
    algorithm: str = "ultra_realistic"
    scale: float = 100.0
    octaves: int = 8
    ridge_influence: float = 0.4
    warp_strength: float = 0.3

    # Erosion
    hydraulic_iterations: int = 50
    thermal_iterations: int = 5
    erosion_rate: float = 0.3

    # Vegetation
    vegetation_enabled: bool = True
    vegetation_density: float = 0.5
    vegetation_spacing: float = 5.0
    use_clustering: bool = True
    cluster_size: int = 5

    # PBR Materials
    primary_material: str = "rock"
    secondary_material: Optional[str] = "snow"

    # Lighting
    sun_azimuth: float = 135.0
    sun_elevation: float = 45.0
    sun_intensity: float = 1.2
    ambient_strength: float = 0.4

    # Atmosphere
    fog_density: float = 0.015
    fog_enabled: bool = True
    atmosphere_enabled: bool = True

    # HDRI
    hdri_time: str = "midday"  # sunrise, morning, midday, afternoon, sunset, twilight, night

    # Camera
    camera_distance: float = 350.0
    camera_elevation: float = 25.0
    camera_azimuth: float = 45.0

    # Height scale
    height_scale: float = 50.0

    # Seed
    seed: Optional[int] = None

    # Tags
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MountainPresets:
    """
    Collection de presets professionnels

    Usage:
        presets = MountainPresets()
        evian_preset = presets.get_preset("evian_alps")

        # Appliquer à l'UI
        app.apply_preset(evian_preset)
    """

    def __init__(self):
        self._presets = {}
        self._initialize_presets()

    def _initialize_presets(self):
        """Initialize all presets"""

        # =====================================================================
        # EVIAN ALPS - Style publicité Evian
        # =====================================================================
        self._presets["evian_alps"] = TerrainPreset(
            name="Evian Alps",
            description="Montagnes alpines immaculées style publicité Evian. "
                        "Pics enneigés, pureté, lumière naturelle claire.",
            category=PresetCategory.ALPS,
            difficulty="easy",

            # Terrain
            resolution=(512, 512),
            algorithm="ultra_realistic",
            scale=120.0,
            octaves=10,
            ridge_influence=0.5,  # Pics bien définis
            warp_strength=0.25,   # Formes douces mais naturelles

            # Erosion légère (montagnes jeunes)
            hydraulic_iterations=30,
            thermal_iterations=3,
            erosion_rate=0.2,

            # Végétation alpine
            vegetation_enabled=True,
            vegetation_density=0.4,  # Modérée
            vegetation_spacing=6.0,
            use_clustering=True,
            cluster_size=4,

            # Matériaux (rock + snow)
            primary_material="rock",
            secondary_material="snow",

            # Lighting doux et clair
            sun_azimuth=120.0,
            sun_elevation=50.0,  # Soleil assez haut
            sun_intensity=1.3,   # Lumière intense
            ambient_strength=0.5,  # Ambiance claire

            # Atmosphère pure
            fog_density=0.008,  # Très peu de brouillard
            fog_enabled=True,
            atmosphere_enabled=True,

            # HDRI midday
            hdri_time="midday",

            # Caméra
            camera_distance=400.0,
            camera_elevation=20.0,
            camera_azimuth=45.0,

            height_scale=55.0,

            tags=["alps", "evian", "pristine", "snow", "advertising"]
        )

        # =====================================================================
        # THREE PEAKS - 3 sommets majestueux
        # =====================================================================
        self._presets["three_peaks"] = TerrainPreset(
            name="Three Peaks",
            description="Trois sommets majestueux dominant le paysage. "
                        "Composition dramatique avec pics bien définis.",
            category=PresetCategory.PEAKS,
            difficulty="medium",

            # Terrain avec contrôle pour 3 pics
            resolution=(512, 512),
            algorithm="ridged",  # Meilleur pour pics définis
            scale=80.0,
            octaves=12,
            ridge_influence=0.7,  # Très prononcé
            warp_strength=0.15,   # Peu de warp pour pics nets

            # Erosion modérée
            hydraulic_iterations=40,
            thermal_iterations=4,
            erosion_rate=0.25,

            # Végétation limitée (haute altitude)
            vegetation_enabled=True,
            vegetation_density=0.25,  # Faible
            vegetation_spacing=8.0,
            use_clustering=True,
            cluster_size=3,

            # Matériaux
            primary_material="rock",
            secondary_material="snow",

            # Lighting dramatique
            sun_azimuth=100.0,
            sun_elevation=35.0,  # Soleil plus bas pour ombres
            sun_intensity=1.4,
            ambient_strength=0.3,  # Ombres prononcées

            # Atmosphère
            fog_density=0.012,
            fog_enabled=True,
            atmosphere_enabled=True,

            # HDRI afternoon (lumière dorée)
            hdri_time="afternoon",

            # Caméra angle dramatique
            camera_distance=450.0,
            camera_elevation=18.0,
            camera_azimuth=60.0,

            height_scale=70.0,  # Plus haut pour majestueux

            tags=["peaks", "dramatic", "majestic", "mountains"]
        )

        # =====================================================================
        # SKI SLOPE - Piste de ski poudreuse
        # =====================================================================
        self._presets["ski_slope"] = TerrainPreset(
            name="Powder Ski Slope",
            description="Piste de ski avec poudreuse fraîche. "
                        "Pentes régulières, neige immaculée, végétation en bas.",
            category=PresetCategory.SKI,
            difficulty="easy",

            # Terrain avec pentes ski (pas trop raides)
            resolution=(512, 512),
            algorithm="hybrid",
            scale=150.0,
            octaves=8,
            ridge_influence=0.2,  # Peu de ridges (pentes douces)
            warp_strength=0.4,    # Formes naturelles

            # Erosion minimale (neige fraîche)
            hydraulic_iterations=20,
            thermal_iterations=2,
            erosion_rate=0.15,

            # Végétation (arbres en bas seulement)
            vegetation_enabled=True,
            vegetation_density=0.6,  # Dense en bas
            vegetation_spacing=4.0,
            use_clustering=True,
            cluster_size=6,

            # Matériaux (snow dominant)
            primary_material="snow",
            secondary_material="rock",

            # Lighting matin clair
            sun_azimuth=90.0,
            sun_elevation=30.0,  # Lumière rasante sur neige
            sun_intensity=1.5,   # Réflexion neige
            ambient_strength=0.6,  # Ambiance claire

            # Atmosphère claire
            fog_density=0.005,  # Très peu
            fog_enabled=True,
            atmosphere_enabled=True,

            # HDRI morning
            hdri_time="morning",

            # Caméra vue piste
            camera_distance=300.0,
            camera_elevation=30.0,
            camera_azimuth=30.0,

            height_scale=40.0,  # Modéré pour piste

            tags=["ski", "snow", "powder", "slope", "winter"]
        )

        # =====================================================================
        # MATTERHORN - Pic emblématique
        # =====================================================================
        self._presets["matterhorn"] = TerrainPreset(
            name="Matterhorn Peak",
            description="Pic pyramidal emblématique style Matterhorn. "
                        "Forme distinctive, arêtes prononcées, haute altitude.",
            category=PresetCategory.PEAKS,
            difficulty="hard",

            # Terrain pyramidal
            resolution=(512, 512),
            algorithm="ridged",
            scale=60.0,
            octaves=14,
            ridge_influence=0.8,  # Très prononcé
            warp_strength=0.1,    # Minimal pour forme nette

            # Erosion forte (montagne ancienne)
            hydraulic_iterations=60,
            thermal_iterations=8,
            erosion_rate=0.4,

            # Végétation très limitée
            vegetation_enabled=True,
            vegetation_density=0.15,
            vegetation_spacing=10.0,
            use_clustering=False,  # Arbres isolés
            cluster_size=2,

            # Matériaux
            primary_material="rock",
            secondary_material="snow",

            # Lighting dramatique
            sun_azimuth=110.0,
            sun_elevation=40.0,
            sun_intensity=1.3,
            ambient_strength=0.25,

            # Atmosphère
            fog_density=0.01,
            fog_enabled=True,
            atmosphere_enabled=True,

            # HDRI sunset
            hdri_time="sunset",

            # Caméra
            camera_distance=500.0,
            camera_elevation=15.0,
            camera_azimuth=50.0,

            height_scale=80.0,  # Très haut

            tags=["matterhorn", "iconic", "pyramid", "peak", "dramatic"]
        )

        # =====================================================================
        # MONT BLANC - Plus haut sommet des Alpes
        # =====================================================================
        self._presets["mont_blanc"] = TerrainPreset(
            name="Mont Blanc Massif",
            description="Massif du Mont Blanc - plus haut sommet des Alpes. "
                        "Immense, enneigé, majestueux.",
            category=PresetCategory.ALPS,
            difficulty="medium",

            # Terrain massif
            resolution=(768, 768),  # Plus grande résolution
            algorithm="ultra_realistic",
            scale=140.0,
            octaves=12,
            ridge_influence=0.45,
            warp_strength=0.3,

            # Erosion glaciaire
            hydraulic_iterations=45,
            thermal_iterations=5,
            erosion_rate=0.3,

            # Végétation alpine
            vegetation_enabled=True,
            vegetation_density=0.35,
            vegetation_spacing=7.0,
            use_clustering=True,
            cluster_size=5,

            # Matériaux
            primary_material="snow",
            secondary_material="rock",

            # Lighting
            sun_azimuth=130.0,
            sun_elevation=45.0,
            sun_intensity=1.4,
            ambient_strength=0.45,

            # Atmosphère
            fog_density=0.01,
            fog_enabled=True,
            atmosphere_enabled=True,

            # HDRI midday
            hdri_time="midday",

            # Caméra large
            camera_distance=600.0,
            camera_elevation=22.0,
            camera_azimuth=40.0,

            height_scale=90.0,  # Maximum

            tags=["mont_blanc", "alps", "highest", "massive", "snow"]
        )

        # =====================================================================
        # DOLOMITES - Formations rocheuses spectaculaires
        # =====================================================================
        self._presets["dolomites"] = TerrainPreset(
            name="Dolomites Towers",
            description="Tours et formations rocheuses des Dolomites. "
                        "Parois verticales, couleur distinctive, dramatique.",
            category=PresetCategory.ALPS,
            difficulty="hard",

            # Terrain vertical
            resolution=(512, 512),
            algorithm="swiss",  # Meilleur pour formations verticales
            scale=70.0,
            octaves=14,
            ridge_influence=0.75,
            warp_strength=0.2,

            # Erosion prononcée
            hydraulic_iterations=55,
            thermal_iterations=7,
            erosion_rate=0.35,

            # Végétation (pins et mélèzes)
            vegetation_enabled=True,
            vegetation_density=0.3,
            vegetation_spacing=6.0,
            use_clustering=True,
            cluster_size=4,

            # Matériaux (roche distinctive)
            primary_material="rock",
            secondary_material="grass",

            # Lighting chaud (lumière dorée)
            sun_azimuth=115.0,
            sun_elevation=38.0,
            sun_intensity=1.5,
            ambient_strength=0.35,

            # Atmosphère
            fog_density=0.012,
            fog_enabled=True,
            atmosphere_enabled=True,

            # HDRI afternoon/sunset
            hdri_time="afternoon",

            # Caméra
            camera_distance=380.0,
            camera_elevation=25.0,
            camera_azimuth=55.0,

            height_scale=75.0,

            tags=["dolomites", "towers", "vertical", "dramatic", "italy"]
        )

    def get_preset(self, name: str) -> Optional[TerrainPreset]:
        """Get preset by name"""
        return self._presets.get(name)

    def list_presets(self, category: Optional[PresetCategory] = None) -> List[str]:
        """List all preset names, optionally filtered by category"""
        if category:
            return [
                name for name, preset in self._presets.items()
                if preset.category == category
            ]
        return list(self._presets.keys())

    def get_all_presets(self) -> Dict[str, TerrainPreset]:
        """Get all presets"""
        return self._presets.copy()

    def get_presets_by_category(self, category: PresetCategory) -> Dict[str, TerrainPreset]:
        """Get all presets in a category"""
        return {
            name: preset for name, preset in self._presets.items()
            if preset.category == category
        }

    def get_preset_info(self, name: str) -> Optional[str]:
        """Get detailed info about a preset"""
        preset = self.get_preset(name)
        if not preset:
            return None

        info = f"""
{preset.name}
{'=' * len(preset.name)}

Description: {preset.description}
Category: {preset.category.value}
Difficulty: {preset.difficulty}

Terrain:
  - Algorithm: {preset.algorithm}
  - Resolution: {preset.resolution[0]}x{preset.resolution[1]}
  - Scale: {preset.scale}
  - Ridge influence: {preset.ridge_influence}

Vegetation:
  - Enabled: {preset.vegetation_enabled}
  - Density: {preset.vegetation_density}
  - Clustering: {preset.use_clustering}

Lighting:
  - Sun: {preset.sun_azimuth}° azimuth, {preset.sun_elevation}° elevation
  - HDRI: {preset.hdri_time}

Tags: {', '.join(preset.tags)}
"""
        return info


# Test
if __name__ == "__main__":
    print("Mountain Presets System - Test\n")

    presets = MountainPresets()

    # List all
    print("Available presets:")
    for name in presets.list_presets():
        preset = presets.get_preset(name)
        print(f"  - {name}: {preset.description[:60]}...")

    # Show detailed info
    print("\n" + "=" * 70)
    print("EVIAN ALPS PRESET DETAILS:")
    print("=" * 70)
    print(presets.get_preset_info("evian_alps"))

    # Show by category
    print("\n" + "=" * 70)
    print("SKI PRESETS:")
    print("=" * 70)
    ski_presets = presets.get_presets_by_category(PresetCategory.SKI)
    for name, preset in ski_presets.items():
        print(f"\n{preset.name}:")
        print(f"  {preset.description}")
