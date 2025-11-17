"""
Système de Presets Professionnels pour Mountain Studio Pro

Presets complets incluant:
- Paramètres terrain (type, érosion, seed)
- Paramètres végétation (densité, espèces)
- Paramètres caméra (angle, focale, mouvement)
- Style de rendu (prompt, modèle AI, post-processing)

Catégories:
- VFX Production (pour films, publicités)
- Game Development (Unreal/Unity assets)
- Landscape Photography (style photo pro)
- Artistic/Concept Art (styles artistiques)
- Quick Tests (tests rapides)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal, Any
import json
from pathlib import Path


@dataclass
class TerrainPreset:
    """Paramètres de génération du terrain"""
    width: int = 2048
    height: int = 2048
    mountain_type: Literal['alpine', 'volcanic', 'rolling', 'massive', 'rocky'] = 'alpine'

    # Noise parameters
    scale: float = 100.0
    octaves: int = 8
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int = 42

    # Advanced features
    domain_warp_strength: float = 0.3
    use_ridged_multifractal: bool = True

    # Erosion
    apply_hydraulic_erosion: bool = True
    apply_thermal_erosion: bool = True
    erosion_iterations: int = 50000
    erosion_strength: float = 0.5


@dataclass
class VegetationPreset:
    """Paramètres de végétation"""
    enabled: bool = True
    density: float = 0.5  # 0-1
    min_spacing: float = 3.0  # meters
    use_clustering: bool = True
    cluster_size: int = 5
    cluster_spread: float = 10.0

    # Species mix (will be auto-calculated if None)
    species_override: Optional[Dict[str, float]] = None

    # Tree variation
    scale_variation: float = 0.3  # 0-1, how much trees vary in size
    rotation_random: bool = True
    age_variation: float = 0.4


@dataclass
class CameraPreset:
    """Paramètres caméra pour rendu/vidéo"""
    # Single shot
    focal_length: int = 35  # mm
    aperture: str = 'f/11'
    angle: Literal['wide-angle', 'normal', 'telephoto', 'ultra-wide'] = 'wide-angle'
    shot_type: Literal['aerial', 'ground-level', 'low-angle', 'high-angle', 'eye-level'] = 'ground-level'
    composition: Literal['rule-of-thirds', 'centered', 'leading-lines', 'golden-ratio'] = 'rule-of-thirds'

    # Video (if applicable)
    num_frames: int = 24
    camera_movement: Literal['static', 'orbit', 'dolly', 'pan', 'custom'] = 'orbit'
    movement_speed: float = 0.5  # 0-1
    interpolation_strength: float = 0.25  # For temporal consistency


@dataclass
class RenderPreset:
    """Paramètres de rendu AI"""
    # Time/Weather
    season: Literal['spring', 'summer', 'autumn', 'winter'] = 'summer'
    time_of_day: Literal['dawn', 'morning', 'midday', 'afternoon', 'sunset', 'dusk', 'night'] = 'sunset'
    weather: Literal['clear', 'cloudy', 'overcast', 'foggy', 'stormy', 'snowy'] = 'clear'

    # AI Model
    model_backend: Literal['sdxl', 'comfyui'] = 'sdxl'
    model_name: str = 'epicrealism_xl'  # EpicRealism XL for photorealism

    # Generation params
    steps: int = 40
    cfg_scale: float = 7.5
    sampler: str = 'DPM++ 2M Karras'

    # Prompt
    photographer_style: Optional[str] = 'nat_geo'
    quality_level: Literal['standard', 'high', 'ultra', 'vfx'] = 'ultra'

    # Output
    output_resolution: int = 2048


@dataclass
class ExportPreset:
    """Paramètres d'export"""
    # Maps to export
    export_heightmap: bool = True
    export_normal_map: bool = True
    export_depth_map: bool = True
    export_ao_map: bool = True
    export_roughness_map: bool = True
    export_albedo_map: bool = True
    export_splatmap: bool = False  # PBR splatmap

    # 3D exports
    export_obj: bool = False
    export_fbx: bool = False
    export_blend: bool = False

    # Vegetation
    export_vegetation_instances: bool = False
    vegetation_format: Literal['json', 'csv', 'unreal', 'unity'] = 'json'

    # Format
    image_format: Literal['png', 'exr', 'tiff'] = 'png'
    use_16bit: bool = False
    use_32bit_exr: bool = False


@dataclass
class CompletePreset:
    """Preset complet combinant tous les paramètres"""
    name: str
    description: str
    category: Literal['vfx_production', 'game_dev', 'photography', 'artistic', 'quick_test']

    terrain: TerrainPreset
    vegetation: VegetationPreset
    camera: CameraPreset
    render: RenderPreset
    export: ExportPreset

    # Metadata
    author: str = "Mountain Studio Pro"
    version: str = "2.0"
    tags: List[str] = None
    thumbnail: Optional[str] = None  # Path to preview image


class PresetManager:
    """Gestionnaire de presets professionnels"""

    def __init__(self, presets_dir: Optional[Path] = None):
        if presets_dir is None:
            presets_dir = Path(__file__).parent / "presets"

        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(exist_ok=True)

        # Load built-in presets
        self.presets = self._create_builtin_presets()

    def _create_builtin_presets(self) -> Dict[str, CompletePreset]:
        """Crée les presets intégrés professionnels"""

        presets = {}

        # ========================================
        # VFX PRODUCTION PRESETS
        # ========================================

        presets['vfx_epic_mountain'] = CompletePreset(
            name="VFX Epic Mountain",
            description="Epic mountain shot for film/commercial production. Dramatic alpine peaks at golden hour.",
            category='vfx_production',
            terrain=TerrainPreset(
                width=4096,
                height=4096,
                mountain_type='alpine',
                scale=150.0,
                octaves=10,
                persistence=0.55,
                lacunarity=2.1,
                seed=42,
                domain_warp_strength=0.4,
                use_ridged_multifractal=True,
                apply_hydraulic_erosion=True,
                apply_thermal_erosion=True,
                erosion_iterations=100000,
                erosion_strength=0.6
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.4,
                min_spacing=4.0,
                use_clustering=True,
                cluster_size=8,
                cluster_spread=15.0,
                scale_variation=0.35,
                age_variation=0.5
            ),
            camera=CameraPreset(
                focal_length=35,
                aperture='f/11',
                angle='wide-angle',
                shot_type='ground-level',
                composition='rule-of-thirds',
                num_frames=48,
                camera_movement='orbit',
                movement_speed=0.3,
                interpolation_strength=0.2
            ),
            render=RenderPreset(
                season='summer',
                time_of_day='sunset',
                weather='clear',
                model_backend='sdxl',
                model_name='epicrealism_xl',
                steps=50,
                cfg_scale=7.5,
                sampler='DPM++ 2M Karras',
                photographer_style='galen_rowell',
                quality_level='vfx',
                output_resolution=4096
            ),
            export=ExportPreset(
                export_heightmap=True,
                export_normal_map=True,
                export_depth_map=True,
                export_ao_map=True,
                export_roughness_map=True,
                export_albedo_map=True,
                export_splatmap=True,
                export_obj=True,
                export_fbx=True,
                export_vegetation_instances=True,
                vegetation_format='json',
                image_format='exr',
                use_32bit_exr=True
            ),
            tags=['vfx', 'production', 'epic', 'cinematic', '4k']
        )

        presets['vfx_misty_forest'] = CompletePreset(
            name="VFX Misty Forest Mountain",
            description="Atmospheric foggy mountain forest for moody scenes. Cinematic quality.",
            category='vfx_production',
            terrain=TerrainPreset(
                width=4096,
                height=4096,
                mountain_type='massive',
                scale=120.0,
                octaves=9,
                persistence=0.5,
                lacunarity=2.0,
                seed=123,
                domain_warp_strength=0.35,
                erosion_iterations=80000,
                erosion_strength=0.7
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.75,
                min_spacing=3.5,
                use_clustering=True,
                cluster_size=12,
                cluster_spread=20.0
            ),
            camera=CameraPreset(
                focal_length=85,
                aperture='f/5.6',
                angle='telephoto',
                shot_type='high-angle',
                composition='centered',
                num_frames=60,
                camera_movement='dolly',
                movement_speed=0.2
            ),
            render=RenderPreset(
                season='autumn',
                time_of_day='dawn',
                weather='foggy',
                model_name='juggernaut_xl',
                steps=45,
                cfg_scale=8.0,
                photographer_style='michael_kenna',
                quality_level='vfx'
            ),
            export=ExportPreset(
                export_heightmap=True,
                export_normal_map=True,
                export_ao_map=True,
                export_splatmap=True,
                image_format='exr',
                use_32bit_exr=True
            ),
            tags=['vfx', 'atmospheric', 'forest', 'fog', 'cinematic']
        )

        # ========================================
        # GAME DEVELOPMENT PRESETS
        # ========================================

        presets['game_unreal_landscape'] = CompletePreset(
            name="Game: Unreal Engine Landscape",
            description="Optimized for Unreal Engine 5. Includes all necessary maps and vegetation instances.",
            category='game_dev',
            terrain=TerrainPreset(
                width=2048,
                height=2048,
                mountain_type='alpine',
                scale=100.0,
                octaves=8,
                persistence=0.5,
                lacunarity=2.0,
                seed=999,
                domain_warp_strength=0.3,
                erosion_iterations=50000,
                erosion_strength=0.5
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.6,
                min_spacing=5.0,
                use_clustering=True,
                cluster_size=10
            ),
            camera=CameraPreset(
                focal_length=50,
                aperture='f/8',
                angle='normal',
                shot_type='eye-level',
                composition='rule-of-thirds'
            ),
            render=RenderPreset(
                season='summer',
                time_of_day='midday',
                weather='clear',
                model_name='protovision_xl',
                steps=35,
                cfg_scale=7.5,
                quality_level='high',
                output_resolution=2048
            ),
            export=ExportPreset(
                export_heightmap=True,
                export_normal_map=True,
                export_roughness_map=True,
                export_albedo_map=True,
                export_splatmap=True,
                export_obj=True,
                export_vegetation_instances=True,
                vegetation_format='unreal',
                image_format='png'
            ),
            tags=['game', 'unreal', 'ue5', 'landscape', 'optimized']
        )

        presets['game_unity_terrain'] = CompletePreset(
            name="Game: Unity Terrain",
            description="Optimized for Unity. Standard 2k textures with splatmap for terrain shader.",
            category='game_dev',
            terrain=TerrainPreset(
                width=2048,
                height=2048,
                mountain_type='rolling',
                scale=80.0,
                octaves=7,
                seed=777,
                erosion_iterations=40000
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.55,
                min_spacing=4.5,
                use_clustering=True
            ),
            camera=CameraPreset(
                focal_length=50,
                angle='normal',
                shot_type='eye-level'
            ),
            render=RenderPreset(
                season='summer',
                time_of_day='afternoon',
                weather='clear',
                steps=30,
                quality_level='high'
            ),
            export=ExportPreset(
                export_heightmap=True,
                export_normal_map=True,
                export_splatmap=True,
                export_vegetation_instances=True,
                vegetation_format='unity',
                image_format='png'
            ),
            tags=['game', 'unity', 'terrain', 'optimized', '2k']
        )

        # ========================================
        # LANDSCAPE PHOTOGRAPHY PRESETS
        # ========================================

        presets['photo_golden_hour_alpine'] = CompletePreset(
            name="Photo: Golden Hour Alpine",
            description="Classic landscape photography. Alpine mountain at golden hour, National Geographic style.",
            category='photography',
            terrain=TerrainPreset(
                width=3072,
                height=2048,  # 3:2 ratio like DSLR
                mountain_type='alpine',
                scale=120.0,
                octaves=9,
                seed=2024,
                erosion_iterations=70000
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.35,
                min_spacing=4.0
            ),
            camera=CameraPreset(
                focal_length=24,
                aperture='f/11',
                angle='ultra-wide',
                shot_type='ground-level',
                composition='rule-of-thirds'
            ),
            render=RenderPreset(
                season='summer',
                time_of_day='sunset',
                weather='clear',
                model_name='realvis_xl_v4',
                steps=40,
                cfg_scale=7.0,
                photographer_style='galen_rowell',
                quality_level='ultra',
                output_resolution=3072
            ),
            export=ExportPreset(
                export_heightmap=False,
                export_normal_map=False,
                image_format='png'
            ),
            tags=['photography', 'landscape', 'golden-hour', 'nat-geo']
        )

        presets['photo_black_white_ansel'] = CompletePreset(
            name="Photo: B&W Ansel Adams Style",
            description="Black and white dramatic landscape in the style of Ansel Adams. Zone system aesthetic.",
            category='photography',
            terrain=TerrainPreset(
                width=3072,
                height=2048,
                mountain_type='massive',
                scale=140.0,
                octaves=9,
                seed=1927,  # Ansel Adams birth year
                erosion_iterations=60000
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.3
            ),
            camera=CameraPreset(
                focal_length=50,
                aperture='f/16',
                angle='normal',
                shot_type='eye-level',
                composition='golden-ratio'
            ),
            render=RenderPreset(
                season='autumn',
                time_of_day='afternoon',
                weather='clear',
                model_name='epicrealism_xl',
                steps=45,
                photographer_style='ansel_adams',
                quality_level='vfx'
            ),
            export=ExportPreset(
                image_format='png'
            ),
            tags=['photography', 'black-white', 'ansel-adams', 'fine-art']
        )

        # ========================================
        # ARTISTIC / CONCEPT ART PRESETS
        # ========================================

        presets['art_fantasy_peaks'] = CompletePreset(
            name="Art: Fantasy Mountain Peaks",
            description="Dramatic fantasy-style mountains for concept art. Exaggerated features.",
            category='artistic',
            terrain=TerrainPreset(
                width=2048,
                height=2048,
                mountain_type='rocky',
                scale=200.0,
                octaves=10,
                persistence=0.6,
                seed=8888,
                domain_warp_strength=0.5,
                erosion_strength=0.3  # Less erosion for sharper fantasy look
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.25
            ),
            camera=CameraPreset(
                focal_length=24,
                aperture='f/8',
                angle='ultra-wide',
                shot_type='low-angle',
                composition='centered'
            ),
            render=RenderPreset(
                season='summer',
                time_of_day='sunset',
                weather='dramatic',
                model_name='dreamshaper_xl',
                steps=40,
                quality_level='ultra'
            ),
            export=ExportPreset(
                image_format='png'
            ),
            tags=['art', 'fantasy', 'concept-art', 'dramatic']
        )

        presets['art_minimalist_zen'] = CompletePreset(
            name="Art: Minimalist Zen Mountain",
            description="Minimalist, peaceful mountain landscape. Clean composition, muted colors.",
            category='artistic',
            terrain=TerrainPreset(
                width=2048,
                height=2048,
                mountain_type='rolling',
                scale=60.0,
                octaves=6,
                persistence=0.4,
                seed=108,  # Buddhist number
                erosion_iterations=30000
            ),
            vegetation=VegetationPreset(
                enabled=True,
                density=0.15  # Sparse for minimalism
            ),
            camera=CameraPreset(
                focal_length=85,
                aperture='f/5.6',
                angle='telephoto',
                shot_type='eye-level',
                composition='centered'
            ),
            render=RenderPreset(
                season='spring',
                time_of_day='morning',
                weather='foggy',
                model_name='dreamshaper_xl',
                steps=35,
                photographer_style='michael_kenna',
                quality_level='high'
            ),
            export=ExportPreset(
                image_format='png'
            ),
            tags=['art', 'minimalist', 'zen', 'peaceful']
        )

        # ========================================
        # QUICK TEST PRESETS
        # ========================================

        presets['test_quick_preview'] = CompletePreset(
            name="Test: Quick Preview",
            description="Fast generation for testing. Low resolution, minimal processing.",
            category='quick_test',
            terrain=TerrainPreset(
                width=512,
                height=512,
                mountain_type='alpine',
                scale=50.0,
                octaves=6,
                seed=42,
                apply_hydraulic_erosion=False,
                apply_thermal_erosion=False
            ),
            vegetation=VegetationPreset(
                enabled=False
            ),
            camera=CameraPreset(
                focal_length=50,
                angle='normal',
                shot_type='eye-level'
            ),
            render=RenderPreset(
                time_of_day='midday',
                weather='clear',
                steps=20,
                cfg_scale=7.0,
                quality_level='standard',
                output_resolution=512
            ),
            export=ExportPreset(
                export_heightmap=True,
                export_normal_map=False,
                image_format='png'
            ),
            tags=['test', 'quick', 'preview', 'low-res']
        )

        presets['test_erosion_comparison'] = CompletePreset(
            name="Test: Erosion Comparison",
            description="Medium quality for testing erosion effects.",
            category='quick_test',
            terrain=TerrainPreset(
                width=1024,
                height=1024,
                mountain_type='alpine',
                scale=80.0,
                octaves=7,
                seed=42,
                apply_hydraulic_erosion=True,
                apply_thermal_erosion=True,
                erosion_iterations=25000
            ),
            vegetation=VegetationPreset(
                enabled=False
            ),
            camera=CameraPreset(
                focal_length=50,
                angle='normal'
            ),
            render=RenderPreset(
                time_of_day='afternoon',
                steps=25,
                quality_level='standard',
                output_resolution=1024
            ),
            export=ExportPreset(
                export_heightmap=True,
                export_normal_map=True,
                image_format='png'
            ),
            tags=['test', 'erosion', 'medium', '1k']
        )

        return presets

    def get_preset(self, preset_name: str) -> Optional[CompletePreset]:
        """Récupère un preset par nom"""
        return self.presets.get(preset_name)

    def list_presets(self, category: Optional[str] = None) -> List[str]:
        """Liste tous les presets, optionnellement filtrés par catégorie"""
        if category is None:
            return list(self.presets.keys())

        return [
            name for name, preset in self.presets.items()
            if preset.category == category
        ]

    def get_presets_by_category(self) -> Dict[str, List[str]]:
        """Retourne les presets groupés par catégorie"""
        categorized = {
            'vfx_production': [],
            'game_dev': [],
            'photography': [],
            'artistic': [],
            'quick_test': []
        }

        for name, preset in self.presets.items():
            categorized[preset.category].append(name)

        return categorized

    def save_preset(self, preset: CompletePreset, preset_name: str):
        """Sauvegarde un preset personnalisé"""
        filepath = self.presets_dir / f"{preset_name}.json"

        preset_dict = asdict(preset)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(preset_dict, f, indent=2, ensure_ascii=False)

        # Ajouter au dictionnaire
        self.presets[preset_name] = preset

    def load_custom_preset(self, preset_name: str) -> Optional[CompletePreset]:
        """Charge un preset personnalisé depuis un fichier"""
        filepath = self.presets_dir / f"{preset_name}.json"

        if not filepath.exists():
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            preset_dict = json.load(f)

        # Reconstruire les dataclasses
        preset = CompletePreset(
            name=preset_dict['name'],
            description=preset_dict['description'],
            category=preset_dict['category'],
            terrain=TerrainPreset(**preset_dict['terrain']),
            vegetation=VegetationPreset(**preset_dict['vegetation']),
            camera=CameraPreset(**preset_dict['camera']),
            render=RenderPreset(**preset_dict['render']),
            export=ExportPreset(**preset_dict['export']),
            author=preset_dict.get('author', 'Custom'),
            version=preset_dict.get('version', '1.0'),
            tags=preset_dict.get('tags', []),
            thumbnail=preset_dict.get('thumbnail')
        )

        self.presets[preset_name] = preset
        return preset

    def export_preset_to_dict(self, preset_name: str) -> Optional[Dict]:
        """Exporte un preset en dictionnaire pour affichage UI"""
        preset = self.get_preset(preset_name)
        if preset is None:
            return None

        return asdict(preset)

    def search_presets(self, query: str) -> List[str]:
        """Recherche des presets par nom, description ou tags"""
        query_lower = query.lower()
        results = []

        for name, preset in self.presets.items():
            # Rechercher dans nom
            if query_lower in name.lower():
                results.append(name)
                continue

            # Rechercher dans description
            if query_lower in preset.description.lower():
                results.append(name)
                continue

            # Rechercher dans tags
            if preset.tags and any(query_lower in tag.lower() for tag in preset.tags):
                results.append(name)
                continue

        return results


# Fonction utilitaire pour tester
def test_presets():
    """Test du système de presets"""

    manager = PresetManager()

    print("=" * 80)
    print("PRESETS PAR CATÉGORIE")
    print("=" * 80)

    categorized = manager.get_presets_by_category()
    for category, preset_names in categorized.items():
        print(f"\n{category.upper()}:")
        for name in preset_names:
            preset = manager.get_preset(name)
            print(f"  - {preset.name}")
            print(f"    {preset.description}")

    print("\n" + "=" * 80)
    print("DÉTAILS PRESET VFX EPIC MOUNTAIN")
    print("=" * 80)

    preset = manager.get_preset('vfx_epic_mountain')
    if preset:
        print(f"\nNom: {preset.name}")
        print(f"Catégorie: {preset.category}")
        print(f"Description: {preset.description}")
        print(f"\nTerrain:")
        print(f"  - Résolution: {preset.terrain.width}x{preset.terrain.height}")
        print(f"  - Type: {preset.terrain.mountain_type}")
        print(f"  - Seed: {preset.terrain.seed}")
        print(f"  - Érosion hydraulique: {preset.terrain.apply_hydraulic_erosion}")
        print(f"  - Itérations: {preset.terrain.erosion_iterations}")
        print(f"\nVégétation:")
        print(f"  - Activée: {preset.vegetation.enabled}")
        print(f"  - Densité: {preset.vegetation.density}")
        print(f"  - Clustering: {preset.vegetation.use_clustering}")
        print(f"\nRendu:")
        print(f"  - Modèle: {preset.render.model_name}")
        print(f"  - Steps: {preset.render.steps}")
        print(f"  - Temps: {preset.render.time_of_day}")
        print(f"  - Saison: {preset.render.season}")
        print(f"\nExport:")
        print(f"  - Format: {preset.export.image_format}")
        print(f"  - OBJ: {preset.export.export_obj}")
        print(f"  - Végétation instances: {preset.export.export_vegetation_instances}")

    print("\n" + "=" * 80)
    print("RECHERCHE: 'fog'")
    print("=" * 80)
    results = manager.search_presets('fog')
    for name in results:
        preset = manager.get_preset(name)
        print(f"  - {preset.name}: {preset.description}")


if __name__ == "__main__":
    test_presets()
