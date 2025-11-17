"""
Configuration Centralisée - Mountain Studio Pro v2.0

Toutes les configurations de l'application en un seul endroit:
- Chemins et dossiers
- Paramètres par défaut
- Backends AI (SDXL, ComfyUI)
- Performance et optimisation
- Export et formats
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


# ========================================
# PATHS & DIRECTORIES
# ========================================

class AppPaths:
    """Chemins et dossiers de l'application"""

    # Root directory
    ROOT_DIR = Path(__file__).parent.parent

    # Core modules
    CORE_DIR = ROOT_DIR / "core"
    TERRAIN_DIR = CORE_DIR / "terrain"
    VEGETATION_DIR = CORE_DIR / "vegetation"
    RENDERING_DIR = CORE_DIR / "rendering"
    EXPORT_DIR = CORE_DIR / "export"

    # Config
    CONFIG_DIR = ROOT_DIR / "config"
    PRESETS_DIR = CONFIG_DIR / "presets"

    # Services
    SERVICES_DIR = ROOT_DIR / "services"

    # UI
    UI_DIR = ROOT_DIR / "ui"

    # Output directories
    OUTPUT_DIR = ROOT_DIR / "output"
    HEIGHTMAPS_DIR = OUTPUT_DIR / "heightmaps"
    TEXTURES_DIR = OUTPUT_DIR / "textures"
    VIDEOS_DIR = OUTPUT_DIR / "videos"
    EXPORTS_DIR = OUTPUT_DIR / "exports"

    # Cache
    CACHE_DIR = ROOT_DIR / ".cache"
    MODEL_CACHE_DIR = CACHE_DIR / "models"
    TEMP_DIR = CACHE_DIR / "temp"

    @classmethod
    def ensure_dirs(cls):
        """Crée tous les dossiers nécessaires"""
        dirs_to_create = [
            cls.OUTPUT_DIR,
            cls.HEIGHTMAPS_DIR,
            cls.TEXTURES_DIR,
            cls.VIDEOS_DIR,
            cls.EXPORTS_DIR,
            cls.CACHE_DIR,
            cls.MODEL_CACHE_DIR,
            cls.TEMP_DIR,
            cls.PRESETS_DIR
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(exist_ok=True, parents=True)

        logger.info("Dossiers initialisés")


# ========================================
# DEFAULT SETTINGS
# ========================================

@dataclass
class TerrainDefaults:
    """Paramètres par défaut pour génération terrain"""

    # Resolution
    width: int = 2048
    height: int = 2048

    # Noise
    scale: float = 100.0
    octaves: int = 8
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int = 42

    # Mountain type
    mountain_type: str = 'alpine'  # alpine, volcanic, rolling, massive, rocky

    # Advanced
    domain_warp_strength: float = 0.3
    use_ridged_multifractal: bool = True

    # Erosion
    apply_hydraulic_erosion: bool = True
    apply_thermal_erosion: bool = True
    erosion_iterations: int = 50000
    erosion_strength: float = 0.5

    # Hydraulic erosion specific
    hydraulic_droplet_lifetime: int = 30
    hydraulic_inertia: float = 0.05
    hydraulic_sediment_capacity: float = 4.0
    hydraulic_deposition_speed: float = 0.3
    hydraulic_erosion_speed: float = 0.3

    # Thermal erosion specific
    thermal_talus_angle: float = 0.7  # ~35 degrees
    thermal_erosion_amount: float = 0.5


@dataclass
class VegetationDefaults:
    """Paramètres par défaut pour végétation"""

    enabled: bool = True
    density: float = 0.5  # 0-1
    min_spacing: float = 3.0  # meters

    # Clustering
    use_clustering: bool = True
    cluster_size: int = 5
    cluster_spread: float = 10.0

    # Variation
    scale_variation: float = 0.3
    rotation_random: bool = True
    age_variation: float = 0.4

    # Export
    export_instances: bool = False
    export_density_map: bool = True


@dataclass
class RenderDefaults:
    """Paramètres par défaut pour rendu AI"""

    # Backend
    backend: str = 'sdxl'  # 'sdxl' or 'comfyui'
    model_name: str = 'epicrealism_xl'

    # Generation parameters
    steps: int = 40
    cfg_scale: float = 7.5
    sampler: str = 'DPM++ 2M Karras'

    # Resolution
    width: int = 2048
    height: int = 2048

    # Scene
    season: str = 'summer'
    time_of_day: str = 'sunset'
    weather: str = 'clear'

    # Quality
    quality_level: str = 'ultra'  # standard, high, ultra, vfx
    photographer_style: Optional[str] = 'nat_geo'

    # Video
    num_frames: int = 24
    fps: int = 24
    camera_movement: str = 'orbit'
    interpolation_strength: float = 0.25


@dataclass
class ExportDefaults:
    """Paramètres par défaut pour export"""

    # Maps to export
    export_heightmap: bool = True
    export_normal_map: bool = True
    export_depth_map: bool = True
    export_ao_map: bool = True
    export_roughness_map: bool = True
    export_albedo_map: bool = False
    export_splatmap: bool = False

    # 3D
    export_obj: bool = False
    export_fbx: bool = False
    export_blend: bool = False

    # Format
    image_format: str = 'png'  # png, exr, tiff
    use_16bit: bool = False
    use_32bit_exr: bool = False

    # Vegetation
    export_vegetation_instances: bool = False
    vegetation_format: str = 'json'  # json, csv, unreal, unity


# ========================================
# AI MODEL CONFIGURATIONS
# ========================================

class AIModelConfig:
    """Configuration des modèles AI"""

    # Stable Diffusion XL Models
    SDXL_MODELS = {
        'epicrealism_xl': {
            'name': 'EpicRealism XL',
            'checkpoint': 'epicrealism_xl_v10.safetensors',
            'description': 'Best for photorealistic landscapes',
            'recommended_steps': 40,
            'recommended_cfg': 7.5,
            'recommended_sampler': 'DPM++ 2M Karras'
        },
        'juggernaut_xl': {
            'name': 'Juggernaut XL',
            'checkpoint': 'juggernaut_xl_v9.safetensors',
            'description': 'Dramatic detailed landscapes',
            'recommended_steps': 45,
            'recommended_cfg': 8.0,
            'recommended_sampler': 'DPM++ 2M SDE Karras'
        },
        'realvis_xl_v4': {
            'name': 'RealVisXL V4',
            'checkpoint': 'realvisxl_v4.safetensors',
            'description': 'Ultra-realistic nature photography',
            'recommended_steps': 35,
            'recommended_cfg': 7.0,
            'recommended_sampler': 'Euler a'
        },
        'protovision_xl': {
            'name': 'ProtoVision XL',
            'checkpoint': 'protovision_xl_v6.safetensors',
            'description': 'VFX-quality versatile',
            'recommended_steps': 40,
            'recommended_cfg': 7.5,
            'recommended_sampler': 'DPM++ 2M Karras'
        },
        'dreamshaper_xl': {
            'name': 'DreamShaper XL',
            'checkpoint': 'dreamshaper_xl.safetensors',
            'description': 'Artistic realistic style',
            'recommended_steps': 40,
            'recommended_cfg': 8.0,
            'recommended_sampler': 'Euler a'
        }
    }

    # ComfyUI Configuration
    COMFYUI_CONFIG = {
        'default_url': 'http://127.0.0.1:8188',
        'timeout': 300,  # seconds
        'check_interval': 1.0,  # seconds between status checks
        'max_retries': 3
    }

    # ControlNet Models
    CONTROLNET_MODELS = {
        'depth': 'control_v11f1p_sd15_depth',
        'canny': 'control_v11p_sd15_canny',
        'normal': 'control_v11p_sd15_normalbae',
        'tile': 'control_v11f1e_sd15_tile'
    }


# ========================================
# PERFORMANCE SETTINGS
# ========================================

@dataclass
class PerformanceConfig:
    """Configuration performance et optimisation"""

    # GPU
    use_gpu: bool = True
    gpu_device_id: int = 0

    # Memory
    max_vram_gb: float = 8.0
    enable_cpu_offload: bool = True

    # Processing
    use_numba: bool = True  # JIT compilation
    use_cupy: bool = False  # GPU arrays (requires CuPy installed)

    # Multiprocessing
    num_workers: int = 4
    use_multiprocessing: bool = True

    # Caching
    enable_model_caching: bool = True
    enable_heightmap_caching: bool = True
    cache_max_size_gb: float = 5.0


# ========================================
# APPLICATION SETTINGS
# ========================================

@dataclass
class AppSettings:
    """Settings généraux de l'application"""

    # Info
    app_name: str = "Mountain Studio Pro"
    version: str = "2.0.0"
    author: str = "Mountain Studio Team"

    # UI
    theme: str = 'dark'  # dark, light
    language: str = 'fr'  # fr, en
    window_width: int = 1600
    window_height: int = 1000

    # Logging
    log_level: str = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_file: str = 'mountain_studio.log'

    # Auto-save
    auto_save_enabled: bool = True
    auto_save_interval: int = 300  # seconds

    # Defaults
    terrain: TerrainDefaults = field(default_factory=TerrainDefaults)
    vegetation: VegetationDefaults = field(default_factory=VegetationDefaults)
    render: RenderDefaults = field(default_factory=RenderDefaults)
    export: ExportDefaults = field(default_factory=ExportDefaults)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


# ========================================
# CONFIG MANAGER
# ========================================

class ConfigManager:
    """Gestionnaire de configuration centralisé"""

    def __init__(self, config_file: Optional[Path] = None):
        if config_file is None:
            config_file = AppPaths.CONFIG_DIR / "settings.json"

        self.config_file = Path(config_file)
        self.settings = AppSettings()

        # Charger config si existe
        if self.config_file.exists():
            self.load()
        else:
            # Créer config par défaut
            self.save()

    def load(self):
        """Charge la configuration depuis le fichier"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruire settings
            self.settings = self._dict_to_settings(data)

            logger.info(f"Configuration chargée depuis {self.config_file}")

        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            logger.info("Utilisation configuration par défaut")

    def save(self):
        """Sauvegarde la configuration"""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(exist_ok=True, parents=True)

            # Convert to dict
            data = self._settings_to_dict(self.settings)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration sauvegardée dans {self.config_file}")

        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")

    def _settings_to_dict(self, settings: AppSettings) -> Dict[str, Any]:
        """Convertit AppSettings en dict"""
        from dataclasses import asdict
        return asdict(settings)

    def _dict_to_settings(self, data: Dict[str, Any]) -> AppSettings:
        """Convertit dict en AppSettings"""

        # Reconstruct nested dataclasses
        terrain = TerrainDefaults(**data.get('terrain', {}))
        vegetation = VegetationDefaults(**data.get('vegetation', {}))
        render = RenderDefaults(**data.get('render', {}))
        export = ExportDefaults(**data.get('export', {}))
        performance = PerformanceConfig(**data.get('performance', {}))

        # Remove nested from data
        app_data = {k: v for k, v in data.items()
                    if k not in ['terrain', 'vegetation', 'render', 'export', 'performance']}

        settings = AppSettings(
            **app_data,
            terrain=terrain,
            vegetation=vegetation,
            render=render,
            export=export,
            performance=performance
        )

        return settings

    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de config"""
        try:
            # Support dot notation: "terrain.width"
            parts = key.split('.')
            value = self.settings

            for part in parts:
                value = getattr(value, part)

            return value

        except AttributeError:
            return default

    def set(self, key: str, value: Any):
        """Définit une valeur de config"""
        try:
            # Support dot notation
            parts = key.split('.')

            if len(parts) == 1:
                setattr(self.settings, key, value)
            else:
                # Navigate to parent
                parent = self.settings
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                # Set value
                setattr(parent, parts[-1], value)

        except AttributeError as e:
            logger.error(f"Impossible de définir {key}: {e}")

    def reset_to_defaults(self):
        """Réinitialise à la configuration par défaut"""
        self.settings = AppSettings()
        self.save()
        logger.info("Configuration réinitialisée aux valeurs par défaut")


# ========================================
# GLOBAL CONFIG INSTANCE
# ========================================

# Instance globale accessible partout
_global_config = None


def get_config() -> ConfigManager:
    """Récupère l'instance globale de configuration"""
    global _global_config

    if _global_config is None:
        _global_config = ConfigManager()

    return _global_config


def init_config(config_file: Optional[Path] = None):
    """Initialise la configuration globale"""
    global _global_config
    _global_config = ConfigManager(config_file)

    # Ensure directories
    AppPaths.ensure_dirs()

    return _global_config


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def get_terrain_defaults() -> TerrainDefaults:
    """Récupère les defaults terrain"""
    return get_config().settings.terrain


def get_vegetation_defaults() -> VegetationDefaults:
    """Récupère les defaults végétation"""
    return get_config().settings.vegetation


def get_render_defaults() -> RenderDefaults:
    """Récupère les defaults rendu"""
    return get_config().settings.render


def get_export_defaults() -> ExportDefaults:
    """Récupère les defaults export"""
    return get_config().settings.export


def get_performance_config() -> PerformanceConfig:
    """Récupère la config performance"""
    return get_config().settings.performance


# Test
if __name__ == "__main__":
    # Initialiser config
    config = init_config()

    print("=" * 80)
    print("MOUNTAIN STUDIO PRO - CONFIGURATION")
    print("=" * 80)

    print(f"\nApp: {config.settings.app_name} v{config.settings.version}")
    print(f"Theme: {config.settings.theme}")
    print(f"Language: {config.settings.language}")

    print("\nTERRAIN DEFAULTS:")
    print(f"  Resolution: {config.settings.terrain.width}x{config.settings.terrain.height}")
    print(f"  Mountain Type: {config.settings.terrain.mountain_type}")
    print(f"  Hydraulic Erosion: {config.settings.terrain.apply_hydraulic_erosion}")
    print(f"  Erosion Iterations: {config.settings.terrain.erosion_iterations}")

    print("\nVEGETATION DEFAULTS:")
    print(f"  Enabled: {config.settings.vegetation.enabled}")
    print(f"  Density: {config.settings.vegetation.density}")
    print(f"  Clustering: {config.settings.vegetation.use_clustering}")

    print("\nRENDER DEFAULTS:")
    print(f"  Backend: {config.settings.render.backend}")
    print(f"  Model: {config.settings.render.model_name}")
    print(f"  Steps: {config.settings.render.steps}")
    print(f"  Quality: {config.settings.render.quality_level}")

    print("\nPERFORMANCE:")
    print(f"  Use GPU: {config.settings.performance.use_gpu}")
    print(f"  Use Numba: {config.settings.performance.use_numba}")
    print(f"  Max VRAM: {config.settings.performance.max_vram_gb}GB")

    print("\nPATHS:")
    print(f"  Root: {AppPaths.ROOT_DIR}")
    print(f"  Output: {AppPaths.OUTPUT_DIR}")
    print(f"  Cache: {AppPaths.CACHE_DIR}")

    print("\nAI MODELS:")
    for model_id, model_info in AIModelConfig.SDXL_MODELS.items():
        print(f"  {model_info['name']}: {model_info['description']}")

    # Test get/set
    print("\nTEST GET/SET:")
    print(f"  terrain.width = {config.get('terrain.width')}")
    config.set('terrain.width', 4096)
    print(f"  après set(4096) = {config.get('terrain.width')}")

    # Save
    config.save()
    print(f"\nConfiguration sauvegardée dans {config.config_file}")
