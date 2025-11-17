"""
VFX Prompt Generator Ultra-Réaliste
Génère des prompts optimisés pour Stable Diffusion XL avec qualité VFX professionnelle

Structure des prompts basée sur les meilleures pratiques VFX 2025:
- [SUBJECT] Description principale du terrain
- [ENVIRONMENT] Conditions atmosphériques et contexte
- [COMPOSITION] Cadrage et éléments visuels
- [LIGHTING] Éclairage naturel/artificiel
- [CAMERA] Paramètres photographiques réalistes
- [PHOTOGRAPHER] Style photographique professionnel
- [TECHNICAL] Keywords VFX modernes (UE5, RTX, SSAO, hypersharp, etc.)

Références:
- Stable Diffusion XL VFX Guide 2025
- Professional Photography Prompting Techniques
- Unreal Engine 5 Photorealism Keywords
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TerrainContext:
    """Contexte du terrain pour génération de prompts"""
    mountain_type: str  # alpine, volcanic, rolling, massive, rocky
    elevation_range: Tuple[float, float]  # min, max elevation (0-1)
    dominant_biome: str  # alpine, subalpine, montane_forest, valley_floor
    vegetation_density: float  # 0-1
    dominant_species: List[str]  # pine, spruce, fir, deciduous
    has_snow: bool
    has_water: bool
    season: str  # spring, summer, autumn, winter
    time_of_day: Literal['dawn', 'morning', 'midday', 'afternoon', 'sunset', 'dusk', 'night']
    weather: Literal['clear', 'cloudy', 'overcast', 'foggy', 'stormy', 'snowy']


@dataclass
class CameraSettings:
    """Paramètres photographiques pour prompts réalistes"""
    focal_length: int  # mm (14-200)
    aperture: str  # f/1.4, f/2.8, f/5.6, f/11, f/16
    angle: Literal['wide-angle', 'normal', 'telephoto', 'ultra-wide']
    shot_type: Literal['aerial', 'ground-level', 'low-angle', 'high-angle', 'eye-level']
    composition: Literal['rule-of-thirds', 'centered', 'leading-lines', 'golden-ratio']


class VFXPromptGenerator:
    """
    Générateur de prompts ultra-réalistes pour Stable Diffusion XL
    Optimisé pour rendus VFX professionnels
    """

    def __init__(self):
        self.vfx_quality_keywords = self._init_vfx_keywords()
        self.lighting_presets = self._init_lighting_presets()
        self.photographer_styles = self._init_photographer_styles()
        self.model_recommendations = self._init_model_recommendations()

    def _init_vfx_keywords(self) -> Dict[str, List[str]]:
        """
        Keywords VFX modernes basés sur recherche 2025
        """
        return {
            'ultra_quality': [
                'hypersharp',
                'gigapixel',
                '16k resolution',
                '8k textures',
                'ultra-detailed',
                'photorealistic',
                'hyperrealistic',
                'cinema-quality'
            ],
            'rendering_tech': [
                'Unreal Engine 5',
                'UE5 nanite',
                'UE5 lumen',
                'RTX ray tracing',
                'path tracing',
                'global illumination',
                'SSAO',
                'screen space reflections',
                'volumetric lighting',
                'subsurface scattering'
            ],
            'photography': [
                'DSLR photography',
                'professional cinematography',
                'National Geographic style',
                'landscape photography masterpiece',
                'award-winning photograph',
                'cinematic composition',
                'cinematic lighting',
                'professional color grading'
            ],
            'technical_quality': [
                'physically-based rendering',
                'PBR materials',
                'accurate color science',
                'high dynamic range',
                'HDR',
                'RAW photograph',
                'Bayer sensor',
                'zero noise',
                'perfect focus'
            ]
        }

    def _init_lighting_presets(self) -> Dict[str, str]:
        """
        Presets d'éclairage naturel réaliste
        """
        return {
            'dawn': 'soft golden hour lighting, warm orange and pink sky, long shadows, gentle rim light, low sun angle, magical morning atmosphere',
            'morning': 'bright morning sunlight, clear blue sky, crisp lighting, soft shadows, fresh atmosphere, high contrast',
            'midday': 'harsh overhead sunlight, deep blue sky, strong shadows, high contrast, intense brightness, neutral white balance',
            'afternoon': 'warm afternoon sunlight, golden tones, moderate shadows, pleasant lighting, clear atmosphere',
            'sunset': 'golden hour lighting, dramatic orange and red sky, long shadows, warm color temperature, magical atmosphere, sun rays through clouds',
            'dusk': 'blue hour lighting, deep blue sky, soft diffused light, minimal shadows, cool color temperature, serene atmosphere',
            'night': 'moonlight illumination, starry sky, deep shadows, cool blue tones, milky way visible, long exposure atmosphere',

            # Météo
            'clear': 'crystal clear atmosphere, perfect visibility, sharp details, vibrant colors, clean air',
            'cloudy': 'diffused cloud lighting, soft shadows, even illumination, dramatic cloud formations, overcast sky',
            'overcast': 'completely diffused lighting, no harsh shadows, flat even light, grey sky, muted colors',
            'foggy': 'atmospheric fog, reduced visibility, soft diffused light, ethereal atmosphere, volumetric haze',
            'stormy': 'dramatic storm lighting, dark clouds, contrasting light breaks, moody atmosphere, heavy atmosphere',
            'snowy': 'bright snow reflection, cool color temperature, soft diffused light, high contrast with snow, winter atmosphere'
        }

    def _init_photographer_styles(self) -> Dict[str, str]:
        """
        Styles de photographes professionnels célèbres
        """
        return {
            'ansel_adams': 'in the style of Ansel Adams, dramatic black and white landscape, perfect tonal range, zone system, large format camera aesthetic',
            'galen_rowell': 'in the style of Galen Rowell, vibrant mountain photography, perfect golden hour timing, dramatic natural lighting',
            'art_wolfe': 'in the style of Art Wolfe, epic landscape composition, rich colors, perfect natural light',
            'michael_kenna': 'in the style of Michael Kenna, minimalist landscape, long exposure, ethereal atmosphere, fine art photography',
            'jimmy_chin': 'in the style of Jimmy Chin, epic mountain photography, dramatic adventure aesthetic, National Geographic quality',
            'marc_adamus': 'in the style of Marc Adamus, epic landscape photography, perfect light capture, dramatic natural scenes',
            'max_rive': 'in the style of Max Rive, dramatic mountain landscapes, epic scale, perfect composition, vivid colors',

            # Styles génériques
            'nat_geo': 'National Geographic photography style, editorial quality, perfect storytelling, award-winning composition',
            'cinematic': 'cinematic landscape photography, movie-quality framing, dramatic composition, epic scale',
            'fine_art': 'fine art landscape photography, museum-quality print, perfect technical execution, artistic vision'
        }

    def _init_model_recommendations(self) -> Dict[str, Dict]:
        """
        Modèles SDXL recommandés pour terrains ultra-réalistes
        """
        return {
            'epicrealism_xl': {
                'name': 'EpicRealism XL',
                'description': 'Meilleur pour photorealism landscapes, excellent pour montagnes',
                'strengths': ['photorealistic textures', 'natural lighting', 'realistic materials'],
                'cfg_scale': 7.5,
                'steps': 40,
                'sampler': 'DPM++ 2M Karras'
            },
            'juggernaut_xl': {
                'name': 'Juggernaut XL',
                'description': 'Excellent pour landscapes dramatiques, très détaillé',
                'strengths': ['dramatic scenes', 'high detail', 'versatile'],
                'cfg_scale': 8.0,
                'steps': 45,
                'sampler': 'DPM++ 2M SDE Karras'
            },
            'realvis_xl_v4': {
                'name': 'RealVisXL V4',
                'description': 'Ultra-réaliste, idéal pour nature photography',
                'strengths': ['ultra-realistic', 'natural colors', 'perfect lighting'],
                'cfg_scale': 7.0,
                'steps': 35,
                'sampler': 'Euler a'
            },
            'protovision_xl': {
                'name': 'ProtoVision XL',
                'description': 'Très versatile, excellent pour VFX',
                'strengths': ['VFX-quality', 'consistent results', 'good for variations'],
                'cfg_scale': 7.5,
                'steps': 40,
                'sampler': 'DPM++ 2M Karras'
            },
            'dreamshaper_xl': {
                'name': 'DreamShaper XL',
                'description': 'Artistique mais réaliste, bon pour atmosphères',
                'strengths': ['atmospheric', 'artistic realism', 'good mood'],
                'cfg_scale': 8.0,
                'steps': 40,
                'sampler': 'Euler a'
            }
        }

    def generate_prompt(
        self,
        terrain_context: TerrainContext,
        camera_settings: CameraSettings,
        style: str = 'photorealistic',
        photographer_style: Optional[str] = None,
        quality_level: Literal['standard', 'high', 'ultra', 'vfx'] = 'ultra',
        include_negative: bool = True
    ) -> Dict[str, str]:
        """
        Génère un prompt complet ultra-réaliste

        Args:
            terrain_context: Contexte du terrain
            camera_settings: Paramètres caméra
            style: Style général (photorealistic, cinematic, artistic)
            photographer_style: Style d'un photographe spécifique
            quality_level: Niveau de qualité
            include_negative: Inclure negative prompt

        Returns:
            Dict avec 'positive' et 'negative' prompts
        """

        # SECTION 1: SUBJECT - Description du terrain
        subject = self._build_subject(terrain_context)

        # SECTION 2: ENVIRONMENT - Atmosphère et contexte
        environment = self._build_environment(terrain_context)

        # SECTION 3: COMPOSITION - Cadrage
        composition = self._build_composition(camera_settings)

        # SECTION 4: LIGHTING - Éclairage
        lighting = self._build_lighting(terrain_context)

        # SECTION 5: CAMERA - Technique photographique
        camera = self._build_camera_specs(camera_settings)

        # SECTION 6: PHOTOGRAPHER - Style
        photographer = ""
        if photographer_style and photographer_style in self.photographer_styles:
            photographer = self.photographer_styles[photographer_style]

        # SECTION 7: TECHNICAL - Keywords VFX
        technical = self._build_technical_keywords(quality_level)

        # Assembler le prompt
        positive_parts = [
            subject,
            environment,
            composition,
            lighting,
            camera,
            photographer,
            technical
        ]

        positive_prompt = ", ".join([p for p in positive_parts if p])

        # Negative prompt
        negative_prompt = ""
        if include_negative:
            negative_prompt = self._build_negative_prompt(quality_level)

        logger.info(f"Prompt généré: {len(positive_prompt)} caractères")

        return {
            'positive': positive_prompt,
            'negative': negative_prompt,
            'metadata': {
                'terrain_type': terrain_context.mountain_type,
                'time_of_day': terrain_context.time_of_day,
                'weather': terrain_context.weather,
                'quality_level': quality_level
            }
        }

    def _build_subject(self, context: TerrainContext) -> str:
        """Construit la description du sujet principal"""

        # Type de montagne
        mountain_descriptions = {
            'alpine': 'majestic alpine mountain range, dramatic jagged peaks, steep rocky slopes',
            'volcanic': 'imposing volcanic mountain, conical peak, dramatic elevation, volcanic rock formations',
            'rolling': 'rolling mountain landscape, gentle slopes, rounded peaks, pastoral terrain',
            'massive': 'massive mountain massif, towering peaks, grand scale, imposing presence',
            'rocky': 'rugged rocky mountain, exposed rock faces, dramatic cliff formations, wild terrain'
        }

        base = mountain_descriptions.get(context.mountain_type, 'mountain landscape')

        # Ajouter végétation si présente
        if context.vegetation_density > 0.5:
            if 'pine' in context.dominant_species or 'spruce' in context.dominant_species:
                base += ', dense coniferous forest, pine and spruce trees'
            elif 'deciduous' in context.dominant_species:
                base += ', lush deciduous forest, mixed hardwood trees'
            else:
                base += ', natural forest coverage'
        elif context.vegetation_density > 0.2:
            base += ', scattered alpine trees, sparse forest'

        # Neige
        if context.has_snow:
            base += ', snow-capped peaks, pristine white snow, glacial features'

        # Eau
        if context.has_water:
            base += ', mountain lake, crystal clear water, natural reflections'

        return base

    def _build_environment(self, context: TerrainContext) -> str:
        """Construit la description de l'environnement"""

        parts = []

        # Saison
        season_desc = {
            'spring': 'spring season, fresh green growth, melting snow, vibrant nature',
            'summer': 'summer season, lush vegetation, clear skies, vibrant colors',
            'autumn': 'autumn season, golden foliage, warm colors, fall atmosphere',
            'winter': 'winter season, heavy snow coverage, frozen landscape, icy conditions'
        }

        if context.season in season_desc:
            parts.append(season_desc[context.season])

        # Météo
        if context.weather in self.lighting_presets:
            parts.append(self.lighting_presets[context.weather])

        # Biome
        biome_desc = {
            'alpine': 'alpine tundra environment, high altitude, sparse vegetation',
            'subalpine': 'subalpine environment, tree line transition, scattered conifers',
            'montane_forest': 'montane forest environment, dense tree coverage, rich ecosystem',
            'valley_floor': 'mountain valley environment, lush vegetation, diverse flora'
        }

        if context.dominant_biome in biome_desc:
            parts.append(biome_desc[context.dominant_biome])

        return ", ".join(parts)

    def _build_composition(self, camera: CameraSettings) -> str:
        """Construit la description de la composition"""

        parts = []

        # Type de prise de vue
        shot_descriptions = {
            'aerial': 'aerial view, bird\'s eye perspective, expansive vista',
            'ground-level': 'ground level perspective, human eye view, intimate scale',
            'low-angle': 'low angle shot, dramatic upward perspective, imposing presence',
            'high-angle': 'high angle view, elevated perspective, overview composition',
            'eye-level': 'eye-level perspective, natural viewing angle, balanced composition'
        }

        if camera.shot_type in shot_descriptions:
            parts.append(shot_descriptions[camera.shot_type])

        # Composition
        composition_desc = {
            'rule-of-thirds': 'rule of thirds composition, balanced framing, professional layout',
            'centered': 'centered composition, symmetrical framing, focused subject',
            'leading-lines': 'leading lines composition, dynamic flow, visual guidance',
            'golden-ratio': 'golden ratio composition, perfect proportions, harmonious layout'
        }

        if camera.composition in composition_desc:
            parts.append(composition_desc[camera.composition])

        return ", ".join(parts)

    def _build_lighting(self, context: TerrainContext) -> str:
        """Construit la description de l'éclairage"""

        # Moment de la journée
        if context.time_of_day in self.lighting_presets:
            return self.lighting_presets[context.time_of_day]

        return "natural lighting, realistic illumination"

    def _build_camera_specs(self, camera: CameraSettings) -> str:
        """Construit les spécifications caméra"""

        parts = []

        # Focale
        parts.append(f'{camera.focal_length}mm lens')

        # Ouverture
        parts.append(f'{camera.aperture} aperture')

        # Type d'angle
        angle_desc = {
            'wide-angle': 'wide-angle lens, expansive field of view',
            'normal': 'normal lens, natural perspective',
            'telephoto': 'telephoto lens, compressed perspective',
            'ultra-wide': 'ultra-wide lens, dramatic perspective'
        }

        if camera.angle in angle_desc:
            parts.append(angle_desc[camera.angle])

        # Ajouts techniques
        parts.extend([
            'professional DSLR camera',
            'full-frame sensor',
            'perfect focus',
            'optimal exposure'
        ])

        return ", ".join(parts)

    def _build_technical_keywords(self, quality_level: str) -> str:
        """Construit les keywords techniques VFX"""

        keywords = []

        if quality_level in ['ultra', 'vfx']:
            keywords.extend(self.vfx_quality_keywords['ultra_quality'][:4])
            keywords.extend(self.vfx_quality_keywords['rendering_tech'][:6])
            keywords.extend(self.vfx_quality_keywords['photography'][:4])
            keywords.extend(self.vfx_quality_keywords['technical_quality'][:4])
        elif quality_level == 'high':
            keywords.extend(self.vfx_quality_keywords['ultra_quality'][:2])
            keywords.extend(self.vfx_quality_keywords['rendering_tech'][:3])
            keywords.extend(self.vfx_quality_keywords['photography'][:2])
            keywords.extend(self.vfx_quality_keywords['technical_quality'][:2])
        else:  # standard
            keywords.append('photorealistic')
            keywords.append('8k resolution')
            keywords.append('professional photography')

        return ", ".join(keywords)

    def _build_negative_prompt(self, quality_level: str) -> str:
        """Construit le negative prompt"""

        # Défauts communs à éviter
        common_negatives = [
            'blurry', 'out of focus', 'low quality', 'low resolution',
            'pixelated', 'jpeg artifacts', 'compression artifacts',
            'noise', 'grain', 'distorted', 'deformed',
            'unrealistic', 'fake', 'artificial', 'rendered look',
            'oversaturated', 'undersaturated', 'wrong colors',
            'bad lighting', 'flat lighting', 'overexposed', 'underexposed',
            'amateur', 'snapshot', 'phone camera'
        ]

        # Défauts spécifiques terrains
        terrain_negatives = [
            'flat terrain', 'unnaturally smooth', 'repetitive patterns',
            'tiling texture', 'obvious repetition', 'symmetrical',
            'floating objects', 'disconnected elements',
            'cartoon', 'anime', 'illustration', 'painting', 'drawing',
            'text', 'watermark', 'signature', 'logo'
        ]

        # Défauts géométriques
        geometry_negatives = [
            'distorted perspective', 'wrong proportions', 'warped',
            'stretched', 'squashed', 'deformed geometry'
        ]

        if quality_level in ['ultra', 'vfx']:
            all_negatives = common_negatives + terrain_negatives + geometry_negatives
        elif quality_level == 'high':
            all_negatives = common_negatives + terrain_negatives
        else:
            all_negatives = common_negatives[:10]

        return ", ".join(all_negatives)

    def create_preset_prompts(self) -> Dict[str, Dict]:
        """
        Crée des presets de prompts professionnels prêts à l'emploi
        """

        presets = {}

        # PRESET 1: Epic Alpine Sunset
        presets['epic_alpine_sunset'] = {
            'name': 'Epic Alpine Sunset',
            'description': 'Dramatic alpine mountain at golden hour',
            'terrain_context': TerrainContext(
                mountain_type='alpine',
                elevation_range=(0.5, 1.0),
                dominant_biome='subalpine',
                vegetation_density=0.3,
                dominant_species=['pine', 'spruce'],
                has_snow=True,
                has_water=False,
                season='summer',
                time_of_day='sunset',
                weather='clear'
            ),
            'camera_settings': CameraSettings(
                focal_length=35,
                aperture='f/11',
                angle='wide-angle',
                shot_type='ground-level',
                composition='rule-of-thirds'
            ),
            'photographer_style': 'galen_rowell',
            'quality_level': 'vfx'
        }

        # PRESET 2: Misty Mountain Morning
        presets['misty_morning'] = {
            'name': 'Misty Mountain Morning',
            'description': 'Ethereal foggy mountain landscape',
            'terrain_context': TerrainContext(
                mountain_type='massive',
                elevation_range=(0.3, 0.8),
                dominant_biome='montane_forest',
                vegetation_density=0.7,
                dominant_species=['fir', 'spruce'],
                has_snow=False,
                has_water=True,
                season='autumn',
                time_of_day='dawn',
                weather='foggy'
            ),
            'camera_settings': CameraSettings(
                focal_length=85,
                aperture='f/5.6',
                angle='telephoto',
                shot_type='high-angle',
                composition='centered'
            ),
            'photographer_style': 'michael_kenna',
            'quality_level': 'ultra'
        }

        # PRESET 3: Dramatic Storm Peak
        presets['storm_peak'] = {
            'name': 'Dramatic Storm Peak',
            'description': 'Stormy dramatic mountain peak',
            'terrain_context': TerrainContext(
                mountain_type='rocky',
                elevation_range=(0.6, 1.0),
                dominant_biome='alpine',
                vegetation_density=0.1,
                dominant_species=['pine'],
                has_snow=True,
                has_water=False,
                season='winter',
                time_of_day='afternoon',
                weather='stormy'
            ),
            'camera_settings': CameraSettings(
                focal_length=24,
                aperture='f/8',
                angle='ultra-wide',
                shot_type='low-angle',
                composition='leading-lines'
            ),
            'photographer_style': 'jimmy_chin',
            'quality_level': 'vfx'
        }

        # PRESET 4: Peaceful Valley
        presets['peaceful_valley'] = {
            'name': 'Peaceful Valley',
            'description': 'Serene mountain valley landscape',
            'terrain_context': TerrainContext(
                mountain_type='rolling',
                elevation_range=(0.1, 0.5),
                dominant_biome='valley_floor',
                vegetation_density=0.8,
                dominant_species=['deciduous', 'pine'],
                has_snow=False,
                has_water=True,
                season='spring',
                time_of_day='morning',
                weather='clear'
            ),
            'camera_settings': CameraSettings(
                focal_length=50,
                aperture='f/8',
                angle='normal',
                shot_type='eye-level',
                composition='golden-ratio'
            ),
            'photographer_style': 'nat_geo',
            'quality_level': 'ultra'
        }

        # PRESET 5: Volcanic Majesty
        presets['volcanic_majesty'] = {
            'name': 'Volcanic Majesty',
            'description': 'Imposing volcanic mountain peak',
            'terrain_context': TerrainContext(
                mountain_type='volcanic',
                elevation_range=(0.7, 1.0),
                dominant_biome='alpine',
                vegetation_density=0.2,
                dominant_species=['pine'],
                has_snow=True,
                has_water=False,
                season='summer',
                time_of_day='midday',
                weather='clear'
            ),
            'camera_settings': CameraSettings(
                focal_length=70,
                aperture='f/11',
                angle='normal',
                shot_type='ground-level',
                composition='centered'
            ),
            'photographer_style': 'ansel_adams',
            'quality_level': 'vfx'
        }

        return presets

    def get_recommended_model(self, style: str = 'photorealistic') -> Dict:
        """
        Retourne le modèle SDXL recommandé selon le style
        """

        recommendations = {
            'photorealistic': 'epicrealism_xl',
            'dramatic': 'juggernaut_xl',
            'natural': 'realvis_xl_v4',
            'vfx': 'protovision_xl',
            'artistic': 'dreamshaper_xl'
        }

        model_key = recommendations.get(style, 'epicrealism_xl')
        return self.model_recommendations[model_key]

    def auto_generate_from_heightmap(
        self,
        heightmap: np.ndarray,
        biome_map: Optional[np.ndarray] = None,
        vegetation_density_map: Optional[np.ndarray] = None,
        time_of_day: str = 'sunset',
        weather: str = 'clear',
        season: str = 'summer'
    ) -> Dict[str, str]:
        """
        Génère automatiquement un prompt optimal à partir d'une heightmap

        Args:
            heightmap: Heightmap (0-1)
            biome_map: Carte de biomes (optionnel)
            vegetation_density_map: Carte de densité végétation (optionnel)
            time_of_day: Moment de la journée
            weather: Météo
            season: Saison

        Returns:
            Prompt complet
        """

        # Analyser la heightmap
        elevation_mean = np.mean(heightmap)
        elevation_max = np.max(heightmap)
        elevation_min = np.min(heightmap)
        elevation_range = (elevation_min, elevation_max)

        # Déterminer le type de montagne
        if elevation_max > 0.8 and elevation_mean > 0.5:
            mountain_type = 'alpine'
        elif elevation_max > 0.9:
            mountain_type = 'rocky'
        elif elevation_max < 0.5:
            mountain_type = 'rolling'
        else:
            mountain_type = 'massive'

        # Analyser biome dominant
        dominant_biome = 'montane_forest'
        if biome_map is not None:
            from core.vegetation.biome_classifier import BiomeType
            unique, counts = np.unique(biome_map, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]

            biome_names = {
                BiomeType.ROCKY_CLIFF: 'alpine',
                BiomeType.ALPINE: 'alpine',
                BiomeType.SUBALPINE: 'subalpine',
                BiomeType.MONTANE_FOREST: 'montane_forest',
                BiomeType.VALLEY_FLOOR: 'valley_floor'
            }
            dominant_biome = biome_names.get(dominant_idx, 'montane_forest')

        # Analyser végétation
        vegetation_density = 0.5
        if vegetation_density_map is not None:
            vegetation_density = np.mean(vegetation_density_map)

        # Déterminer espèces dominantes
        dominant_species = []
        if elevation_mean > 0.6:
            dominant_species = ['pine', 'spruce']
        elif elevation_mean > 0.3:
            dominant_species = ['pine', 'spruce', 'fir']
        else:
            dominant_species = ['deciduous', 'pine']

        # Neige et eau
        has_snow = elevation_max > 0.7 or season == 'winter'
        has_water = elevation_min < 0.15

        # Créer contexte
        terrain_context = TerrainContext(
            mountain_type=mountain_type,
            elevation_range=elevation_range,
            dominant_biome=dominant_biome,
            vegetation_density=vegetation_density,
            dominant_species=dominant_species,
            has_snow=has_snow,
            has_water=has_water,
            season=season,
            time_of_day=time_of_day,
            weather=weather
        )

        # Paramètres caméra par défaut (dramatique)
        camera_settings = CameraSettings(
            focal_length=35,
            aperture='f/11',
            angle='wide-angle',
            shot_type='ground-level',
            composition='rule-of-thirds'
        )

        # Générer prompt
        return self.generate_prompt(
            terrain_context=terrain_context,
            camera_settings=camera_settings,
            photographer_style='nat_geo',
            quality_level='vfx'
        )


# Fonction utilitaire pour tester
def test_generator():
    """Test du générateur de prompts"""

    generator = VFXPromptGenerator()

    # Test avec preset
    presets = generator.create_preset_prompts()
    preset = presets['epic_alpine_sunset']

    result = generator.generate_prompt(
        terrain_context=preset['terrain_context'],
        camera_settings=preset['camera_settings'],
        photographer_style=preset['photographer_style'],
        quality_level=preset['quality_level']
    )

    print("=" * 80)
    print("POSITIVE PROMPT:")
    print("=" * 80)
    print(result['positive'])
    print()
    print("=" * 80)
    print("NEGATIVE PROMPT:")
    print("=" * 80)
    print(result['negative'])
    print()
    print("=" * 80)
    print("METADATA:")
    print("=" * 80)
    for key, value in result['metadata'].items():
        print(f"{key}: {value}")
    print()

    # Recommandation modèle
    model = generator.get_recommended_model('photorealistic')
    print("=" * 80)
    print("RECOMMENDED MODEL:")
    print("=" * 80)
    print(f"Model: {model['name']}")
    print(f"Description: {model['description']}")
    print(f"CFG Scale: {model['cfg_scale']}")
    print(f"Steps: {model['steps']}")
    print(f"Sampler: {model['sampler']}")
    print()


if __name__ == "__main__":
    test_generator()
