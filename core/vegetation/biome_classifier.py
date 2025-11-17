"""
Classificateur de biomes basé sur altitude, pente, orientation et moisture
Détermine où placer quels types de végétation

Biomes supportés:
- Alpine (haute altitude, peu de végétation)
- Subalpine (pins, épicéas dispersés)
- Montane Forest (forêt dense de conifères)
- Valley Floor (forêt mixte, végétation dense)
- Rocky/Cliff (pas de végétation, rochers exposés)
"""

import numpy as np
from typing import Tuple, Dict
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class BiomeType(IntEnum):
    """Types de biomes"""
    ROCKY_CLIFF = 0      # Falaises, rochers exposés
    ALPINE = 1           # Haute altitude, toundra alpine
    SUBALPINE = 2        # Limite des arbres, pins dispersés
    MONTANE_FOREST = 3   # Forêt de conifères
    VALLEY_FLOOR = 4     # Fond de vallée, forêt mixte
    WATER = 5            # Lacs, rivières (pas de végétation)


class BiomeClassifier:
    """
    Classifie chaque pixel du terrain en biome
    Basé sur des règles écologiques réalistes
    """

    def __init__(self, width: int, height: int):
        """
        Args:
            width: Largeur de la heightmap
            height: Hauteur de la heightmap
        """
        self.width = width
        self.height = height

    def classify(
        self,
        heightmap: np.ndarray,
        moisture_map: Optional[np.ndarray] = None,
        temperature_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classifie le terrain en biomes

        Args:
            heightmap: Heightmap normalisée (0-1)
            moisture_map: Map d'humidité optionnelle (0-1)
            temperature_map: Map de température optionnelle (0-1)

        Returns:
            Biome map (array d'int avec BiomeType)
        """
        logger.info("Classification des biomes...")

        biome_map = np.zeros((self.height, self.width), dtype=np.int32)

        # Calculer pente et orientation
        slope_map = self._calculate_slope(heightmap)
        aspect_map = self._calculate_aspect(heightmap)

        # Générer moisture si non fourni
        if moisture_map is None:
            moisture_map = self._generate_moisture_map(heightmap, slope_map, aspect_map)

        # Générer température si non fourni (basé sur altitude)
        if temperature_map is None:
            temperature_map = self._generate_temperature_map(heightmap)

        # Classifier chaque pixel
        biome_map = self._apply_classification_rules(
            heightmap,
            slope_map,
            aspect_map,
            moisture_map,
            temperature_map
        )

        logger.info(f"Biomes classifiés: "
                   f"Rocky={np.sum(biome_map==BiomeType.ROCKY_CLIFF)}, "
                   f"Alpine={np.sum(biome_map==BiomeType.ALPINE)}, "
                   f"Subalpine={np.sum(biome_map==BiomeType.SUBALPINE)}, "
                   f"Montane={np.sum(biome_map==BiomeType.MONTANE_FOREST)}, "
                   f"Valley={np.sum(biome_map==BiomeType.VALLEY_FLOOR)}")

        return biome_map

    def _calculate_slope(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Calcule la pente en chaque point (0-1+)
        0 = plat, 1 = 45°, >1 = plus raide
        """
        grad_y, grad_x = np.gradient(heightmap)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        return slope

    def _calculate_aspect(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Calcule l'orientation de la pente (0-360°)
        0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest
        """
        grad_y, grad_x = np.gradient(heightmap)
        aspect = np.arctan2(grad_y, grad_x) * 180 / np.pi
        aspect = (aspect + 360) % 360  # 0-360
        return aspect

    def _generate_moisture_map(
        self,
        heightmap: np.ndarray,
        slope_map: np.ndarray,
        aspect_map: np.ndarray
    ) -> np.ndarray:
        """
        Génère une map d'humidité basée sur:
        - Altitude (bas = plus humide)
        - Pente (plat = accumule eau)
        - Orientation (nord = plus humide)
        """
        moisture = np.zeros_like(heightmap)

        # Altitude (inversé: bas = humide)
        altitude_factor = 1.0 - heightmap
        moisture += altitude_factor * 0.4

        # Pente (plat = accumule eau)
        slope_factor = 1.0 - np.clip(slope_map * 2, 0, 1)
        moisture += slope_factor * 0.3

        # Orientation (nord = plus humide que sud)
        # Nord = 0°, Sud = 180°
        aspect_factor = np.cos(aspect_map * np.pi / 180)  # -1 (sud) à 1 (nord)
        aspect_factor = (aspect_factor + 1) / 2  # 0 à 1
        moisture += aspect_factor * 0.2

        # Ajouter bruit Perlin pour variation
        import noise as pnoise
        noise_layer = np.zeros_like(heightmap)
        for y in range(self.height):
            for x in range(self.width):
                noise_layer[y, x] = pnoise.pnoise2(
                    x / 100.0,
                    y / 100.0,
                    octaves=3
                )
        noise_layer = (noise_layer - noise_layer.min()) / (noise_layer.max() - noise_layer.min())
        moisture += noise_layer * 0.1

        # Normaliser
        moisture = np.clip(moisture, 0, 1)

        return moisture

    def _generate_temperature_map(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Génère une map de température basée sur altitude
        Lapse rate: température diminue avec altitude
        """
        # Température décroît avec altitude
        # 1.0 = chaud (bas), 0.0 = froid (haut)
        temperature = 1.0 - heightmap

        return temperature

    def _apply_classification_rules(
        self,
        heightmap: np.ndarray,
        slope_map: np.ndarray,
        aspect_map: np.ndarray,
        moisture_map: np.ndarray,
        temperature_map: np.ndarray
    ) -> np.ndarray:
        """
        Applique les règles de classification écologiques
        """
        biome_map = np.zeros((self.height, self.width), dtype=np.int32)

        # RÈGLE 1: Pentes raides = Rocky Cliff (pas de végétation)
        steep_threshold = 0.8  # ~38°
        biome_map[slope_map > steep_threshold] = BiomeType.ROCKY_CLIFF

        # RÈGLE 2: Haute altitude = Alpine (toundra)
        alpine_threshold = 0.75
        alpine_mask = (heightmap > alpine_threshold) & (slope_map <= steep_threshold)
        biome_map[alpine_mask] = BiomeType.ALPINE

        # RÈGLE 3: Subalpine (transition, arbres dispersés)
        subalpine_threshold_low = 0.6
        subalpine_threshold_high = 0.75
        subalpine_mask = (heightmap > subalpine_threshold_low) & \
                        (heightmap <= subalpine_threshold_high) & \
                        (slope_map <= steep_threshold)
        biome_map[subalpine_mask] = BiomeType.SUBALPINE

        # RÈGLE 4: Montane Forest (forêt de conifères)
        montane_threshold_low = 0.3
        montane_threshold_high = 0.6
        montane_mask = (heightmap > montane_threshold_low) & \
                      (heightmap <= montane_threshold_high) & \
                      (slope_map <= steep_threshold)
        biome_map[montane_mask] = BiomeType.MONTANE_FOREST

        # RÈGLE 5: Valley Floor (fond de vallée)
        valley_mask = (heightmap <= montane_threshold_low) & \
                     (slope_map <= steep_threshold)
        biome_map[valley_mask] = BiomeType.VALLEY_FLOOR

        # RÈGLE 6: Zones très plates et basses = potentiellement eau
        water_threshold = 0.05
        very_flat = slope_map < 0.05
        very_low = heightmap < water_threshold
        water_mask = very_flat & very_low
        biome_map[water_mask] = BiomeType.WATER

        return biome_map

    def get_biome_info(self, biome_type: BiomeType) -> Dict:
        """
        Retourne les informations sur un biome
        (végétation, caractéristiques, etc.)
        """
        biome_info = {
            BiomeType.ROCKY_CLIFF: {
                'name': 'Rocky Cliff',
                'vegetation_density': 0.0,
                'tree_species': [],
                'color': (128, 128, 128),  # Gris roche
                'description': 'Falaises rocheuses, pas de végétation'
            },
            BiomeType.ALPINE: {
                'name': 'Alpine Tundra',
                'vegetation_density': 0.1,
                'tree_species': [],  # Pas d'arbres, seulement herbes/mousses
                'color': (200, 180, 160),  # Beige/brun clair
                'description': 'Toundra alpine, végétation basse'
            },
            BiomeType.SUBALPINE: {
                'name': 'Subalpine',
                'vegetation_density': 0.3,
                'tree_species': ['pine', 'spruce'],
                'color': (100, 140, 100),  # Vert foncé clairsemé
                'description': 'Limite des arbres, pins et épicéas dispersés'
            },
            BiomeType.MONTANE_FOREST: {
                'name': 'Montane Forest',
                'vegetation_density': 0.7,
                'tree_species': ['pine', 'spruce', 'fir'],
                'color': (60, 100, 60),  # Vert foncé dense
                'description': 'Forêt de conifères dense'
            },
            BiomeType.VALLEY_FLOOR: {
                'name': 'Valley Floor',
                'vegetation_density': 0.8,
                'tree_species': ['pine', 'spruce', 'fir', 'deciduous'],
                'color': (80, 120, 80),  # Vert moyen
                'description': 'Forêt mixte dense, fond de vallée'
            },
            BiomeType.WATER: {
                'name': 'Water',
                'vegetation_density': 0.0,
                'tree_species': [],
                'color': (100, 150, 200),  # Bleu
                'description': 'Eau (lac, rivière)'
            }
        }

        return biome_info.get(biome_type, biome_info[BiomeType.ALPINE])
