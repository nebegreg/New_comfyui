"""
PBR Splatmap Generator - Système de matériaux multicouches avancé

Génère des splatmaps pour blend de matériaux PBR réalistes basés sur:
- Altitude (neige en haut, herbe en bas)
- Pente (roche sur falaises, herbe sur plat)
- Orientation (mousse au nord, sec au sud)
- Humidité (zones humides vs sèches)
- Weathering (altération, lichens, mousse)

Compatible avec:
- Unreal Engine 5 (Landscape Material)
- Unity URP/HDRP (Terrain Shader)
- Blender (Shader Nodes)
- Substance Designer

Format de sortie:
- Splatmap multicouche (RGBA per layer)
- Jusqu'à 8 matériaux (2 textures RGBA)
- Export PNG ou EXR 32-bit

Références:
- Unreal Engine 5 Landscape Material Best Practices
- Unity Terrain Splatmap Documentation
- Physically-Based Material Layering (SIGGRAPH 2017)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaterialLayer:
    """Définition d'une couche de matériau PBR"""
    name: str
    id: int  # 0-7

    # Conditions de placement
    altitude_min: float  # 0-1
    altitude_max: float  # 0-1
    altitude_optimal: float  # 0-1
    altitude_falloff: float  # Distance de transition

    slope_min: float  # 0-1 (0=plat, 1=vertical)
    slope_max: float
    slope_optimal: float
    slope_falloff: float

    # Orientation (aspect) - préférence
    # None = pas de préférence, sinon angle en degrés (0=Nord, 180=Sud)
    aspect_preference: Optional[float] = None
    aspect_tolerance: float = 90.0  # Degrés de tolérance

    # Humidité
    moisture_min: float = 0.0
    moisture_max: float = 1.0
    moisture_optimal: float = 0.5
    moisture_falloff: float = 0.3

    # Weathering/altération
    weathering_strength: float = 0.0  # 0-1, strength de l'effet weathering

    # Blending
    blend_sharpness: float = 0.5  # 0=doux, 1=net
    noise_scale: float = 10.0  # Échelle du bruit de variation
    noise_strength: float = 0.1  # Force du bruit

    # PBR Properties (pour info/export)
    base_color: Tuple[int, int, int] = (128, 128, 128)
    roughness: float = 0.5
    metallic: float = 0.0

    # Description
    description: str = ""


class PBRSplatmapGenerator:
    """
    Générateur de splatmaps multicouches pour matériaux PBR
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.materials = self._create_default_materials()

    def _create_default_materials(self) -> Dict[str, MaterialLayer]:
        """
        Crée les matériaux par défaut pour terrains de montagne
        Inspiré par la réalité: distribution naturelle des matériaux
        """

        materials = {}

        # LAYER 0: SNOW - Neige haute altitude
        materials['snow'] = MaterialLayer(
            name='snow',
            id=0,
            altitude_min=0.7,
            altitude_max=1.0,
            altitude_optimal=0.85,
            altitude_falloff=0.15,
            slope_min=0.0,
            slope_max=0.6,  # Pas de neige sur falaises raides
            slope_optimal=0.1,
            slope_falloff=0.2,
            moisture_min=0.0,
            moisture_max=1.0,
            moisture_optimal=0.5,
            moisture_falloff=0.5,
            blend_sharpness=0.6,
            noise_scale=20.0,
            noise_strength=0.15,
            base_color=(250, 250, 255),
            roughness=0.2,
            metallic=0.0,
            description="Fresh snow, high altitude, pristine white"
        )

        # LAYER 1: ROCK_CLIFF - Roche exposée (falaises)
        materials['rock_cliff'] = MaterialLayer(
            name='rock_cliff',
            id=1,
            altitude_min=0.0,
            altitude_max=1.0,
            altitude_optimal=0.6,
            altitude_falloff=0.4,
            slope_min=0.5,  # Falaises raides uniquement
            slope_max=1.5,
            slope_optimal=0.8,
            slope_falloff=0.2,
            moisture_min=0.0,
            moisture_max=0.5,
            moisture_optimal=0.2,
            moisture_falloff=0.3,
            blend_sharpness=0.8,  # Transitions nettes pour falaises
            noise_scale=5.0,
            noise_strength=0.2,
            base_color=(120, 115, 110),
            roughness=0.7,
            metallic=0.0,
            description="Exposed rock cliff faces, steep terrain"
        )

        # LAYER 2: ROCK_GROUND - Roche de sol (haute altitude)
        materials['rock_ground'] = MaterialLayer(
            name='rock_ground',
            id=2,
            altitude_min=0.5,
            altitude_max=0.85,
            altitude_optimal=0.65,
            altitude_falloff=0.15,
            slope_min=0.2,
            slope_max=0.7,
            slope_optimal=0.4,
            slope_falloff=0.2,
            moisture_min=0.0,
            moisture_max=0.6,
            moisture_optimal=0.3,
            moisture_falloff=0.3,
            blend_sharpness=0.5,
            noise_scale=15.0,
            noise_strength=0.2,
            base_color=(130, 120, 105),
            roughness=0.8,
            metallic=0.0,
            description="Rocky ground, alpine zone, scattered rocks"
        )

        # LAYER 3: ALPINE_GRASS - Herbe alpine
        materials['alpine_grass'] = MaterialLayer(
            name='alpine_grass',
            id=3,
            altitude_min=0.5,
            altitude_max=0.75,
            altitude_optimal=0.6,
            altitude_falloff=0.15,
            slope_min=0.0,
            slope_max=0.4,
            slope_optimal=0.15,
            slope_falloff=0.15,
            moisture_min=0.3,
            moisture_max=0.8,
            moisture_optimal=0.6,
            moisture_falloff=0.2,
            aspect_preference=0.0,  # Préfère nord (plus humide)
            aspect_tolerance=120.0,
            blend_sharpness=0.4,
            noise_scale=25.0,
            noise_strength=0.15,
            base_color=(140, 155, 110),
            roughness=0.9,
            metallic=0.0,
            description="Alpine grass, high altitude meadows"
        )

        # LAYER 4: FOREST_GRASS - Herbe de forêt
        materials['forest_grass'] = MaterialLayer(
            name='forest_grass',
            id=4,
            altitude_min=0.2,
            altitude_max=0.6,
            altitude_optimal=0.4,
            altitude_falloff=0.2,
            slope_min=0.0,
            slope_max=0.35,
            slope_optimal=0.1,
            slope_falloff=0.15,
            moisture_min=0.4,
            moisture_max=1.0,
            moisture_optimal=0.7,
            moisture_falloff=0.2,
            blend_sharpness=0.3,
            noise_scale=30.0,
            noise_strength=0.2,
            base_color=(90, 120, 70),
            roughness=0.95,
            metallic=0.0,
            description="Lush forest floor grass, rich soil"
        )

        # LAYER 5: DIRT - Terre/sol
        materials['dirt'] = MaterialLayer(
            name='dirt',
            id=5,
            altitude_min=0.0,
            altitude_max=0.5,
            altitude_optimal=0.25,
            altitude_falloff=0.25,
            slope_min=0.0,
            slope_max=0.45,
            slope_optimal=0.2,
            slope_falloff=0.2,
            moisture_min=0.2,
            moisture_max=0.7,
            moisture_optimal=0.45,
            moisture_falloff=0.25,
            blend_sharpness=0.4,
            noise_scale=20.0,
            noise_strength=0.25,
            base_color=(100, 80, 60),
            roughness=0.85,
            metallic=0.0,
            description="Exposed dirt and soil, transition zones"
        )

        # LAYER 6: MOSS_WET - Mousse (zones humides nord)
        materials['moss_wet'] = MaterialLayer(
            name='moss_wet',
            id=6,
            altitude_min=0.0,
            altitude_max=0.65,
            altitude_optimal=0.35,
            altitude_falloff=0.3,
            slope_min=0.15,
            slope_max=0.65,
            slope_optimal=0.35,
            slope_falloff=0.2,
            moisture_min=0.6,
            moisture_max=1.0,
            moisture_optimal=0.85,
            moisture_falloff=0.2,
            aspect_preference=0.0,  # Nord (humide)
            aspect_tolerance=80.0,
            weathering_strength=0.7,
            blend_sharpness=0.5,
            noise_scale=12.0,
            noise_strength=0.3,
            base_color=(60, 90, 50),
            roughness=0.8,
            metallic=0.0,
            description="Wet moss on north-facing rocks, humid areas"
        )

        # LAYER 7: SCREE - Éboulis (pentes moyennes)
        materials['scree'] = MaterialLayer(
            name='scree',
            id=7,
            altitude_min=0.4,
            altitude_max=0.8,
            altitude_optimal=0.6,
            altitude_falloff=0.2,
            slope_min=0.3,
            slope_max=0.7,
            slope_optimal=0.5,
            slope_falloff=0.15,
            moisture_min=0.0,
            moisture_max=0.4,
            moisture_optimal=0.15,
            moisture_falloff=0.2,
            blend_sharpness=0.6,
            noise_scale=8.0,
            noise_strength=0.35,
            base_color=(115, 110, 100),
            roughness=0.75,
            metallic=0.0,
            description="Scree slopes, loose rocks, debris"
        )

        return materials

    def generate_splatmap(
        self,
        heightmap: np.ndarray,
        moisture_map: Optional[np.ndarray] = None,
        custom_materials: Optional[Dict[str, MaterialLayer]] = None,
        apply_weathering: bool = True,
        smooth_transitions: bool = True,
        smooth_sigma: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère les splatmaps à partir du terrain

        Args:
            heightmap: Heightmap normalisée (0-1)
            moisture_map: Carte d'humidité (0-1), optionnel
            custom_materials: Matériaux personnalisés, sinon utilise défaut
            apply_weathering: Appliquer effets weathering
            smooth_transitions: Lisser les transitions entre matériaux
            smooth_sigma: Force du lissage (si smooth_transitions=True)

        Returns:
            Tuple de (splatmap1, splatmap2) où chaque map est RGBA (4 channels)
            splatmap1: Layers 0-3 (R, G, B, A)
            splatmap2: Layers 4-7 (R, G, B, A)
        """

        logger.info("Génération splatmaps PBR...")

        # Utiliser matériaux custom ou défaut
        materials = custom_materials if custom_materials else self.materials

        # Calculer maps dérivées
        slope_map = self._calculate_slope(heightmap)
        aspect_map = self._calculate_aspect(heightmap)

        # Générer moisture si non fourni
        if moisture_map is None:
            moisture_map = self._generate_moisture_map(heightmap, slope_map, aspect_map)

        # Calculer weight pour chaque matériau
        material_weights = {}

        for mat_name, material in materials.items():
            weight = self._calculate_material_weight(
                material,
                heightmap,
                slope_map,
                aspect_map,
                moisture_map
            )
            material_weights[mat_name] = weight

        # Appliquer weathering
        if apply_weathering:
            material_weights = self._apply_weathering(
                material_weights,
                materials,
                heightmap,
                slope_map,
                moisture_map
            )

        # Normaliser pour que somme = 1.0
        material_weights = self._normalize_weights(material_weights)

        # Smooth transitions
        if smooth_transitions:
            for mat_name in material_weights.keys():
                material_weights[mat_name] = gaussian_filter(
                    material_weights[mat_name],
                    sigma=smooth_sigma
                )
            # Re-normaliser après smooth
            material_weights = self._normalize_weights(material_weights)

        # Créer splatmaps (2 textures RGBA pour 8 layers)
        splatmap1 = self._pack_splatmap(material_weights, materials, layers=[0, 1, 2, 3])
        splatmap2 = self._pack_splatmap(material_weights, materials, layers=[4, 5, 6, 7])

        logger.info("Splatmaps générées avec succès")

        return splatmap1, splatmap2

    def _calculate_slope(self, heightmap: np.ndarray) -> np.ndarray:
        """Calcule la pente (0-1+)"""
        grad_y, grad_x = np.gradient(heightmap)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        return slope

    def _calculate_aspect(self, heightmap: np.ndarray) -> np.ndarray:
        """Calcule l'orientation de la pente (0-360°)"""
        grad_y, grad_x = np.gradient(heightmap)
        aspect = np.arctan2(grad_y, grad_x) * 180 / np.pi
        aspect = (aspect + 360) % 360
        return aspect

    def _generate_moisture_map(
        self,
        heightmap: np.ndarray,
        slope_map: np.ndarray,
        aspect_map: np.ndarray
    ) -> np.ndarray:
        """Génère une moisture map basique"""
        moisture = np.zeros_like(heightmap)

        # Bas = plus humide
        moisture += (1.0 - heightmap) * 0.4

        # Plat accumule eau
        slope_factor = 1.0 - np.clip(slope_map * 2, 0, 1)
        moisture += slope_factor * 0.3

        # Nord = plus humide
        aspect_factor = np.cos(aspect_map * np.pi / 180)
        aspect_factor = (aspect_factor + 1) / 2
        moisture += aspect_factor * 0.2

        # Bruit
        noise = np.random.random((self.height, self.width)) * 0.1
        moisture += noise

        moisture = np.clip(moisture, 0, 1)

        return moisture

    def _calculate_material_weight(
        self,
        material: MaterialLayer,
        heightmap: np.ndarray,
        slope_map: np.ndarray,
        aspect_map: np.ndarray,
        moisture_map: np.ndarray
    ) -> np.ndarray:
        """
        Calcule le poids d'un matériau pour chaque pixel
        Retourne map de weights (0-1)
        """

        weight = np.ones((self.height, self.width), dtype=np.float32)

        # ALTITUDE
        altitude_weight = self._evaluate_range(
            heightmap,
            material.altitude_min,
            material.altitude_max,
            material.altitude_optimal,
            material.altitude_falloff
        )
        weight *= altitude_weight

        # SLOPE
        slope_weight = self._evaluate_range(
            slope_map,
            material.slope_min,
            material.slope_max,
            material.slope_optimal,
            material.slope_falloff
        )
        weight *= slope_weight

        # MOISTURE
        moisture_weight = self._evaluate_range(
            moisture_map,
            material.moisture_min,
            material.moisture_max,
            material.moisture_optimal,
            material.moisture_falloff
        )
        weight *= moisture_weight

        # ASPECT (orientation)
        if material.aspect_preference is not None:
            aspect_weight = self._evaluate_aspect(
                aspect_map,
                material.aspect_preference,
                material.aspect_tolerance
            )
            weight *= aspect_weight

        # NOISE pour variation
        if material.noise_strength > 0:
            noise = self._generate_perlin_noise(material.noise_scale)
            # Noise centré sur 1.0 avec variation
            noise = 1.0 + (noise - 0.5) * material.noise_strength
            weight *= noise

        # Appliquer sharpness
        if material.blend_sharpness > 0.5:
            # Rendre transitions plus nettes
            power = 1.0 + (material.blend_sharpness - 0.5) * 4.0
            weight = np.power(weight, power)

        weight = np.clip(weight, 0, 1)

        return weight

    def _evaluate_range(
        self,
        value_map: np.ndarray,
        min_val: float,
        max_val: float,
        optimal_val: float,
        falloff: float
    ) -> np.ndarray:
        """
        Évalue si valeurs sont dans range acceptable
        Retourne weight map (0-1)
        """

        weight = np.ones_like(value_map)

        # Hors range = 0
        weight[value_map < min_val] = 0.0
        weight[value_map > max_val] = 0.0

        # Transition douce avec falloff
        # En dessous de optimal
        below_optimal = (value_map < optimal_val) & (value_map >= min_val)
        if np.any(below_optimal):
            distance = optimal_val - value_map[below_optimal]
            max_distance = optimal_val - min_val
            if max_distance > 0:
                transition = 1.0 - np.clip(distance / (falloff * max_distance), 0, 1)
                transition = np.power(transition, 2)  # Courbe douce
                weight[below_optimal] = transition

        # Au dessus de optimal
        above_optimal = (value_map > optimal_val) & (value_map <= max_val)
        if np.any(above_optimal):
            distance = value_map[above_optimal] - optimal_val
            max_distance = max_val - optimal_val
            if max_distance > 0:
                transition = 1.0 - np.clip(distance / (falloff * max_distance), 0, 1)
                transition = np.power(transition, 2)
                weight[above_optimal] = transition

        return weight

    def _evaluate_aspect(
        self,
        aspect_map: np.ndarray,
        preferred_aspect: float,
        tolerance: float
    ) -> np.ndarray:
        """
        Évalue préférence d'orientation
        preferred_aspect: 0-360 degrés
        tolerance: degrés de tolérance
        """

        # Calculer différence angulaire
        diff = np.abs(aspect_map - preferred_aspect)

        # Gérer wrap-around (360° = 0°)
        diff = np.minimum(diff, 360 - diff)

        # Weight basé sur différence
        weight = np.ones_like(aspect_map)
        weight = 1.0 - np.clip(diff / tolerance, 0, 1)
        weight = np.power(weight, 2)  # Courbe douce

        return weight

    def _generate_perlin_noise(self, scale: float) -> np.ndarray:
        """Génère bruit Perlin pour variation"""
        try:
            import noise as pnoise

            noise_map = np.zeros((self.height, self.width))

            for y in range(self.height):
                for x in range(self.width):
                    noise_map[y, x] = pnoise.pnoise2(
                        x / scale,
                        y / scale,
                        octaves=3,
                        persistence=0.5,
                        lacunarity=2.0
                    )
        except ImportError:
            # Fallback to opensimplex if noise not available
            from opensimplex import OpenSimplex

            noise_gen = OpenSimplex(seed=42)
            noise_map = np.zeros((self.height, self.width))

            for y in range(self.height):
                for x in range(self.width):
                    noise_map[y, x] = noise_gen.noise2(x / scale, y / scale)

        # Normaliser 0-1
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())

        return noise_map

    def _apply_weathering(
        self,
        material_weights: Dict[str, np.ndarray],
        materials: Dict[str, MaterialLayer],
        heightmap: np.ndarray,
        slope_map: np.ndarray,
        moisture_map: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Applique effets weathering (altération, mousse, lichens)
        """

        # Identifier matériaux avec weathering
        for mat_name, material in materials.items():
            if material.weathering_strength > 0 and mat_name in material_weights:
                # Weathering plus fort dans zones humides
                weathering_boost = moisture_map * material.weathering_strength

                # Weathering plus fort sur pentes modérées (accumulation)
                slope_factor = 1.0 - np.abs(slope_map - 0.3) / 0.7
                slope_factor = np.clip(slope_factor, 0, 1)

                weathering_boost *= slope_factor

                # Appliquer boost
                material_weights[mat_name] += weathering_boost * 0.3

        return material_weights

    def _normalize_weights(
        self,
        material_weights: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Normalise les weights pour que somme = 1.0 à chaque pixel
        """

        # Sommer tous les weights
        total_weight = np.zeros((self.height, self.width), dtype=np.float32)

        for weight_map in material_weights.values():
            total_weight += weight_map

        # Éviter division par zéro
        total_weight = np.maximum(total_weight, 1e-6)

        # Normaliser
        normalized = {}
        for mat_name, weight_map in material_weights.items():
            normalized[mat_name] = weight_map / total_weight

        return normalized

    def _pack_splatmap(
        self,
        material_weights: Dict[str, np.ndarray],
        materials: Dict[str, MaterialLayer],
        layers: List[int]
    ) -> np.ndarray:
        """
        Pack 4 layers dans une texture RGBA

        Args:
            material_weights: Dict de weight maps
            materials: Dict de matériaux
            layers: Liste de 4 IDs de layers à packer

        Returns:
            RGBA image (H, W, 4) uint8
        """

        splatmap = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Map layer ID -> channel
        channel_map = {layers[i]: i for i in range(4)}

        # Remplir channels
        for mat_name, material in materials.items():
            if material.id in channel_map and mat_name in material_weights:
                channel = channel_map[material.id]
                splatmap[:, :, channel] = material_weights[mat_name]

        # Convertir en uint8 (0-255)
        splatmap = (splatmap * 255).astype(np.uint8)

        return splatmap

    def export_splatmaps(
        self,
        splatmap1: np.ndarray,
        splatmap2: np.ndarray,
        output_dir: str,
        prefix: str = "terrain",
        format: Literal['png', 'exr'] = 'png'
    ):
        """
        Exporte les splatmaps

        Args:
            splatmap1: Layers 0-3
            splatmap2: Layers 4-7
            output_dir: Dossier de sortie
            prefix: Préfixe des fichiers
            format: Format (png ou exr)
        """

        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        if format == 'png':
            # Export PNG
            img1 = Image.fromarray(splatmap1, mode='RGBA')
            img2 = Image.fromarray(splatmap2, mode='RGBA')

            img1.save(output_path / f"{prefix}_splatmap_0-3.png")
            img2.save(output_path / f"{prefix}_splatmap_4-7.png")

            logger.info(f"Splatmaps exportées en PNG dans {output_dir}")

        elif format == 'exr':
            # Export EXR 32-bit
            import OpenEXR
            import Imath

            # Convert to float32
            splatmap1_float = splatmap1.astype(np.float32) / 255.0
            splatmap2_float = splatmap2.astype(np.float32) / 255.0

            # TODO: Implémenter export EXR si OpenEXR disponible
            logger.warning("Export EXR nécessite OpenEXR, export PNG à la place")

            # Fallback PNG
            self.export_splatmaps(splatmap1, splatmap2, output_dir, prefix, format='png')

    def export_material_info(self, output_path: str):
        """
        Exporte les informations de matériaux en JSON
        Utile pour configurer shaders dans game engines
        """

        import json

        material_info = {}

        for mat_name, material in self.materials.items():
            material_info[mat_name] = {
                'id': material.id,
                'name': material.name,
                'description': material.description,
                'base_color_rgb': material.base_color,
                'roughness': material.roughness,
                'metallic': material.metallic,
                'altitude_range': [material.altitude_min, material.altitude_max],
                'slope_range': [material.slope_min, material.slope_max]
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(material_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Material info exporté: {output_path}")


# Fonction utilitaire pour tester
def test_splatmap_generator():
    """Test du générateur de splatmap"""

    # Créer heightmap de test
    width, height = 512, 512
    generator = PBRSplatmapGenerator(width, height)

    # Heightmap simple (montagne centrale)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    heightmap = 1.0 - (distance / max_dist)
    heightmap = np.clip(heightmap, 0, 1)
    heightmap = np.power(heightmap, 1.5)  # Pic plus prononcé

    print("Génération splatmaps...")
    splatmap1, splatmap2 = generator.generate_splatmap(
        heightmap,
        smooth_transitions=True,
        smooth_sigma=1.5
    )

    print(f"Splatmap 1 shape: {splatmap1.shape}")
    print(f"Splatmap 2 shape: {splatmap2.shape}")

    # Export
    generator.export_splatmaps(
        splatmap1,
        splatmap2,
        output_dir="/tmp/test_splatmaps",
        prefix="test"
    )

    # Export material info
    generator.export_material_info("/tmp/test_splatmaps/materials.json")

    print("Test terminé! Vérifiez /tmp/test_splatmaps/")


if __name__ == "__main__":
    test_splatmap_generator()
