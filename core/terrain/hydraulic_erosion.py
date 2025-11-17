"""
Système d'érosion hydraulique avancé pour génération de terrain ultra-réaliste
Basé sur les recherches:
- Olsen 2004: "Realtime Procedural Terrain Generation"
- Mei et al. 2007: "Fast Hydraulic Erosion Simulation"
- Stava et al. 2008: "Interactive Terrain Modeling Using Hydraulic Erosion"

Simule l'écoulement d'eau, le transport de sédiments et la déposition
pour créer des formations géologiques réalistes: vallées, ravines, deltas
"""

import numpy as np
from typing import Tuple, Optional
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)


class HydraulicErosionSystem:
    """
    Système d'érosion hydraulique par simulation de gouttes de pluie

    Chaque goutte suit une trajectoire physique réaliste:
    1. Tombe à une position aléatoire
    2. Suit la pente (gradient du terrain)
    3. Érode le terrain selon sa vitesse
    4. Transporte des sédiments
    5. Dépose les sédiments quand elle ralentit
    6. S'évapore progressivement
    """

    def __init__(self, width: int, height: int):
        """
        Args:
            width: Largeur de la heightmap
            height: Hauteur de la heightmap
        """
        self.width = width
        self.height = height

    def apply_erosion(
        self,
        heightmap: np.ndarray,
        num_iterations: int = 100000,
        erosion_radius: int = 3,
        inertia: float = 0.05,
        sediment_capacity_factor: float = 4.0,
        min_sediment_capacity: float = 0.01,
        erode_speed: float = 0.3,
        deposit_speed: float = 0.3,
        evaporate_speed: float = 0.01,
        gravity: float = 4.0,
        max_droplet_lifetime: int = 30,
        initial_water_volume: float = 1.0,
        initial_speed: float = 1.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Applique l'érosion hydraulique sur une heightmap

        Args:
            heightmap: Heightmap d'entrée (0-1), sera modifié en place
            num_iterations: Nombre de gouttes à simuler (50k-200k recommandé)
            erosion_radius: Rayon d'érosion autour de chaque goutte (pixels)
            inertia: Inertie de la goutte (0-1, plus haut = suit moins la pente)
            sediment_capacity_factor: Capacité de transport (plus haut = plus d'érosion)
            min_sediment_capacity: Capacité minimale même sur terrain plat
            erode_speed: Vitesse d'érosion (0-1)
            deposit_speed: Vitesse de déposition (0-1)
            evaporate_speed: Vitesse d'évaporation de l'eau (0-1)
            gravity: Force de gravité (accélération)
            max_droplet_lifetime: Durée de vie maximale d'une goutte
            initial_water_volume: Volume d'eau initial de chaque goutte
            initial_speed: Vitesse initiale
            seed: Seed pour reproductibilité

        Returns:
            Heightmap érodée (modifie aussi en place)
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Démarrage érosion hydraulique: {num_iterations} iterations, "
                   f"rayon={erosion_radius}, erosion={erode_speed}")

        # Pré-calculer les offsets pour le rayon d'érosion (optimisation)
        erosion_brush_indices, erosion_brush_weights = self._initialize_brush_indices(
            erosion_radius
        )

        # Convertir en float32 pour performance
        heightmap = heightmap.astype(np.float32)

        # Simulation de chaque goutte
        for i in range(num_iterations):
            # Position de départ aléatoire
            pos_x = np.random.randint(erosion_radius, self.width - erosion_radius)
            pos_y = np.random.randint(erosion_radius, self.height - erosion_radius)

            # Simuler la goutte
            heightmap = self._simulate_droplet(
                heightmap,
                float(pos_x),
                float(pos_y),
                erosion_brush_indices,
                erosion_brush_weights,
                erosion_radius,
                inertia,
                sediment_capacity_factor,
                min_sediment_capacity,
                erode_speed,
                deposit_speed,
                evaporate_speed,
                gravity,
                max_droplet_lifetime,
                initial_water_volume,
                initial_speed
            )

            # Log progress
            if (i + 1) % 10000 == 0:
                logger.debug(f"Érosion: {i+1}/{num_iterations} gouttes simulées")

        logger.info(f"Érosion hydraulique terminée: {num_iterations} gouttes")
        return heightmap

    def _initialize_brush_indices(
        self,
        radius: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pré-calcule les indices et poids pour le brush d'érosion
        Optimisation pour éviter de recalculer à chaque goutte

        Returns:
            (indices, weights) où:
            - indices: array de (x, y) offsets depuis le centre
            - weights: poids d'érosion pour chaque point (1 au centre, 0 au bord)
        """
        indices = []
        weights = []

        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                # Distance au centre
                sqr_dist = x * x + y * y
                if sqr_dist <= radius * radius:
                    indices.append((x, y))
                    # Poids décroissant avec la distance (distribution gaussienne)
                    weight = 1.0 - np.sqrt(sqr_dist) / radius
                    weights.append(weight)

        return np.array(indices, dtype=np.int32), np.array(weights, dtype=np.float32)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _simulate_droplet(
        heightmap: np.ndarray,
        start_x: float,
        start_y: float,
        brush_indices: np.ndarray,
        brush_weights: np.ndarray,
        erosion_radius: int,
        inertia: float,
        sediment_capacity_factor: float,
        min_sediment_capacity: float,
        erode_speed: float,
        deposit_speed: float,
        evaporate_speed: float,
        gravity: float,
        max_droplet_lifetime: int,
        initial_water_volume: float,
        initial_speed: float
    ) -> np.ndarray:
        """
        Simule le parcours d'une goutte d'eau (JIT compilé pour performance)

        Cette fonction est le cœur de l'algorithme. Elle est compilée par Numba
        pour atteindre des performances proches du C/C++.
        """
        height, width = heightmap.shape

        # État de la goutte
        pos_x, pos_y = start_x, start_y
        dir_x, dir_y = 0.0, 0.0
        speed = initial_speed
        water = initial_water_volume
        sediment = 0.0

        for lifetime in range(max_droplet_lifetime):
            # Position actuelle (indices)
            node_x = int(pos_x)
            node_y = int(pos_y)

            # Vérifier si hors limites
            if node_x < 0 or node_x >= width - 1 or node_y < 0 or node_y >= height - 1:
                break

            # Calculer offset dans la cellule (pour interpolation)
            cell_offset_x = pos_x - node_x
            cell_offset_y = pos_y - node_y

            # Hauteurs des 4 coins de la cellule
            height_NW = heightmap[node_y, node_x]
            height_NE = heightmap[node_y, node_x + 1]
            height_SW = heightmap[node_y + 1, node_x]
            height_SE = heightmap[node_y + 1, node_x + 1]

            # Calculer gradient par interpolation bilinéaire
            gradient_x = (height_NE - height_NW) * (1.0 - cell_offset_y) + \
                        (height_SE - height_SW) * cell_offset_y
            gradient_y = (height_SW - height_NW) * (1.0 - cell_offset_x) + \
                        (height_SE - height_NE) * cell_offset_x

            # Hauteur interpolée à la position exacte
            current_height = height_NW * (1.0 - cell_offset_x) * (1.0 - cell_offset_y) + \
                           height_NE * cell_offset_x * (1.0 - cell_offset_y) + \
                           height_SW * (1.0 - cell_offset_x) * cell_offset_y + \
                           height_SE * cell_offset_x * cell_offset_y

            # Calculer nouvelle direction avec inertie
            # L'inertie fait que la goutte ne suit pas instantanément la pente
            dir_x = dir_x * inertia - gradient_x * (1.0 - inertia)
            dir_y = dir_y * inertia - gradient_y * (1.0 - inertia)

            # Normaliser la direction
            length = np.sqrt(dir_x * dir_x + dir_y * dir_y)
            if length != 0:
                dir_x /= length
                dir_y /= length

            # Avancer la goutte
            pos_x += dir_x
            pos_y += dir_y

            # Vérifier nouvelle position
            new_node_x = int(pos_x)
            new_node_y = int(pos_y)

            if new_node_x < 0 or new_node_x >= width - 1 or \
               new_node_y < 0 or new_node_y >= height - 1:
                break

            # Calculer nouvelle hauteur
            new_offset_x = pos_x - new_node_x
            new_offset_y = pos_y - new_node_y

            new_height_NW = heightmap[new_node_y, new_node_x]
            new_height_NE = heightmap[new_node_y, new_node_x + 1]
            new_height_SW = heightmap[new_node_y + 1, new_node_x]
            new_height_SE = heightmap[new_node_y + 1, new_node_x + 1]

            new_height = new_height_NW * (1.0 - new_offset_x) * (1.0 - new_offset_y) + \
                        new_height_NE * new_offset_x * (1.0 - new_offset_y) + \
                        new_height_SW * (1.0 - new_offset_x) * new_offset_y + \
                        new_height_SE * new_offset_x * new_offset_y

            # Différence de hauteur (positif = montée, négatif = descente)
            delta_height = new_height - current_height

            # Calculer capacité de transport de sédiments
            # Plus la goutte va vite et descend, plus elle peut transporter
            sediment_capacity = max(
                -delta_height * speed * water * sediment_capacity_factor,
                min_sediment_capacity
            )

            # Si la goutte transporte plus que sa capacité OU monte
            if sediment > sediment_capacity or delta_height > 0:
                # Déposer des sédiments
                if delta_height > 0:
                    # Montée: déposer tout ce qui empêche de monter
                    amount_to_deposit = min(delta_height, sediment)
                else:
                    # Capacité dépassée: déposer l'excédent
                    amount_to_deposit = (sediment - sediment_capacity) * deposit_speed

                sediment -= amount_to_deposit

                # Déposer sur le terrain (node actuel)
                heightmap[node_y, node_x] += amount_to_deposit

            else:
                # Éroder le terrain
                amount_to_erode = min(
                    (sediment_capacity - sediment) * erode_speed,
                    -delta_height  # Ne pas éroder plus que nécessaire pour descendre
                )

                # Appliquer érosion avec le brush
                for i in range(len(brush_indices)):
                    offset_x, offset_y = brush_indices[i]
                    weight = brush_weights[i]

                    # Position du point érodé
                    erode_x = node_x + offset_x
                    erode_y = node_y + offset_y

                    # Vérifier limites
                    if 0 <= erode_x < width and 0 <= erode_y < height:
                        # Quantité érodée proportionnelle au poids
                        weighted_erode_amount = amount_to_erode * weight

                        # Limiter pour ne pas créer de vallées négatives
                        delta_sediment = min(
                            heightmap[erode_y, erode_x],
                            weighted_erode_amount
                        )

                        heightmap[erode_y, erode_x] -= delta_sediment
                        sediment += delta_sediment

            # Mettre à jour vitesse (accélération gravitationnelle)
            speed = np.sqrt(max(0, speed * speed + delta_height * gravity))

            # Évaporation de l'eau
            water *= (1.0 - evaporate_speed)

            # Si plus d'eau, arrêter
            if water < 0.01:
                break

        return heightmap


class RainErosionSystem:
    """
    Système d'érosion par pluie uniforme
    Alternative plus simple pour érosion globale douce
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def apply_rain_erosion(
        self,
        heightmap: np.ndarray,
        rain_amount: float = 0.01,
        dissolution_rate: float = 0.1,
        num_iterations: int = 10
    ) -> np.ndarray:
        """
        Applique une érosion par pluie uniforme
        Adoucit les formes, crée un aspect plus naturel

        Args:
            heightmap: Heightmap d'entrée
            rain_amount: Quantité de pluie par itération
            dissolution_rate: Taux de dissolution du terrain
            num_iterations: Nombre d'itérations
        """
        heightmap = heightmap.copy()

        for _ in range(num_iterations):
            # Calculer gradients
            grad_y, grad_x = np.gradient(heightmap)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Dissolution proportionnelle à la pente
            dissolution = gradient_magnitude * rain_amount * dissolution_rate

            # Appliquer dissolution
            heightmap -= dissolution

            # Redistribution (lissage très léger)
            from scipy.ndimage import gaussian_filter
            heightmap = gaussian_filter(heightmap, sigma=0.3)

        return heightmap
