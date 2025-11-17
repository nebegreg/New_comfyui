"""
Système d'érosion thermique pour génération de terrain réaliste
Simule les éboulements et la formation de talus d'éboulis

L'érosion thermique crée:
- Falaises abruptes avec angle de repos réaliste
- Talus d'éboulis au pied des pentes raides
- Formations rocheuses avec cassures naturelles
- Transition douce entre zones plates et pentes

Basé sur:
- Musgrave et al. 1989: "The synthesis and rendering of eroded fractal terrains"
- Chiba et al. 1998: "Erosion modeling on terrain"
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import convolve
import logging

logger = logging.getLogger(__name__)


class ThermalErosionSystem:
    """
    Système d'érosion thermique basé sur l'angle de repos des matériaux

    Principe:
    - Chaque matériau a un angle de repos maximum (talus angle)
    - Si la pente dépasse cet angle, le matériau s'éboule
    - Le matériau éboulé se dépose en bas de pente
    - Formation progressive de talus d'éboulis réalistes
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
        talus_angle: float = 0.7,
        num_iterations: int = 50,
        erosion_amount: float = 0.5,
        material_hardness: float = 1.0
    ) -> np.ndarray:
        """
        Applique l'érosion thermique sur une heightmap

        Args:
            heightmap: Heightmap d'entrée (0-1)
            talus_angle: Angle de repos maximum (en hauteur/distance)
                        0.5 = ~26°, 0.7 = ~35°, 1.0 = 45°, 1.5 = ~56°
                        Roche dure: 0.7-1.0, Sable/gravier: 0.5-0.7
            num_iterations: Nombre d'itérations (30-100 recommandé)
            erosion_amount: Quantité de matériau transféré (0-1)
            material_hardness: Dureté du matériau (0-1, affecte la vitesse d'érosion)

        Returns:
            Heightmap érodée
        """
        heightmap = heightmap.copy().astype(np.float32)

        logger.info(f"Démarrage érosion thermique: {num_iterations} iterations, "
                   f"talus_angle={talus_angle:.2f}")

        # Kernel 3x3 pour détecter les voisins
        # Structure: détecter les 8 voisins autour de chaque pixel
        neighbors_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)

        for iteration in range(num_iterations):
            # Copie pour modifications
            new_heightmap = heightmap.copy()

            # Pour chaque pixel, calculer différence avec voisins
            height_diff = self._calculate_height_differences(heightmap)

            # Masque des pixels qui doivent s'éroder (pente > talus_angle)
            should_erode = height_diff > talus_angle

            if not np.any(should_erode):
                # Plus rien à éroder
                logger.debug(f"Érosion thermique terminée anticipativement à l'itération {iteration}")
                break

            # Quantité de matériau à éroder
            # Proportionnel à l'excès de pente et à la dureté
            erode_amount = (height_diff - talus_angle) * erosion_amount * (1.0 - material_hardness * 0.5)
            erode_amount = np.maximum(erode_amount, 0)  # Pas de valeurs négatives

            # Calculer vers où le matériau s'écoule (voisin le plus bas)
            flow_directions = self._calculate_flow_directions(heightmap)

            # Transférer le matériau
            new_heightmap = self._transfer_material(
                new_heightmap,
                erode_amount,
                flow_directions
            )

            heightmap = new_heightmap

            # Log progress
            if (iteration + 1) % 10 == 0:
                logger.debug(f"Érosion thermique: {iteration+1}/{num_iterations} itérations")

        logger.info(f"Érosion thermique terminée: {num_iterations} itérations")
        return heightmap

    def _calculate_height_differences(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Calcule la différence de hauteur maximale avec les voisins
        Retourne un array avec la pente maximale pour chaque pixel
        """
        height, width = heightmap.shape
        max_diff = np.zeros_like(heightmap)

        # Vérifier les 8 voisins
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Distance pour calcul de pente
        distances = {
            (-1, -1): np.sqrt(2), (-1, 0): 1, (-1, 1): np.sqrt(2),
            (0, -1): 1,                       (0, 1): 1,
            (1, -1): np.sqrt(2),  (1, 0): 1,  (1, 1): np.sqrt(2)
        }

        for dy, dx in offsets:
            # Décaler heightmap dans cette direction
            shifted = np.roll(np.roll(heightmap, dy, axis=0), dx, axis=1)

            # Calculer différence de hauteur (pente)
            # diff positif = voisin plus bas
            height_diff = heightmap - shifted

            # Convertir en pente (hauteur / distance)
            slope = height_diff / distances[(dy, dx)]

            # Garder le maximum
            max_diff = np.maximum(max_diff, slope)

        return max_diff

    def _calculate_flow_directions(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Calcule vers quel voisin le matériau s'écoule (le plus bas)
        Retourne un array d'indices (0-7 pour les 8 directions, -1 si aucune)
        """
        height, width = heightmap.shape
        flow_dir = np.full((height, width), -1, dtype=np.int8)

        # Les 8 directions
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        min_height = np.full_like(heightmap, np.inf)

        for idx, (dy, dx) in enumerate(offsets):
            # Décaler heightmap
            shifted = np.roll(np.roll(heightmap, dy, axis=0), dx, axis=1)

            # Trouver où ce voisin est le plus bas
            is_lower = shifted < min_height
            min_height = np.where(is_lower, shifted, min_height)
            flow_dir = np.where(is_lower, idx, flow_dir)

        return flow_dir

    def _transfer_material(
        self,
        heightmap: np.ndarray,
        erode_amount: np.ndarray,
        flow_directions: np.ndarray
    ) -> np.ndarray:
        """
        Transfère le matériau érodé vers les voisins selon flow_directions
        """
        height, width = heightmap.shape
        new_heightmap = heightmap.copy()

        # Les 8 directions correspondant aux indices 0-7
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Pour chaque direction
        for idx, (dy, dx) in enumerate(offsets):
            # Masque des pixels qui s'écoulent dans cette direction
            flows_this_way = (flow_directions == idx)

            if not np.any(flows_this_way):
                continue

            # Quantité à transférer
            transfer_amount = erode_amount * flows_this_way

            # Enlever du pixel source
            new_heightmap -= transfer_amount

            # Ajouter au pixel destination (décalé)
            transferred = np.roll(np.roll(transfer_amount, -dy, axis=0), -dx, axis=1)
            new_heightmap += transferred

        return new_heightmap

    def apply_scree_formation(
        self,
        heightmap: np.ndarray,
        slope_threshold: float = 0.6,
        scree_amount: float = 0.1,
        num_iterations: int = 20
    ) -> np.ndarray:
        """
        Simule la formation de talus d'éboulis (scree) au pied des pentes
        Crée des accumulations réalistes de débris rocheux

        Args:
            heightmap: Heightmap d'entrée
            slope_threshold: Seuil de pente pour formation d'éboulis
            scree_amount: Quantité de débris générés
            num_iterations: Nombre d'itérations

        Returns:
            Heightmap avec talus d'éboulis
        """
        heightmap = heightmap.copy()

        logger.info(f"Formation de talus d'éboulis: {num_iterations} iterations")

        for _ in range(num_iterations):
            # Calculer gradient
            grad_y, grad_x = np.gradient(heightmap)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Zones avec forte pente = production d'éboulis
            produces_scree = gradient_magnitude > slope_threshold

            # Direction de la pente (vers le bas)
            # Normaliser les gradients
            grad_length = np.maximum(gradient_magnitude, 1e-10)
            grad_x_norm = grad_x / grad_length
            grad_y_norm = grad_y / grad_length

            # Éboulis tombe dans la direction de la pente
            # Simulation simple: déplacer un peu de matière vers le bas
            scree_deposit = np.zeros_like(heightmap)

            # Pour les zones productrices d'éboulis
            for y in range(1, heightmap.shape[0] - 1):
                for x in range(1, heightmap.shape[1] - 1):
                    if produces_scree[y, x]:
                        # Direction de chute (suivre la pente)
                        dx = int(np.sign(grad_x_norm[y, x]))
                        dy = int(np.sign(grad_y_norm[y, x]))

                        # Déposer l'éboulis 1-2 pixels plus bas
                        target_y = np.clip(y + dy, 0, heightmap.shape[0] - 1)
                        target_x = np.clip(x + dx, 0, heightmap.shape[1] - 1)

                        # Quantité d'éboulis
                        scree_qty = scree_amount * gradient_magnitude[y, x]

                        # Transfert
                        heightmap[y, x] -= scree_qty * 0.5
                        scree_deposit[target_y, target_x] += scree_qty

            # Appliquer dépôts d'éboulis
            heightmap += scree_deposit

        logger.info("Formation de talus d'éboulis terminée")
        return heightmap


class AdvancedThermalErosion:
    """
    Version avancée avec différenciation de matériaux
    Permet d'avoir des roches dures et des zones plus friables
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def apply_multi_material_erosion(
        self,
        heightmap: np.ndarray,
        material_map: np.ndarray,
        material_properties: dict,
        num_iterations: int = 50
    ) -> np.ndarray:
        """
        Érosion avec différents types de matériaux

        Args:
            heightmap: Heightmap d'entrée
            material_map: Map des matériaux (0-N pour N types de matériaux)
            material_properties: Dict avec propriétés par type:
                {
                    0: {'talus_angle': 0.8, 'hardness': 0.9},  # Roche dure
                    1: {'talus_angle': 0.5, 'hardness': 0.3},  # Gravier
                    ...
                }
            num_iterations: Nombre d'itérations

        Returns:
            Heightmap érodée avec variations réalistes selon matériaux
        """
        heightmap = heightmap.copy()

        # Créer une érosion différente pour chaque matériau
        for mat_id, props in material_properties.items():
            # Masque de ce matériau
            material_mask = (material_map == mat_id)

            if not np.any(material_mask):
                continue

            # Appliquer érosion avec paramètres spécifiques
            eroder = ThermalErosionSystem(self.width, self.height)

            # Créer une heightmap masquée (seulement ce matériau)
            masked_heightmap = np.where(material_mask, heightmap, 0)

            # Éroder
            eroded = eroder.apply_erosion(
                masked_heightmap,
                talus_angle=props.get('talus_angle', 0.7),
                num_iterations=num_iterations,
                erosion_amount=0.5,
                material_hardness=props.get('hardness', 0.5)
            )

            # Fusionner résultat
            heightmap = np.where(material_mask, eroded, heightmap)

        return heightmap
