"""
Système de placement de végétation réaliste
Utilise Poisson Disc Sampling pour distribution naturelle
Supporte clustering (groupes d'arbres) et spacing écologique

Algorithmes utilisés:
- Poisson Disc Sampling (distribution uniforme mais naturelle)
- Ecosystem simulation (competition, spacing)
- Terrain-based placement (slope, elevation, moisture)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreeInstance:
    """Représente une instance d'arbre"""
    x: float              # Position X (pixels)
    y: float              # Position Z (pixels)
    elevation: float      # Hauteur (valeur heightmap)
    species: str          # Espèce ('pine', 'spruce', 'fir', etc.)
    scale: float          # Échelle (0.8-1.2 pour variation)
    rotation: float       # Rotation en degrés (0-360)
    age: float            # Âge relatif (0-1)
    health: float         # Santé (0-1, affecte apparence)


class VegetationPlacer:
    """
    Place la végétation de manière réaliste sur le terrain

    Techniques:
    - Poisson disc sampling pour espacement naturel
    - Clustering pour groupes d'arbres réalistes
    - Règles écologiques (altitude, pente, orientation, moisture)
    - Competition entre arbres (pas trop proches)
    """

    def __init__(
        self,
        width: int,
        height: int,
        heightmap: np.ndarray,
        biome_map: np.ndarray
    ):
        """
        Args:
            width: Largeur du terrain
            height: Hauteur du terrain
            heightmap: Heightmap du terrain (0-1)
            biome_map: Map des biomes (BiomeType)
        """
        self.width = width
        self.height = height
        self.heightmap = heightmap
        self.biome_map = biome_map

        self.tree_instances: List[TreeInstance] = []

    def place_vegetation(
        self,
        density: float = 0.5,
        min_spacing: float = 3.0,
        max_attempts: int = 30,
        use_clustering: bool = True,
        cluster_size: int = 5,
        cluster_radius: float = 10.0,
        seed: Optional[int] = None
    ) -> List[TreeInstance]:
        """
        Place tous les arbres sur le terrain

        Args:
            density: Densité globale (0-1)
            min_spacing: Espacement minimum entre arbres (pixels)
            max_attempts: Tentatives max par point (Poisson disc)
            use_clustering: Activer le clustering (groupes d'arbres)
            cluster_size: Nombre d'arbres par cluster
            cluster_radius: Rayon des clusters
            seed: Seed pour reproductibilité

        Returns:
            Liste de TreeInstance
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Placement végétation: density={density}, spacing={min_spacing}, "
                   f"clustering={'ON' if use_clustering else 'OFF'}")

        self.tree_instances = []

        # Placer arbres par biome
        from .biome_classifier import BiomeType

        biomes_with_trees = [
            BiomeType.SUBALPINE,
            BiomeType.MONTANE_FOREST,
            BiomeType.VALLEY_FLOOR
        ]

        for biome_type in biomes_with_trees:
            # Masque du biome
            biome_mask = (self.biome_map == biome_type)

            if not np.any(biome_mask):
                continue

            # Densité ajustée selon biome
            biome_density = self._get_biome_density(biome_type, density)

            # Placer arbres dans ce biome
            biome_trees = self._place_in_biome(
                biome_mask,
                biome_type,
                biome_density,
                min_spacing,
                max_attempts,
                use_clustering,
                cluster_size,
                cluster_radius
            )

            self.tree_instances.extend(biome_trees)

        logger.info(f"Placement terminé: {len(self.tree_instances)} arbres placés")

        return self.tree_instances

    def _get_biome_density(self, biome_type: int, base_density: float) -> float:
        """Ajuste la densité selon le biome"""
        from .biome_classifier import BiomeType

        density_multipliers = {
            BiomeType.SUBALPINE: 0.3,        # Arbres dispersés
            BiomeType.MONTANE_FOREST: 1.0,   # Densité normale
            BiomeType.VALLEY_FLOOR: 1.2      # Plus dense
        }

        multiplier = density_multipliers.get(biome_type, 1.0)
        return base_density * multiplier

    def _place_in_biome(
        self,
        biome_mask: np.ndarray,
        biome_type: int,
        density: float,
        min_spacing: float,
        max_attempts: int,
        use_clustering: bool,
        cluster_size: int,
        cluster_radius: float
    ) -> List[TreeInstance]:
        """Place les arbres dans un biome spécifique"""

        trees = []

        if use_clustering:
            # Placement par clusters
            trees = self._poisson_disc_clustering(
                biome_mask,
                biome_type,
                density,
                min_spacing,
                max_attempts,
                cluster_size,
                cluster_radius
            )
        else:
            # Placement Poisson disc standard
            trees = self._poisson_disc_sampling(
                biome_mask,
                biome_type,
                density,
                min_spacing,
                max_attempts
            )

        return trees

    def _poisson_disc_sampling(
        self,
        valid_mask: np.ndarray,
        biome_type: int,
        density: float,
        min_spacing: float,
        max_attempts: int
    ) -> List[TreeInstance]:
        """
        Poisson Disc Sampling classique
        Garantit espacement minimum entre points

        Algorithme:
        1. Commence avec un point aléatoire
        2. Génère candidats autour des points actifs
        3. Valide candidats (espacement + contraintes terrain)
        4. Ajoute valides à la liste
        5. Répète jusqu'à épuisement
        """
        trees = []

        # Grille pour accélération spatiale
        cell_size = min_spacing / np.sqrt(2)
        grid_width = int(np.ceil(self.width / cell_size))
        grid_height = int(np.ceil(self.height / cell_size))
        grid = [[] for _ in range(grid_width * grid_height)]

        # Points actifs (peuvent générer voisins)
        active_list = []

        # Nombre cible de points (basé sur densité et surface)
        valid_area = np.sum(valid_mask)
        target_count = int(valid_area * density / (min_spacing ** 2))

        # Point de départ
        while len(trees) < target_count and len(active_list) < 1000:
            # Trouver point de départ valide
            if len(active_list) == 0:
                # Nouveau point aléatoire dans zone valide
                valid_indices = np.argwhere(valid_mask)
                if len(valid_indices) == 0:
                    break

                idx = np.random.randint(len(valid_indices))
                y, x = valid_indices[idx]

                # Créer arbre
                tree = self._create_tree_instance(x, y, biome_type)
                trees.append(tree)

                # Ajouter à grille et liste active
                self._add_to_grid(grid, grid_width, cell_size, x, y)
                active_list.append((x, y))
                continue

            # Choisir point actif aléatoire
            active_idx = np.random.randint(len(active_list))
            active_x, active_y = active_list[active_idx]

            # Générer candidats autour de ce point
            found_candidate = False

            for _ in range(max_attempts):
                # Angle et distance aléatoires
                angle = np.random.random() * 2 * np.pi
                radius = min_spacing + np.random.random() * min_spacing

                # Position candidate
                cand_x = active_x + radius * np.cos(angle)
                cand_y = active_y + radius * np.sin(angle)

                # Vérifier limites
                if cand_x < 0 or cand_x >= self.width or \
                   cand_y < 0 or cand_y >= self.height:
                    continue

                # Vérifier masque valide
                if not valid_mask[int(cand_y), int(cand_x)]:
                    continue

                # Vérifier espacement avec voisins
                if not self._is_valid_position(
                    grid, grid_width, cell_size,
                    cand_x, cand_y, min_spacing
                ):
                    continue

                # Candidat valide!
                tree = self._create_tree_instance(cand_x, cand_y, biome_type)
                trees.append(tree)

                self._add_to_grid(grid, grid_width, cell_size, cand_x, cand_y)
                active_list.append((cand_x, cand_y))
                found_candidate = True
                break

            # Si pas de candidat trouvé, retirer de liste active
            if not found_candidate:
                active_list.pop(active_idx)

        return trees

    def _poisson_disc_clustering(
        self,
        valid_mask: np.ndarray,
        biome_type: int,
        density: float,
        min_spacing: float,
        max_attempts: int,
        cluster_size: int,
        cluster_radius: float
    ) -> List[TreeInstance]:
        """
        Poisson disc avec clustering
        Crée des groupes d'arbres pour aspect plus naturel
        """
        trees = []

        # D'abord placer les centres de clusters (Poisson disc)
        cluster_spacing = cluster_radius * 2.5  # Espacement entre clusters

        cluster_centers = self._poisson_disc_sampling(
            valid_mask,
            biome_type,
            density * 0.5,  # Moins de centres que d'arbres individuels
            cluster_spacing,
            max_attempts
        )

        # Pour chaque centre, créer un cluster d'arbres
        for center_tree in cluster_centers:
            # Nombre d'arbres dans ce cluster (variation)
            num_trees = np.random.randint(
                max(1, cluster_size - 2),
                cluster_size + 3
            )

            # Placer arbres autour du centre
            for _ in range(num_trees):
                # Position aléatoire dans le rayon
                angle = np.random.random() * 2 * np.pi
                radius = np.random.random() * cluster_radius

                x = center_tree.x + radius * np.cos(angle)
                y = center_tree.y + radius * np.sin(angle)

                # Vérifier validité
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    continue

                if not valid_mask[int(y), int(x)]:
                    continue

                # Créer arbre
                tree = self._create_tree_instance(x, y, biome_type)

                # Variation au sein du cluster
                tree.scale *= np.random.uniform(0.9, 1.1)
                tree.age *= np.random.uniform(0.8, 1.0)

                trees.append(tree)

        return trees

    def _create_tree_instance(
        self,
        x: float,
        y: float,
        biome_type: int
    ) -> TreeInstance:
        """Crée une instance d'arbre avec propriétés réalistes"""

        # Obtenir espèce selon biome
        species = self._select_species(biome_type)

        # Élévation
        elevation = self.heightmap[int(y), int(x)]

        # Échelle (variation naturelle)
        scale = np.random.uniform(0.85, 1.15)

        # Rotation aléatoire
        rotation = np.random.random() * 360

        # Âge (corrélation avec élévation: plus bas = plus vieux)
        age = 0.5 + (1.0 - elevation) * 0.3 + np.random.random() * 0.2
        age = np.clip(age, 0, 1)

        # Santé (généralement haute, quelques arbres malades)
        health = np.random.choice([0.9, 1.0], p=[0.1, 0.9])
        health += np.random.uniform(-0.05, 0.05)
        health = np.clip(health, 0, 1)

        return TreeInstance(
            x=x,
            y=y,
            elevation=elevation,
            species=species,
            scale=scale,
            rotation=rotation,
            age=age,
            health=health
        )

    def _select_species(self, biome_type: int) -> str:
        """Sélectionne une espèce d'arbre selon le biome"""
        from .biome_classifier import BiomeType

        species_distribution = {
            BiomeType.SUBALPINE: {
                'pine': 0.6,
                'spruce': 0.4
            },
            BiomeType.MONTANE_FOREST: {
                'pine': 0.4,
                'spruce': 0.4,
                'fir': 0.2
            },
            BiomeType.VALLEY_FLOOR: {
                'pine': 0.2,
                'spruce': 0.3,
                'fir': 0.3,
                'deciduous': 0.2
            }
        }

        distribution = species_distribution.get(
            biome_type,
            {'pine': 1.0}
        )

        # Sélection pondérée
        species_list = list(distribution.keys())
        probabilities = list(distribution.values())

        return np.random.choice(species_list, p=probabilities)

    def _add_to_grid(
        self,
        grid: List,
        grid_width: int,
        cell_size: float,
        x: float,
        y: float
    ):
        """Ajoute un point à la grille spatiale"""
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        grid_idx = grid_y * grid_width + grid_x
        grid[grid_idx].append((x, y))

    def _is_valid_position(
        self,
        grid: List,
        grid_width: int,
        cell_size: float,
        x: float,
        y: float,
        min_spacing: float
    ) -> bool:
        """Vérifie si position respecte espacement minimum"""
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)

        # Vérifier cellules voisines
        search_radius = 2  # Chercher 2 cellules autour

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy

                if check_x < 0 or check_x >= grid_width:
                    continue
                if check_y < 0 or check_y >= int(self.height / cell_size):
                    continue

                cell_idx = check_y * grid_width + check_x
                if cell_idx >= len(grid):
                    continue

                # Vérifier distance avec points dans cette cellule
                for px, py in grid[cell_idx]:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < min_spacing:
                        return False

        return True

    def export_instances(self, filepath: str):
        """
        Exporte les instances pour utilisation externe
        (Blender, Unreal, Unity, etc.)
        """
        import json

        data = {
            'tree_count': len(self.tree_instances),
            'terrain_size': [int(self.width), int(self.height)],
            'instances': [
                {
                    'position': [float(t.x), float(t.elevation), float(t.y)],  # X, Height, Z
                    'species': t.species,
                    'scale': float(t.scale),
                    'rotation': float(t.rotation),
                    'age': float(t.age),
                    'health': float(t.health)
                }
                for t in self.tree_instances
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Instances exportées: {filepath}")

    def generate_density_map(self) -> np.ndarray:
        """
        Génère une density map pour utilisation en ControlNet
        ou comme texture pour shaders

        Returns:
            Density map (0-1) avec densité d'arbres par zone
        """
        density_map = np.zeros((self.height, self.width), dtype=np.float32)

        # Rayon d'influence pour chaque arbre
        influence_radius = 10  # pixels

        for tree in self.tree_instances:
            x, y = int(tree.x), int(tree.y)

            # Dessiner densité autour de l'arbre
            y_min = max(0, y - influence_radius)
            y_max = min(self.height, y + influence_radius + 1)
            x_min = max(0, x - influence_radius)
            x_max = min(self.width, x + influence_radius + 1)

            for py in range(y_min, y_max):
                for px in range(x_min, x_max):
                    dist = np.sqrt((px - x)**2 + (py - y)**2)

                    if dist <= influence_radius:
                        # Gradient radial
                        density_value = 1.0 - (dist / influence_radius)
                        density_map[py, px] = max(
                            density_map[py, px],
                            density_value
                        )

        return density_map
