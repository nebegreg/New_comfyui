"""
Générateur de terrain 3D professionnel
Crée des heightmaps, normal maps, depth maps et textures réalistes
Pour utilisation par des graphistes professionnels
"""

import numpy as np
from PIL import Image
import noise
from opensimplex import OpenSimplex
from typing import Tuple, Optional, Dict
import cv2
from scipy.ndimage import gaussian_filter


class TerrainGenerator:
    """Génère des terrains 3D réalistes avec toutes les maps nécessaires"""

    def __init__(self, width: int = 2048, height: int = 2048):
        self.width = width
        self.height = height
        self.heightmap = None
        self.normal_map = None
        self.depth_map = None
        self.ao_map = None  # Ambient Occlusion
        self.roughness_map = None

    def generate_heightmap(self,
                          scale: float = 100.0,
                          octaves: int = 8,
                          persistence: float = 0.5,
                          lacunarity: float = 2.0,
                          mountain_type: str = 'alpine',
                          seed: int = 0) -> np.ndarray:
        """
        Génère une heightmap réaliste avec Perlin/Simplex noise

        Args:
            scale: Échelle du terrain (plus petit = plus détaillé)
            octaves: Nombre de niveaux de détail
            persistence: Amplitude des détails (0-1)
            lacunarity: Fréquence des détails (typiquement 2.0)
            mountain_type: Type de montagne pour paramètres adaptés
            seed: Seed pour reproductibilité

        Returns:
            Heightmap normalisée (0-1)
        """
        heightmap = np.zeros((self.height, self.width))

        # Paramètres selon le type de montagne
        if mountain_type == 'alpine':
            # Montagnes alpines - pics aigus
            base_frequency = 1.0
            erosion_strength = 0.3
            peak_sharpness = 1.5
        elif mountain_type == 'volcanic':
            # Volcanique - pic central prononcé
            base_frequency = 0.8
            erosion_strength = 0.2
            peak_sharpness = 2.0
        elif mountain_type == 'rolling':
            # Collines douces
            base_frequency = 0.5
            erosion_strength = 0.5
            peak_sharpness = 0.8
        elif mountain_type == 'massive':
            # Massif - larges formations
            base_frequency = 0.6
            erosion_strength = 0.4
            peak_sharpness = 1.2
        else:  # rocky
            # Rocheux - irrégulier
            base_frequency = 1.2
            erosion_strength = 0.1
            peak_sharpness = 1.8

        # Génération multi-octaves
        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width - 0.5
                ny = y / self.height - 0.5

                # Perlin noise multi-octave
                elevation = 0
                amplitude = 1.0
                frequency = base_frequency

                for octave in range(octaves):
                    sample_x = nx * frequency * scale
                    sample_y = ny * frequency * scale

                    # Perlin noise
                    perlin_value = noise.pnoise2(
                        sample_x, sample_y,
                        octaves=1,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=self.width,
                        repeaty=self.height,
                        base=seed + octave
                    )

                    elevation += perlin_value * amplitude
                    amplitude *= persistence
                    frequency *= lacunarity

                heightmap[y, x] = elevation

        # Normaliser
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

        # Appliquer une courbe pour accentuer les pics
        heightmap = np.power(heightmap, peak_sharpness)

        # Ajouter un gradient radial pour créer un pic central (optionnel)
        if mountain_type in ['volcanic', 'alpine', 'massive']:
            center_x, center_y = self.width // 2, self.height // 2
            y_coords, x_coords = np.ogrid[:self.height, :self.width]
            distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            radial_gradient = 1.0 - (distance / max_distance)
            radial_gradient = np.clip(radial_gradient, 0, 1)

            # Mélanger avec le heightmap
            blend_factor = 0.3 if mountain_type == 'alpine' else 0.5
            heightmap = heightmap * (1 - blend_factor) + radial_gradient * blend_factor

        # Simulation d'érosion simple
        if erosion_strength > 0:
            heightmap = self._apply_erosion(heightmap, iterations=3, strength=erosion_strength)

        # Lisser légèrement pour réalisme
        heightmap = gaussian_filter(heightmap, sigma=0.5)

        # Normaliser à nouveau
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

        self.heightmap = heightmap
        return heightmap

    def _apply_erosion(self, heightmap: np.ndarray, iterations: int = 3, strength: float = 0.3) -> np.ndarray:
        """Applique une simulation d'érosion hydraulique simplifiée"""
        eroded = heightmap.copy()

        for _ in range(iterations):
            # Calculer les gradients
            grad_y, grad_x = np.gradient(eroded)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Érosion proportionnelle au gradient (pentes raides s'érodent plus)
            erosion = gradient_magnitude * strength * 0.01
            eroded -= erosion

            # Lisser légèrement
            eroded = gaussian_filter(eroded, sigma=0.3)

        return eroded

    def generate_normal_map(self, strength: float = 1.0) -> np.ndarray:
        """
        Génère une normal map à partir de la heightmap

        Args:
            strength: Force des normales (1.0 = normal, >1.0 = plus prononcé)

        Returns:
            Normal map RGB (0-255)
        """
        if self.heightmap is None:
            raise ValueError("Générez d'abord une heightmap avec generate_heightmap()")

        # Calculer les gradients
        zy, zx = np.gradient(self.heightmap)

        # Appliquer la force
        zx *= strength
        zy *= strength

        # Calculer les normales
        normal = np.dstack((-zx, -zy, np.ones_like(self.heightmap)))

        # Normaliser
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = normal / (norm + 1e-10)

        # Convertir de [-1,1] à [0,255]
        normal_map = ((normal + 1.0) * 127.5).astype(np.uint8)

        self.normal_map = normal_map
        return normal_map

    def generate_depth_map(self, near: float = 0.0, far: float = 1.0) -> np.ndarray:
        """
        Génère une depth map (Z-depth) pour rendu

        Args:
            near: Distance proche (0-1)
            far: Distance lointaine (0-1)

        Returns:
            Depth map grayscale (0-255)
        """
        if self.heightmap is None:
            raise ValueError("Générez d'abord une heightmap avec generate_heightmap()")

        # La depth map est essentiellement la heightmap inversée et normalisée
        depth = self.heightmap.copy()

        # Inverser (points hauts = proche, points bas = loin)
        depth = 1.0 - depth

        # Appliquer les limites near/far
        depth = near + depth * (far - near)

        # Convertir en 0-255
        depth_map = (depth * 255).astype(np.uint8)

        self.depth_map = depth_map
        return depth_map

    def generate_ambient_occlusion(self, samples: int = 16, radius: float = 0.05) -> np.ndarray:
        """
        Génère une map d'ambient occlusion

        Args:
            samples: Nombre d'échantillons pour le calcul
            radius: Rayon d'échantillonnage

        Returns:
            AO map (0-255)
        """
        if self.heightmap is None:
            raise ValueError("Générez d'abord une heightmap avec generate_heightmap()")

        ao_map = np.ones((self.height, self.width), dtype=np.float32)

        # Méthode simplifiée: utiliser les gradients locaux
        # Les zones avec beaucoup de variation = plus d'occlusion
        for i in range(3):
            kernel_size = int(radius * min(self.width, self.height) * (i + 1))
            if kernel_size < 3:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1

            blurred = cv2.GaussianBlur(self.heightmap, (kernel_size, kernel_size), 0)
            diff = np.abs(self.heightmap - blurred)
            ao_map -= diff * 0.3

        # Normaliser
        ao_map = np.clip(ao_map, 0, 1)
        ao_map = (ao_map * 255).astype(np.uint8)

        self.ao_map = ao_map
        return ao_map

    def generate_roughness_map(self, base_roughness: float = 0.5) -> np.ndarray:
        """
        Génère une roughness map pour PBR

        Args:
            base_roughness: Rugosité de base (0-1)

        Returns:
            Roughness map (0-255)
        """
        if self.heightmap is None:
            raise ValueError("Générez d'abord une heightmap avec generate_heightmap()")

        # La rugosité varie selon les pentes
        grad_y, grad_x = np.gradient(self.heightmap)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        # Normaliser la pente
        slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-10)

        # Zones plates = moins rugueux, pentes raides = plus rugueux
        roughness = base_roughness + slope * (1.0 - base_roughness)

        # Ajouter du bruit pour variation
        noise_layer = np.random.random((self.height, self.width)) * 0.1
        roughness = np.clip(roughness + noise_layer, 0, 1)

        roughness_map = (roughness * 255).astype(np.uint8)

        self.roughness_map = roughness_map
        return roughness_map

    def export_all_maps(self, output_dir: str, prefix: str = "terrain"):
        """
        Exporte toutes les maps générées

        Args:
            output_dir: Dossier de sortie
            prefix: Préfixe des fichiers
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        maps_to_export = {
            'heightmap': (self.heightmap, True),  # True = grayscale
            'normal': (self.normal_map, False),   # False = RGB
            'depth': (self.depth_map, True),
            'ao': (self.ao_map, True),
            'roughness': (self.roughness_map, True)
        }

        for map_name, (map_data, is_grayscale) in maps_to_export.items():
            if map_data is not None:
                filepath = os.path.join(output_dir, f"{prefix}_{map_name}.png")

                if is_grayscale and map_data.dtype == np.float64:
                    # Convertir en uint8
                    map_data = (map_data * 255).astype(np.uint8)

                if is_grayscale and len(map_data.shape) == 2:
                    img = Image.fromarray(map_data, mode='L')
                else:
                    img = Image.fromarray(map_data, mode='RGB')

                img.save(filepath)
                print(f"✓ Exporté: {filepath}")

    def get_3d_mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne les données pour créer un mesh 3D

        Returns:
            vertices, faces, normals
        """
        if self.heightmap is None:
            raise ValueError("Générez d'abord une heightmap avec generate_heightmap()")

        # Créer les vertices
        vertices = []
        for y in range(self.height):
            for x in range(self.width):
                z = self.heightmap[y, x]
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Créer les faces (triangles)
        faces = []
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                # Indices des 4 coins du quad
                i0 = y * self.width + x
                i1 = y * self.width + (x + 1)
                i2 = (y + 1) * self.width + (x + 1)
                i3 = (y + 1) * self.width + x

                # Deux triangles par quad
                faces.append([i0, i1, i2])
                faces.append([i0, i2, i3])

        faces = np.array(faces)

        # Les normales sont déjà calculées dans la normal map
        normals = None
        if self.normal_map is not None:
            normals = (self.normal_map.astype(np.float32) / 127.5) - 1.0

        return vertices, faces, normals

    def apply_ai_texture_guidance(self, base_color: Tuple[int, int, int] = (139, 69, 19)) -> np.ndarray:
        """
        Crée une texture de base guidée par la heightmap
        Pour être utilisée comme base pour génération AI

        Args:
            base_color: Couleur de base RGB

        Returns:
            Texture RGB (H, W, 3)
        """
        if self.heightmap is None:
            raise ValueError("Générez d'abord une heightmap avec generate_heightmap()")

        texture = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Gradient altitudinal (comme dans la nature)
        # Bas = vert (forêt), moyen = brun (roche), haut = blanc (neige)

        for i in range(3):  # RGB
            channel = np.zeros((self.height, self.width))

            # Zone basse (0-0.4): Vert/brun foncé
            mask_low = self.heightmap < 0.4
            if i == 1:  # Green
                channel[mask_low] = 100 + self.heightmap[mask_low] * 100
            else:
                channel[mask_low] = 50 + self.heightmap[mask_low] * 50

            # Zone moyenne (0.4-0.7): Brun/gris
            mask_mid = (self.heightmap >= 0.4) & (self.heightmap < 0.7)
            channel[mask_mid] = base_color[i] * (1.0 - (self.heightmap[mask_mid] - 0.4) / 0.3)

            # Zone haute (0.7-1.0): Blanc (neige)
            mask_high = self.heightmap >= 0.7
            blend = (self.heightmap[mask_high] - 0.7) / 0.3
            channel[mask_high] = base_color[i] * (1 - blend) + 255 * blend

            texture[:, :, i] = channel

        # Ajouter du bruit pour texture
        noise_r = np.random.randint(-10, 10, (self.height, self.width))
        noise_g = np.random.randint(-10, 10, (self.height, self.width))
        noise_b = np.random.randint(-10, 10, (self.height, self.width))

        texture[:, :, 0] = np.clip(texture[:, :, 0] + noise_r, 0, 255)
        texture[:, :, 1] = np.clip(texture[:, :, 1] + noise_g, 0, 255)
        texture[:, :, 2] = np.clip(texture[:, :, 2] + noise_b, 0, 255)

        return texture.astype(np.uint8)
