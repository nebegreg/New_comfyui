"""
Water System - Rivers, Lakes, Waterfalls
=========================================

Generates realistic water features for mountain terrains:
- Rivers following terrain flow
- Lakes in natural depressions
- Waterfalls at steep drops
- Water level simulation
- Realistic water rendering

Author: Mountain Studio Pro Team
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from scipy.ndimage import binary_dilation, label, maximum_filter
from skimage.morphology import watershed

logger = logging.getLogger(__name__)


class WaterSystem:
    """
    Complete water system for terrain.

    Generates rivers, lakes, and waterfalls based on terrain topology.
    """

    def __init__(self, heightmap: np.ndarray):
        """
        Initialize water system.

        Args:
            heightmap: Terrain heightmap (H, W) normalized [0, 1]
        """
        self.heightmap = heightmap.copy()
        self.h, self.w = heightmap.shape

        # Water features
        self.rivers: List[np.ndarray] = []
        self.lakes: Dict = {}
        self.waterfalls: List[Dict] = []

        # Flow accumulation map
        self.flow_accumulation: Optional[np.ndarray] = None

        logger.info(f"WaterSystem initialized for {self.w}x{self.h} terrain")

    def generate_all(self, river_threshold: int = 100, lake_min_size: int = 50):
        """
        Generate all water features.

        Args:
            river_threshold: Flow accumulation threshold for rivers
            lake_min_size: Minimum lake size in pixels
        """
        logger.info("Generating water features...")

        # 1. Calculate flow accumulation
        self.flow_accumulation = self.calculate_flow_accumulation()

        # 2. Extract rivers
        self.rivers = self.extract_rivers(threshold=river_threshold)

        # 3. Find and fill lakes
        self.lakes = self.generate_lakes(min_size=lake_min_size)

        # 4. Detect waterfalls
        self.waterfalls = self.detect_waterfalls()

        logger.info(f"Generated {len(self.rivers)} rivers, {len(self.lakes)} lakes, {len(self.waterfalls)} waterfalls")

    def calculate_flow_accumulation(self) -> np.ndarray:
        """
        Calculate flow accumulation using D8 algorithm.

        Returns:
            Flow accumulation map (H, W) - higher values = more flow
        """
        logger.info("Calculating flow accumulation...")

        flow = np.ones((self.h, self.w), dtype=np.float32)

        # D8 flow directions (8 neighbors)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

        # Process pixels from highest to lowest
        pixels = []
        for i in range(self.h):
            for j in range(self.w):
                pixels.append((self.heightmap[i, j], i, j))

        pixels.sort(reverse=True)  # Highest first

        for height, i, j in pixels:
            if flow[i, j] == 0:
                continue

            # Find steepest descent direction
            max_slope = -np.inf
            best_dir = None

            for di, dj in directions:
                ni, nj = i + di, j + dj

                if 0 <= ni < self.h and 0 <= nj < self.w:
                    slope = (self.heightmap[i, j] - self.heightmap[ni, nj])

                    if slope > max_slope:
                        max_slope = slope
                        best_dir = (ni, nj)

            # Flow to steepest neighbor
            if best_dir and max_slope > 0:
                ni, nj = best_dir
                flow[ni, nj] += flow[i, j]

        logger.debug(f"Flow accumulation range: [{flow.min():.1f}, {flow.max():.1f}]")

        return flow

    def extract_rivers(self, threshold: int = 100) -> List[np.ndarray]:
        """
        Extract river paths from flow accumulation.

        Args:
            threshold: Flow threshold for river formation

        Returns:
            List of river polylines
        """
        logger.info(f"Extracting rivers (threshold={threshold})...")

        if self.flow_accumulation is None:
            self.flow_accumulation = self.calculate_flow_accumulation()

        # Binary river mask
        river_mask = self.flow_accumulation > threshold

        # Trace river paths
        rivers = []
        labeled, num_features = label(river_mask)

        for label_id in range(1, num_features + 1):
            river_pixels = np.argwhere(labeled == label_id)

            if len(river_pixels) > 10:  # Minimum river length
                rivers.append(river_pixels)

        logger.info(f"Extracted {len(rivers)} rivers")

        return rivers

    def carve_rivers(self, depth: float = 0.05) -> np.ndarray:
        """
        Carve river valleys into heightmap.

        Args:
            depth: River carving depth

        Returns:
            Modified heightmap with carved rivers
        """
        logger.info(f"Carving rivers (depth={depth})...")

        carved = self.heightmap.copy()

        for river_pixels in self.rivers:
            for i, j in river_pixels:
                # Carve with smooth falloff
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj

                        if 0 <= ni < self.h and 0 <= nj < self.w:
                            distance = np.sqrt(di ** 2 + dj ** 2)
                            falloff = max(0, 1 - distance / 3.0)

                            carved[ni, nj] -= depth * falloff

        carved = np.clip(carved, 0, 1)

        logger.info("Rivers carved")

        return carved

    def generate_lakes(self, min_size: int = 50, water_level: float = 0.5) -> Dict:
        """
        Generate lakes in natural depressions.

        Args:
            min_size: Minimum lake size in pixels
            water_level: Water fill level

        Returns:
            Dictionary of lakes {id: {'mask': array, 'depth': float, 'area': int}}
        """
        logger.info(f"Generating lakes (min_size={min_size})...")

        # Find local minima (potential lake locations)
        local_min = self.heightmap == maximum_filter(self.heightmap, size=5, mode='constant', cval=1.0)

        # Watershed to find depressions
        markers = label(local_min)[0]
        labels = watershed(self.heightmap, markers)

        lakes = {}

        for region_id in range(1, labels.max() + 1):
            mask = labels == region_id
            area = np.sum(mask)

            if area < min_size:
                continue

            # Calculate depression depth
            region_heights = self.heightmap[mask]
            min_height = region_heights.min()
            max_edge_height = self.heightmap[binary_dilation(mask) & ~mask].min()

            depth = max_edge_height - min_height

            if depth > 0.02:  # Minimum depression depth
                lakes[len(lakes)] = {
                    'mask': mask,
                    'depth': depth,
                    'area': area,
                    'water_level': min_height + depth * water_level
                }

        logger.info(f"Generated {len(lakes)} lakes")

        return lakes

    def fill_lakes(self, water_level_offset: float = 0.01) -> np.ndarray:
        """
        Fill lakes with water (modify heightmap to flat water level).

        Args:
            water_level_offset: Offset above minimum to fill

        Returns:
            Heightmap with filled lakes
        """
        filled = self.heightmap.copy()

        for lake_id, lake in self.lakes.items():
            mask = lake['mask']
            water_level = lake['water_level'] + water_level_offset

            # Set all lake pixels to water level
            filled[mask] = water_level

        return filled

    def detect_waterfalls(self, min_drop: float = 0.1) -> List[Dict]:
        """
        Detect waterfall locations (steep drops along rivers).

        Args:
            min_drop: Minimum height drop for waterfall

        Returns:
            List of waterfall dictionaries
        """
        logger.info(f"Detecting waterfalls (min_drop={min_drop})...")

        waterfalls = []

        for river_pixels in self.rivers:
            # Sort pixels by flow direction (upstream to downstream)
            sorted_pixels = sorted(river_pixels, key=lambda p: self.heightmap[p[0], p[1]], reverse=True)

            for idx in range(len(sorted_pixels) - 1):
                i1, j1 = sorted_pixels[idx]
                i2, j2 = sorted_pixels[idx + 1]

                height_drop = self.heightmap[i1, j1] - self.heightmap[i2, j2]

                if height_drop > min_drop:
                    waterfalls.append({
                        'top': (i1, j1),
                        'bottom': (i2, j2),
                        'drop': height_drop
                    })

        logger.info(f"Detected {len(waterfalls)} waterfalls")

        return waterfalls

    def get_water_mask(self) -> np.ndarray:
        """
        Get binary mask of all water (rivers + lakes).

        Returns:
            Water mask (H, W) bool
        """
        water_mask = np.zeros((self.h, self.w), dtype=bool)

        # Add rivers
        for river_pixels in self.rivers:
            for i, j in river_pixels:
                # Dilate river to make it visible
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.h and 0 <= nj < self.w:
                            water_mask[ni, nj] = True

        # Add lakes
        for lake in self.lakes.values():
            water_mask |= lake['mask']

        return water_mask

    def get_water_depth_map(self) -> np.ndarray:
        """
        Get water depth map (for rendering).

        Returns:
            Water depth (H, W) float [0-1]
        """
        depth_map = np.zeros((self.h, self.w), dtype=np.float32)

        # Lakes have depth
        for lake in self.lakes.values():
            mask = lake['mask']
            water_level = lake['water_level']

            # Depth = water_level - terrain_height
            depth_map[mask] = water_level - self.heightmap[mask]

        # Rivers have shallow depth
        for river_pixels in self.rivers:
            for i, j in river_pixels:
                depth_map[i, j] = max(depth_map[i, j], 0.01)

        return depth_map

    def __repr__(self):
        return f"WaterSystem(rivers={len(self.rivers)}, lakes={len(self.lakes)}, waterfalls={len(self.waterfalls)})"


class WaterRenderer:
    """
    Render water with realistic effects.

    Features:
    - Reflection
    - Refraction
    - Caustics
    - Foam
    - Waves
    """

    def __init__(self):
        self.water_color = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        self.transparency = 0.7
        self.reflection_strength = 0.5
        self.wave_height = 0.01
        self.wave_frequency = 10.0

    def render_water(self, terrain_image: np.ndarray, water_mask: np.ndarray,
                     water_depth: np.ndarray) -> np.ndarray:
        """
        Render water on terrain image.

        Args:
            terrain_image: Base terrain render (H, W, 3)
            water_mask: Water mask (H, W) bool
            water_depth: Water depth map (H, W)

        Returns:
            Image with water rendered
        """
        result = terrain_image.copy()

        # Simple water rendering (blend with water color based on depth)
        for c in range(3):
            water_layer = self.water_color[c] * np.ones_like(water_depth)
            alpha = np.clip(water_depth * 5.0, 0, self.transparency)

            result[:, :, c] = np.where(
                water_mask,
                result[:, :, c] * (1 - alpha) + water_layer * alpha,
                result[:, :, c]
            )

        return result
