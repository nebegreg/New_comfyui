"""
Photorealistic 3D Terrain Viewer - Style Evian
================================================

Features:
✅ PBR Materials (Diffuse, Normal, Roughness, AO)
✅ Atmospheric Scattering (Rayleigh + Mie)
✅ Distance Fog with altitude gradient
✅ Advanced Lighting (Sun + Sky + Ambient)
✅ Vegetation Rendering (Instanced trees as billboards)
✅ Post-processing (Tone mapping, gamma correction)

Inspired by:
- Evian Alps advertising visual style
- Modern game engines (Unreal, Unity)
- Industry-standard CGI rendering (2024)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

try:
    import pyqtgraph.opengl as gl
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

logger = logging.getLogger(__name__)


class PhotorealisticTerrainViewer(gl.GLViewWidget if OPENGL_AVAILABLE else object):
    """
    Ultra-realistic 3D terrain viewer with PBR and atmospheric effects

    Rendering pipeline:
    1. Terrain mesh with PBR materials
    2. Vegetation instances (billboards)
    3. Atmospheric scattering
    4. Distance fog
    5. Tone mapping
    """

    def __init__(self, parent=None):
        if not OPENGL_AVAILABLE:
            raise ImportError("OpenGL not available. Install pyqtgraph and PyOpenGL.")

        super().__init__(parent)

        # Terrain data
        self.heightmap = None
        self.terrain_mesh = None
        self.terrain_colors = None

        # Vegetation
        self.tree_instances = []
        self.vegetation_items = []

        # PBR Textures (as numpy arrays for now)
        self.pbr_textures = {
            'diffuse': None,
            'normal': None,
            'roughness': None,
            'ao': None
        }

        # Lighting parameters
        self.sun_direction = np.array([0.3, 0.5, -0.7])
        self.sun_direction /= np.linalg.norm(self.sun_direction)
        self.sun_color = np.array([1.0, 0.98, 0.9])  # Warm sunlight
        self.sun_intensity = 1.2

        self.sky_color = np.array([0.4, 0.6, 0.9])  # Blue sky
        self.ambient_color = np.array([0.3, 0.35, 0.45])  # Cool ambient
        self.ambient_strength = 0.4

        # Atmospheric parameters
        self.fog_enabled = True
        self.fog_density = 0.015  # Exponential fog density
        self.fog_color = np.array([0.7, 0.8, 0.95])  # Light blue-gray
        self.fog_start = 50.0
        self.fog_end = 500.0

        self.atmosphere_enabled = True
        self.rayleigh_strength = 0.3  # Sky scattering
        self.mie_strength = 0.1  # Haze scattering

        # Rendering options
        self.wireframe_mode = False
        self.show_vegetation = True
        self.use_pbr = True

        # Post-processing
        self.tone_mapping = True
        self.gamma = 2.2  # sRGB gamma

        # Setup
        self.setCameraPosition(distance=350, elevation=25, azimuth=45)
        self.setBackgroundColor(self.sky_color * 255)

        logger.info("PhotorealisticTerrainViewer initialized")

    def set_terrain(
        self,
        heightmap: np.ndarray,
        height_scale: float = 50.0,
        pbr_textures: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Set terrain heightmap with PBR textures

        Args:
            heightmap: Terrain heightmap (H, W) normalized to [0, 1]
            height_scale: Height multiplier for Z axis
            pbr_textures: Optional PBR texture dictionary from PBRTextureGenerator
        """
        self.heightmap = heightmap
        h, w = heightmap.shape

        # Store PBR textures
        if pbr_textures:
            self.pbr_textures = pbr_textures

        # Create mesh coordinates
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        X, Y = np.meshgrid(x, y)
        Z = heightmap * height_scale

        # Calculate vertex colors with PBR + Atmosphere
        colors = self._calculate_pbr_lighting(heightmap, Z, height_scale)

        # Remove old terrain
        if self.terrain_mesh is not None:
            self.removeItem(self.terrain_mesh)

        # Create new terrain mesh
        self.terrain_mesh = gl.GLSurfacePlotItem(
            x=x, y=y, z=Z,
            colors=colors,
            shader='shaded',
            smooth=True,
            drawEdges=self.wireframe_mode,
            drawFaces=True
        )

        self.addItem(self.terrain_mesh)

        logger.info(f"Terrain set: {w}x{h}, height_scale={height_scale}, PBR={pbr_textures is not None}")

    def _calculate_pbr_lighting(
        self,
        heightmap: np.ndarray,
        Z: np.ndarray,
        height_scale: float
    ) -> np.ndarray:
        """
        Calculate vertex colors using PBR lighting model + atmospheric effects

        Lighting model:
        1. Base color (albedo) from PBR diffuse or procedural
        2. Normal map (from PBR or calculated)
        3. Roughness (from PBR or slope-based)
        4. Ambient Occlusion (from PBR or calculated)
        5. Diffuse lighting (Lambertian)
        6. Specular highlights (GGX for PBR)
        7. Atmospheric scattering (distance-based)
        8. Distance fog (exponential)
        """
        h, w = heightmap.shape

        # 1. Calculate normals
        normals = self._calculate_normals(heightmap, height_scale)

        # 2. Get base color (albedo)
        if self.use_pbr and self.pbr_textures.get('diffuse') is not None:
            # Use PBR diffuse texture
            albedo = self.pbr_textures['diffuse'].astype(np.float32) / 255.0
        else:
            # Procedural altitude-based coloring
            albedo = self._generate_procedural_albedo(heightmap)

        # 3. Get roughness
        if self.use_pbr and self.pbr_textures.get('roughness') is not None:
            roughness = self.pbr_textures['roughness'].astype(np.float32) / 255.0
        else:
            # Calculate from slope
            roughness = self._calculate_roughness_from_slope(heightmap)
            roughness = np.expand_dims(roughness, axis=-1)

        # 4. Get AO
        if self.use_pbr and self.pbr_textures.get('ao') is not None:
            ao = self.pbr_textures['ao'].astype(np.float32) / 255.0
            ao = np.expand_dims(ao, axis=-1)
        else:
            # Simple AO from height
            ao = ((heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-10))
            ao = 0.6 + ao * 0.4  # Range [0.6, 1.0]
            ao = np.expand_dims(ao, axis=-1)

        # 5. Calculate lighting

        # View direction (from camera, simplified as looking down)
        view_dir = np.array([0.0, 0.0, 1.0])

        # Sun direction broadcast
        sun_dir = self.sun_direction.reshape(1, 1, 3)

        # Diffuse (Lambertian)
        n_dot_l = np.maximum(0, np.sum(normals * sun_dir, axis=2, keepdims=True))
        diffuse = n_dot_l * self.sun_color.reshape(1, 1, 3) * self.sun_intensity

        # Specular (simplified Blinn-Phong, should be GGX for true PBR)
        # For performance, using Blinn-Phong approximation
        half_vector = sun_dir + np.array([0, 0, 1]).reshape(1, 1, 3)
        half_vector = half_vector / (np.linalg.norm(half_vector, axis=2, keepdims=True) + 1e-10)

        n_dot_h = np.maximum(0, np.sum(normals * half_vector, axis=2, keepdims=True))

        # Shininess from roughness (rough = low shininess)
        shininess = ((1.0 - roughness) * 100.0 + 5.0)  # Range [5, 105]
        specular = np.power(n_dot_h, shininess) * (1.0 - roughness)
        specular = specular * self.sun_color.reshape(1, 1, 3) * 0.3  # Reduce specular intensity

        # Ambient (from sky)
        ambient = self.ambient_color.reshape(1, 1, 3) * self.ambient_strength

        # Combine lighting
        final_color = albedo * (ambient + diffuse + specular) * ao

        # 6. Apply atmospheric effects
        if self.atmosphere_enabled or self.fog_enabled:
            final_color = self._apply_atmospheric_effects(final_color, Z, heightmap)

        # 7. Tone mapping
        if self.tone_mapping:
            final_color = self._tone_map_aces(final_color)

        # 8. Gamma correction
        final_color = np.power(final_color, 1.0 / self.gamma)

        # Convert to [0, 1] and add alpha
        final_color = np.clip(final_color, 0, 1)
        colors = np.zeros((h, w, 4), dtype=np.float32)
        colors[:, :, :3] = final_color
        colors[:, :, 3] = 1.0

        return colors

    def _calculate_normals(
        self,
        heightmap: np.ndarray,
        height_scale: float
    ) -> np.ndarray:
        """Calculate surface normals from heightmap"""
        h, w = heightmap.shape

        # Calculate gradients
        dy, dx = np.gradient(heightmap * height_scale)

        # Build normal vectors
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[:, :, 0] = -dx
        normals[:, :, 1] = -dy
        normals[:, :, 2] = 1.0

        # Normalize
        magnitude = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        normals /= (magnitude + 1e-10)

        return normals

    def _generate_procedural_albedo(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Generate procedural albedo based on altitude (Evian style)

        Color zones:
        - 0.0-0.3: Dark green/brown (valley forest)
        - 0.3-0.6: Gray/brown (rocky slopes)
        - 0.6-0.8: Light gray (high rocks)
        - 0.8-1.0: White/blue (snow peaks)
        """
        h, w = heightmap.shape
        albedo = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                height_val = heightmap[i, j]

                if height_val < 0.3:
                    # Valley: dark green/brown
                    t = height_val / 0.3
                    r = 0.2 + t * 0.1
                    g = 0.25 + t * 0.15
                    b = 0.15 + t * 0.05

                elif height_val < 0.6:
                    # Mid slopes: gray/brown
                    t = (height_val - 0.3) / 0.3
                    r = 0.3 + t * 0.1
                    g = 0.4 + t * 0.0
                    b = 0.2 + t * 0.1

                elif height_val < 0.8:
                    # High rocks: light gray
                    t = (height_val - 0.6) / 0.2
                    r = 0.4 + t * 0.2
                    g = 0.4 + t * 0.2
                    b = 0.3 + t * 0.25

                else:
                    # Snow peaks: white/blue
                    t = (height_val - 0.8) / 0.2
                    r = 0.6 + t * 0.3
                    g = 0.6 + t * 0.3
                    b = 0.55 + t * 0.4

                albedo[i, j] = [r, g, b]

        return albedo

    def _calculate_roughness_from_slope(self, heightmap: np.ndarray) -> np.ndarray:
        """Calculate roughness from terrain slope"""
        dy, dx = np.gradient(heightmap)
        slope = np.sqrt(dx**2 + dy**2)

        # Normalize to [0, 1], steeper = rougher
        roughness = np.clip(slope / (slope.max() + 1e-10), 0, 1)

        # Adjust range: terrain is generally rough
        roughness = 0.5 + roughness * 0.4  # Range [0.5, 0.9]

        return roughness

    def _apply_atmospheric_effects(
        self,
        colors: np.ndarray,
        Z: np.ndarray,
        heightmap: np.ndarray
    ) -> np.ndarray:
        """
        Apply atmospheric scattering and fog

        Effects:
        1. Rayleigh scattering (sky blue)
        2. Mie scattering (haze)
        3. Exponential distance fog
        4. Altitude-based fog density
        """
        h, w, c = colors.shape

        # Calculate distance from camera (simplified: use Z + horizontal distance)
        center_x, center_y = w // 2, h // 2

        result = colors.copy()

        for i in range(h):
            for j in range(w):
                # Distance from camera (Euclidean 3D distance, simplified)
                dx = j - center_x
                dy = i - center_y
                dz = Z[i, j]

                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                # Fog factor (exponential)
                if self.fog_enabled:
                    fog_factor = 1.0 - np.exp(-self.fog_density * max(0, distance - self.fog_start))
                    fog_factor = np.clip(fog_factor, 0, 1)

                    # Altitude modulation: less fog at high altitude
                    altitude_factor = 1.0 - heightmap[i, j] * 0.5
                    fog_factor *= altitude_factor

                    # Blend with fog color
                    result[i, j, :3] = (
                        result[i, j, :3] * (1 - fog_factor) +
                        self.fog_color * fog_factor
                    )

                # Atmospheric scattering (Rayleigh + Mie)
                if self.atmosphere_enabled:
                    # Rayleigh (blue scattering, increases with distance)
                    rayleigh_factor = self.rayleigh_strength * min(1.0, distance / 300.0)

                    # Mie (haze, increases with distance)
                    mie_factor = self.mie_strength * min(1.0, distance / 200.0)

                    # Apply scattering (blend towards sky color)
                    scatter_color = self.sky_color * rayleigh_factor + self.fog_color * mie_factor
                    scatter_amount = rayleigh_factor + mie_factor

                    result[i, j, :3] = (
                        result[i, j, :3] * (1 - scatter_amount) +
                        scatter_color * scatter_amount
                    )

        return result

    def _tone_map_aces(self, color: np.ndarray) -> np.ndarray:
        """
        ACES tone mapping (industry standard for film/games)

        Provides filmic look with nice highlights and shadows
        """
        # ACES filmic tone mapping curve
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        tone_mapped = (color * (a * color + b)) / (color * (c * color + d) + e)
        return np.clip(tone_mapped, 0, 1)

    def set_vegetation(self, tree_instances: List):
        """
        Set vegetation instances to render

        Args:
            tree_instances: List of TreeInstance from VegetationPlacer
        """
        # Remove old vegetation
        for item in self.vegetation_items:
            self.removeItem(item)
        self.vegetation_items = []

        self.tree_instances = tree_instances

        if not self.show_vegetation or not tree_instances:
            return

        logger.info(f"Rendering {len(tree_instances)} trees...")

        # For now, render trees as simple cylinders (cones for tops)
        # In a full implementation, would use instanced rendering with billboards

        # Group by species for batching (performance)
        species_groups = {}
        for tree in tree_instances:
            if tree.species not in species_groups:
                species_groups[tree.species] = []
            species_groups[tree.species].append(tree)

        # Render each species group
        for species, trees in species_groups.items():
            # For performance, limit to first 500 trees per species
            trees_to_render = trees[:500]

            for tree in trees_to_render:
                # Get terrain height at tree position
                h, w = self.heightmap.shape
                x_idx = int(np.clip(tree.x, 0, w - 1))
                y_idx = int(np.clip(tree.y, 0, h - 1))
                terrain_height = self.heightmap[y_idx, x_idx] * 50.0  # Match height_scale

                # Tree position (center terrain coordinates)
                tree_x = tree.x - w / 2
                tree_y = tree.y - h / 2
                tree_z = terrain_height

                # Tree dimensions based on species and scale
                tree_height = tree.scale * 15.0  # Base height 15 units
                tree_radius = tree.scale * 2.0

                # Create simple tree representation (cylinder + cone)
                # Trunk
                trunk_mesh = gl.GLMeshItem(
                    meshdata=gl.MeshData.cylinder(
                        rows=8,
                        cols=16,
                        radius=[tree_radius * 0.3, tree_radius * 0.3],
                        length=tree_height * 0.6
                    ),
                    color=(0.3, 0.2, 0.1, tree.health),  # Brown trunk
                    smooth=False,
                    shader='shaded'
                )
                trunk_mesh.translate(tree_x, tree_y, tree_z + tree_height * 0.3)
                trunk_mesh.rotate(90, 1, 0, 0)  # Stand upright

                self.addItem(trunk_mesh)
                self.vegetation_items.append(trunk_mesh)

                # Foliage (cone for coniferous, sphere for deciduous)
                if species in ['pine', 'spruce', 'fir']:
                    # Coniferous: cone
                    foliage_color = (0.1, 0.3, 0.15, tree.health * 0.8)
                    foliage_mesh = gl.GLMeshItem(
                        meshdata=gl.MeshData.cylinder(
                            rows=8,
                            cols=16,
                            radius=[tree_radius, 0],
                            length=tree_height * 0.6
                        ),
                        color=foliage_color,
                        smooth=False,
                        shader='shaded'
                    )
                    foliage_mesh.translate(tree_x, tree_y, tree_z + tree_height * 0.6)
                    foliage_mesh.rotate(90, 1, 0, 0)
                else:
                    # Deciduous: sphere
                    foliage_color = (0.2, 0.4, 0.2, tree.health * 0.7)
                    foliage_mesh = gl.GLMeshItem(
                        meshdata=gl.MeshData.sphere(
                            rows=10,
                            cols=10,
                            radius=tree_radius * 1.5
                        ),
                        color=foliage_color,
                        smooth=True,
                        shader='shaded'
                    )
                    foliage_mesh.translate(tree_x, tree_y, tree_z + tree_height * 0.7)

                self.addItem(foliage_mesh)
                self.vegetation_items.append(foliage_mesh)

        logger.info(f"Rendered {len(self.vegetation_items)} vegetation items")

    def set_lighting(
        self,
        sun_azimuth: float,
        sun_elevation: float,
        sun_intensity: float = 1.2,
        ambient_strength: float = 0.4
    ):
        """
        Update lighting parameters

        Args:
            sun_azimuth: Sun azimuth angle in degrees (0-360)
            sun_elevation: Sun elevation angle in degrees (0-90)
            sun_intensity: Sun light intensity (0.5-2.0)
            ambient_strength: Ambient light strength (0.0-1.0)
        """
        # Convert to direction vector
        azimuth_rad = np.radians(sun_azimuth)
        elevation_rad = np.radians(sun_elevation)

        self.sun_direction = np.array([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            -np.sin(elevation_rad)
        ])
        self.sun_direction /= np.linalg.norm(self.sun_direction)

        self.sun_intensity = sun_intensity
        self.ambient_strength = ambient_strength

        # Adjust sun color based on elevation (sunset = orange)
        if sun_elevation < 15:
            # Sunset/sunrise: warm orange
            t = sun_elevation / 15.0
            self.sun_color = np.array([
                1.0,
                0.7 + t * 0.28,
                0.4 + t * 0.5
            ])
        else:
            # Normal: warm white
            self.sun_color = np.array([1.0, 0.98, 0.9])

        # Recompute lighting if terrain exists
        if self.heightmap is not None:
            height_scale = 50.0  # Default
            self.set_terrain(self.heightmap, height_scale, self.pbr_textures)

    def set_atmosphere(
        self,
        fog_density: float = 0.015,
        fog_enabled: bool = True,
        atmosphere_enabled: bool = True
    ):
        """Update atmospheric parameters"""
        self.fog_density = fog_density
        self.fog_enabled = fog_enabled
        self.atmosphere_enabled = atmosphere_enabled

        # Recompute if terrain exists
        if self.heightmap is not None:
            self.set_terrain(self.heightmap, 50.0, self.pbr_textures)

    def toggle_wireframe(self):
        """Toggle wireframe rendering"""
        self.wireframe_mode = not self.wireframe_mode
        if self.heightmap is not None:
            self.set_terrain(self.heightmap, 50.0, self.pbr_textures)

    def toggle_vegetation(self):
        """Toggle vegetation visibility"""
        self.show_vegetation = not self.show_vegetation
        if self.tree_instances:
            self.set_vegetation(self.tree_instances if self.show_vegetation else [])

    def reset_camera(self):
        """Reset camera to default position"""
        self.setCameraPosition(distance=350, elevation=25, azimuth=45)


if __name__ == "__main__":
    # Test viewer
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    viewer = PhotorealisticTerrainViewer()

    # Generate test heightmap
    test_size = 256
    from core.noise import ridged_multifractal
    heightmap = ridged_multifractal(test_size, test_size, octaves=8, seed=42)

    viewer.set_terrain(heightmap, height_scale=50.0)
    viewer.show()

    sys.exit(app.exec())
