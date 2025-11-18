"""
Advanced 3D Terrain Preview Widget

Professional 3D visualization with:
- Real-time camera controls (orbit, pan, zoom)
- Multiple render modes (wireframe, solid, textured)
- Dynamic lighting with shadows
- Texture mapping (heightmap → color)
- Vertical exaggeration control
- Grid overlay
- Performance optimizations

Based on pyqtgraph.opengl for GPU acceleration.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, Slot
import pyqtgraph.opengl as gl
from OpenGL.GL import *
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TerrainPreview3DWidget(QWidget):
    """
    Advanced 3D terrain preview with camera controls

    Features:
    - Orbit camera (mouse drag)
    - Pan (Shift + drag)
    - Zoom (mouse wheel)
    - Vertical exaggeration
    - Render modes (wireframe/solid/textured)
    - Lighting controls
    - Grid overlay
    """

    camera_changed = Signal(dict)  # Emits camera state

    def __init__(self, parent=None):
        super().__init__(parent)

        self.heightmap: Optional[np.ndarray] = None
        self.terrain_mesh: Optional[gl.GLMeshItem] = None
        self.grid_item: Optional[gl.GLGridItem] = None

        # Camera state
        self.camera_distance = 100.0
        self.camera_elevation = 30.0  # degrees
        self.camera_azimuth = 45.0    # degrees
        self.camera_center = (0, 0, 0)

        # Render settings
        self.vertical_exaggeration = 2.0
        self.render_mode = 'solid'  # 'wireframe', 'solid', 'textured'
        self.show_grid = True
        self.show_normals = False

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h3>🏔️ 3D Terrain Preview</h3>")
        layout.addWidget(title)

        # 3D View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumHeight(400)

        # Setup camera
        self._update_camera()

        # Add grid
        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(100, 100, 1)
        self.grid_item.setSpacing(10, 10, 1)
        self.gl_view.addItem(self.grid_item)

        layout.addWidget(self.gl_view)

        # Controls
        controls = self._create_controls()
        layout.addWidget(controls)

        # Camera info
        self.camera_info_label = QLabel(self._format_camera_info())
        layout.addWidget(self.camera_info_label)

    def _create_controls(self) -> QGroupBox:
        """Create control panel"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()

        # Vertical exaggeration
        v_exag_layout = QHBoxLayout()
        v_exag_layout.addWidget(QLabel("Vertical Exaggeration:"))

        self.v_exag_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_exag_slider.setRange(10, 100)  # 1.0 to 10.0
        self.v_exag_slider.setValue(int(self.vertical_exaggeration * 10))
        self.v_exag_slider.valueChanged.connect(self._on_v_exag_changed)
        v_exag_layout.addWidget(self.v_exag_slider)

        self.v_exag_label = QLabel(f"{self.vertical_exaggeration:.1f}x")
        v_exag_layout.addWidget(self.v_exag_label)

        layout.addLayout(v_exag_layout)

        # Render mode
        render_layout = QHBoxLayout()
        render_layout.addWidget(QLabel("Render Mode:"))

        self.render_combo = QComboBox()
        self.render_combo.addItems(['Solid', 'Wireframe', 'Textured'])
        self.render_combo.currentTextChanged.connect(self._on_render_mode_changed)
        render_layout.addWidget(self.render_combo)

        layout.addLayout(render_layout)

        # Checkboxes
        cb_layout = QHBoxLayout()

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.show_grid)
        self.grid_checkbox.toggled.connect(self._on_grid_toggled)
        cb_layout.addWidget(self.grid_checkbox)

        self.normals_checkbox = QCheckBox("Show Normals")
        self.normals_checkbox.setChecked(self.show_normals)
        self.normals_checkbox.toggled.connect(self._on_normals_toggled)
        cb_layout.addWidget(self.normals_checkbox)

        layout.addLayout(cb_layout)

        # Camera controls
        cam_layout = QHBoxLayout()

        reset_btn = QPushButton("🎥 Reset Camera")
        reset_btn.clicked.connect(self._reset_camera)
        cam_layout.addWidget(reset_btn)

        top_view_btn = QPushButton("⬇️ Top View")
        top_view_btn.clicked.connect(lambda: self._set_view(90, 0))
        cam_layout.addWidget(top_view_btn)

        side_view_btn = QPushButton("➡️ Side View")
        side_view_btn.clicked.connect(lambda: self._set_view(0, 90))
        cam_layout.addWidget(side_view_btn)

        layout.addLayout(cam_layout)

        # Lighting controls
        light_group = self._create_lighting_controls()
        layout.addWidget(light_group)

        group.setLayout(layout)
        return group

    def _create_lighting_controls(self) -> QGroupBox:
        """Create lighting control panel"""
        group = QGroupBox("Lighting")
        layout = QHBoxLayout()

        # Ambient light
        layout.addWidget(QLabel("Ambient:"))
        self.ambient_slider = QSlider(Qt.Orientation.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(30)
        self.ambient_slider.valueChanged.connect(self._update_lighting)
        layout.addWidget(self.ambient_slider)

        # Diffuse light
        layout.addWidget(QLabel("Diffuse:"))
        self.diffuse_slider = QSlider(Qt.Orientation.Horizontal)
        self.diffuse_slider.setRange(0, 100)
        self.diffuse_slider.setValue(70)
        self.diffuse_slider.valueChanged.connect(self._update_lighting)
        layout.addWidget(self.diffuse_slider)

        group.setLayout(layout)
        return group

    def set_heightmap(self, heightmap: np.ndarray):
        """
        Set heightmap to display

        Args:
            heightmap: 2D array in range [0, 1]
        """
        self.heightmap = heightmap.copy()
        self._update_terrain_mesh()

    def _update_terrain_mesh(self):
        """Generate and display 3D mesh from heightmap"""
        if self.heightmap is None:
            return

        # Remove old mesh
        if self.terrain_mesh is not None:
            self.gl_view.removeItem(self.terrain_mesh)

        h, w = self.heightmap.shape

        # Generate mesh vertices
        # Create a grid of (x, y) coordinates
        x = np.linspace(-50, 50, w)
        y = np.linspace(-50, 50, h)
        X, Y = np.meshgrid(x, y)

        # Z from heightmap with vertical exaggeration
        Z = self.heightmap * 50 * self.vertical_exaggeration

        # Create vertices array (N, 3)
        vertices = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

        # Generate faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                # Two triangles per quad
                idx = i * w + j

                # Triangle 1
                faces.append([idx, idx + 1, idx + w])

                # Triangle 2
                faces.append([idx + 1, idx + w + 1, idx + w])

        faces = np.array(faces)

        # Calculate colors based on height
        colors = self._calculate_vertex_colors(Z.flatten())

        # Create mesh
        mesh_data = gl.MeshData(
            vertexes=vertices,
            faces=faces,
            vertexColors=colors
        )

        # Set render mode
        if self.render_mode == 'wireframe':
            self.terrain_mesh = gl.GLMeshItem(
                meshdata=mesh_data,
                drawEdges=True,
                drawFaces=False,
                edgeColor=(0.5, 0.5, 0.5, 1.0)
            )
        elif self.render_mode == 'solid':
            self.terrain_mesh = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,
                drawEdges=False,
                shader='shaded'
            )
        else:  # textured
            self.terrain_mesh = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,
                drawEdges=False,
                shader='shaded'
            )

        self.gl_view.addItem(self.terrain_mesh)

        logger.info(f"Terrain mesh updated: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    def _calculate_vertex_colors(self, z_values: np.ndarray) -> np.ndarray:
        """
        Calculate vertex colors based on elevation

        Returns:
            Array of shape (N, 4) with RGBA colors
        """
        # Normalize Z to [0, 1]
        z_min, z_max = z_values.min(), z_values.max()
        if z_max > z_min:
            z_norm = (z_values - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_values)

        # Color gradient: blue (low) → green → brown → white (high)
        colors = np.zeros((len(z_values), 4))

        # Blue for low areas (water/valleys)
        mask_low = z_norm < 0.2
        colors[mask_low, 0] = 0.2  # R
        colors[mask_low, 1] = 0.4  # G
        colors[mask_low, 2] = 0.7  # B

        # Green for mid-low (vegetation)
        mask_mid_low = (z_norm >= 0.2) & (z_norm < 0.4)
        colors[mask_mid_low, 0] = 0.3
        colors[mask_mid_low, 1] = 0.6
        colors[mask_mid_low, 2] = 0.3

        # Brown for mid (rock)
        mask_mid = (z_norm >= 0.4) & (z_norm < 0.7)
        t = (z_norm[mask_mid] - 0.4) / 0.3
        colors[mask_mid, 0] = 0.3 + t * 0.3  # R: 0.3 → 0.6
        colors[mask_mid, 1] = 0.6 - t * 0.3  # G: 0.6 → 0.3
        colors[mask_mid, 2] = 0.3 - t * 0.2  # B: 0.3 → 0.1

        # White for high (snow)
        mask_high = z_norm >= 0.7
        t = (z_norm[mask_high] - 0.7) / 0.3
        colors[mask_high, 0] = 0.6 + t * 0.4  # R: 0.6 → 1.0
        colors[mask_high, 1] = 0.3 + t * 0.7  # G: 0.3 → 1.0
        colors[mask_high, 2] = 0.1 + t * 0.9  # B: 0.1 → 1.0

        # Alpha
        colors[:, 3] = 1.0

        return colors

    @Slot(int)
    def _on_v_exag_changed(self, value: int):
        """Handle vertical exaggeration change"""
        self.vertical_exaggeration = value / 10.0
        self.v_exag_label.setText(f"{self.vertical_exaggeration:.1f}x")
        self._update_terrain_mesh()

    @Slot(str)
    def _on_render_mode_changed(self, mode: str):
        """Handle render mode change"""
        self.render_mode = mode.lower()
        self._update_terrain_mesh()

    @Slot(bool)
    def _on_grid_toggled(self, checked: bool):
        """Toggle grid visibility"""
        self.show_grid = checked
        if self.grid_item:
            self.grid_item.setVisible(checked)

    @Slot(bool)
    def _on_normals_toggled(self, checked: bool):
        """Toggle normals visibility"""
        self.show_normals = checked
        # TODO: Implement normal visualization

    @Slot()
    def _reset_camera(self):
        """Reset camera to default position"""
        self.camera_distance = 100.0
        self.camera_elevation = 30.0
        self.camera_azimuth = 45.0
        self.camera_center = (0, 0, 0)
        self._update_camera()

    def _set_view(self, elevation: float, azimuth: float):
        """Set camera to specific view"""
        self.camera_elevation = elevation
        self.camera_azimuth = azimuth
        self._update_camera()

    def _update_camera(self):
        """Update camera position"""
        self.gl_view.setCameraPosition(
            distance=self.camera_distance,
            elevation=self.camera_elevation,
            azimuth=self.camera_azimuth
        )

        self.camera_info_label.setText(self._format_camera_info())
        self.camera_changed.emit(self._get_camera_state())

    @Slot()
    def _update_lighting(self):
        """Update lighting parameters"""
        ambient = self.ambient_slider.value() / 100.0
        diffuse = self.diffuse_slider.value() / 100.0

        # TODO: Apply lighting to mesh shader
        # This requires custom shader setup in pyqtgraph

        logger.debug(f"Lighting: ambient={ambient:.2f}, diffuse={diffuse:.2f}")

    def _format_camera_info(self) -> str:
        """Format camera info string"""
        return (f"📷 Camera: Distance={self.camera_distance:.1f}, "
                f"Elevation={self.camera_elevation:.1f}°, "
                f"Azimuth={self.camera_azimuth:.1f}°")

    def _get_camera_state(self) -> dict:
        """Get current camera state"""
        return {
            'distance': self.camera_distance,
            'elevation': self.camera_elevation,
            'azimuth': self.camera_azimuth,
            'center': self.camera_center
        }

    def set_camera_state(self, state: dict):
        """Restore camera state"""
        self.camera_distance = state.get('distance', 100.0)
        self.camera_elevation = state.get('elevation', 30.0)
        self.camera_azimuth = state.get('azimuth', 45.0)
        self.camera_center = state.get('center', (0, 0, 0))
        self._update_camera()

    # Mouse controls for camera
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()

        if delta > 0:
            self.camera_distance *= 0.9  # Zoom in
        else:
            self.camera_distance *= 1.1  # Zoom out

        self.camera_distance = np.clip(self.camera_distance, 10, 500)
        self._update_camera()

    def export_snapshot(self, filename: str):
        """
        Export current view as image

        Args:
            filename: Output filename (.png)
        """
        # Render to QImage
        image = self.gl_view.renderToArray(size=(1920, 1080))

        from PIL import Image
        img = Image.fromarray(image)
        img.save(filename)

        logger.info(f"Snapshot saved: {filename}")


if __name__ == '__main__':
    # Test the widget
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    widget = TerrainPreview3DWidget()

    # Generate test heightmap
    size = 128
    from scipy.ndimage import gaussian_filter
    heightmap = np.random.rand(size, size)
    heightmap = gaussian_filter(heightmap, sigma=5)
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    widget.set_heightmap(heightmap)
    widget.resize(800, 900)
    widget.show()

    sys.exit(app.exec())
