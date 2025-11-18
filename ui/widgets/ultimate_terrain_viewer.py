"""
Ultimate Terrain Viewer - Complete Integration
Mountain Studio Pro

Integrates:
- Advanced OpenGL viewer with shadows
- FPS camera controls
- HDRI panoramic generator
- Real-time shadow mapping
- Complete UI controls

Author: Mountain Studio Pro
"""

import numpy as np
from pathlib import Path
from typing import Optional
import logging

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QComboBox, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QProgressBar,
    QTabWidget, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont

try:
    from ui.widgets.advanced_terrain_viewer import AdvancedTerrainViewer, OPENGL_AVAILABLE
except ImportError:
    OPENGL_AVAILABLE = False

from core.rendering.hdri_generator import HDRIPanoramicGenerator, TimeOfDay, AI_AVAILABLE
from core.terrain.heightmap_generator_v2 import HeightmapGeneratorV2
from core.terrain.advanced_algorithms import spectral_synthesis, stream_power_erosion, MOUNTAIN_PRESETS

logger = logging.getLogger(__name__)


class UltimateTerrainViewer(QMainWindow):
    """
    Ultimate terrain viewer with all advanced features.

    Features:
    - Advanced OpenGL rendering with shadows
    - FPS camera controls
    - HDRI generation and loading
    - Real-time parameter adjustment
    - Export capabilities
    """

    terrain_generated = Signal(np.ndarray)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize ultimate viewer."""
        super().__init__(parent)

        if not OPENGL_AVAILABLE:
            QMessageBox.critical(self, "Error", "PyOpenGL is required for Ultimate Viewer")
            raise ImportError("PyOpenGL not available")

        self.setWindowTitle("Mountain Studio Pro - Ultimate Terrain Viewer")
        self.resize(1600, 900)

        # Current terrain
        self._current_heightmap = None
        self._terrain_scale = 100.0
        self._height_scale = 20.0

        # HDRI generator
        self._hdri_generator = None

        # Setup UI
        self._setup_ui()

        # FPS update timer
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps_display)
        self._fps_timer.start(500)  # Update FPS every 0.5s

        logger.info("Ultimate Terrain Viewer initialized")

    def _setup_ui(self):
        """Setup complete UI."""
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)

        # Left: 3D Viewer
        self.viewer = AdvancedTerrainViewer()
        splitter.addWidget(self.viewer)

        # Right: Control Panel
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        # Set splitter sizes (70% viewer, 30% controls)
        splitter.setSizes([1120, 480])

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")
        self._fps_label = QLabel("FPS: 0")
        self._camera_label = QLabel("Camera: [0, 0, 0]")
        self.statusBar().addPermanentWidget(self._camera_label)
        self.statusBar().addPermanentWidget(self._fps_label)

    def _create_control_panel(self) -> QWidget:
        """Create comprehensive control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for different categories
        tabs = QTabWidget()

        # Tabs
        tabs.addTab(self._create_terrain_tab(), "Terrain")
        tabs.addTab(self._create_rendering_tab(), "Rendering")
        tabs.addTab(self._create_lighting_tab(), "Lighting")
        tabs.addTab(self._create_camera_tab(), "Camera")
        tabs.addTab(self._create_hdri_tab(), "HDRI Skybox")
        tabs.addTab(self._create_export_tab(), "Export")

        layout.addWidget(tabs)

        # Help text
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(150)
        help_text.setHtml("""
        <h4>Controls:</h4>
        <b>WASD</b> - Move camera<br>
        <b>Space/Shift</b> - Move up/down<br>
        <b>Mouse</b> - Look around (click to capture)<br>
        <b>R</b> - Reset camera<br>
        <b>C</b> - Toggle collision<br>
        <br>
        <i>Click in viewport to capture mouse for FPS controls</i>
        """)
        layout.addWidget(help_text)

        return panel

    def _create_terrain_tab(self) -> QWidget:
        """Create terrain generation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Preset selection
        preset_group = QGroupBox("Mountain Presets")
        preset_layout = QVBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Alps",
            "Himalayas",
            "Scottish Highlands",
            "Grand Canyon",
            "Rocky Mountains"
        ])
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)

        generate_btn = QPushButton("Generate Terrain")
        generate_btn.clicked.connect(self._on_generate_terrain)
        preset_layout.addWidget(generate_btn)

        layout.addWidget(preset_group)

        # Terrain parameters
        params_group = QGroupBox("Terrain Parameters")
        params_layout = QGridLayout(params_group)

        # Size
        params_layout.addWidget(QLabel("Size:"), 0, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(128, 2048)
        self.size_spin.setValue(512)
        self.size_spin.setSingleStep(128)
        params_layout.addWidget(self.size_spin, 0, 1)

        # Terrain scale
        params_layout.addWidget(QLabel("Terrain Scale:"), 1, 0)
        self.terrain_scale_spin = QDoubleSpinBox()
        self.terrain_scale_spin.setRange(10.0, 1000.0)
        self.terrain_scale_spin.setValue(100.0)
        self.terrain_scale_spin.setSingleStep(10.0)
        params_layout.addWidget(self.terrain_scale_spin, 1, 1)

        # Height scale
        params_layout.addWidget(QLabel("Height Scale:"), 2, 0)
        self.height_scale_spin = QDoubleSpinBox()
        self.height_scale_spin.setRange(1.0, 100.0)
        self.height_scale_spin.setValue(20.0)
        self.height_scale_spin.setSingleStep(5.0)
        params_layout.addWidget(self.height_scale_spin, 2, 1)

        # LOD
        params_layout.addWidget(QLabel("LOD:"), 3, 0)
        self.lod_combo = QComboBox()
        self.lod_combo.addItems(["1 (High)", "2 (Medium)", "4 (Low)"])
        self.lod_combo.setCurrentIndex(1)
        params_layout.addWidget(self.lod_combo, 3, 1)

        layout.addWidget(params_group)

        # Load heightmap
        load_group = QGroupBox("Load Heightmap")
        load_layout = QVBoxLayout(load_group)

        load_btn = QPushButton("Load from File...")
        load_btn.clicked.connect(self._on_load_heightmap)
        load_layout.addWidget(load_btn)

        layout.addWidget(load_group)

        layout.addStretch()

        return widget

    def _create_rendering_tab(self) -> QWidget:
        """Create rendering settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Shadows
        shadow_group = QGroupBox("Shadows")
        shadow_layout = QVBoxLayout(shadow_group)

        self.shadows_check = QCheckBox("Enable Shadows")
        self.shadows_check.setChecked(True)
        self.shadows_check.toggled.connect(self.viewer.set_shadows_enabled)
        shadow_layout.addWidget(self.shadows_check)

        shadow_layout.addWidget(QLabel("Shadow Quality:"))
        self.shadow_quality_combo = QComboBox()
        self.shadow_quality_combo.addItems(["Low (1024)", "Medium (2048)", "High (4096)"])
        self.shadow_quality_combo.setCurrentIndex(1)
        self.shadow_quality_combo.currentIndexChanged.connect(self._on_shadow_quality_changed)
        shadow_layout.addWidget(self.shadow_quality_combo)

        layout.addWidget(shadow_group)

        # Fog
        fog_group = QGroupBox("Atmospheric Fog")
        fog_layout = QVBoxLayout(fog_group)

        self.fog_check = QCheckBox("Enable Fog")
        self.fog_check.setChecked(True)
        self.fog_check.toggled.connect(self.viewer.set_fog_enabled)
        fog_layout.addWidget(self.fog_check)

        fog_layout.addWidget(QLabel("Fog Density:"))
        self.fog_density_slider = QSlider(Qt.Horizontal)
        self.fog_density_slider.setRange(0, 100)
        self.fog_density_slider.setValue(10)
        self.fog_density_slider.valueChanged.connect(self._on_fog_density_changed)
        fog_layout.addWidget(self.fog_density_slider)

        layout.addWidget(fog_group)

        # Display options
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.wireframe_check = QCheckBox("Wireframe Mode")
        self.wireframe_check.toggled.connect(self.viewer.set_wireframe)
        display_layout.addWidget(self.wireframe_check)

        layout.addWidget(display_group)

        layout.addStretch()

        return widget

    def _create_lighting_tab(self) -> QWidget:
        """Create lighting controls tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Sun position
        sun_group = QGroupBox("Sun Position")
        sun_layout = QGridLayout(sun_group)

        sun_layout.addWidget(QLabel("Azimuth (°):"), 0, 0)
        self.sun_azimuth_slider = QSlider(Qt.Horizontal)
        self.sun_azimuth_slider.setRange(0, 360)
        self.sun_azimuth_slider.setValue(135)
        self.sun_azimuth_slider.valueChanged.connect(self._on_sun_changed)
        sun_layout.addWidget(self.sun_azimuth_slider, 0, 1)
        self.sun_azimuth_label = QLabel("135°")
        sun_layout.addWidget(self.sun_azimuth_label, 0, 2)

        sun_layout.addWidget(QLabel("Elevation (°):"), 1, 0)
        self.sun_elevation_slider = QSlider(Qt.Horizontal)
        self.sun_elevation_slider.setRange(-90, 90)
        self.sun_elevation_slider.setValue(45)
        self.sun_elevation_slider.valueChanged.connect(self._on_sun_changed)
        sun_layout.addWidget(self.sun_elevation_slider, 1, 1)
        self.sun_elevation_label = QLabel("45°")
        sun_layout.addWidget(self.sun_elevation_label, 1, 2)

        layout.addWidget(sun_group)

        # Ambient
        ambient_group = QGroupBox("Ambient Light")
        ambient_layout = QVBoxLayout(ambient_group)

        ambient_layout.addWidget(QLabel("Strength:"))
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(30)
        self.ambient_slider.valueChanged.connect(self._on_ambient_changed)
        ambient_layout.addWidget(self.ambient_slider)

        layout.addWidget(ambient_group)

        # Shadow bias
        bias_group = QGroupBox("Shadow Settings")
        bias_layout = QVBoxLayout(bias_group)

        bias_layout.addWidget(QLabel("Shadow Bias:"))
        self.bias_slider = QSlider(Qt.Horizontal)
        self.bias_slider.setRange(1, 100)
        self.bias_slider.setValue(50)
        self.bias_slider.valueChanged.connect(self._on_bias_changed)
        bias_layout.addWidget(self.bias_slider)

        layout.addWidget(bias_group)

        layout.addStretch()

        return widget

    def _create_camera_tab(self) -> QWidget:
        """Create camera settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Camera mode (FPS only for now)
        mode_group = QGroupBox("Camera Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.addWidget(QLabel("FPS Camera (WASD + Mouse)"))
        layout.addWidget(mode_group)

        # Camera settings
        settings_group = QGroupBox("Movement Settings")
        settings_layout = QGridLayout(settings_group)

        settings_layout.addWidget(QLabel("Speed:"), 0, 0)
        self.camera_speed_slider = QSlider(Qt.Horizontal)
        self.camera_speed_slider.setRange(1, 100)
        self.camera_speed_slider.setValue(int(self.viewer._camera.speed))
        self.camera_speed_slider.valueChanged.connect(self._on_camera_speed_changed)
        settings_layout.addWidget(self.camera_speed_slider, 0, 1)

        settings_layout.addWidget(QLabel("Sensitivity:"), 1, 0)
        self.camera_sens_slider = QSlider(Qt.Horizontal)
        self.camera_sens_slider.setRange(1, 50)
        self.camera_sens_slider.setValue(int(self.viewer._camera.sensitivity * 100))
        self.camera_sens_slider.valueChanged.connect(self._on_camera_sens_changed)
        settings_layout.addWidget(self.camera_sens_slider, 1, 1)

        layout.addWidget(settings_group)

        # Collision
        collision_group = QGroupBox("Collision")
        collision_layout = QVBoxLayout(collision_group)

        self.collision_check = QCheckBox("Enable Terrain Collision")
        self.collision_check.setChecked(True)
        self.collision_check.toggled.connect(self._on_collision_toggled)
        collision_layout.addWidget(self.collision_check)

        layout.addWidget(collision_group)

        # Reset button
        reset_btn = QPushButton("Reset Camera (R)")
        reset_btn.clicked.connect(lambda: self.viewer._camera.reset())
        layout.addWidget(reset_btn)

        layout.addStretch()

        return widget

    def _create_hdri_tab(self) -> QWidget:
        """Create HDRI generation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Generation
        gen_group = QGroupBox("Generate HDRI Skybox")
        gen_layout = QVBoxLayout(gen_group)

        gen_layout.addWidget(QLabel("Time of Day:"))
        self.hdri_time_combo = QComboBox()
        self.hdri_time_combo.addItems([
            "Sunrise", "Morning", "Midday", "Afternoon",
            "Sunset", "Twilight", "Night"
        ])
        self.hdri_time_combo.setCurrentIndex(2)  # Midday
        gen_layout.addWidget(self.hdri_time_combo)

        gen_layout.addWidget(QLabel("Resolution:"))
        self.hdri_res_combo = QComboBox()
        self.hdri_res_combo.addItems([
            "2048x1024 (Low)",
            "4096x2048 (Medium)",
            "8192x4096 (High)"
        ])
        self.hdri_res_combo.setCurrentIndex(1)
        gen_layout.addWidget(self.hdri_res_combo)

        gen_layout.addWidget(QLabel("Cloud Density:"))
        self.cloud_density_slider = QSlider(Qt.Horizontal)
        self.cloud_density_slider.setRange(0, 100)
        self.cloud_density_slider.setValue(30)
        gen_layout.addWidget(self.cloud_density_slider)

        if AI_AVAILABLE:
            self.ai_enhance_check = QCheckBox("AI Enhancement (requires 10+ GB VRAM)")
            gen_layout.addWidget(self.ai_enhance_check)

        self.hdri_progress = QProgressBar()
        self.hdri_progress.setVisible(False)
        gen_layout.addWidget(self.hdri_progress)

        generate_hdri_btn = QPushButton("Generate HDRI")
        generate_hdri_btn.clicked.connect(self._on_generate_hdri)
        gen_layout.addWidget(generate_hdri_btn)

        layout.addWidget(gen_group)

        # Load
        load_group = QGroupBox("Load HDRI")
        load_layout = QVBoxLayout(load_group)

        load_hdri_btn = QPushButton("Load HDRI File...")
        load_hdri_btn.clicked.connect(self._on_load_hdri)
        load_layout.addWidget(load_hdri_btn)

        layout.addWidget(load_group)

        layout.addStretch()

        return widget

    def _create_export_tab(self) -> QWidget:
        """Create export tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Terrain export
        terrain_group = QGroupBox("Export Terrain")
        terrain_layout = QVBoxLayout(terrain_group)

        export_terrain_btn = QPushButton("Export for Flame...")
        export_terrain_btn.clicked.connect(self._on_export_terrain)
        terrain_layout.addWidget(export_terrain_btn)

        layout.addWidget(terrain_group)

        # Screenshot
        screenshot_group = QGroupBox("Screenshot")
        screenshot_layout = QVBoxLayout(screenshot_group)

        screenshot_btn = QPushButton("Capture Screenshot...")
        screenshot_btn.clicked.connect(self._on_screenshot)
        screenshot_layout.addWidget(screenshot_btn)

        layout.addWidget(screenshot_group)

        layout.addStretch()

        return widget

    # ========== Event Handlers ==========

    def _on_generate_terrain(self):
        """Generate terrain from preset."""
        size = self.size_spin.value()
        preset_name = self.preset_combo.currentText().lower().replace(" ", "_")

        self.statusBar().showMessage(f"Generating {preset_name} terrain...")

        try:
            if preset_name == "custom":
                # Simple spectral synthesis
                heightmap = spectral_synthesis(size, beta=2.0, seed=42)
            else:
                # Use preset
                if preset_name in MOUNTAIN_PRESETS:
                    from core.terrain.advanced_algorithms import combine_algorithms
                    heightmap = combine_algorithms(size, **MOUNTAIN_PRESETS[preset_name], seed=42)
                else:
                    heightmap = spectral_synthesis(size, beta=2.0, seed=42)

            self._current_heightmap = heightmap
            self._update_terrain()

            self.statusBar().showMessage(f"Terrain generated: {size}x{size}", 3000)

        except Exception as e:
            logger.error(f"Terrain generation failed: {e}")
            QMessageBox.critical(self, "Error", f"Terrain generation failed:\n{str(e)}")
            self.statusBar().showMessage("Generation failed", 3000)

    def _on_load_heightmap(self):
        """Load heightmap from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Heightmap", "",
            "Images (*.png *.jpg *.tif *.tiff *.exr);;All Files (*)"
        )

        if not filename:
            return

        try:
            from PIL import Image
            img = Image.open(filename).convert('L')
            heightmap = np.array(img, dtype=np.float32) / 255.0

            self._current_heightmap = heightmap
            self._update_terrain()

            self.statusBar().showMessage(f"Loaded: {Path(filename).name}", 3000)

        except Exception as e:
            logger.error(f"Failed to load heightmap: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load heightmap:\n{str(e)}")

    def _update_terrain(self):
        """Update viewer with current heightmap."""
        if self._current_heightmap is None:
            return

        terrain_scale = self.terrain_scale_spin.value()
        height_scale = self.height_scale_spin.value()
        lod = int(self.lod_combo.currentText().split()[0])

        self.viewer.set_terrain(
            self._current_heightmap,
            terrain_scale=terrain_scale,
            height_scale=height_scale,
            lod=lod
        )

        self.terrain_generated.emit(self._current_heightmap)

    def _on_shadow_quality_changed(self, index):
        """Handle shadow quality change."""
        qualities = [1024, 2048, 4096]
        self.viewer.set_shadow_quality(qualities[index])

    def _on_fog_density_changed(self, value):
        """Handle fog density change."""
        self.viewer._fog_density = value / 100000.0  # Scale to reasonable range

    def _on_sun_changed(self):
        """Handle sun position change."""
        azimuth = self.sun_azimuth_slider.value()
        elevation = self.sun_elevation_slider.value()

        self.sun_azimuth_label.setText(f"{azimuth}°")
        self.sun_elevation_label.setText(f"{elevation}°")

        # Convert to direction vector
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        x = np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = np.sin(elevation_rad)
        z = np.cos(elevation_rad) * np.sin(azimuth_rad)

        direction = np.array([x, y, z], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        self.viewer._light_dir = -direction  # Light direction is opposite to sun position

    def _on_ambient_changed(self, value):
        """Handle ambient strength change."""
        self.viewer._ambient_strength = value / 100.0

    def _on_bias_changed(self, value):
        """Handle shadow bias change."""
        self.viewer._shadow_bias = value / 10000.0

    def _on_camera_speed_changed(self, value):
        """Handle camera speed change."""
        self.viewer._camera.speed = float(value)

    def _on_camera_sens_changed(self, value):
        """Handle camera sensitivity change."""
        self.viewer._camera.sensitivity = value / 100.0

    def _on_collision_toggled(self, checked):
        """Handle collision toggle."""
        self.viewer._camera.collision_enabled = checked

    def _on_generate_hdri(self):
        """Generate HDRI panorama."""
        self.statusBar().showMessage("Generating HDRI...")
        self.hdri_progress.setVisible(True)
        self.hdri_progress.setRange(0, 0)  # Indeterminate

        try:
            # Get settings
            time_str = self.hdri_time_combo.currentText().lower()
            time_of_day = TimeOfDay(time_str)

            res_map = {0: (2048, 1024), 1: (4096, 2048), 2: (8192, 4096)}
            resolution = res_map[self.hdri_res_combo.currentIndex()]

            cloud_density = self.cloud_density_slider.value() / 100.0

            # Create generator if needed
            if self._hdri_generator is None or self._hdri_generator.resolution != resolution:
                self._hdri_generator = HDRIPanoramicGenerator(resolution)

            # Generate
            hdri = self._hdri_generator.generate_procedural(
                time_of_day=time_of_day,
                cloud_density=cloud_density
            )

            # Save preview
            output_dir = Path.home() / "mountain_studio_hdri"
            output_dir.mkdir(exist_ok=True)

            self._hdri_generator.export_ldr(
                hdri,
                str(output_dir / f"hdri_{time_str}_preview.png")
            )

            self._hdri_generator.export_exr(
                hdri,
                str(output_dir / f"hdri_{time_str}.exr")
            )

            self.statusBar().showMessage(f"HDRI saved to {output_dir}", 5000)
            QMessageBox.information(
                self, "Success",
                f"HDRI generated and saved to:\n{output_dir}"
            )

        except Exception as e:
            logger.error(f"HDRI generation failed: {e}")
            QMessageBox.critical(self, "Error", f"HDRI generation failed:\n{str(e)}")
            self.statusBar().showMessage("HDRI generation failed", 3000)

        finally:
            self.hdri_progress.setVisible(False)

    def _on_load_hdri(self):
        """Load HDRI from file."""
        QMessageBox.information(
            self, "Not Implemented",
            "HDRI loading will be implemented in future version.\n"
            "For now, you can generate HDRIs using the Generate button."
        )

    def _on_export_terrain(self):
        """Export terrain for Flame."""
        if self._current_heightmap is None:
            QMessageBox.warning(self, "Warning", "No terrain to export")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory",
            str(Path.home())
        )

        if not output_dir:
            return

        try:
            from core.export.professional_exporter import ProfessionalExporter

            exporter = ProfessionalExporter(output_dir)
            exporter.export_for_flame(self._current_heightmap)

            self.statusBar().showMessage(f"Exported to {output_dir}", 3000)
            QMessageBox.information(self, "Success", f"Terrain exported to:\n{output_dir}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _on_screenshot(self):
        """Capture screenshot."""
        QMessageBox.information(
            self, "Not Implemented",
            "Screenshot capture will be implemented in future version."
        )

    def _update_fps_display(self):
        """Update FPS and camera position in status bar."""
        fps = self.viewer.get_fps()
        self._fps_label.setText(f"FPS: {fps}")

        pos = self.viewer._camera.position
        self._camera_label.setText(f"Camera: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

    def closeEvent(self, event):
        """Handle window close."""
        self.viewer.cleanup()
        event.accept()


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = UltimateTerrainViewer()
    viewer.show()

    # Generate default terrain
    default_terrain = spectral_synthesis(512, beta=2.0, seed=42)
    viewer._current_heightmap = default_terrain
    viewer._update_terrain()

    sys.exit(app.exec())
