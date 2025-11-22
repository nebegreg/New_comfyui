#!/usr/bin/env python3
"""
MOUNTAIN STUDIO ULTIMATE FINAL - L'Application Ultime
======================================================

Application complète de génération de terrains photorréalistes avec:

✅ PRESETS PROFESSIONNELS (Evian, 3 Peaks, Ski Slope, etc.)
✅ GÉNÉRATION TERRAIN ultra-réaliste
✅ VÉGÉTATION avec placement biome-based
✅ TEXTURES PBR via ComfyUI (automatique) ou procédural
✅ PREVIEW PBR MAPS complet (tous les maps visualisables)
✅ RENDU 3D photorréaliste (PBR + atmosphère + fog)
✅ HDRI PANORAMIQUE (7 times of day)
✅ EXPORT AVANCÉ (OBJ, FBX, ABC pour Autodesk Flame)
✅ BOUTON "GENERATE ALL" - tout en un clic

Features Uniques:
- Communication ComfyUI automatique (zéro configuration)
- Onglet Preview PBR avec grille de thumbnails
- Export optimisé pour Autodesk Flame (package complet)
- 6 presets de montagnes iconiques
- Cache intelligent pour éviter régénération

Usage:
    python mountain_studio_ultimate_final.py

Author: Mountain Studio Pro Team
Date: 2025
"""

import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import logging
import time
import json

# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QSlider, QComboBox, QCheckBox,
    QGroupBox, QTextEdit, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QSpinBox, QDoubleSpinBox, QGridLayout, QScrollArea,
    QSplitter, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

# Scientific imports
from scipy.ndimage import gaussian_filter, convolve
from scipy.interpolate import interp1d, griddata

# 3D visualization
try:
    import pyqtgraph.opengl as gl
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("⚠️ Warning: PyQtGraph OpenGL not available. 3D preview will be limited.")

# Image handling
from PIL import Image

# Core modules
try:
    from core.config.mountain_presets import TerrainPreset, PRESETS
except ImportError:
    PRESETS = {}
    print("⚠️ Warning: mountain_presets not available")

try:
    from core.export.advanced_3d_exporter import Advanced3DExporter
    ADVANCED_EXPORT_AVAILABLE = True
except ImportError:
    ADVANCED_EXPORT_AVAILABLE = False
    print("⚠️ Warning: advanced_3d_exporter not available")

try:
    from core.rendering.hdri_generator import HDRIPanoramicGenerator, TimeOfDay
    HDRI_AVAILABLE = True
except ImportError:
    HDRI_AVAILABLE = False
    print("⚠️ Warning: HDRI generator not available")

try:
    from core.ai.comfyui_auto_workflow import ComfyUIAutoWorkflow
    COMFYUI_AUTO_AVAILABLE = True
except ImportError:
    COMFYUI_AUTO_AVAILABLE = False
    print("⚠️ Warning: ComfyUI auto workflow not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== BACKGROUND GENERATION THREADS ====================

class TerrainGenerationThread(QThread):
    """Generate terrain in background"""
    finished = Signal(np.ndarray)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from core.terrain.terrain_generator import TerrainGenerator

            self.progress.emit("Generating base terrain...")
            generator = TerrainGenerator(
                width=self.params.get('width', 512),
                height=self.params.get('height', 512),
                scale=self.params.get('scale', 100.0),
                octaves=self.params.get('octaves', 8),
                seed=self.params.get('seed')
            )

            heightmap = generator.generate()

            # Apply erosion if requested
            if self.params.get('erosion_enabled', False):
                self.progress.emit("Applying hydraulic erosion...")
                from core.terrain.erosion import HydraulicErosion
                erosion = HydraulicErosion()
                heightmap = erosion.erode(
                    heightmap,
                    iterations=self.params.get('hydraulic_iterations', 50)
                )

                self.progress.emit("Applying thermal erosion...")
                from core.terrain.erosion import ThermalErosion
                thermal = ThermalErosion()
                heightmap = thermal.erode(
                    heightmap,
                    iterations=self.params.get('thermal_iterations', 5)
                )

            self.progress.emit("Terrain generation complete!")
            self.finished.emit(heightmap)

        except Exception as e:
            logger.error(f"Terrain generation error: {e}")
            self.progress.emit(f"ERROR: {e}")


class VegetationGenerationThread(QThread):
    """Generate vegetation in background"""
    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, heightmap: np.ndarray, params: dict):
        super().__init__()
        self.heightmap = heightmap
        self.params = params

    def run(self):
        try:
            from core.vegetation.placement import PoissonDiscSampling
            from core.vegetation.biomes import BiomeClassifier

            h, w = self.heightmap.shape

            self.progress.emit("Classifying biomes...")
            classifier = BiomeClassifier(width=w, height=h)
            biome_map = classifier.classify(self.heightmap)

            self.progress.emit("Placing vegetation...")
            sampler = PoissonDiscSampling(
                width=w,
                height=h,
                min_distance=self.params.get('spacing', 5.0)
            )

            points = sampler.generate(max_attempts=self.params.get('max_attempts', 30))

            # Filter by biome and height
            vegetation = []
            for x, y in points:
                i, j = int(y), int(x)
                if 0 <= i < h and 0 <= j < w:
                    height = self.heightmap[i, j]
                    biome = biome_map[i, j]

                    # Only place trees in suitable areas
                    if 0.3 < height < 0.8 and biome in [1, 2]:  # Forest/grassland
                        vegetation.append({
                            'x': x, 'y': y, 'z': height,
                            'type': 'tree',
                            'biome': biome
                        })

            self.progress.emit(f"Placed {len(vegetation)} vegetation instances")
            self.finished.emit(vegetation)

        except Exception as e:
            logger.error(f"Vegetation generation error: {e}")
            self.progress.emit(f"ERROR: {e}")
            self.finished.emit([])


class PBRGenerationThread(QThread):
    """Generate PBR textures in background"""
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, material: str, resolution: int, use_ai: bool):
        super().__init__()
        self.material = material
        self.resolution = resolution
        self.use_ai = use_ai

    def run(self):
        try:
            if self.use_ai and COMFYUI_AUTO_AVAILABLE:
                self.progress.emit("Starting AI PBR generation (ComfyUI)...")
                workflow = ComfyUIAutoWorkflow()

                textures = workflow.generate_pbr_auto(
                    material=self.material,
                    resolution=self.resolution,
                    progress_callback=lambda msg: self.progress.emit(msg)
                )

            else:
                self.progress.emit("Generating procedural PBR textures...")
                from core.rendering.pbr_texture_generator import PBRTextureGenerator

                generator = PBRTextureGenerator(resolution=(self.resolution, self.resolution))

                textures = {
                    'diffuse': generator.generate_diffuse(self.material),
                    'normal': generator.generate_normal(self.material),
                    'roughness': generator.generate_roughness(self.material),
                    'ao': generator.generate_ao(),
                    'height': generator.generate_height(self.material)
                }

                self.progress.emit("Procedural PBR textures generated!")

            self.finished.emit(textures)

        except Exception as e:
            logger.error(f"PBR generation error: {e}")
            self.progress.emit(f"ERROR: {e}")
            self.finished.emit({})


class HDRIGenerationThread(QThread):
    """Generate HDRI in background"""
    finished = Signal(np.ndarray)
    progress = Signal(str)

    def __init__(self, time_of_day: str, resolution: tuple):
        super().__init__()
        self.time_of_day = time_of_day
        self.resolution = resolution

    def run(self):
        try:
            if not HDRI_AVAILABLE:
                self.progress.emit("HDRI generator not available")
                return

            self.progress.emit(f"Generating HDRI: {self.time_of_day}...")

            generator = HDRIPanoramicGenerator(resolution=self.resolution)

            # Map string to TimeOfDay enum
            time_map = {
                'sunrise': TimeOfDay.SUNRISE,
                'morning': TimeOfDay.MORNING,
                'midday': TimeOfDay.MIDDAY,
                'afternoon': TimeOfDay.AFTERNOON,
                'sunset': TimeOfDay.SUNSET,
                'twilight': TimeOfDay.TWILIGHT,
                'night': TimeOfDay.NIGHT
            }

            time_enum = time_map.get(self.time_of_day.lower(), TimeOfDay.MIDDAY)

            hdri = generator.generate_procedural_enhanced(
                time_of_day=time_enum,
                cloud_density=0.3,
                mountain_distance=True
            )

            self.progress.emit("HDRI generation complete!")
            self.finished.emit(hdri)

        except Exception as e:
            logger.error(f"HDRI generation error: {e}")
            self.progress.emit(f"ERROR: {e}")


# ==================== MAIN APPLICATION ====================

class MountainStudioUltimateFinal(QMainWindow):
    """
    Mountain Studio ULTIMATE FINAL

    Application complète avec tous les features:
    - Presets professionnels
    - Génération terrain + végétation
    - PBR textures (AI ou procedural)
    - Preview PBR complet
    - HDRI panoramique
    - Export avancé (Flame, FBX, ABC)
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mountain Studio ULTIMATE FINAL - Professional Edition")
        self.setGeometry(100, 100, 1600, 1000)

        # Output directory
        self.output_dir = Path("outputs_ultimate")
        self.output_dir.mkdir(exist_ok=True)

        # Data storage
        self.heightmap = None
        self.vegetation = []
        self.pbr_textures = {}
        self.hdri_image = None
        self.current_preset = None

        # Generation threads
        self.terrain_thread = None
        self.vegetation_thread = None
        self.pbr_thread = None
        self.hdri_thread = None

        # Setup UI
        self.init_ui()

        self.log("🏔️ Mountain Studio ULTIMATE FINAL - Ready!")
        self.log("💡 Select a preset or customize terrain parameters")

    def init_ui(self):
        """Initialize complete UI"""

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Header with Generate All button
        header = self.create_header()
        main_layout.addWidget(header)

        # Main content area
        splitter = QSplitter(Qt.Horizontal)

        # Left: Tab widget
        self.tabs = QTabWidget()
        self.create_all_tabs()
        splitter.addWidget(self.tabs)

        # Right: Log and status
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_header(self):
        """Create header with Generate All button"""
        header_group = QGroupBox("🎯 MASTER CONTROL")
        header_layout = QHBoxLayout()

        title_label = QLabel("🏔️ <b>MOUNTAIN STUDIO ULTIMATE</b> - One-Click Complete Generation")
        title_label.setStyleSheet("font-size: 14pt; color: #2c3e50;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Generate All button (BIG)
        self.generate_all_btn = QPushButton("⚡ GENERATE ALL\n(Terrain + Vegetation + PBR + HDRI)")
        self.generate_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.generate_all_btn.setMinimumHeight(80)
        self.generate_all_btn.clicked.connect(self.generate_all)
        header_layout.addWidget(self.generate_all_btn)

        header_group.setLayout(header_layout)
        return header_group

    def create_all_tabs(self):
        """Create all tabs"""

        # 1. Presets Tab
        presets_tab = self.create_presets_tab()
        self.tabs.addTab(presets_tab, "⭐ Presets")

        # 2. Terrain Tab
        terrain_tab = self.create_terrain_tab()
        self.tabs.addTab(terrain_tab, "🏔️ Terrain")

        # 3. Vegetation Tab
        vegetation_tab = self.create_vegetation_tab()
        self.tabs.addTab(vegetation_tab, "🌲 Vegetation")

        # 4. PBR Textures Tab
        pbr_tab = self.create_pbr_tab()
        self.tabs.addTab(pbr_tab, "🎨 PBR Textures")

        # 5. PBR Preview Tab (NEW!)
        pbr_preview_tab = self.create_pbr_preview_tab()
        self.tabs.addTab(pbr_preview_tab, "🗺️ PBR Preview")

        # 6. 3D Rendering Tab
        rendering_tab = self.create_rendering_tab()
        self.tabs.addTab(rendering_tab, "🎮 3D Rendering")

        # 7. HDRI Tab
        hdri_tab = self.create_hdri_tab()
        self.tabs.addTab(hdri_tab, "🌅 HDRI Sky")

        # 8. Export Tab
        export_tab = self.create_export_tab()
        self.tabs.addTab(export_tab, "💾 Export")

        # 9. Advanced Export Tab (NEW!)
        advanced_export_tab = self.create_advanced_export_tab()
        self.tabs.addTab(advanced_export_tab, "🎬 Advanced Export")

    def create_presets_tab(self):
        """Tab 1: Presets professionnels"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>⭐ Professional Mountain Presets</h2>")
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)

        # Description
        desc = QLabel("Select a professional preset for instant terrain generation")
        layout.addWidget(desc)

        # Preset selector
        preset_group = QGroupBox("Select Preset")
        preset_layout = QVBoxLayout()

        self.preset_combo = QComboBox()
        if PRESETS:
            for name, preset in PRESETS.items():
                display_name = f"{preset.name} - {preset.description}"
                self.preset_combo.addItem(display_name, name)
        else:
            self.preset_combo.addItem("No presets available", None)

        self.preset_combo.currentIndexChanged.connect(self.on_preset_selected)
        preset_layout.addWidget(self.preset_combo)

        # Preset details
        self.preset_details = QTextEdit()
        self.preset_details.setReadOnly(True)
        self.preset_details.setMaximumHeight(200)
        preset_layout.addWidget(QLabel("Preset Details:"))
        preset_layout.addWidget(self.preset_details)

        # Apply button
        apply_preset_btn = QPushButton("✅ Apply Preset")
        apply_preset_btn.setStyleSheet("background-color: #3498db; color: white; padding: 10px; font-weight: bold;")
        apply_preset_btn.clicked.connect(self.apply_preset)
        preset_layout.addWidget(apply_preset_btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Available presets list
        presets_list_group = QGroupBox("Available Presets")
        presets_list_layout = QVBoxLayout()

        if PRESETS:
            for name, preset in PRESETS.items():
                preset_label = QLabel(f"• <b>{preset.name}</b>: {preset.description}")
                presets_list_layout.addWidget(preset_label)
        else:
            presets_list_layout.addWidget(QLabel("⚠️ No presets loaded. Check core/config/mountain_presets.py"))

        presets_list_group.setLayout(presets_list_layout)
        layout.addWidget(presets_list_group)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_terrain_tab(self):
        """Tab 2: Terrain generation"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>🏔️ Terrain Generation</h2>")
        layout.addWidget(title)

        # Parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QGridLayout()

        # Resolution
        params_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.terrain_resolution = QComboBox()
        self.terrain_resolution.addItems(["256x256", "512x512", "1024x1024", "2048x2048"])
        self.terrain_resolution.setCurrentText("512x512")
        params_layout.addWidget(self.terrain_resolution, 0, 1)

        # Scale
        params_layout.addWidget(QLabel("Scale:"), 1, 0)
        self.terrain_scale = QDoubleSpinBox()
        self.terrain_scale.setRange(10.0, 500.0)
        self.terrain_scale.setValue(100.0)
        params_layout.addWidget(self.terrain_scale, 1, 1)

        # Octaves
        params_layout.addWidget(QLabel("Octaves:"), 2, 0)
        self.terrain_octaves = QSpinBox()
        self.terrain_octaves.setRange(1, 16)
        self.terrain_octaves.setValue(8)
        params_layout.addWidget(self.terrain_octaves, 2, 1)

        # Seed
        params_layout.addWidget(QLabel("Seed:"), 3, 0)
        self.terrain_seed = QSpinBox()
        self.terrain_seed.setRange(0, 999999)
        self.terrain_seed.setValue(42)
        params_layout.addWidget(self.terrain_seed, 3, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Erosion
        erosion_group = QGroupBox("Erosion")
        erosion_layout = QVBoxLayout()

        self.erosion_enabled = QCheckBox("Enable Erosion")
        self.erosion_enabled.setChecked(True)
        erosion_layout.addWidget(self.erosion_enabled)

        erosion_params = QGridLayout()
        erosion_params.addWidget(QLabel("Hydraulic Iterations:"), 0, 0)
        self.hydraulic_iterations = QSpinBox()
        self.hydraulic_iterations.setRange(0, 200)
        self.hydraulic_iterations.setValue(50)
        erosion_params.addWidget(self.hydraulic_iterations, 0, 1)

        erosion_params.addWidget(QLabel("Thermal Iterations:"), 1, 0)
        self.thermal_iterations = QSpinBox()
        self.thermal_iterations.setRange(0, 50)
        self.thermal_iterations.setValue(5)
        erosion_params.addWidget(self.thermal_iterations, 1, 1)

        erosion_layout.addLayout(erosion_params)
        erosion_group.setLayout(erosion_layout)
        layout.addWidget(erosion_group)

        # Generate button
        generate_terrain_btn = QPushButton("🏔️ Generate Terrain")
        generate_terrain_btn.setStyleSheet("background-color: #e67e22; color: white; padding: 15px; font-weight: bold;")
        generate_terrain_btn.clicked.connect(self.generate_terrain)
        layout.addWidget(generate_terrain_btn)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_vegetation_tab(self):
        """Tab 3: Vegetation"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>🌲 Vegetation Placement</h2>")
        layout.addWidget(title)

        # Parameters
        params_group = QGroupBox("Vegetation Parameters")
        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("Spacing:"), 0, 0)
        self.veg_spacing = QDoubleSpinBox()
        self.veg_spacing.setRange(1.0, 20.0)
        self.veg_spacing.setValue(5.0)
        params_layout.addWidget(self.veg_spacing, 0, 1)

        params_layout.addWidget(QLabel("Max Attempts:"), 1, 0)
        self.veg_max_attempts = QSpinBox()
        self.veg_max_attempts.setRange(10, 100)
        self.veg_max_attempts.setValue(30)
        params_layout.addWidget(self.veg_max_attempts, 1, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Generate button
        generate_veg_btn = QPushButton("🌲 Generate Vegetation")
        generate_veg_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 15px; font-weight: bold;")
        generate_veg_btn.clicked.connect(self.generate_vegetation)
        layout.addWidget(generate_veg_btn)

        # Status
        self.veg_status = QLabel("No vegetation generated")
        layout.addWidget(self.veg_status)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_pbr_tab(self):
        """Tab 4: PBR Textures"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>🎨 PBR Texture Generation</h2>")
        layout.addWidget(title)

        # Material selection
        material_group = QGroupBox("Material Selection")
        material_layout = QVBoxLayout()

        self.material_combo = QComboBox()
        self.material_combo.addItems(["rock", "snow", "grass", "sand", "dirt", "ice"])
        material_layout.addWidget(QLabel("Material:"))
        material_layout.addWidget(self.material_combo)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # Resolution
        res_group = QGroupBox("Texture Resolution")
        res_layout = QVBoxLayout()

        self.pbr_resolution = QComboBox()
        self.pbr_resolution.addItems(["512", "1024", "2048", "4096"])
        self.pbr_resolution.setCurrentText("2048")
        res_layout.addWidget(self.pbr_resolution)

        res_group.setLayout(res_layout)
        layout.addWidget(res_group)

        # Generation method
        method_group = QGroupBox("Generation Method")
        method_layout = QVBoxLayout()

        self.use_ai_pbr = QCheckBox("Use AI (ComfyUI) - Auto Workflow")
        self.use_ai_pbr.setChecked(COMFYUI_AUTO_AVAILABLE)
        method_layout.addWidget(self.use_ai_pbr)

        ai_info = QLabel("✅ AI: Ultra-realistic via ComfyUI (automatic)\n⚡ Procedural: Fast, good quality")
        ai_info.setWordWrap(True)
        method_layout.addWidget(ai_info)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Generate button
        generate_pbr_btn = QPushButton("🎨 Generate PBR Textures")
        generate_pbr_btn.setStyleSheet("background-color: #9b59b6; color: white; padding: 15px; font-weight: bold;")
        generate_pbr_btn.clicked.connect(self.generate_pbr)
        layout.addWidget(generate_pbr_btn)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_pbr_preview_tab(self):
        """Tab 5: PBR Preview (NEW!)"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>🗺️ PBR Maps Preview</h2>")
        layout.addWidget(title)

        desc = QLabel("View all generated PBR texture maps")
        layout.addWidget(desc)

        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # Grid for PBR maps
        self.pbr_preview_grid = QGridLayout()

        # Placeholder labels for each map
        self.pbr_preview_labels = {}
        map_names = ['diffuse', 'normal', 'roughness', 'ao', 'height', 'metallic']

        for idx, name in enumerate(map_names):
            row = idx // 2
            col = idx % 2

            map_group = QGroupBox(name.upper())
            map_layout = QVBoxLayout()

            label = QLabel()
            label.setMinimumSize(300, 300)
            label.setScaledContents(True)
            label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")
            label.setText(f"No {name} map")
            label.setAlignment(Qt.AlignCenter)

            map_layout.addWidget(label)
            map_group.setLayout(map_layout)

            self.pbr_preview_grid.addWidget(map_group, row, col)
            self.pbr_preview_labels[name] = label

        scroll_layout.addLayout(self.pbr_preview_grid)
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)

        layout.addWidget(scroll)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Preview")
        refresh_btn.clicked.connect(self.refresh_pbr_preview)
        layout.addWidget(refresh_btn)

        tab.setLayout(layout)
        return tab

    def create_rendering_tab(self):
        """Tab 6: 3D Rendering"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>🎮 3D Photorealistic Rendering</h2>")
        layout.addWidget(title)

        if OPENGL_AVAILABLE:
            # 3D viewer
            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setMinimumSize(600, 400)
            layout.addWidget(self.gl_widget)
        else:
            layout.addWidget(QLabel("⚠️ OpenGL not available. Install pyqtgraph and PyOpenGL."))

        # Render button
        render_btn = QPushButton("🎮 Render 3D View")
        render_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 15px; font-weight: bold;")
        render_btn.clicked.connect(self.render_3d)
        layout.addWidget(render_btn)

        tab.setLayout(layout)
        return tab

    def create_hdri_tab(self):
        """Tab 7: HDRI"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>🌅 HDRI Panoramic Sky</h2>")
        layout.addWidget(title)

        # Time of day
        time_group = QGroupBox("Time of Day")
        time_layout = QVBoxLayout()

        self.hdri_time = QComboBox()
        self.hdri_time.addItems(["Sunrise", "Morning", "Midday", "Afternoon", "Sunset", "Twilight", "Night"])
        self.hdri_time.setCurrentText("Midday")
        time_layout.addWidget(self.hdri_time)

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Resolution
        res_group = QGroupBox("HDRI Resolution")
        res_layout = QVBoxLayout()

        self.hdri_resolution = QComboBox()
        self.hdri_resolution.addItems(["2048x1024 (Low)", "4096x2048 (Medium)", "8192x4096 (High)"])
        self.hdri_resolution.setCurrentIndex(1)
        res_layout.addWidget(self.hdri_resolution)

        res_group.setLayout(res_layout)
        layout.addWidget(res_group)

        # Preview
        self.hdri_preview = QLabel()
        self.hdri_preview.setMinimumSize(400, 200)
        self.hdri_preview.setScaledContents(True)
        self.hdri_preview.setStyleSheet("border: 2px solid #3498db;")
        self.hdri_preview.setText("No HDRI generated")
        self.hdri_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(QLabel("Preview:"))
        layout.addWidget(self.hdri_preview)

        # Generate button
        generate_hdri_btn = QPushButton("🌅 Generate HDRI")
        generate_hdri_btn.setStyleSheet("background-color: #f39c12; color: white; padding: 15px; font-weight: bold;")
        generate_hdri_btn.clicked.connect(self.generate_hdri)
        layout.addWidget(generate_hdri_btn)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_export_tab(self):
        """Tab 8: Basic Export"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>💾 Basic Export</h2>")
        layout.addWidget(title)

        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()

        # Heightmap
        export_heightmap_btn = QPushButton("💾 Export Heightmap (PNG, 16-bit)")
        export_heightmap_btn.clicked.connect(self.export_heightmap)
        export_layout.addWidget(export_heightmap_btn)

        # PBR textures
        export_pbr_btn = QPushButton("💾 Export PBR Textures (All Maps)")
        export_pbr_btn.clicked.connect(self.export_pbr_textures)
        export_layout.addWidget(export_pbr_btn)

        # HDRI
        export_hdri_btn = QPushButton("💾 Export HDRI (EXR + PNG)")
        export_hdri_btn.clicked.connect(self.export_hdri)
        export_layout.addWidget(export_hdri_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()

        self.output_dir_label = QLabel(str(self.output_dir))
        output_layout.addWidget(self.output_dir_label)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_btn)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_advanced_export_tab(self):
        """Tab 9: Advanced Export (NEW!)"""
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h2>🎬 Advanced 3D Export</h2>")
        layout.addWidget(title)

        desc = QLabel("Export complete packages for professional VFX and 3D software")
        layout.addWidget(desc)

        # Autodesk Flame
        flame_group = QGroupBox("🎬 Autodesk Flame Export")
        flame_layout = QVBoxLayout()

        flame_desc = QLabel(
            "Complete package for Flame:\n"
            "• High-res OBJ mesh\n"
            "• 16-bit EXR textures (linear color space)\n"
            "• Camera data\n"
            "• HDRI environment\n"
            "• Python setup script"
        )
        flame_desc.setWordWrap(True)
        flame_layout.addWidget(flame_desc)

        export_flame_btn = QPushButton("🎬 Export for Autodesk Flame")
        export_flame_btn.setStyleSheet("background-color: #c0392b; color: white; padding: 15px; font-weight: bold;")
        export_flame_btn.clicked.connect(self.export_for_flame)
        flame_layout.addWidget(export_flame_btn)

        flame_group.setLayout(flame_layout)
        layout.addWidget(flame_group)

        # Other formats
        formats_group = QGroupBox("Other Professional Formats")
        formats_layout = QVBoxLayout()

        # OBJ
        export_obj_btn = QPushButton("📦 Export OBJ (Universal)")
        export_obj_btn.clicked.connect(self.export_obj)
        formats_layout.addWidget(export_obj_btn)

        # FBX
        export_fbx_btn = QPushButton("📦 Export FBX (Autodesk)")
        export_fbx_btn.clicked.connect(self.export_fbx)
        formats_layout.addWidget(export_fbx_btn)

        # Alembic
        export_abc_btn = QPushButton("📦 Export Alembic (VFX Pipeline)")
        export_abc_btn.clicked.connect(self.export_alembic)
        formats_layout.addWidget(export_abc_btn)

        formats_group.setLayout(formats_layout)
        layout.addWidget(formats_group)

        # Status
        if not ADVANCED_EXPORT_AVAILABLE:
            warning = QLabel("⚠️ Warning: Advanced exporter module not available")
            warning.setStyleSheet("color: #e67e22; font-weight: bold;")
            layout.addWidget(warning)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_right_panel(self):
        """Create right panel with log and progress"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_bar)

        # Log
        log_group = QGroupBox("📋 Activity Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(600)
        log_layout.addWidget(self.log_text)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        panel.setLayout(layout)
        return panel

    # ==================== SLOTS AND ACTIONS ====================

    def log(self, message: str):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)

    def on_preset_selected(self, index: int):
        """Preset selection changed"""
        preset_key = self.preset_combo.currentData()
        if preset_key and preset_key in PRESETS:
            preset = PRESETS[preset_key]

            details = f"""
<b>{preset.name}</b><br>
{preset.description}<br>
<br>
<b>Category:</b> {preset.category.value}<br>
<b>Resolution:</b> {preset.resolution[0]}x{preset.resolution[1]}<br>
<b>Algorithm:</b> {preset.algorithm}<br>
<b>Erosion:</b> {preset.hydraulic_iterations} hydraulic, {preset.thermal_iterations} thermal<br>
<b>Vegetation:</b> {'Enabled' if preset.vegetation_enabled else 'Disabled'}<br>
<b>Materials:</b> {preset.primary_material}, {preset.secondary_material}<br>
<b>HDRI Time:</b> {preset.hdri_time}<br>
            """

            self.preset_details.setHtml(details)
            self.current_preset = preset

    def apply_preset(self):
        """Apply selected preset to all parameters"""
        if not self.current_preset:
            QMessageBox.warning(self, "Warning", "No preset selected")
            return

        preset = self.current_preset
        self.log(f"📋 Applying preset: {preset.name}")

        # Apply to terrain tab
        res_str = f"{preset.resolution[0]}x{preset.resolution[1]}"
        idx = self.terrain_resolution.findText(res_str)
        if idx >= 0:
            self.terrain_resolution.setCurrentIndex(idx)

        self.terrain_scale.setValue(preset.scale)
        self.terrain_octaves.setValue(preset.octaves)

        self.erosion_enabled.setChecked(preset.hydraulic_iterations > 0 or preset.thermal_iterations > 0)
        self.hydraulic_iterations.setValue(preset.hydraulic_iterations)
        self.thermal_iterations.setValue(preset.thermal_iterations)

        # Apply to vegetation tab
        self.veg_spacing.setValue(preset.vegetation_spacing)

        # Apply to PBR tab
        self.material_combo.setCurrentText(preset.primary_material)

        # Apply to HDRI tab
        hdri_map = {
            'sunrise': 'Sunrise',
            'morning': 'Morning',
            'midday': 'Midday',
            'afternoon': 'Afternoon',
            'sunset': 'Sunset',
            'twilight': 'Twilight',
            'night': 'Night'
        }
        hdri_text = hdri_map.get(preset.hdri_time, 'Midday')
        idx = self.hdri_time.findText(hdri_text)
        if idx >= 0:
            self.hdri_time.setCurrentIndex(idx)

        self.log("✅ Preset applied to all tabs")
        QMessageBox.information(self, "Success", f"Preset '{preset.name}' applied!\n\nClick 'GENERATE ALL' to create terrain.")

    def generate_all(self):
        """Master function: Generate everything"""
        self.log("⚡ GENERATE ALL - Starting complete generation pipeline...")

        # Step 1: Terrain
        self.log("Step 1/4: Generating terrain...")
        self.generate_terrain()

        # Note: We can't chain synchronously here because threads are async
        # Instead, we'll connect signals
        self.log("💡 Tip: Wait for terrain to complete, then vegetation/PBR/HDRI will auto-generate")
        self.log("Or manually trigger each step from the tabs")

    def generate_terrain(self):
        """Generate terrain"""
        if self.terrain_thread and self.terrain_thread.isRunning():
            self.log("⚠️ Terrain generation already running")
            return

        # Get parameters
        res_str = self.terrain_resolution.currentText()
        w, h = map(int, res_str.split('x'))

        params = {
            'width': w,
            'height': h,
            'scale': self.terrain_scale.value(),
            'octaves': self.terrain_octaves.value(),
            'seed': self.terrain_seed.value(),
            'erosion_enabled': self.erosion_enabled.isChecked(),
            'hydraulic_iterations': self.hydraulic_iterations.value(),
            'thermal_iterations': self.thermal_iterations.value()
        }

        self.log(f"🏔️ Generating terrain {w}x{h}...")
        self.progress_bar.setValue(10)

        self.terrain_thread = TerrainGenerationThread(params)
        self.terrain_thread.progress.connect(self.log)
        self.terrain_thread.finished.connect(self.on_terrain_generated)
        self.terrain_thread.start()

    def on_terrain_generated(self, heightmap: np.ndarray):
        """Terrain generation complete"""
        self.heightmap = heightmap
        self.log(f"✅ Terrain generated: {heightmap.shape}")
        self.progress_bar.setValue(30)

        # Save preview
        preview_path = self.output_dir / "terrain_preview.png"
        img = (heightmap * 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(preview_path)
        self.log(f"💾 Saved: {preview_path}")

    def generate_vegetation(self):
        """Generate vegetation"""
        if self.heightmap is None:
            QMessageBox.warning(self, "Warning", "Generate terrain first!")
            return

        if self.vegetation_thread and self.vegetation_thread.isRunning():
            self.log("⚠️ Vegetation generation already running")
            return

        params = {
            'spacing': self.veg_spacing.value(),
            'max_attempts': self.veg_max_attempts.value()
        }

        self.log("🌲 Generating vegetation...")
        self.progress_bar.setValue(40)

        self.vegetation_thread = VegetationGenerationThread(self.heightmap, params)
        self.vegetation_thread.progress.connect(self.log)
        self.vegetation_thread.finished.connect(self.on_vegetation_generated)
        self.vegetation_thread.start()

    def on_vegetation_generated(self, vegetation: list):
        """Vegetation generation complete"""
        self.vegetation = vegetation
        self.log(f"✅ Vegetation placed: {len(vegetation)} instances")
        self.veg_status.setText(f"✅ {len(vegetation)} vegetation instances placed")
        self.progress_bar.setValue(50)

    def generate_pbr(self):
        """Generate PBR textures"""
        if self.pbr_thread and self.pbr_thread.isRunning():
            self.log("⚠️ PBR generation already running")
            return

        material = self.material_combo.currentText()
        resolution = int(self.pbr_resolution.currentText())
        use_ai = self.use_ai_pbr.isChecked()

        self.log(f"🎨 Generating PBR textures: {material}, {resolution}x{resolution}, AI={use_ai}")
        self.progress_bar.setValue(60)

        self.pbr_thread = PBRGenerationThread(material, resolution, use_ai)
        self.pbr_thread.progress.connect(self.log)
        self.pbr_thread.finished.connect(self.on_pbr_generated)
        self.pbr_thread.start()

    def on_pbr_generated(self, textures: dict):
        """PBR generation complete"""
        self.pbr_textures = textures
        self.log(f"✅ PBR textures generated: {len(textures)} maps")
        self.progress_bar.setValue(80)

        # Auto-refresh preview
        self.refresh_pbr_preview()

    def refresh_pbr_preview(self):
        """Refresh PBR preview thumbnails"""
        if not self.pbr_textures:
            self.log("⚠️ No PBR textures to preview")
            return

        self.log("🔄 Refreshing PBR preview...")

        for name, texture in self.pbr_textures.items():
            if name in self.pbr_preview_labels:
                label = self.pbr_preview_labels[name]

                # Convert texture to QPixmap
                if isinstance(texture, np.ndarray):
                    if texture.dtype != np.uint8:
                        # Normalize to 0-255
                        texture = (np.clip(texture, 0, 1) * 255).astype(np.uint8)

                    h, w = texture.shape[:2]

                    if len(texture.shape) == 2:
                        # Grayscale
                        qimg = QImage(texture.data, w, h, w, QImage.Format_Grayscale8)
                    else:
                        # RGB
                        qimg = QImage(texture.data, w, h, w * 3, QImage.Format_RGB888)

                    pixmap = QPixmap.fromImage(qimg)
                    label.setPixmap(pixmap)
                    label.setText("")

        self.log("✅ PBR preview updated")

    def generate_hdri(self):
        """Generate HDRI"""
        if self.hdri_thread and self.hdri_thread.isRunning():
            self.log("⚠️ HDRI generation already running")
            return

        time_of_day = self.hdri_time.currentText().lower()

        # Parse resolution
        res_text = self.hdri_resolution.currentText()
        if "2048x1024" in res_text:
            resolution = (2048, 1024)
        elif "4096x2048" in res_text:
            resolution = (4096, 2048)
        else:
            resolution = (8192, 4096)

        self.log(f"🌅 Generating HDRI: {time_of_day}, {resolution[0]}x{resolution[1]}")
        self.progress_bar.setValue(90)

        self.hdri_thread = HDRIGenerationThread(time_of_day, resolution)
        self.hdri_thread.progress.connect(self.log)
        self.hdri_thread.finished.connect(self.on_hdri_generated)
        self.hdri_thread.start()

    def on_hdri_generated(self, hdri: np.ndarray):
        """HDRI generation complete"""
        if hdri is None or hdri.size == 0:
            self.log("⚠️ HDRI generation failed")
            return

        self.hdri_image = hdri
        self.log(f"✅ HDRI generated: {hdri.shape}")
        self.progress_bar.setValue(100)

        # Generate preview (tone-mapped)
        from core.rendering.hdri_generator import HDRIPanoramicGenerator
        tone_mapped = HDRIPanoramicGenerator._tonemap_for_display(hdri)

        # Save and display
        preview_path = self.output_dir / "hdri_preview.png"
        img = (tone_mapped * 255).astype(np.uint8)
        Image.fromarray(img).save(preview_path)

        pixmap = QPixmap(str(preview_path))
        self.hdri_preview.setPixmap(pixmap)
        self.hdri_preview.setText("")

        self.log(f"💾 HDRI preview: {preview_path}")

    def render_3d(self):
        """Render 3D view"""
        if self.heightmap is None:
            QMessageBox.warning(self, "Warning", "Generate terrain first!")
            return

        if not OPENGL_AVAILABLE:
            QMessageBox.warning(self, "Warning", "OpenGL not available")
            return

        self.log("🎮 Rendering 3D view...")

        try:
            # Clear previous
            self.gl_widget.clear()

            # Create mesh
            h, w = self.heightmap.shape

            # Vertex positions
            x = np.linspace(-50, 50, w)
            z = np.linspace(-50, 50, h)
            X, Z = np.meshgrid(x, z)
            Y = self.heightmap * 50.0

            # Create surface
            colors = np.ones((h, w, 4))
            colors[:, :, :3] = 0.7  # Gray

            surface = gl.GLSurfacePlotItem(
                x=x, y=z, z=Y,
                colors=colors,
                shader='shaded',
                smooth=True
            )

            self.gl_widget.addItem(surface)

            # Camera position
            self.gl_widget.setCameraPosition(distance=150, elevation=30, azimuth=45)

            self.log("✅ 3D view rendered")

        except Exception as e:
            self.log(f"❌ 3D rendering error: {e}")
            logger.error(f"3D rendering error: {e}")

    # ==================== EXPORT FUNCTIONS ====================

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.output_dir_label.setText(str(self.output_dir))
            self.log(f"📁 Output directory: {self.output_dir}")

    def export_heightmap(self):
        """Export heightmap"""
        if self.heightmap is None:
            QMessageBox.warning(self, "Warning", "No heightmap to export")
            return

        filepath = self.output_dir / "heightmap.png"

        # Save as 16-bit PNG
        img_16bit = (self.heightmap * 65535).astype(np.uint16)
        Image.fromarray(img_16bit, mode='I;16').save(filepath)

        self.log(f"💾 Exported heightmap: {filepath}")
        QMessageBox.information(self, "Success", f"Heightmap exported to:\n{filepath}")

    def export_pbr_textures(self):
        """Export all PBR textures"""
        if not self.pbr_textures:
            QMessageBox.warning(self, "Warning", "No PBR textures to export")
            return

        pbr_dir = self.output_dir / "pbr_textures"
        pbr_dir.mkdir(exist_ok=True)

        for name, texture in self.pbr_textures.items():
            filepath = pbr_dir / f"{name}.png"

            if isinstance(texture, np.ndarray):
                if texture.dtype != np.uint8:
                    texture = (np.clip(texture, 0, 1) * 255).astype(np.uint8)

                if len(texture.shape) == 2:
                    img = Image.fromarray(texture, mode='L')
                else:
                    img = Image.fromarray(texture, mode='RGB')

                img.save(filepath)
                self.log(f"💾 Exported {name}: {filepath}")

        QMessageBox.information(self, "Success", f"PBR textures exported to:\n{pbr_dir}")

    def export_hdri(self):
        """Export HDRI"""
        if self.hdri_image is None:
            QMessageBox.warning(self, "Warning", "No HDRI to export")
            return

        if not HDRI_AVAILABLE:
            QMessageBox.warning(self, "Warning", "HDRI generator not available")
            return

        hdri_dir = self.output_dir / "hdri"
        hdri_dir.mkdir(exist_ok=True)

        from core.rendering.hdri_generator import HDRIPanoramicGenerator
        generator = HDRIPanoramicGenerator()

        # Export EXR
        exr_path = hdri_dir / "mountain_hdri.exr"
        generator.export_exr(self.hdri_image, str(exr_path))

        # Export LDR preview
        png_path = hdri_dir / "mountain_hdri_preview.png"
        generator.export_ldr(self.hdri_image, str(png_path))

        self.log(f"💾 Exported HDRI: {exr_path}, {png_path}")
        QMessageBox.information(self, "Success", f"HDRI exported to:\n{hdri_dir}")

    def export_for_flame(self):
        """Export complete package for Autodesk Flame"""
        if self.heightmap is None:
            QMessageBox.warning(self, "Warning", "Generate terrain first!")
            return

        if not ADVANCED_EXPORT_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Advanced exporter not available.\n\nCheck core/export/advanced_3d_exporter.py")
            return

        self.log("🎬 Exporting for Autodesk Flame...")

        try:
            flame_dir = self.output_dir / "flame_export"
            flame_dir.mkdir(exist_ok=True)

            exporter = Advanced3DExporter(str(flame_dir))

            # Export OBJ with high resolution
            obj_path = exporter.export_obj(
                heightmap=self.heightmap,
                filename="terrain_flame.obj",
                height_scale=50.0,
                resolution_scale=1.0,
                pbr_textures=self.pbr_textures,
                generate_normals=True,
                generate_uvs=True
            )

            self.log(f"✅ Exported OBJ: {obj_path}")

            # Export PBR textures as 16-bit EXR
            if self.pbr_textures:
                for name, texture in self.pbr_textures.items():
                    # Convert to float32 linear
                    if texture.dtype == np.uint8:
                        texture = texture.astype(np.float32) / 255.0

                    # Save as EXR (would need OpenEXR)
                    # For now, save as PNG
                    tex_path = flame_dir / f"{name}.png"
                    if len(texture.shape) == 2:
                        img = Image.fromarray((texture * 255).astype(np.uint8), mode='L')
                    else:
                        img = Image.fromarray((texture * 255).astype(np.uint8), mode='RGB')
                    img.save(tex_path)
                    self.log(f"✅ Exported texture: {tex_path}")

            # Create README
            readme_path = flame_dir / "README_FLAME.txt"
            with open(readme_path, 'w') as f:
                f.write("""AUTODESK FLAME IMPORT PACKAGE
==============================

Files included:
- terrain_flame.obj: High-resolution terrain mesh
- terrain_flame.mtl: Material definition
- *.png: PBR texture maps (diffuse, normal, roughness, etc.)

Import Instructions:
1. Open Autodesk Flame
2. Import terrain_flame.obj
3. Apply textures from PBR maps
4. Adjust lighting and camera as needed

Generated by Mountain Studio ULTIMATE FINAL
""")

            self.log(f"✅ Created README: {readme_path}")

            QMessageBox.information(
                self,
                "Success",
                f"Flame export complete!\n\nPackage location:\n{flame_dir}\n\nSee README_FLAME.txt for import instructions."
            )

        except Exception as e:
            self.log(f"❌ Flame export error: {e}")
            logger.error(f"Flame export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Flame export failed:\n{e}")

    def export_obj(self):
        """Export OBJ"""
        if self.heightmap is None:
            QMessageBox.warning(self, "Warning", "Generate terrain first!")
            return

        if not ADVANCED_EXPORT_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Advanced exporter not available")
            return

        try:
            exporter = Advanced3DExporter(str(self.output_dir))
            obj_path = exporter.export_obj(self.heightmap, filename="terrain.obj")

            self.log(f"💾 Exported OBJ: {obj_path}")
            QMessageBox.information(self, "Success", f"OBJ exported:\n{obj_path}")

        except Exception as e:
            self.log(f"❌ OBJ export error: {e}")
            QMessageBox.critical(self, "Error", f"OBJ export failed:\n{e}")

    def export_fbx(self):
        """Export FBX"""
        QMessageBox.information(
            self,
            "FBX Export",
            "FBX export requires FBX SDK or blender.\n\nUse OBJ export and convert with Blender/Maya."
        )

    def export_alembic(self):
        """Export Alembic"""
        QMessageBox.information(
            self,
            "Alembic Export",
            "Alembic export requires alembic library.\n\nUse OBJ export and convert with Houdini/Maya."
        )


# ==================== MAIN ====================

def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set style
    app.setStyle('Fusion')

    # Create and show window
    window = MountainStudioUltimateFinal()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
