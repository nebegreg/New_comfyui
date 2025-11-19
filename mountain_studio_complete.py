#!/usr/bin/env python3
"""
🏔️ Mountain Studio COMPLETE - Edition Photorealistic
=====================================================

VERSION COMPLÈTE avec TOUTES les fonctionnalités:

✅ Terrain ultra-réaliste (érosion hydraulique + thermique)
✅ Viewer 3D PHOTOREALISTIC (PBR + atmosphère style Evian)
✅ Végétation intégrée (arbres avec Poisson disc sampling)
✅ Textures AI via ComfyUI (avec fallback procédural)
✅ PBR complet (Diffuse, Normal, Roughness, AO, Height, Metallic)
✅ HDRI environnement
✅ Exports professionnels
✅ Interface intuitive

Basé sur les standards 2024/2025 pour rendus photoréalistes.
Inspiré du style visuel des publicités Evian (Alpes françaises).

Author: Mountain Studio Team
License: MIT
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QGroupBox, QGridLayout,
    QTabWidget, QProgressBar, QMessageBox, QComboBox, QCheckBox,
    QTextEdit, QScrollArea, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

# Scientific
from scipy.ndimage import gaussian_filter

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available")

# Import custom modules
try:
    from ui.widgets.photorealistic_terrain_viewer import PhotorealisticTerrainViewer
    PHOTOREALISTIC_VIEWER_AVAILABLE = True
except ImportError:
    PHOTOREALISTIC_VIEWER_AVAILABLE = False
    print("⚠️ Photorealistic viewer not available")

try:
    from core.terrain.heightmap_generator_v2 import HeightmapGenerator
    TERRAIN_GEN_AVAILABLE = True
except ImportError:
    TERRAIN_GEN_AVAILABLE = False
    print("⚠️ Terrain generator not available")

try:
    from core.vegetation.vegetation_placer import VegetationPlacer
    from core.vegetation.biome_classifier import BiomeClassifier
    VEGETATION_AVAILABLE = True
except ImportError:
    VEGETATION_AVAILABLE = False
    print("⚠️ Vegetation system not available")

try:
    from core.rendering.pbr_texture_generator import PBRTextureGenerator
    PBR_AVAILABLE = True
except ImportError:
    PBR_AVAILABLE = False
    print("⚠️ PBR generator not available")

try:
    from core.ai.comfyui_integration import ComfyUIClient, generate_complete_pbr_set
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("⚠️ ComfyUI integration not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TERRAIN GENERATION THREAD
# =============================================================================

class TerrainGenerationThread(QThread):
    """Background thread for terrain generation"""

    progress = Signal(int)
    log_message = Signal(str)
    finished_terrain = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.log_message.emit("🏔️ Génération du terrain...")
            self.progress.emit(10)

            if TERRAIN_GEN_AVAILABLE:
                # Use advanced generator
                generator = HeightmapGenerator(
                    width=self.params['width'],
                    height=self.params['height']
                )

                terrain = generator.generate(
                    algorithm='hybrid',
                    scale=self.params['scale'],
                    octaves=self.params['octaves'],
                    ridge_influence=self.params['ridge_influence'],
                    warp_strength=self.params['warp_strength'],
                    hydraulic_iterations=self.params['hydraulic_iterations'],
                    thermal_iterations=self.params['thermal_iterations'],
                    seed=self.params['seed']
                )
            else:
                # Simple fallback
                from core.noise import fractional_brownian_motion
                terrain = fractional_brownian_motion(
                    self.params['width'],
                    self.params['height'],
                    octaves=self.params['octaves'],
                    seed=self.params['seed']
                )

            self.progress.emit(100)
            self.log_message.emit("✅ Terrain généré!")
            self.finished_terrain.emit(terrain)

        except Exception as e:
            logger.exception("Terrain generation failed")
            self.error.emit(str(e))


# =============================================================================
# VEGETATION GENERATION THREAD
# =============================================================================

class VegetationGenerationThread(QThread):
    """Background thread for vegetation placement"""

    progress = Signal(int)
    log_message = Signal(str)
    finished_vegetation = Signal(list)
    error = Signal(str)

    def __init__(self, heightmap: np.ndarray, params: Dict):
        super().__init__()
        self.heightmap = heightmap
        self.params = params

    def run(self):
        try:
            self.log_message.emit("🌲 Placement de la végétation...")
            self.progress.emit(10)

            if not VEGETATION_AVAILABLE:
                raise ImportError("Vegetation system not available")

            # Classify biomes
            self.log_message.emit("  📊 Classification des biomes...")
            classifier = BiomeClassifier()
            biome_map = classifier.classify(self.heightmap)

            self.progress.emit(30)

            # Place vegetation
            self.log_message.emit("  🌳 Placement des arbres (Poisson disc)...")
            placer = VegetationPlacer(
                width=self.heightmap.shape[1],
                height=self.heightmap.shape[0],
                heightmap=self.heightmap,
                biome_map=biome_map
            )

            trees = placer.place_vegetation(
                density=self.params['density'],
                min_spacing=self.params['spacing'],
                use_clustering=self.params['clustering'],
                cluster_size=self.params['cluster_size'],
                seed=self.params['seed']
            )

            self.progress.emit(100)
            self.log_message.emit(f"✅ {len(trees)} arbres placés!")
            self.finished_vegetation.emit(trees)

        except Exception as e:
            logger.exception("Vegetation generation failed")
            self.error.emit(str(e))


# =============================================================================
# PBR TEXTURE GENERATION THREAD
# =============================================================================

class PBRGenerationThread(QThread):
    """Background thread for PBR texture generation"""

    progress = Signal(int)
    log_message = Signal(str)
    finished_pbr = Signal(dict)
    error = Signal(str)

    def __init__(self, heightmap: np.ndarray, params: Dict):
        super().__init__()
        self.heightmap = heightmap
        self.params = params

    def run(self):
        try:
            self.log_message.emit("🎨 Génération des textures PBR...")
            self.progress.emit(10)

            material_type = self.params.get('material_type', 'rock')
            resolution = self.params.get('resolution', 2048)
            use_comfyui = self.params.get('use_comfyui', True)

            if use_comfyui and COMFYUI_AVAILABLE:
                # Try ComfyUI first
                self.log_message.emit("  🤖 Tentative avec ComfyUI (AI)...")
                self.progress.emit(20)

                pbr_textures = generate_complete_pbr_set(
                    heightmap=self.heightmap,
                    material_type=material_type,
                    resolution=resolution,
                    use_comfyui=True,
                    make_seamless=True
                )

                if pbr_textures.get('source') == 'comfyui':
                    self.log_message.emit("  ✅ Textures AI générées!")
                else:
                    self.log_message.emit("  ⚠️ ComfyUI non disponible, fallback procédural")

            else:
                # Procedural fallback
                self.log_message.emit("  🔧 Génération procédurale...")
                self.progress.emit(20)

                if PBR_AVAILABLE:
                    generator = PBRTextureGenerator(resolution=resolution)
                    pbr_textures = generator.generate_from_heightmap(
                        self.heightmap,
                        material_type=material_type,
                        make_seamless=True,
                        detail_level=1.0
                    )
                    pbr_textures['source'] = 'procedural'
                else:
                    raise ImportError("PBR generator not available")

            self.progress.emit(100)
            self.log_message.emit(f"✅ Textures PBR générées ({pbr_textures['source']})!")
            self.finished_pbr.emit(pbr_textures)

        except Exception as e:
            logger.exception("PBR generation failed")
            self.error.emit(str(e))


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MountainStudioComplete(QMainWindow):
    """
    Mountain Studio COMPLETE - Edition Photorealistic

    Application complète pour la génération de terrains montagneux photoréalistes
    avec végétation, textures PBR, et rendu 3D avancé style Evian.
    """

    def __init__(self):
        super().__init__()

        self.terrain = None
        self.tree_instances = []
        self.pbr_textures = None

        self.output_dir = Path.home() / "MountainStudio_Output"
        self.output_dir.mkdir(exist_ok=True)

        self.init_ui()

        logger.info("🏔️ Mountain Studio COMPLETE initialized")
        self.log("🏔️ Bienvenue dans Mountain Studio COMPLETE!")
        self.log(f"📁 Répertoire de sortie: {self.output_dir}")
        self.log("")
        self.log("✨ Fonctionnalités disponibles:")
        self.log(f"  - Viewer 3D photoréaliste: {PHOTOREALISTIC_VIEWER_AVAILABLE}")
        self.log(f"  - Génération terrain avancée: {TERRAIN_GEN_AVAILABLE}")
        self.log(f"  - Système de végétation: {VEGETATION_AVAILABLE}")
        self.log(f"  - Textures PBR: {PBR_AVAILABLE}")
        self.log(f"  - AI via ComfyUI: {COMFYUI_AVAILABLE}")
        self.log("")
        self.log("💡 Commencez par générer un terrain (onglet 'Terrain')!")

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Mountain Studio COMPLETE - Photorealistic Edition")
        self.setGeometry(100, 100, 1800, 1000)

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel: Controls (500px)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(500)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_terrain_tab(), "🏔️ Terrain")
        self.tabs.addTab(self._create_vegetation_tab(), "🌲 Végétation")
        self.tabs.addTab(self._create_pbr_tab(), "🎨 Textures PBR")
        self.tabs.addTab(self._create_rendering_tab(), "💡 Rendu 3D")
        self.tabs.addTab(self._create_export_tab(), "💾 Export")

        left_layout.addWidget(self.tabs)

        # Progress bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Log area
        log_group = QGroupBox("📋 Journal")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)

        main_layout.addWidget(left_panel)

        # Right panel: 3D Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        viewer_group = QGroupBox("🎮 Aperçu 3D Photorealistic (Style Evian)")
        viewer_layout = QVBoxLayout()

        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            self.viewer_3d = PhotorealisticTerrainViewer()
            viewer_layout.addWidget(self.viewer_3d)

            # Viewer controls
            viewer_controls = QHBoxLayout()

            self.wireframe_btn = QPushButton("🔲 Wireframe")
            self.wireframe_btn.setCheckable(True)
            self.wireframe_btn.clicked.connect(self.toggle_wireframe)
            viewer_controls.addWidget(self.wireframe_btn)

            self.vegetation_btn = QPushButton("🌲 Arbres")
            self.vegetation_btn.setCheckable(True)
            self.vegetation_btn.setChecked(True)
            self.vegetation_btn.clicked.connect(self.toggle_vegetation_display)
            viewer_controls.addWidget(self.vegetation_btn)

            self.reset_camera_btn = QPushButton("📷 Reset Camera")
            self.reset_camera_btn.clicked.connect(self.reset_camera)
            viewer_controls.addWidget(self.reset_camera_btn)

            viewer_layout.addLayout(viewer_controls)
        else:
            viewer_layout.addWidget(QLabel("⚠️ Viewer photoréaliste non disponible"))

        viewer_group.setLayout(viewer_layout)
        right_layout.addWidget(viewer_group)

        main_layout.addWidget(right_panel, stretch=1)

    def _create_terrain_tab(self) -> QWidget:
        """Create terrain generation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Resolution
        res_group = QGroupBox("📐 Résolution")
        res_layout = QGridLayout()

        res_layout.addWidget(QLabel("Largeur:"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(128, 2048)
        self.width_spin.setValue(512)
        self.width_spin.setSingleStep(64)
        res_layout.addWidget(self.width_spin, 0, 1)

        res_layout.addWidget(QLabel("Hauteur:"), 1, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(128, 2048)
        self.height_spin.setValue(512)
        self.height_spin.setSingleStep(64)
        res_layout.addWidget(self.height_spin, 1, 1)

        res_group.setLayout(res_layout)
        scroll_layout.addWidget(res_group)

        # Noise parameters
        noise_group = QGroupBox("🌊 Paramètres de bruit")
        noise_layout = QGridLayout()

        noise_layout.addWidget(QLabel("Scale:"), 0, 0)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 500)
        self.scale_slider.setValue(100)
        self.scale_label = QLabel("100")
        self.scale_slider.valueChanged.connect(lambda v: self.scale_label.setText(str(v)))
        noise_layout.addWidget(self.scale_slider, 0, 1)
        noise_layout.addWidget(self.scale_label, 0, 2)

        noise_layout.addWidget(QLabel("Octaves:"), 1, 0)
        self.octaves_spin = QSpinBox()
        self.octaves_spin.setRange(1, 12)
        self.octaves_spin.setValue(8)
        noise_layout.addWidget(self.octaves_spin, 1, 1, 1, 2)

        noise_layout.addWidget(QLabel("Influence Ridge:"), 2, 0)
        self.ridge_slider = QSlider(Qt.Horizontal)
        self.ridge_slider.setRange(0, 100)
        self.ridge_slider.setValue(40)
        self.ridge_label = QLabel("0.40")
        self.ridge_slider.valueChanged.connect(lambda v: self.ridge_label.setText(f"{v/100:.2f}"))
        noise_layout.addWidget(self.ridge_slider, 2, 1)
        noise_layout.addWidget(self.ridge_label, 2, 2)

        noise_layout.addWidget(QLabel("Domain Warp:"), 3, 0)
        self.warp_slider = QSlider(Qt.Horizontal)
        self.warp_slider.setRange(0, 100)
        self.warp_slider.setValue(30)
        self.warp_label = QLabel("0.30")
        self.warp_slider.valueChanged.connect(lambda v: self.warp_label.setText(f"{v/100:.2f}"))
        noise_layout.addWidget(self.warp_slider, 3, 1)
        noise_layout.addWidget(self.warp_label, 3, 2)

        noise_group.setLayout(noise_layout)
        scroll_layout.addWidget(noise_group)

        # Erosion
        erosion_group = QGroupBox("💧 Érosion")
        erosion_layout = QGridLayout()

        erosion_layout.addWidget(QLabel("Hydraulique (iter):"), 0, 0)
        self.hydraulic_spin = QSpinBox()
        self.hydraulic_spin.setRange(0, 100)
        self.hydraulic_spin.setValue(50)
        erosion_layout.addWidget(self.hydraulic_spin, 0, 1)

        erosion_layout.addWidget(QLabel("Thermique (iter):"), 1, 0)
        self.thermal_spin = QSpinBox()
        self.thermal_spin.setRange(0, 20)
        self.thermal_spin.setValue(5)
        erosion_layout.addWidget(self.thermal_spin, 1, 1)

        erosion_group.setLayout(erosion_layout)
        scroll_layout.addWidget(erosion_group)

        # Seed
        seed_group = QGroupBox("🎲 Seed")
        seed_layout = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        seed_layout.addWidget(self.seed_spin)
        randomize_btn = QPushButton("🎲 Aléatoire")
        randomize_btn.clicked.connect(lambda: self.seed_spin.setValue(np.random.randint(0, 999999)))
        seed_layout.addWidget(randomize_btn)
        seed_group.setLayout(seed_layout)
        scroll_layout.addWidget(seed_group)

        # Generate button
        self.generate_terrain_btn = QPushButton("🏔️ GÉNÉRER TERRAIN")
        self.generate_terrain_btn.setStyleSheet(
            "QPushButton { background-color: #2ecc71; color: white; font-weight: bold; padding: 12px; font-size: 14px; }"
        )
        self.generate_terrain_btn.clicked.connect(self.generate_terrain)
        scroll_layout.addWidget(self.generate_terrain_btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return tab

    def _create_vegetation_tab(self) -> QWidget:
        """Create vegetation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not VEGETATION_AVAILABLE:
            layout.addWidget(QLabel("⚠️ Système de végétation non disponible"))
            return tab

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Density
        density_group = QGroupBox("🌲 Densité")
        density_layout = QGridLayout()

        density_layout.addWidget(QLabel("Densité globale:"), 0, 0)
        self.veg_density_slider = QSlider(Qt.Horizontal)
        self.veg_density_slider.setRange(1, 100)
        self.veg_density_slider.setValue(50)
        self.veg_density_label = QLabel("0.50")
        self.veg_density_slider.valueChanged.connect(lambda v: self.veg_density_label.setText(f"{v/100:.2f}"))
        density_layout.addWidget(self.veg_density_slider, 0, 1)
        density_layout.addWidget(self.veg_density_label, 0, 2)

        density_layout.addWidget(QLabel("Espacement min (px):"), 1, 0)
        self.veg_spacing_spin = QSpinBox()
        self.veg_spacing_spin.setRange(1, 20)
        self.veg_spacing_spin.setValue(5)
        density_layout.addWidget(self.veg_spacing_spin, 1, 1, 1, 2)

        density_group.setLayout(density_layout)
        scroll_layout.addWidget(density_group)

        # Clustering
        cluster_group = QGroupBox("🌳 Groupements (Clustering)")
        cluster_layout = QGridLayout()

        self.veg_clustering_check = QCheckBox("Activer clustering")
        self.veg_clustering_check.setChecked(True)
        cluster_layout.addWidget(self.veg_clustering_check, 0, 0, 1, 2)

        cluster_layout.addWidget(QLabel("Taille cluster:"), 1, 0)
        self.veg_cluster_size_spin = QSpinBox()
        self.veg_cluster_size_spin.setRange(3, 15)
        self.veg_cluster_size_spin.setValue(5)
        cluster_layout.addWidget(self.veg_cluster_size_spin, 1, 1)

        cluster_group.setLayout(cluster_layout)
        scroll_layout.addWidget(cluster_group)

        # Info
        info_label = QLabel(
            "ℹ️ La végétation est placée selon:\n"
            "• Biomes (altitude, pente)\n"
            "• Poisson disc sampling (distribution naturelle)\n"
            "• Clustering (groupes d'arbres réalistes)\n\n"
            "Les arbres apparaîtront dans le viewer 3D."
        )
        info_label.setWordWrap(True)
        scroll_layout.addWidget(info_label)

        # Generate button
        self.generate_vegetation_btn = QPushButton("🌲 PLACER VÉGÉTATION")
        self.generate_vegetation_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 12px; font-size: 14px; }"
        )
        self.generate_vegetation_btn.clicked.connect(self.generate_vegetation)
        self.generate_vegetation_btn.setEnabled(False)  # Enabled after terrain generation
        scroll_layout.addWidget(self.generate_vegetation_btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return tab

    def _create_pbr_tab(self) -> QWidget:
        """Create PBR textures tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Material type
        material_group = QGroupBox("🪨 Type de matériau")
        material_layout = QVBoxLayout()

        material_layout.addWidget(QLabel("Matériau:"))
        self.material_combo = QComboBox()
        self.material_combo.addItems(['rock', 'grass', 'snow', 'sand', 'dirt'])
        material_layout.addWidget(self.material_combo)

        material_group.setLayout(material_layout)
        scroll_layout.addWidget(material_group)

        # Resolution
        res_group = QGroupBox("📐 Résolution textures")
        res_layout = QVBoxLayout()

        res_layout.addWidget(QLabel("Résolution:"))
        self.pbr_res_combo = QComboBox()
        self.pbr_res_combo.addItems(['512', '1024', '2048', '4096'])
        self.pbr_res_combo.setCurrentText('2048')
        res_layout.addWidget(self.pbr_res_combo)

        res_group.setLayout(res_layout)
        scroll_layout.addWidget(res_group)

        # ComfyUI
        comfyui_group = QGroupBox("🤖 ComfyUI (AI)")
        comfyui_layout = QVBoxLayout()

        self.use_comfyui_check = QCheckBox("Utiliser ComfyUI pour génération AI")
        self.use_comfyui_check.setChecked(COMFYUI_AVAILABLE)
        self.use_comfyui_check.setEnabled(COMFYUI_AVAILABLE)
        comfyui_layout.addWidget(self.use_comfyui_check)

        if COMFYUI_AVAILABLE:
            status_label = QLabel("✅ ComfyUI disponible")
            status_label.setStyleSheet("color: green;")
        else:
            status_label = QLabel("⚠️ ComfyUI non disponible (fallback procédural)")
            status_label.setStyleSheet("color: orange;")
        comfyui_layout.addWidget(status_label)

        comfyui_info = QLabel(
            "ℹ️ ComfyUI doit tourner sur localhost:8188\n"
            "Voir le guide COMFYUI_GUIDE.md pour setup."
        )
        comfyui_info.setWordWrap(True)
        comfyui_layout.addWidget(comfyui_info)

        comfyui_group.setLayout(comfyui_layout)
        scroll_layout.addWidget(comfyui_group)

        # Info
        info_label = QLabel(
            "📦 Textures PBR générées:\n"
            "• Diffuse/Albedo (couleur)\n"
            "• Normal (relief)\n"
            "• Roughness (rugosité)\n"
            "• AO (ombres ambiantes)\n"
            "• Height (displacement)\n"
            "• Metallic (réflectivité)\n\n"
            "Toutes seamless/tileable!"
        )
        info_label.setWordWrap(True)
        scroll_layout.addWidget(info_label)

        # Generate button
        self.generate_pbr_btn = QPushButton("🎨 GÉNÉRER TEXTURES PBR")
        self.generate_pbr_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 12px; font-size: 14px; }"
        )
        self.generate_pbr_btn.clicked.connect(self.generate_pbr)
        self.generate_pbr_btn.setEnabled(False)  # Enabled after terrain generation
        scroll_layout.addWidget(self.generate_pbr_btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return tab

    def _create_rendering_tab(self) -> QWidget:
        """Create 3D rendering controls tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not PHOTOREALISTIC_VIEWER_AVAILABLE:
            layout.addWidget(QLabel("⚠️ Viewer photoréaliste non disponible"))
            return tab

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Sun
        sun_group = QGroupBox("☀️ Soleil")
        sun_layout = QGridLayout()

        sun_layout.addWidget(QLabel("Azimuth (0-360°):"), 0, 0)
        self.sun_azimuth_slider = QSlider(Qt.Horizontal)
        self.sun_azimuth_slider.setRange(0, 360)
        self.sun_azimuth_slider.setValue(135)
        self.sun_azimuth_label = QLabel("135°")
        self.sun_azimuth_slider.valueChanged.connect(lambda v: self.sun_azimuth_label.setText(f"{v}°"))
        self.sun_azimuth_slider.valueChanged.connect(self.update_lighting)
        sun_layout.addWidget(self.sun_azimuth_slider, 0, 1)
        sun_layout.addWidget(self.sun_azimuth_label, 0, 2)

        sun_layout.addWidget(QLabel("Elevation (0-90°):"), 1, 0)
        self.sun_elevation_slider = QSlider(Qt.Horizontal)
        self.sun_elevation_slider.setRange(0, 90)
        self.sun_elevation_slider.setValue(45)
        self.sun_elevation_label = QLabel("45°")
        self.sun_elevation_slider.valueChanged.connect(lambda v: self.sun_elevation_label.setText(f"{v}°"))
        self.sun_elevation_slider.valueChanged.connect(self.update_lighting)
        sun_layout.addWidget(self.sun_elevation_slider, 1, 1)
        sun_layout.addWidget(self.sun_elevation_label, 1, 2)

        sun_layout.addWidget(QLabel("Intensité:"), 2, 0)
        self.sun_intensity_slider = QSlider(Qt.Horizontal)
        self.sun_intensity_slider.setRange(50, 200)
        self.sun_intensity_slider.setValue(120)
        self.sun_intensity_label = QLabel("1.20")
        self.sun_intensity_slider.valueChanged.connect(lambda v: self.sun_intensity_label.setText(f"{v/100:.2f}"))
        self.sun_intensity_slider.valueChanged.connect(self.update_lighting)
        sun_layout.addWidget(self.sun_intensity_slider, 2, 1)
        sun_layout.addWidget(self.sun_intensity_label, 2, 2)

        sun_group.setLayout(sun_layout)
        scroll_layout.addWidget(sun_group)

        # Atmosphere
        atmo_group = QGroupBox("🌫️ Atmosphère & Brouillard")
        atmo_layout = QGridLayout()

        self.fog_enabled_check = QCheckBox("Activer brouillard")
        self.fog_enabled_check.setChecked(True)
        self.fog_enabled_check.stateChanged.connect(self.update_atmosphere)
        atmo_layout.addWidget(self.fog_enabled_check, 0, 0, 1, 3)

        atmo_layout.addWidget(QLabel("Densité brouillard:"), 1, 0)
        self.fog_density_slider = QSlider(Qt.Horizontal)
        self.fog_density_slider.setRange(1, 50)
        self.fog_density_slider.setValue(15)
        self.fog_density_label = QLabel("0.015")
        self.fog_density_slider.valueChanged.connect(lambda v: self.fog_density_label.setText(f"{v/1000:.3f}"))
        self.fog_density_slider.valueChanged.connect(self.update_atmosphere)
        atmo_layout.addWidget(self.fog_density_slider, 1, 1)
        atmo_layout.addWidget(self.fog_density_label, 1, 2)

        self.atmosphere_enabled_check = QCheckBox("Activer scattering atmosphérique")
        self.atmosphere_enabled_check.setChecked(True)
        self.atmosphere_enabled_check.stateChanged.connect(self.update_atmosphere)
        atmo_layout.addWidget(self.atmosphere_enabled_check, 2, 0, 1, 3)

        atmo_group.setLayout(atmo_layout)
        scroll_layout.addWidget(atmo_group)

        # Height scale
        height_group = QGroupBox("📏 Échelle hauteur")
        height_layout = QGridLayout()

        height_layout.addWidget(QLabel("Multiplicateur:"), 0, 0)
        self.height_scale_slider = QSlider(Qt.Horizontal)
        self.height_scale_slider.setRange(10, 200)
        self.height_scale_slider.setValue(50)
        self.height_scale_label = QLabel("50")
        self.height_scale_slider.valueChanged.connect(lambda v: self.height_scale_label.setText(str(v)))
        self.height_scale_slider.valueChanged.connect(self.update_height_scale)
        height_layout.addWidget(self.height_scale_slider, 0, 1)
        height_layout.addWidget(self.height_scale_label, 0, 2)

        height_group.setLayout(height_layout)
        scroll_layout.addWidget(height_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return tab

    def _create_export_tab(self) -> QWidget:
        """Create export tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Output directory
        dir_group = QGroupBox("📁 Répertoire de sortie")
        dir_layout = QVBoxLayout()
        self.output_dir_label = QLabel(str(self.output_dir))
        self.output_dir_label.setWordWrap(True)
        dir_layout.addWidget(self.output_dir_label)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Quick exports
        quick_group = QGroupBox("⚡ Exports rapides")
        quick_layout = QVBoxLayout()

        export_heightmap_btn = QPushButton("💾 Exporter Heightmap (PNG 16-bit)")
        export_heightmap_btn.clicked.connect(self.export_heightmap)
        quick_layout.addWidget(export_heightmap_btn)

        export_pbr_btn = QPushButton("💾 Exporter Textures PBR")
        export_pbr_btn.clicked.connect(self.export_pbr_textures)
        quick_layout.addWidget(export_pbr_btn)

        export_vegetation_btn = QPushButton("💾 Exporter Végétation (JSON)")
        export_vegetation_btn.clicked.connect(self.export_vegetation)
        quick_layout.addWidget(export_vegetation_btn)

        export_all_btn = QPushButton("📦 EXPORTER TOUT")
        export_all_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; padding: 10px; }"
        )
        export_all_btn.clicked.connect(self.export_all)
        quick_layout.addWidget(export_all_btn)

        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)

        layout.addStretch()

        return tab

    # =========================================================================
    # TERRAIN GENERATION
    # =========================================================================

    def generate_terrain(self):
        """Generate terrain in background thread"""
        if not self.generate_terrain_btn.isEnabled():
            return

        self.generate_terrain_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("🏔️ Génération du terrain...")

        params = {
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'scale': self.scale_slider.value(),
            'octaves': self.octaves_spin.value(),
            'ridge_influence': self.ridge_slider.value() / 100.0,
            'warp_strength': self.warp_slider.value() / 100.0,
            'hydraulic_iterations': self.hydraulic_spin.value(),
            'thermal_iterations': self.thermal_spin.value(),
            'seed': self.seed_spin.value()
        }

        self.terrain_thread = TerrainGenerationThread(params)
        self.terrain_thread.progress.connect(self.progress_bar.setValue)
        self.terrain_thread.log_message.connect(self.log)
        self.terrain_thread.finished_terrain.connect(self.on_terrain_generated)
        self.terrain_thread.error.connect(self.on_error)
        self.terrain_thread.start()

    def on_terrain_generated(self, terrain: np.ndarray):
        """Handle terrain generation completion"""
        self.terrain = terrain
        self.log(f"✅ Terrain généré: {terrain.shape}")

        # Update 3D viewer
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            height_scale = self.height_scale_slider.value()
            self.viewer_3d.set_terrain(terrain, height_scale, self.pbr_textures)

        # Enable dependent buttons
        self.generate_vegetation_btn.setEnabled(VEGETATION_AVAILABLE)
        self.generate_pbr_btn.setEnabled(PBR_AVAILABLE or COMFYUI_AVAILABLE)

        self.generate_terrain_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        QMessageBox.information(self, "Succès", "Terrain généré avec succès!")

    # =========================================================================
    # VEGETATION
    # =========================================================================

    def generate_vegetation(self):
        """Generate vegetation in background thread"""
        if self.terrain is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        if not self.generate_vegetation_btn.isEnabled():
            return

        self.generate_vegetation_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("🌲 Placement de la végétation...")

        params = {
            'density': self.veg_density_slider.value() / 100.0,
            'spacing': float(self.veg_spacing_spin.value()),
            'clustering': self.veg_clustering_check.isChecked(),
            'cluster_size': self.veg_cluster_size_spin.value(),
            'seed': self.seed_spin.value()
        }

        self.veg_thread = VegetationGenerationThread(self.terrain, params)
        self.veg_thread.progress.connect(self.progress_bar.setValue)
        self.veg_thread.log_message.connect(self.log)
        self.veg_thread.finished_vegetation.connect(self.on_vegetation_generated)
        self.veg_thread.error.connect(self.on_error)
        self.veg_thread.start()

    def on_vegetation_generated(self, trees: List):
        """Handle vegetation generation completion"""
        self.tree_instances = trees
        self.log(f"✅ Végétation placée: {len(trees)} arbres")

        # Update 3D viewer
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            self.viewer_3d.set_vegetation(trees)

        self.generate_vegetation_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        QMessageBox.information(self, "Succès", f"{len(trees)} arbres placés!")

    # =========================================================================
    # PBR TEXTURES
    # =========================================================================

    def generate_pbr(self):
        """Generate PBR textures in background thread"""
        if self.terrain is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        if not self.generate_pbr_btn.isEnabled():
            return

        self.generate_pbr_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("🎨 Génération des textures PBR...")

        params = {
            'material_type': self.material_combo.currentText(),
            'resolution': int(self.pbr_res_combo.currentText()),
            'use_comfyui': self.use_comfyui_check.isChecked()
        }

        self.pbr_thread = PBRGenerationThread(self.terrain, params)
        self.pbr_thread.progress.connect(self.progress_bar.setValue)
        self.pbr_thread.log_message.connect(self.log)
        self.pbr_thread.finished_pbr.connect(self.on_pbr_generated)
        self.pbr_thread.error.connect(self.on_error)
        self.pbr_thread.start()

    def on_pbr_generated(self, pbr_textures: Dict):
        """Handle PBR generation completion"""
        self.pbr_textures = pbr_textures
        source = pbr_textures.get('source', 'unknown')
        self.log(f"✅ Textures PBR générées ({source})!")

        # Update 3D viewer with new textures
        if PHOTOREALISTIC_VIEWER_AVAILABLE and self.terrain is not None:
            height_scale = self.height_scale_slider.value()
            self.viewer_3d.set_terrain(self.terrain, height_scale, pbr_textures)

        self.generate_pbr_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        QMessageBox.information(
            self,
            "Succès",
            f"Textures PBR générées ({source})!\n\n"
            "Le rendu 3D a été mis à jour avec les nouvelles textures."
        )

    # =========================================================================
    # 3D RENDERING CONTROLS
    # =========================================================================

    def update_lighting(self):
        """Update 3D viewer lighting"""
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            azimuth = self.sun_azimuth_slider.value()
            elevation = self.sun_elevation_slider.value()
            intensity = self.sun_intensity_slider.value() / 100.0

            self.viewer_3d.set_lighting(azimuth, elevation, intensity, 0.4)

    def update_atmosphere(self):
        """Update atmospheric parameters"""
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            fog_density = self.fog_density_slider.value() / 1000.0
            fog_enabled = self.fog_enabled_check.isChecked()
            atmosphere_enabled = self.atmosphere_enabled_check.isChecked()

            self.viewer_3d.set_atmosphere(fog_density, fog_enabled, atmosphere_enabled)

    def update_height_scale(self):
        """Update terrain height scale"""
        if PHOTOREALISTIC_VIEWER_AVAILABLE and self.terrain is not None:
            height_scale = self.height_scale_slider.value()
            self.viewer_3d.set_terrain(self.terrain, height_scale, self.pbr_textures)

    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            self.viewer_3d.toggle_wireframe()

    def toggle_vegetation_display(self):
        """Toggle vegetation display"""
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            self.viewer_3d.toggle_vegetation()

    def reset_camera(self):
        """Reset camera to default"""
        if PHOTOREALISTIC_VIEWER_AVAILABLE:
            self.viewer_3d.reset_camera()

    # =========================================================================
    # EXPORTS
    # =========================================================================

    def export_heightmap(self):
        """Export heightmap as PNG 16-bit"""
        if self.terrain is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        try:
            filepath = self.output_dir / "heightmap_16bit.png"
            terrain_uint16 = (self.terrain * 65535).astype(np.uint16)
            img = Image.fromarray(terrain_uint16, mode='I;16')
            img.save(filepath)

            self.log(f"💾 Heightmap exporté: {filepath}")
            QMessageBox.information(self, "Succès", f"Heightmap exporté:\n{filepath}")
        except Exception as e:
            self.log(f"❌ Erreur export: {e}")
            QMessageBox.critical(self, "Erreur", f"Export échoué:\n{e}")

    def export_pbr_textures(self):
        """Export PBR textures"""
        if self.pbr_textures is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord des textures PBR!")
            return

        try:
            material = self.material_combo.currentText()
            prefix = f"terrain_{material}"

            exported = {}
            for name, texture in self.pbr_textures.items():
                if isinstance(texture, str):  # Skip 'source' key
                    continue

                filename = f"{prefix}_{name}.png"
                filepath = self.output_dir / filename

                if len(texture.shape) == 2:
                    img = Image.fromarray(texture, mode='L')
                else:
                    img = Image.fromarray(texture, mode='RGB')

                img.save(filepath)
                exported[name] = filepath
                self.log(f"  💾 {name}: {filepath}")

            QMessageBox.information(
                self,
                "Succès",
                f"{len(exported)} textures PBR exportées:\n{self.output_dir}"
            )
        except Exception as e:
            self.log(f"❌ Erreur export PBR: {e}")
            QMessageBox.critical(self, "Erreur", f"Export PBR échoué:\n{e}")

    def export_vegetation(self):
        """Export vegetation instances as JSON"""
        if not self.tree_instances:
            QMessageBox.warning(self, "Attention", "Générez d'abord de la végétation!")
            return

        try:
            if VEGETATION_AVAILABLE:
                from core.vegetation.vegetation_placer import VegetationPlacer

                # Create temporary placer to use export method
                h, w = self.terrain.shape
                placer = VegetationPlacer(w, h, self.terrain, np.zeros_like(self.terrain))
                placer.tree_instances = self.tree_instances

                filepath = self.output_dir / "vegetation_instances.json"
                placer.export_instances(str(filepath))

                self.log(f"💾 Végétation exportée: {filepath}")
                QMessageBox.information(
                    self,
                    "Succès",
                    f"Végétation exportée:\n{filepath}\n\n"
                    f"{len(self.tree_instances)} arbres"
                )
        except Exception as e:
            self.log(f"❌ Erreur export végétation: {e}")
            QMessageBox.critical(self, "Erreur", f"Export végétation échoué:\n{e}")

    def export_all(self):
        """Export everything"""
        if self.terrain is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        self.log("📦 Export complet...")

        # Export heightmap
        self.export_heightmap()

        # Export PBR if available
        if self.pbr_textures:
            self.export_pbr_textures()

        # Export vegetation if available
        if self.tree_instances:
            self.export_vegetation()

        # Create README
        readme_path = self.output_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("Mountain Studio COMPLETE - Export Package\n")
            f.write("=" * 60 + "\n\n")
            f.write("Contenu:\n")
            f.write(f"- Terrain: {self.terrain.shape}\n")
            f.write(f"- Textures PBR: {self.pbr_textures is not None}\n")
            f.write(f"- Végétation: {len(self.tree_instances)} arbres\n\n")
            f.write("Fichiers:\n")
            f.write("- heightmap_16bit.png: Heightmap 16-bit\n")
            if self.pbr_textures:
                f.write("- terrain_*_*.png: Textures PBR\n")
            if self.tree_instances:
                f.write("- vegetation_instances.json: Instances d'arbres\n")
            f.write("\nGénéré avec Mountain Studio COMPLETE\n")

        self.log(f"💾 README: {readme_path}")
        self.log("✅ Export complet terminé!")

        QMessageBox.information(
            self,
            "Succès",
            f"Export complet terminé!\n\nRépertoire:\n{self.output_dir}"
        )

    # =========================================================================
    # LOGGING & ERRORS
    # =========================================================================

    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        logger.info(message)

    def on_error(self, error: str):
        """Handle error"""
        self.log(f"❌ Erreur: {error}")
        self.progress_bar.setValue(0)

        # Re-enable buttons
        self.generate_terrain_btn.setEnabled(True)
        if self.terrain is not None:
            self.generate_vegetation_btn.setEnabled(VEGETATION_AVAILABLE)
            self.generate_pbr_btn.setEnabled(PBR_AVAILABLE or COMFYUI_AVAILABLE)

        QMessageBox.critical(self, "Erreur", f"Une erreur s'est produite:\n{error}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MountainStudioComplete()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
