"""
Interface PySide6 Professionnelle pour Simulation de Montagne
Outil destiné aux graphistes professionnels
"""

import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QPushButton, QComboBox, QProgressBar, QTextEdit, QFileDialog,
    QSplitter, QCheckBox, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Old modules (still used for some features)
from comfyui_integration import ComfyUIIntegration, StableDiffusionDirect
from temporal_consistency import VideoCoherenceManager
from video_generator import VideoGenerator

# New core modules (Mountain Studio Pro v2.0)
from core.config.preset_manager import PresetManager
from core.terrain.heightmap_generator_v2 import HeightmapGeneratorV2  # Ultra-realistic V2
from core.terrain.hydraulic_erosion import HydraulicErosionSystem
from core.terrain.thermal_erosion import ThermalErosionSystem
from core.vegetation.biome_classifier import BiomeClassifier
from core.vegetation.vegetation_placer import VegetationPlacer
from core.rendering.pbr_splatmap_generator import PBRSplatmapGenerator
from core.rendering.vfx_prompt_generator import VFXPromptGenerator
from core.export.professional_exporter import ProfessionalExporter
from core.ai.comfyui_integration import ComfyUIClient, generate_pbr_textures, generate_landscape_image


class GenerationThread(QThread):
    """Thread pour génération asynchrone"""
    progress = Signal(int, str)
    finished = Signal(object, str)  # (result, type)
    error = Signal(str)

    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params

    def run(self):
        try:
            if self.task_type == "terrain":
                self.generate_terrain()
            elif self.task_type == "texture":
                self.generate_texture()
            elif self.task_type == "video":
                self.generate_video()
        except Exception as e:
            self.error.emit(f"Erreur: {str(e)}")

    def generate_terrain(self):
        """Generate terrain using new HeightmapGenerator with advanced erosion"""
        self.progress.emit(5, "Initialisation générateur...")

        # Load preset if specified
        preset = None
        if self.params.get('use_preset') and self.params.get('preset_name'):
            self.progress.emit(8, f"Chargement preset '{self.params['preset_name']}'...")
            preset_mgr = PresetManager()
            preset = preset_mgr.get_preset(self.params['preset_name'])

            # Extract terrain parameters from preset
            terrain_cfg = preset.get('terrain', {})
            erosion_cfg = preset.get('erosion', {})
            hydraulic_cfg = erosion_cfg.get('hydraulic', {})

            # Use preset values
            resolution = preset.get('resolution', self.params['resolution'])
            scale = terrain_cfg.get('base_scale', 100.0)
            octaves = terrain_cfg.get('octaves', 8)
            persistence = terrain_cfg.get('persistence', 0.5)
            lacunarity = terrain_cfg.get('lacunarity', 2.0)
            seed = terrain_cfg.get('seed', 42)
            apply_hydraulic = hydraulic_cfg.get('enabled', True)
            apply_thermal = erosion_cfg.get('thermal', {}).get('enabled', True)
            erosion_iters = hydraulic_cfg.get('iterations', 50) * 1000
        else:
            # Use UI parameters
            resolution = self.params['resolution']
            scale = self.params.get('scale', 100.0)
            octaves = self.params.get('octaves', 8)
            persistence = self.params.get('persistence', 0.5)
            lacunarity = self.params.get('lacunarity', 2.0)
            seed = self.params.get('seed', 42)
            apply_hydraulic = self.params.get('hydraulic_enabled', True)
            apply_thermal = self.params.get('thermal_enabled', True)
            erosion_iters = self.params.get('hydraulic_iterations', 50) * 1000

        # Create ULTRA-REALISTIC terrain generator V2
        terrain_gen = HeightmapGeneratorV2(width=resolution, height=resolution)

        self.progress.emit(15, "Génération heightmap ULTRA-RÉALISTE...")

        # Use preset name if available, otherwise use mountain type
        preset_name = self.params.get('preset_name') if self.params.get('use_preset') else None

        heightmap = terrain_gen.generate(
            mountain_type=self.params.get('mountain_type', 'ultra_realistic'),
            preset=preset_name,
            scale=1.0,  # V2 uses normalized scale
            octaves=max(12, octaves),  # V2 needs higher octaves for quality
            lacunarity=lacunarity,
            gain=persistence,  # V2 uses 'gain' instead of 'persistence'
            warp_strength=self.params.get('warp_strength', 0.5),
            erosion_strength=0.7,
            apply_hydraulic_erosion=apply_hydraulic,
            apply_thermal_erosion=apply_thermal,
            erosion_iterations=erosion_iters if not preset_name else None,  # Auto if preset
            seed=seed
        )

        self.progress.emit(40, "Génération normal map...")
        normal_map = terrain_gen.generate_normal_map(
            heightmap,
            strength=self.params.get('normal_strength', 1.0)
        )

        self.progress.emit(55, "Génération depth map...")
        depth_map = terrain_gen.generate_depth_map(heightmap)

        self.progress.emit(65, "Génération AO...")
        ao_map = terrain_gen.generate_ambient_occlusion(
            heightmap,
            samples=16,
            radius=10
        )

        # Get vegetation parameters (from preset or UI)
        if preset is not None:
            veg_cfg = preset.get('vegetation', {})
            generate_veg = veg_cfg.get('enabled', False) or self.params.get('generate_vegetation', False)
            veg_density = veg_cfg.get('density', 0.5)
            veg_spacing = veg_cfg.get('min_spacing', 3.0)
            veg_clustering = veg_cfg.get('use_clustering', True)
        else:
            generate_veg = self.params.get('generate_vegetation', False)
            veg_density = self.params.get('vegetation_density', 0.5)
            veg_spacing = self.params.get('vegetation_spacing', 3.0)
            veg_clustering = self.params.get('use_clustering', True)

        # Biome classification
        biome_map = None
        if generate_veg or self.params.get('generate_biomes', False):
            self.progress.emit(75, "Classification biomes...")
            biome_classifier = BiomeClassifier(
                width=resolution,
                height=resolution
            )
            biome_map = biome_classifier.classify(heightmap)

        # Vegetation placement
        tree_instances = None
        density_map = None
        if generate_veg and biome_map is not None:
            self.progress.emit(85, "Placement végétation...")
            placer = VegetationPlacer(
                resolution,
                resolution,
                heightmap,
                biome_map
            )
            tree_instances = placer.place_vegetation(
                density=veg_density,
                min_spacing=veg_spacing,
                use_clustering=veg_clustering
            )
            density_map = placer.generate_density_map()

        # PBR Splatmap generation
        splatmaps = None
        if self.params.get('generate_splatmaps', True):  # Generate by default
            self.progress.emit(92, "Génération PBR splatmaps...")
            splatmap_gen = PBRSplatmapGenerator(resolution, resolution)

            # Generate splatmap from heightmap only (moisture auto-generated)
            splatmap1, splatmap2 = splatmap_gen.generate_splatmap(
                heightmap,
                moisture_map=None,  # Auto-generated internally
                custom_materials=None,  # Use default materials
                apply_weathering=True,
                smooth_transitions=True
            )

            splatmaps = [splatmap1, splatmap2]

        self.progress.emit(100, "Terminé!")

        result = {
            'heightmap': heightmap,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'ao_map': ao_map,
            'biome_map': biome_map,
            'tree_instances': tree_instances,
            'density_map': density_map,
            'splatmaps': splatmaps,
            'terrain_gen': terrain_gen
        }

        self.finished.emit(result, 'terrain')

    def generate_texture(self):
        """Generate AI texture using Stable Diffusion via ComfyUI"""
        try:
            # Get parameters
            prompt = self.params.get('texture_prompt', 'photorealistic mountain terrain, high detail, 4K')
            width = self.params.get('texture_width', 1024)
            height = self.params.get('texture_height', 1024)
            server_address = self.params.get('comfyui_server', '127.0.0.1:8188')
            seed = self.params.get('seed', -1)

            self.progress.emit(10, "Connecting to ComfyUI...")

            # Generate PBR textures
            self.progress.emit(30, "Generating textures with AI...")
            logger.info(f"Generating texture: '{prompt}'")

            textures = generate_pbr_textures(
                prompt=prompt,
                width=width,
                height=height,
                server_address=server_address,
                seed=seed
            )

            if textures is None:
                raise Exception("Failed to generate textures from ComfyUI")

            self.progress.emit(90, "Processing results...")

            # Store results
            result = {
                'diffuse': textures.get('diffuse'),
                'normal': textures.get('normal'),
                'roughness': textures.get('roughness'),
                'ao': textures.get('ao'),
                'height': textures.get('height'),
                'prompt': prompt
            }

            self.progress.emit(100, "Texture generation complete!")
            logger.info("✓ Texture generation successful")

            self.finished.emit(result, 'texture')

        except Exception as e:
            logger.error(f"Texture generation failed: {e}")
            self.progress.emit(0, f"Error: {str(e)}")
            self.finished.emit(None, 'texture')

    def generate_video(self):
        """Generate coherent video using VideoCoherenceManager"""
        try:
            # Get parameters
            num_frames = self.params.get('video_frames', 120)  # 4 seconds @ 30fps
            fps = self.params.get('video_fps', 30)
            movement_type = self.params.get('video_movement', 'orbit')  # orbit, flyover, pan, zoom
            prompt = self.params.get('video_prompt', 'cinematic mountain landscape flyover')
            resolution = self.params.get('video_resolution', 512)
            server_address = self.params.get('comfyui_server', '127.0.0.1:8188')

            self.progress.emit(5, "Initializing video generation...")

            # Create video manager
            video_manager = VideoCoherenceManager(
                width=resolution,
                height=resolution,
                comfyui_server=server_address
            )

            self.progress.emit(15, f"Generating {num_frames} frames...")

            # Generate frames with coherence
            logger.info(f"Generating video: {num_frames} frames, {movement_type} movement")

            frames = []
            if movement_type == 'orbit':
                # Orbital camera movement around terrain
                frames = video_manager.generate_orbit_sequence(
                    base_prompt=prompt,
                    num_frames=num_frames,
                    seed=self.params.get('seed', 42)
                )
            elif movement_type == 'flyover':
                # Flyover movement
                frames = video_manager.generate_flyover_sequence(
                    base_prompt=prompt,
                    num_frames=num_frames,
                    seed=self.params.get('seed', 42)
                )
            else:
                # Default: use simple camera params
                camera_params = [
                    {'angle': i * 360 / num_frames} for i in range(num_frames)
                ]
                frames = video_manager.generate_coherent_sequence(
                    base_prompt=prompt,
                    num_frames=num_frames,
                    camera_params=camera_params,
                    seed=self.params.get('seed', 42)
                )

            if not frames or len(frames) == 0:
                raise Exception("No frames generated")

            self.progress.emit(85, "Encoding video...")

            # Create video generator
            from video_generator import VideoGenerator
            video_gen = VideoGenerator(width=resolution, height=resolution, fps=fps)

            # Add all frames
            for idx, frame in enumerate(frames):
                if (idx % 10) == 0:
                    progress = 85 + int((idx / len(frames)) * 10)
                    self.progress.emit(progress, f"Encoding frame {idx+1}/{len(frames)}...")
                video_gen.add_frame(frame)

            # Save video
            output_path = self.params.get('video_output', 'output/mountain_video.mp4')
            video_gen.save(output_path)

            self.progress.emit(100, "Video generation complete!")
            logger.info(f"✓ Video saved: {output_path}")

            result = {
                'video_path': output_path,
                'frames': frames,
                'num_frames': len(frames),
                'fps': fps,
                'movement': movement_type
            }

            self.finished.emit(result, 'video')

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            self.progress.emit(0, f"Error: {str(e)}")
            self.finished.emit(None, 'video')


class MountainProUI(QMainWindow):
    """Interface professionnelle principale"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mountain Studio Pro - Outil Professionnel pour Graphistes")
        self.setGeometry(100, 100, 1600, 900)

        # État - Terrain data
        self.current_terrain = None
        self.current_heightmap = None
        self.current_normal_map = None
        self.current_depth_map = None
        self.current_ao_map = None
        self.current_biome_map = None
        self.current_tree_instances = None
        self.current_density_map = None
        self.current_splatmaps = None
        self.current_vfx_prompt = None
        self.current_texture = None
        self.generation_thread = None
        self.generation_metadata = {}

        # Backends
        self.comfyui = None
        self.sd_direct = None
        self.video_manager = None

        # 3D View state
        self.terrain_surface = None
        self.wireframe_mode = False

        self.init_ui()

    def init_ui(self):
        """Initialise l'interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # Panel gauche - Contrôles
        left_panel = self.create_controls_panel()
        splitter.addWidget(left_panel)

        # Panel central - Visualisation 3D
        center_panel = self.create_3d_view_panel()
        splitter.addWidget(center_panel)

        # Panel droit - Preview & Export
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800, 400])
        main_layout.addWidget(splitter)

    def create_controls_panel(self):
        """Panel de contrôles à gauche"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Titre
        title = QLabel("<h2>⛰️ Mountain Studio Pro</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Tabs pour organisation
        tabs = QTabWidget()

        # Tab 1: Terrain
        terrain_tab = self.create_terrain_controls()
        tabs.addTab(terrain_tab, "🗻 Terrain")

        # Tab 2: Vegetation (NEW)
        vegetation_tab = self.create_vegetation_controls()
        tabs.addTab(vegetation_tab, "🌲 Végétation")

        # Tab 3: Texture AI
        texture_tab = self.create_texture_controls()
        tabs.addTab(texture_tab, "🎨 Texture AI")

        # Tab 4: Caméra & Rendu
        camera_tab = self.create_camera_controls()
        tabs.addTab(camera_tab, "🎥 Caméra")

        # Tab 5: Export
        export_tab = self.create_export_controls()
        tabs.addTab(export_tab, "💾 Export")

        layout.addWidget(tabs)

        # Status bar en bas
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Prêt")
        layout.addWidget(self.status_label)

        return panel

    def create_terrain_controls(self):
        """Contrôles de génération de terrain"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Type de montagne
        group = QGroupBox("Type de Montagne")
        group_layout = QVBoxLayout()
        self.mountain_type_combo = QComboBox()
        self.mountain_type_combo.addItems([
            "Alpine", "Volcanic", "Rolling", "Massive", "Rocky"
        ])
        group_layout.addWidget(self.mountain_type_combo)
        group.setLayout(group_layout)
        layout.addWidget(group)

        # Paramètres génération
        params_group = QGroupBox("Paramètres de Génération")
        params_layout = QVBoxLayout()

        # Résolution
        params_layout.addWidget(QLabel("Résolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["512", "1024", "2048", "4096"])
        self.resolution_combo.setCurrentText("2048")
        params_layout.addWidget(self.resolution_combo)

        # Scale
        params_layout.addWidget(QLabel("Scale (détail):"))
        self.scale_slider = self.create_slider_with_value(10, 200, 100)
        params_layout.addLayout(self.scale_slider['layout'])

        # Octaves
        params_layout.addWidget(QLabel("Octaves (complexité):"))
        self.octaves_slider = self.create_slider_with_value(1, 12, 8)
        params_layout.addLayout(self.octaves_slider['layout'])

        # Persistence
        params_layout.addWidget(QLabel("Persistence:"))
        self.persistence_slider = self.create_slider_with_value(10, 90, 50, scale=0.01)
        params_layout.addLayout(self.persistence_slider['layout'])

        # Lacunarity
        params_layout.addWidget(QLabel("Lacunarity:"))
        self.lacunarity_slider = self.create_slider_with_value(10, 40, 20, scale=0.1)
        params_layout.addLayout(self.lacunarity_slider['layout'])

        # Normal strength
        params_layout.addWidget(QLabel("Normal Map Strength:"))
        self.normal_strength_slider = self.create_slider_with_value(5, 30, 10, scale=0.1)
        params_layout.addLayout(self.normal_strength_slider['layout'])

        # Seed
        params_layout.addWidget(QLabel("Seed:"))
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999)
        self.seed_spinbox.setValue(42)
        params_layout.addWidget(self.seed_spinbox)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Presets (NEW)
        preset_group = QGroupBox("⚡ Presets Professionnels")
        preset_layout = QVBoxLayout()

        self.use_preset_checkbox = QCheckBox("Utiliser un preset")
        self.use_preset_checkbox.stateChanged.connect(self.toggle_preset_mode)
        preset_layout.addWidget(self.use_preset_checkbox)

        self.preset_combo = QComboBox()
        preset_mgr = PresetManager()
        for preset_name in preset_mgr.list_presets():
            self.preset_combo.addItem(preset_name)
        self.preset_combo.setEnabled(False)
        preset_layout.addWidget(self.preset_combo)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Érosion avancée (NEW)
        erosion_group = QGroupBox("🌊 Érosion Avancée")
        erosion_layout = QVBoxLayout()

        # Hydraulic erosion
        self.hydraulic_checkbox = QCheckBox("Érosion Hydraulique")
        self.hydraulic_checkbox.setChecked(True)
        erosion_layout.addWidget(self.hydraulic_checkbox)

        erosion_layout.addWidget(QLabel("Itérations hydrauliques:"))
        self.hydraulic_iter_slider = self.create_slider_with_value(10, 200, 50)
        erosion_layout.addLayout(self.hydraulic_iter_slider['layout'])

        erosion_layout.addWidget(QLabel("Quantité pluie:"))
        self.rain_slider = self.create_slider_with_value(1, 20, 10, scale=0.001)
        erosion_layout.addLayout(self.rain_slider['layout'])

        # Thermal erosion
        self.thermal_checkbox = QCheckBox("Érosion Thermique")
        self.thermal_checkbox.setChecked(True)
        erosion_layout.addWidget(self.thermal_checkbox)

        erosion_layout.addWidget(QLabel("Itérations thermiques:"))
        self.thermal_iter_slider = self.create_slider_with_value(10, 100, 30)
        erosion_layout.addLayout(self.thermal_iter_slider['layout'])

        erosion_group.setLayout(erosion_layout)
        layout.addWidget(erosion_group)

        # Bouton génération
        self.generate_terrain_btn = QPushButton("🗻 Générer Terrain 3D (avec Érosion)")
        self.generate_terrain_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-weight: bold; }")
        self.generate_terrain_btn.clicked.connect(self.generate_terrain)
        layout.addWidget(self.generate_terrain_btn)

        layout.addStretch()

        return widget

    def create_vegetation_controls(self):
        """Contrôles de végétation procédurale (NEW)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Enable vegetation
        self.vegetation_enabled_checkbox = QCheckBox("Activer la génération de végétation")
        self.vegetation_enabled_checkbox.setChecked(False)
        self.vegetation_enabled_checkbox.stateChanged.connect(self.toggle_vegetation_controls)
        layout.addWidget(self.vegetation_enabled_checkbox)

        # Biome classification
        biome_group = QGroupBox("🏔️ Classification Biomes")
        biome_layout = QVBoxLayout()

        self.biome_checkbox = QCheckBox("Classifier les biomes automatiquement")
        self.biome_checkbox.setChecked(True)
        self.biome_checkbox.setEnabled(False)
        biome_layout.addWidget(self.biome_checkbox)

        biome_info = QLabel("Les biomes déterminent le type de végétation:\n" +
                           "• Alpine Tundra (haute altitude)\n" +
                           "• Subalpine (pins épars)\n" +
                           "• Montane Forest (forêt dense)\n" +
                           "• Valley Floor (vallées)\n" +
                           "• Water (lacs)")
        biome_info.setStyleSheet("color: #888; font-size: 9pt;")
        biome_layout.addWidget(biome_info)

        biome_group.setLayout(biome_layout)
        layout.addWidget(biome_group)

        # Vegetation placement
        veg_group = QGroupBox("🌲 Placement Végétation")
        veg_layout = QVBoxLayout()

        veg_layout.addWidget(QLabel("Densité globale:"))
        self.veg_density_slider = self.create_slider_with_value(10, 100, 50, scale=0.01)
        veg_layout.addLayout(self.veg_density_slider['layout'])

        veg_layout.addWidget(QLabel("Espacement minimum (m):"))
        self.veg_spacing_slider = self.create_slider_with_value(1, 10, 3, scale=0.1)
        veg_layout.addLayout(self.veg_spacing_slider['layout'])

        self.veg_clustering_checkbox = QCheckBox("Activer le clustering (groupes d'arbres)")
        self.veg_clustering_checkbox.setChecked(True)
        veg_layout.addWidget(self.veg_clustering_checkbox)

        veg_group.setLayout(veg_layout)
        veg_group.setEnabled(False)
        layout.addWidget(veg_group)

        # Store reference for enabling/disabling
        self.veg_controls_group = veg_group
        self.biome_controls_group = biome_group

        layout.addStretch()

        return widget

    def create_texture_controls(self):
        """Contrôles de texture AI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Backend
        backend_group = QGroupBox("Backend AI")
        backend_layout = QVBoxLayout()

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Stable Diffusion XL", "ComfyUI"])
        backend_layout.addWidget(self.backend_combo)

        self.comfyui_address = QLineEdit("127.0.0.1:8188")
        backend_layout.addWidget(QLabel("Adresse ComfyUI:"))
        backend_layout.addWidget(self.comfyui_address)

        self.init_backend_btn = QPushButton("🚀 Initialiser Backend")
        self.init_backend_btn.clicked.connect(self.initialize_backend)
        backend_layout.addWidget(self.init_backend_btn)

        backend_group.setLayout(backend_layout)
        layout.addWidget(backend_group)

        # Prompt
        prompt_group = QGroupBox("Prompt de Texture")
        prompt_layout = QVBoxLayout()

        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText("Décrivez la texture souhaitée...")
        self.prompt_text.setMaximumHeight(100)
        prompt_layout.addWidget(self.prompt_text)

        self.negative_prompt_text = QTextEdit()
        self.negative_prompt_text.setPlaceholderText("Negative prompt...")
        self.negative_prompt_text.setMaximumHeight(80)
        prompt_layout.addWidget(QLabel("Negative Prompt:"))
        prompt_layout.addWidget(self.negative_prompt_text)

        # Auto-generate prompt
        self.auto_prompt_btn = QPushButton("✨ Auto-générer Prompt")
        self.auto_prompt_btn.clicked.connect(self.auto_generate_prompt)
        prompt_layout.addWidget(self.auto_prompt_btn)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # Paramètres génération
        gen_group = QGroupBox("Paramètres Génération")
        gen_layout = QVBoxLayout()

        gen_layout.addWidget(QLabel("Steps:"))
        self.steps_slider = self.create_slider_with_value(20, 100, 40)
        gen_layout.addLayout(self.steps_slider['layout'])

        gen_layout.addWidget(QLabel("Detail Level:"))
        self.detail_slider = self.create_slider_with_value(0, 100, 85)
        gen_layout.addLayout(self.detail_slider['layout'])

        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)

        # Bouton génération texture
        self.generate_texture_btn = QPushButton("🎨 Générer Texture AI")
        self.generate_texture_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 10px; font-weight: bold; }")
        self.generate_texture_btn.clicked.connect(self.generate_texture)
        layout.addWidget(self.generate_texture_btn)

        layout.addStretch()

        return widget

    def create_camera_controls(self):
        """Contrôles caméra et vidéo"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Caméra
        camera_group = QGroupBox("Position Caméra")
        camera_layout = QVBoxLayout()

        camera_layout.addWidget(QLabel("Angle Horizontal:"))
        self.h_angle_slider = self.create_slider_with_value(-180, 180, 0)
        camera_layout.addLayout(self.h_angle_slider['layout'])

        camera_layout.addWidget(QLabel("Angle Vertical:"))
        self.v_angle_slider = self.create_slider_with_value(-90, 90, 15)
        camera_layout.addLayout(self.v_angle_slider['layout'])

        camera_layout.addWidget(QLabel("Focale (mm):"))
        self.focal_slider = self.create_slider_with_value(24, 200, 50)
        camera_layout.addLayout(self.focal_slider['layout'])

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Vidéo
        video_group = QGroupBox("Génération Vidéo Cohérente")
        video_layout = QVBoxLayout()

        video_layout.addWidget(QLabel("Nombre de Frames:"))
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(3, 60)
        self.num_frames_spin.setValue(12)
        video_layout.addWidget(self.num_frames_spin)

        video_layout.addWidget(QLabel("Type de Mouvement:"))
        self.movement_combo = QComboBox()
        self.movement_combo.addItems(["Orbit", "Pan", "Zoom", "Flyover", "Static"])
        video_layout.addWidget(self.movement_combo)

        video_layout.addWidget(QLabel("Strength (cohérence):"))
        self.video_strength_slider = self.create_slider_with_value(10, 50, 25, scale=0.01)
        video_layout.addLayout(self.video_strength_slider['layout'])

        self.interpolate_checkbox = QCheckBox("Interpolation entre frames")
        self.interpolate_checkbox.setChecked(True)
        video_layout.addWidget(self.interpolate_checkbox)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Bouton génération vidéo
        self.generate_video_btn = QPushButton("🎬 Générer Vidéo Cohérente")
        self.generate_video_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 10px; font-weight: bold; }")
        self.generate_video_btn.clicked.connect(self.generate_video)
        layout.addWidget(self.generate_video_btn)

        layout.addStretch()

        return widget

    def create_export_controls(self):
        """Contrôles d'export"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        export_group = QGroupBox("Export Professionnel")
        export_layout = QVBoxLayout()

        export_layout.addWidget(QLabel("Sélectionnez les maps à exporter:"))

        self.export_heightmap_cb = QCheckBox("Heightmap (.EXR 32-bit)")
        self.export_heightmap_cb.setChecked(True)
        export_layout.addWidget(self.export_heightmap_cb)

        self.export_normal_cb = QCheckBox("Normal Map")
        self.export_normal_cb.setChecked(True)
        export_layout.addWidget(self.export_normal_cb)

        self.export_depth_cb = QCheckBox("Depth Map (Z-Depth)")
        self.export_depth_cb.setChecked(True)
        export_layout.addWidget(self.export_depth_cb)

        self.export_ao_cb = QCheckBox("Ambient Occlusion")
        self.export_ao_cb.setChecked(True)
        export_layout.addWidget(self.export_ao_cb)

        self.export_roughness_cb = QCheckBox("Roughness Map")
        self.export_roughness_cb.setChecked(True)
        export_layout.addWidget(self.export_roughness_cb)

        self.export_texture_cb = QCheckBox("Texture AI (si générée)")
        self.export_texture_cb.setChecked(True)
        export_layout.addWidget(self.export_texture_cb)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Format
        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout()

        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG", "EXR", "TIFF", "Tous"])
        self.export_format_combo.setCurrentText("Tous")
        format_layout.addWidget(self.export_format_combo)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Boutons export
        self.export_all_btn = QPushButton("💾 Exporter Toutes les Maps")
        self.export_all_btn.clicked.connect(self.export_all_maps)
        layout.addWidget(self.export_all_btn)

        self.export_mesh_btn = QPushButton("📐 Exporter Mesh 3D (.OBJ)")
        self.export_mesh_btn.clicked.connect(self.export_mesh)
        layout.addWidget(self.export_mesh_btn)

        self.export_flame_btn = QPushButton("🔥 Export pour Autodesk Flame (OBJ+MTL+Textures)")
        self.export_flame_btn.setStyleSheet("QPushButton { background-color: #E91E63; color: white; padding: 10px; font-weight: bold; }")
        self.export_flame_btn.clicked.connect(self.export_for_flame)
        layout.addWidget(self.export_flame_btn)

        layout.addStretch()

        # Log
        layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        return widget

    def create_3d_view_panel(self):
        """Vue 3D centrale"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h3>Vue 3D Interactive</h3>"))

        # Vue 3D avec pyqtgraph.opengl
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=40)
        layout.addWidget(self.gl_view)

        # Contrôles de vue
        view_controls = QHBoxLayout()
        reset_view_btn = QPushButton("Reset Vue")
        reset_view_btn.clicked.connect(self.reset_3d_view)
        view_controls.addWidget(reset_view_btn)

        wireframe_btn = QPushButton("Toggle Wireframe")
        wireframe_btn.clicked.connect(self.toggle_wireframe)
        view_controls.addWidget(wireframe_btn)

        view_controls.addStretch()
        layout.addLayout(view_controls)

        return widget

    def create_preview_panel(self):
        """Panel de preview à droite"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h3>Preview Maps</h3>"))

        # Tabs pour différentes maps
        preview_tabs = QTabWidget()

        # Tab Heightmap
        self.heightmap_preview = QLabel()
        self.heightmap_preview.setAlignment(Qt.AlignCenter)
        self.heightmap_preview.setMinimumSize(350, 350)
        self.heightmap_preview.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        preview_tabs.addTab(self.heightmap_preview, "Heightmap")

        # Tab Normal
        self.normal_preview = QLabel()
        self.normal_preview.setAlignment(Qt.AlignCenter)
        self.normal_preview.setMinimumSize(350, 350)
        self.normal_preview.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        preview_tabs.addTab(self.normal_preview, "Normal")

        # Tab Depth
        self.depth_preview = QLabel()
        self.depth_preview.setAlignment(Qt.AlignCenter)
        self.depth_preview.setMinimumSize(350, 350)
        self.depth_preview.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        preview_tabs.addTab(self.depth_preview, "Depth")

        # Tab Texture
        self.texture_preview = QLabel()
        self.texture_preview.setAlignment(Qt.AlignCenter)
        self.texture_preview.setMinimumSize(350, 350)
        self.texture_preview.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        preview_tabs.addTab(self.texture_preview, "Texture AI")

        layout.addWidget(preview_tabs)

        return widget

    def create_slider_with_value(self, min_val, max_val, default, scale=1.0):
        """Crée un slider avec affichage de valeur"""
        layout = QHBoxLayout()

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val / scale), int(max_val / scale))
        slider.setValue(int(default / scale))

        value_label = QLabel(str(default))

        def update_label(val):
            value_label.setText(f"{val * scale:.2f}")

        slider.valueChanged.connect(update_label)

        layout.addWidget(slider)
        layout.addWidget(value_label)

        return {'layout': layout, 'slider': slider, 'label': value_label, 'scale': scale}

    # =========== SLOTS ===========

    def generate_terrain(self):
        """Lance la génération de terrain avec nouvelles features"""
        self.log("🗻 Génération du terrain avec érosion avancée...")

        params = {
            # Base parameters
            'resolution': int(self.resolution_combo.currentText()),
            'scale': self.scale_slider['slider'].value() * self.scale_slider['scale'],
            'octaves': self.octaves_slider['slider'].value(),
            'persistence': self.persistence_slider['slider'].value() * self.persistence_slider['scale'],
            'lacunarity': self.lacunarity_slider['slider'].value() * self.lacunarity_slider['scale'],
            'mountain_type': self.mountain_type_combo.currentText().lower(),
            'normal_strength': self.normal_strength_slider['slider'].value() * self.normal_strength_slider['scale'],
            'seed': self.seed_spinbox.value(),

            # Preset parameters (NEW)
            'use_preset': self.use_preset_checkbox.isChecked(),
            'preset_name': self.preset_combo.currentText() if self.use_preset_checkbox.isChecked() else None,

            # Erosion parameters (NEW)
            'hydraulic_enabled': self.hydraulic_checkbox.isChecked(),
            'hydraulic_iterations': self.hydraulic_iter_slider['slider'].value(),
            'rain_amount': self.rain_slider['slider'].value() * self.rain_slider['scale'],
            'evaporation': 0.5,  # Default
            'thermal_enabled': self.thermal_checkbox.isChecked(),
            'thermal_iterations': self.thermal_iter_slider['slider'].value(),
            'talus_angle': 0.7,  # Default

            # Vegetation parameters (NEW)
            'generate_biomes': self.vegetation_enabled_checkbox.isChecked(),
            'generate_vegetation': self.vegetation_enabled_checkbox.isChecked(),
            'vegetation_density': self.veg_density_slider['slider'].value() * self.veg_density_slider['scale'] if self.vegetation_enabled_checkbox.isChecked() else 0.5,
            'vegetation_spacing': self.veg_spacing_slider['slider'].value() * self.veg_spacing_slider['scale'] if self.vegetation_enabled_checkbox.isChecked() else 3.0,
            'use_clustering': self.veg_clustering_checkbox.isChecked() if self.vegetation_enabled_checkbox.isChecked() else True
        }

        self.generation_thread = GenerationThread("terrain", params)
        self.generation_thread.progress.connect(self.update_progress)
        self.generation_thread.finished.connect(self.on_terrain_generated)
        self.generation_thread.error.connect(self.on_error)
        self.generation_thread.start()

        self.generate_terrain_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

    def on_terrain_generated(self, result, result_type):
        """Callback terrain généré avec nouvelles features"""
        # Store all results
        self.current_terrain = result['terrain_gen']
        self.current_heightmap = result['heightmap']
        self.current_normal_map = result.get('normal_map')
        self.current_depth_map = result.get('depth_map')
        self.current_ao_map = result.get('ao_map')
        self.current_biome_map = result.get('biome_map')
        self.current_tree_instances = result.get('tree_instances')
        self.current_density_map = result.get('density_map')
        self.current_splatmaps = result.get('splatmaps')

        self.log("✓ Terrain généré avec succès!")

        # Afficher les previews
        self.display_preview(result['heightmap'], self.heightmap_preview)
        self.display_preview(result['normal_map'], self.normal_preview)
        self.display_preview(result['depth_map'], self.depth_preview)

        # Display biome map if available
        if result.get('biome_map') is not None:
            self.log(f"✓ Biomes classifiés")

        # Display vegetation stats if available
        if result.get('tree_instances') is not None:
            num_trees = len(result['tree_instances'])
            self.log(f"✓ {num_trees} arbres placés")

            # Count species
            species_counts = {}
            for tree in result['tree_instances']:
                species_counts[tree.species] = species_counts.get(tree.species, 0) + 1

            for species, count in species_counts.items():
                self.log(f"  → {species}: {count} arbres")

        # Afficher en 3D
        self.display_3d_mesh(result['heightmap'])

        self.generate_terrain_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def display_preview(self, data, label_widget):
        """Affiche une preview"""
        if data is None:
            return

        if len(data.shape) == 2:
            # Grayscale
            if data.dtype == np.float64 or data.dtype == np.float32:
                data = (data * 255).astype(np.uint8)
            h, w = data.shape
            qimage = QImage(data.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # RGB
            h, w, c = data.shape
            qimage = QImage(data.data, w, h, w * c, QImage.Format_RGB888)

        # Resize pour preview
        pixmap = QPixmap.fromImage(qimage).scaled(
            350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label_widget.setPixmap(pixmap)

    def display_3d_mesh(self, heightmap):
        """Affiche le mesh 3D"""
        self.gl_view.clear()

        # Sous-échantillonner pour performance
        step = max(1, heightmap.shape[0] // 200)
        h_sub = heightmap[::step, ::step]

        # Créer mesh
        z = h_sub * 20  # Amplifier la hauteur pour visualisation
        self.terrain_surface = gl.GLSurfacePlotItem(
            z=z,
            shader='heightColor',
            computeNormals=True,
            smooth=True
        )
        self.terrain_surface.scale(1, 1, 1)
        self.terrain_surface.translate(-h_sub.shape[0]/2, -h_sub.shape[1]/2, 0)

        # Apply wireframe mode if enabled
        if self.wireframe_mode:
            self.terrain_surface.setGLOptions('additive')
            self.terrain_surface.shader().setUniformValue('drawEdges', True)

        self.gl_view.addItem(self.terrain_surface)

        # Grille
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.gl_view.addItem(grid)

    def reset_3d_view(self):
        """Reset la vue 3D"""
        self.gl_view.setCameraPosition(distance=40, elevation=30, azimuth=45)

    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        if self.terrain_surface is None:
            return

        self.wireframe_mode = not self.wireframe_mode

        # For pyqtgraph GLSurfacePlotItem, we need to set drawMode
        # 'lines' for wireframe, 'triangles' for solid
        if hasattr(self.terrain_surface, 'opts'):
            if self.wireframe_mode:
                # Enable wireframe
                self.terrain_surface.opts['drawEdges'] = True
                self.terrain_surface.opts['drawFaces'] = False
            else:
                # Disable wireframe (solid mode)
                self.terrain_surface.opts['drawEdges'] = False
                self.terrain_surface.opts['drawFaces'] = True

            # Force update
            self.terrain_surface.meshDataChanged()

        logger.info(f"Wireframe mode: {'ON' if self.wireframe_mode else 'OFF'}")

    def toggle_preset_mode(self, state):
        """Toggle preset mode ON/OFF"""
        enabled = (state == Qt.CheckState.Checked.value)
        self.preset_combo.setEnabled(enabled)

        # Disable/enable manual controls when using preset
        controls = [
            self.scale_slider,
            self.octaves_slider,
            self.persistence_slider,
            self.lacunarity_slider,
            self.hydraulic_iter_slider,
            self.rain_slider,
            self.thermal_iter_slider
        ]

        for control in controls:
            control['slider'].setEnabled(not enabled)

    def toggle_vegetation_controls(self, state):
        """Toggle vegetation controls ON/OFF"""
        enabled = (state == Qt.CheckState.Checked.value)
        self.veg_controls_group.setEnabled(enabled)
        self.biome_controls_group.setEnabled(enabled)
        self.biome_checkbox.setEnabled(enabled)

    def initialize_backend(self):
        """Initialise le backend AI"""
        backend = self.backend_combo.currentText()
        self.log(f"🚀 Initialisation {backend}...")

        if backend == "ComfyUI":
            address = self.comfyui_address.text()
            self.comfyui = ComfyUIIntegration(address)
            if self.comfyui.test_connection():
                self.log("✓ ComfyUI connecté!")
            else:
                self.log("❌ Échec connexion ComfyUI")
        else:
            self.sd_direct = StableDiffusionDirect()
            if self.sd_direct.load_model():
                self.log("✓ Stable Diffusion chargé!")
            else:
                self.log("❌ Échec chargement SD")

    def auto_generate_prompt(self):
        """Auto-génère un prompt basé sur les paramètres"""
        prompt_gen = MountainPromptGenerator()

        params = {
            'mountain_type': self.mountain_type_combo.currentText().lower(),
            'mountain_height': 75,
            'tree_density': 60,
            'tree_type': 'pine',
            'sky_type': 'dramatic',
            'lighting': 'golden',
            'weather': 'clear',
            'season': 'summer',
            'camera_desc': 'professional photography, cinematic view'
        }

        prompt, negative = prompt_gen.generate_prompt(params)
        self.prompt_text.setText(prompt)
        self.negative_prompt_text.setText(negative)

        self.log("✨ Prompt auto-généré")

    def generate_texture(self):
        """Génère une texture AI ultra-réaliste avec ComfyUI"""
        if self.current_heightmap is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        self.log("🎨 Génération texture AI avec ComfyUI...")

        # Generate VFX prompt
        vfx_gen = VFXPromptGenerator()

        # Analyze terrain characteristics
        elevation_stats = {
            'mean': float(np.mean(self.current_heightmap)),
            'std': float(np.std(self.current_heightmap)),
            'min': float(np.min(self.current_heightmap)),
            'max': float(np.max(self.current_heightmap))
        }

        # Generate prompt using auto-generation from heightmap
        prompt_result = vfx_gen.auto_generate_from_heightmap(
            self.current_heightmap,
            biome_map=self.current_biome_map,
            vegetation_density_map=self.current_density_map,
            time_of_day='sunset' if elevation_stats['mean'] > 0.5 else 'golden_hour',
            weather='clear',
            season='summer'
        )

        # Store prompt
        self.current_vfx_prompt = prompt_result

        # Try to generate AI texture with ComfyUI
        self.log("🤖 Connexion à ComfyUI pour génération AI...")

        try:
            # Generate landscape image with AI
            landscape = generate_landscape_image(
                self.current_heightmap,
                prompt=prompt_result['positive'],
                style=style,
                server_address="127.0.0.1:8188",
                seed=-1
            )

            if landscape is not None:
                self.log("✓ Texture AI générée avec succès!")

                # Display result
                dialog = QMessageBox(self)
                dialog.setWindowTitle("Texture AI Générée")
                dialog.setText("Texture ultra-réaliste générée avec ComfyUI!")
                dialog.setInformativeText("La texture a été générée en utilisant l'IA. Voulez-vous la sauvegarder?")
                dialog.setStandardButtons(QMessageBox.Save | QMessageBox.Close)

                if dialog.exec() == QMessageBox.Save:
                    filepath, _ = QFileDialog.getSaveFileName(
                        self,
                        "Sauvegarder Texture AI",
                        "ai_texture.png",
                        "PNG Images (*.png);;All Files (*)"
                    )
                    if filepath:
                        Image.fromarray(landscape).save(filepath)
                        self.log(f"✓ Texture AI sauvegardée: {filepath}")

            else:
                raise Exception("ComfyUI not available")

        except Exception as e:
            self.log(f"⚠ ComfyUI non disponible ({e}), affichage du prompt...")

            # Fallback: show prompt for manual use
            prompt_text = f"""PROMPT POSITIF:
{prompt_result['positive']}

PROMPT NÉGATIF:
{prompt_result['negative']}

PARAMÈTRES RECOMMANDÉS:
{prompt_result['metadata']['recommended_model']} - {prompt_result['metadata']['steps']} steps, CFG {prompt_result['metadata']['cfg_scale']}

NOTE: ComfyUI n'est pas disponible. Lancez ComfyUI sur http://127.0.0.1:8188
pour la génération automatique, ou utilisez ce prompt manuellement.
"""

            dialog = QMessageBox(self)
            dialog.setWindowTitle("Prompt VFX (ComfyUI non disponible)")
            dialog.setText("ComfyUI non détecté - Prompt généré pour usage manuel")
            dialog.setDetailedText(prompt_text)
            dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Save)

            result = dialog.exec()

            if result == QMessageBox.Save:
                filepath, _ = QFileDialog.getSaveFileName(
                    self,
                    "Sauvegarder Prompt",
                    "vfx_prompt.txt",
                    "Text Files (*.txt);;All Files (*)"
                )
                if filepath:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(prompt_text)
                    self.log(f"✓ Prompt sauvegardé: {filepath}")

        self.log("✓ Génération texture terminée")

    def generate_video(self):
        """Génère une vidéo cohérente"""
        self.log("🎬 Génération vidéo cohérente...")
        QMessageBox.information(self, "Info", "Fonction en développement")

    def export_all_maps(self):
        """Exporte toutes les maps avec ProfessionalExporter"""
        if self.current_heightmap is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        folder = QFileDialog.getExistingDirectory(self, "Choisir dossier d'export")
        if folder:
            self.log("📦 Export en cours...")

            # Create exporter
            exporter = ProfessionalExporter(folder)

            # Export complete package
            exported_files = exporter.export_complete_package(
                heightmap=self.current_heightmap,
                normal_map=self.current_normal_map,
                depth_map=self.current_depth_map,
                ao_map=self.current_ao_map,
                splatmaps=self.current_splatmaps,
                tree_instances=self.current_tree_instances,
                vfx_prompt=self.current_vfx_prompt,
                metadata=self.generation_metadata,
                export_mesh=True,
                mesh_subsample=2  # Subsample for performance
            )

            num_files = len(exported_files)
            self.log(f"✓ {num_files} fichiers exportés vers: {folder}")

            # Show summary
            file_list = "\n".join([f"• {Path(f).name}" for f in exported_files.values()])
            QMessageBox.information(
                self,
                "Export Réussi",
                f"Export terminé: {num_files} fichiers\n\n{file_list}"
            )

    def export_mesh(self):
        """Exporte le mesh 3D en OBJ"""
        if self.current_heightmap is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter Mesh 3D",
            "",
            "OBJ Files (*.obj);;All Files (*)"
        )

        if filepath:
            self.log("🎨 Export mesh 3D en cours...")

            # Use temporary folder for export
            import tempfile
            temp_dir = tempfile.mkdtemp()

            exporter = ProfessionalExporter(temp_dir)
            exported_obj = exporter.export_mesh_obj(
                self.current_heightmap,
                "terrain.obj",
                scale_x=1.0,
                scale_y=50.0,  # Amplify height
                scale_z=1.0,
                subsample=2  # For performance
            )

            # Move to desired location
            import shutil
            shutil.move(exported_obj, filepath)
            shutil.rmtree(temp_dir)

            self.log(f"✓ Mesh 3D exporté: {filepath}")
            QMessageBox.information(self, "Succès", f"Mesh 3D exporté:\n{filepath}")

    def export_for_flame(self):
        """Export complet pour Autodesk Flame (OBJ+MTL+Textures)"""
        if self.current_heightmap is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        folder = QFileDialog.getExistingDirectory(self, "Choisir dossier d'export Flame")
        if folder:
            self.log("🔥 Export pour Autodesk Flame en cours...")

            try:
                # Create exporter
                exporter = ProfessionalExporter(folder)

                # Export using dedicated Flame method
                exported_files = exporter.export_for_autodesk_flame(
                    heightmap=self.current_heightmap,
                    normal_map=self.current_normal_map,
                    depth_map=self.current_depth_map,
                    ao_map=self.current_ao_map,
                    diffuse_map=None,  # Will be auto-generated from heightmap
                    roughness_map=None,
                    splatmaps=self.current_splatmaps,
                    tree_instances=self.current_tree_instances,
                    mesh_subsample=2,
                    scale_y=50.0
                )

                num_files = len(exported_files)
                self.log(f"✓ Export Flame réussi: {num_files} fichiers")

                # Build file list for display
                file_list = []
                file_list.append("📁 STRUCTURE EXPORTÉE:")
                file_list.append(f"  • terrain.obj")
                file_list.append(f"  • terrain.mtl")
                file_list.append(f"  📂 textures/")

                if exported_files.get('diffuse'):
                    file_list.append(f"    • diffuse.png")
                if exported_files.get('normal'):
                    file_list.append(f"    • normal.png")
                if exported_files.get('ao'):
                    file_list.append(f"    • ao.png")
                if exported_files.get('displacement'):
                    file_list.append(f"    • height.png (displacement)")
                if exported_files.get('depth'):
                    file_list.append(f"    • depth.png")

                splatmap_count = sum(1 for k in exported_files.keys() if k.startswith('splatmap'))
                if splatmap_count > 0:
                    file_list.append(f"    • {splatmap_count} splatmaps PBR")

                if exported_files.get('vegetation'):
                    file_list.append(f"  • vegetation.json")

                file_list.append(f"  • README_FLAME.txt")

                summary = "\n".join(file_list)

                QMessageBox.information(
                    self,
                    "Export Flame Réussi",
                    f"Export terminé avec succès!\n\n{summary}\n\nLocalisation: {folder}"
                )

                self.log(f"📦 Tous les fichiers sont dans: {folder}")

            except Exception as e:
                error_msg = f"Erreur lors de l'export Flame: {str(e)}"
                self.log(f"❌ {error_msg}")
                QMessageBox.critical(self, "Erreur Export", error_msg)

    def export_obj(self, filepath, vertices, faces):
        """Exporte en format OBJ"""
        with open(filepath, 'w') as f:
            f.write("# Mountain Studio Pro - 3D Mesh Export\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Faces
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def update_progress(self, value, text):
        """Met à jour la barre de progression"""
        self.progress_bar.setValue(value)
        self.status_label.setText(text)

    def on_error(self, error_msg):
        """Gestion d'erreur"""
        self.log(f"❌ {error_msg}")
        QMessageBox.critical(self, "Erreur", error_msg)
        self.generate_terrain_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def log(self, message):
        """Ajoute un message au log"""
        self.log_text.append(message)
        self.status_label.setText(message)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Style moderne

    # Dark theme
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = MountainProUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
