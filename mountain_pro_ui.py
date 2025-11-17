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

from terrain_generator import TerrainGenerator
from prompt_generator import MountainPromptGenerator
from comfyui_integration import ComfyUIIntegration, StableDiffusionDirect
from temporal_consistency import VideoCoherenceManager
from video_generator import VideoGenerator


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
        self.progress.emit(10, "Génération heightmap...")
        terrain_gen = TerrainGenerator(
            width=self.params['resolution'],
            height=self.params['resolution']
        )

        heightmap = terrain_gen.generate_heightmap(
            scale=self.params['scale'],
            octaves=self.params['octaves'],
            persistence=self.params['persistence'],
            lacunarity=self.params['lacunarity'],
            mountain_type=self.params['mountain_type'],
            seed=self.params['seed']
        )

        self.progress.emit(40, "Génération normal map...")
        normal_map = terrain_gen.generate_normal_map(strength=self.params['normal_strength'])

        self.progress.emit(60, "Génération depth map...")
        depth_map = terrain_gen.generate_depth_map()

        self.progress.emit(80, "Génération AO et roughness...")
        ao_map = terrain_gen.generate_ambient_occlusion()
        roughness_map = terrain_gen.generate_roughness_map()

        self.progress.emit(100, "Terminé!")

        result = {
            'heightmap': heightmap,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'ao_map': ao_map,
            'roughness_map': roughness_map,
            'terrain_gen': terrain_gen
        }

        self.finished.emit(result, 'terrain')

    def generate_texture(self):
        # Utiliser Stable Diffusion pour texturer
        self.progress.emit(20, "Initialisation SD...")
        # TODO: Implémenter
        pass

    def generate_video(self):
        # Génération vidéo cohérente
        self.progress.emit(10, "Préparation...")
        # TODO: Implémenter
        pass


class MountainProUI(QMainWindow):
    """Interface professionnelle principale"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mountain Studio Pro - Outil Professionnel pour Graphistes")
        self.setGeometry(100, 100, 1600, 900)

        # État
        self.current_terrain = None
        self.current_heightmap = None
        self.current_texture = None
        self.generation_thread = None

        # Backends
        self.comfyui = None
        self.sd_direct = None
        self.video_manager = None

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

        # Tab 2: Texture AI
        texture_tab = self.create_texture_controls()
        tabs.addTab(texture_tab, "🎨 Texture AI")

        # Tab 3: Caméra & Rendu
        camera_tab = self.create_camera_controls()
        tabs.addTab(camera_tab, "🎥 Caméra")

        # Tab 4: Export
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

        # Bouton génération
        self.generate_terrain_btn = QPushButton("🗻 Générer Terrain 3D")
        self.generate_terrain_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-weight: bold; }")
        self.generate_terrain_btn.clicked.connect(self.generate_terrain)
        layout.addWidget(self.generate_terrain_btn)

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
        """Lance la génération de terrain"""
        self.log("🗻 Génération du terrain...")

        params = {
            'resolution': int(self.resolution_combo.currentText()),
            'scale': self.scale_slider['slider'].value() * self.scale_slider['scale'],
            'octaves': self.octaves_slider['slider'].value(),
            'persistence': self.persistence_slider['slider'].value() * self.persistence_slider['scale'],
            'lacunarity': self.lacunarity_slider['slider'].value() * self.lacunarity_slider['scale'],
            'mountain_type': self.mountain_type_combo.currentText().lower(),
            'normal_strength': self.normal_strength_slider['slider'].value() * self.normal_strength_slider['scale'],
            'seed': self.seed_spinbox.value()
        }

        self.generation_thread = GenerationThread("terrain", params)
        self.generation_thread.progress.connect(self.update_progress)
        self.generation_thread.finished.connect(self.on_terrain_generated)
        self.generation_thread.error.connect(self.on_error)
        self.generation_thread.start()

        self.generate_terrain_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

    def on_terrain_generated(self, result, result_type):
        """Callback terrain généré"""
        self.current_terrain = result['terrain_gen']
        self.current_heightmap = result['heightmap']

        self.log("✓ Terrain généré avec succès!")

        # Afficher les previews
        self.display_preview(result['heightmap'], self.heightmap_preview)
        self.display_preview(result['normal_map'], self.normal_preview)
        self.display_preview(result['depth_map'], self.depth_preview)

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
        surface = gl.GLSurfacePlotItem(
            z=z,
            shader='heightColor',
            computeNormals=True,
            smooth=True
        )
        surface.scale(1, 1, 1)
        surface.translate(-h_sub.shape[0]/2, -h_sub.shape[1]/2, 0)

        self.gl_view.addItem(surface)

        # Grille
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.gl_view.addItem(grid)

    def reset_3d_view(self):
        """Reset la vue 3D"""
        self.gl_view.setCameraPosition(distance=40, elevation=30, azimuth=45)

    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        # TODO
        pass

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
        """Génère une texture AI"""
        self.log("🎨 Génération texture AI...")
        QMessageBox.information(self, "Info", "Fonction en développement")

    def generate_video(self):
        """Génère une vidéo cohérente"""
        self.log("🎬 Génération vidéo cohérente...")
        QMessageBox.information(self, "Info", "Fonction en développement")

    def export_all_maps(self):
        """Exporte toutes les maps"""
        if self.current_terrain is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        folder = QFileDialog.getExistingDirectory(self, "Choisir dossier d'export")
        if folder:
            self.current_terrain.export_all_maps(folder, prefix="mountain_pro")
            self.log(f"✓ Maps exportées vers: {folder}")
            QMessageBox.information(self, "Succès", f"Maps exportées vers:\n{folder}")

    def export_mesh(self):
        """Exporte le mesh 3D"""
        if self.current_terrain is None:
            QMessageBox.warning(self, "Attention", "Générez d'abord un terrain!")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Exporter Mesh 3D", "", "OBJ Files (*.obj)")
        if filepath:
            vertices, faces, normals = self.current_terrain.get_3d_mesh_data()
            self.export_obj(filepath, vertices, faces)
            self.log(f"✓ Mesh exporté: {filepath}")
            QMessageBox.information(self, "Succès", f"Mesh exporté:\n{filepath}")

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
