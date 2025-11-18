#!/usr/bin/env python3
"""
EXEMPLE: Intégration Presets + Végétation dans Mountain Studio v2
==================================================================

Ce fichier montre comment ajouter:
✅ Sélecteur de presets professionnels
✅ Génération réaliste de sapins/arbres
✅ Export végétation pour game engines

Pour intégrer dans mountain_studio_ultimate_v2.py:
1. Copier les imports
2. Copier les méthodes _create_presets_tab() et _create_vegetation_tab()
3. Ajouter les tabs dans init_ui()
4. Ajouter les méthodes apply_preset(), generate_vegetation(), export_vegetation()
"""

# ==============================================================================
# IMPORTS À AJOUTER
# ==============================================================================

from config.professional_presets import PresetManager, CompletePreset
from core.vegetation.species_distribution import SpeciesDistributor, SpeciesProfile
from core.vegetation.vegetation_placer import VegetationPlacer, TreeInstance
from core.vegetation.biome_classifier import BiomeClassifier, BiomeType


# ==============================================================================
# DANS __init__() - Initialiser les managers
# ==============================================================================

def example_init_managers(self):
    """Ajouter dans MountainStudioUltimate.__init__()"""

    # Preset manager
    self.preset_manager = PresetManager()

    # Vegetation components
    self.species_distributor = SpeciesDistributor()
    self.vegetation_placer = None  # Will be created after terrain generation
    self.tree_instances = []


# ==============================================================================
# NOUVEAU TAB: PRESETS
# ==============================================================================

def _create_presets_tab(self) -> QWidget:
    """Create presets selection tab"""
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QTextEdit, QGroupBox

    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Category selection
    category_group = QGroupBox("📁 Category")
    category_layout = QVBoxLayout()

    self.preset_category_combo = QComboBox()
    self.preset_category_combo.addItems([
        'All',
        'VFX Production',
        'Game Development',
        'Photography',
        'Artistic',
        'Quick Test'
    ])
    self.preset_category_combo.currentTextChanged.connect(self.on_preset_category_changed)
    category_layout.addWidget(self.preset_category_combo)

    category_group.setLayout(category_layout)
    layout.addWidget(category_group)

    # Preset selection
    preset_group = QGroupBox("🎯 Preset")
    preset_layout = QVBoxLayout()

    self.preset_combo = QComboBox()
    self.preset_combo.currentTextChanged.connect(self.on_preset_selected)
    preset_layout.addWidget(self.preset_combo)

    # Load initial presets
    self.update_preset_list('All')

    preset_group.setLayout(preset_layout)
    layout.addWidget(preset_group)

    # Preset description
    desc_group = QGroupBox("📋 Description")
    desc_layout = QVBoxLayout()

    self.preset_description = QTextEdit()
    self.preset_description.setReadOnly(True)
    self.preset_description.setMaximumHeight(150)
    desc_layout.addWidget(self.preset_description)

    desc_group.setLayout(desc_layout)
    layout.addWidget(desc_group)

    # Preset details
    details_group = QGroupBox("⚙️ Parameters")
    details_layout = QVBoxLayout()

    self.preset_details = QTextEdit()
    self.preset_details.setReadOnly(True)
    self.preset_details.setMaximumHeight(200)
    details_layout.addWidget(self.preset_details)

    details_group.setLayout(details_layout)
    layout.addWidget(details_group)

    # Apply button
    apply_btn = QPushButton("✅ APPLY PRESET")
    apply_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px; }")
    apply_btn.clicked.connect(self.apply_preset)
    layout.addWidget(apply_btn)

    layout.addStretch()
    return tab


def on_preset_category_changed(self, category: str):
    """Update preset list when category changes"""
    self.update_preset_list(category)


def update_preset_list(self, category: str):
    """Update preset combo box based on category"""
    self.preset_combo.clear()

    if category == 'All':
        presets = self.preset_manager.list_presets()
    else:
        # Map UI category to preset category
        category_map = {
            'VFX Production': 'vfx_production',
            'Game Development': 'game_dev',
            'Photography': 'photography',
            'Artistic': 'artistic',
            'Quick Test': 'quick_test'
        }
        preset_category = category_map.get(category)
        if preset_category:
            presets = self.preset_manager.list_presets(category=preset_category)
        else:
            presets = []

    self.preset_combo.addItems(presets)


def on_preset_selected(self, preset_name: str):
    """Show preset details when selected"""
    if not preset_name:
        return

    preset = self.preset_manager.get_preset(preset_name)
    if not preset:
        return

    # Update description
    self.preset_description.setText(f"{preset.name}\n\n{preset.description}")

    # Update details
    details = f"""
**Terrain**
- Resolution: {preset.terrain.width}x{preset.terrain.height}
- Type: {preset.terrain.mountain_type}
- Scale: {preset.terrain.scale}
- Octaves: {preset.terrain.octaves}
- Seed: {preset.terrain.seed}
- Hydraulic Erosion: {preset.terrain.apply_hydraulic_erosion} ({preset.terrain.erosion_iterations} iterations)
- Thermal Erosion: {preset.terrain.apply_thermal_erosion}

**Vegetation**
- Enabled: {preset.vegetation.enabled}
- Density: {preset.vegetation.density}
- Min Spacing: {preset.vegetation.min_spacing}m
- Clustering: {preset.vegetation.use_clustering}

**Render**
- Season: {preset.render.season}
- Time: {preset.render.time_of_day}
- Weather: {preset.render.weather}
- Quality: {preset.render.quality_level}

**Export**
- Heightmap: {preset.export.export_heightmap}
- Normal Map: {preset.export.export_normal_map}
- PBR Splatmap: {preset.export.export_splatmap}
- OBJ: {preset.export.export_obj}
- Vegetation Instances: {preset.export.export_vegetation_instances}
"""

    self.preset_details.setText(details)


def apply_preset(self):
    """Apply selected preset to all parameters"""
    preset_name = self.preset_combo.currentText()
    if not preset_name:
        QMessageBox.warning(self, "Warning", "No preset selected!")
        return

    preset = self.preset_manager.get_preset(preset_name)
    if not preset:
        return

    # Apply terrain parameters
    self.width_spin.setValue(preset.terrain.width)
    self.height_spin.setValue(preset.terrain.height)
    self.scale_slider.setValue(int(preset.terrain.scale))
    self.octaves_spin.setValue(preset.terrain.octaves)
    self.ridge_slider.setValue(int(preset.terrain.domain_warp_strength * 100))
    self.warp_slider.setValue(int(preset.terrain.domain_warp_strength * 100))

    if preset.terrain.apply_hydraulic_erosion:
        iterations = min(100, preset.terrain.erosion_iterations // 1000)
        self.hydraulic_spin.setValue(iterations)
    else:
        self.hydraulic_spin.setValue(0)

    if preset.terrain.apply_thermal_erosion:
        self.thermal_spin.setValue(5)
    else:
        self.thermal_spin.setValue(0)

    self.seed_spin.setValue(preset.terrain.seed)

    self.log(f"✅ Applied preset: {preset.name}")
    QMessageBox.information(self, "Preset Applied", f"Preset '{preset.name}' has been applied!\n\nClick 'GENERATE TERRAIN' to create.")


# ==============================================================================
# NOUVEAU TAB: VÉGÉTATION
# ==============================================================================

def _create_vegetation_tab(self) -> QWidget:
    """Create vegetation generation tab"""
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QSlider, QSpinBox, QPushButton, QLabel, QGroupBox, QGridLayout, QComboBox

    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Enable/Disable
    self.vegetation_enabled = QCheckBox("Enable Vegetation Generation")
    self.vegetation_enabled.setChecked(True)
    layout.addWidget(self.vegetation_enabled)

    # Density settings
    density_group = QGroupBox("🌲 Density")
    density_layout = QGridLayout()

    density_layout.addWidget(QLabel("Overall Density:"), 0, 0)
    self.vegetation_density_slider = QSlider(Qt.Horizontal)
    self.vegetation_density_slider.setRange(0, 100)
    self.vegetation_density_slider.setValue(50)
    self.vegetation_density_label = QLabel("0.50")
    self.vegetation_density_slider.valueChanged.connect(
        lambda v: self.vegetation_density_label.setText(f"{v/100:.2f}")
    )
    density_layout.addWidget(self.vegetation_density_slider, 0, 1)
    density_layout.addWidget(self.vegetation_density_label, 0, 2)

    density_layout.addWidget(QLabel("Min Spacing (m):"), 1, 0)
    self.vegetation_spacing_spin = QDoubleSpinBox()
    self.vegetation_spacing_spin.setRange(1.0, 20.0)
    self.vegetation_spacing_spin.setValue(4.0)
    self.vegetation_spacing_spin.setSingleStep(0.5)
    density_layout.addWidget(self.vegetation_spacing_spin, 1, 1, 1, 2)

    density_group.setLayout(density_layout)
    layout.addWidget(density_group)

    # Clustering
    clustering_group = QGroupBox("🌳 Clustering")
    clustering_layout = QGridLayout()

    self.vegetation_clustering = QCheckBox("Use Clustering (Natural Groups)")
    self.vegetation_clustering.setChecked(True)
    clustering_layout.addWidget(self.vegetation_clustering, 0, 0, 1, 3)

    clustering_layout.addWidget(QLabel("Cluster Size:"), 1, 0)
    self.vegetation_cluster_size_spin = QSpinBox()
    self.vegetation_cluster_size_spin.setRange(3, 20)
    self.vegetation_cluster_size_spin.setValue(8)
    clustering_layout.addWidget(self.vegetation_cluster_size_spin, 1, 1, 1, 2)

    clustering_layout.addWidget(QLabel("Cluster Radius (m):"), 2, 0)
    self.vegetation_cluster_radius_spin = QDoubleSpinBox()
    self.vegetation_cluster_radius_spin.setRange(5.0, 50.0)
    self.vegetation_cluster_radius_spin.setValue(15.0)
    self.vegetation_cluster_radius_spin.setSingleStep(1.0)
    clustering_layout.addWidget(self.vegetation_cluster_radius_spin, 2, 1, 1, 2)

    clustering_group.setLayout(clustering_layout)
    layout.addWidget(clustering_group)

    # Species selection
    species_group = QGroupBox("🌲 Species Mix")
    species_layout = QVBoxLayout()

    species_layout.addWidget(QLabel("Automatic based on terrain altitude/biome"))
    species_layout.addWidget(QLabel("Available species: Pine, Spruce, Fir, Larch, Oak, Birch"))

    species_group.setLayout(species_layout)
    layout.addWidget(species_group)

    # Generate button
    self.generate_vegetation_btn = QPushButton("🌲 GENERATE VEGETATION")
    self.generate_vegetation_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 10px; }")
    self.generate_vegetation_btn.clicked.connect(self.generate_vegetation)
    layout.addWidget(self.generate_vegetation_btn)

    # Stats
    stats_group = QGroupBox("📊 Statistics")
    stats_layout = QVBoxLayout()
    self.vegetation_stats_label = QLabel("No vegetation generated yet.")
    stats_layout.addWidget(self.vegetation_stats_label)
    stats_group.setLayout(stats_layout)
    layout.addWidget(stats_group)

    # Export
    export_group = QGroupBox("💾 Export")
    export_layout = QGridLayout()

    export_layout.addWidget(QLabel("Format:"), 0, 0)
    self.vegetation_export_format = QComboBox()
    self.vegetation_export_format.addItems(['JSON (Generic)', 'Unreal Engine', 'Unity'])
    export_layout.addWidget(self.vegetation_export_format, 0, 1)

    self.export_vegetation_btn = QPushButton("💾 Export Vegetation Instances")
    self.export_vegetation_btn.clicked.connect(self.export_vegetation)
    export_layout.addWidget(self.export_vegetation_btn, 1, 0, 1, 2)

    export_group.setLayout(export_layout)
    layout.addWidget(export_group)

    layout.addStretch()
    return tab


def generate_vegetation(self):
    """Generate realistic vegetation on terrain"""
    if self.terrain is None:
        QMessageBox.warning(self, "Warning", "Generate terrain first!")
        return

    if not self.vegetation_enabled.isChecked():
        QMessageBox.information(self, "Info", "Vegetation is disabled. Enable it first.")
        return

    self.log("🌲 Generating vegetation...")
    self.generate_vegetation_btn.setEnabled(False)

    try:
        # Create biome classifier
        from core.vegetation.biome_classifier import BiomeClassifier

        biome_classifier = BiomeClassifier(self.terrain)
        biome_map = biome_classifier.classify()

        # Create vegetation placer
        h, w = self.terrain.shape
        placer = VegetationPlacer(w, h, self.terrain, biome_map)

        # Place vegetation
        density = self.vegetation_density_slider.value() / 100.0
        spacing = self.vegetation_spacing_spin.value()
        use_clustering = self.vegetation_clustering.isChecked()
        cluster_size = self.vegetation_cluster_size_spin.value()
        cluster_radius = self.vegetation_cluster_radius_spin.value()

        self.tree_instances = placer.place_vegetation(
            density=density,
            min_spacing=spacing,
            use_clustering=use_clustering,
            cluster_size=cluster_size,
            cluster_radius=cluster_radius,
            seed=self.seed_spin.value()
        )

        # Update stats
        species_count = {}
        for tree in self.tree_instances:
            species_count[tree.species] = species_count.get(tree.species, 0) + 1

        stats_text = f"Total trees: {len(self.tree_instances)}\n\nBy species:\n"
        for species, count in sorted(species_count.items()):
            stats_text += f"  {species}: {count}\n"

        self.vegetation_stats_label.setText(stats_text)

        self.log(f"✅ Vegetation generated: {len(self.tree_instances)} trees")
        QMessageBox.information(self, "Success", f"Generated {len(self.tree_instances)} trees!")

    except Exception as e:
        self.log(f"❌ Vegetation generation error: {e}")
        QMessageBox.critical(self, "Error", f"Vegetation generation failed:\n{e}")

    finally:
        self.generate_vegetation_btn.setEnabled(True)


def export_vegetation(self):
    """Export vegetation instances for game engines"""
    if not self.tree_instances:
        QMessageBox.warning(self, "Warning", "Generate vegetation first!")
        return

    format_type = self.vegetation_export_format.currentText()

    try:
        output_path = self.output_dir / "vegetation_instances.json"

        if format_type == 'JSON (Generic)':
            data = {
                'version': '1.0',
                'total_count': len(self.tree_instances),
                'instances': [
                    {
                        'x': tree.x,
                        'y': tree.y,
                        'elevation': tree.elevation,
                        'species': tree.species,
                        'scale': tree.scale,
                        'rotation': tree.rotation,
                        'age': tree.age,
                        'health': tree.health
                    }
                    for tree in self.tree_instances
                ]
            }

        elif format_type == 'Unreal Engine':
            data = {
                'version': 'Unreal Engine 5',
                'instances': [
                    {
                        'asset': f'/Game/Trees/{tree.species.capitalize()}_01',
                        'transform': {
                            'translation': [float(tree.x), float(tree.y), float(tree.elevation * 100)],
                            'rotation': [0.0, 0.0, float(tree.rotation)],
                            'scale': [float(tree.scale), float(tree.scale), float(tree.scale)]
                        }
                    }
                    for tree in self.tree_instances
                ]
            }

        elif format_type == 'Unity':
            data = {
                'version': 'Unity',
                'treeInstances': [
                    {
                        'prototypeIndex': 0,  # You'd map species to prototype index
                        'position': {'x': float(tree.x), 'y': float(tree.elevation), 'z': float(tree.y)},
                        'widthScale': float(tree.scale),
                        'heightScale': float(tree.scale),
                        'rotation': float(tree.rotation),
                        'color': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
                        'lightmapColor': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0}
                    }
                    for tree in self.tree_instances
                ]
            }

        # Save
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.log(f"💾 Exported vegetation: {output_path}")
        QMessageBox.information(self, "Success", f"Vegetation exported to:\n{output_path}")

    except Exception as e:
        self.log(f"❌ Export error: {e}")
        QMessageBox.critical(self, "Error", f"Export failed:\n{e}")


# ==============================================================================
# DANS init_ui() - Ajouter les nouveaux tabs
# ==============================================================================

def example_add_tabs_to_ui(self):
    """Ajouter après les tabs existants dans init_ui()"""

    # Tab 7: Presets
    presets_tab = self._create_presets_tab()
    self.tabs.addTab(presets_tab, "🎯 Presets")

    # Tab 8: Vegetation
    vegetation_tab = self._create_vegetation_tab()
    self.tabs.addTab(vegetation_tab, "🌲 Vegetation")


# ==============================================================================
# EXEMPLE COMPLET D'UTILISATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("EXEMPLE: Presets + Végétation pour Mountain Studio v2")
    print("=" * 80)
    print()
    print("Pour intégrer dans mountain_studio_ultimate_v2.py:")
    print()
    print("1. Ajouter les imports en haut du fichier:")
    print("   from config.professional_presets import PresetManager")
    print("   from core.vegetation.vegetation_placer import VegetationPlacer")
    print("   from core.vegetation.biome_classifier import BiomeClassifier")
    print()
    print("2. Dans __init__(), ajouter:")
    print("   self.preset_manager = PresetManager()")
    print("   self.tree_instances = []")
    print()
    print("3. Copier les méthodes:")
    print("   - _create_presets_tab()")
    print("   - _create_vegetation_tab()")
    print("   - apply_preset()")
    print("   - generate_vegetation()")
    print("   - export_vegetation()")
    print()
    print("4. Dans init_ui(), ajouter les tabs:")
    print("   self.tabs.addTab(self._create_presets_tab(), '🎯 Presets')")
    print("   self.tabs.addTab(self._create_vegetation_tab(), '🌲 Vegetation')")
    print()
    print("=" * 80)
    print()
    print("Voir QUICK_START_GUIDE.md pour utilisation complète!")
    print()
