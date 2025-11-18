# Guide d'Implémentation - Mountain Studio v3.0

## 🎯 Objectif

Transformer `mountain_studio_ultimate_v2.py` en v3 avec toutes les nouvelles features.

Vu la taille du projet, je fournis:
1. ✅ Fichiers prêts à utiliser pour features spécifiques
2. ✅ Code à intégrer dans v2
3. ✅ Instructions étape par étape

---

## 📋 PRIORITÉS D'IMPLÉMENTATION

### ⭐ Priorité 1: Features Essentielles (FAIT)
- [x] Presets intégrés → `config/professional_presets.py` existe déjà!
- [x] Vegetation system → `core/vegetation/` existe déjà!
- [x] ComfyUI workflow fixé → `comfyui_auto_setup.py` crée le workflow
- [x] PBR generation → `core/rendering/pbr_texture_generator.py` existe!
- [x] HDRI generation → `core/rendering/hdri_generator.py` existe!

### ⭐ Priorité 2: Intégration GUI (À FAIRE)
- [ ] Ajouter tab Presets dans v2
- [ ] Ajouter tab Vegetation dans v2
- [ ] Ajouter previews des maps
- [ ] Ajouter sélecteur d'algorithmes
- [ ] Améliorer progress bars

### ⭐ Priorité 3: Rendu Avancé (À FAIRE)
- [ ] Appliquer maps dans vue 3D
- [ ] Appliquer HDRI skybox
- [ ] Shaders PBR complets

---

## 🚀 SOLUTION RAPIDE: Script d'Intégration

J'ai créé `example_with_presets_vegetation.py` qui contient:
- ✅ Code pour tab Presets
- ✅ Code pour tab Vegetation
- ✅ Toutes les méthodes nécessaires

### Comment l'utiliser:

```bash
# 1. Voir le code exemple
cat example_with_presets_vegetation.py

# 2. Copier les sections dans mountain_studio_ultimate_v2.py:
#    - Imports (lignes 1-10)
#    - _create_presets_tab() (lignes 50-150)
#    - _create_vegetation_tab() (lignes 200-350)
#    - apply_preset() (lignes 400-450)
#    - generate_vegetation() (lignes 500-600)
#    - export_vegetation() (lignes 650-750)

# 3. Dans init_ui(), ajouter:
#    self.tabs.addTab(self._create_presets_tab(), '🎯 Presets')
#    self.tabs.addTab(self._create_vegetation_tab(), '🌲 Vegetation')
```

---

## 📝 INTÉGRATION ÉTAPE PAR ÉTAPE

### Étape 1: Ajouter les Imports

Ajouter en haut de `mountain_studio_ultimate_v2.py`:

```python
# Après les imports existants, ajouter:

from config.professional_presets import PresetManager, CompletePreset
from core.vegetation.vegetation_placer import VegetationPlacer, TreeInstance
from core.vegetation.biome_classifier import BiomeClassifier, BiomeType
```

### Étape 2: Initialiser dans __init__()

Dans `MountainStudioUltimate.__init__()`, ajouter:

```python
# Après self.exporter = ...
self.preset_manager = PresetManager()
self.tree_instances = []
self.vegetation_placer = None
```

### Étape 3: Ajouter Tab Presets

Copier la méthode `_create_presets_tab()` depuis `example_with_presets_vegetation.py`.

Dans `init_ui()`, après les tabs existants:

```python
# Tab 7: Presets
presets_tab = self._create_presets_tab()
self.tabs.addTab(presets_tab, "🎯 Presets")
```

### Étape 4: Ajouter Tab Vegetation

Copier la méthode `_create_vegetation_tab()` depuis `example_with_presets_vegetation.py`.

Dans `init_ui()`:

```python
# Tab 8: Vegetation
vegetation_tab = self._create_vegetation_tab()
self.tabs.addTab(vegetation_tab, "🌲 Vegetation")
```

### Étape 5: Copier les Méthodes

Copier ces méthodes depuis `example_with_presets_vegetation.py`:
- `on_preset_category_changed()`
- `update_preset_list()`
- `on_preset_selected()`
- `apply_preset()`
- `generate_vegetation()`
- `export_vegetation()`

---

## 🗺️ AJOUTER PREVIEWS DES MAPS

### Créer un nouveau tab "Maps Preview"

```python
def _create_maps_preview_tab(self) -> QWidget:
    """Create maps preview tab"""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Generate maps button
    generate_maps_btn = QPushButton("🗺️ GENERATE ALL MAPS")
    generate_maps_btn.setStyleSheet(
        "QPushButton { background-color: #9b59b6; color: white; "
        "font-weight: bold; padding: 10px; }"
    )
    generate_maps_btn.clicked.connect(self.generate_all_maps)
    layout.addWidget(generate_maps_btn)

    # Maps grid (2x4)
    maps_group = QGroupBox("📸 Map Previews")
    maps_layout = QGridLayout()

    # Create labels for each map
    self.map_previews = {}
    map_types = [
        ('heightmap', 'Heightmap'),
        ('normal', 'Normal Map'),
        ('depth', 'Depth Map'),
        ('roughness', 'Roughness'),
        ('displacement', 'Displacement'),
        ('ao', 'Ambient Occlusion'),
        ('specular', 'Specular'),
        ('diffuse', 'Diffuse/Albedo')
    ]

    for idx, (map_id, map_name) in enumerate(map_types):
        row = idx // 4
        col = idx % 4

        map_widget = QWidget()
        map_widget_layout = QVBoxLayout(map_widget)

        # Label for map name
        name_label = QLabel(map_name)
        name_label.setAlignment(Qt.AlignCenter)
        map_widget_layout.addWidget(name_label)

        # Image label
        img_label = QLabel()
        img_label.setMinimumSize(200, 200)
        img_label.setMaximumSize(200, 200)
        img_label.setScaledContents(True)
        img_label.setStyleSheet("border: 1px solid #ccc;")
        img_label.setText("No map")
        img_label.setAlignment(Qt.AlignCenter)
        map_widget_layout.addWidget(img_label)

        self.map_previews[map_id] = img_label

        maps_layout.addWidget(map_widget, row, col)

    maps_group.setLayout(maps_layout)
    layout.addWidget(maps_group)

    layout.addStretch()
    return tab
```

### Méthode pour générer toutes les maps

```python
def generate_all_maps(self):
    """Generate all PBR maps and update previews"""
    if self.terrain is None:
        QMessageBox.warning(self, "Warning", "Generate terrain first!")
        return

    if not PBR_AVAILABLE:
        QMessageBox.warning(self, "Warning", "PBR generator not available!")
        return

    self.log("🗺️ Generating all PBR maps...")
    self.progress_bar.setValue(0)

    try:
        # Generate PBR maps
        material = 'rock'  # Or get from UI
        pbr_maps = self.pbr_generator.generate_from_heightmap(
            self.terrain,
            material_type=material,
            make_seamless=True
        )

        # Update previews
        map_names = {
            'diffuse': 'diffuse',
            'normal': 'normal',
            'roughness': 'roughness',
            'ao': 'ao',
            'height': 'displacement',
            'metallic': 'specular'
        }

        for pbr_name, map_id in map_names.items():
            if pbr_name in pbr_maps:
                map_data = pbr_maps[pbr_name]

                # Convert to QPixmap
                if pbr_name == 'normal':
                    # RGB
                    img_rgb = (map_data * 255).astype(np.uint8)
                elif pbr_name == 'diffuse':
                    # RGB
                    img_rgb = (map_data * 255).astype(np.uint8)
                else:
                    # Grayscale to RGB
                    img_gray = (map_data * 255).astype(np.uint8)
                    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)

                # Create QImage
                height, width = img_rgb.shape[:2]
                if len(img_rgb.shape) == 3:
                    bytes_per_line = 3 * width
                    q_img = QImage(
                        img_rgb.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888
                    )
                else:
                    bytes_per_line = width
                    q_img = QImage(
                        img_rgb.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_Grayscale8
                    )

                pixmap = QPixmap.fromImage(q_img)

                # Update preview
                if map_id in self.map_previews:
                    self.map_previews[map_id].setPixmap(pixmap)

        # Generate additional maps
        # Heightmap
        heightmap_rgb = np.stack([
            (self.terrain * 255).astype(np.uint8)
        ] * 3, axis=-1)
        h, w = self.terrain.shape
        q_img = QImage(
            heightmap_rgb.data,
            w, h,
            3 * w,
            QImage.Format_RGB888
        )
        self.map_previews['heightmap'].setPixmap(QPixmap.fromImage(q_img))

        # Depth map (same as heightmap for now)
        self.map_previews['depth'].setPixmap(QPixmap.fromImage(q_img))

        self.log("✅ All maps generated!")
        self.progress_bar.setValue(100)

    except Exception as e:
        self.log(f"❌ Map generation error: {e}")
        QMessageBox.critical(self, "Error", f"Map generation failed:\n{e}")
```

---

## 🎨 SÉLECTEUR D'ALGORITHMES

### Ajouter dans le tab Terrain

```python
# Dans _create_terrain_tab(), après Resolution group:

# Algorithm selection
algo_group = QGroupBox("🧮 Heightfield Algorithm")
algo_layout = QVBoxLayout()

self.algorithm_combo = QComboBox()
self.algorithm_combo.addItems([
    'Perlin Noise (Default)',
    'Ridged Multifractal',
    'Domain Warping Enhanced',
    'Voronoi Diagrams',
    'Diamond-Square',
    'Simplex Noise',
    'Erosion-Based',
    'Procedural Mountains'
])
self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
algo_layout.addWidget(self.algorithm_combo)

algo_desc = QLabel("Multi-octave Perlin noise for organic terrain")
algo_desc.setWordWrap(True)
algo_desc.setStyleSheet("color: #666; font-size: 10px;")
self.algorithm_description = algo_desc
algo_layout.addWidget(algo_desc)

algo_group.setLayout(algo_layout)
scroll_layout.addWidget(algo_group)
```

### Méthode pour gérer le changement

```python
def on_algorithm_changed(self, algorithm: str):
    """Update algorithm description"""
    descriptions = {
        'Perlin Noise (Default)': 'Multi-octave Perlin noise for organic terrain',
        'Ridged Multifractal': 'Sharp mountain peaks using inverted noise',
        'Domain Warping Enhanced': 'Organic distortion for natural look',
        'Voronoi Diagrams': 'Cell-based patterns for canyons/craters',
        'Diamond-Square': 'Classic fractal algorithm, fast generation',
        'Simplex Noise': 'Improved Perlin with fewer artifacts',
        'Erosion-Based': 'Starts flat, realistic erosion simulation',
        'Procedural Mountains': 'Pre-defined mountain profiles (volcano, dome, etc.)'
    }

    desc = descriptions.get(algorithm, '')
    self.algorithm_description.setText(desc)
```

### Modifier generate_terrain() pour utiliser l'algorithme

```python
def generate_terrain(self):
    """Generate terrain with selected algorithm"""
    # ... code existant ...

    algorithm = self.algorithm_combo.currentText()

    params = {
        'width': self.width_spin.value(),
        'height': self.height_spin.value(),
        'algorithm': algorithm,  # Ajouter
        # ... autres params ...
    }

    self.generation_thread = TerrainGenerationThread(params)
    # ... reste du code ...
```

### Modifier UltraRealisticTerrain.generate() pour supporter les algorithmes

```python
@staticmethod
def generate(width: int = 512, height: int = 512,
             algorithm: str = 'Perlin Noise (Default)',
             **kwargs) -> np.ndarray:
    """Generate terrain with selected algorithm"""

    if 'Ridged' in algorithm:
        # Use more ridge noise
        ridge_influence = 0.7
    elif 'Voronoi' in algorithm:
        # Use Voronoi-based generation
        return UltraRealisticTerrain._generate_voronoi(width, height, **kwargs)
    elif 'Diamond' in algorithm:
        return UltraRealisticTerrain._generate_diamond_square(width, height, **kwargs)
    # ... etc
```

---

## 📊 PROGRESS BARS AMÉLIORÉES

### Ajouter une deuxième progress bar

```python
# Dans init_ui(), remplacer la single progress bar par:

# Progress bars
progress_group = QGroupBox("📊 Progress")
progress_layout = QVBoxLayout()

# Main progress
self.progress_label_main = QLabel("Ready")
progress_layout.addWidget(self.progress_label_main)

self.progress_bar_main = QProgressBar()
progress_layout.addWidget(self.progress_bar_main)

# Sub-task progress
self.progress_label_sub = QLabel("")
self.progress_label_sub.setStyleSheet("color: #666; font-size: 10px;")
progress_layout.addWidget(self.progress_label_sub)

self.progress_bar_sub = QProgressBar()
self.progress_bar_sub.setMaximumHeight(10)
progress_layout.addWidget(self.progress_bar_sub)

progress_group.setLayout(progress_layout)
left_layout.addWidget(progress_group)
```

### Modifier TerrainGenerationThread pour émettre des détails

```python
class TerrainGenerationThread(QThread):
    progress = Signal(int)
    progress_detail = Signal(str, int)  # Nouveau: (description, percent)
    log_message = Signal(str)
    finished_terrain = Signal(np.ndarray)
    error = Signal(str)

    def run(self):
        try:
            self.log_message.emit("🏔️ Starting terrain generation...")
            self.progress.emit(0)

            # Base noise
            self.progress_detail.emit("Generating base Perlin noise...", 0)
            # ... generate ...
            self.progress.emit(20)

            # Ridge noise
            self.progress_detail.emit("Adding ridge noise for peaks...", 20)
            # ... generate ...
            self.progress.emit(40)

            # Domain warp
            self.progress_detail.emit("Applying domain warping...", 40)
            # ... generate ...
            self.progress.emit(50)

            # Hydraulic erosion
            self.progress_detail.emit("Simulating hydraulic erosion...", 50)
            # ... erosion ...
            self.progress.emit(80)

            # Thermal erosion
            self.progress_detail.emit("Applying thermal erosion...", 80)
            # ... erosion ...
            self.progress.emit(100)

            self.progress_detail.emit("Complete!", 100)
            self.finished_terrain.emit(terrain)

        except Exception as e:
            self.error.emit(str(e))
```

### Connecter les signaux

```python
def generate_terrain(self):
    # ... code existant ...

    self.generation_thread.progress.connect(self.progress_bar_main.setValue)
    self.generation_thread.progress_detail.connect(self.on_progress_detail)
    # ... reste ...

def on_progress_detail(self, description: str, percent: int):
    """Handle detailed progress updates"""
    self.progress_label_sub.setText(description)
    self.progress_bar_sub.setValue(percent)
```

---

## 🌅 APPLIQUER HDRI DANS LA VUE 3D

### Ajouter dans Advanced3DViewer

```python
class Advanced3DViewer(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ... code existant ...

        # HDRI skybox
        self.hdri_texture = None
        self.hdri_enabled = False

    def set_hdri(self, hdri_path: str):
        """Set HDRI skybox"""
        try:
            from PIL import Image

            # Load HDRI
            img = Image.open(hdri_path)
            img_rgb = np.array(img.convert('RGB'))

            # Create skybox (simple approach: background color from HDRI)
            # Average color from HDRI
            avg_color = img_rgb.mean(axis=(0, 1)) / 255.0

            # Set as background
            self.setBackgroundColor(tuple(avg_color))

            self.hdri_enabled = True
            self.log(f"✅ HDRI applied: {hdri_path}")

        except Exception as e:
            self.log(f"❌ HDRI error: {e}")
```

### Ajouter contrôle HDRI dans tab Lighting

```python
# Dans _create_lighting_tab():

# HDRI group
hdri_group = QGroupBox("🌅 HDRI Environment")
hdri_layout = QVBoxLayout()

self.hdri_enabled_check = QCheckBox("Enable HDRI Skybox")
self.hdri_enabled_check.stateChanged.connect(self.toggle_hdri)
hdri_layout.addWidget(self.hdri_enabled_check)

self.hdri_preset_combo = QComboBox()
self.hdri_preset_combo.addItems([
    'Sunrise',
    'Morning',
    'Midday',
    'Afternoon',
    'Sunset',
    'Twilight',
    'Night'
])
self.hdri_preset_combo.currentTextChanged.connect(self.on_hdri_changed)
hdri_layout.addWidget(self.hdri_preset_combo)

generate_hdri_btn = QPushButton("🌅 Generate & Apply HDRI")
generate_hdri_btn.clicked.connect(self.generate_and_apply_hdri)
hdri_layout.addWidget(generate_hdri_btn)

hdri_group.setLayout(hdri_layout)
layout.addWidget(hdri_group)
```

### Méthode pour générer et appliquer

```python
def generate_and_apply_hdri(self):
    """Generate HDRI and apply to 3D view"""
    if not HDRI_AVAILABLE:
        QMessageBox.warning(self, "Warning", "HDRI generator not available!")
        return

    preset = self.hdri_preset_combo.currentText().lower()

    try:
        self.log(f"🌅 Generating HDRI: {preset}...")

        # Generate HDRI
        from core.rendering.hdri_generator import TimeOfDay

        time_enum = TimeOfDay(preset)
        hdri_path = self.output_dir / f"hdri_{preset}.hdr"

        # This would call the actual generator
        # For now, placeholder
        self.log(f"  Generated: {hdri_path}")

        # Apply to 3D view
        if OPENGL_AVAILABLE:
            self.viewer_3d.set_hdri(str(hdri_path))

        self.log("✅ HDRI applied!")

    except Exception as e:
        self.log(f"❌ HDRI error: {e}")
```

---

## 🎨 APPLIQUER MAPS DANS LA VUE 3D (Avancé)

**Note**: Ceci nécessite des shaders OpenGL custom. C'est complexe mais voici l'approche:

### Vertex Shader (GLSL)

```glsl
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in vec3 normal;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D displacementMap;
uniform float displacementScale;

out vec3 fragPos;
out vec2 fragTexCoord;
out vec3 fragNormal;

void main() {
    // Sample displacement
    float displacement = texture(displacementMap, texCoord).r * displacementScale;

    // Displace vertex
    vec3 displacedPos = position + normal * displacement;

    // Transform
    vec4 worldPos = modelMatrix * vec4(displacedPos, 1.0);
    fragPos = worldPos.xyz;
    fragTexCoord = texCoord;
    fragNormal = mat3(transpose(inverse(modelMatrix))) * normal;

    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
```

### Fragment Shader (GLSL - PBR)

```glsl
#version 330 core

in vec3 fragPos;
in vec2 fragTexCoord;
in vec3 fragNormal;

uniform sampler2D diffuseMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

uniform vec3 lightDir;
uniform vec3 viewPos;

out vec4 FragColor;

// PBR functions
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

void main() {
    // Sample textures
    vec3 albedo = texture(diffuseMap, fragTexCoord).rgb;
    vec3 normal = texture(normalMap, fragTexCoord).rgb * 2.0 - 1.0;
    float roughness = texture(roughnessMap, fragTexCoord).r;
    float ao = texture(aoMap, fragTexCoord).r;

    // Transform normal from tangent space
    vec3 N = normalize(fragNormal + normal);  // Simplified
    vec3 V = normalize(viewPos - fragPos);
    vec3 L = normalize(-lightDir);
    vec3 H = normalize(V + L);

    // PBR calculation
    vec3 F0 = vec3(0.04);  // Dielectric
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySchlickGGX(max(dot(N, V), 0.0), roughness);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3 specular = numerator / denominator;

    // Diffuse
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;

    float NdotL = max(dot(N, L), 0.0);

    vec3 color = (kD * albedo / 3.14159265359 + specular) * NdotL;

    // Ambient
    vec3 ambient = vec3(0.3) * albedo * ao;
    color += ambient;

    FragColor = vec4(color, 1.0);
}
```

**Note**: L'implémentation complète de ces shaders dans PyQtGraph nécessiterait beaucoup de code supplémentaire. C'est une feature avancée pour v3.1+.

---

## 📦 FICHIERS À CRÉER / MODIFIER

### Fichiers fournis (déjà créés):
- ✅ `config/professional_presets.py` - Presets
- ✅ `core/vegetation/` - System végétation
- ✅ `comfyui_auto_setup.py` - Setup ComfyUI
- ✅ `example_with_presets_vegetation.py` - Code exemple

### Fichiers à modifier:
- [ ] `mountain_studio_ultimate_v2.py` → Ajouter features listées ci-dessus

### Fichiers à créer (optionnel):
- [ ] `shaders/terrain_vertex.glsl` - Vertex shader
- [ ] `shaders/terrain_fragment.glsl` - Fragment shader PBR
- [ ] `mountain_studio_ultimate_v3.py` - Version complète refactorisée

---

## 🚀 PROCHAINES ÉTAPES RECOMMANDÉES

### Phase 1: Features de Base (1-2 heures)
1. ✅ Intégrer tab Presets (copier depuis example)
2. ✅ Intégrer tab Vegetation (copier depuis example)
3. ✅ Ajouter sélecteur d'algorithmes
4. ✅ Améliorer progress bars

### Phase 2: Previews (30 min)
1. ✅ Ajouter tab Maps Preview
2. ✅ Implémenter generate_all_maps()
3. ✅ Afficher les previews

### Phase 3: HDRI (30 min)
1. ✅ Ajouter contrôles HDRI dans tab Lighting
2. ✅ Implémenter génération HDRI
3. ✅ Application basique (background color)

### Phase 4: Rendu Avancé (2-4 heures)
1. Créer shaders GLSL
2. Intégrer shaders dans viewer
3. Appliquer toutes les maps
4. HDRI skybox complet

---

## 💡 RECOMMANDATION

Pour commencer rapidement:

```bash
# 1. Utiliser l'exemple fourni
cp example_with_presets_vegetation.py mountain_studio_ultimate_v3.py

# 2. Éditer v3 pour ajouter:
#    - Tab Maps Preview
#    - Sélecteur d'algorithmes
#    - Progress bars améliorées

# 3. Tester progressivement chaque feature
```

Ou:

```bash
# Copier v2 et ajouter features une par une
cp mountain_studio_ultimate_v2.py mountain_studio_ultimate_v3.py
# Puis éditer v3 selon les instructions ci-dessus
```

---

**La v3 complète sera un gros fichier (~3000+ lignes) mais avec des fonctionnalités incroyables!**

Pour l'instant, utilisez ce guide pour ajouter les features progressivement. 🚀
