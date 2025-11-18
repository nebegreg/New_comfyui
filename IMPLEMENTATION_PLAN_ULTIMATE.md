# Plan d'Implémentation Ultimate - Mountain Studio Pro

## Vue d'Ensemble

Implémentation de 3 fonctionnalités avancées:
1. **HDRI Panoramique 360°** - Génération de skybox/environnement
2. **Ombres Temps Réel** - Shadow mapping avec shaders custom
3. **Caméra FPS** - Contrôles WASD + mouse look

**Ressources disponibles:** 24 GB VRAM

---

## 1. HDRI Panoramique 360°

### Architecture
```
core/rendering/hdri_generator.py
├── HDRIPanoramicGenerator
│   ├── generate_equirectangular()
│   ├── generate_cubemap()
│   ├── export_hdr()
│   └── export_exr()
```

### Approches Techniques

**Approche 1: Stable Diffusion XL + Panorama**
- Modèle: stabilityai/stable-diffusion-xl-base-1.0
- Custom pipeline pour équirectangulaire (2:1 ratio)
- ControlNet pour cohérence de perspective
- Format: 8192x4096 ou 4096x2048
- VRAM: ~10-12 GB

**Approche 2: 6-face Cubemap Assembly**
- Générer 6 vues (front, back, left, right, top, bottom)
- Assembler en cubemap cohérente
- Plus de contrôle sur chaque face
- VRAM: ~8-10 GB

**Approche 3: Procedural + AI Enhancement**
- Générer skybox procédural de base
- Améliorer avec AI pour détails
- Plus rapide, cohérence garantie
- VRAM: ~6-8 GB

**Choix: Approche 1 + 3 Hybride**
- Base procédurale pour gradient ciel + montagnes lointaines
- AI enhancement optionnel
- Format output: .hdr (Radiance HDR) et .exr (OpenEXR)

### Caractéristiques
- Résolution: 4096x2048 (équirectangulaire)
- HDR range: 0.01 - 100.0 (exposition)
- Presets: Sunrise, Midday, Sunset, Night, Stormy
- Time-of-day control
- Sun position control
- Cloud density
- Atmospheric scattering

---

## 2. Ombres Temps Réel avec Shadow Mapping

### Architecture
```
ui/widgets/advanced_terrain_viewer.py
├── AdvancedTerrainViewer (QOpenGLWidget)
│   ├── Shadow Mapping Pipeline
│   │   ├── ShadowMapFBO (1024x1024 depth texture)
│   │   ├── Depth pass (light POV)
│   │   └── Render pass (camera POV + shadow test)
│   ├── Custom Shaders
│   │   ├── terrain_vertex.glsl
│   │   ├── terrain_fragment.glsl
│   │   ├── shadow_depth.vert
│   │   └── shadow_depth.frag
│   └── Lighting System
│       ├── Directional light (sun)
│       ├── Shadow bias
│       └── PCF filtering
```

### Shaders GLSL

**Terrain Vertex Shader:**
```glsl
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec3 Color;
out vec4 FragPosLightSpace;

void main() {
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    Color = color;
    FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
```

**Terrain Fragment Shader:**
```glsl
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec3 Color;
in vec4 FragPosLightSpace;

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform sampler2D shadowMap;
uniform float ambientStrength;
uniform float shadowBias;

out vec4 FragColor;

float ShadowCalculation(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.z > 1.0) return 0.0;

    float currentDepth = projCoords.z;

    // PCF (Percentage Closer Filtering)
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - shadowBias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    return shadow;
}

void main() {
    // Ambient
    vec3 ambient = ambientStrength * Color;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDirNorm = normalize(-lightDir);
    float diff = max(dot(norm, lightDirNorm), 0.0);
    vec3 diffuse = diff * Color;

    // Shadow
    float shadow = ShadowCalculation(FragPosLightSpace);
    vec3 lighting = ambient + (1.0 - shadow) * diffuse;

    // Fog
    float distance = length(viewPos - FragPos);
    float fogFactor = exp(-0.00005 * distance * distance);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    vec3 fogColor = vec3(0.7, 0.8, 0.9);
    vec3 finalColor = mix(fogColor, lighting, fogFactor);

    FragColor = vec4(finalColor, 1.0);
}
```

### Pipeline
1. **Shadow Pass:**
   - Bind shadow FBO
   - Render terrain from light POV
   - Store depth in texture

2. **Render Pass:**
   - Bind default framebuffer
   - Render terrain from camera POV
   - Sample shadow map
   - Calculate shadow factor
   - Apply lighting

### Performance
- Shadow map: 2048x2048 (quality) ou 1024x1024 (performance)
- PCF kernel: 3x3 (quality) ou single sample (performance)
- Cascade shadow maps pour grandes distances (optionnel)

---

## 3. Caméra FPS Complète

### Architecture
```
core/camera/fps_camera.py
├── FPSCamera
│   ├── Position (x, y, z)
│   ├── Rotation (yaw, pitch, roll)
│   ├── Movement
│   │   ├── Forward/Backward (W/S)
│   │   ├── Strafe Left/Right (A/D)
│   │   ├── Up/Down (Space/Shift)
│   │   └── Speed control
│   ├── Mouse Look
│   │   ├── Yaw (horizontal)
│   │   ├── Pitch (vertical)
│   │   └── Sensitivity
│   └── Collision
│       ├── Heightmap query
│       ├── Min height above terrain
│       └── Smooth interpolation
```

### Contrôles
```
WASD - Déplacement horizontal
Space - Monter
Shift - Descendre
Mouse - Rotation caméra
Scroll - Vitesse de déplacement
R - Reset position
C - Toggle collision
```

### Mathématiques
```python
# Direction vectors
front = Vector3(
    cos(yaw) * cos(pitch),
    sin(pitch),
    sin(yaw) * cos(pitch)
)

right = normalize(cross(front, world_up))
up = normalize(cross(right, front))

# View matrix
view = lookAt(position, position + front, up)
```

### Collision avec Terrain
```python
def get_terrain_height(x, z, heightmap):
    """Get interpolated height at (x, z)"""
    # Bilinear interpolation
    x_grid = x / terrain_scale * heightmap.shape[1]
    z_grid = z / terrain_scale * heightmap.shape[0]

    x0, x1 = floor(x_grid), ceil(x_grid)
    z0, z1 = floor(z_grid), ceil(z_grid)

    # Clamp to bounds
    # Bilinear interpolation
    return height

def update_camera_position(camera, heightmap):
    terrain_height = get_terrain_height(camera.x, camera.z, heightmap)
    min_height = terrain_height + camera_offset  # e.g., 2.0 meters

    if camera.y < min_height:
        camera.y = lerp(camera.y, min_height, 0.1)  # Smooth
```

---

## 4. Intégration UI

### Widget Principal
```
ui/widgets/ultimate_terrain_viewer.py
├── UltimateTerrainViewer (QMainWindow)
│   ├── OpenGL Viewport (AdvancedTerrainViewer)
│   ├── Control Panel
│   │   ├── Camera Mode (Orbit / FPS)
│   │   ├── Rendering Options
│   │   │   ├── Shadows On/Off
│   │   │   ├── Shadow Quality
│   │   │   ├── Fog On/Off
│   │   │   └── Wireframe
│   │   ├── Lighting
│   │   │   ├── Sun Position (azimuth/elevation)
│   │   │   ├── Ambient Strength
│   │   │   └── Shadow Bias
│   │   └── HDRI Skybox
│   │       ├── Load HDRI
│   │       ├── Generate New
│   │       ├── Time of Day
│   │       └── Exposure
│   └── Status Bar
│       ├── FPS Counter
│       ├── Camera Position
│       └── Performance Metrics
```

---

## 5. Dépendances

### Nouvelles
```python
# OpenGL moderne
ModernGL>=5.8.0  # ou PyOpenGL>=3.1.7

# HDR/EXR support
OpenEXR>=3.2.0
Imath>=3.1.9

# Diffusers pour HDRI AI
diffusers>=0.27.0
transformers>=4.38.0
accelerate>=0.27.0

# Maths
pyrr>=0.10.3  # Matrices/vectors OpenGL
```

### Installation
```bash
pip install moderngl PyOpenGL PyOpenGL-accelerate
pip install OpenEXR Imath
pip install diffusers transformers accelerate
pip install pyrr
```

---

## 6. Structure Fichiers

```
New_comfyui/
├── core/
│   ├── camera/
│   │   ├── __init__.py
│   │   └── fps_camera.py
│   └── rendering/
│       ├── hdri_generator.py
│       └── shaders/
│           ├── terrain_vertex.glsl
│           ├── terrain_fragment.glsl
│           ├── shadow_depth.vert
│           ├── shadow_depth.frag
│           ├── skybox_vertex.glsl
│           └── skybox_fragment.glsl
├── ui/
│   └── widgets/
│       ├── ultimate_terrain_viewer.py
│       └── hdri_generator_widget.py
└── examples/
    ├── example_ultimate_viewer.py
    └── example_hdri_generation.py
```

---

## 7. Plan d'Implémentation

### Phase 1: HDRI Generator (2-3 heures)
1. ✅ Create hdri_generator.py
2. ✅ Implement procedural skybox
3. ✅ Add AI enhancement (optional)
4. ✅ Export .hdr and .exr
5. ✅ Create GUI widget

### Phase 2: FPS Camera (1-2 heures)
1. ✅ Create fps_camera.py
2. ✅ Implement movement (WASD)
3. ✅ Implement mouse look
4. ✅ Add terrain collision
5. ✅ Test with existing viewer

### Phase 3: Shadow Mapping (3-4 heures)
1. ✅ Create shader files
2. ✅ Implement shadow FBO
3. ✅ Create AdvancedTerrainViewer (PyOpenGL/ModernGL)
4. ✅ Integrate shaders
5. ✅ Test shadow rendering
6. ✅ Optimize performance

### Phase 4: Ultimate Viewer (2-3 heures)
1. ✅ Create UltimateTerrainViewer
2. ✅ Integrate all features
3. ✅ Create control panel UI
4. ✅ Add HDRI skybox rendering
5. ✅ Test complete workflow

### Phase 5: Testing & Polish (1-2 heures)
1. ✅ Performance testing
2. ✅ Bug fixes
3. ✅ Documentation
4. ✅ Examples

**Total estimé: 9-14 heures de développement**

---

## 8. Risques et Mitigations

### Risque 1: Performance OpenGL
- **Mitigation**: LOD system, frustum culling, instancing
- **Fallback**: Quality presets (Low/Medium/High/Ultra)

### Risque 2: HDRI AI trop lent
- **Mitigation**: Génération asynchrone avec progress bar
- **Fallback**: Cache des HDRIs générés, presets procéduraux

### Risque 3: Compatibilité OpenGL
- **Mitigation**: Vérifier OpenGL version au démarrage
- **Fallback**: Fallback vers viewer basique si OpenGL < 3.3

### Risque 4: Collisions caméra complexes
- **Mitigation**: Collision simple heightmap-based
- **Fallback**: Toggle collision on/off

---

## 9. Métriques de Succès

### Performance
- ✅ 60 FPS @ 1024² terrain avec ombres (quality: Medium)
- ✅ 30 FPS @ 2048² terrain avec ombres (quality: High)
- ✅ Shadow map 2048x2048 sans lag

### Qualité
- ✅ Ombres douces et réalistes (PCF)
- ✅ HDRI 4096x2048 exportable
- ✅ Contrôles caméra FPS fluides (<50ms input lag)

### Fonctionnalité
- ✅ Génération HDRI en <30s (procédural) ou <2min (AI)
- ✅ Toggle ombres en temps réel
- ✅ Switch Orbit/FPS sans lag

---

## 10. Documentation

### Pour l'Utilisateur
- Guide d'utilisation Ultimate Viewer
- Tutoriel génération HDRI
- Contrôles caméra FPS
- Optimisation performance

### Pour le Développeur
- Architecture OpenGL
- Shaders documentation
- Extension du système d'ombres
- Ajout de nouveaux presets HDRI

---

**Status**: Ready for implementation
**Estimation**: 9-14 heures
**Priority**: High
**Complexity**: Advanced
