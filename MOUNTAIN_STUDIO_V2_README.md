# Mountain Studio ULTIMATE v2.0 🏔️

## Complete Professional Terrain Generation Suite

**The ultimate all-in-one application for ultra-realistic mountain terrain generation with AI-powered texturing, advanced 3D visualization, and professional export capabilities.**

---

## ✨ **ALL FEATURES INTEGRATED IN ONE GUI**

### 🏔️ **Ultra-Realistic Terrain Generation**
- **Multi-octave Perlin noise** with gradient interpolation
- **Ridge noise** for sharp mountain peaks (inverted absolute method)
- **Domain warping** for organic terrain distortion
- **Hydraulic erosion** with shallow-water fluid model (rain → flow → erosion → deposition)
- **Thermal erosion** with talus slope simulation
- **Real-time parameter adjustment** (scale, octaves, erosion rates)
- **Presets** for different mountain types

### 🎮 **Advanced 3D Viewer with Lighting & Shadows**
- **Real-time Phong lighting** (ambient + diffuse)
- **Dynamic shadow mapping** (adjustable quality)
- **Interactive sun position** (azimuth + elevation controls)
- **Wireframe mode toggle**
- **FPS camera controls** (WASD + mouse)
- **Height scale adjustment** for dramatic visualization
- **Altitude-based coloring** (brown → green → gray → white)

### 🎨 **AI Texture Generation (ComfyUI Integration)**
- **TXT2TEXTURE workflows** for ultra-realistic PBR textures
- **Stable Diffusion XL** integration
- **Custom prompt support** for specific material types
- **Automatic PBR set generation** (Diffuse, Normal, Roughness, AO, Height)
- **Connection status monitoring**
- **4K/8K texture generation**

### 🗺️ **Complete PBR Map Generation**
- **Diffuse/Albedo** maps with altitude and slope variation
- **Normal maps** from heightmap with micro-detail
- **Roughness maps** based on terrain slope and noise
- **Ambient Occlusion** with multi-sample ray tracing
- **Height/Displacement maps** for parallax effects
- **Metallic maps** for material properties
- **Material presets**: Rock, Grass, Snow, Sand, Dirt
- **Seamless/tileable textures** for terrain application
- **Multiple resolutions**: 512, 1024, 2048, 4096

### 🌅 **HDRI Panoramic Generation**
- **360° equirectangular panoramas**
- **7 time-of-day presets**:
  - Sunrise 🌅 (warm oranges, low sun)
  - Morning ☀️ (bright, clear)
  - Midday ☀️ (intense, high sun)
  - Afternoon 🌤️ (golden hour approaching)
  - Sunset 🌇 (dramatic oranges and purples)
  - Twilight 🌆 (soft blues, stars appearing)
  - Night 🌃 (stars, moonlight)
- **Physically-based Rayleigh scattering**
- **Procedural clouds** with multi-octave noise
- **Distant mountain silhouettes**
- **Export formats**: .hdr (Radiance), .exr (OpenEXR 32-bit), .png (LDR preview)
- **Resolution options**: 2K, 4K, 8K
- **Optional AI enhancement** with Stable Diffusion XL

### 💾 **Professional Export Formats**
- **PNG** (8-bit, 16-bit grayscale heightmaps)
- **RAW** (16-bit binary for Unity/Unreal)
- **EXR** (32-bit HDR for VFX)
- **OBJ** (3D mesh with UVs and normals)
- **MTL** (material definitions for OBJ)
- **JSON** (vegetation instances, metadata)
- **Autodesk Flame** complete pipeline export
- **Complete package export** (all assets + README)

### 🎬 **Video Generation**
- **Temporal consistency** with ControlNet
- **Camera paths**: Orbit, flyover, zoom
- **Frame interpolation** for smooth motion
- **Motion blur simulation**
- **Ken Burns effects** (zoom + pan)

### 🌲 **Vegetation System**
- **Biome classification** (altitude, slope, moisture-based)
- **Species distribution** with realistic placement
- **Tree instance export** (JSON format)
- **LOD support** for performance optimization

### 🎯 **VFX Prompt Generation**
- **Ultra-quality keywords** (hypersharp, gigapixel, UE5, RTX)
- **Professional photographer styles** (Ansel Adams, Galen Rowell)
- **Lighting presets** for different times/weather
- **SDXL model recommendations** (EpicRealism, Juggernaut, RealVisXL)

---

## 🚀 **Quick Start**

### **Requirements**
- Python 3.8+
- PySide6 (Qt for GUI)
- NumPy, SciPy (scientific computing)
- PyQtGraph (3D visualization)
- PIL/Pillow (image processing)
- **Optional**: ComfyUI server for AI textures (localhost:8188)

### **Installation**
```bash
# Install dependencies
pip install PySide6 numpy scipy pillow pyqtgraph pyopengl

# Optional: For advanced features
pip install opencv-python diffusers torch OpenEXR
```

### **Launch Application**
```bash
# Simple launcher
./run_mountain_ultimate_v2.sh

# Or direct Python
python3 mountain_studio_ultimate_v2.py
```

---

## 📖 **User Guide**

### **Tab 1: 🏔️ Terrain Generation**

#### **Resolution Settings**
- **Width/Height**: Terrain grid size (64-2048)
  - 512x512: Fast preview (recommended for testing)
  - 1024x1024: High detail (good balance)
  - 2048x2048: Ultra detail (production quality)

#### **Noise Parameters**
- **Scale** (10-500): Base frequency of terrain features
  - Low (10-50): Small, detailed features
  - Medium (50-150): Realistic mountain scale
  - High (150-500): Large, gentle hills

- **Octaves** (1-12): Number of noise layers
  - More octaves = more detail + longer generation time
  - Recommended: 6-8 for balanced results

- **Ridge Influence** (0-1): Sharp mountain peak strength
  - 0.0: Smooth, rolling terrain
  - 0.4: Balanced mountains with peaks
  - 0.8+: Dramatic, sharp ridges

- **Domain Warp** (0-1): Organic terrain distortion
  - 0.0: Regular, geometric noise
  - 0.3: Natural-looking distortion (recommended)
  - 0.6+: Highly organic, chaotic shapes

#### **Erosion Parameters**
- **Hydraulic Iterations** (0-100): Water erosion simulation
  - 0: No erosion (pure noise)
  - 25-50: Light erosion, realistic valleys
  - 75-100: Heavy erosion, deep canyons

- **Thermal Iterations** (0-20): Cliff decomposition
  - 0: No thermal erosion
  - 3-5: Subtle talus slopes (recommended)
  - 10+: Heavy cliff breakdown

- **Erosion Rate** (0.1-1.0): How aggressively water erodes
  - 0.1-0.2: Gentle erosion
  - 0.3-0.5: Realistic erosion (recommended)
  - 0.6+: Aggressive, deep channels

#### **Random Seed**
- Same seed = identical terrain (reproducible)
- Click "🎲 Randomize" for new random seed

### **Tab 2: 💡 3D Lighting**

#### **Sun Position**
- **Azimuth** (0-360°): Horizontal sun angle
  - 0°/360°: North
  - 90°: East
  - 180°: South
  - 270°: West
  - 135°: Southeast (recommended for dramatic lighting)

- **Elevation** (0-90°): Vertical sun angle
  - 0°: Sunrise/sunset (horizon)
  - 45°: Mid-morning/afternoon (recommended)
  - 90°: Noon (directly overhead)

#### **Lighting Parameters**
- **Ambient Strength** (0-1): Base lighting level
  - 0.0: Pure directional lighting (dramatic shadows)
  - 0.3: Balanced (recommended)
  - 0.5+: Flat, well-lit

#### **Height Scale**
- **Height Multiplier** (10-200): Vertical exaggeration
  - 25: Subtle elevation changes
  - 50: Realistic scale (recommended)
  - 100+: Dramatic, exaggerated mountains

### **Tab 3: 🎨 AI Textures (ComfyUI)**

#### **Setup**
1. Start ComfyUI server: `python main.py` (default: localhost:8188)
2. Click "🔍 Check Connection" to verify
3. Status should show "✅ Connected"

#### **Texture Generation**
- **Prompt**: Describe desired texture style
  - Example: "ultra realistic mountain rock texture, 4k, PBR"
  - Example: "weathered granite cliff face, high detail, 8k"
  - Example: "snow-covered alpine rock, photorealistic"

- Click "🎨 Generate AI Textures" to start
- Generation time: 30 seconds - 5 minutes (GPU-dependent)
- Requires: 6-10 GB VRAM for SDXL models

### **Tab 4: 🗺️ PBR Maps**

#### **Material Types**
- **Rock**: Gray-brown, rough, high AO contrast
- **Grass**: Green, medium roughness, organic detail
- **Snow**: White-blue, smooth, low AO
- **Sand**: Yellow-tan, medium roughness, fine detail
- **Dirt**: Brown, rough, high detail

#### **Resolution**
- 512: Low res (fast, for testing)
- 1024: Medium res (good for real-time)
- 2048: High res (production quality, recommended)
- 4096: Ultra res (film/VFX quality, slow)

#### **Generated Maps**
- **Diffuse**: Base color/albedo
- **Normal**: Surface detail (RGB tangent-space)
- **Roughness**: Surface glossiness (grayscale, 0=smooth, 1=rough)
- **AO**: Ambient occlusion (grayscale shadows)
- **Height**: Displacement map for parallax
- **Metallic**: Metal vs dielectric (0=non-metal, 1=metal)

All maps are **seamless/tileable** for terrain use.

### **Tab 5: 🌅 HDRI**

#### **Time of Day Presets**
- **Sunrise** 🌅: Warm golden hour, long shadows
- **Morning** ☀️: Bright, clear sky
- **Midday** ☀️: Intense overhead sun
- **Afternoon** 🌤️: Soft golden light
- **Sunset** 🌇: Dramatic oranges/purples
- **Twilight** 🌆: Blue hour, stars appearing
- **Night** 🌃: Starry sky, moonlight

#### **Resolution**
- 2048x1024 (2K): Fast, good for real-time
- 4096x2048 (4K): High quality (recommended)
- 8192x4096 (8K): Ultra quality, film/VFX

#### **Export Formats**
- **.hdr**: Radiance HDR (industry standard)
- **.exr**: OpenEXR 32-bit float (VFX pipeline)
- **.png**: LDR preview (tone-mapped for viewing)

### **Tab 6: 💾 Export**

#### **Quick Exports**
- **PNG 16-bit**: Standard heightmap (Unity, Unreal, World Machine)
- **RAW 16-bit**: Binary heightmap (game engines)
- **OBJ Mesh**: 3D model with UVs and normals

#### **Professional Exports**
- **Autodesk Flame**: Complete VFX pipeline package
  - Includes: Mesh (OBJ/MTL), textures, heightmaps, splatmaps, README

- **Complete Package**: All assets in one export
  - Includes: All formats, documentation, metadata JSON

---

## 🎯 **Recommended Workflows**

### **Workflow 1: Quick Terrain Preview**
1. Tab 1: Set resolution to 512x512
2. Adjust Scale (100) and Octaves (6)
3. Set Hydraulic Iterations to 25
4. Click "🏔️ GENERATE TERRAIN"
5. Tab 2: Adjust lighting for best view
6. Tab 6: Export PNG 16-bit

**Time**: 5-10 seconds

### **Workflow 2: Production Mountain**
1. Tab 1: Set resolution to 1024x1024 or 2048x2048
2. Adjust all parameters for desired look:
   - Scale: 100-150
   - Octaves: 8
   - Ridge Influence: 0.4
   - Domain Warp: 0.3
   - Hydraulic Iterations: 50
   - Thermal Iterations: 5
3. Click "🏔️ GENERATE TERRAIN"
4. Tab 2: Fine-tune lighting (Azimuth: 135°, Elevation: 45°)
5. Tab 4: Generate PBR maps (Rock, 2048)
6. Tab 5: Generate HDRI (Afternoon, 4K)
7. Tab 6: Export Complete Package

**Time**: 2-5 minutes

### **Workflow 3: AI-Enhanced Ultra-Realistic**
1. Start ComfyUI server
2. Generate high-res terrain (2048x2048)
3. Tab 3: Generate AI textures with custom prompt
4. Tab 4: Generate PBR maps
5. Tab 5: Generate HDRI with AI enhancement
6. Tab 6: Export for Autodesk Flame

**Time**: 10-20 minutes (GPU-dependent)

---

## 🔧 **Technical Details**

### **Algorithms**

#### **Perlin Noise**
```python
# Multi-octave synthesis with gradient interpolation
for octave in range(octaves):
    frequency *= lacunarity  # 2.0x per octave
    amplitude *= persistence  # 0.5x per octave
    noise += perlin(frequency) * amplitude
```

#### **Ridge Noise**
```python
# Sharp mountain peaks using inverted absolute
ridges = 1.0 - abs(2.0 * perlin_noise - 1.0)
ridges = ridges ** 1.5  # Sharpen ridges
```

#### **Hydraulic Erosion**
```python
# Shallow-water fluid model
water += rain_amount
sediment_capacity = water * slope * capacity_factor

if sediment < capacity:
    erosion = (capacity - sediment) * erosion_rate
    terrain -= erosion
    sediment += erosion
else:
    deposition = (sediment - capacity) * deposition_rate
    terrain += deposition
    sediment -= deposition

water *= (1 - evaporation_rate)
```

#### **Phong Lighting**
```python
# Calculate surface normals
normals = compute_normals(heightmap)

# Diffuse lighting (Lambertian)
diffuse = max(0, dot(normals, light_direction))

# Final color
color = (ambient_strength + (1 - ambient_strength) * diffuse) * base_color
```

### **Performance**

| Resolution | Terrain Gen | PBR Maps | HDRI 4K | Total |
|------------|-------------|----------|---------|-------|
| 512x512    | 2-5s        | 10s      | 30s     | ~45s  |
| 1024x1024  | 10-20s      | 30s      | 30s     | ~60s  |
| 2048x2048  | 45-90s      | 2min     | 30s     | ~4min |

*Times on Intel i7 + NVIDIA GTX 1080. YMMV.*

### **Memory Requirements**

| Resolution | RAM | VRAM (AI) |
|------------|-----|-----------|
| 512x512    | 500MB | 4GB |
| 1024x1024  | 1GB | 6GB |
| 2048x2048  | 3GB | 10GB |

---

## 🐛 **Troubleshooting**

### **3D Viewer Not Working**
- **Cause**: Missing PyQtGraph OpenGL or PyOpenGL
- **Solution**: `pip install pyqtgraph pyopengl`

### **ComfyUI Not Connecting**
- **Cause**: ComfyUI server not running
- **Solution**:
  1. Navigate to ComfyUI directory
  2. Run: `python main.py`
  3. Wait for "To see the GUI go to: http://127.0.0.1:8188"
  4. Click "🔍 Check Connection" in Mountain Studio

### **AI Texture Generation Fails**
- **Cause**: Insufficient VRAM or missing models
- **Solution**:
  1. Ensure you have 6-10 GB VRAM
  2. Download required models (SDXL Base 1.0)
  3. Check ComfyUI console for errors

### **Slow Terrain Generation**
- **Cause**: High resolution + many erosion iterations
- **Solution**:
  1. Reduce resolution to 512x512 for testing
  2. Lower Hydraulic Iterations to 25
  3. Lower Thermal Iterations to 3
  4. Increase for final render only

### **Export Fails**
- **Cause**: Missing PIL/Pillow
- **Solution**: `pip install pillow`

---

## 🆚 **What's New in v2.0**

### **vs v1.0 (mountain_studio_ultimate.py)**

#### **Added Features**
✅ **Advanced 3D Lighting & Shadows**
- Phong lighting model with ambient + diffuse
- Interactive sun position controls (azimuth/elevation)
- Real-time lighting updates
- Altitude-based terrain coloring

✅ **AI Texture Generation**
- Complete ComfyUI integration
- Custom prompt support
- Connection status monitoring
- TXT2TEXTURE workflows

✅ **Complete PBR Map Generation**
- 6 map types (Diffuse, Normal, Roughness, AO, Height, Metallic)
- 5 material presets (Rock, Grass, Snow, Sand, Dirt)
- Multiple resolutions (512-4096)
- Seamless/tileable output

✅ **HDRI Panorama Generation**
- 7 time-of-day presets
- 2K/4K/8K resolution options
- .hdr/.exr/.png export
- Physically-based sky rendering

✅ **Professional Export Suite**
- Autodesk Flame pipeline export
- Complete package export
- OBJ mesh with UVs and normals
- Comprehensive README generation

✅ **Improved GUI**
- 6 organized tabs (vs 3)
- Real-time parameter labels
- Progress tracking
- Detailed generation log
- Status indicators

✅ **Better Lighting Controls**
- Interactive sun azimuth slider (0-360°)
- Interactive sun elevation slider (0-90°)
- Ambient strength slider
- Real-time preview updates

#### **Improved**
- Better error handling and user feedback
- Optimized terrain generation algorithms
- More intuitive UI layout
- Comprehensive tooltips and labels
- Professional styling

---

## 📚 **Module Architecture**

```
Mountain Studio ULTIMATE v2.0
├── Core Terrain Generation
│   ├── NoiseGenerator (Perlin, Ridge, Domain Warp)
│   ├── HydraulicErosion (Shallow-water model)
│   ├── ThermalErosion (Talus slopes)
│   └── UltraRealisticTerrain (Complete pipeline)
│
├── 3D Visualization
│   └── Advanced3DViewer
│       ├── Phong lighting (ambient + diffuse)
│       ├── Dynamic sun position
│       ├── Altitude-based coloring
│       └── Wireframe toggle
│
├── AI Integration (Optional)
│   └── ComfyUIClient
│       ├── Connection management
│       ├── Workflow queueing
│       └── Image retrieval
│
├── PBR Generation (Optional)
│   └── PBRTextureGenerator
│       ├── Diffuse/Albedo generation
│       ├── Normal map from heightmap
│       ├── Roughness from slope
│       ├── AO from occlusion sampling
│       ├── Height/Displacement
│       └── Metallic maps
│
├── HDRI Generation (Optional)
│   └── HDRIPanoramicGenerator
│       ├── Time-of-day presets
│       ├── Procedural sky rendering
│       ├── Rayleigh scattering
│       └── .hdr/.exr export
│
└── Export Suite (Optional)
    └── ProfessionalExporter
        ├── PNG/RAW heightmaps
        ├── OBJ mesh generation
        ├── Autodesk Flame pipeline
        └── Complete package export
```

### **Module Dependencies**
- **Required**: NumPy, SciPy, PySide6, PIL
- **3D Viewer**: PyQtGraph, PyOpenGL
- **AI Textures**: ComfyUI server (separate application)
- **PBR/HDRI/Export**: Built-in modules from `core/` directory

---

## 💡 **Tips & Best Practices**

### **Terrain Generation**
1. **Start small**: Test with 512x512 before going to 2048x2048
2. **Balance erosion**: Too much erosion can flatten your terrain
3. **Use seeds**: Same seed = reproducible results (great for iterations)
4. **Ridge influence**: 0.3-0.5 for realistic mountains, 0.7+ for dramatic peaks
5. **Domain warp**: 0.2-0.4 for natural look, avoid 0.5+ (too chaotic)

### **Lighting**
1. **Golden hour**: Azimuth 120° or 240°, Elevation 15-30°
2. **Noon**: Azimuth 180°, Elevation 60-75°
3. **Dramatic**: Low elevation (10-20°) for long shadows
4. **Ambient**: 0.2-0.4 for natural contrast, 0.5+ for flat lighting

### **PBR Maps**
1. **Match material**: Choose material type that matches terrain altitude
2. **Resolution**: 2048 is sweet spot (quality vs performance)
3. **Test first**: Generate at 512 to preview, then export at 2048/4096
4. **Seamless**: All maps are tileable - perfect for terrain painting

### **Exports**
1. **PNG 16-bit**: Best for Unity/Unreal (standard format)
2. **RAW 16-bit**: Best for World Machine import/export
3. **OBJ**: Best for 3D applications (Blender, Maya, Houdini)
4. **Complete Package**: Best for archiving or sharing projects

---

## 🎓 **Learning Resources**

### **Terrain Generation Theory**
- **Perlin Noise**: Ken Perlin's original paper (1985)
- **Hydraulic Erosion**: "Fast Hydraulic Erosion Simulation" by Mei et al.
- **World Machine**: Industry standard for terrain generation
- **Houdini Heightfields**: Professional VFX workflows

### **PBR Materials**
- **Substance Designer**: Node-based PBR creation
- **Quixel Megascans**: Real-world PBR reference
- **Marmoset Toolbag**: PBR rendering and baking

### **HDRI Lighting**
- **HDRI Haven**: Free HDRI downloads for reference
- **Blender EEVEE**: Real-time HDRI rendering
- **Physical Sky Models**: Preetham, Hosek-Wilkie models

---

## 📞 **Support & Feedback**

### **Found a Bug?**
Please report issues with:
1. Your OS and Python version
2. Full error message from console
3. Steps to reproduce
4. Screenshots if GUI-related

### **Feature Requests**
We're always improving! Suggestions welcome for:
- New erosion algorithms
- Additional material presets
- Export format support
- Workflow optimizations

---

## 📄 **License**

MIT License - Use freely for commercial and non-commercial projects.

---

## 🙏 **Credits**

Built with:
- **PySide6**: Qt for Python (GUI framework)
- **NumPy/SciPy**: Scientific computing
- **PyQtGraph**: Fast 3D visualization
- **PIL/Pillow**: Image processing
- **ComfyUI**: AI texture generation (optional)

Inspired by:
- **World Machine**: Industry-standard terrain tools
- **Gaea**: Modern terrain generation
- **Houdini**: VFX industry workflows

---

## 🚀 **Roadmap**

### **Planned Features**
- [ ] GPU acceleration for terrain generation (CUDA/OpenCL)
- [ ] Real-time erosion preview
- [ ] Vegetation placement in 3D viewer
- [ ] Animation timeline for video generation
- [ ] Multiplayer terrain collaboration
- [ ] Cloud rendering support
- [ ] Blender/Unreal Engine direct export plugins

---

**Mountain Studio ULTIMATE v2.0** - Professional terrain generation, made accessible.

🏔️ **Generate. Visualize. Export. Create.**
