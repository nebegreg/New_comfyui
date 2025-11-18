# 🏔️ Mountain Studio Ultimate

**Ultra-Realistic Mountain Terrain Generator**

Version 3.0 - Standalone Edition

---

## ✨ Features

### Ultra-Realistic Terrain Generation
- **Multi-octave Perlin noise** for organic base terrain
- **Ridge noise** for sharp mountain peaks and crêtes
- **Domain warping** for natural-looking distortions
- **Hydraulic erosion** using shallow-water fluid model
  - Rain simulation
  - Water flow and sediment transport
  - Erosion and deposition
  - Realistic river valleys and canyons
- **Thermal erosion** for cliff decomposition
  - Talus slope formation
  - Natural rockfall simulation

### Professional Interface
- **Real-time 3D preview** with PyQtGraph OpenGL
- **2D heightmap preview**
- **Intuitive parameter controls**
  - Basic: Resolution, scale, detail, persistence, lacunarity
  - Advanced: Ridge influence, domain warping
  - Erosion: Hydraulic and thermal iterations
- **Progress tracking** with detailed logs
- **Professional export options**
  - PNG (16-bit heightmap)
  - RAW (16-bit binary for game engines)

### Standalone Application
- **No external module dependencies** on custom code
- **Single file** - no import errors
- **Cross-platform** (Windows, Linux, macOS)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install PySide6 numpy scipy pillow pyqtgraph PyOpenGL
```

### Run Application

**Option 1: Simple launcher**
```bash
./run_mountain_studio.sh
```

**Option 2: Direct Python**
```bash
python3 mountain_studio_ultimate.py
```

**Option 3: From any directory**
```bash
python3 /path/to/mountain_studio_ultimate.py
```

---

## 📖 User Guide

### 1. Basic Parameters

**Resolution**: Terrain size (256x256 to 2048x2048)
- **256x256**: Fast generation, preview quality
- **512x512**: Good balance (recommended for testing)
- **1024x1024**: High quality
- **2048x2048**: Production quality (slow)

**Scale**: Base frequency of terrain features
- **Low (10-50)**: Large, smooth features
- **Medium (50-100)**: Balanced (recommended: 100)
- **High (100-200)**: Small, detailed features

**Detail (Octaves)**: Number of noise layers
- **1-3**: Simple, smooth terrain
- **4-8**: Realistic detail (recommended: 8)
- **9-12**: Very detailed (slower)

**Persistence**: Amplitude decrease per octave
- **0.1-0.3**: Smooth, gentle hills
- **0.4-0.6**: Balanced mountains (recommended: 0.5)
- **0.7-0.9**: Sharp, dramatic peaks

**Lacunarity**: Frequency increase per octave
- **1.5-1.9**: Smooth transitions
- **2.0-2.5**: Realistic (recommended: 2.0)
- **2.6-3.5**: Highly detailed

### 2. Advanced Parameters

**Ridge Influence** (0.0-1.0): Controls sharp mountain peaks
- **0.0**: No ridges (smooth terrain)
- **0.2-0.5**: Realistic mountains (recommended: 0.4)
- **0.6-1.0**: Very sharp, dramatic ridges

**Domain Warping** (0.0-1.0): Organic distortion
- **0.0**: No warping (regular grid)
- **0.2-0.4**: Natural look (recommended: 0.3)
- **0.5-1.0**: Heavily distorted

### 3. Erosion Parameters

**Hydraulic Erosion Iterations**:
- **0**: No water erosion
- **20-50**: Subtle erosion (recommended: 50)
- **100-200**: Heavy erosion (canyons, deep valleys)

**Thermal Erosion Iterations**:
- **0**: No cliff erosion
- **3-5**: Realistic slopes (recommended: 5)
- **10-20**: Very smooth slopes

### 4. Generation Process

1. **Configure parameters** in the tabs
2. **Click "Generate Ultra-Realistic Terrain"**
3. **Wait** (progress bar shows status)
4. **View results** in 3D and 2D previews
5. **Export** if satisfied

Generation time:
- 256x256: ~5-10 seconds
- 512x512: ~15-30 seconds
- 1024x1024: ~1-2 minutes
- 2048x2048: ~5-10 minutes

### 5. Export Options

**PNG Export (16-bit)**:
- Standard image format
- 16-bit grayscale (65,536 height levels)
- Compatible with most software
- Lossless compression

**RAW Export (16-bit)**:
- Binary format
- Direct heightmap data
- For game engines (Unity, Unreal, etc.)
- No compression, maximum quality

---

## 🎨 Recommended Presets

### Gentle Hills
```
Resolution: 512x512
Scale: 150
Octaves: 6
Persistence: 0.4
Lacunarity: 2.0
Ridge: 0.1
Warp: 0.2
Hydraulic: 30
Thermal: 3
```

### Realistic Mountains (Default)
```
Resolution: 512x512
Scale: 100
Octaves: 8
Persistence: 0.5
Lacunarity: 2.0
Ridge: 0.4
Warp: 0.3
Hydraulic: 50
Thermal: 5
```

### Dramatic Peaks
```
Resolution: 512x512
Scale: 80
Octaves: 10
Persistence: 0.6
Lacunarity: 2.2
Ridge: 0.7
Warp: 0.4
Hydraulic: 40
Thermal: 7
```

### Grand Canyon Style
```
Resolution: 1024x1024
Scale: 120
Octaves: 8
Persistence: 0.5
Lacunarity: 2.0
Ridge: 0.3
Warp: 0.2
Hydraulic: 150
Thermal: 10
```

---

## 🔬 Technical Details

### Algorithms

**Perlin Noise**:
- Gradient-based coherent noise
- Smooth interpolation (cubic)
- Multi-octave synthesis (fractal noise)

**Ridge Noise**:
- Inverted absolute noise: `1 - abs(2*noise - 1)`
- Creates sharp mountain ridges
- Sharpening with power function

**Domain Warping**:
- Coordinate space distortion
- Uses offset fields from Perlin noise
- Creates organic, natural patterns

**Hydraulic Erosion**:
- Shallow-water fluid model
- Rain → Flow → Erosion → Deposition cycle
- Sediment transport simulation
- Gravity-based water flow

**Thermal Erosion**:
- Talus angle calculation (default: 0.7 ≈ 35°)
- Material transfer down slopes
- Cliff face decomposition
- Scree/talus formation

### Performance

Optimizations:
- NumPy vectorization where possible
- Background thread for generation
- Efficient neighbor iteration
- Clipping to avoid overflow

Memory usage:
- 256x256: ~1 MB
- 512x512: ~2 MB
- 1024x1024: ~8 MB
- 2048x2048: ~32 MB

---

## 🛠️ Troubleshooting

### "No module named 'PySide6'"
```bash
pip install PySide6
```

### "PyOpenGL not available"
```bash
pip install PyOpenGL PyOpenGL_accelerate
```

### "3D preview not working"
- Install PyQtGraph: `pip install pyqtgraph`
- Install OpenGL: `pip install PyOpenGL`
- 2D preview will still work

### Slow generation
- Reduce resolution (512x512 instead of 2048x2048)
- Reduce octaves (6 instead of 12)
- Reduce erosion iterations (30 instead of 100)

### Not realistic enough
- **Increase erosion**: Try 100-150 hydraulic iterations
- **Adjust ridge influence**: 0.5-0.7 for sharper peaks
- **Add domain warping**: 0.4-0.5 for more organic look
- **Increase octaves**: 10-12 for more detail

### Too noisy/chaotic
- **Decrease octaves**: 4-6 for smoother terrain
- **Decrease persistence**: 0.3-0.4 for gentler slopes
- **Increase thermal erosion**: 10-15 for smoother slopes

---

## 📚 Additional Resources

### Terrain Generation Theory
- [World Machine Documentation](https://www.world-machine.com/)
- [GPU Gems 3: Terrain Generation](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu)
- [Perlin Noise FAQ](https://adrianb.io/2014/08/09/perlinnoise.html)

### Game Engine Integration
- **Unity**: Import as 16-bit RAW, set resolution in import settings
- **Unreal Engine**: Import as 16-bit RAW, create Landscape
- **Blender**: Import as 16-bit PNG, use as displacement map

### Heightmap Format
- **PNG**: Standard image, easy to preview
- **RAW**: Binary 16-bit unsigned int, little-endian
- Value range: 0-65535 (0.0-1.0 normalized)

---

## 🎯 Tips & Tricks

1. **Start with low resolution** (256-512) for testing parameters
2. **Use presets** as starting points, then tweak
3. **Increase erosion** for more realistic features
4. **Ridge influence** = mountain peaks sharpness
5. **Domain warping** = organic, natural look
6. **Export to PNG** for easy preview in image viewers
7. **Export to RAW** for game engines (highest quality)
8. **Random seed** = different terrain with same parameters

---

## 🐛 Known Issues

- 3D preview may be slow on large terrains (>1024)
  - Solution: Subsample to 200x200 for display
- Very high erosion iterations (>200) may be slow
  - Solution: Use 50-100 for good balance
- RAW export has no header (import settings needed)
  - Solution: Remember resolution when importing

---

## 📝 Version History

**v3.0** (2025-01-18) - Ultimate Edition
- Complete rewrite as standalone application
- Added ultra-realistic algorithms
- Hydraulic erosion with shallow-water model
- Thermal erosion with talus slopes
- Ridge noise for mountain peaks
- Domain warping for organic look
- Professional export options
- Real-time previews

**v2.1** - P0 Improvements
- Smooth camera movement
- Enhanced HDRI quality
- UX polish

**v2.0** - Ultimate Features
- Advanced terrain algorithms
- ComfyUI integration
- 3D preview

**v1.0** - Initial Release
- Basic Perlin noise generation

---

## 📧 Support

For issues, questions, or suggestions:
- Check this README first
- Review the Troubleshooting section
- Check the logs in the application
- Verify all dependencies are installed

---

## 🏆 Credits

**Algorithms based on:**
- World Machine (terrain generation industry standard)
- GPU Gems 3 (NVIDIA)
- Research papers on procedural terrain generation (2024)

**Technologies:**
- Python 3
- PySide6 (Qt for Python)
- NumPy (numerical computing)
- SciPy (scientific computing)
- PyQtGraph (3D visualization)

---

## 📜 License

MIT License - Free to use, modify, and distribute

---

**🏔️ Happy Mountain Creating! 🏔️**
