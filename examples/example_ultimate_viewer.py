#!/usr/bin/env python3
"""
Example: Ultimate Terrain Viewer
Mountain Studio Pro

Demonstrates:
- Advanced OpenGL viewer with shadows
- FPS camera controls
- Real-time rendering
- HDRI generation
- Complete UI integration

Usage:
    python examples/example_ultimate_viewer.py

Controls:
    WASD - Move camera
    Space/Shift - Up/Down
    Mouse - Look around (click to capture)
    R - Reset camera
    C - Toggle collision
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PySide6.QtWidgets import QApplication

from ui.widgets.ultimate_terrain_viewer import UltimateTerrainViewer
from core.terrain.advanced_algorithms import spectral_synthesis, stream_power_erosion


def main():
    """Run ultimate viewer with example terrain."""
    app = QApplication(sys.argv)

    # Create viewer
    viewer = UltimateTerrainViewer()
    viewer.show()

    # Generate example terrain
    print("Generating example terrain (Alps preset)...")
    terrain_size = 512

    # Generate base terrain with spectral synthesis
    heightmap = spectral_synthesis(terrain_size, beta=2.2, seed=42)

    # Apply erosion for realism
    heightmap = stream_power_erosion(
        heightmap,
        iterations=50,
        K_erosion=0.015,
        m_area_exp=0.5,
        n_slope_exp=1.0
    )

    # Set in viewer
    viewer._current_heightmap = heightmap
    viewer._update_terrain()

    print("Terrain loaded!")
    print("\nControls:")
    print("  WASD - Move camera")
    print("  Space/Shift - Up/Down")
    print("  Mouse - Look around (click in viewport to capture mouse)")
    print("  R - Reset camera")
    print("  C - Toggle terrain collision")
    print("\nExplore the tabs for more options!")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
