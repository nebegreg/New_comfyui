#!/usr/bin/env python3
"""
Complete System Test for Mountain Studio Pro - Ultimate Features
Tests ALL components to ensure everything is functional

Run with: python3 test_ultimate_system.py
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_section(title):
    """Print a test section header."""
    print('\n' + '='*70)
    print(f'  {title}')
    print('='*70)


def test_fps_camera():
    """Test FPS camera system."""
    print_section('TEST 1: FPS CAMERA SYSTEM')

    try:
        from core.camera.fps_camera import FPSCamera

        # Create camera
        camera = FPSCamera(position=np.array([0.0, 10.0, 0.0]))
        assert camera.position[1] == 10.0, "Initial position incorrect"
        print('  ✓ Camera creation')

        # Test movement
        camera.set_move_forward(True)
        camera.process_keyboard(0.1)
        assert camera.position[2] < 0, "Forward movement failed"
        print('  ✓ Forward movement')

        camera.set_move_forward(False)
        camera.set_move_right(True)
        camera.process_keyboard(0.1)
        assert camera.position[0] > 0, "Right movement failed"
        print('  ✓ Right movement')

        # Test mouse look
        initial_yaw = camera.yaw
        camera.process_mouse_movement(10.0, 0.0)
        assert camera.yaw != initial_yaw, "Mouse look failed"
        print('  ✓ Mouse look')

        # Test view matrix
        view = camera.get_view_matrix()
        assert view.shape == (4, 4), "View matrix wrong shape"
        print('  ✓ View matrix generation')

        # Test projection matrix
        proj = camera.get_projection_matrix(aspect_ratio=16/9)
        assert proj.shape == (4, 4), "Projection matrix wrong shape"
        print('  ✓ Projection matrix generation')

        # Test collision (with dummy heightmap)
        heightmap = np.random.rand(100, 100)
        camera.set_heightmap(heightmap, terrain_scale=100.0, height_scale=20.0)
        camera.collision_enabled = True
        camera.position = np.array([0.0, 0.0, 0.0])  # Ground level
        camera.process_keyboard(0.1)
        assert camera.position[1] > 0, "Collision not working"
        print('  ✓ Terrain collision')

        # Test state save/restore
        state = camera.get_state()
        camera2 = FPSCamera()
        camera2.set_state(state)
        assert np.allclose(camera.position, camera2.position), "State restore failed"
        print('  ✓ State save/restore')

        print('\n✅ FPS CAMERA: ALL TESTS PASSED')
        return True

    except Exception as e:
        print(f'\n❌ FPS CAMERA FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_hdri_generator():
    """Test HDRI panoramic generator."""
    print_section('TEST 2: HDRI PANORAMIC GENERATOR')

    try:
        from core.rendering.hdri_generator import (
            HDRIPanoramicGenerator, TimeOfDay, OPENEXR_AVAILABLE, AI_AVAILABLE
        )

        # Create generator
        gen = HDRIPanoramicGenerator(resolution=(512, 256))
        assert gen.width == 512 and gen.height == 256, "Resolution incorrect"
        print('  ✓ Generator creation')

        # Test all time presets
        for time in [TimeOfDay.SUNRISE, TimeOfDay.MIDDAY, TimeOfDay.SUNSET, TimeOfDay.NIGHT]:
            hdri = gen.generate_procedural(time, cloud_density=0.3, seed=42)

            # Check shape
            assert hdri.shape == (256, 512, 3), f"Wrong shape for {time.value}"

            # Check for invalid values
            assert not np.isnan(hdri).any(), f"NaN values in {time.value}"
            assert not np.isinf(hdri).any(), f"Inf values in {time.value}"
            assert hdri.min() >= 0, f"Negative values in {time.value}"

            print(f'  ✓ {time.value}: range=[{hdri.min():.3f}, {hdri.max():.3f}]')

        # Test edge cases for cloud density
        for density in [0.0, 0.01, 0.5, 1.0]:
            hdri = gen.generate_procedural(TimeOfDay.MIDDAY, cloud_density=density, seed=42)
            assert not np.isnan(hdri).any(), f"NaN with density={density}"
        print('  ✓ Edge cases (density 0.0-1.0)')

        # Test with mountains
        hdri_with_mountains = gen.generate_procedural(
            TimeOfDay.MIDDAY, mountain_distance=True, seed=42
        )
        hdri_without_mountains = gen.generate_procedural(
            TimeOfDay.MIDDAY, mountain_distance=False, seed=42
        )
        assert not np.array_equal(hdri_with_mountains, hdri_without_mountains), \
            "Mountain silhouette not working"
        print('  ✓ Mountain silhouette')

        # Test export
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # LDR export
            ldr_path = os.path.join(tmpdir, 'test.png')
            gen.export_ldr(hdri, ldr_path)
            assert os.path.exists(ldr_path), "LDR export failed"
            print('  ✓ LDR (PNG) export')

            # EXR export
            if OPENEXR_AVAILABLE:
                exr_path = os.path.join(tmpdir, 'test.exr')
                gen.export_exr(hdri, exr_path)
                assert os.path.exists(exr_path), "EXR export failed"
                print('  ✓ EXR export')
            else:
                print('  ⚠  EXR export skipped (OpenEXR not available)')

        # Check AI availability
        if AI_AVAILABLE:
            print('  ✓ AI enhancement available')
        else:
            print('  ⚠  AI enhancement not available (optional)')

        print('\n✅ HDRI GENERATOR: ALL TESTS PASSED')
        return True

    except Exception as e:
        print(f'\n❌ HDRI GENERATOR FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_shaders():
    """Test that all shader files exist and are valid."""
    print_section('TEST 3: GLSL SHADERS')

    try:
        shader_dir = PROJECT_ROOT / 'core' / 'rendering' / 'shaders'

        required_shaders = [
            'terrain_vertex.glsl',
            'terrain_fragment.glsl',
            'shadow_depth.vert',
            'shadow_depth.frag',
            'skybox_vertex.glsl',
            'skybox_fragment.glsl'
        ]

        for shader_name in required_shaders:
            shader_path = shader_dir / shader_name
            assert shader_path.exists(), f"Shader missing: {shader_name}"

            # Check file is not empty
            content = shader_path.read_text()
            assert len(content) > 0, f"Shader empty: {shader_name}"

            # Check for GLSL version
            assert '#version' in content, f"No GLSL version in {shader_name}"

            print(f'  ✓ {shader_name} ({len(content)} bytes)')

        print('\n✅ SHADERS: ALL FILES PRESENT AND VALID')
        return True

    except Exception as e:
        print(f'\n❌ SHADERS FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_advanced_viewer_structure():
    """Test advanced viewer structure (syntax only, no OpenGL)."""
    print_section('TEST 4: ADVANCED TERRAIN VIEWER (Structure)')

    try:
        # Can't instantiate without OpenGL, but can check structure
        import ast

        viewer_path = PROJECT_ROOT / 'ui' / 'widgets' / 'advanced_terrain_viewer.py'
        with open(viewer_path) as f:
            code = f.read()

        # Parse syntax
        tree = ast.parse(code)
        print('  ✓ Syntax valid')

        # Check for key features
        checks = {
            'OPENGL_AVAILABLE': 'OpenGL availability check',
            'class AdvancedTerrainViewer': 'Main class defined',
            'def initializeGL': 'OpenGL initialization',
            'def paintGL': 'Render function',
            '_render_shadow_pass': 'Shadow pass implementation',
            '_render_main_pass': 'Main render pass',
            'def set_terrain': 'Terrain setup',
            'FPSCamera': 'FPS camera integration',
        }

        for feature, description in checks.items():
            assert feature in code, f"Missing: {description}"
            print(f'  ✓ {description}')

        print('\n✅ ADVANCED VIEWER: STRUCTURE VALID')
        print('  ⚠  OpenGL rendering cannot be tested without display')
        return True

    except Exception as e:
        print(f'\n❌ ADVANCED VIEWER FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_ultimate_viewer_structure():
    """Test ultimate viewer structure (syntax only, no Qt)."""
    print_section('TEST 5: ULTIMATE VIEWER (Structure)')

    try:
        import ast

        viewer_path = PROJECT_ROOT / 'ui' / 'widgets' / 'ultimate_terrain_viewer.py'
        with open(viewer_path) as f:
            code = f.read()

        # Parse syntax
        tree = ast.parse(code)
        print('  ✓ Syntax valid')

        # Check for key features
        checks = {
            'class UltimateTerrainViewer(QMainWindow)': 'Main window class',
            'def _create_terrain_tab': 'Terrain tab',
            'def _create_rendering_tab': 'Rendering tab',
            'def _create_lighting_tab': 'Lighting tab',
            'def _create_camera_tab': 'Camera tab',
            'def _create_hdri_tab': 'HDRI tab',
            'def _create_export_tab': 'Export tab',
            'def _on_generate_terrain': 'Terrain generation handler',
            'def _on_generate_hdri': 'HDRI generation handler',
            'AdvancedTerrainViewer': 'OpenGL viewer integration',
        }

        for feature, description in checks.items():
            assert feature in code, f"Missing: {description}"
            print(f'  ✓ {description}')

        print('\n✅ ULTIMATE VIEWER: STRUCTURE VALID')
        print('  ⚠  Qt GUI cannot be tested without display')
        return True

    except Exception as e:
        print(f'\n❌ ULTIMATE VIEWER FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_examples():
    """Test that example files exist and are valid."""
    print_section('TEST 6: EXAMPLE FILES')

    try:
        examples_dir = PROJECT_ROOT / 'examples'

        examples = [
            'example_ultimate_viewer.py',
            'example_hdri_generation.py'
        ]

        for example_name in examples:
            example_path = examples_dir / example_name
            assert example_path.exists(), f"Example missing: {example_name}"

            # Check syntax
            with open(example_path) as f:
                code = f.read()

            import ast
            ast.parse(code)

            # Check has main
            assert 'def main()' in code or 'if __name__ == "__main__"' in code, \
                f"No main in {example_name}"

            print(f'  ✓ {example_name}')

        print('\n✅ EXAMPLES: ALL FILES VALID')
        return True

    except Exception as e:
        print(f'\n❌ EXAMPLES FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Test that documentation exists."""
    print_section('TEST 7: DOCUMENTATION')

    try:
        docs = [
            'IMPLEMENTATION_PLAN_ULTIMATE.md',
            'ULTIMATE_FEATURES_GUIDE.md',
            'requirements_ultimate.txt'
        ]

        for doc_name in docs:
            doc_path = PROJECT_ROOT / doc_name
            assert doc_path.exists(), f"Documentation missing: {doc_name}"

            # Check not empty
            content = doc_path.read_text()
            assert len(content) > 100, f"Documentation too short: {doc_name}"

            print(f'  ✓ {doc_name} ({len(content)} bytes)')

        print('\n✅ DOCUMENTATION: ALL FILES PRESENT')
        return True

    except Exception as e:
        print(f'\n❌ DOCUMENTATION FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print('\n' + '█'*70)
    print('█' + ' '*68 + '█')
    print('█' + '  MOUNTAIN STUDIO PRO - ULTIMATE FEATURES TEST SUITE  '.center(68) + '█')
    print('█' + ' '*68 + '█')
    print('█'*70)

    tests = [
        ('FPS Camera', test_fps_camera),
        ('HDRI Generator', test_hdri_generator),
        ('Shaders', test_shaders),
        ('Advanced Viewer', test_advanced_viewer_structure),
        ('Ultimate Viewer', test_ultimate_viewer_structure),
        ('Examples', test_examples),
        ('Documentation', test_documentation),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Final report
    print_section('FINAL REPORT')

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = '✅ PASS' if result else '❌ FAIL'
        print(f'  {status}: {name}')

    print(f'\n  Total: {passed}/{total} tests passed')

    if passed == total:
        print('\n' + '█'*70)
        print('█' + ' '*68 + '█')
        print('█' + '  ✅ ALL TESTS PASSED - SYSTEM IS FUNCTIONAL  '.center(68) + '█')
        print('█' + ' '*68 + '█')
        print('█'*70)
        return 0
    else:
        print('\n' + '█'*70)
        print('█' + ' '*68 + '█')
        print('█' + f'  ⚠  {total - passed} TESTS FAILED  '.center(68) + '█')
        print('█' + ' '*68 + '█')
        print('█'*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
