#!/usr/bin/env python3
"""
Test Complet du Système Mountain Studio Pro

Vérifie que tous les modules fonctionnent correctement.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    print("="*70)
    print("TEST 1: IMPORTS")
    print("="*70)

    tests = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("PIL", "Pillow"),
        ("core.terrain.advanced_algorithms", "Advanced Algorithms"),
        ("core.terrain.heightmap_generator_v2", "Heightmap Generator V2"),
        ("core.rendering.pbr_texture_generator", "PBR Generator"),
        ("core.export.professional_exporter", "Professional Exporter"),
        ("core.ai.comfyui_installer", "ComfyUI Installer"),
    ]

    failed = []

    for module_name, display_name in tests:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            failed.append(display_name)

    if failed:
        print(f"\n❌ {len(failed)} imports failed")
        return False
    else:
        print(f"\n✅ All imports successful")
        return True


def test_terrain_generation():
    """Test terrain generation"""
    print("\n" + "="*70)
    print("TEST 2: TERRAIN GENERATION")
    print("="*70)

    try:
        from core.terrain.advanced_algorithms import (
            spectral_synthesis,
            stream_power_erosion,
            glacial_erosion,
            combine_algorithms,
            MOUNTAIN_PRESETS
        )

        # Test spectral synthesis
        print("\n  Testing Spectral Synthesis...")
        terrain = spectral_synthesis(256, beta=2.0, seed=42)
        assert terrain.shape == (256, 256), "Wrong shape"
        assert 0 <= terrain.min() <= terrain.max() <= 1, "Wrong range"
        print(f"    ✓ Spectral synthesis: shape={terrain.shape}, range=[{terrain.min():.3f}, {terrain.max():.3f}]")

        # Test stream power erosion
        print("\n  Testing Stream Power Erosion...")
        eroded = stream_power_erosion(terrain.copy(), iterations=20)
        assert eroded.shape == (256, 256), "Wrong shape"
        print(f"    ✓ Stream power erosion: range=[{eroded.min():.3f}, {eroded.max():.3f}]")

        # Test glacial erosion
        print("\n  Testing Glacial Erosion...")
        glaciated = glacial_erosion(eroded.copy(), altitude_threshold=0.7, strength=0.3)
        assert glaciated.shape == (256, 256), "Wrong shape"
        print(f"    ✓ Glacial erosion: range=[{glaciated.min():.3f}, {glaciated.max():.3f}]")

        # Test combined (Alps)
        print("\n  Testing Combined Generation (Alps)...")
        alps = combine_algorithms(256, **MOUNTAIN_PRESETS['alps'], seed=42)
        assert alps.shape == (256, 256), "Wrong shape"
        print(f"    ✓ Alps terrain: range=[{alps.min():.3f}, {alps.max():.3f}]")

        print("\n✅ All terrain tests passed")
        return terrain

    except Exception as e:
        print(f"\n❌ Terrain generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pbr_generation(heightmap):
    """Test PBR texture generation"""
    print("\n" + "="*70)
    print("TEST 3: PBR TEXTURE GENERATION")
    print("="*70)

    try:
        from core.rendering.pbr_texture_generator import PBRTextureGenerator

        generator = PBRTextureGenerator(resolution=256)

        print("\n  Generating PBR textures...")
        pbr_textures = generator.generate_from_heightmap(
            heightmap,
            material_type='rock',
            make_seamless=True,
            detail_level=1.0
        )

        # Verify all maps
        expected_maps = ['diffuse', 'normal', 'roughness', 'ao', 'height', 'metallic']
        for map_name in expected_maps:
            assert map_name in pbr_textures, f"Missing map: {map_name}"
            print(f"    ✓ {map_name}: {pbr_textures[map_name].shape}")

        print("\n✅ PBR generation passed")
        return pbr_textures

    except Exception as e:
        print(f"\n❌ PBR generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_export(heightmap, pbr_textures):
    """Test export functionality"""
    print("\n" + "="*70)
    print("TEST 4: EXPORT")
    print("="*70)

    try:
        from core.export.professional_exporter import ProfessionalExporter
        from core.terrain.heightmap_generator import HeightmapGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\n  Export directory: {tmpdir}")

            # Generate derivative maps
            print("\n  Generating derivative maps...")
            terrain_gen = HeightmapGenerator(256, 256)

            normal_map = terrain_gen.generate_normal_map(heightmap=heightmap, strength=1.0)
            depth_map = terrain_gen.generate_depth_map(heightmap=heightmap)
            ao_map = terrain_gen.generate_ambient_occlusion(heightmap=heightmap, samples=8)

            print(f"    ✓ Normal map: {normal_map.shape}")
            print(f"    ✓ Depth map: {depth_map.shape}")
            print(f"    ✓ AO map: {ao_map.shape}")

            # Export for Flame
            print("\n  Exporting for Autodesk Flame...")
            exporter = ProfessionalExporter(tmpdir)

            exported_files = exporter.export_for_autodesk_flame(
                heightmap=heightmap,
                normal_map=normal_map,
                depth_map=depth_map,
                ao_map=ao_map,
                diffuse_map=pbr_textures['diffuse'],
                roughness_map=pbr_textures['roughness'],
                splatmaps=None,
                tree_instances=None,
                mesh_subsample=2,
                scale_y=50.0
            )

            print(f"\n    ✓ Exported {len(exported_files)} files:")
            for key, path in list(exported_files.items())[:5]:
                size_kb = Path(path).stat().st_size / 1024
                print(f"      - {key}: {size_kb:.1f} KB")

        print("\n✅ Export tests passed")
        return True

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comfyui_installer():
    """Test ComfyUI installer (without actual installation)"""
    print("\n" + "="*70)
    print("TEST 5: COMFYUI INSTALLER")
    print("="*70)

    try:
        from core.ai.comfyui_installer import ComfyUIInstaller

        installer = ComfyUIInstaller()

        # Get recommended models
        print("\n  Checking recommended models...")
        models = installer.get_recommended_models()
        print(f"    ✓ {len(models)} models available:")
        for model in models[:3]:
            print(f"      - {model.name} ({model.size_mb} MB)")

        # Get recommended nodes
        print("\n  Checking recommended custom nodes...")
        nodes = installer.get_recommended_custom_nodes()
        print(f"    ✓ {len(nodes)} nodes available:")
        for node in nodes[:3]:
            print(f"      - {node.name}")

        print("\n✅ ComfyUI installer tests passed")
        return True

    except Exception as e:
        print(f"\n❌ ComfyUI installer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmarks():
    """Performance benchmarks"""
    print("\n" + "="*70)
    print("TEST 6: PERFORMANCE BENCHMARKS")
    print("="*70)

    try:
        from core.terrain.advanced_algorithms import spectral_synthesis, stream_power_erosion
        from core.rendering.pbr_texture_generator import PBRTextureGenerator
        import time

        sizes = [256, 512]

        for size in sizes:
            print(f"\n  Benchmarking {size}x{size}:")

            # Spectral synthesis
            start = time.time()
            terrain = spectral_synthesis(size, beta=2.0, seed=42)
            t_spectral = time.time() - start
            print(f"    Spectral synthesis: {t_spectral:.3f}s")

            # Erosion
            start = time.time()
            eroded = stream_power_erosion(terrain, iterations=20)
            t_erosion = time.time() - start
            print(f"    Stream power erosion (20 iter): {t_erosion:.3f}s")

            # PBR generation
            start = time.time()
            generator = PBRTextureGenerator(resolution=size)
            pbr = generator.generate_from_heightmap(terrain, material_type='rock')
            t_pbr = time.time() - start
            print(f"    PBR generation (6 maps): {t_pbr:.3f}s")

            total = t_spectral + t_erosion + t_pbr
            print(f"    Total: {total:.3f}s")

        print("\n✅ Performance benchmarks complete")
        return True

    except Exception as e:
        print(f"\n❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MOUNTAIN STUDIO PRO - SYSTEM TEST" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    print()

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Terrain Generation
    heightmap = test_terrain_generation()
    results.append(("Terrain Generation", heightmap is not None))

    if heightmap is not None:
        # Test 3: PBR Generation
        pbr_textures = test_pbr_generation(heightmap)
        results.append(("PBR Generation", pbr_textures is not None))

        if pbr_textures is not None:
            # Test 4: Export
            results.append(("Export", test_export(heightmap, pbr_textures)))
    else:
        results.append(("PBR Generation", False))
        results.append(("Export", False))

    # Test 5: ComfyUI Installer
    results.append(("ComfyUI Installer", test_comfyui_installer()))

    # Test 6: Performance
    results.append(("Performance", test_performance_benchmarks()))

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print("\n" + "="*70)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print("="*70)

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! System is fully functional.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} tests failed. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
