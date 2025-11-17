"""
Test script for Autodesk Flame export
Generates a simple terrain and exports it for Flame
"""

import numpy as np
from core.terrain.heightmap_generator import HeightmapGenerator
from core.export.professional_exporter import ProfessionalExporter
import os


def test_flame_export():
    """Test complet de l'export Autodesk Flame"""

    print("=" * 80)
    print("TEST EXPORT AUTODESK FLAME")
    print("=" * 80)

    # 1. Générer un terrain simple
    print("\n[1/4] Génération terrain 512x512...")
    terrain_gen = HeightmapGenerator(512, 512)

    heightmap = terrain_gen.generate(
        mountain_type='alpine',
        scale=100.0,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        seed=42,
        apply_hydraulic_erosion=True,
        apply_thermal_erosion=True,
        erosion_iterations=10000,  # Rapide pour test
        domain_warp_strength=0.3,
        use_ridged_multifractal=True
    )
    print(f"✓ Heightmap généré: {heightmap.shape}")

    # 2. Générer maps dérivées
    print("\n[2/4] Génération normal map, depth map, AO...")
    normal_map = terrain_gen.generate_normal_map(strength=1.0)
    depth_map = terrain_gen.generate_depth_map()
    ao_map = terrain_gen.generate_ambient_occlusion(samples=8)

    print(f"✓ Normal map: {normal_map.shape}")
    print(f"✓ Depth map: {depth_map.shape}")
    print(f"✓ AO map: {ao_map.shape}")

    # 3. Créer dossier export
    print("\n[3/4] Préparation export...")
    export_dir = "test_flame_export_output"
    os.makedirs(export_dir, exist_ok=True)
    print(f"✓ Dossier: {export_dir}/")

    # 4. Export pour Flame
    print("\n[4/4] Export pour Autodesk Flame...")
    exporter = ProfessionalExporter(export_dir)

    exported_files = exporter.export_for_autodesk_flame(
        heightmap=heightmap,
        normal_map=normal_map,
        depth_map=depth_map,
        ao_map=ao_map,
        diffuse_map=None,  # Auto-généré
        roughness_map=None,
        splatmaps=None,
        tree_instances=None,
        mesh_subsample=2,
        scale_y=50.0
    )

    # Afficher résultats
    print("\n" + "=" * 80)
    print("EXPORT TERMINÉ")
    print("=" * 80)
    print(f"\nNombre de fichiers: {len(exported_files)}")
    print("\nFICHIERS EXPORTÉS:")
    print("-" * 80)

    for key, filepath in exported_files.items():
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✓ {filename:30s} ({filesize:8.1f} KB)")

    print("\n" + "=" * 80)
    print(f"📁 Tous les fichiers sont dans: {export_dir}/")
    print("=" * 80)

    print("\n✅ TEST RÉUSSI!")
    print("\nPOUR IMPORTER DANS FLAME:")
    print("  1. Ouvrez Autodesk Flame")
    print("  2. Importez terrain.obj")
    print("  3. Les textures seront automatiquement chargées via le .mtl")
    print(f"  4. Voir {export_dir}/README_FLAME.txt pour plus de détails\n")

    return exported_files


if __name__ == "__main__":
    test_flame_export()
