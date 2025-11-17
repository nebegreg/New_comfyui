#!/usr/bin/env python3
"""
Test complet de tous les nouveaux modules Mountain Studio Pro v2.0

Execute ce script pour vérifier que tous les modules fonctionnent:
    python test_all_modules.py

Options:
    --quick    : Test rapide (résolutions réduites)
    --full     : Test complet avec export fichiers
    --visual   : Génère visualisations (nécessite matplotlib)
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Couleurs terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.OKCYAN}→ {msg}{Colors.ENDC}")


def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def test_imports():
    """Test que tous les modules peuvent être importés"""
    print_header("TEST 1: IMPORTS DES MODULES")

    modules_to_test = [
        ('core.terrain.hydraulic_erosion', 'HydraulicErosionSystem'),
        ('core.terrain.thermal_erosion', 'ThermalErosionSystem'),
        ('core.terrain.heightmap_generator', 'HeightmapGenerator'),
        ('core.vegetation.biome_classifier', 'BiomeClassifier', 'BiomeType'),
        ('core.vegetation.species_distribution', 'SpeciesDistributor', 'SpeciesProfile'),
        ('core.vegetation.vegetation_placer', 'VegetationPlacer', 'TreeInstance'),
        ('core.rendering.vfx_prompt_generator', 'VFXPromptGenerator', 'TerrainContext'),
        ('core.rendering.pbr_splatmap_generator', 'PBRSplatmapGenerator', 'MaterialLayer'),
        ('config.professional_presets', 'PresetManager', 'CompletePreset'),
        ('config.app_config', 'ConfigManager', 'AppSettings', 'AppPaths'),
    ]

    success_count = 0
    fail_count = 0

    for module_info in modules_to_test:
        module_name = module_info[0]
        classes = module_info[1:]

        try:
            module = __import__(module_name, fromlist=classes)

            # Vérifier que les classes existent
            for class_name in classes:
                if not hasattr(module, class_name):
                    raise ImportError(f"Class {class_name} not found in {module_name}")

            print_success(f"{module_name}: {', '.join(classes)}")
            success_count += 1

        except ImportError as e:
            print_error(f"{module_name}: {e}")
            fail_count += 1

    print(f"\n{Colors.BOLD}Résultat: {success_count}/{len(modules_to_test)} modules OK{Colors.ENDC}")

    if fail_count > 0:
        print_warning(f"{fail_count} modules échoués - vérifier l'installation")
        return False

    return True


def test_terrain_generation(quick=False):
    """Test génération de terrain avec érosion"""
    print_header("TEST 2: GÉNÉRATION DE TERRAIN")

    try:
        from core.terrain.heightmap_generator import HeightmapGenerator

        # Résolution selon mode
        size = 512 if quick else 1024
        print_info(f"Résolution: {size}x{size}")

        # Test 1: Sans érosion (rapide)
        print_info("Test 1: Génération sans érosion...")
        start = time.time()

        gen = HeightmapGenerator(size, size)
        heightmap_no_erosion = gen.generate(
            mountain_type='alpine',
            scale=80.0,
            octaves=7,
            seed=42,
            apply_hydraulic_erosion=False,
            apply_thermal_erosion=False
        )

        time_no_erosion = time.time() - start
        print_success(f"Généré en {time_no_erosion:.2f}s - Shape: {heightmap_no_erosion.shape}, Range: [{heightmap_no_erosion.min():.3f}, {heightmap_no_erosion.max():.3f}]")

        # Test 2: Avec érosion
        print_info("Test 2: Génération avec érosion hydraulique + thermique...")
        start = time.time()

        iterations = 10000 if quick else 30000

        heightmap_with_erosion = gen.generate(
            mountain_type='alpine',
            scale=80.0,
            octaves=7,
            seed=42,
            apply_hydraulic_erosion=True,
            apply_thermal_erosion=True,
            erosion_iterations=iterations
        )

        time_with_erosion = time.time() - start
        print_success(f"Généré en {time_with_erosion:.2f}s ({iterations} iterations)")

        # Comparer différence
        diff = np.mean(np.abs(heightmap_with_erosion - heightmap_no_erosion))
        print_info(f"Différence moyenne érosion: {diff:.4f}")

        if diff < 0.001:
            print_warning("Érosion semble avoir peu d'effet - normal si peu d'itérations")

        # Test 3: Maps dérivées
        print_info("Test 3: Génération normal/depth maps...")
        normal_map = gen.generate_normal_map(strength=1.0)
        depth_map = gen.generate_depth_map()

        print_success(f"Normal map: {normal_map.shape} dtype={normal_map.dtype}")
        print_success(f"Depth map: {depth_map.shape} dtype={depth_map.dtype}")

        return {
            'heightmap': heightmap_with_erosion,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'generator': gen
        }

    except Exception as e:
        print_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vegetation(terrain_result, quick=False):
    """Test système de végétation"""
    print_header("TEST 3: VÉGÉTATION PROCÉDURALE")

    if terrain_result is None:
        print_warning("Terrain result is None, skipping vegetation test")
        return None

    try:
        from core.vegetation.biome_classifier import BiomeClassifier, BiomeType
        from core.vegetation.vegetation_placer import VegetationPlacer
        from core.vegetation.species_distribution import SpeciesDistributor

        heightmap = terrain_result['heightmap']
        size = heightmap.shape[0]

        # Test 1: Classification biomes
        print_info("Test 1: Classification des biomes...")
        start = time.time()

        classifier = BiomeClassifier(size, size)
        biome_map = classifier.classify(heightmap)

        time_classify = time.time() - start

        # Compter biomes
        unique, counts = np.unique(biome_map, return_counts=True)
        biome_stats = dict(zip(unique, counts))

        print_success(f"Classifié en {time_classify:.2f}s")
        for biome_id, count in biome_stats.items():
            biome_info = classifier.get_biome_info(biome_id)
            percentage = (count / biome_map.size) * 100
            print_info(f"  {biome_info['name']}: {percentage:.1f}% ({count} pixels)")

        # Test 2: Distribution espèces
        print_info("Test 2: Distribution des espèces...")

        distributor = SpeciesDistributor()
        all_species = distributor.get_all_species()
        print_success(f"Espèces disponibles: {', '.join(all_species)}")

        # Test point médian
        mid_y, mid_x = size // 2, size // 2
        elevation = heightmap[mid_y, mid_x]
        suitable = distributor.get_suitable_species(
            elevation=elevation,
            temperature=1.0 - elevation,
            moisture=0.5,
            slope=0.2
        )
        print_info(f"  Au centre (elev={elevation:.2f}): {suitable}")

        # Test 3: Placement arbres
        print_info("Test 3: Placement des arbres (Poisson disc)...")
        start = time.time()

        density = 0.3 if quick else 0.5

        placer = VegetationPlacer(heightmap, biome_map, size, size)
        tree_instances = placer.place_vegetation(
            density=density,
            min_spacing=3.0,
            use_clustering=True,
            cluster_size=5
        )

        time_place = time.time() - start

        print_success(f"Placé {len(tree_instances)} arbres en {time_place:.2f}s")

        # Stats espèces
        species_counts = {}
        for tree in tree_instances:
            species_counts[tree.species] = species_counts.get(tree.species, 0) + 1

        for species, count in species_counts.items():
            percentage = (count / len(tree_instances)) * 100
            print_info(f"  {species}: {count} arbres ({percentage:.1f}%)")

        # Test 4: Density map
        print_info("Test 4: Génération density map...")
        density_map = placer.generate_density_map(tree_instances, radius=10.0)
        print_success(f"Density map: {density_map.shape}, max={density_map.max():.2f}")

        return {
            'biome_map': biome_map,
            'tree_instances': tree_instances,
            'density_map': density_map,
            'placer': placer
        }

    except Exception as e:
        print_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vfx_prompts(terrain_result, vegetation_result):
    """Test génération de prompts VFX"""
    print_header("TEST 4: PROMPTS VFX ULTRA-RÉALISTES")

    try:
        from core.rendering.vfx_prompt_generator import VFXPromptGenerator

        generator = VFXPromptGenerator()

        # Test 1: Presets
        print_info("Test 1: Chargement des presets...")
        presets = generator.create_preset_prompts()

        print_success(f"{len(presets)} presets disponibles:")
        for preset_name, preset_data in presets.items():
            print_info(f"  • {preset_data['name']}")

        # Test 2: Générer depuis un preset
        print_info("Test 2: Génération depuis preset 'epic_alpine_sunset'...")
        preset = presets['epic_alpine_sunset']

        result = generator.generate_prompt(
            terrain_context=preset['terrain_context'],
            camera_settings=preset['camera_settings'],
            photographer_style=preset['photographer_style'],
            quality_level=preset['quality_level']
        )

        print_success(f"Prompt positif: {len(result['positive'])} caractères")
        print_info(f"Preview: {result['positive'][:150]}...")

        # Vérifier keywords VFX
        vfx_keywords = ['hypersharp', 'UE5', 'RTX', 'gigapixel', '16k', 'photorealistic']
        found_keywords = [kw for kw in vfx_keywords if kw in result['positive']]
        print_success(f"Keywords VFX trouvés: {', '.join(found_keywords)}")

        print_success(f"Negative prompt: {len(result['negative'])} caractères")

        # Test 3: Auto-générer depuis heightmap
        if terrain_result and vegetation_result:
            print_info("Test 3: Auto-génération depuis heightmap...")

            auto_result = generator.auto_generate_from_heightmap(
                heightmap=terrain_result['heightmap'],
                biome_map=vegetation_result['biome_map'],
                vegetation_density_map=vegetation_result['density_map'],
                time_of_day='sunset',
                weather='clear',
                season='summer'
            )

            print_success(f"Auto-prompt: {len(auto_result['positive'])} caractères")
            print_info(f"Preview: {auto_result['positive'][:150]}...")

        # Test 4: Recommandations modèles
        print_info("Test 4: Recommandations modèles SDXL...")
        styles = ['photorealistic', 'dramatic', 'natural', 'vfx', 'artistic']

        for style in styles:
            model = generator.get_recommended_model(style)
            print_info(f"  {style}: {model['name']} ({model['steps']} steps, CFG {model['cfg_scale']})")

        return result

    except Exception as e:
        print_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_presets():
    """Test système de presets"""
    print_header("TEST 5: PRESETS PROFESSIONNELS")

    try:
        from config.professional_presets import PresetManager

        manager = PresetManager()

        # Test 1: Lister presets
        print_info("Test 1: Liste des presets par catégorie...")

        categorized = manager.get_presets_by_category()
        total_presets = 0

        for category, preset_names in categorized.items():
            if preset_names:
                print_success(f"{category}: {len(preset_names)} presets")
                for name in preset_names:
                    preset = manager.get_preset(name)
                    print_info(f"  • {preset.name}")
                total_presets += len(preset_names)

        print_success(f"Total: {total_presets} presets")

        # Test 2: Charger preset spécifique
        print_info("Test 2: Détails preset 'vfx_epic_mountain'...")
        preset = manager.get_preset('vfx_epic_mountain')

        if preset:
            print_success(f"Nom: {preset.name}")
            print_info(f"  Catégorie: {preset.category}")
            print_info(f"  Description: {preset.description}")
            print_info(f"  Terrain: {preset.terrain.width}x{preset.terrain.height}, {preset.terrain.mountain_type}")
            print_info(f"  Érosion: {preset.terrain.erosion_iterations} iterations")
            print_info(f"  Végétation: density={preset.vegetation.density}, clustering={preset.vegetation.use_clustering}")
            print_info(f"  Rendu: {preset.render.model_name}, {preset.render.steps} steps")
            print_info(f"  Tags: {', '.join(preset.tags) if preset.tags else 'None'}")

        # Test 3: Recherche
        print_info("Test 3: Recherche 'fog'...")
        results = manager.search_presets('fog')

        if results:
            print_success(f"Trouvé {len(results)} résultats:")
            for name in results:
                preset = manager.get_preset(name)
                print_info(f"  • {preset.name}: {preset.description}")
        else:
            print_warning("Aucun résultat pour 'fog'")

        return manager

    except Exception as e:
        print_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pbr_splatmaps(terrain_result, quick=False):
    """Test génération splatmaps PBR"""
    print_header("TEST 6: PBR SPLATMAPPING")

    if terrain_result is None:
        print_warning("Terrain result is None, skipping splatmap test")
        return None

    try:
        from core.rendering.pbr_splatmap_generator import PBRSplatmapGenerator

        heightmap = terrain_result['heightmap']
        size = heightmap.shape[0]

        print_info(f"Résolution: {size}x{size}")

        # Test 1: Lister matériaux
        print_info("Test 1: Matériaux PBR disponibles...")

        generator = PBRSplatmapGenerator(size, size)

        for mat_name, material in generator.materials.items():
            print_success(f"Layer {material.id}: {material.name}")
            print_info(f"  Altitude: {material.altitude_min:.2f}-{material.altitude_max:.2f}")
            print_info(f"  Pente: {material.slope_min:.2f}-{material.slope_max:.2f}")
            print_info(f"  {material.description}")

        # Test 2: Générer splatmaps
        print_info("Test 2: Génération splatmaps...")
        start = time.time()

        splatmap1, splatmap2 = generator.generate_splatmap(
            heightmap,
            apply_weathering=True,
            smooth_transitions=True,
            smooth_sigma=1.0
        )

        time_gen = time.time() - start

        print_success(f"Généré en {time_gen:.2f}s")
        print_success(f"Splatmap 1 (layers 0-3): {splatmap1.shape}, dtype={splatmap1.dtype}")
        print_success(f"Splatmap 2 (layers 4-7): {splatmap2.shape}, dtype={splatmap2.dtype}")

        # Vérifier que chaque pixel a bien une somme = 255 (ou proche)
        sum_map = splatmap1.sum(axis=2) + splatmap2.sum(axis=2)
        print_info(f"Somme weights: mean={sum_map.mean():.1f}, min={sum_map.min()}, max={sum_map.max()}")

        if abs(sum_map.mean() - 255) > 10:
            print_warning("Somme weights n'est pas ~255, vérifier normalisation")

        return {
            'splatmap1': splatmap1,
            'splatmap2': splatmap2,
            'generator': generator
        }

    except Exception as e:
        print_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_configuration():
    """Test système de configuration"""
    print_header("TEST 7: CONFIGURATION CENTRALISÉE")

    try:
        from config.app_config import init_config, get_config, AppPaths

        # Test 1: Initialisation
        print_info("Test 1: Initialisation configuration...")
        config = init_config()

        print_success(f"App: {config.settings.app_name} v{config.settings.version}")
        print_success(f"Theme: {config.settings.theme}, Language: {config.settings.language}")

        # Test 2: Defaults
        print_info("Test 2: Paramètres par défaut...")

        print_info(f"Terrain:")
        print_info(f"  Résolution: {config.settings.terrain.width}x{config.settings.terrain.height}")
        print_info(f"  Type: {config.settings.terrain.mountain_type}")
        print_info(f"  Érosion: {config.settings.terrain.apply_hydraulic_erosion}")
        print_info(f"  Iterations: {config.settings.terrain.erosion_iterations}")

        print_info(f"Végétation:")
        print_info(f"  Enabled: {config.settings.vegetation.enabled}")
        print_info(f"  Density: {config.settings.vegetation.density}")

        print_info(f"Rendu:")
        print_info(f"  Backend: {config.settings.render.backend}")
        print_info(f"  Model: {config.settings.render.model_name}")
        print_info(f"  Steps: {config.settings.render.steps}")

        # Test 3: Get/Set
        print_info("Test 3: Get/Set avec dot notation...")

        original_width = config.get('terrain.width')
        print_success(f"terrain.width = {original_width}")

        config.set('terrain.width', 4096)
        new_width = config.get('terrain.width')
        print_success(f"Après set(4096): {new_width}")

        # Restore
        config.set('terrain.width', original_width)

        # Test 4: Paths
        print_info("Test 4: Chemins de l'application...")

        AppPaths.ensure_dirs()

        print_success(f"Root: {AppPaths.ROOT_DIR}")
        print_info(f"  Core: {AppPaths.CORE_DIR}")
        print_info(f"  Output: {AppPaths.OUTPUT_DIR}")
        print_info(f"  Cache: {AppPaths.CACHE_DIR}")

        # Vérifier que dossiers existent
        if AppPaths.OUTPUT_DIR.exists():
            print_success("Dossiers output créés")
        else:
            print_warning("Dossiers output non créés")

        return config

    except Exception as e:
        print_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_test_results(terrain_result, vegetation_result, splatmap_result, output_dir="test_output"):
    """Exporte les résultats de test"""
    print_header("EXPORT DES RÉSULTATS")

    try:
        from PIL import Image
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        exported_files = []

        # Export heightmap
        if terrain_result:
            heightmap = terrain_result['heightmap']
            heightmap_img = (heightmap * 255).astype(np.uint8)
            img = Image.fromarray(heightmap_img, mode='L')
            filepath = output_path / "test_heightmap.png"
            img.save(filepath)
            exported_files.append(filepath)
            print_success(f"Heightmap: {filepath}")

            # Normal map
            if 'normal_map' in terrain_result:
                normal_map = terrain_result['normal_map']
                img = Image.fromarray(normal_map, mode='RGB')
                filepath = output_path / "test_normal_map.png"
                img.save(filepath)
                exported_files.append(filepath)
                print_success(f"Normal map: {filepath}")

        # Export vegetation density
        if vegetation_result and 'density_map' in vegetation_result:
            density_map = vegetation_result['density_map']
            density_img = (density_map * 255).astype(np.uint8)
            img = Image.fromarray(density_img, mode='L')
            filepath = output_path / "test_vegetation_density.png"
            img.save(filepath)
            exported_files.append(filepath)
            print_success(f"Vegetation density: {filepath}")

        # Export splatmaps
        if splatmap_result:
            splatmap1 = splatmap_result['splatmap1']
            img = Image.fromarray(splatmap1, mode='RGBA')
            filepath = output_path / "test_splatmap_0-3.png"
            img.save(filepath)
            exported_files.append(filepath)
            print_success(f"Splatmap 1: {filepath}")

            splatmap2 = splatmap_result['splatmap2']
            img = Image.fromarray(splatmap2, mode='RGBA')
            filepath = output_path / "test_splatmap_4-7.png"
            img.save(filepath)
            exported_files.append(filepath)
            print_success(f"Splatmap 2: {filepath}")

        print(f"\n{Colors.BOLD}Total: {len(exported_files)} fichiers exportés dans {output_dir}/{Colors.ENDC}")

        return exported_files

    except Exception as e:
        print_error(f"Erreur export: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_visualization(terrain_result, vegetation_result, splatmap_result, output_dir="test_output"):
    """Crée une visualisation complète"""
    print_header("VISUALISATION")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Mountain Studio Pro v2.0 - Test Results', fontsize=16, fontweight='bold')

        # Heightmap
        if terrain_result:
            axes[0, 0].imshow(terrain_result['heightmap'], cmap='terrain')
            axes[0, 0].set_title('Heightmap (avec érosion)')
            axes[0, 0].axis('off')

        # Normal map
        if terrain_result and 'normal_map' in terrain_result:
            axes[0, 1].imshow(terrain_result['normal_map'])
            axes[0, 1].set_title('Normal Map')
            axes[0, 1].axis('off')

        # Biome map
        if vegetation_result and 'biome_map' in vegetation_result:
            axes[0, 2].imshow(vegetation_result['biome_map'], cmap='tab10')
            axes[0, 2].set_title('Biome Classification')
            axes[0, 2].axis('off')

        # Vegetation
        if vegetation_result:
            axes[1, 0].imshow(terrain_result['heightmap'], cmap='terrain', alpha=0.7)

            if 'tree_instances' in vegetation_result:
                trees = vegetation_result['tree_instances']
                x_coords = [t.x for t in trees]
                y_coords = [t.y for t in trees]
                axes[1, 0].scatter(x_coords, y_coords, c='green', s=0.5, alpha=0.5)
                axes[1, 0].set_title(f'Vegetation ({len(trees)} trees)')
            axes[1, 0].axis('off')

        # Vegetation density
        if vegetation_result and 'density_map' in vegetation_result:
            axes[1, 1].imshow(vegetation_result['density_map'], cmap='YlGn')
            axes[1, 1].set_title('Vegetation Density Map')
            axes[1, 1].axis('off')

        # Splatmap visualization (simplified - show layer 0)
        if splatmap_result:
            # Show snow layer (R channel of splatmap1)
            snow_layer = splatmap_result['splatmap1'][:, :, 0]
            axes[1, 2].imshow(snow_layer, cmap='Blues')
            axes[1, 2].set_title('PBR Splatmap (Snow Layer)')
            axes[1, 2].axis('off')

        plt.tight_layout()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        filepath = output_path / "test_visualization.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print_success(f"Visualisation sauvegardée: {filepath}")

        return filepath

    except ImportError:
        print_warning("matplotlib non installé, skip visualisation")
        print_info("  Installez avec: pip install matplotlib")
        return None
    except Exception as e:
        print_error(f"Erreur visualisation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Test complet Mountain Studio Pro v2.0')
    parser.add_argument('--quick', action='store_true', help='Test rapide (résolutions réduites)')
    parser.add_argument('--full', action='store_true', help='Test complet avec exports')
    parser.add_argument('--visual', action='store_true', help='Générer visualisations (nécessite matplotlib)')

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "    MOUNTAIN STUDIO PRO v2.0 - TEST COMPLET".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"{Colors.ENDC}\n")

    if args.quick:
        print_info("Mode: TEST RAPIDE (résolutions réduites)")
    elif args.full:
        print_info("Mode: TEST COMPLET (avec exports)")
    else:
        print_info("Mode: TEST STANDARD")

    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"NumPy: {np.__version__}")

    # Exécuter les tests
    results = {}

    # Test 1: Imports
    if not test_imports():
        print_error("\n❌ Échec des imports - arrêt des tests")
        sys.exit(1)

    # Test 2: Terrain
    results['terrain'] = test_terrain_generation(quick=args.quick)

    # Test 3: Végétation
    results['vegetation'] = test_vegetation(results['terrain'], quick=args.quick)

    # Test 4: VFX Prompts
    results['prompts'] = test_vfx_prompts(results['terrain'], results['vegetation'])

    # Test 5: Presets
    results['presets'] = test_presets()

    # Test 6: Splatmaps
    results['splatmaps'] = test_pbr_splatmaps(results['terrain'], quick=args.quick)

    # Test 7: Configuration
    results['config'] = test_configuration()

    # Export si demandé
    if args.full:
        export_test_results(
            results['terrain'],
            results['vegetation'],
            results['splatmaps']
        )

    # Visualisation si demandé
    if args.visual:
        create_visualization(
            results['terrain'],
            results['vegetation'],
            results['splatmaps']
        )

    # Résumé final
    print_header("RÉSUMÉ FINAL")

    success_count = sum(1 for r in results.values() if r is not None)
    total_tests = len(results)

    if success_count == total_tests:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ TOUS LES TESTS RÉUSSIS ({success_count}/{total_tests}){Colors.ENDC}\n")
        print_success("Mountain Studio Pro v2.0 est prêt à l'emploi!")
        print_info("Prochaine étape: Lire REFACTORING_V2.md pour intégration UI")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠ CERTAINS TESTS ÉCHOUÉS ({success_count}/{total_tests} réussis){Colors.ENDC}\n")
        print_warning("Vérifier les erreurs ci-dessus")

    print()


if __name__ == "__main__":
    main()
