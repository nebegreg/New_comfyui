#!/usr/bin/env python3
"""
MOUNTAIN STUDIO PRO v2.0 - DÉMONSTRATION COMPLÈTE
Script de démonstration montrant toutes les capacités professionnelles

Génère un terrain complet avec:
- Érosion hydraulique et thermique réaliste
- Végétation procédurale (arbres, biomes)
- Prompts VFX ultra-réalistes
- PBR splatmaps 8 layers
- Export complet (maps, instances, prompts)

Temps d'exécution: ~2-5 minutes (selon résolution)
"""

import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Import nouveaux modules
from core.terrain.heightmap_generator import HeightmapGenerator
from core.vegetation.biome_classifier import BiomeClassifier
from core.vegetation.vegetation_placer import VegetationPlacer
from core.rendering.vfx_prompt_generator import VFXPromptGenerator
from core.rendering.pbr_splatmap_generator import PBRSplatmapGenerator
from config.professional_presets import PresetManager
from config.app_config import init_config, AppPaths


class Colors:
    """Couleurs pour affichage terminal"""
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


def print_step(step_num, total_steps, msg):
    print(f"{Colors.OKCYAN}[{step_num}/{total_steps}] {msg}{Colors.ENDC}")


def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.OKBLUE}→ {msg}{Colors.ENDC}")


def demo_quick(resolution=1024):
    """
    Démonstration rapide (1024x1024)
    Temps: ~2-3 minutes
    """
    print_header("MOUNTAIN STUDIO PRO v2.0 - DÉMONSTRATION RAPIDE")

    total_steps = 8
    output_dir = "demo_output_quick"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    start_time = time.time()

    # ÉTAPE 1: Configuration
    print_step(1, total_steps, "Initialisation configuration...")
    config = init_config()
    AppPaths.ensure_dirs()
    print_success(f"Application: {config.settings.app_name} v{config.settings.version}")

    # ÉTAPE 2: Génération terrain avec érosion
    print_step(2, total_steps, f"Génération terrain {resolution}x{resolution} avec érosion avancée...")
    step_start = time.time()

    gen = HeightmapGenerator(resolution, resolution)
    heightmap = gen.generate(
        mountain_type='alpine',
        scale=100.0,
        octaves=8,
        seed=42,
        apply_hydraulic_erosion=True,
        apply_thermal_erosion=True,
        erosion_iterations=30000,  # Rapide mais réaliste
        domain_warp_strength=0.4,
        use_ridged_multifractal=True
    )

    step_time = time.time() - step_start
    print_success(f"Terrain généré en {step_time:.1f}s")
    print_info(f"  Range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")

    # ÉTAPE 3: Normal/Depth/AO maps
    print_step(3, total_steps, "Génération maps dérivées (normal, depth, AO)...")
    step_start = time.time()

    normal_map = gen.generate_normal_map(strength=1.2)
    depth_map = gen.generate_depth_map()
    ao_map = gen.generate_ambient_occlusion(radius=3.0)

    step_time = time.time() - step_start
    print_success(f"Maps générées en {step_time:.1f}s")

    # ÉTAPE 4: Classification biomes
    print_step(4, total_steps, "Classification écologique des biomes...")
    step_start = time.time()

    classifier = BiomeClassifier(resolution, resolution)
    biome_map = classifier.classify(heightmap)

    # Stats biomes
    unique, counts = np.unique(biome_map, return_counts=True)
    step_time = time.time() - step_start
    print_success(f"Biomes classifiés en {step_time:.1f}s")
    for biome_id, count in zip(unique, counts):
        biome_info = classifier.get_biome_info(biome_id)
        percentage = (count / biome_map.size) * 100
        print_info(f"  {biome_info['name']}: {percentage:.1f}%")

    # ÉTAPE 5: Placement végétation
    print_step(5, total_steps, "Placement végétation procédurale (Poisson disc + clustering)...")
    step_start = time.time()

    placer = VegetationPlacer(resolution, resolution, heightmap, biome_map)
    trees = placer.place_vegetation(
        density=0.4,
        min_spacing=3.0,
        use_clustering=True,
        cluster_size=8
    )

    density_map = placer.generate_density_map()

    step_time = time.time() - step_start
    print_success(f"{len(trees)} arbres placés en {step_time:.1f}s")

    # Stats espèces
    if len(trees) > 0:
        species_counts = {}
        for tree in trees:
            species_counts[tree.species] = species_counts.get(tree.species, 0) + 1

        for species, count in species_counts.items():
            percentage = (count / len(trees)) * 100
            print_info(f"  {species}: {count} arbres ({percentage:.1f}%)")

    # ÉTAPE 6: PBR Splatmaps
    print_step(6, total_steps, "Génération PBR splatmaps (8 matériaux)...")
    step_start = time.time()

    splatmap_gen = PBRSplatmapGenerator(resolution, resolution)
    splatmap1, splatmap2 = splatmap_gen.generate_splatmap(
        heightmap,
        apply_weathering=True,
        smooth_transitions=True
    )

    step_time = time.time() - step_start
    print_success(f"Splatmaps générées en {step_time:.1f}s")
    print_info("  8 matériaux: snow, rock_cliff, rock_ground, alpine_grass, forest_grass, dirt, moss_wet, scree")

    # ÉTAPE 7: Prompts VFX
    print_step(7, total_steps, "Génération prompts VFX ultra-réalistes...")
    step_start = time.time()

    prompt_gen = VFXPromptGenerator()
    prompt_result = prompt_gen.auto_generate_from_heightmap(
        heightmap,
        biome_map,
        vegetation_density_map=density_map,
        time_of_day='sunset',
        weather='clear',
        season='summer'
    )

    step_time = time.time() - step_start
    print_success(f"Prompt généré en {step_time:.1f}s")
    print_info(f"  Longueur: {len(prompt_result['positive'])} caractères")
    print_info(f"  Preview: {prompt_result['positive'][:120]}...")

    # Recommandation modèle
    model = prompt_gen.get_recommended_model('photorealistic')
    print_info(f"  Modèle recommandé: {model['name']} ({model['steps']} steps, CFG {model['cfg_scale']})")

    # ÉTAPE 8: Export tout
    print_step(8, total_steps, "Export fichiers...")
    step_start = time.time()

    # Heightmap
    heightmap_img = (heightmap * 255).astype(np.uint8)
    Image.fromarray(heightmap_img).save(f"{output_dir}/heightmap.png")

    # Normal map
    Image.fromarray(normal_map).save(f"{output_dir}/normal_map.png")

    # Depth map
    Image.fromarray(depth_map).save(f"{output_dir}/depth_map.png")

    # AO map
    Image.fromarray(ao_map).save(f"{output_dir}/ao_map.png")

    # Biome map (coloré)
    biome_colors = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for biome_id in unique:
        biome_info = classifier.get_biome_info(biome_id)
        mask = (biome_map == biome_id)
        biome_colors[mask] = biome_info['color']
    Image.fromarray(biome_colors).save(f"{output_dir}/biome_map.png")

    # Vegetation density
    density_img = (density_map * 255).astype(np.uint8)
    Image.fromarray(density_img).save(f"{output_dir}/vegetation_density.png")

    # Splatmaps
    splatmap_gen.export_splatmaps(splatmap1, splatmap2, output_dir, prefix="terrain")
    splatmap_gen.export_material_info(f"{output_dir}/materials.json")

    # Vegetation instances
    if len(trees) > 0:
        placer.export_instances(f"{output_dir}/vegetation_instances.json")

    # Prompt
    with open(f"{output_dir}/prompt.txt", 'w', encoding='utf-8') as f:
        f.write("=== POSITIVE PROMPT ===\n")
        f.write(prompt_result['positive'])
        f.write("\n\n=== NEGATIVE PROMPT ===\n")
        f.write(prompt_result['negative'])
        f.write("\n\n=== METADATA ===\n")
        for key, value in prompt_result['metadata'].items():
            f.write(f"{key}: {value}\n")

    step_time = time.time() - step_start
    print_success(f"Fichiers exportés en {step_time:.1f}s")

    # RÉSUMÉ
    total_time = time.time() - start_time
    print_header("GÉNÉRATION TERMINÉE")

    print_success(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print_info(f"Dossier: {output_dir}/")
    print()
    print("Fichiers générés:")
    print("  • heightmap.png - Heightmap 8-bit")
    print("  • normal_map.png - Normal map RGB")
    print("  • depth_map.png - Depth map")
    print("  • ao_map.png - Ambient occlusion")
    print("  • biome_map.png - Classification biomes (coloré)")
    print("  • vegetation_density.png - Density map végétation")
    print("  • terrain_splatmap_0-3.png - Splatmap layers 0-3 (RGBA)")
    print("  • terrain_splatmap_4-7.png - Splatmap layers 4-7 (RGBA)")
    print("  • materials.json - Info matériaux PBR")
    print(f"  • trees_blender.json - {len(trees)} instances d'arbres")
    print("  • prompt.txt - Prompt VFX complet")
    print()
    print(f"{Colors.BOLD}🎉 Prêt pour production VFX/Game Dev!{Colors.ENDC}")


def demo_preset_vfx():
    """
    Démonstration avec preset VFX professionnel
    Génère un rendu 4K pour production
    """
    print_header("MOUNTAIN STUDIO PRO v2.0 - WORKFLOW VFX PRODUCTION")

    output_dir = "demo_output_vfx_4k"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    total_steps = 9
    start_time = time.time()

    # ÉTAPE 1: Charger preset
    print_step(1, total_steps, "Chargement preset VFX professionnel...")

    manager = PresetManager()
    preset = manager.get_preset('vfx_epic_mountain')

    print_success(f"Preset: {preset.name}")
    print_info(f"  Description: {preset.description}")
    print_info(f"  Résolution: {preset.terrain.width}x{preset.terrain.height}")
    print_info(f"  Érosion: {preset.terrain.erosion_iterations} iterations")
    print_info(f"  Modèle AI: {preset.render.model_name}")
    print()

    print(f"{Colors.WARNING}⚠ Ce preset génère du 4K (4096x4096) - cela peut prendre 5-10 minutes{Colors.ENDC}")
    print(f"{Colors.WARNING}⚠ RAM recommandée: 16GB+{Colors.ENDC}")

    response = input(f"\n{Colors.BOLD}Continuer? (o/n): {Colors.ENDC}")
    if response.lower() != 'o':
        print("Annulé.")
        return

    # ÉTAPE 2: Terrain 4K
    print_step(2, total_steps, f"Génération terrain {preset.terrain.width}x{preset.terrain.height}...")
    step_start = time.time()

    gen = HeightmapGenerator(preset.terrain.width, preset.terrain.height)
    heightmap = gen.generate(
        mountain_type=preset.terrain.mountain_type,
        seed=preset.terrain.seed,
        apply_hydraulic_erosion=preset.terrain.apply_hydraulic_erosion,
        apply_thermal_erosion=preset.terrain.apply_thermal_erosion,
        erosion_iterations=preset.terrain.erosion_iterations,
        domain_warp_strength=preset.terrain.domain_warp_strength
    )

    step_time = time.time() - step_start
    print_success(f"Terrain 4K généré en {step_time:.1f}s")

    # ÉTAPE 3-9: Continue comme demo_quick mais en 4K...
    print_step(3, total_steps, "Génération maps dérivées...")
    normal_map = gen.generate_normal_map(strength=1.5)
    depth_map = gen.generate_depth_map()
    ao_map = gen.generate_ambient_occlusion(radius=5.0)
    print_success("Maps générées")

    print_step(4, total_steps, "Classification biomes...")
    classifier = BiomeClassifier(preset.terrain.width, preset.terrain.height)
    biome_map = classifier.classify(heightmap)
    print_success("Biomes classifiés")

    print_step(5, total_steps, "Placement végétation...")
    placer = VegetationPlacer(preset.terrain.width, preset.terrain.height, heightmap, biome_map)
    trees = placer.place_vegetation(
        density=preset.vegetation.density,
        use_clustering=preset.vegetation.use_clustering,
        cluster_size=preset.vegetation.cluster_size
    )
    density_map = placer.generate_density_map()
    print_success(f"{len(trees)} arbres placés")

    print_step(6, total_steps, "Génération splatmaps PBR...")
    splatmap_gen = PBRSplatmapGenerator(preset.terrain.width, preset.terrain.height)
    splatmap1, splatmap2 = splatmap_gen.generate_splatmap(heightmap, apply_weathering=True)
    print_success("Splatmaps générées")

    print_step(7, total_steps, "Génération prompt VFX...")
    prompt_gen = VFXPromptGenerator()
    prompt_result = prompt_gen.generate_prompt(
        terrain_context=preset.render.terrain_context,
        camera_settings=preset.camera_settings,
        photographer_style=preset.render.photographer_style,
        quality_level=preset.render.quality_level
    )
    print_success("Prompt VFX généré")

    print_step(8, total_steps, "Export fichiers 4K...")

    # Export 16-bit heightmap pour VFX
    heightmap_16bit = (heightmap * 65535).astype(np.uint16)
    Image.fromarray(heightmap_16bit, mode='I;16').save(f"{output_dir}/heightmap_16bit.png")

    # Autres exports
    Image.fromarray(normal_map).save(f"{output_dir}/normal_map_4k.png")
    Image.fromarray(depth_map).save(f"{output_dir}/depth_map_4k.png")
    Image.fromarray(ao_map).save(f"{output_dir}/ao_map_4k.png")

    splatmap_gen.export_splatmaps(splatmap1, splatmap2, output_dir, format='png')
    splatmap_gen.export_material_info(f"{output_dir}/materials.json")

    if len(trees) > 0:
        placer.export_instances(f"{output_dir}/vegetation_instances_4k.json")

    density_img = (density_map * 255).astype(np.uint8)
    Image.fromarray(density_img).save(f"{output_dir}/vegetation_density_4k.png")

    with open(f"{output_dir}/prompt_vfx.txt", 'w', encoding='utf-8') as f:
        f.write("=== PROMPT VFX PRODUCTION ===\n")
        f.write(f"Preset: {preset.name}\n")
        f.write(f"Resolution: {preset.terrain.width}x{preset.terrain.height}\n")
        f.write(f"Model: {preset.render.model_name}\n")
        f.write(f"Steps: {preset.render.steps}\n")
        f.write(f"CFG: {preset.render.cfg_scale}\n\n")
        f.write("POSITIVE:\n")
        f.write(prompt_result['positive'])
        f.write("\n\nNEGATIVE:\n")
        f.write(prompt_result['negative'])

    print_success("Export 4K terminé")

    total_time = time.time() - start_time
    print_header("WORKFLOW VFX TERMINÉ")

    print_success(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print_info(f"Dossier: {output_dir}/")
    print()
    print(f"{Colors.BOLD}✅ Asset 4K prêt pour production VFX{Colors.ENDC}")


def show_menu():
    """Menu interactif"""
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MOUNTAIN STUDIO PRO v2.0 - DÉMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"{Colors.ENDC}\n")

    print("Choisissez une démonstration:\n")
    print(f"{Colors.OKGREEN}1. Démonstration Rapide (1024x1024, ~2-3 min){Colors.ENDC}")
    print("   Génère un terrain complet avec toutes les features")
    print()
    print(f"{Colors.WARNING}2. Workflow VFX Production (4096x4096, ~5-10 min){Colors.ENDC}")
    print("   Utilise preset professionnel 'VFX Epic Mountain'")
    print()
    print(f"{Colors.OKBLUE}3. Liste des Presets Disponibles{Colors.ENDC}")
    print()
    print("0. Quitter")
    print()


def list_presets():
    """Liste tous les presets"""
    print_header("PRESETS PROFESSIONNELS DISPONIBLES")

    manager = PresetManager()
    categorized = manager.get_presets_by_category()

    for category, preset_names in categorized.items():
        if preset_names:
            print(f"\n{Colors.BOLD}{category.upper().replace('_', ' ')}:{Colors.ENDC}")
            for name in preset_names:
                preset = manager.get_preset(name)
                print(f"  • {Colors.OKGREEN}{preset.name}{Colors.ENDC}")
                print(f"    {preset.description}")
                print(f"    {preset.terrain.width}x{preset.terrain.height}, {preset.terrain.mountain_type}")

    print()


def main():
    """Point d'entrée principal"""

    while True:
        show_menu()

        try:
            choice = input(f"{Colors.BOLD}Votre choix: {Colors.ENDC}")

            if choice == '1':
                demo_quick()
            elif choice == '2':
                demo_preset_vfx()
            elif choice == '3':
                list_presets()
            elif choice == '0':
                print(f"\n{Colors.OKGREEN}Au revoir!{Colors.ENDC}\n")
                break
            else:
                print(f"{Colors.FAIL}Choix invalide{Colors.ENDC}")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Interrompu par l'utilisateur{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.FAIL}Erreur: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()

        input(f"\n{Colors.BOLD}Appuyez sur Entrée pour continuer...{Colors.ENDC}")


if __name__ == "__main__":
    main()
