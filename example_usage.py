"""
Exemple d'utilisation directe des modules sans l'interface Gradio
Pour les utilisateurs avancés qui veulent intégrer dans leurs propres scripts
"""

from camera_system import CameraSystem
from prompt_generator import MountainPromptGenerator
from comfyui_integration import StableDiffusionDirect
from video_generator import VideoGenerator
from PIL import Image
import os


def example_single_image():
    """Exemple : Générer une seule image de montagne"""
    print("🏔️ Exemple 1: Génération d'une image unique")
    print("-" * 50)

    # Initialiser les composants
    camera = CameraSystem()
    prompt_gen = MountainPromptGenerator()
    sd = StableDiffusionDirect()

    # Configurer la caméra
    camera.set_camera(
        horizontal=45,      # Angle horizontal
        vertical=15,        # Angle vertical
        focal=50,           # Focale 50mm
        height=20,          # Hauteur moyenne
        distance=100        # Distance moyenne
    )

    # Obtenir la description de caméra
    camera_desc = camera.get_camera_description()
    depth_desc = camera.get_depth_of_field()

    # Paramètres de la scène
    params = {
        'mountain_type': 'alpine',
        'mountain_height': 75,
        'tree_density': 60,
        'tree_type': 'pine',
        'sky_type': 'sunset',
        'lighting': 'golden',
        'weather': 'clear',
        'season': 'autumn',
        'camera_desc': f"{camera_desc}, {depth_desc}"
    }

    # Générer le prompt
    prompt, negative_prompt = prompt_gen.generate_prompt(params)
    prompt = prompt_gen.add_detail_enhancement(prompt, detail_level=85)

    print(f"📝 Prompt généré:")
    print(f"   {prompt[:150]}...")
    print()

    # Générer l'image
    print("🎨 Génération de l'image en cours...")
    print("⏳ Cela peut prendre quelques minutes...")

    image = sd.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        steps=40,
        seed=42
    )

    if image:
        # Sauvegarder
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/example_mountain.png"
        image.save(output_path)
        print(f"✓ Image sauvegardée: {output_path}")
    else:
        print("❌ Erreur lors de la génération")

    print()


def example_camera_path():
    """Exemple : Générer un chemin de caméra pour une animation"""
    print("🎥 Exemple 2: Génération d'un chemin de caméra")
    print("-" * 50)

    camera = CameraSystem()

    # Configuration initiale
    camera.set_camera(
        horizontal=0,
        vertical=10,
        focal=50,
        height=20,
        distance=100
    )

    # Générer différents types de chemins
    paths = {
        'orbit': camera.generate_camera_path(10, 'orbit'),
        'pan': camera.generate_camera_path(10, 'pan'),
        'zoom': camera.generate_camera_path(10, 'zoom'),
        'flyover': camera.generate_camera_path(10, 'flyover')
    }

    for path_name, frames in paths.items():
        print(f"\n📹 Chemin '{path_name}': {len(frames)} frames")
        print(f"   Premier frame: horizontal={frames[0]['horizontal']:.1f}°, "
              f"vertical={frames[0]['vertical']:.1f}°, focal={frames[0]['focal']:.1f}mm")
        print(f"   Dernier frame: horizontal={frames[-1]['horizontal']:.1f}°, "
              f"vertical={frames[-1]['vertical']:.1f}°, focal={frames[-1]['focal']:.1f}mm")

    print()


def example_prompt_variations():
    """Exemple : Générer différentes variations de prompts"""
    print("📝 Exemple 3: Variations de prompts")
    print("-" * 50)

    prompt_gen = MountainPromptGenerator()

    # Différentes configurations
    configurations = [
        {
            'name': 'Matin d\'hiver',
            'params': {
                'mountain_type': 'alpine',
                'mountain_height': 80,
                'tree_density': 40,
                'tree_type': 'sparse',
                'sky_type': 'clear',
                'lighting': 'soft',
                'weather': 'snow',
                'season': 'winter',
                'camera_desc': 'eye-level view, standard lens'
            }
        },
        {
            'name': 'Coucher de soleil d\'automne',
            'params': {
                'mountain_type': 'rolling',
                'mountain_height': 50,
                'tree_density': 70,
                'tree_type': 'mixed',
                'sky_type': 'sunset',
                'lighting': 'golden',
                'weather': 'clear',
                'season': 'autumn',
                'camera_desc': 'elevated viewpoint, telephoto lens'
            }
        },
        {
            'name': 'Orage dramatique',
            'params': {
                'mountain_type': 'massive',
                'mountain_height': 90,
                'tree_density': 30,
                'tree_type': 'pine',
                'sky_type': 'stormy',
                'lighting': 'dramatic',
                'weather': 'rain',
                'season': 'summer',
                'camera_desc': 'low angle view, wide angle lens'
            }
        }
    ]

    for config in configurations:
        prompt, _ = prompt_gen.generate_prompt(config['params'])
        print(f"\n🎨 {config['name']}:")
        print(f"   {prompt[:120]}...")

    print()


def example_video_effects():
    """Exemple : Créer des effets vidéo sur une image statique"""
    print("🎬 Exemple 4: Effets vidéo")
    print("-" * 50)

    # Créer une image de test (normalement vous utiliseriez une vraie image générée)
    print("ℹ️  Cet exemple nécessite une image existante")
    print("   Générez d'abord une image avec example_single_image()")
    print()

    # Code d'exemple (décommenté si vous avez une image)
    """
    video_gen = VideoGenerator()

    # Charger une image
    image = Image.open("outputs/example_mountain.png")

    # Créer un effet de zoom
    zoom_frames = video_gen.add_zoom_effect(image, num_frames=30, zoom_factor=1.5)

    # Créer la vidéo
    video_gen.create_video_from_images(
        zoom_frames,
        "outputs/zoom_effect.mp4",
        fps=24
    )

    print("✓ Vidéo avec effet de zoom créée: outputs/zoom_effect.mp4")
    """


def main():
    """Exécute tous les exemples"""
    print("\n" + "=" * 50)
    print("  🏔️ EXEMPLES D'UTILISATION")
    print("=" * 50 + "\n")

    # Note: Décommentez l'exemple que vous voulez exécuter
    # Attention: example_single_image() nécessite un GPU et peut prendre du temps

    # example_single_image()        # Nécessite GPU et temps
    example_camera_path()          # Rapide
    example_prompt_variations()    # Rapide
    example_video_effects()        # Info seulement

    print("\n" + "=" * 50)
    print("  ✓ Exemples terminés")
    print("=" * 50 + "\n")

    print("💡 Conseils:")
    print("   - Décommentez example_single_image() pour générer une vraie image")
    print("   - Assurez-vous d'avoir un GPU avec CUDA pour de meilleures performances")
    print("   - Modifiez les paramètres pour expérimenter différents styles")
    print()


if __name__ == "__main__":
    main()
