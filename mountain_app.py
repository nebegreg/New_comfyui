"""
Application de simulation de montagne ultra-réaliste
Interface graphique avec Gradio pour générer des montagnes avec Stable Diffusion/ComfyUI
"""

import gradio as gr
from PIL import Image
import numpy as np
import os
import time
from datetime import datetime
from typing import Optional, List

from camera_system import CameraSystem
from prompt_generator import MountainPromptGenerator
from comfyui_integration import ComfyUIIntegration, StableDiffusionDirect
from video_generator import VideoGenerator


class MountainSimulationApp:
    """Application principale de simulation de montagne"""

    def __init__(self):
        self.camera = CameraSystem()
        self.prompt_gen = MountainPromptGenerator()
        self.comfyui = None
        self.sd_direct = None
        self.video_gen = VideoGenerator()
        self.use_comfyui = False
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_backend(self, backend: str, comfyui_server: str = "127.0.0.1:8188"):
        """Initialise le backend de génération (ComfyUI ou Stable Diffusion direct)"""
        if backend == "ComfyUI":
            self.comfyui = ComfyUIIntegration(comfyui_server)
            self.use_comfyui = True
            return "✓ ComfyUI initialisé"
        else:
            self.sd_direct = StableDiffusionDirect()
            success = self.sd_direct.load_model()
            self.use_comfyui = False
            if success:
                return "✓ Stable Diffusion chargé avec succès"
            else:
                return "⚠ Erreur lors du chargement de Stable Diffusion"

    def generate_single_image(self,
                             # Paramètres de montagne
                             mountain_type: str,
                             mountain_height: float,
                             tree_density: float,
                             tree_type: str,
                             # Paramètres de ciel et météo
                             sky_type: str,
                             lighting: str,
                             weather: str,
                             season: str,
                             # Paramètres de caméra
                             horizontal_angle: float,
                             vertical_angle: float,
                             focal_length: float,
                             camera_height: float,
                             camera_distance: float,
                             # Paramètres de génération
                             width: int,
                             height: int,
                             steps: int,
                             seed: int,
                             detail_level: int) -> tuple:
        """Génère une seule image de montagne"""

        # Configuration de la caméra
        self.camera.set_camera(horizontal_angle, vertical_angle, focal_length,
                               camera_height, camera_distance)
        camera_desc = self.camera.get_camera_description()
        depth_desc = self.camera.get_depth_of_field()

        # Génération du prompt
        params = {
            'mountain_type': mountain_type.lower(),
            'mountain_height': mountain_height,
            'tree_density': tree_density,
            'tree_type': tree_type.lower().replace(' ', '_'),
            'sky_type': sky_type.lower().replace(' ', '_'),
            'lighting': lighting.lower(),
            'weather': weather.lower(),
            'season': season.lower(),
            'camera_desc': f"{camera_desc}, {depth_desc}"
        }

        prompt, negative_prompt = self.prompt_gen.generate_prompt(params)
        prompt = self.prompt_gen.add_detail_enhancement(prompt, detail_level)

        # Génération de l'image
        status = "🎨 Génération de l'image en cours...\n\n"
        status += f"📝 Prompt: {prompt[:200]}...\n\n"
        status += f"🎥 Caméra: {camera_desc}\n"
        status += f"🔍 {depth_desc}\n"

        image = None
        if self.use_comfyui and self.comfyui:
            workflow = self.comfyui.generate_workflow(prompt, negative_prompt,
                                                      width, height, steps, seed)
            prompt_id = self.comfyui.queue_prompt(workflow)
            if prompt_id:
                status += "\n⏳ En attente de ComfyUI...\n"
                image = self.comfyui.get_image(prompt_id)
        elif self.sd_direct:
            image = self.sd_direct.generate_image(prompt, negative_prompt,
                                                  width, height, steps, seed)

        if image:
            # Sauvegarder l'image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mountain_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath)

            status += f"\n✓ Image générée avec succès!\n"
            status += f"💾 Sauvegardée: {filepath}\n"

            return image, status, prompt, negative_prompt
        else:
            status += "\n❌ Erreur lors de la génération\n"
            return None, status, prompt, negative_prompt

    def generate_video_sequence(self,
                                # Paramètres de scène (identiques)
                                mountain_type: str, mountain_height: float,
                                tree_density: float, tree_type: str,
                                sky_type: str, lighting: str, weather: str, season: str,
                                # Paramètres de caméra initiaux
                                horizontal_angle: float, vertical_angle: float,
                                focal_length: float, camera_height: float,
                                camera_distance: float,
                                # Paramètres de génération
                                width: int, height: int, steps: int, seed: int,
                                detail_level: int,
                                # Paramètres vidéo
                                num_frames: int, camera_path: str,
                                fps: int, add_transitions: bool) -> tuple:
        """Génère une séquence d'images pour créer une vidéo"""

        status = f"🎬 Génération de vidéo - {num_frames} frames\n"
        status += f"🎥 Type de mouvement: {camera_path}\n\n"

        # Configuration initiale de la caméra
        self.camera.set_camera(horizontal_angle, vertical_angle, focal_length,
                               camera_height, camera_distance)

        # Générer le chemin de caméra
        camera_frames = self.camera.generate_camera_path(num_frames, camera_path.lower())

        images = []
        prompts_used = []

        for i, cam_params in enumerate(camera_frames):
            status += f"📸 Frame {i+1}/{num_frames}\n"

            # Mettre à jour la caméra
            self.camera.set_camera(
                cam_params['horizontal'],
                cam_params['vertical'],
                cam_params['focal'],
                cam_params['height'],
                cam_params['distance']
            )

            camera_desc = self.camera.get_camera_description()
            depth_desc = self.camera.get_depth_of_field()

            # Générer le prompt pour ce frame
            params = {
                'mountain_type': mountain_type.lower(),
                'mountain_height': mountain_height,
                'tree_density': tree_density,
                'tree_type': tree_type.lower().replace(' ', '_'),
                'sky_type': sky_type.lower().replace(' ', '_'),
                'lighting': lighting.lower(),
                'weather': weather.lower(),
                'season': season.lower(),
                'camera_desc': f"{camera_desc}, {depth_desc}"
            }

            prompt, negative_prompt = self.prompt_gen.generate_prompt(params)
            prompt = self.prompt_gen.add_detail_enhancement(prompt, detail_level)
            prompts_used.append(prompt)

            # Générer l'image
            frame_seed = seed + i  # Seed différent pour chaque frame

            image = None
            if self.use_comfyui and self.comfyui:
                workflow = self.comfyui.generate_workflow(prompt, negative_prompt,
                                                          width, height, steps, frame_seed)
                prompt_id = self.comfyui.queue_prompt(workflow)
                if prompt_id:
                    image = self.comfyui.get_image(prompt_id)
            elif self.sd_direct:
                image = self.sd_direct.generate_image(prompt, negative_prompt,
                                                      width, height, steps, frame_seed)

            if image:
                images.append(image)
                status += f"  ✓ Frame {i+1} généré\n"
            else:
                status += f"  ❌ Erreur frame {i+1}\n"

            yield None, status, "\n\n".join(prompts_used)

        if images:
            # Créer la vidéo
            status += f"\n🎞️ Assemblage de la vidéo ({len(images)} frames)...\n"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"mountain_video_{timestamp}.mp4"
            video_path = os.path.join(self.output_dir, video_filename)

            success = self.video_gen.create_video_from_images(
                images, video_path, fps, add_transitions
            )

            if success:
                status += f"\n✓ Vidéo créée avec succès!\n"
                status += f"💾 {video_path}\n"
                status += f"📊 {len(images)} frames, {fps} FPS, "
                status += f"{len(images)/fps:.1f} secondes\n"

                yield video_path, status, "\n\n".join(prompts_used)
            else:
                status += "\n❌ Erreur lors de la création de la vidéo\n"
                yield None, status, "\n\n".join(prompts_used)
        else:
            status += "\n❌ Aucune image générée\n"
            yield None, status, "\n\n".join(prompts_used)

    def create_interface(self):
        """Crée l'interface Gradio"""

        with gr.Blocks(title="Simulation de Montagne Ultra-Réaliste", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🏔️ Simulation de Montagne Ultra-Réaliste
            Générez des images et vidéos de montagnes photoréalistes avec Stable Diffusion / ComfyUI
            """)

            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("### Backend de génération")
                with gr.Row():
                    backend_choice = gr.Radio(
                        ["ComfyUI", "Stable Diffusion Direct"],
                        value="Stable Diffusion Direct",
                        label="Moteur de génération"
                    )
                    comfyui_server = gr.Textbox(
                        value="127.0.0.1:8188",
                        label="Adresse serveur ComfyUI",
                        visible=False
                    )
                    init_btn = gr.Button("🚀 Initialiser", variant="primary")
                    init_status = gr.Textbox(label="Status", interactive=False)

                def update_server_visibility(choice):
                    return gr.update(visible=(choice == "ComfyUI"))

                backend_choice.change(
                    update_server_visibility,
                    inputs=[backend_choice],
                    outputs=[comfyui_server]
                )

                init_btn.click(
                    self.initialize_backend,
                    inputs=[backend_choice, comfyui_server],
                    outputs=[init_status]
                )

            with gr.Tab("🖼️ Image Unique"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🏔️ Paramètres de Montagne")

                        mountain_type = gr.Dropdown(
                            ["Alpine", "Rolling", "Volcanic", "Massive", "Rocky"],
                            value="Alpine",
                            label="Type de montagne"
                        )
                        mountain_height = gr.Slider(0, 100, 70, label="Hauteur relative")

                        gr.Markdown("### 🌲 Végétation")
                        tree_type = gr.Dropdown(
                            ["Pine", "Spruce", "Mixed", "Sparse", "Dense"],
                            value="Pine",
                            label="Type d'arbres"
                        )
                        tree_density = gr.Slider(0, 100, 60, label="Densité de végétation")

                        gr.Markdown("### ☁️ Ciel et Météo")
                        sky_type = gr.Dropdown(
                            ["Clear", "Cloudy", "Sunset", "Sunrise", "Stormy", "Overcast", "Partly Cloudy"],
                            value="Partly Cloudy",
                            label="Type de ciel"
                        )
                        lighting = gr.Dropdown(
                            ["Golden", "Midday", "Dramatic", "Soft", "Backlit"],
                            value="Dramatic",
                            label="Éclairage"
                        )
                        weather = gr.Dropdown(
                            ["Clear", "Fog", "Snow", "Rain"],
                            value="Clear",
                            label="Météo"
                        )
                        season = gr.Dropdown(
                            ["Spring", "Summer", "Autumn", "Winter"],
                            value="Summer",
                            label="Saison"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 Paramètres de Caméra")

                        horizontal_angle = gr.Slider(-180, 180, 0, label="Angle horizontal (°)")
                        vertical_angle = gr.Slider(-90, 90, 10, label="Angle vertical (°)")
                        focal_length = gr.Slider(24, 200, 50, label="Focale (mm)")
                        camera_height = gr.Slider(0, 100, 20, label="Hauteur caméra")
                        camera_distance = gr.Slider(10, 500, 100, label="Distance")

                        gr.Markdown("### 🎨 Paramètres de Génération")
                        img_width = gr.Slider(512, 2048, 1024, step=64, label="Largeur")
                        img_height = gr.Slider(512, 2048, 768, step=64, label="Hauteur")
                        steps = gr.Slider(20, 100, 40, label="Steps de diffusion")
                        seed = gr.Number(value=42, label="Seed (aléatoire)")
                        detail_level = gr.Slider(0, 100, 85, label="Niveau de détail")

                        generate_btn = gr.Button("🎨 Générer l'image", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        output_image = gr.Image(label="Image générée", type="pil")
                    with gr.Column():
                        generation_status = gr.Textbox(label="Status", lines=15)
                        prompt_display = gr.Textbox(label="Prompt utilisé", lines=5)
                        negative_prompt_display = gr.Textbox(label="Negative prompt", lines=3)

                generate_btn.click(
                    self.generate_single_image,
                    inputs=[
                        mountain_type, mountain_height, tree_density, tree_type,
                        sky_type, lighting, weather, season,
                        horizontal_angle, vertical_angle, focal_length,
                        camera_height, camera_distance,
                        img_width, img_height, steps, seed, detail_level
                    ],
                    outputs=[output_image, generation_status, prompt_display, negative_prompt_display]
                )

            with gr.Tab("🎬 Génération Vidéo"):
                gr.Markdown("""
                ### Générez une vidéo avec mouvement de caméra
                Cette fonction génère plusieurs images avec différentes positions de caméra puis les assemble en vidéo.
                """)

                with gr.Row():
                    with gr.Column():
                        # Réutiliser les mêmes paramètres de scène
                        v_mountain_type = gr.Dropdown(
                            ["Alpine", "Rolling", "Volcanic", "Massive", "Rocky"],
                            value="Alpine", label="Type de montagne"
                        )
                        v_mountain_height = gr.Slider(0, 100, 70, label="Hauteur relative")
                        v_tree_type = gr.Dropdown(
                            ["Pine", "Spruce", "Mixed", "Sparse", "Dense"],
                            value="Pine", label="Type d'arbres"
                        )
                        v_tree_density = gr.Slider(0, 100, 60, label="Densité de végétation")
                        v_sky_type = gr.Dropdown(
                            ["Clear", "Cloudy", "Sunset", "Sunrise", "Stormy", "Overcast", "Partly Cloudy"],
                            value="Sunset", label="Type de ciel"
                        )
                        v_lighting = gr.Dropdown(
                            ["Golden", "Midday", "Dramatic", "Soft", "Backlit"],
                            value="Golden", label="Éclairage"
                        )
                        v_weather = gr.Dropdown(
                            ["Clear", "Fog", "Snow", "Rain"],
                            value="Clear", label="Météo"
                        )
                        v_season = gr.Dropdown(
                            ["Spring", "Summer", "Autumn", "Winter"],
                            value="Autumn", label="Saison"
                        )

                    with gr.Column():
                        v_horizontal_angle = gr.Slider(-180, 180, 0, label="Angle horizontal initial (°)")
                        v_vertical_angle = gr.Slider(-90, 90, 15, label="Angle vertical initial (°)")
                        v_focal_length = gr.Slider(24, 200, 50, label="Focale (mm)")
                        v_camera_height = gr.Slider(0, 100, 25, label="Hauteur caméra")
                        v_camera_distance = gr.Slider(10, 500, 150, label="Distance")

                        v_width = gr.Slider(512, 2048, 1024, step=64, label="Largeur")
                        v_height = gr.Slider(512, 2048, 576, step=64, label="Hauteur")
                        v_steps = gr.Slider(20, 100, 30, label="Steps (réduit pour vidéo)")
                        v_seed = gr.Number(value=42, label="Seed de départ")
                        v_detail = gr.Slider(0, 100, 80, label="Niveau de détail")

                    with gr.Column():
                        gr.Markdown("### 🎬 Paramètres Vidéo")
                        num_frames = gr.Slider(3, 30, 8, step=1, label="Nombre de frames")
                        camera_path_type = gr.Dropdown(
                            ["Orbit", "Pan", "Zoom", "Flyover", "Static"],
                            value="Orbit",
                            label="Type de mouvement caméra"
                        )
                        video_fps = gr.Slider(12, 60, 24, step=1, label="FPS de la vidéo")
                        add_transitions = gr.Checkbox(value=True, label="Transitions douces")

                        generate_video_btn = gr.Button("🎬 Générer la vidéo", variant="primary", size="lg")

                with gr.Row():
                    video_output = gr.Video(label="Vidéo générée")
                    video_status = gr.Textbox(label="Status de génération", lines=20)

                with gr.Row():
                    video_prompts = gr.Textbox(label="Prompts utilisés", lines=10)

                generate_video_btn.click(
                    self.generate_video_sequence,
                    inputs=[
                        v_mountain_type, v_mountain_height, v_tree_density, v_tree_type,
                        v_sky_type, v_lighting, v_weather, v_season,
                        v_horizontal_angle, v_vertical_angle, v_focal_length,
                        v_camera_height, v_camera_distance,
                        v_width, v_height, v_steps, v_seed, v_detail,
                        num_frames, camera_path_type, video_fps, add_transitions
                    ],
                    outputs=[video_output, video_status, video_prompts]
                )

            gr.Markdown("""
            ---
            ### 📖 Guide d'utilisation:
            1. **Configuration**: Choisissez votre backend (ComfyUI ou Stable Diffusion) et initialisez-le
            2. **Image Unique**: Ajustez tous les paramètres et générez une image
            3. **Vidéo**: Configurez le mouvement de caméra et générez une séquence animée

            **Types de mouvements caméra**:
            - **Orbit**: Rotation à 360° autour de la montagne
            - **Pan**: Panoramique horizontal
            - **Zoom**: Zoom progressif sur la scène
            - **Flyover**: Survol cinématique des montagnes
            - **Static**: Aucun mouvement (pour tester)
            """)

        return interface


def main():
    """Point d'entrée de l'application"""
    app = MountainSimulationApp()
    interface = app.create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()
