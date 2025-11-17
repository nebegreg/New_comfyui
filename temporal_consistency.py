"""
Système de cohérence temporelle pour génération vidéo
Évite les changements abrupts entre frames
Utilise ControlNet, img2img, et frame interpolation
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Optional, Tuple
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler
)
from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector
import os


class TemporalConsistencyEngine:
    """Assure la cohérence temporelle entre les frames vidéo"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.controlnet_pipe = None
        self.img2img_pipe = None
        self.canny_detector = CannyDetector()
        self.depth_detector = None
        self.previous_frame = None
        self.previous_latent = None

    def load_models(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Charge les modèles pour cohérence temporelle"""
        print("🔧 Chargement des modèles de cohérence temporelle...")

        try:
            # ControlNet pour guidance structurelle
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16
            )

            # Pipeline ControlNet
            self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_name,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(self.device)

            # Pipeline img2img pour cohérence frame-to-frame
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(self.device)

            # Utiliser DDIM scheduler pour meilleure cohérence
            self.controlnet_pipe.scheduler = DDIMScheduler.from_config(
                self.controlnet_pipe.scheduler.config
            )
            self.img2img_pipe.scheduler = DDIMScheduler.from_config(
                self.img2img_pipe.scheduler.config
            )

            # Activer attention slicing pour économiser VRAM
            self.controlnet_pipe.enable_attention_slicing()
            self.img2img_pipe.enable_attention_slicing()

            print("✓ Modèles chargés avec succès")
            return True

        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles: {e}")
            return False

    def generate_first_frame(self,
                            prompt: str,
                            negative_prompt: str,
                            width: int,
                            height: int,
                            steps: int,
                            seed: int,
                            control_image: Optional[Image.Image] = None) -> Image.Image:
        """
        Génère la première frame avec ControlNet si une image de contrôle est fournie

        Args:
            prompt: Prompt de génération
            negative_prompt: Negative prompt
            width, height: Dimensions
            steps: Nombre de steps
            seed: Seed
            control_image: Image de contrôle (heightmap, canny, etc.)

        Returns:
            Première frame générée
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if control_image is not None and self.controlnet_pipe is not None:
            # Générer avec ControlNet pour structure cohérente
            control_image = control_image.resize((width, height))

            # Extraire les contours Canny
            canny_image = self.canny_detector(control_image)

            image = self.controlnet_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                controlnet_conditioning_scale=0.5
            ).images[0]
        else:
            # Génération standard sans contrôle
            if self.img2img_pipe is not None:
                # Créer une image initiale aléatoire
                init_image = Image.new('RGB', (width, height), color=(128, 128, 128))
                image = self.img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=1.0,
                    num_inference_steps=steps,
                    generator=generator
                ).images[0]
            else:
                raise ValueError("Aucun pipeline disponible")

        # Sauvegarder pour la prochaine frame
        self.previous_frame = image

        return image

    def generate_next_frame(self,
                           prompt: str,
                           negative_prompt: str,
                           steps: int,
                           seed: int,
                           strength: float = 0.3,
                           control_image: Optional[Image.Image] = None) -> Image.Image:
        """
        Génère la frame suivante en utilisant la frame précédente comme base

        Args:
            prompt: Prompt (peut varier légèrement)
            negative_prompt: Negative prompt
            steps: Nombre de steps
            seed: Seed
            strength: Force de transformation (0.1-0.5 pour cohérence)
            control_image: Image de contrôle (heightmap de la même montagne)

        Returns:
            Frame suivante cohérente
        """
        if self.previous_frame is None:
            raise ValueError("Appelez d'abord generate_first_frame()")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if control_image is not None and self.controlnet_pipe is not None:
            # Utiliser ControlNet + img2img pour cohérence maximale
            width, height = self.previous_frame.size
            control_image = control_image.resize((width, height))
            canny_image = self.canny_detector(control_image)

            # Générer avec guidance structurelle ET frame précédente
            image = self.controlnet_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=steps,
                generator=generator,
                controlnet_conditioning_scale=0.3  # Plus faible pour variation
            ).images[0]

            # Blend avec frame précédente
            blend_factor = 1.0 - strength
            image = Image.blend(self.previous_frame, image, strength)

        else:
            # Img2img pur avec frame précédente
            image = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=self.previous_frame,
                strength=strength,  # Faible strength = haute cohérence
                num_inference_steps=steps,
                generator=generator
            ).images[0]

        # Sauvegarder pour la prochaine frame
        self.previous_frame = image

        return image

    def reset(self):
        """Reset l'état pour une nouvelle séquence"""
        self.previous_frame = None
        self.previous_latent = None


class FrameInterpolator:
    """Interpolation entre frames pour fluidité"""

    def __init__(self):
        self.method = "optical_flow"

    def interpolate_optical_flow(self,
                                 frame1: Image.Image,
                                 frame2: Image.Image,
                                 num_intermediate: int = 3) -> List[Image.Image]:
        """
        Interpole entre deux frames avec optical flow

        Args:
            frame1: Première frame
            frame2: Deuxième frame
            num_intermediate: Nombre de frames intermédiaires

        Returns:
            Liste de frames interpolées
        """
        # Convertir en numpy
        img1 = np.array(frame1)
        img2 = np.array(frame2)

        # Convertir en grayscale pour optical flow
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Calculer optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Générer les frames intermédiaires
        interpolated = []
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)

            # Warping de l'image 1 vers l'image 2
            h, w = flow.shape[:2]
            flow_t = flow * t
            map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
            map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
            map_x += flow_t[:, :, 0]
            map_y += flow_t[:, :, 1]

            warped1 = cv2.remap(img1, map_x, map_y, cv2.INTER_LINEAR)

            # Blend simple
            alpha = t
            blended = cv2.addWeighted(warped1, 1-alpha, img2, alpha, 0)

            interpolated.append(Image.fromarray(blended))

        return interpolated

    def interpolate_simple_blend(self,
                                 frame1: Image.Image,
                                 frame2: Image.Image,
                                 num_intermediate: int = 3) -> List[Image.Image]:
        """
        Interpolation simple par blending

        Args:
            frame1: Première frame
            frame2: Deuxième frame
            num_intermediate: Nombre de frames intermédiaires

        Returns:
            Liste de frames interpolées
        """
        interpolated = []
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            blended = Image.blend(frame1, frame2, alpha)
            interpolated.append(blended)

        return interpolated


class VideoCoherenceManager:
    """Gestionnaire principal pour vidéos cohérentes"""

    def __init__(self, device: str = "cuda"):
        self.temporal_engine = TemporalConsistencyEngine(device)
        self.interpolator = FrameInterpolator()
        self.heightmap_sequence = []

    def load_models(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Charge les modèles nécessaires"""
        return self.temporal_engine.load_models(model_name)

    def generate_coherent_video(self,
                               base_prompt: str,
                               negative_prompt: str,
                               camera_params: List[dict],
                               heightmap_base: Optional[np.ndarray],
                               width: int = 1024,
                               height: int = 768,
                               steps: int = 30,
                               seed: int = 42,
                               strength: float = 0.25,
                               interpolate: bool = True,
                               interpolation_frames: int = 2) -> List[Image.Image]:
        """
        Génère une vidéo cohérente avec la même montagne vue sous différents angles

        Args:
            base_prompt: Prompt de base
            negative_prompt: Negative prompt
            camera_params: Liste de paramètres de caméra pour chaque frame
            heightmap_base: Heightmap de la montagne (reste identique)
            width, height: Dimensions
            steps: Steps de diffusion
            seed: Seed de base
            strength: Force de variation entre frames (0.1-0.4 recommandé)
            interpolate: Ajouter des frames interpolées
            interpolation_frames: Nombre de frames interpolées entre chaque frame générée

        Returns:
            Liste de toutes les frames
        """
        all_frames = []

        # Convertir heightmap en image de contrôle
        control_image = None
        if heightmap_base is not None:
            control_image = Image.fromarray(
                (heightmap_base * 255).astype(np.uint8),
                mode='L'
            ).convert('RGB')

        # Générer la première frame
        print(f"🎬 Génération frame 1/{len(camera_params)}")
        first_frame = self.temporal_engine.generate_first_frame(
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            control_image=control_image
        )
        all_frames.append(first_frame)

        # Générer les frames suivantes avec cohérence
        for i, cam_params in enumerate(camera_params[1:], start=2):
            print(f"🎬 Génération frame {i}/{len(camera_params)}")

            # Variation subtile du prompt selon la caméra
            # mais SANS changer la montagne elle-même
            varied_prompt = base_prompt  # On garde le même prompt !

            # Générer frame cohérente
            frame = self.temporal_engine.generate_next_frame(
                prompt=varied_prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed + i,  # Légère variation de seed
                strength=strength,  # Faible strength = haute cohérence
                control_image=control_image  # Même heightmap !
            )

            # Interpoler avec la frame précédente si demandé
            if interpolate and len(all_frames) > 0:
                print(f"  ↳ Interpolation de {interpolation_frames} frames...")
                interpolated = self.interpolator.interpolate_optical_flow(
                    all_frames[-1], frame, interpolation_frames
                )
                all_frames.extend(interpolated)

            all_frames.append(frame)

        # Reset pour la prochaine vidéo
        self.temporal_engine.reset()

        return all_frames

    def generate_from_heightmap_rotation(self,
                                        heightmap: np.ndarray,
                                        base_prompt: str,
                                        negative_prompt: str,
                                        num_frames: int = 24,
                                        width: int = 1024,
                                        height: int = 768,
                                        steps: int = 30,
                                        seed: int = 42) -> List[Image.Image]:
        """
        Génère une rotation autour de la même montagne (heightmap)

        La heightmap reste identique, seul l'angle de vue change

        Args:
            heightmap: Heightmap 3D de la montagne
            base_prompt: Prompt de base
            negative_prompt: Negative prompt
            num_frames: Nombre de frames pour rotation complète
            width, height: Dimensions
            steps: Steps
            seed: Seed

        Returns:
            Frames de rotation
        """
        # TODO: Implémenter la rotation réelle du heightmap en 3D
        # Pour l'instant, on utilise le système de cohérence avec même heightmap

        camera_params = [{'angle': i * 360 / num_frames} for i in range(num_frames)]

        return self.generate_coherent_video(
            base_prompt=base_prompt,
            negative_prompt=negative_prompt,
            camera_params=camera_params,
            heightmap_base=heightmap,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            strength=0.2,  # Très faible pour cohérence maximale
            interpolate=True,
            interpolation_frames=1
        )
