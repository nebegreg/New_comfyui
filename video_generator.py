"""
Générateur de vidéo à partir de séquences d'images
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import os


class VideoGenerator:
    """Crée des vidéos à partir de séquences d'images"""

    def __init__(self):
        self.fps = 24
        self.codec = 'mp4v'

    def create_video_from_images(self, images: List[Image.Image], output_path: str,
                                 fps: int = 24, add_transitions: bool = True,
                                 transition_frames: int = 5) -> bool:
        """
        Crée une vidéo à partir d'une liste d'images

        Args:
            images: Liste d'images PIL
            output_path: Chemin de sortie pour la vidéo
            fps: Images par seconde
            add_transitions: Ajouter des transitions douces entre les images
            transition_frames: Nombre de frames de transition

        Returns:
            bool: True si succès
        """
        if not images:
            print("Aucune image à traiter")
            return False

        try:
            # Convertir la première image pour obtenir les dimensions
            first_frame = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            height, width, _ = first_frame.shape

            # Créer le writer vidéo
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if add_transitions and len(images) > 1:
                # Avec transitions douces
                for i in range(len(images)):
                    current_img = cv2.cvtColor(np.array(images[i]), cv2.COLOR_RGB2BGR)

                    # Écrire l'image principale plusieurs fois
                    for _ in range(fps // 2):  # Maintenir chaque image pendant 0.5 seconde
                        out.write(current_img)

                    # Transition vers l'image suivante
                    if i < len(images) - 1:
                        next_img = cv2.cvtColor(np.array(images[i + 1]), cv2.COLOR_RGB2BGR)

                        for t in range(transition_frames):
                            alpha = t / transition_frames
                            blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
                            out.write(blended)
            else:
                # Sans transitions
                for img in images:
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    for _ in range(fps):  # Maintenir chaque image pendant 1 seconde
                        out.write(frame)

            out.release()
            print(f"Vidéo créée avec succès: {output_path}")
            return True

        except Exception as e:
            print(f"Erreur lors de la création de la vidéo: {e}")
            return False

    def add_camera_motion_blur(self, image: Image.Image, motion_vector: tuple) -> Image.Image:
        """Ajoute un flou de mouvement pour simuler le mouvement de caméra"""
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Créer un noyau de flou directionnel
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Appliquer le flou
        blurred = cv2.filter2D(img_cv, -1, kernel)
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        return Image.fromarray(blurred_rgb)

    def interpolate_frames(self, img1: Image.Image, img2: Image.Image, num_frames: int = 5) -> List[Image.Image]:
        """
        Interpole entre deux images pour créer des frames intermédiaires

        Args:
            img1: Première image
            img2: Deuxième image
            num_frames: Nombre de frames à générer entre les deux images

        Returns:
            Liste d'images interpolées
        """
        frames = []
        arr1 = np.array(img1).astype(np.float32)
        arr2 = np.array(img2).astype(np.float32)

        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0.5
            interpolated = (1 - alpha) * arr1 + alpha * arr2
            interpolated = interpolated.astype(np.uint8)
            frames.append(Image.fromarray(interpolated))

        return frames

    def add_zoom_effect(self, image: Image.Image, num_frames: int = 30, zoom_factor: float = 1.5) -> List[Image.Image]:
        """
        Crée un effet de zoom sur une image statique

        Args:
            image: Image de base
            num_frames: Nombre de frames à générer
            zoom_factor: Facteur de zoom final

        Returns:
            Liste d'images avec effet de zoom
        """
        frames = []
        width, height = image.size
        center_x, center_y = width // 2, height // 2

        for i in range(num_frames):
            progress = i / (num_frames - 1)
            current_zoom = 1.0 + (zoom_factor - 1.0) * progress

            # Calculer les nouvelles dimensions
            new_width = int(width / current_zoom)
            new_height = int(height / current_zoom)

            # Calculer les coordonnées de crop
            left = center_x - new_width // 2
            top = center_y - new_height // 2
            right = left + new_width
            bottom = top + new_height

            # Crop et resize
            cropped = image.crop((left, top, right, bottom))
            zoomed = cropped.resize((width, height), Image.Resampling.LANCZOS)
            frames.append(zoomed)

        return frames

    def add_pan_effect(self, image: Image.Image, num_frames: int = 30, direction: str = 'left') -> List[Image.Image]:
        """
        Crée un effet de panoramique sur une image statique

        Args:
            image: Image de base (doit être plus large que la résolution finale)
            num_frames: Nombre de frames à générer
            direction: Direction du pan ('left', 'right', 'up', 'down')

        Returns:
            Liste d'images avec effet de panoramique
        """
        frames = []
        width, height = image.size
        target_width = width // 2  # On suppose que l'image est 2x plus large
        target_height = height

        for i in range(num_frames):
            progress = i / (num_frames - 1)

            if direction == 'left':
                left = int((width - target_width) * progress)
                top = 0
            elif direction == 'right':
                left = int((width - target_width) * (1 - progress))
                top = 0
            elif direction == 'up':
                left = 0
                top = int((height - target_height) * progress)
            else:  # down
                left = 0
                top = int((height - target_height) * (1 - progress))

            right = left + target_width
            bottom = top + target_height

            cropped = image.crop((left, top, right, bottom))
            frames.append(cropped)

        return frames

    def save_frames(self, frames: List[Image.Image], output_dir: str, prefix: str = "frame") -> List[str]:
        """Sauvegarde une liste de frames dans un dossier"""
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            frame.save(path)
            paths.append(path)

        return paths
