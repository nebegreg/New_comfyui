"""
Système de caméra pour la simulation de montagne
Gère les angles, la focale et génère les prompts appropriés
"""

import numpy as np
from typing import Dict, Tuple, List


class CameraSystem:
    """Gère les paramètres de caméra et génère les prompts pour la génération d'images"""

    def __init__(self):
        self.horizontal_angle = 0  # -180 à 180 degrés
        self.vertical_angle = 0    # -90 à 90 degrés
        self.distance = 100        # Distance de la scène
        self.focal_length = 50     # Focale en mm (24-200mm)
        self.height = 10           # Hauteur de la caméra

    def set_camera(self, horizontal: float, vertical: float, focal: float, height: float, distance: float):
        """Configure les paramètres de la caméra"""
        self.horizontal_angle = np.clip(horizontal, -180, 180)
        self.vertical_angle = np.clip(vertical, -90, 90)
        self.focal_length = np.clip(focal, 24, 200)
        self.height = np.clip(height, 0, 100)
        self.distance = np.clip(distance, 10, 500)

    def get_camera_description(self) -> str:
        """Génère une description de la vue de caméra pour le prompt"""
        descriptions = []

        # Angle vertical
        if self.vertical_angle < -30:
            descriptions.append("low angle view, looking up at towering mountains")
        elif self.vertical_angle < -10:
            descriptions.append("slightly low angle, dramatic mountain perspective")
        elif self.vertical_angle > 30:
            descriptions.append("high angle aerial view, bird's eye perspective")
        elif self.vertical_angle > 10:
            descriptions.append("elevated viewpoint, overlooking mountain landscape")
        else:
            descriptions.append("eye-level view, immersive mountain scene")

        # Distance et focale
        if self.focal_length < 35:
            descriptions.append("wide angle lens, expansive vista")
        elif self.focal_length > 85:
            descriptions.append("telephoto lens, compressed perspective, distant peaks")
        else:
            descriptions.append("standard lens, natural perspective")

        # Hauteur
        if self.height < 5:
            descriptions.append("ground level")
        elif self.height > 50:
            descriptions.append("high altitude viewpoint")
        else:
            descriptions.append("medium elevation")

        return ", ".join(descriptions)

    def generate_camera_path(self, num_frames: int, path_type: str = "orbit") -> List[Dict]:
        """Génère un chemin de caméra pour une animation"""
        frames = []

        if path_type == "orbit":
            # Rotation autour des montagnes
            for i in range(num_frames):
                angle = (i / num_frames) * 360 - 180
                frames.append({
                    'horizontal': angle,
                    'vertical': self.vertical_angle,
                    'focal': self.focal_length,
                    'height': self.height,
                    'distance': self.distance
                })

        elif path_type == "pan":
            # Panoramique horizontal
            start_angle = self.horizontal_angle - 45
            for i in range(num_frames):
                angle = start_angle + (i / num_frames) * 90
                frames.append({
                    'horizontal': angle,
                    'vertical': self.vertical_angle,
                    'focal': self.focal_length,
                    'height': self.height,
                    'distance': self.distance
                })

        elif path_type == "zoom":
            # Zoom progressif
            start_focal = 24
            end_focal = 200
            for i in range(num_frames):
                focal = start_focal + (i / num_frames) * (end_focal - start_focal)
                frames.append({
                    'horizontal': self.horizontal_angle,
                    'vertical': self.vertical_angle,
                    'focal': focal,
                    'height': self.height,
                    'distance': self.distance
                })

        elif path_type == "flyover":
            # Survol des montagnes
            for i in range(num_frames):
                progress = i / num_frames
                height = self.height + np.sin(progress * np.pi) * 30
                distance = self.distance - progress * 50
                vertical = self.vertical_angle - progress * 20
                frames.append({
                    'horizontal': self.horizontal_angle,
                    'vertical': vertical,
                    'focal': self.focal_length,
                    'height': height,
                    'distance': distance
                })
        else:
            # Statique
            for i in range(num_frames):
                frames.append({
                    'horizontal': self.horizontal_angle,
                    'vertical': self.vertical_angle,
                    'focal': self.focal_length,
                    'height': self.height,
                    'distance': self.distance
                })

        return frames

    def get_depth_of_field(self) -> str:
        """Calcule la description de la profondeur de champ basée sur la focale"""
        if self.focal_length > 85:
            return "shallow depth of field, background blur, bokeh effect"
        elif self.focal_length < 35:
            return "deep depth of field, everything in focus"
        else:
            return "moderate depth of field"
