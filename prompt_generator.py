"""
Générateur de prompts pour la simulation de montagne
Crée des prompts optimisés pour Stable Diffusion
"""

from typing import Dict
import random


class MountainPromptGenerator:
    """Génère des prompts optimisés pour créer des scènes de montagne réalistes"""

    def __init__(self):
        self.quality_tags = [
            "photorealistic", "highly detailed", "8k uhd", "professional photography",
            "sharp focus", "physically-based rendering", "extreme detail description",
            "masterpiece", "best quality"
        ]

        self.mountain_types = {
            "alpine": "jagged alpine peaks, snow-capped mountains, rocky cliffs",
            "rolling": "rolling mountain hills, gentle slopes, layered ridges",
            "volcanic": "volcanic mountain, dramatic peak, rugged terrain",
            "massive": "massive mountain range, towering peaks, dramatic elevation",
            "rocky": "rocky mountain faces, exposed stone, weathered rock formations"
        }

        self.tree_types = {
            "pine": "dense pine forest, coniferous trees, evergreen coverage",
            "spruce": "spruce trees, northern forest, dark green foliage",
            "mixed": "mixed forest, variety of trees, natural vegetation",
            "sparse": "sparse tree coverage, scattered pines, alpine treeline",
            "dense": "dense forest coverage, thick woodland, lush vegetation"
        }

        self.sky_types = {
            "clear": "clear blue sky, perfect weather, bright sunlight",
            "cloudy": "dramatic clouds, volumetric clouds, dynamic sky",
            "sunset": "golden hour, sunset lighting, warm orange glow, dramatic sky",
            "sunrise": "sunrise, morning light, soft pink and orange hues",
            "stormy": "dramatic storm clouds, moody atmosphere, dark clouds",
            "overcast": "overcast sky, soft diffused lighting, grey clouds",
            "partly_cloudy": "partly cloudy, scattered clouds, dynamic lighting"
        }

        self.lighting_conditions = {
            "golden": "golden hour lighting, warm sunlight, long shadows",
            "midday": "bright midday sun, clear visibility, strong lighting",
            "dramatic": "dramatic lighting, god rays, volumetric light",
            "soft": "soft natural lighting, diffused light, gentle illumination",
            "backlit": "backlit scene, rim lighting, silhouetted peaks"
        }

        self.weather_effects = {
            "clear": "clear weather, high visibility",
            "fog": "morning fog, mist in valleys, atmospheric haze",
            "snow": "fresh snow, winter scene, pristine white coverage",
            "rain": "after rain, wet surfaces, dramatic atmosphere"
        }

        self.seasons = {
            "spring": "spring season, fresh green vegetation, blooming flowers",
            "summer": "summer, lush green forest, vibrant colors",
            "autumn": "autumn colors, fall foliage, orange and red leaves",
            "winter": "winter landscape, snow covered, frozen"
        }

    def generate_prompt(self, params: Dict) -> tuple[str, str]:
        """
        Génère un prompt complet basé sur les paramètres

        Args:
            params: Dictionnaire avec les paramètres de la scène
                - mountain_type: Type de montagne
                - mountain_height: Hauteur relative (0-100)
                - tree_density: Densité des arbres (0-100)
                - tree_type: Type d'arbres
                - sky_type: Type de ciel
                - lighting: Conditions d'éclairage
                - weather: Conditions météo
                - season: Saison
                - camera_desc: Description de la caméra

        Returns:
            tuple: (prompt, negative_prompt)
        """
        prompt_parts = []

        # Tags de qualité
        prompt_parts.append(", ".join(random.sample(self.quality_tags, 4)))

        # Description de la montagne
        mountain_type = params.get('mountain_type', 'alpine')
        mountain_height = params.get('mountain_height', 50)

        mountain_desc = self.mountain_types.get(mountain_type, self.mountain_types['alpine'])

        if mountain_height > 75:
            mountain_desc += ", towering peaks, extreme elevation, massive scale"
        elif mountain_height > 50:
            mountain_desc += ", tall mountains, impressive height"
        elif mountain_height > 25:
            mountain_desc += ", moderate elevation, prominent peaks"
        else:
            mountain_desc += ", gentle mountains, rolling terrain"

        prompt_parts.append(mountain_desc)

        # Arbres et végétation
        tree_density = params.get('tree_density', 50)
        tree_type = params.get('tree_type', 'pine')

        if tree_density > 75:
            tree_desc = self.tree_types.get(tree_type, self.tree_types['pine'])
            tree_desc += ", thick forest coverage, abundant vegetation"
        elif tree_density > 50:
            tree_desc = self.tree_types.get(tree_type, self.tree_types['pine'])
        elif tree_density > 25:
            tree_desc = self.tree_types.get('sparse', "scattered trees, light vegetation")
        else:
            tree_desc = "minimal vegetation, alpine meadows, above treeline"

        if tree_density > 0:
            prompt_parts.append(tree_desc)

        # Ciel et météo
        sky_type = params.get('sky_type', 'clear')
        prompt_parts.append(self.sky_types.get(sky_type, self.sky_types['clear']))

        # Conditions d'éclairage
        lighting = params.get('lighting', 'dramatic')
        prompt_parts.append(self.lighting_conditions.get(lighting, self.lighting_conditions['dramatic']))

        # Météo
        weather = params.get('weather', 'clear')
        prompt_parts.append(self.weather_effects.get(weather, self.weather_effects['clear']))

        # Saison
        season = params.get('season', 'summer')
        prompt_parts.append(self.seasons.get(season, self.seasons['summer']))

        # Description de la caméra
        camera_desc = params.get('camera_desc', '')
        if camera_desc:
            prompt_parts.append(camera_desc)

        # Détails supplémentaires pour le réalisme
        prompt_parts.append("natural landscape, realistic terrain, authentic mountain scene")
        prompt_parts.append("high dynamic range, rich colors, natural color grading")

        # Prompt final
        prompt = ", ".join(prompt_parts)

        # Negative prompt pour éviter les artefacts
        negative_prompt = (
            "low quality, blurry, distorted, unrealistic, artificial, cartoon, anime, "
            "painting, drawing, illustration, cgi, 3d render, bad anatomy, deformed, "
            "ugly, artifacts, watermark, text, signature, low resolution, pixelated, "
            "oversaturated, people, person, human, building, city, urban"
        )

        return prompt, negative_prompt

    def add_detail_enhancement(self, prompt: str, detail_level: int = 80) -> str:
        """Ajoute des tags pour améliorer les détails"""
        if detail_level > 80:
            prompt += ", ultra detailed, hyper realistic, exceptional detail, professional photography"
        elif detail_level > 60:
            prompt += ", highly detailed, realistic, professional quality"
        else:
            prompt += ", detailed, natural"

        return prompt
