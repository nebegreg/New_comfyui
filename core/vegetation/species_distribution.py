"""
Distribution d'espèces d'arbres basée sur des règles écologiques

Gère:
- Zones de distribution (altitude, température, moisture)
- Mixité des espèces
- Ratio entre espèces
- Règles de competition
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SpeciesProfile:
    """Profil écologique d'une espèce d'arbre"""
    name: str
    common_name: str

    # Altitude range (0-1)
    min_elevation: float
    max_elevation: float
    optimal_elevation: float

    # Température range (0-1)
    min_temperature: float
    max_temperature: float

    # Moisture range (0-1)
    min_moisture: float
    max_moisture: float
    optimal_moisture: float

    # Slope tolerance (0-1+)
    max_slope: float

    # Spacing (meters or pixels)
    min_spacing: float
    typical_spacing: float

    # Visual properties
    typical_height: float  # meters
    typical_crown_radius: float  # meters

    # Description for prompts
    description: str


class SpeciesDistributor:
    """
    Distribue les espèces selon leurs profils écologiques
    """

    def __init__(self):
        self.species_database = self._create_species_database()

    def _create_species_database(self) -> Dict[str, SpeciesProfile]:
        """Crée la base de données d'espèces"""

        species = {
            'pine': SpeciesProfile(
                name='pine',
                common_name='Pine (Pinus)',
                min_elevation=0.2,
                max_elevation=0.8,
                optimal_elevation=0.5,
                min_temperature=0.3,
                max_temperature=0.9,
                min_moisture=0.2,
                max_moisture=0.8,
                optimal_moisture=0.5,
                max_slope=0.7,
                min_spacing=3.0,
                typical_spacing=5.0,
                typical_height=25.0,
                typical_crown_radius=4.0,
                description='tall pine tree, coniferous, needle foliage, brown bark'
            ),

            'spruce': SpeciesProfile(
                name='spruce',
                common_name='Spruce (Picea)',
                min_elevation=0.3,
                max_elevation=0.85,
                optimal_elevation=0.6,
                min_temperature=0.2,
                max_temperature=0.7,
                min_moisture=0.4,
                max_moisture=0.9,
                optimal_moisture=0.7,
                max_slope=0.6,
                min_spacing=3.5,
                typical_spacing=5.5,
                typical_height=30.0,
                typical_crown_radius=3.5,
                description='tall spruce tree, conical shape, dense dark green foliage'
            ),

            'fir': SpeciesProfile(
                name='fir',
                common_name='Fir (Abies)',
                min_elevation=0.25,
                max_elevation=0.75,
                optimal_elevation=0.5,
                min_temperature=0.25,
                max_temperature=0.75,
                min_moisture=0.5,
                max_moisture=0.9,
                optimal_moisture=0.7,
                max_slope=0.5,
                min_spacing=4.0,
                typical_spacing=6.0,
                typical_height=35.0,
                typical_crown_radius=5.0,
                description='majestic fir tree, silver-green needles, symmetrical conical shape'
            ),

            'deciduous': SpeciesProfile(
                name='deciduous',
                common_name='Deciduous Mix',
                min_elevation=0.0,
                max_elevation=0.5,
                optimal_elevation=0.25,
                min_temperature=0.4,
                max_temperature=1.0,
                min_moisture=0.5,
                max_moisture=1.0,
                optimal_moisture=0.8,
                max_slope=0.4,
                min_spacing=4.0,
                typical_spacing=7.0,
                typical_height=20.0,
                typical_crown_radius=6.0,
                description='deciduous tree, broad leaves, rounded crown, mixed hardwood'
            )
        }

        return species

    def get_suitable_species(
        self,
        elevation: float,
        temperature: float,
        moisture: float,
        slope: float
    ) -> List[str]:
        """
        Retourne les espèces adaptées à ces conditions

        Args:
            elevation: Altitude (0-1)
            temperature: Température (0-1)
            moisture: Humidité (0-1)
            slope: Pente (0-1+)

        Returns:
            Liste de noms d'espèces adaptées
        """
        suitable = []

        for species_name, profile in self.species_database.items():
            # Vérifier si conditions dans les ranges
            if not (profile.min_elevation <= elevation <= profile.max_elevation):
                continue
            if not (profile.min_temperature <= temperature <= profile.max_temperature):
                continue
            if not (profile.min_moisture <= moisture <= profile.max_moisture):
                continue
            if slope > profile.max_slope:
                continue

            suitable.append(species_name)

        return suitable

    def calculate_suitability_score(
        self,
        species_name: str,
        elevation: float,
        temperature: float,
        moisture: float,
        slope: float
    ) -> float:
        """
        Calcule un score de convenance (0-1)
        1.0 = conditions parfaites, 0.0 = impossible
        """
        if species_name not in self.species_database:
            return 0.0

        profile = self.species_database[species_name]

        score = 1.0

        # Pénalité selon distance à l'optimal
        # Elevation
        if elevation < profile.min_elevation or elevation > profile.max_elevation:
            return 0.0
        elev_dist = abs(elevation - profile.optimal_elevation)
        elev_range = (profile.max_elevation - profile.min_elevation) / 2
        score *= 1.0 - (elev_dist / elev_range) * 0.5

        # Temperature
        if temperature < profile.min_temperature or temperature > profile.max_temperature:
            return 0.0
        temp_range = (profile.max_temperature - profile.min_temperature)
        temp_optimal = (profile.min_temperature + profile.max_temperature) / 2
        temp_dist = abs(temperature - temp_optimal)
        score *= 1.0 - (temp_dist / temp_range) * 0.3

        # Moisture
        if moisture < profile.min_moisture or moisture > profile.max_moisture:
            return 0.0
        moist_dist = abs(moisture - profile.optimal_moisture)
        moist_range = (profile.max_moisture - profile.min_moisture) / 2
        score *= 1.0 - (moist_dist / moist_range) * 0.4

        # Slope
        if slope > profile.max_slope:
            return 0.0
        slope_factor = 1.0 - (slope / profile.max_slope) * 0.2
        score *= slope_factor

        return score

    def get_species_mix(
        self,
        elevation: float,
        temperature: float,
        moisture: float,
        slope: float
    ) -> Dict[str, float]:
        """
        Retourne un mix d'espèces avec leurs proportions

        Returns:
            Dict {species_name: proportion} où sum(proportions) = 1.0
        """
        # Calculer scores pour toutes les espèces
        scores = {}
        for species_name in self.species_database.keys():
            score = self.calculate_suitability_score(
                species_name,
                elevation,
                temperature,
                moisture,
                slope
            )
            if score > 0:
                scores[species_name] = score

        if not scores:
            # Aucune espèce convenable, retourner pine par défaut
            return {'pine': 1.0}

        # Normaliser en proportions
        total_score = sum(scores.values())
        proportions = {
            species: score / total_score
            for species, score in scores.items()
        }

        return proportions

    def get_species_description(self, species_name: str) -> str:
        """Retourne la description pour prompts AI"""
        if species_name not in self.species_database:
            return "generic tree"

        return self.species_database[species_name].description

    def get_all_species(self) -> List[str]:
        """Retourne toutes les espèces disponibles"""
        return list(self.species_database.keys())
