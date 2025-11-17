"""
Système de végétation procédurale pour terrains ultra-réalistes
"""

from .biome_classifier import BiomeClassifier
from .vegetation_placer import VegetationPlacer, TreeInstance
from .species_distribution import SpeciesDistributor

__all__ = [
    'BiomeClassifier',
    'VegetationPlacer',
    'TreeInstance',
    'SpeciesDistributor'
]
