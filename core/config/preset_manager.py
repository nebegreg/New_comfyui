"""
Preset Manager for Mountain Studio Pro v2.0
Manages professional terrain generation presets
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class PresetManager:
    """
    Gestion centralisée des presets de génération de terrain

    Chaque preset contient:
    - Paramètres de terrain (scale, octaves, persistence, lacunarity)
    - Configuration d'érosion (hydraulique, thermique)
    - Paramètres de végétation
    - Paramètres de rendu PBR
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: Dossier de configuration (défaut: ./config)
        """
        if config_dir is None:
            config_dir = os.path.join(os.getcwd(), 'config')

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Fichier de presets
        self.presets_file = self.config_dir / 'presets.json'

        # Charger ou créer presets par défaut
        if self.presets_file.exists():
            with open(self.presets_file, 'r') as f:
                self.presets = json.load(f)
        else:
            self.presets = self._create_default_presets()
            self._save_presets()

    def _create_default_presets(self) -> Dict:
        """Crée les 12 presets professionnels par défaut"""

        return {
            # === PRESETS RAPIDES ===

            "quick_preview": {
                "name": "Quick Preview",
                "description": "Aperçu rapide pour tests (512x512, peu d'érosion)",
                "resolution": 512,
                "terrain": {
                    "base_scale": 80.0,
                    "octaves": 6,
                    "persistence": 0.5,
                    "lacunarity": 2.0,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 20,
                        "rain_amount": 0.008,
                        "evaporation": 0.5,
                        "capacity": 0.01,
                        "deposition": 0.1,
                        "erosion": 0.3
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 10,
                        "talus_angle": 0.7
                    }
                },
                "vegetation": {
                    "enabled": False
                }
            },

            "balanced_quality": {
                "name": "Balanced Quality",
                "description": "Équilibre qualité/vitesse (1024x1024)",
                "resolution": 1024,
                "terrain": {
                    "base_scale": 100.0,
                    "octaves": 8,
                    "persistence": 0.5,
                    "lacunarity": 2.0,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 50,
                        "rain_amount": 0.01,
                        "evaporation": 0.5,
                        "capacity": 0.01,
                        "deposition": 0.1,
                        "erosion": 0.3
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 30,
                        "talus_angle": 0.7
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.5,
                    "min_spacing": 3.0,
                    "use_clustering": True
                }
            },

            "high_detail_4k": {
                "name": "High Detail 4K",
                "description": "Haute qualité pour production (4096x4096)",
                "resolution": 4096,
                "terrain": {
                    "base_scale": 150.0,
                    "octaves": 10,
                    "persistence": 0.55,
                    "lacunarity": 2.1,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 100,
                        "rain_amount": 0.012,
                        "evaporation": 0.5,
                        "capacity": 0.01,
                        "deposition": 0.1,
                        "erosion": 0.3
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 50,
                        "talus_angle": 0.65
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.6,
                    "min_spacing": 2.5,
                    "use_clustering": True
                }
            },

            # === PRESETS VFX PROFESSIONNELS ===

            "vfx_epic_mountain": {
                "name": "VFX Epic Mountain",
                "description": "Montagne épique pour VFX (pics dramatiques, érosion forte)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 120.0,
                    "octaves": 10,
                    "persistence": 0.6,
                    "lacunarity": 2.2,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 80,
                        "rain_amount": 0.015,
                        "evaporation": 0.4,
                        "capacity": 0.012,
                        "deposition": 0.08,
                        "erosion": 0.35
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 40,
                        "talus_angle": 0.6
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.4,
                    "min_spacing": 3.5,
                    "use_clustering": True
                }
            },

            "vfx_gentle_hills": {
                "name": "VFX Gentle Hills",
                "description": "Collines douces pour VFX (paysages paisibles)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 200.0,
                    "octaves": 6,
                    "persistence": 0.45,
                    "lacunarity": 1.8,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 60,
                        "rain_amount": 0.012,
                        "evaporation": 0.6,
                        "capacity": 0.01,
                        "deposition": 0.12,
                        "erosion": 0.25
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 25,
                        "talus_angle": 0.75
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.7,
                    "min_spacing": 2.0,
                    "use_clustering": True
                }
            },

            "vfx_volcanic": {
                "name": "VFX Volcanic",
                "description": "Terrain volcanique (falaises abruptes, peu de végétation)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 90.0,
                    "octaves": 9,
                    "persistence": 0.65,
                    "lacunarity": 2.5,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 40,
                        "rain_amount": 0.008,
                        "evaporation": 0.7,
                        "capacity": 0.008,
                        "deposition": 0.15,
                        "erosion": 0.2
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 60,
                        "talus_angle": 0.5
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.2,
                    "min_spacing": 5.0,
                    "use_clustering": False
                }
            },

            # === PRESETS SPÉCIALISÉS ===

            "extreme_alps": {
                "name": "Extreme Alps",
                "description": "Alpes extrêmes (pics acérés, haute altitude)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 80.0,
                    "octaves": 12,
                    "persistence": 0.7,
                    "lacunarity": 2.4,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 70,
                        "rain_amount": 0.01,
                        "evaporation": 0.3,
                        "capacity": 0.015,
                        "deposition": 0.05,
                        "erosion": 0.4
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 50,
                        "talus_angle": 0.55
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.3,
                    "min_spacing": 4.0,
                    "use_clustering": True
                }
            },

            "desert_dunes": {
                "name": "Desert Dunes",
                "description": "Dunes de désert (ondulations douces, pas de végétation)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 250.0,
                    "octaves": 5,
                    "persistence": 0.4,
                    "lacunarity": 1.6,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": False
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 20,
                        "talus_angle": 0.8
                    }
                },
                "vegetation": {
                    "enabled": False
                }
            },

            "canyon_erosion": {
                "name": "Canyon Erosion",
                "description": "Canyon avec forte érosion hydraulique",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 110.0,
                    "octaves": 8,
                    "persistence": 0.55,
                    "lacunarity": 2.0,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 150,
                        "rain_amount": 0.02,
                        "evaporation": 0.4,
                        "capacity": 0.02,
                        "deposition": 0.05,
                        "erosion": 0.45
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 30,
                        "talus_angle": 0.6
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.4,
                    "min_spacing": 3.0,
                    "use_clustering": True
                }
            },

            "coastal_cliffs": {
                "name": "Coastal Cliffs",
                "description": "Falaises côtières (forte érosion, dénivelé important)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 95.0,
                    "octaves": 9,
                    "persistence": 0.6,
                    "lacunarity": 2.1,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 90,
                        "rain_amount": 0.018,
                        "evaporation": 0.5,
                        "capacity": 0.015,
                        "deposition": 0.08,
                        "erosion": 0.38
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 45,
                        "talus_angle": 0.58
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.35,
                    "min_spacing": 3.5,
                    "use_clustering": True
                }
            },

            "rolling_plains": {
                "name": "Rolling Plains",
                "description": "Plaines ondulées (végétation dense, peu de relief)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 300.0,
                    "octaves": 4,
                    "persistence": 0.35,
                    "lacunarity": 1.5,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 30,
                        "rain_amount": 0.01,
                        "evaporation": 0.7,
                        "capacity": 0.008,
                        "deposition": 0.15,
                        "erosion": 0.2
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 15,
                        "talus_angle": 0.85
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.8,
                    "min_spacing": 1.5,
                    "use_clustering": True
                }
            },

            "glacial_valley": {
                "name": "Glacial Valley",
                "description": "Vallée glaciaire (U-shaped valley, érosion spécifique)",
                "resolution": 2048,
                "terrain": {
                    "base_scale": 130.0,
                    "octaves": 8,
                    "persistence": 0.5,
                    "lacunarity": 2.0,
                    "seed": 42
                },
                "erosion": {
                    "hydraulic": {
                        "enabled": True,
                        "iterations": 100,
                        "rain_amount": 0.015,
                        "evaporation": 0.3,
                        "capacity": 0.018,
                        "deposition": 0.12,
                        "erosion": 0.35
                    },
                    "thermal": {
                        "enabled": True,
                        "iterations": 40,
                        "talus_angle": 0.65
                    }
                },
                "vegetation": {
                    "enabled": True,
                    "density": 0.45,
                    "min_spacing": 3.0,
                    "use_clustering": True
                }
            }
        }

    def _save_presets(self):
        """Sauvegarde les presets dans le fichier JSON"""
        with open(self.presets_file, 'w') as f:
            json.dump(self.presets, f, indent=2)

    def list_presets(self) -> List[str]:
        """Liste tous les presets disponibles"""
        return list(self.presets.keys())

    def get_preset(self, name: str) -> Dict:
        """
        Récupère un preset par son nom

        Args:
            name: Nom du preset

        Returns:
            Configuration complète du preset

        Raises:
            KeyError: Si le preset n'existe pas
        """
        if name not in self.presets:
            raise KeyError(f"Preset '{name}' not found. Available: {self.list_presets()}")

        return self.presets[name]

    def get_preset_info(self, name: str) -> Dict:
        """
        Récupère les informations d'un preset (nom, description)

        Args:
            name: Nom du preset

        Returns:
            Dict avec 'name' et 'description'
        """
        preset = self.get_preset(name)
        return {
            'name': preset.get('name', name),
            'description': preset.get('description', '')
        }

    def add_preset(self, name: str, config: Dict):
        """
        Ajoute ou met à jour un preset personnalisé

        Args:
            name: Nom du preset
            config: Configuration complète
        """
        self.presets[name] = config
        self._save_presets()

    def delete_preset(self, name: str):
        """
        Supprime un preset personnalisé

        Args:
            name: Nom du preset à supprimer
        """
        if name in self.presets:
            del self.presets[name]
            self._save_presets()

    def export_preset(self, name: str, filepath: str):
        """
        Exporte un preset vers un fichier JSON

        Args:
            name: Nom du preset
            filepath: Chemin du fichier d'export
        """
        preset = self.get_preset(name)
        with open(filepath, 'w') as f:
            json.dump(preset, f, indent=2)

    def import_preset(self, name: str, filepath: str):
        """
        Importe un preset depuis un fichier JSON

        Args:
            name: Nom à donner au preset importé
            filepath: Chemin du fichier à importer
        """
        with open(filepath, 'r') as f:
            config = json.load(f)

        self.add_preset(name, config)
