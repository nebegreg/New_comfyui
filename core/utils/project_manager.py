"""
Project Manager - Save/Load Complete Sessions
==============================================

Saves and loads complete Mountain Studio projects with all data:
- Terrain heightmaps
- PBR textures
- HDRI images
- Vegetation placements
- All parameters and settings
- Camera positions
- Render settings

Format: .mtsp (Mountain Studio Project) - ZIP archive with JSON + binary data

Author: Mountain Studio Pro Team
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import zipfile
import io
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MountainStudioProject:
    """
    Complete project container.

    Contains all data and settings for a Mountain Studio session.
    Can be saved to .mtsp file and loaded later.
    """

    VERSION = "1.0.0"

    def __init__(self):
        """Initialize empty project"""
        self.metadata = {
            'version': self.VERSION,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'name': 'Untitled Project',
            'description': '',
            'author': ''
        }

        # Core data
        self.heightmap: Optional[np.ndarray] = None
        self.vegetation: list = []
        self.pbr_textures: Dict = {}
        self.hdri_image: Optional[np.ndarray] = None

        # Parameters
        self.terrain_params: Dict = {}
        self.erosion_params: Dict = {}
        self.vegetation_params: Dict = {}
        self.pbr_params: Dict = {}
        self.hdri_params: Dict = {}

        # Render settings
        self.camera_position: Dict = {
            'distance': 350.0,
            'elevation': 25.0,
            'azimuth': 45.0
        }

        self.lighting: Dict = {
            'sun_azimuth': 135.0,
            'sun_elevation': 45.0,
            'sun_intensity': 1.2,
            'ambient_strength': 0.4
        }

        self.atmosphere: Dict = {
            'fog_enabled': True,
            'fog_density': 0.015,
            'atmosphere_enabled': True
        }

        # Preset info
        self.preset_used: Optional[str] = None

        # Custom data
        self.custom_data: Dict = {}

    def set_metadata(self, name: str = None, description: str = None, author: str = None):
        """Set project metadata"""
        if name:
            self.metadata['name'] = name
        if description:
            self.metadata['description'] = description
        if author:
            self.metadata['author'] = author
        self.metadata['modified'] = datetime.now().isoformat()

    def set_terrain(self, heightmap: np.ndarray, params: Dict):
        """Set terrain data"""
        self.heightmap = heightmap
        self.terrain_params = params
        self.metadata['modified'] = datetime.now().isoformat()

    def set_vegetation(self, vegetation: list, params: Dict):
        """Set vegetation data"""
        self.vegetation = vegetation
        self.vegetation_params = params
        self.metadata['modified'] = datetime.now().isoformat()

    def set_pbr(self, textures: Dict, params: Dict):
        """Set PBR textures"""
        self.pbr_textures = textures
        self.pbr_params = params
        self.metadata['modified'] = datetime.now().isoformat()

    def set_hdri(self, hdri: np.ndarray, params: Dict):
        """Set HDRI"""
        self.hdri_image = hdri
        self.hdri_params = params
        self.metadata['modified'] = datetime.now().isoformat()

    def set_camera(self, distance: float, elevation: float, azimuth: float):
        """Set camera position"""
        self.camera_position = {
            'distance': distance,
            'elevation': elevation,
            'azimuth': azimuth
        }
        self.metadata['modified'] = datetime.now().isoformat()

    def set_lighting(self, sun_azimuth: float, sun_elevation: float,
                     sun_intensity: float, ambient_strength: float):
        """Set lighting parameters"""
        self.lighting = {
            'sun_azimuth': sun_azimuth,
            'sun_elevation': sun_elevation,
            'sun_intensity': sun_intensity,
            'ambient_strength': ambient_strength
        }
        self.metadata['modified'] = datetime.now().isoformat()

    def set_atmosphere(self, fog_enabled: bool, fog_density: float, atmosphere_enabled: bool):
        """Set atmosphere parameters"""
        self.atmosphere = {
            'fog_enabled': fog_enabled,
            'fog_density': fog_density,
            'atmosphere_enabled': atmosphere_enabled
        }
        self.metadata['modified'] = datetime.now().isoformat()

    def save(self, filepath: str):
        """
        Save project to .mtsp file.

        The .mtsp file is a ZIP archive containing:
        - project.json: Metadata and parameters
        - heightmap.npy: Terrain heightmap (if exists)
        - vegetation.pkl: Vegetation data (if exists)
        - pbr/: PBR texture maps (if exist)
        - hdri.npy: HDRI image (if exists)

        Args:
            filepath: Path to save file (will add .mtsp extension if missing)
        """
        filepath = Path(filepath)
        if filepath.suffix != '.mtsp':
            filepath = filepath.with_suffix('.mtsp')

        logger.info(f"Saving project: {filepath}")

        try:
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Save metadata and parameters as JSON
                project_data = {
                    'metadata': self.metadata,
                    'terrain_params': self.terrain_params,
                    'erosion_params': self.erosion_params,
                    'vegetation_params': self.vegetation_params,
                    'pbr_params': self.pbr_params,
                    'hdri_params': self.hdri_params,
                    'camera_position': self.camera_position,
                    'lighting': self.lighting,
                    'atmosphere': self.atmosphere,
                    'preset_used': self.preset_used,
                    'custom_data': self.custom_data,
                    'has_heightmap': self.heightmap is not None,
                    'has_vegetation': len(self.vegetation) > 0,
                    'has_pbr': len(self.pbr_textures) > 0,
                    'has_hdri': self.hdri_image is not None
                }

                zf.writestr('project.json', json.dumps(project_data, indent=2))

                # Save heightmap
                if self.heightmap is not None:
                    heightmap_bytes = io.BytesIO()
                    np.save(heightmap_bytes, self.heightmap)
                    zf.writestr('heightmap.npy', heightmap_bytes.getvalue())
                    logger.debug(f"Saved heightmap: {self.heightmap.shape}")

                # Save vegetation
                if self.vegetation:
                    vegetation_bytes = io.BytesIO()
                    pickle.dump(self.vegetation, vegetation_bytes)
                    zf.writestr('vegetation.pkl', vegetation_bytes.getvalue())
                    logger.debug(f"Saved vegetation: {len(self.vegetation)} instances")

                # Save PBR textures
                if self.pbr_textures:
                    for name, texture in self.pbr_textures.items():
                        if isinstance(texture, np.ndarray):
                            # Save as PNG
                            img_bytes = io.BytesIO()
                            if texture.dtype != np.uint8:
                                texture = (np.clip(texture, 0, 1) * 255).astype(np.uint8)

                            if len(texture.shape) == 2:
                                img = Image.fromarray(texture, mode='L')
                            else:
                                img = Image.fromarray(texture, mode='RGB')

                            img.save(img_bytes, format='PNG')
                            zf.writestr(f'pbr/{name}.png', img_bytes.getvalue())
                            logger.debug(f"Saved PBR texture: {name}")

                # Save HDRI
                if self.hdri_image is not None:
                    hdri_bytes = io.BytesIO()
                    np.save(hdri_bytes, self.hdri_image)
                    zf.writestr('hdri.npy', hdri_bytes.getvalue())
                    logger.debug(f"Saved HDRI: {self.hdri_image.shape}")

            logger.info(f"Project saved successfully: {filepath} ({filepath.stat().st_size / 1024 / 1024:.2f} MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to save project: {e}", exc_info=True)
            return False

    def load(self, filepath: str):
        """
        Load project from .mtsp file.

        Args:
            filepath: Path to .mtsp file

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Project file not found: {filepath}")
            return False

        logger.info(f"Loading project: {filepath}")

        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                # Load project.json
                project_json = zf.read('project.json').decode('utf-8')
                project_data = json.loads(project_json)

                # Restore metadata and parameters
                self.metadata = project_data['metadata']
                self.terrain_params = project_data.get('terrain_params', {})
                self.erosion_params = project_data.get('erosion_params', {})
                self.vegetation_params = project_data.get('vegetation_params', {})
                self.pbr_params = project_data.get('pbr_params', {})
                self.hdri_params = project_data.get('hdri_params', {})
                self.camera_position = project_data.get('camera_position', {})
                self.lighting = project_data.get('lighting', {})
                self.atmosphere = project_data.get('atmosphere', {})
                self.preset_used = project_data.get('preset_used')
                self.custom_data = project_data.get('custom_data', {})

                # Load heightmap
                if project_data.get('has_heightmap'):
                    heightmap_bytes = io.BytesIO(zf.read('heightmap.npy'))
                    self.heightmap = np.load(heightmap_bytes)
                    logger.debug(f"Loaded heightmap: {self.heightmap.shape}")

                # Load vegetation
                if project_data.get('has_vegetation'):
                    vegetation_bytes = io.BytesIO(zf.read('vegetation.pkl'))
                    self.vegetation = pickle.load(vegetation_bytes)
                    logger.debug(f"Loaded vegetation: {len(self.vegetation)} instances")

                # Load PBR textures
                if project_data.get('has_pbr'):
                    self.pbr_textures = {}
                    for item in zf.namelist():
                        if item.startswith('pbr/') and item.endswith('.png'):
                            name = Path(item).stem
                            img_bytes = io.BytesIO(zf.read(item))
                            img = Image.open(img_bytes)
                            self.pbr_textures[name] = np.array(img)
                            logger.debug(f"Loaded PBR texture: {name}")

                # Load HDRI
                if project_data.get('has_hdri'):
                    hdri_bytes = io.BytesIO(zf.read('hdri.npy'))
                    self.hdri_image = np.load(hdri_bytes)
                    logger.debug(f"Loaded HDRI: {self.hdri_image.shape}")

            logger.info(f"Project loaded successfully: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load project: {e}", exc_info=True)
            return False

    def get_summary(self) -> str:
        """Get project summary"""
        summary = f"""
Mountain Studio Project: {self.metadata['name']}
Version: {self.metadata['version']}
Created: {self.metadata['created']}
Modified: {self.metadata['modified']}

Data:
- Terrain: {'Yes' if self.heightmap is not None else 'No'}
- Vegetation: {len(self.vegetation)} instances
- PBR Textures: {len(self.pbr_textures)} maps
- HDRI: {'Yes' if self.hdri_image is not None else 'No'}

Preset Used: {self.preset_used or 'Custom'}

Description: {self.metadata.get('description', 'No description')}
"""
        return summary

    def __repr__(self):
        return (f"MountainStudioProject(name='{self.metadata['name']}', "
                f"terrain={'Yes' if self.heightmap is not None else 'No'}, "
                f"vegetation={len(self.vegetation)}, "
                f"pbr={len(self.pbr_textures)}, "
                f"hdri={'Yes' if self.hdri_image is not None else 'No'})")


class ProjectManager:
    """
    Manager for recent projects and project operations.

    Tracks recent projects, provides quick open, auto-save, etc.
    """

    def __init__(self, projects_dir: str = "projects"):
        """Initialize project manager"""
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(exist_ok=True)

        self.config_file = self.projects_dir / "projects.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load project manager configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {'recent_projects': [], 'auto_save_enabled': True}

    def _save_config(self):
        """Save configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def add_recent(self, filepath: str):
        """Add project to recent list"""
        filepath = str(Path(filepath).absolute())

        # Remove if already exists
        if filepath in self.config['recent_projects']:
            self.config['recent_projects'].remove(filepath)

        # Add to front
        self.config['recent_projects'].insert(0, filepath)

        # Keep only last 10
        self.config['recent_projects'] = self.config['recent_projects'][:10]

        self._save_config()

    def get_recent_projects(self) -> list:
        """Get list of recent projects"""
        # Filter out non-existent files
        recent = []
        for filepath in self.config['recent_projects']:
            if Path(filepath).exists():
                recent.append(filepath)

        # Update config if filtered
        if len(recent) != len(self.config['recent_projects']):
            self.config['recent_projects'] = recent
            self._save_config()

        return recent

    def create_project(self, name: str, description: str = "", author: str = "") -> MountainStudioProject:
        """Create new project"""
        project = MountainStudioProject()
        project.set_metadata(name=name, description=description, author=author)
        return project

    def save_project(self, project: MountainStudioProject, filepath: str) -> bool:
        """Save project and add to recent"""
        if project.save(filepath):
            self.add_recent(filepath)
            return True
        return False

    def load_project(self, filepath: str) -> Optional[MountainStudioProject]:
        """Load project and add to recent"""
        project = MountainStudioProject()
        if project.load(filepath):
            self.add_recent(filepath)
            return project
        return None

    def auto_save(self, project: MountainStudioProject, name: str = "autosave"):
        """Auto-save project"""
        if not self.config.get('auto_save_enabled', True):
            return

        autosave_path = self.projects_dir / f"{name}.mtsp"
        project.save(str(autosave_path))
        logger.info(f"Auto-saved: {autosave_path}")

    def list_projects(self) -> list:
        """List all projects in projects directory"""
        return [str(p) for p in self.projects_dir.glob("*.mtsp")]


# Global project manager
_project_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    """Get global project manager instance"""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
