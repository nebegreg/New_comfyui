"""
ComfyUI Auto-Installer and Model Manager

Handles automatic installation of:
- ComfyUI checkpoints/models
- Custom nodes for PBR generation
- Workflows
- Dependencies

Designed for professional VFX pipeline integration.
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model to download"""
    name: str
    url: str
    filename: str
    category: str  # 'checkpoint', 'lora', 'controlnet', 'vae'
    size_mb: int
    sha256: Optional[str] = None
    description: str = ""
    required_for: List[str] = None  # Features that need this model


@dataclass
class CustomNodeInfo:
    """Information about a custom node"""
    name: str
    git_url: str
    description: str
    required_for: List[str] = None


class ComfyUIInstaller:
    """
    Manages ComfyUI models, custom nodes, and configuration

    Features:
    - Download models with progress bars
    - Install custom nodes from git
    - Verify checksums
    - Track installed components
    - UI integration for path selection
    """

    def __init__(self, comfyui_path: Optional[str] = None):
        """
        Args:
            comfyui_path: Path to ComfyUI installation. If None, will prompt user.
        """
        self.comfyui_path = Path(comfyui_path) if comfyui_path else None
        self.models_path = None
        self.custom_nodes_path = None

        # Track what's installed
        self.installed_file = Path.home() / '.mountain_studio' / 'comfyui_installed.json'
        self.installed_components = self._load_installed()

        if self.comfyui_path:
            self._setup_paths()

    def _setup_paths(self):
        """Setup paths to ComfyUI directories"""
        self.models_path = self.comfyui_path / 'models'
        self.custom_nodes_path = self.comfyui_path / 'custom_nodes'

        logger.info(f"ComfyUI paths configured:")
        logger.info(f"  Models: {self.models_path}")
        logger.info(f"  Custom nodes: {self.custom_nodes_path}")

    def set_comfyui_path(self, path: str) -> bool:
        """
        Set and validate ComfyUI path

        Returns:
            True if valid ComfyUI installation found
        """
        path = Path(path)

        # Check if it looks like a ComfyUI installation
        if not (path / 'models').exists():
            logger.error(f"Not a valid ComfyUI path: {path}")
            return False

        if not (path / 'custom_nodes').exists():
            logger.error(f"custom_nodes directory not found: {path}")
            return False

        self.comfyui_path = path
        self._setup_paths()
        return True

    def _load_installed(self) -> Dict:
        """Load record of installed components"""
        if self.installed_file.exists():
            with open(self.installed_file, 'r') as f:
                return json.load(f)
        return {'models': {}, 'nodes': {}, 'workflows': {}}

    def _save_installed(self):
        """Save record of installed components"""
        self.installed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.installed_file, 'w') as f:
            json.dump(self.installed_components, f, indent=2)

    def get_recommended_models(self) -> List[ModelInfo]:
        """
        Get list of recommended models for terrain/PBR generation

        Returns:
            List of ModelInfo objects
        """
        models = [
            # Main checkpoint for texture generation
            ModelInfo(
                name="Realistic Vision V5.1",
                url="https://civitai.com/api/download/models/130072",
                filename="realisticVisionV51_v51VAE.safetensors",
                category="checkpoint",
                size_mb=2132,
                description="Best checkpoint for realistic terrain textures",
                required_for=["pbr_textures", "landscape_generation"]
            ),

            # Alternative: SD XL for high quality
            ModelInfo(
                name="SD XL Base 1.0",
                url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                filename="sd_xl_base_1.0.safetensors",
                category="checkpoint",
                size_mb=6938,
                description="SDXL for highest quality textures (slower)",
                required_for=["high_quality_pbr"]
            ),

            # VAE for better colors
            ModelInfo(
                name="VAE-ft-mse-840000",
                url="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                filename="vae-ft-mse-840000-ema-pruned.safetensors",
                category="vae",
                size_mb=335,
                description="Improved VAE for better color accuracy",
                required_for=["pbr_textures"]
            ),

            # ControlNet for normal map guidance
            ModelInfo(
                name="ControlNet Normal",
                url="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/diffusion_pytorch_model.safetensors",
                filename="control_v11p_sd15_normalbae.safetensors",
                category="controlnet",
                size_mb=1445,
                description="ControlNet for normal map generation",
                required_for=["normal_map_generation"]
            ),

            # ControlNet for depth
            ModelInfo(
                name="ControlNet Depth",
                url="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.safetensors",
                filename="control_v11f1p_sd15_depth.safetensors",
                category="controlnet",
                size_mb=1445,
                description="ControlNet for depth-based generation",
                required_for=["depth_guided_generation"]
            ),
        ]

        return models

    def get_recommended_custom_nodes(self) -> List[CustomNodeInfo]:
        """
        Get list of recommended custom nodes for PBR workflow

        Returns:
            List of CustomNodeInfo objects
        """
        nodes = [
            CustomNodeInfo(
                name="ComfyUI-Manager",
                git_url="https://github.com/ltdrdata/ComfyUI-Manager.git",
                description="Essential for managing other custom nodes",
                required_for=["node_management"]
            ),

            CustomNodeInfo(
                name="ComfyUI_Comfyroll_CustomNodes",
                git_url="https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git",
                description="Useful utility nodes",
                required_for=["utility"]
            ),

            CustomNodeInfo(
                name="ComfyUI-Impact-Pack",
                git_url="https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
                description="Advanced image processing",
                required_for=["image_processing"]
            ),

            CustomNodeInfo(
                name="comfyui_controlnet_aux",
                git_url="https://github.com/Fannovel16/comfyui_controlnet_aux.git",
                description="Preprocessors for ControlNet",
                required_for=["normal_map_generation", "depth_generation"]
            ),

            # Note: PBRify nodes might not exist yet, this is aspirational
            CustomNodeInfo(
                name="ComfyUI-PBRify",
                git_url="https://github.com/example/ComfyUI-PBRify.git",  # Placeholder
                description="PBR texture generation (if available)",
                required_for=["pbr_workflow"]
            ),
        ]

        return nodes

    def is_model_installed(self, model: ModelInfo) -> bool:
        """Check if a model is already installed"""
        if not self.models_path:
            return False

        model_path = self.models_path / model.category / model.filename
        return model_path.exists()

    def is_node_installed(self, node: CustomNodeInfo) -> bool:
        """Check if a custom node is already installed"""
        if not self.custom_nodes_path:
            return False

        # Extract repo name from git URL
        repo_name = node.git_url.split('/')[-1].replace('.git', '')
        node_path = self.custom_nodes_path / repo_name

        return node_path.exists()

    def download_model(
        self,
        model: ModelInfo,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Download a model with progress tracking

        Args:
            model: ModelInfo object
            progress_callback: Function(current_mb, total_mb, percentage)

        Returns:
            True if successful
        """
        if not self.models_path:
            logger.error("ComfyUI path not set")
            return False

        # Create category directory
        category_path = self.models_path / model.category
        category_path.mkdir(parents=True, exist_ok=True)

        output_path = category_path / model.filename

        # Check if already exists
        if output_path.exists():
            logger.info(f"Model already exists: {model.filename}")
            return True

        logger.info(f"Downloading {model.name}...")
        logger.info(f"  URL: {model.url}")
        logger.info(f"  Size: {model.size_mb} MB")

        try:
            # Download with progress
            response = requests.get(model.url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            progress_callback(
                                downloaded / (1024 * 1024),
                                total_size / (1024 * 1024),
                                (downloaded / total_size * 100) if total_size > 0 else 0
                            )

            # Verify checksum if provided
            if model.sha256:
                if not self._verify_checksum(output_path, model.sha256):
                    logger.error(f"Checksum verification failed: {model.filename}")
                    output_path.unlink()
                    return False

            # Record installation
            self.installed_components['models'][model.name] = {
                'filename': model.filename,
                'category': model.category,
                'installed_at': str(output_path)
            }
            self._save_installed()

            logger.info(f"✓ Successfully downloaded: {model.name}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {model.name}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def install_custom_node(self, node: CustomNodeInfo) -> bool:
        """
        Install a custom node from git

        Args:
            node: CustomNodeInfo object

        Returns:
            True if successful
        """
        if not self.custom_nodes_path:
            logger.error("ComfyUI path not set")
            return False

        # Extract repo name
        repo_name = node.git_url.split('/')[-1].replace('.git', '')
        node_path = self.custom_nodes_path / repo_name

        if node_path.exists():
            logger.info(f"Custom node already exists: {node.name}")
            return True

        logger.info(f"Installing custom node: {node.name}")
        logger.info(f"  From: {node.git_url}")

        try:
            # Clone the repository
            import subprocess
            result = subprocess.run(
                ['git', 'clone', node.git_url, str(node_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return False

            # Check for requirements.txt
            requirements_file = node_path / 'requirements.txt'
            if requirements_file.exists():
                logger.info(f"Installing node dependencies...")
                result = subprocess.run(
                    ['pip', 'install', '-r', str(requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode != 0:
                    logger.warning(f"Some dependencies failed to install: {result.stderr}")

            # Record installation
            self.installed_components['nodes'][node.name] = {
                'git_url': node.git_url,
                'installed_at': str(node_path)
            }
            self._save_installed()

            logger.info(f"✓ Successfully installed: {node.name}")
            return True

        except Exception as e:
            logger.error(f"Error installing {node.name}: {e}")
            if node_path.exists():
                import shutil
                shutil.rmtree(node_path)
            return False

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file SHA256 checksum"""
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        actual = sha256.hexdigest()
        return actual.lower() == expected_sha256.lower()

    def get_installation_status(self) -> Dict:
        """
        Get status of recommended components

        Returns:
            Dict with installation status
        """
        status = {
            'models': [],
            'nodes': [],
            'comfyui_path_set': self.comfyui_path is not None
        }

        for model in self.get_recommended_models():
            status['models'].append({
                'name': model.name,
                'installed': self.is_model_installed(model),
                'size_mb': model.size_mb,
                'description': model.description,
                'required_for': model.required_for or []
            })

        for node in self.get_recommended_custom_nodes():
            status['nodes'].append({
                'name': node.name,
                'installed': self.is_node_installed(node),
                'description': node.description,
                'required_for': node.required_for or []
            })

        return status

    def install_all_required(self, features: List[str]) -> Tuple[int, int]:
        """
        Install all models and nodes required for specific features

        Args:
            features: List of feature names (e.g., ['pbr_textures', 'normal_map_generation'])

        Returns:
            (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        # Install required models
        for model in self.get_recommended_models():
            if not model.required_for:
                continue

            if any(feature in model.required_for for feature in features):
                if not self.is_model_installed(model):
                    if self.download_model(model):
                        successful += 1
                    else:
                        failed += 1

        # Install required nodes
        for node in self.get_recommended_custom_nodes():
            if not node.required_for:
                continue

            if any(feature in node.required_for for feature in features):
                if not self.is_node_installed(node):
                    if self.install_custom_node(node):
                        successful += 1
                    else:
                        failed += 1

        return successful, failed


# Standalone functions for CLI usage
def download_model_cli(url: str, output_path: str, expected_size_mb: int = 0):
    """
    Download a model from command line

    Args:
        url: Download URL
        output_path: Where to save
        expected_size_mb: Expected size for progress bar
    """
    print(f"Downloading to: {output_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✓ Download complete: {output_path}")


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create installer
    installer = ComfyUIInstaller()

    # Set ComfyUI path (adjust to your installation)
    comfyui_path = input("Enter ComfyUI path: ").strip()
    if installer.set_comfyui_path(comfyui_path):
        print("\n✓ ComfyUI path validated")

        # Show status
        status = installer.get_installation_status()
        print("\n" + "="*60)
        print("INSTALLATION STATUS")
        print("="*60)

        print("\nModels:")
        for model in status['models']:
            status_icon = "✓" if model['installed'] else "✗"
            print(f"  {status_icon} {model['name']} ({model['size_mb']} MB)")
            print(f"     {model['description']}")

        print("\nCustom Nodes:")
        for node in status['nodes']:
            status_icon = "✓" if node['installed'] else "✗"
            print(f"  {status_icon} {node['name']}")
            print(f"     {node['description']}")

        # Offer to install missing
        print("\n" + "="*60)
        response = input("\nInstall all required for PBR textures? (y/n): ")
        if response.lower() == 'y':
            successful, failed = installer.install_all_required(['pbr_textures'])
            print(f"\n✓ Installed: {successful}")
            if failed > 0:
                print(f"✗ Failed: {failed}")
    else:
        print("✗ Invalid ComfyUI path")
