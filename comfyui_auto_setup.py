#!/usr/bin/env python3
"""
ComfyUI Auto-Setup pour Mountain Studio ULTIMATE v2.0
=====================================================

Installe automatiquement:
✅ Modèles SDXL requis (sd_xl_base_1.0.safetensors)
✅ Custom nodes manquants
✅ Workflows corrigés
✅ Dépendances Python

Résout les erreurs:
- "ImageSegmentation does not exist"
- "sd_xl_base_1.0.safetensors not in []"
- Seed négatif (-1)
"""

import os
import sys
import json
import urllib.request
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComfyUIAutoSetup:
    """Auto-installation complète de ComfyUI pour Mountain Studio"""

    # Modèles requis avec URLs
    REQUIRED_MODELS = {
        'sd_xl_base_1.0.safetensors': {
            'url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors',
            'category': 'checkpoints',
            'size_gb': 6.9,
            'description': 'SDXL Base 1.0 - required for AI texture generation'
        },
        'sd_xl_refiner_1.0.safetensors': {
            'url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors',
            'category': 'checkpoints',
            'size_gb': 6.1,
            'description': 'SDXL Refiner 1.0 - optional, for enhanced quality'
        },
        'sdxl_vae.safetensors': {
            'url': 'https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors',
            'category': 'vae',
            'size_gb': 0.3,
            'description': 'SDXL VAE - recommended for best quality'
        }
    }

    # Custom nodes requis
    REQUIRED_CUSTOM_NODES = {
        'ComfyUI-Manager': {
            'git_url': 'https://github.com/ltdrdata/ComfyUI-Manager.git',
            'description': 'ComfyUI Manager - for installing other nodes'
        },
        'comfyui_controlnet_aux': {
            'git_url': 'https://github.com/Fannovel16/comfyui_controlnet_aux.git',
            'description': 'ControlNet Auxiliary - includes segmentation'
        },
        'ComfyUI-Impact-Pack': {
            'git_url': 'https://github.com/ltdrdata/ComfyUI-Impact-Pack.git',
            'description': 'Impact Pack - advanced image processing'
        }
    }

    def __init__(self, comfyui_path: Optional[str] = None):
        """
        Args:
            comfyui_path: Path to ComfyUI installation. If None, will search.
        """
        if comfyui_path:
            self.comfyui_path = Path(comfyui_path)
        else:
            self.comfyui_path = self.find_comfyui_installation()

        if self.comfyui_path:
            self.models_path = self.comfyui_path / 'models'
            self.custom_nodes_path = self.comfyui_path / 'custom_nodes'
            logger.info(f"✅ ComfyUI found: {self.comfyui_path}")
        else:
            logger.warning("⚠️ ComfyUI not found. Please specify path.")

    def find_comfyui_installation(self) -> Optional[Path]:
        """Search for ComfyUI installation in common locations"""
        search_paths = [
            Path.home() / 'ComfyUI',
            Path.cwd() / 'ComfyUI',
            Path('/opt/ComfyUI'),
            Path('C:/ComfyUI') if os.name == 'nt' else None
        ]

        for path in search_paths:
            if path and path.exists() and (path / 'main.py').exists():
                return path

        return None

    def check_model_installed(self, model_name: str) -> bool:
        """Check if a model is already installed"""
        if not self.models_path:
            return False

        model_info = self.REQUIRED_MODELS[model_name]
        category = model_info['category']
        model_path = self.models_path / category / model_name

        return model_path.exists()

    def check_custom_node_installed(self, node_name: str) -> bool:
        """Check if a custom node is already installed"""
        if not self.custom_nodes_path:
            return False

        node_path = self.custom_nodes_path / node_name
        return node_path.exists()

    def download_model(self, model_name: str, progress_callback=None):
        """Download a model with progress tracking"""
        if self.check_model_installed(model_name):
            logger.info(f"✅ Model already installed: {model_name}")
            return True

        model_info = self.REQUIRED_MODELS[model_name]
        url = model_info['url']
        category = model_info['category']

        # Create category directory
        category_path = self.models_path / category
        category_path.mkdir(parents=True, exist_ok=True)

        output_path = category_path / model_name

        logger.info(f"📥 Downloading {model_name} ({model_info['size_gb']} GB)...")
        logger.info(f"   URL: {url}")
        logger.info(f"   Destination: {output_path}")
        logger.info("   This may take several minutes depending on your connection...")

        try:
            def reporthook(block_num, block_size, total_size):
                if progress_callback:
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    progress_callback(percent)
                else:
                    if total_size > 0:
                        downloaded = block_num * block_size
                        percent = min(100, (downloaded / total_size) * 100)
                        if block_num % 100 == 0:  # Update every 100 blocks
                            logger.info(f"   Progress: {percent:.1f}%")

            urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
            logger.info(f"✅ Downloaded: {model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return False

    def install_custom_node(self, node_name: str):
        """Install a custom node from git"""
        if self.check_custom_node_installed(node_name):
            logger.info(f"✅ Custom node already installed: {node_name}")
            return True

        node_info = self.REQUIRED_CUSTOM_NODES[node_name]
        git_url = node_info['git_url']

        logger.info(f"📦 Installing custom node: {node_name}")
        logger.info(f"   Git: {git_url}")

        try:
            # Clone repository
            node_path = self.custom_nodes_path / node_name
            result = subprocess.run(
                ['git', 'clone', git_url, str(node_path)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"❌ Git clone failed: {result.stderr}")
                return False

            # Install dependencies if requirements.txt exists
            requirements_path = node_path / 'requirements.txt'
            if requirements_path.exists():
                logger.info(f"   Installing dependencies for {node_name}...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    logger.warning(f"⚠️ Dependencies installation had issues: {result.stderr}")

            logger.info(f"✅ Installed: {node_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to install {node_name}: {e}")
            return False

    def create_fixed_workflow(self) -> Dict:
        """Create a fixed ComfyUI workflow with correct nodes and parameters"""
        workflow = {
            "4": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "3": {
                "inputs": {
                    "seed": 42,  # FIX: Use valid seed (not -1)
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "5": {
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": "ultra realistic mountain rock texture, 4k, PBR, professional photography, high detail",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "low quality, blurry, cartoon, anime, sketch",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "MountainStudio_Texture",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

        return workflow

    def save_workflow(self, workflow: Dict, output_path: str):
        """Save workflow to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        logger.info(f"💾 Saved workflow: {output_path}")

    def run_full_setup(self, install_models: bool = True, install_nodes: bool = True):
        """Run complete setup"""
        logger.info("=" * 80)
        logger.info("ComfyUI AUTO-SETUP for Mountain Studio ULTIMATE v2.0")
        logger.info("=" * 80)

        if not self.comfyui_path:
            logger.error("❌ ComfyUI not found. Please install ComfyUI first:")
            logger.error("   git clone https://github.com/comfyanonymous/ComfyUI.git")
            return False

        success = True

        # 1. Install custom nodes
        if install_nodes:
            logger.info("\n📦 Installing Custom Nodes...")
            for node_name in self.REQUIRED_CUSTOM_NODES.keys():
                if not self.install_custom_node(node_name):
                    success = False

        # 2. Download models
        if install_models:
            logger.info("\n📥 Downloading Models...")
            logger.info("⚠️ Warning: This will download ~7-13 GB of models!")

            # Only download base model (required)
            base_model = 'sd_xl_base_1.0.safetensors'
            if not self.download_model(base_model):
                success = False

            # VAE is smaller, try to download
            vae_model = 'sdxl_vae.safetensors'
            self.download_model(vae_model)  # Don't fail if this doesn't work

        # 3. Create fixed workflow
        logger.info("\n📄 Creating Fixed Workflow...")
        workflow = self.create_fixed_workflow()
        workflow_path = Path.cwd() / 'mountain_studio_workflow_fixed.json'
        self.save_workflow(workflow, str(workflow_path))

        logger.info("\n" + "=" * 80)
        if success:
            logger.info("✅ SETUP COMPLETE!")
            logger.info("\nNext steps:")
            logger.info("1. Restart ComfyUI server if it's running")
            logger.info("2. Start ComfyUI: python main.py")
            logger.info("3. Load workflow: mountain_studio_workflow_fixed.json")
            logger.info("4. Generate textures in Mountain Studio!")
        else:
            logger.info("⚠️ SETUP COMPLETED WITH WARNINGS")
            logger.info("Some components may not have installed correctly.")
            logger.info("Check the logs above for details.")
        logger.info("=" * 80)

        return success

    def check_installation_status(self) -> Dict[str, bool]:
        """Check what's already installed"""
        status = {}

        # Check models
        for model_name in self.REQUIRED_MODELS.keys():
            status[f"model_{model_name}"] = self.check_model_installed(model_name)

        # Check custom nodes
        for node_name in self.REQUIRED_CUSTOM_NODES.keys():
            status[f"node_{node_name}"] = self.check_custom_node_installed(node_name)

        return status


def main():
    """CLI interface for auto-setup"""
    import argparse

    parser = argparse.ArgumentParser(description='ComfyUI Auto-Setup for Mountain Studio')
    parser.add_argument('--comfyui-path', type=str, help='Path to ComfyUI installation')
    parser.add_argument('--skip-models', action='store_true', help='Skip model downloads')
    parser.add_argument('--skip-nodes', action='store_true', help='Skip custom node installation')
    parser.add_argument('--check-only', action='store_true', help='Only check installation status')

    args = parser.parse_args()

    setup = ComfyUIAutoSetup(args.comfyui_path)

    if args.check_only:
        logger.info("=" * 80)
        logger.info("INSTALLATION STATUS CHECK")
        logger.info("=" * 80)

        status = setup.check_installation_status()

        for item, installed in status.items():
            icon = "✅" if installed else "❌"
            logger.info(f"{icon} {item}")

        logger.info("=" * 80)
    else:
        setup.run_full_setup(
            install_models=not args.skip_models,
            install_nodes=not args.skip_nodes
        )


if __name__ == '__main__':
    main()
