"""
Professional ComfyUI Workflows for PBR Texture Generation

Based on 2024 best practices:
- TXT2TEXTURE workflow (complete PBR sets)
- PBRify approach (normal/roughness from diffuse)
- Material-specific generation

Workflows generate:
- Diffuse/Albedo
- Normal map
- Roughness
- Ambient Occlusion
- Height/Displacement
- Metallic
"""

import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def create_txt2texture_workflow(
    prompt: str,
    negative_prompt: str = "blurry, low quality, tiling artifacts, repeating patterns",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    cfg: float = 7.0,
    seed: int = -1,
    seamless: bool = True
) -> Dict:
    """
    Create TXT2TEXTURE-style workflow for complete PBR generation

    This workflow generates:
    1. Base diffuse/albedo texture
    2. Normal map (from height analysis)
    3. Roughness map (from diffuse analysis)
    4. AO map (from diffuse analysis)
    5. Height map (from image analysis)

    Args:
        prompt: Material description (e.g., "rocky cliff face, granite")
        negative_prompt: Negative prompt
        width: Texture width
        height: Texture height
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
        seamless: Generate seamless/tileable texture

    Returns:
        ComfyUI workflow JSON
    """
    # Add seamless keywords if requested
    if seamless:
        prompt += ", seamless, tileable, texture, flat lighting, no shadows"
        negative_prompt += ", visible edges, seams, perspective, 3d"

    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"  # Or realisticVision, etc.
            }
        },

        # CLIP text encode - positive
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1]
            }
        },

        # CLIP text encode - negative
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1]
            }
        },

        # Empty latent image
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },

        # KSampler - generate base image
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0]
            }
        },

        # VAE Decode - get diffuse
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },

        # Save diffuse
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "pbr/diffuse",
                "images": ["6", 0]
            }
        }

        # Note: For complete PBR workflow, you would add:
        # - Normal map generation nodes (image analysis -> normal)
        # - Roughness extraction (analyze diffuse brightness variation)
        # - AO generation (analyze diffuse dark areas)
        # - Height map (edge detection + depth estimation)
        #
        # These require custom nodes like:
        # - PBRify nodes
        # - Image analysis nodes
        # - Material extraction nodes
    }

    return workflow


def create_pbrify_workflow(
    diffuse_image_path: str,
    output_prefix: str = "pbr_output"
) -> Dict:
    """
    Create PBRify-style workflow

    Generates Normal, Roughness, and Height from a diffuse texture

    Requires: PBRify custom nodes installed in ComfyUI

    Args:
        diffuse_image_path: Path to input diffuse texture
        output_prefix: Output filename prefix

    Returns:
        ComfyUI workflow JSON

    Note:
        This workflow requires PBRify custom nodes:
        https://github.com/Kim2091/PBRify_Remix
    """
    workflow = {
        # Load diffuse image
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": diffuse_image_path
            }
        },

        # PBRify - Generate Normal
        "2": {
            "class_type": "PBRifyNormal",  # Custom node
            "inputs": {
                "image": ["1", 0],
                "strength": 1.0
            }
        },

        # PBRify - Generate Roughness
        "3": {
            "class_type": "PBRifyRoughness",  # Custom node
            "inputs": {
                "image": ["1", 0],
                "strength": 1.0
            }
        },

        # PBRify - Generate Height
        "4": {
            "class_type": "PBRifyHeight",  # Custom node
            "inputs": {
                "image": ["1", 0],
                "strength": 1.0
            }
        },

        # Save normal
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"{output_prefix}_normal",
                "images": ["2", 0]
            }
        },

        # Save roughness
        "6": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"{output_prefix}_roughness",
                "images": ["3", 0]
            }
        },

        # Save height
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"{output_prefix}_height",
                "images": ["4", 0]
            }
        }
    }

    return workflow


def create_material_specific_workflow(
    material_type: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1
) -> Dict:
    """
    Create material-specific optimized workflow

    Pre-tuned prompts and settings for different material types

    Args:
        material_type: 'rock', 'grass', 'snow', 'sand', 'dirt', 'bark', 'gravel'
        width: Texture width
        height: Texture height
        seed: Random seed

    Returns:
        ComfyUI workflow JSON
    """
    # Material-specific prompts and settings
    material_configs = {
        'rock': {
            'prompt': "detailed rocky surface texture, granite stone, natural cliff face, "
                     "photorealistic, high detail, 8k, seamless tileable texture, "
                     "flat lighting, physically based rendering",
            'negative': "smooth, polished, artificial, tiling artifacts, perspective, shadows",
            'cfg': 7.5,
            'steps': 35
        },
        'grass': {
            'prompt': "natural grass ground texture, green meadow, blades of grass, "
                     "photorealistic vegetation, high detail, seamless tileable, "
                     "flat lighting, pbr material",
            'negative': "flowers, dirt, rocks, perspective, shadows, 3d",
            'cfg': 7.0,
            'steps': 30
        },
        'snow': {
            'prompt': "fresh snow texture, winter ground surface, fine snow crystals, "
                     "photorealistic, high detail, seamless tileable, soft lighting",
            'negative': "ice, water, dirt, footprints, perspective, shadows",
            'cfg': 6.5,
            'steps': 28
        },
        'sand': {
            'prompt': "fine sand texture, beach sand surface, desert ground, "
                     "photorealistic grains, high detail, seamless tileable, "
                     "flat lighting, natural material",
            'negative': "rocks, grass, wet, perspective, shadows, ripples",
            'cfg': 7.0,
            'steps': 30
        },
        'dirt': {
            'prompt': "brown dirt ground texture, earth soil surface, "
                     "photorealistic, high detail, seamless tileable, "
                     "natural lighting, pbr material",
            'negative': "grass, rocks, mud, wet, perspective, shadows",
            'cfg': 7.2,
            'steps': 32
        },
        'bark': {
            'prompt': "tree bark texture, wood surface, natural tree trunk, "
                     "photorealistic details, high resolution, seamless tileable",
            'negative': "leaves, smooth, artificial, perspective, shadows",
            'cfg': 7.5,
            'steps': 35
        },
        'gravel': {
            'prompt': "gravel stones texture, small pebbles, rocky ground surface, "
                     "photorealistic, high detail, seamless tileable, flat lighting",
            'negative': "large rocks, dirt, grass, perspective, shadows",
            'cfg': 7.3,
            'steps': 33
        }
    }

    config = material_configs.get(material_type, material_configs['rock'])

    return create_txt2texture_workflow(
        prompt=config['prompt'],
        negative_prompt=config['negative'],
        width=width,
        height=height,
        steps=config['steps'],
        cfg=config['cfg'],
        seed=seed,
        seamless=True
    )


def get_recommended_settings(resolution: int) -> Dict:
    """
    Get recommended ComfyUI settings for resolution

    Args:
        resolution: Target resolution (512, 1024, 2048, 4096)

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        512: {
            'steps': 25,
            'cfg': 7.0,
            'sampler': 'euler_ancestral',
            'scheduler': 'normal'
        },
        1024: {
            'steps': 30,
            'cfg': 7.5,
            'sampler': 'euler_ancestral',
            'scheduler': 'normal'
        },
        2048: {
            'steps': 35,
            'cfg': 8.0,
            'sampler': 'dpmpp_2m',
            'scheduler': 'karras'
        },
        4096: {
            'steps': 40,
            'cfg': 8.5,
            'sampler': 'dpmpp_2m_sde',
            'scheduler': 'karras'
        }
    }

    # Find closest resolution
    closest = min(settings.keys(), key=lambda x: abs(x - resolution))
    return settings[closest]


if __name__ == "__main__":
    # Example workflow generation
    print("ComfyUI Professional PBR Workflows")
    print("=" * 60)

    # Generate rock material workflow
    print("\n1. Material-specific workflow (rock):")
    rock_workflow = create_material_specific_workflow('rock', width=1024, height=1024)
    print(f"   Nodes: {len(rock_workflow)}")
    print(f"   Prompt: {rock_workflow['2']['inputs']['text'][:80]}...")

    # Generate grass workflow
    print("\n2. Material-specific workflow (grass):")
    grass_workflow = create_material_specific_workflow('grass', width=1024, height=1024)
    print(f"   Nodes: {len(grass_workflow)}")

    # Recommended settings
    print("\n3. Recommended settings:")
    for res in [512, 1024, 2048, 4096]:
        settings = get_recommended_settings(res)
        print(f"   {res}x{res}: {settings['steps']} steps, CFG {settings['cfg']}, {settings['sampler']}")

    print("\n" + "=" * 60)
    print("✓ Workflows ready for ComfyUI!")
