"""
AI Enhancement Module for Mountain Studio Pro

Integrates with ComfyUI for AI-powered texture generation and enhancement.
"""

from .comfyui_integration import (
    ComfyUIClient,
    generate_pbr_textures,
    enhance_terrain_texture,
    generate_landscape_image
)

__all__ = [
    'ComfyUIClient',
    'generate_pbr_textures',
    'enhance_terrain_texture',
    'generate_landscape_image'
]
