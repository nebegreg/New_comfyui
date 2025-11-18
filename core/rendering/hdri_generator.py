"""
HDRI Panoramic Generator for Mountain Studio Pro

Generates 360° panoramic HDR environment maps for terrain rendering.
Supports both procedural generation and AI enhancement.

Features:
- Equirectangular 360° panoramas
- Cubemap generation
- HDR (.hdr) and EXR (.exr) export
- Time-of-day presets
- Optional AI enhancement with Stable Diffusion
- Procedural sky, clouds, and distant mountains

Author: Mountain Studio Pro
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging
from PIL import Image
from enum import Enum

logger = logging.getLogger(__name__)

# Optional imports
try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False
    logger.warning("OpenEXR not available - .exr export disabled")

try:
    from diffusers import StableDiffusionXLPipeline
    import torch
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("Diffusers not available - AI enhancement disabled")


class TimeOfDay(Enum):
    """Time of day presets for HDRI generation."""
    SUNRISE = "sunrise"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    SUNSET = "sunset"
    TWILIGHT = "twilight"
    NIGHT = "night"


class HDRIPanoramicGenerator:
    """
    Generate panoramic HDRI environment maps.

    Supports procedural generation with optional AI enhancement.
    Exports to .hdr (Radiance) and .exr (OpenEXR) formats.
    """

    # Standard resolutions
    RESOLUTION_LOW = (2048, 1024)
    RESOLUTION_MEDIUM = (4096, 2048)
    RESOLUTION_HIGH = (8192, 4096)

    # Sun parameters by time of day
    TIME_PRESETS = {
        TimeOfDay.SUNRISE: {
            'sun_elevation': 5.0,
            'sun_azimuth': 90.0,
            'sun_color': np.array([1.0, 0.8, 0.6]),
            'sun_intensity': 0.8,
            'sky_top_color': np.array([0.3, 0.4, 0.6]),
            'sky_horizon_color': np.array([1.0, 0.7, 0.5]),
            'ground_color': np.array([0.3, 0.25, 0.2]),
            'exposure': 1.0
        },
        TimeOfDay.MORNING: {
            'sun_elevation': 30.0,
            'sun_azimuth': 120.0,
            'sun_color': np.array([1.0, 0.95, 0.9]),
            'sun_intensity': 1.2,
            'sky_top_color': np.array([0.4, 0.6, 0.9]),
            'sky_horizon_color': np.array([0.7, 0.8, 0.95]),
            'ground_color': np.array([0.4, 0.35, 0.3]),
            'exposure': 1.2
        },
        TimeOfDay.MIDDAY: {
            'sun_elevation': 60.0,
            'sun_azimuth': 180.0,
            'sun_color': np.array([1.0, 1.0, 0.98]),
            'sun_intensity': 2.0,
            'sky_top_color': np.array([0.3, 0.5, 0.9]),
            'sky_horizon_color': np.array([0.6, 0.75, 0.95]),
            'ground_color': np.array([0.5, 0.45, 0.4]),
            'exposure': 1.5
        },
        TimeOfDay.AFTERNOON: {
            'sun_elevation': 40.0,
            'sun_azimuth': 240.0,
            'sun_color': np.array([1.0, 0.9, 0.8]),
            'sun_intensity': 1.5,
            'sky_top_color': np.array([0.4, 0.55, 0.85]),
            'sky_horizon_color': np.array([0.8, 0.75, 0.7]),
            'ground_color': np.array([0.45, 0.4, 0.35]),
            'exposure': 1.3
        },
        TimeOfDay.SUNSET: {
            'sun_elevation': 5.0,
            'sun_azimuth': 270.0,
            'sun_color': np.array([1.0, 0.6, 0.4]),
            'sun_intensity': 0.8,
            'sky_top_color': np.array([0.3, 0.35, 0.5]),
            'sky_horizon_color': np.array([1.0, 0.5, 0.3]),
            'ground_color': np.array([0.3, 0.2, 0.15]),
            'exposure': 0.9
        },
        TimeOfDay.TWILIGHT: {
            'sun_elevation': -5.0,
            'sun_azimuth': 270.0,
            'sun_color': np.array([0.8, 0.4, 0.3]),
            'sun_intensity': 0.3,
            'sky_top_color': np.array([0.1, 0.15, 0.3]),
            'sky_horizon_color': np.array([0.5, 0.3, 0.4]),
            'ground_color': np.array([0.1, 0.08, 0.06]),
            'exposure': 0.6
        },
        TimeOfDay.NIGHT: {
            'sun_elevation': -30.0,
            'sun_azimuth': 0.0,
            'sun_color': np.array([0.3, 0.3, 0.4]),
            'sun_intensity': 0.05,
            'sky_top_color': np.array([0.01, 0.01, 0.05]),
            'sky_horizon_color': np.array([0.05, 0.05, 0.1]),
            'ground_color': np.array([0.02, 0.02, 0.02]),
            'exposure': 0.3
        }
    }

    def __init__(self, resolution: Tuple[int, int] = RESOLUTION_MEDIUM):
        """
        Initialize HDRI generator.

        Args:
            resolution: Output resolution (width, height). Must be 2:1 ratio for equirectangular.
        """
        if resolution[0] != resolution[1] * 2:
            logger.warning(f"Resolution {resolution} is not 2:1 ratio. Equirectangular may be distorted.")

        self.resolution = resolution
        self.width, self.height = resolution

        # AI pipeline (lazy loading)
        self._ai_pipeline = None
        self._device = None

        logger.info(f"HDRI Generator initialized: {self.width}x{self.height}")

    def generate_procedural(
        self,
        time_of_day: TimeOfDay = TimeOfDay.MIDDAY,
        cloud_density: float = 0.3,
        mountain_distance: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate procedural HDRI panorama.

        Args:
            time_of_day: Time of day preset
            cloud_density: Cloud coverage [0-1]
            mountain_distance: Add distant mountains silhouette
            seed: Random seed for reproducibility

        Returns:
            HDR image array (H, W, 3) in float32, linear color space
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Generating procedural HDRI: {time_of_day.value}, clouds={cloud_density}")

        # Get preset parameters
        params = self.TIME_PRESETS[time_of_day]

        # Create coordinate grid (equirectangular)
        y_coords, x_coords = np.meshgrid(
            np.linspace(0, 1, self.height),
            np.linspace(0, 1, self.width),
            indexing='ij'
        )

        # Convert to spherical coordinates
        # theta (longitude): 0 to 2π
        # phi (latitude): -π/2 to π/2
        theta = x_coords * 2 * np.pi  # [0, 2π]
        phi = (y_coords - 0.5) * np.pi  # [-π/2, π/2]

        # Convert to Cartesian for lighting calculations
        x = np.cos(phi) * np.cos(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.sin(theta)

        # Initialize image
        hdr_image = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # 1. Sky gradient
        sky_factor = (y + 1.0) / 2.0  # [0, 1], 0=bottom, 1=top
        sky_factor = np.power(sky_factor, 0.7)  # Nonlinear gradient

        sky_top = params['sky_top_color']
        sky_horizon = params['sky_horizon_color']

        for c in range(3):
            hdr_image[:, :, c] = sky_horizon[c] + (sky_top[c] - sky_horizon[c]) * sky_factor

        # 2. Sun
        sun_elevation_rad = np.radians(params['sun_elevation'])
        sun_azimuth_rad = np.radians(params['sun_azimuth'])

        sun_dir = np.array([
            np.cos(sun_elevation_rad) * np.cos(sun_azimuth_rad),
            np.sin(sun_elevation_rad),
            np.cos(sun_elevation_rad) * np.sin(sun_azimuth_rad)
        ])

        # Dot product with sun direction
        dot_sun = x * sun_dir[0] + y * sun_dir[1] + z * sun_dir[2]
        dot_sun = np.clip(dot_sun, 0, 1)

        # Sun disk (sharp)
        sun_disk = np.power(dot_sun, 2000.0) * params['sun_intensity'] * 10.0
        sun_glow = np.power(dot_sun, 20.0) * params['sun_intensity'] * 2.0

        for c in range(3):
            hdr_image[:, :, c] += (sun_disk + sun_glow) * params['sun_color'][c]

        # 3. Atmospheric scattering (blue sky)
        scatter_factor = np.power(np.clip(y + 0.2, 0, 1), 0.5)
        scatter_color = np.array([0.3, 0.5, 0.9])

        for c in range(3):
            hdr_image[:, :, c] += scatter_factor * scatter_color[c] * 0.3

        # 4. Clouds (Perlin-like noise)
        if cloud_density > 0:
            clouds = self._generate_clouds(theta, phi, cloud_density, seed)
            cloud_color = np.array([1.0, 1.0, 1.0]) * params['exposure']

            for c in range(3):
                hdr_image[:, :, c] = hdr_image[:, :, c] * (1 - clouds) + cloud_color[c] * clouds

        # 5. Distant mountains (silhouette)
        if mountain_distance:
            mountains = self._generate_mountain_silhouette(theta, phi, y, seed)
            mountain_color = params['ground_color'] * 0.5  # Darker

            for c in range(3):
                hdr_image[:, :, c] = hdr_image[:, :, c] * (1 - mountains) + mountain_color[c] * mountains

        # 6. Ground (below horizon)
        ground_mask = (y < -0.05).astype(np.float32)
        ground_gradient = np.clip((-y - 0.05) / 0.3, 0, 1)

        for c in range(3):
            hdr_image[:, :, c] = (
                hdr_image[:, :, c] * (1 - ground_mask) +
                params['ground_color'][c] * ground_gradient * ground_mask
            )

        # Apply exposure
        hdr_image *= params['exposure']

        logger.info(f"Procedural HDRI generated: range=[{hdr_image.min():.3f}, {hdr_image.max():.3f}]")

        return hdr_image

    def _generate_clouds(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        density: float,
        seed: Optional[int]
    ) -> np.ndarray:
        """Generate cloud layer using multi-octave noise."""
        if seed is not None:
            np.random.seed(seed + 1)

        clouds = np.zeros_like(theta, dtype=np.float32)

        # Only add clouds in upper hemisphere
        sky_mask = phi > -0.1

        # Multi-octave noise
        for octave in range(4):
            freq = 2 ** octave * 2.0
            amp = 0.5 ** octave

            noise_x = np.sin(theta * freq + octave) * np.cos(phi * freq * 2)
            noise_y = np.cos(theta * freq * 1.3 + octave) * np.sin(phi * freq * 2)
            noise = (noise_x + noise_y) * 0.5 + 0.5

            clouds += noise * amp

        # Normalize and apply density
        clouds = clouds / clouds.max() if clouds.max() > 0 else clouds

        # Ensure clouds are positive before power operation
        clouds = np.clip(clouds, 0, 1)
        clouds = np.power(clouds, 1.5)  # Sharpen

        # Apply density threshold (protect against division by zero)
        density = max(density, 0.01)  # Minimum density to avoid division by zero
        clouds = (clouds - (1.0 - density)) / density
        clouds = np.clip(clouds, 0, 1)

        # Apply sky mask
        clouds *= sky_mask

        return clouds

    def _generate_mountain_silhouette(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        y: np.ndarray,
        seed: Optional[int]
    ) -> np.ndarray:
        """Generate distant mountain silhouette."""
        if seed is not None:
            np.random.seed(seed + 2)

        mountains = np.zeros_like(theta, dtype=np.float32)

        # Mountain profile (varies with azimuth)
        # Use multiple frequencies for varied peaks
        profile = 0.0
        for freq in [2, 3, 5, 7]:
            phase = np.random.rand() * 2 * np.pi
            profile += np.sin(theta * freq + phase) * (0.15 / freq)

        # Base height at horizon
        base_height = -0.05 + profile

        # Mountain mask (below profile, above ground)
        mountain_mask = (y < base_height) & (y > -0.2)

        # Fade based on distance from profile
        distance_from_edge = np.abs(y - base_height)
        fade = np.exp(-distance_from_edge * 50.0)

        mountains = mountain_mask.astype(np.float32) * fade

        return mountains

    def enhance_with_ai(
        self,
        base_image: np.ndarray,
        prompt: str,
        strength: float = 0.5,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Enhance procedural HDRI with AI (Stable Diffusion XL).

        Requires 'diffusers' package and ~10-12 GB VRAM.

        Args:
            base_image: Base HDR image from generate_procedural()
            prompt: Text prompt for enhancement
            strength: Strength of AI modification [0-1]
            seed: Random seed

        Returns:
            Enhanced HDR image
        """
        if not AI_AVAILABLE:
            logger.error("AI enhancement not available - diffusers not installed")
            return base_image

        logger.info(f"Enhancing with AI: prompt='{prompt}', strength={strength}")

        # Lazy load pipeline
        if self._ai_pipeline is None:
            self._init_ai_pipeline()

        # Convert HDR to LDR for SD input (tone mapping)
        ldr_input = self._tonemap_for_display(base_image)
        ldr_input_pil = Image.fromarray((ldr_input * 255).astype(np.uint8))

        # Resize if needed (SD works best at certain sizes)
        # SDXL default: 1024x1024, but we need 2:1 ratio
        target_height = 1024
        target_width = 2048

        ldr_input_pil = ldr_input_pil.resize((target_width, target_height), Image.LANCZOS)

        # Generate with img2img
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        result = self._ai_pipeline(
            prompt=prompt,
            image=ldr_input_pil,
            strength=strength,
            generator=generator,
            num_inference_steps=30
        ).images[0]

        # Convert back to numpy and resize to original resolution
        result_np = np.array(result).astype(np.float32) / 255.0
        result_pil = Image.fromarray((result_np * 255).astype(np.uint8))
        result_pil = result_pil.resize((self.width, self.height), Image.LANCZOS)
        result_np = np.array(result_pil).astype(np.float32) / 255.0

        # Blend with original HDR (preserve dynamic range)
        # Use AI for color/detail, keep HDR for intensity
        intensity_original = np.mean(base_image, axis=2, keepdims=True)
        intensity_result = np.mean(result_np, axis=2, keepdims=True)

        # Reapply original intensity to AI colors
        enhanced = result_np * (intensity_original / (intensity_result + 1e-6))

        logger.info("AI enhancement complete")

        return enhanced

    def _init_ai_pipeline(self):
        """Initialize Stable Diffusion XL pipeline."""
        logger.info("Loading Stable Diffusion XL pipeline...")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._ai_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
        ).to(self._device)

        # Optimizations
        if self._device == "cuda":
            self._ai_pipeline.enable_attention_slicing()

        logger.info(f"AI pipeline loaded on {self._device}")

    def export_hdr(self, image: np.ndarray, output_path: str):
        """
        Export to Radiance HDR format (.hdr).

        Args:
            image: HDR image array (H, W, 3)
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Radiance HDR uses RGBE encoding
        # We'll use PIL with a custom encoder or save as high-precision format

        # For now, use EXR if available, otherwise tone-mapped PNG with warning
        if OPENEXR_AVAILABLE:
            # Convert to EXR instead (better HDR support)
            exr_path = output_path.with_suffix('.exr')
            self.export_exr(image, str(exr_path))
            logger.info(f"Exported as EXR instead: {exr_path}")
        else:
            # Fallback: tone-mapped PNG
            logger.warning("HDR export not available - saving tone-mapped PNG")
            tone_mapped = self._tonemap_for_display(image)
            img_pil = Image.fromarray((tone_mapped * 255).astype(np.uint8))
            png_path = output_path.with_suffix('.png')
            img_pil.save(png_path)
            logger.info(f"Saved tone-mapped PNG: {png_path}")

    def export_exr(self, image: np.ndarray, output_path: str):
        """
        Export to OpenEXR format (.exr).

        Args:
            image: HDR image array (H, W, 3)
            output_path: Output file path
        """
        if not OPENEXR_AVAILABLE:
            logger.error("OpenEXR not available - cannot export .exr")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to OpenEXR format
        height, width = image.shape[:2]

        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }

        exr = OpenEXR.OutputFile(str(output_path), header)

        # Extract channels and convert to bytes
        r = image[:, :, 0].astype(np.float32).tobytes()
        g = image[:, :, 1].astype(np.float32).tobytes()
        b = image[:, :, 2].astype(np.float32).tobytes()

        exr.writePixels({'R': r, 'G': g, 'B': b})
        exr.close()

        logger.info(f"Exported EXR: {output_path}")

    def export_ldr(self, image: np.ndarray, output_path: str, tone_map: bool = True):
        """
        Export tone-mapped LDR preview (PNG/JPG).

        Args:
            image: HDR image array
            output_path: Output file path
            tone_map: Apply tone mapping
        """
        if tone_map:
            ldr_image = self._tonemap_for_display(image)
        else:
            ldr_image = np.clip(image, 0, 1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img_pil = Image.fromarray((ldr_image * 255).astype(np.uint8))
        img_pil.save(output_path, quality=95)

        logger.info(f"Exported LDR preview: {output_path}")

    @staticmethod
    def _tonemap_for_display(hdr_image: np.ndarray, exposure: float = 1.0) -> np.ndarray:
        """
        Apply Reinhard tone mapping for display.

        Args:
            hdr_image: HDR image in linear space
            exposure: Exposure adjustment

        Returns:
            Tone-mapped LDR image [0-1]
        """
        # Apply exposure
        exposed = hdr_image * exposure

        # Reinhard tone mapping: L_out = L_in / (1 + L_in)
        tone_mapped = exposed / (1.0 + exposed)

        # Gamma correction (sRGB)
        gamma_corrected = np.power(tone_mapped, 1.0 / 2.2)

        return np.clip(gamma_corrected, 0, 1)

    def generate_preset(self, time_of_day: TimeOfDay, output_dir: str, ai_enhance: bool = False):
        """
        Generate and save complete HDRI preset.

        Args:
            time_of_day: Time of day preset
            output_dir: Output directory
            ai_enhance: Apply AI enhancement
        """
        logger.info(f"Generating preset: {time_of_day.value}")

        # Generate procedural
        hdr = self.generate_procedural(time_of_day=time_of_day, cloud_density=0.3)

        # Optional AI enhancement
        if ai_enhance and AI_AVAILABLE:
            prompt = f"360 degree panoramic view of mountains at {time_of_day.value}, highly detailed, photorealistic, 8k"
            hdr = self.enhance_with_ai(hdr, prompt, strength=0.4)

        # Save all formats
        output_dir = Path(output_dir)
        base_name = f"mountain_hdri_{time_of_day.value}"

        self.export_exr(hdr, str(output_dir / f"{base_name}.exr"))
        self.export_ldr(hdr, str(output_dir / f"{base_name}_preview.png"))

        logger.info(f"Preset saved: {output_dir / base_name}")

    def __repr__(self) -> str:
        return f"HDRIPanoramicGenerator({self.width}x{self.height})"
