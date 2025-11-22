"""
Post-Processing Effects System
===============================

Comprehensive post-processing pipeline for cinematic quality rendering:

Effects:
- Bloom (glow on highlights)
- Depth of Field (focus blur)
- SSAO (Screen Space Ambient Occlusion)
- Color Grading (LUTs)
- Tone Mapping (ACES, Reinhard, Filmic)
- Vignette
- Chromatic Aberration
- Film Grain

Author: Mountain Studio Pro Team
"""

import numpy as np
import logging
from typing import Optional, Tuple
from enum import Enum
from scipy.ndimage import gaussian_filter, convolve

logger = logging.getLogger(__name__)

# OpenGL imports
try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    logger.warning("OpenGL not available - GPU post-processing disabled")


class ToneMappingOperator(Enum):
    """Tone mapping operators"""
    ACES = "aces"             # Cinematic (Unreal Engine default)
    REINHARD = "reinhard"     # Classic
    FILMIC = "filmic"         # Blender-style
    UNCHARTED2 = "uncharted2" # Game-style
    NONE = "none"             # Linear (no tone mapping)


class PostProcessingPipeline:
    """
    Complete post-processing pipeline.

    Can be used with NumPy arrays (CPU) or OpenGL textures (GPU).
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize post-processing pipeline.

        Args:
            use_gpu: Use GPU acceleration if available
        """
        self.use_gpu = use_gpu and OPENGL_AVAILABLE

        # Effect parameters
        self.bloom_enabled = True
        self.bloom_threshold = 0.8
        self.bloom_intensity = 0.3
        self.bloom_radius = 10.0

        self.dof_enabled = False
        self.dof_focus_distance = 100.0
        self.dof_aperture = 2.0
        self.dof_focal_length = 50.0

        self.ssao_enabled = True
        self.ssao_radius = 0.5
        self.ssao_bias = 0.025
        self.ssao_samples = 16

        self.color_grading_enabled = True
        self.tone_mapping = ToneMappingOperator.ACES
        self.exposure = 1.0
        self.contrast = 1.0
        self.saturation = 1.0

        self.vignette_enabled = True
        self.vignette_intensity = 0.3
        self.vignette_smoothness = 0.5

        self.chromatic_aberration_enabled = False
        self.chromatic_aberration_strength = 0.5

        self.film_grain_enabled = True
        self.film_grain_intensity = 0.05

        logger.info(f"PostProcessingPipeline initialized (GPU={self.use_gpu})")

    # ==================== BLOOM ====================

    def apply_bloom(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bloom effect (glow on bright areas).

        Args:
            image: Input HDR image (H, W, 3) float32 [0-inf]

        Returns:
            Image with bloom
        """
        # Extract bright areas
        bright_mask = np.max(image, axis=2) > self.bloom_threshold
        bright_pixels = image * bright_mask[:, :, np.newaxis]

        # Gaussian blur for glow
        bloomed = np.zeros_like(bright_pixels)
        for c in range(3):
            bloomed[:, :, c] = gaussian_filter(bright_pixels[:, :, c],
                                               sigma=self.bloom_radius)

        # Add bloom to original
        result = image + bloomed * self.bloom_intensity

        return result

    # ==================== DEPTH OF FIELD ====================

    def apply_dof(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply depth of field (focus blur).

        Args:
            image: Input image (H, W, 3)
            depth_map: Depth map (H, W) - distance from camera

        Returns:
            Image with depth of field
        """
        # Calculate circle of confusion (blur amount) for each pixel
        coc = self._calculate_coc(depth_map)

        # Apply variable blur based on CoC
        result = np.zeros_like(image)

        # Sample blur at different levels
        blur_levels = [0, 2, 4, 8, 16]

        for c in range(3):
            # Create multi-scale blurred versions
            blurred_versions = []
            for sigma in blur_levels:
                if sigma == 0:
                    blurred_versions.append(image[:, :, c])
                else:
                    blurred_versions.append(gaussian_filter(image[:, :, c], sigma=sigma))

            # Blend based on CoC
            # Normalize CoC to [0, 1]
            coc_norm = np.clip(coc / np.max(blur_levels), 0, 1)

            # Linear interpolation between blur levels
            for i in range(len(blur_levels) - 1):
                level_min = blur_levels[i] / np.max(blur_levels)
                level_max = blur_levels[i + 1] / np.max(blur_levels)

                mask = (coc_norm >= level_min) & (coc_norm < level_max)

                if np.any(mask):
                    t = (coc_norm - level_min) / (level_max - level_min)
                    t = np.clip(t, 0, 1)

                    blended = (1 - t) * blurred_versions[i] + t * blurred_versions[i + 1]
                    result[:, :, c] = np.where(mask, blended, result[:, :, c])

            # Handle max level
            mask = coc_norm >= (blur_levels[-1] / np.max(blur_levels))
            result[:, :, c] = np.where(mask, blurred_versions[-1], result[:, :, c])

        return result

    def _calculate_coc(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Calculate circle of confusion from depth.

        Args:
            depth_map: Depth map (distance from camera)

        Returns:
            Circle of confusion map (blur radius in pixels)
        """
        # Thin lens equation
        # CoC = (aperture * focal_length * |depth - focus_distance|) / (depth * (focus_distance - focal_length))

        # Avoid division by zero
        depth_safe = np.maximum(depth_map, 0.1)

        denominator = depth_safe * (self.dof_focus_distance - self.dof_focal_length)
        denominator = np.maximum(np.abs(denominator), 0.1)

        coc = (self.dof_aperture * self.dof_focal_length *
               np.abs(depth_safe - self.dof_focus_distance)) / denominator

        # Clamp to reasonable values
        coc = np.clip(coc, 0, 20)

        return coc

    # ==================== SSAO ====================

    def apply_ssao(self, image: np.ndarray, depth_map: np.ndarray,
                   normal_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Screen Space Ambient Occlusion.

        Args:
            image: Input image (H, W, 3)
            depth_map: Depth map (H, W)
            normal_map: Normal map (H, W, 3) - optional

        Returns:
            Image with SSAO
        """
        h, w = depth_map.shape

        # Generate sample kernel
        kernel = self._generate_ssao_kernel()

        # Generate noise texture (4x4 tiled)
        noise = self._generate_ssao_noise()

        # Calculate occlusion factor for each pixel
        occlusion = np.ones((h, w), dtype=np.float32)

        # Simplified SSAO (in real impl, would use normals and proper sampling)
        # Here we approximate: darker in concave areas (higher depth variance)

        # Calculate local depth variance as proxy for occlusion
        from scipy.ndimage import generic_filter

        def depth_variance(values):
            return np.var(values)

        # Variance in local neighborhood indicates geometry complexity
        variance = generic_filter(depth_map, depth_variance, size=5)

        # Higher variance = more occlusion
        occlusion = 1.0 - np.clip(variance * self.ssao_radius * 10.0, 0, 0.7)

        # Apply bilateral blur to smooth occlusion
        occlusion = gaussian_filter(occlusion, sigma=2.0)

        # Apply to image
        result = image * occlusion[:, :, np.newaxis]

        return result

    def _generate_ssao_kernel(self) -> np.ndarray:
        """Generate random sample kernel for SSAO"""
        kernel = []
        for i in range(self.ssao_samples):
            # Random samples in hemisphere
            sample = np.random.rand(3) * 2.0 - 1.0
            sample[2] = abs(sample[2])  # Hemisphere (positive z)
            sample = sample / np.linalg.norm(sample)

            # Scale samples (more near center)
            scale = i / self.ssao_samples
            scale = 0.1 + scale * scale * 0.9

            sample *= scale

            kernel.append(sample)

        return np.array(kernel)

    def _generate_ssao_noise(self, size: int = 4) -> np.ndarray:
        """Generate noise texture for SSAO"""
        noise = np.random.rand(size, size, 3) * 2.0 - 1.0
        noise[:, :, 2] = 0  # Only rotate in xy plane
        return noise

    # ==================== TONE MAPPING ====================

    def apply_tone_mapping(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        Apply tone mapping to convert HDR to LDR.

        Args:
            hdr_image: HDR image [0-inf]

        Returns:
            LDR image [0-1]
        """
        # Apply exposure
        exposed = hdr_image * self.exposure

        # Apply tone mapping operator
        if self.tone_mapping == ToneMappingOperator.ACES:
            return self._tone_map_aces(exposed)
        elif self.tone_mapping == ToneMappingOperator.REINHARD:
            return self._tone_map_reinhard(exposed)
        elif self.tone_mapping == ToneMappingOperator.FILMIC:
            return self._tone_map_filmic(exposed)
        elif self.tone_mapping == ToneMappingOperator.UNCHARTED2:
            return self._tone_map_uncharted2(exposed)
        else:  # NONE
            return np.clip(exposed, 0, 1)

    def _tone_map_aces(self, hdr: np.ndarray) -> np.ndarray:
        """ACES Filmic tone mapping (Unreal Engine)"""
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        tone_mapped = (hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e)
        return np.clip(tone_mapped, 0, 1)

    def _tone_map_reinhard(self, hdr: np.ndarray) -> np.ndarray:
        """Reinhard tone mapping"""
        return hdr / (1.0 + hdr)

    def _tone_map_filmic(self, hdr: np.ndarray) -> np.ndarray:
        """Filmic tone mapping (Blender-style)"""
        x = np.maximum(0, hdr - 0.004)
        tone_mapped = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        return tone_mapped

    def _tone_map_uncharted2(self, hdr: np.ndarray) -> np.ndarray:
        """Uncharted 2 tone mapping"""
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30

        def uncharted2_tonemap_partial(x):
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

        curr = uncharted2_tonemap_partial(hdr * 2.0)
        white_scale = 1.0 / uncharted2_tonemap_partial(11.2)

        return np.clip(curr * white_scale, 0, 1)

    # ==================== COLOR GRADING ====================

    def apply_color_grading(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color grading (contrast, saturation, etc.).

        Args:
            image: Input image [0-1]

        Returns:
            Color graded image
        """
        result = image.copy()

        # Contrast
        if self.contrast != 1.0:
            result = (result - 0.5) * self.contrast + 0.5
            result = np.clip(result, 0, 1)

        # Saturation
        if self.saturation != 1.0:
            # Convert to grayscale
            gray = np.dot(result, [0.299, 0.587, 0.114])
            gray = gray[:, :, np.newaxis]

            # Lerp between grayscale and original
            result = gray + (result - gray) * self.saturation
            result = np.clip(result, 0, 1)

        return result

    # ==================== VIGNETTE ====================

    def apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """
        Apply vignette effect (darkening at edges).

        Args:
            image: Input image

        Returns:
            Image with vignette
        """
        h, w = image.shape[:2]

        # Create vignette mask
        y, x = np.ogrid[:h, :w]

        # Center coordinates
        center_y, center_x = h / 2, w / 2

        # Distance from center (normalized)
        dist_y = (y - center_y) / (h / 2)
        dist_x = (x - center_x) / (w / 2)
        distance = np.sqrt(dist_y ** 2 + dist_x ** 2)

        # Vignette function (smooth falloff)
        vignette = 1.0 - self.vignette_intensity * np.power(distance, 1.0 / self.vignette_smoothness)
        vignette = np.clip(vignette, 0, 1)

        # Apply
        result = image * vignette[:, :, np.newaxis]

        return result

    # ==================== CHROMATIC ABERRATION ====================

    def apply_chromatic_aberration(self, image: np.ndarray) -> np.ndarray:
        """
        Apply chromatic aberration (color fringing at edges).

        Args:
            image: Input RGB image

        Returns:
            Image with chromatic aberration
        """
        h, w = image.shape[:2]

        # Shift amount (in pixels)
        shift = int(self.chromatic_aberration_strength)

        result = np.zeros_like(image)

        # Red channel: shift outward
        result[:, :, 0] = self._shift_channel(image[:, :, 0], shift)

        # Green channel: no shift
        result[:, :, 1] = image[:, :, 1]

        # Blue channel: shift inward
        result[:, :, 2] = self._shift_channel(image[:, :, 2], -shift)

        return result

    def _shift_channel(self, channel: np.ndarray, shift: int) -> np.ndarray:
        """Radial shift of a channel"""
        h, w = channel.shape

        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Radial direction
        dy = y - center_y
        dx = x - center_x
        distance = np.sqrt(dy ** 2 + dx ** 2)

        # Avoid division by zero
        distance = np.maximum(distance, 1)

        # Shift coordinates
        new_y = (y + dy / distance * shift).astype(int)
        new_x = (x + dx / distance * shift).astype(int)

        # Clamp to valid range
        new_y = np.clip(new_y, 0, h - 1)
        new_x = np.clip(new_x, 0, w - 1)

        return channel[new_y, new_x]

    # ==================== FILM GRAIN ====================

    def apply_film_grain(self, image: np.ndarray) -> np.ndarray:
        """
        Apply film grain noise.

        Args:
            image: Input image

        Returns:
            Image with film grain
        """
        h, w = image.shape[:2]

        # Generate noise
        noise = np.random.randn(h, w) * self.film_grain_intensity

        # Add to each channel
        result = image + noise[:, :, np.newaxis]
        result = np.clip(result, 0, 1)

        return result

    # ==================== FULL PIPELINE ====================

    def process(self, hdr_image: np.ndarray,
                depth_map: Optional[np.ndarray] = None,
                normal_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply full post-processing pipeline.

        Args:
            hdr_image: Input HDR image [0-inf]
            depth_map: Optional depth map for DOF and SSAO
            normal_map: Optional normal map for SSAO

        Returns:
            Final LDR image [0-1]
        """
        result = hdr_image.copy()

        # 1. Bloom (on HDR)
        if self.bloom_enabled:
            result = self.apply_bloom(result)
            logger.debug("Applied bloom")

        # 2. Tone mapping (HDR → LDR)
        result = self.apply_tone_mapping(result)
        logger.debug(f"Applied tone mapping: {self.tone_mapping.value}")

        # 3. SSAO (on LDR)
        if self.ssao_enabled and depth_map is not None:
            result = self.apply_ssao(result, depth_map, normal_map)
            logger.debug("Applied SSAO")

        # 4. Depth of Field
        if self.dof_enabled and depth_map is not None:
            result = self.apply_dof(result, depth_map)
            logger.debug("Applied depth of field")

        # 5. Color grading
        if self.color_grading_enabled:
            result = self.apply_color_grading(result)
            logger.debug("Applied color grading")

        # 6. Vignette
        if self.vignette_enabled:
            result = self.apply_vignette(result)
            logger.debug("Applied vignette")

        # 7. Chromatic aberration
        if self.chromatic_aberration_enabled:
            result = self.apply_chromatic_aberration(result)
            logger.debug("Applied chromatic aberration")

        # 8. Film grain
        if self.film_grain_enabled:
            result = self.apply_film_grain(result)
            logger.debug("Applied film grain")

        # Gamma correction (sRGB)
        result = np.power(result, 1.0 / 2.2)

        return np.clip(result, 0, 1)

    def __repr__(self):
        effects = []
        if self.bloom_enabled:
            effects.append("Bloom")
        if self.ssao_enabled:
            effects.append("SSAO")
        if self.dof_enabled:
            effects.append("DOF")
        if self.vignette_enabled:
            effects.append("Vignette")

        return f"PostProcessingPipeline({', '.join(effects)}, {self.tone_mapping.value})"
