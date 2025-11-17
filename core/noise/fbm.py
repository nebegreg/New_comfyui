"""
Fractional Brownian Motion (fBm) and Variants

Advanced multi-octave noise functions for ultra-realistic terrain generation.

Based on research by:
- Musgrave et al. (1989) "The Synthesis and Rendering of Eroded Fractal Terrains"
- Ebert et al. (2003) "Texturing & Modeling: A Procedural Approach"
"""

import numpy as np
from typing import Callable, Optional
from .vectorized_noise import perlin_noise_2d, simplex_noise_2d


def fractional_brownian_motion(
    width: int,
    height: int,
    octaves: int = 8,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    seed: int = 0,
    noise_func: str = 'simplex'
) -> np.ndarray:
    """
    Fractional Brownian Motion - Standard multi-octave noise

    Classic fBm sums multiple octaves of noise with decreasing amplitude
    and increasing frequency.

    Args:
        width: Output width
        height: Output height
        octaves: Number of noise octaves (8-16 for ultra-realistic)
        lacunarity: Frequency multiplier per octave (2.0 standard)
        persistence: Amplitude multiplier per octave (0.5 standard)
        frequency: Base frequency
        amplitude: Base amplitude
        seed: Random seed
        noise_func: 'perlin' or 'simplex' (simplex recommended)

    Returns:
        2D array of terrain heights in range [0, 1]

    Notes:
        Higher octaves = more detail
        lacunarity=2.0 is standard, 2.5-3.0 for sharper features
        persistence=0.5 gives natural 1/f falloff
    """
    result = np.zeros((height, width), dtype=np.float32)

    # Choose noise function
    if noise_func == 'perlin':
        noise_fn = perlin_noise_2d
    elif noise_func == 'simplex':
        noise_fn = simplex_noise_2d
    else:
        raise ValueError(f"Unknown noise function: {noise_func}")

    # Accumulate octaves
    max_value = 0.0
    current_amplitude = amplitude
    current_frequency = frequency

    for octave in range(octaves):
        # Generate noise at current frequency
        noise = noise_fn(
            width, height,
            frequency=current_frequency,
            seed=seed + octave
        )

        # Add to result with current amplitude
        result += noise * current_amplitude

        # Track max for normalization
        max_value += current_amplitude

        # Update for next octave
        current_amplitude *= persistence
        current_frequency *= lacunarity

    # Normalize to [0, 1]
    result = (result + max_value) / (2.0 * max_value)

    return np.clip(result, 0.0, 1.0)


def turbulence(
    width: int,
    height: int,
    octaves: int = 8,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    seed: int = 0,
    noise_func: str = 'simplex'
) -> np.ndarray:
    """
    Turbulence - Absolute value of fBm

    Creates more chaotic, cloud-like patterns by taking absolute values.
    Useful for erosion patterns, clouds, and weathering.

    Args:
        Same as fractional_brownian_motion

    Returns:
        2D array in range [0, 1]

    Use cases:
        - Erosion intensity maps
        - Cloud patterns
        - Surface weathering
        - Micro-detail overlays
    """
    result = np.zeros((height, width), dtype=np.float32)

    if noise_func == 'perlin':
        noise_fn = perlin_noise_2d
    elif noise_func == 'simplex':
        noise_fn = simplex_noise_2d
    else:
        raise ValueError(f"Unknown noise function: {noise_func}")

    max_value = 0.0
    current_amplitude = amplitude
    current_frequency = frequency

    for octave in range(octaves):
        noise = noise_fn(
            width, height,
            frequency=current_frequency,
            seed=seed + octave
        )

        # Take absolute value for turbulence effect
        result += np.abs(noise) * current_amplitude

        max_value += current_amplitude
        current_amplitude *= persistence
        current_frequency *= lacunarity

    # Normalize to [0, 1]
    if max_value > 0:
        result = result / max_value

    return np.clip(result, 0.0, 1.0)


def billow(
    width: int,
    height: int,
    octaves: int = 8,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    seed: int = 0,
    noise_func: str = 'simplex'
) -> np.ndarray:
    """
    Billow noise - Inverted turbulence for puffy/cloud-like shapes

    Creates billowy, puffy patterns. Good for clouds and rolling hills.

    Args:
        Same as fractional_brownian_motion

    Returns:
        2D array in range [0, 1]

    Use cases:
        - Rolling hills
        - Cumulus clouds
        - Sand dunes
        - Gentle terrain
    """
    result = np.zeros((height, width), dtype=np.float32)

    if noise_func == 'perlin':
        noise_fn = perlin_noise_2d
    elif noise_func == 'simplex':
        noise_fn = simplex_noise_2d
    else:
        raise ValueError(f"Unknown noise function: {noise_func}")

    max_value = 0.0
    current_amplitude = amplitude
    current_frequency = frequency

    for octave in range(octaves):
        noise = noise_fn(
            width, height,
            frequency=current_frequency,
            seed=seed + octave
        )

        # Transform to billow: abs() * 2 - 1
        billowed = np.abs(noise) * 2.0 - 1.0
        result += billowed * current_amplitude

        max_value += current_amplitude
        current_amplitude *= persistence
        current_frequency *= lacunarity

    # Normalize to [0, 1]
    result = (result + max_value) / (2.0 * max_value)

    return np.clip(result, 0.0, 1.0)


def multiscale_noise(
    width: int,
    height: int,
    scales: list = [1.0, 4.0, 16.0, 64.0],
    weights: Optional[list] = None,
    seed: int = 0,
    noise_func: str = 'simplex'
) -> np.ndarray:
    """
    Multi-scale noise with custom frequency scales

    More flexible than standard fBm - you specify exact scales.
    Useful when you need specific feature sizes.

    Args:
        width: Output width
        height: Output height
        scales: List of frequency scales (e.g., [1, 4, 16])
        weights: Amplitude weights for each scale (optional)
        seed: Random seed
        noise_func: 'perlin' or 'simplex'

    Returns:
        2D array in range [0, 1]

    Example:
        # Mountain with specific feature sizes
        noise = multiscale_noise(
            1024, 1024,
            scales=[0.5, 2.0, 8.0, 32.0, 128.0],  # Custom sizes
            weights=[1.0, 0.8, 0.6, 0.4, 0.2]      # Custom importance
        )
    """
    if weights is None:
        # Default: equal weight per scale
        weights = [1.0] * len(scales)

    if len(weights) != len(scales):
        raise ValueError("weights must match scales length")

    result = np.zeros((height, width), dtype=np.float32)

    if noise_func == 'perlin':
        noise_fn = perlin_noise_2d
    elif noise_func == 'simplex':
        noise_fn = simplex_noise_2d
    else:
        raise ValueError(f"Unknown noise function: {noise_func}")

    # Accumulate scales
    total_weight = 0.0

    for i, (scale, weight) in enumerate(zip(scales, weights)):
        noise = noise_fn(
            width, height,
            frequency=scale,
            seed=seed + i
        )

        result += noise * weight
        total_weight += weight

    # Normalize
    if total_weight > 0:
        result = (result + total_weight) / (2.0 * total_weight)

    return np.clip(result, 0.0, 1.0)


def erosion_fbm(
    width: int,
    height: int,
    octaves: int = 10,
    lacunarity: float = 2.5,
    persistence: float = 0.6,
    erosion_strength: float = 0.3,
    seed: int = 0
) -> np.ndarray:
    """
    fBm with erosion-like amplitude modulation

    Lower octaves modulate higher octaves to create erosion-like patterns.
    More realistic than standard fBm.

    Args:
        width: Output width
        height: Output height
        octaves: Number of octaves (10+ recommended)
        lacunarity: Frequency multiplier
        persistence: Base amplitude falloff
        erosion_strength: How much lower octaves erode higher ones (0-1)
        seed: Random seed

    Returns:
        2D array in range [0, 1]

    Notes:
        This creates natural-looking erosion patterns without
        running an actual erosion simulation.
    """
    result = np.zeros((height, width), dtype=np.float32)

    # Generate all octaves first
    octave_layers = []
    current_frequency = 1.0

    for octave in range(octaves):
        noise = simplex_noise_2d(
            width, height,
            frequency=current_frequency,
            seed=seed + octave
        )
        octave_layers.append(noise)
        current_frequency *= lacunarity

    # Combine with modulation
    max_value = 0.0
    current_amplitude = 1.0

    for i, noise in enumerate(octave_layers):
        # Modulation factor from previous octaves
        if i > 0 and erosion_strength > 0:
            # Lower octaves reduce amplitude in high areas
            modulation = 1.0 - erosion_strength * np.clip(result / max_value if max_value > 0 else 0, 0, 1)
            effective_amplitude = current_amplitude * modulation
        else:
            effective_amplitude = current_amplitude

        result += noise * effective_amplitude
        max_value += current_amplitude

        current_amplitude *= persistence

    # Normalize
    result = (result + max_value) / (2.0 * max_value)

    return np.clip(result, 0.0, 1.0)


if __name__ == "__main__":
    # Test and visualize different fBm variants
    import matplotlib.pyplot as plt
    import time

    size = 512
    seed = 42

    print("Generating fBm variants...")

    # Standard fBm
    start = time.time()
    fbm = fractional_brownian_motion(size, size, octaves=8, seed=seed)
    print(f"fBm: {(time.time()-start)*1000:.1f}ms")

    # Turbulence
    start = time.time()
    turb = turbulence(size, size, octaves=8, seed=seed)
    print(f"Turbulence: {(time.time()-start)*1000:.1f}ms")

    # Billow
    start = time.time()
    bill = billow(size, size, octaves=8, seed=seed)
    print(f"Billow: {(time.time()-start)*1000:.1f}ms")

    # Erosion fBm
    start = time.time()
    erosion = erosion_fbm(size, size, octaves=10, seed=seed)
    print(f"Erosion fBm: {(time.time()-start)*1000:.1f}ms")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(fbm, cmap='terrain')
    axes[0, 0].set_title('Standard fBm')

    axes[0, 1].imshow(turb, cmap='gray')
    axes[0, 1].set_title('Turbulence')

    axes[1, 0].imshow(bill, cmap='terrain')
    axes[1, 0].set_title('Billow')

    axes[1, 1].imshow(erosion, cmap='terrain')
    axes[1, 1].set_title('Erosion fBm')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/fbm_variants.png', dpi=150)
    print("\nVisualization saved to /tmp/fbm_variants.png")
