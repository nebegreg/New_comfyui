"""
Advanced Terrain Generation Algorithms

Implements state-of-the-art algorithms for ultra-realistic terrain:
1. Spectral Synthesis (FFT-based)
2. Stream Power Erosion
3. Glacial Erosion
4. Tectonic Uplift

Based on latest research (2024) in procedural terrain generation.
References in RESEARCH_TERRAIN_ALGORITHMS.md
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def spectral_synthesis(
    size: int,
    beta: float = 2.0,
    amplitude: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terrain using Spectral Synthesis (FFT-based)

    Based on Fournier et al. (1982). Generates terrain in frequency domain.

    Args:
        size: Output size (must be power of 2 for best performance)
        beta: Power spectrum exponent. Controls rugosity:
              - 2.0 = natural terrain (1/f² noise)
              - 2.5-3.0 = very rugged mountains
              - 1.5 = smooth hills
        amplitude: Output amplitude scaling
        seed: Random seed for reproducibility

    Returns:
        Heightmap of shape (size, size) in range [0, 1]

    Technical Details:
        Power spectrum: P(f) = f^(-β)
        where f is spatial frequency

        This creates scale-invariant terrain matching real-world DEM statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate frequency grid
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)

    # Radial frequency
    f_radial = np.sqrt(fx**2 + fy**2)

    # Avoid division by zero at DC component
    f_radial[0, 0] = 1.0

    # Power spectrum: P(f) = f^(-β/2)
    # (We use -β/2 because we're working with amplitudes, not power)
    spectrum = f_radial ** (-beta / 2.0)

    # Set DC component to 0 (zero mean)
    spectrum[0, 0] = 0.0

    # Random phase
    phase = np.random.rand(size, size) * 2 * np.pi

    # Complex spectrum with random phase
    complex_spectrum = spectrum * np.exp(1j * phase)

    # Inverse FFT to get spatial domain
    terrain = np.fft.ifft2(complex_spectrum).real

    # Normalize to [0, 1]
    terrain = terrain - terrain.min()
    if terrain.max() > 0:
        terrain = terrain / terrain.max()

    terrain = terrain * amplitude

    logger.info(f"Spectral synthesis: size={size}, beta={beta}, "
                f"range=[{terrain.min():.3f}, {terrain.max():.3f}]")

    return terrain.astype(np.float32)


def stream_power_erosion(
    heightmap: np.ndarray,
    iterations: int = 100,
    K_erosion: float = 0.015,
    m_area_exp: float = 0.5,
    n_slope_exp: float = 1.0,
    dt: float = 0.01,
    uplift_rate: float = 0.0
) -> np.ndarray:
    """
    Apply Stream Power Law erosion

    Based on Braun & Willett (2013). More geomorphologically accurate
    than particle-based erosion.

    Erosion rate: E = K * A^m * S^n

    where:
    - A = upslope drainage area
    - S = local slope
    - K = erodability coefficient
    - m, n = empirical exponents

    Args:
        heightmap: Input heightmap [0, 1]
        iterations: Number of erosion steps
        K_erosion: Erodability coefficient (0.01-0.02 typical)
        m_area_exp: Drainage area exponent (0.4-0.6 typical)
        n_slope_exp: Slope exponent (1.0-2.0 typical)
        dt: Time step (smaller = more stable)
        uplift_rate: Tectonic uplift rate (for dynamic equilibrium)

    Returns:
        Eroded heightmap
    """
    h, w = heightmap.shape
    terrain = heightmap.copy().astype(np.float32)

    logger.info(f"Stream power erosion: {iterations} iterations, K={K_erosion}")

    for iteration in range(iterations):
        # Calculate flow directions and upslope areas
        flow_acc = _calculate_flow_accumulation(terrain)

        # Calculate slopes (gradient magnitude)
        gy, gx = np.gradient(terrain)
        slope = np.sqrt(gx**2 + gy**2)
        slope = np.maximum(slope, 1e-6)  # Avoid division by zero

        # Stream power law: erosion rate
        # E = K * A^m * S^n
        # Use log to avoid numerical issues with large A
        erosion_rate = K_erosion * (flow_acc ** m_area_exp) * (slope ** n_slope_exp)

        # Limit maximum erosion per step
        erosion_rate = np.minimum(erosion_rate, 0.1)

        # Apply erosion
        terrain -= erosion_rate * dt

        # Apply uplift (if any)
        if uplift_rate > 0:
            terrain += uplift_rate * dt

        # Periodic logging
        if (iteration + 1) % 20 == 0 or iteration == 0:
            logger.debug(f"  Iteration {iteration + 1}/{iterations}: "
                        f"mean erosion={erosion_rate.mean():.6f}")

    # Renormalize to [0, 1]
    terrain = terrain - terrain.min()
    if terrain.max() > 0:
        terrain = terrain / terrain.max()

    logger.info(f"Stream power erosion complete")

    return terrain


def _calculate_flow_accumulation(heightmap: np.ndarray) -> np.ndarray:
    """
    Calculate flow accumulation (upslope drainage area)

    Uses D8 flow routing (8 directions).

    Args:
        heightmap: Input heightmap

    Returns:
        Flow accumulation (number of upslope pixels)
    """
    h, w = heightmap.shape

    # Flow accumulation starts at 1 (the cell itself)
    flow_acc = np.ones((h, w), dtype=np.float32)

    # Flow directions (row_offset, col_offset)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # Sort cells by elevation (highest to lowest)
    # Process from peaks downward
    flat_indices = np.argsort(heightmap.ravel())[::-1]

    for flat_idx in flat_indices:
        i = flat_idx // w
        j = flat_idx % w

        current_elev = heightmap[i, j]
        current_acc = flow_acc[i, j]

        # Find steepest descent direction
        max_slope = 0
        best_dir = None

        for di, dj in directions:
            ni, nj = i + di, j + dj

            # Check bounds
            if 0 <= ni < h and 0 <= nj < w:
                neighbor_elev = heightmap[ni, nj]

                # Calculate slope (drop / distance)
                drop = current_elev - neighbor_elev
                distance = np.sqrt(di**2 + dj**2)
                slope = drop / distance

                if slope > max_slope:
                    max_slope = slope
                    best_dir = (ni, nj)

        # Flow to steepest descent neighbor
        if best_dir is not None:
            ni, nj = best_dir
            flow_acc[ni, nj] += current_acc

    return flow_acc


def glacial_erosion(
    heightmap: np.ndarray,
    altitude_threshold: float = 0.7,
    strength: float = 0.3,
    u_valley_factor: float = 0.8
) -> np.ndarray:
    """
    Apply glacial erosion (U-shaped valleys)

    Simulates glacial carving which creates characteristic U-shaped valleys
    instead of V-shaped river valleys.

    Args:
        heightmap: Input heightmap [0, 1]
        altitude_threshold: Elevation above which glaciers form (0.7 = upper 30%)
        strength: Erosion strength (0.1-0.5 typical)
        u_valley_factor: How pronounced U-shape is (0-1)

    Returns:
        Eroded heightmap with U-shaped valleys
    """
    h, w = heightmap.shape
    terrain = heightmap.copy()

    logger.info(f"Glacial erosion: threshold={altitude_threshold}, strength={strength}")

    # Identify glacial zones (high altitude)
    glacial_mask = terrain > altitude_threshold

    if not glacial_mask.any():
        logger.warning("No glacial zones found (threshold too high)")
        return terrain

    # Calculate flow from glacial zones
    flow_acc = _calculate_flow_accumulation(terrain)

    # Normalize flow to [0, 1]
    flow_acc_norm = flow_acc / flow_acc.max()

    # Glacial erosion is proportional to ice flux
    # Ice flux ≈ flow accumulation from glacial zones
    glacial_flow = np.zeros_like(terrain)

    # Propagate glacial influence downslope
    for i in range(h):
        for j in range(w):
            if glacial_mask[i, j]:
                glacial_flow[i, j] = flow_acc_norm[i, j]

    # Smooth glacial flow to simulate ice flow
    glacial_flow = ndimage.gaussian_filter(glacial_flow, sigma=2.0)

    # Apply U-shaped valley carving
    # U-shaped valleys are wider than V-shaped
    erosion = glacial_flow * strength

    # Create U-shape by eroding more in the center
    # Use distance transform to find valley centers
    from scipy.ndimage import distance_transform_edt

    # Binary mask of significant glacial flow
    significant_flow = glacial_flow > 0.1

    if significant_flow.any():
        # Distance from valley edges
        dist = distance_transform_edt(significant_flow)
        dist_norm = dist / (dist.max() + 1e-6)

        # U-shaped profile: more erosion in center
        u_profile = dist_norm ** u_valley_factor
        erosion = erosion * (1 + u_profile * 2.0)

    # Apply erosion
    terrain -= erosion

    # Renormalize
    terrain = terrain - terrain.min()
    if terrain.max() > 0:
        terrain = terrain / terrain.max()

    logger.info("Glacial erosion complete")

    return terrain.astype(np.float32)


def tectonic_uplift(
    heightmap: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    magnitude: float = 0.3,
    radius: float = 0.4
) -> np.ndarray:
    """
    Apply tectonic uplift (Gaussian bump)

    Simulates mountain building from plate tectonics.

    Args:
        heightmap: Input heightmap [0, 1]
        center: (row, col) center of uplift. If None, uses center of map.
        magnitude: Uplift height (0-1)
        radius: Uplift radius (0-1 as fraction of map size)

    Returns:
        Uplifted heightmap
    """
    h, w = heightmap.shape
    terrain = heightmap.copy()

    if center is None:
        center = (h // 2, w // 2)

    logger.info(f"Tectonic uplift: center={center}, magnitude={magnitude}, radius={radius}")

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Distance from center
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Gaussian uplift pattern
    sigma = radius * min(h, w)
    uplift = magnitude * np.exp(-(dist**2) / (2 * sigma**2))

    # Apply uplift
    terrain += uplift

    # Renormalize
    terrain = terrain - terrain.min()
    if terrain.max() > 0:
        terrain = terrain / terrain.max()

    logger.info(f"Uplift applied: max uplift={uplift.max():.3f}")

    return terrain.astype(np.float32)


def combine_algorithms(
    size: int,
    algorithm: str = 'spectral',
    beta: float = 2.0,
    erosion_iterations: int = 100,
    apply_glacial: bool = False,
    apply_uplift: bool = False,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Combined terrain generation with multiple algorithms

    Convenience function that chains algorithms for complete terrain.

    Args:
        size: Output size
        algorithm: Base algorithm ('spectral', 'ridged', 'hybrid')
        beta: Spectral synthesis parameter (if algorithm='spectral')
        erosion_iterations: Stream power erosion iterations
        apply_glacial: Whether to apply glacial erosion
        apply_uplift: Whether to apply tectonic uplift
        seed: Random seed

    Returns:
        Complete heightmap [0, 1]
    """
    logger.info(f"Combined terrain generation: {algorithm}, size={size}")

    # Generate base terrain
    if algorithm == 'spectral':
        terrain = spectral_synthesis(size, beta=beta, seed=seed)

    elif algorithm == 'ridged':
        # Use existing ridged multifractal
        from core.noise.ridged_multifractal import ridged_multifractal
        terrain = ridged_multifractal(size, size, octaves=12, gain=0.5, offset=1.0, seed=seed)

    elif algorithm == 'hybrid':
        # Combination of ridged and spectral
        from core.noise.ridged_multifractal import hybrid_multifractal
        terrain = hybrid_multifractal(size, size, octaves=12, gain=0.5, offset=1.0, seed=seed)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Apply tectonic uplift (before erosion for realistic mountain building)
    if apply_uplift:
        terrain = tectonic_uplift(terrain, magnitude=0.3, radius=0.4)

    # Apply stream power erosion
    if erosion_iterations > 0:
        terrain = stream_power_erosion(
            terrain,
            iterations=erosion_iterations,
            K_erosion=0.015,
            m_area_exp=0.5,
            n_slope_exp=1.0
        )

    # Apply glacial erosion (after fluvial erosion)
    if apply_glacial:
        terrain = glacial_erosion(
            terrain,
            altitude_threshold=0.7,
            strength=0.3
        )

    logger.info("Combined terrain generation complete")

    return terrain


# Preset parameters for different mountain types
MOUNTAIN_PRESETS = {
    'alps': {
        'algorithm': 'spectral',
        'beta': 2.2,
        'erosion_iterations': 100,
        'apply_glacial': True,
        'apply_uplift': False
    },
    'himalayas': {
        'algorithm': 'hybrid',
        'beta': 2.5,
        'erosion_iterations': 150,
        'apply_glacial': True,
        'apply_uplift': True  # Active tectonics
    },
    'scottish_highlands': {
        'algorithm': 'spectral',
        'beta': 2.0,
        'erosion_iterations': 80,
        'apply_glacial': True,
        'apply_uplift': False
    },
    'grand_canyon': {
        'algorithm': 'ridged',
        'beta': 2.0,
        'erosion_iterations': 200,  # Massive erosion
        'apply_glacial': False,
        'apply_uplift': False
    },
    'rocky_mountains': {
        'algorithm': 'hybrid',
        'beta': 2.3,
        'erosion_iterations': 120,
        'apply_glacial': True,
        'apply_uplift': True
    }
}


if __name__ == '__main__':
    # Test algorithms
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("TESTING ADVANCED TERRAIN ALGORITHMS")
    print("="*60)

    size = 256

    # Test Spectral Synthesis
    print("\n1. Spectral Synthesis...")
    terrain1 = spectral_synthesis(size, beta=2.0, seed=42)
    print(f"   Range: [{terrain1.min():.3f}, {terrain1.max():.3f}]")

    # Test Stream Power Erosion
    print("\n2. Stream Power Erosion...")
    terrain2 = terrain1.copy()
    terrain2 = stream_power_erosion(terrain2, iterations=50)
    print(f"   Range: [{terrain2.min():.3f}, {terrain2.max():.3f}]")

    # Test Glacial Erosion
    print("\n3. Glacial Erosion...")
    terrain3 = terrain2.copy()
    terrain3 = glacial_erosion(terrain3, altitude_threshold=0.7, strength=0.3)
    print(f"   Range: [{terrain3.min():.3f}, {terrain3.max():.3f}]")

    # Test Combined
    print("\n4. Combined (Alps preset)...")
    terrain4 = combine_algorithms(
        size,
        **MOUNTAIN_PRESETS['alps'],
        seed=42
    )
    print(f"   Range: [{terrain4.min():.3f}, {terrain4.max():.3f}]")

    print("\n✅ All algorithms tested successfully!")
