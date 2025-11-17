"""
Ultra-Fast Vectorized Noise Module for Mountain Studio Pro v2.0

This module provides production-quality, vectorized noise implementations
that are 100-1000x faster than nested loop approaches.

All functions are fully vectorized using NumPy operations for maximum performance.
"""

from .vectorized_noise import (
    perlin_noise_2d,
    simplex_noise_2d,
    cellular_noise_2d,
    value_noise_2d
)

from .fbm import (
    fractional_brownian_motion,
    turbulence,
    billow
)

from .ridged_multifractal import (
    ridged_multifractal,
    hybrid_multifractal,
    swiss_turbulence
)

from .domain_warp import (
    domain_warp_2d,
    advanced_domain_warp,
    flow_noise
)

__all__ = [
    # Base noise functions
    'perlin_noise_2d',
    'simplex_noise_2d',
    'cellular_noise_2d',
    'value_noise_2d',

    # fBm variants
    'fractional_brownian_motion',
    'turbulence',
    'billow',

    # Multifractal variants (best for mountains)
    'ridged_multifractal',
    'hybrid_multifractal',
    'swiss_turbulence',

    # Domain warping
    'domain_warp_2d',
    'advanced_domain_warp',
    'flow_noise'
]

__version__ = '2.0.0'
