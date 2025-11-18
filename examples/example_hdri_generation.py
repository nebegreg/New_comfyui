#!/usr/bin/env python3
"""
Example: HDRI Panoramic Generation
Mountain Studio Pro

Demonstrates:
- Procedural HDRI generation
- Multiple time-of-day presets
- Export to EXR and preview PNG
- Optional AI enhancement

Usage:
    python examples/example_hdri_generation.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from core.rendering.hdri_generator import HDRIPanoramicGenerator, TimeOfDay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate HDRI panoramas for all time presets."""
    output_dir = Path.home() / "mountain_studio_hdri_examples"
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Resolution options
    # resolution = (2048, 1024)  # Low
    resolution = (4096, 2048)  # Medium
    # resolution = (8192, 4096)  # High (requires more VRAM for AI)

    generator = HDRIPanoramicGenerator(resolution=resolution)

    # Generate all time-of-day presets
    times = [
        TimeOfDay.SUNRISE,
        TimeOfDay.MIDDAY,
        TimeOfDay.SUNSET,
        TimeOfDay.NIGHT
    ]

    for time_of_day in times:
        print(f"\n{'='*60}")
        print(f"Generating: {time_of_day.value.upper()}")
        print(f"{'='*60}")

        # Generate procedural HDRI
        hdri = generator.generate_procedural(
            time_of_day=time_of_day,
            cloud_density=0.3,
            mountain_distance=True,
            seed=42
        )

        # Save files
        base_name = f"mountain_hdri_{time_of_day.value}"

        # Export EXR (HDR format)
        exr_path = output_dir / f"{base_name}.exr"
        generator.export_exr(hdri, str(exr_path))
        print(f"  ✓ Saved EXR: {exr_path}")

        # Export tone-mapped preview (PNG)
        preview_path = output_dir / f"{base_name}_preview.png"
        generator.export_ldr(hdri, str(preview_path), tone_map=True)
        print(f"  ✓ Saved preview: {preview_path}")

    print(f"\n{'='*60}")
    print(f"✅ All HDRIs generated successfully!")
    print(f"{'='*60}")
    print(f"\nOutput location: {output_dir}")
    print(f"\nFiles generated:")
    for f in sorted(output_dir.glob("*.exr")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("*_preview.png")):
        print(f"  - {f.name}")

    print("\n💡 Tip: Use the .exr files in 3D software for HDR lighting")
    print("        Use the .png files for quick preview")

    # Optional: AI enhancement example
    print("\n" + "="*60)
    print("AI ENHANCEMENT (Optional)")
    print("="*60)
    print("To enhance HDRIs with AI:")
    print("  1. Install diffusers: pip install diffusers transformers accelerate")
    print("  2. Uncomment the AI enhancement section below")
    print("  3. Requires ~10-12 GB VRAM")

    # Uncomment to enable AI enhancement:
    """
    try:
        print("\nGenerating AI-enhanced HDRI (this may take 1-2 minutes)...")
        base_hdri = generator.generate_procedural(TimeOfDay.SUNSET, cloud_density=0.3)

        enhanced_hdri = generator.enhance_with_ai(
            base_hdri,
            prompt="360 degree panoramic view of majestic mountains at sunset, "
                   "highly detailed, photorealistic, dramatic clouds, 8k",
            strength=0.4,
            seed=42
        )

        generator.export_exr(enhanced_hdri, str(output_dir / "mountain_hdri_sunset_ai_enhanced.exr"))
        generator.export_ldr(enhanced_hdri, str(output_dir / "mountain_hdri_sunset_ai_enhanced_preview.png"))

        print("✓ AI-enhanced HDRI saved!")

    except Exception as e:
        print(f"⚠ AI enhancement failed (this is optional): {e}")
    """


if __name__ == "__main__":
    main()
