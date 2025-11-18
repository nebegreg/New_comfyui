#!/usr/bin/env python3
"""
Mountain Studio Pro - Professional Launcher
Launches the Ultimate Terrain Viewer with comprehensive error handling

Usage:
    python3 launch_mountain_studio.py [OPTIONS]

Options:
    --mode MODE          Launch mode: 'viewer' (default) or 'hdri'
    --no-gui             Run in CLI mode (HDRI generation only)
    --test               Run system tests
    --check-deps         Check dependencies
    --help               Show this help

Examples:
    # Launch main viewer
    python3 launch_mountain_studio.py

    # Generate HDRI examples
    python3 launch_mountain_studio.py --mode hdri

    # Check dependencies
    python3 launch_mountain_studio.py --check-deps

    # Run tests
    python3 launch_mountain_studio.py --test
"""

import sys
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class DependencyChecker:
    """Check and report on system dependencies."""

    def __init__(self):
        self.missing_required = []
        self.missing_optional = []

    def check_all(self, verbose=True):
        """Check all dependencies."""
        if verbose:
            print('='*70)
            print('  DEPENDENCY CHECK')
            print('='*70)

        # Required dependencies
        required = [
            ('numpy', 'NumPy'),
            ('PIL', 'Pillow'),
            ('PySide6.QtCore', 'PySide6'),
        ]

        for module, name in required:
            if self._check_import(module):
                if verbose:
                    print(f'  ✓ {name}')
            else:
                self.missing_required.append(name)
                if verbose:
                    print(f'  ❌ {name} (REQUIRED)')

        # OpenGL (required for 3D viewer)
        if self._check_import('OpenGL.GL'):
            if verbose:
                print('  ✓ PyOpenGL (required for 3D rendering)')
        else:
            self.missing_required.append('PyOpenGL')
            if verbose:
                print('  ❌ PyOpenGL (REQUIRED for 3D rendering)')

        # Optional dependencies
        optional = [
            ('OpenEXR', 'OpenEXR (for .exr export)'),
            ('diffusers', 'Diffusers (for AI HDRI enhancement)'),
            ('torch', 'PyTorch (for AI features)'),
        ]

        if verbose:
            print('\n  Optional Dependencies:')

        for module, name in optional:
            if self._check_import(module):
                if verbose:
                    print(f'  ✓ {name}')
            else:
                self.missing_optional.append(name)
                if verbose:
                    print(f'  ⚠  {name} (optional)')

        return len(self.missing_required) == 0

    def _check_import(self, module_name):
        """Try to import a module."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def print_installation_help(self):
        """Print help for installing missing dependencies."""
        if not self.missing_required and not self.missing_optional:
            return

        print('\n' + '='*70)
        print('  INSTALLATION HELP')
        print('='*70)

        if self.missing_required:
            print('\n❌ REQUIRED dependencies missing:')
            print('   Install with:')
            print('   pip install PyOpenGL PyOpenGL-accelerate PySide6 numpy Pillow\n')

        if self.missing_optional:
            print('\n⚠  OPTIONAL dependencies missing:')
            print('   For HDRI .exr export:')
            print('     pip install OpenEXR Imath')
            print('\n   For AI enhancement (requires 10+ GB VRAM):')
            print('     pip install diffusers transformers accelerate torch')


def launch_viewer():
    """Launch the Ultimate Terrain Viewer."""
    print('\n' + '█'*70)
    print('█' + ' '*68 + '█')
    print('█' + '  MOUNTAIN STUDIO PRO - ULTIMATE VIEWER  '.center(68) + '█')
    print('█' + ' '*68 + '█')
    print('█'*70 + '\n')

    try:
        from PySide6.QtWidgets import QApplication
        from ui.widgets.ultimate_terrain_viewer import UltimateTerrainViewer
        from core.terrain.advanced_algorithms import spectral_synthesis, stream_power_erosion

        # Check OpenGL
        try:
            from OpenGL.GL import glGetString, GL_VERSION
            logger.info('OpenGL available - 3D rendering enabled')
        except ImportError:
            logger.error('PyOpenGL not found - 3D rendering disabled')
            logger.error('Install with: pip install PyOpenGL PyOpenGL-accelerate')
            return False

        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName('Mountain Studio Pro')

        # Create main window
        logger.info('Initializing Ultimate Viewer...')
        viewer = UltimateTerrainViewer()

        # Generate default terrain
        logger.info('Generating default terrain (512x512 Alps)...')
        terrain_size = 512
        terrain = spectral_synthesis(terrain_size, beta=2.2, seed=42)
        terrain = stream_power_erosion(terrain, iterations=50, K_erosion=0.015)

        viewer._current_heightmap = terrain
        viewer._update_terrain()

        # Show window
        viewer.show()
        logger.info('Viewer launched successfully!')
        logger.info('')
        logger.info('Controls:')
        logger.info('  WASD - Move camera')
        logger.info('  Space/Shift - Up/Down')
        logger.info('  Mouse - Look around (click in viewport to capture mouse)')
        logger.info('  R - Reset camera')
        logger.info('  C - Toggle collision')
        logger.info('')
        logger.info('Explore the tabs for shadows, HDRI, lighting, and more!')

        # Run application
        return app.exec() == 0

    except ImportError as e:
        logger.error(f'Import failed: {e}')
        logger.error('Please install required dependencies.')
        return False
    except Exception as e:
        logger.error(f'Failed to launch viewer: {e}')
        import traceback
        traceback.print_exc()
        return False


def launch_hdri_generator():
    """Launch HDRI generation example."""
    print('\n' + '█'*70)
    print('█' + ' '*68 + '█')
    print('█' + '  MOUNTAIN STUDIO PRO - HDRI GENERATOR  '.center(68) + '█')
    print('█' + ' '*68 + '█')
    print('█'*70 + '\n')

    try:
        # Import example
        examples_dir = PROJECT_ROOT / 'examples'
        sys.path.insert(0, str(examples_dir))

        # Run HDRI generation example
        from example_hdri_generation import main as hdri_main
        hdri_main()
        return True

    except ImportError as e:
        logger.error(f'Import failed: {e}')
        return False
    except Exception as e:
        logger.error(f'HDRI generation failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """Run system tests."""
    try:
        from test_ultimate_system import main as test_main
        return test_main() == 0
    except Exception as e:
        logger.error(f'Tests failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description='Mountain Studio Pro - Ultimate Features Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--mode',
        choices=['viewer', 'hdri'],
        default='viewer',
        help='Launch mode (default: viewer)'
    )

    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in CLI mode (HDRI only)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run system tests'
    )

    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies'
    )

    args = parser.parse_args()

    # Check dependencies first
    checker = DependencyChecker()
    deps_ok = checker.check_all(verbose=args.check_deps or args.test)

    if args.check_deps:
        checker.print_installation_help()
        return 0 if deps_ok else 1

    if not deps_ok:
        logger.error('Missing required dependencies!')
        checker.print_installation_help()
        return 1

    # Run tests
    if args.test:
        return 0 if run_tests() else 1

    # Launch based on mode
    if args.mode == 'hdri' or args.no_gui:
        success = launch_hdri_generator()
    else:
        success = launch_viewer()

    return 0 if success else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n\nInterrupted by user.')
        sys.exit(130)
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
