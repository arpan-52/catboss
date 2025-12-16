#!/usr/bin/env python3
"""
Smart casacore detection for build system.
Finds casacore in multiple locations automatically.
"""
import subprocess
import os
import sys
from pathlib import Path


def find_casacore():
    """
    Find casacore installation using multiple methods.
    Returns: (include_dirs, library_dirs, libraries)
    """

    # Method 1: Try pkg-config (most reliable)
    try:
        print("Trying pkg-config for casacore...")
        cflags = subprocess.check_output(
            ['pkg-config', '--cflags', 'casacore'],
            stderr=subprocess.DEVNULL
        ).decode().strip().split()

        libs = subprocess.check_output(
            ['pkg-config', '--libs', 'casacore'],
            stderr=subprocess.DEVNULL
        ).decode().strip().split()

        # Parse flags
        include_dirs = [f[2:] for f in cflags if f.startswith('-I')]
        library_dirs = [f[2:] for f in libs if f.startswith('-L')]
        libraries = [f[2:] for f in libs if f.startswith('-l')]

        print(f"✓ Found via pkg-config:")
        print(f"  Include: {include_dirs}")
        print(f"  Libs: {library_dirs}")
        return include_dirs, library_dirs, libraries

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  pkg-config not found or casacore not registered")

    # Method 2: Check common installation paths
    common_paths = [
        '/usr/include/casacore',
        '/usr/local/include/casacore',
        '/opt/homebrew/include/casacore',  # macOS ARM
        '/opt/local/include/casacore',      # MacPorts
        os.path.expanduser('~/.local/include/casacore'),
        os.path.expanduser('~/anaconda3/include/casacore'),
        os.path.expanduser('~/miniconda3/include/casacore'),
    ]

    # Add conda environment if present
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        common_paths.insert(0, os.path.join(conda_prefix, 'include', 'casacore'))

    print("Checking common paths...")
    for path in common_paths:
        if os.path.exists(path):
            include_dir = str(Path(path).parent)
            # Guess library path
            lib_dir = str(Path(path).parent.parent / 'lib')

            print(f"✓ Found casacore at: {path}")
            print(f"  Include: {include_dir}")
            print(f"  Lib (guessed): {lib_dir}")

            return (
                [include_dir],
                [lib_dir] if os.path.exists(lib_dir) else [],
                ['casa_tables', 'casa_ms', 'casa_casa']
            )

    # Method 3: Try to import python-casacore and use its path
    try:
        print("Trying to find casacore via python-casacore...")
        import casacore
        casacore_path = Path(casacore.__file__).parent

        # Check if there's a bundled casacore
        possible_include = casacore_path / 'include'
        possible_lib = casacore_path / 'lib'

        if possible_include.exists():
            print(f"✓ Found bundled casacore in python-casacore")
            return (
                [str(possible_include)],
                [str(possible_lib)] if possible_lib.exists() else [],
                ['casa_tables', 'casa_ms', 'casa_casa']
            )
    except ImportError:
        print("  python-casacore not installed")

    # Method 4: Environment variable override
    if 'CASACORE_ROOT' in os.environ:
        root = Path(os.environ['CASACORE_ROOT'])
        print(f"Using CASACORE_ROOT: {root}")
        return (
            [str(root / 'include')],
            [str(root / 'lib')],
            ['casa_tables', 'casa_ms', 'casa_casa']
        )

    print("✗ Could not find casacore!")
    print("\nPlease install casacore using one of:")
    print("  - Ubuntu/Debian: sudo apt-get install casacore-dev")
    print("  - macOS: brew install casacore")
    print("  - Conda: conda install -c conda-forge casacore")
    print("  - Or set CASACORE_ROOT environment variable")
    sys.exit(1)


if __name__ == "__main__":
    includes, lib_dirs, libs = find_casacore()
    print("\nFinal configuration:")
    print(f"  include_dirs = {includes}")
    print(f"  library_dirs = {lib_dirs}")
    print(f"  libraries = {libs}")
