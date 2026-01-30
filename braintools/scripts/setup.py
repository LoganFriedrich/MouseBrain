"""
SCI-Connectome Pipeline - Napari Plugin Installer

Installs the napari plugin for the BrainGlobe cell detection pipeline.

USAGE:
------
    conda activate brainglobe-env
    cd y:\\2_Connectome
    python setup.py

Or just double-click: INSTALL_NAPARI_PLUGINS.bat

WHAT YOU GET:
-------------
1. Pipeline Dashboard  - See your brains, run pipeline steps
2. Setup & Tuning      - Configure voxel size, orientation, detection params
3. Manual Crop         - Crop brain volumes (remove spinal cord)
4. Experiments         - Browse and compare experiment runs
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Install the napari plugin."""
    plugin_dir = Path(__file__).parent / "Tissue" / "3D_Cleared" / "util_Scripts" / "sci_connectome_napari"

    print("\n" + "=" * 60)
    print("SCI-Connectome Pipeline - Napari Plugin Installer")
    print("=" * 60)

    if plugin_dir.exists():
        print("\nInstalling SCI-Connectome Pipeline...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", str(plugin_dir)
        ], check=True)
    else:
        print(f"\nERROR: Plugin not found at {plugin_dir}")
        return

    print("\n" + "=" * 60)
    print("Done! Restart napari to see the plugin.")
    print("=" * 60)
    print("\nIn napari, go to:")
    print("  Plugins > SCI-Connectome Pipeline > [choose widget]")
    print()

if __name__ == "__main__":
    main()
