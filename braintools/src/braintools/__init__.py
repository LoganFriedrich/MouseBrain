"""
BrainTools - SCI-Connectome Tissue Analysis Tools

A unified package for BrainGlobe-based cell detection, registration,
and analysis workflows.

Includes:
- 3D Cleared Brain Pipeline (cellfinder, brainreg)
- 2D Slice Pipeline (SliceAtlas)
- Experiment/Calibration Tracking

Usage:
    braintool          # Launch napari with all plugins
    braintool --check  # Verify dependencies
    braintool --help   # Show help
"""

__version__ = "0.1.0"

# Path configuration - ALWAYS prefer Y: drive over UNC paths
# UNC paths cause massive slowdowns with dask/zarr/napari
from pathlib import Path

def _get_tissue_root():
    """Get the Tissue root, preferring Y: drive to avoid UNC path issues."""
    # 1. Prefer Y: drive (fast local mount)
    y_drive = Path(r"Y:\2_Connectome\Tissue")
    if y_drive.exists() and (y_drive / "3D_Cleared").exists():
        return y_drive

    # 2. Fall back to computing from __file__ (may be UNC path, slow)
    return Path(__file__).parent.parent.parent.parent

TISSUE_ROOT = _get_tissue_root()
BRAINTOOLS_ROOT = TISSUE_ROOT  # Alias for compatibility

# Sub-pipeline paths
CLEARED_3D_ROOT = TISSUE_ROOT / "3D_Cleared"
SLICES_2D_ROOT = TISSUE_ROOT / "2D_Slices"
INJURY_ROOT = TISSUE_ROOT / "Injury"

# Data paths
BRAINS_ROOT = CLEARED_3D_ROOT / "1_Brains"
DATA_SUMMARY_DIR = CLEARED_3D_ROOT / "2_Data_Summary"
SCRIPTS_DIR = CLEARED_3D_ROOT / "util_Scripts"
