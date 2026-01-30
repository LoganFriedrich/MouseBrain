"""
BrainTools - SCI-Connectome Tissue Analysis Tools

A unified package for BrainGlobe-based cell detection, registration,
and analysis workflows.

Includes:
- 3D Cleared Brain Pipeline (cellfinder, brainreg) — install with braintools[3d]
- 2D Slice Pipeline (SliceAtlas + Annotator) — install with braintools[2d]
- Experiment/Calibration Tracking

Usage:
    braintool          # Launch napari with all plugins
    braintool --check  # Verify dependencies
    braintool --help   # Show help
"""

__version__ = "0.2.0"

# Re-export path configuration from config module
# config.py is the single source of truth for all paths
from braintools.config import (
    ROOT_PATH,
    TISSUE_ROOT,
    CLEARED_3D_DIR,
    NUCLEI_DETECTION_DIR,
    BRAINS_ROOT,
    DATA_SUMMARY_DIR,
    SCRIPTS_DIR,
    MODELS_DIR,
    CALIBRATION_RUNS_CSV,
    EXPERIMENTS_CSV,
    REGION_COUNTS_CSV,
    REGION_COUNTS_ARCHIVE_CSV,
    ELIFE_COUNTS_CSV,
    ELIFE_COUNTS_ARCHIVE_CSV,
    parse_brain_name,
    format_subject_id,
    format_cohort_id,
    get_brain_hierarchy,
    validate_paths,
    print_config,
)

# Convenience aliases matching old __init__.py names
BRAINTOOLS_ROOT = TISSUE_ROOT
CLEARED_3D_ROOT = CLEARED_3D_DIR
SLICES_2D_ROOT = TISSUE_ROOT / "2D_Slices"
INJURY_ROOT = TISSUE_ROOT / "Injury"
