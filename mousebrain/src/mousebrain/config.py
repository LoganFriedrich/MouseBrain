#!/usr/bin/env python3
"""
config.py - Central configuration for SCI-Connectome pipeline paths (mousebrain package).

This module provides portable path detection so the pipeline works on any system,
not just the lab's Y: drive setup.

Path Resolution Order:
1. Environment variable SCI_CONNECTOME_ROOT (if set)
2. Auto-detect from script location (walks up to find 3_Nuclei_Detection)
3. Fallback to Y:\\2_Connectome (for backwards compatibility with lab setup)

Usage in other scripts:
    from mousebrain.config import BRAINS_ROOT, SCRIPTS_DIR, MODELS_DIR, DATA_SUMMARY_DIR

    # These paths are now portable:
    brain_folder = BRAINS_ROOT / "349_CNT_01_02"

Environment Variable Override:
    Set SCI_CONNECTOME_ROOT to the root of your installation:

    Windows (cmd):
        set SCI_CONNECTOME_ROOT=C:\\Projects\\SCI-Connectome

    Windows (PowerShell):
        $env:SCI_CONNECTOME_ROOT = "C:\\Projects\\SCI-Connectome"

    Linux/Mac:
        export SCI_CONNECTOME_ROOT=/home/user/SCI-Connectome
"""

import os
import sys
from pathlib import Path
from typing import Optional
import warnings

# =============================================================================
# PATH DETECTION
# =============================================================================

def _find_repo_root() -> Optional[Path]:
    """
    Auto-detect the repository root by walking up from this script's location.

    Looks for the characteristic folder structure (NEW: Tissue/3D_Cleared):
    - Tissue/3D_Cleared/  (or legacy 3_Nuclei_Detection/)
      - util_Scripts/  (where this config.py lives)
      - 1_Brains/
      - 2_Data_Summary/

    Returns the root (2_Connectome equivalent).
    """
    # Start from this file's location
    current = Path(__file__).resolve().parent

    # Walk up looking for the pipeline structure
    for _ in range(10):  # Max 10 levels up to prevent infinite loop
        # Check if we're in util_Scripts under 3D_Cleared (new structure)
        if current.name == "util_Scripts":
            parent = current.parent
            if parent.name == "3D_Cleared":
                # New structure: Tissue/3D_Cleared -> return Tissue's parent (2_Connectome)
                tissue_dir = parent.parent
                if tissue_dir.name == "Tissue":
                    return tissue_dir.parent
            # Legacy: 3_Nuclei_Detection
            if parent.name == "3_Nuclei_Detection":
                return parent.parent

        # Check if current directory contains Tissue/3D_Cleared (new)
        cleared_3d = current / "Tissue" / "3D_Cleared"
        if cleared_3d.exists() and cleared_3d.is_dir():
            if (cleared_3d / "util_Scripts").exists():
                return current

        # Legacy: Check for 3_Nuclei_Detection
        nuclei_detection = current / "3_Nuclei_Detection"
        if nuclei_detection.exists() and nuclei_detection.is_dir():
            if (nuclei_detection / "util_Scripts").exists():
                return current

        # Move up one level
        if current.parent == current:
            break  # Reached filesystem root
        current = current.parent

    return None


def _get_root_path() -> Path:
    """
    Get the root path for the SCI-Connectome installation.

    Resolution order:
    1. SCI_CONNECTOME_ROOT environment variable
    2. Lab's Y: drive (preferred - avoids UNC path issues)
    3. Auto-detect from script location
    """
    # 1. Check environment variable first
    env_root = os.environ.get("SCI_CONNECTOME_ROOT")
    if env_root:
        env_path = Path(env_root)
        if env_path.exists():
            return env_path
        else:
            warnings.warn(
                f"SCI_CONNECTOME_ROOT is set to '{env_root}' but path doesn't exist. "
                f"Falling back to auto-detection."
            )

    # 2. Prefer Y: drive to avoid UNC path issues with zarr/dask
    y_drive = Path(r"Y:\2_Connectome")
    if y_drive.exists() and (y_drive / "Tissue" / "3D_Cleared").exists():
        return y_drive

    # 3. Try auto-detection (may return UNC path on network drives)
    detected = _find_repo_root()
    if detected and detected.exists():
        return detected

    # If nothing works, return Y: drive for error messages
    return y_drive


# =============================================================================
# EXPORTED PATHS
# =============================================================================

# Root of the installation (e.g., Y:\2_Connectome or wherever user installed)
ROOT_PATH = _get_root_path()

# Tissue processing root (NEW structure)
TISSUE_ROOT = ROOT_PATH / "Tissue"

# Main directories - try new structure first, fall back to legacy
if (TISSUE_ROOT / "3D_Cleared").exists():
    # New structure: Tissue/3D_Cleared
    CLEARED_3D_DIR = TISSUE_ROOT / "3D_Cleared"
else:
    # Legacy structure: 3_Nuclei_Detection
    CLEARED_3D_DIR = ROOT_PATH / "3_Nuclei_Detection"

# Alias for backwards compatibility
NUCLEI_DETECTION_DIR = CLEARED_3D_DIR

# Where brain data lives
BRAINS_ROOT = CLEARED_3D_DIR / "1_Brains"

# Where summary data and experiment tracking lives
DATA_SUMMARY_DIR = CLEARED_3D_DIR / "2_Data_Summary"

# Where the utility scripts live (this directory)
SCRIPTS_DIR = CLEARED_3D_DIR / "util_Scripts"

# Where trained models are stored
MODELS_DIR = CLEARED_3D_DIR / "util_Brainglobe" / "Trained_Models"

# Tracker CSV paths
# Calibration runs = optimization/tuning experiments (detection params, model training, etc.)
CALIBRATION_RUNS_CSV = DATA_SUMMARY_DIR / "calibration_runs.csv"
EXPERIMENTS_CSV = CALIBRATION_RUNS_CSV  # Backwards compatibility alias

# Production results = final per-region cell counts from routine pipeline runs
REGION_COUNTS_CSV = DATA_SUMMARY_DIR / "region_counts.csv"  # Current (one row per brain)
REGION_COUNTS_ARCHIVE_CSV = DATA_SUMMARY_DIR / "region_counts_archive.csv"  # Historical

# eLife-grouped results (same archiving pattern)
ELIFE_COUNTS_CSV = DATA_SUMMARY_DIR / "elife_counts.csv"  # Current (one row per brain)
ELIFE_COUNTS_ARCHIVE_CSV = DATA_SUMMARY_DIR / "elife_counts_archive.csv"  # Historical

# =============================================================================
# LEGACY ALIASES (for backwards compatibility)
# =============================================================================

# Some scripts use these names
DEFAULT_BRAINGLOBE_ROOT = BRAINS_ROOT
DEFAULT_MODELS_DIR = MODELS_DIR
DEFAULT_TRACKER_PATH = EXPERIMENTS_CSV
SUMMARY_DATA_DIR = DATA_SUMMARY_DIR


# =============================================================================
# VALIDATION AND DEBUGGING
# =============================================================================

def validate_paths(verbose: bool = False) -> bool:
    """
    Check that essential paths exist.

    Args:
        verbose: If True, print status of each path

    Returns:
        True if all essential paths exist, False otherwise
    """
    paths_to_check = {
        "ROOT_PATH": ROOT_PATH,
        "NUCLEI_DETECTION_DIR": NUCLEI_DETECTION_DIR,
        "BRAINS_ROOT": BRAINS_ROOT,
        "SCRIPTS_DIR": SCRIPTS_DIR,
    }

    all_exist = True
    for name, path in paths_to_check.items():
        exists = path.exists()
        if verbose:
            status = "OK" if exists else "MISSING"
            print(f"  {name}: {path} [{status}]")
        if not exists:
            all_exist = False

    return all_exist


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("SCI-Connectome Configuration")
    print("=" * 60)

    env_var = os.environ.get("SCI_CONNECTOME_ROOT", "(not set)")
    print(f"\nEnvironment variable SCI_CONNECTOME_ROOT: {env_var}")

    print(f"\nDetected/configured paths:")
    print(f"  ROOT_PATH:            {ROOT_PATH}")
    print(f"  NUCLEI_DETECTION_DIR: {NUCLEI_DETECTION_DIR}")
    print(f"  BRAINS_ROOT:          {BRAINS_ROOT}")
    print(f"  DATA_SUMMARY_DIR:     {DATA_SUMMARY_DIR}")
    print(f"  SCRIPTS_DIR:          {SCRIPTS_DIR}")
    print(f"  MODELS_DIR:           {MODELS_DIR}")
    print(f"  EXPERIMENTS_CSV:      {EXPERIMENTS_CSV}")

    print(f"\nPath validation:")
    validate_paths(verbose=True)
    print("=" * 60)


# =============================================================================
# BRAIN NAME PARSING - Data Hierarchy
# =============================================================================

# See 2_Data_Summary/DATA_HIERARCHY.md for full documentation
#
# Hierarchy (smallest to largest):
#   Brain (374) -> Subject (CNT_01_01) -> Cohort (CNT_01) -> Experiment -> Project (CNT)
#
# Naming convention: {BRAIN#}_{PROJECT}_{COHORT#}_{SUBJECT#}
# Example: 349_CNT_01_02 = Brain 349, Connectome project, Cohort 01, Subject 02

# Known project codes
PROJECT_CODES = {
    "CNT": "Connectome",
    "ENCR": "Enhancer",
    # Add more as needed
}


def parse_brain_name(brain_name: str) -> dict:
    """
    Parse a brain/folder name into its hierarchical components.

    Args:
        brain_name: Name like "349_CNT_01_02" or "349_CNT_01_02_1p625x_z4"

    Returns:
        dict with keys: brain_id, project_code, project_name, cohort, subject,
                        subject_full, cohort_full, imaging_params (if present)

    Example:
        >>> parse_brain_name("349_CNT_01_02_1p625x_z4")
        {
            'brain_id': '349',
            'project_code': 'CNT',
            'project_name': 'Connectome',
            'cohort': '01',
            'subject': '02',
            'subject_full': 'CNT_01_02',
            'cohort_full': 'CNT_01',
            'imaging_params': '1p625x_z4',
            'raw': '349_CNT_01_02_1p625x_z4'
        }
    """
    result = {
        'brain_id': None,
        'project_code': None,
        'project_name': None,
        'cohort': None,
        'subject': None,
        'subject_full': None,
        'cohort_full': None,
        'imaging_params': None,
        'raw': brain_name
    }

    if not brain_name:
        return result

    parts = brain_name.split('_')

    if len(parts) < 4:
        # Can't parse, return what we have
        return result

    # Standard format: BRAIN_PROJECT_COHORT_SUBJECT[_imaging_params...]
    result['brain_id'] = parts[0]
    result['project_code'] = parts[1]
    result['project_name'] = PROJECT_CODES.get(parts[1], parts[1])
    result['cohort'] = parts[2]
    result['subject'] = parts[3]

    # Build full identifiers
    result['cohort_full'] = f"{parts[1]}_{parts[2]}"
    result['subject_full'] = f"{parts[1]}_{parts[2]}_{parts[3]}"

    # Anything after subject is imaging parameters
    if len(parts) > 4:
        result['imaging_params'] = '_'.join(parts[4:])

    return result


def format_subject_id(project: str, cohort: str, subject: str) -> str:
    """Format a full subject ID from components."""
    return f"{project}_{cohort}_{subject}"


def format_cohort_id(project: str, cohort: str) -> str:
    """Format a full cohort ID from components."""
    return f"{project}_{cohort}"


def get_brain_hierarchy(brain_name: str) -> str:
    """
    Get a human-readable hierarchy string for a brain.

    Example:
        >>> get_brain_hierarchy("349_CNT_01_02")
        "Brain 349 | Subject CNT_01_02 | Cohort CNT_01 | Project Connectome"
    """
    parsed = parse_brain_name(brain_name)

    parts = []
    if parsed['brain_id']:
        parts.append(f"Brain {parsed['brain_id']}")
    if parsed['subject_full']:
        parts.append(f"Subject {parsed['subject_full']}")
    if parsed['cohort_full']:
        parts.append(f"Cohort {parsed['cohort_full']}")
    if parsed['project_name']:
        parts.append(f"Project {parsed['project_name']}")

    return " | ".join(parts) if parts else brain_name


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    """When run directly, print configuration info."""
    print_config()

    # Also demo the brain parser
    print("\n" + "=" * 60)
    print("Brain Name Parser Demo")
    print("=" * 60)
    test_names = [
        "349_CNT_01_02",
        "349_CNT_01_02_1p625x_z4",
        "374_ENCR_03_05",
    ]
    for name in test_names:
        print(f"\n{name}:")
        parsed = parse_brain_name(name)
        for k, v in parsed.items():
            if v:
                print(f"  {k}: {v}")
