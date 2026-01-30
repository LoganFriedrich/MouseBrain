"""
config.py - Path configuration for BrainSlice

Handles automatic detection of project root and defines standard paths.
Uses environment variable override or auto-detection from script location.

Usage:
    from braintools.pipeline_2d.sliceatlas.core.config import DATA_DIR, TRACKER_CSV
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


# =============================================================================
# PATH DETECTION
# =============================================================================

def _get_root_path() -> Path:
    """
    Detect BrainSlice root directory.

    Priority:
    1. Environment variable BRAINSLICE_ROOT
    2. Auto-detect by walking up from this file
    3. Fallback to parent of brainslice package
    """
    # Check environment variable
    env_root = os.environ.get("BRAINSLICE_ROOT")
    if env_root:
        root = Path(env_root)
        if root.exists():
            return root

    # Auto-detect: walk up from this file looking for BrainSlice marker
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Max 5 levels up
        if (current / "brainslice").is_dir() and (current / "environment.yml").exists():
            return current
        current = current.parent

    # Fallback: assume we're in brainslice/core/
    return Path(__file__).resolve().parent.parent.parent


# =============================================================================
# STANDARD PATHS
# =============================================================================

BRAINSLICE_ROOT = _get_root_path()

# Data directory (where sample folders live)
DATA_DIR = BRAINSLICE_ROOT / "BrainSlice_Data"

# Tracker CSV for calibration runs
TRACKER_CSV = DATA_DIR / "calibration_runs.csv"

# Models directory (for custom StarDist models, etc.)
MODELS_DIR = BRAINSLICE_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SAMPLE NAME PARSING
# =============================================================================

# Project code mappings (short code → full name)
PROJECT_CODES = {
    "E": "ENCR",
    "C": "CNT",
    "ENCR": "ENCR",
    "CNT": "CNT",
}


def parse_sample_name(sample_name: str) -> Dict[str, Any]:
    """
    Parse sample ID into components.

    Expected formats:
    - {P}{CC}_{SS}_S{N}: e.g., "E02_01_S12" (ENCR cohort 02, subject 01, slice 12)
    - {PROJECT}_{COHORT}_{SUBJECT}_S{N}: e.g., "ENCR_02_01_S12"
    - {PROJECT}_{COHORT}_{SLICE}: e.g., "ENCR_001_slice_12"
    - Or any custom format (returns as-is if can't parse)

    Args:
        sample_name: Sample identifier string

    Returns:
        Dict with parsed components:
        - sample_id: Full sample name
        - project: Project code (expanded to full name)
        - project_short: Short project code
        - cohort: Cohort number (e.g., "02")
        - subject: Subject number (e.g., "01")
        - slice_num: Slice number if parseable
        - raw: Original string
    """
    import re

    result = {
        "sample_id": sample_name,
        "raw": sample_name,
        "project": "",
        "project_short": "",
        "cohort": "",
        "subject": "",
        "slice_num": "",
    }

    # Pattern 1: Compact format - E02_01_S12 or E02_01_S12_inset
    # {P}{CC}_{SS}_S{N} where P=project letter, CC=cohort, SS=subject, N=slice
    compact_match = re.match(
        r'^([A-Z])(\d{2})_(\d{2})_S(\d+)(?:_(.+))?$',
        sample_name
    )
    if compact_match:
        short_code = compact_match.group(1)
        result["project_short"] = short_code
        result["project"] = PROJECT_CODES.get(short_code, short_code)
        result["cohort"] = compact_match.group(2)
        result["subject"] = compact_match.group(3)
        result["slice_num"] = compact_match.group(4)
        if compact_match.group(5):
            result["suffix"] = compact_match.group(5)
        return result

    # Pattern 2: Full format - ENCR_02_01_S12
    full_match = re.match(
        r'^([A-Z]+)_(\d{2})_(\d{2})_S(\d+)(?:_(.+))?$',
        sample_name
    )
    if full_match:
        full_code = full_match.group(1)
        result["project"] = PROJECT_CODES.get(full_code, full_code)
        result["project_short"] = full_code[0] if full_code else ""
        result["cohort"] = full_match.group(2)
        result["subject"] = full_match.group(3)
        result["slice_num"] = full_match.group(4)
        if full_match.group(5):
            result["suffix"] = full_match.group(5)
        return result

    # Fallback: generic underscore splitting
    parts = sample_name.split("_")
    result["project"] = parts[0] if parts else ""
    result["project_short"] = result["project"][0] if result["project"] else ""

    # Try to extract slice number from any part
    for part in parts:
        if part.lower().startswith("slice"):
            result["slice_num"] = part.lower().replace("slice", "")
        elif part.startswith("S") and part[1:].isdigit():
            result["slice_num"] = part[1:]
        elif part.isdigit() and len(parts) > 2:
            result["slice_num"] = part

    # Cohort is typically second part
    if len(parts) >= 2:
        result["cohort"] = parts[1]

    return result


def parse_filename(filepath: str) -> Dict[str, Any]:
    """
    Parse a filename (with or without path) into sample components.

    Args:
        filepath: Path or filename (e.g., "y:/data/E02_01_S12.nd2")

    Returns:
        Same dict as parse_sample_name, with additional 'extension' key
    """
    path = Path(filepath)
    stem = path.stem  # filename without extension

    result = parse_sample_name(stem)
    result["extension"] = path.suffix.lower()
    result["filename"] = path.name

    return result


def sample_id_from_path(filepath: str) -> str:
    """
    Extract a canonical sample ID from a file path.

    Args:
        filepath: Path to ND2 or TIFF file

    Returns:
        Sample ID string (stem of filename)
    """
    return Path(filepath).stem


def get_sample_dir(sample_id: str) -> Path:
    """
    Get the data directory for a specific sample.

    Args:
        sample_id: Sample identifier

    Returns:
        Path to sample's data directory
    """
    return DATA_DIR / sample_id


def get_sample_subdir(sample_id: str, subdir: str) -> Path:
    """
    Get a specific subdirectory for a sample.

    Args:
        sample_id: Sample identifier
        subdir: Subdirectory name (e.g., "0_Raw", "2_Registered")

    Returns:
        Path to subdirectory (created if doesn't exist)
    """
    path = get_sample_dir(sample_id) / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# STANDARD SUBDIRECTORY NAMES
# =============================================================================

class SampleDirs:
    """Standard subdirectory names for sample data."""
    RAW = "0_Raw"
    PREPROCESSED = "1_Preprocessed"
    REGISTERED = "2_Registered"
    DETECTED = "3_Detected"
    QUANTIFIED = "4_Quantified"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_paths() -> Dict[str, bool]:
    """
    Validate that required paths exist.

    Returns:
        Dict mapping path names to existence status
    """
    return {
        "BRAINSLICE_ROOT": BRAINSLICE_ROOT.exists(),
        "DATA_DIR": DATA_DIR.exists(),
        "MODELS_DIR": MODELS_DIR.exists(),
    }


# =============================================================================
# DEBUG INFO
# =============================================================================

if __name__ == "__main__":
    print("BrainSlice Configuration")
    print("=" * 50)
    print(f"BRAINSLICE_ROOT: {BRAINSLICE_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"TRACKER_CSV: {TRACKER_CSV}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print()
    print("Path validation:")
    for name, exists in validate_paths().items():
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")
