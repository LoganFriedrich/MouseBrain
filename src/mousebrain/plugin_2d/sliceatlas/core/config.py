"""
config.py - Path configuration for BrainSlice (2D slice analysis)

Imports canonical paths from mousebrain.config when running as part of the
monorepo, with fallback for standalone brainslice usage.

Usage:
    from mousebrain.plugin_2d.sliceatlas.core.config import DATA_DIR, TRACKER_CSV
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


# =============================================================================
# PATH DETECTION — unified with mousebrain.config
# =============================================================================

try:
    from mousebrain.config import (
        SLICES_2D_DIR, SLICES_2D_SUBJECTS, SLICES_2D_DATA_SUMMARY,
        SLICES_2D_TRACKER_CSV, PIPELINE_ROOT, TISSUE_ROOT,
    )
    DATA_DIR = SLICES_2D_SUBJECTS  # samples live here now
    TRACKER_CSV = SLICES_2D_TRACKER_CSV
    MODELS_DIR = TISSUE_ROOT / "MouseBrain" / "models" if TISSUE_ROOT else Path("models")
except ImportError:
    # Standalone fallback (brainslice installed without mousebrain)
    SLICES_2D_DIR = None
    SLICES_2D_SUBJECTS = None
    SLICES_2D_DATA_SUMMARY = None
    SLICES_2D_TRACKER_CSV = None
    PIPELINE_ROOT = None
    TISSUE_ROOT = None

    # Legacy path detection
    _env_root = os.environ.get("BRAINSLICE_ROOT")
    if _env_root:
        _root = Path(_env_root)
    else:
        _root = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = _root / "BrainSlice_Data"
    TRACKER_CSV = DATA_DIR / "calibration_runs.csv"
    MODELS_DIR = _root / "models"

# Legacy alias
BRAINSLICE_ROOT = SLICES_2D_DIR or Path(__file__).resolve().parent.parent.parent


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
    Get the data directory for a specific sample, organized by project/subject.

    For a sample like "E02_01_S12_DCN", resolves to:
        DATA_DIR / ENCR / ENCR_02_01 /

    Args:
        sample_id: Sample identifier (filename stem)

    Returns:
        Path to subject's data directory
    """
    parsed = parse_sample_name(sample_id)
    project = parsed.get("project", "UNKNOWN")
    cohort = parsed.get("cohort", "00")
    subject = parsed.get("subject", "00")

    if project and cohort and subject:
        subject_id = f"{project}_{cohort}_{subject}"
        return DATA_DIR / project / subject_id

    # Fallback for unparseable names
    return DATA_DIR / sample_id


def get_sample_subdir(sample_id: str, subdir: str) -> Path:
    """
    Get a specific subdirectory for a sample.

    Args:
        sample_id: Sample identifier
        subdir: Subdirectory name (e.g., "0_Raw", "3_Detected")

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
    RAW_ATLAS = "0_Raw_Atlas"
    RAW_HD = "0_Raw_HD"
    PREPROCESSED = "1_Preprocessed"
    REGISTERED = "2_Registered"
    DETECTED = "3_Detected"
    QUANTIFIED = "4_Quantified"
    QUANTIFIED_CORRECTED = "4_Quantified_Corrected"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_paths() -> Dict[str, bool]:
    """
    Validate that required paths exist.

    Returns:
        Dict mapping path names to existence status
    """
    results = {}
    if DATA_DIR:
        results["DATA_DIR"] = DATA_DIR.exists()
    if TRACKER_CSV:
        results["TRACKER_CSV"] = TRACKER_CSV.exists()
    if SLICES_2D_DIR:
        results["SLICES_2D_DIR"] = SLICES_2D_DIR.exists()
    return results


# =============================================================================
# DEBUG INFO
# =============================================================================

if __name__ == "__main__":
    print("BrainSlice 2D Configuration")
    print("=" * 50)
    print(f"SLICES_2D_DIR:    {SLICES_2D_DIR}")
    print(f"DATA_DIR:         {DATA_DIR}")
    print(f"TRACKER_CSV:      {TRACKER_CSV}")
    print(f"MODELS_DIR:       {MODELS_DIR}")
    print()
    print("Path validation:")
    for name, exists in validate_paths().items():
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")
