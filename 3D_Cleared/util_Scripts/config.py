#!/usr/bin/env python3
"""
config.py - Backward-compatibility shim.

The canonical version of this module now lives in the mousebrain package:
    from mousebrain.config import BRAINS_ROOT, SCRIPTS_DIR, ...

This shim re-exports everything so existing scripts that do
    from config import BRAINS_ROOT
continue to work without changes.
"""

# Try mousebrain package first (preferred)
try:
    from mousebrain.config import *
    from mousebrain.config import _find_repo_root, _get_root_path, PROJECT_CODES
except ImportError:
    # Fallback: if braintools is not installed, use inline version
    # This keeps standalone scripts working even without pip install
    import os
    import sys
    from pathlib import Path
    from typing import Optional
    import warnings

    def _find_repo_root() -> Optional[Path]:
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if current.name == "util_Scripts":
                parent = current.parent
                if parent.name == "3D_Cleared":
                    grandparent = parent.parent
                    # New restructured: Tissue/MouseBrain_Pipeline/3D_Cleared/util_Scripts
                    if grandparent.name == "MouseBrain_Pipeline":
                        tissue_dir = grandparent.parent
                        if tissue_dir.name == "Tissue":
                            return tissue_dir.parent
                    # Current: Tissue/3D_Cleared/util_Scripts
                    if grandparent.name == "Tissue":
                        return grandparent.parent
                if parent.name == "3_Nuclei_Detection":
                    return parent.parent
            # New restructured: Tissue/MouseBrain_Pipeline/3D_Cleared
            new_cleared = current / "Tissue" / "MouseBrain_Pipeline" / "3D_Cleared"
            if new_cleared.exists() and new_cleared.is_dir():
                if (new_cleared / "util_Scripts").exists():
                    return current
            # Current: Tissue/3D_Cleared
            cleared_3d = current / "Tissue" / "3D_Cleared"
            if cleared_3d.exists() and cleared_3d.is_dir():
                if (cleared_3d / "util_Scripts").exists():
                    return current
            # Legacy: 3_Nuclei_Detection
            nuclei_detection = current / "3_Nuclei_Detection"
            if nuclei_detection.exists() and nuclei_detection.is_dir():
                if (nuclei_detection / "util_Scripts").exists():
                    return current
            if current.parent == current:
                break
            current = current.parent
        return None

    def _get_root_path() -> Path:
        env_root = os.environ.get("SCI_CONNECTOME_ROOT")
        if env_root:
            env_path = Path(env_root)
            if env_path.exists():
                return env_path
        y_drive = Path(r"Y:\2_Connectome")
        if y_drive.exists():
            if ((y_drive / "Tissue" / "MouseBrain_Pipeline" / "3D_Cleared").exists()
                    or (y_drive / "Tissue" / "3D_Cleared").exists()):
                return y_drive
        detected = _find_repo_root()
        if detected and detected.exists():
            return detected
        return y_drive

    ROOT_PATH = _get_root_path()
    TISSUE_ROOT = ROOT_PATH / "Tissue"

    # Tool package location
    if (TISSUE_ROOT / "MouseBrain").exists():
        MOUSEBRAIN_ROOT = TISSUE_ROOT / "MouseBrain"
    else:
        MOUSEBRAIN_ROOT = TISSUE_ROOT / "mousebrain"

    # Pipeline data root
    if (TISSUE_ROOT / "MouseBrain_Pipeline").exists():
        PIPELINE_ROOT = TISSUE_ROOT / "MouseBrain_Pipeline"
    else:
        PIPELINE_ROOT = None

    # 3D Cleared dir - try restructured, then current, then legacy
    if PIPELINE_ROOT and (PIPELINE_ROOT / "3D_Cleared").exists():
        CLEARED_3D_DIR = PIPELINE_ROOT / "3D_Cleared"
    elif (TISSUE_ROOT / "3D_Cleared").exists():
        CLEARED_3D_DIR = TISSUE_ROOT / "3D_Cleared"
    else:
        CLEARED_3D_DIR = ROOT_PATH / "3_Nuclei_Detection"

    NUCLEI_DETECTION_DIR = CLEARED_3D_DIR
    BRAINS_ROOT = CLEARED_3D_DIR / "1_Brains"
    DATA_SUMMARY_DIR = CLEARED_3D_DIR / "2_Data_Summary"
    SCRIPTS_DIR = CLEARED_3D_DIR / "util_Scripts"
    MODELS_DIR = CLEARED_3D_DIR / "util_Brainglobe" / "Trained_Models"
    CALIBRATION_RUNS_CSV = DATA_SUMMARY_DIR / "calibration_runs.csv"
    EXPERIMENTS_CSV = CALIBRATION_RUNS_CSV
    REGION_COUNTS_CSV = DATA_SUMMARY_DIR / "region_counts.csv"
    REGION_COUNTS_ARCHIVE_CSV = DATA_SUMMARY_DIR / "region_counts_archive.csv"
    ELIFE_COUNTS_CSV = DATA_SUMMARY_DIR / "elife_counts.csv"
    ELIFE_COUNTS_ARCHIVE_CSV = DATA_SUMMARY_DIR / "elife_counts_archive.csv"
    DEFAULT_BRAINGLOBE_ROOT = BRAINS_ROOT
    DEFAULT_MODELS_DIR = MODELS_DIR
    DEFAULT_TRACKER_PATH = EXPERIMENTS_CSV
    SUMMARY_DATA_DIR = DATA_SUMMARY_DIR

    PROJECT_CODES = {"CNT": "Connectome", "ENCR": "Enhancer"}

    def parse_brain_name(brain_name: str) -> dict:
        result = {
            'brain_id': None, 'project_code': None, 'project_name': None,
            'cohort': None, 'subject': None, 'subject_full': None,
            'cohort_full': None, 'imaging_params': None, 'raw': brain_name
        }
        if not brain_name:
            return result
        parts = brain_name.split('_')
        if len(parts) < 4:
            return result
        result['brain_id'] = parts[0]
        result['project_code'] = parts[1]
        result['project_name'] = PROJECT_CODES.get(parts[1], parts[1])
        result['cohort'] = parts[2]
        result['subject'] = parts[3]
        result['cohort_full'] = f"{parts[1]}_{parts[2]}"
        result['subject_full'] = f"{parts[1]}_{parts[2]}_{parts[3]}"
        if len(parts) > 4:
            result['imaging_params'] = '_'.join(parts[4:])
        return result

    def format_subject_id(project, cohort, subject):
        return f"{project}_{cohort}_{subject}"

    def format_cohort_id(project, cohort):
        return f"{project}_{cohort}"

    def get_brain_hierarchy(brain_name):
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

    def validate_paths(verbose=False):
        paths_to_check = {
            "ROOT_PATH": ROOT_PATH, "NUCLEI_DETECTION_DIR": NUCLEI_DETECTION_DIR,
            "BRAINS_ROOT": BRAINS_ROOT, "SCRIPTS_DIR": SCRIPTS_DIR,
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
        print("=" * 60)
        print("SCI-Connectome Configuration (FALLBACK - braintools not installed)")
        print("=" * 60)
        print(f"\nDetected/configured paths:")
        print(f"  ROOT_PATH:            {ROOT_PATH}")
        print(f"  BRAINS_ROOT:          {BRAINS_ROOT}")
        print(f"  DATA_SUMMARY_DIR:     {DATA_SUMMARY_DIR}")
        print(f"  SCRIPTS_DIR:          {SCRIPTS_DIR}")
        print(f"\nPath validation:")
        validate_paths(verbose=True)
        print("=" * 60)

if __name__ == "__main__":
    print_config()
