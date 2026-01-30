#!/usr/bin/env python3
"""
Temporary script to update hardcoded paths to use config module.
Run once and delete.
"""

import re
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

# Files to update and their replacements
updates = {
    # Napari widgets
    "sci_connectome_napari/sci_connectome_napari/pipeline_widget.py": {
        "old": """# Paths - default locations
BRAINS_ROOT = Path(r"Y:/2_Connectome/3_Nuclei_Detection/1_Brains")
SCRIPTS_DIR = Path(r"Y:/2_Connectome/3_Nuclei_Detection/util_Scripts")""",
        "new": """# Import paths from central config (auto-detects repo location)
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))
from config import BRAINS_ROOT, SCRIPTS_DIR"""
    },
    "sci_connectome_napari/sci_connectome_napari/tuning_widget.py": {
        "old": """# Paths - default locations
BRAINS_ROOT = Path(r"Y:/2_Connectome/3_Nuclei_Detection/1_Brains")
SCRIPTS_DIR = Path(r"Y:/2_Connectome/3_Nuclei_Detection/util_Scripts")""",
        "new": """# Import paths from central config (auto-detects repo location)
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))
from config import BRAINS_ROOT, SCRIPTS_DIR, MODELS_DIR"""
    },
    "sci_connectome_napari/sci_connectome_napari/experiments_widget.py": {
        "old": """# Paths - default locations
BRAINS_ROOT = Path(r"Y:/2_Connectome/3_Nuclei_Detection/1_Brains")
SCRIPTS_DIR = Path(r"Y:/2_Connectome/3_Nuclei_Detection/util_Scripts")

# Try to import experiment tracker
try:
    import sys
    sys.path.insert(0, str(SCRIPTS_DIR))
    from experiment_tracker import ExperimentTracker, EXP_TYPES
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    EXP_TYPES = ["detection", "training", "classification", "counts"]""",
        "new": """# Import paths from central config (auto-detects repo location)
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))
from config import BRAINS_ROOT, SCRIPTS_DIR

# Try to import experiment tracker
try:
    from experiment_tracker import ExperimentTracker, EXP_TYPES
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    EXP_TYPES = ["detection", "training", "classification", "counts"]"""
    },
    "sci_connectome_napari/sci_connectome_napari/manual_crop_widget.py": {
        "old": """# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\\2_Connectome\\3_Nuclei_Detection\\1_Brains")""",
        "new": """# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from central config (auto-detects repo location)
import sys as _sys
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in _sys.path:
    _sys.path.insert(0, str(_config_dir))
from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT"""
    },
}

# Also update hardcoded models path in tuning_widget.py
tuning_models_update = {
    "sci_connectome_napari/sci_connectome_napari/tuning_widget.py": {
        "old": 'models_dir = Path(r"Y:\\2_Connectome\\3_Nuclei_Detection\\util_Brainglobe\\Trained_Models")',
        "new": 'models_dir = MODELS_DIR'
    }
}

def update_file(filepath, old_text, new_text):
    """Update a file by replacing old_text with new_text."""
    full_path = SCRIPTS_DIR / filepath
    if not full_path.exists():
        print(f"  SKIP: {filepath} not found")
        return False

    content = full_path.read_text(encoding='utf-8')

    if old_text not in content:
        # Try with different line endings
        old_text_crlf = old_text.replace('\n', '\r\n')
        if old_text_crlf in content:
            old_text = old_text_crlf
        else:
            print(f"  SKIP: Pattern not found in {filepath}")
            return False

    new_content = content.replace(old_text, new_text)
    full_path.write_text(new_content, encoding='utf-8')
    print(f"  OK: {filepath}")
    return True


def main():
    print("Updating files to use config module...\n")

    # Main updates
    for filepath, change in updates.items():
        update_file(filepath, change["old"], change["new"])

    # Additional tuning widget update
    for filepath, change in tuning_models_update.items():
        update_file(filepath, change["old"], change["new"])

    print("\nDone! You can delete this script (update_paths.py)")


if __name__ == "__main__":
    main()
