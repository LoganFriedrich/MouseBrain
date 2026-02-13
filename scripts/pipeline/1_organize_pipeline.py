#!/usr/bin/env python3
r"""
1_organize_pipeline.py (v1.0.0)

================================================================================
WHAT IS THIS?
================================================================================
This is Script 1 in the BrainGlobe pipeline. It's the "get organized" script
that sets up folder structures and moves files into place. It doesn't do any
actual image processing - just housekeeping.

Think of it as: "Make sure everything is in the right place before we start"

Run this BEFORE Script 2 (extract_and_analyze.py).

================================================================================
WHAT IT DOES
================================================================================
1. Scans for .ims files in your brains directory
2. Creates the pipeline folder structure for each brain/magnification
3. Moves IMS files into their proper 0_Raw_IMS folders
4. Migrates any old folder structures from previous script versions
5. Destroys thumbs.db files (they break napari drag-drop)
6. Reports what it found and what needs processing

================================================================================
HOW TO RUN
================================================================================
Open Anaconda Prompt, then:

    conda activate brainglobe-env
    cd <your-repo-path>/3_Nuclei_Detection/util_Scripts
    python 1_organize_pipeline.py

Options:
    python 1_organize_pipeline.py              # Normal - organize all
    python 1_organize_pipeline.py --inspect    # Dry run - just show status
    python 1_organize_pipeline.py "C:\path"    # Scan specific folder

================================================================================
FOLDER STRUCTURE CREATED
================================================================================
Before running:
    1_Brains/
    └── 349_CNT_01_02/
        └── 349_CNT_01_02_1.625x_z4.ims     ← Sitting in mouse folder

After running:
    1_Brains/
    └── 349_CNT_01_02/
        └── 349_CNT_01_02_1p625x_z4/        ← Pipeline folder (. → p in name)
            ├── 0_Raw_IMS/
            │   └── 349_CNT_01_02_1.625x_z4.ims   ← Moved here
            ├── 1_Extracted_Full/            ← Script 2 output (full)
            ├── 2_Cropped_For_Registration/  ← Script 2 output (cropped)
            ├── 3_Registered_Atlas/          ← Script 3 output
            ├── 4_Cell_Candidates/           ← Script 4 output
            ├── 5_Classified_Cells/          ← Script 5 output
            └── 6_Region_Analysis/           ← Script 6 output

Note: Decimals become 'p' in folder names (1.625x → 1p625x) because napari
hates dots in folder paths.

================================================================================
FILENAME CONVENTION (IMPORTANT!)
================================================================================
Your .ims files MUST be named like this:

    NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims

Examples:
    349_CNT_01_02_1.625x_z4.ims
    350_SCI_02_05_1.9x_z3.37.ims

What each part means:
    349         = Brain/sample number
    CNT         = Project code (CNT, SCI, etc.)
    01          = Cohort number
    02          = Animal number within cohort
    1.625x      = Magnification (used to calculate voxel size!)
    z4          = Z-step in microns (your voxel Z size!)

Files that don't match this pattern will be skipped with a warning.

================================================================================
REQUIREMENTS
================================================================================
Just Python standard library - no special packages needed for this script!
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# VERSION
# =============================================================================
SCRIPT_VERSION = "1.0.1"

# =============================================================================
# PROGRESS HELPERS
# =============================================================================
def timestamp():
    """Get current time as formatted string."""
    return datetime.now().strftime("%H:%M:%S")

# =============================================================================
# DEFAULT PATHS - edit for your setup
# =============================================================================
from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT

# =============================================================================
# PIPELINE FOLDER NAMES
# =============================================================================
FOLDER_RAW_IMS = "0_Raw_IMS"
FOLDER_EXTRACTED_FULL = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"
FOLDER_CLASSIFICATION = "5_Classified_Cells"
FOLDER_ANALYSIS = "6_Region_Analysis"

ALL_PIPELINE_FOLDERS = [
    FOLDER_RAW_IMS,
    FOLDER_EXTRACTED_FULL,
    FOLDER_CROPPED,
    FOLDER_REGISTRATION,
    FOLDER_DETECTION,
    FOLDER_CLASSIFICATION,
    FOLDER_ANALYSIS,
]

# Old folder names from previous versions (for migration)
OLD_FOLDER_NAMES = {
    "0_Raw_IMS_From_Miami": FOLDER_RAW_IMS,
    "1_Extracted_Channels_from_1_ims_to_brainglobe": FOLDER_EXTRACTED_FULL,
    "1_Channels": FOLDER_EXTRACTED_FULL,
    "2_Registered_Atlas_from_brainreg": FOLDER_REGISTRATION,
    "2_Registration": FOLDER_REGISTRATION,
    "3_Detected_Cell_Candidates_from_cellfinder": FOLDER_DETECTION,
    "3_Detection": FOLDER_DETECTION,
    "4_Classified_Cells_from_cellfinder": FOLDER_CLASSIFICATION,
    "5_Cell_Counts_by_Region_from_brainglobe_segmentation": FOLDER_ANALYSIS,
    "4_Analysis": FOLDER_ANALYSIS,
}


# =============================================================================
# THUMBS.DB ANNIHILATION
# =============================================================================

def annihilate_thumbs_db(path, recursive=True):
    """
    Destroy all thumbs.db files. Nuke them from orbit.
    It's the only way to be sure.
    
    Skips 0-byte files (these are our prevention decoys).
    
    Returns count of files destroyed.
    """
    path = Path(path)
    destroyed = 0
    
    def try_destroy(thumbs):
        """Attempt to destroy a single thumbs.db file."""
        nonlocal destroyed
        try:
            # Skip 0-byte files - these are our decoys
            if thumbs.stat().st_size == 0:
                return
            
            # Remove hidden/system attributes on Windows
            if sys.platform == 'win32':
                import subprocess
                subprocess.run(['attrib', '-h', '-s', '-r', str(thumbs)], 
                               capture_output=True, check=False)
            thumbs.unlink()
            destroyed += 1
        except Exception as e:
            print(f"    Warning: Could not destroy {thumbs}: {e}")
    
    if recursive:
        for pattern in ["Thumbs.db", "thumbs.db", "THUMBS.DB"]:
            for thumbs in path.rglob(pattern):
                try_destroy(thumbs)
    else:
        for name in ["Thumbs.db", "thumbs.db", "THUMBS.DB"]:
            thumbs = path / name
            if thumbs.exists():
                try_destroy(thumbs)
    
    return destroyed


def create_thumbs_decoy(folder):
    """
    Create a read-only, hidden decoy Thumbs.db file.
    Windows won't overwrite a read-only file, preventing thumbs.db creation.
    """
    folder = Path(folder)
    decoy = folder / "Thumbs.db"
    
    if decoy.exists():
        return False
    
    try:
        decoy.write_bytes(b'')
        if sys.platform == 'win32':
            import subprocess
            subprocess.run(['attrib', '+h', '+r', str(decoy)], 
                           capture_output=True, check=False)
        else:
            decoy.chmod(0o444)
        return True
    except:
        return False


# =============================================================================
# FILENAME PARSING
# =============================================================================

def decimals_to_p(s):
    """Convert decimals to 'p' in a string (for folder names)."""
    return s.replace('.', 'p')


def p_to_decimals(s):
    """Convert 'p' back to decimals (for parsing folder names)."""
    return re.sub(r'(\d)p(\d)', r'\1.\2', s)


def parse_filename(filename):
    """
    Parse an IMS filename to extract components.
    
    Expected format: NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims
    Example: 349_CNT_01_02_1.625x_z4.ims
    
    Returns dict with parsed values or None if invalid.
    """
    stem = Path(filename).stem
    
    # Pattern: digits_letters_digits_digits_decimals+x_z+decimals
    pattern = r'^(\d+)_([A-Za-z]+)_(\d+)_(\d+)_(\d+\.?\d*)x_z(\d+\.?\d*)$'
    match = re.match(pattern, stem)
    
    if not match:
        return None
    
    return {
        'number': match.group(1),
        'project': match.group(2),
        'cohort': match.group(3),
        'animal': match.group(4),
        'magnification': float(match.group(5)),
        'z_step': float(match.group(6)),
        'stem': stem,
        'mouse_id': f"{match.group(1)}_{match.group(2)}_{match.group(3)}_{match.group(4)}",
        'mag_str': f"{match.group(5)}x_z{match.group(6)}",
        'pipeline_folder': decimals_to_p(stem),  # Full name with p notation
    }


def validate_filename(filename):
    """
    Validate that a filename matches expected convention.
    Returns (is_valid, reason).
    """
    parsed = parse_filename(filename)
    if parsed is None:
        return False, "doesn't match pattern NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims"
    return True, "valid"


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_pipeline_folder(mouse_folder, pipeline_name):
    """Get path to pipeline folder for a specific magnification."""
    return Path(mouse_folder) / pipeline_name


def get_raw_ims_folder(mouse_folder, pipeline_name):
    """Get path to 0_Raw_IMS folder inside pipeline."""
    return get_pipeline_folder(mouse_folder, pipeline_name) / FOLDER_RAW_IMS


def create_pipeline_structure(mouse_folder, pipeline_name):
    """
    Create the full pipeline folder structure.
    Returns the pipeline folder path.
    """
    pipeline = get_pipeline_folder(mouse_folder, pipeline_name)
    
    for subfolder in ALL_PIPELINE_FOLDERS:
        folder = pipeline / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        create_thumbs_decoy(folder)
    
    return pipeline


# =============================================================================
# IMS FILE DISCOVERY
# =============================================================================

def find_ims_files(root_path):
    """
    Find all IMS files in the directory structure.
    
    Looks in:
    - Direct children of root (mouse folders)
    - {mouse}/0_Original/ (old v3.0.x structure)
    - {mouse}/{pipeline}/0_Raw_IMS*/ (various versions)
    """
    root_path = Path(root_path)
    ims_files = []
    
    for subdir in root_path.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        
        # Skip obvious non-mouse folders
        if any(skip in subdir.name.lower() for skip in ['script', 'backup', 'archive', 'old']):
            continue
        
        # Check directly in mouse folder
        for ims_file in subdir.glob('*.ims'):
            ims_files.append(ims_file)
        
        # Check old 0_Original at mouse level
        old_original = subdir / "0_Original"
        if old_original.exists():
            for ims_file in old_original.glob('*.ims'):
                ims_files.append(ims_file)
        
        # Check pipeline subfolders for various raw folder names
        for pipeline_folder in subdir.iterdir():
            if pipeline_folder.is_dir():
                # Check all possible raw folder names
                for raw_name in [FOLDER_RAW_IMS, "0_Raw_IMS_From_Miami"]:
                    raw_folder = pipeline_folder / raw_name
                    if raw_folder.exists():
                        for ims_file in raw_folder.glob('*.ims'):
                            ims_files.append(ims_file)
    
    # Also check root directly (unlikely but handle it)
    for ims_file in root_path.glob('*.ims'):
        ims_files.append(ims_file)
    
    # Remove duplicates and sort
    return sorted(set(ims_files))


# =============================================================================
# STATUS CHECKING
# =============================================================================

def get_ims_status(ims_path):
    """
    Determine the status of an IMS file.
    
    Returns:
        (status, reason, details)
        
        Status:
        - "invalid": Filename doesn't match expected format
        - "needs_organization": File needs to be moved/folders created
        - "organized": File is in correct location with folder structure
    """
    ims_path = Path(ims_path)
    
    # Check filename validity
    is_valid, validation_reason = validate_filename(ims_path.name)
    if not is_valid:
        return "invalid", validation_reason, {}
    
    parsed = parse_filename(ims_path.name)
    pipeline_name = parsed['pipeline_folder']
    
    # Figure out where IMS currently is and where it should be
    current_parent = ims_path.parent.name
    
    # Is it in the correct 0_Raw_IMS folder?
    if current_parent == FOLDER_RAW_IMS:
        # Check if pipeline folder name matches
        pipeline_folder = ims_path.parent.parent
        if pipeline_folder.name == pipeline_name:
            # Check all subfolders exist
            all_exist = all(
                (pipeline_folder / subfolder).exists() 
                for subfolder in ALL_PIPELINE_FOLDERS
            )
            if all_exist:
                return "organized", "ready for processing", parsed
            else:
                return "needs_organization", "missing pipeline subfolders", parsed
        else:
            return "needs_organization", f"wrong pipeline folder (in {pipeline_folder.name})", parsed
    
    # Is it in an old-style raw folder?
    elif current_parent in ["0_Raw_IMS_From_Miami", "0_Original"]:
        return "needs_organization", f"old folder structure ({current_parent})", parsed
    
    # Is it directly in a mouse folder?
    else:
        return "needs_organization", "not yet organized", parsed


def get_mouse_folder_for_ims(ims_path):
    """
    Determine the mouse folder for an IMS file based on its current location.
    """
    ims_path = Path(ims_path)
    parent = ims_path.parent
    
    # If in a raw folder, go up two levels
    if parent.name in [FOLDER_RAW_IMS, "0_Raw_IMS_From_Miami", "0_Original"]:
        if parent.name == "0_Original":
            return parent.parent  # 0_Original is at mouse level
        else:
            return parent.parent.parent  # 0_Raw_IMS is inside pipeline
    
    # Otherwise assume it's directly in mouse folder
    return parent


# =============================================================================
# ORGANIZATION / MIGRATION
# =============================================================================

def organize_ims_file(ims_path, dry_run=False):
    """
    Move an IMS file to its proper location and create folder structure.
    
    Returns (success, message, actions_taken).
    """
    ims_path = Path(ims_path)
    
    if not ims_path.exists():
        return False, f"file not found: {ims_path}", []
    
    parsed = parse_filename(ims_path.name)
    pipeline_name = parsed['pipeline_folder']
    
    mouse_folder = get_mouse_folder_for_ims(ims_path)
    actions = []
    
    # Create pipeline structure
    pipeline = get_pipeline_folder(mouse_folder, pipeline_name)
    raw_folder = get_raw_ims_folder(mouse_folder, pipeline_name)
    target_path = raw_folder / ims_path.name
    
    # Check if already in correct location
    if ims_path == target_path or ims_path.resolve() == target_path.resolve():
        # Already in correct place - just ensure folder structure exists
        if not dry_run:
            create_pipeline_structure(mouse_folder, pipeline_name)
        return True, "already organized", ["Already in correct location"]
    
    if not dry_run:
        create_pipeline_structure(mouse_folder, pipeline_name)
        actions.append(f"Created pipeline structure: {pipeline_name}/")
    else:
        actions.append(f"Would create pipeline structure: {pipeline_name}/")
    
    # Move IMS file if needed
    if not dry_run:
        # Ensure target directory exists
        raw_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ims_path), str(target_path))
        actions.append(f"Moved IMS to {pipeline_name}/{FOLDER_RAW_IMS}/")
        
        # Clean up old empty folders
        old_parent = ims_path.parent
        if old_parent.name in ["0_Original", "0_Raw_IMS_From_Miami"]:
            try:
                if not any(old_parent.iterdir()):
                    old_parent.rmdir()
                    actions.append(f"Removed empty {old_parent.name}/")
            except:
                pass
    else:
        actions.append(f"Would move IMS to {pipeline_name}/{FOLDER_RAW_IMS}/")
    
    return True, "organized", actions


def migrate_old_folders(mouse_folder, pipeline_name, dry_run=False):
    """
    Rename old folder names to new standardized names.
    
    Returns list of actions taken.
    """
    pipeline = get_pipeline_folder(mouse_folder, pipeline_name)
    actions = []
    
    if not pipeline.exists():
        return actions
    
    for old_name, new_name in OLD_FOLDER_NAMES.items():
        if old_name == new_name:
            continue
        
        old_folder = pipeline / old_name
        new_folder = pipeline / new_name
        
        if old_folder.exists() and old_folder.is_dir():
            if new_folder.exists():
                # Merge if new exists but is empty
                if not any(f for f in new_folder.iterdir() 
                          if f.name not in ['Thumbs.db', 'desktop.ini']):
                    if not dry_run:
                        shutil.rmtree(str(new_folder))
                        shutil.move(str(old_folder), str(new_folder))
                    actions.append(f"Renamed {old_name}/ → {new_name}/")
                else:
                    # Both have content - try to merge
                    if not dry_run:
                        for item in old_folder.iterdir():
                            if item.name in ['Thumbs.db', 'desktop.ini']:
                                continue
                            dest = new_folder / item.name
                            if not dest.exists():
                                shutil.move(str(item), str(dest))
                        # Remove old if empty
                        remaining = [f for f in old_folder.iterdir() 
                                    if f.name not in ['Thumbs.db', 'desktop.ini']]
                        if not remaining:
                            shutil.rmtree(str(old_folder))
                    actions.append(f"Merged {old_name}/ into {new_name}/")
            else:
                if not dry_run:
                    shutil.move(str(old_folder), str(new_folder))
                actions.append(f"Renamed {old_name}/ → {new_name}/")
    
    # Also check for old short-name pipeline folders (e.g., 1p625x_z4 instead of full name)
    mag_only = decimals_to_p(parse_filename(p_to_decimals(pipeline_name) + ".ims")['mag_str']) \
               if '_' in pipeline_name else None
    
    if mag_only and mag_only != pipeline_name:
        old_short = mouse_folder / mag_only
        if old_short.exists() and old_short.is_dir():
            if pipeline.exists():
                # Merge contents
                if not dry_run:
                    for item in old_short.iterdir():
                        dest = pipeline / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                    shutil.rmtree(str(old_short))
                actions.append(f"Merged old short-name {mag_only}/ into {pipeline_name}/")
            else:
                if not dry_run:
                    shutil.move(str(old_short), str(pipeline))
                actions.append(f"Renamed {mag_only}/ → {pipeline_name}/")
    
    return actions


# =============================================================================
# SCANNING AND REPORTING
# =============================================================================

def scan_and_report(root_path):
    """
    Scan for IMS files and report their status.
    """
    ims_files = find_ims_files(root_path)
    
    results = {
        'organized': [],
        'needs_organization': [],
        'invalid': [],
    }
    
    for ims_path in ims_files:
        status, reason, details = get_ims_status(ims_path)
        results[status].append((ims_path, reason, details))
    
    return results


def print_scan_results(results):
    """Print a formatted summary of scan results."""
    total = sum(len(v) for v in results.values())
    
    print(f"\nFound {total} .ims file(s):\n")
    
    # Organized (ready)
    for ims_path, reason, details in results['organized']:
        pipeline = details.get('pipeline_folder', '?')
        print(f"  ✓ {ims_path.parent.parent.parent.name}/{pipeline}/ - {reason}")
    
    # Needs organization
    for ims_path, reason, details in results['needs_organization']:
        rel_path = f"{ims_path.parent.name}/{ims_path.name}"
        print(f"  ○ {rel_path} - {reason}")
    
    # Invalid
    for ims_path, reason, details in results['invalid']:
        print(f"  ✗ {ims_path.name}")
        print(f"      SKIP: {reason}")
    
    if results['invalid']:
        print(f"\n  ⚠ {len(results['invalid'])} file(s) skipped - rename to: NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims")
    
    return len(results['needs_organization'])


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Organize BrainGlobe pipeline folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {SCRIPT_VERSION}

This script organizes your folder structure. Run it before Script 2.

Folder Structure Created:
  MouseFolder/
  └── 349_CNT_01_02_1p625x_z4/
      ├── 0_Raw_IMS/
      ├── 1_Extracted_Full/
      ├── 2_Cropped_For_Registration/
      ├── 3_Registered_Atlas/
      ├── 4_Cell_Candidates/
      ├── 5_Classified_Cells/
      └── 6_Region_Analysis/

Examples:
  python 1_organize_pipeline.py
  python 1_organize_pipeline.py --inspect
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help=f'Path to scan (default: {DEFAULT_BRAINGLOBE_ROOT})')
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Dry run - show what would happen without making changes')
    
    args = parser.parse_args()
    
    # Determine root path
    root_path = Path(args.path) if args.path else DEFAULT_BRAINGLOBE_ROOT
    
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        print(f"\nEdit DEFAULT_BRAINGLOBE_ROOT in the script or provide a path as argument.")
        sys.exit(1)
    
    print("=" * 60)
    print("BrainGlobe Pipeline Organizer")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Initial thumbs.db sweep
    print(f"\n[{timestamp()}] Scanning: {root_path}")
    print(f"[{timestamp()}] Sweeping for thumbs.db files...")
    destroyed = annihilate_thumbs_db(root_path)
    if destroyed > 0:
        print(f"  Destroyed {destroyed} thumbs.db file(s)")
    
    # Scan and report
    results = scan_and_report(root_path)
    n_needs_work = print_scan_results(results)
    
    if n_needs_work == 0:
        print("\n✓ All files are organized. Ready for Script 2!")
        return
    
    if args.inspect:
        print(f"\n{n_needs_work} file(s) would be organized.")
        print("Run without --inspect to make changes.")
        return
    
    # Confirm
    response = input(f"\nOrganize {n_needs_work} file(s)? [Enter to continue, 'q' to quit]: ").strip()
    if response.lower() == 'q':
        print("Cancelled.")
        return
    
    print("\n" + "=" * 60)
    print("Organizing...")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for ims_path, reason, details in results['needs_organization']:
        parsed = parse_filename(ims_path.name)
        pipeline_name = parsed['pipeline_folder']
        mouse_folder = get_mouse_folder_for_ims(ims_path)
        
        print(f"\n  {ims_path.name}")
        
        try:
            # Migrate any old folder names first
            migrate_actions = migrate_old_folders(mouse_folder, pipeline_name)
            for action in migrate_actions:
                print(f"    {action}")
            
            # After migration, the IMS file may have moved - recalculate its path
            # The file should now be in the new 0_Raw_IMS folder if migration happened
            new_raw_folder = get_raw_ims_folder(mouse_folder, pipeline_name)
            potential_new_path = new_raw_folder / ims_path.name
            
            if potential_new_path.exists():
                ims_path = potential_new_path
            elif not ims_path.exists():
                # File moved somewhere else during migration, try to find it
                # Check if it's directly in the pipeline folder
                pipeline_folder = get_pipeline_folder(mouse_folder, pipeline_name)
                for possible_raw in [FOLDER_RAW_IMS, "0_Raw_IMS_From_Miami"]:
                    check_path = pipeline_folder / possible_raw / ims_path.name
                    if check_path.exists():
                        ims_path = check_path
                        break
            
            # Now organize the IMS file (move to correct location if needed)
            ok, msg, actions = organize_ims_file(ims_path)
            for action in actions:
                print(f"    {action}")
            
            if ok:
                success += 1
                print(f"    ✓ Organized")
            else:
                failed += 1
                print(f"    ✗ {msg}")
        
        except Exception as e:
            failed += 1
            print(f"    ✗ Error: {e}")
    
    # Final thumbs.db sweep
    print("\nFinal thumbs.db sweep...")
    destroyed = annihilate_thumbs_db(root_path)
    if destroyed > 0:
        print(f"  Destroyed {destroyed} additional thumbs.db file(s)")
    
    print("\n" + "=" * 60)
    print(f"Complete: {success} organized, {failed} failed")
    print("=" * 60)
    
    if success > 0 and failed == 0:
        print("\n✓ Ready for Script 2 (extract_and_analyze.py)!")


if __name__ == '__main__':
    main()
    
    # If double-clicked, pause
    if len(sys.argv) == 1:
        print()
        input("Press Enter to close...")
