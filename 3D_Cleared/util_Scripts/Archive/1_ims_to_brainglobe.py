#!/usr/bin/env python3
"""
1_ims_to_brainglobe.py (v3.2.0)

================================================================================
WHAT IS THIS?
================================================================================
This is Script 1 in the BrainGlobe pipeline. It converts Imaris .ims files 
(from the LaVision UltraMicroscope lightsheet) into the TIFF slice format that 
BrainGlobe tools (brainreg, cellfinder) need.

Think of it as: "Get my microscope data ready for brain analysis"

================================================================================
PREREQUISITES
================================================================================
1. Your .ims files must be named correctly (see FILENAME FORMAT below)
2. You need the brainglobe conda environment
3. Files should be in mouse folders inside the 1_Brains directory

================================================================================
HOW TO RUN
================================================================================
Open Anaconda Prompt, then:

    conda activate brainglobe-env
    cd Y:\2_Connectome\3_Nuclei_Detection\BrainGlobe\0_Scripts
    python 1_ims_to_brainglobe.py

The script will:
1. Scan for .ims files
2. Show you what it found
3. Ask for confirmation
4. Extract all channels to TIFF slices
5. Create the pipeline folder structure

================================================================================
FILENAME FORMAT (IMPORTANT!)
================================================================================
Your .ims files MUST be named like this:

    NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims

Examples:
    349_CNT_01_02_1.625x_z4.ims
    350_SCI_02_05_1.9x_z3.37.ims

What each part means:
    349         = Brain/sample number
    CNT         = Project code (CNT=control, SCI=spinal cord injury, etc.)
    01          = Cohort number
    02          = Animal number within cohort
    1.625x      = Magnification (CRITICAL - used to calculate voxel size!)
    z4          = Z-step in microns (CRITICAL - your voxel Z size!)

WHY THIS MATTERS: The script calculates your XY voxel size from the 
magnification (camera pixel size / magnification). If your filename is 
wrong, your voxel sizes will be wrong, and registration will fail!

================================================================================
FOLDER STRUCTURE CREATED
================================================================================
Before running:
    1_Brains/
    +-- 349_CNT_01_02/
        +-- 349_CNT_01_02_1.625x_z4.ims     <- Your file goes here

After running:
    1_Brains/
    +-- 349_CNT_01_02/                       <- Mouse folder
        +-- 349_CNT_01_02_1p625x_z4/         <- Pipeline folder (note: . -> p)
            +-- 0_Raw_IMS_From_Miami/        <- Original moved here
            |   +-- 349_CNT_01_02_1.625x_z4.ims
            +-- 1_Extracted_Channels_from_1_ims_to_brainglobe/  <- This script
            |   +-- ch0/
            |   |   +-- Z0000.tif
            |   |   +-- Z0001.tif
            |   |   +-- ... (one TIFF per Z-slice)
            |   +-- ch1/
            |   +-- metadata.json            <- Voxel sizes, dimensions, etc.
            +-- 2_Registered_Atlas_from_brainreg/        <- Script 2
            +-- 3_Detected_Cell_Candidates_from_cellfinder/  <- Script 3
            +-- 4_Classified_Cells_from_cellfinder/          <- Script 4
            +-- 5_Cell_Counts_by_Region_from_brainglobe_segmentation/  <- Script 5

Note: Decimals in folder names become 'p' (1.625x -> 1p625x) because napari
hates dots in folder names.

================================================================================
COMMAND LINE OPTIONS
================================================================================
    python 1_ims_to_brainglobe.py              # Normal mode - scan and process
    python 1_ims_to_brainglobe.py --inspect    # Dry run - just show what would happen
    python 1_ims_to_brainglobe.py --force      # Reprocess everything (even if done)
    python 1_ims_to_brainglobe.py "C:\path"   # Scan a specific folder

================================================================================
HOW LONG DOES IT TAKE?
================================================================================
Depends on file size. A 20GB .ims file takes roughly:
    - 1-2 minutes to load each channel
    - 1-2 minutes to save each channel as TIFFs
    - Total: ~5-10 minutes per channel

The script shows progress as it works.

================================================================================
WHAT IF SOMETHING GOES WRONG?
================================================================================
Common issues:

1. "doesn't match pattern" error
   -> Your filename isn't in the right format. Rename it!

2. Script seems to hang
   -> Large files take time. Watch for progress updates.

3. "Missing required packages" error
   -> Make sure you activated the brainglobe-env first

4. Napari can't drag-drop the output folder
   -> There's probably a thumbs.db file. The script tries to kill these,
      but Windows is persistent. Delete any thumbs.db files manually.

================================================================================
SETTINGS (edit in script if needed)
================================================================================
    DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")
    CAMERA_PIXEL_SIZE = 6.5      # Andor Neo/Zyla sCMOS pixel size in microns
    DEFAULT_ORIENTATION = "iar"  # inferior-anterior-right

================================================================================
REQUIREMENTS
================================================================================
    pip install imaris-ims-file-reader tifffile numpy h5py

================================================================================
"""

# =============================================================================
# CHECK REQUIREMENTS
# =============================================================================
import sys
import os

def _check_requirements():
    """Check if required packages are available."""
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import tifffile
    except ImportError:
        missing.append("tifffile")
    try:
        import h5py
    except ImportError:
        missing.append("h5py")
    
    if missing:
        print("=" * 60)
        print("ERROR: Missing required packages:", ", ".join(missing))
        print()
        print("Run this script from Anaconda Prompt:")
        print("    conda activate brainglobe-env")
        print("    python 1_ims_to_brainglobe.py")
        print()
        print("Or install missing packages:")
        print(f"    pip install {' '.join(missing)}")
        print("=" * 60)
        input("\nPress Enter to close...")
        sys.exit(1)

_check_requirements()

# =============================================================================
# MAIN SCRIPT
# =============================================================================

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# =============================================================================
# VERSION - increment this when making changes that affect output
# =============================================================================
SCRIPT_VERSION = "3.2.0"

# =============================================================================
# DEFAULT PATHS - edit these for your setup
# =============================================================================
DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Camera pixel size in microns (Andor Neo/Zyla sCMOS)
CAMERA_PIXEL_SIZE = 6.5

# Default orientation for brainreg (inferior-anterior-right)
DEFAULT_ORIENTATION = "iar"

# Pipeline folder names - named for the tool/script that generates each output
FOLDER_RAW_IMS = "0_Raw_IMS_From_Miami"
FOLDER_CHANNELS = "1_Extracted_Channels_from_1_ims_to_brainglobe"
FOLDER_REGISTRATION = "2_Registered_Atlas_from_brainreg"
FOLDER_DETECTION = "3_Detected_Cell_Candidates_from_cellfinder"
FOLDER_CLASSIFICATION = "4_Classified_Cells_from_cellfinder"
FOLDER_ANALYSIS = "5_Cell_Counts_by_Region_from_brainglobe_segmentation"


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
        # Find all thumbs.db files recursively (case variations)
        for pattern in ["Thumbs.db", "thumbs.db", "THUMBS.DB"]:
            for thumbs in path.rglob(pattern):
                try_destroy(thumbs)
    else:
        # Just check the immediate directory
        for name in ["Thumbs.db", "thumbs.db", "THUMBS.DB"]:
            thumbs = path / name
            if thumbs.exists():
                try_destroy(thumbs)
    
    return destroyed


def create_desktop_ini_to_prevent_thumbs(folder):
    """
    Create a desktop.ini that might help prevent thumbs.db creation.
    (This is a best-effort attempt - Windows can be stubborn)
    """
    try:
        desktop_ini = folder / "desktop.ini"
        if not desktop_ini.exists():
            desktop_ini.write_text("[ViewState]\nMode=4\nVid=\nFolderType=Generic\n")
            if sys.platform == 'win32':
                import subprocess
                subprocess.run(['attrib', '+h', '+s', str(desktop_ini)], 
                               capture_output=True, check=False)
    except:
        pass  # Best effort only


def create_thumbs_decoy(folder):
    """
    Create a read-only, hidden decoy Thumbs.db file.
    Windows won't overwrite a read-only file, preventing thumbs.db creation.
    
    This is the nuclear prevention option.
    """
    folder = Path(folder)
    decoy = folder / "Thumbs.db"
    
    # Don't create if already exists (might be a real one we haven't nuked yet)
    if decoy.exists():
        return False
    
    try:
        # Create empty file
        decoy.write_bytes(b'')
        
        # Make it read-only and hidden on Windows
        if sys.platform == 'win32':
            import subprocess
            subprocess.run(['attrib', '+h', '+r', str(decoy)], 
                           capture_output=True, check=False)
        else:
            # On Unix, just make it read-only
            decoy.chmod(0o444)
        
        return True
    except:
        return False


# =============================================================================
# FILENAME PARSING AND VALIDATION
# =============================================================================

def decimals_to_p(s):
    """Convert decimals to 'p' in a string (for folder names)."""
    return s.replace('.', 'p')


def p_to_decimals(s):
    """Convert 'p' back to decimals (for parsing folder names)."""
    # Only convert 'p' that's between digits
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
        'mag_str': f"{match.group(5)}x_z{match.group(6)}",
        'mag_folder': decimals_to_p(stem),  # Full name with p notation
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


def calculate_voxel_size_xy(magnification):
    """Calculate XY voxel size from magnification."""
    return CAMERA_PIXEL_SIZE / magnification


# =============================================================================
# PATH HELPERS (NEW STRUCTURE)
# =============================================================================

def get_raw_ims_folder(mouse_folder, mag_folder_name):
    """Get path to 0_Raw_IMS_From_Miami folder inside pipeline."""
    return get_pipeline_folder(mouse_folder, mag_folder_name) / FOLDER_RAW_IMS


def get_pipeline_folder(mouse_folder, mag_folder_name):
    """Get path to pipeline folder for a specific magnification."""
    return Path(mouse_folder) / mag_folder_name


def get_channels_folder(mouse_folder, mag_folder_name):
    """Get path to 1_Channels folder."""
    return get_pipeline_folder(mouse_folder, mag_folder_name) / FOLDER_CHANNELS


def get_channel_folder(mouse_folder, mag_folder_name, channel):
    """Get path to specific channel folder."""
    return get_channels_folder(mouse_folder, mag_folder_name) / f"ch{channel}"


def get_metadata_path(mouse_folder, mag_folder_name):
    """Get path to metadata.json."""
    return get_channels_folder(mouse_folder, mag_folder_name) / "metadata.json"


def create_pipeline_structure(mouse_folder, mag_folder_name):
    """
    Create the full pipeline folder structure.
    Returns the pipeline folder path.
    """
    pipeline = get_pipeline_folder(mouse_folder, mag_folder_name)
    
    # Create main pipeline folder and subfolders
    for subfolder in [FOLDER_RAW_IMS, FOLDER_CHANNELS, FOLDER_REGISTRATION, FOLDER_DETECTION, FOLDER_CLASSIFICATION, FOLDER_ANALYSIS]:
        (pipeline / subfolder).mkdir(parents=True, exist_ok=True)
    
    return pipeline


# =============================================================================
# IMS FILE DISCOVERY
# =============================================================================

def find_ims_files(root_path):
    """
    Find all IMS files in the directory structure.
    
    Looks in:
    - Direct children of root (mouse folders) - for unprocessed files
    - {mouse}/0_Original/ - for old v3.0.x organized files
    - {mouse}/{pipeline}/0_Raw_IMS_From_Miami/ - for new v3.1.0 organized files
    """
    root_path = Path(root_path)
    ims_files = []
    
    for subdir in root_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            # Skip non-mouse folders
            if 'script' in subdir.name.lower():
                continue
            
            # Check directly in mouse folder (old structure or not yet organized)
            for ims_file in subdir.glob('*.ims'):
                ims_files.append(ims_file)
            
            # Check old 0_Original at mouse level (v3.0.x structure)
            old_original = subdir / "0_Original"
            if old_original.exists():
                for ims_file in old_original.glob('*.ims'):
                    ims_files.append(ims_file)
            
            # Check pipeline folders for 0_Raw_IMS_From_Miami (new v3.1.0 structure)
            for pipeline_folder in subdir.iterdir():
                if pipeline_folder.is_dir():
                    raw_folder = pipeline_folder / FOLDER_RAW_IMS
                    if raw_folder.exists():
                        for ims_file in raw_folder.glob('*.ims'):
                            ims_files.append(ims_file)
    
    # Also check root directly
    for ims_file in root_path.glob('*.ims'):
        ims_files.append(ims_file)
    
    # Remove duplicates (in case file is somehow in both places)
    return sorted(set(ims_files))


# =============================================================================
# IMS METADATA EXTRACTION
# =============================================================================

def get_voxel_sizes_from_ims(filepath):
    """
    Extract voxel size information from IMS file.
    Returns dict with voxel sizes and dimensions.
    """
    import h5py
    
    result = {
        'voxel_size_x': None,
        'voxel_size_y': None,
        'voxel_size_z': None,
        'size_x': None,
        'size_y': None,
        'size_z': None,
        'metadata_source': 'none',
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Get dimensions
            if 'DataSetInfo/Image' in f:
                img_info = f['DataSetInfo/Image']
                for dim in ['X', 'Y', 'Z']:
                    attr_name = f'Size{dim}' if f'Size{dim}' in img_info.attrs else dim
                    if attr_name in img_info.attrs:
                        val = img_info.attrs[attr_name]
                        if hasattr(val, '__iter__'):
                            val = val[0]
                        if isinstance(val, bytes):
                            val = val.decode()
                        try:
                            result[f'size_{dim.lower()}'] = int(float(val))
                        except:
                            pass
            
            # Try to get voxel sizes from metadata
            if 'DataSetInfo/Image' in f:
                img_info = f['DataSetInfo/Image']
                for dim in ['X', 'Y', 'Z']:
                    for attr_name in [f'ExtMax{dim}', f'ExtMin{dim}']:
                        if attr_name in img_info.attrs:
                            result['metadata_source'] = 'ims_file'
                            break
    except Exception as e:
        pass
    
    return result


def get_channel_info(filepath):
    """Get information about available channels."""
    import h5py
    
    channels = []
    try:
        with h5py.File(filepath, 'r') as f:
            i = 0
            while True:
                channel_path = f'DataSetInfo/Channel {i}'
                if channel_path in f:
                    channel_info = f[channel_path]
                    name = f"Channel{i}"
                    try:
                        if 'Name' in channel_info.attrs:
                            name_val = channel_info.attrs['Name'][0]
                            if isinstance(name_val, bytes):
                                name_val = name_val.decode()
                            name = str(name_val)
                    except:
                        pass
                    channels.append({'index': i, 'name': name})
                    i += 1
                else:
                    break
    except:
        pass
    
    return channels


def is_valid_voxel_size(value):
    """Check if a voxel size is reasonable (not placeholder garbage)."""
    if value is None:
        return False
    try:
        v = float(value)
        return 0.1 <= v <= 100.0
    except:
        return False


# =============================================================================
# PROCESSING STATUS CHECK
# =============================================================================

def load_metadata(mouse_folder, mag_folder_name):
    """
    Load existing metadata from JSON.
    Checks new v3 location first, then falls back to old v2.x locations.
    Returns None if not found.
    """
    mouse_folder = Path(mouse_folder)
    
    # New v3 location: {mouse}/{mag_folder}/1_Channels/metadata.json
    json_path = get_metadata_path(mouse_folder, mag_folder_name)
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Old v2.x location: {mouse}/{stem}/processed.json
    # Need to convert mag_folder back to stem (1p625x_z4 â†’ 1.625x_z4)
    stem_guess = p_to_decimals(mag_folder_name)
    old_json = mouse_folder / stem_guess / "processed.json"
    if old_json.exists():
        try:
            with open(old_json, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Also check for {mouse}/{stem}_processed.json (v1.x/early v2.0)
    old_flat_json = mouse_folder / f"{stem_guess}_processed.json"
    if old_flat_json.exists():
        try:
            with open(old_flat_json, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return None


def get_ims_location_status(ims_path):
    """
    Determine where the IMS file currently is.
    
    Returns:
        'in_raw_folder': File is in pipeline's 0_Raw_IMS_From_Miami folder (correct)
        'in_mouse_folder': File is directly in mouse folder (needs moving)
        'unknown': Can't determine
    """
    ims_path = Path(ims_path)
    
    if ims_path.parent.name == FOLDER_RAW_IMS:
        return 'in_raw_folder'
    else:
        return 'in_mouse_folder'


def needs_processing(ims_path):
    """
    Check if an IMS file needs processing.
    
    Returns:
        (status: str, reason: str)
        
        Status can be:
        - "needs_processing": Full extraction needed
        - "needs_organization": Just needs file/folder moves
        - "up_to_date": Already processed correctly
        - "invalid": Filename doesn't match expected format
    """
    ims_path = Path(ims_path)
    
    # Check filename validity
    is_valid, validation_reason = validate_filename(ims_path.name)
    if not is_valid:
        return "invalid", validation_reason
    
    parsed = parse_filename(ims_path.name)
    mag_folder_name = parsed['mag_folder']  # Full name: 349_CNT_01_02_1p625x_z4
    stem = parsed['stem']
    mag_only = decimals_to_p(parsed['mag_str'])  # Just magnification: 1p625x_z4
    
    # Determine mouse folder based on where IMS currently is
    if ims_path.parent.name == FOLDER_RAW_IMS:
        # IMS is in a raw folder - parent.parent is pipeline, parent.parent.parent is mouse
        mouse_folder = ims_path.parent.parent.parent
        pipeline_name = ims_path.parent.parent.name
        ims_in_correct_raw = pipeline_name == mag_folder_name
    elif ims_path.parent.name == "0_Original":
        # Old v3.0.x structure: IMS in mouse/0_Original/
        mouse_folder = ims_path.parent.parent
        ims_in_correct_raw = False
    else:
        # IMS is directly in mouse folder
        mouse_folder = ims_path.parent
        ims_in_correct_raw = False
    
    # Check for old v3.0.x short-name folder (e.g., 1p625x_z4 instead of full name)
    old_v3_short = mouse_folder / mag_only
    old_v3_short_channels = old_v3_short / "1_Channels" / "ch0"
    has_old_v3_short = (old_v3_short.exists() and 
                        mag_only != mag_folder_name and
                        old_v3_short_channels.exists() and 
                        len(list(old_v3_short_channels.glob('Z*.tif'))) > 0)
    
    # Check for old folder names that need renaming (e.g., "1_Channels" â†’ new name)
    pipeline_folder = mouse_folder / mag_folder_name
    old_channel_folder = pipeline_folder / "1_Channels"
    has_old_folder_names = (old_channel_folder.exists() and 
                            "1_Channels" != FOLDER_CHANNELS and
                            len(list((old_channel_folder / "ch0").glob('Z*.tif') if (old_channel_folder / "ch0").exists() else [])) > 0)
    
    # Check for old 0_Original at mouse level
    old_original = mouse_folder / "0_Original"
    has_old_original = old_original.exists() and (old_original / ims_path.name).exists()
    
    # Check for old v2.x structure: {stem}/ch0/
    old_output = mouse_folder / stem
    old_ch0 = old_output / "ch0"
    has_old_v2_structure = old_ch0.exists() and len(list(old_ch0.glob('Z*.tif'))) > 0
    
    # Check for old flat structure: {stem}_ch0/
    old_flat_ch0 = mouse_folder / f"{stem}_ch0"
    has_old_flat = old_flat_ch0.exists() and len(list(old_flat_ch0.glob('Z*.tif'))) > 0
    
    # Check for new v3.1 structure: {full_mag_folder}/1_Channels/ch0/
    new_ch0 = get_channel_folder(mouse_folder, mag_folder_name, 0)
    has_new_structure = new_ch0.exists() and len(list(new_ch0.glob('Z*.tif'))) > 0
    
    # Decision logic
    if has_new_structure:
        # Channels are in correct location with correct folder name
        if not ims_in_correct_raw:
            return "needs_organization", "IMS file needs moving to pipeline raw folder"
        return "up_to_date", "up to date"
    
    elif has_old_folder_names:
        return "needs_organization", "old folder names (1_Channels â†’ new name)"
    
    elif has_old_v3_short:
        return "needs_organization", f"old v3.0.x short folder name ({mag_only})"
    
    elif has_old_original:
        return "needs_organization", "IMS in old 0_Original location"
    
    elif has_old_v2_structure:
        return "needs_organization", "old v2.x structure (can migrate)"
    
    elif has_old_flat:
        return "needs_organization", "old flat structure (can migrate)"
    
    else:
        # No extracted channels found - need full processing
        return "needs_processing", "not processed"


# =============================================================================
# ORGANIZATION / MIGRATION
# =============================================================================

def organize_ims_file(ims_path, mag_folder_name):
    """
    Move IMS file to 0_Raw_IMS_From_Miami folder inside its pipeline.
    Returns new path.
    """
    ims_path = Path(ims_path)
    
    # Determine mouse folder based on current location
    if ims_path.parent.name == FOLDER_RAW_IMS:
        # Already in a raw folder - check if it's the right one
        pipeline_folder = ims_path.parent.parent
        if pipeline_folder.name == mag_folder_name:
            # Already in correct place
            return ims_path
        # Wrong pipeline folder - need to move
        mouse_folder = pipeline_folder.parent
    elif ims_path.parent.name == "0_Original":
        # Old v3.0.x structure: mouse/0_Original/
        mouse_folder = ims_path.parent.parent
    else:
        mouse_folder = ims_path.parent
    
    # Create pipeline structure and get raw IMS folder
    create_pipeline_structure(mouse_folder, mag_folder_name)
    raw_folder = get_raw_ims_folder(mouse_folder, mag_folder_name)
    
    new_path = raw_folder / ims_path.name
    
    if ims_path != new_path:
        print(f"    Moving IMS to {mag_folder_name}/{FOLDER_RAW_IMS}/...")
        sys.stdout.flush()
        shutil.move(str(ims_path), str(new_path))
        
        # Clean up old 0_Original if it exists and is now empty
        old_original = mouse_folder / "0_Original"
        if old_original.exists():
            try:
                if not any(old_original.iterdir()):
                    old_original.rmdir()
                    print(f"    Removed empty 0_Original/")
            except:
                pass
    
    return new_path


def migrate_old_structure(ims_path):
    """
    Migrate from old folder structures to new numbered structure.
    
    Handles:
    - v3.0.x: {mag_only}/1_Channels/ â†’ {full_name}/1_Channels/
    - v3.0.x: mouse/0_Original/*.ims â†’ {full_name}/0_Raw_IMS_From_Miami/*.ims
    - v2.0.x: {stem}/ch0/ â†’ {full_name}/1_Channels/ch0/
    - early v2.0: {stem}_ch0/ â†’ {full_name}/1_Channels/ch0/
    
    Returns (success, message).
    """
    ims_path = Path(ims_path)
    parsed = parse_filename(ims_path.name)
    mag_folder_name = parsed['mag_folder']  # Full name with p notation
    stem = parsed['stem']
    
    # Extract just the magnification part for detecting old v3.0.x folders
    mag_only = decimals_to_p(parsed['mag_str'])  # e.g., "1p625x_z4"
    
    # Determine mouse folder based on current IMS location
    if ims_path.parent.name == FOLDER_RAW_IMS:
        # Already in a raw folder
        mouse_folder = ims_path.parent.parent.parent
    elif ims_path.parent.name == "0_Original":
        # Old v3.0.x structure: mouse/0_Original/
        mouse_folder = ims_path.parent.parent
    else:
        mouse_folder = ims_path.parent
    
    channels_migrated = []
    actions = []
    
    # =================================================================
    # CHECK FOR v3.0.x SHORT-NAME FOLDERS (e.g., 1p625x_z4 instead of 349_CNT_01_02_1p625x_z4)
    # =================================================================
    old_v3_pipeline = mouse_folder / mag_only
    if old_v3_pipeline.exists() and old_v3_pipeline.is_dir() and mag_only != mag_folder_name:
        # Found old short-name folder, need to rename it
        new_pipeline = mouse_folder / mag_folder_name
        
        if new_pipeline.exists():
            # Merge contents - need to handle nested folders properly
            for item in old_v3_pipeline.iterdir():
                dest = new_pipeline / item.name
                if dest.exists():
                    # If it's a directory, merge contents recursively
                    if item.is_dir() and dest.is_dir():
                        # Check if dest is empty or has fewer items
                        dest_items = list(dest.iterdir())
                        item_items = list(item.iterdir())
                        if len(item_items) > len(dest_items):
                            # Old folder has more content, remove empty dest and move
                            shutil.rmtree(str(dest))
                            shutil.move(str(item), str(dest))
                            actions.append(f"Replaced empty {mag_folder_name}/{item.name}/ with {mag_only}/{item.name}/")
                        else:
                            # Recursively merge contents
                            for subitem in item.iterdir():
                                subdest = dest / subitem.name
                                if not subdest.exists():
                                    shutil.move(str(subitem), str(subdest))
                                    actions.append(f"Moved {mag_only}/{item.name}/{subitem.name} â†’ {mag_folder_name}/{item.name}/")
                else:
                    shutil.move(str(item), str(dest))
                    actions.append(f"Moved {mag_only}/{item.name} â†’ {mag_folder_name}/{item.name}")
            # Remove old folder if empty
            try:
                shutil.rmtree(str(old_v3_pipeline))
                actions.append(f"Removed {mag_only}/")
            except:
                pass
        else:
            # Just rename the folder
            shutil.move(str(old_v3_pipeline), str(new_pipeline))
            actions.append(f"Renamed {mag_only}/ â†’ {mag_folder_name}/")
    
    # =================================================================
    # CHECK FOR OLD 0_Original AT MOUSE LEVEL
    # =================================================================
    old_original = mouse_folder / "0_Original"
    if old_original.exists() and old_original.is_dir():
        # Look for this IMS file in old 0_Original
        old_ims_location = old_original / ims_path.name
        if old_ims_location.exists() or ims_path.parent.name == "0_Original":
            # Create new structure and move IMS
            create_pipeline_structure(mouse_folder, mag_folder_name)
            new_raw_folder = get_raw_ims_folder(mouse_folder, mag_folder_name)
            new_ims_path = new_raw_folder / ims_path.name
            
            if old_ims_location.exists() and not new_ims_path.exists():
                shutil.move(str(old_ims_location), str(new_ims_path))
                actions.append(f"Moved 0_Original/{ims_path.name} â†’ {mag_folder_name}/{FOLDER_RAW_IMS}/")
                ims_path = new_ims_path
        
        # Clean up old 0_Original if empty
        try:
            if not any(old_original.iterdir()):
                old_original.rmdir()
                actions.append("Removed empty 0_Original/")
        except:
            pass
    
    # =================================================================
    # CREATE NEW STRUCTURE
    # =================================================================
    pipeline = create_pipeline_structure(mouse_folder, mag_folder_name)
    channels_folder = get_channels_folder(mouse_folder, mag_folder_name)
    
    # =================================================================
    # RENAME OLD FOLDER NAMES TO NEW DESCRIPTIVE NAMES
    # =================================================================
    old_to_new_folders = [
        ("1_Channels", FOLDER_CHANNELS),
        ("2_Registration", FOLDER_REGISTRATION),
        ("3_Detection", FOLDER_DETECTION),
        ("4_Analysis", FOLDER_ANALYSIS),  # Old 4 â†’ new 5
    ]
    
    for old_name, new_name in old_to_new_folders:
        if old_name == new_name:
            continue
        old_folder = pipeline / old_name
        new_folder = pipeline / new_name
        if old_folder.exists() and old_folder.is_dir():
            if new_folder.exists():
                # Merge contents if new folder exists but is empty
                if not any(new_folder.iterdir()):
                    shutil.rmtree(str(new_folder))
                    shutil.move(str(old_folder), str(new_folder))
                    actions.append(f"Renamed {old_name}/ â†’ {new_name}/")
                else:
                    # Move contents from old to new
                    for item in old_folder.iterdir():
                        dest = new_folder / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                    try:
                        old_folder.rmdir()
                        actions.append(f"Merged {old_name}/ into {new_name}/")
                    except:
                        pass
            else:
                shutil.move(str(old_folder), str(new_folder))
                actions.append(f"Renamed {old_name}/ â†’ {new_name}/")
    
    # Update channels_folder reference after potential rename
    channels_folder = get_channels_folder(mouse_folder, mag_folder_name)
    
    # =================================================================
    # CHECK FOR v2.0.x STRUCTURE: {stem}/ch0/
    # =================================================================
    old_output = mouse_folder / stem
    if old_output.exists() and old_output.is_dir():
        for ch_idx in range(10):
            old_ch = old_output / f"ch{ch_idx}"
            new_ch = channels_folder / f"ch{ch_idx}"
            
            if old_ch.exists() and old_ch.is_dir():
                if new_ch.exists():
                    shutil.rmtree(str(new_ch))
                shutil.move(str(old_ch), str(new_ch))
                channels_migrated.append(ch_idx)
                actions.append(f"Moved {stem}/ch{ch_idx} â†’ {mag_folder_name}/{FOLDER_CHANNELS}/ch{ch_idx}/")
        
        # Move metadata
        old_json = old_output / "processed.json"
        if old_json.exists():
            new_json = get_metadata_path(mouse_folder, mag_folder_name)
            shutil.move(str(old_json), str(new_json))
        
        # Remove old folder if empty
        try:
            old_output.rmdir()
            actions.append(f"Removed empty {stem}/")
        except:
            pass
    
    # =================================================================
    # CHECK FOR v3.0.x CHANNELS (in both old short-name and new full-name folders)
    # =================================================================
    # Check new location first
    new_channels_folder = mouse_folder / mag_folder_name / "1_Channels"
    if new_channels_folder.exists():
        for ch_idx in range(10):
            ch_folder = new_channels_folder / f"ch{ch_idx}"
            if ch_folder.exists() and ch_folder.is_dir() and ch_idx not in channels_migrated:
                if len(list(ch_folder.glob('Z*.tif'))) > 0:
                    channels_migrated.append(ch_idx)
    
    # Also check if there's still an old short-name folder with channels (shouldn't happen after merge above)
    old_short_channels = mouse_folder / mag_only / "1_Channels"
    if old_short_channels.exists() and mag_only != mag_folder_name:
        for ch_idx in range(10):
            ch_folder = old_short_channels / f"ch{ch_idx}"
            if ch_folder.exists() and ch_folder.is_dir() and ch_idx not in channels_migrated:
                if len(list(ch_folder.glob('Z*.tif'))) > 0:
                    channels_migrated.append(ch_idx)
    
    # =================================================================
    # CHECK FOR EARLY v2.0 FLAT STRUCTURE: {stem}_ch0/
    # =================================================================
    for ch_idx in range(10):
        old_flat = mouse_folder / f"{stem}_ch{ch_idx}"
        new_ch = channels_folder / f"ch{ch_idx}"
        
        if old_flat.exists() and old_flat.is_dir():
            if new_ch.exists():
                shutil.rmtree(str(new_ch))
            shutil.move(str(old_flat), str(new_ch))
            if ch_idx not in channels_migrated:
                channels_migrated.append(ch_idx)
            actions.append(f"Moved {stem}_ch{ch_idx}/ â†’ {mag_folder_name}/{FOLDER_CHANNELS}/ch{ch_idx}/")
    
    # Move old JSON if present
    old_json = mouse_folder / f"{stem}_processed.json"
    if old_json.exists():
        new_json = get_metadata_path(mouse_folder, mag_folder_name)
        shutil.move(str(old_json), str(new_json))
    
    # Print actions
    for action in actions:
        print(f"      {action}")
    
    if channels_migrated or actions:
        # Update metadata
        if channels_migrated:
            update_metadata_paths(mouse_folder, mag_folder_name, channels_migrated)
        msg_parts = []
        if channels_migrated:
            msg_parts.append(f"migrated {len(channels_migrated)} channel(s)")
        if actions:
            msg_parts.append(f"{len(actions)} folder operation(s)")
        return True, ", ".join(msg_parts)
    
    return False, "no old structure found"


def update_metadata_paths(mouse_folder, mag_folder_name, channels):
    """Update metadata.json with new paths."""
    json_path = get_metadata_path(mouse_folder, mag_folder_name)
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
    else:
        metadata = {}
    
    metadata['script_version'] = SCRIPT_VERSION
    metadata['pipeline_folder'] = mag_folder_name
    metadata['channels_extracted'] = sorted(channels)
    metadata['output_folders'] = {f'ch{ch}': f'ch{ch}' for ch in channels}
    metadata['output_format'] = 'individual_slices'
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# MAIN CONVERSION
# =============================================================================

def convert_ims_to_tiff(ims_path, force_move=True):
    """
    Convert an IMS file to TIFF slices for BrainGlobe/cellfinder.
    
    Sets up full pipeline structure and extracts all channels.
    """
    from imaris_ims_file_reader.ims import ims
    import tifffile
    
    ims_path = Path(ims_path)
    parsed = parse_filename(ims_path.name)
    
    mag_folder_name = parsed['mag_folder']
    
    # Determine mouse folder based on current IMS location
    if ims_path.parent.name == FOLDER_RAW_IMS:
        # IMS is in pipeline/0_Raw_IMS_From_Miami/
        mouse_folder = ims_path.parent.parent.parent
    else:
        # IMS is directly in mouse folder
        mouse_folder = ims_path.parent
    
    # Move IMS to pipeline's raw folder if needed
    if force_move and ims_path.parent.name != FOLDER_RAW_IMS:
        ims_path = organize_ims_file(ims_path, mag_folder_name)
    
    # Create pipeline structure
    pipeline = create_pipeline_structure(mouse_folder, mag_folder_name)
    channels_folder = get_channels_folder(mouse_folder, mag_folder_name)
    
    # Get file info
    file_size_gb = ims_path.stat().st_size / (1024**3)
    
    # Get metadata
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        voxel_info = get_voxel_sizes_from_ims(str(ims_path))
        channels = get_channel_info(str(ims_path))
    finally:
        sys.stderr = old_stderr
    
    num_channels = len(channels)
    
    # Calculate voxel sizes from filename
    voxel_xy = calculate_voxel_size_xy(parsed['magnification'])
    voxel_z = parsed['z_step']
    
    # Show info
    dims = f"{voxel_info['size_x']}x{voxel_info['size_y']}x{voxel_info['size_z']}"
    print(f"    File size: {file_size_gb:.1f} GB | Dimensions: {dims}")
    print(f"    Voxels (Âµm): X={voxel_xy:.2f}, Y={voxel_xy:.2f}, Z={voxel_z:.2f}")
    print(f"    Channels: {num_channels}")
    print(f"    Output: {mag_folder_name}/{FOLDER_CHANNELS}/")
    sys.stdout.flush()
    
    # Open IMS file
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        ims_data = ims(str(ims_path), ResolutionLevelLock=0, squeeze_output=False)
    finally:
        sys.stderr = old_stderr
    
    # Track results
    output_folders = {}
    channels_extracted = []
    total_start = time.time()
    num_slices = 0
    
    # Process each channel
    for ch_idx in range(num_channels):
        ch_name = channels[ch_idx]['name'] if ch_idx < len(channels) else f"Channel{ch_idx}"
        print(f"    Channel {ch_idx} ({ch_name}):")
        sys.stdout.flush()
        
        # Create channel folder
        ch_folder = channels_folder / f"ch{ch_idx}"
        ch_folder.mkdir(exist_ok=True)
        output_folders[f'ch{ch_idx}'] = f"ch{ch_idx}"
        
        # Create thumbs.db decoy to prevent Windows from creating one
        create_thumbs_decoy(ch_folder)
        
        # Load channel data
        print(f"      Loading... ", end='')
        sys.stdout.flush()
        start_time = time.time()
        
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            volume = np.array(ims_data[0, ch_idx, :, :, :])
        finally:
            sys.stderr = old_stderr
        
        volume = np.squeeze(volume)
        load_time = time.time() - start_time
        num_slices = volume.shape[0]
        print(f"done ({load_time:.1f}s) | Shape: {volume.shape}")
        sys.stdout.flush()
        
        # Save individual slices
        print(f"      Saving {num_slices} slices... ", end='')
        sys.stdout.flush()
        start_time = time.time()
        
        for z in range(num_slices):
            slice_path = ch_folder / f"Z{z:04d}.tif"
            tifffile.imwrite(str(slice_path), volume[z, :, :])
            
            if (z + 1) % 100 == 0:
                print(f"{z+1}...", end='')
                sys.stdout.flush()
        
        save_time = time.time() - start_time
        print(f" done ({save_time:.1f}s)")
        sys.stdout.flush()
        
        channels_extracted.append(ch_idx)
        del volume
        
        # Annihilate any thumbs.db that might have appeared
        annihilate_thumbs_db(ch_folder, recursive=False)
    
    total_time = time.time() - total_start
    print(f"    Total extraction time: {total_time:.1f}s")
    
    # Build metadata
    metadata = {
        'script_version': SCRIPT_VERSION,
        'processed_date': datetime.now().isoformat(),
        'source_file': ims_path.name,
        'source_path': str(ims_path),
        'pipeline_folder': mag_folder_name,
        'magnification': parsed['magnification'],
        'z_step': parsed['z_step'],
        'dimensions': {
            'x': int(voxel_info['size_x']) if voxel_info['size_x'] else None,
            'y': int(voxel_info['size_y']) if voxel_info['size_y'] else None,
            'z': int(voxel_info['size_z']) if voxel_info['size_z'] else num_slices,
        },
        'voxel_size_um': {
            'x': round(voxel_xy, 4),
            'y': round(voxel_xy, 4),
            'z': round(voxel_z, 4),
        },
        'channels_available': num_channels,
        'channels_extracted': channels_extracted,
        'channel_names': {ch['index']: ch['name'] for ch in channels},
        'output_format': 'individual_slices',
        'slices_per_channel': num_slices,
        'orientation': DEFAULT_ORIENTATION,
        'output_folders': output_folders,
        'brainreg_command': (
            f"brainreg \"{channels_folder / 'ch0'}\" \"{pipeline / FOLDER_REGISTRATION}\" "
            f"-v {voxel_z:.2f} {voxel_xy:.2f} {voxel_xy:.2f} "
            f"--orientation {DEFAULT_ORIENTATION} --atlas allen_mouse_25um"
        ),
    }
    
    # Save metadata
    json_path = get_metadata_path(mouse_folder, mag_folder_name)
    print(f"    Saving metadata: {mag_folder_name}/{FOLDER_CHANNELS}/metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final thumbs.db sweep
    destroyed = annihilate_thumbs_db(pipeline)
    if destroyed > 0:
        print(f"    Destroyed {destroyed} thumbs.db file(s)")
    
    return metadata


# =============================================================================
# SCANNING AND REPORTING
# =============================================================================

def scan_and_report(root_path):
    """
    Scan for IMS files and report their status.
    """
    ims_files = find_ims_files(root_path)
    
    all_files = []
    needs_work = []
    invalid_files = []
    
    for ims_path in ims_files:
        status, reason = needs_processing(ims_path)
        all_files.append((ims_path, status, reason))
        
        if status in ("needs_processing", "needs_organization"):
            needs_work.append((ims_path, status, reason))
        elif status == "invalid":
            invalid_files.append((ims_path, status, reason))
    
    return all_files, needs_work, invalid_files


def print_scan_results(all_files, needs_work, invalid_files):
    """Print a summary of the scan results."""
    print(f"\nFound {len(all_files)} .ims file(s):\n")
    
    for ims_path, status, reason in all_files:
        # Get relative path for cleaner display
        if ims_path.parent.name == FOLDER_RAW_IMS:
            # IMS in pipeline/0_Raw_IMS_From_Miami/
            mag_folder = ims_path.parent.parent.name
            mouse_folder = ims_path.parent.parent.parent.name
            rel_path = f"{mouse_folder}/{mag_folder}/{FOLDER_RAW_IMS}/{ims_path.name}"
        else:
            rel_path = f"{ims_path.parent.name}/{ims_path.name}"
        
        if status == "up_to_date":
            print(f"  âœ“ {rel_path}")
        elif status == "invalid":
            print(f"  âœ— {rel_path}")
            print(f"      SKIP: {reason}")
        elif status == "needs_organization":
            print(f"  â†» {rel_path} ({reason})")
        else:
            print(f"  â—‹ {rel_path} ({reason})")
    
    if invalid_files:
        print(f"\n  âš  {len(invalid_files)} file(s) skipped - rename to: NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims")
    
    return len(needs_work)


def write_voxel_summary(root_path, all_processing_info):
    """Write a summary of voxel sizes for all processed brains."""
    summary_path = Path(root_path) / "VOXEL_SIZES.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VOXEL SIZES FOR NAPARI / BRAINREG\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Script version: {SCRIPT_VERSION}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Enter voxel sizes as: Z, Y, X (in microns)\n")
        f.write("-" * 70 + "\n\n")
        
        for info in sorted(all_processing_info, key=lambda x: x.get('source_file', '')):
            name = info.get('source_file', 'Unknown')
            pipeline = info.get('pipeline_folder', '?')
            voxel = info.get('voxel_size_um', {})
            vx = voxel.get('x', '?')
            vy = voxel.get('y', '?')
            vz = voxel.get('z', '?')
            
            if isinstance(vx, float):
                voxel_str = f"{vz:.2f}, {vy:.2f}, {vx:.2f}"
            else:
                voxel_str = f"{vz}, {vy}, {vx}"
            
            f.write(f"{name}\n")
            f.write(f"    Pipeline: {pipeline}/\n")
            f.write(f"    Voxels (Z,Y,X): {voxel_str} Âµm\n")
            
            slices = info.get('slices_per_channel', 0)
            if slices:
                f.write(f"    Slices: {slices}\n")
            
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("For brainreg, use: -v Z Y X\n")
        f.write("For napari scale, use: (Z, Y, X)\n")
    
    return summary_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert Imaris .ims files to TIFF for BrainGlobe/brainreg',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {SCRIPT_VERSION}

Output Structure:
  MouseFolder/
  â””â”€â”€ 349_CNT_01_02_1p625x_z4/
      â”œâ”€â”€ 0_Raw_IMS_From_Miami/
      â”œâ”€â”€ 1_Extracted_Channels_from_1_ims_to_brainglobe/
      â”œâ”€â”€ 2_Registered_Atlas_from_brainreg/
      â”œâ”€â”€ 3_Detected_Cell_Candidates_from_cellfinder/
      â”œâ”€â”€ 4_Classified_Cells_from_cellfinder/
      â””â”€â”€ 5_Cell_Counts_by_Region_from_brainglobe_segmentation/

Examples:
  python 1_ims_to_brainglobe.py
  python 1_ims_to_brainglobe.py --inspect
  python 1_ims_to_brainglobe.py --force
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help=f'Path to scan (default: {DEFAULT_BRAINGLOBE_ROOT})')
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Only scan and report, do not process')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force reprocess all files regardless of status')
    
    args = parser.parse_args()
    
    # Determine root path
    if args.path:
        root_path = Path(args.path)
    else:
        root_path = DEFAULT_BRAINGLOBE_ROOT
    
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        print(f"\nEdit DEFAULT_BRAINGLOBE_ROOT in the script or provide a path as argument.")
        sys.exit(1)
    
    print("=" * 60)
    print("IMS to BrainGlobe Converter")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Initial thumbs.db sweep
    print(f"\nScanning: {root_path}")
    print("Sweeping for thumbs.db files...")
    destroyed = annihilate_thumbs_db(root_path)
    if destroyed > 0:
        print(f"  Destroyed {destroyed} thumbs.db file(s)")
    
    # Handle single file
    if root_path.is_file() and root_path.suffix.lower() == '.ims':
        status, reason = needs_processing(root_path)
        all_files = [(root_path, status, reason)]
        if status == "invalid":
            invalid_files = [(root_path, status, reason)]
            needs_work = []
        elif status in ("needs_processing", "needs_organization") or args.force:
            needs_work = [(root_path, status if not args.force else "needs_processing", 
                          reason if not args.force else "forced")]
            invalid_files = []
        else:
            needs_work = []
            invalid_files = []
    else:
        # Scan directory
        all_files, needs_work, invalid_files = scan_and_report(root_path)
        
        if args.force:
            needs_work = [(p, "needs_processing", "forced") for p, s, _ in all_files if s != "invalid"]
    
    n_needs_work = print_scan_results(all_files, needs_work, invalid_files)
    
    if n_needs_work == 0:
        print("\nAll files are up to date. Nothing to do.")
        # Still collect info for summary
        all_processing_info = []
        for ims_path, status, reason in all_files:
            if status == "up_to_date":
                parsed = parse_filename(ims_path.name)
                if parsed:
                    if ims_path.parent.name == FOLDER_RAW_IMS:
                        # IMS in pipeline/0_Raw_IMS_From_Miami/
                        mouse_folder = ims_path.parent.parent.parent
                    else:
                        mouse_folder = ims_path.parent
                    info = load_metadata(mouse_folder, parsed['mag_folder'])
                    if info:
                        all_processing_info.append(info)
        if all_processing_info:
            summary_path = write_voxel_summary(root_path, all_processing_info)
            print(f"\nVoxel summary updated: {summary_path}")
        return
    
    if args.inspect:
        print(f"\n{n_needs_work} file(s) would be processed.")
        return
    
    # Confirm
    response = input(f"\nProcess {n_needs_work} file(s)? [Enter to continue, 'q' to quit]: ").strip()
    if response.lower() == 'q':
        print("Cancelled.")
        return
    
    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)
    
    success = 0
    organized = 0
    failed = 0
    all_processing_info = []
    
    for ims_path, status, reason in needs_work:
        if ims_path.parent.name == FOLDER_RAW_IMS:
            # IMS in pipeline/0_Raw_IMS_From_Miami/
            mag_folder = ims_path.parent.parent.name
            mouse_name = ims_path.parent.parent.parent.name
            rel_path = f"{mouse_name}/{mag_folder}/{FOLDER_RAW_IMS}/{ims_path.name}"
        else:
            rel_path = f"{ims_path.parent.name}/{ims_path.name}"
        
        print(f"\n  [{success + organized + failed + 1}/{n_needs_work}] {rel_path}")
        
        try:
            if status == "needs_organization":
                # Try migration first
                print("    Reorganizing...")
                
                parsed = parse_filename(ims_path.name)
                mag_folder_name = parsed['mag_folder']
                
                # Move IMS file if needed
                if ims_path.parent.name != FOLDER_RAW_IMS:
                    ims_path = organize_ims_file(ims_path, mag_folder_name)
                
                # Migrate old channel folders
                did_migrate, msg = migrate_old_structure(ims_path)
                
                if did_migrate:
                    organized += 1
                    print(f"    âœ“ Reorganized ({msg})")
                    
                    # Load updated metadata
                    # IMS is now in mouse_folder/mag_folder/0_Raw_IMS/
                    mouse_folder = ims_path.parent.parent.parent
                    info = load_metadata(mouse_folder, mag_folder_name)
                    if info:
                        all_processing_info.append(info)
                else:
                    # Fall back to full processing
                    print(f"    Migration incomplete, processing...")
                    info = convert_ims_to_tiff(ims_path)
                    all_processing_info.append(info)
                    success += 1
                    print(f"    âœ“ Done")
            else:
                # Full processing
                info = convert_ims_to_tiff(ims_path)
                all_processing_info.append(info)
                success += 1
                print(f"    âœ“ Done")
        except Exception as e:
            failed += 1
            print(f"    âœ— Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Collect info from already-processed files
    for ims_path, status, reason in all_files:
        if status == "up_to_date":
            parsed = parse_filename(ims_path.name)
            if parsed:
                if ims_path.parent.name == FOLDER_RAW_IMS:
                    # IMS in pipeline/0_Raw_IMS_From_Miami/
                    mouse_folder = ims_path.parent.parent.parent
                else:
                    mouse_folder = ims_path.parent
                info = load_metadata(mouse_folder, parsed['mag_folder'])
                if info:
                    all_processing_info.append(info)
    
    # Write voxel summary
    if all_processing_info:
        summary_path = write_voxel_summary(root_path, all_processing_info)
        print(f"\nVoxel summary: {summary_path}")
    
    # Final thumbs.db sweep
    print("\nFinal thumbs.db sweep...")
    destroyed = annihilate_thumbs_db(root_path)
    if destroyed > 0:
        print(f"  Destroyed {destroyed} additional thumbs.db file(s)")
    
    print("\n" + "=" * 60)
    if organized > 0:
        print(f"Complete: {success} processed, {organized} reorganized, {failed} failed")
    else:
        print(f"Complete: {success} succeeded, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\nCheck failed files - likely missing voxel info in filename.")
        print("Expected format: NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims")


if __name__ == '__main__':
    main()
    
    # If double-clicked (no args passed), pause so window doesn't close
    if len(sys.argv) == 1:
        print()
        input("Press Enter to close...")
