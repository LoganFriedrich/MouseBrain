#!/usr/bin/env python3
"""
2_brainreg_registration.py (v1.2.0)

================================================================================
WHAT IS THIS?
================================================================================
This is Script 2 in the BrainGlobe pipeline. It takes the channel images 
extracted by Script 1 and registers them to the Allen Mouse Brain Atlas using 
brainreg. This alignment allows you to identify which brain regions your cells 
are in.

Registration = "warping" your brain images to match a standard atlas so that
every voxel in your image corresponds to a known anatomical location.

================================================================================
PREREQUISITES
================================================================================
1. You must have already run Script 1 (1_ims_to_brainglobe.py) on your brains
2. You need the brainglobe conda environment activated
3. The Allen Mouse 10µm atlas will be downloaded automatically on first run

================================================================================
HOW TO RUN
================================================================================
Open Anaconda Prompt, then:

    conda activate brainglobe-env
    cd Y:\\2_Connectome\\3_Nuclei_Detection\\BrainGlobe\\0_Scripts
    python 2_brainreg_registration.py

The script will show you an interactive menu listing all available brains and 
let you choose which ones to process.

================================================================================
INTERACTIVE MENU OPTIONS
================================================================================
When you run the script, you'll see a menu like this:

    [READY TO REGISTER] (3 brain(s)):
      1. 349_CNT_01_02/349_CNT_01_02_1p625x_z4
      2. 350_SCI_01_01/350_SCI_01_01_1p9x_z3
      3. 351_CNT_02_01/351_CNT_02_01_1p625x_z4

    [ALREADY COMPLETED] (1 brain(s)):
      C1. 143_test/143_test_1p625x_z4

    OPTIONS:
      all        - Process all unregistered brains
      1,2,3      - Process specific brains by number
      1          - Process just brain #1
      reprocess  - Redo a completed brain
      retry      - Retry a failed brain
      q          - Quit

================================================================================
COMMAND LINE OPTIONS
================================================================================
    python 2_brainreg_registration.py              # Interactive mode (default)
    python 2_brainreg_registration.py --inspect    # Just show status, don't process
    python 2_brainreg_registration.py --batch      # Non-interactive, process all pending
    python 2_brainreg_registration.py --channel 0  # Use channel 0 instead of 1
    python 2_brainreg_registration.py --quiet      # Less verbose brainreg output
    python 2_brainreg_registration.py --n-free-cpus 106  # Leave 106 cores free

================================================================================
INPUT (from Script 1)
================================================================================
    349_CNT_01_02/                                   ← Mouse folder
    └── 349_CNT_01_02_1p625x_z4/                     ← Pipeline folder  
        └── 1_Extracted_Channels_from_1_ims_to_brainglobe/
            ├── ch0/                                 ← Background channel
            ├── ch1/                                 ← Signal channel (DEFAULT)
            └── metadata.json                        ← Voxel sizes, etc.

================================================================================
OUTPUT (what this script creates)
================================================================================
    349_CNT_01_02/
    └── 349_CNT_01_02_1p625x_z4/
        └── 2_Registered_Atlas_from_brainreg/
            ├── registered_atlas.tiff       ← Atlas labels in your image space
            ├── registered_hemispheres.tiff ← Left/right hemisphere labels
            ├── boundaries.tiff             ← Region outlines (for overlay)
            ├── brainreg.json               ← Brainreg's parameters
            ├── QC_registration_overview.png ← LOOK AT THIS to verify alignment
            ├── registration_metadata.json   ← Our metadata
            └── _archive/                    ← Previous registration attempts
                └── 20241216_143052/         ← Timestamped backup

================================================================================
VERSIONING / ARCHIVING
================================================================================
When you reprocess a brain (via 'reprocess' or 'retry'), the previous 
registration output is NOT deleted. Instead, it's moved to an _archive 
subfolder with a timestamp. This lets you:

    - Compare different registration attempts
    - Roll back if a new registration is worse
    - Keep a history of what you've tried

To restore an old registration, just move its contents back to the main folder.

================================================================================
HOW LONG DOES IT TAKE?
================================================================================
Each brain takes roughly 10-60 minutes depending on:
    - Image size (more slices = longer)
    - Computer speed
    - Whether the atlas needs to be downloaded (first run only)

The script streams brainreg's progress so you can see what's happening.

================================================================================
QC (QUALITY CONTROL)
================================================================================
After each registration, check the QC_registration_overview.png file!
It shows your brain with the atlas boundaries overlaid. If the boundaries 
don't line up with your brain structures, the registration may have failed.

Common issues:
    - Wrong orientation → boundaries are rotated/flipped
    - Wrong voxel sizes → boundaries are stretched/squished
    - Poor tissue quality → boundaries don't match anatomy

================================================================================
SETTINGS
================================================================================
These can be changed in the script if needed:

    DEFAULT_SIGNAL_CHANNEL = 1      # Which channel to register (ch1 = signal)
    DEFAULT_ATLAS = "allen_mouse_10um"  # Atlas resolution
    DEFAULT_ORIENTATION = "iar"     # inferior-anterior-right
    DEFAULT_N_FREE_CPUS = 106       # CPUs to leave free (for 180-core server)
                                    # Set to None to use brainreg's default

================================================================================
REQUIREMENTS
================================================================================
    pip install brainreg numpy tifffile matplotlib

================================================================================
"""

# =============================================================================
# CHECK REQUIREMENTS
# =============================================================================
import sys
import subprocess

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
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    # Check if brainreg is available as a command
    try:
        result = subprocess.run(['brainreg', '--version'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            missing.append("brainreg")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append("brainreg")
    
    if missing:
        print("=" * 60)
        print("ERROR: Missing required packages:", ", ".join(missing))
        print()
        print("Run this script from Anaconda Prompt:")
        print("    conda activate brainglobe-env")
        print("    python 2_brainreg_registration.py")
        print()
        print("Install missing packages:")
        if "brainreg" in missing:
            print("    pip install brainreg")
        other = [p for p in missing if p != "brainreg"]
        if other:
            print(f"    pip install {' '.join(other)}")
        print("=" * 60)
        input("\nPress Enter to close...")
        sys.exit(1)

_check_requirements()

# =============================================================================
# MAIN SCRIPT
# =============================================================================

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# =============================================================================
# VERSION
# =============================================================================
SCRIPT_VERSION = "1.2.0"

# =============================================================================
# DEFAULT PATHS AND SETTINGS
# =============================================================================
DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Default channel for registration (ch1 = signal channel typically)
DEFAULT_SIGNAL_CHANNEL = 1

# Atlas settings
DEFAULT_ATLAS = "allen_mouse_10um"
DEFAULT_ORIENTATION = "iar"  # inferior-anterior-right

# CPU settings - for servers with many cores, limit brainreg to prevent overload
# Set to number of cores to LEAVE FREE (e.g., 106 on a 180-core server = use 74 cores)
# Set to None to let brainreg use its default behavior
DEFAULT_N_FREE_CPUS = 106

# Pipeline folder names (must match Script 1)
FOLDER_CHANNELS = "1_Extracted_Channels_from_1_ims_to_brainglobe"
FOLDER_REGISTRATION = "2_Registered_Atlas_from_brainreg"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text, char="="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(text)
    print(f"{char * 60}")


def print_step(step_num, total, text):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {text}")
    print("-" * 50)


def find_pipeline_folders(root_path):
    """
    Find all pipeline folders that have completed channel extraction.
    
    Returns list of tuples: (mouse_folder, pipeline_folder, metadata_path)
    """
    root_path = Path(root_path)
    results = []
    
    print(f"Scanning: {root_path}")
    print("Looking for completed channel extractions...")
    
    # Iterate through mouse folders
    for mouse_folder in root_path.iterdir():
        if not mouse_folder.is_dir():
            continue
        if mouse_folder.name.startswith('.') or mouse_folder.name.startswith('_'):
            continue
        
        # Look for pipeline folders inside mouse folder
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            
            # Check for channels folder
            channels_folder = pipeline_folder / FOLDER_CHANNELS
            if not channels_folder.exists():
                continue
            
            # Check for metadata.json
            metadata_path = channels_folder / "metadata.json"
            if not metadata_path.exists():
                print(f"  Warning: {pipeline_folder.name} has channels but no metadata.json")
                continue
            
            # Check for at least one channel folder with TIFFs
            has_channel_data = False
            for ch_folder in channels_folder.iterdir():
                if ch_folder.is_dir() and ch_folder.name.startswith('ch'):
                    tiffs = list(ch_folder.glob('Z*.tif'))
                    if len(tiffs) > 0:
                        has_channel_data = True
                        break
            
            if has_channel_data:
                results.append((mouse_folder, pipeline_folder, metadata_path))
                print(f"  ✓ Found: {mouse_folder.name}/{pipeline_folder.name}")
    
    return results


def load_metadata(metadata_path):
    """Load metadata.json from a pipeline folder."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def check_registration_status(pipeline_folder):
    """
    Check if registration has already been completed.
    
    Returns:
        (status: str, reason: str)
        
        Status can be:
        - "completed": Registration output exists
        - "needs_registration": No output found
        - "incomplete": Partial output (might have failed)
    """
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    
    if not reg_folder.exists():
        return "needs_registration", "registration folder doesn't exist"
    
    # Check for key brainreg outputs
    required_files = [
        "registered_atlas.tiff",
        "brainreg.json"
    ]
    
    missing = []
    for fname in required_files:
        if not (reg_folder / fname).exists():
            missing.append(fname)
    
    if missing:
        # Check if folder is essentially empty (just created by Script 1)
        # vs actually having a failed/partial registration
        contents = list(reg_folder.iterdir())
        # Filter out hidden files, desktop.ini, thumbs.db, and archive folders
        real_contents = [f for f in contents 
                        if not f.name.startswith('.') 
                        and not f.name.startswith('_archive')
                        and f.name.lower() not in ('desktop.ini', 'thumbs.db')]
        
        if len(real_contents) == 0:
            # Empty folder - never attempted registration
            return "needs_registration", "not yet registered"
        else:
            # Has some files but not the required ones - partial/failed
            return "incomplete", f"missing: {', '.join(missing)}"
    
    return "completed", "registration complete"


def archive_existing_registration(reg_folder):
    """
    Archive existing registration output instead of deleting it.
    
    Creates an _archive subfolder with timestamped copies of previous runs.
    This allows comparing different registration attempts.
    
    Returns:
        archive_path if archived, None if nothing to archive
    """
    reg_folder = Path(reg_folder)
    
    if not reg_folder.exists():
        return None
    
    # Check if there's anything worth archiving
    contents = [f for f in reg_folder.iterdir() 
                if not f.name.startswith('_archive') 
                and not f.name.startswith('.')
                and f.name.lower() not in ('desktop.ini', 'thumbs.db')]
    
    if not contents:
        return None
    
    # Create archive folder
    archive_base = reg_folder / "_archive"
    archive_base.mkdir(exist_ok=True)
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_base / timestamp
    archive_path.mkdir()
    
    # Move all non-archive contents to archive
    moved = 0
    for item in contents:
        dest = archive_path / item.name
        shutil.move(str(item), str(dest))
        moved += 1
    
    return archive_path if moved > 0 else None


def get_channel_folder(pipeline_folder, channel):
    """Get the path to a specific channel's folder."""
    return pipeline_folder / FOLDER_CHANNELS / f"ch{channel}"


def run_brainreg(input_folder, output_folder, voxel_sizes, orientation, atlas, 
                 n_free_cpus=None, verbose=True):
    """
    Run brainreg registration.
    
    Args:
        input_folder: Path to channel folder with TIFF slices
        output_folder: Path where brainreg should save output
        voxel_sizes: Dict with 'x', 'y', 'z' in microns
        orientation: Orientation string (e.g., 'iar')
        atlas: Atlas name (e.g., 'allen_mouse_10um')
        n_free_cpus: Number of CPU cores to leave free (None = brainreg default)
        verbose: Whether to show brainreg output
    
    Returns:
        (success: bool, message: str, duration: float)
    """
    # Build the brainreg command
    # Note: brainreg expects voxel sizes as Z Y X
    vz = voxel_sizes['z']
    vy = voxel_sizes['y']
    vx = voxel_sizes['x']
    
    cmd = [
        'brainreg',
        str(input_folder),
        str(output_folder),
        '-v', str(vz), str(vy), str(vx),
        '--orientation', orientation,
        '--atlas', atlas,
    ]
    
    # Add CPU limit if specified
    if n_free_cpus is not None:
        cmd.extend(['--n-free-cpus', str(n_free_cpus)])
    
    print(f"\n    Command: {' '.join(cmd)}")
    print(f"\n    Running brainreg (this may take 10-60 minutes)...")
    print("    " + "=" * 50)
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        # Run brainreg and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if verbose:
                print(f"    | {line}")
                sys.stdout.flush()
        
        process.wait()
        duration = time.time() - start_time
        
        print("    " + "=" * 50)
        
        if process.returncode == 0:
            return True, "brainreg completed successfully", duration
        else:
            return False, f"brainreg failed with return code {process.returncode}", duration
            
    except Exception as e:
        duration = time.time() - start_time
        return False, f"brainreg error: {str(e)}", duration


def generate_qc_images(pipeline_folder, metadata):
    """
    Generate QC images for visual verification of registration.
    
    Creates PNG files showing:
    - Middle slices in all three planes
    - Overlay of brain boundaries on the original image
    """
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    channels_folder = pipeline_folder / FOLDER_CHANNELS
    
    print("\n    Generating QC images...")
    
    qc_images_created = []
    
    try:
        # Load the boundaries image (atlas outlines in sample space)
        boundaries_path = reg_folder / "boundaries.tiff"
        if not boundaries_path.exists():
            print("    Warning: boundaries.tiff not found, skipping QC overlay")
            return qc_images_created
        
        print("    Loading boundaries.tiff...")
        boundaries = tifffile.imread(str(boundaries_path))
        print(f"    Boundaries shape: {boundaries.shape}")
        
        # Load a sample of the original signal channel for overlay
        signal_ch = metadata.get('channels_extracted', [0])[0]
        if len(metadata.get('channels_extracted', [])) > 1:
            signal_ch = 1  # Prefer ch1 if available
        
        ch_folder = channels_folder / f"ch{signal_ch}"
        tiff_files = sorted(ch_folder.glob('Z*.tif'))
        
        if not tiff_files:
            print(f"    Warning: No TIFF files found in ch{signal_ch}")
            return qc_images_created
        
        # Load middle slice for Z
        mid_z = len(tiff_files) // 2
        print(f"    Loading slice Z{mid_z:04d} for QC...")
        mid_slice_z = tifffile.imread(str(tiff_files[mid_z]))
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Registration QC: {pipeline_folder.name}", fontsize=14, fontweight='bold')
        
        # Get dimensions
        nz, ny, nx = boundaries.shape
        mid_z_idx = nz // 2
        mid_y_idx = ny // 2
        mid_x_idx = nx // 2
        
        # Row 1: Original signal channel slices
        # Axial (XY plane at middle Z)
        axes[0, 0].imshow(mid_slice_z, cmap='gray')
        axes[0, 0].set_title(f'Signal Ch{signal_ch} - Axial (Z={mid_z})')
        axes[0, 0].axis('off')
        
        # For coronal and sagittal, we'd need to load more slices
        # For now, show the boundaries in those planes
        axes[0, 1].imshow(boundaries[mid_z_idx, :, :], cmap='gray')
        axes[0, 1].set_title(f'Boundaries - Axial (Z={mid_z_idx})')
        axes[0, 1].axis('off')
        
        axes[0, 2].text(0.5, 0.5, f'Voxel sizes:\nZ: {metadata["voxel_size_um"]["z"]:.2f} µm\n'
                        f'Y: {metadata["voxel_size_um"]["y"]:.2f} µm\n'
                        f'X: {metadata["voxel_size_um"]["x"]:.2f} µm\n\n'
                        f'Orientation: {metadata.get("orientation", "iar")}\n'
                        f'Dimensions: {nx} x {ny} x {nz}',
                        transform=axes[0, 2].transAxes, fontsize=12,
                        verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace')
        axes[0, 2].axis('off')
        axes[0, 2].set_title('Parameters')
        
        # Row 2: Boundary views in different planes
        # Coronal (XZ plane at middle Y)
        axes[1, 0].imshow(boundaries[:, mid_y_idx, :], cmap='gray', aspect='auto')
        axes[1, 0].set_title(f'Boundaries - Coronal (Y={mid_y_idx})')
        axes[1, 0].axis('off')
        
        # Sagittal (YZ plane at middle X)
        axes[1, 1].imshow(boundaries[:, :, mid_x_idx], cmap='gray', aspect='auto')
        axes[1, 1].set_title(f'Boundaries - Sagittal (X={mid_x_idx})')
        axes[1, 1].axis('off')
        
        # Overlay: boundaries on signal (if shapes match)
        if boundaries.shape[0] == len(tiff_files):
            # Load the matching Z slice from boundaries
            boundary_slice = boundaries[mid_z, :, :]
            
            # Resize boundary to match signal if needed
            if boundary_slice.shape != mid_slice_z.shape:
                from scipy.ndimage import zoom
                zoom_factors = (mid_slice_z.shape[0] / boundary_slice.shape[0],
                               mid_slice_z.shape[1] / boundary_slice.shape[1])
                boundary_slice = zoom(boundary_slice, zoom_factors, order=0)
            
            # Create overlay
            axes[1, 2].imshow(mid_slice_z, cmap='gray')
            axes[1, 2].contour(boundary_slice > 0, colors='red', linewidths=0.5, alpha=0.7)
            axes[1, 2].set_title('Overlay: Signal + Boundaries')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, f'Shape mismatch:\n'
                           f'Signal: {len(tiff_files)} slices\n'
                           f'Boundaries: {boundaries.shape[0]} slices',
                           transform=axes[1, 2].transAxes, fontsize=10,
                           verticalalignment='center', horizontalalignment='center')
            axes[1, 2].axis('off')
            axes[1, 2].set_title('Overlay (shape mismatch)')
        
        plt.tight_layout()
        
        # Save QC image
        qc_path = reg_folder / "QC_registration_overview.png"
        plt.savefig(str(qc_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        qc_images_created.append(qc_path)
        print(f"    ✓ Saved: QC_registration_overview.png")
        
    except Exception as e:
        print(f"    Warning: QC image generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return qc_images_created


def save_registration_metadata(pipeline_folder, input_metadata, registration_params, 
                               duration, success, qc_images):
    """Save metadata about the registration run."""
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    
    reg_metadata = {
        'script_version': SCRIPT_VERSION,
        'registration_date': datetime.now().isoformat(),
        'success': success,
        'duration_seconds': round(duration, 1),
        'input_metadata': {
            'source_file': input_metadata.get('source_file'),
            'pipeline_folder': input_metadata.get('pipeline_folder'),
            'voxel_size_um': input_metadata.get('voxel_size_um'),
        },
        'registration_params': registration_params,
        'qc_images': [str(p.name) for p in qc_images],
        'output_folder': str(reg_folder),
    }
    
    metadata_path = reg_folder / "registration_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(reg_metadata, f, indent=2)
    
    print(f"    ✓ Saved: registration_metadata.json")
    
    return metadata_path


# =============================================================================
# SCANNING AND REPORTING
# =============================================================================

def scan_and_report(root_path, signal_channel):
    """
    Scan for pipelines and report their registration status.
    
    Returns:
        (all_pipelines, needs_registration, completed, incomplete)
    """
    pipelines = find_pipeline_folders(root_path)
    
    all_pipelines = []
    needs_registration = []
    completed = []
    incomplete = []
    
    for mouse_folder, pipeline_folder, metadata_path in pipelines:
        status, reason = check_registration_status(pipeline_folder)
        
        # Check if the requested channel exists
        ch_folder = get_channel_folder(pipeline_folder, signal_channel)
        if not ch_folder.exists():
            # Try to find an available channel
            channels_folder = pipeline_folder / FOLDER_CHANNELS
            available_channels = [d.name for d in channels_folder.iterdir() 
                                 if d.is_dir() and d.name.startswith('ch')]
            if not available_channels:
                status = "no_channels"
                reason = "no channel folders found"
            else:
                reason += f" (ch{signal_channel} not found, available: {', '.join(available_channels)})"
        
        entry = (mouse_folder, pipeline_folder, metadata_path, status, reason)
        all_pipelines.append(entry)
        
        if status == "needs_registration":
            needs_registration.append(entry)
        elif status == "completed":
            completed.append(entry)
        elif status == "incomplete":
            incomplete.append(entry)
    
    return all_pipelines, needs_registration, completed, incomplete


def print_scan_results(all_pipelines, needs_registration, completed, incomplete, signal_channel):
    """Print a summary of the scan results."""
    print(f"\nFound {len(all_pipelines)} pipeline(s) with extracted channels:\n")
    
    for mouse_folder, pipeline_folder, metadata_path, status, reason in all_pipelines:
        rel_path = f"{mouse_folder.name}/{pipeline_folder.name}"
        
        if status == "completed":
            print(f"  ✓ {rel_path}")
        elif status == "needs_registration":
            print(f"  ○ {rel_path} (needs registration)")
        elif status == "incomplete":
            print(f"  ⚠ {rel_path} ({reason})")
        else:
            print(f"  ✗ {rel_path} ({reason})")
    
    print(f"\nUsing channel: ch{signal_channel}")
    print(f"\nSummary:")
    print(f"  - Ready to register: {len(needs_registration)}")
    print(f"  - Already completed: {len(completed)}")
    print(f"  - Incomplete/failed: {len(incomplete)}")
    
    return len(needs_registration)


# =============================================================================
# INTERACTIVE SELECTION
# =============================================================================

def interactive_select_pipelines(needs_registration, completed, incomplete, all_pipelines):
    """
    Interactive menu to select which pipelines to process.
    
    Returns:
        List of pipelines to process (same format as needs_registration)
    """
    while True:
        print("\n" + "=" * 60)
        print("PIPELINE SELECTION")
        print("=" * 60)
        
        # Show unprocessed brains
        if needs_registration:
            print(f"\n[READY TO REGISTER] ({len(needs_registration)} brain(s)):")
            for idx, (mouse, pipeline, meta, status, reason) in enumerate(needs_registration):
                print(f"  {idx + 1}. {mouse.name}/{pipeline.name}")
        else:
            print("\n[READY TO REGISTER] None - all brains are registered!")
        
        # Show completed brains
        if completed:
            print(f"\n[ALREADY COMPLETED] ({len(completed)} brain(s)):")
            for idx, (mouse, pipeline, meta, status, reason) in enumerate(completed):
                print(f"  C{idx + 1}. {mouse.name}/{pipeline.name}")
        
        # Show incomplete/failed
        if incomplete:
            print(f"\n[INCOMPLETE/FAILED] ({len(incomplete)} brain(s)):")
            for idx, (mouse, pipeline, meta, status, reason) in enumerate(incomplete):
                print(f"  F{idx + 1}. {mouse.name}/{pipeline.name} - {reason}")
        
        # Options
        print("\n" + "-" * 60)
        print("OPTIONS:")
        if needs_registration:
            print("  all        - Process all unregistered brains")
            print("  1,2,3      - Process specific brains by number (comma-separated)")
            print("  1          - Process just brain #1")
        if completed:
            print("  reprocess  - Choose from completed brains to redo")
        if incomplete:
            print("  retry      - Choose from failed/incomplete to retry")
        print("  q          - Quit without processing")
        print("-" * 60)
        
        choice = input("\nYour choice: ").strip().lower()
        
        # Handle quit
        if choice == 'q' or choice == 'quit':
            return []
        
        # Handle 'all'
        if choice == 'all':
            if not needs_registration:
                print("\nNo unregistered brains to process!")
                continue
            return needs_registration
        
        # Handle 'reprocess'
        if choice == 'reprocess':
            if not completed:
                print("\nNo completed brains to reprocess!")
                continue
            return _select_from_list(completed, "REPROCESS")
        
        # Handle 'retry'
        if choice == 'retry':
            if not incomplete:
                print("\nNo incomplete/failed brains to retry!")
                continue
            return _select_from_list(incomplete, "RETRY")
        
        # Handle number selection (e.g., "1" or "1,3,5")
        if choice and needs_registration:
            try:
                # Parse numbers
                if ',' in choice:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                else:
                    indices = [int(choice) - 1]
                
                # Validate indices
                selected = []
                invalid = []
                for i in indices:
                    if 0 <= i < len(needs_registration):
                        selected.append(needs_registration[i])
                    else:
                        invalid.append(i + 1)
                
                if invalid:
                    print(f"\nInvalid number(s): {invalid}. Valid range: 1-{len(needs_registration)}")
                    continue
                
                if selected:
                    # Confirm selection
                    print(f"\nSelected {len(selected)} brain(s):")
                    for mouse, pipeline, meta, status, reason in selected:
                        print(f"  - {mouse.name}/{pipeline.name}")
                    
                    confirm = input("\nProceed with these? [Enter=yes, n=no]: ").strip().lower()
                    if confirm != 'n':
                        return selected
                    continue
                    
            except ValueError:
                print(f"\nCouldn't parse '{choice}'. Enter numbers like: 1 or 1,3,5")
                continue
        
        print("\nInvalid choice. Please try again.")


def _select_from_list(pipeline_list, action_name):
    """Helper to select from a list of pipelines (for reprocess/retry)."""
    print(f"\n{action_name} - Select brain(s):")
    for idx, (mouse, pipeline, meta, status, reason) in enumerate(pipeline_list):
        print(f"  {idx + 1}. {mouse.name}/{pipeline.name}")
    
    print("\nEnter number(s) or 'all', 'back' to go back:")
    choice = input("Your choice: ").strip().lower()
    
    if choice == 'back' or choice == 'b':
        return None  # Signal to go back to main menu
    
    if choice == 'all':
        return pipeline_list
    
    try:
        if ',' in choice:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
        else:
            indices = [int(choice) - 1]
        
        selected = []
        for i in indices:
            if 0 <= i < len(pipeline_list):
                selected.append(pipeline_list[i])
        
        return selected if selected else None
        
    except ValueError:
        print("Invalid selection.")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run brainreg atlas registration on extracted channel data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {SCRIPT_VERSION}

This script registers brain images to the Allen Mouse Brain Atlas.
It uses output from Script 1 (1_ims_to_brainglobe.py).

Examples:
  python 2_brainreg_registration.py              # Interactive mode
  python 2_brainreg_registration.py --inspect    # Just show status
  python 2_brainreg_registration.py --channel 0  # Use channel 0
  python 2_brainreg_registration.py --batch      # Non-interactive, process all
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help=f'Path to scan (default: {DEFAULT_BRAINGLOBE_ROOT})')
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Only scan and report, do not process')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Non-interactive batch mode - process all pending')
    parser.add_argument('--channel', '-c', type=int, default=DEFAULT_SIGNAL_CHANNEL,
                        help=f'Channel to use for registration (default: {DEFAULT_SIGNAL_CHANNEL})')
    parser.add_argument('--atlas', '-a', default=DEFAULT_ATLAS,
                        help=f'Atlas to use (default: {DEFAULT_ATLAS})')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce brainreg output verbosity')
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS,
                        help=f'Number of CPU cores to leave unused (default: {DEFAULT_N_FREE_CPUS})')
    
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
    
    print_header("BrainReg Atlas Registration")
    print(f"Version: {SCRIPT_VERSION}")
    print(f"Atlas: {args.atlas}")
    print(f"Orientation: {DEFAULT_ORIENTATION}")
    print(f"Signal Channel: ch{args.channel}")
    if args.n_free_cpus:
        print(f"CPU Limit: leaving {args.n_free_cpus} cores free")
    
    # Scan for pipelines
    all_pipelines, needs_registration, completed, incomplete = scan_and_report(
        root_path, args.channel
    )
    
    # Quick summary
    print(f"\nFound {len(all_pipelines)} pipeline(s):")
    print(f"  - Ready to register: {len(needs_registration)}")
    print(f"  - Already completed: {len(completed)}")
    print(f"  - Incomplete/failed: {len(incomplete)}")
    
    if args.inspect:
        # Detailed status report
        print("\nDetailed status:")
        for mouse, pipeline, meta, status, reason in all_pipelines:
            rel_path = f"{mouse.name}/{pipeline.name}"
            if status == "completed":
                print(f"  ✓ {rel_path}")
            elif status == "needs_registration":
                print(f"  ○ {rel_path}")
            else:
                print(f"  ⚠ {rel_path} ({reason})")
        return
    
    if len(all_pipelines) == 0:
        print("\nNo pipelines found. Run Script 1 first to extract channels.")
        return
    
    # Select pipelines to process
    if args.batch:
        # Non-interactive: process all pending
        to_process = needs_registration
        if not to_process:
            print("\nNo brains need registration. Use --inspect to see status.")
            return
        print(f"\nBatch mode: processing {len(to_process)} brain(s)")
    else:
        # Interactive selection
        to_process = interactive_select_pipelines(
            needs_registration, completed, incomplete, all_pipelines
        )
        
        if to_process is None:
            # User selected 'back' from submenu, restart selection
            to_process = interactive_select_pipelines(
                needs_registration, completed, incomplete, all_pipelines
            )
        
        if not to_process:
            print("\nNo brains selected. Exiting.")
            return
    
    n_to_process = len(to_process)
    print_header(f"Processing {n_to_process} brain(s)", char="-")
    
    success_count = 0
    failed_count = 0
    
    for idx, (mouse_folder, pipeline_folder, metadata_path, status, reason) in enumerate(to_process):
        rel_path = f"{mouse_folder.name}/{pipeline_folder.name}"
        print_step(idx + 1, n_to_process, rel_path)
        
        try:
            # Load metadata
            print("    Loading metadata...")
            metadata = load_metadata(metadata_path)
            
            voxel_sizes = metadata.get('voxel_size_um', {})
            if not all(k in voxel_sizes for k in ['x', 'y', 'z']):
                print(f"    ✗ Error: Missing voxel size info in metadata")
                failed_count += 1
                continue
            
            print(f"    Voxel sizes (µm): Z={voxel_sizes['z']:.2f}, Y={voxel_sizes['y']:.2f}, X={voxel_sizes['x']:.2f}")
            
            # Get channel folder
            ch_folder = get_channel_folder(pipeline_folder, args.channel)
            if not ch_folder.exists():
                # Try fallback to ch0
                ch_folder = get_channel_folder(pipeline_folder, 0)
                if not ch_folder.exists():
                    print(f"    ✗ Error: No channel folder found")
                    failed_count += 1
                    continue
                print(f"    Note: Using ch0 (ch{args.channel} not found)")
            
            # Check channel has data
            tiff_files = list(ch_folder.glob('Z*.tif'))
            print(f"    Input: {ch_folder.name}/ ({len(tiff_files)} slices)")
            
            if len(tiff_files) == 0:
                print(f"    ✗ Error: No TIFF slices found in {ch_folder.name}")
                failed_count += 1
                continue
            
            # Set up output folder
            reg_folder = pipeline_folder / FOLDER_REGISTRATION
            
            # If reprocessing (completed or incomplete), archive existing output
            if status in ("completed", "incomplete") and reg_folder.exists():
                archive_path = archive_existing_registration(reg_folder)
                if archive_path:
                    print(f"    Archived previous registration to: _archive/{archive_path.name}/")
            
            reg_folder.mkdir(parents=True, exist_ok=True)
            print(f"    Output: {FOLDER_REGISTRATION}/")
            
            # Registration parameters
            reg_params = {
                'atlas': args.atlas,
                'orientation': DEFAULT_ORIENTATION,
                'voxel_sizes_um': voxel_sizes,
                'input_channel': ch_folder.name,
            }
            
            # Run brainreg
            success, message, duration = run_brainreg(
                input_folder=ch_folder,
                output_folder=reg_folder,
                voxel_sizes=voxel_sizes,
                orientation=DEFAULT_ORIENTATION,
                atlas=args.atlas,
                n_free_cpus=args.n_free_cpus,
                verbose=not args.quiet
            )
            
            print(f"\n    Duration: {duration/60:.1f} minutes")
            
            if success:
                print(f"    ✓ {message}")
                
                # Generate QC images
                qc_images = generate_qc_images(pipeline_folder, metadata)
                
                # Save our metadata
                save_registration_metadata(
                    pipeline_folder, metadata, reg_params, duration, success, qc_images
                )
                
                success_count += 1
            else:
                print(f"    ✗ {message}")
                failed_count += 1
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # Summary
    print_header("Complete")
    print(f"Processed: {success_count + failed_count}")
    print(f"  - Succeeded: {success_count}")
    print(f"  - Failed: {failed_count}")
    
    if success_count > 0:
        print(f"\nCheck QC images in each {FOLDER_REGISTRATION}/ folder")
        print("Look for: QC_registration_overview.png")


if __name__ == '__main__':
    main()
    
    # If double-clicked (no args passed), pause so window doesn't close
    if len(sys.argv) == 1:
        print()
        input("Press Enter to close...")
