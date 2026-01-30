#!/usr/bin/env python3
"""
run_registration.py

Wrapper for brainreg atlas registration that auto-logs to experiment tracker.

This registers your brain sample to an atlas (typically Allen Mouse Brain)
using the BrainGlobe brainreg tool.

================================================================================
WHAT IT DOES
================================================================================
1. Takes cropped brain data (from 2_Cropped_For_Registration/)
2. Runs brainreg to register it to the Allen Mouse Brain Atlas
3. Outputs registered atlas annotations and transforms
4. Logs everything to the experiment tracker

================================================================================
USAGE
================================================================================
Basic (uses settings from metadata.json):
    python run_registration.py --brain 349_CNT_01_02_1p625x_z4

With custom atlas:
    python run_registration.py --brain 349_CNT_01_02_1p625x_z4 --atlas allen_mouse_50um

With custom voxel sizes (overrides metadata):
    python run_registration.py --brain 349_CNT_01_02_1p625x_z4 -v 4.0 4.0 4.0

Dry run:
    python run_registration.py --brain 349_CNT_01_02_1p625x_z4 --dry-run

================================================================================
OUTPUT
================================================================================
Results go to: {pipeline}/3_Registered_Atlas/
    - registered_atlas.tiff     : Atlas annotations in sample space
    - registered_hemispheres.tiff : Left/right hemisphere labels
    - deformation_field_*.tiff  : Transform fields
    - brainreg.json             : Settings used

================================================================================
REQUIREMENTS
================================================================================
    conda activate brainglobe-env
    # brainreg should already be installed
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Pipeline folder names
FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"

# Default registration settings
DEFAULT_ATLAS = "allen_mouse_25um"
DEFAULT_ORIENTATION = "iar"  # inferior-anterior-right

# Available atlases (common ones)
COMMON_ATLASES = [
    "allen_mouse_25um",
    "allen_mouse_50um",
    "allen_mouse_10um",
    "kim_unified_25um",
    "osten_mouse_25um",
]


# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT) -> Optional[Path]:
    """Find a brain's pipeline folder."""
    root = Path(root)
    
    # Direct match
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name == pipeline_folder.name:
                return pipeline_folder
            if brain_name == f"{mouse_folder.name}/{pipeline_folder.name}":
                return pipeline_folder
    
    # Partial match
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name in pipeline_folder.name:
                return pipeline_folder
    
    return None


def get_input_data(pipeline_folder: Path, channel: int = 0) -> Tuple[Optional[Path], str]:
    """
    Get input data folder, preferring cropped data over full extracted.
    
    Returns:
        (path_to_channel, source_name)
    """
    # Prefer cropped data
    cropped = pipeline_folder / FOLDER_CROPPED / f"ch{channel}"
    if cropped.exists() and len(list(cropped.glob("Z*.tif"))) > 0:
        return cropped, "cropped"
    
    # Fall back to full extracted
    extracted = pipeline_folder / FOLDER_EXTRACTED / f"ch{channel}"
    if extracted.exists() and len(list(extracted.glob("Z*.tif"))) > 0:
        return extracted, "full"
    
    return None, "none"


def get_metadata(pipeline_folder: Path) -> Dict:
    """Load metadata, checking both cropped and extracted folders."""
    # Try cropped first
    cropped_meta = pipeline_folder / FOLDER_CROPPED / "metadata.json"
    if cropped_meta.exists():
        with open(cropped_meta, 'r') as f:
            return json.load(f)
    
    # Fall back to extracted
    extracted_meta = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    if extracted_meta.exists():
        with open(extracted_meta, 'r') as f:
            return json.load(f)
    
    return {}


# =============================================================================
# REGISTRATION
# =============================================================================

def run_brainreg(
    input_folder: Path,
    output_folder: Path,
    voxel_sizes: Tuple[float, float, float],
    orientation: str = DEFAULT_ORIENTATION,
    atlas: str = DEFAULT_ATLAS,
    additional_args: str = None,
) -> Tuple[bool, float, str]:
    """
    Run brainreg registration.
    
    Args:
        input_folder: Path to input channel folder
        output_folder: Path for registration output
        voxel_sizes: (z, y, x) voxel sizes in microns
        orientation: Three-letter orientation code
        atlas: Atlas name
        additional_args: Extra arguments to pass to brainreg
    
    Returns:
        (success, duration_seconds, error_message)
    """
    vz, vy, vx = voxel_sizes
    
    # Build command
    cmd = [
        "brainreg",
        str(input_folder),
        str(output_folder),
        "-v", str(vz), str(vy), str(vx),
        "--orientation", orientation,
        "--atlas", atlas,
    ]
    
    # Add any extra arguments
    if additional_args:
        cmd.extend(additional_args.split())
    
    print(f"\nRunning: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output show in terminal
            text=True,
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return True, duration, ""
        else:
            return False, duration, f"Exit code: {result.returncode}"
    
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def verify_registration(output_folder: Path) -> bool:
    """Check if registration produced expected outputs."""
    expected_files = [
        "registered_atlas.tiff",
        "brainreg.json",
    ]
    
    for filename in expected_files:
        if not (output_folder / filename).exists():
            return False
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run brainreg atlas registration with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (reads voxel sizes from metadata)
  python run_registration.py --brain 349_CNT_01_02_1p625x_z4

  # With custom atlas
  python run_registration.py --brain 349_CNT_01_02_1p625x_z4 --atlas allen_mouse_50um

  # Override voxel sizes
  python run_registration.py --brain 349_CNT_01_02_1p625x_z4 -v 4.0 4.0 4.0

  # Custom orientation
  python run_registration.py --brain 349_CNT_01_02_1p625x_z4 --orientation sal

  # Dry run
  python run_registration.py --brain 349_CNT_01_02_1p625x_z4 --dry-run

Available atlases:
  allen_mouse_25um   (default, high resolution)
  allen_mouse_50um   (faster, good for testing)
  allen_mouse_10um   (highest resolution, slow)
  kim_unified_25um
  osten_mouse_25um
        """
    )
    
    # Required
    parser.add_argument('--brain', '-b', required=True,
                        help='Brain/pipeline to process')
    
    # Registration settings
    parser.add_argument('--atlas', '-a', default=None,
                        help=f'Atlas to use (default: {DEFAULT_ATLAS})')
    parser.add_argument('--orientation', '-o', default=None,
                        help=f'Orientation code (default: {DEFAULT_ORIENTATION})')
    parser.add_argument('-v', '--voxel-sizes', nargs=3, type=float,
                        metavar=('Z', 'Y', 'X'),
                        help='Voxel sizes in microns (overrides metadata)')
    parser.add_argument('--args', help='Additional brainreg arguments')
    
    # Options
    parser.add_argument('--channel', '-c', type=int, default=0,
                        help='Channel to register (default: 0)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Re-run even if registration exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without running')
    parser.add_argument('--notes', help='Notes to add to experiment log')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT,
                        help='Root folder for brains')
    
    args = parser.parse_args()
    
    # Find brain
    pipeline_folder = find_brain(args.brain, args.root)
    if not pipeline_folder:
        print(f"Error: Brain not found: {args.brain}")
        print(f"Searched in: {args.root}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Atlas Registration")
    print(f"{'='*60}")
    print(f"Brain: {pipeline_folder.name}")
    print(f"Pipeline: {pipeline_folder}")
    
    # Get input data
    input_folder, data_source = get_input_data(pipeline_folder, args.channel)
    if not input_folder:
        print(f"\nError: No input data found")
        print(f"  Checked: {pipeline_folder / FOLDER_CROPPED}")
        print(f"  Checked: {pipeline_folder / FOLDER_EXTRACTED}")
        print(f"\nRun crop optimization or extraction first.")
        sys.exit(1)
    
    print(f"Input: {input_folder}")
    print(f"Source: {data_source}")
    
    # Get metadata
    metadata = get_metadata(pipeline_folder)
    
    # Determine voxel sizes
    if args.voxel_sizes:
        voxel_sizes = tuple(args.voxel_sizes)
        print(f"Voxel sizes (from args): {voxel_sizes}")
    else:
        voxel = metadata.get('voxel_size_um', {})
        voxel_sizes = (
            voxel.get('z', 4.0),
            voxel.get('y', 4.0),
            voxel.get('x', 4.0),
        )
        print(f"Voxel sizes (from metadata): {voxel_sizes}")
    
    # Determine orientation
    orientation = args.orientation or metadata.get('orientation', DEFAULT_ORIENTATION)
    print(f"Orientation: {orientation}")
    
    # Determine atlas
    atlas = args.atlas or DEFAULT_ATLAS
    print(f"Atlas: {atlas}")
    
    # Check output folder
    output_folder = pipeline_folder / FOLDER_REGISTRATION
    if output_folder.exists() and verify_registration(output_folder):
        if not args.force:
            print(f"\nRegistration already exists: {output_folder}")
            print("Use --force to re-run")
            sys.exit(0)
        else:
            print(f"\nRe-running registration (--force)")
    
    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would run brainreg with:")
        print(f"  Input: {input_folder}")
        print(f"  Output: {output_folder}")
        print(f"  Voxel sizes: {voxel_sizes}")
        print(f"  Orientation: {orientation}")
        print(f"  Atlas: {atlas}")
        if args.args:
            print(f"  Extra args: {args.args}")
        sys.exit(0)
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Log experiment start
    exp_id = tracker.log_registration(
        brain=pipeline_folder.name,
        atlas=atlas,
        orientation=orientation,
        voxel_z=voxel_sizes[0],
        voxel_y=voxel_sizes[1],
        voxel_x=voxel_sizes[2],
        output_path=str(output_folder),
        additional_args=args.args,
        status="started",
        notes=args.notes,
    )
    
    print(f"\nExperiment: {exp_id}")
    
    # Run registration
    output_folder.mkdir(parents=True, exist_ok=True)
    
    success, duration, error = run_brainreg(
        input_folder=input_folder,
        output_folder=output_folder,
        voxel_sizes=voxel_sizes,
        orientation=orientation,
        atlas=atlas,
        additional_args=args.args,
    )
    
    # Update experiment status
    if success and verify_registration(output_folder):
        tracker.update_status(
            exp_id,
            status="completed",
            duration_seconds=round(duration, 1),
        )
        
        print(f"\n{'='*60}")
        print(f"REGISTRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Output: {output_folder}")
        print(f"Experiment: {exp_id}")
        
        # Prompt for rating
        print()
        try:
            rating = input("Rate this registration (1-5, or Enter to skip): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add a note (or Enter to skip): ").strip()
                tracker.rate_experiment(exp_id, int(rating), note if note else None)
        except (EOFError, KeyboardInterrupt):
            pass
    
    else:
        tracker.update_status(
            exp_id,
            status="failed",
            duration_seconds=round(duration, 1),
            notes=f"Error: {error}" if error else "Registration verification failed",
        )
        
        print(f"\n{'='*60}")
        print(f"REGISTRATION FAILED")
        print(f"{'='*60}")
        print(f"Duration: {duration/60:.1f} minutes")
        if error:
            print(f"Error: {error}")
        print(f"Experiment: {exp_id}")
        sys.exit(1)


if __name__ == '__main__':
    main()
