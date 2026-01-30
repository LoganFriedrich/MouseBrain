#!/usr/bin/env python3
"""
4_detect_cells.py

Script 4 in the BrainGlobe pipeline: Cell candidate detection.

Wraps cellfinder detection with presets and auto-logs to experiment tracker.
Run this AFTER Script 3 (register_to_atlas.py).

This runs cellfinder's cell candidate detection with preset parameter
combinations or custom values, and automatically logs everything.

================================================================================
HOW TO USE
================================================================================
Interactive mode (recommended for first runs):
    python 4_detect_cells.py

With presets:
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset sensitive
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset balanced
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset conservative

Custom parameters:
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --ball-xy 6 --ball-z 15

================================================================================
PRESETS
================================================================================
    sensitive    - Catches more cells, more false positives
                   ball_xy=4, ball_z=10, soma=12, threshold=8
                   
    balanced     - Good default starting point
                   ball_xy=6, ball_z=15, soma=16, threshold=10
                   
    conservative - Fewer false positives, may miss dim cells
                   ball_xy=8, ball_z=20, soma=20, threshold=12
                   
    large_cells  - For larger neurons (motor neurons, Purkinje, etc.)
                   ball_xy=10, ball_z=25, soma=25, threshold=10

================================================================================
REQUIREMENTS
================================================================================
    - cellfinder must be installed
    - Registered atlas in 3_Registered_Atlas folder
    - experiment_tracker.py in same directory or PYTHONPATH
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from experiment_tracker import ExperimentTracker
except ImportError:
    print("ERROR: experiment_tracker.py not found!")
    print("Make sure it's in the same directory as this script.")
    sys.exit(1)

# =============================================================================
# VERIFY CELLFINDER AVAILABILITY
# =============================================================================
import os

def check_cellfinder_api_available():
    """Check if cellfinder Python API is available (replaces deprecated CLI check)."""
    try:
        from cellfinder.core.detect.detect import main as detect_main
        from brainglobe_utils.IO.image.load import read_with_dask
        from brainglobe_utils.IO.cells import save_cells
        return True
    except ImportError as e:
        print("=" * 70)
        print("ERROR: cellfinder Python API not found!")
        print("=" * 70)
        print()
        print(f"Import error: {e}")
        print()
        print("Make sure cellfinder is installed: pip install cellfinder")
        print(f"Current Python: {sys.executable}")
        print()
        sys.exit(1)

# Check cellfinder on import (using Python API now, not deprecated CLI)
check_cellfinder_api_available()

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.1"

from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT, parse_brain_name

# Pipeline folders (must match other scripts)
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"

# Detection presets
PRESETS = {
    'sensitive': {
        'description': 'Catches more cells, more false positives',
        'ball_xy_size': 4,
        'ball_z_size': 10,
        'soma_diameter': 12,
        'threshold': 8,
    },
    'balanced': {
        'description': 'Good default starting point',
        'ball_xy_size': 6,
        'ball_z_size': 15,
        'soma_diameter': 16,
        'threshold': 10,
    },
    'conservative': {
        'description': 'Fewer false positives, may miss dim cells',
        'ball_xy_size': 8,
        'ball_z_size': 20,
        'soma_diameter': 20,
        'threshold': 12,
    },
    'large_cells': {
        'description': 'For larger neurons (motor, Purkinje, etc.)',
        'ball_xy_size': 10,
        'ball_z_size': 25,
        'soma_diameter': 25,
        'threshold': 10,
    },
}

DEFAULT_N_FREE_CPUS = 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def find_pipeline(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """
    Find pipeline folder for a brain.
    
    Args:
        brain_name: Either just the pipeline name (349_CNT_01_02_1p625x_z4)
                   or mouse/pipeline format
    
    Returns:
        (pipeline_folder, mouse_folder, metadata) or (None, None, None)
    """
    root = Path(root)
    
    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            # Match by pipeline name or full path
            full_name = f"{mouse_dir.name}/{pipeline_dir.name}"
            if brain_name in [pipeline_dir.name, full_name]:
                # Check for registration
                reg_folder = pipeline_dir / FOLDER_REGISTRATION
                crop_folder = pipeline_dir / FOLDER_CROPPED
                
                if not (reg_folder / "brainreg.json").exists():
                    print(f"Warning: No registration found for {brain_name}")
                    print("Run Script 3 (register_to_atlas.py) first!")
                    return None, None, None
                
                # Load metadata
                metadata_path = crop_folder / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                return pipeline_dir, mouse_dir, metadata
    
    return None, None, None


def list_available_brains(root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """List all brains that can be processed."""
    root = Path(root)
    brains = []
    
    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            reg_folder = pipeline_dir / FOLDER_REGISTRATION
            det_folder = pipeline_dir / FOLDER_DETECTION
            
            has_registration = (reg_folder / "brainreg.json").exists()
            has_detection = det_folder.exists() and len(list(det_folder.glob("*.xml"))) > 0
            
            if has_registration:
                brains.append({
                    'name': f"{mouse_dir.name}/{pipeline_dir.name}",
                    'pipeline': pipeline_dir,
                    'mouse': mouse_dir,
                    'registered': True,
                    'detected': has_detection,
                })
    
    return brains


def count_cells_in_xml(xml_path: Path) -> int:
    """Count cells in a cellfinder XML file."""
    if not xml_path.exists():
        return 0
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        return content.count('<Marker>')
    except:
        return 0


def run_cellfinder_detect(
    signal_path: Path,
    background_path: Path,
    output_path: Path,
    voxel_sizes: tuple,
    params: dict,
    n_free_cpus: int = 2,
) -> tuple:
    """
    Run cellfinder detection using Python API (not deprecated CLI).

    Returns:
        (success, duration, cells_found)
    """
    # Import cellfinder Python API
    from cellfinder.core.detect.detect import main as detect_main
    from brainglobe_utils.IO.image.load import read_with_dask
    from brainglobe_utils.IO.cells import save_cells

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{timestamp()}] Running cellfinder detection...")
    print(f"    Signal: {signal_path}")
    print(f"    Background: {background_path}")
    print(f"    Output: {output_path}")
    print(f"    Parameters:")
    print(f"        ball_xy_size: {params['ball_xy_size']}")
    print(f"        ball_z_size: {params['ball_z_size']}")
    print(f"        soma_diameter: {params['soma_diameter']}")
    print(f"        threshold: {params['threshold']}")
    print()

    start_time = time.time()

    try:
        # Load signal array with dask (memory efficient)
        print(f"    Loading signal data from {signal_path}...")
        signal_array = read_with_dask(str(signal_path))
        print(f"    Signal array shape: {signal_array.shape}")

        # Run detection using Python API
        print(f"    Running detection...")
        cells = detect_main(
            signal_array,
            start_plane=0,
            end_plane=-1,  # All planes
            voxel_sizes=(float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2])),
            soma_diameter=float(params['soma_diameter']),
            ball_xy_size=float(params['ball_xy_size']),
            ball_z_size=float(params['ball_z_size']),
            n_sds_above_mean_thresh=float(params['threshold']),  # This is the threshold parameter
            n_free_cpus=n_free_cpus,
        )

        duration = time.time() - start_time
        cells_found = len(cells)

        # Save results to XML
        cells_xml = output_path / "detected_cells.xml"
        save_cells(cells, str(cells_xml))
        print(f"    Saved {cells_found} cells to {cells_xml}")

        return True, duration, cells_found

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, time.time() - start_time, 0


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_select_brain(brains):
    """Interactive brain selection."""
    print("\n" + "=" * 60)
    print("AVAILABLE BRAINS")
    print("=" * 60)
    
    ready = []
    already_done = []
    
    for i, brain in enumerate(brains):
        if brain['detected']:
            already_done.append((i, brain['name']))
        else:
            ready.append((i, brain['name']))
    
    if ready:
        print("\n[READY FOR DETECTION]")
        for idx, name in ready:
            print(f"  {idx + 1}. {name}")
    
    if already_done:
        print("\n[ALREADY DETECTED]")
        for idx, name in already_done:
            print(f"  {idx + 1}. {name} (has results)")
    
    print("\n" + "-" * 60)
    print("Enter number to select, or 'q' to quit")
    print("-" * 60)
    
    while True:
        response = input("\nSelection: ").strip()
        
        if response.lower() == 'q':
            return None
        
        try:
            idx = int(response) - 1
            if 0 <= idx < len(brains):
                return brains[idx]
            else:
                print(f"Invalid number. Enter 1-{len(brains)}")
        except ValueError:
            print("Enter a number or 'q'")


def interactive_select_preset():
    """Interactive preset selection."""
    print("\n" + "=" * 60)
    print("DETECTION PRESETS")
    print("=" * 60)
    
    for i, (name, preset) in enumerate(PRESETS.items()):
        print(f"\n  {i + 1}. {name}")
        print(f"     {preset['description']}")
        print(f"     ball_xy={preset['ball_xy_size']}, ball_z={preset['ball_z_size']}, "
              f"soma={preset['soma_diameter']}, threshold={preset['threshold']}")
    
    print(f"\n  {len(PRESETS) + 1}. custom (enter your own parameters)")
    
    print("\n" + "-" * 60)
    
    while True:
        response = input("\nSelect preset (1-5): ").strip()
        
        try:
            idx = int(response)
            if 1 <= idx <= len(PRESETS):
                preset_name = list(PRESETS.keys())[idx - 1]
                return preset_name, PRESETS[preset_name].copy()
            elif idx == len(PRESETS) + 1:
                # Custom
                params = {}
                params['ball_xy_size'] = int(input("  ball_xy_size (default 6): ").strip() or "6")
                params['ball_z_size'] = int(input("  ball_z_size (default 15): ").strip() or "15")
                params['soma_diameter'] = int(input("  soma_diameter (default 16): ").strip() or "16")
                params['threshold'] = int(input("  threshold (default 10): ").strip() or "10")
                return "custom", params
            else:
                print(f"Enter 1-{len(PRESETS) + 1}")
        except ValueError:
            print("Enter a number")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder detection with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
    sensitive     - ball_xy=4, ball_z=10, soma=12, threshold=8
    balanced      - ball_xy=6, ball_z=15, soma=16, threshold=10 (default)
    conservative  - ball_xy=8, ball_z=20, soma=20, threshold=12
    large_cells   - ball_xy=10, ball_z=25, soma=25, threshold=10

Examples:
    python 4_detect_cells.py                                    # Interactive mode
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset balanced
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --ball-xy 5 --ball-z 12
        """
    )
    
    parser.add_argument('--brain', '-b', help='Brain/pipeline to process')
    parser.add_argument('--preset', '-p', choices=list(PRESETS.keys()),
                        help='Parameter preset')
    
    # Custom parameters
    parser.add_argument('--ball-xy', type=int, help='Ball filter XY size')
    parser.add_argument('--ball-z', type=int, help='Ball filter Z size')
    parser.add_argument('--soma-diameter', type=int, help='Expected soma diameter')
    parser.add_argument('--threshold', type=int, help='Detection threshold')
    
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS)
    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    parser.add_argument('--routine', action='store_true',
                        help='Routine processing mode: auto-use paradigm-best settings if available')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrainGlobe Cell Detection")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Select brain
    if args.brain:
        pipeline_folder, mouse_folder, metadata = find_pipeline(args.brain, args.root)
        if not pipeline_folder:
            print(f"ERROR: Brain not found: {args.brain}")
            sys.exit(1)
        brain_name = f"{mouse_folder.name}/{pipeline_folder.name}"

        # Check registration approval
        reg_folder = pipeline_folder / FOLDER_REGISTRATION
        approval_file = reg_folder / ".registration_approved"
        if not approval_file.exists():
            print("\n" + "="*60)
            print("WARNING: REGISTRATION NOT YET APPROVED")
            print("="*60)
            print("\nBefore detecting cells, you must review and approve registration QC.")
            print("\nSteps:")
            print("  1. Review QC images:")
            print(f"     {reg_folder / 'QC_registration_detailed.png'}")
            print("  2. If registration looks good, approve it:")
            print(f"     python util_approve_registration.py --brain {args.brain}")
            print("\nThis ensures registration quality before expensive cell detection.")
            print("="*60)
            sys.exit(1)
    else:
        # Interactive selection
        brains = list_available_brains(args.root)
        if not brains:
            print("\nNo registered brains found!")
            print("Run Script 3 (3_register_to_atlas.py) first.")
            sys.exit(1)
        
        brain_info = interactive_select_brain(brains)
        if not brain_info:
            print("Cancelled.")
            return
        
        pipeline_folder = brain_info['pipeline']
        mouse_folder = brain_info['mouse']
        brain_name = brain_info['name']
        
        # Load metadata
        metadata_path = pipeline_folder / FOLDER_CROPPED / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    # Check for paradigm-best settings
    tracker = ExperimentTracker()
    parsed = parse_brain_name(pipeline_folder.name)
    imaging_paradigm = parsed.get('imaging_params', '')
    paradigm_settings = None

    if imaging_paradigm:
        paradigm_settings = tracker.get_paradigm_detection_settings(imaging_paradigm)
        if paradigm_settings:
            print(f"\n[Paradigm Best Found] Settings for '{imaging_paradigm}':")
            print(f"  Source: {paradigm_settings.get('source_brain', 'unknown')}")
            print(f"  ball_xy={paradigm_settings['ball_xy']:.0f}, "
                  f"ball_z={paradigm_settings['ball_z']:.0f}, "
                  f"soma={paradigm_settings['soma_diameter']:.0f}, "
                  f"threshold={paradigm_settings['threshold']}")

    # Select parameters
    if args.routine and paradigm_settings:
        # Routine mode: auto-use paradigm-best settings
        print("\n[Routine Mode] Using paradigm-best detection settings.")
        preset_name = "paradigm_best"
        params = {
            'ball_xy_size': int(paradigm_settings['ball_xy']),
            'ball_z_size': int(paradigm_settings['ball_z']),
            'soma_diameter': int(paradigm_settings['soma_diameter']),
            'threshold': int(paradigm_settings['threshold']),
        }
    elif args.preset:
        preset_name = args.preset
        params = PRESETS[preset_name].copy()
    elif any([args.ball_xy, args.ball_z, args.soma_diameter, args.threshold]):
        # Custom from command line
        preset_name = "custom"
        params = {
            'ball_xy_size': args.ball_xy or 6,
            'ball_z_size': args.ball_z or 15,
            'soma_diameter': args.soma_diameter or 16,
            'threshold': args.threshold or 10,
        }
    elif not args.brain:
        # Interactive preset selection - offer paradigm-best first if available
        if paradigm_settings:
            print("\n" + "=" * 60)
            print("PARADIGM-BEST SETTINGS AVAILABLE")
            print("=" * 60)
            print(f"\nUse proven settings from '{imaging_paradigm}' paradigm? (y/n)")
            use_paradigm = input("Selection [y]: ").strip().lower()
            if use_paradigm in ('', 'y', 'yes'):
                preset_name = "paradigm_best"
                params = {
                    'ball_xy_size': int(paradigm_settings['ball_xy']),
                    'ball_z_size': int(paradigm_settings['ball_z']),
                    'soma_diameter': int(paradigm_settings['soma_diameter']),
                    'threshold': int(paradigm_settings['threshold']),
                }
            else:
                preset_name, params = interactive_select_preset()
        else:
            preset_name, params = interactive_select_preset()
    else:
        # Non-interactive with brain specified - use paradigm-best if available, else balanced
        if paradigm_settings:
            print("\n[Auto] Using paradigm-best settings (use --preset to override)")
            preset_name = "paradigm_best"
            params = {
                'ball_xy_size': int(paradigm_settings['ball_xy']),
                'ball_z_size': int(paradigm_settings['ball_z']),
                'soma_diameter': int(paradigm_settings['soma_diameter']),
                'threshold': int(paradigm_settings['threshold']),
            }
        else:
            preset_name = "balanced"
            params = PRESETS["balanced"].copy()
    
    # Override individual params if specified
    if args.ball_xy:
        params['ball_xy_size'] = args.ball_xy
    if args.ball_z:
        params['ball_z_size'] = args.ball_z
    if args.soma_diameter:
        params['soma_diameter'] = args.soma_diameter
    if args.threshold:
        params['threshold'] = args.threshold
    
    # Get paths
    crop_folder = pipeline_folder / FOLDER_CROPPED
    det_folder = pipeline_folder / FOLDER_DETECTION
    
    # Determine signal and background channels
    channels = metadata.get('channels', {})
    signal_ch = channels.get('signal_channel', 0)
    background_ch = channels.get('background_channel', 1)
    
    signal_path = crop_folder / f"ch{signal_ch}"
    background_path = crop_folder / f"ch{background_ch}"
    
    # Get voxel sizes
    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (
        voxel.get('z', 4),
        voxel.get('y', 4),
        voxel.get('x', 4),
    )
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Brain: {brain_name}")
        print(f"Preset: {preset_name}")
        print(f"Parameters: {params}")
        print(f"Signal: {signal_path}")
        print(f"Background: {background_path}")
        print(f"Voxel sizes: {voxel_sizes}")
        return

    # Log detection run (tracker already initialized for paradigm check)
    exp_id = tracker.log_detection(
        brain=brain_name,
        preset=preset_name,
        ball_xy=params['ball_xy_size'],
        ball_z=params['ball_z_size'],
        soma_diameter=params['soma_diameter'],
        threshold=params['threshold'],
        voxel_z=voxel_sizes[0],
        voxel_xy=voxel_sizes[1],
        input_path=str(crop_folder),
        output_path=str(det_folder),
        notes=args.notes,
        status="started",
        script_version=SCRIPT_VERSION,
    )
    
    print(f"\n{'='*60}")
    print(f"Detection Run: {exp_id}")
    print(f"Brain: {brain_name}")
    print(f"Preset: {preset_name}")
    print(f"{'='*60}")
    
    # Run detection
    success, duration, cells_found = run_cellfinder_detect(
        signal_path=signal_path,
        background_path=background_path,
        output_path=det_folder,
        voxel_sizes=voxel_sizes,
        params=params,
        n_free_cpus=args.n_free_cpus,
    )
    
    # Update tracker
    tracker.update_status(
        exp_id,
        status="completed" if success else "failed",
        duration_seconds=round(duration, 1),
        det_cells_found=cells_found,
    )
    
    print(f"\n{'='*60}")
    if success:
        print(f"COMPLETED in {duration/60:.1f} minutes")
        print(f"Cells detected: {cells_found}")
    else:
        print(f"FAILED after {duration/60:.1f} minutes")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")
    
    # Interactive rating
    if success:
        try:
            rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add a note (or Enter to skip): ").strip()
                tracker.rate_experiment(exp_id, int(rating), note if note else None)
                print("Rating saved!")
        except (EOFError, KeyboardInterrupt):
            pass

        # Next step guidance
        print("\n" + "=" * 60)
        print("WHAT TO DO NEXT")
        print("=" * 60)
        print("\n1. Run cell classification:")
        print(f"   python 5_classify_cells.py --brain {brain_name.split('/')[-1]}")
        print("\n" + "-" * 60)
        print("OR just run: python RUN_PIPELINE.py")
        print("   (it will guide you through everything)")
        print("=" * 60)


if __name__ == '__main__':
    main()
