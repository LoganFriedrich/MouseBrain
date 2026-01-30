#!/usr/bin/env python3
"""
run_detection.py

Wrapper for cellfinder detection that auto-logs to experiment tracker.

This runs cellfinder's cell candidate detection with specified parameters
and automatically logs everything to the experiment CSV.

Usage:
    python run_detection.py --brain 349_CNT_01_02_1p625x_z4 --soma-diameter 16
    python run_detection.py --brain 349_CNT_01_02_1p625x_z4 --preset sensitive
    python run_detection.py --list-presets

The script will:
1. Find the cropped data for the specified brain
2. Log the run to experiment_log.csv (started)
3. Run cellfinder detection
4. Update the log with results (completed/failed, candidates found)
5. Optionally prompt for notes/rating

Parameters can be specified individually or use presets for common configurations.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Folder names (must match pipeline scripts)
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_DETECTION = "3_Detected_Cell_Candidates"

# Default detection parameters (cellfinder defaults)
DEFAULT_PARAMS = {
    'soma_diameter': 16,
    'ball_xy_size': 6,
    'ball_z_size': 15,
    'ball_overlap_fraction': 0.6,
    'max_cluster_size': 100000,
    'soma_spread_factor': 1.4,
    'log_sigma_size': 0.2,
    'n_free_cpus': 2,
}

# Preset configurations for common use cases
PRESETS = {
    'default': {
        **DEFAULT_PARAMS,
        'description': 'Cellfinder defaults - good starting point'
    },
    'sensitive': {
        **DEFAULT_PARAMS,
        'soma_diameter': 14,
        'ball_xy_size': 5,
        'ball_overlap_fraction': 0.7,
        'description': 'More sensitive - catches smaller/dimmer cells, more false positives'
    },
    'conservative': {
        **DEFAULT_PARAMS,
        'soma_diameter': 18,
        'ball_xy_size': 7,
        'ball_overlap_fraction': 0.5,
        'description': 'More conservative - fewer false positives, might miss some cells'
    },
    'large_cells': {
        **DEFAULT_PARAMS,
        'soma_diameter': 20,
        'ball_xy_size': 8,
        'ball_z_size': 18,
        'description': 'For larger cell bodies (e.g., motor neurons)'
    },
    'small_cells': {
        **DEFAULT_PARAMS,
        'soma_diameter': 12,
        'ball_xy_size': 4,
        'ball_z_size': 12,
        'description': 'For smaller cell bodies'
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_brain_data(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """
    Find the cropped data for a brain.
    
    Returns:
        (signal_path, background_path, output_path, metadata) or None
    """
    import json
    
    # Brain name could be just the pipeline folder or mouse/pipeline
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            
            # Check if this matches
            if brain_name in [pipeline_folder.name, f"{mouse_folder.name}/{pipeline_folder.name}"]:
                cropped = pipeline_folder / FOLDER_CROPPED
                if not cropped.exists():
                    print(f"Error: Cropped folder not found: {cropped}")
                    return None
                
                # Load metadata to get channel info
                metadata_path = cropped / "metadata.json"
                if not metadata_path.exists():
                    print(f"Error: Metadata not found: {metadata_path}")
                    return None
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Get channels
                signal_ch = metadata.get('channels', {}).get('signal_channel', 0)
                bg_ch = metadata.get('channels', {}).get('background_channel', 1)
                
                signal_path = cropped / f"ch{signal_ch}"
                bg_path = cropped / f"ch{bg_ch}" if bg_ch is not None else None
                
                # Output path
                output_path = pipeline_folder / FOLDER_DETECTION
                
                return signal_path, bg_path, output_path, metadata
    
    print(f"Error: Brain not found: {brain_name}")
    return None


def list_available_brains(root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """List all brains with cropped data ready for detection."""
    brains = []
    
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            
            cropped = pipeline_folder / FOLDER_CROPPED
            if cropped.exists() and (cropped / "metadata.json").exists():
                brains.append(f"{mouse_folder.name}/{pipeline_folder.name}")
    
    return sorted(brains)


def count_detected_cells(output_path: Path) -> int:
    """Count cells in detection output."""
    # Cellfinder outputs cells as XML or can be read from npy
    cells_xml = output_path / "detected_cells.xml"
    cells_npy = output_path / "detected_cells.npy"
    
    if cells_npy.exists():
        import numpy as np
        cells = np.load(cells_npy)
        return len(cells)
    elif cells_xml.exists():
        # Parse XML (simplified - just count cell tags)
        with open(cells_xml, 'r') as f:
            content = f.read()
        return content.count('<Cell ')
    
    return 0


def run_detection(
    signal_path: Path,
    background_path: Path,
    output_path: Path,
    voxel_z: float,
    voxel_y: float,
    voxel_x: float,
    params: dict,
) -> tuple:
    """
    Run cellfinder detection.
    
    Returns:
        (success: bool, duration_seconds: float, candidates_found: int)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "cellfinder",
        "--signal", str(signal_path),
        "--output", str(output_path),
        "--voxel-sizes", str(voxel_z), str(voxel_y), str(voxel_x),
        "--soma-diameter", str(params['soma_diameter']),
        "--ball-xy-size", str(params['ball_xy_size']),
        "--ball-z-size", str(params['ball_z_size']),
        "--ball-overlap-fraction", str(params['ball_overlap_fraction']),
        "--max-cluster-size", str(params['max_cluster_size']),
        "--soma-spread-factor", str(params['soma_spread_factor']),
        "--log-sigma-size", str(params['log_sigma_size']),
        "--n-free-cpus", str(params['n_free_cpus']),
        "--no-classify",  # Detection only, no classification
    ]
    
    if background_path and background_path.exists():
        cmd.extend(["--background", str(background_path)])
    
    print(f"\nRunning: {' '.join(cmd[:5])}...")
    print(f"  Signal: {signal_path}")
    print(f"  Background: {background_path}")
    print(f"  Output: {output_path}")
    print(f"  Soma diameter: {params['soma_diameter']}")
    print(f"  Ball XY/Z: {params['ball_xy_size']}/{params['ball_z_size']}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            candidates = count_detected_cells(output_path)
            return True, duration, candidates
        else:
            return False, duration, 0
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error: {e}")
        return False, duration, 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder detection with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py --brain 349_CNT_01_02_1p625x_z4
  python run_detection.py --brain 349_CNT_01_02_1p625x_z4 --preset sensitive
  python run_detection.py --brain 349_CNT_01_02_1p625x_z4 --soma-diameter 14 --ball-xy-size 5
  python run_detection.py --list-presets
  python run_detection.py --list-brains
        """
    )
    
    parser.add_argument('--brain', '-b', help='Brain/pipeline to process')
    parser.add_argument('--preset', '-p', choices=list(PRESETS.keys()),
                        help='Use a preset configuration')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets')
    parser.add_argument('--list-brains', action='store_true',
                        help='List available brains')
    
    # Individual parameters (override preset)
    parser.add_argument('--soma-diameter', type=int)
    parser.add_argument('--ball-xy-size', type=int)
    parser.add_argument('--ball-z-size', type=int)
    parser.add_argument('--ball-overlap-fraction', type=float)
    parser.add_argument('--max-cluster-size', type=int)
    parser.add_argument('--soma-spread-factor', type=float)
    parser.add_argument('--log-sigma-size', type=float)
    parser.add_argument('--n-free-cpus', type=int, default=2)
    
    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without running')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT,
                        help='Root path for brain data')
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_presets:
        print("\nAvailable presets:\n")
        for name, preset in PRESETS.items():
            print(f"  {name}:")
            print(f"    {preset.get('description', '')}")
            print(f"    soma_diameter={preset['soma_diameter']}, ball_xy={preset['ball_xy_size']}, ball_z={preset['ball_z_size']}")
            print()
        return
    
    if args.list_brains:
        brains = list_available_brains(args.root)
        print(f"\nAvailable brains with cropped data ({len(brains)}):\n")
        for brain in brains:
            print(f"  {brain}")
        return
    
    if not args.brain:
        parser.error("--brain is required")
    
    # Find brain data
    result = find_brain_data(args.brain, args.root)
    if not result:
        sys.exit(1)
    
    signal_path, bg_path, output_path, metadata = result
    
    # Build parameters
    if args.preset:
        params = {k: v for k, v in PRESETS[args.preset].items() if k != 'description'}
    else:
        params = DEFAULT_PARAMS.copy()
    
    # Override with explicit arguments
    if args.soma_diameter:
        params['soma_diameter'] = args.soma_diameter
    if args.ball_xy_size:
        params['ball_xy_size'] = args.ball_xy_size
    if args.ball_z_size:
        params['ball_z_size'] = args.ball_z_size
    if args.ball_overlap_fraction:
        params['ball_overlap_fraction'] = args.ball_overlap_fraction
    if args.max_cluster_size:
        params['max_cluster_size'] = args.max_cluster_size
    if args.soma_spread_factor:
        params['soma_spread_factor'] = args.soma_spread_factor
    if args.log_sigma_size:
        params['log_sigma_size'] = args.log_sigma_size
    params['n_free_cpus'] = args.n_free_cpus
    
    # Get voxel sizes
    voxel = metadata.get('voxel_size_um', {})
    voxel_z = voxel.get('z', 4.0)
    voxel_y = voxel.get('y', 4.0)
    voxel_x = voxel.get('x', 4.0)
    
    # Dry run?
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nBrain: {args.brain}")
        print(f"Signal: {signal_path}")
        print(f"Background: {bg_path}")
        print(f"Output: {output_path}")
        print(f"Voxels (Z,Y,X): {voxel_z}, {voxel_y}, {voxel_x}")
        print(f"\nParameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        return
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Get channel info for logging
    channels = metadata.get('channels', {})
    signal_ch = f"ch{channels.get('signal_channel', 0)}"
    bg_ch = f"ch{channels.get('background_channel', 1)}" if channels.get('background_channel') is not None else None
    
    # Log start
    exp_id = tracker.log_detection(
        brain=args.brain,
        signal_channel=signal_ch,
        background_channel=bg_ch,
        soma_diameter=params['soma_diameter'],
        ball_xy_size=params['ball_xy_size'],
        ball_z_size=params['ball_z_size'],
        ball_overlap_fraction=params['ball_overlap_fraction'],
        max_cluster_size=params['max_cluster_size'],
        soma_spread_factor=params['soma_spread_factor'],
        log_sigma_size=params['log_sigma_size'],
        n_free_cpus=params['n_free_cpus'],
        output_path=str(output_path),
        notes=args.notes,
        status="started",
    )
    
    print(f"\n{'='*60}")
    print(f"Detection Run: {exp_id}")
    print(f"{'='*60}")
    
    # Run detection
    success, duration, candidates = run_detection(
        signal_path=signal_path,
        background_path=bg_path,
        output_path=output_path,
        voxel_z=voxel_z,
        voxel_y=voxel_y,
        voxel_x=voxel_x,
        params=params,
    )
    
    # Update log
    tracker.update_status(
        exp_id,
        status="completed" if success else "failed",
        duration_seconds=round(duration, 1),
        det_candidates_found=candidates,
    )
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print(f"COMPLETED in {duration/60:.1f} minutes")
        print(f"Candidates found: {candidates}")
    else:
        print(f"FAILED after {duration/60:.1f} minutes")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")
    
    # Prompt for rating?
    try:
        rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
        if rating and rating.isdigit() and 1 <= int(rating) <= 5:
            note = input("Add a note (or Enter to skip): ").strip()
            tracker.rate_experiment(exp_id, int(rating), note if note else None)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == '__main__':
    main()
