#!/usr/bin/env python3
"""
run_classification.py

Wrapper for cellfinder classification that auto-logs to experiment tracker.

This runs a trained network on detected cell candidates to filter out
false positives, and automatically logs everything.

Usage:
    python run_classification.py --brain 349_CNT_01_02_1p625x_z4 --model path/to/model.h5
    python run_classification.py --brain 349_CNT_01_02_1p625x_z4 --detection det_20241218_abc
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

FOLDER_DETECTION = "3_Detected_Cell_Candidates"
FOLDER_CLASSIFICATION = "4_Classified_Cells"

DEFAULT_PARAMS = {
    'cube_size': 50,
    'batch_size': 32,
    'n_free_cpus': 2,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_detection_output(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find detection output for a brain."""
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name in [pipeline_folder.name, f"{mouse_folder.name}/{pipeline_folder.name}"]:
                detection_folder = pipeline_folder / FOLDER_DETECTION
                if detection_folder.exists():
                    # Look for detected cells
                    candidates = detection_folder / "detected_cells.xml"
                    if not candidates.exists():
                        candidates = detection_folder / "detected_cells.npy"
                    if candidates.exists():
                        output_folder = pipeline_folder / FOLDER_CLASSIFICATION
                        return candidates, output_folder, pipeline_folder
    return None, None, None


def count_cells(xml_path: Path) -> int:
    """Count cells in XML file."""
    if not xml_path.exists():
        return 0
    with open(xml_path, 'r') as f:
        content = f.read()
    return content.count('<Cell ')


def run_classification(
    candidates_path: Path,
    model_path: Path,
    output_path: Path,
    signal_path: Path,
    voxel_sizes: tuple,
    params: dict,
) -> tuple:
    """
    Run cellfinder classification.
    
    Returns:
        (success, duration, cells_found, rejected)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "cellfinder_classify",
        "--signal", str(signal_path),
        "--cells", str(candidates_path),
        "--output", str(output_path),
        "--model", str(model_path),
        "--voxel-sizes", str(voxel_sizes[0]), str(voxel_sizes[1]), str(voxel_sizes[2]),
        "--cube-size", str(params['cube_size']),
        "--batch-size", str(params['batch_size']),
        "--n-free-cpus", str(params['n_free_cpus']),
    ]
    
    print(f"\nRunning: cellfinder_classify")
    print(f"  Candidates: {candidates_path}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            cells_found = count_cells(output_path / "cells.xml")
            rejected = count_cells(output_path / "rejected.xml")
            return True, duration, cells_found, rejected
        else:
            return False, duration, 0, 0
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error: {e}")
        return False, duration, 0, 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder classification with auto-logging'
    )
    
    parser.add_argument('--brain', '-b', required=True,
                        help='Brain/pipeline to process')
    parser.add_argument('--model', '-m', type=Path, required=True,
                        help='Path to trained classification model')
    parser.add_argument('--detection', '-d',
                        help='Detection experiment ID to link to')
    
    # Parameters
    parser.add_argument('--cube-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--n-free-cpus', type=int, default=2)
    
    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    
    args = parser.parse_args()
    
    # Find data
    candidates_path, output_path, pipeline_folder = find_detection_output(args.brain, args.root)
    
    if not candidates_path:
        print(f"Error: No detection output found for {args.brain}")
        print("Run detection first with run_detection.py")
        sys.exit(1)
    
    # Get signal path and voxel sizes from metadata
    import json
    cropped_meta = pipeline_folder / "2_Cropped_For_Registration" / "metadata.json"
    with open(cropped_meta, 'r') as f:
        metadata = json.load(f)
    
    signal_ch = metadata.get('channels', {}).get('signal_channel', 0)
    signal_path = pipeline_folder / "2_Cropped_For_Registration" / f"ch{signal_ch}"
    
    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (voxel.get('z', 4), voxel.get('y', 4), voxel.get('x', 4))
    
    # Build params
    params = DEFAULT_PARAMS.copy()
    if args.cube_size:
        params['cube_size'] = args.cube_size
    if args.batch_size:
        params['batch_size'] = args.batch_size
    params['n_free_cpus'] = args.n_free_cpus
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Brain: {args.brain}")
        print(f"Candidates: {candidates_path}")
        print(f"Model: {args.model}")
        print(f"Output: {output_path}")
        return
    
    # Log and run
    tracker = ExperimentTracker()
    
    exp_id = tracker.log_classification(
        brain=args.brain,
        model_path=str(args.model),
        candidates_path=str(candidates_path),
        cube_size=params['cube_size'],
        batch_size=params['batch_size'],
        n_free_cpus=params['n_free_cpus'],
        output_path=str(output_path),
        parent_experiment=args.detection,
        notes=args.notes,
        status="started",
    )
    
    print(f"\n{'='*60}")
    print(f"Classification Run: {exp_id}")
    print(f"{'='*60}")
    
    success, duration, cells_found, rejected = run_classification(
        candidates_path=candidates_path,
        model_path=args.model,
        output_path=output_path,
        signal_path=signal_path,
        voxel_sizes=voxel_sizes,
        params=params,
    )
    
    tracker.update_status(
        exp_id,
        status="completed" if success else "failed",
        duration_seconds=round(duration, 1),
        class_cells_found=cells_found,
        class_rejected=rejected,
    )
    
    print(f"\n{'='*60}")
    if success:
        print(f"COMPLETED in {duration/60:.1f} minutes")
        print(f"Cells found: {cells_found}")
        print(f"Rejected: {rejected}")
    else:
        print(f"FAILED after {duration/60:.1f} minutes")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")
    
    try:
        rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
        if rating and rating.isdigit() and 1 <= int(rating) <= 5:
            note = input("Add a note (or Enter to skip): ").strip()
            tracker.rate_experiment(exp_id, int(rating), note if note else None)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == '__main__':
    main()
