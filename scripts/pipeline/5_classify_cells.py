#!/usr/bin/env python3
"""
5_classify_cells.py

Script 5 in the BrainGlobe pipeline: Cell classification.

Runs a trained network on detected cell candidates to filter out
false positives, and automatically logs to experiment tracker.

Run this AFTER Script 4 (4_detect_cells.py).

================================================================================
HOW TO USE
================================================================================
    python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --model path/to/model.h5
    python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4  # Uses default model
    python 5_classify_cells.py  # Interactive mode

================================================================================
REQUIREMENTS
================================================================================
    - cellfinder must be installed
    - Detection output in 4_Cell_Candidates folder (from Script 4)
    - experiment_tracker.py in same directory or PYTHONPATH
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.tracker import ExperimentTracker
except ImportError:
    print("ERROR: mousebrain.tracker not found!")
    print("Make sure mousebrain package is installed.")
    sys.exit(1)

# Try to import cellfinder classification components
try:
    from brainglobe_utils.cells.cells import Cell
    from brainglobe_utils.IO.cells import get_cells, save_cells
    from cellfinder.core.classify.cube_generator import CubeGeneratorFromFile
    from cellfinder.core.classify.classify import main as classify_main
    CELLFINDER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import cellfinder components: {e}")
    CELLFINDER_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
from mousebrain.config import MODELS_DIR as DEFAULT_MODELS_DIR
from mousebrain.config import parse_brain_name

FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_DETECTION = "4_Cell_Candidates"
FOLDER_CLASSIFICATION = "5_Classified_Cells"

DEFAULT_CUBE_SIZE = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_FREE_CPUS = 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def find_pipeline(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find pipeline folder for a brain."""
    root = Path(root)
    
    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            full_name = f"{mouse_dir.name}/{pipeline_dir.name}"
            if brain_name in [pipeline_dir.name, full_name]:
                return pipeline_dir, mouse_dir
    
    return None, None


def find_detection_output(pipeline_folder: Path):
    """Find detection output (cell candidates)."""
    det_folder = pipeline_folder / FOLDER_DETECTION
    
    if not det_folder.exists():
        return None
    
    # Look for detected cells (try both XML and NPY formats)
    for pattern in ["detected_cells.xml", "cell_classification.xml", "*.xml"]:
        files = list(det_folder.glob(pattern))
        if files:
            return files[0]
    
    return None


def _best_model_in_dir(model_dir):
    """Find the best model file in a training output directory.

    Priority: best_model.keras > best_model.h5 > lowest-loss .keras > lowest-loss .h5
    """
    # Check for explicitly saved best model
    for ext in ('.keras', '.h5'):
        best = model_dir / f"best_model{ext}"
        if best.exists():
            return best

    # Fall back to lowest-loss checkpoint
    for pattern in ('*.keras', '*.h5'):
        checkpoints = list(model_dir.glob(f"model-epoch.*{pattern.replace('*', '')}"))
        if checkpoints:
            # Parse loss from filename: model-epoch.19-loss-0.0898.h5
            def parse_loss(p):
                try:
                    return float(p.stem.split('loss-')[1])
                except (IndexError, ValueError):
                    return float('inf')
            checkpoints.sort(key=parse_loss)
            return checkpoints[0]

    # Last resort: any model file
    for pattern in ('*.keras', '*.h5'):
        files = list(model_dir.glob(pattern))
        if files:
            return files[0]

    return None


def find_default_model():
    """Find the best model from the most recent training run.

    Sorts training directories by timestamp in name (most recent first),
    then picks the best model file from that directory.
    """
    if not DEFAULT_MODELS_DIR.exists():
        return None

    # Collect directories that contain at least one model file
    candidates = []
    for model_dir in DEFAULT_MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        best = _best_model_in_dir(model_dir)
        if best:
            candidates.append((model_dir, best))

    if candidates:
        # Sort by best model file's modification time (most recent first)
        candidates.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
        chosen_dir, chosen_model = candidates[0]
        print(f"  Default model dir: {chosen_dir.name}")
        print(f"  Default model:     {chosen_model.name}")
        return chosen_model

    return None


def count_cells_in_xml(xml_path: Path) -> int:
    """Count cells in XML file."""
    if not xml_path.exists():
        return 0
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        return content.count('<Marker>')
    except:
        return 0


def list_available_brains(root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """List all brains that have detection output."""
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
            
            det_folder = pipeline_dir / FOLDER_DETECTION
            class_folder = pipeline_dir / FOLDER_CLASSIFICATION
            
            candidates = find_detection_output(pipeline_dir)
            has_detection = candidates is not None
            has_classification = class_folder.exists() and len(list(class_folder.glob("cells.xml"))) > 0
            
            if has_detection:
                n_candidates = count_cells_in_xml(candidates) if candidates else 0
                brains.append({
                    'name': f"{mouse_dir.name}/{pipeline_dir.name}",
                    'pipeline': pipeline_dir,
                    'mouse': mouse_dir,
                    'candidates': candidates,
                    'n_candidates': n_candidates,
                    'classified': has_classification,
                })
    
    return brains


def run_cellfinder_classify(
    signal_path: Path,
    candidates_path: Path,
    output_path: Path,
    model_path: Path,
    voxel_sizes: tuple,
    cube_size: int = DEFAULT_CUBE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_free_cpus: int = DEFAULT_N_FREE_CPUS,
    background_path: Path = None,
) -> tuple:
    """
    Run cellfinder classification using Python API.

    Returns:
        (success, duration, cells_found, rejected)
    """
    if not CELLFINDER_AVAILABLE:
        print("ERROR: cellfinder classification components not available!")
        print("Make sure cellfinder is installed: pip install cellfinder")
        return False, 0, 0, 0

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{timestamp()}] Running cellfinder classification...")
    print(f"    Signal: {signal_path}")
    print(f"    Background: {background_path}")
    print(f"    Candidates: {candidates_path}")
    print(f"    Model: {model_path}")
    print(f"    Output: {output_path}")
    print()

    start_time = time.time()

    try:
        # Load candidate cells from XML
        print(f"[{timestamp()}] Loading candidate cells...")
        cells = get_cells(str(candidates_path))
        print(f"    Loaded {len(cells)} candidates")

        if len(cells) == 0:
            print("ERROR: No candidates found in XML file")
            return False, time.time() - start_time, 0, 0

        # Convert voxel_sizes from (z, y, x) microns
        voxel_sizes_zyx = (float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2]))

        print(f"[{timestamp()}] Loading image data...")
        print(f"    Voxel sizes (z,y,x): {voxel_sizes_zyx}")

        # Load images as dask arrays for memory efficiency
        from brainglobe_utils.IO.image.load import read_with_dask

        signal_array = read_with_dask(str(signal_path))
        print(f"    Signal shape: {signal_array.shape}")

        if background_path and background_path.exists():
            background_array = read_with_dask(str(background_path))
            print(f"    Background shape: {background_array.shape}")
        else:
            background_array = None
            print("    Background: None")

        print(f"[{timestamp()}] Setting up classification...")
        print(f"    Batch size: {batch_size}")
        print(f"    Cube size: {cube_size}")
        print(f"    n_free_cpus: {n_free_cpus}")

        # Run classification using cellfinder's main function
        from cellfinder.core.classify.classify import main as classify_main

        cube_depth = 20  # Standard depth for cellfinder

        print(f"[{timestamp()}] Running classification network...")

        # Determine model type
        # .keras and .h5 checkpoints from training (save_weights_only=False) are
        # full models -> load via trained_model (tf.keras.models.load_model)
        # Weights-only files -> load via model_weights (build_model + load_weights)
        if model_path.suffix in ('.keras', '.h5'):
            trained_model_arg = model_path
            model_weights_arg = None
            print(f"    Model format: {model_path.suffix} (full model)")
        else:
            trained_model_arg = None
            model_weights_arg = model_path
            print(f"    Model format: {model_path.suffix} (weights)")

        classified_cells = classify_main(
            points=cells,
            signal_array=signal_array,
            background_array=background_array,
            n_free_cpus=n_free_cpus,
            voxel_sizes=voxel_sizes_zyx,
            network_voxel_sizes=(5.0, 1.0, 1.0),  # Standard network voxel sizes
            batch_size=batch_size,
            cube_height=cube_size,
            cube_width=cube_size,
            cube_depth=cube_depth,
            trained_model=trained_model_arg,
            model_weights=model_weights_arg,
            network_depth="50",  # ResNet50
        )

        duration = time.time() - start_time

        # Separate cells and rejected
        cells_list = [c for c in classified_cells if c.type == Cell.CELL]
        rejected_list = [c for c in classified_cells if c.type == Cell.NO_CELL]

        cells_found = len(cells_list)
        rejected_count = len(rejected_list)

        print(f"[{timestamp()}] Classification complete!")
        print(f"    Cells: {cells_found}")
        print(f"    Rejected: {rejected_count}")

        # Save results
        cells_output = output_path / "cells.xml"
        rejected_output = output_path / "rejected.xml"

        save_cells(cells_list, str(cells_output))
        save_cells(rejected_list, str(rejected_output))

        print(f"    Saved to: {output_path}")

        return True, duration, cells_found, rejected_count

    except Exception as e:
        import traceback
        print(f"ERROR during classification: {e}")
        traceback.print_exc()
        return False, time.time() - start_time, 0, 0


def interactive_select_brain(brains):
    """Interactive brain selection."""
    print("\n" + "=" * 60)
    print("BRAINS WITH DETECTION OUTPUT")
    print("=" * 60)
    
    ready = []
    already_done = []
    
    for i, brain in enumerate(brains):
        if brain['classified']:
            already_done.append((i, brain['name'], brain['n_candidates']))
        else:
            ready.append((i, brain['name'], brain['n_candidates']))
    
    if ready:
        print("\n[READY FOR CLASSIFICATION]")
        for idx, name, n in ready:
            print(f"  {idx + 1}. {name} ({n} candidates)")
    
    if already_done:
        print("\n[ALREADY CLASSIFIED]")
        for idx, name, n in already_done:
            print(f"  {idx + 1}. {name} ({n} candidates)")
    
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder classification with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 5_classify_cells.py                               # Interactive
    python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --model model.h5
    python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4  # Uses default model
    python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --no-prefilter  # Skip pre-filter (not recommended)
    python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --routine       # Auto pre-filter + paradigm model
        """
    )
    
    parser.add_argument('--brain', '-b', help='Brain/pipeline to process')
    parser.add_argument('--model', '-m', type=Path, help='Path to trained model')
    parser.add_argument('--detection', '-d', help='Detection experiment ID to link to')
    
    # Parameters
    parser.add_argument('--cube-size', type=int, default=DEFAULT_CUBE_SIZE)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS)
    
    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    parser.add_argument('--routine', action='store_true',
                        help='Routine processing mode: auto-use paradigm-best model if available')
    parser.add_argument('--no-prefilter', action='store_true',
                        help='Skip atlas pre-filter (not recommended - includes surface artifacts)')

    args = parser.parse_args()

    # Prefilter is ON by default; --no-prefilter opts out
    args.prefilter = not args.no_prefilter

    print("=" * 60)
    print("BrainGlobe Cell Classification")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)

    # Select brain first (needed for paradigm model lookup)
    if args.brain:
        pipeline_folder, mouse_folder = find_pipeline(args.brain, args.root)
        if not pipeline_folder:
            print(f"ERROR: Brain not found: {args.brain}")
            sys.exit(1)
        brain_name = f"{mouse_folder.name}/{pipeline_folder.name}"
        candidates = find_detection_output(pipeline_folder)
        if not candidates:
            print(f"ERROR: No detection output found for {args.brain}")
            print("Run Script 4 (4_detect_cells.py) first!")
            sys.exit(1)
    else:
        # Interactive selection
        brains = list_available_brains(args.root)
        if not brains:
            print("\nNo brains with detection output found!")
            print("Run Script 4 (4_detect_cells.py) first!")
            sys.exit(1)
        
        brain_info = interactive_select_brain(brains)
        if not brain_info:
            print("Cancelled.")
            return
        
        pipeline_folder = brain_info['pipeline']
        mouse_folder = brain_info['mouse']
        brain_name = brain_info['name']
        candidates = brain_info['candidates']

    # Check for paradigm-best model
    tracker = ExperimentTracker()
    parsed = parse_brain_name(pipeline_folder.name)
    imaging_paradigm = parsed.get('imaging_params', '')
    paradigm_model = None

    if imaging_paradigm:
        paradigm_model = tracker.get_paradigm_model(imaging_paradigm)
        if paradigm_model:
            paradigm_model_path = Path(paradigm_model)
            if paradigm_model_path.exists():
                print(f"\n[Paradigm Best Model Found] For '{imaging_paradigm}':")
                print(f"  Model: {paradigm_model_path.name}")
            else:
                print(f"\n[Warning] Paradigm model path not found: {paradigm_model}")
                paradigm_model = None

    # Find model (priority: explicit > paradigm-best > default)
    if args.model:
        model_path = args.model
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        print(f"Using specified model: {model_path}")
    elif args.routine and paradigm_model:
        # Routine mode: auto-use paradigm-best model
        print("\n[Routine Mode] Using paradigm-best model.")
        model_path = Path(paradigm_model)
    elif paradigm_model and not args.brain:
        # Interactive mode with paradigm model available
        print(f"\nUse paradigm-best model? (y/n)")
        use_paradigm = input("Selection [y]: ").strip().lower()
        if use_paradigm in ('', 'y', 'yes'):
            model_path = Path(paradigm_model)
            print(f"Using paradigm-best model: {model_path}")
        else:
            model_path = find_default_model()
            if not model_path:
                print("ERROR: No default model found! Specify with --model")
                sys.exit(1)
            print(f"Using default model: {model_path}")
    elif paradigm_model:
        # Non-interactive with paradigm model available
        print("\n[Auto] Using paradigm-best model (use --model to override)")
        model_path = Path(paradigm_model)
    else:
        model_path = find_default_model()
        if not model_path:
            print("ERROR: No model specified and no default model found!")
            print("Train a model or specify with --model")
            sys.exit(1)
        print(f"Using default model: {model_path}")

    # Load metadata
    crop_folder = pipeline_folder / FOLDER_CROPPED
    metadata_path = crop_folder / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: No metadata found at {metadata_path}")
        metadata = {}
    
    # Get paths and voxel sizes
    channels = metadata.get('channels', {})
    signal_ch = channels.get('signal_channel', 0)
    bg_ch = channels.get('background_channel', 1)
    signal_path = crop_folder / f"ch{signal_ch}"
    background_path = crop_folder / f"ch{bg_ch}"

    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (
        voxel.get('z', 4),
        voxel.get('y', 4),
        voxel.get('x', 4),
    )

    output_path = pipeline_folder / FOLDER_CLASSIFICATION

    # Optional pre-filter step
    if args.prefilter:
        print(f"\n[{timestamp()}] Running atlas pre-filter...")
        try:
            from mousebrain.prefilter import prefilter_candidates

            registration_path = pipeline_folder / "3_Registered_Atlas"
            if not registration_path.exists():
                print("WARNING: No registration folder found, skipping pre-filter")
            else:
                result = prefilter_candidates(
                    candidates_xml=candidates,
                    registration_path=registration_path,
                )

                total = result['stats']['total']
                interior = result['stats']['interior']
                suspicious = result['stats']['suspicious']

                print(f"    Total candidates: {total}")
                print(f"    Interior (keep): {interior}")
                print(f"    Suspicious (removed): {suspicious}")
                if total > 0:
                    print(f"    Pre-filter removed {suspicious/total*100:.1f}% of candidates")

                # Show category breakdown
                for cat, cnt in sorted(result.get('category_counts', {}).items(), key=lambda x: -x[1]):
                    print(f"      {cat}: {cnt:,}")

                # Save interior candidates as the new input for classification
                prefiltered_dir = pipeline_folder / FOLDER_DETECTION / "prefiltered"
                prefiltered_dir.mkdir(parents=True, exist_ok=True)
                prefiltered_xml = prefiltered_dir / "interior_candidates.xml"

                # Use helper function to save XML
                from mousebrain.prefilter import _coords_to_xml
                _coords_to_xml(result['interior_coords'], prefiltered_xml)

                # Use prefiltered candidates for classification
                candidates = prefiltered_xml
                print(f"    Using {interior} pre-filtered candidates for classification")

                # Log pre-filter to tracker
                tracker.log_prefilter(
                    brain=brain_name,
                    total=total,
                    interior=interior,
                    suspicious=suspicious,
                    tracing_type='descending',
                    output_path=str(prefiltered_xml),
                    status="completed",
                )
        except ImportError:
            print("WARNING: mousebrain.prefilter not found, skipping pre-filter")
        except Exception as e:
            print(f"WARNING: Pre-filter failed: {e}")
            print("Continuing with unfiltered candidates...")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Brain: {brain_name}")
        print(f"Candidates: {candidates}")
        print(f"Model: {model_path}")
        print(f"Output: {output_path}")
        return

    # Log classification run (tracker already initialized for paradigm check)
    exp_id = tracker.log_classification(
        brain=brain_name,
        model_path=str(model_path),
        candidates_path=str(candidates),
        cube_size=args.cube_size,
        batch_size=args.batch_size,
        n_free_cpus=args.n_free_cpus,
        output_path=str(output_path),
        parent_experiment=args.detection,
        notes=args.notes,
        status="started",
        script_version=SCRIPT_VERSION,
    )
    
    print(f"\n{'='*60}")
    print(f"Classification Run: {exp_id}")
    print(f"Brain: {brain_name}")
    print(f"{'='*60}")
    
    # Run classification
    success, duration, cells_found, rejected = run_cellfinder_classify(
        signal_path=signal_path,
        candidates_path=candidates,
        output_path=output_path,
        model_path=model_path,
        voxel_sizes=voxel_sizes,
        cube_size=args.cube_size,
        batch_size=args.batch_size,
        n_free_cpus=args.n_free_cpus,
        background_path=background_path,
    )
    
    # Update tracker
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
        print(f"Cells classified: {cells_found}")
        print(f"Rejected: {rejected}")
        if cells_found + rejected > 0:
            pct = cells_found / (cells_found + rejected) * 100
            print(f"Acceptance rate: {pct:.1f}%")
    else:
        print(f"FAILED after {duration/60:.1f} minutes")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")
    
    if success:
        try:
            rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add a note (or Enter to skip): ").strip()
                tracker.rate_experiment(exp_id, int(rating), note if note else None)
        except (EOFError, KeyboardInterrupt):
            pass

        # Next step guidance
        print("\n" + "=" * 60)
        print("WHAT TO DO NEXT")
        print("=" * 60)
        print("\n1. Run region counting:")
        print(f"   python 6_count_regions.py --brain {brain_name.split('/')[-1]}")
        print("\n" + "-" * 60)
        print("OR just run: python RUN_PIPELINE.py")
        print("   (it will guide you through everything)")
        print("=" * 60)


if __name__ == '__main__':
    main()
