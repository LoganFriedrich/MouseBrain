#!/usr/bin/env python3
"""
compare_model_classification.py

Compare classification results between two model revisions across multiple brains.

Runs classification with Model A (previous) and Model B (newest) on each brain's
most recent full detection, saving results side-by-side for comparison.

Expected result: peripheral/meningeal false positives should be classified as
rejected (not cells) by the newer models.

================================================================================
USAGE
================================================================================
    # Run full comparison (all 3 brains, both models)
    python compare_model_classification.py

    # Dry run to verify paths
    python compare_model_classification.py --dry-run

    # Override models
    python compare_model_classification.py --model-a path/to/modelA.keras --model-b path/to/modelB.keras

================================================================================
OUTPUT
================================================================================
    For each brain, saves to:
        {brain}/5_Classified_Cells/compare_modelA/cells.xml + rejected.xml
        {brain}/5_Classified_Cells/compare_modelB/cells.xml + rejected.xml

    Summary report saved to:
        2_Data_Summary/reports/model_comparison_{timestamp}.txt
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.tracker import ExperimentTracker
    from mousebrain.config import BRAINS_ROOT, MODELS_DIR, DATA_SUMMARY_DIR
except ImportError:
    try:
        from braintools.tracker import ExperimentTracker
        from braintools.config import BRAINS_ROOT, MODELS_DIR, DATA_SUMMARY_DIR
    except ImportError:
        # Fall back to local shims
        from experiment_tracker import ExperimentTracker
        from config import BRAINS_ROOT, MODELS_DIR, DATA_SUMMARY_DIR

# Heavy imports (cellfinder/keras/tensorflow) are deferred to avoid slow startup.
# They are loaded on first call to _ensure_cellfinder().
Cell = None
get_cells = None
save_cells = None
read_with_dask = None
classify_main = None
CELLFINDER_AVAILABLE = None


def _ensure_cellfinder():
    """Lazy-load cellfinder and brainglobe_utils on first use."""
    global Cell, get_cells, save_cells, read_with_dask, classify_main, CELLFINDER_AVAILABLE
    if CELLFINDER_AVAILABLE is not None:
        return CELLFINDER_AVAILABLE
    try:
        print(f"[{timestamp()}] Loading cellfinder (this may take a minute)...", flush=True)
        from brainglobe_utils.cells.cells import Cell as _Cell
        from brainglobe_utils.IO.cells import get_cells as _get_cells, save_cells as _save_cells
        from brainglobe_utils.IO.image.load import read_with_dask as _read_with_dask
        from cellfinder.core.classify.classify import main as _classify_main
        Cell = _Cell
        get_cells = _get_cells
        save_cells = _save_cells
        read_with_dask = _read_with_dask
        classify_main = _classify_main
        CELLFINDER_AVAILABLE = True
        print(f"[{timestamp()}] cellfinder loaded OK.", flush=True)
    except ImportError as e:
        print(f"WARNING: Could not import cellfinder components: {e}")
        CELLFINDER_AVAILABLE = False
    return CELLFINDER_AVAILABLE


# =============================================================================
# CONFIGURATION
# =============================================================================

# The 3 newer brains with 1p625x_z4 imaging paradigm
BRAINS = [
    "357_CNT_02_08/357_CNT_02_08_1p625x_z4",
    "367_CNT_03_07/367_CNT_03_07_1p625x_z4",
    "368_CNT_03_08/368_CNT_03_08_1p625x_z4",
]

# Model A: previous revision (Jan 23, 100 epochs, best loss 0.036) - converted to Keras 2 .h5
DEFAULT_MODEL_A = MODELS_DIR / "model_20260123_142837" / "model_keras2.h5"
MODEL_A_LABEL = "Jan23_100ep"

# Model B: newest revision (Jan 28 manual, 40 epochs, best epoch 37 loss 0.057) - converted to Keras 2 .h5
DEFAULT_MODEL_B = MODELS_DIR / "model_20260128_manual" / "model-epoch.37-loss-0.057_keras2.h5"
MODEL_B_LABEL = "Jan28_40ep"

CLASSIFY_PARAMS = {
    "cube_size": 50,
    "cube_depth": 20,
    "batch_size": 32,
    "n_free_cpus": 2,
    "network_depth": "50",
    "network_voxel_sizes": (5.0, 1.0, 1.0),
}


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def load_brain_metadata(brain_path: Path) -> dict:
    """Load voxel sizes and channel info from metadata.json."""
    crop_folder = brain_path / "2_Cropped_For_Registration"
    metadata_path = crop_folder / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"  Warning: No metadata at {metadata_path}, using defaults")
        metadata = {}

    channels = metadata.get('channels', {})
    signal_ch = channels.get('signal_channel', 0)
    bg_ch = channels.get('background_channel', 1)

    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (
        voxel.get('z', 4.0),
        voxel.get('y', 4.0),
        voxel.get('x', 4.0),
    )

    return {
        'signal_path': crop_folder / f"ch{signal_ch}",
        'background_path': crop_folder / f"ch{bg_ch}",
        'voxel_sizes': voxel_sizes,
    }


def run_classification(
    brain_name: str,
    brain_path: Path,
    candidates_path: Path,
    model_path: Path,
    model_label: str,
    metadata: dict,
    tracker: ExperimentTracker,
) -> dict:
    """
    Run classification with a specific model and save to a labeled output directory.

    Returns dict with results.
    """
    output_dir = brain_path / "5_Classified_Cells" / f"compare_{model_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Brain:  {brain_name}")
    print(f"  Model:  {model_label} ({model_path.name})")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load candidates
    print(f"[{timestamp()}] Loading candidates from {candidates_path.name}...")
    cells = get_cells(str(candidates_path))
    n_candidates = len(cells)
    print(f"  Loaded {n_candidates} candidates")

    if n_candidates == 0:
        print("  ERROR: No candidates found!")
        return {'success': False, 'brain': brain_name, 'model': model_label}

    # Load image data
    signal_path = metadata['signal_path']
    background_path = metadata['background_path']
    voxel_sizes = metadata['voxel_sizes']

    print(f"[{timestamp()}] Loading image data...")
    signal_array = read_with_dask(str(signal_path))
    print(f"  Signal shape: {signal_array.shape}")

    if background_path.exists():
        background_array = read_with_dask(str(background_path))
        print(f"  Background shape: {background_array.shape}")
    else:
        background_array = None
        print("  Background: None")

    print(f"[{timestamp()}] Voxel sizes (z,y,x): {voxel_sizes}")

    # Determine model format for cellfinder classify API
    # Our converted .h5 files are full Keras 2 models (saved with model.save()),
    # so load via trained_model → tf.keras.models.load_model()
    # .keras files would need conversion first (use convert_keras3_to_h5.py)
    trained_model_arg = model_path
    model_weights_arg = None

    # Log to tracker
    exp_id = tracker.log_classification(
        brain=brain_name,
        model_path=str(model_path),
        candidates_path=str(candidates_path),
        cube_size=CLASSIFY_PARAMS['cube_size'],
        batch_size=CLASSIFY_PARAMS['batch_size'],
        output_path=str(output_dir),
        notes=f"Model comparison: {model_label}",
        status="started",
    )

    # Run classification
    print(f"[{timestamp()}] Running classification with {model_label}...")
    try:
        classified_cells = classify_main(
            points=cells,
            signal_array=signal_array,
            background_array=background_array,
            n_free_cpus=CLASSIFY_PARAMS['n_free_cpus'],
            voxel_sizes=voxel_sizes,
            network_voxel_sizes=CLASSIFY_PARAMS['network_voxel_sizes'],
            batch_size=CLASSIFY_PARAMS['batch_size'],
            cube_height=CLASSIFY_PARAMS['cube_size'],
            cube_width=CLASSIFY_PARAMS['cube_size'],
            cube_depth=CLASSIFY_PARAMS['cube_depth'],
            trained_model=trained_model_arg,
            model_weights=model_weights_arg,
            network_depth=CLASSIFY_PARAMS['network_depth'],
        )

        duration = time.time() - start_time

        # Separate cells and rejected
        cells_list = [c for c in classified_cells if c.type == Cell.CELL]
        rejected_list = [c for c in classified_cells if c.type == Cell.NO_CELL]

        n_cells = len(cells_list)
        n_rejected = len(rejected_list)
        rejection_pct = (n_rejected / n_candidates * 100) if n_candidates > 0 else 0

        # Save results
        save_cells(cells_list, str(output_dir / "cells.xml"))
        save_cells(rejected_list, str(output_dir / "rejected.xml"))

        print(f"[{timestamp()}] Classification complete! ({duration:.1f}s)")
        print(f"  Candidates:  {n_candidates}")
        print(f"  Accepted:    {n_cells}")
        print(f"  Rejected:    {n_rejected} ({rejection_pct:.1f}%)")

        # Update tracker
        tracker.update_status(
            exp_id,
            status="completed",
            duration=duration,
            class_cells_found=n_cells,
            class_rejected=n_rejected,
        )

        return {
            'success': True,
            'brain': brain_name,
            'model': model_label,
            'model_path': str(model_path),
            'candidates': n_candidates,
            'accepted': n_cells,
            'rejected': n_rejected,
            'rejection_pct': rejection_pct,
            'duration': duration,
            'output_dir': str(output_dir),
            'exp_id': exp_id,
        }

    except Exception as e:
        import traceback
        duration = time.time() - start_time
        print(f"  ERROR: {e}")
        traceback.print_exc()
        tracker.update_status(exp_id, status="failed", duration=duration)
        return {
            'success': False,
            'brain': brain_name,
            'model': model_label,
            'error': str(e),
        }


def generate_report(results: list, model_a_path: Path, model_b_path: Path) -> str:
    """Generate comparison report text."""
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL CLASSIFICATION COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model A ({MODEL_A_LABEL}): {model_a_path}")
    lines.append(f"Model B ({MODEL_B_LABEL}): {model_b_path}")
    lines.append("")

    # Group results by brain
    by_brain = {}
    for r in results:
        if r['success']:
            brain = r['brain']
            if brain not in by_brain:
                by_brain[brain] = {}
            by_brain[brain][r['model']] = r

    # Summary table
    lines.append("-" * 80)
    lines.append(f"{'Brain':<40} {'Model':<15} {'Candidates':>10} {'Accepted':>10} {'Rejected':>10} {'Rej%':>8}")
    lines.append("-" * 80)

    for brain in sorted(by_brain.keys()):
        for model_label in [MODEL_A_LABEL, MODEL_B_LABEL]:
            if model_label in by_brain[brain]:
                r = by_brain[brain][model_label]
                lines.append(
                    f"{brain:<40} {model_label:<15} {r['candidates']:>10} "
                    f"{r['accepted']:>10} {r['rejected']:>10} {r['rejection_pct']:>7.1f}%"
                )
        lines.append("")

    # Delta analysis
    lines.append("=" * 80)
    lines.append("DELTA ANALYSIS (Model B vs Model A)")
    lines.append("=" * 80)
    lines.append("")

    for brain in sorted(by_brain.keys()):
        a = by_brain[brain].get(MODEL_A_LABEL)
        b = by_brain[brain].get(MODEL_B_LABEL)
        if a and b:
            delta_accepted = b['accepted'] - a['accepted']
            delta_rejected = b['rejected'] - a['rejected']
            delta_pct = b['rejection_pct'] - a['rejection_pct']
            lines.append(f"{brain}:")
            lines.append(f"  Accepted change:  {delta_accepted:+d} cells")
            lines.append(f"  Rejected change:  {delta_rejected:+d} cells")
            lines.append(f"  Rejection % change: {delta_pct:+.1f}%")
            lines.append("")

    # Interpretation
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("If edge/meningeal rejection is working correctly:")
    lines.append("  - Rejection % should be > 0% (unlike the Jan 10 model which was 0%)")
    lines.append("  - Peripheral false positives appear in rejected.xml")
    lines.append("  - Real interior cells remain in cells.xml")
    lines.append("")
    lines.append("To visually verify, load in napari:")
    lines.append("  1. Load brain signal channel")
    lines.append("  2. Load cells.xml and rejected.xml from each compare_* folder")
    lines.append("  3. Check that rejected points cluster at brain periphery/meninges")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare classification results between two model revisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model-a', type=Path, default=DEFAULT_MODEL_A,
                        help=f'Previous model (default: {DEFAULT_MODEL_A.name})')
    parser.add_argument('--model-b', type=Path, default=DEFAULT_MODEL_B,
                        help=f'Newest model (default: {DEFAULT_MODEL_B.name})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without running classification')
    parser.add_argument('--brains', nargs='+', default=BRAINS,
                        help='Brain paths to process (default: all 3 newer brains)')

    args = parser.parse_args()

    if not args.dry_run:
        if not _ensure_cellfinder():
            print("ERROR: cellfinder not available. Install with: pip install cellfinder")
            sys.exit(1)

    model_a = args.model_a
    model_b = args.model_b

    print("=" * 80)
    print("MODEL CLASSIFICATION COMPARISON")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\nModel A ({MODEL_A_LABEL}): {model_a}")
    print(f"Model B ({MODEL_B_LABEL}): {model_b}")

    # Validate models exist
    if not model_a.exists():
        print(f"\nERROR: Model A not found: {model_a}")
        sys.exit(1)
    if not model_b.exists():
        print(f"\nERROR: Model B not found: {model_b}")
        sys.exit(1)

    # Validate brains and collect info
    brain_info = []
    print(f"\nBrains to process ({len(args.brains)}):")
    for brain_name in args.brains:
        brain_path = BRAINS_ROOT / brain_name
        if not brain_path.exists():
            print(f"  WARNING: Brain not found: {brain_path}")
            continue

        candidates_path = brain_path / "4_Cell_Candidates" / "detected_cells.xml"
        if not candidates_path.exists():
            print(f"  WARNING: No detection for {brain_name}")
            continue

        metadata = load_brain_metadata(brain_path)
        if not metadata['signal_path'].exists():
            print(f"  WARNING: No signal data for {brain_name}")
            continue

        brain_info.append({
            'name': brain_name,
            'path': brain_path,
            'candidates': candidates_path,
            'metadata': metadata,
        })
        print(f"  {brain_name} - OK")

    if not brain_info:
        print("\nERROR: No valid brains found!")
        sys.exit(1)

    total_runs = len(brain_info) * 2
    print(f"\nTotal classification runs: {total_runs}")

    if args.dry_run:
        print("\n=== DRY RUN - would run the following ===")
        for bi in brain_info:
            for label, model in [(MODEL_A_LABEL, model_a), (MODEL_B_LABEL, model_b)]:
                output = bi['path'] / "5_Classified_Cells" / f"compare_{label}"
                print(f"\n  Brain: {bi['name']}")
                print(f"  Model: {label} ({model.name})")
                print(f"  Input: {bi['candidates']}")
                print(f"  Output: {output}")
        print("\nDry run complete. Remove --dry-run to execute.")
        return

    # Run all classifications
    tracker = ExperimentTracker()
    all_results = []
    run_count = 0

    overall_start = time.time()

    for bi in brain_info:
        for model_label, model_path in [(MODEL_A_LABEL, model_a), (MODEL_B_LABEL, model_b)]:
            run_count += 1
            print(f"\n\n{'#'*80}")
            print(f"# RUN {run_count}/{total_runs}")
            print(f"{'#'*80}")

            result = run_classification(
                brain_name=bi['name'],
                brain_path=bi['path'],
                candidates_path=bi['candidates'],
                model_path=model_path,
                model_label=model_label,
                metadata=bi['metadata'],
                tracker=tracker,
            )
            all_results.append(result)

    overall_duration = time.time() - overall_start

    # Generate and save report
    report = generate_report(all_results, model_a, model_b)

    reports_dir = DATA_SUMMARY_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, 'w') as f:
        f.write(report)

    # Print report
    print(f"\n\n{report}")
    print(f"\nReport saved to: {report_path}")
    print(f"Total time: {overall_duration/3600:.1f} hours")
    print("\nDone! Load results in napari to visually inspect edge rejection.")


if __name__ == "__main__":
    main()
