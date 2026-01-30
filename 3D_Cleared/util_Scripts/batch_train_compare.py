#!/usr/bin/env python3
"""
batch_train_compare.py

CLI for overnight training + classification + comparison workflow.

Run this, go to sleep, come back to see results in napari.

================================================================================
USAGE (PowerShell)
================================================================================
    # Full workflow: train (150 epochs default), classify, compare
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4

    # Deep training with 200 epochs
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4 --epochs 200

    # Just classify with latest model and compare (skip training)
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4 --skip-training

    # Specify detection to classify (otherwise uses most recent)
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4 --detection det_20260122_123640_69eb32

================================================================================
WHAT IT DOES
================================================================================
    1. Finds training data for the brain
    2. Trains a new model (unless --skip-training)
    3. Finds most recent detection for the brain
    4. Runs classification with new/latest model
    5. Compares to previous classification
    6. Saves summary to {brain}/comparison_summary.txt
    7. Prints results

When done, open napari and load the brain - classification results will be there.

================================================================================
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from experiment_tracker import ExperimentTracker
    from config import BRAINS_ROOT, MODELS_DIR, DATA_SUMMARY_DIR
except ImportError as e:
    print(f"ERROR: Required module not found: {e}")
    sys.exit(1)


def find_brain_path(brain_name: str) -> Path:
    """Find brain folder by name (searches subdirectories)."""
    # Direct match
    direct = BRAINS_ROOT / brain_name
    if direct.exists():
        return direct

    # Search in subdirectories (e.g., 368_CNT_03_08/368_CNT_03_08_1p625x_z4)
    for subdir in BRAINS_ROOT.iterdir():
        if subdir.is_dir():
            candidate = subdir / brain_name
            if candidate.exists():
                return candidate

    # Fuzzy match
    for path in BRAINS_ROOT.rglob("*"):
        if path.is_dir() and brain_name in path.name:
            return path

    return None


def find_training_data(brain_path: Path) -> tuple:
    """Find cells and non_cells training data folders."""
    # Check in brain folder first
    training_dir = brain_path / "training_data"
    if training_dir.exists():
        cells = training_dir / "cells"
        non_cells = training_dir / "non_cells"
        if cells.exists() and non_cells.exists():
            return cells, non_cells

    # Check in shared training location
    # Pattern: Trained_Models/{resolution}/training_data
    brain_name = brain_path.name
    # Extract resolution pattern like "1p625x_z4"
    parts = brain_name.split("_")
    for i, part in enumerate(parts):
        if "x" in part.lower() and i + 1 < len(parts) and parts[i+1].startswith("z"):
            resolution = f"{part}_{parts[i+1]}"
            shared_training = MODELS_DIR / resolution / "training_data"
            if shared_training.exists():
                cells = shared_training / "cells"
                non_cells = shared_training / "non_cells"
                if cells.exists() and non_cells.exists():
                    return cells, non_cells

    return None, None


def find_latest_model() -> Path:
    """Find the most recently created model."""
    model_files = []
    for ext in ['*.keras', '*.h5']:
        for f in MODELS_DIR.rglob(ext):
            # Skip checkpoints
            if 'epoch' in f.stem.lower() or 'checkpoint' in f.stem.lower():
                continue
            model_files.append((f.stat().st_mtime, f))

    if not model_files:
        return None

    model_files.sort(key=lambda x: x[0], reverse=True)
    return model_files[0][1]


def find_detection_xml(brain_path: Path, tracker: ExperimentTracker, detection_id: str = None) -> Path:
    """Find detection XML for classification."""
    brain_name = brain_path.name

    if detection_id:
        # Specific detection requested
        run = tracker.get_experiment(detection_id)
        if run and run.get('output_path'):
            xml_path = Path(run['output_path'])
            if xml_path.exists():
                return xml_path

    # Find most recent completed detection for this brain
    detections = tracker.search(
        brain=brain_name,
        exp_type="detection",
        status="completed",
        limit=5,
        sort_by="created_at",
        descending=True
    )

    for det in detections:
        output_path = det.get('output_path')
        if output_path:
            xml_path = Path(output_path)
            if xml_path.exists():
                return xml_path

    # Fallback: check standard location
    candidates_dir = brain_path / "4_Cell_Candidates"
    if candidates_dir.exists():
        xmls = sorted(candidates_dir.glob("*.xml"), key=lambda x: x.stat().st_mtime, reverse=True)
        if xmls:
            return xmls[0]

    return None


def run_training(cells_dir: Path, non_cells_dir: Path, epochs: int = 150, learning_rate: float = 0.0001) -> bool:
    """Run training script."""
    import subprocess

    script = Path(__file__).parent / "util_train_model.py"
    cmd = [
        sys.executable, str(script),
        "--cells", str(cells_dir),
        "--non-cells", str(non_cells_dir),
        "--epochs", str(epochs),
        "--learning-rate", str(learning_rate)
    ]

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"Cells: {cells_dir}")
    print(f"Non-cells: {non_cells_dir}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    return result.returncode == 0


def resolve_detection_xml(detection_path: Path) -> Path:
    """Resolve detection path to an actual XML file.

    The tracker may store a directory path (e.g. optimization_runs/test_YYYYMMDD_HHMMSS)
    rather than a specific XML file. This function finds the XML inside.
    """
    if detection_path.is_file() and detection_path.suffix == '.xml':
        return detection_path

    if detection_path.is_dir():
        # Look for XML files in the directory
        for pattern in ["detected_cells.xml", "cell_classification.xml", "*.xml"]:
            xmls = list(detection_path.glob(pattern))
            if xmls:
                return xmls[0]

    return None


def ensure_detection_in_pipeline(detection_xml: Path, brain_path: Path) -> bool:
    """Ensure detection XML is in 4_Cell_Candidates/ so 5_classify_cells.py can find it.

    If the detection is in optimization_runs or elsewhere, copies it to the
    standard pipeline location.
    """
    import shutil

    candidates_dir = brain_path / "4_Cell_Candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    target = candidates_dir / "detected_cells.xml"

    # If detection is already in the right place, nothing to do
    if detection_xml.parent == candidates_dir:
        return True

    # Copy to standard location (preserve original)
    print(f"  Copying detection to pipeline: {target}")
    shutil.copy2(detection_xml, target)
    return True


def run_classification(brain_path: Path, detection_xml: Path, model_path: Path, tracker: ExperimentTracker) -> dict:
    """Run classification and return results."""
    import subprocess

    script = Path(__file__).parent / "5_classify_cells.py"

    # Resolve detection path (may be directory from tracker)
    resolved_xml = resolve_detection_xml(detection_xml)
    if not resolved_xml:
        print(f"ERROR: No XML found at detection path: {detection_xml}")
        return None

    # Ensure detection is in standard pipeline location
    ensure_detection_in_pipeline(resolved_xml, brain_path)

    # Call 5_classify_cells.py with correct arguments
    cmd = [
        sys.executable, str(script),
        "--brain", brain_path.name,
        "--model", str(model_path),
    ]

    print(f"\n{'='*60}")
    print("CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Detection: {resolved_xml}")
    print(f"Model: {model_path}")
    print(f"Brain: {brain_path.name}")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Classification failed (see output above)")
        return None

    # Find output in 5_Classified_Cells
    output_dir = brain_path / "5_Classified_Cells"

    # Count cells in output
    cells_found = 0
    rejected = 0

    try:
        from brainglobe_utils.IO.cells import get_cells

        cells_xml = output_dir / "cells.xml"
        if cells_xml.exists():
            cells = get_cells(str(cells_xml))
            cells_found = len(cells) if cells else 0

        rejected_xml = output_dir / "rejected.xml"
        if rejected_xml.exists():
            rej = get_cells(str(rejected_xml))
            rejected = len(rej) if rej else 0
    except Exception as e:
        print(f"Warning: Could not count cells: {e}")

    return {
        'cells_found': cells_found,
        'rejected': rejected,
        'output_dir': output_dir,
        'model': model_path
    }


def compare_classifications(brain_name: str, new_result: dict, tracker: ExperimentTracker) -> dict:
    """Compare new classification to previous."""

    # Find previous classification
    prev_runs = tracker.search(
        brain=brain_name,
        exp_type="classification",
        status="completed",
        limit=10,
        sort_by="created_at",
        descending=True
    )

    # Find the best or most recent (excluding current)
    prev_run = None
    for run in prev_runs[1:]:  # Skip first (likely the one we just ran)
        if run.get('marked_best'):
            prev_run = run
            break
        elif prev_run is None:
            prev_run = run

    if not prev_run:
        return {
            'has_previous': False,
            'new_cells': new_result['cells_found'],
            'new_rejected': new_result['rejected']
        }

    prev_cells = int(prev_run.get('class_cells_found', 0) or 0)
    prev_rejected = int(prev_run.get('class_rejected', 0) or 0)

    diff = new_result['cells_found'] - prev_cells
    pct = (diff / prev_cells * 100) if prev_cells > 0 else 0

    return {
        'has_previous': True,
        'new_cells': new_result['cells_found'],
        'new_rejected': new_result['rejected'],
        'prev_cells': prev_cells,
        'prev_rejected': prev_rejected,
        'prev_exp_id': prev_run.get('exp_id'),
        'diff': diff,
        'pct_change': pct
    }


def save_summary(brain_path: Path, training_done: bool, classification: dict, comparison: dict, model_path: Path):
    """Save summary file for later reference."""
    summary_path = brain_path / "comparison_summary.txt"

    lines = [
        "=" * 60,
        "BATCH TRAIN & COMPARE SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Brain: {brain_path.name}",
        f"Model: {model_path}",
        f"Training: {'Completed' if training_done else 'Skipped'}",
        "",
        "CLASSIFICATION RESULTS",
        "-" * 40,
        f"Cells found: {classification['cells_found']}",
        f"Rejected: {classification['rejected']}",
        f"Acceptance rate: {classification['cells_found'] / (classification['cells_found'] + classification['rejected']) * 100:.1f}%" if (classification['cells_found'] + classification['rejected']) > 0 else "N/A",
        "",
    ]

    if comparison['has_previous']:
        lines.extend([
            "COMPARISON TO PREVIOUS",
            "-" * 40,
            f"Previous cells: {comparison['prev_cells']}",
            f"Previous rejected: {comparison['prev_rejected']}",
            f"Previous run: {comparison['prev_exp_id']}",
            "",
            f"Difference: {comparison['diff']:+d} cells ({comparison['pct_change']:+.1f}%)",
            "",
        ])
    else:
        lines.extend([
            "COMPARISON",
            "-" * 40,
            "No previous classification found for comparison.",
            "",
        ])

    lines.extend([
        "NEXT STEPS",
        "-" * 40,
        "1. Open napari: braintool",
        f"2. Load brain: {brain_path.name}",
        "3. Classification results are in 5_Classified_Cells/",
        "4. Use 'Load Historical Run' to compare visually",
        "",
        "=" * 60,
    ])

    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description='Overnight training + classification + comparison workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4 --epochs 200
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4 --skip-training
    python batch_train_compare.py --brain 368_CNT_03_08_1p625x_z4 --detection det_20260122_123640
        """
    )

    parser.add_argument('--brain', required=True, help='Brain name or path')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, use latest model')
    parser.add_argument('--detection', help='Specific detection exp_id to classify (default: most recent)')
    parser.add_argument('--model', help='Specific model path to use (default: latest)')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs (default: 150)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BATCH TRAIN & COMPARE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize tracker
    tracker = ExperimentTracker()

    # Find brain
    brain_path = find_brain_path(args.brain)
    if not brain_path:
        print(f"ERROR: Brain not found: {args.brain}")
        print(f"Searched in: {BRAINS_ROOT}")
        sys.exit(1)

    print(f"Brain: {brain_path}")

    training_done = False

    # Training
    if not args.skip_training:
        cells_dir, non_cells_dir = find_training_data(brain_path)
        if cells_dir and non_cells_dir:
            n_cells = len(list(cells_dir.glob("*.tif")))
            n_non = len(list(non_cells_dir.glob("*.tif")))
            print(f"Training data: {n_cells} cells, {n_non} non-cells")

            if n_cells > 0 and n_non > 0:
                training_done = run_training(cells_dir, non_cells_dir, epochs=args.epochs, learning_rate=args.learning_rate)
                if not training_done:
                    print("WARNING: Training failed, continuing with latest model")
            else:
                print("WARNING: Training data empty, skipping training")
        else:
            print("WARNING: No training data found, skipping training")
    else:
        print("Training: Skipped (--skip-training)")

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model()

    if not model_path or not model_path.exists():
        print("ERROR: No model found")
        sys.exit(1)

    print(f"Model: {model_path}")

    # Find detection
    detection_xml = find_detection_xml(brain_path, tracker, args.detection)
    if not detection_xml:
        print("ERROR: No detection XML found")
        print("Run detection first, or specify --detection")
        sys.exit(1)

    # Classification
    classification = run_classification(brain_path, detection_xml, model_path, tracker)
    if not classification:
        print("ERROR: Classification failed")
        sys.exit(1)

    # Comparison
    comparison = compare_classifications(brain_path.name, classification, tracker)

    # Save summary
    summary_path = save_summary(brain_path, training_done, classification, comparison, model_path)

    # Print final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Cells found: {classification['cells_found']}")
    print(f"Rejected: {classification['rejected']}")

    if comparison['has_previous']:
        print(f"\nComparison to previous ({comparison['prev_exp_id'][:20]}...):")
        print(f"  Previous: {comparison['prev_cells']} cells")
        print(f"  Difference: {comparison['diff']:+d} ({comparison['pct_change']:+.1f}%)")

    print(f"\nSummary saved to: {summary_path}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\nTo view results in napari:")
    print("  braintool")
    print(f"  Load brain: {brain_path.name}")
    print("  Results in: 5_Classified_Cells/")


if __name__ == '__main__':
    main()
