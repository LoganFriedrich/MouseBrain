#!/usr/bin/env python3
"""
util_train_model.py

Utility: Train a custom cellfinder classification model.

Trains a network using curated cell data, with auto-logging to experiment tracker.

================================================================================
HOW TO USE
================================================================================
    python util_train_model.py --cells path/to/cells --non-cells path/to/non_cells
    python util_train_model.py --yaml training_config.yaml
    python util_train_model.py --continue-from models/best_model.h5

================================================================================
REQUIREMENTS
================================================================================
    - cellfinder must be installed
    - Curated training data (cell and non-cell images)
    - experiment_tracker.py in same directory or PYTHONPATH
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from experiment_tracker import ExperimentTracker
except ImportError:
    print("ERROR: experiment_tracker.py not found!")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
from config import MODELS_DIR as DEFAULT_MODELS_DIR

# Default training parameters
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_FRACTION = 0.1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def run_cellfinder_train(
    yaml_paths: list,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    continue_from: Path = None,
    augment: bool = True,
    n_free_cpus: int = 2,
) -> tuple:
    """
    Run cellfinder training using the BrainGlobe Python API directly.

    Args:
        yaml_paths: List of paths to training.yml config file(s)
        output_dir: Where to save the trained model
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        test_fraction: Fraction of data for testing
        continue_from: Path to existing model to continue training
        augment: Whether to use data augmentation
        n_free_cpus: Number of CPUs to leave free (default: 2)

    Returns:
        (success, duration, best_loss, best_accuracy)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{timestamp()}] Running cellfinder training via Python API...")
    print(f"    YAML sources: {len(yaml_paths)}")
    for yp in yaml_paths:
        print(f"      - {yp}")
    print(f"    Output: {output_dir}")
    print(f"    Epochs: {epochs}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Batch size: {batch_size}")
    print(f"    Test fraction: {test_fraction}")
    print(f"    n_free_cpus: {n_free_cpus}")
    print(f"    Augmentation: {augment}")
    if continue_from:
        print(f"    Continue from: {continue_from}")
    print()

    start_time = time.time()

    try:
        # Import the training module directly
        from cellfinder.core.train.train_yml import run as train_run

        # Call the Python API directly - this gives us better control
        # Enable save_progress to write training.csv for monitoring
        train_run(
            output_dir=output_dir,
            yaml_file=[str(p) for p in yaml_paths],
            n_free_cpus=n_free_cpus,
            trained_model=str(continue_from) if continue_from else None,
            model_weights=None,  # Will use default
            network_depth="50",
            learning_rate=learning_rate,
            continue_training=bool(continue_from),
            test_fraction=test_fraction,
            batch_size=batch_size,
            no_augment=not augment,
            tensorboard=True,  # Enable TensorBoard for monitoring
            save_weights=False,
            no_save_checkpoints=False,
            save_progress=True,  # Write training.csv for monitoring
            epochs=epochs,
        )

        duration = time.time() - start_time

        # Parse training.csv for final metrics
        csv_file = output_dir / "training.csv"
        best_loss = None
        best_accuracy = None

        if csv_file.exists():
            try:
                import csv
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        best_loss = float(last_row.get('val_loss', 0))
                        best_accuracy = float(last_row.get('val_accuracy', 0))
                        print(f"\n[{timestamp()}] Final metrics: loss={best_loss:.4f}, accuracy={best_accuracy:.4f}")
            except Exception as e:
                print(f"Warning: Could not parse training.csv: {e}")

        return True, duration, best_loss, best_accuracy

    except Exception as e:
        import traceback
        print(f"\n[{timestamp()}] ERROR during training: {e}")
        traceback.print_exc()
        return False, time.time() - start_time, None, None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder training with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python util_train_model.py --cells data/cells --non-cells data/non_cells
    python util_train_model.py --yaml training_data/training.yml
    python util_train_model.py --continue-from models/best.h5 --cells data/cells --non-cells data/non_cells
        """
    )

    parser.add_argument('--yaml', '-y', type=Path, nargs='+',
                        help='Path(s) to training.yml file(s). Multiple sources supported.')
    parser.add_argument('--cells', '-c', type=Path,
                        help='Path to cells training data')
    parser.add_argument('--non-cells', '-n', type=Path,
                        help='Path to non-cells training data')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output directory (default: timestamped in models dir)')
    parser.add_argument('--name', help='Model name (for output folder)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--test-fraction', type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument('--continue-from', type=Path, help='Continue from existing model')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--n-free-cpus', type=int, default=2,
                        help='Number of CPUs to leave free (default: 2)')

    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()

    # Handle --yaml option: parse training.yml to get cells/non-cells paths
    if args.yaml:
        yaml_paths = args.yaml if isinstance(args.yaml, list) else [args.yaml]

        # Validate all paths exist
        for yp in yaml_paths:
            if not yp.exists():
                print(f"ERROR: YAML file not found: {yp}")
                sys.exit(1)

        # Parse first YAML to get cells/non-cells paths for counting
        try:
            import yaml
            with open(yaml_paths[0]) as f:
                config = yaml.safe_load(f)
        except ImportError:
            config = {'data': []}
            with open(yaml_paths[0]) as f:
                current_item = {}
                for line in f:
                    line = line.strip()
                    if line.startswith('- '):
                        if current_item:
                            config['data'].append(current_item)
                        current_item = {}
                        if ':' in line[2:]:
                            key, val = line[2:].split(':', 1)
                            current_item[key.strip()] = val.strip().strip("'\"")
                    elif ':' in line and current_item is not None:
                        key, val = line.split(':', 1)
                        current_item[key.strip()] = val.strip().strip("'\"")
                if current_item:
                    config['data'].append(current_item)

        for item in config.get('data', []):
            if item.get('type') == 'cell':
                args.cells = Path(item.get('cube_dir', ''))
            elif item.get('type') == 'no_cell':
                args.non_cells = Path(item.get('cube_dir', ''))

        print(f"Loaded {len(yaml_paths)} YAML source(s)")
        for yp in yaml_paths:
            print(f"  - {yp}")

    # Determine YAML paths - either provided directly or infer from cells/non-cells
    if args.yaml:
        yaml_paths = args.yaml if isinstance(args.yaml, list) else [args.yaml]
    else:
        yaml_paths = None

    if not yaml_paths:
        # If --cells and --non-cells provided, look for training.yml in parent
        if args.cells and args.non_cells:
            # Look for training.yml in the parent folder of cells/
            potential_yaml = args.cells.parent / "training.yml"
            if potential_yaml.exists():
                yaml_paths = [potential_yaml]
            else:
                # Create a temporary training.yml
                yaml_paths = [args.cells.parent / "training.yml"]
                yaml_content = f"""data:
- bg_channel: 1
  cell_def: ''
  cube_dir: {args.cells}
  signal_channel: 0
  type: cell
- bg_channel: 1
  cell_def: ''
  cube_dir: {args.non_cells}
  signal_channel: 0
  type: no_cell
"""
                with open(yaml_paths[0], 'w') as f:
                    f.write(yaml_content)
                print(f"Created training.yml: {yaml_paths[0]}")
        else:
            print("ERROR: Must provide --yaml or both --cells and --non-cells")
            sys.exit(1)

    # Validate inputs
    if not all(yp.exists() for yp in yaml_paths):
        for yp in yaml_paths:
            if not yp.exists():
                print(f"ERROR: YAML file not found: {yp}")
        sys.exit(1)

    # If cells/non-cells not set, try to get from yaml for counting
    if not args.cells or not args.non_cells:
        # Already parsed above in the yaml handling block
        pass

    if args.cells and not args.cells.exists():
        print(f"ERROR: Cells path not found: {args.cells}")
        sys.exit(1)
    if args.non_cells and not args.non_cells.exists():
        print(f"ERROR: Non-cells path not found: {args.non_cells}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_dir = args.output
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.name or "model"
        output_dir = DEFAULT_MODELS_DIR / f"{name}_{timestamp_str}"

    print("=" * 60)
    print("BrainGlobe Cell Classification Training")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"YAML sources: {len(yaml_paths)}")
        for yp in yaml_paths:
            print(f"  - {yp}")
        print(f"Cells: {args.cells}")
        print(f"Non-cells: {args.non_cells}")
        print(f"Output: {output_dir}")
        print(f"Epochs: {args.epochs}")
        return

    # Count training samples
    n_cells = len(list(args.cells.glob("*.tif*"))) if args.cells else 0
    n_non_cells = len(list(args.non_cells.glob("*.tif*"))) if args.non_cells else 0
    print(f"\nTraining data: {n_cells} cells, {n_non_cells} non-cells")

    # Initialize tracker
    tracker = ExperimentTracker()

    brain_name = args.name or (args.cells.parent.name if args.cells else yaml_paths[0].parent.name)

    exp_id = tracker.log_training(
        brain=brain_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        augment=not args.no_augment,
        pretrained=str(args.continue_from) if args.continue_from else None,
        input_path=str(yaml_paths[0].parent),
        output_path=str(output_dir),
        notes=args.notes,
        status="started",
        script_version=SCRIPT_VERSION,
    )

    print(f"\n{'='*60}")
    print(f"Training Run: {exp_id}")
    print(f"{'='*60}")

    # Run training
    success, duration, best_loss, best_accuracy = run_cellfinder_train(
        yaml_paths=yaml_paths,
        output_dir=output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        test_fraction=args.test_fraction,
        continue_from=args.continue_from,
        augment=not args.no_augment,
        n_free_cpus=args.n_free_cpus,
    )
    
    # Update tracker
    tracker.update_status(
        exp_id,
        status="completed" if success else "failed",
        duration_seconds=round(duration, 1),
        train_loss=best_loss,
        train_accuracy=best_accuracy,
    )
    
    print(f"\n{'='*60}")
    if success:
        print(f"COMPLETED in {duration/60:.1f} minutes")
        print(f"Model saved to: {output_dir}")
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


if __name__ == '__main__':
    main()
