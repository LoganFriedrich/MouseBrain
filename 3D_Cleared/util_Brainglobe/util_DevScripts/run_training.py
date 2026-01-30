#!/usr/bin/env python3
"""
run_training.py

Wrapper for cellfinder training that auto-logs to experiment tracker.

This trains a neural network to classify cell candidates as real cells
or false positives, using manually curated training data.

================================================================================
TRAINING DATA
================================================================================
You need two folders of cell cubes:
- cells/     : Examples of REAL cells (true positives)
- non_cells/ : Examples of NOT cells (false positives, debris, etc.)

These are typically created by:
1. Running detection on a brain
2. Opening results in napari
3. Manually sorting candidates into cells vs non_cells folders

================================================================================
USAGE
================================================================================
Basic training:
    python run_training.py --cells path/to/cells --non-cells path/to/non_cells

Continue from existing model:
    python run_training.py --cells cells/ --non-cells non_cells/ \\
        --continue-from path/to/existing_model.h5

Custom parameters:
    python run_training.py --cells cells/ --non-cells non_cells/ \\
        --epochs 100 --learning-rate 0.0001 --batch-size 16

================================================================================
OUTPUT
================================================================================
Trained model saved to: util_Brainglobe/Trained_Models/model_YYYYMMDD_HHMMSS/
    - model.h5           : The trained network
    - training_log.csv   : Loss/accuracy per epoch
    - training_config.json : Parameters used

================================================================================
REQUIREMENTS
================================================================================
    conda activate brainglobe-env
    # cellfinder should already be installed
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODELS_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\util_Brainglobe\Trained_Models")

# Default training parameters
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 32
DEFAULT_NETWORK_DEPTH = 50  # ResNet50


# =============================================================================
# TRAINING
# =============================================================================

def count_training_data(cells_folder: Path, non_cells_folder: Path) -> Tuple[int, int]:
    """Count training examples in each folder."""
    # Training data can be .tif cubes or folders of slices
    cells_count = 0
    non_cells_count = 0
    
    # Count .tif files
    cells_count += len(list(cells_folder.glob("*.tif")))
    cells_count += len(list(cells_folder.glob("*.tiff")))
    
    non_cells_count += len(list(non_cells_folder.glob("*.tif")))
    non_cells_count += len(list(non_cells_folder.glob("*.tiff")))
    
    # Count subdirectories (each is one example)
    for item in cells_folder.iterdir():
        if item.is_dir():
            cells_count += 1
    
    for item in non_cells_folder.iterdir():
        if item.is_dir():
            non_cells_count += 1
    
    return cells_count, non_cells_count


def run_cellfinder_train(
    cells_folder: Path,
    non_cells_folder: Path,
    output_folder: Path,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    network_depth: int = DEFAULT_NETWORK_DEPTH,
    continue_from: Path = None,
    augment: bool = True,
) -> Tuple[bool, float, float, float]:
    """
    Run cellfinder training.
    
    Returns:
        (success, duration_seconds, final_loss, final_accuracy)
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "cellfinder_train",
        "--yaml", str(output_folder / "training_config.yaml"),  # Will be created
        "-y", str(cells_folder),
        "-n", str(non_cells_folder),
        "-o", str(output_folder),
        "--epochs", str(epochs),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--network-depth", str(network_depth),
    ]
    
    if continue_from:
        cmd.extend(["--continue-training", str(continue_from)])
    
    if augment:
        cmd.append("--augment")
    
    print(f"\nRunning: cellfinder_train")
    print(f"  Cells: {cells_folder}")
    print(f"  Non-cells: {non_cells_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Try to parse final metrics from training log
            final_loss, final_accuracy = parse_training_results(output_folder)
            return True, duration, final_loss, final_accuracy
        else:
            return False, duration, 0.0, 0.0
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error: {e}")
        return False, duration, 0.0, 0.0


def parse_training_results(output_folder: Path) -> Tuple[float, float]:
    """Parse final loss and accuracy from training log."""
    log_path = output_folder / "training_log.csv"
    
    if not log_path.exists():
        return 0.0, 0.0
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return 0.0, 0.0
        
        # Parse last line (final epoch)
        last_line = lines[-1].strip()
        parts = last_line.split(',')
        
        # Typically: epoch, loss, accuracy, val_loss, val_accuracy
        if len(parts) >= 5:
            val_loss = float(parts[3])
            val_accuracy = float(parts[4])
            return val_loss, val_accuracy
        elif len(parts) >= 3:
            loss = float(parts[1])
            accuracy = float(parts[2])
            return loss, accuracy
    
    except Exception:
        pass
    
    return 0.0, 0.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train cellfinder classification network with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python run_training.py --cells path/to/cells --non-cells path/to/non_cells

  # Continue from existing model
  python run_training.py --cells cells/ --non-cells non_cells/ \\
      --continue-from path/to/model.h5

  # Custom parameters
  python run_training.py --cells cells/ --non-cells non_cells/ \\
      --epochs 100 --learning-rate 0.0001 --batch-size 16

  # Dry run
  python run_training.py --cells cells/ --non-cells non_cells/ --dry-run

Training Data:
  cells/     - Folder of real cell examples (cubes or tiff stacks)
  non_cells/ - Folder of non-cell examples (false positives)
        """
    )
    
    # Required
    parser.add_argument('--cells', '-y', type=Path, required=True,
                        help='Folder of cell (positive) examples')
    parser.add_argument('--non-cells', '-n', type=Path, required=True,
                        help='Folder of non-cell (negative) examples')
    
    # Training parameters
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--learning-rate', '-lr', type=float, default=DEFAULT_LEARNING_RATE,
                        help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument('--batch-size', '-bs', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--network-depth', type=int, default=DEFAULT_NETWORK_DEPTH,
                        choices=[18, 34, 50, 101, 152],
                        help=f'ResNet depth (default: {DEFAULT_NETWORK_DEPTH})')
    parser.add_argument('--continue-from', type=Path,
                        help='Continue training from existing model')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    
    # Output
    parser.add_argument('--output', '-o', type=Path,
                        help='Output folder (default: auto-generated in Trained_Models)')
    parser.add_argument('--name', help='Model name (used in output folder name)')
    
    # Options
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without running')
    parser.add_argument('--notes', help='Notes to add to experiment log')
    
    args = parser.parse_args()
    
    # Validate input folders
    if not args.cells.exists():
        print(f"Error: Cells folder not found: {args.cells}")
        sys.exit(1)
    
    if not args.non_cells.exists():
        print(f"Error: Non-cells folder not found: {args.non_cells}")
        sys.exit(1)
    
    # Count training data
    num_cells, num_non_cells = count_training_data(args.cells, args.non_cells)
    
    print(f"\n{'='*60}")
    print(f"Cell Classification Training")
    print(f"{'='*60}")
    print(f"Cells folder: {args.cells}")
    print(f"Non-cells folder: {args.non_cells}")
    print(f"Training examples: {num_cells} cells, {num_non_cells} non-cells")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Network depth: ResNet{args.network_depth}")
    print(f"Augmentation: {'disabled' if args.no_augment else 'enabled'}")
    
    if args.continue_from:
        print(f"Continuing from: {args.continue_from}")
    
    # Warn if data is unbalanced
    if num_cells == 0 or num_non_cells == 0:
        print(f"\nError: Need examples in both folders!")
        sys.exit(1)
    
    ratio = num_cells / num_non_cells if num_non_cells > 0 else float('inf')
    if ratio < 0.5 or ratio > 2.0:
        print(f"\nWarning: Unbalanced training data (ratio: {ratio:.2f})")
        print("  Consider balancing for better results")
    
    # Determine output folder
    if args.output:
        output_folder = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part = f"_{args.name}" if args.name else ""
        output_folder = DEFAULT_MODELS_ROOT / f"model_{timestamp}{name_part}"
    
    print(f"Output: {output_folder}")
    
    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would train model with {num_cells + num_non_cells} examples")
        sys.exit(0)
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Log experiment start
    exp_id = tracker.log_training(
        brain=f"training_{datetime.now().strftime('%Y%m%d')}",
        network_depth=args.network_depth,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        continue_from=str(args.continue_from) if args.continue_from else None,
        num_positive=num_cells,
        num_negative=num_non_cells,
        augment="disabled" if args.no_augment else "enabled",
        output_path=str(output_folder),
        status="started",
        notes=args.notes,
    )
    
    print(f"\nExperiment: {exp_id}")
    
    # Run training
    success, duration, final_loss, final_accuracy = run_cellfinder_train(
        cells_folder=args.cells,
        non_cells_folder=args.non_cells,
        output_folder=output_folder,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        network_depth=args.network_depth,
        continue_from=args.continue_from,
        augment=not args.no_augment,
    )
    
    # Update experiment status
    if success:
        tracker.update_status(
            exp_id,
            status="completed",
            duration_seconds=round(duration, 1),
            train_final_loss=round(final_loss, 4),
            train_final_accuracy=round(final_accuracy, 4),
        )
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Model saved to: {output_folder}")
        print(f"Experiment: {exp_id}")
        
        # Prompt for rating
        print()
        try:
            rating = input("Rate this training run (1-5, or Enter to skip): ").strip()
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
        )
        
        print(f"\n{'='*60}")
        print(f"TRAINING FAILED")
        print(f"{'='*60}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Experiment: {exp_id}")
        sys.exit(1)


if __name__ == '__main__':
    main()
