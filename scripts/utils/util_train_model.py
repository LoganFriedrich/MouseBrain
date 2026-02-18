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
    python util_train_model.py --yaml t.yml --post-run "python 5_classify_cells.py --brain {brain} --model {output_dir}"

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
    from mousebrain.tracker import ExperimentTracker
except ImportError:
    print("ERROR: mousebrain.tracker not found!")
    print("Make sure mousebrain package is installed.")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

from mousebrain.config import BRAINS_ROOT, MODELS_DIR

# Aliases for backward compat
DEFAULT_BRAINGLOBE_ROOT = BRAINS_ROOT
DEFAULT_MODELS_DIR = MODELS_DIR

# Default training parameters
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.00005
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_FRACTION = 0.1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def discover_training_yamls(paradigm: str) -> list:
    """Auto-discover all training.yml files for a given paradigm (e.g. '1p625x_z4').

    Searches:
      1. Shared pool: MODELS_DIR/{paradigm}/training_data/training.yml
      2. Per-brain:   BRAINS_ROOT/{mouse}/{brain}/training_data/training.yml
         where {brain} ends with _{paradigm}

    Returns list of Paths, shared pool first, then per-brain sorted alphabetically.
    Only includes YAMLs whose cube directories actually contain TIFFs.
    """
    found = []

    # 1. Shared model pool
    shared = MODELS_DIR / paradigm / "training_data" / "training.yml"
    if shared.exists():
        found.append(shared)

    # 2. Per-brain directories
    per_brain = []
    if BRAINS_ROOT.exists():
        for mouse_dir in sorted(BRAINS_ROOT.iterdir()):
            if not mouse_dir.is_dir():
                continue
            for brain_dir in sorted(mouse_dir.iterdir()):
                if not brain_dir.is_dir():
                    continue
                if not brain_dir.name.endswith(f"_{paradigm}"):
                    continue
                yml = brain_dir / "training_data" / "training.yml"
                if yml.exists():
                    # Check that at least one cube dir has TIFFs
                    cells_dir = brain_dir / "training_data" / "cells"
                    non_cells_dir = brain_dir / "training_data" / "non_cells"
                    has_data = False
                    for d in [cells_dir, non_cells_dir]:
                        if d.exists() and any(d.glob("*.tif")):
                            has_data = True
                            break
                    if has_data:
                        per_brain.append(yml)

    found.extend(per_brain)
    return found


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
    early_stopping: bool = True,
    reduce_lr: bool = True,
    nan_guard: bool = True,
    patience: int = 15,
) -> tuple:
    """
    Run cellfinder training with custom Keras callbacks for safety.

    Uses cellfinder's internal components (model builder, data generators) but
    runs model.fit() ourselves so we can add EarlyStopping, ReduceLROnPlateau,
    and TerminateOnNaN callbacks.

    Args:
        yaml_paths: List of paths to training.yml config file(s)
        output_dir: Where to save the trained model
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        test_fraction: Fraction of data for testing
        continue_from: Path to existing model to continue training
        augment: Whether to use data augmentation
        n_free_cpus: Number of CPUs to leave free
        early_stopping: Enable early stopping when val_loss stops improving
        reduce_lr: Enable learning rate reduction on plateau
        nan_guard: Enable NaN termination guard
        patience: Early stopping patience (epochs)

    Returns:
        (success, duration, best_loss, best_accuracy)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{timestamp()}] Running cellfinder training with safety callbacks...")
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
    print(f"    Early stopping: {early_stopping} (patience={patience})")
    print(f"    Reduce LR on plateau: {reduce_lr}")
    print(f"    NaN guard: {nan_guard}")
    if continue_from:
        print(f"    Continue from: {continue_from}")
    print()

    start_time = time.time()

    try:
        return _run_with_callbacks(
            yaml_paths, output_dir, epochs, learning_rate, batch_size,
            test_fraction, continue_from, augment, n_free_cpus,
            early_stopping, reduce_lr, nan_guard, patience,
        )
    except Exception as e:
        print(f"\n[{timestamp()}] Custom training loop failed: {e}")
        print(f"[{timestamp()}] Falling back to cellfinder's built-in training...")
        try:
            return _run_cellfinder_fallback(
                yaml_paths, output_dir, epochs, learning_rate, batch_size,
                test_fraction, continue_from, augment, n_free_cpus,
            )
        except Exception as e2:
            import traceback
            print(f"\n[{timestamp()}] ERROR during fallback training: {e2}")
            traceback.print_exc()
            return False, time.time() - start_time, None, None


def _run_with_callbacks(
    yaml_paths, output_dir, epochs, learning_rate, batch_size,
    test_fraction, continue_from, augment, n_free_cpus,
    early_stopping, reduce_lr, nan_guard, patience,
) -> tuple:
    """Custom training loop with Keras callbacks for safety."""
    import numpy as np

    start_time = time.time()

    # --- 1. Parse YAML to get data paths ---
    # Accumulate signal/background file lists from all YAML sources
    all_signal = []
    all_background = []
    all_labels = []

    for yp in yaml_paths:
        try:
            import yaml as _yaml
            with open(yp) as f:
                config = _yaml.safe_load(f)
        except ImportError:
            # Manual YAML parsing fallback
            config = {'data': []}
            with open(yp) as f:
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
            cube_dir = Path(item.get('cube_dir', ''))
            item_type = item.get('type', '')
            signal_ch = int(item.get('signal_channel', 0))
            bg_ch = int(item.get('bg_channel', 1))

            if not cube_dir.exists():
                print(f"[{timestamp()}] Warning: cube_dir not found: {cube_dir}")
                continue

            # Find all signal channel TIFFs
            signal_files = sorted(cube_dir.glob(f"*Ch{signal_ch}.tif*"))
            for sf in signal_files:
                # Find matching background file
                bg_name = sf.name.replace(f"Ch{signal_ch}.", f"Ch{bg_ch}.")
                bf = sf.parent / bg_name
                if bf.exists():
                    all_signal.append(str(sf))
                    all_background.append(str(bf))
                    all_labels.append(1 if item_type == 'cell' else 0)

    if not all_signal:
        raise ValueError("No training data found in YAML sources")

    labels_array = np.array(all_labels)
    n_cells = int(np.sum(labels_array == 1))
    n_non_cells = int(np.sum(labels_array == 0))
    print(f"[{timestamp()}] Loaded {len(all_signal)} samples: {n_cells} cells, {n_non_cells} non-cells")

    # --- 2. Split into train/validation ---
    from sklearn.model_selection import train_test_split

    (signal_train, signal_test,
     bg_train, bg_test,
     labels_train, labels_test) = train_test_split(
        all_signal, all_background, all_labels,
        test_size=test_fraction,
        random_state=42,
        stratify=all_labels,
    )

    print(f"[{timestamp()}] Train: {len(signal_train)} samples, Validation: {len(signal_test)} samples")

    # --- 3. Build model ---
    from cellfinder.core.classify.tools import get_model

    if continue_from and Path(continue_from).exists():
        print(f"[{timestamp()}] Loading model from: {continue_from}")
        model = get_model(
            existing_model=str(continue_from),
            network_depth=None,
            learning_rate=learning_rate,
            continue_training=True,
        )
    else:
        print(f"[{timestamp()}] Building new ResNet-50 model...")
        model = get_model(
            network_depth="50-layer",
            learning_rate=learning_rate,
        )

    # --- 4. Create data generators ---
    from cellfinder.core.classify.cube_generator import CubeGeneratorFromDisk

    training_generator = CubeGeneratorFromDisk(
        signal_train, bg_train,
        labels=labels_train,
        batch_size=batch_size,
        shuffle=True,
        train=True,
        augment=augment,
    )

    validation_generator = CubeGeneratorFromDisk(
        signal_test, bg_test,
        labels=labels_test,
        batch_size=batch_size,
        train=True,
    )

    # --- 5. Set up callbacks ---
    import tensorflow as tf

    callbacks = []

    # Model checkpoint - save best model
    checkpoint_path = str(output_dir / "model-epoch.{epoch:02d}-loss-{val_loss:.4f}.h5")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=False,
        verbose=1,
    ))

    # CSV logger for monitoring
    csv_path = str(output_dir / "training.csv")
    callbacks.append(tf.keras.callbacks.CSVLogger(csv_path))

    # TensorBoard
    tb_dir = str(output_dir / "tensorboard")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tb_dir))

    # --- Safety callbacks (the whole point of this custom loop) ---
    if nan_guard:
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
        print(f"[{timestamp()}] NaN guard: ON - training will abort if NaN loss detected")

    if reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(patience // 2, 3),
            min_lr=1e-7,
            verbose=1,
        ))
        print(f"[{timestamp()}] Reduce LR: ON - LR halved if val_loss plateaus for {max(patience // 2, 3)} epochs")

    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ))
        print(f"[{timestamp()}] Early stopping: ON - training stops if val_loss doesn't improve for {patience} epochs")

    # Also save the best model separately
    best_model_path = str(output_dir / "best_model.h5")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    ))

    # --- 6. Train! ---
    print(f"\n[{timestamp()}] Starting training for {epochs} epochs...")
    print(f"{'='*60}")

    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False,
        epochs=epochs,
        callbacks=callbacks,
    )

    duration = time.time() - start_time

    # --- 7. Save final model ---
    final_model_path = output_dir / "model.keras"
    try:
        model.save(str(final_model_path))
        print(f"\n[{timestamp()}] Final model saved: {final_model_path}")
    except Exception:
        # Fallback to .h5 format
        final_model_path = output_dir / "model.h5"
        model.save(str(final_model_path))
        print(f"\n[{timestamp()}] Final model saved: {final_model_path}")

    # --- 8. Extract best metrics ---
    best_loss = None
    best_accuracy = None

    if 'val_loss' in history.history:
        val_losses = history.history['val_loss']
        # Filter out NaN values
        valid_losses = [(i, l) for i, l in enumerate(val_losses) if not np.isnan(l)]
        if valid_losses:
            best_epoch, best_loss = min(valid_losses, key=lambda x: x[1])
            if 'val_accuracy' in history.history:
                best_accuracy = history.history['val_accuracy'][best_epoch]
            print(f"\n[{timestamp()}] Best validation loss: {best_loss:.4f} at epoch {best_epoch + 1}")
            if best_accuracy is not None:
                print(f"[{timestamp()}] Best validation accuracy: {best_accuracy:.4f}")

    actual_epochs = len(history.history.get('loss', []))
    if actual_epochs < epochs:
        print(f"\n[{timestamp()}] Training stopped early at epoch {actual_epochs}/{epochs}")
        if early_stopping:
            print(f"[{timestamp()}] (Early stopping triggered - best weights restored)")

    return True, duration, best_loss, best_accuracy


def _run_cellfinder_fallback(
    yaml_paths, output_dir, epochs, learning_rate, batch_size,
    test_fraction, continue_from, augment, n_free_cpus,
) -> tuple:
    """Fallback: use cellfinder's built-in training (no custom callbacks)."""
    start_time = time.time()

    from cellfinder.core.train.train_yml import run as train_run

    train_run(
        output_dir=output_dir,
        yaml_file=[str(p) for p in yaml_paths],
        n_free_cpus=n_free_cpus,
        trained_model=str(continue_from) if continue_from else None,
        model_weights=None,
        network_depth="50-layer",
        learning_rate=learning_rate,
        continue_training=bool(continue_from),
        test_fraction=test_fraction,
        batch_size=batch_size,
        no_augment=not augment,
        tensorboard=True,
        save_weights=False,
        no_save_checkpoints=False,
        save_progress=True,
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder training with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simplest: auto-discover all training data for paradigm
    python util_train_model.py --paradigm 1p625x_z4

    # Explicit YAML sources
    python util_train_model.py --yaml training1.yml training2.yml

    # Direct paths
    python util_train_model.py --cells data/cells --non-cells data/non_cells
        """
    )

    parser.add_argument('--paradigm', '-p',
                        help='Auto-discover all training.yml for this imaging paradigm '
                             '(e.g. "1p625x_z4"). Finds shared pool + all per-brain data.')
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
    import os as _os
    _cpu_count = _os.cpu_count() or 4
    _default_free = max(_cpu_count - 30, 2) if _cpu_count > 63 else 2
    parser.add_argument('--n-free-cpus', type=int, default=_default_free,
                        help=f'Number of CPUs to leave free (default: {_default_free}, '
                             f'auto-computed for {_cpu_count} CPUs)')

    # Safety callbacks (all ON by default)
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping (NOT recommended)')
    parser.add_argument('--no-reduce-lr', action='store_true',
                        help='Disable automatic learning rate reduction (NOT recommended)')
    parser.add_argument('--no-nan-guard', action='store_true',
                        help='Disable NaN termination guard (NOT recommended)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience in epochs (default: 15)')

    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true')

    # Post-training automation
    parser.add_argument('--post-run', nargs='+',
                        help='Command(s) to run after successful training. '
                             'Supports template vars: {output_dir}, {brain}, {exp_id}. '
                             'Example: --post-run "python 5_classify_cells.py --brain {brain} --model {output_dir}"')
    
    args = parser.parse_args()

    # Handle --paradigm: auto-discover all training YAMLs
    if args.paradigm:
        if args.yaml:
            print("ERROR: --paradigm and --yaml are mutually exclusive.")
            sys.exit(1)
        discovered = discover_training_yamls(args.paradigm)
        if not discovered:
            print(f"ERROR: No training data found for paradigm '{args.paradigm}'")
            print(f"  Searched: {MODELS_DIR / args.paradigm / 'training_data'}")
            print(f"  Searched: {BRAINS_ROOT} / */  *_{args.paradigm} / training_data/")
            sys.exit(1)
        args.yaml = discovered
        print(f"Auto-discovered {len(discovered)} training source(s) for paradigm '{args.paradigm}':")
        for yp in discovered:
            print(f"  - {yp}")
        # Default name from paradigm if not set
        if not args.name:
            args.name = args.paradigm

    # Handle --yaml option: parse training.yml to get cells/non-cells paths
    if args.yaml:
        yaml_paths = args.yaml if isinstance(args.yaml, list) else [args.yaml]

        # Validate all paths exist
        for yp in yaml_paths:
            if not yp.exists():
                print(f"ERROR: YAML file not found: {yp}")
                sys.exit(1)

        # Parse ALL YAMLs to get cells/non-cells paths for counting
        all_cell_dirs = []
        all_non_cell_dirs = []
        for yp in yaml_paths:
            try:
                import yaml
                with open(yp) as f:
                    config = yaml.safe_load(f)
            except ImportError:
                config = {'data': []}
                with open(yp) as f:
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
                    all_cell_dirs.append(Path(item.get('cube_dir', '')))
                elif item.get('type') == 'no_cell':
                    all_non_cell_dirs.append(Path(item.get('cube_dir', '')))

        # Set args.cells/non_cells to first entry (for downstream compatibility)
        if all_cell_dirs:
            args.cells = all_cell_dirs[0]
        if all_non_cell_dirs:
            args.non_cells = all_non_cell_dirs[0]
        # Store all dirs for accurate counting
        args._all_cell_dirs = all_cell_dirs
        args._all_non_cell_dirs = all_non_cell_dirs

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
    # Count cubes across all YAML sources (each cube = Ch0 + Ch1 pair)
    cell_dirs = getattr(args, '_all_cell_dirs', [args.cells] if args.cells else [])
    non_cell_dirs = getattr(args, '_all_non_cell_dirs', [args.non_cells] if args.non_cells else [])
    n_cells = sum(len(list(d.glob("*Ch0.tif"))) for d in cell_dirs if d and d.exists())
    n_non_cells = sum(len(list(d.glob("*Ch0.tif"))) for d in non_cell_dirs if d and d.exists())
    print(f"\nTraining data: {n_cells} cells, {n_non_cells} non-cells (across {len(cell_dirs)} source(s))")

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
        early_stopping=not args.no_early_stopping,
        reduce_lr=not args.no_reduce_lr,
        nan_guard=not args.no_nan_guard,
        patience=args.patience,
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
        # Run post-training commands if specified
        if args.post_run:
            print(f"\n{'='*60}")
            print("POST-TRAINING COMMANDS")
            print(f"{'='*60}")
            for cmd_template in args.post_run:
                cmd = cmd_template.format(
                    output_dir=output_dir,
                    brain=brain_name,
                    exp_id=exp_id,
                )
                print(f"\n[{timestamp()}] Running: {cmd}")
                try:
                    result = subprocess.run(
                        cmd, shell=True, cwd=str(Path(__file__).parent),
                    )
                    if result.returncode == 0:
                        print(f"[{timestamp()}] Post-run completed successfully")
                    else:
                        print(f"[{timestamp()}] Post-run exited with code {result.returncode}")
                except Exception as e:
                    print(f"[{timestamp()}] Post-run error: {e}")

        # Interactive rating (skipped if stdin is not a terminal)
        try:
            if sys.stdin.isatty():
                rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
                if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                    note = input("Add a note (or Enter to skip): ").strip()
                    tracker.rate_experiment(exp_id, int(rating), note if note else None)
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == '__main__':
    main()
