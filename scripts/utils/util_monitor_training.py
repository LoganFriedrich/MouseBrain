#!/usr/bin/env python3
"""
util_monitor_training.py

Monitors a running cellfinder training and executes post-training actions.

Watches training.csv in the output directory for epoch completion.
When training finishes, runs configured post-actions automatically.

================================================================================
HOW TO USE
================================================================================
    # Monitor and run classification when done
    python util_monitor_training.py --training-dir path/to/model_output --epochs 100 \
        --then "python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --model {training_dir}"

    # Monitor with folder cleanup actions (one-time maintenance)
    python util_monitor_training.py --training-dir path/to/model_output --epochs 100 \
        --cleanup-folders

    # Just monitor, no post-actions
    python util_monitor_training.py --training-dir path/to/model_output --epochs 100
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.config import BRAINS_ROOT
except ImportError:
    BRAINS_ROOT = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\3D_Cleared\1_Brains")

SCRIPT_DIR = Path(__file__).parent


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def check_training_progress(training_dir: Path, expected_epochs: int) -> tuple:
    """
    Check training.csv for progress.

    Returns: (is_complete, current_epoch, total_epochs, last_loss, last_accuracy)
    """
    csv_file = training_dir / "training.csv"
    if not csv_file.exists():
        return False, 0, expected_epochs, None, None

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return False, 0, expected_epochs, None, None

        current_epoch = len(rows)
        last_row = rows[-1]
        last_loss = last_row.get('val_loss', 'nan')
        last_acc = last_row.get('val_accuracy', 'nan')

        is_complete = current_epoch >= expected_epochs
        return is_complete, current_epoch, expected_epochs, last_loss, last_acc

    except Exception:
        return False, 0, expected_epochs, None, None


def merge_training_data(src_dir: Path, dst_dir: Path, dry_run: bool = False) -> int:
    """
    Merge training_data from src into dst, keeping more complete version.

    Copies any files from src that don't exist in dst (or are newer/larger).
    Returns number of files copied.
    """
    copied = 0
    for subdir_name in ["cells", "non_cells"]:
        src_sub = src_dir / subdir_name
        dst_sub = dst_dir / subdir_name

        if not src_sub.exists():
            continue

        dst_sub.mkdir(parents=True, exist_ok=True)

        src_files = list(src_sub.iterdir())
        dst_files = {f.name for f in dst_sub.iterdir()} if dst_sub.exists() else set()

        new_files = [f for f in src_files if f.name not in dst_files]
        if new_files:
            print(f"    {subdir_name}: {len(new_files)} new files to copy "
                  f"({len(src_files)} src, {len(dst_files)} dst)")
            if not dry_run:
                for f in new_files:
                    shutil.copy2(str(f), str(dst_sub / f.name))
            copied += len(new_files)
        else:
            print(f"    {subdir_name}: dst already has all {len(dst_files)} files")

    # Copy training.yml if src has one and dst doesn't (or src is newer)
    src_yml = src_dir / "training.yml"
    dst_yml = dst_dir / "training.yml"
    if src_yml.exists() and (not dst_yml.exists() or src_yml.stat().st_mtime > dst_yml.stat().st_mtime):
        if not dry_run:
            shutil.copy2(str(src_yml), str(dst_yml))
        print(f"    training.yml: copied from src")
        copied += 1

    return copied


def move_flat_to_nested(brain_id: str, dry_run: bool = False) -> bool:
    """
    Move/merge a flat brain folder into its proper nested mouse folder.

    If nested already exists, merges training_data (flat may have more cubes).
    Then removes the flat stub.

    Example: 1_Brains/349_CNT_01_02_1p625x_z4/ → 1_Brains/349_CNT_01_02/349_CNT_01_02_1p625x_z4/

    Returns True if moved (or already nested), False on error.
    """
    flat_path = BRAINS_ROOT / brain_id
    if not flat_path.exists():
        print(f"  [SKIP] {brain_id}: flat folder doesn't exist (already cleaned up)")
        return True

    # Parse mouse name from brain_id: everything before the magnification suffix
    # Format: {BRAIN#}_{PROJECT}_{COHORT}_{SUBJECT}_{MAG}x_z{ZSTEP}
    parts = brain_id.split('_')
    # Find the magnification part (contains 'x')
    mag_idx = None
    for i, part in enumerate(parts):
        if 'x' in part and part.replace('p', '').replace('.', '').replace('x', '').isdigit():
            mag_idx = i
            break

    if mag_idx is None:
        print(f"  [ERROR] Could not parse mouse name from: {brain_id}")
        return False

    mouse_name = '_'.join(parts[:mag_idx])
    mouse_dir = BRAINS_ROOT / mouse_name
    nested_path = mouse_dir / brain_id

    if nested_path.exists():
        # Nested already exists - merge training_data from flat into nested
        flat_td = flat_path / "training_data"
        nested_td = nested_path / "training_data"

        if flat_td.exists():
            print(f"  [MERGE] Merging training_data from flat -> nested")
            copied = merge_training_data(flat_td, nested_td, dry_run=dry_run)
            if copied > 0:
                print(f"  [MERGE] {copied} file(s) {'would be ' if dry_run else ''}copied")
            else:
                print(f"  [MERGE] No new files to copy")

        # Remove the flat stub after merge
        if flat_path.resolve() != nested_path.resolve():
            if dry_run:
                print(f"  [DRY RUN] Would remove flat stub: {flat_path}")
            else:
                print(f"  [CLEAN] Removing flat stub: {flat_path}")
                shutil.rmtree(flat_path)
        return True

    # Nested doesn't exist - move flat there entirely
    if dry_run:
        print(f"  [DRY RUN] Would move: {flat_path} → {nested_path}")
        return True

    mouse_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [MOVE] {flat_path} → {nested_path}")
    try:
        shutil.move(str(flat_path), str(nested_path))
        return True
    except Exception as e:
        print(f"  [ERROR] Move failed: {e}")
        return False


def update_training_yml(brain_id: str, dry_run: bool = False) -> bool:
    """Update training.yml paths after folder move."""
    # Parse mouse name
    parts = brain_id.split('_')
    mag_idx = None
    for i, part in enumerate(parts):
        if 'x' in part and part.replace('p', '').replace('.', '').replace('x', '').isdigit():
            mag_idx = i
            break

    if mag_idx is None:
        return False

    mouse_name = '_'.join(parts[:mag_idx])
    yml_path = BRAINS_ROOT / mouse_name / brain_id / "training_data" / "training.yml"

    if not yml_path.exists():
        print(f"  [SKIP] No training.yml for {brain_id}")
        return True

    with open(yml_path, 'r') as f:
        content = f.read()

    old_flat = f"1_Brains\\{brain_id}\\training_data"
    new_nested = f"1_Brains\\{mouse_name}\\{brain_id}\\training_data"

    if old_flat not in content:
        print(f"  [OK] training.yml already has correct paths")
        return True

    new_content = content.replace(old_flat, new_nested)

    if dry_run:
        print(f"  [DRY RUN] Would update {yml_path}")
        return True

    with open(yml_path, 'w') as f:
        f.write(new_content)
    print(f"  [UPDATED] {yml_path}")
    return True


def update_tracker_csv(brain_ids: list, dry_run: bool = False) -> bool:
    """Update calibration_runs.csv to reflect moved folder paths."""
    csv_path = BRAINS_ROOT.parent / "2_Data_Summary" / "calibration_runs.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No calibration_runs.csv found")
        return True

    with open(csv_path, 'r') as f:
        content = f.read()

    changes = 0
    for brain_id in brain_ids:
        parts = brain_id.split('_')
        mag_idx = None
        for i, part in enumerate(parts):
            if 'x' in part and part.replace('p', '').replace('.', '').replace('x', '').isdigit():
                mag_idx = i
                break
        if mag_idx is None:
            continue

        mouse_name = '_'.join(parts[:mag_idx])

        # Fix: 1_Brains\349_CNT_01_02_1p625x_z4\ → 1_Brains\349_CNT_01_02\349_CNT_01_02_1p625x_z4\
        old = f"1_Brains\\{brain_id}\\"
        new = f"1_Brains\\{mouse_name}\\{brain_id}\\"
        if old in content:
            count = content.count(old)
            content = content.replace(old, new)
            changes += count
            print(f"  [FIX] {brain_id}: {count} path(s) updated")

    if changes == 0:
        print(f"  [OK] No flat paths found in tracker CSV")
        return True

    if dry_run:
        print(f"  [DRY RUN] Would update {changes} path(s) in {csv_path}")
        return True

    with open(csv_path, 'w') as f:
        f.write(content)
    print(f"  [UPDATED] {csv_path} ({changes} paths fixed)")
    return True


def run_post_commands(commands: list, training_dir: Path):
    """Run post-training commands with template substitution."""
    for cmd_template in commands:
        cmd = cmd_template.format(training_dir=training_dir)
        print(f"\n[{timestamp()}] Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, cwd=str(SCRIPT_DIR))
            if result.returncode == 0:
                print(f"[{timestamp()}] Command completed successfully")
            else:
                print(f"[{timestamp()}] Command exited with code {result.returncode}")
        except Exception as e:
            print(f"[{timestamp()}] Command error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor cellfinder training and run post-actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Monitor and auto-classify when done
    python util_monitor_training.py --training-dir path/to/output --epochs 100 \\
        --then "python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --model {training_dir}"

    # Monitor with folder cleanup
    python util_monitor_training.py --training-dir path/to/output --epochs 100 --cleanup-folders

    # Dry run (show what would happen)
    python util_monitor_training.py --training-dir path/to/output --epochs 100 --cleanup-folders --dry-run
        """
    )

    parser.add_argument('--training-dir', '-d', type=Path, required=True,
                        help='Training output directory to monitor')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Expected number of epochs (default: 100)')
    parser.add_argument('--poll-interval', type=int, default=300,
                        help='Seconds between progress checks (default: 300)')
    parser.add_argument('--then', nargs='+',
                        help='Command(s) to run after training completes. '
                             'Use {training_dir} as placeholder.')
    parser.add_argument('--cleanup-folders', action='store_true',
                        help='Move flat brain folders (349, 357) into nested structure after training')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    parser.add_argument('--skip-monitor', action='store_true',
                        help='Skip monitoring loop, run post-actions immediately (for testing)')

    args = parser.parse_args()

    training_dir = args.training_dir

    print("=" * 60)
    print("Training Monitor")
    print("=" * 60)
    print(f"Watching: {training_dir}")
    print(f"Expected epochs: {args.epochs}")
    print(f"Poll interval: {args.poll_interval}s")
    if args.then:
        print(f"Post-commands: {len(args.then)}")
        for cmd in args.then:
            print(f"  - {cmd}")
    if args.cleanup_folders:
        print("Folder cleanup: enabled")
    if args.dry_run:
        print("Mode: DRY RUN")
    if args.skip_monitor:
        print("Mode: SKIP MONITOR (running post-actions immediately)")
    print("=" * 60)

    if args.skip_monitor:
        # Check current progress for info, then skip to post-actions
        is_complete, current_epoch, total, loss, acc = check_training_progress(
            training_dir, args.epochs
        )
        print(f"\n[{timestamp()}] Current progress: {current_epoch}/{total} epochs")
        print(f"[{timestamp()}] Skipping monitor loop, proceeding to post-actions...")
    else:
        # Monitor loop
        last_epoch = -1
        while True:
            is_complete, current_epoch, total, loss, acc = check_training_progress(
                training_dir, args.epochs
            )

            if current_epoch != last_epoch:
                loss_str = f"{float(loss):.4f}" if loss and loss != 'nan' else "n/a"
                acc_str = f"{float(acc):.4f}" if acc and acc != 'nan' else "n/a"
                print(f"[{timestamp()}] Epoch {current_epoch}/{total} "
                      f"| val_loss={loss_str} | val_acc={acc_str}")
                last_epoch = current_epoch

            if is_complete:
                print(f"\n[{timestamp()}] Training COMPLETE! ({current_epoch} epochs)")
                break

            time.sleep(args.poll_interval)

    # === POST-TRAINING ACTIONS ===
    print(f"\n{'='*60}")
    print("POST-TRAINING ACTIONS")
    print(f"{'='*60}")

    # Folder cleanup if requested
    if args.cleanup_folders:
        print(f"\n--- Folder Cleanup ---")
        flat_brains = ["349_CNT_01_02_1p625x_z4", "357_CNT_02_08_1p625x_z4"]

        for brain_id in flat_brains:
            print(f"\nProcessing {brain_id}:")
            move_flat_to_nested(brain_id, dry_run=args.dry_run)
            update_training_yml(brain_id, dry_run=args.dry_run)

        print(f"\n--- Updating Tracker CSV ---")
        update_tracker_csv(flat_brains, dry_run=args.dry_run)

    # Run post-commands if specified
    if args.then:
        print(f"\n--- Post-Commands ---")
        if args.dry_run:
            for cmd in args.then:
                print(f"  [DRY RUN] Would run: {cmd.format(training_dir=training_dir)}")
        else:
            run_post_commands(args.then, training_dir)

    print(f"\n{'='*60}")
    print(f"[{timestamp()}] All post-training actions complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
