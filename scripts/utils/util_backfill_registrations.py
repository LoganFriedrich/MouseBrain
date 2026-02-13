#!/usr/bin/env python3
"""
util_backfill_registrations.py

Scans existing 3_Registered_Atlas folders and creates tracker entries
for registrations that happened before tracking was implemented.

Usage:
    python util_backfill_registrations.py
    python util_backfill_registrations.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from mousebrain.config import BRAINS_ROOT
from mousebrain.tracker import ExperimentTracker


def backfill_registrations(dry_run: bool = False):
    """
    Scan all brains and backfill registration tracking.

    Args:
        dry_run: If True, only report what would be done without making changes.

    Returns:
        Number of registrations backfilled.
    """
    tracker = ExperimentTracker()

    # Get already-tracked registrations
    existing = tracker.search(exp_type="registration")
    tracked_brains = {r.get('brain') for r in existing}

    print("=" * 60)
    print("Registration Backfill Utility")
    print("=" * 60)
    print(f"\nAlready tracked: {len(tracked_brains)} registration(s)")
    print(f"Scanning: {BRAINS_ROOT}")

    if dry_run:
        print("\n*** DRY RUN - No changes will be made ***\n")

    backfilled = 0
    skipped = 0

    for mouse_dir in BRAINS_ROOT.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue

        # Skip non-brain directories
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
            continue

        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            brain_name = pipeline_dir.name

            # Skip if already tracked
            if brain_name in tracked_brains:
                skipped += 1
                continue

            reg_folder = pipeline_dir / "3_Registered_Atlas"
            brainreg_json = reg_folder / "brainreg.json"

            # Skip if no registration exists
            if not brainreg_json.exists():
                continue

            # Skip archive/optimization folders
            if "_archive" in str(reg_folder) or "_crop_optimization" in str(reg_folder):
                continue

            # Found untracked registration - parse metadata
            try:
                with open(brainreg_json) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"  Warning: Could not parse {brainreg_json}: {e}")
                continue

            # Parse voxel sizes (stored as strings in a list)
            voxel_sizes = meta.get('voxel_sizes', ['4.0', '4.0', '4.0'])
            try:
                voxel_z = float(voxel_sizes[0])
                voxel_xy = float(voxel_sizes[1])
            except (IndexError, ValueError, TypeError):
                voxel_z = 4.0
                voxel_xy = 4.0

            # Parse registration steps
            try:
                affine_n_steps = int(meta.get('affine_n_steps', '6'))
                freeform_n_steps = int(meta.get('freeform_n_steps', '6'))
            except (ValueError, TypeError):
                affine_n_steps = 6
                freeform_n_steps = 6

            # Get timestamp from folder mtime (when brainreg completed)
            try:
                created_at = datetime.fromtimestamp(reg_folder.stat().st_mtime)
            except:
                created_at = datetime.now()

            # Check if approved
            approved = (reg_folder / ".registration_approved").exists()

            # Determine initial status
            initial_status = "approved" if approved else "completed"

            print(f"\nFound: {brain_name}")
            print(f"  Atlas: {meta.get('atlas', 'unknown')}")
            print(f"  Orientation: {meta.get('orientation', 'unknown')}")
            print(f"  Voxel sizes: Z={voxel_z}, XY={voxel_xy}")
            print(f"  Status: {initial_status}")
            print(f"  Estimated date: {created_at.strftime('%Y-%m-%d %H:%M')}")

            if dry_run:
                print(f"  [DRY RUN] Would backfill as: reg_...")
                backfilled += 1
                continue

            # Log to tracker
            exp_id = tracker.log_registration(
                brain=brain_name,
                atlas=meta.get('atlas', 'allen_mouse_10um'),
                orientation=meta.get('orientation', 'unknown'),
                voxel_z=voxel_z,
                voxel_xy=voxel_xy,
                affine_n_steps=affine_n_steps,
                freeform_n_steps=freeform_n_steps,
                output_path=str(reg_folder),
                status="completed",  # Start as completed, then approve if needed
                notes="Backfilled from existing registration",
            )

            # If approved, mark it
            if approved:
                tracker.approve_registration(exp_id)

            print(f"  Backfilled: {exp_id}")
            backfilled += 1

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Previously tracked: {len(tracked_brains)}")
    print(f"  Skipped (already tracked): {skipped}")
    print(f"  Backfilled: {backfilled}")
    print(f"{'=' * 60}")

    if dry_run and backfilled > 0:
        print(f"\nRun without --dry-run to actually backfill {backfilled} registration(s)")

    return backfilled


def main():
    parser = argparse.ArgumentParser(
        description='Backfill registration tracking for existing brains',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python util_backfill_registrations.py             # Actually backfill
    python util_backfill_registrations.py --dry-run   # Preview what would be done
        """
    )

    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Preview what would be done without making changes')

    args = parser.parse_args()

    backfill_registrations(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
