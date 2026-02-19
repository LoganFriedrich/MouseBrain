#!/usr/bin/env python3
"""
batch_classify_all.py

Run cellfinder classification (with pre-filter) on all brains that have
detections in 4_Cell_Candidates/. Uses the default model (most recent
best_model.h5).

Usage:
    python batch_classify_all.py              # Classify all brains
    python batch_classify_all.py --dry-run    # Show what would run
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = sys.executable
SCRIPT = Path(__file__).parent / "5_classify_cells.py"

# All brains to classify (in order of candidate count, smallest first)
BRAINS = [
    "349_CNT_01_02_1p625x_z4",
    "357_CNT_02_08_1p625x_z4",
    "367_CNT_03_07_1p625x_z4",
    "368_CNT_03_08_1p625x_z4",
]


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("Batch Classification -- All Brains")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model:   (default -- most recent best_model.h5)")
    print(f"Brains:  {len(BRAINS)}")
    print("=" * 60)

    results = {}

    for i, brain in enumerate(BRAINS, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(BRAINS)}] {brain}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

        cmd = [PYTHON, str(SCRIPT), "--brain", brain]

        if dry_run:
            print(f"  [DRY RUN] Would run: {' '.join(str(c) for c in cmd)}")
            results[brain] = "skipped (dry run)"
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, check=False)
            elapsed = (time.time() - t0) / 60
            if proc.returncode == 0:
                results[brain] = f"OK ({elapsed:.1f} min)"
                print(f"\n  [OK] {brain} completed in {elapsed:.1f} min")
            else:
                results[brain] = f"FAILED (exit {proc.returncode}, {elapsed:.1f} min)"
                print(f"\n  [FAIL] {brain} failed (exit {proc.returncode}) after {elapsed:.1f} min")
        except Exception as e:
            elapsed = (time.time() - t0) / 60
            results[brain] = f"ERROR: {e} ({elapsed:.1f} min)"
            print(f"\n  [ERROR] {brain}: {e}")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("BATCH COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    for brain, status in results.items():
        print(f"  {brain}: {status}")
    print()


if __name__ == "__main__":
    main()
