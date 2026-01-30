#!/usr/bin/env python3
"""
util_approve_registration.py

Mark registration as approved after manual QC review.

This creates a `.registration_approved` file that subsequent steps check for.

Usage:
    python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4
    python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4 --view

The script will:
1. Show you the QC images
2. Ask if registration looks good
3. Create approval marker if you confirm
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
FOLDER_REGISTERED = "3_Registered_Atlas"
APPROVAL_MARKER = ".registration_approved"

# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find a brain's pipeline folder."""
    root = Path(root)

    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name == pipeline_folder.name or brain_name in pipeline_folder.name:
                return pipeline_folder

    return None


def check_approval_status(pipeline_folder: Path):
    """Check if registration has been approved."""
    reg_folder = pipeline_folder / FOLDER_REGISTERED
    approval_file = reg_folder / APPROVAL_MARKER

    if approval_file.exists():
        with open(approval_file, 'r') as f:
            data = f.read().strip()
        return True, data
    return False, None


def approve_registration(pipeline_folder: Path):
    """Create approval marker file."""
    reg_folder = pipeline_folder / FOLDER_REGISTERED
    approval_file = reg_folder / APPROVAL_MARKER

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(approval_file, 'w') as f:
        f.write(f"Approved: {timestamp}\n")

    print(f"✓ Registration approved and marked for: {pipeline_folder.name}")
    return True


def view_qc_images(pipeline_folder: Path):
    """Open QC images for review."""
    import subprocess
    import platform

    reg_folder = pipeline_folder / FOLDER_REGISTERED
    qc_detailed = reg_folder / "QC_registration_detailed.png"
    qc_overview = reg_folder / "QC_registration_overview.png"

    images_to_view = []
    if qc_detailed.exists():
        images_to_view.append(qc_detailed)
    if qc_overview.exists():
        images_to_view.append(qc_overview)

    if not images_to_view:
        print("Warning: No QC images found!")
        print(f"  Looked in: {reg_folder}")
        return False

    print(f"\nOpening QC images:")
    for img in images_to_view:
        print(f"  - {img.name}")

        # Open image with default viewer
        if platform.system() == 'Windows':
            subprocess.run(['start', str(img)], shell=True)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(img)])
        else:  # Linux
            subprocess.run(['xdg-open', str(img)])

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Approve registration after QC review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review QC and approve
  python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4

  # Just view QC images without approving
  python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4 --view

  # Check approval status
  python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4 --status
        """
    )

    parser.add_argument('--brain', '-b', required=True,
                       help='Brain/pipeline to approve')
    parser.add_argument('--view', action='store_true',
                       help='Just view QC images without approving')
    parser.add_argument('--status', action='store_true',
                       help='Check approval status')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT,
                       help='Root folder for brains')

    args = parser.parse_args()

    # Find brain
    pipeline_folder = find_brain(args.brain, args.root)
    if not pipeline_folder:
        print(f"Error: Brain not found: {args.brain}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Registration Approval")
    print(f"{'='*60}")
    print(f"Brain: {pipeline_folder.name}")
    print(f"Path: {pipeline_folder}")

    # Check if registration exists
    reg_folder = pipeline_folder / FOLDER_REGISTERED
    if not reg_folder.exists():
        print(f"\nError: No registration found for this brain")
        print(f"  Run registration first: python 3_register_to_atlas.py --brain {args.brain}")
        sys.exit(1)

    # Check current status
    approved, approval_data = check_approval_status(pipeline_folder)

    if args.status:
        if approved:
            print(f"\n✓ Registration is APPROVED")
            print(f"  {approval_data}")
        else:
            print(f"\n⚠ Registration NOT YET APPROVED")
            print(f"  Review QC images and approve if registration looks good")
        sys.exit(0)

    # View QC images
    print(f"\n{'='*60}")
    print("QC Image Review")
    print(f"{'='*60}")

    if not view_qc_images(pipeline_folder):
        print("\nCould not open QC images")
        sys.exit(1)

    if args.view:
        print("\n(View-only mode, not creating approval marker)")
        sys.exit(0)

    # Ask for approval
    print(f"\n{'='*60}")
    print("Review the QC images that just opened.")
    print("Check that:")
    print("  - Brain structures align with atlas")
    print("  - Region boundaries match anatomical features")
    print("  - Registration looks good across dorsal-ventral axis")
    print(f"{'='*60}")

    if approved:
        print(f"\n⚠ This registration is already approved:")
        print(f"  {approval_data}")
        response = input("\nRe-approve? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Keeping existing approval")
            sys.exit(0)
    else:
        response = input("\nDoes the registration look good? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\n✗ Registration NOT approved")
            print("  Re-run registration if needed, or adjust parameters")
            sys.exit(1)

    # Create approval marker
    approve_registration(pipeline_folder)

    print(f"\n{'='*60}")
    print("✓ Registration approved!")
    print(f"{'='*60}")
    print("\nYou can now proceed to cell detection:")
    print(f"  python 4_detect_cells.py --brain {args.brain}")
    print()


if __name__ == '__main__':
    main()
