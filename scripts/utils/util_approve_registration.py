#!/usr/bin/env python3
"""
util_approve_registration.py

Mark registration as approved after manual QC review.

This creates a `.registration_approved` file that subsequent steps check for.
The file names the registration it approves (mousebrain.registration_approval):
registering the brain again needs a new approval, and an approval written
before approvals recorded their registration does not count.

Usage:
    python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4
    python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4 --view

The script will:
1. Show you the QC images
2. Ask if registration looks good
3. Create approval marker if you confirm
"""

import argparse
import sys
from pathlib import Path
from mousebrain.registration_approval import (APPROVED, NEEDS_REGISTRATION, NOT_REGISTERED,
                                              REJECTED, RUNNING, RegistrationChanged,
                                              approval_state, approve as record_approval,
                                              current_registration_id, reject)

# =============================================================================
# CONFIGURATION
# =============================================================================

from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
FOLDER_REGISTERED = "3_Registered_Atlas"
APPROVAL_MARKER = ".registration_approved"

# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find a brain's pipeline folder.

    An exact folder name wins. Otherwise the name must match exactly ONE brain
    folder: a part of a name that fits several brains (two imaging setups of one
    mouse share its number) must not approve whichever is listed first.
    """
    from mousebrain.config import find_brain_folder
    found, partial = find_brain_folder(brain_name, root)
    if found is None and len(partial) > 1:
        print(f"Error: '{brain_name}' matches more than one brain; give the full name:")
        for p in partial:
            print(f"  {p.name}")
    return found


def check_approval_status(pipeline_folder: Path):
    """(approved, explanation) for the registration that is in the folder now."""
    state, text = approval_state(pipeline_folder / FOLDER_REGISTERED)
    return state == APPROVED, text


def ask_yes_no(question: str):
    """True for yes/y, False for no/n; anything else asks again. None when
    there is nobody to answer (end of input) -- nothing is recorded then.

    WHY IT ASKS AGAIN: "no" RECORDS A REJECTION under the person's name. A
    bare Enter or a typo must not count as "no", or a person who only pressed
    Enter would reject the registration and send the brain back to "Register"."""
    while True:
        try:
            response = input(question).strip().lower()
        except EOFError:
            return None
        if response in ('yes', 'y'):
            return True
        if response in ('no', 'n'):
            return False
        print("  Please answer yes or no.")


def approve_registration(pipeline_folder: Path, overrule_rejection: bool = False,
                         expected_id=None):
    """Record that the person running this script approved the registration.
    ``overrule_rejection``: the person was told the registration had been
    rejected and said they approve it anyway. ``expected_id``: the
    registration they were shown (registration_approval.approve refuses
    another one)."""
    record_approval(pipeline_folder / FOLDER_REGISTERED,
                    note="approved with util_approve_registration.py"
                    + (" (overruling a rejection)" if overrule_rejection else ""),
                    allow_rejected=overrule_rejection, expected_id=expected_id)
    print(f"[OK] Registration approved and marked for: {pipeline_folder.name}")
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
  python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4

  # Just view QC images without approving
  python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4 --view

  # Check approval status
  python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4 --status
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
    if approval_state(reg_folder)[0] == NOT_REGISTERED:
        print(f"\nError: No registration found for this brain")
        print(f"  Run registration first: python 3_register_to_atlas.py --brain {args.brain}")
        sys.exit(1)

    # The registration this run shows the person (status, QC pictures). The
    # yes/no below is about THAT one: if a batch registers the brain again
    # while they look, the answer is refused instead of signing (or turning
    # down) a registration nobody saw. Taken before the state is read.
    shown_id = current_registration_id(reg_folder)

    # A registration that is there but cannot be approved (it did not finish
    # ...): say why and stop, instead of showing pictures and asking a
    # question whose "yes" would be refused.
    state, why = approval_state(reg_folder)
    overrule = False
    if state == REJECTED and not (args.status or args.view):
        # A person turned this registration down. Approving it now needs an
        # explicit "yes, overrule": nobody approves a rejected registration
        # by answering the ordinary question.
        print(f"\n[!] This registration was {why}")
        if not ask_yes_no("\nApprove it anyway, overruling the rejection? (yes/no): "):
            print("Rejection kept. Nothing changed.")
            sys.exit(1)
        overrule = True
    elif state == RUNNING and not args.view:
        print(f"\n[!] Nothing to approve yet: {why}")
        sys.exit(1)
    elif state in NEEDS_REGISTRATION and not args.view:
        print(f"\n[!] Nothing to approve: {why}")
        print(f"  Register it again: python 3_register_to_atlas.py --batch --brain {pipeline_folder.name}")
        sys.exit(1)

    # Check current status
    approved, approval_data = check_approval_status(pipeline_folder)

    if args.status:
        if approved:
            print(f"\n[OK] Registration is APPROVED")
            print(f"  {approval_data}")
        else:
            print(f"\n[!] Registration NOT APPROVED: {approval_data}")
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

    if overrule:
        pass  # already asked above
    elif approved:
        print(f"\n[!] This registration is already approved:")
        print(f"  {approval_data}")
        if not ask_yes_no("\nRe-approve? (yes/no): "):
            print("Keeping existing approval")
            sys.exit(0)
    else:
        answer = ask_yes_no("\nDoes the registration look good? (yes/no): ")
        if answer is None:
            print("\nNo answer: nothing recorded (neither approved nor rejected).")
            sys.exit(1)
        if not answer:
            # "no" is a rejection, and it is recorded: every reader shows the
            # brain back at "Register" and nobody approves it by accident.
            try:
                reject(reg_folder, note="rejected with util_approve_registration.py",
                       expected_id=shown_id)
            except RegistrationChanged as e:
                print(f"\n[FAIL] Nothing rejected: {e}")
                print("  Run this script again to see the new registration.")
                sys.exit(1)
            print("\n[FAIL] Registration REJECTED (recorded)")
            print("  Adjust the crop or parameters, then register the brain again")
            sys.exit(1)

    # Create approval marker
    try:
        approve_registration(pipeline_folder, overrule_rejection=overrule,
                             expected_id=shown_id)
    except RegistrationChanged as e:
        print(f"\n[FAIL] Nothing approved: {e}")
        print("  Run this script again to see the new registration.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("[OK] Registration approved!")
    print(f"{'='*60}")
    print("\nYou can now proceed to cell detection:")
    print(f"  python 4_detect_cells.py --brain {args.brain}")
    print()


if __name__ == '__main__':
    main()
