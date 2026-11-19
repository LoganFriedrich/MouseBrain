#!/usr/bin/env python3
"""
RUN_PIPELINE.py - Interactive Pipeline Guide

Just run this script. It will ask you questions and guide you through everything.

    conda activate brainglobe-env
    python RUN_PIPELINE.py

That's it.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
from mousebrain.config import BRAINS_ROOT, find_script, iter_brains
from mousebrain.pipeline_status import brain_status


def clear():
    """Clear screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def ask(question, options=None, allow_quit=True):
    """Ask user a question. Returns their choice."""
    print(f"\n{question}")

    if options:
        print()
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        if allow_quit:
            print(f"  [q] Quit")
        print()

        while True:
            choice = input(">>> ").strip().lower()
            if choice == 'q' and allow_quit:
                return None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return idx
            except ValueError:
                pass
            print("  Try again - enter a number")
    else:
        return input(">>> ").strip()


def find_brains():
    """Find all brain folders."""
    brains = []

    if not BRAINS_ROOT.exists():
        return brains

    # Which folders are brains is ONE rule, config.iter_brains.
    for mouse_dir, pipeline_dir in iter_brains(BRAINS_ROOT, report_skipped=True):
        # Check if this looks like a pipeline
        has_raw = (pipeline_dir / "0_Raw_IMS").exists()
        has_extracted = (pipeline_dir / "1_Extracted_Full").exists()
        has_cropped = (pipeline_dir / "2_Cropped_For_Registration").exists()

        if has_raw or has_extracted or has_cropped:
            brains.append({
                'name': pipeline_dir.name,
                'full_name': f"{mouse_dir.name}/{pipeline_dir.name}",
                'path': pipeline_dir,
                'mouse': mouse_dir.name,
            })

    return brains


def get_brain_status(brain_path):
    """Get detailed status of a brain.

    The rules live in mousebrain.pipeline_status, shared with the napari
    dashboard and the overnight batch, so every reader puts a brain at the same
    step -- and at the step the next script will actually take up.
    """
    return brain_status(brain_path, include_prefilter=True)


def run_script(script_name, args=None):
    """Run a script and wait for it to finish.

    find_script: the steps live in scripts/pipeline, the util_*.py helpers in
    scripts/utils. Looking only beside this file made the pre-filter action
    fail with "can't open file".
    """
    script_path = find_script(script_name)
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    subprocess.run(cmd)

    print(f"\n{'='*60}")
    input("Press Enter to continue...")


def show_brain_status(brain):
    """Show status of a specific brain and offer actions."""
    clear()
    status = get_brain_status(brain['path'])

    print("=" * 60)
    print(f"BRAIN: {brain['full_name']}")
    print("=" * 60)

    # Progress bar
    steps = ['Extract', 'Crop', 'Register', 'Approve', 'Detect', 'Pre-filter', 'Classify', 'Count']
    progress = ""
    for i, step in enumerate(steps):
        if i < status['step_num'] - 1:
            progress += f" [{step}]"
        elif i == status['step_num'] - 1 and status['step'] != 'done':
            progress += f" -->{step}<--"
        else:
            progress += f"  {step}  "

    print(f"\nProgress: Step {status['step_num']} of 8")
    print(f"Current: {status['step_name']}")

    print("\n" + "-" * 60)

    # Show checkmarks
    checks = [
        ("IMS file present", status['has_ims']),
        ("Images extracted", status['extracted']),
        ("Cropped for registration", status['cropped']),
        ("Registered to atlas", status['registered']),
        ("Registration approved", status['approved']),
        ("Cells detected", status['detected']),
        ("Candidates pre-filtered", status['prefiltered']),
        ("Cells classified", status['classified']),
        ("Regions counted", status['counted']),
    ]

    # A stale output exists but was made from something replaced since (see
    # mousebrain.pipeline_status): shown as [!] with the reason, and the step
    # is offered again.
    stale_of = {"Cells detected": "detect", "Cells classified": "classify",
                "Regions counted": "count"}
    for label, done in checks:
        why = status['stale'].get(stale_of.get(label, ""))
        mark = "[!]" if why else ("[X]" if done else "[ ]")
        line = f"  {mark} {label}" + (f" -- STALE: {why}" if why else "")
        # A registration that is in the folder but does not count (it did not
        # finish ...): say why, so "not registered" is not a mystery.
        if label == "Registered to atlas" and status.get('register_problem'):
            line = f"  [!] {label} -- {status['register_problem']}"
        print(line)
    if status['registered'] and not status['approved']:
        print(f"\n  Registration: {status['approval_text']}")

    print("-" * 60)

    if status['step'] == 'done':
        print("\nThis brain is fully processed!")
        choice = ask("What would you like to do?", [
            "View results",
            "Reprocess a step",
            "Go back to brain list",
        ])
        if choice == 2:
            return 'back'
        # TODO: implement view results and reprocess
        return 'back'

    # Offer next action
    print(f"\nNext step: {status['step_name']}")

    choice = ask("What would you like to do?", [
        f"Run this step ({status['step_name']})",
        "Skip to a different step",
        "Go back to brain list",
    ])

    if choice is None or choice == 2:
        return 'back'

    if choice == 1:
        # Let them pick a step
        available = []
        if status['extracted']:
            available.append(('crop', 'Manual crop (napari)'))
        if status['cropped']:
            available.append(('register', 'Register to atlas'))
        if status['registered']:
            available.append(('approve', 'Approve registration'))
        if status['approved']:
            available.append(('detect', 'Detect cells'))
        if status['detected']:
            available.append(('prefilter', 'Pre-filter candidates'))
        if status['prefiltered'] or status['detected']:
            available.append(('classify', 'Classify cells'))
        if status['classified']:
            available.append(('count', 'Count by region'))

        if not available:
            print("\nNo steps available to skip to.")
            input("Press Enter...")
            return 'continue'

        step_choice = ask("Which step?", [s[1] for s in available])
        if step_choice is None:
            return 'continue'
        status['step'] = available[step_choice][0]

    # Run the appropriate step
    brain_name = brain['name']

    if status['step'] == 'extract':
        run_script('2_extract_and_analyze.py')

    elif status['step'] == 'crop':
        print("\nOpening napari for manual cropping...")
        print("Use the Manual Crop plugin from the Plugins menu.")
        subprocess.run(['napari'])

    elif status['step'] == 'register':
        from mousebrain.registration_approval import REJECTED, RUNNING
        if status.get('approval') == RUNNING:
            print(f"\n{status['approval_text']}")
            input("Press Enter to continue...")
        elif status.get('approval') == REJECTED:
            # The batches never register a rejected brain again; a person
            # decides. --reprocess registers this one brain again.
            print(f"\nThis registration was {status['approval_text']}")
            choice = ask("Register it again now?", [
                "Yes, register it again (the rejected one goes into _archive)",
                "No, not yet",
            ])
            if choice == 0:
                run_script('3_register_to_atlas.py',
                           ['--batch', '--brain', brain_name, '--reprocess'])
        else:
            run_script('3_register_to_atlas.py')

    elif status['step'] == 'approve':
        # The registration the person is about to look at. The answer below
        # is about THAT one: approve()/reject() refuse it when a batch has
        # registered the brain again while the person was looking
        # (registration_approval.RegistrationChanged), instead of signing a
        # registration nobody saw. Taken before the QC picture is opened.
        from mousebrain.registration_approval import (RegistrationChanged,
                                                      current_registration_id)
        shown_id = current_registration_id(brain['path'] / "3_Registered_Atlas")

        # Show where the QC images are
        qc_path = brain['path'] / "3_Registered_Atlas" / "QC_registration_detailed.png"
        print(f"\nFirst, look at the QC image to check if registration is good:")
        print(f"  {qc_path}")

        if qc_path.exists():
            # Try to open it
            if os.name == 'nt':
                os.startfile(str(qc_path))

        choice = ask("\nDoes the registration look good?", [
            "Yes, approve it",
            "No, reject it (recorded)",
            "Let me look at it first",
        ])

        if choice == 0:
            # The person said yes just now, so the approval is written here,
            # naming the registration they looked at. (This used to run the
            # approval script with a --yes it does not accept, so it never
            # approved anything.)
            from mousebrain.registration_approval import approve
            try:
                approve(brain['path'] / "3_Registered_Atlas",
                        note="approved in RUN_PIPELINE after viewing the QC picture",
                        expected_id=shown_id)
                print("\n[OK] Registration approved.")
            except RegistrationChanged as e:
                print(f"\n[FAIL] Nothing approved: {e}")
                print("Choose this brain again to see the new registration.")
            except ValueError as e:
                print(f"\n[FAIL] {e}")
            input("Press Enter to continue...")
        elif choice == 1:
            # "No" is a rejection, recorded so that every reader shows the
            # brain back at "Register" -- it used to be forgotten as soon as
            # this menu closed, and the brain was offered for approval again.
            from mousebrain.registration_approval import reject
            try:
                reject(brain['path'] / "3_Registered_Atlas",
                       note="rejected in RUN_PIPELINE after viewing the QC picture",
                       expected_id=shown_id)
                print("\n[OK] Registration rejected (recorded). The brain is back at 'Register'.")
            except RegistrationChanged as e:
                print(f"\n[FAIL] Nothing rejected: {e}")
                print("Choose this brain again to see the new registration.")
                input("Press Enter to continue...")
                return 'continue'
            except ValueError as e:
                print(f"\n[FAIL] {e}")
            # Not registered again here: on the same images and settings it
            # would take hours to give the same fit. Once something is changed
            # the brain's "Register" step offers it (a re-crop makes the
            # batches register it again by themselves).
            print("Change what made it bad first (crop, orientation, voxel size),")
            print("then choose this brain's 'Register' step to register it again.")
            input("Press Enter to continue...")

    elif status['step'] == 'detect':
        # No choice to make: step 4 uses the settled method for this brain's
        # imaging setup (the tracker run marked Paradigm Best), and says how to
        # settle one if there is none.
        print("\nCell detection finds candidate cells in your images,")
        print("using the settled detection method for this kind of imaging.")
        run_script('4_detect_cells.py', ['--brain', brain_name])

    elif status['step'] == 'prefilter':
        print("\nPre-filtering candidates using atlas registration...")
        print("This removes candidates outside the brain (meningeal/surface).")
        run_script('util_atlas_prefilter.py', ['--brain', brain_name])

    elif status['step'] == 'classify':
        # No choice to make: step 5 classifies with the settled method of this
        # imaging setup (the classification run marked Paradigm Best) and its
        # own pre-filter. A pre-filter saved or tuned by hand is tuning, and
        # step 5 refuses it (it used to be offered here).
        print("\nClassification checks each candidate cell, using the settled")
        print("classification method for this kind of imaging.")
        run_script('5_classify_cells.py', ['--brain', brain_name])

    elif status['step'] == 'count':
        run_script('6_count_regions.py', ['--brain', brain_name])

    return 'continue'


def main_menu():
    """Main menu - shows brains and lets user pick one."""
    while True:
        clear()

        print("=" * 60)
        print("  BRAINGLOBE CELL DETECTION PIPELINE")
        print("=" * 60)
        print()

        brains = find_brains()

        if not brains:
            print("No brains found!")
            print(f"\nLooking in: {BRAINS_ROOT}")
            print("\nTo get started:")
            print("  1. Create a folder for your mouse (e.g., '001_CNT_01_01')")
            print("  2. Inside that, create a pipeline folder (e.g., '001_CNT_01_01_1p625x_z4')")
            print("  3. Put your .ims file in a '0_Raw_IMS' subfolder")
            print("  4. Run this script again")
            print()
            input("Press Enter to exit...")
            return

        # Show brains with their status
        print("YOUR BRAINS:")
        print("-" * 60)

        for i, brain in enumerate(brains, 1):
            status = get_brain_status(brain['path'])

            # Simple progress indicator
            progress = status['step_num'] - 1
            bar = "[" + "=" * progress + ">" + " " * (8 - progress) + "]" if progress < 8 else "[========]"

            if status['step'] == 'done':
                state = "COMPLETE"
            else:
                state = f"Step {status['step_num']}: {status['step_name']}"

            print(f"  [{i}] {brain['name']}")
            print(f"      {bar} {state}")
            print()

        print("-" * 60)
        print("  [m] Open napari (manual crop tool)")
        print("  [h] View experiment history")
        print("  [q] Quit")
        print("-" * 60)

        choice = input("\nPick a brain (number) or option: ").strip().lower()

        if choice == 'q':
            print("\nGoodbye!")
            return

        if choice == 'm':
            print("\nOpening napari...")
            subprocess.run(['napari'])
            continue

        if choice == 'h':
            run_script('util_experiments.py')
            continue

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(brains):
                while True:
                    result = show_brain_status(brains[idx])
                    if result == 'back':
                        break
                    # Refresh status and continue
        except ValueError:
            print("Enter a number or letter option")
            input("Press Enter...")


def main():
    """Entry point."""
    # Quick dependency check
    try:
        import numpy
        import tifffile
    except ImportError as e:
        print("=" * 60)
        print("ERROR: Missing dependencies!")
        print("=" * 60)
        print(f"\n{e}")
        print("\nMake sure you activated the environment:")
        print("  conda activate brainglobe-env")
        print()
        input("Press Enter to exit...")
        sys.exit(1)

    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == '__main__':
    main()
