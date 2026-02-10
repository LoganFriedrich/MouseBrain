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
from config import BRAINS_ROOT


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

    for mouse_dir in BRAINS_ROOT.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
            continue

        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

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
    """Get detailed status of a brain."""
    p = brain_path

    status = {
        'has_ims': len(list((p / "0_Raw_IMS").glob("*.ims"))) > 0 if (p / "0_Raw_IMS").exists() else False,
        'extracted': (p / "1_Extracted_Full" / "metadata.json").exists() if (p / "1_Extracted_Full").exists() else False,
        'cropped': (p / "2_Cropped_For_Registration" / "metadata.json").exists() if (p / "2_Cropped_For_Registration").exists() else False,
        'registered': (p / "3_Registered_Atlas" / "brainreg.json").exists() if (p / "3_Registered_Atlas").exists() else False,
        'approved': (p / "3_Registered_Atlas" / ".registration_approved").exists() if (p / "3_Registered_Atlas").exists() else False,
        'detected': len(list((p / "4_Cell_Candidates").glob("*.xml"))) > 0 if (p / "4_Cell_Candidates").exists() else False,
        'prefiltered': (p / "4_Cell_Candidates" / "prefilter_report.json").exists() if (p / "4_Cell_Candidates").exists() else False,
        'classified': (p / "5_Classified_Cells" / "cells.xml").exists() if (p / "5_Classified_Cells").exists() else False,
        'counted': len(list((p / "6_Region_Analysis").glob("*.csv"))) > 0 if (p / "6_Region_Analysis").exists() else False,
    }

    # Determine current step
    if not status['extracted']:
        status['step'] = 'extract'
        status['step_name'] = 'Extract images from IMS file'
        status['step_num'] = 1
    elif not status['cropped']:
        status['step'] = 'crop'
        status['step_name'] = 'Crop to remove spinal cord'
        status['step_num'] = 2
    elif not status['registered']:
        status['step'] = 'register'
        status['step_name'] = 'Register to atlas'
        status['step_num'] = 3
    elif not status['approved']:
        status['step'] = 'approve'
        status['step_name'] = 'Review and approve registration'
        status['step_num'] = 4
    elif not status['detected']:
        status['step'] = 'detect'
        status['step_name'] = 'Detect cells'
        status['step_num'] = 5
    elif not status['prefiltered']:
        status['step'] = 'prefilter'
        status['step_name'] = 'Pre-filter candidates (atlas)'
        status['step_num'] = 6
    elif not status['classified']:
        status['step'] = 'classify'
        status['step_name'] = 'Classify cells'
        status['step_num'] = 7
    elif not status['counted']:
        status['step'] = 'count'
        status['step_name'] = 'Count cells by region'
        status['step_num'] = 8
    else:
        status['step'] = 'done'
        status['step_name'] = 'Complete!'
        status['step_num'] = 9

    return status


def run_script(script_name, args=None):
    """Run a script and wait for it to finish."""
    script_path = SCRIPT_DIR / script_name
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

    for label, done in checks:
        mark = "[X]" if done else "[ ]"
        print(f"  {mark} {label}")

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
        run_script('3_register_to_atlas.py')

    elif status['step'] == 'approve':
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
            "No, I need to re-register",
            "Let me look at it first",
        ])

        if choice == 0:
            run_script('util_approve_registration.py', ['--brain', brain_name, '--yes'])
        elif choice == 1:
            print("\nRe-running registration...")
            run_script('3_register_to_atlas.py')

    elif status['step'] == 'detect':
        # Ask about detection settings
        print("\nCell detection finds candidate cells in your images.")

        choice = ask("Which detection preset?", [
            "Balanced (recommended for most cases)",
            "Sensitive (catches more, but more false positives)",
            "Conservative (fewer false positives, may miss some)",
            "Large cells (for motor neurons, Purkinje cells)",
        ])

        presets = ['balanced', 'sensitive', 'conservative', 'large_cells']
        if choice is not None:
            run_script('4_detect_cells.py', ['--brain', brain_name, '--preset', presets[choice]])

    elif status['step'] == 'prefilter':
        print("\nPre-filtering candidates using atlas registration...")
        print("This removes candidates outside the brain (meningeal/surface).")

        choice = ask("Pre-filter options:", [
            "Standard (remove outside-brain only)",
            "With suspicious region flagging",
            "Skip pre-filtering (go straight to classify)",
        ])

        if choice == 0:
            run_script('util_atlas_prefilter.py', ['--brain', brain_name])
        elif choice == 1:
            run_script('util_atlas_prefilter.py', ['--brain', brain_name, '--flag-suspicious'])
        # choice == 2 means skip, do nothing

    elif status['step'] == 'classify':
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
            print("  1. Create a folder for your mouse (e.g., '349_CNT_01_02')")
            print("  2. Inside that, create a pipeline folder (e.g., '349_CNT_01_02_1p625x_z4')")
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
