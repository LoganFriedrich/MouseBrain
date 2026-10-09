#!/usr/bin/env python3
"""
4_detect_cells.py

Script 4 in the BrainGlobe pipeline: Cell candidate detection.

Wraps cellfinder detection with presets and auto-logs to experiment tracker.

Normally run this AFTER Script 3 (register_to_atlas.py), on the cropped images
registration was computed from -- that is the path that leads to counts per
brain region. It can also run BEFORE registration, on the whole extracted
stack, which is useful when registration is waiting on something; see
WHICH IMAGES below for what that does and does not give you.

This runs cellfinder's cell candidate detection with preset parameter
combinations or custom values, and automatically logs everything.

================================================================================
HOW TO USE
================================================================================
Interactive mode (recommended for first runs):
    python 4_detect_cells.py

With presets:
    python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --preset sensitive
    python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --preset balanced
    python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --preset conservative

With the settings already proven for this kind of imaging (from the tracker):
    python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --routine

Custom parameters:
    python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --ball-xy 6 --ball-z 15

Before the brain has been cropped or registered:
    python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --source full --routine

================================================================================
WHICH IMAGES  (--source)
================================================================================
    auto     (default) the best available, in this order:
             2_Cropped_For_Registration_Manual, then
             2_Cropped_For_Registration, then
             1_Extracted_Full
    manual   insist on the hand-made crop
    cropped  insist on the automatic crop
    full     insist on the whole extracted stack

Detection itself never reads the atlas, so it works on any of these. What
differs is what you can do afterwards:

    A CROP (manual/cropped) is the images registration was computed on, so
    cells found there can be classified (Script 5) and counted per brain region
    (Script 6). This script refuses to run on a crop whose registration has not
    been approved by a person -- that gate is the whole reason detection waits
    for step 3 at all.

    THE WHOLE STACK (full) has no registration and cannot have one: the atlas
    is fitted to a crop. So there are no region counts, and the coordinates are
    in whole-stack space, which does not line up with a crop made later. A run
    here answers "do these settings find cells on this brain, and how many" --
    a trial run, worth doing while registration waits on a better scan, a
    manual crop, or an approval. Results go into a subfolder named after the
    source so Scripts 5 and 6 cannot pick them up by accident.

Every run of either kind is logged in the tracker, with the source recorded in
its notes, so two runs are never silently compared across spaces.

================================================================================
PRESETS
================================================================================
    sensitive    - Catches more cells, more false positives
                   ball_xy=4, ball_z=10, soma=12, threshold=8
                   
    balanced     - Good default starting point
                   ball_xy=6, ball_z=15, soma=16, threshold=10
                   
    conservative - Fewer false positives, may miss dim cells
                   ball_xy=8, ball_z=20, soma=20, threshold=12
                   
    large_cells  - For larger neurons (motor neurons, Purkinje, etc.)
                   ball_xy=10, ball_z=25, soma=25, threshold=10

================================================================================
REQUIREMENTS
================================================================================
    - cellfinder must be installed
    - the mousebrain package must be importable (it holds the tracker)
    - extracted images: a ch0 folder of .tif files in one of the folders listed
      under WHICH IMAGES above, which Script 2 produces
    - an APPROVED registration in 3_Registered_Atlas, if and only if detecting
      on a crop (see WHICH IMAGES)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.tracker import ExperimentTracker
except ImportError:
    print("ERROR: mousebrain.tracker not found!")
    print("Make sure mousebrain package is installed.")
    sys.exit(1)

# =============================================================================
# VERIFY CELLFINDER AVAILABILITY
# =============================================================================
import os

def check_cellfinder_api_available():
    """Check if cellfinder Python API is available (replaces deprecated CLI check)."""
    try:
        from cellfinder.core.detect.detect import main as detect_main
        from brainglobe_utils.IO.image.load import read_with_dask
        from brainglobe_utils.IO.cells import save_cells
        return True
    except ImportError as e:
        print("=" * 70)
        print("ERROR: cellfinder Python API not found!")
        print("=" * 70)
        print()
        print(f"Import error: {e}")
        print()
        print("Make sure cellfinder is installed: pip install cellfinder")
        print(f"Current Python: {sys.executable}")
        print()
        sys.exit(1)

# Check cellfinder on import (using Python API now, not deprecated CLI)
check_cellfinder_api_available()

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.1"

from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT, parse_brain_name

# Pipeline folders (must match other scripts)
FOLDER_FULL = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_CROPPED_MANUAL = "2_Cropped_For_Registration_Manual"
FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"

# Where detection reads its images from, best first. This is the same order the
# napari plugin uses, deliberately: a hand-made crop beats a machine-made one,
# and the uncropped stack is what is left when neither crop exists.
#
# WHY the uncropped stack is allowed at all: cell detection does not read the
# atlas or the registration. Cropping and registration exist to put cells into
# ATLAS space, which is what counting per region needs -- a later step. So a
# brain that has only been extracted can be detected on, and that is worth doing
# when registration is waiting on something (a better stitch, a manual crop, a
# person's approval). What it cannot do is produce region counts; see
# describe_input_choice() for what that costs and why.
INPUT_SOURCES = {
    "manual": FOLDER_CROPPED_MANUAL,
    "cropped": FOLDER_CROPPED,
    "full": FOLDER_FULL,
}
INPUT_PRIORITY = ("manual", "cropped", "full")

# The crop folders hold the images registration was computed on, so cells found
# in them can be mapped into the atlas. The uncropped stack cannot -- its voxel
# coordinates are offset from the crop's by however much was trimmed off.
SOURCES_IN_REGISTERED_SPACE = ("manual", "cropped")

# Detection presets
PRESETS = {
    'sensitive': {
        'description': 'Catches more cells, more false positives',
        'ball_xy_size': 4,
        'ball_z_size': 10,
        'soma_diameter': 12,
        'threshold': 8,
    },
    'balanced': {
        'description': 'Good default starting point',
        'ball_xy_size': 6,
        'ball_z_size': 15,
        'soma_diameter': 16,
        'threshold': 10,
    },
    'conservative': {
        'description': 'Fewer false positives, may miss dim cells',
        'ball_xy_size': 8,
        'ball_z_size': 20,
        'soma_diameter': 20,
        'threshold': 12,
    },
    'large_cells': {
        'description': 'For larger neurons (motor, Purkinje, etc.)',
        'ball_xy_size': 10,
        'ball_z_size': 25,
        'soma_diameter': 25,
        'threshold': 10,
    },
}

DEFAULT_N_FREE_CPUS = 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def resolve_input(pipeline_dir: Path, requested: str = "auto"):
    """Decide which folder of images detection should read.

    Args:
        pipeline_dir: the brain's pipeline folder.
        requested: "auto" to take the best available (see INPUT_PRIORITY), or
            one of INPUT_SOURCES to insist on exactly that folder.

    Returns:
        (source_name, folder, metadata) -- or (None, None, None) if there are no
        images to detect on. metadata is the folder's own metadata.json, which is
        where the voxel sizes and the signal/background channel roles come from.

    WHY this is a function and not two lines inline: three places need the same
    answer (the named-brain path, the interactive path, and listing what can be
    processed), and when they each decided for themselves they disagreed -- the
    listing offered brains the run then refused.
    """
    pipeline_dir = Path(pipeline_dir)

    if requested != "auto" and requested not in INPUT_SOURCES:
        raise ValueError("unknown input source %r (expected auto or one of %s)"
                         % (requested, ", ".join(INPUT_SOURCES)))

    order = INPUT_PRIORITY if requested == "auto" else (requested,)
    for source in order:
        folder = pipeline_dir / INPUT_SOURCES[source]
        # ch0 specifically, not just the folder: an empty folder is the normal
        # state of a crop nobody has made yet, and it must not look like data.
        if not (folder / "ch0").is_dir():
            continue
        if not any((folder / "ch0").glob("*.tif")):
            continue

        metadata_path = folder / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        return source, folder, metadata

    return None, None, None


def describe_input_choice(source: str) -> str:
    """What detecting on this folder means for the rest of the pipeline.

    Printed before a run so the person starting it knows what they will and will
    not be able to do with the result. Not a warning for its own sake: detecting
    on the uncropped stack is a legitimate and useful thing to do, it just does
    not lead anywhere near region counts without redoing it.
    """
    if source in SOURCES_IN_REGISTERED_SPACE:
        return ("Detecting on %s -- the same images registration uses, so these "
                "cells can be classified and counted per brain region as usual."
                % INPUT_SOURCES[source])

    return (
        "Detecting on %s -- the WHOLE extracted stack, uncropped and "
        "unregistered.\n"
        "\n"
        "What this gives you:\n"
        "  * a real cell count for these detection settings on this brain\n"
        "  * candidate coordinates you can load in napari and look at\n"
        "\n"
        "What it does NOT give you, and why:\n"
        "  * no counts per brain region. Those need the atlas, and the atlas is\n"
        "    fitted during registration (step 3), which has not happened.\n"
        "  * these coordinates are in whole-stack space. A later registration is\n"
        "    computed on a CROPPED stack, so its coordinates start from a\n"
        "    different corner. Cells found now cannot simply be reused against\n"
        "    that atlas -- detection is rerun on the cropped images instead.\n"
        "\n"
        "So this is a trial run: it answers whether the settings find cells and\n"
        "roughly how many, on this brain, today. Treat the number as provisional\n"
        "until the brain has been cropped and registered.\n"
        "\n"
        "Results go in a clearly named subfolder of %s so that steps 5 and 6\n"
        "cannot pick them up by accident -- they only ever read the top level."
        % (INPUT_SOURCES[source], FOLDER_DETECTION)
    )


def find_pipeline(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT,
                  source: str = "auto"):
    """
    Find pipeline folder for a brain.

    Args:
        brain_name: Either just the pipeline name (101_PROJ_01_02_2p5x_z5)
                   or mouse/pipeline format
        source: which images to detect on -- see resolve_input().

    Returns:
        (pipeline_folder, mouse_folder, source_name, input_folder, metadata)
        or five Nones if the brain or its images could not be found.

    This no longer refuses a brain that has not been registered. It used to, and
    that was the wrong gate in the wrong place: detection does not read the
    registration at all, so a missing atlas is a reason not to COUNT REGIONS, not
    a reason not to detect. The approval gate that does belong -- do not spend
    hours detecting on images whose registration a person has not checked -- is
    still enforced in main(), and only for the crop folders it applies to.
    """
    root = Path(root)

    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue

        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            # Match by pipeline name or full path
            full_name = f"{mouse_dir.name}/{pipeline_dir.name}"
            if brain_name in [pipeline_dir.name, full_name]:
                source_name, input_folder, metadata = resolve_input(pipeline_dir, source)
                if source_name is None:
                    print(f"ERROR: no images to detect on for {brain_name}")
                    if source == "auto":
                        print(f"  Looked for a ch0 folder of .tif files in: "
                              f"{', '.join(INPUT_SOURCES[s] for s in INPUT_PRIORITY)}")
                        print("  Run Script 2 (2_extract_and_analyze.py) first.")
                    else:
                        print(f"  --source {source} means {INPUT_SOURCES[source]}, "
                              f"and there are no .tif files in its ch0 folder.")
                    return None, None, None, None, None

                return pipeline_dir, mouse_dir, source_name, input_folder, metadata

    return None, None, None, None, None


def list_available_brains(root: Path = DEFAULT_BRAINGLOBE_ROOT, source: str = "auto"):
    """List every brain that has images detection could run on.

    Registration is REPORTED, not required. A brain that is extracted but not
    yet registered is a perfectly valid thing to detect on (see resolve_input);
    hiding it from this list meant the only way to do that was to not use this
    script, so people didn't, and then the run was never logged in the tracker.
    """
    root = Path(root)
    brains = []

    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
            continue

        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            source_name, input_folder, metadata = resolve_input(pipeline_dir, source)
            if source_name is None:
                continue

            reg_folder = pipeline_dir / FOLDER_REGISTRATION
            det_folder = pipeline_dir / FOLDER_DETECTION

            brains.append({
                'name': f"{mouse_dir.name}/{pipeline_dir.name}",
                'pipeline': pipeline_dir,
                'mouse': mouse_dir,
                'source': source_name,
                'input_folder': input_folder,
                'metadata': metadata,
                'registered': (reg_folder / "brainreg.json").exists(),
                'approved': (reg_folder / ".registration_approved").exists(),
                'detected': det_folder.exists() and len(list(det_folder.glob("*.xml"))) > 0,
            })

    return brains


def count_cells_in_xml(xml_path: Path) -> int:
    """Count cells in a cellfinder XML file."""
    if not xml_path.exists():
        return 0
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        return content.count('<Marker>')
    except:
        return 0


def run_cellfinder_detect(
    signal_path: Path,
    background_path: Path,
    output_path: Path,
    voxel_sizes: tuple,
    params: dict,
    n_free_cpus: int = 2,
) -> tuple:
    """
    Run cellfinder detection using Python API (not deprecated CLI).

    Returns:
        (success, duration, cells_found)

    IMPORTANT - WINDOWS MULTIPROCESSING LIMIT:
        Windows WaitForMultipleObjects allows max 63 handles. If
        (cpu_count - n_free_cpus) > 63, cellfinder crashes with:
            "ValueError: need at most 63 handles, got a sequence of length N"
        On machines with many CPUs (e.g. 104), n_free_cpus MUST be set high
        enough to keep the worker pool under 63. This is enforced below.
    """
    # Import cellfinder Python API
    from cellfinder.core.detect.detect import main as detect_main
    from brainglobe_utils.IO.image.load import read_with_dask
    from brainglobe_utils.IO.cells import save_cells
    import multiprocessing

    # ENFORCE WINDOWS 63-HANDLE LIMIT (see docstring)
    max_workers = 60  # safely under the 63 limit
    total_cpus = multiprocessing.cpu_count()
    if total_cpus - n_free_cpus > max_workers:
        n_free_cpus = total_cpus - max_workers
        print(f"    [NOTE] Adjusted n_free_cpus to {n_free_cpus} (Windows 63-handle limit, {total_cpus} CPUs)")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{timestamp()}] Running cellfinder detection...")
    print(f"    Signal: {signal_path}")
    print(f"    Background: {background_path}")
    print(f"    Output: {output_path}")
    print(f"    Parameters:")
    print(f"        ball_xy_size: {params['ball_xy_size']}")
    print(f"        ball_z_size: {params['ball_z_size']}")
    print(f"        soma_diameter: {params['soma_diameter']}")
    print(f"        threshold: {params['threshold']}")
    print()

    start_time = time.time()

    try:
        # Load signal array with dask (memory efficient)
        print(f"    Loading signal data from {signal_path}...")
        signal_array = read_with_dask(str(signal_path))
        print(f"    Signal array shape: {signal_array.shape}")

        # Run detection using Python API
        print(f"    Running detection...")
        cells = detect_main(
            signal_array,
            start_plane=0,
            end_plane=-1,  # All planes
            voxel_sizes=(float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2])),
            soma_diameter=float(params['soma_diameter']),
            max_cluster_size=100000,
            ball_xy_size=float(params['ball_xy_size']),
            ball_z_size=float(params['ball_z_size']),
            ball_overlap_fraction=0.6,
            soma_spread_factor=1.4,
            n_free_cpus=n_free_cpus,
            log_sigma_size=0.2,
            n_sds_above_mean_thresh=float(params['threshold']),  # This is the threshold parameter
        )

        duration = time.time() - start_time
        cells_found = len(cells)

        # Save results to XML
        cells_xml = output_path / "detected_cells.xml"
        save_cells(cells, str(cells_xml))
        print(f"    Saved {cells_found} cells to {cells_xml}")

        return True, duration, cells_found

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, time.time() - start_time, 0


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_select_brain(brains):
    """Interactive brain selection."""
    print("\n" + "=" * 60)
    print("AVAILABLE BRAINS")
    print("=" * 60)
    
    ready = []
    already_done = []

    def label(brain):
        """Say which images this brain would be detected on, and in what state.

        Without this the list was just names, and two brains that would be
        processed completely differently -- one against a registered crop, one
        against a raw stack -- looked identical on screen.
        """
        where = INPUT_SOURCES[brain['source']]
        if brain['source'] not in SOURCES_IN_REGISTERED_SPACE:
            return f"{brain['name']}  [{where} -- not cropped, no region counts]"
        if not brain['registered']:
            return f"{brain['name']}  [{where} -- not registered yet]"
        if not brain['approved']:
            return f"{brain['name']}  [{where} -- registration NOT approved]"
        return f"{brain['name']}  [{where}]"

    for i, brain in enumerate(brains):
        if brain['detected']:
            already_done.append((i, label(brain)))
        else:
            ready.append((i, label(brain)))

    if ready:
        print("\n[READY FOR DETECTION]")
        for idx, name in ready:
            print(f"  {idx + 1}. {name}")

    if already_done:
        print("\n[ALREADY DETECTED]")
        for idx, name in already_done:
            print(f"  {idx + 1}. {name} (has results)")

    print("\n" + "-" * 60)
    print("Enter number to select, or 'q' to quit")
    print("-" * 60)
    
    while True:
        response = input("\nSelection: ").strip()
        
        if response.lower() == 'q':
            return None
        
        try:
            idx = int(response) - 1
            if 0 <= idx < len(brains):
                return brains[idx]
            else:
                print(f"Invalid number. Enter 1-{len(brains)}")
        except ValueError:
            print("Enter a number or 'q'")


def interactive_select_preset():
    """Interactive preset selection."""
    print("\n" + "=" * 60)
    print("DETECTION PRESETS")
    print("=" * 60)
    
    for i, (name, preset) in enumerate(PRESETS.items()):
        print(f"\n  {i + 1}. {name}")
        print(f"     {preset['description']}")
        print(f"     ball_xy={preset['ball_xy_size']}, ball_z={preset['ball_z_size']}, "
              f"soma={preset['soma_diameter']}, threshold={preset['threshold']}")
    
    print(f"\n  {len(PRESETS) + 1}. custom (enter your own parameters)")
    
    print("\n" + "-" * 60)
    
    while True:
        response = input("\nSelect preset (1-5): ").strip()
        
        try:
            idx = int(response)
            if 1 <= idx <= len(PRESETS):
                preset_name = list(PRESETS.keys())[idx - 1]
                return preset_name, PRESETS[preset_name].copy()
            elif idx == len(PRESETS) + 1:
                # Custom
                params = {}
                params['ball_xy_size'] = int(input("  ball_xy_size (default 6): ").strip() or "6")
                params['ball_z_size'] = int(input("  ball_z_size (default 15): ").strip() or "15")
                params['soma_diameter'] = int(input("  soma_diameter (default 16): ").strip() or "16")
                params['threshold'] = int(input("  threshold (default 10): ").strip() or "10")
                return "custom", params
            else:
                print(f"Enter 1-{len(PRESETS) + 1}")
        except ValueError:
            print("Enter a number")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cellfinder detection with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
    sensitive     - ball_xy=4, ball_z=10, soma=12, threshold=8
    balanced      - ball_xy=6, ball_z=15, soma=16, threshold=10 (default)
    conservative  - ball_xy=8, ball_z=20, soma=20, threshold=12
    large_cells   - ball_xy=10, ball_z=25, soma=25, threshold=10

Examples:
    python 4_detect_cells.py                                    # Interactive mode
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset balanced
    python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --ball-xy 5 --ball-z 12
        """
    )
    
    parser.add_argument('--brain', '-b', help='Brain/pipeline to process')
    parser.add_argument('--preset', '-p', choices=list(PRESETS.keys()),
                        help='Parameter preset')
    
    # Custom parameters
    parser.add_argument('--ball-xy', type=int, help='Ball filter XY size')
    parser.add_argument('--ball-z', type=int, help='Ball filter Z size')
    parser.add_argument('--soma-diameter', type=int, help='Expected soma diameter')
    parser.add_argument('--threshold', type=int, help='Detection threshold')
    
    parser.add_argument('--source', choices=['auto'] + list(INPUT_SOURCES),
                        default='auto',
                        help='Which images to detect on. auto (default) takes the '
                             'best available: a manual crop, else an automatic '
                             'crop, else the whole extracted stack. "full" insists '
                             'on the uncropped stack, which works before '
                             'registration but cannot produce region counts.')
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS)
    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    parser.add_argument('--routine', action='store_true',
                        help='Routine processing mode: auto-use paradigm-best settings if available')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrainGlobe Cell Detection")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Select brain
    if args.brain:
        (pipeline_folder, mouse_folder, input_source, input_folder,
         metadata) = find_pipeline(args.brain, args.root, args.source)
        if not pipeline_folder:
            print(f"ERROR: nothing to detect for: {args.brain}")
            sys.exit(1)
        brain_name = f"{mouse_folder.name}/{pipeline_folder.name}"
    else:
        # Interactive selection
        brains = list_available_brains(args.root, args.source)
        if not brains:
            print("\nNo brains with extracted images found!")
            print("Run Script 2 (2_extract_and_analyze.py) first.")
            sys.exit(1)

        brain_info = interactive_select_brain(brains)
        if not brain_info:
            print("Cancelled.")
            return

        pipeline_folder = brain_info['pipeline']
        mouse_folder = brain_info['mouse']
        brain_name = brain_info['name']
        input_source = brain_info['source']
        input_folder = brain_info['input_folder']
        metadata = brain_info['metadata']

    # Say what these images are before anything expensive starts.
    print("\n" + "-" * 60)
    print(describe_input_choice(input_source))
    print("-" * 60)

    # The approval gate applies to the crop folders and only to them. Its purpose
    # is to stop hours of detection being spent on images whose registration
    # nobody has looked at -- which is a statement about a registration that
    # EXISTS. There is no registration to approve when detecting on the
    # uncropped stack, so demanding approval there would be demanding the
    # impossible, and the whole point of this path is that registration is still
    # waiting on something.
    if input_source in SOURCES_IN_REGISTERED_SPACE:
        reg_folder = pipeline_folder / FOLDER_REGISTRATION
        approval_file = reg_folder / ".registration_approved"
        if not approval_file.exists():
            print("\n" + "=" * 60)
            print("STOPPING: REGISTRATION NOT YET APPROVED")
            print("=" * 60)
            print("\nThese are the images registration was computed on, so cell")
            print("detection here is the expensive step that follows it -- and it")
            print("should not run until a person has confirmed the registration is")
            print("actually good.")
            print("\nSteps:")
            print("  1. Review QC images:")
            print(f"     {reg_folder / 'QC_registration_detailed.png'}")
            print("  2. If registration looks good, approve it:")
            print(f"     python util_approve_registration.py --brain {pipeline_folder.name}")
            print("\nOr, if registration is not ready and you want to try detection")
            print("settings on the raw stack in the meantime:")
            print(f"     python 4_detect_cells.py --brain {pipeline_folder.name} --source full")
            print("=" * 60)
            sys.exit(1)

    # Check for paradigm-best settings
    tracker = ExperimentTracker()
    parsed = parse_brain_name(pipeline_folder.name)
    imaging_paradigm = parsed.get('imaging_params', '')
    paradigm_settings = None

    if imaging_paradigm:
        paradigm_settings = tracker.get_paradigm_detection_settings(imaging_paradigm)
        if paradigm_settings:
            print(f"\n[Paradigm Best Found] Settings for '{imaging_paradigm}':")
            print(f"  Source: {paradigm_settings.get('source_brain', 'unknown')}")
            print(f"  ball_xy={paradigm_settings['ball_xy']:.0f}, "
                  f"ball_z={paradigm_settings['ball_z']:.0f}, "
                  f"soma={paradigm_settings['soma_diameter']:.0f}, "
                  f"threshold={paradigm_settings['threshold']}")

    # Select parameters
    if args.routine and paradigm_settings:
        # Routine mode: auto-use paradigm-best settings
        print("\n[Routine Mode] Using paradigm-best detection settings.")
        preset_name = "paradigm_best"
        params = {
            'ball_xy_size': int(paradigm_settings['ball_xy']),
            'ball_z_size': int(paradigm_settings['ball_z']),
            'soma_diameter': int(paradigm_settings['soma_diameter']),
            'threshold': int(paradigm_settings['threshold']),
        }
    elif args.preset:
        preset_name = args.preset
        params = PRESETS[preset_name].copy()
    elif any([args.ball_xy, args.ball_z, args.soma_diameter, args.threshold]):
        # Custom from command line
        preset_name = "custom"
        params = {
            'ball_xy_size': args.ball_xy or 6,
            'ball_z_size': args.ball_z or 15,
            'soma_diameter': args.soma_diameter or 16,
            'threshold': args.threshold or 10,
        }
    elif not args.brain:
        # Interactive preset selection - offer paradigm-best first if available
        if paradigm_settings:
            print("\n" + "=" * 60)
            print("PARADIGM-BEST SETTINGS AVAILABLE")
            print("=" * 60)
            print(f"\nUse proven settings from '{imaging_paradigm}' paradigm? (y/n)")
            use_paradigm = input("Selection [y]: ").strip().lower()
            if use_paradigm in ('', 'y', 'yes'):
                preset_name = "paradigm_best"
                params = {
                    'ball_xy_size': int(paradigm_settings['ball_xy']),
                    'ball_z_size': int(paradigm_settings['ball_z']),
                    'soma_diameter': int(paradigm_settings['soma_diameter']),
                    'threshold': int(paradigm_settings['threshold']),
                }
            else:
                preset_name, params = interactive_select_preset()
        else:
            preset_name, params = interactive_select_preset()
    else:
        # Non-interactive with brain specified - use paradigm-best if available, else balanced
        if paradigm_settings:
            print("\n[Auto] Using paradigm-best settings (use --preset to override)")
            preset_name = "paradigm_best"
            params = {
                'ball_xy_size': int(paradigm_settings['ball_xy']),
                'ball_z_size': int(paradigm_settings['ball_z']),
                'soma_diameter': int(paradigm_settings['soma_diameter']),
                'threshold': int(paradigm_settings['threshold']),
            }
        else:
            preset_name = "balanced"
            params = PRESETS["balanced"].copy()
    
    # Override individual params if specified
    if args.ball_xy:
        params['ball_xy_size'] = args.ball_xy
    if args.ball_z:
        params['ball_z_size'] = args.ball_z
    if args.soma_diameter:
        params['soma_diameter'] = args.soma_diameter
    if args.threshold:
        params['threshold'] = args.threshold
    
    # Get paths
    det_folder = pipeline_folder / FOLDER_DETECTION
    if input_source not in SOURCES_IN_REGISTERED_SPACE:
        # Keep results off the top level of 4_Cell_Candidates. Steps 5 and 6 glob
        # the top level only, so a run whose coordinates are NOT in the space the
        # atlas will be fitted to stays invisible to them. Without this, a trial
        # run on the uncropped stack would sit exactly where the real one belongs
        # and later get classified and counted against the wrong coordinates --
        # silently, and with a perfectly plausible number coming out.
        det_folder = det_folder / ("from_" + INPUT_SOURCES[input_source])

    # Determine signal and background channels
    channels = metadata.get('channels', {})
    signal_ch = channels.get('signal_channel', 0)
    background_ch = channels.get('background_channel', 1)

    signal_path = input_folder / f"ch{signal_ch}"
    background_path = input_folder / f"ch{background_ch}"

    # Get voxel sizes
    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (
        voxel.get('z', 4),
        voxel.get('y', 4),
        voxel.get('x', 4),
    )
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Brain: {brain_name}")
        print(f"Preset: {preset_name}")
        print(f"Parameters: {params}")
        print(f"Signal: {signal_path}")
        print(f"Background: {background_path}")
        print(f"Voxel sizes: {voxel_sizes}")
        print(f"Output: {det_folder}")
        return

    # Log detection run (tracker already initialized for paradigm check)
    exp_id = tracker.log_detection(
        brain=brain_name,
        preset=preset_name,
        ball_xy=params['ball_xy_size'],
        ball_z=params['ball_z_size'],
        soma_diameter=params['soma_diameter'],
        threshold=params['threshold'],
        voxel_z=voxel_sizes[0],
        voxel_xy=voxel_sizes[1],
        input_path=str(input_folder),
        output_path=str(det_folder),
        # The source goes in the notes as well as the path, because the notes are
        # what a person reads in the tracker table. A run on the uncropped stack
        # and a run on the registered crop are not comparable as counts, and the
        # record has to say which one this was without anyone decoding a path.
        notes=" | ".join(filter(None, [
            args.notes,
            "images: %s (%s)" % (
                INPUT_SOURCES[input_source],
                "registered space"
                if input_source in SOURCES_IN_REGISTERED_SPACE
                else "whole-stack space, pre-registration, no region counts"),
        ])),
        status="started",
        script_version=SCRIPT_VERSION,
    )
    
    print(f"\n{'='*60}")
    print(f"Detection Run: {exp_id}")
    print(f"Brain: {brain_name}")
    print(f"Preset: {preset_name}")
    print(f"{'='*60}")
    
    # Run detection
    success, duration, cells_found = run_cellfinder_detect(
        signal_path=signal_path,
        background_path=background_path,
        output_path=det_folder,
        voxel_sizes=voxel_sizes,
        params=params,
        n_free_cpus=args.n_free_cpus,
    )
    
    # Update tracker
    tracker.update_status(
        exp_id,
        status="completed" if success else "failed",
        duration_seconds=round(duration, 1),
        det_cells_found=cells_found,
    )
    
    print(f"\n{'='*60}")
    if success:
        print(f"COMPLETED in {duration/60:.1f} minutes")
        print(f"Cells detected: {cells_found}")
    else:
        print(f"FAILED after {duration/60:.1f} minutes")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")
    
    # Interactive rating
    if success:
        try:
            rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add a note (or Enter to skip): ").strip()
                tracker.rate_experiment(exp_id, int(rating), note if note else None)
                print("Rating saved!")
        except (EOFError, KeyboardInterrupt):
            pass

        # Next step guidance. Which next step it IS depends on what was detected
        # on: classification and counting only make sense for cells that can be
        # placed in the atlas.
        print("\n" + "=" * 60)
        print("WHAT TO DO NEXT")
        print("=" * 60)
        if input_source in SOURCES_IN_REGISTERED_SPACE:
            print("\n1. Run cell classification:")
            print(f"   python 5_classify_cells.py --brain {brain_name.split('/')[-1]}")
            print("\n" + "-" * 60)
            print("OR just run: python RUN_PIPELINE.py")
            print("   (it will guide you through everything)")
        else:
            print(f"\nCandidates were written to:")
            print(f"   {det_folder}")
            print("\nThis was a trial run on the uncropped stack, so there is no")
            print("next pipeline step from here -- classification and region counts")
            print("both need the atlas. What you can do now:")
            print("\n1. Look at the candidates on the images, in napari:")
            print("   mousebrain")
            print("   then Plugins -> BrainTools -> 3D: 2. Setup & Tuning,")
            print("   pick this brain and load it. The plugin reads the uncropped")
            print("   stack when there is no crop, so the layers will line up.")
            print("\n2. Judge the count and the settings, and re-run this script")
            print("   with different parameters if they need changing. Every run is")
            print("   logged in the tracker, so the comparison is kept for you.")
            print("\n3. When the brain is finally cropped and registered, run")
            print("   detection AGAIN on the crop -- these coordinates do not")
            print("   transfer. Then steps 5 and 6 follow as normal.")
        print("=" * 60)


if __name__ == '__main__':
    main()
