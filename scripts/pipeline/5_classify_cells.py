#!/usr/bin/env python3
"""
5_classify_cells.py

Script 5 in the BrainGlobe pipeline: Cell classification.

Writes a brain's PRODUCTION classification (5_Classified_Cells/cells.xml,
which step 6 counts) with the settled method of its imaging setup: the
classification run a person marked Paradigm Best (tracker
get_paradigm_classification) -- its model file, cube and batch size, and, for
a selective classification, its tracing type. With no Paradigm Best it stops;
it never falls back to "the newest model". Every run is logged to the tracker
and recorded beside its files (promoted_from.json).

Method "selective" (the usual one): pre-filter the candidates, map each to
its atlas region, TRUST candidates in deep regions, run the classifier only on
the rest, and install trusted + classifier-approved as cells.xml.
Method "full": pre-filter, then run the classifier on every candidate.

Run this AFTER Script 4 (4_detect_cells.py).

================================================================================
HOW TO USE
================================================================================
    python 5_classify_cells.py --brain 001_CNT_01_01_1p625x_z4
    python 5_classify_cells.py --brain 001_CNT_01_01_1p625x_z4 --dry-run   # show the method
    python 5_classify_cells.py  # Interactive mode

================================================================================
REQUIREMENTS
================================================================================
    - cellfinder must be installed
    - Detection output in 4_Cell_Candidates folder (from Script 4)
    - experiment_tracker.py in same directory or PYTHONPATH
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.tracker import ExperimentTracker
except ImportError:
    print("ERROR: mousebrain.tracker not found!")
    print("Make sure mousebrain package is installed.")
    sys.exit(1)

# Try to import cellfinder classification components
try:
    from brainglobe_utils.cells.cells import Cell
    from brainglobe_utils.IO.cells import get_cells, save_cells
    from cellfinder.core.classify.cube_generator import CubeGeneratorFromFile
    from cellfinder.core.classify.classify import main as classify_main
    CELLFINDER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import cellfinder components: {e}")
    CELLFINDER_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT, iter_brains, explain_brain_not_found
from mousebrain.config import parse_brain_name
from mousebrain.config import registered_candidate_folders, registered_input
from mousebrain.classification_runs import (discard_staged, follow_files_into_archive,
                                            follow_selective_rows, install_staged,
                                            staging_folder, sweep_stopped_runs)

FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_DETECTION = "4_Cell_Candidates"
FOLDER_CLASSIFICATION = "5_Classified_Cells"

DEFAULT_CUBE_SIZE = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_FREE_CPUS = 2

# Fixed parts of the classifier input, the same for every model this tool
# trains (cellfinder's standard cube depth and its ResNet-50 network).
CUBE_DEPTH = 20
NETWORK_DEPTH = "50"

# The settled pre-filter: these exact settings, written out rather than taken
# from prefilter_candidates' defaults, so changing a default there can never
# change production quietly. They are the settings the settled selective
# classification was made with (the pre-filter's defaults when the settled method was made).
PREFILTER_SETTINGS = {
    "method": "atlas_erosion",
    "surface_depth_um": 100.0,
    "withdrawal_depth_um": 100.0,
    "include_ventricle_walls": True,
    "include_fiber_tracts": True,
    "filter_depth_um": 100.0,
    "dark_percentile": 5.0,
    "extreme_oob_um": 500.0,
    "atlas_name": "allen_mouse_10um",
    "tracing_type": "descending",
}
PREFILTERED_NAME = "interior_candidates.xml"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def find_pipeline(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find pipeline folder for a brain."""
    root = Path(root)

    # A brains folder that is missing or cannot be read is not "brain not
    # found": stop and name the folder, so an unconnected data drive is not
    # mistaken for a brain that does not exist.
    from mousebrain.config import brains_root_problem
    problem = brains_root_problem(root)
    if problem:
        sys.exit("ERROR: " + problem)

    # Which folders are brains is ONE rule, config.iter_brains.
    for mouse_dir, pipeline_dir in iter_brains(root):
        full_name = f"{mouse_dir.name}/{pipeline_dir.name}"
        if brain_name in [pipeline_dir.name, full_name]:
            return pipeline_dir, mouse_dir

    # Say why, when the brain is there but in a folder the rule passes over.
    why = explain_brain_not_found(brain_name, root)
    if why:
        print(why)
    return None, None


def find_detection_output(pipeline_folder: Path):
    """Find detection output (cell candidates) in the space the atlas was fitted to.

    The top level of 4_Cell_Candidates first; then ``from_<folder>`` when the
    atlas was fitted to that folder's images (config.registered_candidate_folders
    says why). A run on images in any other space is never returned.

    WHY IT CALLS pipeline_status.find_candidates: "the candidates" is ONE rule.
    The batch QC, the pre-filter and Setup & Tuning read the file that function
    names, and this step must classify that same file. A second copy of the
    search here took the first .xml in directory-listing order, which a network
    share does not guarantee to be by name -- so with two non-standard .xml files
    in a folder, this step could classify a different file from the one the
    others had checked.
    """
    from mousebrain.pipeline_status import find_candidates
    return find_candidates(pipeline_folder)


def count_cells_in_xml(xml_path: Path) -> int:
    """Count cells in XML file."""
    if not xml_path.exists():
        return 0
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        return content.count('<Marker>')
    except:
        return 0


def list_available_brains(root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """List all brains that have detection output."""
    root = Path(root)
    brains = []
    
    # Which folders are brains is ONE rule, config.iter_brains.
    for mouse_dir, pipeline_dir in iter_brains(root, report_skipped=True):
        det_folder = pipeline_dir / FOLDER_DETECTION
        class_folder = pipeline_dir / FOLDER_CLASSIFICATION
        
        candidates = find_detection_output(pipeline_dir)
        has_detection = candidates is not None
        has_classification = class_folder.exists() and len(list(class_folder.glob("cells.xml"))) > 0
        
        if has_detection:
            n_candidates = count_cells_in_xml(candidates) if candidates else 0
            brains.append({
                'name': f"{mouse_dir.name}/{pipeline_dir.name}",
                'pipeline': pipeline_dir,
                'mouse': mouse_dir,
                'candidates': candidates,
                'n_candidates': n_candidates,
                'classified': has_classification,
            })

    return brains


def run_cellfinder_classify(
    signal_path: Path,
    candidates_path: Path,
    output_path: Path,
    model_path: Path,
    voxel_sizes: tuple,
    cube_size: int = DEFAULT_CUBE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_free_cpus: int = DEFAULT_N_FREE_CPUS,
    background_path: Path = None,
) -> tuple:
    """
    Run cellfinder classification using Python API.

    Returns:
        (success, duration, cells_found, rejected)
    """
    if not CELLFINDER_AVAILABLE:
        print("ERROR: cellfinder classification components not available!")
        print("Make sure cellfinder is installed: pip install cellfinder")
        return False, 0, 0, 0

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{timestamp()}] Running cellfinder classification...")
    print(f"    Signal: {signal_path}")
    print(f"    Background: {background_path}")
    print(f"    Candidates: {candidates_path}")
    print(f"    Model: {model_path}")
    print(f"    Output: {output_path}")
    print()

    start_time = time.time()

    try:
        # Load candidate cells from XML
        print(f"[{timestamp()}] Loading candidate cells...")
        cells = get_cells(str(candidates_path))
        print(f"    Loaded {len(cells)} candidates")

        if len(cells) == 0:
            print("ERROR: No candidates found in XML file")
            return False, time.time() - start_time, 0, 0

        # Convert voxel_sizes from (z, y, x) microns
        voxel_sizes_zyx = (float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2]))

        print(f"[{timestamp()}] Loading image data...")
        print(f"    Voxel sizes (z,y,x): {voxel_sizes_zyx}")

        # Load images as dask arrays for memory efficiency
        from brainglobe_utils.IO.image.load import read_with_dask

        signal_array = read_with_dask(str(signal_path))
        print(f"    Signal shape: {signal_array.shape}")

        if background_path and background_path.exists():
            background_array = read_with_dask(str(background_path))
            print(f"    Background shape: {background_array.shape}")
        else:
            background_array = None
            print("    Background: None")

        print(f"[{timestamp()}] Setting up classification...")
        print(f"    Batch size: {batch_size}")
        print(f"    Cube size: {cube_size}")
        print(f"    n_free_cpus: {n_free_cpus}")

        # Run classification using cellfinder's main function
        from cellfinder.core.classify.classify import main as classify_main

        cube_depth = CUBE_DEPTH

        print(f"[{timestamp()}] Running classification network...")

        # Determine model type
        # .keras and .h5 checkpoints from training (save_weights_only=False) are
        # full models -> load via trained_model (tf.keras.models.load_model)
        # Weights-only files -> load via model_weights (build_model + load_weights)
        if model_path.suffix in ('.keras', '.h5'):
            trained_model_arg = model_path
            model_weights_arg = None
            print(f"    Model format: {model_path.suffix} (full model)")
        else:
            trained_model_arg = None
            model_weights_arg = model_path
            print(f"    Model format: {model_path.suffix} (weights)")

        classified_cells = classify_main(
            points=cells,
            signal_array=signal_array,
            background_array=background_array,
            n_free_cpus=n_free_cpus,
            voxel_sizes=voxel_sizes_zyx,
            network_voxel_sizes=(5.0, 1.0, 1.0),  # Standard network voxel sizes
            batch_size=batch_size,
            cube_height=cube_size,
            cube_width=cube_size,
            cube_depth=cube_depth,
            trained_model=trained_model_arg,
            model_weights=model_weights_arg,
            network_depth=NETWORK_DEPTH,  # ResNet50
        )

        duration = time.time() - start_time

        # Separate cells and rejected
        cells_list = [c for c in classified_cells if c.type == Cell.CELL]
        rejected_list = [c for c in classified_cells if c.type == Cell.NO_CELL]

        cells_found = len(cells_list)
        rejected_count = len(rejected_list)

        print(f"[{timestamp()}] Classification complete!")
        print(f"    Cells: {cells_found}")
        print(f"    Rejected: {rejected_count}")

        # Save results
        cells_output = output_path / "cells.xml"
        rejected_output = output_path / "rejected.xml"

        save_cells(cells_list, str(cells_output))
        save_cells(rejected_list, str(rejected_output))

        print(f"    Saved to: {output_path}")

        return True, duration, cells_found, rejected_count

    except Exception as e:
        import traceback
        print(f"ERROR during classification: {e}")
        traceback.print_exc()
        return False, time.time() - start_time, 0, 0


def interactive_select_brain(brains):
    """Interactive brain selection."""
    print("\n" + "=" * 60)
    print("BRAINS WITH DETECTION OUTPUT")
    print("=" * 60)
    
    ready = []
    already_done = []
    
    for i, brain in enumerate(brains):
        if brain['classified']:
            already_done.append((i, brain['name'], brain['n_candidates']))
        else:
            ready.append((i, brain['name'], brain['n_candidates']))
    
    if ready:
        print("\n[READY FOR CLASSIFICATION]")
        for idx, name, n in ready:
            print(f"  {idx + 1}. {name} ({n} candidates)")
    
    if already_done:
        print("\n[ALREADY CLASSIFIED]")
        for idx, name, n in already_done:
            print(f"  {idx + 1}. {name} ({n} candidates)")
    
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


def settled_method_or_stop(tracker, imaging_paradigm: str, args) -> dict:
    """The classification method step 5 runs, or stop (exit 1) saying why.

    The method is the run a person marked Paradigm Best for the brain's
    imaging setup (tracker.get_paradigm_classification): its model file, cube
    and batch size, and -- for a selective classification -- its tracing type.
    Nothing else is ever used.

    WHY: step 5 writes THE classification of a brain, which step 6 counts.
    Taking "the newest model file" in the models folder, or letting options swap
    in any model, skip the pre-filter or classify a tuned pre-filter, would put
    all of those straight into production. Two models on
    the same brain gave counts several-fold apart, so a count could not be
    traced to a method anyone had settled on. Trying something else is tuning:
    Setup & Tuning writes such runs to optimization_runs/<brain>/, never here.
    """
    def stop(*lines):
        for line in lines:
            print(line)
        print("Nothing was classified, and the production classification was not touched.")
        sys.exit(1)

    if not imaging_paradigm:
        stop("ERROR: the imaging setup (magnification + z-step) cannot be read from the "
             "brain's name, so its settled classification method cannot be looked up.",
             "Brain names end in <mag>x_z<step>, e.g. 1p625x_z4.")
    method = tracker.get_paradigm_classification(imaging_paradigm)
    if method is None:
        stop(f"ERROR: no classification run is marked Paradigm Best for imaging setup "
             f"'{imaging_paradigm}'.",
             "Step 5 runs only the settled method: the classification (or selective",
             "classification) run a person marked Paradigm Best for this imaging setup.",
             "To settle one: python scripts/utils/util_mark_paradigm_classifier.py --run <exp_id>",
             "(see docs/pipeline/README.md, 'The settled classification method').")
    if method['missing']:
        stop(f"ERROR: the Paradigm Best classification run {method['source_exp_id']} does not "
             f"record: {', '.join(method['missing'])}.",
             "A method is not finished with values the run never used. Record them on the",
             "run with util_mark_paradigm_classifier.py (--cube-size / --batch-size), or mark",
             "a run that records them.")
    model = Path(method['model_path'])
    if not model.is_file():
        stop(f"ERROR: the settled model file is missing: {model}",
             f"(from Paradigm Best run {method['source_exp_id']}). It is never replaced by",
             "another model file. Put the file back, or settle another run.")
    refused = []
    if args.model is not None:
        try:
            same = Path(args.model).is_file() and os.path.samefile(str(args.model), str(model))
        except OSError:
            same = False
        if not same:
            refused.append(f"--model {args.model} (the settled model is {model})")
    if args.no_prefilter:
        refused.append("--no-prefilter (the settled method pre-filters)")
    if args.candidates:
        refused.append("--candidates (a saved or tuned pre-filter is not the settled pre-filter)")
    if args.tracing_type and args.tracing_type != (method['tracing_type'] or args.tracing_type):
        refused.append(f"--tracing-type {args.tracing_type} (settled: {method['tracing_type']})")
    if args.cube_size is not None and args.cube_size != method['cube_size']:
        refused.append(f"--cube-size {args.cube_size} (settled: {method['cube_size']})")
    if args.batch_size is not None and args.batch_size != method['batch_size']:
        refused.append(f"--batch-size {args.batch_size} (settled: {method['batch_size']})")
    if refused:
        stop("ERROR: step 5 writes the production classification with the settled method "
             "only. Refused:",
             *["    " + r for r in refused],
             "To try another model or setting, use Setup & Tuning (it writes to",
             "optimization_runs/<brain>/). To change the method, mark another run Paradigm Best.")
    print(f"\n[Settled method] imaging setup '{imaging_paradigm}': "
          f"{method['method']} classification, Paradigm Best {method['source_exp_id']}")
    print(f"  Model: {model}")
    print(f"  Cube size {method['cube_size']}, batch size {method['batch_size']}"
          + (f", tracing type {method['tracing_type']}" if method['tracing_type'] else ""))
    return method


def run_prefilter_or_stop(tracker, brain_name, pipeline_folder, candidates, registration_path):
    """Pre-filter the brain's candidates with PREFILTER_SETTINGS into
    4_Cell_Candidates/prefiltered/interior_candidates.xml, or stop (exit 1).

    The previous interior_candidates.xml (and anything describing it) is moved
    to prefiltered/_archive/<time>/ first: it is the input an earlier
    classification recorded, and was overwritten with no copy before.
    Returns what the record beside the production files says about it."""
    print(f"\n[{timestamp()}] Running atlas pre-filter...")
    try:
        from mousebrain.prefilter import prefilter_candidates, _coords_to_xml
        if not registration_path.exists():
            raise RuntimeError("no registration folder (%s)" % registration_path)
        result = prefilter_candidates(candidates_xml=candidates,
                                      registration_path=registration_path,
                                      **PREFILTER_SETTINGS)
        total = result['stats']['total']
        interior = result['stats']['interior']
        suspicious = result['stats']['suspicious']
        print(f"    Total candidates: {total}")
        print(f"    Interior (keep): {interior}")
        print(f"    Suspicious (removed): {suspicious}")
        if total > 0:
            print(f"    Pre-filter removed {suspicious/total*100:.1f}% of candidates")
        for cat, cnt in sorted(result.get('category_counts', {}).items(), key=lambda x: -x[1]):
            print(f"      {cat}: {cnt:,}")

        prefiltered_dir = pipeline_folder / FOLDER_DETECTION / "prefiltered"
        prefiltered_dir.mkdir(parents=True, exist_ok=True)
        from mousebrain.candidates_io import archive_previous
        previous, _ = archive_previous(prefiltered_dir, PREFILTERED_NAME)
        if previous:
            print(f"    Previous pre-filter output kept in {previous.parent}")
        prefiltered_xml = prefiltered_dir / PREFILTERED_NAME
        _coords_to_xml(result['interior_coords'], prefiltered_xml)
        print(f"    Using {interior} pre-filtered candidates for classification")
        pf_id = tracker.log_prefilter(
            brain=brain_name, total=total, interior=interior, suspicious=suspicious,
            tracing_type=PREFILTER_SETTINGS.get('tracing_type', 'descending'),
            # What was filtered: step 6 follows this back to the detection
            # run to record its settings with the counts.
            input_path=str(candidates), output_path=str(prefiltered_xml),
            status="completed")
        return {"exp_id": pf_id, "settings": dict(PREFILTER_SETTINGS),
                "input": str(candidates), "output": str(prefiltered_xml),
                "total": total, "interior": interior, "removed": suspicious}
    except Exception as e:
        print(f"\nERROR: The pre-filter failed: {e}")
        try:
            tracker.log_prefilter(
                brain=brain_name, tracing_type=PREFILTER_SETTINGS.get('tracing_type', 'descending'),
                input_path=str(candidates), status="failed",
                notes=f"step 5 stopped: {e}"[:500])
        except Exception as log_error:
            print(f"    [!] The failure could not be recorded in the tracker: {log_error}")
        print("Step 5 stopped. Nothing was classified, and the production")
        print("classification was not touched. Fix the cause above and run step 5 again.")
        sys.exit(1)


def run_selective(**kwargs):
    """mousebrain.selective_classifier.run_selective_classification (imported
    when used: it needs the atlas tools)."""
    from mousebrain.selective_classifier import run_selective_classification
    return run_selective_classification(**kwargs)


def _file_fingerprint(path: Path) -> dict:
    """Size and modification time of a file, so the record says WHICH file of
    that name was used (a model file replaced under the same name shows)."""
    try:
        st = Path(path).stat()
        return {"bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    except OSError:
        return {}


def _candidates_record(candidates: Path) -> dict:
    """The record step 4 wrote beside the candidates (which detection run made
    them, with its settings), or {} when there is none."""
    try:
        from mousebrain.candidates_io import provenance_path
        return json.loads(provenance_path(candidates).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Write the production classification with the settled method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 5_classify_cells.py                                        # Interactive
    python 5_classify_cells.py --brain 001_CNT_01_01_1p625x_z4          # The settled method
    python 5_classify_cells.py --brain 001_CNT_01_01_1p625x_z4 --dry-run  # Show it, change nothing

The method (model file, cube and batch size, selective or full, tracing type)
is the classification run marked Paradigm Best for the brain's imaging setup.
--model, --cube-size, --batch-size and --tracing-type are accepted only when
they equal it; --no-prefilter and --candidates are refused. Trying anything
else is done in Setup & Tuning, which never writes 5_Classified_Cells.
        """
    )

    parser.add_argument('--brain', '-b', help='Brain/pipeline to process')
    parser.add_argument('--model', '-m', type=Path,
                        help='Accepted only if it is the settled model file (a check)')
    parser.add_argument('--detection', '-d', help='Detection experiment ID to link to')

    # Parameters: the settled method's own; given here only as a check.
    parser.add_argument('--cube-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS,
                        help='CPUs left free while classifying (speed only, not the method)')

    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show the brain, candidates and settled method; change nothing')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    parser.add_argument('--routine', action='store_true',
                        help='Kept for old command lines; the settled method is always used')
    parser.add_argument('--no-prefilter', action='store_true',
                        help='Refused: the settled method pre-filters')
    parser.add_argument('--candidates', type=Path,
                        help='Refused: a saved or tuned pre-filter is not the settled method')
    parser.add_argument('--selective', action='store_true',
                        help='Kept for old command lines; selective or full comes from the '
                             'Paradigm Best run')
    parser.add_argument('--tracing-type', default=None,
                        choices=['descending', 'ascending', 'unknown'],
                        help='Accepted only if it is the settled tracing type (a check)')

    args = parser.parse_args()

    print("=" * 60)
    print("BrainGlobe Cell Classification")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)

    # Select brain first (needed for paradigm model lookup)
    if args.brain:
        pipeline_folder, mouse_folder = find_pipeline(args.brain, args.root)
        if not pipeline_folder:
            print(f"ERROR: Brain not found: {args.brain}")
            sys.exit(1)
        brain_name = f"{mouse_folder.name}/{pipeline_folder.name}"
        candidates = find_detection_output(pipeline_folder)
        if not candidates:
            print(f"ERROR: No detection output found for {args.brain}")
            print("Run Script 4 (4_detect_cells.py) first!")
            sys.exit(1)
    else:
        # Interactive selection
        brains = list_available_brains(args.root)
        if not brains:
            print("\nNo brains with detection output found!")
            print("Run Script 4 (4_detect_cells.py) first!")
            sys.exit(1)
        
        brain_info = interactive_select_brain(brains)
        if not brain_info:
            print("Cancelled.")
            return
        
        pipeline_folder = brain_info['pipeline']
        mouse_folder = brain_info['mouse']
        brain_name = brain_info['name']
        candidates = brain_info['candidates']

    tracker = ExperimentTracker()

    # =========================================================================
    # THE SETTLED METHOD: the run marked Paradigm Best, or nothing
    # =========================================================================
    parsed = parse_brain_name(pipeline_folder.name)
    imaging_paradigm = parsed.get('imaging_params', '')
    method = settled_method_or_stop(tracker, imaging_paradigm, args)
    model_path = Path(method['model_path'])
    cube_size = method['cube_size']
    batch_size = method['batch_size']

    # Load metadata -- from the images the atlas was fitted to, which are the
    # images detection's coordinates refer to (not always the crop: a brain
    # delivered already cut is registered on its full extraction).
    _source, crop_folder, _meta = registered_input(pipeline_folder)
    metadata_path = crop_folder / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: No metadata found at {metadata_path}")
        metadata = {}

    # Get paths and voxel sizes
    channels = metadata.get('channels', {})
    signal_ch = channels.get('signal_channel', 0)
    bg_ch = channels.get('background_channel', 1)
    signal_path = crop_folder / f"ch{signal_ch}"
    background_path = crop_folder / f"ch{bg_ch}"

    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (
        voxel.get('z', 4),
        voxel.get('y', 4),
        voxel.get('x', 4),
    )

    output_path = pipeline_folder / FOLDER_CLASSIFICATION
    registration_path = pipeline_folder / "3_Registered_Atlas"

    if args.dry_run:
        print("\n=== DRY RUN (nothing is pre-filtered, classified or written) ===")
        print(f"Brain: {brain_name}")
        print(f"Candidates: {candidates}")
        print(f"Method: {method['method']} (Paradigm Best {method['source_exp_id']})")
        print(f"Model: {model_path}")
        print(f"Cube size: {cube_size}  Batch size: {batch_size}")
        if method['method'] == 'selective':
            print(f"Tracing type: {method['tracing_type']}")
        print(f"Pre-filter: {PREFILTER_SETTINGS}")
        print(f"Output: {output_path}")
        return

    # The pre-filter, with its own settings (PREFILTER_SETTINGS). It either
    # runs or step 5 stops: a pre-filter that failed must not print a warning
    # and classify the UNFILTERED candidates -- surface glow and all -- into
    # the production classification with nothing on the run saying so.
    prefilter_record = run_prefilter_or_stop(tracker, brain_name, pipeline_folder,
                                             candidates, registration_path)
    prefilter_source = candidates
    candidates = Path(prefilter_record['output'])

    # The record written beside the production files (promoted_from.json,
    # "method"): everything that made them, so a count can be traced to its
    # model file and settings without the tracker.
    method_record = {
        "method": method['method'],
        "paradigm_best_exp_id": method['source_exp_id'],
        "paradigm_best_brain": method['source_brain'],
        "imaging_paradigm": imaging_paradigm,
        "model_path": str(model_path),
        "model_file": _file_fingerprint(model_path),
        "cube_size": cube_size,
        "batch_size": batch_size,
        "cube_depth": CUBE_DEPTH,
        "network_depth": NETWORK_DEPTH,
        "tracing_type": method['tracing_type'],
        "prefilter": prefilter_record,
        "candidates": str(prefilter_source),
        "candidates_record": _candidates_record(prefilter_source),
        "script": "5_classify_cells.py %s" % SCRIPT_VERSION,
    }

    if method['method'] == 'selective':
        run_selective_production(
            tracker, args, brain_name, pipeline_folder, candidates, model_path, method,
            registration_path, signal_path, background_path, voxel_sizes,
            method_record)
        return

    # =========================================================================
    # FULL CLASSIFICATION (the Paradigm Best is a plain classification run)
    # =========================================================================
    exp_id = tracker.log_classification(
        brain=brain_name,
        model_path=str(model_path),
        candidates_path=str(candidates),
        cube_size=cube_size,
        batch_size=batch_size,
        n_free_cpus=args.n_free_cpus,
        output_path=str(output_path),
        parent_experiment=args.detection,
        notes=args.notes or ("settled method: Paradigm Best %s" % method['source_exp_id']),
        status="started",
        script_version=SCRIPT_VERSION,
    )

    print(f"\n{'='*60}")
    print(f"Classification Run: {exp_id}")
    print(f"Brain: {brain_name}")
    print(f"{'='*60}")

    # If this stops part-way (an error, Ctrl-C, a crash inside the tool),
    # the tracker row is set to failed instead of staying 'started' forever.
    with tracker.finish_or_fail(exp_id):
        # The run writes into 5_Classified_Cells/_in_progress/<time>/, never
        # into the production folder (mousebrain.classification_runs says
        # why): the production cells.xml stays in place for the whole
        # multi-hour run, so a stop no handler sees -- the process killed, a
        # restart, power or the share lost -- leaves it untouched. Only a run
        # that finished swaps its files in, in one short step that archives
        # the previous ones.
        for left in sweep_stopped_runs(pipeline_folder):
            print(f"    An earlier run that was stopped left files; kept in {left}")
        staged = staging_folder(pipeline_folder)
        try:
            # Run classification
            success, duration, cells_found, rejected = run_cellfinder_classify(
                signal_path=signal_path,
                candidates_path=candidates,
                output_path=staged,
                model_path=model_path,
                voxel_sizes=voxel_sizes,
                cube_size=cube_size,
                batch_size=batch_size,
                n_free_cpus=args.n_free_cpus,
                background_path=background_path,
            )
        except BaseException:
            discard_staged(staged, pipeline_folder, "failed")
            raise
        if success and (staged / "cells.xml").exists():
            method_record["counts"] = {"cells": cells_found, "rejected": rejected}
            _install(tracker, pipeline_folder, staged, exp_id, method_record)
        else:
            success = False
            kept = discard_staged(staged, pipeline_folder, "failed")
            if kept:
                print(f"    What the failed run wrote is kept in {kept}; "
                      f"the production classification is unchanged")

        # Update tracker
        tracker.update_status(
            exp_id,
            status="completed" if success else "failed",
            duration_seconds=round(duration, 1),
            class_cells_found=cells_found,
            class_rejected=rejected,
        )

    print(f"\n{'='*60}")
    if success:
        print(f"COMPLETED in {duration/60:.1f} minutes")
        print(f"Cells classified: {cells_found}")
        print(f"Rejected: {rejected}")
        if cells_found + rejected > 0:
            pct = cells_found / (cells_found + rejected) * 100
            print(f"Acceptance rate: {pct:.1f}%")
    else:
        print(f"FAILED after {duration/60:.1f} minutes")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")

    if success:
        _rate_and_next(tracker, exp_id, brain_name)


def run_selective_production(tracker, args, brain_name, pipeline_folder, candidates,
                             model_path, method, registration_path, signal_path,
                             background_path, voxel_sizes, method_record):
    """Selective classification as the production classification.

    Written into the staging folder (as a full classification is), installed
    as cells.xml + rejected.xml + its audit files only when it finished with a
    result; the previous production files go to _archive/<time>/. The
    classifier is run with on_classifier_error="raise": a classifier that could
    not run must stop the run, not leave a result of trusted cells only."""
    output_path = pipeline_folder / FOLDER_CLASSIFICATION
    if not registration_path.exists():
        print("ERROR: No registration folder found -- required for selective classification")
        print("Nothing was classified.")
        sys.exit(1)

    sel_exp_id = tracker.log_selective_classification(
        brain=brain_name,
        model_path=str(model_path),
        tracing_type=method['tracing_type'],
        candidates_path=str(candidates),
        output_path=str(output_path),
        parent_experiment=args.detection,
        notes=args.notes or ("settled method: Paradigm Best %s" % method['source_exp_id']),
        status="started",
        script_version=SCRIPT_VERSION,
        cube_size=method['cube_size'],
        batch_size=method['batch_size'],
    )
    print(f"\nSelective Classification Run: {sel_exp_id}")

    success = False
    result = None
    with tracker.finish_or_fail(sel_exp_id):
        for left in sweep_stopped_runs(pipeline_folder):
            print(f"    An earlier run that was stopped left files; kept in {left}")
        staged = staging_folder(pipeline_folder)
        try:
            result = run_selective(
                brain_path=pipeline_folder,
                candidates_xml=candidates,
                model_path=model_path,
                registration_path=registration_path,
                tracing_type=method['tracing_type'],
                signal_path=signal_path,
                background_path=background_path,
                voxel_sizes=voxel_sizes,
                cube_size=method['cube_size'],
                batch_size=method['batch_size'],
                n_free_cpus=args.n_free_cpus,
                output_dir=staged,
                on_classifier_error="raise",
            )
        except BaseException:
            discard_staged(staged, pipeline_folder, "failed")
            raise
        # The selective run names its files for what they are; production
        # names them as every reader expects (step 6 counts cells.xml).
        for src, dst in (("cells_selective.xml", "cells.xml"),
                         ("selective_rejected.xml", "rejected.xml")):
            if (staged / src).is_file():
                (staged / src).replace(staged / dst)
        success = bool(result is not None and result.final_cells > 0
                       and (staged / "cells.xml").is_file())
        if success:
            method_record["counts"] = {
                "candidates": result.total_candidates,
                "trusted": result.trust_count,
                "sent_to_classifier": result.classify_count,
                "classifier_approved": result.classified_cells,
                "classifier_rejected": result.classified_rejected,
                "cells": result.final_cells,
            }
            _install(tracker, pipeline_folder, staged, sel_exp_id, method_record)
        else:
            kept = discard_staged(staged, pipeline_folder, "failed")
            print("    The selective classification found no cells; the production "
                  "classification is unchanged" + (f" (what it wrote is in {kept})" if kept else ""))

        r = result
        tracker.update_status(
            sel_exp_id,
            status="completed" if success else "failed",
            duration_seconds=round(r.duration_seconds, 1) if r else None,
            class_cells_found=r.final_cells if r else None,
            class_rejected=r.classified_rejected if r else None,
            prefilter_total=r.total_candidates if r else None,
            prefilter_interior=r.trust_count if r else None,
            prefilter_suspicious=r.classify_count if r else None,
            notes="; ".join(filter(None, [
                args.notes or ("settled method: Paradigm Best %s" % method['source_exp_id']),
                ("trust=%d classify=%d approved=%d final=%d"
                 % (r.trust_count, r.classify_count, r.classified_cells, r.final_cells))
                if r else ""])),
        )

    print(f"\nExperiment ID: {sel_exp_id}")
    if success:
        _rate_and_next(tracker, sel_exp_id, brain_name)
    else:
        sys.exit(1)


def _install(tracker, pipeline_folder, staged, exp_id, method_record):
    """Make a finished run's staged files the production classification, keep
    the previous ones in _archive/<time>/, point the tracker rows at where the
    files are now, and take the counts made from the replaced classification
    out of the production count tables."""
    output_path = pipeline_folder / FOLDER_CLASSIFICATION
    archived, prev_id, _ = install_staged(
        staged, pipeline_folder, exp_id,
        source="5_classify_cells.py %s" % SCRIPT_VERSION, how="step 5",
        method=method_record)
    if archived:
        print(f"    Previous production classification archived to {archived}")
    print(f"    Production classification: {output_path / 'cells.xml'}")
    # This run's row names the file it wrote; the previous writer's row
    # follows its files into the archive.
    tracker.update_status(exp_id, output_path=str(output_path / "cells.xml"))
    follow_files_into_archive(tracker, prev_id, archived, pipeline_folder)
    follow_selective_rows(tracker, pipeline_folder, archived)
    # Counts made from the classification just replaced leave the
    # production count tables (region_counts_io.withdraw_counts).
    from mousebrain.region_counts_io import withdraw_counts_or_warn
    withdraw_counts_or_warn(pipeline_folder.name,
                            "classification replaced by step 5 run %s" % exp_id)


def _rate_and_next(tracker, exp_id, brain_name):
    try:
        rating = input("\nRate this run (1-5, or Enter to skip): ").strip()
        if rating and rating.isdigit() and 1 <= int(rating) <= 5:
            note = input("Add a note (or Enter to skip): ").strip()
            tracker.rate_experiment(exp_id, int(rating), note if note else None)
    except (EOFError, KeyboardInterrupt):
        pass

    # Next step guidance
    print("\n" + "=" * 60)
    print("WHAT TO DO NEXT")
    print("=" * 60)
    print("\n1. Run region counting:")
    print(f"   python 6_count_regions.py --brain {brain_name.split('/')[-1]}")
    print("\n" + "-" * 60)
    print("OR just run: python RUN_PIPELINE.py")
    print("   (it will guide you through everything)")
    print("=" * 60)


if __name__ == '__main__':
    main()
