"""
batch_encr.py - Batch colocalization analysis for a project's HD-region ND2 files.

Loads each ND2 file, runs threshold/Otsu nuclei detection on the nuclear
channel, measures signal colocalization in the other channel, saves CSV
results + QC overlay images, and records the run in the tracker.

Usage:
    python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project ENCR
    python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project ENCR --threshold 1.5 --dilation 50
    python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project ENCR --output <folder>

Where the files are searched, in order (the first root holding ND2 files wins):
  1. new layout:    <2D_Slices>/1_Subjects/<PROJECT>/<PROJECT>_XX_XX/0_Raw_HD and 0_Raw
  2. legacy layout: <2D_Slices>/<PROJECT>/<PROJECT>_XX_XX_HD_Regions[/Corrected]
(<2D_Slices> is mousebrain.config.SLICES_2D_DIR, i.e. from CONNECTOME_ROOT;
--input replaces both with one explicit root). The project code can also be
given as the MOUSEBRAIN_2D_PROJECT environment variable. When no root holds
any ND2 file the script names the roots it searched and exits 1.

METHOD (fixed by the code)
  DETECTION:       threshold+LoG, nucleus diameter BATCH_NUCLEUS_DIAMETER_UM
                   (10-25 um; converted to pixel area per image from its pixel size)
  COLOCALIZATION:  background_mean with a sigma threshold
  BACKGROUND:      gmm (with a percentile fallback)

PARAMETERS (chosen per run; the banner shows each value and where it came from)
  sigma threshold, background percentile, soma dilation (px), background
  exclusion dilation (iterations), detector threshold_fraction / log_threshold.
  Each is taken from, in order:
    1. the command line (--threshold, --bg-percentile, --soma-dilation, --dilation)
    2. a colocalization run of this project marked best in the tracker (sigma
       threshold and background percentile only; see get_tuned_parameters for
       which runs count)
    3. the approved method the registry judges this batch's entries by
       (batch_approved_method)
    4. BUILTIN_PARAMETERS
  --ignore-tracker skips step 2. A marked run's sigma is used only at the soma
  and exclusion dilations it was made at: when the batch would take a number
  from the marked run and run at other dilations, it stops before processing
  unless --soma-dilation / --dilation are typed (dilation_mismatches). With
  both --threshold and --bg-percentile typed the marked run supplies nothing
  and is not checked.
  Marking a run best CHOOSES the batch's numbers; it does not change the
  approved method. The registry judges every entry against the approved method
  only, so numbers taken from a marked run that differ from it are registered
  as stale until the same numbers are written into the approved method.

How background_mean works:
  1. Detect nuclei in the nuclear channel (threshold + LoG, diameter range above).
  2. Dilate each nucleus (soma dilation) to measure cytoplasmic signal.
  3. Dilate all nuclei generously (exclusion dilation) to create background
     exclusion zones.
  4. Background = mean of the signal channel in tissue OUTSIDE exclusion zones.
  5. Cell is positive if: soma signal > background_mean + sigma * std.

Keep the reasoning for a method change in your own method log, not here.
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.config import DATA_DIR, get_sample_dir, SampleDirs, parse_sample_name

# WHY: the legacy layout lives one level ABOVE the 1_Subjects tree, so discovery
# needs the 2D root itself, not only DATA_DIR (= <2D_Slices>/1_Subjects). Guarded
# so a standalone sliceatlas install (no mousebrain.config) still imports.
try:
    from mousebrain.config import SLICES_2D_DIR
except ImportError:
    SLICES_2D_DIR = None


# Project code: --project on the command line, else MOUSEBRAIN_2D_PROJECT.
# The data root is <DATA_DIR>/<PROJECT>; there is no built-in lab path.
import os as _os
PROJECT = _os.environ.get("MOUSEBRAIN_2D_PROJECT")
ENCR_ROOT = (Path(DATA_DIR) / PROJECT) if (DATA_DIR and PROJECT) else None
DEFAULT_OUTPUT = (Path(DATA_DIR).parent / "2_Data_Summary" / "batch_results") if DATA_DIR else None

# Default channel indices for ENCR ND2 files
# Channel 0 = 488nm (green/eYFP signal)
# Channel 1 = 561nm (red/nuclear)
RED_IDX = 1
GREEN_IDX = 0

# Nucleus diameter range (micrometers) the batch detects with: 10-25 um, the
# default of the detector it calls (core.detection.detect_with_log_augmentation).
# It is passed to the detector explicitly and recorded with every output.
# OPEN QUESTION, deliberately not settled in code: the approved-method default
# (analysis_registry.DEFAULT_METHOD['size_filter_um']) says 8-25 um. Which range
# is approved is for the lab to decide; until then the batch detects 10-25 um,
# says so in its banner and records 10-25 um, so its registry entries show as
# not matching an approved method that says otherwise.
BATCH_NUCLEUS_DIAMETER_UM = (10, 25)

# The method the batch runs. These are fixed by the code; they are recorded in
# each registry entry so an approved method naming something else shows the
# entry as not produced with it.
# NOT recorded yet: the region each cell's signal is measured in. The batch
# measures a non-overlapping ring around each nucleus that leaves the nucleus
# out (ColocalizationAnalyzer.measure_cytoplasm_intensities). Adding a key for
# it changes every entry's method hash, and what the key and its approved value
# should be is the lab's decision: a second open question, written up with the
# size range in plugin_2d/ALGORITHM_REFERENCE.md, section 6.
BATCH_FIXED_METHOD = {
    "detection": "threshold+log",
    "log_decision_tree": True,
    "colocalization": "background_mean",
    "background": "gmm",
}

# Last-resort values, used only when neither the command line, a marked-best
# run nor the approved method gives one. Equal to the detector's and the
# registry's built-in defaults.
BUILTIN_PARAMETERS = {
    "sigma_threshold": 0,
    "soma_dilation": 6,
    "bg_exclusion_dilation": 50,
    "background_percentile": 10,
    "threshold_fraction": 0.20,
    "log_threshold": 0.005,
}

# The registry analysis the batch registers its outputs under.
REGISTRY_ANALYSIS = "ENCR_Detection"


def batch_approved_method() -> dict:
    """The approved method this batch's registry entries are judged by.

    WHY ask the registry module for THIS analysis: the approved method is
    what the registry compares each entry's method hash with. A registry that
    can give an analysis its own approved method (approved_method_for) may
    judge REGISTRY_ANALYSIS by something other than the general approved
    method; filling unset options from, and comparing the banner against, the
    general one would then announce "approved" numbers the registry reports
    as stale. A registry without per-analysis methods has one approved method
    (get_approved_method) and that is the one used. A per-analysis entry of
    null means "no approved method": the batch then fills unset options from
    BUILTIN_PARAMETERS and the banner compares with nothing.

    Raises RuntimeError when an approved-method file is set but unreadable.
    """
    from mousebrain import analysis_registry as ar
    per_analysis = getattr(ar, "approved_method_for", None)
    if per_analysis is not None:
        return dict(per_analysis(REGISTRY_ANALYSIS) or {})
    return ar.get_approved_method()


def approved_method_is_shared() -> bool:
    """True when the batch's approved method is the registry's SHARED one.

    WHY the batch asks: its banner tells a person how to make the batch's
    numbers approved -- by writing them into the approved-method file. When
    the file does not name this batch's analysis (no file, a flat file, or a
    per-analysis file without an entry for REGISTRY_ANALYSIS), the batch is
    judged by the shared method, and so is every other analysis that follows
    it (the ROI analysis included): editing the shared method changes what
    their existing entries are judged against too, and their later runs
    record it. The person must be told that before editing the file, not
    after. When the file gives REGISTRY_ANALYSIS its own entry, only that
    entry needs editing and the warning is not given.

    approved_method_for(..., follows_approved_method=False) answers exactly
    "does the file name this analysis with a method": it returns that method,
    or None when the analysis is not named (or named with null, in which case
    the batch has no approved method and the banner lists no difference). A
    registry module without approved_method_for has only the shared method.
    """
    try:
        from mousebrain import analysis_registry as ar
    except ImportError:
        return False
    per_analysis = getattr(ar, "approved_method_for", None)
    if per_analysis is None:
        return True
    return per_analysis(REGISTRY_ANALYSIS, follows_approved_method=False) is None


# command-line option (argparse dest) -> approved-method key
_CLI_TO_METHOD_KEY = {
    "threshold": "sigma_threshold",
    "soma_dilation": "soma_dilation",
    "dilation": "bg_exclusion_dilation",
    "bg_percentile": "background_percentile",
}
# marked-best run (get_tuned_parameters key) -> approved-method key
_TUNED_TO_METHOD_KEY = {
    "threshold": "sigma_threshold",
    "bg_percentile": "background_percentile",
}


def resolve_parameters(cli: dict, tuned: dict, approved: dict) -> tuple:
    """Decide each batch parameter and where it came from.

    Precedence, per parameter: a value typed on the command line, then a
    marked-best run (sigma threshold and background percentile only), then
    the approved method, then BUILTIN_PARAMETERS.

    WHY "typed" means "not None": the options used to default to a number and
    a tracker value replaced the option whenever it EQUALLED that number, so
    typing the default on purpose (e.g. --threshold 0) was overridden.

    Args:
        cli: argparse values keyed by dest (threshold, soma_dilation,
            dilation, bg_percentile); None = not given.
        tuned: get_tuned_parameters() result ({} when none).
        approved: batch_approved_method() result.

    Returns:
        (values, sources): both keyed by approved-method key (sigma_threshold,
        soma_dilation, bg_exclusion_dilation, background_percentile,
        threshold_fraction, log_threshold).
    """
    values, sources = {}, {}
    from_cli = {_CLI_TO_METHOD_KEY[k]: v for k, v in cli.items()
                if k in _CLI_TO_METHOD_KEY and v is not None}
    from_tuned = {_TUNED_TO_METHOD_KEY[k]: v for k, v in tuned.items()
                  if k in _TUNED_TO_METHOD_KEY}
    runs = ", ".join(tuned.get("source_runs", []))
    for key, builtin in BUILTIN_PARAMETERS.items():
        if key in from_cli:
            values[key], sources[key] = from_cli[key], "command line"
        elif key in from_tuned:
            values[key], sources[key] = from_tuned[key], f"marked-best run {runs}"
        elif key in approved:
            values[key], sources[key] = approved[key], "approved method"
        else:
            values[key], sources[key] = builtin, "built-in default"
    return values, sources


def batch_method_params(approved: dict, values: dict) -> dict:
    """The method parameters a batch run actually used, for the registry.

    Exactly what the batch decides: the resolved values, the fixed method
    names and the diameter range -- nothing else.

    WHY: entries used to record the approved method whatever the run used, so
    a run with --threshold 1.5 hashed like an approved run and could never be
    found stale. WHY nothing else: a key of the approved method that the
    batch does not control (say an approved-method file naming another
    detection method under its own key) was not used by the run, and copying
    it in would record it as if it had been. Leaving it out makes the entry
    differ from that approved method, so it is reported stale and the banner
    lists the key (main()). A value equal to the approved one keeps the
    approved spelling (10 vs 10.0) so an approved run hashes exactly like the
    approved method.
    """
    used = dict(BATCH_FIXED_METHOD)
    used.update(values)
    used["size_filter_um"] = list(BATCH_NUCLEUS_DIAMETER_UM)
    return {key: (approved[key] if key in approved and approved[key] == value
                  else value)
            for key, value in used.items()}


class TunedParameterConflict(Exception):
    """Two or more marked-best runs of this batch's project give different numbers.

    WHY this stops the batch instead of taking the newest: choosing one of two
    marked runs silently is a guess about which one the person meant. The
    message names the runs (and their samples) so the person can clear the
    marks they did not mean with the Run History panel's "Unmark Best" button
    -- marking another run only clears marks of the SAME sample -- or type
    both values.
    """


def _recorded_note_int(row: dict, key: str):
    """An integer setting written as '<key>=<n>' in a run's notes, or None.

    Colocalization rows have no column for the soma dilation or the background
    exclusion dilation, so the widget and the batch write both into the notes
    ("soma_dilation=6; bg_exclusion_dilation=50").
    """
    import re
    m = re.search(r'(?<![A-Za-z_])' + re.escape(key) + r'=(-?\d+)',
                  row.get('notes', '') or '')
    return int(m.group(1)) if m else None


def _recorded_soma_dilation(row: dict):
    """The soma dilation (px) written in a run's notes, or None."""
    return _recorded_note_int(row, 'soma_dilation')


def _recorded_exclusion_dilation(row: dict):
    """The background exclusion dilation (iterations) in a run's notes, or None."""
    return _recorded_note_int(row, 'bg_exclusion_dilation')


def batch_approval_problem(row: dict):
    """Why a marked colocalization run cannot supply the batch's numbers, or None.

    The batch runs one fixed method: green signal channel, GMM background,
    background_mean classification with a sigma threshold. A marked-best run
    gives the batch its numbers only when it was run with that same
    classification, finished, and recorded the sigma it used -- otherwise its
    number means something else (a fold change, a dual-channel threshold) and
    would become the batch's sigma threshold unnoticed.

    NOT the same measurement: the napari widget measures each cell's signal in
    a dilated disk around the nucleus (measure_soma_intensities; disks of
    neighboring cells may overlap), the batch in a non-overlapping ring that
    leaves out the nucleus (measure_cytoplasm_intensities), and the widget
    detects nuclei with its own detection settings. A sigma chosen on the
    widget is therefore applied to a related, not identical, measurement. A run
    with soma dilation 0 measured the NUCLEUS itself (measure_nuclei_intensities),
    which has nothing in common with the batch's ring, so it is refused.

    Shared with the napari widget's "Mark Selected as Best" button, so a person
    marking a run is told at once when the batch will not use it.
    """
    from ..tracker.schema import records_sigma

    if row.get('run_type') != 'colocalization':
        return "it is not a colocalization run"
    if row.get('status') != 'completed':
        return f"it did not complete (status '{row.get('status', '')}')"
    if row.get('coloc_signal_channel') != 'green':
        return (f"it measured signal channel '{row.get('coloc_signal_channel', '')}'; "
                "the batch measures the green channel only")
    if row.get('coloc_threshold_method') != 'background_mean':
        return (f"it classified with '{row.get('coloc_threshold_method', '')}'; "
                "the batch classifies with background_mean")
    if row.get('coloc_background_method') != 'gmm':
        return (f"it estimated background with '{row.get('coloc_background_method', '')}'; "
                "the batch uses gmm")
    if not records_sigma(row):
        return ("it was logged before runs recorded their sigma threshold; "
                "re-run it and mark the new run")
    soma = _recorded_soma_dilation(row)
    if soma is None:
        return "its soma dilation is not recorded"
    if soma <= 0:
        return ("it measured the nucleus itself (soma dilation 0); the batch "
                "measures a ring around each nucleus")
    if _recorded_exclusion_dilation(row) is None:
        return "its background exclusion dilation is not recorded"
    try:
        float(row.get('coloc_threshold_value', ''))
    except (TypeError, ValueError):
        return "it has no sigma threshold recorded"
    return None


def _run_belongs_to_batch(row: dict, project, sample_ids) -> bool:
    """True when a tracker row is about this batch's project.

    A row counts when its sample is one of the files the batch is about to
    process, or when the tracker filed it under the batch's project (its
    'project' column, parsed from the sample name). WHY both: the tracker's
    project comes from parsing the sample name, which for a compact name gives
    the parsed code rather than the folder the batch was started on; the file
    list catches those.
    """
    if project is None and sample_ids is None:
        return True
    if sample_ids is not None and row.get('sample_id', '') in sample_ids:
        return True
    return project is not None and row.get('project', '') == project


def get_tuned_parameters(verbose: bool = True, project=None, sample_ids=None) -> dict:
    """Parameters chosen by marking a colocalization run best in the tracker.

    Only runs of THIS batch's project count (see _run_belongs_to_batch).
    WHY: the 2D tracker is shared by every project, and a run marked best for
    one project's slices used to set another project's batch (or stop it with
    a conflict). main() passes the project and its file stems; called with
    neither, every marked run counts.

    Only runs a person has marked best count. WHY: the batch used to fall back
    to the newest completed run of any kind when nothing was marked -- which
    was usually the previous batch's own row -- so "tuned" parameters were
    whatever ran last, approved by nobody.

    A marked-best run supplies the sigma threshold and the background
    percentile, and only when batch_approval_problem() finds nothing wrong with
    it (a refused run is printed with its reason when *verbose*). It does NOT
    supply the soma dilation or the background exclusion dilation: those come
    from the command line or the approved method. The dilations the run was
    made at (recorded in its notes) are returned as well, because a sigma is
    chosen for one measurement: main() stops when the batch would run at other
    dilations than the marked run, unless the person typed them (see
    dilation_mismatches). Runs that agree on the sigma and percentile but were
    made at different dilations count as disagreeing.

    mark_as_best keeps one best run PER SAMPLE, so several samples can each have
    one. When the qualifying runs agree they are used; when they disagree,
    TunedParameterConflict is raised (see its docstring for why).

    Returns:
        {} when no qualifying run is marked best, else
        {'threshold': float, 'bg_percentile': float (when recorded),
         'source_runs': [run ids]}.
    """
    try:
        from ..tracker import SliceTracker
        tracker = SliceTracker()
    except Exception as e:
        if verbose:
            print(f"[!] tracker unavailable, no marked-best run can be used: {e}")
        return {}

    # limit=None: search() keeps only the 50 newest rows by default, and an
    # approval made weeks ago must not drop out of sight as new runs are logged.
    rows = tracker.search(run_type='colocalization', limit=None)
    best_rows = [r for r in rows if r.get('marked_best') == 'True']
    elsewhere = [r for r in best_rows
                 if not _run_belongs_to_batch(r, project, sample_ids)]
    best_rows = [r for r in best_rows
                 if _run_belongs_to_batch(r, project, sample_ids)]
    if elsewhere and verbose:
        print(f"  {len(elsewhere)} marked-best run(s) of other projects not used")

    approvals = {}
    for row in best_rows:
        problem = batch_approval_problem(row)
        if problem:
            if verbose:
                print(f"[!] marked-best run {row.get('run_id', '')} "
                      f"({row.get('sample_id', '')}) not used: {problem}")
            continue
        sigma = float(row['coloc_threshold_value'])
        pct = row.get('coloc_background_percentile', '')
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            pct = None
        key = (sigma, pct, _recorded_soma_dilation(row),
               _recorded_exclusion_dilation(row))
        approvals.setdefault(key, []).append(
            f"{row.get('run_id', '')} ({row.get('sample_id', '')})")

    if not approvals:
        return {}
    if len(approvals) > 1:
        lines = [f"  sigma={s}, background percentile={p}, soma dilation={sd}, "
                 f"exclusion dilation={ed}: {', '.join(ids)}"
                 for (s, p, sd, ed), ids in sorted(approvals.items(),
                                                   key=lambda kv: str(kv[0]))]
        raise TunedParameterConflict(
            "marked-best runs give different parameters:\n" + "\n".join(lines))

    (sigma, pct, soma, excl), ids = next(iter(approvals.items()))
    params = {'threshold': sigma,
              'run_soma_dilation': soma,
              'run_bg_exclusion_dilation': excl,
              'source_runs': sorted(i.split(' (')[0] for i in ids)}
    if pct is not None:
        params['bg_percentile'] = pct
    return params


# get_tuned_parameters key -> resolved-value key, for the dilations a marked
# run was made at (it records them but does not supply them).
_RUN_DILATION_KEYS = {
    'run_soma_dilation': 'soma_dilation',
    'run_bg_exclusion_dilation': 'bg_exclusion_dilation',
}


def dilation_mismatches(tuned: dict, values: dict) -> list:
    """Dilations the batch would run at that differ from the marked run's.

    WHY: a sigma threshold is chosen by looking at one measurement -- the
    signal in a region of a given size around each nucleus, against a
    background with the nuclei excluded by a given dilation. The same sigma
    at another soma or exclusion dilation is a different measurement, while
    the banner would still credit the sigma to the marked run. So the batch
    states the difference instead of guessing which one was meant.

    Returns [(key, the run's value, the batch's value), ...]; empty when they
    match or no marked run supplied a sigma.
    """
    out = []
    for run_key, key in _RUN_DILATION_KEYS.items():
        recorded = tuned.get(run_key)
        if recorded is not None and values.get(key) != recorded:
            out.append((key, recorded, values.get(key)))
    return out


# Layout labels reported in the banner and returned by discover_nd2_files().
LAYOUT_NEW = "1_Subjects"
LAYOUT_LEGACY = "legacy"


def _scan_new_layout(root: Path, project: str) -> list:
    """ND2 files in the 1_Subjects layout under ONE project root.

    <root>/<PROJECT>_XX_XX/0_Raw_HD/*.nd2 and <root>/<PROJECT>_XX_XX/0_Raw/*.nd2
    """
    nd2_files = []
    for subject_dir in sorted(root.glob("%s_*" % project)):
        if not subject_dir.is_dir():
            continue
        # HD region files
        hd_dir = subject_dir / SampleDirs.RAW_HD
        if hd_dir.exists():
            nd2_files.extend(sorted(hd_dir.glob("*.nd2")))
        # Standard raw files
        raw_dir = subject_dir / SampleDirs.RAW
        if raw_dir.exists():
            nd2_files.extend(sorted(raw_dir.glob("*.nd2")))
    return nd2_files


def _scan_legacy_layout(root: Path, project: str) -> list:
    """ND2 files in the pre-1_Subjects layout under ONE project root.

    <root>/<PROJECT>_XX_XX_HD_Regions/Corrected/*.nd2 first, then the files in
    <root>/<PROJECT>_XX_XX_HD_Regions/ itself whose stem has no Corrected copy.
    WHY Corrected wins: it holds the operator-corrected re-export of the same
    slice; taking both would count that slice twice.
    """
    nd2_files = []
    for hd_dir in sorted(root.glob("%s_*_HD_Regions" % project)):
        if not hd_dir.is_dir():
            continue
        corrected = hd_dir / "Corrected"
        corrected_files = sorted(corrected.glob("*.nd2")) if corrected.exists() else []
        nd2_files.extend(corrected_files)
        corrected_stems = {f.stem for f in corrected_files}
        for f in sorted(hd_dir.glob("*.nd2")):
            if f.stem not in corrected_stems:
                nd2_files.append(f)
    return nd2_files


def find_nd2_files(root: Path, project: str = None) -> list:
    """Find the ND2 files under ONE project root, whichever layout it holds.

    The 1_Subjects layout is tried first; the legacy layout only when the
    former yields nothing under this root (see the module docstring for the
    two trees). Returns [] when the root does not exist or holds no ND2 file.
    """
    root = Path(root)
    project = project or PROJECT or root.name
    if not root.is_dir():
        return []
    return _scan_new_layout(root, project) or _scan_legacy_layout(root, project)


def input_roots(project: str, explicit_input=None) -> list:
    """Project roots to search, in priority order.

    --input names exactly one root and is searched alone. Otherwise the new
    layout root (<1_Subjects>/<project>) comes first and the legacy root
    (<2D_Slices>/<project>) second.
    WHY the legacy root is still searched: raw slices that were never moved
    into 1_Subjects/ sit in the old per-project folder. Searching only the
    new root made the documented invocation find 0 files and exit 0 as if
    the batch had run.
    """
    if explicit_input is not None:
        return [Path(explicit_input)]
    roots = []
    if not project:
        return roots
    if DATA_DIR:
        roots.append(Path(DATA_DIR) / project)
    if SLICES_2D_DIR:
        roots.append(Path(SLICES_2D_DIR) / project)
    return roots


def discover_nd2_files(roots, project: str = None) -> tuple:
    """Search roots in order; return (files, root_used, layout).

    Within each root the 1_Subjects layout is tried before the legacy one.
    The first (root, layout) that holds any ND2 file wins and nothing later
    is merged in. WHY: the two layouts can hold copies of the same slices
    while a migration is under way; merging them would process each slice
    twice. (files, root_used, layout) is ([], None, None) when nothing was
    found anywhere.
    """
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        proj = project or PROJECT or root.name
        for layout, scan in ((LAYOUT_NEW, _scan_new_layout), (LAYOUT_LEGACY, _scan_legacy_layout)):
            files = scan(root, proj)
            if files:
                return files, root, layout
    return [], None, None


# Steps of process_single in which a failure can be about the SAMPLE: taking
# the two channels out of its image, detecting its nuclei, measuring its
# cells. All three work on an image that is already in memory.
# NOT among them, on purpose:
#   "read"    -- getting the file off the disk (load_image). Whatever the
#                reader raises there keeps the earlier entry: it reports a
#                file it does not recognize with an OSError, and a file whose
#                end is missing (cut off, or still being copied) with a
#                ValueError. Neither can be told from a share that drops out
#                mid-read, and a file that is only half there says nothing
#                about the sample.
#   "tracker", "save" -- tracker writes and saving outputs.
SAMPLE_STEPS = ("channels", "detect", "colocalize")

# The errors that can say something about the sample's DATA: what the channel
# extraction, the detector and the measuring code raise for an image they
# cannot use (a wrong number of channels or dimensions, a missing metadata
# key, an index past the image's channels). They count only inside
# SAMPLE_STEPS; the same error types raised while READING the file do not
# (see above).
SAMPLE_DATA_ERRORS = (ValueError, KeyError, IndexError)

# Which finished samples show that the software can get past a step. A sample
# that ended with no nuclei read its image and ran the detector, but never
# reached the measuring step, so it is no evidence about that step.
STEP_EVIDENCE = {
    "channels": ("completed", "no_nuclei"),
    "detect": ("completed", "no_nuclei"),
    "colocalize": ("completed",),
}

# Libraries the sample steps import only when they first run. Checked once
# before the first sample (check_environment): a missing or broken one would
# otherwise surface as each sample's own error.
REQUIRED_MODULES = (
    "nd2", "pandas", "matplotlib", "scipy.ndimage", "scipy.stats",
    "scipy.interpolate", "skimage.feature", "skimage.filters",
    "skimage.measure", "skimage.morphology", "skimage.segmentation",
    "sklearn.mixture",
)


def check_environment() -> list:
    """Import every library the sample steps need; return [(module, error)].

    WHY before the loop: those libraries are imported lazily inside the
    reading, detecting and measuring code. In an environment that lacks one
    (or holds a broken build, which can fail with an ImportError, an OSError
    from a DLL or a ValueError about binary sizes), every sample would fail
    the same way, one after another. Checking first stops the batch with one
    message naming the library, before any sample is attempted.
    """
    import importlib
    problems = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:  # any failure to import is the environment's
            problems.append((name, f"{type(e).__name__}: {e}"))
    return problems


def failure_is_the_samples(step: str, exc: BaseException) -> bool:
    """True when a run failed because of the sample, not the machinery around it.

    WHY this distinction: when a sample's re-run fails, its earlier registry
    entry is retired (main()) only if the failure says something about the
    sample -- an image with the wrong channels or shape, a detector that
    rejects it.
    A share that cannot be reached, a tracker file another program holds
    open, a full disk, a machine out of memory, a library that is missing or
    broken, or a defect in the code says nothing about the sample
    (infrastructure failures are not scientific outcomes); retiring a good
    result because of one would drop it for no reason, so those keep the
    earlier entry current.

    So the rule is a positive list: a failure counts as the sample's only
    when it happened while taking the channels out of the loaded image,
    detecting or measuring (SAMPLE_STEPS) AND is one of the data errors in
    SAMPLE_DATA_ERRORS. Everything else is infrastructure and keeps the
    entry:
      - ANY error while reading the file from disk (step "read"), whatever
        its type. The ND2 reader raises an OSError for a file it does not
        recognize and a ValueError for a file whose end is missing (cut off,
        or still being copied); both keep the entry, as does a share that
        drops out mid-read.
      - at any step: ImportError and ModuleNotFoundError, OSError,
        MemoryError, TypeError, AttributeError, NameError, RuntimeError,
        EOFError and anything not listed.

    WHAT THIS RULE CANNOT KNOW: the three listed errors are also what a
    defect in the code or in a library typically raises (a missing table
    column, an array of the wrong shape). An error type alone does not say
    whose fault it is. main() therefore adds a second condition -- another
    sample of the same run must have got past the step that failed
    (STEP_EVIDENCE) -- which rules out a defect that hits every image. A
    defect that hits only SOME images, raises one of the listed errors and
    happens in a run where another sample got past that step is still taken
    for the sample's, and that sample's earlier entry is retired. Nothing is
    deleted by that: the files are in _archived/ and a re-run after the
    defect is fixed registers the sample again.
    """
    if step not in SAMPLE_STEPS:
        return False
    # ImportError is not a data error even where a library raises a subclass
    # of a listed type for it; checked first so the order of the list cannot
    # change that.
    if isinstance(exc, (ImportError, OSError, MemoryError)):
        return False
    return isinstance(exc, SAMPLE_DATA_ERRORS)


def process_single(
    nd2_path: Path,
    output_dir: Path,
    threshold: float = 0,
    dilation: int = BUILTIN_PARAMETERS['bg_exclusion_dilation'],
    bg_percentile: float = BUILTIN_PARAMETERS['background_percentile'],
    soma_dilation: int = BUILTIN_PARAMETERS['soma_dilation'],
    sigma_threshold: float = BUILTIN_PARAMETERS['sigma_threshold'],
    min_diameter_um: float = BATCH_NUCLEUS_DIAMETER_UM[0],
    max_diameter_um: float = BATCH_NUCLEUS_DIAMETER_UM[1],
    threshold_fraction: float = BUILTIN_PARAMETERS['threshold_fraction'],
    log_threshold: float = BUILTIN_PARAMETERS['log_threshold'],
) -> dict:
    """
    Process a single ND2 file through detection + colocalization.

    Detection uses threshold method (tuned for sparse retrograde-labeled
    neurons). Colocalization uses background_mean with soma dilation to
    measure cytoplasmic signal.

    Every detection parameter is passed to the detector explicitly, and the
    same values are written to the tracker rows. WHY: the detector used to run
    on its own defaults while the banner, tracker and registry stated other
    values (pixel areas that were never passed, a diameter range it did not
    use), so the record could not be trusted.

    Parameters:
        threshold: threshold drawn on the QC figures (main() passes the sigma).
        soma_dilation: Dilation radius (px) to create soma ROIs around nuclei.
            Retrograde tracer signal is cytoplasmic, so we must measure green
            in the dilated disk around each nucleus.
        sigma_threshold: Number of std devs above background mean for positive
            classification. 0 = background mean IS the threshold.
        min_diameter_um, max_diameter_um: accepted nucleus diameter range.
        threshold_fraction, log_threshold: detector settings (see
            core.detection.detect_with_log_augmentation).

    Returns dict with summary stats, or dict with 'error' key on failure.
    A failed run also carries 'failed_at' (the step) and 'failure': 'sample'
    or 'infrastructure' (see failure_is_the_samples).
    """
    from ..core.io import load_image, extract_channels
    from ..core.detection import detect_with_log_augmentation, diameter_range_to_area_px
    from ..core.colocalization import ColocalizationAnalyzer
    from ..core.visualization import save_all_qc_figures

    # Initialize tracker
    try:
        from ..tracker import SliceTracker
        tracker = SliceTracker()
    except Exception:
        tracker = None
    from ..tracker.schema import COLOC_SIGMA_RECORDED_VERSION

    result = {
        'file': nd2_path.name,
        'path': str(nd2_path),
        'animal': '',
        'region': '',
    }
    det_run_id = None
    coloc_run_id = None
    step = "read"   # the step running now; says whose failure an error is

    try:
        # Parse animal and region from filename (e.g., E01_01_S13_DCN.nd2)
        stem = nd2_path.stem
        parts = stem.split('_')
        if len(parts) >= 3:
            result['animal'] = f"{parts[0]}_{parts[1]}"
            result['region'] = '_'.join(parts[3:]) if len(parts) > 3 else parts[2]

        # Parse sample ID
        sample_id = nd2_path.stem

        # Read the file. Its own step, and not one of SAMPLE_STEPS: nothing
        # the reader raises may retire the sample's earlier registry entry
        # (a cut-off or half-copied file raises a ValueError here, which
        # would otherwise count as a data error of the sample).
        step = "read"
        print(f"  Loading {nd2_path.name}...")
        data, metadata = load_image(nd2_path)

        # From here on the image is in memory: an error is about its content.
        step = "channels"
        red, green = extract_channels(data, red_idx=RED_IDX, green_idx=GREEN_IDX)

        # Physical size conversion
        voxel = metadata.get('voxel_size_um', {})
        pixel_um = voxel.get('x', 1.0) if isinstance(voxel, dict) else voxel[0]
        print(f"  Pixel size: {pixel_um:.2f} um/px")

        # The pixel-area range the detector derives from the diameters, so the
        # tracker row states the areas actually applied to this image.
        min_area_px, max_area_px = diameter_range_to_area_px(
            min_diameter_um, max_diameter_um, pixel_um)

        # Log detection run to tracker
        step = "tracker"
        det_run_id = None
        if tracker:
            det_run_id = tracker.log_detection(
                sample_id=sample_id,
                channel="red",
                model="threshold+log",
                prob_thresh=0.0,
                nms_thresh=0.0,
                min_area=min_area_px,
                max_area=max_area_px,
                input_path=str(nd2_path),
                notes=(f"diameter {min_diameter_um}-{max_diameter_um} um; "
                       f"pixel {pixel_um} um; threshold_fraction={threshold_fraction}; "
                       f"log_threshold={log_threshold}"),
                status="started",
            )

        # Detect using threshold + LoG augmentation with decision tree
        step = "detect"
        print(f"  Detecting nuclei (threshold+LoG, {min_diameter_um}-{max_diameter_um} um "
              f"= {min_area_px}-{max_area_px} px area, {pixel_um:.2f} um/px)...")
        labels, details = detect_with_log_augmentation(
            red, pixel_um=pixel_um,
            min_diameter_um=min_diameter_um,
            max_diameter_um=max_diameter_um,
            threshold_fraction=threshold_fraction,
            log_threshold=log_threshold,
        )
        n_nuclei = details['filtered_count']
        result['n_nuclei'] = n_nuclei
        decision = details.get('decision', '?')
        n_thresh = details.get('n_threshold', 0)
        n_log = details.get('n_log_new', 0)
        print(f"  Found {n_nuclei} nuclei ({decision}: {n_thresh} threshold + {n_log} LoG)")

        # Update tracker with detection results
        step = "tracker"
        if tracker and det_run_id:
            tracker.update_status(det_run_id, status="completed", det_nuclei_found=n_nuclei)

        if n_nuclei == 0:
            result['status'] = 'no_nuclei'
            return result

        # Log colocalization run to tracker
        step = "tracker"
        coloc_run_id = None
        if tracker:
            coloc_run_id = tracker.log_colocalization(
                sample_id=sample_id,
                signal_channel="green",
                background_method="gmm",
                background_percentile=bg_percentile,
                threshold_method="background_mean",
                # The sigma is what background_mean classification uses; the
                # version marker says this row's value can be trusted as one
                # (see tracker.schema.COLOC_SIGMA_RECORDED_VERSION).
                threshold_value=sigma_threshold,
                parent_run=det_run_id,
                input_path=str(nd2_path),
                status="started",
                script_version=COLOC_SIGMA_RECORDED_VERSION,
                # No column holds these two; the note keeps them with the run.
                notes=(f"soma_dilation={soma_dilation}; "
                       f"bg_exclusion_dilation={dilation}"),
            )

        # Colocalize using background_mean (PI's method):
        # Mask out all possible somas from green channel, average remaining
        # tissue = background. Cell is positive if green > background mean.
        step = "colocalize"
        print(f"  Running colocalization (background_mean)...")
        analyzer = ColocalizationAnalyzer(
            background_method='gmm',
            background_percentile=bg_percentile,
        )
        background = analyzer.estimate_background(green, labels, dilation_iterations=dilation)
        tissue_mask = analyzer.estimate_tissue_mask(labels, dilation)
        # Measure cytoplasmic signal via Voronoi territories (non-overlapping)
        # This prevents neighboring cells' green signal from contaminating each other
        measurements = analyzer.measure_cytoplasm_intensities(
            green, labels, expansion_px=soma_dilation)
        classified = analyzer.classify_positive_negative(
            measurements, background,
            method='background_mean',
            signal_image=green,
            nuclei_labels=labels,
            sigma_threshold=sigma_threshold,
        )
        summary = analyzer.get_summary_statistics(classified)

        # Update tracker with colocalization results
        step = "tracker"
        if tracker and coloc_run_id:
            tracker.update_status(
                coloc_run_id,
                status="completed",
                coloc_positive_cells=summary['positive_cells'],
                coloc_negative_cells=summary['negative_cells'],
                coloc_positive_fraction=summary['positive_fraction'],
                coloc_background_value=summary['background_used'],
            )

        result.update({
            'n_positive': summary['positive_cells'],
            'n_negative': summary['negative_cells'],
            'positive_fraction': round(summary['positive_fraction'], 4),
            'mean_fold_change': round(summary['mean_fold_change'], 2),
            'median_fold_change': round(summary['median_fold_change'], 2),
            'background': round(summary['background_used'], 2),
            'status': 'completed',
        })

        # Route output to per-subject pipeline folder
        step = "save"
        parsed = parse_sample_name(sample_id)
        subject_id = f"{parsed['project']}_{parsed['cohort']}_{parsed['subject']}"
        subject_base = DATA_DIR / parsed['project'] / subject_id if DATA_DIR else output_dir
        quant_dir = subject_base / SampleDirs.QUANTIFIED
        quant_dir.mkdir(parents=True, exist_ok=True)
        detect_dir = subject_base / SampleDirs.DETECTED
        detect_dir.mkdir(parents=True, exist_ok=True)

        # Fallback sample_dir for backward compatibility
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save measurements CSV to quantification directory and export dir
        classified.to_csv(quant_dir / f"{sample_id}_measurements.csv", index=False)
        classified.to_csv(sample_dir / "measurements.csv", index=False)

        # Save QC figures to detection directory (primary) and fallback
        print(f"  Saving QC images...")
        save_all_qc_figures(
            output_dir=detect_dir,
            green_channel=green,
            nuclei_labels=labels,
            measurements_df=classified,
            tissue_mask=tissue_mask,
            threshold=threshold,
            background=background,
            prefix=sample_id,
        )
        # Also save to sample_dir for backward compatibility
        save_all_qc_figures(
            output_dir=sample_dir,
            green_channel=green,
            nuclei_labels=labels,
            measurements_df=classified,
            tissue_mask=tissue_mask,
            threshold=threshold,
            background=background,
            prefix=sample_id,
        )

        print(f"  Done: {summary['positive_cells']} pos, {summary['negative_cells']} neg ({summary['positive_fraction']*100:.1f}%)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['status'] = 'error'
        result['error'] = str(e)
        result['failed_at'] = step
        result['failure'] = ('sample' if failure_is_the_samples(step, e)
                             else 'infrastructure')
        print(f"  ERROR ({step}): {e}")

        # Update tracker on error
        if tracker and det_run_id:
            tracker.update_status(det_run_id, status="error")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch colocalization analysis for ENCR ND2 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--project', default=PROJECT,
                        help='Project code, e.g. ENCR (default: MOUSEBRAIN_2D_PROJECT)')
    parser.add_argument('--input', type=Path, default=None,
                        help='Project root directory (default: <2D data dir>/<project>)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Output directory (default: {DEFAULT_OUTPUT})')
    # The four options below default to None = "not typed". Each unset one is
    # filled from a marked-best run, then the approved method, then the
    # built-in default (resolve_parameters). There are no --min-area /
    # --max-area options: the detector never received them, and its size
    # range is the diameter range BATCH_NUCLEUS_DIAMETER_UM.
    parser.add_argument('--threshold', type=float, default=None,
                        help='Sigma threshold: positive if green > bg_mean + N*std; '
                             '0 = bg mean IS the threshold (default: marked-best run, '
                             'else approved method)')
    parser.add_argument('--soma-dilation', type=int, default=None,
                        help='Soma dilation radius (px) for cytoplasmic signal '
                             'measurement (default: approved method)')
    parser.add_argument('--dilation', type=int, default=None,
                        help='Nuclei exclusion dilation iterations for background '
                             '(default: approved method)')
    parser.add_argument('--bg-percentile', type=float, default=None,
                        help='Background percentile fallback (default: marked-best run, '
                             'else approved method)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without processing')
    parser.add_argument('--ignore-tracker', action='store_true',
                        help='Do not use marked-best runs; unset options come from '
                             'the approved method')
    args = parser.parse_args()

    try:
        approved = batch_approved_method()
    except ImportError:
        approved = {}
    except RuntimeError as e:   # an approved-method file is set but unreadable
        parser.error(str(e))

    # Resolve the input BEFORE the banner so the banner reports what will
    # actually be read. WHY: it used to print 'Input: None' because the
    # default root was only filled in after the banner.
    roots = input_roots(args.project, args.input)
    if not roots:
        parser.error("give --project (or MOUSEBRAIN_2D_PROJECT) and set CONNECTOME_ROOT, "
                     "or pass --input explicitly")
    if args.output is None:
        parser.error("output folder unknown: set CONNECTOME_ROOT or pass --output")
    nd2_files, root_used, layout = discover_nd2_files(roots, args.project)
    searched = "; ".join(str(r) for r in roots)

    tuned = {}
    # Set when marked-best runs disagree and the typed numbers settle it, so
    # the banner says that instead of "no run is marked best".
    conflict_settled_by_typing = False
    if not args.ignore_tracker:
        try:
            tuned = get_tuned_parameters(
                project=args.project, sample_ids={f.stem for f in nd2_files})
        except TunedParameterConflict as e:
            # Typing both numbers a marked run could supply settles the
            # question; otherwise the person has to choose.
            if args.threshold is None or args.bg_percentile is None:
                parser.error(
                    f"{e}\nTo choose: in BrainSlice open the sample of each run "
                    "you did NOT mean, select the run in Run History and click "
                    "'Unmark Best'. Or type both --threshold and --bg-percentile, "
                    "or use --ignore-tracker.")
            print(f"[!] {e}\n    Not used: --threshold and --bg-percentile were typed.")
            conflict_settled_by_typing = True
    values, sources = resolve_parameters(
        {'threshold': args.threshold, 'soma_dilation': args.soma_dilation,
         'dilation': args.dilation, 'bg_percentile': args.bg_percentile},
        tuned, approved)
    # A sigma chosen at other dilations than the batch runs at: stop, unless
    # the person typed the batch's dilation (then it was chosen knowingly and
    # the banner repeats the difference). WHY only when a number came from the
    # marked run: the check protects the marked run's sigma (and percentile)
    # from being applied to another measurement. When the person typed both,
    # the marked run supplies nothing and its dilations say nothing about this
    # batch -- the same reason typing both settles disagreeing marked runs.
    uses_marked_run = any(src.startswith("marked-best") for src in sources.values())
    mismatched = dilation_mismatches(tuned, values) if uses_marked_run else []
    untyped = [m for m in mismatched if sources[m[0]] != "command line"]
    if untyped:
        option = {'soma_dilation': '--soma-dilation',
                  'bg_exclusion_dilation': '--dilation'}
        lines = [f"  {k}: marked-best run {', '.join(tuned['source_runs'])} was "
                 f"made at {run}, the batch would run at {mine} [{sources[k]}]"
                 for k, run, mine in untyped]
        parser.error(
            "the marked-best run's sigma was chosen at other dilations than the "
            "batch would use:\n" + "\n".join(lines) + "\nTo choose: type "
            + " and ".join(f"{option[k]} {run}" for k, run, _ in untyped)
            + " to run at the marked run's dilations (or type the batch's own "
            "value to keep it), or mark a run made at the batch's dilations, "
            "or use --ignore-tracker.")
    # What this run will record in the registry, and where it parts from the
    # approved method (those entries are reported stale by check_staleness).
    method_params = batch_method_params(approved, values)
    differs = sorted(k for k in set(method_params) | set(approved)
                     if method_params.get(k) != approved.get(k))

    print("=" * 60)
    print("ENCR Batch Colocalization Analysis")
    print(f"Project:   {args.project}")
    if root_used is not None:
        print(f"Input:     {root_used}  [{layout} layout]")
    else:
        print("Input:     (no ND2 files found)")
    print(f"Searched:  {searched}")
    print(f"Output:    {args.output}")
    d_min, d_max = BATCH_NUCLEUS_DIAMETER_UM
    print(f"Detection: threshold+LoG, nucleus diameter {d_min}-{d_max} um "
          f"(pixel area per image from its pixel size)")
    print(f"           threshold_fraction={values['threshold_fraction']} "
          f"[{sources['threshold_fraction']}], "
          f"log_threshold={values['log_threshold']} [{sources['log_threshold']}]")
    print(f"Background: GMM (percentile={values['background_percentile']}) "
          f"[{sources['background_percentile']}]")
    print(f"Coloc:     background_mean, sigma_threshold={values['sigma_threshold']} "
          f"[{sources['sigma_threshold']}]")
    print(f"Soma dil:  {values['soma_dilation']} px [{sources['soma_dilation']}]")
    print(f"BG excl:   {values['bg_exclusion_dilation']} dilation iterations "
          f"[{sources['bg_exclusion_dilation']}]")
    for k, run, mine in mismatched:
        print(f"[!] {k}: the marked-best run was made at {run}; this batch runs "
              f"at {mine} (typed on the command line)")
    if conflict_settled_by_typing:
        print("Tracker:   marked-best runs disagree; none used, because "
              "--threshold and --bg-percentile were typed")
    elif not tuned and not args.ignore_tracker:
        print("Tracker:   no run of this project is marked best for the batch's "
              "method; mark one in the BrainSlice Run History panel to have the "
              "batch use its sigma and percentile")
    if not approved:
        # Nothing to be stale against (the registry was told this analysis
        # has no approved method); listing every key as "different" would
        # only be noise.
        print(f"Approved:  none for {REGISTRY_ANALYSIS}; entries are registered "
              "with the parameters above, with no approved method to compare")
    elif differs:
        print("[!] NOT the approved method -- these outputs will be registered as "
              "stale against it:")
        for k in differs:
            print(f"      {k}: this run {method_params.get(k)!r}, "
                  f"approved {approved.get(k)!r}")
        # Approved keys the batch has no setting for: no command-line option
        # or code path changes them, so they will differ on every run until
        # the approved method stops naming them.
        uncontrolled = sorted(set(approved) - set(method_params))
        if uncontrolled:
            print(f"    The batch has no setting for {', '.join(uncontrolled)}; "
                  "it never runs with them, so its entries stay stale until the "
                  "approved method leaves them out.")
        # WHY spelled out: marking a run best looks like approving it, but
        # the registry only knows the approved method; without this line a
        # person who marked a run gets stale entries and no reason.
        if any(sources[k].startswith("marked-best") for k in sources):
            print("    A marked-best run chooses the batch's numbers; it does not "
                  "change the approved method.")
        # Printed whatever the numbers came from (typed, a marked run or the
        # defaults): any difference listed above is one a person may copy
        # into the approved-method file, and on a shared registry that edit
        # changes every analysis's approved method. WHY always: the warning
        # used to appear only for marked-run numbers, so its absence after a
        # typed-only run read as "this file is the batch's alone".
        print("    To make these numbers approved, write them into the "
              "approved-method file (see plugin_2d/ALGORITHM_REFERENCE.md, "
              "section 6).")
        if approved_method_is_shared():
            print("    [!] On this registry that file is the approved method "
                  "of EVERY analysis it holds, not only this batch's: "
                  "editing it also changes what the other analyses' entries "
                  "are judged against. Read section 6 before editing it.")
    print("=" * 60)

    print(f"\nFound {len(nd2_files)} ND2 files")

    if not nd2_files:
        # WHY exit 1: a batch that found nothing did no work. Exiting 0 let a
        # wrong or empty root pass unnoticed in scripted and scheduled runs.
        print(f"[!] no ND2 files under {searched}")
        sys.exit(1)

    if args.dry_run:
        for f in nd2_files:
            print(f"  {f.relative_to(root_used)}")
        return

    # Every library the sample steps need, before the first sample: see
    # check_environment for why.
    missing = check_environment()
    if missing:
        print("[!] this environment cannot run the batch; no sample was "
              "attempted and no registry entry was changed:")
        for name, err in missing:
            print(f"      {name}: {err}")
        print("    Activate the environment MouseBrain is installed in (or "
              "install the missing library) and run the batch again.")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Instantiate AnalysisRegistry once for the batch run
    try:
        from mousebrain.analysis_registry import AnalysisRegistry
        # Declared as following the shared approved method: its entries are
        # judged by it unless approved_method.json names this analysis.
        registry = AnalysisRegistry(analysis_name=REGISTRY_ANALYSIS,
                                    follows_approved_method=True)
    except Exception as _reg_init_err:
        registry = None
        print(f"  Registry warning: {_reg_init_err}")

    # Process each file
    all_results = []
    replaced_by_no_nuclei = []   # (sample, earlier n_nuclei, files archived)
    not_replaced = []            # (sample, earlier n_nuclei): files would not move
    retired_by_error = []        # samples whose earlier entry a failure retired
    not_retired = []             # samples whose entry should retire, files would not move
    kept_after_error = []        # failed samples whose earlier entry stays current
    kept_unproven = []           # (sample, step): no other sample got past the step
    sample_failures = []         # (sample, step) of runs that failed on the sample
    registry_errors = []         # (sample, message): a registry call that raised
    retired_not_registered = []  # (sample, earlier n, files archived, message):
                                 # no nuclei; earlier entry retired, new one not written
    start_time = time.time()

    for i, nd2_path in enumerate(nd2_files):
        print(f"\n[{i+1}/{len(nd2_files)}] {nd2_path.name}")
        result = process_single(
            nd2_path,
            args.output,
            threshold=values['sigma_threshold'],
            dilation=values['bg_exclusion_dilation'],
            bg_percentile=values['background_percentile'],
            soma_dilation=values['soma_dilation'],
            sigma_threshold=values['sigma_threshold'],
            min_diameter_um=BATCH_NUCLEUS_DIAMETER_UM[0],
            max_diameter_um=BATCH_NUCLEUS_DIAMETER_UM[1],
            threshold_fraction=values['threshold_fraction'],
            log_threshold=values['log_threshold'],
        )
        all_results.append(result)

        # Register output for successfully completed samples
        if result.get('status') == 'completed' and registry is not None:
            try:
                sample_id = nd2_path.stem
                sample_dir = args.output / sample_id
                output_files = {}
                if sample_dir.exists():
                    for f in sample_dir.iterdir():
                        if f.suffix == '.csv':
                            output_files['measurements'] = str(f)
                        elif f.suffix == '.png' and 'overlay' in f.name:
                            output_files['overlay'] = str(f)
                registry.register_output(
                    sample=sample_id,
                    category="detection",
                    files=output_files,
                    results={
                        'n_nuclei': result.get('n_nuclei', 0),
                        'n_positive': result.get('n_positive', 0),
                        'positive_fraction': result.get('positive_fraction', 0),
                        'background': result.get('background', 0),
                    },
                    # What this run used, not the approved method: see
                    # batch_method_params for why.
                    method_params=method_params,
                    source_files={'nd2': str(nd2_path)},
                )
            except Exception as _reg_err:
                print(f"  Registry warning: {_reg_err}")

        # A finished run that found no nuclei is a result, and it is
        # registered like one: 0 nuclei, no files, the parameters it ran with.
        # WHY: only completed samples used to be registered, so a sample whose
        # re-run found nothing kept its earlier entry -- with the earlier
        # count -- marked current, and the registry reported a number the
        # newest run contradicts. The earlier entry's files are archived first
        # (invalidate moves them to _archived/<timestamp>/ beside them), so no
        # current entry is left pointing at, or next to, another run's files.
        # WHEN THEY WILL NOT MOVE (a file open in another program, a share
        # dropping out): invalidate leaves the earlier entry current with its
        # files in place and returns nothing. Registering 0 nuclei over it
        # then would leave the earlier run's files in the active folder with
        # no entry naming them -- and whatever mirrors that folder would take
        # them for current. So the 0-nuclei result is NOT registered in that
        # case; the earlier entry stays and the summary says what to do.
        elif result.get('status') == 'no_nuclei' and registry is not None:
            sample_id = nd2_path.stem
            earlier = None
            archived = 0
            retired = False   # True once the earlier entry's files have moved
            try:
                earlier = registry.get_entry(sample_id)
                if (earlier and earlier.get('is_current', True)
                        and earlier.get('outputs')):
                    # Counted before the move (afterwards they are gone from
                    # here) and reported only once invalidate says it moved
                    # them all.
                    present = sum(
                        1 for rel in earlier['outputs'].values()
                        if (Path(registry.registry_root) / rel).exists())
                    if not registry.invalidate(sample=sample_id):
                        not_replaced.append(
                            (sample_id,
                             (earlier.get('results') or {}).get('n_nuclei')))
                        continue
                    archived = present
                    retired = True
                registry.register_output(
                    sample=sample_id,
                    category="detection",
                    files={},
                    # No cells, so no positive fraction and no background
                    # measured: null, never a made-up 0.
                    results={
                        'n_nuclei': 0,
                        'n_positive': 0,
                        'positive_fraction': None,
                        'background': None,
                    },
                    method_params=method_params,
                    source_files={'nd2': str(nd2_path)},
                )
                if earlier:
                    replaced_by_no_nuclei.append(
                        (sample_id,
                         (earlier.get('results') or {}).get('n_nuclei'),
                         archived))
            except Exception as _reg_err:
                print(f"  Registry warning: {_reg_err}")
                if retired:
                    # The earlier entry is already retired and its files are
                    # in _archived/, and the 0-nuclei result did not get
                    # registered: the sample has NO current entry. Reported
                    # as exactly that -- the general "entry is as it was"
                    # line would be false here.
                    retired_not_registered.append(
                        (sample_id,
                         (earlier.get('results') or {}).get('n_nuclei'),
                         archived, str(_reg_err)))
                else:
                    registry_errors.append((sample_id, str(_reg_err)))

        # A run that FAILED produced no result. Whether its earlier entry is
        # retired is decided after the loop (below), once it is known which
        # steps the other samples of this run got past.
        elif result.get('status') == 'error' and registry is not None:
            if result.get('failure') == 'sample':
                sample_failures.append((nd2_path.stem, result.get('failed_at')))
            else:
                try:
                    earlier = registry.get_entry(nd2_path.stem)
                    if earlier and earlier.get('is_current', True):
                        kept_after_error.append(nd2_path.stem)
                except Exception as _reg_err:
                    print(f"  Registry warning: {_reg_err}")
                    registry_errors.append((nd2_path.stem, str(_reg_err)))

    # A failed sample's earlier entry is retired (moved to _archived/,
    # recorded as no longer current) when the failure is the sample's -- the
    # newest attempt on it failed, so the older count must not stand as
    # current -- AND another sample of this run got past the step that failed
    # (STEP_EVIDENCE). WHY the second condition: when no sample got past a
    # step, its failures say more about the software (a code defect, a broken
    # library) than about each sample, and retiring entries on that evidence
    # would drop good results for no reason. WHY per step: a sample with no
    # nuclei never reaches the measuring step, so it cannot vouch for it; if
    # every sample that has nuclei fails while measuring, that is the
    # software, however many no-nuclei samples the run also had.
    finished = {r.get('status') for r in all_results}
    for sample_id, failed_step in sample_failures:
        try:
            earlier = registry.get_entry(sample_id)
            if not (earlier and earlier.get('is_current', True)):
                continue   # nothing current to retire
            if not finished.intersection(STEP_EVIDENCE.get(failed_step, ())):
                kept_unproven.append((sample_id, failed_step))
            elif registry.invalidate(sample=sample_id):
                retired_by_error.append(sample_id)
            else:
                # invalidate could not move a file: the entry is still
                # current. Said in the summary, not only in invalidate's own
                # line among the per-sample output.
                not_retired.append(sample_id)
        except Exception as _reg_err:
            print(f"  Registry warning: {_reg_err}")
            registry_errors.append((sample_id, str(_reg_err)))

    # Write summary CSV
    summary_path = args.output / "summary.csv"
    fieldnames = [
        'file', 'animal', 'region', 'n_nuclei', 'n_positive', 'n_negative',
        'positive_fraction', 'mean_fold_change', 'median_fold_change',
        'background', 'status', 'error', 'path',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    elapsed = time.time() - start_time
    completed = sum(1 for r in all_results if r.get('status') == 'completed')
    errored = sum(1 for r in all_results if r.get('status') == 'error')
    no_nuclei = sum(1 for r in all_results if r.get('status') == 'no_nuclei')

    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE")
    print(f"  Processed: {len(all_results)} files in {elapsed/60:.1f} min")
    print(f"  Completed: {completed}")
    print(f"  No nuclei: {no_nuclei}")
    for sample_id, earlier_n, archived in replaced_by_no_nuclei:
        what = (f"{archived} file(s) moved to _archived/" if archived
                else "no files to archive")
        print(f"    {sample_id}: earlier registry entry ({earlier_n} nuclei) "
              f"replaced by this run's 0 nuclei; {what}")
    for sample_id, earlier_n in not_replaced:
        print(f"    [!] {sample_id}: this run found 0 nuclei, but the earlier "
              f"registry entry ({earlier_n} nuclei) is KEPT CURRENT: its files "
              "could not be moved to _archived/ (in use?). Close whatever has "
              "them open and run again.")
    for sample_id, earlier_n, archived, message in retired_not_registered:
        print(f"    [!] {sample_id}: earlier registry entry ({earlier_n} "
              f"nuclei) was retired, {archived} file(s) moved to _archived/, "
              "but this run's 0-nuclei result could NOT be registered: "
              f"{message}. The sample has no current entry now; run the "
              "batch again.")
    print(f"  Errors:    {errored}")
    for r in all_results:
        if r.get('status') == 'error':
            print(f"    {r.get('file', '')}: [{r.get('failure', '')} failure "
                  f"while {r.get('failed_at', '')}] {r.get('error', '')}")
    for sample_id in retired_by_error:
        print(f"    {sample_id}: earlier registry entry retired (no longer "
              "current, its files moved to _archived/) -- the sample's own "
              "re-run failed")
    for sample_id in not_retired:
        print(f"    [!] {sample_id}: the sample's own re-run failed, but its "
              "earlier registry entry could not be retired and is STILL "
              "CURRENT: its files could not be moved to _archived/ (in use?). "
              "Close whatever has them open and run again.")
    for sample_id, failed_step in kept_unproven:
        print(f"    {sample_id}: earlier registry entry kept current -- no "
              f"other sample of this run got past the '{failed_step}' step, so "
              "the failure may be the software's, not the sample's; fix the "
              "cause and run again")
    for sample_id in kept_after_error:
        print(f"    {sample_id}: earlier registry entry kept current -- the "
              "failure was not shown to be the sample's; fix it and run again")
    for sample_id, message in registry_errors:
        print(f"    [!] {sample_id}: the registry could not be read or updated "
              f"for this sample ({message}); its entry is as it was before "
              "this run unless a line above says otherwise")
    print(f"  Summary:   {summary_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
