#!/usr/bin/env python3
"""
tracker.py (mousebrain package)

Calibration run tracking system for the BrainGlobe pipeline.

This module is now part of the mousebrain package and tracks CALIBRATION runs
(tuning/optimization) - detection parameter tests, model training, classification
tests, etc. These are iterative experiments to find optimal settings.

For PRODUCTION results (final cell counts per region), see region_counts.csv
which is updated by 6_count_regions.py.

Two-file architecture:
  - calibration_runs.csv: This tracker - all tuning runs with parameters
  - region_counts.csv: Production results - one row per brain, wide format

This module is imported by the run_* wrapper scripts and the napari tuning widget.

Usage as a module:
    from mousebrain.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    exp_id = tracker.log_detection(brain="001_CNT_01_01_1p625x_z4", ...)
    tracker.update_status(exp_id, status="completed", cells_found=1234)
    tracker.rate_experiment(exp_id, rating=4, notes="Good results")

Usage as CLI:
    python -m mousebrain.tracker                    # Show recent runs
    python -m mousebrain.tracker --search "001"     # Search by brain
    python -m mousebrain.tracker --stats            # Show statistics
"""

import csv
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import hashlib
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default location for the calibration runs CSV
from mousebrain.config import CALIBRATION_RUNS_CSV as DEFAULT_TRACKER_PATH, parse_brain_name
from mousebrain.file_lock import exclusive_file_lock

# Experiment types
EXP_TYPES = ["detection", "training", "classification", "selective_classification", "registration", "counts", "prefilter"]

# CSV columns (order matters for readability)
CSV_COLUMNS = [
    # Identity
    "exp_id",
    "exp_type",
    "brain",
    "created_at",

    # Data Hierarchy (auto-filled from brain name)
    "brain_id",       # e.g., "001" - individual tissue sample
    "subject",        # e.g., "CNT_01_01" - the animal/unit of study
    "cohort",         # e.g., "CNT_01" - group processed together
    "project_code",   # e.g., "CNT" - umbrella project code
    "project_name",   # e.g., "Connectome" - human-readable project name

    # Status. Written by the code: started, completed, failed, approved
    # (registration backfill), data_exported (training export). A run whose
    # process stops without a result is set to failed (finish_or_fail).
    "status",
    "duration_seconds",

    # Common parameters
    "atlas",
    "voxel_z",
    "voxel_xy",

    # Detection parameters
    "det_preset",
    "det_ball_xy",
    "det_ball_z",
    "det_soma_diameter",
    "det_threshold",
    "det_cells_found",
    "det_scope",        # "full" or "partial" - whether full brain or subset was analyzed
    "det_z_start",      # Starting Z slice (for partial detection)
    "det_z_end",        # Ending Z slice (for partial detection)
    "det_z_center",     # Center Z slice (for partial detection)

    # Training parameters
    "train_epochs",
    "train_learning_rate",
    "train_augment",
    "train_pretrained",
    "train_loss",
    "train_accuracy",

    # Classification parameters
    "class_model_path",
    "class_cube_size",
    "class_batch_size",
    "class_cells_found",
    "class_rejected",

    # Pre-filter parameters (atlas-based candidate filtering)
    "prefilter_total",            # Total candidates before filtering
    "prefilter_interior",         # Candidates inside brain
    "prefilter_outside",          # Candidates at region_id=0 (outside brain)
    "prefilter_suspicious",       # Candidates in suspicious regions
    "prefilter_flag_suspicious",  # Whether suspicious filtering was enabled
    "prefilter_tracing_type",     # descending/ascending/unknown

    # Counting parameters
    "count_regions",
    "count_total_cells",
    "count_output_csv",

    # Registration parameters
    "reg_atlas",            # Atlas used (e.g., "allen_mouse_10um")
    "reg_orientation",      # Brain orientation (e.g., "prs")
    "reg_voxel_z",          # Voxel size Z in microns
    "reg_voxel_xy",         # Voxel size XY in microns
    "reg_affine_n_steps",   # Number of affine registration steps
    "reg_freeform_n_steps", # Number of freeform registration steps
    "reg_approved",         # True if user approved the registration
    "reg_approved_at",      # ISO timestamp of approval
    "reg_qc_images",        # Paths to QC images (comma-separated)

    # Paths
    "input_path",
    "output_path",
    "model_path",

    # Linkage
    "parent_experiment",  # Links classification to detection, etc.

    # User feedback
    "rating",  # 1-5 stars
    "notes",
    "tags",

    # Metadata
    "script_version",
    "hostname",

    # QC and documentation
    "qc_image_path",  # Path to auto-generated QC image

    # Comparison and archival tracking (added for tuning widget workflow)
    "marked_best",      # True if user marked this as best for its brain
    "archived_by",      # exp_id that replaced/superseded this experiment
    "archived_at",      # ISO timestamp when archived
    "diff_vs_best",     # Total difference points when compared to best
    "diff_gained",      # Points gained vs reference
    "diff_lost",        # Points lost vs reference
    "comparison_ref",   # exp_id this was compared against

    # Imaging paradigm (for auto-applying settings to similar brains)
    "imaging_paradigm",  # Extracted from brain name, e.g., "1p625x_z4"
    "paradigm_best",     # True if this is the best run for its paradigm (not just brain)

    # The rest of the detection settings (added 2026-10-29). A detection method
    # is eight settings, not the four above: before these columns existed, a
    # Paradigm Best run carried only half its method, and step 4 filled the rest
    # with cellfinder's defaults -- a different method (re-detecting part of a
    # settled run that way found only about a third of its cells at the same
    # positions; with these four restored, all of them). They sit at the END so a
    # file written before them gains them without any existing column moving.
    "det_ball_overlap",  # ball_overlap_fraction
    "det_soma_spread",   # soma_spread_factor
    "det_log_sigma",     # log_sigma_size
    "det_max_cluster",   # max_cluster_size
]

# Run types that are a classification METHOD. One imaging setup has at most
# one Paradigm Best among them (mark_paradigm_best), and step 5 runs that one
# (get_paradigm_classification): 'selective_classification' trusts candidates
# in deep regions and classifies the rest; 'classification' classifies all.
CLASSIFICATION_TYPES = ("classification", "selective_classification")

# The eight settings that make up a detection method: tracker column -> the
# cellfinder argument it feeds. get_paradigm_detection_settings() returns them
# under the short names on the left of each pair.
DETECTION_METHOD_COLUMNS = {
    'ball_xy': 'det_ball_xy',
    'ball_z': 'det_ball_z',
    'soma_diameter': 'det_soma_diameter',
    'threshold': 'det_threshold',
    'ball_overlap': 'det_ball_overlap',
    'soma_spread': 'det_soma_spread',
    'log_sigma': 'det_log_sigma',
    'max_cluster': 'det_max_cluster',
}


# How the CSV is written (see ExperimentTracker._csv_lock and _rewrite).
#
# WHY A LOCK, A TEMPORARY FILE, AND THE FILE'S OWN COLUMN ORDER
# The tracker is one CSV on the shared drive, written by the pipeline scripts,
# the napari widget and the training script -- on several machines at once.
# Every status change rewrites the whole file. Before this, that rewrite:
#   * wrote in place, so a crash part-way left a truncated tracker;
#   * had no lock, so two machines updating at once lost one update;
#   * wrote only the columns THIS code version knows, in this version's order,
#     so the first status change made by an older copy deleted every newer
#     column from every row.
# Now each write holds an exclusive lock file beside the CSV, writes a
# temporary file in the file's own column order (keeping columns this version
# does not know), and swaps it in with os.replace.
#
# The lock is mousebrain.file_lock (see ExperimentTracker._csv_lock): a file
# created with O_EXCL, which the file server grants to exactly one machine. A
# holder that crashed leaves it behind, so a lock untouched for
# LOCK_STALE_SECONDS -- measured by the FILE SERVER's clock, not this
# machine's, because the machines' clocks differ -- is taken over. A live
# holder refreshes its lock in the background, so a rewrite that is slow on a
# slow share is never mistaken for a crashed one.
LOCK_STALE_SECONDS = 120
LOCK_WAIT_SECONDS = 300
REPLACE_RETRY_SECONDS = 30  # a reader on Windows can hold the file open briefly


def same_brain_name(logged: Optional[str], brain: str) -> bool:
    """True when the tracker's `brain` column names exactly this brain.

    Scripts log "<mouse>/<brain>", the napari widget logs "<brain>", so a row
    matches when its name IS the brain or ENDS in "/<brain>" (a backslash is
    taken as the same separator). Case is ignored, as folder names on the
    file server are. A name that merely CONTAINS the brain is a different
    brain and never matches.
    """
    name = (logged or '').strip().replace(chr(92), '/').lower()
    want = (brain or '').strip().replace(chr(92), '/').lower()
    if not want:
        return False
    return name == want or name.endswith('/' + want)


def _bare_brain_name(logged: Optional[str]) -> str:
    """The brain's own name from a tracker `brain` value: "<mouse>/<brain>"
    (scripts) and "<brain>" (napari widget) both give "<brain>". Passed to
    same_brain_name, it matches the rows of this brain in either form."""
    name = (logged or '').strip().replace(chr(92), '/').rstrip('/')
    return name.rsplit('/', 1)[-1]


# =============================================================================
# EXPERIMENT TRACKER CLASS
# =============================================================================

class ExperimentTracker:
    """
    Central experiment tracking for BrainGlobe pipeline.

    All experiments are logged to a single CSV file with consistent columns.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize tracker.

        Args:
            csv_path: Path to experiments CSV. Uses default if not specified.
        """
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_TRACKER_PATH
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create the CSV with headers if it does not exist.

        REFUSES to create a new one at an unconfigured location. The tracker is the
        record of every calibration run ever done, and a second empty copy is the
        worst possible failure here: it looks exactly like a working tracker, later
        runs append to it, and the real file is
        quietly orphaned while everything appears to work. Reading an unconfigured
        path just finds nothing, which is obvious; writing one is not.

        This guards against an unconfigured run creating a fresh empty tracker
        in the working directory (see mousebrain.config._get_root_path). The path
        sentinel is now harmless on its own; this is the second line of defence,
        because the tracker is the file that must never be silently forked.
        """
        if not self.csv_path.exists():
            from . import config as _cfg
            using_default = self.csv_path == DEFAULT_TRACKER_PATH
            if using_default and not _cfg.is_configured():
                raise RuntimeError(
                    "refusing to create a new calibration tracker at %s. "
                    "mousebrain is not configured, so that is not where this lab's "
                    "tracker lives, and a second empty one would look real while the "
                    "actual record of every calibration run was orphaned. %s"
                    % (self.csv_path, _cfg.UNCONFIGURED_MESSAGE))
            # The default tracker is created only INSIDE an existing
            # 2_Data_Summary folder, never together with it. A configured root
            # whose pipeline folder is not visible right now (a share that is
            # half connected) must not get a fresh, empty tracker that later
            # runs append to while the real one sits unseen. A path given on
            # purpose (csv_path=...) is not a guess, so its folders are made.
            if using_default and not self.csv_path.parent.is_dir():
                raise RuntimeError(
                    "refusing to create a new calibration tracker: its folder %s "
                    "does not exist. On a working installation that folder is "
                    "always there, so either the data drive is not connected "
                    "right now (reconnect it and try again) or CONNECTOME_ROOT "
                    "points at the wrong place (check with: mousebrain --paths). "
                    "Only on a brand-new installation, create that one folder by "
                    "hand; the tracker then starts in it." % self.csv_path.parent)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
        else:
            self._add_missing_columns()

    # -----------------------------------------------------------------
    # Safe writing (see the note above the class)
    # -----------------------------------------------------------------
    @property
    def _lock_path(self) -> Path:
        return self.csv_path.with_name(self.csv_path.name + ".lock")

    @contextmanager
    def _csv_lock(self):
        """Hold the tracker's write lock for the duration of the block.

        The lock is mousebrain.file_lock, the same lock the analysis registry
        and the 2D slice tracker use. WHY not a copy of its own: the tracker
        used to keep an older copy of that algorithm, which (1) never
        refreshed a live lock, so a rewrite that took longer than
        LOCK_STALE_SECONDS on a slow share was judged abandoned and taken over
        -- two writers at once, and the second rewrite silently dropped the
        first one's rows; (2) took a lock over by moving it aside and putting
        it back when it turned out to be fresh, which leaves the lock's name
        free for a moment so a third writer can take it; and (3) released the
        lock with one delete, which Windows refuses while another process has
        the lock file open for a moment to read who holds it -- the lock then
        stayed behind and every writer waited it out. file_lock refreshes a
        live lock in the background, takes over an abandoned one only while
        holding a one-at-a-time breaker, and retries the release. The limits
        are read here, at call time, so a test or a caller can change them."""
        with exclusive_file_lock(self.csv_path, "Tracker",
                                 stale_seconds=LOCK_STALE_SECONDS,
                                 wait_seconds=LOCK_WAIT_SECONDS):
            yield

    def _replace(self, tmp: Path) -> None:
        deadline = time.monotonic() + REPLACE_RETRY_SECONDS
        while True:
            try:
                os.replace(tmp, self.csv_path)
                return
            except PermissionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)

    def _rewrite(self, mutate):
        """Read every row, let `mutate(rows)` change them, write them back.

        Under the lock; in the file's own column order, keeping columns this
        version does not know and adding any column it does know at the end;
        through a temporary file. Returns what `mutate` returns.
        """
        with self._csv_lock():
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                rows = list(reader)
            if any(None in row for row in rows):
                raise RuntimeError(
                    f"{self.csv_path}: some rows have more values than the header has "
                    f"columns, so the file cannot be rewritten without losing them. "
                    f"Repair it first.")
            result = mutate(rows)
            columns = header + [c for c in CSV_COLUMNS if c not in header]
            extra = [k for row in rows for k in row if k not in columns]
            columns += list(dict.fromkeys(extra))
            tmp = self.csv_path.with_name(self.csv_path.name + ".tmp-%d" % os.getpid())
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({c: row.get(c, "") for c in columns})
            self._replace(tmp)
            return result

    def _add_missing_columns(self):
        """Give a tracker written before the newest columns those columns.

        Rows are APPENDED with the current column list, so appending to a file
        whose header is shorter would shift every later row against its header
        -- values landing under the wrong names, which has happened to a tracker
        before and needed a repair. The header is brought
        up to date, every existing value kept under its own name, before any
        row is written. Written to a temporary file and swapped in, so a failure
        part-way leaves the original untouched.
        """
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            header = next(csv.reader(f), None) or []
        if all(col in header for col in CSV_COLUMNS):
            return
        # Existing columns keep their order (including any this version no
        # longer knows); new ones go at the end -- which is what _rewrite does.
        self._rewrite(lambda rows: None)

    def _generate_exp_id(self, exp_type: str, brain: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Short hash for uniqueness. It includes a random part: two runs logged
        # by one process within the same second (prefilter + classification in
        # step 5, say) used to get the SAME id, and every later update to
        # either then changed only the first.
        hash_input = f"{exp_type}_{brain}_{timestamp}_{os.getpid()}_{uuid.uuid4().hex}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{exp_type[:3]}_{timestamp}_{short_hash}"

    def _get_hostname(self) -> str:
        """Get current hostname."""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"

    def _add_hierarchy_fields(self, row: Dict[str, Any], brain: str) -> Dict[str, Any]:
        """
        Auto-populate hierarchy fields from brain name.

        Parses brain names like "001_CNT_01_01_1p625x_z4" into:
          - brain_id: 001
          - subject: CNT_01_01
          - cohort: CNT_01
          - project_code: CNT
          - project_name: Connectome
          - imaging_paradigm: 1p625x_z4

        See 2_Data_Summary/DATA_HIERARCHY.md for full documentation.
        """
        parsed = parse_brain_name(brain)

        row["brain_id"] = parsed.get("brain_id", "")
        row["subject"] = parsed.get("subject_full", "")
        row["cohort"] = parsed.get("cohort_full", "")
        row["project_code"] = parsed.get("project_code", "")
        row["project_name"] = parsed.get("project_name", "")
        row["imaging_paradigm"] = parsed.get("imaging_params", "")

        return row

    def _write_row(self, row: Dict[str, Any]):
        """Append a row to the CSV."""
        # Ensure all columns exist with empty string as default
        full_row = {col: row.get(col, "") for col in CSV_COLUMNS}

        # Append in the FILE's column order, not this code's: the two differ
        # whenever the file has a column this version does not know, and writing
        # in the wrong order would put values under the wrong names. Under the
        # lock, so an append never lands inside another machine's rewrite.
        with self._csv_lock():
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                header = next(csv.reader(f), None) or CSV_COLUMNS
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                writer.writerow(full_row)

    def _read_all(self) -> List[Dict[str, str]]:
        """Read all experiments from CSV."""
        if not self.csv_path.exists():
            return []

        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _update_row(self, exp_id: str, updates: Dict[str, Any]) -> bool:
        """Update a specific row by exp_id. One locked rewrite (see _rewrite):
        columns this code version does not know are kept, never dropped."""
        return self._update_rows({exp_id: updates})

    def _update_rows(self, updates_by_id: Dict[str, Dict[str, Any]]) -> bool:
        """Apply several rows' updates in ONE locked rewrite, so a change
        that must touch more than one row (moving a 'best' mark) can never be
        left half done. True when the first id given was found."""
        def mutate(rows):
            found = set()
            for row in rows:
                upd = updates_by_id.get(row.get('exp_id'))
                if upd is not None and row['exp_id'] not in found:
                    row.update({k: str(v) if v is not None else "" for k, v in upd.items()})
                    found.add(row['exp_id'])
            return found
        found = self._rewrite(mutate)
        first = next(iter(updates_by_id), None)
        return first in found

    @contextmanager
    def finish_or_fail(self, exp_id: Optional[str]):
        """Wrap a run's work: if the block ends while the run is still
        'started' -- an exception, Ctrl-C, sys.exit, or a path that forgot to
        record a result -- the run is set to 'failed' with the reason in its
        notes. Rows stuck at 'started' forever used to be indistinguishable
        from runs still going. (A process killed outright -- task end, power
        loss -- runs no code at all and still leaves 'started'.)"""
        try:
            yield
        except BaseException as e:
            self._fail_if_unfinished(exp_id, "stopped before finishing: %s: %s"
                                     % (type(e).__name__, e))
            raise
        else:
            self._fail_if_unfinished(exp_id, "ended without recording a result")

    def _fail_if_unfinished(self, exp_id: Optional[str], why: str) -> None:
        if not exp_id:
            return
        self.fail_unfinished([exp_id], why)

    def fail_unfinished(self, exp_ids, why: str) -> List[str]:
        """Set each of these runs that is still 'started' to 'failed', with
        `why` added to its notes, in ONE locked rewrite. A run that already
        finished (completed, failed, ...) is left exactly as it is. Returns
        the ids that were changed. Never raises: it is called on the way out
        of something else (an error, the napari window closing), and must
        not hide that.

        WHY: a 'started' row that will never finish looks exactly like a run
        still going, so nobody can tell a crashed run from a running one.
        The code that started a run marks it failed whenever it can see the
        run will not finish (finish_or_fail; the tuning widget on close)."""
        wanted = {e for e in (exp_ids or []) if e}
        if not wanted:
            return []
        try:
            def mutate(rows):
                changed = []
                for row in rows:
                    if row.get('exp_id') in wanted and row.get('status') == 'started':
                        row['status'] = 'failed'
                        row['notes'] = "; ".join(filter(None, [row.get('notes', ''), why]))
                        changed.append(row['exp_id'])
                return changed
            changed = self._rewrite(mutate) or []
            for exp_id in changed:
                print("[Tracker] Run %s marked failed: %s" % (exp_id, why))
            return changed
        except Exception as e:  # never hide the original error behind this one
            print("[Tracker] [!] Could not mark run(s) %s failed: %s"
                  % (", ".join(sorted(wanted)), e))
            return []

    def unfinished(self, older_than_hours: float = 0) -> List[Dict[str, str]]:
        """Runs still at 'started' that were begun more than
        `older_than_hours` ago (0: every one), oldest first.

        A run is left at 'started' only when its process was killed outright
        (Task Manager, a power cut) or by code from before runs were marked
        failed on the way out -- or it is genuinely still running. This lists
        them so a person can tell which is which (util_experiments.py
        unfinished). A row whose start time cannot be read is always listed:
        its age is unknown, and leaving it out would hide it."""
        cutoff = None
        if older_than_hours and older_than_hours > 0:
            cutoff = datetime.now().timestamp() - older_than_hours * 3600
        out = []
        for row in self._read_all():
            if row.get('status') != 'started':
                continue
            if cutoff is not None:
                try:
                    started = datetime.fromisoformat(
                        (row.get('created_at') or '').strip()).timestamp()
                except ValueError:
                    started = None
                if started is not None and started > cutoff:
                    continue
            out.append(row)
        out.sort(key=lambda r: r.get('created_at') or '')
        return out

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def log_detection(
        self,
        brain: str,
        preset: Optional[str] = None,
        ball_xy: Optional[float] = None,
        ball_z: Optional[float] = None,
        soma_diameter: Optional[float] = None,
        threshold: Optional[int] = None,
        voxel_z: Optional[float] = None,
        voxel_xy: Optional[float] = None,
        atlas: str = "allen_mouse_10um",
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
        # Detection scope parameters (for partial vs full detection)
        det_scope: Optional[str] = None,      # "full" or "partial"
        det_z_start: Optional[int] = None,    # Starting Z slice (for partial)
        det_z_end: Optional[int] = None,      # Ending Z slice (for partial)
        det_z_center: Optional[int] = None,   # Center Z slice (for partial)
        # The other half of the method (see DETECTION_METHOD_COLUMNS)
        ball_overlap: Optional[float] = None,
        soma_spread: Optional[float] = None,
        log_sigma: Optional[float] = None,
        max_cluster: Optional[int] = None,
    ) -> str:
        """
        Log a cell detection experiment.

        Hierarchy fields (brain_id, subject, cohort, project_code, project_name)
        are automatically populated from the brain name.

        Args:
            brain: Brain name
            preset: Detection preset used
            ball_xy/ball_z: Ball filter sizes
            soma_diameter: Expected cell diameter
            threshold: Detection threshold
            voxel_z/voxel_xy: Voxel sizes in microns
            atlas: Atlas used for registration
            input_path: Path to input data
            output_path: Path where results are saved
            notes: User notes
            tags: Comma-separated tags
            status: Run status
            script_version: Version of detection script
            det_scope: "full" for whole brain, "partial" for subset of Z slices
            det_z_start: Starting Z slice for partial detection
            det_z_end: Ending Z slice for partial detection
            det_z_center: Center Z slice for partial detection

        Returns:
            exp_id: Unique experiment identifier
        """
        exp_id = self._generate_exp_id("detection", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "detection",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "atlas": atlas,
            "voxel_z": voxel_z,
            "voxel_xy": voxel_xy,
            "det_preset": preset,
            "det_ball_xy": ball_xy,
            "det_ball_z": ball_z,
            "det_soma_diameter": soma_diameter,
            "det_threshold": threshold,
            "det_ball_overlap": ball_overlap,
            "det_soma_spread": soma_spread,
            "det_log_sigma": log_sigma,
            "det_max_cluster": max_cluster,
            "det_scope": det_scope,
            "det_z_start": det_z_start,
            "det_z_end": det_z_end,
            "det_z_center": det_z_center,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "tags": tags,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        # Auto-fill hierarchy from brain name
        row = self._add_hierarchy_fields(row, brain)

        self._write_row(row)
        return exp_id

    def log_training(
        self,
        brain: str,
        epochs: int,
        learning_rate: float = 0.0001,
        augment: bool = True,
        pretrained: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a training experiment. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("training", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "training",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "train_epochs": epochs,
            "train_learning_rate": learning_rate,
            "train_augment": augment,
            "train_pretrained": pretrained,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_classification(
        self,
        brain: str,
        model_path: str,
        candidates_path: Optional[str] = None,
        cube_size: int = 50,
        batch_size: int = 32,
        n_free_cpus: int = 2,
        output_path: Optional[str] = None,
        parent_experiment: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a classification experiment. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("classification", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "classification",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "class_model_path": model_path,
            "class_cube_size": cube_size,
            "class_batch_size": batch_size,
            "input_path": candidates_path,
            "output_path": output_path,
            "model_path": model_path,
            "parent_experiment": parent_experiment,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_prefilter(
        self,
        brain: str,
        total: int = 0,
        interior: int = 0,
        outside: int = 0,
        suspicious: int = 0,
        flag_suspicious: bool = False,
        tracing_type: str = "descending",
        parent_experiment: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log an atlas pre-filter run. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("prefilter", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "prefilter",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "prefilter_total": total,
            "prefilter_interior": interior,
            "prefilter_outside": outside,
            "prefilter_suspicious": suspicious,
            "prefilter_flag_suspicious": flag_suspicious,
            "prefilter_tracing_type": tracing_type,
            "parent_experiment": parent_experiment,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_selective_classification(
        self,
        brain: str,
        model_path: str,
        total_candidates: int = 0,
        trust_count: int = 0,
        classify_count: int = 0,
        classified_cells: int = 0,
        classified_rejected: int = 0,
        final_cells: int = 0,
        tracing_type: str = "descending",
        candidates_path: Optional[str] = None,
        output_path: Optional[str] = None,
        parent_experiment: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
        cube_size: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> str:
        """Log a selective classification run. Hierarchy fields auto-populated from brain name.

        cube_size / batch_size: recorded so the row is the WHOLE method. A row
        without them cannot be a settled method (get_paradigm_classification
        lists them as missing), which is what happened to the runs made before
        2026-11-19: they recorded the model and the tracing type only.
        """
        exp_id = self._generate_exp_id("selective_classification", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "selective_classification",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "class_model_path": model_path,
            "class_cells_found": final_cells,
            "class_rejected": classified_rejected,
            "class_cube_size": cube_size if cube_size is not None else "",
            "class_batch_size": batch_size if batch_size is not None else "",
            "prefilter_total": total_candidates,
            "prefilter_interior": trust_count,
            "prefilter_suspicious": classify_count,
            "prefilter_tracing_type": tracing_type,
            "input_path": candidates_path,
            "output_path": output_path,
            "model_path": model_path,
            "parent_experiment": parent_experiment,
            # The counts are known only at the end, and this is called at the
            # start: the old default wrote "trust=0 classify=0 ..." into every
            # row, which read as a result. The caller writes them when it ends.
            "notes": notes or "",
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_counts(
        self,
        brain: str,
        cells_path: str,
        atlas: str = "allen_mouse_10um",
        output_path: Optional[str] = None,
        parent_experiment: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a regional counting experiment. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("counts", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "counts",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "atlas": atlas,
            "input_path": cells_path,
            "output_path": output_path,
            "parent_experiment": parent_experiment,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_registration(
        self,
        brain: str,
        atlas: str = "allen_mouse_10um",
        orientation: str = "prs",
        voxel_z: Optional[float] = None,
        voxel_xy: Optional[float] = None,
        affine_n_steps: int = 6,
        freeform_n_steps: int = 6,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """
        Log a brain registration experiment.

        Tracks atlas registration using BrainGlobe's brainreg.

        Args:
            brain: Brain name
            atlas: Atlas used for registration (e.g., "allen_mouse_10um")
            orientation: Brain orientation string (e.g., "prs" for posterior-right-superior)
            voxel_z: Voxel size in Z (microns)
            voxel_xy: Voxel size in XY (microns)
            affine_n_steps: Number of affine registration steps
            freeform_n_steps: Number of freeform registration steps
            input_path: Path to input data (cropped brain)
            output_path: Path to registration output folder
            notes: User notes
            status: Run status
            script_version: Version of registration script

        Returns:
            exp_id: Unique experiment identifier
        """
        exp_id = self._generate_exp_id("registration", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "registration",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "atlas": atlas,
            "reg_atlas": atlas,
            "reg_orientation": orientation,
            "reg_voxel_z": voxel_z,
            "reg_voxel_xy": voxel_xy,
            "voxel_z": voxel_z,
            "voxel_xy": voxel_xy,
            "reg_affine_n_steps": affine_n_steps,
            "reg_freeform_n_steps": freeform_n_steps,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def approve_registration(self, exp_id: str) -> bool:
        """
        Mark a registration as approved by the user.

        Called after user visually confirms registration quality.

        Args:
            exp_id: The registration experiment ID to approve

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'reg_approved': 'True',
            'reg_approved_at': datetime.now().isoformat(),
            'status': 'approved',
        })

    def add_registration_qc_images(self, exp_id: str, image_paths: List[str]) -> bool:
        """
        Add QC image paths to a registration experiment.

        Args:
            exp_id: The registration experiment ID
            image_paths: List of paths to QC images

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'reg_qc_images': ','.join(str(p) for p in image_paths),
        })

    # =========================================================================
    # UPDATE METHODS
    # =========================================================================

    def update_status(
        self,
        exp_id: str,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        expect_status: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Update experiment status and results.

        expect_status: apply the update only while the run's status is still
            this (e.g. 'started'), checked and written in one locked rewrite.
            If the run has meanwhile been set to something else, its status
            and results are left as they are, a line saying what arrived is
            added to its notes, and False is returned.

            WHY: the Setup & Tuning panel marks a run it started 'failed'
            when the panel is closed while the run is still going (see
            fail_unfinished). The run's thread can still finish afterwards;
            its result used to turn that row into 'completed' while its notes
            said it had been cut off -- a row contradicting itself. A run's
            end is recorded once: a late result is kept in the notes (so the
            cells it found and the file it wrote can still be found) but never
            overwrites the end already recorded.

        Common kwargs:
            det_cells_found: int - Detection cell count
            class_cells_found: int - Classification cell count
            class_rejected: int - Rejected candidates
            train_loss: float - Training loss
            train_accuracy: float - Training accuracy
            count_total_cells: int - Total cells counted
        """
        updates = {}

        if status:
            updates['status'] = status
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds

        # Map common result fields
        result_fields = [
            'det_cells_found', 'class_cells_found', 'class_rejected',
            'train_loss', 'train_accuracy', 'count_total_cells',
            'count_output_csv', 'output_path',
            # What a selective classification found, known only at its end
            # (log_selective_classification is called before it starts).
            'prefilter_total', 'prefilter_interior', 'prefilter_suspicious',
            'notes',
            # Comparison fields (for auto-compare after training)
            'diff_vs_best', 'diff_gained', 'diff_lost', 'comparison_ref'
        ]
        for field in result_fields:
            if field in kwargs and kwargs[field] is not None:
                updates[field] = kwargs[field]

        if expect_status is None:
            return self._update_row(exp_id, updates)

        def mutate(rows):
            row = next((r for r in rows if r.get('exp_id') == exp_id), None)
            if row is None:
                return "missing"
            if row.get('status') != expect_status:
                if updates == {'status': row.get('status')}:
                    # The same end again (a run already marked failed now
                    # reports failure): nothing new to keep.
                    return "same"
                arrived = ", ".join("%s=%s" % (k, v) for k, v in updates.items())
                late = ("a result arrived after this run was already '%s' and was "
                        "not applied: %s" % (row.get('status') or '', arrived))
                row['notes'] = "; ".join(filter(None, [row.get('notes', ''), late]))
                return "late"
            row.update({k: str(v) if v is not None else "" for k, v in updates.items()})
            return "ok"
        outcome = self._rewrite(mutate)
        if outcome == "late":
            print("[Tracker] [!] Run %s had already ended; a later result was added "
                  "to its notes, not applied" % exp_id)
        return outcome == "ok"

    def rate_experiment(
        self,
        exp_id: str,
        rating: int,
        notes: Optional[str] = None
    ) -> bool:
        """
        Rate an experiment (1-5 stars) with optional notes.
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")

        updates = {'rating': rating}
        if notes:
            # Append to existing notes
            rows = self._read_all()
            for row in rows:
                if row['exp_id'] == exp_id:
                    existing = row.get('notes', '')
                    if existing:
                        updates['notes'] = f"{existing}; {notes}"
                    else:
                        updates['notes'] = notes
                    break

        return self._update_row(exp_id, updates)

    def add_tags(self, exp_id: str, tags: List[str]) -> bool:
        """Add tags to an experiment."""
        rows = self._read_all()
        for row in rows:
            if row['exp_id'] == exp_id:
                existing = row.get('tags', '')
                if existing:
                    existing_tags = set(existing.split(','))
                    existing_tags.update(tags)
                    new_tags = ','.join(sorted(existing_tags))
                else:
                    new_tags = ','.join(sorted(tags))
                return self._update_row(exp_id, {'tags': new_tags})
        return False

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, str]]:
        """Get a single experiment by ID."""
        for row in self._read_all():
            if row['exp_id'] == exp_id:
                return row
        return None

    def search(
        self,
        brain: Optional[str] = None,
        exp_type: Optional[str] = None,
        status: Optional[str] = None,
        rating_min: Optional[int] = None,
        limit: Optional[int] = None,
        sort_by: str = "created_at",
        descending: bool = True,
        brain_contains: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Search calibration runs with filters.

        brain: ONE brain, matched as a whole name (see same_brain_name). Code
            that asks for "this brain's runs" uses this.
        brain_contains: any brain whose name contains the text. For a person
            typing part of a name into a search box -- never for code that
            means one brain.
        limit: at most this many rows; None (the default) returns every match.

        WHY whole-name and no default limit: this used to match part of the
        name and return only the 50 newest rows. A brain's run list therefore
        also showed runs of every brain whose name merely contained this one,
        and once 50 newer rows existed a brain's older runs -- its marked
        Best among them -- silently dropped out of the list and out of every
        "does this detection have a classification" lookup. A caller that
        wants a page asks for one with limit.
        """
        rows = self._read_all()

        # Filter
        if brain:
            rows = [r for r in rows if same_brain_name(r.get('brain'), brain)]
        if brain_contains:
            needle = brain_contains.lower()
            rows = [r for r in rows if needle in (r.get('brain') or '').lower()]
        if exp_type:
            rows = [r for r in rows if r.get('exp_type') == exp_type]
        if status:
            rows = [r for r in rows if r.get('status') == status]
        if rating_min:
            rows = [r for r in rows if r.get('rating') and int(r['rating']) >= rating_min]

        # Sort
        def sort_key(r):
            val = r.get(sort_by, '')
            # Handle numeric sorting
            if sort_by in ['rating', 'duration_seconds', 'det_cells_found']:
                try:
                    return float(val) if val else 0
                except:
                    return 0
            return val

        rows.sort(key=sort_key, reverse=descending)

        return rows if limit is None else rows[:limit]

    def get_recent(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get most recent experiments."""
        return self.search(limit=limit, sort_by="created_at", descending=True)

    def get_best(self, exp_type: str, brain: Optional[str] = None, limit: int = 5) -> List[Dict[str, str]]:
        """Get highest-rated experiments of a type."""
        return self.search(
            brain=brain,
            exp_type=exp_type,
            rating_min=1,
            limit=limit,
            sort_by="rating",
            descending=True
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        rows = self._read_all()

        stats = {
            'total': len(rows),
            'by_type': {},
            'by_status': {},
            'by_brain': {},
            'rated': 0,
            'avg_rating': 0,
        }

        ratings = []

        for row in rows:
            exp_type = row.get('exp_type', 'unknown')
            status = row.get('status', 'unknown')
            brain = row.get('brain', 'unknown')
            rating = row.get('rating', '')

            stats['by_type'][exp_type] = stats['by_type'].get(exp_type, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['by_brain'][brain] = stats['by_brain'].get(brain, 0) + 1

            if rating and rating.isdigit():
                stats['rated'] += 1
                ratings.append(int(rating))

        if ratings:
            stats['avg_rating'] = sum(ratings) / len(ratings)

        return stats

    def compare_experiments(self, exp_ids: List[str]) -> List[Dict[str, str]]:
        """Get multiple experiments for comparison."""
        rows = self._read_all()
        return [r for r in rows if r['exp_id'] in exp_ids]

    # =========================================================================
    # MARK-AS-BEST AND ARCHIVAL METHODS
    # =========================================================================

    def mark_as_best(self, exp_id: str) -> bool:
        """
        Mark an experiment as the 'best' for its brain.

        This automatically:
        1. Clears marked_best from any previous best for the same brain
        2. Sets marked_best=True and rating=5 for this experiment

        Args:
            exp_id: The experiment ID to mark as best

        Returns:
            True if successful
        """
        # Clear and set in ONE locked rewrite. They used to be separate
        # whole-file rewrites, so a crash or another machine's write in between
        # could leave the brain with no best, or two (and readers take the
        # first in file order -- the older one).
        def mutate(rows):
            target = next((r for r in rows if r.get('exp_id') == exp_id), None)
            if target is None:
                return False
            # The brain is matched as a whole name, in either form (see
            # same_brain_name and _bare_brain_name): the napari widget logs
            # "<brain>", the pipeline scripts "<mouse>/<brain>". An exact
            # string compare left a Best logged in the other form standing,
            # so the brain had two, and "Load best" took the first in file
            # order -- the OLDER run.
            bare = _bare_brain_name(target.get('brain'))
            for row in rows:
                if (row is not target
                        and same_brain_name(row.get('brain'), bare)
                        and row.get('exp_type') == target.get('exp_type')
                        and row.get('marked_best') == 'True'):
                    row['marked_best'] = ''
            target['marked_best'] = 'True'
            target['rating'] = '5'
            return True
        return self._rewrite(mutate)

    def archive_experiment(self, exp_id: str, archived_by: Optional[str] = None) -> bool:
        """
        Archive an experiment (mark it as superseded).

        Used when a new experiment replaces an old one as "best".

        Args:
            exp_id: The experiment ID to archive
            archived_by: The exp_id that replaced this one (optional)

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'archived_by': archived_by or '',
            'archived_at': datetime.now().isoformat(),
            'marked_best': '',  # Clear best status
            # And its Paradigm Best: an archived (superseded) run used to stay
            # the settled detection method for every brain of its imaging
            # setup, because only the brain-level mark was cleared.
            'paradigm_best': '',
        })

    # =========================================================================
    # IMAGING PARADIGM METHODS
    # =========================================================================

    def mark_paradigm_best(self, exp_id: str, fill_empty: Optional[Dict[str, Any]] = None,
                           note: Optional[str] = None) -> bool:
        """
        Mark an experiment as best for its imaging paradigm.

        This is different from mark_as_best which is per-brain.
        Paradigm best applies to ALL brains with the same imaging parameters
        (e.g., all 1p625x_z4 brains will use these detection settings).

        Args:
            exp_id: The experiment ID to mark as paradigm best
            fill_empty: {column: value} written into the run's row in the SAME
                locked rewrite, but only into columns that are empty. A column
                that already holds a different value is never overwritten: the
                mark is refused and nothing is written. WHY: a run made before
                a column existed (a classification's cube and batch size) can
                only become a settled method once the row says the whole
                method; the values a person fills in must never replace what
                the run itself recorded.
            note: text added to the run's notes (why it was marked, and where
                any filled-in value comes from).

        Returns:
            True if successful
        """
        # One locked rewrite: clear the paradigm's other Paradigm Best, clear
        # THIS BRAIN's other best (the run also becomes the brain's best, and a
        # brain must never have two), and set this one. Other brains' bests are
        # theirs and are left alone.
        #
        # A classification and a selective classification are one family here
        # (CLASSIFICATION_TYPES): both are "the classification method", so an
        # imaging setup has one Paradigm Best across the two, never one of each.
        fill_empty = dict(fill_empty or {})

        def family(row):
            t = row.get('exp_type')
            return CLASSIFICATION_TYPES if t in CLASSIFICATION_TYPES else (t,)

        def mutate(rows):
            target = next((r for r in rows if r.get('exp_id') == exp_id), None)
            if target is None:
                return "missing"
            paradigm = target.get('imaging_paradigm') or ''
            if not paradigm:
                return "no_paradigm"
            for col, value in fill_empty.items():
                have = (target.get(col) or '').strip()
                if have and have != str(value):
                    return "conflict:%s holds %s, not %s" % (col, have, value)
            kinds = family(target)
            bare = _bare_brain_name(target.get('brain'))  # either name form, as in mark_as_best
            for row in rows:
                if row is target or row.get('exp_type') not in kinds:
                    continue
                if row.get('imaging_paradigm') == paradigm and row.get('paradigm_best') == 'True':
                    row['paradigm_best'] = ''
                if same_brain_name(row.get('brain'), bare) and row.get('marked_best') == 'True':
                    row['marked_best'] = ''
            for col, value in fill_empty.items():
                target[col] = str(value)
            if note:
                target['notes'] = "; ".join(filter(None, [target.get('notes', ''), note]))
            target['paradigm_best'] = 'True'
            target['marked_best'] = 'True'
            target['rating'] = '5'
            return "ok"
        outcome = self._rewrite(mutate)
        if outcome == "no_paradigm":
            print(f"[Tracker] Cannot mark paradigm best: no imaging_paradigm for {exp_id}")
        elif outcome == "missing":
            print(f"[Tracker] Cannot mark paradigm best: no run {exp_id}")
        elif str(outcome).startswith("conflict:"):
            print(f"[Tracker] Cannot mark paradigm best: {outcome[9:]} -- nothing was written")
        return outcome == "ok"

    def get_best_for_paradigm(
        self,
        imaging_paradigm: str,
        exp_type: str = "detection",
        strict: bool = False,
    ) -> Optional[Dict[str, str]]:
        """
        Get the best experiment for an imaging paradigm.

        Args:
            imaging_paradigm: e.g., "1p625x_z4"
            exp_type: "detection", "classification", or "training"
            strict: True returns ONLY the run a person marked Paradigm Best.
                False (for display only) falls back to the newest run marked
                Best for any one brain of the setup.

        WHY strict exists: the fallback used to feed the settled method.
        With no Paradigm Best -- never marked, or archived by mistake -- one
        brain's Best quietly became the detection method for every brain of
        its imaging setup, and step 4 logged it as 'paradigm_best'. Anything
        that RUNS a method (get_paradigm_detection_settings,
        get_paradigm_model) asks with strict=True, so no Paradigm Best means
        no settled method, and the caller stops and says so.

        Returns:
            The paradigm-best experiment dict, or None
        """
        # An archived run has been superseded and is never the settled method.
        rows = [r for r in self._read_all() if not (r.get('archived_at') or '').strip()]

        # Look for paradigm_best first
        for row in rows:
            if (row.get('imaging_paradigm') == imaging_paradigm and
                row.get('exp_type') == exp_type and
                row.get('paradigm_best') == 'True'):
                return row

        if strict:
            return None

        # Display-only fallback: the newest marked_best of this paradigm
        candidates = [
            r for r in rows
            if (r.get('imaging_paradigm') == imaging_paradigm and
                r.get('exp_type') == exp_type and
                r.get('marked_best') == 'True')
        ]

        if candidates:
            # Return most recent
            candidates.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return candidates[0]

        return None

    def get_paradigm_detection_settings(self, imaging_paradigm: str) -> Optional[Dict[str, Any]]:
        """
        The settled detection method for an imaging paradigm.

        The method is the run a person marked Paradigm Best for that imaging
        setup (magnification + z-step). All eight of its settings come from
        that run's row (DETECTION_METHOD_COLUMNS) -- none are filled in here.

        Args:
            imaging_paradigm: e.g., "1p625x_z4"

        Returns:
            None if no run is marked best for the paradigm. Otherwise a dict
            with the eight settings (ball_xy, ball_z, soma_diameter, threshold,
            ball_overlap, soma_spread, log_sigma, max_cluster), plus preset,
            source_exp_id, source_brain, and 'missing': the names of any of
            the eight the run's row does not record. A caller must NOT run
            detection while 'missing' is non-empty -- it would be filling half
            the method with someone else's numbers.
        """
        # strict: only the run marked Paradigm Best is the settled method
        # (see get_best_for_paradigm for why one brain's Best never is).
        best = self.get_best_for_paradigm(imaging_paradigm, exp_type="detection",
                                          strict=True)
        if not best:
            return None

        # No defaults for absent values. These used to read `or 6`, `or 15`
        # ... -- the generic numbers -- so a row missing a setting quietly
        # became a different method. An absent value is reported in 'missing'.
        # Numbers go through float() first because the CSV may hold "8.0".
        settings = {'missing': []}
        for name, column in DETECTION_METHOD_COLUMNS.items():
            raw = (best.get(column) or '').strip()
            if not raw:
                settings[name] = None
                settings['missing'].append(name)
            elif name in ('threshold', 'max_cluster'):
                settings[name] = int(float(raw))
            else:
                settings[name] = float(raw)
        settings.update({
            'preset': best.get('det_preset') or 'custom',
            'source_exp_id': best.get('exp_id'),
            'source_brain': best.get('brain'),
        })
        return settings

    def get_paradigm_classification(self, imaging_paradigm: str) -> Optional[Dict[str, Any]]:
        """
        The settled classification method for an imaging paradigm: the
        classification or selective classification run a person marked
        Paradigm Best for it. Step 5 runs exactly this and nothing else.

        Returns None when no such run is marked. Otherwise a dict:
            method        'selective' (selective_classification row) or 'full'
            model_path    the model file the run used (class_model_path)
            cube_size, batch_size   ints, or None when the row lacks them
            tracing_type  selective only (prefilter_tracing_type)
            source_exp_id, source_brain, created_at
            missing       names of anything the row does not record. A caller
                          must NOT classify while it is non-empty: it would be
                          finishing the method with numbers the run never used.

        WHY strict and pinned to one row: the model used to be "the newest
        model file in the models folder" whenever nothing was marked, and two
        models on the same brain gave counts that differ several-fold. The
        settled method is the one run a person confirmed, with its own model
        file and settings -- never whichever file is newest.
        """
        rows = [r for r in self._read_all() if not (r.get('archived_at') or '').strip()]
        best = next((r for r in rows
                     if r.get('imaging_paradigm') == imaging_paradigm
                     and r.get('exp_type') in CLASSIFICATION_TYPES
                     and r.get('paradigm_best') == 'True'), None)
        if best is None:
            return None
        selective = best.get('exp_type') == 'selective_classification'
        missing = []

        def as_int(col, name):
            raw = (best.get(col) or '').strip()
            if not raw:
                missing.append(name)
                return None
            return int(float(raw))
        model = (best.get('class_model_path') or best.get('model_path') or '').strip()
        if not model or model == 'default':
            missing.append('model_path')
        method = {
            'method': 'selective' if selective else 'full',
            'model_path': model or None,
            'cube_size': as_int('class_cube_size', 'cube_size'),
            'batch_size': as_int('class_batch_size', 'batch_size'),
            'tracing_type': None,
            'source_exp_id': best.get('exp_id'),
            'source_brain': best.get('brain'),
            'created_at': best.get('created_at'),
        }
        if selective:
            method['tracing_type'] = (best.get('prefilter_tracing_type') or '').strip() or None
            if not method['tracing_type']:
                missing.append('tracing_type')
        method['missing'] = missing
        return method

    def get_paradigm_model(self, imaging_paradigm: str) -> Optional[str]:
        """
        Get the best model path for an imaging paradigm.

        Returns the model_path from the paradigm-best classification run
        (a plain or a selective classification: get_paradigm_classification).

        Args:
            imaging_paradigm: e.g., "1p625x_z4"

        Returns:
            Model path string, or None
        """
        # strict, as for detection: a run marked Best for one brain is never
        # the model for every brain of the setup.
        method = self.get_paradigm_classification(imaging_paradigm)
        return (method or {}).get('model_path') or None

    def update_rating(self, exp_id: str, rating: int, notes: Optional[str] = None) -> bool:
        """
        Update experiment rating (1-5) with optional notes.

        This is a convenience alias for rate_experiment.
        """
        return self.rate_experiment(exp_id, rating, notes)

    def log_comparison(
        self,
        exp_id: str,
        ref_id: str,
        gained: int,
        lost: int,
    ) -> bool:
        """
        Log comparison results between two experiments.

        Args:
            exp_id: The experiment being evaluated
            ref_id: The reference experiment (e.g., current best)
            gained: Number of points gained vs reference
            lost: Number of points lost vs reference

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'comparison_ref': ref_id,
            'diff_gained': gained,
            'diff_lost': lost,
            'diff_vs_best': gained + lost,  # Total difference
        })

    def get_best_for_brain(self, brain: str, exp_type: str = 'detection') -> Optional[Dict[str, str]]:
        """
        Get the marked-best experiment for a specific brain.

        Args:
            brain: Brain name to search for
            exp_type: Experiment type ('detection', 'classification', etc.)

        Returns:
            The best experiment dict, or None if not found
        """
        # The brain is matched as a whole name (see same_brain_name).
        rows = [r for r in self._read_all()
                if same_brain_name(r.get('brain'), brain)
                and r.get('exp_type') == exp_type]
        for row in rows:
            if row.get('marked_best') == 'True':
                return row

        # Fall back to the highest-rated run that has not been archived (a
        # former best keeps its rating after it is superseded).
        rated = [r for r in rows
                 if (r.get('rating') or '').strip().isdigit()
                 and not (r.get('archived_at') or '').strip()]
        rated.sort(key=lambda r: (int(r['rating']), r.get('created_at', '')), reverse=True)
        return rated[0] if rated else None


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_experiment_row(row: Dict[str, str], verbose: bool = False):
    """Print a single experiment row in a readable format."""
    exp_id = row.get('exp_id', '?')
    exp_type = row.get('exp_type', '?')
    brain = row.get('brain', '?')
    status = row.get('status', '?')
    created = row.get('created_at', '?')[:16]  # Trim to date + time
    rating = row.get('rating', '')

    # Status symbols (ASCII-safe for Windows)
    status_symbols = {
        'completed': '[OK]',
        'failed': '[X]',
        'started': '[..]',
        'running': '[>>]',
        'cancelled': '[--]',
        'approved': '[OK]',
    }
    symbol = status_symbols.get(status, '[?]')

    # Rating stars (ASCII-safe)
    if rating and rating.isdigit():
        r = int(rating)
        stars = '*' * r + '.' * (5 - r)
    else:
        stars = '-----'

    # Type abbreviation
    type_abbrev = {'detection': 'DET', 'training': 'TRN', 'classification': 'CLS', 'registration': 'REG', 'counts': 'CNT'}
    type_str = type_abbrev.get(exp_type, exp_type[:3].upper())

    print(f"  {symbol} [{type_str}] {exp_id}  {brain}  {stars}  {created}")

    if verbose:
        # Show more details
        duration = row.get('duration_seconds', '')
        if duration:
            mins = float(duration) / 60
            print(f"           Duration: {mins:.1f} min")

        cells = row.get('det_cells_found') or row.get('class_cells_found') or row.get('count_total_cells')
        if cells:
            print(f"           Cells: {cells}")

        notes = row.get('notes', '')
        if notes:
            print(f"           Notes: {notes[:60]}...")


def main():
    """CLI interface for calibration run tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BrainGlobe Calibration Run Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tracks CALIBRATION runs (tuning/optimization experiments).
For PRODUCTION results, see region_counts.csv.

Examples:
    python -m mousebrain.tracker                    # Show recent runs
    python -m mousebrain.tracker --search "001"     # Search by brain
    python -m mousebrain.tracker --type detection   # Filter by type
    python -m mousebrain.tracker --stats            # Show statistics
    python -m mousebrain.tracker --best detection   # Best detection runs
        """
    )

    parser.add_argument('--search', '-s', help='Search term (brain name, exp_id, etc.)')
    parser.add_argument('--type', '-t', choices=EXP_TYPES, help='Filter by experiment type')
    parser.add_argument('--status', choices=['completed', 'failed', 'started', 'running'])
    parser.add_argument('--best', metavar='TYPE', help='Show best-rated of a type')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--limit', '-n', type=int, default=20, help='Number of results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show more details')
    parser.add_argument('--csv', help='Path to experiments CSV (default: standard location)')

    args = parser.parse_args()

    # Initialize tracker
    tracker = ExperimentTracker(args.csv)

    print("=" * 70)
    print("BrainGlobe Calibration Run Tracker")
    print(f"CSV: {tracker.csv_path}")
    print("(For production results, see region_counts.csv)")
    print("=" * 70)

    if args.stats:
        stats = tracker.get_statistics()
        print(f"\nTotal calibration runs: {stats['total']}")
        print(f"Rated: {stats['rated']} (avg: {stats['avg_rating']:.1f}*)")

        print("\nBy type:")
        for t, count in sorted(stats['by_type'].items()):
            print(f"  {t}: {count}")

        print("\nBy status:")
        for s, count in sorted(stats['by_status'].items()):
            print(f"  {s}: {count}")

        print("\nBy brain (top 10):")
        sorted_brains = sorted(stats['by_brain'].items(), key=lambda x: x[1], reverse=True)[:10]
        for b, count in sorted_brains:
            print(f"  {b}: {count}")
        return

    if args.best:
        print(f"\nBest {args.best} runs:")
        rows = tracker.get_best(args.best, limit=args.limit)
    elif args.search or args.type or args.status:
        print(f"\nSearch results:")
        rows = tracker.search(
            brain_contains=args.search,
            exp_type=args.type,
            status=args.status,
            limit=args.limit
        )
    else:
        print(f"\nRecent calibration runs:")
        rows = tracker.get_recent(limit=args.limit)

    if not rows:
        print("  No calibration runs found.")
    else:
        for row in rows:
            print_experiment_row(row, verbose=args.verbose)

    print()


if __name__ == '__main__':
    main()
