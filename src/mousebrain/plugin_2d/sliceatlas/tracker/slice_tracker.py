"""
slice_tracker.py - Calibration run tracking for BrainSlice pipeline.

Tracks all calibration runs (registration, detection, colocalization, quantification)
to a CSV file for reproducibility and history.

Usage:
    from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

    tracker = SliceTracker()
    run_id = tracker.log_detection(sample_id="ENCR_001_slice12", ...)
    tracker.update_status(run_id, status="completed", det_nuclei_found=1234)
    tracker.mark_as_best(run_id)
"""

import csv
import json
import os
import socket
import hashlib
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from mousebrain.file_lock import exclusive_file_lock

from .schema import CSV_COLUMNS, RUN_TYPES


# HOW THE CSV IS WRITTEN (see SliceTracker._rewrite and _write_row)
#
# The 2D tracker is one CSV that the napari widget and the batch script write,
# possibly at the same time and from different machines. Every status change
# rewrites the whole file. That rewrite used to:
#   * write in place, so a crash part-way left a truncated tracker;
#   * take no lock, so the widget and the batch each wrote back the rows they
#     had read, and one of them lost the other's new rows and updates.
# Now every write holds an exclusive lock file beside the CSV
# (mousebrain.file_lock), and a rewrite goes through a temporary file swapped
# in with os.replace, keeping every column the file has (known to this
# version or not) in the file's own order.
LOCK_STALE_SECONDS = 120
LOCK_WAIT_SECONDS = 300
REPLACE_RETRY_SECONDS = 30  # a reader on Windows can hold the file open briefly
# The napari window writes from its main thread, which cannot repaint while it
# waits, so it opens the tracker with much shorter waits (see
# SliceTracker.__init__ and the widget's _open_tracker). The module values are
# for the batch, which can afford to wait.

# HOW THE CSV IS READ: every read opens it as 'utf-8-sig'. WHY: a spreadsheet
# program saving the tracker as "CSV UTF-8" puts a byte-order mark (BOM) in
# front of the first column name. Read as plain 'utf-8', the first column was
# then called BOM+'run_id', not 'run_id': the next open "added" a second, empty
# run_id column, and from then on no existing run could be found by its id
# (status updates, mark-as-best, failing an unfinished run, the batch's
# marked-best lookup all missed every old row). 'utf-8-sig' drops the mark
# and reads a file without one exactly as 'utf-8' does. Writes stay plain
# 'utf-8' (no mark), so the next rewrite also removes it.
READ_ENCODING = "utf-8-sig"


# The one word for a run that did not finish. WHY: the batch used to write
# 'error' and the widget 'failed' for the same state, so anything looking for
# failed runs missed the batch's. update_status turns 'error' into this.
FAILED = "failed"


# The columns update_status() may write: a run's results and output paths,
# never its identity or its parameters (those are fixed when the run is
# logged). Anything else given to update_status() is refused (see there).
RESULT_FIELDS = (
    'det_nuclei_found', 'det_raw_count', 'det_removed_by_size',
    'det_removed_by_border', 'det_removed_by_morphology', 'det_removed_by_confidence',
    'coloc_positive_cells', 'coloc_negative_cells', 'coloc_positive_fraction',
    'coloc_background_value',
    'coloc_dual_cells', 'coloc_ch1_only_cells', 'coloc_ch2_only_cells', 'coloc_neither_cells',
    'quant_total_regions', 'quant_top_region', 'quant_top_region_count',
    'pa_particles_found', 'pa_particles_raw', 'pa_bg_value', 'pa_mean_intensity',
    'pa_positive_particles',
    'output_path', 'labels_path', 'measurements_path', 'region_counts_path',
    'reg_transform_path', 'reg_quality_score',
)


class TrackerNotConfiguredError(RuntimeError):
    """SliceTracker() was asked for the default tracker on an install that
    does not know where the lab's data lives (see SliceTracker.__init__)."""


class SliceTracker:
    """
    Central calibration run tracking for BrainSlice pipeline.

    All runs are logged to a single CSV file with consistent columns.
    """

    def __init__(self, csv_path: Optional[Path] = None, *,
                 lock_wait_seconds: Optional[float] = None,
                 replace_retry_seconds: Optional[float] = None):
        """
        Initialize tracker.

        Args:
            csv_path: Path to calibration runs CSV. Uses the configured 2D
                tracker (<MouseBrain_Pipeline>/2D_Slices/2_Data_Summary/
                calibration_runs.csv) if not specified.
            lock_wait_seconds / replace_retry_seconds: how long one write
                waits for another writer's lock, and keeps retrying the swap
                while something holds the CSV open. None = the module values
                (LOCK_WAIT_SECONDS, REPLACE_RETRY_SECONDS). WHY settable: a
                GUI writes on the thread that draws its window, so a long
                wait there freezes the window; it passes short waits and
                reports a write that could not be made instead.

        Raises:
            TrackerNotConfiguredError: no csv_path was given and mousebrain is
                not configured (CONNECTOME_ROOT unset). WHY: the default path
                is then under a temporary stand-in folder; runs logged there
                look recorded but never reach the lab's tracker. Callers show
                this reason to the person instead of running unlogged in
                silence.
        """
        if csv_path is None:
            from mousebrain import config as _cfg
            if not _cfg.is_configured():
                raise TrackerNotConfiguredError(
                    "mousebrain is not configured, so the 2D tracker's default "
                    "location is a temporary stand-in and runs logged there would "
                    "be lost. " + _cfg.UNCONFIGURED_MESSAGE)
            from ..core.config import TRACKER_CSV
            if TRACKER_CSV is None:
                raise TrackerNotConfiguredError(
                    "the 2D tracker has no configured location. " + _cfg.UNCONFIGURED_MESSAGE)
            csv_path = TRACKER_CSV

        self.csv_path = Path(csv_path)
        self._lock_wait_seconds = lock_wait_seconds
        self._replace_retry_seconds = replace_retry_seconds
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create the CSV with headers if it does not exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with self._lock():
                if not self.csv_path.exists():
                    tmp = self._tmp_path()
                    with open(tmp, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                        writer.writeheader()
                    self._replace(tmp)
        else:
            self._add_missing_columns()

    def _add_missing_columns(self):
        """Give a tracker written before the newest columns those columns,
        every existing value kept under its own name, before anything is
        appended to it (see _write_row)."""
        with open(self.csv_path, 'r', newline='', encoding=READ_ENCODING) as f:
            header = next(csv.reader(f), None) or []
        if all(col in header for col in CSV_COLUMNS) and len(set(header)) == len(header):
            return
        # Also rewritten: a header that names one column twice -- left by an
        # older version that read a byte-order-marked file (see READ_ENCODING)
        # and added a second, empty run_id. The rewrite folds the two into one
        # (see _parse), so the old rows are found by their ids again.
        self._rewrite(lambda rows: None)

    # -----------------------------------------------------------------
    # Safe writing (see the note at the top of this module)
    # -----------------------------------------------------------------
    @property
    def _lock_path(self) -> Path:
        return self.csv_path.with_name(self.csv_path.name + ".lock")

    def _lock(self):
        """Hold the tracker's write lock for the duration of the block."""
        return exclusive_file_lock(self.csv_path, "SliceTracker",
                                   stale_seconds=LOCK_STALE_SECONDS,
                                   wait_seconds=(LOCK_WAIT_SECONDS
                                                 if self._lock_wait_seconds is None
                                                 else self._lock_wait_seconds))

    def _tmp_path(self) -> Path:
        return self.csv_path.with_name(
            "%s.tmp-%d-%s" % (self.csv_path.name, os.getpid(), uuid.uuid4().hex[:8]))

    def _replace(self, tmp: Path) -> None:
        """Swap the finished temporary file in. Retries while a reader on
        Windows briefly holds the CSV open; removes the temporary file if the
        swap finally fails, so nothing is left beside the tracker."""
        budget = (REPLACE_RETRY_SECONDS if self._replace_retry_seconds is None
                  else self._replace_retry_seconds)
        deadline = time.monotonic() + budget
        try:
            while True:
                try:
                    os.replace(tmp, self.csv_path)
                    return
                except PermissionError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def _parse(self, f):
        """(header, rows) of an open CSV, every column name once.

        A header that names a column twice is folded into one column. WHY:
        an older version of this code, reading a file saved with a byte-order
        mark, took its first column for an unknown one and appended a second,
        empty 'run_id' (see READ_ENCODING); read as dicts, the empty
        right-hand copy hid every real id. Each row keeps the one non-empty
        value of the repeated column. Two DIFFERENT non-empty values under
        one name cannot be folded without choosing one, so that raises
        RuntimeError: a person has to look.
        """
        reader = csv.reader(f)
        raw_header = next(reader, None) or []
        header = list(dict.fromkeys(raw_header))
        positions = {name: [i for i, h in enumerate(raw_header) if h == name]
                     for name in header}
        rows = []
        for line_no, values in enumerate(reader, start=2):
            if not values:
                continue  # a blank line, as csv.DictReader skips it
            row = {}
            for name, idx in positions.items():
                found = [values[i] for i in idx if i < len(values)]
                filled = list(dict.fromkeys(v for v in found if v != ""))
                if len(filled) > 1:
                    raise RuntimeError(
                        f"{self.csv_path}: line {line_no} has different values "
                        f"{filled} under the repeated column '{name}'. Repair it first.")
                row[name] = filled[0] if filled else ("" if found else None)
            if len(values) > len(raw_header):
                # Kept the way csv.DictReader keeps them, so _read_locked
                # can refuse to rewrite a row whose extra values it would drop.
                row[None] = values[len(raw_header):]
            rows.append(row)
        return header, rows

    def _read_locked(self):
        """(header, rows) of the file. Call with the lock held."""
        with open(self.csv_path, 'r', newline='', encoding=READ_ENCODING) as f:
            header, rows = self._parse(f)
        if any(None in row for row in rows):
            # A row with more values than the header has columns: rewriting
            # would drop the extra values. A person has to look.
            raise RuntimeError(
                f"{self.csv_path}: some rows have more values than the header has "
                f"columns, so the file cannot be rewritten without losing them. "
                f"Repair it first.")
        return header, rows

    def _rewrite_locked(self, mutate):
        """The body of _rewrite; the caller holds the lock."""
        header, rows = self._read_locked()
        result = mutate(rows)
        columns = header + [c for c in CSV_COLUMNS if c not in header]
        extra = [k for row in rows for k in row if k not in columns]
        columns += list(dict.fromkeys(extra))
        tmp = self._tmp_path()
        try:
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({c: row.get(c, "") for c in columns})
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        self._replace(tmp)
        return result

    def _rewrite(self, mutate):
        """Read every row, let `mutate(rows)` change them, write them back.

        Under the lock; in the file's own column order, keeping columns this
        version does not know and adding any column it does know at the end;
        through a temporary file, so a failure part-way leaves the file as it
        was. Returns what `mutate` returns.
        """
        with self._lock():
            return self._rewrite_locked(mutate)

    def _generate_run_id(self, run_type: str, sample_id: str) -> str:
        """Generate a unique run ID.

        The id keeps its readable form (type, second, 6 hex digits), but the
        hash includes a random part. WHY: it used to hash only the type,
        sample, second and process id, so two runs of one sample started in
        the same second got the same id -- and update_status() then changed
        the first matching row, i.e. the wrong run.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{run_type}_{sample_id}_{timestamp}_{os.getpid()}_{uuid.uuid4().hex}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{run_type[:3]}_{timestamp}_{short_hash}"

    def _get_hostname(self) -> str:
        """Get current hostname."""
        try:
            return socket.gethostname()
        except:
            return "unknown"

    def _add_hierarchy_fields(self, row: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """Auto-populate hierarchy fields from sample name."""
        from ..core.config import parse_sample_name

        parsed = parse_sample_name(sample_id)
        row["project"] = parsed.get("project", "")
        row["cohort"] = parsed.get("cohort", "")
        row["slice_num"] = parsed.get("slice_num", "")

        return row

    def _write_row(self, row: Dict[str, Any]):
        """Append a row to the CSV.

        In the FILE's column order, not this code's. WHY: the two differ
        whenever the file was written by another version of this code, and
        appending in the code's order put a row's values under the wrong
        column names (channel names under input_path, an area under
        output_path ...) -- which the next rewrite then made permanent. If
        the file lacks a column this version writes, the row goes in through
        one rewrite that adds the column first, so no value is dropped.
        Under the lock, so an append never lands inside another writer's
        rewrite.
        """
        full_row = {col: ("" if row.get(col) is None else row.get(col)) for col in CSV_COLUMNS}

        with self._lock():
            with open(self.csv_path, 'r', newline='', encoding=READ_ENCODING) as f:
                header = next(csv.reader(f), None) or []
            # A header naming a column twice goes through the rewrite, which
            # folds it into one (see _parse), never through a plain append.
            if (header and len(set(header)) == len(header)
                    and all(col in header for col in CSV_COLUMNS)):
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerow({c: full_row.get(c, "") for c in header})
            else:
                self._rewrite_locked(lambda rows: rows.append(full_row))

    def _read_all(self) -> List[Dict[str, str]]:
        """Read all runs from CSV."""
        if not self.csv_path.exists():
            return []

        with open(self.csv_path, 'r', newline='', encoding=READ_ENCODING) as f:
            return self._parse(f)[1]

    def _update_row(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update a specific row by run_id. One locked rewrite (see _rewrite)."""
        return self._update_rows({run_id: updates})

    def _update_rows(self, updates_by_id: Dict[str, Dict[str, Any]]) -> bool:
        """Apply several rows' updates in ONE locked rewrite, so a change that
        must touch more than one row (moving a 'best' mark) can never be left
        half done. True when the first id given was found."""
        def mutate(rows):
            found = set()
            for row in rows:
                rid = row.get('run_id')
                upd = updates_by_id.get(rid)
                if upd is not None and rid not in found:
                    row.update({k: str(v) if v is not None else "" for k, v in upd.items()})
                    found.add(rid)
            return found
        found = self._rewrite(mutate)
        first = next(iter(updates_by_id), None)
        return first in found

    # -----------------------------------------------------------------
    # No run left 'started'
    # -----------------------------------------------------------------
    @contextmanager
    def finish_or_fail(self, run_id: Optional[str]):
        """Wrap a run's work: if the block ends while the run is still
        'started' -- an exception, Ctrl-C, or a path that forgot to record a
        result -- the run is set to 'failed' with the reason in its notes.
        WHY: a row left at 'started' forever cannot be told apart from a run
        still going. (A process killed outright runs no code at all and still
        leaves 'started'.)"""
        try:
            yield
        except BaseException as e:
            self.fail_if_unfinished(run_id, "stopped before finishing: %s: %s"
                                    % (type(e).__name__, e))
            raise
        else:
            self.fail_if_unfinished(run_id, "ended without recording a result")

    def fail_if_unfinished(self, run_id: Optional[str], why: str) -> bool:
        """Set a run that is still 'started' to 'failed', adding `why` to its
        notes. A run that already finished (completed, or failed) is never
        touched. WHY: a problem in a LATER step must not overwrite a result
        that was recorded -- the batch used to turn a completed detection row
        into 'error' when its colocalization step crashed. Never raises: it
        runs on error paths, where it must not hide the original error.
        True when the row was changed."""
        if not run_id:
            return False
        try:
            # A plain read first: a caller may call this on every exit path,
            # and a run that already finished needs no locked rewrite.
            run = self.get_run(run_id)
            if run is None or run.get('status') != 'started':
                return False

            def mutate(rows):
                for row in rows:
                    if row.get('run_id') == run_id and row.get('status') == 'started':
                        row['status'] = FAILED
                        row['notes'] = "; ".join(filter(None, [row.get('notes', ''), why]))
                        return True
                return False
            changed = bool(self._rewrite(mutate))
            if changed:
                print("[SliceTracker] Run %s marked failed: %s" % (run_id, why))
            return changed
        except Exception as e:
            print("[SliceTracker] [!] Could not mark run %s failed: %s" % (run_id, e))
            return False

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def log_registration(
        self,
        sample_id: str,
        atlas: str = "allen_mouse_10um",
        orientation: str = "coronal",
        method: str = "abba_manual",
        ap_position: Optional[float] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a slice registration run."""
        run_id = self._generate_run_id("registration", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "registration",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "reg_atlas": atlas,
            "reg_orientation": orientation,
            "reg_method": method,
            "reg_ap_position": ap_position,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_detection(
        self,
        sample_id: str,
        channel: str = "red",
        model: str = "2D_versatile_fluo",
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        scale: float = 1.0,
        min_area: int = 50,
        max_area: int = 5000,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
        backend: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a nuclei detection run.

        backend / parameters: the detection backend and EVERY setting it ran
        with, stored as det_backend and as JSON in det_preprocessing. WHY:
        the fixed det_* arguments are StarDist's; a threshold or
        threshold+log run's own settings were recorded nowhere, so the run
        could not be reproduced from its row.
        """
        run_id = self._generate_run_id("detection", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "detection",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "det_channel": channel,
            "det_model": model,
            "det_prob_thresh": prob_thresh,
            "det_nms_thresh": nms_thresh,
            "det_scale": scale,
            "det_min_area": min_area,
            "det_max_area": max_area,
            "det_backend": backend,
            "det_preprocessing": (json.dumps(parameters, sort_keys=True, default=str)
                                  if parameters else None),
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_colocalization(
        self,
        sample_id: str,
        signal_channel: str = "green",
        background_method: str = "percentile",
        background_percentile: float = 10.0,
        background_value: Optional[float] = None,
        threshold_method: str = "fold_change",
        threshold_value: float = 2.0,
        parent_run: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a colocalization analysis run."""
        run_id = self._generate_run_id("colocalization", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "colocalization",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "coloc_signal_channel": signal_channel,
            "coloc_background_method": background_method,
            "coloc_background_percentile": background_percentile,
            "coloc_background_value": background_value,
            "coloc_threshold_method": threshold_method,
            "coloc_threshold_value": threshold_value,
            "parent_run": parent_run,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_particle_analysis(
        self,
        sample_id: str,
        detection_channel: str = '0',
        threshold_value: float = 0.0,
        measurement_channel: str = '1',
        min_area: int = 10,
        max_area: int = 50000,
        min_circularity: float = 0.0,
        max_circularity: float = 1.0,
        bg_method: str = 'percentile',
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = 'started',
        script_version: str = '1.0.0',
        min_pct_above_bg: Optional[float] = None,
        results: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
    ) -> str:
        """Log a particle analysis run.

        min_pct_above_bg: the positive rule (a particle is positive when at
        least this % of its pixels are above background).
        results: result columns (names from RESULT_FIELDS) to write in the
        same row, for a run logged once it has finished. WHY: the batch in
        the napari window logs each image after analyzing it, in ONE write
        with its final status, so no row of it can be left 'started' by a
        second write the file refuses. A name that is not a result column
        raises ValueError before anything is written; None values are left
        blank.
        """
        results = dict(results or {})
        unknown = sorted(k for k in results if k not in RESULT_FIELDS)
        if unknown:
            raise ValueError(
                "log_particle_analysis cannot record %s: not a result column"
                % ", ".join(unknown))
        run_id = self._generate_run_id('particle_analysis', sample_id)

        row = {
            'run_id': run_id,
            'run_type': 'particle_analysis',
            'sample_id': sample_id,
            'created_at': datetime.now().isoformat(),
            'status': status,
            'pa_detection_channel': detection_channel,
            'pa_threshold_value': threshold_value,
            'pa_measurement_channel': measurement_channel,
            'pa_min_area': min_area,
            'pa_max_area': max_area,
            'pa_min_circularity': min_circularity,
            'pa_max_circularity': max_circularity,
            'pa_bg_method': bg_method,
            'pa_min_pct_above_bg': min_pct_above_bg,
            'input_path': input_path,
            'output_path': output_path,
            'notes': notes,
            'script_version': script_version,
            'hostname': self._get_hostname(),
        }
        if duration_seconds is not None:
            row['duration_seconds'] = duration_seconds
        for field, value in results.items():
            if value is not None:
                row[field] = value

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_particle_reclassification(
        self,
        sample_id: str,
        parent_run: Optional[str],
        bg_value: float,
        min_pct_above_bg: float,
        particles_found: int,
        positive_particles: int,
        input_path: Optional[str] = None,
        measurements_path: Optional[str] = None,
        script_version: str = '1.0.0',
    ) -> str:
        """Log a re-classification of an analysis's particles: the same
        particles, classified again with another background and/or positive
        rule. Logged complete in one write (it is instantaneous).

        WHY its own row, not an edit of the analysis's row: re-classifying
        rewrites the saved results (particles.csv) without finding the
        particles again, and used to leave no trace at all -- the analysis's
        row still stated the old background and counts. The original row
        keeps describing what the analysis produced; this row says what the
        results were changed to and when, and its parent_run is that
        analysis (blank when the particles on screen have no tracker row).
        The particle-finding settings are copied from the parent so the row
        stands on its own.
        """
        run_id = self._generate_run_id('particle_analysis', sample_id)
        parent = self.get_run(parent_run) if parent_run else None
        copied = {col: parent.get(col, '') for col in (
            'pa_detection_channel', 'pa_threshold_value', 'pa_measurement_channel',
            'pa_min_area', 'pa_max_area', 'pa_min_circularity',
            'pa_max_circularity', 'pa_particles_raw')} if parent else {}
        row = dict(copied)
        row.update({
            'run_id': run_id,
            'run_type': 'particle_analysis',
            'sample_id': sample_id,
            'created_at': datetime.now().isoformat(),
            'status': 'completed',
            'pa_bg_method': 'manual_reclassify',
            'pa_bg_value': bg_value,
            'pa_min_pct_above_bg': min_pct_above_bg,
            'pa_particles_found': particles_found,
            'pa_positive_particles': positive_particles,
            'parent_run': parent_run,
            'input_path': input_path,
            'measurements_path': measurements_path,
            'notes': ("re-classified the particles of %s" % parent_run if parent_run
                      else "re-classified particles that have no tracker row"),
            'script_version': script_version,
            'hostname': self._get_hostname(),
        })
        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_quantification(
        self,
        sample_id: str,
        parent_run: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a quantification run."""
        run_id = self._generate_run_id("quantification", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "quantification",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "parent_run": parent_run,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    # =========================================================================
    # UPDATE METHODS
    # =========================================================================

    def update_status(
        self,
        run_id: str,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Update run status and results.

        Common kwargs:
            det_nuclei_found: int - Detection nuclei count
            coloc_positive_cells: int - Positive cells count
            coloc_negative_cells: int - Negative cells count
            coloc_positive_fraction: float - Fraction positive
            quant_total_regions: int - Number of regions with cells

        A keyword that is not one of RESULT_FIELDS raises ValueError, before
        anything is written. WHY: an unknown keyword used to be dropped
        without a word, so a caller recording a result under a misspelt or
        unsupported name believed it was recorded when nothing was. A value
        of None is left out (the column keeps what it has).
        """
        unknown = sorted(k for k in kwargs if k not in RESULT_FIELDS)
        if unknown:
            raise ValueError(
                "update_status cannot record %s: not a result column (allowed: %s)"
                % (", ".join(unknown), ", ".join(RESULT_FIELDS)))
        updates = {}

        if status == 'error':
            status = FAILED  # one word for one state (see FAILED)
        if status:
            updates['status'] = status
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds

        for field in RESULT_FIELDS:
            if field in kwargs and kwargs[field] is not None:
                updates[field] = kwargs[field]

        return self._update_row(run_id, updates)

    def record_result_if_started(
        self,
        run_id: str,
        status: str,
        duration_seconds: Optional[float] = None,
        *,
        lock_wait_seconds: Optional[float] = None,
        replace_retry_seconds: Optional[float] = None,
        append_note: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Write a run's result (as update_status would) only while its row
        still says 'started'; True when it was written, False when the row is
        missing or already finished (nothing written).

        For a result that could not be written when the run finished (the
        napari window keeps it and tries again later). WHY only a 'started'
        row: by the time of a later try the row may have been finished some
        other way, and a late write must never overwrite that. The check and
        the write are one locked rewrite, so nothing can change the row in
        between.

        lock_wait_seconds / replace_retry_seconds: this one write's waits
        (None = the tracker's own). WHY: the window retries on the thread
        that draws it and passes 0, so a file still held open costs it no
        wait at all -- the write fails at once (OSError, TimeoutError among
        them) and is tried again later. Unknown result names raise
        ValueError before anything is written, as in update_status.

        append_note: text added to the row's notes (after a '; '), e.g. why
        a run is being marked failed.
        """
        unknown = sorted(k for k in kwargs if k not in RESULT_FIELDS)
        if unknown:
            raise ValueError(
                "record_result_if_started cannot record %s: not a result column"
                % ", ".join(unknown))
        if status == 'error':
            status = FAILED
        updates = {'status': status}
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds
        updates.update({k: v for k, v in kwargs.items() if v is not None})

        def mutate(rows):
            for row in rows:
                if row.get('run_id') == run_id:
                    if row.get('status') != 'started':
                        return False
                    row.update({k: str(v) for k, v in updates.items()})
                    if append_note:
                        row['notes'] = "; ".join(filter(None, [row.get('notes', ''),
                                                               append_note]))
                    return True
            return False

        saved = (self._lock_wait_seconds, self._replace_retry_seconds)
        if lock_wait_seconds is not None:
            self._lock_wait_seconds = lock_wait_seconds
        if replace_retry_seconds is not None:
            self._replace_retry_seconds = replace_retry_seconds
        try:
            return bool(self._rewrite(mutate))
        finally:
            self._lock_wait_seconds, self._replace_retry_seconds = saved

    def rate_run(self, run_id: str, rating: int, notes: Optional[str] = None) -> bool:
        """Rate a run (1-5 stars) with optional notes."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")

        def mutate(rows):
            for row in rows:
                if row.get('run_id') == run_id:
                    row['rating'] = str(rating)
                    if notes:
                        existing = row.get('notes', '')
                        row['notes'] = f"{existing}; {notes}" if existing else notes
                    return True
            return False

        # The notes are read and changed inside one locked rewrite, so a note
        # another writer added in between is never lost.
        return bool(self._rewrite(mutate))

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_run(self, run_id: str) -> Optional[Dict[str, str]]:
        """Get a single run by ID."""
        for row in self._read_all():
            if row['run_id'] == run_id:
                return row
        return None

    def search(
        self,
        sample_id: Optional[str] = None,
        run_type: Optional[str] = None,
        status: Optional[str] = None,
        rating_min: Optional[int] = None,
        limit: int = 50,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> List[Dict[str, str]]:
        """Search runs with filters."""
        rows = self._read_all()

        if sample_id:
            # Exact match. WHY: a substring match made slice S1's run history
            # list S10-S19 too, and Load Selected Run could then put another
            # slice's measurements over the image on screen.
            rows = [r for r in rows if r.get('sample_id', '') == sample_id]
        if run_type:
            rows = [r for r in rows if r.get('run_type') == run_type]
        if status:
            rows = [r for r in rows if r.get('status') == status]
        if rating_min:
            rows = [r for r in rows if r.get('rating') and int(r['rating']) >= rating_min]

        def sort_key(r):
            val = r.get(sort_by, '')
            if sort_by in ['rating', 'duration_seconds', 'det_nuclei_found']:
                try:
                    return float(val) if val else 0
                except:
                    return 0
            return val

        rows.sort(key=sort_key, reverse=descending)
        return rows[:limit]

    def get_recent(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get most recent runs."""
        return self.search(limit=limit, sort_by="created_at", descending=True)

    def get_best_for_sample(
        self, sample_id: str, run_type: str = 'detection'
    ) -> Optional[Dict[str, str]]:
        """Get the marked-best run for a specific sample."""
        rows = self._read_all()
        for row in rows:
            if (row.get('sample_id') == sample_id and
                row.get('run_type') == run_type and
                row.get('marked_best') == 'True'):
                return row

        # Fall back to highest-rated
        candidates = [r for r in rows if
                      r.get('sample_id') == sample_id and
                      r.get('run_type') == run_type and
                      r.get('rating')]
        if candidates:
            candidates.sort(key=lambda x: int(x.get('rating', 0)), reverse=True)
            return candidates[0]

        return None

    # =========================================================================
    # MARK-AS-BEST
    # =========================================================================

    def mark_as_best(self, run_id: str) -> bool:
        """
        Mark a run as the 'best' for its sample.

        Automatically clears marked_best from previous best for same sample+type.
        """
        run = self.get_run(run_id)
        if not run:
            return False

        sample_id = run.get('sample_id', '')
        run_type = run.get('run_type', '')

        # Clear the previous best for this sample+type and mark this one in
        # ONE locked rewrite (it used to be one rewrite per cleared row plus
        # one), so the move can never be left half done.
        def mutate(rows):
            found = False
            for row in rows:
                if row.get('run_id') == run_id and not found:
                    row['marked_best'] = 'True'
                    row['rating'] = '5'
                    found = True
                elif (row.get('sample_id') == sample_id and
                      row.get('run_type') == run_type and
                      row.get('marked_best') == 'True'):
                    row['marked_best'] = ''
            return found

        return bool(self._rewrite(mutate))

    def unmark_best(self, run_id: str) -> bool:
        """Clear a run's marked-best flag (its rating is left as it is).

        WHY: mark_as_best only clears the previous best of the SAME sample, so
        without this a mark on another sample could never be taken back -- and
        the 2D batch stops when marked runs of one project disagree.
        """
        if not self.get_run(run_id):
            return False
        return self._update_row(run_id, {'marked_best': ''})

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        rows = self._read_all()

        stats = {
            'total': len(rows),
            'by_type': {},
            'by_status': {},
            'by_sample': {},
            'rated': 0,
            'avg_rating': 0,
        }

        ratings = []

        for row in rows:
            run_type = row.get('run_type', 'unknown')
            status = row.get('status', 'unknown')
            sample = row.get('sample_id', 'unknown')
            rating = row.get('rating', '')

            stats['by_type'][run_type] = stats['by_type'].get(run_type, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['by_sample'][sample] = stats['by_sample'].get(sample, 0) + 1

            if rating:
                stats['rated'] += 1
                ratings.append(int(rating))

        if ratings:
            stats['avg_rating'] = sum(ratings) / len(ratings)

        return stats
