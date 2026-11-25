"""
atlas_fit.py - Save and log the 2D atlas fit that region counts were made with.

WHY: the Align tab holds the atlas position, orientation and transform only in
memory. Once napari closes, nothing would say which fit a set of region counts
came from, so the counts could not be checked, reproduced or compared with a
later fit. Run Quantification records the fit in two places:

- a JSON file ``<sample>_atlas_fit.json`` beside the sample's exported counts
  (only when the counts are exported -- with export off there is no results
  folder to put it beside), and
- a ``registration`` row in the slice calibration tracker, carrying the atlas,
  orientation, position, method, the fit as JSON in ``notes``, and the JSON
  file's path in ``reg_transform_path``. The ``notes`` JSON leaves out the
  two fields that speak about the COUNTS (``counts_status`` and
  ``atlas_fit_used``): whether the count finished is the row's ``status``
  column, the only thing finish_fit_record() can update.

The one rule: a fit file never claims a fit the counts beside it were not
made with. Everything below serves that rule.

The record is made in two halves, because counting runs in a background
worker that can fail after it has started:

1. begin_fit_record() -- when Run Quantification is pressed, BEFORE the
   worker can touch a count file. It captures the fit the worker is handed,
   notes the state of the count files already in the results folder, and --
   when the counts will be exported -- replaces any earlier fit file with a
   ``counts_status: in_progress`` record (keeping the earlier file's bytes in
   memory). WHY now and not later: the counts are exported in the worker
   thread, and the finished fit file is written only when the window handles
   the worker's report. If napari is closed or crashes between the two, the
   folder would otherwise hold NEW counts beside the PREVIOUS run's fit file,
   still saying ``complete``. With the in-progress record in place, that gap
   leaves a file saying "a count was started and never reported back; no fit
   is vouched for". If the earlier fit file can be neither replaced nor
   removed (for example another program holds it), the context says
   ``blocked`` and the caller must NOT count: counting would put new counts
   beside a file claiming an old fit. Only then is the tracker row logged as
   ``started``. A tracker that cannot be written (for example its CSV is open
   in a spreadsheet program) does NOT stop the fit file from being written
   later: the error is kept and reported.
2. finish_fit_record() -- when the worker reports back.
   - Success: the JSON file is written (atomically) beside the counts that
     were just exported, and the row is set to ``completed``. If the file
     cannot be written, the in-progress record is REMOVED rather than left
     claiming a count is still running.
   - Failure, count files unchanged: the row is set to ``failed`` and the
     earlier fit file is put back exactly as it was -- it still describes the
     counts in the folder (or the in-progress record is removed, when there
     was no earlier file).
   - Failure, count files changed: the folder holds counts the failed run
     partly rewrote, so no fit can be vouched for. The fit file is replaced by
     a record saying so (``counts_status: incomplete``), or removed if even
     that cannot be written. (The export itself is all-or-nothing -- see
     RegionQuantifier.export_results -- so this is a safety net, not the
     normal path.)

Counts exported WITHOUT an Align-tab fit (no atlas loaded, or the Align tab
had no labels to give) replace the file with a record that says no Align-tab
fit was used and which labels were counted with instead.

Neither function raises: every problem is returned as text so the window can
show it (nothing fails silently).
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ATLAS_FIT_SUFFIX = "_atlas_fit.json"

# The count files RegionQuantifier.export_results writes for one sample.
COUNT_FILE_SUFFIXES = ("_cells.csv", "_regions.csv", "_summary.txt")

# What the counts were assigned with when no Align-tab fit was used. These
# strings are written into the replacement fit file.
LABELS_STORED = "stored_atlas_labels"     # labels already held by the main widget
LABELS_NONE = "none_all_cells_unassigned"  # all-zero labels: every cell gets region 0

# Labels handed in by a program calling quantify_sample(): their source is
# not known to the quantifier, so no Align-tab fit can be claimed for them.
LABELS_CALLER = "supplied_by_caller"

# counts_status values written into the fit file.
COUNTS_COMPLETE = "complete"        # the counts beside this file were made with this record
COUNTS_INCOMPLETE = "incomplete"    # a failed run partly rewrote the counts
COUNTS_IN_PROGRESS = "in_progress"  # a count started and has not reported back

# Fields of the record that say what became of the COUNTS, as opposed to what
# the fit was. They belong at the top of the fit file (rewritten when the
# worker reports back) and nowhere else; see _fit_only().
_COUNT_CLAIM_FIELDS = ("counts_status", "atlas_fit_used")


def atlas_fit_path(output_dir: Path, sample_id: str) -> Path:
    """Where a sample's atlas fit is saved: beside its exported counts."""
    return Path(output_dir) / f"{sample_id}{ATLAS_FIT_SUFFIX}"


def _count_files_state(output_dir: Optional[Path], sample_id: str):
    """(size, modification time) of each count file, None where it is missing.

    WHY: a failed export may have rewritten some count files before failing.
    Comparing this before and after the run is how finish_fit_record() tells
    "the folder still holds the old counts" from "the folder holds a mix".
    """
    if output_dir is None:
        return None
    state = {}
    for suffix in COUNT_FILE_SUFFIXES:
        p = Path(output_dir) / f"{sample_id}{suffix}"
        try:
            st = p.stat()
            state[suffix] = (int(st.st_size), int(st.st_mtime_ns))
        except OSError:
            state[suffix] = None
    return state


def _ap_position_um(fit: Dict[str, Any]) -> Optional[float]:
    """The slice's anterior-posterior position in micrometers, coronal fits only.

    The tracker column holds micrometers, while the Align tab holds a slice
    index along the atlas's first axis; for a coronal slice that index times
    the atlas's AP resolution is the AP position. Other orientations have no
    AP position, so the column stays empty and the index is in the JSON.
    """
    resolution = fit.get("atlas_resolution_um")
    if fit.get("orientation") != "coronal" or not resolution:
        return None
    return float(fit.get("atlas_position", 0)) * float(resolution[0])


def _fit_only(record: Dict[str, Any]) -> Dict[str, Any]:
    """The record without its verdict on the counts: what the fit was, only.

    Used wherever the record is written down BEFORE or INSTEAD OF a finished
    count: the tracker row's notes, and the copy of the run kept inside an
    in-progress or incomplete fit file (whose own counts_status would
    otherwise sit beside a nested "complete").


    WHY not the whole record: the row is logged as 'started', before anything
    is counted, and its notes are never rewritten afterwards (only the status
    and the paths are). The record is built saying the counts are complete,
    because that is what the finished fit file must say; copied into notes it
    would say "complete" on a row whose status ends as 'failed', or stays
    'started' because the window was closed. So the notes describe the fit
    only, and the status column alone says what became of the count.
    """
    return {k: v for k, v in record.items() if k not in _COUNT_CLAIM_FIELDS}


def _write_json_atomically(path: Path, record: Dict[str, Any]) -> None:
    """Write to a temporary file in the same folder, then swap it in.

    WHY: a plain open(path, 'w') truncates the old file first, so a crash
    mid-write would leave a half-written fit file beside the counts.
    """
    _write_atomically(path, lambda f: json.dump(record, f, indent=2), "w")


def _write_bytes_atomically(path: Path, data: bytes) -> None:
    """Put a file's earlier bytes back, by the same swap-in as above."""
    _write_atomically(path, lambda f: f.write(data), "wb")


def _write_atomically(path: Path, write, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        kwargs = {} if "b" in mode else {"encoding": "utf-8"}
        with os.fdopen(fd, mode, **kwargs) as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_or_remove(path: Path, record: Dict[str, Any]) -> Optional[str]:
    """Write record to path; if that fails, remove whatever file is there.

    Returns None on success, otherwise text for the person. WHY remove: the
    file at path describes an earlier fit, and the counts beside it are no
    longer that fit's -- an absent file is honest, a stale one is not.
    """
    try:
        _write_json_atomically(path, record)
        return None
    except Exception as e:
        write_error = f"{type(e).__name__}: {e}"
    if not path.exists():
        return (f"the atlas fit file could not be written ({write_error}); "
                f"no fit file is beside the counts")
    try:
        path.unlink()
        return (f"the atlas fit file could not be written ({write_error}); "
                f"the previous fit file was removed because it describes "
                f"a different fit than these counts")
    except OSError as e:
        return (f"the atlas fit file could not be written ({write_error}) and "
                f"the previous one could not be removed ({e}). "
                f"{path} DESCRIBES A DIFFERENT FIT than the counts beside it: "
                f"delete it by hand")


def _read_previous(path: Path):
    """(bytes, parsed record or None) of the fit file at path, or (None, None)."""
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        parsed = None
    return data, parsed


def _mark_in_progress(path: Path, sample_id: str, record: Dict[str, Any],
                      previous_record) -> Dict[str, Any]:
    """Replace the fit file at path with an in-progress record.

    Returns {'blocked': bool, 'file_error': text or None}. 'blocked' is True
    only when a file is still there that was not replaced -- i.e. an earlier
    fit file that would end up beside new counts if counting went ahead.
    """
    marker = {
        "sample_id": sample_id,
        "recorded_at": datetime.now().isoformat(),
        "counts_status": COUNTS_IN_PROGRESS,
        "atlas_fit_used": None,
        "note": ("Run Quantification was started for this sample and has not "
                 "reported back. The count files in this folder may be from "
                 "this run or an earlier one, so no atlas fit is vouched for "
                 "them. If this file stays (the window was closed or crashed "
                 "while counting), run Quantification again."),
        "run_record": _fit_only(record),
        "previous_record": previous_record,
    }
    error = _write_or_remove(path, marker)
    blocked = error is not None and path.exists()
    return {"blocked": blocked, "file_error": error}


def begin_fit_record(
    fit: Optional[Dict[str, Any]],
    sample_id: str,
    tracker=None,
    input_path: Optional[str] = None,
    labels_source: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Capture what this count is about to be made with; log a 'started' row.

    Args:
        fit: AlignmentWidget.get_current_registration(), taken at the moment
            the labels are handed to the worker -- or None when the counts do
            not use an Align-tab fit.
        sample_id: the sample the counts are for.
        tracker: SliceTracker, or None when the tracker could not be opened.
            The caller warns the person in that case; nothing is logged.
        input_path: the image the fit was made on.
        labels_source: when fit is None, LABELS_STORED or LABELS_NONE.
        output_dir: the folder the counts will be exported to, or None.

    Returns:
        A context dict to hand to finish_fit_record() when the worker ends.
        Always returned, even when the tracker cannot be written; its
        'tracker_error' then says why. With output_dir set, the fit file is
        replaced by an in-progress record before this returns; if an earlier
        fit file could not be replaced or removed, 'blocked' is True,
        'file_error' says why, no tracker row is logged, and the caller must
        not count (see the module docstring).
    """
    if fit is not None:
        record = {
            "sample_id": sample_id,
            "recorded_at": datetime.now().isoformat(),
            "input_path": str(input_path) if input_path else None,
            "counts_status": COUNTS_COMPLETE,
            "atlas_fit_used": True,
            **fit,
        }
    else:
        record = {
            "sample_id": sample_id,
            "recorded_at": datetime.now().isoformat(),
            "input_path": str(input_path) if input_path else None,
            "counts_status": COUNTS_COMPLETE,
            "atlas_fit_used": False,
            "labels_source": labels_source or LABELS_NONE,
        }

    counts_before = _count_files_state(output_dir, sample_id)
    previous_bytes = None
    begin_file_error = None
    if output_dir is not None:
        # Invalidate the earlier fit file BEFORE the counts can change; see
        # the module docstring for why this cannot wait for the worker.
        path = atlas_fit_path(output_dir, sample_id)
        previous_bytes, previous_record = _read_previous(path)
        marked = _mark_in_progress(path, sample_id, record, previous_record)
        begin_file_error = marked["file_error"]
        if marked["blocked"]:
            return {"record": record, "sample_id": sample_id, "tracker": tracker,
                    "run_id": None, "tracker_error": None,
                    "output_dir": output_dir, "counts_before": counts_before,
                    "previous_bytes": previous_bytes, "blocked": True,
                    "file_error": begin_file_error}

    run_id = None
    tracker_error = None
    if fit is not None and tracker is not None:
        # Logged now as 'started' so a count that never finishes still leaves
        # a trace; finish_fit_record() sets it to completed or failed. A
        # tracker failure is kept, not raised: the fit file must still be
        # written (or the stale one cleared) once the counts are exported.
        try:
            run_id = tracker.log_registration(
                sample_id=sample_id,
                atlas=fit.get("atlas") or "",
                orientation=fit.get("orientation") or "",
                method=fit.get("method") or "",
                ap_position=_ap_position_um(fit),
                input_path=str(input_path) if input_path else None,
                notes=json.dumps(_fit_only(record), sort_keys=True),
                status="started",
            )
        except Exception as e:
            tracker_error = f"{type(e).__name__}: {e}"

    return {"record": record, "sample_id": sample_id, "tracker": tracker,
            "run_id": run_id, "tracker_error": tracker_error,
            "output_dir": output_dir, "counts_before": counts_before,
            "previous_bytes": previous_bytes, "blocked": False,
            "file_error": begin_file_error}


def finish_fit_record(
    context: Optional[Dict[str, Any]],
    success: bool,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Finish the record once the counting worker has reported back.

    Args:
        context: what begin_fit_record() returned (None: nothing to do).
        success: whether the counts were made (and, with export on, written).
        output_dir: the folder the counts were exported to, or None (then the
            folder given to begin_fit_record() is used).

    Returns:
        {'json_path': Path or None, 'run_id': str or None,
         'file_error': text or None, 'tracker_error': text or None,
         'counts_incomplete': bool}
        Never raises; the caller shows any error text to the person.
    """
    result = {"json_path": None, "run_id": None, "file_error": None,
              "tracker_error": None, "counts_incomplete": False}
    if not context:
        return result
    run_id = context.get("run_id")
    tracker = context.get("tracker")
    sample_id = context["sample_id"]
    result["run_id"] = run_id
    result["tracker_error"] = context.get("tracker_error")
    if output_dir is None:
        output_dir = context.get("output_dir")

    def _set_status(**kw):
        if tracker is None or not run_id:
            return
        try:
            tracker.update_status(run_id, **kw)
        except Exception as e:
            result["tracker_error"] = f"{type(e).__name__}: {e}"

    if not success:
        _set_status(status="failed")
        if output_dir is None:
            return result
        before = context.get("counts_before")
        after = _count_files_state(output_dir, sample_id)
        if before is None or after == before:
            # The failed run changed no count file, so the folder still holds
            # the previous run's counts: put back the fit file describing
            # them (begin_fit_record replaced it with an in-progress record).
            path = atlas_fit_path(output_dir, sample_id)
            previous = context.get("previous_bytes")
            try:
                if previous is not None:
                    _write_bytes_atomically(path, previous)
                elif path.exists():
                    path.unlink()
            except Exception as e:
                result["file_error"] = (
                    f"the previous atlas fit file could not be put back "
                    f"({type(e).__name__}: {e}); the fit file now says a count "
                    f"is in progress, so no fit is claimed for the counts in "
                    f"the folder (they are the earlier run's)")
            return result
        # The failed run rewrote some count files before failing: the folder
        # holds a mix, and no fit describes it. Say so in place of the old
        # fit file (whose fit the new files were not made with).
        result["counts_incomplete"] = True
        path = atlas_fit_path(output_dir, sample_id)
        incomplete = {
            "sample_id": sample_id,
            "recorded_at": datetime.now().isoformat(),
            "counts_status": COUNTS_INCOMPLETE,
            "atlas_fit_used": None,
            "note": ("A count failed after rewriting some of this sample's "
                     "count files. The counts in this folder may be a mix of "
                     "two runs and no atlas fit describes them. Run "
                     "Quantification again."),
            "failed_run_record": _fit_only(context["record"]),
        }
        result["file_error"] = _write_or_remove(path, incomplete)
        if result["file_error"] is None:
            result["json_path"] = path
        return result

    json_path = None
    if output_dir is not None:
        # Replaces the in-progress record begin_fit_record() wrote; if this
        # write fails, that record is removed rather than left behind.
        path = atlas_fit_path(output_dir, sample_id)
        result["file_error"] = _write_or_remove(path, context["record"])
        if result["file_error"] is None:
            json_path = path
    result["json_path"] = json_path

    _set_status(
        status="completed",
        output_path=str(json_path) if json_path else None,
        reg_transform_path=str(json_path) if json_path else None,
    )
    return result
