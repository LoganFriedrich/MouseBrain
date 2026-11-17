"""Writing the production region-count CSVs, and where their provenance comes from.

Used by step 6 (6_count_regions.py). Three things this module fixes:

1. THE ARCHIVE APPEND. When a brain is counted again its previous row moves to
   region_counts_archive.csv (elife_counts_archive.csv for the eLife groups).
   The append wrote the row with column names = the sorted union of every
   column seen, but never rewrote the archive's header. Region columns differ
   from brain to brain, so every row after the first was written in an order
   that was not the header's -- its values sat under the wrong names, and the
   labels cannot be recovered from the file. Now a row is written in the
   archive's own header order; when it brings new columns, the whole archive
   is rewritten with the header extended (temporary file + os.replace).

2. THE MAIN FILE was rewritten in place, so a crash part-way truncated the file
   an integrator pulls. It is now written to a temporary file and swapped in.

3. PROVENANCE. The det_* and voxel columns of region_counts.csv were never
   filled ("should come from parent experiment"). detection_provenance()
   follows the tracker from the classification that wrote cells.xml, through
   a pre-filter if there was one, to the detection run that produced the
   candidates, and returns that run's settings. When the chain cannot be
   followed the columns stay empty -- a guess would be worse than a gap -- and
   the caller says so.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p))) if p else ""


def write_rows_atomic(path, columns: List[str], rows: Iterable[dict]) -> None:
    """Write a whole CSV through a temporary file swapped in at the end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-%d" % os.getpid())
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})
    os.replace(tmp, path)


def counts_tables_lock(region_counts_csv):
    """Hold the one lock that guards the production count tables.

    WHY: region_counts.csv, elife_counts.csv and their archives are each
    read, changed and written back -- by step 6 after every brain, by
    regenerate_derived_counts, and when a brain's counts are withdrawn. Two of
    those at once (two step 6 windows, or step 6 while the derived tables are
    rebuilt) used to lose an update: both read the old file, and the second
    write dropped the first one's row. Every reader-writer of these tables
    holds this lock around its whole read-change-write. It is a file
    ``region_counts.csv.lock`` beside region_counts.csv
    (mousebrain.file_lock says how a lock left by a crashed run is taken over).
    """
    from mousebrain.file_lock import exclusive_file_lock
    return exclusive_file_lock(Path(region_counts_csv), "count tables")


def append_archive_row(archive_path, row: dict) -> Path:
    """Add one row to an archive CSV, in the archive's own column order
    (append_archive_rows)."""
    return append_archive_rows(archive_path, [row])


def append_archive_rows(archive_path, new_rows: List[dict]) -> Path:
    """Add rows to an archive CSV, in the archive's own column order.

    New columns are appended to the header (whole file rewritten through a
    temporary file); existing rows keep every value under its own name. An
    archive whose rows already have more values than its header (written by
    the old code) cannot be extended safely: it is renamed aside, kept as it
    is, a fresh archive is started with these rows, and a warning is printed.
    Returns the archive path.
    """
    archive_path = Path(archive_path)
    new_rows = list(new_rows)
    if not new_rows:
        return archive_path
    incoming = list(dict.fromkeys(k for r in new_rows for k in r))
    if not archive_path.exists():
        write_rows_atomic(archive_path, incoming, new_rows)
        return archive_path

    with open(archive_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if any(None in r for r in rows):
        aside = archive_path.with_name("%s.malformed-%s%s" % (
            archive_path.stem, datetime.now().strftime("%Y%m%d_%H%M%S"), archive_path.suffix))
        os.replace(archive_path, aside)
        print("    [!] %s has rows that do not match its header (written by an older "
              "version); kept unchanged as %s and a new archive was started."
              % (archive_path.name, aside.name))
        write_rows_atomic(archive_path, incoming, new_rows)
        return archive_path

    new_cols = [c for c in incoming if c not in header]
    if new_cols:
        write_rows_atomic(archive_path, header + new_cols, rows + new_rows)
    else:
        with open(archive_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            for row in new_rows:
                w.writerow({c: row.get(c, "") for c in header})
    return archive_path


# ----------------------------------------------------------------- provenance
def _same_brain(row_brain: str, brain_folder: str) -> bool:
    row_brain = row_brain or ""
    return row_brain == brain_folder or row_brain.endswith("/" + brain_folder)


def _points_at(row_path: str, target: Path) -> bool:
    """A tracker path names `target` itself or the folder holding it."""
    if not row_path:
        return False
    n = _norm(row_path)
    return n == _norm(target) or n == _norm(Path(target).parent)


def detection_provenance(tracker_rows: List[dict], brain_folder: str, cells_xml,
                         candidates_hint=None) -> Optional[Dict[str, str]]:
    """The detection run behind a brain's production cells.xml, or None.

    Chain: the newest completed classification of this brain whose output is
    cells.xml's folder (and that is not newer than cells.xml) -> its
    parent_experiment if that is a detection run; else its input candidates
    -> through any pre-filter run whose output they are -> the newest
    completed detection run whose output is those candidates (file or folder)
    and that started before the classification.

    Returns {'exp_id', 'preset', 'ball_xy', 'ball_z', 'soma_diameter',
    'threshold', 'voxel_xy', 'voxel_z', 'classification_exp_id', 'model'}.
    """
    cells_xml = Path(cells_xml)
    by_id = {r.get("exp_id"): r for r in tracker_rows}

    def ok(r, kind):
        return (r.get("exp_type") == kind and r.get("status") == "completed"
                and _same_brain(r.get("brain"), brain_folder))

    cla = [r for r in tracker_rows
           if (ok(r, "classification") or ok(r, "selective_classification"))
           and _points_at(r.get("output_path"), cells_xml)]
    try:
        cells_time = datetime.fromtimestamp(cells_xml.stat().st_mtime).isoformat()
    except OSError:
        cells_time = "9999"
    cla = [r for r in cla if (r.get("created_at") or "") <= cells_time]
    cla.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    classification = cla[0] if cla else None

    detection = None
    if classification is not None:
        parent = by_id.get(classification.get("parent_experiment") or "")
        if parent is not None and parent.get("exp_type") == "detection":
            detection = parent
    if detection is None:
        candidates = (classification or {}).get("input_path") or candidates_hint
        seen = set()
        while candidates and _norm(candidates) not in seen:
            seen.add(_norm(candidates))
            pf = [r for r in tracker_rows if ok(r, "prefilter")
                  and _points_at(r.get("output_path"), Path(candidates))]
            if not pf:
                break
            candidates = sorted(pf, key=lambda r: r.get("created_at") or "")[-1].get("input_path")
        before = (classification or {}).get("created_at") or "9999"
        if candidates:
            dets = [r for r in tracker_rows if ok(r, "detection")
                    and _points_at(r.get("output_path"), Path(candidates))
                    and (r.get("created_at") or "") <= before]
            dets.sort(key=lambda r: r.get("created_at") or "", reverse=True)
            detection = dets[0] if dets else None
    if detection is None:
        return None
    return {
        "exp_id": detection.get("exp_id", ""),
        "preset": detection.get("det_preset", ""),
        "ball_xy": detection.get("det_ball_xy", ""),
        "ball_z": detection.get("det_ball_z", ""),
        "soma_diameter": detection.get("det_soma_diameter", ""),
        "threshold": detection.get("det_threshold", ""),
        "voxel_xy": detection.get("voxel_xy", ""),
        "voxel_z": detection.get("voxel_z", ""),
        "classification_exp_id": (classification or {}).get("exp_id", ""),
        "model": (classification or {}).get("class_model_path", ""),
    }
