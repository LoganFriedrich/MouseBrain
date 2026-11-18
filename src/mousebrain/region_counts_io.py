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


# ---------------------------------------------------------------- withdrawal
COUNT_TABLES = (("region_counts.csv", "region_counts_archive.csv"),
                ("elife_counts.csv", "elife_counts_archive.csv"))


def _read_table(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _summary_copies(summary_dir: Path, brain_folder: str) -> List[Path]:
    """The per-brain copies step 6 leaves in the summary folder,
    ``<mouse>_<brain>_counts.csv`` (an integrator imports every ``*_counts.csv``
    there)."""
    tables = {t for pair in COUNT_TABLES for t in pair}
    out = []
    for f in summary_dir.glob("*_counts.csv"):
        if f.name in tables:
            continue
        if f.name == brain_folder + "_counts.csv" or f.name.endswith("_" + brain_folder + "_counts.csv"):
            out.append(f)
    return out


# The analysis-registry name step 6 enters a brain's counts under
# (6_count_regions.REGISTRY_ANALYSIS); each entry's sample is the brain
# folder's name.
REGISTRY_ANALYSIS = "3D_Region_Counts"

_DEFAULT = object()


def _registry_manifest(registry_root) -> Optional[Path]:
    """registry.json of the region-count analysis, or None when there is none.

    The root is MouseBrain's own registry (analysis_registry.default_registry_root
    unless given). An AnalysisRegistry is never built just to look: building
    one creates its folders, and an empty ``exports/3D_Region_Counts`` would
    show up as an analysis with nothing in it to whoever mirrors the registry."""
    if registry_root is _DEFAULT:
        from mousebrain.analysis_registry import default_registry_root
        registry_root = default_registry_root()
    if not registry_root:
        return None
    manifest = Path(registry_root) / "exports" / REGISTRY_ANALYSIS / "registry.json"
    return manifest if manifest.is_file() else None


def _registry_entry_current(manifest: Optional[Path], brain_folder: str) -> bool:
    """Whether the region-count registry holds a CURRENT entry for this brain.
    A manifest that cannot be read counts as yes: invalidate() then says what
    is wrong with it, instead of the brain's entry being passed over."""
    if manifest is None:
        return False
    import json
    try:
        entries = (json.loads(manifest.read_text(encoding="utf-8")) or {}).get("entries") or {}
    except (OSError, ValueError, AttributeError):
        return True
    return any(e.get("sample") == brain_folder and e.get("is_current", True)
               for e in entries.values() if isinstance(e, dict))


def _withdraw_registry_entry(manifest: Optional[Path], brain_folder: str) -> List[str]:
    """Invalidate the brain's current region-count registry entry (its files
    move to ``_archived/<time>/`` in the registry, the entry is marked not
    current). Raises when the entry could not be invalidated, so the caller
    says so: an entry left current is mirrored by an integrator as the
    brain's counts."""
    if not _registry_entry_current(manifest, brain_folder):
        return []
    from mousebrain.analysis_registry import AnalysisRegistry
    registry = AnalysisRegistry(analysis_name=REGISTRY_ANALYSIS,
                                registry_root=manifest.parents[2],
                                follows_approved_method=False)
    registry.invalidate(sample=brain_folder, include_own_folder=True)
    if _registry_entry_current(manifest, brain_folder):
        raise RuntimeError("its entry in the analysis registry (%s) is still current: "
                           "its files could not all be moved aside (see the [!] lines "
                           "above)" % manifest)
    return ["analysis registry entry (%s)" % REGISTRY_ANALYSIS]


def _withdraw_tables(summary_dir: Path, brain_folder: str, reason: str, stamp) -> List[str]:
    """Move the brain's rows out of the count tables and its summary-folder
    copy aside. The caller holds counts_tables_lock."""
    done = []
    for table, archive in COUNT_TABLES:
        path = summary_dir / table
        if not path.exists():
            continue
        header, rows = _read_table(path)
        out = [r for r in rows if _same_brain(r.get("brain"), brain_folder)]
        if not out:
            continue
        for r in out:
            r["archived_at"] = stamp.isoformat()
            r["withdrawn_reason"] = reason
        append_archive_rows(summary_dir / archive, out)
        write_rows_atomic(path, header,
                          [r for r in rows if not _same_brain(r.get("brain"), brain_folder)])
        done.append(table)
    for copy in _summary_copies(summary_dir, brain_folder):
        aside = copy.with_name("%s.withdrawn-%s.csv" % (copy.stem, stamp.strftime("%Y%m%d_%H%M%S")))
        os.replace(copy, aside)
        done.append(copy.name)
    return done


def withdraw_counts(brain_folder: str, reason: str, summary_dir=None,
                    registry_root=_DEFAULT, lock_held: bool = False) -> List[str]:
    """Take a brain's counts out of the production tables AND out of
    MouseBrain's analysis registry, because what they were made from no
    longer stands for the brain.

    WHY: region_counts.csv and elife_counts.csv are what an integrator pulls,
    and the registry's ``3D_Region_Counts`` entries are what it mirrors. When
    a brain's registration was archived (step 3 registers it again), rejected
    by a person, or made out of date by a re-crop, or its classification was
    replaced (step 5, a promotion), its counts stayed in all three until
    somebody counted the brain again -- counts of a registration or a
    classification that no longer stands for the brain, indistinguishable
    from current ones. Withdrawing only the tables (as an earlier version
    did) left the registry entry current, so the integrator kept mirroring
    it.

    What happens:
      * the brain's row in each table moves to that table's archive
        (``archived_at`` and ``withdrawn_reason`` added; archived, never
        deleted);
      * step 6's per-brain copy in the summary folder is renamed
        ``<name>.withdrawn-<time>.csv`` beside it, so it is no longer a
        ``*_counts.csv`` an integrator imports;
      * the brain's current ``3D_Region_Counts`` registry entry is
        invalidated: its files move to the registry's ``_archived/<time>/``
        and the entry is marked not current (AnalysisRegistry.invalidate).
    The brain's own ``6_Region_Analysis`` files are left where they are (the
    status readers show them stale). Counting the brain again puts a fresh
    row and a fresh registry entry back; ``6_count_regions.py --all`` counts
    every brain that has an approved registration and no row in
    region_counts.csv, so a withdrawn brain is picked up by it.

    All of it happens while holding the count-tables lock
    (counts_tables_lock), the same lock step 6 holds while it checks that
    what it counted is still the brain's and writes its rows and registry
    entry. So a withdrawal and a count can never interleave: either the
    count lands first and is then withdrawn, or the withdrawal lands first
    and the count sees the change and writes nothing. ``lock_held``: the
    caller already holds that lock (step 6 undoing its own half-written
    count); it is not taken again (the lock is not re-entrant).

    ``brain_folder`` is the brain folder's name; a table row matches when its
    ``brain`` is that name or ``<mouse>/<that name>``. ``summary_dir``
    defaults to the configured 2_Data_Summary; ``registry_root`` to
    MouseBrain's registry root (None: no registry is looked at). Nothing is
    locked or written when the brain has no counts in any of them. Returns
    what was withdrawn. Raises when something could not be withdrawn (see
    withdraw_counts_or_warn).
    """
    if summary_dir is None:
        from mousebrain import config
        summary_dir = config.DATA_SUMMARY_DIR
    summary_dir = Path(summary_dir)
    manifest = _registry_manifest(registry_root)

    def in_tables() -> bool:
        if not summary_dir.is_dir():
            return False
        for table, _archive in COUNT_TABLES:
            path = summary_dir / table
            if path.exists() and any(_same_brain(r.get("brain"), brain_folder)
                                     for r in _read_table(path)[1]):
                return True
        return bool(_summary_copies(summary_dir, brain_folder))

    # The registry is checked whatever the tables say: a brain whose row
    # already left the tables can still have a current registry entry.
    if not in_tables() and not _registry_entry_current(manifest, brain_folder):
        return []

    def work() -> List[str]:
        done = []
        if summary_dir.is_dir():
            done += _withdraw_tables(summary_dir, brain_folder, reason, datetime.now())
        done += _withdraw_registry_entry(manifest, brain_folder)
        return done

    if lock_held or not summary_dir.is_dir():
        # No summary folder: nothing of the tables to guard (and the lock
        # would have to create the folder in a lab tree to sit in it).
        done = work()
    else:
        with counts_tables_lock(summary_dir / COUNT_TABLES[0][0]):
            done = work()
    if done:
        print("    Counts of %s withdrawn from %s (%s); count the brain again to "
              "restore them." % (brain_folder, ", ".join(done), reason))
    return done


def withdraw_counts_or_warn(brain_folder: str, reason: str, summary_dir=None,
                            registry_root=_DEFAULT) -> List[str]:
    """withdraw_counts for a step whose own work is already done: a problem
    (the tables locked by a run that does not finish, a registry file that
    would not move ...) is printed with what to do, never raised, so it
    cannot undo or hide that work."""
    try:
        return withdraw_counts(brain_folder, reason, summary_dir, registry_root)
    except Exception as e:
        print("    [!] The counts of %s could NOT all be withdrawn from region_counts.csv / "
              "elife_counts.csv / the analysis registry (%s): %s. They are out of date "
              "(%s); count the brain again (step 6) to replace them."
              % (brain_folder, REGISTRY_ANALYSIS, e, reason))
        return []


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


# How much later than its record cells.xml may be and still be the file the
# record describes. Both times are file times from the same file system (the
# record is written just after cells.xml is put in place), so only the
# file system's time resolution needs room.
RECORD_TOLERANCE_S = 5.0


def _record_exp_id(cells_xml) -> str:
    """The exp_id the production record beside cells.xml names, or "" when
    there is no readable record or the record no longer describes cells.xml
    (mousebrain.classification_runs writes the record).

    WHY the time check: anything that writes cells.xml without writing the
    record -- an older copy of step 5 still installed somewhere, a file put
    back by hand -- leaves a record naming a run that did not make the file.
    Trusting it would give counts the detection settings of a run they did
    not come from. A cells.xml modified after its record is therefore not
    described by it, and the caller falls back to the time-guarded search.
    """
    import json
    cells_xml = Path(cells_xml)
    rec = cells_xml.with_name("promoted_from.json")
    try:
        exp_id = str((json.loads(rec.read_text(encoding="utf-8")) or {}).get("exp_id") or "")
        if exp_id and cells_xml.stat().st_mtime > rec.stat().st_mtime + RECORD_TOLERANCE_S:
            return ""
        return exp_id
    except (OSError, ValueError, AttributeError):
        return ""


def detection_provenance(tracker_rows: List[dict], brain_folder: str, cells_xml,
                         candidates_hint=None) -> Optional[Dict[str, str]]:
    """The detection run behind a brain's production cells.xml, or None.

    Chain: the classification run the production record beside cells.xml
    (promoted_from.json) names; for a brain without one, the newest completed
    classification of this brain whose output is cells.xml or its folder (and
    that is not newer than cells.xml) -> its
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

    def classifying(r):
        return ok(r, "classification") or ok(r, "selective_classification")

    # The production record beside cells.xml (promoted_from.json, written by a
    # promotion and by step 5) names the run whose files these are. WHY it is
    # read first: a run promoted from Setup & Tuning wrote into its own run
    # folder, so its tracker row never names 5_Classified_Cells and the search
    # below cannot find it -- counts of a promoted classification had no
    # provenance at all. The record is the one statement of which run made
    # the file; the search is for brains classified before records existed,
    # and for a cells.xml written after its record (_record_exp_id).
    classification = None
    record_id = _record_exp_id(cells_xml)
    if record_id and record_id in by_id and classifying(by_id[record_id]):
        classification = by_id[record_id]
    if classification is None:
        cla = [r for r in tracker_rows
               if classifying(r) and _points_at(r.get("output_path"), cells_xml)]
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
