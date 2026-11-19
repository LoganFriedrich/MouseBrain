"""Where a classification is written, and how one becomes production.

WHY A TUNING RUN NO LONGER WRITES 5_Classified_Cells
``<brain>/5_Classified_Cells/cells.xml`` is THE brain's classification: step 6
counts it, and every status reader treats it as "classified". The napari
Setup & Tuning widget used to write each of its classification runs straight
into that folder, so trying a model in the GUI silently replaced the production
classification (step 5's output), with no copy kept. Every tuning run's tracker
row also pointed at that one shared folder, so "load the previous run" loaded
the newest file and a comparison compared a run with itself.

Now each GUI run gets its own folder, beside the GUI's detection runs:
``<2_Data_Summary>/optimization_runs/<brain>/class_<time>/`` (cells.xml +
rejected.xml), and that folder is the run's ``output_path`` in the tracker.
Production changes only when a person promotes a run (promote_to_production,
the "Promote to production" button), which first moves the current production
files into ``5_Classified_Cells/_archive/<time>/``.

STEP 5 KEEPS THE PREVIOUS PRODUCTION CLASSIFICATION TOO
Step 5 writes the production classification itself, and it used to write over
the files already there with no copy kept, while every step 5 tracker row
named the same shared folder -- so no row could be matched to the file it
wrote. Step 5 now goes through the same helpers as a promotion:

* staging_folder: step 5 runs cellfinder into
  ``5_Classified_Cells/_in_progress/<time>/``, NOT into the production folder.
  WHY: a classification takes hours, and a stop that no handler sees (the
  process killed, the machine restarted, power or the file share lost) must
  leave the previous production files exactly where they were. Archiving them
  first and putting them back "on failure" left a brain with no cells.xml after
  any such stop, with nothing telling a person where its files had gone.
* install_staged: only once the run has finished and saved its files, in one
  short step: archive_production moves the current production files (and the
  record below) into ``5_Classified_Cells/_archive/<time>/``, the staged files
  move in (a rename on the same disk), and write_production_record writes
  ``promoted_from.json`` beside them naming the run that made them (its tracker
  exp_id), how ("step 5" or "promoted"), when, and where the previous
  production went;
* discard_staged: a run that fails moves what it wrote into
  ``_archive/<time>_failed/``; a run that was hard-stopped leaves its
  ``_in_progress/<time>/`` folder behind, and the next step 5 run of the brain
  moves it into ``_archive/<time>_stopped/`` (sweep_stopped_runs). Production
  is never touched by either.

The run that wrote the files has its tracker ``output_path`` set to the new
``cells.xml``, and the run that wrote the previous files (named by their
record) has its ``output_path`` moved to their place in the archive -- as step
4 does with candidates (mousebrain.candidates_io).

SELECTIVE CLASSIFICATION IS A PRODUCTION METHOD (2026-11-19)
Step 5 runs the method of the run marked Paradigm Best for the brain's imaging
setup (tracker.get_paradigm_classification). When that is a selective
classification, its merged result is installed as ``cells.xml`` and the
classifier's rejects as ``rejected.xml``, exactly like a full classification,
so step 6 and every status reader need nothing new. Its audit files are
installed beside them (SELECTIVE_FILES: the report with every count, the
trusted candidates, the candidates sent to the classifier) and are archived
with them when a later run replaces them. The names a selective run wrote
before it was a production method (``cells_selective.xml``,
``selective_rejected.xml``, written straight into 5_Classified_Cells beside
the production files) are archived too, so after one step 5 run the folder
holds one classification and its own audit files, never a second, older
result next to it. The record (``promoted_from.json``) carries the whole
method that made the files (``method``): which Paradigm Best run, the model
file with its size and time, every setting, the pre-filter and its settings.
"""

import getpass
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

PRODUCTION_FOLDER = "5_Classified_Cells"
# The record of which run the production files came from. The name is kept
# from when only a promotion wrote it; step 5 writes the same record.
PRODUCTION_RECORD = "promoted_from.json"
# Audit files a selective classification installs beside cells.xml.
SELECTIVE_FILES = ("selective_report.json", "trust_cells.xml", "classify_input.xml")
# Names a selective run wrote straight into 5_Classified_Cells before step 5
# made it a production method; archived with the production files.
LEGACY_SELECTIVE_FILES = ("cells_selective.xml", "selective_rejected.xml")
PRODUCTION_FILES = (("cells.xml", "rejected.xml", PRODUCTION_RECORD)
                    + SELECTIVE_FILES + LEGACY_SELECTIVE_FILES)
PROMOTED_TAG = "promoted_to_production"


def gui_run_folder(data_summary_dir, brain_name: str, when: Optional[datetime] = None) -> Path:
    """The folder a napari classification run writes to (not created here)."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(data_summary_dir) / "optimization_runs" / brain_name / ("class_" + stamp)


def _new_archive_folder(prod: Path, label: str = "") -> Path:
    """A fresh ``_archive/<time>[_label]/`` (``_2``, ``_3`` ... when two
    archives are made within the same second)."""
    stem = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_" + label if label else "")
    dest = prod / "_archive" / stem
    n = 1
    while dest.exists():
        n += 1
        dest = prod / "_archive" / ("%s_%d" % (stem, n))
    dest.mkdir(parents=True)
    return dest


def production_record(brain_folder) -> dict:
    """The brain's production record (promoted_from.json), or {} when there
    is none or it cannot be read."""
    path = Path(brain_folder) / PRODUCTION_FOLDER / PRODUCTION_RECORD
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def archive_production(brain_folder) -> Tuple[Optional[Path], Optional[str]]:
    """Move the brain's production files into ``5_Classified_Cells/_archive/<time>/``.

    Returns (the archive folder, the exp_id of the run that wrote them as
    their record names it -- None when there is no record), or (None, None)
    when there were no production files.
    """
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    present = [prod / n for n in PRODUCTION_FILES if (prod / n).exists()]
    if not present:
        return None, None
    prev_id = production_record(brain_folder).get("exp_id") or None
    dest = _new_archive_folder(prod)
    for f in present:
        shutil.move(str(f), str(dest / f.name))
    return dest, prev_id


IN_PROGRESS = "_in_progress"


def staging_folder(brain_folder) -> Path:
    """A fresh ``5_Classified_Cells/_in_progress/<time>/`` for a step 5 run to
    write into (created here). See the module docstring for why a run never
    writes the production folder directly."""
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = prod / IN_PROGRESS / stem
    n = 1
    while dest.exists():
        n += 1
        dest = prod / IN_PROGRESS / ("%s_%d" % (stem, n))
    dest.mkdir(parents=True)
    return dest


def discard_staged(staged, brain_folder, label: str = "failed") -> Optional[Path]:
    """Move a run's staging folder, whatever it holds, into
    ``_archive/<time>_<label>/`` (kept, never deleted). Production is not
    touched. Returns the archive folder, or None when there was nothing."""
    staged = Path(staged)
    if not staged.is_dir():
        return None
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    dest = _new_archive_folder(prod, label)
    for f in list(staged.iterdir()):
        shutil.move(str(f), str(dest / f.name))
    try:
        staged.rmdir()
        (prod / IN_PROGRESS).rmdir()  # only if now empty
    except OSError:
        pass
    return dest


def sweep_stopped_runs(brain_folder) -> List[Path]:
    """Move staging folders left by runs that were hard-stopped (killed,
    restarted, power lost) into ``_archive/<time>_stopped/``. Called at the
    start of a step 5 run, so a leftover never sits beside a live one. Their
    files are incomplete or unrecorded; they are kept only so nothing is
    deleted unseen."""
    in_progress = Path(brain_folder) / PRODUCTION_FOLDER / IN_PROGRESS
    if not in_progress.is_dir():
        return []
    return [d for d in (discard_staged(p, brain_folder, "stopped")
                        for p in sorted(in_progress.iterdir()) if p.is_dir()) if d is not None]


def install_staged(staged, brain_folder, exp_id: Optional[str], source, how: str = "step 5",
                   method: Optional[dict] = None
                   ) -> Tuple[Optional[Path], Optional[str], dict]:
    """Make a finished run's staged files the brain's production classification.

    In one short step: the current production files go to
    ``_archive/<time>/`` (archive_production), the staged cells.xml and
    rejected.xml move in, and the record is written. Returns (the archive
    folder or None, the previous record's exp_id or None, the new record).
    Raises FileNotFoundError, touching nothing, when the run left no cells.xml.
    """
    staged = Path(staged)
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    if not (staged / "cells.xml").is_file():
        raise FileNotFoundError("no cells.xml in %s -- nothing to install" % staged)
    archived, prev_id = archive_production(brain_folder)
    moved = []
    for name in ("cells.xml", "rejected.xml") + SELECTIVE_FILES:
        if (staged / name).is_file():
            shutil.move(str(staged / name), str(prod / name))
            moved.append(name)
    record = write_production_record(brain_folder, exp_id, source, moved, archived, how=how,
                                     method=method)
    # Anything else the run left in its staging folder is kept in the archive.
    if any(staged.iterdir()):
        discard_staged(staged, brain_folder, "run_extras")
    else:
        try:
            staged.rmdir()
            (prod / IN_PROGRESS).rmdir()
        except OSError:
            pass
    return archived, prev_id, record


def write_production_record(brain_folder, exp_id: Optional[str], source, files: List[str],
                            archived: Optional[Path], how: str,
                            method: Optional[dict] = None) -> dict:
    """Write ``promoted_from.json``: which run the production files came from,
    and (``method``, from step 5) the whole method that made them."""
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    try:
        who = getpass.getuser()
    except Exception:
        who = "unknown"
    record = {"exp_id": exp_id or "", "source": str(source), "files": list(files),
              "how": how,
              "promoted_at": datetime.now().isoformat(timespec="seconds"),
              "promoted_by": who,
              "previous_production_archived_to": (
                  [str(archived / f.name) for f in sorted(Path(archived).iterdir())]
                  if archived is not None and Path(archived).is_dir() else [])}
    if method is not None:
        record["method"] = method
    (prod / PRODUCTION_RECORD).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def follow_files_into_archive(tracker, prev_id: Optional[str], archived: Optional[Path],
                              brain_folder) -> None:
    """Point the previous production run's tracker row at its files' new place
    in the archive, so no row names a file that a later run replaced.

    Only a row that names the production folder (or a file in it) is moved: a
    promoted napari run's row names its own run folder, which still holds its
    files. Never raises: the files are already where they belong; a tracker
    problem is printed."""
    if tracker is None or not prev_id or archived is None:
        return
    import os

    def norm(p):
        return os.path.normcase(os.path.normpath(str(p)))
    prod = norm(Path(brain_folder) / PRODUCTION_FOLDER)
    try:
        row = tracker.get_experiment(prev_id) or {}
        named = norm(row.get("output_path") or "")
        if named == prod or os.path.dirname(named) == prod:
            tracker.update_status(prev_id, output_path=str(archived))
    except Exception as e:
        print("[!] the previous run's tracker row could not be pointed at the archive: %s" % e)


def follow_selective_rows(tracker, brain_folder, archived: Optional[Path]) -> List[str]:
    """Point the rows of selective classifications that wrote their files
    straight into 5_Classified_Cells (before step 5 made selective a
    production method) at the archive those files just moved to.

    Only when this archive holds such files (LEGACY_SELECTIVE_FILES), and only
    rows of this brain that name the production folder itself. Several such
    rows may name it (each overwrote the one before); all are pointed at the
    archive, which holds the files of the last of them. Returns the exp_ids
    changed; never raises (the files are already where they belong)."""
    if tracker is None or archived is None:
        return []
    if not any((Path(archived) / n).exists() for n in LEGACY_SELECTIVE_FILES):
        return []

    def norm(p):
        return os.path.normcase(os.path.normpath(str(p)))
    prod = norm(Path(brain_folder) / PRODUCTION_FOLDER)
    name = Path(brain_folder).name
    changed = []
    try:
        for row in tracker._read_all():
            brain = row.get("brain") or ""
            if (row.get("exp_type") == "selective_classification"
                    and (brain == name or brain.endswith("/" + name))
                    and norm(row.get("output_path") or "") == prod):
                tracker.update_status(row["exp_id"], output_path=str(archived))
                changed.append(row["exp_id"])
    except Exception as e:
        print("[!] earlier selective runs' tracker rows could not be pointed at the archive: %s" % e)
    return changed


def promote_to_production(run_folder, brain_folder, exp_id: Optional[str] = None,
                          tracker=None, summary_dir=None) -> dict:
    """Make a classification run the brain's production classification.

    The current production files (if any) move into
    ``5_Classified_Cells/_archive/<time>/`` -- kept, never deleted -- then the
    run's cells.xml (and rejected.xml) are copied in, with
    ``promoted_from.json`` saying which run they came from. The run's tracker
    row gets the tag ``promoted_to_production``. Region counts made before
    this are then older than cells.xml, so the status readers show them stale,
    and the brain's row is withdrawn from the production count tables
    (region_counts_io.withdraw_counts; ``summary_dir`` defaults to the
    configured 2_Data_Summary).

    Raises FileNotFoundError when the run has no cells.xml, and ValueError,
    touching nothing, when the run folder IS the production folder.
    """
    run_folder = Path(run_folder)
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    if not (run_folder / "cells.xml").is_file():
        raise FileNotFoundError("no cells.xml in %s -- nothing to promote" % run_folder)
    # Promoting the production folder onto itself would archive its files
    # and then find nothing to copy back: the brain would be left with no
    # classification. A step 5 run's tracker row names that folder, and the
    # same folder can be written two ways (drive letter or network path), so
    # this asks the file system whether they are the same folder.
    if prod.is_dir() and os.path.samefile(str(run_folder), str(prod)):
        raise ValueError("%s is the production classification already -- nothing to promote"
                         % run_folder)
    prod.mkdir(parents=True, exist_ok=True)

    archived, prev_id = archive_production(brain_folder)

    copied = []
    for name in ("cells.xml", "rejected.xml"):
        if (run_folder / name).is_file():
            # copyfile, not copy2: the copy gets the time it was written
            # (stamped by the file server). copy2 kept the run's own, older
            # time, so counts made after that run but before the promotion
            # looked newer than cells.xml and were never shown stale.
            shutil.copyfile(str(run_folder / name), str(prod / name))
            copied.append(name)
    record = write_production_record(brain_folder, exp_id, run_folder, copied, archived,
                                     how="promoted")

    # The counts were made from the classification just replaced.
    from mousebrain.region_counts_io import withdraw_counts_or_warn
    withdraw_counts_or_warn(Path(brain_folder).name,
                            "classification replaced: promoted run %s" % (exp_id or run_folder.name),
                            summary_dir)

    if tracker is not None:
        follow_files_into_archive(tracker, prev_id, archived, brain_folder)
    if tracker is not None and exp_id:
        try:
            tracker.add_tags(exp_id, [PROMOTED_TAG])
        except Exception as e:
            print("[!] promoted, but the tracker row could not be tagged: %s" % e)
    return record
