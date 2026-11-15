"""Where a napari classification run is written, and how one becomes production.

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
"""

import getpass
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

PRODUCTION_FOLDER = "5_Classified_Cells"
PRODUCTION_FILES = ("cells.xml", "rejected.xml", "promoted_from.json")
PROMOTED_TAG = "promoted_to_production"


def gui_run_folder(data_summary_dir, brain_name: str, when: Optional[datetime] = None) -> Path:
    """The folder a napari classification run writes to (not created here)."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(data_summary_dir) / "optimization_runs" / brain_name / ("class_" + stamp)


def promote_to_production(run_folder, brain_folder, exp_id: Optional[str] = None,
                          tracker=None) -> dict:
    """Make a classification run the brain's production classification.

    The current production files (if any) move into
    ``5_Classified_Cells/_archive/<time>/`` -- kept, never deleted -- then the
    run's cells.xml (and rejected.xml) are copied in, with
    ``promoted_from.json`` saying which run they came from. The run's tracker
    row gets the tag ``promoted_to_production``. Region counts made before
    this are then older than cells.xml, so the status readers show them stale.

    Raises FileNotFoundError when the run has no cells.xml.
    """
    run_folder = Path(run_folder)
    prod = Path(brain_folder) / PRODUCTION_FOLDER
    if not (run_folder / "cells.xml").is_file():
        raise FileNotFoundError("no cells.xml in %s -- nothing to promote" % run_folder)
    prod.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = []
    present = [prod / n for n in PRODUCTION_FILES if (prod / n).exists()]
    if present:
        dest = prod / "_archive" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        for f in present:
            shutil.move(str(f), str(dest / f.name))
            archived.append(str(dest / f.name))

    copied = []
    for name in ("cells.xml", "rejected.xml"):
        if (run_folder / name).is_file():
            shutil.copy2(str(run_folder / name), str(prod / name))
            copied.append(name)
    try:
        who = getpass.getuser()
    except Exception:
        who = "unknown"
    record = {"exp_id": exp_id or "", "source": str(run_folder), "files": copied,
              "promoted_at": datetime.now().isoformat(timespec="seconds"),
              "promoted_by": who, "previous_production_archived_to": archived}
    (prod / "promoted_from.json").write_text(json.dumps(record, indent=1), encoding="utf-8")

    if tracker is not None and exp_id:
        try:
            tracker.add_tags(exp_id, [PROMOTED_TAG])
        except Exception as e:
            print("[!] promoted, but the tracker row could not be tagged: %s" % e)
    return record
