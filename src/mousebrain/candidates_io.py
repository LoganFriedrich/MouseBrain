"""Step 4's candidates file: kept, not overwritten, and tied to the run that wrote it.

WHY
Step 4 writes ``4_Cell_Candidates/detected_cells.xml`` (or the same name under
``from_<folder>/``) -- a fixed name. Overwriting it on every run would leave
each run's tracker row, which names only the folder, pointing at a file a later
run replaced: two runs with different settings could not be told apart on disk,
and a candidates file that reached classification and counting could have been
written by a run nobody could identify.

So, just before a successful run saves its candidates:
  * the previous ``detected_cells.xml`` (with its record, below) moves into
    ``_archive/<time>/`` in the same folder -- the pattern registration and
    production classification use. Step 5 and the status readers look only at
    the top of the folder, so an archived file is never classified;
  * the new file gets a record beside it, ``detected_cells.provenance.json``,
    naming the tracker run that wrote it;
  * step 4 sets that run's tracker ``output_path`` to the file itself, and the
    previous run's ``output_path`` to where its file now lives in the archive.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

CANDIDATES_NAME = "detected_cells.xml"
PROVENANCE_SUFFIX = ".provenance.json"


def provenance_path(xml_path) -> Path:
    xml_path = Path(xml_path)
    return xml_path.with_name(xml_path.stem + PROVENANCE_SUFFIX)


def archive_previous(folder, name: str = CANDIDATES_NAME) -> Tuple[Optional[Path], Optional[str]]:
    """Move an existing candidates file (and its record) into
    ``<folder>/_archive/<time>/``. Returns (where it went, the exp_id of the
    run that wrote it if its record says), or (None, None) if there was none."""
    folder = Path(folder)
    xml = folder / name
    if not xml.exists():
        return None, None
    prev_id = None
    rec = provenance_path(xml)
    if rec.exists():
        try:
            prev_id = json.loads(rec.read_text(encoding="utf-8")).get("exp_id")
        except (OSError, ValueError):
            prev_id = None
    dest_dir = folder / "_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.move(str(xml), str(dest))
    if rec.exists():
        shutil.move(str(rec), str(dest_dir / rec.name))
    return dest, prev_id


def write_provenance(xml_path, exp_id: Optional[str], settings: Optional[dict] = None,
                     archived_previous: Optional[Path] = None) -> Path:
    """Record beside the candidates file which run wrote it."""
    rec = provenance_path(xml_path)
    rec.write_text(json.dumps({
        "exp_id": exp_id or "",
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings or {},
        "previous_file_archived_to": str(archived_previous) if archived_previous else "",
    }, indent=1), encoding="utf-8")
    return rec


def restore(archived: Optional[Path], folder, name: str = CANDIDATES_NAME) -> bool:
    """Put an archived candidates file (and its record) back when the run that
    archived it did not write a new one -- a failed run leaves the previous
    candidates where they were. Returns True if anything was put back."""
    folder = Path(folder)
    if archived is None or (folder / name).exists() or not Path(archived).exists():
        return False
    archived = Path(archived)
    shutil.move(str(archived), str(folder / name))
    rec = provenance_path(archived)
    if rec.exists():
        shutil.move(str(rec), str(provenance_path(folder / name)))
    try:
        archived.parent.rmdir()  # only if now empty
    except OSError:
        pass
    return True
