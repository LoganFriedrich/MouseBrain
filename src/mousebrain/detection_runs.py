"""Where in the stack the cells of a Setup & Tuning detection run are.

WHY THIS EXISTS
A detection test of part of the stack ("current slice +/- 25" or a Z range)
runs cellfinder on a slice of the brain that starts at plane ``z_start``, so
the Z of every cell it finds is counted from that plane, not from the first
plane of the brain. The widget added ``z_start`` only to the points it drew
right after the run; the run's file (``detected_cells.xml``) kept the slice's
own Z. Every later load of the run -- Load Selected, Load Best, the run list --
drew its cells ``z_start`` planes too shallow, on the wrong cells.

THE RULE NOW
* A run's file holds positions counted from the first plane of the WHOLE
  stack. The widget adds ``z_start`` before saving, and writes a marker,
  ``whole_stack_z.json``, into the run folder BEFORE the file, saying so.
* A run folder without that marker holds a file saved the old way. When such
  a run tested part of the stack -- tracker ``det_scope`` "partial", or, for
  rows logged before that column existed, ``notes`` beginning ``Z<start>-<end>``
  with a start above 0 -- its cells are shifted by that first plane when
  loaded, in memory only; the file is not changed by loading. Only files in a
  Setup & Tuning run folder of their own (``test_<time>``) are shifted. The data-repair script for this fix rewrites such
  files once, writing the marker, so they read the same afterwards.
* A whole-stack run started at plane 0: its positions were always right.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

MARKER_NAME = "whole_stack_z.json"


def write_whole_stack_marker(run_folder, z_start: int, z_end: Optional[int] = None,
                             note: str = "saved with whole-stack Z") -> Path:
    """Write the marker saying the run folder's file holds whole-stack Z."""
    run_folder = Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    marker = run_folder / MARKER_NAME
    marker.write_text(json.dumps({
        "z_counted_from": "first plane of the whole stack",
        "z_start_added": int(z_start or 0),
        "z_end": None if z_end is None else int(z_end),
        "note": note,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=1), encoding="utf-8")
    return marker


def has_whole_stack_marker(points_file) -> bool:
    """Whether the folder of a run's file says the file holds whole-stack Z."""
    if not points_file:
        return False
    return (Path(points_file).parent / MARKER_NAME).is_file()


def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


_NOTES_RANGE = re.compile(r"\s*Z(\d+)-(\d+)")


def run_first_plane(run: Optional[dict]) -> int:
    """The first plane a Setup & Tuning detection run tested (0 = whole stack).

    From ``det_scope``/``det_z_start`` when the row has them. Rows logged
    before those columns existed have an empty ``det_scope``; the widget has
    always written the range it tested into ``notes`` as ``Z<start>-<end>``,
    so for those rows the start is read from there."""
    if not run:
        return 0
    scope = (run.get("det_scope") or "").strip().lower()
    if scope == "partial":
        return _int(run.get("det_z_start"))
    if scope:
        return 0  # "full"
    m = _NOTES_RANGE.match(run.get("notes") or "")
    return int(m.group(1)) if m else 0


def z_shift_on_load(run: Optional[dict], points_file) -> int:
    """Planes to add to the Z of each cell read from ``points_file``, the file
    of tracker run ``run``: the first plane the run tested, for a Setup &
    Tuning test run (its file sits in a ``test_<time>`` folder of its own)
    saved before the rule above; else 0."""
    if not run or not points_file:
        return 0
    if not Path(points_file).parent.name.startswith("test_"):
        return 0  # not a Setup & Tuning test run's own folder
    if has_whole_stack_marker(points_file):
        return 0
    return run_first_plane(run)
