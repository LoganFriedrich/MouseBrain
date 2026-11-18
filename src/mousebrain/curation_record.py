"""The curation answers a person gave for ONE brain, and the training cubes cut from them.

WHY THIS EXISTS
Curation (Setup & Tuning, tab "Curate/Train") asks a person, candidate by
candidate, "is this a cell?". Each answer is training data for a classifier,
so each answer must be kept exactly once, with the label the person gave
LAST, and must be tied to the brain whose image it was given on. Before this
module:

* every answer was appended to one pooled pair of files per imaging setup
  (``<models>/<setup>/training_data/curated_cells.xml`` and
  ``curated_non_cells.xml``) with no brain recorded, so the export later cut
  cubes for every brain's coordinates out of whichever image was on screen,
  and wrote them under names without a brain, over the cubes of other brains;
* a changed answer (cell, then not a cell) stayed in BOTH files, and a
  repeated answer was written twice, so the classifier was trained on the
  same point with both labels;
* the files were rewritten in place, so an interrupted save could leave an
  unreadable file, and an unreadable file was then silently replaced by an
  empty one -- every earlier answer lost.

WHAT HOLDS NOW
* Each brain has its own folder inside its imaging setup's pool:
  ``<models>/<setup>/training_data/<brain>/`` holding ``curated_cells.xml``,
  ``curated_non_cells.xml`` (the answers), ``cells/`` and ``non_cells/``
  (the cubes cut from that brain's own image) and ``training.yml``.
* A point (z, y, x, whole voxels) is in at most one of the two files, once.
  ``set_label`` moves it when the answer changes; ``add_unlabelled`` adds only
  points that have no answer yet (used for suggestions and old batches, which
  must never overwrite what a person answered).
* Both files are written to a temporary file first and swapped in, and a file
  that exists but cannot be read stops the save with an error instead of
  being replaced.
* The pool's own ``training.yml`` lists every brain folder's cube folders, so
  training on "the pool" trains on every brain's cubes, each cut from its own
  brain. Cubes that an older version left directly in the pool's ``cells/``
  and ``non_cells/`` carry no brain and are still listed (they are what older
  models were trained on); which brain they came from cannot be told from
  their names.

Coordinates are compared as whole voxels with ``int()``, exactly as they are
written to the XML, so a point read back from the file matches the
candidate it was saved from.
"""

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LABELS = ("cells", "non_cells")
XML_NAMES = {"cells": "curated_cells.xml", "non_cells": "curated_non_cells.xml"}
POOL_FOLDER = "training_data"

# BrainGlobe training cubes: 50 x 50 voxels in-plane, 20 planes deep.
CUBE_XY = 50
CUBE_Z = 20

_CUBE_NAME = re.compile(r"^pCellz(-?\d+)y(-?\d+)x(-?\d+)Ch(\d+)\.tiff?$")


class CurationRecordError(Exception):
    """A saved answers file exists but cannot be read; nothing was changed."""


def _other(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, not {label!r}")
    return LABELS[1] if label == LABELS[0] else LABELS[0]


def point_key(coord) -> Tuple[int, int, int]:
    """(z, y, x) as whole voxels, the way the answers files store them."""
    return int(coord[0]), int(coord[1]), int(coord[2])


def imaging_setup(brain_name: str) -> str:
    """The imaging setup a brain's answers are pooled under ('default' if unknown)."""
    from mousebrain.config import parse_brain_name
    try:
        parsed = parse_brain_name(brain_name) or {}
    except Exception:
        parsed = {}
    return parsed.get("imaging_params") or "default"


def pool_dir(models_dir, brain_name: str) -> Path:
    """``<models>/<setup>/training_data``: the pool that brain's folder sits in."""
    return Path(models_dir) / imaging_setup(brain_name) / POOL_FOLDER


def brain_folder(models_dir, brain_name: str) -> Path:
    """``<models>/<setup>/training_data/<brain>``: this brain's answers and cubes."""
    return pool_dir(models_dir, brain_name) / brain_name


# --------------------------------------------------------------------------
# The answers files
# --------------------------------------------------------------------------

def read_points(xml_path) -> List[Tuple[int, int, int]]:
    """Points (z, y, x) in a curated XML file, in file order, without repeats.

    A missing file holds no points. A file that exists but cannot be parsed
    raises CurationRecordError: treating it as empty would let the next save
    overwrite every answer in it.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError) as e:
        raise CurationRecordError(
            f"Cannot read the saved answers in {xml_path}: {e}. Nothing was "
            f"changed; move the damaged file aside (it may be partly "
            f"recoverable by hand) and answer again.") from e
    seen, out = set(), []
    for marker in root.iter("Marker"):
        try:
            x = int(marker.find("MarkerX").text)
            y = int(marker.find("MarkerY").text)
            z = int(marker.find("MarkerZ").text)
        except (AttributeError, TypeError, ValueError):
            continue
        key = (z, y, x)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _xml_bytes(points: Sequence[Tuple[int, int, int]]) -> bytes:
    root = ET.Element("CellCounter_Marker_File")
    props = ET.SubElement(root, "Image_Properties")
    ET.SubElement(props, "Image_Filename").text = "training_data"
    data = ET.SubElement(root, "Marker_Data")
    ET.SubElement(data, "Current_Type").text = "1"
    mtype = ET.SubElement(data, "Marker_Type")
    ET.SubElement(mtype, "Type").text = "1"
    for z, y, x in points:
        m = ET.SubElement(mtype, "Marker")
        ET.SubElement(m, "MarkerX").text = str(x)
        ET.SubElement(m, "MarkerY").text = str(y)
        ET.SubElement(m, "MarkerZ").text = str(z)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    import io
    buf = io.BytesIO()
    tree.write(buf, encoding="UTF-8", xml_declaration=True)
    return buf.getvalue()


def _write_whole(path: Path, payload: bytes) -> None:
    """Write ``payload`` to a temporary file beside ``path`` and swap it in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class CurationRecord:
    """The answers for one brain: its two curated XML files in its own folder."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def xml_path(self, label: str) -> Path:
        _other(label)
        return self.folder / XML_NAMES[label]

    @property
    def cells_xml(self) -> Path:
        return self.xml_path("cells")

    @property
    def non_cells_xml(self) -> Path:
        return self.xml_path("non_cells")

    def read(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """{'cells': [...], 'non_cells': [...]}, each point once.

        A point found in BOTH files (left there by an older version) belongs
        to neither until a person answers it again: it is left out here, so it
        is neither trained on nor counted as reviewed, and ``conflicts()``
        names it.
        """
        cells = read_points(self.cells_xml)
        non = read_points(self.non_cells_xml)
        both = set(cells) & set(non)
        return {"cells": [p for p in cells if p not in both],
                "non_cells": [p for p in non if p not in both]}

    def conflicts(self) -> List[Tuple[int, int, int]]:
        """Points that sit in both files (answered both ways by an older version)."""
        cells = read_points(self.cells_xml)
        non = set(read_points(self.non_cells_xml))
        return [p for p in cells if p in non]

    def _write(self, cells, non) -> None:
        # Both files are rebuilt from the full lists and swapped in one after
        # the other; the point being moved is removed from the file it leaves
        # first, so an interruption between the two swaps loses that one
        # answer (it is asked again) rather than leaving it under both labels.
        self.folder.mkdir(parents=True, exist_ok=True)
        old_cells = set(read_points(self.cells_xml))
        first = ("non_cells", "cells") if len(old_cells - set(cells)) == 0 else ("cells", "non_cells")
        lists = {"cells": cells, "non_cells": non}
        for label in first:
            _write_whole(self.xml_path(label), _xml_bytes(lists[label]))

    def set_label(self, coords: Iterable, label: str) -> Dict[str, int]:
        """Record a person's answer ``label`` for each point; the last answer wins.

        A point that had the other label is moved (taken out of the other
        file); a point that already has this label is not written again.
        Returns counts: added, moved, unchanged.
        """
        other = _other(label)
        cells = read_points(self.cells_xml)
        non = read_points(self.non_cells_xml)
        lists = {"cells": cells, "non_cells": non}
        mine, theirs = lists[label], lists[other]
        mine_set, theirs_set = set(mine), set(theirs)
        added = moved = unchanged = 0
        for c in coords:
            k = point_key(c)
            if k in theirs_set:
                theirs_set.discard(k)
                moved += 1
                if k not in mine_set:
                    mine.append(k)
                    mine_set.add(k)
            elif k in mine_set:
                unchanged += 1
            else:
                mine.append(k)
                mine_set.add(k)
                added += 1
        lists[other] = [p for p in theirs if p in theirs_set]
        lists[label] = mine
        if added or moved:
            self._write(lists["cells"], lists["non_cells"])
        return {"added": added, "moved": moved, "unchanged": unchanged}

    def add_unlabelled(self, coords: Iterable, label: str) -> Dict[str, int]:
        """Give ``label`` only to points that have no answer yet.

        For suggestions (pre-filter results) and old answer batches: they
        never overwrite an answer a person gave. Returns counts: added,
        already_answered.
        """
        _other(label)
        cells = read_points(self.cells_xml)
        non = read_points(self.non_cells_xml)
        known = set(cells) | set(non)
        lists = {"cells": cells, "non_cells": non}
        added = already = 0
        for c in coords:
            k = point_key(c)
            if k in known:
                already += 1
                continue
            known.add(k)
            lists[label].append(k)
            added += 1
        if added:
            self._write(lists["cells"], lists["non_cells"])
        return {"added": added, "already_answered": already}

    def answers_for(self, candidates) -> Tuple[Set[int], Set[int]]:
        """(indices answered 'cell', indices answered 'not a cell') among ``candidates``.

        Lets a review carry on where it stopped -- after the layer is chosen
        again or napari is restarted -- instead of asking every candidate again.
        """
        labels = self.read()
        cells, non = set(labels["cells"]), set(labels["non_cells"])
        confirmed, rejected = set(), set()
        if candidates is None:
            return confirmed, rejected
        for i, c in enumerate(candidates):
            k = point_key(c)
            if k in cells:
                confirmed.add(i)
            elif k in non:
                rejected.add(i)
        return confirmed, rejected


# --------------------------------------------------------------------------
# Cubes and training.yml
# --------------------------------------------------------------------------

def cube_name(key, channel: int) -> str:
    z, y, x = key
    return f"pCellz{z}y{y}x{x}Ch{channel}.tif"


def write_brain_cubes(folder, labels: Dict[str, List[Tuple[int, int, int]]],
                      signal, background=None) -> Dict[str, int]:
    """Cut the cubes for one brain's answers from THAT brain's image.

    ``signal`` / ``background`` are the brain's own arrays (z, y, x). Writes
    ``<folder>/cells`` and ``<folder>/non_cells``. A cube already there for a
    point that no longer has that label (the answer changed, or the point is
    in both files) is removed, so the folders always match the answers. Points
    too close to the image edge for a whole cube are skipped and counted.
    """
    import numpy as np
    import tifffile

    folder = Path(folder)
    sig = np.asarray(signal)
    bg = np.asarray(background) if background is not None else None
    z_max, y_max, x_max = sig.shape
    hz, hxy = CUBE_Z // 2, CUBE_XY // 2
    out = {"cells": 0, "non_cells": 0, "skipped": 0, "removed": 0}
    for label in LABELS:
        d = folder / label
        d.mkdir(parents=True, exist_ok=True)
        wanted = set(labels.get(label, []))
        for f in d.iterdir():
            m = _CUBE_NAME.match(f.name)
            if m and (int(m.group(1)), int(m.group(2)), int(m.group(3))) not in wanted:
                f.unlink()
                out["removed"] += 1
        for key in labels.get(label, []):
            z, y, x = key
            if (z - hz < 0 or z + hz > z_max or y - hxy < 0 or y + hxy > y_max
                    or x - hxy < 0 or x + hxy > x_max):
                out["skipped"] += 1
                continue
            sl = (slice(z - hz, z + hz), slice(y - hxy, y + hxy), slice(x - hxy, x + hxy))
            tifffile.imwrite(str(d / cube_name(key, 0)), sig[sl].astype(np.uint16))
            if bg is not None:
                tifffile.imwrite(str(d / cube_name(key, 1)), bg[sl].astype(np.uint16))
            out[label] += 1
    return out


def _has_cubes(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.tif*"))


def pool_cube_dirs(td) -> List[Tuple[str, Path]]:
    """(label, folder) for every cube folder of a pool or of one brain's folder.

    Covers ``<td>/cells``, ``<td>/non_cells`` (cubes from older versions, no
    brain) and ``<td>/<brain>/cells``, ``<td>/<brain>/non_cells``.
    """
    td = Path(td)
    found = [(label, td / label) for label in LABELS]
    if td.is_dir():
        for sub in sorted(td.iterdir()):
            if sub.is_dir() and sub.name not in LABELS:
                found.extend((label, sub / label) for label in LABELS)
    return found


def count_cubes(td) -> Tuple[int, int]:
    """(cell cubes, non-cell cubes) under a pool or brain folder (signal channel counted once per file)."""
    n = {"cells": 0, "non_cells": 0}
    for label, d in pool_cube_dirs(td):
        if d.is_dir():
            n[label] += len(list(d.glob("*.tif*")))
    return n["cells"], n["non_cells"]


def write_training_yaml(td) -> Optional[Path]:
    """Write ``<td>/training.yml`` naming every cube folder under ``td`` that holds cubes.

    Used for a pool (one entry pair per brain folder, plus the older brainless
    folders) and for a single brain's folder. Returns None (and removes a
    stale file) when there are no cubes.
    """
    import yaml

    td = Path(td)
    entries = []
    for label, d in pool_cube_dirs(td):
        if _has_cubes(d):
            entries.append({"bg_channel": 1, "cell_def": "", "cube_dir": str(d),
                            "signal_channel": 0,
                            "type": "cell" if label == "cells" else "no_cell"})
    path = td / "training.yml"
    if not entries:
        if path.exists():
            path.unlink()
        return None
    _write_whole(path, yaml.safe_dump({"data": entries}, default_flow_style=False).encode("utf-8"))
    return path
