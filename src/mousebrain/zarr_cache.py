"""The ch*.zarr fast-load caches beside a brain's TIFF folders: complete or not used.

WHY
The napari widget keeps a Zarr copy of each channel (``ch0.zarr`` beside
``ch0/``) because it loads in seconds where the TIFFs take minutes. Treating
"the folder exists" as "the cache is complete", and creating the full-size
array at its final name and filling it plane by plane, would mean:
  * a conversion cut short (napari closed, a crash, a failed whole-volume save)
    leaves a cache that looks finished. Its missing planes read as zeros, and
    every later load shows -- and every detection on the loaded layers runs
    on -- a partly blank brain, with no message;
  * an empty cache folder makes the widget skip conversion for good;
  * a re-crop rewrites the TIFFs but leaves the old cache, so loads show the
    previous crop.

The rules:
  * A cache is written under ``<name>.zarr.partial`` and renamed to its real
    name only when every plane is in; then a completion record,
    ``<name>.zarr.complete.json``, is written LAST, beside it.
  * A cache is used only when that record exists, its plane count matches the
    number of TIFFs now in the folder, and no TIFF is newer than the record
    says. Anything else is "not usable": the loader reads the TIFFs instead and
    says why, and the widget rebuilds the cache.
  * Rebuilding removes the unusable cache folder first. It is a copy made from
    the TIFFs, which are never touched.
  * Every build writes into a partial folder of its OWN,
    ``<name>.zarr.partial-<random>``. Builds sharing one partial folder, each
    starting by deleting it, would let a second build of the same channel (a
    second Load, or Convert pressed during the automatic conversion) delete
    the first one's planes under it; the first could then rename a folder with
    planes missing and record it as complete. With separate folders two builds
    never touch each other's planes, and the last one to finish wins with a
    complete cache.
  * A partial folder left by a build that was cut short is removed by the
    next build of the same channel that completes, but only when nothing has
    been written into it for ABANDONED_AFTER_S (a running build writes a chunk
    every few planes, so a folder that quiet is not being built any more). It
    is unfinished machine output; the TIFFs it was made from are untouched.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

COMPLETE_SUFFIX = ".complete.json"
PARTIAL_SUFFIX = ".partial"
# A partial folder untouched this long belongs to a build that stopped.
ABANDONED_AFTER_S = 24 * 3600


def _new_partial(zarr_path: Path) -> Path:
    """A partial folder no other build uses (see the module docstring)."""
    import uuid
    return zarr_path.with_name("%s%s-%s" % (zarr_path.name, PARTIAL_SUFFIX, uuid.uuid4().hex[:12]))


def _newest_write(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def remove_abandoned_partials(zarr_path, now: Optional[float] = None) -> list:
    """Remove partial folders of this cache that no build has written to for
    ABANDONED_AFTER_S. Returns the folders removed."""
    import time
    zarr_path = Path(zarr_path)
    now = time.time() if now is None else now
    removed = []
    prefix = zarr_path.name + PARTIAL_SUFFIX
    try:
        siblings = list(zarr_path.parent.iterdir())
    except OSError:
        return removed
    for p in siblings:
        if not p.is_dir() or not (p.name == prefix or p.name.startswith(prefix + "-")):
            continue
        try:
            if now - _newest_write(p) < ABANDONED_AFTER_S:
                continue  # maybe another build still writing
            shutil.rmtree(p)
            removed.append(p)
        except OSError:
            pass
    return removed


def zarr_path_for(tiff_folder) -> Path:
    tiff_folder = Path(tiff_folder)
    return tiff_folder.parent / (tiff_folder.name + ".zarr")


def record_path(zarr_path) -> Path:
    zarr_path = Path(zarr_path)
    return zarr_path.with_name(zarr_path.name + COMPLETE_SUFFIX)


def _tiffs(tiff_folder) -> Tuple[int, float]:
    """(number of TIFFs, newest modification time) from one listing."""
    n, newest = 0, 0.0
    try:
        with os.scandir(tiff_folder) as it:
            for e in it:
                if e.name.lower().endswith((".tif", ".tiff")) and e.is_file():
                    n += 1
                    newest = max(newest, e.stat().st_mtime)
    except FileNotFoundError:
        pass
    return n, newest


def cache_state(tiff_folder, zarr_path=None) -> Tuple[bool, str]:
    """(usable, why) for the cache of this TIFF folder."""
    zarr_path = Path(zarr_path) if zarr_path else zarr_path_for(tiff_folder)
    if not zarr_path.exists():
        return False, "no cache"
    rec = record_path(zarr_path)
    if not rec.exists():
        return False, "cache has no completion record (interrupted, empty, or made before records existed)"
    try:
        info = json.loads(rec.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, "completion record unreadable: %s" % e
    n, newest = _tiffs(tiff_folder)
    if info.get("n_planes") != n:
        return False, "cache has %s planes, the folder has %d TIFFs" % (info.get("n_planes"), n)
    if newest > float(info.get("source_newest_mtime", 0)) + 2.0:
        return False, "the TIFFs were rewritten after the cache was made (a new crop?)"
    return True, "complete"


def _finish(partial: Path, zarr_path: Path, tiff_folder, shape, dtype) -> Path:
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    old_rec = record_path(zarr_path)
    if old_rec.exists():
        old_rec.unlink()
    os.replace(partial, zarr_path)
    n, newest = _tiffs(tiff_folder)
    info = {"n_planes": int(shape[0]), "shape": [int(s) for s in shape], "dtype": str(dtype),
            "source_folder": str(tiff_folder), "source_tiffs": n, "source_newest_mtime": newest,
            "completed_at": datetime.now().isoformat(timespec="seconds")}
    tmp = old_rec.with_name(old_rec.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    os.replace(tmp, old_rec)  # written LAST: its presence means complete
    remove_abandoned_partials(zarr_path)
    return zarr_path


def build(tiff_folder, zarr_path=None, progress: Optional[Callable[[int, int], None]] = None,
          read_plane=None, open_array=None) -> Path:
    """Build the cache from the TIFFs, plane by plane, under a partial name."""
    tiff_folder = Path(tiff_folder)
    zarr_path = Path(zarr_path) if zarr_path else zarr_path_for(tiff_folder)
    if read_plane is None:
        import tifffile
        read_plane = tifffile.imread
    if open_array is None:
        import zarr
        open_array = zarr.open
    files = sorted(f for f in tiff_folder.iterdir() if f.suffix.lower() in (".tif", ".tiff"))
    try:
        from natsort import natsorted
        files = natsorted(files, key=lambda f: f.name)
    except ImportError:
        pass
    if not files:
        raise FileNotFoundError("no TIFF files in %s" % tiff_folder)
    first = read_plane(str(files[0]))
    shape = (len(files),) + tuple(first.shape)
    partial = _new_partial(zarr_path)
    z = open_array(str(partial), mode="w", shape=shape, dtype=first.dtype,
                   chunks=(10,) + tuple(first.shape))
    for i, f in enumerate(files):
        z[i] = first if i == 0 else read_plane(str(f))
        if progress:
            progress(i + 1, len(files))
    return _finish(partial, zarr_path, tiff_folder, shape, first.dtype)


def write_array(volume, tiff_folder, zarr_path=None, save=None) -> Path:
    """Cache a volume already in memory (the crop tools), same rules."""
    zarr_path = Path(zarr_path) if zarr_path else zarr_path_for(tiff_folder)
    if save is None:
        import zarr
        save = zarr.save
    partial = _new_partial(zarr_path)
    save(str(partial), volume)
    return _finish(partial, zarr_path, tiff_folder, volume.shape, volume.dtype)
