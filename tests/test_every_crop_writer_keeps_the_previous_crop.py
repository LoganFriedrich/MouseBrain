"""Every tool that writes a crop builds it aside and swaps it in when complete.

The napari crop already wrote the new crop to 2_Cropped_For_Registration.partial
and swapped it in, archiving the previous crop. Step 2's --auto-crop and
util_apply_crop_from_reference.py (--force) still wrote over the crop already
there, plane by plane: stopped part-way, the folder held old and new planes
(of different heights when the crop line moved) under the old metadata.json,
the registration then read as out of date, and the batches registered the
mixed folder unattended. Identifiers are invented; a small array stands in for
the .ims file.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
STEM = "921_PROJ_09_21_2.5x_z5"
DEPTH, ROWS, COLS = 4, 12, 10
OLD_ROWS = 9
CROP = "2_Cropped_For_Registration"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _brain(tmp_path):
    """A brain with a finished extraction and an earlier crop of OLD_ROWS rows."""
    b = tmp_path / "M9" / STEM.replace(".", "p")
    (b / "0_Raw_IMS").mkdir(parents=True)
    ims = b / "0_Raw_IMS" / (STEM + ".ims")
    ims.write_bytes(b"")
    rng = np.random.default_rng(1)
    for ch in (0, 1):
        full = b / "1_Extracted_Full" / f"ch{ch}"
        old = b / CROP / f"ch{ch}"
        full.mkdir(parents=True)
        old.mkdir(parents=True)
        for z in range(DEPTH):
            plane = rng.integers(0, 1000, size=(ROWS, COLS)).astype(np.uint16)
            tifffile.imwrite(str(full / f"Z{z:04d}.tif"), plane)
            tifffile.imwrite(str(old / f"Z{z:04d}.tif"), plane[:OLD_ROWS])
    (b / "1_Extracted_Full" / "metadata.json").write_text("{}")
    (b / CROP / "metadata.json").write_text(json.dumps({"crop_y_end": OLD_ROWS - 1, "old": True}))
    return b, ims


def _crop_is_the_old_one(b):
    for ch in (0, 1):
        planes = sorted((b / CROP / f"ch{ch}").glob("Z*.tif"))
        assert len(planes) == DEPTH
        for f in planes:
            assert tifffile.imread(str(f)).shape == (OLD_ROWS, COLS), \
                "%s/%s was overwritten by a crop that never finished" % (f.parent.name, f.name)
    assert json.loads((b / CROP / "metadata.json").read_text()).get("old") is True


def _die_after(monkeypatch, n):
    """tifffile.imwrite that stops the run (a closed window, a restart) after n planes."""
    real = tifffile.imwrite
    count = []

    def write(path, data, *a, **k):
        if len(count) >= n:
            raise KeyboardInterrupt
        count.append(path)
        return real(path, data, *a, **k)
    monkeypatch.setattr(tifffile, "imwrite", write)
    return real


# ----------------------------------------------------------- step 2 --auto-crop
def _step2(monkeypatch, new_rows):
    mod = _load("extract_staged_crop", "scripts/pipeline/2_extract_and_analyze.py")
    monkeypatch.setattr(mod, "get_ims_info", lambda path: {
        "size_x": COLS, "size_y": ROWS, "size_z": DEPTH,
        "channels": [{"index": 0, "name": "c0"}, {"index": 1, "name": "c1"}]})
    monkeypatch.setattr(mod, "detect_crop_boundary",
                        lambda volume, *a, **k: (0, new_rows - 1, np.ones(ROWS), {}))
    monkeypatch.setattr(mod, "generate_crop_qc_plot", lambda *a, **k: None)
    return mod


def test_step_2_auto_crop_stopped_part_way_leaves_the_previous_crop(tmp_path, monkeypatch):
    b, ims = _brain(tmp_path)
    step2 = _step2(monkeypatch, new_rows=6)
    real = _die_after(monkeypatch, DEPTH + 1)     # stopped in the second channel
    with pytest.raises(KeyboardInterrupt):
        step2.extract_and_analyze(ims, b, skip_crop=False)
    monkeypatch.setattr(tifffile, "imwrite", real)
    _crop_is_the_old_one(b)


def test_step_2_auto_crop_archives_the_previous_crop(tmp_path, monkeypatch):
    b, ims = _brain(tmp_path)
    step2 = _step2(monkeypatch, new_rows=6)
    step2.extract_and_analyze(ims, b, skip_crop=False)
    for ch in (0, 1):
        for f in sorted((b / CROP / f"ch{ch}").glob("Z*.tif")):
            assert tifffile.imread(str(f)).shape == (6, COLS)
    assert json.loads((b / CROP / "metadata.json").read_text())["crop"]["cropped_y"] == 6
    old = sorted((b / CROP / "_archive").glob("*/ch0/Z*.tif"))
    assert len(old) == DEPTH and tifffile.imread(str(old[0])).shape == (OLD_ROWS, COLS)
    assert list((b / CROP / "_archive").glob("*/metadata.json"))
    assert not (b / (CROP + ".partial")).exists()


# ------------------------------------------------- util_apply_crop_from_reference
def _reference_tool(monkeypatch):
    util = _load("apply_crop_reference_staged", "scripts/utils/util_apply_crop_from_reference.py")
    import mousebrain.zarr_cache as zc
    # The zarr cache is optional and its writer is tested elsewhere.
    monkeypatch.setattr(zc, "write_array", lambda *a, **k: None)
    return util


def test_crop_from_a_reference_stopped_part_way_leaves_the_previous_crop(tmp_path, monkeypatch):
    b, _ims = _brain(tmp_path)
    util = _reference_tool(monkeypatch)
    real = _die_after(monkeypatch, DEPTH + 1)
    with pytest.raises(KeyboardInterrupt):
        util.apply_crop(b / "1_Extracted_Full", b / CROP, 6, {})
    monkeypatch.setattr(tifffile, "imwrite", real)
    _crop_is_the_old_one(b)


def test_crop_from_a_reference_archives_the_previous_crop(tmp_path, monkeypatch):
    b, _ims = _brain(tmp_path)
    util = _reference_tool(monkeypatch)
    assert util.apply_crop(b / "1_Extracted_Full", b / CROP, 6, {})
    assert tifffile.imread(str(b / CROP / "ch1" / "Z0000.tif")).shape == (6, COLS)
    assert json.loads((b / CROP / "metadata.json").read_text())["crop_y_end"] == 6
    old = sorted((b / CROP / "_archive").glob("*/ch1/Z*.tif"))
    assert len(old) == DEPTH and tifffile.imread(str(old[0])).shape == (OLD_ROWS, COLS)
    assert not (b / (CROP + ".partial")).exists()


def test_crop_from_a_reference_with_nothing_to_crop_keeps_the_crop(tmp_path, monkeypatch):
    b, _ims = _brain(tmp_path)
    util = _reference_tool(monkeypatch)
    empty = tmp_path / "empty_extraction"
    empty.mkdir()
    assert util.apply_crop(empty, b / CROP, 6, {}) is False
    _crop_is_the_old_one(b)
    assert not (b / (CROP + ".partial")).exists()


# --------------------------------------------- manual crop widget's own writer
def test_the_manual_crop_widget_writer_archives_the_previous_crop(tmp_path):
    mcw = pytest.importorskip("mousebrain.plugin_3d.manual_crop_widget")
    b, _ims = _brain(tmp_path)
    mcw.save_cropped_volume(b / "1_Extracted_Full", b / CROP, 6, {})
    assert tifffile.imread(str(b / CROP / "ch0" / "Z0000.tif")).shape == (6, COLS)
    old = sorted((b / CROP / "_archive").glob("*/ch0/Z*.tif"))
    assert len(old) == DEPTH and tifffile.imread(str(old[0])).shape == (OLD_ROWS, COLS)
