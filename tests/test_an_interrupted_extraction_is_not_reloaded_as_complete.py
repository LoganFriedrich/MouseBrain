"""An extraction stopped part-way is extracted again, never reused as complete.

Step 2 reuses planes already in 1_Extracted_Full instead of reading the .ims
again. It used to reuse them whenever a channel held ANY plane, taking the
depth from the files it found -- so an extraction stopped part-way came back
as a complete brain with a short stack, and metadata.json recorded the short
depth. Identifiers are invented; a small array stands in for the .ims file.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
STEM = "907_PROJ_09_07_2.5x_z5"
DEPTH, ROWS, COLS = 6, 12, 10


def _step2(monkeypatch, volume):
    spec = importlib.util.spec_from_file_location(
        "extract_step_complete", ROOT / "scripts/pipeline/2_extract_and_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_ims_info", lambda path: {
        "size_x": COLS, "size_y": ROWS, "size_z": DEPTH,
        "channels": [{"index": 0, "name": "c0"}, {"index": 1, "name": "c1"}]})
    # The .ims reader: indexing [0, channel, :, :, :] gives that channel's stack.
    reader = types.ModuleType("imaris_ims_file_reader.ims")
    reader.ims = lambda *a, **k: volume[np.newaxis]
    package = types.ModuleType("imaris_ims_file_reader")
    package.ims = reader
    monkeypatch.setitem(sys.modules, "imaris_ims_file_reader", package)
    monkeypatch.setitem(sys.modules, "imaris_ims_file_reader.ims", reader)
    return mod


def _volume():
    rng = np.random.default_rng(0)
    return rng.integers(0, 1000, size=(2, DEPTH, ROWS, COLS)).astype(np.uint16)


def _brain(tmp_path):
    b = tmp_path / "M9" / STEM.replace(".", "p")
    (b / "0_Raw_IMS").mkdir(parents=True)
    ims = b / "0_Raw_IMS" / (STEM + ".ims")
    ims.write_bytes(b"")
    return b, ims


def _planes(b, ch):
    return sorted((b / "1_Extracted_Full" / f"ch{ch}").glob("Z*.tif"))


def test_a_short_stack_left_by_an_interrupted_extraction_is_extracted_again(tmp_path, monkeypatch):
    volume = _volume()
    step2 = _step2(monkeypatch, volume)
    b, ims = _brain(tmp_path)
    # What a run stopped in channel 1 leaves: ch0 whole, ch1 three planes in.
    for ch, n in ((0, DEPTH), (1, 3)):
        folder = b / "1_Extracted_Full" / f"ch{ch}"
        folder.mkdir(parents=True)
        for z in range(n):
            tifffile.imwrite(str(folder / f"Z{z:04d}.tif"), volume[ch, z])

    full_meta, _ = step2.extract_and_analyze(ims, b, skip_crop=True)
    assert full_meta["dimensions"]["z"] == DEPTH
    for ch in (0, 1):
        assert len(_planes(b, ch)) == DEPTH
        assert np.array_equal(tifffile.imread(str(_planes(b, ch)[-1])), volume[ch, -1])
    meta = json.loads((b / "1_Extracted_Full" / "metadata.json").read_text())
    assert meta["dimensions"]["z"] == DEPTH


def test_a_stack_short_in_every_channel_is_extracted_again(tmp_path, monkeypatch):
    volume = _volume()
    step2 = _step2(monkeypatch, volume)
    b, ims = _brain(tmp_path)
    for ch in (0, 1):
        folder = b / "1_Extracted_Full" / f"ch{ch}"
        folder.mkdir(parents=True)
        for z in range(DEPTH - 2):
            tifffile.imwrite(str(folder / f"Z{z:04d}.tif"), volume[ch, z])
    full_meta, _ = step2.extract_and_analyze(ims, b, skip_crop=True)
    assert full_meta["dimensions"]["z"] == DEPTH
    assert len(_planes(b, 1)) == DEPTH


def test_a_finished_extraction_is_reused(tmp_path, monkeypatch):
    volume = _volume()
    step2 = _step2(monkeypatch, volume)
    b, ims = _brain(tmp_path)
    for ch in (0, 1):
        folder = b / "1_Extracted_Full" / f"ch{ch}"
        folder.mkdir(parents=True)
        for z in range(DEPTH):
            tifffile.imwrite(str(folder / f"Z{z:04d}.tif"), volume[ch, z])

    def must_not_read(*a, **k):
        raise AssertionError("a finished extraction was read from the .ims again")
    sys.modules["imaris_ims_file_reader.ims"].ims = must_not_read
    full_meta, _ = step2.extract_and_analyze(ims, b, skip_crop=True)
    assert full_meta["dimensions"]["z"] == DEPTH
