"""A re-crop makes the registration fitted to the earlier crop out of date.

A re-crop does not rewrite 2_Cropped_For_Registration in place (an
interrupted one would mix old and new planes under the old metadata). And the
registration fitted to the earlier crop must stop counting as "registered"
and "approved": otherwise step 4's approval gate would let detection run on
images the atlas was never fitted to. Identifiers are invented.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space
from mousebrain.pipeline_status import brain_status

ROOT = Path(__file__).resolve().parents[1]
BRAIN = "905_PROJ_09_05_2p5x_z5"
T0 = 1_700_000_000
CROP = "2_Cropped_For_Registration"


def _write_planes(folder, n, rows, value):
    (folder / "ch0").mkdir(parents=True, exist_ok=True)
    for z in range(n):
        tifffile.imwrite(str(folder / "ch0" / ("Z%04d.tif" % z)),
                         np.full((rows, 4), value, dtype=np.uint16))


def _approved_brain(tmp_path):
    b = tmp_path / "M9" / BRAIN
    _write_planes(b / "1_Extracted_Full", 3, 10, 7)
    (b / "1_Extracted_Full" / "metadata.json").write_text("{}")
    _write_planes(b / CROP, 3, 8, 7)
    (b / CROP / "metadata.json").write_text(json.dumps({"crop_y_end": 8}))
    for f in (b / CROP / "ch0").iterdir():
        os.utime(f, (T0, T0))
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", b / CROP)
    for n in ra.FINGERPRINT_FILES:
        (reg / n).write_bytes(b"r")
        os.utime(reg / n, (T0 + 100, T0 + 100))
    ra.approve(reg, approved_by="tester")
    return b, reg


def _recrop(b, t):
    for f in (b / CROP / "ch0").iterdir():
        os.utime(f, (t, t))


def test_a_registration_older_than_its_images_is_out_of_date(tmp_path):
    b, reg = _approved_brain(tmp_path)
    assert brain_status(b)["approved"]
    _recrop(b, T0 + 5000)
    s = brain_status(b)
    assert not s["registered"] and not s["approved"]
    assert s["step"] == "register"
    assert "register" in s["stale"] and s["register_problem"]
    # Step 4's gate and every other "is it approved?" reader.
    assert not ra.is_approved(reg)
    assert ra.approval_state(reg)[0] == ra.OUT_OF_DATE
    with pytest.raises(ValueError):
        ra.approve(reg, approved_by="tester")


def test_a_registration_whose_images_are_gone_is_out_of_date(tmp_path):
    b, reg = _approved_brain(tmp_path)
    for f in (b / CROP / "ch0").iterdir():
        f.unlink()
    s = brain_status(b)
    assert s["step"] == "register" and not s["approved"]
    assert "gone" in s["register_problem"]


def test_counts_on_an_out_of_date_registration_are_stale(tmp_path):
    b, reg = _approved_brain(tmp_path)
    (b / "6_Region_Analysis").mkdir()
    counts = b / "6_Region_Analysis" / "cell_counts_by_region.csv"
    counts.write_text("a\n")
    os.utime(counts, (T0 + 9000, T0 + 9000))
    _recrop(b, T0 + 5000)
    s = brain_status(b)
    assert "count" in s["stale"] and not s["counted"]


def test_step_3_registers_an_out_of_date_registration_again(tmp_path):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    spec = importlib.util.spec_from_file_location(
        "register_out_of_date", ROOT / "scripts/pipeline/3_register_to_atlas.py")
    step3 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = step3
    spec.loader.exec_module(step3)
    b, reg = _approved_brain(tmp_path)
    assert step3.check_registration_status(b)[0] == "complete"
    _recrop(b, T0 + 5000)
    assert step3.check_registration_status(b)[0] == "out_of_date"


# ------------------------------------------------------------------ re-crop
def _crop_tool():
    spec = importlib.util.spec_from_file_location(
        "crop_subprocess_test", ROOT / "src/mousebrain/plugin_3d/crop_subprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ask_for_crop(b, y):
    (b / ".crop_position.json").write_text(json.dumps(
        {"y_crop": y, "rows_kept": y, "total_rows": 10}))


def test_a_re_crop_keeps_the_previous_crop(tmp_path):
    b, _reg = _approved_brain(tmp_path)
    _ask_for_crop(b, 5)
    assert _crop_tool().do_crop(b)
    assert tifffile.imread(str(b / CROP / "ch0" / "Z0000.tif")).shape == (5, 4)
    assert json.loads((b / CROP / "metadata.json").read_text())["crop_y_end"] == 5
    old = list((b / CROP / "_archive").glob("*/ch0/Z0000.tif"))
    assert len(old) == 1 and tifffile.imread(str(old[0])).shape == (8, 4)
    assert not (b / (CROP + ".partial")).exists()
    assert brain_status(b)["step"] == "register"  # the registration was for the old crop


def test_an_interrupted_re_crop_leaves_the_previous_crop_untouched(tmp_path, monkeypatch):
    b, _reg = _approved_brain(tmp_path)
    _ask_for_crop(b, 5)
    tool = _crop_tool()
    written = []
    real_write = tifffile.imwrite

    def write_then_die(path, data, *a, **k):
        if len(written) == 1:
            raise KeyboardInterrupt  # the window is closed part-way
        written.append(path)
        return real_write(path, data, *a, **k)
    monkeypatch.setattr(tifffile, "imwrite", write_then_die)
    with pytest.raises(KeyboardInterrupt):
        tool.do_crop(b)
    monkeypatch.setattr(tifffile, "imwrite", real_write)
    for f in sorted((b / CROP / "ch0").glob("Z*.tif")):
        assert tifffile.imread(str(f)).shape == (8, 4), f.name
    assert json.loads((b / CROP / "metadata.json").read_text())["crop_y_end"] == 8
    assert brain_status(b)["approved"]  # nothing changed for the registration either
