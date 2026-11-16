"""registered_from.json names images a FINISHED registration was fitted to.

Step 3 used to write the record before brainreg started, so a failed run left a
record naming images no registration had been fitted to. And when a recorded
folder held no images, config.registered_input quietly returned the caller's
default folder instead -- another image space, in which every cell position is
wrong while the counts still look plausible. Identifiers are invented.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import (REGISTERED_SPACE_FILE,
                               record_registered_space, registered_input)

ROOT = Path(__file__).resolve().parents[1]
BRAIN = "902_PROJ_09_02_2p5x_z5"


def _images(folder):
    (folder / "ch0").mkdir(parents=True)
    (folder / "ch0" / "Z0000.tif").write_bytes(b"x")
    (folder / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')


def _brain(tmp_path):
    b = tmp_path / "M9" / BRAIN
    _images(b / "1_Extracted_Full")
    _images(b / "2_Cropped_For_Registration")
    return b


def test_a_recorded_folder_without_images_is_an_error_not_a_fallback(tmp_path):
    b = _brain(tmp_path)
    record_registered_space(b / "3_Registered_Atlas", "full", b / "1_Extracted_Full")
    for f in (b / "1_Extracted_Full" / "ch0").iterdir():
        f.unlink()
    with pytest.raises(FileNotFoundError) as e:  # config.RegisteredImagesMissing
        registered_input(b)  # the crop has images: it must NOT be returned instead
    assert "1_Extracted_Full" in str(e.value)


def test_an_unrecorded_registration_still_falls_back(tmp_path):
    b = _brain(tmp_path)
    assert registered_input(b)[1] == b / "2_Cropped_For_Registration"


def _step3(monkeypatch):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    spec = importlib.util.spec_from_file_location("register_space_record",
                                                  ROOT / "scripts/pipeline/3_register_to_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # No tracker row, no atlas download for the orientation picture, no QC picture.
    monkeypatch.setattr(mod, "TRACKER_AVAILABLE", False)
    monkeypatch.setattr(mod, "draw_registration_qc", lambda *a, **k: None)
    import mousebrain.orientation_check as oc

    def no_picture(*a, **k):
        raise RuntimeError("not drawn in tests")
    monkeypatch.setattr(oc, "orientation_check", no_picture)
    return mod


class _Result:
    def __init__(self, code):
        self.returncode = code


def test_a_failed_brainreg_leaves_no_record_of_a_registered_space(tmp_path, monkeypatch):
    step3 = _step3(monkeypatch)
    b = _brain(tmp_path)

    def failing(cmd, **k):
        (Path(cmd[2]) / "brainreg.json").write_text("{}")  # what brainreg writes first
        return _Result(1)
    monkeypatch.setattr(step3.subprocess, "run", failing)
    with pytest.raises(RuntimeError):
        step3.run_brainreg(b, {})
    assert not (b / "3_Registered_Atlas" / REGISTERED_SPACE_FILE).exists()


def test_a_finished_registration_records_its_images(tmp_path, monkeypatch):
    step3 = _step3(monkeypatch)
    b = _brain(tmp_path)

    def finishing(cmd, **k):
        for name in ra.FINGERPRINT_FILES:
            (Path(cmd[2]) / name).write_text("{}")
        return _Result(0)
    monkeypatch.setattr(step3.subprocess, "run", finishing)
    assert step3.run_brainreg(b, {})
    assert (b / "3_Registered_Atlas" / REGISTERED_SPACE_FILE).exists()
    assert registered_input(b)[1] == b / "2_Cropped_For_Registration"
