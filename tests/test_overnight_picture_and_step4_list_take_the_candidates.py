"""The overnight sample picture and step 4's brain list read "the candidates"
by the one rule (pipeline_status.find_candidates), like step 5.

The overnight picture took the first .xml the folder listing gave at the top
of 4_Cell_Candidates, and step 4's brain list called a brain "detected" when
any .xml sat there. A brain registered on its full extraction keeps the run
step 5 classifies in 4_Cell_Candidates/from_1_Extracted_Full: the picture drew
no cells and the list said "not detected", while step 5 classified that run.
"""
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
BRAIN = "101_PROJ_01_02_2p5x_z5"


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, PIPELINE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _images(folder: Path, n_z=8):
    import tifffile
    (folder / "ch0").mkdir(parents=True)
    for z in range(n_z):
        tifffile.imwrite(str(folder / "ch0" / ("Z%04d.tif" % z)),
                         (np.arange(64, dtype=np.uint16).reshape(8, 8) + z))
    (folder / "metadata.json").write_text(json.dumps({"voxel_size_um": {"x": 4, "y": 4, "z": 4}}),
                                          encoding="utf-8")


def _markers(zs):
    rows = "".join("<Marker><MarkerX>3</MarkerX><MarkerY>4</MarkerY><MarkerZ>%d</MarkerZ></Marker>\n" % z
                   for z in zs)
    return ("<CellCounter_Marker_File><Marker_Data><Marker_Type>\n%s</Marker_Type>"
            "</Marker_Data></CellCounter_Marker_File>\n" % rows)


def _brain_registered_on_full(root: Path) -> Path:
    brain = root / "mouse_a" / BRAIN
    (brain / "0_Raw_IMS").mkdir(parents=True)
    (brain / "0_Raw_IMS" / (BRAIN + ".ims")).write_bytes(b"x")
    _images(brain / "1_Extracted_Full")
    reg = brain / "3_Registered_Atlas"
    reg.mkdir()
    (reg / "brainreg.json").write_text("{}", encoding="utf-8")
    (reg / "registered_from.json").write_text(json.dumps({"source": "full"}), encoding="utf-8")
    run = brain / "4_Cell_Candidates" / "from_1_Extracted_Full"
    run.mkdir(parents=True)
    (run / "detected_cells.xml").write_text(_markers([4, 4, 4]), encoding="utf-8")
    return brain


@pytest.fixture
def overnight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the module opens its log file in the working folder
    mod = _load("overnight_batch.py", "overnight_picture_under_test")
    yield mod
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler) and str(tmp_path) in h.baseFilename:
            logging.root.removeHandler(h)
            h.close()


def test_the_overnight_picture_draws_the_run_step_5_classifies(overnight, tmp_path, monkeypatch):
    from mousebrain.pipeline_status import find_candidates
    brain = _brain_registered_on_full(tmp_path / "1_Brains")
    expected = brain / "4_Cell_Candidates" / "from_1_Extracted_Full" / "detected_cells.xml"
    assert find_candidates(brain) == expected  # what step 5 classifies

    read = []
    real_parse = overnight.parse_cells_from_xml

    def spy(path):
        read.append(Path(path))
        return real_parse(path)
    monkeypatch.setattr(overnight, "parse_cells_from_xml", spy)
    titles = []
    import matplotlib.axes
    real_title = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                        lambda self, t, *a, **k: (titles.append(t), real_title(self, t, *a, **k))[1])

    out = tmp_path / "pictures"
    out.mkdir()
    made = overnight.generate_cell_detection_images(brain, out)
    assert made, "no picture was made"
    assert read == [expected]
    # The middle picture (Z=4) shows that run's three cells.
    assert any("Z=4 (3 cells)" in t for t in titles), titles


def test_step_4_lists_a_brain_whose_run_is_kept_in_registered_space_as_detected(tmp_path):
    root = tmp_path / "1_Brains"
    _brain_registered_on_full(root)
    step4 = _load("4_detect_cells.py", "detect_list_under_test")
    listed = {b["pipeline"].name: b for b in step4.list_available_brains(root)}
    assert listed[BRAIN]["detected"] is True


def test_step_4_does_not_count_a_run_outside_registered_space_as_detected(tmp_path):
    """Registered on the crop, a trial run on the full extraction is not what
    step 5 classifies, so the brain is not detected."""
    root = tmp_path / "1_Brains"
    brain = _brain_registered_on_full(root)
    _images(brain / "2_Cropped_For_Registration")
    (brain / "3_Registered_Atlas" / "registered_from.json").write_text(
        json.dumps({"source": "cropped"}), encoding="utf-8")
    step4 = _load("4_detect_cells.py", "detect_list_under_test2")
    listed = {b["pipeline"].name: b for b in step4.list_available_brains(root)}
    assert listed[BRAIN]["detected"] is False
