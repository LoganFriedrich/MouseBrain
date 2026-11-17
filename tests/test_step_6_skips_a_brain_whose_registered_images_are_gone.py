"""Step 6 does not count a brain whose registered images are gone -- and
counts the others.

Cell positions only mean something in the images the atlas was fitted to.
When those are gone, step 6 must not count the brain against another folder,
nor fall back to its direct atlas lookup (it used to catch every error and
write direct-lookup counts). And one such brain must not abort the whole
batch, leaving every brain after it uncounted. Identifiers are invented.
"""
import contextlib
import importlib.util
import sys
from pathlib import Path

import pytest

from mousebrain.config import record_registered_space

STEP6 = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "6_count_regions.py"


@pytest.fixture
def step6(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("count_step_gone", STEP6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _brain(root, name, images_there=True):
    b = root / ("mouse_" + name) / name
    crop = b / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    if images_there:
        (crop / "ch0" / "Z0000.tif").write_bytes(b"x")
    reg = b / "3_Registered_Atlas"
    reg.mkdir(parents=True)
    (reg / "brainreg.json").write_text("{}")
    record_registered_space(reg, "cropped", crop)
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text("<Marker></Marker>")
    return b


class _Tracker:
    def log_counts(self, **k):
        return "count_1"

    @contextlib.contextmanager
    def finish_or_fail(self, exp_id):
        yield

    def update_status(self, *a, **k):
        pass

    def _read_all(self):
        return []


def test_the_batch_counts_the_other_brains_and_not_the_one_without_images(step6, tmp_path,
                                                                         monkeypatch, capsys):
    root = tmp_path / "1_Brains"
    _brain(root, "901_PROJ_09_01_2p5x_z5", images_there=False)
    _brain(root, "902_PROJ_09_02_2p5x_z5")
    counted = []

    def fake_counts(cells_path, registration_path, output_path, atlas="x", **kw):
        counted.append(Path(registration_path).parent.name)
        return False, 0.0, 0, None, ""
    monkeypatch.setattr(step6, "run_brainglobe_counts", fake_counts)
    monkeypatch.setattr(step6, "ExperimentTracker", _Tracker)
    monkeypatch.setattr(sys, "argv", ["6", "--all", "--root", str(root)])
    step6.main()
    out = capsys.readouterr().out
    assert counted == ["902_PROJ_09_02_2p5x_z5"]
    assert "Not counted" in out and "holds no images" in out


def test_the_counting_does_not_fall_back_to_direct_lookup(step6, tmp_path, monkeypatch):
    from mousebrain.config import RegisteredImagesMissing
    fell_back = []

    def transform(*a, **k):
        raise RegisteredImagesMissing("the registration was fitted to images that are gone")
    monkeypatch.setattr(step6, "run_counts_brainglobe_transform", transform)
    monkeypatch.setattr(step6, "run_counts_direct_lookup", lambda *a, **k: fell_back.append(1))
    with pytest.raises(RegisteredImagesMissing):
        step6.run_brainglobe_counts(tmp_path / "cells.xml", tmp_path / "reg", tmp_path / "out")
    assert fell_back == []


@pytest.mark.parametrize("succeeded, status", [(True, "completed"), (False, "failed")])
def test_a_counted_brain_has_its_result_on_its_tracker_row(step6, tmp_path, monkeypatch,
                                                          succeeded, status):
    """The count's result lands on its tracker row: completed with the cell
    total when the count worked, failed when it did not. (Once the skip above
    was added, the update sat unreachable after a ``continue`` and every
    successful count ended as 'failed: ended without recording a result'.)"""
    from mousebrain.tracker import ExperimentTracker
    root = tmp_path / "1_Brains"
    _brain(root, "903_PROJ_09_03_2p5x_z5")
    csv_path = tmp_path / "tracker" / "calibration_runs.csv"
    monkeypatch.setattr(step6, "ExperimentTracker", lambda: ExperimentTracker(csv_path))
    monkeypatch.setattr(step6, "run_brainglobe_counts",
                        lambda *a, **k: (succeeded, 1.5, 7 if succeeded else 0, None,
                                         "brainglobe_transform" if succeeded else ""))
    monkeypatch.setattr(sys, "argv", ["6", "--all", "--root", str(root)])
    step6.main()
    rows = [r for r in ExperimentTracker(csv_path)._read_all()
            if r.get("exp_type") == "counts"]
    assert len(rows) == 1
    assert rows[0]["status"] == status
    if succeeded:
        assert str(rows[0]["count_total_cells"]) == "7"
