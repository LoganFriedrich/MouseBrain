"""Step 5 archives the production classification before writing a new one,
and each tracker row names the file its run wrote.

Writing over 5_Classified_Cells/cells.xml with no copy kept would lose the
previous classification, and a step 5 row that named only the shared folder
could not be matched to its file. Identifiers are invented.
"""
import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mousebrain.config import record_registered_space
from mousebrain.tracker import ExperimentTracker
from settled_method_helpers import fake_prefilter, settle

STEP5 = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "5_classify_cells.py"
NAME = "821_PROJ_08_21_2p5x_z5"


@pytest.fixture
def step5(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("classify_step_keep", STEP5)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    csv_path = tmp_path / "tracker" / "calibration_runs.csv"
    monkeypatch.setattr(mod, "ExperimentTracker", lambda: ExperimentTracker(csv_path))
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(EOFError()))
    mod.tracker_csv = csv_path
    # Step 5 runs only the settled method (a Paradigm Best run) and always
    # pre-filters; here a full classification with a stand-in pre-filter.
    model = tmp_path / "model.h5"
    model.write_text("m")
    settle(csv_path, NAME, model)
    fake_prefilter(monkeypatch)
    return mod


def _brain(root):
    b = root / ("mouse_" + NAME) / NAME
    crop = b / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    (crop / "ch0" / "Z0000.tif").write_bytes(b"x")
    reg = b / "3_Registered_Atlas"
    reg.mkdir(parents=True)
    (reg / "brainreg.json").write_text("{}")
    record_registered_space(reg, "cropped", crop)
    det = b / "4_Cell_Candidates"
    det.mkdir()
    (det / "detected_cells.xml").write_text("<Marker></Marker>")
    return b


def _run(step5, monkeypatch, root, model, write=None, ok=True):
    def fake_classify(signal_path, candidates_path, output_path, model_path, voxel_sizes, **k):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        if write is not None:
            (Path(output_path) / "cells.xml").write_text(write)
            (Path(output_path) / "rejected.xml").write_text("rej-" + write)
        return ok, 1.0, (1 if ok else 0), 0
    monkeypatch.setattr(step5, "run_cellfinder_classify", fake_classify)
    monkeypatch.setattr(sys, "argv", ["5", "--brain", NAME, "--root", str(root)])
    step5.main()


def _rows(step5):
    # The Paradigm Best row settle() made is not one of step 5's runs.
    return [r for r in ExperimentTracker(step5.tracker_csv)._read_all()
            if r.get("exp_type") == "classification" and r.get("paradigm_best") != "True"]


def test_the_previous_production_classification_is_archived(step5, tmp_path, monkeypatch):
    root = tmp_path / "1_Brains"
    b = _brain(root)
    model = tmp_path / "model.h5"
    model.write_text("m")
    _run(step5, monkeypatch, root, model, write="first")
    _run(step5, monkeypatch, root, model, write="second")
    prod = b / "5_Classified_Cells"
    assert (prod / "cells.xml").read_text() == "second"
    archived = list((prod / "_archive").glob("*/cells.xml"))
    assert [p.read_text() for p in archived] == ["first"]


def test_each_row_names_the_file_its_run_wrote(step5, tmp_path, monkeypatch):
    root = tmp_path / "1_Brains"
    b = _brain(root)
    model = tmp_path / "model.h5"
    model.write_text("m")
    _run(step5, monkeypatch, root, model, write="first")
    _run(step5, monkeypatch, root, model, write="second")
    first, second = sorted(_rows(step5), key=lambda r: r["created_at"])
    prod = b / "5_Classified_Cells"
    assert Path(second["output_path"]) == prod / "cells.xml"
    assert (Path(first["output_path"]) / "cells.xml").read_text() == "first"
    record = json.loads((prod / "promoted_from.json").read_text())
    assert record["exp_id"] == second["exp_id"]


def test_a_failed_run_leaves_the_previous_classification_in_place(step5, tmp_path, monkeypatch):
    root = tmp_path / "1_Brains"
    b = _brain(root)
    model = tmp_path / "model.h5"
    model.write_text("m")
    _run(step5, monkeypatch, root, model, write="good")
    _run(step5, monkeypatch, root, model, write="half-written", ok=False)
    assert (b / "5_Classified_Cells" / "cells.xml").read_text() == "good"


def test_the_previous_classification_stays_in_place_while_the_run_works(
        step5, tmp_path, monkeypatch):
    """A classification takes hours, and a stop no handler sees (the process
    killed, a restart, power lost) must leave the production files where they
    were. So while cellfinder runs, the previous cells.xml is still in
    5_Classified_Cells and the run writes somewhere else."""
    root = tmp_path / "1_Brains"
    b = _brain(root)
    prod = b / "5_Classified_Cells"
    model = tmp_path / "model.h5"
    model.write_text("m")
    _run(step5, monkeypatch, root, model, write="first")
    seen = {}

    def fake_classify(signal_path, candidates_path, output_path, model_path, voxel_sizes, **k):
        seen["prod_cells"] = (prod / "cells.xml").read_text()
        seen["prod_record"] = (prod / "promoted_from.json").exists()
        seen["writes_to"] = Path(output_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        (Path(output_path) / "cells.xml").write_text("second")
        return True, 1.0, 1, 0
    monkeypatch.setattr(step5, "run_cellfinder_classify", fake_classify)
    monkeypatch.setattr(sys, "argv", ["5", "--brain", NAME, "--root", str(root)])
    step5.main()
    assert seen["prod_cells"] == "first" and seen["prod_record"]
    assert seen["writes_to"] != prod
    assert (prod / "cells.xml").read_text() == "second"


def test_files_left_by_a_hard_stopped_run_never_reach_production(step5, tmp_path, monkeypatch):
    """A hard stop leaves the run's staging folder behind. The next run keeps
    it in the archive and production only ever holds a finished run's files."""
    root = tmp_path / "1_Brains"
    b = _brain(root)
    prod = b / "5_Classified_Cells"
    model = tmp_path / "model.h5"
    model.write_text("m")
    _run(step5, monkeypatch, root, model, write="first")
    left = prod / "_in_progress" / "20000101_000000"
    left.mkdir(parents=True)
    (left / "cells.xml").write_text("half")
    assert (prod / "cells.xml").read_text() == "first"
    _run(step5, monkeypatch, root, model, write="second")
    assert (prod / "cells.xml").read_text() == "second"
    assert not (prod / "_in_progress").exists()
    stopped = list((prod / "_archive").glob("*_stopped/cells.xml"))
    assert [p.read_text() for p in stopped] == ["half"]
