"""A brain's counts leave the production count tables as soon as what they
were made from is replaced.

When step 3 archived a brain's registration, or its classification was
replaced (step 5, a promotion), its row stayed in region_counts.csv and
elife_counts.csv -- the files an integrator pulls -- until someone counted the
brain again. Identifiers are invented.
"""
import builtins
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

import mousebrain.config as config
from mousebrain.classification_runs import promote_to_production
from mousebrain.config import record_registered_space
from mousebrain.tracker import ExperimentTracker

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "pipeline"
NAME = "851_PROJ_08_51_2p5x_z5"
OTHER = "852_PROJ_08_52_2p5x_z5"
KEY = "mouse_851/" + NAME
OTHER_KEY = "mouse_852/" + OTHER


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _read(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def summary(tmp_path, monkeypatch):
    s = tmp_path / "2_Data_Summary"
    s.mkdir()
    monkeypatch.setattr(config, "DATA_SUMMARY_DIR", s)
    _write(s / "region_counts.csv", [{"brain": KEY, "total_cells": "5", "region_AAA": "5"},
                                     {"brain": OTHER_KEY, "total_cells": "3", "region_AAA": "3"}])
    _write(s / "elife_counts.csv", [{"brain": KEY, "total_cells": "5", "group_A": "5"},
                                    {"brain": OTHER_KEY, "total_cells": "3", "group_A": "3"}])
    (s / ("mouse_851_%s_counts.csv" % NAME)).write_text("region_acronym,cell_count\nAAA,5\n")
    (s / ("mouse_852_%s_counts.csv" % OTHER)).write_text("region_acronym,cell_count\nAAA,3\n")
    return s


def _assert_withdrawn(s):
    for table in ("region_counts.csv", "elife_counts.csv"):
        assert [r["brain"] for r in _read(s / table)] == [OTHER_KEY], table
        archived = _read(s / table.replace(".csv", "_archive.csv"))
        assert [r["brain"] for r in archived] == [KEY], table
        assert archived[0]["withdrawn_reason"] and archived[0]["total_cells"] == "5"
    copies = sorted(p.name for p in s.glob("*_counts.csv")
                    if p.name not in ("region_counts.csv", "elife_counts.csv"))
    assert copies == ["mouse_852_%s_counts.csv" % OTHER]
    assert len(list(s.glob("mouse_851_*.withdrawn-*.csv"))) == 1


def _brain(tmp_path):
    b = tmp_path / "1_Brains" / "mouse_851" / NAME
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
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text("old")
    return b


def test_archiving_the_registration_withdraws_the_counts(tmp_path, summary):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    step3 = _load("register_step_withdraw", PIPELINE / "3_register_to_atlas.py")
    b = _brain(tmp_path)
    assert step3.archive_existing_registration(b)
    _assert_withdrawn(summary)


def test_promoting_a_classification_withdraws_the_counts(tmp_path, summary):
    b = _brain(tmp_path)
    run = tmp_path / "optimization_runs" / NAME / "class_20000101_000000"
    run.mkdir(parents=True)
    (run / "cells.xml").write_text("new")
    promote_to_production(run, b, exp_id="cla_5")
    _assert_withdrawn(summary)


def test_a_step_5_classification_withdraws_the_counts(tmp_path, summary, monkeypatch):
    step5 = _load("classify_step_withdraw", PIPELINE / "5_classify_cells.py")
    b = _brain(tmp_path)
    monkeypatch.setattr(step5, "ExperimentTracker",
                        lambda: ExperimentTracker(tmp_path / "tracker" / "calibration_runs.csv"))
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(EOFError()))

    def fake_classify(signal_path, candidates_path, output_path, model_path, voxel_sizes, **k):
        (Path(output_path) / "cells.xml").write_text("new")
        return True, 1.0, 1, 0
    monkeypatch.setattr(step5, "run_cellfinder_classify", fake_classify)
    model = tmp_path / "model.h5"
    model.write_text("m")
    monkeypatch.setattr(sys, "argv", ["5", "--brain", NAME, "--root", str(tmp_path / "1_Brains"),
                                      "--model", str(model), "--no-prefilter"])
    step5.main()
    _assert_withdrawn(summary)


def test_a_failed_step_5_run_leaves_the_counts(tmp_path, summary, monkeypatch):
    step5 = _load("classify_step_keep_counts", PIPELINE / "5_classify_cells.py")
    _brain(tmp_path)
    monkeypatch.setattr(step5, "ExperimentTracker",
                        lambda: ExperimentTracker(tmp_path / "tracker" / "calibration_runs.csv"))
    monkeypatch.setattr(step5, "run_cellfinder_classify", lambda *a, **k: (False, 1.0, 0, 0))
    model = tmp_path / "model.h5"
    model.write_text("m")
    monkeypatch.setattr(sys, "argv", ["5", "--brain", NAME, "--root", str(tmp_path / "1_Brains"),
                                      "--model", str(model), "--no-prefilter"])
    step5.main()
    assert [r["brain"] for r in _read(summary / "region_counts.csv")] == [KEY, OTHER_KEY]
