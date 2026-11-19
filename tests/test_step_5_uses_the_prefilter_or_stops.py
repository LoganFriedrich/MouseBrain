"""Step 5 classifies pre-filtered candidates or nothing, and only with its
own (settled) pre-filter.

When its own pre-filter failed, step 5 printed a warning and classified the
UNFILTERED candidates into the production classification. Since 2026-11-19
step 5 writes production with the settled method only, so a pre-filter saved
or tuned by hand (--candidates) is refused there, and RUN_PIPELINE no longer
offers one. Identifiers are invented.
"""
import builtins
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

import mousebrain.prefilter as prefilter
from mousebrain.config import record_registered_space
from mousebrain.tracker import ExperimentTracker
from settled_method_helpers import settle

ROOT = Path(__file__).resolve().parents[1]
STEP5 = ROOT / "scripts" / "pipeline" / "5_classify_cells.py"
NAME = "861_PROJ_08_61_2p5x_z5"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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


def _saved_prefilter(b, source=None, age=0):
    folder = b / "4_Cell_Candidates" / "prefiltered_20000101_000000"
    folder.mkdir()
    xml = folder / "interior_candidates.xml"
    xml.write_text("<Marker></Marker>")
    src = source if source is not None else str(b / "4_Cell_Candidates" / "detected_cells.xml")
    (folder / "prefilter_report.json").write_text(json.dumps({"source_xml": src}))
    if age:
        t = time.time() - age
        os.utime(xml, (t, t))
    return xml


@pytest.fixture
def step5(tmp_path, monkeypatch):
    mod = _load("classify_step_prefilter", STEP5)
    csv_path = tmp_path / "tracker" / "calibration_runs.csv"
    monkeypatch.setattr(mod, "ExperimentTracker", lambda: ExperimentTracker(csv_path))
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(EOFError()))
    mod.classified = []

    def fake_classify(signal_path, candidates_path, output_path, model_path, voxel_sizes, **k):
        mod.classified.append(Path(candidates_path))
        (Path(output_path) / "cells.xml").parent.mkdir(parents=True, exist_ok=True)
        (Path(output_path) / "cells.xml").write_text("new")
        return True, 1.0, 1, 0
    monkeypatch.setattr(mod, "run_cellfinder_classify", fake_classify)
    model = tmp_path / "model.h5"
    model.write_text("m")
    mod.model = model
    settle(csv_path, NAME, model)
    return mod


def _argv(monkeypatch, step5, root, *extra):
    monkeypatch.setattr(sys, "argv", ["5", "--brain", NAME, "--root", str(root)] + list(extra))


def test_a_failed_prefilter_stops_step_5(step5, tmp_path, monkeypatch):
    root = tmp_path / "1_Brains"
    b = _brain(root)

    def broken(**k):
        raise RuntimeError("the atlas annotation could not be read")
    monkeypatch.setattr(prefilter, "prefilter_candidates", broken)
    _argv(monkeypatch, step5, root)
    with pytest.raises(SystemExit) as e:
        step5.main()
    assert e.value.code not in (0, None)
    assert step5.classified == []
    assert not (b / "5_Classified_Cells" / "cells.xml").exists()


def test_a_saved_prefilter_is_refused(step5, tmp_path, monkeypatch, capsys):
    """A saved or tuned pre-filter is not the settled method: production is
    never made from it, and nothing is pre-filtered or classified."""
    root = tmp_path / "1_Brains"
    b = _brain(root)
    time.sleep(0.01)
    saved = _saved_prefilter(b)
    ran = []
    monkeypatch.setattr(prefilter, "prefilter_candidates", lambda **k: ran.append(1))
    _argv(monkeypatch, step5, root, "--candidates", str(saved))
    with pytest.raises(SystemExit) as e:
        step5.main()
    assert e.value.code not in (0, None)
    assert step5.classified == [] and ran == []
    assert not (b / "5_Classified_Cells" / "cells.xml").exists()
    assert "--candidates" in capsys.readouterr().out


def test_run_pipeline_runs_step_5_without_offering_a_saved_prefilter(tmp_path, monkeypatch):
    rp = _load("run_pipeline_prefilter", ROOT / "scripts" / "pipeline" / "RUN_PIPELINE.py")
    b = _brain(tmp_path / "1_Brains")
    _saved_prefilter(b)
    from mousebrain.pipeline_status import brain_status
    status = dict(brain_status(b), step="classify", step_name="Classify cells", step_num=7)
    monkeypatch.setattr(rp, "get_brain_status", lambda p: status)
    answers = [0]  # "Run this step"; there is no second question
    monkeypatch.setattr(rp, "ask", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(rp, "clear", lambda: None)
    ran = []
    monkeypatch.setattr(rp, "run_script", lambda name, args=None: ran.append((name, args)))
    rp.show_brain_status({"name": NAME, "full_name": "mouse_" + NAME + "/" + NAME, "path": b,
                          "mouse": "mouse_" + NAME})
    assert ran == [("5_classify_cells.py", ["--brain", NAME])]
