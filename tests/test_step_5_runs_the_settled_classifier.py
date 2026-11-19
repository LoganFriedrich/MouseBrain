"""Step 5 writes the production classification with the settled method only.

The settled method is the classification run a person marked Paradigm Best
for the brain's imaging setup: its model file, cube and batch size, and -- for
a selective classification -- its tracing type. Before 2026-11-19 step 5 took
"the newest model file" when nothing was marked, any --model or pre-filter
choice went straight into production, and a selective classification could
only write files beside production (cells_selective.xml), never be counted.
Identifiers are invented.
"""
import builtins
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import mousebrain.selective_classifier as sc
from mousebrain.classification_runs import (archive_production, follow_selective_rows,
                                            install_staged, staging_folder)
from mousebrain.config import record_registered_space
from mousebrain.region_counts_io import detection_provenance
from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker
from settled_method_helpers import fake_prefilter, settle

ROOT = Path(__file__).resolve().parents[1]
STEP5 = ROOT / "scripts" / "pipeline" / "5_classify_cells.py"
MARK = ROOT / "scripts" / "utils" / "util_mark_paradigm_classifier.py"
NAME = "871_PROJ_08_71_2p5x_z5"
SETUP = "2p5x_z5"


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


class FakeResult:
    def __init__(self, final=3):
        self.total_candidates = 5
        self.trust_count = 2
        self.classify_count = 3
        self.classified_cells = final - 2 if final else 0
        self.classified_rejected = 3 - self.classified_cells
        self.final_cells = final
        self.duration_seconds = 1.0


@pytest.fixture
def env(tmp_path, monkeypatch):
    step5 = _load("classify_step_settled", STEP5)
    csv_path = tmp_path / "tracker" / "calibration_runs.csv"
    monkeypatch.setattr(step5, "ExperimentTracker", lambda: ExperimentTracker(csv_path))
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(EOFError()))
    root = tmp_path / "1_Brains"
    b = _brain(root)
    model = tmp_path / "models" / "settled" / "best_model.h5"
    model.parent.mkdir(parents=True)
    model.write_text("settled")
    # A NEWER model beside it: it must never be taken for being newest.
    newer = tmp_path / "models" / "zz_newer" / "best_model.h5"
    newer.parent.mkdir(parents=True)
    newer.write_text("newer")
    monkeypatch.setattr(step5, "DEFAULT_MODELS_DIR", tmp_path / "models", raising=False)
    seen = {"selective": [], "full": [], "prefilter": []}

    def fake_selective(**kw):
        seen["selective"].append(kw)
        out = Path(kw["output_dir"])
        (out / "cells_selective.xml").write_text("selective-cells")
        (out / "selective_rejected.xml").write_text("selective-rejected")
        (out / "selective_report.json").write_text('{"final_cells": 3}')
        (out / "trust_cells.xml").write_text("trust")
        (out / "classify_input.xml").write_text("classify")
        return FakeResult(env_state["final"])

    def fake_full(signal_path, candidates_path, output_path, model_path, voxel_sizes, **k):
        seen["full"].append(Path(model_path))
        (Path(output_path) / "cells.xml").write_text("full-cells")
        return True, 1.0, 1, 0
    env_state = {"final": 3}
    monkeypatch.setattr(step5, "run_selective", fake_selective, raising=False)
    monkeypatch.setattr(step5, "run_cellfinder_classify", fake_full)
    fake_prefilter(monkeypatch, seen=seen["prefilter"])
    return dict(step5=step5, csv=csv_path, root=root, brain=b, model=model, newer=newer,
                seen=seen, state=env_state, tmp=tmp_path)


def _run(env, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["5", "--brain", NAME, "--root", str(env["root"])]
                        + list(extra))
    env["step5"].main()


def _rows(env, kind):
    return [r for r in ExperimentTracker(env["csv"])._read_all()
            if r.get("exp_type") == kind and r.get("paradigm_best") != "True"]


# ------------------------------------------------------------------ refusals

def test_no_paradigm_best_means_nothing_is_classified(env, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run(env, monkeypatch)
    assert e.value.code not in (0, None)
    assert env["seen"]["full"] == [] and env["seen"]["selective"] == []
    assert not (env["brain"] / "5_Classified_Cells" / "cells.xml").exists()
    assert "Paradigm Best" in capsys.readouterr().out


def test_a_paradigm_best_without_cube_size_is_refused(env, monkeypatch, capsys):
    settle(env["csv"], NAME, env["model"], method="selective", cube=None)
    with pytest.raises(SystemExit):
        _run(env, monkeypatch)
    assert env["seen"]["selective"] == []
    assert "cube_size" in capsys.readouterr().out


def test_a_missing_settled_model_is_never_replaced_by_another(env, monkeypatch):
    settle(env["csv"], NAME, env["model"], method="selective")
    env["model"].unlink()
    with pytest.raises(SystemExit):
        _run(env, monkeypatch)
    assert env["seen"]["selective"] == [] and env["seen"]["full"] == []


@pytest.mark.parametrize("extra", [["--no-prefilter"], ["--cube-size", "40"],
                                   ["--tracing-type", "ascending"], ["--model", "OTHER"]])
def test_a_departure_from_the_settled_method_is_refused(env, monkeypatch, extra):
    settle(env["csv"], NAME, env["model"], method="selective")
    extra = [str(env["newer"]) if x == "OTHER" else x for x in extra]
    with pytest.raises(SystemExit) as e:
        _run(env, monkeypatch, *extra)
    assert e.value.code not in (0, None)
    assert env["seen"]["selective"] == [] and env["seen"]["prefilter"] == []


def test_the_settled_values_given_again_are_accepted(env, monkeypatch):
    settle(env["csv"], NAME, env["model"], method="selective")
    _run(env, monkeypatch, "--model", str(env["model"]), "--cube-size", "50",
         "--batch-size", "32", "--tracing-type", "descending", "--selective")
    assert len(env["seen"]["selective"]) == 1


# ------------------------------------------------------------ selective path

def test_the_settled_selective_method_writes_production(env, monkeypatch):
    b = env["brain"]
    prod = b / "5_Classified_Cells"
    prod.mkdir()
    (prod / "cells.xml").write_text("old-full")
    (prod / "cells_selective.xml").write_text("old-side-file")
    best = settle(env["csv"], NAME, env["model"], method="selective")
    _run(env, monkeypatch)

    kw = env["seen"]["selective"][0]
    assert Path(kw["model_path"]) == env["model"]
    assert (kw["cube_size"], kw["batch_size"], kw["tracing_type"]) == (50, 32, "descending")
    assert kw["on_classifier_error"] == "raise"
    assert Path(kw["output_dir"]) != prod  # staged, never written in place
    # the settled pre-filter, with its settings written out
    assert env["seen"]["prefilter"][0]["method"] == "atlas_erosion"
    assert env["seen"]["prefilter"][0]["surface_depth_um"] == 100.0

    assert (prod / "cells.xml").read_text() == "selective-cells"
    assert (prod / "rejected.xml").read_text() == "selective-rejected"
    for name in ("selective_report.json", "trust_cells.xml", "classify_input.xml"):
        assert (prod / name).exists(), name
    assert not (prod / "cells_selective.xml").exists()
    archived = list((prod / "_archive").glob("*/cells.xml"))
    assert [p.read_text() for p in archived] == ["old-full"]
    assert (archived[0].parent / "cells_selective.xml").read_text() == "old-side-file"

    record = json.loads((prod / "promoted_from.json").read_text())
    run = _rows(env, "selective_classification")[0]
    assert record["exp_id"] == run["exp_id"]
    m = record["method"]
    assert m["method"] == "selective" and m["paradigm_best_exp_id"] == best
    assert Path(m["model_path"]) == env["model"] and m["model_file"]["bytes"] == 7
    assert m["prefilter"]["settings"]["method"] == "atlas_erosion"
    assert m["counts"]["cells"] == 3

    assert run["status"] == "completed"
    assert Path(run["output_path"]) == prod / "cells.xml"
    assert (run["class_cube_size"], run["class_batch_size"]) == ("50", "32")
    assert "final=3" in run["notes"] and "trust=0" not in run["notes"]


def test_a_selective_run_with_no_cells_leaves_production(env, monkeypatch):
    prod = env["brain"] / "5_Classified_Cells"
    prod.mkdir()
    (prod / "cells.xml").write_text("keep")
    settle(env["csv"], NAME, env["model"], method="selective")
    env["state"]["final"] = 0
    with pytest.raises(SystemExit):
        _run(env, monkeypatch)
    assert (prod / "cells.xml").read_text() == "keep"
    assert _rows(env, "selective_classification")[0]["status"] == "failed"


def test_the_previous_prefilter_output_is_kept(env, monkeypatch):
    pre = env["brain"] / "4_Cell_Candidates" / "prefiltered"
    pre.mkdir()
    (pre / "interior_candidates.xml").write_text("march")
    settle(env["csv"], NAME, env["model"], method="selective")
    _run(env, monkeypatch)
    kept = list((pre / "_archive").glob("*/interior_candidates.xml"))
    assert [p.read_text() for p in kept] == ["march"]


def test_a_settled_full_classification_uses_its_own_model(env, monkeypatch):
    settle(env["csv"], NAME, env["model"], method="full")
    _run(env, monkeypatch)
    assert env["seen"]["full"] == [env["model"]]
    assert (env["brain"] / "5_Classified_Cells" / "cells.xml").read_text() == "full-cells"


# ------------------------------------------------------------------ tracker

def _tracker(tmp_path, rows):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            full = {c: "" for c in CSV_COLUMNS}
            full.update(r)
            w.writerow(full)
    return ExperimentTracker(csv_path=path)


def _row(exp_id, exp_type, **extra):
    r = {"exp_id": exp_id, "exp_type": exp_type, "brain": "m/" + NAME,
         "imaging_paradigm": SETUP, "status": "completed",
         "class_model_path": "models/" + exp_id + ".h5", "prefilter_tracing_type": "descending"}
    r.update(extra)
    return r


def test_one_paradigm_best_across_both_kinds_of_classification(tmp_path):
    t = _tracker(tmp_path, [_row("cla_old", "classification", paradigm_best="True",
                                 marked_best="True", class_cube_size="50",
                                 class_batch_size="32"),
                            _row("sel_new", "selective_classification")])
    assert t.mark_paradigm_best("sel_new", fill_empty={"class_cube_size": 50,
                                                       "class_batch_size": 32},
                                note="settled")
    rows = {r["exp_id"]: r for r in t._read_all()}
    assert rows["cla_old"]["paradigm_best"] == "" and rows["cla_old"]["marked_best"] == ""
    assert rows["sel_new"]["paradigm_best"] == "True"
    assert (rows["sel_new"]["class_cube_size"], rows["sel_new"]["notes"]) == ("50", "settled")
    m = t.get_paradigm_classification(SETUP)
    assert (m["method"], m["source_exp_id"], m["cube_size"], m["missing"]) == \
        ("selective", "sel_new", 50, [])
    assert t.get_paradigm_model(SETUP) == "models/sel_new.h5"


def test_a_recorded_value_is_never_overwritten_by_the_mark(tmp_path):
    t = _tracker(tmp_path, [_row("sel_x", "selective_classification", class_cube_size="40")])
    before = t.csv_path.read_bytes()
    assert not t.mark_paradigm_best("sel_x", fill_empty={"class_cube_size": 50})
    assert t.csv_path.read_bytes() == before


def test_the_marking_tool_dry_run_writes_nothing_and_the_real_run_marks(tmp_path):
    model = tmp_path / "m.h5"
    model.write_text("m")
    t = _tracker(tmp_path, [_row("sel_x", "selective_classification",
                                 class_model_path=str(model))])
    before = t.csv_path.read_bytes()
    args = [sys.executable, str(MARK), "--run", "sel_x", "--cube-size", "50",
            "--batch-size", "32", "--tracker", str(t.csv_path)]
    dry = subprocess.run(args + ["--dry-run"], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert t.csv_path.read_bytes() == before
    real = subprocess.run(args + ["--why", "a person confirmed it"],
                          capture_output=True, text=True)
    assert real.returncode == 0, real.stdout + real.stderr
    m = ExperimentTracker(t.csv_path).get_paradigm_classification(SETUP)
    assert (m["source_exp_id"], m["cube_size"], m["batch_size"]) == ("sel_x", 50, 32)


def test_the_marking_tool_refuses_a_run_without_its_model_file(tmp_path):
    t = _tracker(tmp_path, [_row("sel_x", "selective_classification",
                                 class_cube_size="50", class_batch_size="32")])
    before = t.csv_path.read_bytes()
    r = subprocess.run([sys.executable, str(MARK), "--run", "sel_x", "--tracker",
                        str(t.csv_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "model file" in r.stdout
    assert t.csv_path.read_bytes() == before


# -------------------------------------------------- selective module, files

def test_the_selective_classifier_writes_where_it_is_told_and_can_stop(tmp_path, monkeypatch):
    cells = [sc.CellRegionInfo(1, 2, 3, 5, "A", "a", "G", "left", (0, 0, 0)),
             sc.CellRegionInfo(4, 5, 6, 0, "", "", None, "left", (0, 0, 0))]
    monkeypatch.setattr(sc, "map_cells_to_regions", lambda *a, **k: cells)
    monkeypatch.setattr(sc, "categorize_cells",
                        lambda c, tt: ([c[0]], [c[1]], {"G": 1}, {"unmapped": 1}))
    # the classifier cannot be imported
    monkeypatch.setitem(sys.modules, "cellfinder.core.classify.classify", None)
    out = tmp_path / "staged"
    with pytest.raises(ImportError):
        sc.run_selective_classification(
            brain_path=tmp_path / "brain", candidates_xml=tmp_path / "c.xml",
            model_path=tmp_path / "m.h5", registration_path=tmp_path / "reg",
            output_dir=out, on_classifier_error="raise")
    assert (out / "trust_cells.xml").exists()
    assert not (tmp_path / "brain" / "5_Classified_Cells").exists()
    r = sc.run_selective_classification(
        brain_path=tmp_path / "brain", candidates_xml=tmp_path / "c.xml",
        model_path=tmp_path / "m.h5", registration_path=tmp_path / "reg", output_dir=out)
    assert r.final_cells == 1 and (out / "cells_selective.xml").exists()


def test_side_files_of_earlier_selective_runs_are_archived_and_rows_follow(tmp_path):
    b = tmp_path / "1_Brains" / "m" / NAME
    prod = b / "5_Classified_Cells"
    prod.mkdir(parents=True)
    for n in ("cells.xml", "cells_selective.xml", "selective_rejected.xml",
              "selective_report.json", "trust_cells.xml", "classify_input.xml"):
        (prod / n).write_text(n)
    t = _tracker(tmp_path, [_row("sel_march", "selective_classification",
                                 output_path=str(prod))])
    staged = staging_folder(b)
    (staged / "cells.xml").write_text("new")
    (staged / "selective_report.json").write_text("{}")
    archived, _, record = install_staged(staged, b, "sel_now", source="test",
                                         method={"method": "selective"})
    assert sorted(p.name for p in archived.iterdir()) == sorted(
        ["cells.xml", "cells_selective.xml", "selective_rejected.xml",
         "selective_report.json", "trust_cells.xml", "classify_input.xml"])
    assert record["method"] == {"method": "selective"}
    assert sorted(p.name for p in prod.iterdir() if p.is_file()) == sorted(
        ["cells.xml", "selective_report.json", "promoted_from.json"])
    assert follow_selective_rows(t, b, archived) == ["sel_march"]
    assert Path(t.get_experiment("sel_march")["output_path"]) == archived


# ------------------------------------------------------------- step 6, batch

def test_counts_say_the_classification_was_selective(tmp_path):
    b = tmp_path / NAME
    prod = b / "5_Classified_Cells"
    prod.mkdir(parents=True)
    (prod / "cells.xml").write_text("x")
    (prod / "promoted_from.json").write_text(json.dumps({"exp_id": "sel_1"}))
    cand = b / "4_Cell_Candidates" / "detected_cells.xml"
    rows = [{"exp_id": "det_1", "exp_type": "detection", "status": "completed",
             "brain": NAME, "output_path": str(cand), "created_at": "2000-01-01T00:00:00"},
            {"exp_id": "sel_1", "exp_type": "selective_classification", "status": "completed",
             "brain": NAME, "parent_experiment": "det_1", "class_model_path": "m.h5",
             "created_at": "2000-01-02T00:00:00"}]
    prov = detection_provenance(rows, NAME, prod / "cells.xml")
    assert prov["classification_exp_id"] == "sel_1"
    assert prov["classification_method"] == "selective"


def test_the_overnight_batch_leaves_the_model_to_step_5(tmp_path, monkeypatch):
    ob = _load("overnight_settled", ROOT / "scripts" / "pipeline" / "overnight_batch.py")
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(ob.subprocess, "run", run)
    ok, _ = ob.run_classification("brain_x", tmp_path)
    assert ok and "--model" not in ran[0]
    assert not hasattr(ob, "find_paradigm_model")
