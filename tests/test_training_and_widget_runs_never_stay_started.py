"""A run that will never finish does not stay at 'started', and the leftovers are listed.

Steps 3-6, training (util_train_model.py) and the tuning widget's detection
and classification runs all mark their run failed when they stop part-way. A
row left at 'started' forever -- a training that crashed, or a widget run cut
off by closing napari -- is indistinguishable from a run still going, so the
widget marks its unfinished runs failed when it closes, and
tracker.unfinished() / `util_experiments.py unfinished` lists what is still
'started' so a person can find it (a process killed outright can still leave
one).
"""
import csv
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tracker(tmp_path, rows=()):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            full = {c: "" for c in CSV_COLUMNS}
            full.update(r)
            w.writerow(full)
    return ExperimentTracker(csv_path=path)


def _status(tracker, exp_id):
    return tracker.get_experiment(exp_id)["status"]


# --- training ---------------------------------------------------------------

def test_a_training_that_crashes_is_marked_failed(tmp_path, monkeypatch):
    mod = _load("util_train_model_under_test", REPO / "scripts" / "utils" / "util_train_model.py")
    tracker = _tracker(tmp_path)
    monkeypatch.setattr(mod, "ExperimentTracker", lambda *a, **k: tracker)

    def crash(**kwargs):
        raise RuntimeError("cellfinder fell over")
    monkeypatch.setattr(mod, "run_cellfinder_train", crash)

    cells, non_cells = tmp_path / "pool" / "cells", tmp_path / "pool" / "non_cells"
    cells.mkdir(parents=True)
    non_cells.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [
        "util_train_model.py", "--cells", str(cells), "--non-cells", str(non_cells),
        "--output", str(tmp_path / "model_out"), "--name", "TEST_MODEL"])
    with pytest.raises(RuntimeError):
        mod.main()

    rows = [r for r in tracker._read_all() if r["exp_type"] == "training"]
    assert len(rows) == 1
    assert rows[0]["status"] == "failed", "a crashed training must not stay 'started'"
    assert "cellfinder fell over" in rows[0]["notes"]


# --- tuning widget ----------------------------------------------------------

def _widget_closing_with(tmp_path, statuses):
    tw = pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
    tracker = _tracker(tmp_path, [
        {"exp_id": exp_id, "brain": "9_PROJ_01_09_1p625x_z4", "exp_type": exp_type,
         "status": status, "created_at": "2026-01-01T00:00:00"}
        for exp_id, (exp_type, status) in statuses.items()])
    widget = SimpleNamespace(tracker=tracker, session_doc=None,
                             last_run_id="det_open", _last_class_exp_id="cls_open",
                             _runs_in_progress=set(statuses))
    event = SimpleNamespace(accepted=False, ignored=False)
    event.accept = lambda: setattr(event, "accepted", True)
    event.ignore = lambda: setattr(event, "ignored", True)
    tw.TuningWidget.closeEvent(widget, event)
    return tracker, event


def test_closing_napari_marks_the_widgets_unfinished_runs_failed(tmp_path):
    tracker, event = _widget_closing_with(tmp_path, {
        "det_open": ("detection", "started"),
        "cls_open": ("classification", "started"),
        "det_done": ("detection", "completed"),
    })
    assert event.accepted
    assert _status(tracker, "det_open") == "failed"
    assert _status(tracker, "cls_open") == "failed"
    assert "closed" in tracker.get_experiment("det_open")["notes"]
    assert _status(tracker, "det_done") == "completed", "a finished run is never touched"


# --- listing leftovers ------------------------------------------------------

def test_leftover_started_runs_are_listed(tmp_path):
    now = datetime.now()
    old = (now - timedelta(hours=30)).isoformat(timespec="seconds")
    new = (now - timedelta(minutes=5)).isoformat(timespec="seconds")
    tracker = _tracker(tmp_path, [
        {"exp_id": "cnt_old", "exp_type": "counts", "status": "started", "created_at": old},
        {"exp_id": "det_new", "exp_type": "detection", "status": "started", "created_at": new},
        {"exp_id": "det_done", "exp_type": "detection", "status": "completed", "created_at": old},
        {"exp_id": "trn_noage", "exp_type": "training", "status": "started", "created_at": ""},
    ])
    assert {r["exp_id"] for r in tracker.unfinished()} == {"cnt_old", "det_new", "trn_noage"}
    assert {r["exp_id"] for r in tracker.unfinished(older_than_hours=24)} == {"cnt_old", "trn_noage"}


def test_util_experiments_lists_them(tmp_path, monkeypatch, capsys):
    tracker = _tracker(tmp_path, [
        {"exp_id": "cnt_left_open", "brain": "9_PROJ_01_09_1p625x_z4", "exp_type": "counts",
         "status": "started", "created_at": "2026-01-01T00:00:00"},
        {"exp_id": "cnt_finished", "brain": "9_PROJ_01_09_1p625x_z4", "exp_type": "counts",
         "status": "completed", "created_at": "2026-01-01T00:00:00"},
    ])
    mod = _load("util_experiments_under_test", REPO / "scripts" / "utils" / "util_experiments.py")
    monkeypatch.setattr(sys, "argv", ["util_experiments.py", "--csv", str(tracker.csv_path),
                                      "unfinished"])
    mod.main()
    out = capsys.readouterr().out
    assert "cnt_left_open" in out and "cnt_finished" not in out
