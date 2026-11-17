"""A training started from Setup & Tuning is finished when it says so, and
auto-classify then classifies with THAT training's model.

What went wrong:
* The widget decided a training had finished by counting the model files in
  the whole models folder: any other model file appearing meanwhile counted,
  and a training that ended without a model was reported only by count.
* A second Start replaced the first training's handle and added a second
  timer; without auto-classify the process was not tracked at all.
* Auto-classify called a method that does not exist, and before that compared
  the new model with the model list's first entry (which has no file) in a
  way that raised; and the "this run is automatic" flag stayed set after an
  attempt that never started, so the next classification by hand was loaded
  and compared without asking.

Now the training script writes ``<its output folder>/DONE.json`` naming its
model; the widget names that folder, keeps the process, refuses a second start
while one runs, and classifies with the recorded model -- only on the brain
the training was started for.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "util_train_model_done_test", REPO / "scripts" / "utils" / "util_train_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script_run(tmp_path, monkeypatch, writes_model):
    from mousebrain.tracker import ExperimentTracker
    mod = _load_script()
    tracker = ExperimentTracker(csv_path=tmp_path / "calibration_runs.csv")
    monkeypatch.setattr(mod, "ExperimentTracker", lambda *a, **k: tracker)

    def fake_train(**kw):
        if writes_model:
            (kw["output_dir"]).mkdir(parents=True, exist_ok=True)
            (kw["output_dir"] / "model.keras").write_bytes(b"model")
        else:
            kw["output_dir"].mkdir(parents=True, exist_ok=True)
        return True, 1.0, 0.1, 0.9
    monkeypatch.setattr(mod, "run_cellfinder_train", fake_train)
    cells, non_cells = tmp_path / "pool" / "cells", tmp_path / "pool" / "non_cells"
    cells.mkdir(parents=True)
    non_cells.mkdir(parents=True)
    out = tmp_path / "models" / "model_x"
    monkeypatch.setattr(sys, "argv", [
        "util_train_model.py", "--cells", str(cells), "--non-cells", str(non_cells),
        "--output", str(out), "--name", "TEST_MODEL"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    mod.main()
    rows = [r for r in tracker._read_all() if r["exp_type"] == "training"]
    return out, rows


# --- the training script ----------------------------------------------------

def test_a_finished_training_records_its_model_in_its_folder(tmp_path, monkeypatch):
    out, rows = _script_run(tmp_path, monkeypatch, writes_model=True)
    record = json.loads((out / "DONE.json").read_text(encoding="utf-8"))
    assert Path(record["model_path"]) == out / "model.keras"
    assert record["exp_id"] == rows[0]["exp_id"]
    assert rows[0]["status"] == "completed"


def test_a_training_that_left_no_model_is_not_finished(tmp_path, monkeypatch):
    out, rows = _script_run(tmp_path, monkeypatch, writes_model=False)
    assert not (out / "DONE.json").exists()
    assert rows[0]["status"] == "failed"


# --- the widget -------------------------------------------------------------

napari = pytest.importorskip("napari")
import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from tuning_widget_stand_in import BRAIN_B  # noqa: E402
from test_a_classification_run_keeps_its_brain import (  # noqa: E402
    _FakeClassification, _widget_for)


class _FakeProcess:
    made = []

    def __init__(self, cmd, **kw):
        self.cmd, self.kw, self.code = cmd, kw, None
        _FakeProcess.made.append(self)

    def poll(self):
        return self.code


class _FakeTimer:
    def __init__(self, *a):
        self.timeout = SimpleNamespace(connect=lambda fn: None)
        self.running = False

    def start(self, _ms):
        self.running = True

    def stop(self):
        self.running = False


class _Check:
    def __init__(self, on):
        self.on = on

    def isChecked(self):
        return self.on


class _Combo:
    """The model list as refresh_models fills it: entry 0 is the default
    model and has no file."""
    def __init__(self, files=()):
        self.items = [None] + [str(f) for f in files]
        self.index = 0

    def count(self):
        return len(self.items)

    def itemData(self, i):
        return self.items[i]

    def currentData(self):
        return self.items[self.index]

    def currentIndex(self):
        return self.index

    def currentText(self):
        return self.items[self.index] or "Default"

    def setCurrentIndex(self, i):
        self.index = i


def _training_widget(tmp_path, monkeypatch, auto=True):
    """A Setup & Tuning widget on a brain that can be classified (candidates,
    loaded images), with a training pool; processes and timers are fakes."""
    import qtpy.QtCore
    w, _tracker_ = _widget_for(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(tw, "MODELS_DIR", models)
    monkeypatch.setattr(tw.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    _FakeProcess.made = []
    pool = tmp_path / "pool"
    for sub in ("cells", "non_cells"):
        (pool / sub).mkdir(parents=True)
        (pool / sub / "a_Ch0.tif").write_bytes(b"x")
    w._get_selected_yaml_paths = lambda: []
    w.training_data_dir = pool
    w.training_data_path = SimpleNamespace(text=lambda: "")
    w.auto_classify_after_training = _Check(auto)
    w.last_run_id = "det_1"
    w.model_combo = _Combo()
    w.refresh_models = lambda: None
    return w, models


def _output_dir(process, models):
    """Where the training writes: the folder the widget named (--output), or
    the script's own default when the widget names none."""
    if "--output" in process.cmd:
        return Path(process.cmd[process.cmd.index("--output") + 1])
    return models / "model_default"


def _finish(process, output_dir, record=True):
    """The training process ends. With record=True it finished properly:
    model file plus its finished record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if record:
        (output_dir / "model.keras").write_bytes(b"m")
        (output_dir / "DONE.json").write_text(json.dumps(
            {"exp_id": "train_1", "model_path": str(output_dir / "model.keras")}),
            encoding="utf-8")
    process.code = 0


def test_a_second_start_is_refused_while_a_training_runs(tmp_path, monkeypatch):
    w, _models = _training_widget(tmp_path, monkeypatch, auto=False)
    w.start_training()
    w.start_training()
    assert len(_FakeProcess.made) == 1
    assert "Training Already Running" in [m[1] for m in w.msgs.seen]


def test_a_training_without_auto_classify_is_still_followed(tmp_path, monkeypatch):
    w, _models = _training_widget(tmp_path, monkeypatch, auto=False)
    w.start_training()
    timer = getattr(w, "_training_check_timer", None)
    assert timer is not None and timer.running
    assert getattr(w, "_training_process", None) is _FakeProcess.made[0]


def test_auto_classify_classifies_with_the_model_this_training_made(tmp_path, monkeypatch):
    w, models = _training_widget(tmp_path, monkeypatch)
    w.start_training()
    process = _FakeProcess.made[0]
    out = _output_dir(process, models)
    _finish(process, out)
    newer = models / "newer_elsewhere"
    newer.mkdir()
    (newer / "model.keras").write_bytes(b"m")  # the newest file, not this training's
    w.model_combo = _Combo([newer / "model.keras", out / "model.keras"])
    w._check_training_complete()
    assert len(_FakeClassification.made) == 1, w.msgs.seen
    assert _FakeClassification.made[0].kw["model_path"] == str(out / "model.keras")


def test_another_model_appearing_does_not_finish_a_training(tmp_path, monkeypatch):
    """The training ended without recording a model while another model
    appeared in the models folder: not finished, nothing classified."""
    w, models = _training_widget(tmp_path, monkeypatch)
    w.start_training()
    process = _FakeProcess.made[0]
    (models / "someone_elses").mkdir()
    (models / "someone_elses" / "model.keras").write_bytes(b"m")
    w.model_combo = _Combo([models / "someone_elses" / "model.keras"])
    _finish(process, _output_dir(process, models), record=False)
    w._check_training_complete()
    assert _FakeClassification.made == []
    assert "Training Did Not Finish" in [m[1] for m in w.msgs.seen]


def test_auto_classify_skips_a_brain_that_is_no_longer_selected(tmp_path, monkeypatch):
    w, models = _training_widget(tmp_path, monkeypatch)
    w.start_training()
    process = _FakeProcess.made[0]
    out = _output_dir(process, models)
    _finish(process, out)
    w.model_combo = _Combo([out / "model.keras"])
    w.current_brain = w.current_brain.parent / BRAIN_B
    w._check_training_complete()
    assert _FakeClassification.made == []
    assert "Not Classified Automatically" in [m[1] for m in w.msgs.seen]


def test_an_automatic_attempt_that_stops_early_leaves_no_mark(tmp_path, monkeypatch):
    """An automatic classification that cannot start (its brain has no
    candidates) must not make the next classification by hand count as
    automatic: that one is offered for loading, not loaded and compared."""
    w, models = _training_widget(tmp_path, monkeypatch)
    w.start_training()
    process = _FakeProcess.made[0]
    out = _output_dir(process, models)
    _finish(process, out)
    w.model_combo = _Combo([out / "model.keras"])
    candidates = w.current_brain / "4_Cell_Candidates" / "detected_cells.xml"
    kept = candidates.read_text(encoding="utf-8")
    candidates.unlink()
    w._check_training_complete()
    assert _FakeClassification.made == []
    assert "No Candidates Found" in [m[1] for m in w.msgs.seen]

    candidates.write_text(kept, encoding="utf-8")
    w._auto_compare_classification = lambda *a, **k: pytest.fail("compared as automatic")
    w.run_classification()
    worker = _FakeClassification.made[0]
    worker.running = False
    worker.finished.emit(True, "done", 1, 0, worker.kw["exp_id"])
    assert "Classification Complete" in [m[1] for m in w.msgs.seen if m[0] == "question"]
