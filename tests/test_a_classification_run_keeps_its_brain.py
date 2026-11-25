"""A Setup & Tuning classification run finishes as the run it started as.

The finish handler must not find the run's results through the widget's
_last_class_output, which selecting another brain resets: the run's own
results would then never be offered, and "Load classified cells" would load
the newly selected brain's production cells as if they were the run's result.
"""
from types import SimpleNamespace

import numpy as np
import pytest

napari = pytest.importorskip("napari")
tifffile = pytest.importorskip("tifffile")

import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from mousebrain.brain_context import reset_per_brain_state, tag_layer  # noqa: E402
from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _tracker, _widget  # noqa: E402

XML = ("<CellCounter_Marker_File><Marker_Data><Marker_Type><Type>1</Type>"
       "<Marker><MarkerX>1</MarkerX><MarkerY>1</MarkerY><MarkerZ>1</MarkerZ></Marker>"
       "</Marker_Type></Marker_Data></CellCounter_Marker_File>")


class _Value:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v


class _Btn:
    def __init__(self):
        self.enabled = True
        self.text = ""

    def setEnabled(self, v):
        self.enabled = v

    def setText(self, t):
        self.text = t


class _Sig:
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def emit(self, *a):
        for fn in self.slots:
            fn(*a)


class _FakeClassification:
    made = []

    def __init__(self, **kw):
        self.kw = kw
        self.finished = _Sig()
        self.progress = _Sig()
        self.running = False
        _FakeClassification.made.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running


def _brain(root, name):
    b = root / "M" / name
    for ch in ("ch0", "ch1"):
        (b / "2_Cropped_For_Registration" / ch).mkdir(parents=True)
        tifffile.imwrite(str(b / "2_Cropped_For_Registration" / ch / "Z0000.tif"),
                         np.zeros((4, 4), np.uint16))
    (b / "4_Cell_Candidates").mkdir()
    (b / "4_Cell_Candidates" / "detected_cells.xml").write_text(XML, encoding="utf-8")
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text(XML, encoding="utf-8")
    return b


def _widget_for(tmp_path, monkeypatch, layer_folder_name="2_Cropped_For_Registration"):
    monkeypatch.setattr(tw, "DATA_SUMMARY_DIR", tmp_path / "summary")
    monkeypatch.setattr(tw, "ClassificationWorker", _FakeClassification)
    _FakeClassification.made = []
    brain_a = _brain(tmp_path, BRAIN_A)
    _brain(tmp_path, BRAIN_B)
    tracker = _tracker(tmp_path, [])
    w = _widget(tmp_path, monkeypatch, tracker, brain=BRAIN_A)
    assert w.current_brain == brain_a
    for name in ("Signal", "Background"):
        layer = napari.layers.Image(np.zeros((1, 4, 4), np.uint16), name=f"{BRAIN_A} - {name}")
        tag_layer(layer, brain_a, brain_a / layer_folder_name)
        w.viewer.layers.append(layer)
    w.metadata = {"voxel_size_um": {"x": 4, "y": 4, "z": 5}}
    w.model_combo = SimpleNamespace(currentData=lambda: None, currentIndex=lambda: 0,
                                    currentText=lambda: "Default")
    w.classify_cube_size, w.classify_batch_size = _Value(50), _Value(32)
    w.classify_btn, w.classify_status, w.promote_class_btn = _Btn(), _Btn(), _Btn()
    w.brain_combo = _Btn()
    w._runs_in_progress = set()
    w._generate_classification_qc_image = lambda **k: None
    w.loaded = []
    w._load_classified_cells = lambda *a, folder=None: w.loaded.append(folder)
    return w, tracker


def test_a_classification_finishing_after_a_brain_change_is_its_own_run(tmp_path, monkeypatch):
    w, tracker = _widget_for(tmp_path, monkeypatch)
    w.run_classification()
    assert len(_FakeClassification.made) == 1, w.msgs.seen
    worker = _FakeClassification.made[0]
    run_folder = worker.kw["output_path"]
    exp_id = worker.kw["exp_id"]
    held = w.brain_combo.enabled is False

    w.current_brain = w.current_brain.parent / BRAIN_B
    reset_per_brain_state(w)

    worker.running = False
    worker.finished.emit(True, "done", 1, 0, exp_id)
    assert tracker.get_experiment(exp_id)["status"] == "completed"
    # nothing loaded, above all not brain B's production cells
    assert w.loaded == []
    assert held and w.brain_combo.enabled is True
    assert str(run_folder) in w.classify_status.text


def test_a_classification_on_the_same_brain_loads_its_own_folder(tmp_path, monkeypatch):
    w, tracker = _widget_for(tmp_path, monkeypatch)
    w.run_classification()
    worker = _FakeClassification.made[0]
    worker.running = False
    worker.finished.emit(True, "done", 1, 0, worker.kw["exp_id"])
    assert w.loaded == [worker.kw["output_path"]]
