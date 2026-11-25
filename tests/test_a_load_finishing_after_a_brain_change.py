"""A load that finishes after another brain was selected serves the brain it
loaded, and puts nothing of the selected brain over it.

The layers are named and tagged for the brain whose load started, so the rest
of the finish handler must serve that brain too. Serving the brain selected
NOW would load that brain's Best and recent detections and classifications
over the other brain's images, report "Loaded: <selected brain>", and start a
Zarr conversion for it.
"""
import numpy as np
import pytest

napari = pytest.importorskip("napari")

from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _tracker, _widget  # noqa: E402


class _Check:
    def isChecked(self):
        return True


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _s):
        pass

    def setEnabled(self, _v):
        pass


def _finish_load(tmp_path, monkeypatch, selected):
    w = _widget(tmp_path, monkeypatch, _tracker(tmp_path, []), brain=selected)
    brain_a = w.current_brain.parent / BRAIN_A
    w.viewer.add_image = lambda data, name="", **k: w.viewer.layers.append(
        napari.layers.Image(np.asarray(data), name=name)) or w.viewer.layers[-1]
    w.viewer.reset_view = lambda: None
    w._loading_source = "cropped"
    w._loading_brain = brain_a
    w._loading_folder = brain_a / "2_Cropped_For_Registration"
    w.load_brain_btn = w.brain_status_label = _Label()
    for name in ("load_best_detection_cb", "load_recent_detection_cb",
                 "load_best_classification_cb", "load_recent_classification_cb"):
        setattr(w, name, _Check())
    w.calls = []
    w._load_detection_results = lambda best=True: w.calls.append(("det", best)) or False
    w._load_classification_results = lambda best=True: w.calls.append(("class", best)) or False
    w._auto_start_zarr_conversion = lambda *a: w.calls.append(("zarr",) + a)
    stack = np.zeros((2, 4, 4), np.uint16)
    w._on_brain_load_finished(True, "ok", stack, stack)
    return w


def test_nothing_of_the_newly_selected_brain_is_loaded_over_the_loaded_one(tmp_path, monkeypatch):
    w = _finish_load(tmp_path, monkeypatch, selected=BRAIN_B)
    assert w.calls == []
    text = w.brain_status_label.text
    assert BRAIN_A in text and BRAIN_B in text and "selected now" in text
    # the images are the loaded brain's, tagged so
    assert {l.metadata["source_brain"] for l in w.viewer.layers} == {
        str(w.current_brain.parent / BRAIN_A)}


def test_a_load_of_the_selected_brain_loads_its_extras(tmp_path, monkeypatch):
    w = _finish_load(tmp_path, monkeypatch, selected=BRAIN_A)
    assert ("det", True) in w.calls and ("zarr", w._loading_folder) in w.calls
    assert w.brain_status_label.text.startswith(f"Loaded: {BRAIN_A}")
