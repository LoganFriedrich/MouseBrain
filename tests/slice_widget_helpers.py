"""A stand-in for the BrainSlice napari widget, for tests that drive its run
methods without opening a window.

The real widget is a Qt window with hundreds of controls. These tests need
only the methods under test and controls that answer value(), text(),
isChecked() ... with fixed answers, so the stand-in hands out a generic
control for any attribute named like a control (..._spin, ..._combo,
..._btn ...) and binds the widget's own methods to itself on demand.
"""
import inspect
import types
from types import SimpleNamespace

import numpy as np

from mousebrain.plugin_2d.sliceatlas.napari_plugin import workers
from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import BrainSliceWidget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

CONTROL_SUFFIXES = ("_spin", "_combo", "_check", "_btn", "_label", "_edit", "_table", "_ch2")


class Control:
    """Answers every question a control is asked with a fixed value."""

    def __init__(self, value=1, text="", checked=False, index=0):
        self._value, self._text, self._checked, self._index = value, text, checked, index
        self.enabled = True

    def value(self):
        return self._value

    def currentText(self):
        return self._text

    def text(self):
        return self._text

    def currentIndex(self):
        return self._index

    def isChecked(self):
        return self._checked

    def setText(self, text):
        self._text = text

    def setEnabled(self, enabled):
        self.enabled = enabled

    def isEnabled(self):
        return self.enabled

    def setValue(self, value):
        self._value = value

    def __getattr__(self, name):
        # Table and layout calls the tests do not look at.
        return lambda *a, **k: None


class FakeWidget:
    """BrainSliceWidget's methods on a plain object (see the module note)."""

    def __getattr__(self, name):
        if isinstance(inspect.getattr_static(BrainSliceWidget, name, None), staticmethod):
            return getattr(BrainSliceWidget, name)
        attr = getattr(BrainSliceWidget, name, None)
        if callable(attr) and not name.startswith("__"):
            bound = types.MethodType(attr, self)
            setattr(self, name, bound)
            return bound
        if name.endswith(CONTROL_SUFFIXES):
            control = Control()
            setattr(self, name, control)
            return control
        raise AttributeError(name)


class StubWorker:
    """A worker that never runs: the test calls the result handler itself."""

    made = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.progress = SimpleNamespace(connect=lambda *_: None)
        self.finished = SimpleNamespace(connect=lambda *_: None)
        self.started = False
        StubWorker.made.append(self)

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started


def make_widget(tmp_path, monkeypatch, sample, backend="StarDist"):
    """A stand-in widget with an image loaded and a tracker in tmp_path.
    Workers are stubs that never run."""
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
    monkeypatch.setattr(main_widget, "QMessageBox",
                        SimpleNamespace(warning=lambda *a, **k: None,
                                        critical=lambda *a, **k: None))
    StubWorker.made = []
    for name in ("DetectionWorker", "ColocalizationWorker", "DualColocalizationWorker",
                 "QuantificationWorker"):
        monkeypatch.setattr(workers, name, StubWorker)

    w = FakeWidget()
    w.tracker = SliceTracker(csv_path=tmp_path / "runs.csv")
    w.session_run_ids = []
    w._unwritable_run_ids = set()
    w._tracker_error = None
    w.last_run_id = None
    w._detection_run_id = None
    w._coloc_run_id = None
    w._labels_run_id = None
    w._analysis_running = None
    w.detection_worker = w.coloc_worker = w.quant_worker = None
    w.current_file = tmp_path / f"{sample}.nd2"
    w.sample_id_edit = Control(text=sample)
    w.backend_combo = Control(text=backend)
    w.model_combo = Control(text="2D_versatile_fluo")
    w.coloc_mode_combo = Control(text="Single Channel")
    w.bg_method_combo = Control(text="gmm")
    w.thresh_method_combo = Control(text="fold_change")
    w.red_channel = np.ones((8, 8))
    w.green_channel = np.ones((8, 8))
    w.channels = [w.red_channel, w.green_channel]
    w.nuclei_labels = None
    w.cell_measurements = None
    w._pixel_size_um = None
    w.inset_widget = SimpleNamespace(get_detection_settings=lambda: {
        "use_insets": False, "inset_manager": None, "detect_in_base": True})
    w._get_current_slice = lambda arr: arr
    w.viewer = SimpleNamespace(
        layers=[], add_labels=lambda *a, **k: SimpleNamespace(contour=0),
        add_points=lambda *a, **k: None)
    return w


def labels_with_one_nucleus():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:4, 2:4] = 1
    return labels
