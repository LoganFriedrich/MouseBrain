"""Stand-ins for driving the Setup & Tuning widget's methods without a Qt window.

The widget's methods are called on a plain object (_Widget) that holds only the
attributes a test gives it and finds every method on TuningWidget; the viewer
is a list of real napari points layers; message boxes are recorded.
"""
import csv
import types

import numpy as np

from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

import mousebrain.plugin_3d.tuning_widget as tw
import napari

BRAIN_A = "101_PROJ_01_01_2p5x_z5"
BRAIN_B = "102_PROJ_01_02_2p5x_z5"


class _Layers(list):
    """viewer.layers: a list with a .selection, as napari's LayerList has."""
    def __init__(self, *a):
        super().__init__(*a)
        self.selection = []


class _Viewer:
    def __init__(self):
        self.layers = _Layers()

    def add_points(self, data, name="", **kw):
        layer = napari.layers.Points(np.asarray(data), name=name)
        self.layers.append(layer)
        return layer


class _Messages:
    """Records every message box; a question is answered Yes."""
    Yes, No = 0x4000, 0x10000   # flag values, combined with |

    def __init__(self):
        self.seen = []

    def _rec(self, kind):
        def f(parent, title, text, *a, **k):
            self.seen.append((kind, title, text))
            return self.Yes if kind == "question" else None
        return f

    def __getattr__(self, name):
        if name in ("warning", "information", "critical", "question"):
            return self._rec(name)
        raise AttributeError(name)


class _Widget:
    """Stands in for TuningWidget: own attributes first, then its methods."""
    def __getattr__(self, name):
        fn = getattr(tw.TuningWidget, name)
        if isinstance(fn, types.FunctionType):
            return types.MethodType(fn, self)
        return fn


def _tracker(tmp_path, rows):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"exp_type": "detection", "status": "completed",
                        "imaging_paradigm": "2p5x_z5",
                        "created_at": "2000-01-0%dT00:00:00" % (i + 1), **r})
    return ExperimentTracker(path)


def _widget(tmp_path, monkeypatch, tracker, brain=BRAIN_B):
    msgs = _Messages()
    monkeypatch.setattr(tw, "QMessageBox", msgs)
    w = _Widget()
    w.tracker = tracker
    w.viewer = _Viewer()
    w.current_brain = tmp_path / "M" / brain
    w.session_doc = None
    w.session_run_ids = []
    w._refresh_context_runs = lambda: None
    w.msgs = msgs
    return w


def _run_layer(w, brain, exp_id, role, name):
    """A points layer as the widget makes it for a run of ``brain``."""
    layer = napari.layers.Points(np.zeros((2, 3)), name=name)
    layer.metadata.update({"source_brain": str(w.current_brain.parent / brain),
                           "source_folder": ""})
    if exp_id:
        layer.metadata["exp_id"] = exp_id
    if role:
        layer.metadata["cell_type"] = role
    w.viewer.layers.append(layer)
    return layer


def _row(tracker, exp_id):
    return tracker.get_experiment(exp_id)


