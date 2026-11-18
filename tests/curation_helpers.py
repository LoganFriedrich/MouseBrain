"""A Setup & Tuning stand-in set up for curation (tab 'Curate/Train').

Builds on tuning_widget_stand_in: the widget's own methods run on a plain
object; labels and the curation widget are minimal stand-ins; the models
folder is a temporary folder.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import napari

import mousebrain.plugin_3d.tuning_widget as tw
from mousebrain.brain_context import reset_per_brain_state

from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _Widget, _Messages, _Viewer  # noqa: F401


class _Label:
    def __init__(self, text=""):
        self.text_ = text

    def setText(self, t):
        self.text_ = t

    def text(self):
        return self.text_

    def setStyleSheet(self, *_):
        pass


def curation_widget(tmp_path, monkeypatch, brain=BRAIN_A):
    """A widget with ``brain`` selected and an empty temporary models folder."""
    msgs = _Messages()
    monkeypatch.setattr(tw, "QMessageBox", msgs)
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(tw, "MODELS_DIR", models)
    w = _Widget()
    w.viewer = _Viewer()
    w.tracker = None
    w.session_doc = None
    w.msgs = msgs
    w.models = models
    for name in ("training_path_label", "training_data_path", "train_cells_count_label",
                 "train_non_cells_count_label", "export_status_label"):
        setattr(w, name, _Label())
    w.curation_widget = SimpleNamespace(candidates=None, candidates_layer=None)
    select_brain(w, tmp_path, brain)
    return w


def select_brain(w, tmp_path, brain):
    """What choosing another brain does to the per-brain state."""
    w.current_brain = tmp_path / "M" / brain
    reset_per_brain_state(w)


def candidates(w, points, brain=None):
    """Give the curation widget a candidates layer (tagged with ``brain`` if given)."""
    layer = napari.layers.Points(np.asarray(points, dtype=float), name="Candidates")
    if brain is not None:
        layer.metadata["source_brain"] = str(w.current_brain.parent / brain)
    w.curation_widget.candidates = layer.data.copy()
    w.curation_widget.candidates_layer = layer
    return layer


def image(w, brain, name="Signal", value=1, shape=(30, 60, 60)):
    """An image layer loaded for ``brain`` (tagged as the widget tags its loads)."""
    layer = napari.layers.Image(np.full(shape, value, dtype=np.uint16), name=name)
    layer.metadata["source_brain"] = str(w.current_brain.parent / brain)
    layer.metadata["source_folder"] = ""
    w.viewer.layers.append(layer)
    return layer


def xml_points(path):
    """(z, y, x) of every marker in a curated XML file, repeats kept."""
    root = ET.parse(str(path)).getroot()
    return [(int(m.find("MarkerZ").text), int(m.find("MarkerY").text), int(m.find("MarkerX").text))
            for m in root.iter("Marker")]


def all_saved(models: Path, kind: str):
    """Every saved answer of one kind anywhere under the models folder, repeats kept."""
    out = []
    for f in sorted(Path(models).rglob(f"curated_{kind}.xml")):
        out.extend(xml_points(f))
    return out


def layer_points(w, role):
    layer = w._get_layer_by_type(role)
    return [] if layer is None else [tuple(int(v) for v in p) for p in layer.data]
