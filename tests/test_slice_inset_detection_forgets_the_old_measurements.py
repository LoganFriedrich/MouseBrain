"""A detection with insets replaces the nuclei just as a whole-image
detection does, so it forgets what was measured on the nuclei it replaces.

The whole-image detection forgets the measurements and turns Quantify off; the
inset path must too. Otherwise Quantify would count the previous colocalization
as if it described the new nuclei, and leaving the image would save the old
measurements.csv beside the new nuclei.npz as one pair.
"""
import numpy as np
import pandas as pd

from mousebrain.plugin_2d.sliceatlas.core import detection, inset_detection

from slice_widget_helpers import browse, finish_load, image_files, make_loading_widget


class FakeInsetPipeline:
    """Returns new nuclei made of two labels, one from an inset."""

    def __init__(self, manager, detector):
        pass

    def run_full_detection(self, **kwargs):
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[5:7, 5:7] = 2
        props = pd.DataFrame({
            "area": [4, 4], "from_inset": [True, False],
            "centroid_y_base": [1.5, 5.5], "centroid_x_base": [1.5, 5.5]})
        return {"merged_labels": labels, "merged_properties": props,
                "total_cells": 2}


def _insets(monkeypatch):
    monkeypatch.setattr(inset_detection, "InsetDetectionPipeline", FakeInsetPipeline)
    monkeypatch.setattr(detection, "NucleiDetector", lambda **k: None)
    manager = type("Manager", (), {"insets": [object()]})()
    params = {"model": "2D_versatile_fluo", "prob_thresh": 0.4, "nms_thresh": 0.3,
              "filter_size": False}
    settings = {"inset_manager": manager, "detect_in_base": True}
    return params, settings


def test_a_detection_with_insets_forgets_the_old_measurements(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    # A first detection, measured by colocalization: Quantify is on.
    w.nuclei_labels = np.ones((8, 8), dtype=np.int32)
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})
    w.quant_btn.setEnabled(True)

    params, settings = _insets(monkeypatch)
    w._run_inset_detection(params, settings)

    assert int(w.nuclei_labels.max()) == 2, "the inset detection did not run"
    assert w.cell_measurements is None
    assert not w.quant_btn.isEnabled()
    assert w.coloc_btn.isEnabled()      # the new nuclei can be measured


def test_leaving_after_an_inset_detection_saves_no_old_measurements(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    w.nuclei_labels = np.ones((8, 8), dtype=np.int32)
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})

    params, settings = _insets(monkeypatch)
    w._run_inset_detection(params, settings)
    browse(w, monkeypatch, b)

    saved = a.parent / "SAMPLE_A_analysis"
    assert (saved / "nuclei.npz").exists()
    assert not (saved / "measurements.csv").exists(), \
        "measurements of the replaced nuclei saved beside the new nuclei"
