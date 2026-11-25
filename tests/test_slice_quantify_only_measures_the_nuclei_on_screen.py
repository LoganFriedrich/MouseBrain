"""Colocalize and Quantify are off until the image and nuclei on screen have
been detected and measured.

If both stayed on across a new image and across a new detection, the
previous image's nuclei could be colocalized against the new image and the
previous detection's measurements counted as if they described the new
nuclei.
"""
import pandas as pd

from slice_widget_helpers import (browse, finish_load, image_files,
                                  labels_with_one_nucleus, make_loading_widget)


def _measured(w):
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})
    w.coloc_btn.setEnabled(True)
    w.quant_btn.setEnabled(True)


def test_a_new_detection_forgets_the_old_measurements(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    _measured(w)

    w._run_detection()
    w._on_detect_finished(True, "done", 1, labels_with_one_nucleus())
    assert w.cell_measurements is None
    assert not w.quant_btn.isEnabled()
    assert w.coloc_btn.isEnabled()      # the new nuclei can be measured


def test_a_new_image_turns_colocalize_and_quantify_off(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    _measured(w)
    browse(w, monkeypatch, b)
    finish_load(w)
    assert not w.coloc_btn.isEnabled()
    assert not w.quant_btn.isEnabled()
