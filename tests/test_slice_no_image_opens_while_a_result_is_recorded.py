"""No image opens while a detection or colocalization result is being
handled -- including while it is written to the tracker.

The result handlers must not free the window first thing and then write the
tracker on the window's own thread, which lets the window answer clicks that
were waiting. A queued Browse / Load would be honored right then: the image on
screen would be saved and cleared, and the handler would go on to draw its
nuclei on the image being loaded and to turn Colocalize / Quantify back on for
it.
"""
import numpy as np
import pandas as pd

from mousebrain.plugin_2d.sliceatlas.core import config as slice_config

from slice_widget_helpers import (StubLoader, browse, finish_load, image_files,
                                  labels_with_one_nucleus, make_loading_widget)
from test_slice_inset_detection_forgets_the_old_measurements import _insets


def _window_with_a(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    assert w.current_file == a
    return w, a, b


def _click_browse_while_recording(w, monkeypatch, b):
    """The tracker write lets the window answer a Browse click on b."""
    clicks = []

    def saving_to_tracker():
        clicks.append(len(StubLoader.made))
        browse(w, monkeypatch, b)
    w._saving_to_tracker = saving_to_tracker
    return clicks


def _assert_nothing_opened(w, a, loads_before):
    assert len(StubLoader.made) == loads_before, "an image started loading"
    assert w._loading_file is None
    assert w.current_file == a
    assert w._analysis_running is None      # free once the handler is done


def test_a_detection_result_keeps_the_image_until_it_is_handled(tmp_path, monkeypatch):
    w, a, b = _window_with_a(tmp_path, monkeypatch)
    w._run_detection()
    assert w._analysis_running == "detection"
    loads = len(StubLoader.made)
    clicks = _click_browse_while_recording(w, monkeypatch, b)

    labels = labels_with_one_nucleus()
    w._on_detect_finished(True, "Detected 1 nuclei", 1, labels)

    assert clicks, "the handler did not write the tracker"
    _assert_nothing_opened(w, a, loads)
    assert w.nuclei_labels is labels         # still this image's nuclei


def test_a_colocalization_result_keeps_the_image_until_it_is_handled(tmp_path, monkeypatch):
    w, a, b = _window_with_a(tmp_path, monkeypatch)
    monkeypatch.setattr(slice_config, "get_sample_dir", lambda stem: tmp_path / "out" / stem)
    w.nuclei_labels = labels_with_one_nucleus()
    w._analysis_running = "colocalization"
    w._coloc_run_id = w.tracker.log_colocalization(sample_id="SAMPLE_A")
    w._visualize_colocalization = lambda m: None
    w._update_diagnostic_plot = lambda: None
    w._refresh_run_history = lambda: None
    loads = len(StubLoader.made)
    clicks = _click_browse_while_recording(w, monkeypatch, b)

    measurements = pd.DataFrame({"label": [1], "is_positive": [True], "fold_change": [3.0]})
    summary = {"positive_cells": 1, "negative_cells": 0, "positive_fraction": 1.0,
               "background_used": 10.0, "mean_fold_change": 3.0}
    w._on_coloc_finished(True, "done", measurements, summary, None)

    assert clicks, "the handler did not write the tracker"
    _assert_nothing_opened(w, a, loads)
    assert w.cell_measurements is measurements


def test_an_inset_detection_keeps_the_image_until_it_is_handled(tmp_path, monkeypatch):
    w, a, b = _window_with_a(tmp_path, monkeypatch)
    params, settings = _insets(monkeypatch)
    w.inset_widget.get_detection_settings = lambda: dict(settings, use_insets=True)
    w._inset_detection_plan = lambda backend: (True, None)
    w._refresh_inset_status = lambda: None
    loads = len(StubLoader.made)
    clicks = _click_browse_while_recording(w, monkeypatch, b)

    w._run_detection()

    assert clicks, "the inset detection did not write the tracker"
    _assert_nothing_opened(w, a, loads)
    assert int(w.nuclei_labels.max()) == 2
