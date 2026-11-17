"""A new image starts with none of the previous image's analysis: no ignore
regions, tissue mask, background diagnostics or region counts; and a saved
run loaded from the history brings none of the last live run's.

Clearing for a new image missed these, so the previous image's ignore
regions were applied to and saved with the next image, and its tissue mask
and background were drawn in the next image's diagnostic plot.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import BrainSliceWidget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker
from slice_widget_helpers import (StubLoader, browse, finish_load, image_files,
                                  make_loading_widget)


def test_opening_an_image_clears_every_per_image_result(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    w._pa_ignore_shapes_layer = w.viewer.add_shapes(
        [np.array([[0, 0], [0, 4], [4, 4]])], name="Ignore Regions")
    w._tissue_mask = np.ones((8, 8), bool)
    w._background_diagnostics = {"method": "gmm"}
    w._tissue_pixels = np.ones(4)
    w._coloc_background_surface = np.ones((8, 8))
    w.region_counts = pd.DataFrame({"region": ["R1"], "count": [3]})

    browse(w, monkeypatch, b)
    assert w._pa_ignore_shapes_layer is None
    assert "Ignore Regions" not in w.viewer.names()
    assert w._tissue_mask is None and w._background_diagnostics is None
    assert w._tissue_pixels is None and w._coloc_background_surface is None
    assert w.region_counts is None


def test_no_image_opens_over_a_batch_waiting_for_rois(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._batch_roi_mode = True
    browse(w, monkeypatch, a)
    assert StubLoader.made == []
    assert "batch" in w.status_label.text()


class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def row(self):
        return 0


def test_a_saved_run_brings_no_live_runs_tissue_mask(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    csv_path = tmp_path / "m.csv"
    pd.DataFrame({"label": [1], "centroid_y": [1.0], "centroid_x": [1.0],
                  "is_positive": [True], "fold_change": [3.0]}).to_csv(csv_path, index=False)
    run_id = t.log_colocalization(sample_id="SAMPLE_A")
    t.update_status(run_id, status="completed", measurements_path=str(csv_path),
                    coloc_background_value=10.0)
    w = SimpleNamespace(
        tracker=t,
        run_history_table=SimpleNamespace(selectedItems=lambda: [_Item(run_id)],
                                          item=lambda r, c: _Item(run_id)),
        sample_id_edit=SimpleNamespace(text=lambda: "SAMPLE_A"),
        current_file=tmp_path / "SAMPLE_A.nd2",
        cell_measurements=None,
        _tissue_mask=np.ones((8, 8), bool),
        _background_diagnostics={"method": "gmm"},
        _tissue_pixels=np.ones(4),
        _coloc_background_surface=np.ones((8, 8)),
        region_counts=pd.DataFrame({"region": ["R1"]}),
        _visualize_colocalization=lambda m: None,
        _update_diagnostic_plot=lambda: None,
        coloc_result_label=SimpleNamespace(setText=lambda _t: None),
        status_label=SimpleNamespace(setText=lambda _t: None),
        quant_btn=SimpleNamespace(setEnabled=lambda b: None),
    )
    BrainSliceWidget._load_selected_run(w)
    assert w.cell_measurements is not None
    assert w._tissue_mask is None and w._background_diagnostics is None
    assert w._tissue_pixels is None and w._coloc_background_surface is None
    assert w.region_counts is None
