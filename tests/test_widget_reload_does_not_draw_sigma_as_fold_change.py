"""Reloading a colocalization run from the Run History panel does not draw a
recorded sigma as a fold-change threshold.

Defect covered: a background_mean run records its SIGMA in
coloc_threshold_value, but "Load selected run" read that column as the
fold-change threshold the diagnostic plots draw.
"""

from types import SimpleNamespace

import pytest

from test_encr_batch_takes_only_approved_runs import (  # noqa: F401  (fixture)
    SAMPLE_A, _coloc, tracker)
from test_slice_widget_marks_runs_for_the_batch import _Spin, _Table

def test_loading_a_background_mean_run_does_not_use_its_sigma_as_fold_change(
        tracker, monkeypatch, tmp_path):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: None, information=lambda *a: None))
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("label,centroid_y,centroid_x,is_positive,fold_change\n"
                        "1,5,5,True,3.0\n2,9,9,False,1.1\n", encoding="utf-8")
    run = _coloc(tracker, SAMPLE_A, value=0.5)
    tracker._update_row(run, {"measurements_path": str(csv_path),
                              "coloc_background_value": 10})
    fake = SimpleNamespace(
        run_history_table=_Table(run), tracker=tracker,
        _visualize_colocalization=lambda m: None, _update_diagnostic_plot=lambda: None,
        coloc_result_label=_Spin(""), status_label=_Spin(""), quant_btn=_Spin(""),
        _coloc_threshold="unset")
    mw.BrainSliceWidget._load_selected_run(fake)
    # 0.5 is a sigma; the fold-change plots must fall back to their own setting.
    assert fake._coloc_threshold is None


def test_loading_a_fold_change_run_keeps_its_fold_change(tracker, monkeypatch, tmp_path):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: None, information=lambda *a: None))
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("label,centroid_y,centroid_x,is_positive,fold_change\n"
                        "1,5,5,True,3.0\n", encoding="utf-8")
    run = _coloc(tracker, SAMPLE_A, method="fold_change", value=2.5)
    tracker._update_row(run, {"measurements_path": str(csv_path),
                              "coloc_background_value": 10})
    fake = SimpleNamespace(
        run_history_table=_Table(run), tracker=tracker,
        _visualize_colocalization=lambda m: None, _update_diagnostic_plot=lambda: None,
        coloc_result_label=_Spin(""), status_label=_Spin(""), quant_btn=_Spin(""),
        _coloc_threshold="unset")
    mw.BrainSliceWidget._load_selected_run(fake)
    assert fake._coloc_threshold == 2.5
