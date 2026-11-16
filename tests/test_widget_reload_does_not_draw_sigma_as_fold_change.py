"""The napari widget records a background_mean run's sigma as its threshold,
and reloading such a run does not draw that sigma as a fold-change threshold.

Defects covered:
- the widget logged the fold-change spin box as the threshold of a
  background_mean run, so a row a person marked best carried the wrong number;
- a background_mean run records its SIGMA in coloc_threshold_value, but "Load
  selected run" read that column as the fold-change threshold the diagnostic
  plots draw.
"""

from types import SimpleNamespace

import pytest

from test_encr_batch_takes_only_approved_runs import (  # noqa: F401  (fixture)
    COLOC_SIGMA_RECORDED_VERSION, SAMPLE_A, _coloc, be, tracker)


# --- the napari widget records the sigma of a background_mean run ------------

class _Spin:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v

    def isChecked(self):
        return self.v

    def currentText(self):
        return self.v

    def text(self):
        return self.v

    def setText(self, *_):
        pass

    def setEnabled(self, *_):
        pass


def _run_widget_coloc(tracker, monkeypatch, method, local=False):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import workers
    # The run is logged only after the input checks pass, just before the
    # worker starts; the worker is a stand-in that never runs, and the warning
    # box is replaced so no window opens.
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(warning=lambda *a, **k: None))

    class _Worker:
        def __init__(self, *a, **k):
            self.progress = self.finished = SimpleNamespace(connect=lambda *_: None)

        def start(self):
            pass

    monkeypatch.setattr(workers, "ColocalizationWorker", _Worker)
    fake = SimpleNamespace(
        nuclei_labels=object(), green_channel=object(), red_channel=None,
        coloc_mode_combo=_Spin("Single Channel"),
        status_label=_Spin(""), coloc_btn=_Spin(""),
        bg_method_combo=_Spin("gmm"), bg_percentile_spin=_Spin(10.0),
        thresh_method_combo=_Spin(method), thresh_value_spin=_Spin(2.0),
        sigma_threshold_spin=_Spin(0.75), bg_dilation_spin=_Spin(50),
        area_fraction_spin=_Spin(0.5), bg_local_check=_Spin(local),
        bg_block_size_spin=_Spin(256), soma_dilation_spin=_Spin(6),
        tracker=tracker, last_run_id=None,
        _get_current_slice=lambda img: img,
        _on_coloc_progress=None, _on_coloc_finished=None,
    )
    mw.BrainSliceWidget._run_colocalization(fake)
    return tracker.get_run(fake.last_run_id)


def test_widget_logs_the_sigma_for_background_mean(tracker, monkeypatch):
    row = _run_widget_coloc(tracker, monkeypatch, "background_mean")
    assert float(row["coloc_threshold_value"]) == 0.75
    assert row["script_version"] == COLOC_SIGMA_RECORDED_VERSION


def test_widget_logs_the_fold_change_for_fold_change(tracker, monkeypatch):
    row = _run_widget_coloc(tracker, monkeypatch, "fold_change")
    assert float(row["coloc_threshold_value"]) == 2.0


def test_widget_marks_a_local_background_as_a_different_method(tracker, monkeypatch):
    row = _run_widget_coloc(tracker, monkeypatch, "background_mean", local=True)
    assert row["coloc_background_method"] == "gmm+local"
    assert be.batch_approval_problem(row) is not None


# --- stand-in for the run-history table: one selected row ----------------------

class _Table:
    def __init__(self, run_id):
        self.run_id = run_id

    def selectedItems(self):
        return [SimpleNamespace(row=lambda: 0)]

    def item(self, row, col):
        return SimpleNamespace(text=lambda: self.run_id)


# --- reloading a run from the history panel -----------------------------------

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
