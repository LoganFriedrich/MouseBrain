"""The BrainSlice widget records the sigma of a background_mean run.

Defect covered: the widget logged the fold-change spin box as the threshold of
a background_mean run, so even a correctly chosen row carried the wrong number.
"""

from types import SimpleNamespace

import pytest

from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

from test_encr_batch_takes_only_approved_runs import (  # noqa: F401  (fixture)
    COLOC_SIGMA_RECORDED_VERSION, SAMPLE_A, SAMPLE_B, _coloc, tracker)

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
    # The fake widget has no signal image, so the method stops right after
    # logging the run; the warning box is replaced so no window opens.
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(warning=lambda *a, **k: None))
    fake = SimpleNamespace(
        nuclei_labels=object(), green_channel=object(), red_channel=None,
        coloc_mode_combo=_Spin("Single Channel"),
        status_label=_Spin(""), coloc_btn=_Spin(""),
        bg_method_combo=_Spin("gmm"), bg_percentile_spin=_Spin(10.0),
        thresh_method_combo=_Spin(method), thresh_value_spin=_Spin(2.0),
        sigma_threshold_spin=_Spin(0.75), bg_dilation_spin=_Spin(50),
        area_fraction_spin=_Spin(0.5), bg_local_check=_Spin(local),
        bg_block_size_spin=_Spin(256), soma_dilation_spin=_Spin(6),
        sample_id_edit=_Spin(SAMPLE_A), current_file=None,
        tracker=tracker, last_run_id=None,
        _get_current_slice=lambda img: None,
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


# --- a selected row of the run-history panel ---------------------------------

class _Table:
    def __init__(self, run_id):
        self.run_id = run_id

    def selectedItems(self):
        return [SimpleNamespace(row=lambda: 0)]

    def item(self, row, col):
        return SimpleNamespace(text=lambda: self.run_id)
