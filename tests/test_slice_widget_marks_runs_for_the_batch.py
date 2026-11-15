"""The BrainSlice widget records the sigma of a background_mean run, and its Run
History panel can mark a run best for the 2D batch and take a mark back.

Defects covered:
- the widget logged the fold-change spin box as the threshold of a
  background_mean run, so even a correctly chosen row carried the wrong number;
- no 2D run could be marked best at all, so the batch never used an approved run;
- a mark on another sample (which can stop the batch when it disagrees) could
  only be cleared from a terminal, and a mark older than the 20 newest runs was
  not even listed.

The message after marking names the project whose batch will use the run,
because the batch counts only marks of its own project.
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


# --- the run-history panel can mark a run best -------------------------------

def test_history_panel_marks_a_run_best_and_the_batch_uses_it(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    run = _coloc(tracker, SAMPLE_A, value=0.25)
    fake = SimpleNamespace(run_history_table=_Table(run), tracker=tracker,
                           _refresh_run_history=lambda: None)
    handler = getattr(mw.BrainSliceWidget, "_mark_selected_run_best", None)
    assert handler is not None, "the run-history panel has no way to mark a run best"
    handler(fake)
    assert tracker.get_run(run)["marked_best"] == "True"
    assert be.get_tuned_parameters()["threshold"] == 0.25
    assert "will use" in shown[-1][2]


def test_history_panel_says_when_the_batch_will_not_use_a_marked_run(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    run = _coloc(tracker, SAMPLE_A, method="fold_change", value=2.0)
    fake = SimpleNamespace(run_history_table=_Table(run), tracker=tracker,
                           _refresh_run_history=lambda: None)
    handler = getattr(mw.BrainSliceWidget, "_mark_selected_run_best", None)
    assert handler is not None, "the run-history panel has no way to mark a run best"
    handler(fake)
    assert "will NOT use" in shown[-1][2]


# --- a disagreement can be settled in the GUI ---------------------------------

def test_unmarking_one_of_two_disagreeing_runs_settles_the_conflict(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    keep = _coloc(tracker, SAMPLE_A, value=0.0)
    drop = _coloc(tracker, SAMPLE_B, value=1.0)
    tracker.mark_as_best(keep)
    tracker.mark_as_best(drop)
    with pytest.raises(be.TunedParameterConflict):
        be.get_tuned_parameters()
    # Marking another run of a THIRD sample does not clear either mark ...
    tracker.mark_as_best(_coloc(tracker, "PRJ01_09_S9_RZ", value=0.0))
    with pytest.raises(be.TunedParameterConflict):
        be.get_tuned_parameters()
    # ... the panel's Unmark Best button on the unwanted run does.
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    handler = getattr(mw.BrainSliceWidget, "_unmark_selected_run_best", None)
    assert handler is not None, "the run-history panel has no way to clear a best mark"
    handler(SimpleNamespace(run_history_table=_Table(f"★ {drop}"), tracker=tracker,
                            _refresh_run_history=lambda: None))
    assert tracker.get_run(drop)["marked_best"] == ""
    assert be.get_tuned_parameters()["threshold"] == 0.0


# --- an old mark is always listed, so it can always be cleared ----------------

class _HistoryTable:
    """Records what _refresh_run_history puts in the table."""

    def __init__(self):
        self.rows = []

    def setRowCount(self, n):
        self.rows = self.rows[:n]

    def rowCount(self):
        return len(self.rows)

    def insertRow(self, row):
        self.rows.insert(row, {})

    def setItem(self, row, col, item):
        self.rows[row][col] = item

    def resizeColumnsToContents(self):
        pass


def test_a_marked_run_older_than_the_newest_twenty_is_listed_and_can_be_unmarked(
        tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    monkeypatch.setattr(mw, "QTableWidgetItem", lambda text: text)
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    # Twenty-one runs logged within a second would share a run id; give each
    # its own.
    counter = iter(range(1000))
    monkeypatch.setattr(tracker, "_generate_run_id",
                        lambda run_type, sample_id: "col_%03d" % next(counter))
    old = _coloc(tracker, SAMPLE_A, value=1.0)
    tracker._update_row(old, {"created_at": "2000-01-01T00:00:00"})
    tracker.mark_as_best(old)
    for k in range(20):   # twenty newer completed runs of the same sample
        newer = _coloc(tracker, SAMPLE_A, value=0.0)
        tracker._update_row(newer, {"created_at": "2001-01-01T00:00:%02d" % k})

    table = _HistoryTable()
    fake = SimpleNamespace(tracker=tracker, run_history_table=table,
                           sample_id_edit=_Spin(SAMPLE_A), current_file=None,
                           session_run_ids=set())
    mw.BrainSliceWidget._refresh_run_history(fake)
    listed = [r[0] for r in table.rows]
    assert len(listed) == 21
    starred = [text for text in listed if old in text]
    assert starred and starred[0].startswith("★")

    handler = mw.BrainSliceWidget._unmark_selected_run_best
    handler(SimpleNamespace(run_history_table=_Table(starred[0]), tracker=tracker,
                            _refresh_run_history=lambda: None))
    assert tracker.get_run(old)["marked_best"] == ""


def test_mark_message_names_the_project_whose_batch_uses_the_run(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    run = _coloc(tracker, SAMPLE_A, value=0.25)
    mw.BrainSliceWidget._mark_selected_run_best(SimpleNamespace(
        run_history_table=_Table(run), tracker=tracker,
        _refresh_run_history=lambda: None))
    text = shown[-1][2]
    project = tracker.get_run(run)["project"]
    assert project and f"project '{project}'" in text
    assert SAMPLE_A in text
    assert "for this project" not in text
