"""A region count that succeeds is shown in the Results tab and logged.

When step 6 finished from the Results tab ("Run Region Counting"), the finish
handler filled a comparison table that the tab no longer has: every
successful count raised an error, the status line stayed at "Complete!
Loading results..." for good, the tables were not filled and the session log
entry was skipped. Now the tab's own tables are filled from the files step 6
wrote, the status line gives the totals, and the count is logged.
"""
import pytest

pytest.importorskip("napari")
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QTableWidget  # noqa: E402

from tuning_widget_stand_in import BRAIN_A, _tracker, _widget  # noqa: E402

_APP = QApplication.instance() or QApplication([])

BY_REGION = (
    "region_acronym,region_name,cell_count,left_count,right_count\n"
    "AAA,Region one,30,10,20\n"
    "BBB,Region two,12,6,6\n"
    "CCC,Region three,0,0,0\n"
)
GROUPED = (
    "elife_group,group_id,cell_count,left_count,right_count,description,constituent_regions\n"
    "Group one,1,42,16,26,Two regions,AAA=30; BBB=12\n"
)


class _Log:
    def __init__(self):
        self.counted = []
        self.totals = []

    def log_region_count(self, brain_name, total_cells=0, regions_counted=0, output_path=""):
        self.counted.append(brain_name)
        self.totals.append((total_cells, regions_counted))


def _counted_widget(tmp_path, monkeypatch):
    w = _widget(tmp_path, monkeypatch, _tracker(tmp_path, []), brain=BRAIN_A)
    out = w.current_brain / "6_Region_Analysis"
    out.mkdir(parents=True)
    (out / "cell_counts_by_region.csv").write_text(BY_REGION, encoding="utf-8")
    (out / "cell_counts_elife_grouped.csv").write_text(GROUPED, encoding="utf-8")
    w.region_count_status = QLabel()
    w.run_regions_btn = QPushButton()
    w.brain_combo = QPushButton()
    w.region_count_worker = None
    w.allen_table = QTableWidget()
    w.allen_table.setColumnCount(4)
    w.allen_summary = QLabel()
    w.elife_table = QTableWidget()
    w.elife_table.setColumnCount(4)
    w.session_doc = _Log()
    return w


def test_a_successful_count_fills_the_results_tables_and_is_logged(tmp_path, monkeypatch):
    w = _counted_widget(tmp_path, monkeypatch)
    error = None
    try:
        w._on_region_count_finished(True, "done", w.current_brain)
    except Exception as e:  # the old handler raised here
        error = e
    assert error is None, f"the finish handler failed: {error!r}"
    status = w.region_count_status.text()
    assert "Loading results" not in status
    assert "42 cells in 2 regions" in status
    assert w.allen_table.rowCount() == 2
    assert w.elife_table.rowCount() == 1
    assert w.session_doc.counted == [BRAIN_A]
    assert w.session_doc.totals == [(42, 2)]   # the real totals, not the defaults
    assert w.run_regions_btn.isEnabled()


def test_counts_that_cannot_be_shown_are_still_logged_and_said(tmp_path, monkeypatch):
    w = _counted_widget(tmp_path, monkeypatch)

    def broken():
        raise ValueError("table broke")
    w._refresh_all_results_views = broken
    error = None
    try:
        w._on_region_count_finished(True, "done", w.current_brain)
    except Exception as e:
        error = e
    assert error is None, f"the finish handler failed: {error!r}"
    assert "could not be shown" in w.region_count_status.text()
    assert "table broke" in w.region_count_status.text()
    assert w.session_doc.counted == [BRAIN_A]
    assert w.session_doc.totals == [(42, 2)]
