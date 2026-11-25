"""The Run History panel always lists a sample's marked-best runs.

The panel lists the 20 newest completed runs of the sample and, in addition,
every marked-best run. The 2D batch stops when marked runs of its project
disagree and tells the person to clear the unwanted mark in this panel; a
mark older than those 20 runs must still be selectable, or the stop could not
be cleared in the GUI.
"""

import csv
from types import SimpleNamespace

import pytest

from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker
from mousebrain.plugin_2d.sliceatlas.tracker.schema import CSV_COLUMNS

SAMPLE = "PRJ_01_02_S3_RA"


class _HistoryTable:
    """Just enough of a QTableWidget to record what the panel lists."""

    def __init__(self):
        self.cells = {}
        self.n = 0

    def setRowCount(self, n):
        self.n = n
        self.cells = {k: v for k, v in self.cells.items() if k[0] < n}

    def rowCount(self):
        return self.n

    def insertRow(self, row):
        self.n += 1

    def setItem(self, row, col, item):
        self.cells[(row, col)] = item

    def resizeColumnsToContents(self):
        pass

    def listed_ids(self):
        return [self.cells[(r, 0)].lstrip("● ★ ").strip() for r in range(self.n)]


def _completed_coloc(t, sample):
    run = t.log_colocalization(sample_id=sample, signal_channel="green",
                               background_method="gmm", threshold_method="background_mean",
                               threshold_value=1.0, status="started")
    t.update_status(run, status="completed")
    return run


def test_an_old_marked_run_is_listed_behind_newer_runs(tmp_path, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    monkeypatch.setattr(mw, "QTableWidgetItem", lambda text: text)
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    # Run ids carry a one-second timestamp; give each run its own id.
    counter = iter(range(1000))
    monkeypatch.setattr(t, "_generate_run_id",
                        lambda run_type, sample_id: f"col_{next(counter):04d}")
    old_marked = _completed_coloc(t, SAMPLE)
    t.mark_as_best(old_marked)
    for _ in range(25):
        _completed_coloc(t, SAMPLE)
    # created_at has one-second resolution; make the marked run the oldest.
    rows = t._read_all()
    for r in rows:
        if r["run_id"] == old_marked:
            r["created_at"] = "2000-01-01T00:00:00"
    with open(t.csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in CSV_COLUMNS} for r in rows)

    table = _HistoryTable()
    fake = SimpleNamespace(run_history_table=table, tracker=t,
                           sample_id_edit=SimpleNamespace(text=lambda: SAMPLE),
                           current_file=None, session_run_ids=set())
    mw.BrainSliceWidget._refresh_run_history(fake)
    listed = table.listed_ids()
    assert old_marked in listed
    assert len(listed) == 21          # the 20 newest plus the marked one
    assert len(set(listed)) == len(listed)


def test_a_marked_run_of_a_longer_named_sample_is_not_listed(tmp_path, monkeypatch):
    # The tracker's search matches sample ids by substring. A sample whose
    # name starts with the open sample's name is another sample: its starred
    # run must not be listed (where "Unmark Best" could be clicked on it).
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    monkeypatch.setattr(mw, "QTableWidgetItem", lambda text: text)
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    counter = iter(range(1000))
    monkeypatch.setattr(t, "_generate_run_id",
                        lambda run_type, sample_id: f"col_{next(counter):04d}")
    other_marked = _completed_coloc(t, SAMPLE + "_v2")
    t.mark_as_best(other_marked)
    for _ in range(25):
        _completed_coloc(t, SAMPLE)
    rows = t._read_all()
    for r in rows:
        if r["run_id"] == other_marked:
            r["created_at"] = "2000-01-01T00:00:00"
    with open(t.csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in CSV_COLUMNS} for r in rows)

    table = _HistoryTable()
    fake = SimpleNamespace(run_history_table=table, tracker=t,
                           sample_id_edit=SimpleNamespace(text=lambda: SAMPLE),
                           current_file=None, session_run_ids=set())
    mw.BrainSliceWidget._refresh_run_history(fake)
    assert other_marked not in table.listed_ids()
    assert len(table.listed_ids()) == 20
