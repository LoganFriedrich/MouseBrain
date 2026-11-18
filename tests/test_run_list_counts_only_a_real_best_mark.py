"""The Setup & Tuning run list shows as Best only a run the tracker counts as Best.

The tracker writes 'True' for a Best and every tracker reader compares with
'True'. The run list's 'Best only' filter, its star, its sort and the layer
colors tested the value for truth instead, so a row holding the text 'False'
showed as Best in the list while Load best and the tracker did not count it.
"""
import pytest

pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
pytest.importorskip("napari")

from tuning_widget_stand_in import BRAIN_B, _tracker, _widget  # noqa: E402


class _List:
    def __init__(self):
        self.items = []

    def clear(self):
        self.items = []

    def addItem(self, item):
        self.items.append(item.text())


class _Box:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked


def _run_list(tmp_path, monkeypatch, best_only):
    t = _tracker(tmp_path, [
        {"exp_id": "says_false", "brain": BRAIN_B, "marked_best": "False",
         "det_cells_found": "11"},
        {"exp_id": "real_best", "brain": BRAIN_B, "marked_best": "True",
         "det_cells_found": "22"},
        {"exp_id": "newest_plain", "brain": BRAIN_B, "det_cells_found": "33"},
    ])
    w = _widget(tmp_path, monkeypatch, t)
    del w._refresh_context_runs  # the real method, found on TuningWidget
    w.runs_list = _List()
    w.filter_best_only = _Box(best_only)
    w.filter_this_session = _Box(False)
    w._refresh_context_runs()
    return w.runs_list.items


def test_best_only_lists_only_the_run_marked_true(tmp_path, monkeypatch):
    items = _run_list(tmp_path, monkeypatch, best_only=True)
    assert len(items) == 1 and "22 cells" in items[0]


def test_a_false_mark_gets_no_star_and_no_place_at_the_top(tmp_path, monkeypatch):
    items = _run_list(tmp_path, monkeypatch, best_only=False)
    assert "22 cells" in items[0]          # the Best first
    assert "33 cells" in items[1]          # then the newest
    starred = [i for i in items if "★" in i]
    assert len(starred) == 1 and "22 cells" in starred[0]
