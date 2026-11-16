"""The run history shows this slice's runs only, and loads only those.

Runs were matched to the sample by substring, so slice S1's history also
listed S10-S19, and Load Selected Run loaded the chosen run's measurements
with no check that the run belonged to the image on screen -- drawing
another slice's cells over this one and enabling Run Quantification on them.
"""
import types
from types import SimpleNamespace

import pandas as pd

from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import BrainSliceWidget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

S1 = "PRJ_01_01_S1"
S10 = "PRJ_01_01_S10"


def _coloc_run(t, sample, measurements_path):
    run_id = t.log_colocalization(sample_id=sample)
    t.update_status(run_id, status="completed", measurements_path=str(measurements_path))
    return run_id


def test_search_by_sample_is_an_exact_match(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    own = t.log_detection(sample_id=S1)
    t.log_detection(sample_id=S10)
    t.log_detection(sample_id="X" + S1)
    assert [r["run_id"] for r in t.search(sample_id=S1)] == [own]


class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def row(self):
        return 0


class _Box:
    def __init__(self):
        self.shown = []

    def warning(self, _parent, title, text, *a):
        self.shown.append((title, text))


def test_a_run_of_another_slice_is_not_loaded_over_this_one(tmp_path, monkeypatch):
    box = _Box()
    monkeypatch.setattr(main_widget, "QMessageBox", box)
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    csv_path = tmp_path / "other_measurements.csv"
    pd.DataFrame({"label": [1], "centroid_y": [1.0], "centroid_x": [1.0],
                  "is_positive": [True], "fold_change": [3.0]}).to_csv(csv_path, index=False)
    other = _coloc_run(t, S10, csv_path)

    loaded = []
    w = SimpleNamespace(
        tracker=t,
        run_history_table=SimpleNamespace(selectedItems=lambda: [_Item(other)],
                                          item=lambda r, c: _Item(other)),
        sample_id_edit=SimpleNamespace(text=lambda: ""),
        current_file=tmp_path / f"{S1}.nd2",
        cell_measurements=None,
        _visualize_colocalization=lambda m: loaded.append(m),
        _update_diagnostic_plot=lambda: None,
        coloc_result_label=SimpleNamespace(setText=lambda _t: None),
        status_label=SimpleNamespace(setText=lambda _t: None),
        quant_btn=SimpleNamespace(setEnabled=lambda b: loaded.append(("quant", b))),
    )
    BrainSliceWidget._load_selected_run(w)
    assert w.cell_measurements is None and loaded == []
    assert box.shown and S10 in box.shown[-1][1]

    # The same run IS loaded when its own slice is on screen.
    w.current_file = tmp_path / f"{S10}.nd2"
    BrainSliceWidget._load_selected_run(w)
    assert w.cell_measurements is not None and len(w.cell_measurements) == 1
