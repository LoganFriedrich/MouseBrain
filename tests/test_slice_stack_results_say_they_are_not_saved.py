"""A folder opened as one stack has no single image to file results under,
so nothing of its analysis is saved -- and the window now says so.

Its results used to vanish on the next image with only a console line, and
a run on a stack with the Sample ID box emptied crashed on the missing
image name.
"""
import numpy as np
import pandas as pd

from slice_widget_helpers import StubWorker, make_loading_widget, make_widget


def _stack_widget(w, tmp_path):
    w.current_file = None
    w.current_folder = tmp_path / "STACK_FOLDER"
    w.current_folder.mkdir()
    w.is_folder_load = True
    return w


def test_leaving_a_stack_with_results_says_they_are_not_saved(tmp_path, monkeypatch):
    w = _stack_widget(make_loading_widget(tmp_path, monkeypatch), tmp_path)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})

    w._save_analysis_state()

    assert "NOT saved" in w.status_label.text()
    assert not any(tmp_path.rglob("*_analysis"))


def test_detection_on_a_stack_with_no_sample_id_runs_under_the_folder_name(tmp_path, monkeypatch):
    w = _stack_widget(make_widget(tmp_path, monkeypatch, "unused"), tmp_path)
    w.sample_id_edit.setText("")

    w._run_detection()

    assert StubWorker.made, "detection did not start"
    rows = w.tracker.search(sample_id="STACK_FOLDER")
    assert len(rows) == 1
    np.testing.assert_array_equal(w._detection_source, w.red_channel)
