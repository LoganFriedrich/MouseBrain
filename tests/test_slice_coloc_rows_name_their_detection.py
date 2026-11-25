"""A colocalization row from the napari window names the detection whose
nuclei it measured, and a dual-channel row keeps its four classes apart.

The window's colocalization rows carry parent_run, so a colocalization can be
traced back to the detection settings behind it. Dual-channel runs must not
store the 'both channels' count in coloc_positive_cells and the 'neither' count
in coloc_negative_cells -- the columns a single-channel run uses for
positive/negative -- or the row would read as a single-channel result it never
was.
"""
import pandas as pd

from slice_widget_helpers import Control, labels_with_one_nucleus, make_widget

SAMPLE = "PRJ_02_01_S11"


def _rows(w, run_type):
    return [r for r in w.tracker.search(sample_id=SAMPLE, limit=None)
            if r["run_type"] == run_type]


def _detect(w):
    w._run_detection()
    w._on_detect_finished(True, "Detected 1 nuclei", 1, labels_with_one_nucleus())
    # The newest detection row is this one.
    return w.tracker.search(sample_id=SAMPLE, run_type="detection", limit=1)[0]["run_id"]


def test_a_colocalization_row_names_the_detection_it_measured(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    _detect(w)                       # an earlier detection, replaced by the next
    det_id = _detect(w)

    w._run_colocalization()

    (coloc,) = _rows(w, "colocalization")
    assert coloc["parent_run"] == det_id


def test_a_dual_colocalization_row_names_its_detection_and_keeps_four_classes(
        tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    det_id = _detect(w)
    w.coloc_mode_combo = Control(text="Dual Channel")
    w.current_file = None            # no measurements file to write here
    w._visualize_dual_colocalization = lambda m: None

    w._run_colocalization()
    summary = {"total_nuclei": 10, "ch1_name": "red", "ch2_name": "green",
               "n_dual": 4, "n_red_only": 3, "n_green_only": 2, "n_neither": 1}
    w._on_dual_coloc_finished(True, "done", pd.DataFrame({"label": range(10)}), summary, None)

    (coloc,) = _rows(w, "colocalization")
    assert coloc["status"] == "completed"
    assert coloc["parent_run"] == det_id
    assert (coloc["coloc_dual_cells"], coloc["coloc_ch1_only_cells"],
            coloc["coloc_ch2_only_cells"], coloc["coloc_neither_cells"]) == ("4", "3", "2", "1")
    # Not presented as a single-channel positive/negative count.
    assert coloc["coloc_positive_cells"] == "" and coloc["coloc_negative_cells"] == ""

