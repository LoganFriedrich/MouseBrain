"""'Load best detection' in Setup & Tuning loads a Best run made in the widget.

A finished widget detection run records the file it wrote (not its run
FOLDER) as its output; an older row's folder is resolved to the file inside
it. Load best reads that as a candidates file, so it loads the Best run
whether it came from the widget or from a script, and the loaded layer records
the brain, the run and its role like every run layer.
"""
from pathlib import Path

import pytest

pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
pytest.importorskip("napari")

from tuning_widget_stand_in import BRAIN_B, _row, _tracker, _widget  # noqa: E402


_XML = ('<?xml version="1.0" encoding="UTF-8"?><CellCounter_Marker_File><Marker_Data>'
        '<Marker_Type><Type>2</Type>'
        '<Marker><MarkerX>5</MarkerX><MarkerY>6</MarkerY><MarkerZ>7</MarkerZ></Marker>'
        '<Marker><MarkerX>8</MarkerX><MarkerY>9</MarkerY><MarkerZ>10</MarkerZ></Marker>'
        '</Marker_Type></Marker_Data></CellCounter_Marker_File>')


def test_load_best_loads_a_widget_run_recorded_by_its_folder(tmp_path, monkeypatch):
    run_dir = tmp_path / "optimization_runs" / BRAIN_B / "test_20000101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "detected_cells.xml").write_text(_XML, encoding="utf-8")
    t = _tracker(tmp_path, [{"exp_id": "w_best", "brain": BRAIN_B, "marked_best": "True",
                             "output_path": str(run_dir)}])
    w = _widget(tmp_path, monkeypatch, t)
    assert w._load_detection_results(best=True)
    layer = w.viewer.layers[-1]
    assert len(layer.data) == 2
    assert layer.metadata.get("exp_id") == "w_best"
    assert layer.metadata.get("cell_type") == "det_best"
    assert layer.metadata.get("source_brain") == str(w.current_brain)


class _Btn:
    def setEnabled(self, *a):
        pass

    def setText(self, *a):
        pass


def test_a_finished_widget_run_records_its_file_and_its_layer(tmp_path, monkeypatch):
    from brainglobe_utils.cells.cells import Cell
    t = _tracker(tmp_path, [])
    run_dir = tmp_path / "optimization_runs" / BRAIN_B / "test_20000101_000000"
    run_dir.mkdir(parents=True)
    exp_id = t.log_detection(brain=BRAIN_B, output_path=str(run_dir), status="started")
    w = _widget(tmp_path, monkeypatch, t)
    w.test_detection_btn = _Btn()
    w.last_run_id = exp_id
    w._last_output_path = run_dir
    w._detection_z_offset = 0
    w._refresh_prefilter_runs = lambda: None
    w._generate_detection_qc_image = lambda: None
    cells = [Cell((5, 6, 7), Cell.UNKNOWN), Cell((8, 9, 10), Cell.UNKNOWN)]
    w._on_detection_finished(True, "done", len(cells), cells)
    recorded = Path(_row(t, exp_id)["output_path"])
    assert recorded == run_dir / "detected_cells.xml" and recorded.is_file()
    layer = w.viewer.layers[-1]
    assert layer.metadata.get("exp_id") == exp_id
    assert layer.metadata.get("cell_type") == "det_recent"
    assert layer.metadata.get("source_brain") == str(w.current_brain)
