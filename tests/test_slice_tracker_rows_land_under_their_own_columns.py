"""A row appended to the 2D tracker lands under its own column names.

Rows used to be appended in the code's column order whatever the file's
header said. After a version added columns (the particle-analysis ones), the
first particle row written to a file with the older header put its values
under the wrong names -- channel names under input_path, an area under
output_path -- and the next rewrite made that permanent.
"""
import csv

from mousebrain.plugin_2d.sliceatlas.tracker.schema import CSV_COLUMNS
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def _check_particle_row(row):
    assert row["pa_detection_channel"] == "chan_det"
    assert row["pa_measurement_channel"] == "chan_meas"
    assert row["pa_max_area"] == "157"
    assert row["input_path"] == "in.nd2"
    assert row["output_path"] == ""
    assert row["marked_best"] == ""
    assert row["hostname"] != ""


def _log_particle(t):
    return t.log_particle_analysis(
        sample_id="PRJ_01_01_S01", detection_channel="chan_det",
        measurement_channel="chan_meas", max_area=157, input_path="in.nd2")


def test_a_file_written_before_the_newest_columns_gets_them_before_a_row_is_added(tmp_path):
    path = tmp_path / "calibration_runs.csv"
    older = [c for c in CSV_COLUMNS if not c.startswith("pa_")]
    _write(path, older, [{"run_id": "det_1", "run_type": "detection",
                          "sample_id": "PRJ_01_01_S01", "status": "completed",
                          "input_path": "old.nd2"}])
    t = SliceTracker(csv_path=path)
    run_id = _log_particle(t)
    t.update_status("det_1", status="completed")  # the rewrite that used to fix the damage in place
    _, rows = _rows(path)
    by_id = {r["run_id"]: r for r in rows}
    _check_particle_row(by_id[run_id])
    assert by_id["det_1"]["input_path"] == "old.nd2"


def test_a_file_in_another_column_order_is_appended_in_its_own_order(tmp_path):
    path = tmp_path / "calibration_runs.csv"
    header = CSV_COLUMNS[:3] + ["column_from_a_newer_version"] + list(reversed(CSV_COLUMNS[3:]))
    _write(path, header, [])
    t = SliceTracker(csv_path=path)
    run_id = _log_particle(t)
    cols, rows = _rows(path)
    assert cols == header
    (row,) = rows
    assert row["run_id"] == run_id and row["column_from_a_newer_version"] == ""
    _check_particle_row(row)


def test_a_header_that_loses_a_column_after_opening_still_gets_the_value(tmp_path):
    """Another version rewrote the file without a column after this tracker
    opened it: the append adds the column back instead of dropping the value."""
    path = tmp_path / "calibration_runs.csv"
    t = SliceTracker(csv_path=path)
    _write(path, [c for c in CSV_COLUMNS if c != "pa_max_area"], [])
    run_id = _log_particle(t)
    _, rows = _rows(path)
    (row,) = rows
    assert row["run_id"] == run_id
    _check_particle_row(row)
