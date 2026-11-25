"""Two runs of one sample started in the same second get different run ids.

An id hashing only type, sample, second and process id would give a second
run in the same second the first run's id, and update_status() on the second
run would change the first run's row.
"""
import csv

from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker


def test_two_runs_in_one_second_are_updated_separately(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    first = t.log_registration(sample_id="SUBJ_Y_S01", atlas="a", orientation="coronal",
                               method="m", status="started")
    second = t.log_registration(sample_id="SUBJ_Y_S01", atlas="a", orientation="coronal",
                                method="m", status="started")
    assert first != second

    t.update_status(first, status="completed")
    t.update_status(second, status="failed")
    with open(tmp_path / "runs.csv", newline="", encoding="utf-8") as f:
        statuses = {r["run_id"]: r["status"] for r in csv.DictReader(f)}
    assert statuses == {first: "completed", second: "failed"}
