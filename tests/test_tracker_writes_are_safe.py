"""The calibration tracker is rewritten safely, and runs never stay 'started'.

The tracker is one CSV on a shared drive, written by several machines. Every
status change rewrites the whole file, and that rewrite used to (1) keep only
the columns the running code knew -- so an older copy deleted every newer column
from every row on its first status change -- (2) write in place, so a crash
truncated it, and (3) take no lock, so concurrent updates were lost. And a step
that stopped part-way left its row at 'started' forever.
"""
import csv
import os
import time
from pathlib import Path

import pytest

from mousebrain import tracker as tracker_mod
from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def _tracker_with_header(tmp_path, header, rows):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return ExperimentTracker(path), path


def test_a_column_this_version_does_not_know_survives_a_status_change(tmp_path):
    header = CSV_COLUMNS[:5] + ["column_from_a_newer_version"] + CSV_COLUMNS[5:]
    t, path = _tracker_with_header(tmp_path, header, [
        {"exp_id": "det_1", "exp_type": "detection", "brain": "B", "status": "started",
         "column_from_a_newer_version": "keep me"}])
    assert t.update_status("det_1", status="completed")
    cols, rows = _read(path)
    assert cols == header  # the file's own order, unknown column in place
    assert rows[0]["column_from_a_newer_version"] == "keep me"
    assert rows[0]["status"] == "completed"


def test_a_file_in_another_column_order_keeps_its_order_and_values(tmp_path):
    header = list(reversed(CSV_COLUMNS))
    t, path = _tracker_with_header(tmp_path, header, [
        {"exp_id": "det_1", "brain": "B", "status": "started", "notes": "n", "tags": "t"}])
    t.update_status("det_1", status="failed")
    cols, rows = _read(path)
    assert cols == header
    assert (rows[0]["notes"], rows[0]["tags"], rows[0]["status"]) == ("n", "t", "failed")


def test_no_temporary_or_lock_file_is_left_behind(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"exp_id": "a", "status": "started"}])
    t.update_status("a", status="completed")
    t.log_detection(brain="B")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration_runs.csv"]


def test_a_failed_rewrite_leaves_the_file_as_it_was(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"exp_id": "a", "status": "started"}])
    before = path.read_bytes()

    def boom(rows):
        rows[0]["status"] = "half-written"
        raise RuntimeError("crash mid-update")
    with pytest.raises(RuntimeError):
        t._rewrite(boom)
    assert path.read_bytes() == before
    assert not t._lock_path.exists()


def test_a_held_lock_is_waited_for_and_never_ignored(tmp_path, monkeypatch):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"exp_id": "a", "status": "started"}])
    t._lock_path.write_text("another machine pid 1 at now token x\n")
    monkeypatch.setattr(tracker_mod, "LOCK_WAIT_SECONDS", 1)
    before = path.read_bytes()
    with pytest.raises(TimeoutError):
        t.update_status("a", status="completed")
    assert path.read_bytes() == before
    assert t._lock_path.exists()  # someone else's lock is not ours to remove


def test_a_lock_left_by_a_crashed_writer_is_taken_over(tmp_path, monkeypatch):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"exp_id": "a", "status": "started"}])
    t._lock_path.write_text("crashed machine pid 1 at long ago token y\n")
    old = time.time() - 10 * tracker_mod.LOCK_STALE_SECONDS
    os.utime(t._lock_path, (old, old))
    assert t.update_status("a", status="completed")
    assert _read(path)[1][0]["status"] == "completed"
    assert not t._lock_path.exists()


def test_a_run_that_raises_is_marked_failed(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [])
    exp_id = t.log_detection(brain="B", status="started")
    with pytest.raises(KeyboardInterrupt):
        with t.finish_or_fail(exp_id):
            raise KeyboardInterrupt()
    row = t.get_experiment(exp_id)
    assert row["status"] == "failed" and "KeyboardInterrupt" in row["notes"]


def test_a_run_that_ends_without_a_result_is_marked_failed_and_a_finished_one_is_not(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [])
    forgot = t.log_detection(brain="B", status="started")
    done = t.log_detection(brain="B", status="started")
    with t.finish_or_fail(forgot):
        pass
    with t.finish_or_fail(done):
        t.update_status(done, status="completed")
    assert t.get_experiment(forgot)["status"] == "failed"
    assert t.get_experiment(done)["status"] == "completed"


@pytest.mark.parametrize("script", ["3_register_to_atlas.py", "4_detect_cells.py",
                                    "5_classify_cells.py", "6_count_regions.py"])
def test_every_step_guards_its_tracker_row(script):
    assert "finish_or_fail(" in (PIPELINE / script).read_text(encoding="utf-8")
