"""The 2D calibration tracker is rewritten safely.

The 2D tracker is one CSV that the napari widget and the batch script both
write. Every status change rewrites the whole file, so the rewrite must not
(1) write in place (a crash part-way would truncate the tracker), (2) take no
lock (a writer that had read the file before another writer's change would
write it back without that change), or (3) keep only the columns the running
code knows (a column added by a newer version would vanish on the first status
change).
"""
import csv
import os
import time

import pytest

from mousebrain.plugin_2d.sliceatlas.tracker import slice_tracker as st_mod
from mousebrain.plugin_2d.sliceatlas.tracker.schema import CSV_COLUMNS
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker


def _lock_of(path):
    return path.with_name(path.name + ".lock")


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
    return SliceTracker(csv_path=path), path


def test_a_column_this_version_does_not_know_survives_a_status_change(tmp_path):
    header = CSV_COLUMNS[:5] + ["column_from_a_newer_version"] + CSV_COLUMNS[5:]
    t, path = _tracker_with_header(tmp_path, header, [
        {"run_id": "det_1", "run_type": "detection", "sample_id": "S", "status": "started",
         "column_from_a_newer_version": "keep me"}])
    assert t.update_status("det_1", status="completed")
    cols, rows = _read(path)
    assert cols == header
    assert rows[0]["column_from_a_newer_version"] == "keep me"
    assert rows[0]["status"] == "completed"


def test_no_temporary_or_lock_file_is_left_behind(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"run_id": "a", "status": "started"}])
    t.update_status("a", status="completed")
    t.log_detection(sample_id="PRJ_01_01_S01")
    t.mark_as_best("a")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration_runs.csv"]


def test_a_failed_rewrite_leaves_the_file_as_it_was(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"run_id": "a", "status": "started"}])
    before = path.read_bytes()

    def boom(rows):
        rows[0]["status"] = "half-written"
        raise RuntimeError("crash mid-update")
    with pytest.raises(RuntimeError):
        t._rewrite(boom)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration_runs.csv"]


def test_a_crash_while_writing_the_new_file_leaves_the_old_one_whole(tmp_path, monkeypatch):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [
        {"run_id": "a", "status": "started"}, {"run_id": "b", "status": "started"}])
    before = path.read_bytes()
    real_to_list = csv.DictWriter._dict_to_list
    calls = {"n": 0}

    def dies_on_the_second_row(self, rowdict):  # both writerow and writerows go through it
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk went away")
        return real_to_list(self, rowdict)
    monkeypatch.setattr(csv.DictWriter, "_dict_to_list", dies_on_the_second_row)
    with pytest.raises(OSError):
        t.update_status("a", status="completed")
    monkeypatch.undo()
    assert path.read_bytes() == before, "an in-place rewrite would have truncated the tracker"


def test_a_held_lock_is_waited_for_and_never_ignored(tmp_path, monkeypatch):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"run_id": "a", "status": "started"}])
    _lock_of(path).write_text("another machine pid 1 at now token x\n")
    monkeypatch.setattr(st_mod, "LOCK_WAIT_SECONDS", 1, raising=False)
    before = path.read_bytes()
    with pytest.raises(TimeoutError):
        t.update_status("a", status="completed")
    with pytest.raises(TimeoutError):
        t.log_detection(sample_id="PRJ_01_01_S01")
    assert path.read_bytes() == before
    assert _lock_of(path).exists()  # someone else's lock is not ours to remove


def test_a_lock_left_by_a_crashed_writer_is_taken_over(tmp_path):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [{"run_id": "a", "status": "started"}])
    _lock_of(path).write_text("crashed machine pid 1 at long ago token y\n")
    old = time.time() - 3600
    os.utime(_lock_of(path), (old, old))
    assert t.update_status("a", status="completed")
    assert _read(path)[1][0]["status"] == "completed"
    assert not _lock_of(path).exists()


def test_a_second_writer_does_not_lose_the_first_writers_row(tmp_path):
    """Two trackers on one file (the widget and the batch): every row and
    every update of both is kept."""
    path = tmp_path / "calibration_runs.csv"
    widget = SliceTracker(csv_path=path)
    batch = SliceTracker(csv_path=path)
    w = widget.log_detection(sample_id="PRJ_01_01_S01")
    b = batch.log_detection(sample_id="PRJ_01_01_S02")
    widget.update_status(w, status="completed", det_nuclei_found=3)
    batch.update_status(b, status="completed", det_nuclei_found=4)
    rows = {r["run_id"]: r for r in _read(path)[1]}
    assert rows[w]["det_nuclei_found"] == "3" and rows[b]["det_nuclei_found"] == "4"


def test_moving_the_best_mark_is_one_write(tmp_path, monkeypatch):
    t, path = _tracker_with_header(tmp_path, CSV_COLUMNS, [
        {"run_id": "old", "run_type": "detection", "sample_id": "S", "status": "completed",
         "marked_best": "True"},
        {"run_id": "new", "run_type": "detection", "sample_id": "S", "status": "completed"}])
    replaces = []
    real = os.replace
    monkeypatch.setattr(st_mod.os, "replace", lambda a, b: (replaces.append(a), real(a, b)))
    assert t.mark_as_best("new")
    rows = {r["run_id"]: r for r in _read(path)[1]}
    assert rows["old"]["marked_best"] == "" and rows["new"]["marked_best"] == "True"
    assert len(replaces) == 1, "the old mark and the new one must change together"
