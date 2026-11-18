"""A finished run whose result the tracker file refused is saved once it can be.

When a run finishes while calibration_runs.csv is held open (a spreadsheet
program has it), the napari window cannot write the result and -- so as not
to freeze again -- does not try a second write. It used to drop the result
there, so the row said 'started' for ever although the run had finished. The
window now keeps the result and writes it at the next chance (the next run it
starts, a timer, the program closing), without waiting on a file that still
refuses it, and only while the row still says 'started'.
"""
import contextlib
import sys
import time

import pytest

from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

from slice_widget_helpers import labels_with_one_nucleus, make_widget

SAMPLE = "PRJ_05_06_S11"
METRICS = {"raw_count": 9, "filtered_count": 7, "removed_by_size": 2,
           "removed_by_border": 0, "removed_by_morphology": 0,
           "removed_by_confidence": 0, "backend": "stardist"}

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing rules")


@contextlib.contextmanager
def _held_open_like_a_spreadsheet(path):
    """Hold `path` open the way a spreadsheet program does: others may read
    it, nobody may write, rename over or delete it."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NORMAL = 0x80000000, 1, 3, 0x80
    handle = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None,
                             OPEN_EXISTING, NORMAL, None)
    assert handle not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    try:
        yield
    finally:
        k32.CloseHandle(handle)


def _window(tmp_path, monkeypatch):
    """The stand-in window, its tracker opened with the window's short waits."""
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    w.tracker = SliceTracker(
        csv_path=tmp_path / "runs.csv",
        lock_wait_seconds=main_widget.GUI_TRACKER_LOCK_WAIT_SECONDS,
        replace_retry_seconds=main_widget.GUI_TRACKER_REPLACE_RETRY_SECONDS)
    w._unrecorded_results = {}
    w._unrecorded_retry_timer = None
    return w


def _detection_finished_while_held_open(w):
    """Start a detection, then let it finish while the tracker is held open.
    Returns its run id."""
    w._run_detection()
    run_id = w._detection_run_id
    with _held_open_like_a_spreadsheet(w.tracker.csv_path):
        w._on_detect_finished(True, "Detected 7 nuclei", 7, labels_with_one_nucleus(), METRICS)
    assert w.tracker.get_run(run_id)["status"] == "started"
    assert "still says 'started'" in w.status_label.text()
    return run_id


def test_a_refused_result_is_saved_when_the_next_run_starts(tmp_path, monkeypatch):
    w = _window(tmp_path, monkeypatch)
    first = _detection_finished_while_held_open(w)

    # The spreadsheet is closed; the person starts the next detection.
    w._analysis_running = None
    w._run_detection()

    row = w.tracker.get_run(first)
    assert row["status"] == "completed"
    assert (row["det_nuclei_found"], row["det_raw_count"], row["det_removed_by_size"]) == (
        "7", "9", "2")
    second = w._detection_run_id
    assert second != first and w.tracker.get_run(second)["status"] == "started"


def test_a_try_while_the_file_still_refuses_does_not_wait_and_keeps_the_result(
        tmp_path, monkeypatch):
    w = _window(tmp_path, monkeypatch)
    first = _detection_finished_while_held_open(w)

    with _held_open_like_a_spreadsheet(w.tracker.csv_path):
        started = time.monotonic()
        w._retry_unrecorded_results()
        elapsed = time.monotonic() - started
    assert elapsed < 1.5, "a retry must not freeze the window"
    assert w.tracker.get_run(first)["status"] == "started"

    w._retry_unrecorded_results()  # the file is free now
    assert w.tracker.get_run(first)["status"] == "completed"
    assert first not in w._unrecorded_results


def test_a_kept_result_never_overwrites_a_row_that_was_finished_another_way(
        tmp_path, monkeypatch):
    w = _window(tmp_path, monkeypatch)
    first = _detection_finished_while_held_open(w)
    # Finished some other way meanwhile (e.g. a person's repair).
    other = SliceTracker(csv_path=w.tracker.csv_path)
    other.fail_if_unfinished(first, "repaired by hand")

    w._retry_unrecorded_results()

    row = w.tracker.get_run(first)
    assert row["status"] == "failed" and row["det_nuclei_found"] == ""
    assert first not in w._unrecorded_results


def test_the_program_closing_gives_a_kept_result_its_last_try(tmp_path, monkeypatch, capsys):
    w = _window(tmp_path, monkeypatch)
    first = _detection_finished_while_held_open(w)

    with _held_open_like_a_spreadsheet(w.tracker.csv_path):
        w._last_try_unrecorded_results()
    assert first in capsys.readouterr().out.split("say 'started':")[-1]

    w._last_try_unrecorded_results()
    assert w.tracker.get_run(first)["status"] == "completed"


def test_the_tracker_writes_a_kept_result_only_over_a_started_row(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    started = t.log_detection(sample_id=SAMPLE)
    done = t.log_detection(sample_id=SAMPLE)
    t.update_status(done, status="completed", det_nuclei_found=3)

    assert t.record_result_if_started(started, status="completed", det_nuclei_found=5) is True
    assert t.record_result_if_started(done, status="completed", det_nuclei_found=99) is False
    assert t.record_result_if_started("det_00000000_000000_ffffff", status="completed") is False
    assert t.get_run(started)["det_nuclei_found"] == "5"
    assert t.get_run(done)["det_nuclei_found"] == "3"
    with pytest.raises(ValueError):
        t.record_result_if_started(started, status="completed", det_nucleus_found=1)


def test_a_failed_run_the_file_refused_to_mark_is_marked_failed_later(tmp_path, monkeypatch):
    # A detection that FAILS while the file is held open: marking its row
    # failed is refused too, and the row used to say 'started' for ever.
    w = _window(tmp_path, monkeypatch)
    w._run_detection()
    run_id = w._detection_run_id
    with _held_open_like_a_spreadsheet(w.tracker.csv_path):
        w._on_detect_finished(False, "model could not be loaded", 0, None, {})
    assert w.tracker.get_run(run_id)["status"] == "started"

    w._analysis_running = None
    w._run_detection()  # the spreadsheet is closed; the next run starts

    row = w.tracker.get_run(run_id)
    assert row["status"] == "failed"
    assert "model could not be loaded" in row["notes"]


def test_several_kept_results_wait_together_and_are_all_written_once_the_file_is_free(
        tmp_path, monkeypatch):
    w = _window(tmp_path, monkeypatch)
    ids = [w.tracker.log_detection(sample_id=SAMPLE) for _ in range(3)]
    for n, run_id in enumerate(ids):
        w._unwritable_run_ids.add(run_id)
        w._unrecorded_results[run_id] = {"status": "completed", "det_nuclei_found": n + 1}

    with _held_open_like_a_spreadsheet(w.tracker.csv_path):
        w._retry_unrecorded_results()
    assert sorted(w._unrecorded_results) == sorted(ids)
    assert [w.tracker.get_run(r)["status"] for r in ids] == ["started"] * 3

    w._retry_unrecorded_results()

    assert [w.tracker.get_run(r)["det_nuclei_found"] for r in ids] == ["1", "2", "3"]
    assert not w._unrecorded_results and not (w._unwritable_run_ids & set(ids))
