"""The calibration tracker's lock is the shared file lock: kept fresh, let go reliably.

The tracker uses the shared lock of mousebrain.file_lock, not its own copy. A
live lock must be refreshed: otherwise a rewrite of calibration_runs.csv that
takes longer than the stale limit (a slow share) would be judged abandoned, a
second writer would take it over, both would rewrite the file, and the slower
rewrite would drop the other's update. And the release must survive Windows
refusing a delete while another process has the lock file open for a moment (a
waiter reading who holds it), or the lock would stay behind and every later
writer would wait it out.
"""
import csv
import threading
import time

from mousebrain import tracker as tracker_mod
from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker


def _tracker(tmp_path, ids):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for exp_id in ids:
            w.writerow({"exp_id": exp_id, "exp_type": "detection", "status": "started"})
    return ExperimentTracker(path)


def _status(tracker, exp_id):
    return tracker.get_experiment(exp_id)["status"]


def test_a_slow_rewrite_keeps_its_lock_and_loses_no_other_update(tmp_path, monkeypatch):
    t = _tracker(tmp_path, ["slow_run", "quick_run"])
    monkeypatch.setattr(tracker_mod, "LOCK_STALE_SECONDS", 1)
    inside = threading.Event()

    def slow(rows):
        for row in rows:
            if row["exp_id"] == "slow_run":
                row["status"] = "completed"
        inside.set()
        time.sleep(3)  # three times the stale limit: a slow share

    slow_writer = threading.Thread(target=t._rewrite, args=(slow,))
    slow_writer.start()
    assert inside.wait(10)
    # A second writer arrives while the slow rewrite still holds the lock. It
    # must wait for it, never take over a lock that is still in use.
    assert t.update_status("quick_run", status="completed")
    slow_writer.join(30)
    assert _status(t, "slow_run") == "completed"
    assert _status(t, "quick_run") == "completed", (
        "the slow rewrite's lock was taken over and one of the two updates was lost")


def test_the_lock_is_let_go_even_while_a_reader_has_it_open(tmp_path):
    t = _tracker(tmp_path, ["a"])
    opened, done = threading.Event(), threading.Event()

    def reader():
        # What a waiter does several times a second: open the lock to read
        # who holds it. On Windows a file open like this cannot be deleted.
        with open(t._lock_path, "rb"):
            opened.set()
            time.sleep(0.5)
        done.set()

    def mutate(rows):
        rows[0]["status"] = "completed"
        threading.Thread(target=reader).start()
        assert opened.wait(10)

    t._rewrite(mutate)
    done.wait(10)
    assert _status(t, "a") == "completed"
    assert not t._lock_path.exists(), (
        "the lock was left behind because it was open for a moment at release")
