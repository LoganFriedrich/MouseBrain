"""batch_process's pipeline_status.csv survives a crash and two writers at once.

The file is written before every step, so it must not be rewritten in place
without a lock: a crash part-way would leave it cut short, and two batches
that each read it, change their own brain's row and write it back would drop
each other's change.
"""
import csv
import importlib.util
import threading
import time
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


@pytest.fixture
def bp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("batch_process_lock", PIPELINE / "batch_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PIPELINE_STATUS_CSV", tmp_path / "pipeline_status.csv")
    return mod


def test_a_crash_while_writing_leaves_the_file_as_it_was(bp, monkeypatch):
    bp.update_brain_status("MOUSE_A/BRAIN_1", {"status": "PASS"})
    bp.update_brain_status("MOUSE_A/BRAIN_2", {"status": "PASS"})
    before = bp.PIPELINE_STATUS_CSV.read_bytes()

    class DiskFull(csv.DictWriter):
        def writerow(self, row):
            if row.get("brain_name") == "MOUSE_A/BRAIN_2":
                raise OSError("no space left on device")
            return super().writerow(row)

    with monkeypatch.context() as m:
        m.setattr(csv, "DictWriter", DiskFull)
        with pytest.raises(OSError):
            bp.update_brain_status("MOUSE_A/BRAIN_1", {"status": "FAIL"})
    assert bp.PIPELINE_STATUS_CSV.read_bytes() == before
    assert set(bp.load_pipeline_status()) == {"MOUSE_A/BRAIN_1", "MOUSE_A/BRAIN_2"}


def test_two_writers_at_once_both_keep_their_change(bp, monkeypatch):
    bp.update_brain_status("MOUSE_A/BRAIN_0", {"status": "PASS"})
    real_load = bp.load_pipeline_status

    def slow_load():
        rows = real_load()
        time.sleep(0.3)  # the other writer reads in this window
        return rows

    monkeypatch.setattr(bp, "load_pipeline_status", slow_load)
    errors = []

    def write(name):
        try:
            bp.update_brain_status(name, {"status": "PENDING", "current_step": "detect"})
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=write, args=("MOUSE_A/BRAIN_%d" % i,)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert set(real_load()) == {"MOUSE_A/BRAIN_0", "MOUSE_A/BRAIN_1", "MOUSE_A/BRAIN_2"}
    assert not list(bp.PIPELINE_STATUS_CSV.parent.glob("pipeline_status.csv.*")), \
        "no lock or temporary file is left behind"


_WORKER = r'''
import importlib.util, sys, time, traceback
from pathlib import Path
script, csv_path, me, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("bp_worker", script)
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)
bp.PIPELINE_STATUS_CSV = csv_path
lock = csv_path.with_name(csv_path.name + ".lock")
for k in range(n):
    try:
        bp.update_brain_status("MOUSE_W/BRAIN_%d_%d" % (me, k), {"status": "PENDING"})
    except Exception:
        traceback.print_exc()
        print("FAILED", k)
        continue
    # Right after a release the lock must be gone or belong to someone else.
    try:
        held_by_me = ("pid %d " % __import__("os").getpid()) in lock.read_text()
    except OSError:
        held_by_me = False
    print("LEFT_BEHIND" if held_by_me else "OK", k)
'''


def test_four_processes_writing_at_once_keep_every_row_and_never_stall(tmp_path):
    """Real processes, not threads: on Windows a waiter that opens the lock
    file while polling makes the holder's delete fail. If the failure were
    swallowed, the finished holder's lock would stay, and every writer would
    stall for the two-minute stale takeover -- some updates would then time out
    and their rows would never be written."""
    import os
    import subprocess
    import sys

    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER, encoding="utf-8")
    csv_path = tmp_path / "pipeline_status.csv"
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    n_proc, n_each = 4, 8
    started = time.monotonic()
    procs = [subprocess.Popen([sys.executable, str(worker), str(PIPELINE / "batch_process.py"),
                               str(csv_path), str(i), str(n_each)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
             for i in range(n_proc)]
    outputs = []
    try:
        for p in procs:
            outputs.append(p.communicate(timeout=90)[0])
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    elapsed = time.monotonic() - started
    joined = "\n".join(outputs)
    assert "FAILED" not in joined and "TimeoutError" not in joined, joined
    assert "LEFT_BEHIND" not in joined, joined
    assert "Took over a lock" not in joined, joined
    spec = importlib.util.spec_from_file_location("bp_check", PIPELINE / "batch_process.py")
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    bp.PIPELINE_STATUS_CSV = csv_path
    rows = bp.load_pipeline_status()
    expected = {"MOUSE_W/BRAIN_%d_%d" % (i, k) for i in range(n_proc) for k in range(n_each)}
    assert expected <= set(rows), sorted(expected - set(rows))
    assert not csv_path.with_name(csv_path.name + ".lock").exists()
    assert elapsed < 60, "took %.0f s: a writer stalled on a lock nobody held" % elapsed


def test_a_stale_lock_that_cannot_be_removed_still_ends_the_wait(bp, monkeypatch):
    """A lock left by a crashed batch that cannot be removed (a program keeps
    it open, or the folder refuses it) must not make the batch wait forever:
    the wait ends at its limit with a message naming the lock file, and the
    row is not written without the lock."""
    import functools
    import os

    from mousebrain import file_lock as fl

    lock = bp.PIPELINE_STATUS_CSV.with_name("pipeline_status.csv.lock")
    lock.write_text("a batch that crashed\n", encoding="utf-8")
    long_ago = time.time() - 1000
    os.utime(lock, (long_ago, long_ago))

    class LockCannotBeRemoved:
        def __getattr__(self, name):
            return getattr(os, name)

        def _refuse(self, path):
            if Path(path) == lock:
                raise PermissionError(32, "being used by another process", str(path))

        def rename(self, src, dst):
            self._refuse(src)
            return os.rename(src, dst)

        def unlink(self, path):
            self._refuse(path)
            return os.unlink(path)

    monkeypatch.setattr(fl, "os", LockCannotBeRemoved())
    monkeypatch.setattr(bp, "exclusive_file_lock",
                        functools.partial(fl.exclusive_file_lock, wait_seconds=2))

    outcome = {}

    def write():
        try:
            bp.update_brain_status("MOUSE_A/BRAIN_1", {"status": "PENDING"})
            outcome["result"] = "written"
        except Exception as e:
            outcome["result"] = e

    t = threading.Thread(target=write, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "the batch was still waiting for the lock after 20 s"
    err = outcome.get("result")
    assert isinstance(err, TimeoutError), err
    assert "pipeline_status.csv.lock" in str(err)
    assert "MOUSE_A/BRAIN_1" not in bp.load_pipeline_status()
    assert lock.read_text(encoding="utf-8") == "a batch that crashed\n"


def test_a_lock_left_by_a_crashed_batch_is_taken_over(bp, capsys):
    """A batch ended in the middle of a write leaves its lock behind. The next
    change waits no longer than the stale time, takes the lock over with a
    line saying so, and is written."""
    import os

    lock = bp.PIPELINE_STATUS_CSV.with_name("pipeline_status.csv.lock")
    lock.write_text("a batch that crashed\n", encoding="utf-8")
    long_ago = time.time() - 1000
    os.utime(lock, (long_ago, long_ago))

    bp.update_brain_status("MOUSE_A/BRAIN_1", {"status": "PENDING"})

    assert "Took over a lock" in capsys.readouterr().out
    assert "MOUSE_A/BRAIN_1" in bp.load_pipeline_status()
    assert not list(bp.PIPELINE_STATUS_CSV.parent.glob("pipeline_status.csv.*")), \
        "no lock or temporary file is left behind"
