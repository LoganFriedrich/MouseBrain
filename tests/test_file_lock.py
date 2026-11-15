"""mousebrain.file_lock: taking over a lock a crashed holder left behind.

Everything happens in pytest's tmp_path.
"""
import os
import time

import pytest

from mousebrain import file_lock
from mousebrain.file_lock import exclusive_file_lock


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_an_old_empty_lock_is_taken_over(tmp_path):
    # A process that crashed between creating the lock and writing its holder
    # line leaves an EMPTY lock file. It used to be skipped by the takeover,
    # so every later writer waited the full wait time and failed.
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    lock.write_bytes(b"")
    _age(lock, 3600)

    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=2):
        assert lock.read_text(encoding="utf-8").strip()  # ours now
    assert not lock.exists()


def test_a_young_empty_lock_is_waited_for(tmp_path):
    # A holder that has just created its lock and not yet written to it.
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    lock.write_bytes(b"")
    with pytest.raises(TimeoutError):
        with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=0.5):
            pass
    assert lock.exists()


def test_a_lock_retaken_since_it_was_judged_stale_is_never_moved(tmp_path, monkeypatch):
    # The lock read as stale, but by the time of the takeover another process
    # has taken it afresh (its content changed). The takeover must leave it
    # where it is: moving it aside and back could race a third process.
    lock = tmp_path / "registry.json.lock"
    lock.write_text("crashed-host pid 1 token a\n", encoding="utf-8")
    _age(lock, 3600)
    renamed = []
    real_rename = os.rename

    def rename(src, dst, *a, **k):
        renamed.append((src, dst))
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(file_lock.os, "rename", rename)
    lock.write_text("live-host pid 2 token b\n", encoding="utf-8")
    _age(lock, 3600)  # the new holder has not touched it yet

    assert file_lock._take_over_stale_lock(
        lock, "crashed-host pid 1 token a\n", 120, "Test") == file_lock._TRY_NOW
    assert renamed == []
    assert lock.read_text(encoding="utf-8") == "live-host pid 2 token b\n"


def _open_for(path, seconds):
    """Open *path* the way a waiter reading the holder line does, and close
    it after *seconds* (on Windows the holder cannot delete it meanwhile)."""
    import threading
    f = open(path, "rb")
    timer = threading.Timer(seconds, f.close)
    timer.start()
    return timer


def test_a_release_while_a_waiter_is_reading_the_lock_still_removes_it(tmp_path):
    # A waiter reads the lock several times a second. Python opens files
    # without delete sharing, so on Windows a release that tried the delete
    # once failed whenever it coincided with such a read, left the lock
    # behind, and every writer then waited for it to age out.
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=2):
        timer = _open_for(lock, 0.5)
    timer.join()
    assert not lock.exists()


def test_a_release_never_removes_a_lock_someone_else_holds_now(tmp_path):
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=2):
        # Taken over and taken afresh by another process while we worked.
        lock.write_text("other-host pid 9 token z\n", encoding="utf-8")
    assert lock.read_text(encoding="utf-8") == "other-host pid 9 token z\n"


def test_the_lock_and_its_clock_probe_can_live_in_another_folder(tmp_path):
    target = tmp_path / "data" / "registry.json"
    target.parent.mkdir()
    lock = tmp_path / "work" / "data.lock"
    lock.parent.mkdir()
    lock.write_text("crashed-host pid 1 token a\n", encoding="utf-8")
    _age(lock, 3600)
    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=2,
                             lock_path=lock):
        assert lock.exists()
        assert list(target.parent.iterdir()) == []  # nothing beside the target
    assert not lock.exists()
    assert list(target.parent.iterdir()) == []
    assert list(lock.parent.iterdir()) == []  # probes and the old lock cleaned up


def _run_with_limit(fn, limit):
    """Run *fn* in a thread; return (finished, exception). A loop that never
    ends must fail the test, not hang the suite."""
    import threading
    out = {}

    def run():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - reported to the test
            out["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(limit)
    return (not t.is_alive()), out.get("error")


def test_a_folder_this_account_cannot_create_files_in_times_out(tmp_path, monkeypatch):
    # The create fails with PermissionError and there is no lock file (a share
    # or ACL that refuses new files). This used to retry at once forever, a
    # busy loop that never reached the wait limit.
    def refuse(*a, **k):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(file_lock.os, "open", refuse)
    target = tmp_path / "registry.json"

    def take():
        with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=1):
            pass

    started = time.monotonic()
    finished, error = _run_with_limit(take, 10)
    assert finished, "the waiter never gave up"
    assert isinstance(error, TimeoutError)
    assert "permission denied" in str(error)
    assert time.monotonic() - started < 5


def test_a_stale_lock_that_cannot_be_moved_times_out(tmp_path, monkeypatch):
    # A lock left by a crashed holder that this account may not rename
    # (another program has it open, or an ACL refuses): wait and give up,
    # never spin on the refused rename.
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    lock.write_text("crashed-host pid 1 token a\n", encoding="utf-8")
    _age(lock, 3600)
    attempts = []

    def refuse(src, dst, *a, **k):
        attempts.append(src)
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(file_lock.os, "rename", refuse)

    def take():
        with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=1):
            pass

    finished, error = _run_with_limit(take, 10)
    assert finished, "the waiter never gave up"
    assert isinstance(error, TimeoutError)
    assert len(attempts) < 50  # waited between attempts, did not spin
    assert lock.exists()


def test_a_touched_lock_is_dated_by_the_file_server_not_this_machines_clock(tmp_path, monkeypatch):
    # A waiter judges a lock's age by the file server's clock. Setting the
    # modification time with os.utime(path, None) dates it by THIS machine's
    # clock instead, so a holder whose clock ran behind the server's made its
    # own live lock look abandoned the moment it touched it. Simulate that
    # clock: any time stamped by this machine is 1000 s behind.
    real_utime = os.utime

    def utime_with_a_slow_clock(path, times=None, **kwargs):
        if times is None:
            behind = time.time() - 1000
            times = (behind, behind)
        return real_utime(path, times, **kwargs)

    monkeypatch.setattr(file_lock.os, "utime", utime_with_a_slow_clock)
    target = tmp_path / "registry.json"
    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=2) as handle:
        _age(handle.path, 3600)  # not refreshed for an hour...
        handle.touch()           # ...until now
        seen = handle.path.read_text(encoding="utf-8")
        verdict = file_lock._take_over_stale_lock(handle.path, seen, 120, "Test")
        assert verdict == file_lock._WAIT, "a just-touched live lock was judged abandoned"
        assert handle.path.read_text(encoding="utf-8") == seen  # holder line unchanged


def test_a_touch_never_writes_into_a_lock_someone_else_holds(tmp_path):
    target = tmp_path / "registry.json"
    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=2) as handle:
        handle.path.write_text("other-host pid 2 token b\n", encoding="utf-8")
        _age(handle.path, 3600)
        assert handle.touch() is False
        assert handle.path.read_text(encoding="utf-8") == "other-host pid 2 token b\n"
        assert time.time() - handle.path.stat().st_mtime > 3000
        handle.path.unlink()  # let the test's release find nothing of ours


def test_a_lock_held_through_one_long_step_never_looks_abandoned(tmp_path):
    # One step inside the block (a large file copied over a slow share) can
    # last longer than the stale age without the holder ever calling
    # touch(). The lock must stay fresh by itself meanwhile.
    target = tmp_path / "registry.json"
    with exclusive_file_lock(target, "Test", stale_seconds=1, wait_seconds=2) as handle:
        seen = handle.path.read_text(encoding="utf-8")
        _age(handle.path, 3600)
        time.sleep(1.0)  # the long step; the holder does not touch the lock
        verdict = file_lock._take_over_stale_lock(handle.path, seen, 1, "Test")
        assert verdict == file_lock._WAIT, "a live lock was taken over during one long step"
    assert not handle.path.exists()
