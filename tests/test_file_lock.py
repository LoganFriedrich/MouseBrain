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
    # line leaves an EMPTY lock file. The takeover must not skip it, or every
    # later writer would wait the full wait time and fail.
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
    # or ACL that refuses new files). It must not retry at once forever, a
    # busy loop that never reaches the wait limit.
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


def test_a_stale_lock_that_cannot_be_removed_times_out(tmp_path, monkeypatch):
    # A lock left by a crashed holder that this account may not delete
    # (another program has it open, or an ACL refuses): wait and give up,
    # never spin on the refused delete.
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    lock.write_text("crashed-host pid 1 token a\n", encoding="utf-8")
    _age(lock, 3600)
    attempts = []

    real_unlink = os.unlink

    def refuse(path, *a, **k):
        if str(path) != str(lock):
            return real_unlink(path, *a, **k)  # the breaker file may go
        attempts.append(path)
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(file_lock.os, "unlink", refuse)

    def take():
        with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=1):
            pass

    finished, error = _run_with_limit(take, 10)
    assert finished, "the waiter never gave up"
    assert isinstance(error, TimeoutError)
    assert attempts, "the takeover was never tried"
    assert len(attempts) < 50  # waited between attempts, did not spin
    assert lock.exists()
    assert not (tmp_path / "registry.json.lock.break").exists()


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


# ---------------------------------------------------------------------------
# Several waiters on one abandoned lock: takeovers happen one at a time
# ---------------------------------------------------------------------------

class _OsWithARace:
    """Stands in for the ``os`` module inside file_lock and replays, step by
    step, what two other processes do while one waiter takes over a lock.

    *before_removal* runs once, at the last moment before the waiter under
    test acts on the lock (when it asks for the breaker, or -- in the older
    scheme -- when it moves the lock aside). *while_name_is_free* runs right
    after the lock has been moved away from its name, if that ever happens."""

    def __init__(self, lock, before_removal, while_name_is_free):
        self._lock = str(lock)
        self._before = before_removal
        self._free = while_name_is_free
        self._fired = False

    def __getattr__(self, name):
        return getattr(os, name)

    def _fire(self):
        if not self._fired:
            self._fired = True
            self._before()

    def open(self, path, *a, **k):
        if str(path) == self._lock + ".break":
            self._fire()
        return os.open(path, *a, **k)

    def rename(self, src, dst, *a, **k):
        if str(src) == self._lock:
            self._fire()
        result = os.rename(src, dst, *a, **k)
        if str(src) == self._lock:
            self._free()
        return result


def test_a_fresh_lock_taken_during_a_takeover_is_never_moved_or_removed(tmp_path, monkeypatch):
    # Waiters A and B both judge the crashed holder's lock abandoned. A is
    # quicker: it removes the old lock and takes a fresh one of its own just
    # as B is about to act. B must not move A's fresh lock aside, notice, and
    # put it back: while it was aside a third writer C could create the
    # lock, so the put-back would fail, A and C would both be writing (a lost
    # row), and A's lock would be stranded under another name.
    lock = tmp_path / "registry.json.lock"
    crashed = "crashed-host pid 1 token a\n"
    fresh = "host-of-A pid 2 token b\n"
    lock.write_bytes(crashed.encode("utf-8"))
    _age(lock, 3600)
    third_writer_got_in = []

    def waiter_a_takes_over():
        os.unlink(lock)
        lock.write_bytes(fresh.encode("utf-8"))

    def waiter_c_tries_to_create():
        try:
            os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            third_writer_got_in.append(True)
        except FileExistsError:
            pass

    monkeypatch.setattr(file_lock, "os", _OsWithARace(
        lock, waiter_a_takes_over, waiter_c_tries_to_create))
    verdict = file_lock._take_over_stale_lock(lock, crashed, 120, "waiter B",
                                              server_offset=0.0)
    monkeypatch.undo()

    assert not third_writer_got_in, "the lock's name was free while A held the lock"
    assert lock.read_bytes().decode("utf-8") == fresh, "A's fresh lock was disturbed"
    assert sorted(p.name for p in tmp_path.iterdir()) == [lock.name]
    assert verdict != file_lock._GONE


def test_a_fresh_empty_lock_taken_during_a_takeover_is_left_alone(tmp_path, monkeypatch):
    # The same race, where the abandoned lock was EMPTY (its holder crashed
    # before writing its line) and A's fresh lock is, for an instant, empty
    # too: the holder line cannot tell them apart, the modification time does.
    lock = tmp_path / "registry.json.lock"
    lock.write_bytes(b"")
    _age(lock, 3600)

    def waiter_a_takes_over():
        os.unlink(lock)
        lock.write_bytes(b"")  # created, line not yet written

    monkeypatch.setattr(file_lock, "os", _OsWithARace(lock, waiter_a_takes_over, lambda: None))
    verdict = file_lock._take_over_stale_lock(lock, "", 120, "waiter B", server_offset=0.0)
    monkeypatch.undo()

    assert lock.exists(), "A's fresh lock was removed"
    assert verdict == file_lock._WAIT
    assert sorted(p.name for p in tmp_path.iterdir()) == [lock.name]


def test_a_waiter_leaves_the_lock_alone_while_another_is_taking_it_over(tmp_path):
    # Only the waiter holding <lock>.break may remove the lock.
    lock = tmp_path / "registry.json.lock"
    breaker = tmp_path / "registry.json.lock.break"
    crashed = "crashed-host pid 1 token a\n"
    lock.write_bytes(crashed.encode("utf-8"))
    _age(lock, 3600)
    breaker.write_bytes(b"")  # another waiter is in the middle of a takeover

    assert file_lock._take_over_stale_lock(lock, crashed, 120, "Test") == file_lock._WAIT
    assert lock.read_bytes().decode("utf-8") == crashed
    assert breaker.exists(), "a live breaker was removed"


def test_a_breaker_left_by_a_process_that_died_does_not_block_takeovers(tmp_path, capsys):
    # A process that died in the few milliseconds it held <lock>.break leaves
    # it behind. Without a way out, no abandoned lock could ever be taken
    # over again and every writer would time out.
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    breaker = tmp_path / "registry.json.lock.break"
    lock.write_text("crashed-host pid 1 token a\n", encoding="utf-8")
    _age(lock, 3600)
    breaker.write_bytes(b"")
    _age(breaker, 3600)

    with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=5):
        assert "crashed-host" not in lock.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "Removed registry.json.lock.break" in out
    assert "Took over a lock" in out
    assert list(tmp_path.iterdir()) == []


def test_many_waiters_on_one_abandoned_lock_never_hold_it_together(tmp_path):
    # Real threads racing for real: every one of them judges the same
    # abandoned lock stale at the same moment.
    import threading
    target = tmp_path / "registry.json"
    lock = tmp_path / "registry.json.lock"
    inside = []
    together = []
    errors = []
    guard = threading.Lock()

    def writer(start):
        try:
            start.wait()
            with exclusive_file_lock(target, "Test", stale_seconds=120, wait_seconds=30):
                with guard:
                    inside.append(1)
                    if len(inside) > 1:
                        together.append(len(inside))
                time.sleep(0.02)
                with guard:
                    inside.pop()
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    for _ in range(3):
        lock.write_text("crashed-host pid 1 token a\n", encoding="utf-8")
        _age(lock, 3600)
        start = threading.Event()
        threads = [threading.Thread(target=writer, args=(start,), daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(40)
        assert not any(t.is_alive() for t in threads)
        assert not errors, errors
        assert not together, "two writers held the lock at once"
        assert list(tmp_path.iterdir()) == []
