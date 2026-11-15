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
        lock, "crashed-host pid 1 token a\n", 120, "Test") is True
    assert renamed == []
    assert lock.read_text(encoding="utf-8") == "live-host pid 2 token b\n"
