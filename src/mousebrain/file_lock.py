"""An exclusive lock file for a file that several machines rewrite.

WHY THIS EXISTS
A file on a shared drive that is read, changed and written back (a manifest, a
CSV) loses an update whenever two processes do that at the same time: both read
the old content, each adds its own change, and the second write replaces the
first. Writing through a temporary file and os.replace only prevents a TORN
file; it does not prevent a LOST update. Holding this lock around the whole
read-change-write does.

HOW IT WORKS
The lock is a file ``<target>.lock`` created with O_EXCL, which the file server
grants to exactly one process on one machine. It holds a line naming its holder
(machine, process id, time, a random token). The holder removes it when done,
and only if it still holds its own token.

A holder that crashed leaves the lock behind. A lock that has not been touched
for ``stale_seconds`` -- measured by the FILE SERVER's clock (the modification
time of a probe file written there just now), never this machine's clock,
because machines' clocks differ -- is taken over: it is read again to check it
is still the lock that was judged stale, renamed aside (only one machine's
rename can succeed), and checked once more before it is removed. A holder that
works for a long time calls ``touch()`` on the handle it was given so its lock
never looks abandoned.

An EMPTY (or unreadable) lock is judged by its age the same way. WHY: a
process that crashed between creating the lock and writing its holder line
leaves an empty file; never taking that over made every later writer wait the
full ``wait_seconds`` and fail, one after another.

A waiter that cannot get the lock within ``wait_seconds`` raises TimeoutError
naming the holder and the lock file, so a person can see what is holding it.

This is the same algorithm the calibration tracker uses for its CSV
(mousebrain.tracker), in a form any module can use.
"""

import os
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union

__all__ = ["exclusive_file_lock", "LockHandle"]


def _server_now(folder: Path) -> float:
    """The file server's current time: the modification time of a file
    written there just now."""
    probe = folder / (".clock-%s" % uuid.uuid4().hex[:8])
    try:
        probe.write_bytes(b"")
        return probe.stat().st_mtime
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def _read_or(path: Path, default: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def _take_over_stale_lock(lock: Path, seen: str, stale_seconds: float, label: str) -> bool:
    """Remove *lock* if it is still the one we saw (content *seen*, which may
    be empty) and it has not been touched for *stale_seconds* by the server's
    clock. Returns True when the caller should try to take the lock again."""
    try:
        age = _server_now(lock.parent) - lock.stat().st_mtime
    except FileNotFoundError:
        return True
    except OSError:
        return False  # cannot judge its age now (e.g. a delete pending): wait
    if age < stale_seconds:
        return False
    # Check it is still the same lock BEFORE renaming it, so a lock another
    # process has just taken is (almost always) never moved at all. Moving a
    # fresh lock and putting it back afterwards could fail if a third
    # process created a lock in between, leaving two writers each believing
    # they hold it.
    if _read_or(lock, seen) != seen:
        return True
    aside = lock.with_name(lock.name + ".stale-" + uuid.uuid4().hex[:8])
    try:
        os.rename(lock, aside)
    except OSError:
        return True
    content = _read_or(aside, seen)
    if content != seen:
        # A fresh lock was taken in the instant between the check and the
        # rename: not ours to break.
        try:
            os.rename(aside, lock)
        except OSError:
            print("[%s] [!] Moved a live lock aside by mistake and could not put it "
                  "back (%s); its holder (%s) and another writer may both be "
                  "writing. Check the file it protects." % (
                      label, aside, content.strip() or "?"))
        return True
    print("[%s] [!] Took over a lock left %.0f s ago by: %s"
          % (label, age, seen.strip() or "(an empty lock file)"))
    try:
        aside.unlink()
    except OSError:
        pass
    return True


class LockHandle:
    """What a holder gets inside ``with exclusive_file_lock(...)``."""

    def __init__(self, path: Path):
        self.path = path

    def touch(self) -> None:
        """Mark the lock as still in use (refreshes its modification time), so
        a long piece of work is not mistaken for a crashed holder."""
        try:
            os.utime(self.path, None)
        except OSError:
            pass


@contextmanager
def exclusive_file_lock(
    target: Union[str, Path],
    label: str,
    stale_seconds: float = 120,
    wait_seconds: float = 300,
) -> Iterator[LockHandle]:
    """Hold ``<target>.lock`` for the duration of the block.

    Args:
        target: The file being protected; the lock sits beside it.
        label: Short name used in printed messages (ASCII).
        stale_seconds: A lock untouched for this long (server clock) is
            treated as left behind by a crashed holder and taken over.
        wait_seconds: How long to wait for a live holder before raising
            TimeoutError.
    """
    target = Path(target)
    lock = target.with_name(target.name + ".lock")
    mine = "%s pid %d at %s token %s\n" % (
        socket.gethostname(), os.getpid(),
        datetime.now().isoformat(timespec="seconds"), uuid.uuid4().hex)
    deadline = time.monotonic() + wait_seconds
    told = False
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(mine)
            break
        except (FileExistsError, PermissionError):
            try:
                seen = lock.read_text(encoding="utf-8", errors="replace")
            except OSError:
                seen = ""
            # An empty or unreadable lock is judged by its age too (see the
            # module docstring); a holder that has just created it is far
            # younger than stale_seconds, so it is never taken over.
            if _take_over_stale_lock(lock, seen, stale_seconds, label):
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "%s is locked by %s for longer than %d s. If nothing is writing "
                    "to it, delete %s." % (
                        target, seen.strip() or "another process", wait_seconds, lock))
            if not told:
                print("[%s] Waiting for the lock on %s (held by %s)"
                      % (label, target.name, seen.strip() or "?"))
                told = True
            time.sleep(0.2)
    try:
        yield LockHandle(lock)
    finally:
        try:
            if lock.read_text(encoding="utf-8", errors="replace") == mine:
                lock.unlink()
        except OSError:
            pass
