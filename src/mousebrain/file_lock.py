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
because machines' clocks differ -- is taken over: it is renamed aside (only one
machine's rename can succeed) and checked to still be the lock that was judged
stale before it is removed. A holder that works for a long time calls
``touch()`` on the handle it was given so its lock never looks abandoned.

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


def _take_over_stale_lock(lock: Path, seen: str, stale_seconds: float, label: str) -> bool:
    """Remove *lock* if it is still the one we saw (content *seen*) and it has
    not been touched for *stale_seconds* by the server's clock. Returns True
    when the caller should try to take the lock again."""
    try:
        age = _server_now(lock.parent) - lock.stat().st_mtime
    except FileNotFoundError:
        return True
    if age < stale_seconds:
        return False
    aside = lock.with_name(lock.name + ".stale-" + uuid.uuid4().hex[:8])
    try:
        os.rename(lock, aside)
    except OSError:
        return True
    try:
        content = aside.read_text(encoding="utf-8", errors="replace")
    except OSError:
        content = seen
    if content != seen:
        try:
            os.rename(aside, lock)  # a fresh lock taken meanwhile: not ours to break
        except OSError:
            pass
        return True
    print("[%s] [!] Took over a lock left %.0f s ago by: %s" % (label, age, seen.strip()))
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
            if seen and _take_over_stale_lock(lock, seen, stale_seconds, label):
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
