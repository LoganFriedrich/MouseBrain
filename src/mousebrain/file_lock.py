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
because machines' clocks differ -- is taken over (see the next section).

TAKING OVER AN ABANDONED LOCK: ONE BREAKER AT A TIME
Several waiters usually judge the same abandoned lock stale at the same
moment. If each were free to remove "the stale lock", the slower one would
remove the FRESH lock the quicker one had just taken in its place, and two
writers would be inside the read-change-write at once -- the lost update this
lock exists to prevent. (An earlier version moved the lock aside, looked at
it, and put it back when it turned out to be a fresh one; while it was aside
the lock's name was free, a third writer could create it, and the put-back
then failed.) So:

- a waiter that wants to break a lock first creates ``<lock>.break`` with
  O_EXCL -- only one waiter on one machine can hold it;
- holding it, it looks at the lock AGAIN (its age and its holder line) and
  deletes it only if it is still the very lock it judged abandoned;
  otherwise it touches nothing;
- it never moves a lock aside and never puts one back, and it does no other
  work on the shared drive while it holds the breaker (the file server's
  clock was read beforehand).

A second waiter can therefore look only after the first has finished, and by
then it sees no lock or a fresh one (a different holder line, a new
modification time) and leaves it alone.

A breaker's own crash leaves ``<lock>.break`` behind. It is held for a few
milliseconds, so one untouched for max(stale_seconds, BREAKER_STALE_SECONDS)
is deleted (checking first that it is still the same file) and the takeover
is tried again on a later pass, never in the same one.

WHAT THIS DOES NOT PROTECT AGAINST: a holder that stopped completely for
longer than ``stale_seconds`` (its machine asleep, its link to the share
down) and then carries on. Its lock was rightly judged abandoned and taken
over, and it does not know. No lock file with a timeout can prevent that;
the background refresh below makes it need a whole-process stall, not merely
a slow step.

KEEPING A LIVE LOCK FRESH
While the block runs, a background thread refreshes the lock every quarter
of ``stale_seconds``, so no single long step inside it (one large file copied
over a slow share, a long pass over many entries) can make a live lock look
abandoned. A holder may also call ``touch()`` on the handle it was given.

A refresh REWRITES the holder line through an open handle; it never sets the
modification time with os.utime. WHY: os.utime(path, None) stamps the file
with THIS machine's clock (on Windows CPython reads the local system time),
while a waiter judges the lock's age by the file server's clock. A holder
whose clock ran two minutes behind the server's would make its own live lock
look abandoned the moment it touched it, and a waiter would take it over --
two writers at once, the lost update this lock exists to prevent. A write is
timestamped by the file server itself. The refresh first checks the lock
still holds this holder's line, and the handle it keeps open meanwhile stops
anyone renaming or deleting the file under it, so it never rewrites a lock
someone else has taken over.

An EMPTY (or unreadable) lock is judged by its age the same way. WHY: a
process that crashed between creating the lock and writing its holder line
leaves an empty file; never taking that over would make every later writer
wait the full ``wait_seconds`` and fail, one after another.

A waiter that cannot get the lock within ``wait_seconds`` raises TimeoutError
naming the holder and the lock file, so a person can see what is holding it.
An account that may not create files in the lock's folder at all waits the
same way and gets a TimeoutError saying "permission denied" -- every path
round the waiting loop that does not sleep is bounded by ``wait_seconds``.

RELEASING ON WINDOWS
Windows refuses to delete a file while another process has it open, and a
waiter opens the lock for a moment several times a second to read who holds
it. A release that tried the delete once would fail now and then, leave the
lock behind, and make every waiter sit out ``stale_seconds`` and then "take
over" a lock whose holder had finished long ago. So the release retries
the delete for up to RELEASE_RETRY_SECONDS, re-reading the lock before each
attempt so it never deletes a lock someone else has taken over since, and a
waiter's read of the lock is a single open-read-close. A lock that still
cannot be deleted is reported with a [!] line naming it.

WHERE THE LOCK LIVES
By default ``<target>.lock`` beside the file it protects. A caller whose
folder is copied wholesale by someone else (the analysis registry: an
integrator mirrors every file under it) passes ``lock_path`` to keep the
lock, its clock probes and taken-over leftovers out of that folder.

FILES IT MAY LEAVE IN THE LOCK'S FOLDER
- ``<lock>`` itself, only while a holder holds it (or after a crash, until
  it is taken over);
- ``.clock-<8 hex>``: the clock probe, deleted at once; one can stay
  behind only if the process dies in that instant;
- ``<lock>.break``: held for a few milliseconds by a waiter taking over an
  abandoned lock; it stays behind only if that process dies in that
  instant, and is then cleared by a later waiter (see above).
All three are safe to delete when nothing is holding the lock.

The calibration tracker (mousebrain.tracker) keeps its own, older copy of
this algorithm for its CSV; it does not use this module.
"""

import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

__all__ = ["exclusive_file_lock", "release_lock", "LockHandle", "RELEASE_RETRY_SECONDS",
           "BREAKER_STALE_SECONDS"]

# How long a release keeps retrying the delete while another process has the
# lock file open for a moment (see "RELEASING ON WINDOWS" above).
RELEASE_RETRY_SECONDS = 10

# A ``<lock>.break`` file (see "TAKING OVER AN ABANDONED LOCK") is held for a
# few milliseconds. One older than this -- or than the lock's own stale time,
# whichever is longer -- was left by a process that died holding it.
BREAKER_STALE_SECONDS = 60


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


def _read_or(path: Path, default):
    """The lock's content, opened, read and closed in one go: while a reader
    has it open the holder cannot delete it on Windows (see "RELEASING ON
    WINDOWS"). *default* when it cannot be read."""
    try:
        with open(path, "rb") as f:
            data = f.read(4096)
        return data.decode("utf-8", errors="replace")
    except OSError:
        return default


# What _take_over_stale_lock tells the waiter to do next.
_TRY_NOW = "try now"   # the lock was taken over, or changed hands: try at once
_GONE = "gone"         # there is no lock file (anymore)
_WAIT = "wait"         # a live (or unjudgeable) lock: wait, then try again


def _take_over_stale_lock(lock: Path, seen: str, stale_seconds: float, label: str,
                          server_offset: Optional[float] = None) -> str:
    """Remove *lock* if it is still the one we saw (content *seen*, which may
    be empty) and it has not been touched for *stale_seconds* by the server's
    clock. Returns _TRY_NOW, _GONE or _WAIT (see above).

    WHY "gone" is its own answer: it means two different things. After the
    create failed because the lock EXISTED, it vanishing means its holder has
    just let go, and trying again at once is right. After the create failed
    with PermissionError, a missing lock means this account cannot create
    files in the lock's folder at all -- trying again at once, every time,
    would be an endless busy loop that never reached the wait limit. The waiter
    tells the two apart (see exclusive_file_lock).

    *server_offset* is the file server's clock minus this machine's, measured
    once by a waiter (see exclusive_file_lock); without it the server's clock
    is read now with a probe file -- BEFORE the breaker is taken, so nothing
    slow happens while it is held.

    The lock is removed only while holding ``<lock>.break``, after looking at
    it again; it is never moved aside (see "TAKING OVER AN ABANDONED LOCK" in
    the module docstring for why)."""
    try:
        mtime = lock.stat().st_mtime
        if server_offset is None:
            server_offset = _server_now(lock.parent) - time.time()
        age = time.time() + server_offset - mtime
    except FileNotFoundError:
        return _GONE
    except OSError:
        return _WAIT  # cannot judge its age now (e.g. a delete pending)
    if age < stale_seconds:
        return _WAIT
    # A first look without the breaker: a lock that has already changed hands
    # needs no breaker at all.
    if _read_or(lock, seen) != seen:
        return _TRY_NOW

    breaker = lock.with_name(lock.name + ".break")
    try:
        fd = os.open(str(breaker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Another waiter is breaking the lock right now (wait for it), or one
        # died doing so (clear its breaker; the takeover is tried on a later
        # pass, so a waiter never clears a breaker and takes it in one go).
        _clear_abandoned_breaker(breaker, stale_seconds, server_offset, label)
        return _WAIT
    except OSError:
        return _WAIT  # not allowed to create it: the wait limit decides
    try:
        os.close(fd)
        # Holding the breaker, no other waiter can remove or replace the lock:
        # only its own holder can change it. Look again, and delete it only
        # if it is still the lock judged abandoned above -- same holder line,
        # still untouched for stale_seconds. A fresh lock another waiter took
        # before we got the breaker has a different holder line (or, for the
        # instant before its line is written, a new modification time).
        try:
            mtime = os.stat(str(lock)).st_mtime
        except FileNotFoundError:
            return _GONE
        except OSError:
            return _WAIT
        age = time.time() + server_offset - mtime
        if age < stale_seconds:
            return _WAIT
        content = _read_or(lock, None)
        if content is None:
            # Unreadable now: gone, or open in a way that refuses reading.
            return _WAIT if lock.exists() else _GONE
        if content != seen:
            return _TRY_NOW
        try:
            os.unlink(str(lock))
        except FileNotFoundError:
            return _GONE
        except OSError:
            # Not allowed to delete it (or a program holds it open): wait.
            # Trying again at once would spin on the same refusal without
            # ever timing out.
            return _WAIT
        print("[%s] [!] Took over a lock left %.0f s ago by: %s"
              % (label, age, seen.strip() or "(an empty lock file)"))
        return _TRY_NOW
    finally:
        try:
            os.unlink(str(breaker))
        except OSError:
            pass


def _clear_abandoned_breaker(breaker: Path, stale_seconds: float,
                             server_offset: float, label: str) -> None:
    """Delete ``<lock>.break`` if a process died holding it.

    A live breaker is held for a few milliseconds, so one untouched for
    max(stale_seconds, BREAKER_STALE_SECONDS) by the server's clock is
    abandoned. It is looked at twice, and deleted only if it is the same file
    both times, so a breaker another waiter has just cleared and retaken is
    left alone."""
    limit = max(stale_seconds, BREAKER_STALE_SECONDS)
    try:
        first = os.stat(str(breaker))
        if time.time() + server_offset - first.st_mtime < limit:
            return
        again = os.stat(str(breaker))
        if (again.st_mtime, again.st_ctime) != (first.st_mtime, first.st_ctime):
            return
        os.unlink(str(breaker))
    except OSError:
        return
    print("[%s] [!] Removed %s, left by a process that stopped while taking "
          "over an abandoned lock" % (label, breaker.name))


def _unlink_retrying(path: Path, still_mine: Optional[Callable[[], bool]] = None) -> bool:
    """Delete *path*, retrying for RELEASE_RETRY_SECONDS while Windows refuses
    because another process has it open for a moment. *still_mine*, if given,
    is asked before every attempt; when it says False the file is left alone
    (someone else holds it now). Returns True when the file is gone or no
    longer ours, False when it could not be deleted in time."""
    deadline = time.monotonic() + RELEASE_RETRY_SECONDS
    while True:
        if still_mine is not None and not still_mine():
            return True
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        except OSError:
            return False


def release_lock(lock: Path, mine: str, label: str) -> bool:
    """Delete *lock* if it still holds this holder's line *mine*.

    WHY it retries: see "RELEASING ON WINDOWS" in the module docstring. The
    lock is re-read before every attempt, so a lock that was taken over (and
    perhaps taken afresh by someone else) is never deleted. Returns True when
    the lock is gone or no longer ours."""
    def still_mine() -> bool:
        content = _read_or(lock, None)
        if content is None:  # gone, or open in a way that refuses reading too
            return lock.exists()
        return content == mine

    if _unlink_retrying(lock, still_mine):
        return True
    print("[%s] [!] Could not remove the lock %s within %d s (another program has "
          "it open). Writers will wait until it ages out; if nothing is writing, "
          "delete it." % (label, lock, RELEASE_RETRY_SECONDS))
    return False


class LockHandle:
    """What a holder gets inside ``with exclusive_file_lock(...)``."""

    def __init__(self, path: Path, mine: str = ""):
        self.path = path
        self._mine = mine.encode("utf-8")
        self._touching = threading.Lock()

    def touch(self) -> bool:
        """Mark the lock as still in use, so a long piece of work is not
        mistaken for a crashed holder.

        The holder line is written again through an open handle, so the FILE
        SERVER stamps the new modification time (see "KEEPING A LIVE LOCK
        FRESH" in the module docstring for why never os.utime). Nothing is
        written unless the lock still holds this holder's line. Returns True
        when the lock was refreshed."""
        with self._touching:  # the heartbeat thread and the holder never write at once
            try:
                with open(self.path, "r+b") as f:
                    if f.read(len(self._mine) + 1) != self._mine:
                        return False  # no longer ours: never write into it
                    f.seek(0)
                    f.write(self._mine)
                    f.truncate()
                return True
            except OSError:
                return False


def _keep_fresh(handle: LockHandle, every: float, stop: threading.Event) -> None:
    """Refresh *handle*'s lock every *every* seconds until *stop* is set."""
    while not stop.wait(every):
        handle.touch()


@contextmanager
def exclusive_file_lock(
    target: Union[str, Path],
    label: str,
    stale_seconds: float = 120,
    wait_seconds: float = 300,
    lock_path: Union[str, Path, None] = None,
) -> Iterator[LockHandle]:
    """Hold ``<target>.lock`` (or *lock_path*) for the duration of the block.

    Args:
        target: The file being protected; the lock sits beside it.
        label: Short name used in printed messages (ASCII).
        stale_seconds: A lock untouched for this long (server clock) is
            treated as left behind by a crashed holder and taken over.
        wait_seconds: How long to wait for a live holder before raising
            TimeoutError.
        lock_path: Where the lock file goes instead of beside *target* (see
            "WHERE THE LOCK LIVES"). Its folder is created when missing.
    """
    target = Path(target)
    if lock_path is None:
        lock = target.with_name(target.name + ".lock")
    else:
        lock = Path(lock_path)
        lock.parent.mkdir(parents=True, exist_ok=True)
    mine = "%s pid %d at %s token %s\n" % (
        socket.gethostname(), os.getpid(),
        datetime.now().isoformat(timespec="seconds"), uuid.uuid4().hex)
    deadline = time.monotonic() + wait_seconds
    told = False
    server_offset: Optional[float] = None
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(mine)
            break
        except (FileExistsError, PermissionError) as create_error:
            seen = _read_or(lock, "")
            # The server's clock is read ONCE per wait (one probe file), not
            # on every pass: a probe written five times a second by every
            # waiter would be needless traffic on the share.
            if server_offset is None:
                try:
                    server_offset = _server_now(lock.parent) - time.time()
                except OSError:
                    server_offset = None
            # An empty or unreadable lock is judged by its age too (see the
            # module docstring); a holder that has just created it is far
            # younger than stale_seconds, so it is never taken over.
            verdict = _take_over_stale_lock(lock, seen, stale_seconds, label, server_offset)
            # Every way round this loop that does not sleep is bounded by the
            # deadline, so no refusal can turn it into an endless busy loop.
            cannot_create = verdict == _GONE and isinstance(create_error, PermissionError)
            if verdict != _WAIT and not cannot_create and time.monotonic() <= deadline:
                continue
            if time.monotonic() > deadline:
                if cannot_create:
                    raise TimeoutError(
                        "Could not create the lock %s for %s within %d s: permission "
                        "denied (%s). Check that this account may create and delete "
                        "files in %s." % (lock, target, wait_seconds, create_error,
                                         lock.parent))
                raise TimeoutError(
                    "%s is locked by %s for longer than %d s. If nothing is writing "
                    "to it, delete %s." % (
                        target, seen.strip() or "another process", wait_seconds, lock))
            if not told:
                if cannot_create:
                    print("[%s] [!] Cannot create the lock %s (permission denied); "
                          "retrying for %d s" % (label, lock, wait_seconds))
                else:
                    print("[%s] Waiting for the lock on %s (held by %s)"
                          % (label, target.name, seen.strip() or "?"))
                told = True
            time.sleep(0.2)
    handle = LockHandle(lock, mine)
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_keep_fresh, args=(handle, max(stale_seconds / 4.0, 0.05), stop),
        name="lock-heartbeat %s" % lock.name, daemon=True)
    heartbeat.start()
    try:
        yield handle
    finally:
        # Stop refreshing BEFORE the release, so a refresh can never hold the
        # file open while the release deletes it.
        stop.set()
        heartbeat.join()
        release_lock(lock, mine, label)
