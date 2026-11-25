"""A person's approval of ONE registration -- written, read and retired here only.

WHY APPROVAL EXISTS
Registration fits the atlas to a brain. A bad fit still produces region counts
-- every number looks plausible -- but the cells are assigned to the wrong
regions. So cell detection on the registered images (step 4) waits until a
person has looked at the registration's QC pictures and said it is good. The
approval is the file ``3_Registered_Atlas/.registration_approved``.

WHY THE APPROVAL NAMES ITS REGISTRATION
A marker that held only a date (or nothing) would be an approval of
"whatever registration is in this folder", not of the one the person looked at:

* Re-registering moves the old registration into ``_archive/<time>/``; a
  marker left behind (an archive that skips files starting with a dot) would
  show the NEW, never-reviewed registration as approved, and step 4 would run
  on it.
* A batch that wrote the marker itself straight after registering would send
  a registration nobody had looked at on to detection, classification and
  region counts.
* An Approve button pressed before any registration exists would be inherited
  by the registration made later.

So the marker records the registration it approves -- the size and
modification time of the registration's own output files -- and every reader
compares that record with what is in the folder. A marker for a different
registration, or a plain-text marker that does not say which registration it
was for, counts as NOT approved. Approving needs a person (the util script, the
Approve buttons); no automated run writes one.

WHEN A REGISTRATION COUNTS AS A REGISTRATION AT ALL
brainreg writes ``brainreg.json`` FIRST, before it fits anything, so that file
alone proves only that a registration STARTED. A run that failed or was killed
leaves it behind; a reader that trusted it would take the brain as registered:
step 3 would skip it as done, the overnight batch would call it a failure every
night, and a person could approve it. A registration is complete only when every one of its output
files (FINGERPRINT_FILES) is there -- registration_complete(), the one test
every reader uses. An incomplete registration cannot be approved.

A REGISTRATION WHOSE IMAGES CHANGED IS OUT OF DATE
The atlas is fitted to one set of images (``registered_from.json`` names the
folder). A re-crop rewrites those images; the registration then describes
images that are no longer there; it must not stay "registered" and "approved"
for step 4 to run on. A registration older than the newest image it was fitted
to, or whose recorded images are gone, is OUT_OF_DATE: no reader counts it as
registered, nobody can approve it, step 4's approval gate is closed, and the
brain is at "register" again (step 3's --batch registers it).

A REGISTRATION THAT IS RUNNING IS NOT "INCOMPLETE"
Because brainreg.json comes first, a registration that is running right now
looks exactly like one that stopped half-way. Taking it for the latter would
make step 3's --batch archive the running brainreg's folder and start a
second brainreg into it. So step 3 writes ``.registration_running`` (which
computer, which process, when) before brainreg starts and removes it when it
ends, however it ends. While that marker is LIVE every reader says RUNNING and
step 3 refuses to archive or register the brain. A marker whose process has
gone (the computer restarted, the window was closed) is not live: the
registration is INCOMPLETE and is registered again. See running_registration()
for how "live" is decided.

The marker names TWO processes: step 3 itself and, once it has started it,
brainreg. Killing step 3 (a batch's time limit, a closed dashboard) does not
stop brainreg, which goes on writing into the folder. Judged by step 3's
process alone, the marker would look dead, and a retry would archive the folder
under the still-running brainreg and start a second one into it. The marker
stays live while EITHER process is alive.

Step 3's own check after brainreg passes its claim (``own=``): the marker is
its own, so to that check the brain is not "being registered by someone
else". Without that, every finished registration would read as RUNNING to the
run that made it, fail its verification and be archived as failed.

A REJECTED REGISTRATION STAYS REJECTED
Rejecting by only moving an existing approval away (and doing nothing for a
registration never approved) would let the next reader show the registration
as merely "waiting for approval", and anyone could approve what a person had
just turned down. So reject() writes ``.registration_rejected``
naming the registration (who, when, its fingerprint). Every reader then shows
it as REJECTED, the brain is back at the register step, and approve() refuses
it -- except the approval script, which asks the person whether they mean to
overrule the rejection. Registering again archives the rejection with the
registration it was for.

NOTHING UNATTENDED REGISTERS A REJECTED BRAIN AGAIN. The batches (step 3
--batch, overnight_batch, batch_process) skip it and report who rejected it,
when, and what to change: registering it again on the same images and the
same settings would only take hours to produce the same fit and ask the same
question again. A person asks for it (the dashboard's "Register again",
step 3 --batch --brain <brain> --reprocess, or step 3's interactive menu).
A re-crop made after the rejection makes the registration OUT_OF_DATE, which
the batches do register again: the change a rejection asks for has been made.

What happens to approvals:
  * approve()                    -- a person approved what is there now; a
                                    window passes ``expected_id`` (the
                                    registration it showed) and approve()
                                    refuses when the folder now holds another
                                    one (RegistrationChanged)
  * reject()                     -- a person turned the registration down; the
                                    rejection is recorded and any approval
                                    goes into ``_archive/``; ``expected_id``
                                    as for approve()

WHY EVERY WINDOW PASSES THE REGISTRATION IT SHOWED
A person approves or rejects what they LOOKED AT: the status line, the QC
picture. Between looking and clicking, a batch can finish registering the
brain again. Without ``expected_id`` the click would sign (or turn down) the
new registration nobody had seen. Every window that asks (Setup & Tuning, the
pipeline dashboard, RUN_PIPELINE, the approval script) records
fingerprint_id() of the registration at the moment it shows it, and passes it
here; a registration that changed since is refused with RegistrationChanged
and nothing is written. Use current_registration_id() for that record.
  * retire_for_new_registration() -- registration is about to run again: the
                                     approval and any rejection move into
                                     ``_archive/`` with the registration they
                                     were for
  * withdraw()                   -- take an approval back without rejecting

COUNTS EXIST ONLY FOR AN APPROVED REGISTRATION
Region counts are cells placed in the regions of one registration's atlas
fit. Step 6 counts a brain only while approval_state() says APPROVED, and the
brain's counts leave production -- its rows in region_counts.csv /
elife_counts.csv and its 3D_Region_Counts entry in the analysis registry
(withdraw_counts_from_production) -- when reject() turns the registration
down, when withdraw() takes its approval back, when a re-crop makes it out of
date (crop_subprocess.swap_in) and when step 3 archives it to register again.
"""

import getpass
import hashlib
import json
import os
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

APPROVAL_MARKER = ".registration_approved"
REJECTION_MARKER = ".registration_rejected"
RUNNING_MARKER = ".registration_running"

# A running-marker written on ANOTHER computer cannot be checked against that
# computer's processes, so it is taken as live until it is this old. Longer
# than any registration takes; a registration that is still really running
# after this long is better looked at by a person anyway.
RUNNING_MAX_HOURS_OTHER_HOST = 24

# The registration's own output files. Their sizes and modification times
# identify one registration: a new run rewrites all of them. QC pictures and
# registered_from.json are deliberately NOT included -- redrawing a QC picture
# is not a new registration.
FINGERPRINT_FILES = ("brainreg.json", "registered_atlas.tiff", "boundaries.tiff",
                     "deformation_field_0.tiff")

# What approval_state() can say.
NOT_REGISTERED = "not_registered"      # no brainreg.json: nothing to approve
RUNNING = "running"                    # step 3 is registering it right now
INCOMPLETE = "incomplete"              # started, but an output file is missing
REJECTED = "rejected"                  # a person rejected THIS registration
OUT_OF_DATE = "out_of_date"            # its images were made again, or are gone
UNAPPROVED = "unapproved"              # registered, no marker
APPROVED = "approved"                  # marker names this registration
OTHER_REGISTRATION = "other_registration"  # marker names a different one
UNRECORDED = "unrecorded"              # old-style marker: does not say which

# The states in which the brain is at the register step: there is nothing a
# person could approve. (RUNNING is among them: the registration is not there
# yet. Step 3 refuses to start a second one; see running_registration.)
NEEDS_REGISTRATION = (NOT_REGISTERED, RUNNING, INCOMPLETE, REJECTED, OUT_OF_DATE)

# What a person does about a rejected registration. Said wherever a rejected
# brain is skipped, so nobody has to look it up.
REJECTED_ADVICE = ("change what made it bad, then register it again: a re-crop "
                   "(after which the batches register it again by themselves), or "
                   "the orientation / voxel sizes in the crop's metadata.json and "
                   "then 'Register again' in the pipeline dashboard or "
                   "3_register_to_atlas.py --batch --brain <brain> --reprocess")

_STATE_TEXT = {
    NOT_REGISTERED: "not registered yet",
    INCOMPLETE: ("the registration did not finish (it failed or was stopped): "
                 "register the brain again"),
    UNAPPROVED: "registered, waiting for a person to approve it",
    APPROVED: "approved",
    OTHER_REGISTRATION: ("the approval on file is for an EARLIER registration of this "
                         "brain; this one has not been reviewed"),
    UNRECORDED: ("the approval on file does not say which registration it was for "
                 "(written before approvals recorded that, or by an automated run); "
                 "review the current registration and approve it again"),
}


def _reg(reg_folder) -> Path:
    return Path(reg_folder)


def missing_registration_files(reg_folder) -> list:
    """The registration output files (FINGERPRINT_FILES) not in this folder."""
    reg = _reg(reg_folder)
    return [n for n in FINGERPRINT_FILES if not (reg / n).is_file()]


def registration_complete(reg_folder) -> bool:
    """True only when a registration FINISHED in this folder: every output
    file is there. brainreg.json alone is written before brainreg fits
    anything, so it is no proof (see the module docstring)."""
    return not missing_registration_files(reg_folder)


def registration_fingerprint(reg_folder) -> Optional[Dict[str, Dict[str, int]]]:
    """{file name: {size, mtime}} for the registration in this folder, or None
    when there is no registration (no brainreg.json). Modification times are
    whole seconds, so the same file reads the same from every machine."""
    reg = _reg(reg_folder)
    if not (reg / "brainreg.json").is_file():
        return None
    fp = {}
    for name in FINGERPRINT_FILES:
        f = reg / name
        if f.is_file():
            st = f.stat()
            fp[name] = {"size": int(st.st_size), "mtime": int(st.st_mtime)}
    return fp


def fingerprint_id(fp: Optional[dict]) -> str:
    """A short id of a fingerprint, for display and comparison."""
    if not fp:
        return ""
    return hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:12]


def registration_time(reg_folder) -> Optional[float]:
    """When the registration in this folder was written (newest of its output
    files), or None when there is none. Outputs older than this were made on an
    earlier registration."""
    reg = _reg(reg_folder)
    if not (reg / "brainreg.json").is_file():
        return None
    times = [(reg / n).stat().st_mtime for n in FINGERPRINT_FILES if (reg / n).is_file()]
    return max(times) if times else None


def read_marker(reg_folder) -> Optional[dict]:
    """The approval marker's content, or None when there is no marker. An
    plain-text marker (a date, or empty) comes back as {'legacy': text}."""
    marker = _reg(reg_folder) / APPROVAL_MARKER
    if not marker.is_file():
        return None
    try:
        text = marker.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"legacy": "", "error": str(e)}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "registration" in data:
            return data
    except ValueError:
        pass
    return {"legacy": text.strip()}


_UNSET = object()


def registered_images_time(pipeline_dir) -> Optional[float]:
    """When the images the atlas is fitted to were last written: the newest
    ``ch*/*.tif`` of the folder config.registered_input names, or None when
    there are none. One directory listing per channel; the times come with
    the listing. Raises config.RegisteredImagesMissing when the registration
    records a folder whose images are gone."""
    import os
    from mousebrain.config import registered_input
    _source, folder, _meta = registered_input(pipeline_dir)
    if folder is None or not Path(folder).is_dir():
        return None
    newest = None
    for ch in Path(folder).glob("ch*"):
        if not ch.is_dir():
            continue
        with os.scandir(ch) as it:
            for entry in it:
                if entry.name.lower().endswith((".tif", ".tiff")):
                    t = entry.stat().st_mtime
                    newest = t if newest is None or t > newest else newest
    return newest


# File times are compared with this much slack: copies and network shares
# round modification times.
_SLACK_SECONDS = 2.0


def images_problem(reg_folder, images_time=_UNSET) -> str:
    """Why the registration in this folder no longer fits its images, or "".

    "" also when nothing can be said (no registration, no images found).
    ``images_time``: the caller's own registered_images_time() result, so a
    reader that already listed the images does not list them twice; left
    out, the images are listed here."""
    from mousebrain.config import RegisteredImagesMissing
    reg = _reg(reg_folder)
    reg_t = registration_time(reg)
    if reg_t is None:
        return ""
    if images_time is _UNSET:
        try:
            images_time = registered_images_time(reg.parent)
        except RegisteredImagesMissing as e:
            return "the images it was fitted to are gone: %s" % e
    if images_time is not None and reg_t + _SLACK_SECONDS < images_time:
        return ("its images were made again after it was registered (a re-crop or "
                "re-extraction), so the atlas fit describes the earlier images")
    return ""


def read_rejection(reg_folder) -> Optional[dict]:
    """The rejection marker's content, or None when there is none."""
    marker = _reg(reg_folder) / REJECTION_MARKER
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class RegistrationRunning(RuntimeError):
    """Step 3 refused a brain: another run is registering it right now."""


def _pid_alive(pid, started_epoch: Optional[float]) -> bool:
    """Whether process ``pid`` on THIS computer is still the one that wrote the
    marker. A process id can be reused after the process ends, so on Windows a
    process created after the marker was written is not it.

    WHY NOT os.kill(pid, 0) ON WINDOWS: there os.kill with any signal other
    than the two console events TERMINATES the process."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # Access denied means the process exists (another user's); any
            # other error (invalid parameter) means there is no such process.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if k32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
                return False  # it has ended (259 = STILL_ACTIVE)
            if started_epoch is not None:
                times = [wintypes.FILETIME() for _ in range(4)]
                if k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                    ft = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                    created = (ft - 116444736000000000) / 1e7
                    if created > started_epoch + _SLACK_SECONDS:
                        return False  # the id now belongs to a newer process
            return True
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def running_registration(reg_folder, own: Optional[dict] = None) -> Optional[dict]:
    """The running-marker in this folder, or None when there is none.

    ``own``: the caller's claim (claim_registration()'s result). A marker
    with the same token is the caller's own run, and is reported as None:
    to the run that wrote it, the folder is not being registered by anyone
    else.

    The returned dict is the marker's content plus ``live`` (True while the
    registration that wrote it can still be running) and ``text`` (what to
    tell a person). LIVE means:
      * written on THIS computer: its process is still running (and is the
        same process, not a later one given the same id);
      * written on ANOTHER computer: written less than
        RUNNING_MAX_HOURS_OTHER_HOST hours ago (that computer's processes
        cannot be seen from here);
      * unreadable: the file is less than that many hours old.
    On this computer the marker is live while step 3's process OR the
    brainreg it started (``child_pid``, see record_registration_child) is
    alive: brainreg outlives a step 3 that was killed.
    A marker that is not live was left by a run that ended without removing
    it: the computer restarted, or the window was closed."""
    marker = _reg(reg_folder) / RUNNING_MARKER
    try:
        mtime = marker.stat().st_mtime
    except OSError:
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            info = {}
    except (OSError, ValueError):
        info = {}
    if own is not None and own.get("token") and info.get("token") == own.get("token"):
        return None
    host = str(info.get("host") or "")

    def _epoch(key):
        value = info.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    started = _epoch("started_epoch")
    age_h = (datetime.now().timestamp() - (started if started is not None else mtime)) / 3600.0
    if host and host.lower() == socket.gethostname().lower():
        live = (_pid_alive(info.get("pid"), started)
                 or (info.get("child_pid") is not None
                     and _pid_alive(info.get("child_pid"), _epoch("child_started_epoch"))))
    else:
        live = age_h < RUNNING_MAX_HOURS_OTHER_HOST
    where = "on %s, process %s%s, started %s" % (
        host or "an unknown computer", info.get("pid", "?"),
        (" (brainreg: process %s)" % info["child_pid"]) if info.get("child_pid") else "",
        info.get("started_at", "?"))
    if live:
        text = ("being registered right now (%s); wait for it to finish. If you are "
                "sure that run is no longer going, delete %s in the registration "
                "folder" % (where, RUNNING_MARKER))
    else:
        text = ("the registration did not finish: the run %s is no longer running "
                "-- register the brain again" % where)
    info.update(live=live, text=text)
    return info


def claim_registration(reg_folder) -> dict:
    """Step 3 is about to register this brain: write the running-marker.

    Created exclusively, so of two runs starting together only one gets it.
    A marker that is no longer live (see running_registration) is replaced.
    Raises RegistrationRunning when a live run holds it. Returns the marker's
    content; hand it to release_registration() when the run ends."""
    reg = _reg(reg_folder)
    reg.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    record = {"host": socket.gethostname(), "pid": os.getpid(),
              "started_at": now.isoformat(timespec="seconds"),
              "started_epoch": now.timestamp(),
              "token": "%s-%d-%f" % (socket.gethostname(), os.getpid(), now.timestamp())}
    path = reg / RUNNING_MARKER
    text = json.dumps(record, indent=1)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held = running_registration(reg)
        if held is not None and held["live"]:
            raise RegistrationRunning(held["text"])
        _write_atomic(path, text)  # a dead run's marker: take it over
        mine = running_registration(reg)
        if mine is None or mine.get("token") != record["token"]:
            raise RegistrationRunning(
                (mine or {}).get("text") or "another run took this brain at the same moment")
        return record
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return record


def record_registration_child(reg_folder, record: dict, child_pid: int) -> None:
    """Step 3 has started brainreg (process ``child_pid``): add it to the
    run's running-marker, so the marker stays live while brainreg runs even
    when step 3 itself is killed (see running_registration).

    ``record`` is claim_registration()'s result; it is updated in place. Only
    the caller's OWN marker is rewritten (same token), never another run's.
    Call it right after the process is created: the time written here is the
    one a reused process id is checked against."""
    path = _reg(reg_folder) / RUNNING_MARKER
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(current, dict) or current.get("token") != record.get("token"):
        return
    record["child_pid"] = int(child_pid)
    record["child_started_epoch"] = datetime.now().timestamp()
    _write_atomic(path, json.dumps(record, indent=1))


def release_registration(reg_folder, record: dict) -> None:
    """The run that wrote ``record`` has ended: remove its running-marker --
    only its OWN (a marker another run wrote since is left alone)."""
    held = running_registration(reg_folder)
    if held is not None and held.get("token") == record.get("token"):
        try:
            (_reg(reg_folder) / RUNNING_MARKER).unlink()
        except OSError:
            pass


def approval_state(reg_folder, images_time=_UNSET, own: Optional[dict] = None) -> Tuple[str, str]:
    """(state, explanation) for the registration in this folder. Only
    APPROVED means a person approved the registration that is there now; the
    states in NEEDS_REGISTRATION mean the brain has to be registered again.

    ``images_time``: see images_problem(). ``own``: the caller's claim,
    see running_registration() -- only step 3 checking its own run passes it."""
    # Running first: a registration being made right now has a brainreg.json
    # and no other output yet, which is exactly what a stopped one looks like.
    running = running_registration(reg_folder, own=own)
    if running is not None:
        if running["live"]:
            return RUNNING, running["text"]
        # Its run ended without removing the marker: whatever is in the folder
        # was not checked by step 3 after brainreg, even if every file is there.
        return INCOMPLETE, running["text"]
    current = registration_fingerprint(reg_folder)
    if current is None:
        return NOT_REGISTERED, _STATE_TEXT[NOT_REGISTERED]
    missing = missing_registration_files(reg_folder)
    if missing:
        return INCOMPLETE, "%s (missing: %s)" % (_STATE_TEXT[INCOMPLETE], ", ".join(missing))
    # Out of date BEFORE rejected: a re-crop made after the rejection is the
    # change the rejection asked for, so the brain is registrable again (by
    # the batches too) instead of still "rejected".
    problem = images_problem(reg_folder, images_time)
    if problem:
        return OUT_OF_DATE, "out of date: %s -- register the brain again" % problem
    rejection = read_rejection(reg_folder)
    if rejection is not None and rejection.get("registration") in (None, current):
        # A rejection that cannot be read, or does not say which registration
        # it was for, is still a person's "no" about what is in this folder.
        return REJECTED, "rejected by %s at %s%s -- %s" % (
            rejection.get("rejected_by") or "someone", rejection.get("rejected_at") or "?",
            (" (" + rejection["note"] + ")") if rejection.get("note") else "",
            REJECTED_ADVICE)
    marker = read_marker(reg_folder)
    if marker is None:
        return UNAPPROVED, _STATE_TEXT[UNAPPROVED]
    if "legacy" in marker:
        return UNRECORDED, _STATE_TEXT[UNRECORDED]
    if marker.get("registration") != current:
        return OTHER_REGISTRATION, _STATE_TEXT[OTHER_REGISTRATION]
    who = marker.get("approved_by") or "someone"
    when = marker.get("approved_at") or "?"
    return APPROVED, "approved by %s at %s" % (who, when)


def is_approved(reg_folder) -> bool:
    """True only when a person approved the registration that is there now."""
    return approval_state(reg_folder)[0] == APPROVED


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp-%d" % os.getpid())
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _who(name: Optional[str]) -> str:
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


class RegistrationChanged(ValueError):
    """approve() or reject() refused: the folder holds another registration
    than the one the person was shown (registered again since they looked)."""


def current_registration_id(reg_folder) -> str:
    """fingerprint_id() of the registration in this folder now ("" when there
    is none). A window records this when it SHOWS a registration (status line,
    QC picture) -- before it reads the status it displays, so that a
    registration finished in between is refused rather than signed -- and
    passes it to approve()/reject() as ``expected_id``."""
    return fingerprint_id(registration_fingerprint(reg_folder))


def _refuse_if_changed(reg: Path, current: dict, expected_id: Optional[str], what: str) -> None:
    if expected_id is not None and fingerprint_id(current) != expected_id:
        raise RegistrationChanged(
            "the registration in %s is not the one you were shown: it was "
            "registered again since. Look at the new registration, then "
            "%s it." % (reg, what))


def approve(reg_folder, approved_by: Optional[str] = None, note: str = "",
            allow_rejected: bool = False, expected_id: Optional[str] = None) -> dict:
    """Record that a person approved the registration in this folder.

    ``expected_id``: the fingerprint_id() of the registration the person was
    shown ("" when they were shown none). When given and the folder now holds
    another registration, RegistrationChanged is raised and nothing is
    written: the click was about what they saw, and a registration finished
    by a script after they looked has not been seen by anyone.

    Call this only from something a person did (a button, the approval
    script's yes/no question). Raises ValueError when there is nothing a
    person could approve: no registration (an approval written before the
    registration exists would be inherited by whatever lands there later), or
    a registration that did not finish, or one whose images changed since
    (OUT_OF_DATE). A REJECTED registration is refused
    too, unless ``allow_rejected`` -- passed only after a person was asked
    whether they mean to overrule the rejection and said yes (the approval
    script's question, the pipeline dashboard's overrule question); the
    rejection then goes into ``_archive/``.
    """
    reg = _reg(reg_folder)
    current = registration_fingerprint(reg)
    if current is None:
        raise ValueError("no registration to approve in %s (brainreg.json not found)" % reg)
    _refuse_if_changed(reg, current, expected_id, "approve")
    state, text = approval_state(reg)
    if state in (RUNNING, INCOMPLETE, OUT_OF_DATE):
        raise ValueError("cannot approve the registration in %s: %s" % (reg, text))
    if state == REJECTED:
        if not allow_rejected:
            raise ValueError("this registration was %s" % text)
        _move_marker(reg, "rejection_overruled", REJECTION_MARKER)
    record = {
        "approved_at": datetime.now().isoformat(timespec="seconds"),
        "approved_by": _who(approved_by),
        "host": socket.gethostname(),
        "note": note,
        "registration": current,
        "registration_id": fingerprint_id(current),
    }
    _write_atomic(reg / APPROVAL_MARKER, json.dumps(record, indent=1))
    return record


def reject(reg_folder, rejected_by: Optional[str] = None, note: str = "",
           expected_id: Optional[str] = None) -> dict:
    """Record that a person looked at the registration in this folder and
    turned it down. Any approval goes into ``_archive/``; every reader shows
    the brain back at the register step until it is registered again (step 3
    then archives this rejection with the registration it was for). Raises
    ValueError when there is no registration in the folder.

    ``expected_id``: as for approve(). A "no" is about the registration the
    person looked at; a registration finished after they looked has not been
    seen, so turning it down is refused (RegistrationChanged) and nothing is
    written -- a rejection blocks the batches from registering the brain
    again, so a wrong one costs as much as a wrong approval."""
    reg = _reg(reg_folder)
    current = registration_fingerprint(reg)
    if current is None:
        raise ValueError("no registration to reject in %s (brainreg.json not found)" % reg)
    _refuse_if_changed(reg, current, expected_id, "decide about")
    running = running_registration(reg)
    if running is not None and running["live"]:
        raise ValueError("nothing to reject yet in %s: %s" % (reg, running["text"]))
    _move_marker(reg, "withdrawn_rejected")
    record = {
        "rejected_at": datetime.now().isoformat(timespec="seconds"),
        "rejected_by": _who(rejected_by),
        "host": socket.gethostname(),
        "note": note,
        "registration": current,
        "registration_id": fingerprint_id(current),
    }
    _write_atomic(reg / REJECTION_MARKER, json.dumps(record, indent=1))
    # The rejection is recorded first: it is the person's decision and must
    # stand whatever happens next. Then the brain's counts -- regions of the
    # atlas fit just turned down -- leave the production count tables and the
    # analysis registry (withdraw_counts_from_production says why).
    withdraw_counts_from_production(reg, "registration rejected by %s" % record["rejected_by"])
    return record


def withdraw_counts_from_production(reg_folder, reason: str) -> None:
    """Take the brain's region counts out of production: its rows in
    region_counts.csv / elife_counts.csv and its ``3D_Region_Counts`` entry
    in the analysis registry (mousebrain.region_counts_io.withdraw_counts).

    WHY: counts are cells placed in the regions of ONE registration's atlas
    fit. When that registration is rejected, its approval is taken back, or
    a re-crop makes it out of date, the counts describe a fit that no longer
    stands for the brain, and step 6 counts only an approved registration.
    Before, rejecting a registration withdrew nothing: the brain's counts
    stayed in the tables and the registry, and an integrator imported them
    as current. The brain gets fresh counts by being registered (if needed),
    approved, and counted again (``6_count_regions.py --all`` picks it up).

    Never raises: the event that called it (a rejection a person made, a
    crop that is already in place) is done and must not be undone or hidden
    by a problem here; that problem is printed with what to do."""
    try:
        from mousebrain.region_counts_io import withdraw_counts_or_warn
        withdraw_counts_or_warn(_reg(reg_folder).parent.name, reason)
    except Exception as e:  # the import itself (a broken installation)
        print("    [!] The counts of %s could NOT be withdrawn (%s): %s. Count the brain "
              "again (step 6) once its registration is approved."
              % (_reg(reg_folder).parent.name, reason, e))


def _move_marker(reg: Path, label: str, name: str = APPROVAL_MARKER) -> Optional[Path]:
    marker = reg / name
    if not marker.exists():
        return None
    dest_dir = reg / "_archive" / ("%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), label))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.move(str(marker), str(dest))
    return dest


def retire_for_new_registration(reg_folder) -> Optional[Path]:
    """Registration is about to run (again): move any approval -- and any
    rejection -- out of the way, into ``_archive/``, so the new registration
    starts unapproved and unrejected. Returns where the approval went, or None
    if there was none."""
    reg = _reg(reg_folder)
    _move_marker(reg, "rejection_of_previous_registration", REJECTION_MARKER)
    return _move_marker(reg, "approval_of_previous_registration")


def withdraw(reg_folder, reason: str = "rejected") -> Optional[Path]:
    """Take an approval back: it goes into ``_archive/`` -- kept, not
    deleted -- and the brain is unapproved. To turn the registration DOWN use
    reject(), which also records that it was turned down. The brain's counts
    leave production too (withdraw_counts_from_production): step 6 counts
    only an approved registration."""
    moved = _move_marker(_reg(reg_folder), "withdrawn_" + reason)
    if moved is not None:
        withdraw_counts_from_production(reg_folder, "approval withdrawn (%s)" % reason)
    return moved
