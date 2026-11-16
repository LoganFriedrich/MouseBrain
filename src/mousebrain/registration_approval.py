"""A person's approval of ONE registration -- written, read and retired here only.

WHY APPROVAL EXISTS
Registration fits the atlas to a brain. A bad fit still produces region counts
-- every number looks plausible -- but the cells are assigned to the wrong
regions. So cell detection on the registered images (step 4) waits until a
person has looked at the registration's QC pictures and said it is good. The
approval is the file ``3_Registered_Atlas/.registration_approved``.

WHY THE APPROVAL NAMES ITS REGISTRATION
The marker used to hold only a date (or nothing). That made it an approval of
"whatever registration is in this folder", not of the one the person looked at:

* Re-registering moved the old registration into ``_archive/<time>/`` but left
  the marker behind (the archive skipped every file starting with a dot), so
  the NEW, never-reviewed registration was shown as approved and step 4 ran on
  it.
* The overnight batch wrote the marker itself straight after registering, so
  a registration nobody had looked at went on to detection, classification
  and region counts.
* The tuning widget's Approve button could be pressed before any registration
  existed; the registration made later inherited the approval.

Now the marker records the registration it approves -- the size and
modification time of the registration's own output files -- and every reader
compares that record with what is in the folder. A marker for a different
registration, or an old marker that does not say which registration it was
for, counts as NOT approved. Approving needs a person (the util script, the
Approve buttons); no automated run writes one.

WHEN A REGISTRATION COUNTS AS A REGISTRATION AT ALL
brainreg writes ``brainreg.json`` FIRST, before it fits anything, so that file
alone proves only that a registration STARTED. A run that failed or was killed
left it behind and every reader took the brain as registered: step 3 skipped
it as done, the overnight batch called it a failure every night, and a person
could approve it. A registration is complete only when every one of its output
files (FINGERPRINT_FILES) is there -- registration_complete(), the one test
every reader uses. An incomplete registration cannot be approved.

What happens to approvals:
  * approve()                    -- a person approved what is there now
  * retire_for_new_registration() -- registration is about to run again: the
                                     marker moves into ``_archive/`` with the
                                     registration it was for
  * withdraw()                   -- a person rejected the registration
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

# The registration's own output files. Their sizes and modification times
# identify one registration: a new run rewrites all of them. QC pictures and
# registered_from.json are deliberately NOT included -- redrawing a QC picture
# is not a new registration.
FINGERPRINT_FILES = ("brainreg.json", "registered_atlas.tiff", "boundaries.tiff",
                     "deformation_field_0.tiff")

# What approval_state() can say.
NOT_REGISTERED = "not_registered"      # no brainreg.json: nothing to approve
INCOMPLETE = "incomplete"              # started, but an output file is missing
UNAPPROVED = "unapproved"              # registered, no marker
APPROVED = "approved"                  # marker names this registration
OTHER_REGISTRATION = "other_registration"  # marker names a different one
UNRECORDED = "unrecorded"              # old-style marker: does not say which

# The states in which the brain needs registering (again) before anything
# else: there is nothing a person could approve.
NEEDS_REGISTRATION = (NOT_REGISTERED, INCOMPLETE)

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
    old-style marker (plain text or empty) comes back as {'legacy': text}."""
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


def approval_state(reg_folder) -> Tuple[str, str]:
    """(state, explanation) for the registration in this folder. Only
    APPROVED means a person approved the registration that is there now; the
    states in NEEDS_REGISTRATION mean the brain has to be registered again."""
    current = registration_fingerprint(reg_folder)
    if current is None:
        return NOT_REGISTERED, _STATE_TEXT[NOT_REGISTERED]
    missing = missing_registration_files(reg_folder)
    if missing:
        return INCOMPLETE, "%s (missing: %s)" % (_STATE_TEXT[INCOMPLETE], ", ".join(missing))
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


def approve(reg_folder, approved_by: Optional[str] = None, note: str = "") -> dict:
    """Record that a person approved the registration in this folder.

    Call this only from something a person did (a button, the approval
    script's yes/no question). Raises ValueError when there is nothing a
    person could approve: no registration (an approval written before the
    registration exists would be inherited by whatever lands there later), or
    a registration that did not finish.
    """
    reg = _reg(reg_folder)
    current = registration_fingerprint(reg)
    if current is None:
        raise ValueError("no registration to approve in %s (brainreg.json not found)" % reg)
    state, text = approval_state(reg)
    if state == INCOMPLETE:
        raise ValueError("cannot approve the registration in %s: %s" % (reg, text))
    if approved_by is None:
        try:
            approved_by = getpass.getuser()
        except Exception:
            approved_by = "unknown"
    record = {
        "approved_at": datetime.now().isoformat(timespec="seconds"),
        "approved_by": approved_by,
        "host": socket.gethostname(),
        "note": note,
        "registration": current,
        "registration_id": fingerprint_id(current),
    }
    _write_atomic(reg / APPROVAL_MARKER, json.dumps(record, indent=1))
    return record


def _move_marker(reg: Path, label: str) -> Optional[Path]:
    marker = reg / APPROVAL_MARKER
    if not marker.exists():
        return None
    dest_dir = reg / "_archive" / ("%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), label))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / APPROVAL_MARKER
    shutil.move(str(marker), str(dest))
    return dest


def retire_for_new_registration(reg_folder) -> Optional[Path]:
    """Registration is about to run (again): move any approval out of the way,
    into ``_archive/``, so the new registration starts unapproved. Returns
    where it went, or None if there was none."""
    return _move_marker(_reg(reg_folder), "approval_of_previous_registration")


def withdraw(reg_folder, reason: str = "rejected") -> Optional[Path]:
    """A person rejected the registration: the approval (if any) goes into
    ``_archive/`` -- kept, not deleted -- and the brain is unapproved."""
    return _move_marker(_reg(reg_folder), "withdrawn_" + reason)
