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

A REGISTRATION WHOSE IMAGES CHANGED IS OUT OF DATE
The atlas is fitted to one set of images (``registered_from.json`` names the
folder). A re-crop rewrites those images; the registration then describes
images that are no longer there, yet it stayed "registered" and "approved" and
step 4 ran on it. Now a registration older than the newest image it was fitted
to, or whose recorded images are gone, is OUT_OF_DATE: no reader counts it as
registered, nobody can approve it, step 4's approval gate is closed, and the
brain is at "register" again (step 3's --batch registers it).

A REJECTED REGISTRATION STAYS REJECTED
Rejecting used to only move an existing approval away (and do nothing at all
for a registration never approved), so the next reader showed the
registration as merely "waiting for approval" and anyone could approve what a
person had just turned down. reject() now writes ``.registration_rejected``
naming the registration (who, when, its fingerprint). Every reader then shows
it as REJECTED, the brain is back at the register step, and approve() refuses
it -- except the approval script, which asks the person whether they mean to
overrule the rejection. Registering again archives the rejection with the
registration it was for.

What happens to approvals:
  * approve()                    -- a person approved what is there now
  * reject()                     -- a person turned the registration down; the
                                    rejection is recorded and any approval
                                    goes into ``_archive/``
  * retire_for_new_registration() -- registration is about to run again: the
                                     approval and any rejection move into
                                     ``_archive/`` with the registration they
                                     were for
  * withdraw()                   -- take an approval back without rejecting
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

# The registration's own output files. Their sizes and modification times
# identify one registration: a new run rewrites all of them. QC pictures and
# registered_from.json are deliberately NOT included -- redrawing a QC picture
# is not a new registration.
FINGERPRINT_FILES = ("brainreg.json", "registered_atlas.tiff", "boundaries.tiff",
                     "deformation_field_0.tiff")

# What approval_state() can say.
NOT_REGISTERED = "not_registered"      # no brainreg.json: nothing to approve
INCOMPLETE = "incomplete"              # started, but an output file is missing
REJECTED = "rejected"                  # a person rejected THIS registration
OUT_OF_DATE = "out_of_date"            # its images were made again, or are gone
UNAPPROVED = "unapproved"              # registered, no marker
APPROVED = "approved"                  # marker names this registration
OTHER_REGISTRATION = "other_registration"  # marker names a different one
UNRECORDED = "unrecorded"              # old-style marker: does not say which

# The states in which the brain needs registering (again) before anything
# else: there is nothing a person could approve.
NEEDS_REGISTRATION = (NOT_REGISTERED, INCOMPLETE, REJECTED, OUT_OF_DATE)

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


def approval_state(reg_folder, images_time=_UNSET) -> Tuple[str, str]:
    """(state, explanation) for the registration in this folder. Only
    APPROVED means a person approved the registration that is there now; the
    states in NEEDS_REGISTRATION mean the brain has to be registered again.

    ``images_time``: see images_problem()."""
    current = registration_fingerprint(reg_folder)
    if current is None:
        return NOT_REGISTERED, _STATE_TEXT[NOT_REGISTERED]
    missing = missing_registration_files(reg_folder)
    if missing:
        return INCOMPLETE, "%s (missing: %s)" % (_STATE_TEXT[INCOMPLETE], ", ".join(missing))
    rejection = read_rejection(reg_folder)
    if rejection is not None and rejection.get("registration") in (None, current):
        # A rejection that cannot be read, or does not say which registration
        # it was for, is still a person's "no" about what is in this folder.
        return REJECTED, "rejected by %s at %s%s -- register the brain again" % (
            rejection.get("rejected_by") or "someone", rejection.get("rejected_at") or "?",
            (" (" + rejection["note"] + ")") if rejection.get("note") else "")
    problem = images_problem(reg_folder, images_time)
    if problem:
        return OUT_OF_DATE, "out of date: %s -- register the brain again" % problem
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


def approve(reg_folder, approved_by: Optional[str] = None, note: str = "",
            allow_rejected: bool = False) -> dict:
    """Record that a person approved the registration in this folder.

    Call this only from something a person did (a button, the approval
    script's yes/no question). Raises ValueError when there is nothing a
    person could approve: no registration (an approval written before the
    registration exists would be inherited by whatever lands there later), or
    a registration that did not finish, or one whose images changed since
    (OUT_OF_DATE). A REJECTED registration is refused
    too, unless ``allow_rejected`` -- only the approval script passes it,
    after asking the person whether they mean to overrule the rejection; the
    rejection then goes into ``_archive/``.
    """
    reg = _reg(reg_folder)
    current = registration_fingerprint(reg)
    if current is None:
        raise ValueError("no registration to approve in %s (brainreg.json not found)" % reg)
    state, text = approval_state(reg)
    if state in (INCOMPLETE, OUT_OF_DATE):
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


def reject(reg_folder, rejected_by: Optional[str] = None, note: str = "") -> dict:
    """Record that a person looked at the registration in this folder and
    turned it down. Any approval goes into ``_archive/``; every reader shows
    the brain back at the register step until it is registered again (step 3
    then archives this rejection with the registration it was for). Raises
    ValueError when there is no registration in the folder."""
    reg = _reg(reg_folder)
    current = registration_fingerprint(reg)
    if current is None:
        raise ValueError("no registration to reject in %s (brainreg.json not found)" % reg)
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
    return record


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
    reject(), which also records that it was turned down."""
    return _move_marker(_reg(reg_folder), "withdrawn_" + reason)
