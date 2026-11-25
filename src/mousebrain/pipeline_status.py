"""Which pipeline step a brain is at -- one answer, shared by every reader.

WHY THIS MODULE EXISTS
The pipeline keeps no explicit "this brain is at step N" record. The step is
worked out from which files exist in the brain's numbered folders. Until this
module, every reader worked it out for itself -- RUN_PIPELINE, the napari
pipeline dashboard, the overnight batch -- each with its own idea of which file
proves which step. They disagreed, and two of them were wrong in ways that kept
brains at the wrong step for months:

* RUN_PIPELINE's "prefilter" stage looked for
  ``4_Cell_Candidates/prefilter_report.json`` at the TOP of the candidates
  folder. Nothing ever writes it there: the pre-filter writes its report into a
  ``prefiltered_<time>/`` subfolder, and step 5's built-in pre-filter writes
  ``prefiltered/interior_candidates.xml`` with no report at all. So every brain
  past detection -- including brains already classified and counted -- was
  shown at "Pre-filter" forever.
* "Detected" meant "an .xml at the top of 4_Cell_Candidates". Step 4 writes a
  run on images that are not the registered crop into ``from_<folder>/`` (so a
  trial run can never be classified by mistake); once the atlas is fitted to
  those very images that subfolder IS the registered space and step 5 reads it
  (``config.registered_candidate_folders``). The readers did not look there, so
  a brain that step 5 had already classified was shown at "Detect".

The rules below are the ones the pipeline steps themselves use, so what a
reader shows is what the next step will actually do.

THE RULES (first missing item, in this order, is the brain's step)
  extract   ``1_Extracted_Full/metadata.json`` -- written last by extraction,
            so it exists only when every channel finished.
  crop      ``metadata.json`` in ``2_Cropped_For_Registration`` or
            ``2_Cropped_For_Registration_Manual``; OR the registration records
            that the atlas was fitted to the full extraction (a delivery that
            arrived already cut has nothing to crop).
  register  a COMPLETE registration in ``3_Registered_Atlas``: every output
            file brainreg writes (registration_approval.registration_complete).
            ``brainreg.json`` alone is not enough -- brainreg writes it before
            it registers, so a failed or stopped run leaves it behind.
            A registration a person REJECTED does not count either, nor one
            that is OUT OF DATE (its images were made again after it, or are
            gone): the brain is back at "register"
            (registration_approval). Nor one that is RUNNING right now
            (step 3 holds .registration_running): the brain is at
            "register" with "being registered right now", and step 3
            refuses to start a second registration of it.
  approve   a person's approval of THIS registration (see
            ``mousebrain.registration_approval``). An approval of an earlier
            registration, or one that does not say which registration it was
            for, is not an approval.
  detect    a candidates .xml in a registered-space candidates folder: the top
            of ``4_Cell_Candidates``, or ``from_<the registered folder>``.
  prefilter optional. Done when any pre-filter output exists: a
            ``prefilter_report.json`` in any ``prefilter*`` subfolder, or step
            5's ``prefiltered/interior_candidates.xml``. Never the brain's step
            once it is classified -- step 5 pre-filters on its own by default.
  classify  ``5_Classified_Cells/cells.xml`` (the production classification).
  count     any ``.csv`` in ``6_Region_Analysis`` AND a row for the brain in
            the production table ``2_Data_Summary/region_counts.csv``
            (region_counts_io.production_row, the rule ``6_count_regions.py
            --all`` uses). A withdrawal (rejection, approval taken back,
            re-crop, re-registration, classification replaced) takes the row
            out and leaves the brain's own files, so files without a row are
            shown with the reason and the brain is back at "count". When the
            table cannot be read the brain is at "count" too, with a reason
            that says so (never "withdrawn").

STALE OUTPUTS (reported in ``status['stale']`` and treated as NOT done)
An output made from something that has since been replaced is stale: it
exists, but it describes an earlier state of the brain. Nothing is deleted or
moved -- the step is simply shown again, with the reason, so a person re-runs
it. What each output depends on:
  registration    the images it was fitted to (registered_from.json names
                  them). Older than the newest of them (a re-crop), or the
                  images are gone: stale, and its approval no longer counts.
  counts          the registration (regions come from the atlas fit) and the
                  classification. Older than either, or made on a registration
                  that is stale: stale. So re-registering always makes the
                  counts stale. And (brain_status) counts that are no longer
                  in the production table: stale, even on an approved,
                  current registration -- the case of a rejection a person
                  overruled.
  classification  the candidates it classified, and the images. Older than
                  either: stale.
  detection       the images the atlas is fitted to: its coordinates are
                  positions in those images, not in the atlas, so re-fitting
                  the atlas to the SAME images does not change them. Older
                  than the images (a re-crop): stale. When the registration
                  does not record which images it was fitted to, older than
                  the registration: stale, because nothing can say the images
                  are the same.
A stale earlier output makes everything after it stale too.
"""

from pathlib import Path
from typing import Dict, List, Optional

from mousebrain.config import (RegisteredImagesMissing, registered_candidate_folders,
                               registered_space)
from mousebrain.region_counts_io import (NOT_IN_PRODUCTION, TABLE_UNREADABLE,
                                         production_row)
from mousebrain.registration_approval import (APPROVED, NEEDS_REGISTRATION,
                                              NOT_REGISTERED, approval_state,
                                              images_problem,
                                              registered_images_time,
                                              registration_complete,
                                              registration_time)

FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_CROPPED_MANUAL = "2_Cropped_For_Registration_Manual"
FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"
FOLDER_CLASSIFICATION = "5_Classified_Cells"
FOLDER_ANALYSIS = "6_Region_Analysis"

# (key, what the person is asked to do next). The order is the pipeline order.
STEPS = [
    ("extract", "Extract images from IMS file"),
    ("crop", "Crop to remove spinal cord"),
    ("register", "Register to atlas"),
    ("approve", "Review and approve registration"),
    ("detect", "Detect cells"),
    ("prefilter", "Pre-filter candidates (atlas)"),
    ("classify", "Classify cells"),
    ("count", "Count cells by region"),
]

# The flag in the status dict that says each step is done.
_DONE_FLAG = {
    "extract": "extracted", "crop": "cropped", "register": "registered",
    "approve": "approved", "detect": "detected", "prefilter": "prefiltered",
    "classify": "classified", "count": "counted",
}


def steps(include_prefilter: bool = True) -> List[tuple]:
    """The step list a reader shows. The napari dashboard has no pre-filter row."""
    return [s for s in STEPS if include_prefilter or s[0] != "prefilter"]


def find_candidates(pipeline_dir) -> Optional[Path]:
    """The candidates file step 5 would classify, or None.

    THE rule for "the candidates": step 5 (``find_detection_output``) calls
    this, so the file it classifies is the file every other reader checks.
    Each registered-space folder in turn, and in each the standard name first,
    then the legacy names, then any .xml -- the first BY NAME, because a
    network share does not promise to list a folder in name order.
    """
    for folder in registered_candidate_folders(pipeline_dir):
        if not folder.is_dir():
            continue
        for pattern in ("detected_cells.xml", "cell_classification.xml", "*.xml"):
            hits = sorted(folder.glob(pattern))
            if hits:
                return hits[0]
    return None


def _prefiltered(det: Path) -> bool:
    if not det.is_dir():
        return False
    if (det / "prefilter_report.json").exists():  # legacy location
        return True
    if any(det.glob("prefilter*/prefilter_report.json")):
        return True
    return (det / "prefiltered" / "interior_candidates.xml").exists()


# File times are compared with this much slack: copies and network shares
# round modification times, and a step that finishes within the same couple of
# seconds as its input is not "older" than it.
_SLACK_SECONDS = 2.0


def _mtime(path: Optional[Path]) -> Optional[float]:
    try:
        return path.stat().st_mtime if path is not None else None
    except OSError:
        return None


def _newest_csv(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    csvs = list(folder.glob("*.csv"))
    return max(csvs, key=lambda f: f.stat().st_mtime) if csvs else None


def images_time(pipeline_dir) -> Optional[float]:
    """When the images the atlas is fitted to were last written (newest
    ``ch*/*.tif`` of that folder), or None when there are none
    (registration_approval.registered_images_time). None too when the
    registration's recorded images are gone (config.RegisteredImagesMissing):
    a status reader must still be able to show the brain -- stale_outputs
    and approval_state say that the images are gone."""
    try:
        return registered_images_time(pipeline_dir)
    except RegisteredImagesMissing:
        return None


def _images(pipeline_dir):
    """(newest image time or None, why the registered images are gone or "")."""
    try:
        return registered_images_time(pipeline_dir), ""
    except RegisteredImagesMissing as e:
        return None, str(e)


def _older(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and a + _SLACK_SECONDS < b


def stale_outputs(pipeline_dir, candidates: Optional[Path] = None,
                  images=None) -> Dict[str, str]:
    """{'register'|'detect'|'classify'|'count': why} for outputs made from
    something that has since been replaced. See STALE OUTPUTS in the module
    docstring. ``images``: brain_status's own _images() result, so the image
    folders are listed once per brain."""
    p = Path(pipeline_dir)
    reg_t = registration_time(p / FOLDER_REGISTRATION)
    img_t, gone = images if images is not None else _images(p)
    cand = candidates if candidates is not None else find_candidates(p)
    cand_t = _mtime(cand)
    cells_t = _mtime(p / FOLDER_CLASSIFICATION / "cells.xml")
    counts_t = _mtime(_newest_csv(p / FOLDER_ANALYSIS))
    stale = {}

    if reg_t is not None:
        problem = (("the images it was fitted to are gone: " + gone) if gone
                   else images_problem(p / FOLDER_REGISTRATION, img_t))
        if problem:
            stale["register"] = problem

    if _older(cand_t, img_t):
        stale["detect"] = ("the candidates are older than the images they are positions in "
                           "(the images were made again since)")
    elif not registered_space(p) and _older(cand_t, reg_t):
        stale["detect"] = ("the candidates are older than the registration, and the registration "
                           "does not record which images it was fitted to")

    if cells_t is not None:
        if "detect" in stale:
            stale["classify"] = "made from stale candidates"
        elif _older(cells_t, cand_t):
            stale["classify"] = "older than the candidates it should be made from"
        elif _older(cells_t, img_t):
            stale["classify"] = "older than the images"

    if counts_t is not None:
        if "register" in stale:
            stale["count"] = "counted on a registration that no longer fits its images"
        elif "classify" in stale:
            stale["count"] = "counted from a stale classification"
        elif _older(counts_t, reg_t):
            stale["count"] = "counted on an earlier registration (the brain was registered again since)"
        elif _older(counts_t, cells_t):
            stale["count"] = "counted from an earlier classification"
    return stale


def brain_status(pipeline_dir, include_prefilter: bool = True) -> Dict:
    """Everything a status reader needs about one brain folder.

    Returns a dict with one boolean per item (has_ims, extracted, cropped,
    registered, approved, detected, prefiltered, classified, counted) -- a
    stale output counts as not done -- plus ``stale`` ({step: why}),
    ``approval`` / ``approval_text`` (see registration_approval),
    ``register_problem`` (why the registration in the folder does not count as
    one, e.g. it did not finish; "" otherwise), ``candidates``
    (the file step 5 would classify, or None), ``step`` (a key of STEPS, or
    'done'), ``step_name`` and ``step_num`` (1-based position in
    ``steps(include_prefilter)``; len+1 when done).
    """
    p = Path(pipeline_dir)
    raw = p / "0_Raw_IMS"
    reg = p / FOLDER_REGISTRATION
    det = p / FOLDER_DETECTION
    analysis = p / FOLDER_ANALYSIS

    recorded_space = registered_space(p)
    # The registered images are listed once, and only for a finished
    # registration (nothing else needs them), then shared by the approval
    # state and the stale-output rules.
    images = _images(p) if registration_complete(reg) else (None, "")
    approval, approval_text = (approval_state(reg) if images[1]
                               else approval_state(reg, images_time=images[0]))
    candidates = find_candidates(p)
    status = {
        "has_ims": raw.is_dir() and any(raw.glob("*.ims")),
        "extracted": (p / FOLDER_EXTRACTED / "metadata.json").exists(),
        "cropped": ((p / FOLDER_CROPPED / "metadata.json").exists()
                    or (p / FOLDER_CROPPED_MANUAL / "metadata.json").exists()
                    or recorded_space == "full"),
        # Finished, and not one that has to be registered again (a person
        # rejected it ...): registration_approval.NEEDS_REGISTRATION.
        "registered": registration_complete(reg) and approval not in NEEDS_REGISTRATION,
        "approved": approval == APPROVED,
        "detected": candidates is not None,
        "prefiltered": _prefiltered(det),
        "classified": (p / FOLDER_CLASSIFICATION / "cells.xml").exists(),
        "counted": analysis.is_dir() and any(analysis.glob("*.csv")),
    }

    status["approval"] = approval
    status["approval_text"] = approval_text
    # Why a registration that is there does not count (it did not finish ...),
    # for the readers' "Registered to atlas" line; "" when there is none or
    # it counts.
    status["register_problem"] = (approval_text if approval in NEEDS_REGISTRATION
                                  and approval != NOT_REGISTERED else "")
    status["candidates"] = candidates
    # True when the brain's count files are there but region_counts.csv could
    # not be read, so nobody can tell whether it is counted. The batches hold
    # the brain at the count step on this instead of counting it again.
    status["count_unknown"] = False
    status["stale"] = (stale_outputs(p, candidates, images=images)
                       if registration_complete(reg) else {})
    if status["register_problem"] and status["counted"]:
        # Regions come from the atlas fit; counts made on a fit that no longer
        # counts (a person rejected it ...) are not counts of this brain.
        status["stale"]["count"] = ("counted on a registration that no longer counts: "
                                    + status["register_problem"])
    if status["counted"] and "count" not in status["stale"]:
        # The brain's own count files are there and current. "Counted" also
        # needs its row in the production table (region_counts_io.production_row,
        # the rule 6_count_regions.py --all uses): a withdrawal takes the row
        # out and leaves these files. Without this, a brain whose rejection
        # a person overruled -- the SAME registration approved again -- read
        # as Complete with no counts in production, and nothing asked for it
        # to be counted again.
        in_production = production_row(p.name)
        if in_production == NOT_IN_PRODUCTION:
            status["stale"]["count"] = (
                "its counts are not in the production table region_counts.csv "
                "(withdrawn when its registration was rejected, its approval taken "
                "back, or it was cropped or registered again; or never written "
                "there) -- count it again")
        elif in_production == TABLE_UNREADABLE:
            status["count_unknown"] = True
            status["stale"]["count"] = (
                "region_counts.csv could not be read, so it cannot be told whether "
                "its counts are in production -- check that the file opens, then look again")
    for key in status["stale"]:
        status[_DONE_FLAG[key]] = False

    shown = steps(include_prefilter)
    status["step"], status["step_name"], status["step_num"] = "done", "Complete!", len(shown) + 1
    for num, (key, name) in enumerate(shown, 1):
        if key == "prefilter" and status["classified"]:
            continue  # optional, and step 5 already did its own
        if not status[_DONE_FLAG[key]]:
            status["step"], status["step_name"], status["step_num"] = key, name, num
            break
    return status


__all__ = ["STEPS", "steps", "brain_status", "find_candidates", "stale_outputs", "images_time"]
