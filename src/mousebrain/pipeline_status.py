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
  register  ``3_Registered_Atlas/brainreg.json``.
  approve   a person's approval of THIS registration (see
            ``mousebrain.registration_approval``).
  detect    a candidates .xml in a registered-space candidates folder: the top
            of ``4_Cell_Candidates``, or ``from_<the registered folder>``.
  prefilter optional. Done when any pre-filter output exists: a
            ``prefilter_report.json`` in any ``prefilter*`` subfolder, or step
            5's ``prefiltered/interior_candidates.xml``. Never the brain's step
            once it is classified -- step 5 pre-filters on its own by default.
  classify  ``5_Classified_Cells/cells.xml`` (the production classification).
  count     any ``.csv`` in ``6_Region_Analysis``.
"""

from pathlib import Path
from typing import Dict, List, Optional

from mousebrain.config import registered_candidate_folders, registered_space

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

    Same search as step 5 (``find_detection_output``): each registered-space
    folder in turn, and in each the standard name first, then the legacy
    names, then any .xml.
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
    if (det / "prefilter_report.json").exists():  # where an old version looked
        return True
    if any(det.glob("prefilter*/prefilter_report.json")):
        return True
    return (det / "prefiltered" / "interior_candidates.xml").exists()


def _approved(reg: Path) -> bool:
    return (reg / ".registration_approved").exists()


def brain_status(pipeline_dir, include_prefilter: bool = True) -> Dict:
    """Everything a status reader needs about one brain folder.

    Returns a dict with one boolean per item (has_ims, extracted, cropped,
    registered, approved, detected, prefiltered, classified, counted), plus
    ``step`` (a key of STEPS, or 'done'), ``step_name`` and ``step_num``
    (1-based position in ``steps(include_prefilter)``; len+1 when done).
    """
    p = Path(pipeline_dir)
    raw = p / "0_Raw_IMS"
    reg = p / FOLDER_REGISTRATION
    det = p / FOLDER_DETECTION
    analysis = p / FOLDER_ANALYSIS

    recorded_space = registered_space(p)
    status = {
        "has_ims": raw.is_dir() and any(raw.glob("*.ims")),
        "extracted": (p / FOLDER_EXTRACTED / "metadata.json").exists(),
        "cropped": ((p / FOLDER_CROPPED / "metadata.json").exists()
                    or (p / FOLDER_CROPPED_MANUAL / "metadata.json").exists()
                    or recorded_space == "full"),
        "registered": (reg / "brainreg.json").exists(),
        "approved": _approved(reg),
        "detected": find_candidates(p) is not None,
        "prefiltered": _prefiltered(det),
        "classified": (p / FOLDER_CLASSIFICATION / "cells.xml").exists(),
        "counted": analysis.is_dir() and any(analysis.glob("*.csv")),
    }

    shown = steps(include_prefilter)
    status["step"], status["step_name"], status["step_num"] = "done", "Complete!", len(shown) + 1
    for num, (key, name) in enumerate(shown, 1):
        if key == "prefilter" and status["classified"]:
            continue  # optional, and step 5 already did its own
        if not status[_DONE_FLAG[key]]:
            status["step"], status["step_name"], status["step_num"] = key, name, num
            break
    return status


__all__ = ["STEPS", "steps", "brain_status", "find_candidates"]
