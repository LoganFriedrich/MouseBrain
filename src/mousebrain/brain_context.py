"""What the napari Setup & Tuning widget holds for ONE brain, and how it lets go.

WHY THIS EXISTS
The widget keeps a dozen pieces of per-brain state (the brain's metadata, the
last pre-filter result, the selected classification, the training-data folder,
the last run ids ...) and napari's image layers are global to the viewer.
Selecting another brain changed only ``current_brain``: everything else stayed.
So detection could run on brain A's pixels and be logged under brain B, Save
could write A's metadata into B's folders, and a pre-filter result made on A
could be saved into B. On disk this showed as a training export logged under
one brain with its cubes in another brain's folder, and an extraction's
metadata.json overwritten by the crop's.

Four rules, all applied here:
  1. Changing brain resets every per-brain field (PER_BRAIN_DEFAULTS).
  2. Image layers are tagged with the brain and folder they were loaded from
     (``layer.metadata['source_brain']`` / ``['source_folder']``); detection and
     classification use only layers of the selected brain, and refuse when the
     viewer holds only another brain's images.
  3. Metadata is saved only into the folder it was loaded from.
  4. Points layers that show a detection run carry the same brain tag, the run
     id and their role (Best / recent); the Best buttons act only on layers of
     the selected brain and find them by that record, never by name (see
     "Detection-run points layers" below).
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple

# field name -> fresh value (a callable makes a new object each time)
PER_BRAIN_DEFAULTS = {
    "metadata": dict,
    "_metadata_source": None,
    "last_run_id": None,
    "last_run_cells": 0,
    "_prefilter_result": None,
    "_prefilter_source_exp_id": None,
    "_last_prefilter_exp_id": None,
    "training_data_dir": None,
    "_cells_xml": None,
    "_non_cells_xml": None,
    "_selected_class_run": None,
    "_selected_cells_xml": None,
    "_detection_z_offset": 0,
    "_last_output_path": None,
    "_last_class_output": None,
    "_last_class_exp_id": None,
    "_auto_classify_triggered": False,
    # Which registration the approval label / the QC pictures showed: Approve
    # signs only that one (tuning_widget.approve_registration).
    "_shown_registration_id": None,
    "_qc_view_registration_id": None,
}


def reset_per_brain_state(obj) -> None:
    """Give every per-brain field of the widget its fresh value."""
    for name, default in PER_BRAIN_DEFAULTS.items():
        setattr(obj, name, default() if callable(default) else default)


def tag_layer(layer, brain: Path, folder: Optional[Path]) -> None:
    """Record on an image layer which brain and folder its pixels came from."""
    layer.metadata["source_brain"] = str(brain)
    layer.metadata["source_folder"] = str(folder) if folder else ""


def _role(name: str) -> Optional[str]:
    low = name.lower()
    if "signal" in low:
        return "signal"
    if "background" in low:
        return "background"
    return None


def pick_image_layers(layers: Iterable, brain: Path) -> Tuple[object, object, bool]:
    """(signal, background, other_images_present) among the viewer's image layers.

    Only layers tagged as loaded from ``brain`` count. ``other_images_present``
    is True when signal/background layers exist that belong to another brain
    (or carry no tag, so their brain is unknown) -- the caller must then refuse
    rather than silently read the selected brain's images from disk.
    """
    signal = background = None
    other = False
    for layer in layers:
        role = _role(getattr(layer, "name", ""))
        if role is None or not hasattr(layer, "data"):
            continue
        meta = getattr(layer, "metadata", None) or {}
        if meta.get("source_brain") != str(brain):
            other = True
            continue
        if role == "signal":
            signal = layer
        else:
            background = layer
    return signal, background, other


def layer_source_folder(layer) -> str:
    """The folder a tagged layer's pixels were read from ('' if unknown)."""
    return (getattr(layer, "metadata", None) or {}).get("source_folder", "") or ""


def metadata_save_target(metadata_source) -> Optional[Path]:
    """Where Save may write metadata.json: the folder it was loaded from, or
    None when nothing was loaded (then Save must refuse -- writing a new
    metadata.json would forge "extracted"/"cropped" for the status readers)."""
    return Path(metadata_source) if metadata_source else None


# ---------------------------------------------------------------------------
# Detection-run points layers: which brain and run they show
# ---------------------------------------------------------------------------
#
# The curation buttons (Mark as Best, Promote Recent, Keep Best, Difference)
# act on "the Best layer" and "the recent layer". They used to find them by
# NAME as well as by metadata, and in the whole viewer. A Best layer kept the
# dot marker of a this-session run in its name, so it was also found as the recent
# layer: Keep Best hid the Best, and Promote archived the very run it then
# marked best. And a Best layer of ANOTHER brain still open in the viewer was
# archived -- in the tracker too -- when this brain's Best was marked.
#
# So a detection-run layer is identified only by what was recorded on it when
# it was made: the brain (``source_brain``, the same tag image layers carry),
# the run (``exp_id``) and its role (``cell_type`` 'det_best' / 'det_recent').
# A layer without them is never treated as a run.

RUN_LAYER_ROLES = ("det_best", "det_recent")


def tag_run_layer(layer, brain: Path, exp_id: Optional[str], role: Optional[str]) -> None:
    """Record on a points layer the brain and calibration run it shows."""
    meta = layer.metadata
    meta["source_brain"] = str(brain)
    meta["source_folder"] = ""
    if exp_id:
        meta["exp_id"] = exp_id
    if role:
        meta["cell_type"] = role


def run_layers(layers: Iterable, brain: Path, role: str) -> list:
    """The layers among ``layers`` that show a run of ``brain`` in ``role``.

    Matched on metadata only, never on the layer's name, and only for layers
    tagged with this brain. ``layers`` must already be points layers."""
    found = []
    for layer in layers:
        meta = getattr(layer, "metadata", None) or {}
        if meta.get("cell_type") != role:
            continue
        if meta.get("source_brain") != str(brain):
            continue
        found.append(layer)
    return found


# Name prefixes the widget has put on run layers over time. Stripped before a
# new one is set so a layer never carries two contradictory ones.
_RUN_NAME_PREFIXES = ("BEST:", "RECENT:", "[archived]", "[not best]",
                      "\u2605", "\u25cf", "\u25c6")  # star, dot, diamond


def run_layer_base_name(name: str) -> str:
    """A run layer's name without any Best / recent / archived prefix."""
    name = (name or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in _RUN_NAME_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                changed = True
    return name


def run_points_file(output_path) -> Optional[Path]:
    """The candidates file a detection run's ``output_path`` points at.

    Widget runs used to record their run FOLDER (the file inside it is
    ``detected_cells.xml``); scripts record the file itself. Returns None when
    there is no such file, so a caller says so instead of loading nothing."""
    if not output_path:
        return None
    p = Path(output_path)
    if p.is_dir():
        p = p / "detected_cells.xml"
    return p if p.is_file() else None
