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

Three rules, all applied here:
  1. Changing brain resets every per-brain field (PER_BRAIN_DEFAULTS).
  2. Image layers are tagged with the brain and folder they were loaded from
     (``layer.metadata['source_brain']`` / ``['source_folder']``); detection and
     classification use only layers of the selected brain, and refuse when the
     viewer holds only another brain's images.
  3. Metadata is saved only into the folder it was loaded from.
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
