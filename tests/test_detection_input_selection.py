"""Which images cell detection runs on, and what each choice commits you to.

Detection does not read the atlas. Cropping and registration exist to put cells
into ATLAS space, which is what counting per region needs -- a later step. So a
brain that has only been extracted can legitimately be detected on, and that
matters whenever registration is waiting on something: a better scan, a manual
crop, a person's approval.

The two dangers these tests pin down:

1. An empty crop folder must not look like data. An un-made crop is the normal
   state of a brain between extraction and cropping, and if it counted as a
   source, detection would read a folder with no images in it.

2. A run on the uncropped stack must not be mistakable for the real one. Its
   coordinates start from a different corner than a crop's do, so if its output
   sat where Script 5 and Script 6 look, those cells would be classified and
   counted against the wrong coordinates -- silently, with a plausible number
   coming out the far end.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "pipeline" / "4_detect_cells.py")


@pytest.fixture(scope="module")
def detect():
    """The script, imported as a module.

    Its filename starts with a digit, so it cannot be imported by name.
    """
    pytest.importorskip("cellfinder", reason="4_detect_cells checks for cellfinder on import")
    spec = importlib.util.spec_from_file_location("detect_cells_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_brain(tmp_path, folders_with_images=(), empty_folders=(), metadata=None):
    """A pipeline folder, with the named subfolders holding a ch0 of tiffs."""
    pipeline = tmp_path / "101_PROJ_01_02_2p5x_z5"
    for name in list(folders_with_images) + list(empty_folders):
        (pipeline / name / "ch0").mkdir(parents=True)
    for name in folders_with_images:
        (pipeline / name / "ch0" / "Z0000.tif").write_bytes(b"not really a tiff")
        meta = metadata if metadata is not None else {
            "voxel_size_um": {"x": 4.0, "y": 4.0, "z": 4.0},
            "channels": {"signal_channel": 0, "background_channel": 1},
        }
        (pipeline / name / "metadata.json").write_text(json.dumps(meta))
    return pipeline


def test_a_manual_crop_wins_over_an_automatic_one(detect, tmp_path):
    """A person's crop is the one they meant to use."""
    pipeline = make_brain(tmp_path, folders_with_images=(
        detect.FOLDER_CROPPED_MANUAL, detect.FOLDER_CROPPED, detect.FOLDER_FULL))
    source, folder, meta = detect.resolve_input(pipeline)
    assert source == "manual"
    assert folder.name == detect.FOLDER_CROPPED_MANUAL


def test_a_crop_wins_over_the_uncropped_stack(detect, tmp_path):
    pipeline = make_brain(tmp_path, folders_with_images=(
        detect.FOLDER_CROPPED, detect.FOLDER_FULL))
    source, folder, meta = detect.resolve_input(pipeline)
    assert source == "cropped"


def test_the_uncropped_stack_is_used_when_nothing_has_been_cropped(detect, tmp_path):
    """The case this whole feature exists for."""
    pipeline = make_brain(tmp_path, folders_with_images=(detect.FOLDER_FULL,),
                          empty_folders=(detect.FOLDER_CROPPED,))
    source, folder, meta = detect.resolve_input(pipeline)
    assert source == "full"
    assert folder.name == detect.FOLDER_FULL
    assert meta["voxel_size_um"]["z"] == 4.0, "voxel sizes must come from the folder used"


def test_an_empty_crop_folder_is_not_data(detect, tmp_path):
    """The folder exists from the moment the brain is organised; that is not data.

    If an existing-but-empty crop folder counted, detection would be pointed at
    a folder with no images in it and fail deep inside cellfinder, long after the
    point where the real problem could still be named.
    """
    pipeline = make_brain(tmp_path, folders_with_images=(detect.FOLDER_FULL,),
                          empty_folders=(detect.FOLDER_CROPPED,
                                         detect.FOLDER_CROPPED_MANUAL))
    source, _, _ = detect.resolve_input(pipeline)
    assert source == "full"


def test_asking_for_a_source_that_has_no_images_fails_rather_than_substituting(detect, tmp_path):
    """--source cropped means that folder, not "whatever you can find".

    Someone who names a source is making a statement about which coordinate
    space they want. Quietly giving them a different one would be worse than
    refusing.
    """
    pipeline = make_brain(tmp_path, folders_with_images=(detect.FOLDER_FULL,))
    source, folder, meta = detect.resolve_input(pipeline, "cropped")
    assert source is None and folder is None


def test_a_brain_with_nothing_extracted_has_no_input(detect, tmp_path):
    pipeline = make_brain(tmp_path)
    assert detect.resolve_input(pipeline) == (None, None, None)


def test_an_unknown_source_is_a_mistake_not_a_fallback(detect, tmp_path):
    pipeline = make_brain(tmp_path, folders_with_images=(detect.FOLDER_FULL,))
    with pytest.raises(ValueError):
        detect.resolve_input(pipeline, "croped")


def test_only_the_crops_are_treated_as_registered_space(detect):
    """The fact the output location and the approval gate both hang on."""
    assert set(detect.SOURCES_IN_REGISTERED_SPACE) == {"manual", "cropped"}
    assert "full" not in detect.SOURCES_IN_REGISTERED_SPACE


def test_detecting_on_the_uncropped_stack_is_explained_not_just_allowed(detect):
    """A person starting this run has to be told what it will not give them."""
    text = detect.describe_input_choice("full")
    assert "region" in text, "must say region counts are not possible"
    assert "trial run" in text
    assert detect.FOLDER_DETECTION in text, "must say where results go"

    crop_text = detect.describe_input_choice("cropped")
    assert "as usual" in crop_text


def test_registration_is_reported_for_unregistered_brains_not_used_to_hide_them(
        detect, tmp_path):
    """Listing used to require registration, so this path was unreachable by CLI.

    The consequence was not that people did not do it -- they did it in napari --
    but that those runs were never logged in the tracker.
    """
    root = tmp_path / "1_Brains"
    mouse = root / "101_PROJ_01_02"
    mouse.mkdir(parents=True)
    make_brain(mouse, folders_with_images=(detect.FOLDER_FULL,))

    brains = detect.list_available_brains(root)
    assert len(brains) == 1, "an extracted, unregistered brain must be offered"
    assert brains[0]["registered"] is False
    assert brains[0]["approved"] is False
    assert brains[0]["source"] == "full"
