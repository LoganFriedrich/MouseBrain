"""Everything after registration must work in the space the atlas was fitted to.

Cells placed in the wrong space land in the wrong structures, and every number
downstream still looks perfectly reasonable -- a region count does not announce
that it counted the wrong voxels. So this is not a tidiness question.

The steps used to assume that space was the crop folder. That holds only when the
crop was made here. A delivery that arrives already cut has nothing for the crop
step to do, and is registered on the full extraction -- at which point the full
extraction IS registered space, and the assumption is wrong in the direction that
hides itself.

So registration records what it used, and later steps ask.
"""
import json
from pathlib import Path

import pytest

from mousebrain import config


def brain_with(tmp_path, folders_with_images=(), empty=()):
    pipeline = tmp_path / "101_PROJ_01_02_2p5x_z5"
    for name in list(folders_with_images) + list(empty):
        (pipeline / name / "ch0").mkdir(parents=True)
    for name in folders_with_images:
        (pipeline / name / "ch0" / "Z0000.tif").write_bytes(b"x")
        (pipeline / name / "metadata.json").write_text(
            json.dumps({"voxel_size_um": {"x": 4.0, "y": 4.0, "z": 4.0}}),
            encoding="utf-8")
    return pipeline


# ------------------------------------------------------------ choosing images
def test_a_hand_crop_is_preferred(tmp_path):
    p = brain_with(tmp_path, ("2_Cropped_For_Registration_Manual",
                              "2_Cropped_For_Registration", "1_Extracted_Full"))
    assert config.resolve_pipeline_input(p)[0] == "manual"


def test_a_delivery_that_arrived_already_cut_can_still_be_registered(tmp_path):
    """The case that was impossible before: no crop, because none was needed."""
    p = brain_with(tmp_path, ("1_Extracted_Full",),
                   empty=("2_Cropped_For_Registration",))
    source, folder, metadata = config.resolve_pipeline_input(p)
    assert source == "full"
    assert folder.name == "1_Extracted_Full"
    assert metadata["voxel_size_um"]["z"] == 4.0


def test_an_empty_crop_folder_is_not_data(tmp_path):
    """It exists from the moment a brain is organised. That is not a crop."""
    p = brain_with(tmp_path, ("1_Extracted_Full",),
                   empty=("2_Cropped_For_Registration",
                          "2_Cropped_For_Registration_Manual"))
    assert config.resolve_pipeline_input(p)[0] == "full"


def test_a_brain_with_no_images_resolves_to_nothing(tmp_path):
    assert config.resolve_pipeline_input(brain_with(tmp_path)) == (None, None, None)


def test_insisting_on_a_source_that_is_empty_does_not_substitute(tmp_path):
    p = brain_with(tmp_path, ("1_Extracted_Full",))
    assert config.resolve_pipeline_input(p, "cropped")[0] is None


# --------------------------------------------------------- recording the space
def test_registration_records_what_it_was_fitted_to(tmp_path):
    p = brain_with(tmp_path, ("1_Extracted_Full",))
    config.record_registered_space(p / "3_Registered_Atlas", "full",
                                   p / "1_Extracted_Full")
    assert config.registered_space(p) == "full"


def test_a_brain_registered_before_this_existed_reports_nothing(tmp_path):
    """And the caller must fall back, not treat the brain as unregistered."""
    p = brain_with(tmp_path, ("2_Cropped_For_Registration",))
    assert config.registered_space(p) == ""


def test_detection_treats_the_recorded_space_as_the_registered_one(tmp_path):
    """The whole point: a full-stack run on a pre-cut delivery is NOT a trial run."""
    pytest.importorskip("cellfinder",
                        reason="4_detect_cells checks for cellfinder on import")
    import importlib.util
    import sys

    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "pipeline" / "4_detect_cells.py")
    spec = importlib.util.spec_from_file_location("detect_for_space", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    p = brain_with(tmp_path, ("1_Extracted_Full",))
    assert "full" not in module.sources_in_registered_space(p), (
        "with nothing recorded, only a crop counts")

    config.record_registered_space(p / "3_Registered_Atlas", "full",
                                   p / "1_Extracted_Full")
    assert "full" in module.sources_in_registered_space(p)
    assert "cropped" not in module.sources_in_registered_space(p), (
        "once recorded, the recorded space is the only registered one")
