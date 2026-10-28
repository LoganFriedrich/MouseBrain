"""Classification and counting read the images the atlas was fitted to.

A brain delivered already cut is registered on its full extraction and has an
empty 2_Cropped_For_Registration. Steps 5 and 6 used to read the crop folder by
name, and step 5 looked for detection results only at the top of
4_Cell_Candidates, where step 4 does not put a run on the full extraction made
before registration recorded that space -- so such a brain could not be
classified or counted.
"""
import importlib.util
import json
from pathlib import Path

from mousebrain.config import (record_registered_space, registered_candidate_folders,
                               registered_input)

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _brain(tmp_path, with_crop=False, with_manual=False, registered_on=None):
    brain = tmp_path / "1_CNT_01_01_1p625x_z4"
    for folder, present in (("1_Extracted_Full", True),
                            ("2_Cropped_For_Registration", with_crop),
                            ("2_Cropped_For_Registration_Manual", with_manual)):
        (brain / folder).mkdir(parents=True)
        if present:
            (brain / folder / "ch0").mkdir()
            (brain / folder / "ch0" / "Z0000.tif").write_bytes(b"")
            (brain / folder / "metadata.json").write_text(json.dumps({"from": folder}))
    if registered_on:
        record_registered_space(brain / "3_Registered_Atlas", registered_on,
                                brain / {"full": "1_Extracted_Full",
                                         "cropped": "2_Cropped_For_Registration",
                                         "manual": "2_Cropped_For_Registration_Manual"}[registered_on])
    return brain


def test_brain_registered_on_full_extraction_reads_the_full_extraction(tmp_path):
    brain = _brain(tmp_path, registered_on="full")
    source, folder, meta = registered_input(brain)
    assert source == "full"
    assert folder == brain / "1_Extracted_Full"
    assert meta == {"from": "1_Extracted_Full"}


def test_recorded_space_wins_over_a_crop_that_exists(tmp_path):
    brain = _brain(tmp_path, with_crop=True, with_manual=True, registered_on="cropped")
    assert registered_input(brain, fallback=("manual", "cropped"))[1] == \
        brain / "2_Cropped_For_Registration"


def test_unrecorded_brain_keeps_each_callers_old_assumption(tmp_path):
    brain = _brain(tmp_path, with_crop=True, with_manual=True)
    assert registered_input(brain)[1] == brain / "2_Cropped_For_Registration"
    assert registered_input(brain, fallback=("manual", "cropped"))[1] == \
        brain / "2_Cropped_For_Registration_Manual"


def test_unrecorded_brain_without_a_crop_still_names_the_crop_folder(tmp_path):
    brain = _brain(tmp_path)
    source, folder, meta = registered_input(brain)
    assert (source, folder, meta) == ("cropped", brain / "2_Cropped_For_Registration", {})


def test_candidate_folders(tmp_path):
    det = "4_Cell_Candidates"
    assert registered_candidate_folders(_brain(tmp_path / "a")) == \
        [tmp_path / "a" / "1_CNT_01_01_1p625x_z4" / det]
    brain = _brain(tmp_path / "b", registered_on="full")
    assert registered_candidate_folders(brain) == \
        [brain / det, brain / det / "from_1_Extracted_Full"]


def _step5():
    spec = importlib.util.spec_from_file_location("classify_step", PIPELINE / "5_classify_cells.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step5_finds_a_full_extraction_run_once_registered_there(tmp_path):
    step5 = _step5()
    brain = _brain(tmp_path)
    run = brain / "4_Cell_Candidates" / "from_1_Extracted_Full"
    run.mkdir(parents=True)
    (run / "detected_cells.xml").write_text("<x/>")
    # not registered on the full extraction: that run is in another space
    assert step5.find_detection_output(brain) is None
    record_registered_space(brain / "3_Registered_Atlas", "full", brain / "1_Extracted_Full")
    assert step5.find_detection_output(brain) == run / "detected_cells.xml"


def test_step5_prefers_the_top_level(tmp_path):
    step5 = _step5()
    brain = _brain(tmp_path, registered_on="full")
    top = brain / "4_Cell_Candidates"
    (top / "from_1_Extracted_Full").mkdir(parents=True)
    (top / "from_1_Extracted_Full" / "detected_cells.xml").write_text("<x/>")
    (top / "detected_cells.xml").write_text("<x/>")
    assert step5.find_detection_output(brain) == top / "detected_cells.xml"
