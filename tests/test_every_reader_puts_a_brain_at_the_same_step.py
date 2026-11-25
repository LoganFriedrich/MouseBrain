"""Every status reader puts a brain at the step the pipeline will actually take up.

A reader must not look for a pre-filter report at the top of
4_Cell_Candidates, where nothing writes one: RUN_PIPELINE would show every
brain past detection at "Pre-filter", including brains already classified and
counted. And a brain whose candidates sit in from_<registered folder> (where
step 5 finds and classifies them) must not be shown at "Detect" by any reader.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from mousebrain.config import record_registered_space
from mousebrain.pipeline_status import brain_status
from mousebrain.registration_approval import approve

from count_withdrawal_helpers import put_in_production

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _readers():
    readers = {"pipeline_status": lambda p: brain_status(p),
               "RUN_PIPELINE": _load("run_pipeline", PIPELINE / "RUN_PIPELINE.py").get_brain_status}
    try:
        from mousebrain.plugin_3d.pipeline_widget import PipelineDashboard
        readers["dashboard"] = lambda p: PipelineDashboard.get_brain_status(None, p)
    except Exception:  # no Qt in this environment: the shared function is still tested
        pass
    return readers


def _brain(tmp_path, registered_on="cropped"):
    b = tmp_path / "M1" / "101_PROJ_01_01_2p5x_z5"
    for folder in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / folder / "ch0").mkdir(parents=True)
        (b / folder / "ch0" / "Z0000.tif").write_bytes(b"")
        (b / folder / "metadata.json").write_text(json.dumps({}))
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, registered_on, b / {"cropped": "2_Cropped_For_Registration",
                                                     "full": "1_Extracted_Full"}[registered_on])
    # A finished registration: all of brainreg's output files, not only the
    # brainreg.json it writes before it starts.
    for name in ("brainreg.json", "registered_atlas.tiff", "boundaries.tiff",
                 "deformation_field_0.tiff"):
        (reg / name).write_text("{}")
    approve(reg, approved_by="tester")
    return b


def _classified_and_counted(b, candidates_dir="4_Cell_Candidates"):
    (b / candidates_dir).mkdir(parents=True, exist_ok=True)
    (b / candidates_dir / "detected_cells.xml").write_text("<x/>")
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text("<x/>")
    (b / "6_Region_Analysis").mkdir()
    (b / "6_Region_Analysis" / "cell_counts_by_region.csv").write_text("a\n")


@pytest.mark.parametrize("prefilter_output", [
    None,
    "prefiltered_20000101_000000/prefilter_report.json",
    "prefiltered/interior_candidates.xml",
])
def test_a_counted_brain_is_done_not_stuck_at_prefilter(tmp_path, monkeypatch, prefilter_output):
    b = _brain(tmp_path)
    _classified_and_counted(b)
    put_in_production(tmp_path, monkeypatch, b.name)
    if prefilter_output:
        out = b / "4_Cell_Candidates" / prefilter_output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
    for name, read in _readers().items():
        assert read(b)["step"] == "done", name


def test_a_prefilter_report_in_its_subfolder_counts_as_prefiltered(tmp_path):
    b = _brain(tmp_path)
    (b / "4_Cell_Candidates" / "prefiltered_20000101_000000").mkdir(parents=True)
    (b / "4_Cell_Candidates" / "detected_cells.xml").write_text("<x/>")
    (b / "4_Cell_Candidates" / "prefiltered_20000101_000000" / "prefilter_report.json").write_text("{}")
    assert brain_status(b)["prefiltered"]
    assert brain_status(b)["step"] == "classify"


def test_candidates_in_the_registered_from_folder_count_as_detected(tmp_path):
    b = _brain(tmp_path, registered_on="full")
    det = b / "4_Cell_Candidates" / "from_1_Extracted_Full"
    det.mkdir(parents=True)
    (det / "detected_cells.xml").write_text("<x/>")
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text("<x/>")
    for name, read in _readers().items():
        s = read(b)
        assert s["detected"], name
        assert s["step"] == "count", name


def test_a_trial_run_off_the_registered_space_is_not_detection(tmp_path):
    # Registered on the crop: a run kept in from_1_Extracted_Full is a trial on
    # the uncropped stack, and must not count (step 5 will not read it).
    b = _brain(tmp_path, registered_on="cropped")
    det = b / "4_Cell_Candidates" / "from_1_Extracted_Full"
    det.mkdir(parents=True)
    (det / "detected_cells.xml").write_text("<x/>")
    assert brain_status(b)["step"] == "detect"
