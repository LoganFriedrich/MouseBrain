"""Every pipeline step agrees on which folders are brains.

Each step used to carry its own idea of which mouse folders to skip: extraction
and registration skipped script/backup/archive, detection, classification,
counting and RUN_PIPELINE also skipped summary, the organizer also skipped old,
and the lookups by name skipped nothing -- not even an underscore folder. So a
brain kept in a folder one step skipped was processed by another, and the steps
disagreed about which brains exist.

The overnight batch went further: it took only brains with images in the
automatic crop folder, so a brain cropped by hand, or registered on its full
extraction, never ran overnight and nothing said so.
"""
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
BRAIN = "101_PROJ_01_02_2p5x_z5"


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, PIPELINE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _images(folder: Path):
    (folder / "ch0").mkdir(parents=True)
    (folder / "ch0" / "Z0000.tif").write_bytes(b"x")
    (folder / "metadata.json").write_text(json.dumps({"voxel_size_um": {"x": 4, "y": 4, "z": 4}}),
                                          encoding="utf-8")


def _brain(folder: Path) -> Path:
    """A brain at every step: raw file, images, candidates, cells, counts."""
    (folder / "0_Raw_IMS").mkdir(parents=True)
    (folder / "0_Raw_IMS" / (folder.name + ".ims")).write_bytes(b"x")
    _images(folder / "1_Extracted_Full")
    _images(folder / "2_Cropped_For_Registration")
    (folder / "4_Cell_Candidates").mkdir()
    (folder / "4_Cell_Candidates" / "detected_cells.xml").write_text("<Marker/>", encoding="utf-8")
    (folder / "5_Classified_Cells").mkdir()
    (folder / "5_Classified_Cells" / "cells.xml").write_text("<Marker/>", encoding="utf-8")
    (folder / "6_Region_Analysis").mkdir()
    (folder / "6_Region_Analysis" / "counts.csv").write_text("a\n1\n", encoding="utf-8")
    return folder


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "1_Brains"
    _brain(root / "mouse_a" / BRAIN)
    # Folders that some step used to treat as mice:
    _brain(root / "_incoming" / "102_PROJ_01_03_2p5x_z5")       # underscore
    _brain(root / "Data_Summary_copy" / "103_PROJ_01_04_2p5x_z5")  # 'summary'
    _brain(root / "old_deliveries" / "104_PROJ_01_05_2p5x_z5")   # 'old'
    return root


def test_the_listing_steps_list_the_same_brains(root):
    extract = _load("2_extract_and_analyze.py", "extract_under_test")
    register = _load("3_register_to_atlas.py", "register_under_test")
    run_pipeline = _load("RUN_PIPELINE.py", "run_pipeline_under_test")
    seen = {
        "extract": sorted(p.name for _ims, p, _m in extract.find_organized_pipelines(root)),
        "register": sorted(p.name for p, _m, _meta in register.find_extracted_pipelines(root)),
    }
    run_pipeline.BRAINS_ROOT = root
    seen["RUN_PIPELINE"] = sorted(b["name"] for b in run_pipeline.find_brains())
    assert seen == {k: [BRAIN] for k in seen}


@pytest.mark.parametrize("script", ["5_classify_cells.py", "6_count_regions.py"])
def test_a_step_never_finds_a_brain_by_name_in_a_folder_the_others_skip(root, script):
    mod = _load(script, script.replace(".py", "_under_test").lstrip("0123456789_"))
    assert mod.find_pipeline(BRAIN, root)[0] == root / "mouse_a" / BRAIN
    assert mod.find_pipeline("102_PROJ_01_03_2p5x_z5", root) == (None, None)
    listed = sorted(b["pipeline"].name for b in mod.list_available_brains(root))
    assert listed == [BRAIN]


# ---------------------------------------------------------------- overnight batch

@pytest.fixture
def overnight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the module opens its log file in the working folder
    mod = _load("overnight_batch.py", "overnight_under_test")
    yield mod
    for h in list(logging.getLogger("overnight_under_test").handlers) + list(logging.root.handlers):
        if isinstance(h, logging.FileHandler) and str(tmp_path) in h.baseFilename:
            logging.root.removeHandler(h)
            h.close()


def _run_dry(mod, root, caplog, monkeypatch):
    monkeypatch.setattr(mod, "BRAINS_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["overnight_batch.py", "--dry-run"])
    with caplog.at_level(logging.INFO):
        mod.main()
    return caplog.text


def test_overnight_takes_a_hand_cropped_brain(overnight, tmp_path, caplog, monkeypatch):
    root = tmp_path / "1_Brains"
    brain = root / "mouse_a" / BRAIN
    _images(brain / "1_Extracted_Full")
    _images(brain / "2_Cropped_For_Registration_Manual")
    text = _run_dry(overnight, root, caplog, monkeypatch)
    assert "  - %s [" % BRAIN in text


def test_overnight_takes_a_brain_registered_on_its_full_extraction(overnight, tmp_path, caplog,
                                                                     monkeypatch):
    root = tmp_path / "1_Brains"
    brain = root / "mouse_a" / BRAIN
    _images(brain / "1_Extracted_Full")
    (brain / "3_Registered_Atlas").mkdir()
    (brain / "3_Registered_Atlas" / "brainreg.json").write_text("{}", encoding="utf-8")
    text = _run_dry(overnight, root, caplog, monkeypatch)
    assert "  - %s [" % BRAIN in text


def test_overnight_names_a_brain_it_leaves_out(overnight, tmp_path, caplog, monkeypatch):
    root = tmp_path / "1_Brains"
    _images(root / "mouse_a" / BRAIN / "1_Extracted_Full")
    text = _run_dry(overnight, root, caplog, monkeypatch)
    assert "  - %s [" % BRAIN not in text
    assert "%s: not cropped and not registered yet" % BRAIN in text


def test_overnight_leaves_out_only_brains_never_a_folder_beside_them(overnight, tmp_path, caplog,
                                                                     monkeypatch):
    """A folder in a mouse folder that is not a brain (training data, a stray
    copy of a numbered output folder) is passed over without a word. It used
    to be logged as left out with "no extracted images yet (run step 2)",
    sending the operator to extract a folder that holds no brain. A brain that
    has its raw file but no extraction is still named, with that reason."""
    root = tmp_path / "1_Brains"
    _images(root / "mouse_a" / BRAIN / "2_Cropped_For_Registration")
    (root / "mouse_a" / "training_data" / "cells").mkdir(parents=True)
    (root / "mouse_a" / "3_Registered_Atlas").mkdir()
    waiting = root / "mouse_b" / "102_PROJ_01_03_2p5x_z5" / "0_Raw_IMS"
    waiting.mkdir(parents=True)
    (waiting / "102_PROJ_01_03_2p5x_z5.ims").write_bytes(b"x")
    text = _run_dry(overnight, root, caplog, monkeypatch)
    assert "  - %s [" % BRAIN in text
    assert "training_data" not in text
    assert "3_Registered_Atlas:" not in text
    assert "102_PROJ_01_03_2p5x_z5: no extracted images yet (run step 2)" in text
