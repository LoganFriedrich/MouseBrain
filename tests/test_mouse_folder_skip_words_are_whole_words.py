"""Which mouse folders the skip words pass over, and that each one says so.

The skip words (script, backup, archive, summary, old) keep whole-mouse backup
and archive copies out of every brain list. The four long words are matched
anywhere in the name, as every step always matched them. Matched as letters
anywhere in the name, "old" would also drop a mouse folder named for a project such as "GOLD"
or "HOLD" from every step, list and lookup, with no message. And a brain kept
in a passed-over folder was missing from lists, or "not found" by name, with no
reason given.
"""
import importlib.util
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


def _brain(root: Path, mouse: str) -> Path:
    folder = root / mouse / BRAIN
    (folder / "0_Raw_IMS").mkdir(parents=True)
    (folder / "0_Raw_IMS" / (BRAIN + ".ims")).write_bytes(b"x")
    return folder


@pytest.mark.parametrize("mouse", ["GOLD_01", "Threshold_Test", "HOLDOUT_A", "scaffold-3"])
@pytest.mark.parametrize("step", ["4_detect_cells.py", "5_classify_cells.py", "6_count_regions.py"])
def test_a_mouse_folder_that_only_contains_the_letters_is_a_mouse(tmp_path, mouse, step):
    root = tmp_path / "1_Brains"
    brain = _brain(root, mouse)
    mod = _load(step, "step_" + step.split("_")[0] + "_words")
    if step.startswith("4_"):
        brain.joinpath("1_Extracted_Full", "ch0").mkdir(parents=True)
        brain.joinpath("1_Extracted_Full", "ch0", "Z0000.tif").write_bytes(b"x")
        brain.joinpath("1_Extracted_Full", "metadata.json").write_text("{}", encoding="utf-8")
    assert mod.find_pipeline(BRAIN, root)[0] == brain


@pytest.mark.parametrize("mouse", ["Mouse_Backup", "MouseBackup", "old_deliveries",
                                   "Archive2024", "scripts", "mouse_a archived", "Summaries"])
def test_a_skip_word_in_a_mouse_folder_name_passes_it_over_and_says_why(tmp_path, capsys, mouse):
    """Looked up by name in step 5, a brain in a backup copy of a mouse is not
    used, and the message names the folder and the word."""
    root = tmp_path / "1_Brains"
    _brain(root, mouse)
    mod = _load("5_classify_cells.py", "step_5_words_skip")
    assert mod.find_pipeline(BRAIN, root) == (None, None)
    out = capsys.readouterr().out
    assert mouse in out and "not treated as a mouse folder" in out


def test_a_listing_names_each_mouse_folder_it_passes_over(tmp_path, capsys, monkeypatch):
    root = tmp_path / "1_Brains"
    _brain(root, "mouse_a")
    _brain(root, "mouse_a_backup")
    _brain(root, "_kept_aside")
    rp = _load("RUN_PIPELINE.py", "run_pipeline_words")
    monkeypatch.setattr(rp, "BRAINS_ROOT", root)
    found = rp.find_brains()
    assert [b["full_name"] for b in found] == ["mouse_a/" + BRAIN]
    out = capsys.readouterr().out
    assert "Not treated as a mouse folder: mouse_a_backup" in out and "'backup'" in out
    assert "_kept_aside" not in out  # an underscore is the documented quiet way


# ---------------------------------------------------------------- the long words
# script, backup, archive and summary were matched ANYWHERE in the name by
# every step that had them. Only "old" needed to become a whole word; matching
# the long words as whole words too would make a backup copy of a whole mouse
# named, say, "brainbackup" a mouse again, processed by every step.

GLUED = ["brainbackup", "mybackups", "archivedmice", "mousescripts", "cohortsummary"]


def _full_brain(root: Path, mouse: str) -> Path:
    folder = _brain(root, mouse)
    folder.joinpath("1_Extracted_Full", "ch0").mkdir(parents=True)
    folder.joinpath("1_Extracted_Full", "ch0", "Z0000.tif").write_bytes(b"x")
    folder.joinpath("1_Extracted_Full", "metadata.json").write_text(
        '{"voxel_size_um": {"x": 4.0, "y": 4.0, "z": 4.0}, "channels": {"signal_channel": 0}}',
        encoding="utf-8")
    return folder


@pytest.mark.parametrize("mouse", GLUED)
def test_a_long_skip_word_glued_into_a_name_still_passes_the_folder_over(tmp_path, mouse):
    root = tmp_path / "1_Brains"
    real = _full_brain(root, "mouse_a")
    _full_brain(root, mouse)

    organizer = _load("1_organize_pipeline.py", "step_1_glued")
    assert [p.parents[2].name for p in organizer.find_ims_files(root)] == ["mouse_a"]

    extract = _load("2_extract_and_analyze.py", "step_2_glued")
    assert [m.name for _ims, _p, m in extract.find_organized_pipelines(root)] == ["mouse_a"]

    register = _load("3_register_to_atlas.py", "step_3_glued")
    assert [m.name for _p, m, _meta in register.find_extracted_pipelines(root)] == ["mouse_a"]

    for step in ("4_detect_cells.py", "5_classify_cells.py", "6_count_regions.py"):
        mod = _load(step, "step_" + step.split("_")[0] + "_glued")
        assert mod.find_pipeline(BRAIN, root)[0] == real, step
