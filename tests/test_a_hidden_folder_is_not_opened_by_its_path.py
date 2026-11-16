"""A folder the brain rule hides is not opened by giving its two-level path.

By the one folder rule a folder starting with '_' and anything inside a backup
or archive mouse folder is not a brain: never listed, never processed, never
found by name. Two lookups still joined whatever name they were given to the
brains root first, so "<mouse>/_superseded_x" or "<backup mouse>/<brain>"
returned the hidden folder with no message -- and the pre-filter then wrote
into an archive, and the training-cube cutter cut cubes from one.
Identifiers are invented.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BRAIN = "101_PROJ_01_02_2p5x_z5"
MOUSE = "101_PROJ_01_02"


def _load(rel, alias):
    spec = importlib.util.spec_from_file_location(alias, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _brain(folder: Path) -> Path:
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (folder / sub / "ch0").mkdir(parents=True)
        (folder / sub / "ch0" / "Z0000.tif").write_bytes(b"x")
        (folder / sub / "metadata.json").write_text(
            '{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}', encoding="utf-8")
    (folder / "4_Cell_Candidates").mkdir()
    (folder / "5_Classified_Cells").mkdir()
    return folder


@pytest.fixture
def root(tmp_path):
    """A real brain, an archive beside its outputs, and a backup of the mouse."""
    root = tmp_path / "1_Brains"
    _brain(root / MOUSE / BRAIN)
    _brain(root / MOUSE / "_superseded_x")
    _brain(root / (MOUSE + "_backup") / BRAIN)
    return root


HIDDEN = [MOUSE + "/_superseded_x", MOUSE + "_backup/" + BRAIN,
          MOUSE + "\\_superseded_x"]


def _lookups(root, monkeypatch):
    import mousebrain.prefilter as pf
    cubes = _load("scripts/utils/util_create_training_cubes.py", "training_cubes_hidden")
    monkeypatch.setattr(pf, "BRAINS_ROOT", root)
    monkeypatch.setattr(cubes, "BRAINS_ROOT", root)
    return {"prefilter": pf.find_brain_path, "cubes": cubes.find_brain_path}


@pytest.mark.parametrize("tool", ["prefilter", "cubes"])
@pytest.mark.parametrize("given", HIDDEN)
def test_a_hidden_folder_given_by_path_is_not_found_and_says_why(
        root, monkeypatch, capsys, tool, given):
    find = _lookups(root, monkeypatch)[tool]
    assert find(given) is None
    out = capsys.readouterr().out
    assert "[i]" in out and ("not a brain" in out or "not treated as a mouse folder" in out), out


@pytest.mark.parametrize("tool", ["prefilter", "cubes"])
@pytest.mark.parametrize("given", [BRAIN, MOUSE + "/" + BRAIN])
def test_the_real_brain_is_found_by_name_and_by_mouse_path(root, monkeypatch, capsys, tool, given):
    find = _lookups(root, monkeypatch)[tool]
    assert find(given) == root / MOUSE / BRAIN
    assert "[i]" not in capsys.readouterr().out


@pytest.mark.parametrize("tool", ["prefilter", "cubes"])
def test_a_brain_directly_in_the_root_is_still_found(tmp_path, monkeypatch, tool):
    """The old one-level layout keeps working; a '_' folder there does not."""
    root = tmp_path / "1_Brains"
    _brain(root / BRAIN)
    _brain(root / "_set_aside")
    find = _lookups(root, monkeypatch)[tool]
    assert find(BRAIN) == root / BRAIN
    assert find("_set_aside") is None
