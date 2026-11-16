"""Every lookup of a brain by name says why when the folder rule hides it.

A brain kept in a mouse folder whose name has a skip word (a backup copy of a
mouse), or in a folder starting with '_', is not a brain by the one folder
rule. Steps 4-6 and the utilities that use config.find_brain_folder already
said so when asked for it by name. The pre-filter, steps 2 and 3,
util_optimize_crop.py and util_create_training_cubes.py printed only "not
found" (or nothing), so the operator could not tell a typo from a folder the
rule passes over.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BRAIN = "101_PROJ_01_02_2p5x_z5"


def _load(rel, alias):
    spec = importlib.util.spec_from_file_location(alias, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _brain(folder: Path) -> Path:
    (folder / "0_Raw_IMS").mkdir(parents=True)
    (folder / "0_Raw_IMS" / (folder.name + ".ims")).write_bytes(b"x")
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (folder / sub / "ch0").mkdir(parents=True)
        (folder / sub / "ch0" / "Z0000.tif").write_bytes(b"x")
        (folder / sub / "metadata.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture(params=["mouse_backup", "_set_aside"])
def hidden(request, tmp_path):
    """The only copy of the brain sits in a folder the rule passes over."""
    root = tmp_path / "1_Brains"
    _brain(root / request.param / BRAIN)
    return root, request.param


def _says_why(out, mouse):
    assert BRAIN in out and mouse in out and "not treated as a mouse folder" in out, out


def test_the_prefilter_says_why(hidden, monkeypatch, capsys):
    import mousebrain.prefilter as pf
    root, mouse = hidden
    monkeypatch.setattr(pf, "BRAINS_ROOT", root)
    assert pf.find_brain_path(BRAIN) is None
    _says_why(capsys.readouterr().out, mouse)


def test_util_optimize_crop_says_why(hidden, capsys):
    root, mouse = hidden
    mod = _load("scripts/utils/util_optimize_crop.py", "optimize_crop_why")
    assert mod.find_pipeline(BRAIN, root) == (None, None)
    _says_why(capsys.readouterr().out, mouse)


def test_util_create_training_cubes_says_why(hidden, monkeypatch, capsys):
    root, mouse = hidden
    mod = _load("scripts/utils/util_create_training_cubes.py", "training_cubes_why")
    monkeypatch.setattr(mod, "BRAINS_ROOT", root)
    assert mod.find_brain_path(BRAIN) is None
    _says_why(capsys.readouterr().out, mouse)


@pytest.mark.parametrize("step", ["2_extract_and_analyze.py", "3_register_to_atlas.py"])
def test_steps_2_and_3_say_why(hidden, monkeypatch, capsys, step):
    root, mouse = hidden
    mod = _load("scripts/pipeline/" + step, "step_%s_why" % step[0])
    monkeypatch.setattr(sys, "argv", [step, str(root), "--brain", BRAIN, "--inspect"])
    mod.main()
    _says_why(capsys.readouterr().out, mouse)


def test_a_brain_that_is_found_prints_no_reason(tmp_path, monkeypatch, capsys):
    """The reason is printed only when the lookup fails."""
    import mousebrain.prefilter as pf
    root = tmp_path / "1_Brains"
    _brain(root / "mouse_a" / BRAIN)
    _brain(root / "mouse_backup" / BRAIN)
    monkeypatch.setattr(pf, "BRAINS_ROOT", root)
    assert pf.find_brain_path(BRAIN) == root / "mouse_a" / BRAIN
    assert "not treated" not in capsys.readouterr().out
