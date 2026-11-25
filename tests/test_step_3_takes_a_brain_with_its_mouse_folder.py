"""Step 3 takes a brain given as <mouse folder>/<brain folder>.

Steps 4 to 6 print and accept a brain as "<mouse folder>/<brain folder>".
Step 3 accepts that form too: comparing --brain with the brain folder name
alone would fit nothing, a brain waiting to be registered would be answered
with "No pending registration found" and nothing would be registered.
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


def _step3(monkeypatch):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    return _load("scripts/pipeline/3_register_to_atlas.py", "register_by_mouse_path")


@pytest.mark.parametrize("given", [MOUSE + "/" + BRAIN, MOUSE + "\\" + BRAIN])
def test_step_3_registers_the_brain_given_with_its_mouse_folder(tmp_path, monkeypatch, capsys, given):
    reg = _step3(monkeypatch)
    root = tmp_path / "1_Brains"
    _brain(root / MOUSE / BRAIN)
    _brain(root / MOUSE / ("9" + BRAIN))  # its name contains the other's
    registered = []

    class Stop(Exception):
        pass

    def fake_brainreg(pipeline_folder, *a, **k):
        registered.append(Path(pipeline_folder).name)
        raise Stop
    monkeypatch.setattr(reg, "run_brainreg", fake_brainreg)
    monkeypatch.setattr(sys, "argv", ["3_register_to_atlas.py", str(root), "--batch", "--brain", given])
    try:
        reg.main()
    except SystemExit:
        pass
    assert registered == [BRAIN], capsys.readouterr().out


def test_step_3_still_takes_a_part_of_a_name(tmp_path, monkeypatch):
    reg = _step3(monkeypatch)
    pipelines = [(Path("m1") / "101_PROJ_01_02_2p5x_z5", Path("m1"), {}),
                 (Path("m1") / "9101_PROJ_01_02_2p5x_z5", Path("m1"), {}),
                 (Path("m2") / "102_PROJ_01_03_2p5x_z5", Path("m2"), {})]
    assert reg.brain_filter_matches("101_PROJ", pipelines) == [0, 1]
    assert reg.brain_filter_matches("101_PROJ_01_02_2p5x_z5", pipelines) == [0]
    assert reg.brain_filter_matches("m2/102_PROJ_01_03_2p5x_z5", pipelines) == [2]
    assert reg.brain_filter_matches("nothing_like_it", pipelines) == []
