"""A brains folder that cannot be read is reported, never passed over quietly.

The shared brain listing (config.iter_brains) treated a brains root that is
missing or cannot be listed as a root with no brains in it, and a mouse folder
that cannot be listed as a mouse with no brains. Nothing was printed. With the
data drive not connected the overnight batch logged "Found 0 brain(s)" and
ended normally, a step asked for a real brain by name answered "Brain not
found", and the brains of an unreadable mouse folder dropped out of every list
without a line anywhere. A brain that silently never runs looks exactly like
one that is not there.
"""
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
BRAIN = "101_PROJ_01_02_2p5x_z5"
OTHER = "102_PROJ_01_03_2p5x_z5"


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


@pytest.fixture
def two_mice(tmp_path, monkeypatch):
    """Two mouse folders with a cropped brain each; the second cannot be listed."""
    root = tmp_path / "1_Brains"
    _images(root / "mouse_a" / BRAIN / "2_Cropped_For_Registration")
    _images(root / "mouse_b" / OTHER / "2_Cropped_For_Registration")
    locked = root / "mouse_b"
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self == locked:
            raise PermissionError(13, "Access is denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    return root


# ---------------------------------------------------------------- the listing

def test_a_missing_brains_root_is_reported(tmp_path, capsys):
    from mousebrain.config import brains_root_problem, iter_brains
    root = tmp_path / "not_connected" / "1_Brains"
    problems = []
    assert list(iter_brains(root, problems=problems)) == []
    out = capsys.readouterr().out
    assert "[!] Could not read the brains folder" in out and str(root) in out
    assert [p for p, _why in problems] == [root]
    assert str(root) in brains_root_problem(root)


def test_a_readable_brains_root_has_no_problem(tmp_path, capsys):
    from mousebrain.config import brains_root_problem, iter_brains
    root = tmp_path / "1_Brains"
    _images(root / "mouse_a" / BRAIN / "2_Cropped_For_Registration")
    assert brains_root_problem(root) == ""
    assert [b.name for _m, b in iter_brains(root)] == [BRAIN]
    assert "[!]" not in capsys.readouterr().out


def test_a_mouse_folder_that_cannot_be_listed_is_reported(two_mice, capsys):
    """Its brains are missing from the list, the other mouse's are not, and
    one line says so -- also in the quiet listings the napari lists use."""
    from mousebrain.config import iter_brains
    problems = []
    assert [b.name for _m, b in iter_brains(two_mice, problems=problems)] == [BRAIN]
    out = capsys.readouterr().out
    assert "[!] Could not read the mouse folder" in out and "mouse_b" in out
    assert "Access is denied" in out
    assert [p.name for p, _why in problems] == ["mouse_b"]


def test_looking_a_brain_up_by_name_reports_a_missing_root(tmp_path, capsys):
    from mousebrain.config import find_brain_folder
    root = tmp_path / "not_connected" / "1_Brains"
    assert find_brain_folder(BRAIN, root) == (None, [])
    assert "Could not read the brains folder" in capsys.readouterr().out


# ---------------------------------------------------------------- steps 4 to 6

@pytest.mark.parametrize("script", ["4_detect_cells.py", "5_classify_cells.py",
                                    "6_count_regions.py"])
def test_a_step_does_not_call_a_brain_not_found_when_the_root_is_missing(tmp_path, script):
    """--brain stops and names the brains folder, instead of reporting a
    real brain as not found."""
    mod = _load(script, script.replace(".py", "_unreadable_root").lstrip("0123456789_"))
    root = tmp_path / "not_connected" / "1_Brains"
    with pytest.raises(SystemExit) as stopped:
        mod.find_pipeline(BRAIN, root)
    message = str(stopped.value.code)
    assert "Could not read the brains folder" in message and str(root) in message


# ---------------------------------------------------------------- overnight batch

@pytest.fixture
def overnight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the module opens its log file in the working folder
    mod = _load("overnight_batch.py", "overnight_unreadable_root")
    yield mod
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler) and str(tmp_path) in h.baseFilename:
            logging.root.removeHandler(h)
            h.close()


def _run_dry(mod, root, caplog, monkeypatch):
    monkeypatch.setattr(mod, "BRAINS_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["overnight_batch.py", "--dry-run"])
    with caplog.at_level(logging.INFO):
        code = mod.main()
    return code, caplog.text


def test_overnight_fails_when_the_brains_folder_is_missing(overnight, tmp_path, caplog,
                                                           monkeypatch):
    root = tmp_path / "not_connected" / "1_Brains"
    code, text = _run_dry(overnight, root, caplog, monkeypatch)
    assert code == 1
    assert "Could not read the brains folder" in text and str(root) in text
    assert "Nothing was processed" in text
    assert "Found 0 brain(s)" not in text


def test_overnight_logs_a_mouse_folder_it_could_not_read(overnight, two_mice, caplog,
                                                         monkeypatch):
    code, text = _run_dry(overnight, two_mice, caplog, monkeypatch)
    assert "  - %s [" % BRAIN in text
    assert "mouse_b" in text and "were NOT listed" in text
