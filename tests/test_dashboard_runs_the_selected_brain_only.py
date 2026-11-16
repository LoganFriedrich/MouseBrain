"""The pipeline dashboard's Run button acts on the selected brain, and only on it.

It started extraction with no arguments -- step 2's interactive form, which asks
a question and waits for Enter, while the dashboard captures the output and
gives no way to answer -- and, once past that, extracted every brain waiting.
It started registration with --batch only, which registers every brain still
waiting, not the one selected.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
BRAIN = "101_PROJ_01_02_2p5x_z5"


class _Widget:
    def setText(self, *_):
        pass

    def setEnabled(self, *_):
        pass


def _run_next(step, tmp_path):
    import mousebrain.plugin_3d.pipeline_widget as pw
    started = []
    fake = SimpleNamespace(
        current_brain=tmp_path / "mouse_a" / BRAIN,
        get_brain_status=lambda p: {"step": step, "step_name": step},
        message_label=_Widget(), run_next_btn=_Widget(),
        run_script=lambda script, args: started.append((script, list(args))))
    pw.PipelineDashboard.run_next_step(fake)
    return started


def test_extraction_runs_unattended_on_the_selected_brain(tmp_path):
    assert _run_next("extract", tmp_path) == [
        ("2_extract_and_analyze.py", ["--batch", "--brain", BRAIN])]


def test_registration_runs_on_the_selected_brain(tmp_path):
    assert _run_next("register", tmp_path) == [
        ("3_register_to_atlas.py", ["--batch", "--brain", BRAIN])]


def test_a_new_brain_from_the_wizard_is_extracted_alone(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.pipeline_widget as pw
    new_brain = tmp_path / "mouse_a" / BRAIN

    class Wizard:
        def __init__(self, parent):
            self.start_extraction = True
            self.pipeline_dir = new_brain

        def exec_(self):
            return True

    monkeypatch.setattr(pw, "SetupWizard", Wizard)
    item = SimpleNamespace(data=lambda role: str(new_brain))
    started = []
    fake = SimpleNamespace(
        refresh_brains=lambda: None,
        brain_list=SimpleNamespace(count=lambda: 1, item=lambda i: item,
                                   setCurrentItem=lambda i: None),
        on_brain_selected=lambda i: None,
        run_script=lambda script, args: started.append((script, list(args))))
    pw.PipelineDashboard.add_new_brain(fake)
    assert started == [("2_extract_and_analyze.py", ["--batch", "--brain", BRAIN])]


def _registrable(root, name):
    crop = root / "mouse_a" / name / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "ch0" / "Z0000.tif").write_bytes(b"x")
    (crop / "metadata.json").write_text(json.dumps({"voxel_size_um": {"x": 4, "y": 4, "z": 4}}),
                                        encoding="utf-8")


def test_registration_by_exact_name_does_not_take_a_longer_name_too(tmp_path, monkeypatch):
    """The exact name the dashboard passes is also part of another brain's name
    (brain 1 and brain 101 of the same animal and setup)."""
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    spec = importlib.util.spec_from_file_location("register_exact", PIPELINE / "3_register_to_atlas.py")
    reg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reg
    spec.loader.exec_module(reg)
    root = tmp_path / "1_Brains"
    short = "1_PROJ_01_02_2p5x_z5"
    _registrable(root, short)
    _registrable(root, "10" + short)  # "101_PROJ_..." contains "1_PROJ_..."
    registered = []

    class Stop(Exception):
        pass

    def fake_brainreg(pipeline_folder, *a, **k):
        registered.append(Path(pipeline_folder).name)
        raise Stop
    monkeypatch.setattr(reg, "run_brainreg", fake_brainreg)
    monkeypatch.setattr(sys, "argv", ["3_register_to_atlas.py", str(root), "--batch", "--brain", short])
    try:
        reg.main()
    except SystemExit:
        pass
    assert registered == [short]
