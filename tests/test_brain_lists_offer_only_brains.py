"""Every brain list and brain lookup follows ONE rule for which folders are brains.

The pipeline steps had learned that an archive kept beside a brain (a folder whose
name starts with an underscore) is not a brain. The napari brain lists and the
utilities had not: the dashboard, Setup & Tuning, the crop tool and the 3D cell
viewer offered the archive as a brain, and the utilities could load, crop, draw or
approve it in the real brain's place. The rule now lives in one function,
config.iter_brains, and everything that lists or looks up brains uses it.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousebrain import config

REPO = Path(__file__).resolve().parents[1]
BRAIN = "101_PROJ_01_02_2p5x_z5"
OTHER_SETUP = "101_PROJ_01_02_1p6x_z4"


def _complete(folder: Path) -> Path:
    """A folder with every numbered output a brain list looks for."""
    (folder / "0_Raw_IMS").mkdir(parents=True)
    (folder / "0_Raw_IMS" / (folder.name + ".ims")).write_bytes(b"not really")
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (folder / sub / "ch0").mkdir(parents=True)
        (folder / sub / "ch0" / "Z0000.tif").write_bytes(b"x")
        (folder / sub / "metadata.json").write_text("{}", encoding="utf-8")
    (folder / "3_Registered_Atlas").mkdir()
    (folder / "3_Registered_Atlas" / "brainreg.json").write_text("{}", encoding="utf-8")
    (folder / "6_Region_Analysis").mkdir()
    (folder / "6_Region_Analysis" / "cell_counts_by_region.csv").write_text("a\n1\n", encoding="utf-8")
    (folder / "6_Region_Analysis" / "all_points_information.csv").write_text("a\n1\n", encoding="utf-8")
    return folder


@pytest.fixture
def root(tmp_path):
    """One real brain, an archive of it beside it, and a whole-mouse backup."""
    root = tmp_path / "1_Brains"
    _complete(root / "mouse_a" / BRAIN)
    _complete(root / "mouse_a" / "_superseded_replaced_by_better_stitch")
    _complete(root / "mouse_a_backup" / BRAIN)
    _complete(root / "_incoming" / "loose_copy")
    return root


def _names(pairs):
    return [b.name for _m, b in pairs]


# ---------------------------------------------------------------- the rule itself
# Checked through entry points that existed before the shared rule, so each
# test shows the behaviour that changed, not only that a new function exists.

def _run_pipeline(monkeypatch, root):
    spec = importlib.util.spec_from_file_location(
        "run_pipeline_under_test", REPO / "scripts" / "pipeline" / "RUN_PIPELINE.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BRAINS_ROOT", root)
    return mod


def test_only_the_real_brain_is_a_brain(root, monkeypatch):
    """RUN_PIPELINE's list: not the archive beside the brain, not the
    whole-mouse backup, not a loose copy in an underscore folder."""
    found = _run_pipeline(monkeypatch, root).find_brains()
    assert [b["full_name"] for b in found] == ["mouse_a/" + BRAIN]


def test_a_root_that_does_not_exist_lists_nothing(tmp_path):
    """A lookup in a brains root that is not there says 'not found'; it used
    to stop the utility with a FileNotFoundError."""
    assert _script("util_registration_qc.py").find_brain(BRAIN, tmp_path / "gone") is None


def test_an_exact_name_wins_and_an_ambiguous_part_is_refused(tmp_path):
    """A brain whose name is part of another brain's name (a rescan kept as
    its own brain, listed first) is found by its exact name, not the other."""
    root = tmp_path / "1_Brains"
    _complete(root / "mouse_0" / (BRAIN + "_rescan"))
    _complete(root / "mouse_a" / BRAIN)
    _complete(root / "mouse_a" / OTHER_SETUP)
    util = _script("util_registration_qc.py")
    assert util.find_brain(BRAIN, root) == root / "mouse_a" / BRAIN
    assert util.find_brain("mouse_a/" + OTHER_SETUP, root) == root / "mouse_a" / OTHER_SETUP
    assert util.find_brain("101_PROJ_01_02", root) is None
    assert util.find_brain("1p6x", root) == root / "mouse_a" / OTHER_SETUP


# ---------------------------------------------------------------- the napari lists

class _Combo:
    def __init__(self):
        self.items = []

    def clear(self):
        self.items = []

    def addItem(self, *args):
        self.items.append(args)

    def blockSignals(self, *_):
        return False


class _List(_Combo):
    def addItem(self, item):
        from qtpy.QtCore import Qt
        self.items.append(item.data(Qt.UserRole))


def _offered(items):
    """Folder names of every entry carrying a path."""
    out = []
    for entry in items:
        path = entry if isinstance(entry, str) else (entry[1] if len(entry) > 1 else None)
        if path:
            out.append(Path(str(path)).name)
    return out


def test_the_pipeline_dashboard_lists_only_brains(root, monkeypatch):
    import mousebrain.plugin_3d.pipeline_widget as pw
    from mousebrain.pipeline_status import brain_status
    monkeypatch.setattr(pw, "BRAINS_ROOT", root)
    fake = SimpleNamespace(brain_list=_List(),
                           get_brain_status=lambda p: brain_status(p, include_prefilter=False))
    pw.PipelineDashboard.refresh_brains(fake)
    assert [Path(p).name for p in fake.brain_list.items] == [BRAIN]


def test_setup_and_tuning_lists_only_brains(root, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    monkeypatch.setattr(tw, "BRAINS_ROOT", root)
    fake = SimpleNamespace(brain_combo=_Combo())
    tw.TuningWidget.refresh_brains(fake)
    assert _offered(fake.brain_combo.items) == [BRAIN]


def test_the_cross_brain_comparison_lists_only_brains(root, monkeypatch):
    import mousebrain.config as cfg
    import mousebrain.plugin_3d.tuning_widget as tw
    monkeypatch.setattr(tw, "BRAINS_ROOT", root, raising=False)
    monkeypatch.setattr(cfg, "BRAINS_ROOT", root)
    fake = SimpleNamespace(cross_brain_list=_List())
    tw.TuningWidget._load_available_brains_for_comparison(fake)
    assert [Path(str(p)).name for p in fake.cross_brain_list.items] == [BRAIN]


def test_the_3d_cell_viewer_lists_only_brains(root, monkeypatch):
    import mousebrain.plugin_3d.cell_viewer_3d as cv
    monkeypatch.setattr(cv, "BRAINS_ROOT", root)
    fake = SimpleNamespace(brain_combo=_Combo())
    cv.Cell3DViewer.refresh_brains(fake)
    assert _offered(fake.brain_combo.items) == [BRAIN]


def test_the_crop_tool_lists_only_brains(root, monkeypatch):
    import mousebrain.plugin_3d.manual_crop_widget as mc
    monkeypatch.setattr(mc, "DEFAULT_BRAINGLOBE_ROOT", root)
    fake = SimpleNamespace(brain_combo=_Combo())
    mc.ManualCropWidget._populate_brain_list(fake)
    assert _offered(fake.brain_combo.items) == [BRAIN]


# ---------------------------------------------------------------- lookups by name

def _script(name):
    path = REPO / "scripts" / "utils" / name
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def shadowed(tmp_path):
    """The brain, plus a whole-mouse backup copy of it that lists first."""
    root = tmp_path / "1_Brains"
    _complete(root / "a_backup_of_mouse_a" / BRAIN)
    _complete(root / "mouse_a" / BRAIN)
    return root


@pytest.mark.parametrize("script", ["util_registration_qc.py", "util_manual_crop.py",
                                    "util_approve_registration.py"])
def test_a_utility_finds_the_brain_not_a_copy_kept_aside(shadowed, script):
    mod = _script(script)
    assert mod.find_brain(BRAIN, shadowed) == shadowed / "mouse_a" / BRAIN


@pytest.mark.parametrize("script", ["util_registration_qc.py", "util_manual_crop.py"])
def test_a_utility_refuses_a_name_that_fits_two_brains(tmp_path, script):
    root = tmp_path / "1_Brains"
    _complete(root / "mouse_a" / BRAIN)
    _complete(root / "mouse_a" / OTHER_SETUP)
    assert _script(script).find_brain("101_PROJ_01_02", root) is None


def test_the_prefilter_finds_the_brain_not_a_copy_kept_aside(shadowed, monkeypatch):
    import mousebrain.prefilter as pf
    monkeypatch.setattr(pf, "BRAINS_ROOT", shadowed)
    assert pf.find_brain_path(BRAIN) == shadowed / "mouse_a" / BRAIN


def test_the_orientation_check_finds_the_brain_not_a_copy_kept_aside(shadowed, monkeypatch):
    import mousebrain.orientation_check as oc
    monkeypatch.setattr(config, "BRAINS_ROOT", shadowed)
    assert oc.find_brain(BRAIN) == shadowed / "mouse_a" / BRAIN


def test_nothing_lists_brains_by_hand():
    """A rule applied in most places is not a rule: no code walks <root>/<mouse>/<brain>
    itself. The exceptions are named, with the reason they are not brain lists."""
    import re
    allowed = {
        # config.iter_brains is the rule.
        ("src/mousebrain/config.py", "root.iterdir"),
        ("src/mousebrain/config.py", "mouse_dir.iterdir"),
        # Training-data pools: these collect training cubes wherever they are
        # kept and are not lists of brains to process.
        ("src/mousebrain/plugin_3d/tuning_widget.py", "mouse_dir.iterdir"),
        ("src/mousebrain/plugin_3d/tuning_widget.py", "BRAINS_ROOT.iterdir"),
        ("scripts/utils/util_train_model.py", "mouse_dir.iterdir"),
        ("scripts/utils/util_train_model.py", "BRAINS_ROOT.iterdir"),
        # Step 1 also looks for .ims files lying loose in a mouse folder; it
        # applies is_mouse_folder and is_pipeline_candidate itself.
        ("scripts/pipeline/1_organize_pipeline.py", "root_path.iterdir"),
    }
    pattern = re.compile(r"\b(BRAINS_ROOT|root|root_path|mouse_dir|mouse_folder|scan_path)"
                         r"\.(iterdir|r?glob)\(")
    # A glob is flagged when its receiver is the brains root itself (or a step's
    # scan_path): a glob there such as "*/*<name>*" is a brain lookup by hand.
    # Globs on a mouse folder or a step's own root look for files (.ims), not brains.
    glob_flagged = {"BRAINS_ROOT", "scan_path"}
    offenders = []
    for base in ("src/mousebrain", "scripts"):
        for path in sorted((REPO / base).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if "/plugin_2d/" in rel:
                continue  # 2D slices are not organised as mouse/brain folders
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                m = pattern.search(line)
                if not m or m.group(2).endswith("glob") and m.group(1) not in glob_flagged:
                    continue
                if (rel, "%s.%s" % (m.group(1), m.group(2))) in allowed:
                    continue
                offenders.append("%s:%d: %s" % (rel, n, line.strip()))
    assert not offenders, "\n".join(offenders)


# ---------------------------------------------------------------- the mousebrain launcher
# `mousebrain --crop <brain>` used to find the brain with its own glob for
# "*/*<name>*" and open the first match the folder listing gave.

def _launch_crop(monkeypatch, capsys, root, name):
    """Run the launcher with napari replaced by a stand-in whose Viewer stops
    the launch, so the test sees which brain it chose without opening a window."""
    import sys
    import mousebrain
    from mousebrain import cli
    opened = []

    def _viewer(*_a, **_k):
        opened.append(True)
        raise RuntimeError("stand-in viewer: stop here")

    monkeypatch.setitem(sys.modules, "napari", SimpleNamespace(Viewer=_viewer, run=lambda: None))
    monkeypatch.setattr(mousebrain, "BRAINS_ROOT", root)
    monkeypatch.setattr(config, "BRAINS_ROOT", root)
    rc = cli.launch_manual_crop(name)
    return rc, opened, capsys.readouterr().out


def test_the_launcher_opens_the_brain_not_a_backup_copy(shadowed, monkeypatch, capsys):
    _rc, opened, out = _launch_crop(monkeypatch, capsys, shadowed, BRAIN)
    assert opened, out
    assert "Path: %s" % (shadowed / "mouse_a" / BRAIN) in out
    assert "a_backup_of_mouse_a" not in out


def test_the_launcher_refuses_a_name_that_fits_two_brains(tmp_path, monkeypatch, capsys):
    root = tmp_path / "1_Brains"
    _complete(root / "mouse_a" / BRAIN)
    _complete(root / "mouse_a" / OTHER_SETUP)
    rc, opened, out = _launch_crop(monkeypatch, capsys, root, "101_PROJ_01_02")
    assert rc == 1 and not opened
    assert BRAIN in out and OTHER_SETUP in out


def test_the_launcher_lists_only_brains_when_a_name_is_not_found(root, monkeypatch, capsys):
    rc, opened, out = _launch_crop(monkeypatch, capsys, root, "no_such_brain")
    assert rc == 1 and not opened
    listed = [line.strip()[2:] for line in out.split("Available brains:")[1].splitlines()
              if line.strip().startswith("- ")]
    assert listed == [BRAIN]
