"""Step 1's merge of an old short-name folder deletes nothing it did not move.

Older layouts named a brain's folder by its imaging setup only (``2p5x_z5``).
When both that folder and the full-name pipeline folder exist, step 1 merges
the old one in. It must remove the old folder only when nothing is left in
it: the full-name folder's numbered subfolders exist (empty) before the merge,
so the old 0_Raw_IMS -- with the brain's raw file in it -- is not moved, and
deleting the folder whatever was left would delete the raw file.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "1_organize_pipeline.py"
BRAIN = "101_PROJ_01_02_2p5x_z5"
SHORT = "2p5x_z5"


@pytest.fixture(scope="module")
def organizer():
    spec = importlib.util.spec_from_file_location("organize_merge_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mouse(tmp_path, organizer):
    mouse = tmp_path / "101_PROJ_01_02"
    organizer.create_pipeline_structure(mouse, BRAIN)  # numbered folders, all empty
    old = mouse / SHORT
    (old / "0_Raw_IMS").mkdir(parents=True)
    (old / "0_Raw_IMS" / "101_PROJ_01_02_2.5x_z5.ims").write_bytes(b"the only copy")
    return mouse, old


def test_the_raw_file_survives_the_merge(organizer, tmp_path):
    mouse, old = _mouse(tmp_path, organizer)
    organizer.migrate_old_folders(mouse, BRAIN)
    ims = mouse / BRAIN / "0_Raw_IMS" / "101_PROJ_01_02_2.5x_z5.ims"
    assert ims.is_file(), "the raw file was deleted"
    assert ims.read_bytes() == b"the only copy"
    assert not old.exists(), "fully merged, so the old folder goes"


def test_what_cannot_be_merged_is_kept_and_named(organizer, tmp_path):
    mouse, old = _mouse(tmp_path, organizer)
    (old / "notes.txt").write_text("old notes", encoding="utf-8")
    (mouse / BRAIN / "notes.txt").write_text("new notes", encoding="utf-8")
    actions = organizer.migrate_old_folders(mouse, BRAIN)
    assert (old / "notes.txt").read_text(encoding="utf-8") == "old notes"
    assert (mouse / BRAIN / "notes.txt").read_text(encoding="utf-8") == "new notes"
    assert any("notes.txt" in a and "Kept" in a for a in actions), actions
    # the part that could be merged was
    assert (mouse / BRAIN / "0_Raw_IMS" / "101_PROJ_01_02_2.5x_z5.ims").is_file()
