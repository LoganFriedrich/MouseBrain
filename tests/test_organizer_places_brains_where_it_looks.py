"""A brain must be placed where this script's own scanner then looks for it.

The bug: a newly delivered .ims dropped into the brains root -- the normal way a brain
arrives -- was placed FLAT, as <brains root>/403_CNT_05_03_1p625x_z4/, because the code
assumed its parent directory was already the mouse folder. The scanner looks for
mouse/pipeline/0_Raw_IMS, so the brain then vanished from the listing entirely: not
"ready", not "needs work, just absent. A 50 GB delivery sat in the pipeline invisible
to every later step (2026-10-07), and the flat name also collides with the
training_data folders, which use that shape already.

PANOs are a separate case: preliminary ~8 um overview scans that the pipeline does not
use. Refusing them is correct; what was wrong was telling the operator to rename one
into the production form, which would have pulled an 8 um volume into a pipeline
calibrated at 4 um.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "1_organize_pipeline.py"


@pytest.fixture(scope="module")
def organizer():
    spec = importlib.util.spec_from_file_location("organize_pipeline_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _drop(root, name, body=b"not really an ims"):
    root.mkdir(parents=True, exist_ok=True)
    f = root / name
    f.write_bytes(body)
    return f


def test_a_newly_delivered_brain_is_nested_under_its_mouse_folder(organizer, tmp_path):
    f = _drop(tmp_path, "403_CNT_05_03_1.625x_z4.ims")
    ok, msg, _ = organizer.organize_ims_file(f)
    assert ok, msg
    expected = (tmp_path / "403_CNT_05_03" / "403_CNT_05_03_1p625x_z4"
                / "0_Raw_IMS" / "403_CNT_05_03_1.625x_z4.ims")
    assert expected.is_file(), "placed at %s instead" % list(tmp_path.rglob("*.ims"))
    assert not (tmp_path / "403_CNT_05_03_1p625x_z4").exists(), (
        "a flat pipeline folder in the brains root is the bug: the scanner cannot see "
        "it, and the name already means training_data")


def test_the_scanner_then_finds_what_was_just_placed(organizer, tmp_path):
    """The actual regression: placing and finding must agree."""
    f = _drop(tmp_path, "403_CNT_05_03_1.625x_z4.ims")
    organizer.organize_ims_file(f)
    results = organizer.scan_and_report(tmp_path)
    found = [p for p, *_ in results["organized"]] + [p for p, *_ in results.get("needs_work", [])]
    assert any("403_CNT_05_03" in str(p) for p in found), (
        "the brain disappeared from the scan after being organized: %r" % results)


def test_a_brain_already_in_place_is_left_alone(organizer, tmp_path):
    raw = tmp_path / "403_CNT_05_03" / "403_CNT_05_03_1p625x_z4" / "0_Raw_IMS"
    f = _drop(raw, "403_CNT_05_03_1.625x_z4.ims")
    ok, msg, _ = organizer.organize_ims_file(f)
    assert ok and "already" in msg.lower(), msg
    assert f.is_file()


def test_a_pano_is_refused_without_advising_a_rename(organizer):
    ok, reason = organizer.validate_filename("404_CNT_05_05_PANO.ims")
    assert ok is False
    assert "PANO" in reason
    assert "rename" in reason.lower(), "it should say explicitly NOT to rename it"
    assert "do not rename" in reason.lower()


def test_a_production_name_still_validates(organizer):
    ok, _ = organizer.validate_filename("403_CNT_05_03_1.625x_z4.ims")
    assert ok is True


def test_console_output_is_ascii_only():
    """Windows consoles are cp1252; one non-ASCII character crashes the whole script
    the moment output is piped or logged rather than shown in a terminal."""
    src = SCRIPT.read_text(encoding="utf-8")
    bad = sorted({c for c in src if ord(c) > 127})
    assert not bad, "non-ASCII in a script that prints to a Windows console: %r" % bad
    src.encode("cp1252")


def test_it_can_run_without_a_person_at_a_console():
    """Both prompts must be conditional, or the script cannot be piped, logged or
    scheduled -- it dies with EOFError after doing its work."""
    src = SCRIPT.read_text(encoding="utf-8")
    for call in [l for l in src.splitlines() if "input(" in l and "def " not in l]:
        assert "isatty" in src, "input() must be guarded by a tty check"
    assert "--yes" in src and "isatty" in src
