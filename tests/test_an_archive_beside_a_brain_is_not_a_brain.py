"""A folder beside a brain is not automatically a brain.

Every pipeline step scans <brains root>/<mouse>/<brain>, and each step used to
decide for itself what counted -- in practice, any directory. People put things
beside a brain: an archive of a superseded delivery, a scratch copy, a folder
kept while something is checked. Give one of those a 0_Raw_IMS and every step
scans it, finds it complete, and reports the real brain as already done.

Observed 2026-10-13. A brain was re-delivered with a better stitch, its previous
outputs were archived into a folder inside the brain folder, and the next
extraction found the ARCHIVE, announced "all extractions complete", and did
nothing. The two steps queued behind it then failed for reasons that pointed
nowhere near the cause -- which is the expensive part: the symptom was three
steps away from the mistake.
"""
from pathlib import Path

import pytest

from mousebrain.config import is_pipeline_candidate


def make(tmp_path, name):
    folder = tmp_path / name
    (folder / "0_Raw_IMS").mkdir(parents=True)
    (folder / "0_Raw_IMS" / "101_PROJ_01_02_2.5x_z5.ims").write_bytes(b"not really")
    return folder


def test_a_real_brain_folder_is_scanned(tmp_path):
    assert is_pipeline_candidate(make(tmp_path, "101_PROJ_01_02_2p5x_z5"))


@pytest.mark.parametrize("name", [
    "_superseded_2026-10-13_replaced_by_better_stitch",
    "_incoming",
    "_scratch",
    "_old",
])
def test_an_underscore_folder_is_skipped_however_complete_it_looks(tmp_path, name):
    """Complete is exactly the problem: it looks finished, so nothing runs."""
    assert not is_pipeline_candidate(make(tmp_path, name))


def test_a_hidden_folder_is_skipped(tmp_path):
    assert not is_pipeline_candidate(make(tmp_path, ".cache"))


def test_a_file_is_not_a_brain(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello", encoding="utf-8")
    assert not is_pipeline_candidate(f)


def test_something_that_does_not_exist_is_not_a_brain(tmp_path):
    assert not is_pipeline_candidate(tmp_path / "gone")


def test_every_pipeline_step_uses_the_shared_rule():
    """Ten scanners existed; a rule applied in nine places is not a rule.

    This is the test that matters: the defect was not that the rule was wrong,
    it was that each step decided separately and one of them decided wrongly.
    """
    import re

    scripts = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
    offenders = []
    for path in sorted(scripts.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"for pipeline_dir in mouse_dir\.iterdir\(\):\n(.{0,160})",
                                 text, re.S):
            if "is_pipeline_candidate" not in match.group(1):
                line = text[:match.start()].count("\n") + 1
                offenders.append("%s:%d" % (path.name, line))
    assert not offenders, (
        "These scan every directory as a brain, so an archive or scratch folder "
        "beside a brain will be picked up as one:\n  " + "\n  ".join(offenders))
