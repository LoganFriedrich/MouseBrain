"""A brain named by its folder path parses like the brain's own name.

WHY (2026-10-22): the calibration-run tracker logs some runs with the brain as
"<mouse>/<brain>". Splitting that on '_' across the slash made the subject
"CNT_01_02/349" and the imaging paradigm "CNT_01_02_1p625x_z4" -- a paradigm
per brain -- so "the best settings for this kind of imaging" never matched a
run on another brain. Identifiers here are synthetic.
"""
import pytest

from mousebrain.config import parse_brain_name

PLAIN = "349_PRJ_01_02_1p625x_z4"


@pytest.mark.parametrize("given", [
    "349_PRJ_01_02/349_PRJ_01_02_1p625x_z4",
    "349_PRJ_01_02\\349_PRJ_01_02_1p625x_z4",
    "some/root/349_PRJ_01_02/349_PRJ_01_02_1p625x_z4/",
])
def test_a_path_parses_as_its_last_folder(given):
    got = parse_brain_name(given)
    want = parse_brain_name(PLAIN)
    for key in ("brain_id", "project_code", "cohort", "subject", "subject_full",
                "cohort_full", "imaging_params"):
        assert got[key] == want[key], key
    assert got["imaging_params"] == "1p625x_z4"
    assert got["subject_full"] == "PRJ_01_02"
    assert got["raw"] == given          # the text as given is kept


def test_a_plain_name_is_unchanged():
    got = parse_brain_name(PLAIN)
    assert (got["brain_id"], got["subject_full"], got["imaging_params"]) == (
        "349", "PRJ_01_02", "1p625x_z4")
