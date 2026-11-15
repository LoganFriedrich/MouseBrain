"""batch_process starts steps 2 and 3 with arguments they accept, and its QC
checks recognise what those steps actually write.

It passed a --yes that neither step accepts, so argparse stopped step 2 and
every brain failed at its first step. Behind that, the extraction check looked
for TIFFs at the top of 1_Extracted_Full (extraction writes ch0/Z0000.tif ...)
and the crop check for a downsampled.zarr no crop writer makes.
"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _bp():
    spec = importlib.util.spec_from_file_location("batch_process", PIPELINE / "batch_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flags_passed(script):
    text = (PIPELINE / "batch_process.py").read_text(encoding="utf-8")
    calls = re.findall(r"run_step\('%s', \[([^\]]*)\]\)" % re.escape(script), text)
    assert calls, script
    return {f for call in calls for f in re.findall(r"'(--[a-z-]+)'", call)}


@pytest.mark.parametrize("script,needs", [("2_extract_and_analyze.py", None),
                                          ("3_register_to_atlas.py", "brainreg")])
def test_every_flag_batch_process_passes_is_accepted(script, needs):
    if needs:
        pytest.importorskip(needs, reason="%s checks for %s on import" % (script, needs))
    out = subprocess.run([sys.executable, str(PIPELINE / script), "--help"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    for flag in _flags_passed(script):
        assert flag in out.stdout, "%s does not accept %s" % (script, flag)


def _brain(tmp_path):
    b = tmp_path / "101_PROJ_01_02_2p5x_z5"
    for folder in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / folder / "ch0").mkdir(parents=True)
        (b / folder / "ch0" / "Z0000.tif").write_bytes(b"x" * 2048)
        (b / folder / "metadata.json").write_text(json.dumps({}))
    return b


def test_extraction_check_finds_the_channel_folders(tmp_path, monkeypatch):
    bp = _bp()
    monkeypatch.setitem(bp.QC_THRESHOLDS, "min_extracted_size_mb", 0)
    status, msg = bp.qc_check_extraction(_brain(tmp_path))
    assert status == bp.QCStatus.PASS, msg


def test_crop_check_recognises_a_crop(tmp_path):
    bp = _bp()
    status, msg = bp.qc_check_crop(_brain(tmp_path))
    assert status == bp.QCStatus.PASS, msg


def test_an_uncropped_brain_still_stops_for_a_crop(tmp_path):
    bp = _bp()
    b = _brain(tmp_path)
    (b / "2_Cropped_For_Registration" / "metadata.json").unlink()
    assert bp.qc_check_crop(b)[0] == bp.QCStatus.FAIL
