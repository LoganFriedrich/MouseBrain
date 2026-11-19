"""The overnight batch's report says why step 5 or step 6 stopped.

Step 6 prints why it refused a brain (its registration is not approved) or
could not publish a count as the LAST lines of its normal output, with
nothing on its error output. The overnight batch kept only the start of the
error output, so its report for such a failure was blank.

Identifiers are invented.
"""
import importlib.util
import subprocess
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_overnight_report_says_why_step_6_refused(tmp_path, monkeypatch):
    ob = _load("overnight_failure_text", PIPELINE / "overnight_batch.py")
    said = "Counting brain_x\n" + "x" * 900 + "\n[FAILED] Not counted: its registration is not approved\n"

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout=said, stderr="")
    monkeypatch.setattr(ob.subprocess, "run", run)

    ok, msg = ob.run_counting("brain_x", tmp_path)
    assert not ok
    assert "Not counted: its registration is not approved" in msg
    ok, msg = ob.run_classification("brain_x", tmp_path)
    assert not ok
    assert "Not counted: its registration is not approved" in msg
