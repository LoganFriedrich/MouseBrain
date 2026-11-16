"""In the pipeline dashboard only an explicit Reject rejects a registration.

The approval question was Yes/No, and anything but Yes recorded a rejection:
closing the dialog (Esc, the close button) to look at the QC picture wrote
.registration_rejected under the person's name and sent the brain back to
"Register". Now there are three buttons -- Approve, Reject, Not yet -- and
closing is Not yet. A rejected brain's Run button asks before registering it
again (--reprocess) and can overrule the rejection. Identifiers are invented.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space
from mousebrain.pipeline_status import brain_status

MOUSE = "M9"
BRAIN = "906_PROJ_09_06_2p5x_z5"
T0 = 1_700_000_000


class _Widget:
    def setText(self, *_):
        pass

    def setEnabled(self, *_):
        pass


def _brain(root):
    b = root / MOUSE / BRAIN
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / sub / "ch0").mkdir(parents=True)
        (b / sub / "ch0" / "Z0000.tif").write_bytes(b"x")
        os.utime(b / sub / "ch0" / "Z0000.tif", (T0, T0))
        (b / sub / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    crop = b / "2_Cropped_For_Registration"
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", crop)
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"r")
        os.utime(reg / name, (T0 + 100, T0 + 100))
    return b


def _dashboard(b, answers):
    pw = pytest.importorskip("mousebrain.plugin_3d.pipeline_widget")
    started, asked = [], []
    queue = list(answers)

    def choose(title, text, choices, escape):
        asked.append(title)
        return queue.pop(0) if queue else escape  # nothing left: the dialog was closed
    fake = SimpleNamespace(
        current_brain=b, message_label=_Widget(), run_next_btn=_Widget(),
        get_brain_status=brain_status, view_qc_image=lambda: None,
        refresh_brains=lambda: None, update_status_display=lambda: None,
        run_script=lambda script, args: started.append((script, list(args))),
        _choose=choose)
    for name in ("_approve_current_registration", "_reject_current_registration",
                 "_overrule_rejection"):
        setattr(fake, name, (lambda n: lambda: getattr(pw.PipelineDashboard, n)(fake))(name))
    return pw, fake, started, asked


def test_closing_the_approval_question_records_nothing(tmp_path):
    b = _brain(tmp_path)
    pw, fake, started, asked = _dashboard(b, [])  # closed (Esc / close button)
    pw.PipelineDashboard.run_next_step(fake)
    reg = b / "3_Registered_Atlas"
    assert asked == ["Approve Registration"]
    assert not (reg / ra.REJECTION_MARKER).exists()
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED


def test_reject_records_a_rejection_and_approve_an_approval(tmp_path):
    b = _brain(tmp_path)
    pw, fake, _s, _a = _dashboard(b, ["reject"])
    pw.PipelineDashboard.run_next_step(fake)
    assert ra.approval_state(b / "3_Registered_Atlas")[0] == ra.REJECTED

    b2 = _brain(tmp_path / "second")
    pw, fake, _s, _a = _dashboard(b2, ["approve"])
    pw.PipelineDashboard.run_next_step(fake)
    assert ra.is_approved(b2 / "3_Registered_Atlas")


def test_a_rejected_brain_is_registered_again_only_when_asked(tmp_path):
    b = _brain(tmp_path)
    ra.reject(b / "3_Registered_Atlas", rejected_by="tester")
    pw, fake, started, _a = _dashboard(b, [])  # closed: nothing happens
    pw.PipelineDashboard.run_next_step(fake)
    assert started == []
    pw, fake, started, _a = _dashboard(b, ["register"])
    pw.PipelineDashboard.run_next_step(fake)
    assert started == [("3_register_to_atlas.py", ["--batch", "--brain", BRAIN, "--reprocess"])]


def test_a_rejection_can_be_overruled_in_the_dashboard(tmp_path):
    b = _brain(tmp_path)
    reg = b / "3_Registered_Atlas"
    ra.reject(reg, rejected_by="tester")
    pw, fake, _s, _a = _dashboard(b, ["overrule"])  # second question closed: nothing
    pw.PipelineDashboard.run_next_step(fake)
    assert ra.approval_state(reg)[0] == ra.REJECTED
    pw, fake, started, _a = _dashboard(b, ["overrule", "yes"])
    pw.PipelineDashboard.run_next_step(fake)
    assert started == [] and ra.is_approved(reg)
    assert list((reg / "_archive").glob("*/" + ra.REJECTION_MARKER))
