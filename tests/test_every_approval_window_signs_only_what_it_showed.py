"""Every window that asks about a registration acts only on the one it showed.

Setup & Tuning's Approve already passed the registration it showed. The
pipeline dashboard (Approve button, Run Next Step's approve/reject question,
Overrule), RUN_PIPELINE's approve/reject question, the approval script and
Setup & Tuning's Reject did not: they approved or rejected whatever
registration was in the folder at the moment of the answer. A batch that
registered the brain again while the person was looking at the QC picture had
its new, unseen registration approved -- or turned down -- under that person's
name. Each test below re-registers the brain between "shown" and "answered"
and checks that nothing was recorded. Identifiers are invented.
"""
import builtins
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space

ROOT = Path(__file__).resolve().parents[1]
MOUSE = "M8"
BRAIN = "811_PROJ_08_11_2p5x_z5"
T0 = 1_700_000_000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _register(reg, t, size):
    reg.mkdir(parents=True, exist_ok=True)
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"r" * size)
        os.utime(reg / name, (t, t))


def _brain(root):
    b = root / MOUSE / BRAIN
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / sub / "ch0").mkdir(parents=True)
        (b / sub / "ch0" / "Z0000.tif").write_bytes(b"x")
        os.utime(b / sub / "ch0" / "Z0000.tif", (T0, T0))
        (b / sub / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", b / "2_Cropped_For_Registration")
    _register(reg, T0 + 100, size=10)
    return b, reg


def _registered_again(reg):
    """What a batch finishing a new registration of the brain leaves."""
    _register(reg, T0 + 500, size=20)


def _nothing_recorded(reg):
    assert ra.read_marker(reg) is None, "an approval was written"
    assert ra.read_rejection(reg) is None, "a rejection was written"
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED


# --------------------------------------------------------------- dashboard
class _Any:
    """Stands in for any Qt widget: every method does nothing."""
    def __getattr__(self, name):
        return lambda *a, **k: None


class _Messages:
    Yes, No = 1, 0

    def __init__(self, answer=1):
        self.seen, self.answer = [], answer

    def question(self, *a, **k):
        return self.answer

    def warning(self, parent, title, text, *a, **k):
        self.seen.append(("warning", title, text))

    def information(self, parent, title, text, *a, **k):
        self.seen.append(("information", title, text))


def _dashboard(monkeypatch, b, answers, between=None):
    pw = pytest.importorskip("mousebrain.plugin_3d.pipeline_widget")
    from mousebrain.pipeline_status import brain_status
    msgs = _Messages()
    monkeypatch.setattr(pw, "QMessageBox", msgs)
    queue = list(answers)

    def choose(title, text, choices, escape):
        if between is not None:
            between()          # what happens while the question is open
        return queue.pop(0) if queue else escape
    fake = SimpleNamespace(
        current_brain=b, message_label=_Any(), run_next_btn=_Any(), progress_bar=_Any(),
        status_label=_Any(), next_step_label=_Any(), crop_btn=_Any(), view_qc_btn=_Any(),
        approve_btn=_Any(), view_results_btn=_Any(), check_labels=[_Any() for _ in range(8)],
        get_brain_status=lambda p: brain_status(p, include_prefilter=False),
        view_qc_image=lambda: None, refresh_brains=lambda: None,
        run_script=lambda *a, **k: None, _choose=choose)
    fake.update_status_display = lambda: pw.PipelineDashboard.update_status_display(fake)
    for name in ("_approve_current_registration", "_reject_current_registration",
                 "_overrule_rejection"):
        setattr(fake, name, (lambda n: lambda: getattr(pw.PipelineDashboard, n)(fake))(name))
    return pw, fake, msgs


def test_dashboard_run_next_step_does_not_approve_a_registration_made_while_asking(
        tmp_path, monkeypatch):
    b, reg = _brain(tmp_path)
    pw, fake, msgs = _dashboard(monkeypatch, b, ["approve"], between=lambda: _registered_again(reg))
    pw.PipelineDashboard.run_next_step(fake)
    _nothing_recorded(reg)
    assert any("Registered again" in title for _k, title, _t in msgs.seen)


def test_dashboard_run_next_step_does_not_reject_a_registration_made_while_asking(
        tmp_path, monkeypatch):
    b, reg = _brain(tmp_path)
    pw, fake, _m = _dashboard(monkeypatch, b, ["reject"], between=lambda: _registered_again(reg))
    pw.PipelineDashboard.run_next_step(fake)
    _nothing_recorded(reg)


def test_dashboard_approve_button_signs_only_the_registration_on_display(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path)
    pw, fake, msgs = _dashboard(monkeypatch, b, [])
    fake.update_status_display()      # the person reads the status
    _registered_again(reg)            # a script registers the brain again
    pw.PipelineDashboard.approve_registration(fake)   # "Yes" to "have you reviewed it?"
    _nothing_recorded(reg)
    # Once the new registration has been displayed, it can be approved.
    fake.update_status_display()
    pw.PipelineDashboard.approve_registration(fake)
    assert ra.is_approved(reg)


def test_dashboard_overrule_does_not_approve_a_registration_made_while_asking(
        tmp_path, monkeypatch):
    b, reg = _brain(tmp_path)
    ra.reject(reg, rejected_by="tester")
    answers = ["overrule", "yes"]
    calls = []

    def between():
        calls.append(1)
        if len(calls) == 2:           # while "Overrule the rejection?" is open
            _registered_again(reg)
    pw, fake, _m = _dashboard(monkeypatch, b, answers, between=between)
    pw.PipelineDashboard.run_next_step(fake)
    assert ra.read_marker(reg) is None
    assert ra.approval_state(reg)[0] != ra.APPROVED


# ------------------------------------------------------------- RUN_PIPELINE
def _run_pipeline(monkeypatch, reg, final_answer):
    rp = _load("run_pipeline_shown", "scripts/pipeline/RUN_PIPELINE.py")
    answers = [0, final_answer]   # "Run this step (approve)", then the answer

    def ask(question, *a, **k):
        if "look good" in question:
            _registered_again(reg)    # a batch finishes while the person looks
        return answers.pop(0)
    monkeypatch.setattr(rp, "ask", ask)
    monkeypatch.setattr(rp, "run_script", lambda *a, **k: None)
    monkeypatch.setattr(rp, "clear", lambda: None)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    return rp


@pytest.mark.parametrize("answer", [0, 1], ids=["approve", "reject"])
def test_run_pipeline_answer_is_about_the_registration_shown(tmp_path, monkeypatch, answer):
    b, reg = _brain(tmp_path)
    rp = _run_pipeline(monkeypatch, reg, answer)
    rp.show_brain_status({"name": BRAIN, "full_name": MOUSE + "/" + BRAIN, "path": b,
                          "mouse": MOUSE})
    _nothing_recorded(reg)


# ------------------------------------------------------- approval script
@pytest.mark.parametrize("answer", ["yes", "no"])
def test_the_approval_script_answer_is_about_the_registration_shown(tmp_path, monkeypatch,
                                                                    answer):
    b, reg = _brain(tmp_path)
    util = _load("approve_util_shown", "scripts/utils/util_approve_registration.py")
    monkeypatch.setattr(util, "find_brain", lambda *a, **k: b)

    def view(*a, **k):
        _registered_again(reg)    # registered again while the pictures are open
        return True
    monkeypatch.setattr(util, "view_qc_images", view)
    monkeypatch.setattr(sys, "argv", ["util_approve_registration.py", "--brain", BRAIN])
    monkeypatch.setattr(builtins, "input", lambda *a, **k: answer)
    with pytest.raises(SystemExit):
        util.main()
    _nothing_recorded(reg)


# ---------------------------------------------------------- Setup & Tuning
def test_setup_and_tuning_reject_is_about_the_registration_shown(tmp_path, monkeypatch):
    tw = pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
    b, reg = _brain(tmp_path)
    msgs = _Messages()
    monkeypatch.setattr(tw, "QMessageBox", msgs)

    class _W:
        def __getattr__(self, name):
            fn = getattr(tw.TuningWidget, name)
            if isinstance(fn, types.FunctionType):
                return types.MethodType(fn, self)
            return fn
    w = _W()
    w.current_brain = b
    w.approval_status = _Any()
    w._update_registration_status()     # the person sees registration 1
    _registered_again(reg)
    w.reject_registration()
    _nothing_recorded(reg)
    assert any(kind == "warning" for kind, _t, _x in msgs.seen)
    # The registration now on display can be rejected.
    w.reject_registration()
    assert ra.approval_state(reg)[0] == ra.REJECTED


def test_reject_refuses_another_registration_than_expected(tmp_path):
    b, reg = _brain(tmp_path)
    shown = ra.current_registration_id(reg)
    _registered_again(reg)
    with pytest.raises(ra.RegistrationChanged):
        ra.reject(reg, rejected_by="tester", expected_id=shown)
    _nothing_recorded(reg)
