"""Approve in Setup & Tuning signs the registration the person was shown.

The status label on the Registration tab was refreshed only when the brain
changed and after Approve or Reject; Approve then signed whatever registration
was in the folder at the moment of the click. A registration finished by a
script after the person looked was approved unseen. Now the widget remembers
which registration the label (or the QC view) showed, Approve refuses when the
folder holds another one, and opening the Registration tab refreshes the label.
"""
import os
import types
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space

tw = pytest.importorskip("mousebrain.plugin_3d.tuning_widget")


def _register(reg, t, size=10):
    reg.mkdir(parents=True, exist_ok=True)
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"r" * size)
        os.utime(reg / name, (t, t))


def _brain(tmp_path, t0=1_700_000_000):
    b = tmp_path / "M1" / "101_PROJ_01_01_2p5x_z5"
    crop = b / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "ch0" / "Z0000.tif").write_bytes(b"")
    os.utime(crop / "ch0" / "Z0000.tif", (t0, t0))
    (crop / "metadata.json").write_text("{}")
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", crop)
    _register(reg, t0 + 100)
    return b, reg


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, *a):
        pass

    def setWordWrap(self, *a):
        pass


class _Messages:
    def __init__(self):
        self.seen = []

    def __getattr__(self, name):
        if name in ("warning", "information", "critical"):
            return lambda parent, title, text, *a, **k: self.seen.append((name, title, text))
        raise AttributeError(name)


class _Widget:
    def __getattr__(self, name):
        fn = getattr(tw.TuningWidget, name)
        if isinstance(fn, types.FunctionType):
            return types.MethodType(fn, self)
        return fn


def _widget(monkeypatch, brain):
    msgs = _Messages()
    monkeypatch.setattr(tw, "QMessageBox", msgs)
    w = _Widget()
    w.current_brain = brain
    w.approval_status = _Label()
    w.msgs = msgs
    return w


def test_approve_refuses_a_registration_made_after_the_person_looked(tmp_path, monkeypatch):
    brain, reg = _brain(tmp_path)
    w = _widget(monkeypatch, brain)
    w._update_registration_status()                 # the person sees registration 1
    _register(reg, 1_700_000_500, size=20)          # a script registers it again
    w.approve_registration()
    assert not ra.is_approved(reg)
    assert ra.read_marker(reg) is None
    assert any(kind == "warning" for kind, _, _ in w.msgs.seen)
    assert "PENDING APPROVAL" in w.approval_status.text   # label now shows the new one


def test_approve_signs_the_registration_that_was_shown(tmp_path, monkeypatch):
    brain, reg = _brain(tmp_path)
    w = _widget(monkeypatch, brain)
    w._update_registration_status()
    w.approve_registration()
    assert ra.is_approved(reg)


def test_approve_refuses_while_the_qc_pictures_are_of_an_earlier_registration(
        tmp_path, monkeypatch):
    brain, reg = _brain(tmp_path)
    w = _widget(monkeypatch, brain)
    w._qc_view_registration_id = ra.fingerprint_id(ra.registration_fingerprint(reg))
    _register(reg, 1_700_000_500, size=20)          # registered again after the pictures
    w._update_registration_status()                 # e.g. the tab was opened again
    w.approve_registration()
    assert ra.read_marker(reg) is None


def test_opening_the_registration_tab_refreshes_the_status(tmp_path, monkeypatch):
    brain, reg = _brain(tmp_path)
    w = _widget(monkeypatch, brain)
    w._registration_tab_index = 2
    w.approval_status.setText("Status: NOT REGISTERED")   # shown before it was registered
    w._on_tab_changed(2)
    assert "PENDING APPROVAL" in w.approval_status.text


def test_the_approval_api_refuses_another_registration_than_expected(tmp_path):
    brain, reg = _brain(tmp_path)
    shown = ra.fingerprint_id(ra.registration_fingerprint(reg))
    _register(reg, 1_700_000_500, size=20)
    with pytest.raises(ValueError):
        ra.approve(reg, approved_by="tester", expected_id=shown)
    assert ra.read_marker(reg) is None
    now = ra.fingerprint_id(ra.registration_fingerprint(reg))
    ra.approve(reg, approved_by="tester", expected_id=now)
    assert ra.is_approved(reg)
