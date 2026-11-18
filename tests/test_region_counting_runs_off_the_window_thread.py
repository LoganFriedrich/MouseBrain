"""Region counting from the Results tab runs step 6 off napari's thread, with
no time limit.

It ran ``subprocess.run(..., timeout=300)`` on napari's own thread: the window
froze for the whole count, and a whole brain takes far longer than five
minutes, so the count was killed part-way and its tracker row stayed
'started'. Now step 6 runs in a worker thread to its end, its printed lines
feed the status line, a second start is refused meanwhile, and the results
are shown only for the brain that was counted.
"""
import subprocess
import threading
from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari")
import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _tracker, _widget  # noqa: E402

STEP6 = """\
print("Counting regions for the brain")
print("Wrote cell_counts_by_region.csv")
"""


class _Status:
    def __init__(self):
        self.texts = []
        self.enabled = True

    def setText(self, t):
        self.texts.append(t)

    def setStyleSheet(self, _s):
        pass

    def setEnabled(self, v):
        self.enabled = v


@pytest.fixture
def counting(tmp_path, monkeypatch):
    """A widget on a classified, registered brain; step 6 is a small script
    that prints two lines. Every subprocess call is recorded with its thread
    and its arguments."""
    script = tmp_path / "scripts" / "6_count_regions.py"
    script.parent.mkdir()
    script.write_text(STEP6, encoding="utf-8")
    monkeypatch.setattr(tw, "find_script", lambda name: script)

    calls = []
    real_popen = subprocess.Popen

    def popen(*a, **k):
        calls.append(("Popen", threading.get_ident(), k))
        return real_popen(*a, **k)

    def run(*a, **k):
        calls.append(("run", threading.get_ident(), k))
        return subprocess.CompletedProcess(a[0] if a else k.get("args"), 0, "", "")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", run)

    # The test drives the worker's run() itself, on this thread, after the
    # button handler has returned (no Qt event loop here).
    worker_class = getattr(tw, "RegionCountWorker", None)
    if worker_class is not None:
        monkeypatch.setattr(worker_class, "start", lambda self: None)

    w = _widget(tmp_path, monkeypatch, _tracker(tmp_path, []), brain=BRAIN_A)
    (w.current_brain / "5_Classified_Cells").mkdir(parents=True)
    (w.current_brain / "5_Classified_Cells" / "cells.xml").write_text(
        "<Marker></Marker>", encoding="utf-8")
    # A finished registration a person approved: counting refuses anything
    # less (a bare brainreg.json is only a registration that STARTED).
    from mousebrain import registration_approval as ra
    reg = w.current_brain / "3_Registered_Atlas"
    reg.mkdir()
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_text("{}", encoding="utf-8")
    ra.approve(reg, approved_by="tester")
    w.region_count_status = _Status()
    w.run_regions_btn = _Status()
    w.brain_combo = _Status()
    w.loaded = []
    w._load_region_comparison = lambda: w.loaded.append(w.current_brain)
    return w, calls


def test_counting_does_not_run_step_6_on_the_windows_thread(counting):
    w, calls = counting
    w.run_region_counting()
    here = threading.get_ident()
    assert [c for c in calls if c[1] == here] == [], "step 6 ran on napari's thread"
    assert w.run_regions_btn.enabled is False and w.brain_combo.enabled is False


def test_the_count_runs_to_its_end_with_no_time_limit(counting):
    w, calls = counting
    w.run_region_counting()
    worker = getattr(w, "region_count_worker", None)  # none: it ran inside the call
    if worker is not None:
        worker.run()  # what the thread does
    assert calls and all("timeout" not in c[2] for c in calls)
    assert any("Wrote cell_counts_by_region.csv" in t for t in w.region_count_status.texts)
    assert w.loaded == [w.current_brain]
    assert w.run_regions_btn.enabled is True and w.brain_combo.enabled is True


def test_a_second_start_is_refused_while_a_count_runs(counting):
    w, calls = counting
    w.region_count_worker = SimpleNamespace(isRunning=lambda: True,
                                            brain=w.current_brain)
    w.run_region_counting()
    assert calls == []
    assert "Counting Running" in [m[1] for m in w.msgs.seen]


def test_a_count_finishing_after_a_brain_change_shows_nothing_of_the_new_brain(counting):
    w, calls = counting
    w.run_region_counting()
    worker = getattr(w, "region_count_worker", None)  # none: it ran inside the call
    w.current_brain = w.current_brain.parent / BRAIN_B
    if worker is not None:
        worker.run()
    assert w.loaded == []
    assert BRAIN_A in w.region_count_status.texts[-1]


def test_the_window_does_not_close_under_a_running_count(monkeypatch):
    messages = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: messages.append(a[1]))
    widget = SimpleNamespace(region_count_worker=SimpleNamespace(isRunning=lambda: True),
                             session_doc=None, _runs_in_progress=set(), tracker=None)
    event = SimpleNamespace(accepted=False, ignored=False)
    event.accept = lambda: setattr(event, "accepted", True)
    event.ignore = lambda: setattr(event, "ignored", True)
    tw.TuningWidget.closeEvent(widget, event)
    assert event.ignored and not event.accepted
    assert messages == ["Region Counting Running"]
