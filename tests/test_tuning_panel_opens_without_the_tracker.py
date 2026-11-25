"""Setup & Tuning opens when the tracker cannot be opened, and says why in red.

The calibration tracker refuses to create a new calibration_runs.csv when its
folder is not visible (a share reconnecting, a new install), so a fresh empty
tracker never orphans the real one. That refusal must not escape from the
panel's constructor, or the whole Setup & Tuning panel fails to open. The
panel opens without a tracker and shows in red why runs are not being logged.
"""
import pytest

pytest.importorskip("napari")


def _refuse(*a, **k):
    raise RuntimeError("refusing to create a new calibration tracker: its folder "
                       "X does not exist")


def test_the_panel_opens_and_shows_why_runs_are_not_logged(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from unittest.mock import MagicMock
    from qtpy.QtWidgets import QApplication, QLabel
    import mousebrain.plugin_3d.tuning_widget as tw
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(tw, "ExperimentTracker", _refuse)
    monkeypatch.setattr(tw, "TRACKER_AVAILABLE", True)
    w = None
    try:
        w = tw.TuningWidget(MagicMock())
        assert w.tracker is None
        shown = [lbl.text() for lbl in w.findChildren(QLabel)
                 if lbl.isVisibleTo(w) and "NOT be logged" in lbl.text()]
        assert len(shown) == 1 and "does not exist" in shown[0]
    except RuntimeError as e:
        pytest.fail("the Setup & Tuning panel did not open: %s" % e)
    finally:
        if w is not None:
            w.deleteLater()
        app.processEvents()


def test_no_warning_while_the_tracker_is_open():
    import mousebrain.plugin_3d.tuning_widget as tw
    from types import SimpleNamespace
    w = SimpleNamespace(tracker=object(), _tracker_error=None)
    assert tw.TuningWidget._tracker_warning_text(w) == ""
