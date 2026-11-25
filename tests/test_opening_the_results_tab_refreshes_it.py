"""Opening the Results tab refreshes the Results tab, and only that tab.

The tab-change handler must not compare the new tab with a hard-coded index:
tabs get added, and the index would then point at another tab (index 8 is
"9. Curate/Train"), so opening Results would not refresh it and opening
another tab would run the Results refresh.
"""
import pytest

pytest.importorskip("napari")


def _refuse(*a, **k):
    raise RuntimeError("no tracker in this test")


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from unittest.mock import MagicMock
    from qtpy.QtWidgets import QApplication
    import mousebrain.plugin_3d.tuning_widget as tw
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(tw, "ExperimentTracker", _refuse)
    monkeypatch.setattr(tw, "TRACKER_AVAILABLE", True)
    w = tw.TuningWidget(MagicMock())
    w.refreshed = []
    w._refresh_all_results_views = lambda: w.refreshed.append(True)
    try:
        yield w
    finally:
        w.deleteLater()
        app.processEvents()


def _tab(w, word):
    found = [i for i in range(w.tabs.count()) if word in w.tabs.tabText(i)]
    assert len(found) == 1, f"no single tab named '{word}'"
    return found[0]


def test_opening_results_refreshes_it(panel):
    panel._on_tab_changed(_tab(panel, "Results"))
    assert panel.refreshed == [True]


def test_opening_curate_train_does_not_refresh_results(panel):
    panel._on_tab_changed(_tab(panel, "Curate/Train"))
    assert panel.refreshed == []
