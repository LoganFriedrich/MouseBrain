"""A particle analysis whose tracker row cannot be written is not run.

When the tracker refused the new particle-analysis row (calibration_runs.csv
open in a spreadsheet program, another writer's lock not freed in time) the
error was swallowed: the analysis ran and saved its results with no row and
no word to the person -- while detection and colocalization refuse to start
in the same case. It must refuse the same way: say so, run nothing, and give
the Run button back.
"""
from types import SimpleNamespace

from mousebrain.plugin_2d.sliceatlas.core import particle_analysis
from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

from slice_widget_helpers import make_particle_widget

SAMPLE = "PRJ_03_04_S02"


class _AppendRefused(SliceTracker):
    """A tracker whose file refuses a new row (open in another program)."""

    def log_particle_analysis(self, *a, **k):
        raise PermissionError("the tracker file is open in another program")


def test_a_particle_analysis_whose_row_is_refused_does_not_run_and_says_so(
        tmp_path, monkeypatch):
    w = make_particle_widget(tmp_path, monkeypatch, SAMPLE, background=20.0)
    w.tracker = _AppendRefused(csv_path=tmp_path / "runs.csv")
    shown = []
    monkeypatch.setattr(main_widget, "QMessageBox", SimpleNamespace(
        warning=lambda _parent, title, text: shown.append((title, text)),
        critical=lambda *a, **k: None))
    analyzed = []
    real = particle_analysis.ParticleAnalyzer
    monkeypatch.setattr(particle_analysis, "ParticleAnalyzer",
                        lambda *a, **k: (analyzed.append(1), real(*a, **k))[1])

    w._pa_run_analysis()

    assert not analyzed, "an analysis with no tracker row must not run"
    assert w._pa_run_id is None
    assert "Particle analysis not started" in w.status_label.text()
    assert "open in another program" in w.status_label.text()
    assert shown and shown[0][0] == "Particle analysis not started"
    assert w.pa_run_btn.isEnabled() is True


def test_a_particle_analysis_with_its_row_written_still_runs(tmp_path, monkeypatch):
    w = make_particle_widget(tmp_path, monkeypatch, SAMPLE, background=20.0)

    w._pa_run_analysis()

    runs = w.tracker.search(sample_id=SAMPLE, run_type="particle_analysis")
    assert [r["status"] for r in runs] == ["completed"]
    assert w.pa_run_btn.isEnabled() is True
