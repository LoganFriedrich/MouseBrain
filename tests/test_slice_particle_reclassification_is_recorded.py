"""Re-classifying particles leaves a tracker row naming the analysis.

Changing the Background or Min % above background after a particle analysis
re-classifies the same particles and rewrites the saved results
(particles.csv). It used to leave no trace in the tracker: the analysis's row
still stated the old background and counts, so the saved results no longer
matched any record. Now each settled change is logged as its own
particle_analysis row whose parent_run is the analysis that found the
particles; the analysis's own row is left as it was.
"""
from slice_widget_helpers import Control, make_particle_widget

SAMPLE = "PRJ_03_02_S05"


def _rows(w):
    rows = [r for r in w.tracker.search(sample_id=SAMPLE, limit=None)
            if r["run_type"] == "particle_analysis"]
    return sorted(rows, key=lambda r: r["created_at"])


def _analysed(tmp_path, monkeypatch):
    w = make_particle_widget(tmp_path, monkeypatch, SAMPLE, background=20.0)
    w._pa_run_analysis()
    (analysis,) = _rows(w)
    assert analysis["status"] == "completed"
    return w, analysis


def test_a_reclassification_is_logged_naming_the_analysis_and_leaves_its_row_alone(
        tmp_path, monkeypatch):
    w, analysis = _analysed(tmp_path, monkeypatch)

    w.pa_bg_manual_spin = Control(value=250.0)   # above every particle pixel
    w._pa_reclassify_live()
    w._pa_reclass_timer.fire()                    # the values stopped changing

    rows = _rows(w)
    assert len(rows) == 2, "the re-classification left no row"
    first, second = rows
    assert analysis["pa_positive_particles"] == "1"
    assert first == analysis, "the analysis's own row was changed"
    assert second["parent_run"] == analysis["run_id"]
    assert second["status"] == "completed"
    assert float(second["pa_bg_value"]) == 250.0
    assert float(second["pa_min_pct_above_bg"]) == 50.0
    assert (second["pa_particles_found"], second["pa_positive_particles"]) == ("1", "0")
    # The particle-finding settings are the analysis's own.
    assert second["pa_threshold_value"] == analysis["pa_threshold_value"]


def test_changing_back_to_the_recorded_values_logs_nothing(tmp_path, monkeypatch):
    w, analysis = _analysed(tmp_path, monkeypatch)
    w.pa_bg_manual_spin = Control(value=250.0)
    w._pa_reclassify_live()
    w.pa_bg_manual_spin = Control(value=20.0)     # back before the pause ended
    w._pa_reclassify_live()
    w._pa_reclass_timer.fire()
    assert _rows(w) == [analysis]


def test_a_waiting_reclassification_is_recorded_before_the_next_analysis(tmp_path, monkeypatch):
    w, analysis = _analysed(tmp_path, monkeypatch)
    w.pa_pos_pct_spin = Control(value=100.0)
    w._pa_reclassify_live()                       # no pause: Run is clicked at once
    w._pa_run_analysis()

    rows = _rows(w)
    assert len(rows) == 3
    reclass = rows[1]
    assert reclass["parent_run"] == analysis["run_id"]
    assert float(reclass["pa_min_pct_above_bg"]) == 100.0


def test_recording_at_program_exit_survives_a_deleted_timer(tmp_path, monkeypatch):
    """The exit signal can fire after Qt deleted the timer; stopping it then
    raises RuntimeError. The waiting change is still recorded, nothing raises."""
    w, analysis = _analysed(tmp_path, monkeypatch)
    w.pa_bg_manual_spin = Control(value=250.0)
    w._pa_reclassify_live()

    class DeletedTimer:
        def stop(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    w._pa_reclass_timer = DeletedTimer()
    w._pa_record_reclassification()               # must not raise
    assert len(_rows(w)) == 2
