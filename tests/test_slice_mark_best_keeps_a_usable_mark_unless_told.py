"""Marking a run the batch refuses never silently clears a mark the batch uses.

Defect covered: "Mark Selected as Best" marked the run first and checked it
afterwards. Marking clears the sample's previous mark, so marking a run the
batch refuses (a fold-change run, say) over a usable mark left the batch with
no mark to use: it silently took other numbers, and the message did not say
the earlier mark was gone.
"""

from types import SimpleNamespace

import pytest

from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

try:
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import COLOC_SIGMA_RECORDED_VERSION
except ImportError:
    COLOC_SIGMA_RECORDED_VERSION = "1.1.0"

SAMPLE = "PRJ_01_02_S3_RA"


class _Box:
    """Stands in for QMessageBox: records what is shown, answers questions."""
    Yes, No = 1, 2

    def __init__(self, answer):
        self.answer = answer
        self.questions, self.infos, self.warnings = [], [], []

    def question(self, parent, title, text, *a):
        self.questions.append(text)
        return self.answer

    def information(self, parent, title, text):
        self.infos.append(text)

    def warning(self, parent, title, text):
        self.warnings.append(text)


def _coloc(t, method="background_mean", value=0.5):
    run = t.log_colocalization(
        sample_id=SAMPLE, signal_channel="green", background_method="gmm",
        background_percentile=10.0, threshold_method=method, threshold_value=value,
        status="started", script_version=COLOC_SIGMA_RECORDED_VERSION,
        notes="soma_dilation=6; bg_exclusion_dilation=50")
    t.update_status(run, status="completed")
    return run


def _widget(t, run_id):
    item = SimpleNamespace(row=lambda: 0, text=lambda: run_id)
    table = SimpleNamespace(selectedItems=lambda: [item], item=lambda r, c: item)
    return SimpleNamespace(run_history_table=table, tracker=t,
                           _refresh_run_history=lambda: None)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    counter = iter(range(1000))
    monkeypatch.setattr(t, "_generate_run_id",
                        lambda run_type, sample_id: f"col_{next(counter):04d}")
    usable = _coloc(t)
    t.mark_as_best(usable)
    refused = _coloc(t, method="fold_change", value=2.0)
    return mw, t, usable, refused


def _marked(t, run_id):
    return t.get_run(run_id).get("marked_best") == "True"


def test_declining_keeps_the_usable_mark(setup, monkeypatch):
    mw, t, usable, refused = setup
    box = _Box(_Box.No)
    monkeypatch.setattr(mw, "QMessageBox", box)
    mw.BrainSliceWidget._mark_selected_run_best(_widget(t, refused))
    assert len(box.questions) == 1 and usable in box.questions[0]
    assert _marked(t, usable) and not _marked(t, refused)


def test_confirming_marks_it_and_names_the_cleared_run(setup, monkeypatch):
    mw, t, usable, refused = setup
    box = _Box(_Box.Yes)
    monkeypatch.setattr(mw, "QMessageBox", box)
    mw.BrainSliceWidget._mark_selected_run_best(_widget(t, refused))
    assert _marked(t, refused) and not _marked(t, usable)
    assert f"The mark on run {usable} was cleared" in box.infos[0]
    assert "will NOT use" in box.infos[0]


def test_a_usable_run_replaces_a_mark_without_asking_and_says_so(setup, monkeypatch):
    mw, t, usable, _ = setup
    newer = _coloc(t, value=0.7)
    box = _Box(_Box.No)
    monkeypatch.setattr(mw, "QMessageBox", box)
    mw.BrainSliceWidget._mark_selected_run_best(_widget(t, newer))
    assert box.questions == []
    assert _marked(t, newer) and not _marked(t, usable)
    assert f"The mark on run {usable} was cleared" in box.infos[0]
