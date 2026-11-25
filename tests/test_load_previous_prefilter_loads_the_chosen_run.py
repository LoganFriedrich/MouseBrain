"""'Load Previous' on the Pre-filter tab loads the pre-filter run chosen in
the list, and only a run of the brain selected.

The run is looked up with an argument tracker.search() accepts; a lookup that
raised (caught by the button) would end every press in "Could not load
previous result" with nothing loaded.
"""
from types import SimpleNamespace

from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _tracker, _widget

MARKERS = ("<CellCounter_Marker_File><Marker_Data><Marker_Type>%s</Marker_Type>"
           "</Marker_Data></CellCounter_Marker_File>")


def _xml(n):
    one = "<Marker><MarkerX>3</MarkerX><MarkerY>2</MarkerY><MarkerZ>1</MarkerZ></Marker>"
    return MARKERS % (one * n)


class _Stub:
    def __init__(self):
        self.text = None

    def setText(self, t):
        self.text = t

    def setEnabled(self, _v):
        pass


def _press(tmp_path, monkeypatch, run_brain, selected_brain):
    out = tmp_path / "prefilter_out"
    (out / "prefiltered_001").mkdir(parents=True)
    (out / "prefiltered_001" / "interior_candidates.xml").write_text(_xml(4), encoding="utf-8")
    (out / "prefiltered_001" / "suspicious_candidates.xml").write_text(_xml(1), encoding="utf-8")
    tracker = _tracker(tmp_path, [{"exp_id": "pf_1", "exp_type": "prefilter",
                                   "brain": run_brain, "output_path": str(out)}])
    w = _widget(tmp_path, monkeypatch, tracker, brain=selected_brain)
    item = SimpleNamespace(data=lambda _role: "pf_1")
    w.prefilter_history_list = SimpleNamespace(currentItem=lambda: item)
    w._prefilter_result = None
    w._display_prefilter_results = lambda result: None
    w.prefilter_results_label = _Stub()
    w.prefilter_save_btn = w.prefilter_to_training_btn = _Stub()
    w.prefilter_status = _Stub()
    w._load_previous_prefilter()
    return w


def test_the_chosen_previous_prefilter_is_loaded(tmp_path, monkeypatch):
    w = _press(tmp_path, monkeypatch, BRAIN_B, BRAIN_B)
    assert not [m for m in w.msgs.seen if m[0] == "warning"], w.msgs.seen
    assert w._prefilter_result["stats"] == {"total": 5, "interior": 4, "suspicious": 1}
    assert w.prefilter_status.text == "Loaded: pf_1"


def test_a_previous_prefilter_of_another_brain_is_not_loaded(tmp_path, monkeypatch):
    w = _press(tmp_path, monkeypatch, BRAIN_A, BRAIN_B)
    assert w._prefilter_result is None
    assert [m[1] for m in w.msgs.seen] == ["Run Belongs To Another Brain"]
