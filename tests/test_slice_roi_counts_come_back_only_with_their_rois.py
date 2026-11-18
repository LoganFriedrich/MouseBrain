"""ROI counts (roi_detail.csv) come back only together with the ROIs they
were counted in (rois.json).

The restore read roi_detail.csv BEFORE rois.json, so its check "belongs to
rois.json, which was not restored" could never fire: with rois.json
unreadable, the counts came back with no ROIs on screen to explain them.
"""
import pandas as pd

from test_slice_saved_analysis_is_one_state import _draw_roi, _open


def test_counts_are_not_restored_when_their_rois_are_unreadable(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _draw_roi(w)
    w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [3]})
    w._save_analysis_state()
    folder = image.parent / "PRJ_01_01_S1_analysis"
    (folder / "rois.json").write_text("{ not json")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    w2, _ = _open(tmp_path, monkeypatch)
    assert w2._roi_detail_data is None
    status = w2.status_label.text()
    assert "rois.json" in status and "roi_detail.csv" in status

    # Saving the image again keeps both files exactly as they were.
    w2._save_analysis_state()
    assert (folder / "rois.json").read_bytes() == before["rois.json"]
    assert (folder / "roi_detail.csv").read_bytes() == before["roi_detail.csv"]


def test_counts_come_back_with_readable_rois(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _draw_roi(w)
    w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [3]})
    w._save_analysis_state()

    w2, _ = _open(tmp_path, monkeypatch)
    assert list(w2._roi_detail_data["total"]) == [3]
    assert w2.roi_shapes_layer is not None and len(w2.roi_shapes_layer.data) == 1
