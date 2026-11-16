"""Saving an ROI file never leaves a half-written file in its place.

save_rois_json used to write over the file in place: a failure mid-write
left a truncated, unreadable file, and the ROIs saved before -- the only
copy of a person's drawing -- were gone.
"""

import json

import pytest

from mousebrain.plugin_2d.sliceatlas.core import roi as roi_mod

SQUARE = [[0, 0], [0, 50], [50, 50], [50, 0]]


def test_a_failed_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "slice.rois.json"
    roi_mod.save_rois_json(path, [{"name": "Left", "vertices": SQUARE}],
                           "slice.nd2", (100, 100))
    before = path.read_text()

    def dies_mid_write(data, f, **kw):
        f.write('{"version": 1, "ro')
        raise OSError("disk full")

    monkeypatch.setattr(roi_mod.json, "dump", dies_mid_write)
    with pytest.raises(OSError):
        roi_mod.save_rois_json(path, [{"name": "Right", "vertices": SQUARE}],
                               "slice.nd2", (100, 100))
    monkeypatch.undo()

    assert path.read_text() == before
    assert json.loads(before)["rois"][0]["name"] == "Left"
    assert [p.name for p in tmp_path.iterdir()] == ["slice.rois.json"]
