"""An ROI file without an image size opens with its ROIs, and Save & Next makes it countable.

The napari widget's "Save ROIs" writes polygons but no image size. Counting
needs the size, so such a file is not countable -- but its polygons are a
person's drawing and read fine. The editor must open with them loaded, so one
Save & Next (changing nothing) re-saves them with the size. Treating the file
as unreadable would open the editor empty and ask for a redraw.
"""

import json
from argparse import Namespace

from mousebrain.plugin_2d.sliceatlas.cli import run_roi

from roi_cli_helpers import (
    SAMPLE, SQUARE, FakeEditor, batch_args, make_dataset, nd2_of, pin_registry,
    rois_of, write_measurements,
)


def _write_shapeless(path):
    """What the widget's Save ROIs writes: polygons, no image_shape."""
    path.write_text(json.dumps({
        "version": 1, "image": path.name, "created": "2000-01-01T00:00:00",
        "rois": [{"name": "Left", "vertices": SQUARE}],
    }))
    return path


def _dataset(tmp_path, monkeypatch):
    pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    # batch_results/ is searched by every version of the counter.
    write_measurements(base / "batch_results" / SAMPLE / "measurements.csv")
    return base


def test_draw_opens_with_the_rois_and_saves_them_with_the_image_size(tmp_path, monkeypatch):
    base = make_dataset(tmp_path)
    rois = _write_shapeless(rois_of(base))
    editor = FakeEditor(monkeypatch, keep=True)

    run_roi.cmd_draw(Namespace(image=str(nd2_of(base)), output=None, red_ch=None,
                               green_ch=None, measurements=None))

    assert editor.preloaded == [[{"name": "Left", "vertices": SQUARE}]]
    saved = json.loads(rois.read_text())
    assert saved["image_shape"] == [100, 100]
    assert [r["vertices"] for r in saved["rois"]] == [SQUARE]


def test_batch_save_and_next_makes_the_sample_countable(tmp_path, monkeypatch):
    base = _dataset(tmp_path, monkeypatch)
    rois = _write_shapeless(rois_of(base))
    editor = FakeEditor(monkeypatch, keep=True)

    run_roi.cmd_batch(batch_args(base))

    assert editor.preloaded == [[{"name": "Left", "vertices": SQUARE}]]
    saved = json.loads(rois.read_text())
    assert saved["image_shape"] == [100, 100]
    assert [r["vertices"] for r in saved["rois"]] == [SQUARE]
    # Counted in the same run: the per-sample counts file was written.
    assert list(tmp_path.rglob(f"{SAMPLE}_roi_counts.csv"))


def test_batch_lists_it_as_its_own_group_and_skip_done_opens_it(tmp_path, monkeypatch, capsys):
    base = _dataset(tmp_path, monkeypatch)
    rois = _write_shapeless(rois_of(base))
    before = rois.read_text()
    editor = FakeEditor(monkeypatch, keep=True)

    run_roi.cmd_batch(batch_args(base, skip_done=True))

    out = capsys.readouterr().out
    assert run_roi.NO_SHAPE_FIX in out
    assert "ROI file unreadable" not in out
    assert len(editor.opened) == 1 and editor.preloaded[0]
    # The file without the size is kept beside the new one, never overwritten.
    kept = [p for p in rois.parent.iterdir() if p.name.endswith(".noshape.bak")]
    assert len(kept) == 1 and kept[0].read_text() == before


def test_count_only_leaves_it_out_and_says_how_to_fix_it(tmp_path, monkeypatch, capsys):
    base = _dataset(tmp_path, monkeypatch)
    write_measurements(base / "batch_results" / "Q01_02_S06_ZONE" / "measurements.csv")
    (base / "images" / "Q01_02_S06_ZONE.nd2").write_bytes(b"")
    _write_shapeless(rois_of(base))
    FakeEditor(monkeypatch)
    from roi_cli_helpers import write_rois
    write_rois(rois_of(base, "Q01_02_S06_ZONE"))

    run_roi.cmd_batch(batch_args(base, count_only=True))

    out = capsys.readouterr().out
    assert f"{SAMPLE}: {run_roi.NO_SHAPE_FIX}" in out
    assert not list(tmp_path.rglob(f"{SAMPLE}_roi_counts.csv"))
    assert list(tmp_path.rglob("Q01_02_S06_ZONE_roi_counts.csv"))
