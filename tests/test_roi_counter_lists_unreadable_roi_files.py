"""An unreadable ROI file is listed as unreadable, never as done.

It must not count as "already has ROIs": --skip-done would never open it, and
the counting step would then skip the sample, so the sample would silently
have no counts. A file that is JSON but lacks what counting needs is reported
too, not allowed to crash the whole count.
"""

import json

from mousebrain.plugin_2d.sliceatlas.cli import run_roi

from roi_cli_helpers import (
    PROJECT, SAMPLE, FakeEditor, batch_args, make_dataset, pin_registry, rois_of,
    write_measurements, write_rois,
)

OTHER = "Q01_02_S09_ZONE"


def _dataset(tmp_path, monkeypatch, samples=(SAMPLE,)):
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path, samples=samples)
    for s in samples:
        write_measurements(reg / "exports" / f"{PROJECT}_Detection" / s / "measurements.csv")
    return base, reg


def test_skip_done_opens_a_sample_whose_roi_file_is_unreadable(tmp_path, monkeypatch, capsys):
    base, _ = _dataset(tmp_path, monkeypatch)
    rois_of(base).write_text("{ broken")
    editor = FakeEditor(monkeypatch, drawn=[])

    run_roi.cmd_batch(batch_args(base, skip_done=True))

    assert len(editor.opened) == 1 and SAMPLE in editor.opened[0]
    out = capsys.readouterr().out
    assert "ROI file unreadable: 1" in out
    assert "Already have ROIs: 0" in out


def test_a_file_without_an_image_shape_does_not_stop_the_count(tmp_path, monkeypatch, capsys):
    base, reg = _dataset(tmp_path, monkeypatch, samples=(SAMPLE, OTHER))
    write_rois(rois_of(base, SAMPLE))
    rois_of(base, OTHER).write_text(json.dumps({"rois": [{"name": "Left", "vertices": []}]}))

    rc = run_roi.cmd_batch(batch_args(base, count_only=True))

    out = capsys.readouterr().out
    assert rc == 0
    assert "SUMMARY: 1 samples counted, 1 left out" in out
    # Left out, and named with the fix (not as unreadable: its polygons read).
    assert [l for l in out.splitlines() if OTHER in l and run_roi.NO_SHAPE_FIX in l]


# A file saved by the napari widget's "Save ROIs": polygons, no image shape.
def _file_without_shape(path):
    path.write_text(json.dumps({
        "version": 1, "n_rois": 1, "roi_names": ["Left"],
        "rois": [{"name": "Left", "vertices": [[0, 0], [0, 50], [50, 50], [50, 0]]}],
    }))
    return path


def test_view_shows_a_roi_file_without_an_image_shape(tmp_path, monkeypatch, capsys):
    """Viewing does not need the shape, so such a file must still open."""
    from argparse import Namespace
    image = tmp_path / "slice.nd2"
    image.write_bytes(b"")
    rois = _file_without_shape(tmp_path / "slice_rois.json")
    editor = FakeEditor(monkeypatch)

    rc = run_roi.cmd_view(Namespace(image=str(image), rois=str(rois), red_ch=None,
                                    green_ch=None, measurements=None))

    assert rc == 0
    assert len(editor.shown_shapes) == 1 and len(editor.shown_shapes[0]) == 1


def test_count_refuses_a_roi_file_without_an_image_shape(tmp_path, capsys):
    """Counting needs the shape: one clear error, not a crash."""
    from argparse import Namespace
    meas = write_measurements(tmp_path / "measurements.csv")
    rois = _file_without_shape(tmp_path / "slice_rois.json")

    rc = run_roi.cmd_count(Namespace(measurements=str(meas), rois=str(rois), output=None))

    assert rc == 1
    assert "image_shape" in capsys.readouterr().out


def test_batch_lists_a_roi_file_without_an_image_shape_as_not_yet_countable(
        tmp_path, monkeypatch, capsys):
    """Not done (counting needs the shape), but not unreadable either: it has
    its own line telling the person how to add the shape (see
    test_roi_file_without_image_size_is_reopened_with_its_rois.py)."""
    base, _ = _dataset(tmp_path, monkeypatch)
    _file_without_shape(rois_of(base))

    run_roi.cmd_batch(batch_args(base, count_only=True))

    out = capsys.readouterr().out
    assert f"{run_roi.NO_SHAPE_FIX}: 1" in out
    assert "ROI file unreadable" not in out
    assert "Already have ROIs: 0" in out
