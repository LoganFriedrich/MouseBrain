"""Closing the ROI editor never deletes an ROI file.

Closing the window with no ROIs used to delete the sample's ROI file -- after
Clear, and also when the file could not be read and the editor had opened
empty. The file is a person's drawing and the only copy of it.
"""

from mousebrain.plugin_2d.sliceatlas.cli import run_roi

from roi_cli_helpers import (
    PROJECT, SAMPLE, SQUARE, FakeEditor, batch_args, make_dataset, pin_registry,
    rois_of, write_measurements, write_rois,
)

UNREADABLE = "{ this is not json"


def _dataset(tmp_path, monkeypatch):
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    write_measurements(reg / "exports" / f"{PROJECT}_Detection" / SAMPLE / "measurements.csv")
    return base


def _files_holding(folder, text):
    return [p for p in folder.iterdir() if p.is_file() and p.read_text() == text]


def test_clearing_the_rois_keeps_the_old_file_as_a_backup(tmp_path, monkeypatch, capsys):
    base = _dataset(tmp_path, monkeypatch)
    rois = write_rois(rois_of(base))
    before = rois.read_text()
    FakeEditor(monkeypatch, drawn=[])

    run_roi.cmd_batch(batch_args(base))

    kept = _files_holding(rois.parent, before)
    assert len(kept) == 1 and kept[0].name.endswith(".cleared.bak")
    # The sample now has no ROIs, so it is not counted.
    assert not rois.exists()
    assert kept[0].name in capsys.readouterr().out


def test_an_unreadable_file_survives_closing_the_editor_empty(tmp_path, monkeypatch):
    base = _dataset(tmp_path, monkeypatch)
    rois = rois_of(base)
    rois.write_text(UNREADABLE)
    FakeEditor(monkeypatch, drawn=[])

    run_roi.cmd_batch(batch_args(base))

    assert rois.read_text() == UNREADABLE


def test_drawing_over_an_unreadable_file_keeps_the_old_one(tmp_path, monkeypatch):
    base = _dataset(tmp_path, monkeypatch)
    rois = rois_of(base)
    rois.write_text(UNREADABLE)
    FakeEditor(monkeypatch, drawn=[SQUARE])

    run_roi.cmd_batch(batch_args(base))

    kept = _files_holding(rois.parent, UNREADABLE)
    assert len(kept) == 1 and kept[0].name.endswith(".unreadable.bak")
    assert '"Left"' in rois.read_text()
