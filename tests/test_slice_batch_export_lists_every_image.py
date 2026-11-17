"""Export Batch Results lists every image of the batch, each with what
happened to it: counted, no ROI drawn, skipped, or failed and why.

The summary CSV used to hold only the images whose ROIs were counted. An
image that was skipped, counted with no ROI drawn, or could not be read at
all simply was not there -- indistinguishable from an image that was never
in the folder -- and the read failures were printed to the console only.
"""
from slice_batch_helpers import (draw_roi_around_everything, export_batch,
                                 make_batch_widget, two_channel_image)


def _batch(tmp_path, monkeypatch):
    images = {
        "A_01.tif": two_channel_image(1),
        "A_02.tif": two_channel_image(2),
        "A_03.tif": two_channel_image(3),
        "A_04.tif": None,                        # cannot be read
        "A_05.tif": RuntimeError("damaged file"),
    }
    w, folder = make_batch_widget(tmp_path, monkeypatch, images)
    w._batch_run_analysis()
    # A_01: an ROI drawn and counted.
    draw_roi_around_everything(w)
    w._batch_count_and_next()
    # A_02: counted with no ROI drawn.
    w._batch_count_and_next()
    # A_03: skipped.
    w._batch_skip_image()
    return w


def test_the_summary_has_a_row_for_every_image_saying_what_happened(tmp_path, monkeypatch):
    w = _batch(tmp_path, monkeypatch)
    (tmp_path / "out").mkdir()
    summary = export_batch(w, monkeypatch, tmp_path / "out" / "x_summary.csv")

    status = {}
    for _, row in summary.iterrows():
        status.setdefault(row["sample"], set()).add(str(row["status"]))
    assert set(status) == {"A_01.tif", "A_02.tif", "A_03.tif", "A_04.tif", "A_05.tif"}
    assert status["A_01.tif"] == {"counted"}
    assert status["A_02.tif"] == {"no ROI drawn"}
    assert status["A_03.tif"] == {"skipped"}
    assert status["A_04.tif"] == {"failed: could not be read"}
    assert status["A_05.tif"] == {"failed: damaged file"}
    # The counted image's counts are still there.
    total = summary[(summary["sample"] == "A_01.tif") & (summary["roi"] == "TOTAL")]
    assert int(total["total"].iloc[0]) == 1


def test_failed_images_are_named_in_a_message_when_the_run_ends(tmp_path, monkeypatch):
    w = _batch(tmp_path, monkeypatch)
    text = "\n".join(w.messages)
    assert "A_04.tif" in text and "could not be read" in text
    assert "A_05.tif" in text and "damaged file" in text


def test_going_back_and_skipping_keeps_a_counted_image_counted(tmp_path, monkeypatch):
    images = {"A_01.tif": two_channel_image(1), "A_02.tif": two_channel_image(1)}
    w, folder = make_batch_widget(tmp_path, monkeypatch, images)
    w._batch_run_analysis()
    draw_roi_around_everything(w)
    w._batch_count_and_next()       # A_01 counted
    w._batch_prev_image()           # back to A_01
    w._batch_skip_image()           # ... and on again without counting
    w._batch_skip_image()           # A_02 skipped
    (tmp_path / "out").mkdir()
    summary = export_batch(w, monkeypatch, tmp_path / "out" / "x_summary.csv")
    by_sample = summary.groupby("sample")["status"].agg(set).to_dict()
    assert by_sample == {"A_01.tif": {"counted"}, "A_02.tif": {"skipped"}}


def test_a_new_batch_folder_starts_with_nothing_of_the_last_batch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
    w, folder = make_batch_widget(tmp_path, monkeypatch,
                                  {"A_01.tif": None, "A_02.tif": two_channel_image(1)})
    w._batch_run_analysis()
    w._batch_skip_image()                   # the ROI drawing ends
    assert w._batch_failed
    other = tmp_path / "batch_two"
    other.mkdir()
    (other / "B_01.tif").write_bytes(b"")
    monkeypatch.setattr(main_widget, "QFileDialog", SimpleNamespace(
        getExistingDirectory=lambda *a, **k: str(other)))
    w._batch_select_folder()
    assert w._batch_failed == {} and w._batch_results == {}
    assert w._batch_export_btn.isEnabled() is False
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(main_widget, "QFileDialog", SimpleNamespace(
        getSaveFileName=lambda *a, **k: (str(out / "x_summary.csv"), "")))
    w._batch_export()
    assert not (out / "x_summary.csv").exists()
