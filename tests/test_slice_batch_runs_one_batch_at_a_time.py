"""Run Batch Analysis works through the folder it was started on, and
nothing can change that folder or open another image until it is done.

The batch runs on the window's own thread and lets the window answer clicks
between images. Only the Run button used to be turned off, so choosing
another batch folder part-way replaced the batch's file list and emptied its
results while the run kept going: the images analyzed after the click landed
in the new batch's results under the old batch's file names, and the ones
before it were lost. An image opened part-way was analyzed against nothing
the batch knew about.
"""
from types import SimpleNamespace

from slice_batch_helpers import make_batch_widget, two_channel_image


def _run_with_click_between_images(w, monkeypatch, click):
    """Run Batch Analysis; `click(w)` happens while the window answers
    clicks after the first image."""
    from qtpy.QtWidgets import QApplication
    calls = {"n": 0}

    def process_events(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:     # between the first and the second image
            click(w)
    monkeypatch.setattr(QApplication, "processEvents", staticmethod(process_events))
    w._batch_enter_roi_mode = lambda: None
    w._batch_run_analysis()


def test_choosing_another_batch_folder_mid_run_is_refused(tmp_path, monkeypatch):
    images = {"A_01.tif": two_channel_image(1), "A_02.tif": two_channel_image(2),
              "A_03.tif": two_channel_image(3)}
    w, folder = make_batch_widget(tmp_path, monkeypatch, images)
    other = tmp_path / "batch_two"
    other.mkdir()
    (other / "B_01.tif").write_bytes(b"")

    from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget

    def choose_other_folder(w):
        monkeypatch.setattr(main_widget, "QFileDialog", SimpleNamespace(
            getExistingDirectory=lambda *a, **k: str(other)))
        w._batch_select_folder()

    _run_with_click_between_images(w, monkeypatch, choose_other_folder)

    assert w._batch_folder == folder
    assert [p.name for p in w._batch_files] == ["A_01.tif", "A_02.tif", "A_03.tif"]
    assert sorted(w._batch_results) == ["A_01.tif", "A_02.tif", "A_03.tif"]
    assert w._batch_results["A_03.tif"]["n_particles"] == 3
    assert "Batch folder not changed" in w.status_label.text()


def test_no_image_can_be_opened_and_nothing_else_run_mid_batch(tmp_path, monkeypatch):
    w, folder = make_batch_widget(tmp_path, monkeypatch,
                                  {"A_01.tif": two_channel_image(1),
                                   "A_02.tif": two_channel_image(1)})
    other_image = tmp_path / "elsewhere" / "C_01.tif"
    other_image.parent.mkdir()
    other_image.write_bytes(b"")
    seen = {}

    def click(w):
        seen["open"] = w._open_image(other_image, peek=False)
        seen["particles"] = w._refuse_if_analysis_running("Particle analysis")
        seen["run_again"] = w._analysis_in_progress()

    _run_with_click_between_images(w, monkeypatch, click)

    assert seen["open"] is False            # the image was not opened
    assert w._chosen_file != other_image
    assert seen["particles"] is True        # single-image analysis refused
    assert seen["run_again"]                # the window says what is running
    # ... and once the batch is done the window is free again.
    assert w._analysis_in_progress() is None
