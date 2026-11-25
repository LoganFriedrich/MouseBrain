"""A load that fails leaves the window with no image, never the previous
image's pixels under the new file's name.

If current_file named the new file as soon as the image is chosen, then after
a failed load it would name the new file while the previous image's pixels
stayed on screen; a detection or a save would file the previous image's
results under the new name.
"""
from slice_widget_helpers import browse, finish_load, image_files, make_loading_widget, StubWorker


def test_failed_load_clears_the_old_pixels_and_names_nothing(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    assert w.current_file == a and w.red_channel is not None

    browse(w, monkeypatch, b)
    # While B loads, the window still names A (whose pixels are on screen).
    assert w.current_file == a
    finish_load(w, success=False, message="unreadable file")

    assert w.current_file is None
    assert w.red_channel is None and w.channels == []
    assert w.viewer.layers == []
    assert not w.detect_btn.isEnabled()
    assert not w.pa_run_btn.isEnabled()

    # Nothing can be run on stale pixels.
    made = len(StubWorker.made)
    w._run_detection()
    assert len(StubWorker.made) == made
    assert w.tracker.search(run_type="detection") == []


def test_load_button_tries_the_failed_file_again(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w, success=False, message="busy")
    w._load_image()
    finish_load(w)
    assert w.current_file == a
