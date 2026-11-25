"""The 2D window opens one image at a time, and never opens an image while an
analysis runs, or runs an analysis while an image opens.

A second load started while one is running could finish first, so that the
older one's pixels would be shown under the newer file's name. A detection
started during a load would return nuclei of the previous image and draw
them on the new one.
"""
import numpy as np

from slice_widget_helpers import (StubLoader, StubWorker, browse, finish_load,
                                  image_files, labels_with_one_nucleus,
                                  make_loading_widget)


def test_a_second_load_is_refused_while_one_runs(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    browse(w, monkeypatch, b)           # A is still loading
    assert len(StubLoader.made) == 1
    assert "still loading" in w.status_label.text()

    w._queue_files, w._queue_idx = [a, b], 0
    w._queue_next()
    assert len(StubLoader.made) == 1 and w._queue_idx == 0

    finish_load(w)
    assert w.current_file == a
    assert w.sample_id_edit.text() == "SAMPLE_A"


def test_a_load_result_nobody_waits_for_is_not_applied(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    pixels = finish_load(w)
    finish_load(w, shape=(4, 4))        # a stray second result
    assert w.red_channel.shape == pixels.shape


def test_no_detection_starts_while_an_image_loads(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    browse(w, monkeypatch, b)           # B loading, A's pixels still held
    assert not w.detect_btn.isEnabled()
    made = len(StubWorker.made)
    w._run_detection()
    assert len(StubWorker.made) == made
    assert w.tracker.search(run_type="detection") == []


def test_no_image_opens_while_detection_runs(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    w._run_detection()
    assert w._analysis_running == "detection"
    loads = len(StubLoader.made)
    browse(w, monkeypatch, b)
    assert len(StubLoader.made) == loads
    assert w.current_file == a


def test_a_detection_result_for_pixels_no_longer_shown_is_dropped(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    w._run_detection()
    run_id = w._detection_run_id
    w.red_channel = np.zeros((8, 8))    # the pixels changed under it
    w._on_detect_finished(True, "done", 1, labels_with_one_nucleus())
    assert w.nuclei_labels is None
    assert w.tracker.get_run(run_id)["status"] == "failed"
