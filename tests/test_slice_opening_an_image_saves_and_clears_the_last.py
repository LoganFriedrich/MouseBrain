"""Opening another image saves the results of the image on screen and then
forgets them, whichever way the image is opened.

Browse File used to switch to the new image without saving or clearing, so
image A's particles stayed in memory and the next Save State wrote them into
image B's _analysis folder.
"""
import pandas as pd
import numpy as np

from slice_widget_helpers import browse, finish_load, image_files, make_loading_widget


def _give_particles(w):
    w._pa_results = pd.DataFrame({"label": [1, 2], "area": [10, 12]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)


def test_browse_saves_the_last_image_and_leaves_nothing_of_it_for_the_next(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    _give_particles(w)

    browse(w, monkeypatch, b)
    finish_load(w)

    # A's results were saved beside A ...
    assert (a.parent / "SAMPLE_A_analysis" / "particles.csv").exists()
    # ... and none of them are left to be saved as B's.
    assert w._pa_results is None and w._pa_labels is None
    w._save_analysis_state()
    assert not (b.parent / "SAMPLE_B_analysis").exists()


def test_the_queue_and_the_load_button_go_the_same_way(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._queue_files, w._queue_idx = [a, b], 0
    w._queue_load_current()
    finish_load(w)
    _give_particles(w)

    w._queue_next()
    finish_load(w)
    assert (a.parent / "SAMPLE_A_analysis" / "particles.csv").exists()
    assert w._pa_results is None

    # Reloading B with the Load button: B's results are saved under B first.
    _give_particles(w)
    w._load_image()
    assert (b.parent / "SAMPLE_B_analysis" / "particles.csv").exists()
    assert w._pa_results is None
