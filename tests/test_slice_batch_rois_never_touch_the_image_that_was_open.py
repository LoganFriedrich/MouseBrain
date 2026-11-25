"""Drawing ROIs on the batch's images never writes into the folder of the
image that was open before the batch, and leaves no batch image's particles
in the window's own state.

Entering the batch's ROI drawing must not put the batch image's particles in
the window's single-image state while the window still names the image
opened before the batch as the one on screen: the next save -- Save State,
Count in ROI(s), or opening another image afterwards -- would write the batch
image's particles into THAT image's _analysis folder, over its own results.
"""
import numpy as np
import pandas as pd

from slice_batch_helpers import make_batch_widget, two_channel_image
from slice_widget_helpers import finish_load


def _window_with_image_open(tmp_path, monkeypatch):
    w, folder = make_batch_widget(
        tmp_path, monkeypatch,
        {"A_01.tif": two_channel_image(1), "A_02.tif": two_channel_image(2)})
    image = tmp_path / "single" / "PRJ_01_01_S1.tif"
    image.parent.mkdir()
    image.write_bytes(b"")
    w._open_image(image, peek=False)
    finish_load(w, shape=(40, 40))
    # The image's own particle analysis: seven particles.
    labels = np.zeros((40, 40), dtype=np.int32)
    for k in range(7):
        labels[k * 5:k * 5 + 3, 1:4] = k + 1
    w._pa_labels = labels
    w._pa_results = pd.DataFrame({"label": range(1, 8), "area": [9] * 7,
                                  "is_positive": [True] * 7})
    return w, image


def _saved_particles(image):
    return pd.read_csv(image.parent / f"{image.stem}_analysis" / "particles.csv")


def test_the_open_image_keeps_its_own_particles(tmp_path, monkeypatch):
    w, image = _window_with_image_open(tmp_path, monkeypatch)
    w._batch_run_analysis()                 # ROI drawing starts on A_01

    # Its own results were saved when the batch took over the window ...
    assert len(_saved_particles(image)) == 7
    # ... and a save made during the ROI drawing (Save State, Count in
    # ROI(s)) does not put a batch image's particles there.
    w._save_analysis_state()
    assert len(_saved_particles(image)) == 7
    assert w.current_file is None
    assert w._pa_labels is None and w._pa_results is None


def test_the_batch_overlay_is_drawn_without_the_single_image_state(tmp_path, monkeypatch):
    w, image = _window_with_image_open(tmp_path, monkeypatch)
    w._batch_run_analysis()
    assert "Positive/Negative" in w.viewer.names()
    w._batch_skip_image()                   # on to A_02
    assert "Positive/Negative" in w.viewer.names()
    assert w._pa_labels is None and w._pa_results is None
