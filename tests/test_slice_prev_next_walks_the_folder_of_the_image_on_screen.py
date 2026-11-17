"""Prev/Next walk the folder of the image on screen, and the position label
names that image, however it was opened.

The list Prev/Next walk was built once, from the first image loaded: after
Browse File or the image queue opened an image in another folder, Prev/Next
kept walking the first folder and the label named an image not on screen.
"""
from slice_widget_helpers import browse, finish_load, image_files, make_loading_widget


def test_browse_into_another_folder_moves_prev_next_there(tmp_path, monkeypatch):
    first = image_files(tmp_path / "one", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    second = image_files(tmp_path / "two", ["SAMPLE_C.tif", "SAMPLE_D.tif", "SAMPLE_E.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, first[0])
    finish_load(w)

    browse(w, monkeypatch, second[1])
    finish_load(w)
    assert w._nav_siblings == second
    assert w._img_nav_label.text().startswith("2/3: SAMPLE_D.tif")

    w._nav_next_image()
    w._nav_load_selected()
    finish_load(w)
    assert w.current_file == second[2]


def test_the_queue_moves_the_position_label(tmp_path, monkeypatch):
    first = image_files(tmp_path / "one", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    second = image_files(tmp_path / "two", ["SAMPLE_C.tif", "SAMPLE_D.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, first[0])
    finish_load(w)

    w._queue_files, w._queue_idx = [second[1]], 0
    w._queue_load_current()
    finish_load(w)
    assert w._img_nav_label.text().startswith("2/2: SAMPLE_D.tif")
