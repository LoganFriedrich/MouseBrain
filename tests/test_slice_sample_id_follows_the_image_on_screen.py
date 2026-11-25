"""The Sample ID box names the image on screen, however it was opened.

It must not be set by Browse File only: after Prev/Next or the image queue
every tracker row and Quantify export would otherwise go under the previously
browsed image's name.
"""
from slice_widget_helpers import browse, choose_folder, finish_load, image_files, make_loading_widget


def test_prev_next_puts_detection_rows_under_the_new_image(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    assert w.sample_id_edit.text() == "SAMPLE_A"

    w._nav_next_image()
    w._nav_load_selected()
    finish_load(w)
    assert w.sample_id_edit.text() == "SAMPLE_B"

    w._run_detection()
    rows = w.tracker.search(run_type="detection")
    assert [r["sample_id"] for r in rows] == ["SAMPLE_B"]


def test_the_queue_names_each_image(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._queue_files, w._queue_idx = [a, b], 0
    w._queue_load_current()
    finish_load(w)
    w._queue_next()
    finish_load(w)
    assert w.sample_id_edit.text() == "SAMPLE_B"


def test_choosing_a_folder_does_not_rename_the_image_still_on_screen(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    stack = tmp_path / "STACK_X"
    image_files(stack, ["p1.tif", "p2.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    browse(w, monkeypatch, a)
    finish_load(w)
    choose_folder(w, monkeypatch, stack)
    assert w.sample_id_edit.text() == "SAMPLE_A"
    # Load opens the stack, and only now does the name change.
    w._load_image()
    assert w.sample_id_edit.text() == "STACK_X"
