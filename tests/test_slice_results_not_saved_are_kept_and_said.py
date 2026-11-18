"""When the results of the image on screen cannot be saved, opening another
image does not throw them away without asking, and the window keeps saying
that they were not saved.

The save made on the way to the next image used to fail with only a status
line (a file of the _analysis folder open in another program); the results
were then cleared and the status line overwritten at once by "Loading
image...", so the only trace was in the console. A folder stack's results,
which are never saved, vanished the same way.
"""
import numpy as np
import pandas as pd

from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget

from slice_widget_helpers import Control, finish_load, image_files, make_loading_widget


def _particles(w):
    w._pa_results = pd.DataFrame({"label": [1, 2], "area": [10, 12]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)


def _locked(monkeypatch, name="particles.csv"):
    """`name` cannot be written: open in another program."""
    real = main_widget._write_state_file
    state = {"locked": True}

    def write(path, payload):
        if state["locked"] and path.name == name:
            raise PermissionError(13, "The file is open in another program", str(path))
        return real(path, payload)
    monkeypatch.setattr(main_widget, "_write_state_file", write)
    return state


def _answer(w, leave):
    asked = []

    def ask(problem):
        asked.append(problem)
        return leave
    w._ask_leave_unsaved = ask
    return asked


def _red_line(w):
    return w.not_saved_label.text()


def test_an_image_whose_results_cannot_be_saved_is_not_left_unless_asked(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._open_image(a, peek=False)
    finish_load(w)
    _particles(w)
    _locked(monkeypatch)
    asked = _answer(w, leave=False)

    opened = w._open_image(b, peek=False)

    assert asked, "the person was not asked"
    assert opened is False
    assert w.current_file == a and w._loading_file is None
    assert w._pa_results is not None and len(w._pa_results) == 2, "results thrown away"
    assert "NOT" in _red_line(w) and "SAMPLE_A" in _red_line(w)
    assert "Loading image" not in w.status_label.text()


def test_once_the_file_is_free_save_state_saves_and_clears_the_red_line(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._open_image(a, peek=False)
    finish_load(w)
    _particles(w)
    lock = _locked(monkeypatch)
    _answer(w, leave=False)
    w._open_image(b, peek=False)
    assert "NOT" in _red_line(w)

    lock["locked"] = False               # closed in the other program
    w._manual_save_state()

    assert (a.parent / "SAMPLE_A_analysis" / "particles.csv").exists()
    assert _red_line(w) == ""
    assert w._open_image(b, peek=False) is True


def test_leaving_anyway_says_what_was_lost_and_keeps_saying_it(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._open_image(a, peek=False)
    finish_load(w)
    _particles(w)
    _locked(monkeypatch)
    _answer(w, leave=True)

    assert w._open_image(b, peek=False) is True
    finish_load(w)                       # "Loading image...", load, restore

    assert w.current_file == b
    assert "NOT saved" in _red_line(w) and "SAMPLE_A" in _red_line(w)


def test_leaving_a_stack_says_its_results_were_not_saved_and_keeps_saying_it(tmp_path, monkeypatch):
    (b,) = image_files(tmp_path / "imgs", ["SAMPLE_B.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w.current_file = None
    w.current_folder = tmp_path / "STACK_FOLDER"
    w.current_folder.mkdir()
    w.is_folder_load = True
    _particles(w)

    w._open_image(b, peek=False)
    assert "NOT saved" in _red_line(w) and "STACK_FOLDER" in _red_line(w)
    finish_load(w)
    assert "NOT saved" in _red_line(w), "the message was lost on the way to the next image"
    assert not any(tmp_path.rglob("*_analysis"))


def test_a_batch_is_not_started_while_the_image_on_screen_cannot_be_saved(tmp_path, monkeypatch):
    (a,) = image_files(tmp_path / "imgs", ["SAMPLE_A.tif"])
    batch = image_files(tmp_path / "batch", ["SAMPLE_X.tif"])
    w = make_loading_widget(tmp_path, monkeypatch)
    w._open_image(a, peek=False)
    finish_load(w)
    _particles(w)
    _locked(monkeypatch)
    w._batch_files = batch
    w._batch_busy = lambda: None
    w._batch_progress = Control()
    ran = []

    def analyze(*args, **kwargs):
        ran.append(args)
        raise ValueError("not analyzed in this test")
    monkeypatch.setattr(main_widget, "analyze_batch_image", analyze)

    w._batch_run_analysis()

    assert ran == [], "the batch ran though the image on screen could not be saved"
    assert w.current_file == a and len(w._pa_results) == 2
    assert "NOT" in _red_line(w)
