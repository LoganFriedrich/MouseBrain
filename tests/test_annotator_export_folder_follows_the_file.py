"""The slice annotator's export folder follows the loaded file, unless the
person chose a folder themselves.

Before this, the folder was filled in only while the box was empty: after the
first file, every later file exported into the first file's folder.
"""
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest

from annotator_helpers import make_widget


def _load_file(widget, path):
    data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    meta = {"file_path": str(path), "n_channels": 2,
            "channel_names": ["red", "green"], "shape": data.shape}
    widget._on_load_finished(True, "loaded", data, meta)


def _load_folder(widget, folder):
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    meta = {"folder": str(folder), "n_slices": 3, "n_channels": 2,
            "channel_names": ["red", "green"], "shape": data.shape}
    widget._on_load_finished(True, "loaded", data, meta)


def _folder(widget):
    return Path(widget.output_folder_edit.text())


def test_each_load_points_the_export_at_its_own_folder(tmp_path):
    widget, _ = make_widget()
    first, second, sections = tmp_path / "first", tmp_path / "second", tmp_path / "sections"

    _load_file(widget, first / "SAMPLE_A.nd2")
    assert _folder(widget) == first
    _load_file(widget, second / "SAMPLE_B.nd2")
    assert _folder(widget) == second
    _load_folder(widget, sections)
    assert _folder(widget) == sections


def test_a_typed_folder_is_kept_until_the_box_is_cleared(tmp_path):
    widget, _ = make_widget()
    _load_file(widget, tmp_path / "first" / "SAMPLE_A.nd2")

    box = widget.output_folder_edit
    box.selectAll()
    QTest.keyClicks(box, str(tmp_path / "my_exports"))
    _load_file(widget, tmp_path / "second" / "SAMPLE_B.nd2")
    assert _folder(widget) == tmp_path / "my_exports"

    box.selectAll()
    QTest.keyClick(box, Qt.Key_Backspace)
    assert box.text() == ""
    _load_file(widget, tmp_path / "third" / "SAMPLE_C.nd2")
    assert _folder(widget) == tmp_path / "third"


def test_a_browsed_folder_is_kept(monkeypatch, tmp_path):
    widget, module = make_widget()
    chosen = tmp_path / "browsed"
    monkeypatch.setattr(module.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(chosen)))
    _load_file(widget, tmp_path / "first" / "SAMPLE_A.nd2")
    widget._browse_output_folder()
    _load_file(widget, tmp_path / "second" / "SAMPLE_B.nd2")
    assert _folder(widget) == chosen
