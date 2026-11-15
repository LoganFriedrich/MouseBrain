"""Starting a new load in the slice annotator while another is running
cancels the old one quietly.

Before this, the cancelled worker's "Loading cancelled" report was handled as
an ordinary result: a "Load Error" box appeared for a load the person had
replaced on purpose, and both load buttons were switched back on while the
new file was still loading.
"""
import threading
from pathlib import Path

import numpy as np
from qtpy.QtCore import QThread, Signal

from annotator_helpers import RecordingMessageBox, make_widget, process_events


class FakeLoader(QThread):
    """Behaves like the real loader: blocks until released or cancelled,
    and reports a cancellation through `finished` exactly as it does."""

    progress = Signal(str)
    finished = Signal(bool, str, object, object)

    def __init__(self, file_path, lazy=False, z_projection='max', parent=None):
        super().__init__(parent)
        self.file_path = Path(file_path)
        self._cancelled = False
        self.release = threading.Event()

    def cancel(self):
        self._cancelled = True
        self.release.set()

    def run(self):
        self.release.wait(10)
        if self._cancelled:
            self.finished.emit(False, "Loading cancelled", None, None)
            return
        # Not constant: a flat image would not survive the layer set-up, and
        # the test must see the replacement load finish all the way through.
        data = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
        meta = {"file_path": str(self.file_path), "n_channels": 2,
                "channel_names": ["A", "B"], "shape": data.shape}
        self.finished.emit(True, "loaded", data, meta)


def test_replacing_a_load_shows_no_error_and_keeps_the_buttons_off(monkeypatch, tmp_path):
    widget, module = make_widget()
    RecordingMessageBox.shown = []
    monkeypatch.setattr(module, "QMessageBox", RecordingMessageBox)
    monkeypatch.setattr(module, "ND2LoaderWorker", FakeLoader)

    first, second = tmp_path / "first.nd2", tmp_path / "second.nd2"
    widget._load_file(first)
    widget._load_file(second)          # cancels the first load
    new_worker = widget.loader_worker
    for _ in range(5):
        process_events()               # deliver the old worker's report

    assert RecordingMessageBox.shown == []
    assert not widget.load_file_btn.isEnabled(), "the new load is still running"
    assert widget.current_data is None

    new_worker.release.set()
    new_worker.wait()
    for _ in range(5):
        process_events()

    assert RecordingMessageBox.shown == []
    assert widget.current_file == second
    assert widget.current_data is not None, "the new load finished"
    assert len(widget.channel_layers) == 2, "both channels of the new file are shown"
    assert all(layer in widget.viewer.layers for layer in widget.channel_layers)
    assert widget.load_file_btn.isEnabled()
    assert widget.load_folder_btn.isEnabled()


def test_a_real_load_failure_is_still_reported(monkeypatch, tmp_path):
    widget, module = make_widget()
    RecordingMessageBox.shown = []
    monkeypatch.setattr(module, "QMessageBox", RecordingMessageBox)

    class FailingLoader(FakeLoader):
        def run(self):
            self.finished.emit(False, "Failed to load broken.nd2: bad header", None, None)

    monkeypatch.setattr(module, "ND2LoaderWorker", FailingLoader)
    widget._load_file(tmp_path / "broken.nd2")
    widget.loader_worker.wait()
    for _ in range(5):
        process_events()

    assert [box[0] for box in RecordingMessageBox.shown] == ["warning"]
    assert widget.load_file_btn.isEnabled()
