"""Shared set-up for the slice annotator tests: a Qt application and a
viewer model with no window, so the widget runs exactly as in napari but
nothing is shown on screen."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def make_widget():
    from napari.components import ViewerModel
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import annotator_widget

    return annotator_widget.SliceAnnotatorWidget(ViewerModel()), annotator_widget


class RecordingMessageBox:
    """Stands in for QMessageBox: records every box instead of showing it."""

    shown = []

    @classmethod
    def warning(cls, *args):
        cls.shown.append(("warning",) + args[1:])

    @classmethod
    def information(cls, *args):
        cls.shown.append(("information",) + args[1:])

    @classmethod
    def critical(cls, *args):
        cls.shown.append(("critical",) + args[1:])


def process_events():
    _APP.processEvents()
