"""BrainSlice napari plugin for interactive slice analysis."""

from .main_widget import BrainSliceWidget
from .inset_widget import InsetWidget
from .alignment_widget import AlignmentWidget
from .workers import (
    ImageLoaderWorker,
    FolderLoaderWorker,
    DetectionWorker,
    ColocalizationWorker,
    QuantificationWorker,
)

__all__ = [
    'BrainSliceWidget',
    'InsetWidget',
    'AlignmentWidget',
    'ImageLoaderWorker',
    'FolderLoaderWorker',
    'DetectionWorker',
    'ColocalizationWorker',
    'QuantificationWorker',
]
