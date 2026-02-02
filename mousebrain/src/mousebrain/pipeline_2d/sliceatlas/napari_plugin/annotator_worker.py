"""
loader_worker.py - Background worker for loading ND2 files.

Runs file loading in a separate thread to keep the UI responsive.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

from qtpy.QtCore import QThread, Signal


class ND2LoaderWorker(QThread):
    """
    Background worker for loading ND2 files.

    Emits:
        progress: Status message updates
        finished: (success, message, data, metadata) when complete
    """

    progress = Signal(str)
    finished = Signal(bool, str, object, object)  # success, message, data, metadata

    def __init__(
        self,
        file_path: Union[str, Path],
        lazy: bool = False,
        z_projection: Optional[str] = 'max',
        parent=None,
    ):
        """
        Initialize loader worker.

        Args:
            file_path: Path to ND2 file
            lazy: Use dask lazy loading (not recommended, MIP won't work)
            z_projection: Z projection mode - 'max' (default MIP), 'mean', 'min', 'sum',
                          or None to keep full optical z-stack
        """
        super().__init__(parent)
        self.file_path = Path(file_path)
        self.lazy = lazy
        self.z_projection = z_projection
        self._cancelled = False

    def cancel(self):
        """Request cancellation of loading."""
        self._cancelled = True

    def run(self):
        """Load the ND2 file in background thread."""
        try:
            self.progress.emit(f"Loading {self.file_path.name}...")

            # Import here to avoid issues with lazy imports in threads
            from mousebrain.pipeline_2d.sliceatlas.core.io import load_nd2, load_nd2_lazy

            if self._cancelled:
                self.finished.emit(False, "Loading cancelled", None, None)
                return

            # Load based on mode
            if self.lazy:
                self.progress.emit("Using lazy loading for large file...")
                data, metadata = load_nd2_lazy(self.file_path)
            else:
                self.progress.emit("Loading full data into memory...")
                data, metadata = load_nd2(self.file_path, z_projection=self.z_projection)

            if self._cancelled:
                self.finished.emit(False, "Loading cancelled", None, None)
                return

            # Report success
            shape_str = ' x '.join(str(s) for s in data.shape)
            n_channels = metadata.get('n_channels', 1)
            n_z = metadata.get('n_z', 1)

            msg = f"Loaded {self.file_path.name}: {shape_str} ({n_channels} channels, {n_z} Z slices)"
            self.progress.emit(msg)
            self.finished.emit(True, msg, data, metadata)

        except Exception as e:
            import traceback
            error_msg = f"Failed to load {self.file_path.name}: {str(e)}"
            print(f"Loader error:\n{traceback.format_exc()}")
            self.finished.emit(False, error_msg, None, None)


class FolderLoaderWorker(QThread):
    """
    Background worker for loading all ND2 files from a folder as tissue sections.

    Each ND2 file is treated as one physical tissue section. MIP is applied to
    flatten the optical z-stack in each file, then all files are stacked into
    (N_tissue_slices, C, Y, X) so the slider navigates between tissue sections.

    Emits:
        progress: (current, total, filename) during loading
        finished: (success, message, data, metadata) when complete
    """

    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(bool, str, object, object)

    def __init__(
        self,
        folder_path: Union[str, Path],
        z_projection: str = 'max',
        parent=None,
    ):
        """
        Initialize folder loader.

        Args:
            folder_path: Path to folder containing ND2 files
            z_projection: Projection mode for each file - 'max' (default MIP), 'mean', etc.
        """
        super().__init__(parent)
        self.folder_path = Path(folder_path)
        self.z_projection = z_projection
        self._cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def run(self):
        """Load all ND2 files from folder with MIP applied to each."""
        try:
            from mousebrain.pipeline_2d.sliceatlas.core.io import load_folder

            # Define progress callback
            def progress_callback(current, total, filename):
                if self._cancelled:
                    raise InterruptedError("Loading cancelled")
                self.progress.emit(current, total, filename)

            # Use load_folder which applies MIP and stacks properly
            stacked, metadata = load_folder(
                self.folder_path,
                z_projection=self.z_projection,
                progress_callback=progress_callback,
            )

            if self._cancelled:
                self.finished.emit(False, "Loading cancelled", None, None)
                return

            # Report success
            n_slices = metadata.get('n_slices', stacked.shape[0])
            n_channels = metadata.get('n_channels', 1)
            shape_str = ' x '.join(str(s) for s in stacked.shape)

            msg = f"Loaded {n_slices} tissue sections from {self.folder_path.name}: {shape_str} ({n_channels} channels)"
            self.finished.emit(True, msg, stacked, metadata)

        except InterruptedError:
            self.finished.emit(False, "Loading cancelled", None, None)
        except Exception as e:
            import traceback
            error_msg = f"Failed to load folder: {str(e)}"
            print(f"Folder loader error:\n{traceback.format_exc()}")
            self.finished.emit(False, error_msg, None, None)
