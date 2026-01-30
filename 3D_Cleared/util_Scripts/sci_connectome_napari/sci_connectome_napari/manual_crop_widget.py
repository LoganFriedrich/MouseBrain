"""
Manual Crop Widget for Napari

Works with images already loaded in napari viewer.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QProgressBar, QMessageBox, QTextEdit, QScrollArea,
    QGroupBox, QRadioButton, QButtonGroup, QComboBox, QFileDialog
)
from qtpy.QtCore import Qt, QThread, Signal
import napari
from napari.utils import notifications


# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from central config (auto-detects repo location)
import sys as _sys
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in _sys.path:
    _sys.path.insert(0, str(_config_dir))
from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"


# =============================================================================
# BACKGROUND WORKER THREAD
# =============================================================================

class CropWorker(QThread):
    """Background thread for cropping operation."""
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(bool, str)  # success, message

    def __init__(self, source_folder, output_folder, y_crop, metadata, signal_channel='ch0'):
        super().__init__()
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.y_crop = y_crop
        self.metadata = metadata
        self.signal_channel = signal_channel  # Which channel has the cells

    def run(self):
        """Run the crop operation in background."""
        try:
            import tifffile

            self.output_folder.mkdir(parents=True, exist_ok=True)

            # Find all channels
            channels = sorted([d for d in self.source_folder.iterdir()
                             if d.is_dir() and d.name.startswith("ch")])

            total_files = sum(len(list(ch.glob("Z*.tif"))) for ch in channels)
            processed = 0

            for ch_folder in channels:
                ch_name = ch_folder.name
                self.progress.emit(processed, total_files, f"Processing {ch_name}...")

                out_ch = self.output_folder / ch_name
                out_ch.mkdir(exist_ok=True)

                tiff_files = sorted(ch_folder.glob("Z*.tif"))

                for tiff_path in tiff_files:
                    img = tifffile.imread(str(tiff_path))
                    cropped = img[:self.y_crop, :]  # Keep everything BEFORE y_crop (top portion)

                    out_path = out_ch / tiff_path.name
                    tifffile.imwrite(str(out_path), cropped)

                    processed += 1
                    if processed % 10 == 0:  # Update every 10 files
                        self.progress.emit(processed, total_files,
                                         f"Processing {ch_name}: {tiff_path.name}")

            # Save metadata
            self.progress.emit(total_files, total_files, "Saving metadata...")

            first_ch = channels[0]
            first_tif = sorted(first_ch.glob("Z*.tif"))[0]
            first_img = tifffile.imread(str(first_tif))
            orig_y, x = first_img.shape
            z = len(list(first_ch.glob("Z*.tif")))

            crop_meta = self.metadata.copy() if self.metadata else {}
            crop_meta['crop_y_end'] = self.y_crop  # Crop line position (everything before this is kept)
            crop_meta['crop_y_original'] = orig_y
            crop_meta['crop_y_new'] = self.y_crop  # New height = y_crop (kept top portion)
            crop_meta['crop_method'] = 'napari_plugin_manual'
            crop_meta['dimensions_cropped'] = {
                'z': z,
                'y': self.y_crop,  # New Y dimension
                'x': x,
            }
            # Channel assignment - critical for detection!
            crop_meta['signal_channel'] = self.signal_channel
            crop_meta['channels_swapped'] = (self.signal_channel == 'ch1')

            with open(self.output_folder / "metadata.json", 'w') as f:
                json.dump(crop_meta, f, indent=2)

            self.finished.emit(True, str(self.output_folder))

        except Exception as e:
            self.finished.emit(False, str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_metadata_from_path(image_path: Path):
    """Try to find metadata for a loaded image based on its path."""
    # Try to find the pipeline folder from the image path
    path_parts = image_path.parts

    # Look for pattern: .../BRAIN_NAME/1_Extracted_Full/ch0/...
    for i, part in enumerate(path_parts):
        if part == FOLDER_EXTRACTED and i > 0:
            pipeline_folder = Path(*path_parts[:i])
            meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    return json.load(f), pipeline_folder

    return None, None


def find_pipeline_folder_from_layer(layer):
    """Try to find the pipeline folder from a layer's source."""
    if hasattr(layer, 'source') and hasattr(layer.source, 'path') and layer.source.path is not None:
        path = Path(layer.source.path)
        return get_metadata_from_path(path)
    return None, None


def save_cropped_volume(source_folder: Path, output_folder: Path, y_crop: int, metadata: dict, progress_callback=None):
    """Crop and save all channels from source to output."""
    import tifffile

    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all channels in source
    channels = sorted([d for d in source_folder.iterdir() if d.is_dir() and d.name.startswith("ch")])

    total_files = sum(len(list(ch.glob("Z*.tif"))) for ch in channels)
    processed = 0

    for ch_folder in channels:
        ch_name = ch_folder.name
        out_ch = output_folder / ch_name
        out_ch.mkdir(exist_ok=True)

        tiff_files = sorted(ch_folder.glob("Z*.tif"))

        for tiff_path in tiff_files:
            img = tifffile.imread(str(tiff_path))
            cropped = img[:y_crop, :]  # Keep everything BEFORE y_crop (top portion)

            out_path = out_ch / tiff_path.name
            tifffile.imwrite(str(out_path), cropped)

            processed += 1
            if progress_callback:
                progress_callback(processed, total_files)

    # Get original dimensions
    first_ch = channels[0]
    first_tif = sorted(first_ch.glob("Z*.tif"))[0]
    first_img = tifffile.imread(str(first_tif))
    orig_y, x = first_img.shape
    z = len(list(first_ch.glob("Z*.tif")))

    # Save metadata
    crop_meta = metadata.copy() if metadata else {}
    crop_meta['crop_y_end'] = y_crop  # Crop line position (everything before this is kept)
    crop_meta['crop_y_original'] = orig_y
    crop_meta['crop_y_new'] = y_crop  # New height = y_crop (kept top portion)
    crop_meta['crop_method'] = 'napari_plugin_manual'
    crop_meta['dimensions_cropped'] = {
        'z': z,
        'y': y_crop,  # New Y dimension
        'x': x,
    }

    with open(output_folder / "metadata.json", 'w') as f:
        json.dump(crop_meta, f, indent=2)


# =============================================================================
# WIDGET
# =============================================================================

class ManualCropWidget(QWidget):
    """Widget for manual brain cropping - works with loaded napari layers."""

    def __init__(self, napari_viewer: napari.Viewer, pipeline_folder: Optional[Path] = None, metadata: Optional[dict] = None):
        super().__init__()
        self.viewer = napari_viewer
        self.crop_line_layer = None
        self.active_image = None
        self.metadata = metadata
        self.pipeline_folder = Path(pipeline_folder) if pipeline_folder else None
        self.crop_worker = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout with scroll area
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Title
        title = QLabel("Manual Crop Tool")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Brain selector section
        brain_group = QGroupBox("1. Select Brain")
        brain_layout = QVBoxLayout()
        brain_group.setLayout(brain_layout)

        # Dropdown for available brains
        selector_layout = QHBoxLayout()
        self.brain_combo = QComboBox()
        self.brain_combo.setMinimumWidth(200)
        self.brain_combo.currentIndexChanged.connect(self._on_brain_selected)
        selector_layout.addWidget(self.brain_combo, stretch=1)

        # Refresh button
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(30)
        self.refresh_btn.setToolTip("Refresh brain list")
        self.refresh_btn.clicked.connect(self._populate_brain_list)
        selector_layout.addWidget(self.refresh_btn)

        brain_layout.addLayout(selector_layout)

        # Browse button
        self.browse_btn = QPushButton("Browse for Folder...")
        self.browse_btn.clicked.connect(self._browse_for_brain)
        brain_layout.addWidget(self.browse_btn)

        # Load button
        self.load_btn = QPushButton("Load Selected Brain")
        self.load_btn.clicked.connect(self._load_selected_brain)
        self.load_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.load_btn.setEnabled(False)
        brain_layout.addWidget(self.load_btn)

        layout.addWidget(brain_group)

        # Populate the brain list
        self._populate_brain_list()

        # Instructions
        instructions = QLabel(
            "2. Identify which channel has cells (signal)\n"
            "3. Click 'Add Crop Line' and adjust Y position\n"
            "4. Click 'Apply Crop' to save"
        )
        instructions.setStyleSheet("color: #666; margin: 10px 0;")
        layout.addWidget(instructions)

        # Channel identification - KEY FEATURE
        channel_group = QGroupBox("Signal Channel (which has the cells?)")
        channel_layout = QVBoxLayout()
        channel_group.setLayout(channel_layout)

        self.channel_button_group = QButtonGroup(self)
        self.ch0_radio = QRadioButton("ch0 is signal (default)")
        self.ch1_radio = QRadioButton("ch1 is signal (swapped)")
        self.channel_button_group.addButton(self.ch0_radio, 0)
        self.channel_button_group.addButton(self.ch1_radio, 1)
        self.ch0_radio.setChecked(True)

        channel_layout.addWidget(QLabel("Look at the loaded channels - which one shows bright cell bodies?"))
        channel_layout.addWidget(self.ch0_radio)
        channel_layout.addWidget(self.ch1_radio)

        self.channel_help = QLabel("Tip: Toggle layer visibility to compare ch0 vs ch1")
        self.channel_help.setStyleSheet("color: #888; font-size: 9pt; font-style: italic;")
        channel_layout.addWidget(self.channel_help)

        layout.addWidget(channel_group)

        # Add crop line button
        self.add_line_btn = QPushButton("Add Crop Line")
        self.add_line_btn.clicked.connect(self._add_crop_line)
        self.add_line_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.add_line_btn)

        # Y position controls
        position_layout = QVBoxLayout()
        position_layout.addWidget(QLabel("Crop Position (Y):"))

        # Spinbox and +/- buttons
        controls_layout = QHBoxLayout()

        self.minus_btn = QPushButton("-10")
        self.minus_btn.clicked.connect(lambda: self._adjust_crop(-10))
        self.minus_btn.setEnabled(False)
        controls_layout.addWidget(self.minus_btn)

        self.minus_small_btn = QPushButton("-1")
        self.minus_small_btn.clicked.connect(lambda: self._adjust_crop(-1))
        self.minus_small_btn.setEnabled(False)
        controls_layout.addWidget(self.minus_small_btn)

        self.y_spinbox = QSpinBox()
        self.y_spinbox.setMinimum(0)
        self.y_spinbox.setMaximum(10000)
        self.y_spinbox.setValue(0)
        self.y_spinbox.setEnabled(False)
        self.y_spinbox.valueChanged.connect(self._on_spinbox_changed)
        controls_layout.addWidget(self.y_spinbox)

        self.plus_small_btn = QPushButton("+1")
        self.plus_small_btn.clicked.connect(lambda: self._adjust_crop(1))
        self.plus_small_btn.setEnabled(False)
        controls_layout.addWidget(self.plus_small_btn)

        self.plus_btn = QPushButton("+10")
        self.plus_btn.clicked.connect(lambda: self._adjust_crop(10))
        self.plus_btn.setEnabled(False)
        controls_layout.addWidget(self.plus_btn)

        position_layout.addLayout(controls_layout)
        layout.addLayout(position_layout)

        # Info label
        self.info_label = QLabel("Load an image and add crop line to start")
        self.info_label.setStyleSheet("color: gray; margin: 10px 0;")
        layout.addWidget(self.info_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        self.log_output.setVisible(False)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; font-size: 9pt;")
        layout.addWidget(self.log_output)

        # Apply button
        self.apply_btn = QPushButton("Apply Crop")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_crop)
        self.apply_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #4CAF50; color: white;")
        layout.addWidget(self.apply_btn)

        layout.addStretch()

    def _populate_brain_list(self):
        """Scan for available brains and populate the dropdown."""
        import time
        t0 = time.time()

        self.brain_combo.clear()
        self.brain_combo.addItem("-- Select a brain --", None)

        brains_found = []

        # CRITICAL: Force Y: drive to avoid UNC path slowness
        scan_path = DEFAULT_BRAINGLOBE_ROOT
        y_drive_path = Path(r"Y:\2_Connectome\Tissue\3D_Cleared\1_Brains")
        if y_drive_path.exists():
            scan_path = y_drive_path

        print(f"[{time.time()-t0:.2f}s] Scanning brains in: {scan_path}")

        try:
            # Use glob pattern to find all pipeline folders at once (faster than nested iterdir)
            # Pattern: mouse_folder/pipeline_folder
            all_folders = list(scan_path.glob("*/*"))
            print(f"[{time.time()-t0:.2f}s] Found {len(all_folders)} potential brain folders")

            for pipeline_folder in all_folders:
                if not pipeline_folder.is_dir() or pipeline_folder.name.startswith('.'):
                    continue

                # Quick check: does extracted folder exist?
                extracted_folder = pipeline_folder / FOLDER_EXTRACTED
                if not extracted_folder.exists():
                    continue

                # Quick check: does it have any channel folders with TIFFs?
                # Just check for one TIFF file existence (fast)
                has_data = any((extracted_folder / f"ch{i}").exists() for i in [0, 1])
                if not has_data:
                    continue

                # Check crop status (quick existence check)
                cropped_folder = pipeline_folder / FOLDER_CROPPED
                has_crop = cropped_folder.exists() and (cropped_folder / "ch0").exists()
                status = "✓ cropped" if has_crop else "○ needs crop"
                brains_found.append((pipeline_folder.name, pipeline_folder, status))

            print(f"[{time.time()-t0:.2f}s] Found {len(brains_found)} brains with extracted data")

        except Exception as e:
            notifications.show_warning(f"Error scanning brains: {e}")

        # Sort by name and add to combo
        brains_found.sort(key=lambda x: x[0])
        for name, path, status in brains_found:
            display = f"{name} [{status}]"
            self.brain_combo.addItem(display, path)

        if len(brains_found) == 0:
            self.brain_combo.addItem("No brains found with extracted data", None)

    def _on_brain_selected(self, index):
        """Handle brain selection change."""
        path = self.brain_combo.currentData()
        self.load_btn.setEnabled(path is not None)

    def _browse_for_brain(self):
        """Open file dialog to browse for a brain folder."""
        # Use Y: drive to avoid UNC slowness
        browse_start = Path(r"Y:\2_Connectome\Tissue\3D_Cleared\1_Brains")
        if not browse_start.exists():
            browse_start = DEFAULT_BRAINGLOBE_ROOT

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Brain Pipeline Folder (contains 1_Extracted_Full)",
            str(browse_start)
        )
        if folder:
            folder_path = Path(folder)

            # Validate it has extracted data
            extracted = folder_path / FOLDER_EXTRACTED
            if not extracted.exists():
                # Maybe they selected the extracted folder directly
                if folder_path.name == FOLDER_EXTRACTED:
                    folder_path = folder_path.parent
                    extracted = folder_path / FOLDER_EXTRACTED
                elif (folder_path / "ch0").exists():
                    # They selected a channel folder
                    folder_path = folder_path.parent.parent
                    extracted = folder_path / FOLDER_EXTRACTED

            if not extracted.exists():
                QMessageBox.warning(
                    self,
                    "Invalid Folder",
                    f"Selected folder doesn't contain extracted data.\n\n"
                    f"Expected to find: {FOLDER_EXTRACTED}/ch0/Z*.tif\n\n"
                    f"Run extraction first:\n"
                    f"python 2_extract_and_analyze.py"
                )
                return

            # Add to combo if not already there
            found = False
            for i in range(self.brain_combo.count()):
                if self.brain_combo.itemData(i) == folder_path:
                    self.brain_combo.setCurrentIndex(i)
                    found = True
                    break

            if not found:
                self.brain_combo.addItem(f"{folder_path.name} [browsed]", folder_path)
                self.brain_combo.setCurrentIndex(self.brain_combo.count() - 1)

    def _load_selected_brain(self):
        """Load the selected brain into napari."""
        path = self.brain_combo.currentData()
        if not path:
            notifications.show_warning("Please select a brain first.")
            return

        print(f"[DEBUG] Loading brain from combo: {self.brain_combo.currentText()}")
        print(f"[DEBUG] Path from combo data: {path}")

        self.pipeline_folder = path
        extracted_folder = path / FOLDER_EXTRACTED
        print(f"[DEBUG] Extracted folder: {extracted_folder}")
        print(f"[DEBUG] Extracted folder exists: {extracted_folder.exists()}")

        # Try to load metadata
        meta_path = extracted_folder / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)

        # Clear existing layers
        self.viewer.layers.clear()

        # Load channels
        self.load_btn.setEnabled(False)
        self.load_btn.setText("Loading...")

        try:
            import tifffile
            import dask
            import dask.array as da

            ch_folders = sorted([d for d in extracted_folder.iterdir()
                               if d.is_dir() and d.name.startswith("ch")])

            for ch_folder in ch_folders:
                tiff_files = sorted(ch_folder.glob("Z*.tif"))
                if not tiff_files:
                    continue

                # Use dask for lazy loading
                sample = tifffile.imread(str(tiff_files[0]))
                dtype = sample.dtype

                lazy_arrays = [
                    da.from_delayed(
                        dask.delayed(tifffile.imread)(str(f)),
                        shape=sample.shape,
                        dtype=dtype
                    ) for f in tiff_files
                ]
                volume = da.stack(lazy_arrays, axis=0)

                self.viewer.add_image(
                    volume,
                    name=ch_folder.name,
                    blending='additive'
                )

            notifications.show_info(f"Loaded {path.name}")

        except Exception as e:
            notifications.show_error(f"Error loading brain: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.load_btn.setEnabled(True)
            self.load_btn.setText("Load Selected Brain")

    def _add_crop_line(self):
        """Add crop line to the viewer based on active image layer."""
        # Find active image layer
        if not self.viewer.layers:
            notifications.show_warning("No image loaded. Please load a brain first.")
            return

        # Get the selected layer or first image layer
        image_layer = self.viewer.layers.selection.active if self.viewer.layers.selection.active else None

        if not image_layer or not isinstance(image_layer, napari.layers.Image):
            # Find first image layer
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    image_layer = layer
                    break

        if not image_layer:
            notifications.show_warning("No image layer found. Please load a brain first.")
            return

        self.active_image = image_layer
        shape = image_layer.data.shape

        # Try to get metadata
        self.metadata, self.pipeline_folder = find_pipeline_folder_from_layer(image_layer)

        # Determine dimensions (handle 3D and 4D data)
        if len(shape) == 3:
            z, y, x = shape
        elif len(shape) == 4:
            _, z, y, x = shape
        else:
            notifications.show_error(f"Unexpected image dimensions: {shape}")
            return

        # Set up spinbox
        self.y_spinbox.setMaximum(y - 1)
        self.y_spinbox.setValue(y // 2)
        self.y_spinbox.setEnabled(True)

        # Enable controls
        self.minus_btn.setEnabled(True)
        self.minus_small_btn.setEnabled(True)
        self.plus_btn.setEnabled(True)
        self.plus_small_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)

        # Create crop line
        self._update_crop_line(y // 2, shape)

        notifications.show_info("Crop line added! Adjust position with +/- buttons.")

    def _update_crop_line(self, y_pos, shape):
        """Update or create the crop line at the given Y position."""
        if len(shape) == 3:
            z, y, x = shape
        elif len(shape) == 4:
            _, z, y, x = shape
        else:
            return

        # Remove old line
        if self.crop_line_layer and self.crop_line_layer in self.viewer.layers:
            self.viewer.layers.remove(self.crop_line_layer)

        # Create a 3D image plane at the crop position
        # This works better with napari's multiscale rendering than shapes
        thickness = 200  # Make it very thick to be visible
        y_start = max(0, y_pos - thickness//2)
        y_end = min(y-1, y_pos + thickness//2)

        # Create a binary mask volume with just the crop plane lit up
        crop_plane = np.zeros((z, y, x), dtype=np.uint8)
        crop_plane[:, y_start:y_end, :] = 255

        self.crop_line_layer = self.viewer.add_image(
            crop_plane,
            name='Crop Line (everything ABOVE is kept)',
            colormap='yellow',
            opacity=0.5,
            blending='additive',
            scale=self.active_image.scale if hasattr(self.active_image, 'scale') else None
        )

        # Update info
        rows_kept = y_pos  # Keeping everything from 0 to y_pos (top portion)
        percent_kept = 100 * rows_kept / y
        self.info_label.setText(
            f"Crop at Y={y_pos} | Keep {rows_kept}/{y} rows ({percent_kept:.1f}%)"
        )

    def _adjust_crop(self, delta):
        """Adjust crop position by delta."""
        if not self.active_image:
            return

        new_val = self.y_spinbox.value() + delta
        new_val = max(0, min(new_val, self.y_spinbox.maximum()))
        self.y_spinbox.setValue(new_val)

    def _on_spinbox_changed(self, value):
        """Handle spinbox value change."""
        if not self.active_image:
            return

        shape = self.active_image.data.shape
        self._update_crop_line(value, shape)

    def _apply_crop(self):
        """Apply crop - either start background worker or save position for CLI."""
        if not self.active_image:
            notifications.show_error("No image loaded.")
            return

        # If pipeline folder not set, ask user to select it
        if not self.pipeline_folder:
            # Use Y: drive to avoid UNC slowness
            browse_start = Path(r"Y:\2_Connectome\Tissue\3D_Cleared\1_Brains")
            if not browse_start.exists():
                browse_start = DEFAULT_BRAINGLOBE_ROOT

            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Brain Pipeline Folder (contains 1_Extracted_Full)",
                str(browse_start)
            )
            if not folder:
                return

            folder_path = Path(folder)

            # Validate the selected folder has extracted data
            extracted = folder_path / FOLDER_EXTRACTED
            if not extracted.exists():
                # Maybe they selected the mouse folder - check for pipeline subfolder
                subfolders = [d for d in folder_path.iterdir() if d.is_dir()]
                for sub in subfolders:
                    if (sub / FOLDER_EXTRACTED).exists():
                        folder_path = sub
                        extracted = folder_path / FOLDER_EXTRACTED
                        print(f"[DEBUG] Auto-corrected to subfolder: {folder_path}")
                        break

            if not extracted.exists():
                QMessageBox.critical(
                    self,
                    "Invalid Folder",
                    f"Selected folder doesn't contain extracted data.\n\n"
                    f"Expected: {folder_path / FOLDER_EXTRACTED}\n\n"
                    f"Make sure you select the pipeline folder that contains "
                    f"1_Extracted_Full/ch0/Z*.tif"
                )
                return

            self.pipeline_folder = folder_path
            print(f"[DEBUG] Set pipeline folder from dialog: {self.pipeline_folder}")

            # Try to load metadata
            meta_path = self.pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)

        y_pos = self.y_spinbox.value()
        shape = self.active_image.data.shape

        if len(shape) == 3:
            _, y, _ = shape
        elif len(shape) == 4:
            _, _, y, _ = shape
        else:
            return

        rows_kept = y_pos  # Keeping everything from 0 to y_pos (top portion)
        source_folder = self.pipeline_folder / FOLDER_EXTRACTED
        output_folder = self.pipeline_folder / FOLDER_CROPPED

        # VALIDATION: Check source folder exists before proceeding
        print(f"[DEBUG] Pipeline folder: {self.pipeline_folder}")
        print(f"[DEBUG] Source folder: {source_folder}")
        if not source_folder.exists():
            QMessageBox.critical(
                self,
                "Source Folder Not Found",
                f"Cannot find extracted data at:\n{source_folder}\n\n"
                f"Pipeline folder: {self.pipeline_folder}\n\n"
                f"Make sure you selected the correct brain folder."
            )
            return

        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm Crop Position",
            f"Use crop position Y={y_pos}?\n\n"
            f"This will keep {rows_kept} of {y} rows ({100*rows_kept/y:.1f}%).\n\n"
            f"Output: {output_folder}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Disable controls
        self.apply_btn.setEnabled(False)
        self.add_line_btn.setEnabled(False)
        self.minus_btn.setEnabled(False)
        self.minus_small_btn.setEnabled(False)
        self.plus_btn.setEnabled(False)
        self.plus_small_btn.setEnabled(False)
        self.y_spinbox.setEnabled(False)

        # Show progress UI
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_output.setVisible(True)
        self.log_output.clear()
        self.log_output.append("Starting crop operation...")

        # Get channel assignment
        signal_channel = self.channel_button_group.checkedId()  # 0 = ch0, 1 = ch1
        channels_swapped = (signal_channel == 1)

        # Save crop position to temp file
        crop_file = self.pipeline_folder / ".crop_position.json"
        with open(crop_file, 'w') as f:
            json.dump({
                'y_crop': y_pos,
                'rows_kept': rows_kept,
                'total_rows': y,
                'timestamp': time.time(),
                'signal_channel': f"ch{signal_channel}",
                'channels_swapped': channels_swapped,
            }, f, indent=2)

        # Spawn subprocess to do the cropping (so it continues after napari closes)
        import subprocess
        import sys
        crop_script = Path(__file__).parent / "crop_subprocess.py"

        # Launch in a new terminal window so user can see progress
        if sys.platform == 'win32':
            # Windows: open new cmd window
            subprocess.Popen([
                'cmd', '/c', 'start', 'cmd', '/k',
                sys.executable, str(crop_script), str(self.pipeline_folder)
            ], shell=True)
        else:
            # Unix: try to open in new terminal
            subprocess.Popen([
                'x-terminal-emulator', '-e',
                sys.executable, str(crop_script), str(self.pipeline_folder)
            ])

        # Close napari
        self.viewer.close()

    def _on_crop_progress(self, current, total, message):
        """Update progress bar and log."""
        percent = int(100 * current / total) if total > 0 else 0
        self.progress.setValue(percent)
        self.log_output.append(message)
        # Auto-scroll to bottom
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def _on_crop_finished(self, success, message):
        """Handle crop completion."""
        if success:
            self.log_output.append(f"\n✓ Crop complete!")
            self.log_output.append(f"Output: {message}")
            self.progress.setValue(100)

            QMessageBox.information(
                self,
                "Crop Complete",
                f"Crop applied successfully!\n\n"
                f"Output saved to:\n{message}\n\n"
                f"Napari will now close."
            )
            # Close napari after successful crop
            self.viewer.close()
        else:
            self.log_output.append(f"\n✗ Error: {message}")
            notifications.show_error(f"Crop failed: {message}")

            # Re-enable controls
            self.apply_btn.setEnabled(True)
            self.add_line_btn.setEnabled(True)
            self.minus_btn.setEnabled(True)
            self.minus_small_btn.setEnabled(True)
            self.plus_btn.setEnabled(True)
            self.plus_small_btn.setEnabled(True)
            self.y_spinbox.setEnabled(True)


# =============================================================================
# NAPARI PLUGIN REGISTRATION
# =============================================================================

def napari_experimental_provide_dock_widget():
    """Provide the widget to napari."""
    return ManualCropWidget
