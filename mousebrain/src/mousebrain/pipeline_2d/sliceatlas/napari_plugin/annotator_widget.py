"""
main_widget.py - Main Slice Annotator widget for napari.

Provides the central tabbed interface for:
- Loading ND2 files
- Managing channels (color, contrast, gamma, opacity)
- Creating annotations (shapes, text, callouts)
- Exporting to TIFF with flattened visualization settings

Usage:
    # In napari
    Plugins -> Slice Annotator
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QGroupBox, QFormLayout, QLineEdit, QScrollArea,
    QFrame, QSplitter, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QColorDialog,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor

import napari

from mousebrain.pipeline_2d.sliceatlas.napari_plugin.annotator_worker import ND2LoaderWorker
from mousebrain.pipeline_2d.sliceatlas.core.channel_model import ChannelSettings, create_default_channel_settings, DEFAULT_CHANNEL_COLORS
from mousebrain.pipeline_2d.sliceatlas.core.image_utils import auto_contrast, COLORMAPS


class SliceAnnotatorWidget(QWidget):
    """
    Main widget for Slice Annotator napari plugin.

    Provides tabbed interface for loading, channel management, annotation, and export.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # State
        self.current_file: Optional[Path] = None
        self.current_data = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.channel_settings: List[ChannelSettings] = []
        self.channel_layers: List[napari.layers.Image] = []
        self.loader_worker: Optional[ND2LoaderWorker] = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with file info
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.file_label)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self._create_load_tab()
        self._create_channels_tab()
        self._create_annotate_tab()
        self._create_export_tab()

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _create_load_tab(self):
        """Create the Load tab for file selection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        # Load single file button
        btn_layout = QHBoxLayout()
        self.load_file_btn = QPushButton("Load ND2 File...")
        self.load_file_btn.clicked.connect(self._on_load_file)
        btn_layout.addWidget(self.load_file_btn)

        self.load_folder_btn = QPushButton("Load Folder...")
        self.load_folder_btn.clicked.connect(self._on_load_folder)
        btn_layout.addWidget(self.load_folder_btn)
        file_layout.addLayout(btn_layout)

        # Recent files (placeholder for future)
        file_layout.addWidget(QLabel("Recent Files:"))
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(100)
        self.recent_list.itemDoubleClicked.connect(self._on_recent_clicked)
        file_layout.addWidget(self.recent_list)

        layout.addWidget(file_group)

        # Load options group
        options_group = QGroupBox("Load Options")
        options_layout = QFormLayout(options_group)

        self.lazy_checkbox = QCheckBox()
        self.lazy_checkbox.setChecked(True)
        self.lazy_checkbox.setToolTip("Use lazy loading for large files (recommended)")
        options_layout.addRow("Lazy Loading:", self.lazy_checkbox)

        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["None (Full Stack)", "Max Projection", "Mean Projection", "Min Projection"])
        self.projection_combo.setToolTip("Z-projection mode (None keeps full z-stack)")
        options_layout.addRow("Z-Projection:", self.projection_combo)

        layout.addWidget(options_group)

        # File info display
        info_group = QGroupBox("File Information")
        info_layout = QFormLayout(info_group)

        self.info_shape = QLabel("-")
        self.info_channels = QLabel("-")
        self.info_voxel = QLabel("-")
        self.info_dtype = QLabel("-")

        info_layout.addRow("Shape:", self.info_shape)
        info_layout.addRow("Channels:", self.info_channels)
        info_layout.addRow("Voxel Size:", self.info_voxel)
        info_layout.addRow("Data Type:", self.info_dtype)

        layout.addWidget(info_group)

        # Transform group (rotation, flip)
        transform_group = QGroupBox("Transform")
        transform_layout = QVBoxLayout(transform_group)

        # Rotation row
        rotate_layout = QHBoxLayout()
        rotate_label = QLabel("Rotate:")
        rotate_label.setStyleSheet("color: #DDD;")
        rotate_layout.addWidget(rotate_label)

        self.rotate_ccw_btn = QPushButton("↺ 90°")
        self.rotate_ccw_btn.setToolTip("Rotate 90° counter-clockwise")
        self.rotate_ccw_btn.clicked.connect(lambda: self._rotate_data(-90))
        rotate_layout.addWidget(self.rotate_ccw_btn)

        self.rotate_cw_btn = QPushButton("↻ 90°")
        self.rotate_cw_btn.setToolTip("Rotate 90° clockwise")
        self.rotate_cw_btn.clicked.connect(lambda: self._rotate_data(90))
        rotate_layout.addWidget(self.rotate_cw_btn)

        self.rotate_180_btn = QPushButton("180°")
        self.rotate_180_btn.setToolTip("Rotate 180°")
        self.rotate_180_btn.clicked.connect(lambda: self._rotate_data(180))
        rotate_layout.addWidget(self.rotate_180_btn)

        rotate_layout.addStretch()
        transform_layout.addLayout(rotate_layout)

        # Flip row
        flip_layout = QHBoxLayout()
        flip_label = QLabel("Flip:")
        flip_label.setStyleSheet("color: #DDD;")
        flip_layout.addWidget(flip_label)

        self.flip_h_btn = QPushButton("↔ Horizontal")
        self.flip_h_btn.setToolTip("Flip horizontally (left-right)")
        self.flip_h_btn.clicked.connect(lambda: self._flip_data('horizontal'))
        flip_layout.addWidget(self.flip_h_btn)

        self.flip_v_btn = QPushButton("↕ Vertical")
        self.flip_v_btn.setToolTip("Flip vertically (up-down)")
        self.flip_v_btn.clicked.connect(lambda: self._flip_data('vertical'))
        flip_layout.addWidget(self.flip_v_btn)

        flip_layout.addStretch()
        transform_layout.addLayout(flip_layout)

        layout.addWidget(transform_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Load")

    def _create_channels_tab(self):
        """Create the Channels tab for visualization settings."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scroll area for channel list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout(self.channels_container)
        self.channels_layout.setAlignment(Qt.AlignTop)

        # Placeholder
        self.no_channels_label = QLabel("Load a file to see channels")
        self.no_channels_label.setStyleSheet("color: gray; padding: 20px;")
        self.channels_layout.addWidget(self.no_channels_label)

        scroll.setWidget(self.channels_container)
        layout.addWidget(scroll)

        # Global controls
        controls_layout = QHBoxLayout()

        self.auto_contrast_btn = QPushButton("Auto Contrast All")
        self.auto_contrast_btn.clicked.connect(self._on_auto_contrast_all)
        controls_layout.addWidget(self.auto_contrast_btn)

        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self._on_reset_all)
        controls_layout.addWidget(self.reset_btn)

        layout.addLayout(controls_layout)

        # Blending mode
        blend_layout = QHBoxLayout()
        blend_label = QLabel("Blending:")
        blend_label.setStyleSheet("color: #DDD;")
        blend_layout.addWidget(blend_label)
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["Additive", "Translucent"])
        self.blend_combo.currentTextChanged.connect(self._on_blending_changed)
        blend_layout.addWidget(self.blend_combo)
        blend_layout.addStretch()
        layout.addLayout(blend_layout)

        self.tabs.addTab(tab, "Channels")

    def _create_annotate_tab(self):
        """Create the Annotate tab for drawing tools."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tool palette
        tools_group = QGroupBox("Drawing Tools")
        tools_layout = QVBoxLayout(tools_group)

        # Shape tools row
        shapes_layout = QHBoxLayout()
        self.rect_btn = QPushButton("Rectangle")
        self.rect_btn.clicked.connect(lambda: self._add_shapes_layer('rectangle'))
        shapes_layout.addWidget(self.rect_btn)

        self.ellipse_btn = QPushButton("Ellipse")
        self.ellipse_btn.clicked.connect(lambda: self._add_shapes_layer('ellipse'))
        shapes_layout.addWidget(self.ellipse_btn)

        self.polygon_btn = QPushButton("Polygon")
        self.polygon_btn.clicked.connect(lambda: self._add_shapes_layer('polygon'))
        shapes_layout.addWidget(self.polygon_btn)

        tools_layout.addLayout(shapes_layout)

        # Line tools row
        lines_layout = QHBoxLayout()
        self.line_btn = QPushButton("Line")
        self.line_btn.clicked.connect(lambda: self._add_shapes_layer('line'))
        lines_layout.addWidget(self.line_btn)

        self.arrow_btn = QPushButton("Arrow")
        self.arrow_btn.clicked.connect(lambda: self._add_shapes_layer('path'))
        lines_layout.addWidget(self.arrow_btn)

        self.freehand_btn = QPushButton("Freehand")
        self.freehand_btn.clicked.connect(lambda: self._add_shapes_layer('path'))
        lines_layout.addWidget(self.freehand_btn)

        tools_layout.addLayout(lines_layout)

        layout.addWidget(tools_group)

        # Style options
        style_group = QGroupBox("Style")
        style_layout = QFormLayout(style_group)

        # Edge color
        self.edge_color_btn = QPushButton()
        self.edge_color_btn.setStyleSheet("background-color: #FFFF00; min-width: 60px;")
        self.edge_color_btn.clicked.connect(self._pick_edge_color)
        self._current_edge_color = QColor(255, 255, 0)
        style_layout.addRow("Edge Color:", self.edge_color_btn)

        # Edge width
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.5, 20.0)
        self.edge_width_spin.setValue(2.0)
        self.edge_width_spin.setSingleStep(0.5)
        style_layout.addRow("Edge Width:", self.edge_width_spin)

        # Face color (optional fill)
        self.fill_checkbox = QCheckBox("Fill shapes")
        self.fill_checkbox.setChecked(False)
        style_layout.addRow("", self.fill_checkbox)

        layout.addWidget(style_group)

        # Text annotation
        text_group = QGroupBox("Text Annotations")
        text_layout = QVBoxLayout(text_group)

        # Text input field
        text_input_layout = QHBoxLayout()
        text_input_label = QLabel("Text:")
        text_input_label.setStyleSheet("color: #DDD;")
        text_input_layout.addWidget(text_input_label)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter annotation text...")
        self.text_input.setText("Label")
        text_input_layout.addWidget(self.text_input)
        text_layout.addLayout(text_input_layout)

        # Font size and color row
        font_layout = QHBoxLayout()
        font_label = QLabel("Size:")
        font_label.setStyleSheet("color: #DDD;")
        font_layout.addWidget(font_label)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 72)
        self.font_size_spin.setValue(14)
        font_layout.addWidget(self.font_size_spin)

        color_label = QLabel("Color:")
        color_label.setStyleSheet("color: #DDD;")
        font_layout.addWidget(color_label)
        self.text_color_btn = QPushButton()
        self.text_color_btn.setStyleSheet("background-color: #FFFF00; min-width: 40px;")
        self.text_color_btn.clicked.connect(self._pick_text_color)
        self._current_text_color = QColor(255, 255, 0)
        font_layout.addWidget(self.text_color_btn)

        font_layout.addStretch()
        text_layout.addLayout(font_layout)

        # Text annotation buttons
        text_btn_layout = QHBoxLayout()

        self.add_text_box_btn = QPushButton("Text Box")
        self.add_text_box_btn.setToolTip("Draw a rectangle, text appears inside")
        self.add_text_box_btn.clicked.connect(self._add_text_box)
        text_btn_layout.addWidget(self.add_text_box_btn)

        self.add_text_btn = QPushButton("Text Label")
        self.add_text_btn.setToolTip("Click to place text at location")
        self.add_text_btn.clicked.connect(self._add_text_label)
        text_btn_layout.addWidget(self.add_text_btn)

        text_layout.addLayout(text_btn_layout)

        layout.addWidget(text_group)

        # Annotation list
        list_group = QGroupBox("Annotations")
        list_layout = QVBoxLayout(list_group)

        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(150)
        list_layout.addWidget(self.annotation_list)

        # Delete button
        self.delete_annotation_btn = QPushButton("Delete Selected")
        self.delete_annotation_btn.clicked.connect(self._delete_selected_annotation)
        list_layout.addWidget(self.delete_annotation_btn)

        layout.addWidget(list_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Annotate")

    def _create_export_tab(self):
        """Create the Export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Export scope
        scope_group = QGroupBox("Export Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems([
            "Current Slice Only",
            "All Slices (Individual Files)",
            "All Slices (Stack)",
            "Max Intensity Projection",
        ])
        scope_layout.addWidget(self.scope_combo)

        layout.addWidget(scope_group)

        # Format options
        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout(format_group)

        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "Composite View (Current Settings)",
            "RGB Composite (8-bit)",
            "16-bit Per Channel (Separate Files)",
        ])
        self.format_combo.setToolTip(
            "Composite View: Exact screen capture with annotations\n"
            "RGB Composite: Programmatic composite of visible channels\n"
            "16-bit: Raw channel data in separate files"
        )
        format_layout.addWidget(self.format_combo)

        self.include_annotations_cb = QCheckBox("Include annotations")
        self.include_annotations_cb.setChecked(True)
        format_layout.addWidget(self.include_annotations_cb)

        layout.addWidget(format_group)

        # Scale bar options
        scalebar_group = QGroupBox("Scale Bar")
        scalebar_layout = QFormLayout(scalebar_group)

        self.scalebar_cb = QCheckBox()
        self.scalebar_cb.setChecked(True)
        scalebar_layout.addRow("Add Scale Bar:", self.scalebar_cb)

        self.scalebar_size_spin = QSpinBox()
        self.scalebar_size_spin.setRange(1, 10000)
        self.scalebar_size_spin.setValue(100)
        self.scalebar_size_spin.setSuffix(" um")
        scalebar_layout.addRow("Length:", self.scalebar_size_spin)

        self.scalebar_position_combo = QComboBox()
        self.scalebar_position_combo.addItems(["Bottom Right", "Bottom Left", "Top Right", "Top Left"])
        scalebar_layout.addRow("Position:", self.scalebar_position_combo)

        layout.addWidget(scalebar_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        folder_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Select output folder...")
        folder_layout.addWidget(self.output_folder_edit)

        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self._browse_output_folder)
        folder_layout.addWidget(self.browse_output_btn)

        output_layout.addLayout(folder_layout)

        # Filename pattern
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Filename:"))
        self.filename_pattern_edit = QLineEdit()
        self.filename_pattern_edit.setText("{filename}_{slice:03d}")
        self.filename_pattern_edit.setToolTip("Use {filename}, {slice}, {channel} placeholders")
        pattern_layout.addWidget(self.filename_pattern_edit)
        output_layout.addLayout(pattern_layout)

        layout.addWidget(output_group)

        # Export button
        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.export_btn.clicked.connect(self._on_export)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        self.tabs.addTab(tab, "Export")

    # =========================================================================
    # Load tab handlers
    # =========================================================================

    def _on_load_file(self):
        """Handle Load ND2 File button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ND2 File",
            "",
            "ND2 Files (*.nd2);;All Files (*.*)",
        )

        if file_path:
            self._load_file(Path(file_path))

    def _on_load_folder(self):
        """Handle Load Folder button click."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with ND2 Files",
        )

        if folder_path:
            self._load_folder(Path(folder_path))

    def _on_recent_clicked(self, item: QListWidgetItem):
        """Handle double-click on recent file."""
        path = Path(item.data(Qt.UserRole))
        if path.exists():
            self._load_file(path)

    def _load_file(self, file_path: Path):
        """Load an ND2 file."""
        # Cancel any existing load
        if self.loader_worker and self.loader_worker.isRunning():
            self.loader_worker.cancel()
            self.loader_worker.wait()

        # Get options
        lazy = self.lazy_checkbox.isChecked()
        projection_text = self.projection_combo.currentText()
        z_projection = None
        if "Max" in projection_text:
            z_projection = 'max'
        elif "Mean" in projection_text:
            z_projection = 'mean'
        elif "Min" in projection_text:
            z_projection = 'min'

        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.load_file_btn.setEnabled(False)
        self.status_label.setText(f"Loading {file_path.name}...")

        # Start worker
        self.loader_worker = ND2LoaderWorker(file_path, lazy=lazy, z_projection=z_projection)
        self.loader_worker.progress.connect(self._on_load_progress)
        self.loader_worker.finished.connect(self._on_load_finished)
        self.loader_worker.start()

    def _load_folder(self, folder_path: Path):
        """Load all ND2 files from a folder as stacked tissue sections.

        Each ND2 file = one tissue section. MIP is applied to flatten optical
        z-stacks, then files are stacked so slider navigates tissue sections.
        """
        from mousebrain.pipeline_2d.sliceatlas.napari_plugin.annotator_worker import FolderLoaderWorker

        # Cancel any existing load
        if self.loader_worker and self.loader_worker.isRunning():
            self.loader_worker.cancel()
            self.loader_worker.wait()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.load_folder_btn.setEnabled(False)
        self.status_label.setText(f"Loading folder {folder_path.name}...")

        # Use max projection by default for folder loading
        self.loader_worker = FolderLoaderWorker(folder_path, z_projection='max')
        self.loader_worker.progress.connect(self._on_folder_load_progress)
        self.loader_worker.finished.connect(self._on_load_finished)
        self.loader_worker.start()

    def _on_load_progress(self, message: str):
        """Handle load progress update."""
        self.status_label.setText(message)

    def _on_folder_load_progress(self, current: int, total: int, filename: str):
        """Handle folder load progress."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Loading {current}/{total}: {filename}")

    def _on_load_finished(self, success: bool, message: str, data, metadata):
        """Handle load completion for both single files and folders."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.load_file_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)

        if not success:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.warning(self, "Load Error", message)
            return

        self.status_label.setText(message)

        # Store data
        self.current_data = data
        self.metadata = metadata

        # Handle both single file and folder loading
        # Single file has 'file_path', folder has 'folder'
        if 'file_path' in metadata:
            self.current_file = Path(metadata['file_path'])
            display_name = self.current_file.name
        elif 'folder' in metadata:
            self.current_file = Path(metadata['folder'])
            n_slices = metadata.get('n_slices', 0)
            display_name = f"{self.current_file.name}/ ({n_slices} sections)"
        else:
            self.current_file = Path('')
            display_name = "Unknown"

        # Update file label
        self.file_label.setText(f"Source: {display_name}")

        # Update info display
        self._update_file_info()

        # Add to recent files (only for single files)
        if 'file_path' in metadata:
            self._add_to_recent(self.current_file)

        # Create channel layers
        self._create_channel_layers()

        # Update channels tab
        self._update_channels_ui()

        # Set default output folder
        if not self.output_folder_edit.text():
            output_folder = self.current_file.parent if 'file_path' in metadata else self.current_file
            self.output_folder_edit.setText(str(output_folder))

    def _update_file_info(self):
        """Update file information display."""
        if not self.metadata:
            return

        shape = self.metadata.get('shape', ())
        self.info_shape.setText(' x '.join(str(s) for s in shape))

        n_channels = self.metadata.get('n_channels', 1)
        channel_names = self.metadata.get('channel_names', [])
        if channel_names:
            self.info_channels.setText(f"{n_channels}: {', '.join(channel_names)}")
        else:
            self.info_channels.setText(str(n_channels))

        voxel = self.metadata.get('voxel_size_um', {})
        voxel_str = f"X: {voxel.get('x', 1):.3f}, Y: {voxel.get('y', 1):.3f}"
        if voxel.get('z'):
            voxel_str += f", Z: {voxel.get('z', 1):.3f}"
        voxel_str += " um"
        self.info_voxel.setText(voxel_str)

        self.info_dtype.setText(self.metadata.get('dtype', 'unknown'))

    def _add_to_recent(self, file_path: Path):
        """Add file to recent files list."""
        # Check if already in list
        for i in range(self.recent_list.count()):
            item = self.recent_list.item(i)
            if item.data(Qt.UserRole) == str(file_path):
                # Move to top
                self.recent_list.takeItem(i)
                break

        # Add to top
        item = QListWidgetItem(file_path.name)
        item.setData(Qt.UserRole, str(file_path))
        self.recent_list.insertItem(0, item)

        # Keep max 10
        while self.recent_list.count() > 10:
            self.recent_list.takeItem(self.recent_list.count() - 1)

    # =========================================================================
    # Transform operations (rotate, flip)
    # =========================================================================

    def _rotate_data(self, angle: int):
        """Rotate all data by the specified angle (90, -90, or 180 degrees).

        Rotates the entire stack so all slices are transformed consistently.
        """
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first before rotating.")
            return

        # Determine number of 90° rotations (k for np.rot90)
        # Positive angle = clockwise, but np.rot90 is counter-clockwise
        # So we negate: 90° CW = k=-1, 90° CCW = k=1, 180° = k=2
        if angle == 90:
            k = -1  # Clockwise
        elif angle == -90:
            k = 1   # Counter-clockwise
        elif angle == 180:
            k = 2
        else:
            return

        self.status_label.setText(f"Rotating {angle}°...")

        # Rotate on the last two axes (Y, X) regardless of data dimensions
        # Data shapes: (C, Y, X), (N_slices, C, Y, X), etc.
        axes = (-2, -1)  # Always rotate Y, X plane
        self.current_data = np.rot90(self.current_data, k=k, axes=axes)

        # Update metadata shape
        self.metadata['shape'] = self.current_data.shape

        # Recreate layers with rotated data
        self._create_channel_layers()
        self._update_channels_ui()
        self._update_file_info()

        self.status_label.setText(f"Rotated {angle}°")

    def _flip_data(self, direction: str):
        """Flip all data horizontally or vertically.

        Args:
            direction: 'horizontal' (left-right) or 'vertical' (up-down)
        """
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first before flipping.")
            return

        self.status_label.setText(f"Flipping {direction}...")

        # Flip on the appropriate axis
        # Data shapes: (C, Y, X), (N_slices, C, Y, X), etc.
        # Y is second-to-last axis (-2), X is last axis (-1)
        if direction == 'horizontal':
            # Flip left-right = flip along X axis (last axis)
            self.current_data = np.flip(self.current_data, axis=-1)
        elif direction == 'vertical':
            # Flip up-down = flip along Y axis (second-to-last axis)
            self.current_data = np.flip(self.current_data, axis=-2)
        else:
            return

        # Recreate layers with flipped data
        self._create_channel_layers()
        self._update_channels_ui()

        self.status_label.setText(f"Flipped {direction}")

    # =========================================================================
    # Channel layer management
    # =========================================================================

    def _create_channel_layers(self):
        """Create napari layers for each channel.

        Data shapes:
        - Single file with MIP: (C, Y, X)
        - Single file without MIP: (Z_optical, C, Y, X)
        - Folder loading: (N_tissue_slices, C, Y, X) where slider = tissue sections
        """
        if self.current_data is None:
            return

        # Clear existing channel layers
        for layer in self.channel_layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.channel_layers = []

        data = self.current_data
        n_channels = self.metadata.get('n_channels', 1)
        channel_names = self.metadata.get('channel_names', [f"Ch{i}" for i in range(n_channels)])

        # Determine dtype max for contrast
        if np.issubdtype(data.dtype, np.integer):
            dtype_max = float(np.iinfo(data.dtype).max)
        else:
            dtype_max = float(np.max(data)) if np.max(data) > 0 else 1.0

        # Create default channel settings
        self.channel_settings = create_default_channel_settings(
            n_channels, channel_names, dtype_max
        )

        # Data shapes:
        # - 4D (Z/Slices, C, Y, X): folder loading or single file without MIP
        # - 3D (C, Y, X): single file with MIP applied
        # - 2D (Y, X): single channel, single slice
        if data.ndim == 4:
            # (N_slices, C, Y, X) - each position on slider is a tissue section
            for i in range(n_channels):
                ch_data = data[:, i, :, :]  # (N_slices, Y, X)
                self._add_channel_layer(ch_data, i, channel_names[i])
        elif data.ndim == 3:
            # (C, Y, X) - MIP applied, no slider needed
            for i in range(n_channels):
                ch_data = data[i, :, :]  # (Y, X)
                self._add_channel_layer(ch_data, i, channel_names[i])
        else:
            # Single channel (Y, X)
            self._add_channel_layer(data, 0, channel_names[0] if channel_names else "Channel_0")

    def _add_channel_layer(self, data: np.ndarray, index: int, name: str):
        """Add a single channel as a napari image layer."""
        settings = self.channel_settings[index]

        # Calculate auto contrast
        min_val, max_val = auto_contrast(data)
        settings.contrast_limits = (min_val, max_val)

        # Get colormap
        colormap = settings.colormap
        if colormap in ['gray', 'white']:
            colormap = 'gray'

        layer = self.viewer.add_image(
            data,
            name=name,
            colormap=colormap,
            contrast_limits=settings.contrast_limits,
            gamma=settings.gamma,
            opacity=settings.opacity,
            blending='additive',
        )

        self.channel_layers.append(layer)

    # =========================================================================
    # Channels tab handlers
    # =========================================================================

    def _update_channels_ui(self):
        """Update the channels tab with current channel settings."""
        # Clear existing widgets
        while self.channels_layout.count() > 0:
            item = self.channels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.channel_settings:
            self.no_channels_label = QLabel("Load a file to see channels")
            self.no_channels_label.setStyleSheet("color: gray; padding: 20px;")
            self.channels_layout.addWidget(self.no_channels_label)
            return

        # Create widget for each channel
        for i, settings in enumerate(self.channel_settings):
            widget = self._create_channel_widget(i, settings)
            self.channels_layout.addWidget(widget)

    def _create_channel_widget(self, index: int, settings: ChannelSettings) -> QWidget:
        """Create a widget for controlling a single channel."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.StyledPanel)
        # Use transparent background to inherit from napari theme
        widget.setStyleSheet("""
            QFrame {
                background: rgba(60, 60, 60, 150);
                border: 1px solid #555;
                border-radius: 4px;
                margin: 2px;
                padding: 5px;
            }
            QLabel {
                color: #DDD;
            }
        """)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header row: visibility checkbox + name + color
        header = QHBoxLayout()

        visible_cb = QCheckBox()
        visible_cb.setChecked(settings.visible)
        visible_cb.stateChanged.connect(lambda state, i=index: self._on_channel_visibility(i, state))
        header.addWidget(visible_cb)

        name_label = QLabel(settings.name)
        name_label.setStyleSheet("font-weight: bold; color: #FFF;")
        header.addWidget(name_label)

        header.addStretch()

        color_combo = QComboBox()
        color_combo.addItems(list(COLORMAPS.keys()))
        color_combo.setCurrentText(settings.colormap)
        color_combo.currentTextChanged.connect(lambda color, i=index: self._on_channel_color(i, color))
        header.addWidget(color_combo)

        layout.addLayout(header)

        # Contrast row
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))

        min_spin = QDoubleSpinBox()
        min_spin.setRange(0, 65535)
        min_spin.setValue(settings.contrast_limits[0])
        min_spin.setDecimals(0)
        min_spin.valueChanged.connect(lambda val, i=index: self._on_contrast_min(i, val))
        contrast_layout.addWidget(min_spin)

        contrast_layout.addWidget(QLabel("-"))

        max_spin = QDoubleSpinBox()
        max_spin.setRange(0, 65535)
        max_spin.setValue(settings.contrast_limits[1])
        max_spin.setDecimals(0)
        max_spin.valueChanged.connect(lambda val, i=index: self._on_contrast_max(i, val))
        contrast_layout.addWidget(max_spin)

        auto_btn = QPushButton("Auto")
        auto_btn.setMaximumWidth(50)
        auto_btn.clicked.connect(lambda _, i=index: self._on_auto_contrast(i))
        contrast_layout.addWidget(auto_btn)

        layout.addLayout(contrast_layout)

        # Gamma and opacity row
        gamma_layout = QHBoxLayout()

        gamma_layout.addWidget(QLabel("Gamma:"))
        gamma_spin = QDoubleSpinBox()
        gamma_spin.setRange(0.1, 3.0)
        gamma_spin.setValue(settings.gamma)
        gamma_spin.setSingleStep(0.1)
        gamma_spin.valueChanged.connect(lambda val, i=index: self._on_gamma(i, val))
        gamma_layout.addWidget(gamma_spin)

        gamma_layout.addWidget(QLabel("Opacity:"))
        opacity_spin = QDoubleSpinBox()
        opacity_spin.setRange(0.0, 1.0)
        opacity_spin.setValue(settings.opacity)
        opacity_spin.setSingleStep(0.1)
        opacity_spin.valueChanged.connect(lambda val, i=index: self._on_opacity(i, val))
        gamma_layout.addWidget(opacity_spin)

        layout.addLayout(gamma_layout)

        return widget

    def _on_channel_visibility(self, index: int, state: int):
        """Handle channel visibility toggle."""
        visible = state == Qt.Checked
        self.channel_settings[index].visible = visible
        if index < len(self.channel_layers):
            self.channel_layers[index].visible = visible

    def _on_channel_color(self, index: int, color: str):
        """Handle channel color change."""
        self.channel_settings[index].colormap = color
        if index < len(self.channel_layers):
            try:
                self.channel_layers[index].colormap = color
            except ValueError:
                self.channel_layers[index].colormap = 'gray'

    def _on_contrast_min(self, index: int, value: float):
        """Handle contrast min change."""
        settings = self.channel_settings[index]
        settings.contrast_limits = (value, settings.contrast_limits[1])
        if index < len(self.channel_layers):
            self.channel_layers[index].contrast_limits = settings.contrast_limits

    def _on_contrast_max(self, index: int, value: float):
        """Handle contrast max change."""
        settings = self.channel_settings[index]
        settings.contrast_limits = (settings.contrast_limits[0], value)
        if index < len(self.channel_layers):
            self.channel_layers[index].contrast_limits = settings.contrast_limits

    def _on_auto_contrast(self, index: int):
        """Handle auto contrast button for single channel."""
        if index >= len(self.channel_layers):
            return

        layer = self.channel_layers[index]
        min_val, max_val = auto_contrast(layer.data)

        self.channel_settings[index].contrast_limits = (min_val, max_val)
        layer.contrast_limits = (min_val, max_val)

        # Update UI
        self._update_channels_ui()

    def _on_gamma(self, index: int, value: float):
        """Handle gamma change."""
        self.channel_settings[index].gamma = value
        if index < len(self.channel_layers):
            self.channel_layers[index].gamma = value

    def _on_opacity(self, index: int, value: float):
        """Handle opacity change."""
        self.channel_settings[index].opacity = value
        if index < len(self.channel_layers):
            self.channel_layers[index].opacity = value

    def _on_auto_contrast_all(self):
        """Apply auto contrast to all channels."""
        for i in range(len(self.channel_settings)):
            self._on_auto_contrast(i)

    def _on_reset_all(self):
        """Reset all channels to defaults."""
        if not self.current_data:
            return

        n_channels = len(self.channel_settings)
        channel_names = [s.name for s in self.channel_settings]

        # Get dtype max
        if np.issubdtype(self.current_data.dtype, np.integer):
            dtype_max = float(np.iinfo(self.current_data.dtype).max)
        else:
            dtype_max = 65535.0

        self.channel_settings = create_default_channel_settings(n_channels, channel_names, dtype_max)

        # Apply to layers
        for i, settings in enumerate(self.channel_settings):
            if i < len(self.channel_layers):
                layer = self.channel_layers[i]
                layer.visible = settings.visible
                layer.contrast_limits = settings.contrast_limits
                layer.gamma = settings.gamma
                layer.opacity = settings.opacity
                try:
                    layer.colormap = settings.colormap
                except ValueError:
                    layer.colormap = 'gray'

        self._update_channels_ui()

    def _on_blending_changed(self, text: str):
        """Handle blending mode change."""
        blending = 'additive' if text == "Additive" else 'translucent'
        for layer in self.channel_layers:
            layer.blending = blending

    # =========================================================================
    # Annotate tab handlers
    # =========================================================================

    def _add_shapes_layer(self, shape_type: str):
        """Add a shapes layer for drawing on the current slice.

        Shapes are slice-aware - they're tied to the current tissue section
        and only appear on that slice.
        """
        edge_color = [
            self._current_edge_color.redF(),
            self._current_edge_color.greenF(),
            self._current_edge_color.blueF(),
            1.0,
        ]
        edge_width = self.edge_width_spin.value()
        face_color = edge_color if self.fill_checkbox.isChecked() else [0, 0, 0, 0]

        # Determine ndim from loaded data to make shapes slice-aware
        # For 4D data (N_slices, C, Y, X), shapes need 3D coords (slice, y, x)
        ndim = 2  # Default for 2D/3D single-slice data
        if self.current_data is not None and self.current_data.ndim == 4:
            ndim = 3  # Shapes need (slice, y, x) coordinates

        layer = self.viewer.add_shapes(
            name=f"Annotations ({shape_type})",
            edge_color=edge_color,
            edge_width=edge_width,
            face_color=face_color,
            ndim=ndim,
        )

        # Set the mode for the shape type
        layer.mode = 'add_' + shape_type if shape_type in ['rectangle', 'ellipse', 'polygon', 'line', 'path'] else 'add_rectangle'

        # Update annotation list
        self._update_annotation_list()

    def _pick_edge_color(self):
        """Open color picker for edge color."""
        color = QColorDialog.getColor(self._current_edge_color, self)
        if color.isValid():
            self._current_edge_color = color
            self.edge_color_btn.setStyleSheet(f"background-color: {color.name()}; min-width: 60px;")

    def _pick_text_color(self):
        """Open color picker for text color."""
        color = QColorDialog.getColor(self._current_text_color, self)
        if color.isValid():
            self._current_text_color = color
            self.text_color_btn.setStyleSheet(f"background-color: {color.name()}; min-width: 40px;")

    def _add_text_box(self):
        """Add a text box annotation (rectangle with text inside).

        Creates a Shapes layer where each rectangle displays the entered text.
        Draw a rectangle to place a text box - the text appears at the center.
        """
        import pandas as pd

        # Get the text to display
        text = self.text_input.text().strip()
        if not text:
            text = "Label"

        # Determine ndim for slice-awareness
        ndim = 2
        if self.current_data is not None and self.current_data.ndim == 4:
            ndim = 3

        # Get colors
        text_color = [
            self._current_text_color.redF(),
            self._current_text_color.greenF(),
            self._current_text_color.blueF(),
            1.0,
        ]
        # Semi-transparent background for the box
        box_color = [
            self._current_text_color.redF() * 0.3,
            self._current_text_color.greenF() * 0.3,
            self._current_text_color.blueF() * 0.3,
            0.5,
        ]

        # Create unique layer name
        existing = [l.name for l in self.viewer.layers if l.name.startswith("TextBox:")]
        layer_name = f"TextBox: {text}"
        if layer_name in existing:
            layer_name = f"TextBox: {text} ({len(existing) + 1})"

        # Create shapes layer with text display
        layer = self.viewer.add_shapes(
            data=None,
            name=layer_name,
            edge_color=text_color,
            edge_width=2,
            face_color=box_color,
            ndim=ndim,
            features={'text': []},
            text={
                'string': '{text}',
                'size': self.font_size_spin.value(),
                'color': text_color,
                'anchor': 'center',
            },
        )

        # Set to add rectangles
        layer.mode = 'add_rectangle'

        # Store default text
        layer.metadata['default_text'] = text

        # Auto-fill text when shapes are added
        @layer.events.data.connect
        def on_shapes_change(event):
            n_shapes = len(layer.data) if layer.data is not None else 0
            n_texts = len(layer.features.get('text', []))
            if n_shapes > n_texts:
                default = layer.metadata.get('default_text', 'Label')
                current_texts = list(layer.features.get('text', []))
                while len(current_texts) < n_shapes:
                    current_texts.append(default)
                layer.features = pd.DataFrame({'text': current_texts})

        self._update_annotation_list()
        self.status_label.setText(f"Draw rectangle to place text box: '{text}'")

    def _add_text_label(self):
        """Add a text annotation layer.

        Creates a Points layer with text display enabled. The user clicks to
        place text at specific locations. Text is slice-aware for 4D data.
        """
        import pandas as pd

        # Get the text to display
        text = self.text_input.text().strip()
        if not text:
            text = "Label"

        # Determine ndim from loaded data to make text slice-aware
        ndim = 2
        if self.current_data is not None and self.current_data.ndim == 4:
            ndim = 3  # Points need (slice, y, x) coordinates

        # Get text color
        text_color = [
            self._current_text_color.redF(),
            self._current_text_color.greenF(),
            self._current_text_color.blueF(),
            1.0,
        ]

        # Create a unique layer name
        existing_text_layers = [l.name for l in self.viewer.layers if l.name.startswith("Text:")]
        layer_name = f"Text: {text}"
        if layer_name in existing_text_layers:
            layer_name = f"Text: {text} ({len(existing_text_layers) + 1})"

        # Create points layer with text display
        # We start with empty data and the user clicks to add points
        layer = self.viewer.add_points(
            data=None,
            name=layer_name,
            size=self.font_size_spin.value(),
            face_color=text_color,
            edge_color=text_color,
            symbol='disc',  # Small marker at text position
            ndim=ndim,
            features={'text': []},  # Will store text for each point
            text={
                'string': '{text}',
                'size': self.font_size_spin.value(),
                'color': text_color,
                'anchor': 'upper_left',
            },
        )

        # Set mode to add points
        layer.mode = 'add'

        # Store the default text for new points on this layer
        layer.metadata['default_text'] = text

        # Connect to data change to auto-fill text for new points
        @layer.events.data.connect
        def on_data_change(event):
            # When new points are added, add text to features
            n_points = len(layer.data)
            n_texts = len(layer.features.get('text', []))
            if n_points > n_texts:
                # New point added - add the default text
                default = layer.metadata.get('default_text', 'Label')
                current_texts = list(layer.features.get('text', []))
                while len(current_texts) < n_points:
                    current_texts.append(default)
                layer.features = pd.DataFrame({'text': current_texts})

        self._update_annotation_list()
        self.status_label.setText(f"Click to place text: '{text}'")

    def _update_annotation_list(self):
        """Update the annotation list widget."""
        self.annotation_list.clear()

        for layer in self.viewer.layers:
            if isinstance(layer, (napari.layers.Shapes, napari.layers.Points)):
                # Include Annotations, Labels, Text, and TextBox layers
                if any(x in layer.name for x in ['Annotation', 'Label', 'Text:', 'TextBox:']):
                    item = QListWidgetItem(layer.name)
                    item.setData(Qt.UserRole, layer.name)
                    self.annotation_list.addItem(item)

    def _delete_selected_annotation(self):
        """Delete selected annotation layer."""
        item = self.annotation_list.currentItem()
        if not item:
            return

        layer_name = item.data(Qt.UserRole)
        for layer in list(self.viewer.layers):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)
                break

        self._update_annotation_list()

    # =========================================================================
    # Export tab handlers
    # =========================================================================

    def _browse_output_folder(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def _on_export(self):
        """Handle export button click."""
        if self.current_data is None:
            QMessageBox.warning(self, "Export Error", "No data loaded")
            return

        output_folder = self.output_folder_edit.text()
        if not output_folder:
            QMessageBox.warning(self, "Export Error", "Please select an output folder")
            return

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get export settings
        scope = self.scope_combo.currentText()
        format_type = self.format_combo.currentText()
        include_annotations = self.include_annotations_cb.isChecked()

        self.status_label.setText("Exporting...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        try:
            if "Composite View" in format_type:
                self._export_composite_view(output_path, scope)
            elif "RGB Composite" in format_type:
                self._export_rgb_composite(output_path, scope)
            else:
                self._export_16bit_channels(output_path, scope)

            self.status_label.setText("Export complete!")
            QMessageBox.information(self, "Export Complete", f"Files saved to:\n{output_path}")

        except Exception as e:
            self.status_label.setText(f"Export error: {e}")
            QMessageBox.critical(self, "Export Error", str(e))

        finally:
            self.progress_bar.setVisible(False)

    def _get_slice_filename(self, z_idx: int, suffix: str = "") -> str:
        """Get filename for a specific slice, using original name if from folder loading.

        Args:
            z_idx: Slice index
            suffix: Optional suffix to append (e.g., "_composite")

        Returns:
            Filename without extension
        """
        # Check if we have original filenames from folder loading
        file_names = self.metadata.get('file_names', [])
        if file_names and z_idx < len(file_names):
            # Use original filename (remove .nd2 extension)
            original_name = file_names[z_idx]
            if original_name.lower().endswith('.nd2'):
                original_name = original_name[:-4]
            return f"{original_name}{suffix}"
        else:
            # Fallback to generic naming
            filename_base = self.current_file.stem if self.current_file else "export"
            n_z = self.current_data.shape[0] if self.current_data is not None and self.current_data.ndim == 4 else 1
            if n_z > 1:
                return f"{filename_base}_Z{z_idx:04d}{suffix}"
            else:
                return f"{filename_base}{suffix}"

    def _export_composite_view(self, output_path: Path, scope: str):
        """
        Export composite view matching exactly what's displayed in napari.

        This captures the current visualization settings (colors, contrast, gamma,
        opacity) and composites visible channels into a single RGB image.
        """
        from mousebrain.pipeline_2d.sliceatlas.core.io import save_tiff
        from mousebrain.pipeline_2d.sliceatlas.core.image_utils import composite_channels

        data = self.current_data
        filename_base = self.current_file.stem if self.current_file else "export"

        # Collect settings from all visible channels
        visible_indices = []
        colors = []
        contrast_limits = []
        gammas = []
        opacities = []

        for i, settings in enumerate(self.channel_settings):
            if settings.visible:
                visible_indices.append(i)
                colors.append(settings.colormap)
                contrast_limits.append(settings.contrast_limits)
                gammas.append(settings.gamma)
                opacities.append(settings.opacity)

        if not visible_indices:
            raise ValueError("No visible channels to export")

        # Determine data dimensions and slices to export
        if data.ndim == 4:
            n_z = data.shape[0]
        else:
            n_z = 1
            data = data[np.newaxis, ...]  # Add Z dimension for consistent indexing

        # Determine which slices to export based on scope
        if "Current Slice" in scope:
            current_step = self.viewer.dims.current_step
            z_indices = [current_step[0] if len(current_step) > 0 else 0]
        elif "Max Intensity" in scope:
            z_indices = ['mip']
        else:
            # All slices
            z_indices = list(range(n_z))

        # Get blending mode
        blend_mode = 'additive' if self.blend_combo.currentText() == "Additive" else 'composite'

        # Export each slice
        for z_idx in z_indices:
            if z_idx == 'mip':
                # Max intensity projection
                channels = [np.max(data[:, i, :, :], axis=0) for i in visible_indices]
                output_filename = f"{filename_base}_composite_MIP"
            else:
                channels = [data[z_idx, i, :, :] for i in visible_indices]
                output_filename = self._get_slice_filename(z_idx, "_composite")

            # Create composite image using current settings
            rgb = composite_channels(
                channels=channels,
                colors=colors,
                contrast_limits=contrast_limits,
                gammas=gammas,
                opacities=opacities,
                mode=blend_mode,
            )

            # Save as TIFF
            output_file = output_path / f"{output_filename}.tiff"
            save_tiff(rgb, output_file, metadata=self.metadata, compress=True, imagej=True)

    def _export_rgb_composite(self, output_path: Path, scope: str):
        """Export as RGB composite."""
        from mousebrain.pipeline_2d.sliceatlas.core.io import save_tiff
        from mousebrain.pipeline_2d.sliceatlas.core.image_utils import composite_channels

        data = self.current_data
        filename_base = self.current_file.stem if self.current_file else "export"

        # Get visible channels and their settings
        visible_indices = [i for i, s in enumerate(self.channel_settings) if s.visible]

        if not visible_indices:
            raise ValueError("No visible channels to export")

        # Determine which slices to export
        if data.ndim == 4:
            n_z = data.shape[0]
        else:
            n_z = 1
            data = data[np.newaxis, ...]  # Add Z dimension

        if "Current Slice" in scope:
            # Get current slice from viewer
            current_step = self.viewer.dims.current_step
            z_indices = [current_step[0] if len(current_step) > 0 else 0]
        elif "Max Intensity" in scope:
            # MIP
            z_indices = ['mip']
        else:
            # All slices
            z_indices = list(range(n_z))

        # Export each slice
        for z_idx in z_indices:
            if z_idx == 'mip':
                # Get max intensity projection
                channels = [np.max(data[:, i, :, :], axis=0) for i in visible_indices]
                output_filename = f"{filename_base}_MIP"
            else:
                channels = [data[z_idx, i, :, :] for i in visible_indices]
                output_filename = self._get_slice_filename(z_idx, "")

            # Get colors and settings for visible channels
            colors = [self.channel_settings[i].colormap for i in visible_indices]
            contrast_limits = [self.channel_settings[i].contrast_limits for i in visible_indices]
            gammas = [self.channel_settings[i].gamma for i in visible_indices]
            opacities = [self.channel_settings[i].opacity for i in visible_indices]

            # Composite
            rgb = composite_channels(
                channels, colors, contrast_limits, gammas, opacities,
                mode='additive' if self.blend_combo.currentText() == "Additive" else 'composite',
            )

            # Save
            output_file = output_path / f"{output_filename}.tiff"
            save_tiff(rgb, output_file, metadata=self.metadata, compress=True, imagej=True)

            if "All Slices (Individual" not in scope:
                break  # Only one file for current slice or MIP

    def _export_16bit_channels(self, output_path: Path, scope: str):
        """Export as 16-bit per channel."""
        from mousebrain.pipeline_2d.sliceatlas.core.io import save_tiff

        data = self.current_data
        filename_base = self.current_file.stem if self.current_file else "export"

        # Determine which slices to export
        if data.ndim == 4:
            n_z = data.shape[0]
        else:
            n_z = 1
            data = data[np.newaxis, ...]

        if "Current Slice" in scope:
            current_step = self.viewer.dims.current_step
            z_indices = [current_step[0] if len(current_step) > 0 else 0]
        elif "Max Intensity" in scope:
            z_indices = ['mip']
        else:
            z_indices = list(range(n_z))

        # Export each channel separately
        for i, settings in enumerate(self.channel_settings):
            if not settings.visible:
                continue

            for z_idx in z_indices:
                if z_idx == 'mip':
                    ch_data = np.max(data[:, i, :, :], axis=0)
                    output_filename = f"{filename_base}_Ch{i}_{settings.name}_MIP"
                else:
                    ch_data = data[z_idx, i, :, :]
                    # Use original filename if available
                    base_name = self._get_slice_filename(z_idx, "")
                    output_filename = f"{base_name}_Ch{i}_{settings.name}"

                output_file = output_path / f"{output_filename}.tiff"
                save_tiff(ch_data, output_file, metadata=self.metadata, compress=True, imagej=True)
