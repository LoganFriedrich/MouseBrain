"""
main_widget.py - Main BrainSlice napari widget

Provides a tabbed interface for:
1. Load - Load images and select channels
2. Insets - Add high-resolution region-of-interest overlays
3. Detect - Nuclei detection with StarDist
4. Colocalize - Measure signal and classify positive/negative
5. Quantify - Assign to regions and export results
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QProgressBar, QCheckBox, QLineEdit,
)
from qtpy.QtCore import Qt

import napari


class BrainSliceWidget(QWidget):
    """Main BrainSlice widget for napari."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # State
        self.current_file: Optional[Path] = None
        self.current_folder: Optional[Path] = None
        self.is_folder_load: bool = False
        self.stack_data: Optional[np.ndarray] = None  # For folder stacks
        self.red_channel: Optional[np.ndarray] = None
        self.green_channel: Optional[np.ndarray] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.nuclei_labels: Optional[np.ndarray] = None
        self.atlas_labels: Optional[np.ndarray] = None
        self.cell_measurements = None  # DataFrame
        self.region_counts = None  # DataFrame
        self.roi_shapes_layer = None  # napari Shapes layer for ROI drawing
        self._roi_counts_data = None  # List of dicts for export
        self._tissue_mask = None
        self._coloc_background = None
        self._coloc_threshold = None
        self._coloc_summary = None
        self._background_diagnostics = None
        self._tissue_pixels = None
        self._diag_canvas = None

        # Tracker
        try:
            from ..tracker import SliceTracker
            self.tracker = SliceTracker()
        except Exception:
            self.tracker = None

        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_run_ids = []
        self.last_run_id: Optional[str] = None

        # Workers
        self.loader_worker = None
        self.detection_worker = None
        self.coloc_worker = None
        self.quant_worker = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("BrainSlice")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.tabs.addTab(self._create_load_tab(), "1. Load")

        # Alignment widget for atlas overlay
        from .alignment_widget import AlignmentWidget
        self.alignment_widget = AlignmentWidget(self)
        self.tabs.addTab(self.alignment_widget, "2. Align")

        # Inset widget (imported here to avoid circular import)
        from .inset_widget import InsetWidget
        self.inset_widget = InsetWidget(self)
        self.tabs.addTab(self.inset_widget, "3. Insets")

        self.tabs.addTab(self._create_detect_tab(), "4. Detect")
        self.tabs.addTab(self._create_coloc_tab(), "5. Colocalize")
        self.tabs.addTab(self._create_quantify_tab(), "6. Quantify")

        # Annotator widget (ND2 annotation & export)
        from .annotator_widget import SliceAnnotatorWidget
        self.annotator_widget = SliceAnnotatorWidget(self.viewer)
        self.tabs.addTab(self.annotator_widget, "7. Annotate")

        # Status bar
        self.status_label = QLabel("Ready - Load an image to begin")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def _create_load_tab(self) -> QWidget:
        """Create the Load tab for image loading."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # File/Folder selection
        file_group = QGroupBox("Image Source")
        file_layout = QVBoxLayout()

        # Single file row
        file_row = QHBoxLayout()
        self.file_label = QLabel("No file/folder selected")
        self.file_label.setWordWrap(True)
        file_row.addWidget(self.file_label, stretch=1)
        file_layout.addLayout(file_row)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("Browse File...")
        self.browse_btn.clicked.connect(self._browse_file)
        btn_layout.addWidget(self.browse_btn)

        self.browse_folder_btn = QPushButton("Browse Folder...")
        self.browse_folder_btn.clicked.connect(self._browse_folder)
        btn_layout.addWidget(self.browse_folder_btn)

        file_layout.addLayout(btn_layout)

        # Folder loading indicator
        self.is_folder_load = False
        self.folder_info_label = QLabel("")
        self.folder_info_label.setStyleSheet("color: blue; font-size: 10px;")
        file_layout.addWidget(self.folder_info_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Loading options
        options_group = QGroupBox("Loading Options")
        options_layout = QVBoxLayout()

        # Z-projection method
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-projection:"))
        self.z_projection_combo = QComboBox()
        self.z_projection_combo.addItems([
            'Max Intensity',
            'Mean',
            'First Z only',
            'All Z (slider)',
        ])
        self.z_projection_combo.setToolTip(
            "How to handle multiple Z-planes:\n"
            "- Max Intensity: Maximum projection (recommended)\n"
            "- Mean: Average all Z-planes\n"
            "- First Z only: Take just the first plane\n"
            "- All Z: Keep as 3D stack with slider"
        )
        z_layout.addWidget(self.z_projection_combo)
        options_layout.addLayout(z_layout)

        # Contrast settings
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItems([
            'Auto (napari) - Recommended',
            'Percentile (1-99%)',
            'Percentile (0.5-99.5%)',
            'Full range',
        ])
        self.contrast_combo.setToolTip(
            "How to set display contrast:\n"
            "- Auto: Let napari auto-adjust (recommended)\n"
            "- Percentile: Use 1st-99th percentile (avoids outliers)\n"
            "- Full range: Use actual min/max values"
        )
        contrast_layout.addWidget(self.contrast_combo)
        options_layout.addLayout(contrast_layout)

        # Rotation settings
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation:"))
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems([
            'None',
            '90° CCW',
            '90° CW',
            '180°',
        ])
        self.rotation_combo.setCurrentIndex(1)  # Default to 90° CCW
        self.rotation_combo.setToolTip(
            "Rotate images on load:\n"
            "- 90° CCW: Counter-clockwise (default for this dataset)\n"
            "- 90° CW: Clockwise\n"
            "- 180°: Flip upside down"
        )
        rotation_layout.addWidget(self.rotation_combo)
        options_layout.addLayout(rotation_layout)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Channel selection
        channel_group = QGroupBox("Channel Assignment")
        channel_layout = QVBoxLayout()

        # Red (nuclear) channel
        red_layout = QHBoxLayout()
        red_layout.addWidget(QLabel("Nuclear (red):"))
        self.red_channel_spin = QSpinBox()
        self.red_channel_spin.setRange(0, 10)
        self.red_channel_spin.setValue(1)  # Default to channel 1 (561nm)
        red_layout.addWidget(self.red_channel_spin)
        channel_layout.addLayout(red_layout)

        # Green (signal) channel
        green_layout = QHBoxLayout()
        green_layout.addWidget(QLabel("Signal (green):"))
        self.green_channel_spin = QSpinBox()
        self.green_channel_spin.setRange(0, 10)
        self.green_channel_spin.setValue(0)  # Default to channel 0 (488nm)
        green_layout.addWidget(self.green_channel_spin)
        channel_layout.addLayout(green_layout)

        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)

        # Load button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self._load_image)
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        # Metadata display
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.metadata_label)

        layout.addStretch()
        return widget

    def _create_detect_tab(self) -> QWidget:
        """Create the Detect tab for nuclei detection."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Model selection
        model_group = QGroupBox("StarDist Model")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            '2D_versatile_fluo',
            '2D_versatile_he',
            '2D_paper_dsb2018',
        ])
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Detection parameters
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()

        # Probability threshold
        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Prob Threshold:"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setSingleStep(0.05)
        self.prob_spin.setValue(0.5)
        prob_layout.addWidget(self.prob_spin)
        param_layout.addLayout(prob_layout)

        # NMS threshold
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS Threshold:"))
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.4)
        nms_layout.addWidget(self.nms_spin)
        param_layout.addLayout(nms_layout)

        # Size filtering
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min/Max Area:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10000)
        self.min_area_spin.setValue(50)
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 100000)
        self.max_area_spin.setValue(5000)
        size_layout.addWidget(self.min_area_spin)
        size_layout.addWidget(QLabel("-"))
        size_layout.addWidget(self.max_area_spin)
        param_layout.addLayout(size_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Run button
        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self._run_detection)
        self.detect_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)

        # Results
        self.detect_result_label = QLabel("")
        layout.addWidget(self.detect_result_label)

        layout.addStretch()
        return widget

    def _create_coloc_tab(self) -> QWidget:
        """Create the Colocalize tab for intensity measurement."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Background estimation
        bg_group = QGroupBox("Background Estimation")
        bg_layout = QVBoxLayout()

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(['percentile', 'mode', 'mean'])
        method_layout.addWidget(self.bg_method_combo)
        bg_layout.addLayout(method_layout)

        percentile_layout = QHBoxLayout()
        percentile_layout.addWidget(QLabel("Percentile:"))
        self.bg_percentile_spin = QDoubleSpinBox()
        self.bg_percentile_spin.setRange(1.0, 50.0)
        self.bg_percentile_spin.setValue(10.0)
        percentile_layout.addWidget(self.bg_percentile_spin)
        bg_layout.addLayout(percentile_layout)

        dilation_layout = QHBoxLayout()
        dilation_layout.addWidget(QLabel("Nuclei exclusion radius:"))
        self.bg_dilation_spin = QSpinBox()
        self.bg_dilation_spin.setRange(5, 200)
        self.bg_dilation_spin.setValue(50)
        self.bg_dilation_spin.setToolTip(
            "Dilation iterations for excluding signal around nuclei from background.\n"
            "Increase for non-nuclear signals (e.g., eYFP in soma/processes)."
        )
        dilation_layout.addWidget(self.bg_dilation_spin)
        bg_layout.addLayout(dilation_layout)

        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)

        # Classification threshold
        thresh_group = QGroupBox("Positive/Negative Classification")
        thresh_layout = QVBoxLayout()

        thresh_method_layout = QHBoxLayout()
        thresh_method_layout.addWidget(QLabel("Method:"))
        self.thresh_method_combo = QComboBox()
        self.thresh_method_combo.addItems(['fold_change', 'absolute', 'percentile'])
        thresh_method_layout.addWidget(self.thresh_method_combo)
        thresh_layout.addLayout(thresh_method_layout)

        thresh_value_layout = QHBoxLayout()
        thresh_value_layout.addWidget(QLabel("Threshold:"))
        self.thresh_value_spin = QDoubleSpinBox()
        self.thresh_value_spin.setRange(0.1, 100.0)
        self.thresh_value_spin.setValue(2.0)
        self.thresh_value_spin.setSingleStep(0.5)
        thresh_value_layout.addWidget(self.thresh_value_spin)
        thresh_layout.addLayout(thresh_value_layout)

        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)

        # Run button
        self.coloc_btn = QPushButton("Run Colocalization Analysis")
        self.coloc_btn.clicked.connect(self._run_colocalization)
        self.coloc_btn.setEnabled(False)
        layout.addWidget(self.coloc_btn)

        # Results
        self.coloc_result_label = QLabel("")
        self.coloc_result_label.setWordWrap(True)
        layout.addWidget(self.coloc_result_label)

        # --- ROI Counting ---
        roi_group = QGroupBox("ROI Counting")
        roi_layout = QVBoxLayout()

        roi_btn_layout = QHBoxLayout()
        self.draw_roi_btn = QPushButton("Draw ROI")
        self.draw_roi_btn.clicked.connect(self._add_roi_layer)
        roi_btn_layout.addWidget(self.draw_roi_btn)

        self.count_roi_btn = QPushButton("Count in ROI(s)")
        self.count_roi_btn.clicked.connect(self._count_all_rois)
        roi_btn_layout.addWidget(self.count_roi_btn)

        self.export_roi_btn = QPushButton("Export")
        self.export_roi_btn.clicked.connect(self._export_roi_counts)
        roi_btn_layout.addWidget(self.export_roi_btn)
        roi_layout.addLayout(roi_btn_layout)

        self.roi_results_table = QTableWidget()
        self.roi_results_table.setColumnCount(5)
        self.roi_results_table.setHorizontalHeaderLabels(
            ["ROI", "Total Nuclei", "Green+", "Green-", "Fraction"]
        )
        self.roi_results_table.setMaximumHeight(200)
        roi_layout.addWidget(self.roi_results_table)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        # Diagnostics Group
        diag_group = QGroupBox("Diagnostics")
        diag_layout = QVBoxLayout()

        # Plot selector + save button row
        diag_ctrl_layout = QHBoxLayout()
        self.diag_plot_combo = QComboBox()
        self.diag_plot_combo.addItems([
            'Fold Change Histogram',
            'Intensity vs Area',
            'Overlay Image',
            'Annotated Overlay',
            'Background Mask',
            'GMM Diagnostic',
        ])
        self.diag_plot_combo.currentIndexChanged.connect(self._update_diagnostic_plot)
        diag_ctrl_layout.addWidget(self.diag_plot_combo)

        self.save_qc_btn = QPushButton("Save QC Images")
        self.save_qc_btn.clicked.connect(self._save_qc_images)
        diag_ctrl_layout.addWidget(self.save_qc_btn)
        diag_layout.addLayout(diag_ctrl_layout)

        # Matplotlib canvas
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure as MplFigure
        self._diag_figure = MplFigure(figsize=(6, 4), dpi=100)
        self._diag_canvas = FigureCanvas(self._diag_figure)
        self._diag_canvas.setMinimumHeight(300)
        diag_layout.addWidget(self._diag_canvas)

        diag_group.setLayout(diag_layout)
        layout.addWidget(diag_group)

        layout.addStretch()
        return widget

    def _create_quantify_tab(self) -> QWidget:
        """Create the Quantify tab for regional counting."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Atlas selection (placeholder)
        atlas_group = QGroupBox("Atlas (Optional)")
        atlas_layout = QVBoxLayout()
        atlas_layout.addWidget(QLabel("Atlas registration not yet implemented."))
        atlas_layout.addWidget(QLabel("Cells will be counted without region assignment."))
        atlas_group.setLayout(atlas_layout)
        layout.addWidget(atlas_group)

        # Export options
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Sample ID:"))
        self.sample_id_edit = QLineEdit()
        self.sample_id_edit.setPlaceholderText("e.g., ENCR_001_slice12")
        sample_layout.addWidget(self.sample_id_edit)
        export_layout.addLayout(sample_layout)

        self.export_csv_check = QCheckBox("Export to CSV")
        self.export_csv_check.setChecked(True)
        export_layout.addWidget(self.export_csv_check)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Run button
        self.quant_btn = QPushButton("Run Quantification")
        self.quant_btn.clicked.connect(self._run_quantification)
        self.quant_btn.setEnabled(False)
        layout.addWidget(self.quant_btn)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            'Metric', 'Value', '', ''
        ])
        layout.addWidget(self.results_table)

        layout.addStretch()
        return widget

    # =========================================================================
    # ACTION HANDLERS
    # =========================================================================

    def _browse_file(self):
        """Open file browser to select single image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.nd2 *.tif *.tiff);;All Files (*)"
        )

        if file_path:
            self.current_file = Path(file_path)
            self.current_folder = None
            self.is_folder_load = False
            self.file_label.setText(str(self.current_file.name))
            self.folder_info_label.setText("")
            self.load_btn.setEnabled(True)

            # Auto-set sample ID from filename
            self.sample_id_edit.setText(self.current_file.stem)

    def _browse_folder(self):
        """Open folder browser to select folder of images as stack."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Images",
            "",
        )

        if folder_path:
            self.current_folder = Path(folder_path)
            self.current_file = None
            self.is_folder_load = True
            print(f"[BrainSlice] _browse_folder: folder={self.current_folder}, is_folder_load={self.is_folder_load}")

            # Count images in folder
            from ..core.io import find_images_in_folder
            try:
                files = find_images_in_folder(self.current_folder)
                n_files = len(files)
                if n_files == 0:
                    self.file_label.setText("No supported images found")
                    self.folder_info_label.setText("")
                    self.load_btn.setEnabled(False)
                else:
                    self.file_label.setText(str(self.current_folder.name))
                    self.folder_info_label.setText(f"Found {n_files} images - will load as stack")
                    self.load_btn.setEnabled(True)
                    # Auto-set sample ID from folder name
                    self.sample_id_edit.setText(self.current_folder.name)
            except Exception as e:
                self.file_label.setText(f"Error: {e}")
                self.load_btn.setEnabled(False)

    def _load_image(self):
        """Load the selected image or folder."""
        print(f"[BrainSlice] _load_image called: is_folder_load={self.is_folder_load}, folder={self.current_folder}, file={self.current_file}")
        if self.is_folder_load:
            self._load_folder()
        else:
            self._load_single_file()

    def _get_z_projection_mode(self) -> str:
        """Get z-projection mode from combo box."""
        text = self.z_projection_combo.currentText()
        if 'Max' in text:
            return 'max'
        elif 'Mean' in text:
            return 'mean'
        elif 'First' in text:
            return 'first'
        elif 'All' in text:
            return 'all'
        return 'max'

    def _load_single_file(self):
        """Load a single image file."""
        print(f"[BrainSlice] _load_single_file called: current_file={self.current_file}")
        if self.current_file is None:
            print("[BrainSlice] WARNING: current_file is None, returning early!")
            return

        self.status_label.setText("Loading image...")
        self.load_btn.setEnabled(False)

        from .workers import ImageLoaderWorker

        self.loader_worker = ImageLoaderWorker(
            self.current_file,
            red_idx=self.red_channel_spin.value(),
            green_idx=self.green_channel_spin.value(),
            z_projection=self._get_z_projection_mode(),
        )
        self.loader_worker.progress.connect(self._on_load_progress)
        self.loader_worker.finished.connect(self._on_load_finished)
        self.loader_worker.start()

    def _load_folder(self):
        """Load folder of images as a stack."""
        if self.current_folder is None:
            return

        print(f"[BrainSlice] Starting folder load: {self.current_folder}")
        self.status_label.setText("Loading folder as stack...")
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        try:
            from .workers import FolderLoaderWorker

            z_mode = self._get_z_projection_mode()
            print(f"[BrainSlice] Z-projection mode: {z_mode}")

            self.loader_worker = FolderLoaderWorker(
                self.current_folder,
                red_idx=self.red_channel_spin.value(),
                green_idx=self.green_channel_spin.value(),
                z_projection=z_mode,
            )
            self.loader_worker.progress.connect(self._on_folder_load_progress)
            self.loader_worker.finished.connect(self._on_folder_load_finished)
            self.loader_worker.start()
            print("[BrainSlice] Worker started")
        except Exception as e:
            import traceback
            print(f"[BrainSlice] ERROR starting worker: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Error: {e}")
            self.load_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_folder_load_progress(self, current: int, total: int, filename: str):
        """Handle folder load progress updates."""
        self.status_label.setText(f"Loading {filename} ({current}/{total})")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_folder_load_finished(self, success: bool, message: str, stack, metadata):
        """Handle folder load completion."""
        print(f"[BrainSlice] Folder load finished: success={success}, message={message}")
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            # Apply rotation to entire stack
            stack = self._apply_rotation(stack)
            self.stack_data = stack  # (N_slices, C, Y, X)
            self.metadata = metadata

            # Extract red and green channels from stack
            red_idx = self.red_channel_spin.value()
            green_idx = self.green_channel_spin.value()

            # Stack is (S, C, Y, X) - take channels for each slice
            red_stack = stack[:, red_idx, :, :]  # (S, Y, X)
            green_stack = stack[:, green_idx, :, :]  # (S, Y, X)

            # Store for detection
            self.red_channel = red_stack
            self.green_channel = green_stack

            # Calculate contrast limits (Auto = None lets napari decide)
            red_limits = self._get_contrast_limits(red_stack)
            green_limits = self._get_contrast_limits(green_stack)

            # Add to napari as stack with proper contrast
            self.viewer.layers.clear()
            self.viewer.add_image(
                red_stack,
                name="Nuclear (red) Stack",
                colormap='red',
                blending='additive',
                contrast_limits=red_limits,
            )
            self.viewer.add_image(
                green_stack,
                name="Signal (green) Stack",
                colormap='green',
                blending='additive',
                contrast_limits=green_limits,
            )

            # Update UI
            n_slices = metadata.get('n_slices', stack.shape[0])
            self.status_label.setText(f"Loaded {n_slices} slices as stack")
            self.metadata_label.setText(
                f"Shape: {metadata.get('shape', 'Unknown')}, "
                f"Slices: {n_slices}, "
                f"Channels: {metadata.get('channels', 'Unknown')}"
            )

            # Enable detection
            self.detect_btn.setEnabled(True)

            # Notify inset widget that base is loaded
            self.inset_widget.on_base_loaded()

        else:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.warning(self, "Load Error", message)

    def _on_load_progress(self, message: str):
        """Handle load progress updates."""
        self.status_label.setText(message)

    def _get_contrast_limits(self, image: np.ndarray) -> tuple:
        """Calculate contrast limits based on user selection."""
        contrast_mode = self.contrast_combo.currentText()

        if 'Auto' in contrast_mode:
            return None  # Let napari decide (recommended)
        elif 'Full range' in contrast_mode:
            return (float(image.min()), float(image.max()))
        elif '0.5-99.5' in contrast_mode:
            lo = np.percentile(image, 0.5)
            hi = np.percentile(image, 99.5)
            return (float(lo), float(hi))
        else:  # Percentile 1-99%
            lo = np.percentile(image, 1)
            hi = np.percentile(image, 99)
            return (float(lo), float(hi))

    def _get_rotation_k(self) -> int:
        """Get rotation value (k for np.rot90) from dropdown."""
        rotation_mode = self.rotation_combo.currentText()
        if '90° CCW' in rotation_mode:
            return 1  # 90° counter-clockwise
        elif '90° CW' in rotation_mode:
            return 3  # 90° clockwise (= 270° CCW)
        elif '180' in rotation_mode:
            return 2  # 180°
        return 0  # No rotation

    def _apply_rotation(self, data: np.ndarray) -> np.ndarray:
        """Apply rotation to image data based on dropdown selection."""
        k = self._get_rotation_k()
        if k == 0:
            return data

        # Handle different array shapes
        if data.ndim == 2:
            return np.rot90(data, k=k)
        elif data.ndim == 3:
            # (S, Y, X) or (C, Y, X) - rotate last two axes
            return np.rot90(data, k=k, axes=(-2, -1))
        elif data.ndim == 4:
            # (S, C, Y, X) - rotate last two axes
            return np.rot90(data, k=k, axes=(-2, -1))
        return data

    def _get_current_slice(self, data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract current slice from 3D stack, or return 2D data as-is.

        This handles the case where data is a 3D stack (S, Y, X) from folder loading.
        Detection and colocalization require 2D images, so we extract the current
        viewer slice.

        Args:
            data: 2D image (Y, X) or 3D stack (S, Y, X)

        Returns:
            2D image (Y, X) for the current slice, or None if input is None
        """
        if data is None:
            return None
        if data.ndim == 3:
            # Stack - use current viewer slice
            current_slice_idx = self.viewer.dims.current_step[0]
            return data[current_slice_idx]
        return data

    def _on_load_finished(self, success: bool, message: str, red, green, metadata):
        """Handle load completion."""
        self.load_btn.setEnabled(True)

        if success:
            # Apply rotation
            red = self._apply_rotation(red)
            green = self._apply_rotation(green)

            self.red_channel = red
            self.green_channel = green
            self.metadata = metadata

            # Calculate contrast limits (Auto = None lets napari decide)
            red_limits = self._get_contrast_limits(red)
            green_limits = self._get_contrast_limits(green)

            # Add to napari with proper contrast
            self.viewer.layers.clear()
            self.viewer.add_image(
                red,
                name="Nuclear (red)",
                colormap='red',
                blending='additive',
                contrast_limits=red_limits,
            )
            self.viewer.add_image(
                green,
                name="Signal (green)",
                colormap='green',
                blending='additive',
                contrast_limits=green_limits,
            )

            # Update UI
            z_info = metadata.get('z_projection', '')
            if z_info:
                z_info = f", Z: {z_info}"
            self.status_label.setText(f"Loaded: {self.current_file.name}{z_info}")
            self.metadata_label.setText(
                f"Shape: {metadata.get('shape', 'Unknown')}, "
                f"Channels: {metadata.get('channels', 'Unknown')}"
            )

            # Enable detection
            self.detect_btn.setEnabled(True)

            # Notify inset widget that base is loaded
            self.inset_widget.on_base_loaded()

        else:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.warning(self, "Load Error", message)

    def _run_detection(self):
        """Run nuclei detection."""
        if self.red_channel is None:
            QMessageBox.warning(self, "Error", "Load an image first")
            return

        self.status_label.setText("Running detection...")
        self.detect_btn.setEnabled(False)

        params = {
            'model': self.model_combo.currentText(),
            'prob_thresh': self.prob_spin.value(),
            'nms_thresh': self.nms_spin.value(),
            'min_area': self.min_area_spin.value(),
            'max_area': self.max_area_spin.value(),
            'filter_size': True,
        }

        # Log to tracker
        if self.tracker:
            sample_id = self.sample_id_edit.text() or self.current_file.stem
            self.last_run_id = self.tracker.log_detection(
                sample_id=sample_id,
                model=params['model'],
                prob_thresh=params['prob_thresh'],
                nms_thresh=params['nms_thresh'],
                min_area=params['min_area'],
                max_area=params['max_area'],
                status='started',
            )

        # Check if we should use inset detection
        inset_settings = self.inset_widget.get_detection_settings()
        use_insets = (
            inset_settings['use_insets'] and
            inset_settings['inset_manager'] is not None and
            len(inset_settings['inset_manager'].insets) > 0
        )

        if use_insets:
            # Run inset-aware detection (synchronous for now)
            self._run_inset_detection(params, inset_settings)
        else:
            # Standard detection - extract current slice if dealing with stack
            image = self._get_current_slice(self.red_channel)
            if image is None:
                QMessageBox.warning(self, "Error", "No image data available")
                self.detect_btn.setEnabled(True)
                return

            from .workers import DetectionWorker
            self.detection_worker = DetectionWorker(image, params)
            self.detection_worker.progress.connect(self._on_detect_progress)
            self.detection_worker.finished.connect(self._on_detect_finished)
            self.detection_worker.start()

    def _run_inset_detection(self, params: Dict[str, Any], inset_settings: Dict[str, Any]):
        """Run detection using insets at full resolution."""
        try:
            from ..core.inset_detection import InsetDetectionPipeline
            from ..core.detection import NucleiDetector

            self.status_label.setText("Running inset detection pipeline...")

            # Create detector and pipeline
            detector = NucleiDetector(model_name=params['model'])
            pipeline = InsetDetectionPipeline(
                inset_settings['inset_manager'],
                detector,
            )

            # Run detection
            results = pipeline.run_full_detection(
                channel=0,  # Red channel
                detect_in_base=inset_settings['detect_in_base'],
                prob_thresh=params['prob_thresh'],
                nms_thresh=params['nms_thresh'],
            )

            # Filter by size
            if params.get('filter_size', True) and results['merged_properties'] is not None:
                df = results['merged_properties']
                mask = (
                    (df['area'] >= params['min_area']) &
                    (df['area'] <= params['max_area'])
                )
                results['merged_properties'] = df[mask].reset_index(drop=True)
                results['total_cells'] = len(results['merged_properties'])

            # Store results
            self.nuclei_labels = results['merged_labels']
            count = results['total_cells']

            # Update tracker
            if self.tracker and self.last_run_id:
                self.tracker.update_status(
                    self.last_run_id,
                    status='completed',
                    det_nuclei_found=count,
                )
                self.session_run_ids.append(self.last_run_id)

            # Visualize results
            # Remove old detection layers
            for layer in list(self.viewer.layers):
                if 'Nuclei' in layer.name:
                    self.viewer.layers.remove(layer)

            # Add merged labels
            self.viewer.add_labels(
                results['merged_labels'],
                name=f"Nuclei ({count})"
            )

            # Show inset vs base detections differently
            if results['merged_properties'] is not None and len(results['merged_properties']) > 0:
                df = results['merged_properties']

                # Inset detections
                inset_cells = df[df['from_inset']]
                if len(inset_cells) > 0:
                    coords = inset_cells[['centroid_y_base', 'centroid_x_base']].values
                    self.viewer.add_points(
                        coords,
                        name=f"Inset Detections ({len(inset_cells)})",
                        size=8,
                        face_color='transparent',
                        edge_color='cyan',
                        edge_width=0.5,
                    )

                # Base detections
                base_cells = df[~df['from_inset']]
                if len(base_cells) > 0:
                    coords = base_cells[['centroid_y_base', 'centroid_x_base']].values
                    self.viewer.add_points(
                        coords,
                        name=f"Base Detections ({len(base_cells)})",
                        size=8,
                        face_color='transparent',
                        edge_color='yellow',
                        edge_width=0.5,
                    )

            # Update UI
            n_insets = len(inset_settings['inset_manager'].insets)
            message = f"Detected {count} nuclei ({n_insets} insets used)"
            self.status_label.setText(message)
            self.detect_result_label.setText(message)
            self.detect_btn.setEnabled(True)

            # Enable colocalization
            self.coloc_btn.setEnabled(True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Error: {e}")
            self.detect_btn.setEnabled(True)
            if self.tracker and self.last_run_id:
                self.tracker.update_status(self.last_run_id, status='failed')

    def _on_detect_progress(self, message: str):
        """Handle detection progress updates."""
        self.status_label.setText(message)

    def _on_detect_finished(self, success: bool, message: str, count: int, labels):
        """Handle detection completion."""
        self.detect_btn.setEnabled(True)

        if success:
            self.nuclei_labels = labels

            # Update tracker
            if self.tracker and self.last_run_id:
                self.tracker.update_status(
                    self.last_run_id,
                    status='completed',
                    det_nuclei_found=count,
                )
                self.session_run_ids.append(self.last_run_id)

            # Add labels to napari
            # Remove old detection layer if exists
            for layer in list(self.viewer.layers):
                if 'Nuclei' in layer.name:
                    self.viewer.layers.remove(layer)

            self.viewer.add_labels(labels, name=f"Nuclei ({count})")

            # Update UI
            self.status_label.setText(message)
            self.detect_result_label.setText(f"Detected {count} nuclei")

            # Enable colocalization
            self.coloc_btn.setEnabled(True)

        else:
            self.status_label.setText(f"Error: {message}")
            if self.tracker and self.last_run_id:
                self.tracker.update_status(self.last_run_id, status='failed')

    def _run_colocalization(self):
        """Run colocalization analysis."""
        print(f"[BrainSlice] _run_colocalization called")
        print(f"[BrainSlice]   nuclei_labels: {type(self.nuclei_labels)}, shape={getattr(self.nuclei_labels, 'shape', None)}")
        print(f"[BrainSlice]   green_channel: {type(self.green_channel)}, shape={getattr(self.green_channel, 'shape', None)}")
        if self.nuclei_labels is None:
            QMessageBox.warning(self, "Error", "Run detection first")
            return
        if self.green_channel is None:
            QMessageBox.warning(self, "Error", "No signal channel loaded")
            return

        self.status_label.setText("Running colocalization analysis...")
        self.coloc_btn.setEnabled(False)

        params = {
            'background_method': self.bg_method_combo.currentText(),
            'background_percentile': self.bg_percentile_spin.value(),
            'threshold_method': self.thresh_method_combo.currentText(),
            'threshold_value': self.thresh_value_spin.value(),
            'dilation_iterations': self.bg_dilation_spin.value(),
        }

        # Log to tracker
        if self.tracker:
            sample_id = self.sample_id_edit.text() or self.current_file.stem
            self.last_run_id = self.tracker.log_colocalization(
                sample_id=sample_id,
                signal_channel='green',
                background_method=params['background_method'],
                background_percentile=params['background_percentile'],
                threshold_method=params['threshold_method'],
                threshold_value=params['threshold_value'],
                status='started',
            )

        from .workers import ColocalizationWorker

        # Extract current slice if dealing with stack
        signal_image = self._get_current_slice(self.green_channel)
        if signal_image is None:
            QMessageBox.warning(self, "Error", "No signal channel data available")
            self.coloc_btn.setEnabled(True)
            return

        # nuclei_labels should already be 2D from detection on current slice
        # but handle edge case where it might still be 3D
        labels = self._get_current_slice(self.nuclei_labels)
        if labels is None:
            QMessageBox.warning(self, "Error", "No nuclei labels available")
            self.coloc_btn.setEnabled(True)
            return

        self.coloc_worker = ColocalizationWorker(
            signal_image,
            labels,
            params,
        )
        self.coloc_worker.progress.connect(self._on_coloc_progress)
        self.coloc_worker.finished.connect(self._on_coloc_finished)
        self.coloc_worker.start()

    def _on_coloc_progress(self, message: str):
        """Handle colocalization progress updates."""
        self.status_label.setText(message)

    def _on_coloc_finished(self, success: bool, message: str, measurements, summary, tissue_mask):
        """Handle colocalization completion."""
        print(f"[BrainSlice] _on_coloc_finished: success={success}, message={message}")
        if measurements is not None:
            print(f"[BrainSlice]   measurements: {len(measurements)} rows")
        if summary is not None:
            print(f"[BrainSlice]   summary: {summary}")
        self.coloc_btn.setEnabled(True)

        if success:
            self.cell_measurements = measurements
            self._tissue_mask = tissue_mask
            self._coloc_background = summary['background_used']
            self._coloc_threshold = summary.get('threshold_value', self.thresh_value_spin.value())
            self._background_diagnostics = getattr(self.coloc_worker, 'background_diagnostics', None)
            self._tissue_pixels = getattr(self.coloc_worker, 'tissue_pixels', None)
            self._coloc_summary = summary

            # Update tracker
            if self.tracker and self.last_run_id:
                self.tracker.update_status(
                    self.last_run_id,
                    status='completed',
                    coloc_positive_cells=summary['positive_cells'],
                    coloc_negative_cells=summary['negative_cells'],
                    coloc_positive_fraction=summary['positive_fraction'],
                    coloc_background_value=summary['background_used'],
                )
                self.session_run_ids.append(self.last_run_id)

            # Visualize results - color nuclei by positive/negative
            self._visualize_colocalization(measurements)

            # Update diagnostic plot
            self._update_diagnostic_plot()

            # Update UI
            self.status_label.setText(message)
            self.coloc_result_label.setText(
                f"Positive: {summary['positive_cells']} ({summary['positive_fraction']*100:.1f}%)\n"
                f"Negative: {summary['negative_cells']}\n"
                f"Background: {summary['background_used']:.1f}\n"
                f"Mean fold change: {summary['mean_fold_change']:.2f}"
            )

            # Enable quantification
            self.quant_btn.setEnabled(True)

        else:
            self.status_label.setText(f"Error: {message}")
            if self.tracker and self.last_run_id:
                self.tracker.update_status(self.last_run_id, status='failed')

    def _visualize_colocalization(self, measurements):
        """Visualize colocalization results in napari."""
        if measurements is None or len(measurements) == 0:
            return

        # Determine column names - handle both standard and inset detection data
        # Inset detection uses centroid_y_base/centroid_x_base for coordinates in base image space
        if 'centroid_y_base' in measurements.columns:
            y_col, x_col = 'centroid_y_base', 'centroid_x_base'
        else:
            y_col, x_col = 'centroid_y', 'centroid_x'

        # Create points for positive and negative cells
        positive = measurements[measurements['is_positive']]
        negative = measurements[~measurements['is_positive']]

        # Remove old colocalization layers
        for layer in list(self.viewer.layers):
            if 'Positive' in layer.name or 'Negative' in layer.name:
                self.viewer.layers.remove(layer)

        # Add positive cells (green)
        if len(positive) > 0:
            pos_coords = positive[[y_col, x_col]].values
            self.viewer.add_points(
                pos_coords,
                name=f"Positive ({len(positive)})",
                size=10,
                face_color='transparent',
                edge_color='lime',
                edge_width=0.5,
            )

        # Add negative cells (red)
        if len(negative) > 0:
            neg_coords = negative[[y_col, x_col]].values
            self.viewer.add_points(
                neg_coords,
                name=f"Negative ({len(negative)})",
                size=10,
                face_color='transparent',
                edge_color='red',
                edge_width=0.5,
            )

    def _update_diagnostic_plot(self):
        """Update the diagnostic plot based on combo selection."""
        if self.cell_measurements is None or self._diag_canvas is None:
            return

        from ..core.visualization import (
            create_overlay_image,
            create_annotated_overlay,
            create_background_mask_overlay,
            create_fold_change_histogram,
            create_intensity_scatter,
            create_gmm_diagnostic,
        )

        plot_type = self.diag_plot_combo.currentText()
        threshold = self._coloc_threshold or self.thresh_value_spin.value()
        background = self._coloc_background or 0.0

        # Clear current figure
        self._diag_figure.clear()

        try:
            if plot_type == 'Fold Change Histogram':
                fig = create_fold_change_histogram(
                    self.cell_measurements, threshold, background
                )
            elif plot_type == 'Intensity vs Area':
                fig = create_intensity_scatter(
                    self.cell_measurements, background, threshold
                )
            elif plot_type == 'Overlay Image':
                green = self._get_current_slice(self.green_channel)
                labels = self._get_current_slice(self.nuclei_labels)
                if green is not None and labels is not None:
                    fig = create_overlay_image(green, labels, self.cell_measurements)
                else:
                    return
            elif plot_type == 'Annotated Overlay':
                green = self._get_current_slice(self.green_channel)
                labels = self._get_current_slice(self.nuclei_labels)
                if green is not None and labels is not None:
                    fig = create_annotated_overlay(green, labels, self.cell_measurements)
                else:
                    return
            elif plot_type == 'Background Mask':
                green = self._get_current_slice(self.green_channel)
                labels = self._get_current_slice(self.nuclei_labels)
                if green is not None and labels is not None and self._tissue_mask is not None:
                    fig = create_background_mask_overlay(green, labels, self._tissue_mask)
                else:
                    return
            elif plot_type == 'GMM Diagnostic':
                if self._tissue_pixels is not None and self._background_diagnostics is not None:
                    fig = create_gmm_diagnostic(self._tissue_pixels, self._background_diagnostics)
                else:
                    return
            else:
                return

            # Copy the generated figure content onto our embedded figure
            # We need to replace the canvas figure's content
            import matplotlib.pyplot as plt

            # Get the axes from the generated figure and recreate on our figure
            src_axes = fig.get_axes()
            if src_axes:
                src_ax = src_axes[0]
                # Simpler approach: just swap the figure on the canvas
                self._diag_canvas.figure = fig
                fig.set_canvas(self._diag_canvas)
                fig.tight_layout()

            self._diag_canvas.draw()
            plt.close(self._diag_figure)  # Close old figure
            self._diag_figure = fig  # Keep reference to new one

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[BrainSlice] Diagnostic plot error: {e}")

    def _save_qc_images(self):
        """Save all QC images to a folder."""
        if self.cell_measurements is None:
            QMessageBox.warning(self, "Error", "Run colocalization first")
            return

        default_dir = str(self.current_file.parent) if self.current_file else str(Path.home())
        output_dir = QFileDialog.getExistingDirectory(self, "Select QC Output Folder", default_dir)
        if not output_dir:
            return

        try:
            from ..core.visualization import save_all_qc_figures

            green = self._get_current_slice(self.green_channel)
            labels = self._get_current_slice(self.nuclei_labels)
            threshold = self._coloc_threshold or self.thresh_value_spin.value()
            background = self._coloc_background or 0.0
            prefix = self.current_file.stem if self.current_file else "qc"

            saved = save_all_qc_figures(
                output_dir=Path(output_dir),
                green_channel=green,
                nuclei_labels=labels,
                measurements_df=self.cell_measurements,
                tissue_mask=self._tissue_mask,
                threshold=threshold,
                background=background,
                roi_counts=self._roi_counts_data,
                background_diagnostics=getattr(self, '_background_diagnostics', None),
                tissue_pixels=getattr(self, '_tissue_pixels', None),
                summary=getattr(self, '_coloc_summary', None),
                prefix=prefix,
            )

            self.status_label.setText(f"Saved {len(saved)} files to {Path(output_dir).name}/")
            QMessageBox.information(self, "Success", f"Saved {len(saved)} QC images")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to save QC images: {e}")

    # ----- ROI Counting Methods -----

    def _add_roi_layer(self):
        """Add or activate a Shapes layer for drawing ROIs."""
        # Check if our layer still exists in the viewer
        if self.roi_shapes_layer is not None:
            if self.roi_shapes_layer not in self.viewer.layers:
                self.roi_shapes_layer = None

        if self.roi_shapes_layer is None:
            self.roi_shapes_layer = self.viewer.add_shapes(
                name="ROIs",
                edge_color="yellow",
                edge_width=2,
                face_color="transparent",
            )

        self.viewer.layers.selection.active = self.roi_shapes_layer
        self.roi_shapes_layer.mode = 'add_polygon'
        self.status_label.setText("Draw ROI polygon. Press Escape when done.")

    def _count_all_rois(self):
        """Count colocalized cells in all drawn ROIs."""
        if self.cell_measurements is None:
            QMessageBox.warning(self, "Error", "Run colocalization first")
            return

        if self.roi_shapes_layer is None or len(self.roi_shapes_layer.data) == 0:
            QMessageBox.warning(self, "Error", "Draw at least one ROI first")
            return

        from ..core.colocalization import filter_measurements_by_roi

        # Determine image shape from red channel
        if self.red_channel is not None:
            img = self._get_current_slice(self.red_channel)
            image_shape = img.shape[:2] if img is not None else (1, 1)
        else:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        results = []
        for i, shape_data in enumerate(self.roi_shapes_layer.data):
            vertices = np.array(shape_data)  # Nx2 (y, x)
            filtered = filter_measurements_by_roi(
                self.cell_measurements, vertices, image_shape
            )

            total = len(filtered)
            positive = int(filtered['is_positive'].sum()) if total > 0 else 0
            negative = total - positive
            fraction = positive / total if total > 0 else 0.0

            results.append({
                'roi': f"ROI {i+1}",
                'total': total,
                'positive': positive,
                'negative': negative,
                'fraction': fraction,
            })

        # Add totals row
        t_total = sum(r['total'] for r in results)
        t_pos = sum(r['positive'] for r in results)
        t_neg = sum(r['negative'] for r in results)
        t_frac = t_pos / t_total if t_total > 0 else 0.0
        results.append({
            'roi': 'TOTAL',
            'total': t_total,
            'positive': t_pos,
            'negative': t_neg,
            'fraction': t_frac,
        })

        self._roi_counts_data = results

        # Update table
        self.roi_results_table.setRowCount(len(results))
        for row_idx, r in enumerate(results):
            self.roi_results_table.setItem(row_idx, 0, QTableWidgetItem(r['roi']))
            self.roi_results_table.setItem(row_idx, 1, QTableWidgetItem(str(r['total'])))
            self.roi_results_table.setItem(row_idx, 2, QTableWidgetItem(str(r['positive'])))
            self.roi_results_table.setItem(row_idx, 3, QTableWidgetItem(str(r['negative'])))
            self.roi_results_table.setItem(row_idx, 4, QTableWidgetItem(f"{r['fraction']*100:.1f}%"))

        self.status_label.setText(f"Counted cells in {len(results)-1} ROI(s)")

    def _export_roi_counts(self):
        """Export ROI counts to CSV."""
        if self._roi_counts_data is None:
            QMessageBox.warning(self, "Error", "Run ROI counting first")
            return

        default_name = "roi_counts.csv"
        if self.current_file:
            default_name = f"{self.current_file.stem}_roi_counts.csv"
            default_dir = self.current_file.parent
        else:
            default_dir = Path.home()

        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Counts",
            str(default_dir / default_name),
            "CSV Files (*.csv)"
        )
        if not path:
            return

        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['roi', 'total', 'positive', 'negative', 'fraction'])
            writer.writeheader()
            writer.writerows(self._roi_counts_data)

        self.status_label.setText(f"Exported to {Path(path).name}")

    def _run_quantification(self):
        """Run regional quantification."""
        if self.cell_measurements is None:
            QMessageBox.warning(self, "Error", "Run colocalization first")
            return

        self.status_label.setText("Running quantification...")
        self.quant_btn.setEnabled(False)

        sample_id = self.sample_id_edit.text() or 'sample'

        # Determine output directory
        output_dir = None
        if self.export_csv_check.isChecked() and self.current_file:
            output_dir = self.current_file.parent / "BrainSlice_Results"

        from .workers import QuantificationWorker

        # Try to get registered atlas labels from alignment widget
        atlas_labels = None
        atlas_manager = None
        if hasattr(self, 'alignment_widget') and self.alignment_widget.atlas_loaded:
            atlas_labels = self.alignment_widget.get_registered_atlas_labels()
            atlas_manager = self.alignment_widget.atlas_manager

        # Fall back to stored atlas_labels or create dummy
        if atlas_labels is None:
            if self.atlas_labels is not None:
                # Use stored atlas_labels, extracting current slice if 3D
                atlas_labels = self._get_current_slice(self.atlas_labels)
                if atlas_labels is None:
                    atlas_labels = self.atlas_labels
            else:
                # Create dummy atlas_labels (all zeros = no regions)
                # Get current slice image to determine shape
                current_image = self._get_current_slice(self.red_channel)
                if current_image is not None:
                    atlas_labels = np.zeros(current_image.shape, dtype=np.int32)
                elif self.red_channel.ndim == 3:
                    atlas_labels = np.zeros(
                        (self.red_channel.shape[1], self.red_channel.shape[2]),
                        dtype=np.int32
                    )
                else:
                    atlas_labels = np.zeros(
                        (self.red_channel.shape[0], self.red_channel.shape[1]),
                        dtype=np.int32
                    )

        self.quant_worker = QuantificationWorker(
            self.cell_measurements,
            atlas_labels,
            atlas_manager=atlas_manager,
            output_dir=output_dir,
            sample_id=sample_id,
        )
        self.quant_worker.progress.connect(self._on_quant_progress)
        self.quant_worker.finished.connect(self._on_quant_finished)
        self.quant_worker.start()

    def _on_quant_progress(self, message: str):
        """Handle quantification progress updates."""
        self.status_label.setText(message)

    def _on_quant_finished(self, success: bool, message: str, cell_data, region_counts, summary):
        """Handle quantification completion."""
        self.quant_btn.setEnabled(True)

        if success:
            self.cell_measurements = cell_data
            self.region_counts = region_counts

            # Update results table
            self.results_table.setRowCount(0)
            metrics = [
                ('Total Cells', summary['total_cells']),
                ('Positive Cells', summary['positive_cells']),
                ('Negative Cells', summary['negative_cells']),
                ('Positive Fraction', f"{summary['positive_fraction']*100:.1f}%"),
                ('Regions with Cells', summary['regions_with_cells']),
                ('Top Region', summary['top_region']),
            ]

            for metric, value in metrics:
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(metric))
                self.results_table.setItem(row, 1, QTableWidgetItem(str(value)))

            self.status_label.setText(message)

            if self.export_csv_check.isChecked():
                output_dir = self.current_file.parent / "BrainSlice_Results"
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Results exported to:\n{output_dir}"
                )

        else:
            self.status_label.setText(f"Error: {message}")
