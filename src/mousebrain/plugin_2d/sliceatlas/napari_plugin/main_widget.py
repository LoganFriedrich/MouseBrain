"""
main_widget.py - Main BrainSlice napari widget

Provides a tabbed interface for:
1. Load - Load images and select channels
2. Insets - Add high-resolution region-of-interest overlays
3. Detect - Nuclei detection (Threshold default, StarDist/Cellpose optional)
4. Colocalize - Measure signal in soma regions and classify positive/negative
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
        self._coloc_background_surface = None
        self._diag_canvas = None
        self._pixel_size_um: Optional[float] = None
        self._size_manually_set: bool = False
        self._peeked_metadata: Optional[Dict[str, Any]] = None

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
        self.red_channel_name_label = QLabel("")
        self.red_channel_name_label.setStyleSheet("color: #FF8888; font-size: 10px;")
        red_layout.addWidget(self.red_channel_name_label)
        channel_layout.addLayout(red_layout)

        # Green (signal) channel
        green_layout = QHBoxLayout()
        green_layout.addWidget(QLabel("Signal (green):"))
        self.green_channel_spin = QSpinBox()
        self.green_channel_spin.setRange(0, 10)
        self.green_channel_spin.setValue(0)  # Default to channel 0 (488nm)
        green_layout.addWidget(self.green_channel_spin)
        self.green_channel_name_label = QLabel("")
        self.green_channel_name_label.setStyleSheet("color: #88FF88; font-size: 10px;")
        green_layout.addWidget(self.green_channel_name_label)
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

        # ── Backend & Model Selection ──
        backend_group = QGroupBox("Detection Backend")
        backend_layout = QVBoxLayout()

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['Threshold', 'StarDist', 'Cellpose'])
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        backend_row.addWidget(self.backend_combo)
        backend_layout.addLayout(backend_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            '2D_versatile_fluo',
            '2D_versatile_he',
            '2D_paper_dsb2018',
        ])
        model_row.addWidget(self.model_combo)
        self.model_row_widget = QWidget()
        self.model_row_widget.setLayout(model_row)
        self.model_row_widget.setVisible(False)  # Hidden for Threshold (default)
        backend_layout.addWidget(self.model_row_widget)

        backend_group.setLayout(backend_layout)
        layout.addWidget(backend_group)

        # ── Preprocessing ──
        preproc_group = QGroupBox("Preprocessing")
        preproc_layout = QVBoxLayout()

        # Background subtraction
        bgsub_row = QHBoxLayout()
        self.preproc_bgsub_check = QCheckBox("Background subtraction")
        self.preproc_bgsub_check.setToolTip(
            "Subtract slowly-varying illumination. Helps detect dim nuclei\n"
            "in unevenly-lit regions. Recommended for most images."
        )
        bgsub_row.addWidget(self.preproc_bgsub_check)
        bgsub_row.addWidget(QLabel("sigma:"))
        self.preproc_bgsub_sigma_spin = QDoubleSpinBox()
        self.preproc_bgsub_sigma_spin.setRange(10.0, 200.0)
        self.preproc_bgsub_sigma_spin.setSingleStep(10.0)
        self.preproc_bgsub_sigma_spin.setValue(50.0)
        bgsub_row.addWidget(self.preproc_bgsub_sigma_spin)
        preproc_layout.addLayout(bgsub_row)

        # CLAHE
        clahe_row = QHBoxLayout()
        self.preproc_clahe_check = QCheckBox("CLAHE")
        self.preproc_clahe_check.setToolTip(
            "Contrast Limited Adaptive Histogram Equalization.\n"
            "Enhances local contrast so dim nuclei near bright ones become visible."
        )
        clahe_row.addWidget(self.preproc_clahe_check)
        clahe_row.addWidget(QLabel("clip:"))
        self.preproc_clahe_clip_spin = QDoubleSpinBox()
        self.preproc_clahe_clip_spin.setRange(0.005, 0.10)
        self.preproc_clahe_clip_spin.setSingleStep(0.005)
        self.preproc_clahe_clip_spin.setDecimals(3)
        self.preproc_clahe_clip_spin.setValue(0.02)
        clahe_row.addWidget(self.preproc_clahe_clip_spin)
        preproc_layout.addLayout(clahe_row)

        # Gaussian blur
        gauss_row = QHBoxLayout()
        self.preproc_gauss_check = QCheckBox("Gaussian blur")
        self.preproc_gauss_check.setToolTip(
            "Light denoising. Smooths speckle noise and small debris.\n"
            "sigma=1.0 is light, 2.0 is moderate."
        )
        gauss_row.addWidget(self.preproc_gauss_check)
        gauss_row.addWidget(QLabel("sigma:"))
        self.preproc_gauss_sigma_spin = QDoubleSpinBox()
        self.preproc_gauss_sigma_spin.setRange(0.5, 5.0)
        self.preproc_gauss_sigma_spin.setSingleStep(0.5)
        self.preproc_gauss_sigma_spin.setValue(1.0)
        gauss_row.addWidget(self.preproc_gauss_sigma_spin)
        preproc_layout.addLayout(gauss_row)

        # Preview button
        self.preproc_preview_btn = QPushButton("Preview Preprocessing")
        self.preproc_preview_btn.setToolTip("Show preprocessing result in napari")
        self.preproc_preview_btn.clicked.connect(self._preview_preprocessing)
        self.preproc_preview_btn.setEnabled(False)
        preproc_layout.addWidget(self.preproc_preview_btn)

        preproc_group.setLayout(preproc_layout)
        self.preproc_group = preproc_group
        self.preproc_group.setVisible(False)  # Hidden for Threshold (default)
        layout.addWidget(preproc_group)

        # ── Detection Parameters ──
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()

        # Threshold parameters (visible by default — Threshold is default backend)
        self.threshold_params_widget = QWidget()
        thresh_det_layout = QVBoxLayout()
        thresh_det_layout.setContentsMargins(0, 0, 0, 0)

        thresh_method_row = QHBoxLayout()
        thresh_method_row.addWidget(QLabel("Method:"))
        self.thresh_detect_method_combo = QComboBox()
        self.thresh_detect_method_combo.addItems(['otsu', 'percentile', 'manual'])
        self.thresh_detect_method_combo.setToolTip(
            "Otsu: automatic threshold (good default for bimodal images)\n"
            "Percentile: threshold at Nth percentile intensity\n"
            "Manual: user-specified threshold value"
        )
        self.thresh_detect_method_combo.currentTextChanged.connect(
            self._on_thresh_detect_method_changed
        )
        thresh_method_row.addWidget(self.thresh_detect_method_combo)
        thresh_det_layout.addLayout(thresh_method_row)

        thresh_pct_row = QHBoxLayout()
        thresh_pct_row.addWidget(QLabel("Percentile:"))
        self.thresh_detect_percentile_spin = QDoubleSpinBox()
        self.thresh_detect_percentile_spin.setRange(80.0, 99.9)
        self.thresh_detect_percentile_spin.setSingleStep(0.5)
        self.thresh_detect_percentile_spin.setValue(99.0)
        self.thresh_detect_percentile_spin.setToolTip(
            "Intensity percentile to use as threshold.\n"
            "99 = only brightest 1% of pixels."
        )
        thresh_pct_row.addWidget(self.thresh_detect_percentile_spin)
        self.thresh_percentile_row = QWidget()
        self.thresh_percentile_row.setLayout(thresh_pct_row)
        self.thresh_percentile_row.setVisible(False)
        thresh_det_layout.addWidget(self.thresh_percentile_row)

        thresh_manual_row = QHBoxLayout()
        thresh_manual_row.addWidget(QLabel("Manual value:"))
        self.thresh_detect_manual_spin = QDoubleSpinBox()
        self.thresh_detect_manual_spin.setRange(0.0, 65535.0)
        self.thresh_detect_manual_spin.setSingleStep(100.0)
        self.thresh_detect_manual_spin.setValue(500.0)
        thresh_manual_row.addWidget(self.thresh_detect_manual_spin)
        self.thresh_manual_row = QWidget()
        self.thresh_manual_row.setLayout(thresh_manual_row)
        self.thresh_manual_row.setVisible(False)
        thresh_det_layout.addWidget(self.thresh_manual_row)

        # Hysteresis thresholding (captures full nucleus extent)
        self.thresh_hysteresis_check = QCheckBox("Hysteresis (expand to full boundary)")
        self.thresh_hysteresis_check.setChecked(True)
        self.thresh_hysteresis_check.setToolTip(
            "Use hysteresis thresholding to capture full nucleus extent.\n"
            "The main threshold finds bright cores; a lower threshold\n"
            "expands to the true boundary of each nucleus.\n"
            "Fixes undersized detections and catches dimmer nuclei."
        )
        self.thresh_hysteresis_check.stateChanged.connect(
            self._on_hysteresis_check_changed
        )
        thresh_det_layout.addWidget(self.thresh_hysteresis_check)

        thresh_hyst_row = QHBoxLayout()
        thresh_hyst_row.addWidget(QLabel("Low fraction:"))
        self.thresh_hysteresis_low_spin = QDoubleSpinBox()
        self.thresh_hysteresis_low_spin.setRange(0.1, 0.9)
        self.thresh_hysteresis_low_spin.setSingleStep(0.05)
        self.thresh_hysteresis_low_spin.setValue(0.5)
        self.thresh_hysteresis_low_spin.setToolTip(
            "Low threshold = high threshold x this fraction.\n"
            "0.5 = low is half the high threshold (good default).\n"
            "Lower values capture more of each nucleus boundary\n"
            "but may merge nearby objects."
        )
        thresh_hyst_row.addWidget(self.thresh_hysteresis_low_spin)
        self.thresh_hysteresis_row = QWidget()
        self.thresh_hysteresis_row.setLayout(thresh_hyst_row)
        thresh_det_layout.addWidget(self.thresh_hysteresis_row)

        thresh_opening_row = QHBoxLayout()
        thresh_opening_row.addWidget(QLabel("Opening radius:"))
        self.thresh_opening_spin = QSpinBox()
        self.thresh_opening_spin.setRange(0, 10)
        self.thresh_opening_spin.setValue(0)
        self.thresh_opening_spin.setToolTip(
            "Morphological opening to remove small speckle noise.\n"
            "0 = disabled (default). 1-2 = light cleanup.\n"
            "Warning: opening erodes nucleus boundaries."
        )
        thresh_opening_row.addWidget(self.thresh_opening_spin)
        thresh_det_layout.addLayout(thresh_opening_row)

        thresh_gauss_row = QHBoxLayout()
        thresh_gauss_row.addWidget(QLabel("Gaussian sigma:"))
        self.thresh_gauss_spin = QDoubleSpinBox()
        self.thresh_gauss_spin.setRange(0.0, 5.0)
        self.thresh_gauss_spin.setSingleStep(0.5)
        self.thresh_gauss_spin.setValue(1.0)
        self.thresh_gauss_spin.setToolTip(
            "Gaussian blur before thresholding. Smooths noise.\n"
            "0 = no blur. 1.0 = light smoothing."
        )
        thresh_gauss_row.addWidget(self.thresh_gauss_spin)
        thresh_det_layout.addLayout(thresh_gauss_row)

        # ── Morphology cleanup group ──
        morph_group = QGroupBox("Post-Detection Cleanup")
        morph_layout = QVBoxLayout()
        morph_layout.setContentsMargins(4, 4, 4, 4)

        closing_row = QHBoxLayout()
        closing_row.addWidget(QLabel("Closing radius:"))
        self.thresh_closing_spin = QSpinBox()
        self.thresh_closing_spin.setRange(0, 10)
        self.thresh_closing_spin.setValue(0)
        self.thresh_closing_spin.setToolTip(
            "Morphological closing to bridge small gaps in nucleus masks.\n"
            "0 = disabled (default). 1-3 = typical for fragmented nuclei."
        )
        closing_row.addWidget(self.thresh_closing_spin)
        morph_layout.addLayout(closing_row)

        self.thresh_fill_holes_check = QCheckBox("Fill holes")
        self.thresh_fill_holes_check.setChecked(True)
        self.thresh_fill_holes_check.setToolTip(
            "Fill internal holes in binary mask before labeling.\n"
            "Saturated nuclei can have internal voids from noise."
        )
        morph_layout.addWidget(self.thresh_fill_holes_check)

        self.thresh_split_check = QCheckBox("Split touching nuclei (watershed)")
        self.thresh_split_check.setChecked(False)
        self.thresh_split_check.setToolTip(
            "Use distance-transform watershed to split merged nuclei.\n"
            "Useful when hysteresis merges two adjacent bright nuclei."
        )
        self.thresh_split_check.stateChanged.connect(
            self._on_split_check_changed
        )
        morph_layout.addWidget(self.thresh_split_check)

        split_fp_row = QHBoxLayout()
        split_fp_row.addWidget(QLabel("Split footprint:"))
        self.thresh_split_footprint_spin = QSpinBox()
        self.thresh_split_footprint_spin.setRange(3, 30)
        self.thresh_split_footprint_spin.setValue(10)
        self.thresh_split_footprint_spin.setToolTip(
            "Footprint size for watershed peak detection.\n"
            "Larger = peaks must be farther apart to split.\n"
            "10 = good default for typical nuclear spacing."
        )
        split_fp_row.addWidget(self.thresh_split_footprint_spin)
        self.thresh_split_footprint_row = QWidget()
        self.thresh_split_footprint_row.setLayout(split_fp_row)
        self.thresh_split_footprint_row.setVisible(False)
        morph_layout.addWidget(self.thresh_split_footprint_row)

        solidity_row = QHBoxLayout()
        solidity_row.addWidget(QLabel("Min solidity:"))
        self.thresh_solidity_spin = QDoubleSpinBox()
        self.thresh_solidity_spin.setRange(0.0, 1.0)
        self.thresh_solidity_spin.setSingleStep(0.05)
        self.thresh_solidity_spin.setValue(0.0)
        self.thresh_solidity_spin.setToolTip(
            "Minimum solidity (area / convex_hull_area).\n"
            "0 = no filtering. Nuclei are typically > 0.8.\n"
            "Debris/artifacts are often < 0.7."
        )
        solidity_row.addWidget(self.thresh_solidity_spin)
        morph_layout.addLayout(solidity_row)

        circ_row = QHBoxLayout()
        circ_row.addWidget(QLabel("Min circularity:"))
        self.thresh_circularity_spin = QDoubleSpinBox()
        self.thresh_circularity_spin.setRange(0.0, 1.0)
        self.thresh_circularity_spin.setSingleStep(0.05)
        self.thresh_circularity_spin.setValue(0.0)
        self.thresh_circularity_spin.setToolTip(
            "Minimum circularity (4*pi*area/perimeter^2).\n"
            "0 = no filtering. Perfect circle = 1.0.\n"
            "Use 0.4-0.6 to reject elongated artifacts."
        )
        circ_row.addWidget(self.thresh_circularity_spin)
        morph_layout.addLayout(circ_row)

        morph_group.setLayout(morph_layout)
        thresh_det_layout.addWidget(morph_group)

        self.threshold_params_widget.setLayout(thresh_det_layout)
        param_layout.addWidget(self.threshold_params_widget)

        # StarDist parameters (hidden by default)
        self.stardist_params_widget = QWidget()
        stardist_layout = QVBoxLayout()
        stardist_layout.setContentsMargins(0, 0, 0, 0)

        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Prob Threshold:"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setSingleStep(0.05)
        self.prob_spin.setValue(0.5)
        self.prob_spin.setToolTip("Lower = more detections (more false positives)")
        prob_layout.addWidget(self.prob_spin)
        stardist_layout.addLayout(prob_layout)

        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS Threshold:"))
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.4)
        self.nms_spin.setToolTip("Controls overlap tolerance between adjacent nuclei")
        nms_layout.addWidget(self.nms_spin)
        stardist_layout.addLayout(nms_layout)

        self.stardist_params_widget.setLayout(stardist_layout)
        self.stardist_params_widget.setVisible(False)
        param_layout.addWidget(self.stardist_params_widget)

        # Cellpose parameters (hidden by default)
        self.cellpose_params_widget = QWidget()
        cellpose_layout = QVBoxLayout()
        cellpose_layout.setContentsMargins(0, 0, 0, 0)

        diam_layout = QHBoxLayout()
        diam_layout.addWidget(QLabel("Diameter:"))
        self.diameter_spin = QSpinBox()
        self.diameter_spin.setRange(0, 500)
        self.diameter_spin.setValue(30)
        self.diameter_spin.setToolTip(
            "Expected nucleus diameter in pixels. 0 = auto-estimate.\n"
            "Typical: 10-30 for confocal, 30-80 for widefield."
        )
        diam_layout.addWidget(self.diameter_spin)
        cellpose_layout.addLayout(diam_layout)

        self.cellpose_params_widget.setLayout(cellpose_layout)
        self.cellpose_params_widget.setVisible(False)
        param_layout.addWidget(self.cellpose_params_widget)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # ── Post-Detection Filters ──
        filter_group = QGroupBox("Post-Detection Filters")
        filter_layout = QVBoxLayout()

        # Size filtering
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min/Max Area:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10000)
        self.min_area_spin.setValue(10)
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 100000)
        self.max_area_spin.setValue(5000)
        size_layout.addWidget(self.min_area_spin)
        size_layout.addWidget(QLabel("-"))
        size_layout.addWidget(self.max_area_spin)
        filter_layout.addLayout(size_layout)

        # Physical area label (updated when pixel size is known)
        self._area_um_label = QLabel("")
        self._area_um_label.setStyleSheet("color: gray; font-size: 10px; margin-left: 4px;")
        filter_layout.addWidget(self._area_um_label)

        # Connect spinbox changes to update area label and track manual edits
        self.min_area_spin.valueChanged.connect(self._on_area_spin_changed)
        self.max_area_spin.valueChanged.connect(self._on_area_spin_changed)

        # Solidity filter
        solidity_layout = QHBoxLayout()
        solidity_layout.addWidget(QLabel("Min Solidity:"))
        self.min_solidity_spin = QDoubleSpinBox()
        self.min_solidity_spin.setRange(0.0, 1.0)
        self.min_solidity_spin.setSingleStep(0.05)
        self.min_solidity_spin.setValue(0.0)
        self.min_solidity_spin.setToolTip(
            "Solidity = area / convex_hull_area.\n"
            "Debris/artifacts typically < 0.7, real nuclei > 0.8.\n"
            "0 = no filtering."
        )
        solidity_layout.addWidget(self.min_solidity_spin)
        filter_layout.addLayout(solidity_layout)

        # Border-touching removal
        self.remove_border_check = QCheckBox("Remove border-touching nuclei")
        self.remove_border_check.setToolTip(
            "Remove partial nuclei that touch image edges.\n"
            "These have incorrect area and intensity measurements."
        )
        filter_layout.addWidget(self.remove_border_check)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # ── Run Button ──
        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self._run_detection)
        self.detect_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)

        # ── Results & Metrics ──
        self.detect_result_label = QLabel("")
        layout.addWidget(self.detect_result_label)

        self.detect_metrics_label = QLabel("")
        self.detect_metrics_label.setWordWrap(True)
        self.detect_metrics_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.detect_metrics_label)

        layout.addStretch()
        return widget

    def _on_backend_changed(self, backend_text: str):
        """Toggle visibility of backend-specific parameters."""
        is_threshold = backend_text == 'Threshold'
        is_stardist = backend_text == 'StarDist'
        is_cellpose = backend_text == 'Cellpose'

        # Threshold params
        self.threshold_params_widget.setVisible(is_threshold)

        # StarDist/Cellpose need model, preprocessing
        self.model_row_widget.setVisible(not is_threshold)
        self.preproc_group.setVisible(not is_threshold)
        self.stardist_params_widget.setVisible(is_stardist)
        self.cellpose_params_widget.setVisible(is_cellpose)

        # Update model list
        if is_stardist:
            self.model_combo.clear()
            self.model_combo.addItems([
                '2D_versatile_fluo',
                '2D_versatile_he',
                '2D_paper_dsb2018',
            ])
        elif is_cellpose:
            self.model_combo.clear()
            self.model_combo.addItems([
                'nuclei',
                'cyto',
                'cyto2',
                'cyto3',
            ])

    def _on_thresh_detect_method_changed(self, method: str):
        """Toggle threshold detection sub-parameters."""
        self.thresh_percentile_row.setVisible(method == 'percentile')
        self.thresh_manual_row.setVisible(method == 'manual')

    def _on_hysteresis_check_changed(self, state):
        """Toggle hysteresis low fraction visibility."""
        self.thresh_hysteresis_row.setVisible(bool(state))

    def _on_split_check_changed(self, state):
        """Toggle watershed split footprint visibility."""
        self.thresh_split_footprint_row.setVisible(bool(state))

    def _on_thresh_method_changed(self, method: str):
        """Toggle visibility of area_fraction parameter."""
        self.area_fraction_widget.setVisible(method == 'area_fraction')

    def _on_coloc_mode_changed(self, mode_text: str):
        """Show/hide Channel 2 controls based on mode."""
        is_dual = mode_text == 'Dual Channel'
        self.ch2_group.setVisible(is_dual)

    def _preview_preprocessing(self):
        """Show preprocessing effect on current nuclear channel in napari."""
        image = self._get_current_slice(self.red_channel)
        if image is None:
            return

        from ..core.detection import preprocess_for_detection

        preprocessed = preprocess_for_detection(
            image,
            background_subtraction=self.preproc_bgsub_check.isChecked(),
            bg_sigma=self.preproc_bgsub_sigma_spin.value(),
            clahe=self.preproc_clahe_check.isChecked(),
            clahe_clip_limit=self.preproc_clahe_clip_spin.value(),
            gaussian_sigma=(self.preproc_gauss_sigma_spin.value()
                            if self.preproc_gauss_check.isChecked() else 0.0),
        )

        # Remove old preview layer
        for layer in list(self.viewer.layers):
            if 'Preprocessed' in layer.name:
                self.viewer.layers.remove(layer)

        self.viewer.add_image(
            preprocessed,
            name="Preprocessed (nuclear)",
            colormap='gray',
            blending='additive',
        )

    def _create_coloc_tab(self) -> QWidget:
        """Create the Colocalize tab for intensity measurement."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Mode selector: Single or Dual channel
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.coloc_mode_combo = QComboBox()
        self.coloc_mode_combo.addItems(['Single Channel', 'Dual Channel'])
        self.coloc_mode_combo.currentTextChanged.connect(self._on_coloc_mode_changed)
        mode_layout.addWidget(self.coloc_mode_combo)
        layout.addLayout(mode_layout)

        # Background estimation
        bg_group = QGroupBox("Background Estimation")
        bg_layout = QVBoxLayout()

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(['gmm', 'percentile', 'mode', 'mean'])
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

        # Local background estimation
        self.bg_local_check = QCheckBox("Local background estimation")
        self.bg_local_check.setToolTip(
            "Estimate background spatially across the tissue instead of\n"
            "a single tissue-wide value. Recommended when background\n"
            "varies across the section (e.g., uneven illumination)."
        )
        bg_layout.addWidget(self.bg_local_check)

        local_bg_row = QHBoxLayout()
        local_bg_row.addWidget(QLabel("Block size:"))
        self.bg_block_size_spin = QSpinBox()
        self.bg_block_size_spin.setRange(64, 1024)
        self.bg_block_size_spin.setSingleStep(64)
        self.bg_block_size_spin.setValue(256)
        self.bg_block_size_spin.setToolTip("Size of spatial blocks for local estimation (pixels)")
        local_bg_row.addWidget(self.bg_block_size_spin)
        bg_layout.addLayout(local_bg_row)

        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)

        # Soma measurement
        soma_group = QGroupBox("Soma Measurement")
        soma_layout = QVBoxLayout()

        soma_dil_row = QHBoxLayout()
        soma_dil_row.addWidget(QLabel("Soma dilation (px):"))
        self.soma_dilation_spin = QSpinBox()
        self.soma_dilation_spin.setRange(0, 50)
        self.soma_dilation_spin.setValue(5)
        self.soma_dilation_spin.setToolTip(
            "Dilate each nucleus ROI to include the surrounding soma.\n"
            "Signal (eYFP, etc.) is cytoplasmic, not nuclear — measure\n"
            "intensity in this dilated region instead of the nucleus alone.\n"
            "0 = measure only within nucleus (old behavior)."
        )
        soma_dil_row.addWidget(self.soma_dilation_spin)
        soma_layout.addLayout(soma_dil_row)

        soma_group.setLayout(soma_layout)
        layout.addWidget(soma_group)

        # Classification threshold
        thresh_group = QGroupBox("Positive/Negative Classification")
        thresh_layout = QVBoxLayout()

        thresh_method_layout = QHBoxLayout()
        thresh_method_layout.addWidget(QLabel("Method:"))
        self.thresh_method_combo = QComboBox()
        self.thresh_method_combo.addItems(['fold_change', 'area_fraction', 'absolute', 'percentile'])
        self.thresh_method_combo.currentTextChanged.connect(self._on_thresh_method_changed)
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

        # Area fraction parameter (visible only when area_fraction method selected)
        area_frac_layout = QHBoxLayout()
        area_frac_layout.addWidget(QLabel("Area Fraction:"))
        self.area_fraction_spin = QDoubleSpinBox()
        self.area_fraction_spin.setRange(0.1, 1.0)
        self.area_fraction_spin.setSingleStep(0.05)
        self.area_fraction_spin.setValue(0.5)
        self.area_fraction_spin.setToolTip(
            "Fraction of nucleus pixels that must exceed threshold.\n"
            "0.5 = at least 50% of pixels must be bright enough."
        )
        area_frac_layout.addWidget(self.area_fraction_spin)
        self.area_fraction_widget = QWidget()
        self.area_fraction_widget.setLayout(area_frac_layout)
        self.area_fraction_widget.setVisible(False)
        thresh_layout.addWidget(self.area_fraction_widget)

        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)

        # --- Channel 2 params (visible only in Dual mode) ---
        self.ch2_group = QGroupBox("Channel 2 (Green / eYFP)")
        ch2_layout = QVBoxLayout()

        ch2_bg_layout = QHBoxLayout()
        ch2_bg_layout.addWidget(QLabel("BG method:"))
        self.bg_method_combo_ch2 = QComboBox()
        self.bg_method_combo_ch2.addItems(['gmm', 'percentile', 'mode', 'mean'])
        ch2_bg_layout.addWidget(self.bg_method_combo_ch2)
        ch2_layout.addLayout(ch2_bg_layout)

        ch2_dil_layout = QHBoxLayout()
        ch2_dil_layout.addWidget(QLabel("BG exclusion radius:"))
        self.bg_dilation_spin_ch2 = QSpinBox()
        self.bg_dilation_spin_ch2.setRange(5, 200)
        self.bg_dilation_spin_ch2.setValue(50)
        self.bg_dilation_spin_ch2.setToolTip("Background exclusion dilation for green channel.\nShould be generous to exclude eYFP+ somas.")
        ch2_dil_layout.addWidget(self.bg_dilation_spin_ch2)
        ch2_layout.addLayout(ch2_dil_layout)

        ch2_soma_layout = QHBoxLayout()
        ch2_soma_layout.addWidget(QLabel("Soma dilation (px):"))
        self.soma_dilation_spin_ch2 = QSpinBox()
        self.soma_dilation_spin_ch2.setRange(0, 50)
        self.soma_dilation_spin_ch2.setValue(15)
        self.soma_dilation_spin_ch2.setToolTip("eYFP is cytoplasmic — dilate generously to capture soma signal.\nRecommended: 15-20px.")
        ch2_soma_layout.addWidget(self.soma_dilation_spin_ch2)
        ch2_layout.addLayout(ch2_soma_layout)

        ch2_thresh_layout = QHBoxLayout()
        ch2_thresh_layout.addWidget(QLabel("Threshold:"))
        self.thresh_value_spin_ch2 = QDoubleSpinBox()
        self.thresh_value_spin_ch2.setRange(0.1, 100.0)
        self.thresh_value_spin_ch2.setValue(2.0)
        self.thresh_value_spin_ch2.setSingleStep(0.5)
        ch2_thresh_layout.addWidget(self.thresh_value_spin_ch2)
        ch2_layout.addLayout(ch2_thresh_layout)

        self.ch2_group.setLayout(ch2_layout)
        self.ch2_group.setVisible(False)
        layout.addWidget(self.ch2_group)

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

        # --- Run History ---
        history_group = QGroupBox("Run History")
        history_layout = QVBoxLayout()

        self.run_history_table = QTableWidget()
        self.run_history_table.setColumnCount(5)
        self.run_history_table.setHorizontalHeaderLabels(
            ["Run ID", "Date", "Positive", "Fraction", "Method"]
        )
        self.run_history_table.setMaximumHeight(150)
        self.run_history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.run_history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        history_layout.addWidget(self.run_history_table)

        history_btn_layout = QHBoxLayout()
        self.load_run_btn = QPushButton("Load Selected Run")
        self.load_run_btn.clicked.connect(self._load_selected_run)
        history_btn_layout.addWidget(self.load_run_btn)

        self.refresh_history_btn = QPushButton("Refresh")
        self.refresh_history_btn.clicked.connect(self._refresh_run_history)
        history_btn_layout.addWidget(self.refresh_history_btn)
        history_layout.addLayout(history_btn_layout)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

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
            'Background Surface',
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

            # Peek at metadata for channel auto-detection and size calibration
            self._peek_and_configure(self.current_file)

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

    def _peek_and_configure(self, file_path: Path):
        """Peek at file metadata to auto-detect channels and calibrate sizes."""
        try:
            from ..core.io import peek_metadata, guess_channel_roles

            meta = peek_metadata(file_path)
            self._peeked_metadata = meta

            # Auto-detect channel roles from channel names/wavelengths
            channels = meta.get('channels', [])
            if channels:
                roles = guess_channel_roles(meta)
                self.red_channel_spin.setValue(roles['nuclear'])
                self.green_channel_spin.setValue(roles['signal'])

                # Show channel names next to spinboxes
                self._update_channel_labels(channels)
            else:
                self.red_channel_name_label.setText("")
                self.green_channel_name_label.setText("")

            # Store pixel size and calibrate detection parameters
            voxel = meta.get('voxel_size_um')
            if voxel and voxel.get('x', 1.0) != 1.0:
                self._pixel_size_um = voxel['x']
                self._size_manually_set = False
                self._calibrate_from_pixel_size()
            else:
                self._pixel_size_um = None

            # Update area label if it exists
            if hasattr(self, '_area_um_label'):
                self._update_area_label()

        except Exception as e:
            print(f"[BrainSlice] Metadata peek failed (non-fatal): {e}")

    def _update_channel_labels(self, channels):
        """Update channel name labels next to spinboxes."""
        red_idx = self.red_channel_spin.value()
        green_idx = self.green_channel_spin.value()
        if red_idx < len(channels):
            self.red_channel_name_label.setText(channels[red_idx])
        else:
            self.red_channel_name_label.setText("")
        if green_idx < len(channels):
            self.green_channel_name_label.setText(channels[green_idx])
        else:
            self.green_channel_name_label.setText("")

    def _calibrate_from_pixel_size(self):
        """Auto-set detection size filters based on pixel size in microns."""
        if self._pixel_size_um is None or self._pixel_size_um <= 0:
            return
        if self._size_manually_set:
            return

        px = self._pixel_size_um
        pi = 3.14159

        # Min area: 5 um diameter nucleus -> pixel area
        min_diam_um = 5.0
        min_area_px = max(3, int(pi * (min_diam_um / (2 * px)) ** 2))

        # Max area: 30 um diameter nucleus -> pixel area
        max_diam_um = 30.0
        max_area_px = max(min_area_px + 10, int(pi * (max_diam_um / (2 * px)) ** 2))

        # Cellpose diameter: 12 um typical nucleus
        typical_diam_um = 12.0
        diameter_px = max(5, int(typical_diam_um / px))

        self.min_area_spin.setValue(min_area_px)
        self.max_area_spin.setValue(max_area_px)
        self.diameter_spin.setValue(diameter_px)

        print(f"[BrainSlice] Auto-calibrated from {px:.3f} um/px: "
              f"area={min_area_px}-{max_area_px} px, diameter={diameter_px} px")

    def _update_area_label(self):
        """Update the physical area label below min/max area spinboxes."""
        if not hasattr(self, '_area_um_label'):
            return
        if self._pixel_size_um is None or self._pixel_size_um <= 0:
            self._area_um_label.setText("")
            return

        px2 = self._pixel_size_um ** 2
        min_um2 = self.min_area_spin.value() * px2
        max_um2 = self.max_area_spin.value() * px2
        self._area_um_label.setText(
            f"({min_um2:.0f} - {max_um2:,.0f} um\u00b2 at {self._pixel_size_um:.2f} um/px)"
        )

    def _on_area_spin_changed(self):
        """Handle area spinbox changes — mark as manual and update label."""
        self._size_manually_set = True
        self._update_area_label()

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

            # Enable detection and preprocessing preview
            self.detect_btn.setEnabled(True)
            self.preproc_preview_btn.setEnabled(True)

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
            try:
                # Apply rotation
                red = self._apply_rotation(red)
                green = self._apply_rotation(green)

                self.red_channel = red
                self.green_channel = green
                self.metadata = metadata
                print(f"[BrainSlice] Load finished: red={red.shape}, green={green.shape}")

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

                # Update pixel size from full load metadata (in case peek missed it)
                voxel = metadata.get('voxel_size_um')
                if voxel and voxel.get('x', 1.0) != 1.0:
                    self._pixel_size_um = voxel['x']
                    if not self._size_manually_set:
                        self._calibrate_from_pixel_size()
                    self._update_area_label()

                # Update channel name labels
                channels = metadata.get('channels', [])
                if channels:
                    self._update_channel_labels(channels)

                # Update UI
                z_info = metadata.get('z_projection', '')
                if z_info:
                    z_info = f", Z: {z_info}"
                self.status_label.setText(f"Loaded: {self.current_file.name}{z_info}")

                # Build metadata display with pixel size info
                meta_parts = [
                    f"Shape: {metadata.get('shape', 'Unknown')}",
                    f"Channels: {metadata.get('channels', 'Unknown')}",
                ]
                if self._pixel_size_um is not None:
                    meta_parts.append(f"Pixel: {self._pixel_size_um:.3f} um/px")
                self.metadata_label.setText(", ".join(meta_parts))

            except Exception as e:
                import traceback
                print(f"[BrainSlice] ERROR in _on_load_finished: {e}")
                traceback.print_exc()

            # Always enable detection and preview buttons after successful load
            self.detect_btn.setEnabled(True)
            self.preproc_preview_btn.setEnabled(True)
            print("[BrainSlice] Detect button enabled")

            # Notify inset widget that base is loaded
            try:
                self.inset_widget.on_base_loaded()
            except Exception as e:
                print(f"[BrainSlice] Inset widget notification failed (non-fatal): {e}")

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
        self.detect_metrics_label.setText("")

        try:
            # Build params dict with all new controls
            backend = self.backend_combo.currentText().lower()
            params = {
                'backend': backend,
                # Filters (shared across all backends)
                'filter_size': True,
                'min_area': self.min_area_spin.value(),
                'max_area': self.max_area_spin.value(),
                'min_solidity': self.min_solidity_spin.value(),
                'remove_border': self.remove_border_check.isChecked(),
            }

            if backend == 'threshold':
                # Threshold-specific params
                params['threshold_method'] = self.thresh_detect_method_combo.currentText()
                params['threshold_percentile'] = self.thresh_detect_percentile_spin.value()
                params['manual_threshold'] = self.thresh_detect_manual_spin.value()
                params['opening_radius'] = self.thresh_opening_spin.value()
                params['closing_radius'] = self.thresh_closing_spin.value()
                params['fill_holes'] = self.thresh_fill_holes_check.isChecked()
                params['split_touching'] = self.thresh_split_check.isChecked()
                params['split_footprint_size'] = self.thresh_split_footprint_spin.value()
                params['gaussian_sigma'] = self.thresh_gauss_spin.value()
                params['use_hysteresis'] = self.thresh_hysteresis_check.isChecked()
                params['hysteresis_low_fraction'] = self.thresh_hysteresis_low_spin.value()
                params['min_solidity'] = self.thresh_solidity_spin.value()
                params['min_circularity'] = self.thresh_circularity_spin.value()
            else:
                # StarDist / Cellpose params
                params['model'] = self.model_combo.currentText()
                params['prob_thresh'] = self.prob_spin.value()
                params['nms_thresh'] = self.nms_spin.value()
                params['diameter'] = self.diameter_spin.value()
                params['background_subtraction'] = self.preproc_bgsub_check.isChecked()
                params['bg_sigma'] = self.preproc_bgsub_sigma_spin.value()
                params['clahe'] = self.preproc_clahe_check.isChecked()
                params['clahe_clip_limit'] = self.preproc_clahe_clip_spin.value()
                params['gaussian_sigma'] = (self.preproc_gauss_sigma_spin.value()
                                            if self.preproc_gauss_check.isChecked() else 0.0)
                params['auto_n_tiles'] = True

            print(f"[BrainSlice] Detection params: backend={backend}")

            # Log to tracker
            if self.tracker:
                sample_id = self.sample_id_edit.text() or self.current_file.stem
                self.last_run_id = self.tracker.log_detection(
                    sample_id=sample_id,
                    model=params.get('model', backend),
                    prob_thresh=params.get('prob_thresh', 0.0),
                    nms_thresh=params.get('nms_thresh', 0.0),
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

                print(f"[BrainSlice] Starting detection worker: image shape={image.shape}, dtype={image.dtype}")
                from .workers import DetectionWorker
                self.detection_worker = DetectionWorker(image, params)
                self.detection_worker.progress.connect(self._on_detect_progress)
                self.detection_worker.finished.connect(self._on_detect_finished)
                self.detection_worker.start()
                print("[BrainSlice] Detection worker started (model loading may take a moment...)")

        except Exception as e:
            import traceback
            print(f"[BrainSlice] ERROR in _run_detection: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Detection error: {e}")
            self.detect_btn.setEnabled(True)
            QMessageBox.critical(self, "Detection Error", f"Failed to start detection:\n{e}")

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
                name=f"Nuclei ({count})",
                contour=2,
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

    def _on_detect_finished(self, success: bool, message: str, count: int, labels, metrics=None):
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

            self.viewer.add_labels(labels, name=f"Nuclei ({count})", contour=2)

            # Update UI
            self.status_label.setText(message)
            self.detect_result_label.setText(f"Detected {count} nuclei")

            # Display detection metrics
            if metrics:
                self._display_detection_metrics(metrics)

            # Enable colocalization
            self.coloc_btn.setEnabled(True)

        else:
            self.status_label.setText(f"Error: {message}")
            self.detect_metrics_label.setText("")
            if self.tracker and self.last_run_id:
                self.tracker.update_status(self.last_run_id, status='failed')

    def _display_detection_metrics(self, metrics: dict):
        """Display detection metrics in the UI."""
        lines = []
        raw = metrics.get('raw_count', 0)
        filtered = metrics.get('filtered_count', 0)

        if raw != filtered:
            lines.append(f"Raw detections: {raw} -> Filtered: {filtered}")
            removed_parts = []
            for key, label in [
                ('removed_by_size', 'size'),
                ('removed_by_border', 'border'),
                ('removed_by_confidence', 'confidence'),
                ('removed_by_morphology', 'morphology'),
            ]:
                n = metrics.get(key, 0)
                if n > 0:
                    removed_parts.append(f"{label}: -{n}")
            if removed_parts:
                lines.append("  Removed: " + ", ".join(removed_parts))

        # Size stats with physical units if available
        size_stats = metrics.get('size_stats')
        if size_stats:
            size_line = (
                f"Size: mean={size_stats['mean']:.0f}px  "
                f"median={size_stats['median']:.0f}px  "
                f"std={size_stats['std']:.0f}px"
            )
            if self._pixel_size_um is not None:
                px2 = self._pixel_size_um ** 2
                mean_um2 = size_stats['mean'] * px2
                median_um2 = size_stats['median'] * px2
                size_line += f"\n      ({mean_um2:.0f} / {median_um2:.0f} um\u00b2)"
            lines.append(size_line)

        # Confidence stats
        conf_stats = metrics.get('confidence_stats')
        if conf_stats:
            lines.append(
                f"Confidence: mean={conf_stats['mean']:.2f}  "
                f"min={conf_stats['min']:.2f}"
            )

        # Backend and preprocessing info
        backend = metrics.get('backend', 'stardist')
        preproc = metrics.get('preprocessing', {})
        if preproc:
            active = [k for k, v in preproc.items()
                      if v and k not in ('bg_sigma', 'clahe_clip_limit', 'gaussian_sigma')]
            if active:
                lines.append(f"Backend: {backend} | Preprocess: {', '.join(active)}")
        else:
            lines.append(f"Backend: {backend}")

        # Threshold-specific info
        if backend == 'threshold':
            thresh_val = metrics.get('threshold_value', 0)
            thresh_method = metrics.get('threshold_method', '?')
            if metrics.get('use_hysteresis'):
                thresh_low = metrics.get('threshold_low', 0)
                lines.append(
                    f"Threshold ({thresh_method}): high={thresh_val:.1f}, "
                    f"low={thresh_low:.1f} (hysteresis)"
                )
            else:
                lines.append(f"Threshold ({thresh_method}): {thresh_val:.1f}")
            n_splits = metrics.get('n_watershed_splits', 0)
            if n_splits > 0:
                lines.append(f"Watershed splits: +{n_splits} nuclei")

        self.detect_metrics_label.setText("\n".join(lines))

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

        # Branch on mode
        if self.coloc_mode_combo.currentText() == 'Dual Channel':
            self._run_dual_colocalization()
            return

        self.status_label.setText("Running colocalization analysis...")
        self.coloc_btn.setEnabled(False)

        params = {
            'background_method': self.bg_method_combo.currentText(),
            'background_percentile': self.bg_percentile_spin.value(),
            'threshold_method': self.thresh_method_combo.currentText(),
            'threshold_value': self.thresh_value_spin.value(),
            'dilation_iterations': self.bg_dilation_spin.value(),
            'area_fraction': self.area_fraction_spin.value(),
            'use_local_background': self.bg_local_check.isChecked(),
            'bg_block_size': self.bg_block_size_spin.value(),
            'soma_dilation': self.soma_dilation_spin.value(),
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

        # Get nuclear channel for Manders/Pearson validation metrics
        nuclear_image = self._get_current_slice(self.red_channel)

        # When using area_fraction method, pass signal_image and labels to worker
        # so it can forward them to classify_positive_negative
        if params['threshold_method'] == 'area_fraction':
            self.coloc_worker = ColocalizationWorker(
                signal_image,
                labels,
                params,
                signal_image_for_area=signal_image,
                labels_for_area=labels,
                nuclear_image=nuclear_image,
            )
        else:
            self.coloc_worker = ColocalizationWorker(
                signal_image,
                labels,
                params,
                nuclear_image=nuclear_image,
            )
        self.coloc_worker.progress.connect(self._on_coloc_progress)
        self.coloc_worker.finished.connect(self._on_coloc_finished)
        self.coloc_worker.start()

    def _run_dual_colocalization(self):
        """Run dual-channel colocalization (both red and green as independent signals)."""
        self.status_label.setText("Running dual-channel colocalization...")
        self.coloc_btn.setEnabled(False)

        # Ch1 params (red / mCherry — nuclear)
        params_ch1 = {
            'background_method': self.bg_method_combo.currentText(),
            'background_percentile': self.bg_percentile_spin.value(),
            'threshold_method': self.thresh_method_combo.currentText(),
            'threshold_value': self.thresh_value_spin.value(),
            'dilation_iterations': self.bg_dilation_spin.value(),
            'area_fraction': self.area_fraction_spin.value(),
            'soma_dilation': self.soma_dilation_spin.value(),
        }

        # Ch2 params (green / eYFP — cytoplasmic)
        params_ch2 = {
            'background_method': self.bg_method_combo_ch2.currentText(),
            'background_percentile': self.bg_percentile_spin.value(),
            'threshold_method': 'fold_change',
            'threshold_value': self.thresh_value_spin_ch2.value(),
            'dilation_iterations': self.bg_dilation_spin_ch2.value(),
            'area_fraction': 0.5,
            'soma_dilation': self.soma_dilation_spin_ch2.value(),
        }

        # Log to tracker
        if self.tracker:
            sample_id = self.sample_id_edit.text() or self.current_file.stem
            self.last_run_id = self.tracker.log_colocalization(
                sample_id=sample_id,
                signal_channel='dual',
                background_method=params_ch1['background_method'],
                background_percentile=params_ch1['background_percentile'],
                threshold_method=params_ch1['threshold_method'],
                threshold_value=params_ch1['threshold_value'],
                status='started',
            )

        from .workers import DualColocalizationWorker

        signal_1 = self._get_current_slice(self.red_channel)
        signal_2 = self._get_current_slice(self.green_channel)
        labels = self._get_current_slice(self.nuclei_labels)

        if signal_1 is None or signal_2 is None or labels is None:
            QMessageBox.warning(self, "Error", "Missing channels or labels")
            self.coloc_btn.setEnabled(True)
            return

        self.coloc_worker = DualColocalizationWorker(
            signal_1, signal_2, labels, params_ch1, params_ch2,
        )
        self.coloc_worker.progress.connect(self._on_coloc_progress)
        self.coloc_worker.finished.connect(self._on_dual_coloc_finished)
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
            self._coloc_background_surface = getattr(self.coloc_worker, 'background_surface', None)
            self._coloc_summary = summary

            # Auto-save measurements CSV
            measurements_path = None
            if self.current_file is not None:
                from ..core.config import get_sample_dir, SampleDirs
                stem = self.current_file.stem
                sample_dir = get_sample_dir(stem)
                results_dir = sample_dir / SampleDirs.QUANTIFIED
                results_dir.mkdir(parents=True, exist_ok=True)
                run_tag = self.last_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                measurements_path = results_dir / f"{stem}_{run_tag}_measurements.csv"
                measurements.to_csv(measurements_path, index=False)

            # Update tracker
            if self.tracker and self.last_run_id:
                update_kwargs = dict(
                    status='completed',
                    coloc_positive_cells=summary['positive_cells'],
                    coloc_negative_cells=summary['negative_cells'],
                    coloc_positive_fraction=summary['positive_fraction'],
                    coloc_background_value=summary['background_used'],
                )
                if measurements_path is not None:
                    update_kwargs['measurements_path'] = str(measurements_path)
                self.tracker.update_status(self.last_run_id, **update_kwargs)
                self.session_run_ids.append(self.last_run_id)

            # Refresh run history panel
            self._refresh_run_history()

            # Visualize results - color nuclei by positive/negative
            self._visualize_colocalization(measurements)

            # Update diagnostic plot
            self._update_diagnostic_plot()

            # Update UI
            self.status_label.setText(message)
            result_text = (
                f"Positive: {summary['positive_cells']} ({summary['positive_fraction']*100:.1f}%)\n"
                f"Negative: {summary['negative_cells']}\n"
                f"Background: {summary['background_used']:.1f}\n"
                f"Mean fold change: {summary['mean_fold_change']:.2f}"
            )

            # Append Manders/Pearson validation metrics if available
            coloc_metrics = summary.get('coloc_metrics')
            if coloc_metrics:
                result_text += (
                    f"\n--- Validation Metrics ---\n"
                    f"Pearson r: {coloc_metrics['pearson_r']:.4f}\n"
                    f"Manders M1 (red in green): {coloc_metrics['manders_m1']:.4f}\n"
                    f"Manders M2 (green in red): {coloc_metrics['manders_m2']:.4f}"
                )

            self.coloc_result_label.setText(result_text)

            # Enable quantification
            self.quant_btn.setEnabled(True)

        else:
            self.status_label.setText(f"Error: {message}")
            if self.tracker and self.last_run_id:
                self.tracker.update_status(self.last_run_id, status='failed')

    def _on_dual_coloc_finished(self, success: bool, message: str, measurements, summary, tissue_mask):
        """Handle dual-channel colocalization completion."""
        print(f"[BrainSlice] _on_dual_coloc_finished: success={success}")
        self.coloc_btn.setEnabled(True)

        if success:
            self.cell_measurements = measurements
            self._coloc_summary = summary

            # Auto-save CSV
            measurements_path = None
            if self.current_file is not None:
                from ..core.config import get_sample_dir, SampleDirs
                stem = self.current_file.stem
                sample_dir = get_sample_dir(stem)
                results_dir = sample_dir / SampleDirs.QUANTIFIED
                results_dir.mkdir(parents=True, exist_ok=True)
                tag = self.last_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                measurements_path = results_dir / f"{stem}_{tag}_dual_measurements.csv"
                measurements.to_csv(measurements_path, index=False)

            # Update tracker
            if self.tracker and self.last_run_id:
                update_kwargs = dict(
                    status='completed',
                    coloc_positive_cells=summary.get('n_dual', 0),
                    coloc_negative_cells=summary.get('n_neither', 0),
                )
                if measurements_path is not None:
                    update_kwargs['measurements_path'] = str(measurements_path)
                self.tracker.update_status(self.last_run_id, **update_kwargs)
                self.session_run_ids.append(self.last_run_id)

            # Visualize with 4-category coloring
            self._visualize_dual_colocalization(measurements)

            # Update result label
            ch1 = summary.get('ch1_name', 'red')
            ch2 = summary.get('ch2_name', 'green')
            result_text = (
                f"DUAL-CHANNEL RESULTS\n"
                f"Total nuclei: {summary['total_nuclei']}\n"
                f"Red+ (mCherry): {summary.get(f'n_{ch1}_positive', 0)} "
                f"({summary.get(f'fraction_{ch1}', 0)*100:.1f}%)\n"
                f"Green+ (eYFP): {summary.get(f'n_{ch2}_positive', 0)} "
                f"({summary.get(f'fraction_{ch2}', 0)*100:.1f}%)\n"
                f"Dual+ (both): {summary.get('n_dual', 0)} "
                f"({summary.get('fraction_dual', 0)*100:.1f}%)\n"
                f"Red-only: {summary.get(f'n_{ch1}_only', 0)}\n"
                f"Green-only: {summary.get(f'n_{ch2}_only', 0)}\n"
                f"Neither: {summary.get('n_neither', 0)}"
            )
            self.coloc_result_label.setText(result_text)
            self.status_label.setText(message)

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

    def _visualize_dual_colocalization(self, measurements):
        """Visualize dual-channel results with 4 colored point layers."""
        if measurements is None or len(measurements) == 0:
            return

        if 'centroid_y_base' in measurements.columns:
            y_col, x_col = 'centroid_y_base', 'centroid_x_base'
        else:
            y_col, x_col = 'centroid_y', 'centroid_x'

        # Remove old colocalization layers
        for layer in list(self.viewer.layers):
            if any(tag in layer.name for tag in ['Positive', 'Negative', 'Dual+', 'Red-only', 'Green-only', 'Neither']):
                self.viewer.layers.remove(layer)

        # Category -> color mapping
        categories = {
            'dual':       ('Dual+', '#FFFF00'),
            'red_only':   ('Red-only', '#FF4444'),
            'green_only': ('Green-only', '#44FF44'),
            'neither':    ('Neither', '#888888'),
        }

        for cat, (name_prefix, color) in categories.items():
            subset = measurements[measurements['classification'] == cat]
            if len(subset) > 0:
                coords = subset[[y_col, x_col]].values
                self.viewer.add_points(
                    coords,
                    name=f"{name_prefix} ({len(subset)})",
                    size=10,
                    face_color='transparent',
                    edge_color=color,
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
            create_background_surface_plot,
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
            elif plot_type == 'Background Surface':
                bg_surface = getattr(self, '_coloc_background_surface', None)
                if bg_surface is not None:
                    labels = self._get_current_slice(self.nuclei_labels)
                    fig = create_background_surface_plot(bg_surface, labels)
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

    # ----- Run History Methods -----

    def _refresh_run_history(self):
        """Populate the run history table from the tracker."""
        self.run_history_table.setRowCount(0)

        if not self.tracker:
            return

        # Get sample_id to filter runs
        sample_id = self.sample_id_edit.text()
        if not sample_id and self.current_file:
            sample_id = self.current_file.stem

        if not sample_id:
            return

        runs = self.tracker.search(
            sample_id=sample_id,
            run_type='colocalization',
            status='completed',
            limit=20,
        )

        for run in runs:
            row = self.run_history_table.rowCount()
            self.run_history_table.insertRow(row)

            run_id = run.get('run_id', '')
            created = run.get('created_at', '')[:16]  # trim seconds
            pos = run.get('coloc_positive_cells', '?')
            frac = run.get('coloc_positive_fraction', '')
            if frac:
                try:
                    frac = f"{float(frac)*100:.1f}%"
                except ValueError:
                    pass
            method = run.get('coloc_background_method', '')

            # Mark session runs and best
            display_id = run_id
            if run_id in self.session_run_ids:
                display_id = f"● {run_id}"
            if run.get('marked_best') == 'True':
                display_id = f"★ {run_id}"

            self.run_history_table.setItem(row, 0, QTableWidgetItem(display_id))
            self.run_history_table.setItem(row, 1, QTableWidgetItem(created))
            self.run_history_table.setItem(row, 2, QTableWidgetItem(str(pos)))
            self.run_history_table.setItem(row, 3, QTableWidgetItem(str(frac)))
            self.run_history_table.setItem(row, 4, QTableWidgetItem(method))

        self.run_history_table.resizeColumnsToContents()

    def _load_selected_run(self):
        """Load a historical run's measurements from disk and visualize."""
        selected = self.run_history_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Select a run from the history table.")
            return

        row = selected[0].row()
        display_id = self.run_history_table.item(row, 0).text()
        # Strip markers (● ★)
        run_id = display_id.lstrip('● ★ ').strip()

        if not self.tracker:
            return

        run = self.tracker.get_run(run_id)
        if not run:
            QMessageBox.warning(self, "Error", f"Run {run_id} not found in tracker.")
            return

        measurements_path = run.get('measurements_path', '')
        if not measurements_path or not Path(measurements_path).exists():
            QMessageBox.warning(
                self, "No Data",
                f"Measurements CSV not found for run {run_id}.\n"
                f"Path: {measurements_path or '(not recorded)'}\n\n"
                "Only runs that saved measurements to disk can be reloaded."
            )
            return

        try:
            import pandas as pd
            measurements = pd.read_csv(measurements_path)

            # Validate required columns
            required = {'label', 'centroid_y', 'centroid_x', 'is_positive', 'fold_change'}
            if not required.issubset(set(measurements.columns)):
                QMessageBox.warning(
                    self, "Invalid Data",
                    f"CSV is missing required columns.\nFound: {list(measurements.columns)}"
                )
                return

            # Store and visualize
            self.cell_measurements = measurements
            self._coloc_background = float(run.get('coloc_background_value', 0))
            self._coloc_threshold = float(run.get('coloc_threshold_value', 2.0))
            self._coloc_summary = {
                'total_cells': len(measurements),
                'positive_cells': int(measurements['is_positive'].sum()),
                'negative_cells': int((~measurements['is_positive']).sum()),
                'positive_fraction': float(measurements['is_positive'].mean()),
                'mean_fold_change': float(measurements['fold_change'].mean()),
                'median_fold_change': float(measurements['fold_change'].median()),
                'background_used': self._coloc_background,
            }

            # Visualize
            self._visualize_colocalization(measurements)
            self._update_diagnostic_plot()

            # Update result label
            s = self._coloc_summary
            self.coloc_result_label.setText(
                f"Loaded run {run_id}\n"
                f"Positive: {s['positive_cells']} ({s['positive_fraction']*100:.1f}%)\n"
                f"Negative: {s['negative_cells']}\n"
                f"Background: {s['background_used']:.1f}\n"
                f"Mean fold change: {s['mean_fold_change']:.2f}"
            )

            self.status_label.setText(f"Loaded historical run {run_id}")
            self.quant_btn.setEnabled(True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to load run: {e}")

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
            is_dual_mode = 'classification' in filtered.columns

            if is_dual_mode and total > 0:
                n_dual = int((filtered['classification'] == 'dual').sum())
                n_red = int((filtered['classification'] == 'red_only').sum())
                n_green = int((filtered['classification'] == 'green_only').sum())
                n_neither = int((filtered['classification'] == 'neither').sum())
                results.append({
                    'roi': f"ROI {i+1}",
                    'total': total,
                    'dual': n_dual,
                    'red_only': n_red,
                    'green_only': n_green,
                    'neither': n_neither,
                    'frac_dual': n_dual / total if total > 0 else 0.0,
                    '_dual_mode': True,
                })
            else:
                positive = int(filtered['is_positive'].sum()) if total > 0 else 0
                negative = total - positive
                fraction = positive / total if total > 0 else 0.0
                results.append({
                    'roi': f"ROI {i+1}",
                    'total': total,
                    'positive': positive,
                    'negative': negative,
                    'fraction': fraction,
                    '_dual_mode': False,
                })

        # Add totals row
        is_dual = results and results[0].get('_dual_mode', False)
        if is_dual:
            t_total = sum(r['total'] for r in results)
            t_dual = sum(r.get('dual', 0) for r in results)
            t_red = sum(r.get('red_only', 0) for r in results)
            t_green = sum(r.get('green_only', 0) for r in results)
            t_neither = sum(r.get('neither', 0) for r in results)
            results.append({
                'roi': 'TOTAL', 'total': t_total,
                'dual': t_dual, 'red_only': t_red,
                'green_only': t_green, 'neither': t_neither,
                'frac_dual': t_dual / t_total if t_total > 0 else 0.0,
                '_dual_mode': True,
            })
        else:
            t_total = sum(r['total'] for r in results)
            t_pos = sum(r['positive'] for r in results)
            t_neg = sum(r['negative'] for r in results)
            t_frac = t_pos / t_total if t_total > 0 else 0.0
            results.append({
                'roi': 'TOTAL', 'total': t_total,
                'positive': t_pos, 'negative': t_neg,
                'fraction': t_frac, '_dual_mode': False,
            })

        self._roi_counts_data = results

        # Update table
        if is_dual:
            self.roi_results_table.setColumnCount(7)
            self.roi_results_table.setHorizontalHeaderLabels(
                ["ROI", "Total", "Dual+", "Red+", "Green+", "Neither", "Frac Dual"]
            )
            self.roi_results_table.setRowCount(len(results))
            for row_idx, r in enumerate(results):
                self.roi_results_table.setItem(row_idx, 0, QTableWidgetItem(r['roi']))
                self.roi_results_table.setItem(row_idx, 1, QTableWidgetItem(str(r['total'])))
                self.roi_results_table.setItem(row_idx, 2, QTableWidgetItem(str(r.get('dual', 0))))
                self.roi_results_table.setItem(row_idx, 3, QTableWidgetItem(str(r.get('red_only', 0))))
                self.roi_results_table.setItem(row_idx, 4, QTableWidgetItem(str(r.get('green_only', 0))))
                self.roi_results_table.setItem(row_idx, 5, QTableWidgetItem(str(r.get('neither', 0))))
                self.roi_results_table.setItem(row_idx, 6, QTableWidgetItem(f"{r.get('frac_dual', 0)*100:.1f}%"))
        else:
            self.roi_results_table.setColumnCount(5)
            self.roi_results_table.setHorizontalHeaderLabels(
                ["ROI", "Total Nuclei", "Green+", "Green-", "Fraction"]
            )
            self.roi_results_table.setRowCount(len(results))
            for row_idx, r in enumerate(results):
                self.roi_results_table.setItem(row_idx, 0, QTableWidgetItem(r['roi']))
                self.roi_results_table.setItem(row_idx, 1, QTableWidgetItem(str(r['total'])))
                self.roi_results_table.setItem(row_idx, 2, QTableWidgetItem(str(r.get('positive', 0))))
                self.roi_results_table.setItem(row_idx, 3, QTableWidgetItem(str(r.get('negative', 0))))
                self.roi_results_table.setItem(row_idx, 4, QTableWidgetItem(f"{r.get('fraction', 0)*100:.1f}%"))

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
            from ..core.config import get_sample_dir, SampleDirs
            stem = self.current_file.stem
            sample_dir = get_sample_dir(stem)
            output_dir = sample_dir / SampleDirs.QUANTIFIED
            output_dir.mkdir(parents=True, exist_ok=True)

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
                from ..core.config import get_sample_dir, SampleDirs
                stem = self.current_file.stem
                sample_dir = get_sample_dir(stem)
                output_dir = sample_dir / SampleDirs.QUANTIFIED
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Results exported to:\n{output_dir}"
                )

        else:
            self.status_label.setText(f"Error: {message}")
