"""
main_widget.py - Main BrainSlice napari widget

Provides a tabbed interface for:
1. Load - Load images and select channels
2. Align - Atlas alignment
3. Insets - Add high-resolution region-of-interest overlays
4. Signal - Particle analysis (primary) or nuclei detection + classification (secondary)
5. ROI Count - Draw ROIs and count positive/negative within regions
6. Quantify - Assign to regions and export results
7. Annotate - ND2 annotation and export
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem, QSlider,
    QMessageBox, QProgressBar, QCheckBox, QLineEdit, QScrollArea,
)
from qtpy.QtCore import Qt, QTimer

import napari


class _NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            return self.text() < other.text()


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
        self.channels: list = []  # All channels as list of 2D arrays
        self.channel_names: list = []  # Channel names from metadata
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
        self.particle_worker = None

        # Particle analysis state
        self._pa_labels = None  # Particle labels array
        self._pa_results = None  # Results DataFrame
        self._pa_summary = None  # Summary dict

        # Image navigation
        self._nav_siblings = []
        self._nav_selected_idx = -1

        # Ignore regions
        self._pa_ignore_shapes_layer = None

        # ROI naming
        self._roi_names = []  # List of names matching ROI draw order

        # Image queue state (folder of individual images)
        self._queue_files = []
        self._queue_idx = -1

        # Batch processing state
        self._batch_folder = None
        self._batch_files = []
        self._batch_results = {}
        self._batch_current_idx = -1
        self._batch_settings = None
        self._batch_roi_mode = False

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
        self.tabs.addTab(self._scrollable(self._create_load_tab()), "1. Load")

        # Alignment widget for atlas overlay
        from .alignment_widget import AlignmentWidget
        self.alignment_widget = AlignmentWidget(self)
        self.tabs.addTab(self.alignment_widget, "2. Align")

        # Inset widget (imported here to avoid circular import)
        from .inset_widget import InsetWidget
        self.inset_widget = InsetWidget(self)
        self.tabs.addTab(self.inset_widget, "3. Insets")

        self.tabs.addTab(self._scrollable(self._create_coloc_tab()), "4. Detect && Classify")
        self.tabs.addTab(self._scrollable(self._create_roi_tab()), "5. ROI Count")
        self.tabs.addTab(self._scrollable(self._create_quantify_tab()), "6. Quantify")

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

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        """Wrap a widget in a scroll area so tall tabs are scrollable."""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(0)  # NoFrame
        return scroll

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

        # --- Image Queue (browse folder of individual images) ---
        queue_group = QGroupBox("Image Queue")
        queue_layout = QVBoxLayout()

        queue_info = QLabel(
            "Load a folder of images and step through them one at a time.")
        queue_info.setWordWrap(True)
        queue_info.setStyleSheet("color: #888888; font-size: 11px;")
        queue_layout.addWidget(queue_info)

        queue_btn_row = QHBoxLayout()
        self._queue_browse_btn = QPushButton("Browse Image Folder...")
        self._queue_browse_btn.clicked.connect(self._queue_browse_folder)
        queue_btn_row.addWidget(self._queue_browse_btn)
        queue_layout.addLayout(queue_btn_row)

        self._queue_status_label = QLabel("")
        self._queue_status_label.setWordWrap(True)
        queue_layout.addWidget(self._queue_status_label)

        nav_row = QHBoxLayout()
        self._queue_prev_btn = QPushButton("Previous")
        self._queue_prev_btn.setEnabled(False)
        self._queue_prev_btn.clicked.connect(self._queue_prev)
        nav_row.addWidget(self._queue_prev_btn)

        self._queue_nav_label = QLabel("")
        self._queue_nav_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._queue_nav_label)

        self._queue_next_btn = QPushButton("Next")
        self._queue_next_btn.setEnabled(False)
        self._queue_next_btn.clicked.connect(self._queue_next)
        nav_row.addWidget(self._queue_next_btn)
        queue_layout.addLayout(nav_row)

        queue_group.setLayout(queue_layout)
        layout.addWidget(queue_group)

        # Metadata display
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.metadata_label)

        layout.addStretch()
        return widget

    def _create_detect_tab(self) -> QWidget:
        """Detection has been merged into the Signal tab's Nuclei mode."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(QLabel("Detection has moved to the Signal tab (Nuclei mode)."))
        return widget

    def _on_backend_changed(self, backend_text: str):
        """Toggle visibility of backend-specific parameters."""
        is_threshold = backend_text == 'Threshold'
        is_log = backend_text == 'Threshold+LoG'
        is_stardist = backend_text == 'StarDist'
        is_cellpose = backend_text == 'Cellpose'

        # Threshold params (basic threshold only)
        self.threshold_params_widget.setVisible(is_threshold)

        # Threshold+LoG params
        self.log_params_widget.setVisible(is_log)

        # StarDist/Cellpose need model, preprocessing
        self.model_row_widget.setVisible(not is_threshold and not is_log)
        self.preproc_group.setVisible(not is_threshold and not is_log)
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
        self.thresh_percentile_row.setVisible(method in ('percentile', 'zscore'))
        self.thresh_manual_row.setVisible(method == 'manual')
        # Relabel and re-default the percentile spin for zscore mode
        if method == 'zscore':
            self.thresh_percentile_label.setText("Z-score cutoff:")
            self.thresh_detect_percentile_spin.setRange(1.0, 20.0)
            self.thresh_detect_percentile_spin.setValue(5.0)
            self.thresh_detect_percentile_spin.setToolTip(
                "Number of standard deviations above background.\n"
                "5.0 = good default for sparse fluorescent nuclei.\n"
                "Lower catches dimmer nuclei but adds noise."
            )
        else:
            self.thresh_percentile_label.setText("Percentile:")
            self.thresh_detect_percentile_spin.setRange(80.0, 99.9)
            self.thresh_detect_percentile_spin.setValue(99.0)
            self.thresh_detect_percentile_spin.setToolTip(
                "Intensity percentile to use as threshold.\n"
                "99 = only brightest 1% of pixels."
            )

    def _on_hysteresis_check_changed(self, state):
        """Toggle hysteresis low fraction visibility."""
        self.thresh_hysteresis_row.setVisible(bool(state))

    def _on_split_check_changed(self, state):
        """Toggle watershed split footprint visibility."""
        self.thresh_split_footprint_row.setVisible(bool(state))

    def _on_thresh_method_changed(self, method: str):
        """Toggle visibility of method-specific parameters."""
        self.area_fraction_widget.setVisible(method == 'area_fraction')
        self.sigma_threshold_widget.setVisible(method == 'background_mean')
        self.thresh_value_widget.setVisible(method != 'background_mean')

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
        """Create the Signal tab (particle analysis primary, nuclei classification secondary)."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # ---- Image Navigation (auto-discovers folder siblings) ----
        nav_group = QGroupBox("Image Navigation")
        nav_outer = QVBoxLayout()
        self._img_nav_label = QLabel("(load an image first)")
        self._img_nav_label.setAlignment(Qt.AlignCenter)
        self._img_nav_label.setWordWrap(True)
        nav_outer.addWidget(self._img_nav_label)
        nav_btn_row = QHBoxLayout()
        self._img_prev_btn = QPushButton("< Previous")
        self._img_prev_btn.clicked.connect(self._nav_prev_image)
        nav_btn_row.addWidget(self._img_prev_btn)
        self._img_next_btn = QPushButton("Next >")
        self._img_next_btn.clicked.connect(self._nav_next_image)
        nav_btn_row.addWidget(self._img_next_btn)
        self._img_load_btn = QPushButton("Load")
        self._img_load_btn.setStyleSheet("font-weight: bold;")
        self._img_load_btn.clicked.connect(self._nav_load_selected)
        nav_btn_row.addWidget(self._img_load_btn)
        self._img_save_state_btn = QPushButton("Save State")
        self._img_save_state_btn.setToolTip("Save current analysis state to disk")
        self._img_save_state_btn.clicked.connect(self._manual_save_state)
        nav_btn_row.addWidget(self._img_save_state_btn)
        self._img_export_tiff_btn = QPushButton("Export TIFF")
        self._img_export_tiff_btn.setToolTip(
            "Flatten all visible layers as currently displayed to a TIFF file")
        self._img_export_tiff_btn.clicked.connect(self._export_analyzed_tiff)
        nav_btn_row.addWidget(self._img_export_tiff_btn)
        nav_outer.addLayout(nav_btn_row)
        nav_group.setLayout(nav_outer)
        layout.addWidget(nav_group)

        # ---- Top-level mode selector: Particle (primary) or Nuclei (secondary) ----
        sig_mode_layout = QHBoxLayout()
        sig_mode_layout.addWidget(QLabel("Mode:"))
        self._sig_mode_combo = QComboBox()
        self._sig_mode_combo.addItems(['Particle', 'Nuclei'])
        self._sig_mode_combo.currentTextChanged.connect(self._on_sig_mode_changed)
        sig_mode_layout.addWidget(self._sig_mode_combo)
        layout.addLayout(sig_mode_layout)

        # =========================================================================
        # PARTICLE MODE CONTAINER (visible by default)
        # =========================================================================
        self._sig_particle_container = QWidget()
        pa_layout = QVBoxLayout()
        pa_layout.setSpacing(4)
        pa_layout.setContentsMargins(0, 0, 0, 0)
        self._sig_particle_container.setLayout(pa_layout)

        # Debounce timers
        self._pa_thresh_timer = QTimer()
        self._pa_thresh_timer.setSingleShot(True)
        self._pa_thresh_timer.setInterval(80)
        self._pa_thresh_timer.timeout.connect(self._pa_update_threshold_view)

        self._pa_bg_mask_timer = QTimer()
        self._pa_bg_mask_timer.setSingleShot(True)
        self._pa_bg_mask_timer.setInterval(80)
        self._pa_bg_mask_timer.timeout.connect(self._pa_update_signal_previews)

        # Extra state
        self._pa_binary_view = False
        self._pa_original_visibility = {}
        self._pa_bg_shapes_layer = None

        # --- Channels & Display ---
        chan_group = QGroupBox("Channels & Display")
        chan_layout = QVBoxLayout(chan_group)

        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Detect:"))
        self.pa_det_combo = QComboBox()
        self.pa_det_combo.setToolTip("Channel to binarize (find objects in)")
        self.pa_det_combo.currentIndexChanged.connect(self._pa_on_det_channel_changed)
        ch_row.addWidget(self.pa_det_combo)
        ch_row.addWidget(QLabel("Measure:"))
        self.pa_meas_combo = QComboBox()
        self.pa_meas_combo.setToolTip("Channel to measure intensity in")
        ch_row.addWidget(self.pa_meas_combo)
        chan_layout.addLayout(ch_row)

        # Display channel selector for contrast controls
        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Display:"))
        self.pa_display_combo = QComboBox()
        self.pa_display_combo.setToolTip("Channel to adjust contrast/gamma for")
        self.pa_display_combo.currentIndexChanged.connect(self._pa_on_display_channel_changed)
        disp_row.addWidget(self.pa_display_combo)
        chan_layout.addLayout(disp_row)

        # Contrast min/max/gamma sliders
        self._pa_contrast_min_slider, self._pa_contrast_min_spin = \
            self._pa_make_slider_spinbox(chan_layout, "Min:", 0, 65535, 0,
                                         self._pa_on_contrast_changed)
        self._pa_contrast_max_slider, self._pa_contrast_max_spin = \
            self._pa_make_slider_spinbox(chan_layout, "Max:", 0, 65535, 65535,
                                         self._pa_on_contrast_changed)
        self._pa_gamma_slider, self._pa_gamma_spin = \
            self._pa_make_slider_spinbox(chan_layout, "Gamma:", 0.1, 5.0, 1.0,
                                         self._pa_on_gamma_changed,
                                         is_float=True, step=0.05)

        auto_btn = QPushButton("Auto Contrast")
        auto_btn.clicked.connect(self._pa_auto_contrast)
        chan_layout.addWidget(auto_btn)

        pa_layout.addWidget(chan_group)

        # --- Detection Threshold ---
        thresh_group_pa = QGroupBox("Detection Threshold")
        thresh_layout_pa = QVBoxLayout(thresh_group_pa)

        self._pa_thresh_slider, self._pa_thresh_spin = \
            self._pa_make_slider_spinbox(thresh_layout_pa, "Threshold:", 0, 65535, 500,
                                         self._pa_on_thresh_changed)

        self.pa_mask_info = QLabel("")
        thresh_layout_pa.addWidget(self.pa_mask_info)

        btn_row = QHBoxLayout()
        auto_thresh_btn = QPushButton("Auto (Otsu)")
        auto_thresh_btn.clicked.connect(self._pa_auto_threshold)
        btn_row.addWidget(auto_thresh_btn)
        self.pa_binary_toggle = QCheckBox("Show Binary")
        self.pa_binary_toggle.setToolTip("ImageJ-style: white objects, black background")
        self.pa_binary_toggle.toggled.connect(self._pa_toggle_binary_view)
        btn_row.addWidget(self.pa_binary_toggle)
        thresh_layout_pa.addLayout(btn_row)

        pa_layout.addWidget(thresh_group_pa)

        # --- Background (manual ROI) ---
        bg_group_pa = QGroupBox("Background (draw rectangles on 'Background ROIs' layer)")
        bg_layout_pa = QVBoxLayout(bg_group_pa)

        draw_bg_btn = QPushButton("Draw Background ROIs")
        draw_bg_btn.setToolTip("Activates the Background ROIs layer so you can draw rectangles")
        draw_bg_btn.clicked.connect(self._pa_activate_bg_drawing)
        bg_layout_pa.addWidget(draw_bg_btn)

        self.pa_bg_value_label = QLabel("Background: -- (draw rectangles to measure)")
        bg_layout_pa.addWidget(self.pa_bg_value_label)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("BG value:"))
        self.pa_bg_manual_spin = QDoubleSpinBox()
        self.pa_bg_manual_spin.setRange(0, 999999)
        self.pa_bg_manual_spin.setDecimals(1)
        self.pa_bg_manual_spin.setValue(0)
        self.pa_bg_manual_spin.setToolTip(
            "Background intensity - set from ROIs or type manually.\n"
            "Press Apply to update classification after changing.")
        bg_row.addWidget(self.pa_bg_manual_spin)
        self._pa_apply_bg_btn = QPushButton("Apply")
        self._pa_apply_bg_btn.setMaximumWidth(60)
        self._pa_apply_bg_btn.clicked.connect(self._pa_on_bg_value_changed)
        bg_row.addWidget(self._pa_apply_bg_btn)
        clear_bg_btn = QPushButton("Clear ROIs")
        clear_bg_btn.setMaximumWidth(80)
        clear_bg_btn.clicked.connect(self._pa_clear_bg_rois)
        bg_row.addWidget(clear_bg_btn)
        bg_layout_pa.addLayout(bg_row)

        signal_row = QHBoxLayout()
        self.pa_show_signal_mask = QCheckBox("Show signal fill")
        self.pa_show_signal_mask.setToolTip(
            "Highlight measurement channel pixels above the BG value")
        self.pa_show_signal_mask.toggled.connect(self._pa_on_signal_mask_toggled)
        signal_row.addWidget(self.pa_show_signal_mask)
        self.pa_show_signal_outlines = QCheckBox("Show signal outlines")
        self.pa_show_signal_outlines.setToolTip(
            "Show outlines around regions above BG in measurement channel")
        self.pa_show_signal_outlines.toggled.connect(self._pa_on_signal_outlines_toggled)
        signal_row.addWidget(self.pa_show_signal_outlines)
        bg_layout_pa.addLayout(signal_row)

        pa_layout.addWidget(bg_group_pa)

        # --- Particle Filters ---
        filter_group = QGroupBox("Particle Filters")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Area min:"))
        self.pa_min_area = QSpinBox()
        self.pa_min_area.setMinimum(1)
        self.pa_min_area.setMaximum(1000000)
        self.pa_min_area.setValue(1)
        self.pa_min_area.setToolTip(
            "Minimum particle area in pixels (connected component size).\n"
            "This is the total number of pixels in the detected object,\n"
            "NOT a kernel size. Example: a 5x5 square = 25 pixels.\n"
            "Default 10 filters out noise specks (< ~3px diameter)."
        )
        filter_layout.addWidget(self.pa_min_area)

        filter_layout.addWidget(QLabel("max:"))
        self.pa_max_area = QSpinBox()
        self.pa_max_area.setMinimum(1)
        self.pa_max_area.setMaximum(1000000)
        self.pa_max_area.setValue(50000)
        self.pa_max_area.setToolTip(
            "Maximum particle area in pixels (connected component size).\n"
            "Filters out large merged objects or artifacts.\n"
            "Default 50000 is very permissive."
        )
        filter_layout.addWidget(self.pa_max_area)

        filter_layout.addWidget(QLabel("Circ min:"))
        self.pa_min_circ = QDoubleSpinBox()
        self.pa_min_circ.setRange(0.0, 1.0)
        self.pa_min_circ.setSingleStep(0.05)
        self.pa_min_circ.setValue(0.0)
        filter_layout.addWidget(self.pa_min_circ)

        pa_layout.addWidget(filter_group)

        # --- Positive Classification ---
        pos_group = QGroupBox("Positive Classification")
        pos_layout = QHBoxLayout(pos_group)
        pos_layout.addWidget(QLabel("Min % above BG:"))
        self.pa_pos_pct_spin = QDoubleSpinBox()
        self.pa_pos_pct_spin.setRange(0, 100)
        self.pa_pos_pct_spin.setSingleStep(5)
        self.pa_pos_pct_spin.setValue(50.0)
        self.pa_pos_pct_spin.setSuffix("%")
        self.pa_pos_pct_spin.setToolTip(
            "% of pixels within the particle that must exceed background to count as positive.\n"
            "Edit the value, then press Apply to reclassify.")
        pos_layout.addWidget(self.pa_pos_pct_spin)
        self._pa_apply_pct_btn = QPushButton("Apply")
        self._pa_apply_pct_btn.setMaximumWidth(60)
        self._pa_apply_pct_btn.clicked.connect(self._pa_reclassify_live)
        pos_layout.addWidget(self._pa_apply_pct_btn)
        pa_layout.addWidget(pos_group)

        # --- Ignore Regions ---
        ignore_group = QGroupBox("Ignore Regions")
        ignore_layout = QHBoxLayout(ignore_group)
        self._pa_draw_ignore_btn = QPushButton("Draw Ignore Region")
        self._pa_draw_ignore_btn.setToolTip(
            "Draw polygons around areas to exclude from analysis.\n"
            "Particles inside these regions will be removed from results.")
        self._pa_draw_ignore_btn.clicked.connect(self._pa_activate_ignore_drawing)
        ignore_layout.addWidget(self._pa_draw_ignore_btn)
        self._pa_clear_ignore_btn = QPushButton("Clear")
        self._pa_clear_ignore_btn.setMaximumWidth(60)
        self._pa_clear_ignore_btn.clicked.connect(self._pa_clear_ignore_regions)
        ignore_layout.addWidget(self._pa_clear_ignore_btn)
        pa_layout.addWidget(ignore_group)

        # --- Watershed ---
        ws_group = QGroupBox("Split Touching Particles")
        ws_layout = QHBoxLayout(ws_group)
        self.pa_watershed_check = QCheckBox("Watershed split")
        self.pa_watershed_check.setToolTip(
            "Automatically split touching/merged nuclei using watershed")
        self.pa_watershed_check.setChecked(True)
        ws_layout.addWidget(self.pa_watershed_check)
        pa_layout.addWidget(ws_group)

        # --- Run button ---
        self.pa_run_btn = QPushButton("Run Particle Analysis")
        self.pa_run_btn.setEnabled(False)
        self.pa_run_btn.clicked.connect(self._pa_run_analysis)
        self.pa_run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        pa_layout.addWidget(self.pa_run_btn)

        # --- Results ---
        self.pa_summary_label = QLabel("")
        self.pa_summary_label.setWordWrap(True)
        pa_layout.addWidget(self.pa_summary_label)

        self.pa_results_table = QTableWidget()
        self.pa_results_table.setMaximumHeight(200)
        self.pa_results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pa_results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.pa_results_table.setSortingEnabled(True)
        self.pa_results_table.cellClicked.connect(self._pa_on_table_row_clicked)
        pa_layout.addWidget(self.pa_results_table)

        export_row = QHBoxLayout()
        self.pa_export_btn = QPushButton("Export CSV")
        self.pa_export_btn.setEnabled(False)
        self.pa_export_btn.clicked.connect(self._pa_export_csv)
        export_row.addWidget(self.pa_export_btn)
        self.pa_export_fig_btn = QPushButton("Export Figure")
        self.pa_export_fig_btn.setEnabled(False)
        self.pa_export_fig_btn.clicked.connect(self._pa_export_figure)
        export_row.addWidget(self.pa_export_fig_btn)
        pa_layout.addLayout(export_row)

        self._pa_append_folder_btn = QPushButton("Append to Folder CSV")
        self._pa_append_folder_btn.setEnabled(False)
        self._pa_append_folder_btn.setToolTip(
            "Append this image's particle results to a master CSV\n"
            "in the image folder (no ROIs needed).")
        self._pa_append_folder_btn.clicked.connect(self._pa_append_to_folder_csv)
        pa_layout.addWidget(self._pa_append_folder_btn)

        # --- Batch Settings ---
        batch_settings_row = QHBoxLayout()
        self._pa_save_settings_btn = QPushButton("Save Settings")
        self._pa_save_settings_btn.setToolTip("Save current particle analysis settings to a JSON file")
        self._pa_save_settings_btn.clicked.connect(self._pa_save_settings)
        batch_settings_row.addWidget(self._pa_save_settings_btn)
        self._pa_load_settings_btn = QPushButton("Load Settings")
        self._pa_load_settings_btn.setToolTip("Load particle analysis settings from a JSON file")
        self._pa_load_settings_btn.clicked.connect(self._pa_load_settings)
        batch_settings_row.addWidget(self._pa_load_settings_btn)
        pa_layout.addLayout(batch_settings_row)

        # --- Batch Processing ---
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout()

        batch_info = QLabel("Tune settings on one image, then process a folder.")
        batch_info.setWordWrap(True)
        batch_info.setStyleSheet("color: #888888; font-size: 11px;")
        batch_layout.addWidget(batch_info)

        self._batch_folder_btn = QPushButton("Select Batch Folder")
        self._batch_folder_btn.clicked.connect(self._batch_select_folder)
        batch_layout.addWidget(self._batch_folder_btn)

        self._batch_status_label = QLabel("")
        self._batch_status_label.setWordWrap(True)
        batch_layout.addWidget(self._batch_status_label)

        self._batch_run_btn = QPushButton("Run Batch Analysis")
        self._batch_run_btn.setEnabled(False)
        self._batch_run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._batch_run_btn.clicked.connect(self._batch_run_analysis)
        batch_layout.addWidget(self._batch_run_btn)

        self._batch_progress = QProgressBar()
        self._batch_progress.setVisible(False)
        batch_layout.addWidget(self._batch_progress)

        # ROI annotation navigation (visible during ROI annotation mode)
        self._batch_roi_nav = QWidget()
        roi_nav_layout = QHBoxLayout()
        roi_nav_layout.setContentsMargins(0, 0, 0, 0)
        self._batch_prev_btn = QPushButton("Previous")
        self._batch_prev_btn.clicked.connect(self._batch_prev_image)
        roi_nav_layout.addWidget(self._batch_prev_btn)
        self._batch_nav_label = QLabel("")
        self._batch_nav_label.setAlignment(Qt.AlignCenter)
        roi_nav_layout.addWidget(self._batch_nav_label)
        self._batch_count_next_btn = QPushButton("Count && Next")
        self._batch_count_next_btn.clicked.connect(self._batch_count_and_next)
        roi_nav_layout.addWidget(self._batch_count_next_btn)
        self._batch_skip_btn = QPushButton("Skip")
        self._batch_skip_btn.clicked.connect(self._batch_skip_image)
        roi_nav_layout.addWidget(self._batch_skip_btn)
        self._batch_roi_nav.setLayout(roi_nav_layout)
        self._batch_roi_nav.setVisible(False)
        batch_layout.addWidget(self._batch_roi_nav)

        self._batch_export_btn = QPushButton("Export Batch Results")
        self._batch_export_btn.setEnabled(False)
        self._batch_export_btn.clicked.connect(self._batch_export)
        batch_layout.addWidget(self._batch_export_btn)

        batch_group.setLayout(batch_layout)
        pa_layout.addWidget(batch_group)

        layout.addWidget(self._sig_particle_container)

        # =========================================================================
        # NUCLEI MODE CONTAINER (hidden by default)
        # =========================================================================
        self._sig_nuclei_container = QWidget()
        nuc_layout = QVBoxLayout()
        nuc_layout.setContentsMargins(0, 0, 0, 0)
        self._sig_nuclei_container.setLayout(nuc_layout)
        self._sig_nuclei_container.setVisible(False)

        # ── Detection Backend & Model Selection ──
        backend_group = QGroupBox("Detection Backend")
        backend_layout = QVBoxLayout()

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['Threshold', 'Threshold+LoG', 'StarDist', 'Cellpose'])
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
        nuc_layout.addWidget(backend_group)

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
        nuc_layout.addWidget(preproc_group)

        # ── Detection Parameters ──
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()

        # Threshold parameters (visible by default -- Threshold is default backend)
        self.threshold_params_widget = QWidget()
        thresh_det_layout = QVBoxLayout()
        thresh_det_layout.setContentsMargins(0, 0, 0, 0)

        thresh_method_row = QHBoxLayout()
        thresh_method_row.addWidget(QLabel("Method:"))
        self.thresh_detect_method_combo = QComboBox()
        self.thresh_detect_method_combo.addItems(['zscore', 'otsu', 'percentile', 'manual'])
        self.thresh_detect_method_combo.setToolTip(
            "Zscore: z-score peak detection -- finds bright spots above background\n"
            "  (recommended for sparse fluorescent nuclei)\n"
            "Otsu: automatic threshold (good for bimodal images)\n"
            "Percentile: threshold at Nth percentile intensity\n"
            "Manual: user-specified threshold value"
        )
        self.thresh_detect_method_combo.currentTextChanged.connect(
            self._on_thresh_detect_method_changed
        )
        thresh_method_row.addWidget(self.thresh_detect_method_combo)
        thresh_det_layout.addLayout(thresh_method_row)

        thresh_pct_row = QHBoxLayout()
        self.thresh_percentile_label = QLabel("Z-score cutoff:")
        thresh_pct_row.addWidget(self.thresh_percentile_label)
        self.thresh_detect_percentile_spin = QDoubleSpinBox()
        self.thresh_detect_percentile_spin.setRange(1.0, 20.0)
        self.thresh_detect_percentile_spin.setSingleStep(0.5)
        self.thresh_detect_percentile_spin.setValue(5.0)
        self.thresh_detect_percentile_spin.setToolTip(
            "Number of standard deviations above background.\n"
            "5.0 = good default for sparse fluorescent nuclei.\n"
            "Lower catches dimmer nuclei but adds noise."
        )
        thresh_pct_row.addWidget(self.thresh_detect_percentile_spin)
        self.thresh_percentile_row = QWidget()
        self.thresh_percentile_row.setLayout(thresh_pct_row)
        self.thresh_percentile_row.setVisible(True)  # Visible by default (zscore is default)
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

        # Threshold+LoG parameters (hidden by default)
        self.log_params_widget = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_info = QLabel(
            "Production detection: artifact masking + threshold\n"
            "+ LoG blob union + sparse signal handling."
        )
        log_info.setStyleSheet("color: #888888; font-size: 10px; font-style: italic;")
        log_info.setWordWrap(True)
        log_layout.addWidget(log_info)

        from qtpy.QtWidgets import QFormLayout
        log_form = QFormLayout()

        self.log_pixel_um_spin = QDoubleSpinBox()
        self.log_pixel_um_spin.setRange(0.01, 100.0)
        self.log_pixel_um_spin.setSingleStep(0.01)
        self.log_pixel_um_spin.setDecimals(3)
        self.log_pixel_um_spin.setValue(self._pixel_size_um if self._pixel_size_um else 1.0)
        self.log_pixel_um_spin.setToolTip(
            "Physical pixel size in microns.\n"
            "Auto-populated from ND2 metadata on load."
        )
        log_form.addRow("Pixel size (um):", self.log_pixel_um_spin)

        self.log_min_diam_spin = QDoubleSpinBox()
        self.log_min_diam_spin.setRange(1.0, 100.0)
        self.log_min_diam_spin.setSingleStep(1.0)
        self.log_min_diam_spin.setValue(10.0)
        self.log_min_diam_spin.setToolTip(
            "Minimum expected nucleus diameter in microns.\n"
            "Objects smaller than this are filtered out."
        )
        log_form.addRow("Min diameter (um):", self.log_min_diam_spin)

        self.log_max_diam_spin = QDoubleSpinBox()
        self.log_max_diam_spin.setRange(1.0, 200.0)
        self.log_max_diam_spin.setSingleStep(1.0)
        self.log_max_diam_spin.setValue(25.0)
        self.log_max_diam_spin.setToolTip(
            "Maximum expected nucleus diameter in microns.\n"
            "Objects larger than this are filtered out."
        )
        log_form.addRow("Max diameter (um):", self.log_max_diam_spin)

        self.log_thresh_fraction_spin = QDoubleSpinBox()
        self.log_thresh_fraction_spin.setRange(0.01, 1.0)
        self.log_thresh_fraction_spin.setSingleStep(0.01)
        self.log_thresh_fraction_spin.setDecimals(2)
        self.log_thresh_fraction_spin.setValue(0.20)
        self.log_thresh_fraction_spin.setToolTip(
            "Fraction of Otsu threshold for initial segmentation.\n"
            "0.20 = 20% of Otsu (good for sparse fluorescent nuclei)."
        )
        log_form.addRow("Threshold fraction:", self.log_thresh_fraction_spin)

        self.log_sensitivity_spin = QDoubleSpinBox()
        self.log_sensitivity_spin.setRange(0.0001, 0.1)
        self.log_sensitivity_spin.setSingleStep(0.001)
        self.log_sensitivity_spin.setDecimals(4)
        self.log_sensitivity_spin.setValue(0.005)
        self.log_sensitivity_spin.setToolTip(
            "LoG blob detection sensitivity.\n"
            "Lower = more sensitive (more blobs detected).\n"
            "0.005 = good default for fluorescent nuclei."
        )
        log_form.addRow("LoG sensitivity:", self.log_sensitivity_spin)

        log_layout.addLayout(log_form)
        self.log_params_widget.setLayout(log_layout)
        self.log_params_widget.setVisible(False)
        param_layout.addWidget(self.log_params_widget)

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
        nuc_layout.addWidget(param_group)

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
        nuc_layout.addWidget(filter_group)

        # ── Run Detection Button ──
        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self._run_detection)
        self.detect_btn.setEnabled(False)
        nuc_layout.addWidget(self.detect_btn)

        # ── Detection Results & Metrics ──
        self.detect_result_label = QLabel("")
        nuc_layout.addWidget(self.detect_result_label)

        self.detect_metrics_label = QLabel("")
        self.detect_metrics_label.setWordWrap(True)
        self.detect_metrics_label.setStyleSheet("color: #888888; font-size: 11px;")
        nuc_layout.addWidget(self.detect_metrics_label)

        # ---- Classification (after detection) ----
        nuc_classify_label = QLabel("--- Signal Classification ---")
        nuc_classify_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        nuc_layout.addWidget(nuc_classify_label)

        # Channel Mode selector: Single or Dual channel
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Channel Mode:"))
        self.coloc_mode_combo = QComboBox()
        self.coloc_mode_combo.addItems(['Single Channel', 'Dual Channel'])
        self.coloc_mode_combo.currentTextChanged.connect(self._on_coloc_mode_changed)
        mode_layout.addWidget(self.coloc_mode_combo)
        nuc_layout.addLayout(mode_layout)

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
        self.bg_dilation_spin.setValue(10)
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
        nuc_layout.addWidget(bg_group)

        # Soma measurement
        soma_group = QGroupBox("Soma Measurement")
        soma_layout = QVBoxLayout()

        soma_dil_row = QHBoxLayout()
        soma_dil_row.addWidget(QLabel("Soma dilation (px):"))
        self.soma_dilation_spin = QSpinBox()
        self.soma_dilation_spin.setRange(0, 50)
        self.soma_dilation_spin.setValue(6)
        self.soma_dilation_spin.setToolTip(
            "Dilate each nucleus ROI to include the surrounding soma.\n"
            "Signal (eYFP, etc.) is cytoplasmic, not nuclear -- measure\n"
            "intensity in this dilated region instead of the nucleus alone.\n"
            "6 = validated default for ENCR retrograde tracer.\n"
            "0 = measure only within nucleus (misses cytoplasmic signal)."
        )
        soma_dil_row.addWidget(self.soma_dilation_spin)
        soma_layout.addLayout(soma_dil_row)

        soma_group.setLayout(soma_layout)
        nuc_layout.addWidget(soma_group)

        # Classification threshold
        thresh_group = QGroupBox("Positive/Negative Classification")
        thresh_layout = QVBoxLayout()

        thresh_method_layout = QHBoxLayout()
        thresh_method_layout.addWidget(QLabel("Method:"))
        self.thresh_method_combo = QComboBox()
        self.thresh_method_combo.addItems(['background_mean', 'fold_change', 'area_fraction', 'absolute', 'percentile'])
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
        self.thresh_value_widget = QWidget()
        self.thresh_value_widget.setLayout(thresh_value_layout)
        self.thresh_value_widget.setVisible(False)  # hidden by default since background_mean is default
        thresh_layout.addWidget(self.thresh_value_widget)

        # Sigma threshold (for background_mean method)
        sigma_thresh_layout = QHBoxLayout()
        sigma_thresh_layout.addWidget(QLabel("Sigma threshold:"))
        self.sigma_threshold_spin = QDoubleSpinBox()
        self.sigma_threshold_spin.setRange(0.0, 10.0)
        self.sigma_threshold_spin.setValue(0.0)
        self.sigma_threshold_spin.setSingleStep(0.5)
        self.sigma_threshold_spin.setToolTip(
            "Std devs above background mean for positive classification.\n"
            "0 = background mean IS the threshold (PI's method).\n"
            "Only used with background_mean method."
        )
        sigma_thresh_layout.addWidget(self.sigma_threshold_spin)
        self.sigma_threshold_widget = QWidget()
        self.sigma_threshold_widget.setLayout(sigma_thresh_layout)
        self.sigma_threshold_widget.setVisible(True)  # visible by default since background_mean is default
        thresh_layout.addWidget(self.sigma_threshold_widget)

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
        nuc_layout.addWidget(thresh_group)

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
        self.soma_dilation_spin_ch2.setToolTip("eYFP is cytoplasmic -- dilate generously to capture soma signal.\nRecommended: 15-20px.")
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
        nuc_layout.addWidget(self.ch2_group)

        # Run button
        self.coloc_btn = QPushButton("Run Signal Analysis")
        self.coloc_btn.clicked.connect(self._run_colocalization)
        self.coloc_btn.setEnabled(False)
        nuc_layout.addWidget(self.coloc_btn)

        # Results
        self.coloc_result_label = QLabel("")
        self.coloc_result_label.setWordWrap(True)
        nuc_layout.addWidget(self.coloc_result_label)

        layout.addWidget(self._sig_nuclei_container)

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

    def _on_sig_mode_changed(self, mode):
        """Toggle between Particle and Nuclei mode in the Signal tab."""
        is_particle = (mode == 'Particle')
        self._sig_particle_container.setVisible(is_particle)
        self._sig_nuclei_container.setVisible(not is_particle)

    def _create_roi_tab(self) -> QWidget:
        """Create the ROI Count tab for region-of-interest counting."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Data source info
        self._roi_source_label = QLabel("Data source: (run Detect & Classify first)")
        self._roi_source_label.setWordWrap(True)
        layout.addWidget(self._roi_source_label)

        # ROI Drawing
        roi_draw_group = QGroupBox("ROI Drawing")
        roi_draw_layout = QVBoxLayout()

        roi_btn_row = QHBoxLayout()
        self._roi_add_left_btn = QPushButton("Add Left ROI")
        self._roi_add_left_btn.clicked.connect(lambda: self._add_named_roi("Left"))
        roi_btn_row.addWidget(self._roi_add_left_btn)

        self._roi_add_right_btn = QPushButton("Add Right ROI")
        self._roi_add_right_btn.clicked.connect(lambda: self._add_named_roi("Right"))
        roi_btn_row.addWidget(self._roi_add_right_btn)
        roi_draw_layout.addLayout(roi_btn_row)

        custom_row = QHBoxLayout()
        self._roi_custom_name = QLineEdit()
        self._roi_custom_name.setPlaceholderText("Custom name...")
        custom_row.addWidget(self._roi_custom_name)
        self._roi_add_custom_btn = QPushButton("Add ROI")
        self._roi_add_custom_btn.clicked.connect(
            lambda: self._add_named_roi(
                self._roi_custom_name.text().strip() or None))
        custom_row.addWidget(self._roi_add_custom_btn)
        roi_draw_layout.addLayout(custom_row)

        # Save/Load ROIs
        roi_io_row = QHBoxLayout()
        self._roi_save_btn = QPushButton("Save ROIs")
        self._roi_save_btn.setToolTip("Save ROI polygons + names to a JSON file for reuse")
        self._roi_save_btn.clicked.connect(self._save_rois)
        roi_io_row.addWidget(self._roi_save_btn)
        self._roi_load_btn = QPushButton("Load ROIs")
        self._roi_load_btn.setToolTip(
            "Load saved ROI polygons onto current image.\n"
            "Use napari's transform tools to adjust position/scale.")
        self._roi_load_btn.clicked.connect(self._load_rois)
        roi_io_row.addWidget(self._roi_load_btn)
        self._roi_clear_btn = QPushButton("Clear ROIs")
        self._roi_clear_btn.clicked.connect(self._clear_rois)
        roi_io_row.addWidget(self._roi_clear_btn)
        roi_draw_layout.addLayout(roi_io_row)

        # ROI name list (shows current ROIs and their names)
        self._roi_names_label = QLabel("ROIs: (none)")
        self._roi_names_label.setWordWrap(True)
        self._roi_names_label.setStyleSheet("color: #888888; font-size: 11px;")
        roi_draw_layout.addWidget(self._roi_names_label)

        roi_draw_group.setLayout(roi_draw_layout)
        layout.addWidget(roi_draw_group)

        # Counting
        count_group = QGroupBox("Count")
        count_layout = QVBoxLayout()

        count_btn_layout = QHBoxLayout()
        self.count_roi_btn = QPushButton("Count in ROI(s)")
        self.count_roi_btn.clicked.connect(self._count_all_rois)
        count_btn_layout.addWidget(self.count_roi_btn)

        self.export_roi_btn = QPushButton("Export CSV")
        self.export_roi_btn.clicked.connect(self._export_roi_counts)
        count_btn_layout.addWidget(self.export_roi_btn)

        self._append_folder_btn = QPushButton("Append to Folder CSV")
        self._append_folder_btn.setToolTip(
            "Append this image's ROI results to a master CSV in the image folder.\n"
            "Creates the file if it doesn't exist. Adds a 'sample' column\n"
            "with the image filename so all images accumulate in one file.")
        self._append_folder_btn.clicked.connect(self._append_to_folder_csv)
        count_btn_layout.addWidget(self._append_folder_btn)
        count_layout.addLayout(count_btn_layout)

        self.roi_results_table = QTableWidget()
        self.roi_results_table.setColumnCount(5)
        self.roi_results_table.setHorizontalHeaderLabels(
            ["ROI", "Total", "Positive", "Negative", "Fraction"]
        )
        self.roi_results_table.setMaximumHeight(250)
        count_layout.addWidget(self.roi_results_table)

        count_group.setLayout(count_layout)
        layout.addWidget(count_group)

        # --- Image Navigation + Export (duplicated from Detect & Classify for convenience) ---
        roi_nav_group = QGroupBox("Image Navigation")
        roi_nav_outer = QVBoxLayout()
        self._roi_img_nav_label = QLabel("(load an image first)")
        self._roi_img_nav_label.setAlignment(Qt.AlignCenter)
        self._roi_img_nav_label.setWordWrap(True)
        roi_nav_outer.addWidget(self._roi_img_nav_label)
        roi_nav_btn_row = QHBoxLayout()
        self._roi_img_prev_btn = QPushButton("< Previous")
        self._roi_img_prev_btn.clicked.connect(self._nav_prev_image)
        roi_nav_btn_row.addWidget(self._roi_img_prev_btn)
        self._roi_img_next_btn = QPushButton("Next >")
        self._roi_img_next_btn.clicked.connect(self._nav_next_image)
        roi_nav_btn_row.addWidget(self._roi_img_next_btn)
        self._roi_img_load_btn = QPushButton("Load")
        self._roi_img_load_btn.setStyleSheet("font-weight: bold;")
        self._roi_img_load_btn.clicked.connect(self._nav_load_selected)
        roi_nav_btn_row.addWidget(self._roi_img_load_btn)
        self._roi_save_state_btn = QPushButton("Save State")
        self._roi_save_state_btn.setToolTip("Save current analysis state to disk")
        self._roi_save_state_btn.clicked.connect(self._manual_save_state)
        roi_nav_btn_row.addWidget(self._roi_save_state_btn)
        self._roi_export_tiff_btn = QPushButton("Export TIFF")
        self._roi_export_tiff_btn.clicked.connect(self._export_analyzed_tiff)
        roi_nav_btn_row.addWidget(self._roi_export_tiff_btn)
        roi_nav_outer.addLayout(roi_nav_btn_row)
        roi_nav_group.setLayout(roi_nav_outer)
        layout.addWidget(roi_nav_group)

        layout.addStretch()
        return widget

    # =========================================================================
    # PARTICLE ANALYSIS TAB
    # =========================================================================

    @staticmethod
    def _pa_make_slider_spinbox(parent_layout, label_text, min_val, max_val, default,
                                on_change=None, is_float=False, step=1):
        """Create a linked slider+spinbox pair. Returns (slider, spinbox)."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))

        slider = QSlider(Qt.Horizontal)
        if is_float:
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
        else:
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
        row.addWidget(slider)

        if is_float:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setDecimals(2)
            spinbox.setSingleStep(step)
            spinbox.setValue(default)
            spinbox.setMaximumWidth(70)
        else:
            spinbox = QSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setSingleStep(step)
            spinbox.setValue(default)
            spinbox.setMaximumWidth(70)
        row.addWidget(spinbox)

        def _slider_to_spin(val):
            spinbox.blockSignals(True)
            spinbox.setValue(val / 100.0 if is_float else val)
            spinbox.blockSignals(False)
            if on_change:
                on_change(spinbox.value() if is_float else val)

        def _spin_to_slider(val):
            slider.blockSignals(True)
            slider.setValue(int(val * 100) if is_float else int(val))
            slider.blockSignals(False)
            if on_change:
                on_change(val)

        slider.valueChanged.connect(_slider_to_spin)
        spinbox.valueChanged.connect(_spin_to_slider)

        parent_layout.addLayout(row)
        return slider, spinbox

    def _create_particle_tab(self) -> QWidget:
        """Particle UI has been merged into the Signal tab."""
        # All particle UI is now created in _create_coloc_tab() particle container.
        # This method is kept as a stub for reference.
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(QLabel("Particle analysis has moved to the Signal tab."))
        return widget

    # =========================================================================
    # =========================================================================
    # =========================================================================
    # IMAGE NAVIGATION (from Detect & Classify tab)
    # =========================================================================

    def _clear_analysis_state(self):
        """Clear all analysis state from the previous image."""
        # Remove analysis layers from viewer
        for name in ('Particles', 'Positive/Negative', 'Selected Particle',
                     'Threshold Mask', 'Binary', 'Signal Mask', 'Signal Outlines',
                     'ROIs', 'Background ROIs'):
            for layer in list(self.viewer.layers):
                if layer.name == name:
                    self.viewer.layers.remove(layer)

        # Clear particle state
        self._pa_labels = None
        self._pa_results = None
        self._pa_summary = None
        self._pa_bg_shapes_layer = None

        # Clear ROI state
        self.roi_shapes_layer = None
        self._roi_names = []
        self._roi_counts_data = None
        if hasattr(self, '_roi_detail_data'):
            self._roi_detail_data = None
        if hasattr(self, '_roi_names_label'):
            self._update_roi_names_label()

        # Clear colocalization state
        self.cell_measurements = None
        self.nuclei_labels = None
        self._coloc_background = None
        self._coloc_threshold = None
        self._coloc_summary = None

        # Reset UI elements
        self.pa_summary_label.setText("")
        self.pa_results_table.setRowCount(0)
        self.pa_results_table.setColumnCount(0)
        self.pa_export_btn.setEnabled(False)
        self.pa_export_fig_btn.setEnabled(False)
        self._pa_append_folder_btn.setEnabled(False)
        self.coloc_result_label.setText("")

    def _save_analysis_state(self):
        """Save current analysis state to disk alongside the image file."""
        if self.current_file is None:
            print("[BrainSlice] Save skipped: no current file")
            return
        if self._pa_results is None and self.cell_measurements is None:
            print("[BrainSlice] Save skipped: no results to save")
            return

        import json

        try:
            analysis_dir = self.current_file.parent / f"{self.current_file.stem}_analysis"
            analysis_dir.mkdir(exist_ok=True)

            # Save particle results
            if self._pa_results is not None and len(self._pa_results) > 0:
                self._pa_results.to_csv(analysis_dir / "particles.csv", index=False)

            # Save particle labels
            if self._pa_labels is not None:
                np.savez_compressed(analysis_dir / "labels.npz", labels=self._pa_labels)

            # Save ignore regions
            if (self._pa_ignore_shapes_layer is not None
                    and self._pa_ignore_shapes_layer in self.viewer.layers
                    and len(self._pa_ignore_shapes_layer.data) > 0):
                ignore_data = []
                for shape_data in self._pa_ignore_shapes_layer.data:
                    ignore_data.append(np.array(shape_data).tolist())
                with open(analysis_dir / "ignore_regions.json", 'w') as f:
                    json.dump({'regions': ignore_data}, f, indent=2)

            # Save background ROIs
            if (self._pa_bg_shapes_layer is not None
                    and self._pa_bg_shapes_layer in self.viewer.layers
                    and len(self._pa_bg_shapes_layer.data) > 0):
                bg_rois = []
                for shape_data in self._pa_bg_shapes_layer.data:
                    bg_rois.append(np.array(shape_data).tolist())
                bg_data = {
                    'regions': bg_rois,
                    'bg_value': self.pa_bg_manual_spin.value(),
                }
                with open(analysis_dir / "bg_rois.json", 'w') as f:
                    json.dump(bg_data, f, indent=2)

            # Save ROIs
            if self.roi_shapes_layer is not None and self.roi_shapes_layer in self.viewer.layers:
                if len(self.roi_shapes_layer.data) > 0:
                    rois = []
                    for i, shape_data in enumerate(self.roi_shapes_layer.data):
                        rois.append({
                            'name': self._get_roi_name(i),
                            'vertices': np.array(shape_data).tolist(),
                        })
                    roi_data = {
                        'version': 1,
                        'roi_names': self._roi_names,
                        'rois': rois,
                    }
                    with open(analysis_dir / "rois.json", 'w') as f:
                        json.dump(roi_data, f, indent=2)

            # Save settings + layer visibility
            settings = self._pa_get_settings()
            layer_visibility = {}
            for layer in self.viewer.layers:
                layer_visibility[layer.name] = layer.visible
            settings['layer_visibility'] = layer_visibility
            with open(analysis_dir / "settings.json", 'w') as f:
                json.dump(settings, f, indent=2)

            # Save ROI counts if available
            if hasattr(self, '_roi_detail_data') and self._roi_detail_data is not None:
                self._roi_detail_data.to_csv(analysis_dir / "roi_detail.csv", index=False)

            print(f"[BrainSlice] Analysis saved to {analysis_dir.name}/")

        except Exception as e:
            print(f"[BrainSlice] ERROR saving analysis: {e}")
            import traceback
            traceback.print_exc()

    def _manual_save_state(self):
        """Explicitly save analysis state (triggered by Save State button)."""
        self._save_analysis_state()
        if self.current_file:
            self.status_label.setText(
                f"State saved for {self.current_file.stem}")

    def _restore_analysis_state(self):
        """Restore analysis state from disk if available for current image."""
        if self.current_file is None:
            return False

        analysis_dir = self.current_file.parent / f"{self.current_file.stem}_analysis"
        if not analysis_dir.exists():
            return False

        import pandas as pd
        import json
        restored = []

        try:
            # Restore settings first
            settings_path = analysis_dir / "settings.json"
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = json.load(f)
                self._pa_apply_settings(settings)
                restored.append("settings")

            # Restore particle labels
            labels_path = analysis_dir / "labels.npz"
            if labels_path.exists():
                data = np.load(labels_path)
                self._pa_labels = data['labels']
                restored.append("labels")

                # Add labels layer to viewer
                scale = self._pa_get_scale()
                self.viewer.add_labels(
                    self._pa_labels, name='Particles',
                    opacity=0.5, scale=scale)

            # Restore particle results
            results_path = analysis_dir / "particles.csv"
            if results_path.exists():
                self._pa_results = pd.read_csv(results_path)
                restored.append(f"particles ({len(self._pa_results)})")

                # Populate table
                self._pa_populate_table(self._pa_results)
                self.pa_export_btn.setEnabled(True)
                self.pa_export_fig_btn.setEnabled(True)
                self._pa_append_folder_btn.setEnabled(True)

                # Draw classification overlay
                if self._pa_labels is not None and 'is_positive' in self._pa_results.columns:
                    self._pa_draw_classification_overlay()

                # Register click callback
                particles_layer = self._pa_find_layer('Particles')
                if particles_layer is not None:
                    self._pa_register_click_callback(particles_layer)

            # Restore ROIs
            rois_path = analysis_dir / "rois.json"
            if rois_path.exists():
                with open(rois_path) as f:
                    roi_data = json.load(f)
                self._roi_names = roi_data.get('roi_names', [])
                rois = roi_data.get('rois', [])
                if rois:
                    self._add_roi_layer()
                    self.roi_shapes_layer.data = []
                    for roi in rois:
                        verts = np.array(roi['vertices'])
                        self.roi_shapes_layer.add_polygons([verts])
                    self.roi_shapes_layer.mode = 'pan_zoom'
                    self._update_roi_names_label()
                    restored.append(f"ROIs ({len(rois)})")

            # Restore background ROIs
            bg_rois_path = analysis_dir / "bg_rois.json"
            if bg_rois_path.exists():
                with open(bg_rois_path) as f:
                    bg_data = json.load(f)
                bg_regions = bg_data.get('regions', [])
                bg_value = bg_data.get('bg_value', 0)
                if bg_regions:
                    self._pa_setup_bg_shapes_layer()
                    for region in bg_regions:
                        verts = np.array(region)
                        self._pa_bg_shapes_layer.add_rectangles([verts])
                    self._pa_bg_shapes_layer.mode = 'pan_zoom'
                    if bg_value > 0:
                        self.pa_bg_manual_spin.setValue(bg_value)
                    restored.append(f"BG ROIs ({len(bg_regions)})")

            # Restore ignore regions
            ignore_path = analysis_dir / "ignore_regions.json"
            if ignore_path.exists():
                with open(ignore_path) as f:
                    ignore_data = json.load(f)
                regions = ignore_data.get('regions', [])
                if regions:
                    self._pa_activate_ignore_drawing()
                    self._pa_ignore_shapes_layer.data = []
                    for region in regions:
                        verts = np.array(region)
                        self._pa_ignore_shapes_layer.add_polygons([verts])
                    self._pa_ignore_shapes_layer.mode = 'pan_zoom'
                    restored.append(f"ignore regions ({len(regions)})")

            # Restore layer visibility
            if settings_path.exists():
                with open(settings_path) as f:
                    s = json.load(f)
                vis = s.get('layer_visibility', {})
                if vis:
                    for layer in self.viewer.layers:
                        if layer.name in vis:
                            layer.visible = vis[layer.name]

            if restored:
                self.status_label.setText(
                    f"Restored: {', '.join(restored)}")
                print(f"[BrainSlice] Restored analysis from {analysis_dir.name}/: "
                      f"{', '.join(restored)}")
                return True

        except Exception as e:
            print(f"[BrainSlice] Failed to restore analysis: {e}")
            import traceback
            traceback.print_exc()

        return False

    def _nav_get_siblings(self):
        """Get sorted list of same-type image files in the current file's folder."""
        if not self.current_file or not self.current_file.exists():
            return [], -1
        folder = self.current_file.parent
        ext = self.current_file.suffix.lower()
        siblings = sorted(f for f in folder.iterdir()
                          if f.suffix.lower() == ext and f.is_file())
        try:
            idx = siblings.index(self.current_file)
        except ValueError:
            idx = -1
        return siblings, idx

    def _nav_update_label(self):
        """Update navigation labels on both Detect & Classify and ROI tabs."""
        # Rebuild siblings list if needed
        if not self._nav_siblings and self.current_file:
            self._nav_siblings, _ = self._nav_get_siblings()
            if self._nav_siblings:
                try:
                    self._nav_selected_idx = self._nav_siblings.index(self.current_file)
                except ValueError:
                    self._nav_selected_idx = -1

        if not self._nav_siblings or self._nav_selected_idx < 0:
            self._img_nav_label.setText("(load an image first)")
            if hasattr(self, '_roi_img_nav_label'):
                self._roi_img_nav_label.setText("(load an image first)")
            return

        selected = self._nav_siblings[self._nav_selected_idx]
        total = len(self._nav_siblings)
        # Check if this image has been analyzed before
        analysis_dir = selected.parent / f"{selected.stem}_analysis"
        marker = ""
        if analysis_dir.exists():
            # Get timestamp from settings.json or folder mtime
            settings_file = analysis_dir / "settings.json"
            try:
                if settings_file.exists():
                    mtime = settings_file.stat().st_mtime
                else:
                    mtime = analysis_dir.stat().st_mtime
                from datetime import datetime as _dt
                ts = _dt.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                marker = f" [analyzed {ts}]"
            except Exception:
                marker = " [analyzed]"
        text = f"{self._nav_selected_idx + 1}/{total}: {selected.name}{marker}"
        can_prev = self._nav_selected_idx > 0
        can_next = self._nav_selected_idx < total - 1

        self._img_nav_label.setText(text)
        self._img_prev_btn.setEnabled(can_prev)
        self._img_next_btn.setEnabled(can_next)
        if hasattr(self, '_roi_img_nav_label'):
            self._roi_img_nav_label.setText(text)
            self._roi_img_prev_btn.setEnabled(can_prev)
            self._roi_img_next_btn.setEnabled(can_next)

    def _nav_load_file(self, fpath):
        """Load a file through the normal Load tab pipeline."""
        self._save_analysis_state()
        self._clear_analysis_state()
        self.current_file = fpath
        self.is_folder_load = False
        self.file_label.setText(str(fpath.name))
        self.load_btn.setEnabled(True)
        self._peek_and_configure(fpath)
        self._load_image()
        self._nav_update_label()

    def _nav_prev_image(self):
        """Select previous image (does not load)."""
        if self._nav_selected_idx > 0:
            self._nav_selected_idx -= 1
            self._nav_update_label()

    def _nav_next_image(self):
        """Select next image (does not load)."""
        if self._nav_selected_idx < len(self._nav_siblings) - 1:
            self._nav_selected_idx += 1
            self._nav_update_label()

    def _nav_load_selected(self):
        """Load the currently selected image."""
        if (self._nav_selected_idx < 0 or
                self._nav_selected_idx >= len(self._nav_siblings)):
            return
        fpath = self._nav_siblings[self._nav_selected_idx]
        self._nav_load_file(fpath)

    def _export_analyzed_tiff(self):
        """Export all visible layers composited at full base image resolution as TIFF."""
        if not self.current_file:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        default_name = f"{self.current_file.stem}_Analyzed.tiff"
        default_dir = self.current_file.parent
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Analyzed TIFF",
            str(default_dir / default_name),
            "TIFF Files (*.tiff *.tif)"
        )
        if not path:
            return

        try:
            # Get base image shape
            if self.red_channel is not None:
                base = self._get_current_slice(self.red_channel)
            elif self.channels:
                base = self._get_current_slice(self.channels[0])
            else:
                QMessageBox.warning(self, "Error", "No base image")
                return
            h, w = base.shape[:2]
            scale = self._pa_get_scale()

            # Start with black canvas
            canvas = np.zeros((h, w, 3), dtype=np.float64)

            for layer in self.viewer.layers:
                if not layer.visible:
                    continue
                opacity = layer.opacity

                # Image layers
                if isinstance(layer, napari.layers.Image):
                    data = layer.data
                    if data.ndim == 3:
                        data = data.max(axis=0)
                    if data.shape[:2] != (h, w):
                        continue
                    data = data.astype(np.float64)
                    cmin, cmax = layer.contrast_limits
                    data = np.clip((data - cmin) / max(cmax - cmin, 1), 0, 1)
                    cmap_name = str(getattr(layer.colormap, 'name', layer.colormap))
                    if 'red' in cmap_name or 'magenta' in cmap_name:
                        rgb = (1.0, 0.0, 0.0)
                    elif 'green' in cmap_name:
                        rgb = (0.0, 1.0, 0.0)
                    elif 'blue' in cmap_name or 'cyan' in cmap_name:
                        rgb = (0.0, 0.0, 1.0)
                    elif 'gray' in cmap_name:
                        rgb = (1.0, 1.0, 1.0)
                    else:
                        rgb = (1.0, 1.0, 1.0)
                    for c in range(3):
                        canvas[:, :, c] += data * rgb[c] * opacity

                # Labels layers (Particles, Positive/Negative, etc.)
                elif isinstance(layer, napari.layers.Labels):
                    label_data = layer.data
                    if label_data.shape[:2] != (h, w):
                        continue
                    cmap = layer.colormap
                    if hasattr(cmap, 'color_dict'):
                        for lv, rgba in cmap.color_dict.items():
                            if lv is None or lv == 0:
                                continue
                            mask = label_data == lv
                            if not mask.any():
                                continue
                            for c in range(3):
                                canvas[:, :, c][mask] = (
                                    canvas[:, :, c][mask] * (1 - opacity)
                                    + rgba[c] * opacity)
                    else:
                        # Default label colors: use napari's color for each label
                        unique_labels = np.unique(label_data)
                        for lv in unique_labels:
                            if lv == 0:
                                continue
                            mask = label_data == lv
                            color = layer.get_color(lv)
                            if color is not None:
                                for c in range(3):
                                    canvas[:, :, c][mask] = (
                                        canvas[:, :, c][mask] * (1 - opacity)
                                        + color[c] * opacity)

                # Shapes layers (ROIs, Ignore Regions)
                elif isinstance(layer, napari.layers.Shapes):
                    from skimage.draw import polygon as draw_polygon, polygon_perimeter
                    edge_color = np.array(layer.edge_color[0]
                                          if len(layer.edge_color) > 0
                                          else [1, 1, 1, 1])
                    face_color = np.array(layer.face_color[0]
                                          if len(layer.face_color) > 0
                                          else [0, 0, 0, 0])
                    sy = scale[0] if len(scale) > 0 else 1.0
                    sx = scale[1] if len(scale) > 1 else 1.0

                    for shape_data in layer.data:
                        verts = np.array(shape_data)
                        if sy != 1.0 or sx != 1.0:
                            verts = verts.copy()
                            verts[:, 0] /= sy
                            verts[:, 1] /= sx
                        rows_v = verts[:, 0].astype(int)
                        cols_v = verts[:, 1].astype(int)

                        # Fill
                        if face_color[3] > 0.01:
                            rr, cc = draw_polygon(rows_v, cols_v, shape=(h, w))
                            for c in range(3):
                                canvas[rr, cc, c] = (
                                    canvas[rr, cc, c] * (1 - face_color[3] * opacity)
                                    + face_color[c] * face_color[3] * opacity)

                        # Edge
                        if edge_color[3] > 0.01:
                            rr, cc = polygon_perimeter(rows_v, cols_v,
                                                       shape=(h, w), clip=True)
                            # Thicken edge
                            from scipy import ndimage as _ndi
                            edge_mask = np.zeros((h, w), dtype=bool)
                            edge_mask[rr, cc] = True
                            edge_mask = _ndi.binary_dilation(edge_mask, iterations=1)
                            for c in range(3):
                                canvas[:, :, c][edge_mask] = edge_color[c]

            # Draw scale bar in bottom-right corner
            px_um = self._pixel_size_um
            if px_um and px_um > 0:
                # Pick a nice round scale bar length
                img_width_um = w * px_um
                for bar_um in [50, 100, 200, 500, 1000, 2000, 5000]:
                    if bar_um >= img_width_um * 0.08 and bar_um <= img_width_um * 0.25:
                        break
                bar_px = int(bar_um / px_um)
                bar_h = max(3, h // 200)
                margin = max(10, h // 50)
                x_start = w - margin - bar_px
                y_start = h - margin - bar_h
                # White bar
                canvas[y_start:y_start+bar_h, x_start:x_start+bar_px, :] = 1.0
                # Label text
                if bar_um >= 1000:
                    label = f"{bar_um/1000:.0f} mm"
                else:
                    label = f"{bar_um:.0f} um"

                # Render text with Pillow
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont
                    # Convert canvas to uint8 for Pillow
                    canvas_u8 = np.clip(canvas * 255, 0, 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(canvas_u8)
                    draw = ImageDraw.Draw(pil_img)
                    # Try to get a reasonable font size
                    font_size = max(12, h // 40)
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except Exception:
                        try:
                            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                        except Exception:
                            font = ImageFont.load_default()
                    # Position text centered above the bar
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    text_x = x_start + (bar_px - text_w) // 2
                    text_y = y_start - text_h - 4
                    # Draw text with dark outline for readability
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx != 0 or dy != 0:
                                draw.text((text_x+dx, text_y+dy), label,
                                          fill=(0, 0, 0), font=font)
                    draw.text((text_x, text_y), label,
                              fill=(255, 255, 255), font=font)
                    canvas = np.array(pil_img).astype(np.float64) / 255.0
                except Exception as text_err:
                    print(f"[BrainSlice] Could not render scale bar text: {text_err}")

                print(f"[BrainSlice] Scale bar: {label} ({bar_px} px)")

            # Clamp and convert to uint8
            canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)

            from tifffile import imwrite
            imwrite(path, canvas)
            self.status_label.setText(
                f"Exported {w}x{h} to {Path(path).name}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to export: {e}")

    # =========================================================================
    # IMAGE QUEUE (folder navigation)
    # =========================================================================

    def _queue_browse_folder(self):
        """Select a folder and discover all images for sequential loading."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder")
        if not folder:
            return
        folder = Path(folder)
        files = sorted(
            list(folder.glob('*.nd2')) +
            list(folder.glob('*.tif')) +
            list(folder.glob('*.tiff'))
        )
        if not files:
            QMessageBox.warning(self, "No Images",
                "No ND2 or TIFF files found in the selected folder.")
            return

        self._queue_files = files
        self._queue_idx = 0
        self._queue_status_label.setText(
            f"{len(files)} images in {folder.name}")
        self._queue_prev_btn.setEnabled(False)
        self._queue_next_btn.setEnabled(len(files) > 1)
        self._queue_load_current()

    def _queue_load_current(self):
        """Load the current queue image using the normal Load tab pipeline."""
        if self._queue_idx < 0 or self._queue_idx >= len(self._queue_files):
            return
        fpath = self._queue_files[self._queue_idx]
        total = len(self._queue_files)

        self._queue_nav_label.setText(
            f"{self._queue_idx + 1}/{total}: {fpath.name}")
        self._queue_prev_btn.setEnabled(self._queue_idx > 0)
        self._queue_next_btn.setEnabled(self._queue_idx < total - 1)

        # Save and clear previous analysis before loading new file
        self._save_analysis_state()
        self._clear_analysis_state()

        # Set the file path and trigger normal load
        self.current_file = fpath
        self.is_folder_load = False
        self.file_label.setText(str(fpath))
        self.load_btn.setEnabled(True)

        # Auto-load the image
        self._load_image()

    def _queue_prev(self):
        """Load previous image in queue."""
        if self._queue_idx > 0:
            self._queue_idx -= 1
            self._queue_load_current()

    def _queue_next(self):
        """Load next image in queue."""
        if self._queue_idx < len(self._queue_files) - 1:
            self._queue_idx += 1
            self._queue_load_current()

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def _pa_get_settings(self):
        """Capture current particle + load settings as a dict."""
        settings = {
            'version': 2,
            # Particle analysis params
            'threshold': float(self._pa_thresh_spin.value()),
            'min_area': self.pa_min_area.value(),
            'max_area': self.pa_max_area.value(),
            'min_circularity': self.pa_min_circ.value(),
            'watershed': self.pa_watershed_check.isChecked(),
            'bg_value': self.pa_bg_manual_spin.value(),
            'min_pct_above_bg': self.pa_pos_pct_spin.value(),
            # Load params
            'rotation': self.rotation_combo.currentText(),
            'z_projection': self.z_projection_combo.currentText(),
            'red_channel_idx': self.red_channel_spin.value(),
            'green_channel_idx': self.green_channel_spin.value(),
        }
        # Particle channel indices (relative to loaded channels)
        if self.pa_det_combo.currentIndex() >= 0:
            settings['detect_channel_idx'] = self.pa_det_combo.currentIndex()
        if self.pa_meas_combo.currentIndex() >= 0:
            settings['measure_channel_idx'] = self.pa_meas_combo.currentIndex()
        # Source image info
        if self.current_file:
            settings['source_image'] = str(self.current_file.name)
        # Pixel size
        if self._pixel_size_um:
            settings['pixel_size_um'] = self._pixel_size_um
        return settings

    def _pa_apply_settings(self, settings):
        """Apply saved particle + load settings to the UI controls."""
        # Particle params
        if 'threshold' in settings:
            self._pa_thresh_spin.setValue(int(settings['threshold']))
            self._pa_thresh_slider.setValue(int(settings['threshold']))
        if 'min_area' in settings:
            self.pa_min_area.setValue(settings['min_area'])
        if 'max_area' in settings:
            self.pa_max_area.setValue(settings['max_area'])
        if 'min_circularity' in settings:
            self.pa_min_circ.setValue(settings['min_circularity'])
        if 'watershed' in settings:
            self.pa_watershed_check.setChecked(settings['watershed'])
        if 'bg_value' in settings:
            self.pa_bg_manual_spin.setValue(settings['bg_value'])
        if 'min_pct_above_bg' in settings:
            self.pa_pos_pct_spin.setValue(settings['min_pct_above_bg'])
        # Load params
        if 'rotation' in settings:
            idx = self.rotation_combo.findText(settings['rotation'])
            if idx >= 0:
                self.rotation_combo.setCurrentIndex(idx)
        if 'z_projection' in settings:
            idx = self.z_projection_combo.findText(settings['z_projection'])
            if idx >= 0:
                self.z_projection_combo.setCurrentIndex(idx)
        if 'red_channel_idx' in settings:
            self.red_channel_spin.setValue(settings['red_channel_idx'])
        if 'green_channel_idx' in settings:
            self.green_channel_spin.setValue(settings['green_channel_idx'])

    def _pa_save_settings(self):
        """Save current particle settings to a JSON file."""
        import json
        settings = self._pa_get_settings()
        settings['created'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        default_name = "particle_settings.json"
        if self.current_file:
            default_name = f"{self.current_file.stem}_settings.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Particle Settings", default_name,
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            with open(path, 'w') as f:
                json.dump(settings, f, indent=2)
            self.status_label.setText(f"Settings saved to {Path(path).name}")

    def _pa_load_settings(self):
        """Load particle settings from a JSON file."""
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Particle Settings", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                settings = json.load(f)
            self._pa_apply_settings(settings)
            self._batch_settings = settings
            source = settings.get('source_image', 'unknown')
            self.status_label.setText(
                f"Settings loaded from {Path(path).name} (tuned on: {source})")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load settings: {e}")

    def _batch_select_folder(self):
        """Select a folder of images for batch processing."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Batch Folder")
        if not folder:
            return
        folder = Path(folder)
        # Find all ND2 and TIFF files
        files = sorted(
            list(folder.glob('*.nd2')) +
            list(folder.glob('*.tif')) +
            list(folder.glob('*.tiff'))
        )
        if not files:
            QMessageBox.warning(self, "No Images",
                "No ND2 or TIFF files found in the selected folder.")
            return

        self._batch_folder = folder
        self._batch_files = files
        self._batch_results = {}
        self._batch_current_idx = -1

        # Check metadata consistency
        warnings = self._batch_check_metadata(files)

        status_parts = [f"Found {len(files)} images in {folder.name}"]
        if warnings:
            status_parts.append(f"[!] {len(warnings)} metadata warning(s)")
            # Show warning dialog
            warn_text = "Some images may have different acquisition settings:\n\n"
            for w in warnings[:10]:  # cap at 10
                warn_text += f"  - {w}\n"
            if len(warnings) > 10:
                warn_text += f"  ... and {len(warnings) - 10} more\n"
            warn_text += "\nSettings tuned on one image may not apply to all. Proceed?"
            reply = QMessageBox.warning(
                self, "Metadata Mismatch", warn_text,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.No:
                self._batch_files = []
                self._batch_status_label.setText("Batch cancelled.")
                return

        self._batch_status_label.setText(" | ".join(status_parts))
        self._batch_run_btn.setEnabled(True)

    def _batch_check_metadata(self, files):
        """Compare ND2 metadata across batch files. Returns list of warning strings."""
        warnings = []
        if not files or not files[0].suffix.lower() == '.nd2':
            return warnings

        try:
            from ..core.io import load_image
            # Load reference metadata from first file
            _, ref_meta = load_image(str(files[0]), metadata_only=True)
            if ref_meta is None:
                return warnings
            ref_px = ref_meta.get('pixel_size_um')
            ref_obj = ref_meta.get('objective', '')

            for f in files[1:]:
                try:
                    _, meta = load_image(str(f), metadata_only=True)
                    if meta is None:
                        continue
                    px = meta.get('pixel_size_um')
                    obj = meta.get('objective', '')
                    if ref_px and px and abs(px - ref_px) > 0.01:
                        warnings.append(
                            f"{f.name}: pixel size {px:.3f} vs reference {ref_px:.3f}")
                    if ref_obj and obj and obj != ref_obj:
                        warnings.append(
                            f"{f.name}: objective '{obj}' vs reference '{ref_obj}'")
                except Exception:
                    pass
        except Exception:
            # metadata_only may not be supported -- skip validation
            pass
        return warnings

    def _batch_run_analysis(self):
        """Run particle analysis on all batch images with current settings."""
        if not self._batch_files:
            return

        settings = self._pa_get_settings()
        self._batch_settings = settings

        self._batch_run_btn.setEnabled(False)
        self._batch_progress.setVisible(True)
        self._batch_progress.setMaximum(len(self._batch_files))
        self._batch_progress.setValue(0)
        self._batch_results = {}

        from ..core.particle_analysis import ParticleAnalyzer
        from ..core.io import load_image
        import pandas as pd
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed, find_boundaries
        from skimage.feature import peak_local_max
        from skimage.measure import regionprops

        analyzer = ParticleAnalyzer()
        det_ch = settings.get('detect_channel_idx', 1)
        meas_ch = settings.get('measure_channel_idx', 0)
        threshold = settings['threshold']
        min_area = settings['min_area']
        max_area = settings['max_area']
        min_circ = settings['min_circularity']
        do_watershed = settings['watershed']
        bg_value = settings['bg_value']
        min_pct = settings['min_pct_above_bg']

        # Load params
        rotation_text = settings.get('rotation', 'None')
        z_proj_text = settings.get('z_projection', 'Max Intensity')
        red_ch_idx = settings.get('red_channel_idx', 1)
        green_ch_idx = settings.get('green_channel_idx', 0)

        # Determine rotation k value
        if '90' in rotation_text and 'CCW' in rotation_text:
            rot_k = 1
        elif '90' in rotation_text and 'CW' in rotation_text:
            rot_k = 3
        elif '180' in rotation_text:
            rot_k = 2
        else:
            rot_k = 0

        from qtpy.QtWidgets import QApplication

        errors = []
        for i, fpath in enumerate(self._batch_files):
            self._batch_progress.setValue(i)
            self._batch_status_label.setText(
                f"Processing {i+1}/{len(self._batch_files)}: {fpath.name}")
            QApplication.processEvents()

            try:
                channels, meta = load_image(str(fpath))
                if channels is None:
                    errors.append(f"{fpath.name}: failed to load")
                    continue

                # Apply z-projection to each channel
                processed = []
                for ch in channels:
                    if ch.ndim == 3:
                        if 'Max' in z_proj_text:
                            ch = ch.max(axis=0)
                        elif 'Mean' in z_proj_text:
                            ch = ch.mean(axis=0)
                        else:  # First Z
                            ch = ch[0]
                    # Apply rotation
                    if rot_k > 0:
                        ch = np.rot90(ch, k=rot_k)
                    processed.append(ch)
                channels = processed

                # Map channels like the Load tab does:
                # red_ch_idx and green_ch_idx are the raw channel indices
                # det_ch and meas_ch are relative to [green, red] = [0, 1]
                if max(red_ch_idx, green_ch_idx) >= len(channels):
                    errors.append(f"{fpath.name}: not enough channels "
                                  f"(need idx {max(red_ch_idx, green_ch_idx)}, "
                                  f"have {len(channels)})")
                    continue

                # Build the 2-channel list as the Load tab would:
                # channels[0] = green (signal), channels[1] = red (nuclear)
                mapped_channels = [
                    channels[green_ch_idx],  # index 0 = green/signal
                    channels[red_ch_idx],    # index 1 = red/nuclear
                ]
                # If there are extra channels, append them
                for ci, ch in enumerate(channels):
                    if ci not in (green_ch_idx, red_ch_idx):
                        mapped_channels.append(ch)

                det_img = mapped_channels[det_ch].astype(np.float64)
                meas_img = mapped_channels[meas_ch].astype(np.float64)

                # Store mapped channels for ROI annotation display later
                channels = mapped_channels

                # Binarize
                mask = analyzer.binarize(det_img, threshold)

                # Optional watershed
                if do_watershed:
                    distance = ndi.distance_transform_edt(mask)
                    min_dist = max(3, int(np.sqrt(min_area / np.pi)))
                    coords = peak_local_max(distance, min_distance=min_dist,
                                            labels=mask.astype(int))
                    local_max = np.zeros(mask.shape, dtype=bool)
                    if len(coords) > 0:
                        local_max[tuple(coords.T)] = True
                    markers, _ = ndi.label(local_max)
                    ws_labels = watershed(-distance, markers, mask=mask)
                    boundaries = find_boundaries(ws_labels, mode='inner')
                    mask[boundaries] = False

                labels, particle_props = analyzer.find_particles(
                    mask, min_area=min_area, max_area=max_area,
                    min_circularity=min_circ, max_circularity=1.0,
                )
                n = int(labels.max())

                if n > 0 and bg_value > 0:
                    from skimage.measure import regionprops_table
                    table = regionprops_table(
                        labels, intensity_image=meas_img,
                        properties=['label', 'mean_intensity', 'max_intensity', 'area'],
                    )
                    measurements = pd.DataFrame(table)
                    median_vals, integrated_vals = [], []
                    for lbl in measurements['label'].values:
                        px_vals = meas_img[labels == lbl].astype(np.float64)
                        median_vals.append(float(np.median(px_vals)))
                        integrated_vals.append(float(np.sum(px_vals)))
                    measurements['median_intensity'] = median_vals
                    measurements['integrated_intensity'] = integrated_vals
                    measurements['background'] = bg_value
                    measurements['mean_above_background'] = (
                        measurements['mean_intensity'] - bg_value)
                    measurements['snr'] = (
                        measurements['mean_above_background'] / max(bg_value, 1e-10))

                    results = pd.merge(particle_props, measurements, on='label',
                                       suffixes=('', '_meas'))
                    if 'area_meas' in results.columns:
                        results = results.drop(columns=['area_meas'])

                    # Per-pixel positivity
                    pct_above = []
                    for lbl in results['label'].values:
                        px_vals = meas_img[labels == int(lbl)]
                        n_above = int((px_vals > bg_value).sum())
                        pct_above.append(100.0 * n_above / max(len(px_vals), 1))
                    results['pct_above_bg'] = pct_above
                    results['is_positive'] = results['pct_above_bg'] >= min_pct
                    results['fold_change'] = (
                        results['mean_intensity'] / max(bg_value, 1e-10))

                    # Centroids
                    rp = {r.label: r for r in regionprops(labels)}
                    results['centroid_y'] = results['label'].map(
                        lambda l: rp[l].centroid[0] if l in rp else np.nan)
                    results['centroid_x'] = results['label'].map(
                        lambda l: rp[l].centroid[1] if l in rp else np.nan)
                else:
                    results = particle_props.copy()
                    if 'pct_above_bg' not in results.columns:
                        results['pct_above_bg'] = 0.0
                        results['is_positive'] = False

                self._batch_results[fpath.name] = {
                    'path': fpath,
                    'particles': results,
                    'labels': labels,
                    'channels': channels,
                    'n_particles': n,
                    'roi_done': False,
                    'summary': None,
                    'detail': None,
                }
            except Exception as e:
                errors.append(f"{fpath.name}: {e}")

        self._batch_progress.setValue(len(self._batch_files))

        n_ok = len(self._batch_results)
        status = f"Batch complete: {n_ok}/{len(self._batch_files)} images processed"
        if errors:
            status += f" ({len(errors)} errors)"
            print(f"[BrainSlice] Batch errors:")
            for err in errors:
                print(f"  {err}")
        self._batch_status_label.setText(status)
        self._batch_run_btn.setEnabled(True)
        self._batch_progress.setVisible(False)

        # Enter ROI annotation mode if we have results
        if self._batch_results:
            self._batch_enter_roi_mode()

    def _batch_enter_roi_mode(self):
        """Enter sequential ROI annotation mode."""
        self._batch_roi_mode = True
        self._batch_roi_nav.setVisible(True)
        self._batch_current_idx = 0
        self._batch_load_current_image()

    def _batch_load_current_image(self):
        """Load the current batch image into napari for ROI drawing."""
        if self._batch_current_idx < 0:
            return
        names = list(self._batch_results.keys())
        if self._batch_current_idx >= len(names):
            # All done
            self._batch_finish_roi_mode()
            return

        name = names[self._batch_current_idx]
        data = self._batch_results[name]
        total = len(names)
        done = sum(1 for d in self._batch_results.values() if d['roi_done'])

        self._batch_nav_label.setText(
            f"{self._batch_current_idx + 1}/{total} ({done} done) -- {name}")

        # Clear viewer layers
        self.viewer.layers.clear()

        # Load channels
        channels = data['channels']
        scale = [self._pixel_size_um, self._pixel_size_um] if self._pixel_size_um else [1, 1]

        for ci, ch in enumerate(channels):
            ch_data = ch
            if ch_data.ndim == 3:
                ch_data = ch_data.max(axis=0)
            cmap = 'green' if ci == 0 else 'red' if ci == 1 else 'gray'
            self.viewer.add_image(
                ch_data, name=f"Ch {ci}", colormap=cmap,
                blending='additive', scale=scale,
            )

        # Show particles
        labels = data['labels']
        if labels.max() > 0:
            self.viewer.add_labels(
                labels, name='Particles', opacity=0.5, scale=scale)

        # Show positive/negative overlay if available
        results = data['particles']
        if 'is_positive' in results.columns and labels.max() > 0:
            self._pa_labels = labels
            self._pa_results = results
            self._pa_draw_classification_overlay()

        # Add ROI shapes layer
        self.roi_shapes_layer = None
        self._add_roi_layer()

        self.status_label.setText(
            f"Draw ROIs on {name}, then click 'Count & Next'")

    def _batch_count_and_next(self):
        """Count ROIs for current image and advance to next."""
        names = list(self._batch_results.keys())
        if self._batch_current_idx < 0 or self._batch_current_idx >= len(names):
            return

        name = names[self._batch_current_idx]
        data = self._batch_results[name]

        # Use the current particle results for this image
        measurements = data['particles']

        if self.roi_shapes_layer is not None and len(self.roi_shapes_layer.data) > 0:
            from ..core.colocalization import filter_measurements_by_roi
            import pandas as pd

            # Get image shape
            ch0 = data['channels'][0]
            if ch0.ndim == 3:
                ch0 = ch0.max(axis=0)
            image_shape = ch0.shape[:2]

            scale = [self._pixel_size_um, self._pixel_size_um] if self._pixel_size_um else [1, 1]
            scale_y = scale[0]
            scale_x = scale[1]

            roi_assignment = pd.Series('Outside', index=measurements.index)
            summary_results = []

            for i, shape_data in enumerate(self.roi_shapes_layer.data):
                vertices = np.array(shape_data)
                if scale_y != 1.0 or scale_x != 1.0:
                    vertices = vertices.copy()
                    vertices[:, 0] /= scale_y
                    vertices[:, 1] /= scale_x
                filtered = filter_measurements_by_roi(
                    measurements, vertices, image_shape)

                roi_name = self._get_roi_name(i)
                for idx in filtered.index:
                    if roi_assignment[idx] == 'Outside':
                        roi_assignment[idx] = roi_name

                total = len(filtered)
                positive = int(filtered['is_positive'].sum()) if total > 0 else 0
                negative = total - positive
                fraction = positive / total if total > 0 else 0.0
                summary_results.append({
                    'sample': name,
                    'roi': roi_name,
                    'total': total,
                    'positive': positive,
                    'negative': negative,
                    'fraction': fraction,
                })

            # Totals
            t_total = sum(r['total'] for r in summary_results)
            t_pos = sum(r['positive'] for r in summary_results)
            t_neg = sum(r['negative'] for r in summary_results)
            summary_results.append({
                'sample': name,
                'roi': 'TOTAL',
                'total': t_total,
                'positive': t_pos,
                'negative': t_neg,
                'fraction': t_pos / t_total if t_total > 0 else 0.0,
            })

            detail = measurements.copy()
            detail.insert(0, 'roi', roi_assignment)
            detail.insert(0, 'sample', name)

            data['summary'] = summary_results
            data['detail'] = detail
            data['roi_done'] = True
        else:
            # No ROIs drawn -- still mark as done with no ROI data
            data['roi_done'] = True
            data['summary'] = []
            data['detail'] = None

        # Advance
        self._batch_current_idx += 1
        self._batch_load_current_image()

    def _batch_prev_image(self):
        """Go back to previous image in batch."""
        if self._batch_current_idx > 0:
            self._batch_current_idx -= 1
            self._batch_load_current_image()

    def _batch_skip_image(self):
        """Skip current image without counting."""
        self._batch_current_idx += 1
        self._batch_load_current_image()

    def _batch_finish_roi_mode(self):
        """All images processed -- exit ROI mode."""
        self._batch_roi_mode = False
        self._batch_roi_nav.setVisible(False)
        self._batch_export_btn.setEnabled(True)

        done = sum(1 for d in self._batch_results.values() if d['roi_done'])
        total = len(self._batch_results)
        self._batch_status_label.setText(
            f"Batch ROI annotation complete: {done}/{total} images. Ready to export.")
        self.status_label.setText("Click 'Export Batch Results' to save.")

    def _batch_export(self):
        """Export accumulated batch results to CSV files."""
        if not self._batch_results:
            return

        import pandas as pd

        default_name = "batch_results"
        if self._batch_folder:
            default_name = f"{self._batch_folder.name}_batch"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch Results",
            f"{default_name}_summary.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        # Build master summary
        all_summary = []
        all_detail = []
        for name, data in self._batch_results.items():
            if data['summary']:
                all_summary.extend(data['summary'])
            if data['detail'] is not None:
                all_detail.append(data['detail'])

        # Write summary
        if all_summary:
            summary_df = pd.DataFrame(all_summary)
            summary_df.to_csv(path, index=False)

        # Write detail
        if all_detail:
            detail_df = pd.concat(all_detail, ignore_index=True)
            detail_path = path.replace('_summary.csv', '_detail.csv')
            if detail_path == path:
                detail_path = path.replace('.csv', '_detail.csv')
            detail_df.to_csv(detail_path, index=False)
            self.status_label.setText(
                f"Exported {Path(path).name} + {Path(detail_path).name}")
        else:
            self.status_label.setText(f"Exported {Path(path).name}")

        # Also save the settings used
        if self._batch_settings:
            import json
            settings_path = path.replace('_summary.csv', '_settings.json')
            if settings_path == path:
                settings_path = path.replace('.csv', '_settings.json')
            with open(settings_path, 'w') as f:
                json.dump(self._batch_settings, f, indent=2)

    # -- helpers ---------------------------------------------------------------

    def _pa_find_layer(self, name):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _pa_remove_layer(self, name):
        for layer in list(self.viewer.layers):
            if layer.name == name:
                self.viewer.layers.remove(layer)

    def _pa_get_scale(self):
        """Return [pixel_um, pixel_um] scale if available, else [1, 1]."""
        px = self._pixel_size_um
        return [px, px] if px and px > 0 else [1, 1]

    def _pa_get_bg(self):
        val = self.pa_bg_manual_spin.value()
        return val if val > 0 else 0.0

    def _pa_scroll_to_results(self):
        """Scroll the particle tab's scroll area to show the results table."""
        from qtpy.QtWidgets import QScrollArea
        widget = self.pa_results_table.parent()
        while widget is not None:
            if isinstance(widget, QScrollArea):
                widget.ensureWidgetVisible(self.pa_results_table)
                return
            widget = widget.parent()

    # -- channel combos --------------------------------------------------------

    def _update_particle_channel_combos(self):
        """Update particle analysis channel combos when image is loaded."""
        if not hasattr(self, 'pa_det_combo'):
            return

        self.pa_det_combo.blockSignals(True)
        self.pa_meas_combo.blockSignals(True)
        self.pa_display_combo.blockSignals(True)
        self.pa_det_combo.clear()
        self.pa_meas_combo.clear()
        self.pa_display_combo.clear()

        for i, name in enumerate(self.channel_names):
            self.pa_det_combo.addItem(f"{i}: {name}")
            self.pa_meas_combo.addItem(f"{i}: {name}")
            self.pa_display_combo.addItem(f"{i}: {name}")

        if len(self.channel_names) >= 2:
            self.pa_det_combo.setCurrentIndex(0)   # red = detect
            self.pa_meas_combo.setCurrentIndex(1)  # green = measure
            self.pa_display_combo.setCurrentIndex(0)  # default to red

        self.pa_det_combo.blockSignals(False)
        self.pa_meas_combo.blockSignals(False)
        self.pa_display_combo.blockSignals(False)

        # Update threshold range based on bit depth from metadata
        if self.channels:
            det_idx = max(0, self.pa_det_combo.currentIndex())
            if det_idx < len(self.channels):
                img = self.channels[det_idx]
                # Use significant bit depth from metadata if available
                bits = None
                if self.metadata and isinstance(self.metadata, dict):
                    bits = self.metadata.get('bits_per_component')
                if bits:
                    img_max = (2 ** bits) - 1
                elif img.dtype == np.uint8:
                    img_max = 255
                elif img.dtype == np.uint16:
                    img_max = 65535
                else:
                    img_max = max(int(np.max(img)) + 1, 1)
                self._pa_thresh_slider.setMaximum(img_max)
                self._pa_thresh_spin.setMaximum(img_max)
                self._pa_contrast_min_slider.setMaximum(img_max)
                self._pa_contrast_min_spin.setMaximum(img_max)
                self._pa_contrast_max_slider.setMaximum(img_max)
                self._pa_contrast_max_spin.setMaximum(img_max)
                self._pa_thresh_slider.setValue(int(img_max * 0.3))

        # Enable run button
        self.pa_run_btn.setEnabled(len(self.channels) >= 2)

        # Setup background ROI shapes layer
        self._pa_setup_bg_shapes_layer()

    # -- contrast / gamma ------------------------------------------------------

    def _pa_on_det_channel_changed(self, idx):
        """Update threshold slider range when detection channel changes."""
        if 0 <= idx < len(self.channels):
            det_img = self.channels[idx]
            img_max = int(det_img.max())
            for w in (self._pa_thresh_slider,):
                w.setMaximum(img_max)
            for w in (self._pa_thresh_spin,):
                w.setMaximum(img_max)

    def _pa_on_display_channel_changed(self, idx):
        """Update contrast slider range and sync when display channel changes."""
        if 0 <= idx < len(self.channels):
            img = self.channels[idx]
            img_max = int(img.max())
            for w in (self._pa_contrast_min_slider, self._pa_contrast_max_slider):
                w.setMaximum(img_max)
            for w in (self._pa_contrast_min_spin, self._pa_contrast_max_spin):
                w.setMaximum(img_max)
            self._pa_sync_contrast_to_layer()

    def _pa_sync_contrast_to_layer(self):
        disp_idx = self.pa_display_combo.currentIndex()
        if disp_idx < 0 or disp_idx >= len(self.channel_names):
            return
        name = self.channel_names[disp_idx]
        for layer in self.viewer.layers:
            if layer.name == name:
                cmin, cmax = layer.contrast_limits
                for w in (self._pa_contrast_min_slider, self._pa_contrast_min_spin,
                          self._pa_contrast_max_slider, self._pa_contrast_max_spin):
                    w.blockSignals(True)
                self._pa_contrast_min_slider.setValue(int(cmin))
                self._pa_contrast_min_spin.setValue(int(cmin))
                self._pa_contrast_max_slider.setValue(int(cmax))
                self._pa_contrast_max_spin.setValue(int(cmax))
                for w in (self._pa_contrast_min_slider, self._pa_contrast_min_spin,
                          self._pa_contrast_max_slider, self._pa_contrast_max_spin):
                    w.blockSignals(False)
                break

    def _pa_on_contrast_changed(self, _val=None):
        disp_idx = self.pa_display_combo.currentIndex()
        if disp_idx < 0 or disp_idx >= len(self.channel_names):
            return
        cmin = self._pa_contrast_min_spin.value()
        cmax = max(self._pa_contrast_max_spin.value(), cmin + 1)
        for layer in self.viewer.layers:
            if layer.name == self.channel_names[disp_idx]:
                layer.contrast_limits = (cmin, cmax)
                break

    def _pa_on_gamma_changed(self, val):
        disp_idx = self.pa_display_combo.currentIndex()
        if disp_idx < 0 or disp_idx >= len(self.channel_names):
            return
        for layer in self.viewer.layers:
            if layer.name == self.channel_names[disp_idx]:
                layer.gamma = val
                break

    def _pa_auto_contrast(self):
        disp_idx = self.pa_display_combo.currentIndex()
        if disp_idx < 0 or disp_idx >= len(self.channels):
            return
        ch = self.channels[disp_idx]
        for layer in self.viewer.layers:
            if layer.name == self.channel_names[disp_idx]:
                layer.contrast_limits = (float(ch.min()), float(ch.max()))
                layer.gamma = 1.0
                break
        self._pa_sync_contrast_to_layer()
        self._pa_gamma_slider.blockSignals(True)
        self._pa_gamma_spin.blockSignals(True)
        self._pa_gamma_slider.setValue(100)
        self._pa_gamma_spin.setValue(1.0)
        self._pa_gamma_slider.blockSignals(False)
        self._pa_gamma_spin.blockSignals(False)

    # -- threshold / binary view -----------------------------------------------

    def _pa_on_thresh_changed(self, _value=None):
        self._pa_thresh_timer.start()

    def _pa_update_threshold_view(self):
        if not self.channels:
            return
        det_idx = self.pa_det_combo.currentIndex()
        if det_idx < 0 or det_idx >= len(self.channels):
            return

        det_img = self._get_current_slice(self.channels[det_idx])
        if det_img is None:
            return
        threshold = self._pa_thresh_spin.value()
        mask = det_img.astype(np.float64) > threshold

        px_count = int(mask.sum())
        pct = px_count / mask.size * 100
        self.pa_mask_info.setText(f"Mask: {px_count} px ({pct:.1f}%)")

        scale = self._pa_get_scale()
        if self._pa_binary_view:
            binary_img = mask.astype(np.float32) * 255
            existing = self._pa_find_layer('Binary')
            if existing is not None:
                existing.data = binary_img
            else:
                self.viewer.add_image(
                    binary_img, name='Binary',
                    colormap='gray', blending='opaque',
                    contrast_limits=(0, 255), scale=scale,
                )
            self._pa_remove_layer('Threshold Mask')
        else:
            mask_uint8 = mask.astype(np.uint8)
            existing = self._pa_find_layer('Threshold Mask')
            if existing is not None:
                existing.data = mask_uint8
            else:
                from napari.utils.colormaps import DirectLabelColormap
                white_cmap = DirectLabelColormap(
                    color_dict={1: (1.0, 1.0, 1.0, 1.0), None: (0, 0, 0, 0)}
                )
                self.viewer.add_labels(
                    mask_uint8, name='Threshold Mask',
                    opacity=0.25, colormap=white_cmap, scale=scale,
                )
            self._pa_remove_layer('Binary')

    def _pa_toggle_binary_view(self, checked):
        self._pa_binary_view = checked
        for name in self.channel_names:
            for layer in self.viewer.layers:
                if layer.name == name:
                    if checked:
                        self._pa_original_visibility[name] = layer.visible
                        layer.visible = False
                    else:
                        layer.visible = self._pa_original_visibility.get(name, True)
        self._pa_update_threshold_view()

    def _pa_auto_threshold(self):
        if not self.channels:
            return
        det_idx = self.pa_det_combo.currentIndex()
        if det_idx < 0 or det_idx >= len(self.channels):
            return
        det_img = self._get_current_slice(self.channels[det_idx])
        if det_img is None:
            return
        from skimage.filters import threshold_otsu
        val = int(threshold_otsu(det_img))
        self._pa_thresh_slider.blockSignals(True)
        self._pa_thresh_spin.blockSignals(True)
        self._pa_thresh_slider.setValue(val)
        self._pa_thresh_spin.setValue(val)
        self._pa_thresh_slider.blockSignals(False)
        self._pa_thresh_spin.blockSignals(False)
        self._pa_thresh_timer.start()

    # -- background ROI --------------------------------------------------------

    def _pa_setup_bg_shapes_layer(self):
        """Create (or recreate) the Background ROIs shapes layer."""
        self._pa_remove_layer('Background ROIs')
        scale = self._pa_get_scale()
        self._pa_bg_shapes_layer = self.viewer.add_shapes(
            name="Background ROIs",
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
            scale=scale,
        )
        self._pa_bg_shapes_layer.mode = 'add_rectangle'
        self._pa_bg_shapes_layer.events.data.connect(self._pa_on_bg_rois_changed)

    def _pa_activate_bg_drawing(self):
        # Recreate if the layer was removed from the viewer
        if (self._pa_bg_shapes_layer is None
                or self._pa_bg_shapes_layer not in self.viewer.layers):
            self._pa_setup_bg_shapes_layer()
        if self._pa_bg_shapes_layer is not None:
            self.viewer.layers.selection.active = self._pa_bg_shapes_layer
            self._pa_bg_shapes_layer.mode = 'add_rectangle'

    def _pa_on_bg_rois_changed(self, event=None):
        if self._pa_bg_shapes_layer is None or len(self._pa_bg_shapes_layer.data) == 0:
            self.pa_bg_value_label.setText(
                "Background: -- (draw rectangles to measure)")
            return

        meas_idx = self.pa_meas_combo.currentIndex()
        if meas_idx < 0 or meas_idx >= len(self.channels):
            return

        meas_img = self._get_current_slice(self.channels[meas_idx])
        if meas_img is None:
            return
        meas_img = meas_img.astype(np.float64)
        roi_means = []
        all_px_min = float('inf')
        all_px_max = float('-inf')

        px = self._pixel_size_um if (self._pixel_size_um and self._pixel_size_um > 0) else None
        for shape_data in self._pa_bg_shapes_layer.data:
            rows = shape_data[:, 0]
            cols = shape_data[:, 1]
            # Convert from physical coords back to pixels if scaled
            if px:
                rows = rows / px
                cols = cols / px
            r_min = int(max(0, rows.min()))
            r_max = int(min(meas_img.shape[0], rows.max()))
            c_min = int(max(0, cols.min()))
            c_max = int(min(meas_img.shape[1], cols.max()))
            if r_max > r_min and c_max > c_min:
                roi_patch = meas_img[r_min:r_max, c_min:c_max]
                roi_means.append(float(roi_patch.mean()))
                all_px_min = min(all_px_min, float(roi_patch.min()))
                all_px_max = max(all_px_max, float(roi_patch.max()))

        if roi_means:
            avg = np.mean(roi_means)
            self.pa_bg_value_label.setText(
                f"Background: {avg:.1f}  ({len(roi_means)} ROI(s), "
                f"pixel range: {all_px_min:.0f}-{all_px_max:.0f}, "
                f"mean range: {min(roi_means):.0f}-{max(roi_means):.0f})"
            )
            self.pa_bg_manual_spin.blockSignals(True)
            self.pa_bg_manual_spin.setValue(avg)
            self.pa_bg_manual_spin.blockSignals(False)
            if self.pa_show_signal_mask.isChecked() or self.pa_show_signal_outlines.isChecked():
                self._pa_bg_mask_timer.start()

    def _pa_on_bg_value_changed(self, _val=None):
        if self.pa_show_signal_mask.isChecked() or self.pa_show_signal_outlines.isChecked():
            self._pa_bg_mask_timer.start()
        # Live reclassify if we have results
        self._pa_reclassify_live()

    def _pa_update_signal_previews(self):
        if self.pa_show_signal_mask.isChecked():
            self._pa_update_bg_mask_preview()
        if self.pa_show_signal_outlines.isChecked():
            self._pa_update_signal_outlines()

    def _pa_on_signal_mask_toggled(self, checked):
        if checked:
            self._pa_update_bg_mask_preview()
        else:
            self._pa_remove_layer('Signal Mask')

    def _pa_on_signal_outlines_toggled(self, checked):
        if checked:
            self._pa_update_signal_outlines()
        else:
            self._pa_remove_layer('Signal Outlines')

    def _pa_update_bg_mask_preview(self):
        if not self.channels:
            return
        meas_idx = self.pa_meas_combo.currentIndex()
        if meas_idx < 0 or meas_idx >= len(self.channels):
            return
        bg_val = self.pa_bg_manual_spin.value()
        if bg_val <= 0:
            return
        meas_img = self._get_current_slice(self.channels[meas_idx])
        if meas_img is None:
            return
        meas_f64 = meas_img.astype(np.float64)
        signal_mask = (meas_f64 > bg_val).astype(np.uint8)
        n_above = int(signal_mask.sum())
        print(f"[Signal Mask] meas_ch={meas_idx}, bg={bg_val:.1f}, "
              f"img range={float(meas_f64.min()):.0f}-{float(meas_f64.max()):.0f}, "
              f"pixels above bg={n_above}")
        scale = self._pa_get_scale()
        existing = self._pa_find_layer('Signal Mask')
        if existing is not None:
            existing.data = signal_mask
        else:
            from napari.utils.colormaps import DirectLabelColormap
            green_cmap = DirectLabelColormap(
                color_dict={1: (0.0, 1.0, 0.0, 1.0), None: (0, 0, 0, 0)}
            )
            self.viewer.add_labels(
                signal_mask, name='Signal Mask',
                opacity=0.3, colormap=green_cmap, scale=scale,
            )

    def _pa_update_signal_outlines(self):
        if not self.channels:
            return
        meas_idx = self.pa_meas_combo.currentIndex()
        if meas_idx < 0 or meas_idx >= len(self.channels):
            return
        bg_val = self.pa_bg_manual_spin.value()
        if bg_val <= 0:
            return
        meas_img = self._get_current_slice(self.channels[meas_idx])
        if meas_img is None:
            return
        from scipy import ndimage as ndi
        signal_mask = meas_img.astype(np.float64) > bg_val
        dilated = ndi.binary_dilation(signal_mask, iterations=1)
        outline = (dilated & ~signal_mask).astype(np.uint8)
        scale = self._pa_get_scale()
        existing = self._pa_find_layer('Signal Outlines')
        if existing is not None:
            existing.data = outline
        else:
            from napari.utils.colormaps import DirectLabelColormap
            green_cmap = DirectLabelColormap(
                color_dict={1: (0.0, 1.0, 0.0, 1.0), None: (0, 0, 0, 0)}
            )
            self.viewer.add_labels(
                outline, name='Signal Outlines',
                opacity=0.8, colormap=green_cmap, scale=scale,
            )

    def _pa_clear_bg_rois(self):
        if self._pa_bg_shapes_layer is not None:
            self._pa_bg_shapes_layer.data = []
        self.pa_bg_value_label.setText("Background: -- (draw rectangles to measure)")
        self.pa_bg_manual_spin.setValue(0)
        self._pa_remove_layer('Signal Mask')

    # -- run analysis ----------------------------------------------------------

    def _pa_run_analysis(self):
        """Run the full particle analysis pipeline with all new features."""
        if not self.channels or len(self.channels) < 2:
            QMessageBox.warning(self, "Error", "Load an image with at least 2 channels first")
            return

        det_idx = self.pa_det_combo.currentIndex()
        meas_idx = self.pa_meas_combo.currentIndex()
        if det_idx < 0 or meas_idx < 0:
            return

        det_img = self._get_current_slice(self.channels[det_idx])
        meas_img = self._get_current_slice(self.channels[meas_idx])
        if det_img is None or meas_img is None:
            return

        det_img = det_img.astype(np.float64)
        meas_img = meas_img.astype(np.float64)

        self.pa_run_btn.setEnabled(False)
        self.status_label.setText("Running particle analysis...")

        # Tracker logging
        import time as _time_mod
        start_time = _time_mod.time()
        pa_run_id = None
        if self.tracker:
            sample_id = self.current_file.stem if self.current_file else ""
            try:
                pa_run_id = self.tracker.log_particle_analysis(
                    sample_id=sample_id,
                    detection_channel=self.pa_det_combo.currentText(),
                    threshold_value=float(self._pa_thresh_spin.value()),
                    measurement_channel=self.pa_meas_combo.currentText(),
                    min_area=self.pa_min_area.value(),
                    max_area=self.pa_max_area.value(),
                    min_circularity=self.pa_min_circ.value(),
                    max_circularity=1.0,
                    bg_method='manual_roi',
                    input_path=str(self.current_file) if self.current_file else None,
                )
                self.last_run_id = pa_run_id
                self.session_run_ids.append(pa_run_id)
            except Exception:
                pa_run_id = None

        try:
            from ..core.particle_analysis import ParticleAnalyzer
            analyzer = ParticleAnalyzer()

            # Binarize
            mask = analyzer.binarize(det_img, float(self._pa_thresh_spin.value()))

            # Apply ignore regions
            ignore_mask = self._pa_get_ignore_mask(mask.shape)
            if ignore_mask is not None:
                mask[ignore_mask] = False

            # Optional watershed
            if self.pa_watershed_check.isChecked():
                from scipy import ndimage as ndi
                from skimage.segmentation import watershed, find_boundaries
                from skimage.feature import peak_local_max
                distance = ndi.distance_transform_edt(mask)
                min_dist = max(3, int(np.sqrt(self.pa_min_area.value() / np.pi)))
                coords = peak_local_max(distance, min_distance=min_dist,
                                        labels=mask.astype(int))
                local_max = np.zeros(mask.shape, dtype=bool)
                if len(coords) > 0:
                    local_max[tuple(coords.T)] = True
                markers, _ = ndi.label(local_max)
                ws_labels = watershed(-distance, markers, mask=mask)
                boundaries = find_boundaries(ws_labels, mode='inner')
                mask[boundaries] = False

            labels, particle_props = analyzer.find_particles(
                mask,
                min_area=self.pa_min_area.value(),
                max_area=self.pa_max_area.value(),
                min_circularity=self.pa_min_circ.value(),
                max_circularity=1.0,
            )
            n = int(labels.max())

            bg_value = self._pa_get_bg()

            import pandas as pd
            measurements = pd.DataFrame()
            if n > 0:
                if bg_value > 0:
                    from skimage.measure import regionprops_table
                    table = regionprops_table(
                        labels, intensity_image=meas_img,
                        properties=['label', 'mean_intensity', 'max_intensity', 'area'],
                    )
                    measurements = pd.DataFrame(table)
                    median_vals, integrated_vals = [], []
                    for lbl in measurements['label'].values:
                        px_vals = meas_img[labels == lbl].astype(np.float64)
                        median_vals.append(float(np.median(px_vals)))
                        integrated_vals.append(float(np.sum(px_vals)))
                    measurements['median_intensity'] = median_vals
                    measurements['integrated_intensity'] = integrated_vals
                    measurements['background'] = bg_value
                    measurements['mean_above_background'] = (
                        measurements['mean_intensity'] - bg_value)
                    measurements['snr'] = (
                        measurements['mean_above_background'] / max(bg_value, 1e-10))
                    measurements = measurements[[
                        'label', 'area', 'mean_intensity', 'median_intensity',
                        'max_intensity', 'integrated_intensity',
                        'background', 'mean_above_background', 'snr',
                    ]]
                else:
                    measurements = analyzer.measure_intensity(
                        labels, meas_img,
                        background_method='percentile',
                        background_percentile=50.0,
                    )
                    bg_value = float(measurements['background'].iloc[0]) \
                        if len(measurements) > 0 else 0.0

                results = pd.merge(particle_props, measurements, on='label',
                                   suffixes=('', '_meas'))
                if 'area_meas' in results.columns:
                    results = results.drop(columns=['area_meas'])

                # Per-pixel positivity
                pct_above = []
                for lbl in results['label'].values:
                    px_vals = meas_img[labels == int(lbl)]
                    n_above = int((px_vals > bg_value).sum())
                    pct_above.append(100.0 * n_above / max(len(px_vals), 1))
                results['pct_above_bg'] = pct_above
                results['is_positive'] = (
                    results['pct_above_bg'] >= self.pa_pos_pct_spin.value())
                results['fold_change'] = (
                    results['mean_intensity'] / max(bg_value, 1e-10))

                # Centroids
                from skimage.measure import regionprops
                rp = {r.label: r for r in regionprops(labels)}
                results['centroid_y'] = results['label'].map(
                    lambda l: rp[l].centroid[0] if l in rp else np.nan)
                results['centroid_x'] = results['label'].map(
                    lambda l: rp[l].centroid[1] if l in rp else np.nan)
            else:
                results = particle_props
                if 'pct_above_bg' not in results.columns:
                    results = results.copy()
                    results['pct_above_bg'] = 0.0
                    results['is_positive'] = False

            self._pa_labels = labels
            self._pa_results = results
            self._pa_summary = {
                'n_particles': n,
                'background': bg_value,
            }

            # Remove preview layers and show results
            for lname in ('Particles', 'Positive/Negative',
                           'Threshold Mask', 'Binary', 'Signal Mask', 'Signal Outlines'):
                self._pa_remove_layer(lname)
            if self._pa_binary_view:
                self.pa_binary_toggle.setChecked(False)
            self.pa_show_signal_mask.setChecked(False)

            scale = self._pa_get_scale()
            if n > 0:
                particles_layer = self.viewer.add_labels(
                    labels, name='Particles', opacity=0.5, scale=scale)
                self._pa_register_click_callback(particles_layer)

                if bg_value > 0 and len(measurements) > 0:
                    self._pa_draw_classification_overlay()

            # Summary text
            parts = [f"Particles: {n}"]
            if n > 0 and len(measurements) > 0:
                parts.append(f"BG: {bg_value:.0f}")
                if 'mean_intensity' in measurements.columns:
                    parts.append(
                        f"Mean int: {measurements['mean_intensity'].mean():.0f}")
                if 'mean_above_background' in measurements.columns:
                    parts.append(
                        f"Above BG: {measurements['mean_above_background'].mean():.0f}")
                if 'area' in particle_props.columns:
                    parts.append(f"Mean area: {particle_props['area'].mean():.0f} px")
                if 'is_positive' in results.columns:
                    n_pos = int(results['is_positive'].sum())
                    parts.append(f"Positive: {n_pos}/{n}")
            self.pa_summary_label.setText(" | ".join(parts))
            self.status_label.setText(f"Done - {n} particles found")

            self._pa_populate_table(results)
            self.pa_export_btn.setEnabled(n > 0)
            self.pa_export_fig_btn.setEnabled(n > 0)
            self._pa_append_folder_btn.setEnabled(n > 0)

            # Auto-scroll to show results table
            self._pa_scroll_to_results()

            # Update tracker
            if self.tracker and pa_run_id:
                try:
                    duration = _time_mod.time() - start_time
                    self.tracker.update_status(
                        pa_run_id,
                        status='completed',
                        duration_seconds=duration,
                        pa_particles_found=n,
                        pa_bg_value=bg_value,
                    )
                except Exception:
                    pass

            # Auto-save analysis state
            self._save_analysis_state()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Particle analysis error: {exc}")
            if self.tracker and pa_run_id:
                try:
                    self.tracker.update_status(pa_run_id, status='failed')
                except Exception:
                    pass
        finally:
            self.pa_run_btn.setEnabled(True)

    # -- results table ---------------------------------------------------------

    def _pa_draw_classification_overlay(self):
        """Draw green/red ring outlines around positive/negative particles."""
        if self._pa_labels is None or self._pa_results is None:
            return
        labels = self._pa_labels
        results = self._pa_results
        if labels.max() == 0 or len(results) == 0:
            return
        if 'is_positive' not in results.columns:
            return

        from scipy import ndimage as ndi
        from napari.utils.colormaps import DirectLabelColormap

        self._pa_remove_layer('Positive/Negative')
        scale = self._pa_get_scale()

        pos_outline = np.zeros(labels.shape, dtype=np.uint8)
        neg_outline = np.zeros(labels.shape, dtype=np.uint8)
        for _, row in results.iterrows():
            lbl = int(row['label'])
            single = (labels == lbl)
            outer = ndi.binary_dilation(single, iterations=6)
            inner = ndi.binary_dilation(single, iterations=4)
            ring = outer & ~inner
            if row['is_positive']:
                pos_outline[ring] = 1
            else:
                neg_outline[ring] = 1
        class_overlay = np.zeros(labels.shape, dtype=np.uint8)
        class_overlay[pos_outline > 0] = 1
        class_overlay[neg_outline > 0] = 2
        class_cmap = DirectLabelColormap(color_dict={
            1: (0.0, 1.0, 0.0, 1.0),
            2: (1.0, 0.0, 0.0, 1.0),
            None: (0, 0, 0, 0),
        })
        self.viewer.add_labels(
            class_overlay, name='Positive/Negative',
            opacity=0.3, colormap=class_cmap, scale=scale,
        )

    def _pa_reclassify_live(self, _val=None):
        """Re-classify particles and redraw outlines when % or BG threshold changes."""
        if self._pa_results is None or self._pa_labels is None:
            return
        if len(self._pa_results) == 0:
            return

        labels = self._pa_labels
        bg_value = self.pa_bg_manual_spin.value()
        if bg_value <= 0:
            return

        # Recompute pct_above_bg with current BG value
        meas_idx = self.pa_meas_combo.currentIndex()
        if meas_idx < 0 or meas_idx >= len(self.channels):
            return
        meas_img = self._get_current_slice(self.channels[meas_idx])
        if meas_img is None:
            return
        meas_img = meas_img.astype(np.float64)

        pct_above = []
        for lbl in self._pa_results['label'].values:
            px = meas_img[labels == int(lbl)]
            n_above = int((px > bg_value).sum())
            n_total = len(px)
            pct_above.append(100.0 * n_above / max(n_total, 1))
        self._pa_results['pct_above_bg'] = pct_above
        self._pa_results['background'] = bg_value
        self._pa_results['mean_above_background'] = self._pa_results['mean_intensity'] - bg_value

        # Reclassify
        min_pct = self.pa_pos_pct_spin.value()
        self._pa_results['is_positive'] = self._pa_results['pct_above_bg'] >= min_pct

        # Update summary
        n = len(self._pa_results)
        n_pos = int(self._pa_results['is_positive'].sum())
        self.pa_summary_label.setText(
            self.pa_summary_label.text().rsplit('Positive:', 1)[0]
            + f"Positive: {n_pos}/{n}"
        )
        # Redraw outlines and table
        self._pa_draw_classification_overlay()
        self._pa_populate_table(self._pa_results)

        # Auto-save after reclassification
        self._save_analysis_state()

    def _pa_populate_table(self, df):
        """Populate the results table with particle analysis results."""
        # Disable sorting while populating to avoid interference
        self.pa_results_table.setSortingEnabled(False)

        if df is None or len(df) == 0:
            self.pa_results_table.setRowCount(0)
            self.pa_results_table.setColumnCount(0)
            return

        show_cols = [c for c in [
            'label', 'area', 'mean_intensity', 'background',
            'pct_above_bg', 'is_positive', 'snr', 'circularity',
        ] if c in df.columns]

        self.pa_results_table.setColumnCount(len(show_cols))
        self.pa_results_table.setHorizontalHeaderLabels(show_cols)
        self.pa_results_table.setRowCount(min(len(df), 200))

        for row_idx, (_, row) in enumerate(df.head(200).iterrows()):
            for col_idx, col in enumerate(show_cols):
                val = row[col]
                text = f"{val:.2f}" if isinstance(val, float) else str(val)
                item = _NumericTableItem(text)
                self.pa_results_table.setItem(row_idx, col_idx, item)

        self.pa_results_table.resizeColumnsToContents()
        self.pa_results_table.setSortingEnabled(True)

    def _pa_activate_ignore_drawing(self):
        """Create or activate the Ignore Regions shapes layer."""
        if (self._pa_ignore_shapes_layer is None
                or self._pa_ignore_shapes_layer not in self.viewer.layers):
            scale = self._pa_get_scale()
            self._pa_ignore_shapes_layer = self.viewer.add_shapes(
                name="Ignore Regions",
                edge_color="red",
                edge_width=2,
                face_color=[1.0, 0.0, 0.0, 0.15],
                scale=scale,
            )
        self.viewer.layers.selection.active = self._pa_ignore_shapes_layer
        self._pa_ignore_shapes_layer.mode = 'add_polygon'
        self.status_label.setText(
            "Draw polygon around area to ignore. Press Escape when done.")

    def _pa_clear_ignore_regions(self):
        """Clear all ignore regions."""
        if (self._pa_ignore_shapes_layer is not None
                and self._pa_ignore_shapes_layer in self.viewer.layers):
            self._pa_ignore_shapes_layer.data = []
        self.status_label.setText("Ignore regions cleared")

    def _pa_get_ignore_mask(self, shape):
        """Build a boolean mask of pixels to ignore (True = ignore)."""
        if (self._pa_ignore_shapes_layer is None
                or self._pa_ignore_shapes_layer not in self.viewer.layers
                or len(self._pa_ignore_shapes_layer.data) == 0):
            return None

        from skimage.draw import polygon as draw_polygon
        mask = np.zeros(shape, dtype=bool)
        scale = self._pa_get_scale()
        sy = scale[0] if len(scale) > 0 else 1.0
        sx = scale[1] if len(scale) > 1 else 1.0

        for shape_data in self._pa_ignore_shapes_layer.data:
            verts = np.array(shape_data)
            # Convert from world to pixel coords if scaled
            if sy != 1.0 or sx != 1.0:
                verts = verts.copy()
                verts[:, 0] /= sy
                verts[:, 1] /= sx
            rr, cc = draw_polygon(verts[:, 0], verts[:, 1], shape=shape)
            mask[rr, cc] = True
        return mask

    def _pa_append_to_folder_csv(self):
        """Append particle results (no ROIs) to a master CSV in the image folder."""
        if self._pa_results is None or len(self._pa_results) == 0:
            QMessageBox.warning(self, "Error", "Run particle analysis first")
            return
        if not self.current_file:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        import pandas as pd
        import json

        folder = self.current_file.parent
        sample_name = self.current_file.stem
        master_path = folder / f"{folder.name}_particles.csv"

        detail = self._pa_results.copy()
        detail.insert(0, 'sample', sample_name)

        if master_path.exists():
            existing = pd.read_csv(master_path, comment='#')
            if sample_name in existing.get('sample', pd.Series()).values:
                reply = QMessageBox.question(
                    self, "Duplicate",
                    f"'{sample_name}' already exists in {master_path.name}.\n"
                    "Replace its rows?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    existing = existing[existing['sample'] != sample_name]
                else:
                    return
            combined = pd.concat([existing, detail], ignore_index=True)
        else:
            combined = detail

        settings = self._pa_get_settings()
        with open(master_path, 'w', newline='') as f:
            f.write(f"# Settings: {json.dumps(settings)}\n")
            combined.to_csv(f, index=False)

        n_samples = combined['sample'].nunique()
        self.status_label.setText(
            f"Appended {sample_name} to {master_path.name} "
            f"({n_samples} samples, {len(combined)} particles)")

    # -- click interaction -----------------------------------------------------

    def _pa_register_click_callback(self, particles_layer):
        """Register a mouse callback on the Particles layer for click-to-select."""
        widget_ref = self  # prevent closure over self in nested function

        @particles_layer.mouse_drag_callbacks.append
        def _on_particle_click(layer, event):
            if event.type != 'mouse_press':
                return
            # Convert world position to data coordinates
            try:
                coords = layer.world_to_data(event.position)
                row = int(round(coords[-2]))
                col = int(round(coords[-1]))
                if (0 <= row < layer.data.shape[-2] and
                        0 <= col < layer.data.shape[-1]):
                    label_val = int(layer.data[row, col])
                    if label_val > 0:
                        widget_ref._pa_highlight_particle(label_val, layer)
                    else:
                        # Clicked background -- clear highlight
                        widget_ref._pa_clear_highlight()
            except Exception:
                pass

    def _pa_highlight_particle(self, label_val, particles_layer=None):
        """Highlight a particle with a bright outline ring and select in table."""
        if self._pa_results is None or self._pa_labels is None:
            return
        match = self._pa_results[self._pa_results['label'] == label_val]
        if match.empty:
            return
        row_idx = match.index[0]
        if row_idx < self.pa_results_table.rowCount():
            self.pa_results_table.selectRow(row_idx)
            item = self.pa_results_table.item(row_idx, 0)
            if item is not None:
                self.pa_results_table.scrollToItem(item)

        # Draw bright outline ring on a dedicated highlight layer
        from scipy import ndimage as ndi
        single = (self._pa_labels == label_val)
        outer = ndi.binary_dilation(single, iterations=8)
        inner = ndi.binary_dilation(single, iterations=5)
        ring = (outer & ~inner).astype(np.uint8)

        scale = self._pa_get_scale()
        existing = self._pa_find_layer('Selected Particle')
        if existing is not None:
            existing.data = ring
        else:
            from napari.utils.colormaps import DirectLabelColormap
            cmap = DirectLabelColormap(color_dict={
                1: (0.0, 1.0, 1.0, 1.0),  # cyan ring
                None: (0, 0, 0, 0),
            })
            self.viewer.add_labels(
                ring, name='Selected Particle',
                opacity=1.0, colormap=cmap, scale=scale,
            )

        area = int(match.iloc[0]['area']) if 'area' in match.columns else '?'
        self.status_label.setText(
            f"Particle {label_val} | area={area} px")

    def _pa_clear_highlight(self):
        """Remove the particle highlight ring and deselect table."""
        self._pa_remove_layer('Selected Particle')
        self.pa_results_table.clearSelection()
        self.status_label.setText("")

    def _pa_on_table_row_clicked(self, row, col):
        """Center the viewer on the clicked particle and highlight it."""
        if self._pa_results is None or row >= len(self._pa_results):
            return
        particle = self._pa_results.iloc[row]
        if 'centroid_y' not in particle or 'centroid_x' not in particle:
            return
        cy, cx = float(particle['centroid_y']), float(particle['centroid_x'])

        # Apply scale for world coordinates
        scale = self._pa_get_scale()
        world_y = cy * scale[0] if len(scale) > 0 else cy
        world_x = cx * scale[1] if len(scale) > 1 else cx

        # Pan viewer to particle
        self.viewer.camera.center = (world_y, world_x)

        # Highlight with ring
        label_val = int(particle['label'])
        self._pa_highlight_particle(label_val)

        area = int(particle['area']) if 'area' in particle else '?'
        self.status_label.setText(
            f"Particle {label_val} | area={area} px")

    # -- export ----------------------------------------------------------------

    def _pa_export_csv(self):
        """Export particle analysis results to CSV with settings metadata."""
        if self._pa_results is None or len(self._pa_results) == 0:
            return

        default_name = "particle_analysis.csv"
        if self.current_file:
            default_name = f"{self.current_file.stem}_particles.csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Particle Results", default_name,
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            import json
            settings = self._pa_get_settings()
            settings['export_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            # Write settings as comment header, then CSV data
            with open(path, 'w', newline='') as f:
                f.write(f"# Settings: {json.dumps(settings)}\n")
                self._pa_results.to_csv(f, index=False)
            # Also save settings JSON alongside
            settings_path = path.replace('.csv', '_settings.json')
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            self.status_label.setText(f"Exported to {Path(path).name}")

    def _pa_export_figure(self):
        """Export a QC figure (detection / measurement / intensity histogram)."""
        if self._pa_labels is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "particle_qc.png", "PNG (*.png)"
        )
        if not path:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        det_idx = self.pa_det_combo.currentIndex()
        meas_idx = self.pa_meas_combo.currentIndex()
        det_img = self._get_current_slice(self.channels[det_idx])
        meas_img = self._get_current_slice(self.channels[meas_idx])
        if det_img is None or meas_img is None:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        def _scalebar(ax, pixel_um):
            if not pixel_um:
                return
            img_width_um = ax.get_xlim()[1] * pixel_um
            target = img_width_um * 0.15
            nice_lengths = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
            bar_um = min(nice_lengths, key=lambda x: abs(x - target))
            bar_px = bar_um / pixel_um
            y_max = ax.get_ylim()[0]
            x_max = ax.get_xlim()[1]
            margin = x_max * 0.03
            y_pos = y_max - margin
            x_end = x_max - margin
            x_start = x_end - bar_px
            ax.plot([x_start, x_end], [y_pos, y_pos], color='white', linewidth=3)
            label = f"{bar_um} um" if bar_um >= 1 else f"{bar_um*1000:.0f} nm"
            ax.text((x_start + x_end) / 2, y_pos - margin * 0.8, label,
                    color='white', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

        axes[0].imshow(det_img, cmap='gray',
                       vmax=np.percentile(det_img, 99.5))
        axes[0].set_title(f"Detection (ch{det_idx})")
        axes[0].axis('off')
        _scalebar(axes[0], self._pixel_size_um)

        axes[1].imshow(meas_img, cmap='gray',
                       vmax=np.percentile(meas_img, 99.5))
        if self._pa_labels is not None and self._pa_labels.max() > 0:
            from scipy import ndimage
            for lbl in range(1, int(self._pa_labels.max()) + 1):
                single = (self._pa_labels == lbl).astype(np.uint8)
                contour = ndimage.binary_dilation(single) & ~single.astype(bool)
                axes[1].contour(contour, colors=['cyan'], linewidths=0.5)
        axes[1].set_title(f"Particles on ch{meas_idx}")
        axes[1].axis('off')
        _scalebar(axes[1], self._pixel_size_um)

        if (self._pa_results is not None
                and 'mean_intensity' in self._pa_results.columns):
            bg = (self._pa_results['background'].iloc[0]
                  if 'background' in self._pa_results.columns else 0)
            self._pa_results['mean_intensity'].hist(
                ax=axes[2], bins=20, color='steelblue', edgecolor='white')
            axes[2].axvline(bg, color='red', linestyle='--', label=f'BG={bg:.0f}')
            axes[2].legend()
        axes[2].set_title("Intensity Distribution")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        self.status_label.setText(f"Figure saved: {Path(path).name}")

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

            # Auto-load the image
            self._load_image()

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
                # Auto-populate LoG pixel_um spinner
                if hasattr(self, 'log_pixel_um_spin'):
                    self.log_pixel_um_spin.setValue(voxel['x'])
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

            # Update pixel size from metadata before creating layers
            voxel = metadata.get('voxel_size_um')
            if voxel and voxel.get('x', 1.0) != 1.0:
                self._pixel_size_um = voxel['x']
            scale = self._pa_get_scale()

            # Add to napari as stack with proper contrast
            self.viewer.layers.clear()
            self.viewer.add_image(
                red_stack,
                name="Nuclear (red) Stack",
                colormap='red',
                blending='additive',
                contrast_limits=red_limits,
                scale=scale,
            )
            self.viewer.add_image(
                green_stack,
                name="Signal (green) Stack",
                colormap='green',
                blending='additive',
                contrast_limits=green_limits,
                scale=scale,
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

    def _on_load_finished(self, success: bool, message: str, red, green, metadata, full_data=None):
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

                # Store all channels for particle analysis.
                # red/green are already rotated and in the correct order
                # (channels[0]=red=nuclear, channels[1]=green=signal).
                red_idx = self.red_channel_spin.value()
                green_idx = self.green_channel_spin.value()
                if (full_data is not None and full_data.ndim == 3
                        and full_data.shape[0] > 2):
                    # >2 channels: red and green first, then extras (rotated)
                    self.channels = [red, green]
                    used = {red_idx, green_idx}
                    for i in range(full_data.shape[0]):
                        if i not in used:
                            self.channels.append(self._apply_rotation(full_data[i]))
                else:
                    self.channels = [red, green]

                # Channel names must match the napari layer names used below
                # channels[0]=red, channels[1]=green; layers added in same order
                if len(self.channels) == 2:
                    self.channel_names = ["Nuclear (red)", "Signal (green)"]
                else:
                    # >2 channels: named red/green first, then extras from metadata
                    ch_meta = metadata.get('channels', [])
                    self.channel_names = ["Nuclear (red)", "Signal (green)"]
                    used = {red_idx, green_idx}
                    for i in range(full_data.shape[0]):
                        if i not in used:
                            name = str(ch_meta[i]) if i < len(ch_meta) else f"Channel {i}"
                            self.channel_names.append(name)

                # Update particle analysis channel combos
                self._update_particle_channel_combos()

                print(f"[BrainSlice] Load finished: red={red.shape}, green={green.shape}, total_channels={len(self.channels)}")

                # Update pixel size from full load metadata BEFORE creating layers
                # so all layers (image + overlays) get consistent scale
                voxel = metadata.get('voxel_size_um')
                if voxel and voxel.get('x', 1.0) != 1.0:
                    self._pixel_size_um = voxel['x']
                    if not self._size_manually_set:
                        self._calibrate_from_pixel_size()
                    self._update_area_label()
                    if hasattr(self, 'log_pixel_um_spin'):
                        self.log_pixel_um_spin.setValue(voxel['x'])

                # Calculate contrast limits (Auto = None lets napari decide)
                red_limits = self._get_contrast_limits(red)
                green_limits = self._get_contrast_limits(green)

                # Add to napari with proper contrast and physical scale
                scale = self._pa_get_scale()
                self.viewer.layers.clear()
                self.viewer.add_image(
                    red,
                    name="Nuclear (red)",
                    colormap='red',
                    blending='additive',
                    contrast_limits=red_limits,
                    scale=scale,
                )
                self.viewer.add_image(
                    green,
                    name="Signal (green)",
                    colormap='green',
                    blending='additive',
                    contrast_limits=green_limits,
                    scale=scale,
                )

                # Enable scale bar
                if self._pixel_size_um and self._pixel_size_um > 0:
                    self.viewer.scale_bar.visible = True
                    self.viewer.scale_bar.unit = 'um'
                    self.viewer.scale_bar.colored = True
                    self.viewer.scale_bar.font_size = 12

                # (pixel_size_um already updated above, before layer creation)

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

            # Update image navigation labels
            self._nav_update_label()

            # Try to restore previous analysis
            self._restore_analysis_state()

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
            elif backend == 'threshold+log':
                # Threshold+LoG production pipeline params
                params['pixel_um'] = self.log_pixel_um_spin.value()
                params['min_diameter_um'] = self.log_min_diam_spin.value()
                params['max_diameter_um'] = self.log_max_diam_spin.value()
                params['threshold_fraction'] = self.log_thresh_fraction_spin.value()
                params['log_threshold'] = self.log_sensitivity_spin.value()
                params['gaussian_sigma'] = self.thresh_gauss_spin.value()
                params['use_hysteresis'] = self.thresh_hysteresis_check.isChecked()
                params['hysteresis_low_fraction'] = self.thresh_hysteresis_low_spin.value()
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
            lbl_layer = self.viewer.add_labels(
                results['merged_labels'],
                name=f"Nuclei ({count})",
            )
            lbl_layer.contour = 2

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
                        border_color='cyan',
                        border_width=0.5,
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
                        border_color='yellow',
                        border_width=0.5,
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
        print(f"[BrainSlice] _on_detect_finished called: success={success}, count={count}, "
              f"labels type={type(labels)}, labels shape={getattr(labels, 'shape', 'N/A')}")

        if success:
            self.nuclei_labels = labels

            # Update tracker (wrapped so failures don't block layer creation)
            try:
                if self.tracker and self.last_run_id:
                    self.tracker.update_status(
                        self.last_run_id,
                        status='completed',
                        det_nuclei_found=count,
                    )
                    self.session_run_ids.append(self.last_run_id)
            except Exception as e:
                print(f"[BrainSlice] WARNING: Tracker update failed (non-fatal): {e}")

            # Add labels to napari
            try:
                # Remove old detection layer if exists
                for layer in list(self.viewer.layers):
                    if 'Nuclei' in layer.name:
                        self.viewer.layers.remove(layer)

                print(f"[BrainSlice] Adding labels layer: shape={labels.shape}, "
                      f"dtype={labels.dtype}, max_label={labels.max()}")
                lbl_layer = self.viewer.add_labels(labels, name=f"Nuclei ({count})")
                lbl_layer.contour = 2
                print(f"[BrainSlice] Labels layer added successfully")
            except Exception as e:
                import traceback
                print(f"[BrainSlice] ERROR adding labels layer: {e}")
                traceback.print_exc()

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
                try:
                    self.tracker.update_status(self.last_run_id, status='failed')
                except Exception:
                    pass

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

        # Threshold+LoG-specific info
        elif backend == 'threshold+log':
            decision = metrics.get('decision', '?')
            n_thresh = metrics.get('n_threshold', 0)
            n_log = metrics.get('n_log_new', 0)
            thresh_val = metrics.get('threshold_value', 0)
            n_artifact = metrics.get('n_artifact_pixels', 0)
            lines.append(f"Decision: {decision}")
            lines.append(
                f"Threshold: {n_thresh} nuclei | LoG added: +{n_log} | "
                f"Threshold value: {thresh_val:.1f}"
            )
            if n_artifact > 0:
                lines.append(f"Artifact pixels masked: {n_artifact:,}")

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
            'sigma_threshold': self.sigma_threshold_spin.value(),
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

        # Ch1 params (red / mCherry -- nuclear)
        params_ch1 = {
            'background_method': self.bg_method_combo.currentText(),
            'background_percentile': self.bg_percentile_spin.value(),
            'threshold_method': self.thresh_method_combo.currentText(),
            'threshold_value': self.thresh_value_spin.value(),
            'sigma_threshold': self.sigma_threshold_spin.value(),
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

            # Auto-save measurements CSV (non-fatal if fails)
            measurements_path = None
            try:
                if self.current_file is not None:
                    from ..core.config import get_sample_dir, SampleDirs
                    stem = self.current_file.stem
                    sample_dir = get_sample_dir(stem)
                    results_dir = sample_dir / SampleDirs.QUANTIFIED
                    results_dir.mkdir(parents=True, exist_ok=True)
                    run_tag = self.last_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                    measurements_path = results_dir / f"{stem}_{run_tag}_measurements.csv"
                    measurements.to_csv(measurements_path, index=False)
            except Exception as e:
                print(f"[BrainSlice] WARNING: Auto-save measurements failed (non-fatal): {e}")

            # Update tracker (non-fatal if fails)
            try:
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
            except Exception as e:
                print(f"[BrainSlice] WARNING: Tracker update failed (non-fatal): {e}")

            # Refresh run history panel
            try:
                self._refresh_run_history()
            except Exception as e:
                print(f"[BrainSlice] WARNING: Run history refresh failed (non-fatal): {e}")

            # Visualize results - color nuclei by positive/negative
            try:
                self._visualize_colocalization(measurements)
                print(f"[BrainSlice] Colocalization visualization added successfully")
            except Exception as e:
                import traceback
                print(f"[BrainSlice] ERROR adding colocalization layers: {e}")
                traceback.print_exc()

            # Update diagnostic plot
            try:
                self._update_diagnostic_plot()
            except Exception as e:
                print(f"[BrainSlice] WARNING: Diagnostic plot update failed (non-fatal): {e}")

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
                try:
                    self.tracker.update_status(self.last_run_id, status='failed')
                except Exception:
                    pass

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
                border_color='lime',
                border_width=0.5,
            )

        # Add negative cells (red)
        if len(negative) > 0:
            neg_coords = negative[[y_col, x_col]].values
            self.viewer.add_points(
                neg_coords,
                name=f"Negative ({len(negative)})",
                size=10,
                face_color='transparent',
                border_color='red',
                border_width=0.5,
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
                    border_color=color,
                    border_width=0.5,
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
        if self.roi_shapes_layer is not None:
            if self.roi_shapes_layer not in self.viewer.layers:
                self.roi_shapes_layer = None

        if self.roi_shapes_layer is None:
            self.roi_shapes_layer = self.viewer.add_shapes(
                name="ROIs",
                edge_color="white",
                edge_width=2,
                face_color=[1.0, 1.0, 1.0, 0.08],
            )

        self.viewer.layers.selection.active = self.roi_shapes_layer
        self.roi_shapes_layer.mode = 'add_polygon'
        self.status_label.setText("Draw ROI polygon. Press Escape when done.")

    def _add_named_roi(self, name=None):
        """Start drawing a new ROI with a given name."""
        if name is None:
            name = f"ROI {len(self._roi_names) + 1}"
        self._roi_names.append(name)
        self._update_roi_names_label()
        self._add_roi_layer()
        self.status_label.setText(f"Draw '{name}' ROI polygon. Press Escape when done.")

    def _get_roi_name(self, index):
        """Get the name for an ROI by index, falling back to numbered."""
        if index < len(self._roi_names):
            return self._roi_names[index]
        return f"ROI {index + 1}"

    def _update_roi_names_label(self):
        """Update the ROI names display label."""
        if hasattr(self, '_roi_names_label'):
            if self._roi_names:
                names_str = ", ".join(self._roi_names)
                self._roi_names_label.setText(f"ROIs: {names_str}")
            else:
                self._roi_names_label.setText("ROIs: (none)")

    def _save_rois(self):
        """Save current ROI polygons and names to a JSON file."""
        if self.roi_shapes_layer is None or len(self.roi_shapes_layer.data) == 0:
            QMessageBox.warning(self, "Error", "No ROIs to save")
            return

        import json
        default_name = "rois.json"
        if self.current_file:
            default_name = f"{self.current_file.stem}_rois.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs", default_name,
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return

        rois = []
        for i, shape_data in enumerate(self.roi_shapes_layer.data):
            verts = np.array(shape_data).tolist()
            rois.append({
                'name': self._get_roi_name(i),
                'vertices': verts,
            })

        data = {
            'version': 1,
            'n_rois': len(rois),
            'roi_names': self._roi_names,
            'rois': rois,
        }
        if self.current_file:
            data['source_image'] = str(self.current_file.name)
        if self._pixel_size_um:
            data['pixel_size_um'] = self._pixel_size_um

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.status_label.setText(f"Saved {len(rois)} ROIs to {Path(path).name}")

    def _load_rois(self):
        """Load ROI polygons from a JSON file onto the current viewer."""
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROIs", "",
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return

        try:
            with open(path) as f:
                data = json.load(f)

            rois = data.get('rois', [])
            if not rois:
                QMessageBox.warning(self, "Error", "No ROIs found in file")
                return

            # Restore names
            self._roi_names = data.get('roi_names', [])
            if not self._roi_names:
                self._roi_names = [r.get('name', f"ROI {i+1}")
                                   for i, r in enumerate(rois)]

            # Create/clear shapes layer
            self._add_roi_layer()
            self.roi_shapes_layer.data = []

            # Add each ROI polygon
            for roi in rois:
                verts = np.array(roi['vertices'])
                self.roi_shapes_layer.add_polygons([verts])

            self.roi_shapes_layer.mode = 'pan_zoom'
            self._update_roi_names_label()
            self.status_label.setText(
                f"Loaded {len(rois)} ROIs from {Path(path).name} "
                f"-- use transform tools to adjust if needed")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load ROIs: {e}")

    def _clear_rois(self):
        """Clear all ROIs and names."""
        if self.roi_shapes_layer is not None:
            if self.roi_shapes_layer in self.viewer.layers:
                self.roi_shapes_layer.data = []
        self._roi_names = []
        self._update_roi_names_label()
        self.status_label.setText("ROIs cleared")

    def _get_roi_data_source(self):
        """Get the measurements DataFrame for ROI counting.

        Returns whichever results are available: cell_measurements from
        nuclei-based Signal analysis, or _pa_results from particle analysis.
        """
        if self.cell_measurements is not None and self._pa_results is not None:
            # Both available -- prefer the most recently computed
            # (cell_measurements from nuclei mode, _pa_results from particle mode)
            # For now, prefer particle results since it's the primary mode
            return self._pa_results
        if self._pa_results is not None:
            return self._pa_results
        if self.cell_measurements is not None:
            return self.cell_measurements
        return None

    def _count_all_rois(self):
        """Count positive/negative cells in all drawn ROIs."""
        measurements = self._get_roi_data_source()
        if measurements is None:
            QMessageBox.warning(self, "Error",
                "Run Signal Analysis (Particle or Nuclei mode) first")
            return

        if self.roi_shapes_layer is None or len(self.roi_shapes_layer.data) == 0:
            QMessageBox.warning(self, "Error", "Draw at least one ROI first")
            return

        from ..core.colocalization import filter_measurements_by_roi

        # Determine image shape -- try red_channel first, fall back to channels[0]
        img = None
        if self.red_channel is not None:
            img = self._get_current_slice(self.red_channel)
        elif self.channels:
            img = self._get_current_slice(self.channels[0])
        if img is None:
            QMessageBox.warning(self, "Error", "No image loaded")
            return
        image_shape = img.shape[:2]

        # Update source label if it exists (ROI Count tab)
        if hasattr(self, '_roi_source_label'):
            source_name = 'particle' if measurements is self._pa_results else 'nuclei'
            n = len(measurements)
            self._roi_source_label.setText(
                f"Data source: {source_name} results ({n} objects)")

        # Determine scale to convert ROI vertices from world to pixel coords
        scale = self._pa_get_scale()
        scale_y = scale[0] if len(scale) > 0 else 1.0
        scale_x = scale[1] if len(scale) > 1 else 1.0

        # Build per-particle ROI assignment for detailed export
        import pandas as pd
        roi_assignment = pd.Series('Outside', index=measurements.index)

        results = []
        for i, shape_data in enumerate(self.roi_shapes_layer.data):
            vertices = np.array(shape_data)  # Nx2 (y, x) in world coords
            # Convert from world coordinates to pixel coordinates
            if scale_y != 1.0 or scale_x != 1.0:
                vertices = vertices.copy()
                vertices[:, 0] /= scale_y
                vertices[:, 1] /= scale_x
            filtered = filter_measurements_by_roi(
                measurements, vertices, image_shape
            )

            # Assign ROI name to particles (first match wins)
            roi_name = self._get_roi_name(i)
            for idx in filtered.index:
                if roi_assignment[idx] == 'Outside':
                    roi_assignment[idx] = roi_name

            total = len(filtered)
            is_dual_mode = 'classification' in filtered.columns

            if is_dual_mode and total > 0:
                n_dual = int((filtered['classification'] == 'dual').sum())
                n_red = int((filtered['classification'] == 'red_only').sum())
                n_green = int((filtered['classification'] == 'green_only').sum())
                n_neither = int((filtered['classification'] == 'neither').sum())
                results.append({
                    'roi': self._get_roi_name(i),
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
                    'roi': self._get_roi_name(i),
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
        # Store per-particle detail with ROI assignment
        detail = measurements.copy()
        detail.insert(0, 'roi', roi_assignment)
        self._roi_detail_data = detail

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
                ["ROI", "Total", "Positive", "Negative", "Fraction"]
            )
            self.roi_results_table.setRowCount(len(results))
            for row_idx, r in enumerate(results):
                self.roi_results_table.setItem(row_idx, 0, QTableWidgetItem(r['roi']))
                self.roi_results_table.setItem(row_idx, 1, QTableWidgetItem(str(r['total'])))
                self.roi_results_table.setItem(row_idx, 2, QTableWidgetItem(str(r.get('positive', 0))))
                self.roi_results_table.setItem(row_idx, 3, QTableWidgetItem(str(r.get('negative', 0))))
                self.roi_results_table.setItem(row_idx, 4, QTableWidgetItem(f"{r.get('fraction', 0)*100:.1f}%"))

        self.status_label.setText(f"Counted cells in {len(results)-1} ROI(s)")

        # Auto-save analysis state
        self._save_analysis_state()

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

        import json
        settings = self._pa_get_settings()
        settings['export_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        settings['roi_names'] = self._roi_names

        # Primary output: per-particle detail with roi column
        if hasattr(self, '_roi_detail_data') and self._roi_detail_data is not None:
            with open(path, 'w', newline='') as f:
                f.write(f"# Settings: {json.dumps(settings)}\n")
                self._roi_detail_data.to_csv(f, index=False)
        else:
            # Fallback: summary only if no detail available
            is_dual = (self._roi_counts_data
                       and self._roi_counts_data[0].get('_dual_mode', False))
            if is_dual:
                fieldnames = ['roi', 'total', 'dual', 'red_only',
                              'green_only', 'neither', 'frac_dual']
            else:
                fieldnames = ['roi', 'total', 'positive', 'negative', 'fraction']
            clean_data = [{k: v for k, v in r.items() if not k.startswith('_')}
                          for r in self._roi_counts_data]
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(clean_data)

        # Also write summary alongside
        summary_path = path.replace('.csv', '_summary.csv')
        if summary_path != path:
            is_dual = (self._roi_counts_data
                       and self._roi_counts_data[0].get('_dual_mode', False))
            if is_dual:
                fieldnames = ['roi', 'total', 'dual', 'red_only',
                              'green_only', 'neither', 'frac_dual']
            else:
                fieldnames = ['roi', 'total', 'positive', 'negative', 'fraction']
            clean_data = [{k: v for k, v in r.items() if not k.startswith('_')}
                          for r in self._roi_counts_data]
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(clean_data)

        self.status_label.setText(
            f"Exported to {Path(path).name} + {Path(summary_path).name}")

    def _append_to_folder_csv(self):
        """Append current image's ROI results to a master CSV in the image folder."""
        if not hasattr(self, '_roi_detail_data') or self._roi_detail_data is None:
            QMessageBox.warning(self, "Error", "Run ROI counting first")
            return

        # Determine output path
        if self.current_file:
            folder = self.current_file.parent
            sample_name = self.current_file.stem
        else:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        master_path = folder / f"{folder.name}_roi_results.csv"

        import pandas as pd

        # Add sample column
        detail = self._roi_detail_data.copy()
        detail.insert(0, 'sample', sample_name)

        # Append or create
        if master_path.exists():
            # Read existing to check for duplicate sample
            existing = pd.read_csv(master_path, comment='#')
            if sample_name in existing.get('sample', pd.Series()).values:
                reply = QMessageBox.question(
                    self, "Duplicate",
                    f"'{sample_name}' already exists in {master_path.name}.\n"
                    "Replace its rows?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    existing = existing[existing['sample'] != sample_name]
                else:
                    return
            combined = pd.concat([existing, detail], ignore_index=True)
        else:
            combined = detail

        # Write with settings header
        import json
        settings = self._pa_get_settings()
        settings['roi_names'] = self._roi_names
        with open(master_path, 'w', newline='') as f:
            f.write(f"# Settings: {json.dumps(settings)}\n")
            combined.to_csv(f, index=False)

        n_samples = combined['sample'].nunique()
        self.status_label.setText(
            f"Appended {sample_name} to {master_path.name} "
            f"({n_samples} samples, {len(combined)} particles)")

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
