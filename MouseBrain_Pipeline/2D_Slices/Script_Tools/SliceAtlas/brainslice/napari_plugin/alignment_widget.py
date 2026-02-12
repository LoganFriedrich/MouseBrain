"""
alignment_widget.py - Atlas alignment and overlay widget for BrainSlice

Provides interactive atlas alignment with:
- Automatic registration (position detection + alignment)
- Manual alignment controls (position, rotation, scale)
- Atlas overlay with opacity control
- ABBA export/import as fallback
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QSlider, QCheckBox, QFileDialog, QMessageBox, QProgressBar,
)
from qtpy.QtCore import Qt, QThread, Signal


class AutoDetectWorker(QThread):
    """Worker thread for automatic atlas position detection."""

    progress = Signal(int, int)  # current, total
    finished = Signal(bool, str, object)  # success, message, result

    def __init__(self, image, atlas_manager, orientation):
        super().__init__()
        self.image = image
        self.atlas_manager = atlas_manager
        self.orientation = orientation

    def run(self):
        try:
            from ..core.registration import find_best_atlas_slice

            def progress_cb(current, total):
                self.progress.emit(current, total)

            result = find_best_atlas_slice(
                image=self.image,
                atlas_manager=self.atlas_manager,
                orientation=self.orientation,
                step=5,  # Check every 5th slice for speed
                progress_callback=progress_cb,
            )

            self.finished.emit(True, "Position detected", result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None)


class AutoAlignWorker(QThread):
    """Worker thread for automatic alignment refinement (legacy)."""

    finished = Signal(bool, str, object)  # success, message, result

    def __init__(self, image, atlas_slice, method='similarity'):
        super().__init__()
        self.image = image
        self.atlas_slice = atlas_slice
        self.method = method

    def run(self):
        try:
            from ..core.registration import register_to_atlas

            result = register_to_atlas(
                image=self.image,
                atlas_slice=self.atlas_slice,
                method=self.method,
            )

            self.finished.emit(True, "Alignment complete", result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None)


class DeepSliceWorker(QThread):
    """Worker thread for DeepSlice batch position detection."""

    progress = Signal(str)  # status message
    finished = Signal(bool, str, object)  # success, message, predictions

    def __init__(self, folder_path, use_cache=True):
        super().__init__()
        self.folder_path = folder_path
        self.use_cache = use_cache

    def run(self):
        try:
            from ..core.deepslice_wrapper import DeepSliceWrapper, is_deepslice_available

            if not is_deepslice_available():
                self.finished.emit(
                    False,
                    "DeepSlice not installed. Install with: pip install DeepSlice",
                    None
                )
                return

            wrapper = DeepSliceWrapper(species='mouse')

            def progress_cb(msg):
                self.progress.emit(msg)

            predictions = wrapper.predict_folder(
                self.folder_path,
                ensemble=True,
                use_cache=self.use_cache,
                progress_callback=progress_cb,
            )

            self.finished.emit(True, f"Detected {predictions['n_slices']} slices", predictions)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None)


class ElastixRegistrationWorker(QThread):
    """Worker thread for SimpleElastix registration."""

    finished = Signal(bool, str, object)  # success, message, result

    def __init__(self, image, atlas_ref, atlas_annot, grid_spacing=8):
        super().__init__()
        self.image = image
        self.atlas_ref = atlas_ref
        self.atlas_annot = atlas_annot
        self.grid_spacing = grid_spacing

    def run(self):
        try:
            from ..core.elastix_registration import warp_atlas_to_image, is_elastix_available

            if not is_elastix_available():
                self.finished.emit(
                    False,
                    "SimpleElastix not installed. Install with: pip install SimpleITK-SimpleElastix",
                    None
                )
                return

            result = warp_atlas_to_image(
                atlas_reference=self.atlas_ref,
                atlas_annotation=self.atlas_annot,
                image=self.image,
                grid_spacing=self.grid_spacing,
            )

            self.finished.emit(True, "Registration complete", result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None)


class AlignmentWidget(QWidget):
    """Widget for atlas alignment and overlay controls."""

    def __init__(self, parent_widget):
        """
        Initialize alignment widget.

        Args:
            parent_widget: Parent BrainSliceWidget with viewer and data
        """
        super().__init__()
        self.parent_widget = parent_widget
        self.viewer = parent_widget.viewer

        # Atlas state
        self.atlas_manager = None
        self.atlas_loaded = False
        self.current_atlas_position = 0
        self.current_orientation = 'coronal'

        # Transform state for manual alignment
        self.transform_offset_x = 0
        self.transform_offset_y = 0
        self.transform_rotation = 0.0
        self.transform_scale = 1.0

        # Layer references
        self.atlas_ref_layer = None
        self.atlas_labels_layer = None

        # Auto-registration state (legacy)
        self.auto_detect_worker = None
        self.auto_align_worker = None
        self.last_detection_result = None
        self.registration_transform = None

        # DeepSlice state
        self.deepslice_worker = None
        self.deepslice_predictions = None
        self.deepslice_wrapper = None

        # Elastix registration state
        self.elastix_worker = None
        self.warped_annotation = None
        self.deformation_field = None

        # Boundary overlay
        self.boundaries_layer = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Atlas Selection
        atlas_group = QGroupBox("Atlas")
        atlas_layout = QVBoxLayout()

        # Atlas type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Atlas:"))
        self.atlas_type_combo = QComboBox()
        self.atlas_type_combo.addItems([
            'Allen Mouse Brain (10um)',
            'Allen Mouse Brain (25um)',
        ])
        type_layout.addWidget(self.atlas_type_combo)
        atlas_layout.addLayout(type_layout)

        # Load atlas button
        self.load_atlas_btn = QPushButton("Load Atlas")
        self.load_atlas_btn.clicked.connect(self._load_atlas)
        atlas_layout.addWidget(self.load_atlas_btn)

        # Atlas status
        self.atlas_status_label = QLabel("No atlas loaded")
        self.atlas_status_label.setStyleSheet("color: gray;")
        atlas_layout.addWidget(self.atlas_status_label)

        atlas_group.setLayout(atlas_layout)
        layout.addWidget(atlas_group)

        # Orientation and Position
        position_group = QGroupBox("Atlas Position")
        position_layout = QVBoxLayout()

        # Orientation selector
        orient_layout = QHBoxLayout()
        orient_layout.addWidget(QLabel("Orientation:"))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(['Coronal', 'Sagittal', 'Horizontal'])
        self.orientation_combo.currentTextChanged.connect(self._on_orientation_changed)
        orient_layout.addWidget(self.orientation_combo)
        position_layout.addLayout(orient_layout)

        # Position slider
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Position:"))
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(50)
        self.position_slider.valueChanged.connect(self._on_position_changed)
        pos_layout.addWidget(self.position_slider)
        self.position_label = QLabel("50%")
        self.position_label.setMinimumWidth(40)
        pos_layout.addWidget(self.position_label)
        position_layout.addLayout(pos_layout)

        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        # Overlay Controls
        overlay_group = QGroupBox("Overlay")
        overlay_layout = QVBoxLayout()

        # Show/hide checkboxes
        self.show_reference_check = QCheckBox("Show atlas reference")
        self.show_reference_check.setChecked(True)
        self.show_reference_check.stateChanged.connect(self._update_overlay_visibility)
        overlay_layout.addWidget(self.show_reference_check)

        self.show_regions_check = QCheckBox("Show region labels (filled)")
        self.show_regions_check.setChecked(False)
        self.show_regions_check.stateChanged.connect(self._update_overlay_visibility)
        overlay_layout.addWidget(self.show_regions_check)

        self.show_boundaries_check = QCheckBox("Show region boundaries (outlines)")
        self.show_boundaries_check.setChecked(True)
        self.show_boundaries_check.stateChanged.connect(self._update_overlay_visibility)
        overlay_layout.addWidget(self.show_boundaries_check)

        # Opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("30%")
        self.opacity_label.setMinimumWidth(40)
        opacity_layout.addWidget(self.opacity_label)
        overlay_layout.addLayout(opacity_layout)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        # Manual Transform Controls
        transform_group = QGroupBox("Manual Alignment")
        transform_layout = QVBoxLayout()

        # X offset
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X offset:"))
        self.x_offset_spin = QSpinBox()
        self.x_offset_spin.setRange(-2000, 2000)
        self.x_offset_spin.setValue(0)
        self.x_offset_spin.valueChanged.connect(self._on_transform_changed)
        x_layout.addWidget(self.x_offset_spin)
        transform_layout.addLayout(x_layout)

        # Y offset
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y offset:"))
        self.y_offset_spin = QSpinBox()
        self.y_offset_spin.setRange(-2000, 2000)
        self.y_offset_spin.setValue(0)
        self.y_offset_spin.valueChanged.connect(self._on_transform_changed)
        y_layout.addWidget(self.y_offset_spin)
        transform_layout.addLayout(y_layout)

        # Rotation
        rot_layout = QHBoxLayout()
        rot_layout.addWidget(QLabel("Rotation (deg):"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setValue(0)
        self.rotation_spin.valueChanged.connect(self._on_transform_changed)
        rot_layout.addWidget(self.rotation_spin)
        transform_layout.addLayout(rot_layout)

        # Scale
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 5.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(1.0)
        self.scale_spin.valueChanged.connect(self._on_transform_changed)
        scale_layout.addWidget(self.scale_spin)
        transform_layout.addLayout(scale_layout)

        # Reset button
        self.reset_transform_btn = QPushButton("Reset Transform")
        self.reset_transform_btn.clicked.connect(self._reset_transform)
        transform_layout.addWidget(self.reset_transform_btn)

        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)

        # DeepSlice Position Detection (batch)
        deepslice_group = QGroupBox("DeepSlice Position Detection")
        deepslice_layout = QVBoxLayout()

        deepslice_layout.addWidget(QLabel("Detect atlas positions for all slices:"))

        # DeepSlice batch button
        self.deepslice_btn = QPushButton("Detect All Positions (DeepSlice)")
        self.deepslice_btn.clicked.connect(self._run_deepslice)
        self.deepslice_btn.setToolTip(
            "Run DeepSlice ML model on entire folder to detect\n"
            "atlas positions for all slices at once.\n"
            "Results are cached for fast subsequent use."
        )
        deepslice_layout.addWidget(self.deepslice_btn)

        # DeepSlice status
        self.deepslice_status_label = QLabel("No positions detected")
        self.deepslice_status_label.setStyleSheet("color: gray; font-size: 10px;")
        deepslice_layout.addWidget(self.deepslice_status_label)

        deepslice_group.setLayout(deepslice_layout)
        layout.addWidget(deepslice_group)

        # Automatic Registration with Elastix
        auto_group = QGroupBox("Registration (Current Slice)")
        auto_layout = QVBoxLayout()

        # Auto-detect position button (legacy fallback)
        self.auto_detect_btn = QPushButton("Detect Position (Single Slice)")
        self.auto_detect_btn.clicked.connect(self._auto_detect_position)
        self.auto_detect_btn.setToolTip(
            "Find atlas position for current slice only.\n"
            "For better results, use DeepSlice batch detection above."
        )
        auto_layout.addWidget(self.auto_detect_btn)

        # Progress bar for detection
        self.auto_progress = QProgressBar()
        self.auto_progress.setVisible(False)
        auto_layout.addWidget(self.auto_progress)

        # Detection result label
        self.auto_detect_label = QLabel("")
        self.auto_detect_label.setStyleSheet("color: gray; font-size: 10px;")
        auto_layout.addWidget(self.auto_detect_label)

        # Elastix registration button
        self.elastix_btn = QPushButton("Register Atlas (Elastix)")
        self.elastix_btn.clicked.connect(self._run_elastix_registration)
        self.elastix_btn.setEnabled(False)  # Enable after position detected
        self.elastix_btn.setToolTip(
            "Warp atlas to match current slice using\n"
            "affine + B-spline deformation (SimpleElastix)."
        )
        auto_layout.addWidget(self.elastix_btn)

        # Elastix status label
        self.elastix_status_label = QLabel("")
        self.elastix_status_label.setStyleSheet("color: gray; font-size: 10px;")
        auto_layout.addWidget(self.elastix_status_label)

        # Legacy auto-align button (hidden by default)
        self.auto_align_btn = QPushButton("Auto-Align (Legacy)")
        self.auto_align_btn.clicked.connect(self._auto_align)
        self.auto_align_btn.setEnabled(False)
        self.auto_align_btn.setVisible(False)  # Hidden by default
        auto_layout.addWidget(self.auto_align_btn)

        self.auto_align_label = QLabel("")
        self.auto_align_label.setStyleSheet("color: gray; font-size: 10px;")
        self.auto_align_label.setVisible(False)
        auto_layout.addWidget(self.auto_align_label)

        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)

        # ABBA Integration (fallback - collapsed by default)
        abba_group = QGroupBox("ABBA Export/Import (Advanced)")
        abba_group.setCheckable(True)
        abba_group.setChecked(False)  # Collapsed by default
        abba_layout = QVBoxLayout()

        abba_layout.addWidget(QLabel("For manual registration in Fiji:"))

        self.export_abba_btn = QPushButton("Export for ABBA...")
        self.export_abba_btn.clicked.connect(self._export_for_abba)
        abba_layout.addWidget(self.export_abba_btn)

        self.import_abba_btn = QPushButton("Import ABBA Registration...")
        self.import_abba_btn.clicked.connect(self._import_abba_registration)
        abba_layout.addWidget(self.import_abba_btn)

        self.abba_status_label = QLabel("")
        self.abba_status_label.setStyleSheet("color: gray; font-size: 10px;")
        abba_layout.addWidget(self.abba_status_label)

        abba_group.setLayout(abba_layout)
        layout.addWidget(abba_group)

        layout.addStretch()

    def _load_atlas(self):
        """Load the selected atlas."""
        try:
            from ..core.atlas_utils import DualAtlasManager

            atlas_text = self.atlas_type_combo.currentText()
            if '10um' in atlas_text:
                atlas_name = 'allen_mouse_10um'
            else:
                atlas_name = 'allen_mouse_25um'

            self.atlas_status_label.setText(f"Loading {atlas_name}...")
            self.atlas_status_label.setStyleSheet("color: blue;")

            # Load atlas
            self.atlas_manager = DualAtlasManager(brain_atlas=atlas_name)
            self.atlas_loaded = True

            # Get atlas info for slider range
            atlas_info = self.atlas_manager.get_atlas_info('brain')
            shape = atlas_info['shape']

            # Set slider range based on orientation
            self._update_slider_range()

            self.atlas_status_label.setText(f"Loaded: {atlas_name}")
            self.atlas_status_label.setStyleSheet("color: green;")

            # Show initial overlay
            self._update_atlas_overlay()

        except Exception as e:
            self.atlas_status_label.setText(f"Error: {e}")
            self.atlas_status_label.setStyleSheet("color: red;")
            import traceback
            traceback.print_exc()

    def _update_slider_range(self):
        """Update position slider range based on orientation."""
        if not self.atlas_loaded:
            return

        atlas = self.atlas_manager.brain_atlas
        orientation = self.orientation_combo.currentText().lower()

        if orientation == 'coronal':
            max_val = atlas.shape[0] - 1
        elif orientation == 'sagittal':
            max_val = atlas.shape[2] - 1
        else:  # horizontal
            max_val = atlas.shape[1] - 1

        self.position_slider.setMaximum(max_val)
        self.position_slider.setValue(max_val // 2)

    def _on_orientation_changed(self, text):
        """Handle orientation change."""
        self.current_orientation = text.lower()
        self._update_slider_range()
        self._update_atlas_overlay()

    def _on_position_changed(self, value):
        """Handle position slider change."""
        self.current_atlas_position = value
        max_val = self.position_slider.maximum()
        pct = int(100 * value / max_val) if max_val > 0 else 0
        self.position_label.setText(f"{pct}%")
        self._update_atlas_overlay()

    def _on_opacity_changed(self, value):
        """Handle opacity slider change."""
        self.opacity_label.setText(f"{value}%")
        opacity = value / 100.0

        if self.atlas_ref_layer is not None:
            self.atlas_ref_layer.opacity = opacity
        if self.atlas_labels_layer is not None:
            self.atlas_labels_layer.opacity = opacity * 0.7  # Labels slightly more transparent

    def _update_overlay_visibility(self):
        """Update visibility of overlay layers."""
        if self.atlas_ref_layer is not None:
            self.atlas_ref_layer.visible = self.show_reference_check.isChecked()
        if self.atlas_labels_layer is not None:
            self.atlas_labels_layer.visible = self.show_regions_check.isChecked()
        if self.boundaries_layer is not None:
            self.boundaries_layer.visible = self.show_boundaries_check.isChecked()

    def _on_transform_changed(self):
        """Handle manual transform parameter change."""
        self.transform_offset_x = self.x_offset_spin.value()
        self.transform_offset_y = self.y_offset_spin.value()
        self.transform_rotation = self.rotation_spin.value()
        self.transform_scale = self.scale_spin.value()

        self._update_atlas_overlay()

    def _reset_transform(self):
        """Reset transform to defaults."""
        self.x_offset_spin.setValue(0)
        self.y_offset_spin.setValue(0)
        self.rotation_spin.setValue(0)
        self.scale_spin.setValue(1.0)

    # =========================================================================
    # AUTOMATIC REGISTRATION
    # =========================================================================

    def _get_current_image(self) -> Optional[np.ndarray]:
        """Get current image for registration (handles stacks)."""
        red = self.parent_widget.red_channel
        if red is None:
            return None

        if red.ndim == 3:
            # Stack - use current slice
            current_slice = self.viewer.dims.current_step[0]
            return red[current_slice]
        else:
            return red

    def _auto_detect_position(self):
        """Automatically detect best matching atlas position."""
        if not self.atlas_loaded:
            QMessageBox.warning(self, "No Atlas", "Load an atlas first")
            return

        image = self._get_current_image()
        if image is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        # Update UI
        self.auto_detect_btn.setEnabled(False)
        self.auto_progress.setVisible(True)
        self.auto_progress.setValue(0)
        self.auto_detect_label.setText("Scanning atlas slices...")
        self.auto_detect_label.setStyleSheet("color: blue; font-size: 10px;")

        # Start worker
        orientation = self.orientation_combo.currentText().lower()
        self.auto_detect_worker = AutoDetectWorker(
            image=image,
            atlas_manager=self.atlas_manager,
            orientation=orientation,
        )
        self.auto_detect_worker.progress.connect(self._on_detect_progress)
        self.auto_detect_worker.finished.connect(self._on_detect_finished)
        self.auto_detect_worker.start()

    def _on_detect_progress(self, current: int, total: int):
        """Handle detection progress."""
        self.auto_progress.setMaximum(total)
        self.auto_progress.setValue(current)

    def _on_detect_finished(self, success: bool, message: str, result):
        """Handle detection completion."""
        self.auto_detect_btn.setEnabled(True)
        self.auto_progress.setVisible(False)

        if success and result is not None:
            self.last_detection_result = result
            best_idx = result['best_position_idx']
            best_score = result['best_score']
            n_slices = result['n_slices']

            # Update position slider to detected position
            self.position_slider.setValue(best_idx)

            # Update label with result
            pct = int(100 * best_idx / n_slices) if n_slices > 0 else 0
            self.auto_detect_label.setText(
                f"Position: {best_idx}/{n_slices} ({pct}%)  Score: {best_score:.3f}"
            )
            self.auto_detect_label.setStyleSheet("color: green; font-size: 10px;")

            # Enable auto-align button
            self.auto_align_btn.setEnabled(True)

            # Update overlay
            self._update_atlas_overlay()

        else:
            self.auto_detect_label.setText(f"Error: {message}")
            self.auto_detect_label.setStyleSheet("color: red; font-size: 10px;")

    def _auto_align(self):
        """Automatically align image to atlas using registration."""
        if self.last_detection_result is None:
            QMessageBox.warning(
                self, "No Detection",
                "Run Auto-Detect Position first"
            )
            return

        image = self._get_current_image()
        if image is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        # Get atlas slice at detected position
        atlas_slice = self.last_detection_result['reference_slice']

        # Update UI
        self.auto_align_btn.setEnabled(False)
        self.auto_align_label.setText("Computing alignment...")
        self.auto_align_label.setStyleSheet("color: blue; font-size: 10px;")

        # Start worker
        self.auto_align_worker = AutoAlignWorker(
            image=image,
            atlas_slice=atlas_slice,
            method='similarity',
        )
        self.auto_align_worker.finished.connect(self._on_align_finished)
        self.auto_align_worker.start()

    def _on_align_finished(self, success: bool, message: str, result):
        """Handle alignment completion."""
        self.auto_align_btn.setEnabled(True)

        if success and result is not None:
            self.registration_transform = result.get('transform')

            # Extract transform parameters and update UI
            translation = result.get('translation', (0, 0))
            rotation = result.get('rotation', 0)
            scale = result.get('scale', 1.0)
            method_used = result.get('method_used', 'unknown')

            # Update manual controls with computed values
            self.x_offset_spin.setValue(int(translation[1]))  # dx
            self.y_offset_spin.setValue(int(translation[0]))  # dy
            self.rotation_spin.setValue(rotation)
            self.scale_spin.setValue(scale)

            # Update label
            self.auto_align_label.setText(
                f"Aligned ({method_used}): "
                f"shift=({translation[0]:.0f}, {translation[1]:.0f}), "
                f"rot={rotation:.1f}°, scale={scale:.2f}"
            )
            self.auto_align_label.setStyleSheet("color: green; font-size: 10px;")

            # Update overlay with new transform
            self._update_atlas_overlay()

        else:
            self.auto_align_label.setText(f"Error: {message}")
            self.auto_align_label.setStyleSheet("color: red; font-size: 10px;")

    # =========================================================================
    # DEEPSLICE BATCH DETECTION
    # =========================================================================

    def _run_deepslice(self):
        """Run DeepSlice batch position detection on folder."""
        # Get folder path from parent widget
        folder_path = getattr(self.parent_widget, 'current_folder', None)
        if folder_path is None:
            QMessageBox.warning(
                self, "No Folder",
                "Load a folder of images first, then run DeepSlice."
            )
            return

        # Update UI
        self.deepslice_btn.setEnabled(False)
        self.deepslice_status_label.setText("Running DeepSlice...")
        self.deepslice_status_label.setStyleSheet("color: blue; font-size: 10px;")

        # Start worker
        self.deepslice_worker = DeepSliceWorker(folder_path, use_cache=True)
        self.deepslice_worker.progress.connect(self._on_deepslice_progress)
        self.deepslice_worker.finished.connect(self._on_deepslice_finished)
        self.deepslice_worker.start()

    def _on_deepslice_progress(self, message: str):
        """Handle DeepSlice progress update."""
        self.deepslice_status_label.setText(message)

    def _on_deepslice_finished(self, success: bool, message: str, predictions):
        """Handle DeepSlice completion."""
        self.deepslice_btn.setEnabled(True)

        if success and predictions is not None:
            self.deepslice_predictions = predictions

            # Create wrapper for position lookups
            from ..core.deepslice_wrapper import DeepSliceWrapper
            self.deepslice_wrapper = DeepSliceWrapper(species='mouse')
            self.deepslice_wrapper._predictions_by_filename = {
                s['filename']: s for s in predictions.get('slices', [])
            }

            self.deepslice_status_label.setText(
                f"Detected: {predictions['n_slices']} slices"
            )
            self.deepslice_status_label.setStyleSheet("color: green; font-size: 10px;")

            # Enable registration button
            self.elastix_btn.setEnabled(True)

            # Update position for current slice
            self._update_position_from_deepslice()

        else:
            self.deepslice_status_label.setText(f"Error: {message}")
            self.deepslice_status_label.setStyleSheet("color: red; font-size: 10px;")

    def _update_position_from_deepslice(self):
        """Update atlas position from DeepSlice for current slice."""
        if self.deepslice_wrapper is None or not self.atlas_loaded:
            return

        # Get current slice filename
        current_idx = 0
        if hasattr(self.viewer, 'dims'):
            current_idx = self.viewer.dims.current_step[0]

        # Get filename for current slice
        filenames = getattr(self.parent_widget, 'loaded_filenames', None)
        if filenames and current_idx < len(filenames):
            filename = filenames[current_idx]
        else:
            # Try to construct filename
            filename = f"slice_s{current_idx+1:03d}"

        # Get atlas resolution
        atlas_text = self.atlas_type_combo.currentText()
        resolution = 10 if '10um' in atlas_text else 25

        # Get position from DeepSlice
        position = self.deepslice_wrapper.get_position_for_slice(filename, resolution)
        if position is not None:
            self.position_slider.setValue(position)
            self.auto_detect_label.setText(f"Position from DeepSlice: {position}")
            self.auto_detect_label.setStyleSheet("color: green; font-size: 10px;")

    # =========================================================================
    # ELASTIX REGISTRATION
    # =========================================================================

    def _run_elastix_registration(self):
        """Run SimpleElastix registration on current slice."""
        if not self.atlas_loaded:
            QMessageBox.warning(self, "No Atlas", "Load an atlas first")
            return

        image = self._get_current_image()
        if image is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        # Get atlas slices at current position
        atlas_ref = self.atlas_manager.get_reference_slice(
            'brain',
            position_idx=self.current_atlas_position,
            orientation=self.current_orientation,
        )
        atlas_annot = self.atlas_manager.get_annotation_slice(
            'brain',
            position_idx=self.current_atlas_position,
            orientation=self.current_orientation,
        )

        # Update UI
        self.elastix_btn.setEnabled(False)
        self.elastix_status_label.setText("Running registration...")
        self.elastix_status_label.setStyleSheet("color: blue; font-size: 10px;")

        # Start worker
        self.elastix_worker = ElastixRegistrationWorker(
            image=image,
            atlas_ref=atlas_ref,
            atlas_annot=atlas_annot,
            grid_spacing=8,
        )
        self.elastix_worker.finished.connect(self._on_elastix_finished)
        self.elastix_worker.start()

    def _on_elastix_finished(self, success: bool, message: str, result):
        """Handle Elastix registration completion."""
        self.elastix_btn.setEnabled(True)

        if success and result is not None:
            self.warped_annotation = result.get('warped_annotation')
            self.deformation_field = result.get('deformation_field')

            # Get affine params for display
            affine = result.get('affine_params', {})
            translation = affine.get('translation', (0, 0))
            rotation = affine.get('rotation_deg', 0)
            scale = affine.get('scale', 1.0)

            self.elastix_status_label.setText(
                f"Registered: shift=({translation[0]:.0f}, {translation[1]:.0f}), "
                f"rot={rotation:.1f}°, scale={scale:.2f}"
            )
            self.elastix_status_label.setStyleSheet("color: green; font-size: 10px;")

            # Update overlay with warped atlas
            self._update_atlas_overlay_with_warp(result)

        else:
            self.elastix_status_label.setText(f"Error: {message}")
            self.elastix_status_label.setStyleSheet("color: red; font-size: 10px;")

    def _update_atlas_overlay_with_warp(self, registration_result):
        """Update atlas overlay using warped registration result."""
        try:
            warped_ref = registration_result.get('warped_reference')
            warped_annot = registration_result.get('warped_annotation')
            deform_field = registration_result.get('deformation_field')

            opacity = self.opacity_slider.value() / 100.0

            # Remove old layers
            for layer in [self.atlas_ref_layer, self.atlas_labels_layer, self.boundaries_layer]:
                if layer is not None:
                    try:
                        self.viewer.layers.remove(layer)
                    except ValueError:
                        pass

            # Add warped reference
            if warped_ref is not None:
                self.atlas_ref_layer = self.viewer.add_image(
                    warped_ref,
                    name="Atlas Reference (Warped)",
                    colormap='gray',
                    opacity=opacity,
                    blending='additive',
                    visible=self.show_reference_check.isChecked(),
                )

            # Add warped labels
            if warped_annot is not None:
                self.atlas_labels_layer = self.viewer.add_labels(
                    warped_annot.astype(np.int32),
                    name="Atlas Regions (Warped)",
                    opacity=opacity * 0.5,
                    visible=self.show_regions_check.isChecked(),
                )

            # Add boundary outlines
            if warped_annot is not None and self.show_boundaries_check.isChecked():
                self._add_boundary_shapes(warped_annot, deform_field)

        except Exception as e:
            print(f"[AlignmentWidget] Error updating warped overlay: {e}")
            import traceback
            traceback.print_exc()

    def _add_boundary_shapes(self, annotation, deformation_field=None):
        """Add boundary outlines as Shapes layer."""
        try:
            from ..core.boundaries import extract_region_boundaries, boundaries_to_napari_shapes

            # Extract boundaries (use major regions to reduce clutter)
            boundaries = extract_region_boundaries(
                annotation,
                deformation_field=None,  # Already warped
                simplify_tolerance=2.0,
            )

            if not boundaries:
                return

            # Convert to napari format
            shapes_data, layer_kwargs = boundaries_to_napari_shapes(
                boundaries,
                self.atlas_manager,
                shape_type='path',
            )

            if shapes_data:
                # Remove old boundaries layer
                if self.boundaries_layer is not None:
                    try:
                        self.viewer.layers.remove(self.boundaries_layer)
                    except ValueError:
                        pass

                self.boundaries_layer = self.viewer.add_shapes(
                    shapes_data,
                    shape_type='path',
                    edge_color=layer_kwargs.get('edge_color', 'white'),
                    edge_width=1.5,
                    name="Atlas Boundaries",
                    visible=self.show_boundaries_check.isChecked(),
                )

        except Exception as e:
            print(f"[AlignmentWidget] Error adding boundaries: {e}")
            import traceback
            traceback.print_exc()

    def _update_atlas_overlay(self):
        """Update the atlas overlay in napari."""
        if not self.atlas_loaded:
            return

        try:
            # Get atlas slices at current position
            ref_slice = self.atlas_manager.get_reference_slice(
                'brain',
                position_idx=self.current_atlas_position,
                orientation=self.current_orientation,
            )
            annot_slice = self.atlas_manager.get_annotation_slice(
                'brain',
                position_idx=self.current_atlas_position,
                orientation=self.current_orientation,
            )

            # Apply transform (scale, rotate)
            if self.transform_scale != 1.0 or self.transform_rotation != 0:
                from scipy.ndimage import zoom, rotate
                if self.transform_scale != 1.0:
                    ref_slice = zoom(ref_slice, self.transform_scale, order=1)
                    annot_slice = zoom(annot_slice, self.transform_scale, order=0)
                if self.transform_rotation != 0:
                    ref_slice = rotate(ref_slice, self.transform_rotation, reshape=False, order=1)
                    annot_slice = rotate(annot_slice, self.transform_rotation, reshape=False, order=0)

            # Calculate translation for napari
            translate = [self.transform_offset_y, self.transform_offset_x]

            opacity = self.opacity_slider.value() / 100.0

            # Remove old layers if they exist
            if self.atlas_ref_layer is not None:
                try:
                    self.viewer.layers.remove(self.atlas_ref_layer)
                except ValueError:
                    pass
            if self.atlas_labels_layer is not None:
                try:
                    self.viewer.layers.remove(self.atlas_labels_layer)
                except ValueError:
                    pass

            # Add reference image
            self.atlas_ref_layer = self.viewer.add_image(
                ref_slice,
                name="Atlas Reference",
                colormap='gray',
                opacity=opacity,
                blending='additive',
                translate=translate,
                visible=self.show_reference_check.isChecked(),
            )

            # Add region labels
            self.atlas_labels_layer = self.viewer.add_labels(
                annot_slice,
                name="Atlas Regions",
                opacity=opacity * 0.7,
                translate=translate,
                visible=self.show_regions_check.isChecked(),
            )

        except Exception as e:
            print(f"[AlignmentWidget] Error updating overlay: {e}")
            import traceback
            traceback.print_exc()

    def _export_for_abba(self):
        """Export current image for ABBA registration."""
        if self.parent_widget.red_channel is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export for ABBA",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )

        if not file_path:
            return

        try:
            from ..core.io import save_tiff

            # Get current slice if stack, otherwise full image
            red = self.parent_widget.red_channel
            green = self.parent_widget.green_channel

            if red.ndim == 3:
                # Stack - export current slice
                current_slice = self.viewer.dims.current_step[0]
                red_slice = red[current_slice]
                green_slice = green[current_slice]
            else:
                red_slice = red
                green_slice = green

            # Combine channels for ABBA (C, Y, X format)
            combined = np.stack([green_slice, red_slice], axis=0)

            save_tiff(combined, file_path)

            self.abba_status_label.setText(f"Exported: {Path(file_path).name}")
            self.abba_status_label.setStyleSheet("color: green; font-size: 10px;")

            QMessageBox.information(
                self,
                "Export Complete",
                f"Image exported to:\n{file_path}\n\n"
                "Open this in ABBA (Fiji plugin) to perform registration.\n"
                "Then export the registration and import it back here."
            )

        except Exception as e:
            self.abba_status_label.setText(f"Error: {e}")
            self.abba_status_label.setStyleSheet("color: red; font-size: 10px;")
            QMessageBox.warning(self, "Export Error", str(e))

    def _import_abba_registration(self):
        """Import registration from ABBA."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import ABBA Registration",
            "",
            "ABBA Files (*.json *.xml *.state);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Parse ABBA registration file
            # ABBA exports in various formats - we'll support the common ones
            import json

            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    reg_data = json.load(f)
                else:
                    # For XML or other formats, show message
                    QMessageBox.information(
                        self,
                        "Format Note",
                        "Currently supporting JSON format.\n"
                        "Please export from ABBA as JSON if possible."
                    )
                    return

            # Extract transform parameters from ABBA format
            # This depends on ABBA's export format
            if 'slices' in reg_data:
                # QuPath/ABBA format
                slice_data = reg_data['slices'][0] if reg_data['slices'] else {}
                transform = slice_data.get('transform', {})

                # Apply transform
                self.x_offset_spin.setValue(int(transform.get('translateX', 0)))
                self.y_offset_spin.setValue(int(transform.get('translateY', 0)))
                self.rotation_spin.setValue(float(transform.get('rotation', 0)))
                self.scale_spin.setValue(float(transform.get('scale', 1.0)))

            self.abba_status_label.setText(f"Imported: {Path(file_path).name}")
            self.abba_status_label.setStyleSheet("color: green; font-size: 10px;")

        except Exception as e:
            self.abba_status_label.setText(f"Error: {e}")
            self.abba_status_label.setStyleSheet("color: red; font-size: 10px;")
            QMessageBox.warning(self, "Import Error", str(e))
            import traceback
            traceback.print_exc()

    def get_current_registration(self) -> Dict[str, Any]:
        """Get current registration parameters."""
        return {
            'orientation': self.current_orientation,
            'atlas_position': self.current_atlas_position,
            'transform': {
                'offset_x': self.transform_offset_x,
                'offset_y': self.transform_offset_y,
                'rotation': self.transform_rotation,
                'scale': self.transform_scale,
            }
        }

    def get_registered_atlas_labels(self) -> Optional[np.ndarray]:
        """Get atlas labels transformed to image coordinates."""
        if not self.atlas_loaded or self.parent_widget.red_channel is None:
            return None

        try:
            # Get annotation slice
            annot = self.atlas_manager.get_annotation_slice(
                'brain',
                position_idx=self.current_atlas_position,
                orientation=self.current_orientation,
            )

            # Apply transforms
            from scipy.ndimage import zoom, rotate, shift

            if self.transform_scale != 1.0:
                annot = zoom(annot, self.transform_scale, order=0)
            if self.transform_rotation != 0:
                annot = rotate(annot, self.transform_rotation, reshape=False, order=0)

            # Resize/pad to match image size
            target = self.parent_widget.red_channel
            if target.ndim == 3:
                target_shape = target.shape[1:]  # (Y, X)
            else:
                target_shape = target.shape

            # Create output array
            result = np.zeros(target_shape, dtype=annot.dtype)

            # Calculate paste position with offset
            offset_y = self.transform_offset_y
            offset_x = self.transform_offset_x

            # Determine overlap region
            src_y_start = max(0, -offset_y)
            src_x_start = max(0, -offset_x)
            dst_y_start = max(0, offset_y)
            dst_x_start = max(0, offset_x)

            src_y_end = min(annot.shape[0], target_shape[0] - offset_y)
            src_x_end = min(annot.shape[1], target_shape[1] - offset_x)
            dst_y_end = min(target_shape[0], annot.shape[0] + offset_y)
            dst_x_end = min(target_shape[1], annot.shape[1] + offset_x)

            # Ensure valid ranges
            if src_y_end > src_y_start and src_x_end > src_x_start:
                h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
                w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
                result[dst_y_start:dst_y_start+h, dst_x_start:dst_x_start+w] = \
                    annot[src_y_start:src_y_start+h, src_x_start:src_x_start+w]

            return result

        except Exception as e:
            print(f"[AlignmentWidget] Error getting registered labels: {e}")
            import traceback
            traceback.print_exc()
            return None
