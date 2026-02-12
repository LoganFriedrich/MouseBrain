"""
Curation Widget for rapid cell candidate review.

Features:
- Isolation view: Focus on one candidate at a time
- Keyboard shortcuts for rapid Y/N decisions
- Z-context mini viewer
- Progress tracking
- Integration with session documenter
"""

import sys
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QProgressBar, QSlider, QCheckBox,
    QScrollArea, QShortcut
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont, QKeySequence
import numpy as np

# Import paths from central config
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))


class CurationWidget(QWidget):
    """
    Widget for curating cell detection candidates.

    Provides an isolation view for reviewing candidates one at a time,
    with keyboard shortcuts for rapid classification.
    """

    # Signals
    candidate_confirmed = Signal(int)  # index of confirmed candidate
    candidate_rejected = Signal(int)   # index of rejected candidate
    curation_complete = Signal(int, int, int)  # confirmed, rejected, skipped

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.candidates = None  # points array
        self.candidates_layer = None  # napari layer
        self.current_index = 0
        self.review_indices = None  # indices to review (for sampling)
        self.review_position = 0  # position within review_indices
        self.confirmed = set()
        self.rejected = set()
        self.skipped = set()
        self.review_mode = False

        # Settings
        self.z_context = 5  # slices above/below
        self.zoom_level = 100  # percentage

        self.setup_ui()
        self.setup_shortcuts()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Title
        title = QLabel("Cell Curation")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Rapid review of detection candidates")
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)

        # Candidates source
        source_group = QGroupBox("1. Select Candidates")
        source_layout = QVBoxLayout()
        source_group.setLayout(source_layout)

        source_layout.addWidget(QLabel(
            "Select a points layer containing detection candidates:"
        ))

        layer_layout = QHBoxLayout()
        self.layer_label = QLabel("No layer selected")
        self.layer_label.setStyleSheet("color: gray; font-style: italic;")
        layer_layout.addWidget(self.layer_label, stretch=1)

        use_selected_btn = QPushButton("Use Selected Layer")
        use_selected_btn.clicked.connect(self.use_selected_layer)
        layer_layout.addWidget(use_selected_btn)

        source_layout.addLayout(layer_layout)

        # Sampling option for QC
        sample_layout = QHBoxLayout()
        self.sample_check = QCheckBox("Sample subset for QC:")
        self.sample_check.setToolTip("Review only a random percentage of candidates")
        sample_layout.addWidget(self.sample_check)
        self.sample_spin = QSpinBox()
        self.sample_spin.setMinimum(1)
        self.sample_spin.setMaximum(100)
        self.sample_spin.setValue(1)  # Default 1%
        self.sample_spin.setSuffix("%")
        sample_layout.addWidget(self.sample_spin)
        sample_layout.addStretch()
        source_layout.addLayout(sample_layout)

        layout.addWidget(source_group)

        # Progress section
        progress_group = QGroupBox("2. Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        stats_layout = QHBoxLayout()

        self.total_label = QLabel("Total: 0")
        stats_layout.addWidget(self.total_label)

        self.confirmed_label = QLabel("Confirmed: 0")
        self.confirmed_label.setStyleSheet("color: green;")
        stats_layout.addWidget(self.confirmed_label)

        self.rejected_label = QLabel("Rejected: 0")
        self.rejected_label.setStyleSheet("color: red;")
        stats_layout.addWidget(self.rejected_label)

        self.skipped_label = QLabel("Skipped: 0")
        self.skipped_label.setStyleSheet("color: orange;")
        stats_layout.addWidget(self.skipped_label)

        progress_layout.addLayout(stats_layout)
        layout.addWidget(progress_group)

        # Settings
        settings_group = QGroupBox("3. View Settings")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)

        # Z context
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z Context (slices):"))
        self.z_context_spin = QSpinBox()
        self.z_context_spin.setMinimum(1)
        self.z_context_spin.setMaximum(20)
        self.z_context_spin.setValue(5)
        self.z_context_spin.valueChanged.connect(self.on_z_context_changed)
        z_layout.addWidget(self.z_context_spin)
        settings_layout.addLayout(z_layout)

        # Zoom
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)
        settings_layout.addLayout(zoom_layout)

        # Hide other layers
        self.hide_others_check = QCheckBox("Hide other points layers during review")
        self.hide_others_check.setChecked(True)
        settings_layout.addWidget(self.hide_others_check)

        # Isolation mode - only show current point
        self.isolate_check = QCheckBox("Isolate current point (hide all others)")
        self.isolate_check.setChecked(True)
        self.isolate_check.setToolTip("Only show the current candidate being reviewed")
        settings_layout.addWidget(self.isolate_check)

        layout.addWidget(settings_group)

        # Review controls
        review_group = QGroupBox("4. Review Mode")
        review_layout = QVBoxLayout()
        review_group.setLayout(review_layout)

        review_layout.addWidget(QLabel(
            "Keyboard shortcuts:\n"
            "  Y/C = Confirm cell\n"
            "  N/X = Reject candidate\n"
            "  S/Space = Skip\n"
            "  ← → = Navigate without marking\n"
            "  Esc = Exit review mode"
        ))

        self.start_review_btn = QPushButton("Start Review Mode")
        self.start_review_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;"
        )
        self.start_review_btn.clicked.connect(self.toggle_review_mode)
        review_layout.addWidget(self.start_review_btn)

        # Current candidate display
        self.current_info = QLabel("Current: -")
        self.current_info.setFont(QFont("Arial", 12, QFont.Bold))
        self.current_info.setAlignment(Qt.AlignCenter)
        review_layout.addWidget(self.current_info)

        # Manual navigation
        nav_layout = QHBoxLayout()

        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(self.go_previous)
        nav_layout.addWidget(prev_btn)

        confirm_btn = QPushButton("Confirm (Y)")
        confirm_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        confirm_btn.clicked.connect(self.confirm_current)
        nav_layout.addWidget(confirm_btn)

        reject_btn = QPushButton("Reject (N)")
        reject_btn.setStyleSheet("background-color: #f44336; color: white;")
        reject_btn.clicked.connect(self.reject_current)
        nav_layout.addWidget(reject_btn)

        skip_btn = QPushButton("Skip (S)")
        skip_btn.clicked.connect(self.skip_current)
        nav_layout.addWidget(skip_btn)

        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self.go_next)
        nav_layout.addWidget(next_btn)

        review_layout.addLayout(nav_layout)
        layout.addWidget(review_group)

        # Results
        results_group = QGroupBox("5. Save Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        save_layout = QHBoxLayout()

        save_confirmed_btn = QPushButton("Save Confirmed Cells")
        save_confirmed_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        save_confirmed_btn.clicked.connect(self.save_confirmed)
        save_layout.addWidget(save_confirmed_btn)

        save_rejected_btn = QPushButton("Save Rejected")
        save_rejected_btn.setStyleSheet("background-color: #f44336; color: white;")
        save_rejected_btn.clicked.connect(self.save_rejected)
        save_layout.addWidget(save_rejected_btn)

        results_layout.addLayout(save_layout)
        layout.addWidget(results_group)

        layout.addStretch()

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Confirm shortcuts
        QShortcut(QKeySequence("Y"), self).activated.connect(self.confirm_current)
        QShortcut(QKeySequence("C"), self).activated.connect(self.confirm_current)

        # Reject shortcuts
        QShortcut(QKeySequence("N"), self).activated.connect(self.reject_current)
        QShortcut(QKeySequence("X"), self).activated.connect(self.reject_current)

        # Skip shortcuts
        QShortcut(QKeySequence("S"), self).activated.connect(self.skip_current)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.skip_current)

        # Navigation
        QShortcut(QKeySequence("Left"), self).activated.connect(self.go_previous)
        QShortcut(QKeySequence("Right"), self).activated.connect(self.go_next)

        # Exit
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.exit_review_mode)

    def use_selected_layer(self):
        """Use the currently selected layer as candidates source."""
        selected = self.viewer.layers.selection.active

        if selected is None:
            QMessageBox.warning(self, "Error", "No layer selected")
            return

        if selected._type_string != 'points':
            QMessageBox.warning(self, "Error", "Selected layer must be a Points layer")
            return

        self.candidates_layer = selected
        self.candidates = selected.data.copy()
        self.current_index = 0
        self.confirmed.clear()
        self.rejected.clear()
        self.skipped.clear()

        self.layer_label.setText(f"Layer: {selected.name} ({len(self.candidates)} candidates)")
        self.layer_label.setStyleSheet("color: #2196F3; font-weight: bold;")

        self.progress_bar.setMaximum(len(self.candidates))
        self.total_label.setText(f"Total: {len(self.candidates)}")
        self.update_stats()

    def toggle_review_mode(self):
        """Toggle review mode on/off."""
        if not self.review_mode:
            self.start_review_mode()
        else:
            self.exit_review_mode()

    def start_review_mode(self):
        """Enter review mode."""
        if self.candidates is None or len(self.candidates) == 0:
            QMessageBox.warning(self, "Error", "No candidates loaded. Select a points layer first.")
            return

        self.review_mode = True
        self.start_review_btn.setText("Exit Review Mode")
        self.start_review_btn.setStyleSheet(
            "background-color: #f44336; color: white; padding: 10px; font-size: 14px;"
        )

        # Set up review indices (with optional sampling)
        if self.sample_check.isChecked():
            sample_pct = self.sample_spin.value() / 100
            n_sample = max(1, int(len(self.candidates) * sample_pct))
            self.review_indices = np.random.choice(
                len(self.candidates), size=n_sample, replace=False
            )
            self.review_indices = np.sort(self.review_indices)  # Keep in order
            print(f"[Connectome] Sampling {n_sample} candidates ({self.sample_spin.value()}%) for QC review")
        else:
            self.review_indices = np.arange(len(self.candidates))

        self.review_position = 0
        self.progress_bar.setMaximum(len(self.review_indices))

        # Hide other POINTS layers if requested (keep image layers visible!)
        if self.hide_others_check.isChecked():
            for layer in self.viewer.layers:
                if layer != self.candidates_layer:
                    # Only hide other points layers, keep images visible for review
                    if layer._type_string == 'points':
                        layer.visible = False

        # Focus on first unreviewed candidate
        self.go_to_next_unreviewed()

    def exit_review_mode(self):
        """Exit review mode."""
        self.review_mode = False
        self.start_review_btn.setText("Start Review Mode")
        self.start_review_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;"
        )

        # Show all layers again
        for layer in self.viewer.layers:
            layer.visible = True

        # Emit completion signal
        self.curation_complete.emit(
            len(self.confirmed),
            len(self.rejected),
            len(self.skipped)
        )

    def go_to_next_unreviewed(self):
        """Go to the next unreviewed candidate in the review set."""
        if self.review_indices is None:
            return

        # Search from current position forward
        for pos in range(self.review_position, len(self.review_indices)):
            idx = self.review_indices[pos]
            if idx not in self.confirmed and idx not in self.rejected and idx not in self.skipped:
                self.review_position = pos
                self.current_index = idx
                self.focus_on_current()
                return

        # Wrap around from beginning
        for pos in range(0, self.review_position):
            idx = self.review_indices[pos]
            if idx not in self.confirmed and idx not in self.rejected and idx not in self.skipped:
                self.review_position = pos
                self.current_index = idx
                self.focus_on_current()
                return

        # All reviewed
        reviewed_count = len([i for i in self.review_indices if i in self.confirmed or i in self.rejected or i in self.skipped])
        QMessageBox.information(
            self, "Complete",
            f"All {reviewed_count} sampled candidates have been reviewed!\n\n"
            f"Confirmed: {len(self.confirmed)}\n"
            f"Rejected: {len(self.rejected)}\n"
            f"Skipped: {len(self.skipped)}"
        )
        self.exit_review_mode()

    def focus_on_current(self):
        """Focus the view on the current candidate."""
        if self.candidates is None or self.current_index >= len(self.candidates):
            return

        pos = self.candidates[self.current_index]

        # Update the viewer position
        # Assuming ZYX ordering
        if len(pos) >= 3:
            z, y, x = pos[0], pos[1], pos[2]

            # Set the Z position (dims slider)
            if hasattr(self.viewer, 'dims'):
                self.viewer.dims.set_point(0, z)

            # Center the camera on the point
            self.viewer.camera.center = (0, y, x)

            # Apply zoom
            zoom_factor = self.zoom_slider.value() / 100
            self.viewer.camera.zoom = zoom_factor

        self.update_current_info()
        self.update_candidate_colors()

    def update_current_info(self):
        """Update the current candidate info display."""
        if self.candidates is None:
            self.current_info.setText("Current: -")
            return

        # Use review set size if sampling
        if self.review_indices is not None:
            total = len(self.review_indices)
            reviewed = len([i for i in self.review_indices
                           if i in self.confirmed or i in self.rejected or i in self.skipped])
            position = self.review_position + 1
        else:
            total = len(self.candidates)
            reviewed = len(self.confirmed) + len(self.rejected) + len(self.skipped)
            position = self.current_index + 1

        status = ""
        if self.current_index in self.confirmed:
            status = " [CONFIRMED]"
        elif self.current_index in self.rejected:
            status = " [REJECTED]"
        elif self.current_index in self.skipped:
            status = " [SKIPPED]"

        self.current_info.setText(
            f"Current: {position} / {total}{status}"
        )
        self.progress_bar.setValue(reviewed)

    def update_candidate_colors(self):
        """Update the colors of candidates in the layer."""
        if self.candidates_layer is None or self.candidates is None:
            return

        isolate = self.isolate_check.isChecked() if hasattr(self, 'isolate_check') else True

        face_colors = []
        edge_colors = []
        sizes = []

        for i in range(len(self.candidates)):
            if i == self.current_index:
                # Current: bright visible ring (transparent face, yellow edge)
                face_colors.append([1, 1, 0, 0.0])  # Transparent face
                edge_colors.append([1, 1, 0, 1.0])  # Bright yellow edge
                sizes.append(20)  # Larger for visibility
            elif isolate:
                # Isolation mode: hide all other points completely
                face_colors.append([0, 0, 0, 0.0])
                edge_colors.append([0, 0, 0, 0.0])
                sizes.append(0)
            elif i in self.confirmed:
                face_colors.append([0, 1, 0, 0.0])  # Transparent
                edge_colors.append([0, 1, 0, 0.3])  # Dim green edge
                sizes.append(10)
            elif i in self.rejected:
                face_colors.append([1, 0, 0, 0.0])  # Transparent
                edge_colors.append([1, 0, 0, 0.3])  # Dim red edge
                sizes.append(10)
            elif i in self.skipped:
                face_colors.append([1, 0.5, 0, 0.0])  # Transparent
                edge_colors.append([1, 0.5, 0, 0.3])  # Dim orange edge
                sizes.append(10)
            else:
                # Unreviewed: very dim
                face_colors.append([0, 0.7, 1, 0.0])  # Transparent
                edge_colors.append([0, 0.7, 1, 0.1])  # Nearly invisible
                sizes.append(8)

        self.candidates_layer.face_color = face_colors
        self.candidates_layer.border_color = edge_colors  # Updated from deprecated edge_color
        self.candidates_layer.size = sizes
        self.candidates_layer.border_width = 0.1  # Updated from deprecated edge_width
        self.candidates_layer.refresh()

    def update_stats(self):
        """Update the statistics labels."""
        self.confirmed_label.setText(f"Confirmed: {len(self.confirmed)}")
        self.rejected_label.setText(f"Rejected: {len(self.rejected)}")
        self.skipped_label.setText(f"Skipped: {len(self.skipped)}")

    def confirm_current(self):
        """Confirm the current candidate as a real cell."""
        if not self.review_mode or self.candidates is None:
            return

        self.confirmed.add(self.current_index)
        self.rejected.discard(self.current_index)
        self.skipped.discard(self.current_index)

        self.candidate_confirmed.emit(self.current_index)
        self.update_stats()
        self.update_current_info()
        self.go_to_next_unreviewed()

    def reject_current(self):
        """Reject the current candidate as not a real cell."""
        if not self.review_mode or self.candidates is None:
            return

        self.rejected.add(self.current_index)
        self.confirmed.discard(self.current_index)
        self.skipped.discard(self.current_index)

        self.candidate_rejected.emit(self.current_index)
        self.update_stats()
        self.update_current_info()
        self.go_to_next_unreviewed()

    def skip_current(self):
        """Skip the current candidate (unsure)."""
        if not self.review_mode or self.candidates is None:
            return

        self.skipped.add(self.current_index)
        self.update_stats()
        self.update_current_info()
        self.go_to_next_unreviewed()

    def go_previous(self):
        """Go to the previous candidate without changing its status."""
        if self.candidates is None:
            return

        if self.review_indices is not None and len(self.review_indices) > 0:
            # Navigate within review set
            self.review_position = (self.review_position - 1) % len(self.review_indices)
            self.current_index = self.review_indices[self.review_position]
        else:
            self.current_index = (self.current_index - 1) % len(self.candidates)
        self.focus_on_current()

    def go_next(self):
        """Go to the next candidate without changing its status."""
        if self.candidates is None:
            return

        if self.review_indices is not None and len(self.review_indices) > 0:
            # Navigate within review set
            self.review_position = (self.review_position + 1) % len(self.review_indices)
            self.current_index = self.review_indices[self.review_position]
        else:
            self.current_index = (self.current_index + 1) % len(self.candidates)
        self.focus_on_current()

    def on_z_context_changed(self, value):
        """Handle Z context slider change."""
        self.z_context = value

    def on_zoom_changed(self, value):
        """Handle zoom slider change."""
        self.zoom_level = value
        self.zoom_label.setText(f"{value}%")
        if self.review_mode:
            self.viewer.camera.zoom = value / 100

    def save_confirmed(self):
        """Save confirmed cells to disk AND as a napari layer."""
        if self.candidates is None or len(self.confirmed) == 0:
            QMessageBox.warning(self, "Error", "No confirmed cells to save")
            return

        confirmed_points = self.candidates[list(self.confirmed)]

        # Save to disk
        save_path = self._save_points_to_xml(confirmed_points, "cells")

        # Also add as napari layer for visualization
        layer_name = f"Confirmed Cells ({len(confirmed_points)})"
        self.viewer.add_points(
            confirmed_points,
            name=layer_name,
            size=10,
            face_color='green',
            border_color='darkgreen',
            symbol='disc',
        )

        if save_path:
            QMessageBox.information(
                self, "Saved",
                f"Saved {len(confirmed_points)} confirmed cells:\n"
                f"- To disk: {save_path}\n"
                f"- To layer: '{layer_name}'"
            )
            # Log to session documenter if available
            self._log_to_session("confirmed", len(confirmed_points), save_path)
        else:
            QMessageBox.information(
                self, "Saved",
                f"Saved {len(confirmed_points)} confirmed cells as '{layer_name}'\n"
                "(Disk save failed - check console for errors)"
            )

    def save_rejected(self):
        """Save rejected candidates to disk AND as a napari layer."""
        if self.candidates is None or len(self.rejected) == 0:
            QMessageBox.warning(self, "Error", "No rejected candidates to save")
            return

        rejected_points = self.candidates[list(self.rejected)]

        # Save to disk
        save_path = self._save_points_to_xml(rejected_points, "non_cells")

        # Also add as napari layer for visualization
        layer_name = f"Rejected Candidates ({len(rejected_points)})"
        self.viewer.add_points(
            rejected_points,
            name=layer_name,
            size=10,
            face_color='red',
            border_color='darkred',
            symbol='x',
        )

        if save_path:
            QMessageBox.information(
                self, "Saved",
                f"Saved {len(rejected_points)} rejected candidates:\n"
                f"- To disk: {save_path}\n"
                f"- To layer: '{layer_name}'"
            )
            # Log to session documenter if available
            self._log_to_session("rejected", len(rejected_points), save_path)
        else:
            QMessageBox.information(
                self, "Saved",
                f"Saved {len(rejected_points)} rejected candidates as '{layer_name}'\n"
                "(Disk save failed - check console for errors)"
            )

    def _save_points_to_xml(self, points, cell_type: str) -> Path:
        """
        Save points to XML format compatible with cellfinder/BrainGlobe.

        Args:
            points: numpy array of points (Z, Y, X)
            cell_type: "cells" or "non_cells"

        Returns:
            Path to saved file, or None if failed
        """
        from datetime import datetime
        import xml.etree.ElementTree as ET
        from qtpy.QtWidgets import QFileDialog

        # Ask user where to save
        default_name = f"curated_{cell_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {cell_type}",
            default_name,
            "XML files (*.xml);;All files (*.*)"
        )

        if not save_path:
            return None

        save_path = Path(save_path)

        try:
            # Create XML in BrainGlobe/cellfinder format
            root = ET.Element("CellCounter_Marker_File")

            # Image properties (placeholder)
            img_props = ET.SubElement(root, "Image_Properties")
            ET.SubElement(img_props, "Image_Filename").text = "curated_data"

            # Marker data
            marker_data = ET.SubElement(root, "Marker_Data")
            ET.SubElement(marker_data, "Current_Type").text = "1"

            marker_type = ET.SubElement(marker_data, "Marker_Type")
            # Type 1 = cells, Type 2 = non_cells (cellfinder convention)
            ET.SubElement(marker_type, "Type").text = "1" if cell_type == "cells" else "2"

            # Add each point as a marker
            for point in points:
                marker = ET.SubElement(marker_type, "Marker")
                # Points are in (Z, Y, X) order in napari, convert to XML format
                z, y, x = int(point[0]), int(point[1]), int(point[2])
                ET.SubElement(marker, "MarkerX").text = str(x)
                ET.SubElement(marker, "MarkerY").text = str(y)
                ET.SubElement(marker, "MarkerZ").text = str(z)

            # Write to file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(str(save_path), encoding="UTF-8", xml_declaration=True)

            print(f"[Connectome] Saved {len(points)} {cell_type} to {save_path}")
            return save_path

        except Exception as e:
            print(f"[Connectome] Error saving to XML: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_to_session(self, curation_type: str, count: int, save_path: Path):
        """Log curation action to session documenter if available."""
        try:
            # Try to find the tuning widget's session documenter
            for widget in self.viewer.window.dock_widgets.values():
                if hasattr(widget, 'widget'):
                    w = widget.widget()
                    if hasattr(w, 'session_doc') and w.session_doc is not None:
                        # Log the curation
                        if hasattr(w.session_doc, 'log_curation'):
                            w.session_doc.log_curation(
                                confirmed=count if curation_type == "confirmed" else 0,
                                rejected=count if curation_type == "rejected" else 0,
                                skipped=0
                            )
                        # Also log the export/save
                        if hasattr(w.session_doc, 'log_export'):
                            w.session_doc.log_export(str(save_path), f"curated_{curation_type}")
                        print(f"[Connectome] Logged curation to session documenter")
                        return
        except Exception as e:
            print(f"[Connectome] Could not log to session: {e}")

    def get_curation_results(self):
        """Get the curation results."""
        return {
            'total': len(self.candidates) if self.candidates is not None else 0,
            'confirmed': list(self.confirmed),
            'rejected': list(self.rejected),
            'skipped': list(self.skipped),
            'confirmed_count': len(self.confirmed),
            'rejected_count': len(self.rejected),
            'skipped_count': len(self.skipped),
        }
