"""
inset_widget.py - napari widget for managing high-resolution insets

Provides UI for:
- Adding inset images
- Visualizing alignment
- Manual position adjustment
- Running detection on insets
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QFileDialog, QMessageBox, QSpinBox,
    QCheckBox, QHeaderView,
)
from qtpy.QtCore import Qt, Signal

import napari


class InsetWidget(QWidget):
    """Widget for managing high-resolution inset images."""

    # Signal emitted when insets change
    insets_changed = Signal()

    def __init__(
        self,
        parent_widget: 'BrainSliceWidget',
    ):
        super().__init__()
        self.parent_widget = parent_widget
        self.viewer = parent_widget.viewer

        # InsetManager (created when base image is loaded)
        self.inset_manager = None

        # Track inset layers
        self.inset_layers: Dict[str, napari.layers.Layer] = {}

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Info label
        info_label = QLabel(
            "Add high-resolution insets for regions of interest.\n"
            "Insets will be aligned to the base image and used for detection."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)

        # Add insets group
        add_group = QGroupBox("Add Insets")
        add_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.add_single_btn = QPushButton("Add Inset...")
        self.add_single_btn.clicked.connect(self._add_single_inset)
        btn_layout.addWidget(self.add_single_btn)

        self.find_matching_btn = QPushButton("Find Matching")
        self.find_matching_btn.clicked.connect(self._find_matching_insets)
        self.find_matching_btn.setToolTip("Find insets with matching names in same folder")
        btn_layout.addWidget(self.find_matching_btn)

        add_layout.addLayout(btn_layout)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        # Insets table
        table_group = QGroupBox("Loaded Insets")
        table_layout = QVBoxLayout()

        self.insets_table = QTableWidget()
        self.insets_table.setColumnCount(5)
        self.insets_table.setHorizontalHeaderLabels([
            'Name', 'Region', 'Aligned', 'Scale', 'Actions'
        ])
        self.insets_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.insets_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.insets_table.itemSelectionChanged.connect(self._on_selection_changed)
        table_layout.addWidget(self.insets_table)

        # Table action buttons
        table_btn_layout = QHBoxLayout()

        self.show_btn = QPushButton("Show/Hide")
        self.show_btn.clicked.connect(self._toggle_visibility)
        table_btn_layout.addWidget(self.show_btn)

        self.align_btn = QPushButton("Re-align")
        self.align_btn.clicked.connect(self._realign_selected)
        table_btn_layout.addWidget(self.align_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        table_btn_layout.addWidget(self.remove_btn)

        table_layout.addLayout(table_btn_layout)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # Manual position adjustment
        adjust_group = QGroupBox("Manual Position (Selected)")
        adjust_layout = QVBoxLayout()

        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("X:"))
        self.pos_x_spin = QSpinBox()
        self.pos_x_spin.setRange(0, 100000)
        pos_layout.addWidget(self.pos_x_spin)

        pos_layout.addWidget(QLabel("Y:"))
        self.pos_y_spin = QSpinBox()
        self.pos_y_spin.setRange(0, 100000)
        pos_layout.addWidget(self.pos_y_spin)

        self.apply_pos_btn = QPushButton("Apply")
        self.apply_pos_btn.clicked.connect(self._apply_position)
        pos_layout.addWidget(self.apply_pos_btn)

        adjust_layout.addLayout(pos_layout)

        # Live update checkbox
        self.live_update_check = QCheckBox("Live update from layer transform")
        self.live_update_check.setChecked(True)
        adjust_layout.addWidget(self.live_update_check)

        adjust_group.setLayout(adjust_layout)
        layout.addWidget(adjust_group)

        # Detection options
        detect_group = QGroupBox("Inset Detection")
        detect_layout = QVBoxLayout()

        self.use_insets_check = QCheckBox("Use insets for detection (full resolution)")
        self.use_insets_check.setChecked(True)
        detect_layout.addWidget(self.use_insets_check)

        self.detect_base_check = QCheckBox("Also detect in base image (excluding inset regions)")
        self.detect_base_check.setChecked(True)
        detect_layout.addWidget(self.detect_base_check)

        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        # Composite visualization
        composite_group = QGroupBox("Visualization")
        composite_layout = QVBoxLayout()

        self.show_composite_btn = QPushButton("Show Composite")
        self.show_composite_btn.clicked.connect(self._show_composite)
        composite_layout.addWidget(self.show_composite_btn)

        self.show_mask_btn = QPushButton("Show Inset Regions")
        self.show_mask_btn.clicked.connect(self._show_inset_mask)
        composite_layout.addWidget(self.show_mask_btn)

        composite_group.setLayout(composite_layout)
        layout.addWidget(composite_group)

        layout.addStretch()

        # Initial state
        self._update_ui_state()

    def _update_ui_state(self):
        """Update UI based on current state."""
        has_base = self.parent_widget.red_channel is not None
        has_insets = self.inset_manager is not None and len(self.inset_manager.insets) > 0

        self.add_single_btn.setEnabled(has_base)
        self.find_matching_btn.setEnabled(has_base)
        self.show_composite_btn.setEnabled(has_insets)
        self.show_mask_btn.setEnabled(has_insets)

    def on_base_loaded(self):
        """Called when base image is loaded in parent widget."""
        from ..core.insets import InsetManager

        # Get current slice if data is a 3D stack (from folder loading)
        # This ensures InsetManager works with 2D base images
        red = self.parent_widget._get_current_slice(self.parent_widget.red_channel)
        green = self.parent_widget._get_current_slice(self.parent_widget.green_channel)

        # Create InsetManager with base image (2D or stacked 2-channel)
        if red is not None and green is not None:
            base_image = np.stack([red, green])  # (C=2, Y, X)
        elif red is not None:
            base_image = red  # (Y, X)
        else:
            base_image = None

        if base_image is None:
            return

        self.inset_manager = InsetManager(base_image, self.parent_widget.metadata)

        # Clear previous insets
        self.insets_table.setRowCount(0)
        self.inset_layers.clear()

        self._update_ui_state()

        # Auto-find matching insets
        if self.parent_widget.current_file:
            self._find_matching_insets()

    def _add_single_inset(self):
        """Open file dialog to add a single inset."""
        if self.inset_manager is None:
            QMessageBox.warning(self, "Error", "Load a base image first")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Inset Image",
            str(self.parent_widget.current_file.parent) if self.parent_widget.current_file else "",
            "Image Files (*.nd2 *.tif *.tiff);;All Files (*)"
        )

        if file_path:
            self._load_inset(Path(file_path))

    def _find_matching_insets(self):
        """Find and load insets with matching naming convention."""
        if self.inset_manager is None or self.parent_widget.current_file is None:
            return

        from ..core.insets import find_matching_insets

        matching = find_matching_insets(self.parent_widget.current_file)

        if not matching:
            self.parent_widget.status_label.setText("No matching insets found")
            return

        loaded = 0
        for inset_path in matching:
            if self._load_inset(inset_path):
                loaded += 1

        self.parent_widget.status_label.setText(f"Loaded {loaded} matching insets")

    def _load_inset(self, file_path: Path) -> bool:
        """Load a single inset file."""
        try:
            name = self.inset_manager.add_inset(file_path, auto_align=True)
            inset = self.inset_manager.insets[name]

            # Add to table
            row = self.insets_table.rowCount()
            self.insets_table.insertRow(row)

            self.insets_table.setItem(row, 0, QTableWidgetItem(name))
            self.insets_table.setItem(row, 1, QTableWidgetItem(inset.region))
            self.insets_table.setItem(row, 2, QTableWidgetItem(
                "Yes" if inset.aligned else "No"
            ))
            self.insets_table.setItem(row, 3, QTableWidgetItem(
                f"{inset.scale_factor:.2f}x"
            ))

            # Store name in row data
            self.insets_table.item(row, 0).setData(Qt.UserRole, name)

            # Add to napari
            self._add_inset_to_viewer(name)

            self._update_ui_state()
            self.insets_changed.emit()

            return True

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load inset: {e}")
            return False

    def _add_inset_to_viewer(self, inset_name: str):
        """Add inset image layer to napari viewer."""
        inset = self.inset_manager.insets[inset_name]

        # Get display channel (red channel for visibility)
        if inset.image.ndim == 3:
            display_data = inset.image[0]  # Red channel
        else:
            display_data = inset.image

        # Calculate transform to position inset
        if inset.aligned and inset.position is not None:
            # Scale and translate
            scale = (1.0 / inset.scale_factor, 1.0 / inset.scale_factor)
            translate = (inset.position[1], inset.position[0])  # (y, x)
        else:
            scale = (1.0, 1.0)
            translate = (0, 0)

        # Add as image layer with distinctive styling
        layer = self.viewer.add_image(
            display_data,
            name=f"Inset: {inset_name}",
            colormap='magenta',
            blending='additive',
            opacity=0.7,
            scale=scale,
            translate=translate,
        )

        # Store reference
        self.inset_layers[inset_name] = layer

        # Connect to transform changes for live update
        layer.events.affine.connect(lambda event, n=inset_name: self._on_layer_transform_changed(n))

    def _on_layer_transform_changed(self, inset_name: str):
        """Handle layer transform changes (user drag)."""
        if not self.live_update_check.isChecked():
            return

        if inset_name not in self.inset_layers:
            return

        layer = self.inset_layers[inset_name]

        # Extract translate from layer transform
        # napari uses (y, x) order
        translate = layer.translate
        if len(translate) >= 2:
            new_x = int(translate[1])
            new_y = int(translate[0])

            # Update InsetManager
            self.inset_manager.set_inset_position(inset_name, (new_x, new_y))

            # Update UI if this is selected
            self._update_position_spinboxes()

    def _on_selection_changed(self):
        """Handle table selection change."""
        self._update_position_spinboxes()

    def _update_position_spinboxes(self):
        """Update position spinboxes for selected inset."""
        selected = self.insets_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        name_item = self.insets_table.item(row, 0)
        if name_item is None:
            return

        inset_name = name_item.data(Qt.UserRole)
        if inset_name not in self.inset_manager.insets:
            return

        inset = self.inset_manager.insets[inset_name]
        if inset.position:
            self.pos_x_spin.setValue(inset.position[0])
            self.pos_y_spin.setValue(inset.position[1])

    def _apply_position(self):
        """Apply manual position to selected inset."""
        selected = self.insets_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "Select an inset first")
            return

        row = selected[0].row()
        name_item = self.insets_table.item(row, 0)
        inset_name = name_item.data(Qt.UserRole)

        new_pos = (self.pos_x_spin.value(), self.pos_y_spin.value())

        # Update InsetManager
        self.inset_manager.set_inset_position(inset_name, new_pos)

        # Update layer transform
        if inset_name in self.inset_layers:
            layer = self.inset_layers[inset_name]
            layer.translate = (new_pos[1], new_pos[0])  # (y, x)

        # Update table
        self.insets_table.setItem(row, 2, QTableWidgetItem("Yes"))

        self.parent_widget.status_label.setText(f"Updated position for {inset_name}")

    def _toggle_visibility(self):
        """Toggle visibility of selected inset layer."""
        selected = self.insets_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        name_item = self.insets_table.item(row, 0)
        inset_name = name_item.data(Qt.UserRole)

        if inset_name in self.inset_layers:
            layer = self.inset_layers[inset_name]
            layer.visible = not layer.visible

    def _realign_selected(self):
        """Re-run alignment for selected inset."""
        selected = self.insets_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "Select an inset first")
            return

        row = selected[0].row()
        name_item = self.insets_table.item(row, 0)
        inset_name = name_item.data(Qt.UserRole)

        inset = self.inset_manager.insets[inset_name]
        success = self.inset_manager._align_inset(inset)

        if success:
            # Update layer transform
            if inset_name in self.inset_layers:
                layer = self.inset_layers[inset_name]
                layer.translate = (inset.position[1], inset.position[0])

            # Update table
            self.insets_table.setItem(row, 2, QTableWidgetItem("Yes"))

            self.parent_widget.status_label.setText(
                f"Aligned {inset_name} (score: {inset.alignment_score:.2f})"
            )
        else:
            self.parent_widget.status_label.setText(f"Alignment failed for {inset_name}")

    def _remove_selected(self):
        """Remove selected inset."""
        selected = self.insets_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        name_item = self.insets_table.item(row, 0)
        inset_name = name_item.data(Qt.UserRole)

        # Remove from napari
        if inset_name in self.inset_layers:
            layer = self.inset_layers[inset_name]
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
            del self.inset_layers[inset_name]

        # Remove from manager
        if inset_name in self.inset_manager.insets:
            del self.inset_manager.insets[inset_name]

        # Remove from table
        self.insets_table.removeRow(row)

        self._update_ui_state()
        self.insets_changed.emit()

    def _show_composite(self):
        """Show composite image with insets overlaid."""
        if self.inset_manager is None or len(self.inset_manager.insets) == 0:
            return

        composite = self.inset_manager.create_composite()

        # Remove old composite layer
        for layer in list(self.viewer.layers):
            if 'Composite' in layer.name:
                self.viewer.layers.remove(layer)

        # Add as new layer
        if composite.ndim == 3:
            # Multi-channel
            for i, name in enumerate(['Red', 'Green']):
                self.viewer.add_image(
                    composite[i],
                    name=f"Composite {name}",
                    colormap='red' if i == 0 else 'green',
                    blending='additive',
                )
        else:
            self.viewer.add_image(
                composite,
                name="Composite",
                colormap='gray',
            )

    def _show_inset_mask(self):
        """Show mask of inset regions."""
        if self.inset_manager is None or len(self.inset_manager.insets) == 0:
            return

        mask = self.inset_manager.create_inset_mask()

        # Remove old mask layer
        for layer in list(self.viewer.layers):
            if 'Inset Regions' in layer.name:
                self.viewer.layers.remove(layer)

        self.viewer.add_labels(
            mask.astype(np.int32),
            name="Inset Regions",
            opacity=0.3,
        )

    def get_detection_settings(self) -> Dict[str, Any]:
        """Get inset detection settings for use by detection pipeline."""
        return {
            'use_insets': self.use_insets_check.isChecked(),
            'detect_in_base': self.detect_base_check.isChecked(),
            'inset_manager': self.inset_manager,
        }
