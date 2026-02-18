"""
cell_viewer_3d.py — Lightweight 3D cell viewer for napari.

Loads detected cells as colored points (by eLife functional group) overlaid
on a ghost atlas volume. Uses native napari 3D rendering for fast, interactive
exploration of spatial cell distributions.
"""

import csv
import json
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QCheckBox, QSpinBox, QScrollArea,
    QFrame, QSizePolicy,
)
from qtpy.QtCore import Qt

from mousebrain.config import BRAINS_ROOT
from mousebrain.region_mapping import get_elife_group, ELIFE_GROUPS

# =============================================================================
# CONSTANTS
# =============================================================================

FOLDER_REGISTERED = "3_Registered_Atlas"
FOLDER_ANALYSIS = "6_Region_Analysis"
CELL_CSV = "all_points_information.csv"


# =============================================================================
# WIDGET
# =============================================================================

class Cell3DViewer(QWidget):
    """Napari widget for interactive 3D visualization of detected cells."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_brain = None
        self.current_brain_path = None
        self._atlas_name = None
        self._cell_data = None
        self._elife_colors = None
        self.setup_ui()
        self.refresh_brains()

    # -------------------------------------------------------------------------
    # UI SETUP
    # -------------------------------------------------------------------------

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Brain Selector ---
        brain_group = QGroupBox("Brain")
        brain_layout = QVBoxLayout()
        brain_group.setLayout(brain_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Brain:"))
        self.brain_combo = QComboBox()
        self.brain_combo.currentTextChanged.connect(self._on_brain_changed)
        row.addWidget(self.brain_combo, stretch=1)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh brain list")
        refresh_btn.clicked.connect(self.refresh_brains)
        row.addWidget(refresh_btn)
        brain_layout.addLayout(row)

        self.brain_info = QLabel("")
        self.brain_info.setStyleSheet("color: #888; font-size: 11px;")
        self.brain_info.setWordWrap(True)
        brain_layout.addWidget(self.brain_info)

        layout.addWidget(brain_group)

        # --- Display Options ---
        opts_group = QGroupBox("Display")
        opts_layout = QVBoxLayout()
        opts_group.setLayout(opts_layout)

        self.show_atlas_cb = QCheckBox("Show atlas regions (ghost)")
        self.show_atlas_cb.setChecked(True)
        opts_layout.addWidget(self.show_atlas_cb)

        self.show_boundaries_cb = QCheckBox("Show brain outline")
        self.show_boundaries_cb.setChecked(True)
        opts_layout.addWidget(self.show_boundaries_cb)

        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("Atlas downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 10)
        self.downsample_spin.setValue(4)
        self.downsample_spin.setToolTip("Higher = faster loading, lower resolution ghost")
        ds_row.addWidget(self.downsample_spin)
        ds_row.addWidget(QLabel("x"))
        ds_row.addStretch()
        opts_layout.addLayout(ds_row)

        pt_row = QHBoxLayout()
        pt_row.addWidget(QLabel("Point size:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 30)
        self.point_size_spin.setValue(5)
        pt_row.addWidget(self.point_size_spin)
        pt_row.addStretch()
        opts_layout.addLayout(pt_row)

        layout.addWidget(opts_group)

        # --- eLife Group Filter ---
        elife_group = QGroupBox("eLife Groups")
        elife_layout = QVBoxLayout()
        elife_group.setLayout(elife_layout)

        elife_btn_row = QHBoxLayout()
        select_all = QPushButton("All")
        select_all.setFixedWidth(40)
        select_all.clicked.connect(lambda: self._set_all_elife(True))
        select_none = QPushButton("None")
        select_none.setFixedWidth(40)
        select_none.clicked.connect(lambda: self._set_all_elife(False))
        elife_btn_row.addWidget(select_all)
        elife_btn_row.addWidget(select_none)
        elife_btn_row.addStretch()
        elife_layout.addLayout(elife_btn_row)

        # Scrollable list of eLife group checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll_widget = QWidget()
        self.elife_checks_layout = QVBoxLayout()
        self.elife_checks_layout.setContentsMargins(2, 2, 2, 2)
        self.elife_checks_layout.setSpacing(1)
        scroll_widget.setLayout(self.elife_checks_layout)
        scroll.setWidget(scroll_widget)
        elife_layout.addWidget(scroll)

        self.elife_checkboxes = {}
        self._build_elife_checkboxes()

        layout.addWidget(elife_group)

        # --- Load Button ---
        self.load_btn = QPushButton("Load 3D View")
        self.load_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; font-size: 13px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self.load_3d_view)
        layout.addWidget(self.load_btn)

        # --- Clear Button ---
        self.clear_btn = QPushButton("Clear 3D Layers")
        self.clear_btn.setStyleSheet(
            "QPushButton { padding: 4px; }"
        )
        self.clear_btn.clicked.connect(self._clear_3d_layers)
        layout.addWidget(self.clear_btn)

        # --- Status ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        self.status_label = QLabel("Select a brain to begin.")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    # -------------------------------------------------------------------------
    # BRAIN DISCOVERY
    # -------------------------------------------------------------------------

    def refresh_brains(self):
        """Populate brain dropdown from BRAINS_ROOT."""
        self.brain_combo.blockSignals(True)
        self.brain_combo.clear()
        self.brain_combo.addItem("Select a brain...", None)

        if not BRAINS_ROOT.exists():
            self.brain_combo.blockSignals(False)
            return

        for mouse_dir in sorted(BRAINS_ROOT.iterdir()):
            if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
                continue
            if any(s in mouse_dir.name.lower()
                   for s in ['script', 'backup', 'archive', 'summary']):
                continue

            for pipeline_dir in sorted(mouse_dir.iterdir()):
                if not pipeline_dir.is_dir():
                    continue
                # Only show brains that have region analysis results
                analysis_dir = pipeline_dir / FOLDER_ANALYSIS
                cell_csv = analysis_dir / CELL_CSV
                if cell_csv.exists():
                    self.brain_combo.addItem(pipeline_dir.name, str(pipeline_dir))

        self.brain_combo.blockSignals(False)

    def _on_brain_changed(self, text):
        """Handle brain selection."""
        path = self.brain_combo.currentData()
        if not path:
            self.current_brain = None
            self.current_brain_path = None
            self.brain_info.setText("")
            self.load_btn.setEnabled(False)
            return

        self.current_brain_path = Path(path)
        self.current_brain = self.current_brain_path.name

        # Auto-detect atlas
        self._atlas_name = self._detect_atlas()

        # Count cells
        cell_csv = self.current_brain_path / FOLDER_ANALYSIS / CELL_CSV
        try:
            with open(cell_csv, 'r') as f:
                cell_count = sum(1 for _ in f) - 1  # minus header
        except Exception:
            cell_count = 0

        # Check for eLife grouped data
        elife_csv = self.current_brain_path / FOLDER_ANALYSIS / "cell_counts_elife_grouped.csv"
        has_elife = elife_csv.exists()

        info_parts = [
            f"{cell_count:,} cells detected",
            f"Atlas: {self._atlas_name or 'unknown'}",
        ]
        if has_elife:
            info_parts.append("eLife grouping available")

        self.brain_info.setText(" | ".join(info_parts))
        self.brain_info.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.load_btn.setEnabled(True)
        self.status_label.setText(f"Ready to load {self.current_brain}")

    def _detect_atlas(self):
        """Read atlas name from brainreg.json."""
        brainreg = self.current_brain_path / FOLDER_REGISTERED / "brainreg.json"
        if brainreg.exists():
            try:
                with open(brainreg, 'r') as f:
                    meta = json.load(f)
                return meta.get('atlas', 'allen_mouse_25um')
            except Exception:
                pass
        return 'allen_mouse_25um'

    # -------------------------------------------------------------------------
    # ELIFE GROUP CHECKBOXES
    # -------------------------------------------------------------------------

    def _build_elife_checkboxes(self):
        """Create checkboxes for each eLife group."""
        group_colors = _build_elife_hex_colors()
        sorted_groups = sorted(ELIFE_GROUPS.items(), key=lambda x: x[1]['id'])

        for group_name, group_data in sorted_groups:
            cb = QCheckBox(f"{group_name}")
            cb.setChecked(True)

            color = group_colors.get(group_name, '#888888')
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")

            self.elife_checkboxes[group_name] = cb
            self.elife_checks_layout.addWidget(cb)

        # Add unmapped checkbox
        cb = QCheckBox("[Unmapped regions]")
        cb.setChecked(True)
        cb.setStyleSheet("QCheckBox { color: #888; }")
        self.elife_checkboxes['[Unmapped]'] = cb
        self.elife_checks_layout.addWidget(cb)

    def _set_all_elife(self, checked):
        for cb in self.elife_checkboxes.values():
            cb.setChecked(checked)

    def _get_enabled_groups(self):
        return {name for name, cb in self.elife_checkboxes.items() if cb.isChecked()}

    # -------------------------------------------------------------------------
    # 3D LOADING
    # -------------------------------------------------------------------------

    def load_3d_view(self):
        """Load cells + ghost atlas into napari 3D viewer."""
        if not self.current_brain_path:
            return

        self.load_btn.setEnabled(False)
        self.status_label.setText("Loading...")
        self.status_label.setStyleSheet("color: #FF9800; font-size: 11px;")
        # Force UI update
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            self._do_load()
            self.status_label.setText(f"Loaded: {self.current_brain}")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #F44336; font-size: 11px;")
            import traceback
            traceback.print_exc()
        finally:
            self.load_btn.setEnabled(True)

    def _do_load(self):
        """Core loading logic."""
        # Clear previous 3D layers
        self._clear_3d_layers()

        ds = self.downsample_spin.value()
        point_size = self.point_size_spin.value()
        enabled_groups = self._get_enabled_groups()

        # 1. Load cell data
        self.status_label.setText("Loading cell coordinates...")
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()

        cell_csv = self.current_brain_path / FOLDER_ANALYSIS / CELL_CSV
        coords, structure_names, hemispheres = _load_cells(cell_csv)

        # 2. Build colors and filter by enabled eLife groups
        group_colors = _build_elife_rgba_colors()
        name_to_group = _build_name_to_group()

        display_coords = []
        display_colors = []

        for i in range(len(coords)):
            name = structure_names[i]
            hemi = hemispheres[i]
            group = name_to_group.get(name) or get_elife_group(name) or '[Unmapped]'

            if group not in enabled_groups:
                continue

            # Color: bright for left, dark for right
            if group in group_colors:
                color = group_colors[group]['bright' if hemi == 'left' else 'dark']
            else:
                color = [0.53, 0.53, 0.53, 1.0] if hemi == 'left' else [0.37, 0.37, 0.37, 1.0]

            # Coordinates in atlas voxel space, scaled by downsample factor
            display_coords.append(coords[i] / ds)
            display_colors.append(color)

        if not display_coords:
            self.status_label.setText("No cells to display with current filters.")
            return

        display_coords = np.array(display_coords)
        display_colors = np.array(display_colors)

        # 3. Load ghost atlas
        if self.show_atlas_cb.isChecked() or self.show_boundaries_cb.isChecked():
            self.status_label.setText("Loading atlas volume...")
            QApplication.processEvents()

            try:
                from brainglobe_atlasapi import BrainGlobeAtlas
                atlas = BrainGlobeAtlas(self._atlas_name)
                annotation = atlas.annotation  # 3D uint32 array
            except Exception as e:
                print(f"Warning: Could not load atlas: {e}")
                annotation = None

            if annotation is not None:
                # Downsample atlas for performance
                ds_annotation = annotation[::ds, ::ds, ::ds]

                if self.show_atlas_cb.isChecked():
                    # Add as Labels layer — ghost regions
                    self.viewer.add_labels(
                        ds_annotation,
                        name="[3D] Atlas Regions",
                        opacity=0.15,
                        blending='translucent',
                    )

                if self.show_boundaries_cb.isChecked():
                    # Create brain outline mask (where annotation > 0)
                    brain_mask = (ds_annotation > 0).astype(np.uint8)
                    self.viewer.add_image(
                        brain_mask,
                        name="[3D] Brain Outline",
                        opacity=0.08,
                        blending='additive',
                        colormap='gray',
                        rendering='iso',
                        iso_threshold=0.5,
                    )

        # 4. Add cell points
        self.status_label.setText(f"Adding {len(display_coords):,} cells...")
        QApplication.processEvents()

        self.viewer.add_points(
            display_coords,
            face_color=display_colors,
            size=point_size,
            name="[3D] Detected Cells",
            edge_width=0,
            shading='spherical',
            out_of_slice_display=True,
        )

        # 5. Switch to 3D
        self.viewer.dims.ndisplay = 3

        self.status_label.setText(
            f"Loaded {len(display_coords):,} cells "
            f"({len(enabled_groups)} groups) — {self.current_brain}"
        )

    def _clear_3d_layers(self):
        """Remove layers added by this widget."""
        to_remove = [
            layer for layer in self.viewer.layers
            if layer.name.startswith("[3D]")
        ]
        for layer in to_remove:
            self.viewer.layers.remove(layer)


# =============================================================================
# DATA HELPERS (module-level, reusable)
# =============================================================================

def _load_cells(csv_path):
    """Load cell coordinates from all_points_information.csv.

    Returns atlas-space voxel coordinates (Nx3), structure names, hemispheres.
    """
    coords = []
    structure_names = []
    hemispheres = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append([
                float(row['coordinate_atlas_axis_0']),
                float(row['coordinate_atlas_axis_1']),
                float(row['coordinate_atlas_axis_2']),
            ])
            structure_names.append(row.get('structure_name', ''))
            hemispheres.append(row.get('hemisphere', ''))

    return np.array(coords) if coords else np.empty((0, 3)), structure_names, hemispheres


def _build_name_to_group():
    """Build reverse lookup: structure name/acronym -> eLife group."""
    name_to_group = {}
    for group_name, group_data in ELIFE_GROUPS.items():
        for fn in group_data.get('full_names', []):
            name_to_group[fn] = group_name
        for acr in group_data.get('acronyms', []):
            name_to_group[acr] = group_name
    return name_to_group


def _build_elife_hex_colors():
    """Hex colors for eLife groups (for checkbox styling)."""
    from matplotlib import colormaps
    tab20 = colormaps['tab20']
    tab20b = colormaps['tab20b']
    sorted_groups = sorted(ELIFE_GROUPS.items(), key=lambda x: x[1]['id'])

    colors = {}
    for i, (group_name, _) in enumerate(sorted_groups):
        rgba = tab20(i / 20) if i < 20 else tab20b((i - 20) / 20)
        colors[group_name] = f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'
    return colors


def _build_elife_rgba_colors():
    """RGBA float colors for eLife groups (for napari point coloring).

    Returns dict: {group_name: {'bright': [r,g,b,a], 'dark': [r,g,b,a]}}
    """
    from matplotlib import colormaps
    tab20 = colormaps['tab20']
    tab20b = colormaps['tab20b']
    sorted_groups = sorted(ELIFE_GROUPS.items(), key=lambda x: x[1]['id'])

    colors = {}
    for i, (group_name, _) in enumerate(sorted_groups):
        rgba = tab20(i / 20) if i < 20 else tab20b((i - 20) / 20)
        r, g, b = rgba[0], rgba[1], rgba[2]
        colors[group_name] = {
            'bright': [r, g, b, 1.0],
            'dark': [r * 0.7, g * 0.7, b * 0.7, 1.0],
        }
    # Unmapped
    colors['[Unmapped]'] = {
        'bright': [0.53, 0.53, 0.53, 1.0],
        'dark': [0.37, 0.37, 0.37, 1.0],
    }
    return colors
