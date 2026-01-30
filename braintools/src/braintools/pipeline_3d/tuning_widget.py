"""
BrainGlobe Tuning Widget for napari.

Helps scientists figure out the right settings for their specific tissue:
- Voxel size calibration
- Orientation detection
- Crop optimization
- Detection parameter tuning
- Network training curation

This is for the iterative "development" phase before running the full pipeline.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTextEdit, QTabWidget, QFormLayout, QMessageBox, QSlider,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QScrollArea, QProgressDialog, QSplitter, QApplication,
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem, QDialog
)
import napari
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont, QColor
import numpy as np

# Import paths from central config
from braintools.config import BRAINS_ROOT, SCRIPTS_DIR, MODELS_DIR, DATA_SUMMARY_DIR

# Try to import experiment tracker and comparison data
try:
    from braintools.tracker import ExperimentTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

try:
    from util_compare_to_published import PUBLISHED_REFERENCE, map_to_elife_regions
    REFERENCE_AVAILABLE = True
except ImportError:
    PUBLISHED_REFERENCE = {}
    REFERENCE_AVAILABLE = False

# Import session documenter
try:
    from braintools.pipeline_3d.session_documenter import SessionDocumenterWidget
    SESSION_DOC_AVAILABLE = True
except ImportError:
    SESSION_DOC_AVAILABLE = False


class BrainLoaderWorker(QThread):
    """Background worker for loading brain images - uses Zarr if available for FAST loading."""
    progress = Signal(str)
    finished = Signal(bool, str, object, object)  # success, message, signal_stack, bg_stack

    def __init__(self, signal_path, background_path, use_dask=True):
        super().__init__()
        self.signal_path = signal_path
        self.background_path = background_path
        self.use_dask = use_dask

    def run(self):
        try:
            signal_stack = None
            bg_stack = None

            # Check for Zarr first (instant loading!)
            zarr_signal = self.signal_path.parent / f"{self.signal_path.name}.zarr"
            zarr_bg = self.background_path.parent / f"{self.background_path.name}.zarr" if self.background_path else None

            if zarr_signal.exists():
                try:
                    import zarr
                    import dask.array as da
                    self.progress.emit("Loading signal from Zarr (fast!)...")
                    print(f"[SCI-Connectome] Zarr path: {zarr_signal}")
                    # Open zarr array and wrap with dask
                    z = zarr.open_array(str(zarr_signal), mode='r')
                    if self.use_dask:
                        signal_stack = da.from_array(z, chunks=z.chunks)
                    else:
                        signal_stack = z[:]
                    print(f"[SCI-Connectome] Loaded signal from Zarr: {signal_stack.shape}")

                    if zarr_bg and zarr_bg.exists():
                        self.progress.emit("Loading background from Zarr (fast!)...")
                        z = zarr.open_array(str(zarr_bg), mode='r')
                        if self.use_dask:
                            bg_stack = da.from_array(z, chunks=z.chunks)
                        else:
                            bg_stack = z[:]

                    self.finished.emit(True, "Loaded from Zarr (instant!)", signal_stack, bg_stack)
                    return
                except Exception as e:
                    # Zarr v3 format not compatible with zarr 2.x - fall through to TIFF
                    print(f"[SCI-Connectome] Zarr load failed ({e}), falling back to TIFF...")

            # Fall back to TIFF loading with Dask for lazy loading
            import tifffile
            from natsort import natsorted

            signal_files = natsorted(self.signal_path.glob("*.tif*"))
            if signal_files:
                if self.use_dask:
                    try:
                        import dask
                        import dask.array as da
                        self.progress.emit(f"Lazy-loading signal ({len(signal_files)} slices)...")
                        # Lazy load - doesn't actually read data yet
                        signal_stack = da.from_delayed(
                            dask.delayed(tifffile.imread)([str(f) for f in signal_files]),
                            shape=(len(signal_files), *tifffile.imread(str(signal_files[0])).shape),
                            dtype=tifffile.imread(str(signal_files[0])).dtype
                        )
                        # Actually, simpler approach - use tifffile's built-in lazy loading
                        signal_stack = tifffile.imread([str(f) for f in signal_files], aszarr=True)
                        signal_stack = da.from_zarr(signal_stack)
                        print(f"[SCI-Connectome] Lazy-loaded signal: {signal_stack.shape}")
                    except Exception as e:
                        print(f"[SCI-Connectome] Dask loading failed, falling back to full load: {e}")
                        self.progress.emit(f"Loading signal ({len(signal_files)} slices)...")
                        signal_stack = tifffile.imread([str(f) for f in signal_files])
                else:
                    self.progress.emit(f"Loading signal ({len(signal_files)} slices)...")
                    signal_stack = tifffile.imread([str(f) for f in signal_files])

            # Load background channel
            if self.background_path and self.background_path.exists():
                bg_files = natsorted(self.background_path.glob("*.tif*"))
                if bg_files:
                    if self.use_dask:
                        try:
                            import dask.array as da
                            self.progress.emit(f"Lazy-loading background ({len(bg_files)} slices)...")
                            bg_stack = tifffile.imread([str(f) for f in bg_files], aszarr=True)
                            bg_stack = da.from_zarr(bg_stack)
                        except:
                            self.progress.emit(f"Loading background ({len(bg_files)} slices)...")
                            bg_stack = tifffile.imread([str(f) for f in bg_files])
                    else:
                        self.progress.emit(f"Loading background ({len(bg_files)} slices)...")
                        bg_stack = tifffile.imread([str(f) for f in bg_files])

            self.finished.emit(True, "Load complete", signal_stack, bg_stack)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None, None)


class ZarrConverterWorker(QThread):
    """Convert TIFF stack to Zarr for fast future loading."""
    progress = Signal(str, int)  # message, percent
    finished = Signal(bool, str)  # success, message

    def __init__(self, tiff_folder, zarr_path):
        super().__init__()
        self.tiff_folder = tiff_folder
        self.zarr_path = zarr_path

    def run(self):
        try:
            import zarr
            import tifffile
            from natsort import natsorted

            tiff_files = natsorted(self.tiff_folder.glob("*.tif*"))
            if not tiff_files:
                self.finished.emit(False, "No TIFF files found")
                return

            # Get shape from first file
            first = tifffile.imread(str(tiff_files[0]))
            shape = (len(tiff_files), *first.shape)
            dtype = first.dtype

            self.progress.emit(f"Creating Zarr array {shape}...", 0)

            # Create Zarr array with good chunking for 3D viewing
            z_store = zarr.open(
                str(self.zarr_path),
                mode='w',
                shape=shape,
                dtype=dtype,
                chunks=(10, shape[1], shape[2]),  # 10 Z slices per chunk
            )

            # Copy data
            for i, tiff_file in enumerate(tiff_files):
                z_store[i] = tifffile.imread(str(tiff_file))
                if i % 50 == 0:
                    pct = int(100 * i / len(tiff_files))
                    self.progress.emit(f"Converting {i}/{len(tiff_files)}...", pct)

            self.progress.emit("Done!", 100)
            self.finished.emit(True, f"Converted to {self.zarr_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))


class DetectionWorker(QThread):
    """Background worker for running cellfinder detection using Python API (like native plugin)."""
    progress = Signal(str)
    finished = Signal(bool, str, int, object)  # success, message, cell_count, cells

    def __init__(self, signal_array, background_array, voxel_sizes, params):
        super().__init__()
        self.signal_array = signal_array
        self.background_array = background_array
        self.voxel_sizes = voxel_sizes  # (z, y, x) in microns
        self.params = params

    def run(self):
        try:
            # Use cellfinder Python API directly (same as native napari plugin)
            from cellfinder.core.main import main as cellfinder_main

            self.progress.emit("Running cell detection (Python API)...")
            print(f"\n[SCI-Connectome] Starting detection with cellfinder Python API")
            print(f"  Signal shape: {self.signal_array.shape}")
            print(f"  Background shape: {self.background_array.shape}")
            print(f"  Voxel sizes (ZYX): {self.voxel_sizes}")
            print(f"  Parameters: ball_xy={self.params['ball_xy_size']}, ball_z={self.params['ball_z_size']}, "
                  f"soma={self.params['soma_diameter']}, thresh={self.params['threshold']}")

            # Call cellfinder main directly with numpy arrays
            # skip_classification=True for detection only (faster)

            # Windows has a 63 handle limit for multiprocessing
            # Cap workers to avoid "need at most 63 handles" error
            import os
            total_cpus = os.cpu_count() or 8
            n_free = int(self.params.get('n_free_cpus', 2))
            workers = total_cpus - n_free
            if workers > 60:  # Leave headroom for Windows limit
                n_free = total_cpus - 60
                print(f"[SCI-Connectome] Capping workers to 60 (Windows limit), n_free_cpus={n_free}")

            detected_cells = cellfinder_main(
                signal_array=self.signal_array,
                background_array=self.background_array,
                voxel_sizes=self.voxel_sizes,
                # Ball filter params
                ball_xy_size=int(self.params['ball_xy_size']),
                ball_z_size=int(self.params['ball_z_size']),
                ball_overlap_fraction=self.params.get('ball_overlap_fraction', 0.6),
                # Soma params
                soma_diameter=int(self.params['soma_diameter']),
                soma_spread_factor=self.params.get('soma_spread_factor', 1.4),
                # Threshold params
                n_sds_above_mean_thresh=self.params['threshold'],
                log_sigma_size=self.params.get('log_sigma_size', 0.2),
                # Cluster params
                max_cluster_size=int(self.params.get('max_cluster_size', 100000000)),
                # Performance - capped for Windows
                n_free_cpus=n_free,
                # Skip classification for speed
                skip_classification=True,
            )

            cell_count = len(detected_cells) if detected_cells else 0
            print(f"[SCI-Connectome] Detection complete: {cell_count} candidates found")

            self.finished.emit(True, f"Detection complete: {cell_count} candidates found", cell_count, detected_cells)

        except ImportError as e:
            self.finished.emit(False, f"cellfinder not installed properly: {e}", 0, None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, f"Error: {str(e)}", 0, None)


class ClassificationWorker(QThread):
    """Background worker for running cell classification using cellfinder Python API."""
    finished = Signal(bool, str, int, int, str)  # success, message, cells_found, rejected, exp_id
    progress = Signal(str)

    def __init__(self, signal_array, background_array, points, voxel_sizes,
                 model_path, output_path, cube_size=50, batch_size=32, exp_id=None):
        super().__init__()
        self.signal_array = signal_array
        self.background_array = background_array
        self.points = points  # List of Cell objects
        self.voxel_sizes = voxel_sizes
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.exp_id = exp_id

    def run(self):
        try:
            from cellfinder.core.classify.classify import main as classify_main
            from brainglobe_utils.cells.cells import Cell
            from brainglobe_utils.IO.cells import save_cells

            print(f"\n[SCI-Connectome] Starting classification with Python API...")
            print(f"  Signal shape: {self.signal_array.shape}")
            print(f"  Candidates: {len(self.points)}")
            print(f"  Model: {self.model_path}")
            print(f"  Cube size: {self.cube_size}, Batch size: {self.batch_size}")

            # Ensure output directory exists
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Determine if model_path is weights (.h5) or full model (.keras)
            model_path = Path(self.model_path) if self.model_path else None
            trained_model = None
            model_weights = None

            if model_path:
                if model_path.suffix == '.keras':
                    trained_model = model_path
                elif model_path.suffix == '.h5':
                    model_weights = model_path
                else:
                    # Assume it's weights
                    model_weights = model_path

            # Network voxel sizes (what the model was trained on) - typically 5um isotropic
            network_voxel_sizes = (5, 2, 2)  # Z, Y, X in microns

            # Cube dimensions - model expects (height=50, width=50, depth=20)
            # The depth (Z) is smaller because Z resolution is typically lower
            cube_height = self.cube_size  # Y dimension
            cube_width = self.cube_size   # X dimension
            cube_depth = 20               # Z dimension - fixed to match trained model

            print(f"  Cube dimensions: height={cube_height}, width={cube_width}, depth={cube_depth}")

            # Run classification
            def progress_callback(batch_num):
                total_batches = len(self.points) // self.batch_size + 1
                print(f"  Classified batch {batch_num}/{total_batches}")

            classified_cells = classify_main(
                points=self.points,
                signal_array=self.signal_array,
                background_array=self.background_array,
                n_free_cpus=2,
                voxel_sizes=self.voxel_sizes,
                network_voxel_sizes=network_voxel_sizes,
                batch_size=self.batch_size,
                cube_height=cube_height,
                cube_width=cube_width,
                cube_depth=cube_depth,
                trained_model=trained_model,
                model_weights=model_weights,
                network_depth="50",
                max_workers=3,
                callback=progress_callback,
            )

            # Separate cells and rejected
            cells = [c for c in classified_cells if c.type == Cell.CELL]
            rejected = [c for c in classified_cells if c.type == Cell.UNKNOWN]

            cells_found = len(cells)
            rejected_count = len(rejected)

            print(f"[SCI-Connectome] Classification complete: {cells_found} cells, {rejected_count} rejected")

            # Save results
            if cells:
                save_cells(cells, self.output_path / "cells.xml")
            if rejected:
                save_cells(rejected, self.output_path / "rejected.xml")

            self.finished.emit(
                True,
                f"Classification complete: {cells_found} cells found, {rejected_count} rejected",
                cells_found,
                rejected_count,
                self.exp_id
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), 0, 0, self.exp_id)

    def _count_cells_in_xml(self, xml_path):
        """Count cells in XML file."""
        if not xml_path.exists():
            return 0
        try:
            with open(xml_path, 'r') as f:
                content = f.read()
            return content.count('<Marker>')
        except:
            return 0


class TuningWidget(QWidget):
    """Widget for tuning BrainGlobe settings."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_brain = None
        self.metadata = {}
        self.detection_worker = None
        self.brain_loader_worker = None
        self.last_run_id = None
        self.last_run_cells = 0
        self._loading_source = None  # Track source during async load
        self.view_mode = "cell_detection"  # 'cell_detection' or 'registration_qc'

        # Session tracking - track runs made during this napari session
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_run_ids = []  # Run IDs created during this session

        # Training data management - auto-save on every Y/N during curation
        self.training_data_dir = None  # Set when "Create Training Layers" is clicked
        self._cells_xml = None  # Path to auto-save XML for cells
        self._non_cells_xml = None  # Path to auto-save XML for non-cells

        # Classification selection for region counting
        self._selected_class_run = None
        self._selected_cells_xml = None

        # Initialize tracker if available
        self.tracker = ExperimentTracker() if TRACKER_AVAILABLE else None

        # Initialize session documenter for auto-documentation
        if SESSION_DOC_AVAILABLE:
            self.session_doc = SessionDocumenterWidget(self)
        else:
            self.session_doc = None

        self.setup_ui()

    def setup_ui(self):
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
        title = QLabel("BrainGlobe Setup & Tuning")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Figure out the right settings for your tissue")
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)

        # Brain selector
        brain_group = QGroupBox("Active Brain")
        brain_group_layout = QVBoxLayout()
        brain_group.setLayout(brain_group_layout)

        brain_layout = QHBoxLayout()
        brain_layout.addWidget(QLabel("Brain:"))
        self.brain_combo = QComboBox()
        self.brain_combo.currentTextChanged.connect(self.on_brain_changed)
        brain_layout.addWidget(self.brain_combo, stretch=1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_brains)
        brain_layout.addWidget(refresh_btn)
        brain_group_layout.addLayout(brain_layout)

        # Load brain button - the key friction reducer
        load_layout = QHBoxLayout()
        self.load_brain_btn = QPushButton("Load Brain into Napari")
        self.load_brain_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.load_brain_btn.clicked.connect(self.load_brain_into_napari)
        self.load_brain_btn.setToolTip("Load signal and background channels - uses Zarr if available!")
        load_layout.addWidget(self.load_brain_btn)

        # Convert to Zarr button - one-time conversion for instant future loads
        self.convert_zarr_btn = QPushButton("Convert to Zarr")
        self.convert_zarr_btn.setStyleSheet("background-color: #9C27B0; color: white;")
        self.convert_zarr_btn.clicked.connect(self.convert_to_zarr)
        self.convert_zarr_btn.setToolTip("One-time conversion - future loads will be instant!")
        load_layout.addWidget(self.convert_zarr_btn)

        brain_group_layout.addLayout(load_layout)

        layout.addWidget(brain_group)

        # View mode selector
        mode_group = QGroupBox("Workflow Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        mode_label = QLabel("Choose your workflow:")
        mode_layout.addWidget(mode_label)

        mode_button_layout = QVBoxLayout()
        self.mode_group = QButtonGroup()

        self.mode_cell_detection = QRadioButton("Cell Detection & Tuning (default)")
        self.mode_cell_detection.setChecked(True)
        self.mode_cell_detection.setToolTip("Tune detection parameters on full-resolution brain")
        self.mode_cell_detection.toggled.connect(self._on_mode_changed)
        self.mode_group.addButton(self.mode_cell_detection, 0)
        mode_button_layout.addWidget(self.mode_cell_detection)

        self.mode_registration = QRadioButton("Registration QC & Approval")
        self.mode_registration.setToolTip("Verify registration quality before proceeding")
        self.mode_registration.toggled.connect(self._on_mode_changed)
        self.mode_group.addButton(self.mode_registration, 1)
        mode_button_layout.addWidget(self.mode_registration)

        mode_layout.addLayout(mode_button_layout)
        layout.addWidget(mode_group)

        # Load options - what additional data to load with the brain (for cell detection mode)
        load_options_group = QGroupBox("Load Options (with brain)")
        load_options_layout = QVBoxLayout()
        load_options_group.setLayout(load_options_layout)

        # Note: Registration boundaries are now handled in "Registration QC & Approval" mode
        # No longer loaded in cell detection mode

        # Detection loading options
        det_label = QLabel("<b>Detection:</b>")
        load_options_layout.addWidget(det_label)

        self.load_best_detection_cb = QCheckBox("Best detection (highest-rated)")
        self.load_best_detection_cb.setChecked(False)
        self.load_best_detection_cb.setToolTip("Load detection points from the highest-rated experiment")
        load_options_layout.addWidget(self.load_best_detection_cb)

        self.load_recent_detection_cb = QCheckBox("Most recent detection")
        self.load_recent_detection_cb.setChecked(False)
        self.load_recent_detection_cb.setToolTip("Load detection points from the most recent experiment")
        load_options_layout.addWidget(self.load_recent_detection_cb)

        # Classification loading options
        class_label = QLabel("<b>Classification:</b>")
        load_options_layout.addWidget(class_label)

        self.load_best_classification_cb = QCheckBox("Best classification (highest-rated)")
        self.load_best_classification_cb.setChecked(True)
        self.load_best_classification_cb.setToolTip("Load cells.xml + rejected.xml from best classification run")
        load_options_layout.addWidget(self.load_best_classification_cb)

        self.load_recent_classification_cb = QCheckBox("Most recent classification")
        self.load_recent_classification_cb.setChecked(False)
        self.load_recent_classification_cb.setToolTip("Load cells.xml + rejected.xml from most recent classification")
        load_options_layout.addWidget(self.load_recent_classification_cb)

        # Color legend for cell markers - comprehensive legend for all layer types
        legend_label = QLabel(
            "<b>Detection:</b> "
            "<span style='color: #00FF00;'>Best</span> | "
            "<span style='color: #FFA500;'>Recent</span> | "
            "<span style='color: #FF0080;'>Only-Best</span> | "
            "<span style='color: #00BFFF;'>Only-Recent</span><br>"
            "<b>Classification:</b> "
            "<span style='color: #98FB98;'>Prev-Cells</span> | "
            "<span style='color: #DDA0DD;'>Prev-Rej</span> | "
            "<span style='color: #00CED1;'>New-Cells</span> | "
            "<span style='color: #FF69B4;'>New-Rej</span><br>"
            "<b>Diff:</b> "
            "<span style='color: #7FFF00;'>Gained</span> | "
            "<span style='color: #FF4500;'>Lost</span> | "
            "<b>Training:</b> "
            "<span style='color: #FFD700;'>Cells</span> | "
            "<span style='color: #8B0000;'>Non-Cells</span>"
        )
        legend_label.setToolTip(
            "DETECTION TUNING:\n"
            "  Green = Best detection (user-marked)\n"
            "  Orange = Most recent detection\n"
            "  Hot Pink = Only in Best (diff)\n"
            "  Sky Blue = Only in Recent (diff)\n\n"
            "CLASSIFICATION:\n"
            "  Pale Green = Previous model cells\n"
            "  Plum = Previous model rejected\n"
            "  Turquoise = New model cells\n"
            "  Hot Pink = New model rejected\n\n"
            "DIFFERENCES:\n"
            "  Chartreuse = Gained (new found, prev missed)\n"
            "  Orange-Red = Lost (prev found, new missed)\n\n"
            "TRAINING:\n"
            "  Gold = Training cells bucket\n"
            "  Dark Red = Training non-cells bucket\n\n"
            "Gray = Archived layers"
        )
        load_options_layout.addWidget(legend_label)

        brain_group_layout.addWidget(load_options_group)

        # Status display
        self.brain_status_label = QLabel("No brain loaded")
        self.brain_status_label.setStyleSheet("color: gray; font-style: italic;")
        brain_group_layout.addWidget(self.brain_status_label)

        # Session controls
        if SESSION_DOC_AVAILABLE:
            session_layout = QHBoxLayout()

            add_note_btn = QPushButton("Add Note")
            add_note_btn.setToolTip("Add a note to the current session")
            add_note_btn.clicked.connect(self.add_session_note)
            session_layout.addWidget(add_note_btn)

            view_log_btn = QPushButton("View Log")
            view_log_btn.setToolTip("Open the live session log file")
            view_log_btn.clicked.connect(self.view_session_log)
            session_layout.addWidget(view_log_btn)

            end_session_btn = QPushButton("End Session")
            end_session_btn.setToolTip("End session and generate report")
            end_session_btn.setStyleSheet("background-color: #FF9800; color: white;")
            end_session_btn.clicked.connect(self.end_session_and_report)
            session_layout.addWidget(end_session_btn)

            brain_group_layout.addLayout(session_layout)

        layout.addWidget(brain_group)

        # Tabs for different tuning tasks
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # =================================================================
        # TABS - Sequential workflow from setup to results
        # =================================================================

        # Tab 1: Setup (Voxel + Orientation)
        self.tabs.addTab(self.create_setup_tab(), "1. Setup")

        # Tab 2: Cropping
        self.tabs.addTab(self.create_crop_tab(), "2. Crop")

        # Tab 3: Registration QC
        self.tabs.addTab(self.create_qc_tab(), "3. Registration")

        # Tab 4: Detection Tuning
        self.tabs.addTab(self.create_detection_tab(), "4. Detection")

        # Tab 5: Detection Comparison (NEW)
        self.tabs.addTab(self.create_detection_compare_tab(), "5. Det Compare")

        # Tab 6: Classification
        self.tabs.addTab(self.create_classification_tab(), "6. Classify")

        # Tab 7: Classification Comparison (NEW)
        self.tabs.addTab(self.create_classification_compare_tab(), "7. Class Compare")

        # Tab 8: Curation & Training
        self.tabs.addTab(self.create_curation_tab(), "8. Curate/Train")

        # Tab 9: Results
        self.tabs.addTab(self.create_results_tab(), "9. Results")

        # Auto-refresh when switching tabs
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Save button
        save_btn = QPushButton("Save Settings to Metadata")
        save_btn.clicked.connect(self.save_metadata)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        layout.addWidget(save_btn)

        self.refresh_brains()

    def _on_tab_changed(self, index):
        """Handle tab changes - auto-refresh data for certain tabs."""
        # Tab 9 (Results) is index 8 - auto-refresh the results views
        if index == 8:
            self._refresh_all_results_views()

    # ==========================================================================
    # TAB CREATION METHODS
    # ==========================================================================

    def create_setup_tab(self):
        """Tab 1: Setup - Voxel size and orientation combined."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        info = QLabel(
            "Configure physical parameters for this brain before processing."
        )
        info.setWordWrap(True)
        content_layout.addWidget(info)

        # =================================================================
        # VOXEL SIZE SECTION
        # =================================================================
        voxel_group = QGroupBox("Voxel Size")
        voxel_layout = QVBoxLayout()
        voxel_group.setLayout(voxel_layout)

        voxel_info = QLabel(
            "Voxel size tells BrainGlobe the physical size of each pixel.\n"
            "Calculated from magnification and camera pixel size."
        )
        voxel_info.setWordWrap(True)
        voxel_layout.addWidget(voxel_info)

        # Calculator
        calc_layout = QFormLayout()

        self.camera_pixel = QDoubleSpinBox()
        self.camera_pixel.setRange(0.1, 100)
        self.camera_pixel.setValue(6.5)
        self.camera_pixel.setSuffix(" µm")
        self.camera_pixel.setToolTip("Physical size of camera sensor pixels (Andor Neo/Zyla = 6.5µm)")
        self.camera_pixel.valueChanged.connect(self.calculate_voxel)
        calc_layout.addRow("Camera pixel size:", self.camera_pixel)

        self.magnification = QDoubleSpinBox()
        self.magnification.setRange(0.1, 100)
        self.magnification.setValue(1.625)
        self.magnification.setSuffix("x")
        self.magnification.setToolTip("Objective magnification from filename (e.g., 1.625x)")
        self.magnification.valueChanged.connect(self.calculate_voxel)
        calc_layout.addRow("Magnification:", self.magnification)

        self.z_step = QDoubleSpinBox()
        self.z_step.setRange(0.1, 100)
        self.z_step.setValue(4.0)
        self.z_step.setSuffix(" µm")
        self.z_step.setToolTip("Z-step from filename (e.g., z4 = 4µm)")
        self.z_step.valueChanged.connect(self.calculate_voxel)
        calc_layout.addRow("Z-step:", self.z_step)

        voxel_layout.addLayout(calc_layout)

        # Result
        result_layout = QFormLayout()
        self.voxel_xy_label = QLabel("4.0 µm")
        self.voxel_xy_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout.addRow("XY (lateral):", self.voxel_xy_label)

        self.voxel_z_label = QLabel("4.0 µm")
        self.voxel_z_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout.addRow("Z (axial):", self.voxel_z_label)

        voxel_layout.addLayout(result_layout)

        parse_btn = QPushButton("Parse from Filename")
        parse_btn.clicked.connect(self.parse_voxel_from_filename)
        voxel_layout.addWidget(parse_btn)

        content_layout.addWidget(voxel_group)

        # =================================================================
        # ORIENTATION SECTION
        # =================================================================
        orient_group = QGroupBox("Orientation")
        orient_layout = QVBoxLayout()
        orient_group.setLayout(orient_layout)

        orient_info = QLabel(
            "Orientation tells BrainGlobe which way your brain is facing.\n"
            "Use 3-letter codes: i=inferior, s=superior, a=anterior, p=posterior, l=left, r=right"
        )
        orient_info.setWordWrap(True)
        orient_layout.addWidget(orient_info)

        self.orientation_combo = QComboBox()
        orientations = [
            ("iar", "Inferior-Anterior-Right (ventral side up, nose forward)"),
            ("sal", "Superior-Anterior-Left (dorsal side up, nose forward)"),
            ("asr", "Anterior-Superior-Right (nose up, dorsal forward)"),
            ("pir", "Posterior-Inferior-Right (tail up, ventral forward)"),
            ("custom", "Custom..."),
        ]
        for code, desc in orientations:
            self.orientation_combo.addItem(f"{code} - {desc}", code)
        self.orientation_combo.currentIndexChanged.connect(self.on_orientation_changed)
        orient_layout.addWidget(self.orientation_combo)

        self.custom_orientation = QLineEdit()
        self.custom_orientation.setPlaceholderText("e.g., iar")
        self.custom_orientation.setMaxLength(3)
        self.custom_orientation.setEnabled(False)
        orient_layout.addWidget(self.custom_orientation)

        test_btn = QPushButton("Load Sample Slice to Check")
        test_btn.clicked.connect(self.load_sample_slice)
        orient_layout.addWidget(test_btn)

        content_layout.addWidget(orient_group)

        content_layout.addStretch()
        return widget

    def create_voxel_tab(self):
        """Tab for voxel size calibration (LEGACY - now part of Setup tab)."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Info
        info = QLabel(
            "Voxel size tells BrainGlobe the physical size of each pixel.\n"
            "This is calculated from your magnification and camera pixel size."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Calculator
        calc_group = QGroupBox("Voxel Calculator")
        calc_layout = QFormLayout()
        calc_group.setLayout(calc_layout)

        self.camera_pixel = QDoubleSpinBox()
        self.camera_pixel.setRange(0.1, 100)
        self.camera_pixel.setValue(6.5)
        self.camera_pixel.setSuffix(" µm")
        self.camera_pixel.setToolTip("Physical size of camera sensor pixels (Andor Neo/Zyla = 6.5µm)")
        self.camera_pixel.valueChanged.connect(self.calculate_voxel)
        calc_layout.addRow("Camera pixel size:", self.camera_pixel)

        self.magnification = QDoubleSpinBox()
        self.magnification.setRange(0.1, 100)
        self.magnification.setValue(1.625)
        self.magnification.setSuffix("x")
        self.magnification.setToolTip("Objective magnification from filename (e.g., 1.625x)")
        self.magnification.valueChanged.connect(self.calculate_voxel)
        calc_layout.addRow("Magnification:", self.magnification)

        self.z_step = QDoubleSpinBox()
        self.z_step.setRange(0.1, 100)
        self.z_step.setValue(4.0)
        self.z_step.setSuffix(" µm")
        self.z_step.setToolTip("Z-step from filename (e.g., z4 = 4µm)")
        self.z_step.valueChanged.connect(self.calculate_voxel)
        calc_layout.addRow("Z-step:", self.z_step)

        layout.addWidget(calc_group)

        # Result
        result_group = QGroupBox("Calculated Voxel Size")
        result_layout = QFormLayout()
        result_group.setLayout(result_layout)

        self.voxel_xy_label = QLabel("4.0 µm")
        self.voxel_xy_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout.addRow("XY (lateral):", self.voxel_xy_label)

        self.voxel_z_label = QLabel("4.0 µm")
        self.voxel_z_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout.addRow("Z (axial):", self.voxel_z_label)

        layout.addWidget(result_group)

        # Parse from filename
        parse_btn = QPushButton("Parse from Filename")
        parse_btn.clicked.connect(self.parse_voxel_from_filename)
        layout.addWidget(parse_btn)

        layout.addStretch()
        return widget

    def calculate_voxel(self):
        """Calculate voxel size from inputs."""
        camera = self.camera_pixel.value()
        mag = self.magnification.value()
        z = self.z_step.value()

        xy = camera / mag
        self.voxel_xy_label.setText(f"{xy:.2f} µm")
        self.voxel_z_label.setText(f"{z:.2f} µm")

        self.metadata['voxel_size_um'] = {
            'x': round(xy, 3),
            'y': round(xy, 3),
            'z': round(z, 3),
        }

    def parse_voxel_from_filename(self):
        """Try to parse magnification and z-step from filename."""
        print("[DEBUG] Button clicked: parse_voxel_from_filename")
        if not self.current_brain:
            return

        name = self.current_brain.name
        # Look for patterns like 1.625x or 1p625x
        import re

        # Magnification
        mag_match = re.search(r'(\d+[p.]\d+)x', name)
        if mag_match:
            mag_str = mag_match.group(1).replace('p', '.')
            self.magnification.setValue(float(mag_str))

        # Z-step
        z_match = re.search(r'z(\d+(?:[p.]\d+)?)', name)
        if z_match:
            z_str = z_match.group(1).replace('p', '.')
            self.z_step.setValue(float(z_str))

    def create_orientation_tab(self):
        """Tab for orientation settings."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "Orientation tells BrainGlobe which way your brain is facing.\n"
            "Use 3-letter codes: first letter = what's at index 0 of each axis.\n\n"
            "Letters: i=inferior, s=superior, a=anterior, p=posterior, l=left, r=right"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Common orientations
        orient_group = QGroupBox("Common Orientations")
        orient_layout = QVBoxLayout()
        orient_group.setLayout(orient_layout)

        self.orientation_combo = QComboBox()
        orientations = [
            ("iar", "Inferior-Anterior-Right (ventral side up, nose forward)"),
            ("sal", "Superior-Anterior-Left (dorsal side up, nose forward)"),
            ("asr", "Anterior-Superior-Right (nose up, dorsal forward)"),
            ("pir", "Posterior-Inferior-Right (tail up, ventral forward)"),
            ("custom", "Custom..."),
        ]
        for code, desc in orientations:
            self.orientation_combo.addItem(f"{code} - {desc}", code)
        self.orientation_combo.currentIndexChanged.connect(self.on_orientation_changed)
        orient_layout.addWidget(self.orientation_combo)

        self.custom_orientation = QLineEdit()
        self.custom_orientation.setPlaceholderText("e.g., iar")
        self.custom_orientation.setMaxLength(3)
        self.custom_orientation.setEnabled(False)
        orient_layout.addWidget(self.custom_orientation)

        layout.addWidget(orient_group)

        # Visual guide
        guide_group = QGroupBox("How to Determine Orientation")
        guide_layout = QVBoxLayout()
        guide_group.setLayout(guide_layout)

        guide_text = QTextEdit()
        guide_text.setReadOnly(True)
        guide_text.setHtml("""
        <h4>Look at your first Z-slice (Z=0):</h4>
        <ol>
            <li><b>First letter:</b> What's at low Z? (inferior/superior)</li>
            <li><b>Second letter:</b> What's at low Y? (anterior/posterior)</li>
            <li><b>Third letter:</b> What's at low X? (left/right)</li>
        </ol>

        <h4>Tips:</h4>
        <ul>
            <li>Load a slice in napari and look at the anatomy</li>
            <li>Olfactory bulbs = anterior</li>
            <li>Cerebellum = posterior</li>
            <li>If brain appears upside-down after registration, flip i↔s</li>
        </ul>
        """)
        guide_layout.addWidget(guide_text)

        layout.addWidget(guide_group)

        # Test button
        test_btn = QPushButton("Load Sample Slice to Check")
        test_btn.clicked.connect(self.load_sample_slice)
        layout.addWidget(test_btn)

        layout.addStretch()
        return widget

    def on_orientation_changed(self, index):
        """Handle orientation selection."""
        code = self.orientation_combo.currentData()
        self.custom_orientation.setEnabled(code == "custom")
        if code != "custom":
            self.metadata['orientation'] = code

    def load_sample_slice(self):
        """Load a sample slice to check orientation."""
        print("[DEBUG] Button clicked: load_sample_slice")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        # Try to find extracted data (priority order for sample slice)
        extracted = self.current_brain / "1_Extracted_Full" / "ch0"
        if not extracted.exists():
            extracted = self.current_brain / "2_Cropped_For_Registration_Manual" / "ch0"
        if not extracted.exists():
            extracted = self.current_brain / "2_Cropped_For_Registration" / "ch0"

        if not extracted.exists():
            QMessageBox.warning(self, "Error", "No extracted data found. Run extraction first.")
            return

        # Load middle slice
        slices = sorted(extracted.glob("Z*.tif"))
        if not slices:
            return

        middle = slices[len(slices) // 2]

        try:
            import tifffile
            img = tifffile.imread(middle)
            self.viewer.add_image(img, name=f"Sample: {middle.name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load slice: {e}")

    def create_crop_tab(self):
        """Tab for crop optimization."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "Cropping removes spinal cord to improve registration.\n"
            "Adjust the automatic detection or set manual crop boundaries."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Auto-detection settings
        auto_group = QGroupBox("Automatic Detection Settings")
        auto_layout = QFormLayout()
        auto_group.setLayout(auto_layout)

        self.width_threshold = QDoubleSpinBox()
        self.width_threshold.setRange(0.1, 0.9)
        self.width_threshold.setValue(0.25)
        self.width_threshold.setSingleStep(0.05)
        self.width_threshold.setToolTip("Cord is detected where width < this fraction of max brain width")
        auto_layout.addRow("Width threshold:", self.width_threshold)

        self.margin_slices = QSpinBox()
        self.margin_slices.setRange(0, 100)
        self.margin_slices.setValue(10)
        self.margin_slices.setToolTip("Extra slices to keep above detected crop point")
        auto_layout.addRow("Safety margin (slices):", self.margin_slices)

        layout.addWidget(auto_group)

        # Manual override
        manual_group = QGroupBox("Manual Override")
        manual_layout = QFormLayout()
        manual_group.setLayout(manual_layout)

        self.use_manual_crop = QCheckBox("Use manual Y crop")
        manual_layout.addRow(self.use_manual_crop)

        self.manual_y_max = QSpinBox()
        self.manual_y_max.setRange(0, 10000)
        self.manual_y_max.setValue(0)
        self.manual_y_max.setToolTip("Maximum Y index to keep (0 = use auto)")
        manual_layout.addRow("Crop Y at:", self.manual_y_max)

        layout.addWidget(manual_group)

        # Preview
        preview_btn = QPushButton("Open Manual Crop Tool")
        preview_btn.clicked.connect(self.open_manual_crop)
        layout.addWidget(preview_btn)

        optimize_btn = QPushButton("Run Crop Optimization")
        optimize_btn.clicked.connect(self.run_crop_optimization)
        layout.addWidget(optimize_btn)

        layout.addStretch()
        return widget

    def open_manual_crop(self):
        """Open the manual crop widget."""
        print("[DEBUG] Button clicked: open_manual_crop")
        # The ManualCropWidget should be available
        QMessageBox.information(
            self, "Manual Crop",
            "Use Plugins > Manual Crop to interactively set crop boundaries.\n\n"
            "The tool will show you a max projection and let you drag a line."
        )

    def run_crop_optimization(self):
        """Run the crop optimization script."""
        print("[DEBUG] Button clicked: run_crop_optimization")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        script = SCRIPTS_DIR / "util_optimize_crop.py"
        if script.exists():
            subprocess.Popen([sys.executable, str(script), "--brain", self.current_brain.name])
        else:
            QMessageBox.warning(self, "Error", "Crop optimization script not found")

    def create_detection_tab(self):
        """Tab for detection parameter tuning - ALL cellfinder parameters exposed."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "Test detection parameters on a small region before running on the whole brain.\n"
            "ALL cellfinder parameters are exposed - experiment to find what works for YOUR tissue!"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Parameter presets (quick starting points)
        preset_group = QGroupBox("Quick Start Presets (click to load)")
        preset_layout = QVBoxLayout()
        preset_group.setLayout(preset_layout)

        preset_btn_layout = QHBoxLayout()
        for preset_name, tooltip in [
            ("proven", "YOUR tested settings - ball_xy=10, ball_z=10, soma=10, thresh=8, spread=1.0"),
            ("sensitive", "Low thresholds, catches more - ball_xy=4, ball_z=10, soma=12, thresh=8"),
            ("balanced", "Default settings - ball_xy=6, ball_z=15, soma=16, thresh=10"),
            ("conservative", "High thresholds, fewer FPs - ball_xy=8, ball_z=20, soma=20, thresh=12"),
        ]:
            btn = QPushButton(preset_name)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, p=preset_name: self.load_preset(p))
            preset_btn_layout.addWidget(btn)
        preset_layout.addLayout(preset_btn_layout)
        layout.addWidget(preset_group)

        # ALL detection parameters - organized by category
        params_group = QGroupBox("Detection Parameters (ALL cellfinder options)")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)

        # --- Core Ball Filter Parameters ---
        params_layout.addRow(QLabel("<b>Ball Filter (core detection):</b>"))

        self.ball_xy = QDoubleSpinBox()
        self.ball_xy.setRange(1, 100)
        self.ball_xy.setValue(6)
        self.ball_xy.setSuffix(" µm")
        self.ball_xy.setToolTip("Lateral (XY) size of the spherical filter kernel in microns")
        params_layout.addRow("Ball XY size:", self.ball_xy)

        self.ball_z = QDoubleSpinBox()
        self.ball_z.setRange(1, 100)
        self.ball_z.setValue(15)
        self.ball_z.setSuffix(" µm")
        self.ball_z.setToolTip("Axial (Z) size of the spherical filter kernel in microns")
        params_layout.addRow("Ball Z size:", self.ball_z)

        self.ball_overlap_fraction = QDoubleSpinBox()
        self.ball_overlap_fraction.setRange(0.1, 1.0)
        self.ball_overlap_fraction.setValue(0.6)
        self.ball_overlap_fraction.setSingleStep(0.05)
        self.ball_overlap_fraction.setToolTip("Fraction of ball filter overlap needed to retain a voxel (higher = stricter)")
        params_layout.addRow("Ball overlap fraction:", self.ball_overlap_fraction)

        # --- Soma Parameters ---
        params_layout.addRow(QLabel("<b>Soma (cell body):</b>"))

        self.soma_diameter = QDoubleSpinBox()
        self.soma_diameter.setRange(1, 200)
        self.soma_diameter.setValue(16)
        self.soma_diameter.setSuffix(" µm")
        self.soma_diameter.setToolTip("Expected in-plane soma diameter in microns")
        params_layout.addRow("Soma diameter:", self.soma_diameter)

        self.soma_spread_factor = QDoubleSpinBox()
        self.soma_spread_factor.setRange(0.5, 5.0)
        self.soma_spread_factor.setValue(1.4)
        self.soma_spread_factor.setSingleStep(0.1)
        self.soma_spread_factor.setToolTip("Cell spread factor for determining cluster split thresholds")
        params_layout.addRow("Soma spread factor:", self.soma_spread_factor)

        # --- Thresholding Parameters ---
        params_layout.addRow(QLabel("<b>Intensity Thresholding:</b>"))

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.5, 100)
        self.threshold.setValue(10)
        self.threshold.setToolTip("Per-plane intensity threshold in standard deviations above mean (n_sds_above_mean_thresh)")
        params_layout.addRow("Threshold (SDs above mean):", self.threshold)

        self.tiled_threshold = QDoubleSpinBox()
        self.tiled_threshold.setRange(0.5, 100)
        self.tiled_threshold.setValue(10)
        self.tiled_threshold.setToolTip("Per-tile intensity threshold in standard deviations (n_sds_above_mean_tiled_thresh)")
        params_layout.addRow("Tiled threshold (SDs):", self.tiled_threshold)

        self.log_sigma_size = QDoubleSpinBox()
        self.log_sigma_size.setRange(0.01, 1.0)
        self.log_sigma_size.setValue(0.2)
        self.log_sigma_size.setSingleStep(0.05)
        self.log_sigma_size.setToolTip("How tight the detection filter is (0.1=tight/sharp, 0.5=loose/blurry). Expressed as fraction of soma diameter.")
        params_layout.addRow("Filter width (% of soma):", self.log_sigma_size)

        # --- Cluster Handling ---
        params_layout.addRow(QLabel("<b>Cluster Handling:</b>"))

        self.max_cluster_size = QSpinBox()
        self.max_cluster_size.setRange(1000, 1000000000)  # Up to 1 billion
        self.max_cluster_size.setValue(100000)
        self.max_cluster_size.setSuffix(" µm³")
        self.max_cluster_size.setToolTip("Maximum cluster volume before labeling as artifact (cubic microns)")
        params_layout.addRow("Max cluster size:", self.max_cluster_size)

        # NOTE: Cluster splitting is handled internally by cellfinder and cannot be
        # configured through the Python API. The splitting parameters shown in cellfinder
        # output are internal algorithm behavior, not user-configurable settings.

        # --- Channel Options ---
        params_layout.addRow(QLabel("<b>Channels:</b>"))

        self.swap_channels = QCheckBox("Swap signal/background channels")
        self.swap_channels.setToolTip("Check if ch0/ch1 are reversed (signal in ch1, background in ch0)")
        self.swap_channels.setChecked(True)  # Default ON since channels got swapped during crop
        params_layout.addRow(self.swap_channels)

        # Button to actually swap the loaded layers right now
        swap_btn_layout = QHBoxLayout()
        self.swap_layers_btn = QPushButton("Swap Layers Now")
        self.swap_layers_btn.setToolTip("Swap the Signal and Background layer names/colors in napari")
        self.swap_layers_btn.setStyleSheet("background-color: #FF9800; color: white;")
        self.swap_layers_btn.clicked.connect(self._swap_loaded_layers)
        swap_btn_layout.addWidget(self.swap_layers_btn)
        swap_btn_layout.addStretch()
        params_layout.addRow("", swap_btn_layout)

        # --- Performance Options ---
        params_layout.addRow(QLabel("<b>Performance:</b>"))

        self.use_gpu = QCheckBox("Use GPU (CUDA)")
        self.use_gpu.setToolTip("Use GPU acceleration if available")
        self.use_gpu.setChecked(False)
        params_layout.addRow(self.use_gpu)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 100)
        self.batch_size.setValue(0)
        self.batch_size.setSpecialValueText("Auto")
        self.batch_size.setToolTip("Number of planes to process simultaneously (0 = auto)")
        params_layout.addRow("Batch size:", self.batch_size)

        self.n_free_cpus = QSpinBox()
        self.n_free_cpus.setRange(1, 80)
        self.n_free_cpus.setValue(2)  # Default: leave 2 CPUs free (uses 74 on 76-core machine)
        self.n_free_cpus.setToolTip("Number of CPUs to leave free (cellfinder uses the rest)")
        params_layout.addRow("CPUs to leave free:", self.n_free_cpus)

        layout.addWidget(params_group)

        # Connect all detection parameters to session tracking
        self._connect_param_tracking()

        # Detection scope
        test_group = QGroupBox("Run Detection")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)

        # Scope selection: Current slice, Z range, or Entire brain
        from qtpy.QtWidgets import QRadioButton, QButtonGroup
        scope_layout = QHBoxLayout()

        self.scope_group = QButtonGroup(self)
        self.scope_current = QRadioButton("Current ±25 slices")
        self.scope_current.setToolTip("Run on current napari Z position ± 25 slices")
        self.scope_range = QRadioButton("Z Range")
        self.scope_range.setToolTip("Run on a specific Z range")
        self.scope_all = QRadioButton("Entire Brain")
        self.scope_all.setToolTip("Run on the entire brain (can take a long time)")

        self.scope_group.addButton(self.scope_current, 0)
        self.scope_group.addButton(self.scope_range, 1)
        self.scope_group.addButton(self.scope_all, 2)
        self.scope_current.setChecked(True)  # Default to current slice

        scope_layout.addWidget(self.scope_current)
        scope_layout.addWidget(self.scope_range)
        scope_layout.addWidget(self.scope_all)
        scope_layout.addStretch()
        test_layout.addLayout(scope_layout)

        # Z range input (only visible when "Z Range" is selected)
        self.z_range_widget = QWidget()
        region_layout = QHBoxLayout(self.z_range_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(QLabel("Z range:"))
        self.test_z_start = QSpinBox()
        self.test_z_start.setRange(0, 10000)
        self.test_z_start.setToolTip("Start Z slice")
        region_layout.addWidget(self.test_z_start)
        region_layout.addWidget(QLabel("to"))
        self.test_z_end = QSpinBox()
        self.test_z_end.setRange(0, 10000)
        self.test_z_end.setValue(50)
        self.test_z_end.setToolTip("End Z slice")
        region_layout.addWidget(self.test_z_end)
        region_layout.addStretch()
        test_layout.addWidget(self.z_range_widget)
        self.z_range_widget.hide()  # Hidden until "Z Range" selected

        # Show/hide Z range based on selection
        self.scope_range.toggled.connect(lambda checked: self.z_range_widget.setVisible(checked))

        # Slice count label
        self.z_range_label = QLabel("(~50 slices around current position)")
        self.z_range_label.setStyleSheet("color: gray;")
        self.test_z_start.valueChanged.connect(self.update_z_range_label)
        self.test_z_end.valueChanged.connect(self.update_z_range_label)
        self.scope_group.buttonClicked.connect(self._update_scope_label)
        test_layout.addWidget(self.z_range_label)

        self.test_detection_btn = QPushButton("Run Detection")
        self.test_detection_btn.clicked.connect(self.run_test_detection)
        self.test_detection_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        test_layout.addWidget(self.test_detection_btn)

        layout.addWidget(test_group)

        # =================================================================
        # CONTEXT PANEL - Calibration Runs for Current Brain
        # =================================================================
        context_group = QGroupBox("Calibration Runs for This Brain")
        context_layout = QVBoxLayout()
        context_group.setLayout(context_layout)

        # Quick filters row
        filter_row = QHBoxLayout()
        self.filter_best_only = QCheckBox("Best only")
        self.filter_best_only.setToolTip("Show only runs marked as best")
        self.filter_best_only.stateChanged.connect(self._refresh_context_runs)
        filter_row.addWidget(self.filter_best_only)

        self.filter_this_session = QCheckBox("This session")
        self.filter_this_session.setToolTip("Show only runs from this napari session")
        self.filter_this_session.stateChanged.connect(self._refresh_context_runs)
        filter_row.addWidget(self.filter_this_session)

        filter_row.addStretch()

        refresh_runs_btn = QPushButton("Refresh")
        refresh_runs_btn.setToolTip("Refresh the runs list")
        refresh_runs_btn.clicked.connect(self._refresh_context_runs)
        filter_row.addWidget(refresh_runs_btn)

        context_layout.addLayout(filter_row)

        # Runs list (simpler than table, works better with napari theme)
        from qtpy.QtWidgets import QListWidget, QListWidgetItem
        self.runs_list = QListWidget()
        self.runs_list.setMinimumHeight(120)
        self.runs_list.setMaximumHeight(200)
        self.runs_list.setSelectionMode(QListWidget.SingleSelection)
        self.runs_list.setToolTip("★ = Best marked | ● = This session\nDouble-click to load a run")
        self.runs_list.setStyleSheet("""
            QListWidget {
                background-color: #363636;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """)
        self.runs_list.itemDoubleClicked.connect(self._load_run_from_list)
        context_layout.addWidget(self.runs_list)

        # Keep runs_table as alias for compatibility
        self.runs_table = self.runs_list

        # Action buttons row
        action_row = QHBoxLayout()
        self.load_selected_btn = QPushButton("Load Selected")
        self.load_selected_btn.setToolTip("Load the selected run's detection results")
        self.load_selected_btn.clicked.connect(self._load_selected_run)
        action_row.addWidget(self.load_selected_btn)

        self.compare_selected_btn = QPushButton("Compare")
        self.compare_selected_btn.setToolTip("Compare selected run with current best")
        self.compare_selected_btn.clicked.connect(self._compare_with_selected)
        action_row.addWidget(self.compare_selected_btn)

        self.mark_best_btn = QPushButton("Mark as Best")
        self.mark_best_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.mark_best_btn.setToolTip("Mark the selected run as the best for this brain")
        self.mark_best_btn.clicked.connect(self._mark_selected_as_best)
        action_row.addWidget(self.mark_best_btn)

        context_layout.addLayout(action_row)

        # Paradigm best row
        paradigm_row = QHBoxLayout()

        self.mark_paradigm_best_btn = QPushButton("Mark as Paradigm Best")
        self.mark_paradigm_best_btn.setStyleSheet("background-color: #9C27B0; color: white;")
        self.mark_paradigm_best_btn.setToolTip(
            "Mark as best for ALL brains with same imaging paradigm.\n"
            "Future brains with matching parameters will use these settings."
        )
        self.mark_paradigm_best_btn.clicked.connect(self._mark_selected_as_paradigm_best)
        paradigm_row.addWidget(self.mark_paradigm_best_btn)

        self.paradigm_status_label = QLabel("")
        self.paradigm_status_label.setStyleSheet("color: #9C27B0; font-style: italic;")
        paradigm_row.addWidget(self.paradigm_status_label)

        context_layout.addLayout(paradigm_row)

        layout.addWidget(context_group)

        layout.addStretch()
        return widget

    def _refresh_context_runs(self):
        """Populate runs list with smart prioritization: ★ Best → ● This session → Recent."""
        print("[DEBUG] Button clicked: _refresh_context_runs")
        from qtpy.QtWidgets import QListWidgetItem

        self.runs_list.clear()
        # Also clear compare list if it exists
        if hasattr(self, 'compare_runs_list'):
            self.compare_runs_list.clear()

        if not self.tracker or not self.current_brain:
            return

        # Get ALL completed detection runs for this brain
        brain_name = self.current_brain.name
        all_runs = self.tracker.search(
            brain=brain_name,
            exp_type="detection"
        )

        # Filter to completed runs only
        completed = [r for r in all_runs if r.get('status') == 'completed']

        # Get all classification runs to check which detections have been classified
        all_classifications = self.tracker.search(exp_type="classification", status="completed")
        classified_detections = set(r.get('parent_experiment') for r in all_classifications)

        # Apply filters
        if self.filter_best_only.isChecked():
            completed = [r for r in completed if r.get('marked_best', False)]
        if self.filter_this_session.isChecked():
            completed = [r for r in completed if r.get('exp_id') in self.session_run_ids]

        # Sort by priority: Best marked → Same session → Created time (newest first)
        def priority_key(run):
            # Higher values sort first with reverse=True
            is_best = 2 if run.get('marked_best', False) else 0
            is_this_session = 1 if run.get('exp_id') in self.session_run_ids else 0
            created = run.get('created_at', '')  # ISO format, so string comparison works
            return (is_best, is_this_session, created)

        sorted_runs = sorted(completed, key=priority_key, reverse=True)

        # Populate list (limit to 50 for performance)
        for run in sorted_runs[:50]:
            exp_id = run.get('exp_id')

            # Build display text with icons
            icons = ""
            if run.get('marked_best', False):
                icons += "★ "
            if exp_id in self.session_run_ids:
                icons += "● "
            if exp_id in classified_detections:
                icons += "C "  # Has classification

            created = run.get('created_at', '')[:16]
            cells = run.get('det_cells_found', '?')
            preset = run.get('det_preset', 'custom') or 'custom'

            # Add scope indicator
            scope = run.get('det_scope', '')
            if scope == 'partial':
                z_start = run.get('det_z_start', '?')
                z_end = run.get('det_z_end', '?')
                scope_text = f"Z{z_start}-{z_end}"
            elif scope == 'full':
                scope_text = "FULL"
            else:
                # Legacy runs without scope tracking
                scope_text = ""

            if scope_text:
                display_text = f"{icons}{created}  |  {cells} cells  |  [{scope_text}]  |  {preset}"
            else:
                display_text = f"{icons}{created}  |  {cells} cells  |  {preset}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, run.get('exp_id'))  # Store exp_id
            self.runs_list.addItem(item)

            # Also add to compare list if it exists
            if hasattr(self, 'compare_runs_list'):
                compare_item = QListWidgetItem(display_text)
                compare_item.setData(Qt.UserRole, run.get('exp_id'))
                self.compare_runs_list.addItem(compare_item)

    def _load_run_from_list(self, item):
        """Load a run when double-clicked in the list."""
        exp_id = item.data(Qt.UserRole)
        if exp_id:
            self._load_historical_run(exp_id)

    def _get_selected_run_id(self):
        """Get the exp_id of the currently selected run in the list."""
        current = self.runs_list.currentItem()
        if current:
            return current.data(Qt.UserRole)
        return None

    def _load_run_from_table(self, item):
        """Legacy method - redirects to list handler."""
        self._load_run_from_list(item)

    def _load_selected_run(self):
        """Load the selected run's detection results into napari."""
        print("[DEBUG] Button clicked: _load_selected_run")
        exp_id = self._get_selected_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a run from the table first.")
            return

        self._load_historical_run(exp_id)

    def _get_compare_selected_run_id(self):
        """Get the exp_id of the selected run in the compare list."""
        if not hasattr(self, 'compare_runs_list'):
            return None
        current = self.compare_runs_list.currentItem()
        if current:
            return current.data(Qt.UserRole)
        return None

    def _load_selected_as_best(self):
        """Load selected run from compare list as the REFERENCE (best) for comparison."""
        print("[DEBUG] Button clicked: _load_selected_as_best")
        exp_id = self._get_compare_selected_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a run from the list first.")
            return
        self._load_historical_run(exp_id, cell_type='det_best')

    def _load_selected_as_recent(self):
        """Load selected run from compare list as the COMPARISON (recent) run."""
        print("[DEBUG] Button clicked: _load_selected_as_recent")
        exp_id = self._get_compare_selected_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a run from the list first.")
            return
        self._load_historical_run(exp_id, cell_type='det_recent')

    def _find_classification_for_detection(self, detection_exp_id: str) -> dict:
        """
        Find the most recent classification run for a given detection.

        The tracker links classification → detection via parent_experiment field.

        Args:
            detection_exp_id: The exp_id of the detection run

        Returns:
            Classification run dict if found, None otherwise
        """
        if not self.tracker:
            return None

        # Get all classification runs
        classifications = self.tracker.search(exp_type="classification", status="completed")

        # Find those linked to this detection
        linked = [r for r in classifications if r.get('parent_experiment') == detection_exp_id]

        if not linked:
            return None

        # Return the most recent one
        linked.sort(key=lambda r: r.get('created_at', ''), reverse=True)
        return linked[0]

    def _load_classification_for_detection(self, detection_exp_id: str, class_run: dict, cell_type=None):
        """
        Load classification results (cells.xml + rejected.xml) for a detection run.

        This is used by smart loading when we detect that classification has been
        run on a detection - we load the split layers instead of raw candidates.

        Args:
            detection_exp_id: The detection run ID (for metadata)
            class_run: The classification run dict from tracker
            cell_type: Optional styling override
        """
        import numpy as np
        from brainglobe_utils.IO.cells import get_cells

        output_path = class_run.get('output_path')
        if not output_path:
            print(f"[SCI-Connectome] Classification has no output_path")
            return

        output_path = Path(output_path)
        cells_xml = output_path / "cells.xml"
        rejected_xml = output_path / "rejected.xml"

        if not cells_xml.exists():
            print(f"[SCI-Connectome] cells.xml not found at {cells_xml}")
            return

        # Load cells
        try:
            cells = get_cells(str(cells_xml))
            cells_count = len(cells) if cells else 0

            if cells:
                points = np.array([[c.z, c.y, c.x] for c in cells])
                # Use 'cells' cell_type for classification results
                self._add_points_layer(points, f"Classified Cells ({cells_count})", cell_type='cells')
            else:
                self._add_points_layer(np.empty((0, 3)), "Classified Cells (0)", cell_type='cells')

        except Exception as e:
            print(f"[SCI-Connectome] Error loading cells.xml: {e}")
            return

        # Load rejected
        rejected_count = 0
        try:
            if rejected_xml.exists():
                rejected = get_cells(str(rejected_xml))
                rejected_count = len(rejected) if rejected else 0
                if rejected:
                    points = np.array([[c.z, c.y, c.x] for c in rejected])
                    self._add_points_layer(points, f"Rejected ({rejected_count})", cell_type='rejected')
                else:
                    self._add_points_layer(np.empty((0, 3)), "Rejected (0)", cell_type='rejected')
            else:
                self._add_points_layer(np.empty((0, 3)), "Rejected (0)", cell_type='rejected')

        except Exception as e:
            print(f"[SCI-Connectome] Error loading rejected.xml: {e}")
            self._add_points_layer(np.empty((0, 3)), "Rejected (0)", cell_type='rejected')

        # Store metadata linking to detection
        for layer in self.viewer.layers:
            if 'Classified' in layer.name or 'Rejected' in layer.name:
                if 'exp_id' not in layer.metadata:
                    layer.metadata['exp_id'] = detection_exp_id
                    layer.metadata['class_exp_id'] = class_run.get('exp_id')

        print(f"[SCI-Connectome] Loaded classification: {cells_count} cells, {rejected_count} rejected")

        # Update status if available
        if hasattr(self, 'classification_load_status'):
            self.classification_load_status.setText(
                f"Loaded: {cells_count} cells, {rejected_count} rejected"
            )

    def _load_historical_run(self, exp_id, cell_type=None, prefer_classification=True):
        """
        Load a specific calibration run by its ID.

        Smart Loading: If this is a detection run and classification exists for it,
        loads the classified results (cells.xml + rejected.xml) instead of raw detection.
        This provides the split layers that are more useful for analysis.

        Args:
            exp_id: The experiment ID to load
            cell_type: Optional styling override ('det_best', 'det_recent', etc.)
            prefer_classification: If True, load classification results when available
        """
        if not self.tracker:
            return

        # Get run details from tracker
        run = self.tracker.get_experiment(exp_id)
        if not run:
            QMessageBox.warning(self, "Not Found", f"Run {exp_id} not found in tracker.")
            return

        # Smart Loading: Check if classification exists for this detection
        exp_type = run.get('exp_type', 'detection')
        if prefer_classification and exp_type == 'detection':
            # First check tracker for linked classification
            class_run = self._find_classification_for_detection(exp_id)

            if class_run:
                # Classification exists in tracker - load it instead
                print(f"[SCI-Connectome] Found classification for detection {exp_id}")
                self._load_classification_for_detection(exp_id, class_run, cell_type)
                return

            # Fallback: Check filesystem even if not in tracker
            if self.current_brain:
                classified_folder = self.current_brain / "5_Classified_Cells"
                cells_xml = classified_folder / "cells.xml"
                if cells_xml.exists():
                    print(f"[SCI-Connectome] Found classification on filesystem (not tracked)")
                    self._load_classified_cells()  # Use existing method
                    return

        output_path = run.get('output_path')
        if not output_path or not Path(output_path).exists():
            QMessageBox.warning(self, "File Not Found", f"Output file not found: {output_path}")
            return

        # Handle both file and directory paths
        output_path = Path(output_path)
        if output_path.is_dir():
            # Look for detected_cells.xml first (our standard name)
            xml_path = output_path / "detected_cells.xml"
            if xml_path.exists():
                output_path = xml_path
                print(f"[SCI-Connectome] Found XML file: {output_path}")
            else:
                # Fall back to any XML file
                xml_files = list(output_path.glob("*.xml"))
                if not xml_files:
                    # This run was logged but XML was never saved (common for old/failed runs)
                    print(f"[SCI-Connectome] No XML files in: {output_path}")
                    print(f"[SCI-Connectome] This run may have been logged but detection results were not saved.")
                    QMessageBox.warning(
                        self, "Missing Data",
                        f"Detection results not found for this run.\n\n"
                        f"The run was logged but the XML file was not saved.\n"
                        f"This can happen with older runs or failed detections."
                    )
                    return
                # Use the most recent XML file
                output_path = sorted(xml_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                print(f"[SCI-Connectome] Found XML file: {output_path}")

        # Load points from XML
        try:
            from brainglobe_utils.IO.cells import get_cells
            cells = get_cells(str(output_path))
            if not cells:
                file_size = output_path.stat().st_size if output_path.exists() else 0
                QMessageBox.information(
                    self, "Empty",
                    f"No cells found in this run.\n\nFile: {output_path.name}\nSize: {file_size} bytes"
                )
                return

            # Convert to napari points format
            points = np.array([[c.z, c.y, c.x] for c in cells])

            # Determine styling based on cell_type override or automatic detection
            is_best = run.get('marked_best', False)
            is_this_session = exp_id in self.session_run_ids
            created = run.get('created_at', '')[:16]

            # If cell_type is explicitly provided, use it for styling (for comparison)
            if cell_type == 'det_best':
                layer_name = f"BEST: Run {created} ({len(cells)} cells)"
                face_color = 'transparent'
                edge_color = '#00FF00'  # Green for reference
                actual_cell_type = 'det_best'
            elif cell_type == 'det_recent':
                layer_name = f"RECENT: Run {created} ({len(cells)} cells)"
                face_color = 'transparent'
                edge_color = '#FFA500'  # Orange for comparison
                actual_cell_type = 'det_recent'
            else:
                # Auto-detect based on run properties
                prefix = "★ " if is_best else ("● " if is_this_session else "")
                layer_name = f"{prefix}Run {created} ({len(cells)} cells)"
                if is_best:
                    face_color = 'transparent'
                    edge_color = '#00FF00'  # Green for best
                    actual_cell_type = 'det_best'
                elif is_this_session:
                    face_color = 'transparent'
                    edge_color = '#00BFFF'  # Sky blue for this session
                    actual_cell_type = 'det_recent'
                else:
                    face_color = 'transparent'
                    edge_color = '#888888'  # Gray for old runs
                    actual_cell_type = None

            # Add points layer
            layer = self.viewer.add_points(
                points,
                name=layer_name,
                size=14,
                face_color=face_color,
                border_color=edge_color,
                border_width=0.1,
                opacity=0.7,
                symbol='o',
            )

            # Store metadata for layer identification
            layer.metadata['exp_id'] = exp_id
            layer.metadata['is_best'] = is_best
            layer.metadata['cell_type'] = actual_cell_type

            print(f"[SCI-Connectome] Loaded run {exp_id}: {len(cells)} cells")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load run: {e}")

    def _compare_with_selected(self):
        """Compare selected run with current best (placeholder)."""
        print("[DEBUG] Button clicked: _compare_with_selected")
        exp_id = self._get_selected_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a run to compare.")
            return

        # TODO: Implement proper comparison visualization
        # For now, just load the selected run - user can visually compare layers
        self._load_historical_run(exp_id)
        QMessageBox.information(
            self,
            "Compare",
            "Selected run loaded. You can now toggle layer visibility to compare visually."
        )

    def _mark_selected_as_best(self):
        """Mark the selected run as the best for this brain."""
        print("[DEBUG] Button clicked: _mark_selected_as_best")
        exp_id = self._get_selected_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a run to mark as best.")
            return

        if not self.tracker:
            QMessageBox.warning(self, "No Tracker", "Experiment tracker not available.")
            return

        # Mark as best in tracker
        self.tracker.mark_as_best(exp_id)
        print(f"[SCI-Connectome] Marked run {exp_id} as best")

        # Update any existing layers in napari with this exp_id
        for layer in self.viewer.layers:
            if hasattr(layer, 'metadata') and layer.metadata.get('exp_id') == exp_id:
                # Update layer styling to best
                if hasattr(layer, 'border_color'):
                    layer.border_color = '#00FF00'  # Green for best
                # Add ★ to layer name if not already there
                if not layer.name.startswith("★"):
                    layer.name = "★ " + layer.name.lstrip("● ")
                layer.metadata['is_best'] = True

        # Refresh the runs table to show updated status
        self._refresh_context_runs()

        QMessageBox.information(self, "Success", f"Run marked as best!")

    def _mark_selected_as_paradigm_best(self):
        """Mark the selected run as best for its imaging paradigm."""
        print("[DEBUG] Button clicked: _mark_selected_as_paradigm_best")
        exp_id = self._get_selected_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a run to mark as paradigm best.")
            return

        if not self.tracker:
            QMessageBox.warning(self, "No Tracker", "Experiment tracker not available.")
            return

        # Get the run to check its paradigm
        run = self.tracker.get_experiment(exp_id)
        if not run:
            QMessageBox.warning(self, "Not Found", f"Run {exp_id} not found in tracker.")
            return

        paradigm = run.get('imaging_paradigm', '')
        if not paradigm:
            QMessageBox.warning(
                self, "No Paradigm",
                "This brain's imaging paradigm could not be detected.\n"
                "Paradigm is extracted from brain name (e.g., '1p625x_z4').\n"
                "Ensure brain name follows format: BRAIN_PROJECT_COHORT_SUBJECT_PARADIGM"
            )
            return

        # Confirm with user
        reply = QMessageBox.question(
            self, "Confirm Paradigm Best",
            f"Mark this run as PARADIGM BEST for '{paradigm}'?\n\n"
            f"These detection settings will be auto-applied to ALL future brains "
            f"with the same imaging paradigm.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Mark as paradigm best in tracker
        success = self.tracker.mark_paradigm_best(exp_id)
        if success:
            print(f"[SCI-Connectome] Marked run {exp_id} as paradigm best for '{paradigm}'")
            self.paradigm_status_label.setText(f"Paradigm: {paradigm} ✓")

            # Update any existing layers
            for layer in self.viewer.layers:
                if hasattr(layer, 'metadata') and layer.metadata.get('exp_id') == exp_id:
                    if hasattr(layer, 'border_color'):
                        layer.border_color = '#9C27B0'  # Purple for paradigm best
                    if "★" not in layer.name and "◆" not in layer.name:
                        layer.name = "◆ " + layer.name.lstrip("● ")

            self._refresh_context_runs()
            QMessageBox.information(
                self, "Success",
                f"Run marked as paradigm best for '{paradigm}'!\n\n"
                f"Future brains with this imaging paradigm will use these settings."
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to mark as paradigm best.")

    def _update_paradigm_status(self):
        """Update the paradigm status label based on current brain."""
        if not self.tracker or not self.current_brain:
            if hasattr(self, 'paradigm_status_label'):
                self.paradigm_status_label.setText("")
            return

        from braintools.config import parse_brain_name
        parsed = parse_brain_name(self.current_brain.name)
        paradigm = parsed.get('imaging_params', '')

        if not paradigm:
            if hasattr(self, 'paradigm_status_label'):
                self.paradigm_status_label.setText("No paradigm detected")
            return

        # Check if there's a paradigm best for this paradigm
        best = self.tracker.get_best_for_paradigm(paradigm, exp_type="detection")
        if hasattr(self, 'paradigm_status_label'):
            if best and best.get('paradigm_best') == 'True':
                self.paradigm_status_label.setText(f"Paradigm: {paradigm} (has best)")
            else:
                self.paradigm_status_label.setText(f"Paradigm: {paradigm}")

    def _offer_paradigm_settings(self):
        """Offer to apply paradigm settings if available for current brain."""
        if not self.tracker or not self.current_brain:
            return

        from braintools.config import parse_brain_name
        parsed = parse_brain_name(self.current_brain.name)
        paradigm = parsed.get('imaging_params', '')

        if not paradigm:
            return

        # Check if this brain already has calibration runs
        existing_runs = self.tracker.search(
            brain=self.current_brain.name,
            exp_type="detection",
            status="completed"
        )
        if existing_runs:
            # Brain has runs, don't auto-suggest
            return

        # Check for paradigm best settings
        settings = self.tracker.get_paradigm_detection_settings(paradigm)
        if not settings:
            return

        # Offer to apply settings
        source_brain = settings.get('source_brain', 'unknown')
        reply = QMessageBox.question(
            self, "Apply Paradigm Settings?",
            f"Found optimized settings for paradigm '{paradigm}'\n"
            f"(from brain: {source_brain})\n\n"
            f"Settings:\n"
            f"  Ball XY: {settings['ball_xy']}\n"
            f"  Ball Z: {settings['ball_z']}\n"
            f"  Soma: {settings['soma_diameter']}\n"
            f"  Threshold: {settings['threshold']}\n\n"
            f"Apply these settings?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._apply_paradigm_settings(settings)

    def _apply_paradigm_settings(self, settings: dict):
        """Apply paradigm detection settings to the UI."""
        if 'ball_xy' in settings and hasattr(self, 'ball_xy'):
            self.ball_xy.setValue(int(settings['ball_xy']))
        if 'ball_z' in settings and hasattr(self, 'ball_z'):
            self.ball_z.setValue(int(settings['ball_z']))
        if 'soma_diameter' in settings and hasattr(self, 'soma_diameter'):
            self.soma_diameter.setValue(int(settings['soma_diameter']))
        if 'threshold' in settings and hasattr(self, 'threshold'):
            self.threshold.setValue(int(settings['threshold']))

        source = settings.get('source_brain', 'paradigm best')
        print(f"[SCI-Connectome] Applied paradigm settings from {source}")

    def load_preset(self, preset_name):
        """Load a preset configuration."""
        presets = {
            "sensitive": {"ball_xy": 4, "ball_z": 10, "soma": 12, "thresh": 8, "overlap": 0.6, "spread": 1.4},
            "balanced": {"ball_xy": 6, "ball_z": 15, "soma": 16, "thresh": 10, "overlap": 0.6, "spread": 1.4},
            "conservative": {"ball_xy": 8, "ball_z": 20, "soma": 20, "thresh": 12, "overlap": 0.6, "spread": 1.4},
            "large_cells": {"ball_xy": 10, "ball_z": 25, "soma": 25, "thresh": 10, "overlap": 0.6, "spread": 1.4},
            # Your proven settings from the spreadsheet
            "proven": {"ball_xy": 10, "ball_z": 10, "soma": 10, "thresh": 8, "overlap": 0.5, "spread": 1.0, "sigma": 0.2, "max_cluster": 100000000},
        }
        if preset_name in presets:
            p = presets[preset_name]
            self.ball_xy.setValue(p["ball_xy"])
            self.ball_z.setValue(p["ball_z"])
            self.soma_diameter.setValue(p["soma"])
            self.threshold.setValue(p["thresh"])
            self.ball_overlap_fraction.setValue(p.get("overlap", 0.6))
            self.soma_spread_factor.setValue(p.get("spread", 1.4))
            if "sigma" in p:
                self.log_sigma_size.setValue(p["sigma"])
            if "max_cluster" in p:
                self.max_cluster_size.setValue(p["max_cluster"])
            print(f"[SCI-Connectome] Loaded preset: {preset_name}")

    def set_z_range_from_viewer(self):
        """Set Z range centered on current napari viewer position."""
        try:
            # Get current Z position from viewer
            current_step = self.viewer.dims.current_step
            if len(current_step) >= 1:
                current_z = current_step[0]

                # Get total Z range from data
                total_z = 1000  # default
                for layer in self.viewer.layers:
                    if hasattr(layer, 'data') and hasattr(layer.data, 'shape'):
                        if len(layer.data.shape) >= 3:
                            total_z = layer.data.shape[0]
                            break

                # Set range centered on current Z (default 50 slices each direction)
                half_range = 25
                z_start = max(0, int(current_z) - half_range)
                z_end = min(total_z, int(current_z) + half_range)

                self.test_z_start.setValue(z_start)
                self.test_z_end.setValue(z_end)

                print(f"[SCI-Connectome] Z range set from viewer: {z_start} to {z_end} (centered on Z={int(current_z)})")
        except Exception as e:
            print(f"[SCI-Connectome] Could not get viewer position: {e}")
            QMessageBox.warning(self, "Error", f"Could not get current frame position: {e}")

    def update_z_range_label(self):
        """Update the Z range slice count label."""
        count = self.test_z_end.value() - self.test_z_start.value()
        self.z_range_label.setText(f"({count} slices)")

    def _update_scope_label(self, button):
        """Update the scope label based on selected radio button."""
        if self.scope_current.isChecked():
            self.z_range_label.setText("(~50 slices around current position)")
        elif self.scope_range.isChecked():
            count = self.test_z_end.value() - self.test_z_start.value()
            self.z_range_label.setText(f"({count} slices)")
        elif self.scope_all.isChecked():
            self.z_range_label.setText("(entire brain - may take several minutes)")

    def run_test_detection(self):
        """Run detection on a test region using cellfinder Python API (like native plugin)."""
        print("[DEBUG] Button clicked: run_test_detection")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        # Check if brain is loaded in napari - use those arrays if available
        signal_layer = None
        background_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                if 'Signal' in layer.name or 'signal' in layer.name:
                    signal_layer = layer
                elif 'Background' in layer.name or 'background' in layer.name:
                    background_layer = layer

        # If not loaded in viewer, load from disk (priority: manual_crop > auto_crop)
        if signal_layer is None or background_layer is None:
            manual_crop_folder = self.current_brain / "2_Cropped_For_Registration_Manual"
            crop_folder = self.current_brain / "2_Cropped_For_Registration"

            if manual_crop_folder.exists() and (manual_crop_folder / "ch0").exists():
                data_folder = manual_crop_folder
                print("[SCI-Connectome] Using MANUAL crop folder (takes priority)")
            else:
                data_folder = crop_folder

            swap = getattr(self, 'swap_channels', None)
            if swap is None or swap.isChecked():
                signal_path = data_folder / "ch1"
                background_path = data_folder / "ch0"
                print("[SCI-Connectome] Using swapped channels: signal=ch1, background=ch0")
            else:
                signal_path = data_folder / "ch0"
                background_path = data_folder / "ch1"

            if not signal_path.exists() or not background_path.exists():
                QMessageBox.warning(self, "Error",
                    "Load the brain into napari first (click 'Load Brain into Napari')\n"
                    "or check that the cropped data exists.")
                return

            # Load the arrays from disk
            try:
                import tifffile
                from natsort import natsorted

                self.test_detection_btn.setEnabled(False)
                self.test_detection_btn.setText("Loading images...")
                QApplication.processEvents()

                signal_files = natsorted(signal_path.glob("*.tif*"))
                bg_files = natsorted(background_path.glob("*.tif*"))

                signal_array = tifffile.imread([str(f) for f in signal_files])
                background_array = tifffile.imread([str(f) for f in bg_files])
                print(f"[SCI-Connectome] Loaded from disk: signal {signal_array.shape}, bg {background_array.shape}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load images: {e}")
                self.test_detection_btn.setEnabled(True)
                self.test_detection_btn.setText("Run Test Detection")
                return
        else:
            signal_array = signal_layer.data
            background_array = background_layer.data
            print(f"[SCI-Connectome] Using loaded layers: signal {signal_array.shape}, bg {background_array.shape}")

        # Get Z range based on scope selection
        total_z = signal_array.shape[0]

        if self.scope_current.isChecked():
            # Current slice ± 25
            current_step = self.viewer.dims.current_step
            current_z = int(current_step[0]) if len(current_step) >= 1 else total_z // 2
            z_start = max(0, current_z - 25)
            z_end = min(total_z, current_z + 25)
            print(f"[SCI-Connectome] Scope: current slice ±25 (Z {z_start}-{z_end}, centered on {current_z})")
        elif self.scope_range.isChecked():
            # User-specified range
            z_start = self.test_z_start.value()
            z_end = self.test_z_end.value()
            if z_end <= z_start:
                QMessageBox.warning(self, "Error", "Z end must be greater than Z start")
                return
            print(f"[SCI-Connectome] Scope: user range Z {z_start}-{z_end}")
        else:  # scope_all
            # Entire brain
            z_start = 0
            z_end = total_z
            print(f"[SCI-Connectome] Scope: entire brain (Z 0-{total_z})")

        # Slice arrays if not using entire brain
        if z_start != 0 or z_end != total_z:
            signal_array = signal_array[z_start:z_end]
            background_array = background_array[z_start:z_end]
        print(f"[SCI-Connectome] Running detection on: {signal_array.shape}")

        # Get voxel sizes from metadata
        voxel = self.metadata.get('voxel_size_um', {'x': 4, 'y': 4, 'z': 4})
        voxel_sizes = (voxel.get('z', 4), voxel.get('y', 4), voxel.get('x', 4))

        # Get ALL detection parameters
        params = {
            # Core ball filter
            'ball_xy_size': self.ball_xy.value(),
            'ball_z_size': self.ball_z.value(),
            'ball_overlap_fraction': self.ball_overlap_fraction.value(),
            # Soma
            'soma_diameter': self.soma_diameter.value(),
            'soma_spread_factor': self.soma_spread_factor.value(),
            # Thresholding
            'threshold': self.threshold.value(),
            'log_sigma_size': self.log_sigma_size.value(),
            # Cluster handling
            'max_cluster_size': self.max_cluster_size.value(),
            # Performance
            'n_free_cpus': self.n_free_cpus.value(),
        }

        # Store Z offset for coordinate adjustment when displaying results
        self._detection_z_offset = z_start

        # Create output directory for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATA_SUMMARY_DIR / "optimization_runs" / self.current_brain.name / f"test_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        self._last_output_path = output_path

        # Log to tracker with detection scope
        if self.tracker:
            # Determine detection scope type
            if self.scope_all.isChecked():
                det_scope = "full"
                z_center = None
            else:
                det_scope = "partial"
                z_center = (z_start + z_end) // 2

            notes_parts = [f"Z{z_start}-{z_end}"]
            self.last_run_id = self.tracker.log_detection(
                brain=self.current_brain.name,
                preset="custom",
                ball_xy=params['ball_xy_size'],
                ball_z=params['ball_z_size'],
                soma_diameter=params['soma_diameter'],
                threshold=params['threshold'],
                voxel_z=voxel_sizes[0],
                voxel_xy=voxel_sizes[1],
                output_path=str(output_path),
                notes=", ".join(notes_parts),
                status="started",
                # Detection scope tracking
                det_scope=det_scope,
                det_z_start=z_start,
                det_z_end=z_end,
                det_z_center=z_center,
            )

        # Print all params to terminal
        print(f"\n[SCI-Connectome] Starting detection with parameters:")
        print(f"  Ball filter: XY={params['ball_xy_size']}, Z={params['ball_z_size']}, overlap={params['ball_overlap_fraction']}")
        print(f"  Soma: diameter={params['soma_diameter']}, spread={params['soma_spread_factor']}")
        print(f"  Threshold: {params['threshold']} SDs, sigma={params['log_sigma_size']}")
        print(f"  Cluster: max={params['max_cluster_size']}")
        print(f"  Z range: {z_start} to {z_end}")
        print(f"  Voxel sizes (ZYX): {voxel_sizes}")

        # Disable button during run
        self.test_detection_btn.setEnabled(False)
        self.test_detection_btn.setText("Running detection...")

        # Start worker thread with numpy arrays (like native cellfinder plugin)
        self.detection_worker = DetectionWorker(
            signal_array, background_array, voxel_sizes, params
        )
        self.detection_worker.progress.connect(self._on_detection_progress)
        self.detection_worker.finished.connect(self._on_detection_finished)
        self.detection_worker.start()

    def _on_detection_progress(self, message):
        """Handle detection progress updates."""
        self.test_detection_btn.setText(message[:30] + "...")

    def _on_detection_finished(self, success, message, cell_count, detected_cells):
        """Handle detection completion - now receives cells directly from Python API."""
        self.test_detection_btn.setEnabled(True)
        self.test_detection_btn.setText("Run Test Detection")

        if success:
            self.last_run_cells = cell_count

            # Save detected cells to XML file for later loading
            if detected_cells and hasattr(self, '_last_output_path') and self._last_output_path:
                try:
                    from brainglobe_utils.IO.cells import save_cells
                    xml_path = self._last_output_path / "detected_cells.xml"
                    save_cells(detected_cells, xml_path)
                    print(f"[SCI-Connectome] Saved {len(detected_cells)} cells to: {xml_path}")
                except Exception as e:
                    print(f"[SCI-Connectome] Warning: Could not save cells to XML: {e}")

            # Update tracker
            if self.tracker and self.last_run_id:
                self.tracker.update_status(
                    self.last_run_id,
                    status="completed",
                    det_cells_found=cell_count,
                )

            # Add to session tracking - track runs made during this session
            if self.last_run_id and self.last_run_id not in self.session_run_ids:
                self.session_run_ids.append(self.last_run_id)

            # Refresh the calibration runs table to show new run
            self._refresh_context_runs()

            # Log to session documenter
            if self.session_doc and self.last_run_id:
                params = {
                    'ball_xy_size': self.ball_xy.value(),
                    'ball_z_size': self.ball_z.value(),
                    'soma_diameter': self.soma_diameter.value(),
                    'threshold': self.threshold.value(),
                }
                z_range = (self.test_z_start.value(), self.test_z_end.value())
                self.session_doc.log_detection_run(
                    run_id=self.last_run_id,
                    params=params,
                    cell_count=cell_count,
                    preset="custom",
                    z_range=z_range,
                )

            # Add detected cells directly to napari as points layer
            # Using same approach as native cellfinder napari plugin (see cellfinder/napari/utils.py)
            if detected_cells and len(detected_cells) > 0:
                z_offset = getattr(self, '_detection_z_offset', 0)

                # Convert Cell objects to array - same as native cellfinder does
                # Cell objects have .x, .y, .z attributes
                # Native plugin does: points = [(c.x, c.y, c.z)] then reorders with [:, [2,1,0]]
                # which gives (z, y, x) order for napari
                points_xyz = np.array([(c.x, c.y, c.z) for c in detected_cells], dtype=np.float64)
                coords = points_xyz[:, [2, 1, 0]]  # Reorder to (z, y, x) for napari

                # Apply Z offset for the slice range we analyzed
                coords[:, 0] += z_offset

                print(f"[SCI-Connectome] Point coordinates (after Z offset={z_offset}):")
                print(f"  Z range: {coords[:, 0].min():.0f} - {coords[:, 0].max():.0f}")
                print(f"  Y range: {coords[:, 1].min():.0f} - {coords[:, 1].max():.0f}")
                print(f"  X range: {coords[:, 2].min():.0f} - {coords[:, 2].max():.0f}")
                print(f"  dtype: {coords.dtype}, shape: {coords.shape}")

                # Create layer name with session indicator
                layer_name = f"● Run {datetime.now().strftime('%H:%M:%S')} ({len(coords)} cells)"

                # Use consistent color scheme for new detection results
                style = self.CELL_COLORS['new']
                border_width = style.get('border_width', 0.1)
                layer = self.viewer.add_points(
                    coords,
                    name=layer_name,
                    size=style['size'],
                    n_dimensional=True,
                    opacity=style['opacity'],
                    symbol=style['symbol'],
                    face_color=style['face'],
                    border_color=style['edge'],
                    border_width=border_width,
                    visible=True,
                )

                # Store exp_id in layer metadata for tracking
                if self.last_run_id:
                    layer.metadata['exp_id'] = self.last_run_id
                    layer.metadata['is_best'] = False

                print(f"[SCI-Connectome] Added {len(coords)} cells to napari (white circle outline = new detection)")
                print(f"[SCI-Connectome] Layer style: size={style['size']}, symbol={style['symbol']}, border_width={border_width}")

                # Auto-generate QC image for this detection run
                self._generate_detection_qc_image()

            QMessageBox.information(self, "Detection Complete", message)
        else:
            if self.tracker and self.last_run_id:
                self.tracker.update_status(self.last_run_id, status="failed")
            QMessageBox.warning(self, "Detection Failed", message)

    def _generate_detection_qc_image(self):
        """
        Generate a QC image after detection run.

        Creates a multi-panel PNG showing:
        - Current napari view (signal + detected cells)
        - Detection parameters used
        - Run metadata (brain hierarchy, timestamp, cell count)

        Saves to: DATA_SUMMARY_DIR/optimization_runs/{brain}/QC_{run_id}.png
        """
        if not self.current_brain or not self.last_run_id:
            return

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("[QC] PIL not available, skipping QC image generation")
            return

        try:
            brain_name = self.current_brain.name

            # Output directory
            qc_dir = DATA_SUMMARY_DIR / "optimization_runs" / brain_name / "qc_images"
            qc_dir.mkdir(parents=True, exist_ok=True)

            # Take napari screenshot
            screenshot = self.viewer.screenshot()
            viewer_img = Image.fromarray(screenshot)

            # Create info panel
            info_width = 400
            info_height = viewer_img.height
            info_img = Image.new('RGB', (info_width, info_height), color=(40, 40, 40))
            draw = ImageDraw.Draw(info_img)

            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 14)
                font_bold = ImageFont.truetype("arialbd.ttf", 16)
                font_title = ImageFont.truetype("arialbd.ttf", 20)
            except:
                font = ImageFont.load_default()
                font_bold = font
                font_title = font

            y_pos = 20
            line_height = 22

            # Title
            draw.text((20, y_pos), "Detection QC Report", fill=(255, 255, 255), font=font_title)
            y_pos += 35

            # Timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            draw.text((20, y_pos), f"Generated: {timestamp}", fill=(180, 180, 180), font=font)
            y_pos += line_height * 2

            # Data hierarchy
            draw.text((20, y_pos), "DATA HIERARCHY", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5

            try:
                from braintools.config import parse_brain_name
                hierarchy = parse_brain_name(brain_name)
                if hierarchy.get("brain_id"):
                    draw.text((20, y_pos), f"Brain ID: {hierarchy.get('brain_id', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Subject: {hierarchy.get('subject_full', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Cohort: {hierarchy.get('cohort_full', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Project: {hierarchy.get('project_name', '')} ({hierarchy.get('project_code', '')})", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    if hierarchy.get('imaging_params'):
                        draw.text((20, y_pos), f"Imaging: {hierarchy.get('imaging_params', '')}", fill=(200, 200, 200), font=font)
                        y_pos += line_height
            except Exception as e:
                draw.text((20, y_pos), f"Brain: {brain_name}", fill=(255, 255, 255), font=font)
                y_pos += line_height

            y_pos += line_height

            # Run info
            draw.text((20, y_pos), "RUN INFO", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5
            draw.text((20, y_pos), f"Run ID: {self.last_run_id}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Cells Found: {self.last_run_cells}", fill=(100, 255, 100), font=font_bold)
            y_pos += line_height
            z_start = self.test_z_start.value()
            z_end = self.test_z_end.value()
            draw.text((20, y_pos), f"Z Range: {z_start} - {z_end}", fill=(255, 255, 255), font=font)
            y_pos += line_height * 2

            # Detection parameters
            draw.text((20, y_pos), "DETECTION PARAMETERS", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5
            draw.text((20, y_pos), f"Ball XY Size: {self.ball_xy.value()}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Ball Z Size: {self.ball_z.value()}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Soma Diameter: {self.soma_diameter.value()}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Threshold: {self.threshold.value()}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Soma Spread: {self.soma_spread_factor.value()}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"LoG Sigma: {self.log_sigma_size.value()}", fill=(255, 255, 255), font=font)
            y_pos += line_height * 2

            # Voxel info
            if self.metadata:
                voxel = self.metadata.get('voxel_size_um', {})
                if voxel:
                    draw.text((20, y_pos), "VOXEL SIZE", fill=(100, 200, 255), font=font_bold)
                    y_pos += line_height + 5
                    draw.text((20, y_pos), f"X: {voxel.get('x', '?')} um", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Y: {voxel.get('y', '?')} um", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Z: {voxel.get('z', '?')} um", fill=(255, 255, 255), font=font)

            # Combine images side by side
            combined_width = viewer_img.width + info_width
            combined_height = max(viewer_img.height, info_height)
            combined = Image.new('RGB', (combined_width, combined_height), color=(40, 40, 40))
            combined.paste(viewer_img, (0, 0))
            combined.paste(info_img, (viewer_img.width, 0))

            # Save
            qc_filename = f"QC_{self.last_run_id}.png"
            qc_path = qc_dir / qc_filename
            combined.save(str(qc_path))

            print(f"[SCI-Connectome] QC image saved: {qc_path}")

            # Log to session documenter
            if self.session_doc:
                self.session_doc.log_export(str(qc_path), export_type="qc_image")

        except Exception as e:
            print(f"[SCI-Connectome] QC image generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _load_detection_results(self):
        """
        Load detection results into napari as points layer.

        Uses yellow/orange for candidates (before classification) to distinguish
        from the cyan/magenta used for classified results.
        """
        if not self.last_run_id:
            return

        # Find the output XML
        runs_dir = DATA_SUMMARY_DIR / "optimization_runs" / self.current_brain.name
        if not runs_dir.exists():
            return

        # Get most recent test folder
        test_folders = sorted(runs_dir.glob("test_*"), reverse=True)
        if not test_folders:
            return

        cells_xml = test_folders[0] / "detected_cells.xml"
        if not cells_xml.exists():
            return

        # Parse XML to get cell coordinates
        try:
            coords = self._parse_cellfinder_xml(cells_xml)
            if coords:
                # Add as points layer - use 'new' style for just-ran detection
                layer_name = f"Candidates {datetime.now().strftime('%H:%M:%S')} ({len(coords)})"
                style = self.CELL_COLORS['new']
                self.viewer.add_points(
                    coords,
                    name=layer_name,
                    size=style['size'],
                    face_color=style['face'],
                    border_color=style['edge'],
                    border_width=style.get('border_width', 0.1),
                    symbol=style['symbol'],
                    opacity=style['opacity'],
                    n_dimensional=True,  # Required for 3D point visibility
                )
        except Exception as e:
            print(f"Error loading detection results: {e}")

    def _parse_cellfinder_xml(self, xml_path):
        """Parse cellfinder XML to get cell coordinates."""
        import re
        coords = []
        with open(xml_path, 'r') as f:
            content = f.read()

        # Parse markers - cellfinder uses <Marker> tags with x,y,z children
        marker_pattern = r'<Marker>.*?<MarkerX>(\d+)</MarkerX>.*?<MarkerY>(\d+)</MarkerY>.*?<MarkerZ>(\d+)</MarkerZ>.*?</Marker>'
        matches = re.findall(marker_pattern, content, re.DOTALL)

        for x, y, z in matches:
            coords.append([int(z), int(y), int(x)])  # napari uses ZYX order

        return np.array(coords) if coords else None

    # ==========================================================================
    # NEW COMPARISON TABS
    # ==========================================================================

    def create_detection_compare_tab(self):
        """Tab 5: Compare detection runs with difference visualization."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        info = QLabel(
            "Compare two detection runs to see where they differ.\n"
            "Select runs from the list below, then generate difference layers."
        )
        info.setWordWrap(True)
        content_layout.addWidget(info)

        # =================================================================
        # RUNS LIST (for selecting runs to compare)
        # =================================================================
        runs_group = QGroupBox("Calibration Runs (select to compare)")
        runs_layout = QVBoxLayout()
        runs_group.setLayout(runs_layout)

        # Runs list widget
        from qtpy.QtWidgets import QListWidget
        self.compare_runs_list = QListWidget()
        self.compare_runs_list.setMinimumHeight(150)
        self.compare_runs_list.setMaximumHeight(200)
        self.compare_runs_list.setSelectionMode(QListWidget.SingleSelection)
        self.compare_runs_list.setToolTip("★ = Best marked | ● = This session\nSelect a run, then load as Reference or Comparison")
        self.compare_runs_list.setStyleSheet("""
            QListWidget {
                background-color: #363636;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """)
        runs_layout.addWidget(self.compare_runs_list)

        # Load buttons
        load_btn_row = QHBoxLayout()
        self.load_as_best_btn = QPushButton("Load as Reference (green)")
        self.load_as_best_btn.setStyleSheet("background-color: #00AA00; color: white;")
        self.load_as_best_btn.clicked.connect(self._load_selected_as_best)
        self.load_as_best_btn.setToolTip("Load selected run as the REFERENCE for comparison")
        load_btn_row.addWidget(self.load_as_best_btn)

        self.load_as_recent_btn = QPushButton("Load as Comparison (orange)")
        self.load_as_recent_btn.setStyleSheet("background-color: #CC6600; color: white;")
        self.load_as_recent_btn.clicked.connect(self._load_selected_as_recent)
        self.load_as_recent_btn.setToolTip("Load selected run as the COMPARISON run")
        load_btn_row.addWidget(self.load_as_recent_btn)

        runs_layout.addLayout(load_btn_row)
        content_layout.addWidget(runs_group)

        # =================================================================
        # BEST DETECTION SECTION
        # =================================================================
        best_group = QGroupBox("Best Detection (Reference)")
        best_layout = QVBoxLayout()
        best_group.setLayout(best_layout)

        self.best_det_label = QLabel("No best detection marked")
        self.best_det_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        best_layout.addWidget(self.best_det_label)

        mark_best_layout = QHBoxLayout()

        self.mark_as_best_btn = QPushButton("Mark Selected Layer as Best")
        self.mark_as_best_btn.setStyleSheet("background-color: #00AA00; color: white; padding: 8px;")
        self.mark_as_best_btn.clicked.connect(self.mark_current_as_best)
        self.mark_as_best_btn.setToolTip(
            "Select a detection points layer in napari, then click to mark it as BEST.\n"
            "This becomes your reference for comparison."
        )
        mark_best_layout.addWidget(self.mark_as_best_btn)

        self.load_best_from_tracker_btn = QPushButton("Load Best from Tracker")
        self.load_best_from_tracker_btn.clicked.connect(lambda: self._load_detection_results(best=True))
        self.load_best_from_tracker_btn.setToolTip("Load highest-rated detection from experiment tracker")
        mark_best_layout.addWidget(self.load_best_from_tracker_btn)

        best_layout.addLayout(mark_best_layout)
        content_layout.addWidget(best_group)

        # =================================================================
        # RECENT DETECTION SECTION
        # =================================================================
        recent_group = QGroupBox("Most Recent Detection")
        recent_layout = QVBoxLayout()
        recent_group.setLayout(recent_layout)

        self.recent_det_label = QLabel("No recent detection loaded")
        self.recent_det_label.setStyleSheet("color: #FFA500;")
        recent_layout.addWidget(self.recent_det_label)

        load_recent_btn = QPushButton("Load Most Recent Detection")
        load_recent_btn.clicked.connect(lambda: self._load_detection_results(best=False))
        load_recent_btn.setStyleSheet("background-color: #CC6600; color: white;")
        recent_layout.addWidget(load_recent_btn)

        content_layout.addWidget(recent_group)

        # =================================================================
        # DIFFERENCE VISUALIZATION
        # =================================================================
        diff_group = QGroupBox("Difference Analysis")
        diff_layout = QVBoxLayout()
        diff_group.setLayout(diff_layout)

        diff_layout.addWidget(QLabel(
            "Generate difference layers showing where detections DON'T overlap.\n"
            "This helps you decide if the new detection found more real cells."
        ))

        # Display options
        self.show_all_z_cb = QCheckBox("Show all Z levels (out of slice display)")
        self.show_all_z_cb.setChecked(False)
        self.show_all_z_cb.setToolTip(
            "When checked, shows ALL detected points regardless of current Z slice.\n"
            "Useful for overview and comparing distributions.\n"
            "When unchecked, only shows points near current slice (default)."
        )
        self.show_all_z_cb.stateChanged.connect(self._toggle_out_of_slice_display)
        diff_layout.addWidget(self.show_all_z_cb)

        # Auto-generate toggle
        self.auto_generate_diff_cb = QCheckBox("Auto-generate differences when both layers loaded")
        self.auto_generate_diff_cb.setChecked(True)
        diff_layout.addWidget(self.auto_generate_diff_cb)

        # Tolerance setting
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Match tolerance (voxels):"))
        self.det_diff_tolerance_spin = QSpinBox()
        self.det_diff_tolerance_spin.setRange(1, 50)
        self.det_diff_tolerance_spin.setValue(5)
        self.det_diff_tolerance_spin.setToolTip(
            "Points within this distance (in voxels) are considered the same cell.\n"
            "Increase if your cells might shift slightly between runs.\n"
            "Note: Uses Euclidean distance in voxel coordinates (Z, Y, X)."
        )
        tolerance_layout.addWidget(self.det_diff_tolerance_spin)
        tolerance_layout.addStretch()
        diff_layout.addLayout(tolerance_layout)

        self.gen_det_diff_btn = QPushButton("Generate Difference Layers")
        self.gen_det_diff_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 10px; font-size: 14px;")
        self.gen_det_diff_btn.clicked.connect(self.generate_detection_difference)
        diff_layout.addWidget(self.gen_det_diff_btn)

        self.det_diff_stats_label = QLabel("")
        self.det_diff_stats_label.setWordWrap(True)
        diff_layout.addWidget(self.det_diff_stats_label)

        content_layout.addWidget(diff_group)

        # =================================================================
        # COMPARISON DECISION
        # =================================================================
        decision_group = QGroupBox("Decision")
        decision_layout = QVBoxLayout()
        decision_group.setLayout(decision_layout)

        decision_layout.addWidget(QLabel(
            "After reviewing differences, decide if the recent detection is better:"
        ))

        decision_btn_layout = QHBoxLayout()

        promote_btn = QPushButton("Promote Recent to Best")
        promote_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        promote_btn.clicked.connect(self.promote_recent_to_best)
        promote_btn.setToolTip("Make the recent detection your new reference (archives old best)")
        decision_btn_layout.addWidget(promote_btn)

        keep_btn = QPushButton("Keep Current Best")
        keep_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 8px;")
        keep_btn.clicked.connect(self.keep_current_best)
        keep_btn.setToolTip("Discard recent detection, keep current best")
        decision_btn_layout.addWidget(keep_btn)

        decision_layout.addLayout(decision_btn_layout)
        content_layout.addWidget(decision_group)

        # =================================================================
        # LEGEND
        # =================================================================
        legend_group = QGroupBox("Color Legend")
        legend_layout = QVBoxLayout()
        legend_group.setLayout(legend_layout)

        legend_text = QLabel(
            "<span style='color: #00FF00;'>Best (green)</span> - Your reference detection<br>"
            "<span style='color: #FFA500;'>Recent (orange)</span> - Latest detection run<br>"
            "<span style='color: #FF0080;'>Only in Best (pink diamond)</span> - Potential false negatives<br>"
            "<span style='color: #00BFFF;'>Only in Recent (blue triangle)</span> - New candidates"
        )
        legend_layout.addWidget(legend_text)
        content_layout.addWidget(legend_group)

        content_layout.addStretch()
        return widget

    def create_classification_tab(self):
        """Tab 6: Classification - Apply model to candidates."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        info = QLabel(
            "Apply a trained model to classify cell candidates.\n"
            "This filters out false positives, keeping only real cells."
        )
        info.setWordWrap(True)
        content_layout.addWidget(info)

        # =================================================================
        # MODEL SELECTION
        # =================================================================
        model_group = QGroupBox("Select Model")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        model_layout.addWidget(QLabel(
            "Choose which model to use for classification.\n"
            "Start with the default, then switch to your trained model."
        ))

        self.model_combo = QComboBox()
        self.model_combo.addItem("Default (BrainGlobe pretrained)")
        self.refresh_models()
        model_layout.addWidget(self.model_combo)

        refresh_models_btn = QPushButton("Refresh Model List")
        refresh_models_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(refresh_models_btn)

        content_layout.addWidget(model_group)

        # =================================================================
        # CLASSIFICATION PARAMETERS
        # =================================================================
        params_group = QGroupBox("Classification Parameters")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)

        self.classify_cube_size = QSpinBox()
        self.classify_cube_size.setRange(20, 100)
        self.classify_cube_size.setValue(50)
        self.classify_cube_size.setToolTip("Size of image cube extracted around each candidate")
        params_layout.addRow("Cube Size:", self.classify_cube_size)

        self.classify_batch_size = QSpinBox()
        self.classify_batch_size.setRange(1, 128)
        self.classify_batch_size.setValue(32)
        self.classify_batch_size.setToolTip("Batch size for classification (higher = faster, more memory)")
        params_layout.addRow("Batch Size:", self.classify_batch_size)

        content_layout.addWidget(params_group)

        # =================================================================
        # RUN CLASSIFICATION
        # =================================================================
        run_group = QGroupBox("Run Classification")
        run_layout = QVBoxLayout()
        run_group.setLayout(run_layout)

        run_layout.addWidget(QLabel(
            "Requires detection results in 4_Cell_Candidates folder."
        ))

        self.classify_btn = QPushButton("Run Classification")
        self.classify_btn.clicked.connect(self.run_classification)
        self.classify_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px; font-weight: bold;")
        run_layout.addWidget(self.classify_btn)

        self.classify_status = QLabel("")
        self.classify_status.setWordWrap(True)
        run_layout.addWidget(self.classify_status)

        content_layout.addWidget(run_group)

        # =================================================================
        # LOAD PREVIOUS RESULTS
        # =================================================================
        load_group = QGroupBox("Load Classification Results")
        load_layout = QVBoxLayout()
        load_group.setLayout(load_layout)

        load_layout.addWidget(QLabel(
            "Load previously saved classification results from 5_Classified_Cells/"
        ))

        self.load_classification_btn = QPushButton("Load Classification Results")
        self.load_classification_btn.clicked.connect(self._load_classified_cells)
        self.load_classification_btn.setStyleSheet("background-color: #7B1FA2; color: white; padding: 8px;")
        load_layout.addWidget(self.load_classification_btn)

        self.classification_load_status = QLabel("")
        self.classification_load_status.setWordWrap(True)
        load_layout.addWidget(self.classification_load_status)

        content_layout.addWidget(load_group)

        content_layout.addStretch()
        return widget

    def create_classification_compare_tab(self):
        """Tab 7: Compare classification results between two model runs."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        info = QLabel(
            "Compare classification results from two different models.\n"
            "See what changed: which cells were gained or lost."
        )
        info.setWordWrap(True)
        content_layout.addWidget(info)

        # =================================================================
        # PREVIOUS CLASSIFICATION
        # =================================================================
        prev_group = QGroupBox("Previous Classification (Reference)")
        prev_layout = QVBoxLayout()
        prev_group.setLayout(prev_layout)

        self.prev_class_label = QLabel("No previous classification loaded")
        self.prev_class_label.setStyleSheet("color: #98FB98;")
        prev_layout.addWidget(self.prev_class_label)

        prev_btn_layout = QHBoxLayout()

        load_prev_btn = QPushButton("Load Previous Results")
        load_prev_btn.clicked.connect(self.load_previous_classification)
        load_prev_btn.setStyleSheet("background-color: #32CD32; color: white;")
        prev_btn_layout.addWidget(load_prev_btn)

        mark_prev_btn = QPushButton("Mark Layer as Previous")
        mark_prev_btn.clicked.connect(self.mark_layer_as_prev_classification)
        prev_btn_layout.addWidget(mark_prev_btn)

        prev_layout.addLayout(prev_btn_layout)
        content_layout.addWidget(prev_group)

        # =================================================================
        # NEW CLASSIFICATION
        # =================================================================
        new_group = QGroupBox("New Classification (Current)")
        new_layout = QVBoxLayout()
        new_group.setLayout(new_layout)

        self.new_class_label = QLabel("No new classification loaded")
        self.new_class_label.setStyleSheet("color: #00CED1;")
        new_layout.addWidget(self.new_class_label)

        new_btn_layout = QHBoxLayout()

        load_new_btn = QPushButton("Load New Results")
        load_new_btn.clicked.connect(self.load_new_classification)
        load_new_btn.setStyleSheet("background-color: #008B8B; color: white;")
        new_btn_layout.addWidget(load_new_btn)

        mark_new_btn = QPushButton("Mark Layer as New")
        mark_new_btn.clicked.connect(self.mark_layer_as_new_classification)
        new_btn_layout.addWidget(mark_new_btn)

        new_layout.addLayout(new_btn_layout)
        content_layout.addWidget(new_group)

        # =================================================================
        # CLASSIFICATION RUNS FROM TRACKER
        # =================================================================
        runs_group = QGroupBox("Classification Runs (from Tracker)")
        runs_layout = QVBoxLayout()
        runs_group.setLayout(runs_layout)

        runs_layout.addWidget(QLabel("Select classification runs to compare:"))

        self.class_compare_runs_list = QListWidget()
        self.class_compare_runs_list.setMaximumHeight(120)
        self.class_compare_runs_list.setSelectionMode(QListWidget.SingleSelection)
        runs_layout.addWidget(self.class_compare_runs_list)

        runs_btn_layout = QHBoxLayout()

        load_prev_from_tracker_btn = QPushButton("Load as Previous")
        load_prev_from_tracker_btn.setStyleSheet("background-color: #32CD32; color: white;")
        load_prev_from_tracker_btn.clicked.connect(self._load_class_run_as_previous)
        runs_btn_layout.addWidget(load_prev_from_tracker_btn)

        load_new_from_tracker_btn = QPushButton("Load as New")
        load_new_from_tracker_btn.setStyleSheet("background-color: #008B8B; color: white;")
        load_new_from_tracker_btn.clicked.connect(self._load_class_run_as_new)
        runs_btn_layout.addWidget(load_new_from_tracker_btn)

        refresh_class_runs_btn = QPushButton("Refresh")
        refresh_class_runs_btn.clicked.connect(self._refresh_class_compare_runs)
        runs_btn_layout.addWidget(refresh_class_runs_btn)

        runs_layout.addLayout(runs_btn_layout)
        content_layout.addWidget(runs_group)

        # =================================================================
        # CLASSIFICATION DIFFERENCE
        # =================================================================
        diff_group = QGroupBox("Classification Difference Analysis")
        diff_layout = QVBoxLayout()
        diff_group.setLayout(diff_layout)

        diff_layout.addWidget(QLabel(
            "Compare what the two models decided differently.\n"
            "Helps evaluate if your trained model is better than default."
        ))

        # Auto-generate toggle
        self.auto_generate_class_diff_cb = QCheckBox("Auto-generate differences when both layers loaded")
        self.auto_generate_class_diff_cb.setChecked(True)
        diff_layout.addWidget(self.auto_generate_class_diff_cb)

        # Tolerance
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Match tolerance (voxels):"))
        self.class_diff_tolerance_spin = QSpinBox()
        self.class_diff_tolerance_spin.setRange(1, 50)
        self.class_diff_tolerance_spin.setValue(5)
        self.class_diff_tolerance_spin.setToolTip(
            "Points within this distance (in voxels) are considered the same cell.\n"
            "Note: Uses Euclidean distance in voxel coordinates (Z, Y, X)."
        )
        tolerance_layout.addWidget(self.class_diff_tolerance_spin)
        tolerance_layout.addStretch()
        diff_layout.addLayout(tolerance_layout)

        gen_class_diff_btn = QPushButton("Generate Classification Difference")
        gen_class_diff_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px;")
        gen_class_diff_btn.clicked.connect(self.generate_classification_difference)
        diff_layout.addWidget(gen_class_diff_btn)

        self.class_diff_stats = QLabel("")
        self.class_diff_stats.setWordWrap(True)
        diff_layout.addWidget(self.class_diff_stats)

        content_layout.addWidget(diff_group)

        # =================================================================
        # LEGEND
        # =================================================================
        legend_group = QGroupBox("Color Legend")
        legend_layout = QVBoxLayout()
        legend_group.setLayout(legend_layout)

        legend_text = QLabel(
            "<b>Previous Results:</b><br>"
            "<span style='color: #98FB98;'>Cells (pale green)</span> | "
            "<span style='color: #DDA0DD;'>Rejected (plum)</span><br><br>"
            "<b>New Results:</b><br>"
            "<span style='color: #00CED1;'>Cells (turquoise)</span> | "
            "<span style='color: #FF69B4;'>Rejected (hot pink)</span><br><br>"
            "<b>Differences:</b><br>"
            "<span style='color: #7FFF00;'>Gained (chartreuse star)</span> - cell in new, not in prev<br>"
            "<span style='color: #FF4500;'>Lost (orange-red x)</span> - cell in prev, not in new"
        )
        legend_layout.addWidget(legend_text)
        content_layout.addWidget(legend_group)

        content_layout.addStretch()
        return widget

    def create_curation_tab(self):
        """Tab 8: Curation and Training - Review cells and build training data."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        info = QLabel(
            "Review detection/classification results and build training data.\n"
            "Curate cells vs non-cells for training a custom network."
        )
        info.setWordWrap(True)
        content_layout.addWidget(info)

        # =================================================================
        # STREAMLINED TRAINING DATA MANAGEMENT
        # =================================================================
        status_group = QGroupBox("Training Data (Auto-Save Enabled)")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)

        # Create Training Layers button - sets up everything
        create_layers_btn = QPushButton("Create Training Layers")
        create_layers_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 8px;")
        create_layers_btn.setToolTip(
            "Creates 'Train: Cells' and 'Train: Non-Cells' layers.\n"
            "Sets default save location to {brain}/training_data/\n"
            "After this, every Y/N during curation auto-saves!"
        )
        create_layers_btn.clicked.connect(self.create_training_layers)
        status_layout.addWidget(create_layers_btn)

        # Training data path display
        self.training_path_label = QLabel("Training data path: (not set)")
        self.training_path_label.setStyleSheet("color: gray; font-style: italic;")
        status_layout.addWidget(self.training_path_label)

        # Status labels
        self.train_cells_count_label = QLabel("Cells bucket: 0 points")
        self.train_cells_count_label.setStyleSheet("color: #FFD700; font-weight: bold;")
        status_layout.addWidget(self.train_cells_count_label)

        self.train_non_cells_count_label = QLabel("Non-cells bucket: 0 points")
        self.train_non_cells_count_label.setStyleSheet("color: #8B0000; font-weight: bold;")
        status_layout.addWidget(self.train_non_cells_count_label)

        self.export_status_label = QLabel("")
        self.export_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        status_layout.addWidget(self.export_status_label)

        # Action buttons row
        action_row = QHBoxLayout()

        refresh_status_btn = QPushButton("Refresh")
        refresh_status_btn.clicked.connect(self.refresh_training_data_counts)
        action_row.addWidget(refresh_status_btn)

        # Export button - now called "Create Training Dataset"
        export_btn = QPushButton("Create Training Dataset")
        export_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        export_btn.setToolTip(
            "Export curated points to BrainGlobe TIFF cube format.\n"
            "Creates cells/ and non_cells/ folders with both channels.\n"
            "Also creates training.yml config file."
        )
        export_btn.clicked.connect(self.export_training_data)
        action_row.addWidget(export_btn)

        status_layout.addLayout(action_row)
        content_layout.addWidget(status_group)

        # =================================================================
        # CURATION MODE
        # =================================================================
        curate_group = QGroupBox("Curation Mode")
        curate_layout = QVBoxLayout()
        curate_group.setLayout(curate_layout)

        curate_layout.addWidget(QLabel(
            "Select a points layer, then use the curation widget.\n"
            "Keyboard: C=confirm cell, X=reject, Arrow keys=navigate"
        ))

        # Embed curation widget
        try:
            from braintools.pipeline_3d.curation_widget import CurationWidget
            self.curation_widget = CurationWidget(self.viewer)
            curate_layout.addWidget(self.curation_widget)

            # Connect signals for training bucket updates
            self.curation_widget.candidate_confirmed.connect(self._on_candidate_confirmed)
            self.curation_widget.candidate_rejected.connect(self._on_candidate_rejected)
        except Exception as e:
            error_label = QLabel(f"Could not load curation widget: {e}")
            error_label.setStyleSheet("color: red;")
            curate_layout.addWidget(error_label)

        content_layout.addWidget(curate_group)

        # =================================================================
        # ADD TO TRAINING BUCKETS
        # =================================================================
        bucket_group = QGroupBox("Add to Training Data")
        bucket_layout = QVBoxLayout()
        bucket_group.setLayout(bucket_layout)

        bucket_layout.addWidget(QLabel(
            "After curation, add confirmed/rejected to training buckets:"
        ))

        btn_layout = QHBoxLayout()

        add_cells_btn = QPushButton("Add Confirmed to Cells Bucket")
        add_cells_btn.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold;")
        add_cells_btn.clicked.connect(self.add_confirmed_to_training)
        btn_layout.addWidget(add_cells_btn)

        add_non_cells_btn = QPushButton("Add Rejected to Non-Cells Bucket")
        add_non_cells_btn.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;")
        add_non_cells_btn.clicked.connect(self.add_rejected_to_training)
        btn_layout.addWidget(add_non_cells_btn)

        bucket_layout.addLayout(btn_layout)
        content_layout.addWidget(bucket_group)

        # =================================================================
        # VISUALIZE TRAINING DATA
        # =================================================================
        viz_group = QGroupBox("Visualize Training Data")
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)

        show_train_btn = QPushButton("Show Training Data in Napari")
        show_train_btn.clicked.connect(self.visualize_training_data)
        viz_layout.addWidget(show_train_btn)

        viz_layout.addWidget(QLabel(
            "<span style='color: #FFD700;'>Gold = Training Cells</span> | "
            "<span style='color: #8B0000;'>Dark Red = Training Non-Cells</span>"
        ))

        content_layout.addWidget(viz_group)

        # =================================================================
        # START TRAINING
        # =================================================================
        train_group = QGroupBox("Train Custom Network")
        train_layout = QVBoxLayout()
        train_group.setLayout(train_layout)

        train_layout.addWidget(QLabel(
            "Once you have sufficient training data (100+ each recommended),\n"
            "train a custom network tuned to your specific tissue."
        ))

        self.training_data_path = QLineEdit()
        self.training_data_path.setPlaceholderText("Path to training data folder...")
        train_layout.addWidget(self.training_data_path)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_training_data)
        train_layout.addWidget(browse_btn)

        # Auto-classify option
        self.auto_classify_after_training = QCheckBox("Auto-classify after training completes")
        self.auto_classify_after_training.setChecked(True)
        self.auto_classify_after_training.setToolTip(
            "When training finishes, automatically run classification\n"
            "with the new model on the same detection candidates.\n"
            "Results will be ready for evaluation."
        )
        train_layout.addWidget(self.auto_classify_after_training)

        start_train_btn = QPushButton("Start Training")
        start_train_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px;")
        start_train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(start_train_btn)

        self.training_status_label = QLabel("")
        self.training_status_label.setStyleSheet("color: #9C27B0; font-style: italic;")
        train_layout.addWidget(self.training_status_label)

        content_layout.addWidget(train_group)

        content_layout.addStretch()
        return widget

    # ==========================================================================
    # STUB METHODS FOR NEW FUNCTIONALITY (to be implemented in later phases)
    # ==========================================================================

    def mark_current_as_best(self):
        """Mark the currently selected layer as best detection.

        Workflow:
        1. Get selected points layer
        2. Archive any existing best layer
        3. Apply det_best styling to selected layer
        4. Update experiment tracker
        5. Log to session documenter
        """
        print("[DEBUG] Button clicked: mark_current_as_best")
        import napari
        from datetime import datetime

        # Get selected layer
        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            QMessageBox.warning(self, "No Selection",
                "Please select a points layer in napari first.")
            return

        layer = selected_layers[0]
        if not isinstance(layer, napari.layers.Points):
            QMessageBox.warning(self, "Wrong Layer Type",
                "Please select a Points layer (cell candidates), not an image layer.")
            return

        # Get layer's exp_id from metadata if available
        exp_id = layer.metadata.get('exp_id') if hasattr(layer, 'metadata') else None

        # Find and archive any existing "best" layer
        existing_best = self._get_layer_by_type('det_best')
        if existing_best and existing_best != layer:
            old_exp_id = existing_best.metadata.get('exp_id') if hasattr(existing_best, 'metadata') else None
            self._archive_layer(existing_best, archived_by=exp_id or 'user selection')

            # Update tracker for archived experiment
            if self.tracker and old_exp_id:
                try:
                    self.tracker.archive_experiment(old_exp_id, archived_by=exp_id)
                except AttributeError:
                    # Method not yet added to tracker
                    pass

        # Apply best styling
        self._set_layer_type(layer, 'det_best')

        # Update name with BEST: prefix
        if not layer.name.startswith('BEST:'):
            # Remove any existing prefix
            name = layer.name
            for prefix in ['RECENT:', '[archived]', 'Best ', 'Recent ']:
                if name.startswith(prefix):
                    name = name[len(prefix):].strip()
            layer.name = f"BEST: {name}"

        # Update tracker
        if self.tracker and exp_id:
            try:
                self.tracker.mark_as_best(exp_id)
            except AttributeError:
                # Method not yet added to tracker - update rating as fallback
                self.tracker.update_rating(exp_id, rating=5, notes="Marked as best")

        # Log to session documenter
        if self.session_doc:
            self.session_doc.log_event('mark_as_best', {
                'layer_name': layer.name,
                'points_count': len(layer.data),
                'exp_id': exp_id,
            })

        QMessageBox.information(self, "Marked as Best",
            f"Layer '{layer.name}' is now marked as BEST.\n"
            f"Points: {len(layer.data)}")

    def promote_recent_to_best(self):
        """Promote the 'recent' detection to become the new 'best'.

        This is a shortcut that:
        1. Finds the det_recent layer
        2. Archives the current det_best
        3. Promotes det_recent to det_best
        """
        print("[DEBUG] Button clicked: promote_recent_to_best")
        import napari

        recent_layer = self._get_layer_by_type('det_recent')
        if not recent_layer:
            QMessageBox.warning(self, "No Recent Detection",
                "No recent detection layer found.\n"
                "Load or run a detection first, then compare.")
            return

        # Get exp_id for recent
        exp_id = recent_layer.metadata.get('exp_id') if hasattr(recent_layer, 'metadata') else None

        # Archive current best if exists
        existing_best = self._get_layer_by_type('det_best')
        if existing_best:
            old_exp_id = existing_best.metadata.get('exp_id') if hasattr(existing_best, 'metadata') else None
            self._archive_layer(existing_best, archived_by=exp_id or 'promoted recent')

            # Update tracker for archived experiment
            if self.tracker and old_exp_id:
                try:
                    self.tracker.archive_experiment(old_exp_id, archived_by=exp_id)
                except AttributeError:
                    pass

        # Promote recent to best
        self._set_layer_type(recent_layer, 'det_best')

        # Update name
        name = recent_layer.name
        for prefix in ['RECENT:', 'Recent ', 'Most recent']:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
        recent_layer.name = f"BEST: {name}"

        # Update tracker
        if self.tracker and exp_id:
            try:
                self.tracker.mark_as_best(exp_id)
            except AttributeError:
                self.tracker.update_rating(exp_id, rating=5, notes="Promoted from recent to best")

        # Log to session documenter
        if self.session_doc:
            self.session_doc.log_event('promote_to_best', {
                'layer_name': recent_layer.name,
                'points_count': len(recent_layer.data),
                'exp_id': exp_id,
                'previous_best_archived': existing_best is not None,
            })

        QMessageBox.information(self, "Promoted to Best",
            f"Recent detection promoted to BEST.\n"
            f"Layer: {recent_layer.name}\n"
            f"Points: {len(recent_layer.data)}")

    def keep_current_best(self):
        """Keep the current best, archive the recent comparison layer.

        User has decided the current best is better than the recent run.
        """
        print("[DEBUG] Button clicked: keep_current_best")
        recent_layer = self._get_layer_by_type('det_recent')
        if not recent_layer:
            QMessageBox.information(self, "Keep Best",
                "No recent detection to discard. Current best remains.")
            return

        best_layer = self._get_layer_by_type('det_best')
        best_exp_id = best_layer.metadata.get('exp_id') if best_layer and hasattr(best_layer, 'metadata') else None

        # Archive the recent layer
        self._archive_layer(recent_layer, archived_by=f"kept best: {best_exp_id}" if best_exp_id else "user kept best")

        # Log to session documenter
        if self.session_doc:
            self.session_doc.log_event('keep_best', {
                'best_layer': best_layer.name if best_layer else None,
                'archived_recent': recent_layer.name,
                'decision': 'kept_current_best',
            })

        QMessageBox.information(self, "Kept Best",
            f"Current BEST retained.\n"
            f"Archived: {recent_layer.name}")

    def _toggle_out_of_slice_display(self, state):
        """Toggle out_of_slice_display for all Points layers."""
        import napari
        show_all = (state == Qt.Checked)

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                layer.out_of_slice_display = show_all

        status = "showing ALL Z levels" if show_all else "showing current slice only"
        print(f"[SCI-Connectome] Points layers: {status}")

    def generate_detection_difference(self):
        """Generate difference layers comparing best vs recent detection.

        Creates two new layers:
        - Only in BEST (potential false negatives in recent)
        - Only in RECENT (new candidates)

        Uses scipy.spatial.cKDTree for efficient nearest-neighbor matching.
        """
        print("[DEBUG] Button clicked: generate_detection_difference")
        import numpy as np

        # Get tolerance from UI if available (in voxels)
        tolerance = getattr(self, 'det_diff_tolerance_spin', None)
        tolerance_voxels = tolerance.value() if tolerance else 5.0

        best_layer = self._get_layer_by_type('det_best')
        recent_layer = self._get_layer_by_type('det_recent')

        if not best_layer:
            QMessageBox.warning(self, "Missing Layer",
                "No 'best' detection layer found.\n"
                "Mark a detection as best first.")
            return

        if not recent_layer:
            QMessageBox.warning(self, "Missing Layer",
                "No 'recent' detection layer found.\n"
                "Load or run a detection for comparison.")
            return

        best_coords = np.array(best_layer.data)
        recent_coords = np.array(recent_layer.data)

        if len(best_coords) == 0 or len(recent_coords) == 0:
            QMessageBox.warning(self, "Empty Layers",
                "One or both layers have no points.")
            return

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            QMessageBox.warning(self, "Missing scipy",
                "scipy is required for difference calculation.\n"
                "Install with: pip install scipy")
            return

        # Build KD-trees for fast nearest neighbor lookup
        tree_best = cKDTree(best_coords)
        tree_recent = cKDTree(recent_coords)

        # Find points in best with no match in recent
        distances_best, _ = tree_recent.query(best_coords)
        only_in_best_mask = distances_best > tolerance_voxels
        only_in_best = best_coords[only_in_best_mask]

        # Find points in recent with no match in best
        distances_recent, _ = tree_best.query(recent_coords)
        only_in_recent_mask = distances_recent > tolerance_voxels
        only_in_recent = recent_coords[only_in_recent_mask]

        # Calculate overlap (matched points)
        overlap_count = len(best_coords) - len(only_in_best)

        # Remove existing diff layers before adding new ones
        for layer in list(self.viewer.layers):
            if hasattr(layer, 'metadata') and layer.metadata.get('cell_type') in ['det_diff_only_best', 'det_diff_only_recent']:
                self.viewer.layers.remove(layer)

        # Create new difference layers
        if len(only_in_best) > 0:
            self._add_points_layer(
                only_in_best,
                f"Diff: Only in BEST ({len(only_in_best)})",
                'det_diff_only_best',
                visible=True
            )

        if len(only_in_recent) > 0:
            self._add_points_layer(
                only_in_recent,
                f"Diff: Only in RECENT ({len(only_in_recent)})",
                'det_diff_only_recent',
                visible=True
            )

        # Update stats display if available
        if hasattr(self, 'det_diff_stats_label'):
            self.det_diff_stats_label.setText(
                f"<b>Comparison Stats</b> (tolerance: {tolerance_voxels} voxels)<br>"
                f"Best: {len(best_coords)} | Recent: {len(recent_coords)}<br>"
                f"<span style='color:#00FF00'>Overlap: {overlap_count}</span> | "
                f"<span style='color:#FF0080'>Only BEST: {len(only_in_best)}</span> | "
                f"<span style='color:#00BFFF'>Only RECENT: {len(only_in_recent)}</span><br>"
                f"Net change: {len(recent_coords) - len(best_coords):+d}"
            )

        # Log to session documenter (if available and has the method)
        if self.session_doc and hasattr(self.session_doc, 'log_diff_generated'):
            self.session_doc.log_diff_generated(
                diff_type='detection',
                tolerance=tolerance_voxels,
                best_count=len(best_coords),
                recent_count=len(recent_coords),
                overlap=overlap_count,
                only_best=len(only_in_best),
                only_recent=len(only_in_recent),
            )

        print(f"[SCI-Connectome] Detection difference generated:")
        print(f"  Tolerance: {tolerance_voxels} voxels")
        print(f"  Best: {len(best_coords)}, Recent: {len(recent_coords)}")
        print(f"  Overlap: {overlap_count}, Only-Best: {len(only_in_best)}, Only-Recent: {len(only_in_recent)}")

    def load_previous_classification(self):
        """Load previous classification results."""
        print("[DEBUG] Button clicked: load_previous_classification")
        self._load_classified_cells()  # Use existing method
        self.prev_class_label.setText("Loaded from 5_Classified_Cells")

    def load_new_classification(self):
        """Load new classification results."""
        print("[DEBUG] Button clicked: load_new_classification")
        # For now, same as previous - will be different after running new classification
        self._load_classified_cells()
        self.new_class_label.setText("Loaded from 5_Classified_Cells")

    def _refresh_class_compare_runs(self):
        """Populate classification runs list from tracker."""
        print("[DEBUG] Button clicked: _refresh_class_compare_runs")
        from qtpy.QtWidgets import QListWidgetItem

        if not hasattr(self, 'class_compare_runs_list'):
            return

        self.class_compare_runs_list.clear()

        if not self.tracker or not self.current_brain:
            return

        # Get all classification runs for this brain
        runs = self.tracker.search(
            brain=self.current_brain.name,
            exp_type="classification",
            status="completed"
        )

        if not runs:
            return

        # Sort by creation time (newest first)
        runs.sort(key=lambda r: r.get('created_at', ''), reverse=True)

        for run in runs:
            exp_id = run.get('exp_id', '')
            created = run.get('created_at', '')[:16]
            cells = run.get('class_cells_found', '?')
            rejected = run.get('class_rejected', '?')
            model = run.get('class_model_path', 'default')
            if model and '/' in model:
                model = model.split('/')[-1]  # Just filename
            if model and '\\' in model:
                model = model.split('\\')[-1]

            # Build display text
            icons = ""
            if run.get('marked_best') == 'True':
                icons += "★ "
            if run.get('paradigm_best') == 'True':
                icons += "◆ "

            display_text = f"{icons}{created} | {cells} cells, {rejected} rej | {model}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, exp_id)
            self.class_compare_runs_list.addItem(item)

    def _get_selected_class_run_id(self):
        """Get exp_id from selected item in class compare runs list."""
        if not hasattr(self, 'class_compare_runs_list'):
            return None

        selected = self.class_compare_runs_list.selectedItems()
        if not selected:
            return None

        return selected[0].data(Qt.UserRole)

    def _load_class_run_as_previous(self):
        """Load selected classification run as previous (reference)."""
        print("[DEBUG] Button clicked: _load_class_run_as_previous")
        exp_id = self._get_selected_class_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a classification run from the list.")
            return

        self._load_classification_run(exp_id, as_previous=True)

    def _load_class_run_as_new(self):
        """Load selected classification run as new (comparison)."""
        print("[DEBUG] Button clicked: _load_class_run_as_new")
        exp_id = self._get_selected_class_run_id()
        if not exp_id:
            QMessageBox.warning(self, "No Selection", "Please select a classification run from the list.")
            return

        self._load_classification_run(exp_id, as_previous=False)

    def _load_classification_run(self, exp_id: str, as_previous: bool = True):
        """Load a classification run from tracker into napari (both cells and rejected)."""
        import numpy as np
        from brainglobe_utils.IO.cells import get_cells

        if not self.tracker:
            return

        run = self.tracker.get_experiment(exp_id)
        if not run:
            QMessageBox.warning(self, "Not Found", f"Run {exp_id} not found.")
            return

        output_path = run.get('output_path')
        if not output_path:
            QMessageBox.warning(self, "No Path", "Classification run has no output path.")
            return

        output_path = Path(output_path)
        cells_xml = output_path / "cells.xml"
        rejected_xml = output_path / "rejected.xml"

        if not cells_xml.exists():
            QMessageBox.warning(self, "Not Found", f"cells.xml not found at {output_path}")
            return

        created = run.get('created_at', '')[:16]
        prefix = "Prev" if as_previous else "New"

        # Load cells
        try:
            cells = get_cells(str(cells_xml))
            cell_count = len(cells) if cells else 0

            if cells:
                points = np.array([[c.z, c.y, c.x] for c in cells])
            else:
                points = np.empty((0, 3))

            # Determine cell_type based on previous/new
            if as_previous:
                cell_type = 'class_prev_cells'
                layer_name = f"{prefix}: {created} ({cell_count} cells)"
                label_widget = self.prev_class_label
            else:
                cell_type = 'class_new_cells'
                layer_name = f"{prefix}: {created} ({cell_count} cells)"
                label_widget = self.new_class_label

            # Add cells layer
            self._add_points_layer(points, layer_name, cell_type)

            # Update label
            if label_widget:
                label_widget.setText(layer_name)

            print(f"[SCI-Connectome] Loaded classification {exp_id}: {cell_count} cells")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load cells: {e}")
            return

        # Load rejected
        rejected_count = 0
        try:
            if rejected_xml.exists():
                rejected = get_cells(str(rejected_xml))
                rejected_count = len(rejected) if rejected else 0

                if rejected:
                    rejected_points = np.array([[c.z, c.y, c.x] for c in rejected])
                else:
                    rejected_points = np.empty((0, 3))
            else:
                rejected_points = np.empty((0, 3))

            # Add rejected layer with appropriate type
            if as_previous:
                rejected_type = 'class_prev_rejected'
                rejected_name = f"{prefix}: {created} ({rejected_count} rejected)"
            else:
                rejected_type = 'class_new_rejected'
                rejected_name = f"{prefix}: {created} ({rejected_count} rejected)"

            self._add_points_layer(rejected_points, rejected_name, rejected_type)
            print(f"[SCI-Connectome] Loaded rejected: {rejected_count}")

        except Exception as e:
            print(f"[SCI-Connectome] Note: Could not load rejected.xml: {e}")
            # Create empty rejected layer anyway
            if as_previous:
                self._add_points_layer(np.empty((0, 3)), f"{prefix}: (0 rejected)", 'class_prev_rejected')
            else:
                self._add_points_layer(np.empty((0, 3)), f"{prefix}: (0 rejected)", 'class_new_rejected')

    def mark_layer_as_prev_classification(self):
        """Mark selected layer as previous classification cells."""
        print("[DEBUG] Button clicked: mark_layer_as_prev_classification")
        import napari

        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            QMessageBox.warning(self, "No Selection",
                "Please select a points layer in napari first.")
            return

        layer = selected_layers[0]
        if not isinstance(layer, napari.layers.Points):
            QMessageBox.warning(self, "Wrong Layer Type",
                "Please select a Points layer, not an image layer.")
            return

        # Apply previous cells styling
        self._set_layer_type(layer, 'class_prev_cells')

        # Update name
        if not any(x in layer.name.lower() for x in ['prev', 'previous']):
            layer.name = f"Prev Cells: {layer.name}"

        # Update label
        if hasattr(self, 'prev_class_label'):
            self.prev_class_label.setText(f"Previous: {layer.name} ({len(layer.data)} cells)")

        # Log
        if self.session_doc:
            self.session_doc.log_event('mark_layer', {
                'action': 'mark_as_prev_classification',
                'layer_name': layer.name,
                'cell_count': len(layer.data),
            })

        print(f"[SCI-Connectome] Marked as previous classification: {layer.name}")

    def mark_layer_as_new_classification(self):
        """Mark selected layer as new classification cells."""
        print("[DEBUG] Button clicked: mark_layer_as_new_classification")
        import napari

        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            QMessageBox.warning(self, "No Selection",
                "Please select a points layer in napari first.")
            return

        layer = selected_layers[0]
        if not isinstance(layer, napari.layers.Points):
            QMessageBox.warning(self, "Wrong Layer Type",
                "Please select a Points layer, not an image layer.")
            return

        # Apply new cells styling
        self._set_layer_type(layer, 'class_new_cells')

        # Update name
        if not any(x in layer.name.lower() for x in ['new', 'classified']):
            layer.name = f"New Cells: {layer.name}"

        # Update label
        if hasattr(self, 'new_class_label'):
            self.new_class_label.setText(f"New: {layer.name} ({len(layer.data)} cells)")

        # Log
        if self.session_doc:
            self.session_doc.log_event('mark_layer', {
                'action': 'mark_as_new_classification',
                'layer_name': layer.name,
                'cell_count': len(layer.data),
            })

        print(f"[SCI-Connectome] Marked as new classification: {layer.name}")

    def generate_classification_difference(self):
        """Generate difference layers comparing two classification results.

        Creates two new layers:
        - GAINED: cells in new classification that weren't in previous
        - LOST: cells in previous classification that aren't in new

        Uses scipy.spatial.cKDTree for efficient nearest-neighbor matching.
        """
        print("[DEBUG] Button clicked: generate_classification_difference")
        import numpy as np

        # Get tolerance from UI if available (in voxels)
        tolerance = getattr(self, 'class_diff_tolerance_spin', None)
        tolerance_voxels = tolerance.value() if tolerance else 5.0

        prev_cells = self._get_layer_by_type('class_prev_cells')
        new_cells = self._get_layer_by_type('class_new_cells')

        if not prev_cells:
            QMessageBox.warning(self, "Missing Layer",
                "No 'previous cells' layer found.\n"
                "Load or mark a layer as previous classification first.")
            return

        if not new_cells:
            QMessageBox.warning(self, "Missing Layer",
                "No 'new cells' layer found.\n"
                "Load or run a new classification first.")
            return

        prev_coords = np.array(prev_cells.data)
        new_coords = np.array(new_cells.data)

        if len(prev_coords) == 0 and len(new_coords) == 0:
            QMessageBox.warning(self, "Empty Layers",
                "Both layers have no points.")
            return

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            QMessageBox.warning(self, "Missing scipy",
                "scipy is required for difference calculation.\n"
                "Install with: pip install scipy")
            return

        gained_coords = np.array([]).reshape(0, 3)
        lost_coords = np.array([]).reshape(0, 3)
        overlap_count = 0

        if len(prev_coords) > 0 and len(new_coords) > 0:
            # Build KD-trees
            tree_prev = cKDTree(prev_coords)
            tree_new = cKDTree(new_coords)

            # GAINED: in new but not in prev
            distances_new, _ = tree_prev.query(new_coords)
            gained_mask = distances_new > tolerance_voxels
            gained_coords = new_coords[gained_mask]

            # LOST: in prev but not in new
            distances_prev, _ = tree_new.query(prev_coords)
            lost_mask = distances_prev > tolerance_voxels
            lost_coords = prev_coords[lost_mask]

            overlap_count = len(prev_coords) - len(lost_coords)

        elif len(new_coords) > 0:
            # All new are gained
            gained_coords = new_coords

        elif len(prev_coords) > 0:
            # All prev are lost
            lost_coords = prev_coords

        # Remove existing diff layers before adding new ones
        for layer in list(self.viewer.layers):
            if hasattr(layer, 'metadata') and layer.metadata.get('cell_type') in ['class_diff_gained', 'class_diff_lost']:
                self.viewer.layers.remove(layer)

        # Create new difference layers
        if len(gained_coords) > 0:
            self._add_points_layer(
                gained_coords,
                f"GAINED: {len(gained_coords)} new cells",
                'class_diff_gained',
                visible=True
            )

        if len(lost_coords) > 0:
            self._add_points_layer(
                lost_coords,
                f"LOST: {len(lost_coords)} cells",
                'class_diff_lost',
                visible=True
            )

        # Update stats display if available
        if hasattr(self, 'class_diff_stats'):
            self.class_diff_stats.setText(
                f"<b>Classification Comparison</b> (tolerance: {tolerance_voxels} voxels)<br>"
                f"Previous: {len(prev_coords)} cells | New: {len(new_coords)} cells<br>"
                f"<span style='color:#00FF00'>Overlap: {overlap_count}</span> | "
                f"<span style='color:#7FFF00'>Gained: {len(gained_coords)}</span> | "
                f"<span style='color:#FF4500'>Lost: {len(lost_coords)}</span><br>"
                f"Net change: {len(new_coords) - len(prev_coords):+d}"
            )

        # Log to session documenter (if available and has the method)
        if self.session_doc and hasattr(self.session_doc, 'log_diff_generated'):
            self.session_doc.log_diff_generated(
                diff_type='classification',
                tolerance=tolerance_voxels,
                best_count=len(prev_coords),
                recent_count=len(new_coords),
                overlap=overlap_count,
                only_best=len(lost_coords),
                only_recent=len(gained_coords),
            )

        print(f"[SCI-Connectome] Classification difference generated:")
        print(f"  Tolerance: {tolerance_voxels} voxels")
        print(f"  Previous: {len(prev_coords)}, New: {len(new_coords)}")
        print(f"  Overlap: {overlap_count}, Gained: {len(gained_coords)}, Lost: {len(lost_coords)}")

    def refresh_training_data_counts(self):
        """Refresh the training data counts display."""
        print("[DEBUG] Button clicked: refresh_training_data_counts")
        # Check for training layers
        train_cells = self._get_layer_by_type('train_cells')
        train_non_cells = self._get_layer_by_type('train_non_cells')

        cells_count = len(train_cells.data) if train_cells else 0
        non_cells_count = len(train_non_cells.data) if train_non_cells else 0

        self.train_cells_count_label.setText(f"Cells bucket: {cells_count} points")
        self.train_non_cells_count_label.setText(f"Non-cells bucket: {non_cells_count} points")

    def _on_candidate_confirmed(self, index: int):
        """Handle candidate confirmation from curation widget.

        The curation widget emits the index of the confirmed candidate.
        We get the actual coordinates from the curation widget's candidates array.
        """
        import numpy as np

        # Get the actual coordinates from the curation widget
        if not hasattr(self, 'curation_widget') or self.curation_widget is None:
            print(f"[SCI-Connectome] Candidate confirmed (index {index}) - no curation widget available")
            return

        if self.curation_widget.candidates is None:
            print(f"[SCI-Connectome] Candidate confirmed (index {index}) - no candidates loaded")
            return

        if index < 0 or index >= len(self.curation_widget.candidates):
            print(f"[SCI-Connectome] Candidate confirmed - index {index} out of range")
            return

        coords = np.array([self.curation_widget.candidates[index]])
        print(f"[SCI-Connectome] Candidate confirmed: index {index}, coords {coords[0]}")

        # Find or create training cells bucket
        train_layer = self._get_layer_by_type('train_cells')
        if train_layer:
            existing_coords = np.array(train_layer.data)
            new_coords = np.vstack([existing_coords, coords])
            train_layer.data = new_coords
        else:
            train_layer = self._add_points_layer(
                coords,
                f"Training Cells ({len(coords)})",
                'train_cells',
                visible=True
            )

        # Update display
        self.refresh_training_data_counts()

        # Auto-save to XML (no dialog, just save)
        self._append_to_training_xml(coords[0], 'cells')

        # Log
        if self.session_doc:
            self.session_doc.log_event('curation_confirmed', {
                'index': index,
                'point': coords.tolist(),
                'total_in_bucket': len(train_layer.data),
            })

    def _on_candidate_rejected(self, index: int):
        """Handle candidate rejection from curation widget.

        The curation widget emits the index of the rejected candidate.
        We get the actual coordinates from the curation widget's candidates array.
        """
        import numpy as np

        # Get the actual coordinates from the curation widget
        if not hasattr(self, 'curation_widget') or self.curation_widget is None:
            print(f"[SCI-Connectome] Candidate rejected (index {index}) - no curation widget available")
            return

        if self.curation_widget.candidates is None:
            print(f"[SCI-Connectome] Candidate rejected (index {index}) - no candidates loaded")
            return

        if index < 0 or index >= len(self.curation_widget.candidates):
            print(f"[SCI-Connectome] Candidate rejected - index {index} out of range")
            return

        coords = np.array([self.curation_widget.candidates[index]])
        print(f"[SCI-Connectome] Candidate rejected: index {index}, coords {coords[0]}")

        # Find or create training non-cells bucket
        train_layer = self._get_layer_by_type('train_non_cells')
        if train_layer:
            existing_coords = np.array(train_layer.data)
            new_coords = np.vstack([existing_coords, coords])
            train_layer.data = new_coords
        else:
            train_layer = self._add_points_layer(
                coords,
                f"Training Non-Cells ({len(coords)})",
                'train_non_cells',
                visible=True
            )

        # Update display
        self.refresh_training_data_counts()

        # Auto-save to XML (no dialog, just save)
        self._append_to_training_xml(coords[0], 'non_cells')

        # Log
        if self.session_doc:
            self.session_doc.log_event('curation_rejected', {
                'index': index,
                'point': coords.tolist(),
                'total_in_bucket': len(train_layer.data),
            })

    def add_confirmed_to_training(self):
        """Add confirmed cells to training bucket.

        Takes selected points from any points layer and adds them to the
        'train_cells' bucket layer for later export to training data.
        """
        print("[DEBUG] Button clicked: add_confirmed_to_training")
        import napari
        import numpy as np

        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            QMessageBox.warning(self, "No Selection",
                "Please select a points layer in napari first.")
            return

        layer = selected_layers[0]
        if not isinstance(layer, napari.layers.Points):
            QMessageBox.warning(self, "Wrong Layer Type",
                "Please select a Points layer.")
            return

        # Get selected points within the layer (if any)
        if hasattr(layer, 'selected_data') and len(layer.selected_data) > 0:
            selected_indices = list(layer.selected_data)
            coords_to_add = layer.data[selected_indices]
        else:
            # If no points selected within layer, use all points
            coords_to_add = layer.data

        if len(coords_to_add) == 0:
            QMessageBox.warning(self, "No Points",
                "Layer has no points to add.")
            return

        # Find or create training cells bucket
        train_layer = self._get_layer_by_type('train_cells')
        if train_layer:
            # Append to existing
            existing_coords = np.array(train_layer.data)
            new_coords = np.vstack([existing_coords, coords_to_add])
            train_layer.data = new_coords
        else:
            # Create new layer
            train_layer = self._add_points_layer(
                coords_to_add,
                f"Training Cells ({len(coords_to_add)})",
                'train_cells',
                visible=True
            )

        # Update count display
        self.refresh_training_data_counts()

        # Log
        if self.session_doc:
            self.session_doc.log_event('training_data_added', {
                'type': 'cells',
                'count': len(coords_to_add),
                'source_layer': layer.name,
                'total_in_bucket': len(train_layer.data),
            })

        print(f"[SCI-Connectome] Added {len(coords_to_add)} points to training cells bucket")

    def add_rejected_to_training(self):
        """Add rejected cells to training non-cells bucket.

        Takes selected points from any points layer and adds them to the
        'train_non_cells' bucket layer for later export to training data.
        """
        print("[DEBUG] Button clicked: add_rejected_to_training")
        import napari
        import numpy as np

        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            QMessageBox.warning(self, "No Selection",
                "Please select a points layer in napari first.")
            return

        layer = selected_layers[0]
        if not isinstance(layer, napari.layers.Points):
            QMessageBox.warning(self, "Wrong Layer Type",
                "Please select a Points layer.")
            return

        # Get selected points within the layer (if any)
        if hasattr(layer, 'selected_data') and len(layer.selected_data) > 0:
            selected_indices = list(layer.selected_data)
            coords_to_add = layer.data[selected_indices]
        else:
            # If no points selected within layer, use all points
            coords_to_add = layer.data

        if len(coords_to_add) == 0:
            QMessageBox.warning(self, "No Points",
                "Layer has no points to add.")
            return

        # Find or create training non-cells bucket
        train_layer = self._get_layer_by_type('train_non_cells')
        if train_layer:
            # Append to existing
            existing_coords = np.array(train_layer.data)
            new_coords = np.vstack([existing_coords, coords_to_add])
            train_layer.data = new_coords
        else:
            # Create new layer
            train_layer = self._add_points_layer(
                coords_to_add,
                f"Training Non-Cells ({len(coords_to_add)})",
                'train_non_cells',
                visible=True
            )

        # Update count display
        self.refresh_training_data_counts()

        # Log
        if self.session_doc:
            self.session_doc.log_event('training_data_added', {
                'type': 'non_cells',
                'count': len(coords_to_add),
                'source_layer': layer.name,
                'total_in_bucket': len(train_layer.data),
            })

        print(f"[SCI-Connectome] Added {len(coords_to_add)} points to training non-cells bucket")

    def visualize_training_data(self):
        """Visualize existing training data from cellfinder training folders.

        Loads cells from:
        - {brain}/training_data/cells/
        - {brain}/training_data/non_cells/
        """
        print("[DEBUG] Button clicked: visualize_training_data")
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain",
                "Load a brain first to visualize its training data.")
            return

        brain_path = BRAINS_ROOT / self.current_brain
        training_path = brain_path / "training_data"

        if not training_path.exists():
            QMessageBox.information(self, "No Training Data",
                f"No training data folder found for {self.current_brain}.\n"
                f"Expected path: {training_path}")
            return

        cells_loaded = 0
        non_cells_loaded = 0

        # Load cells
        cells_dir = training_path / "cells"
        if cells_dir.exists():
            cells_coords = self._load_training_coords(cells_dir)
            if len(cells_coords) > 0:
                self._add_points_layer(
                    cells_coords,
                    f"Training Cells ({len(cells_coords)})",
                    'train_cells',
                    visible=True
                )
                cells_loaded = len(cells_coords)

        # Load non-cells
        non_cells_dir = training_path / "non_cells"
        if non_cells_dir.exists():
            non_cells_coords = self._load_training_coords(non_cells_dir)
            if len(non_cells_coords) > 0:
                self._add_points_layer(
                    non_cells_coords,
                    f"Training Non-Cells ({len(non_cells_coords)})",
                    'train_non_cells',
                    visible=True
                )
                non_cells_loaded = len(non_cells_coords)

        # Update display
        self.refresh_training_data_counts()

        if cells_loaded > 0 or non_cells_loaded > 0:
            QMessageBox.information(self, "Training Data Loaded",
                f"Loaded training data:\n"
                f"  Cells: {cells_loaded}\n"
                f"  Non-cells: {non_cells_loaded}")
        else:
            QMessageBox.information(self, "No Training Data",
                f"Training folder exists but contains no data.")

    def _load_training_coords(self, folder):
        """Load cell coordinates from training folder (contains .tif cubes).

        Training data is stored as 3D TIFF cubes, one per cell.
        The filename encodes the Z, Y, X position.
        Format: z{z}_y{y}_x{x}.tif
        """
        import numpy as np
        import re

        coords = []
        pattern = re.compile(r'z(\d+)_y(\d+)_x(\d+)\.tif')

        for tif_file in folder.glob("*.tif"):
            match = pattern.match(tif_file.name)
            if match:
                z, y, x = int(match.group(1)), int(match.group(2)), int(match.group(3))
                coords.append([z, y, x])

        return np.array(coords) if coords else np.array([]).reshape(0, 3)

    def create_training_layers(self):
        """Create training bucket layers with default save location.

        This is the streamlined workflow entry point:
        1. Creates 'Train: Cells' and 'Train: Non-Cells' layers
        2. Sets default save path to {brain}/training_data/
        3. Enables auto-save on every Y/N during curation
        """
        print("[DEBUG] Button clicked: create_training_layers")
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain",
                "Load a brain first before creating training layers.")
            return

        # Determine brain path - use existing path if already a Path, don't reconstruct
        if isinstance(self.current_brain, Path):
            brain_path = self.current_brain
        else:
            brain_path = Path(self.current_brain)

        # Extract imaging paradigm from brain name for paradigm-specific training
        from braintools.config import parse_brain_name
        brain_name = brain_path.name
        parsed = parse_brain_name(brain_name)
        imaging_paradigm = parsed.get('imaging_params', 'default')

        if not imaging_paradigm:
            imaging_paradigm = 'default'
            print(f"[SCI-Connectome] Warning: Could not parse imaging params from '{brain_name}', using 'default'")

        # Set up training data directory in MODELS_DIR, organized by imaging paradigm
        # This allows pooling training data across all brains with same imaging settings
        self.training_data_dir = MODELS_DIR / imaging_paradigm / "training_data"
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SCI-Connectome] Training data location: {self.training_data_dir}")

        # Set up auto-save XML files
        self._cells_xml = self.training_data_dir / "curated_cells.xml"
        self._non_cells_xml = self.training_data_dir / "curated_non_cells.xml"

        # Create or get training layers
        train_cells = self._get_layer_by_type('train_cells')
        if not train_cells:
            import numpy as np
            train_cells = self._add_points_layer(
                np.empty((0, 3)),
                "Train: Cells (0)",
                'train_cells',
                visible=True
            )

        train_non_cells = self._get_layer_by_type('train_non_cells')
        if not train_non_cells:
            import numpy as np
            train_non_cells = self._add_points_layer(
                np.empty((0, 3)),
                "Train: Non-Cells (0)",
                'train_non_cells',
                visible=True
            )

        # Update UI
        self.training_path_label.setText(f"Auto-save: {self.training_data_dir}")
        self.training_path_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.training_data_path.setText(str(self.training_data_dir))

        # Load existing curated data if any
        cells_count = 0
        non_cells_count = 0
        if self._cells_xml.exists():
            coords = self._load_xml_coordinates(self._cells_xml)
            if len(coords) > 0:
                train_cells.data = coords
                cells_count = len(coords)
        if self._non_cells_xml.exists():
            coords = self._load_xml_coordinates(self._non_cells_xml)
            if len(coords) > 0:
                train_non_cells.data = coords
                non_cells_count = len(coords)

        self.refresh_training_data_counts()

        print(f"[SCI-Connectome] Training layers created. Auto-save to: {self.training_data_dir}")
        if cells_count > 0 or non_cells_count > 0:
            print(f"[SCI-Connectome] Loaded existing curated data: {cells_count} cells, {non_cells_count} non-cells")

    def _load_xml_coordinates(self, xml_path):
        """Load coordinates from an XML file."""
        import numpy as np
        import xml.etree.ElementTree as ET

        coords = []
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            for marker in root.iter('Marker'):
                x_el = marker.find('MarkerX')
                y_el = marker.find('MarkerY')
                z_el = marker.find('MarkerZ')
                if x_el is not None and y_el is not None and z_el is not None:
                    x = int(x_el.text)
                    y = int(y_el.text)
                    z = int(z_el.text)
                    coords.append([z, y, x])  # napari uses ZYX order
        except Exception as e:
            print(f"[SCI-Connectome] Could not load {xml_path}: {e}")

        return np.array(coords) if coords else np.array([]).reshape(0, 3)

    def _append_to_training_xml(self, coord, cell_type: str):
        """Append a single point to the training XML file.

        This is called automatically on every Y/N during curation.
        No dialog, no user interaction - just saves immediately.
        """
        import xml.etree.ElementTree as ET

        if cell_type == 'cells':
            xml_path = self._cells_xml
        else:
            xml_path = self._non_cells_xml

        if xml_path is None:
            # Training layers not set up yet
            return

        # Load existing or create new XML structure
        if xml_path.exists():
            try:
                tree = ET.parse(str(xml_path))
                root = tree.getroot()
            except:
                root = self._create_xml_root(cell_type)
                tree = ET.ElementTree(root)
        else:
            root = self._create_xml_root(cell_type)
            tree = ET.ElementTree(root)

        # Find or create Marker_Type element
        marker_type = root.find('.//Marker_Type')
        if marker_type is None:
            marker_data = root.find('Marker_Data')
            if marker_data is None:
                marker_data = ET.SubElement(root, 'Marker_Data')
            marker_type = ET.SubElement(marker_data, 'Marker_Type')
            ET.SubElement(marker_type, 'Type').text = '1'

        # Add new marker
        marker = ET.SubElement(marker_type, 'Marker')
        z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
        ET.SubElement(marker, 'MarkerX').text = str(x)
        ET.SubElement(marker, 'MarkerY').text = str(y)
        ET.SubElement(marker, 'MarkerZ').text = str(z)

        # Save immediately
        tree.write(str(xml_path), encoding='UTF-8', xml_declaration=True)

    def _create_xml_root(self, cell_type: str):
        """Create XML root structure for training data."""
        import xml.etree.ElementTree as ET

        root = ET.Element('CellCounter_Marker_File')
        image_props = ET.SubElement(root, 'Image_Properties')
        ET.SubElement(image_props, 'Image_Filename').text = 'training_data'

        marker_data = ET.SubElement(root, 'Marker_Data')
        current_type = ET.SubElement(marker_data, 'Current_Type')
        current_type.text = '1'

        marker_type = ET.SubElement(marker_data, 'Marker_Type')
        type_el = ET.SubElement(marker_type, 'Type')
        type_el.text = '1'

        return root

    def _create_training_yaml(self):
        """Create the training.yml file required by BrainGlobe/cellfinder."""
        import yaml

        if not self.training_data_dir:
            return

        yaml_content = {
            'data': [
                {
                    'bg_channel': 1,
                    'cell_def': '',
                    'cube_dir': str(self.training_data_dir / 'cells'),
                    'signal_channel': 0,
                    'type': 'cell'
                },
                {
                    'bg_channel': 1,
                    'cell_def': '',
                    'cube_dir': str(self.training_data_dir / 'non_cells'),
                    'signal_channel': 0,
                    'type': 'no_cell'
                }
            ]
        }

        yaml_path = self.training_data_dir / 'training.yml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"[SCI-Connectome] Created {yaml_path}")

    def export_training_data(self):
        """Export training bucket layers to BrainGlobe/cellfinder training format.

        Creates TIFF cubes in {brain}/training_data/cells/ and non_cells/
        for training a custom classification network.

        BrainGlobe format (VERIFIED from actual training data):
        - Files: pCellz{z}y{y}x{x}Ch{channel}.tif
        - BOTH channels saved (Ch0 = signal, Ch1 = background)
        - Fixed cube size: 50x50 XY, 20 Z
        - Creates training.yml config file
        """
        print("[DEBUG] Button clicked: export_training_data")
        import napari
        import numpy as np
        import tifffile
        from qtpy.QtWidgets import QApplication

        if not self.current_brain:
            QMessageBox.warning(self, "No Brain",
                "Load a brain first to export training data.")
            return

        # Get training bucket layers
        train_cells = self._get_layer_by_type('train_cells')
        train_non_cells = self._get_layer_by_type('train_non_cells')

        cells_count = len(train_cells.data) if train_cells else 0
        non_cells_count = len(train_non_cells.data) if train_non_cells else 0

        if cells_count == 0 and non_cells_count == 0:
            QMessageBox.warning(self, "No Training Data",
                "No training data in buckets to export.\n"
                "Click 'Create Training Layers' first, then curate some candidates.")
            return

        # Get BOTH signal and background layers
        signal_layer = None
        bg_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                name_lower = layer.name.lower()
                if 'signal' in name_lower:
                    signal_layer = layer
                elif 'background' in name_lower or 'bg' in name_lower:
                    bg_layer = layer
                else:
                    # Fallback assignments
                    if signal_layer is None:
                        signal_layer = layer
                    elif bg_layer is None:
                        bg_layer = layer

        if signal_layer is None:
            QMessageBox.warning(self, "No Image",
                "No signal layer found to extract training cubes from.\n"
                "Load brain data first.")
            return

        # BrainGlobe uses fixed cube sizes: 50x50 XY, 20 Z
        cube_xy = 50
        cube_z = 20
        half_xy = cube_xy // 2
        half_z = cube_z // 2

        # Get image data
        signal_data = np.asarray(signal_layer.data)
        z_max, y_max, x_max = signal_data.shape

        # Get background data if available
        has_bg = bg_layer is not None
        if has_bg:
            bg_data = np.asarray(bg_layer.data)
        else:
            print("[SCI-Connectome] Warning: No background layer found, will only export signal channel")
            bg_data = None

        # Set up training data directory
        if self.training_data_dir is None:
            brain_name = self.current_brain.name if hasattr(self.current_brain, 'name') else str(self.current_brain)
            brain_path = BRAINS_ROOT / brain_name
            self.training_data_dir = brain_path / "training_data"

        cells_dir = self.training_data_dir / "cells"
        non_cells_dir = self.training_data_dir / "non_cells"

        cells_dir.mkdir(parents=True, exist_ok=True)
        non_cells_dir.mkdir(parents=True, exist_ok=True)

        cells_exported = 0
        non_cells_exported = 0
        skipped = 0

        self.export_status_label.setText("Exporting cubes (both channels)...")
        QApplication.processEvents()

        def export_cube(coord, output_dir):
            """Export a single cube with both channels."""
            nonlocal skipped
            z, y, x = int(coord[0]), int(coord[1]), int(coord[2])

            # Check bounds with BrainGlobe cube sizes
            if (z - half_z < 0 or z + half_z > z_max or
                y - half_xy < 0 or y + half_xy > y_max or
                x - half_xy < 0 or x + half_xy > x_max):
                skipped += 1
                return False

            # Extract signal cube (Ch0)
            signal_cube = signal_data[
                z - half_z:z + half_z,
                y - half_xy:y + half_xy,
                x - half_xy:x + half_xy
            ]

            # Save signal channel with BrainGlobe naming
            filename_ch0 = output_dir / f"pCellz{z}y{y}x{x}Ch0.tif"
            tifffile.imwrite(str(filename_ch0), signal_cube.astype(np.uint16))

            # Extract and save background cube (Ch1) if available
            if bg_data is not None:
                bg_cube = bg_data[
                    z - half_z:z + half_z,
                    y - half_xy:y + half_xy,
                    x - half_xy:x + half_xy
                ]
                filename_ch1 = output_dir / f"pCellz{z}y{y}x{x}Ch1.tif"
                tifffile.imwrite(str(filename_ch1), bg_cube.astype(np.uint16))

            return True

        # Export cells
        if train_cells and len(train_cells.data) > 0:
            for coord in train_cells.data:
                if export_cube(coord, cells_dir):
                    cells_exported += 1

        # Export non-cells
        if train_non_cells and len(train_non_cells.data) > 0:
            for coord in train_non_cells.data:
                if export_cube(coord, non_cells_dir):
                    non_cells_exported += 1

        # Create training.yml config file
        self._create_training_yaml()

        # Log to tracker
        if self.tracker:
            try:
                brain_name = self.current_brain.name if hasattr(self.current_brain, 'name') else str(self.current_brain)
                self.tracker.log_training(
                    brain=brain_name,
                    epochs=0,  # Not training yet, just exporting
                    output_path=str(self.training_data_dir),
                    notes=f"Training data export (BrainGlobe format): {cells_exported} cells, {non_cells_exported} non-cells",
                    status="data_exported"
                )
            except Exception as e:
                print(f"[SCI-Connectome] Could not log export to tracker: {e}")

        # Log to session
        if self.session_doc:
            self.session_doc.log_event('training_data_exported', {
                'cells': cells_exported,
                'non_cells': non_cells_exported,
                'skipped': skipped,
                'format': 'brainglobe',
                'output_path': str(self.training_data_dir),
            })

        # Update status
        status_msg = f"Exported: {cells_exported} cells, {non_cells_exported} non-cells"
        if skipped > 0:
            status_msg += f" ({skipped} skipped)"
        self.export_status_label.setText(status_msg)

        channels_note = "both channels" if has_bg else "signal channel only"
        QMessageBox.information(self, "Training Dataset Created",
            f"BrainGlobe training data created:\n{self.training_data_dir}\n\n"
            f"Cells: {cells_exported} (x2 files = {channels_note})\n"
            f"Non-cells: {non_cells_exported}\n"
            f"Cube size: 50x50x20 voxels\n"
            f"Config: training.yml created\n"
            f"{f'Skipped (out of bounds): {skipped}' if skipped > 0 else ''}\n\n"
            f"Click 'Train' to start training with this data.")

        print(f"[SCI-Connectome] Training dataset exported to {self.training_data_dir}")

    def create_training_tab(self):
        """Tab for classification and network training."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        info = QLabel(
            "Classify detected candidates and optionally train a custom network.\n\n"
            "Workflow: Detection → Classification → (optional) Curation → Training → Repeat"
        )
        info.setWordWrap(True)
        content_layout.addWidget(info)

        # =====================================================================
        # STEP 1: Model Selection
        # =====================================================================
        model_group = QGroupBox("1. Select Model")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        model_layout.addWidget(QLabel(
            "Choose which model to use for classification.\n"
            "Start with the default, then switch to your trained model."
        ))

        self.model_combo = QComboBox()
        self.model_combo.addItem("Default (BrainGlobe pretrained)")
        self.refresh_models()
        model_layout.addWidget(self.model_combo)

        refresh_models_btn = QPushButton("Refresh Model List")
        refresh_models_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(refresh_models_btn)

        content_layout.addWidget(model_group)

        # =====================================================================
        # STEP 2: Classification - Apply model to candidates
        # =====================================================================
        classify_group = QGroupBox("2. Classify Candidates")
        classify_layout = QVBoxLayout()
        classify_group.setLayout(classify_layout)

        classify_layout.addWidget(QLabel(
            "Apply the selected model to detected cell candidates.\n"
            "This filters out false positives, keeping only real cells.\n"
            "(Requires detection results in 4_Cell_Candidates folder)"
        ))

        # Classification parameters
        param_form = QFormLayout()

        self.classify_cube_size = QSpinBox()
        self.classify_cube_size.setRange(20, 100)
        self.classify_cube_size.setValue(50)
        self.classify_cube_size.setToolTip("Size of image cube extracted around each candidate")
        param_form.addRow("Cube Size:", self.classify_cube_size)

        self.classify_batch_size = QSpinBox()
        self.classify_batch_size.setRange(1, 128)
        self.classify_batch_size.setValue(32)
        self.classify_batch_size.setToolTip("Batch size for classification (higher = faster, more memory)")
        param_form.addRow("Batch Size:", self.classify_batch_size)

        classify_layout.addLayout(param_form)

        # Run classification button
        self.classify_btn = QPushButton("Run Classification")
        self.classify_btn.clicked.connect(self.run_classification)
        self.classify_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        classify_layout.addWidget(self.classify_btn)

        # Classification status
        self.classify_status = QLabel("")
        self.classify_status.setWordWrap(True)
        classify_layout.addWidget(self.classify_status)

        content_layout.addWidget(classify_group)

        # =====================================================================
        # STEP 3: Curation (Optional - for training custom models)
        # =====================================================================
        curate_group = QGroupBox("3. Curate Training Data (Optional)")
        curate_layout = QVBoxLayout()
        curate_group.setLayout(curate_layout)

        curate_layout.addWidget(QLabel(
            "Review classified cells and mark corrections.\n"
            "Use C = confirm cell, X = reject (not a cell).\n"
            "This creates training data for a custom network."
        ))

        # Embed the actual curation widget
        try:
            from braintools.pipeline_3d.curation_widget import CurationWidget
            self.curation_widget = CurationWidget(self.viewer)
            curate_layout.addWidget(self.curation_widget)
        except Exception as e:
            error_label = QLabel(f"Could not load curation widget: {e}")
            error_label.setStyleSheet("color: red;")
            curate_layout.addWidget(error_label)

        content_layout.addWidget(curate_group)

        # =====================================================================
        # STEP 4: Train Custom Network (Optional)
        # =====================================================================
        train_group = QGroupBox("4. Train Custom Network (Optional)")
        train_layout = QVBoxLayout()
        train_group.setLayout(train_layout)

        train_layout.addWidget(QLabel(
            "Train a network on your curated data to improve accuracy.\n"
            "This typically takes 30-60 minutes on GPU."
        ))

        self.training_data_path = QLineEdit()
        self.training_data_path.setPlaceholderText("Path to curated training data...")
        train_layout.addWidget(self.training_data_path)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_training_data)
        train_layout.addWidget(browse_btn)

        train_btn = QPushButton("Start Training")
        train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(train_btn)

        content_layout.addWidget(train_group)

        content_layout.addStretch()
        return widget

    def open_curation(self):
        """Open the curation tool - now embedded in Training tab."""
        # Switch to training tab where curation is embedded
        self.tabs.setCurrentIndex(4)  # Training tab
        QMessageBox.information(
            self, "Curation",
            "The curation tool is now embedded in the Training tab.\n\n"
            "1. Load your detection results\n"
            "2. Use the curation widget to mark cells\n"
            "3. Save the curated data for training"
        )

    def browse_training_data(self):
        """Browse for training data folder."""
        print("[DEBUG] Button clicked: browse_training_data")
        folder = QFileDialog.getExistingDirectory(self, "Select Training Data Folder")
        if folder:
            self.training_data_path.setText(folder)

    def start_training(self):
        """Start network training with auto-detection of all paths.

        Streamlined workflow:
        1. Auto-finds training data from training_data_dir or training_data_path
        2. Finds most recent model to continue from (or starts fresh)
        3. Optionally runs classification when training completes
        """
        print("[DEBUG] Button clicked: start_training")
        # Auto-detect training data location
        training_base = None
        if self.training_data_dir and self.training_data_dir.exists():
            training_base = self.training_data_dir
        elif self.training_data_path.text():
            training_base = Path(self.training_data_path.text())

        if not training_base or not training_base.exists():
            QMessageBox.warning(self, "No Training Data",
                "Training data folder not found.\n\n"
                "Click 'Create Training Layers', curate some candidates,\n"
                "then click 'Create Training Dataset' first.")
            return

        cells_path = training_base / "cells"
        non_cells_path = training_base / "non_cells"

        # Check for TIFF cubes
        n_cells = len(list(cells_path.glob("*.tif*"))) if cells_path.exists() else 0
        n_non_cells = len(list(non_cells_path.glob("*.tif*"))) if non_cells_path.exists() else 0

        if n_cells == 0 or n_non_cells == 0:
            QMessageBox.warning(self, "Missing Training Data",
                f"Training data cubes not found:\n"
                f"  Cells: {n_cells} cubes\n"
                f"  Non-cells: {n_non_cells} cubes\n\n"
                f"Click 'Create Training Dataset' first to export cubes.")
            return

        script = SCRIPTS_DIR / "util_train_model.py"
        if not script.exists():
            QMessageBox.warning(self, "Error",
                f"Training script not found:\n{script}")
            return

        # Build command - always train fresh on all accumulated data
        # This is the proper ML approach: no double-counting from continue-from
        cmd = [
            sys.executable, str(script),
            "--cells", str(cells_path),
            "--non-cells", str(non_cells_path)
        ]

        # Store current model count before training
        self._models_before_training = self._count_models()

        # Check if auto-classify is enabled
        auto_classify = getattr(self, 'auto_classify_after_training', None)
        if auto_classify and auto_classify.isChecked():
            # Store current detection for later classification
            self._training_detection_exp_id = getattr(self, 'last_run_id', None)
            print(f"[SCI-Connectome] Training started. Will auto-classify detection: {self._training_detection_exp_id}")

            self._training_process = subprocess.Popen(cmd)

            # Start a timer to check for training completion
            from qtpy.QtCore import QTimer
            self._training_check_timer = QTimer()
            self._training_check_timer.timeout.connect(self._check_training_complete)
            self._training_check_timer.start(5000)  # Check every 5 seconds

            continue_note = f"Continuing from: {recent_model.name}" if recent_model else "Starting fresh (no prior model)"
            QMessageBox.information(
                self, "Training Started",
                f"Training started in background.\n\n"
                f"Data: {n_cells} cells, {n_non_cells} non-cells\n"
                f"{continue_note}\n\n"
                f"Classification will run automatically when training completes.\n"
                f"You can continue working while training runs."
            )
        else:
            # Simple training without auto-classify
            subprocess.Popen(cmd)

            continue_note = f"Continuing from: {recent_model.name}" if recent_model else "Starting fresh (no prior model)"
            QMessageBox.information(
                self, "Training Started",
                f"Training started in background.\n\n"
                f"Data: {n_cells} cells, {n_non_cells} non-cells\n"
                f"{continue_note}\n\n"
                f"Check the console for progress."
            )

        print(f"[SCI-Connectome] Training started: {n_cells} cells, {n_non_cells} non-cells")
        if recent_model:
            print(f"[SCI-Connectome] Continuing from model: {recent_model}")

    def _find_most_recent_model(self):
        """Find the most recent model file to continue training from."""
        if not MODELS_DIR.exists():
            return None

        # Look for model folders sorted by date
        model_folders = sorted(MODELS_DIR.iterdir(), reverse=True)

        for folder in model_folders:
            if not folder.is_dir():
                continue
            # Look for model.keras or model.h5 in folder
            for model_file in ['model.keras', 'model.h5']:
                model_path = folder / model_file
                if model_path.exists():
                    return model_path

        return None

    def _count_models(self) -> int:
        """Count the number of model files in MODELS_DIR."""
        if not MODELS_DIR.exists():
            return 0

        count = 0
        for item in MODELS_DIR.rglob("*.keras"):
            count += 1
        for item in MODELS_DIR.rglob("*.h5"):
            count += 1
        return count

    def _check_training_complete(self):
        """Check if training has completed by detecting new model files."""
        if not hasattr(self, '_training_process'):
            return

        # Check if process is still running
        poll_result = self._training_process.poll()

        if poll_result is not None:
            # Process finished
            self._training_check_timer.stop()

            # Check if new model was created
            models_after = self._count_models()
            models_before = getattr(self, '_models_before_training', 0)

            if models_after > models_before:
                # New model detected!
                newest_model = self._get_newest_model()
                model_name = Path(newest_model).parent.name + "/" + Path(newest_model).name if newest_model else "unknown"
                print(f"[SCI-Connectome] Training complete! New model: {model_name}")
                self.refresh_models()

                # Auto-classify if checkbox is enabled and we have a detection
                auto_classify_enabled = getattr(self, 'auto_classify_after_training', None)
                detection_exp_id = getattr(self, '_training_detection_exp_id', None)

                if auto_classify_enabled and auto_classify_enabled.isChecked() and detection_exp_id:
                    # Run classification automatically (no confirmation dialog)
                    print(f"[SCI-Connectome] Auto-classifying detection {detection_exp_id}...")
                    self._auto_classify_with_new_model()
                elif detection_exp_id:
                    # Checkbox unchecked but we have a detection - just notify
                    print(f"[SCI-Connectome] Training complete. Auto-classify disabled, skipping classification.")
                else:
                    print(f"[SCI-Connectome] Training complete. No detection loaded to classify.")
            else:
                QMessageBox.warning(
                    self, "Training Issue",
                    f"Training process exited (code: {poll_result}) "
                    f"but no new model was detected.\n\n"
                    f"Check the console for errors."
                )

    def _auto_classify_with_new_model(self):
        """Run classification with the newest model on the stored detection."""
        if not hasattr(self, '_training_detection_exp_id') or not self._training_detection_exp_id:
            print("[SCI-Connectome] No detection exp_id stored for auto-classification")
            return

        # Get the newest model
        newest_model = self._get_newest_model()
        if not newest_model:
            QMessageBox.warning(self, "No Model", "Could not find the newly trained model.")
            return

        # Select the new model in the dropdown
        if hasattr(self, 'model_combo'):
            # Find and select the new model
            for i in range(self.model_combo.count()):
                if newest_model in self.model_combo.itemData(i):
                    self.model_combo.setCurrentIndex(i)
                    break

        # Set flag so we know to run auto-compare after classification
        self._auto_classify_triggered = True

        # Run classification
        print(f"[SCI-Connectome] Auto-classifying with model: {newest_model}")
        self._run_classification_worker()

    def _get_newest_model(self) -> str:
        """Get the path to the most recently modified model file."""
        if not MODELS_DIR.exists():
            return None

        model_files = []
        for ext in ['*.keras', '*.h5']:
            for f in MODELS_DIR.rglob(ext):
                # Skip checkpoints
                if 'epoch' in f.stem.lower() or 'checkpoint' in f.stem.lower():
                    continue
                model_files.append((f.stat().st_mtime, str(f)))

        if not model_files:
            return None

        # Return newest
        model_files.sort(key=lambda x: x[0], reverse=True)
        return model_files[0][1]

    def refresh_models(self):
        """Refresh the list of available models.

        Shows only final trained models (not intermediate epoch checkpoints),
        sorted by modification time with newest first.
        """
        print("[DEBUG] Button clicked: refresh_models")
        import re

        models_dir = MODELS_DIR
        if not models_dir.exists():
            return

        self.model_combo.clear()
        self.model_combo.addItem("Default (BrainGlobe pretrained)")

        # Collect all model files with their modification times
        model_files = []

        # Pattern to identify intermediate checkpoint files (e.g., epoch_001.h5)
        checkpoint_pattern = re.compile(r'epoch[_-]?\d+', re.IGNORECASE)

        def is_final_model(filepath):
            """Check if this is a final model, not an intermediate checkpoint."""
            name = filepath.stem.lower()
            # Skip files that look like epoch checkpoints
            if checkpoint_pattern.search(name):
                return False
            # Skip files with "checkpoint" in the name
            if 'checkpoint' in name:
                return False
            return True

        # Look for model files in subdirectories
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Look for .keras files (full models) - preferred
                for model_file in model_dir.glob("*.keras"):
                    if is_final_model(model_file):
                        model_files.append((
                            model_file.stat().st_mtime,
                            f"{model_dir.name}/{model_file.name}",
                            str(model_file)
                        ))
                # Look for .h5 files (weights only)
                for model_file in model_dir.glob("*.h5"):
                    if is_final_model(model_file):
                        model_files.append((
                            model_file.stat().st_mtime,
                            f"{model_dir.name}/{model_file.name}",
                            str(model_file)
                        ))

        # Also look directly in models dir
        for model_file in models_dir.glob("*.keras"):
            if is_final_model(model_file):
                model_files.append((
                    model_file.stat().st_mtime,
                    model_file.stem,
                    str(model_file)
                ))
        for model_file in models_dir.glob("*.h5"):
            if is_final_model(model_file):
                model_files.append((
                    model_file.stat().st_mtime,
                    model_file.stem,
                    str(model_file)
                ))

        # Sort by modification time, newest first
        model_files.sort(key=lambda x: x[0], reverse=True)

        # Add to combo box
        for _, display_name, file_path in model_files:
            self.model_combo.addItem(display_name, file_path)

    def run_classification(self):
        """
        Run cell classification using the selected model.

        Applies a trained neural network to cell candidates to filter out
        false positives. This is the key step between detection and final counts.
        """
        print("[DEBUG] Button clicked: run_classification")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Load a brain first")
            return

        # Check for candidates
        candidates_folder = self.current_brain / "4_Cell_Candidates"
        if not candidates_folder.exists():
            QMessageBox.warning(
                self, "No Candidates Found",
                "No cell candidates found.\n\n"
                "Run detection first (Detection tab → Run Test Detection),\n"
                "or run the full detection in the Pipeline tab."
            )
            return

        # Find candidates XML
        candidates_xml = None
        for pattern in ["detected_cells.xml", "cell_classification.xml", "*.xml"]:
            files = list(candidates_folder.glob(pattern))
            if files:
                candidates_xml = files[0]
                break

        if not candidates_xml:
            QMessageBox.warning(
                self, "No Candidates Found",
                f"No cell candidates XML found in:\n{candidates_folder}"
            )
            return

        # Get selected model
        model_path = self.model_combo.currentData()
        use_default = self.model_combo.currentIndex() == 0

        if not use_default and not model_path:
            QMessageBox.warning(self, "Error", "Select a model first")
            return

        if not use_default and not Path(model_path).exists():
            QMessageBox.warning(self, "Error", f"Model not found: {model_path}")
            return

        # Get signal channel path (priority: manual_crop > auto_crop)
        manual_crop_folder = self.current_brain / "2_Cropped_For_Registration_Manual"
        crop_folder = self.current_brain / "2_Cropped_For_Registration"

        if manual_crop_folder.exists() and (manual_crop_folder / "ch0").exists():
            data_folder = manual_crop_folder
        else:
            data_folder = crop_folder

        if self.metadata:
            signal_ch = self.metadata.get('channels', {}).get('signal_channel', 0)
        else:
            signal_ch = 0
        signal_path = data_folder / f"ch{signal_ch}"

        if not signal_path.exists():
            QMessageBox.warning(self, "Error", f"Signal channel not found: {signal_path}")
            return

        # Get voxel sizes
        if self.metadata:
            voxel = self.metadata.get('voxel_size_um', {})
            voxel_sizes = (
                voxel.get('z', 4),
                voxel.get('y', 4),
                voxel.get('x', 4),
            )
        else:
            voxel_sizes = (4, 4, 4)

        # Output path
        output_path = self.current_brain / "5_Classified_Cells"
        output_path.mkdir(parents=True, exist_ok=True)

        # Get signal and background arrays from loaded layers
        signal_layer = None
        background_layer = None
        for layer in self.viewer.layers:
            if 'Signal' in layer.name and hasattr(layer, 'data'):
                signal_layer = layer
            elif 'Background' in layer.name and hasattr(layer, 'data'):
                background_layer = layer

        if signal_layer is None:
            QMessageBox.warning(self, "Error", "Signal layer not loaded. Load brain first.")
            return

        signal_array = signal_layer.data
        background_array = background_layer.data if background_layer else signal_array

        # Load cell candidates from XML as Cell objects
        try:
            from brainglobe_utils.IO.cells import get_cells
            points = get_cells(str(candidates_xml))
            print(f"[SCI-Connectome] Loaded {len(points)} candidates from {candidates_xml.name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load candidates: {e}")
            return

        # Log to tracker
        exp_id = None
        if self.tracker:
            exp_id = self.tracker.log_classification(
                brain=self.current_brain.name,
                model_path=model_path if not use_default else "default",
                candidates_path=str(candidates_xml),
                cube_size=self.classify_cube_size.value(),
                batch_size=self.classify_batch_size.value(),
                output_path=str(output_path),
                notes="Run from napari tuning widget (Python API)",
            )

        # Update UI
        self.classify_btn.setEnabled(False)
        self.classify_btn.setText("Running...")
        self.classify_status.setText(
            f"Starting classification...\n"
            f"Candidates: {len(points)} from {candidates_xml.name}\n"
            f"Model: {'Default' if use_default else Path(model_path).name}"
        )
        QApplication.processEvents()

        # Run classification in background thread using Python API
        self.classification_worker = ClassificationWorker(
            signal_array=signal_array,
            background_array=background_array,
            points=points,
            voxel_sizes=voxel_sizes,
            model_path=model_path if not use_default else None,
            output_path=output_path,
            cube_size=self.classify_cube_size.value(),
            batch_size=self.classify_batch_size.value(),
            exp_id=exp_id,
        )
        self.classification_worker.finished.connect(self._on_classification_finished)
        self.classification_worker.start()

    def _on_classification_finished(self, success, message, cells_found, rejected, exp_id):
        """Handle classification completion."""
        self.classify_btn.setEnabled(True)
        self.classify_btn.setText("Run Classification")

        # Check if this was triggered by auto-classify after training
        auto_triggered = getattr(self, '_auto_classify_triggered', False)
        if auto_triggered:
            self._auto_classify_triggered = False  # Reset flag

        if success:
            # Update tracker
            if self.tracker and exp_id:
                self.tracker.update_status(
                    exp_id,
                    status="completed",
                    class_cells_found=cells_found,
                    class_rejected=rejected,
                )

            # Log to session documenter
            model_name = self.model_combo.currentText()
            if self.session_doc and self.session_doc.is_active():
                self.session_doc.documenter.log_classification_run(
                    run_id=exp_id or "unknown",
                    model=model_name,
                    cells_found=cells_found,
                    rejected=rejected,
                    cube_size=self.classify_cube_size.value(),
                    batch_size=self.classify_batch_size.value(),
                )

            # Generate QC image with actual slice visuals
            self._generate_classification_qc_image(
                cells_found=cells_found,
                rejected=rejected,
                model_name=model_name,
                exp_id=exp_id,
            )

            self.classify_status.setText(
                f"Classification complete!\n"
                f"Cells: {cells_found}\n"
                f"Rejected: {rejected}\n"
                f"Acceptance rate: {cells_found/(cells_found+rejected)*100:.1f}%"
                if (cells_found + rejected) > 0 else f"Classification complete!\nNo candidates processed."
            )

            if auto_triggered:
                # Auto-triggered: load results automatically and run comparison
                self._load_classified_cells()
                self._auto_compare_classification(exp_id, cells_found, rejected, model_name)
            else:
                # Manual: offer to load results
                reply = QMessageBox.question(
                    self, "Classification Complete",
                    f"Classification complete!\n\n"
                    f"Cells found: {cells_found}\n"
                    f"Rejected: {rejected}\n\n"
                    f"Load classified cells into napari?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self._load_classified_cells()
        else:
            if self.tracker and exp_id:
                self.tracker.update_status(exp_id, status="failed")

            self.classify_status.setText(f"Classification failed:\n{message}")
            QMessageBox.warning(self, "Classification Failed", message)

    def _auto_compare_classification(self, new_exp_id: str, cells_found: int, rejected: int, model_name: str):
        """
        Automatically compare new classification against previous best/last for this brain.

        Called after auto-classify completes. Loads previous classification, runs spatial
        comparison, displays summary, and updates tracker with diff metrics.
        """
        import numpy as np

        if not self.tracker or not self.current_brain:
            print("[SCI-Connectome] Cannot compare: no tracker or brain loaded")
            return

        brain_name = self.current_brain.name if hasattr(self.current_brain, 'name') else str(self.current_brain)

        # Find previous classification for this brain (excluding the one we just ran)
        prev_classifications = self.tracker.search(
            brain=brain_name,
            exp_type="classification",
            status="completed",
            limit=10,
            sort_by="created_at",
            descending=True
        )

        # Filter out the current run and find the best previous one
        prev_run = None
        for run in prev_classifications:
            if run.get('exp_id') != new_exp_id:
                # Prefer marked_best, otherwise take most recent
                if run.get('marked_best'):
                    prev_run = run
                    break
                elif prev_run is None:
                    prev_run = run

        if not prev_run:
            print(f"[SCI-Connectome] No previous classification found for comparison")
            print(f"[SCI-Connectome] Training & Classification Complete!")
            print(f"  Model: {model_name}")
            print(f"  Cells: {cells_found} | Rejected: {rejected}")
            return

        prev_exp_id = prev_run.get('exp_id')
        prev_cells = prev_run.get('class_cells_found', 0)
        prev_rejected = prev_run.get('class_rejected', 0)

        print(f"\n[SCI-Connectome] === AUTO-COMPARE RESULTS ===")
        print(f"  Model: {model_name}")
        print(f"  New classification: {cells_found} cells ({rejected} rejected)")
        print(f"  Previous ({prev_exp_id[:20]}...): {prev_cells} cells ({prev_rejected} rejected)")

        # Calculate simple count difference
        diff = cells_found - prev_cells
        if prev_cells > 0:
            pct_change = (diff / prev_cells) * 100
            print(f"  Difference: {diff:+d} cells ({pct_change:+.1f}%)")
        else:
            print(f"  Difference: {diff:+d} cells")

        # Load previous classification as 'prev_cells' for visual comparison
        try:
            self._load_classification_run(prev_exp_id, as_previous=True)

            # Mark currently loaded cells as 'new' for comparison
            # The cells just loaded by _load_classified_cells should be marked
            for layer in self.viewer.layers:
                if hasattr(layer, 'metadata'):
                    cell_type = layer.metadata.get('cell_type')
                    if cell_type == 'class_cells':
                        # Mark as new classification
                        layer.metadata['cell_type'] = 'class_new_cells'
                        layer.name = f"New: {cells_found} cells"

            # Run visual difference generation
            print(f"[SCI-Connectome] Generating visual difference layers...")
            self.generate_classification_difference()

        except Exception as e:
            print(f"[SCI-Connectome] Could not load previous for visual comparison: {e}")

        # Update tracker with comparison metrics
        try:
            # Get spatial diff counts from the generated layers
            gained_layer = self._get_layer_by_type('class_diff_gained')
            lost_layer = self._get_layer_by_type('class_diff_lost')

            diff_gained = len(gained_layer.data) if gained_layer else 0
            diff_lost = len(lost_layer.data) if lost_layer else 0

            self.tracker.update_status(
                new_exp_id,
                diff_vs_best=diff,
                diff_gained=diff_gained,
                diff_lost=diff_lost,
                comparison_ref=prev_exp_id
            )

            print(f"  Spatial diff: +{diff_gained} gained, -{diff_lost} lost")

        except Exception as e:
            print(f"[SCI-Connectome] Could not update tracker with diff: {e}")

        print(f"[SCI-Connectome] === COMPARISON COMPLETE ===\n")

    def _load_classified_cells(self):
        """
        Load classified cells into napari with high-contrast, semi-transparent markers.

        Uses cyan for cells and magenta for rejected - these are complementary colors
        that are easily distinguished on grey background and from each other.
        Colors are defined in CELL_COLORS for consistency with legend.

        Always creates both layers (cells and rejected) even if empty, so that
        comparison features work correctly.
        """
        print("[DEBUG] Button clicked: _load_classified_cells")
        import numpy as np

        if not self.current_brain:
            return

        cells_path = self.current_brain / "5_Classified_Cells" / "cells.xml"
        rejected_path = self.current_brain / "5_Classified_Cells" / "rejected.xml"

        # Load cells (confirmed) - cyan discs
        if cells_path.exists():
            self._load_points_from_xml(
                cells_path,
                "Classified Cells",
                cell_type='cells'
            )
        else:
            # Create empty cells layer so UI is consistent
            self._add_points_layer(
                np.empty((0, 3)),
                "Classified Cells (0)",
                cell_type='cells'
            )
            print("[SCI-Connectome] No cells.xml found - created empty Classified Cells layer")

        # Load rejected (non-cells) - magenta crosses, hidden by default
        if rejected_path.exists():
            self._load_points_from_xml(
                rejected_path,
                "Rejected",
                cell_type='rejected'
            )
        else:
            # Create empty rejected layer so comparison features work
            self._add_points_layer(
                np.empty((0, 3)),
                "Rejected (0)",
                cell_type='rejected'
            )
            print("[SCI-Connectome] No rejected.xml found - created empty Rejected layer")

        # Update status label if it exists - count from layers
        if hasattr(self, 'classification_load_status'):
            cells_count = 0
            rejected_count = 0
            for layer in self.viewer.layers:
                if 'Classified Cells' in layer.name:
                    cells_count = len(layer.data)
                elif 'Rejected' in layer.name:
                    rejected_count = len(layer.data)
            self.classification_load_status.setText(
                f"Loaded: {cells_count} cells, {rejected_count} rejected"
            )

    # =========================================================================
    # CONSISTENT SLICE SAMPLING FOR QC
    # =========================================================================

    # Fixed percentile positions for QC slices - these never change between brains
    # This ensures reproducible sampling across all datasets
    QC_SLICE_PERCENTILES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    def _get_qc_slice_indices(self, num_slices: int) -> list:
        """
        Get deterministic QC slice indices that are consistent across all brains.

        Uses fixed percentile positions rather than random sampling, so the
        'same' slices (relative position) are sampled for every brain.

        Args:
            num_slices: Total number of slices in the stack

        Returns:
            List of 10 slice indices at fixed percentile positions
        """
        indices = []
        for pct in self.QC_SLICE_PERCENTILES:
            idx = int(num_slices * pct / 100)
            # Clamp to valid range
            idx = max(0, min(num_slices - 1, idx))
            indices.append(idx)
        return indices

    def _generate_classification_qc_image(self, cells_found: int, rejected: int, model_name: str, exp_id: str = None):
        """
        Generate a QC image after classification showing actual slice images.

        Creates a multi-panel PNG showing 10 representative slices with:
        - The actual image data for each slice
        - Green markers for classified cells
        - Red markers for rejected candidates
        - Summary info: network used, counts, acceptance rate

        The same 10 slice positions (by percentile) are used for every brain
        to enable consistent cross-brain comparison.

        Args:
            cells_found: Number of cells that passed classification
            rejected: Number of candidates rejected
            model_name: Name of the classification model used
            exp_id: Experiment ID for tracking

        Saves to: DATA_SUMMARY_DIR/optimization_runs/{brain}/QC_classify_{exp_id}.png
        """
        if not self.current_brain:
            return

        try:
            from PIL import Image, ImageDraw, ImageFont
            import tifffile
            from natsort import natsorted
        except ImportError as e:
            print(f"[QC] Required library not available: {e}")
            return

        try:
            brain_name = self.current_brain.name

            # Load cell coordinates
            cells_path = self.current_brain / "5_Classified_Cells" / "cells.xml"
            rejected_path = self.current_brain / "5_Classified_Cells" / "rejected.xml"

            cell_coords = self._parse_cellfinder_xml(cells_path) if cells_path.exists() else []
            rejected_coords = self._parse_cellfinder_xml(rejected_path) if rejected_path.exists() else []

            # Build z-index lookup for fast slice filtering
            # Coords are in ZYX order
            cells_by_z = {}
            for coord in cell_coords:
                z = int(coord[0])
                if z not in cells_by_z:
                    cells_by_z[z] = []
                cells_by_z[z].append((int(coord[2]), int(coord[1])))  # XY for drawing

            rejected_by_z = {}
            for coord in rejected_coords:
                z = int(coord[0])
                if z not in rejected_by_z:
                    rejected_by_z[z] = []
                rejected_by_z[z].append((int(coord[2]), int(coord[1])))  # XY for drawing

            # Find signal and background channel images (priority: manual_crop > auto_crop)
            manual_crop_folder = self.current_brain / "2_Cropped_For_Registration_Manual"
            crop_folder = self.current_brain / "2_Cropped_For_Registration"

            if manual_crop_folder.exists() and (manual_crop_folder / "ch0").exists():
                data_folder = manual_crop_folder
            else:
                data_folder = crop_folder

            if self.metadata:
                signal_ch = self.metadata.get('channels', {}).get('signal_channel', 0)
                bg_ch = self.metadata.get('channels', {}).get('background_channel', 1)
            else:
                signal_ch = 0
                bg_ch = 1
            signal_path = data_folder / f"ch{signal_ch}"
            bg_path = data_folder / f"ch{bg_ch}"

            if not signal_path.exists():
                print(f"[QC] Signal path not found: {signal_path}")
                return

            signal_files = natsorted(signal_path.glob("*.tif*"))
            bg_files = natsorted(bg_path.glob("*.tif*")) if bg_path.exists() else []
            if not signal_files:
                print(f"[QC] No TIFF files found in {signal_path}")
                return

            has_background = len(bg_files) == len(signal_files)
            num_slices = len(signal_files)
            qc_indices = self._get_qc_slice_indices(num_slices)

            # Create output directory
            qc_dir = DATA_SUMMARY_DIR / "optimization_runs" / brain_name / "qc_images"
            qc_dir.mkdir(parents=True, exist_ok=True)

            # Setup fonts
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                font_bold = ImageFont.truetype("arialbd.ttf", 14)
                font_title = ImageFont.truetype("arialbd.ttf", 18)
            except:
                font = ImageFont.load_default()
                font_bold = font
                font_title = font

            # Generate individual slice images
            slice_images = []
            slice_width = 400  # Target width for each slice panel
            marker_radius = 8  # Slightly larger for visibility

            for z_idx in qc_indices:
                if z_idx >= len(signal_files):
                    continue

                # Load signal slice
                signal_data = tifffile.imread(str(signal_files[z_idx])).astype(np.float32)

                # Normalize signal to percentile range (matches napari auto-balance)
                s_low, s_high = np.percentile(signal_data, [1, 99.5])
                signal_norm = np.clip((signal_data - s_low) / (s_high - s_low + 1e-6), 0, 1)

                if has_background:
                    # Load background slice
                    bg_data = tifffile.imread(str(bg_files[z_idx])).astype(np.float32)
                    b_low, b_high = np.percentile(bg_data, [1, 99.5])
                    bg_norm = np.clip((bg_data - b_low) / (b_high - b_low + 1e-6), 0, 1)

                    # Additive blending of green (signal) + magenta (background)
                    # Same approach as napari: when balanced, green + magenta = grey
                    # Where signal > background → greenish (true cells)
                    # Where background > signal → magentaish
                    #
                    # RGB channels:
                    #   Green = (0, signal, 0)
                    #   Magenta = (bg, 0, bg)
                    #   Combined = (bg, signal, bg)
                    #
                    # When signal == background: (x, x, x) = grey
                    # When signal > background: more green
                    r_channel = (bg_norm * 255).astype(np.uint8)
                    g_channel = (signal_norm * 255).astype(np.uint8)
                    b_channel = (bg_norm * 255).astype(np.uint8)
                    slice_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
                else:
                    # No background - just use signal as green on black
                    slice_8bit = (signal_norm * 255).astype(np.uint8)
                    slice_rgb = np.stack([np.zeros_like(slice_8bit), slice_8bit, np.zeros_like(slice_8bit)], axis=-1)

                # Convert to RGBA for transparency support
                img = Image.fromarray(slice_rgb).convert('RGBA')

                # Scale to target width
                aspect = img.height / img.width
                new_height = int(slice_width * aspect)
                img = img.resize((slice_width, new_height), Image.Resampling.LANCZOS)
                scale_x = slice_width / signal_data.shape[1]
                scale_y = new_height / signal_data.shape[0]

                # Create transparent overlay for markers
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # High-contrast colors with semi-transparency
                # Cyan for cells (visible on grey), Magenta for rejected (distinct from cyan)
                cell_color = (0, 255, 255, 180)       # Cyan, semi-transparent
                rejected_color = (255, 0, 128, 150)  # Magenta-red, semi-transparent

                # Draw rejected markers (behind cells)
                if z_idx in rejected_by_z:
                    for x, y in rejected_by_z[z_idx]:
                        sx, sy = int(x * scale_x), int(y * scale_y)
                        # X marker for rejected
                        r = marker_radius
                        draw.line([(sx - r, sy - r), (sx + r, sy + r)], fill=rejected_color, width=3)
                        draw.line([(sx - r, sy + r), (sx + r, sy - r)], fill=rejected_color, width=3)

                # Draw cell markers (on top) - filled circles with outline
                if z_idx in cells_by_z:
                    for x, y in cells_by_z[z_idx]:
                        sx, sy = int(x * scale_x), int(y * scale_y)
                        r = marker_radius
                        # Filled circle with border
                        draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)],
                                     fill=(0, 255, 255, 100), outline=cell_color, width=2)

                # Composite overlay onto image
                img = Image.alpha_composite(img, overlay)

                # Convert back to RGB for label drawing
                img = img.convert('RGB')
                draw = ImageDraw.Draw(img)

                # Add slice label with semi-transparent background
                z_cells = len(cells_by_z.get(z_idx, []))
                z_rejected = len(rejected_by_z.get(z_idx, []))
                label = f"Z={z_idx} | Cells: {z_cells} | Rejected: {z_rejected}"
                draw.rectangle([(0, 0), (slice_width, 20)], fill=(0, 0, 0, 180))
                draw.text((5, 3), label, fill=(255, 255, 255), font=font)

                slice_images.append(img)

            if not slice_images:
                print("[QC] No slice images generated")
                return

            # Create info panel
            info_width = 350
            info_height = max(img.height for img in slice_images) if slice_images else 400
            info_img = Image.new('RGB', (info_width, info_height), color=(40, 40, 40))
            draw = ImageDraw.Draw(info_img)

            y_pos = 20
            line_height = 22

            # Title
            draw.text((20, y_pos), "Classification QC Report", fill=(255, 255, 255), font=font_title)
            y_pos += 35

            # Timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            draw.text((20, y_pos), f"Generated: {timestamp}", fill=(180, 180, 180), font=font)
            y_pos += line_height * 2

            # Data hierarchy
            draw.text((20, y_pos), "DATA HIERARCHY", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5
            try:
                from braintools.config import parse_brain_name
                hierarchy = parse_brain_name(brain_name)
                if hierarchy.get("brain_id"):
                    draw.text((20, y_pos), f"Brain ID: {hierarchy.get('brain_id', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Subject: {hierarchy.get('subject_full', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Cohort: {hierarchy.get('cohort_full', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
                    draw.text((20, y_pos), f"Project: {hierarchy.get('project_name', '')}", fill=(255, 255, 255), font=font)
                    y_pos += line_height
            except Exception:
                draw.text((20, y_pos), f"Brain: {brain_name}", fill=(255, 255, 255), font=font)
                y_pos += line_height
            y_pos += line_height

            # Model info
            draw.text((20, y_pos), "CLASSIFICATION MODEL", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5
            draw.text((20, y_pos), f"Network: {model_name}", fill=(255, 255, 255), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Cube size: {self.classify_cube_size.value()}", fill=(200, 200, 200), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"Batch size: {self.classify_batch_size.value()}", fill=(200, 200, 200), font=font)
            y_pos += line_height * 2

            # Results
            draw.text((20, y_pos), "RESULTS", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5
            draw.text((20, y_pos), f"Cells classified: {cells_found}", fill=(100, 255, 100), font=font_bold)
            y_pos += line_height
            draw.text((20, y_pos), f"Rejected: {rejected}", fill=(255, 100, 100), font=font_bold)
            y_pos += line_height
            total = cells_found + rejected
            if total > 0:
                acceptance = cells_found / total * 100
                draw.text((20, y_pos), f"Acceptance rate: {acceptance:.1f}%", fill=(255, 255, 255), font=font)
                y_pos += line_height
            y_pos += line_height

            # Legend (matches marker colors: cyan cells, magenta rejected)
            draw.text((20, y_pos), "LEGEND", fill=(100, 200, 255), font=font_bold)
            y_pos += line_height + 5
            draw.ellipse([(20, y_pos), (32, y_pos + 12)], outline=(0, 255, 255), width=2)
            draw.text((40, y_pos), "Classified cell (cyan)", fill=(0, 255, 255), font=font)
            y_pos += line_height
            draw.line([(20, y_pos), (32, y_pos + 12)], fill=(255, 0, 128), width=2)
            draw.line([(20, y_pos + 12), (32, y_pos)], fill=(255, 0, 128), width=2)
            draw.text((40, y_pos), "Rejected candidate (magenta)", fill=(255, 0, 128), font=font)
            y_pos += line_height * 2

            # Slice info
            draw.text((20, y_pos), f"QC slices: {len(qc_indices)} @ percentiles", fill=(150, 150, 150), font=font)
            y_pos += line_height
            draw.text((20, y_pos), f"{self.QC_SLICE_PERCENTILES}", fill=(150, 150, 150), font=font)

            # Combine: 2 rows of 5 slices + info panel
            # Arrange slices in 2 rows
            n_cols = 5
            n_rows = 2
            slice_height = slice_images[0].height if slice_images else 300

            # Calculate combined image size
            slices_width = n_cols * slice_width
            slices_height = n_rows * slice_height
            combined_width = slices_width + info_width
            combined_height = slices_height

            combined = Image.new('RGB', (combined_width, combined_height), color=(30, 30, 30))

            # Paste slices
            for i, img in enumerate(slice_images[:10]):
                row = i // n_cols
                col = i % n_cols
                x = col * slice_width
                y = row * slice_height
                combined.paste(img, (x, y))

            # Paste info panel on right
            combined.paste(info_img, (slices_width, 0))

            # Save
            run_id = exp_id or datetime.now().strftime('%Y%m%d_%H%M%S')
            qc_filename = f"QC_classify_{run_id}.png"
            qc_path = qc_dir / qc_filename
            combined.save(str(qc_path), quality=95)

            print(f"[SCI-Connectome] Classification QC image saved: {qc_path}")

            # Log to session documenter
            if self.session_doc:
                self.session_doc.log_export(str(qc_path), export_type="classification_qc_image")

            # Update tracker with QC path
            if self.tracker and exp_id:
                self.tracker.update_status(exp_id, qc_image_path=str(qc_path))

        except Exception as e:
            print(f"[SCI-Connectome] Classification QC image generation failed: {e}")
            import traceback
            traceback.print_exc()

    def refresh_brains(self):
        """Refresh the brain list."""
        print("[DEBUG] Button clicked: refresh_brains")
        self.brain_combo.clear()
        self.brain_combo.addItem("Select a brain...")

        if not BRAINS_ROOT.exists():
            return

        for mouse_dir in sorted(BRAINS_ROOT.iterdir()):
            if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
                continue
            if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
                continue

            for pipeline_dir in sorted(mouse_dir.iterdir()):
                if not pipeline_dir.is_dir():
                    continue

                has_structure = any([
                    (pipeline_dir / "0_Raw_IMS").exists(),
                    (pipeline_dir / "1_Extracted_Full").exists(),
                ])

                if has_structure:
                    self.brain_combo.addItem(pipeline_dir.name, str(pipeline_dir))

    def _on_mode_changed(self):
        """Handle workflow mode change."""
        if self.mode_cell_detection.isChecked():
            self.view_mode = "cell_detection"
            print(f"[SCI-Connectome] Switched to Cell Detection & Tuning mode")
        elif self.mode_registration.isChecked():
            self.view_mode = "registration_qc"
            print(f"[SCI-Connectome] Switched to Registration QC & Approval mode")

    def _find_manual_crop(self, brain_folder):
        """Check if manual crop exists, return path if found."""
        manual_crop = Path(brain_folder) / "2_Cropped_For_Registration_Manual"
        if manual_crop.exists():
            print(f"  Found manually cropped brain at: {manual_crop}")
            return manual_crop
        return None

    def on_brain_changed(self, text):
        """Handle brain selection."""
        path = self.brain_combo.currentData()
        if path:
            self.current_brain = Path(path)
            self.load_metadata()
            self.parse_voxel_from_filename()
            self.brain_status_label.setText(f"Selected: {self.current_brain.name}")
            self.brain_status_label.setStyleSheet("color: #2196F3;")

            # Start session documentation for this brain
            if self.session_doc:
                self.session_doc.start_for_brain(self.current_brain.name)
                # Record ALL initial parameters
                initial_params = {
                    # Core ball filter
                    'ball_xy_size': self.ball_xy.value(),
                    'ball_z_size': self.ball_z.value(),
                    'ball_overlap_fraction': self.ball_overlap_fraction.value(),
                    # Soma
                    'soma_diameter': self.soma_diameter.value(),
                    'soma_spread_factor': self.soma_spread_factor.value(),
                    # Thresholding
                    'threshold': self.threshold.value(),
                    'tiled_threshold': self.tiled_threshold.value(),
                    'log_sigma_size': self.log_sigma_size.value(),
                    # Cluster handling
                    'max_cluster_size': self.max_cluster_size.value(),
                    # Performance
                    'use_gpu': self.use_gpu.isChecked(),
                    'batch_size': self.batch_size.value(),
                    'n_free_cpus': self.n_free_cpus.value(),
                }
                self.session_doc.documenter.set_initial_params(initial_params)

            # Refresh the calibration runs table for this brain
            self._refresh_context_runs()

            # Update registration approval status
            self._update_registration_status()

            # Update paradigm status and offer to apply settings
            self._update_paradigm_status()
            self._offer_paradigm_settings()

    def _update_registration_status(self):
        """Update the registration approval status label for current brain."""
        if not self.current_brain:
            return

        approval_file = self.current_brain / "3_Registered_Atlas" / ".registration_approved"
        if approval_file.exists():
            self.approval_status.setText("Status: APPROVED")
            self.approval_status.setStyleSheet("color: green;")
        else:
            # Check if registration exists but is not approved
            brainreg_json = self.current_brain / "3_Registered_Atlas" / "brainreg.json"
            if brainreg_json.exists():
                self.approval_status.setText("Status: PENDING APPROVAL")
                self.approval_status.setStyleSheet("color: orange;")
            else:
                self.approval_status.setText("Status: NOT REGISTERED")
                self.approval_status.setStyleSheet("color: gray;")

    def load_brain_into_napari(self):
        """Load the selected brain's channels into napari - mode-aware loading."""
        print("[DEBUG] Button clicked: load_brain_into_napari")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        # Check view mode
        if self.view_mode == "registration_qc":
            # Registration QC mode - load registration data
            self._load_registration_qc_view()
            return

        # Cell detection mode - load full-res brain
        # Don't start another load if one is in progress
        if self.brain_loader_worker and self.brain_loader_worker.isRunning():
            QMessageBox.warning(self, "Loading", "Brain is already loading...")
            return

        # Try folders in priority order: manual_crop > auto_crop > extracted
        manual_crop_folder = self.current_brain / "2_Cropped_For_Registration_Manual"
        crop_folder = self.current_brain / "2_Cropped_For_Registration"
        extract_folder = self.current_brain / "1_Extracted_Full"

        if manual_crop_folder.exists() and (manual_crop_folder / "ch0").exists():
            data_folder = manual_crop_folder
            source = "manual_crop"
            print(f"  Using MANUAL crop folder (takes priority)")
        elif crop_folder.exists() and (crop_folder / "ch0").exists():
            data_folder = crop_folder
            source = "cropped"
        elif extract_folder.exists() and (extract_folder / "ch0").exists():
            data_folder = extract_folder
            source = "extracted"
        else:
            QMessageBox.warning(
                self, "Error",
                f"No image data found for {self.current_brain.name}\n"
                "Expected ch0/ folder in 2_Cropped_For_Registration or 1_Extracted_Full"
            )
            return

        # Check metadata for which channel is signal
        # Default: ch0=signal, ch1=background
        # If channels_swapped or signal_channel=1, flip them
        metadata_path = data_folder / "metadata.json"
        channels_swapped = False
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                # Check either channels_swapped flag or signal_channel field
                if meta.get('channels_swapped', False):
                    channels_swapped = True
                elif meta.get('channels', {}).get('signal_channel') == 1:
                    channels_swapped = True
                elif meta.get('signal_channel') == 1:
                    channels_swapped = True
            except Exception as e:
                print(f"  Warning: Could not read metadata: {e}")

        if channels_swapped:
            signal_path = data_folder / "ch1"
            background_path = data_folder / "ch0"
            print(f"  Channels swapped per metadata: signal=ch1, background=ch0")
        else:
            signal_path = data_folder / "ch0"
            background_path = data_folder / "ch1"

        # Store source for use in callback
        self._loading_source = source

        # Update UI
        self.brain_status_label.setText(f"Loading {self.current_brain.name}...")
        self.brain_status_label.setStyleSheet("color: orange;")
        self.load_brain_btn.setEnabled(False)
        self.load_brain_btn.setText("Loading...")

        # Print to terminal
        print(f"\n[SCI-Connectome] Loading brain: {self.current_brain.name}")
        print(f"  Source: {source}")
        print(f"  Signal: {signal_path}")

        # Start background loader
        self.brain_loader_worker = BrainLoaderWorker(signal_path, background_path)
        self.brain_loader_worker.progress.connect(self._on_brain_load_progress)
        self.brain_loader_worker.finished.connect(self._on_brain_load_finished)
        self.brain_loader_worker.start()

    def convert_to_zarr(self):
        """Convert current brain's TIFFs to Zarr for instant future loading."""
        print("[DEBUG] Button clicked: convert_to_zarr")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        # Find data folder (priority: manual_crop > auto_crop > extracted)
        manual_crop_folder = self.current_brain / "2_Cropped_For_Registration_Manual"
        crop_folder = self.current_brain / "2_Cropped_For_Registration"
        extract_folder = self.current_brain / "1_Extracted_Full"

        if manual_crop_folder.exists() and (manual_crop_folder / "ch0").exists():
            data_folder = manual_crop_folder
        elif crop_folder.exists() and (crop_folder / "ch0").exists():
            data_folder = crop_folder
        elif extract_folder.exists() and (extract_folder / "ch0").exists():
            data_folder = extract_folder
        else:
            QMessageBox.warning(self, "Error", "No image data found")
            return

        # Check if already converted
        zarr_ch0 = data_folder / "ch0.zarr"
        zarr_ch1 = data_folder / "ch1.zarr"
        if zarr_ch0.exists() and zarr_ch1.exists():
            QMessageBox.information(self, "Already Converted",
                "This brain is already converted to Zarr!\n"
                "Future loads will be instant.")
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Convert to Zarr",
            f"Convert {self.current_brain.name} to Zarr format?\n\n"
            "This takes about the same time as one load, but after that\n"
            "all future loads will be nearly instant (seconds instead of minutes).\n\n"
            "The original TIFF files will NOT be deleted.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Disable buttons
        self.convert_zarr_btn.setEnabled(False)
        self.convert_zarr_btn.setText("Converting...")
        self.load_brain_btn.setEnabled(False)

        # Convert ch0
        self.brain_status_label.setText("Converting ch0 to Zarr...")
        self.brain_status_label.setStyleSheet("color: purple;")

        self._zarr_data_folder = data_folder
        self._zarr_current_channel = 0

        self.zarr_worker = ZarrConverterWorker(data_folder / "ch0", zarr_ch0)
        self.zarr_worker.progress.connect(self._on_zarr_progress)
        self.zarr_worker.finished.connect(self._on_zarr_finished)
        self.zarr_worker.start()

    def _on_zarr_progress(self, message, percent):
        """Handle Zarr conversion progress."""
        self.brain_status_label.setText(f"ch{self._zarr_current_channel}: {message}")
        self.convert_zarr_btn.setText(f"Converting... {percent}%")

    def _on_zarr_finished(self, success, message):
        """Handle Zarr conversion completion."""
        if not success:
            self.convert_zarr_btn.setEnabled(True)
            self.convert_zarr_btn.setText("Convert to Zarr")
            self.load_brain_btn.setEnabled(True)
            self.brain_status_label.setText(f"Conversion failed: {message}")
            self.brain_status_label.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Conversion Failed", message)
            return

        # If we just did ch0, now do ch1
        if self._zarr_current_channel == 0:
            ch1_folder = self._zarr_data_folder / "ch1"
            if ch1_folder.exists():
                self._zarr_current_channel = 1
                zarr_ch1 = self._zarr_data_folder / "ch1.zarr"
                self.zarr_worker = ZarrConverterWorker(ch1_folder, zarr_ch1)
                self.zarr_worker.progress.connect(self._on_zarr_progress)
                self.zarr_worker.finished.connect(self._on_zarr_finished)
                self.zarr_worker.start()
                return

        # All done!
        self.convert_zarr_btn.setEnabled(True)
        self.convert_zarr_btn.setText("Convert to Zarr")
        self.load_brain_btn.setEnabled(True)
        self.brain_status_label.setText("Conversion complete! Future loads will be instant.")
        self.brain_status_label.setStyleSheet("color: green; font-weight: bold;")

        # Only show message box for manual conversions, not auto
        if not getattr(self, '_auto_zarr_conversion', False):
            QMessageBox.information(self, "Conversion Complete",
                "Brain converted to Zarr format!\n\n"
                "Future loads will be nearly instant (seconds instead of minutes).")
        else:
            self._auto_zarr_conversion = False
            print(f"[SCI-Connectome] Auto-zarr conversion complete!")

    def _auto_start_zarr_conversion(self):
        """Auto-start zarr conversion in background after brain loads (if not already done)."""
        if not self.current_brain:
            return

        # Find data folder (priority: manual_crop > auto_crop > extracted)
        manual_crop_folder = self.current_brain / "2_Cropped_For_Registration_Manual"
        crop_folder = self.current_brain / "2_Cropped_For_Registration"
        extract_folder = self.current_brain / "1_Extracted_Full"

        if manual_crop_folder.exists() and (manual_crop_folder / "ch0").exists():
            data_folder = manual_crop_folder
        elif crop_folder.exists() and (crop_folder / "ch0").exists():
            data_folder = crop_folder
        elif extract_folder.exists() and (extract_folder / "ch0").exists():
            data_folder = extract_folder
        else:
            return  # No data folder found

        # Check if zarr already exists
        zarr_ch0 = data_folder / "ch0.zarr"
        zarr_ch1 = data_folder / "ch1.zarr"
        if zarr_ch0.exists() and zarr_ch1.exists():
            print(f"[SCI-Connectome] Zarr already exists, skipping auto-conversion")
            return

        # Start conversion in background (no confirmation needed for auto)
        print(f"[SCI-Connectome] Auto-starting Zarr conversion in background...")
        print(f"  Data folder: {data_folder}")

        # Mark that this is an auto-conversion (suppress completion dialog)
        self._auto_zarr_conversion = True

        # Disable manual convert button while auto is running
        if hasattr(self, 'convert_zarr_btn'):
            self.convert_zarr_btn.setEnabled(False)
            self.convert_zarr_btn.setText("Auto-converting...")

        # Store data folder for conversion
        self._zarr_data_folder = data_folder
        self._zarr_current_channel = 0

        # Start with ch0
        self.zarr_worker = ZarrConverterWorker(data_folder / "ch0", zarr_ch0)
        self.zarr_worker.progress.connect(self._on_zarr_progress)
        self.zarr_worker.finished.connect(self._on_zarr_finished)
        self.zarr_worker.start()

    def _swap_loaded_layers(self):
        """
        Swap the Signal and Background layers in napari.

        Swaps data, names, colors, and contrast limits so that after swap,
        the layers look exactly like they would if loaded correctly:
        - Signal layer = green, visible, on top
        - Background layer = magenta, hidden, on bottom
        """
        print("[DEBUG] Button clicked: _swap_loaded_layers")
        # Find signal and background layers
        signal_layer = None
        bg_layer = None

        for layer in self.viewer.layers:
            if 'Signal' in layer.name:
                signal_layer = layer
            elif 'Background' in layer.name:
                bg_layer = layer

        if not signal_layer or not bg_layer:
            QMessageBox.warning(
                self, "No Layers Found",
                "Could not find Signal and Background layers to swap.\n\n"
                "Load a brain first using 'Load Brain into Napari'."
            )
            return

        # Swap the data
        signal_data = signal_layer.data
        bg_data = bg_layer.data

        signal_layer.data = bg_data
        bg_layer.data = signal_data

        # Swap contrast limits
        signal_contrast = signal_layer.contrast_limits
        bg_contrast = bg_layer.contrast_limits

        signal_layer.contrast_limits = bg_contrast
        bg_layer.contrast_limits = signal_contrast

        # Swap names (keeping the brain name prefix)
        signal_name = signal_layer.name
        bg_name = bg_layer.name

        # Extract brain name from layer name
        if ' - Signal' in signal_name:
            brain_prefix = signal_name.split(' - Signal')[0]
            source_suffix = signal_name.split(' - Signal')[-1]  # e.g., " (cropped)"
        else:
            brain_prefix = self.current_brain.name if self.current_brain else "Brain"
            source_suffix = ""

        signal_layer.name = f"{brain_prefix} - Signal{source_suffix}"
        bg_layer.name = f"{brain_prefix} - Background{source_suffix}"

        # Swap colormaps
        signal_layer.colormap = 'green'
        bg_layer.colormap = 'magenta'

        # Signal visible on top, background hidden on bottom
        signal_layer.visible = True
        bg_layer.visible = False

        # Fix layer order: move background to bottom, signal on top
        # In napari, higher index = on top visually
        signal_idx = self.viewer.layers.index(signal_layer)
        bg_idx = self.viewer.layers.index(bg_layer)

        if bg_idx > signal_idx:
            # Background is above signal, need to swap order
            self.viewer.layers.move(bg_idx, signal_idx)

        # Toggle the checkbox to reflect the new state
        self.swap_channels.setChecked(not self.swap_channels.isChecked())

        # Update status
        swap_state = "swapped" if self.swap_channels.isChecked() else "normal"
        print(f"[SCI-Connectome] Layers swapped! Channels now: {swap_state}")
        self.brain_status_label.setText(f"Channels {swap_state}")
        self.brain_status_label.setStyleSheet("color: green; font-weight: bold;")

    def _on_brain_load_progress(self, message):
        """Handle brain loading progress updates."""
        self.brain_status_label.setText(message)
        print(f"  {message}")

    def _on_brain_load_finished(self, success, message, signal_stack, bg_stack):
        """Handle brain loading completion."""
        self.load_brain_btn.setEnabled(True)
        self.load_brain_btn.setText("Load Brain into Napari")

        if success:
            source = self._loading_source or "unknown"

            # Layer setup:
            # - Signal channel: green colormap
            # - Background channel: magenta colormap
            # - Both use additive blending and gamma=1.2
            # - Contrast limits: napari auto-calculates (user preference)

            # Layer variables for later reference
            bg_layer = None
            signal_layer = None

            # Add background layer FIRST (so it's on bottom in layer list)
            if bg_stack is not None:
                bg_layer = self.viewer.add_image(
                    bg_stack,
                    name=f"{self.current_brain.name} - Background ({source})",
                    colormap='magenta',
                    blending='additive',
                    gamma=1.2,  # User-preferred setting for better visibility
                    visible=True,  # Visible by default for balanced view
                )
                # Let napari auto-calculate contrast limits (user preference)
                print(f"  Added background layer: {bg_stack.shape}")

            # Add signal layer SECOND (so it's on top in layer list)
            if signal_stack is not None:
                signal_layer = self.viewer.add_image(
                    signal_stack,
                    name=f"{self.current_brain.name} - Signal ({source})",
                    colormap='green',
                    blending='additive',
                    gamma=1.2,  # User-preferred setting for better visibility
                )
                print(f"  Added signal layer: {signal_stack.shape}")

            # Set user-preferred contrast limits
            if signal_layer is not None:
                signal_layer.contrast_limits = (0.0, 1872.4285714285713)
            if bg_layer is not None:
                bg_layer.contrast_limits = (0.0, 936.2142857142857)
            print(f"  Applied user contrast limits: signal [0, 1872], background [0, 936]")

            # Reset view
            self.viewer.reset_view()

            # Load additional data based on checkboxes
            extras_loaded = []

            # Note: Registration boundaries are now only loaded in Registration QC mode
            # In Cell Detection mode, they are not needed

            # Load detection results if requested
            if self.load_best_detection_cb.isChecked():
                best_loaded = self._load_detection_results(best=True)
                if best_loaded:
                    extras_loaded.append("best detection")

            if self.load_recent_detection_cb.isChecked():
                recent_loaded = self._load_detection_results(best=False)
                if recent_loaded:
                    extras_loaded.append("recent detection")

            # Load classification results if requested
            if self.load_best_classification_cb.isChecked():
                best_class_loaded = self._load_classification_results(best=True)
                if best_class_loaded:
                    extras_loaded.append("best classification")

            if self.load_recent_classification_cb.isChecked():
                recent_class_loaded = self._load_classification_results(best=False)
                if recent_class_loaded:
                    extras_loaded.append("recent classification")

            status_msg = f"Loaded: {self.current_brain.name} ({source})"
            if extras_loaded:
                status_msg += f" + {', '.join(extras_loaded)}"

            self.brain_status_label.setText(status_msg)
            self.brain_status_label.setStyleSheet("color: green; font-weight: bold;")
            print(f"[SCI-Connectome] Brain loaded successfully!\n")

            # Log to session documenter
            if self.session_doc:
                self.session_doc.log_brain_loaded(source)

            # Auto-start zarr conversion in background if not already done
            self._auto_start_zarr_conversion()

        else:
            self.brain_status_label.setText("Error loading brain")
            self.brain_status_label.setStyleSheet("color: red;")
            print(f"[SCI-Connectome] ERROR: {message}\n")
            QMessageBox.warning(self, "Load Error", f"Failed to load brain:\n{message}")

    def _calculate_contrast_limits(self, stack):
        """
        Calculate good contrast limits for microscopy data.

        Uses percentile-based approach on a sample slice to find where
        the actual signal is, rather than using the full data range
        (which often has outliers that crush the contrast).

        Args:
            stack: 3D numpy/dask array (Z, Y, X)

        Returns:
            tuple: (min, max) contrast limits
        """
        try:
            # Sample the middle slice for speed (avoid edge artifacts)
            if hasattr(stack, 'shape') and len(stack.shape) >= 3:
                mid_z = stack.shape[0] // 2
                # For dask arrays, compute just this slice
                sample = stack[mid_z]
                if hasattr(sample, 'compute'):
                    sample = sample.compute()
            else:
                sample = stack
                if hasattr(sample, 'compute'):
                    sample = sample.compute()

            # Use percentiles to ignore outliers
            # 0.5th percentile for black point (ignore hot pixels at 0)
            # 99.5th percentile for white point (ignore saturated pixels)
            p_low = np.percentile(sample, 0.5)
            p_high = np.percentile(sample, 99.5)

            # For cells, we often want to boost contrast more aggressively
            # Use 99.9th percentile to really make bright cells pop
            p_high_bright = np.percentile(sample, 99.9)

            # If the 99.9th is not much higher than 99.5th, use it
            # (means there are distinct bright features = cells)
            if p_high_bright < p_high * 1.5:
                p_high = p_high_bright

            # Ensure we have a valid range
            if p_high <= p_low:
                p_high = p_low + 1

            return (float(p_low), float(p_high))

        except Exception as e:
            print(f"  Warning: Could not calculate contrast limits: {e}")
            # Return None to let napari use defaults
            return None

    def _load_registration_qc_view(self) -> bool:
        """
        Load registration QC view with brain, boundaries, and region labels.

        Priority order:
        1. Manually cropped brain (if exists) - user may have specifically cropped for registration
        2. Downsampled brain from registration output
        3. Falls back gracefully if registration incomplete

        All files are already in compatible spaces, no expensive transformations needed.
        """
        if not self.current_brain:
            return False

        # Clear viewer for clean registration QC view
        self.viewer.layers.clear()

        atlas_dir = self.current_brain / "3_Registered_Atlas"
        boundaries_path = atlas_dir / "boundaries.tiff"
        brainreg_json_path = atlas_dir / "brainreg.json"

        if not boundaries_path.exists():
            QMessageBox.warning(self, "Error", f"No registration found for {self.current_brain.name}")
            print(f"  No registration boundaries found at {boundaries_path}")
            return False

        try:
            import tifffile
            import json

            print(f"\n[SCI-Connectome] Loading Registration QC view for {self.current_brain.name}")

            # Load metadata to calculate scale
            scale = (1.0, 1.0, 1.0)  # Default - both files already match
            if brainreg_json_path.exists():
                with open(brainreg_json_path) as f:
                    metadata = json.load(f)
                print(f"  Registration atlas: {metadata.get('atlas', 'unknown')}")

            # Try to load manually cropped brain first (higher priority)
            manual_crop_path = self._find_manual_crop(self.current_brain)

            if manual_crop_path and (manual_crop_path / "ch0").exists():
                print(f"  Loading manually cropped brain for registration QC...")
                # User manually cropped, so use their crop for QC
                # Use signal channel
                signal_folder = manual_crop_path / "ch0"
                # Load as tiff sequence
                import tifffile
                from natsort import natsorted
                tiff_files = natsorted(signal_folder.glob("*.tif*"))
                if tiff_files:
                    print(f"    Found {len(tiff_files)} TIFF slices in manual crop")
                    manual_brain = tifffile.imread([str(f) for f in tiff_files])
                    self.viewer.add_image(
                        manual_brain,
                        name="Brain (manually cropped)",
                        colormap='green',
                        blending='additive',
                        gamma=1.2,  # User-preferred setting for better visibility
                        visible=True
                    )
                    print(f"    Added manual crop brain layer: {manual_brain.shape}")
            else:
                # No manual crop, fall back to downsampled
                downsampled_path = atlas_dir / "downsampled.tiff"
                if downsampled_path.exists():
                    print(f"  Loading downsampled brain from registration...")
                    downsampled_brain = tifffile.imread(str(downsampled_path))
                    self.viewer.add_image(
                        downsampled_brain,
                        name="Brain (downsampled)",
                        scale=scale,
                        colormap='green',
                        blending='additive',
                        gamma=1.2,  # User-preferred setting for better visibility
                        visible=True
                    )
                    print(f"    Added registered brain layer: {downsampled_brain.shape}")
                else:
                    QMessageBox.warning(self, "Error", "No brain data found for registration QC")
                    return False

            # Load boundaries
            print(f"  Loading atlas region boundaries...")
            boundaries = tifffile.imread(str(boundaries_path))
            self.viewer.add_image(
                boundaries,
                name="Atlas boundaries",
                scale=scale,
                colormap='gray',
                opacity=0.5,
                blending='additive',
                visible=True  # Show by default in QC mode
            )
            print(f"    Boundaries shape: {boundaries.shape}")

            # Load region labels if available
            atlas_labels_path = atlas_dir / "registered_atlas.tiff"
            if atlas_labels_path.exists():
                print(f"  Loading atlas region labels...")
                atlas_labels = tifffile.imread(str(atlas_labels_path))
                self.viewer.add_labels(
                    atlas_labels,
                    name="Atlas regions",
                    scale=scale,
                    opacity=0.3,
                    visible=False  # Hidden but available
                )
                print(f"    Region labels shape: {atlas_labels.shape}")

            print(f"  ✓ Registration QC view ready")
            self.brain_status_label.setText(f"Registration QC: {self.current_brain.name}")
            self.brain_status_label.setStyleSheet("color: #4CAF50;")
            return True

        except Exception as e:
            print(f"  Failed to load registration QC: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to load registration: {e}")
            return False

    def _load_detection_results(self, best: bool = True) -> bool:
        """
        Load detection results from calibration runs.

        For calibration/tuning: loads results from tracker-recorded calibration runs.
        - best=True: load best/most recent calibration run attempt
        - best=False: load second most recent for comparison

        Priority:
        1. Query tracker for recorded calibration runs (settings, parameters, results)
        2. Fallback to local 4_Cell_Candidates XML files

        Args:
            best: If True, load best run. If False, load previous run for comparison.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.current_brain:
            return False

        # Try tracker first - query calibration runs for this brain
        if self.tracker:
            brain_name = self.current_brain.name
            calibration_runs = self.tracker.search(brain=brain_name, exp_type="detection")

            if calibration_runs:
                # Filter completed runs
                completed = [r for r in calibration_runs if r.get('status') == 'completed']

                if completed:
                    # Sort by creation time (newest first)
                    completed.sort(key=lambda x: x.get('created_at', ''), reverse=True)

                    if best:
                        # Load most recent run (your best current attempt)
                        target_run = completed[0]
                        label_suffix = "(best attempt)"
                        cell_type = 'best'
                    else:
                        # Load second most recent for comparison
                        if len(completed) > 1:
                            target_run = completed[1]
                            label_suffix = "(previous)"
                        else:
                            # Only one run, use it
                            target_run = completed[0]
                            label_suffix = "(only result)"
                        cell_type = 'recent'

                    # Smart Loading: Check if classification exists for this detection
                    detection_exp_id = target_run.get('exp_id')
                    class_run = self._find_classification_for_detection(detection_exp_id)

                    if class_run:
                        # Classification exists - load cells.xml + rejected.xml instead
                        print(f"[SCI-Connectome] Smart loading: found classification for detection")
                        self._load_classification_for_detection(detection_exp_id, class_run, cell_type)
                        return True

                    # No classification - try to load detection from output path
                    output_path = target_run.get('output_path')
                    if output_path:
                        points_loaded = self._load_points_from_xml(
                            Path(output_path),
                            f"{self.current_brain.name} - Detection {label_suffix}",
                            cell_type=cell_type
                        )
                        if points_loaded:
                            return True

        # Fallback: Check filesystem for classification first, then detection
        if self.current_brain:
            classified_folder = self.current_brain / "5_Classified_Cells"
            cells_xml = classified_folder / "cells.xml"
            if cells_xml.exists():
                print(f"[SCI-Connectome] Smart loading: found classification on filesystem")
                self._load_classified_cells()
                return True

        # Fallback: Load directly from 4_Cell_Candidates folder
        candidates_folder = self.current_brain / "4_Cell_Candidates"
        if candidates_folder.exists():
            # Find all Detected_*.xml files
            detected_files = list(candidates_folder.glob("Detected_*.xml"))
            if detected_files:
                # Sort by modification time (newest first)
                detected_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

                if best:
                    # Load most recent
                    xml_file = detected_files[0]
                    label_suffix = "(best attempt)"
                    cell_type = 'best'
                else:
                    # Load second most recent for comparison
                    if len(detected_files) > 1:
                        xml_file = detected_files[1]
                        label_suffix = "(previous)"
                    else:
                        xml_file = detected_files[0]
                        label_suffix = "(only result)"
                    cell_type = 'recent'

                return self._load_points_from_xml(
                    xml_file,
                    f"{self.current_brain.name} - Detection {label_suffix}",
                    cell_type=cell_type
                )

        print(f"  No calibration runs found for {self.current_brain.name}")
        return False

    def _load_classification_results(self, best: bool = True) -> bool:
        """
        Load classification results (cells.xml + rejected.xml).

        For calibration/tuning: loads results from tracker-recorded classification runs.
        - best=True: load best/highest-rated classification
        - best=False: load most recent classification for comparison

        Args:
            best: If True, load best run. If False, load recent for comparison.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.current_brain:
            return False

        # Determine cell_type for styling
        cell_type_cells = 'class_prev_cells' if best else 'class_new_cells'
        cell_type_rejected = 'class_prev_rejected' if best else 'class_new_rejected'
        label_suffix = "(best)" if best else "(recent)"

        # Try tracker first - query classification runs for this brain
        if self.tracker:
            brain_name = self.current_brain.name
            class_runs = self.tracker.search(brain=brain_name, exp_type="classification")

            if class_runs:
                # Filter completed runs
                completed = [r for r in class_runs if r.get('status') == 'completed']

                if completed:
                    # Sort by creation time (newest first) or rating
                    if best:
                        # Prefer highest rated, then most recent
                        def safe_rating(x):
                            try:
                                return -float(x.get('rating', 0) or 0)
                            except (ValueError, TypeError):
                                return 0
                        completed.sort(key=lambda x: (safe_rating(x), x.get('created_at', '')), reverse=True)
                    else:
                        # Most recent
                        completed.sort(key=lambda x: x.get('created_at', ''), reverse=True)

                    target_run = completed[0]

                    # Get output folder
                    output_path = target_run.get('output_path')
                    if output_path:
                        output_folder = Path(output_path)
                        if output_folder.is_file():
                            output_folder = output_folder.parent

                        cells_xml = output_folder / "cells.xml"
                        rejected_xml = output_folder / "rejected.xml"

                        cells_loaded = False
                        rejected_loaded = False

                        if cells_xml.exists():
                            cells_loaded = self._load_points_from_xml(
                                cells_xml,
                                f"Classified Cells {label_suffix}",
                                cell_type=cell_type_cells
                            )

                        if rejected_xml.exists():
                            rejected_loaded = self._load_points_from_xml(
                                rejected_xml,
                                f"Rejected {label_suffix}",
                                cell_type=cell_type_rejected
                            )

                        if cells_loaded or rejected_loaded:
                            n_cells = target_run.get('class_cells_kept', '?')
                            n_rejected = target_run.get('class_cells_rejected', '?')
                            print(f"[SCI-Connectome] Loaded classification {label_suffix}: {n_cells} cells, {n_rejected} rejected")
                            return True

        # Fallback: Load directly from 5_Classified_Cells folder
        classified_folder = self.current_brain / "5_Classified_Cells"
        if classified_folder.exists():
            cells_xml = classified_folder / "cells.xml"
            rejected_xml = classified_folder / "rejected.xml"

            cells_loaded = False
            rejected_loaded = False

            if cells_xml.exists():
                cells_loaded = self._load_points_from_xml(
                    cells_xml,
                    f"Classified Cells {label_suffix}",
                    cell_type=cell_type_cells
                )

            if rejected_xml.exists():
                rejected_loaded = self._load_points_from_xml(
                    rejected_xml,
                    f"Rejected {label_suffix}",
                    cell_type=cell_type_rejected
                )

            if cells_loaded or rejected_loaded:
                print(f"[SCI-Connectome] Loaded classification from filesystem {label_suffix}")
                return True

        print(f"  No classification results found for {self.current_brain.name}")
        return False

    # ==========================================================================
    # COLOR SCHEME FOR ALL LAYER TYPES (14 colors)
    # All layers visible by default - colors chosen for maximum distinguishability
    # NOTE: Size is uniform (12) - only colors/symbols vary for distinction
    # ==========================================================================
    CELL_COLORS = {
        # ----------------------------------------------------------------------
        # DETECTION TUNING (pre-classification candidates)
        # Using 'o' symbol (circle outline only - no fill)
        # Valid napari symbols: disc, o, square, diamond, cross, x, arrow,
        #                       clobber, star, triangle_up, triangle_down
        # NOTE: 'o' renders as outline only, not filled
        # ----------------------------------------------------------------------
        'det_best': {
            'face': 'transparent',    # Hollow so cells visible inside
            'edge': '#00FFFF',        # Cyan border
            'symbol': 'o',            # Circle outline only - see through to cells
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,      # Thin outline width
            'description': 'Best detection (user-marked highest quality)'
        },
        'det_recent': {
            'face': 'transparent',
            'edge': '#FFA500',        # Orange border
            'symbol': 'o',            # Circle outline - different color from best
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'Most recent detection run'
        },
        'det_diff_only_best': {
            'face': 'transparent',
            'edge': '#FFFF00',        # Yellow border
            'symbol': 'diamond',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'In BEST only (potential false negatives in recent)'
        },
        'det_diff_only_recent': {
            'face': 'transparent',
            'edge': '#FFFFFF',        # White border
            'symbol': 'triangle_up',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'In RECENT only (new candidates)'
        },

        # ----------------------------------------------------------------------
        # CLASSIFICATION COMPARISON (post-network results)
        # ----------------------------------------------------------------------
        'class_prev_cells': {
            'face': 'transparent',
            'edge': '#00FFFF',        # Cyan border
            'symbol': 'o',            # Circle outline
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'Previous model: classified as cells'
        },
        'class_prev_rejected': {
            'face': 'transparent',
            'edge': '#FFA500',        # Orange border
            'symbol': 'cross',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'Previous model: rejected'
        },
        'class_new_cells': {
            'face': 'transparent',
            'edge': '#00FFFF',        # Cyan border
            'symbol': 'o',            # Circle outline
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'New model: classified as cells'
        },
        'class_new_rejected': {
            'face': 'transparent',
            'edge': '#FFA500',        # Orange border
            'symbol': 'cross',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'New model: rejected'
        },
        'class_diff_gained': {
            'face': 'transparent',
            'edge': '#FFFF00',        # Yellow border
            'symbol': 'star',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'GAINED: cell in new, not in previous'
        },
        'class_diff_lost': {
            'face': 'transparent',
            'edge': '#FFFFFF',        # White border
            'symbol': 'x',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'LOST: cell in previous, not in new'
        },

        # ----------------------------------------------------------------------
        # TRAINING/CURATION (user-curated data)
        # ----------------------------------------------------------------------
        'train_cells': {
            'face': 'transparent',
            'edge': '#FFFF00',        # Yellow border
            'symbol': 'o',            # Circle outline
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'Training bucket: confirmed cells'
        },
        'train_non_cells': {
            'face': 'transparent',
            'edge': '#FF4444',        # Red border
            'symbol': 'x',
            'size': 14,
            'opacity': 0.2,
            'border_width': 0.1,
            'description': 'Training bucket: confirmed non-cells'
        },

        # ----------------------------------------------------------------------
        # UTILITY
        # ----------------------------------------------------------------------
        'current_focus': {
            'face': 'transparent',
            'edge': '#FFFFFF',        # White border
            'symbol': 'o',            # Circle outline
            'size': 18,               # Slightly larger for focus
            'opacity': 0.3,           # Slightly higher for focus visibility
            'border_width': 0.15,
            'description': 'Currently focused/selected point'
        },
        'archived': {
            'face': 'transparent',
            'edge': '#808080',        # Gray border
            'symbol': 'o',            # Circle outline
            'size': 14,
            'opacity': 0.1,           # Very subtle for archived
            'border_width': 0.08,
            'description': 'Archived/demoted layer'
        },

        # ----------------------------------------------------------------------
        # LEGACY ALIASES (for backward compatibility)
        # Using 'o' (circle outline) symbol so user can see cells inside
        # Colors: Cyan (#00FFFF), Orange (#FFA500), Yellow (#FFFF00), White (#FFFFFF)
        # Avoids green/magenta which match fluorescence channels
        # ----------------------------------------------------------------------
        'best': {'face': 'transparent', 'edge': '#00FFFF', 'symbol': 'o', 'size': 14, 'opacity': 0.2, 'border_width': 0.1},
        'recent': {'face': 'transparent', 'edge': '#FFA500', 'symbol': 'o', 'size': 14, 'opacity': 0.2, 'border_width': 0.1},
        'cells': {'face': 'transparent', 'edge': '#00FFFF', 'symbol': 'o', 'size': 14, 'opacity': 0.2, 'border_width': 0.1},
        'rejected': {'face': 'transparent', 'edge': '#FFA500', 'symbol': 'cross', 'size': 14, 'opacity': 0.2, 'border_width': 0.1},
        'new': {'face': 'transparent', 'edge': '#FFFFFF', 'symbol': 'o', 'size': 14, 'opacity': 0.2, 'border_width': 0.1},  # White circle outline - visible against any background
    }

    # ==========================================================================
    # LAYER MANAGEMENT HELPERS
    # ==========================================================================

    def _get_layer_by_type(self, cell_type: str):
        """
        Find a napari points layer by its cell type based on metadata or naming.

        Args:
            cell_type: One of the CELL_COLORS keys (e.g., 'det_best', 'class_new_cells')

        Returns:
            The napari Points layer or None if not found.
        """
        import napari

        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Points):
                continue

            # Check by stored metadata first (most reliable)
            if hasattr(layer, 'metadata') and layer.metadata.get('cell_type') == cell_type:
                return layer

            # Check by name convention
            type_prefixes = {
                'det_best': ['BEST:', 'Best detection', 'best attempt', '★'],
                'det_recent': ['RECENT:', 'Recent detection', 'Most recent', 'only recent', '●'],
                'det_diff_only_best': ['Diff: Only in BEST'],
                'det_diff_only_recent': ['Diff: Only in RECENT'],
                'class_prev_cells': ['Prev Cells', 'Previous Cells'],
                'class_prev_rejected': ['Prev Rejected', 'Previous Rejected'],
                'class_new_cells': ['New Cells', 'Classified Cells'],
                'class_new_rejected': ['New Rejected', 'Rejected'],
                'class_diff_gained': ['GAINED:', 'Diff: Gained'],
                'class_diff_lost': ['LOST:', 'Diff: Lost'],
                'train_cells': ['Training Cells', 'Train Cells'],
                'train_non_cells': ['Training Non-Cells', 'Train Non-Cells'],
                'archived': ['[archived]'],
            }

            prefixes = type_prefixes.get(cell_type, [])
            for prefix in prefixes:
                if prefix.lower() in layer.name.lower():
                    return layer

        return None

    def _set_layer_type(self, layer, cell_type: str):
        """
        Apply cell type styling to a layer and store metadata.

        Args:
            layer: napari Points layer to style
            cell_type: One of the CELL_COLORS keys
        """
        style = self.CELL_COLORS.get(cell_type)
        if not style:
            print(f"[SCI-Connectome] Warning: Unknown cell_type '{cell_type}'")
            return

        layer.face_color = style['face']
        layer.border_color = style['edge']
        layer.symbol = style['symbol']
        layer.size = style['size']
        layer.opacity = style['opacity']
        layer.border_width = style.get('border_width', 0.1)

        # Store type in metadata for later retrieval
        if not hasattr(layer, 'metadata') or layer.metadata is None:
            layer.metadata = {}
        layer.metadata['cell_type'] = cell_type
        layer.metadata['sci_connectome'] = True

    def _add_points_layer(self, coords, name: str, cell_type: str, visible: bool = True):
        """
        Add a points layer with consistent styling based on cell type.

        Args:
            coords: numpy array of point coordinates (N, 3) in (z, y, x) order
            name: Layer name
            cell_type: One of the CELL_COLORS keys
            visible: Whether layer is visible (default True - never hide by default)

        Returns:
            The created napari Points layer
        """
        style = self.CELL_COLORS.get(cell_type, self.CELL_COLORS.get('new'))

        layer = self.viewer.add_points(
            coords,
            name=name,
            size=style['size'],
            face_color=style['face'],
            border_color=style['edge'],
            symbol=style['symbol'],
            opacity=style['opacity'],
            border_width=style.get('border_width', 0.1),
            n_dimensional=True,
            visible=visible,
        )

        # Store metadata
        if not hasattr(layer, 'metadata') or layer.metadata is None:
            layer.metadata = {}
        layer.metadata['cell_type'] = cell_type
        layer.metadata['sci_connectome'] = True

        return layer

    def _archive_layer(self, layer, archived_by: str = None):
        """
        Archive a layer (demote it) with metadata about when/why.

        Args:
            layer: napari Points layer to archive
            archived_by: exp_id or description of what replaced this layer
        """
        from datetime import datetime

        # Update styling to archived
        self._set_layer_type(layer, 'archived')

        # Rename with archived prefix
        if not layer.name.startswith('[archived]'):
            layer.name = f"[archived] {layer.name}"

        # Store archival metadata
        layer.metadata['archived_at'] = datetime.now().isoformat()
        layer.metadata['archived_by'] = archived_by or 'unknown'
        layer.metadata['original_cell_type'] = layer.metadata.get('cell_type', 'unknown')

        print(f"[SCI-Connectome] Archived layer: {layer.name}")

    def _load_points_from_xml(self, xml_path: Path, layer_name: str, cell_type: str = 'best') -> bool:
        """
        Load cell coordinates from BrainGlobe XML format and add as points layer.

        The XML has MarkerX, MarkerY, MarkerZ tags for each cell.
        Napari expects (Z, Y, X) order for 3D points.

        Args:
            xml_path: Path to the XML file
            layer_name: Name for the napari layer
            cell_type: One of 'best', 'recent', 'cells', 'rejected' - controls color/style
        """
        if not xml_path.exists():
            print(f"  XML file not found: {xml_path}")
            return False

        # If it's a directory, it's not a valid XML file - skip with warning
        if xml_path.is_dir():
            print(f"  Skipping directory (not an XML file): {xml_path.name}")
            return False

        try:
            import xml.etree.ElementTree as ET

            print(f"  Loading points from {xml_path.name}...")
            tree = ET.parse(str(xml_path))
            root = tree.getroot()

            points = []
            for marker in root.iter('Marker'):
                x_elem = marker.find('MarkerX')
                y_elem = marker.find('MarkerY')
                z_elem = marker.find('MarkerZ')

                if x_elem is not None and y_elem is not None and z_elem is not None:
                    x = int(x_elem.text)
                    y = int(y_elem.text)
                    z = int(z_elem.text)
                    # Napari uses (Z, Y, X) order for 3D coordinates
                    points.append([z, y, x])

            if not points:
                print(f"  No points found in {xml_path.name}")
                return False

            coords = np.array(points)
            print(f"  Loaded {len(coords)} points from {xml_path.name}")

            # Get style based on cell type
            style = self.CELL_COLORS.get(cell_type, self.CELL_COLORS['best'])

            # Add as points layer with type-specific styling
            self.viewer.add_points(
                coords,
                name=layer_name,
                size=style['size'],
                n_dimensional=True,
                opacity=style['opacity'],
                symbol=style['symbol'],
                face_color=style['face'],
                border_color=style['edge'],
                border_width=style.get('border_width', 0.1),
                visible=(cell_type != 'rejected'),  # Hide rejected by default
            )
            print(f"  Layer style: size={style['size']}, symbol={style['symbol']}, border={style.get('border_width', 0.1)}, opacity={style['opacity']}")
            return True

        except Exception as e:
            print(f"  Error loading points from XML: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_metadata(self):
        """Load existing metadata if available."""
        if not self.current_brain:
            return

        # Try to load from folders in priority order: manual_crop > auto_crop > extracted
        for folder in ["2_Cropped_For_Registration_Manual", "2_Cropped_For_Registration", "1_Extracted_Full"]:
            meta_path = self.current_brain / folder / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        self.metadata = json.load(f)

                    # Update UI from metadata
                    if 'voxel_size_um' in self.metadata:
                        voxel = self.metadata['voxel_size_um']
                        self.voxel_xy_label.setText(f"{voxel.get('x', 4.0):.2f} µm")
                        self.voxel_z_label.setText(f"{voxel.get('z', 4.0):.2f} µm")
                        self.z_step.setValue(voxel.get('z', 4.0))

                    if 'orientation' in self.metadata:
                        orient = self.metadata['orientation']
                        idx = self.orientation_combo.findData(orient)
                        if idx >= 0:
                            self.orientation_combo.setCurrentIndex(idx)

                    # Load ALL detection params if saved
                    if 'detection_params' in self.metadata:
                        det = self.metadata['detection_params']
                        # Core ball filter
                        if 'ball_xy_size' in det:
                            self.ball_xy.setValue(det['ball_xy_size'])
                        if 'ball_z_size' in det:
                            self.ball_z.setValue(det['ball_z_size'])
                        if 'ball_overlap_fraction' in det:
                            self.ball_overlap_fraction.setValue(det['ball_overlap_fraction'])
                        # Soma
                        if 'soma_diameter' in det:
                            self.soma_diameter.setValue(det['soma_diameter'])
                        if 'soma_spread_factor' in det:
                            self.soma_spread_factor.setValue(det['soma_spread_factor'])
                        # Thresholding
                        if 'threshold' in det:
                            self.threshold.setValue(det['threshold'])
                        if 'tiled_threshold' in det:
                            self.tiled_threshold.setValue(det['tiled_threshold'])
                        if 'log_sigma_size' in det:
                            self.log_sigma_size.setValue(det['log_sigma_size'])
                        # Cluster handling
                        if 'max_cluster_size' in det:
                            self.max_cluster_size.setValue(det['max_cluster_size'])
                        # Performance
                        if 'use_gpu' in det:
                            self.use_gpu.setChecked(det['use_gpu'])
                        if 'batch_size' in det:
                            self.batch_size.setValue(det['batch_size'])
                        if 'n_free_cpus' in det:
                            self.n_free_cpus.setValue(det['n_free_cpus'])

                    # Check if channels are swapped (from manual crop widget)
                    if 'channels_swapped' in self.metadata:
                        self.swap_channels.setChecked(self.metadata['channels_swapped'])
                        if self.metadata['channels_swapped']:
                            print(f"[SCI-Connectome] Metadata says channels are swapped (signal=ch1)")

                    return
                except:
                    pass

    def save_metadata(self):
        """Save current settings to metadata."""
        print("[DEBUG] Button clicked: save_metadata")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        # Update metadata from UI
        self.metadata['voxel_size_um'] = {
            'x': self.camera_pixel.value() / self.magnification.value(),
            'y': self.camera_pixel.value() / self.magnification.value(),
            'z': self.z_step.value(),
        }

        orient = self.orientation_combo.currentData()
        if orient == "custom":
            orient = self.custom_orientation.text()
        self.metadata['orientation'] = orient

        self.metadata['crop_settings'] = {
            'width_threshold': self.width_threshold.value(),
            'margin_slices': self.margin_slices.value(),
            'manual_y_max': self.manual_y_max.value() if self.use_manual_crop.isChecked() else None,
        }

        self.metadata['detection_params'] = {
            # Core ball filter
            'ball_xy_size': self.ball_xy.value(),
            'ball_z_size': self.ball_z.value(),
            'ball_overlap_fraction': self.ball_overlap_fraction.value(),
            # Soma
            'soma_diameter': self.soma_diameter.value(),
            'soma_spread_factor': self.soma_spread_factor.value(),
            # Thresholding
            'threshold': self.threshold.value(),
            'tiled_threshold': self.tiled_threshold.value(),
            'log_sigma_size': self.log_sigma_size.value(),
            # Cluster handling
            'max_cluster_size': self.max_cluster_size.value(),
            # Performance
            'use_gpu': self.use_gpu.isChecked(),
            'batch_size': self.batch_size.value(),
            'n_free_cpus': self.n_free_cpus.value(),
        }

        # Save to all data folders that exist
        for folder in ["2_Cropped_For_Registration_Manual", "2_Cropped_For_Registration", "1_Extracted_Full"]:
            meta_dir = self.current_brain / folder
            if meta_dir.exists():
                meta_path = meta_dir / "metadata.json"
                try:
                    with open(meta_path, 'w') as f:
                        json.dump(self.metadata, f, indent=2)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save: {e}")
                    return

        QMessageBox.information(self, "Saved", "Settings saved to metadata.json")

    def create_qc_tab(self):
        """Tab for registration QC and approval."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "Review registration quality before proceeding with cell detection.\n"
            "Bad registrations will give meaningless region counts!"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # QC Image section
        qc_group = QGroupBox("Registration QC Images")
        qc_layout = QVBoxLayout()
        qc_group.setLayout(qc_layout)

        qc_layout.addWidget(QLabel(
            "Generate QC images showing brain tissue with atlas boundaries overlaid.\n"
            "Use these to verify registration quality before proceeding."
        ))

        # Row of generate buttons
        gen_btn_row = QHBoxLayout()

        generate_slices_btn = QPushButton("Generate Slice Images")
        generate_slices_btn.setStyleSheet("background-color: #2196F3; color: white;")
        generate_slices_btn.setToolTip("Generate individual QC_slice_NNNN.png images at strategic Z positions")
        generate_slices_btn.clicked.connect(self.generate_qc_slices)
        gen_btn_row.addWidget(generate_slices_btn)

        generate_panel_btn = QPushButton("Generate Panel Image")
        generate_panel_btn.setToolTip("Generate single multi-panel QC_registration_detailed.png")
        generate_panel_btn.clicked.connect(self.generate_qc)
        gen_btn_row.addWidget(generate_panel_btn)

        qc_layout.addLayout(gen_btn_row)

        view_qc_btn = QPushButton("View QC Images")
        view_qc_btn.clicked.connect(self.view_qc)
        qc_layout.addWidget(view_qc_btn)

        layout.addWidget(qc_group)

        # Approval section
        approve_group = QGroupBox("Registration Approval")
        approve_layout = QVBoxLayout()
        approve_group.setLayout(approve_layout)

        self.approval_status = QLabel("Status: Unknown")
        self.approval_status.setFont(QFont("Arial", 11, QFont.Bold))
        approve_layout.addWidget(self.approval_status)

        approve_layout.addWidget(QLabel(
            "If registration looks good, approve it to proceed.\n"
            "If it looks bad, adjust crop/orientation and re-register."
        ))

        btn_layout = QHBoxLayout()

        approve_btn = QPushButton("Approve")
        approve_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        approve_btn.clicked.connect(self.approve_registration)
        btn_layout.addWidget(approve_btn)

        reject_btn = QPushButton("Reject")
        reject_btn.setStyleSheet("background-color: #f44336; color: white;")
        reject_btn.clicked.connect(self.reject_registration)
        btn_layout.addWidget(reject_btn)

        approve_layout.addLayout(btn_layout)
        layout.addWidget(approve_group)

        layout.addStretch()
        return widget

    def generate_qc(self, slices_mode=False):
        """Generate QC image(s) for registration."""
        print("[DEBUG] Button clicked: generate_qc")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        script = SCRIPTS_DIR / "util_registration_qc.py"
        if not script.exists():
            QMessageBox.warning(self, "Error", "Registration QC script not found")
            return

        cmd = [sys.executable, str(script), "--brain", self.current_brain.name]
        if slices_mode:
            cmd.append("--slices")

        print(f"[SCI-Connectome] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"[SCI-Connectome] QC generation output:\n{result.stdout}")
                if slices_mode:
                    QMessageBox.information(self, "QC Generated",
                        "Individual slice QC images saved to 3_Registered_Atlas/")
                else:
                    QMessageBox.information(self, "QC Generated",
                        "QC_registration_detailed.png saved to 3_Registered_Atlas/")
            else:
                print(f"[SCI-Connectome] QC generation error:\n{result.stderr}")
                QMessageBox.warning(self, "Error", f"QC generation failed:\n{result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            QMessageBox.warning(self, "Timeout", "QC generation timed out")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"QC generation failed: {e}")

    def generate_qc_slices(self):
        """Generate individual slice QC images."""
        print("[DEBUG] Button clicked: generate_qc_slices")
        self.generate_qc(slices_mode=True)

    def view_qc(self):
        """View the QC image."""
        print("[DEBUG] Button clicked: view_qc")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        import os
        reg_folder = self.current_brain / "3_Registered_Atlas"

        # Try detailed first, then look for slice images
        qc_path = reg_folder / "QC_registration_detailed.png"
        if qc_path.exists():
            if os.name == 'nt':
                os.startfile(str(qc_path))
            else:
                subprocess.run(['open', str(qc_path)])
        else:
            # Check for slice images
            slice_images = list(reg_folder.glob("QC_slice_*.png"))
            if slice_images:
                # Open the first one
                if os.name == 'nt':
                    os.startfile(str(slice_images[0]))
                else:
                    subprocess.run(['open', str(slice_images[0])])
            else:
                QMessageBox.warning(self, "Not Found", "No QC images found. Generate them first.")

    def approve_registration(self):
        """Approve the registration."""
        print("[DEBUG] Button clicked: approve_registration")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        approval_file = self.current_brain / "3_Registered_Atlas" / ".registration_approved"
        try:
            approval_file.touch()
            self.approval_status.setText("Status: APPROVED")
            self.approval_status.setStyleSheet("color: green;")
            QMessageBox.information(self, "Approved", "Registration approved!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to approve: {e}")

    def reject_registration(self):
        """Reject the registration."""
        print("[DEBUG] Button clicked: reject_registration")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        approval_file = self.current_brain / "3_Registered_Atlas" / ".registration_approved"
        if approval_file.exists():
            approval_file.unlink()

        self.approval_status.setText("Status: REJECTED")
        self.approval_status.setStyleSheet("color: red;")
        QMessageBox.information(self, "Rejected", "Registration rejected. Adjust settings and re-register.")

    def create_results_tab(self):
        """
        Tab for viewing and comparing results with multiple views.

        Sub-tabs:
        - Allen Atlas: Full 236-region detail from Allen Mouse Brain Atlas
        - eLife Groups: 25 summary groups matching Wang et al. 2022
        - Cross-Brain: Compare multiple brains side-by-side
        - Run History: Track same brain across optimization runs
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # =====================================================================
        # TOP SECTION: Classification Selection + Run Button
        # =====================================================================
        top_group = QGroupBox("Region Counting")
        top_layout = QVBoxLayout()
        top_group.setLayout(top_layout)

        # Classification selection row
        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Classification:"))

        self.class_selection_status = QLabel("(none selected)")
        self.class_selection_status.setStyleSheet("color: gray; font-style: italic;")
        class_row.addWidget(self.class_selection_status, 1)

        select_class_btn = QPushButton("Select...")
        select_class_btn.clicked.connect(self._show_classification_selector)
        class_row.addWidget(select_class_btn)
        top_layout.addLayout(class_row)

        # Run button row
        run_row = QHBoxLayout()
        self.run_regions_btn = QPushButton("Run Region Counting")
        self.run_regions_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;"
        )
        self.run_regions_btn.clicked.connect(self.run_region_counting)
        run_row.addWidget(self.run_regions_btn)

        self.view_csv_btn = QPushButton("View CSV")
        self.view_csv_btn.clicked.connect(self.view_region_csv)
        run_row.addWidget(self.view_csv_btn)

        refresh_btn = QPushButton("Refresh Views")
        refresh_btn.clicked.connect(self._refresh_all_results_views)
        run_row.addWidget(refresh_btn)
        top_layout.addLayout(run_row)

        self.region_count_status = QLabel("Status: Ready")
        self.region_count_status.setStyleSheet("color: gray; font-style: italic;")
        top_layout.addWidget(self.region_count_status)

        layout.addWidget(top_group)

        # =====================================================================
        # MAIN SECTION: Sub-tabs for different views
        # =====================================================================
        self.results_subtabs = QTabWidget()

        # Sub-tab 1: Allen Atlas (full detail)
        self.results_subtabs.addTab(
            self._create_allen_atlas_view(), "Allen Atlas"
        )

        # Sub-tab 2: eLife Groups (25 summary groups)
        self.results_subtabs.addTab(
            self._create_elife_groups_view(), "eLife Groups"
        )

        # Sub-tab 3: Cross-Brain Comparison
        self.results_subtabs.addTab(
            self._create_cross_brain_view(), "Cross-Brain"
        )

        # Sub-tab 4: Run History
        self.results_subtabs.addTab(
            self._create_run_history_view(), "Run History"
        )

        layout.addWidget(self.results_subtabs, 1)  # Give it stretch

        # =====================================================================
        # BOTTOM SECTION: Export
        # =====================================================================
        export_row = QHBoxLayout()

        report_btn = QPushButton("Generate Report")
        report_btn.setStyleSheet("background-color: #FF5722; color: white; padding: 6px;")
        report_btn.clicked.connect(self.generate_comparison_report)
        export_row.addWidget(report_btn)

        export_btn = QPushButton("Export Session")
        export_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 6px;")
        export_btn.clicked.connect(self.export_results)
        export_row.addWidget(export_btn)

        self.export_status = QLabel("")
        self.export_status.setStyleSheet("color: gray; font-style: italic;")
        export_row.addWidget(self.export_status, 1)

        layout.addLayout(export_row)

        # Keep reference to runs_table for compatibility
        self.runs_table = QTableWidget()  # Hidden, for compatibility

        return widget

    def _create_allen_atlas_view(self):
        """Create the Allen Atlas full-detail view."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel("Full Allen Mouse Brain Atlas regions (236 regions)")
        info.setStyleSheet("color: gray;")
        layout.addWidget(info)

        # Search/filter row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.allen_filter = QLineEdit()
        self.allen_filter.setPlaceholderText("Type to filter regions...")
        self.allen_filter.textChanged.connect(self._filter_allen_table)
        filter_row.addWidget(self.allen_filter)

        self.allen_sort_combo = QComboBox()
        self.allen_sort_combo.addItems(["Sort by Count", "Sort by Name", "Sort by Acronym"])
        self.allen_sort_combo.currentIndexChanged.connect(self._sort_allen_table)
        filter_row.addWidget(self.allen_sort_combo)
        layout.addLayout(filter_row)

        # Table
        self.allen_table = QTableWidget()
        self.allen_table.setColumnCount(4)
        self.allen_table.setHorizontalHeaderLabels(["Acronym", "Region Name", "Count", "%Total"])
        self.allen_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.allen_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.allen_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.allen_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.allen_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.allen_table.setSortingEnabled(True)
        layout.addWidget(self.allen_table)

        # Summary row
        self.allen_summary = QLabel("Total: 0 cells in 0 regions")
        self.allen_summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.allen_summary)

        return widget

    def _create_elife_groups_view(self):
        """Create the eLife 25-group summary view."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "eLife summary groups (Wang et al. 2022) - click row to see constituent regions"
        )
        info.setStyleSheet("color: gray;")
        layout.addWidget(info)

        # Main table (25 groups)
        self.elife_table = QTableWidget()
        self.elife_table.setColumnCount(4)
        self.elife_table.setHorizontalHeaderLabels(["Group", "Count", "Description", "Constituents"])
        self.elife_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.elife_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.elife_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.elife_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.elife_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.elife_table.itemSelectionChanged.connect(self._on_elife_row_selected)
        layout.addWidget(self.elife_table)

        # Constituents detail panel
        detail_group = QGroupBox("Constituent Regions")
        detail_layout = QVBoxLayout()
        detail_group.setLayout(detail_layout)

        self.elife_detail_table = QTableWidget()
        self.elife_detail_table.setColumnCount(3)
        self.elife_detail_table.setHorizontalHeaderLabels(["Acronym", "Region Name", "Count"])
        self.elife_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.elife_detail_table.setMaximumHeight(150)
        detail_layout.addWidget(self.elife_detail_table)

        layout.addWidget(detail_group)

        return widget

    def _create_cross_brain_view(self):
        """Create the cross-brain comparison view."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel("Compare region counts across multiple brains")
        info.setStyleSheet("color: gray;")
        layout.addWidget(info)

        # Brain selection
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Select brains to compare:"))

        self.cross_brain_list = QListWidget()
        self.cross_brain_list.setSelectionMode(QListWidget.MultiSelection)
        self.cross_brain_list.setMaximumHeight(100)
        select_row.addWidget(self.cross_brain_list, 1)

        btn_col = QVBoxLayout()
        load_brains_btn = QPushButton("Load Available")
        load_brains_btn.clicked.connect(self._load_available_brains_for_comparison)
        btn_col.addWidget(load_brains_btn)

        compare_btn = QPushButton("Compare Selected")
        compare_btn.setStyleSheet("background-color: #2196F3; color: white;")
        compare_btn.clicked.connect(self._run_cross_brain_comparison)
        btn_col.addWidget(compare_btn)
        select_row.addLayout(btn_col)

        layout.addLayout(select_row)

        # View mode toggle
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self.cross_view_combo = QComboBox()
        self.cross_view_combo.addItems(["Allen Atlas Regions", "eLife Groups"])
        self.cross_view_combo.currentIndexChanged.connect(self._update_cross_brain_view)
        view_row.addWidget(self.cross_view_combo)
        view_row.addStretch()
        layout.addLayout(view_row)

        # Comparison table (columns = brains, rows = regions)
        self.cross_brain_table = QTableWidget()
        self.cross_brain_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cross_brain_table.setSortingEnabled(True)
        layout.addWidget(self.cross_brain_table)

        return widget

    def _create_run_history_view(self):
        """Create the run history view for tracking optimization progress."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "Track region counts across optimization runs for the current brain.\n"
            "Shows how counts change as detection parameters are tuned."
        )
        info.setStyleSheet("color: gray;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Controls
        ctrl_row = QHBoxLayout()
        load_history_btn = QPushButton("Load Run History")
        load_history_btn.setStyleSheet("background-color: #2196F3; color: white;")
        load_history_btn.clicked.connect(self._load_run_history)
        ctrl_row.addWidget(load_history_btn)

        ctrl_row.addWidget(QLabel("View:"))
        self.history_view_combo = QComboBox()
        self.history_view_combo.addItems(["Top 20 Regions", "eLife Groups", "All Regions"])
        self.history_view_combo.currentIndexChanged.connect(self._update_history_view)
        ctrl_row.addWidget(self.history_view_combo)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # History table (columns = runs, rows = regions)
        self.history_table = QTableWidget()
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.history_table)

        # Trend summary
        self.history_summary = QLabel("")
        self.history_summary.setWordWrap(True)
        layout.addWidget(self.history_summary)

        return widget

    def _show_classification_selector(self):
        """Show dialog to select classification for region counting."""
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain", "Select a brain first")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Classification")
        dialog.setMinimumWidth(600)
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        layout.addWidget(QLabel("Select classification to use for region counting:"))

        # Table of available classifications
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Date", "Source", "Cells", "Model/Path", "Status"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(table)

        # Populate table
        all_entries = []

        # From tracker
        if self.tracker:
            runs = self.tracker.search(
                brain=self.current_brain.name,
                exp_type="classification",
                limit=20
            )
            for run in runs:
                if run.get('status') == 'completed':
                    cells = run.get('class_cells_found') or run.get('class_cells_kept') or '?'
                    model = Path(run.get('class_model_path', '')).stem if run.get('class_model_path') else '?'
                    output_path = run.get('output_path', '')
                    all_entries.append({
                        'date': run.get('created_at', '')[:16],
                        'source': 'Tracker',
                        'cells': str(cells),
                        'model': model[:30],
                        'status': 'completed',
                        'run': run,
                        'cells_xml': Path(output_path) / 'cells.xml' if output_path else None,
                        'sort_key': run.get('created_at', '')
                    })

        # From filesystem
        class_folder = self.current_brain / "5_Classified_Cells"
        if class_folder.exists():
            cells_xml = class_folder / "cells.xml"
            if cells_xml.exists():
                already_in = any(
                    e.get('cells_xml') and e['cells_xml'].exists() and
                    e['cells_xml'].resolve() == cells_xml.resolve()
                    for e in all_entries if e.get('cells_xml')
                )
                if not already_in:
                    try:
                        with open(cells_xml, 'r') as f:
                            n_cells = f.read().count('<Marker>')
                    except:
                        n_cells = '?'
                    mtime = cells_xml.stat().st_mtime
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    all_entries.append({
                        'date': date_str,
                        'source': 'Filesystem',
                        'cells': str(n_cells),
                        'model': '5_Classified_Cells/',
                        'status': 'on disk',
                        'run': None,
                        'cells_xml': cells_xml,
                        'sort_key': datetime.fromtimestamp(mtime).isoformat()
                    })

        all_entries.sort(key=lambda x: x['sort_key'], reverse=True)
        table.setRowCount(len(all_entries))

        for i, entry in enumerate(all_entries):
            table.setItem(i, 0, QTableWidgetItem(entry['date']))
            table.setItem(i, 1, QTableWidgetItem(entry['source']))
            table.setItem(i, 2, QTableWidgetItem(entry['cells']))
            table.setItem(i, 3, QTableWidgetItem(entry['model']))
            table.setItem(i, 4, QTableWidgetItem(entry['status']))

        # Store entries for selection
        dialog._entries = all_entries

        # Buttons
        btn_row = QHBoxLayout()
        select_btn = QPushButton("Use Selected")
        select_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        def on_select():
            row = table.currentRow()
            if row >= 0 and row < len(dialog._entries):
                entry = dialog._entries[row]
                self._selected_class_run = entry.get('run')
                self._selected_cells_xml = entry.get('cells_xml')
                self.class_selection_status.setText(
                    f"{entry['date']} ({entry['cells']} cells) [{entry['source']}]"
                )
                self.class_selection_status.setStyleSheet("color: blue; font-weight: bold;")
                dialog.accept()

        select_btn.clicked.connect(on_select)
        btn_row.addWidget(select_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        dialog.exec_()

    def _refresh_all_results_views(self):
        """Refresh all results sub-tabs."""
        self._load_allen_atlas_view()
        self._load_elife_groups_view()

    def _load_allen_atlas_view(self):
        """Load data into the Allen Atlas view."""
        if not self.current_brain:
            return

        csv_path = self.current_brain / "6_Region_Analysis" / "cell_counts_by_region.csv"
        if not csv_path.exists():
            self.allen_table.setRowCount(0)
            self.allen_summary.setText("No data - run region counting first")
            return

        # Load data
        import csv
        data = []
        total = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                acronym = row.get('region_acronym', '')
                name = row.get('region_name', '')
                try:
                    count = int(row.get('cell_count', 0))
                except:
                    count = 0
                if count > 0:
                    data.append((acronym, name, count))
                    total += count

        # Populate table
        self.allen_table.setSortingEnabled(False)
        self.allen_table.setRowCount(len(data))

        for i, (acronym, name, count) in enumerate(data):
            pct = (count / total * 100) if total > 0 else 0

            item_acr = QTableWidgetItem(acronym)
            item_name = QTableWidgetItem(name)
            item_count = QTableWidgetItem()
            item_count.setData(Qt.DisplayRole, count)
            item_pct = QTableWidgetItem(f"{pct:.1f}%")

            self.allen_table.setItem(i, 0, item_acr)
            self.allen_table.setItem(i, 1, item_name)
            self.allen_table.setItem(i, 2, item_count)
            self.allen_table.setItem(i, 3, item_pct)

        self.allen_table.setSortingEnabled(True)
        self.allen_table.sortItems(2, Qt.DescendingOrder)  # Sort by count

        self.allen_summary.setText(f"Total: {total:,} cells in {len(data)} regions")

        # Store data for filtering
        self._allen_data = data
        self._allen_total = total

    def _filter_allen_table(self, text):
        """Filter Allen table by search text."""
        text = text.lower()
        for row in range(self.allen_table.rowCount()):
            acronym = self.allen_table.item(row, 0).text().lower()
            name = self.allen_table.item(row, 1).text().lower()
            match = text in acronym or text in name
            self.allen_table.setRowHidden(row, not match)

    def _sort_allen_table(self, index):
        """Sort Allen table by selected column."""
        if index == 0:  # Count
            self.allen_table.sortItems(2, Qt.DescendingOrder)
        elif index == 1:  # Name
            self.allen_table.sortItems(1, Qt.AscendingOrder)
        elif index == 2:  # Acronym
            self.allen_table.sortItems(0, Qt.AscendingOrder)

    def _load_elife_groups_view(self):
        """Load data into the eLife groups view."""
        if not self.current_brain:
            return

        csv_path = self.current_brain / "6_Region_Analysis" / "cell_counts_elife_grouped.csv"
        if not csv_path.exists():
            self.elife_table.setRowCount(0)
            return

        # Load data
        import csv
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                group = row.get('elife_group', '')
                try:
                    count = int(row.get('cell_count', 0))
                except:
                    count = 0
                desc = row.get('description', '')
                constituents = row.get('constituent_regions', '')
                data.append((group, count, desc, constituents))

        # Populate table
        self.elife_table.setRowCount(len(data))
        for i, (group, count, desc, constituents) in enumerate(data):
            self.elife_table.setItem(i, 0, QTableWidgetItem(group))

            count_item = QTableWidgetItem()
            count_item.setData(Qt.DisplayRole, count)
            self.elife_table.setItem(i, 1, count_item)

            self.elife_table.setItem(i, 2, QTableWidgetItem(desc))

            # Show constituent count
            n_const = len(constituents.split(';')) if constituents else 0
            self.elife_table.setItem(i, 3, QTableWidgetItem(f"{n_const} regions"))

        # Store for detail view
        self._elife_data = data

    def _on_elife_row_selected(self):
        """Show constituent regions when eLife group row is selected."""
        row = self.elife_table.currentRow()
        if row < 0 or not hasattr(self, '_elife_data') or row >= len(self._elife_data):
            return

        group, count, desc, constituents = self._elife_data[row]

        # Parse constituents: "GRN=3721; MRN=1525; ..."
        self.elife_detail_table.setRowCount(0)
        if not constituents:
            return

        parts = [p.strip() for p in constituents.split(';') if '=' in p]
        self.elife_detail_table.setRowCount(len(parts))

        for i, part in enumerate(parts):
            if '=' in part:
                acr, cnt = part.split('=', 1)
                self.elife_detail_table.setItem(i, 0, QTableWidgetItem(acr.strip()))
                self.elife_detail_table.setItem(i, 1, QTableWidgetItem(""))  # Name not stored
                self.elife_detail_table.setItem(i, 2, QTableWidgetItem(cnt.strip()))

    def _load_available_brains_for_comparison(self):
        """Load list of brains that have region counts."""
        self.cross_brain_list.clear()

        try:
            from braintools.config import BRAINS_ROOT
            for mouse_dir in BRAINS_ROOT.iterdir():
                if not mouse_dir.is_dir():
                    continue
                for brain_dir in mouse_dir.iterdir():
                    if not brain_dir.is_dir():
                        continue
                    counts_csv = brain_dir / "6_Region_Analysis" / "cell_counts_by_region.csv"
                    if counts_csv.exists():
                        item = QListWidgetItem(brain_dir.name)
                        item.setData(Qt.UserRole, brain_dir)
                        self.cross_brain_list.addItem(item)
        except Exception as e:
            print(f"[SCI-Connectome] Error loading brains: {e}")

    def _run_cross_brain_comparison(self):
        """Compare selected brains."""
        selected = self.cross_brain_list.selectedItems()
        if len(selected) < 2:
            QMessageBox.warning(self, "Select Brains", "Select at least 2 brains to compare")
            return

        # Determine view mode
        use_elife = self.cross_view_combo.currentIndex() == 1

        # Load data from each brain
        import csv
        brain_data = {}
        all_regions = set()

        for item in selected:
            brain_path = item.data(Qt.UserRole)
            brain_name = item.text()

            if use_elife:
                csv_path = brain_path / "6_Region_Analysis" / "cell_counts_elife_grouped.csv"
                region_col = 'elife_group'
            else:
                csv_path = brain_path / "6_Region_Analysis" / "cell_counts_by_region.csv"
                region_col = 'region_acronym'

            if csv_path.exists():
                counts = {}
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        region = row.get(region_col, '')
                        try:
                            count = int(row.get('cell_count', 0))
                        except:
                            count = 0
                        if region and count > 0:
                            counts[region] = count
                            all_regions.add(region)
                brain_data[brain_name] = counts

        # Build comparison table
        brain_names = list(brain_data.keys())
        regions = sorted(all_regions)

        self.cross_brain_table.clear()
        self.cross_brain_table.setColumnCount(len(brain_names) + 1)
        self.cross_brain_table.setHorizontalHeaderLabels(["Region"] + brain_names)
        self.cross_brain_table.setRowCount(len(regions))

        for i, region in enumerate(regions):
            self.cross_brain_table.setItem(i, 0, QTableWidgetItem(region))
            for j, brain in enumerate(brain_names):
                count = brain_data[brain].get(region, 0)
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, count)
                self.cross_brain_table.setItem(i, j + 1, item)

    def _update_cross_brain_view(self):
        """Update cross-brain view when mode changes."""
        # Re-run comparison if brains are selected
        if self.cross_brain_list.selectedItems():
            self._run_cross_brain_comparison()

    def _load_run_history(self):
        """
        Load run history for current brain from region_counts_archive.csv.

        Shows the progression of region counts across optimization runs,
        allowing comparison of how counts change as parameters are tuned.
        """
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain", "Select a brain first")
            return

        # Load from archive CSV
        archive_csv = DATA_SUMMARY_DIR / "region_counts_archive.csv"
        if not archive_csv.exists():
            self.history_summary.setText("No archive found - run region counting first")
            return

        # Load and filter for this brain
        import csv
        runs = []
        all_regions = set()

        with open(archive_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            for row in reader:
                brain_name = row.get('brain', '')
                # Match brain name (may include parent folder)
                if self.current_brain.name in brain_name:
                    run_data = {
                        'date': row.get('run_date', row.get('archived_at', ''))[:16],
                        'total': int(row.get('total_cells', 0) or 0),
                        'regions': {}
                    }
                    # Extract region counts (skip hemisphere-specific columns)
                    for col in fieldnames:
                        if col.startswith('region_left_') or col.startswith('region_right_'):
                            continue
                        if col.startswith('region_'):
                            region = col.replace('region_', '')
                            try:
                                count = int(row.get(col, 0) or 0)
                                if count > 0:
                                    run_data['regions'][region] = count
                                    all_regions.add(region)
                            except:
                                pass
                    runs.append(run_data)

        if not runs:
            self.history_summary.setText(f"No archived runs found for {self.current_brain.name}")
            return

        # Sort by date
        runs.sort(key=lambda x: x['date'])

        # Store for view switching
        self._history_runs = runs
        self._history_all_regions = sorted(all_regions)

        # Update display based on current view mode
        self._update_history_view()

    def _update_history_view(self):
        """Update history view based on selected mode (Top 20, eLife Groups, All)."""
        if not hasattr(self, '_history_runs') or not self._history_runs:
            return

        runs = self._history_runs
        view_mode = self.history_view_combo.currentIndex()  # 0=Top 20, 1=eLife, 2=All

        if view_mode == 1:
            # eLife Groups view - aggregate to 25 groups
            self._show_history_elife_view(runs)
        else:
            # Allen regions view (Top 20 or All)
            self._show_history_allen_view(runs, top_n=20 if view_mode == 0 else None)

    def _show_history_allen_view(self, runs, top_n=None):
        """Show Allen atlas region history."""
        # Find top regions by total across all runs
        region_totals = {}
        for run in runs:
            for region, count in run['regions'].items():
                region_totals[region] = region_totals.get(region, 0) + count

        # Sort by total count
        sorted_regions = sorted(region_totals.keys(), key=lambda r: region_totals[r], reverse=True)
        if top_n:
            sorted_regions = sorted_regions[:top_n]

        # Build table: columns = runs, rows = regions
        self.history_table.clear()
        self.history_table.setColumnCount(len(runs) + 2)
        headers = ["Region", "Latest"] + [r['date'] for r in runs]
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(sorted_regions))

        for i, region in enumerate(sorted_regions):
            self.history_table.setItem(i, 0, QTableWidgetItem(region))

            # Latest value (from most recent run)
            latest = runs[-1]['regions'].get(region, 0)
            latest_item = QTableWidgetItem()
            latest_item.setData(Qt.DisplayRole, latest)
            latest_item.setFont(QFont("", -1, QFont.Bold))
            self.history_table.setItem(i, 1, latest_item)

            # Historical values
            for j, run in enumerate(runs):
                count = run['regions'].get(region, 0)
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, count)

                # Color code changes
                if j > 0:
                    prev_count = runs[j-1]['regions'].get(region, 0)
                    if count > prev_count:
                        item.setBackground(QColor(200, 255, 200))  # Light green
                    elif count < prev_count:
                        item.setBackground(QColor(255, 200, 200))  # Light red

                self.history_table.setItem(i, j + 2, item)

        # Summary
        totals = [r['total'] for r in runs]
        change = totals[-1] - totals[0] if len(totals) > 1 else 0
        direction = "+" if change >= 0 else ""
        self.history_summary.setText(
            f"{len(runs)} runs | Total: {totals[0]:,} -> {totals[-1]:,} ({direction}{change:,} cells)"
        )

    def _show_history_elife_view(self, runs):
        """Show eLife groups history with aggregation."""
        try:
            from elife_region_mapping import aggregate_to_elife, ELIFE_GROUPS
        except ImportError:
            self.history_summary.setText("eLife mapping not available")
            return

        # Aggregate each run to eLife groups
        aggregated_runs = []
        for run in runs:
            agg = aggregate_to_elife(run['regions'])
            aggregated_runs.append({
                'date': run['date'],
                'total': run['total'],
                'groups': agg
            })

        # Get all groups
        all_groups = sorted(ELIFE_GROUPS.keys())

        # Build table
        self.history_table.clear()
        self.history_table.setColumnCount(len(runs) + 2)
        headers = ["eLife Group", "Latest"] + [r['date'] for r in runs]
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(all_groups))

        for i, group in enumerate(all_groups):
            self.history_table.setItem(i, 0, QTableWidgetItem(group))

            # Latest value
            latest = aggregated_runs[-1]['groups'].get(group, {}).get('count', 0)
            latest_item = QTableWidgetItem()
            latest_item.setData(Qt.DisplayRole, latest)
            latest_item.setFont(QFont("", -1, QFont.Bold))
            self.history_table.setItem(i, 1, latest_item)

            # Historical values
            for j, agg_run in enumerate(aggregated_runs):
                count = agg_run['groups'].get(group, {}).get('count', 0)
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, count)

                # Color code changes
                if j > 0:
                    prev_count = aggregated_runs[j-1]['groups'].get(group, {}).get('count', 0)
                    if count > prev_count:
                        item.setBackground(QColor(200, 255, 200))
                    elif count < prev_count:
                        item.setBackground(QColor(255, 200, 200))

                self.history_table.setItem(i, j + 2, item)

        # Summary
        totals = [r['total'] for r in runs]
        change = totals[-1] - totals[0] if len(totals) > 1 else 0
        direction = "+" if change >= 0 else ""
        self.history_summary.setText(
            f"{len(runs)} runs | {len(all_groups)} eLife groups | "
            f"Total: {totals[0]:,} -> {totals[-1]:,} ({direction}{change:,} cells)"
        )

    def _on_class_selection_changed(self, index):
        """Not used - kept for compatibility."""
        pass

    def refresh_runs_table(self):
        """
        Refresh the table showing available classifications.

        Shows BOTH:
        - Tracker entries (from calibration_runs.csv)
        - Filesystem entries (cells.xml files found on disk)

        Sorted by date, most recent first.
        """
        print("[DEBUG] Button clicked: refresh_runs_table")
        self.runs_table.setRowCount(0)

        if not self.current_brain:
            return

        all_entries = []

        # 1. Get tracker entries
        if self.tracker:
            runs = self.tracker.search(
                brain=self.current_brain.name,
                exp_type="classification",
                limit=20
            )
            completed = [r for r in runs if r.get('status') == 'completed']

            for run in completed:
                # Get cell count - try multiple field names
                cells = run.get('class_cells_found') or run.get('class_cells_kept') or '?'
                model_path = run.get('class_model_path') or run.get('model_path') or ''
                model_name = Path(model_path).stem if model_path else 'unknown'
                output_path = run.get('output_path', '')

                all_entries.append({
                    'date': run.get('created_at', '')[:16],
                    'source': 'Tracker',
                    'cells': str(cells),
                    'path': model_name[:25],
                    'status': 'completed',
                    'run': run,
                    'cells_xml': Path(output_path) / 'cells.xml' if output_path else None,
                    'sort_key': run.get('created_at', '')
                })

        # 2. Get filesystem entries - scan for cells.xml files
        class_folder = self.current_brain / "5_Classified_Cells"
        if class_folder.exists():
            cells_xml = class_folder / "cells.xml"
            if cells_xml.exists():
                # Check if this is already in tracker entries
                already_tracked = any(
                    e.get('cells_xml') and e['cells_xml'].exists() and
                    e['cells_xml'].samefile(cells_xml)
                    for e in all_entries if e.get('cells_xml')
                )

                if not already_tracked:
                    # Count cells
                    try:
                        with open(cells_xml, 'r') as f:
                            n_cells = f.read().count('<Marker>')
                    except:
                        n_cells = '?'

                    # Get modification time
                    mtime = cells_xml.stat().st_mtime
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

                    all_entries.append({
                        'date': date_str,
                        'source': 'Filesystem',
                        'cells': str(n_cells),
                        'path': '5_Classified_Cells/',
                        'status': 'on disk',
                        'run': None,
                        'cells_xml': cells_xml,
                        'sort_key': datetime.fromtimestamp(mtime).isoformat()
                    })

        # Sort by date, most recent first
        all_entries.sort(key=lambda x: x['sort_key'], reverse=True)

        # Populate table
        self.runs_table.setRowCount(len(all_entries))

        for i, entry in enumerate(all_entries):
            self.runs_table.setItem(i, 0, QTableWidgetItem(entry['date']))
            self.runs_table.setItem(i, 1, QTableWidgetItem(entry['source']))
            self.runs_table.setItem(i, 2, QTableWidgetItem(entry['cells']))
            self.runs_table.setItem(i, 3, QTableWidgetItem(entry['path']))
            self.runs_table.setItem(i, 4, QTableWidgetItem(entry['status']))

            # Use button
            use_btn = QPushButton("Use")
            use_btn.clicked.connect(lambda checked, e=entry: self._select_classification_entry(e))
            self.runs_table.setCellWidget(i, 5, use_btn)

        # Show count
        if all_entries:
            self.class_selection_status.setText(f"Found {len(all_entries)} classification(s) - click 'Use' to select")
            self.class_selection_status.setStyleSheet("color: gray;")
        else:
            self.class_selection_status.setText("No classifications found")
            self.class_selection_status.setStyleSheet("color: orange;")

    def _select_classification_entry(self, entry):
        """Select a classification from the combined table."""
        self._selected_class_run = entry.get('run')
        self._selected_cells_xml = entry.get('cells_xml')

        date = entry.get('date', '?')
        cells = entry.get('cells', '?')
        source = entry.get('source', '?')

        self.class_selection_status.setText(f"Selected: {date} ({cells} cells) [{source}]")
        self.class_selection_status.setStyleSheet("color: blue; font-weight: bold;")

    def _select_specific_classification(self, run):
        """Select a specific classification run from the table."""
        self._selected_class_run = run
        cells = run.get('class_cells_kept', '?')
        date = run.get('created_at', '')[:16]
        self.class_selection_status.setText(f"Selected: {date} ({cells} cells) [manual selection]")
        self.class_selection_status.setStyleSheet("color: blue; font-weight: bold;")

    def load_run_results(self, exp_id):
        """
        Load results from a previous run into napari.

        Uses smart loading: if classification exists for this detection,
        loads the classified results (cells.xml + rejected.xml) instead.
        """
        # Use the unified loading method which handles smart loading
        self._load_historical_run(exp_id, cell_type='det_recent', prefer_classification=True)

    def generate_comparison_report(self):
        """
        Generate comprehensive pipeline comparison report.

        Shows detection settings, classification model, and region counts
        for all runs, compared against eLife reference.
        """
        print("[DEBUG] Button clicked: generate_comparison_report")
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain", "Select a brain first")
            return

        self.export_status.setText("Generating comparison report...")
        self.export_status.setStyleSheet("color: blue;")
        QApplication.processEvents()

        try:
            script = SCRIPTS_DIR / "util_pipeline_comparison.py"
            if not script.exists():
                QMessageBox.warning(self, "Error", "Pipeline comparison script not found")
                return

            # Generate both markdown and CSV
            cmd = [
                sys.executable,
                str(script),
                "--brain", self.current_brain.name,
                "--format", "both"
            ]

            print(f"[SCI-Connectome] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print(f"[SCI-Connectome] Report output:\n{result.stdout}")

                # Find the generated files
                reports_dir = DATA_SUMMARY_DIR / "reports"
                md_file = reports_dir / f"{self.current_brain.name}_comparison.md"
                csv_file = reports_dir / f"{self.current_brain.name}_comparison.csv"

                self.export_status.setText("Report generated!")
                self.export_status.setStyleSheet("color: green;")

                # Offer to open the report
                msg = "Pipeline comparison report generated!\n\n"
                if md_file.exists():
                    msg += f"Markdown: {md_file}\n"
                if csv_file.exists():
                    msg += f"CSV: {csv_file}\n"
                msg += "\nOpen the markdown report now?"

                reply = QMessageBox.question(
                    self, "Report Generated", msg,
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes and md_file.exists():
                    import os
                    if os.name == 'nt':
                        os.startfile(str(md_file))
                    else:
                        subprocess.run(['open', str(md_file)])
            else:
                self.export_status.setText("Report generation failed")
                self.export_status.setStyleSheet("color: red;")
                print(f"[SCI-Connectome] Report error:\n{result.stderr}")
                QMessageBox.warning(self, "Error", f"Report generation failed:\n{result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            self.export_status.setText("Report generation timed out")
            self.export_status.setStyleSheet("color: red;")
        except Exception as e:
            self.export_status.setText(f"Error: {str(e)[:30]}")
            self.export_status.setStyleSheet("color: red;")
            print(f"[SCI-Connectome] Report error: {e}")

    def export_results(self):
        """Export current results with sensible auto-generated names."""
        print("[DEBUG] Button clicked: export_results")
        if not self.current_brain:
            QMessageBox.warning(self, "Error", "Select a brain first")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        brain_name = self.current_brain.name

        # Create export directory
        export_dir = DATA_SUMMARY_DIR / "optimization_runs" / brain_name
        export_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []

        # Export screenshot
        try:
            screenshot_path = export_dir / f"{brain_name}_{timestamp}_screenshot.png"
            screenshot = self.viewer.screenshot()
            from PIL import Image
            img = Image.fromarray(screenshot)
            img.save(str(screenshot_path))
            exported_files.append(screenshot_path.name)
        except Exception as e:
            print(f"Screenshot export failed: {e}")

        # Export settings
        try:
            settings_path = export_dir / f"{brain_name}_{timestamp}_settings.json"
            settings = {
                'brain': brain_name,
                'timestamp': timestamp,
                'voxel_size_um': self.metadata.get('voxel_size_um', {}),
                'orientation': self.metadata.get('orientation', ''),
                'detection_params': {
                    'ball_xy_size': self.ball_xy.value(),
                    'ball_z_size': self.ball_z.value(),
                    'soma_diameter': self.soma_diameter.value(),
                    'threshold': self.threshold.value(),
                },
                'last_run_id': self.last_run_id,
                'last_run_cells': self.last_run_cells,
            }
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            exported_files.append(settings_path.name)
        except Exception as e:
            print(f"Settings export failed: {e}")

        # Export counts if available
        if self.last_run_id and self.tracker:
            try:
                run = self.tracker.get_experiment(self.last_run_id)
                if run:
                    counts_path = export_dir / f"{brain_name}_{timestamp}_run_info.json"
                    with open(counts_path, 'w') as f:
                        json.dump(dict(run), f, indent=2)
                    exported_files.append(counts_path.name)
            except Exception as e:
                print(f"Counts export failed: {e}")

        if exported_files:
            self.export_status.setText(f"Exported: {', '.join(exported_files)}")
            self.export_status.setStyleSheet("color: green;")

            # Log to session documenter
            if self.session_doc:
                for filename in exported_files:
                    self.session_doc.log_export(str(export_dir / filename))

            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(exported_files)} files to:\n{export_dir}"
            )

    def run_region_counting(self):
        """
        Run region counting for the current brain and update comparison table.

        Uses 6_count_regions.py which calls brainglobe-segmentation internally.
        Counts CLASSIFIED cells (from 5_Classified_Cells/cells.xml).
        Results are auto-tracked and compared to eLife reference.
        """
        print("[DEBUG] Button clicked: run_region_counting")
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain", "Select a brain first")
            return

        # Check for classification results (CLASSIFIED cells, not detection candidates!)
        cells_xml = self.current_brain / "5_Classified_Cells" / "cells.xml"
        if not cells_xml.exists():
            QMessageBox.warning(
                self, "No Classification",
                f"No classified cells found at:\n{cells_xml}\n\n"
                "Run classification first (Step 5) to separate true cells from false positives."
            )
            return

        # Count cells to show what will be processed
        try:
            with open(cells_xml, 'r') as f:
                n_cells = f.read().count('<Marker>')
        except:
            n_cells = '?'

        # Check for registration
        reg_folder = self.current_brain / "3_Registered_Atlas"
        if not reg_folder.exists() or not (reg_folder / "brainreg.json").exists():
            QMessageBox.warning(
                self, "No Registration",
                f"Registration not found at:\n{reg_folder}\n\n"
                "Run registration first (Step 3)."
            )
            return

        self.region_count_status.setText(f"Status: Running on {n_cells} classified cells...")
        self.region_count_status.setStyleSheet("color: blue;")
        self.run_regions_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Run 6_count_regions.py via subprocess
            import subprocess
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "6_count_regions.py"),
                "--brain", self.current_brain.name,
            ]

            print(f"[SCI-Connectome] Running region counting on CLASSIFIED cells ({n_cells})")
            print(f"[SCI-Connectome] Source: {cells_xml}")
            print(f"[SCI-Connectome] Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.region_count_status.setText("Status: Complete! Loading results...")
                self.region_count_status.setStyleSheet("color: green;")
                print(f"[SCI-Connectome] Region counting output:\n{result.stdout}")

                # Load results and update comparison table
                self._load_region_comparison()

                # Log to session
                if self.session_doc:
                    self.session_doc.log_region_count(self.current_brain.name)
            else:
                self.region_count_status.setText("Status: Failed - check console")
                self.region_count_status.setStyleSheet("color: red;")
                print(f"[SCI-Connectome] Region counting STDERR:\n{result.stderr}")
                QMessageBox.warning(
                    self, "Region Counting Failed",
                    f"Error running region counting:\n{result.stderr[:500]}"
                )

        except subprocess.TimeoutExpired:
            self.region_count_status.setText("Status: Timeout (>5min)")
            self.region_count_status.setStyleSheet("color: red;")
        except Exception as e:
            self.region_count_status.setText(f"Status: Error - {str(e)[:50]}")
            self.region_count_status.setStyleSheet("color: red;")
            print(f"[SCI-Connectome] Region counting error: {e}")
        finally:
            self.run_regions_btn.setEnabled(True)

    def _load_region_comparison(self):
        """Load region counts and populate comparison table with eLife reference."""
        if not self.current_brain:
            return

        output_csv = self.current_brain / "6_Region_Analysis" / "cell_counts_by_region.csv"
        if not output_csv.exists():
            self.region_count_status.setText("Status: No counts CSV found")
            return

        # Load counts
        import csv
        your_counts = {}
        try:
            with open(output_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    region = row.get('region') or row.get('Region') or row.get('region_name', '')
                    count = row.get('cell_count') or row.get('count') or row.get('cells', '0')
                    if region:
                        try:
                            your_counts[region] = int(count)
                        except ValueError:
                            pass
        except Exception as e:
            print(f"[SCI-Connectome] Error reading CSV: {e}")
            return

        # Map to eLife regions and populate table
        if REFERENCE_AVAILABLE:
            mapped_counts = map_to_elife_regions(your_counts)
            self._populate_comparison_table(mapped_counts)
            self.region_count_status.setText(
                f"Status: Loaded {len(your_counts)} regions, {sum(your_counts.values())} total cells"
            )
        else:
            # Just show raw counts
            self._populate_comparison_table_raw(your_counts)
            self.region_count_status.setText(
                f"Status: Loaded (eLife reference not available)"
            )

    def _populate_comparison_table(self, mapped_counts: dict):
        """Populate the comparison table with mapped counts vs eLife reference."""
        self.compare_table.setRowCount(0)

        # Sort: key recovery regions first, then by count difference
        regions = list(PUBLISHED_REFERENCE.keys())
        key_first = [r for r in regions if any(k in r for k in [
            "Pedunculopontine", "Red Nucleus", "Gigantocellular", "Corticospinal"
        ])]
        other = [r for r in regions if r not in key_first]

        for region in key_first + sorted(other):
            ref_mean, ref_std, n = PUBLISHED_REFERENCE[region]
            your_count = mapped_counts.get(region, 0)

            diff = your_count - ref_mean
            pct = (your_count / ref_mean * 100) if ref_mean > 0 else 0

            row = self.compare_table.rowCount()
            self.compare_table.insertRow(row)

            # Region name (highlight key regions)
            region_item = QTableWidgetItem(region)
            if any(k in region for k in ["Pedunculopontine", "Red Nucleus", "Gigantocellular"]):
                region_item.setForeground(QColor('#FF9800'))  # Orange for key
            self.compare_table.setItem(row, 0, region_item)

            # Your count
            self.compare_table.setItem(row, 1, QTableWidgetItem(str(your_count)))

            # Reference (mean ± std)
            self.compare_table.setItem(row, 2, QTableWidgetItem(f"{ref_mean} ± {ref_std}"))

            # Difference
            diff_item = QTableWidgetItem(f"{diff:+d}")
            if diff > 0:
                diff_item.setForeground(QColor('green'))
            elif diff < 0:
                diff_item.setForeground(QColor('red'))
            self.compare_table.setItem(row, 3, diff_item)

            # Match %
            pct_item = QTableWidgetItem(f"{pct:.0f}%")
            if 80 <= pct <= 120:
                pct_item.setForeground(QColor('green'))
            elif 50 <= pct <= 150:
                pct_item.setForeground(QColor('#FF9800'))
            else:
                pct_item.setForeground(QColor('red'))
            self.compare_table.setItem(row, 4, pct_item)

    def _populate_comparison_table_raw(self, counts: dict):
        """Populate table with raw counts (no eLife mapping)."""
        self.compare_table.setRowCount(0)

        for region, count in sorted(counts.items(), key=lambda x: -x[1]):
            row = self.compare_table.rowCount()
            self.compare_table.insertRow(row)
            self.compare_table.setItem(row, 0, QTableWidgetItem(region))
            self.compare_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.compare_table.setItem(row, 2, QTableWidgetItem("-"))
            self.compare_table.setItem(row, 3, QTableWidgetItem("-"))
            self.compare_table.setItem(row, 4, QTableWidgetItem("-"))

    def view_region_csv(self):
        """Open the region counts CSV in the default application."""
        print("[DEBUG] Button clicked: view_region_csv")
        if not self.current_brain:
            QMessageBox.warning(self, "No Brain", "Select a brain first")
            return

        output_csv = self.current_brain / "6_Region_Analysis" / "cell_counts_by_region.csv"
        if not output_csv.exists():
            QMessageBox.warning(
                self, "No Results",
                f"No region counts found at:\n{output_csv}\n\n"
                "Click 'Run & Compare' first."
            )
            return

        # Open with default application
        import os
        if os.name == 'nt':
            os.startfile(str(output_csv))
        else:
            import subprocess
            subprocess.run(['xdg-open', str(output_csv)])

    def end_session_and_report(self):
        """End the current session and generate a report."""
        print("[DEBUG] Button clicked: end_session_and_report")
        if not self.session_doc or not self.session_doc.is_active():
            QMessageBox.information(
                self, "No Active Session",
                "No active session to end. Select a brain to start a session."
            )
            return

        report_path = self.session_doc.end_session()
        if report_path and report_path.exists():
            # Open the report
            import os
            if os.name == 'nt':
                os.startfile(str(report_path))
            else:
                subprocess.run(['open', str(report_path)])

            QMessageBox.information(
                self, "Session Report Generated",
                f"Session report saved to:\n{report_path}"
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to generate session report")

    def add_session_note(self):
        """Add a note to the current session."""
        print("[DEBUG] Button clicked: add_session_note")
        if not self.session_doc or not self.session_doc.is_active():
            QMessageBox.warning(self, "No Session", "No active session. Select a brain first.")
            return

        from qtpy.QtWidgets import QInputDialog
        note, ok = QInputDialog.getText(
            self, "Add Session Note",
            "Enter a note about the current session:"
        )
        if ok and note:
            self.session_doc.add_note(note)
            QMessageBox.information(self, "Note Added", "Note added to session.")

    def _connect_param_tracking(self):
        """Connect detection parameter spinboxes to session tracking."""
        # Store initial values for change detection
        self._param_values = {}

        # List of (widget, param_name) pairs to track
        tracked_params = [
            (self.ball_xy, "ball_xy_size"),
            (self.ball_z, "ball_z_size"),
            (self.ball_overlap_fraction, "ball_overlap_fraction"),
            (self.soma_diameter, "soma_diameter"),
            (self.soma_spread_factor, "soma_spread_factor"),
            (self.threshold, "threshold"),
            (self.tiled_threshold, "tiled_threshold"),
            (self.log_sigma_size, "log_sigma_size"),
            (self.max_cluster_size, "max_cluster_size"),
        ]

        for widget, param_name in tracked_params:
            # Store initial value
            self._param_values[param_name] = widget.value()

            # Connect to change handler using lambda with default arg to capture param_name
            widget.valueChanged.connect(
                lambda val, pn=param_name, w=widget: self._on_param_changed(pn, val)
            )

    def _on_param_changed(self, param_name: str, new_value):
        """Handle detection parameter change - log to session."""
        if not hasattr(self, '_param_values'):
            return

        old_value = self._param_values.get(param_name)
        if old_value == new_value:
            return  # No actual change

        # Update stored value
        self._param_values[param_name] = new_value

        # Log to session documenter if active
        if self.session_doc and self.session_doc.is_active():
            if hasattr(self.session_doc, 'documenter') and self.session_doc.documenter:
                self.session_doc.documenter.log_param_change(param_name, old_value, new_value)
                self.session_doc.documenter.save_checkpoint()
                print(f"[SCI-Connectome] Tracked: {param_name} changed from {old_value} to {new_value}")

    def view_session_log(self):
        """Open the live session log file in the default text editor."""
        print("[DEBUG] Button clicked: view_session_log")
        if not self.session_doc or not self.session_doc.is_active():
            QMessageBox.warning(self, "No Session", "No active session. Select a brain first.")
            return

        # Get the live log path
        documenter = self.session_doc.documenter
        if not documenter or not documenter.session_id:
            QMessageBox.warning(self, "No Log", "No session log available yet.")
            return

        log_path = documenter.output_dir / f"LIVE_SESSION_{documenter.brain_name}_{documenter.session_id}.md"

        if not log_path.exists():
            # Force a save to create the file
            documenter.save_checkpoint()

        if log_path.exists():
            import os
            import subprocess
            import sys

            # Open the file with the default application
            if sys.platform == 'win32':
                os.startfile(str(log_path))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(log_path)])
            else:
                subprocess.run(['xdg-open', str(log_path)])

            print(f"[SCI-Connectome] Opened log file: {log_path}")
        else:
            QMessageBox.warning(self, "No Log", f"Log file not found:\n{log_path}")

    def closeEvent(self, event):
        """Handle widget close - generate session report if active."""
        if self.session_doc and self.session_doc.is_active():
            reply = QMessageBox.question(
                self, "End Session?",
                "Would you like to generate a session report before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.Yes:
                report_path = self.session_doc.end_session()
                if report_path:
                    QMessageBox.information(
                        self, "Report Generated",
                        f"Session report saved to:\n{report_path}"
                    )

        event.accept()
