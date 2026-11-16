#!/usr/bin/env python3
"""
run_roi.py - CLI for ROI drawing, cell counting, and visualization.

Draw ROI polygons on brain slice images, count cells within ROIs,
and visualize saved ROIs.

Usage:
    brainslice-roi batch /path/to/ENCR --region DCN       # Draw/edit all DCN
    brainslice-roi batch /path/to/ENCR --region DCN --skip-done  # Only new images
    brainslice-roi batch /path/to/ENCR --count-only        # Just recount
    brainslice-roi draw image.nd2 -m measurements.csv      # Single image
    brainslice-roi count measurements.csv --rois rois.json  # Count only

Also accessible as:
    mousebrain roi batch/draw/count/view ...
"""

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_channels(image_path, red_ch=None, green_ch=None):
    """Load image and extract red/green channels.

    Returns (red_image, green_image, image_shape, metadata).
    """
    from mousebrain.plugin_2d.sliceatlas.core.io import (
        load_image, extract_channels, guess_channel_roles,
    )

    data, metadata = load_image(str(image_path))

    roles = guess_channel_roles(metadata)
    r_idx = red_ch if red_ch is not None else roles.get('nuclear', 0)
    g_idx = green_ch if green_ch is not None else roles.get('signal', 1)

    red_image, green_image = extract_channels(data, red_idx=r_idx, green_idx=g_idx)
    image_shape = red_image.shape[:2]

    return red_image, green_image, image_shape, metadata


def _add_measurement_points(viewer, measurements):
    """Add cell measurements as a colored Points layer to the viewer.

    Colors:
        Single mode: green for is_positive=True, gray for False
        Dual mode: yellow=dual, red=red_only, green=green_only, gray=neither
    """
    import numpy as np

    y_col = 'centroid_y_base' if 'centroid_y_base' in measurements.columns else 'centroid_y'
    x_col = 'centroid_x_base' if 'centroid_x_base' in measurements.columns else 'centroid_x'

    coords = measurements[[y_col, x_col]].values
    is_dual = 'classification' in measurements.columns

    if is_dual:
        color_map = {
            'dual': [1.0, 1.0, 0.0, 0.8],
            'red_only': [1.0, 0.3, 0.3, 0.8],
            'green_only': [0.3, 1.0, 0.3, 0.8],
            'neither': [0.5, 0.5, 0.5, 0.5],
        }
        colors = [color_map.get(c, [0.5, 0.5, 0.5, 0.5])
                  for c in measurements['classification']]
    else:
        colors = []
        for pos in measurements['is_positive']:
            if pos:
                colors.append([0.0, 1.0, 0.0, 0.8])
            else:
                colors.append([0.5, 0.5, 0.5, 0.4])

    viewer.add_points(
        coords,
        name="Cells",
        size=8,
        face_color=colors,
        border_color="transparent",
    )


# ---------------------------------------------------------------------------
# ROI Editor Widget
# ---------------------------------------------------------------------------

def _create_roi_editor(viewer, left_layer, right_layer):
    """Create a dock widget with ROI drawing controls.

    Buttons: Draw Left, Draw Right, Clear Left, Clear Right, Save & Next.
    Auto-advances from Left to Right after first polygon is completed.
    """
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    )
    from qtpy.QtCore import Qt

    widget = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    widget.setLayout(layout)

    # Status label
    status = QLabel("Click 'Draw Left' to start")
    status.setWordWrap(True)
    status.setAlignment(Qt.AlignCenter)
    status.setStyleSheet(
        "font-size: 13px; font-weight: bold; padding: 10px; "
        "background: #333; color: #fff; border-radius: 4px;"
    )
    layout.addWidget(status)

    # Left ROI controls
    left_row = QHBoxLayout()
    draw_left_btn = QPushButton("Draw Left")
    clear_left_btn = QPushButton("Clear Left")
    draw_left_btn.setStyleSheet(
        "background-color: #007ACC; color: white; font-weight: bold; "
        "padding: 8px; font-size: 12px;"
    )
    clear_left_btn.setStyleSheet("padding: 8px;")
    left_row.addWidget(draw_left_btn)
    left_row.addWidget(clear_left_btn)
    layout.addLayout(left_row)

    # Right ROI controls
    right_row = QHBoxLayout()
    draw_right_btn = QPushButton("Draw Right")
    clear_right_btn = QPushButton("Clear Right")
    draw_right_btn.setStyleSheet(
        "background-color: #CC6600; color: white; font-weight: bold; "
        "padding: 8px; font-size: 12px;"
    )
    clear_right_btn.setStyleSheet("padding: 8px;")
    right_row.addWidget(draw_right_btn)
    right_row.addWidget(clear_right_btn)
    layout.addLayout(right_row)

    # Spacer
    layout.addSpacing(10)

    # Save button
    save_btn = QPushButton("Save && Next")
    save_btn.setStyleSheet(
        "background-color: #28A745; color: white; font-weight: bold; "
        "padding: 12px; font-size: 14px;"
    )
    layout.addWidget(save_btn)

    # Hint
    hint = QLabel(
        "Tip: click to place vertices.\n"
        "Double-click to finish polygon.\n"
        "Press 3 for select mode (drag vertices).\n"
        "Press Delete to remove selected polygon."
    )
    hint.setWordWrap(True)
    hint.setStyleSheet("color: #aaa; font-size: 10px; padding-top: 8px;")
    layout.addWidget(hint)

    layout.addStretch()

    # --- Button actions ---

    def _update_status():
        """Update status label based on current layer state."""
        has_left = left_layer.nshapes > 0
        has_right = right_layer.nshapes > 0
        if has_left and has_right:
            status.setText("Both ROIs drawn!\nSave & Next when ready.")
            status.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 10px; "
                "background: #1a5c1a; color: #fff; border-radius: 4px;"
            )
        elif has_left and not has_right:
            status.setText("Left done -- now draw Right")
            status.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 10px; "
                "background: #664400; color: #fff; border-radius: 4px;"
            )
        elif not has_left and has_right:
            status.setText("Right done -- still need Left")
            status.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 10px; "
                "background: #333; color: #fff; border-radius: 4px;"
            )
        else:
            status.setText("Click 'Draw Left' to start")
            status.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 10px; "
                "background: #333; color: #fff; border-radius: 4px;"
            )

    def on_draw_left():
        viewer.layers.selection.active = left_layer
        left_layer.mode = 'add_polygon'
        status.setText("Drawing LEFT...\nClick vertices, double-click to finish")
        status.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 10px; "
            "background: #004466; color: #fff; border-radius: 4px;"
        )

    def on_draw_right():
        viewer.layers.selection.active = right_layer
        right_layer.mode = 'add_polygon'
        status.setText("Drawing RIGHT...\nClick vertices, double-click to finish")
        status.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 10px; "
            "background: #664400; color: #fff; border-radius: 4px;"
        )

    def on_clear_left():
        left_layer.data = []
        _update_status()

    def on_clear_right():
        right_layer.data = []
        _update_status()

    def on_save():
        viewer.close()

    draw_left_btn.clicked.connect(on_draw_left)
    draw_right_btn.clicked.connect(on_draw_right)
    clear_left_btn.clicked.connect(on_clear_left)
    clear_right_btn.clicked.connect(on_clear_right)
    save_btn.clicked.connect(on_save)

    # --- Auto-advance: after Left polygon done, switch to Right ---
    def _on_left_change(event=None):
        if left_layer.nshapes > 0 and right_layer.nshapes == 0:
            on_draw_right()
        else:
            _update_status()

    def _on_right_change(event=None):
        _update_status()

    left_layer.events.data.connect(_on_left_change)
    right_layer.events.data.connect(_on_right_change)

    return widget


def _setup_roi_editor(viewer, existing_rois=None):
    """Add Left/Right shape layers, load existing ROIs, add widget.

    Args:
        viewer: napari.Viewer instance (already has image layers)
        existing_rois: list of roi dicts with 'name' and 'vertices', or None

    Returns:
        (left_layer, right_layer) tuple
    """
    import numpy as np

    # Create the two layers
    left_layer = viewer.add_shapes(
        name="Left",
        edge_color="cyan",
        edge_width=3,
        face_color=[0, 0.8, 0.8, 0.08],
    )
    right_layer = viewer.add_shapes(
        name="Right",
        edge_color="orange",
        edge_width=3,
        face_color=[0.8, 0.4, 0, 0.08],
    )

    # Pre-load existing ROIs into the correct layers
    if existing_rois:
        for roi in existing_rois:
            verts = np.array(roi['vertices'])
            name = roi.get('name', '').lower()
            if 'left' in name:
                left_layer.add(verts, shape_type='polygon')
            elif 'right' in name:
                right_layer.add(verts, shape_type='polygon')
            else:
                # Unknown name - put in left if empty, else right
                if left_layer.nshapes == 0:
                    left_layer.add(verts, shape_type='polygon')
                else:
                    right_layer.add(verts, shape_type='polygon')

    # Add the control widget
    widget = _create_roi_editor(viewer, left_layer, right_layer)
    viewer.window.add_dock_widget(widget, name="ROI Controls", area="right")

    # Start in draw-left mode if no existing left ROI
    if left_layer.nshapes == 0:
        viewer.layers.selection.active = left_layer
        left_layer.mode = 'add_polygon'
    elif right_layer.nshapes == 0:
        viewer.layers.selection.active = right_layer
        right_layer.mode = 'add_polygon'

    return left_layer, right_layer


def _extract_rois_from_layers(left_layer, right_layer):
    """Read ROI data from the two shape layers.

    Returns list of roi dicts (may be 0, 1, or 2 entries).
    """
    import numpy as np

    rois = []
    if left_layer.nshapes > 0:
        # Take the last polygon (in case user drew multiple and forgot to clear)
        verts = np.array(left_layer.data[-1]).tolist()
        rois.append({'name': 'Left', 'vertices': verts})
    if right_layer.nshapes > 0:
        verts = np.array(right_layer.data[-1]).tolist()
        rois.append({'name': 'Right', 'vertices': verts})

    return rois


# ---------------------------------------------------------------------------
# Subcommand: draw
# ---------------------------------------------------------------------------

def cmd_draw(args):
    """Open napari to draw ROI polygons, save to JSON on close."""
    import napari
    import pandas as pd

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    print(f"Loading: {image_path.name}")
    t0 = time.time()
    red_image, green_image, image_shape, metadata = _load_image_channels(
        image_path, args.red_ch, args.green_ch
    )
    print(f"  Image shape: {image_shape[0]} x {image_shape[1]} ({time.time()-t0:.1f}s)")

    # Check for existing ROIs
    output_path = Path(args.output) if args.output else image_path.with_suffix('.rois.json')
    state, roi_data, err = _roi_file_state(output_path)
    existing_rois = roi_data['rois'] if state in ROI_PRELOADED else None
    if state == ROI_OK:
        print(f"  Loaded {len(existing_rois)} existing ROI(s) -- edit or redraw")
    elif state == ROI_NO_SHAPE:
        print(f"  Loaded {len(existing_rois)} existing ROI(s). The file has no "
              f"image size; closing the window saves them with it.")
    elif state == ROI_UNREADABLE:
        print(f"  [!] Existing ROI file is unreadable ({err}); it is kept.")

    # Create viewer
    viewer = napari.Viewer(title=f"ROI Draw -- {image_path.name}")

    viewer.add_image(red_image, name="Nuclear (red)", colormap="magenta",
                     blending="additive")
    viewer.add_image(green_image, name="Signal (green)", colormap="green",
                     blending="additive", visible=True)

    # Overlay measurements if provided
    if args.measurements:
        meas_path = Path(args.measurements)
        if not meas_path.exists():
            print(f"Warning: Measurements file not found: {meas_path}")
        else:
            print(f"  Loading measurements: {meas_path.name}")
            meas = pd.read_csv(meas_path)
            _add_measurement_points(viewer, meas)
            print(f"  Overlaid {len(meas)} cells")

    # Setup ROI editor with two layers + widget
    left_layer, right_layer = _setup_roi_editor(viewer, existing_rois)

    # Block until viewer closes
    napari.run()

    # Extract and save (never deleting a file; see _store_edited_rois)
    rois = _extract_rois_from_layers(left_layer, right_layer)
    print(_store_edited_rois(output_path, rois, image_path.name, image_shape, state))

    return 0


# ---------------------------------------------------------------------------
# Subcommand: count
# ---------------------------------------------------------------------------

def cmd_count(args):
    """Pure computation: count cells in ROIs from CSV + JSON."""
    import pandas as pd
    from mousebrain.plugin_2d.sliceatlas.core.roi import (
        load_rois_json, count_cells_in_rois, results_to_dataframe,
        format_results_table,
    )

    meas_path = Path(args.measurements)
    if not meas_path.exists():
        print(f"Error: Measurements file not found: {meas_path}")
        return 1

    rois_path = Path(args.rois)
    if not rois_path.exists():
        print(f"Error: ROI file not found: {rois_path}")
        return 1

    # Load measurements
    print(f"Loading measurements: {meas_path.name}")
    measurements = pd.read_csv(meas_path)

    required = {'centroid_y', 'centroid_x', 'is_positive'}
    missing = required - set(measurements.columns)
    if missing:
        # Check for _base variants
        if 'centroid_y' in missing and 'centroid_y_base' in measurements.columns:
            missing.discard('centroid_y')
        if 'centroid_x' in missing and 'centroid_x_base' in measurements.columns:
            missing.discard('centroid_x')
        if missing:
            print(f"Error: Measurements CSV missing required columns: {missing}")
            return 1

    # Load ROIs
    print(f"Loading ROIs: {rois_path.name}")
    try:
        roi_data = load_rois_json(rois_path, require_image_shape=True)
    except ValueError as e:
        print(f"Error: {e}")
        if _roi_file_state(rois_path)[0] == ROI_NO_SHAPE:
            # The polygons are fine; only the size is missing. Say how to add it.
            print("  Open the image once with brainslice-roi draw (or the sample in "
                  "brainslice-roi batch) and save: the ROIs are kept and saved "
                  "with the image size.")
        return 1

    rois = roi_data['rois']
    image_shape = tuple(roi_data['image_shape'])

    print(f"  Image: {roi_data.get('image', '(unknown)')}")
    print(f"  Image shape: {image_shape[0]} x {image_shape[1]}")
    print(f"  ROIs: {len(rois)}")
    print(f"  Cells: {len(measurements)}")

    # Count
    t0 = time.time()
    results, is_dual = count_cells_in_rois(measurements, rois, image_shape)
    elapsed = time.time() - t0

    # Print results
    mode_str = "dual-channel" if is_dual else "single-channel"
    print(f"\nResults ({mode_str}, {elapsed:.2f}s):\n")
    print(format_results_table(results, is_dual))

    # Save CSV
    output_path = Path(args.output) if args.output else meas_path.parent / f"{meas_path.stem}_roi_counts.csv"
    df = results_to_dataframe(results, is_dual)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: view
# ---------------------------------------------------------------------------

def cmd_view(args):
    """Open napari read-only with image + ROIs overlaid."""
    import napari
    import numpy as np
    import pandas as pd
    from mousebrain.plugin_2d.sliceatlas.core.roi import load_rois_json

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    rois_path = Path(args.rois)
    if not rois_path.exists():
        print(f"Error: ROI file not found: {rois_path}")
        return 1

    print(f"Loading: {image_path.name}")
    t0 = time.time()
    red_image, green_image, image_shape, metadata = _load_image_channels(
        image_path, args.red_ch, args.green_ch
    )
    print(f"  Image shape: {image_shape[0]} x {image_shape[1]} ({time.time()-t0:.1f}s)")

    # Load ROIs
    try:
        roi_data = load_rois_json(rois_path)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    rois = roi_data['rois']
    print(f"  ROIs: {len(rois)}")

    # Create viewer
    viewer = napari.Viewer(title=f"ROI View -- {image_path.name}")

    viewer.add_image(red_image, name="Nuclear (red)", colormap="magenta",
                     blending="additive")
    viewer.add_image(green_image, name="Signal (green)", colormap="green",
                     blending="additive", visible=True)

    # Add ROI shapes
    shapes_data = [np.array(roi['vertices']) for roi in rois]

    viewer.add_shapes(
        shapes_data,
        name="ROIs",
        shape_type='polygon',
        edge_color="yellow",
        edge_width=2,
        face_color="transparent",
        text={'string': [roi['name'] for roi in rois],
              'color': 'yellow', 'size': 12},
    )

    # Overlay measurements if provided
    if args.measurements:
        meas_path = Path(args.measurements)
        if not meas_path.exists():
            print(f"Warning: Measurements file not found: {meas_path}")
        else:
            print(f"  Loading measurements: {meas_path.name}")
            meas = pd.read_csv(meas_path)
            _add_measurement_points(viewer, meas)
            print(f"  Overlaid {len(meas)} cells")

    print(f"\nDisplaying {len(rois)} ROI(s). Close viewer to exit.")
    napari.run()

    return 0


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _registry_root():
    """Root of this tool's analysis registry (see mousebrain.analysis_registry).

    MOUSEBRAIN_REGISTRY_ROOT, else <pipeline root>/Registry. None when nothing
    is configured -- the caller then falls back to a folder beside the data and
    says so, rather than searching the disk for somewhere to write.
    """
    from mousebrain.analysis_registry import default_registry_root
    return default_registry_root()


def _project_code(base_folder, project=None):
    """The project code a dataset's results are filed under.

    --project when given, else the dataset folder's name. WHY the folder name
    is only the default: the detection batch files its results under
    <project>_Detection, and a dataset folder is not always named after its
    project. When they differ, --project says which project to look under.
    """
    return project or Path(base_folder).name


def _parse_sample_name(sample_name):
    """Parse sample name into (mouse, base_region).

    E01_01_S13_DCNv2 -> ('E01_01', 'DCN')
    """
    parts = sample_name.split('_')
    mouse = f'{parts[0]}_{parts[1]}' if len(parts) >= 2 else 'unknown'
    region_suffix = '_'.join(parts[3:]) if len(parts) >= 4 else 'unknown'
    base_region = region_suffix
    for r in ['DCN', 'GRN', 'VEST', 'RN', 'HYP']:
        if region_suffix.upper().startswith(r):
            base_region = r
            break
    return mouse, base_region


def _get_roi_output_dirs(data_folder, project=None):
    """Return (exports_dir, figures_dir) for this dataset under the registry root.

    exports/{project}_ROI_Analysis and figures/{project}_ROI_Analysis (project:
    see _project_code). exports/ is created;
    figures/ is returned as a path only -- nothing in this command writes
    figures, and an empty figures tree would look like an analysis that
    produced nothing. Returns (None, None) when no registry root is configured.
    """
    from mousebrain.plugin_2d.sliceatlas.core.config import roi_analysis_name

    root = _registry_root()
    if root is None:
        return None, None

    analysis = roi_analysis_name(_project_code(data_folder, project))
    exports = root / 'exports' / analysis
    figures = root / 'figures' / analysis
    exports.mkdir(parents=True, exist_ok=True)
    return exports, figures


# ---------------------------------------------------------------------------
# Subcommand: batch
# ---------------------------------------------------------------------------

def _detection_search_dirs(base_folder, project=None, detection_dir=None):
    """Every folder the detection results may be in, in the order searched.

    1. --detection-dir, when given (then nothing else is searched);
    2. <registry root>/exports/<project>_Detection/ -- where the detection
       batch registers each sample's measurements;
    3. the detection batch's default --output folder
       (core.config.DEFAULT_BATCH_OUTPUT), for results that never reached
       the registry;
    4. <dataset folder>/batch_results/, where older batches wrote.

    WHY 2-3 come from the same definitions the batch uses: the counter used to
    look under '<folder name>_Detection' while the batch registered under one
    fixed name, and its fallback folder was not where the batch writes; it
    found results only while the project and the folder happened to share
    that name.
    """
    from mousebrain.plugin_2d.sliceatlas.core.config import (
        DEFAULT_BATCH_OUTPUT, detection_analysis_name,
    )

    if detection_dir:
        return [Path(detection_dir)]
    dirs = []
    root = _registry_root()
    if root is not None:
        dirs.append(root / 'exports' /
                    detection_analysis_name(_project_code(base_folder, project)))
    if DEFAULT_BATCH_OUTPUT is not None:
        dirs.append(Path(DEFAULT_BATCH_OUTPUT))
    dirs.append(Path(base_folder) / 'batch_results')
    return dirs


def _find_detection_dir(base_folder, project=None, detection_dir=None):
    """Return (detection results folder or None, list of folders searched).

    The first folder of _detection_search_dirs that exists is used.
    """
    searched = _detection_search_dirs(base_folder, project, detection_dir)
    for d in searched:
        if d.is_dir():
            return d, searched
    return None, searched


def _discover_samples(base_folder, region=None, project=None, detection_dir=None):
    """Find all samples with measurements.csv and a matching ND2.

    Looks for detection results in the folders _detection_search_dirs names,
    then searches the data folder for matching ND2 files.

    Args:
        base_folder: Path to dataset folder (the project's image folder)
        region: Optional region suffix filter (e.g. 'DCN', 'VEST', 'RN')
        project: Project code (default: the dataset folder's name)
        detection_dir: Explicit detection results folder (overrides the search)

    Returns:
        (samples, left_out). samples: list of dicts with keys sample,
        measurements_path, nd2_path, rois_path, sorted by sample name.
        left_out: list of (sample name, reason) for every sample in the
        detection folder that cannot be drawn or counted. WHY it is returned:
        these samples used to be dropped without a word, and the totals then
        counted only what survived, so a missing ND2 looked like a smaller
        dataset. A sample excluded by --region is not "left out" -- the
        operator asked for that.
    """
    base = Path(base_folder)
    det_dir, searched = _find_detection_dir(base_folder, project, detection_dir)

    if det_dir is None:
        print(f"Error: No detection results found for "
              f"{_project_code(base_folder, project)}. Folders searched:")
        for d in searched:
            print(f"    {d}")
        print("  Run the detection batch first, or give --project / "
              "--detection-dir.")
        return [], []

    print(f"  Detection data: {det_dir}")

    samples = []
    left_out = []
    for meas_dir in sorted(det_dir.iterdir()):
        if not meas_dir.is_dir() or _is_not_a_sample_folder(meas_dir):
            continue
        sample_name = meas_dir.name

        # Apply region filter first: a sample outside the region was not
        # asked for, so it is neither counted nor reported as left out.
        if region:
            parts = sample_name.rsplit('_', 1)
            if len(parts) < 2:
                continue
            suffix = parts[1]
            if not suffix.upper().startswith(region.upper()):
                continue

        meas_file = meas_dir / 'measurements.csv'
        if not meas_file.exists():
            left_out.append((sample_name, f"no measurements.csv in {meas_dir}"))
            continue

        # Find matching ND2 in the data folder (not in detection results)
        nd2_path = None
        for nd2 in base.rglob(f'{sample_name}.nd2'):
            if str(det_dir) not in str(nd2):
                nd2_path = nd2
                break

        if nd2_path is None:
            left_out.append((sample_name, f"no {sample_name}.nd2 under {base}"))
            continue

        # ROI JSON goes next to the ND2
        rois_path = nd2_path.with_suffix('.rois.json')

        samples.append({
            'sample': sample_name,
            'measurements_path': meas_file,
            'nd2_path': nd2_path,
            'rois_path': rois_path,
        })

    return samples, left_out


# Name of the folder, inside the dataset folder, that the counts go to when no
# registry is configured (see cmd_batch).
NO_REGISTRY_OUTPUT = 'roi_counts'


def _is_not_a_sample_folder(folder):
    """True for a folder in the detection results that is not a sample.

    WHY: every other sub-folder without a measurements.csv is reported as a
    left-out sample, so anything that is known not to be a sample must be
    passed over here, or the "left out" count is wrong:
    - names starting with '_' are the registry's own bookkeeping
      (e.g. '_archived', where invalidated results are moved);
    - a folder holding ROI counts (<sample>_roi_counts.csv, directly or one
      level down) and no measurements.csv is this tool's own output, not
      detection results. Older versions wrote the counts to
      <dataset>/batch_results/<animal>/<region>/, which is also a folder
      searched for detection results.
    """
    if folder.name.startswith('_'):
        return True
    if (folder / 'measurements.csv').exists():
        return False
    try:
        return (any(folder.glob('*_roi_counts.csv'))
                or any(folder.glob('*/*_roi_counts.csv')))
    except OSError:
        return False


def _print_left_out(left_out):
    """Print the samples that will not be drawn or counted, with the reason."""
    if not left_out:
        return
    print(f"[!] {len(left_out)} sample(s) left out:")
    for name, reason in left_out:
        print(f"    {name}: {reason}")


# ROI file states (see _roi_file_state).
ROI_NONE = 'none'
ROI_OK = 'ok'
ROI_UNREADABLE = 'unreadable'
ROI_NO_SHAPE = 'no_shape'

# The one line every list and prompt uses for an ROI_NO_SHAPE file, so the
# person always reads the same fix.
NO_SHAPE_FIX = ("ROI file has no image size: open it once in brainslice-roi "
                "batch and click Save & Next")


def _roi_file_state(rois_path):
    """(state, roi data or None, error text or None) for one sample's ROI file.

    WHY the file is READ, not only checked for existence: an existing but
    unreadable file used to count as "already has ROIs", so --skip-done never
    opened it for redrawing and the counting step then skipped the sample --
    the sample silently had no counts. Reading it here lets the lists say
    which files need a person.

    ROI_NO_SHAPE is a file whose polygons read fine but which records no image
    size (what the napari widget's "Save ROIs" writes). WHY it is its own
    state and not "unreadable": the polygons are a person's drawing and are
    perfectly usable -- the editor must open with them loaded, and one Save &
    Next re-saves them with the image size, which makes the file countable.
    Calling it unreadable opened the editor empty and asked for a redraw.
    It is still not "done": counting cannot place polygons without the size.
    """
    from mousebrain.plugin_2d.sliceatlas.core.roi import load_rois_json

    rois_path = Path(rois_path)
    if not rois_path.exists():
        return ROI_NONE, None, None
    try:
        return ROI_OK, load_rois_json(rois_path, require_image_shape=True), None
    except (ValueError, OSError) as e:
        shape_error = str(e)
    try:
        return ROI_NO_SHAPE, load_rois_json(rois_path), shape_error
    except (ValueError, OSError) as e:
        return ROI_UNREADABLE, None, str(e)


# States whose polygons the editor opens with (the person edits or keeps them).
ROI_PRELOADED = (ROI_OK, ROI_NO_SHAPE)


def _set_aside(rois_path, why):
    """Rename an ROI file to <stem>.rois.<time>.<why>.bak beside it; return the new path.

    WHY a rename and never a delete: the ROI file is a person's drawing, the
    only copy of it. Whatever replaces or empties it, the old one stays on
    disk under a name nothing reads as current ROIs, so it can be put back by
    renaming it.
    """
    from datetime import datetime

    rois_path = Path(rois_path)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = rois_path.name[:-len('.json')] if rois_path.name.endswith('.json') else rois_path.name
    dest = rois_path.with_name(f"{stem}.{stamp}.{why}.bak")
    n = 1
    while dest.exists():
        dest = rois_path.with_name(f"{stem}.{stamp}_{n}.{why}.bak")
        n += 1
    rois_path.rename(dest)
    return dest


def _store_edited_rois(rois_path, rois, image_name, image_shape, previous_state):
    """Write what the editor holds when its window closes. Returns a status line.

    * ROIs drawn: they are saved. An unreadable file they replace is first
      set aside (*.unreadable.bak), never overwritten; so is a file without
      an image size (*.noshape.bak), which this save replaces with the same
      polygons plus the image size, making it countable.
    * No ROIs, and the sample had a readable ROI file (with or without the
      image size -- either way the editor showed its polygons): the person
      cleared them. The file is set aside (*.cleared.bak) -- the sample then counts
      as having no ROIs -- and the line names the backup.
    * No ROIs, and the old file was unreadable: it is left exactly as it is
      (it could not be shown, so it was not cleared by anyone).
    * No ROIs and no file: nothing to do.

    WHY: closing the window with nothing drawn used to DELETE the existing
    file -- after a Clear, but also when the file merely could not be read
    and the editor had opened empty.
    """
    from mousebrain.plugin_2d.sliceatlas.core.roi import save_rois_json

    rois_path = Path(rois_path)
    if rois:
        note = ""
        if previous_state == ROI_UNREADABLE and rois_path.exists():
            kept = _set_aside(rois_path, 'unreadable')
            note = f" (the unreadable file was kept as {kept.name})"
        elif previous_state == ROI_NO_SHAPE and rois_path.exists():
            kept = _set_aside(rois_path, 'noshape')
            note = (f" with the image size, so it can be counted (the file "
                    f"without it was kept as {kept.name})")
        save_rois_json(rois_path, rois, image_name, image_shape)
        return f"Saved {len(rois)} ROI(s) -> {rois_path.name}{note}"

    if previous_state in ROI_PRELOADED and rois_path.exists():
        kept = _set_aside(rois_path, 'cleared')
        return (f"No ROIs: the ROIs were cleared. The old file was kept as "
                f"{kept.name} (rename it back to {rois_path.name} to restore it).")
    if previous_state == ROI_UNREADABLE and rois_path.exists():
        return (f"No ROIs drawn. The unreadable file {rois_path.name} was left "
                f"as it is.")
    return "No ROIs drawn. Nothing saved."


def cmd_batch(args):
    """Walk through images for a dataset, draw/edit ROIs, then count all.

    Unified workflow: opens every sample. If ROIs exist, they're loaded as
    editable polygons. If not, starts fresh. Edit or just Save & Next to keep.
    """
    import pandas as pd
    from mousebrain.plugin_2d.sliceatlas.core.roi import (
        count_cells_in_rois, results_to_dataframe,
    )

    base_folder = Path(args.folder)
    if not base_folder.exists():
        print(f"Error: Folder not found: {base_folder}")
        return 1

    project = _project_code(base_folder, getattr(args, 'project', None))
    region = args.region
    samples, left_out = _discover_samples(
        base_folder, region, project=project,
        detection_dir=getattr(args, 'detection_dir', None))

    if not samples:
        region_msg = f" with region '{region}'" if region else ""
        _print_left_out(left_out)
        print(f"No samples found in {base_folder}{region_msg}")
        return 1

    # Split into already-drawn, pending and unreadable (see _roi_file_state)
    states = {s['sample']: _roi_file_state(s['rois_path']) for s in samples}
    done = [s for s in samples if states[s['sample']][0] == ROI_OK]
    pending = [s for s in samples if states[s['sample']][0] == ROI_NONE]
    unreadable = [s for s in samples if states[s['sample']][0] == ROI_UNREADABLE]
    no_shape = [s for s in samples if states[s['sample']][0] == ROI_NO_SHAPE]

    region_label = f" [{region}]" if region else ""
    print(f"Dataset: {base_folder.name}{region_label}  (project {project})")
    _print_left_out(left_out)
    print(f"  Total samples: {len(samples)}"
          + (f" (+{len(left_out)} left out, listed above)" if left_out else ""))
    print(f"  Already have ROIs: {len(done)}")
    print(f"  Need ROIs drawn: {len(pending)}")
    if unreadable:
        print(f"  [!] ROI file unreadable: {len(unreadable)} "
              f"(not counted until redrawn or repaired)")
        for s in unreadable:
            print(f"    {s['sample']}: {s['rois_path']} -- {states[s['sample']][2]}")
    if no_shape:
        print(f"  [!] {NO_SHAPE_FIX}: {len(no_shape)} "
              f"(not counted until then; their ROIs are kept and shown)")
        for s in no_shape:
            print(f"    {s['sample']}: {s['rois_path']}")

    if args.count_only:
        if not done:
            print("\nNo samples have countable ROI files yet. Run without --count-only first.")
            return 1
        # Skip to counting
        samples_to_open = []
    elif args.skip_done:
        # An unreadable file is not "done": it is opened so it can be redrawn.
        # Nor is one without an image size: it is opened with its ROIs loaded,
        # and Save & Next re-saves them with the size.
        samples_to_open = [s for s in samples if states[s['sample']][0] != ROI_OK]
        if not samples_to_open:
            print("\nAll samples already have ROIs. Use --count-only or run without --skip-done to review.")
            # Fall through to counting
    else:
        # Open everything - existing ROIs are loaded as editable
        samples_to_open = samples

    if samples_to_open:
        n = len(samples_to_open)
        has_existing = sum(1 for s in samples_to_open
                           if states[s['sample']][0] in ROI_PRELOADED)
        print(f"\nOpening {n} images ({has_existing} with existing ROIs, {n - has_existing} new)")
        print("For each: draw/edit ROIs, then click 'Save & Next'.\n")
        # Only the drawing step needs napari; --count-only runs without it.
        import napari

    # Draw/edit phase
    for i, sample in enumerate(samples_to_open):
        s_name = sample['sample']
        nd2 = sample['nd2_path']
        meas_file = sample['measurements_path']
        rois_path = sample['rois_path']

        print(f"{'='*60}")
        print(f"[{i+1}/{len(samples_to_open)}] {s_name}")
        print(f"{'='*60}")

        # Load image
        print(f"  Loading: {nd2.name}")
        t0 = time.time()
        try:
            red_image, green_image, image_shape, metadata = _load_image_channels(nd2)
        except Exception as e:
            print(f"  Error loading image: {e}")
            print(f"  Skipping {s_name}")
            continue
        print(f"  Image shape: {image_shape[0]} x {image_shape[1]} ({time.time()-t0:.1f}s)")

        # Load measurements
        meas = pd.read_csv(meas_file)
        print(f"  Cells: {len(meas)}")

        # Load existing ROIs if any (re-read: the file may have changed since
        # the lists above were made)
        state, roi_data, err = _roi_file_state(rois_path)
        existing_rois = roi_data['rois'] if state in ROI_PRELOADED else None
        if state == ROI_OK:
            print(f"  Existing ROIs: {len(existing_rois)} (edit or Save & Next to keep)")
        elif state == ROI_NO_SHAPE:
            print(f"  Existing ROIs: {len(existing_rois)}. The file has no image size:")
            print(f"      Save & Next saves them with it, so the sample can be counted.")
        elif state == ROI_UNREADABLE:
            print(f"  [!] Existing ROI file is unreadable ({err}).")
            print(f"      It is kept. Draw new ROIs to replace it (the old file is")
            print(f"      then renamed to *.unreadable.bak), or close to leave it as it is.")

        # Open napari
        status = {ROI_OK: "EDIT", ROI_NO_SHAPE: "EDIT - NO IMAGE SIZE",
                  ROI_UNREADABLE: "UNREADABLE FILE"}.get(state, "NEW")
        viewer = napari.Viewer(
            title=f"[{i+1}/{len(samples_to_open)}] {s_name} [{status}]"
        )

        viewer.add_image(red_image, name="Nuclear (red)", colormap="magenta",
                         blending="additive")
        viewer.add_image(green_image, name="Signal (green)", colormap="green",
                         blending="additive", visible=True)
        _add_measurement_points(viewer, meas)

        # Setup ROI editor
        left_layer, right_layer = _setup_roi_editor(viewer, existing_rois)

        napari.run()

        # Save whatever is in the layers (never deleting a file; see
        # _store_edited_rois)
        rois = _extract_rois_from_layers(left_layer, right_layer)
        print(f"  {_store_edited_rois(rois_path, rois, nd2.name, image_shape, state)}")

    # Count phase: count all samples that have a readable ROI file
    after = {s['sample']: _roi_file_state(s['rois_path']) for s in samples}
    done = [s for s in samples if after[s['sample']][0] == ROI_OK]
    not_counted = [(s['sample'], f"ROI file unreadable: {after[s['sample']][2]}")
                   for s in samples if after[s['sample']][0] == ROI_UNREADABLE]
    not_counted += [(s['sample'], NO_SHAPE_FIX)
                    for s in samples if after[s['sample']][0] == ROI_NO_SHAPE]

    if not done:
        print("\nNo ROIs saved. Nothing to count.")
        _print_left_out(left_out + not_counted)
        return 0

    # Resolve output directories (registry root preferred)
    exports_dir, figures_dir = _get_roi_output_dirs(base_folder, project)
    registry = None
    if exports_dir is None:
        # WHY not batch_results/ (where this used to write): that folder is
        # searched for DETECTION results, so counts written there were read
        # back on the next run as samples without measurements.
        exports_dir = base_folder / NO_REGISTRY_OUTPUT
        print("[!] registry: no registry root configured (set MOUSEBRAIN_REGISTRY_ROOT "
              f"or CONNECTOME_ROOT). Saving to {exports_dir} instead; nothing is registered.")
        exports_dir.mkdir(exist_ok=True)
    else:
        # One registry for the whole batch. A registry failure is reported and
        # counting continues -- the counts on disk are the primary output.
        try:
            from mousebrain.analysis_registry import AnalysisRegistry
            from mousebrain.plugin_2d.sliceatlas.core.config import roi_analysis_name
            # Follows the shared approved method unless the approved-method
            # file names this analysis (analysis_registry.approved_method_for;
            # docs/pipeline/ANALYSIS_REGISTRY.md).
            registry = AnalysisRegistry(analysis_name=roi_analysis_name(project),
                                        follows_approved_method=True)
        except Exception as e:
            print(f"[!] registry: {e}")

    print(f"\n{'='*60}")
    print(f"COUNTING -- {len(done)} samples with ROIs")
    print(f"{'='*60}\n")

    all_rows = []

    for sample in done:
        s_name = sample['sample']
        meas_file = sample['measurements_path']
        rois_path = sample['rois_path']

        meas = pd.read_csv(meas_file)

        roi_data = after[s_name][1]
        rois = roi_data['rois']
        image_shape = tuple(roi_data['image_shape'])

        results, is_dual = count_cells_in_rois(meas, rois, image_shape)

        # Get the TOTAL row
        total_row = results[-1]  # last row is TOTAL
        total_row_clean = {k: v for k, v in total_row.items() if not k.startswith('_')}
        total_row_clean['sample'] = s_name
        all_rows.append(total_row_clean)

        n_rois = len(results) - 1  # exclude TOTAL row
        if is_dual:
            print(f"  {s_name}: {n_rois} ROI(s), {total_row['total']} cells, "
                  f"{total_row.get('dual', 0)} dual+ ({total_row.get('frac_dual', 0)*100:.1f}%)")
        else:
            print(f"  {s_name}: {n_rois} ROI(s), {total_row['total']} cells, "
                  f"{total_row.get('positive', 0)} positive ({total_row.get('fraction', 0)*100:.1f}%)")

        # Save per-sample ROI counts to exports/{project}_ROI_Analysis/mouse/region/
        mouse, base_region = _parse_sample_name(s_name)
        sample_out = exports_dir / mouse / base_region
        sample_out.mkdir(parents=True, exist_ok=True)

        per_sample_df = results_to_dataframe(results, is_dual)
        per_sample_csv = sample_out / f'{s_name}_roi_counts.csv'
        per_sample_df.to_csv(per_sample_csv, index=False)

        # Registry provenance tracking (non-blocking): records the counts with
        # the method parameters and source files so an integrator can pull
        # them. count_cells_in_rois returns a LIST of rows keyed by 'name';
        # roi_results_from_rows turns that into the ROI-name -> counts mapping
        # the registry stores.
        if registry is not None:
            try:
                from mousebrain.analysis_registry import (
                    get_approved_method, roi_results_from_rows,
                )
                registry.register_roi_counts(
                    sample=s_name,
                    region=base_region,
                    roi_results=roi_results_from_rows(results),
                    # The method this analysis is judged by (see
                    # analysis_registry.approved_method_for).
                    method_params=(registry.approved_method()
                                   or get_approved_method(registry.registry_root)),
                    source_files={
                        "nd2": str(sample['nd2_path']),
                        "roi_json": str(rois_path),
                    },
                )
            except Exception as e:
                print(f"  [!] registry: {e}")

    # Combined summary
    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        # Move 'sample' column to front
        cols = ['sample'] + [c for c in summary_df.columns if c != 'sample']
        summary_df = summary_df[cols]

        region_suffix = f"_{region}" if region else ""
        output_path = Path(args.output) if args.output else (
            exports_dir / f'roi_summary{region_suffix}.csv'
        )
        summary_df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        _print_left_out(left_out + not_counted)
        print(f"SUMMARY: {len(all_rows)} samples counted"
              + (f", {len(left_out) + len(not_counted)} left out"
                 if (left_out or not_counted) else ""))
        print(f"Saved: {output_path}")
        print(f"{'='*60}")

    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='brainslice-roi',
        description='ROI drawing, cell counting, and visualization for 2D brain slices.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow (batch - recommended):
  brainslice-roi batch /path/to/ENCR --region DCN            # Draw/edit all DCN
  brainslice-roi batch /path/to/ENCR --region DCN --skip-done  # Only new images
  brainslice-roi batch /path/to/ENCR --count-only             # Recount all

Workflow (single image):
  brainslice-roi draw image.nd2 -m measurements.csv
  brainslice-roi count measurements.csv --rois image.rois.json
  brainslice-roi view image.nd2 --rois image.rois.json
""",
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # -- draw --
    draw_p = subparsers.add_parser('draw', help='Draw ROI polygons on an image')
    draw_p.add_argument('image', help='Path to ND2 or TIFF image file')
    draw_p.add_argument('-m', '--measurements', default=None,
                        help='Measurements CSV to overlay as points')
    draw_p.add_argument('-o', '--output', default=None,
                        help='Output JSON path (default: <image>.rois.json)')
    draw_p.add_argument('--red-ch', type=int, default=None,
                        help='Red channel index (auto-detected if omitted)')
    draw_p.add_argument('--green-ch', type=int, default=None,
                        help='Green channel index (auto-detected if omitted)')

    # -- count --
    count_p = subparsers.add_parser('count', help='Count cells in ROI polygons')
    count_p.add_argument('measurements', help='Measurements CSV file')
    count_p.add_argument('--rois', required=True, help='ROI JSON file')
    count_p.add_argument('-o', '--output', default=None,
                         help='Output CSV path (default: <measurements>_roi_counts.csv)')

    # -- view --
    view_p = subparsers.add_parser('view', help='View image with saved ROIs overlaid')
    view_p.add_argument('image', help='Path to ND2 or TIFF image file')
    view_p.add_argument('--rois', required=True, help='ROI JSON file')
    view_p.add_argument('-m', '--measurements', default=None,
                        help='Measurements CSV to overlay as points')
    view_p.add_argument('--red-ch', type=int, default=None,
                        help='Red channel index (auto-detected if omitted)')
    view_p.add_argument('--green-ch', type=int, default=None,
                        help='Green channel index (auto-detected if omitted)')

    # -- batch --
    batch_p = subparsers.add_parser(
        'batch',
        help='Walk through all images in a dataset, draw/edit ROIs, count cells',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Walk through all images in a dataset. Each image opens with a button panel:
  - Draw Left / Draw Right  (separate colored layers)
  - Clear Left / Clear Right (to redo)
  - Save & Next             (close and continue)

If ROIs already exist, they're loaded as editable polygons.
After all images, counts cells and saves a summary CSV.
""",
    )
    batch_p.add_argument('folder', help="Dataset folder: the project's image folder")
    batch_p.add_argument('--project', default=None,
                         help='Project code the detection batch filed the results under '
                              "(default: the dataset folder's name)")
    batch_p.add_argument('--detection-dir', default=None,
                         help='Folder holding one sub-folder per sample with '
                              'measurements.csv (default: searched; see the guide)')
    batch_p.add_argument('--region', default=None,
                         help='Filter by brain region suffix (e.g. DCN, VEST, RN, HYP, GRN)')
    batch_p.add_argument('--skip-done', action='store_true',
                         help='Skip samples that already have ROI files (for first-pass drawing)')
    batch_p.add_argument('--count-only', action='store_true',
                         help='Skip drawing, just count samples that already have ROI files')
    batch_p.add_argument('-o', '--output', default=None,
                         help='Output summary CSV path (default: roi_summary[_<region>].csv '
                              'in the ROI analysis folder of the registry, or in '
                              '<folder>/roi_counts/ when no registry is configured)')

    args = parser.parse_args()

    if args.command == 'draw':
        return cmd_draw(args)
    elif args.command == 'count':
        return cmd_count(args)
    elif args.command == 'view':
        return cmd_view(args)
    elif args.command == 'batch':
        return cmd_batch(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
