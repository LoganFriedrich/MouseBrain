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
    from mousebrain.plugin_2d.sliceatlas.core.roi import save_rois_json, load_rois_json

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
    existing_rois = None
    if output_path.exists():
        try:
            roi_data = load_rois_json(output_path)
            existing_rois = roi_data['rois']
            print(f"  Loaded {len(existing_rois)} existing ROI(s) -- edit or redraw")
        except ValueError:
            pass

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

    # Extract and save
    rois = _extract_rois_from_layers(left_layer, right_layer)

    if not rois:
        print("No ROIs drawn. Nothing saved.")
        return 0

    save_rois_json(output_path, rois, image_path.name, image_shape)
    print(f"Saved {len(rois)} ROI(s) to: {output_path}")

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
        roi_data = load_rois_json(rois_path)
    except ValueError as e:
        print(f"Error: {e}")
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

def _find_databases_root(data_folder):
    """Walk up from a data folder to find the Databases/ directory.

    Looks for 2_Connectome (or similar) parent that contains a Databases/ dir.
    Falls back to CONNECTOME_ROOT env var.
    """
    import os

    # Try env var first
    connectome_root = os.environ.get('CONNECTOME_ROOT')
    if connectome_root:
        db = Path(connectome_root) / 'Databases'
        if db.is_dir():
            return db

    # Walk up from data_folder looking for a sibling Databases/
    p = Path(data_folder).resolve()
    for _ in range(10):
        candidate = p / 'Databases'
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            break
        p = p.parent

    return None


def _parse_sample_name(sample_name):
    """Parse sample name into (mouse, base_region).

    E02_01_S13_DCNv2 -> ('E02_01', 'DCN')
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


def _get_roi_output_dirs(data_folder):
    """Return (exports_dir, figures_dir) under Databases/.

    Creates directories if needed. Returns (None, None) if Databases not found.
    """
    db = _find_databases_root(data_folder)
    if db is None:
        return None, None

    project = Path(data_folder).name  # e.g. 'ENCR'
    exports = db / 'exports' / f'{project}_ROI_Analysis'
    figures = db / 'figures' / f'{project}_ROI_Analysis'
    exports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return exports, figures


# ---------------------------------------------------------------------------
# Subcommand: batch
# ---------------------------------------------------------------------------

def _find_detection_dir(base_folder):
    """Find the detection results directory.

    Checks Databases/exports/{project}_Detection/ first,
    falls back to base_folder/batch_results/ for backward compat.
    """
    base = Path(base_folder)
    project = base.name  # e.g. 'ENCR'

    # Primary: Databases/exports/{project}_Detection/
    db = _find_databases_root(base_folder)
    if db is not None:
        det_dir = db / 'exports' / f'{project}_Detection'
        if det_dir.exists():
            return det_dir

    # Fallback: batch_results/ in the data folder
    fallback = base / 'batch_results'
    if fallback.exists():
        return fallback

    return None


def _discover_samples(base_folder, region=None):
    """Find all samples with measurements.csv and a matching ND2.

    Looks for detection results in Databases/exports/{project}_Detection/
    (falls back to base_folder/batch_results/), then searches the data
    folder for matching ND2 files.

    Args:
        base_folder: Path to dataset folder (e.g. .../ENCR/)
        region: Optional region suffix filter (e.g. 'DCN', 'VEST', 'RN')

    Returns:
        List of dicts with keys: sample, measurements_path, nd2_path, rois_path
        Sorted by sample name.
    """
    base = Path(base_folder)
    det_dir = _find_detection_dir(base_folder)

    if det_dir is None:
        print(f"Error: No detection results found for {base.name}")
        return []

    print(f"  Detection data: {det_dir}")

    samples = []
    for meas_dir in sorted(det_dir.iterdir()):
        if not meas_dir.is_dir():
            continue

        meas_file = meas_dir / 'measurements.csv'
        if not meas_file.exists():
            continue

        sample_name = meas_dir.name

        # Apply region filter
        if region:
            parts = sample_name.rsplit('_', 1)
            if len(parts) < 2:
                continue
            suffix = parts[1]
            if not suffix.upper().startswith(region.upper()):
                continue

        # Find matching ND2 in the data folder (not in detection results)
        nd2_path = None
        for nd2 in base.rglob(f'{sample_name}.nd2'):
            if str(det_dir) not in str(nd2):
                nd2_path = nd2
                break

        if nd2_path is None:
            continue

        # ROI JSON goes next to the ND2
        rois_path = nd2_path.with_suffix('.rois.json')

        samples.append({
            'sample': sample_name,
            'measurements_path': meas_file,
            'nd2_path': nd2_path,
            'rois_path': rois_path,
        })

    return samples


def cmd_batch(args):
    """Walk through images for a dataset, draw/edit ROIs, then count all.

    Unified workflow: opens every sample. If ROIs exist, they're loaded as
    editable polygons. If not, starts fresh. Edit or just Save & Next to keep.
    """
    import napari
    import pandas as pd
    from mousebrain.plugin_2d.sliceatlas.core.roi import (
        save_rois_json, load_rois_json, count_cells_in_rois,
        results_to_dataframe,
    )

    base_folder = Path(args.folder)
    if not base_folder.exists():
        print(f"Error: Folder not found: {base_folder}")
        return 1

    region = args.region
    samples = _discover_samples(base_folder, region)

    if not samples:
        region_msg = f" with region '{region}'" if region else ""
        print(f"No samples found in {base_folder}{region_msg}")
        return 1

    # Split into already-drawn and pending
    done = [s for s in samples if s['rois_path'].exists()]
    pending = [s for s in samples if not s['rois_path'].exists()]

    region_label = f" [{region}]" if region else ""
    print(f"Dataset: {base_folder.name}{region_label}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Already have ROIs: {len(done)}")
    print(f"  Need ROIs drawn: {len(pending)}")

    if args.count_only:
        if not done:
            print("\nNo samples have ROI files yet. Run without --count-only first.")
            return 1
        # Skip to counting
        samples_to_open = []
    elif args.skip_done:
        samples_to_open = pending
        if not samples_to_open:
            print("\nAll samples already have ROIs. Use --count-only or run without --skip-done to review.")
            # Fall through to counting
    else:
        # Open everything - existing ROIs are loaded as editable
        samples_to_open = samples

    if samples_to_open:
        n = len(samples_to_open)
        has_existing = sum(1 for s in samples_to_open if s['rois_path'].exists())
        print(f"\nOpening {n} images ({has_existing} with existing ROIs, {n - has_existing} new)")
        print("For each: draw/edit ROIs, then click 'Save & Next'.\n")

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

        # Load existing ROIs if any
        existing_rois = None
        if rois_path.exists():
            try:
                roi_data = load_rois_json(rois_path)
                existing_rois = roi_data['rois']
                print(f"  Existing ROIs: {len(existing_rois)} (edit or Save & Next to keep)")
            except ValueError:
                print(f"  Existing ROI file corrupt, starting fresh")

        # Open napari
        status = "EDIT" if existing_rois else "NEW"
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

        # Save whatever is in the layers
        rois = _extract_rois_from_layers(left_layer, right_layer)

        if not rois:
            print(f"  No ROIs for {s_name}. Skipped.")
            # Delete stale ROI file if it existed
            if rois_path.exists():
                rois_path.unlink()
            continue

        save_rois_json(rois_path, rois, nd2.name, image_shape)
        print(f"  Saved {len(rois)} ROI(s) -> {rois_path.name}")

    # Count phase: count all samples that have ROIs
    done = [s for s in samples if s['rois_path'].exists()]

    if not done:
        print("\nNo ROIs saved. Nothing to count.")
        return 0

    # Resolve output directories (Databases/ preferred)
    exports_dir, figures_dir = _get_roi_output_dirs(base_folder)
    if exports_dir is None:
        print("Warning: Databases/ not found. Saving to batch_results/ as fallback.")
        exports_dir = base_folder / 'batch_results'
        exports_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"COUNTING -- {len(done)} samples with ROIs")
    print(f"{'='*60}\n")

    all_rows = []

    for sample in done:
        s_name = sample['sample']
        meas_file = sample['measurements_path']
        rois_path = sample['rois_path']

        meas = pd.read_csv(meas_file)

        try:
            roi_data = load_rois_json(rois_path)
        except ValueError as e:
            print(f"  [{s_name}] Error loading ROIs: {e}")
            continue

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

        # Save per-sample ROI counts to Databases/exports/.../mouse/region/
        mouse, base_region = _parse_sample_name(s_name)
        sample_out = exports_dir / mouse / base_region
        sample_out.mkdir(parents=True, exist_ok=True)

        per_sample_df = results_to_dataframe(results, is_dual)
        per_sample_csv = sample_out / f'{s_name}_roi_counts.csv'
        per_sample_df.to_csv(per_sample_csv, index=False)

        # Registry provenance tracking (non-blocking)
        try:
            from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method
            registry = AnalysisRegistry(analysis_name="ENCR_ROI_Analysis")
            registry.register_roi_counts(
                sample=s_name,
                region=base_region,
                roi_results=results,
                method_params=get_approved_method(),
                source_files={
                    "nd2": str(sample['nd2_path']),
                    "roi_json": str(rois_path),
                },
            )
        except Exception as e:
            print(f"Registry warning: {e}")

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
        print(f"SUMMARY: {len(all_rows)} samples counted")
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
    batch_p.add_argument('folder', help='Dataset folder (e.g. .../ENCR/)')
    batch_p.add_argument('--region', default=None,
                         help='Filter by brain region suffix (e.g. DCN, VEST, RN, HYP, GRN)')
    batch_p.add_argument('--skip-done', action='store_true',
                         help='Skip samples that already have ROI files (for first-pass drawing)')
    batch_p.add_argument('--count-only', action='store_true',
                         help='Skip drawing, just count samples that already have ROI files')
    batch_p.add_argument('-o', '--output', default=None,
                         help='Output summary CSV path (default: batch_results/roi_summary.csv)')

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
