"""ROI (Region of Interest) utilities for brain slice analysis.

Save/load ROI polygons as JSON, count cells within ROIs, format results.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def save_rois_json(output_path, rois, image_name, image_shape):
    """Save ROI polygons to JSON file.

    Args:
        output_path: Path to write JSON file.
        rois: List of dicts with 'name' (str) and 'vertices' (list of [y, x]).
        image_name: Filename of the source image.
        image_shape: (height, width) tuple of the source image.
    """
    data = {
        'version': 1,
        'image': str(image_name),
        'image_shape': list(image_shape[:2]),
        'created': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'rois': [],
    }
    for roi in rois:
        verts = roi['vertices']
        if isinstance(verts, np.ndarray):
            verts = verts.tolist()
        data['rois'].append({
            'name': roi['name'],
            'vertices': verts,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_rois_json(path):
    """Load ROI polygons from JSON file.

    Args:
        path: Path to .rois.json file.

    Returns:
        Dict with keys: 'version', 'image', 'image_shape', 'created', 'rois'.
        Each roi has 'name' and 'vertices' (list of [y, x] pairs).

    Raises:
        ValueError: If file is not valid ROI JSON.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if 'rois' not in data:
        raise ValueError(f"Not a valid ROI JSON file (missing 'rois' key): {path}")

    return data


def count_cells_in_rois(measurements, rois, image_shape):
    """Count cells within each ROI polygon.

    Args:
        measurements: DataFrame with centroid_y, centroid_x, is_positive columns.
            May also have 'classification' column for dual-channel mode.
        rois: List of roi dicts with 'name' and 'vertices'.
        image_shape: (height, width) tuple.

    Returns:
        (results, is_dual) tuple.
        results: List of dicts, one per ROI + a TOTAL row.
        is_dual: True if dual-channel classification detected.
    """
    from matplotlib.path import Path as MplPath

    is_dual = 'classification' in measurements.columns

    # Determine coordinate columns
    y_col = 'centroid_y_base' if 'centroid_y_base' in measurements.columns else 'centroid_y'
    x_col = 'centroid_x_base' if 'centroid_x_base' in measurements.columns else 'centroid_x'

    coords = measurements[[y_col, x_col]].values  # (N, 2) as (y, x)

    # Build ROI paths (vertices are stored as [y, x])
    roi_paths = []
    for roi in rois:
        verts = np.array(roi['vertices'])
        # MplPath wants (x, y) = (col, row), so swap
        xy = verts[:, ::-1]
        roi_paths.append(MplPath(xy))

    # Assign each cell to an ROI (first match wins)
    roi_membership = np.full(len(measurements), -1, dtype=int)
    for ri, rpath in enumerate(roi_paths):
        # points_to_test as (x, y) = (col, row)
        pts = coords[:, ::-1]
        inside = rpath.contains_points(pts)
        # Only assign cells not yet claimed
        unclaimed = roi_membership == -1
        roi_membership[unclaimed & inside] = ri

    # Count per ROI
    results = []
    for ri, roi in enumerate(rois):
        mask = roi_membership == ri
        roi_meas = measurements[mask]
        result = _count_subset(roi_meas, roi['name'], is_dual)
        results.append(result)

    # Outside
    outside_mask = roi_membership == -1
    outside_meas = measurements[outside_mask]
    results.append(_count_subset(outside_meas, 'Outside', is_dual))

    # Total
    results.append(_count_subset(measurements, 'TOTAL', is_dual))

    return results, is_dual


def _count_subset(df, name, is_dual):
    """Count positive/negative cells in a subset."""
    total = len(df)
    result = {'name': name, 'total': total}

    if is_dual:
        for cat in ['dual', 'red_only', 'green_only', 'neither']:
            result[cat] = int((df['classification'] == cat).sum())
    else:
        pos = int(df['is_positive'].sum()) if total > 0 else 0
        neg = total - pos
        frac = pos / total if total > 0 else 0.0
        result['positive'] = pos
        result['negative'] = neg
        result['fraction'] = round(frac, 4)

    return result


def results_to_dataframe(results, is_dual=False):
    """Convert results list to a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(results)


def format_results_table(results, is_dual=False):
    """Format results as an ASCII table string."""
    lines = []

    if is_dual:
        header = f"{'Region':<12} {'Total':>6} {'Dual':>6} {'Red':>6} {'Green':>6} {'Neither':>7}"
        lines.append(header)
        lines.append('-' * len(header))
        for r in results:
            lines.append(
                f"{r['name']:<12} {r['total']:>6} "
                f"{r.get('dual', 0):>6} {r.get('red_only', 0):>6} "
                f"{r.get('green_only', 0):>6} {r.get('neither', 0):>7}"
            )
    else:
        header = f"{'Region':<12} {'Total':>6} {'Pos':>6} {'Neg':>6} {'Fraction':>9}"
        lines.append(header)
        lines.append('-' * len(header))
        for r in results:
            frac_str = f"{r.get('fraction', 0):.1%}" if r['total'] > 0 else "N/A"
            lines.append(
                f"{r['name']:<12} {r['total']:>6} "
                f"{r.get('positive', 0):>6} {r.get('negative', 0):>6} "
                f"{frac_str:>9}"
            )

    return '\n'.join(lines)
