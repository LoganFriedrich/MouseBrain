#!/usr/bin/env python3
"""
prefilter.py - Canonical location for atlas pre-filter logic.

Pre-filter cell candidates using brain surface edge detection. Separates
detection candidates into:
  - Interior: kept for classification (deep inside brain)
  - Surface: removed (within surface_depth_um of the brain edge)
  - Extreme OOB: removed (candidates far beyond atlas bounds)

Four filtering methods:

1. ATLAS EROSION — blanket erosion of brain mask. Simple, aggressive.
2. PERIPHERAL WITHDRAWAL — only erode peripheral structures (those touching
   the brain boundary or ventricle walls), preserving deep interior regions.
3. BACKGROUND INTENSITY — uses the autofluorescence channel
   (downsampled.tiff). Dark background = non-tissue (surface, ventricle,
   damage). Candidates near dark regions are removed.
4. COMBINED — peripheral withdrawal + background intensity together.

Additionally: EXTREME OOB — candidates far beyond the atlas boundary.
Nearby OOB candidates are KEPT (spinal cord, ventral brainstem). Only
candidates far outside (default >500um) are removed.

The primary purpose is to CLEAN CLASSIFIER TRAINING DATA. Surface artifacts
in the "non-cell" training class poison the classifier. This filter removes
surface junk before it enters the training pipeline.

Usage (CLI):
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --surface-depth 150
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --method peripheral
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

from mousebrain.config import BRAINS_ROOT, MODELS_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "3.0.0"

FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def _brain_to_atlas_permutation(orientation: str) -> Tuple[int, int, int]:
    """Compute axis permutation from brain image space to registered atlas space.

    brainreg reorders brain axes to match the BrainGlobe atlas convention:
      dim 0 = AP (anterior-posterior)
      dim 1 = SI (superior-inferior / dorsal-ventral)
      dim 2 = LR (left-right / medial-lateral)

    The 3-letter orientation code tells which anatomical direction each brain
    axis points to (e.g. 'iar' = axis0 is inferior, axis1 is anterior,
    axis2 is right). From this we determine which brain axis maps to which
    atlas dimension.

    Returns (perm0, perm1, perm2) where atlas[i] indexes brain_axis[perm[i]].
    Example: orientation='iar' -> (1, 0, 2) meaning atlas = (brain_y, brain_z, brain_x).
    """
    letter_to_atlas_dim = {
        'a': 0, 'p': 0,  # anterior/posterior -> AP axis -> atlas dim 0
        's': 1, 'i': 1,  # superior/inferior  -> SI axis -> atlas dim 1
        'l': 2, 'r': 2,  # left/right         -> LR axis -> atlas dim 2
    }

    orientation = orientation.lower().strip()
    if len(orientation) != 3:
        print(f"  [!] Invalid orientation '{orientation}', assuming identity mapping")
        return (0, 1, 2)

    brain_to_atlas = []
    for i, ch in enumerate(orientation):
        atlas_dim = letter_to_atlas_dim.get(ch)
        if atlas_dim is None:
            print(f"  [!] Unknown orientation letter '{ch}', assuming identity")
            return (0, 1, 2)
        brain_to_atlas.append(atlas_dim)

    perm = [0, 0, 0]
    for brain_axis, atlas_dim in enumerate(brain_to_atlas):
        perm[atlas_dim] = brain_axis

    return tuple(perm)


# =============================================================================
# PREFILTER METHODS — each returns a boolean exclusion mask (True = remove)
# =============================================================================

PREFILTER_METHODS = {
    "atlas_erosion": "Atlas Erosion",
    "peripheral": "Peripheral Withdrawal",
    "background": "Background Intensity",
    "combined": "Combined (Peripheral + Background)",
}


def _atlas_erosion_mask(brain_mask, surface_depth_vox):
    """Method 1: Blanket erosion of brain mask."""
    from scipy import ndimage
    print(f"[{timestamp()}] Computing atlas erosion mask (eroding by {surface_depth_vox} voxels)...")
    eroded = ndimage.binary_erosion(brain_mask, iterations=surface_depth_vox)
    deep_count = int(eroded.sum())
    brain_count = int(brain_mask.sum())
    shell_count = brain_count - deep_count
    print(f"  Brain voxels: {brain_count:,}  Deep interior: {deep_count:,}  Surface shell: {shell_count:,}")
    exclusion = brain_mask & ~eroded
    del eroded
    return exclusion


def _find_surface_structure_ids(registered_annotation, brain_mask,
                                include_ventricle_walls=True,
                                include_fiber_tracts=True,
                                atlas=None):
    """Find structure IDs whose voxels touch the brain boundary (annotation=0)."""
    from scipy import ndimage
    print(f"[{timestamp()}] Finding peripheral structures...")
    exterior = ~brain_mask
    dilated_ext = ndimage.binary_dilation(exterior, iterations=1)
    boundary_voxels = brain_mask & dilated_ext
    del exterior, dilated_ext
    surface_ids = set(int(x) for x in np.unique(registered_annotation[boundary_voxels]) if x > 0)
    del boundary_voxels
    n_base = len(surface_ids)
    print(f"  Structures touching brain boundary: {n_base}")

    if include_ventricle_walls and atlas is not None:
        try:
            vs_descendants = atlas.get_structure_descendants('VS')
            ventricle_ids = set()
            for d in vs_descendants:
                if isinstance(d, dict):
                    ventricle_ids.add(d['id'])
                else:
                    ventricle_ids.add(int(d))
            ventricle_mask = np.isin(registered_annotation, list(ventricle_ids))
            if ventricle_mask.any():
                dilated_vent = ndimage.binary_dilation(ventricle_mask, iterations=1)
                vent_adjacent = dilated_vent & brain_mask & ~ventricle_mask
                vent_adj_ids = set(int(x) for x in np.unique(registered_annotation[vent_adjacent]) if x > 0)
                n_vent = len(vent_adj_ids - surface_ids)
                surface_ids |= vent_adj_ids
                surface_ids |= ventricle_ids
                print(f"  + {n_vent} structures adjacent to ventricles")
                del dilated_vent, vent_adjacent
            del ventricle_mask
        except Exception as e:
            print(f"  [!] Could not add ventricle-adjacent structures: {e}")

    if include_fiber_tracts and atlas is not None:
        try:
            ft_descendants = atlas.get_structure_descendants('fiber tracts')
            ft_ids = set()
            for d in ft_descendants:
                if isinstance(d, dict):
                    ft_ids.add(d['id'])
                else:
                    ft_ids.add(int(d))
            ft_at_surface = ft_ids & surface_ids
            ft_at_surface.add(1009)
            n_ft = len(ft_at_surface - surface_ids)
            surface_ids |= ft_at_surface
            if n_ft > 0:
                print(f"  + {n_ft} surface fiber tract structures")
        except Exception as e:
            print(f"  [!] Could not add fiber tract structures: {e}")

    print(f"  Total peripheral structure IDs: {len(surface_ids)}")
    return surface_ids


def _peripheral_withdrawal_mask(registered_annotation, brain_mask,
                                 surface_structure_ids, withdrawal_depth_vox,
                                 include_ventricle_walls=True, atlas=None):
    """Method 2: Only erode peripheral structures, not deep interiors."""
    from scipy import ndimage
    print(f"[{timestamp()}] Computing peripheral withdrawal mask (depth={withdrawal_depth_vox} voxels)...")
    eroded = ndimage.binary_erosion(brain_mask, iterations=withdrawal_depth_vox)
    erosion_shell = brain_mask & ~eroded
    del eroded
    peripheral_mask = np.isin(registered_annotation, list(surface_structure_ids))
    exclusion = erosion_shell & peripheral_mask
    del erosion_shell, peripheral_mask

    if include_ventricle_walls and atlas is not None:
        try:
            vs_descendants = atlas.get_structure_descendants('VS')
            ventricle_ids = set()
            for d in vs_descendants:
                if isinstance(d, dict):
                    ventricle_ids.add(d['id'])
                else:
                    ventricle_ids.add(int(d))
            ventricle_mask = np.isin(registered_annotation, list(ventricle_ids))
            if ventricle_mask.any():
                dilated = ndimage.binary_dilation(ventricle_mask, iterations=withdrawal_depth_vox)
                vent_wall = dilated & brain_mask & ~ventricle_mask
                n_vent_wall = int(vent_wall.sum())
                exclusion = exclusion | vent_wall
                print(f"  Ventricle wall zone: {n_vent_wall:,} voxels")
                del dilated, vent_wall
            del ventricle_mask
        except Exception as e:
            print(f"  [!] Ventricle wall detection failed: {e}")

    n_excluded = int(exclusion.sum())
    print(f"  Peripheral exclusion zone: {n_excluded:,} voxels")
    return exclusion


def _background_proximity_mask(registration_path, brain_mask, atlas_resolution,
                               filter_depth_vox, ann_shape, dark_percentile=5.0):
    """Method 3: Background intensity proximity detection."""
    from scipy import ndimage
    background_path = registration_path / "downsampled.tiff"
    if not background_path.exists():
        print(f"  [!] downsampled.tiff not found -- cannot use background method")
        return None
    print(f"[{timestamp()}] Loading background channel (autofluorescence)...")
    background = tifffile.imread(str(background_path))
    if background.shape != tuple(ann_shape):
        print(f"  [!] Background shape {background.shape} != atlas shape {tuple(ann_shape)}")
        del background
        return None
    brain_bg = background[brain_mask].ravel()
    nonzero = brain_bg[brain_bg > 0]
    if len(nonzero) == 0:
        del background
        return None
    dark_threshold = np.percentile(nonzero, dark_percentile)
    del brain_bg, nonzero
    print(f"  Non-tissue threshold: {dark_threshold:.0f} ({dark_percentile:.0f}th percentile of brain background)")
    filter_size = 2 * filter_depth_vox + 1
    print(f"[{timestamp()}] Computing local minimum ({filter_size}x{filter_size}x{filter_size} window = {filter_size * atlas_resolution:.0f}um)...")
    min_bg = ndimage.minimum_filter(background, size=filter_size)
    del background
    exclusion = min_bg < dark_threshold
    del min_bg
    n_excluded = int(exclusion.sum())
    print(f"  Background exclusion zone: {n_excluded:,} voxels")
    return exclusion


# =============================================================================
# CORE PRE-FILTER FUNCTION
# =============================================================================

def prefilter_candidates(
    candidates_xml: Path,
    registration_path: Path,
    method: str = "atlas_erosion",
    surface_depth_um: float = 100.0,
    withdrawal_depth_um: float = 100.0,
    include_ventricle_walls: bool = True,
    include_fiber_tracts: bool = True,
    filter_depth_um: float = 100.0,
    dark_percentile: float = 5.0,
    extreme_oob_um: float = 500.0,
    atlas_name: str = "allen_mouse_10um",
    tracing_type: str = "descending",
) -> dict:
    """
    Pre-filter cell candidates by surface edge + extreme-OOB criteria.

    Removes ALL candidates near the brain surface -- real labeled neurons
    are never at the tissue edge. Surface fluorescence is autofluorescence,
    tissue damage, or incomplete clearing artifacts.

    Four surface detection methods:
    - atlas_erosion (default): blanket binary erosion on annotation mask
    - peripheral: only erode peripheral structures, preserve deep interior
    - background: use autofluorescence channel intensity
    - combined: peripheral withdrawal + background intensity

    Unmapped candidates (region_id=0) are always KEPT.

    Args:
        candidates_xml: Path to detection candidates XML
        registration_path: Path to 3_Registered_Atlas folder
        method: Prefilter method (atlas_erosion, peripheral, background, combined)
        surface_depth_um: Depth in microns for atlas erosion (default: 100)
        withdrawal_depth_um: Depth in microns for peripheral withdrawal (default: 100)
        include_ventricle_walls: Include ventricle wall zones in peripheral methods
        include_fiber_tracts: Include surface fiber tracts in peripheral methods
        filter_depth_um: Depth in microns for background filter window (default: 100)
        dark_percentile: Percentile threshold for background method (default: 5.0)
        extreme_oob_um: Distance in microns beyond atlas for extreme OOB (default: 500)
        atlas_name: BrainGlobe atlas name (default: allen_mouse_10um)
        tracing_type: 'descending', 'ascending', or 'unknown' (logged, not used for filtering)

    Returns:
        dict with keys:
            interior_coords: list of (z, y, x) tuples to keep
            suspicious_coords: list of (z, y, x) tuples removed (surface + extreme OOB)
            suspicious_details: dict mapping (z,y,x) -> (region_id, acronym, category)
            category_counts: dict mapping category -> count
            stats: summary statistics dict
    """
    from brainglobe_atlasapi import BrainGlobeAtlas
    from scipy import ndimage

    print(f"[{timestamp()}] Loading atlas: {atlas_name}")
    atlas = BrainGlobeAtlas(atlas_name)
    atlas_resolution = atlas.resolution[0]  # typically 10um

    # Load registered atlas annotation
    annotation_path = registration_path / "registered_atlas.tiff"
    if not annotation_path.exists():
        annotation_path = registration_path / "annotation.tiff"
    if not annotation_path.exists():
        raise FileNotFoundError(f"No registered atlas found in {registration_path}")

    registered_annotation = tifffile.imread(str(annotation_path))
    print(f"  Annotation shape: {registered_annotation.shape}, dtype: {registered_annotation.dtype}")

    # Build brain mask from annotation for surface erosion.
    brain_mask = registered_annotation > 0
    print(f"  Brain mask (annotation>0): {int(brain_mask.sum()):,} voxels")

    # Read brain voxel sizes and orientation from brainreg.json
    brainreg_json = registration_path / "brainreg.json"
    brain_voxel_sizes = [4.0, 4.0, 4.0]
    orientation = "iar"

    if brainreg_json.exists():
        with open(brainreg_json) as f:
            brainreg_meta = json.load(f)
        voxel_sizes = brainreg_meta.get('voxel_sizes', ['4.0', '4.0', '4.0'])
        brain_voxel_sizes = [float(v) for v in voxel_sizes[:3]]
        orientation = brainreg_meta.get('orientation', 'iar')

    # Scale factors: brain coordinates -> atlas coordinates (per axis)
    brain_scales = [v / atlas_resolution for v in brain_voxel_sizes]

    # Axis permutation: brain image axes -> registered atlas axes
    axis_perm = _brain_to_atlas_permutation(orientation)

    # Convert depth thresholds to atlas voxels
    surface_depth_vox = max(1, int(round(surface_depth_um / atlas_resolution)))
    withdrawal_depth_vox = max(1, int(round(withdrawal_depth_um / atlas_resolution)))
    filter_depth_vox = max(1, int(round(filter_depth_um / atlas_resolution)))
    extreme_oob_vox = max(1, int(round(extreme_oob_um / atlas_resolution)))

    method_label = PREFILTER_METHODS.get(method, method)
    print(f"  Brain voxel sizes: {brain_voxel_sizes} um")
    print(f"  Atlas resolution: {atlas_resolution}um")
    print(f"  Brain scales: {brain_scales}")
    print(f"  Orientation: {orientation}  Axis permutation: {axis_perm}")
    print(f"  Prefilter method: {method} ({method_label})")
    print(f"  Surface depth: {surface_depth_um}um ({surface_depth_vox} atlas voxels)")
    print(f"  Withdrawal depth: {withdrawal_depth_um}um ({withdrawal_depth_vox} atlas voxels)")
    print(f"  Filter depth: {filter_depth_um}um ({filter_depth_vox} atlas voxels)")
    print(f"  Extreme OOB: {extreme_oob_um}um ({extreme_oob_vox} atlas voxels)")

    ann_shape = registered_annotation.shape
    print(f"  Atlas shape: {ann_shape}")

    # Build exclusion mask based on selected method
    exclusion_mask = None

    if method == "atlas_erosion":
        exclusion_mask = _atlas_erosion_mask(brain_mask, surface_depth_vox)

    elif method == "peripheral":
        surface_ids = _find_surface_structure_ids(
            registered_annotation, brain_mask,
            include_ventricle_walls=include_ventricle_walls,
            include_fiber_tracts=include_fiber_tracts,
            atlas=atlas,
        )
        exclusion_mask = _peripheral_withdrawal_mask(
            registered_annotation, brain_mask,
            surface_ids, withdrawal_depth_vox,
            include_ventricle_walls=include_ventricle_walls,
            atlas=atlas,
        )

    elif method == "background":
        exclusion_mask = _background_proximity_mask(
            registration_path, brain_mask, atlas_resolution,
            filter_depth_vox, ann_shape, dark_percentile=dark_percentile,
        )
        if exclusion_mask is None:
            print(f"  [!] Background method failed, falling back to atlas erosion")
            exclusion_mask = _atlas_erosion_mask(brain_mask, surface_depth_vox)
            method = "atlas_erosion"

    elif method == "combined":
        surface_ids = _find_surface_structure_ids(
            registered_annotation, brain_mask,
            include_ventricle_walls=include_ventricle_walls,
            include_fiber_tracts=include_fiber_tracts,
            atlas=atlas,
        )
        periph_mask = _peripheral_withdrawal_mask(
            registered_annotation, brain_mask,
            surface_ids, withdrawal_depth_vox,
            include_ventricle_walls=include_ventricle_walls,
            atlas=atlas,
        )
        bg_mask = _background_proximity_mask(
            registration_path, brain_mask, atlas_resolution,
            filter_depth_vox, ann_shape, dark_percentile=dark_percentile,
        )
        if bg_mask is not None:
            exclusion_mask = periph_mask | bg_mask
            del bg_mask
        else:
            print(f"  [!] Background unavailable, using peripheral only")
            exclusion_mask = periph_mask
        del periph_mask

    else:
        print(f"  [!] Unknown method '{method}', falling back to atlas_erosion")
        exclusion_mask = _atlas_erosion_mask(brain_mask, surface_depth_vox)
        method = "atlas_erosion"

    del brain_mask  # free memory

    # Parse candidates XML
    print(f"[{timestamp()}] Parsing candidates: {candidates_xml.name}")
    tree = ET.parse(str(candidates_xml))
    root = tree.getroot()

    all_coords = []
    for marker in root.iter('Marker'):
        x = int(marker.find('MarkerX').text)
        y = int(marker.find('MarkerY').text)
        z = int(marker.find('MarkerZ').text)
        all_coords.append((z, y, x))

    total = len(all_coords)
    print(f"  Total candidates: {total}")

    # Classify each candidate
    interior_coords = []
    suspicious_coords = []
    suspicious_details = {}
    out_of_bounds_kept = 0
    extreme_oob_removed = 0
    unmapped = 0
    surface_removed = 0
    category_counts = {}  # category -> count

    print(f"[{timestamp()}] Classifying candidates...")
    for z_brain, y_brain, x_brain in all_coords:
        # Scale each brain axis then permute to atlas space
        brain_scaled = (z_brain * brain_scales[0], y_brain * brain_scales[1], x_brain * brain_scales[2])
        a0 = int(brain_scaled[axis_perm[0]])
        a1 = int(brain_scaled[axis_perm[1]])
        a2 = int(brain_scaled[axis_perm[2]])
        idx = (a0, a1, a2)
        in_bounds = (0 <= a0 < ann_shape[0] and 0 <= a1 < ann_shape[1] and 0 <= a2 < ann_shape[2])

        if not in_bounds:
            # How far beyond the atlas boundary?
            oob_dist = max(
                max(0, a0 - ann_shape[0] + 1), max(0, -a0),
                max(0, a1 - ann_shape[1] + 1), max(0, -a1),
                max(0, a2 - ann_shape[2] + 1), max(0, -a2),
            )

            if oob_dist > extreme_oob_vox:
                # Extreme OOB -- way too far from the atlas to be real
                extreme_oob_removed += 1
                suspicious_coords.append((z_brain, y_brain, x_brain))
                suspicious_details[(z_brain, y_brain, x_brain)] = (0, 'OOB', 'extreme_oob')
                category_counts['extreme_oob'] = category_counts.get('extreme_oob', 0) + 1
            else:
                # Nearby OOB -- likely spinal cord / ventral brainstem, keep
                out_of_bounds_kept += 1
                interior_coords.append((z_brain, y_brain, x_brain))
            continue

        # Surface filter FIRST: remove ANY candidate in the exclusion zone.
        if exclusion_mask[idx]:
            surface_removed += 1
            suspicious_coords.append((z_brain, y_brain, x_brain))
            region_id = int(registered_annotation[idx])
            if region_id > 0 and region_id in atlas.structures:
                acronym = atlas.structures[region_id]['acronym']
            else:
                acronym = 'unmapped' if region_id == 0 else str(region_id)
            suspicious_details[(z_brain, y_brain, x_brain)] = (region_id, acronym, 'surface')
            category_counts['surface'] = category_counts.get('surface', 0) + 1
            continue

        region_id = int(registered_annotation[idx])

        # Unmapped (region_id=0) NOT near surface -- atlas boundary gaps, keep
        if region_id == 0:
            unmapped += 1
            interior_coords.append((z_brain, y_brain, x_brain))
            continue

        # Deep interior -- keep
        interior_coords.append((z_brain, y_brain, x_brain))

    del exclusion_mask, registered_annotation  # free memory

    # Build stats
    stats = {
        'total': total,
        'interior': len(interior_coords),
        'suspicious': len(suspicious_coords),
        'out_of_bounds_kept': out_of_bounds_kept,
        'extreme_oob_removed': extreme_oob_removed,
        'unmapped': unmapped,
        'surface_removed': surface_removed,
        'method_requested': method,
        'tracing_type': tracing_type,
        'surface_depth_um': surface_depth_um,
        'withdrawal_depth_um': withdrawal_depth_um,
        'filter_depth_um': filter_depth_um,
        'dark_percentile': dark_percentile,
        'extreme_oob_um': extreme_oob_um,
        'brain_scales': brain_scales,
        'axis_perm': list(axis_perm),
        'orientation': orientation,
        'atlas_name': atlas_name,
        'annotation_shape': list(ann_shape),
        'include_ventricle_walls': include_ventricle_walls,
        'include_fiber_tracts': include_fiber_tracts,
    }

    # Print summary
    pct_i = len(interior_coords) / total * 100 if total else 0
    pct_s = len(suspicious_coords) / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"PRE-FILTER RESULTS  (method: {method})")
    print(f"{'='*60}")
    print(f"  Total candidates:         {total:>8,}")
    print(f"  Interior (keep):          {len(interior_coords):>8,}  ({pct_i:.1f}%)")
    print(f"    nearby OOB (kept):      {out_of_bounds_kept:>8,}")
    print(f"    unmapped (kept):        {unmapped:>8,}")
    print(f"  Removed:                  {len(suspicious_coords):>8,}  ({pct_s:.1f}%)")
    print(f"    surface edge:           {surface_removed:>8,}")
    print(f"    extreme OOB:            {extreme_oob_removed:>8,}")
    print(f"{'='*60}")

    return {
        'interior_coords': interior_coords,
        'suspicious_coords': suspicious_coords,
        'suspicious_details': suspicious_details,
        'category_counts': category_counts,
        'stats': stats,
    }


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================

def _coords_to_xml(coords: List[Tuple[int, int, int]], output_path: Path):
    """Save coordinate list as CellCounter-compatible XML."""
    root = ET.Element("CellCounter_Marker_File")
    image_props = ET.SubElement(root, "Image_Properties")
    ET.SubElement(image_props, "Image_Filename").text = "prefiltered"

    marker_data = ET.SubElement(root, "Marker_Data")
    marker_type = ET.SubElement(marker_data, "Marker_Type")
    ET.SubElement(marker_type, "Type").text = "1"

    for z, y, x in coords:
        marker = ET.SubElement(marker_type, "Marker")
        ET.SubElement(marker, "MarkerX").text = str(x)
        ET.SubElement(marker, "MarkerY").text = str(y)
        ET.SubElement(marker, "MarkerZ").text = str(z)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), xml_declaration=True, encoding='unicode')
    print(f"  Saved {len(coords)} candidates to {output_path.name}")


def save_prefilter_results(
    result: dict,
    output_dir: Path,
    brain_name: str,
    source_xml: str = "",
) -> dict:
    """
    Save pre-filter results as XML files and a JSON report.

    Creates:
        output_dir/interior_candidates.xml  (keep for classification)
        output_dir/suspicious_candidates.xml (removed by filter)
        output_dir/prefilter_report.json

    Returns dict of saved paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    print(f"\n[{timestamp()}] Saving pre-filter results to {output_dir}")

    # Interior candidates (for classification)
    interior_path = output_dir / "interior_candidates.xml"
    _coords_to_xml(result['interior_coords'], interior_path)
    saved['interior'] = str(interior_path)

    # Suspicious candidates (removed by filter)
    if result['suspicious_coords']:
        suspicious_path = output_dir / "suspicious_candidates.xml"
        _coords_to_xml(result['suspicious_coords'], suspicious_path)
        saved['suspicious'] = str(suspicious_path)

    # JSON report
    report = {
        'brain': brain_name,
        'source_xml': source_xml,
        'timestamp': datetime.now().isoformat(),
        'script_version': SCRIPT_VERSION,
        **result['stats'],
        'category_counts': result.get('category_counts', {}),
        'saved_files': saved,
    }

    report_path = output_dir / "prefilter_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    saved['report'] = str(report_path)

    print(f"  Report: {report_path.name}")
    return saved


# =============================================================================
# BRAIN PATH UTILITIES
# =============================================================================

def _is_brain_pipeline(path: Path) -> bool:
    """Check if a directory looks like a brain pipeline folder."""
    pipeline_markers = [
        "0_Raw_IMS", "1_Extracted_Full", "2_Cropped_For_Registration",
        "3_Registered_Atlas", "4_Cell_Candidates",
    ]
    return any((path / marker).exists() for marker in pipeline_markers)


def find_brain_path(brain_name: str) -> Optional[Path]:
    """Find brain folder by name (searches subdirectories).

    Prefers directories that contain pipeline folders (0_Raw_IMS, etc.)
    over directories that just happen to match the name.
    """
    # Check mouse_id/brain_id pattern first (most common)
    for subdir in BRAINS_ROOT.iterdir():
        if subdir.is_dir():
            candidate = subdir / brain_name
            if candidate.exists() and _is_brain_pipeline(candidate):
                return candidate

    # Direct match (if it's a real pipeline folder)
    direct = BRAINS_ROOT / brain_name
    if direct.exists() and _is_brain_pipeline(direct):
        return direct

    # Fallback: any match with pipeline markers
    for path in BRAINS_ROOT.rglob(brain_name):
        if path.is_dir() and _is_brain_pipeline(path):
            return path

    # Last resort: direct match without validation
    if direct.exists():
        return direct

    return None


def find_latest_candidates(brain_path: Path) -> Optional[Path]:
    """Find the most recent detection candidates XML for a brain."""
    det_dir = brain_path / FOLDER_DETECTION
    if not det_dir.exists():
        return None

    # Look for Detected_*.xml files (dated format), sorted newest first
    xml_files = sorted(det_dir.glob("Detected_*.xml"), reverse=True)
    if xml_files:
        return xml_files[0]

    # Look for detected_cells.xml (cellfinder default name)
    default = det_dir / "detected_cells.xml"
    if default.exists():
        return default

    # Also check subdirectories (detection run folders)
    for subdir in sorted(det_dir.iterdir(), reverse=True):
        if subdir.is_dir():
            for xml_file in sorted(subdir.glob("*.xml"), reverse=True):
                return xml_file

    return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter cell candidates using suspicious region surface mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter candidates for a brain (auto-finds latest detection)
  python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4

  # Adjust surface depth (default 100um) and extreme OOB (default 500um)
  python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --surface-depth 150

  # Use peripheral withdrawal method
  python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --method peripheral

  # Specify detection XML explicitly
  python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 \\
      --candidates 4_Cell_Candidates/Detected_20260109_143022.xml
"""
    )
    parser.add_argument('--brain', required=True, help='Brain folder name')
    parser.add_argument('--candidates', type=Path,
                        help='Path to candidates XML (default: latest in 4_Cell_Candidates)')
    parser.add_argument('--tracing-type', default='descending',
                        choices=['descending', 'ascending', 'unknown'],
                        help='Tracing type for suspicious region filtering (default: descending)')
    parser.add_argument('--method', default='atlas_erosion',
                        choices=list(PREFILTER_METHODS.keys()),
                        help='Prefilter method (default: atlas_erosion)')
    parser.add_argument('--surface-depth', type=float, default=100.0,
                        help='Surface shell depth in microns for atlas erosion (default: 100)')
    parser.add_argument('--withdrawal-depth', type=float, default=100.0,
                        help='Withdrawal depth in microns for peripheral method (default: 100)')
    parser.add_argument('--no-ventricle-walls', action='store_true',
                        help='Exclude ventricle wall zones from peripheral methods')
    parser.add_argument('--no-fiber-tracts', action='store_true',
                        help='Exclude surface fiber tracts from peripheral methods')
    parser.add_argument('--filter-depth', type=float, default=100.0,
                        help='Filter window depth in microns for background method (default: 100)')
    parser.add_argument('--dark-percentile', type=float, default=5.0,
                        help='Dark percentile threshold for background method (default: 5.0)')
    parser.add_argument('--extreme-oob', type=float, default=500.0,
                        help='Distance in microns beyond atlas for extreme OOB removal (default: 500)')
    parser.add_argument('--output', type=Path,
                        help='Output directory (default: 4_Cell_Candidates/prefiltered_{timestamp})')
    parser.add_argument('--atlas', default='allen_mouse_10um',
                        help='BrainGlobe atlas name (default: allen_mouse_10um)')
    parser.add_argument('--view', action='store_true',
                        help='Open napari to visualize results after filtering')

    args = parser.parse_args()

    print("=" * 60)
    print("Atlas Pre-Filter for Cell Candidates")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)

    # Find brain path
    brain_path = find_brain_path(args.brain)
    if brain_path is None:
        print(f"ERROR: Brain not found: {args.brain}")
        print(f"Searched in: {BRAINS_ROOT}")
        sys.exit(1)
    print(f"\nBrain: {brain_path}")

    # Find registration folder
    registration_path = brain_path / FOLDER_REGISTRATION
    if not registration_path.exists():
        print(f"ERROR: Registration folder not found: {registration_path}")
        print("Run registration (step 3) first.")
        sys.exit(1)

    # Find candidates XML
    if args.candidates:
        candidates_xml = args.candidates
        if not candidates_xml.is_absolute():
            candidates_xml = brain_path / candidates_xml
    else:
        candidates_xml = find_latest_candidates(brain_path)

    if candidates_xml is None or not candidates_xml.exists():
        print(f"ERROR: No candidates XML found for {args.brain}")
        print(f"Run detection (step 4) first, or specify --candidates explicitly.")
        sys.exit(1)
    print(f"Candidates: {candidates_xml}")

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = brain_path / FOLDER_DETECTION / f"prefiltered_{ts}"

    # Run pre-filter
    start_time = time.time()
    result = prefilter_candidates(
        candidates_xml=candidates_xml,
        registration_path=registration_path,
        method=args.method,
        surface_depth_um=args.surface_depth,
        withdrawal_depth_um=args.withdrawal_depth,
        include_ventricle_walls=not args.no_ventricle_walls,
        include_fiber_tracts=not args.no_fiber_tracts,
        filter_depth_um=args.filter_depth,
        dark_percentile=args.dark_percentile,
        extreme_oob_um=args.extreme_oob,
        atlas_name=args.atlas,
        tracing_type=args.tracing_type,
    )
    duration = time.time() - start_time

    # Save results
    brain_name = brain_path.name
    saved = save_prefilter_results(result, output_dir, brain_name, str(candidates_xml))

    # Log to tracker
    try:
        from mousebrain.tracker import ExperimentTracker
        tracker = ExperimentTracker()
        exp_id = tracker.log_prefilter(
            brain=brain_name,
            total=result['stats']['total'],
            interior=result['stats']['interior'],
            suspicious=result['stats']['suspicious'],
            tracing_type=args.tracing_type,
            input_path=str(candidates_xml),
            output_path=str(output_dir),
            status="completed",
            script_version=SCRIPT_VERSION,
        )
        print(f"\nLogged to tracker: {exp_id}")
    except Exception as e:
        print(f"\nNote: Could not log to tracker: {e}")

    print(f"\nCompleted in {duration:.1f}s")
    print(f"Output: {output_dir}")
    print(f"\nNext step: Run classification on interior_candidates.xml")
    print(f"  python 5_classify_cells.py --brain {args.brain} "
          f"--candidates {saved['interior']}")

    # Visualize in napari
    if args.view:
        _view_in_napari(result, registration_path, brain_path.name)


def _view_in_napari(result: dict, registration_path: Path, brain_name: str):
    """Open napari with atlas + interior/suspicious point layers."""
    import napari

    stats = result['stats']
    brain_scales = stats['brain_scales']
    axis_perm = stats['axis_perm']

    # Load registered atlas as background
    annotation_path = registration_path / "registered_atlas.tiff"
    if not annotation_path.exists():
        annotation_path = registration_path / "annotation.tiff"
    registered_annotation = tifffile.imread(str(annotation_path))

    def to_display(coords):
        arr = np.array(coords, dtype=np.float32)
        if len(arr) == 0:
            return np.empty((0, 3), dtype=np.float32)
        scaled = np.column_stack([
            arr[:, 0] * brain_scales[0],
            arr[:, 1] * brain_scales[1],
            arr[:, 2] * brain_scales[2],
        ])
        return np.column_stack([
            scaled[:, axis_perm[0]],
            scaled[:, axis_perm[1]],
            scaled[:, axis_perm[2]],
        ])

    interior_pts = to_display(result['interior_coords'])
    suspicious_pts = to_display(result['suspicious_coords'])

    viewer = napari.Viewer(title=f"Pre-Filter: {brain_name}")
    viewer.add_labels(registered_annotation, name="Atlas")

    if len(interior_pts) > 0:
        viewer.add_points(
            interior_pts, name=f"Interior ({len(interior_pts):,})",
            face_color='#00FF00', size=3, opacity=0.6,
        )
    if len(suspicious_pts) > 0:
        viewer.add_points(
            suspicious_pts, name=f"Suspicious ({len(suspicious_pts):,})",
            face_color='#FF0000', size=4, opacity=0.8,
        )

    print(f"\nnapari viewer open -- close window to exit.")
    napari.run()


if __name__ == "__main__":
    main()
