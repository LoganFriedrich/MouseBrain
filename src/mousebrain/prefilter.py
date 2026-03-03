#!/usr/bin/env python3
"""
prefilter.py - Canonical location for atlas pre-filter logic.

Pre-filter cell candidates using brain surface edge detection. Separates
detection candidates into:
  - Interior: kept for classification (deep inside brain)
  - Surface: removed (within surface_depth_um of the brain edge)
  - Extreme OOB: removed (candidates far beyond atlas bounds)

Two filtering criteria:

1. SURFACE EDGE — ANY candidate near the brain surface is removed. Real
   labeled neurons are never at the tissue edge; surface fluorescence is
   autofluorescence, tissue damage, or incomplete clearing. Uses either:
   - Atlas mask erosion (default): binary erosion on annotation > 0
   - Background intensity (optional): uses the autofluorescence channel
     (downsampled.tiff). Dark background = non-tissue (surface, ventricle,
     damage). Candidates near dark regions are removed.

2. EXTREME OOB — candidates far beyond the atlas boundary. Nearby OOB
   candidates are KEPT (spinal cord, ventral brainstem). Only candidates
   far outside (default >500um) are removed.

The primary purpose is to CLEAN CLASSIFIER TRAINING DATA. Surface artifacts
in the "non-cell" training class poison the classifier. This filter removes
surface junk before it enters the training pipeline.

Usage (CLI):
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --surface-depth 150
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --image-edges
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


# =============================================================================
# BACKGROUND-BASED SURFACE DETECTION
# =============================================================================

def _detect_image_surface(registration_path, brain_mask, atlas_resolution,
                          surface_depth_vox, ann_shape):
    """Detect tissue surfaces using background intensity + atlas confirmation.

    Two signals must BOTH agree for a region to be classified as surface:

    1. BACKGROUND: the autofluorescence channel (downsampled.tiff) is dark
       there — meaning non-tissue (exterior, ventricle, damage).
    2. ATLAS: the region is near an atlas structural boundary (edge of
       brain_mask, including internal holes like ventricle annotation gaps).

    A dark spot deep inside registered cortex is just local dimness — the
    atlas says "real tissue here," so we keep candidates there. A dark
    region that lines up with the atlas boundary is a confirmed surface.

    Returns a boolean mask where True = surface exclusion zone, or None
    if downsampled.tiff is not available.
    """
    from scipy import ndimage

    # downsampled.tiff = background/autofluorescence channel (ch0).
    # Registration channel in brainreg: uniform tissue brightness,
    # no labeled cell puncta.
    background_path = registration_path / "downsampled.tiff"
    if not background_path.exists():
        print(f"  [!] downsampled.tiff not found -- falling back to atlas erosion")
        return None

    print(f"[{timestamp()}] Loading background channel for surface detection...")
    background = tifffile.imread(str(background_path))
    if background.shape != tuple(ann_shape):
        print(f"  [!] Background shape {background.shape} != atlas shape {tuple(ann_shape)}")
        print(f"  [!] Falling back to atlas erosion")
        del background
        return None

    # --- Signal 1: Background darkness ---
    # Tissue is bright in the background channel. Dark = non-tissue.
    brain_bg = background[brain_mask].ravel()
    nonzero = brain_bg[brain_bg > 0]
    if len(nonzero) == 0:
        del background
        return None

    dark_threshold = np.percentile(nonzero, 10)
    del brain_bg, nonzero

    # Dark regions in and near the brain (thin outer shell for context)
    search_region = ndimage.binary_dilation(brain_mask, iterations=3)
    dark_mask = (background < dark_threshold) & search_region
    del background, search_region

    dark_count = int(dark_mask.sum())
    print(f"  Dark background (< {dark_threshold:.0f}): {dark_count:,} voxels")

    if dark_count == 0:
        print(f"  [!] No dark regions found -- falling back to atlas erosion")
        del dark_mask
        return None

    # --- Signal 2: Atlas structural boundary ---
    # Inner edge of brain_mask: captures outer brain boundary AND
    # edges of internal holes (ventricle spaces, atlas gaps).
    atlas_inner_edge = brain_mask & ~ndimage.binary_erosion(brain_mask)

    # Wide tolerance: real tissue edges can be sharp or gradually fade
    # over many voxels, and registration is never pixel-perfect.
    # 3x surface_depth (e.g. 300um at default 100um depth) is generous
    # enough to catch gradual fades while still protecting deep interior
    # from false positives due to local dimness.
    boundary_tolerance = surface_depth_vox * 3
    atlas_boundary_zone = ndimage.binary_dilation(
        atlas_inner_edge, iterations=boundary_tolerance
    )
    del atlas_inner_edge

    boundary_count = int(atlas_boundary_zone.sum())
    print(f"  Atlas boundary zone ({boundary_tolerance} vox = "
          f"{boundary_tolerance * atlas_resolution:.0f}um tolerance): "
          f"{boundary_count:,} voxels")

    # --- Confirmed surface: dark background AND near atlas boundary ---
    # Both signals must agree. This prevents:
    # - Dark interior spots from being called surface (atlas says real tissue)
    # - Atlas boundary with bright background from being removed (tissue ok)
    confirmed_surface = dark_mask & atlas_boundary_zone
    del dark_mask, atlas_boundary_zone

    confirmed_count = int(confirmed_surface.sum())
    print(f"  Confirmed surface (dark AND boundary): {confirmed_count:,} voxels")

    if confirmed_count == 0:
        print(f"  [!] No confirmed surfaces -- falling back to atlas erosion")
        del confirmed_surface
        return None

    # Dilate confirmed surfaces to create exclusion zone
    print(f"[{timestamp()}] Building exclusion zone (dilating by {surface_depth_vox} voxels)...")
    exclusion_zone = ndimage.binary_dilation(
        confirmed_surface, iterations=surface_depth_vox
    )
    del confirmed_surface

    exclusion_count = int(exclusion_zone.sum())
    print(f"  Exclusion zone voxels: {exclusion_count:,}")

    return exclusion_zone


# =============================================================================
# CORE PRE-FILTER FUNCTION
# =============================================================================

def prefilter_candidates(
    candidates_xml: Path,
    registration_path: Path,
    atlas_name: str = "allen_mouse_10um",
    tracing_type: str = "descending",
    surface_depth_um: float = 100.0,
    extreme_oob_um: float = 500.0,
    use_image_edges: bool = False,
) -> dict:
    """
    Pre-filter cell candidates by surface edge + extreme-OOB criteria.

    Removes ALL candidates near the brain surface — real labeled neurons
    are never at the tissue edge. Surface fluorescence is autofluorescence,
    tissue damage, or incomplete clearing artifacts.

    Two surface detection modes:
    - Atlas erosion (default): binary erosion on annotation mask
    - Image edges (use_image_edges=True): Sobel edge detection on the
      signal channel, confirmed against atlas outline

    Unmapped candidates (region_id=0) are always KEPT.

    Args:
        candidates_xml: Path to detection candidates XML
        registration_path: Path to 3_Registered_Atlas folder
        atlas_name: BrainGlobe atlas name (default: allen_mouse_10um)
        tracing_type: 'descending', 'ascending', or 'unknown' (logged, not used for filtering)
        surface_depth_um: Depth in microns defining the surface shell (default: 100)
        extreme_oob_um: Distance in microns beyond atlas to consider extreme OOB (default: 500)
        use_image_edges: If True, use CV edge detection on signal channel (default: False)

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
    # The annotation mask defines exactly where atlas regions are — its boundary
    # IS the brain surface. The hemisphere mask extends far beyond the annotation
    # (61% vs 28% of volume), making it too generous for surface definition.
    # With hemisphere mask, surface erosion barely removes anything because
    # candidates at the actual tissue surface appear "deep" relative to the
    # oversized hemisphere boundary.
    brain_mask = registered_annotation > 0
    print(f"  Brain mask (annotation>0): {int(brain_mask.sum()):,} voxels")

    # Read brain voxel sizes from brainreg.json
    brainreg_json = registration_path / "brainreg.json"
    brain_voxel_z = 4.0
    brain_voxel_xy = 4.0

    if brainreg_json.exists():
        with open(brainreg_json) as f:
            brainreg_meta = json.load(f)
        voxel_sizes = brainreg_meta.get('voxel_sizes', ['4.0', '4.0', '4.0'])
        try:
            brain_voxel_z = float(voxel_sizes[0])
            brain_voxel_xy = float(voxel_sizes[1])
        except (IndexError, ValueError):
            pass

    # Scale factors: brain coordinates -> atlas coordinates
    scale_z = brain_voxel_z / atlas_resolution
    scale_xy = brain_voxel_xy / atlas_resolution

    # Convert depth thresholds to atlas voxels
    surface_depth_vox = max(1, int(round(surface_depth_um / atlas_resolution)))
    extreme_oob_vox = max(1, int(round(extreme_oob_um / atlas_resolution)))

    surface_method = "image_edges" if use_image_edges else "atlas_erosion"
    print(f"  Brain voxel: Z={brain_voxel_z}um, XY={brain_voxel_xy}um")
    print(f"  Atlas resolution: {atlas_resolution}um")
    print(f"  Scale factors: Z={scale_z:.3f}, XY={scale_xy:.3f}")
    print(f"  Surface method: {surface_method}")
    print(f"  Surface depth: {surface_depth_um}um ({surface_depth_vox} atlas voxels)")
    print(f"  Extreme OOB: {extreme_oob_um}um ({extreme_oob_vox} atlas voxels)")

    ann_shape = registered_annotation.shape
    print(f"  Atlas shape: {ann_shape} (indexed as [Z, Y, X])")

    # Build surface exclusion mask.
    # Two modes:
    #   1. Image edge detection: Sobel edges on signal, confirmed by atlas boundary
    #   2. Atlas erosion (fallback): binary erosion on annotation mask
    exclusion_mask = None
    if use_image_edges:
        exclusion_mask = _detect_image_surface(
            registration_path, brain_mask, atlas_resolution,
            surface_depth_vox, ann_shape
        )

    if exclusion_mask is None:
        # Atlas erosion mode (default or fallback)
        print(f"[{timestamp()}] Computing surface mask (eroding brain mask by {surface_depth_vox} voxels)...")
        eroded_mask = ndimage.binary_erosion(brain_mask, iterations=surface_depth_vox)
        deep_count = int(eroded_mask.sum())
        brain_count = int(brain_mask.sum())
        surface_count = brain_count - deep_count
        print(f"  Brain voxels: {brain_count:,}  Deep interior: {deep_count:,}  Surface shell: {surface_count:,}")
        # In erosion mode, surface = brain_mask & ~eroded_mask
        # We check eroded_mask[idx] to decide keep/remove
        exclusion_mask = brain_mask & ~eroded_mask
        del eroded_mask
        surface_method = "atlas_erosion"

    # Extend exclusion zone OUTSIDE brain_mask to catch unmapped candidates
    # at the tissue surface. Candidates at annotation=0 are outside brain_mask
    # but many are at the tissue edge where autofluorescence is strongest.
    # Dilate brain_mask outward and add the outer shell to exclusion.
    print(f"[{timestamp()}] Extending exclusion to cover unmapped surface region...")
    outer_envelope = ndimage.binary_dilation(brain_mask, iterations=surface_depth_vox)
    outer_shell = outer_envelope & ~brain_mask
    exclusion_mask = exclusion_mask | outer_shell
    del outer_envelope, outer_shell

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
        # Scale to atlas space
        z_atlas = int(z_brain * scale_z)
        y_atlas = int(y_brain * scale_xy)
        x_atlas = int(x_brain * scale_xy)

        # Registered atlas is in brain image space (pages, rows, columns).
        # Always index as (z, y, x) — no axis swapping.
        idx = (z_atlas, y_atlas, x_atlas)
        in_bounds = (0 <= z_atlas < ann_shape[0] and
                     0 <= y_atlas < ann_shape[1] and
                     0 <= x_atlas < ann_shape[2])

        if not in_bounds:
            # How far beyond the atlas boundary?
            oob_dist = max(
                max(0, z_atlas - ann_shape[0] + 1), max(0, -z_atlas),
                max(0, y_atlas - ann_shape[1] + 1), max(0, -y_atlas),
                max(0, x_atlas - ann_shape[2] + 1), max(0, -x_atlas),
            )

            if oob_dist > extreme_oob_vox:
                # Extreme OOB — way too far from the atlas to be real
                extreme_oob_removed += 1
                suspicious_coords.append((z_brain, y_brain, x_brain))
                suspicious_details[(z_brain, y_brain, x_brain)] = (0, 'OOB', 'extreme_oob')
                category_counts['extreme_oob'] = category_counts.get('extreme_oob', 0) + 1
            else:
                # Nearby OOB — likely spinal cord / ventral brainstem, keep
                out_of_bounds_kept += 1
                interior_coords.append((z_brain, y_brain, x_brain))
            continue

        # Surface filter FIRST: remove ANY candidate in the exclusion zone.
        # This catches both mapped AND unmapped candidates near the surface.
        # The exclusion zone extends outside brain_mask to cover annotation=0
        # voxels at the tissue edge where autofluorescence is strongest.
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

        # Unmapped (region_id=0) NOT near surface — atlas boundary gaps, keep
        if region_id == 0:
            unmapped += 1
            interior_coords.append((z_brain, y_brain, x_brain))
            continue

        # Deep interior — keep
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
        'surface_method': surface_method,
        'tracing_type': tracing_type,
        'surface_depth_um': surface_depth_um,
        'extreme_oob_um': extreme_oob_um,
        'scale_z': scale_z,
        'scale_xy': scale_xy,
        'atlas_name': atlas_name,
        'annotation_shape': list(ann_shape),
    }

    # Print summary
    pct_i = len(interior_coords) / total * 100 if total else 0
    pct_s = len(suspicious_coords) / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"PRE-FILTER RESULTS  (method: {surface_method})")
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
    parser.add_argument('--surface-depth', type=float, default=100.0,
                        help='Surface shell depth in microns (default: 100)')
    parser.add_argument('--extreme-oob', type=float, default=500.0,
                        help='Distance in microns beyond atlas for extreme OOB removal (default: 500)')
    parser.add_argument('--output', type=Path,
                        help='Output directory (default: 4_Cell_Candidates/prefiltered_{timestamp})')
    parser.add_argument('--atlas', default='allen_mouse_10um',
                        help='BrainGlobe atlas name (default: allen_mouse_10um)')
    parser.add_argument('--image-edges', action='store_true',
                        help='Use CV edge detection on signal channel (default: atlas erosion)')
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
        atlas_name=args.atlas,
        tracing_type=args.tracing_type,
        surface_depth_um=args.surface_depth,
        extreme_oob_um=args.extreme_oob,
        use_image_edges=args.image_edges,
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
    scale_z = stats['scale_z']
    scale_xy = stats['scale_xy']

    # Load registered atlas as background
    annotation_path = registration_path / "registered_atlas.tiff"
    if not annotation_path.exists():
        annotation_path = registration_path / "annotation.tiff"
    registered_annotation = tifffile.imread(str(annotation_path))

    # Registered atlas is in brain image space — index as (z, y, x).
    # Scale brain coords to atlas coords for display overlay.
    def to_display(coords):
        arr = np.array(coords, dtype=np.float32)
        if len(arr) == 0:
            return np.empty((0, 3), dtype=np.float32)
        z, y, x = arr[:, 0], arr[:, 1], arr[:, 2]
        return np.column_stack([z * scale_z, y * scale_xy, x * scale_xy])

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
