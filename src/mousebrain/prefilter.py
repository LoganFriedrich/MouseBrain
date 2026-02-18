#!/usr/bin/env python3
"""
prefilter.py - Canonical location for atlas pre-filter logic.

Pre-filter cell candidates using registered atlas and biologically suspicious
region mapping. Separates detection candidates into:
  - Interior: kept for classification (legitimate regions, unmapped, nearby OOB)
  - Suspicious surface: removed (surface shell of biologically suspicious regions)
  - Extreme OOB: removed (candidates far beyond atlas bounds)

Two filtering criteria:

1. SUSPICIOUS SURFACE — candidates in biologically unlikely regions (cerebellar
   cortex, white matter, olfactory, cortical L1-3, etc.) that are near the
   brain surface. Uses binary erosion to distinguish surface shell from deep
   interior. Candidates deep inside suspicious regions are KEPT (could be real
   labeled neurons). Only the outer shell is removed (surface artifacts).

2. EXTREME OOB — candidates far beyond the atlas boundary. Nearby OOB
   candidates are KEPT (spinal cord, ventral brainstem). Only candidates
   far outside (default >500um) are removed.

The primary purpose is to CLEAN CLASSIFIER TRAINING DATA. Surface artifacts
in the "non-cell" training class poison the classifier. This filter removes
surface junk before it enters the training pipeline.

Usage (CLI):
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --surface-depth 150
    python -m mousebrain.prefilter --brain 357_CNT_02_08_1p625x_z4 --tracing-type ascending
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

SCRIPT_VERSION = "2.2.0"

FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


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
) -> dict:
    """
    Pre-filter cell candidates by suspicious-surface + extreme-OOB criteria.

    Two filters:
    1. Suspicious surface — candidates in biologically suspicious regions that
       are within surface_depth_um of the brain edge. Deep candidates in the
       same regions are KEPT.
    2. Extreme OOB — candidates more than extreme_oob_um beyond the atlas
       boundary. Nearby OOB candidates (spinal cord) are KEPT.

    Unmapped candidates (region_id=0) are always KEPT.

    Args:
        candidates_xml: Path to detection candidates XML
        registration_path: Path to 3_Registered_Atlas folder
        atlas_name: BrainGlobe atlas name (default: allen_mouse_10um)
        tracing_type: 'descending', 'ascending', or 'unknown'
        surface_depth_um: Depth in microns defining the surface shell (default: 100)
        extreme_oob_um: Distance in microns beyond atlas to consider extreme OOB (default: 500)

    Returns:
        dict with keys:
            interior_coords: list of (z, y, x) tuples to keep
            suspicious_coords: list of (z, y, x) tuples removed (surface suspicious + extreme OOB)
            suspicious_details: dict mapping (z,y,x) -> (region_id, acronym, category)
            category_counts: dict mapping category -> count
            stats: summary statistics dict
    """
    from brainglobe_atlasapi import BrainGlobeAtlas
    from scipy import ndimage

    from mousebrain.region_mapping import TracingType, is_suspicious_region

    tracing_map = {
        'descending': TracingType.DESCENDING,
        'ascending': TracingType.ASCENDING,
        'unknown': TracingType.UNKNOWN,
    }
    tracing_type_enum = tracing_map.get(tracing_type, TracingType.DESCENDING)

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

    print(f"  Brain voxel: Z={brain_voxel_z}um, XY={brain_voxel_xy}um")
    print(f"  Atlas resolution: {atlas_resolution}um")
    print(f"  Scale factors: Z={scale_z:.3f}, XY={scale_xy:.3f}")
    print(f"  Tracing type: {tracing_type}")
    print(f"  Surface depth: {surface_depth_um}um ({surface_depth_vox} atlas voxels)")
    print(f"  Extreme OOB: {extreme_oob_um}um ({extreme_oob_vox} atlas voxels)")

    # Compute eroded brain interior mask.
    # Voxels in eroded_mask=True are "deep interior" (more than surface_depth
    # from any brain edge). Voxels where eroded_mask=False but brain_mask=True
    # are in the surface shell.
    print(f"[{timestamp()}] Computing surface mask (eroding brain mask by {surface_depth_vox} voxels)...")
    eroded_mask = ndimage.binary_erosion(brain_mask, iterations=surface_depth_vox)
    deep_count = int(eroded_mask.sum())
    brain_count = int(brain_mask.sum())
    surface_count = brain_count - deep_count
    print(f"  Brain voxels: {brain_count:,}  Deep interior: {deep_count:,}  Surface shell: {surface_count:,}")
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
    suspicious_surface = 0
    suspicious_deep_kept = 0
    category_counts = {}  # category -> count

    ann_shape = registered_annotation.shape

    # Determine axis order: brainreg stores as [Y, Z, X] at atlas resolution
    atlas_is_yzx = ann_shape[0] > ann_shape[1]
    if atlas_is_yzx:
        print(f"  Atlas axis order: [Y, Z, X] (dim0={ann_shape[0]}~Y, dim1={ann_shape[1]}~Z)")
    else:
        print(f"  Atlas axis order: [Z, Y, X] (dim0={ann_shape[0]}~Z, dim1={ann_shape[1]}~Y)")

    print(f"[{timestamp()}] Classifying candidates...")
    for z_brain, y_brain, x_brain in all_coords:
        # Scale to atlas space
        z_atlas = int(z_brain * scale_z)
        y_atlas = int(y_brain * scale_xy)
        x_atlas = int(x_brain * scale_xy)

        # Index into registered atlas with correct axis order
        if atlas_is_yzx:
            idx = (y_atlas, z_atlas, x_atlas)
            in_bounds = (0 <= y_atlas < ann_shape[0] and
                         0 <= z_atlas < ann_shape[1] and
                         0 <= x_atlas < ann_shape[2])
        else:
            idx = (z_atlas, y_atlas, x_atlas)
            in_bounds = (0 <= z_atlas < ann_shape[0] and
                         0 <= y_atlas < ann_shape[1] and
                         0 <= x_atlas < ann_shape[2])

        if not in_bounds:
            # How far beyond the atlas boundary?
            if atlas_is_yzx:
                oob_dist = max(
                    max(0, y_atlas - ann_shape[0] + 1), max(0, -y_atlas),
                    max(0, z_atlas - ann_shape[1] + 1), max(0, -z_atlas),
                    max(0, x_atlas - ann_shape[2] + 1), max(0, -x_atlas),
                )
            else:
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

        region_id = int(registered_annotation[idx])

        # Unmapped (region_id=0) — atlas boundary gaps, keep
        if region_id == 0:
            unmapped += 1
            interior_coords.append((z_brain, y_brain, x_brain))
            continue

        # Check suspicious region
        if region_id in atlas.structures:
            acronym = atlas.structures[region_id]['acronym']
            category = is_suspicious_region(acronym, tracing_type_enum)
            if category:
                # Is this candidate in the surface shell or deep interior?
                is_deep = bool(eroded_mask[idx])
                if is_deep:
                    # Deep inside suspicious region — could be real, keep
                    suspicious_deep_kept += 1
                    interior_coords.append((z_brain, y_brain, x_brain))
                else:
                    # Surface shell of suspicious region — likely junk, remove
                    suspicious_surface += 1
                    suspicious_coords.append((z_brain, y_brain, x_brain))
                    suspicious_details[(z_brain, y_brain, x_brain)] = (region_id, acronym, category)
                    category_counts[category] = category_counts.get(category, 0) + 1
                continue

        # Legitimate region — keep
        interior_coords.append((z_brain, y_brain, x_brain))

    del eroded_mask, registered_annotation  # free memory

    # Build stats
    stats = {
        'total': total,
        'interior': len(interior_coords),
        'suspicious': len(suspicious_coords),
        'out_of_bounds_kept': out_of_bounds_kept,
        'extreme_oob_removed': extreme_oob_removed,
        'unmapped': unmapped,
        'suspicious_surface': suspicious_surface,
        'suspicious_deep_kept': suspicious_deep_kept,
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
    print(f"PRE-FILTER RESULTS  (tracing: {tracing_type})")
    print(f"{'='*60}")
    print(f"  Total candidates:         {total:>8,}")
    print(f"  Interior (keep):          {len(interior_coords):>8,}  ({pct_i:.1f}%)")
    print(f"    nearby OOB (kept):      {out_of_bounds_kept:>8,}")
    print(f"    unmapped (kept):        {unmapped:>8,}")
    print(f"    deep suspicious (kept): {suspicious_deep_kept:>8,}")
    print(f"  Removed:                  {len(suspicious_coords):>8,}  ({pct_s:.1f}%)")
    print(f"    suspicious surface:     {suspicious_surface:>8,}")
    print(f"    extreme OOB:            {extreme_oob_removed:>8,}")
    if category_counts:
        print(f"  By category:")
        for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:30s} {cnt:>6,}")
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

    # Atlas is [Y, Z, X]. Points are (z_brain, y_brain, x_brain).
    # Convert to atlas display coords: (y*scale_xy, z*scale_z, x*scale_xy)
    ann_shape = registered_annotation.shape
    atlas_is_yzx = ann_shape[0] > ann_shape[1]

    def to_display(coords):
        arr = np.array(coords, dtype=np.float32)
        if len(arr) == 0:
            return np.empty((0, 3), dtype=np.float32)
        z, y, x = arr[:, 0], arr[:, 1], arr[:, 2]
        if atlas_is_yzx:
            return np.column_stack([y * scale_xy, z * scale_z, x * scale_xy])
        else:
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
