#!/usr/bin/env python3
"""
util_atlas_prefilter.py - Pre-filter cell candidates using registered atlas.

Separates detection candidates into:
  - Interior: inside the brain (valid for classification)
  - Outside: region_id=0 (meninges/surface, pre-rejected)
  - Suspicious: biologically unlikely regions for the tracing type (optional)

This runs BETWEEN detection (step 4) and classification (step 5) to avoid
wasting classification time on candidates that are clearly not cells.

Usage:
    python util_atlas_prefilter.py --brain 357_CNT_02_08_1p625x_z4
    python util_atlas_prefilter.py --brain 357_CNT_02_08_1p625x_z4 --flag-suspicious
    python util_atlas_prefilter.py --brain 357_CNT_02_08_1p625x_z4 --candidates path/to/det.xml
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

# Add parent to path for config/tracker imports
sys.path.insert(0, str(Path(__file__).parent))

from config import BRAINS_ROOT, MODELS_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

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
    flag_suspicious: bool = False,
    tracing_type: str = "descending",
) -> dict:
    """
    Pre-filter cell candidates by atlas region.

    Args:
        candidates_xml: Path to detection candidates XML
        registration_path: Path to 3_Registered_Atlas folder
        atlas_name: BrainGlobe atlas name (default: allen_mouse_10um)
        flag_suspicious: Also flag biologically suspicious regions
        tracing_type: 'descending', 'ascending', or 'unknown'

    Returns:
        dict with keys:
            interior_coords: list of (z, y, x) tuples inside brain
            outside_coords: list of (z, y, x) tuples at region_id=0
            suspicious_coords: list of (z, y, x) tuples in suspicious regions
            suspicious_details: dict mapping (z,y,x) -> (region_id, acronym, category)
            stats: summary statistics dict
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

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
    # atlas_coord = brain_coord * (brain_voxel / atlas_voxel)
    scale_z = brain_voxel_z / atlas_resolution
    scale_xy = brain_voxel_xy / atlas_resolution

    print(f"  Brain voxel: Z={brain_voxel_z}um, XY={brain_voxel_xy}um")
    print(f"  Atlas resolution: {atlas_resolution}um")
    print(f"  Scale factors: Z={scale_z:.3f}, XY={scale_xy:.3f}")

    # Optionally import suspicious region checker
    is_suspicious_fn = None
    tracing_type_enum = None
    if flag_suspicious:
        try:
            from elife_region_mapping import is_suspicious_region, TracingType
            tracing_map = {
                'descending': TracingType.DESCENDING,
                'ascending': TracingType.ASCENDING,
                'unknown': TracingType.UNKNOWN,
            }
            tracing_type_enum = tracing_map.get(tracing_type, TracingType.DESCENDING)
            is_suspicious_fn = is_suspicious_region
            print(f"  Suspicious region filtering: ON ({tracing_type})")
        except ImportError:
            print("  WARNING: Could not import elife_region_mapping, suspicious filtering disabled")
            flag_suspicious = False

    # Parse candidates XML
    print(f"[{timestamp()}] Parsing candidates: {candidates_xml.name}")
    tree = ET.parse(str(candidates_xml))
    root = tree.getroot()

    # Collect all candidate coordinates
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
    outside_coords = []
    suspicious_coords = []
    suspicious_details = {}
    out_of_bounds = 0
    region_breakdown = {}  # region_id -> count for outside/suspicious

    ann_shape = registered_annotation.shape

    print(f"[{timestamp()}] Classifying candidates by atlas region...")
    for z_brain, y_brain, x_brain in all_coords:
        # Scale to atlas space
        z_atlas = int(z_brain * scale_z)
        y_atlas = int(y_brain * scale_xy)
        x_atlas = int(x_brain * scale_xy)

        # Bounds check (annotation shape is ZYX)
        if not (0 <= z_atlas < ann_shape[0] and
                0 <= y_atlas < ann_shape[1] and
                0 <= x_atlas < ann_shape[2]):
            out_of_bounds += 1
            outside_coords.append((z_brain, y_brain, x_brain))
            continue

        region_id = int(registered_annotation[z_atlas, y_atlas, x_atlas])

        if region_id == 0:
            # Outside brain
            outside_coords.append((z_brain, y_brain, x_brain))
            region_breakdown[0] = region_breakdown.get(0, 0) + 1
        elif flag_suspicious and is_suspicious_fn and region_id in atlas.structures:
            acronym = atlas.structures[region_id]['acronym']
            category = is_suspicious_fn(acronym, tracing_type_enum)
            if category:
                suspicious_coords.append((z_brain, y_brain, x_brain))
                suspicious_details[(z_brain, y_brain, x_brain)] = (region_id, acronym, category)
                region_breakdown[region_id] = region_breakdown.get(region_id, 0) + 1
            else:
                interior_coords.append((z_brain, y_brain, x_brain))
        else:
            interior_coords.append((z_brain, y_brain, x_brain))

    # Build stats
    stats = {
        'total': total,
        'interior': len(interior_coords),
        'outside': len(outside_coords),
        'suspicious': len(suspicious_coords),
        'out_of_bounds': out_of_bounds,
        'flag_suspicious': flag_suspicious,
        'tracing_type': tracing_type,
        'scale_z': scale_z,
        'scale_xy': scale_xy,
        'atlas_name': atlas_name,
        'annotation_shape': list(ann_shape),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"PRE-FILTER RESULTS")
    print(f"{'='*60}")
    print(f"  Total candidates:     {total:>8,}")
    print(f"  Interior (keep):      {len(interior_coords):>8,}  ({len(interior_coords)/total*100:.1f}%)")
    print(f"  Outside brain (drop): {len(outside_coords):>8,}  ({len(outside_coords)/total*100:.1f}%)")
    if flag_suspicious:
        print(f"  Suspicious (flag):    {len(suspicious_coords):>8,}  ({len(suspicious_coords)/total*100:.1f}%)")
    if out_of_bounds > 0:
        print(f"  Out of bounds:        {out_of_bounds:>8,}  (included in outside)")
    print(f"{'='*60}")

    return {
        'interior_coords': interior_coords,
        'outside_coords': outside_coords,
        'suspicious_coords': suspicious_coords,
        'suspicious_details': suspicious_details,
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
        output_dir/interior_candidates.xml
        output_dir/outside_candidates.xml
        output_dir/suspicious_candidates.xml (if applicable)
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

    # Outside brain candidates
    outside_path = output_dir / "outside_candidates.xml"
    _coords_to_xml(result['outside_coords'], outside_path)
    saved['outside'] = str(outside_path)

    # Suspicious candidates (optional)
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

def find_brain_path(brain_name: str) -> Optional[Path]:
    """Find brain folder by name (searches subdirectories)."""
    direct = BRAINS_ROOT / brain_name
    if direct.exists():
        return direct

    for subdir in BRAINS_ROOT.iterdir():
        if subdir.is_dir():
            candidate = subdir / brain_name
            if candidate.exists():
                return candidate

    # Fuzzy match
    for path in BRAINS_ROOT.rglob("*"):
        if path.is_dir() and brain_name in path.name:
            return path

    return None


def find_latest_candidates(brain_path: Path) -> Optional[Path]:
    """Find the most recent detection candidates XML for a brain."""
    det_dir = brain_path / FOLDER_DETECTION
    if not det_dir.exists():
        return None

    # Look for Detected_*.xml files, sorted by date (newest first)
    xml_files = sorted(det_dir.glob("Detected_*.xml"), reverse=True)
    if xml_files:
        return xml_files[0]

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
        description="Pre-filter cell candidates using registered atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: filter candidates for a brain (auto-finds latest detection)
  python util_atlas_prefilter.py --brain 357_CNT_02_08_1p625x_z4

  # With suspicious region flagging
  python util_atlas_prefilter.py --brain 357_CNT_02_08_1p625x_z4 --flag-suspicious

  # Specify detection XML explicitly
  python util_atlas_prefilter.py --brain 357_CNT_02_08_1p625x_z4 \\
      --candidates 4_Cell_Candidates/Detected_20260109_143022.xml
"""
    )
    parser.add_argument('--brain', required=True, help='Brain folder name')
    parser.add_argument('--candidates', type=Path,
                        help='Path to candidates XML (default: latest in 4_Cell_Candidates)')
    parser.add_argument('--flag-suspicious', action='store_true',
                        help='Also flag candidates in biologically suspicious regions')
    parser.add_argument('--tracing-type', default='descending',
                        choices=['descending', 'ascending', 'unknown'],
                        help='Tracing type for suspicious region filtering (default: descending)')
    parser.add_argument('--output', type=Path,
                        help='Output directory (default: 4_Cell_Candidates/prefiltered_{timestamp})')
    parser.add_argument('--atlas', default='allen_mouse_10um',
                        help='BrainGlobe atlas name (default: allen_mouse_10um)')

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
        flag_suspicious=args.flag_suspicious,
        tracing_type=args.tracing_type,
    )
    duration = time.time() - start_time

    # Save results
    brain_name = brain_path.name
    saved = save_prefilter_results(result, output_dir, brain_name, str(candidates_xml))

    # Log to tracker
    try:
        from experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        exp_id = tracker.log_prefilter(
            brain=brain_name,
            total=result['stats']['total'],
            interior=result['stats']['interior'],
            outside=result['stats']['outside'],
            suspicious=result['stats']['suspicious'],
            flag_suspicious=args.flag_suspicious,
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


if __name__ == "__main__":
    main()
