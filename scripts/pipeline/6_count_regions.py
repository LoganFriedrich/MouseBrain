#!/usr/bin/env python3
"""
6_count_regions.py

Script 6 in the BrainGlobe pipeline: Regional cell counting.

Counts classified cells by brain region using the registered atlas,
and automatically logs to experiment tracker.

Run this AFTER Script 5 (5_classify_cells.py).

================================================================================
HOW TO USE
================================================================================
    python 6_count_regions.py --brain 349_CNT_01_02_1p625x_z4
    python 6_count_regions.py  # Interactive mode

================================================================================
REQUIREMENTS
================================================================================
    - brainglobe-segmentation must be installed
    - Classification output in 5_Classified_Cells folder
    - Registration in 3_Registered_Atlas folder
    - experiment_tracker.py in same directory or PYTHONPATH
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.tracker import ExperimentTracker
except ImportError:
    print("ERROR: mousebrain.tracker not found!")
    print("Make sure mousebrain package is installed.")
    sys.exit(1)

# =============================================================================
# VERIFY BRAINGLOBE ENVIRONMENT
# =============================================================================
def check_brainglobe_env():
    """Check if brainglobe-utils is available (indicates correct environment)."""
    try:
        import brainglobe_utils
        return True
    except ImportError:
        print("=" * 70)
        print("ERROR: brainglobe-utils not found!")
        print("=" * 70)
        print()
        print("This script requires brainglobe-utils for proper coordinate")
        print("transformation. You need to activate the brainglobe environment.")
        print()
        print("To fix this, run:")
        print("    conda activate brainglobe-env")
        print("    python 6_count_regions.py")
        print()
        print("Or run directly with the brainglobe Python:")
        print("    G:\\Program_Files\\Conda\\envs\\brainglobe-env\\python.exe 6_count_regions.py")
        print()
        print("(The script will use a fallback method, but results may be less accurate)")
        return False

# Check environment
_has_brainglobe_utils = check_brainglobe_env()

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.1"

from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
from mousebrain.config import DATA_SUMMARY_DIR as SUMMARY_DATA_DIR
from mousebrain.config import (
    REGION_COUNTS_CSV, REGION_COUNTS_ARCHIVE_CSV,
    ELIFE_COUNTS_CSV, ELIFE_COUNTS_ARCHIVE_CSV,
    parse_brain_name
)

FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_CLASSIFICATION = "5_Classified_Cells"
FOLDER_ANALYSIS = "6_Region_Analysis"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def find_pipeline(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find pipeline folder for a brain."""
    root = Path(root)
    
    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            full_name = f"{mouse_dir.name}/{pipeline_dir.name}"
            if brain_name in [pipeline_dir.name, full_name]:
                return pipeline_dir, mouse_dir
    
    return None, None


def find_classified_cells(pipeline_folder: Path):
    """Find classified cells output."""
    class_folder = pipeline_folder / FOLDER_CLASSIFICATION
    
    if not class_folder.exists():
        return None
    
    cells_xml = class_folder / "cells.xml"
    if cells_xml.exists():
        return cells_xml
    
    return None


def count_cells_in_xml(xml_path: Path) -> int:
    """Count cells in XML file."""
    if not xml_path.exists():
        return 0
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        return content.count('<Marker>')
    except:
        return 0


def list_available_brains(root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """List all brains that have classification output."""
    root = Path(root)
    brains = []
    
    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            cells_xml = find_classified_cells(pipeline_dir)
            analysis_folder = pipeline_dir / FOLDER_ANALYSIS
            
            has_classification = cells_xml is not None
            has_counts = analysis_folder.exists() and len(list(analysis_folder.glob("*.csv"))) > 0
            
            if has_classification:
                n_cells = count_cells_in_xml(cells_xml) if cells_xml else 0
                brains.append({
                    'name': f"{mouse_dir.name}/{pipeline_dir.name}",
                    'pipeline': pipeline_dir,
                    'mouse': mouse_dir,
                    'cells_xml': cells_xml,
                    'n_cells': n_cells,
                    'has_counts': has_counts,
                })
    
    return brains


def run_brainglobe_counts(
    cells_path: Path,
    registration_path: Path,
    output_path: Path,
    atlas: str = "allen_mouse_10um",
) -> tuple:
    """
    Count cells by region using BrainGlobe's proper coordinate transformation.

    Uses brainglobe-utils to transform cell coordinates through the registration
    deformation fields and assign atlas regions.

    Returns:
        (success, duration, total_cells, output_csv)
    """
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv = output_path / "cell_counts_by_region.csv"

    print(f"\n[{timestamp()}] Running regional cell counting...")
    print(f"    Cells: {cells_path}")
    print(f"    Registration: {registration_path}")
    print(f"    Output: {output_path}")
    print()

    start_time = time.time()

    # Try the proper BrainGlobe method first
    try:
        return run_counts_brainglobe_transform(
            cells_path, registration_path, output_path, output_csv, atlas, start_time
        )
    except ImportError as e:
        print(f"    BrainGlobe utils not available: {e}")
        print(f"    Falling back to direct atlas lookup...")
    except Exception as e:
        print(f"    BrainGlobe transform failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"    Falling back to direct atlas lookup...")

    # Fallback: direct coordinate lookup (less accurate)
    return run_counts_direct_lookup(
        cells_path, registration_path, output_csv, atlas, start_time
    )


def run_counts_brainglobe_transform(
    cells_path: Path,
    registration_path: Path,
    output_path: Path,
    output_csv: Path,
    atlas_name: str,
    start_time: float
) -> tuple:
    """
    Count cells by region using BrainGlobe's coordinate transformation.

    This is the CORRECT method - transforms cell coordinates through the
    registration deformation fields to get proper atlas coordinates.
    """
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_utils.brainmapper.analysis import summarise_points_by_atlas_region
    from brainglobe_utils.brainreg.transform import (
        transform_points_from_raw_to_downsampled_space,
        transform_points_from_downsampled_to_atlas_space,
    )
    import brainglobe_space as bgs
    import numpy as np
    import tifffile
    import xml.etree.ElementTree as ET

    registration_path = Path(registration_path)
    cells_path = Path(cells_path)

    print("    Using BrainGlobe coordinate transformation...")

    # Load atlas
    atlas = BrainGlobeAtlas(atlas_name)

    # Load registration metadata
    brainreg_json = registration_path / "brainreg.json"
    if not brainreg_json.exists():
        raise FileNotFoundError(f"brainreg.json not found in {registration_path}")

    with open(brainreg_json) as f:
        reg_meta = json.load(f)

    orientation = reg_meta.get('orientation', 'iar')
    voxel_sizes = [float(v) for v in reg_meta.get('voxel_sizes', ['4.0', '4.0', '4.0'])]
    print(f"    Orientation: {orientation}, Voxel sizes: {voxel_sizes}")

    # Load deformation fields
    deformation_fields = [
        registration_path / "deformation_field_0.tiff",
        registration_path / "deformation_field_1.tiff",
        registration_path / "deformation_field_2.tiff",
    ]
    for df in deformation_fields:
        if not df.exists():
            raise FileNotFoundError(f"Deformation field not found: {df}")

    # Load downsampled image for reference shape
    downsampled_path = registration_path / "downsampled.tiff"
    if not downsampled_path.exists():
        raise FileNotFoundError(f"downsampled.tiff not found in {registration_path}")

    with tifffile.TiffFile(str(downsampled_path)) as tiff:
        downsampled_shape = tiff.series[0].shape
    print(f"    Downsampled shape: {downsampled_shape}")
    print(f"    Source voxel sizes: {voxel_sizes}")

    # Create the downsampled space (matches atlas orientation and resolution)
    # The atlas uses a specific orientation, and the downsampled space matches it
    atlas_resolution = atlas.resolution[0]  # e.g., 10um for allen_mouse_10um
    downsampled_space = bgs.AnatomicalSpace(
        atlas.space.origin,  # Use atlas orientation (e.g., 'asr' for Allen)
        shape=downsampled_shape,
        resolution=[atlas_resolution] * 3,
    )
    print(f"    Atlas orientation: {atlas.space.origin}, resolution: {atlas_resolution}um")

    # Get path to source image stack folder for reference
    crop_folder = registration_path.parent / "2_Cropped_For_Registration"
    ch_folder = crop_folder / "ch0"
    if not ch_folder.exists():
        ch_folder = crop_folder / "ch1"
    if ch_folder.exists() and list(ch_folder.glob("Z*.tif")):
        # BrainGlobe expects folder path for tif stack
        source_image_path = ch_folder
    else:
        raise FileNotFoundError(f"Cannot find source image stack in {crop_folder}")
    print(f"    Source image reference: {source_image_path}")

    # Parse cells from XML
    tree = ET.parse(str(cells_path))
    root = tree.getroot()

    # Collect points in (z, y, x) order to match source shape dimensions
    points_raw = []
    for marker in root.iter('Marker'):
        x = int(marker.find('MarkerX').text)
        y = int(marker.find('MarkerY').text)
        z = int(marker.find('MarkerZ').text)
        # For "iar" orientation, source shape is (z, y, x)
        points_raw.append([z, y, x])

    points_raw = np.array(points_raw, dtype=np.float64)
    print(f"    Loaded {len(points_raw)} cells from XML")
    if len(points_raw) > 0:
        print(f"    Coord ranges: z=[{int(points_raw[:,0].min())}-{int(points_raw[:,0].max())}], "
              f"y=[{int(points_raw[:,1].min())}-{int(points_raw[:,1].max())}], "
              f"x=[{int(points_raw[:,2].min())}-{int(points_raw[:,2].max())}]")

    # Two-step transformation using BrainGlobe's proper methods:
    # Step 1: Raw -> Downsampled space (handles orientation + resolution scaling)
    print("    Step 1: Transforming to downsampled space (orientation + scaling)...")
    points_downsampled = transform_points_from_raw_to_downsampled_space(
        points_raw,
        source_image_path,  # Reference image for shape
        orientation,        # Source orientation (e.g., 'iar')
        voxel_sizes,        # Source voxel sizes in microns
        downsampled_space,  # Target space (atlas orientation + resolution)
    )
    print(f"    Points in downsampled space: {len(points_downsampled)}")

    # Step 2: Downsampled -> Atlas space (applies deformation fields)
    print("    Step 2: Transforming through deformation fields...")
    points_atlas, points_out_of_bounds = transform_points_from_downsampled_to_atlas_space(
        points_downsampled,
        atlas,
        deformation_field_paths=deformation_fields,
        warn_out_of_bounds=False,
    )
    print(f"    Transformed {len(points_atlas)} points to atlas space")
    if len(points_out_of_bounds) > 0:
        print(f"    ({len(points_out_of_bounds)} points fell outside atlas bounds)")

    # Generate summary
    volumes_csv = registration_path / "volumes.csv"
    points_list_csv = output_path / "all_points_information.csv"
    summary_csv = output_path / "all_points_information_summary.csv"

    print("    Summarizing points by atlas region...")
    summarise_points_by_atlas_region(
        points_in_raw_data_space=points_raw,
        points_in_atlas_space=points_atlas,
        atlas=atlas,
        brainreg_volume_csv_path=volumes_csv if volumes_csv.exists() else None,
        points_list_output_filename=points_list_csv,
        summary_filename=summary_csv,
    )

    print(f"    Generated: {points_list_csv.name}")
    print(f"    Generated: {summary_csv.name}")

    # Convert BrainGlobe output to our format (with hemisphere data)
    return convert_brainglobe_output(points_list_csv, summary_csv, output_csv, start_time)


def convert_brainglobe_output(
    per_cell_csv: Path, summary_csv: Path, output_csv: Path, start_time: float
) -> tuple:
    """
    Convert BrainGlobe's per-cell CSV to our format with hemisphere data.

    Reads all_points_information.csv (per-cell with hemisphere column) to aggregate
    counts by region and hemisphere. Falls back to summary CSV if per-cell is missing.

    Output: region_acronym, region_name, cell_count, left_count, right_count
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    atlas = BrainGlobeAtlas("allen_mouse_10um")

    # Build name-to-acronym mapping
    name_to_acronym = {}
    for rid, info in atlas.structures.items():
        name = info.get('name', '')
        acronym = info.get('acronym', '')
        if name and acronym:
            name_to_acronym[name] = acronym

    # Read per-cell data with hemisphere info
    # Dict of {acronym: {'total': N, 'left': N, 'right': N}}
    region_hemi_counts = {}
    total_cells = 0

    if per_cell_csv.exists():
        print(f"    Reading per-cell data: {per_cell_csv.name}")
        with open(per_cell_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('structure_name', '').strip()
                hemisphere = row.get('hemisphere', '').strip().lower()

                if not name:
                    continue

                acronym = name_to_acronym.get(name, name)

                if acronym not in region_hemi_counts:
                    region_hemi_counts[acronym] = {'total': 0, 'left': 0, 'right': 0}

                region_hemi_counts[acronym]['total'] += 1
                if hemisphere == 'left':
                    region_hemi_counts[acronym]['left'] += 1
                elif hemisphere == 'right':
                    region_hemi_counts[acronym]['right'] += 1
                total_cells += 1
    elif summary_csv.exists():
        # Fallback: read summary CSV (has left/right columns too)
        print(f"    Per-cell CSV not found, falling back to summary: {summary_csv.name}")
        with open(summary_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('structure_name', '')
                try:
                    total = int(float(row.get('total_cells', 0)))
                    left = int(float(row.get('left_cell_count', 0)))
                    right = int(float(row.get('right_cell_count', 0)))
                except (ValueError, TypeError):
                    total = left = right = 0

                if total > 0:
                    acronym = name_to_acronym.get(name, name)
                    if acronym not in region_hemi_counts:
                        region_hemi_counts[acronym] = {'total': 0, 'left': 0, 'right': 0}
                    region_hemi_counts[acronym]['total'] += total
                    region_hemi_counts[acronym]['left'] += left
                    region_hemi_counts[acronym]['right'] += right
                    total_cells += total
    else:
        print(f"    ERROR: Neither per-cell nor summary CSV found")
        duration = time.time() - start_time
        return False, duration, 0, None

    # Build flat region_counts for backward compatibility
    region_counts = {acr: data['total'] for acr, data in region_hemi_counts.items()}

    total_left = sum(d['left'] for d in region_hemi_counts.values())
    total_right = sum(d['right'] for d in region_hemi_counts.values())
    print(f"    Counted {total_cells} cells in {len(region_counts)} regions")
    print(f"    Hemisphere breakdown: {total_left} left, {total_right} right")

    # Build acronym to full name mapping
    acronym_to_name = {}
    for rid, info in atlas.structures.items():
        acr = info.get('acronym', '')
        if acr:
            acronym_to_name[acr] = info.get('name', acr)

    # Write FULL DETAIL format with hemisphere columns
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['region_acronym', 'region_name', 'cell_count', 'left_count', 'right_count'])

        for acronym, count in sorted(region_counts.items(), key=lambda x: -x[1]):
            name = acronym_to_name.get(acronym, acronym)
            hemi = region_hemi_counts[acronym]
            writer.writerow([acronym, name, count, hemi['left'], hemi['right']])

    print(f"    Generated: {output_csv.name}")

    # Write ELIFE-GROUPED format with hemisphere columns
    try:
        from elife_region_mapping import aggregate_to_elife, ELIFE_GROUPS

        elife_csv = output_csv.parent / "cell_counts_elife_grouped.csv"
        aggregated = aggregate_to_elife(region_counts)

        # Also aggregate left and right counts separately
        left_counts = {acr: d['left'] for acr, d in region_hemi_counts.items() if d['left'] > 0}
        right_counts = {acr: d['right'] for acr, d in region_hemi_counts.items() if d['right'] > 0}
        aggregated_left = aggregate_to_elife(left_counts)
        aggregated_right = aggregate_to_elife(right_counts)

        with open(elife_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'elife_group', 'group_id', 'cell_count', 'left_count', 'right_count',
                'constituent_regions', 'description'
            ])

            # Write groups in order by ID
            for group_name in sorted(ELIFE_GROUPS.keys(), key=lambda x: ELIFE_GROUPS[x]["id"]):
                if group_name in aggregated:
                    data = aggregated[group_name]
                    left_total = aggregated_left.get(group_name, {}).get('count', 0)
                    right_total = aggregated_right.get(group_name, {}).get('count', 0)
                    # Format constituents as "GRN=3721; MRN=1525; ..."
                    constituents = "; ".join(
                        f"{k}={v}" for k, v in sorted(
                            data["constituents"].items(), key=lambda x: -x[1]
                        )
                    )
                    writer.writerow([
                        group_name,
                        ELIFE_GROUPS[group_name]["id"],
                        data["count"],
                        left_total,
                        right_total,
                        constituents,
                        ELIFE_GROUPS[group_name]["description"],
                    ])
                else:
                    writer.writerow([
                        group_name,
                        ELIFE_GROUPS[group_name]["id"],
                        0, 0, 0,
                        "",
                        ELIFE_GROUPS[group_name]["description"],
                    ])

            # Add unmapped regions
            if "[Unmapped]" in aggregated:
                data = aggregated["[Unmapped]"]
                left_total = aggregated_left.get("[Unmapped]", {}).get('count', 0)
                right_total = aggregated_right.get("[Unmapped]", {}).get('count', 0)
                top_unmapped = sorted(data["constituents"].items(), key=lambda x: -x[1])[:15]
                constituents = "; ".join(f"{k}={v}" for k, v in top_unmapped)
                if len(data["constituents"]) > 15:
                    constituents += f"; ... (+{len(data['constituents']) - 15} more)"
                writer.writerow([
                    "[Unmapped]",
                    99,
                    data["count"],
                    left_total,
                    right_total,
                    constituents,
                    "Regions not in eLife 25-group mapping",
                ])

        print(f"    Generated: {elife_csv.name} (eLife-style grouping)")
    except ImportError:
        print("    (eLife grouping skipped - mapping module not found)")

    duration = time.time() - start_time
    return True, duration, total_cells, output_csv


def run_counts_direct_lookup(cells_path, registration_path, output_csv, atlas_name, start_time):
    """
    Fallback: Count cells by direct lookup in registered_atlas.tiff.

    This is less accurate than using the deformation fields but works
    when brainglobe-utils isn't available.
    """
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
        import tifffile
        import xml.etree.ElementTree as ET

        registration_path = Path(registration_path)
        cells_path = Path(cells_path)
        output_csv = Path(output_csv)

        print("    Using direct atlas lookup (fallback method)...")

        atlas = BrainGlobeAtlas(atlas_name)
        atlas_resolution = atlas.resolution[0]

        annotation_path = registration_path / "registered_atlas.tiff"
        if not annotation_path.exists():
            annotation_path = registration_path / "annotation.tiff"

        if not annotation_path.exists():
            print(f"ERROR: No registered atlas found in {registration_path}")
            return False, time.time() - start_time, 0, None

        registered_annotation = tifffile.imread(str(annotation_path))
        print(f"    Annotation shape: {registered_annotation.shape}")

        # Get brain voxel sizes
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

        # Scale factor: brain coordinates to atlas coordinates
        # atlas_coord = brain_coord * (brain_voxel / atlas_voxel)
        scale_z = brain_voxel_z / atlas_resolution
        scale_xy = brain_voxel_xy / atlas_resolution

        print(f"    Brain voxel: Z={brain_voxel_z}um, XY={brain_voxel_xy}um")
        print(f"    Scale factors: Z={scale_z:.3f}, XY={scale_xy:.3f}")

        # Parse cells from XML
        tree = ET.parse(str(cells_path))
        root = tree.getroot()

        # Load hemisphere annotation if available
        hemisphere_path = registration_path / "registered_hemispheres.tiff"
        registered_hemispheres = None
        if hemisphere_path.exists():
            registered_hemispheres = tifffile.imread(str(hemisphere_path))
            print(f"    Hemisphere annotation loaded: {registered_hemispheres.shape}")
        else:
            print(f"    Warning: No registered_hemispheres.tiff found, hemisphere data unavailable")

        # Count cells by region with hemisphere tracking
        # Dict of {acronym: {'total': N, 'left': N, 'right': N}}
        region_hemi_counts = {}
        total_cells = 0
        outside_brain = 0
        out_of_bounds = 0

        for marker in root.iter('Marker'):
            # Raw coordinates in brain space
            x_brain = int(marker.find('MarkerX').text)
            y_brain = int(marker.find('MarkerY').text)
            z_brain = int(marker.find('MarkerZ').text)

            # Scale to atlas space
            x = int(x_brain * scale_xy)
            y = int(y_brain * scale_xy)
            z = int(z_brain * scale_z)

            # Check bounds (annotation shape is ZYX)
            if (0 <= z < registered_annotation.shape[0] and
                0 <= y < registered_annotation.shape[1] and
                0 <= x < registered_annotation.shape[2]):

                region_id = int(registered_annotation[z, y, x])

                if region_id > 0 and region_id in atlas.structures:
                    region_info = atlas.structures[region_id]
                    acronym = region_info['acronym']

                    if acronym not in region_hemi_counts:
                        region_hemi_counts[acronym] = {'total': 0, 'left': 0, 'right': 0}
                    region_hemi_counts[acronym]['total'] += 1

                    # Look up hemisphere (1=left, 2=right in BrainGlobe convention)
                    if registered_hemispheres is not None:
                        hemi_val = int(registered_hemispheres[z, y, x])
                        if hemi_val == 1:
                            region_hemi_counts[acronym]['left'] += 1
                        elif hemi_val == 2:
                            region_hemi_counts[acronym]['right'] += 1

                    total_cells += 1
                else:
                    outside_brain += 1
            else:
                out_of_bounds += 1

        # Build flat region_counts for backward compatibility
        region_counts = {acr: data['total'] for acr, data in region_hemi_counts.items()}

        total_left = sum(d['left'] for d in region_hemi_counts.values())
        total_right = sum(d['right'] for d in region_hemi_counts.values())
        print(f"    Counted {total_cells} cells in {len(region_counts)} regions")
        print(f"    Hemisphere breakdown: {total_left} left, {total_right} right")
        if out_of_bounds > 0:
            print(f"    ({out_of_bounds} cells out of annotation bounds after scaling)")
        if outside_brain > 0:
            print(f"    ({outside_brain} cells in unannotated regions / region_id=0)")

        # Build acronym to full name mapping
        acronym_to_name = {}
        for rid, info in atlas.structures.items():
            acr = info.get('acronym', '')
            if acr:
                acronym_to_name[acr] = info.get('name', acr)

        # Write output CSV with hemisphere columns
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['region_acronym', 'region_name', 'cell_count', 'left_count', 'right_count'])

            for acronym, count in sorted(region_counts.items(), key=lambda x: -x[1]):
                name = acronym_to_name.get(acronym, acronym)
                hemi = region_hemi_counts[acronym]
                writer.writerow([acronym, name, count, hemi['left'], hemi['right']])

        duration = time.time() - start_time
        return True, duration, total_cells, output_csv

    except ImportError as e:
        print(f"ERROR: BrainGlobe Atlas API not installed!")
        print(f"Install with: pip install brainglobe-atlasapi")
        print(f"Details: {e}")
        return False, time.time() - start_time, 0, None
    except Exception as e:
        print(f"ERROR during region counting: {e}")
        import traceback
        traceback.print_exc()
        return False, time.time() - start_time, 0, None


def update_region_counts_csv(
    brain_name: str,
    region_counts: Dict[str, int],
    detection_params: Dict[str, Any],
    run_date: Optional[str] = None,
    hemisphere_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Path:
    """
    Update the wide-format region_counts.csv with results from this run.

    This is the PRODUCTION tracking file - one row per brain showing current counts.
    When a brain already exists, the old row is moved to region_counts_archive.csv
    and replaced with the new data.

    Args:
        brain_name: Full brain name (e.g., "349_CNT_01_02_1p625x_z4")
        region_counts: Dict of {region_acronym: cell_count}
        detection_params: Dict of detection parameters used
        run_date: ISO timestamp (defaults to now)
        hemisphere_counts: Optional dict of {acronym: {'left': N, 'right': N}}

    Returns:
        Path to region_counts.csv
    """
    import shutil

    run_date = run_date or datetime.now().isoformat()

    # Parse brain name for hierarchy
    parsed = parse_brain_name(brain_name)

    # Build the data row (wide format)
    row_data = {
        # Identity
        'brain': brain_name,
        'run_date': run_date,

        # Hierarchy from brain name
        'brain_id': parsed.get('brain_id', ''),
        'subject': parsed.get('subject_full', ''),
        'cohort': parsed.get('cohort_full', ''),
        'project_code': parsed.get('project_code', ''),
        'project_name': parsed.get('project_name', ''),

        # Detection parameters (for reproducibility)
        'det_preset': detection_params.get('preset', ''),
        'det_ball_xy': detection_params.get('ball_xy', ''),
        'det_ball_z': detection_params.get('ball_z', ''),
        'det_soma_diameter': detection_params.get('soma_diameter', ''),
        'det_threshold': detection_params.get('threshold', ''),
        'atlas': detection_params.get('atlas', 'allen_mouse_10um'),
        'voxel_xy': detection_params.get('voxel_xy', ''),
        'voxel_z': detection_params.get('voxel_z', ''),

        # Total count
        'total_cells': sum(region_counts.values()),
    }

    # Add hemisphere totals if available
    if hemisphere_counts:
        row_data['total_left'] = sum(d.get('left', 0) for d in hemisphere_counts.values())
        row_data['total_right'] = sum(d.get('right', 0) for d in hemisphere_counts.values())

    # Add all region counts (wide format - each region is a column)
    for region, count in region_counts.items():
        row_data[f'region_{region}'] = count

    # Add hemisphere region counts if available
    if hemisphere_counts:
        for region, hemi in hemisphere_counts.items():
            row_data[f'region_left_{region}'] = hemi.get('left', 0)
            row_data[f'region_right_{region}'] = hemi.get('right', 0)

    # Read existing data
    existing_rows = []
    all_columns = set(row_data.keys())

    if REGION_COUNTS_CSV.exists():
        with open(REGION_COUNTS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_columns.update(row.keys())
                existing_rows.append(row)

    # Check if this brain already exists
    old_row = None
    new_rows = []
    for row in existing_rows:
        if row.get('brain') == brain_name:
            old_row = row
        else:
            new_rows.append(row)

    # Archive old row if it exists
    if old_row:
        archive_exists = REGION_COUNTS_ARCHIVE_CSV.exists()
        with open(REGION_COUNTS_ARCHIVE_CSV, 'a', newline='', encoding='utf-8') as f:
            # Get all archive columns
            archive_columns = list(all_columns)
            if archive_exists:
                with open(REGION_COUNTS_ARCHIVE_CSV, 'r', newline='', encoding='utf-8') as af:
                    reader = csv.DictReader(af)
                    archive_columns = list(set(archive_columns + list(reader.fieldnames or [])))

            # Add archived_at timestamp
            old_row['archived_at'] = datetime.now().isoformat()
            archive_columns = sorted(set(archive_columns) | {'archived_at'})

            writer = csv.DictWriter(f, fieldnames=archive_columns)
            if not archive_exists:
                writer.writeheader()
            writer.writerow({k: old_row.get(k, '') for k in archive_columns})

        print(f"    Archived previous run for {brain_name}")

    # Add new row
    new_rows.append(row_data)

    # Determine final column order: fixed → total regions → left regions → right regions
    fixed_columns = [
        'brain', 'run_date', 'brain_id', 'subject', 'cohort', 'project_code', 'project_name',
        'det_preset', 'det_ball_xy', 'det_ball_z', 'det_soma_diameter', 'det_threshold',
        'atlas', 'voxel_xy', 'voxel_z', 'total_cells',
    ]
    # Add hemisphere totals after total_cells if any row has them
    if any('total_left' in r for r in new_rows) or 'total_left' in all_columns:
        fixed_columns.extend(['total_left', 'total_right'])

    # Separate region columns into total, left, right groups
    total_region_cols = sorted([c for c in all_columns
                                if c.startswith('region_')
                                and not c.startswith('region_left_')
                                and not c.startswith('region_right_')])
    left_region_cols = sorted([c for c in all_columns if c.startswith('region_left_')])
    right_region_cols = sorted([c for c in all_columns if c.startswith('region_right_')])

    final_columns = fixed_columns + total_region_cols + left_region_cols + right_region_cols

    # Write updated CSV
    REGION_COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REGION_COUNTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_columns)
        writer.writeheader()
        for row in new_rows:
            writer.writerow({k: row.get(k, '') for k in final_columns})

    return REGION_COUNTS_CSV


def update_elife_counts_csv(
    brain_name: str,
    region_counts: Dict[str, int],
    detection_params: Dict[str, Any],
    run_date: Optional[str] = None,
    hemisphere_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Optional[Path]:
    """
    Update the wide-format elife_counts.csv with eLife-grouped results.

    Similar to update_region_counts_csv but aggregates to eLife 25 groups.
    Archives old data before replacing.

    Args:
        brain_name: Full brain name (e.g., "349_CNT_01_02_1p625x_z4")
        region_counts: Dict of {region_acronym: cell_count} (raw Allen data)
        detection_params: Dict of detection parameters used
        run_date: ISO timestamp (defaults to now)
        hemisphere_counts: Optional dict of {acronym: {'left': N, 'right': N}}

    Returns:
        Path to elife_counts.csv or None if eLife mapping unavailable
    """
    try:
        from elife_region_mapping import aggregate_to_elife, ELIFE_GROUPS
    except ImportError:
        return None

    run_date = run_date or datetime.now().isoformat()

    # Aggregate to eLife groups (total)
    aggregated = aggregate_to_elife(region_counts)

    # Aggregate left and right hemisphere counts separately
    aggregated_left = {}
    aggregated_right = {}
    if hemisphere_counts:
        left_counts = {acr: d['left'] for acr, d in hemisphere_counts.items() if d.get('left', 0) > 0}
        right_counts = {acr: d['right'] for acr, d in hemisphere_counts.items() if d.get('right', 0) > 0}
        aggregated_left = aggregate_to_elife(left_counts)
        aggregated_right = aggregate_to_elife(right_counts)

    # Parse brain name for hierarchy
    parsed = parse_brain_name(brain_name)

    # Build the data row (wide format)
    row_data = {
        # Identity
        'brain': brain_name,
        'run_date': run_date,

        # Hierarchy from brain name
        'brain_id': parsed.get('brain_id', ''),
        'subject': parsed.get('subject_full', ''),
        'cohort': parsed.get('cohort_full', ''),
        'project_code': parsed.get('project_code', ''),
        'project_name': parsed.get('project_name', ''),

        # Detection parameters (for reproducibility)
        'det_preset': detection_params.get('preset', ''),
        'det_ball_xy': detection_params.get('ball_xy', ''),
        'det_ball_z': detection_params.get('ball_z', ''),
        'det_soma_diameter': detection_params.get('soma_diameter', ''),
        'det_threshold': detection_params.get('threshold', ''),
        'atlas': detection_params.get('atlas', 'allen_mouse_10um'),
        'voxel_xy': detection_params.get('voxel_xy', ''),
        'voxel_z': detection_params.get('voxel_z', ''),

        # Total count
        'total_cells': sum(region_counts.values()),
    }

    # Add hemisphere totals if available
    if hemisphere_counts:
        row_data['total_left'] = sum(d.get('left', 0) for d in hemisphere_counts.values())
        row_data['total_right'] = sum(d.get('right', 0) for d in hemisphere_counts.values())

    # Add eLife group counts (wide format - each group is a column)
    for group_name, group_info in ELIFE_GROUPS.items():
        # Column name is sanitized group name
        col_name = f"group_{group_name.replace(' ', '_').replace('/', '_')}"
        if group_name in aggregated:
            row_data[col_name] = aggregated[group_name]["count"]
        else:
            row_data[col_name] = 0

        # Add hemisphere group counts
        if hemisphere_counts:
            left_col = f"group_left_{group_name.replace(' ', '_').replace('/', '_')}"
            right_col = f"group_right_{group_name.replace(' ', '_').replace('/', '_')}"
            row_data[left_col] = aggregated_left.get(group_name, {}).get('count', 0)
            row_data[right_col] = aggregated_right.get(group_name, {}).get('count', 0)

    # Add unmapped count
    if "[Unmapped]" in aggregated:
        row_data['group_Unmapped'] = aggregated["[Unmapped]"]["count"]
    if hemisphere_counts:
        if "[Unmapped]" in aggregated_left:
            row_data['group_left_Unmapped'] = aggregated_left["[Unmapped]"]["count"]
        if "[Unmapped]" in aggregated_right:
            row_data['group_right_Unmapped'] = aggregated_right["[Unmapped]"]["count"]

    # Read existing data
    existing_rows = []
    all_columns = set(row_data.keys())

    if ELIFE_COUNTS_CSV.exists():
        with open(ELIFE_COUNTS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_columns.update(row.keys())
                existing_rows.append(row)

    # Check if this brain already exists
    old_row = None
    new_rows = []
    for row in existing_rows:
        if row.get('brain') == brain_name:
            old_row = row
        else:
            new_rows.append(row)

    # Archive old row if it exists
    if old_row:
        archive_exists = ELIFE_COUNTS_ARCHIVE_CSV.exists()
        with open(ELIFE_COUNTS_ARCHIVE_CSV, 'a', newline='', encoding='utf-8') as f:
            # Get all archive columns
            archive_columns = list(all_columns)
            if archive_exists:
                with open(ELIFE_COUNTS_ARCHIVE_CSV, 'r', newline='', encoding='utf-8') as af:
                    reader = csv.DictReader(af)
                    archive_columns = list(set(archive_columns + list(reader.fieldnames or [])))

            # Add archived_at timestamp
            old_row['archived_at'] = datetime.now().isoformat()
            archive_columns = sorted(set(archive_columns) | {'archived_at'})

            writer = csv.DictWriter(f, fieldnames=archive_columns)
            if not archive_exists:
                writer.writeheader()
            writer.writerow({k: old_row.get(k, '') for k in archive_columns})

    # Add new row
    new_rows.append(row_data)

    # Determine final column order: fixed → total groups → left groups → right groups
    fixed_columns = [
        'brain', 'run_date', 'brain_id', 'subject', 'cohort', 'project_code', 'project_name',
        'det_preset', 'det_ball_xy', 'det_ball_z', 'det_soma_diameter', 'det_threshold',
        'atlas', 'voxel_xy', 'voxel_z', 'total_cells',
    ]
    if any('total_left' in r for r in new_rows) or 'total_left' in all_columns:
        fixed_columns.extend(['total_left', 'total_right'])

    total_group_cols = sorted([c for c in all_columns
                               if c.startswith('group_')
                               and not c.startswith('group_left_')
                               and not c.startswith('group_right_')])
    left_group_cols = sorted([c for c in all_columns if c.startswith('group_left_')])
    right_group_cols = sorted([c for c in all_columns if c.startswith('group_right_')])

    final_columns = fixed_columns + total_group_cols + left_group_cols + right_group_cols

    # Write updated CSV
    ELIFE_COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ELIFE_COUNTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_columns)
        writer.writeheader()
        for row in new_rows:
            writer.writerow({k: row.get(k, '') for k in final_columns})

    return ELIFE_COUNTS_CSV


# =============================================================================
# REACHING-RELEVANT REGIONS FOR SCI ANALYSIS
# =============================================================================
# Key supraspinal motor regions relevant to skilled reaching behavior
# and likely affected by midline contusion injury.
#
# Reference: Fouad et al. (2021) "The neuroanatomical-functional paradox in SCI"
# Nature Reviews Neurology. DOI: 10.1038/s41582-020-00436-x
#
# For 60kd midline contusion:
# - Dorsal CST is typically most damaged (runs dorsally in rodent cord)
# - Lateral tracts (rubrospinal) may have variable sparing
# - Reticulospinal (ventral) often more preserved
# - The paradox: similar lesions can produce vastly different outcomes

REACHING_REGIONS = {
    # Region name -> (eLife group, tract, role in reaching)
    "Red Nucleus": {
        "elife_group": "Red Nucleus",
        "tract": "Rubrospinal",
        "role": "Skilled forelimb reaching - critical for grasp/retrieval",
        "injury_note": "Lateral cord position; variable sparing in midline injury",
    },
    "Gigantocellular reticular nucleus": {
        "elife_group": "Gigantocellular Reticular Nucleus",
        "tract": "Reticulospinal (medial)",
        "role": "Postural support during reaching, gross movements",
        "injury_note": "Ventral cord; often preserved in dorsal contusion",
    },
    "Pontine Reticular Nuclei": {
        "elife_group": "Pontine Reticular Nuclei",
        "tract": "Reticulospinal (pontine)",
        "role": "Anticipatory postural adjustments",
        "injury_note": "Ventral cord; typically preserved",
    },
    "Corticospinal": {
        "elife_group": "Corticospinal",
        "tract": "Corticospinal (CST)",
        "role": "Fine motor control, digit manipulation",
        "injury_note": "DORSAL cord in rodents; MOST VULNERABLE to midline contusion",
    },
    "Cerebellospinal Nuclei": {
        "elife_group": "Cerebellospinal Nuclei",
        "tract": "Cerebellospinal",
        "role": "Motor coordination, timing",
        "injury_note": "Validates cervical injection (0 in lumbar)",
    },
    "Vestibular Nuclei": {
        "elife_group": "Vestibular Nuclei",
        "tract": "Vestibulospinal",
        "role": "Balance during reaching, postural stability",
        "injury_note": "Variable position; context-dependent sparing",
    },
    "Midbrain Reticular Nuclei": {
        "elife_group": "Midbrain Reticular Nuclei",
        "tract": "Tectospinal/Reticulospinal",
        "role": "Orienting, reaching initiation",
        "injury_note": "Variable sparing",
    },
}

# Tract vulnerability ranking for 60kd midline contusion (most to least damaged)
TRACT_VULNERABILITY_60KD_MIDLINE = [
    ("Corticospinal (CST)", "HIGH", "Dorsal position - directly impacted"),
    ("Rubrospinal", "MODERATE", "Lateral position - partial sparing expected"),
    ("Vestibulospinal", "MODERATE", "Variable cord position"),
    ("Reticulospinal (pontine)", "LOW", "Ventral position - often preserved"),
    ("Reticulospinal (medial)", "LOW", "Ventral position - often preserved"),
    ("Cerebellospinal", "LOW", "Lateral/ventral"),
]


def generate_cross_brain_graphs(
    output_dir: Path = None,
) -> Optional[Path]:
    """
    Generate summary graphs comparing all brains in the database.

    Creates:
    1. Bar chart of key reaching regions across all brains
    2. Heatmap of all eLife groups by brain
    3. Statistical summary with z-scores
    4. Tract vulnerability analysis for 60kd midline contusion

    Called automatically after updating region_counts.csv.

    Args:
        output_dir: Where to save graphs (defaults to DATA_SUMMARY_DIR/reports)

    Returns:
        Path to the generated report directory, or None if failed
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("    Warning: matplotlib not available - skipping graph generation")
        return None

    try:
        from elife_region_mapping import aggregate_to_elife, ELIFE_GROUPS
    except ImportError:
        print("    Warning: eLife mapping not available - skipping graph generation")
        return None

    if output_dir is None:
        output_dir = SUMMARY_DATA_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all brains from region_counts.csv
    if not REGION_COUNTS_CSV.exists():
        print("    No region_counts.csv yet - skipping graph generation")
        return None

    brains_data = {}
    with open(REGION_COUNTS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            brain_name = row.get('brain', '')
            if not brain_name:
                continue

            # Parse region counts (exclude hemisphere-specific columns)
            counts = {}
            left_counts = {}
            right_counts = {}
            for col, val in row.items():
                if col.startswith('region_left_'):
                    region = col[len('region_left_'):]
                    try:
                        left_counts[region] = int(val) if val else 0
                    except ValueError:
                        pass
                elif col.startswith('region_right_'):
                    region = col[len('region_right_'):]
                    try:
                        right_counts[region] = int(val) if val else 0
                    except ValueError:
                        pass
                elif col.startswith('region_'):
                    region = col[7:]  # Remove 'region_' prefix
                    try:
                        counts[region] = int(val) if val else 0
                    except ValueError:
                        pass

            # Store with metadata
            brains_data[brain_name] = {
                'counts': counts,
                'left_counts': left_counts,
                'right_counts': right_counts,
                'cohort': row.get('cohort', ''),
                'project': row.get('project_code', ''),
                'total': int(row.get('total_cells', 0) or 0),
                'total_left': int(row.get('total_left', 0) or 0),
                'total_right': int(row.get('total_right', 0) or 0),
            }

    if len(brains_data) < 1:
        print("    No brain data found - skipping graph generation")
        return None

    print(f"\n    Generating graphs for {len(brains_data)} brain(s)...")

    # Aggregate all brains to eLife groups (total, left, right)
    brains_elife = {}
    has_hemisphere = False
    for brain_name, data in brains_data.items():
        elife_agg = aggregate_to_elife(data['counts'])
        brains_elife[brain_name] = {
            group: elife_agg.get(group, {}).get('count', 0)
            for group in ELIFE_GROUPS.keys()
        }
        brains_elife[brain_name]['_cohort'] = data['cohort']
        brains_elife[brain_name]['_total'] = data['total']

        # Hemisphere aggregation
        if data['left_counts'] or data['right_counts']:
            has_hemisphere = True
            elife_left = aggregate_to_elife(data['left_counts'])
            elife_right = aggregate_to_elife(data['right_counts'])
            brains_elife[brain_name]['_left'] = {
                group: elife_left.get(group, {}).get('count', 0)
                for group in ELIFE_GROUPS.keys()
            }
            brains_elife[brain_name]['_right'] = {
                group: elife_right.get(group, {}).get('count', 0)
                for group in ELIFE_GROUPS.keys()
            }
            brains_elife[brain_name]['_total_left'] = data['total_left']
            brains_elife[brain_name]['_total_right'] = data['total_right']

    # ==========================================================================
    # GRAPH 1: Key Reaching Regions Bar Chart
    # ==========================================================================
    reaching_groups = [info['elife_group'] for info in REACHING_REGIONS.values()]
    reaching_groups = list(dict.fromkeys(reaching_groups))  # Remove duplicates, preserve order

    fig, ax = plt.subplots(figsize=(14, 8))

    brain_names = list(brains_elife.keys())
    x = np.arange(len(reaching_groups))
    width = 0.8 / max(len(brain_names), 1)

    # Color by cohort
    cohorts = list(set(brains_elife[b].get('_cohort', '') for b in brain_names))
    cohort_colors = plt.cm.Set2(np.linspace(0, 1, max(len(cohorts), 1)))
    cohort_color_map = {c: cohort_colors[i] for i, c in enumerate(cohorts)}

    for i, brain in enumerate(brain_names):
        counts = [brains_elife[brain].get(g, 0) for g in reaching_groups]
        cohort = brains_elife[brain].get('_cohort', '')
        color = cohort_color_map.get(cohort, 'steelblue')

        # Short label for legend
        short_name = brain.split('/')[-1][:15] if '/' in brain else brain[:15]
        bars = ax.bar(x + i * width - width * len(brain_names) / 2, counts, width,
                      label=short_name, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Brain Region (eLife Group)', fontsize=12)
    ax.set_ylabel('Cell Count', fontsize=12)
    ax.set_title('Key Reaching/Forelimb Regions Across Brains\n'
                 '(Relevant to skilled reaching behavior and SCI recovery)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(' ', '\n') for g in reaching_groups], fontsize=9, rotation=0)

    # Add vulnerability annotations
    for i, group in enumerate(reaching_groups):
        for region_name, info in REACHING_REGIONS.items():
            if info['elife_group'] == group:
                # Find vulnerability
                tract = info['tract']
                vuln = "?"
                for t, v, _ in TRACT_VULNERABILITY_60KD_MIDLINE:
                    if t in tract or tract in t:
                        vuln = v
                        break
                if vuln == "HIGH":
                    ax.annotate('⚠ HIGH\nDAMAGE', xy=(i, ax.get_ylim()[1] * 0.95),
                               ha='center', fontsize=7, color='red')
                break

    ax.legend(loc='upper right', fontsize=8, title='Brain')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    reaching_graph = output_dir / 'reaching_regions_comparison.png'
    plt.savefig(reaching_graph, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {reaching_graph.name}")

    # ==========================================================================
    # GRAPH 2: Full eLife Heatmap (if multiple brains)
    # ==========================================================================
    if len(brain_names) >= 1:
        all_groups = sorted(ELIFE_GROUPS.keys(), key=lambda x: ELIFE_GROUPS[x]['id'])

        # Build matrix
        matrix = np.zeros((len(brain_names), len(all_groups)))
        for i, brain in enumerate(brain_names):
            for j, group in enumerate(all_groups):
                matrix[i, j] = brains_elife[brain].get(group, 0)

        # Normalize by row (each brain) for visualization
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_norm = matrix / row_sums * 100  # Percentage of total

        fig, ax = plt.subplots(figsize=(16, max(4, len(brain_names) * 0.5 + 2)))

        im = ax.imshow(matrix_norm, aspect='auto', cmap='YlOrRd')

        # Labels
        short_names = [b.split('/')[-1][:20] if '/' in b else b[:20] for b in brain_names]
        ax.set_yticks(np.arange(len(brain_names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xticks(np.arange(len(all_groups)))
        ax.set_xticklabels([g[:15] for g in all_groups], fontsize=8, rotation=45, ha='right')

        # Highlight reaching regions
        for j, group in enumerate(all_groups):
            if group in reaching_groups:
                ax.axvline(j - 0.5, color='blue', linewidth=2, alpha=0.5)
                ax.axvline(j + 0.5, color='blue', linewidth=2, alpha=0.5)

        ax.set_title('eLife Region Distribution Across Brains (% of total)\n'
                     'Blue borders = reaching-relevant regions', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('% of Brain Total', fontsize=10)

        plt.tight_layout()
        heatmap_path = output_dir / 'elife_groups_heatmap.png'
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {heatmap_path.name}")

    # ==========================================================================
    # STATISTICAL SUMMARY
    # ==========================================================================
    stats_lines = []
    stats_lines.append("=" * 80)
    stats_lines.append("CROSS-BRAIN STATISTICAL SUMMARY")
    stats_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    stats_lines.append(f"Brains analyzed: {len(brain_names)}")
    stats_lines.append("=" * 80)

    # Calculate stats for reaching regions
    stats_lines.append("\n" + "=" * 80)
    stats_lines.append("REACHING-RELEVANT REGIONS (for skilled forelimb behavior)")
    stats_lines.append("=" * 80)
    stats_lines.append(f"\n{'Region':<40} {'Mean':>8} {'StdDev':>8} {'Min':>8} {'Max':>8}")
    stats_lines.append("-" * 80)

    region_stats = {}
    for group in reaching_groups:
        values = [brains_elife[b].get(group, 0) for b in brain_names]
        if values:
            mean = np.mean(values)
            std = np.std(values) if len(values) > 1 else 0
            region_stats[group] = {'mean': mean, 'std': std, 'min': min(values), 'max': max(values)}
            stats_lines.append(f"{group:<40} {mean:>8.1f} {std:>8.1f} {min(values):>8} {max(values):>8}")

    # Tract vulnerability analysis
    stats_lines.append("\n" + "=" * 80)
    stats_lines.append("TRACT VULNERABILITY ANALYSIS (60kd Midline Contusion)")
    stats_lines.append("Based on anatomical position in spinal cord")
    stats_lines.append("=" * 80)
    stats_lines.append("\nExpected damage pattern for 60kd midline contusion:")
    stats_lines.append("")

    for tract, vulnerability, reason in TRACT_VULNERABILITY_60KD_MIDLINE:
        marker = "⚠️" if vulnerability == "HIGH" else ("⚡" if vulnerability == "MODERATE" else "✓")
        stats_lines.append(f"  {marker} {tract:<30} {vulnerability:<10} - {reason}")

    stats_lines.append("\n" + "-" * 80)
    stats_lines.append("INTERPRETATION GUIDE:")
    stats_lines.append("  - HIGH vulnerability tracts: Expect significant reduction vs uninjured reference")
    stats_lines.append("  - LOW vulnerability tracts: May show preservation -> potential compensation")
    stats_lines.append("  - The 'anatomical paradox': Similar lesions can produce different outcomes")
    stats_lines.append("  - Correlate these counts with behavioral reaching scores")
    stats_lines.append("-" * 80)

    # Per-brain summary
    if len(brain_names) > 1:
        stats_lines.append("\n" + "=" * 80)
        stats_lines.append("PER-BRAIN Z-SCORES (relative to group mean)")
        stats_lines.append("Positive = above average, Negative = below average")
        stats_lines.append("=" * 80)
        header_line = f"\n{'Brain':<25}"
        for group in reaching_groups[:5]:  # Top 5 reaching regions
            short = group[:12]
            header_line += f" {short:>12}"
        stats_lines.append(header_line)
        stats_lines.append("")
        stats_lines.append("-" * 90)

        for brain in brain_names:
            short_brain = brain.split('/')[-1][:23] if '/' in brain else brain[:23]
            line = f"{short_brain:<25}"
            for group in reaching_groups[:5]:
                val = brains_elife[brain].get(group, 0)
                if group in region_stats and region_stats[group]['std'] > 0:
                    z = (val - region_stats[group]['mean']) / region_stats[group]['std']
                    line += f" {z:>+12.2f}"
                else:
                    line += f" {'N/A':>12}"
            stats_lines.append(line)

    # Reference to anatomical paradox paper
    stats_lines.append("\n" + "=" * 80)
    stats_lines.append("REFERENCES")
    stats_lines.append("=" * 80)
    stats_lines.append("")
    stats_lines.append("Anatomical Paradox:")
    stats_lines.append("  Fouad K, et al. (2021) 'The neuroanatomical-functional paradox in")
    stats_lines.append("  spinal cord injury' Nature Reviews Neurology 17:53-62")
    stats_lines.append("  DOI: 10.1038/s41582-020-00436-x")
    stats_lines.append("")
    stats_lines.append("Reference Data:")
    stats_lines.append("  Wang Z, et al. (2022) 'Brain-wide analysis of the supraspinal")
    stats_lines.append("  connectome reveals anatomical correlates to functional recovery")
    stats_lines.append("  after spinal injury' eLife 11:e76254")
    stats_lines.append("  DOI: 10.7554/eLife.76254")
    stats_lines.append("")
    stats_lines.append("=" * 80)

    # Write stats file
    stats_path = output_dir / 'statistical_summary.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
    print(f"    Saved: {stats_path.name}")

    # ==========================================================================
    # GRAPH 3: Cohort Comparison (if multiple cohorts)
    # ==========================================================================
    cohorts_with_data = {}
    for brain, data in brains_elife.items():
        cohort = data.get('_cohort', 'Unknown')
        if cohort not in cohorts_with_data:
            cohorts_with_data[cohort] = []
        cohorts_with_data[cohort].append(brain)

    if len(cohorts_with_data) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Cohort means for reaching regions
        ax = axes[0]
        cohort_means = {}
        for cohort, brains in cohorts_with_data.items():
            cohort_means[cohort] = {}
            for group in reaching_groups:
                values = [brains_elife[b].get(group, 0) for b in brains]
                cohort_means[cohort][group] = np.mean(values) if values else 0

        x = np.arange(len(reaching_groups))
        width = 0.8 / len(cohorts_with_data)

        for i, (cohort, means) in enumerate(cohort_means.items()):
            values = [means.get(g, 0) for g in reaching_groups]
            ax.bar(x + i * width - width * len(cohorts_with_data) / 2, values, width,
                   label=cohort or 'Unknown', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Brain Region')
        ax.set_ylabel('Mean Cell Count')
        ax.set_title('Cohort Comparison: Reaching Regions')
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace(' ', '\n')[:15] for g in reaching_groups], fontsize=8)
        ax.legend(title='Cohort')
        ax.grid(axis='y', alpha=0.3)

        # Right: Total cells by cohort
        ax = axes[1]
        cohort_totals = {c: [brains_elife[b]['_total'] for b in brains]
                         for c, brains in cohorts_with_data.items()}

        positions = np.arange(len(cohort_totals))
        for i, (cohort, totals) in enumerate(cohort_totals.items()):
            ax.bar(i, np.mean(totals), yerr=np.std(totals) if len(totals) > 1 else 0,
                   capsize=5, label=cohort, edgecolor='black', linewidth=0.5)
            # Scatter individual points
            ax.scatter([i] * len(totals), totals, color='black', s=30, zorder=5)

        ax.set_xlabel('Cohort')
        ax.set_ylabel('Total Cells')
        ax.set_title('Total Supraspinal Neurons by Cohort')
        ax.set_xticks(positions)
        ax.set_xticklabels(list(cohort_totals.keys()))
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        cohort_path = output_dir / 'cohort_comparison.png'
        plt.savefig(cohort_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {cohort_path.name}")

    # ==========================================================================
    # GRAPH 4: Hemisphere Breakdown (L/R per reaching region per brain)
    # ==========================================================================
    if has_hemisphere:
        fig, ax = plt.subplots(figsize=(14, 8))

        brain_names_list = list(brains_elife.keys())
        x = np.arange(len(reaching_groups))
        n_brains = len(brain_names_list)
        width = 0.8 / max(n_brains, 1)

        for i, brain in enumerate(brain_names_list):
            left_data = brains_elife[brain].get('_left', {})
            right_data = brains_elife[brain].get('_right', {})

            left_vals = [left_data.get(g, 0) for g in reaching_groups]
            right_vals = [right_data.get(g, 0) for g in reaching_groups]

            pos = x + i * width - width * n_brains / 2
            short_name = brain.split('/')[-1][:15] if '/' in brain else brain[:15]

            ax.bar(pos, left_vals, width, label=f'{short_name} L',
                   color=plt.cm.Blues(0.6), edgecolor='black', linewidth=0.5)
            ax.bar(pos, right_vals, width, bottom=left_vals, label=f'{short_name} R',
                   color=plt.cm.Reds(0.6), edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Brain Region (eLife Group)', fontsize=12)
        ax.set_ylabel('Cell Count', fontsize=12)
        ax.set_title('Hemisphere Breakdown: Key Reaching Regions\n'
                     '(Left = blue, Right = red, stacked per brain)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace(' ', '\n') for g in reaching_groups], fontsize=9, rotation=0)
        ax.legend(loc='upper right', fontsize=7, ncol=2, title='Brain / Side')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        hemi_graph = output_dir / 'hemisphere_breakdown.png'
        plt.savefig(hemi_graph, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {hemi_graph.name}")

    print(f"    All graphs saved to: {output_dir}")
    return output_dir


def generate_comparison_report(
    brain_name: str,
    region_counts: Dict[str, int],
    output_dir: Path,
    hemisphere_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Optional[Path]:
    """
    Generate a comparison report for routine pipeline runs.

    This report compares the current brain's counts to:
    1. Cervical (C4) reference data from eLife (total)
    2. Hemisphere breakdown (left vs right per key region)
    3. All other brains in the region_counts.csv database

    Args:
        brain_name: Name of the current brain
        region_counts: Dict mapping region acronym -> count
        output_dir: Where to save the report
        hemisphere_counts: Optional dict of {acronym: {'left': N, 'right': N}}

    Returns:
        Path to the generated report, or None if failed
    """
    from datetime import datetime

    # Import eLife mapping and reference data
    try:
        from elife_region_mapping import aggregate_to_elife, ELIFE_GROUPS
        from util_compare_to_published import CERVICAL_REFERENCE, KEY_RECOVERY_REGIONS
    except ImportError:
        print("    Warning: Could not import eLife mapping - skipping comparison report")
        return None

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"COMPARISON REPORT: {brain_name}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("=" * 80)

    # Aggregate to eLife groups
    elife_aggregated = aggregate_to_elife(region_counts)

    # Aggregate hemisphere data if available
    elife_left = {}
    elife_right = {}
    if hemisphere_counts:
        left_counts = {acr: d['left'] for acr, d in hemisphere_counts.items() if d.get('left', 0) > 0}
        right_counts = {acr: d['right'] for acr, d in hemisphere_counts.items() if d.get('right', 0) > 0}
        elife_left = aggregate_to_elife(left_counts)
        elife_right = aggregate_to_elife(right_counts)

    # ==========================================================================
    # SECTION 1: Total Comparison to Cervical Reference
    # ==========================================================================
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("PART 1: TOTAL COMPARISON TO CERVICAL (C4) REFERENCE")
    report_lines.append("Source: Wang et al. (2022) eLife, DOI: 10.7554/eLife.76254")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"{'eLife Group':<45} {'Yours':>8} {'Ref':>8} {'Diff':>8} {'%Ref':>8}")
    report_lines.append("-" * 80)

    total_yours = 0
    total_ref = 0
    key_regions_data = []

    # Sort by importance (key regions first)
    sorted_groups = sorted(CERVICAL_REFERENCE.keys())
    key_first = [g for g in sorted_groups if any(k in g for k in KEY_RECOVERY_REGIONS)]
    other = [g for g in sorted_groups if g not in key_first]
    sorted_groups = key_first + other

    for group in sorted_groups:
        ref_mean, ref_std, n = CERVICAL_REFERENCE[group]
        your_count = elife_aggregated.get(group, {}).get('count', 0)

        diff = your_count - ref_mean
        pct = (your_count / ref_mean * 100) if ref_mean > 0 else 0

        marker = " *" if any(k in group for k in KEY_RECOVERY_REGIONS) else ""
        report_lines.append(f"{group:<43}{marker} {your_count:>8} {ref_mean:>8} {diff:>+8} {pct:>7.1f}%")

        total_yours += your_count
        total_ref += ref_mean

        if any(k in group for k in KEY_RECOVERY_REGIONS):
            key_regions_data.append((group, your_count, ref_mean, pct))

    report_lines.append("-" * 80)
    report_lines.append(f"{'TOTAL':<45} {total_yours:>8} {total_ref:>8}")
    if total_ref > 0:
        report_lines.append(f"\nOverall: {total_yours/total_ref*100:.1f}% of cervical reference")

    # Key regions summary
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("KEY RECOVERY REGIONS (* above):")
    for group, yours, ref, pct in key_regions_data:
        report_lines.append(f"  {group}: {yours} / {ref} ({pct:.0f}%)")

    # ==========================================================================
    # SECTION 1b: Hemisphere Breakdown
    # ==========================================================================
    if hemisphere_counts:
        total_left = sum(d.get('left', 0) for d in hemisphere_counts.values())
        total_right = sum(d.get('right', 0) for d in hemisphere_counts.values())
        total_all = sum(region_counts.values())

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("PART 1b: HEMISPHERE BREAKDOWN (LEFT vs RIGHT)")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Total: {total_all:,} cells  |  Left: {total_left:,}  |  Right: {total_right:,}")
        if total_left + total_right > 0:
            lr_ratio = total_left / total_right if total_right > 0 else float('inf')
            report_lines.append(f"L/R ratio: {lr_ratio:.2f}")
        report_lines.append("")

        # Hemisphere breakdown per key recovery region
        report_lines.append(f"{'eLife Group':<40} {'Total':>8} {'Left':>8} {'Right':>8} {'L/R':>8}")
        report_lines.append("-" * 80)

        for group in sorted_groups:
            total_count = elife_aggregated.get(group, {}).get('count', 0)
            left_count = elife_left.get(group, {}).get('count', 0)
            right_count = elife_right.get(group, {}).get('count', 0)

            if total_count > 0:
                lr = f"{left_count/right_count:.2f}" if right_count > 0 else "inf"
                marker = " *" if any(k in group for k in KEY_RECOVERY_REGIONS) else ""
                report_lines.append(f"{group:<38}{marker} {total_count:>8} {left_count:>8} {right_count:>8} {lr:>8}")

        report_lines.append("-" * 80)
        lr_total = f"{total_left/total_right:.2f}" if total_right > 0 else "inf"
        report_lines.append(f"{'TOTAL':<40} {total_all:>8} {total_left:>8} {total_right:>8} {lr_total:>8}")
        report_lines.append("")
        report_lines.append("Note: L/R ratio ~1.0 indicates symmetric labeling.")
        report_lines.append("      Deviation may reflect injury laterality or injection asymmetry.")

    # ==========================================================================
    # SECTION 2: Comparison to Other Brains
    # ==========================================================================
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("PART 2: COMPARISON TO OTHER BRAINS IN DATABASE")
    report_lines.append("=" * 80)

    # Load other brains from region_counts.csv
    other_brains = {}
    if REGION_COUNTS_CSV.exists():
        with open(REGION_COUNTS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                other_name = row.get('brain', '')
                if other_name and other_name != brain_name:
                    # Parse region counts from row (exclude hemisphere columns)
                    counts = {}
                    for col, val in row.items():
                        if (col.startswith('region_')
                                and not col.startswith('region_left_')
                                and not col.startswith('region_right_')):
                            region = col[7:]  # Remove 'region_' prefix
                            try:
                                counts[region] = int(val) if val else 0
                            except ValueError:
                                pass
                    other_brains[other_name] = counts

    if not other_brains:
        report_lines.append("")
        report_lines.append("No other brains in database yet for comparison.")
        report_lines.append("Process more brains through the pipeline to enable cross-brain comparison.")
    else:
        report_lines.append(f"\nComparing to {len(other_brains)} other brain(s) in database:")
        report_lines.append("")

        # Header row
        brain_names = list(other_brains.keys())[:5]  # Limit to 5 for readability
        header = f"{'eLife Group':<35}"
        header += f"{'THIS':>10}"
        for name in brain_names:
            short_name = name.split('/')[0][:8] if '/' in name else name[:8]
            header += f"{short_name:>10}"
        report_lines.append(header)
        report_lines.append("-" * (35 + 10 + 10 * len(brain_names)))

        # Compare key regions
        for group in key_first:  # Only key recovery regions
            your_count = elife_aggregated.get(group, {}).get('count', 0)
            row = f"{group:<35}{your_count:>10}"

            for other_name in brain_names:
                other_counts = other_brains[other_name]
                other_elife = aggregate_to_elife(other_counts)
                other_val = other_elife.get(group, {}).get('count', 0)
                row += f"{other_val:>10}"

            report_lines.append(row)

        report_lines.append("")
        report_lines.append("Note: Only showing key recovery regions. Full data in region_counts.csv")

    # ==========================================================================
    # SECTION 3: Summary
    # ==========================================================================
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total cells counted: {sum(region_counts.values())}")
    if hemisphere_counts:
        total_left = sum(d.get('left', 0) for d in hemisphere_counts.values())
        total_right = sum(d.get('right', 0) for d in hemisphere_counts.values())
        report_lines.append(f"  Left hemisphere:  {total_left:,}")
        report_lines.append(f"  Right hemisphere: {total_right:,}")
    report_lines.append(f"Regions with cells: {len([v for v in region_counts.values() if v > 0])}")
    report_lines.append(f"eLife groups with cells: {len([g for g in elife_aggregated if g != '[Unmapped]' and elife_aggregated[g].get('count', 0) > 0])}")
    report_lines.append("")
    report_lines.append("Files generated:")
    report_lines.append(f"  - cell_counts_by_region.csv (full Allen Atlas detail, with L/R)")
    report_lines.append(f"  - cell_counts_elife_grouped.csv (eLife 25-group summary, with L/R)")
    report_lines.append(f"  - comparison_report.txt (this file)")
    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Also save CSV version for easier analysis (with hemisphere data)
    csv_report_path = output_dir / "comparison_to_reference.csv"
    with open(csv_report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['elife_group', 'your_count', 'cervical_ref', 'pct_of_ref', 'is_key_region']
        if hemisphere_counts:
            header.extend(['left_count', 'right_count', 'lr_ratio'])
        writer.writerow(header)

        for group in sorted_groups:
            ref_mean, ref_std, n = CERVICAL_REFERENCE[group]
            your_count = elife_aggregated.get(group, {}).get('count', 0)
            pct = (your_count / ref_mean * 100) if ref_mean > 0 else 0
            is_key = any(k in group for k in KEY_RECOVERY_REGIONS)
            row = [group, your_count, ref_mean, f"{pct:.1f}", 'Yes' if is_key else 'No']
            if hemisphere_counts:
                left_c = elife_left.get(group, {}).get('count', 0)
                right_c = elife_right.get(group, {}).get('count', 0)
                lr = f"{left_c/right_c:.2f}" if right_c > 0 else ""
                row.extend([left_c, right_c, lr])
            writer.writerow(row)

    return report_path


def parse_region_counts_from_csv(csv_path: Path) -> tuple:
    """
    Parse the per-brain CSV output into region counts with hemisphere data.

    Returns:
        (region_counts, hemisphere_counts) where:
        - region_counts: Dict[str, int] — {acronym: total_count} (backward compatible)
        - hemisphere_counts: Dict[str, Dict[str, int]] — {acronym: {'left': N, 'right': N}}
          Empty dict if hemisphere data not available in the CSV.
    """
    counts = {}
    hemi_counts = {}

    if not csv_path.exists():
        return counts, hemi_counts

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        has_hemisphere = False

        for row in reader:
            # Try different column name conventions
            region = None
            count = None

            # Format 1: region_acronym + cell_count
            for region_col in ['region_acronym', 'acronym', 'region', 'structure']:
                if region_col in row:
                    region = row[region_col]
                    break

            for count_col in ['cell_count', 'count', 'cells', 'n_cells']:
                if count_col in row:
                    try:
                        count = int(row[count_col])
                    except (ValueError, TypeError):
                        continue
                    break

            if region and count is not None:
                counts[region] = counts.get(region, 0) + count

                # Parse hemisphere columns if present
                left_val = row.get('left_count', '')
                right_val = row.get('right_count', '')
                if left_val or right_val:
                    has_hemisphere = True
                    try:
                        left = int(left_val) if left_val else 0
                    except (ValueError, TypeError):
                        left = 0
                    try:
                        right = int(right_val) if right_val else 0
                    except (ValueError, TypeError):
                        right = 0

                    if region not in hemi_counts:
                        hemi_counts[region] = {'left': 0, 'right': 0}
                    hemi_counts[region]['left'] += left
                    hemi_counts[region]['right'] += right

        if not has_hemisphere:
            hemi_counts = {}

    return counts, hemi_counts


def interactive_select_brain(brains):
    """Interactive brain selection."""
    print("\n" + "=" * 60)
    print("BRAINS WITH CLASSIFIED CELLS")
    print("=" * 60)
    
    ready = []
    already_done = []
    
    for i, brain in enumerate(brains):
        if brain['has_counts']:
            already_done.append((i, brain['name'], brain['n_cells']))
        else:
            ready.append((i, brain['name'], brain['n_cells']))
    
    if ready:
        print("\n[READY FOR COUNTING]")
        for idx, name, n in ready:
            print(f"  {idx + 1}. {name} ({n} cells)")
    
    if already_done:
        print("\n[ALREADY COUNTED]")
        for idx, name, n in already_done:
            print(f"  {idx + 1}. {name} ({n} cells)")
    
    print("\n" + "-" * 60)
    print("Enter number to select, 'all' for all ready, or 'q' to quit")
    print("-" * 60)
    
    while True:
        response = input("\nSelection: ").strip().lower()
        
        if response == 'q':
            return None
        
        if response == 'all':
            return [brains[idx] for idx, _, _ in ready]
        
        try:
            idx = int(response) - 1
            if 0 <= idx < len(brains):
                return [brains[idx]]
            else:
                print(f"Invalid number. Enter 1-{len(brains)}")
        except ValueError:
            print("Enter a number, 'all', or 'q'")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run regional cell counting with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 6_count_regions.py                                    # Interactive
    python 6_count_regions.py --brain 349_CNT_01_02_1p625x_z4
    python 6_count_regions.py --all                              # All pending
        """
    )
    
    parser.add_argument('--brain', '-b', help='Brain/pipeline to process')
    parser.add_argument('--all', '-a', action='store_true', help='Process all pending')
    parser.add_argument('--classification', '-c', help='Classification experiment ID to link to')
    
    parser.add_argument('--atlas', default='allen_mouse_10um')
    parser.add_argument('--notes', help='Notes to add to log')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrainGlobe Regional Cell Counting")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Select brain(s)
    to_process = []
    
    if args.brain:
        pipeline_folder, mouse_folder = find_pipeline(args.brain, args.root)
        if not pipeline_folder:
            print(f"ERROR: Brain not found: {args.brain}")
            sys.exit(1)
        
        cells_xml = find_classified_cells(pipeline_folder)
        if not cells_xml:
            print(f"ERROR: No classified cells found for {args.brain}")
            print("Run Script 5 (5_classify_cells.py) first!")
            sys.exit(1)
        
        to_process = [{
            'name': f"{mouse_folder.name}/{pipeline_folder.name}",
            'pipeline': pipeline_folder,
            'mouse': mouse_folder,
            'cells_xml': cells_xml,
            'n_cells': count_cells_in_xml(cells_xml),
        }]
    
    elif args.all:
        brains = list_available_brains(args.root)
        to_process = [b for b in brains if not b['has_counts']]
        if not to_process:
            print("\nNo brains need counting. All done!")
            return
    
    else:
        # Interactive selection
        brains = list_available_brains(args.root)
        if not brains:
            print("\nNo brains with classified cells found!")
            print("Run Script 5 (5_classify_cells.py) first!")
            sys.exit(1)
        
        to_process = interactive_select_brain(brains)
        if not to_process:
            print("Cancelled.")
            return
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        for brain in to_process:
            print(f"  Would count: {brain['name']} ({brain['n_cells']} cells)")
        return
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    print(f"\nWill process {len(to_process)} brain(s)")
    
    success_count = 0
    failed_count = 0
    
    for i, brain in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] {brain['name']}")
        print("-" * 50)
        
        pipeline_folder = brain['pipeline']
        reg_folder = pipeline_folder / FOLDER_REGISTRATION
        output_path = pipeline_folder / FOLDER_ANALYSIS
        
        exp_id = tracker.log_counts(
            brain=brain['name'],
            cells_path=str(brain['cells_xml']),
            atlas=args.atlas,
            output_path=str(output_path),
            parent_experiment=args.classification,
            notes=args.notes,
            status="started",
            script_version=SCRIPT_VERSION,
        )
        
        success, duration, total_cells, output_csv = run_brainglobe_counts(
            cells_path=brain['cells_xml'],
            registration_path=reg_folder,
            output_path=output_path,
            atlas=args.atlas,
        )
        
        tracker.update_status(
            exp_id,
            status="completed" if success else "failed",
            duration_seconds=round(duration, 1),
            count_total_cells=total_cells,
            count_output_csv=str(output_csv) if output_csv else None,
        )
        
        if success:
            success_count += 1
            print(f"\n    [OK] Completed - {total_cells} cells counted")
            if output_csv:
                print(f"    Output: {output_csv}")

            # Copy to summary directory
            if output_csv and output_csv.exists():
                summary_csv = SUMMARY_DATA_DIR / f"{brain['name'].replace('/', '_')}_counts.csv"
                summary_csv.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(output_csv, summary_csv)
                print(f"    Summary: {summary_csv}")

                # Update wide-format region_counts.csv (production tracking)
                # Parse the per-cell CSV to get region counts + hemisphere data
                region_counts, hemisphere_counts = parse_region_counts_from_csv(output_csv)
                if region_counts:
                    detection_params = {
                        'atlas': args.atlas,
                        # Note: detection params should come from parent experiment
                        # For now, we record what's available
                    }
                    region_csv = update_region_counts_csv(
                        brain_name=brain['name'],
                        region_counts=region_counts,
                        detection_params=detection_params,
                        hemisphere_counts=hemisphere_counts if hemisphere_counts else None,
                    )
                    print(f"    Region counts: {region_csv}")

                    # Also update eLife-grouped counts (with same archiving)
                    elife_csv = update_elife_counts_csv(
                        brain_name=brain['name'],
                        region_counts=region_counts,
                        detection_params=detection_params,
                        hemisphere_counts=hemisphere_counts if hemisphere_counts else None,
                    )
                    if elife_csv:
                        print(f"    eLife counts: {elife_csv}")

                    # Generate comparison report (vs cervical reference + other brains)
                    report_path = generate_comparison_report(
                        brain_name=brain['name'],
                        region_counts=region_counts,
                        output_dir=output_path,
                        hemisphere_counts=hemisphere_counts if hemisphere_counts else None,
                    )
                    if report_path:
                        print(f"    Comparison report: {report_path}")

                    # Generate/update cross-brain graphs and statistics
                    graphs_dir = generate_cross_brain_graphs()
                    if graphs_dir:
                        print(f"    Cross-brain graphs: {graphs_dir}")
                else:
                    print(f"    Warning: Could not parse region counts from {output_csv}")
        else:
            failed_count += 1
            print(f"\n    [FAILED]")
    
    print(f"\n{'='*60}")
    print(f"Complete: {success_count} succeeded, {failed_count} failed")
    print(f"{'='*60}")

    if success_count > 0:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print("\nYour data is ready! Results are in:")
        print(f"\nPer-brain outputs (in each 6_Region_Analysis folder):")
        print(f"  - cell_counts_by_region.csv    (full Allen Atlas detail)")
        print(f"  - cell_counts_elife_grouped.csv (eLife 25-group summary)")
        print(f"  - comparison_report.txt        (vs cervical reference + other brains)")
        print(f"  - comparison_to_reference.csv  (CSV format for analysis)")
        print(f"\nAggregated data (in {SUMMARY_DATA_DIR}):")
        print(f"  - region_counts.csv            (all brains, wide format)")
        print(f"  - elife_counts.csv             (all brains, eLife groups)")
        print(f"\nGraphs and statistics (in {SUMMARY_DATA_DIR / 'reports'}):")
        print(f"  - reaching_regions_comparison.png  (key forelimb regions)")
        print(f"  - elife_groups_heatmap.png         (all regions, all brains)")
        print(f"  - hemisphere_breakdown.png         (left/right per region per brain)")
        print(f"  - cohort_comparison.png            (if multiple cohorts)")
        print(f"  - statistical_summary.txt          (stats, z-scores, tract analysis)")
        print("\nTo compare brains:")
        print("   python util_compare_to_published.py --brain <name>")
        print("\n" + "-" * 60)
        print("OR use napari: Plugins > SCI-Connectome > Setup & Tuning > Results tab")
        print("=" * 60)


if __name__ == '__main__':
    main()
