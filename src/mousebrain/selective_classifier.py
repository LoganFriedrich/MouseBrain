"""
selective_classifier.py - Region-selective cell classification.

Instead of running the cellfinder classifier on ALL candidates, this module
maps each candidate to its atlas region and splits them into two groups:

  TRUST:    Cells in well-mapped eLife functional groups (cortex, thalamus,
            midbrain nuclei, etc.) -- these skip the classifier entirely.

  CLASSIFY: Cells in uncertain regions (fiber tracts, ventricles, meninges,
            unmapped/region_id=0, near brain surface) -- these go through
            the cellfinder classifier as usual.

The final output merges TRUST cells + classifier-approved CLASSIFY cells.

This preserves interior signal that full classification destroys while still
removing genuine artifacts in peripheral/ambiguous regions.

Usage:
    from mousebrain.selective_classifier import run_selective_classification

    result = run_selective_classification(
        brain_path=pipeline_folder,
        candidates_xml=prefiltered_xml,
        model_path=model_path,
        registration_path=reg_path,
    )
    # result.merged_cells_path -> cells_selective.xml
    # result.final_cells -> total cell count
"""

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mousebrain.region_mapping import (
    TracingType,
    get_elife_group,
    is_suspicious_region,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CellRegionInfo:
    """A cell candidate with its atlas region mapping."""
    z: int          # brain voxel coords (original)
    y: int
    x: int
    region_id: int
    acronym: str
    structure_name: str
    elife_group: Optional[str]
    hemisphere: str  # 'left' or 'right'
    atlas_coords: Tuple[float, float, float]
    out_of_bounds: bool = False


@dataclass
class SelectiveResult:
    """Results from selective classification."""
    total_candidates: int
    trust_count: int
    classify_count: int
    classified_cells: int       # from classifier: type==CELL
    classified_rejected: int    # from classifier: type==NO_CELL
    final_cells: int            # trust + classified_cells
    trust_cells_path: Optional[Path] = None
    classify_input_path: Optional[Path] = None
    merged_cells_path: Optional[Path] = None
    report_path: Optional[Path] = None
    category_breakdown: Dict[str, int] = field(default_factory=dict)
    classify_reasons: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


# =============================================================================
# COORDINATE TRANSFORM (shared, canonical implementation)
# =============================================================================

def transform_to_atlas_space(
    points_raw: np.ndarray,
    registration_path: Path,
    atlas_name: str = "allen_mouse_10um",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-step brainglobe coordinate transform: raw voxel -> atlas voxel.

    This is the ONLY correct way to map brain voxel coordinates to atlas
    coordinates. The simple scale+permute approach in prefilter.py is an
    approximation that does not account for non-rigid deformation.

    Steps:
        1. Raw -> Downsampled space (orientation + resolution scaling)
        2. Downsampled -> Atlas space (deformation fields, non-rigid warp)

    Args:
        points_raw: Nx3 array of (z, y, x) in brain voxel coordinates
        registration_path: Path to 3_Registered_Atlas/ folder
        atlas_name: BrainGlobe atlas identifier

    Returns:
        (points_atlas, points_out_of_bounds) - both Nx3 arrays
    """
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_utils.brainreg.transform import (
        transform_points_from_raw_to_downsampled_space,
        transform_points_from_downsampled_to_atlas_space,
    )
    import brainglobe_space as bgs
    import tifffile

    registration_path = Path(registration_path)

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

    # Load deformation fields
    deformation_fields = [
        registration_path / f"deformation_field_{i}.tiff"
        for i in range(3)
    ]
    for df in deformation_fields:
        if not df.exists():
            raise FileNotFoundError(f"Deformation field not found: {df}")

    # Load downsampled image shape
    downsampled_path = registration_path / "downsampled.tiff"
    if not downsampled_path.exists():
        raise FileNotFoundError(f"downsampled.tiff not found in {registration_path}")

    with tifffile.TiffFile(str(downsampled_path)) as tiff:
        downsampled_shape = tiff.series[0].shape

    # Build downsampled space
    atlas_resolution = atlas.resolution[0]
    downsampled_space = bgs.AnatomicalSpace(
        atlas.space.origin,
        shape=downsampled_shape,
        resolution=[atlas_resolution] * 3,
    )

    # Find source image stack for reference
    crop_folder = registration_path.parent / "2_Cropped_For_Registration"
    ch_folder = crop_folder / "ch0"
    if not ch_folder.exists():
        ch_folder = crop_folder / "ch1"
    if not (ch_folder.exists() and list(ch_folder.glob("Z*.tif"))):
        raise FileNotFoundError(f"Cannot find source image stack in {crop_folder}")
    source_image_path = ch_folder

    print(f"    Orientation: {orientation}, Voxel sizes: {voxel_sizes}")
    print(f"    Atlas: {atlas_name}, resolution: {atlas_resolution}um")
    print(f"    Downsampled shape: {downsampled_shape}")

    # Step 1: Raw -> Downsampled (orientation + scaling)
    print("    Step 1: Raw -> Downsampled space...")
    points_downsampled = transform_points_from_raw_to_downsampled_space(
        points_raw,
        source_image_path,
        orientation,
        voxel_sizes,
        downsampled_space,
    )

    # Step 2: Downsampled -> Atlas (deformation fields)
    print("    Step 2: Downsampled -> Atlas space (deformation fields)...")
    points_atlas, points_oob = transform_points_from_downsampled_to_atlas_space(
        points_downsampled,
        atlas,
        deformation_field_paths=deformation_fields,
        warn_out_of_bounds=False,
    )

    print(f"    Transformed {len(points_atlas)} points to atlas space")
    if len(points_oob) > 0:
        print(f"    ({len(points_oob)} points fell outside atlas bounds)")

    return points_atlas, points_oob


# =============================================================================
# REGION MAPPING
# =============================================================================

def parse_xml_coords(xml_path: Path) -> List[Tuple[int, int, int]]:
    """Parse (z, y, x) coordinates from CellCounter XML."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    coords = []
    for marker in root.iter('Marker'):
        x = int(marker.find('MarkerX').text)
        y = int(marker.find('MarkerY').text)
        z = int(marker.find('MarkerZ').text)
        coords.append((z, y, x))
    return coords


def map_cells_to_regions(
    candidates_xml: Path,
    registration_path: Path,
    atlas_name: str = "allen_mouse_10um",
) -> List[CellRegionInfo]:
    """
    Map each cell candidate to its atlas region using the proper 2-step
    brainglobe deformation field transform.

    Args:
        candidates_xml: Path to cell candidates XML
        registration_path: Path to 3_Registered_Atlas/
        atlas_name: BrainGlobe atlas identifier

    Returns:
        List of CellRegionInfo, one per candidate
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    atlas = BrainGlobeAtlas(atlas_name)

    # Parse coordinates from XML
    raw_coords = parse_xml_coords(candidates_xml)
    if not raw_coords:
        return []

    points_raw = np.array(raw_coords, dtype=np.float64)
    print(f"    Loaded {len(points_raw)} candidates from XML")

    # Transform to atlas space
    points_atlas, points_oob = transform_to_atlas_space(
        points_raw, registration_path, atlas_name
    )

    # Build set of out-of-bounds indices for quick lookup
    # points_oob contains the actual coordinate arrays of OOB points
    # We need to track which original indices mapped to OOB
    n_total = len(points_raw)
    n_in_bounds = len(points_atlas)
    n_oob = len(points_oob)

    # The brainglobe transform returns in-bounds and out-of-bounds separately
    # but doesn't tell us which original indices they correspond to.
    # We need to reconstruct this from the counts.
    # Actually, brainglobe returns them in order: first n_in_bounds are in-bounds,
    # remaining are OOB. But the function signature returns them split.
    # We'll handle this by mapping all in-bounds first, then marking the rest as OOB.

    cells = []

    # Map in-bounds cells
    for i, atlas_pt in enumerate(points_atlas):
        z_raw, y_raw, x_raw = int(points_raw[i][0]), int(points_raw[i][1]), int(points_raw[i][2])

        # Clamp to atlas bounds for region lookup
        atlas_pt_clamped = np.clip(
            atlas_pt,
            [0, 0, 0],
            [s - 1 for s in atlas.annotation.shape],
        ).astype(int)

        try:
            region_id = atlas.structure_from_coords(atlas_pt_clamped)
        except Exception:
            region_id = 0

        # Get structure info
        if region_id and region_id in atlas.structures:
            acronym = atlas.structures[region_id].get('acronym', '')
            structure_name = atlas.structures[region_id].get('name', '')
        else:
            acronym = ''
            structure_name = ''
            region_id = 0

        # Get hemisphere
        try:
            hemisphere = atlas.hemisphere_from_coords(atlas_pt_clamped, as_string=True)
        except Exception:
            hemisphere = 'unknown'

        elife_group = get_elife_group(acronym) if acronym else None

        cells.append(CellRegionInfo(
            z=z_raw, y=y_raw, x=x_raw,
            region_id=region_id,
            acronym=acronym,
            structure_name=structure_name,
            elife_group=elife_group,
            hemisphere=hemisphere,
            atlas_coords=tuple(atlas_pt.tolist()),
            out_of_bounds=False,
        ))

    # Mark OOB cells (if any remaining candidates weren't in points_atlas)
    # The brainglobe transform may have fewer outputs than inputs
    for i in range(n_in_bounds, n_total):
        z_raw, y_raw, x_raw = int(points_raw[i][0]), int(points_raw[i][1]), int(points_raw[i][2])
        cells.append(CellRegionInfo(
            z=z_raw, y=y_raw, x=x_raw,
            region_id=0,
            acronym='',
            structure_name='',
            elife_group=None,
            hemisphere='unknown',
            atlas_coords=(0.0, 0.0, 0.0),
            out_of_bounds=True,
        ))

    print(f"    Mapped {len(cells)} cells to atlas regions")
    return cells


# =============================================================================
# CATEGORIZATION
# =============================================================================

def categorize_cells(
    cells: List[CellRegionInfo],
    tracing_type: TracingType = TracingType.DESCENDING,
) -> Tuple[List[CellRegionInfo], List[CellRegionInfo], Dict[str, int], Dict[str, int]]:
    """
    Split cells into TRUST (skip classifier) and CLASSIFY (need classifier).

    CLASSIFY criteria (cells that NEED the classifier):
      - region_id == 0 (unmapped / outside atlas)
      - is_suspicious_region() returns non-None
      - out_of_bounds from coordinate transform

    TRUST criteria (cells that skip the classifier):
      - Maps to a real eLife functional group
      - Not in a suspicious region

    Args:
        cells: List of CellRegionInfo from map_cells_to_regions()
        tracing_type: Type of tracing experiment

    Returns:
        (trust_cells, classify_cells, trust_breakdown, classify_reasons)
        trust_breakdown: eLife group name -> count
        classify_reasons: reason string -> count
    """
    trust_cells = []
    classify_cells = []
    trust_breakdown = {}    # eLife group -> count
    classify_reasons = {}   # reason -> count

    for cell in cells:
        reason = _classify_reason(cell, tracing_type)

        if reason is None:
            # TRUST this cell
            trust_cells.append(cell)
            group = cell.elife_group or "[Other Legitimate]"
            trust_breakdown[group] = trust_breakdown.get(group, 0) + 1
        else:
            # CLASSIFY this cell
            classify_cells.append(cell)
            classify_reasons[reason] = classify_reasons.get(reason, 0) + 1

    return trust_cells, classify_cells, trust_breakdown, classify_reasons


def _classify_reason(cell: CellRegionInfo, tracing_type: TracingType) -> Optional[str]:
    """
    Determine if a cell needs classification. Returns reason string or None.

    Returns None if the cell should be TRUSTED (skip classifier).
    Returns a reason string if the cell needs CLASSIFICATION.
    """
    # Out-of-bounds cells always need classification
    if cell.out_of_bounds:
        return "out_of_bounds"

    # Unmapped cells (region_id=0) always need classification
    if cell.region_id == 0:
        return "unmapped"

    # Check if region is suspicious for this tracing type
    if cell.acronym:
        suspicious_cat = is_suspicious_region(cell.acronym, tracing_type)
        if suspicious_cat is not None:
            return f"suspicious:{suspicious_cat}"

    # Check if cell has a real eLife group
    elife = cell.elife_group
    if elife and elife != "[Unmapped]":
        # This cell is in a legitimate eLife region -- TRUST it
        return None

    # Cell is in a mapped region but not in any eLife group
    # These are ambiguous -- classify them to be safe
    return "no_elife_group"


# =============================================================================
# XML I/O (CellCounter format)
# =============================================================================

def _coords_to_xml(coords: List[Tuple[int, int, int]], output_path: Path):
    """Write (z, y, x) coordinates to CellCounter XML format."""
    root = ET.Element('CellCounter_Marker_File')
    image_props = ET.SubElement(root, 'Image_Properties')
    ET.SubElement(image_props, 'Image_Filename').text = 'cells'

    marker_data = ET.SubElement(root, 'Marker_Data')
    current_type = ET.SubElement(marker_data, 'Current_Type')
    current_type.text = '1'

    marker_type = ET.SubElement(marker_data, 'Marker_Type')
    type_elem = ET.SubElement(marker_type, 'Type')
    type_elem.text = '1'

    for z, y, x in coords:
        marker = ET.SubElement(marker_type, 'Marker')
        ET.SubElement(marker, 'MarkerX').text = str(x)
        ET.SubElement(marker, 'MarkerY').text = str(y)
        ET.SubElement(marker, 'MarkerZ').text = str(z)

    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding='unicode', xml_declaration=True)


def _merge_coords_to_xml(
    coords_list: List[List[Tuple[int, int, int]]],
    output_path: Path,
):
    """Merge multiple coordinate lists into a single XML."""
    all_coords = []
    for coords in coords_list:
        all_coords.extend(coords)
    _coords_to_xml(all_coords, output_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_selective_classification(
    brain_path: Path,
    candidates_xml: Path,
    model_path: Path,
    registration_path: Path,
    tracing_type: str = "descending",
    atlas_name: str = "allen_mouse_10um",
    cube_size: int = 50,
    batch_size: int = 32,
    n_free_cpus: int = 2,
    signal_path: Optional[Path] = None,
    background_path: Optional[Path] = None,
    voxel_sizes: Optional[tuple] = None,
) -> SelectiveResult:
    """
    Full selective classification pipeline.

    1. Map all candidates to atlas regions (2-step brainglobe transform)
    2. Categorize: TRUST vs CLASSIFY
    3. Run cellfinder classifier ONLY on CLASSIFY subset
    4. Merge: TRUST + classifier-approved CLASSIFY cells
    5. Save cells_selective.xml + report

    Args:
        brain_path: Path to brain pipeline folder
        candidates_xml: Path to candidate cells XML (prefiltered or raw)
        model_path: Path to trained classifier model
        registration_path: Path to 3_Registered_Atlas/
        tracing_type: "descending", "ascending", or "unknown"
        atlas_name: BrainGlobe atlas identifier
        cube_size: Classifier cube size
        batch_size: Classifier batch size
        n_free_cpus: CPUs to leave free for classifier
        signal_path: Path to signal channel folder (auto-detected if None)
        background_path: Path to background channel folder (auto-detected if None)
        voxel_sizes: (z, y, x) voxel sizes in microns (auto-detected if None)

    Returns:
        SelectiveResult with all counts and paths
    """
    start_time = time.time()

    # Parse tracing type
    tt = TracingType(tracing_type)

    output_dir = brain_path / "5_Classified_Cells"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"SELECTIVE CLASSIFICATION")
    print(f"{'=' * 60}")
    print(f"  Candidates: {candidates_xml}")
    print(f"  Registration: {registration_path}")
    print(f"  Tracing type: {tracing_type}")
    print(f"  Model: {model_path}")

    # -------------------------------------------------------------------------
    # Step 1: Map candidates to atlas regions
    # -------------------------------------------------------------------------
    print(f"\n--- Step 1: Mapping candidates to atlas regions ---")
    cells = map_cells_to_regions(candidates_xml, registration_path, atlas_name)

    if not cells:
        print("  No candidates found!")
        return SelectiveResult(
            total_candidates=0, trust_count=0, classify_count=0,
            classified_cells=0, classified_rejected=0, final_cells=0,
            duration_seconds=time.time() - start_time,
        )

    # -------------------------------------------------------------------------
    # Step 2: Categorize TRUST vs CLASSIFY
    # -------------------------------------------------------------------------
    print(f"\n--- Step 2: Categorizing cells ---")
    trust_cells, classify_cells, trust_breakdown, classify_reasons = categorize_cells(cells, tt)

    print(f"  Total candidates: {len(cells):,}")
    print(f"  TRUST (skip classifier): {len(trust_cells):,}")
    print(f"  CLASSIFY (need classifier): {len(classify_cells):,}")

    print(f"\n  TRUST breakdown by eLife group:")
    for group, count in sorted(trust_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {group}: {count:,}")

    print(f"\n  CLASSIFY reasons:")
    for reason, count in sorted(classify_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count:,}")

    # Save TRUST cells as XML (for auditing)
    trust_xml = output_dir / "trust_cells.xml"
    trust_coords = [(c.z, c.y, c.x) for c in trust_cells]
    _coords_to_xml(trust_coords, trust_xml)
    print(f"\n  Saved TRUST cells: {trust_xml}")

    # Save CLASSIFY cells as XML (input for classifier)
    classify_xml = output_dir / "classify_input.xml"
    classify_coords = [(c.z, c.y, c.x) for c in classify_cells]
    _coords_to_xml(classify_coords, classify_xml)
    print(f"  Saved CLASSIFY cells: {classify_xml}")

    # -------------------------------------------------------------------------
    # Step 3: Run cellfinder classifier on CLASSIFY subset only
    # -------------------------------------------------------------------------
    classified_cells_count = 0
    classified_rejected_count = 0
    classifier_approved_coords = []

    if classify_cells:
        print(f"\n--- Step 3: Running classifier on {len(classify_cells):,} CLASSIFY cells ---")

        try:
            from brainglobe_utils.cells.cells import Cell
            from brainglobe_utils.IO.cells import get_cells, save_cells
            from cellfinder.core.classify.classify import main as classify_main
            from brainglobe_utils.IO.image.load import read_with_dask

            # Auto-detect paths if not provided
            if signal_path is None:
                crop_folder = brain_path / "2_Cropped_For_Registration"
                signal_path = crop_folder / "ch0"
                if not signal_path.exists():
                    signal_path = crop_folder / "ch1"

            if background_path is None:
                crop_folder = brain_path / "2_Cropped_For_Registration"
                background_path = crop_folder / "ch1"
                if not background_path.exists():
                    background_path = crop_folder / "ch0"

            if voxel_sizes is None:
                metadata_path = brain_path / "2_Cropped_For_Registration" / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    voxel = metadata.get('voxel_size_um', {})
                    voxel_sizes = (
                        voxel.get('z', 4),
                        voxel.get('y', 4),
                        voxel.get('x', 4),
                    )
                else:
                    voxel_sizes = (4.0, 4.0, 4.0)

            # Load cells from the CLASSIFY XML
            classify_cell_objects = get_cells(str(classify_xml))
            print(f"  Loaded {len(classify_cell_objects)} cells for classification")

            # Load images
            signal_array = read_with_dask(str(signal_path))
            bg_array = None
            if background_path and background_path.exists():
                bg_array = read_with_dask(str(background_path))

            # Determine model type
            if model_path.suffix in ('.keras', '.h5'):
                trained_model_arg = model_path
                model_weights_arg = None
            else:
                trained_model_arg = None
                model_weights_arg = model_path

            # Run classifier
            print(f"  Running cellfinder classifier...")
            print(f"    Batch size: {batch_size}, Cube size: {cube_size}")
            print(f"    n_free_cpus: {n_free_cpus}")

            result_cells = classify_main(
                points=classify_cell_objects,
                signal_array=signal_array,
                background_array=bg_array,
                n_free_cpus=n_free_cpus,
                voxel_sizes=tuple(float(v) for v in voxel_sizes),
                network_voxel_sizes=(5.0, 1.0, 1.0),
                batch_size=batch_size,
                cube_height=cube_size,
                cube_width=cube_size,
                cube_depth=20,
                trained_model=trained_model_arg,
                model_weights=model_weights_arg,
                network_depth="50",
            )

            # Separate approved vs rejected
            approved = [c for c in result_cells if c.type == Cell.CELL]
            rejected = [c for c in result_cells if c.type == Cell.NO_CELL]

            classified_cells_count = len(approved)
            classified_rejected_count = len(rejected)

            print(f"  Classifier results:")
            print(f"    Approved: {classified_cells_count:,}")
            print(f"    Rejected: {classified_rejected_count:,}")

            # Extract coordinates from approved cells
            for c in approved:
                classifier_approved_coords.append((c.z, c.y, c.x))

            # Save rejected for auditing
            rejected_xml = output_dir / "selective_rejected.xml"
            save_cells(rejected, str(rejected_xml))

        except ImportError as e:
            print(f"  ERROR: cellfinder not available: {e}")
            print(f"  Treating all CLASSIFY cells as rejected (conservative)")
            classified_rejected_count = len(classify_cells)

        except Exception as e:
            import traceback
            print(f"  ERROR during classification: {e}")
            traceback.print_exc()
            print(f"  Treating all CLASSIFY cells as rejected (conservative)")
            classified_rejected_count = len(classify_cells)
    else:
        print(f"\n--- Step 3: No CLASSIFY cells -- skipping classifier ---")

    # -------------------------------------------------------------------------
    # Step 4: Merge TRUST + classifier-approved CLASSIFY
    # -------------------------------------------------------------------------
    print(f"\n--- Step 4: Merging results ---")

    merged_coords = trust_coords + classifier_approved_coords
    final_count = len(merged_coords)

    merged_xml = output_dir / "cells_selective.xml"
    _coords_to_xml(merged_coords, merged_xml)

    print(f"  TRUST cells:              {len(trust_coords):,}")
    print(f"  Classifier-approved:      {classified_cells_count:,}")
    print(f"  Classifier-rejected:      {classified_rejected_count:,}")
    print(f"  FINAL (merged):           {final_count:,}")
    print(f"  Saved: {merged_xml}")

    # -------------------------------------------------------------------------
    # Step 5: Save report
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    report = {
        "total_candidates": len(cells),
        "trust_count": len(trust_cells),
        "classify_count": len(classify_cells),
        "classified_cells": classified_cells_count,
        "classified_rejected": classified_rejected_count,
        "final_cells": final_count,
        "tracing_type": tracing_type,
        "atlas_name": atlas_name,
        "trust_breakdown": trust_breakdown,
        "classify_reasons": classify_reasons,
        "duration_seconds": round(duration, 1),
        "model_path": str(model_path),
        "candidates_xml": str(candidates_xml),
    }

    report_path = output_dir / "selective_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"SELECTIVE CLASSIFICATION COMPLETE in {duration:.1f}s")
    print(f"  {len(cells):,} candidates -> {final_count:,} cells")
    print(f"  ({len(trust_cells):,} trusted + {classified_cells_count:,} classifier-approved)")
    print(f"{'=' * 60}")

    return SelectiveResult(
        total_candidates=len(cells),
        trust_count=len(trust_cells),
        classify_count=len(classify_cells),
        classified_cells=classified_cells_count,
        classified_rejected=classified_rejected_count,
        final_cells=final_count,
        trust_cells_path=trust_xml,
        classify_input_path=classify_xml,
        merged_cells_path=merged_xml,
        report_path=report_path,
        category_breakdown=trust_breakdown,
        classify_reasons=classify_reasons,
        duration_seconds=duration,
    )
