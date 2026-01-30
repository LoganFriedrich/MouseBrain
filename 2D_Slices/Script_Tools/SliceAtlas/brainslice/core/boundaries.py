"""
boundaries.py - Atlas region boundary extraction and warping

Provides functions for:
- Extracting region boundaries as contours
- Warping contours using deformation fields
- Converting boundaries to napari Shapes layer format
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from skimage.measure import find_contours
from scipy.ndimage import map_coordinates


def extract_region_boundaries(
    annotation: np.ndarray,
    deformation_field: Optional[np.ndarray] = None,
    region_ids: Optional[List[int]] = None,
    simplify_tolerance: float = 1.0,
) -> Dict[int, List[np.ndarray]]:
    """
    Extract region boundaries as contours from atlas annotation.

    Args:
        annotation: 2D atlas annotation array (region IDs per pixel)
        deformation_field: Optional (H, W, 2) displacement vectors for warping
        region_ids: Optional list of region IDs to extract (None = all non-zero)
        simplify_tolerance: Tolerance for contour simplification (higher = simpler)

    Returns:
        Dict mapping region_id -> list of contour arrays (each N×2 in row, col format)
    """
    if region_ids is None:
        region_ids = np.unique(annotation)
        region_ids = region_ids[region_ids > 0]  # Exclude background (0)

    boundaries = {}

    for rid in region_ids:
        # Create binary mask for this region
        mask = (annotation == rid).astype(float)

        # Find contours at 0.5 level (boundary between 0 and 1)
        contours = find_contours(mask, 0.5)

        if not contours:
            continue

        # Simplify contours to reduce point count
        if simplify_tolerance > 0:
            contours = [simplify_contour(c, simplify_tolerance) for c in contours]

        # Warp contours if deformation field provided
        if deformation_field is not None:
            warped_contours = []
            for contour in contours:
                warped = warp_points(contour, deformation_field)
                warped_contours.append(warped)
            contours = warped_contours

        # Filter out very small contours (noise)
        contours = [c for c in contours if len(c) >= 4]

        if contours:
            boundaries[rid] = contours

    return boundaries


def simplify_contour(
    contour: np.ndarray,
    tolerance: float = 1.0,
) -> np.ndarray:
    """
    Simplify contour using Douglas-Peucker algorithm.

    Args:
        contour: Nx2 array of points
        tolerance: Simplification tolerance (higher = fewer points)

    Returns:
        Simplified contour
    """
    if len(contour) < 4:
        return contour

    try:
        from skimage.measure import approximate_polygon
        simplified = approximate_polygon(contour, tolerance=tolerance)
        return simplified
    except ImportError:
        # Fall back to no simplification
        return contour


def warp_points(
    points: np.ndarray,
    deformation_field: np.ndarray,
) -> np.ndarray:
    """
    Apply deformation field to point coordinates.

    Args:
        points: Nx2 array of points in (row, col) format
        deformation_field: (H, W, 2) displacement vectors where
                          [:,:,0] is dy and [:,:,1] is dx

    Returns:
        Warped points Nx2 array
    """
    if deformation_field is None or len(points) == 0:
        return points

    rows = points[:, 0]
    cols = points[:, 1]

    # Clamp coordinates to valid range
    h, w = deformation_field.shape[:2]
    rows_clamped = np.clip(rows, 0, h - 1)
    cols_clamped = np.clip(cols, 0, w - 1)

    # Sample displacement at each point (bilinear interpolation)
    dy = map_coordinates(
        deformation_field[:, :, 0],
        [rows_clamped, cols_clamped],
        order=1,
        mode='constant',
        cval=0,
    )
    dx = map_coordinates(
        deformation_field[:, :, 1],
        [rows_clamped, cols_clamped],
        order=1,
        mode='constant',
        cval=0,
    )

    # Apply displacement
    warped = points.copy()
    warped[:, 0] = rows + dy
    warped[:, 1] = cols + dx

    return warped


def boundaries_to_napari_shapes(
    boundaries: Dict[int, List[np.ndarray]],
    atlas_manager,
    shape_type: str = 'path',
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Convert boundaries to napari Shapes layer data.

    Args:
        boundaries: Dict from extract_region_boundaries
        atlas_manager: DualAtlasManager for region info
        shape_type: 'path' for lines, 'polygon' for filled shapes

    Returns:
        Tuple of (shapes_data, layer_kwargs) for napari add_shapes
    """
    shapes_data = []
    edge_colors = []
    face_colors = []
    properties = {'region_id': [], 'region_name': []}

    for rid, contours in boundaries.items():
        # Get region info for color
        try:
            region_info = atlas_manager.get_region_info(rid)
            rgb = region_info.get('rgb_triplet', [128, 128, 128])
            name = region_info.get('name', f'Region {rid}')
        except Exception:
            rgb = [128, 128, 128]
            name = f'Region {rid}'

        # Convert RGB to RGBA (0-1 range)
        edge_color = [c / 255 for c in rgb] + [1.0]
        face_color = [c / 255 for c in rgb] + [0.2]  # Transparent fill

        for contour in contours:
            # napari expects (N, 2) in (row, col) format - same as our format
            shapes_data.append(contour)
            edge_colors.append(edge_color)
            face_colors.append(face_color)
            properties['region_id'].append(rid)
            properties['region_name'].append(name)

    # Build layer kwargs
    layer_kwargs = {
        'shape_type': shape_type,
        'edge_color': edge_colors,
        'face_color': face_colors if shape_type == 'polygon' else 'transparent',
        'edge_width': 2,
        'name': 'Atlas Boundaries',
        'properties': properties,
    }

    return shapes_data, layer_kwargs


def extract_major_boundaries(
    annotation: np.ndarray,
    atlas_manager,
    deformation_field: Optional[np.ndarray] = None,
    hierarchy_level: int = 2,
) -> Dict[int, List[np.ndarray]]:
    """
    Extract boundaries for major brain regions only.

    Uses atlas hierarchy to get parent regions at specified level,
    reducing visual clutter compared to all fine-grained regions.

    Args:
        annotation: 2D atlas annotation
        atlas_manager: DualAtlasManager for hierarchy info
        deformation_field: Optional deformation to apply
        hierarchy_level: How many levels up in hierarchy (1=direct parent, 2=grandparent)

    Returns:
        Dict of boundaries for major regions
    """
    # Get all unique region IDs
    all_ids = np.unique(annotation)
    all_ids = all_ids[all_ids > 0]

    # Map each ID to its parent at specified level
    parent_map = {}
    for rid in all_ids:
        parent_id = atlas_manager.get_parent_region(rid, level=hierarchy_level)
        if parent_id is not None:
            parent_map[rid] = parent_id
        else:
            # Use original ID if no parent at that level
            parent_map[rid] = rid

    # Create merged annotation with parent IDs
    merged_annotation = annotation.copy()
    for rid, parent_id in parent_map.items():
        merged_annotation[annotation == rid] = parent_id

    # Extract boundaries from merged annotation
    parent_ids = list(set(parent_map.values()))
    return extract_region_boundaries(
        merged_annotation,
        deformation_field=deformation_field,
        region_ids=parent_ids,
    )


def get_brain_outline(
    annotation: np.ndarray,
    deformation_field: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Extract just the outer brain boundary.

    Args:
        annotation: 2D atlas annotation
        deformation_field: Optional deformation to apply

    Returns:
        Single contour array for brain outline, or None if not found
    """
    # Brain mask = any non-zero region
    brain_mask = (annotation > 0).astype(float)

    # Find contours
    contours = find_contours(brain_mask, 0.5)

    if not contours:
        return None

    # Get the largest contour (main brain outline)
    largest = max(contours, key=len)

    # Warp if needed
    if deformation_field is not None:
        largest = warp_points(largest, deformation_field)

    return largest
