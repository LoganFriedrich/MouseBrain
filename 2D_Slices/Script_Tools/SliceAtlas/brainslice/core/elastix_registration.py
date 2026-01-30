"""
elastix_registration.py - SimpleElastix-based hierarchical registration

Provides robust 2D slice-to-atlas registration using:
1. Affine registration (translation, rotation, scale, shear)
2. B-spline deformation (local non-rigid warping)

Uses SimpleITK with SimpleElastix extension for industry-standard
registration quality.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

# Lazy import for SimpleITK
_sitk = None
_elastix_available = None


def _check_elastix():
    """Check if SimpleElastix is available."""
    global _sitk, _elastix_available
    if _elastix_available is None:
        try:
            import SimpleITK as sitk
            # Check if elastix is available
            _ = sitk.ElastixImageFilter()
            _sitk = sitk
            _elastix_available = True
        except (ImportError, AttributeError):
            _elastix_available = False
    return _elastix_available


def is_elastix_available() -> bool:
    """Check if SimpleElastix is installed and available."""
    return _check_elastix()


def register_slice_to_atlas(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    grid_spacing: int = 8,
    max_iterations: int = 500,
    use_bspline: bool = True,
) -> Dict[str, Any]:
    """
    Register brain slice to atlas using hierarchical registration.

    Pipeline: Affine → B-spline (optional)

    Args:
        moving_image: 2D brain slice (to be warped)
        fixed_image: 2D atlas reference slice (target)
        grid_spacing: B-spline control point spacing in voxels
        max_iterations: Maximum optimizer iterations per stage
        use_bspline: Whether to use B-spline deformation (slower but better)

    Returns:
        Dict with:
            'registered_image': Warped moving image
            'transform_params': List of transform parameter maps
            'deformation_field': (H, W, 2) displacement vectors (dy, dx)
            'affine_params': Extracted affine parameters
            'success': Whether registration succeeded
    """
    if not _check_elastix():
        raise ImportError(
            "SimpleITK-SimpleElastix is not installed. "
            "Install with: pip install SimpleITK-SimpleElastix"
        )

    sitk = _sitk

    # Ensure same size
    if moving_image.shape != fixed_image.shape:
        from skimage.transform import resize
        moving_image = resize(
            moving_image,
            fixed_image.shape,
            preserve_range=True,
            anti_aliasing=True,
        )

    # Convert to SimpleITK images
    moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
    fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))

    # Setup elastix
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_sitk)
    elastix.SetMovingImage(moving_sitk)
    elastix.SetLogToConsole(False)

    # Stage 1: Affine registration
    affine_params = sitk.GetDefaultParameterMap("affine")
    affine_params["MaximumNumberOfIterations"] = [str(max_iterations)]
    affine_params["NumberOfResolutions"] = ["4"]
    affine_params["Metric"] = ["AdvancedNormalizedCorrelation"]
    elastix.SetParameterMap(affine_params)

    # Stage 2: B-spline deformation (optional)
    if use_bspline:
        bspline_params = sitk.GetDefaultParameterMap("bspline")
        bspline_params["FinalGridSpacingInVoxels"] = [str(grid_spacing), str(grid_spacing)]
        bspline_params["GridSpacingSchedule"] = ["4.0", "2.0", "1.0"]
        bspline_params["NumberOfResolutions"] = ["3"]
        bspline_params["MaximumNumberOfIterations"] = [str(max_iterations)]
        bspline_params["Metric"] = ["AdvancedNormalizedCorrelation"]
        elastix.AddParameterMap(bspline_params)

    # Execute registration
    try:
        elastix.Execute()
    except Exception as e:
        print(f"[Elastix] Registration failed: {e}")
        return {
            'registered_image': moving_image,
            'transform_params': None,
            'deformation_field': None,
            'affine_params': None,
            'success': False,
            'error': str(e),
        }

    # Get results
    registered = sitk.GetArrayFromImage(elastix.GetResultImage())
    transform_params = elastix.GetTransformParameterMap()

    # Extract deformation field
    deformation_field = _extract_deformation_field(
        sitk, transform_params, moving_sitk, fixed_image.shape
    )

    # Extract affine parameters
    affine_params_dict = _extract_affine_params(transform_params[0])

    return {
        'registered_image': registered,
        'transform_params': transform_params,
        'deformation_field': deformation_field,
        'affine_params': affine_params_dict,
        'success': True,
    }


def _extract_deformation_field(
    sitk,
    transform_params,
    moving_image,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Extract deformation field from transform parameters."""
    try:
        transformix = sitk.TransformixImageFilter()
        transformix.ComputeDeformationFieldOn()
        transformix.SetLogToConsole(False)

        # Use the last transform (B-spline if available, else affine)
        transformix.SetTransformParameterMap(transform_params[-1])
        transformix.SetMovingImage(moving_image)
        transformix.Execute()

        deform_sitk = transformix.GetDeformationField()
        deform_array = sitk.GetArrayFromImage(deform_sitk)

        # deform_array shape is (H, W, 2) with (dx, dy) per pixel
        # We want (H, W, 2) with (dy, dx) to match row, col convention
        if deform_array.ndim == 3 and deform_array.shape[2] == 2:
            # Swap dx, dy to dy, dx
            deformation_field = np.stack([
                deform_array[:, :, 1],  # dy
                deform_array[:, :, 0],  # dx
            ], axis=-1)
        else:
            deformation_field = deform_array

        return deformation_field

    except Exception as e:
        print(f"[Elastix] Could not extract deformation field: {e}")
        return np.zeros((*output_shape, 2), dtype=np.float32)


def _extract_affine_params(param_map) -> Dict[str, Any]:
    """Extract human-readable affine parameters from transform."""
    try:
        transform_params = param_map.get("TransformParameters", [])
        if len(transform_params) >= 6:
            # Affine transform: [a00, a01, a10, a11, tx, ty]
            params = [float(p) for p in transform_params]

            # Calculate rotation and scale from matrix elements
            import math
            a00, a01, a10, a11 = params[0], params[1], params[2], params[3]
            tx, ty = params[4], params[5]

            # Rotation angle (in degrees)
            rotation = math.degrees(math.atan2(a10, a00))

            # Scale (approximate)
            scale_x = math.sqrt(a00**2 + a10**2)
            scale_y = math.sqrt(a01**2 + a11**2)
            scale = (scale_x + scale_y) / 2

            return {
                'translation': (ty, tx),  # row, col convention
                'rotation_deg': rotation,
                'scale': scale,
                'matrix': [[a00, a01], [a10, a11]],
            }
    except Exception as e:
        print(f"[Elastix] Could not extract affine params: {e}")

    return {
        'translation': (0, 0),
        'rotation_deg': 0,
        'scale': 1.0,
        'matrix': [[1, 0], [0, 1]],
    }


def register_affine_only(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    max_iterations: int = 500,
) -> Dict[str, Any]:
    """
    Affine-only registration (faster, no local deformation).

    Args:
        moving_image: Image to warp
        fixed_image: Target image
        max_iterations: Max optimizer iterations

    Returns:
        Registration result dict
    """
    return register_slice_to_atlas(
        moving_image,
        fixed_image,
        use_bspline=False,
        max_iterations=max_iterations,
    )


def apply_transform_to_labels(
    labels: np.ndarray,
    transform_params,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Apply registration transform to label image (atlas annotation).

    Uses nearest-neighbor interpolation to preserve label IDs.

    Args:
        labels: 2D label image (integer region IDs)
        transform_params: Transform from registration
        output_shape: Output image shape

    Returns:
        Warped label image
    """
    if not _check_elastix():
        return labels

    sitk = _sitk

    # Convert labels to SimpleITK
    labels_sitk = sitk.GetImageFromArray(labels.astype(np.float32))

    # Setup transformix
    transformix = sitk.TransformixImageFilter()
    transformix.SetLogToConsole(False)

    # Modify transform to use nearest neighbor interpolation
    for param_map in transform_params:
        param_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

    transformix.SetTransformParameterMap(transform_params)
    transformix.SetMovingImage(labels_sitk)
    transformix.Execute()

    warped_labels = sitk.GetArrayFromImage(transformix.GetResultImage())

    return warped_labels.astype(labels.dtype)


def warp_atlas_to_image(
    atlas_reference: np.ndarray,
    atlas_annotation: np.ndarray,
    image: np.ndarray,
    grid_spacing: int = 8,
) -> Dict[str, Any]:
    """
    High-level function: Register atlas to match a brain slice image.

    The ATLAS is warped to match the IMAGE (atlas moves to image space).

    Args:
        atlas_reference: Atlas reference slice
        atlas_annotation: Atlas annotation slice (region IDs)
        image: User's brain slice image
        grid_spacing: B-spline control point spacing

    Returns:
        Dict with warped atlas reference, annotation, and deformation field
    """
    # Preprocess both images for registration
    from .preprocessing import preprocess_for_registration, preprocess_atlas_slice

    image_proc, _ = preprocess_for_registration(image)
    atlas_proc = preprocess_atlas_slice(atlas_reference)

    # Register: atlas (moving) → image (fixed)
    result = register_slice_to_atlas(
        moving_image=atlas_proc,
        fixed_image=image_proc,
        grid_spacing=grid_spacing,
    )

    if not result['success']:
        return result

    # Warp the actual atlas images (not preprocessed) using the transform
    if result['transform_params'] is not None:
        warped_reference = apply_transform_to_labels(
            atlas_reference,
            result['transform_params'],
            image.shape,
        )
        warped_annotation = apply_transform_to_labels(
            atlas_annotation,
            result['transform_params'],
            image.shape,
        )
    else:
        warped_reference = atlas_reference
        warped_annotation = atlas_annotation

    return {
        'warped_reference': warped_reference,
        'warped_annotation': warped_annotation,
        'deformation_field': result['deformation_field'],
        'affine_params': result['affine_params'],
        'success': True,
    }
