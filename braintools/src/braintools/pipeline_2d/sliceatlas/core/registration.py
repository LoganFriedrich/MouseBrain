"""
registration.py - Automated 2D slice-to-atlas registration for BrainSlice

Provides functions for:
- Auto-detecting which atlas slice best matches an image
- Computing 2D transforms to align image to atlas
- Applying transforms to warp atlas labels to image coordinates

No external dependencies beyond scikit-image and scipy.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np

from skimage.metrics import normalized_mutual_information
from skimage.registration import phase_cross_correlation
from skimage.transform import (
    SimilarityTransform,
    AffineTransform,
    estimate_transform,
    warp,
    resize,
)
from scipy.ndimage import affine_transform as scipy_affine_transform


def find_best_atlas_slice(
    image: np.ndarray,
    atlas_manager,
    orientation: str = 'coronal',
    atlas_type: str = 'brain',
    step: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Scan atlas slices to find the best matching position for an image.

    Uses normalized mutual information which is robust to:
    - Different intensity scales
    - Different contrast
    - Partial overlap

    Args:
        image: 2D grayscale image to match (Y, X)
        atlas_manager: DualAtlasManager instance with loaded atlas
        orientation: 'coronal', 'sagittal', or 'horizontal'
        atlas_type: 'brain' or 'spinal_cord'
        step: Step size for scanning (higher = faster but less precise)
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Dict with:
            'best_position_idx': Best matching atlas slice index
            'best_score': Matching score (higher is better, typically 1.0-2.0)
            'reference_slice': Atlas reference image at best position
            'annotation_slice': Atlas labels at best position
            'n_slices': Total number of slices scanned
    """
    atlas = atlas_manager.get_atlas(atlas_type)

    # Determine number of slices based on orientation
    if orientation == 'coronal':
        n_slices = atlas.reference.shape[0]
    elif orientation == 'sagittal':
        n_slices = atlas.reference.shape[2]
    else:  # horizontal
        n_slices = atlas.reference.shape[1]

    # Normalize and preprocess input image
    image_float = image.astype(np.float32)
    img_min, img_max = image_float.min(), image_float.max()
    if img_max > img_min:
        img_norm = (image_float - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(image_float)

    # Optional: Apply slight blur to reduce noise sensitivity
    try:
        from scipy.ndimage import gaussian_filter
        img_norm = gaussian_filter(img_norm, sigma=2)
    except ImportError:
        pass

    best_score = -np.inf
    best_idx = 0
    scores = []

    # Scan through atlas slices
    indices_to_check = range(0, n_slices, step)
    total_checks = len(list(indices_to_check))

    for i, idx in enumerate(range(0, n_slices, step)):
        if progress_callback:
            progress_callback(i + 1, total_checks)

        # Get atlas slice at this position
        atlas_slice = atlas_manager.get_reference_slice(
            atlas_type=atlas_type,
            position_idx=idx,
            orientation=orientation,
        )

        # Resize atlas to match image dimensions
        atlas_resized = resize(
            atlas_slice.astype(np.float32),
            image.shape,
            preserve_range=True,
            anti_aliasing=True,
        )

        # Normalize atlas slice
        atlas_min, atlas_max = atlas_resized.min(), atlas_resized.max()
        if atlas_max > atlas_min:
            atlas_norm = (atlas_resized - atlas_min) / (atlas_max - atlas_min)
        else:
            atlas_norm = np.zeros_like(atlas_resized)

        # Compute normalized mutual information
        # Higher values indicate better match
        score = normalized_mutual_information(img_norm, atlas_norm)
        scores.append((idx, score))

        if score > best_score:
            best_score = score
            best_idx = idx

    # Refine: do a finer search around the best position
    refined_idx = best_idx
    if step > 1:
        fine_start = max(0, best_idx - step)
        fine_end = min(n_slices, best_idx + step + 1)

        for idx in range(fine_start, fine_end):
            atlas_slice = atlas_manager.get_reference_slice(
                atlas_type=atlas_type,
                position_idx=idx,
                orientation=orientation,
            )
            atlas_resized = resize(
                atlas_slice.astype(np.float32),
                image.shape,
                preserve_range=True,
                anti_aliasing=True,
            )
            atlas_min, atlas_max = atlas_resized.min(), atlas_resized.max()
            if atlas_max > atlas_min:
                atlas_norm = (atlas_resized - atlas_min) / (atlas_max - atlas_min)
            else:
                atlas_norm = np.zeros_like(atlas_resized)

            score = normalized_mutual_information(img_norm, atlas_norm)
            if score > best_score:
                best_score = score
                refined_idx = idx

        best_idx = refined_idx

    # Get final slices at best position
    reference_slice = atlas_manager.get_reference_slice(
        atlas_type=atlas_type,
        position_idx=best_idx,
        orientation=orientation,
    )
    annotation_slice = atlas_manager.get_annotation_slice(
        atlas_type=atlas_type,
        position_idx=best_idx,
        orientation=orientation,
    )

    return {
        'best_position_idx': best_idx,
        'best_score': float(best_score),
        'reference_slice': reference_slice,
        'annotation_slice': annotation_slice,
        'n_slices': n_slices,
        'orientation': orientation,
    }


def register_to_atlas(
    image: np.ndarray,
    atlas_slice: np.ndarray,
    method: str = 'similarity',
) -> Dict[str, Any]:
    """
    Compute 2D transform to align image to atlas slice.

    Uses phase correlation for translation estimation, then optionally
    feature matching for rotation and scale.

    Args:
        image: 2D grayscale image (Y, X)
        atlas_slice: 2D atlas reference slice (Y, X)
        method: 'translation' (shift only), 'similarity' (shift + rotation + uniform scale),
                or 'affine' (full affine transform)

    Returns:
        Dict with:
            'transform': SimilarityTransform or AffineTransform object
            'translation': (dy, dx) shift in pixels
            'rotation': Rotation angle in degrees
            'scale': Scale factor
            'success': Whether registration succeeded
            'method_used': Actual method used (may fall back)
    """
    # Ensure same size
    if atlas_slice.shape != image.shape:
        atlas_slice = resize(
            atlas_slice.astype(np.float32),
            image.shape,
            preserve_range=True,
            anti_aliasing=True,
        )

    # Normalize both images
    def normalize(img):
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return np.zeros_like(img)

    img_norm = normalize(image)
    atlas_norm = normalize(atlas_slice)

    # Step 1: Phase correlation for translation
    try:
        shift, error, diffphase = phase_cross_correlation(
            atlas_norm, img_norm, upsample_factor=10
        )
        translation = (float(shift[0]), float(shift[1]))
    except Exception:
        translation = (0.0, 0.0)

    # Initialize result
    result = {
        'translation': translation,
        'rotation': 0.0,
        'scale': 1.0,
        'success': True,
        'method_used': 'translation',
    }

    if method == 'translation':
        # Just translation
        result['transform'] = SimilarityTransform(translation=translation[::-1])
        return result

    # Step 2: Try feature-based matching for rotation/scale
    try:
        from skimage.feature import ORB, match_descriptors, BRIEF
        from skimage.color import rgb2gray

        # Ensure 8-bit for feature detection
        img_uint8 = (img_norm * 255).astype(np.uint8)
        atlas_uint8 = (atlas_norm * 255).astype(np.uint8)

        # Use ORB detector
        detector = ORB(n_keypoints=500, fast_threshold=0.05)

        detector.detect_and_extract(img_uint8)
        kp_img = detector.keypoints
        desc_img = detector.descriptors

        detector.detect_and_extract(atlas_uint8)
        kp_atlas = detector.keypoints
        desc_atlas = detector.descriptors

        if desc_img is not None and desc_atlas is not None and len(desc_img) > 10 and len(desc_atlas) > 10:
            # Match descriptors
            matches = match_descriptors(desc_img, desc_atlas, cross_check=True)

            if len(matches) >= 4:
                # Get matched keypoints
                src = kp_img[matches[:, 0]]
                dst = kp_atlas[matches[:, 1]]

                # Estimate transform
                transform_type = 'similarity' if method == 'similarity' else 'affine'
                tform = estimate_transform(transform_type, src, dst)

                if tform is not None:
                    # Extract parameters
                    if hasattr(tform, 'rotation'):
                        result['rotation'] = float(np.degrees(tform.rotation))
                    if hasattr(tform, 'scale'):
                        if isinstance(tform.scale, (list, tuple, np.ndarray)):
                            result['scale'] = float(np.mean(tform.scale))
                        else:
                            result['scale'] = float(tform.scale)
                    if hasattr(tform, 'translation'):
                        result['translation'] = (float(tform.translation[1]), float(tform.translation[0]))

                    result['transform'] = tform
                    result['method_used'] = method
                    result['n_matches'] = len(matches)
                    return result

    except ImportError:
        pass
    except Exception as e:
        print(f"[registration] Feature matching failed: {e}")

    # Fall back to translation only
    result['transform'] = SimilarityTransform(translation=translation[::-1])
    result['method_used'] = 'translation'
    return result


def apply_registration(
    atlas_labels: np.ndarray,
    transform,
    output_shape: Tuple[int, int],
    inverse: bool = True,
) -> np.ndarray:
    """
    Warp atlas labels to image coordinates using a transform.

    Uses nearest-neighbor interpolation to preserve label IDs.

    Args:
        atlas_labels: 2D array of atlas region IDs
        transform: Transform object (SimilarityTransform, AffineTransform, etc.)
        output_shape: (height, width) of output image
        inverse: If True, apply inverse transform (atlas → image space)

    Returns:
        2D array of warped atlas labels at output_shape resolution
    """
    # Resize atlas labels to output shape first (nearest neighbor)
    if atlas_labels.shape != output_shape:
        atlas_resized = resize(
            atlas_labels,
            output_shape,
            order=0,  # Nearest neighbor for labels
            preserve_range=True,
            anti_aliasing=False,
        ).astype(atlas_labels.dtype)
    else:
        atlas_resized = atlas_labels

    # Apply transform
    if transform is None:
        return atlas_resized

    # Use inverse transform to go from output to input coordinates
    if inverse and hasattr(transform, 'inverse'):
        transform_to_use = transform.inverse
    else:
        transform_to_use = transform

    # Warp with nearest neighbor interpolation
    warped = warp(
        atlas_resized,
        transform_to_use,
        output_shape=output_shape,
        order=0,  # Nearest neighbor
        preserve_range=True,
        mode='constant',
        cval=0,
    )

    return warped.astype(atlas_labels.dtype)


def compute_registration_quality(
    image: np.ndarray,
    atlas_slice: np.ndarray,
    transform=None,
) -> Dict[str, float]:
    """
    Compute quality metrics for a registration.

    Args:
        image: Original image
        atlas_slice: Atlas reference slice
        transform: Optional transform to apply to atlas before comparison

    Returns:
        Dict with quality metrics:
            'nmi': Normalized mutual information
            'correlation': Pearson correlation coefficient
    """
    # Ensure same size
    if atlas_slice.shape != image.shape:
        atlas_slice = resize(
            atlas_slice.astype(np.float32),
            image.shape,
            preserve_range=True,
        )

    # Apply transform if provided
    if transform is not None:
        atlas_slice = warp(
            atlas_slice,
            transform.inverse if hasattr(transform, 'inverse') else transform,
            output_shape=image.shape,
            preserve_range=True,
        )

    # Normalize
    def normalize(img):
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return np.zeros_like(img)

    img_norm = normalize(image)
    atlas_norm = normalize(atlas_slice)

    # Compute metrics
    nmi = normalized_mutual_information(img_norm, atlas_norm)

    # Pearson correlation
    img_flat = img_norm.flatten()
    atlas_flat = atlas_norm.flatten()
    correlation = np.corrcoef(img_flat, atlas_flat)[0, 1]

    return {
        'nmi': float(nmi),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
    }
