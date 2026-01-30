"""
preprocessing.py - Image preprocessing for atlas registration

Provides functions for:
- Contrast enhancement (CLAHE)
- Edge detection (Sobel, Canny)
- Brain mask extraction
- Multi-scale preprocessing
"""

from typing import Tuple, Optional
import numpy as np

from skimage.filters import sobel, gaussian
from skimage.exposure import equalize_adapthist, rescale_intensity
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion


def preprocess_for_registration(
    image: np.ndarray,
    use_edges: bool = True,
    clahe_clip_limit: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for atlas registration.

    Applies contrast enhancement and edge detection to make
    structural features more prominent for registration.

    Args:
        image: 2D grayscale image (Y, X)
        use_edges: If True, return edge-detected image; else enhanced image
        clahe_clip_limit: CLAHE clip limit (higher = more contrast)

    Returns:
        Tuple of (processed_image, brain_mask)
    """
    # Ensure float and normalize to 0-1
    img_float = image.astype(np.float32)
    img_min, img_max = img_float.min(), img_float.max()
    if img_max > img_min:
        img_norm = (img_float - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_float)

    # CLAHE contrast enhancement
    img_clahe = equalize_adapthist(img_norm, clip_limit=clahe_clip_limit)

    # Edge detection (Sobel)
    if use_edges:
        # Slight blur to reduce noise before edge detection
        img_smooth = gaussian(img_clahe, sigma=1)
        processed = sobel(img_smooth)
        # Normalize edges to 0-1
        e_min, e_max = processed.min(), processed.max()
        if e_max > e_min:
            processed = (processed - e_min) / (e_max - e_min)
    else:
        processed = img_clahe

    # Brain mask extraction
    mask = extract_brain_mask(img_norm)

    return processed, mask


def preprocess_atlas_slice(
    atlas_slice: np.ndarray,
    use_edges: bool = True,
    clahe_clip_limit: float = 0.03,
) -> np.ndarray:
    """
    Preprocess atlas reference slice for registration.

    Same preprocessing as user images to ensure comparable features.

    Args:
        atlas_slice: 2D atlas reference image
        use_edges: If True, return edge-detected image
        clahe_clip_limit: CLAHE clip limit

    Returns:
        Preprocessed atlas slice
    """
    # Normalize
    img_float = atlas_slice.astype(np.float32)
    img_min, img_max = img_float.min(), img_float.max()
    if img_max > img_min:
        img_norm = (img_float - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_float)

    # CLAHE
    img_clahe = equalize_adapthist(img_norm, clip_limit=clahe_clip_limit)

    if use_edges:
        img_smooth = gaussian(img_clahe, sigma=1)
        processed = sobel(img_smooth)
        # Normalize
        e_min, e_max = processed.min(), processed.max()
        if e_max > e_min:
            processed = (processed - e_min) / (e_max - e_min)
        return processed
    else:
        return img_clahe


def extract_brain_mask(
    image: np.ndarray,
    threshold_percentile: float = 10,
    dilation_iterations: int = 5,
    erosion_iterations: int = 2,
) -> np.ndarray:
    """
    Extract brain tissue mask from image.

    Uses thresholding and morphological operations to create
    a mask of the brain region.

    Args:
        image: 2D normalized image (0-1 range)
        threshold_percentile: Percentile for initial threshold
        dilation_iterations: Number of dilation iterations
        erosion_iterations: Number of erosion iterations (cleanup)

    Returns:
        Binary mask (True = brain tissue)
    """
    # Threshold
    threshold = np.percentile(image, threshold_percentile)
    mask = image > threshold

    # Fill holes
    mask = binary_fill_holes(mask)

    # Dilate to close gaps
    mask = binary_dilation(mask, iterations=dilation_iterations)

    # Erode to clean up edges
    mask = binary_erosion(mask, iterations=erosion_iterations)

    # Fill holes again
    mask = binary_fill_holes(mask)

    return mask.astype(bool)


def downsample_for_registration(
    image: np.ndarray,
    target_size: int = 512,
) -> Tuple[np.ndarray, float]:
    """
    Downsample image for faster coarse registration.

    Args:
        image: 2D image to downsample
        target_size: Target size for largest dimension

    Returns:
        Tuple of (downsampled_image, scale_factor)
    """
    from skimage.transform import resize

    h, w = image.shape
    max_dim = max(h, w)

    if max_dim <= target_size:
        return image, 1.0

    scale = target_size / max_dim
    new_h = int(h * scale)
    new_w = int(w * scale)

    downsampled = resize(
        image,
        (new_h, new_w),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(image.dtype)

    return downsampled, scale


def normalize_intensity(
    image: np.ndarray,
    percentile_low: float = 1,
    percentile_high: float = 99,
) -> np.ndarray:
    """
    Normalize image intensity using percentile-based rescaling.

    More robust than min-max normalization for images with outliers.

    Args:
        image: Input image
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        Normalized image in 0-1 range
    """
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)

    if p_high > p_low:
        normalized = np.clip(image, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)

    return normalized.astype(np.float32)
