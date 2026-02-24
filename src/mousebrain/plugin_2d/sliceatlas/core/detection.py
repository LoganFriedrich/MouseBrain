"""
detection.py - Nuclei detection for 2D confocal slice images

Provides multiple detection backends:
- Threshold: Simple intensity thresholding for sparse, bright nuclei (DEFAULT)
- StarDist: Deep learning segmentation for dense, touching nuclei
- Cellpose: Alternative deep learning segmentation

For sparse retrograde-labeled neurons (e.g., DCN projecting neurons),
threshold detection is preferred — it directly finds bright objects in the
nuclear channel without the overhead of a segmentation model.

Usage:
    from brainslice.core.detection import detect_by_threshold, NucleiDetector

    # Threshold detection (recommended for sparse fluorescent nuclei)
    labels, details = detect_by_threshold(image, min_area=10, max_area=500)

    # StarDist detection (for dense DAPI-stained fields)
    detector = NucleiDetector()
    labels, details = detector.detect(image)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np

# Lazy imports
_stardist = None
_csbdeep = None


def _get_stardist():
    """Lazy import StarDist."""
    global _stardist
    if _stardist is None:
        try:
            from stardist.models import StarDist2D
            _stardist = StarDist2D
        except ImportError:
            raise ImportError(
                "StarDist is required for nuclei detection. "
                "Install with: pip install stardist tensorflow"
            )
    return _stardist


def _get_csbdeep():
    """Lazy import csbdeep normalization."""
    global _csbdeep
    if _csbdeep is None:
        try:
            from csbdeep.utils import normalize
            _csbdeep = normalize
        except ImportError:
            # Fallback to simple normalization
            def normalize(x, pmin, pmax):
                lo = np.percentile(x, pmin)
                hi = np.percentile(x, pmax)
                return (x - lo) / (hi - lo + 1e-8)
            _csbdeep = normalize
    return _csbdeep


# ══════════════════════════════════════════════════════════════════════════
# PRE-DETECTION IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def preprocess_for_detection(
    image: np.ndarray,
    background_subtraction: bool = False,
    bg_sigma: float = 50.0,
    clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    clahe_kernel_size: Optional[int] = 128,
    gaussian_sigma: float = 0.0,
) -> np.ndarray:
    """
    Preprocess nuclear channel image before detection.

    Addresses common detection problems:
    - Background subtraction flattens uneven illumination so dim nuclei
      in dark regions become visible (fixes "missing dim nuclei")
    - CLAHE boosts local contrast so dim nuclei near bright ones stand out
    - Gaussian blur smooths speckle noise and small debris

    Order: bg_sub → CLAHE → blur (each step builds on the previous).

    Args:
        image: 2D grayscale nuclear channel image
        background_subtraction: Subtract estimated background illumination.
            Uses large-sigma Gaussian to estimate slowly-varying background.
        bg_sigma: Sigma for background estimation (larger = smoother).
            50.0 works well for most microscopy images.
        clahe: Apply Contrast Limited Adaptive Histogram Equalization.
        clahe_clip_limit: CLAHE clip limit (0.01-0.05 typical).
            Higher = more contrast enhancement.
        clahe_kernel_size: CLAHE tile/kernel size in pixels.
            None = auto (1/8 of image size).
        gaussian_sigma: Gaussian blur sigma. 0 = no blur.
            1.0 is a light denoise, 2.0 is moderate smoothing.

    Returns:
        Preprocessed image as float32, 0-1 range.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.exposure import equalize_adapthist

    # Convert to float32 and normalize to 0-1
    img = image.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)

    # Step 1: Background subtraction
    # Estimates slowly-varying illumination with a large Gaussian, then subtracts.
    # This is equivalent to a top-hat filter but faster.
    if background_subtraction:
        bg = gaussian_filter(img, sigma=bg_sigma)
        img = np.clip(img - bg, 0, None)
        # Re-normalize after subtraction
        s_max = img.max()
        if s_max > 0:
            img = img / s_max

    # Step 2: CLAHE (local contrast enhancement)
    if clahe:
        if clahe_kernel_size is not None:
            # equalize_adapthist expects kernel_size as int or iterable
            kernel = min(clahe_kernel_size, min(img.shape) // 2)
            kernel = max(kernel, 8)  # minimum sensible kernel
        else:
            kernel = None  # auto
        img = equalize_adapthist(img, clip_limit=clahe_clip_limit,
                                 kernel_size=kernel).astype(np.float32)

    # Step 3: Light Gaussian blur (denoise)
    if gaussian_sigma > 0:
        from skimage.filters import gaussian
        img = gaussian(img, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)

    return img


# ══════════════════════════════════════════════════════════════════════════
# THRESHOLD-BASED DETECTION (for sparse fluorescent nuclei)
# ══════════════════════════════════════════════════════════════════════════

def detect_by_threshold(
    image: np.ndarray,
    method: str = 'otsu',
    percentile: float = 99.0,
    manual_threshold: Optional[float] = None,
    min_area: int = 10,
    max_area: int = 5000,
    opening_radius: int = 0,
    closing_radius: int = 0,
    fill_holes: bool = True,
    split_touching: bool = False,
    split_footprint_size: int = 10,
    gaussian_sigma: float = 1.0,
    use_hysteresis: bool = True,
    hysteresis_low_fraction: float = 0.5,
    min_solidity: float = 0.0,
    min_circularity: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect nuclei by intensity thresholding on the nuclear channel.

    This is the correct approach for sparse, bright nuclei (e.g., retrograde-
    labeled neurons). The nuclei are clearly brighter than background, so a
    simple threshold finds them more reliably than a segmentation model trained
    on dense, touching nuclei.

    Steps:
        1. Optional Gaussian blur to denoise
        2. Threshold (Otsu, percentile, or manual)
        3. Hysteresis expansion: use a lower threshold to capture the full
           extent of each nucleus (not just the bright core)
        4. Optional morphological opening to remove small noise
        5. Optional morphological closing to bridge small gaps
        6. Fill holes in binary mask (enabled by default)
        7. Connected component labeling
        8. Optional watershed splitting of merged nuclei
        9. Size filter
       10. Optional solidity/circularity filter

    Hysteresis thresholding (enabled by default) solves two problems:
    - **Undersized detections**: A single threshold captures only the bright
      core of each nucleus. The low threshold expands to the full boundary.
    - **Missing dim nuclei**: Dimmer nuclei whose peak intensity barely
      exceeds the high threshold get properly captured once their connected
      region above the low threshold is included.

    Args:
        image: 2D nuclear channel image (any dtype)
        method: Threshold method
            - 'otsu': Otsu's automatic threshold (good default)
            - 'percentile': Top N% brightest pixels
            - 'manual': Use manual_threshold value directly
        percentile: For percentile method, keep pixels above this percentile
        manual_threshold: For manual method, intensity cutoff (in image units)
        min_area: Minimum nucleus area in pixels
        max_area: Maximum nucleus area in pixels
        opening_radius: Radius for morphological opening (noise removal).
            0 = no opening (default). 1-2 = light cleanup.
        closing_radius: Radius for morphological closing (bridge small gaps
            within nucleus boundaries). 0 = no closing (default). 1-3 typical.
        fill_holes: Fill holes in the binary mask before labeling. True by
            default — saturated H2B nuclei can have internal voids from noise.
        split_touching: Use distance-transform + watershed to split merged
            nuclei. Useful when hysteresis merges two adjacent bright nuclei
            into one connected component. Default False.
        split_footprint_size: Size of local-maximum footprint for watershed
            seed detection. Larger values require peaks to be farther apart
            to be considered separate nuclei. Default 10.
        gaussian_sigma: Gaussian blur sigma before thresholding. 0 = no blur.
        use_hysteresis: If True (default), use hysteresis thresholding to
            capture the full extent of each nucleus. The computed threshold
            becomes the "high" threshold; a "low" threshold at
            high * hysteresis_low_fraction captures nucleus edges.
        hysteresis_low_fraction: Low threshold as fraction of high threshold.
            0.5 = low threshold is half the high threshold. Lower values
            capture more of each nucleus boundary but may merge nearby objects.
        min_solidity: Minimum solidity (area / convex_hull_area) to keep.
            0 = no filtering (default). Nuclei are typically > 0.8.
            Debris and artifacts are often < 0.7.
        min_circularity: Minimum circularity (4*pi*area/perimeter^2).
            0 = no filtering (default). Perfect circle = 1.0.

    Returns:
        Tuple of (labels, details)
        - labels: 2D label image where each nucleus has unique ID (0 = bg)
        - details: dict with 'threshold', 'method', 'raw_count', etc.
    """
    from skimage.filters import threshold_otsu, apply_hysteresis_threshold
    from skimage.morphology import disk, binary_opening, binary_closing, label
    from skimage.measure import regionprops
    from scipy.ndimage import binary_fill_holes

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D")

    # Work on float copy
    img = image.astype(np.float32)

    # Step 1: Optional Gaussian blur
    if gaussian_sigma > 0:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=gaussian_sigma)

    # Step 2: Compute threshold
    if method == 'otsu':
        thresh_val = threshold_otsu(img)
    elif method == 'percentile':
        thresh_val = np.percentile(img, percentile)
    elif method == 'manual':
        if manual_threshold is None:
            raise ValueError("manual_threshold required when method='manual'")
        thresh_val = float(manual_threshold)
    elif method == 'zscore':
        # Z-score detection: "is this pixel significantly brighter than background?"
        # Use robust statistics (median + MAD) to estimate background, since
        # nuclei are sparse outliers that would bias mean/std upward.
        bg_median = float(np.median(img))
        mad = float(np.median(np.abs(img - bg_median)))
        # MAD-to-std conversion factor for Gaussian: std ≈ 1.4826 * MAD
        bg_std = mad * 1.4826
        if bg_std < 1e-6:
            bg_std = float(np.std(img))  # fallback if image is near-constant
        # threshold = k standard deviations above background
        # 'percentile' parameter is repurposed as the z-score cutoff (default 5.0)
        # At typical magnifications (2-3 um/px), real fluorescent nuclei are 5-700+
        # sigma above background. Confocal with few z-stacks means some H2B-mCherry
        # nuclei are partially out of plane and appear dim — z=5 catches these
        # while still rejecting autofluorescence noise (z=3 would be too loose).
        z_cutoff = percentile if percentile != 99.0 else 5.0
        thresh_val = bg_median + z_cutoff * bg_std
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # ── Peak-based detection (zscore method) ──
    # For sparse fluorescent nuclei, find bright spots directly via local
    # maxima then stamp a fixed-radius circular label around each peak.
    # This is the standard "spot detection" approach used by QuPath, CellProfiler,
    # etc. At low magnification (2-8 px per nucleus), circles are accurate enough
    # and we avoid all watershed/merging artifacts.
    if method == 'zscore':
        from skimage.feature import peak_local_max

        # Find intensity peaks above threshold
        # min_distance prevents detecting two peaks within one nucleus
        min_dist = max(2, int(np.sqrt(min_area)))
        coords = peak_local_max(
            img,
            min_distance=min_dist,
            threshold_abs=thresh_val,
        )
        raw_count = len(coords)

        # Stamp circular labels around each peak
        # Radius: geometric mean of min/max expected nucleus radius
        avg_area = (min_area + max_area) / 2
        radius_px = max(1, int(np.sqrt(avg_area / np.pi)))

        labels = np.zeros(img.shape, dtype=np.int32)
        h_img, w_img = img.shape
        yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        circle_template = (yy ** 2 + xx ** 2) <= radius_px ** 2

        for i, (pr, pc) in enumerate(coords, 1):
            # Clip circle to image bounds
            y0 = max(0, pr - radius_px)
            y1 = min(h_img, pr + radius_px + 1)
            x0 = max(0, pc - radius_px)
            x1 = min(w_img, pc + radius_px + 1)
            cy0 = y0 - (pr - radius_px)
            cy1 = circle_template.shape[0] - ((pr + radius_px + 1) - y1)
            cx0 = x0 - (pc - radius_px)
            cx1 = circle_template.shape[1] - ((pc + radius_px + 1) - x1)
            mask = circle_template[cy0:cy1, cx0:cx1]
            region = labels[y0:y1, x0:x1]
            # Don't overwrite existing labels (first-come = brighter peak)
            region[(mask) & (region == 0)] = i

        n_split = 0

    else:
        # ── Classic threshold path (otsu, percentile, manual) ──
        # Step 3: Apply threshold (with optional hysteresis)
        if use_hysteresis:
            low_thresh = thresh_val * hysteresis_low_fraction
            binary = apply_hysteresis_threshold(img, low_thresh, thresh_val)
        else:
            binary = img > thresh_val

        # Step 4: Optional morphological opening (remove small noise specks)
        if opening_radius > 0:
            selem = disk(opening_radius)
            binary = binary_opening(binary, selem)

        # Step 5: Optional morphological closing (bridge small gaps)
        if closing_radius > 0:
            selem = disk(closing_radius)
            binary = binary_closing(binary, selem)

        # Step 6: Fill holes in binary mask
        if fill_holes:
            binary = binary_fill_holes(binary)

        # Step 7: Connected component labeling
        labels = label(binary)
        raw_count = labels.max()

        # Step 8: Optional watershed splitting of merged nuclei
        n_split = 0
        if split_touching and raw_count > 0:
            labels, n_split = _watershed_split(labels, split_footprint_size)

    count_after_split = labels.max()

    # Step 9: Size filter
    removed_by_size = 0
    if count_after_split > 0:
        props = regionprops(labels)
        valid = [p for p in props if min_area <= p.area <= max_area]
        removed_by_size = len(props) - len(valid)

        filtered = np.zeros_like(labels)
        for new_id, prop in enumerate(valid, 1):
            filtered[labels == prop.label] = new_id
        labels = filtered

    # Step 10: Optional solidity/circularity filter
    removed_by_morphology = 0
    count_before_morph = labels.max()
    if count_before_morph > 0 and (min_solidity > 0 or min_circularity > 0):
        props = regionprops(labels)
        valid = []
        for p in props:
            if min_solidity > 0 and p.solidity < min_solidity:
                continue
            if min_circularity > 0 and p.perimeter > 0:
                circ = 4 * np.pi * p.area / (p.perimeter ** 2)
                if circ < min_circularity:
                    continue
            valid.append(p)
        removed_by_morphology = len(props) - len(valid)

        filtered = np.zeros_like(labels)
        for new_id, prop in enumerate(valid, 1):
            filtered[labels == prop.label] = new_id
        labels = filtered

    final_count = labels.max()

    # Build details dict (compatible with existing pipeline interface)
    details = {
        'threshold': float(thresh_val),
        'threshold_low': float(thresh_val * hysteresis_low_fraction) if use_hysteresis else float(thresh_val),
        'use_hysteresis': use_hysteresis,
        'hysteresis_low_fraction': hysteresis_low_fraction,
        'method': method,
        'raw_count': int(raw_count),
        'filtered_count': int(final_count),
        'removed_by_size': int(removed_by_size),
        'removed_by_morphology': int(removed_by_morphology),
        'n_watershed_splits': int(n_split),
        'fill_holes': fill_holes,
        'closing_radius': closing_radius,
        'split_touching': split_touching,
    }

    # Add z-score diagnostics if applicable
    if method == 'zscore':
        details['bg_median'] = float(bg_median)
        details['bg_std'] = float(bg_std)
        details['z_cutoff'] = float(z_cutoff)

    return labels, details


def _watershed_split(
    labels: np.ndarray,
    footprint_size: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Split merged nuclei using distance-transform watershed.

    For each connected component, computes the distance transform and finds
    local maxima. If a component has multiple peaks, watershed splits it into
    separate objects.

    Args:
        labels: Label image from connected component labeling.
        footprint_size: Size of the footprint for local maximum detection.
            Larger values require peaks to be farther apart to be considered
            separate nuclei. 10 is a good default for typical microscopy.

    Returns:
        Tuple of (new_labels, n_splits) where n_splits is the number of
        objects that were split (i.e., extra objects created).
    """
    from scipy.ndimage import distance_transform_edt, label as ndi_label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    binary = labels > 0
    distance = distance_transform_edt(binary)

    # Find local maxima (seeds for watershed)
    coords = peak_local_max(
        distance,
        footprint=np.ones((footprint_size, footprint_size)),
        labels=binary.astype(int),
    )

    # Create marker image from peaks
    markers = np.zeros_like(labels, dtype=int)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    # Expand markers to connected regions
    markers_labeled, _ = ndi_label(markers)

    # Run watershed on negated distance (valleys = boundaries)
    ws_labels = watershed(-distance, markers_labeled, mask=binary)

    n_before = labels.max()
    n_after = ws_labels.max()
    n_splits = max(0, n_after - n_before)

    return ws_labels, n_splits


# ══════════════════════════════════════════════════════════════════════════
# LOG-AUGMENTED DETECTION (threshold + LoG union)
# ══════════════════════════════════════════════════════════════════════════

def detect_with_log_augmentation(
    image: np.ndarray,
    pixel_um: float,
    min_diameter_um: float = 10.0,
    max_diameter_um: float = 25.0,
    threshold_fraction: float = 0.20,
    log_threshold: float = 0.005,
    overlap_radius_factor: float = 1.5,
    gaussian_sigma: float = 1.0,
    use_hysteresis: bool = True,
    hysteresis_low_fraction: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect nuclei using threshold detection augmented with LoG blob detection.

    Uses a decision tree to determine detection strategy based on image quality:

    1. GOOD SIGNAL (Otsu finds >= 5 nuclei):
       Run full LoG augmentation. Otsu detections provide a reference intensity
       floor -- LoG blobs must have peak intensity >= 50% of the dimmest Otsu
       nucleus to be accepted. This prevents noise blobs while catching dim
       real nuclei.

    2. MARGINAL SIGNAL (Otsu finds 1-4 nuclei, but image has dynamic range):
       Run LoG with stricter filtering: blobs must have peak intensity >= the
       median intensity of Otsu detections. Conservative -- only adds blobs
       that look like confirmed detections.

    3. NO SIGNAL (Otsu finds 0 nuclei, or image is essentially flat):
       Skip LoG entirely. Return Otsu-only results (0 or few nuclei).
       Flat = Otsu threshold < 3x image median (no bimodal distribution).

    Args:
        image: 2D nuclear channel image (any dtype).
        pixel_um: Pixel size in micrometers per pixel.
        min_diameter_um: Minimum nucleus diameter in micrometers.
        max_diameter_um: Maximum nucleus diameter in micrometers.
        threshold_fraction: Fraction of Otsu threshold to use as the primary
            detection threshold. 0.20 = 20% of Otsu.
        log_threshold: LoG blob detection threshold on the normalized [0,1]
            image. Lower = more sensitive. 0.005 catches faint circular blobs.
        overlap_radius_factor: LoG blobs within this factor of their radius
            from a threshold centroid are considered overlapping and discarded.
        gaussian_sigma: Gaussian blur sigma for threshold detection.
        use_hysteresis: Whether to use hysteresis thresholding.
        hysteresis_low_fraction: Low threshold fraction for hysteresis.

    Returns:
        Tuple of (labels, details) where:
        - labels: 2D label image (0 = background, 1..N = nuclei). Threshold
          detections get labels 1..M, LoG-only detections get labels M+1..N.
        - details: dict with detection diagnostics including counts from
          each method and combined total.
    """
    from skimage.feature import blob_log
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops
    from scipy.ndimage import gaussian_filter
    import math

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D")

    # Convert size filter from physical to pixel units (diameter-based)
    min_area = int(math.pi * (min_diameter_um / 2 / pixel_um) ** 2)
    max_area = int(math.pi * (max_diameter_um / 2 / pixel_um) ** 2)

    # --- Step 1: Image quality assessment ---
    img_f = gaussian_filter(image.astype(np.float32), sigma=gaussian_sigma)
    otsu_val = float(threshold_otsu(img_f))
    img_median = float(np.median(img_f))
    img_mean = float(np.mean(img_f))

    # Dynamic range: how far above background does signal extend?
    # Flat images have otsu close to median (no bimodal separation)
    otsu_to_median = otsu_val / max(img_median, 1.0)
    is_flat = otsu_to_median < 3.0

    # --- Step 2: Threshold detection (always runs) ---
    manual_thresh = otsu_val * threshold_fraction

    labels_thresh, det_thresh = detect_by_threshold(
        image,
        method='manual',
        manual_threshold=manual_thresh,
        min_area=min_area,
        max_area=max_area,
        gaussian_sigma=gaussian_sigma,
        use_hysteresis=use_hysteresis,
        hysteresis_low_fraction=hysteresis_low_fraction,
    )
    n_threshold = det_thresh['filtered_count']

    # Collect threshold detection properties for intensity reference
    thresh_centroids = []
    thresh_peak_intensities = []
    if n_threshold > 0:
        for prop in regionprops(labels_thresh, intensity_image=image):
            thresh_centroids.append(np.array(prop.centroid))
            thresh_peak_intensities.append(float(prop.max_intensity))

    # --- Step 3: Decision tree ---
    GOOD_SIGNAL_MIN = 5
    MARGINAL_SIGNAL_MIN = 1

    if n_threshold >= GOOD_SIGNAL_MIN:
        decision = 'good_signal'
        # LoG blobs must have peak intensity >= 50% of dimmest Otsu detection
        intensity_floor = min(thresh_peak_intensities) * 0.5
    elif n_threshold >= MARGINAL_SIGNAL_MIN and not is_flat:
        decision = 'marginal_signal'
        # Stricter: blobs must be >= median of Otsu detection intensities
        intensity_floor = float(np.median(thresh_peak_intensities))
    else:
        decision = 'no_signal'
        intensity_floor = float('inf')  # blocks all LoG blobs

    # --- Step 4: LoG blob detection (skipped for no_signal) ---
    n_log_raw = 0
    n_log_rejected_overlap = 0
    n_log_rejected_intensity = 0
    new_blobs = []

    if decision != 'no_signal':
        # Normalize for LoG (standard [0,1] range)
        img_norm = image.astype(np.float64)
        img_range = img_norm.max() - img_norm.min()
        if img_range > 0:
            img_norm = (img_norm - img_norm.min()) / img_range
        else:
            img_norm = np.zeros_like(img_norm)

        min_sigma = (min_diameter_um / 2 / pixel_um) / np.sqrt(2)
        max_sigma = (max_diameter_um / 2 / pixel_um) / np.sqrt(2)

        blobs = blob_log(
            img_norm,
            min_sigma=max(0.5, min_sigma),
            max_sigma=max_sigma,
            num_sigma=10,
            threshold=log_threshold,
            overlap=0.5,
        )
        n_log_raw = len(blobs)

        # --- Step 5: Filter LoG blobs ---
        for blob in blobs:
            y, x, sigma = blob
            yi, xi = int(round(y)), int(round(x))

            # Reject if blob center lands on existing threshold detection
            if 0 <= yi < labels_thresh.shape[0] and 0 <= xi < labels_thresh.shape[1]:
                if labels_thresh[yi, xi] > 0:
                    n_log_rejected_overlap += 1
                    continue

            # Reject if too close to a threshold centroid
            if thresh_centroids:
                blob_radius = sigma * np.sqrt(2)
                too_close = False
                for c in thresh_centroids:
                    dist = np.sqrt((y - c[0]) ** 2 + (x - c[1]) ** 2)
                    if dist < blob_radius * overlap_radius_factor:
                        too_close = True
                        break
                if too_close:
                    n_log_rejected_overlap += 1
                    continue

            # INTENSITY GATE: reject blobs whose peak pixel is below floor
            # This prevents noise blobs on high-background images
            radius_px = max(1, int(round(sigma * np.sqrt(2))))
            y0g = max(0, yi - radius_px)
            y1g = min(image.shape[0], yi + radius_px + 1)
            x0g = max(0, xi - radius_px)
            x1g = min(image.shape[1], xi + radius_px + 1)
            blob_peak = float(image[y0g:y1g, x0g:x1g].max())

            if blob_peak < intensity_floor:
                n_log_rejected_intensity += 1
                continue

            new_blobs.append(blob)

    # --- Step 6: Stamp LoG blobs into the label image ---
    # Use at least the minimum physical nucleus radius for stamps so that
    # LoG detections respect the same size constraint as threshold detections.
    min_radius_px = max(1, int(round(min_diameter_um / 2 / pixel_um)))
    next_label = n_threshold + 1
    h_img, w_img = labels_thresh.shape
    labels_combined = labels_thresh.copy()
    n_log_rejected_size = 0

    for blob in new_blobs:
        y, x, sigma = blob
        blob_radius = max(1, int(round(sigma * np.sqrt(2))))
        # Use the larger of the detected blob radius and the physical
        # minimum — ensures stamps are never below the size filter.
        radius_px = max(blob_radius, min_radius_px)

        y0 = max(0, int(round(y)) - radius_px)
        y1 = min(h_img, int(round(y)) + radius_px + 1)
        x0 = max(0, int(round(x)) - radius_px)
        x1 = min(w_img, int(round(x)) + radius_px + 1)

        yy, xx = np.ogrid[y0 - int(round(y)):y1 - int(round(y)),
                           x0 - int(round(x)):x1 - int(round(x))]
        circle = (yy ** 2 + xx ** 2) <= radius_px ** 2
        region = labels_combined[y0:y1, x0:x1]
        region[(circle) & (region == 0)] = next_label
        next_label += 1

    n_log_new = next_label - n_threshold - 1
    n_combined = n_threshold + n_log_new

    # --- Step 7: Trim smallest 5% of detections ---
    # Removes spurious tiny detections that passed individual filters
    # but are clearly outliers in the size distribution.
    n_trimmed_small = 0
    if n_combined > 10:
        all_props = regionprops(labels_combined)
        all_areas = np.array([p.area for p in all_props])
        p5_area = float(np.percentile(all_areas, 5))
        if p5_area > 0:
            to_remove = [p.label for p in all_props if p.area < p5_area]
            for lbl in to_remove:
                labels_combined[labels_combined == lbl] = 0
                n_trimmed_small += 1
            # Relabel to keep IDs contiguous
            if n_trimmed_small > 0:
                unique_labels = np.unique(labels_combined)
                unique_labels = unique_labels[unique_labels > 0]
                relabeled = np.zeros_like(labels_combined)
                for new_id, old_id in enumerate(unique_labels, 1):
                    relabeled[labels_combined == old_id] = new_id
                labels_combined = relabeled
                n_combined -= n_trimmed_small

    # --- Step 8: Circularity / eccentricity filter ---
    # Remove objects that are clearly not round nuclei (artifacts, debris,
    # elongated smears). Uses eccentricity from the fitted ellipse (robust
    # even at low pixel counts) rather than perimeter-based circularity
    # which is unreliable for small regions with jaggy edges.
    # Eccentricity: 0 = perfect circle, 1 = line. Nuclei viewed from above
    # are roughly round, so eccentricity should be < ~0.85. We use a generous
    # threshold to avoid rejecting legitimate oblique or slightly elongated
    # nuclei.
    max_eccentricity = 0.88  # allows ~2:1 aspect ratio
    min_solidity_val = 0.5   # must be at least half-filled (no hollow/C-shapes)
    n_removed_morph = 0
    if n_combined > 0:
        morph_props = regionprops(labels_combined)
        keep_labels = []
        for p in morph_props:
            # Skip eccentricity check for very small objects (< 6px) where
            # the fitted ellipse is unreliable
            if p.area >= 6 and p.eccentricity > max_eccentricity:
                n_removed_morph += 1
                continue
            if p.solidity < min_solidity_val:
                n_removed_morph += 1
                continue
            keep_labels.append(p.label)

        if n_removed_morph > 0:
            filtered = np.zeros_like(labels_combined)
            for new_id, old_lbl in enumerate(keep_labels, 1):
                filtered[labels_combined == old_lbl] = new_id
            labels_combined = filtered
            n_combined -= n_removed_morph

    details = {
        'method': 'threshold+log',
        'threshold': manual_thresh,
        'otsu_threshold': otsu_val,
        'threshold_fraction': threshold_fraction,
        'log_threshold': log_threshold,
        'n_threshold': int(n_threshold),
        'n_log_raw': int(n_log_raw),
        'n_log_new': int(n_log_new),
        'n_log_rejected_overlap': int(n_log_rejected_overlap),
        'n_log_rejected_intensity': int(n_log_rejected_intensity),
        'n_log_rejected_size': int(n_log_rejected_size),
        'n_trimmed_small': int(n_trimmed_small),
        'filtered_count': int(n_combined),
        'raw_count': int(n_threshold + n_log_raw),
        'min_area': min_area,
        'max_area': max_area,
        'pixel_um': pixel_um,
        'min_diameter_um': min_diameter_um,
        'max_diameter_um': max_diameter_um,
        'decision': decision,
        'intensity_floor': float(intensity_floor) if intensity_floor != float('inf') else None,
        'otsu_to_median': otsu_to_median,
        'is_flat': is_flat,
        'removed_by_size': det_thresh.get('removed_by_size', 0),
        'removed_by_morphology': int(n_removed_morph),
        'n_watershed_splits': det_thresh.get('n_watershed_splits', 0),
        'use_hysteresis': use_hysteresis,
        'hysteresis_low_fraction': hysteresis_low_fraction,
        'fill_holes': det_thresh.get('fill_holes', True),
        'closing_radius': det_thresh.get('closing_radius', 0),
        'split_touching': det_thresh.get('split_touching', False),
    }

    return labels_combined, details


# Available pretrained models
PRETRAINED_MODELS = {
    '2D_versatile_fluo': 'Versatile fluorescence microscopy (recommended)',
    '2D_versatile_he': 'H&E stained histology images',
    '2D_paper_dsb2018': 'Data Science Bowl 2018 nuclei',
}


class NucleiDetector:
    """
    Detect and segment nuclei using StarDist.

    StarDist is well-suited for round/convex nuclei like H2B-labeled cells.
    """

    def __init__(
        self,
        model_name: str = '2D_versatile_fluo',
        custom_model_path: Optional[Path] = None,
    ):
        """
        Initialize detector with StarDist model.

        Args:
            model_name: Pretrained model name (default: 2D_versatile_fluo)
            custom_model_path: Path to custom trained model directory
        """
        StarDist2D = _get_stardist()

        self.model_name = model_name
        self.custom_model_path = custom_model_path

        if custom_model_path:
            # Load custom model
            model_dir = Path(custom_model_path)
            self.model = StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))
        else:
            # Load pretrained model with Windows symlink workaround
            self.model = self._load_pretrained_model(StarDist2D, model_name)

    def _load_pretrained_model(self, StarDist2D, model_name: str):
        """
        Load pretrained model with workaround for Windows symlink issues.
        """
        import os

        try:
            # First attempt: standard loading
            return StarDist2D.from_pretrained(model_name)
        except OSError as e:
            # Windows symlink error - try loading from extracted path directly
            if 'privilege' in str(e).lower() or '1314' in str(e):
                # Model was downloaded but symlink failed
                # Load directly from the extracted folder
                keras_dir = Path(os.path.expanduser('~')) / '.keras' / 'models' / 'StarDist2D'
                model_parent_dir = keras_dir / model_name
                extracted_name = f'{model_name}_extracted'
                extracted_dir = model_parent_dir / extracted_name

                if extracted_dir.exists() and (extracted_dir / 'config.json').exists():
                    # Load from the extracted directory
                    # StarDist expects: basedir/name/config.json
                    # So: basedir = model_parent_dir, name = extracted_name
                    return StarDist2D(None, name=extracted_name, basedir=str(model_parent_dir))
                else:
                    # Try the non-extracted path in case symlink exists
                    model_dir = model_parent_dir / model_name
                    if model_dir.exists() and (model_dir / 'config.json').exists():
                        return StarDist2D(None, name=model_name, basedir=str(model_parent_dir))

            # Re-raise if we couldn't work around it
            raise

    def detect(
        self,
        image: np.ndarray,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        scale: float = 1.0,
        normalize_input: bool = True,
        n_tiles: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect nuclei and return segmentation masks.

        Args:
            image: 2D grayscale image (nuclear channel, e.g., mScarlet)
            prob_thresh: Probability threshold for detection (0-1)
            nms_thresh: Non-maximum suppression threshold (0-1)
            scale: Image rescaling factor (< 1 for downsampling)
            normalize_input: Whether to normalize image before detection
            n_tiles: Tile processing for large images, e.g., (4, 4)

        Returns:
            Tuple of (labels, details)
            - labels: 2D label image where each nucleus has unique ID (0 = background)
            - details: dict with 'coord' (centroids), 'prob' (detection probabilities),
                       'points' (polygon vertices)
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")

        # Normalize if requested
        if normalize_input:
            normalize = _get_csbdeep()
            image = normalize(image, 1, 99.8)

        # Scale if needed
        original_shape = image.shape
        if scale != 1.0:
            from scipy.ndimage import zoom
            image = zoom(image, scale, order=1)

        # Run prediction
        if n_tiles:
            labels, details = self.model.predict_instances(
                image,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=n_tiles,
            )
        else:
            labels, details = self.model.predict_instances(
                image,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
            )

        # Scale labels back if we downsampled
        if scale != 1.0:
            from scipy.ndimage import zoom as zoom_nd
            # Use nearest neighbor for labels to preserve integer values
            labels = zoom_nd(labels, 1.0 / scale, order=0).astype(labels.dtype)

            # Adjust coordinates in details
            if 'coord' in details:
                details['coord'] = details['coord'] / scale

            # Ensure labels match original shape
            if labels.shape != original_shape:
                # Pad or crop to match
                result = np.zeros(original_shape, dtype=labels.dtype)
                h = min(labels.shape[0], original_shape[0])
                w = min(labels.shape[1], original_shape[1])
                result[:h, :w] = labels[:h, :w]
                labels = result

        return labels, details

    def filter_by_size(
        self,
        labels: np.ndarray,
        min_area: int = 50,
        max_area: int = 5000,
    ) -> np.ndarray:
        """
        Filter detected nuclei by size.

        Args:
            labels: Label image from detect()
            min_area: Minimum nucleus area in pixels
            max_area: Maximum nucleus area in pixels

        Returns:
            Filtered label image with consecutive IDs
        """
        from skimage.measure import regionprops

        props = regionprops(labels)
        valid_labels = []

        for prop in props:
            if min_area <= prop.area <= max_area:
                valid_labels.append(prop.label)

        # Create filtered label image with new consecutive IDs
        filtered = np.zeros_like(labels)
        for new_id, old_label in enumerate(valid_labels, 1):
            filtered[labels == old_label] = new_id

        return filtered

    def filter_by_circularity(
        self,
        labels: np.ndarray,
        min_circularity: float = 0.5,
    ) -> np.ndarray:
        """
        Filter detected nuclei by circularity (roundness).

        Circularity = 4 * pi * area / perimeter^2
        Perfect circle = 1.0

        Args:
            labels: Label image from detect()
            min_circularity: Minimum circularity (0-1)

        Returns:
            Filtered label image with consecutive IDs
        """
        from skimage.measure import regionprops

        props = regionprops(labels)
        valid_labels = []

        for prop in props:
            if prop.perimeter > 0:
                circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
                if circularity >= min_circularity:
                    valid_labels.append(prop.label)

        filtered = np.zeros_like(labels)
        for new_id, old_label in enumerate(valid_labels, 1):
            filtered[labels == old_label] = new_id

        return filtered

    def filter_by_confidence(
        self,
        labels: np.ndarray,
        details: Dict[str, Any],
        min_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, int]:
        """
        Filter detections by StarDist confidence/probability score.

        StarDist returns a probability score per detection in details['prob'].
        This allows removing marginal detections that passed the initial
        prob_thresh but have low confidence.

        Args:
            labels: Label image from detect()
            details: Details dict from detect() (must contain 'prob')
            min_confidence: Minimum confidence score to keep (0-1)

        Returns:
            Tuple of (filtered_labels, n_removed)
        """
        if 'prob' not in details or len(details['prob']) == 0:
            return labels, 0

        probs = details['prob']
        # StarDist labels are 1-indexed, probs are 0-indexed
        # prob[i] corresponds to label (i+1)
        n_labels = labels.max()
        if n_labels == 0:
            return labels, 0

        # Build set of labels to keep
        keep_labels = set()
        for i, prob in enumerate(probs):
            label_id = i + 1
            if prob >= min_confidence and label_id <= n_labels:
                keep_labels.add(label_id)

        n_removed = n_labels - len(keep_labels)

        # Create filtered label image with consecutive IDs
        filtered = np.zeros_like(labels)
        for new_id, old_label in enumerate(sorted(keep_labels), 1):
            filtered[labels == old_label] = new_id

        return filtered, n_removed

    def filter_border_touching(
        self,
        labels: np.ndarray,
        border_width: int = 1,
    ) -> Tuple[np.ndarray, int]:
        """
        Remove nuclei that touch image borders.

        Partial nuclei at borders have incorrect area, shape, and
        intensity measurements. Removing them improves downstream
        accuracy.

        Args:
            labels: Label image
            border_width: Width of border region to check (pixels)

        Returns:
            Tuple of (filtered_labels, n_removed)
        """
        from skimage.segmentation import clear_border

        n_before = len(np.unique(labels)) - 1  # exclude bg
        cleared = clear_border(labels, buffer_size=border_width)

        # Relabel consecutively
        unique_labels = np.unique(cleared)
        unique_labels = unique_labels[unique_labels > 0]
        n_after = len(unique_labels)

        relabeled = np.zeros_like(cleared)
        for new_id, old_label in enumerate(unique_labels, 1):
            relabeled[cleared == old_label] = new_id

        return relabeled, n_before - n_after

    def filter_by_morphology(
        self,
        labels: np.ndarray,
        intensity_image: Optional[np.ndarray] = None,
        min_solidity: float = 0.0,
        min_mean_intensity: float = 0.0,
        max_eccentricity: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Filter nuclei by morphological properties and intensity.

        Solidity = area / convex_hull_area. Debris and artifacts typically
        have solidity < 0.7. Real nuclei are typically > 0.8.

        Mean intensity filtering removes dark detections that are likely
        segmentation artifacts (StarDist finding shapes in noise).

        Args:
            labels: Label image
            intensity_image: Nuclear channel image for intensity filtering.
                If None, intensity filtering is skipped.
            min_solidity: Minimum solidity (0-1). 0 = no filtering.
            min_mean_intensity: Minimum mean intensity in nuclear channel.
                0 = no filtering.
            max_eccentricity: Maximum eccentricity (0=circle, 1=line).
                1.0 = no filtering.

        Returns:
            Tuple of (filtered_labels, n_removed)
        """
        from skimage.measure import regionprops

        n_before = len(np.unique(labels)) - 1
        if n_before == 0:
            return labels, 0

        props = regionprops(labels, intensity_image=intensity_image)
        valid_labels = []

        for prop in props:
            # Solidity check
            if min_solidity > 0 and prop.solidity < min_solidity:
                continue
            # Eccentricity check
            if max_eccentricity < 1.0 and prop.eccentricity > max_eccentricity:
                continue
            # Intensity check (only if image provided and threshold set)
            if (intensity_image is not None and min_mean_intensity > 0
                    and prop.intensity_mean < min_mean_intensity):
                continue
            valid_labels.append(prop.label)

        filtered = np.zeros_like(labels)
        for new_id, old_label in enumerate(valid_labels, 1):
            filtered[labels == old_label] = new_id

        return filtered, n_before - len(valid_labels)

    @staticmethod
    def compute_n_tiles(
        image_shape: Tuple[int, int],
        tile_size: int = 1024,
    ) -> Optional[Tuple[int, int]]:
        """
        Auto-calculate n_tiles for large images.

        If any image dimension exceeds tile_size, returns appropriate tiling
        to prevent GPU memory issues. Otherwise returns None (no tiling needed).

        Args:
            image_shape: (height, width) of the image
            tile_size: Maximum tile dimension in pixels

        Returns:
            Tuple of (ny, nx) tiles, or None if image is small enough
        """
        h, w = image_shape[:2]
        if h <= tile_size and w <= tile_size:
            return None
        ny = max(1, (h + tile_size - 1) // tile_size)
        nx = max(1, (w + tile_size - 1) // tile_size)
        return (ny, nx)

    def get_centroids(
        self,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Get centroid coordinates for all detected nuclei.

        Args:
            labels: Label image

        Returns:
            Array of shape (N, 2) with (y, x) coordinates
        """
        from skimage.measure import regionprops

        props = regionprops(labels)
        centroids = np.array([prop.centroid for prop in props])

        return centroids

    def get_properties(
        self,
        labels: np.ndarray,
        intensity_image: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get properties of all detected nuclei.

        Args:
            labels: Label image
            intensity_image: Optional intensity image for intensity measurements

        Returns:
            Dict with arrays for each property:
            - label: nucleus IDs
            - centroid_y, centroid_x: positions
            - area: area in pixels
            - If intensity_image provided:
              - mean_intensity, max_intensity
        """
        from skimage.measure import regionprops_table

        properties = ['label', 'centroid', 'area', 'perimeter', 'eccentricity']

        if intensity_image is not None:
            properties.extend(['intensity_mean', 'intensity_max'])
            props = regionprops_table(
                labels,
                intensity_image=intensity_image,
                properties=properties
            )
        else:
            props = regionprops_table(labels, properties=properties)

        # Rename columns for clarity
        result = {
            'label': props['label'],
            'centroid_y': props['centroid-0'],
            'centroid_x': props['centroid-1'],
            'area': props['area'],
            'perimeter': props['perimeter'],
            'eccentricity': props['eccentricity'],
        }

        if intensity_image is not None:
            result['mean_intensity'] = props['intensity_mean']
            result['max_intensity'] = props['intensity_max']

        return result


def detect_nuclei(
    image: np.ndarray,
    model_name: str = '2D_versatile_fluo',
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    min_area: int = 50,
    max_area: int = 5000,
    preprocess: bool = False,
    background_subtraction: bool = False,
    clahe: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Convenience function for quick nuclei detection.

    Args:
        image: 2D nuclear channel image
        model_name: StarDist model name
        prob_thresh: Probability threshold
        nms_thresh: NMS threshold
        min_area: Minimum nucleus area
        max_area: Maximum nucleus area
        preprocess: Apply default preprocessing (bg_sub + CLAHE)
        background_subtraction: Apply background subtraction only
        clahe: Apply CLAHE only

    Returns:
        Tuple of (labels, count)
        - labels: Filtered label image
        - count: Number of nuclei detected
    """
    # Apply preprocessing if requested
    if preprocess:
        image = preprocess_for_detection(image, background_subtraction=True, clahe=True)
    elif background_subtraction or clahe:
        image = preprocess_for_detection(
            image, background_subtraction=background_subtraction, clahe=clahe)

    detector = NucleiDetector(model_name=model_name)
    labels, _ = detector.detect(image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    labels = detector.filter_by_size(labels, min_area=min_area, max_area=max_area)
    count = len(np.unique(labels)) - 1  # Exclude background

    return labels, count


# Available Cellpose models (for reference when cellpose backend is used)
CELLPOSE_MODELS = {
    'nuclei': 'Nuclear segmentation (recommended for nuclear stains)',
    'cyto': 'Cytoplasm segmentation (original model)',
    'cyto2': 'Improved cytoplasm segmentation',
    'cyto3': 'Latest cytoplasm model',
}


def list_available_models(backend: str = 'stardist') -> Dict[str, str]:
    """List available pretrained models for the given backend."""
    if backend == 'cellpose':
        return CELLPOSE_MODELS.copy()
    return PRETRAINED_MODELS.copy()
