"""
colocalization.py - Signal colocalization analysis for BrainSlice

Measures signal intensity in detected nuclei and classifies as positive/negative
based on comparison to tissue background.

Usage:
    from brainslice.core.colocalization import ColocalizationAnalyzer

    analyzer = ColocalizationAnalyzer()
    background = analyzer.estimate_background(green_channel, nuclei_labels)
    measurements = analyzer.measure_nuclei_intensities(green_channel, nuclei_labels)
    results = analyzer.classify_positive_negative(measurements, background, threshold=2.0)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

# ============================================================================
# Display normalization — per-image auto-normalization
# ============================================================================
# Legacy hardcoded constants kept for backwards compatibility.
DISPLAY_RED_FLOOR = 25
DISPLAY_RED_MAX = 221
DISPLAY_RED_GAMMA = 0.8
DISPLAY_GRN_FLOOR = 32
DISPLAY_GRN_MAX = 1333
DISPLAY_GRN_GAMMA = 2.0


def normalize_for_display_auto(image: np.ndarray,
                                channel: str = 'green') -> np.ndarray:
    """Auto-normalize image for display with background-balanced gamma.

    Both channels map their median (background) to the same target brightness
    so the composite background appears as dim gray (magenta + green = white,
    dim white = gray).

    Strategy:
    1. Floor at p1 (clip hot-pixel noise), max at p99.9 (signal)
    2. Compute per-channel gamma so that the median maps to TARGET_BG
    3. This equalizes background brightness across channels regardless
       of their raw intensity distributions

    Args:
        image: Raw intensity image (any dtype).
        channel: 'green' or 'red' (gamma computed per-channel).

    Returns:
        Normalized float64 array in [0, 1].
    """
    TARGET_BG = 0.06  # background display brightness (dim gray)

    img = image.astype(np.float64)
    p1 = np.percentile(img, 1)
    p60 = np.percentile(img, 60)
    p999 = np.percentile(img, 99.9)

    floor = p1
    display_max = max(p999, floor + 1)
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)

    # Compute gamma so p60 (upper background) maps to TARGET_BG
    # Using p60 instead of p50 ensures the brighter background pixels
    # are also suppressed, not just the median
    ref_linear = np.clip((p60 - floor) / (display_max - floor), 1e-6, 1.0)
    gamma = np.log(TARGET_BG) / np.log(ref_linear)
    gamma = np.clip(gamma, 0.3, 3.0)  # sane bounds

    return np.power(linear, gamma)


def normalize_for_display(image: np.ndarray, display_max: float,
                          floor: float = 0,
                          gamma: float = 1.0) -> np.ndarray:
    """Normalize image for display with floor subtraction and gamma correction.

    Maps [floor, display_max] → [0, 1], then applies gamma curve.
    Matches napari's display pipeline: contrast_limits → gamma.

    Args:
        image: Raw intensity image.
        display_max: Upper contrast limit (maps to 1.0).
        floor: Lower contrast limit (maps to 0.0).
        gamma: Gamma correction exponent. <1 brightens midtones (boosts
            bright signal), >1 darkens midtones (expands dim signal
            differences). 1.0 = linear (no correction).
    """
    linear = np.clip(
        (image.astype(np.float64) - floor) / (display_max - floor), 0, 1
    )
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear


# Lazy imports
_pd = None
_ndimage = None


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_ndimage():
    """Lazy import scipy.ndimage."""
    global _ndimage
    if _ndimage is None:
        from scipy import ndimage
        _ndimage = ndimage
    return _ndimage


class ColocalizationAnalyzer:
    """
    Measure signal intensity in nuclei and classify positive/negative.

    Critical design note: Background is estimated from BRAIN TISSUE
    (not extracellular space) to avoid false positives from comparing
    to the dark regions outside the brain.
    """

    def __init__(
        self,
        background_method: str = 'gmm',
        background_percentile: float = 10.0,
    ):
        """
        Initialize analyzer.

        Args:
            background_method: Method for background estimation
                - 'gmm': Gaussian mixture model (recommended) — fits 1-2 components
                  to tissue intensity, uses the lower component as background.
                  Provides confidence via component separation.
                - 'percentile': Use percentile of tissue intensity
                - 'mode': Use mode of tissue intensity distribution
                - 'mean': Use mean of tissue (excluding high outliers)
            background_percentile: Percentile to use for percentile method
        """
        self.background_method = background_method
        self.background_percentile = background_percentile
        # Populated after estimate_background() when using 'gmm'
        self.background_diagnostics = None

    def estimate_tissue_mask(
        self,
        nuclei_labels: np.ndarray,
        dilation_iterations: int = 20,
    ) -> np.ndarray:
        """
        Create a mask of brain tissue based on where nuclei are detected.

        The idea: nuclei are only found in brain tissue, so we dilate
        the nuclei regions to get an approximation of brain tissue.

        Args:
            nuclei_labels: Label image from detection
            dilation_iterations: How much to dilate (larger = more tissue)

        Returns:
            Boolean mask where True = tissue
        """
        ndimage = _get_ndimage()

        # Start with nuclei locations
        nuclei_mask = nuclei_labels > 0

        # Dilate to include surrounding tissue
        tissue_mask = ndimage.binary_dilation(
            nuclei_mask,
            iterations=dilation_iterations
        )

        return tissue_mask

    def _estimate_background_gmm(self, tissue_pixels: np.ndarray) -> float:
        """
        Estimate background using a Gaussian Mixture Model.

        Fits 1- and 2-component GMMs to tissue-outside-nuclei intensity.
        If a 2-component model is better (by BIC), the lower-mean component
        is taken as true background. If 1-component fits better, the tissue
        is uniform and we use that single component's mean.

        Stores diagnostics in self.background_diagnostics with:
            - n_components: 1 or 2 (which model won)
            - background_mean: mean of the background component
            - background_std: std of the background component
            - bic_1: BIC for 1-component model
            - bic_2: BIC for 2-component model
            - separation: (higher_mean - lower_mean) / lower_std — how well-
              separated the components are. Higher = more confident.
              Only present when n_components=2.
            - background_weight: mixing weight of the background component
            - signal_bleed_mean: mean of the higher component (if 2-component)

        Returns:
            Background intensity (mean of the background component)
        """
        from sklearn.mixture import GaussianMixture

        # Subsample if very large (GMM on millions of pixels is slow)
        if len(tissue_pixels) > 200_000:
            rng = np.random.default_rng(42)
            sample = rng.choice(tissue_pixels, size=200_000, replace=False)
        else:
            sample = tissue_pixels

        X = sample.reshape(-1, 1).astype(np.float64)

        # Fit 1-component
        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm1.fit(X)
        bic1 = gmm1.bic(X)

        # Fit 2-component
        gmm2 = GaussianMixture(n_components=2, random_state=42)
        gmm2.fit(X)
        bic2 = gmm2.bic(X)

        if bic2 < bic1:
            # 2-component model is better — tissue has two populations
            means = gmm2.means_.flatten()
            stds = np.sqrt(gmm2.covariances_.flatten())
            weights = gmm2.weights_.flatten()

            # Lower-mean component is background
            bg_idx = np.argmin(means)
            sig_idx = 1 - bg_idx

            background = float(means[bg_idx])
            separation = float((means[sig_idx] - means[bg_idx]) / (stds[bg_idx] + 1e-10))

            self.background_diagnostics = {
                'n_components': 2,
                'background_mean': float(means[bg_idx]),
                'background_std': float(stds[bg_idx]),
                'background_weight': float(weights[bg_idx]),
                'signal_bleed_mean': float(means[sig_idx]),
                'signal_bleed_std': float(stds[sig_idx]),
                'signal_bleed_weight': float(weights[sig_idx]),
                'separation': separation,
                'bic_1': float(bic1),
                'bic_2': float(bic2),
                'confidence': 'high' if separation > 2.0 else 'moderate' if separation > 1.0 else 'low',
            }
        else:
            # 1-component model is better — tissue is uniform
            background = float(gmm1.means_.flatten()[0])
            std = float(np.sqrt(gmm1.covariances_.flatten()[0]))

            self.background_diagnostics = {
                'n_components': 1,
                'background_mean': background,
                'background_std': std,
                'background_weight': 1.0,
                'bic_1': float(bic1),
                'bic_2': float(bic2),
                'separation': None,
                'confidence': 'high',  # uniform tissue = unambiguous background
            }

        return background

    def estimate_background(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
        dilation_iterations: int = 20,
        cell_body_dilation: int = 8,
    ) -> float:
        """
        Estimate brain tissue background intensity.

        CRITICAL: We estimate background from BRAIN TISSUE, not from
        extracellular/empty space. This prevents false positives where
        any signal looks "bright" compared to the black background.

        The exclusion zone around each nucleus is dilated by cell_body_dilation
        to exclude the cytoplasm/cell body, which extends beyond the nucleus
        boundary and may contain signal fluorescence. Without this, signal
        from positive cell bodies contaminates the background estimate upward.

        Args:
            signal_image: Signal channel (e.g., green/eYFP)
            nuclei_labels: Label image of detected nuclei
            tissue_mask: Optional pre-computed tissue mask
            dilation_iterations: Dilation for tissue mask estimation
            cell_body_dilation: Dilation of nuclei mask to approximate cell
                body extent. Excludes cytoplasm from background sampling.
                Default 8 pixels (~1 cell radius beyond nucleus edge).

        Returns:
            Background intensity value (float)
        """
        ndimage = _get_ndimage()
        self.background_diagnostics = None

        # Get tissue mask if not provided
        if tissue_mask is None:
            tissue_mask = self.estimate_tissue_mask(nuclei_labels, dilation_iterations)

        # Get tissue pixels that are OUTSIDE the cell body zone
        # We dilate nuclei by cell_body_dilation to approximate the full cell
        # body extent (nucleus + cytoplasm), so that signal from positive
        # cells' cytoplasm doesn't contaminate the background estimate.
        nuclei_mask = nuclei_labels > 0
        if cell_body_dilation > 0:
            cell_body_mask = ndimage.binary_dilation(
                nuclei_mask, iterations=cell_body_dilation
            )
        else:
            cell_body_mask = nuclei_mask
        tissue_outside_cells = tissue_mask & (~cell_body_mask)

        # Get intensity values in this region
        tissue_pixels = signal_image[tissue_outside_cells]

        if len(tissue_pixels) == 0:
            # Fallback: use all tissue
            tissue_pixels = signal_image[tissue_mask]

        if len(tissue_pixels) == 0:
            # Ultimate fallback: use whole image
            tissue_pixels = signal_image.ravel()

        # Estimate background based on method
        if self.background_method == 'gmm':
            if len(tissue_pixels) < 100:
                # Too few pixels for GMM, fall back to percentile
                background = np.percentile(tissue_pixels, self.background_percentile)
                self.background_diagnostics = {
                    'n_components': 0,
                    'confidence': 'low',
                    'fallback': 'percentile (too few tissue pixels for GMM)',
                }
            else:
                background = self._estimate_background_gmm(tissue_pixels)

        elif self.background_method == 'percentile':
            background = np.percentile(tissue_pixels, self.background_percentile)

        elif self.background_method == 'mode':
            # Estimate mode using histogram
            hist, bin_edges = np.histogram(tissue_pixels, bins=256)
            mode_idx = np.argmax(hist)
            background = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

        elif self.background_method == 'mean':
            # Mean excluding top 5% (bright outliers)
            cutoff = np.percentile(tissue_pixels, 95)
            background = np.mean(tissue_pixels[tissue_pixels < cutoff])

        else:
            raise ValueError(f"Unknown background method: {self.background_method}")

        # Store background std for z-score classification
        # Use only below-background pixels to avoid signal contamination.
        # The background estimate itself is robust (GMM), but std of ALL
        # tissue pixels is inflated by bright signal areas.
        below_bg = tissue_pixels[tissue_pixels <= background]
        if len(below_bg) > 10:
            self._background_std = float(np.std(below_bg))
        else:
            self._background_std = float(np.std(tissue_pixels))
        self._tissue_pixels = tissue_pixels  # keep for diagnostics

        return float(background)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2b (optional): Local/regional background estimation
    # ══════════════════════════════════════════════════════════════════════

    def estimate_local_background(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
        dilation_iterations: int = 20,
        cell_body_dilation: int = 8,
        block_size: int = 256,
        min_tissue_pixels: int = 100,
    ) -> np.ndarray:
        """
        Estimate spatially-varying background across the tissue.

        Divides the tissue into spatial blocks, estimates background per
        block, then interpolates to produce a smooth background surface.
        This handles the known limitation of tissue-wide background:
        when autofluorescence varies spatially (e.g., cortex vs hippocampus),
        a single background value produces systematic errors.

        Args:
            signal_image: Signal channel image
            nuclei_labels: Label image from detection
            tissue_mask: Optional pre-computed tissue mask
            dilation_iterations: For tissue mask estimation
            cell_body_dilation: For excluding cell bodies from background
            block_size: Size of spatial blocks in pixels
            min_tissue_pixels: Minimum tissue pixels per block for a valid estimate

        Returns:
            2D array (same shape as signal_image) with local background values.
            Pixels outside tissue get the nearest block's estimate.
        """
        ndimage = _get_ndimage()

        if tissue_mask is None:
            tissue_mask = self.estimate_tissue_mask(nuclei_labels, dilation_iterations)

        # Build cell body exclusion mask
        nuclei_mask = nuclei_labels > 0
        if cell_body_dilation > 0:
            cell_body_mask = ndimage.binary_dilation(nuclei_mask, iterations=cell_body_dilation)
        else:
            cell_body_mask = nuclei_mask
        bg_sample_mask = tissue_mask & (~cell_body_mask)

        h, w = signal_image.shape
        # Grid of block centers
        cy_list = list(range(block_size // 2, h, block_size))
        cx_list = list(range(block_size // 2, w, block_size))

        points = []  # (y, x) centers with valid estimates
        values = []  # background value at each center

        for cy in cy_list:
            for cx in cx_list:
                # Block bounds
                y0 = max(0, cy - block_size // 2)
                y1 = min(h, cy + block_size // 2)
                x0 = max(0, cx - block_size // 2)
                x1 = min(w, cx + block_size // 2)

                block_mask = bg_sample_mask[y0:y1, x0:x1]
                block_pixels = signal_image[y0:y1, x0:x1][block_mask]

                if len(block_pixels) < min_tissue_pixels:
                    continue

                # Estimate background for this block using configured method
                if self.background_method == 'gmm' and len(block_pixels) >= 100:
                    bg_val = self._estimate_background_gmm(block_pixels)
                elif self.background_method == 'percentile' or (self.background_method == 'gmm' and len(block_pixels) < 100):
                    bg_val = float(np.percentile(block_pixels, self.background_percentile))
                elif self.background_method == 'mode':
                    hist, bin_edges = np.histogram(block_pixels, bins=256)
                    mode_idx = np.argmax(hist)
                    bg_val = float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)
                elif self.background_method == 'mean':
                    cutoff = np.percentile(block_pixels, 95)
                    bg_val = float(np.mean(block_pixels[block_pixels < cutoff]))
                else:
                    bg_val = float(np.percentile(block_pixels, self.background_percentile))

                points.append((cy, cx))
                values.append(bg_val)

        if len(points) == 0:
            # No valid blocks — fall back to global estimate
            global_bg = self.estimate_background(
                signal_image, nuclei_labels, tissue_mask,
                dilation_iterations, cell_body_dilation)
            return np.full(signal_image.shape, global_bg, dtype=np.float32)

        if len(points) == 1:
            # Only one valid block — uniform background
            return np.full(signal_image.shape, values[0], dtype=np.float32)

        # Interpolate between block centers to create smooth surface
        from scipy.interpolate import griddata

        points_arr = np.array(points)
        values_arr = np.array(values)

        # Create coordinate grid for full image
        yy, xx = np.mgrid[0:h, 0:w]
        grid_coords = np.column_stack([yy.ravel(), xx.ravel()])

        # Interpolate (linear with nearest for extrapolation)
        bg_surface = griddata(
            points_arr, values_arr, grid_coords,
            method='linear', fill_value=np.nan,
        ).reshape(h, w)

        # Fill NaN regions (outside convex hull) with nearest neighbor
        nan_mask = np.isnan(bg_surface)
        if nan_mask.any():
            bg_nearest = griddata(
                points_arr, values_arr, grid_coords,
                method='nearest',
            ).reshape(h, w)
            bg_surface[nan_mask] = bg_nearest[nan_mask]

        return bg_surface.astype(np.float32)

    def measure_nuclei_intensities(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
    ):
        """
        Measure signal intensity for each detected nucleus.

        Args:
            signal_image: Signal channel (e.g., green/eYFP)
            nuclei_labels: Label image from detection

        Returns:
            DataFrame with columns:
            - label: nucleus ID
            - centroid_y, centroid_x: position
            - area: nucleus area in pixels
            - mean_intensity: mean signal in nucleus
            - median_intensity: median signal in nucleus
            - max_intensity: max signal in nucleus
            - integrated_intensity: sum of signal in nucleus
        """
        pd = _get_pandas()
        from skimage.measure import regionprops_table

        # Get basic properties with intensity
        props = regionprops_table(
            nuclei_labels,
            intensity_image=signal_image,
            properties=[
                'label', 'centroid', 'area',
                'intensity_mean', 'intensity_max'
            ]
        )

        df = pd.DataFrame({
            'label': props['label'],
            'centroid_y': props['centroid-0'],
            'centroid_x': props['centroid-1'],
            'area': props['area'],
            'mean_intensity': props['intensity_mean'],
            'max_intensity': props['intensity_max'],
        })

        # Calculate median and integrated intensity manually
        # (not available in regionprops_table)
        median_intensities = []
        integrated_intensities = []

        for label_id in df['label']:
            mask = nuclei_labels == label_id
            intensities = signal_image[mask]
            median_intensities.append(np.median(intensities))
            integrated_intensities.append(np.sum(intensities))

        df['median_intensity'] = median_intensities
        df['integrated_intensity'] = integrated_intensities

        return df

    def measure_soma_intensities(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        soma_dilation: int = 5,
        ring_only: bool = False,
    ):
        """
        Measure signal intensity in dilated soma regions around each nucleus.

        For retrograde-labeled neurons, the signal (e.g., eYFP) is cytoplasmic,
        not nuclear. So we need to look AROUND each nucleus (the soma/cell body)
        to determine whether a cell is positive.

        For each nucleus:
            1. Dilate the nucleus mask by soma_dilation pixels → soma ROI
            2. Measure mean green intensity in the soma ROI

        Args:
            signal_image: Signal channel (e.g., green/eYFP)
            nuclei_labels: Label image from detection
            soma_dilation: Dilation radius in pixels to create soma ROIs.
                Should roughly cover the cytoplasm around the nucleus.
            ring_only: If True, measure only the cytoplasm ring (soma minus
                nucleus). Critical for cytoplasmic-only signals (e.g., eYFP)
                where including nuclear pixels dilutes the measurement with
                near-zero values. Default False for backward compatibility.

        Returns:
            DataFrame with columns:
            - label: nucleus ID
            - centroid_y, centroid_x: nucleus centroid position
            - nucleus_area: area of the nucleus ROI (pixels)
            - soma_area: area of the dilated soma ROI (pixels)
            - soma_mean_intensity: mean signal in the soma ROI
            - soma_max_intensity: max signal in the soma ROI
            - soma_median_intensity: median signal in the soma ROI
        """
        pd = _get_pandas()
        ndimage = _get_ndimage()
        from skimage.measure import regionprops
        from skimage.morphology import disk

        selem = disk(soma_dilation)
        props = regionprops(nuclei_labels)

        rows = []
        for prop in props:
            # Get single-nucleus binary mask
            nucleus_mask = nuclei_labels == prop.label

            # Dilate to create soma region
            soma_mask = ndimage.binary_dilation(nucleus_mask, structure=selem)

            if ring_only:
                # Exclude nuclear pixels — measure only cytoplasm ring
                measurement_mask = soma_mask & ~nucleus_mask
            else:
                measurement_mask = soma_mask

            # Measure signal in the measurement region
            soma_pixels = signal_image[measurement_mask]

            if len(soma_pixels) == 0:
                # Edge case: ring is empty (nucleus fills entire dilation)
                soma_pixels = signal_image[soma_mask]

            rows.append({
                'label': prop.label,
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
                'nucleus_area': prop.area,
                'soma_area': int(measurement_mask.sum()),
                'soma_mean_intensity': float(np.mean(soma_pixels)),
                'soma_max_intensity': float(np.max(soma_pixels)),
                'soma_median_intensity': float(np.median(soma_pixels)),
            })

        return pd.DataFrame(rows)

    def measure_cytoplasm_intensities(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        expansion_px: int = 8,
    ):
        """
        Measure signal intensity in the cytoplasm ring around each nucleus
        using non-overlapping Voronoi-like expansion.

        This is the preferred method for dim cytoplasmic signals (e.g., eYFP)
        in dense tissue. Unlike measure_soma_intensities(), it:
        1. Uses expand_labels() for non-overlapping territory assignment
           (no double-counting between adjacent cells)
        2. Always excludes nuclear pixels (ring-only measurement)
        3. Runs in O(1) operations on the full label image (fast)

        For each nucleus:
            1. expand_labels assigns each pixel to the nearest nucleus
               (up to expansion_px distance)
            2. The cytoplasm ring = expanded territory minus nucleus
            3. Measure signal intensity only in this ring

        Args:
            signal_image: Signal channel (e.g., green/eYFP, dim cytoplasmic)
            nuclei_labels: Label image from nuclear detection
            expansion_px: Maximum expansion distance in pixels. Should roughly
                match the expected cell body radius beyond the nucleus edge.
                8-15 px typical for 20x brainstem neurons.

        Returns:
            DataFrame with columns:
            - label: nucleus ID
            - centroid_y, centroid_x: nucleus centroid position
            - nucleus_area: nucleus area (pixels)
            - cyto_area: cytoplasm ring area (pixels)
            - soma_mean_intensity: mean signal in cytoplasm ring
            - soma_max_intensity: max signal in cytoplasm ring
            - soma_median_intensity: median signal in cytoplasm ring
            - soma_p90_intensity: 90th percentile signal in ring
              (robust to hot pixels, better than max for dim signals)

        Note:
            Output column names use 'soma_*' prefix for compatibility with
            classify_positive_negative() which auto-detects soma columns.
        """
        pd = _get_pandas()
        from skimage.segmentation import expand_labels
        from skimage.measure import regionprops

        # Expand labels — each pixel assigned to nearest nucleus (Voronoi)
        expanded = expand_labels(nuclei_labels, distance=expansion_px)

        # Cytoplasm ring = expanded territory minus original nucleus
        cyto_labels = expanded.copy()
        cyto_labels[nuclei_labels > 0] = 0  # remove nuclear pixels

        # Measure in the ring for each nucleus
        nuc_props = regionprops(nuclei_labels)
        nuc_lookup = {p.label: p for p in nuc_props}

        rows = []
        for label_id in np.unique(expanded):
            if label_id == 0:
                continue

            nuc_prop = nuc_lookup.get(label_id)
            if nuc_prop is None:
                continue

            # Cytoplasm ring pixels
            ring_mask = cyto_labels == label_id
            ring_pixels = signal_image[ring_mask]

            if len(ring_pixels) == 0:
                # Fallback: use full expanded region
                full_mask = expanded == label_id
                ring_pixels = signal_image[full_mask]
                ring_area = int(full_mask.sum())
            else:
                ring_area = int(ring_mask.sum())

            # Nuclear intensity — used by adaptive GMM as fallback when
            # ring intensities are unimodal (cytoplasmic signals compress
            # the distribution, but nuclear intensities often stay bimodal)
            nuc_pixels = signal_image[nuclei_labels == label_id]
            nuc_mean = float(np.mean(nuc_pixels)) if len(nuc_pixels) > 0 else 0.0

            rows.append({
                'label': label_id,
                'centroid_y': nuc_prop.centroid[0],
                'centroid_x': nuc_prop.centroid[1],
                'nucleus_area': nuc_prop.area,
                'cyto_area': ring_area,
                'mean_intensity': nuc_mean,
                # Named soma_* for compat with classify_positive_negative()
                'soma_mean_intensity': float(np.mean(ring_pixels)),
                'soma_max_intensity': float(np.max(ring_pixels)),
                'soma_median_intensity': float(np.median(ring_pixels)),
                'soma_p90_intensity': float(np.percentile(ring_pixels, 90)),
            })

        return pd.DataFrame(rows)

    def classify_positive_negative(
        self,
        measurements,  # DataFrame
        background,  # float or 2D ndarray (from estimate_local_background)
        method: str = 'fold_change',
        threshold: float = 2.0,
        signal_image: Optional[np.ndarray] = None,
        nuclei_labels: Optional[np.ndarray] = None,
        area_fraction: float = 0.5,
        local_snr_radius: int = 100,
    ):
        """
        Classify each nucleus as positive or negative for signal.

        Args:
            measurements: DataFrame from measure_nuclei_intensities()
            background: Background intensity — either a scalar float from
                estimate_background() or a 2D array from estimate_local_background().
                When 2D, each nucleus gets its own background value looked up
                at its centroid position.
            method: Classification method
                - 'background_mean': **CURRENT DEFAULT** (under active
                  development). Estimates tissue green background (excluding
                  black areas and generously dilated somas), then classifies
                  as positive if green ROI > background mean. Simple,
                  interpretable, no tuning needed. Requires signal_image
                  and nuclei_labels.
                All methods below retained for comparison during development:
                - 'fold_change': positive if mean >= threshold * background
                - 'local_snr': positive if local signal-to-noise ratio exceeds
                  threshold. For each cell, estimates background from nearby
                  tissue (within local_snr_radius, excluding all cell bodies),
                  then computes SNR = (signal - local_bg_mean) / local_bg_std.
                  Best for extremely dim signals with spatially varying
                  autofluorescence. Requires signal_image and nuclei_labels.
                - 'area_fraction': positive if >= area_fraction of nucleus
                  pixels exceed background * threshold. Requires signal_image
                  and nuclei_labels. This is the most robust method as it
                  checks spatial extent of signal within each nucleus.
                - 'absolute': positive if mean >= threshold (absolute value)
                - 'percentile': positive if mean >= threshold percentile of all nuclei
            threshold: Threshold value (interpretation depends on method).
                For local_snr: number of standard deviations above local
                background (e.g., 3.0 = 3 sigma). For area_fraction:
                fold-change each pixel must exceed.
            signal_image: Signal channel image (required for local_snr and
                area_fraction methods)
            nuclei_labels: Label image (required for local_snr and
                area_fraction methods)
            area_fraction: Fraction of nucleus area that must exceed the
                threshold to be classified as positive (default 0.5 = 50%).
                Only used with method='area_fraction'.
            local_snr_radius: Radius in pixels for local background estimation
                around each cell (default 100). Only used with method='local_snr'.

        Returns:
            DataFrame with added columns:
            - background: background value used (per-nucleus if local)
            - fold_change: mean_intensity / background
            - is_positive: True/False classification
            - local_snr: (local_snr method only) per-cell signal-to-noise ratio
            - positive_pixel_fraction: (area_fraction method only) fraction
              of nucleus pixels that exceeded threshold
        """
        pd = _get_pandas()
        df = measurements.copy()

        # Determine which intensity column to use for classification.
        # If soma_mean_intensity exists (from measure_soma_intensities), use it.
        # Otherwise use mean_intensity (from measure_nuclei_intensities).
        if 'soma_mean_intensity' in df.columns:
            intensity_col = 'soma_mean_intensity'
        else:
            intensity_col = 'mean_intensity'

        # Handle 2D background surface (from estimate_local_background)
        if isinstance(background, np.ndarray) and background.ndim == 2:
            # Look up per-nucleus background at each centroid
            per_nucleus_bg = []
            h, w = background.shape
            for _, row in df.iterrows():
                y = int(np.clip(round(row['centroid_y']), 0, h - 1))
                x = int(np.clip(round(row['centroid_x']), 0, w - 1))
                per_nucleus_bg.append(float(background[y, x]))
            df['background'] = per_nucleus_bg
        else:
            # Scalar background (existing behavior)
            df['background'] = float(background)

        df['fold_change'] = df[intensity_col] / (df['background'] + 1e-10)

        # Classify based on method
        if method == 'fold_change':
            df['is_positive'] = df['fold_change'] >= threshold

        elif method == 'background_mean':
            # PI's method: simple comparison to tissue background mean.
            #
            # 1. Exclude black/near-black areas (slide, empty regions)
            # 2. Dilate all nuclear ROIs generously to block all somas
            # 3. Average remaining green in tissue = background threshold
            # 4. Cell is positive if its green ROI mean > background mean
            #
            # No standard deviations, no Otsu, no tiers. The background
            # mean IS the threshold.
            if signal_image is None or nuclei_labels is None:
                raise ValueError(
                    "background_mean method requires signal_image and "
                    "nuclei_labels"
                )
            ndimage = _get_ndimage()
            from skimage.segmentation import expand_labels as _expand

            # Step 3a: tissue mask — exclude black/near-black areas
            # Anything below p5 is likely slide or empty space
            tissue_floor = np.percentile(signal_image, 5)
            tissue_mask = signal_image > tissue_floor

            # Step 3b: generous dilation to exclude ALL possible somas
            # Must be large enough that no cell body signal contaminates
            # the background estimate
            soma_dil = 8  # default
            if 'cyto_area' in df.columns and 'nucleus_area' in df.columns:
                avg_nuc = df['nucleus_area'].median()
                avg_ring = df['cyto_area'].median()
                if avg_nuc > 0:
                    import math
                    soma_dil = int(math.sqrt((avg_nuc + avg_ring) / math.pi)
                                   - math.sqrt(avg_nuc / math.pi))
            bg_excl_radius = max(soma_dil * 3, 25)
            expanded_all = _expand(nuclei_labels, distance=bg_excl_radius)
            soma_exclusion = expanded_all > 0

            # Step 3c: background = mean of green in tissue, outside somas
            bg_mask = tissue_mask & ~soma_exclusion
            bg_pixels = signal_image[bg_mask]

            if len(bg_pixels) < 100:
                # Dense image: fall back to tissue excluding just nuclei
                bg_mask_fallback = tissue_mask & (nuclei_labels == 0)
                bg_pixels = signal_image[bg_mask_fallback]

            bg_mean = float(np.mean(bg_pixels))
            # Use below-median pixels for robust std (avoids any signal bleed)
            below_med = bg_pixels[bg_pixels <= np.median(bg_pixels)]
            if len(below_med) > 10:
                bg_std = float(np.std(below_med))
            else:
                bg_std = float(np.std(bg_pixels))
            if bg_std < 1e-6:
                bg_std = max(bg_mean * 0.05, 1.0)

            # Step 4: classify — green ROI above background mean?
            df['is_positive'] = df[intensity_col] > bg_mean

            # How many std devs above background each cell is
            df['sigma_above_bg'] = (
                (df[intensity_col] - bg_mean) / bg_std
            )

            # Overwrite the background column with our computed value
            df['background'] = bg_mean
            df['bg_std'] = bg_std
            df['fold_change'] = df[intensity_col] / max(bg_mean, 1e-10)

            # Binning: how many cells at each sigma level
            sigmas = df['sigma_above_bg'].values
            n_above_0 = int(np.sum(sigmas > 0))
            n_above_1 = int(np.sum(sigmas > 1.0))
            n_above_1p5 = int(np.sum(sigmas > 1.5))
            n_above_2 = int(np.sum(sigmas > 2.0))
            n_above_3 = int(np.sum(sigmas > 3.0))

            self.adaptive_diagnostics = {
                'method_used': 'background_mean',
                'background_mean': bg_mean,
                'background_std': bg_std,
                'bg_excl_radius': bg_excl_radius,
                'tissue_pixels': int(np.sum(tissue_mask)),
                'excluded_pixels': int(np.sum(soma_exclusion & tissue_mask)),
                'bg_pixels_used': len(bg_pixels),
                'n_total': len(df),
                'n_above_bg': n_above_0,
                'n_above_1std': n_above_1,
                'n_above_1p5std': n_above_1p5,
                'n_above_2std': n_above_2,
                'n_above_3std': n_above_3,
            }

        elif method == 'local_snr':
            # Per-cell local signal-to-noise ratio classification.
            # For each cell, sample tissue background within local_snr_radius
            # (excluding ALL cell bodies), compute local mean and std, then
            # score by how many std devs above local background.
            #
            # Reports continuous scores. Binary threshold set by Otsu on the
            # positive scores (data-driven) or by the threshold parameter
            # as a fallback minimum.
            if signal_image is None or nuclei_labels is None:
                raise ValueError(
                    "local_snr method requires signal_image and nuclei_labels"
                )
            ndimage = _get_ndimage()
            from skimage.segmentation import expand_labels as _expand

            h, w = signal_image.shape

            # Pre-compute cell body exclusion mask using expand_labels.
            # Generous exclusion (3x soma_dilation or 25px minimum) ensures
            # signal from positive cell somas doesn't bleed into neighbors'
            # local background estimates. The enhancer-driven eYFP signal
            # can extend well beyond the nucleus boundary.
            # Infer soma_dilation from DataFrame columns or use generous default
            soma_dil = 8  # default if we can't infer
            if 'cyto_area' in df.columns and 'nucleus_area' in df.columns:
                # Estimate from first cell's ring size
                avg_nuc = df['nucleus_area'].median()
                avg_ring = df['cyto_area'].median()
                if avg_nuc > 0:
                    import math
                    soma_dil = int(math.sqrt((avg_nuc + avg_ring) / math.pi)
                                   - math.sqrt(avg_nuc / math.pi))
            excl_radius = max(soma_dil * 3, 25)
            expanded_all = _expand(nuclei_labels, distance=excl_radius)
            cell_body_excl = expanded_all > 0

            local_snr_values = []
            local_bg_means = []
            local_bg_stds = []

            for _, row in df.iterrows():
                cy = int(np.clip(round(row['centroid_y']), 0, h - 1))
                cx = int(np.clip(round(row['centroid_x']), 0, w - 1))

                # Local window bounds
                y0 = max(0, cy - local_snr_radius)
                y1 = min(h, cy + local_snr_radius)
                x0 = max(0, cx - local_snr_radius)
                x1 = min(w, cx + local_snr_radius)

                # Local tissue pixels outside ALL cell bodies
                local_bg_mask = ~cell_body_excl[y0:y1, x0:x1]
                local_pixels = signal_image[y0:y1, x0:x1][local_bg_mask]

                if len(local_pixels) < 20:
                    # Too few background pixels — use global background
                    local_bg_mean = float(background) if not isinstance(
                        background, np.ndarray) else float(np.mean(background))
                    local_bg_std = max(local_bg_mean * 0.1, 1.0)
                else:
                    # Use below-median pixels for std (robust to signal bleed)
                    local_bg_mean = float(np.mean(local_pixels))
                    below_med = local_pixels[local_pixels <= np.median(local_pixels)]
                    if len(below_med) > 5:
                        local_bg_std = float(np.std(below_med))
                    else:
                        local_bg_std = float(np.std(local_pixels))
                    if local_bg_std < 1e-6:
                        local_bg_std = max(local_bg_mean * 0.1, 1.0)

                cell_signal = row[intensity_col]
                snr = (cell_signal - local_bg_mean) / local_bg_std

                local_snr_values.append(float(snr))
                local_bg_means.append(local_bg_mean)
                local_bg_stds.append(local_bg_std)

            df['local_snr'] = local_snr_values
            df['local_bg_mean'] = local_bg_means
            df['local_bg_std'] = local_bg_stds
            # Continuous score — always reported
            df['positive_probability'] = np.clip(
                np.array(local_snr_values) / 10.0, 0.0, 1.0
            )

            # Data-driven threshold via Otsu on positive scores
            from skimage.filters import threshold_otsu
            snr_arr = np.array(local_snr_values)
            snr_positive = snr_arr[snr_arr > 0]
            otsu_thresh = threshold
            if len(snr_positive) >= 4:
                try:
                    otsu_val = threshold_otsu(snr_positive)
                    if otsu_val >= 1.0:  # sanity: at least 1 sigma
                        otsu_thresh = otsu_val
                except ValueError:
                    pass

            df['is_positive'] = snr_arr >= otsu_thresh

            self.adaptive_diagnostics = {
                'method_used': 'local_snr_otsu',
                'otsu_threshold': float(otsu_thresh),
                'fallback_threshold': float(threshold),
                'snr_max': float(snr_arr.max()) if len(snr_arr) > 0 else 0,
                'snr_median': float(np.median(snr_arr)),
                'n_above_zero': int(np.sum(snr_arr > 0)),
            }

        elif method == 'adaptive':
            # Adaptive GMM on nuclear intensities to find subpopulations
            df = self._classify_adaptive_gmm(df, intensity_col, threshold)

        elif method == 'area_fraction':
            if signal_image is None or nuclei_labels is None:
                raise ValueError(
                    "area_fraction method requires signal_image and nuclei_labels"
                )

            positive_fractions = []

            for idx, row in df.iterrows():
                label_id = row['label']
                mask = nuclei_labels == label_id
                pixel_values = signal_image[mask]
                # Use per-nucleus background for cutoff when local bg is used
                cell_bg = row['background']
                intensity_cutoff = cell_bg * threshold
                n_above = np.sum(pixel_values >= intensity_cutoff)
                frac = n_above / len(pixel_values) if len(pixel_values) > 0 else 0.0
                positive_fractions.append(float(frac))

            df['positive_pixel_fraction'] = positive_fractions
            df['is_positive'] = df['positive_pixel_fraction'] >= area_fraction

        elif method == 'absolute':
            df['is_positive'] = df['mean_intensity'] >= threshold

        elif method == 'percentile':
            cutoff = np.percentile(df['mean_intensity'], threshold)
            df['is_positive'] = df['mean_intensity'] >= cutoff

        elif method == 'zscore':
            df = self._classify_zscore(df, intensity_col, background)

        else:
            raise ValueError(f"Unknown classification method: {method}")

        return df

    def _classify_zscore(self, df, intensity_col, background):
        """Classify cells by z-score above background.

        Simple, transparent approach:
        1. Compute z-score for each cell: (intensity - bg_mean) / bg_std
        2. Try Otsu on z-scores to find natural break between populations
        3. If Otsu fails (unimodal), use fixed z > 3.0 cutoff

        No GMM, no cascade, no fallbacks. Works the same for nuclear
        or cytoplasm ring measurements because z-scoring normalizes.
        """
        from skimage.filters import threshold_otsu

        # Background stats — use per-cell background if available
        if 'background' in df.columns:
            bg_values = df['background'].values
        elif isinstance(background, np.ndarray) and background.ndim == 2:
            bg_values = np.full(len(df), float(np.mean(background)))
        else:
            bg_values = np.full(len(df), float(background))

        # Compute z-scores: how many background-stds above background
        # Use the background distribution's std (stored during estimation)
        bg_std = getattr(self, '_background_std', None)
        if bg_std is None or bg_std < 1e-6:
            # Estimate std from the per-cell background values or use
            # a fraction of the mean as fallback
            bg_mean = float(np.mean(bg_values))
            bg_std = max(bg_mean * 0.15, 1.0)

        intensities = df[intensity_col].values
        z_scores = (intensities - bg_values) / bg_std
        df['z_score'] = z_scores

        # Try Otsu on z-scores to find natural population break
        z_positive = z_scores[z_scores > 0]  # only above-background cells
        otsu_worked = False

        if len(z_positive) >= 4:
            try:
                otsu_thresh = threshold_otsu(z_positive)
                # Sanity: Otsu threshold should be at least 2.0 sigma
                if otsu_thresh >= 2.0:
                    z_threshold = otsu_thresh
                    otsu_worked = True
            except ValueError:
                pass

        if not otsu_worked:
            # Fixed cutoff: 3 sigma above background
            z_threshold = 3.0

        df['is_positive'] = z_scores >= z_threshold
        df['positive_probability'] = np.clip(
            (z_scores - z_threshold) / max(z_threshold, 1.0) * 0.5 + 0.5,
            0.0, 1.0
        )
        # Negative cells get probability based on distance below threshold
        neg_mask = ~df['is_positive']
        df.loc[neg_mask, 'positive_probability'] = np.clip(
            z_scores[neg_mask] / z_threshold * 0.5, 0.0, 0.49
        )

        self.adaptive_diagnostics = {
            'method_used': 'zscore_otsu' if otsu_worked else 'zscore_fixed',
            'z_threshold': float(z_threshold),
            'bg_std_used': float(bg_std),
            'otsu_worked': otsu_worked,
            'n_above_bg': int(np.sum(z_scores > 0)),
            'z_score_max': float(z_scores.max()) if len(z_scores) > 0 else 0,
            'z_score_median': float(np.median(z_scores)),
        }

        return df

    def _classify_adaptive_gmm(self, df, intensity_col, fallback_threshold):
        """Classify cells by fitting GMM to their signal intensity distribution.

        Uses a cascade strategy to find bimodal subpopulations:
        1. Try GMM on the primary intensity column (soma or nuclear)
        2. If that fails (unimodal), try GMM on fold_change values
        3. If that fails and nuclear intensities are available as a separate
           column, try GMM on those (nuclear signal often stays bimodal even
           when cytoplasm ring measurements compress the distribution)
        4. Only then fall back to conservative fold_change threshold

        Stores diagnostics in self.adaptive_diagnostics.
        """
        self.adaptive_diagnostics = None

        # Conservative fallback threshold
        safe_threshold = max(fallback_threshold, 2.0)

        # Need enough cells for a meaningful GMM fit
        if len(df) < 8:
            df['is_positive'] = df['fold_change'] >= safe_threshold
            df['positive_probability'] = np.where(
                df['is_positive'], 1.0, 0.0
            )
            self.adaptive_diagnostics = {
                'method_used': 'fallback_fold_change',
                'reason': f'Too few cells ({len(df)}) for adaptive GMM',
                'fallback_threshold': safe_threshold,
            }
            return df

        from sklearn.mixture import GaussianMixture

        # Build cascade of columns to try GMM on
        cascade = [(intensity_col, df[intensity_col].values)]

        # Try fold_change if available
        if 'fold_change' in df.columns:
            cascade.append(('fold_change', df['fold_change'].values))

        # Try nuclear intensities if we're using soma measurements
        if (intensity_col == 'soma_mean_intensity'
                and 'mean_intensity' in df.columns):
            cascade.append(('mean_intensity', df['mean_intensity'].values))

        for col_name, values in cascade:
            X = values.reshape(-1, 1).astype(np.float64)

            gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
            gmm2 = GaussianMixture(n_components=2, random_state=42).fit(X)

            bic1 = gmm1.bic(X)
            bic2 = gmm2.bic(X)

            if bic2 < bic1:
                # 2-component model found subpopulations
                means = gmm2.means_.flatten()
                stds = np.sqrt(gmm2.covariances_.flatten())
                weights = gmm2.weights_.flatten()

                hi_idx = int(np.argmax(means))
                lo_idx = 1 - hi_idx

                separation = float(
                    (means[hi_idx] - means[lo_idx])
                    / (stds[lo_idx] + 1e-10)
                )

                # Use posterior probability for soft classification
                probs = gmm2.predict_proba(X)
                df['positive_probability'] = probs[:, hi_idx]
                df['is_positive'] = df['positive_probability'] > 0.5

                # Threshold at intersection of the two Gaussians
                adaptive_thresh = float(
                    (means[lo_idx] * stds[hi_idx]
                     + means[hi_idx] * stds[lo_idx])
                    / (stds[lo_idx] + stds[hi_idx])
                )

                self.adaptive_diagnostics = {
                    'method_used': 'gmm_2component',
                    'gmm_column': col_name,
                    'bic_1': float(bic1),
                    'bic_2': float(bic2),
                    'negative_mean': float(means[lo_idx]),
                    'negative_std': float(stds[lo_idx]),
                    'negative_weight': float(weights[lo_idx]),
                    'positive_mean': float(means[hi_idx]),
                    'positive_std': float(stds[hi_idx]),
                    'positive_weight': float(weights[hi_idx]),
                    'separation': separation,
                    'adaptive_threshold': adaptive_thresh,
                }

                # ----- Ring rescue for Z-plane occlusion -----
                # When GMM classified on nuclear intensities, some true
                # positives may have low nuclear green because the nucleus
                # "punches a hole" in the cytoplasmic signal at this focal
                # plane.  Their ring (cytoplasm) signal is still strong.
                # Rescue negative cells whose ring fold_change matches
                # the positive population's ring signal.
                if (col_name == 'mean_intensity'
                        and 'soma_mean_intensity' in df.columns
                        and 'fold_change' in df.columns
                        and df['is_positive'].any()
                        and (~df['is_positive']).any()):
                    pos_fc = df.loc[df['is_positive'], 'fold_change']
                    # Threshold: 25th-percentile of positive fold_change,
                    # floored at 1.5x to avoid rescuing background cells
                    rescue_thresh = max(float(pos_fc.quantile(0.25)), 1.5)

                    neg_mask = ~df['is_positive']
                    rescue_mask = neg_mask & (df['fold_change'] >= rescue_thresh)
                    n_rescued = int(rescue_mask.sum())

                    if n_rescued > 0:
                        df.loc[rescue_mask, 'is_positive'] = True
                        # Confident but flagged — not from GMM posterior
                        df.loc[rescue_mask, 'positive_probability'] = 0.75
                        self.adaptive_diagnostics['ring_rescue'] = {
                            'n_rescued': n_rescued,
                            'rescue_threshold_fc': float(rescue_thresh),
                            'pos_fc_median': float(pos_fc.median()),
                            'pos_fc_p25': float(pos_fc.quantile(0.25)),
                        }

                return df

        # All cascade steps failed — no bimodal split found anywhere
        df['is_positive'] = df['fold_change'] >= safe_threshold
        df['positive_probability'] = np.where(
            df['is_positive'], 1.0, 0.0
        )

        tried = [col for col, _ in cascade]
        self.adaptive_diagnostics = {
            'method_used': 'fallback_fold_change',
            'reason': (f'GMM found 1 component on all cascade columns: '
                       f'{tried}'),
            'fallback_threshold': safe_threshold,
            }

        return df

    def get_summary_statistics(
        self,
        classified_measurements,  # DataFrame with is_positive column
    ) -> Dict[str, Any]:
        """
        Get summary statistics from classified measurements.

        Args:
            classified_measurements: DataFrame from classify_positive_negative()

        Returns:
            Dict with summary stats:
            - total_cells: total number of nuclei
            - positive_cells: number classified as positive
            - negative_cells: number classified as negative
            - positive_fraction: fraction positive
            - mean_fold_change: average fold change
            - background_used: background value
        """
        df = classified_measurements

        total = len(df)
        positive = df['is_positive'].sum()
        negative = total - positive

        summary = {
            'total_cells': int(total),
            'positive_cells': int(positive),
            'negative_cells': int(negative),
            'positive_fraction': float(positive / total) if total > 0 else 0.0,
            'mean_fold_change': float(df['fold_change'].mean()) if len(df) > 0 else 0.0,
            'median_fold_change': float(df['fold_change'].median()) if len(df) > 0 else 0.0,
            'background_used': float(df['background'].iloc[0]) if len(df) > 0 else 0.0,
        }

        # Include GMM diagnostics if available
        if self.background_diagnostics is not None:
            summary['background_diagnostics'] = self.background_diagnostics

        # Include adaptive classification diagnostics if available
        if self.adaptive_diagnostics is not None:
            summary['adaptive_diagnostics'] = self.adaptive_diagnostics

        return summary


def compute_colocalization_metrics(
    red_image: np.ndarray,
    green_image: np.ndarray,
    nuclei_labels: np.ndarray,
    background_green: float,
    tissue_mask: Optional[np.ndarray] = None,
    soma_dilation: int = 0,
) -> Dict[str, float]:
    """
    Compute standard colocalization metrics (Pearson, Manders) as validation.

    These whole-image metrics serve as a cross-check against per-cell
    classification. If Manders' M1 says 25% of red signal overlaps green
    but per-cell classification says 80% positive, something is wrong.

    Metrics computed:
    - **Pearson's r**: Overall linear correlation between channels within tissue.
      Ranges -1 (anti-correlated) to +1 (perfectly correlated). Insensitive to
      intensity differences between channels.

    - **Manders' M1**: Fraction of red-channel signal that co-occurs with
      above-background green signal. Answers: "of all nuclear signal, how much
      is in regions with green signal?"

    - **Manders' M2**: Fraction of green-channel signal that co-occurs with
      red nuclei (or soma ROIs). Answers: "of all green signal, how much is
      near detected nuclei?"

    Args:
        red_image: Nuclear channel (2D)
        green_image: Signal channel (2D)
        nuclei_labels: Label image from detection (0 = background)
        background_green: Estimated green background level (used as Manders threshold)
        tissue_mask: Optional tissue mask. If None, uses entire image.
        soma_dilation: If > 0, dilate nuclei to create soma ROIs for Manders' M2.

    Returns:
        Dict with keys: pearson_r, manders_m1, manders_m2
    """
    ndimage = _get_ndimage()

    red = red_image.astype(np.float64)
    green = green_image.astype(np.float64)

    # Restrict analysis to tissue region
    if tissue_mask is not None:
        mask = tissue_mask
    else:
        mask = np.ones(red.shape, dtype=bool)

    # ── Pearson's correlation (within tissue) ──
    r_vals = red[mask]
    g_vals = green[mask]
    if len(r_vals) > 1 and np.std(r_vals) > 0 and np.std(g_vals) > 0:
        pearson_r = float(np.corrcoef(r_vals, g_vals)[0, 1])
    else:
        pearson_r = 0.0

    # ── Manders' thresholded coefficients ──
    # Green signal mask: where green is above background
    green_above_bg = green > background_green

    # Red signal mask: where nuclei are (or soma ROIs if dilated)
    nuclei_mask = nuclei_labels > 0
    if soma_dilation > 0:
        from skimage.morphology import disk
        selem = disk(soma_dilation)
        red_roi_mask = ndimage.binary_dilation(nuclei_mask, structure=selem)
    else:
        red_roi_mask = nuclei_mask

    # M1: fraction of red (nuclear) signal in regions with above-background green
    # "Of all nuclear signal, how much co-occurs with green signal?"
    red_in_green = red[mask & nuclei_mask & green_above_bg].sum()
    red_total = red[mask & nuclei_mask].sum()
    manders_m1 = float(red_in_green / red_total) if red_total > 0 else 0.0

    # M2: fraction of green signal in regions with red nuclei/somas
    # "Of all green signal in tissue, how much is within cell ROIs?"
    green_in_red = green[mask & red_roi_mask].sum()
    green_total = green[mask].sum()
    manders_m2 = float(green_in_red / green_total) if green_total > 0 else 0.0

    return {
        'pearson_r': round(pearson_r, 4),
        'manders_m1': round(manders_m1, 4),
        'manders_m2': round(manders_m2, 4),
    }


def analyze_colocalization(
    signal_image: np.ndarray,
    nuclei_labels: np.ndarray,
    background_method: str = 'percentile',
    background_percentile: float = 10.0,
    threshold_method: str = 'fold_change',
    threshold_value: float = 2.0,
    cell_body_dilation: int = 8,
    area_fraction: float = 0.5,
) -> Tuple[Any, Dict[str, Any]]:  # Returns (DataFrame, summary_dict)
    """
    Convenience function for complete colocalization analysis.

    Args:
        signal_image: Signal channel to measure
        nuclei_labels: Label image from detection
        background_method: Method for background estimation
        background_percentile: Percentile for background (if using percentile method)
        threshold_method: Method for positive/negative classification
            'fold_change': mean intensity >= threshold * background
            'area_fraction': >= area_fraction of nucleus pixels exceed threshold * background
        threshold_value: Threshold for classification
        cell_body_dilation: Pixels to dilate nuclei when excluding from background
            sampling. Prevents cell body signal from contaminating background.
        area_fraction: For area_fraction method, fraction of nucleus that must
            exceed threshold to be positive (default 0.5 = 50%).

    Returns:
        Tuple of (measurements_df, summary_dict)
    """
    analyzer = ColocalizationAnalyzer(
        background_method=background_method,
        background_percentile=background_percentile,
    )

    background = analyzer.estimate_background(
        signal_image, nuclei_labels,
        cell_body_dilation=cell_body_dilation,
    )
    measurements = analyzer.measure_nuclei_intensities(signal_image, nuclei_labels)
    classified = analyzer.classify_positive_negative(
        measurements, background,
        method=threshold_method,
        threshold=threshold_value,
        signal_image=signal_image if threshold_method == 'area_fraction' else None,
        nuclei_labels=nuclei_labels if threshold_method == 'area_fraction' else None,
        area_fraction=area_fraction,
    )
    summary = analyzer.get_summary_statistics(classified)

    return classified, summary


def analyze_dual_colocalization(
    signal_image_1: np.ndarray,
    signal_image_2: np.ndarray,
    nuclei_labels: np.ndarray,
    # Channel 1 params
    background_method_1: str = 'gmm',
    background_percentile_1: float = 10.0,
    threshold_method_1: str = 'fold_change',
    threshold_value_1: float = 2.0,
    cell_body_dilation_1: int = 8,
    area_fraction_1: float = 0.5,
    soma_dilation_1: int = 0,
    # Channel 2 params
    background_method_2: str = 'gmm',
    background_percentile_2: float = 10.0,
    threshold_method_2: str = 'local_snr',
    threshold_value_2: float = 3.0,
    cell_body_dilation_2: int = 8,
    area_fraction_2: float = 0.5,
    soma_dilation_2: int = 5,
    # Measurement mode
    use_cytoplasm_ring: bool = True,
    # Channel labels
    ch1_name: str = 'red',
    ch2_name: str = 'green',
) -> Tuple[Any, Dict[str, Any]]:
    """
    Dual-channel colocalization: classify each nucleus in two signal channels
    independently, then cross-tabulate into dual/ch1-only/ch2-only/neither.

    Designed for retrograde tracing experiments where two viral tracers
    (e.g., H2B-mCherry + eYFP) are co-injected and each channel needs
    independent background estimation and thresholding.

    Args:
        signal_image_1: First signal channel (e.g., mCherry/red, 561nm)
        signal_image_2: Second signal channel (e.g., eYFP/green, 488nm)
        nuclei_labels: Label image from nuclear detection
        background_method_1/2: Background estimation method per channel
        background_percentile_1/2: Percentile for background (if percentile method)
        threshold_method_1/2: Classification method per channel.
            Recommended: 'fold_change' for bright nuclear ch1,
            'local_snr' for dim cytoplasmic ch2.
        threshold_value_1/2: Threshold per channel.
            For fold_change: fold-change multiplier (e.g., 2.0).
            For local_snr: number of std devs above local background (e.g., 3.0).
        cell_body_dilation_1/2: Nuclei exclusion radius for background estimation
        area_fraction_1/2: For area_fraction method
        soma_dilation_1/2: Soma dilation for intensity measurement.
            0 = measure within nucleus (nuclear-localized signals like H2B).
            >0 = measure in dilated soma ring (cytoplasmic signals like eYFP).
        use_cytoplasm_ring: If True (default) and soma_dilation > 0, use
            non-overlapping expand_labels with nuclear pixel exclusion
            (measure_cytoplasm_intensities). If False, use legacy per-nucleus
            dilation (measure_soma_intensities). The new method is faster,
            avoids overlapping regions, and excludes nuclear pixels that
            would dilute cytoplasmic signal measurements.
        ch1_name/ch2_name: Channel labels for output columns

    Returns:
        Tuple of (measurements_df, summary_dict)
        measurements_df has columns:
            label, centroid_y, centroid_x, area,
            mean_intensity_ch1, fold_change_ch1, is_positive_ch1,
            mean_intensity_ch2, fold_change_ch2, is_positive_ch2,
            classification ('dual', 'ch1_only', 'ch2_only', 'neither'),
            is_positive (ch1 | ch2, for backward compat)
    """
    pd = _get_pandas()

    def _measure_channel(analyzer, signal_image, nuclei_labels, soma_dilation,
                         use_ring):
        """Choose measurement method based on soma_dilation and use_ring."""
        if soma_dilation > 0 and use_ring:
            return analyzer.measure_cytoplasm_intensities(
                signal_image, nuclei_labels, expansion_px=soma_dilation,
            )
        elif soma_dilation > 0:
            return analyzer.measure_soma_intensities(
                signal_image, nuclei_labels, soma_dilation=soma_dilation,
                ring_only=True,
            )
        else:
            return analyzer.measure_nuclei_intensities(
                signal_image, nuclei_labels,
            )

    def _classify_channel(analyzer, meas, bg, method, threshold, signal_image,
                          nuclei_labels, area_fraction):
        """Build classify kwargs and run classification."""
        kw = {'method': method, 'threshold': threshold}
        if method == 'area_fraction':
            kw['signal_image'] = signal_image
            kw['nuclei_labels'] = nuclei_labels
            kw['area_fraction'] = area_fraction
        elif method == 'local_snr':
            kw['signal_image'] = signal_image
            kw['nuclei_labels'] = nuclei_labels
        return analyzer.classify_positive_negative(meas, bg, **kw)

    # --- Channel 1 analysis ---
    analyzer1 = ColocalizationAnalyzer(
        background_method=background_method_1,
        background_percentile=background_percentile_1,
    )
    bg1 = analyzer1.estimate_background(
        signal_image_1, nuclei_labels,
        cell_body_dilation=cell_body_dilation_1,
    )
    meas1 = _measure_channel(
        analyzer1, signal_image_1, nuclei_labels,
        soma_dilation_1, use_cytoplasm_ring,
    )
    classified1 = _classify_channel(
        analyzer1, meas1, bg1, threshold_method_1, threshold_value_1,
        signal_image_1, nuclei_labels, area_fraction_1,
    )

    # --- Channel 2 analysis ---
    analyzer2 = ColocalizationAnalyzer(
        background_method=background_method_2,
        background_percentile=background_percentile_2,
    )
    bg2 = analyzer2.estimate_background(
        signal_image_2, nuclei_labels,
        cell_body_dilation=cell_body_dilation_2,
    )
    meas2 = _measure_channel(
        analyzer2, signal_image_2, nuclei_labels,
        soma_dilation_2, use_cytoplasm_ring,
    )
    classified2 = _classify_channel(
        analyzer2, meas2, bg2, threshold_method_2, threshold_value_2,
        signal_image_2, nuclei_labels, area_fraction_2,
    )

    # --- Merge ---
    # Shared spatial columns from ch1
    shared_cols = ['label', 'centroid_y', 'centroid_x', 'area']
    # Handle _base variants if present
    for col in ['centroid_y_base', 'centroid_x_base']:
        if col in classified1.columns:
            shared_cols.append(col)

    merged = classified1[shared_cols].copy()

    # Add ch1 intensity/classification columns with suffix
    intensity_col_1 = 'soma_mean_intensity' if 'soma_mean_intensity' in classified1.columns else 'mean_intensity'
    merged[f'mean_intensity_{ch1_name}'] = classified1[intensity_col_1].values
    merged[f'background_{ch1_name}'] = classified1['background'].values
    merged[f'fold_change_{ch1_name}'] = classified1['fold_change'].values
    merged[f'is_positive_{ch1_name}'] = classified1['is_positive'].values

    # Add ch2 intensity/classification columns with suffix
    intensity_col_2 = 'soma_mean_intensity' if 'soma_mean_intensity' in classified2.columns else 'mean_intensity'
    merged[f'mean_intensity_{ch2_name}'] = classified2[intensity_col_2].values
    merged[f'background_{ch2_name}'] = classified2['background'].values
    merged[f'fold_change_{ch2_name}'] = classified2['fold_change'].values
    merged[f'is_positive_{ch2_name}'] = classified2['is_positive'].values

    # --- Cross-tabulate ---
    pos1 = merged[f'is_positive_{ch1_name}']
    pos2 = merged[f'is_positive_{ch2_name}']
    merged['classification'] = 'neither'
    merged.loc[pos1 & pos2, 'classification'] = 'dual'
    merged.loc[pos1 & ~pos2, 'classification'] = f'{ch1_name}_only'
    merged.loc[~pos1 & pos2, 'classification'] = f'{ch2_name}_only'

    # Backward compat: positive in either channel
    merged['is_positive'] = pos1 | pos2

    # --- Summary ---
    total = len(merged)
    n_ch1 = int(pos1.sum())
    n_ch2 = int(pos2.sum())
    n_dual = int((pos1 & pos2).sum())
    n_ch1_only = int((pos1 & ~pos2).sum())
    n_ch2_only = int((~pos1 & pos2).sum())
    n_neither = int((~pos1 & ~pos2).sum())

    summary = {
        'total_nuclei': total,
        f'n_{ch1_name}_positive': n_ch1,
        f'n_{ch2_name}_positive': n_ch2,
        'n_dual': n_dual,
        f'n_{ch1_name}_only': n_ch1_only,
        f'n_{ch2_name}_only': n_ch2_only,
        'n_neither': n_neither,
        f'fraction_{ch1_name}': float(n_ch1 / total) if total > 0 else 0.0,
        f'fraction_{ch2_name}': float(n_ch2 / total) if total > 0 else 0.0,
        'fraction_dual': float(n_dual / total) if total > 0 else 0.0,
        f'background_{ch1_name}': float(bg1) if not isinstance(bg1, np.ndarray) else float(np.mean(bg1)),
        f'background_{ch2_name}': float(bg2) if not isinstance(bg2, np.ndarray) else float(np.mean(bg2)),
        f'bg_diagnostics_{ch1_name}': analyzer1.background_diagnostics,
        f'bg_diagnostics_{ch2_name}': analyzer2.background_diagnostics,
        'ch1_name': ch1_name,
        'ch2_name': ch2_name,
    }

    return merged, summary


def filter_measurements_by_roi(
    measurements,  # DataFrame with centroid columns
    roi_vertices: np.ndarray,  # Nx2 array of (y, x) polygon vertices
    image_shape: Tuple[int, int],  # (height, width) for rasterization
):
    """
    Filter cell measurements to only those inside an ROI polygon.

    Args:
        measurements: DataFrame with centroid_y/centroid_x (or _base variants)
        roi_vertices: Nx2 array of (row, col) polygon vertices
        image_shape: (height, width) of the image

    Returns:
        Filtered DataFrame containing only cells inside the ROI
    """
    from skimage.draw import polygon as draw_polygon

    rr, cc = draw_polygon(roi_vertices[:, 0], roi_vertices[:, 1], shape=image_shape)
    roi_mask = np.zeros(image_shape, dtype=bool)
    roi_mask[rr, cc] = True

    y_col = 'centroid_y_base' if 'centroid_y_base' in measurements.columns else 'centroid_y'
    x_col = 'centroid_x_base' if 'centroid_x_base' in measurements.columns else 'centroid_x'

    inside = []
    for _, row in measurements.iterrows():
        y, x = int(round(row[y_col])), int(round(row[x_col]))
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            inside.append(roi_mask[y, x])
        else:
            inside.append(False)

    return measurements[inside].copy()
