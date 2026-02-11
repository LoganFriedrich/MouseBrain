"""
colocalization.py - Signal colocalization analysis for BrainSlice

Measures signal intensity in detected nuclei and classifies as positive/negative
based on comparison to tissue background.

Usage:
    from mousebrain.pipeline_2d.sliceatlas.core.colocalization import ColocalizationAnalyzer

    analyzer = ColocalizationAnalyzer()
    background = analyzer.estimate_background(green_channel, nuclei_labels)
    measurements = analyzer.measure_nuclei_intensities(green_channel, nuclei_labels)
    results = analyzer.classify_positive_negative(measurements, background, threshold=2.0)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

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

    def classify_positive_negative(
        self,
        measurements,  # DataFrame
        background,  # float or 2D ndarray (from estimate_local_background)
        method: str = 'fold_change',
        threshold: float = 2.0,
        signal_image: Optional[np.ndarray] = None,
        nuclei_labels: Optional[np.ndarray] = None,
        area_fraction: float = 0.5,
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
                - 'fold_change': positive if mean >= threshold * background
                - 'area_fraction': positive if >= area_fraction of nucleus
                  pixels exceed background * threshold. Requires signal_image
                  and nuclei_labels. This is the most robust method as it
                  checks spatial extent of signal within each nucleus.
                - 'absolute': positive if mean >= threshold (absolute value)
                - 'percentile': positive if mean >= threshold percentile of all nuclei
            threshold: Threshold value (interpretation depends on method).
                For area_fraction: fold-change each pixel must exceed.
            signal_image: Signal channel image (required for area_fraction method)
            nuclei_labels: Label image (required for area_fraction method)
            area_fraction: Fraction of nucleus area that must exceed the
                threshold to be classified as positive (default 0.5 = 50%).
                Only used with method='area_fraction'.

        Returns:
            DataFrame with added columns:
            - background: background value used (per-nucleus if local)
            - fold_change: mean_intensity / background
            - is_positive: True/False classification
            - positive_pixel_fraction: (area_fraction method only) fraction
              of nucleus pixels that exceeded threshold
        """
        pd = _get_pandas()
        df = measurements.copy()

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
            df['fold_change'] = df['mean_intensity'] / (df['background'] + 1e-10)
        else:
            # Scalar background (existing behavior)
            df['background'] = float(background)
            df['fold_change'] = df['mean_intensity'] / (float(background) + 1e-10)

        # Classify based on method
        if method == 'fold_change':
            df['is_positive'] = df['fold_change'] >= threshold

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

        else:
            raise ValueError(f"Unknown classification method: {method}")

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

        return summary


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
