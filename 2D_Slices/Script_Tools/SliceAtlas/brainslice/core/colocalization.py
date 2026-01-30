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
        background_method: str = 'percentile',
        background_percentile: float = 10.0,
    ):
        """
        Initialize analyzer.

        Args:
            background_method: Method for background estimation
                - 'percentile': Use percentile of tissue intensity (recommended)
                - 'mode': Use mode of tissue intensity distribution
                - 'mean': Use mean of tissue (excluding high outliers)
            background_percentile: Percentile to use for percentile method
        """
        self.background_method = background_method
        self.background_percentile = background_percentile

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

    def estimate_background(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
        dilation_iterations: int = 20,
    ) -> float:
        """
        Estimate brain tissue background intensity.

        CRITICAL: We estimate background from BRAIN TISSUE, not from
        extracellular/empty space. This prevents false positives where
        any signal looks "bright" compared to the black background.

        Args:
            signal_image: Signal channel (e.g., green/eYFP)
            nuclei_labels: Label image of detected nuclei
            tissue_mask: Optional pre-computed tissue mask
            dilation_iterations: Dilation for tissue mask estimation

        Returns:
            Background intensity value (float)
        """
        # Get tissue mask if not provided
        if tissue_mask is None:
            tissue_mask = self.estimate_tissue_mask(nuclei_labels, dilation_iterations)

        # Get tissue pixels that are OUTSIDE nuclei
        # (we want background tissue, not the signal-containing cells)
        nuclei_mask = nuclei_labels > 0
        tissue_outside_nuclei = tissue_mask & (~nuclei_mask)

        # Get intensity values in this region
        tissue_pixels = signal_image[tissue_outside_nuclei]

        if len(tissue_pixels) == 0:
            # Fallback: use all tissue
            tissue_pixels = signal_image[tissue_mask]

        if len(tissue_pixels) == 0:
            # Ultimate fallback: use whole image
            tissue_pixels = signal_image.ravel()

        # Estimate background based on method
        if self.background_method == 'percentile':
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
        background: float,
        method: str = 'fold_change',
        threshold: float = 2.0,
    ):
        """
        Classify each nucleus as positive or negative for signal.

        Args:
            measurements: DataFrame from measure_nuclei_intensities()
            background: Background intensity from estimate_background()
            method: Classification method
                - 'fold_change': positive if mean >= threshold * background
                - 'absolute': positive if mean >= threshold (absolute value)
                - 'percentile': positive if mean >= threshold percentile of all nuclei
            threshold: Threshold value (interpretation depends on method)

        Returns:
            DataFrame with added columns:
            - background: background value used
            - fold_change: mean_intensity / background
            - is_positive: True/False classification
        """
        pd = _get_pandas()
        df = measurements.copy()

        df['background'] = background

        # Calculate fold change (even if not using it for classification)
        # Add small epsilon to avoid division by zero
        df['fold_change'] = df['mean_intensity'] / (background + 1e-10)

        # Classify based on method
        if method == 'fold_change':
            df['is_positive'] = df['fold_change'] >= threshold

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

        return {
            'total_cells': int(total),
            'positive_cells': int(positive),
            'negative_cells': int(negative),
            'positive_fraction': float(positive / total) if total > 0 else 0.0,
            'mean_fold_change': float(df['fold_change'].mean()) if len(df) > 0 else 0.0,
            'median_fold_change': float(df['fold_change'].median()) if len(df) > 0 else 0.0,
            'background_used': float(df['background'].iloc[0]) if len(df) > 0 else 0.0,
        }


def analyze_colocalization(
    signal_image: np.ndarray,
    nuclei_labels: np.ndarray,
    background_method: str = 'percentile',
    background_percentile: float = 10.0,
    threshold_method: str = 'fold_change',
    threshold_value: float = 2.0,
) -> Tuple[Any, Dict[str, Any]]:  # Returns (DataFrame, summary_dict)
    """
    Convenience function for complete colocalization analysis.

    Args:
        signal_image: Signal channel to measure
        nuclei_labels: Label image from detection
        background_method: Method for background estimation
        background_percentile: Percentile for background (if using percentile method)
        threshold_method: Method for positive/negative classification
        threshold_value: Threshold for classification

    Returns:
        Tuple of (measurements_df, summary_dict)
    """
    analyzer = ColocalizationAnalyzer(
        background_method=background_method,
        background_percentile=background_percentile,
    )

    background = analyzer.estimate_background(signal_image, nuclei_labels)
    measurements = analyzer.measure_nuclei_intensities(signal_image, nuclei_labels)
    classified = analyzer.classify_positive_negative(
        measurements, background,
        method=threshold_method,
        threshold=threshold_value
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
