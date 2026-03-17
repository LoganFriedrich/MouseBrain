"""
particle_analysis.py - ImageJ-style particle analysis for multi-channel images.

Workflow:
1. Binarize a "detection channel" by hard threshold
2. Find connected-component particles, filter by size/circularity
3. Measure intensity of each particle in a "measurement channel"
4. Report per-particle properties and intensities

This is distinct from colocalization (which detects nuclei then measures signal).
Particle analysis is channel-agnostic: binarize ANY channel, measure ANY other.

Usage:
    from mousebrain.plugin_2d.sliceatlas.core.particle_analysis import ParticleAnalyzer

    analyzer = ParticleAnalyzer()
    mask = analyzer.binarize(green_channel, threshold=500)
    labels, props = analyzer.find_particles(mask, min_area=10, max_area=5000)
    results = analyzer.measure_intensity(labels, red_channel)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np


# Lazy imports
_pd = None
_ndimage = None


def _get_pandas():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_ndimage():
    global _ndimage
    if _ndimage is None:
        from scipy import ndimage
        _ndimage = ndimage
    return _ndimage


class ParticleAnalyzer:
    """
    ImageJ-style particle analysis for multi-channel fluorescence images.

    Binarize a detection channel, find particles (connected components),
    filter by morphology, then measure intensity in a measurement channel.
    """

    def binarize(
        self,
        image: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Apply hard threshold to create binary mask.

        Args:
            image: 2D image (any dtype).
            threshold: Intensity cutoff. Pixels > threshold become True.

        Returns:
            Binary mask (bool array, same shape as image).
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")

        return image.astype(np.float64) > float(threshold)

    def find_particles(
        self,
        binary_mask: np.ndarray,
        min_area: int = 10,
        max_area: int = 50000,
        min_circularity: float = 0.0,
        max_circularity: float = 1.0,
    ) -> Tuple[np.ndarray, 'pandas.DataFrame']:
        """
        Find connected components in a binary mask and filter by properties.

        Args:
            binary_mask: 2D boolean mask from binarize().
            min_area: Minimum particle area in pixels.
            max_area: Maximum particle area in pixels.
            min_circularity: Minimum circularity (4*pi*area/perimeter^2).
                0 = no filtering. Perfect circle = 1.0.
            max_circularity: Maximum circularity. 1.0 = keep all.

        Returns:
            Tuple of:
            - labels: 2D label array (0 = background, 1..N = particles)
            - props_df: DataFrame with columns: label, area, centroid_y,
              centroid_x, circularity, solidity, perimeter, bbox_min_row,
              bbox_min_col, bbox_max_row, bbox_max_col
        """
        from skimage.measure import regionprops, label as sk_label

        pd = _get_pandas()

        if binary_mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {binary_mask.ndim}D")

        # Label connected components
        labels = sk_label(binary_mask.astype(bool))
        raw_count = labels.max()

        if raw_count == 0:
            empty_df = pd.DataFrame(columns=[
                'label', 'area', 'centroid_y', 'centroid_x',
                'circularity', 'solidity', 'perimeter',
                'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col',
            ])
            return labels, empty_df

        # Measure properties
        props = regionprops(labels)
        records = []
        for p in props:
            perim = p.perimeter if p.perimeter > 0 else 1e-10
            circ = (4.0 * np.pi * p.area) / (perim ** 2)
            circ = min(circ, 1.0)  # clamp rounding errors

            records.append({
                'label': p.label,
                'area': p.area,
                'centroid_y': p.centroid[0],
                'centroid_x': p.centroid[1],
                'circularity': circ,
                'solidity': p.solidity,
                'perimeter': p.perimeter,
                'bbox_min_row': p.bbox[0],
                'bbox_min_col': p.bbox[1],
                'bbox_max_row': p.bbox[2],
                'bbox_max_col': p.bbox[3],
            })

        props_df = pd.DataFrame(records)

        # Filter by area
        props_df = props_df[
            (props_df['area'] >= min_area) &
            (props_df['area'] <= max_area)
        ]

        # Filter by circularity
        if min_circularity > 0 or max_circularity < 1.0:
            props_df = props_df[
                (props_df['circularity'] >= min_circularity) &
                (props_df['circularity'] <= max_circularity)
            ]

        # Rebuild labels to keep only surviving particles, relabel 1..N
        keep_labels = set(props_df['label'].values)
        filtered = np.zeros_like(labels)
        new_id = 0
        label_map = {}
        for old_label in sorted(keep_labels):
            new_id += 1
            filtered[labels == old_label] = new_id
            label_map[old_label] = new_id

        props_df = props_df.copy()
        props_df['label'] = props_df['label'].map(label_map)
        props_df = props_df.sort_values('label').reset_index(drop=True)

        return filtered, props_df

    def estimate_background(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        method: str = 'percentile',
        percentile: float = 50.0,
    ) -> float:
        """
        Estimate background intensity in the measurement channel.

        Background is estimated from pixels NOT inside any particle.

        Args:
            image: 2D measurement channel image.
            labels: 2D label array from find_particles().
            method: 'percentile', 'mode', 'mean', or 'median'.
            percentile: Percentile value (for 'percentile' method).

        Returns:
            Background intensity value.
        """
        # Get non-particle pixels
        bg_mask = labels == 0
        bg_pixels = image[bg_mask].astype(np.float64)

        if len(bg_pixels) == 0:
            return 0.0

        if method == 'percentile':
            return float(np.percentile(bg_pixels, percentile))
        elif method == 'median':
            return float(np.median(bg_pixels))
        elif method == 'mean':
            # Exclude top 5% outliers
            cutoff = np.percentile(bg_pixels, 95)
            trimmed = bg_pixels[bg_pixels <= cutoff]
            if len(trimmed) == 0:
                return float(np.mean(bg_pixels))
            return float(np.mean(trimmed))
        elif method == 'mode':
            hist, bin_edges = np.histogram(bg_pixels, bins=256)
            mode_idx = np.argmax(hist)
            return float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)
        else:
            raise ValueError(f"Unknown background method: {method}")

    def measure_intensity(
        self,
        labels: np.ndarray,
        measurement_image: np.ndarray,
        background_method: str = 'percentile',
        background_percentile: float = 50.0,
    ) -> 'pandas.DataFrame':
        """
        Measure intensity of each particle in a measurement channel.

        For each labeled particle, computes mean/median/max/integrated
        intensity in the measurement image, plus background-subtracted values.

        Args:
            labels: 2D label array from find_particles().
            measurement_image: 2D image of the measurement channel.
            background_method: Method for background estimation.
            background_percentile: Percentile for background (if method='percentile').

        Returns:
            DataFrame with columns: label, mean_intensity, median_intensity,
            max_intensity, integrated_intensity, background,
            mean_above_background, snr
        """
        from skimage.measure import regionprops_table

        pd = _get_pandas()

        if labels.max() == 0:
            return pd.DataFrame(columns=[
                'label', 'mean_intensity', 'median_intensity',
                'max_intensity', 'integrated_intensity',
                'background', 'mean_above_background', 'snr',
            ])

        # Estimate background from non-particle pixels
        background = self.estimate_background(
            measurement_image, labels,
            method=background_method,
            percentile=background_percentile,
        )

        # Measure per-particle intensities using regionprops
        table = regionprops_table(
            labels,
            intensity_image=measurement_image.astype(np.float64),
            properties=['label', 'mean_intensity', 'max_intensity', 'area'],
        )
        df = pd.DataFrame(table)

        # Compute additional metrics manually for median and integrated
        median_vals = []
        integrated_vals = []
        for lbl in df['label'].values:
            px = measurement_image[labels == lbl].astype(np.float64)
            median_vals.append(float(np.median(px)))
            integrated_vals.append(float(np.sum(px)))

        df['median_intensity'] = median_vals
        df['integrated_intensity'] = integrated_vals
        df['background'] = background
        df['mean_above_background'] = df['mean_intensity'] - background
        # SNR: signal above background / background (avoid div by zero)
        df['snr'] = df['mean_above_background'] / max(background, 1e-10)

        # Reorder columns
        df = df[['label', 'area', 'mean_intensity', 'median_intensity',
                 'max_intensity', 'integrated_intensity',
                 'background', 'mean_above_background', 'snr']]

        return df

    def run_full_analysis(
        self,
        detection_image: np.ndarray,
        measurement_image: np.ndarray,
        threshold: float,
        min_area: int = 10,
        max_area: int = 50000,
        min_circularity: float = 0.0,
        max_circularity: float = 1.0,
        background_method: str = 'percentile',
        background_percentile: float = 50.0,
    ) -> Tuple[np.ndarray, 'pandas.DataFrame', 'pandas.DataFrame', Dict[str, Any]]:
        """
        Run the complete particle analysis pipeline.

        Convenience method that chains binarize -> find_particles ->
        measure_intensity and returns all results.

        Args:
            detection_image: 2D image to binarize (detection channel).
            measurement_image: 2D image to measure intensities in.
            threshold: Binarization threshold.
            min_area: Minimum particle area.
            max_area: Maximum particle area.
            min_circularity: Minimum circularity filter.
            max_circularity: Maximum circularity filter.
            background_method: Background estimation method.
            background_percentile: Background percentile.

        Returns:
            Tuple of:
            - labels: 2D label array
            - particle_props: DataFrame of particle properties
            - measurements: DataFrame of intensity measurements
            - summary: Dict with summary statistics
        """
        pd = _get_pandas()

        # Step 1: Binarize
        mask = self.binarize(detection_image, threshold)

        # Step 2: Find and filter particles
        labels, particle_props = self.find_particles(
            mask,
            min_area=min_area,
            max_area=max_area,
            min_circularity=min_circularity,
            max_circularity=max_circularity,
        )

        n_particles = int(labels.max())

        # Step 3: Measure intensities
        if n_particles > 0:
            measurements = self.measure_intensity(
                labels, measurement_image,
                background_method=background_method,
                background_percentile=background_percentile,
            )

            # Merge particle props with measurements
            results = pd.merge(particle_props, measurements,
                               on='label', suffixes=('', '_meas'))
            # Drop duplicate 'area' column from merge if present
            if 'area_meas' in results.columns:
                results = results.drop(columns=['area_meas'])
        else:
            results = particle_props
            measurements = pd.DataFrame()

        # Summary statistics
        summary = {
            'n_particles': n_particles,
            'threshold': float(threshold),
            'mask_pixels': int(mask.sum()),
            'mask_fraction': float(mask.sum() / mask.size) if mask.size > 0 else 0.0,
        }

        if n_particles > 0 and len(measurements) > 0:
            bg = float(measurements['background'].iloc[0])
            summary['background'] = bg
            summary['mean_particle_intensity'] = float(measurements['mean_intensity'].mean())
            summary['mean_above_background'] = float(measurements['mean_above_background'].mean())
            summary['mean_snr'] = float(measurements['snr'].mean())
            summary['mean_area'] = float(particle_props['area'].mean())
            summary['median_area'] = float(particle_props['area'].median())

        return labels, results, measurements, summary
