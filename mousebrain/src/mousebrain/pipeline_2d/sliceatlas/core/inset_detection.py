"""
inset_detection.py - Detection on high-resolution insets with coordinate merging

Runs nuclei detection on insets at their full resolution (for accuracy)
and transforms coordinates back to base image space for unified analysis.

Usage:
    from mousebrain.pipeline_2d.sliceatlas.core.inset_detection import InsetDetectionPipeline

    pipeline = InsetDetectionPipeline(inset_manager, detector)
    all_labels, all_measurements = pipeline.run_detection(
        use_composite=True,
        prefer_insets=True
    )
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class InsetDetectionPipeline:
    """
    Pipeline for running detection across base image and insets.

    Strategy:
    1. Run detection on each inset at full resolution
    2. Transform inset detections to base coordinates
    3. Optionally run detection on base image (excluding inset regions)
    4. Merge all detections into unified coordinate space
    """

    def __init__(
        self,
        inset_manager: 'InsetManager',
        detector: Optional['NucleiDetector'] = None,
    ):
        """
        Initialize pipeline.

        Args:
            inset_manager: InsetManager with base image and insets loaded
            detector: NucleiDetector instance (created on-demand if not provided)
        """
        self.inset_manager = inset_manager
        self._detector = detector

    @property
    def detector(self):
        """Get or create detector."""
        if self._detector is None:
            from .detection import NucleiDetector
            self._detector = NucleiDetector()
        return self._detector

    def detect_in_inset(
        self,
        inset_name: str,
        channel: int = 0,
        **detect_params,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run detection on a single inset at full resolution.

        Args:
            inset_name: Name of the inset
            channel: Channel to use for detection
            **detect_params: Parameters passed to detector.detect()

        Returns:
            Tuple of (labels, properties)
            - labels: Label image in inset pixel space
            - properties: Dict of cell properties with 'centroid_y', 'centroid_x' in BASE coords
        """
        inset = self.inset_manager.insets[inset_name]

        # Get detection channel
        if inset.image.ndim == 3:
            image = inset.image[channel]
        else:
            image = inset.image

        # Run detection at full resolution
        labels, _ = self.detector.detect(image, **detect_params)

        # Get properties with centroids
        props = self.detector.get_properties(labels)

        # Transform centroids to base coordinates
        if len(props['label']) > 0:
            inset_coords = np.column_stack([props['centroid_y'], props['centroid_x']])
            base_coords = self.inset_manager.transform_to_base(inset_coords, inset_name)
            props['centroid_y_base'] = base_coords[:, 0]
            props['centroid_x_base'] = base_coords[:, 1]
        else:
            props['centroid_y_base'] = np.array([])
            props['centroid_x_base'] = np.array([])

        # Add inset info
        props['inset_name'] = np.array([inset_name] * len(props['label']))
        props['from_inset'] = np.array([True] * len(props['label']))

        return labels, props

    def detect_in_base_excluding_insets(
        self,
        channel: int = 0,
        **detect_params,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run detection on base image, excluding regions covered by insets.

        Args:
            channel: Channel to use for detection
            **detect_params: Parameters passed to detector.detect()

        Returns:
            Tuple of (labels, properties)
        """
        # Get base image channel
        if self.inset_manager.base_image.ndim == 3:
            image = self.inset_manager.base_image[channel]
        else:
            image = self.inset_manager.base_image

        # Create mask of inset regions
        inset_mask = self.inset_manager.create_inset_mask()

        # Run detection on full base image
        labels, _ = self.detector.detect(image, **detect_params)

        # Remove detections that fall within inset regions
        # For each detected cell, check if centroid is in inset region
        from skimage.measure import regionprops

        props_list = regionprops(labels)
        cells_to_remove = []

        for prop in props_list:
            cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
            if 0 <= cy < inset_mask.shape[0] and 0 <= cx < inset_mask.shape[1]:
                if inset_mask[cy, cx]:
                    cells_to_remove.append(prop.label)

        # Create filtered labels
        if cells_to_remove:
            for label_id in cells_to_remove:
                labels[labels == label_id] = 0

            # Relabel to consecutive integers
            from skimage.segmentation import relabel_sequential
            labels, _, _ = relabel_sequential(labels)

        # Get properties
        props = self.detector.get_properties(labels)

        # Base coords are same as original coords
        props['centroid_y_base'] = props['centroid_y'].copy()
        props['centroid_x_base'] = props['centroid_x'].copy()
        props['inset_name'] = np.array([''] * len(props['label']))
        props['from_inset'] = np.array([False] * len(props['label']))

        return labels, props

    def run_full_detection(
        self,
        channel: int = 0,
        detect_in_base: bool = True,
        **detect_params,
    ) -> Dict[str, Any]:
        """
        Run detection on all insets and optionally base image.

        Args:
            channel: Channel to use for detection
            detect_in_base: Whether to also detect in base image (excluding insets)
            **detect_params: Parameters passed to detector.detect()

        Returns:
            Dict with:
            - 'merged_labels': Label image in base coordinate space
            - 'merged_properties': Combined properties DataFrame
            - 'inset_results': Dict of per-inset results
            - 'base_results': Results from base image (if detect_in_base)
            - 'total_cells': Total cell count
        """
        import pandas as pd

        results = {
            'inset_results': {},
            'base_results': None,
            'merged_properties': None,
            'merged_labels': None,
            'total_cells': 0,
        }

        all_props = []
        current_label = 1

        # Detect in each inset
        for inset_name in self.inset_manager.insets:
            inset = self.inset_manager.insets[inset_name]
            if not inset.aligned:
                continue

            labels, props = self.detect_in_inset(inset_name, channel, **detect_params)

            # Store with adjusted labels for merging
            n_cells = len(props['label'])
            if n_cells > 0:
                props['merged_label'] = np.arange(current_label, current_label + n_cells)
                current_label += n_cells

            results['inset_results'][inset_name] = {
                'labels': labels,
                'properties': props,
                'n_cells': n_cells,
            }
            all_props.append(props)

        # Detect in base image (excluding inset regions)
        if detect_in_base:
            labels, props = self.detect_in_base_excluding_insets(channel, **detect_params)
            n_cells = len(props['label'])
            if n_cells > 0:
                props['merged_label'] = np.arange(current_label, current_label + n_cells)
                current_label += n_cells

            results['base_results'] = {
                'labels': labels,
                'properties': props,
                'n_cells': n_cells,
            }
            all_props.append(props)

        # Merge all properties into single DataFrame
        if all_props:
            merged = {}
            for key in all_props[0].keys():
                merged[key] = np.concatenate([p[key] for p in all_props if len(p[key]) > 0])

            results['merged_properties'] = pd.DataFrame(merged)
            results['total_cells'] = len(results['merged_properties'])

        # Create merged label image in base coordinates
        results['merged_labels'] = self._create_merged_labels(results)

        return results

    def _create_merged_labels(self, results: Dict) -> np.ndarray:
        """Create unified label image in base coordinate space."""
        h, w = self.inset_manager.base_shape
        merged_labels = np.zeros((h, w), dtype=np.int32)

        # First, add base detections
        if results['base_results'] is not None:
            base_labels = results['base_results']['labels']
            base_props = results['base_results']['properties']

            if len(base_props['label']) > 0:
                # Map old labels to merged labels
                label_map = dict(zip(base_props['label'], base_props['merged_label']))
                for old_label, new_label in label_map.items():
                    merged_labels[base_labels == old_label] = new_label

        # Then add inset detections (at base resolution)
        # This is approximate - we draw circles at centroid locations
        for inset_name, inset_results in results['inset_results'].items():
            props = inset_results['properties']
            if len(props['label']) == 0:
                continue

            inset = self.inset_manager.insets[inset_name]

            # For each cell, draw in base coordinates
            for i in range(len(props['label'])):
                cy = int(props['centroid_y_base'][i])
                cx = int(props['centroid_x_base'][i])
                label = int(props['merged_label'][i])

                # Estimate cell radius in base pixels
                area = props['area'][i]
                radius_inset = np.sqrt(area / np.pi)
                radius_base = int(radius_inset / inset.scale_factor)
                radius_base = max(1, radius_base)

                # Draw circle
                y, x = np.ogrid[-radius_base:radius_base+1, -radius_base:radius_base+1]
                mask = x**2 + y**2 <= radius_base**2

                y1 = max(0, cy - radius_base)
                y2 = min(h, cy + radius_base + 1)
                x1 = max(0, cx - radius_base)
                x2 = min(w, cx + radius_base + 1)

                # Adjust mask for edge cases
                my1 = radius_base - (cy - y1)
                my2 = my1 + (y2 - y1)
                mx1 = radius_base - (cx - x1)
                mx2 = mx1 + (x2 - x1)

                if my2 > my1 and mx2 > mx1:
                    merged_labels[y1:y2, x1:x2][mask[my1:my2, mx1:mx2]] = label

        return merged_labels

    def get_cells_in_region(
        self,
        results: Dict,
        bounds: Tuple[int, int, int, int],
    ) -> 'pd.DataFrame':
        """
        Get cells within a bounding box in base coordinates.

        Args:
            results: Results from run_full_detection()
            bounds: (x1, y1, x2, y2) bounding box

        Returns:
            DataFrame of cells in the region
        """
        df = results['merged_properties']
        if df is None or len(df) == 0:
            import pandas as pd
            return pd.DataFrame()

        x1, y1, x2, y2 = bounds
        mask = (
            (df['centroid_x_base'] >= x1) &
            (df['centroid_x_base'] < x2) &
            (df['centroid_y_base'] >= y1) &
            (df['centroid_y_base'] < y2)
        )
        return df[mask]
