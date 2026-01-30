"""
detection.py - Nuclei detection using StarDist

Provides StarDist-based nuclei detection for 2D confocal slice images.

Usage:
    from braintools.pipeline_2d.sliceatlas.core.detection import NucleiDetector

    detector = NucleiDetector()
    labels, details = detector.detect(red_channel_image)
    filtered_labels = detector.filter_by_size(labels, min_area=50, max_area=5000)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
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

    Returns:
        Tuple of (labels, count)
        - labels: Filtered label image
        - count: Number of nuclei detected
    """
    detector = NucleiDetector(model_name=model_name)
    labels, _ = detector.detect(image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    labels = detector.filter_by_size(labels, min_area=min_area, max_area=max_area)
    count = len(np.unique(labels)) - 1  # Exclude background

    return labels, count


def list_available_models() -> Dict[str, str]:
    """List available pretrained StarDist models."""
    return PRETRAINED_MODELS.copy()
