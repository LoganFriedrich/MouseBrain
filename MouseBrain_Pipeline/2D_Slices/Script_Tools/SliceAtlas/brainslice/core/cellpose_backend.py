"""
cellpose_backend.py - Cellpose detection backend for BrainSlice 2D pipeline.

Provides a unified interface for Cellpose-based cell/nuclei detection,
matching the API pattern of the StarDist-based NucleiDetector.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

# Lazy import
_cellpose_available = None
_CellposeModel = None


def _check_cellpose():
    """Check if cellpose is available and import if needed."""
    global _cellpose_available, _CellposeModel

    if _cellpose_available is None:
        try:
            from cellpose.models import Cellpose
            _CellposeModel = Cellpose
            _cellpose_available = True
        except ImportError:
            _cellpose_available = False

    return _cellpose_available


CELLPOSE_MODELS = ['nuclei', 'cyto', 'cyto2', 'cyto3']


class CellposeDetector:
    """
    Cellpose-based cell/nuclei detection with StarDist-compatible output.

    Args:
        model_name: Model type ('nuclei', 'cyto', 'cyto2', 'cyto3')
        gpu: Whether to use GPU acceleration
    """

    def __init__(self, model_name: str = 'nuclei', gpu: bool = True):
        if not _check_cellpose():
            raise ImportError(
                "cellpose is not installed. Install with: pip install cellpose"
            )

        if model_name not in CELLPOSE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {CELLPOSE_MODELS}"
            )

        self.model_name = model_name
        self.gpu = gpu
        self.model = _CellposeModel(gpu=gpu, model_type=model_name)

    def detect(
        self,
        image: np.ndarray,
        diameter: float = 30.0,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Cellpose detection on a 2D image.

        Args:
            image: 2D grayscale image
            diameter: Expected cell diameter in pixels. 0 = auto-estimate.
            flow_threshold: Flow error threshold (higher = fewer cells)
            cellprob_threshold: Cell probability threshold (higher = fewer cells)

        Returns:
            Tuple of (labels, details) matching StarDist output format:
            - labels: 2D integer label image (0 = background)
            - details: dict with 'prob', 'coord', 'flows', 'diams'
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")

        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            **kwargs,
        )

        labels = masks.astype(np.int32)
        n_cells = labels.max()

        # Extract centroids
        if n_cells > 0:
            from scipy.ndimage import center_of_mass
            indices = np.arange(1, n_cells + 1)
            centroids = np.array(center_of_mass(labels > 0, labels, indices))
        else:
            centroids = np.empty((0, 2))

        details = {
            'prob': np.ones(n_cells) if n_cells > 0 else np.empty(0),
            'coord': centroids,
            'flows': flows,
            'diams': diams,
        }

        return labels, details
