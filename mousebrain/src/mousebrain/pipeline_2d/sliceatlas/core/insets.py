"""
insets.py - High-resolution inset image management for BrainSlice

Handles overlaying high-resolution region-of-interest images onto
lower-resolution whole-slice images. Supports:
- Automatic matching via naming conventions
- Scale factor calculation from pixel sizes
- Alignment via template matching or manual positioning
- Composite image creation
- Coordinate transforms for detection results

Naming Convention:
    Base image:   {sample}_overview.nd2  or  {sample}.nd2
    Inset image:  {sample}_inset_{region}.nd2
                  {sample}_inset_{region}_x{X}_y{Y}.nd2  (with position hint)

Examples:
    ENCR_001_slice12_overview.nd2
    ENCR_001_slice12_inset_VTA.nd2
    ENCR_001_slice12_inset_SNc_x1500_y2000.nd2

Usage:
    from mousebrain.pipeline_2d.sliceatlas.core.insets import InsetManager

    manager = InsetManager(base_image, base_metadata)
    manager.add_inset("path/to/inset.nd2")
    composite = manager.create_composite()
    base_coords = manager.transform_to_base(inset_coords, inset_name)
"""

import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# Lazy imports
_cv2 = None


def _get_cv2():
    """Lazy import OpenCV for template matching."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            _cv2 = False  # Mark as unavailable
    return _cv2


class InsetInfo:
    """Information about a single inset image."""

    def __init__(
        self,
        name: str,
        image: np.ndarray,
        metadata: Dict[str, Any],
        region: str = "",
        position_hint: Optional[Tuple[int, int]] = None,
    ):
        self.name = name
        self.image = image  # Shape: (C, Y, X) or (Y, X)
        self.metadata = metadata
        self.region = region
        self.position_hint = position_hint  # (x, y) in base image coords

        # Computed during alignment
        self.scale_factor: float = 1.0  # base_pixel_size / inset_pixel_size
        self.position: Optional[Tuple[int, int]] = None  # (x, y) top-left in base
        self.aligned: bool = False
        self.alignment_score: float = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape

    @property
    def pixel_size(self) -> float:
        """Get pixel size in microns (assumes square pixels)."""
        voxel = self.metadata.get('voxel_size_um', {})
        return voxel.get('x', 1.0)

    def get_base_dimensions(self) -> Tuple[int, int]:
        """Get dimensions when scaled to base image resolution."""
        if self.image.ndim == 3:
            h, w = self.image.shape[1], self.image.shape[2]
        else:
            h, w = self.image.shape
        return (int(h / self.scale_factor), int(w / self.scale_factor))


def parse_inset_name(filename: str) -> Dict[str, Any]:
    """
    Parse inset filename to extract information.

    Patterns recognized:
    - {sample}_inset_{region}.ext
    - {sample}_inset_{region}_x{X}_y{Y}.ext
    - {sample}_roi_{region}.ext
    - {sample}_highres_{region}.ext

    Returns:
        Dict with:
        - base_sample: base sample name (for matching)
        - region: region name
        - position_hint: (x, y) if encoded in name, else None
        - is_inset: True if this looks like an inset file
    """
    stem = Path(filename).stem
    result = {
        'base_sample': '',
        'region': '',
        'position_hint': None,
        'is_inset': False,
        'raw_name': stem,
    }

    # Pattern: {sample}_inset_{region}_x{X}_y{Y}
    match = re.match(r'(.+?)_(?:inset|roi|highres)_([^_]+)_x(\d+)_y(\d+)', stem, re.IGNORECASE)
    if match:
        result['base_sample'] = match.group(1)
        result['region'] = match.group(2)
        result['position_hint'] = (int(match.group(3)), int(match.group(4)))
        result['is_inset'] = True
        return result

    # Pattern: {sample}_inset_{region}
    match = re.match(r'(.+?)_(?:inset|roi|highres)_(.+)', stem, re.IGNORECASE)
    if match:
        result['base_sample'] = match.group(1)
        result['region'] = match.group(2)
        result['is_inset'] = True
        return result

    # Not an inset, might be base/overview
    # Check for _overview suffix
    if stem.endswith('_overview'):
        result['base_sample'] = stem[:-9]  # Remove _overview
    else:
        result['base_sample'] = stem

    return result


def find_matching_insets(
    base_path: Path,
    search_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Find inset files that match a base image.

    Args:
        base_path: Path to base/overview image
        search_dir: Directory to search (default: same as base_path)

    Returns:
        List of paths to matching inset files
    """
    if search_dir is None:
        search_dir = base_path.parent

    base_info = parse_inset_name(base_path.name)
    base_sample = base_info['base_sample']

    matching = []
    for ext in ['.nd2', '.tif', '.tiff']:
        for candidate in search_dir.glob(f'*{ext}'):
            info = parse_inset_name(candidate.name)
            if info['is_inset'] and info['base_sample'] == base_sample:
                matching.append(candidate)

    return sorted(matching)


class InsetManager:
    """
    Manages high-resolution inset images overlaid on a base image.

    Handles:
    - Loading and tracking multiple insets
    - Scale factor calculation from pixel sizes
    - Alignment via template matching or manual positioning
    - Composite image creation
    - Coordinate transforms between inset and base space
    """

    def __init__(
        self,
        base_image: np.ndarray,
        base_metadata: Dict[str, Any],
    ):
        """
        Initialize with base (overview) image.

        Args:
            base_image: Base image array, shape (C, Y, X) or (Y, X)
            base_metadata: Metadata from load_image()
        """
        self.base_image = base_image
        self.base_metadata = base_metadata
        self.base_pixel_size = base_metadata.get('voxel_size_um', {}).get('x', 1.0)

        # Insets by name
        self.insets: Dict[str, InsetInfo] = {}

        # Track which channel to use for alignment
        self.alignment_channel = 0

    @property
    def base_shape(self) -> Tuple[int, int]:
        """Get (height, width) of base image."""
        if self.base_image.ndim == 3:
            return (self.base_image.shape[1], self.base_image.shape[2])
        return (self.base_image.shape[0], self.base_image.shape[1])

    def add_inset(
        self,
        file_path: Union[str, Path],
        region: Optional[str] = None,
        position: Optional[Tuple[int, int]] = None,
        auto_align: bool = True,
    ) -> str:
        """
        Add an inset image.

        Args:
            file_path: Path to inset image file
            region: Region name (auto-detected from filename if not provided)
            position: Manual position (x, y) in base coords (skips auto-align)
            auto_align: Whether to attempt automatic alignment

        Returns:
            Inset name (for reference)
        """
        from .io import load_image

        file_path = Path(file_path)
        data, metadata = load_image(file_path)

        # Parse filename for info
        name_info = parse_inset_name(file_path.name)
        if region is None:
            region = name_info.get('region', file_path.stem)

        # Create InsetInfo
        inset = InsetInfo(
            name=file_path.stem,
            image=data,
            metadata=metadata,
            region=region,
            position_hint=name_info.get('position_hint') or position,
        )

        # Calculate scale factor
        inset_pixel_size = metadata.get('voxel_size_um', {}).get('x', 1.0)
        if inset_pixel_size > 0 and self.base_pixel_size > 0:
            inset.scale_factor = self.base_pixel_size / inset_pixel_size
        else:
            inset.scale_factor = 1.0

        # Align if requested
        if position is not None:
            inset.position = position
            inset.aligned = True
        elif auto_align:
            self._align_inset(inset)

        self.insets[inset.name] = inset
        return inset.name

    def _align_inset(self, inset: InsetInfo) -> bool:
        """
        Align inset to base image using template matching.

        Returns True if alignment succeeded.
        """
        cv2 = _get_cv2()
        if cv2 is False:
            # OpenCV not available, try scipy-based approach
            return self._align_inset_scipy(inset)

        # Get single channel for matching
        if self.base_image.ndim == 3:
            base_ch = self.base_image[self.alignment_channel]
        else:
            base_ch = self.base_image

        if inset.image.ndim == 3:
            inset_ch = inset.image[self.alignment_channel]
        else:
            inset_ch = inset.image

        # Downscale inset to base resolution for matching
        scaled_h, scaled_w = inset.get_base_dimensions()

        # Ensure minimum size
        if scaled_h < 10 or scaled_w < 10:
            inset.aligned = False
            return False

        inset_scaled = cv2.resize(
            inset_ch.astype(np.float32),
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA
        )

        # Normalize for template matching
        base_norm = cv2.normalize(base_ch.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        inset_norm = cv2.normalize(inset_scaled, None, 0, 1, cv2.NORM_MINMAX)

        # Use position hint as search region if available
        if inset.position_hint is not None:
            hx, hy = inset.position_hint
            # Search in region around hint
            margin = max(scaled_h, scaled_w) * 2
            y1 = max(0, hy - margin)
            y2 = min(base_norm.shape[0], hy + scaled_h + margin)
            x1 = max(0, hx - margin)
            x2 = min(base_norm.shape[1], hx + scaled_w + margin)
            search_region = base_norm[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            search_region = base_norm
            offset = (0, 0)

        # Check if template fits in search region
        if (inset_norm.shape[0] >= search_region.shape[0] or
            inset_norm.shape[1] >= search_region.shape[1]):
            # Template larger than search region
            inset.aligned = False
            return False

        # Template matching
        result = cv2.matchTemplate(search_region, inset_norm, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Set position (top-left corner in base coords)
        inset.position = (max_loc[0] + offset[0], max_loc[1] + offset[1])
        inset.alignment_score = max_val
        inset.aligned = max_val > 0.3  # Threshold for "good" alignment

        return inset.aligned

    def _align_inset_scipy(self, inset: InsetInfo) -> bool:
        """Fallback alignment using scipy (slower but no OpenCV needed)."""
        from scipy import ndimage
        from scipy.signal import correlate2d

        # Get channels
        if self.base_image.ndim == 3:
            base_ch = self.base_image[self.alignment_channel]
        else:
            base_ch = self.base_image

        if inset.image.ndim == 3:
            inset_ch = inset.image[self.alignment_channel]
        else:
            inset_ch = inset.image

        # Downscale inset
        scaled_h, scaled_w = inset.get_base_dimensions()
        zoom_factor = (scaled_h / inset_ch.shape[0], scaled_w / inset_ch.shape[1])
        inset_scaled = ndimage.zoom(inset_ch.astype(np.float32), zoom_factor, order=1)

        # Normalize
        base_norm = (base_ch - base_ch.mean()) / (base_ch.std() + 1e-8)
        inset_norm = (inset_scaled - inset_scaled.mean()) / (inset_scaled.std() + 1e-8)

        # Cross-correlation (this is slow for large images)
        # Use downsampled versions for speed
        downsample = 4
        base_small = base_norm[::downsample, ::downsample]
        inset_small = inset_norm[::downsample, ::downsample]

        corr = correlate2d(base_small, inset_small, mode='valid')
        max_idx = np.unravel_index(np.argmax(corr), corr.shape)

        # Scale back to full resolution
        inset.position = (max_idx[1] * downsample, max_idx[0] * downsample)
        inset.alignment_score = corr[max_idx] / (inset_small.size)
        inset.aligned = True

        return True

    def set_inset_position(
        self,
        inset_name: str,
        position: Tuple[int, int],
    ):
        """
        Manually set inset position.

        Args:
            inset_name: Name of the inset
            position: (x, y) top-left corner in base image coordinates
        """
        if inset_name not in self.insets:
            raise ValueError(f"Unknown inset: {inset_name}")

        self.insets[inset_name].position = position
        self.insets[inset_name].aligned = True

    def get_inset_bounds(
        self,
        inset_name: str,
    ) -> Tuple[int, int, int, int]:
        """
        Get bounding box of inset in base image coordinates.

        Returns:
            (x1, y1, x2, y2) - top-left and bottom-right corners
        """
        inset = self.insets[inset_name]
        if not inset.aligned or inset.position is None:
            raise ValueError(f"Inset {inset_name} is not aligned")

        x, y = inset.position
        h, w = inset.get_base_dimensions()
        return (x, y, x + w, y + h)

    def create_composite(
        self,
        channel: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create composite image with insets overlaid on base.

        Insets replace (not blend with) the underlying base image.

        Args:
            channel: Specific channel to composite (None = all channels)

        Returns:
            Composite image array
        """
        from scipy.ndimage import zoom

        # Start with copy of base
        if channel is not None:
            if self.base_image.ndim == 3:
                composite = self.base_image[channel].copy()
            else:
                composite = self.base_image.copy()
        else:
            composite = self.base_image.copy()

        # Overlay each aligned inset
        for inset in self.insets.values():
            if not inset.aligned or inset.position is None:
                continue

            x, y = inset.position
            base_h, base_w = inset.get_base_dimensions()

            # Get inset data for this channel
            if channel is not None and inset.image.ndim == 3:
                inset_data = inset.image[channel]
            elif channel is not None:
                inset_data = inset.image
            else:
                inset_data = inset.image

            # Downscale inset to base resolution
            if inset.scale_factor != 1.0:
                if inset_data.ndim == 3:
                    # Multi-channel
                    scaled_channels = []
                    for c in range(inset_data.shape[0]):
                        scaled = zoom(
                            inset_data[c],
                            1.0 / inset.scale_factor,
                            order=1
                        )
                        scaled_channels.append(scaled)
                    inset_scaled = np.stack(scaled_channels)
                else:
                    inset_scaled = zoom(
                        inset_data,
                        1.0 / inset.scale_factor,
                        order=1
                    )
            else:
                inset_scaled = inset_data

            # Clip to base image bounds
            y1, x1 = max(0, y), max(0, x)
            if composite.ndim == 3:
                y2 = min(composite.shape[1], y + inset_scaled.shape[-2])
                x2 = min(composite.shape[2], x + inset_scaled.shape[-1])
            else:
                y2 = min(composite.shape[0], y + inset_scaled.shape[-2])
                x2 = min(composite.shape[1], x + inset_scaled.shape[-1])

            # Calculate source region in scaled inset
            sy1 = y1 - y
            sx1 = x1 - x
            sy2 = sy1 + (y2 - y1)
            sx2 = sx1 + (x2 - x1)

            # Overlay
            if composite.ndim == 3 and inset_scaled.ndim == 3:
                composite[:, y1:y2, x1:x2] = inset_scaled[:, sy1:sy2, sx1:sx2]
            elif composite.ndim == 2 and inset_scaled.ndim == 2:
                composite[y1:y2, x1:x2] = inset_scaled[sy1:sy2, sx1:sx2]
            elif composite.ndim == 2 and inset_scaled.ndim == 3:
                # Take first channel of inset
                composite[y1:y2, x1:x2] = inset_scaled[0, sy1:sy2, sx1:sx2]

        return composite

    def create_inset_mask(self) -> np.ndarray:
        """
        Create a mask showing where insets are located.

        Returns:
            Boolean array (True = inset region)
        """
        h, w = self.base_shape
        mask = np.zeros((h, w), dtype=bool)

        for inset in self.insets.values():
            if not inset.aligned or inset.position is None:
                continue

            x, y = inset.position
            ih, iw = inset.get_base_dimensions()

            y1, x1 = max(0, y), max(0, x)
            y2 = min(h, y + ih)
            x2 = min(w, x + iw)

            mask[y1:y2, x1:x2] = True

        return mask

    def transform_to_base(
        self,
        coords: np.ndarray,
        inset_name: str,
    ) -> np.ndarray:
        """
        Transform coordinates from inset space to base image space.

        Args:
            coords: Array of (y, x) coordinates in inset pixel space
            inset_name: Name of the inset

        Returns:
            Array of (y, x) coordinates in base pixel space
        """
        inset = self.insets[inset_name]
        if not inset.aligned or inset.position is None:
            raise ValueError(f"Inset {inset_name} is not aligned")

        coords = np.atleast_2d(coords)

        # Scale from inset pixels to base pixels
        scaled_coords = coords / inset.scale_factor

        # Translate by inset position
        offset = np.array([inset.position[1], inset.position[0]])  # (y, x)
        base_coords = scaled_coords + offset

        return base_coords

    def transform_to_inset(
        self,
        coords: np.ndarray,
        inset_name: str,
    ) -> np.ndarray:
        """
        Transform coordinates from base image space to inset space.

        Args:
            coords: Array of (y, x) coordinates in base pixel space
            inset_name: Name of the inset

        Returns:
            Array of (y, x) coordinates in inset pixel space
        """
        inset = self.insets[inset_name]
        if not inset.aligned or inset.position is None:
            raise ValueError(f"Inset {inset_name} is not aligned")

        coords = np.atleast_2d(coords)

        # Translate by inset position
        offset = np.array([inset.position[1], inset.position[0]])  # (y, x)
        translated = coords - offset

        # Scale from base pixels to inset pixels
        inset_coords = translated * inset.scale_factor

        return inset_coords

    def get_inset_for_point(
        self,
        y: int,
        x: int,
    ) -> Optional[str]:
        """
        Get the inset (if any) that contains a given point in base coordinates.

        Args:
            y, x: Point in base image coordinates

        Returns:
            Inset name if point is in an inset, else None
        """
        for name, inset in self.insets.items():
            if not inset.aligned or inset.position is None:
                continue

            ix, iy = inset.position
            ih, iw = inset.get_base_dimensions()

            if iy <= y < iy + ih and ix <= x < ix + iw:
                return name

        return None

    def list_insets(self) -> List[Dict[str, Any]]:
        """
        Get list of all insets with their info.

        Returns:
            List of dicts with inset information
        """
        result = []
        for name, inset in self.insets.items():
            result.append({
                'name': name,
                'region': inset.region,
                'aligned': inset.aligned,
                'position': inset.position,
                'scale_factor': inset.scale_factor,
                'alignment_score': inset.alignment_score,
                'shape': inset.shape,
                'base_dimensions': inset.get_base_dimensions() if inset.aligned else None,
            })
        return result
