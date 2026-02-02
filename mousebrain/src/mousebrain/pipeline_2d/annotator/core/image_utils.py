"""
image_utils.py - Image manipulation utilities for Slice Annotator.

Provides functions for contrast adjustment, gamma correction, and channel compositing.
These are used both for display and for export (flattening visualization settings).

Usage:
    from mousebrain.pipeline_2d.annotator.core.image_utils import apply_contrast, apply_gamma, composite_channels

    # Apply contrast limits
    adjusted = apply_contrast(image, min_val=100, max_val=4000)

    # Apply gamma
    gamma_corrected = apply_gamma(adjusted, gamma=1.2)

    # Composite multiple channels with colors
    rgb = composite_channels(channels, colors, opacities)
"""

from typing import List, Tuple, Optional, Union
import numpy as np


# Standard LUT colors (RGB tuples, 0-255)
COLORMAPS = {
    'gray': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'white': (255, 255, 255),
}


def apply_contrast(
    data: np.ndarray,
    min_val: float,
    max_val: float,
    output_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Apply contrast limits (window/level) to image data.

    Args:
        data: Input array (any dtype)
        min_val: Minimum value (maps to output_range[0])
        max_val: Maximum value (maps to output_range[1])
        output_range: Output value range (default 0-1)

    Returns:
        Float array with values scaled to output_range
    """
    data = data.astype(np.float32)

    # Avoid division by zero
    if max_val <= min_val:
        max_val = min_val + 1

    # Scale to 0-1
    scaled = (data - min_val) / (max_val - min_val)

    # Clip to range
    scaled = np.clip(scaled, 0.0, 1.0)

    # Map to output range
    out_min, out_max = output_range
    return scaled * (out_max - out_min) + out_min


def apply_gamma(
    data: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Apply gamma correction to image data.

    Args:
        data: Input array (should be 0-1 range for proper gamma)
        gamma: Gamma value (< 1 brightens, > 1 darkens midtones)

    Returns:
        Gamma-corrected array (same dtype as input)
    """
    if gamma == 1.0:
        return data

    # Ensure we're working with float
    was_int = np.issubdtype(data.dtype, np.integer)
    if was_int:
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val

    # Apply gamma (only to positive values)
    result = np.power(np.clip(data, 0, None), gamma)

    if was_int:
        result = (result * max_val).astype(data.dtype)

    return result


def normalize_to_uint8(
    data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize array to uint8 (0-255).

    Args:
        data: Input array
        min_val: Min value for scaling (default: data min)
        max_val: Max value for scaling (default: data max)

    Returns:
        uint8 array
    """
    if min_val is None:
        min_val = float(np.min(data))
    if max_val is None:
        max_val = float(np.max(data))

    scaled = apply_contrast(data, min_val, max_val, output_range=(0.0, 255.0))
    return scaled.astype(np.uint8)


def channel_to_rgb(
    channel: np.ndarray,
    color: Union[str, Tuple[int, int, int]],
    contrast_limits: Optional[Tuple[float, float]] = None,
    gamma: float = 1.0,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    Convert single channel to RGB with specified color.

    Args:
        channel: 2D grayscale array
        color: Color name (from COLORMAPS) or RGB tuple
        contrast_limits: (min, max) contrast range
        gamma: Gamma correction value
        opacity: Channel opacity (0-1)

    Returns:
        RGB array with shape (H, W, 3), float32 0-1 range
    """
    # Get color RGB
    if isinstance(color, str):
        rgb = COLORMAPS.get(color.lower(), COLORMAPS['white'])
    else:
        rgb = color

    # Normalize color to 0-1
    r, g, b = [c / 255.0 for c in rgb]

    # Apply contrast
    if contrast_limits is not None:
        min_val, max_val = contrast_limits
        channel = apply_contrast(channel, min_val, max_val)
    else:
        # Auto-scale
        channel = apply_contrast(channel, float(np.min(channel)), float(np.max(channel)))

    # Apply gamma
    if gamma != 1.0:
        channel = apply_gamma(channel, gamma)

    # Apply opacity
    channel = channel * opacity

    # Create RGB by multiplying intensity by color
    h, w = channel.shape
    rgb_out = np.zeros((h, w, 3), dtype=np.float32)
    rgb_out[:, :, 0] = channel * r
    rgb_out[:, :, 1] = channel * g
    rgb_out[:, :, 2] = channel * b

    return rgb_out


def composite_channels(
    channels: List[np.ndarray],
    colors: List[Union[str, Tuple[int, int, int]]],
    contrast_limits: Optional[List[Tuple[float, float]]] = None,
    gammas: Optional[List[float]] = None,
    opacities: Optional[List[float]] = None,
    mode: str = 'additive',
) -> np.ndarray:
    """
    Composite multiple channels into single RGB image.

    Args:
        channels: List of 2D channel arrays
        colors: List of colors for each channel
        contrast_limits: List of (min, max) tuples per channel
        gammas: List of gamma values per channel
        opacities: List of opacity values per channel
        mode: Blending mode - 'additive' or 'composite'

    Returns:
        RGB array with shape (H, W, 3), uint8 0-255 range
    """
    if not channels:
        raise ValueError("No channels provided")

    n_channels = len(channels)
    h, w = channels[0].shape

    # Default values
    if contrast_limits is None:
        contrast_limits = [None] * n_channels
    if gammas is None:
        gammas = [1.0] * n_channels
    if opacities is None:
        opacities = [1.0] * n_channels

    # Extend lists if needed
    while len(colors) < n_channels:
        colors.append('white')
    while len(contrast_limits) < n_channels:
        contrast_limits.append(None)
    while len(gammas) < n_channels:
        gammas.append(1.0)
    while len(opacities) < n_channels:
        opacities.append(1.0)

    # Initialize composite
    composite = np.zeros((h, w, 3), dtype=np.float32)

    for i, channel in enumerate(channels):
        if opacities[i] <= 0:
            continue

        rgb = channel_to_rgb(
            channel,
            colors[i],
            contrast_limits[i],
            gammas[i],
            opacities[i],
        )

        if mode == 'additive':
            composite += rgb
        else:  # composite/over mode
            # Simple alpha compositing
            alpha = opacities[i]
            composite = composite * (1 - alpha) + rgb

    # Clip and convert to uint8
    composite = np.clip(composite, 0.0, 1.0)
    return (composite * 255).astype(np.uint8)


def auto_contrast(
    data: np.ndarray,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> Tuple[float, float]:
    """
    Calculate auto-contrast limits using percentiles.

    Args:
        data: Input array
        percentile_low: Lower percentile for min
        percentile_high: Upper percentile for max

    Returns:
        Tuple of (min_val, max_val)
    """
    # Flatten and sample if very large
    flat = data.ravel()
    if len(flat) > 1000000:
        # Sample for speed
        indices = np.random.choice(len(flat), 1000000, replace=False)
        flat = flat[indices]

    min_val = float(np.percentile(flat, percentile_low))
    max_val = float(np.percentile(flat, percentile_high))

    return min_val, max_val


def get_histogram(
    data: np.ndarray,
    bins: int = 256,
    range_: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of image data.

    Args:
        data: Input array
        bins: Number of histogram bins
        range_: (min, max) range for histogram

    Returns:
        Tuple of (counts, bin_edges)
    """
    flat = data.ravel()

    # Sample if very large
    if len(flat) > 1000000:
        indices = np.random.choice(len(flat), 1000000, replace=False)
        flat = flat[indices]

    if range_ is None:
        range_ = (float(np.min(flat)), float(np.max(flat)))

    counts, bin_edges = np.histogram(flat, bins=bins, range=range_)
    return counts, bin_edges
