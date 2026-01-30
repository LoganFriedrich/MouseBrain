"""
channel_model.py - Channel settings data model.

Stores per-channel visualization settings including colormap, contrast, gamma, opacity.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional


@dataclass
class ChannelSettings:
    """Settings for a single image channel."""

    index: int
    name: str = ""
    visible: bool = True
    colormap: str = "gray"
    contrast_limits: Tuple[float, float] = (0.0, 65535.0)
    gamma: float = 1.0
    opacity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'index': self.index,
            'name': self.name,
            'visible': self.visible,
            'colormap': self.colormap,
            'contrast_limits': list(self.contrast_limits),
            'gamma': self.gamma,
            'opacity': self.opacity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelSettings':
        """Deserialize from dictionary."""
        data = data.copy()
        if 'contrast_limits' in data:
            data['contrast_limits'] = tuple(data['contrast_limits'])
        return cls(**data)

    def apply_to_napari_layer(self, layer) -> None:
        """Apply these settings to a napari image layer."""
        layer.visible = self.visible
        layer.contrast_limits = self.contrast_limits
        layer.gamma = self.gamma
        layer.opacity = self.opacity

        # Set colormap if it's a valid napari colormap
        try:
            layer.colormap = self.colormap
        except ValueError:
            # Fall back to gray if colormap not recognized
            layer.colormap = 'gray'

    @classmethod
    def from_napari_layer(cls, layer, index: int) -> 'ChannelSettings':
        """Create ChannelSettings from a napari image layer."""
        return cls(
            index=index,
            name=layer.name,
            visible=layer.visible,
            colormap=str(layer.colormap.name) if hasattr(layer.colormap, 'name') else 'gray',
            contrast_limits=tuple(layer.contrast_limits),
            gamma=layer.gamma,
            opacity=layer.opacity,
        )


# Default colormaps for multi-channel images
DEFAULT_CHANNEL_COLORS = [
    'green',     # Channel 0 - typically GFP
    'magenta',   # Channel 1 - typically RFP/mCherry
    'cyan',      # Channel 2
    'yellow',    # Channel 3
    'red',       # Channel 4
    'blue',      # Channel 5
    'gray',      # Channel 6
    'orange',    # Channel 7
]


def create_default_channel_settings(
    n_channels: int,
    channel_names: Optional[list] = None,
    dtype_max: float = 65535.0,
) -> list:
    """
    Create default channel settings for a multi-channel image.

    Args:
        n_channels: Number of channels
        channel_names: Optional list of channel names
        dtype_max: Maximum value for contrast limits (based on image dtype)

    Returns:
        List of ChannelSettings objects
    """
    settings = []

    for i in range(n_channels):
        name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel_{i}"
        color = DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]

        settings.append(ChannelSettings(
            index=i,
            name=name,
            visible=True,
            colormap=color,
            contrast_limits=(0.0, dtype_max),
            gamma=1.0,
            opacity=1.0,
        ))

    return settings
