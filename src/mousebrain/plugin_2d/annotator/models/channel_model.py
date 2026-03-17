"""
channel_model.py - Channel settings data model.

Re-exports from the canonical location in sliceatlas.core.channel_model
to maintain backward compatibility for any code importing from here.
"""

from mousebrain.plugin_2d.sliceatlas.core.channel_model import (
    ChannelSettings,
    DEFAULT_CHANNEL_COLORS,
    create_default_channel_settings,
)

__all__ = ["ChannelSettings", "DEFAULT_CHANNEL_COLORS", "create_default_channel_settings"]
