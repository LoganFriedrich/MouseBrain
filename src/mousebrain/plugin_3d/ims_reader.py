"""Napari reader for raw Imaris .ims files (brains / PANO scans).

Registered via mousebrain/napari.yaml so napari's ``File -> Open File(s)...`` and
drag-drop open any ``.ims`` directly -- no CLI arguments, no pipeline organizing.
Each channel is loaded lazily through the Imaris resolution pyramid (multiscale)
and scaled to true microns, so the scale bar and 3-D aspect are physically
correct.

This is the GUI counterpart of the standalone ``view_raw_ims.py`` viewer: the
same 3-D channel/level adapter, exposed to napari as a reader contribution.
"""
from __future__ import annotations

import os

import numpy as np
import dask.array as da

# Distinct colormaps per channel (ch0 gray, ch1 green, ...).
_CHANNEL_COLORS = ["gray", "green", "magenta", "cyan", "red", "blue", "yellow"]


class _ChannelLevel:
    """3-D ``(z, y, x)`` view of one channel at one resolution level.

    dask slices this object with 3-D indices only; we prepend
    ``(timepoint=0, channel=c)`` to build the 5-D index the Imaris reader
    expects. Indexing the reader with a 5-D tuple directly collapses the
    timepoint/channel axes to a 3-D array, which then confuses dask's block
    indexing -- this adapter is what avoids that.
    """

    def __init__(self, reader, c):
        self._r = reader
        self._c = c
        self.shape = tuple(reader.shape[2:])   # (z, y, x)
        self.dtype = reader.dtype
        self.ndim = 3

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self._r[(0, self._c) + idx]


def napari_get_reader(path):
    """napari hook: return a reader callable if ``path`` is an .ims file, else None."""
    p = path[0] if isinstance(path, (list, tuple)) else path
    if isinstance(p, str) and p.lower().endswith(".ims"):
        return read_ims
    return None


def read_ims(path):
    """Read an .ims into napari layers -- one scaled, lazy multiscale image per channel."""
    # Local import so merely loading the plugin manifest never imports the reader.
    from imaris_ims_file_reader.ims import ims

    p = path[0] if isinstance(path, (list, tuple)) else path
    label = os.path.splitext(os.path.basename(p))[0]

    base = ims(p)
    n_levels = int(base.ResolutionLevels)
    n_chan = int(base.Channels)
    zres, yres, xres = base.resolution         # microns (z, y, x) at full resolution
    scale = (float(zres), float(yres), float(xres))

    # One reader handle per resolution level, shared across channels.
    readers = {L: ims(p, ResolutionLevelLock=L) for L in range(n_levels)}

    layers = []
    for c in range(n_chan):
        pyramid = [
            da.from_array(
                _ChannelLevel(readers[L], c),
                chunks=(8, 1024, 1024),
                name=f"{label}_c{c}_L{L}",
                meta=np.empty((0, 0, 0), dtype=readers[L].dtype),
            )
            for L in range(n_levels)
        ]
        meta = {
            "name": f"{label} ch{c}",
            "scale": scale,
            "colormap": _CHANNEL_COLORS[c % len(_CHANNEL_COLORS)],
            "blending": "additive" if n_chan > 1 else "translucent",
        }
        if len(pyramid) > 1:
            # multiscale: napari picks the level to fetch based on zoom (lazy over NAS)
            layers.append((pyramid, {**meta, "multiscale": True}, "image"))
        else:
            layers.append((pyramid[0], meta, "image"))
    return layers
