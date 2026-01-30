"""
BrainSlice - Brain slice atlas alignment and cell quantification

A tool for:
- Aligning 2D confocal brain slice images to the Allen Brain Atlas
- Detecting nuclei using StarDist
- Measuring colocalization between channels
- Quantifying cells per atlas region

Usage:
    from braintools.pipeline_2d.sliceatlas.core import load_image, NucleiDetector, ColocalizationAnalyzer
    from braintools.pipeline_2d.sliceatlas.tracker import SliceTracker
"""

__version__ = "0.1.0"

# Convenience imports
from .core import (
    # Config
    BRAINSLICE_ROOT,
    DATA_DIR,
    # IO
    load_image,
    extract_channels,
    # Detection
    NucleiDetector,
    detect_nuclei,
    # Colocalization
    ColocalizationAnalyzer,
    analyze_colocalization,
    # Quantification
    RegionQuantifier,
    quantify_sample,
    # Atlas
    DualAtlasManager,
)

from .tracker import SliceTracker

__all__ = [
    '__version__',
    'BRAINSLICE_ROOT',
    'DATA_DIR',
    'load_image',
    'extract_channels',
    'NucleiDetector',
    'detect_nuclei',
    'ColocalizationAnalyzer',
    'analyze_colocalization',
    'RegionQuantifier',
    'quantify_sample',
    'DualAtlasManager',
    'SliceTracker',
]
