"""
SCI-Connectome Pipeline Tools for napari.

This plugin provides GUI tools for the BrainGlobe cell detection pipeline:

WIDGETS:
--------
1. Pipeline Dashboard  - See your brains, check their progress, run next step
2. Setup & Tuning      - Configure settings before running the pipeline
3. Manual Crop         - Crop brain volumes (remove spinal cord for registration)
4. Experiments         - Browse and compare your experiment runs
5. Curation            - Rapid review of detection candidates

HOW TO USE:
-----------
In napari: Plugins > SCI-Connectome Pipeline > [choose widget]
"""

__version__ = "0.1.0"

# Lazy imports - widgets are only imported when accessed
# This avoids loading torch/keras/tensorflow at plugin discovery time

def __getattr__(name):
    """Lazy import widgets only when accessed."""
    if name == "PipelineDashboard":
        from .pipeline_widget import PipelineDashboard
        return PipelineDashboard
    elif name == "SetupWizard":
        from .pipeline_widget import SetupWizard
        return SetupWizard
    elif name == "TuningWidget":
        from .tuning_widget import TuningWidget
        return TuningWidget
    elif name == "ManualCropWidget":
        from .manual_crop_widget import ManualCropWidget
        return ManualCropWidget
    elif name == "ExperimentsWidget":
        from .experiments_widget import ExperimentsWidget
        return ExperimentsWidget
    elif name == "CurationWidget":
        from .curation_widget import CurationWidget
        return CurationWidget
    elif name == "SessionDocumenter":
        from .session_documenter import SessionDocumenter
        return SessionDocumenter
    elif name == "SessionDocumenterWidget":
        from .session_documenter import SessionDocumenterWidget
        return SessionDocumenterWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PipelineDashboard",
    "SetupWizard",
    "TuningWidget",
    "ManualCropWidget",
    "ExperimentsWidget",
    "CurationWidget",
    "SessionDocumenter",
    "SessionDocumenterWidget",
]
