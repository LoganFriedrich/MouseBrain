<!-- Parent: ../AGENTS.md -->
# MouseBrain - Tool

## Purpose
Unified Python package for tissue analysis. Provides napari plugins for 3D cleared brain cell detection and 2D slice analysis, plus CLI, path configuration, and calibration run tracking.

## Structure
```
MouseBrain/
├── pyproject.toml                 # Package: mousebrain with [3d], [2d], [all] extras
├── src/mousebrain/
│   ├── __init__.py                # Path re-exports, version
│   ├── config.py                  # Central path configuration (canonical)
│   ├── tracker.py                 # Calibration run tracker (canonical)
│   ├── cli.py                     # `mousebrain` CLI with feature detection
│   ├── pipeline_3d/               # 3D napari plugin
│   │   ├── napari.yaml            # Plugin: "sci-connectome-pipeline"
│   │   ├── tuning_widget.py       # Main widget (10,000+ lines, 9 tabs)
│   │   ├── pipeline_widget.py     # Pipeline execution widget
│   │   ├── manual_crop_widget.py  # Manual cropping
│   │   ├── experiments_widget.py  # Run history browser
│   │   ├── curation_widget.py     # Data curation
│   │   ├── session_documenter.py  # Session documentation
│   │   └── crop_subprocess.py     # Subprocess for cropping
│   └── pipeline_2d/               # 2D napari plugin
│       ├── napari.yaml            # Plugin: "brainslice-2d"
│       ├── sliceatlas/            # Slice analysis core
│       └── annotator/             # ND2 annotation and export
└── scripts/                       # Install/setup scripts
```

## Key APIs
```python
from mousebrain.config import BRAINS_ROOT, SCRIPTS_DIR, DATA_SUMMARY_DIR
from mousebrain.tracker import ExperimentTracker
```

## For AI Agents
- This is the CANONICAL code location. Do not edit the backward-compat shims in `util_Scripts/`.
- `tuning_widget.py` is 10,000+ lines with 9 tabs. Be very careful with edits.
- Install: `cd Tissue/MouseBrain && pip install -e ".[all]"`
- Launch: `mousebrain` CLI command
