<!-- Parent: ../AGENTS.md -->
# Tissue - Domain

## Purpose
All tissue processing tools and data for whole-brain microscopy. Contains the MouseBrain tool and its processing pipelines for 3D cleared brains, 2D slices, and injury analysis.

## Structure
```
Tissue/
├── MouseBrain/                    # TOOL: mousebrain Python package
│   ├── pyproject.toml             # Package definition with [3d], [2d], [all] extras
│   ├── src/mousebrain/
│   │   ├── config.py              # Central path configuration (canonical)
│   │   ├── tracker.py             # Calibration run tracker (canonical)
│   │   ├── cli.py                 # `mousebrain` CLI entry point
│   │   ├── pipeline_3d/           # 3D napari plugin (canonical location)
│   │   └── pipeline_2d/           # 2D napari plugin
│   └── scripts/                   # Install/setup scripts
│
└── MouseBrain_Pipeline/           # DATA: working data being processed
    ├── 3D_Cleared/                # 3D brain data + legacy scripts
    │   ├── 1_Brains/              # Per-brain processing folders
    │   ├── 2_Data_Summary/        # Tracking CSVs, session logs
    │   └── util_Scripts/          # Legacy pipeline scripts (shims to mousebrain)
    ├── 2D_Slices/                 # 2D slice registration and analysis
    │   └── Script_Tools/SliceAtlas/  # Standalone brainslice package
    └── Injury/                    # Injury analysis pipeline
```

## Key Files
- `MouseBrain/src/mousebrain/config.py` — All path constants (BRAINS_ROOT, DATA_SUMMARY_DIR, etc.)
- `MouseBrain/src/mousebrain/tracker.py` — ExperimentTracker API
- `MouseBrain/src/mousebrain/pipeline_3d/tuning_widget.py` — Main 3D napari widget (10,000+ lines)

## For AI Agents
- **Canonical code** lives in `MouseBrain/src/mousebrain/`. The `util_Scripts/` copies are backward-compat shims.
- Import from `mousebrain.config` and `mousebrain.tracker`, not from `config` or `experiment_tracker`.
- The `mousebrain` package has optional extras: `pip install -e ".[3d]"`, `".[2d]"`, or `".[all]"`.
- 3D_Cleared is temporarily at `Tissue/3D_Cleared/` until the batch script `FINISH_RESTRUCTURE.bat` moves it into `MouseBrain_Pipeline/`.
