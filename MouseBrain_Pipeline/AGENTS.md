<!-- Parent: ../AGENTS.md -->
# MouseBrain_Pipeline - Data

## Purpose
Working data directories for tissue analysis. Contains brain imaging data, detection results, calibration tracking CSVs, and legacy pipeline scripts.

## Structure
```
MouseBrain_Pipeline/
├── 3D_Cleared/                        # 3D cleared brain data
│   ├── 1_Brains/                      # Per-brain processing folders
│   │   └── {mouse_id}/{brain_id}/     # Individual brain data
│   │       ├── 0_Raw_IMS/             # Original .ims files
│   │       ├── 1_Extracted_Full/      # Extracted channels
│   │       ├── 2_Cropped_For_Registration/
│   │       ├── 3_Registered_Atlas/    # BrainGlobe output
│   │       ├── 4_Cell_Candidates/     # Detection XML
│   │       ├── 5_Classified_Cells/    # Classification
│   │       └── 6_Region_Analysis/     # Final counts
│   ├── 2_Data_Summary/                # Tracking and results
│   │   ├── calibration_runs.csv       # ALL runs (49+ columns)
│   │   └── sessions/                  # Session logs
│   ├── 0_Extras/                      # Additional resources
│   └── util_Scripts/                  # Legacy pipeline scripts
│       ├── config.py                  # Backward-compat shim → mousebrain.config
│       ├── experiment_tracker.py      # Backward-compat shim → mousebrain.tracker
│       ├── 1-6_*.py                   # Pipeline stage scripts
│       ├── RUN_PIPELINE.py            # Pipeline orchestrator
│       └── sci_connectome_napari/     # Legacy napari plugin (monorepo is canonical)
│
├── 2D_Slices/                         # 2D slice data
│   └── Script_Tools/SliceAtlas/       # Standalone brainslice package
│
└── Injury/                            # Injury analysis data
```

## For AI Agents
- Brain naming format: `{BRAIN#}_{PROJECT}_{COHORT#}_{SUBJECT#}_{MAG}x_z{ZSTEP}`
- `util_Scripts/*.py` are operational scripts that run ON the data. They import from `mousebrain`.
- `util_Scripts/config.py` and `experiment_tracker.py` are SHIMS — they redirect to `mousebrain.config` and `mousebrain.tracker`. Do not add logic to them.
- `sci_connectome_napari/` in util_Scripts is LEGACY. The canonical napari plugin is at `MouseBrain/src/mousebrain/pipeline_3d/`.
