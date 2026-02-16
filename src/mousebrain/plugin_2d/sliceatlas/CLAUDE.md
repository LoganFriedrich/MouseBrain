# sliceatlas/ - CODE Directory

> **This is a CODE directory.** All source code for 2D slice analysis lives here.

## What This Is

The main analysis package for 2D brain slice processing. Handles the full pipeline from raw ND2 files to quantified colocalization results with QC figures.

## Sub-packages

| Directory | Purpose |
|-----------|---------|
| `core/` | Algorithms and configuration (detection, colocalization, IO, visualization, config) |
| `cli/` | Command-line entry points (`run_coloc.py`) |
| `batch/` | Batch processing across multiple samples (`batch_encr.py`) |
| `tracker/` | `SliceTracker` - calibration run logging to CSV |
| `napari_plugin/` | Interactive napari widgets for the GUI |
| `napari_widgets/` | Legacy widget stubs (being migrated to `napari_plugin/`) |

## Key Workflows

1. **Single-sample CLI**: `python -m mousebrain.plugin_2d.sliceatlas.cli.run_coloc <nd2_file>`
2. **Batch processing**: `python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr`
3. **Interactive napari**: Plugins > brainslice-2d > Slice Analysis

## Rules

1. **"Calibration run" not "experiment"**: During parameter tuning, we track "calibration runs." The word "experiment" means a scientific study across cohorts.
2. **Tracker is sacred**: Every detection and colocalization run MUST be logged. Never remove tracker calls.
3. **Config hierarchy**: `sliceatlas/core/config.py` imports paths from `mousebrain.config`. Do not add independent path detection.
4. **Data goes to Pipeline**: Results, CSVs, QC images go to `MouseBrain_Pipeline/2D_Slices/`, never stored in this code tree.
5. **Frame boundary accuracy**: Never describe detection accuracy as "good" or "excellent." Report the error rate and next steps.
