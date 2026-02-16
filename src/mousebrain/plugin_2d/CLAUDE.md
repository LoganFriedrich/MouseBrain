# plugin_2d/ - CODE Directory

> **This is a CODE directory.** All source code lives here. Data lives in `MouseBrain_Pipeline/2D_Slices/`.

## What This Is

The 2D plugin layer for the `mousebrain` monorepo. Contains two sub-packages:

| Sub-package | Purpose | napari plugin name |
|-------------|---------|-------------------|
| `sliceatlas/` | Slice analysis: detection, colocalization, quantification, atlas registration | `brainslice-2d` |
| `annotator/` | ND2 annotation and TIFF export | (integrated into brainslice-2d) |

## Package Structure

```
plugin_2d/
├── __init__.py
├── napari.yaml          # napari plugin manifest ("brainslice-2d")
├── sliceatlas/          # Main analysis package
│   ├── core/            # Algorithms: detection, colocalization, IO, config
│   ├── cli/             # CLI entry points (run_coloc.py)
│   ├── batch/           # Batch processing scripts
│   ├── tracker/         # SliceTracker for calibration run logging
│   ├── napari_plugin/   # napari widgets (main_widget, alignment, inset, annotator)
│   └── napari_widgets/  # Legacy widget stubs (being migrated)
└── annotator/           # ND2 annotation tools
    ├── core/            # Image utilities and IO
    ├── models/          # Channel data model
    ├── widgets/         # napari widget
    └── workers/         # Background loader threads
```

## Rules

1. **Imports**: Always use full package paths: `from mousebrain.plugin_2d.sliceatlas.core.config import ...`
2. **Config**: Path configuration is canonical in `mousebrain.config`. The 2D config (`sliceatlas/core/config.py`) imports from there.
3. **Tracker**: All detection and colocalization runs MUST be logged to the SliceTracker. Never remove tracker code.
4. **Data separation**: No data files in this tree. Raw data, results, and CSVs go to `MouseBrain_Pipeline/2D_Slices/`.
5. **Channel convention**: Channel 0 = 488nm (green/eYFP signal), Channel 1 = 561nm (red/nuclear H2B-mCherry).
