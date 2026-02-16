# annotator/ - CODE Directory

> **This is a CODE directory.** ND2 annotation and TIFF export tools.

## What This Is

A napari-based annotation tool for ND2 microscopy files. Allows users to view multi-channel ND2 images, add annotations, and export annotated regions as TIFF files.

## Sub-packages

| Directory | Purpose |
|-----------|---------|
| `core/` | Image utilities and IO (loading, channel extraction, format conversion) |
| `models/` | Data models (channel metadata) |
| `widgets/` | napari widget for the annotation UI |
| `workers/` | Background thread workers for ND2 loading |

## Rules

1. **Data separation**: Annotation results and exports go to `MouseBrain_Pipeline/2D_Slices/`, not here.
2. **ND2 format**: Uses `nd2` library for reading Nikon ND2 files. Channel order is file-dependent.
3. **Worker threads**: ND2 loading happens in background workers (`workers/loader_worker.py`) to keep napari responsive.
