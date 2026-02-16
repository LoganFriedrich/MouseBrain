# annotator/core/ - CODE Directory

> **This is a CODE directory.** Core utilities for the annotator.

## Key Files

| File | Purpose |
|------|---------|
| `io.py` | ND2/TIFF file loading and saving |
| `image_utils.py` | Image manipulation (cropping, normalization, channel extraction) |

## Rules

1. **Shared IO**: If IO functionality overlaps with `sliceatlas/core/io.py`, prefer the sliceatlas version and import from there. Avoid duplicating image loading logic.
