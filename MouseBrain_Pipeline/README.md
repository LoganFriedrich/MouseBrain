# MouseBrain Pipeline Data

Working data directories for the MouseBrain tissue analysis tool.

## Contents

| Directory | Purpose |
|-----------|---------|
| `3D_Cleared/` | 3D cleared brain processing (registration, detection, classification) |
| `2D_Slices/` | 2D brain slice analysis (registration, colocalization) |
| `Injury/` | Injury analysis pipeline |

## 3D Brain Data Structure

Each brain in `3D_Cleared/1_Brains/` follows the processing pipeline:

```
{mouse_id}/{brain_id}/
├── 0_Raw_IMS/                        → Original .ims microscopy files
├── 1_Extracted_Full/                 → Extracted channels (TIFF)
├── 2_Cropped_For_Registration/       → Cropped + Zarr format
├── 3_Registered_Atlas/               → BrainGlobe registration output
├── 4_Cell_Candidates/                → Detection results (XML)
├── 5_Classified_Cells/               → Classification output
└── 6_Region_Analysis/                → Final per-region cell counts
```

## Tracking

- `3D_Cleared/2_Data_Summary/calibration_runs.csv` — All calibration runs with parameters
- `3D_Cleared/2_Data_Summary/sessions/` — Per-session markdown reports
