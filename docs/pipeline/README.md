# BrainGlobe Pipeline Scripts

Complete automated pipeline for processing lightsheet microscopy data through BrainGlobe tools.

## Pipeline Script Order

```
1_organize_pipeline.py         → Set up folder structure, move IMS files

2_extract_and_analyze.py       → Extract TIFFs, auto-crop spinal cord
   └── util_manual_crop.py     → (Optional) Adjust crop manually in napari

3_register_to_atlas.py         → Register to Allen Mouse Brain Atlas
   ├── (auto) util_registration_qc.py  → Generates detailed QC images
   └── util_approve_registration.py    → (REQUIRED) Review & approve QC

4_detect_cells.py              → Detect cell candidates
5_classify_cells.py            → Classify cells with trained model
6_count_regions.py             → Count cells by brain region
```

**Important:** Step 4 will block until registration QC is approved! This ensures you don't waste compute on badly registered brains.

## Utility Scripts

```
experiment_tracker.py          → Core module: CSV-based experiment logging
util_experiments.py            → Browse/search/rate experiments interactively
util_optimize_crop.py          → Find optimal Y-crop via iterative testing
util_train_model.py            → Train custom classification models
util_manual_crop.py            → Manual crop tool (napari plugin launcher)
util_registration_qc.py        → Generate registration QC visualizations
util_approve_registration.py   → Approve registration after QC review
```

## What's New

| Script | Purpose |
|--------|---------|
| `4_detect_cells.py` | cellfinder detection with presets (sensitive/balanced/conservative) |
| `5_classify_cells.py` | cellfinder classification with trained model |
| `6_count_regions.py` | brainglobe-segmentation regional counting |
| `experiment_tracker.py` | Central CSV logging for all experiments |
| `util_experiments.py` | Interactive CLI for viewing/rating experiments |
| `util_optimize_crop.py` | Find optimal crop using registration quality metrics |
| `util_train_model.py` | Train custom cell classification networks |
| `util_manual_crop.py` | Napari plugin for manual crop adjustment |
| `util_registration_qc.py` | Generate detailed QC images comparing brain to atlas |
| `util_approve_registration.py` | Review & approve registration before cell detection |

## Installation

Copy all `.py` files to your scripts folder:
```
Y:\2_Connectome\Tissue\MouseBrain_Pipeline\3D_Cleared\util_Scripts\
```

The experiment tracker creates its CSV at:
```
Y:\2_Connectome\Tissue\MouseBrain_Pipeline\3D_Cleared\2_Data_Summary\calibration_runs.csv
```

## Usage Examples

### Script 4: Detection
```bash
# Interactive mode
python 4_detect_cells.py

# With preset
python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset balanced

# Custom parameters
python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --ball-xy 5 --ball-z 12
```

### Script 5: Classification
```bash
# Interactive mode
python 5_classify_cells.py

# With specific model
python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --model path/to/model.h5
```

### Script 6: Regional Counting
```bash
# Interactive mode
python 6_count_regions.py

# Process specific brain
python 6_count_regions.py --brain 349_CNT_01_02_1p625x_z4

# Process all pending
python 6_count_regions.py --all
```

### Utility: Browse Experiments
```bash
# Interactive browser
python util_experiments.py

# Quick commands
python util_experiments.py recent 20
python util_experiments.py search "349"
python util_experiments.py best detection
python util_experiments.py stats
```

### Utility: Manual Crop
```bash
# Launch napari with brain loaded and crop tool ready
python util_manual_crop.py --brain 349_CNT_01_02_1p625x_z4

# Or use the napari GUI:
# 1. Launch napari
# 2. Plugins → Connectome Pipeline → Manual Crop
```

### Utility: Registration QC & Approval
```bash
# Generate QC images for existing registration
python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4

# Review and approve registration (REQUIRED before cell detection)
python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4

# Just view QC without approving
python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4 --view

# Check approval status
python util_approve_registration.py --brain 349_CNT_01_02_1p625x_z4 --status
```

### Utility: Optimize Crop
```bash
# Test 0%, 10%, 20%, 30%, 40%, 50% crops
python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4

# Quick mode (0%, 25%, 50%)
python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4 --quick
```

## Detection Presets

| Preset | ball_xy | ball_z | soma | threshold | Use For |
|--------|---------|--------|------|-----------|---------|
| sensitive | 4 | 10 | 12 | 8 | Catching all cells, accept false positives |
| balanced | 6 | 15 | 16 | 10 | Default starting point |
| conservative | 8 | 20 | 20 | 12 | Fewer false positives |
| large_cells | 10 | 25 | 25 | 10 | Motor neurons, Purkinje cells |

## Folder Structure (Updated)

```
1_Brains/
└── 349_CNT_01_02/
    └── 349_CNT_01_02_1p625x_z4/
        ├── 0_Raw_IMS/
        ├── 1_Extracted_Full/
        │   └── QC_area_profile.png       ← Auto-crop detection visualization
        ├── 2_Cropped_For_Registration/
        ├── 3_Registered_Atlas/
        │   ├── QC_registration_detailed.png  ← Registration QC (auto-generated)
        │   └── .registration_approved        ← Approval marker file
        ├── 4_Cell_Candidates/            ← 4_detect_cells.py output
        ├── 5_Classified_Cells/           ← 5_classify_cells.py output
        ├── 6_Region_Analysis/            ← 6_count_regions.py output
        └── _crop_optimization/           ← util_optimize_crop.py output
```

## Experiment Tracking

All runs are logged to `experiments.csv` with:
- Unique experiment ID (e.g., `det_20241218_abc123`)
- All parameters used
- Timing information
- Results (cell counts, etc.)
- User ratings (1-5 stars)
- Notes

This eliminates the months-long manual optimization cycles by keeping a complete record of what was tried and what worked.

## Requirements

```bash
conda activate Y:\2_Connectome\envs\MouseBrain
pip install cellfinder brainreg brainglobe-segmentation brainglobe-atlasapi
pip install imaris-ims-file-reader tifffile numpy scipy matplotlib h5py
```
