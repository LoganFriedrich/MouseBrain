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

**Important:** Step 4 will block until registration QC is approved -- so that you
don't spend hours of compute on a badly registered brain. That gate applies to
the cropped images, which are the ones registration is computed from.

Step 4 can also be run **before** cropping and registration, on the whole
extracted stack, which is useful when registration is waiting on something. It
is a trial run: it gives you a cell count but not counts per brain region. See
[Running detection before registration](#running-detection-before-registration).

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
<CONNECTOME_ROOT>\Tissue\MouseBrain_Pipeline\3D_Cleared\util_Scripts\
```

The experiment tracker creates its CSV at:
```
<CONNECTOME_ROOT>\Tissue\MouseBrain_Pipeline\3D_Cleared\2_Data_Summary\calibration_runs.csv
```

## Usage Examples

### Script 4: Detection
```bash
# Interactive mode
python 4_detect_cells.py

# With preset
python 4_detect_cells.py --brain <brain_id> --preset balanced

# With whatever settings are already proven for this kind of imaging
python 4_detect_cells.py --brain <brain_id> --routine

# Custom parameters
python 4_detect_cells.py --brain <brain_id> --ball-xy 5 --ball-z 12

# On the whole extracted stack, before cropping or registration
python 4_detect_cells.py --brain <brain_id> --source full --routine
```

`--routine` reads the best settings recorded for this brain's imaging paradigm
(magnification and Z-step, taken from the brain's folder name) out of the
calibration tracker, so you do not have to remember them or type them. If
nobody has marked a best run for that paradigm yet, it falls back to the
`balanced` preset and says so on screen.

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
# 2. Plugins → BrainTools → 3D: 3. Manual Crop
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

## Running detection before registration

Normally detection is step 4 of 6, and it runs on the cropped images that step 3
registered to the atlas. Sometimes you want to detect cells before any of that
has happened -- most often because registration is waiting on something: a
better scan of the same brain, a crop you have not made by hand yet, or your own
review of the QC image.

You can. Cell detection never reads the atlas. Cropping and registration exist
to put cells into *atlas* space, and that is only needed when you want counts
per brain region, which is step 6.

**How to do it**

In the terminal:

```bash
python 4_detect_cells.py --brain <brain_id> --source full --routine
```

In the GUI: launch `mousebrain`, open `Plugins -> BrainTools -> 3D: 2. Setup &
Tuning`, pick the brain and press Load. When a brain has no crop, the plugin
loads the uncropped stack automatically -- there is nothing extra to click.
Then tune and run detection as usual.

**What `--source` means**

| `--source` | Reads from | Can produce region counts? |
|---|---|---|
| `auto` (default) | best available: manual crop, else automatic crop, else the whole stack | only if it landed on a crop |
| `manual` | `2_Cropped_For_Registration_Manual` | yes |
| `cropped` | `2_Cropped_For_Registration` | yes |
| `full` | `1_Extracted_Full` | **no** |

A folder only counts as available if it actually contains `ch0/*.tif`. The crop
folders are created empty when the brain is first organised, and an empty folder
is not data.

**What you get, and what you don't**

You get a real cell count for those settings on that brain, candidate
coordinates you can look at on the images in napari, and a logged run in the
calibration tracker like any other.

You do not get counts per brain region, and you cannot carry these cells forward
into steps 5 and 6. The reason is coordinates: the atlas is fitted to a
*cropped* stack, so its coordinates start from a different corner than the whole
stack's do. Cells found on the whole stack do not line up with an atlas fitted
later. When the brain is finally cropped and registered, **detection is run
again** on the crop.

So treat the number as provisional. It answers "do these settings find cells
here, and roughly how many" -- which is exactly what you want to know while
registration waits.

**Where the results go**

Into a subfolder named after the images they came from:

```
4_Cell_Candidates/
├── from_1_Extracted_Full/
│   └── detected_cells.xml      ← a trial run on the whole stack
└── detected_cells.xml          ← the real run, on the registered crop
```

Steps 5 and 6 only ever look at the top level, so a trial run cannot be
classified or counted by mistake. The tracker records which images each run used
in its notes, for the same reason: two runs from different spaces are not
comparable as counts, and nobody should have to decode a path to notice that.

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
        │   └── from_1_Extracted_Full/    ← trial runs made before registration
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
conda activate MouseBrain    # or the full path of the environment if it was created with --prefix
pip install cellfinder brainreg brainglobe-segmentation brainglobe-atlasapi
pip install imaris-ims-file-reader tifffile numpy scipy matplotlib h5py
```
