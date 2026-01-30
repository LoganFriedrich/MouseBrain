# Cell Detection Optimization Workflow

This guide walks you through optimizing cell detection parameters for your tissue.

## Overview

The optimization loop:
1. Run detection with specific parameters
2. Classify the candidates
3. Count cells by region
4. Compare to published reference data
5. Adjust parameters based on results
6. Repeat

## Prerequisites

- Brain registered to atlas (Script 3 complete)
- Registration approved (`python util_approve_registration.py --brain YOUR_BRAIN`)
- cellfinder installed

## Step 1: Run Detection

### Using cellfinder's napari widget (recommended for tuning)

1. Open napari
2. Load your cropped brain image (from `2_Cropped_For_Registration/ch0/`)
3. Go to `Plugins` > `cellfinder` > `Cell detection`
4. Adjust parameters in the widget:
   - **Soma diameter**: Expected cell body size in microns (start with 16)
   - **Ball filter XY/Z**: Spatial filter sizes (start with 6/15)
   - **Threshold**: Detection sensitivity (start with 10)
5. Run on a small region first to test

### Using the CLI (recommended for full runs)

```bash
# With a preset as starting point
python 4_detect_cells.py --brain YOUR_BRAIN --preset balanced

# With custom parameters
python 4_detect_cells.py --brain YOUR_BRAIN --ball-xy 6 --ball-z 15 --soma-diameter 16 --threshold 10
```

### Parameter guidance

| Parameter | Effect when increased | Effect when decreased |
|-----------|----------------------|----------------------|
| soma_diameter | Catches larger cells, misses small | Catches smaller cells, more noise |
| ball_xy_size | Smoother filtering, misses small | More sensitive, more false positives |
| ball_z_size | Better for thick sections | Better for thin sections |
| threshold | Fewer cells, higher confidence | More cells, lower confidence |

## Step 2: Classify Cells

```bash
python 5_classify_cells.py --brain YOUR_BRAIN
```

This uses the trained network to filter detection candidates into real cells vs artifacts.

## Step 3: Count by Region

```bash
python 6_count_regions.py --brain YOUR_BRAIN
```

Outputs `cell_counts_by_region.csv` in the brain's `6_Region_Analysis/` folder.

## Step 4: Compare to Published Data

```bash
# Compare your counts to Wang et al. 2022 eLife reference
python util_compare_to_published.py --brain YOUR_BRAIN

# Just see the reference values
python util_compare_to_published.py --show-reference
```

### Reading the comparison output

```
Region                                        Yours      Ref     Diff   %Match
--------------------------------------------------------------------------------
Red Nucleus                                    1523     1860     -337    81.9%  OK
Gigantocellular reticular nucleus              6543     7997    -1454    81.8%  OK
Corticospinal                                  4521     8713    -4192    51.9%  !!
```

- `OK` = within 80-120% of reference
- `~` = within 50-150% of reference
- `!!` = outside expected range
- `MISS` = you have 0 but reference has cells

### Key regions to watch

These regions are most important for functional recovery analysis:
- **Red Nucleus** - major motor relay
- **Pedunculopontine nucleus (PPN)** - locomotion
- **Gigantocellular reticular nucleus** - motor output
- **Corticospinal** - voluntary movement

If these are consistently under reference, try more sensitive parameters.
If over, try more conservative or check classification.

## Step 5: Review and Adjust

### If counts are too LOW (missing cells):
- Decrease `threshold` (e.g., 10 → 8)
- Decrease `ball_xy_size` (e.g., 6 → 4)
- Check if `soma_diameter` matches your cells

### If counts are too HIGH (false positives):
- Increase `threshold` (e.g., 10 → 12)
- Increase `ball_xy_size` (e.g., 6 → 8)
- Consider retraining the classifier with your false positives

### If specific regions are wrong:
- May indicate registration issues in that area
- Run `python util_registration_qc.py --brain YOUR_BRAIN` to check

## Tracking Your Progress

Every detection run is automatically logged. View your experiment history:

```bash
# See recent runs
python experiment_tracker.py

# See runs for a specific brain
python experiment_tracker.py --search YOUR_BRAIN

# See best-rated detection runs
python experiment_tracker.py --best detection

# See statistics
python experiment_tracker.py --stats
```

### Rating your runs

After each detection, you're prompted to rate it 1-5. Use this to remember which parameter combinations worked best.

## Example Optimization Session

```bash
# Start with balanced preset
python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset balanced
python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4
python 6_count_regions.py --brain 349_CNT_01_02_1p625x_z4
python util_compare_to_published.py --brain 349_CNT_01_02_1p625x_z4

# Review output - if too low, try sensitive
python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --preset sensitive
# ... repeat classify, count, compare

# Fine-tune specific parameters
python 4_detect_cells.py --brain 349_CNT_01_02_1p625x_z4 --ball-xy 5 --threshold 9
# ... repeat

# Check what worked best
python experiment_tracker.py --search 349_CNT --best detection
```

## Reference: Detection Presets

These are starting points, not validated optimal settings:

| Preset | ball_xy | ball_z | soma | threshold | Use case |
|--------|---------|--------|------|-----------|----------|
| sensitive | 4 | 10 | 12 | 8 | Dim labeling, small cells |
| balanced | 6 | 15 | 16 | 10 | General starting point |
| conservative | 8 | 20 | 20 | 12 | Bright labeling, reduce noise |
| large_cells | 10 | 25 | 25 | 10 | Motor neurons, Purkinje cells |

## Reference: Published Baseline

From Wang et al. 2022 eLife (L1-injected uninjured, n=5):

| Region | Average | StdDev |
|--------|---------|--------|
| Corticospinal | 8713 | 433 |
| Gigantocellular reticular nucleus | 7997 | 1520 |
| Pontine Reticular Nuclei | 2185 | 941 |
| Red Nucleus | 1860 | 1050 |
| Pontine Central Gray Area | 1528 | 471 |
| Lateral Reticular Nuclei | 1441 | 568 |
| Hypothalamic Lateral Area | 1170 | 419 |
| Magnocellular Ret. Nuc. | 1097 | 528 |
| Solitariospinal Area | 878 | 252 |
| Midbrain Reticular Nuclei | 646 | 219 |
| Medullary Reticular Nuclei | 635 | 139 |
| Raphe Nuclei | 535 | 258 |

See `util_compare_to_published.py --show-reference` for the full list.
