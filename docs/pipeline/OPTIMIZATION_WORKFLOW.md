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
   - **Soma diameter**: Expected cell body size in microns
   - **Ball filter XY/Z**: Spatial filter sizes
   - **Threshold**: Detection sensitivity
   - and the other four (ball overlap, soma spread, LoG sigma, max cluster)
5. Run on a small region first to test

Setup & Tuning (`Plugins -> BrainTools -> 3D: 2. Setup & Tuning`) does the same
with every run logged, all eight settings included. Its **Settled method (from
tracker)** button starts you from the method already settled for this imaging
setup, if there is one. When a run is right, select it and click **Mark as
Paradigm Best**: that makes it the settled method every brain imaged this way is
detected with (README, "The Settled Detection Method").

### Using the CLI (recommended for full runs)

```bash
# The settled method for this brain's imaging setup
python 4_detect_cells.py --brain YOUR_BRAIN

# A trial with some settings changed (logged as "custom")
python 4_detect_cells.py --brain YOUR_BRAIN --ball-xy 8 --threshold 9

# No settled method yet: give all eight
python 4_detect_cells.py --brain YOUR_BRAIN --ball-xy N --ball-z N --soma-diameter N \
    --threshold N --ball-overlap N --soma-spread N --log-sigma N --max-cluster N
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

If these are consistently under reference, try a lower threshold or smaller
ball filter. If over, try the opposite, or check classification.

## Step 5: Review and Adjust

### If counts are too LOW (missing cells):
- Decrease `threshold`
- Decrease `ball_xy_size`
- Check if `soma_diameter` matches your cells

### If counts are too HIGH (false positives):
- Increase `threshold`
- Increase `ball_xy_size`
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
# Start from the settled method (or, for a new imaging setup, all eight by hand)
python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5
python 5_classify_cells.py --brain 101_PROJ_01_02_2p5x_z5
python 6_count_regions.py --brain 101_PROJ_01_02_2p5x_z5
python util_compare_to_published.py --brain 101_PROJ_01_02_2p5x_z5

# Review output, then try a change
python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --threshold 9
# ... repeat classify, count, compare

# When a run is right: Setup & Tuning -> select it -> Mark as Paradigm Best
```

## Reference: Detection Settings

There are no built-in presets. Detection uses the settled method for the
brain's imaging setup, read from the tracker -- see "The Settled Detection
Method" in README.md for what it contains and how to change it.

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
