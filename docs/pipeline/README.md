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
| `4_detect_cells.py` | cellfinder detection with the settled method (see below) |
| `5_classify_cells.py` | cellfinder classification with trained model |
| `6_count_regions.py` | brainglobe-segmentation regional counting |
| `experiment_tracker.py` | Central CSV logging for all experiments |
| `util_experiments.py` | Interactive CLI for viewing/rating experiments |
| `util_optimize_crop.py` | Find optimal crop using registration quality metrics |
| `util_train_model.py` | Train custom cell classification networks |
| `util_manual_crop.py` | Napari plugin for manual crop adjustment |
| `util_registration_qc.py` | Generate detailed QC images comparing brain to atlas |
| `util_approve_registration.py` | Review & approve registration before cell detection |

## Where the scripts live

The scripts come with the MouseBrain code; nothing is copied anywhere. The
numbered steps are in the repository's `scripts/pipeline/`, the `util_*.py`
scripts in `scripts/utils/`. The napari pipeline dashboard, the tuning
widget's buttons (registration QC, crop optimisation, pipeline comparison,
region counting) and `batch_process.py` find each script by name beside the
installed code, so they always run the version that is installed. To run one
by hand, run it from its folder, e.g.
`python scripts/utils/util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4`.

(Before the reorganisation the scripts were copied into the pipeline's
`3D_Cleared/util_Scripts/`. That folder is data, is empty now, and nothing runs
from it; until this was corrected the dashboard and those buttons answered
"script not found", and registration finished without its QC picture.)

The experiment tracker creates its CSV at:
```
<CONNECTOME_ROOT>\Tissue\MouseBrain_Pipeline\3D_Cleared\2_Data_Summary\calibration_runs.csv
```

## Usage Examples

### Script 4: Detection
```bash
# Interactive mode
python 4_detect_cells.py

# The normal case: the settled method for this brain's imaging setup
python 4_detect_cells.py --brain <brain_id>

# A trial with one setting changed (logged as "custom")
python 4_detect_cells.py --brain <brain_id> --threshold 9

# On the whole extracted stack, before cropping or registration
python 4_detect_cells.py --brain <brain_id> --source full
```

Step 4 reads all eight detection settings from the run marked Paradigm Best
for this brain's imaging setup (magnification and Z-step, taken from the brain's
folder name), so nobody has to remember them or type them. If there is no such
run, or it does not record all eight, step 4 stops and says what to do -- see
"The Settled Detection Method" below.

### Script 5: Classification
```bash
# Interactive mode
python 5_classify_cells.py

# With specific model
python 5_classify_cells.py --brain 349_CNT_01_02_1p625x_z4 --model path/to/model.h5
```

**The pre-filter (on unless `--no-prefilter`).** Before classifying, step 5
removes candidates within 100 um of the brain's edge (surface glow, damage,
uncleared tissue -- labelled neurons are not at the tissue edge) and candidates
far outside the atlas. What is left is written to
`4_Cell_Candidates/prefiltered/interior_candidates.xml` and is what gets
classified. Candidates that land inside the brain outline but in no atlas region
are kept.

To decide where a candidate is, the pre-filter uses BrainGlobe's own transform
from the detection images into the registration -- the same transform step 6
counts with. Since 2026-10-27; before that it used a shortcut that reordered the
image axes but did not reverse them, so on every brain it looked candidates up
millimetres from where they are (up-down on brains imaged `iar`, front-back as
well on `ipr`) and removed an essentially arbitrary set as "surface". Any
classification made before that date went through that wrong pre-filter.

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

### The orientation check (drawn by registration, or on demand)

Registration has to be told how the stack lies: which way each of its three axes
runs, as three letters such as `iar` (the brain's `metadata.json`, key
`orientation`; `iar` when nothing says otherwise). Told wrongly, it still
registers, for over an hour, and every cell then lands in the wrong region while
the numbers look normal. A re-stitched delivery can come back lying differently
from the first stitch.

So **before it starts, registration draws `3_Registered_Atlas/QC_orientation_check.png`**:
BrainGlobe's own "Check orientation" (the button in its registration window), as
a picture. The stack is turned into the atlas's orientation as declared and
averaged three ways -- seen from the front, from above and from the side -- under
the same three views of the atlas, each drawn at its real size in millimetres.

**How to read it:** if the declared orientation is right, each picture of the
brain (bottom row) lies the same way as the atlas above it -- same way up, front at
the same end (olfactory bulbs towards the top of the views from above and from the
side). A brain cut short by the person who prepared it only looks shorter. A
left-right mirror is nearly invisible here, because a brain is nearly symmetric;
look for a known one-sided feature (an injection site, damage) if a mirror is in
question.

To draw it for any brain, or to try another orientation before re-registering:

```bash
mousebrain-orientation-check 349_CNT_01_02_1p625x_z4
mousebrain-orientation-check 349_CNT_01_02_1p625x_z4 --orientation sar
```

A trial orientation is written as `QC_orientation_check_<orientation>.png`, so the
declared one's picture is never overwritten. Each picture is entered in
MouseBrain's analysis registry (analysis `3D_Orientation_Check`) and on the
registration's tracker row. It takes about two minutes (it reads every sixth
slice and pixel). To register with a different orientation, set `orientation` in
the brain's `metadata.json` and run registration again.

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

**How to read `QC_registration_detailed.png`:** each row is a coronal slice
through the registered brain, front of the brain at the top row, back at the
bottom. Left, the brain; middle, the atlas regions registration placed on it;
right, the brain with those regions' outlines drawn over it. A good
registration has the outlines following the brain's edges and its visible
landmarks (the ventricles, the midline, the corpus callosum). The rows are
spread over the stretch the atlas covers. Tissue the atlas does not have --
brainstem beyond where the atlas stops, partway down the medulla -- has no
outlines and gets no region; tissue the brain lost (a torn olfactory bulb)
leaves outlines with nothing under them. Neither is a registration fault.

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
python 4_detect_cells.py --brain <brain_id> --source full
```

In the GUI: launch `mousebrain`, open `Plugins -> BrainTools -> 3D: 2. Setup &
Tuning`, pick the brain and press Load. When a brain has no crop, the plugin
loads the uncropped stack automatically -- there is nothing extra to click.
Then tune and run detection as usual.

### Seeing the candidates again afterwards

**Loading the brain shows the images, not the candidates.** To put a finished
whole-stack run back on screen, load it from the **Context Panel's run list**
(the runs table for the current brain) rather than expecting it to appear with
the images.

This is a consequence of where the results are kept, and it is deliberate. The
plugin's "look in `4_Cell_Candidates` and load what is there" paths all read the
top level of that folder only, and a whole-stack run is written one level down,
in `from_1_Extracted_Full/`. If those scans reached into it, the classification
step could pick up coordinates from a space the atlas was never fitted to and
produce a confident, wrong answer. The run list works instead because it goes by
the output path the tracker recorded for that run, which points straight at the
subfolder.

**What `--source` means**

| `--source` | Reads from | Can produce region counts? |
|---|---|---|
| `auto` (default) | best available: manual crop, else automatic crop, else the whole stack | only if it landed on a crop |
| `manual` | `2_Cropped_For_Registration_Manual` | yes |
| `cropped` | `2_Cropped_For_Registration` | yes |
| `full` | `1_Extracted_Full` | only if the atlas was fitted to the whole stack (see below) |

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

**The exception: a brain registered on the whole stack.** A brain delivered
already cut needs no crop, so registration is run on `1_Extracted_Full` itself.
Registration writes down which images it used (`3_Registered_Atlas/registered_from.json`),
and from then on the whole stack IS the atlas's space: steps 5 and 6 read their
images from `1_Extracted_Full`, and step 5 picks up a run in
`from_1_Extracted_Full/` (after anything at the top level). Nothing needs moving
or re-running. A brain whose registration has no `registered_from.json`
(registered before it was written) is treated as registered on the crop.

**Where the results go**

Into a subfolder named after the images they came from:

```
4_Cell_Candidates/
├── from_1_Extracted_Full/
│   └── detected_cells.xml      ← a trial run on the whole stack
└── detected_cells.xml          ← the real run, on the registered crop
```

Steps 5 and 6 look at the top level, plus `from_<folder>` only when the atlas
was fitted to that folder's images, so a trial run cannot be classified or
counted by mistake. The tracker records which images each run used
in its notes, for the same reason: two runs from different spaces are not
comparable as counts, and nobody should have to decode a path to notice that.

## The Settled Detection Method

Detection always uses **the settled method**: the detection settings someone
tuned on a real brain and then marked as the ones to use for every brain imaged
the same way. There is no menu of presets to choose from.

**Where it comes from.** In napari, `Plugins -> BrainTools -> 3D: 2. Setup &
Tuning`, a person tunes detection on a brain until the cells look right, selects
that run and clicks **Mark as Paradigm Best**. That run, in the calibration
tracker (`2_Data_Summary/calibration_runs.csv`), is the settled method for its
**imaging setup** -- the magnification and z-step at the end of the brain name
(for example `1p625x_z4`). Marking a different run later changes the method for
every brain detected after that; no code changes.

**What it contains.** All eight cellfinder settings, each its own tracker column:

| Setting | Tracker column |
|---------|----------------|
| ball filter XY size | `det_ball_xy` |
| ball filter Z size | `det_ball_z` |
| soma diameter | `det_soma_diameter` |
| threshold (SDs above mean) | `det_threshold` |
| ball overlap fraction | `det_ball_overlap` |
| soma spread factor | `det_soma_spread` |
| LoG sigma size | `det_log_sigma` |
| max cluster size | `det_max_cluster` |

All eight matter. Changing only the last four, with the first four held fixed,
moved most of the cells a run found (see the `det_ball_overlap` comment in
`src/mousebrain/tracker.py`). Runs logged before the last four columns existed
recorded only the first four; such a run cannot be used as the settled method
until its other four are filled in.

**Who uses it.** Step 4, the Pipeline Dashboard (`3D: 1. Pipeline Dashboard`:
the "Detection settings" box shows "Settled method (from tracker)" with the
selected brain's eight numbers and the run they came from), Setup & Tuning's
**Settled method (from tracker)** button, `batch_process.py`,
`overnight_batch.py` and `RUN_PIPELINE.py`.

**When there is none.** If no run is marked Paradigm Best for a brain's imaging
setup, or that run does not record all eight settings, detection does not run:
the dashboard shows the reason in red and step 4 stops with a message saying
what to do. It never falls back to built-in numbers. (Four generic presets --
sensitive, balanced, conservative, large_cells -- used to fill that gap. They
were never tuned on real brains, and the counts they produced looked plausible
and were not, so they were removed.)

**Changing one setting for a trial.** Give it to step 4 on the command line,
e.g. `--threshold 9`. The run is logged as `custom`, never as the settled method.

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
        │   ├── QC_orientation_check.png      ← Was the orientation declared right? (drawn before registering)
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
