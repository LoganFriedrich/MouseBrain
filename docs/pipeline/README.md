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
don't spend hours of compute on a badly registered brain. The approval is for one
registration: registering again needs a new approval, and no automated run ever
approves (see "Registration QC & Approval" below). That gate applies to
the cropped images, which are the ones registration is computed from.

Step 4 can also be run **before** cropping and registration, on the whole
extracted stack, which is useful when registration is waiting on something. It
is a trial run: it gives you a cell count but not counts per brain region. See
[Running detection before registration](#running-detection-before-registration).

## Which step is a brain at?

`RUN_PIPELINE.py`, the napari pipeline dashboard (Plugins -> BrainTools -> 3D:
1. Pipeline Dashboard) and `overnight_batch.py` all show each brain at a step.
Nothing records that step; it is worked out every time from which files exist,
and all three use the same rules (`mousebrain.pipeline_status`), so they always
agree with each other and with what the next script will do. The brain's step
is the first item on this list that is not done yet:

| Step | Done when |
|---|---|
| Extract | `1_Extracted_Full/metadata.json` exists (extraction writes it last, so it means every channel finished) |
| Crop | `metadata.json` exists in `2_Cropped_For_Registration` or `2_Cropped_For_Registration_Manual`, or the brain was registered on its full extraction (it arrived already cut, so there is nothing to crop) |
| Register | `3_Registered_Atlas/brainreg.json` exists |
| Approve | a person approved this registration (see "Registration QC & Approval" below) |
| Detect | a candidates file is where step 5 will look: the top of `4_Cell_Candidates`, or `4_Cell_Candidates/from_<the folder the atlas was fitted to>` |
| Pre-filter | optional (RUN_PIPELINE only). Done when any pre-filter output exists -- a `prefilter_report.json` in a `prefiltered_<time>` folder, or step 5's `prefiltered/interior_candidates.xml`. It is never shown once the brain is classified: step 5 pre-filters on its own by default |
| Classify | `5_Classified_Cells/cells.xml` exists |
| Count | a `.csv` exists in `6_Region_Analysis` |

**Stale outputs.** A file that exists but was made from something replaced since
is shown as `[!] ... STALE: <why>` and its step is offered again (nothing is
deleted or moved). Counts are stale when they are older than the current
registration or the classification; a classification when it is older than its
candidates or the images; detection when it is older than the images the atlas
is fitted to (or, for a registration that does not record its images, older
than the registration). Anything made from a stale output is stale too.

A trial detection run kept in `from_1_Extracted_Full` on a brain that was
registered on its crop does NOT count as detection: those coordinates are not
in the space the atlas was fitted to, and step 5 will not read them.

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

## Batch processing (`batch_process.py`)

`batch_process.py` takes brains through the steps one at a time and stops a
brain at the first check it fails, recording why in
`2_Data_Summary/pipeline_status.csv`. For each brain it runs extraction
(`2_extract_and_analyze.py --batch --brain <brain>`, that brain only), then
checks that `1_Extracted_Full` holds `metadata.json` and `ch*/` TIFFs. It then
needs a crop -- `ch0/` TIFFs plus `metadata.json` in
`2_Cropped_For_Registration_Manual` or `2_Cropped_For_Registration`; with none,
the brain stops as "needs manual crop": crop it by hand and run it again. Then
registration (`3_register_to_atlas.py --batch --brain <brain>`), after which the
brain always stops at "needs registration approval" until a person approves it.
(Until this was fixed every brain failed at extraction: the batch passed steps 2
and 3 an option they do not accept, and the checks looked for files the steps do
not write.)

`--brain` on step 2 takes the exact folder name. Without it step 2 extracts every
brain that still needs it, as before.

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

What step 6 writes, besides the per-brain files in `6_Region_Analysis`:

- `2_Data_Summary/region_counts.csv` and `elife_counts.csv` -- one row per brain,
  keyed by `brain` (`<mouse folder>/<brain folder>`). These are the files an
  integrator pulls. Each is written to a temporary file and swapped in, so a
  crash never leaves a cut-off file.
- When a brain is counted again, its previous row moves to
  `region_counts_archive.csv` / `elife_counts_archive.csv`, written in the
  archive's own column order; new columns extend the archive's header. (Before
  this was fixed, archived rows after the first were written in a different
  order from the header, so their values sat under the wrong column names. If
  step 6 finds an archive like that, it renames it to
  `region_counts_archive.malformed-<date>_<time>.csv`, keeps it untouched, prints
  a warning and starts a new archive.)
- The `det_*` and `voxel_*` columns: the settings of the detection run behind
  the brain's `cells.xml`, found by following the tracker from the
  classification that wrote `cells.xml`, through the pre-filter if there was
  one, to the detection run whose output those candidates are. If the chain
  cannot be followed (runs from before the tracker recorded these links), the
  `det_*` columns stay empty and step 6 prints `[!] The detection run behind
  this cells.xml could not be traced`; the voxel size then comes from the
  images' `metadata.json`. The `subject`, `cohort` and `brain_id` columns come
  from the brain folder name.
- An entry in MouseBrain's analysis registry, `Registry/exports/3D_Region_Counts/`
  (`mousebrain.analysis_registry`): the counts CSV, the total, the detection
  settings, the detection and classification run ids, the model, and the source
  `cells.xml` and registration. A registry problem is printed and does not undo
  the count.

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

**An approval belongs to ONE registration.** Approving writes
`3_Registered_Atlas/.registration_approved`, and that file records which
registration was approved (the size and time of the registration's own output
files: `brainreg.json`, `registered_atlas.tiff`, `boundaries.tiff`,
`deformation_field_0.tiff`). Every place that asks "is this brain approved?" --
step 4, RUN_PIPELINE, the napari pipeline dashboard and Setup & Tuning, the
overnight batch, batch_process -- compares that record with what is in the
folder now. What that means for you:

- **Registering a brain again needs a new approval**, even if the old
  registration was approved. When registration runs, the previous approval is
  moved into `3_Registered_Atlas/_archive/` together with the registration it
  was for. The brain shows "Approve" again until you look at the new QC
  pictures and approve them.
- **Approvals written before this rule** (an empty file, or one holding only
  "Approved: <date>") do not say which registration they were for, so they do
  **not** count. Such a brain shows "Approve" with the explanation "the approval
  on file does not say which registration it was for". Look at its current QC
  pictures and approve it again -- one click or one command.
- **Automated runs never approve.** The overnight batch and batch_process stop a
  brain at "waiting for approval" after registering it; their reports list
  those brains. Only a person approves: `util_approve_registration.py`, the
  Approve button in Setup & Tuning (Registration QC tab), the Approve button in
  the pipeline dashboard, or "Yes, approve it" in RUN_PIPELINE.
- **You cannot approve before registering.** With no `brainreg.json` there is
  nothing to approve, and the buttons say so.
- **Rejecting** (the Reject button) moves the approval into `_archive/` -- kept,
  not deleted -- and the brain is unapproved.

**Re-registering also makes the region counts stale.** The status readers
compare each output with what it was made from (see "Which step is a brain
at?"): counts older than the current registration are shown as `[!] ... STALE`
and the count step is offered again. Detection and classification are positions
in the images, so they stay valid when the atlas is re-fitted to the SAME
images; if the images were made again (a new crop), they are stale too.

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

**How the Best marks behave.** Each brain has at most one run marked Best (per
run type), and each imaging setup at most one Paradigm Best:

- *Mark as Best* moves the brain's Best mark to the selected run; other brains'
  marks are not touched.
- *Mark as Paradigm Best* makes the run the imaging setup's settled method AND
  its brain's Best; the brain's previous Best is cleared (a brain never has two),
  and the previous Paradigm Best of that imaging setup loses that mark.
- *Archiving* a run (when a new run replaces the Best in the comparison view)
  clears both its Best and its Paradigm Best. An archived run is never used as
  the settled method, even if an old tracker row still carries the mark.
- Each change is one locked write of the tracker, so a mark is never left half
  moved.
- *Load best detection* (the check box when loading a brain in Setup & Tuning)
  loads the run MARKED Best for the brain. If none is marked, it loads the
  highest-rated run that has not been archived and labels it "highest rated --
  none is marked best"; if there is none of those either, it loads nothing.

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
        │   ├── .registration_approved        ← A person's approval of THIS registration
        │   └── _archive/<time>/              ← Earlier registrations, each with its approval
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

**How the tracker file is written** (`2_Data_Summary/calibration_runs.csv`). Several
machines write it, and every status change rewrites the whole file, so each write:

- holds a lock file beside it, `calibration_runs.csv.lock`, for the moment of the
  write (well under a second). Another machine that wants to write waits and
  prints `Waiting for the tracker lock (held by ...)`. A lock left by a process
  that crashed is taken over after 2 minutes (by the file server's clock). If a
  write ever reports that the tracker has been locked for 5 minutes and nothing
  is running, delete the `.lock` file.
- keeps the file's own column order and every column in it, including columns
  added by a newer version of the code -- an older copy of the code can no
  longer delete them.
- writes a temporary file and swaps it in, so a crash part-way leaves the
  previous file, never a cut-off one.

**A run is never left at `started`.** Steps 3-6 mark their tracker row `failed`
(with the reason in `notes`) when they stop part-way -- an error, Ctrl-C, a
crash inside brainreg or cellfinder. Only a process killed outright (the task
ended from Task Manager, a power cut) can still leave `started`, because it runs
no code at all on the way out.

## Requirements

```bash
conda activate MouseBrain    # or the full path of the environment if it was created with --prefix
pip install cellfinder brainreg brainglobe-segmentation brainglobe-atlasapi
pip install imaris-ims-file-reader tifffile numpy scipy matplotlib h5py
```
