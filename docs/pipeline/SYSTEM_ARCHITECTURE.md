# MouseBrain System Architecture

> **Note**: Code and data live apart. The source code is this repository (`src/mousebrain/`); the pipeline folder (`MouseBrain_Pipeline/`) holds data only. Do not put code in the pipeline folder.

## Overview

This is a **cell detection and counting pipeline** for whole-brain microscopy images, integrated with the BrainGlobe ecosystem for atlas registration.

**Project Goal**: Make BrainGlobe better by:
1. **Properly tracking what settings were used** for each detection attempt
2. **Enabling proper reload of historical runs** with full parameter context

The system has two main workflows:
1. **Calibration/Tuning** - Find optimal detection parameters for your specific tissue
2. **Production** - Run the full pipeline on all brains with tuned parameters

---

## Key Terminology

| Term | Definition | Use Context |
|------|------------|-------------|
| **Cohort** | A group of brains in same experimental condition (e.g., ENCR_02, CNT_02) | Organization |
| **Brain** | A single sample (e.g., 001_CNT_01_01_1p625x_z4) | Unit of processing |
| **Calibration Run** | A single detection attempt during parameter tuning | Tracked by tracker |
| **Best** | User-marked preferred calibration run for a brain | Marked with ★ |
| **Tracker** | Records ALL calibration runs with settings, parameters, results | Essential for reproducibility |
| **Session** | One napari usage session, runs marked with ● | Current work context |

### IMPORTANT: Calibration Runs ≠ Experiments

| Term | When Used | What It Means |
|------|-----------|---------------|
| **Calibration Run** | During tuning | Each detection attempt with specific parameters |
| **Experiment** | Scientific study | Formal study spanning multiple cohorts |

During calibration/tuning, we track **calibration runs** (settings, options, tools, runs).
We do NOT track "experiments" - that term refers to scientific studies, which is a completely different concept.

---

## Pipeline Root Layout

```
MouseBrain_Pipeline/                      # data only: <CONNECTOME_ROOT>/Tissue/MouseBrain_Pipeline
├── 3D_Cleared/
│   ├── 1_Brains/                         # per-brain folders (next section)
│   └── 2_Data_Summary/                   # calibration_runs.csv, region counts, sessions
├── 2D_Slices/                            # slice datasets (ND2, ROI JSON)
└── Registry/                             # analysis registry (mousebrain.analysis_registry)
    ├── exports/{analysis}/registry.json  # manifest with provenance, one entry per output
    ├── exports/{analysis}/...            # data outputs (CSV); invalidated files -> _archived/<ts>/
    ├── figures/{analysis}/...            # figures by animal/region
    ├── logs/{analysis}.log               # audit trail
    └── approved_method.json              # optional lab-approved method parameters
```

`Registry/` is where this tool records what it produced and how (method
parameters, source files, timestamp, staleness). Its location can be overridden
with `MOUSEBRAIN_REGISTRY_ROOT`. The tool never pushes results out; an
integrator may pull from `Registry/`. Each manifest's header carries the
approved method the analysis is judged by, refreshed on every write
(`approved_method_hash` is null for an analysis with none, such as the 3D
counts and orientation checks); writes hold the analysis's lock,
`Registry/logs/<analysis>.lock`; a damaged manifest is set aside into
`Registry/logs/`, never emptied, a new manifest is started in its place at
once (so an analysis folder is never without one), and the set-aside file is
listed by every reader and in that manifest's header until a person repairs
it; a refused registration changes no
result file. The registry creates no folders besides `exports/`, `figures/`
and `logs/`.
`exports/` and `figures/` hold only results and manifests. See
[ANALYSIS_REGISTRY.md](ANALYSIS_REGISTRY.md).

---

## Folder Structure (per brain)

```
1_Brains/
└── {mouse_id}/                           # e.g., 001_CNT_01_01
    └── {brain_id}/                        # e.g., 001_CNT_01_01_1p625x_z4
        ├── 1_Extracted_Full/              # Raw extracted channels from IMS
        │   ├── ch0/                       # Background channel (typically)
        │   ├── ch1/                       # Signal channel (typically)
        │   └── metadata.json
        │
        ├── 2_Cropped_For_Registration/    # Cropped to remove empty space
        │   ├── ch0/
        │   ├── ch0.zarr/                  # Zarr for fast loading (used only with ch0.zarr.complete.json)
        │   ├── ch1/
        │   ├── ch1.zarr/
        │   └── crop_info.json
        │
        ├── 2_Cropped_For_Registration_Manual/  # Optional: manual crop override
        │   ├── ch0/
        │   └── ch1/
        │
        ├── 3_Registered_Atlas/            # BrainGlobe registration output
        │   ├── brainreg.json              # Registration parameters
        │   ├── downsampled.tiff           # Brain downsampled to atlas resolution
        │   ├── boundaries.tiff            # Atlas region boundaries
        │   ├── registered_atlas.tiff      # Atlas labels in brain space
        │   └── ...
        │
        ├── 4_Cell_Candidates/             # Detection output
        │   ├── Detected_YYYYMMDD_HHMMSS.xml  # Cell coordinates (BrainGlobe XML format)
        │   └── ...
        │
        ├── 5_Classified_Cells/            # Classification output
        │   └── ...
        │
        └── 6_Region_Counts/               # Final counts per atlas region
            └── ...
```

---

## Tracker System

The **Tracker** is the backbone of the calibration workflow. It records:

### What It Tracks (during calibration):
- **Settings**: ball_xy, ball_z, soma_diameter, threshold, etc.
- **Options**: GPU usage, batch size, Z-range
- **Tools**: Which detection algorithm, which atlas
- **Runs**: Each detection execution with timestamp, status, output path
- **Results**: Number of cells detected, runtime, success/failure

### What It Does NOT Track (during calibration):
- "Experiments" in the scientific sense
- Cohort-level analysis
- Cross-brain comparisons

### Key Tracker Operations:
```python
# Log a new calibration run
tracker.log_detection(brain=..., ball_xy=..., threshold=..., output_path=...)

# Search for runs: brain= is ONE brain, matched by whole name, every run returned
tracker.search(brain="001_CNT_01_01_1p625x_z4", exp_type="detection")
# Part of a name, as a person types it into a search box; limit= gives a page
tracker.search(brain_contains="001", limit=50)

# Update run status
tracker.update_status(run_id, status="completed")

# Mark a run as best (highest priority for this brain)
tracker.mark_as_best(run_id)
```

---

## Napari Widget Modes

### Mode 1: Cell Detection & Tuning (default)
**Purpose**: Find optimal detection parameters for your tissue type.

**Loads**:
- Full-resolution brain from Zarr (fast!)
- Previous detection results as points layers

**Which images Load shows.** The images the atlas was fitted to -- the folder
registration recorded (crop, hand crop or full extraction) -- the same images
detection read from disk and classification use. A brain that is not
registered yet (or was registered before registration recorded its folder)
shows its hand crop, else its crop, else its full extraction. If the
registration names a folder that holds no images any more, Load stops with
"Registered images missing" and says which folder; register the brain again.
Convert to Zarr converts the same folder Load shows.

**Workflow**:
1. Select brain from dropdown
2. Click "Load Brain" → loads full-res Zarr
3. Adjust parameters (ball size, threshold, etc.)
4. Click "Run Detection" → generates XML, tracker records run
5. See results as points overlay
6. Compare with previous runs
7. Click "Mark Selected Layer as Best" when satisfied
8. Repeat until parameters are optimal

### Mode 2: Registration QC & Approval
**Purpose**: Verify that atlas registration is correct before trusting cell locations.

**Loads**:
- Downsampled brain (or manual crop if exists)
- Atlas boundaries overlay
- Atlas region labels

**Workflow**:
1. Switch to Registration QC mode
2. Select brain
3. Click "Load Brain" → loads registration view
4. Verify boundaries align with tissue
5. Approve or reject registration. The approval covers this registration only:
   registering the brain again needs a new approval. A rejection is recorded
   and puts the brain back at "Register" for every reader (see
   docs/pipeline/README.md, "Registration QC & Approval").

### The Zarr fast-load caches (`ch0.zarr`, `ch1.zarr`)

Loading a brain from its TIFFs takes minutes; the widget therefore keeps a Zarr
copy of each channel beside its TIFF folder and loads that in seconds. After a
load from TIFFs it builds the copies in the background (or press "Convert to
Zarr").

A copy is used only when it is **complete and current**: it is written into a
folder of its own, `ch0.zarr.partial-<random characters>`, renamed to `ch0.zarr`
when every plane is in, and then `ch0.zarr.complete.json` is written last. A load uses the Zarr copies only if
both channels have that record, the plane count matches the TIFFs in the folder,
and no TIFF is newer than the copy. Otherwise the terminal says
`Not using the Zarr cache (<why>); loading the TIFFs`, the brain loads from its
TIFFs (slower, always right), and the widget builds a new copy, removing the
unusable one first (the TIFFs are never touched).

**One conversion at a time.** While a conversion runs (the status line shows
`ch0: converting 150/600...`), pressing Convert to Zarr says "Conversion
Running" and starts nothing, and a Load does not start a second automatic
conversion. One conversion does both channels of the folder, ch0 then ch1.
(Before, a second conversion could start on the same channel and delete the
first one's planes under it, leaving a copy with planes missing marked
complete.) A `.partial-...` folder left by a conversion that was cut short is
deleted by the next conversion of that channel that finishes, once nothing has
been written into it for a day; it is an unfinished copy, the TIFFs are never
touched. You never need to delete these folders by hand.

Caches made before this rule have no completion record, so each is rebuilt once,
in the background, the next time its brain is loaded. Before the rule, an
interrupted or failed conversion left a cache that loaded as a partly blank
brain, and a new crop left the old crop's cache in use.

### Switching brains in Setup & Tuning

Selecting another brain in the brain list starts that brain from a clean slate:
its own metadata (never the previous brain's), no pre-filter result, no selected
classification, no training-data folder, no "last run". The buttons that act on
those (Save pre-filter results, Send to training, Promote last run to
production) are disabled until you make a new one for this brain.

**While a run is going the brain list is greyed out.** Run Test Detection and
Run Classification hold the list until their run returns, and each run keeps
its own record (brain, tracker run, folder, first plane tested) from start to
finish, so the result is saved and the tracker row completed for the brain
the run was started on. (Before, selecting another brain mid-run reset what
the finish relied on: the detected cells were not saved, the run stayed
"started", or the other brain's production cells were offered as the
classification's result.)

**A load that finishes after you picked another brain** shows the brain it
loaded, and says so on the status line: `Loaded: <brain loaded> -- but <brain
selected> is selected now`. Nothing of the selected brain (its Best or recent
detections and classifications, its Zarr conversion) is put over those
images. Press Load Brain into Napari to load the brain you selected.

Image layers already in the viewer stay there, but every image layer carries the
brain and folder it was loaded from. Run Test Detection and Run Classification
use only layers of the selected brain; if the viewer holds only another brain's
images they stop with "Images of another brain" -- load this brain first. Each
detection run records the image folder it read as `input_path` in the tracker,
and a training-data export records the folder of the images it cut cubes from.

**Run Classification uses only the registered images.** It cuts a small cube
around every candidate from the loaded signal layer, and the candidates are
positions in the images the atlas was fitted to. If the layer was loaded from
any other folder (or carries no record of its folder), it stops with "Loaded
Images Are Not The Registered Ones", naming both folders, and classifies
nothing. Load the brain again -- Load shows the registered images -- and run
it again.

### Which channel holds the cells (the signal)

A brain's `metadata.json` can say which channel is the signal in three ways:
`channels.signal_channel` (written by extraction, read by steps 3-5), and the
older `signal_channel` ("ch1") and `channels_swapped` keys. Load, Run Test
Detection and Run Classification all read them by ONE rule
(`mousebrain.brain_context.channel_roles`), from the metadata of the folder they
read: the keys that are present must agree, none present means ch0. When you
select a brain, the **Swap signal/background channels** checkbox is set from
that rule (unchecked = ch0 is the signal); detection read from disk follows the
checkbox, so change it only if you have looked and the other channel shows the
cells. (Before, the checkbox started checked for every brain and kept the last
brain's setting, while Load and classification each used a rule of their own.)

If a brain's metadata **contradicts itself** (for example `channels_swapped`
true but `channels.signal_channel` 0), nothing decides for you: the status line
shows `[!] the metadata contradicts itself about which channel holds the cells
(...)`, and the checkbox, Load and classification behave as they did before
this rule (the checkbox follows `channels_swapped`; Load shows ch1 as the
signal; classification and steps 3-5 follow `channels.signal_channel`). Look at both channels, then tell
whoever maintains the brain's metadata which one holds the cells so it can be
corrected.

**The crop window** (Plugins -> BrainTools -> 3D: 3. Manual Crop) asks "Signal
Channel (which has the cells?)". It starts on the brain's own answer from its
extraction metadata; leave it unless the other channel shows the cells.
Leaving it changes nothing in the crop's channel record. Choosing the other
channel writes it into the crop's `metadata.json` as `channels.signal_channel`
(and swaps `background_channel` with it), which steps 3-5 and Setup & Tuning
read. (Before, the answer was dropped by the process that does the crop.) If
the extraction metadata contradicts itself, neither channel is chosen and Apply
Crop says "Choose The Signal Channel" until you pick one. If the metadata names
a channel the window does not show (ch2 ...), neither is chosen and, left so,
the crop keeps the metadata's answer.

### Where a test run's cells are (partial-stack runs)

A test of part of the stack ("current slice +/- 25" or a Z range) detects on a
slice that starts at some plane. Its saved file (`detected_cells.xml` in the
run's folder under `optimization_runs/<brain>/test_<time>/`) holds every cell's
plane counted from the first plane of the WHOLE stack, the same as what is
drawn right after the run, and a file `whole_stack_z.json` beside it says so.
Runs saved before this was corrected have no such file: their positions are
counted from the first plane tested, and the widget moves them down by the
run's first plane when it loads them (the tracker's `det_z_start`; for rows
logged before that column existed, the `Z<first>-<last>` the widget wrote at
the start of the run's notes), so Load
Selected, Load Best and the run list show them at the right depth. The file
itself is not changed by loading. (Before, every reload put those cells too
shallow by that many planes.)

**Save (settings to metadata.json)** writes only into the folder the brain's
metadata was loaded from (the hand crop, else the crop, else the extraction). It
never creates a `metadata.json` where there was none -- that file is how the
status readers know a brain was extracted or cropped -- so a brain with no
metadata says "Nothing was saved".

### Mode 3: Results Viewer (future)
**Purpose**: View final cell counts per atlas region.

---

## Detection Results Loading

When loading detection results, the system:

1. **Queries tracker** for all calibration runs for this brain
2. **Filters** to completed runs only
3. **Sorts** by creation time (newest first)
4. **Loads**:
   - `best=True`: Most recent completed run (your best current attempt)
   - `best=False`: Second most recent (for comparison)

**Fallback**: If tracker unavailable, loads directly from `4_Cell_Candidates/*.xml`

### Current Limitation:
There's no UI to load arbitrary old runs (e.g., "load run #3 from 12 years ago").
Only most recent and second-most-recent are accessible via buttons.

**TODO**: Add a dropdown/selector to pick any historical run from tracker.

---

## UI Buttons Reference

| Button | Action |
|--------|--------|
| **Load Brain** | Loads brain into napari (mode-dependent) |
| **Run Detection** | Executes detection with current parameters, records to tracker |
| **Mark Selected Layer as Best** | Moves the selected brain's Best mark to the run the selected points layer shows (refused for a layer that is not a tracked run or belongs to another brain); the previous Best only loses its mark, it is not archived |
| **Load Best** | Loads the run marked Best for the brain (see "How the Best marks behave" in README.md) |
| **Load Most Recent** | Loads second-most recent run for comparison |

---

## Key Design Decisions

1. **Calibration ≠ Experiments**: During calibration, we track runs/settings/results. Not "experiments" in the scientific sense.

2. **Tracker is essential**: Every detection run is recorded. This enables reproducibility and parameter optimization.

3. **Two viewing modes**: Don't mix registration QC with cell detection. Keep them separate.

4. **Local files + Tracker**: Tracker provides metadata/search, but actual cell coordinates live in XML files.

5. **Best = Most Recent**: By default, "best" means your most recent attempt. User can override by marking any layer as best.
