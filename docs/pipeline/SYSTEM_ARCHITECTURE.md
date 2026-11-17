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

A copy is used only when it is **complete and current**: it is written as
`ch0.zarr.partial`, renamed to `ch0.zarr` when every plane is in, and then
`ch0.zarr.complete.json` is written last. A load uses the Zarr copies only if
both channels have that record, the plane count matches the TIFFs in the folder,
and no TIFF is newer than the copy. Otherwise the terminal says
`Not using the Zarr cache (<why>); loading the TIFFs`, the brain loads from its
TIFFs (slower, always right), and the widget builds a new copy, removing the
unusable one first (the TIFFs are never touched).

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

Image layers already in the viewer stay there, but every image layer carries the
brain and folder it was loaded from. Run Test Detection and Run Classification
use only layers of the selected brain; if the viewer holds only another brain's
images they stop with "Images of another brain" -- load this brain first. Each
detection run records the image folder it read as `input_path` in the tracker,
and a training-data export records the folder of the images it cut cubes from.

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
