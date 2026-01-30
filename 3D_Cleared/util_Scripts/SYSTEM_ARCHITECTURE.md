# SCI-Connectome System Architecture

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
| **Brain** | A single sample (e.g., 349_CNT_01_02_1p625x_z4) | Unit of processing |
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

## Folder Structure (per brain)

```
1_Brains/
└── {mouse_id}/                           # e.g., 349_CNT_01_02
    └── {brain_id}/                        # e.g., 349_CNT_01_02_1p625x_z4
        ├── 1_Extracted_Full/              # Raw extracted channels from IMS
        │   ├── ch0/                       # Background channel (typically)
        │   ├── ch1/                       # Signal channel (typically)
        │   └── metadata.json
        │
        ├── 2_Cropped_For_Registration/    # Cropped to remove empty space
        │   ├── ch0/
        │   ├── ch0.zarr/                  # Zarr for fast loading
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

# Search for runs
tracker.search(brain="349_CNT_01_02_1p625x_z4", exp_type="detection")

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
5. Approve or reject registration

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
| **Mark Selected Layer as Best** | Marks currently selected points layer as BEST in tracker |
| **Load Best** | Loads most recent completed calibration run |
| **Load Most Recent** | Loads second-most recent run for comparison |

---

## Key Design Decisions

1. **Calibration ≠ Experiments**: During calibration, we track runs/settings/results. Not "experiments" in the scientific sense.

2. **Tracker is essential**: Every detection run is recorded. This enables reproducibility and parameter optimization.

3. **Two viewing modes**: Don't mix registration QC with cell detection. Keep them separate.

4. **Local files + Tracker**: Tracker provides metadata/search, but actual cell coordinates live in XML files.

5. **Best = Most Recent**: By default, "best" means your most recent attempt. User can override by marking any layer as best.
