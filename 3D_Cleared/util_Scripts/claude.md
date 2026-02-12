# util_Scripts - Pipeline Scripts Documentation

> **Navigation**: [← Back to Project Root](../../../claude.md) | [napari Plugin →](sci_connectome_napari/claude.md)

---

## Overview

This folder contains all Python scripts for the SCI-Connectome cell detection pipeline.

**Key principle**: The tracker (`experiment_tracker.py`) is essential - it records ALL calibration runs.

---

## Script Inventory

### Main Pipeline (Run in Order)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `1_organize_pipeline.py` | Create folder structure | `.ims` files | Pipeline folders |
| `2_extract_and_analyze.py` | Extract channels, auto-crop | `.ims` file | TIFF stacks + metadata |
| `3_register_to_atlas.py` | Atlas registration (brainreg) | Cropped TIFFs | Registration files |
| `4_detect_cells.py` | Cell candidate detection | Registered brain | `Detected_*.xml` |
| `5_classify_cells.py` | Classification (trained model) | Candidates | `cells.xml`, `rejected.xml` |
| `6_count_regions.py` | Count cells per region | Classified cells | `region_counts.csv` |

### Orchestration

| Script | Purpose |
|--------|---------|
| `RUN_PIPELINE.py` | Interactive menu-driven orchestrator with progress tracking |

### Core Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Portable path configuration, brain name parsing |
| `experiment_tracker.py` | Calibration run tracking (CSV-based, 49 columns) |

### Utilities

| Script | Purpose |
|--------|---------|
| `util_approve_registration.py` | Create registration approval marker |
| `util_registration_qc.py` | Generate QC visualizations |
| `util_manual_crop.py` | Launch napari manual crop |
| `util_optimize_crop.py` | Brute-force Y-crop optimization |
| `util_train_model.py` | Train custom classification models |
| `util_estimate_voxel_size.py` | Calculate voxel sizes |
| `util_ims_metadata_dump.py` | Extract IMS file metadata |
| `util_compare_to_published.py` | Compare results to published data |

---

## config.py

### Purpose
Centralized path configuration with automatic repo detection.

### Key Variables
```python
ROOT_PATH                # Root installation (auto-detected or Y:\2_Connectome)
BRAINS_ROOT              # Where brain folders live
DATA_SUMMARY_DIR         # Where tracking CSVs live
SCRIPTS_DIR              # This folder
MODELS_DIR               # Trained classification models
```

### Usage
```python
from config import BRAINS_ROOT, DATA_SUMMARY_DIR
```

### Brain Name Parsing
```python
from config import parse_brain_name

info = parse_brain_name("349_CNT_01_02_1p625x_z4")
# Returns: {
#   'brain_id': '349',
#   'project_code': 'CNT',
#   'project_name': 'Connectome',
#   'cohort': '01',
#   'subject': '02',
#   'imaging_params': '1p625x_z4'
# }
```

---

## experiment_tracker.py

### Purpose
Track all calibration runs with full parameter logging. **This is the most essential component.**

### Storage
CSV file at `2_Data_Summary/calibration_runs.csv` with 49 columns.

### Key Methods
```python
from experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()

# Log a detection run
run_id = tracker.log_detection(
    brain="349_CNT_01_02_1p625x_z4",
    preset="balanced",
    ball_xy=6, ball_z=15,
    soma_diameter=16,
    threshold=10,
    voxel_z=4.0, voxel_xy=4.0,
    output_path="/path/to/output.xml",
    status="started"
)

# Update status when complete
tracker.update_status(run_id, status="completed", det_cells_found=1234)

# Search for runs
runs = tracker.search(brain="349_CNT_01_02_1p625x_z4", exp_type="detection")

# Mark as best
tracker.mark_as_best(run_id)

# Rate a run
tracker.rate_experiment(run_id, rating=5, notes="Good results")
```

### CSV Schema (Key Columns)
- `exp_id` - Unique identifier (e.g., `det_20260107_173747_3295fc`)
- `exp_type` - detection, training, classification, counts
- `brain` - Full brain name
- `created_at` - ISO timestamp
- `status` - started, running, completed, failed
- `det_preset` - balanced, sensitive, conservative, large_cells
- `det_ball_xy`, `det_ball_z` - Ball filter sizes
- `det_soma_diameter` - Expected cell size
- `det_threshold` - Detection sensitivity
- `det_cells_found` - Result count
- `marked_best` - Boolean flag
- `rating` - 1-5 stars

### CLI Usage
```bash
python experiment_tracker.py                    # Show recent runs
python experiment_tracker.py --search "349"     # Search by brain
python experiment_tracker.py --best detection   # Best detection runs
python experiment_tracker.py --stats            # Statistics
```

---

## Detection Presets

| Preset | ball_xy | ball_z | soma | threshold | Use Case |
|--------|---------|--------|------|-----------|----------|
| sensitive | 4 | 10 | 12 | 8 | Dim labeling, small cells |
| balanced | 6 | 15 | 16 | 10 | General starting point |
| conservative | 8 | 20 | 20 | 12 | Bright labeling, reduce noise |
| large_cells | 10 | 25 | 25 | 10 | Motor neurons, Purkinje |

---

## Dependencies

### Core
- numpy, pathlib, json, csv

### Pipeline
- tifffile, scipy, matplotlib, h5py
- brainreg, cellfinder, brainglobe-segmentation

### Optional
- napari (for interactive tools)
- pandas (for some utilities)
- zarr, dask (for fast loading)

---

## Common Workflows

### Full Pipeline Run
```bash
python RUN_PIPELINE.py
# Follow interactive menu
```

### Parameter Tuning (napari)
1. Open napari: `napari`
2. Plugins → SCI-Connectome Pipeline → Setup & Tuning
3. Select brain, load into napari
4. Adjust parameters
5. Run detection (tracker logs automatically)
6. Compare results, mark best

### Check Calibration History
```bash
python experiment_tracker.py --search "349_CNT_01_02_1p625x_z4"
```

---

## See Also
- [Project Root claude.md](../../../claude.md) - Start here for overview
- [napari Plugin claude.md](sci_connectome_napari/claude.md) - Widget documentation
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Technical architecture
