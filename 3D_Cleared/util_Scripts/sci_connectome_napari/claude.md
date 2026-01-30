# sci_connectome_napari - Napari Plugin Documentation

> **Navigation**: [← Back to Project Root](../../../claude.md) | [← util_Scripts](../claude.md) | [SYSTEM_ARCHITECTURE →](../SYSTEM_ARCHITECTURE.md)

---

## Overview

A napari plugin providing interactive tools for the SCI-Connectome cell detection pipeline.

**Key Features:**
- Context Panel for browsing/loading ANY historical calibration run
- Session tracking (● marks runs from current session)
- Mark as Best workflow with full visual feedback
- Two-mode workflow: Cell Detection vs Registration QC

---

## Widget Inventory

| Widget | File | Purpose |
|--------|------|---------|
| **Pipeline Dashboard** | `pipeline_widget.py` | Progress tracking, step navigation |
| **Setup & Tuning** | `tuning_widget.py` | Main configuration (5600+ lines) |
| **Manual Crop** | `manual_crop_widget.py` | Interactive brain cropping |
| **Experiments** | `experiments_widget.py` | Browse/search/compare runs |
| **Curation** | `curation_widget.py` | Manual cell review (Y/N) |
| **Session Documenter** | `session_documenter.py` | Auto-documentation (not a widget) |

---

## TuningWidget (Main Widget)

### Tab Structure
1. **Setup** - Voxel size + Orientation
2. **Crop** - Spinal cord removal
3. **Registration QC** - Verify atlas registration
4. **Detection** - Parameter tuning (main tab) + Context Panel
5. **Det Compare** - Compare detection runs
6. **Classify** - Classification parameters
7. **Class Compare** - Compare classification results
8. **Curate/Train** - Manual review + training
9. **Results** - View final outputs

### State Variables
```python
self.current_brain          # Path to selected brain folder
self.view_mode              # "cell_detection" or "registration_qc"
self.tracker                # ExperimentTracker instance
self.last_run_id            # Most recent detection run ID
self.current_session_id     # Session timestamp (YYYYMMDD_HHMMSS)
self.session_run_ids        # List of run IDs created this session
```

### Worker Threads
- `BrainLoaderWorker` - Loads brain images (Zarr or TIFF)
- `DetectionWorker` - Runs cellfinder detection
- `ClassificationWorker` - Runs classification

### Key Methods
```python
# Brain loading
load_brain_into_napari()        # Mode-aware loading
_on_brain_load_finished()       # Add layers to viewer

# Detection
run_test_detection()            # Run detection with current params
_on_detection_finished()        # Add results as points layer

# Context Panel (Run Management)
_refresh_context_runs()         # Populate runs table
_load_selected_run()            # Load selected run from table
_load_historical_run(exp_id)    # Load specific run by ID
_mark_selected_as_best()        # Mark run as best

# Mode switching
_on_mode_changed()              # Cell Detection vs Registration QC
_load_registration_qc_view()    # Load registration data
_update_registration_status()   # Update approval status label
```

---

## Context Panel (Run Management)

Located in Detection tab. Shows calibration runs for current brain.

### Features
- Shows all calibration runs for current brain
- Smart prioritization: ★ Best → ● This session → Recent
- Quick filters: Best only, This session
- Actions: Load Selected, Compare, Mark as Best

### Table Columns
| Column | Content |
|--------|---------|
| Icons | ★ (best) ● (this session) |
| Date/Time | When run was created |
| Cells | Number of cells found |
| Preset | Detection preset used |
| Rating | User rating (stars) |
| Status | completed/failed/etc |

### Priority Sorting
```python
def priority_key(run):
    is_best = run.get('marked_best', False)
    is_this_session = run.get('exp_id') in self.session_run_ids
    created = run.get('created_at', '')
    return (not is_best, not is_this_session, created)

sorted_runs = sorted(completed, key=priority_key, reverse=True)
```

---

## Session Tracking

Each napari session gets a unique ID:
```python
self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
self.session_run_ids = []  # Populated as runs complete
```

When detection finishes:
```python
if self.last_run_id and self.last_run_id not in self.session_run_ids:
    self.session_run_ids.append(self.last_run_id)
```

Runs from current session are marked with ● in the UI.

---

## Layer Color Scheme

### Detection Layers
| Type | Color | Symbol | Use |
|------|-------|--------|-----|
| `best` | Green #00FF00 | Circle | User-marked best |
| `this_session` | Sky Blue #00BFFF | Circle | Current session runs |
| `old` | Gray #888888 | Circle | Historical runs |
| `new` | White #FFFFFF | Circle | Just-ran detection |
| `rejected` | Orange #FFA500 | Cross | Rejected candidates |

### Style Definition (CELL_COLORS)
```python
CELL_COLORS = {
    'best': {
        'face': 'transparent',
        'edge': '#00FFFF',
        'symbol': 'o',
        'size': 14,
        'opacity': 0.2,
        'border_width': 0.1
    },
    'recent': {
        'face': 'transparent',
        'edge': '#FFA500',
        'symbol': 'o',
        'size': 14,
        'opacity': 0.2,
        'border_width': 0.1
    },
    # ... more styles
}
```

---

## Tracker Integration

### When Detection Runs
```python
# Log to tracker at start
self.last_run_id = self.tracker.log_detection(
    brain=self.current_brain.name,
    preset="custom",
    ball_xy=params['ball_xy_size'],
    ball_z=params['ball_z_size'],
    ...
    status="started"
)

# Update when finished
self.tracker.update_status(
    self.last_run_id,
    status="completed",
    det_cells_found=len(cells)
)

# Add to session tracking
self.session_run_ids.append(self.last_run_id)

# Refresh context panel
self._refresh_context_runs()
```

### When Loading Results
```python
# Query tracker for brain's runs
runs = self.tracker.search(
    brain=self.current_brain.name,
    exp_type="detection",
    status="completed"
)

# Load XML from output_path
output_path = run.get('output_path')
self._load_historical_run(exp_id)
```

### Mark as Best
```python
# Update tracker
self.tracker.mark_as_best(exp_id)

# Update layer styling
layer.edge_color = '#00FF00'  # Green for best
layer.name = "★ " + layer.name.lstrip("● ")
layer.metadata['is_best'] = True

# Refresh table
self._refresh_context_runs()
```

---

## View Modes

### Cell Detection & Tuning Mode (default)
**Purpose**: Tune detection parameters

**Loads**:
- Full-resolution brain from Zarr (`2_Cropped_For_Registration/ch0.zarr`)
- Previous detection results as points layers

**Workflow**:
1. Select brain
2. Load into napari
3. Adjust parameters
4. Run detection
5. Compare results in Context Panel
6. Mark best

### Registration QC & Approval Mode
**Purpose**: Verify atlas registration quality

**Loads**:
- Downsampled brain (`3_Registered_Atlas/downsampled.tiff`)
- Atlas boundaries (`boundaries.tiff`)
- Atlas regions (`registered_atlas.tiff`)

**Workflow**:
1. Switch to Registration QC mode
2. Load brain
3. Verify boundaries align with tissue
4. Approve/reject registration

---

## Layer Metadata

All detection layers store metadata for tracking:
```python
layer.metadata['exp_id'] = self.last_run_id   # Tracker ID
layer.metadata['is_best'] = False              # Best status
```

This allows Mark as Best to find and update the correct layer.

---

## Known Incomplete Features

### Stubs/TODOs
- Detection comparison (`generate_detection_difference()`)
- Classification comparison (`generate_classification_difference()`)
- Layer archival UI (hide/show archived layers)
- Training data integration (partially complete)
- Results tab (stub methods)

---

## Installation

```bash
cd y:\2_Connectome\3_Nuclei_Detection\util_Scripts\sci_connectome_napari
pip install -e .
```

Then in napari: **Plugins → SCI-Connectome Pipeline**

---

## File Structure

```
sci_connectome_napari/
├── sci_connectome_napari/
│   ├── __init__.py              # Package init, widget exports
│   ├── napari.yaml              # Plugin manifest
│   ├── pipeline_widget.py       # Pipeline Dashboard
│   ├── tuning_widget.py         # Setup & Tuning (main, 5600+ lines)
│   ├── manual_crop_widget.py    # Manual Crop tool
│   ├── crop_subprocess.py       # Subprocess for actual cropping
│   ├── experiments_widget.py    # Experiments browser
│   ├── curation_widget.py       # Curation tool
│   └── session_documenter.py    # Session documentation
├── setup.py                     # Package setup
└── claude.md                    # This file
```

---

## Quick Reference

### Load Brain
```python
self.load_brain_into_napari()
# Checks view_mode
# Cell Detection: loads Zarr, adds as Image layers
# Registration QC: loads downsampled + boundaries
```

### Run Detection
```python
self.run_test_detection()
# Creates DetectionWorker
# Logs to tracker
# Results added as Points layer with exp_id in metadata
```

### Load Historical Run
```python
self._load_historical_run(exp_id)
# Gets run from tracker
# Loads XML from output_path
# Adds as Points layer with appropriate styling
```

### Mark as Best
```python
self._mark_selected_as_best()
# Gets selected run from table
# Updates tracker.mark_as_best()
# Updates layer styling (green, ★ prefix)
# Refreshes context panel
```

---

## See Also
- [Project Root claude.md](../../../claude.md) - Start here
- [util_Scripts claude.md](../claude.md) - Script documentation
- [SYSTEM_ARCHITECTURE.md](../SYSTEM_ARCHITECTURE.md) - Technical architecture
