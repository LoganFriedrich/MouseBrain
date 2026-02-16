# tracker/ - CODE Directory

> **This is a CODE directory.** Calibration run tracking for the 2D slice pipeline.

## What This Is

The SliceTracker records every calibration run (registration, detection, colocalization, quantification) to a CSV file for reproducibility and history.

## Key Files

| File | Purpose |
|------|---------|
| `slice_tracker.py` | `SliceTracker` class - the main tracker API |
| `schema.py` | CSV column definitions and run type constants |

## SliceTracker API

```python
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

tracker = SliceTracker()

# Log runs (returns run_id)
run_id = tracker.log_detection(sample_id="E02_01_S13_DCN", channel="red", ...)
run_id = tracker.log_colocalization(sample_id="E02_01_S13_DCN", parent_run=det_run_id, ...)
run_id = tracker.log_registration(sample_id="E02_01_S13_DCN", atlas="allen_mouse_10um", ...)

# Update status
tracker.update_status(run_id, status="completed", det_nuclei_found=198)

# Mark best
tracker.mark_as_best(run_id)

# Search
runs = tracker.search(sample_id="E02_01", run_type="detection")
```

## Rules

1. **THE TRACKER IS SACRED**: Never remove, disable, or bypass tracker code. It is the most essential component for reproducibility.
2. **CSV location**: Defaults to `SLICES_2D_DATA_SUMMARY / "calibration_runs.csv"`, sourced from `mousebrain.config`.
3. **Auto-populated fields**: `_add_hierarchy_fields()` parses sample_id to auto-fill project, cohort, slice_num.
4. **Run IDs are unique**: Generated from type + timestamp + hash. Never reuse or fabricate run IDs.
5. **Update, don't delete**: Use `update_status()` to change run state. Never delete rows from the CSV.
