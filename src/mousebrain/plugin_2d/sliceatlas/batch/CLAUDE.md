# batch/ - CODE Directory

> **This is a CODE directory.** Batch processing scripts for running analysis across multiple samples.

## What This Is

Batch processing entry points that iterate over all ND2 files in a project and run detection + colocalization on each.

## Key Files

| File | Purpose |
|------|---------|
| `batch_encr.py` | Batch colocalization for ENCR project ND2 files |

## batch_encr.py

Processes all ENCR HD region ND2 files through the detection + colocalization pipeline.

### Key behavior:
- Searches `DATA_DIR / "ENCR"` for subject folders, looks in `0_Raw_HD/` and `0_Raw/` subdirectories
- Falls back to old structure (`ENCR_02_*_HD_Regions/`) if new structure not found
- Routes output to per-subject pipeline folders (`3_Detected/`, `4_Quantified/`)
- Logs all runs to SliceTracker (detection + colocalization)
- Writes `summary.csv` with per-file results

### Usage:
```bash
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --dry-run
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --threshold 3.0 --dilation 100
```

## Rules

1. **Tracker calls are mandatory**: Every `process_single()` call must log detection and colocalization runs to the tracker.
2. **Dual output**: Results go to both per-subject pipeline folders AND batch_results/ for backward compatibility.
3. **Channel indices**: `RED_IDX=1`, `GREEN_IDX=0` are hardcoded for ENCR ND2 files. Verify channel order before adding new projects.
