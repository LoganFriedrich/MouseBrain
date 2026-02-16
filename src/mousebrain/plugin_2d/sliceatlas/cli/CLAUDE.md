# cli/ - CODE Directory

> **This is a CODE directory.** Command-line interfaces for slice analysis.

## What This Is

CLI entry points for running slice analysis from the command line.

## Key Files

| File | Purpose |
|------|---------|
| `run_coloc.py` | Main CLI: detection + colocalization + QC figure generation |

## run_coloc.py

The primary single-sample analysis script. Workflow:
1. Load ND2 file and extract red/green channels
2. Detect nuclei on red channel (StarDist + z-score peak finding)
3. Estimate background on green channel (percentile method)
4. Classify nuclei as positive/negative (fold-change or local SNR)
5. Generate publication-quality QC figures (overview + zoom panels)
6. Save measurements CSV

### Key implementation details:
- `_intensity_contour_mask()` generates contours from actual channel intensity, not circular label stamps
- `_smooth_mask()` applies Gaussian smoothing before `contour()` for smooth vector outlines
- Overview contour parameters: `OVR_LW=0.4`, `OVR_ALPHA=0.3`
- Zoom panels use `expand_labels()` for cytoplasm territory assignment
- Arrow annotations use `shrinkB` to keep tips away from cells

## Rules

1. **QC figures are journal-quality**: Contours must follow actual cell morphology, not be circular stamps.
2. **All runs logged to tracker**: Detection and colocalization calls must be tracked.
3. **Output routes to per-subject pipeline folders**: Results go to `SampleDirs.DETECTED` and `SampleDirs.QUANTIFIED` under the subject directory.
