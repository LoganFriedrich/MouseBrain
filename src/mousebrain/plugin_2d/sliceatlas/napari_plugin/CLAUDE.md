# napari_plugin/ - CODE Directory

> **This is a CODE directory.** Interactive napari widgets for the 2D slice analysis GUI.

## What This Is

napari widgets that provide the interactive GUI for slice analysis, atlas alignment, and annotation.

## Key Files

| File | Purpose |
|------|---------|
| `main_widget.py` | Primary analysis widget: load ND2, detect nuclei, run colocalization, view results |
| `alignment_widget.py` | Atlas alignment widget: register slices to Allen Mouse Brain Atlas |
| `inset_widget.py` | Inset handling widget: manage high-magnification inset regions |
| `annotator_widget.py` | Annotation widget: mark and export annotations on ND2 images |
| `annotator_worker.py` | Background worker for annotation tasks |
| `workers.py` | Background workers for long-running operations (detection, registration) |

## main_widget.py

The largest and most complex widget. Key features:
- Load ND2 files and display channels in napari
- Run StarDist nuclei detection with tunable parameters
- Run colocalization analysis (fold-change and local SNR methods)
- View and manage calibration run history
- Mark runs as "best" for a sample
- All operations logged to SliceTracker

## Rules

1. **Tracker integration**: All detection and colocalization operations MUST call `self.tracker.log_detection()` / `self.tracker.log_colocalization()` before starting and `self.tracker.update_status()` on completion or error.
2. **Guard tracker calls**: Always wrap in `if self.tracker:` since tracker initialization can fail.
3. **Worker threads**: Long operations (detection, registration) run in QThread workers to keep the GUI responsive. Never block the main thread.
4. **Channel convention**: Channel 0 = green (488nm), Channel 1 = red (561nm). Configurable in the widget but these are the defaults.
5. **Session tracking**: Track which runs were made this session for priority display in the context panel.
