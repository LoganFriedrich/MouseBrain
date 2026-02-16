# annotator/widgets/ - CODE Directory

> **This is a CODE directory.** napari widget for the annotation UI.

## Key Files

| File | Purpose |
|------|---------|
| `main_widget.py` | Primary annotation widget: load ND2, annotate, export |

## Rules

1. **Worker threads**: Long operations (ND2 loading, export) must run in background workers to keep napari responsive.
2. **Plugin registration**: This widget is registered through the `plugin_2d/napari.yaml` manifest.
