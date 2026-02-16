# annotator/workers/ - CODE Directory

> **This is a CODE directory.** Background thread workers for the annotator.

## Key Files

| File | Purpose |
|------|---------|
| `loader_worker.py` | QThread worker for loading ND2 files in the background |

## Rules

1. **Thread safety**: Workers emit Qt signals for progress/completion. Never modify napari layers directly from a worker thread.
2. **Error handling**: Workers must catch exceptions and emit error signals rather than crashing silently.
