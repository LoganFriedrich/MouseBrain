#!/usr/bin/env python3
"""
experiment_tracker.py - Backward-compatibility shim.

The canonical version of this module now lives in the braintools package:
    from braintools.tracker import ExperimentTracker

This shim re-exports everything so existing scripts that do
    from experiment_tracker import ExperimentTracker
continue to work without changes.
"""

# Try braintools package first (preferred)
try:
    from braintools.tracker import *
    from braintools.tracker import (
        ExperimentTracker,
        EXP_TYPES,
        CSV_COLUMNS,
        print_experiment_row,
        main,
    )
except ImportError:
    # Fallback: braintools not installed, use original inline code
    # This should not normally happen — install braintools first
    raise ImportError(
        "braintools package is required. Install with:\n"
        "  cd Y:\\2_Connectome\\Tissue\\braintools\n"
        "  pip install -e ."
    )

if __name__ == '__main__':
    main()
