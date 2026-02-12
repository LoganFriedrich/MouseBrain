#!/usr/bin/env python3
"""
experiment_tracker.py - Backward-compatibility shim.

The canonical version of this module now lives in the mousebrain package:
    from mousebrain.tracker import ExperimentTracker

This shim re-exports everything so existing scripts that do
    from experiment_tracker import ExperimentTracker
continue to work without changes.
"""

# Try mousebrain package first (preferred)
try:
    from mousebrain.tracker import *
    from mousebrain.tracker import (
        ExperimentTracker,
        EXP_TYPES,
        CSV_COLUMNS,
        print_experiment_row,
        main,
    )
except ImportError:
    # Fallback: mousebrain not installed, use original inline code
    # This should not normally happen — install mousebrain first
    raise ImportError(
        "mousebrain package is required. Install with:\n"
        "  cd Y:\\2_Connectome\\Tissue\\mousebrain\n"
        "  pip install -e ."
    )

if __name__ == '__main__':
    main()
